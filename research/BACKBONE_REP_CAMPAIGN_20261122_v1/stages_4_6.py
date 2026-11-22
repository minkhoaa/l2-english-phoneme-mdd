"""Stages 4, 5, and 6 of BACKBONE_REP_CAMPAIGN_20261122_v1.

Stage 4: Backbone-level WavLM LoRA + 18-Articulatory-Feature Multi-Task (Koel CTC 100% frozen).
Stage 5: Cal-selected Top-2 Paired 3-Seed Confirmation + Fast Distillation to WavLM-Base+.
Stage 6: Pre-gold seal audit, standalone CPU independent recount, blind-spot & speaker analysis,
         GTX 1650 latency benchmark, and Vietnamese campaign report.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from types import MethodType
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import unicodedata

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    IPA_TO_ARPA39,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import compute_multitask_mdd_loss
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    CONTROL,
    enforce_vram,
    read_json,
    sha_file,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import Spec
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.replay_codefix_audit import (
    apply_policy_df,
    independent_count,
    summarize_cohorts,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import collate_sf
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import (
    KEY,
    bootstrap,
    gold_frame,
    keyed_merge,
    read_predictions,
    split_calibrated_summary,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import (
    Logits,
    cal_score,
    choose_policy,
)
from research.SYNTH_3DIR_ISO_V1_20261122.huper_attention import QUERY_BLOCK
from research.SYNTH_3DIR_ISO_V1_20261122.common import timit_fit_candidates, verified_intake, verified_wav

from .articulatory import AF_MATRIX, articulatory_bce_loss
from .backbones import (
    BACKBONE_SPECS,
    PHONE_TO_ID,
    VRAM_CAP_MIB,
    extract_slot_cache,
    koel_voronoi_spans,
    pinned_snapshot,
    release_gpu,
)
from .common import (
    BASE_KOEL_CACHE,
    PROTECTED_SEALED_SPEAKERS,
    SEEDS,
    atomic_save_json,
    check_disk_guard,
    compute_cal_arrays,
    load_eval_gold_stage6_only,
    load_splits_no_eval_gold,
    mark_done,
    open_arm_dir,
    smoke_gold,
    write_checkpoint_card,
)
from .models import (
    DualRepConfig,
    FrameXAttnConfig,
    WavLMLoRAMDD,
    inject_wavlm_lora,
)
from .stages_0_3 import (
    CAN_LP_IDX,
    _cal_metrics_summary,
    _save_cal_arrays,
    _seal_arm_eval,
    infer_dualrep_logits,
    infer_xattn_logits,
    train_one_dualrep_arm,
    train_one_xattn_arm,
)

AF_BUNDLES, PHONE_TO_BUNDLE = np.unique(AF_MATRIX[:-1], axis=0, return_inverse=True)


# ---------------------------------------------------------------------------
# Conditional Tiled Attention: standard during train(), tiled FP32 during eval()
# ---------------------------------------------------------------------------
def adapter_self_attention(attention, hidden_states, attention_mask, gated_position_bias):
    """Call the projection modules, not HF eager attention's raw .weight tensors."""
    b, t, _ = hidden_states.shape
    h, d = attention.num_heads, attention.head_dim

    def proj(linear):
        return linear(hidden_states).reshape(b, t, h, d).transpose(1, 2)

    q, k, v = (proj(m) for m in (attention.q_proj, attention.k_proj, attention.v_proj))
    bias = gated_position_bias.reshape(b, h, t, t)
    kv_mask = attention_mask.ne(1)[:, None, None, :] if attention_mask is not None else None
    if attention.training or t <= 600:
        if kv_mask is not None:
            bias = bias.masked_fill(kv_mask, -torch.inf)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=attention.scaling,
                                             dropout_p=attention.dropout if attention.training else 0.0)
    else:
        out = torch.empty_like(q)
        kt = k.transpose(-1, -2)
        for start in range(0, t, QUERY_BLOCK):
            blk = slice(start, start + QUERY_BLOCK)
            score = (q[:, :, blk] * attention.scaling) @ kt + bias[:, :, blk]
            if kv_mask is not None:
                score.masked_fill_(kv_mask, -torch.inf)
            out[:, :, blk] = score.softmax(-1) @ v
    return attention.out_proj(out.transpose(1, 2).reshape(b, t, h * d)), None


def enable_adapter_attention(wavlm_model: torch.nn.Module) -> None:
    from transformers.models.wavlm.modeling_wavlm import WavLMAttention

    for module in wavlm_model.modules():
        if isinstance(module, WavLMAttention):
            module.torch_multi_head_self_attention = MethodType(adapter_self_attention, module)


def _build_wavlm_lora_model(
    preferred: str,
    device: torch.device,
    longest_wav_path: str,
    longest_row: Mapping[str, Any],
    longest_koel_feat: Mapping[str, np.ndarray],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    allow_fallback: bool,
) -> Tuple[WavLMLoRAMDD, Any, str, int, dict]:
    """Instantiate WavLM LoRA with automatic VRAM probe and OOM-safe fallback to wavlm_base_plus."""
    from transformers import AutoFeatureExtractor, WavLMModel

    order = [preferred]
    if allow_fallback and preferred != "wavlm_base_plus":
        order.append("wavlm_base_plus")
    for bname in order:
        spec = BACKBONE_SPECS[bname]
        top_k = 8 if bname == "wavlm_large" else 6
        release_gpu()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            torch.manual_seed(seed)
            snap = pinned_snapshot(spec.repo_or_path)
            fe = AutoFeatureExtractor.from_pretrained(snap, local_files_only=True)
            base = WavLMModel.from_pretrained(snap, local_files_only=True)
            # Frozen prefix tensors need not require grad; non-reentrant checkpointing
            # still computes gradients for LoRA inside the checkpointed layers.
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.config.layerdrop = 0.0
            base.config.apply_spec_augment = False
            wrapped = inject_wavlm_lora(base, top_k_layers=top_k, r=8, alpha=16.0, dropout=0.05)
            if wrapped != top_k * 4:
                raise RuntimeError(f"Unexpected LoRA projection count: {wrapped}")
            enable_adapter_attention(base)
            # Include the final state so every inserted adapter reaches the loss.
            extract_layers = lora_feature_layers(base, spec.layers)
            model = WavLMLoRAMDD(base, extract_layers=extract_layers, wavlm_dim=spec.hidden_dim).to(device)
            model.train()
            wav, sr = sf.read(longest_wav_path, dtype="float32")
            if sr != 16000 or wav.ndim != 1 or not np.isfinite(wav).all():
                raise ValueError(f"Invalid LoRA probe WAV: {longest_row['uid']}")
            w_in = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"].to(device=device, dtype=torch.float32)
            inp, labels = collate_sf([longest_row], [longest_koel_feat], norm, device, labels=True)
            starts, ends, peaks, t_koel = koel_voronoi_spans(BASE_KOEL_CACHE, longest_row)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                t_start = time.perf_counter()
                out = model.forward_single_utt(w_in, inp, starts, ends, peaks, t_koel)
                loss = compute_multitask_mdd_loss(out, *labels)["loss"]
                loss.backward()
                grad_norms = [float(p.grad.norm()) if p.grad is not None else 0.0
                              for name, p in model.named_parameters() if name.endswith(".lora_b")]
                if len(grad_norms) != wrapped or not all(np.isfinite(g) and g > 0 for g in grad_norms):
                    raise RuntimeError(f"LoRA gradient probe failed: {sum(g > 0 for g in grad_norms)}/{wrapped}")
                seconds = time.perf_counter() - t_start
            peak = enforce_vram(device, limit=3350)
            model.zero_grad(set_to_none=True)
            probe = {"uid": longest_row["uid"], "duration_s": len(wav) / sr,
                     "peak_mib": peak, "adapter_b_gradients": len(grad_norms),
                     "min_adapter_b_grad_norm": min(grad_norms), "forward_backward_s": seconds}
            print(f"[Stage 4 VRAM probe] {bname}: {probe}", flush=True)
            return model, fe, bname, top_k, probe
        except RuntimeError as exc:
            memory_failure = isinstance(exc, torch.cuda.OutOfMemoryError) or str(exc).startswith("GPU budget exceeded:")
            if not memory_failure or bname == order[-1]:
                raise
            print(f"[Stage 4 VRAM probe] {bname}: {exc}; falling back to Base+ before training", flush=True)
            if "base" in locals():
                del base
            if "model" in locals():
                del model
        release_gpu()
    raise RuntimeError("No WavLM LoRA configuration fit in VRAM")


def lora_feature_layers(base, layers):
    return tuple(dict.fromkeys([*layers, base.config.num_hidden_layers]))


def _load_clean_warm_rows(*, max_utts: int = 2400, seed: int = 20261121) -> List[Dict[str, Any]]:
    """Admit unused TIMIT Fit and DoReCo TRAIN Tier A from the pinned intake."""
    fit, cal, eval_prompts = load_splits_no_eval_gold("clean_11")
    excluded = cal + eval_prompts
    exclusions = {
        "base_timit_fit_speakers": {r["speaker"] for r in fit if r["source"].startswith("timit")},
        "base_timit_cal_speakers": {r["speaker"] for r in cal if r["source"].startswith("timit")},
        "cal_eval_speakers": {r["speaker"].strip().upper() for r in excluded} | PROTECTED_SEALED_SPEAKERS,
        "cal_eval_uids": {r["uid"] for r in excluded}, "fit_uids": {r["uid"] for r in fit},
        "cal_eval_shas": {r["audio_sha256"] for r in excluded}, "fit_shas": {r["audio_sha256"] for r in fit},
        "eval_timit_pids": {r["uid"].rsplit("_", 1)[-1].upper() for r in excluded
                            if (r.get("source") or r.get("cohort")).startswith("timit")},
    }
    intake = verified_intake()
    timit = list(timit_fit_candidates(exclusions, intake))
    doreco = [r for r in intake if r["corpus"] == "DoReCo_English" and r["split"] == "TRAIN"]
    blocked_phones = {tuple(r["canonical_phones"]) for r in excluded}
    seen_uids = exclusions["cal_eval_uids"] | exclusions["fit_uids"]
    seen_shas = exclusions["cal_eval_shas"] | exclusions["fit_shas"]
    pools = {"TIMIT": [], "DoReCo_English": []}
    for r in timit + doreco:
        spk = r["speaker_id"].strip().upper()
        dur = float(r.get("duration_sec") or 0.0)
        if (r.get("annotation_tier") != "Tier A" or r.get("ctc_reference_status") != "qualified"
                or not spk or spk in exclusions["cal_eval_speakers"] or not 0.8 <= dur <= 9.5
                or r["uid"] in seen_uids or r["audio_sha256"] in seen_shas):
            continue
        try:
            phones = tuple(IPA_TO_ARPA39[unicodedata.normalize("NFC", t)] for t in r["realized_ipa_tokens"])
            ids = [PHONE_TO_ID[p] for p in phones]
        except KeyError:
            continue
        if len(ids) < 6 or phones in blocked_phones:
            continue
        # Conservative frame bound for CTC with adjacent repeats (AF can merge phone classes).
        if len(ids) + 1 + sum(PHONE_TO_BUNDLE[a] == PHONE_TO_BUNDLE[b] for a, b in zip(ids, ids[1:])) > int(dur * 50) - 1:
            continue
        seen_uids.add(r["uid"])
        seen_shas.add(r["audio_sha256"])
        pools[r["corpus"]].append((r, spk, ids))
    if not all(pools.values()):
        raise RuntimeError("Warm-start needs both admitted corpora; no silent warm-start waiver")
    rng = np.random.default_rng(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    candidates = []
    for i in range(max(map(len, pools.values()))):
        for pool in pools.values():
            if i < len(pool) and len(candidates) < max_utts:
                r, speaker, ids = pool[i]
                candidates.append(dict(uid=r["uid"], corpus=r["corpus"], speaker=speaker,
                                       audio_path=verified_wav(r), audio_sha256=r["audio_sha256"], realized_ids=ids))
        if len(candidates) >= max_utts:
            break
    return candidates


def warm_ctc_loss(phone_logits, af_logits, ids, bundles, phone_to_bundle):
    """Phone and multi-attribute bundle CTC over the audio sequence, no uniform spans."""
    aux_ids = phone_to_bundle[ids]
    num_frames = len(phone_logits)
    min_frames = len(ids) + int((aux_ids[1:] == aux_ids[:-1]).sum())
    if num_frames < min_frames:
        raise ValueError("Insufficient frames for repeated-label AF CTC")
    lens = [num_frames], [len(ids)]
    phone = F.ctc_loss(phone_logits.float().log_softmax(-1)[:, None], ids, *lens, blank=phone_logits.shape[-1] - 1)
    attr, blank = af_logits[:, :-1].float(), af_logits[:, -1:].float()
    bundle_lp = (attr @ bundles.T).log_softmax(-1) + F.logsigmoid(-blank)
    aux_lp = torch.cat([bundle_lp, F.logsigmoid(blank)], dim=-1)
    aux = F.ctc_loss(aux_lp[:, None], aux_ids, *lens, blank=len(bundles))
    return phone + 0.5 * aux


@torch.inference_mode()
def _infer_lora_logits(
    model: WavLMLoRAMDD,
    fe: Any,
    rows: Sequence[Mapping[str, Any]],
    koel_feats: Sequence[Mapping[str, np.ndarray]],
    spans_list: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> List[Logits]:
    model.eval()
    out: List[Logits] = []
    prev_long = False
    for r, kf, (starts, ends, peaks, t_koel) in zip(rows, koel_feats, spans_list, strict=True):
        wav, _ = sf.read(r["audio_path"], dtype="float32")
        is_long = len(wav) > 15 * 16000
        if device.type == "cuda" and (is_long or prev_long):
            torch.cuda.empty_cache()
        prev_long = is_long
        w_in = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"].to(device=device, dtype=torch.float32)
        inp, _ = collate_sf([r], [kf], norm, device, labels=False)
        res = model.forward_single_utt(w_in, inp, starts, ends, peaks, t_koel)
        L = len(r["canonical_phones"])
        out.append(
            Logits(
                res["target_logits"][0, :L].float().cpu().numpy(),
                res["gap_logits"][0, : L + 1].float().cpu().numpy(),
                res["realized_logits"][0, :L].float().cpu().numpy(),
                kf["phone_ids"],
            )
        )
        enforce_vram(device, limit=VRAM_CAP_MIB)
    return out


def train_one_lora_arm(
    arm_name: str,
    preferred_backbone: str,
    seed: int,
    arm_dir: Path,
    fit_rows: Sequence[Dict[str, Any]],
    cal_rows: Sequence[Dict[str, Any]],
    eval_prompts: Sequence[Dict[str, Any]],
    xf: Sequence[Dict[str, np.ndarray]],
    xc: Sequence[Dict[str, np.ndarray]],
    xe: Sequence[Dict[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    epochs: int = 8,
    grad_accum: int = 16,
    warm_rows: Sequence[Dict[str, Any]] = (),
    warm_epochs: int = 2,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    check_disk_guard(arm_dir.parent)
    open_arm_dir(arm_dir)
    t0 = time.perf_counter()
    # Precompute Koel Voronoi spans once (tiny CPU tuples)
    fit_spans = [koel_voronoi_spans(BASE_KOEL_CACHE, r) for r in fit_rows]
    cal_spans = [koel_voronoi_spans(BASE_KOEL_CACHE, r) for r in cal_rows]
    eval_spans = [koel_voronoi_spans(BASE_KOEL_CACHE, r) for r in eval_prompts]

    longest_idx = int(np.argmax([s[3] for s in fit_spans]))
    torch.manual_seed(seed)
    model, fe, used_backbone, top_k, lora_probe = _build_wavlm_lora_model(
        preferred_backbone,
        device,
        fit_rows[longest_idx]["audio_path"],
        fit_rows[longest_idx],
        xf[longest_idx],
        norm,
        seed=seed, allow_fallback=allow_fallback,
    )
    af_table = torch.from_numpy(AF_MATRIX).to(device)
    bundles = torch.from_numpy(AF_BUNDLES).to(device)
    phone_to_bundle = torch.from_numpy(PHONE_TO_BUNDLE).to(device)
    peak = lora_probe["peak_mib"]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "wavlm." in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "wavlm." not in n]

    # Optional Stage 4B Multi-Corpus Clean Phonetic + 18-AF Warm-Start
    warm_hist: List[Dict[str, Any]] = []
    if warm_rows and warm_epochs > 0:
        print(f"[Stage 4 {arm_name}] Warm-start {warm_epochs} ep with phone/AF CTC on {len(warm_rows)} admitted utts...", flush=True)
        w_opt = torch.optim.AdamW(
            [{"params": lora_params, "lr": 2e-4}, {"params": head_params, "lr": 8e-4}],
            weight_decay=0.01,
        )
        w_rng = np.random.default_rng(seed + 77)
        for wep in range(1, warm_epochs + 1):
            model.train()
            order = w_rng.permutation(len(warm_rows))
            w_opt.zero_grad(set_to_none=True)
            w_loss_sum = 0.0
            for step_i, idx in enumerate(order, 1):
                wr = warm_rows[int(idx)]
                wav, _ = sf.read(wr["audio_path"], dtype="float32")
                w_in = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"].to(device=device, dtype=torch.float32)
                ph_lg, af_lg = model.forward_warm_single_utt(w_in)
                r_ids = torch.tensor(wr["realized_ids"], dtype=torch.long, device=device)
                w_loss = warm_ctc_loss(ph_lg, af_lg, r_ids, bundles, phone_to_bundle)
                group_size = min(grad_accum, len(order) - ((step_i - 1) // grad_accum) * grad_accum)
                (w_loss / group_size).backward()
                w_loss_sum += float(w_loss.detach())
                if step_i % grad_accum == 0 or step_i == len(order):
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0, error_if_nonfinite=True)
                    w_opt.step()
                    w_opt.zero_grad(set_to_none=True)
                    peak = max(peak, enforce_vram(device, limit=VRAM_CAP_MIB))
            warm_hist.append({"warm_epoch": wep, "loss": w_loss_sum / max(1, len(order))})
            print(f"  [warm ep {wep}] loss={warm_hist[-1]['loss']:.4f}", flush=True)
        del w_opt

    # Main MDD + Realized + 18-AF Training
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": 1.5e-4}, {"params": head_params, "lr": 8e-4}],
        weight_decay=0.02,
    )
    n_opt_steps = max(1, math.ceil(len(fit_rows) / grad_accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_opt_steps, eta_min=1e-5)
    rng = np.random.default_rng(seed)
    order_hash = hashlib.sha256()

    hist, best_sel, best_ep, best_state = [], -math.inf, 0, None

    for ep in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        print(f"[LoRA {arm_name} seed={seed}] training epoch {ep}/{epochs}", flush=True)
        model.train()
        order = rng.permutation(len(fit_rows))
        order_hash.update(order.astype("<i8").tobytes())
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0

        for step_i, idx in enumerate(order, 1):
            r = fit_rows[int(idx)]
            kf = xf[int(idx)]
            starts, ends, peaks, t_koel = fit_spans[int(idx)]
            wav, _ = sf.read(r["audio_path"], dtype="float32")
            w_in = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"].to(device=device, dtype=torch.float32)
            inp, labels = collate_sf([r], [kf], norm, device, labels=True)
            out_dict = model.forward_single_utt(w_in, inp, starts, ends, peaks, t_koel)
            loss_mdd = compute_multitask_mdd_loss(
                out_dict,
                *labels,
                gap_pos_weight=8.0,
                lambda_gap=0.6,
                lambda_realized=0.25,
                native_fa_weight=1.5,
            )["loss"]
            loss_af = articulatory_bce_loss(out_dict["af_logits"], labels[2], af_table)
            loss = loss_mdd + 0.25 * loss_af
            group_size = min(grad_accum, len(order) - ((step_i - 1) // grad_accum) * grad_accum)
            (loss / group_size).backward()
            loss_sum += float(loss.detach())

            if step_i % grad_accum == 0 or step_i == len(order):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0, error_if_nonfinite=True)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                peak = max(peak, enforce_vram(device, limit=VRAM_CAP_MIB))

        cal_lg = _infer_lora_logits(model, fe, cal_rows, xc, cal_spans, norm, device)
        sel, tap, gap = cal_score(cal_rows, cal_lg)
        hist.append({"epoch": ep, "loss": loss_sum / len(order), "cal_t_ap": tap, "cal_g_ap": gap,
                     "sel": sel, "wall_s": time.perf_counter() - epoch_start})
        atomic_save_json(arm_dir / "TRAIN_PROGRESS.json", {"history": hist, "warm_history": warm_hist, "vram_probe": lora_probe})
        print(
            f"  [LoRA {arm_name} ep={ep}/{epochs}] loss={hist[-1]['loss']:.4f} "
            f"cal_tAP={tap:.4f} cal_gAP={gap:.4f} peak={peak:.1f} MiB",
            flush=True,
        )
        if sel > best_sel:
            best_sel, best_ep = sel, ep
            # Save only trainable parameters (LoRA + head) to keep checkpoint small & fast!
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}

    # Restore best trainable weights
    cur_sd = model.state_dict()
    cur_sd.update(best_state)
    model.load_state_dict(cur_sd, strict=True)
    model.eval()

    cal_logits = _infer_lora_logits(model, fe, cal_rows, xc, cal_spans, norm, device)
    policy = choose_policy(cal_rows, cal_logits, Spec())
    cal_l2, cal_l1 = compute_cal_arrays(cal_rows, cal_logits, policy)
    cal_m = _cal_metrics_summary(cal_rows, cal_logits, policy, xc)
    eval_logits = _infer_lora_logits(model, fe, eval_prompts, xe, eval_spans, norm, device)
    peak = max(peak, enforce_vram(device, limit=VRAM_CAP_MIB))

    cp_path = arm_dir / "lora_and_head.pt"
    torch.save(
        {
            "arm": arm_name,
            "seed": seed,
            "used_backbone": used_backbone,
            "top_k_layers": top_k,
            "extract_layers": list(model.extract_layers),
            "trainable_state_dict": best_state,
            "norm": {k: (v[0].tolist(), v[1].tolist()) for k, v in norm.items()},
            "policy": policy,
            "best_epoch": best_ep,
        },
        cp_path,
    )
    cp_sha = sha_file(cp_path)
    _save_cal_arrays(arm_dir, cal_l2, cal_l1)
    card_sha = write_checkpoint_card(
        arm_dir,
        {
            "arm": arm_name,
            "seed": seed,
            "used_backbone": used_backbone,
            "top_k_layers": top_k,
            "checkpoint_sha256": cp_sha,
            "train_order_sha256": order_hash.hexdigest(),
            "cal_arrays_sha256": sha_file(arm_dir / "cal_arrays.npz"),
            "best_epoch": best_ep,
            "policy": policy,
            "cal_metrics": cal_m,
        },
    )
    seal = _seal_arm_eval(arm_dir, eval_prompts, eval_logits, policy, card_sha)
    summary = {
        "arm": arm_name,
        "seed": seed,
        "used_backbone": used_backbone,
        "top_k_layers": top_k,
        "warm_utts": len(warm_rows),
        "vram_and_gradient_probe": lora_probe,
        "extract_layers": list(model.extract_layers),
        "best_epoch": best_ep,
        "checkpoint_sha256": cp_sha,
        "seal_sha256": sha_file(arm_dir / "PREDICTIONS_SEAL.json"),
        "cal_metrics": cal_m,
        "policy": policy,
        "peak_mib": peak,
        "wall_s": time.perf_counter() - t0,
        "warm_history": warm_hist,
        "history": hist,
        "sealed_cohorts": sorted(seal["files"].keys()),
    }
    atomic_save_json(arm_dir / "SUMMARY.json", summary)
    print(
        f"[LoRA {arm_name} seed={seed} ({used_backbone})] ep={best_ep} "
        f"Cal F1={cal_m['cal_l2_strict_f1']:.2f}% tAP={cal_m['cal_t_ap']:.4f} "
        f"gAP={cal_m['cal_g_ap']:.4f} blindAP={cal_m['cal_blind_spot_ap']:.4f} "
        f"L1_FA/1k={cal_m['cal_l1_fa_per_1k']:.2f} ({summary['wall_s']:.1f}s)",
        flush=True,
    )
    del model
    release_gpu()
    return summary


def run_stage4_wavlm_lora(
    out_dir: Path,
    state: Dict[str, Any],
    fit_rows: Sequence[Dict[str, Any]],
    cal_rows: Sequence[Dict[str, Any]],
    eval_prompts: Sequence[Dict[str, Any]],
    xf: Sequence[Dict[str, np.ndarray]],
    xc: Sequence[Dict[str, np.ndarray]],
    xe: Sequence[Dict[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    preferred_backbone: str = "wavlm_large",
    seed: int = SEEDS[0],
    epochs: int = 8,
    warm_max_utts: int = 2400,
    warm_epochs: int = 2,
) -> Dict[str, Any]:
    stage_dir = out_dir / "stage4_lora"
    if "stage4_lora" in state["completed"] and (stage_dir / "STAGE4_LORA.json").exists():
        return read_json(stage_dir / "STAGE4_LORA.json")

    stage_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    # Arm C1: Direct WavLM LoRA + 18-AF
    c1_key = "stage4_C1_wavlm_lora_direct"
    c1_dir = stage_dir / "C1_wavlm_lora_direct"
    if c1_key in state["completed"] and (c1_dir / "SUMMARY.json").exists():
        results["C1_wavlm_lora_direct"] = read_json(c1_dir / "SUMMARY.json")
    else:
        res = train_one_lora_arm(
            "C1_wavlm_lora_direct",
            preferred_backbone,
            seed,
            c1_dir,
            fit_rows,
            cal_rows,
            eval_prompts,
            xf,
            xc,
            xe,
            norm,
            device,
            epochs=epochs,
            warm_rows=(),
            warm_epochs=0,
            allow_fallback=True,
        )
        mark_done(out_dir, state, c1_key, {"cal_l2_strict_f1": res["cal_metrics"]["cal_l2_strict_f1"]})
        results["C1_wavlm_lora_direct"] = res

    # Arm C2: Clean Multi-Corpus Phonetic+AF Warm-Start -> WavLM LoRA MDD+AF
    c2_key = "stage4_C2_wavlm_lora_warm_af"
    c2_dir = stage_dir / "C2_wavlm_lora_warm_af"
    if c2_key in state["completed"] and (c2_dir / "SUMMARY.json").exists():
        results["C2_wavlm_lora_warm_af"] = read_json(c2_dir / "SUMMARY.json")
    else:
        warm_rows = _load_clean_warm_rows(max_utts=warm_max_utts)
        atomic_save_json(stage_dir / "WARM_FIT_RECEIPT.json", {"rows": warm_rows})
        res = train_one_lora_arm(
            "C2_wavlm_lora_warm_af",
            results["C1_wavlm_lora_direct"]["used_backbone"],
            seed,
            c2_dir,
            fit_rows,
            cal_rows,
            eval_prompts,
            xf,
            xc,
            xe,
            norm,
            device,
            epochs=epochs,
            warm_rows=warm_rows,
            warm_epochs=warm_epochs,
        )
        mark_done(out_dir, state, c2_key, {"cal_l2_strict_f1": res["cal_metrics"]["cal_l2_strict_f1"]})
        results["C2_wavlm_lora_warm_af"] = res

    out = {"stage": "stage4_lora", "arms": results}
    atomic_save_json(stage_dir / "STAGE4_LORA.json", out)
    mark_done(out_dir, state, "stage4_lora", {"arms": sorted(results.keys())})
    return out


# ---------------------------------------------------------------------------
# Stage 5: Cal-Selected Top-2 Paired 3-Seed Confirmation + Fast Distillation
# ---------------------------------------------------------------------------
def _cal_rank_score(cal_m: Mapping[str, float], s0_l1_fa: float) -> float:
    fa_penalty = max(0.0, cal_m["cal_l1_fa_per_1k"] - max(2.0, s0_l1_fa * 1.25)) * 1.5
    return (
        cal_m["cal_l2_strict_f1"]
        + 25.0 * cal_m["cal_t_ap"]
        + 15.0 * cal_m["cal_g_ap"]
        + 10.0 * cal_m["cal_blind_spot_ap"]
        - fa_penalty
    )


def _load_teacher_fit_logits(
    arm_id: str,
    stage_dir: Path,
    stage1_report: Mapping[str, Any],
    out_dir: Path,
    fit_rows: Sequence[Dict[str, Any]],
    xf: Sequence[Dict[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> List[Logits]:
    """Compute teacher logits on Fit rows for Stage 5 fast distillation."""
    top1 = stage1_report["selected_top1"]
    top2 = stage1_report["selected_top2"]
    caches = out_dir / "caches"

    if arm_id.startswith("A"):
        ck = torch.load(stage_dir / "model.pt", map_location="cpu", weights_only=True)
        cfg = DualRepConfig(**ck["config"])
        from .models import DualRepSingleFactorV2
        model = DualRepSingleFactorV2(cfg).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)
        ext1_dir = caches / f"slots_{top2 if arm_id == 'A2_top2' else top1}"
        ext2_dir = (caches / f"slots_{top2}") if arm_id == "A4_top1_top2" else None
        charsiu_dir = (caches / "charsiu_10ms") if arm_id == "A3_top1_charsiu_proto" else None
        proto = None
        aux_norm = None
        if charsiu_dir is not None:
            with np.load(out_dir / f"stage1_probe/prototypes_{top1}.npz") as z:
                proto = {k: z[k] for k in z.files}
            if ck.get("aux_norm") is not None:
                aux_norm = (np.asarray(ck["aux_norm"][0], np.float32), np.asarray(ck["aux_norm"][1], np.float32))
        lg = infer_dualrep_logits(
            model, fit_rows, xf, norm, device, 16,
            ext1_dir=ext1_dir, ext2_dir=ext2_dir, charsiu_dir=charsiu_dir, proto=proto, aux_norm=aux_norm,
        )
        del model
        release_gpu()
        return lg

    if arm_id.startswith("B"):
        ck = torch.load(stage_dir / "model.pt", map_location="cpu", weights_only=True)
        cfg = FrameXAttnConfig(**ck["config"])
        from .models import FrameXAttnMDD
        model = FrameXAttnMDD(cfg).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)
        lg = infer_xattn_logits(
            model, fit_rows, xf, norm, device, 8,
            frame_dir=caches / f"frames_{top1}", include_koel_mid=cfg.use_koel_frame_win,
        )
        del model
        release_gpu()
        return lg

    # C* LoRA teacher
    from transformers import AutoFeatureExtractor, WavLMModel
    ck = torch.load(stage_dir / "lora_and_head.pt", map_location="cpu", weights_only=True)
    bname = ck["used_backbone"]
    spec = BACKBONE_SPECS[bname]
    snap = pinned_snapshot(spec.repo_or_path)
    fe = AutoFeatureExtractor.from_pretrained(snap, local_files_only=True)
    base = WavLMModel.from_pretrained(snap, local_files_only=True)
    inject_wavlm_lora(base, top_k_layers=ck["top_k_layers"], r=8, alpha=16.0, dropout=0.05)
    enable_adapter_attention(base)
    model = WavLMLoRAMDD(base, extract_layers=ck["extract_layers"], wavlm_dim=spec.hidden_dim).to(device)
    sd = model.state_dict()
    sd.update(ck["trainable_state_dict"])
    model.load_state_dict(sd, strict=True)
    spans = [koel_voronoi_spans(BASE_KOEL_CACHE, r) for r in fit_rows]
    lg = _infer_lora_logits(model, fe, fit_rows, xf, spans, norm, device)
    del model
    release_gpu()
    return lg


def run_stage5_confirm_and_distill(
    out_dir: Path,
    state: Dict[str, Any],
    stage0_report: Mapping[str, Any],
    stage1_report: Mapping[str, Any],
    stage2_report: Mapping[str, Any],
    stage3_report: Mapping[str, Any],
    stage4_report: Mapping[str, Any],
    fit_rows: Sequence[Dict[str, Any]],
    cal_rows: Sequence[Dict[str, Any]],
    eval_prompts: Sequence[Dict[str, Any]],
    xf: Sequence[Dict[str, np.ndarray]],
    xc: Sequence[Dict[str, np.ndarray]],
    xe: Sequence[Dict[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    seeds: Sequence[int] = SEEDS,
    dualrep_epochs: int = 14,
    phase2_epochs: int = 8,
    xattn_epochs: int = 12,
    lora_epochs: int = 8,
    warm_max_utts: int = 2400,
    warm_epochs: int = 2,
    smoke: bool = False,
) -> Dict[str, Any]:
    stage_dir = out_dir / "stage5_confirm"
    if "stage5_confirm" in state["completed"] and (stage_dir / "STAGE5_CONFIRM.json").exists():
        return read_json(stage_dir / "STAGE5_CONFIRM.json")

    stage_dir.mkdir(parents=True, exist_ok=True)
    s0_l1_fa = float(stage0_report["mean_cal_l1_fa_per_1k"])

    # Collect all Stage 2–4 candidates with their Stage-1 seed directories
    candidates: List[Tuple[str, Path, Dict[str, Any], float]] = []
    for stage_sub, rep in [
        ("stage2_dualrep", stage2_report),
        ("stage3_xattn", stage3_report),
        ("stage4_lora", stage4_report),
    ]:
        for arm_name, summary in rep["arms"].items():
            score = _cal_rank_score(summary["cal_metrics"], s0_l1_fa)
            candidates.append((arm_name, out_dir / stage_sub / arm_name, summary, score))

    candidates.sort(key=lambda x: x[3], reverse=True)
    s0 = stage0_report["per_seed"][str(seeds[0])]["cal_metrics"]

    def eligible(m):
        return (
            m["cal_l1_fa_per_1k"] <= s0["cal_l1_fa_per_1k"] + 2.0
            and m["cal_t_ap"] >= s0["cal_t_ap"] - 0.005
            and (m["cal_l2_strict_f1"] >= s0["cal_l2_strict_f1"] + 0.5
                 or m["cal_t_ap"] >= s0["cal_t_ap"] + 0.01
                 or m["cal_g_ap"] >= s0["cal_g_ap"] + 0.01)
        )

    ranking = [
        {"arm": a, "cal_rank_score": sc, "cal_metrics": s["cal_metrics"], "passed": eligible(s["cal_metrics"])}
        for a, _, s, sc in candidates
    ]
    top2_selected = (candidates if smoke else [c for c in candidates if eligible(c[2]["cal_metrics"])])[:2]
    print(f"[Stage 5] Cal-selected arms: {[x[0] for x in top2_selected]}; smoke={smoke}", flush=True)

    top1 = stage1_report["selected_top1"]
    top2 = stage1_report["selected_top2"]
    caches = out_dir / "caches"
    confirmed: Dict[str, Dict[str, Any]] = {}

    for arm_name, seed0_dir, seed0_summary, _ in top2_selected:
        arm_seeds: Dict[str, Any] = {str(seeds[0]): {"reused_from": str(seed0_dir), **seed0_summary}}
        for seed in seeds[1:]:
            s_key = f"stage5_{arm_name}_seed_{seed}"
            s_dir = stage_dir / arm_name / f"seed_{seed}"
            if s_key in state["completed"] and (s_dir / "SUMMARY.json").exists():
                arm_seeds[str(seed)] = read_json(s_dir / "SUMMARY.json")
                continue

            if arm_name.startswith("A"):
                cfg = DualRepConfig(**seed0_summary["config"])
                ext1_dir = caches / f"slots_{top2 if arm_name == 'A2_top2' else top1}"
                ext2_dir = (caches / f"slots_{top2}") if arm_name == "A4_top1_top2" else None
                charsiu_dir = (caches / "charsiu_10ms") if arm_name == "A3_top1_charsiu_proto" else None
                proto = None
                if charsiu_dir is not None:
                    with np.load(out_dir / f"stage1_probe/prototypes_{top1}.npz") as z:
                        proto = {k: z[k] for k in z.files}
                res = train_one_dualrep_arm(
                    arm_name, cfg, seed, s_dir, fit_rows, cal_rows, eval_prompts, xf, xc, xe, norm, device,
                    ext1_dir=ext1_dir, ext2_dir=ext2_dir, charsiu_dir=charsiu_dir, proto=proto,
                    epochs=dualrep_epochs, phase2_epochs=phase2_epochs,
                )
            elif arm_name.startswith("B"):
                cfg = FrameXAttnConfig(**seed0_summary["config"])
                res = train_one_xattn_arm(
                    arm_name, cfg, seed, s_dir, fit_rows, cal_rows, eval_prompts, xf, xc, xe, norm, device,
                    frame_dir=caches / f"frames_{top1}", epochs=xattn_epochs,
                )
            else:
                warm_rows = (
                    _load_clean_warm_rows(max_utts=warm_max_utts)
                    if arm_name == "C2_wavlm_lora_warm_af"
                    else ()
                )
                res = train_one_lora_arm(
                    arm_name,
                    seed0_summary["used_backbone"],
                    seed,
                    s_dir,
                    fit_rows,
                    cal_rows,
                    eval_prompts,
                    xf,
                    xc,
                    xe,
                    norm,
                    device,
                    epochs=lora_epochs,
                    warm_rows=warm_rows,
                    warm_epochs=warm_epochs if warm_rows else 0,
                )
            mark_done(out_dir, state, s_key, {"cal_l2_strict_f1": res["cal_metrics"]["cal_l2_strict_f1"]})
            arm_seeds[str(seed)] = res
        confirmed[arm_name] = arm_seeds

    # Distill only a heavy teacher and include a same-seed no-KD Base+ comparator.
    distill_seeds: Dict[str, Any] = {}
    nodistill_seeds: Dict[str, Any] = {}
    best = top2_selected[0] if top2_selected else None
    heavy = {"wavlm_large", "huper_mid", "whisper_small"}
    teacher_backbones = set()
    if best:
        name, _, summary, _ = best
        if name.startswith("C"):
            teacher_backbones.add(summary["used_backbone"])
        else:
            teacher_backbones.add(top2 if name == "A2_top2" else top1)
            if name == "A4_top1_top2":
                teacher_backbones.add(top2)
    if best and (smoke or teacher_backbones & heavy):
        best_arm_name, best_arm_dir, _, _ = best
        print(f"[Stage 5] Distilling teacher {best_arm_name} -> D1_distill_fast (Koel + WavLM-Base+)...", flush=True)
        base_dir = caches / "slots_wavlm_base_plus"
        extract_slot_cache("wavlm_base_plus", eval_prompts, BASE_KOEL_CACHE, base_dir, device)
        teacher_fit_lg = _load_teacher_fit_logits(
            best_arm_name, best_arm_dir, stage1_report, out_dir, fit_rows, xf, norm, device
        )
        fast_spec = BACKBONE_SPECS["wavlm_base_plus"]
        fast_cfg = DualRepConfig(
            arm="D1_distill_fast",
            ext1_layers=len(fast_spec.layers),
            ext1_dim=fast_spec.hidden_dim,
        )
        for seed in seeds:
            for arm_name, teacher, seed_map in (
                ("D0_base_plus_no_kd", None, nodistill_seeds),
                ("D1_distill_fast", teacher_fit_lg, distill_seeds),
            ):
                d_key = f"stage5_{arm_name}_seed_{seed}"
                d_dir = stage_dir / arm_name / f"seed_{seed}"
                if d_key in state["completed"] and (d_dir / "SUMMARY.json").exists():
                    seed_map[str(seed)] = read_json(d_dir / "SUMMARY.json")
                    continue
                res = train_one_dualrep_arm(
                    arm_name, fast_cfg, seed, d_dir, fit_rows, cal_rows, eval_prompts,
                    xf, xc, xe, norm, device, ext1_dir=base_dir,
                    epochs=dualrep_epochs, phase2_epochs=phase2_epochs, teacher_logits=teacher,
                )
                mark_done(out_dir, state, d_key, {"cal_l2_strict_f1": res["cal_metrics"]["cal_l2_strict_f1"]})
                seed_map[str(seed)] = res

    out = {
        "stage": "stage5_confirm",
        "cal_ranking": ranking,
        "selected_top2": [x[0] for x in top2_selected],
        "smoke_gate_override": smoke,
        "confirmed_3seed": confirmed,
        "distill_fast_3seed": distill_seeds,
        "distill_no_kd_3seed": nodistill_seeds,
    }
    atomic_save_json(stage_dir / "STAGE5_CONFIRM.json", out)
    mark_done(out_dir, state, "stage5_confirm", {"selected_top2": out["selected_top2"]})
    return out


# ---------------------------------------------------------------------------
# Stage 6: Pre-Gold Seal Verification, CPU Independent Recount, Latency & Report
# ---------------------------------------------------------------------------
def _verify_sealed_arm(arm_dir, expected_slots, summary):
    card = read_json(arm_dir / "CHECKPOINT_CARD.json")
    ckpt = next(p for p in (arm_dir / "control.pt", arm_dir / "model.pt", arm_dir / "lora_and_head.pt") if p.is_file())
    if sha_file(ckpt) != card["checkpoint_sha256"] or card["checkpoint_sha256"] != summary["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint digest mismatch: {arm_dir}")
    if sha_file(arm_dir / "PREDICTIONS_SEAL.json") != summary["seal_sha256"]:
        raise RuntimeError(f"Prediction seal digest mismatch: {arm_dir}")
    if sha_file(arm_dir / "cal_arrays.npz") != card["cal_arrays_sha256"]:
        raise RuntimeError(f"Cal arrays digest mismatch: {arm_dir}")
    pred = read_predictions(arm_dir)
    if {"gold", "gold_label", "gold_targets", "gold_gaps"} & set(pred.columns):
        raise RuntimeError(f"Gold in prediction columns: {arm_dir}")
    have = pred.set_index(KEY).sort_index()
    if not have.index.equals(expected_slots.index) or not have[["speaker", "canonical_phone"]].equals(expected_slots):
        raise RuntimeError(f"Sealed slot identities do not match the admitted Eval prompts: {arm_dir}")
    t = pred.domain.eq("target").to_numpy()
    score = pred.score.to_numpy()
    if not ((0 <= score) & (score <= 1.000001)).all() or not pred.call.eq(
        apply_policy_df(pred, card["policy"]["tau_t"], card["policy"]["tau_g"])["call"]
    ).all() or pred.loc[t, ["p_o", "p_s"]].isna().any().any():
        raise RuntimeError(f"Invalid or inconsistent Eval calls: {arm_dir}")
    return card


def _read_cal(arm_dir):
    with np.load(arm_dir / "cal_arrays.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def closest_recall_policy(scores, reference_called):
    """Find an achievable Cal recall, without splitting tied scores."""
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Recall matching requires finite positive examples")
    unique, counts = np.unique(scores, return_counts=True)
    thresholds = np.r_[np.nextafter(float(unique[-1]), np.inf), unique[::-1]]
    achieved = np.r_[0, counts[::-1].cumsum()]
    index = int(np.argmin(np.abs(achieved - reference_called)))
    return {"tau": float(thresholds[index]), "cal_called": int(achieved[index]),
            "ref_cal_called": int(reference_called), "cal_positive": len(scores)}


def _lock_recall_match(arm_cal, ref_cal, reference_policy):
    result = {}
    for role, positives in (("t", ["S", "O"]), ("g", ["I"])):
        scores_key, lab_key = f"l2_{role}_sc", f"l2_{role}_g"
        if not np.array_equal(arm_cal[lab_key], ref_cal[lab_key]):
            raise RuntimeError("Reference and candidate Cal label vectors differ")
        yes = np.isin(ref_cal[lab_key], positives)
        k = int((ref_cal[scores_key][yes] >= reference_policy[f"tau_{role}"]).sum())
        result[role] = closest_recall_policy(arm_cal[scores_key][yes], k)
    return result


def _score_sealed_dir(arm_dir, gold_df, eval_can_lp_map, recall_policy):
    pred_df = read_predictions(arm_dir)
    merged = keyed_merge(pred_df, gold_df)
    summary = summarize_cohorts(merged)
    sealed_eval = split_calibrated_summary(merged)
    if set(summary) != set(sealed_eval):
        raise RuntimeError(f"Independent counter cohort mismatch: {arm_dir}")
    for cohort, legacy in sealed_eval.items():
        ind = summary[cohort]
        if cohort == CONTROL:
            for k in ("detected", "n_corrupt"):
                if ind[k] != legacy[k]:
                    raise RuntimeError(f"Independent control count mismatch: {arm_dir}/{k}")
            continue
        pairs = [(ind, legacy)]
        if cohort == "spontaneous":
            pairs.append((ind["held6_only"], legacy["held6_only"]))
        for a, b in pairs:
            for cls in ("strict", "binary", "S", "O", "I"):
                if any(a[cls][k] != b[cls][k] for k in ("tp", "fp", "fn")):
                    raise RuntimeError(f"Independent count mismatch on {arm_dir}/{cohort}/{cls}")
        if cohort == "timit_core" and ind["fa_total"] != legacy["fa_total"]:
            raise RuntimeError(f"Independent native FA mismatch: {arm_dir}")

    # Blind-spot slice (can_lp > -0.2) per L2 cohort
    blind_spot: Dict[str, Any] = {}
    t_df = merged[merged["domain"] == "target"]
    for cohort in ("held6", "epadb_test", "saa27", "spontaneous"):
        sub = t_df[(t_df["cohort"] == cohort) & (t_df["gold_label"].isin(["C", "S", "O"]))]
        if sub.empty:
            continue
        clps = np.asarray(
            [eval_can_lp_map[(cohort, u, int(i))] for u, i in zip(sub["uid"], sub["index"], strict=True)],
            dtype=np.float32,
        )
        bs = sub[clps > -0.2]
        y = (bs["gold_label"] != "C").astype(np.int32).to_numpy()
        sc = bs["score"].to_numpy(dtype=np.float64)
        pred_err = (bs["call"] != "C").to_numpy()
        tp = int(((y == 1) & pred_err).sum())
        fp = int(((y == 0) & pred_err).sum())
        fn = int(((y == 1) & (~pred_err)).sum())
        ap = float(average_precision_score(y, sc)) if len(y) and len(set(y.tolist())) > 1 else None
        blind_spot[cohort] = {
            "n_slots": int(len(y)),
            "n_errors": int(y.sum()),
            "ap": ap,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": float(tp / max(1, tp + fn)),
            "precision": float(tp / max(1, tp + fp)),
        }

    # Worst Held-6 speakers (ABA, ASI)
    h6_sub = merged[merged["cohort"] == "held6"]
    by_spk = {}
    for spk in sorted(h6_sub["speaker"].unique()):
        s_df = h6_sub[h6_sub["speaker"] == spk]
        st = s_df[s_df["domain"] == "target"]
        sg = s_df[s_df["domain"] == "gap"]
        by_spk[spk] = independent_count(
            st["gold_label"].to_numpy(),
            st["call"].to_numpy(),
            sg["gold_label"].to_numpy(),
            sg["call"].to_numpy(),
        )

    recall_matched = None
    if recall_policy is not None:
        rm_df = apply_policy_df(merged, recall_policy["t"]["tau"], recall_policy["g"]["tau"])
        recall_matched = {"cal_policy": recall_policy, "cohorts": split_calibrated_summary(rm_df)}

    return {
        "cohorts": sealed_eval,
        "independent_recount_verified": True,
        "sealed_eval_control": sealed_eval.get(CONTROL),
        "blind_spot_can_lp_gt_neg02": blind_spot,
        "held6_by_speaker": by_spk,
        "cal_recall_matched": recall_matched,
    }


def _bootstrap_arm_vs_s0(arm_dir, s0_dir, gold_df, rm_policy=None):
    a = keyed_merge(read_predictions(arm_dir), gold_df)
    b = read_predictions(s0_dir)
    if rm_policy is not None:
        a = apply_policy_df(a, rm_policy["t"]["tau"], rm_policy["g"]["tau"])
    return bootstrap(a, b)


def _measure_backbone_latency(
    eval_prompts: Sequence[Mapping[str, Any]],
    backbones: Sequence[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Measure matched single-backbone GTX 1650 inference latency on short & Stella WAVs."""
    from .backbones import _load_backbone_model, forward_backbone_layers

    short_wavs = [r["audio_path"] for r in eval_prompts if r["cohort"] == "held6"][:12]
    stella_wavs = [r["audio_path"] for r in eval_prompts if r["cohort"] == "saa27"][:3]
    out: Dict[str, Any] = {}
    for bname in dict.fromkeys(backbones):
        spec = BACKBONE_SPECS[bname]
        release_gpu()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, fe = _load_backbone_model(spec, device)
        try:
            # 2 warmup
            for p in short_wavs[:2]:
                w, _ = sf.read(p, dtype="float32")
                forward_backbone_layers(spec, model, fe, w, spec.layers, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            short_ms = []
            for p in short_wavs[2:]:
                w, _ = sf.read(p, dtype="float32")
                t0 = time.perf_counter()
                forward_backbone_layers(spec, model, fe, w, spec.layers, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                short_ms.append((time.perf_counter() - t0) * 1000.0)
            stella_ms = []
            for p in stella_wavs:
                w, _ = sf.read(p, dtype="float32")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                t0 = time.perf_counter()
                forward_backbone_layers(spec, model, fe, w, spec.layers, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stella_ms.append((time.perf_counter() - t0) * 1000.0)
            peak = enforce_vram(device, limit=VRAM_CAP_MIB)
            out[bname] = {
                "short_p50_ms": float(np.median(short_ms)) if short_ms else None,
                "short_p95_ms": float(np.percentile(short_ms, 95)) if short_ms else None,
                "stella_p50_ms": float(np.median(stella_ms)) if stella_ms else None,
                "peak_mib": peak,
            }
        finally:
            del model
            release_gpu()
    return out


def _render_report_vi(results: Mapping[str, Any]) -> str:
    lines = [
        "# Báo cáo Chiến dịch Nâng cấp Biểu diễn Âm học & Backbone (`BACKBONE_REP_CAMPAIGN_20261122_v1`)",
        "",
        f"- **Split mode:** `{results['preflight']['split_mode']}` (`test_sealed_opened={results['preflight']['test_sealed_opened']}`)",
        f"- **Git HEAD:** `{results['preflight']['git_head']}`",
        f"- **Stage 1 Top-2 Backbone (chọn thuần trên Cal):** `{results['stage1_probe']['selected_top1']}`, `{results['stage1_probe']['selected_top2']}`",
        f"- **Stage 5 (thuần Cal):** `{', '.join(results['stage5_confirm']['selected_top2']) or 'không nhánh nào đạt'}`",
        f"- **Trạng thái:** `{results['status']}`",
        ("- Chỉ smoke kỹ thuật; nhãn Eval giả lập, mọi số điểm dưới đây không có giá trị khoa học."
         if not results['eval_gold_used'] else
         "- Eval là bộ chẩn đoán đã dùng; phần Spontaneous chính chỉ dùng lát Held-6. Không phải xác nhận tập mới."),
        "",
        "## 1. Stage 1 — Cal-Only Ridge Kill-Gate (phá điểm mù `can_lp > -0,2`)",
        "",
        "| Biểu diễn | Cal L2 tAP | Cal L2 Blind-Spot AP (`can_lp > -0.2`) | Cal L2 gAP | Cal L1 FA @ R70 | Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, p in results["stage1_probe"]["probes"].items():
        lines.append(
            f"| `{name}` | {p['cal_l2_target_ap']:.4f} | {p['cal_l2_blind_spot_ap']:.4f} | "
            f"{p['cal_l2_gap_ap']:.4f} | {p['cal_l1_fa_at_r70']*100:.2f}% | {p['composite_cal_score']:.4f} |"
        )

    lines += [
        "",
        "## 2. Xác nhận 3 Seed Ghép Cặp trên Eval (Native Cal-locked, TB ± SD)",
        "",
        "| Hệ thống | Held-6 F1 | EpaDB F1 | Spont (H6) F1 | SAA-27 F1 (tAP) | TIMIT FA/câu | Ctrl Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, agg in results["aggregate_3seed"].items():
        def _ms(k: str, fmt: str = ".2f") -> str:
            v = agg.get(k)
            if not v:
                return "—"
            return f"{v['mean']:{fmt}}±{v['sd']:{fmt}}"

        lines.append(
            f"| `{arm_name}` | {_ms('held6_f1')}% | {_ms('epadb_f1')}% | {_ms('spont_h6_f1')}% | "
            f"{_ms('saa27_f1')}% ({_ms('saa27_tap', '.4f')}) | {_ms('timit_fa', '.3f')} | {_ms('ctrl_recall')}% |"
        )

    lines += [
        "",
        "## 3. Tất cả các nhánh 1-seed sàng lọc (Stage 2–4, seed đầu tiên)",
        "",
        "| Nhánh | Held-6 F1 | Held-6 Blind-Spot AP | EpaDB F1 | SAA-27 F1 | TIMIT FA/câu | Ctrl Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, sc in results["single_seed_eval"].items():
        co = sc["cohorts"]
        h6_f1 = co.get("held6", {}).get("strict", {}).get("f1", 0.0) * 100.0
        bs_ap = sc["blind_spot_can_lp_gt_neg02"].get("held6", {}).get("ap")
        bs_s = f"{bs_ap:.4f}" if bs_ap is not None else "—"
        ep_f1 = co.get("epadb_test", {}).get("strict", {}).get("f1", 0.0) * 100.0
        sa_f1 = co.get("saa27", {}).get("strict", {}).get("f1", 0.0) * 100.0
        tm_fa = co.get("timit_core", {}).get("fa_per_utt", 0.0)
        ctrl_obj = sc.get("sealed_eval_control") or {}
        ct_r = (ctrl_obj.get("recall") if "recall" in ctrl_obj else ctrl_obj.get("strict", {}).get("r", 0.0)) * 100.0
        lines.append(
            f"| `{arm_name}` | {h6_f1:.2f}% | {bs_s} | {ep_f1:.2f}% | {sa_f1:.2f}% | {tm_fa:.3f} | {ct_r:.2f}% |"
        )

    lines += [
        "",
        "## 4. Độ trễ GPU GTX 1650 (chỉ encoder, batch 1; không phải độ trễ end-to-end)",
        "",
        "| Backbone | Short P50 (ms) | Short P95 (ms) | Stella P50 (ms) | Peak VRAM (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for bname, lat in results["latency_benchmark"].items():
        sp50 = f"{lat['short_p50_ms']:.1f}" if lat["short_p50_ms"] is not None else "—"
        sp95 = f"{lat['short_p95_ms']:.1f}" if lat["short_p95_ms"] is not None else "—"
        st50 = f"{lat['stella_p50_ms']:.1f}" if lat["stella_p50_ms"] is not None else "—"
        lines.append(f"| `{bname}` | {sp50} | {sp95} | {st50} | {lat['peak_mib']:.1f} |")

    lines += [
        "",
        "## 5. Bảo toàn & Xác minh",
        "- Mọi nhánh MDD dùng đúng cache CTC đã khóa của Koel; không đo lại PER đầu-cuối trong lượt này.",
        "- Toàn bộ dự đoán Eval được niêm phong SHA-256 trước khi ghép gold và kiểm đếm độc lập trên CPU.",
        "- `test_sealed.jsonl`, `GOAL.md`, và mọi checkpoint cũ (`aef0d8a`, `a9bb69ed…`) giữ nguyên bất biến.",
    ]
    if results["paired_gates"]:
        lines.extend(["", "## 6. Cổng chẩn đoán ghép cặp (Native và Recall-Matched)", ""])
        for name, gate in results["paired_gates"].items():
            lines.append(f"- `{name}`: **{'PASS' if gate['passed'] else 'NO_GO'}**; ΔF1 theo seed (điểm %): {gate['strict_f1_delta_pp_by_seed']}; native/control: {gate['both_operating_points_pass'] }.")
    return "\n".join(lines) + "\n"


def _aggregate_3seed_group(seed_evals: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    def _stat(vals: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}

    h6, ep, sp_h6, sa, sa_tap, tm, ctrl = [], [], [], [], [], [], []
    for _, ev in sorted(seed_evals.items()):
        co = ev["cohorts"]
        if "held6" in co:
            h6.append(co["held6"]["strict"]["f1"] * 100.0)
        if "epadb_test" in co:
            ep.append(co["epadb_test"]["strict"]["f1"] * 100.0)
        if "spontaneous" in co and "held6_only" in co["spontaneous"]:
            sp_h6.append(co["spontaneous"]["held6_only"]["strict"]["f1"] * 100.0)
        if "saa27" in co:
            sa.append(co["saa27"]["strict"]["f1"] * 100.0)
            if co["saa27"].get("target_ap") is not None:
                sa_tap.append(co["saa27"]["target_ap"])
        if "timit_core" in co:
            tm.append(co["timit_core"]["fa_per_utt"])
        if ev.get("sealed_eval_control"):
            ctrl_obj = ev["sealed_eval_control"]
            r_val = ctrl_obj["recall"] if "recall" in ctrl_obj else ctrl_obj.get("strict", {}).get("r", 0.0)
            ctrl.append(r_val * 100.0)

    out = {}
    for k, vals in [
        ("held6_f1", h6),
        ("epadb_f1", ep),
        ("spont_h6_f1", sp_h6),
        ("saa27_f1", sa),
        ("saa27_tap", sa_tap),
        ("timit_fa", tm),
        ("ctrl_recall", ctrl),
    ]:
        if vals:
            out[k] = _stat(vals)
    return out


def _paired_diagnostic_gate(cand, ref):
    """Native and recall-matched safety with a 3-seed effect on the three primary slices."""
    strict, tap, gap, safety = [], [], [], []

    def primary(co):
        return [co["held6"], co["epadb_test"], co["spontaneous"]["held6_only"]]

    for seed in sorted(ref):
        a, b = cand[seed], ref[seed]
        co, base = a["cohorts"], b["cohorts"]
        rm = a["cal_recall_matched"]["cohorts"]
        for key, series in (("target_ap", tap), ("gap_ap", gap)):
            pairs = [(x[key], y[key]) for x, y in zip(primary(co), primary(base), strict=True)]
            series.append(float(np.mean([x - y for x, y in pairs])) if all(None not in pair for pair in pairs) else None)
        strict.append(float(np.mean([x["strict"]["f1"] - y["strict"]["f1"]
                                    for x, y in zip(primary(co), primary(base), strict=True)])) * 100)
        safety.append(all(x["timit_core"]["fa_total"] <= base["timit_core"]["fa_total"]
                          and x[CONTROL]["recall"] >= 0.95 for x in (co, rm)))

    def hits(vals, margin):
        return len(vals) == 3 and all(v is not None and v > 0 for v in vals) and float(np.mean(vals)) >= margin

    effect = hits(strict, 0.5) or hits(tap, 0.01) or hits(gap, 0.01)
    return {"passed": bool(effect and all(safety)), "strict_f1_delta_pp_by_seed": [round(x, 4) for x in strict],
            "target_ap_delta_by_seed": tap, "gap_ap_delta_by_seed": gap, "both_operating_points_pass": safety}


def run_stage6_seal_recount_latency_report(
    out_dir: Path,
    state: Dict[str, Any],
    preflight: Mapping[str, Any],
    stage0_report: Mapping[str, Any],
    stage1_report: Mapping[str, Any],
    stage2_report: Mapping[str, Any],
    stage3_report: Mapping[str, Any],
    stage4_report: Mapping[str, Any],
    stage5_report: Mapping[str, Any],
    eval_prompts: Sequence[Dict[str, Any]],
    xe: Sequence[Dict[str, np.ndarray]],
    device: torch.device,
    *,
    seeds: Sequence[int] = SEEDS,
    smoke: bool = False,
) -> Dict[str, Any]:
    print("[Stage 6] Auditing all seals and locking recall policies before any gold join...", flush=True)
    templates = []
    for row in eval_prompts:
        n = len(row["canonical_phones"])
        for kind, phones in (("target", row["canonical_phones"]), ("gap", [None] * (n + 1))):
            templates.extend((row["cohort"], row["uid"], kind, i, row["speaker"], p) for i, p in enumerate(phones))
    expected = pd.DataFrame(templates, columns=KEY + ["speaker", "canonical_phone"]).set_index(KEY).sort_index()
    baselines = {str(seed): (out_dir / f"stage0_s0_clean/seed_{seed}", stage0_report["per_seed"][str(seed)]) for seed in seeds}
    single = {}
    for stage_sub, rep in (("stage2_dualrep", stage2_report), ("stage3_xattn", stage3_report), ("stage4_lora", stage4_report)):
        for name, s in rep["arms"].items():
            single[name] = (out_dir / stage_sub / name, s)
    confirmed = {}
    for name, seed_map in stage5_report["confirmed_3seed"].items():
        confirmed[name] = {
            seed: (Path(s["reused_from"]) if "reused_from" in s else (out_dir / f"stage5_confirm/{name}/seed_{seed}"), s)
            for seed, s in seed_map.items()
        }
    for name, key in (("D0_base_plus_no_kd", "distill_no_kd_3seed"), ("D1_distill_fast", "distill_fast_3seed")):
        if stage5_report[key]:
            confirmed[name] = {seed: (out_dir / f"stage5_confirm/{name}/seed_{seed}", s) for seed, s in stage5_report[key].items()}
    all_pairs = list(baselines.values()) + list(single.values()) + [p for group in confirmed.values() for p in group.values()]
    cards = {}
    for path, summary in all_pairs:
        if not path.resolve().is_relative_to(out_dir.resolve()):
            raise RuntimeError(f"Unexpected arm directory: {path}")
        if path not in cards:
            cards[path] = _verify_sealed_arm(path, expected, summary)

    policies = {}
    for path, s in all_pairs:
        base = baselines[str(s["seed"])][0]
        card, b_card = cards[path], cards[base]
        if card["train_order_sha256"] != b_card["train_order_sha256"]:
            # B and C intentionally differ in the number of full-corpus epochs; no equal-budget claim.
            if not s["arm"].startswith(("B", "C")):
                raise RuntimeError(f"Paired sample order mismatch: {path}")
        if path != base and path not in policies:
            policies[path] = _lock_recall_match(_read_cal(path), _read_cal(base), b_card["policy"])
    policy_path = out_dir / "PRE_GOLD_EVAL_LOCK.json"
    lock = {
        "scope": "synthetic test labels" if smoke else "authorized historical diagnostic Eval",
        "seals": {str(p.relative_to(out_dir)): sha_file(p / "PREDICTIONS_SEAL.json") for p in cards},
        "recall_matched_cal_policies": {str(p.relative_to(out_dir)): v for p, v in policies.items()},
    }
    if policy_path.exists():
        if read_json(policy_path) != lock:
            raise RuntimeError("Already frozen Eval or Cal digest changed")
    else:
        atomic_save_json(policy_path, lock)

    eval_gold_rows = (smoke_gold(eval_prompts) if smoke else
                      load_eval_gold_stage6_only(preflight["parent_eval_gold_sha256"], eval_prompts))
    gold_df = gold_frame(eval_gold_rows)
    eval_can_lp_map: Dict[Tuple[str, str, int], float] = {}
    for r, kf in zip(eval_prompts, xe, strict=True):
        clps = kf["target_scalars"][:, CAN_LP_IDX]
        for i in range(len(r["canonical_phones"])):
            eval_can_lp_map[(r["cohort"], r["uid"], i)] = float(clps[i])

    by_path = {}
    for path, info in all_pairs:
        if path in by_path:
            continue
        rm = policies.get(path)
        scored = _score_sealed_dir(path, gold_df, eval_can_lp_map, rm)
        if rm is not None:
            ref_path = baselines[str(info["seed"])][0]
            scored["bootstrap_vs_s0_clean"] = _bootstrap_arm_vs_s0(path, ref_path, gold_df)
            scored["cal_recall_matched"]["bootstrap_vs_s0_clean"] = _bootstrap_arm_vs_s0(path, ref_path, gold_df, rm)
        by_path[path] = scored
    s0_eval = {seed: by_path[path] for seed, (path, _) in baselines.items()}
    single_seed_eval = {name: by_path[path] for name, (path, _) in single.items()}
    confirmed_eval = {"S0_clean": s0_eval}
    for name, grp in confirmed.items():
        confirmed_eval[name] = {seed: by_path[path] for seed, (path, _) in grp.items()}
    if "D1_distill_fast" in confirmed:
        for seed, (p, _) in confirmed["D1_distill_fast"].items():
            by_path[p]["bootstrap_vs_no_kd"] = _bootstrap_arm_vs_s0(p, confirmed["D0_base_plus_no_kd"][seed][0], gold_df)

    aggregate_3seed = {name: _aggregate_3seed_group(ev_map) for name, ev_map in confirmed_eval.items()}

    lat_backbones = [stage1_report["selected_top1"], stage1_report["selected_top2"]]
    lat_backbones += [s["used_backbone"] for s in stage4_report["arms"].values()]
    if "D1_distill_fast" in confirmed:
        lat_backbones.append("wavlm_base_plus")
    latency = _measure_backbone_latency(eval_prompts, [b for b in lat_backbones if b is not None], device)
    gates = {} if smoke else {name: _paired_diagnostic_gate(confirmed_eval[name], s0_eval) for name in confirmed}

    results = {
        "status": "COMPLETE_SMOKE_PROCEDURAL_LABELS" if smoke else "COMPLETE_VERIFIED_DIAGNOSTIC",
        "pre_gold_lock_sha256": sha_file(policy_path),
        "verified_unique_arms": len(cards),
        "paired_gates": gates,
        "preflight": dict(preflight),
        "stage0_s0_clean": dict(stage0_report),
        "stage1_probe": dict(stage1_report),
        "stage5_confirm": {
            "cal_ranking": stage5_report["cal_ranking"],
            "selected_top2": stage5_report["selected_top2"],
        },
        "single_seed_eval": single_seed_eval,
        "confirmed_3seed_eval": confirmed_eval,
        "aggregate_3seed": aggregate_3seed,
        "latency_benchmark": latency,
        "koel_recognition": "Same read-only Koel frame cache for all arms; no new PER measurement",
        "eval_gold_used": not smoke,
        "test_sealed_opened": False,
    }
    atomic_save_json(out_dir / "RESULTS.json", results)
    report_md = _render_report_vi(results)
    (out_dir / "CAMPAIGN_REPORT_VI.md").write_text(report_md, encoding="utf-8")
    mark_done(out_dir, state, "stage6_final", {"results_sha256": sha_file(out_dir / "RESULTS.json")})
    print(f"[Stage 6] COMPLETE! Report written to {out_dir / 'CAMPAIGN_REPORT_VI.md'}", flush=True)
    return results
