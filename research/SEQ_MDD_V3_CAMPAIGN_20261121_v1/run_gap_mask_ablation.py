"""Isolated single-variable ablation: masking synthetic TIMIT Gap 'I' on frozen fix12_arch.

Freezes 100% of the shared encoder and target head from fix12_arch/control.pt,
re-initializes ONLY the gap branch from paired seeds (20261121, 20261122, 20261123),
and compares:
  - keep_timit_i : keep all Fit gap labels (including 897 synthetic TIMIT 'I')
  - mask_timit_i : mask ONLY synthetic TIMIT 'I' (domain=='l1' & y_g==1 -> -100),
                   retaining TIMIT 'N' negatives and all L2 gap labels.

No modality dropout, no target/encoder updates, identical gap init SHA and batch
order per seed, verified 0.0 frozen-parameter diff and exact target-prediction parity
with fix12_arch, independent recount, and Calibrate-locked Gap-FA budget comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F

from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    GAP_LABEL_TO_ID,
)

from .data import (
    COHORTS,
    GOAL_COMMIT,
    admitted,
    enforce_vram,
    save_json,
    sha_file,
)
from .repair_prompts import DEFAULT_REPAIR, DEFAULT_RUN, REPAIR_ID, ensure_separate
from .repaired_v2_arch import CodeFixConfig, RepairedCodeFixV2, scale_lp
from .replay_codefix_audit import independent_count
from .run_code_fixes_ablation import collate_codefix, extract_codefix_features
from .run_repaired_control import check_inputs, state_sha
from .scoring import gold_frame, keyed_merge, read_predictions, scores_to_rows, seal_predictions
from .training import domain, release_cuda

DEFAULT_CODEFIX = DEFAULT_REPAIR.parent / "code_fixes_123_20261121_v1"
DEFAULT_SEEDS = (20261121, 20261122, 20261123)
GAP_PARAM_PREFIXES = (
    "gap_layer_logits",
    "gap_layer_scale",
    "gap_h_norm.",
    "gap_h_proj.",
    "gap_lp_proj.",
    "gap_sc_proj.",
    "gap_ac_fuse.",
    "gap_head.",
    "bos_state",
    "eos_state",
)


def is_gap_param(name: str) -> bool:
    return name in GAP_PARAM_PREFIXES or name.startswith(GAP_PARAM_PREFIXES)


def split_param_shas(state_dict: dict[str, torch.Tensor]) -> tuple[str, str]:
    frozen = {k: v for k, v in state_dict.items() if not is_gap_param(k)}
    gap = {k: v for k, v in state_dict.items() if is_gap_param(k)}
    return state_sha(frozen), state_sha(gap)


def reset_gap_branch(model: RepairedCodeFixV2, seed: int) -> None:
    """Re-initialize ONLY gap-branch parameters deterministically from `seed`."""
    torch.manual_seed(seed)
    fresh = RepairedCodeFixV2(
        d_model=model.d_model,
        dropout=0.30,
        scalar_drop_p=0.0,
        gap_lp_drop_p=0.0,
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if is_gap_param(name):
                param.copy_(fresh.state_dict()[name])
    for name, param in model.named_parameters():
        param.requires_grad_(is_gap_param(name))


@torch.inference_mode()
def precompute_frozen_seq(
    model: RepairedCodeFixV2,
    rows,
    feats,
    norm,
    device: torch.device,
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Compute frozen encoder states `seq` (L, d_model) per utterance (eval mode)."""
    model.eval()
    seqs = []
    for start in range(0, len(rows), batch_size):
        rsub = rows[start : start + batch_size]
        fsub = feats[start : start + batch_size]
        inp, _ = collate_codefix(rsub, fsub, norm, device, labels=False)
        B, L, W, _ = inp["lp_win"].shape
        hp = model.target_h_norm(
            model._mix_layers(
                inp["h_peak"], model.target_layer_logits, model.target_layer_scale
            )
        )
        hw = model.target_h_norm(
            model._mix_layers(
                inp["h_win"], model.target_layer_logits, model.target_layer_scale
            )
        )
        e = model.phone_ctx_norm(
            torch.cat(
                [
                    model.phone_emb(inp["phone_ids"]),
                    model.ctx_emb(inp["prev_phone_ids"]),
                    model.ctx_emb(inp["next_phone_ids"]),
                ],
                dim=-1,
            )
        )
        hw_f = model.proj_h_win(hw) + model.h_pos_emb(model.h_pos_ids).view(
            1, 1, -1, model.d_model
        )
        qh = model.h_win_query(e).reshape(B * L, 1, model.d_model)
        kvh = hw_f.reshape(B * L, -1, model.d_model)
        hwa, _ = model.h_win_attn(qh, kvh, kvh, need_weights=False)
        hwa = model.h_win_out_norm(hwa.reshape(B, L, model.d_model))

        pos = model.pos_emb_norm(model.pos_emb(model.pos_ids)).view(1, 1, W, 16).expand(
            B, L, -1, -1
        )
        f_lp = model.win_fuse(
            torch.cat([model.proj_lp_win(scale_lp(inp["lp_win"])), pos], dim=-1)
        )
        ql = model.query_proj(e).reshape(B * L, 1, model.d_model)
        kvl = f_lp.reshape(B * L, W, model.d_model)
        lpa, _ = model.local_attn(ql, kvl, kvl, need_weights=False)
        lpa = model.lp_win_out_norm(lpa.reshape(B, L, model.d_model))

        slot_x = model.slot_proj(
            torch.cat(
                [
                    model.proj_h_peak(hp),
                    hwa,
                    lpa,
                    model.proj_lp_vit(scale_lp(inp["lp_vit"])),
                    e,
                    model.target_scalar_proj(inp["target_scalars"]),
                ],
                dim=-1,
            )
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            slot_x, inp["lengths"].cpu(), batch_first=True, enforce_sorted=False
        )
        seq_out, _ = model.seq_encoder(packed)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(
            seq_out, batch_first=True, total_length=L
        )
        seq = model.seq_norm(model.gru_out_norm(seq_out) + slot_x).cpu().numpy()
        for j, r in enumerate(rsub):
            n = len(r["canonical_phones"])
            seqs.append(seq[j, :n].astype(np.float32))
    return seqs


def prenorm_gap_hidden(feats) -> None:
    """Apply parameter-free per-layer LayerNorm to frozen h_gap once in-place."""
    for f in feats:
        x = torch.from_numpy(f["h_gap"].astype(np.float32))
        f["h_gap_normed"] = F.layer_norm(x, (x.shape[-1],)).numpy().astype(np.float16)


def gap_forward_from_seq(
    model: RepairedCodeFixV2,
    seq: torch.Tensor,
    h_gap: torch.Tensor,
    lp_gap_mean: torch.Tensor,
    lp_gap_max: torch.Tensor,
    gap_scalars: torch.Tensor,
    lengths: torch.Tensor,
    *,
    h_gap_prenormed: bool = False,
) -> torch.Tensor:
    B = seq.shape[0]
    if h_gap_prenormed:
        w = torch.softmax(model.gap_layer_logits, dim=0)
        mixed = torch.einsum("l,...ld->...d", w, h_gap.float()) * model.gap_layer_scale
    else:
        mixed = model._mix_layers(h_gap, model.gap_layer_logits, model.gap_layer_scale)
    hg = model.gap_h_norm(mixed)
    g_h = model.gap_h_proj(hg)
    g_lp = model.gap_lp_proj(
        torch.cat([scale_lp(lp_gap_mean), scale_lp(lp_gap_max)], dim=-1)
    )
    g_sc = model.gap_sc_proj(gap_scalars)
    gap_ac = model.gap_ac_fuse(torch.cat([g_h, g_lp, g_sc], dim=-1))
    left = torch.cat([model.bos_state.expand(B, 1, -1), seq], dim=1)
    eos_idx = lengths.view(B, 1, 1).expand(-1, -1, model.d_model)
    right = F.pad(seq, (0, 0, 0, 1)).scatter(
        1, eos_idx, model.eos_state.expand(B, 1, -1)
    )
    return model.gap_head(torch.cat([left, gap_ac, right], dim=-1))


def collate_gap_batch(rows, feats, seqs, g_mean, g_std, device: torch.device, *, labels: bool):
    lengths = [len(r["canonical_phones"]) for r in rows]
    max_l = max(lengths)
    B = len(rows)
    d_model = seqs[0].shape[-1]
    use_pre = "h_gap_normed" in feats[0]
    hg_key = "h_gap_normed" if use_pre else "h_gap"

    seq_t = np.zeros((B, max_l, d_model), dtype=np.float32)
    h_gap = np.zeros((B, max_l + 1, 6, 1024), dtype=np.float16)
    lp_gm = np.zeros((B, max_l + 1, 79), dtype=np.float32)
    lp_gx = np.zeros((B, max_l + 1, 79), dtype=np.float32)
    g_sc = np.zeros((B, max_l + 1, 8), dtype=np.float32)

    for i, (f, sq, L) in enumerate(zip(feats, seqs, lengths, strict=True)):
        seq_t[i, :L] = sq
        h_gap[i, : L + 1] = f[hg_key]
        lp_gm[i, : L + 1] = f["lp_gap_mean"]
        lp_gx[i, : L + 1] = f["lp_gap_max"]
        g_sc[i, : L + 1] = (f["gap_scalars"] - g_mean) / g_std

    inp = {
        "seq": torch.from_numpy(seq_t).to(device),
        "h_gap": torch.from_numpy(h_gap).to(device),
        "lp_gap_mean": torch.from_numpy(lp_gm).to(device),
        "lp_gap_max": torch.from_numpy(lp_gx).to(device),
        "gap_scalars": torch.from_numpy(g_sc).to(device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "h_gap_prenormed": use_pre,
    }
    if not labels:
        return inp, None, None
    y_g = np.full((B, max_l + 1), -100, dtype=np.int64)
    for i, (r, L) in enumerate(zip(rows, lengths, strict=True)):
        y_g[i, : L + 1] = [GAP_LABEL_TO_ID.get(a, -100) for a in r["gold_gaps"]]
    is_l1 = np.array([domain(r) == "l1" for r in rows], dtype=bool)
    return (
        inp,
        torch.from_numpy(y_g).to(device),
        torch.from_numpy(is_l1).to(device),
    )


def compute_isolated_gap_loss(
    gap_logits: torch.Tensor,
    gap_labels: torch.Tensor,
    is_l1: torch.Tensor,
    *,
    mask_timit_gap_i: bool,
    gap_pos_weight: float = 8.0,
) -> torch.Tensor:
    y = gap_labels
    if mask_timit_gap_i:
        y = torch.where(
            is_l1[:, None] & (gap_labels == 1),
            torch.full_like(gap_labels, -100),
            gap_labels,
        )
    gw = torch.tensor(
        [1.0, gap_pos_weight], device=gap_logits.device, dtype=gap_logits.dtype
    )
    if not (y != -100).any():
        return gap_logits.sum() * 0.0
    return F.cross_entropy(
        gap_logits.reshape(-1, 2), y.reshape(-1), weight=gw, ignore_index=-100
    )


@torch.inference_mode()
def predict_gap_probs(model, rows, feats, seqs, g_mean, g_std, device, batch_size=16):
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        rsub = rows[i : i + batch_size]
        inp, _, _ = collate_gap_batch(
            rsub, feats[i : i + batch_size], seqs[i : i + batch_size], g_mean, g_std, device, labels=False
        )
        gp = F.softmax(gap_forward_from_seq(model, **inp), dim=-1)[..., 1].cpu().numpy()
        for j, r in enumerate(rsub):
            out.append(gp[j, : len(r["canonical_phones"]) + 1])
    return out


def cal_gap_arrays(rows, probs):
    l2_y, l2_p, l1_p = [], [], []
    for r, p in zip(rows, probs, strict=True):
        if domain(r) == "l1":
            l1_p.extend(p.tolist())
        else:
            for g, sc in zip(r["gold_gaps"], p.tolist(), strict=True):
                if g != "unknown":
                    l2_y.append(int(g == "I"))
                    l2_p.append(sc)
    return np.asarray(l2_y, np.int32), np.asarray(l2_p, np.float32), np.asarray(l1_p, np.float32)


def select_cal_gap_policies(l2_y, l2_p, l1_p, *, l2_fp_cap: int = 41):
    """Lock 3 gap thresholds strictly on Calibrate (0 Eval touch)."""
    grid = np.unique(np.linspace(0.10, 0.98, 177))

    # 1. Cal L2 gap F1 max
    best_f1, best_tau = -1.0, 0.75
    for tg in grid:
        pred = l2_p >= tg
        tp = int((pred & (l2_y == 1)).sum())
        fp = int((pred & (l2_y == 0)).sum())
        fn = int(((~pred) & (l2_y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tg)

    # 2. Matched Cal L2 gap FP budget (<= l2_fp_cap, tie-safe: smallest tau with FP <= cap)
    neg_scores = l2_p[l2_y == 0]
    cand_l2 = [float(tg) for tg in grid if int((neg_scores >= tg).sum()) <= l2_fp_cap]
    tau_l2_cap = cand_l2[0] if cand_l2 else float(grid[-1])

    # 3. Matched Cal L1 TIMIT gap FA budget (<= 1 FA on 3,900 Cal TIMIT gaps ~ <= 0.26/1k)
    cand_l1 = [float(tg) for tg in grid if int((l1_p >= tg).sum()) <= 1]
    tau_l1_cap = cand_l1[0] if cand_l1 else float(grid[-1])

    def diag(tg):
        pred = l2_p >= tg
        tp = int((pred & (l2_y == 1)).sum())
        fp = int((pred & (l2_y == 0)).sum())
        fn = int(((~pred) & (l2_y == 1)).sum())
        return {
            "tau_g": round(tg, 4),
            "cal_l2_tp": tp,
            "cal_l2_fp": fp,
            "cal_l2_fn": fn,
            "cal_l2_i_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            "cal_l1_gap_fa": int((l1_p >= tg).sum()),
            "cal_l1_gap_fa_per_1k": float((l1_p >= tg).mean() * 1000.0),
        }

    return {
        "cal_f1": diag(best_tau),
        "cal_l2_fp_matched": diag(tau_l2_cap),
        "cal_l1_fa_le_1": diag(tau_l1_cap),
    }


def evaluate_gap_only(eval_rows, eval_gold_by_key, gap_probs_by_key, tau_g: float) -> dict:
    """Independent gap-only evaluation per cohort at fixed tau_g."""
    by_coh = {}
    for r in eval_rows:
        coh = r["cohort"]
        if coh == "timit_core_corrupt_ctrl":
            continue
        gg = eval_gold_by_key[(coh, r["uid"])]["gold_gaps"]
        gp = gap_probs_by_key[(coh, r["uid"])]
        by_coh.setdefault(coh, ([], [], 0))
        by_coh[coh][0].extend(gg)
        by_coh[coh][1].extend(gp.tolist())
        by_coh[coh] = (by_coh[coh][0], by_coh[coh][1], by_coh[coh][2] + 1)

    out = {}
    for coh, (gg_l, gp_l, n_u) in by_coh.items():
        gg = np.asarray(gg_l)
        gp = np.asarray(gp_l, dtype=np.float32)
        gv = gg != "unknown"
        gg_v, gp_v = gg[gv], gp[gv]
        call = np.where(gp_v >= tau_g, "I", "N")
        if coh == "timit_core":
            fa = int((call == "I").sum())
            out[coh] = {
                "fa_gap": fa,
                "gap_fa_per_1k": fa * 1000.0 / len(gg_v),
                "gap_fa_per_utt": fa / n_u,
            }
        else:
            y = (gg_v == "I").astype(np.int32)
            cnt = independent_count(
                np.array([], dtype=str), np.array([], dtype=str), gg_v, call
            )["I"]
            out[coh] = {
                "gap_ap": float(average_precision_score(y, gp_v)),
                "gap_roc_auc": float(roc_auc_score(y, gp_v)),
                "I": cnt,
            }
    return out


def run_ablation(
    run: Path,
    split: Path,
    base_dir: Path,
    out: Path,
    device: torch.device,
    *,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
):
    ensure_separate(run, out)
    split_sha = check_inputs(run, split)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()

    base_ckpt_path = base_dir / "fix12_arch/control.pt"
    base_eval_dir = base_dir / "eval_fix12_arch"
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=True)
    base_cfg = CodeFixConfig(**base_ckpt["config"])
    base_norm = {k: tuple(t.numpy() for t in v) for k, v in base_ckpt["norm"].items()}
    g_mean, g_std = base_norm["gap_scalars"]
    base_policy = base_ckpt["policy"]
    base_frozen_sha, _ = split_param_shas(base_ckpt["state_dict"])

    base_model = RepairedCodeFixV2(
        d_model=base_cfg.d_model, dropout=base_cfg.dropout, scalar_drop_p=0.0, gap_lp_drop_p=0.0
    )
    base_model.load_state_dict(base_ckpt["state_dict"], strict=True)
    base_model = base_model.to(device).eval()

    fit_rows = admitted(split, "fit.jsonl")
    cal_rows = admitted(split, "cal.jsonl")
    ev_rows = admitted(split, "eval_prompts.jsonl")
    ev_gold_rows = admitted(split, "eval_gold.jsonl")
    ev_gold_by_key = {(r["cohort"], r["uid"]): r for r in ev_gold_rows}
    cache = run / "frames"

    xf = extract_codefix_features(fit_rows, cache, "fit/gap_iso")
    xc = extract_codefix_features(cal_rows, cache, "cal/gap_iso")
    xe = extract_codefix_features(ev_rows, cache, "eval/gap_iso")

    print("Precomputing frozen encoder `seq` states from fix12_arch...", flush=True)
    seq_f = precompute_frozen_seq(base_model, fit_rows, xf, base_norm, device)
    seq_c = precompute_frozen_seq(base_model, cal_rows, xc, base_norm, device)
    seq_e = precompute_frozen_seq(base_model, ev_rows, xe, base_norm, device)
    prenorm_gap_hidden(xf)
    prenorm_gap_hidden(xc)
    prenorm_gap_hidden(xe)

    # Verify target-prediction parity on Eval against sealed eval_fix12_arch
    base_sealed = read_predictions(base_eval_dir)
    base_sealed_t = base_sealed[base_sealed["domain"] == "target"]
    gold_df = gold_frame(ev_gold_rows)

    # Also compute fix12_arch's Cal L2 gap FP count at its native tau_g=0.775 as the matched cap
    base_cal_probs = predict_gap_probs(
        base_model, cal_rows, xc, seq_c, g_mean, g_std, device, batch_size
    )
    bl2_y, bl2_p, _ = cal_gap_arrays(cal_rows, base_cal_probs)
    l2_fp_cap = int(((bl2_p >= base_policy["tau_g"]) & (bl2_y == 0)).sum())
    print(f"Cal L2 matched gap FP cap from fix12_arch@0.775: {l2_fp_cap}", flush=True)

    del base_model
    release_cuda(device)

    all_runs = {}
    for seed in seeds:
        all_runs[str(seed)] = {}
        init_shas = {}
        order_shas = {}
        for arm_name, mask_flag in (("keep_timit_i", False), ("mask_timit_i", True)):
            arm_dir = out / f"seed{seed}_{arm_name}"
            arm_dir.mkdir(parents=True, exist_ok=False)

            model = RepairedCodeFixV2(
                d_model=base_cfg.d_model,
                dropout=base_cfg.dropout,
                scalar_drop_p=0.0,
                gap_lp_drop_p=0.0,
            )
            model.load_state_dict(base_ckpt["state_dict"], strict=True)
            reset_gap_branch(model, seed)
            fr_sha_init, gp_sha_init = split_param_shas(model.state_dict())
            assert fr_sha_init == base_frozen_sha
            init_shas[arm_name] = gp_sha_init

            model = model.to(device)
            gap_params = [p for n, p in model.named_parameters() if is_gap_param(n)]
            opt = torch.optim.AdamW(gap_params, lr=8e-4, weight_decay=0.02)
            n_steps = math.ceil(len(fit_rows) / batch_size)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs * n_steps, eta_min=2e-5
            )
            rng = np.random.default_rng(seed)
            order_hash = hashlib.sha256()

            best_ap, best_ep, best_state = -1.0, 0, None
            hist = []
            for ep in range(1, epochs + 1):
                model.train()
                order = rng.permutation(len(fit_rows))
                order_hash.update(order.astype("<i8").tobytes())
                loss_sum = torch.zeros((), device=device)
                for start in range(0, len(order), batch_size):
                    ix = order[start : start + batch_size]
                    inp, y_g, is_l1 = collate_gap_batch(
                        [fit_rows[i] for i in ix],
                        [xf[i] for i in ix],
                        [seq_f[i] for i in ix],
                        g_mean,
                        g_std,
                        device,
                        labels=True,
                    )
                    opt.zero_grad(set_to_none=True)
                    logits = gap_forward_from_seq(model, **inp)
                    loss = compute_isolated_gap_loss(
                        logits, y_g, is_l1, mask_timit_gap_i=mask_flag, gap_pos_weight=8.0
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(gap_params, 1.0, error_if_nonfinite=True)
                    opt.step()
                    sched.step()
                    loss_sum += loss.detach()
                    enforce_vram(device)
                cprobs = predict_gap_probs(
                    model, cal_rows, xc, seq_c, g_mean, g_std, device, batch_size
                )
                cy, cp, _ = cal_gap_arrays(cal_rows, cprobs)
                cap = float(average_precision_score(cy, cp))
                hist.append(
                    {
                        "epoch": ep,
                        "mean_gap_loss": float(loss_sum / n_steps),
                        "cal_l2_gap_ap": cap,
                    }
                )
                if cap > best_ap:
                    best_ap, best_ep = cap, ep
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }

            order_shas[arm_name] = order_hash.hexdigest()
            fr_sha_end, _ = split_param_shas(best_state)
            assert fr_sha_end == base_frozen_sha, "Frozen encoder/target weights mutated!"
            model.load_state_dict(best_state)
            model.eval()

            # Verify full forward target_logits parity on 16 eval utterances (bit-identical target!)
            max_t_diff = 0.0
            with torch.inference_mode():
                for r_chk, f_chk in zip(ev_rows[:16], xe[:16], strict=True):
                    f_raw = {k: v for k, v in f_chk.items() if k != "h_gap_normed"}
                    inp_full, _ = collate_codefix([r_chk], [f_raw], base_norm, device, labels=False)
                    out_full = model(**inp_full)
                    tp = F.softmax(out_full["target_logits"][0], dim=-1).cpu().numpy()
                    sc_chk = tp[:, 1] + tp[:, 2]
                    ref = base_sealed_t[
                        (base_sealed_t["cohort"] == r_chk["cohort"])
                        & (base_sealed_t["uid"] == r_chk["uid"])
                    ].sort_values("index")["score"].to_numpy()
                    max_t_diff = max(max_t_diff, float(np.max(np.abs(sc_chk - ref))))
            assert max_t_diff < 1e-5, f"Target scores drifted: {max_t_diff}"

            # Lock 3 Calibrate policies strictly on Calibrate
            cprobs = predict_gap_probs(
                model, cal_rows, xc, seq_c, g_mean, g_std, device, batch_size
            )
            cy, cp, cl1 = cal_gap_arrays(cal_rows, cprobs)
            policies = select_cal_gap_policies(cy, cp, cl1, l2_fp_cap=l2_fp_cap)

            # Save checkpoint + seal full predictions (target from fix12_arch + new gap)
            ckpt_p = arm_dir / "control.pt"
            with ckpt_p.open("xb") as f:
                torch.save(
                    {
                        "repair_id": REPAIR_ID,
                        "arm": arm_name,
                        "seed": seed,
                        "mask_timit_gap_i": mask_flag,
                        "state_dict": best_state,
                        "norm": base_ckpt["norm"],
                        "policies": policies,
                    },
                    f,
                )

            eprobs = predict_gap_probs(
                model, ev_rows, xe, seq_e, g_mean, g_std, device, batch_size=1
            )
            # Seal predictions at native cal_f1 gap threshold (keeping exact fix12_arch target calls)
            native_pol = {**base_policy, "tau_g": policies["cal_f1"]["tau_g"]}
            card = {
                "repair_id": REPAIR_ID,
                "goal_commit": GOAL_COMMIT,
                "seed": seed,
                "arm": arm_name,
                "mask_timit_gap_i": mask_flag,
                "best_epoch": best_ep,
                "best_cal_gap_ap": best_ap,
                "frozen_params_sha256": fr_sha_end,
                "gap_init_sha256": gp_sha_init,
                "train_order_sha256": order_shas[arm_name],
                "target_score_max_abs_diff_vs_fix12": max_t_diff,
                "policies": policies,
                "checkpoint_sha256": sha_file(ckpt_p),
                "history": hist,
            }
            card_p = arm_dir / "CHECKPOINT_CARD.json"
            save_json(card_p, card)

            # Seal gap predictions in Parquet before scoring
            records = []
            for r, gp in zip(ev_rows, eprobs, strict=True):
                dummy_t = np.zeros((len(r["canonical_phones"]), 3), dtype=np.float32)
                gap_2 = np.stack([1.0 - gp, gp], axis=-1)
                for rec in scores_to_rows(r, dummy_t, gap_2, native_pol):
                    if rec["domain"] == "gap":
                        records.append(rec)
            seal_predictions(arm_dir, records, sha_file(card_p), name="eval_gap_predictions")

            # Read back sealed gap predictions and join gold for independent evaluation!
            sealed_gap = keyed_merge(
                read_predictions(arm_dir),
                gold_df[gold_df["domain"] == "gap"],
            )
            re_probs_by_key = {
                (c, u): g.sort_values("index")["score"].to_numpy()
                for (c, u), g in sealed_gap.groupby(["cohort", "uid"], sort=False)
            }

            eval_by_policy = {
                pname: evaluate_gap_only(
                    ev_rows, ev_gold_by_key, re_probs_by_key, pinfo["tau_g"]
                )
                for pname, pinfo in policies.items()
            }
            all_runs[str(seed)][arm_name] = {
                "best_epoch": best_ep,
                "best_cal_gap_ap": best_ap,
                "target_score_max_abs_diff_vs_fix12": max_t_diff,
                "policies": policies,
                "eval": eval_by_policy,
            }
            print(
                f"seed={seed} {arm_name:12s} ep={best_ep:2d} cal_gAP={best_ap:.4f} | "
                f"H6 gAP={eval_by_policy['cal_f1']['held6']['gap_ap']:.4f} "
                f"Epa={eval_by_policy['cal_f1']['epadb_test']['gap_ap']:.4f} "
                f"SAA={eval_by_policy['cal_f1']['saa27']['gap_ap']:.4f} "
                f"Spont={eval_by_policy['cal_f1']['spontaneous']['gap_ap']:.4f} | "
                f"TIMIT_gapFA(cal_f1/l2cap/l1cap)="
                f"{eval_by_policy['cal_f1']['timit_core']['fa_gap']}/"
                f"{eval_by_policy['cal_l2_fp_matched']['timit_core']['fa_gap']}/"
                f"{eval_by_policy['cal_l1_fa_le_1']['timit_core']['fa_gap']}",
                flush=True,
            )
            del model, opt, sched
            release_cuda(device)

        assert init_shas["keep_timit_i"] == init_shas["mask_timit_i"]
        assert order_shas["keep_timit_i"] == order_shas["mask_timit_i"]

    # Aggregate 3-seed mean +/- SD and paired deltas (mask - keep)
    agg = {}
    for pname in ("cal_f1", "cal_l2_fp_matched", "cal_l1_fa_le_1"):
        agg[pname] = {}
        for coh in COHORTS:
            if coh == "timit_core":
                vals = {
                    arm: [all_runs[str(s)][arm]["eval"][pname][coh]["fa_gap"] for s in seeds]
                    for arm in ("keep_timit_i", "mask_timit_i")
                }
                deltas = [b - a for a, b in zip(vals["keep_timit_i"], vals["mask_timit_i"])]
                agg[pname][coh] = {
                    "keep_fa_gap_seeds": vals["keep_timit_i"],
                    "mask_fa_gap_seeds": vals["mask_timit_i"],
                    "keep_mean_sd": [float(np.mean(vals["keep_timit_i"])), float(np.std(vals["keep_timit_i"]))],
                    "mask_mean_sd": [float(np.mean(vals["mask_timit_i"])), float(np.std(vals["mask_timit_i"]))],
                    "paired_delta_mask_minus_keep": deltas,
                }
            else:
                def collect(arm, field):
                    if field in ("gap_ap", "gap_roc_auc"):
                        return [all_runs[str(s)][arm]["eval"][pname][coh][field] for s in seeds]
                    return [all_runs[str(s)][arm]["eval"][pname][coh]["I"][field] for s in seeds]

                agg[pname][coh] = {}
                for field in ("gap_ap", "r", "p", "f1", "tp", "fp"):
                    vk = collect("keep_timit_i", field)
                    vm = collect("mask_timit_i", field)
                    agg[pname][coh][field] = {
                        "keep_seeds": vk,
                        "mask_seeds": vm,
                        "keep_mean_sd": [float(np.mean(vk)), float(np.std(vk))],
                        "mask_mean_sd": [float(np.mean(vm)), float(np.std(vm))],
                        "paired_delta_mean": float(np.mean([b - a for a, b in zip(vk, vm)])),
                        "paired_deltas": [float(b - a) for a, b in zip(vk, vm)],
                    }

    summary = {
        "scope": "isolated_single_variable_timit_gap_i_mask_on_frozen_fix12_arch",
        "repair_preflight_sha256": split_sha,
        "base_checkpoint_sha256": sha_file(base_ckpt_path),
        "base_frozen_params_sha256": base_frozen_sha,
        "cal_l2_matched_fp_cap": l2_fp_cap,
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "aggregate_3seed": agg,
        "per_seed": all_runs,
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    save_json(out / "GAP_MASK_ISOLATED_RESULTS.json", summary)
    print(f"Saved {out / 'GAP_MASK_ISOLATED_RESULTS.json'} ({summary['wall_s']}s)", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--splits", type=Path, default=DEFAULT_REPAIR / "splits")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_CODEFIX)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    run_ablation(
        a.legacy_run,
        a.splits,
        a.base_dir,
        a.out,
        torch.device(a.device),
        seeds=DEFAULT_SEEDS,
        epochs=a.epochs,
        batch_size=a.batch_size,
    )


if __name__ == "__main__":
    main()
