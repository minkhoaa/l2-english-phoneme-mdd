"""EXP-5: joint upper-layer q/v LoRA with exact frozen-prefix replay and native KD/PER guards."""

from __future__ import annotations

import argparse
import hashlib
from itertools import groupby
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from research.CONTROLLED_PILOT_AB_20261118_v1.run_pilot_ab import (
    LoRALinear,
    WEIGHTS_DIR,
)
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import ALLOPHONE_TOKEN_IDS
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    COHORTS,
    CONTROL,
    HERE,
    LAYERS,
    SEED,
    admitted,
    enforce_vram,
    forward_frames,
    frame_path,
    identity,
    read_frames,
    read_json,
    read_jsonl,
    save_json,
    sha_file,
    stable_sha,
    wav_input,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import (
    FOUR_LAYERS,
    OFFSETS,
    POINTS,
    build_kernel,
    gap_indices,
    pool_features,
    voronoi_bounds,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import (
    ap_auc,
    brief,
    evaluate_seal,
    read_predictions,
    scores_to_rows,
    seal_predictions,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.stages import (
    comparisons,
    guard_split,
    write_result,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import (
    Logits,
    cal_score,
    choose_policy,
    collate,
    domain,
    load_checkpoint,
    objective,
    probabilities,
    release_cuda,
    sample_probs,
    save_model,
)

PREFIX = 16
RANK = 8
ALPHA = 16


def add_lora(model):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Expected frozen base")
    enc = model.wav2vec2.encoder
    if (
        len(enc.layers) != 24
        or not model.config.do_stable_layer_norm
        or model.config.add_adapter
    ):
        raise ValueError("Unexpected base Koel encoder architecture")
    for i in range(PREFIX, 24):
        for proj in ("q_proj", "v_proj"):
            layer = enc.layers[i].attention
            setattr(
                layer,
                proj,
                LoRALinear(getattr(layer, proj), r=RANK, alpha=ALPHA, dropout=0.0),
            )
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def digest_weights(model):
    def sha(seq):
        h = hashlib.sha256()
        for n, p in seq:
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    frozen = sha((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    lm = sha(model.lm_head.named_parameters())
    return frozen, lm


@torch.no_grad()
def from_wav_prefix(model, row, device):
    w = model.wav2vec2
    enc = w.encoder
    x = w.feature_extractor(wav_input(row, device)).transpose(1, 2)
    x, _ = w.feature_projection(x)
    x = enc.dropout(x + enc.pos_conv_embed(x))
    low = {}
    for i in range(PREFIX):
        x = enc.layers[i](x, attention_mask=None)
        if isinstance(x, tuple):
            x = x[0]
        if i + 1 in (12, 15):
            low[i + 1] = x
    return x, low


def top_forward(model, x, low, *, checkpointed):
    enc = model.wav2vec2.encoder
    feats = dict(low)
    for i in range(PREFIX, 24):
        layer = enc.layers[i]
        if checkpointed:
            x = checkpoint(layer, x, attention_mask=None, use_reentrant=False)
        else:
            x = layer(x, attention_mask=None)
        if isinstance(x, tuple):
            x = x[0]
        if i + 1 in LAYERS:
            feats[i + 1] = x
    # The installed transformers recorder exposes the final block's pre-norm
    # hidden state in output_hidden_states[24]; only the CTC projection uses the norm.
    x = enc.layer_norm(x)
    return (
        torch.stack([feats[k][0] for k in LAYERS], dim=1).half().float(),
        model.lm_head(model.dropout(x))[0].log_softmax(-1),
    )


def fill_prefix(model, rows, cache, device):
    if cache.exists():
        raise FileExistsError(cache)
    cache.mkdir()
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        p = frame_path(cache, row)
        if not p.exists():
            x, low = from_wav_prefix(model, row, device)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as f:
                np.savez(
                    f,
                    x=x.cpu().numpy(),
                    h12=low[12].half().cpu().numpy(),
                    h15=low[15].half().cpu().numpy(),
                )
            enforce_vram(device)
        if i % 400 == 0 or i == len(rows):
            print(
                f"LoRA prefix {i}/{len(rows)} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )


def cached_prefix(cache, row, device):
    with np.load(frame_path(cache, row), allow_pickle=False) as f:
        return (
            torch.from_numpy(f["x"]).to(device),
            {
                k: torch.from_numpy(f[f"h{k}"]).to(device=device, dtype=torch.float32)
                for k in (12, 15)
            },
        )


def differentiable_pool(h, lp, phones, spec, kernel):
    """Backpropagate through pooled high layers; force-align and scalar/CTC features outside autograd."""
    raw = lp.detach().cpu().numpy()
    feat = pool_features(
        h.detach().cpu().numpy().astype(np.float16), raw, phones, spec, kernel
    )
    T = len(raw)
    L = len(phones)
    spans, _ = ctc_viterbi_align_allophone_aware(raw, phones)
    s, e, p = np.asarray(spans).T
    t = np.clip(p[:, None] + OFFSETS, 0, T - 1)
    dev = h.device

    def idx(x):
        return torch.as_tensor(x, device=dev, dtype=torch.long)

    def integral(x):
        return torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)], dim=0)

    def means(pre, lo, hi):
        return (
            ((pre[idx(hi)] - pre[idx(lo)]) / idx(hi - lo)[:, None, None]).half().float()
        )

    def window(x):
        return x[idx(t[:, POINTS])]

    pref = integral(h)
    if spec.window == "full":
        a, b = np.asarray(voronoi_bounds(p, T)).T
        four = idx(FOUR_LAYERS)
        hw = window(h).index_select(2, four)
        hv = means(pref, a, b).index_select(1, four)
        hp = hw.mean(1).half().float()
    else:
        hw = None
        hv = (
            window(h).mean(1).half().float()
            if spec.window == "mean"
            else means(pref, s, e + 1)
        )
        hp = h[idx(p)]
    ids = []
    ends = []
    for k in range(L + 1):
        a = e[k - 1] + 1 if k else 0
        b = s[k] - 1 if k < L else T - 1
        border = e[k - 1] if k else s[0]
        if spec.gap_core:
            skip = {0}
            if k:
                skip.update(ALLOPHONE_TOKEN_IDS[phones[k - 1]])
            if k < L:
                skip.update(ALLOPHONE_TOKEN_IDS[phones[k]])
            if not k:
                skip.add(64)
            ids.append(
                gap_indices(
                    a, b, border, T, raw, ~np.isin(np.arange(79), list(skip)), True
                )
            )
        else:
            if b < a:
                a, b = max(0, border - 1), min(T - 1, border + 1)
            ends.append((a, b + 1))
    if spec.gap_core:
        select = np.zeros((L + 1, 2), int)
        mask = np.zeros((L + 1, 2), np.float32)
        for k, x in enumerate(ids):
            select[k, : len(x)] = x
            mask[k, : len(x)] = 1.0
        keep = torch.from_numpy(mask).to(dev)[:, :, None, None]
        gh = ((h[idx(select)] * keep).sum(1) / keep.sum(1).clamp_min(1)).half().float()
    else:
        a, b = np.asarray(ends).T
        gh = means(pref, a, b)
    features = {"h_vit": hv, "h_peak": hp, "h_gap": gh}
    if hw is not None:
        features["h_window"] = hw
    return feat, features


def logits_and_loss(model, head, norm, spec, row, h, lp, device, kernel):
    static, acoustic = differentiable_pool(h, lp, row["canonical_phones"], spec, kernel)
    x, y = collate([row], [static], norm, device, labels=True)
    x.update({k: v.unsqueeze(0) for k, v in acoustic.items()})
    return objective(head(**x), y, spec)


def smoke_model(run, out, base, fit, device, kernel):
    out.mkdir()
    model = load_base_koel(device)
    adapters = add_lora(model)
    head, norm, spec, _ = load_checkpoint(base, device)
    head.train()
    old, lm = digest_weights(model)
    best = sorted(fit, key=lambda r: (-Path(r["audio_path"]).stat().st_size, r["uid"]))[
        :8
    ]
    tests = []
    max_res = 0.0
    max_alloc = 0.0
    opt = torch.optim.AdamW([*adapters.values(), *head.parameters()], lr=3e-5)
    try:
        # First verify that the split implementation matches the unmodified zero-B model.
        row = best[0]
        with torch.no_grad():
            x, lo = from_wav_prefix(model, row, device)
            h, lp = top_forward(model, x, lo, checkpointed=False)
            hp, pp = forward_frames(model, row, device)
            frame_lp_diff = float(np.max(np.abs(lp.cpu().numpy() - pp)))
            frame_h_diff = float(
                np.max(np.abs(h.cpu().numpy() - hp.astype(np.float32)))
            )
            if frame_lp_diff > 1e-4 or frame_h_diff > 1e-3:
                raise RuntimeError("split-autograd forward parity failed")
            del h, lp, hp, pp, x, lo
        for row in best:
            release_cuda(device)
            torch.cuda.reset_peak_memory_stats(device)
            x, lo = from_wav_prefix(model, row, device)
            h, lp = top_forward(model, x, lo, checkpointed=True)
            loss = logits_and_loss(model, head, norm, spec, row, h, lp, device, kernel)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                [*adapters.values(), *head.parameters()], 1.0
            )
            if (
                not torch.isfinite(grad)
                or not torch.isfinite(loss)
                or not any(
                    p.grad is not None and torch.count_nonzero(p.grad)
                    for p in adapters.values()
                )
            ):
                raise FloatingPointError("Missing or non-finite LoRA gradient")
            opt.step()
            opt.zero_grad(set_to_none=True)
            rv = torch.cuda.max_memory_reserved(device) / 2**20
            al = torch.cuda.max_memory_allocated(device) / 2**20
            max_res = max(max_res, rv)
            max_alloc = max(max_alloc, al)
            tests.append(
                {
                    "uid": row["uid"],
                    "loss": float(loss.detach()),
                    "reserved_mib": rv,
                    "allocated_mib": al,
                }
            )
            del loss, h, lp, x, lo
            if rv > 3400:
                raise RuntimeError(f"Smoke VRAM {rv} MiB > 3400 MiB")
        new, lm_after = digest_weights(model)
        if old != new or lm != lm_after:
            raise RuntimeError("Frozen weights changed during smoke")
        result = {
            "status": "PASS",
            "tests": tests,
            "peak_reserved_mib": max_res,
            "peak_allocated_mib": max_alloc,
            "base_weight_diff": 0.0,
            "lm_head_diff": 0.0,
            "split_lp_max_abs": frame_lp_diff,
            "split_h_max_abs": frame_h_diff,
        }
    except (RuntimeError, FloatingPointError) as exc:
        result = {
            "status": "BLOCKED",
            "tests": tests,
            "reason": str(exc),
            "peak_reserved_mib": max(
                max_res, torch.cuda.max_memory_reserved(device) / 2**20
            ),
            "peak_allocated_mib": max(
                max_alloc, torch.cuda.max_memory_allocated(device) / 2**20
            ),
        }
    save_json(out / "SMOKE_VRAM.json", result)
    del model, head, opt, adapters
    release_cuda(device)
    return result


@torch.no_grad()
def cal_inference(model, head, norm, spec, rows, cache, device, kernel):
    head.eval()
    scored = []
    for row in rows:
        h, lp = top_forward(
            model, *cached_prefix(cache, row, device), checkpointed=False
        )
        feats = pool_features(
            h.cpu().numpy().astype(np.float16),
            lp.cpu().numpy(),
            row["canonical_phones"],
            spec,
            kernel,
        )
        x, _ = collate([row], [feats], norm, device, labels=False)
        v = head(**x)
        v = [
            v[k][0].cpu().numpy()
            for k in ("target_logits", "gap_logits", "realized_logits")
        ]
        scored.append(Logits(*v, feats["phone_ids"]))
        enforce_vram(device)
    return scored


def joined_train(run, out, base, fit, cal, seed, epochs, device, kernel):
    model = load_base_koel(device)
    head, norm, spec, _ = load_checkpoint(base, device)
    cache = out / "prefix16"
    torch.cuda.reset_peak_memory_stats(device)
    fill_prefix(model, fit + cal, cache, device)
    torch.manual_seed(seed)
    adapters = add_lora(model)
    frozen, lm = digest_weights(model)
    opt = torch.optim.AdamW(
        [
            {"params": list(adapters.values()), "lr": 3e-5},
            {"params": head.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.02,
    )
    prob = sample_probs(fit) if spec.corpus_balance else None
    rng = np.random.default_rng(seed)
    best_score = -math.inf
    hist = []
    release_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        head.train()
        opt.zero_grad(set_to_none=True)
        indices = (
            rng.choice(len(fit), size=len(fit), p=prob, replace=True)
            if prob is not None
            else rng.permutation(len(fit))
        )
        losses = torch.zeros((), device=device)
        for i, k in enumerate(indices):
            row = fit[k]
            h, lp = top_forward(
                model, *cached_prefix(cache, row, device), checkpointed=True
            )
            loss = logits_and_loss(model, head, norm, spec, row, h, lp, device, kernel)
            if domain(row) == "l1":
                teacher = torch.from_numpy(read_frames(run / "frames", row)[1]).to(
                    device
                )
                if teacher.shape != lp.shape:
                    raise RuntimeError("Teacher/student frame alignment mismatch")
                kd = F.kl_div(lp, teacher, reduction="batchmean", log_target=True)
                loss = loss + 0.5 * kd
                del kd, teacher
            (loss / min(4, len(indices) - 4 * (i // 4))).backward()
            losses += loss.detach()
            del loss, h, lp
            if i % 4 == 3 or i + 1 == len(indices):
                grad = torch.nn.utils.clip_grad_norm_(
                    [*adapters.values(), *head.parameters()], 1.0
                )
                if not torch.isfinite(grad):
                    raise FloatingPointError("Non-finite LoRA gradients")
                opt.step()
                opt.zero_grad(set_to_none=True)
            enforce_vram(device)
            if (i + 1) % 500 == 0:
                print(
                    f"LoRA {ep}/{epochs} {i + 1}/{len(indices)} ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
        ll = cal_inference(model, head, norm, spec, cal, cache, device, kernel)
        s, t, g = cal_score(cal, ll)
        if s > best_score:
            best_score = s
            best_ep = ep
            best_logits = ll
            best_lora = {k: v.detach().cpu().clone() for k, v in adapters.items()}
            best_head = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
        hist.append(
            {
                "epoch": ep,
                "loss": float(losses / len(indices)),
                "cal_t_ap": t,
                "cal_g_ap": g,
                "wall_s": time.perf_counter() - t0,
            }
        )
        print("LoRA epoch", hist[-1], flush=True)
    if digest_weights(model) != (frozen, lm):
        raise RuntimeError("Frozen weights changed during joint training")
    with torch.no_grad():
        for k, v in adapters.items():
            v.copy_(best_lora[k])
    head.load_state_dict(best_head)
    policy = choose_policy(cal, best_logits, spec)
    with (out / "lora_adapter.pt").open("xb") as f:
        torch.save(best_lora, f)
    peak = enforce_vram(device)
    save_model(
        out,
        head,
        norm,
        spec,
        policy,
        seed,
        best_ep,
        extra={
            "lora_adapter_sha256": sha_file(out / "lora_adapter.pt"),
            "lora_layers_zero_based": list(range(PREFIX, 24)),
            "lora_r": RANK,
            "lora_alpha": ALPHA,
            "native_ctc_kd_lambda": 0.5,
            "fit_sha256": stable_sha(fit),
            "cal_sha256": stable_sha(cal),
            "history": hist,
            "peak_reserved_mib": peak,
            "immutable_backbone": frozen,
            "immutable_lm_head": lm,
            "gradient_accumulation": 4,
            "joint_runner_sha256": sha_file(__file__),
        },
    )
    return model, head.eval(), norm, spec, policy, peak


def guarded_per(out, tokens):
    from research.KOEL_FULL_CONFIG_BENCHMARK_20261120_v1.run_per_and_latency import (
        EPADB_TEST_MANIFEST,
        TIMIT_CORE_MANIFEST,
        _epadb_native79_per,
        _timit_per,
    )

    def per(source):
        return {
            "timit_surface39": _timit_per(source, read_jsonl(TIMIT_CORE_MANIFEST))[
                "surface39"
            ],
            "epadb_native79": _epadb_native79_per(
                source, read_jsonl(EPADB_TEST_MANIFEST)
            ),
        }

    base, student = per(tokens["base"]), per(tokens["lora"])
    if (
        base["timit_surface39"]["edits"] != 698
        or base["epadb_native79"]["edits"] != 3827
    ):
        raise RuntimeError(
            f"Cached PER baseline does not match archived Base Koel: {base}"
        )
    result = {
        "base": base,
        "lora": student,
        "guards": {
            "timit": student["timit_surface39"]["per"] <= 0.123,
            "epadb": student["epadb_native79"]["per"] <= 0.186,
        },
    }
    save_json(out / "PER_GUARD.json", result)
    return result


@torch.inference_mode()
def evaluate_joint(run, out, base, model, head, norm, spec, policy, device, kernel):
    def decode(lp, inv):
        return [inv[k] for k, _ in groupby(np.argmax(lp, axis=1).tolist()) if k != 0]

    inverse = {i: p for p, i in read_json(WEIGHTS_DIR / "vocab.json").items()}
    rows = admitted(run / "splits", "eval_prompts.jsonl")
    records = []
    tokens = {"lora": {}, "base": {}}
    perf = {}
    torch.cuda.reset_peak_memory_stats(device)
    for i, row in enumerate(rows, 1):
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        h, lp = forward_frames(model, row, device, verify=True)
        f = pool_features(h, lp, row["canonical_phones"], spec, kernel)
        x, _ = collate([row], [f], norm, device, labels=False)
        y = head(**x)
        y = [
            y[k][0].cpu().numpy()
            for k in ("target_logits", "gap_logits", "realized_logits")
        ]
        l = Logits(*y, f["phone_ids"])
        records.extend(scores_to_rows(row, *probabilities(l, row, policy), policy))
        torch.cuda.synchronize(device)
        perf.setdefault(row["cohort"], []).append((time.perf_counter() - t) * 1000)
        enforce_vram(device)
        if row["cohort"] in {"epadb_test", "timit_core"}:
            tokens["lora"][row["uid"]] = decode(lp, inverse)
            tokens["base"][row["uid"]] = decode(
                read_frames(run / "frames", row)[1], inverse
            )
        if i % 400 == 0:
            print(f"LoRA EVAL {i}/{len(rows)}", flush=True)
    card_sha = sha_file(out / "CHECKPOINT_CARD.json")
    seal_predictions(out, records, card_sha)
    save_json(out / "CTC_PREDICTIONS.json", tokens)
    save_json(
        out / "CTC_SEAL.json",
        {
            "checkpoint_card_sha256": card_sha,
            "sha256": sha_file(out / "CTC_PREDICTIONS.json"),
        },
    )
    boots = comparisons(run, base)
    ev, ci = evaluate_seal(out, run / "splits", boots)
    tr = read_predictions(out)
    from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import gold_frame, keyed_merge

    tr = keyed_merge(tr, gold_frame(admitted(run / "splits", "eval_gold.jsonl")))
    phone_ap = {}
    for name in COHORTS[:-1]:
        phone_ap[name] = {}
        for ph in ("IH", "AH", "AE", "T"):
            a = tr[
                (tr.cohort == name)
                & (tr.canonical_phone == ph)
                & (tr.gold_label != "unknown")
            ]
            phone_ap[name][ph] = ap_auc(
                a.gold_label.isin(["S", "O"]).to_numpy(), a.score.to_numpy()
            )[0]
    save_json(
        out / "EVAL_METRICS.json",
        {
            "eval": ev,
            "bootstrap": ci,
            "phone_ap": phone_ap,
            "latency": {
                k: {
                    "p50_ms": float(np.median(v)),
                    "p95_ms": float(np.quantile(v, 0.95)),
                    "n": len(v),
                }
                for k, v in perf.items()
            },
        },
    )
    per = guarded_per(out, tokens)
    return ev, ci, per


def run_exp5(run, device, seed=SEED, epochs=6, *, smoke_only=False, out_override=None):
    if device.type != "cuda":
        raise ValueError("EXP-5 requires GPU for the VRAM guard")
    parent = run / "exp4_multiseed"
    d = read_json(parent / "RESULTS.json")
    if d["status"] not in {"GO", "NO_GO"}:
        raise ValueError("EXP-4 not finished")
    base = Path(d["frozen_lora_comparator"])
    design = read_json(parent / "DESIGN.json")
    for role in ("fit", "cal"):
        if sha_file(parent / f"final_{role}.jsonl") != design[f"{role}_sha256"]:
            raise ValueError("EXP-4 split changed")
    fit, cal = (read_jsonl(parent / f"final_{role}.jsonl") for role in ("fit", "cal"))
    ev = [
        r
        for r in admitted(run / "splits", "eval_prompts.jsonl")
        if r["cohort"] != "spontaneous"
    ]
    guard_split(fit, cal, ev)
    out = out_override or run / "exp5_joint_lora"
    out.mkdir(parents=True, exist_ok=False)
    if len({identity(r) for r in fit}) <= 400:
        return write_result(
            out,
            {
                "experiment": "EXP-5",
                "status": "BLOCKED",
                "reason": "At most 400 admitted Fit speakers; not a matched EXP-4/5 test.",
            },
        )
    spec = read_json(base / "CHECKPOINT_CARD.json")["spec"]
    kernel = build_kernel(run / "ctc_delete.so") if spec["ollr"] else None
    result = smoke_model(run, out / "smoke", base, fit, device, kernel)
    save_json(out / "SMOKE_VRAM.json", result)
    if smoke_only or result["status"] != "PASS":
        return write_result(
            out,
            {
                "experiment": "EXP-5",
                "status": result["status"] if smoke_only else "BLOCKED",
                "reason": result.get("reason", "Smoke only; no joint Eval."),
            },
        )
    model, head, norm, spec, policy, peak = joined_train(
        run, out, base, fit, cal, seed, epochs, device, kernel
    )
    e, ci, per = evaluate_joint(
        run, out, base, model, head, norm, spec, policy, device, kernel
    )
    del model, head
    release_cuda(device)
    before = read_json(base / "RESULTS.json")["eval"]
    gain = {k: e[k]["strict"]["f1"] - before[k]["strict"]["f1"] for k in COHORTS[:-1]}
    ok = {
        "three_strict_gains_with_positive_bootstrap": sum(
            gain[k] >= 0.01 and ci["parent"][k]["ci95"][0] > 0 for k in gain
        )
        >= 3,
        "no_native_fa_increase": e["timit_core"]["fa_total"]
        <= before["timit_core"]["fa_total"],
        "vram_train_max_3300": peak <= 3300,
        "control_recall": e[CONTROL]["recall"] >= 0.9,
        "control_ap": e[CONTROL]["ap"] >= 0.95,
        **{f"per_{k}": x for k, x in per["guards"].items()},
    }
    go = all(ok.values())
    return write_result(
        out,
        {
            "experiment": "EXP-5",
            "status": "GO" if go else "NO_GO",
            "gates": ok,
            "promoted": str(out) if go else None,
            "frozen_comparator": str(base),
            "metrics_pct_or_fa": brief(e),
            "strict_f1_gain": gain,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", choices=["cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or a.epochs < 1:
        p.error("Need offline mode and positive epochs")
    run_exp5(
        a.run_dir,
        torch.device(a.device),
        a.seed,
        a.epochs,
        smoke_only=a.smoke_only,
        out_override=a.out_dir,
    )


if __name__ == "__main__":
    main()
