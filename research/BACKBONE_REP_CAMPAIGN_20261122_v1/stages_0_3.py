"""Stages 0, 1, 2, and 3 of BACKBONE_REP_CAMPAIGN_20261122_v1.

Zero Eval gold leakage: none of these stages ever open eval_gold.jsonl.
All architecture selection, epoch selection, and threshold locking happen strictly on Cal.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    CONTINUOUS_SCALAR_NAMES,
    compute_multitask_mdd_loss,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    enforce_vram,
    read_frames,
    read_json,
    sha_file,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import Spec
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import (
    collate_sf,
    infer_sf_logits,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import (
    scores_to_rows,
    seal_predictions,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.single_factor_models import (
    FACTOR_ARMS,
    SingleFactorV2,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import (
    Logits,
    cal_score,
    choose_policy,
    domain,
    probabilities,
)

from .backbones import (
    BACKBONE_SPECS,
    VRAM_CAP_MIB,
    compute_prototype_features,
    extract_charsiu_bounds,
    extract_full_frame_cache,
    extract_slot_cache,
    fit_phone_prototypes,
    koel_voronoi_spans,
    read_charsiu_feats,
    read_frame_cache,
    read_slot_cache,
    release_gpu,
)
from .common import (
    BASE_KOEL_CACHE,
    SEEDS,
    atomic_save_json,
    check_disk_guard,
    compute_cal_arrays,
    is_gap_param_name,
    mark_done,
    open_arm_dir,
    param_group_sha,
    write_checkpoint_card,
)
from .models import (
    FRAME_WIN_GAP,
    FRAME_WIN_TARGET,
    DualRepConfig,
    DualRepSingleFactorV2,
    FrameXAttnConfig,
    FrameXAttnMDD,
)

CAN_LP_IDX = CONTINUOUS_SCALAR_NAMES.index("can_lp_vit_max")
CANDIDATE_BACKBONES = ("wavlm_large", "wavlm_base_plus", "huper_mid", "whisper_small")
XATTN_MICRO_BATCH = 4
XATTN_SLOT_BUDGET = 256


def _seal_arm_eval(
    out_dir: Path,
    eval_prompts: Sequence[Mapping[str, Any]],
    eval_logits: Sequence[Logits],
    policy: Mapping[str, Any],
    card_sha: str,
) -> Dict[str, Any]:
    pred_records: List[Dict[str, Any]] = []
    for row, lg in zip(eval_prompts, eval_logits, strict=True):
        tp, gp = probabilities(lg, row, policy)
        # Compare stored scores and thresholds at the same precision as the recount.
        pred_records.extend(scores_to_rows(row, tp.astype(np.float64), gp.astype(np.float64), policy))
    return seal_predictions(out_dir, pred_records, card_sha)


def _save_cal_arrays(out_dir: Path, cal_l2: Dict[str, np.ndarray], cal_l1: Dict[str, np.ndarray]) -> None:
    np.savez(
        out_dir / "cal_arrays.npz",
        l2_t_sc=cal_l2["t_sc"],
        l2_t_g=cal_l2["t_g"],
        l2_g_sc=cal_l2["g_sc"],
        l2_g_g=cal_l2["g_g"],
        l1_t_sc=cal_l1["t_sc"],
        l1_g_sc=cal_l1["g_sc"],
    )


def _cal_metrics_summary(
    cal_rows: Sequence[Mapping[str, Any]],
    cal_logits: Sequence[Logits],
    policy: Mapping[str, Any],
    cal_koel_feats: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, float]:
    """Compute comprehensive Cal-only diagnostic metrics including blind-spot can_lp > -0.2."""
    sel, tap, gap = cal_score(cal_rows, cal_logits)
    bs_y, bs_s = [], []
    for r, lg, kf in zip(cal_rows, cal_logits, cal_koel_feats, strict=True):
        if domain(r) == "l1":
            continue
        tp, _ = probabilities(lg, r, policy)
        err_sc = tp[:, 1] + tp[:, 2]
        can_lp = kf["target_scalars"][:, CAN_LP_IDX]
        for i, lbl in enumerate(r["gold_targets"]):
            if lbl in {"C", "S", "O"} and can_lp[i] > -0.2:
                bs_y.append(0 if lbl == "C" else 1)
                bs_s.append(float(err_sc[i]))
    bs_ap = float(average_precision_score(bs_y, bs_s)) if bs_y and len(set(bs_y)) > 1 else 0.0
    return {
        "cal_sel": float(sel),
        "cal_t_ap": float(tap),
        "cal_g_ap": float(gap),
        "cal_blind_spot_ap": bs_ap,
        "cal_l2_strict_f1": float(policy["cal_l2_strict_f1"]),
        "cal_l1_fa_per_1k": float(policy["cal_l1_fa_per_1k"]),
        "tau_t": float(policy["tau_t"]),
        "tau_g": float(policy["tau_g"]),
    }


# ---------------------------------------------------------------------------
# Stage 0: Clean S0 3-Seed Baseline (SingleFactorV2 iso_no_hwin)
# ---------------------------------------------------------------------------
def run_stage0_s0_clean(
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
    seeds: Sequence[int] = SEEDS,
    epochs: int = 14,
    batch_size: int = 16,
) -> Dict[str, Any]:
    if "stage0_s0_clean" in state["completed"]:
        return read_json(out_dir / "stage0_s0_clean/STAGE0_S0_CLEAN.json")

    stage_dir = out_dir / "stage0_s0_clean"
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg = FACTOR_ARMS["iso_no_hwin"]
    seed_summaries: Dict[str, Any] = {}

    for seed in seeds:
        s_key = f"stage0_s0_seed_{seed}"
        s_dir = stage_dir / f"seed_{seed}"
        if s_key in state["completed"] and (s_dir / "SUMMARY.json").exists():
            seed_summaries[str(seed)] = read_json(s_dir / "SUMMARY.json")
            continue

        check_disk_guard(out_dir)
        open_arm_dir(s_dir)

        t0 = time.perf_counter()
        release_gpu()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        torch.manual_seed(seed)
        model = SingleFactorV2(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
        n_steps = max(1, math.ceil(len(fit_rows) / batch_size))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_steps, eta_min=2e-5)
        rng = np.random.default_rng(seed)
        order_hash = hashlib.sha256()
        hist, best_sel, best_ep, best_state = [], -math.inf, 0, None

        for ep in range(1, epochs + 1):
            print(f"[Stage 0 S0_clean seed={seed}] training epoch {ep}/{epochs}", flush=True)
            model.train()
            order = rng.permutation(len(fit_rows))
            order_hash.update(order.astype("<i8").tobytes())
            loss_sum = torch.zeros((), device=device)
            for start in range(0, len(order), batch_size):
                ix = order[start : start + batch_size]
                inp, labels = collate_sf(
                    [fit_rows[i] for i in ix], [xf[i] for i in ix], norm, device, labels=True
                )
                opt.zero_grad(set_to_none=True)
                loss = compute_multitask_mdd_loss(
                    model(**inp),
                    *labels,
                    gap_pos_weight=8.0,
                    lambda_gap=0.6,
                    lambda_realized=0.25,
                    native_fa_weight=1.5,
                )["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                sched.step()
                loss_sum += loss.detach()
                enforce_vram(device, limit=VRAM_CAP_MIB)
            sel, tap, gap = cal_score(cal_rows, infer_sf_logits(model, cal_rows, xc, norm, device, batch_size))
            print(f"[Stage 0 S0_clean seed={seed}] Cal tAP={tap:.4f} gAP={gap:.4f}", flush=True)
            hist.append({"epoch": ep, "loss": float(loss_sum / n_steps), "cal_t_ap": tap, "cal_g_ap": gap, "sel": sel})
            if sel > best_sel:
                best_sel, best_ep = sel, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        cal_logits = infer_sf_logits(model, cal_rows, xc, norm, device, batch_size)
        policy = choose_policy(cal_rows, cal_logits, Spec())
        cal_l2, cal_l1 = compute_cal_arrays(cal_rows, cal_logits, policy)
        cal_m = _cal_metrics_summary(cal_rows, cal_logits, policy, xc)
        eval_logits = infer_sf_logits(model, eval_prompts, xe, norm, device, batch_size)
        peak = enforce_vram(device, limit=VRAM_CAP_MIB)

        cp_path = s_dir / "control.pt"
        torch.save(
            {
                "arm": "s0_clean",
                "seed": seed,
                "config": cfg.as_dict(),
                "state_dict": best_state,
                "norm": {k: (v[0].tolist(), v[1].tolist()) for k, v in norm.items()},
                "policy": policy,
                "best_epoch": best_ep,
            },
            cp_path,
        )
        cp_sha = sha_file(cp_path)
        _save_cal_arrays(s_dir, cal_l2, cal_l1)
        card_sha = write_checkpoint_card(
            s_dir,
            {
                "arm": "s0_clean",
                "seed": seed,
                "checkpoint_sha256": cp_sha,
                "train_order_sha256": order_hash.hexdigest(),
                "cal_arrays_sha256": sha_file(s_dir / "cal_arrays.npz"),
                "best_epoch": best_ep,
                "policy": policy,
                "cal_metrics": cal_m,
            },
        )
        seal = _seal_arm_eval(s_dir, eval_prompts, eval_logits, policy, card_sha)
        summary = {
            "arm": "s0_clean",
            "seed": seed,
            "best_epoch": best_ep,
            "checkpoint_sha256": cp_sha,
            "seal_sha256": sha_file(s_dir / "PREDICTIONS_SEAL.json"),
            "cal_metrics": cal_m,
            "policy": policy,
            "peak_mib": peak,
            "wall_s": time.perf_counter() - t0,
            "history": hist,
            "sealed_cohorts": sorted(seal["files"].keys()),
        }
        atomic_save_json(s_dir / "SUMMARY.json", summary)
        mark_done(out_dir, state, s_key, {"dir": str(s_dir), "cal_l2_strict_f1": cal_m["cal_l2_strict_f1"]})
        seed_summaries[str(seed)] = summary
        print(
            f"[Stage 0 S0_clean seed={seed}] ep={best_ep} Cal F1={cal_m['cal_l2_strict_f1']:.2f}% "
            f"tAP={cal_m['cal_t_ap']:.4f} gAP={cal_m['cal_g_ap']:.4f} blindAP={cal_m['cal_blind_spot_ap']:.4f} "
            f"L1_FA/1k={cal_m['cal_l1_fa_per_1k']:.2f} ({summary['wall_s']:.1f}s)",
            flush=True,
        )
        del model
        release_gpu()

    out = {
        "stage": "stage0_s0_clean",
        "seeds": list(seeds),
        "per_seed": seed_summaries,
        "mean_cal_l2_strict_f1": float(np.mean([v["cal_metrics"]["cal_l2_strict_f1"] for v in seed_summaries.values()])),
        "mean_cal_t_ap": float(np.mean([v["cal_metrics"]["cal_t_ap"] for v in seed_summaries.values()])),
        "mean_cal_g_ap": float(np.mean([v["cal_metrics"]["cal_g_ap"] for v in seed_summaries.values()])),
        "mean_cal_l1_fa_per_1k": float(np.mean([v["cal_metrics"]["cal_l1_fa_per_1k"] for v in seed_summaries.values()])),
    }
    atomic_save_json(stage_dir / "STAGE0_S0_CLEAN.json", out)
    mark_done(out_dir, state, "stage0_s0_clean", {"mean_cal_l2_strict_f1": out["mean_cal_l2_strict_f1"]})
    return out


# ---------------------------------------------------------------------------
# Stage 1: Multi-Backbone Fit+Cal Feature Extraction & Cal-Only Ridge Kill-Gate
# ---------------------------------------------------------------------------
def _flatten_probe_slots(
    rows: Sequence[Mapping[str, Any]],
    koel_feats: Sequence[Mapping[str, np.ndarray]],
    *,
    ext_cache_dir: Optional[Path] = None,
    charsiu_dir: Optional[Path] = None,
    proto: Optional[Mapping[str, np.ndarray]] = None,
    max_slots: Optional[int] = None,
    seed: int = 20261122,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_target, y_target, is_l2_mask, can_lp) for valid target slots."""
    xs, ys, is_l2s, can_lps = [], [], [], []
    # Deterministic utterance subsample if max_slots set
    order = np.arange(len(rows))
    if max_slots is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
    count = 0
    for idx in order:
        r = rows[int(idx)]
        kf = koel_feats[int(idx)]
        l2_flag = domain(r) != "l1"
        # Baseline Koel vector: target_scalars (13) + H18 peak (1024, layer idx 2 in (12,15,18,20,22,24))
        parts = [kf["target_scalars"].astype(np.float32), kf["h_peak"][:, 2, :].astype(np.float32)]
        if ext_cache_dir is not None:
            sc = read_slot_cache(ext_cache_dir, r)
            mid_i = sc["slot_mean"].shape[1] // 2
            parts.append(sc["slot_mean"][:, mid_i, :].astype(np.float32))
            parts.append(sc["slot_peak"][:, mid_i, :].astype(np.float32))
        if charsiu_dir is not None:
            parts.append(read_charsiu_feats(charsiu_dir, r))
        if proto is not None and ext_cache_dir is not None:
            parts.append(compute_prototype_features(r, ext_cache_dir, proto))
        feat = np.concatenate(parts, axis=1)
        can_lp = kf["target_scalars"][:, CAN_LP_IDX].astype(np.float32)

        for i, lbl in enumerate(r["gold_targets"]):
            if lbl not in {"C", "S", "O"}:
                continue
            xs.append(feat[i])
            ys.append(0 if lbl == "C" else 1)
            is_l2s.append(l2_flag)
            can_lps.append(can_lp[i])
            count += 1
        if max_slots is not None and count >= max_slots:
            break
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(is_l2s, dtype=bool),
        np.asarray(can_lps, dtype=np.float32),
    )


def _flatten_gap_probe(
    rows: Sequence[Mapping[str, Any]],
    koel_feats: Sequence[Mapping[str, np.ndarray]],
    *,
    ext_cache_dir: Optional[Path] = None,
    max_slots: Optional[int] = None,
    seed: int = 20261122,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, is_l2s = [], [], []
    order = np.arange(len(rows))
    if max_slots is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
    count = 0
    for idx in order:
        r = rows[int(idx)]
        kf = koel_feats[int(idx)]
        l2_flag = domain(r) != "l1"
        parts = [kf["gap_scalars"].astype(np.float32), kf["h_gap"][:, 2, :].astype(np.float32)]
        if ext_cache_dir is not None:
            sc = read_slot_cache(ext_cache_dir, r)
            mid_i = sc["gap_mean"].shape[1] // 2
            parts.append(sc["gap_mean"][:, mid_i, :].astype(np.float32))
            parts.append(sc["gap_max"][:, mid_i, :].astype(np.float32))
        feat = np.concatenate(parts, axis=1)
        for i, lbl in enumerate(r["gold_gaps"]):
            if lbl not in {"N", "I"}:
                continue
            xs.append(feat[i])
            ys.append(0 if lbl == "N" else 1)
            is_l2s.append(l2_flag)
            count += 1
        if max_slots is not None and count >= max_slots:
            break
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(is_l2s, dtype=bool),
    )


def _run_ridge_probe(
    fit_rows,
    cal_rows,
    xf,
    xc,
    *,
    ext_cache_dir: Optional[Path] = None,
    charsiu_dir: Optional[Path] = None,
    proto: Optional[Mapping[str, np.ndarray]] = None,
    max_fit_slots: int = 32000,
) -> Dict[str, float]:
    X_tr, y_tr, _, _ = _flatten_probe_slots(
        fit_rows, xf, ext_cache_dir=ext_cache_dir, charsiu_dir=charsiu_dir, proto=proto, max_slots=max_fit_slots
    )
    X_va, y_va, l2_va, can_lp_va = _flatten_probe_slots(
        cal_rows, xc, ext_cache_dir=ext_cache_dir, charsiu_dir=charsiu_dir, proto=proto, max_slots=None
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    clf = RidgeCV(alphas=(50.0, 250.0, 1000.0, 5000.0, 20000.0))
    clf.fit(X_tr_s, y_tr.astype(np.float32))
    sc_va = clf.predict(X_va_s)

    l2_ap = float(average_precision_score(y_va[l2_va], sc_va[l2_va]))
    bs_mask = l2_va & (can_lp_va > -0.2)
    bs_ap = (
        float(average_precision_score(y_va[bs_mask], sc_va[bs_mask]))
        if bs_mask.any() and len(set(y_va[bs_mask].tolist())) > 1
        else 0.0
    )
    # TIMIT FA rate at 70% Cal L2 recall
    pos_scores = np.sort(sc_va[l2_va & (y_va == 1)])
    thr_70 = float(pos_scores[int(max(0, round(0.30 * (len(pos_scores) - 1))))]) if len(pos_scores) else 0.5
    l1_mask = ~l2_va
    l1_fa_rate = float((sc_va[l1_mask] >= thr_70).mean()) if l1_mask.any() else 0.0

    # Gap probe
    Xg_tr, yg_tr, _ = _flatten_gap_probe(fit_rows, xf, ext_cache_dir=ext_cache_dir, max_slots=min(max_fit_slots, 24000))
    Xg_va, yg_va, gl2_va = _flatten_gap_probe(cal_rows, xc, ext_cache_dir=ext_cache_dir, max_slots=None)
    gscaler = StandardScaler()
    gclf = RidgeCV(alphas=(50.0, 250.0, 1000.0, 5000.0, 20000.0))
    gclf.fit(gscaler.fit_transform(Xg_tr), yg_tr.astype(np.float32))
    gsc_va = gclf.predict(gscaler.transform(Xg_va))
    g_ap = (
        float(average_precision_score(yg_va[gl2_va], gsc_va[gl2_va]))
        if len(set(yg_va[gl2_va].tolist())) > 1
        else 0.0
    )

    composite = 0.50 * l2_ap + 0.30 * bs_ap + 0.20 * g_ap
    return {
        "cal_l2_target_ap": l2_ap,
        "cal_l2_blind_spot_ap": bs_ap,
        "cal_l2_gap_ap": g_ap,
        "cal_l1_fa_at_r70": l1_fa_rate,
        "composite_cal_score": float(composite),
        "n_fit_slots": int(len(y_tr)),
        "n_cal_blind_positives": int(y_va[bs_mask].sum()),
        "n_cal_gap_positives": int(yg_va[gl2_va].sum()),
        "target_ridge_alpha": float(clf.alpha_),
        "gap_ridge_alpha": float(gclf.alpha_),
        "dim": int(X_tr.shape[1]),
    }


def probe_gate(probes, backbones):
    base = probes["koel_only"]
    gates = {}
    for name in backbones:
        p = probes[f"koel+{name}"]
        delta = {k: p[k] - base[k] for k in (
            "cal_l2_target_ap", "cal_l2_blind_spot_ap", "cal_l2_gap_ap", "cal_l1_fa_at_r70"
        )}
        # Advance on the target blind slice or gaps, with a native FA guard.
        guard = delta["cal_l1_fa_at_r70"] <= 0.002 and delta["cal_l2_target_ap"] >= -0.005
        target = p["n_cal_blind_positives"] >= 25 and delta["cal_l2_blind_spot_ap"] >= 0.01 and delta["cal_l2_target_ap"] >= 0.005
        gap = p["n_cal_gap_positives"] >= 25 and delta["cal_l2_gap_ap"] >= 0.02
        gates[name] = {"guard": guard, "target": target, "gap": gap, "passed": guard and (target or gap), "delta": delta}
    return gates


def run_stage1_backbone_probe(
    out_dir: Path,
    state: Dict[str, Any],
    fit_rows: Sequence[Dict[str, Any]],
    cal_rows: Sequence[Dict[str, Any]],
    eval_prompts: Sequence[Dict[str, Any]],
    xf: Sequence[Dict[str, np.ndarray]],
    xc: Sequence[Dict[str, np.ndarray]],
    device: torch.device,
    *,
    backbones: Sequence[str] = CANDIDATE_BACKBONES,
    max_fit_slots: int = 32000,
    smoke: bool = False,
) -> Dict[str, Any]:
    stage_dir = out_dir / "stage1_probe"
    if "stage1_probe" in state["completed"] and (stage_dir / "STAGE1_PROBE.json").exists():
        return read_json(stage_dir / "STAGE1_PROBE.json")

    stage_dir.mkdir(parents=True, exist_ok=True)
    caches_root = out_dir / "caches"
    fit_cal_rows = list(fit_rows) + list(cal_rows)

    extract_receipts: Dict[str, Any] = {}
    for bname in backbones:
        check_disk_guard(out_dir)
        b_dir = caches_root / f"slots_{bname}"
        rec = extract_slot_cache(bname, fit_cal_rows, BASE_KOEL_CACHE, b_dir, device)
        extract_receipts[f"{bname}_fit_cal"] = rec

    # 3. Cal-only Ridge Probe Kill-Gate
    probes: Dict[str, Any] = {}
    print("[Stage 1] Running Cal-only Ridge Kill-Gate probes...", flush=True)
    probes["koel_only"] = _run_ridge_probe(fit_rows, cal_rows, xf, xc, max_fit_slots=max_fit_slots)
    print(f"  koel_only: {probes['koel_only']}", flush=True)

    for bname in backbones:
        b_dir = caches_root / f"slots_{bname}"
        probes[f"koel+{bname}"] = _run_ridge_probe(
            fit_rows, cal_rows, xf, xc, ext_cache_dir=b_dir, max_fit_slots=max_fit_slots
        )
        print(f"  koel+{bname}: {probes[f'koel+{bname}']}", flush=True)

    # Rank backbones strictly by Cal composite score
    ranked = sorted(
        backbones,
        key=lambda b: probes[f"koel+{b}"]["composite_cal_score"],
        reverse=True,
    )
    gates = probe_gate(probes, backbones)
    passing = [b for b in ranked if gates[b]["passed"]]
    # Smoke stresses the heaviest frame windows regardless of toy Cal scores.
    selected = ["wavlm_large", "whisper_small"] if smoke else passing[:2]
    top1, top2 = (selected + [None, None])[:2]
    print(f"[Stage 1] Cal gate: passing={passing}; selected={selected}; smoke={smoke}", flush=True)
    if top1 is not None:
        charsiu_dir = caches_root / "charsiu_10ms"
        extract_receipts["charsiu_fit_cal"] = extract_charsiu_bounds(fit_cal_rows, charsiu_dir, device)
        b_dir = caches_root / f"slots_{top1}"
        proto = fit_phone_prototypes(fit_rows, b_dir, layer_idx=len(BACKBONE_SPECS[top1].layers) // 2)
        np.savez(stage_dir / f"prototypes_{top1}.npz", **proto)
        probes[f"koel+{top1}+charsiu_proto"] = _run_ridge_probe(
            fit_rows, cal_rows, xf, xc, ext_cache_dir=b_dir, charsiu_dir=charsiu_dir,
            proto=proto, max_fit_slots=max_fit_slots,
        )
        extract_receipts["charsiu_eval"] = extract_charsiu_bounds(eval_prompts, charsiu_dir, device)
    for bname in selected:
        check_disk_guard(out_dir)
        b_dir = caches_root / f"slots_{bname}"
        extract_receipts[f"{bname}_eval"] = extract_slot_cache(bname, eval_prompts, BASE_KOEL_CACHE, b_dir, device)

    report = {
        "stage": "stage1_probe",
        "candidate_backbones": list(backbones),
        "ranked_by_cal": ranked,
        "cal_gate": gates,
        "smoke_gate_override": smoke,
        "selected_top1": top1,
        "selected_top2": top2,
        "probes": probes,
        "extract_receipts": extract_receipts,
    }
    atomic_save_json(stage_dir / "STAGE1_PROBE.json", report)
    mark_done(out_dir, state, "stage1_probe", {"top1": top1, "top2": top2})
    return report


# ---------------------------------------------------------------------------
# Stage 2: DualRepSingleFactorV2 (4 Single-Factor Arms + Phase-2 Frozen Gap)
# ---------------------------------------------------------------------------
def _collate_dualrep(
    rows: Sequence[Mapping[str, Any]],
    koel_feats: Sequence[Mapping[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    ext1_dir: Path,
    ext2_dir: Optional[Path] = None,
    charsiu_dir: Optional[Path] = None,
    proto: Optional[Mapping[str, np.ndarray]] = None,
    aux_norm: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    labels: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    args, base_labels = collate_sf(rows, koel_feats, norm, device, labels=labels)
    lengths = [len(r["canonical_phones"]) for r in rows]
    max_l = max(lengths)

    # Load external slot features lazily per batch from NVMe cache
    e1_list = [read_slot_cache(ext1_dir, r) for r in rows]
    for key, extra in [
        ("slot_mean", 0),
        ("slot_peak", 0),
        ("slot_std", 0),
        ("gap_mean", 1),
        ("gap_max", 1),
    ]:
        arr0 = e1_list[0][key]
        buf = np.zeros((len(rows), max_l + extra) + arr0.shape[1:], dtype=np.float16)
        for i, (item, L) in enumerate(zip(e1_list, lengths, strict=True)):
            buf[i, : L + extra] = item[key]
        args[f"ext1_{key}"] = torch.from_numpy(buf).to(device)

    if ext2_dir is not None:
        e2_list = [read_slot_cache(ext2_dir, r) for r in rows]
        for key, extra in [
            ("slot_mean", 0),
            ("slot_peak", 0),
            ("slot_std", 0),
            ("gap_mean", 1),
            ("gap_max", 1),
        ]:
            arr0 = e2_list[0][key]
            buf = np.zeros((len(rows), max_l + extra) + arr0.shape[1:], dtype=np.float16)
            for i, (item, L) in enumerate(zip(e2_list, lengths, strict=True)):
                buf[i, : L + extra] = item[key]
            args[f"ext2_{key}"] = torch.from_numpy(buf).to(device)

    if charsiu_dir is not None and proto is not None:
        aux_list = [
            np.concatenate([read_charsiu_feats(charsiu_dir, r), compute_prototype_features(r, ext1_dir, proto)], axis=1)
            for r in rows
        ]
        dim_a = aux_list[0].shape[1]
        buf = np.zeros((len(rows), max_l, dim_a), dtype=np.float32)
        for i, (a, L) in enumerate(zip(aux_list, lengths, strict=True)):
            if aux_norm is not None:
                a = (a - aux_norm[0]) / aux_norm[1]
            buf[i, :L] = a
        args["aux_scalars"] = torch.from_numpy(buf).to(device)

    return args, base_labels


@torch.inference_mode()
def infer_dualrep_logits(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    koel_feats: Sequence[Mapping[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int,
    *,
    ext1_dir: Path,
    ext2_dir: Optional[Path] = None,
    charsiu_dir: Optional[Path] = None,
    proto: Optional[Mapping[str, np.ndarray]] = None,
    aux_norm: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[Logits]:
    model.eval()
    out: List[Logits] = []
    prior_long = False
    for start, end in xattn_inference_ranges(rows, batch_size, slot_budget=448):
        sub_r = rows[start:end]
        sub_f = koel_feats[start:end]
        long_passage = max(len(r["canonical_phones"]) for r in sub_r) > 128
        if device.type == "cuda" and (long_passage or prior_long):
            torch.cuda.empty_cache()
        prior_long = long_passage
        inp, _ = _collate_dualrep(
            sub_r,
            sub_f,
            norm,
            device,
            ext1_dir=ext1_dir,
            ext2_dir=ext2_dir,
            charsiu_dir=charsiu_dir,
            proto=proto,
            aux_norm=aux_norm,
            labels=False,
        )
        res = model(**inp)
        tl = res["target_logits"].float().cpu().numpy()
        gl = res["gap_logits"].float().cpu().numpy()
        rl = res["realized_logits"].float().cpu().numpy()
        for i, r in enumerate(sub_r):
            L = len(r["canonical_phones"])
            out.append(Logits(tl[i, :L], gl[i, : L + 1], rl[i, :L], sub_f[i]["phone_ids"]))
    return out


def compute_aux_norm(
    fit_rows: Sequence[Mapping[str, Any]],
    ext1_dir: Path,
    charsiu_dir: Path,
    proto: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    mats = [
        np.concatenate([read_charsiu_feats(charsiu_dir, r), compute_prototype_features(r, ext1_dir, proto)], axis=1)
        for r in fit_rows
    ]
    cat = np.concatenate(mats, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std = np.maximum(cat.std(axis=0), 1e-4).astype(np.float32)
    return mean, std


def train_one_dualrep_arm(
    arm_name: str,
    cfg: DualRepConfig,
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
    ext1_dir: Path,
    ext2_dir: Optional[Path] = None,
    charsiu_dir: Optional[Path] = None,
    proto: Optional[Mapping[str, np.ndarray]] = None,
    epochs: int = 14,
    phase2_epochs: int = 8,
    batch_size: int = 16,
    teacher_logits: Optional[Sequence[Logits]] = None,
) -> Dict[str, Any]:
    check_disk_guard(arm_dir.parent)
    open_arm_dir(arm_dir)
    t0 = time.perf_counter()
    release_gpu()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    aux_norm = (
        compute_aux_norm(fit_rows, ext1_dir, charsiu_dir, proto)
        if (charsiu_dir is not None and proto is not None)
        else None
    )
    torch.manual_seed(seed)
    model = DualRepSingleFactorV2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    n_steps = max(1, math.ceil(len(fit_rows) / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_steps, eta_min=2e-5)
    rng = np.random.default_rng(seed)
    order_hash = hashlib.sha256()

    hist, best_sel, best_ep, best_state = [], -math.inf, 0, None

    # Same supervised objective as S0; the optional student has a separate KD loss.
    for ep in range(1, epochs + 1):
        print(f"[DualRep {arm_name} seed={seed}] training epoch {ep}/{epochs}", flush=True)
        model.train()
        order = rng.permutation(len(fit_rows))
        order_hash.update(order.astype("<i8").tobytes())
        loss_sum = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            sub_r = [fit_rows[i] for i in ix]
            sub_f = [xf[i] for i in ix]
            inp, labels = _collate_dualrep(
                sub_r,
                sub_f,
                norm,
                device,
                ext1_dir=ext1_dir,
                ext2_dir=ext2_dir,
                charsiu_dir=charsiu_dir,
                proto=proto,
                aux_norm=aux_norm,
                labels=True,
            )
            opt.zero_grad(set_to_none=True)
            out_dict = model(**inp)
            loss_mdd = compute_multitask_mdd_loss(
                out_dict,
                *labels,
                gap_pos_weight=8.0,
                lambda_gap=0.6,
                lambda_realized=0.25,
                native_fa_weight=1.5,
            )["loss"]
            loss = loss_mdd

            if teacher_logits is not None:
                # Soft KL distillation on target & gap logits (T=2.0)
                T = 2.0
                kd_t_sum = torch.zeros((), device=device)
                kd_g_sum = torch.zeros((), device=device)
                for bi, orig_i in enumerate(ix):
                    L = len(fit_rows[orig_i]["canonical_phones"])
                    t_lg = torch.from_numpy(teacher_logits[orig_i].target).to(device)
                    g_lg = torch.from_numpy(teacher_logits[orig_i].gap).to(device)
                    kd_t_sum = kd_t_sum + F.kl_div(
                        F.log_softmax(out_dict["target_logits"][bi, :L] / T, dim=-1),
                        F.softmax(t_lg / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)
                    kd_g_sum = kd_g_sum + F.kl_div(
                        F.log_softmax(out_dict["gap_logits"][bi, : L + 1] / T, dim=-1),
                        F.softmax(g_lg / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)
                loss = 0.65 * loss + 0.35 * (kd_t_sum + 0.5 * kd_g_sum) / len(ix)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            enforce_vram(device, limit=VRAM_CAP_MIB)

        cal_lg = infer_dualrep_logits(
            model,
            cal_rows,
            xc,
            norm,
            device,
            batch_size,
            ext1_dir=ext1_dir,
            ext2_dir=ext2_dir,
            charsiu_dir=charsiu_dir,
            proto=proto,
            aux_norm=aux_norm,
        )
        sel, tap, gap = cal_score(cal_rows, cal_lg)
        hist.append({"epoch": ep, "loss": float(loss_sum / n_steps), "cal_t_ap": tap, "cal_g_ap": gap, "sel": sel})
        if sel > best_sel:
            best_sel, best_ep = sel, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    cal_lg_p1 = infer_dualrep_logits(
        model, cal_rows, xc, norm, device, batch_size,
        ext1_dir=ext1_dir, ext2_dir=ext2_dir, charsiu_dir=charsiu_dir, proto=proto, aux_norm=aux_norm,
    )
    _, p1_tap, p1_gap = cal_score(cal_rows, cal_lg_p1)

    # Phase 2: Freeze 100% of non-gap params and refine GapHead (keep only if Cal gAP improves)
    frozen_sha_before = param_group_sha(model, lambda n: not is_gap_param_name(n))
    for name, p in model.named_parameters():
        p.requires_grad = is_gap_param_name(name)
    gap_params = [p for p in model.parameters() if p.requires_grad]
    g_opt = torch.optim.AdamW(gap_params, lr=6e-4, weight_decay=0.02)
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=max(1, phase2_epochs * n_steps), eta_min=2e-5)
    best_g_ap, best_g_ep, best_g_state = p1_gap, 0, None
    g_rng = np.random.default_rng(seed + 999)
    pos_w = torch.tensor([1.0, 8.0], device=device)

    for gep in range(1, phase2_epochs + 1):
        model.eval()
        for name, module in model.named_children():
            if is_gap_param_name(name):
                module.train()
        order = g_rng.permutation(len(fit_rows))
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            inp, labels = _collate_dualrep(
                [fit_rows[i] for i in ix], [xf[i] for i in ix], norm, device,
                ext1_dir=ext1_dir, ext2_dir=ext2_dir, charsiu_dir=charsiu_dir, proto=proto, aux_norm=aux_norm, labels=True,
            )
            g_opt.zero_grad(set_to_none=True)
            g_logits = model(**inp)["gap_logits"]
            g_loss = F.cross_entropy(g_logits.reshape(-1, 2), labels[1].reshape(-1), weight=pos_w, ignore_index=-100)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gap_params, 1.0, error_if_nonfinite=True)
            g_opt.step()
            g_sched.step()
        cal_lg_g = infer_dualrep_logits(
            model, cal_rows, xc, norm, device, batch_size,
            ext1_dir=ext1_dir, ext2_dir=ext2_dir, charsiu_dir=charsiu_dir, proto=proto, aux_norm=aux_norm,
        )
        _, _, g_ap = cal_score(cal_rows, cal_lg_g)
        if g_ap > best_g_ap + 1e-4:
            best_g_ap, best_g_ep = g_ap, gep
            best_g_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    phase2_adopted = best_g_state is not None
    if phase2_adopted:
        model.load_state_dict(best_g_state)
    else:
        model.load_state_dict(best_state)
    frozen_sha_after = param_group_sha(model, lambda n: not is_gap_param_name(n))
    if frozen_sha_before != frozen_sha_after:
        raise RuntimeError("Phase 2 modified frozen target/encoder parameters!")

    model.eval()
    cal_logits = infer_dualrep_logits(
        model, cal_rows, xc, norm, device, batch_size,
        ext1_dir=ext1_dir, ext2_dir=ext2_dir, charsiu_dir=charsiu_dir, proto=proto, aux_norm=aux_norm,
    )
    target_drift = max(float(np.max(np.abs(a.target - b.target))) for a, b in zip(cal_lg_p1, cal_logits, strict=True))
    if target_drift > 1e-6:
        raise RuntimeError(f"Gap refinement changed target logits: {target_drift}")
    policy = choose_policy(cal_rows, cal_logits, Spec())
    cal_l2, cal_l1 = compute_cal_arrays(cal_rows, cal_logits, policy)
    cal_m = _cal_metrics_summary(cal_rows, cal_logits, policy, xc)
    eval_logits = infer_dualrep_logits(
        model, eval_prompts, xe, norm, device, batch_size,
        ext1_dir=ext1_dir, ext2_dir=ext2_dir, charsiu_dir=charsiu_dir, proto=proto, aux_norm=aux_norm,
    )
    peak = enforce_vram(device, limit=VRAM_CAP_MIB)

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    cp_path = arm_dir / "model.pt"
    torch.save(
        {
            "arm": arm_name,
            "seed": seed,
            "config": cfg.as_dict(),
            "state_dict": final_state,
            "norm": {k: (v[0].tolist(), v[1].tolist()) for k, v in norm.items()},
            "aux_norm": (aux_norm[0].tolist(), aux_norm[1].tolist()) if aux_norm is not None else None,
            "policy": policy,
            "best_epoch": best_ep,
            "phase2_adopted": phase2_adopted,
            "phase2_epoch": best_g_ep,
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
            "checkpoint_sha256": cp_sha,
            "train_order_sha256": order_hash.hexdigest(),
            "cal_arrays_sha256": sha_file(arm_dir / "cal_arrays.npz"),
            "best_epoch": best_ep,
            "phase2_adopted": phase2_adopted,
            "phase2_epoch": best_g_ep,
            "policy": policy,
            "cal_metrics": cal_m,
        },
    )
    seal = _seal_arm_eval(arm_dir, eval_prompts, eval_logits, policy, card_sha)
    summary = {
        "arm": arm_name,
        "seed": seed,
        "config": cfg.as_dict(),
        "best_epoch": best_ep,
        "phase1_cal_t_ap": p1_tap,
        "phase1_cal_g_ap": p1_gap,
        "phase2_target_max_abs_drift": target_drift,
        "phase2_adopted": phase2_adopted,
        "phase2_epoch": best_g_ep,
        "checkpoint_sha256": cp_sha,
        "seal_sha256": sha_file(arm_dir / "PREDICTIONS_SEAL.json"),
        "cal_metrics": cal_m,
        "policy": policy,
        "peak_mib": peak,
        "wall_s": time.perf_counter() - t0,
        "history": hist,
        "sealed_cohorts": sorted(seal["files"].keys()),
    }
    atomic_save_json(arm_dir / "SUMMARY.json", summary)
    print(
        f"[DualRep {arm_name} seed={seed}] ep={best_ep} (p2={phase2_adopted}:{best_g_ep}) "
        f"Cal F1={cal_m['cal_l2_strict_f1']:.2f}% tAP={cal_m['cal_t_ap']:.4f} "
        f"gAP={cal_m['cal_g_ap']:.4f} blindAP={cal_m['cal_blind_spot_ap']:.4f} "
        f"L1_FA/1k={cal_m['cal_l1_fa_per_1k']:.2f} ({summary['wall_s']:.1f}s)",
        flush=True,
    )
    del model
    release_gpu()
    return summary


def run_stage2_dualrep(
    out_dir: Path,
    state: Dict[str, Any],
    stage1_report: Mapping[str, Any],
    fit_rows: Sequence[Dict[str, Any]],
    cal_rows: Sequence[Dict[str, Any]],
    eval_prompts: Sequence[Dict[str, Any]],
    xf: Sequence[Dict[str, np.ndarray]],
    xc: Sequence[Dict[str, np.ndarray]],
    xe: Sequence[Dict[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    seed: int = SEEDS[0],
    epochs: int = 14,
    phase2_epochs: int = 8,
    batch_size: int = 16,
) -> Dict[str, Any]:
    stage_dir = out_dir / "stage2_dualrep"
    if "stage2_dualrep" in state["completed"] and (stage_dir / "STAGE2_DUALREP.json").exists():
        return read_json(stage_dir / "STAGE2_DUALREP.json")

    stage_dir.mkdir(parents=True, exist_ok=True)
    top1 = stage1_report["selected_top1"]
    top2 = stage1_report["selected_top2"]
    if top1 is None:
        out = {"stage": "stage2_dualrep", "status": "SKIPPED_CAL_GATE", "arms": {}}
        atomic_save_json(stage_dir / "STAGE2_DUALREP.json", out)
        mark_done(out_dir, state, "stage2_dualrep", out)
        return out
    s1 = BACKBONE_SPECS[top1]
    s2 = BACKBONE_SPECS[top2 or top1]
    caches = out_dir / "caches"
    ext1_dir = caches / f"slots_{top1}"
    ext2_dir = caches / f"slots_{top2}"
    charsiu_dir = caches / "charsiu_10ms"
    with np.load(out_dir / f"stage1_probe/prototypes_{top1}.npz") as z:
        proto1 = {k: z[k] for k in z.files}

    arm_specs = [
        (
            "A1_top1",
            DualRepConfig(arm="A1_top1", ext1_layers=len(s1.layers), ext1_dim=s1.hidden_dim),
            dict(ext1_dir=ext1_dir),
        ),
        (
            "A2_top2",
            DualRepConfig(arm="A2_top2", ext1_layers=len(s2.layers), ext1_dim=s2.hidden_dim),
            dict(ext1_dir=ext2_dir),
        ),
        (
            "A3_top1_charsiu_proto",
            DualRepConfig(
                arm="A3_top1_charsiu_proto",
                ext1_layers=len(s1.layers),
                ext1_dim=s1.hidden_dim,
                aux_scalar_dim=11,
            ),
            dict(ext1_dir=ext1_dir, charsiu_dir=charsiu_dir, proto=proto1),
        ),
        (
            "A4_top1_top2",
            DualRepConfig(
                arm="A4_top1_top2",
                ext1_layers=len(s1.layers),
                ext1_dim=s1.hidden_dim,
                ext2_layers=len(s2.layers),
                ext2_dim=s2.hidden_dim,
            ),
            dict(ext1_dir=ext1_dir, ext2_dir=ext2_dir),
        ),
    ]

    results: Dict[str, Any] = {}
    for arm_name, cfg, kwargs in arm_specs:
        if top2 is None and arm_name in {"A2_top2", "A4_top1_top2"}:
            continue
        a_key = f"stage2_{arm_name}"
        a_dir = stage_dir / arm_name
        if a_key in state["completed"] and (a_dir / "SUMMARY.json").exists():
            results[arm_name] = read_json(a_dir / "SUMMARY.json")
            continue
        res = train_one_dualrep_arm(
            arm_name,
            cfg,
            seed,
            a_dir,
            fit_rows,
            cal_rows,
            eval_prompts,
            xf,
            xc,
            xe,
            norm,
            device,
            epochs=epochs,
            phase2_epochs=phase2_epochs,
            batch_size=batch_size,
            **kwargs,
        )
        mark_done(out_dir, state, a_key, {"cal_l2_strict_f1": res["cal_metrics"]["cal_l2_strict_f1"]})
        results[arm_name] = res

    out = {"stage": "stage2_dualrep", "top1": top1, "top2": top2, "arms": results}
    atomic_save_json(stage_dir / "STAGE2_DUALREP.json", out)
    mark_done(out_dir, state, "stage2_dualrep", {"arms": sorted(results.keys())})
    return out


# ---------------------------------------------------------------------------
# Stage 3: FrameXAttnMDD (Monotonic Local Cross-Attention on 20ms Frame Windows)
# ---------------------------------------------------------------------------
def _extract_utt_frame_windows(
    row: Mapping[str, Any],
    ext_frames: np.ndarray,  # (T_ext, L, D) float16
    koel_cache: Path,
    *,
    include_koel_mid: bool = False,
) -> Dict[str, np.ndarray]:
    """Slice local +-8 frame windows around each target peak & inter-peak gap frames."""
    _, _, peaks, t_koel = koel_voronoi_spans(koel_cache, row)
    t_ext, n_layers, dim = ext_frames.shape
    scale = float(t_ext) / max(1.0, float(t_koel))
    n = len(peaks)
    half = FRAME_WIN_TARGET // 2
    offsets = np.arange(-half, half + 1)
    ext_peaks = np.rint(peaks * scale).astype(np.int32).clip(0, t_ext - 1)
    ext_ids = ext_peaks[:, None] + offsets
    t_mask = (ext_ids >= 0) & (ext_ids < t_ext)
    t_win = ext_frames[ext_ids.clip(0, t_ext - 1)]
    t_win[~t_mask] = 0

    g_win = np.zeros((n + 1, FRAME_WIN_GAP, n_layers, dim), dtype=np.float16)
    g_mask = np.zeros((n + 1, FRAME_WIN_GAP), dtype=bool)
    for g in range(n + 1):
        ga = 0 if g == 0 else int(ext_peaks[g - 1])
        gb = t_ext if g == n else int(ext_peaks[g]) + 1
        if gb <= ga:
            gb = min(t_ext, ga + 1)
        span_len = gb - ga
        if span_len <= FRAME_WIN_GAP:
            g_win[g, :span_len] = ext_frames[ga:gb]
            g_mask[g, :span_len] = True
        else:
            ixs = np.linspace(ga, gb - 1, FRAME_WIN_GAP).round().astype(np.int32)
            g_win[g] = ext_frames[ixs]
            g_mask[g] = True

    out = {"ext_t_win": t_win, "ext_t_mask": t_mask, "ext_g_win": g_win, "ext_g_mask": g_mask}
    if include_koel_mid:
        kh, _ = read_frames(koel_cache, row)
        # Layers 15 and 18 are indices 1 and 2 of LAYERS=(12,15,18,20,22,24)
        k_mid = kh[:, [1, 2], :]  # (T_koel, 2, 1024) float16
        ids = peaks[:, None] + offsets
        k_mask = (ids >= 0) & (ids < t_koel)
        k_win = k_mid[ids.clip(0, t_koel - 1)]
        k_win[~k_mask] = 0
        out.update(koel_t_win=k_win, koel_t_mask=k_mask)
    return out


def _collate_xattn(
    rows: Sequence[Mapping[str, Any]],
    koel_feats: Sequence[Mapping[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    frame_dir: Path,
    include_koel_mid: bool,
    labels: bool,
):
    args, base_labels = collate_sf(rows, koel_feats, norm, device, labels=labels)
    lengths = [len(r["canonical_phones"]) for r in rows]
    max_l = max(lengths)
    wins = [
        _extract_utt_frame_windows(
            r, read_frame_cache(frame_dir, r), BASE_KOEL_CACHE, include_koel_mid=include_koel_mid
        )
        for r in rows
    ]
    for key, extra, dtype in [
        ("ext_t_win", 0, np.float16),
        ("ext_t_mask", 0, bool),
        ("ext_g_win", 1, np.float16),
        ("ext_g_mask", 1, bool),
    ]:
        arr0 = wins[0][key]
        buf = np.zeros((len(rows), max_l + extra) + arr0.shape[1:], dtype=dtype)
        for i, (w, L) in enumerate(zip(wins, lengths, strict=True)):
            buf[i, : L + extra] = w[key]
        args[key] = torch.from_numpy(buf).to(device)

    if include_koel_mid:
        for key in ("koel_t_win", "koel_t_mask"):
            arr0 = wins[0][key]
            buf = np.zeros((len(rows), max_l) + arr0.shape[1:], dtype=arr0.dtype)
            for i, (w, L) in enumerate(zip(wins, lengths, strict=True)):
                buf[i, :L] = w[key]
            args[key] = torch.from_numpy(buf).to(device)

    return args, base_labels


def xattn_inference_ranges(rows, batch_size, slot_budget=XATTN_SLOT_BUDGET):
    """Bound padded slot cost while preserving input order; long passages run alone."""
    start = 0
    while start < len(rows):
        end, max_slots = start + 1, len(rows[start]["canonical_phones"]) + 1
        while end < min(len(rows), start + batch_size):
            candidate = max(max_slots, len(rows[end]["canonical_phones"]) + 1)
            if (end - start + 1) * candidate > slot_budget:
                break
            end += 1
            max_slots = candidate
        yield start, end
        start = end


@torch.inference_mode()
def infer_xattn_logits(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    koel_feats: Sequence[Mapping[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int,
    *,
    frame_dir: Path,
    include_koel_mid: bool,
) -> List[Logits]:
    model.eval()
    out: List[Logits] = []
    previous_long = False
    for start, end in xattn_inference_ranges(rows, min(batch_size, XATTN_MICRO_BATCH)):
        sub_r = rows[start:end]
        sub_f = koel_feats[start:end]
        long_passage = max(len(r["canonical_phones"]) for r in sub_r) > 128
        if device.type == "cuda" and (long_passage or previous_long):
            torch.cuda.empty_cache()
        previous_long = long_passage
        inp, _ = _collate_xattn(
            sub_r, sub_f, norm, device, frame_dir=frame_dir, include_koel_mid=include_koel_mid, labels=False
        )
        res = model(**inp)
        enforce_vram(device, limit=VRAM_CAP_MIB)
        tl = res["target_logits"].float().cpu().numpy()
        gl = res["gap_logits"].float().cpu().numpy()
        rl = res["realized_logits"].float().cpu().numpy()
        for i, r in enumerate(sub_r):
            L = len(r["canonical_phones"])
            out.append(Logits(tl[i, :L], gl[i, : L + 1], rl[i, :L], sub_f[i]["phone_ids"]))
    return out


def train_one_xattn_arm(
    arm_name: str,
    cfg: FrameXAttnConfig,
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
    frame_dir: Path,
    epochs: int = 12,
    micro_batch: int = XATTN_MICRO_BATCH,
    grad_accum: int = 4,
) -> Dict[str, Any]:
    check_disk_guard(arm_dir.parent)
    open_arm_dir(arm_dir)
    t0 = time.perf_counter()
    release_gpu()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(seed)
    model = FrameXAttnMDD(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.02)
    eff_bs = micro_batch * grad_accum
    n_steps = max(1, math.ceil(len(fit_rows) / eff_bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_steps, eta_min=2e-5)
    rng = np.random.default_rng(seed)
    order_hash = hashlib.sha256()

    hist, best_sel, best_ep, best_state = [], -math.inf, 0, None

    for ep in range(1, epochs + 1):
        print(f"[FrameXAttn {arm_name} seed={seed}] training epoch {ep}/{epochs}", flush=True)
        model.train()
        order = rng.permutation(len(fit_rows))
        order_hash.update(order.astype("<i8").tobytes())
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        micro_count = 0

        for start in range(0, len(order), micro_batch):
            ix = order[start : start + micro_batch]
            inp, labels = _collate_xattn(
                [fit_rows[i] for i in ix],
                [xf[i] for i in ix],
                norm,
                device,
                frame_dir=frame_dir,
                include_koel_mid=cfg.use_koel_frame_win,
                labels=True,
            )
            out_dict = model(**inp)
            loss_mdd = compute_multitask_mdd_loss(
                out_dict,
                *labels,
                gap_pos_weight=8.0,
                lambda_gap=0.6,
                lambda_realized=0.25,
                native_fa_weight=1.5,
            )["loss"]
            # Encourage non-collapsed attention without changing the shared MDD labels.
            loss = loss_mdd - cfg.entropy_reg_weight * out_dict["attn_entropy"]
            group_start = start // eff_bs * eff_bs
            batch_weight = len(ix) / min(eff_bs, len(order) - group_start)
            (loss * batch_weight).backward()
            loss_sum += float(loss.detach())
            micro_count += 1

            if micro_count % grad_accum == 0 or (start + micro_batch) >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                enforce_vram(device, limit=VRAM_CAP_MIB)

        cal_lg = infer_xattn_logits(
            model, cal_rows, xc, norm, device, micro_batch,
            frame_dir=frame_dir, include_koel_mid=cfg.use_koel_frame_win,
        )
        sel, tap, gap = cal_score(cal_rows, cal_lg)
        hist.append({"epoch": ep, "loss": loss_sum / max(1, micro_count), "cal_t_ap": tap, "cal_g_ap": gap, "sel": sel})
        if sel > best_sel:
            best_sel, best_ep = sel, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    cal_logits = infer_xattn_logits(
        model, cal_rows, xc, norm, device, micro_batch,
        frame_dir=frame_dir, include_koel_mid=cfg.use_koel_frame_win,
    )
    policy = choose_policy(cal_rows, cal_logits, Spec())
    cal_l2, cal_l1 = compute_cal_arrays(cal_rows, cal_logits, policy)
    cal_m = _cal_metrics_summary(cal_rows, cal_logits, policy, xc)
    eval_logits = infer_xattn_logits(
        model, eval_prompts, xe, norm, device, micro_batch,
        frame_dir=frame_dir, include_koel_mid=cfg.use_koel_frame_win,
    )
    peak = enforce_vram(device, limit=VRAM_CAP_MIB)

    cp_path = arm_dir / "model.pt"
    torch.save(
        {
            "arm": arm_name,
            "seed": seed,
            "config": cfg.as_dict(),
            "state_dict": best_state,
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
        "config": cfg.as_dict(),
        "best_epoch": best_ep,
        "checkpoint_sha256": cp_sha,
        "seal_sha256": sha_file(arm_dir / "PREDICTIONS_SEAL.json"),
        "cal_metrics": cal_m,
        "policy": policy,
        "peak_mib": peak,
        "wall_s": time.perf_counter() - t0,
        "history": hist,
        "sealed_cohorts": sorted(seal["files"].keys()),
    }
    atomic_save_json(arm_dir / "SUMMARY.json", summary)
    print(
        f"[FrameXAttn {arm_name} seed={seed}] ep={best_ep} Cal F1={cal_m['cal_l2_strict_f1']:.2f}% "
        f"tAP={cal_m['cal_t_ap']:.4f} gAP={cal_m['cal_g_ap']:.4f} blindAP={cal_m['cal_blind_spot_ap']:.4f} "
        f"L1_FA/1k={cal_m['cal_l1_fa_per_1k']:.2f} ({summary['wall_s']:.1f}s)",
        flush=True,
    )
    del model
    release_gpu()
    return summary


def run_stage3_frame_xattn(
    out_dir: Path,
    state: Dict[str, Any],
    stage1_report: Mapping[str, Any],
    fit_rows: Sequence[Dict[str, Any]],
    cal_rows: Sequence[Dict[str, Any]],
    eval_prompts: Sequence[Dict[str, Any]],
    xf: Sequence[Dict[str, np.ndarray]],
    xc: Sequence[Dict[str, np.ndarray]],
    xe: Sequence[Dict[str, np.ndarray]],
    norm: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    seed: int = SEEDS[0],
    epochs: int = 12,
) -> Dict[str, Any]:
    stage_dir = out_dir / "stage3_xattn"
    if "stage3_xattn" in state["completed"] and (stage_dir / "STAGE3_XATTN.json").exists():
        return read_json(stage_dir / "STAGE3_XATTN.json")

    stage_dir.mkdir(parents=True, exist_ok=True)
    top1 = stage1_report["selected_top1"]
    if top1 is None:
        out = {"stage": "stage3_xattn", "status": "SKIPPED_CAL_GATE", "arms": {}}
        atomic_save_json(stage_dir / "STAGE3_XATTN.json", out)
        mark_done(out_dir, state, "stage3_xattn", out)
        return out
    spec = BACKBONE_SPECS[top1]
    frame_dir = out_dir / f"caches/frames_{top1}"
    check_disk_guard(out_dir)

    all_rows = list(fit_rows) + list(cal_rows) + list(eval_prompts)
    frame_receipt = extract_full_frame_cache(top1, all_rows, frame_dir, device)

    arm_specs = [
        (
            "B1_xattn_top1",
            FrameXAttnConfig(
                arm="B1_xattn_top1",
                ext_layers=len(spec.frame_layers_stage3),
                ext_dim=spec.hidden_dim,
                use_koel_frame_win=False,
            ),
        ),
        (
            "B2_xattn_dual_frame",
            FrameXAttnConfig(
                arm="B2_xattn_dual_frame",
                ext_layers=len(spec.frame_layers_stage3),
                ext_dim=spec.hidden_dim,
                use_koel_frame_win=True,
            ),
        ),
    ]

    results: Dict[str, Any] = {}
    for arm_name, cfg in arm_specs:
        a_key = f"stage3_{arm_name}"
        a_dir = stage_dir / arm_name
        if a_key in state["completed"] and (a_dir / "SUMMARY.json").exists():
            results[arm_name] = read_json(a_dir / "SUMMARY.json")
            continue
        res = train_one_xattn_arm(
            arm_name,
            cfg,
            seed,
            a_dir,
            fit_rows,
            cal_rows,
            eval_prompts,
            xf,
            xc,
            xe,
            norm,
            device,
            frame_dir=frame_dir,
            epochs=epochs,
        )
        mark_done(out_dir, state, a_key, {"cal_l2_strict_f1": res["cal_metrics"]["cal_l2_strict_f1"]})
        results[arm_name] = res

    out = {"stage": "stage3_xattn", "top1": top1, "frame_receipt": frame_receipt, "arms": results}
    atomic_save_json(stage_dir / "STAGE3_XATTN.json", out)
    mark_done(out_dir, state, "stage3_xattn", {"arms": sorted(results.keys())})
    return out
