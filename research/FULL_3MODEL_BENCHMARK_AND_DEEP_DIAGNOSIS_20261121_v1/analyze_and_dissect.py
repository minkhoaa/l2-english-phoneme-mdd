"""Comprehensive cross-dataset scoring and forensic algorithmic dissection of Koel Base, R0, and SM13.

Loads:
  - Newly inferred cohorts: held6, spontaneous, epadb_test, timit_core (+ SM13 decomp on saa27 & dev18)
  - Existing sealed cohorts: dev18 (1,632 utts) and saa27 (27 spks) + saa18 (18 spks)

Computes:
  1. Full 6-cohort benchmark across 3 regimes (Historical Locked, Transferred DEV-18 Cal-F1,
     Threshold-Free PR-AUC / ROC-AUC + Oracle-tau Strict micro-F1) + paired speaker bootstrap CIs.
  2. Native L1 False-Alarm Stress Test on TIMIT Core (vs canonical39 and surface39).
  3. Forensic Dissection:
     - SM13: neural-only vs E4-only vs fused across all 6 cohorts; train-vs-unseen speaker split;
             learned 13-layer scalar-mixing weights; E4 P(S) ceiling & S->O collapse proof.
     - Koel Base: 35/75 unmapped vocab hemorrhage count & missed-error rate; ? abstention rate;
                  native L1 false-alarm phone profile.
     - R0: Base-call conditional rescue vs suppression breakdown; per-phone error profile;
           alignment-tier & utterance-length degradation; cross-domain score/calibration shift;
           gap-head blindness.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.KOEL_V2_FILTER_BENCHMARK_20261119_v1.run_v2_filter_benchmark import FOLDS  # noqa: F401

# Existing sealed runs
DEV18_RUN = REPO_ROOT / "research/KOEL_SM13_BENCHMARK_20261120_v1/runs/dev18_koel_vs_sm13_20261120"
SAA27_RUN = REPO_ROOT / "research/KOEL_SM13_BENCHMARK_20261120_v1/runs/saa27_koel_vs_sm13_20261120"
DEV18_GOLD = REPO_ROOT / "research/KOEL_MDD_BRANCH_ABLATION_20261119_v1/runs/saa_b_20261119/manifests/fit_gold.jsonl"
SAA27_GOLD = REPO_ROOT / "research/KOEL_MDD_BRANCH_ABLATION_20261119_v1/runs/saa_b_20261119/manifests/eval_gold.jsonl"
SM13_CKPT_DIR = REPO_ROOT / "research/KOEL_SM13_BENCHMARK_20261120_v1/seed101/experiments/rq1_sm13_formal_train_01/checkpoints"

# SM13 L2-ARCTIC split membership (from New-Method-Research/data/split_specs/L2_ARCTIC_SPLIT.yaml)
SM13_TRAIN_SPKS = {"ABA", "SKA", "BWC", "LXC", "ASI", "SVBI", "HKK", "HJK", "EBVS", "NJS", "HQTV", "PNV"}
SM13_DEV_SEARCH_SPKS = {"YBAA", "NCC", "RRBI", "YDCK"}
SM13_DEV_CAL_SPKS = {"ERMS", "THV"}
SM13_TEST_SEALED_SPKS = {"ZHAA", "TNI", "MBMPS", "TLV"}
SM13_UNSEEN_L2A_SPKS = {"TXHC", "YKWK"}

# Locked historical thresholds
SM13_LOCKED_TAU_T = 0.8569530248641968
SM13_LOCKED_TAU_G = 0.8660814762115479
R0_LOCKED_DEV18_TAU_T = 0.6194  # mean of 6-fold DEV-18 FP-budget locked thresholds (will read exact from json)

# Transferred DEV-18 Cal-F1 thresholds (mean across the 6 DEV-18 Calibrate folds, zero-touch on Eval!)
# Exact fold values from dev18_koel_vs_sm13_20261120/thresholds.json will be loaded dynamically.


def pr_roc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float | None, float | None]:
    mask = np.isfinite(scores)
    y = y_true[mask]
    s = scores[mask]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return None, None
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def score_decisions(
    gold_t: np.ndarray,
    pred_t: np.ndarray,
    gold_g: np.ndarray,
    pred_g: np.ndarray,
) -> Dict[str, Any]:
    """Compute strict micro-F1, per-class S/O/I P/R/F1, binary F1, S<->O confusions, excluding 'unknown'."""
    t_valid = gold_t != "unknown"
    g_valid = gold_g != "unknown"
    gt = gold_t[t_valid]
    pt = pred_t[t_valid]
    gg = gold_g[g_valid]
    pg = pred_g[g_valid]

    # Map '?' to C / N (no alert)
    pt_c = np.where(np.isin(pt, ["S", "O"]), pt, "C")
    pg_c = np.where(pg == "I", "I", "N")

    def prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "p": p, "r": r, "f1": f1}

    tp_s = int(np.sum((gt == "S") & (pt_c == "S")))
    fp_s = int(np.sum((gt != "S") & (pt_c == "S")))
    fn_s = int(np.sum((gt == "S") & (pt_c != "S")))

    tp_o = int(np.sum((gt == "O") & (pt_c == "O")))
    fp_o = int(np.sum((gt != "O") & (pt_c == "O")))
    fn_o = int(np.sum((gt == "O") & (pt_c != "O")))

    tp_i = int(np.sum((gg == "I") & (pg_c == "I")))
    fp_i = int(np.sum((gg != "I") & (pg_c == "I")))
    fn_i = int(np.sum((gg == "I") & (pg_c != "I")))

    s_to_o = int(np.sum((gt == "S") & (pt_c == "O")))
    o_to_s = int(np.sum((gt == "O") & (pt_c == "S")))

    strict_tp = tp_s + tp_o + tp_i
    strict_fp = fp_s + fp_o + fp_i
    strict_fn = fn_s + fn_o + fn_i

    # Binary error detection
    g_err_t = (gt == "S") | (gt == "O")
    p_err_t = (pt_c == "S") | (pt_c == "O")
    g_err_g = gg == "I"
    p_err_g = pg_c == "I"

    bin_tp = int(np.sum(g_err_t & p_err_t) + np.sum(g_err_g & p_err_g))
    bin_fp = int(np.sum((~g_err_t) & p_err_t) + np.sum((~g_err_g) & p_err_g))
    bin_fn = int(np.sum(g_err_t & (~p_err_t)) + np.sum(g_err_g & (~p_err_g)))

    return {
        "n_target_valid": int(len(gt)),
        "n_target_unknown": int(np.sum(~t_valid)),
        "n_gap_valid": int(len(gg)),
        "n_gap_unknown": int(np.sum(~g_valid)),
        "n_gold_errors": int(np.sum(g_err_t) + np.sum(g_err_g)),
        "strict": prf(strict_tp, strict_fp, strict_fn),
        "binary": prf(bin_tp, bin_fp, bin_fn),
        "S": prf(tp_s, fp_s, fn_s),
        "O": prf(tp_o, fp_o, fn_o),
        "I": prf(tp_i, fp_i, fn_i),
        "s_to_o": s_to_o,
        "o_to_s": o_to_s,
        "target_binary": prf(
            int(np.sum(g_err_t & p_err_t)),
            int(np.sum((~g_err_t) & p_err_t)),
            int(np.sum(g_err_t & (~p_err_t))),
        ),
    }


def apply_r0_tau(p_s: np.ndarray, p_o: np.ndarray, score: np.ndarray, tau_t: float) -> np.ndarray:
    err = score >= tau_t
    kind = np.where(p_o > p_s, "O", "S")
    return np.where(err, kind, "C")


def apply_sm13_tau(
    p_s: np.ndarray, p_o: np.ndarray, gap_i: np.ndarray, tau_t: float, tau_g: float, mode: str = "max"
) -> Tuple[np.ndarray, np.ndarray]:
    """mode='max' matches SM13's official apply_warning_policy: max(p_s, p_o) >= tau_t.
    mode='sum' checks p_s + p_o >= tau_t (diagnostic)."""
    conf = np.maximum(p_s, p_o) if mode == "max" else (p_s + p_o)
    kind = np.where(p_o > p_s, "O", "S")
    pred_t = np.where(conf >= tau_t, kind, "C")
    pred_g = np.where(gap_i >= tau_g, "I", "N")
    return pred_t, pred_g


def oracle_best_f1_1d(
    gold_t: np.ndarray,
    p_s: np.ndarray,
    p_o: np.ndarray,
    score: np.ndarray,
    gold_g: np.ndarray,
    fixed_pred_g: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """Upper-bound oracle threshold search (diagnostic only!) maximizing strict micro-F1."""
    best_f1 = -1.0
    best_tau = 0.5
    best_res = None
    grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), np.quantile(score[np.isfinite(score)], np.linspace(0.5, 0.995, 60))]))
    for tau in grid:
        pt = apply_r0_tau(p_s, p_o, score, float(tau))
        res = score_decisions(gold_t, pt, gold_g, fixed_pred_g)
        if res["strict"]["f1"] > best_f1:
            best_f1 = res["strict"]["f1"]
            best_tau = float(tau)
            best_res = res
    return best_tau, best_res


def oracle_best_f1_sm13(
    gold_t: np.ndarray,
    p_s: np.ndarray,
    p_o: np.ndarray,
    gold_g: np.ndarray,
    gap_i: np.ndarray,
    mode: str = "max",
) -> Tuple[float, float, Dict[str, Any]]:
    """Fast decoupled+joint grid oracle threshold search for SM13 strict micro-F1 (diagnostic only!)."""
    conf_t = np.maximum(p_s, p_o) if mode == "max" else (p_s + p_o)
    t_grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 46), np.quantile(conf_t[np.isfinite(conf_t)], np.linspace(0.5, 0.99, 30))]))
    g_grid = np.unique(np.concatenate([np.linspace(0.20, 0.98, 25), np.quantile(gap_i[np.isfinite(gap_i)], np.linspace(0.90, 0.999, 15))]))
    # First find best tau_g roughly, then joint refine
    best_f1, best_tt, best_tg, best_res = -1.0, 0.5, 0.8, None
    # Coordinate ascent x2
    tg = 0.75
    for _ in range(2):
        pg = np.where(gap_i >= tg, "I", "N")
        for tt in t_grid:
            pt, _ = apply_sm13_tau(p_s, p_o, gap_i, float(tt), tg, mode=mode)
            res = score_decisions(gold_t, pt, gold_g, pg)
            if res["strict"]["f1"] > best_f1:
                best_f1, best_tt, best_tg, best_res = res["strict"]["f1"], float(tt), float(tg), res
        pt, _ = apply_sm13_tau(p_s, p_o, gap_i, best_tt, tg, mode=mode)
        for tg_c in g_grid:
            pg = np.where(gap_i >= float(tg_c), "I", "N")
            res = score_decisions(gold_t, pt, gold_g, pg)
            if res["strict"]["f1"] > best_f1:
                best_f1, best_tg, best_res = res["strict"]["f1"], float(tg_c), res
        tg = best_tg
    return best_tt, best_tg, best_res


def paired_speaker_bootstrap(
    speakers_t: np.ndarray,
    gold_t: np.ndarray,
    pred_a_t: np.ndarray,
    pred_b_t: np.ndarray,
    speakers_g: np.ndarray,
    gold_g: np.ndarray,
    pred_a_g: np.ndarray,
    pred_b_g: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261119,
) -> Dict[str, Any]:
    """Paired speaker-cluster bootstrap for delta strict micro-F1 (A - B)."""
    spks = sorted(set(speakers_t.tolist()))
    # Precompute per-speaker strict tp, fp, fn for A and B
    stats_a, stats_b = [], []
    wins, losses, ties = 0, 0, 0
    for spk in spks:
        mt = speakers_t == spk
        mg = speakers_g == spk
        ra = score_decisions(gold_t[mt], pred_a_t[mt], gold_g[mg], pred_a_g[mg])["strict"]
        rb = score_decisions(gold_t[mt], pred_b_t[mt], gold_g[mg], pred_b_g[mg])["strict"]
        stats_a.append((ra["tp"], ra["fp"], ra["fn"]))
        stats_b.append((rb["tp"], rb["fp"], rb["fn"]))
        if ra["f1"] > rb["f1"] + 1e-12:
            wins += 1
        elif rb["f1"] > ra["f1"] + 1e-12:
            losses += 1
        else:
            ties += 1
    sa = np.array(stats_a, dtype=np.float64)
    sb = np.array(stats_b, dtype=np.float64)

    def micro_f1(arr: np.ndarray) -> float:
        tp, fp, fn = arr.sum(axis=0)
        return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    obs = (micro_f1(sa) - micro_f1(sb)) * 100.0
    rng = np.random.default_rng(seed)
    k = len(spks)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        deltas[b] = (micro_f1(sa[idx]) - micro_f1(sb[idx])) * 100.0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_pp": round(obs, 2),
        "ci95": [round(float(lo), 2), round(float(hi), 2)],
        "speaker_wlt": f"{wins}/{losses}/{ties}",
        "n_speakers": k,
    }


def load_dev18_full() -> Dict[str, Any]:
    """Load full DEV-18 sealed predictions + gold + continuous scores for R0 and SM13."""
    df_pred = pq.read_table(DEV18_RUN / "eval_predictions.parquet").to_pandas()
    # Gold map
    gold_map = {}
    for line in DEV18_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        uid = r["uid"]
        for i, g in enumerate(r["gold_targets"]):
            gold_map[(uid, "target", i)] = g
        for i, g in enumerate(r["gold_gaps"]):
            gold_map[(uid, "gap", i)] = g
    df_pred["gold_label"] = [
        gold_map[(u, d, int(idx))] for u, d, idx in zip(df_pred["uid"], df_pred["domain"], df_pred["index"])
    ]

    # Load continuous scores from koel_reference_and_cal.npz and sm13_raw.jsonl
    npz = np.load(DEV18_RUN / "koel_reference_and_cal.npz", allow_pickle=True)
    df_t = df_pred[df_pred["domain"] == "target"].copy().reset_index(drop=True)
    df_g = df_pred[df_pred["domain"] == "gap"].copy().reset_index(drop=True)

    # Verify exact row alignment between npz and df_t
    assert np.all(npz["t_vanilla"].astype(str) == df_t["vanilla"].astype(str).to_numpy())
    assert np.all(npz["t_r0_locked"].astype(str) == df_t["r0_locked"].astype(str).to_numpy())
    r0_eval = npz["r0_eval"]  # (54134, 4) -> [p_c, p_s, p_o, score]
    df_t["r0_p_s"] = r0_eval[:, 1].astype(np.float64)
    df_t["r0_p_o"] = r0_eval[:, 2].astype(np.float64)
    df_t["r0_score"] = r0_eval[:, 3].astype(np.float64)

    sm13_t_map = {}
    sm13_g_map = {}
    with (DEV18_RUN / "sm13_raw.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r["uid"]
            probs = r["target_probs_cso"]
            for i, pr in enumerate(probs):
                sm13_t_map[(uid, i)] = (float(pr[0]), float(pr[1]), float(pr[2]))
            for i, gp in enumerate(r["gap_prob_i"]):
                sm13_g_map[(uid, i)] = float(gp)

    sm_vals = [sm13_t_map[(u, int(i))] for u, i in zip(df_t["uid"], df_t["index"])]
    df_t["sm13_fused_p_c"] = [v[0] for v in sm_vals]
    df_t["sm13_fused_p_s"] = [v[1] for v in sm_vals]
    df_t["sm13_fused_p_o"] = [v[2] for v in sm_vals]

    df_g["sm13_gap_prob_i"] = [sm13_g_map[(u, int(i))] for u, i in zip(df_g["uid"], df_g["index"])]

    # Thresholds
    th = json.loads((DEV18_RUN / "thresholds.json").read_text(encoding="utf-8"))
    th_lk = json.loads(
        (REPO_ROOT / "research/KOEL_BASE_MDD_HEAD_DEV18_20261119_v1/thresholds.json").read_text(encoding="utf-8")
    )
    return {"df_t": df_t, "df_g": df_g, "thresholds": th, "thresholds_locked": th_lk}


def load_saa27_full() -> Dict[str, Any]:
    df_all = pq.read_table(SAA27_RUN / "saa27_all_systems_predictions.parquet").to_pandas()
    gold_map = {}
    for line in SAA27_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        uid = r["uid"]
        for i, g in enumerate(r["gold_targets"]):
            gold_map[(uid, "target", i)] = g
        for i, g in enumerate(r["gold_gaps"]):
            gold_map[(uid, "gap", i)] = g
    df_all["gold_label"] = [
        gold_map[(u, d, int(idx))] for u, d, idx in zip(df_all["uid"], df_all["domain"], df_all["index"])
    ]
    th = json.loads((SAA27_RUN / "thresholds.json").read_text(encoding="utf-8"))
    return {"df_all": df_all, "thresholds": th}


def analyze_new_cohort(
    cname: str,
    tau_r0_locked: float,
    tau_r0_cal: float,
    tau_sm13_cal_t: float,
    tau_sm13_cal_g: float,
    use_alt_gold: bool = False,
) -> Dict[str, Any]:
    df = pq.read_table(OUT_DIR / f"{cname}_slots.parquet").to_pandas()
    gold_col = "gold_label_alt" if use_alt_gold else "gold_label"
    df_t = df[df["domain"] == "target"].reset_index(drop=True)
    df_g = df[df["domain"] == "gap"].reset_index(drop=True)

    gt = df_t[gold_col].to_numpy()
    gg = df_g[gold_col].to_numpy()
    sp_t = df_t["speaker"].to_numpy()
    sp_g = df_g["speaker"].to_numpy()

    n_utts = int(df_t["uid"].nunique())
    n_spks = int(df_t["speaker"].nunique())

    # Vanilla
    v_t = df_t["vanilla_call"].to_numpy()
    v_g = df_g["vanilla_call"].to_numpy()
    res_vanilla = score_decisions(gt, v_t, gg, v_g)

    # R0 locked & transferred Cal-F1
    r0_ps = df_t["r0_p_s"].to_numpy()
    r0_po = df_t["r0_p_o"].to_numpy()
    r0_sc = df_t["r0_score"].to_numpy()
    r0_t_locked = apply_r0_tau(r0_ps, r0_po, r0_sc, tau_r0_locked)
    r0_t_cal = apply_r0_tau(r0_ps, r0_po, r0_sc, tau_r0_cal)
    res_r0_locked = score_decisions(gt, r0_t_locked, gg, v_g)
    res_r0_cal = score_decisions(gt, r0_t_cal, gg, v_g)

    # SM13 fused locked & transferred Cal-F1
    sm_ps = df_t["sm13_fused_p_s"].to_numpy()
    sm_po = df_t["sm13_fused_p_o"].to_numpy()
    sm_gi = df_g["sm13_gap_prob_i"].to_numpy()
    sm_t_locked, sm_g_locked = apply_sm13_tau(sm_ps, sm_po, sm_gi, SM13_LOCKED_TAU_T, SM13_LOCKED_TAU_G)
    sm_t_cal, sm_g_cal = apply_sm13_tau(sm_ps, sm_po, sm_gi, tau_sm13_cal_t, tau_sm13_cal_g)
    res_sm13_locked = score_decisions(gt, sm_t_locked, gg, sm_g_locked)
    res_sm13_cal = score_decisions(gt, sm_t_cal, gg, sm_g_cal)

    # Threshold-free PR-AUC & ROC-AUC (target binary & gap binary)
    tv = gt != "unknown"
    gv = gg != "unknown"
    y_t = ((gt[tv] == "S") | (gt[tv] == "O")).astype(int)
    y_g = (gg[gv] == "I").astype(int)

    r0_ap, r0_auc = pr_roc(y_t, r0_sc[tv])
    sm_fused_max_ap, sm_fused_max_auc = pr_roc(y_t, np.maximum(sm_ps[tv], sm_po[tv]))
    sm_fused_sum_ap, sm_fused_sum_auc = pr_roc(y_t, (sm_ps + sm_po)[tv])
    sm_neural_ps = df_t["sm13_neural_p_s"].to_numpy()
    sm_neural_po = df_t["sm13_neural_p_o"].to_numpy()
    sm_neural_sum_ap, sm_neural_sum_auc = pr_roc(y_t, (sm_neural_ps + sm_neural_po)[tv])
    sm_e4_ps = df_t["sm13_e4_p_s"].to_numpy()
    sm_e4_po = df_t["sm13_e4_p_o"].to_numpy()
    sm_e4_sum_ap, sm_e4_sum_auc = pr_roc(y_t, (sm_e4_ps + sm_e4_po)[tv])
    sm_gap_ap, sm_gap_auc = pr_roc(y_g, sm_gi[gv])

    # Oracle thresholds (diagnostic upper bound)
    if y_t.sum() > 0:
        or_r0_tau, res_r0_oracle = oracle_best_f1_1d(gt, r0_ps, r0_po, r0_sc, gg, v_g)
        or_sm_tt, or_sm_tg, res_sm13_oracle = oracle_best_f1_sm13(gt, sm_ps, sm_po, gg, sm_gi, mode="max")
        or_smn_tt, or_smn_tg, res_sm13_neural_oracle = oracle_best_f1_sm13(
            gt, sm_neural_ps, sm_neural_po, gg, sm_gi, mode="sum"
        )
        or_sme_tt, or_sme_tg, res_sm13_e4_oracle = oracle_best_f1_sm13(
            gt, sm_e4_ps, sm_e4_po, gg, sm_gi, mode="sum"
        )
        boot_r0_vs_sm13 = paired_speaker_bootstrap(sp_t, gt, r0_t_cal, sm_t_cal, sp_g, gg, v_g, sm_g_cal)
        boot_r0_vs_van = paired_speaker_bootstrap(sp_t, gt, r0_t_cal, v_t, sp_g, gg, v_g, v_g)
        boot_sm13_vs_van = paired_speaker_bootstrap(sp_t, gt, sm_t_cal, v_t, sp_g, gg, sm_g_cal, v_g)
    else:
        or_r0_tau, res_r0_oracle = None, None
        or_sm_tt, or_sm_tg, res_sm13_oracle = None, None, None
        or_smn_tt, or_smn_tg, res_sm13_neural_oracle = None, None, None
        or_sme_tt, or_sme_tg, res_sm13_e4_oracle = None, None, None
        boot_r0_vs_sm13 = boot_r0_vs_van = boot_sm13_vs_van = None

    # Latency & unmapped token stats from utterance_meta.jsonl
    meta_path = OUT_DIR / f"{cname}_utterance_meta.jsonl"
    meta_stats = {}
    if meta_path.exists():
        metas = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines()]
        koel_ms = np.array([m["koel_ms"] for m in metas], dtype=np.float64)
        r0_ms = np.array([m["koel_ms"] + m["r0_head_ms"] for m in metas], dtype=np.float64)
        sm_ms = np.array([m["sm13_ms"] for m in metas], dtype=np.float64)
        durs = np.array([m["duration_s"] for m in metas], dtype=np.float64)
        unmapped_counter = Counter(tok for m in metas for tok in m["unmapped_tokens"])
        total_toks = sum(len(m["native_tokens"]) for m in metas)
        total_unmapped = sum(m["n_unmapped"] for m in metas)
        meta_stats = {
            "total_audio_s": round(float(durs.sum()), 2),
            "median_dur_s": round(float(np.median(durs)), 2),
            "vit_fallback_utts": int(sum(1 for m in metas if m["vit_uniform_fallback"])),
            "total_emitted_koel_tokens": int(total_toks),
            "total_unmapped_koel_tokens": int(total_unmapped),
            "unmapped_rate_pct": round(100.0 * total_unmapped / max(1, total_toks), 3),
            "utts_with_unmapped": int(sum(1 for m in metas if m["n_unmapped"] > 0)),
            "top_unmapped_tokens": unmapped_counter.most_common(15),
            "latency_ms": {
                "vanilla_p50": round(float(np.median(koel_ms)), 1),
                "vanilla_p95": round(float(np.percentile(koel_ms, 95)), 1),
                "r0_p50": round(float(np.median(r0_ms)), 1),
                "r0_p95": round(float(np.percentile(r0_ms, 95)), 1),
                "sm13_p50": round(float(np.median(sm_ms)), 1),
                "sm13_p95": round(float(np.percentile(sm_ms, 95)), 1),
                "sm13_vs_r0_p50_ratio": round(float(np.median(sm_ms) / max(1e-6, np.median(r0_ms))), 2),
            },
        }

    # Detailed R0 & Vanilla & SM13 diagnostic slices
    # 1. Vanilla '?' abstention audit
    q_mask = (v_t == "?") & tv
    q_gold_counts = Counter(gt[q_mask].tolist())
    # 2. R0 conditional on Vanilla greedy call (how much is rescue vs suppression?)
    r0_by_van = {}
    for vc in ["C", "S", "O", "?"]:
        m = (v_t == vc) & tv
        if m.sum() == 0:
            continue
        g_err = (gt[m] == "S") | (gt[m] == "O")
        r0_err = (r0_t_cal[m] == "S") | (r0_t_cal[m] == "O")
        r0_by_van[vc] = {
            "n": int(m.sum()),
            "gold_err": int(g_err.sum()),
            "r0_cal_alert": int(r0_err.sum()),
            "r0_cal_tp": int((g_err & r0_err).sum()),
            "r0_cal_fp": int(((~g_err) & r0_err).sum()),
            "r0_mean_score_on_err": round(float(r0_sc[m][g_err].mean()), 4) if g_err.sum() > 0 else None,
            "r0_mean_score_on_C": round(float(r0_sc[m][~g_err].mean()), 4) if (~g_err).sum() > 0 else None,
        }

    # 3. R0 score distribution shift (mean on C, S, O) & calibration
    score_dist = {}
    for lab in ["C", "S", "O"]:
        ml = gt == lab
        if ml.sum() > 0:
            score_dist[lab] = {
                "n": int(ml.sum()),
                "r0_mean": round(float(r0_sc[ml].mean()), 4),
                "r0_p90": round(float(np.percentile(r0_sc[ml], 90)), 4),
                "sm13_fused_max_mean": round(float(np.maximum(sm_ps[ml], sm_po[ml]).mean()), 4),
                "sm13_fused_ps_max": round(float(sm_ps[ml].max()), 4),
                "sm13_fused_ps_mean": round(float(sm_ps[ml].mean()), 4),
                "sm13_fused_po_mean": round(float(sm_po[ml].mean()), 4),
                "sm13_neural_ps_mean": round(float(sm_neural_ps[ml].mean()), 4),
                "sm13_neural_po_mean": round(float(sm_neural_po[ml].mean()), 4),
                "sm13_e4_ps_mean": round(float(sm_e4_ps[ml].mean()), 4),
                "sm13_e4_ps_max": round(float(sm_e4_ps[ml].max()), 4),
                "sm13_e4_po_mean": round(float(sm_e4_po[ml].mean()), 4),
            }

    # 4. Per-phone breakdown (top error-causing phones for R0 Cal-F1 & Vanilla)
    phone_stats = {}
    for ph, grp in df_t[tv].groupby("canonical_phone"):
        idx_p = grp.index.to_numpy()
        gp = gt[idx_p]
        g_err = (gp == "S") | (gp == "O")
        # Vanilla
        vp = np.isin(v_t[idx_p], ["S", "O"])
        # R0 cal
        rp = np.isin(r0_t_cal[idx_p], ["S", "O"])
        # SM13 cal
        sp_p = np.isin(sm_t_cal[idx_p], ["S", "O"])
        phone_stats[ph] = {
            "n": int(len(gp)),
            "gold_err": int(g_err.sum()),
            "van_tp": int((g_err & vp).sum()), "van_fp": int(((~g_err) & vp).sum()), "van_fn": int((g_err & ~vp).sum()),
            "r0_tp": int((g_err & rp).sum()), "r0_fp": int(((~g_err) & rp).sum()), "r0_fn": int((g_err & ~rp).sum()),
            "sm13_tp": int((g_err & sp_p).sum()), "sm13_fp": int(((~g_err) & sp_p).sum()), "sm13_fn": int((g_err & ~sp_p).sum()),
        }

    # 5. Alignment tier breakdown for R0
    tier_stats = {}
    for tier_val in sorted(df_t["align_tier"].unique()):
        m = (df_t["align_tier"].to_numpy() == tier_val) & tv
        if m.sum() == 0:
            continue
        g_err = (gt[m] == "S") | (gt[m] == "O")
        rp = np.isin(r0_t_cal[m], ["S", "O"])
        ap_t, auc_t = pr_roc(g_err.astype(int), r0_sc[m])
        tier_stats[int(tier_val)] = {
            "n": int(m.sum()),
            "gold_err": int(g_err.sum()),
            "r0_tp": int((g_err & rp).sum()),
            "r0_fp": int(((~g_err) & rp).sum()),
            "r0_fn": int((g_err & ~rp).sum()),
            "r0_pr_auc": round(ap_t, 4) if ap_t is not None else None,
        }

    return {
        "cohort": cname,
        "use_alt_gold": use_alt_gold,
        "n_utts": n_utts,
        "n_speakers": n_spks,
        "n_target_slots": int(len(df_t)),
        "n_gap_slots": int(len(df_g)),
        "vanilla": res_vanilla,
        "r0_locked": res_r0_locked,
        "r0_cal_transfer": res_r0_cal,
        "r0_oracle": {"tau_t": or_r0_tau, "metrics": res_r0_oracle},
        "sm13_locked": res_sm13_locked,
        "sm13_cal_transfer": res_sm13_cal,
        "sm13_oracle": {"tau_t": or_sm_tt, "tau_g": or_sm_tg, "metrics": res_sm13_oracle},
        "sm13_neural_only_oracle": {"tau_t": or_smn_tt, "tau_g": or_smn_tg, "metrics": res_sm13_neural_oracle},
        "sm13_e4_only_oracle": {"tau_t": or_sme_tt, "tau_g": or_sme_tg, "metrics": res_sm13_e4_oracle},
        "threshold_free": {
            "r0_target_pr_auc": r0_ap, "r0_target_roc_auc": r0_auc,
            "sm13_fused_max_target_pr_auc": sm_fused_max_ap, "sm13_fused_max_target_roc_auc": sm_fused_max_auc,
            "sm13_fused_sum_target_pr_auc": sm_fused_sum_ap, "sm13_fused_sum_target_roc_auc": sm_fused_sum_auc,
            "sm13_neural_target_pr_auc": sm_neural_sum_ap, "sm13_neural_target_roc_auc": sm_neural_sum_auc,
            "sm13_e4_target_pr_auc": sm_e4_sum_ap, "sm13_e4_target_roc_auc": sm_e4_sum_auc,
            "sm13_gap_pr_auc": sm_gap_ap, "sm13_gap_roc_auc": sm_gap_auc,
        },
        "bootstrap_cal_transfer": {
            "r0_minus_sm13": boot_r0_vs_sm13,
            "r0_minus_vanilla": boot_r0_vs_van,
            "sm13_minus_vanilla": boot_sm13_vs_van,
        },
        "meta_stats": meta_stats,
        "vanilla_q_abstention": {"n_q_valid": int(q_mask.sum()), "gold_counts": dict(q_gold_counts)},
        "r0_by_vanilla_call": r0_by_van,
        "score_dist": score_dist,
        "phone_stats": phone_stats,
        "tier_stats": tier_stats,
    }


def analyze_sm13_decomp_only(cname: str) -> Dict[str, Any]:
    df = pq.read_table(OUT_DIR / f"{cname}_slots.parquet").to_pandas()
    df_t = df[df["domain"] == "target"].reset_index(drop=True)
    df_g = df[df["domain"] == "gap"].reset_index(drop=True)
    gt = df_t["gold_label"].to_numpy()
    gg = df_g["gold_label"].to_numpy()
    y_t = ((gt == "S") | (gt == "O")).astype(int)
    y_g = (gg == "I").astype(int)

    f_ps, f_po = df_t["sm13_fused_p_s"].to_numpy(), df_t["sm13_fused_p_o"].to_numpy()
    n_ps, n_po = df_t["sm13_neural_p_s"].to_numpy(), df_t["sm13_neural_p_o"].to_numpy()
    e_ps, e_po = df_t["sm13_e4_p_s"].to_numpy(), df_t["sm13_e4_p_o"].to_numpy()
    gi = df_g["sm13_gap_prob_i"].to_numpy()

    out = {
        "cohort": cname,
        "n_utts": int(df_t["uid"].nunique()),
        "n_targets": int(len(df_t)),
        "fused_max_pr_auc": pr_roc(y_t, np.maximum(f_ps, f_po))[0],
        "fused_sum_pr_auc": pr_roc(y_t, f_ps + f_po)[0],
        "neural_sum_pr_auc": pr_roc(y_t, n_ps + n_po)[0],
        "e4_sum_pr_auc": pr_roc(y_t, e_ps + e_po)[0],
        "gap_pr_auc": pr_roc(y_g, gi)[0],
        "fused_oracle": oracle_best_f1_sm13(gt, f_ps, f_po, gg, gi, mode="max")[2],
        "neural_oracle": oracle_best_f1_sm13(gt, n_ps, n_po, gg, gi, mode="sum")[2],
        "e4_oracle": oracle_best_f1_sm13(gt, e_ps, e_po, gg, gi, mode="sum")[2],
        "max_e4_p_s": float(e_ps.max()),
        "mean_e4_p_s_on_S": float(e_ps[gt == "S"].mean()),
        "mean_e4_p_o_on_S": float(e_po[gt == "S"].mean()),
        "max_fused_p_s": float(f_ps.max()),
        "pct_S_where_e4_po_gt_ps": round(100.0 * float(np.mean(e_po[gt == "S"] > e_ps[gt == "S"])), 2),
        "pct_S_where_neural_ps_gt_po": round(100.0 * float(np.mean(n_ps[gt == "S"] > n_po[gt == "S"])), 2),
        "pct_S_where_fused_po_gt_ps": round(100.0 * float(np.mean(f_po[gt == "S"] > f_ps[gt == "S"])), 2),
    }

    # For dev18_sm13_decomp, also split by SM13 speaker training membership!
    if cname == "dev18_sm13_decomp":
        by_group = {}
        for gname, sset in [
            ("SM13_TRAIN_6spks", SM13_TRAIN_SPKS),
            ("SM13_DEV_SEARCH_4spks", SM13_DEV_SEARCH_SPKS),
            ("SM13_DEV_CAL_2spks", SM13_DEV_CAL_SPKS),
            ("SM13_TEST_SEALED_4spks", SM13_TEST_SEALED_SPKS),
            ("SM13_UNSEEN_2spks", SM13_UNSEEN_L2A_SPKS),
        ]:
            mt = df_t["speaker"].isin(sset).to_numpy()
            mg = df_g["speaker"].isin(sset).to_numpy()
            if mt.sum() == 0:
                continue
            by_group[gname] = {
                "n_targets": int(mt.sum()),
                "neural_pr_auc": round(pr_roc(y_t[mt], (n_ps + n_po)[mt])[0], 4),
                "fused_pr_auc": round(pr_roc(y_t[mt], np.maximum(f_ps[mt], f_po[mt]))[0], 4),
                "e4_pr_auc": round(pr_roc(y_t[mt], (e_ps + e_po)[mt])[0], 4),
                "neural_oracle_f1": round(oracle_best_f1_sm13(gt[mt], n_ps[mt], n_po[mt], gg[mg], gi[mg], mode="sum")[2]["strict"]["f1"] * 100, 2),
                "fused_oracle_f1": round(oracle_best_f1_sm13(gt[mt], f_ps[mt], f_po[mt], gg[mg], gi[mg], mode="max")[2]["strict"]["f1"] * 100, 2),
            }
        out["by_sm13_membership"] = by_group
    return out


def inspect_sm13_weights() -> Dict[str, Any]:
    """Inspect learned 13-layer scalar-mixing weights & scale in SM13 checkpoints."""
    t_sd = torch.load(SM13_CKPT_DIR / "target_sm13_seed101_fulltrain.pt", map_location="cpu", weights_only=True)
    g_sd = torch.load(SM13_CKPT_DIR / "gap_sm13_seed101_fulltrain.pt", map_location="cpu", weights_only=True)
    tw = torch.softmax(t_sd["layer_logits"].float(), dim=0).numpy().tolist()
    gw = torch.softmax(g_sd["layer_logits"].float(), dim=0).numpy().tolist()
    return {
        "target_scale": round(float(t_sd["scale"].item()), 4),
        "target_layer_weights_0_to_12": [round(x, 4) for x in tw],
        "gap_scale": round(float(g_sd["scale"].item()), 4),
        "gap_layer_weights_0_to_12": [round(x, 4) for x in gw],
    }


def main() -> None:
    # 1. Load DEV-18 full + compute exact transferred thresholds from DEV-18 Calibrate folds
    dev18 = load_dev18_full()
    th_dev18 = dev18["thresholds"]
    th_lk = dev18["thresholds_locked"]
    # Extract mean Cal-F1 thresholds across the 6 DEV-18 folds
    r0_cal_taus = [th_dev18["folds"][str(i)]["r0_cal_f1"]["target_threshold"] for i in range(6)]
    sm13_cal_tt = [th_dev18["folds"][str(i)]["sm13_cal_f1"]["target_threshold"] for i in range(6)]
    sm13_cal_tg = [th_dev18["folds"][str(i)]["sm13_cal_f1"]["gap_threshold"] for i in range(6)]
    r0_locked_taus = [f["locked_tau"] for f in th_lk["branches"]["gold_only"]["folds"]]

    tau_r0_locked = float(np.mean(r0_locked_taus))
    tau_r0_cal = float(np.mean(r0_cal_taus))
    tau_sm13_cal_t = float(np.mean(sm13_cal_tt))
    tau_sm13_cal_g = float(np.mean(sm13_cal_tg))

    print(f"Transferred DEV-18 thresholds: R0_locked={tau_r0_locked:.4f}, R0_CalF1={tau_r0_cal:.4f}, "
          f"SM13_CalF1=(t={tau_sm13_cal_t:.4f}, g={tau_sm13_cal_g:.4f})")

    # 2. Analyze DEV-18 full (existing sealed + threshold-free PR-AUC + SM13 membership split)
    dt, dg = dev18["df_t"], dev18["df_g"]
    gt_d, gg_d = dt["gold_label"].to_numpy(), dg["gold_label"].to_numpy()
    yt_d = ((gt_d == "S") | (gt_d == "O")).astype(int)
    yg_d = (gg_d == "I").astype(int)

    dev18_summary = {
        "cohort": "dev18",
        "n_utts": int(dt["uid"].nunique()),
        "n_speakers": int(dt["speaker"].nunique()),
        "n_target_slots": int(len(dt)),
        "n_gap_slots": int(len(dg)),
        "vanilla": score_decisions(gt_d, dt["vanilla"].to_numpy(), gg_d, dg["vanilla"].to_numpy()),
        "r0_locked": score_decisions(gt_d, dt["r0_locked"].to_numpy(), gg_d, dg["r0_locked"].to_numpy()),
        "r0_cal_f1_oof": score_decisions(gt_d, dt["r0_cal_f1"].to_numpy(), gg_d, dg["r0_cal_f1"].to_numpy()),
        "sm13_locked": score_decisions(gt_d, dt["sm13_locked"].to_numpy(), gg_d, dg["sm13_locked"].to_numpy()),
        "sm13_cal_f1_oof": score_decisions(gt_d, dt["sm13_cal_f1"].to_numpy(), gg_d, dg["sm13_cal_f1"].to_numpy()),
        "threshold_free": {
            "r0_target_pr_auc": pr_roc(yt_d, dt["r0_score"].to_numpy())[0],
            "r0_target_roc_auc": pr_roc(yt_d, dt["r0_score"].to_numpy())[1],
            "sm13_fused_max_target_pr_auc": pr_roc(yt_d, np.maximum(dt["sm13_fused_p_s"], dt["sm13_fused_p_o"]).to_numpy())[0],
            "sm13_fused_max_target_roc_auc": pr_roc(yt_d, np.maximum(dt["sm13_fused_p_s"], dt["sm13_fused_p_o"]).to_numpy())[1],
            "sm13_fused_sum_target_pr_auc": pr_roc(yt_d, (dt["sm13_fused_p_s"] + dt["sm13_fused_p_o"]).to_numpy())[0],
            "sm13_gap_pr_auc": pr_roc(yg_d, dg["sm13_gap_prob_i"].to_numpy())[0],
            "sm13_gap_roc_auc": pr_roc(yg_d, dg["sm13_gap_prob_i"].to_numpy())[1],
        },
    }
    or_r0_d_tau, or_r0_d_res = oracle_best_f1_1d(
        gt_d, dt["r0_p_s"].to_numpy(), dt["r0_p_o"].to_numpy(), dt["r0_score"].to_numpy(), gg_d, dg["vanilla"].to_numpy()
    )
    or_sm_d_tt, or_sm_d_tg, or_sm_d_res = oracle_best_f1_sm13(
        gt_d, dt["sm13_fused_p_s"].to_numpy(), dt["sm13_fused_p_o"].to_numpy(), gg_d, dg["sm13_gap_prob_i"].to_numpy(), mode="max"
    )
    dev18_summary["r0_oracle"] = {"tau_t": or_r0_d_tau, "metrics": or_r0_d_res}
    dev18_summary["sm13_oracle"] = {"tau_t": or_sm_d_tt, "tau_g": or_sm_d_tg, "metrics": or_sm_d_res}

    # DEV-18 split by SM13 membership (proving SM13 fails EVEN on its own TRAIN speakers!)
    dev18_by_sm13_mem = {}
    for gname, sset in [
        ("SM13_TRAIN_6spks", SM13_TRAIN_SPKS),
        ("SM13_DEV_SEARCH_4spks", SM13_DEV_SEARCH_SPKS),
        ("SM13_DEV_CAL_2spks", SM13_DEV_CAL_SPKS),
        ("SM13_TEST_SEALED_4spks", SM13_TEST_SEALED_SPKS),
        ("SM13_UNSEEN_2spks", SM13_UNSEEN_L2A_SPKS),
    ]:
        mt = dt["speaker"].isin(sset).to_numpy()
        mg = dg["speaker"].isin(sset).to_numpy()
        dev18_by_sm13_mem[gname] = {
            "n_utts": int(dt[mt]["uid"].nunique()),
            "vanilla_f1": round(score_decisions(gt_d[mt], dt["vanilla"].to_numpy()[mt], gg_d[mg], dg["vanilla"].to_numpy()[mg])["strict"]["f1"] * 100, 2),
            "r0_cal_f1": round(score_decisions(gt_d[mt], dt["r0_cal_f1"].to_numpy()[mt], gg_d[mg], dg["r0_cal_f1"].to_numpy()[mg])["strict"]["f1"] * 100, 2),
            "sm13_cal_f1": round(score_decisions(gt_d[mt], dt["sm13_cal_f1"].to_numpy()[mt], gg_d[mg], dg["sm13_cal_f1"].to_numpy()[mg])["strict"]["f1"] * 100, 2),
            "r0_pr_auc": round(pr_roc(yt_d[mt], dt["r0_score"].to_numpy()[mt])[0], 4),
            "sm13_pr_auc": round(pr_roc(yt_d[mt], np.maximum(dt["sm13_fused_p_s"], dt["sm13_fused_p_o"]).to_numpy()[mt])[0], 4),
        }
    dev18_summary["by_sm13_membership"] = dev18_by_sm13_mem

    # DEV-18 per-phone errors for R0 Cal-F1
    dev18_phone_stats = {}
    for ph, grp in dt.groupby("canonical_phone"):
        idx_p = grp.index.to_numpy()
        gp = gt_d[idx_p]
        g_err = (gp == "S") | (gp == "O")
        rp = np.isin(dt["r0_cal_f1"].to_numpy()[idx_p], ["S", "O"])
        vp = np.isin(dt["vanilla"].to_numpy()[idx_p], ["S", "O"])
        dev18_phone_stats[ph] = {
            "n": int(len(gp)), "gold_err": int(g_err.sum()),
            "van_tp": int((g_err & vp).sum()), "van_fp": int(((~g_err) & vp).sum()), "van_fn": int((g_err & ~vp).sum()),
            "r0_tp": int((g_err & rp).sum()), "r0_fp": int(((~g_err) & rp).sum()), "r0_fn": int((g_err & ~rp).sum()),
        }
    dev18_summary["phone_stats"] = dev18_phone_stats

    # 3. Analyze SAA-27 full (existing sealed + threshold-free + transferred DEV-18 Cal-F1 check!)
    saa = load_saa27_full()
    sa = saa["df_all"]
    s_van = sa[sa["variant"] == "vanilla"]
    s_r0 = sa[sa["variant"] == "r0_base_cal_f1"]
    s_sm = sa[sa["variant"] == "sm13_cal_f1"]
    s_r0_lk = sa[sa["variant"] == "r0_base_locked"]
    s_sm_lk = sa[sa["variant"] == "sm13_locked"]

    st_van, sg_van = s_van[s_van["domain"] == "target"], s_van[s_van["domain"] == "gap"]
    st_r0, sg_r0 = s_r0[s_r0["domain"] == "target"], s_r0[s_r0["domain"] == "gap"]
    st_sm, sg_sm = s_sm[s_sm["domain"] == "target"], s_sm[s_sm["domain"] == "gap"]
    gt_s, gg_s = st_van["gold_label"].to_numpy(), sg_van["gold_label"].to_numpy()
    yt_s = ((gt_s == "S") | (gt_s == "O")).astype(int)
    yg_s = (gg_s == "I").astype(int)

    # Also apply transferred DEV-18 Cal-F1 thresholds on SAA-27!
    r0_saa_devcal_t = apply_r0_tau(st_r0["p_s"].to_numpy(), st_r0["p_o"].to_numpy(), st_r0["score"].to_numpy(), tau_r0_cal)
    sm_saa_devcal_t, sm_saa_devcal_g = apply_sm13_tau(
        st_sm["p_s"].to_numpy(), st_sm["p_o"].to_numpy(), sg_sm["score"].to_numpy(), tau_sm13_cal_t, tau_sm13_cal_g
    )
    or_r0_s_tau, or_r0_s_res = oracle_best_f1_1d(
        gt_s, st_r0["p_s"].to_numpy(), st_r0["p_o"].to_numpy(), st_r0["score"].to_numpy(), gg_s, sg_van["pred_call"].to_numpy()
    )
    or_sm_s_tt, or_sm_s_tg, or_sm_s_res = oracle_best_f1_sm13(
        gt_s, st_sm["p_s"].to_numpy(), st_sm["p_o"].to_numpy(), gg_s, sg_sm["score"].to_numpy(), mode="max"
    )
    saa27_summary = {
        "cohort": "saa27",
        "n_utts": int(st_van["uid"].nunique()),
        "n_speakers": int(st_van["speaker"].nunique()),
        "n_target_slots": int(len(st_van)),
        "n_gap_slots": int(len(sg_van)),
        "vanilla": score_decisions(gt_s, st_van["pred_call"].to_numpy(), gg_s, sg_van["pred_call"].to_numpy()),
        "r0_locked": score_decisions(
            gt_s, s_r0_lk[s_r0_lk["domain"] == "target"]["pred_call"].to_numpy(),
            gg_s, s_r0_lk[s_r0_lk["domain"] == "gap"]["pred_call"].to_numpy()
        ),
        "r0_cal_saa18": score_decisions(gt_s, st_r0["pred_call"].to_numpy(), gg_s, sg_r0["pred_call"].to_numpy()),
        "r0_cal_transfer_dev18": score_decisions(gt_s, r0_saa_devcal_t, gg_s, sg_van["pred_call"].to_numpy()),
        "r0_oracle": {"tau_t": or_r0_s_tau, "metrics": or_r0_s_res},
        "sm13_locked": score_decisions(
            gt_s, s_sm_lk[s_sm_lk["domain"] == "target"]["pred_call"].to_numpy(),
            gg_s, s_sm_lk[s_sm_lk["domain"] == "gap"]["pred_call"].to_numpy()
        ),
        "sm13_cal_saa18": score_decisions(gt_s, st_sm["pred_call"].to_numpy(), gg_s, sg_sm["pred_call"].to_numpy()),
        "sm13_cal_transfer_dev18": score_decisions(gt_s, sm_saa_devcal_t, gg_s, sm_saa_devcal_g),
        "sm13_oracle": {"tau_t": or_sm_s_tt, "tau_g": or_sm_s_tg, "metrics": or_sm_s_res},
        "threshold_free": {
            "r0_target_pr_auc": pr_roc(yt_s, st_r0["score"].to_numpy())[0],
            "r0_target_roc_auc": pr_roc(yt_s, st_r0["score"].to_numpy())[1],
            "sm13_fused_max_target_pr_auc": pr_roc(yt_s, np.maximum(st_sm["p_s"], st_sm["p_o"]).to_numpy())[0],
            "sm13_fused_max_target_roc_auc": pr_roc(yt_s, np.maximum(st_sm["p_s"], st_sm["p_o"]).to_numpy())[1],
            "sm13_gap_pr_auc": pr_roc(yg_s, sg_sm["score"].to_numpy())[0],
            "sm13_gap_roc_auc": pr_roc(yg_s, sg_sm["score"].to_numpy())[1],
        },
    }

    # 4. Analyze the 4 newly benchmarked cohorts
    held6_res = analyze_new_cohort("held6", tau_r0_locked, tau_r0_cal, tau_sm13_cal_t, tau_sm13_cal_g)
    spont_res = analyze_new_cohort("spontaneous", tau_r0_locked, tau_r0_cal, tau_sm13_cal_t, tau_sm13_cal_g)
    epadb_res = analyze_new_cohort("epadb_test", tau_r0_locked, tau_r0_cal, tau_sm13_cal_t, tau_sm13_cal_g, use_alt_gold=False)
    epadb_alt_res = analyze_new_cohort("epadb_test", tau_r0_locked, tau_r0_cal, tau_sm13_cal_t, tau_sm13_cal_g, use_alt_gold=True)
    timit_native = analyze_new_cohort("timit_core", tau_r0_locked, tau_r0_cal, tau_sm13_cal_t, tau_sm13_cal_g, use_alt_gold=False)
    timit_surf = analyze_new_cohort("timit_core", tau_r0_locked, tau_r0_cal, tau_sm13_cal_t, tau_sm13_cal_g, use_alt_gold=True)

    # 5. SM13 decomposition on SAA-27 & DEV-18-180 + SM13 learned weights
    sm13_saa_decomp = analyze_sm13_decomp_only("saa27_sm13_decomp")
    sm13_dev_decomp = analyze_sm13_decomp_only("dev18_sm13_decomp")
    sm13_weights = inspect_sm13_weights()

    # 6. TIMIT Native L1 False-Alarm detailed summary
    df_tim = pq.read_table(OUT_DIR / "timit_core_slots.parquet").to_pandas()
    dt_tim = df_tim[df_tim["domain"] == "target"]
    dg_tim = df_tim[df_tim["domain"] == "gap"]
    n_tim_t = len(dt_tim)
    n_tim_g = len(dg_tim)
    n_tim_u = int(dt_tim["uid"].nunique())

    def timit_fa(pt, pg):
        fa_s = int(np.sum(pt == "S"))
        fa_o = int(np.sum(pt == "O"))
        fa_i = int(np.sum(pg == "I"))
        tot = fa_s + fa_o + fa_i
        # Also check what % of target FAs happen where native surface39 != canonical39!
        surf = dt_tim["gold_label_alt"].to_numpy()
        p_err = np.isin(pt, ["S", "O"])
        surf_var = np.isin(surf, ["S", "O"])
        overlap_surf = int(np.sum(p_err & surf_var))
        return {
            "fa_total": tot, "fa_S": fa_s, "fa_O": fa_o, "fa_I": fa_i,
            "fa_per_utt": round(tot / n_tim_u, 2),
            "target_fa_per_1k": round(1000.0 * (fa_s + fa_o) / n_tim_t, 2),
            "pct_target_fa_on_native_surface_variation": round(100.0 * overlap_surf / max(1, fa_s + fa_o), 1),
        }

    v_tim_t = dt_tim["vanilla_call"].to_numpy()
    v_tim_g = dg_tim["vanilla_call"].to_numpy()
    r0_tim_lk = apply_r0_tau(dt_tim["r0_p_s"].to_numpy(), dt_tim["r0_p_o"].to_numpy(), dt_tim["r0_score"].to_numpy(), tau_r0_locked)
    r0_tim_cal = apply_r0_tau(dt_tim["r0_p_s"].to_numpy(), dt_tim["r0_p_o"].to_numpy(), dt_tim["r0_score"].to_numpy(), tau_r0_cal)
    sm_tim_lk_t, sm_tim_lk_g = apply_sm13_tau(dt_tim["sm13_fused_p_s"].to_numpy(), dt_tim["sm13_fused_p_o"].to_numpy(), dg_tim["sm13_gap_prob_i"].to_numpy(), SM13_LOCKED_TAU_T, SM13_LOCKED_TAU_G)
    sm_tim_cal_t, sm_tim_cal_g = apply_sm13_tau(dt_tim["sm13_fused_p_s"].to_numpy(), dt_tim["sm13_fused_p_o"].to_numpy(), dg_tim["sm13_gap_prob_i"].to_numpy(), tau_sm13_cal_t, tau_sm13_cal_g)

    # Top native phones falsely flagged on TIMIT by each system
    timit_top_phones = {}
    for name, pt_arr in [("vanilla", v_tim_t), ("r0_cal", r0_tim_cal), ("r0_locked", r0_tim_lk), ("sm13_cal", sm_tim_cal_t)]:
        c = Counter(dt_tim["canonical_phone"].to_numpy()[np.isin(pt_arr, ["S", "O"])].tolist())
        timit_top_phones[name] = c.most_common(12)

    timit_fa_summary = {
        "n_utts": n_tim_u, "n_targets": n_tim_t, "n_gaps": n_tim_g,
        "native_surface_variation_count": int(np.sum(np.isin(dt_tim["gold_label_alt"].to_numpy(), ["S", "O"]))),
        "vanilla": timit_fa(v_tim_t, v_tim_g),
        "r0_locked": timit_fa(r0_tim_lk, v_tim_g),
        "r0_cal_transfer": timit_fa(r0_tim_cal, v_tim_g),
        "sm13_locked": timit_fa(sm_tim_lk_t, sm_tim_lk_g),
        "sm13_cal_transfer": timit_fa(sm_tim_cal_t, sm_tim_cal_g),
        "top_flagged_phones": timit_top_phones,
    }

    full_report = {
        "transferred_dev18_thresholds": {
            "r0_locked_mean": round(tau_r0_locked, 6),
            "r0_cal_f1_mean": round(tau_r0_cal, 6),
            "sm13_cal_f1_tau_t_mean": round(tau_sm13_cal_t, 6),
            "sm13_cal_f1_tau_g_mean": round(tau_sm13_cal_g, 6),
            "sm13_locked_tau_t": SM13_LOCKED_TAU_T,
            "sm13_locked_tau_g": SM13_LOCKED_TAU_G,
        },
        "cohorts": {
            "dev18": dev18_summary,
            "held6": held6_res,
            "spontaneous": spont_res,
            "saa27": saa27_summary,
            "epadb_test": epadb_res,
            "epadb_test_unmasked_sensitivity": epadb_alt_res,
            "timit_core_surface_sensitivity": timit_surf,
        },
        "timit_native_fa_control": timit_fa_summary,
        "sm13_autopsy": {
            "learned_weights": sm13_weights,
            "saa27_decomp": sm13_saa_decomp,
            "dev18_180_decomp": sm13_dev_decomp,
        },
    }

    out_json = OUT_DIR / "FULL_BENCHMARK_AND_DISSECTION.json"
    out_json.write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved full JSON report: {out_json}")

    # Pretty-print concise summary tables to stdout
    print("\n" + "=" * 110)
    print("MASTER 6-COHORT STRICT MICRO-F1 & THRESHOLD-FREE PR-AUC SUMMARY")
    print("=" * 110)
    header = f"{'Cohort':18s} | {'Utts/Err':11s} | {'Van μF1':7s} | {'R0 Lk':6s} | {'R0 Cal':6s} | {'R0 Or':6s} | {'SM13 Lk':7s} | {'SM13 Cal':8s} | {'SM13 Or':7s} | {'SM13_Neu':8s} | {'R0 AP':6s} | {'SM13 AP':7s} | {'Neu AP':6s}"
    print(header)
    print("-" * len(header))

    def fmt(x):
        return f"{x*100:5.2f}%" if x is not None else "  N/A "

    def fmt_ap(x):
        return f"{x:.4f}" if x is not None else "  N/A "

    # DEV-18
    d = dev18_summary
    print(f"{'L2A DEV-18':18s} | {d['n_utts']:4d}/{d['vanilla']['n_gold_errors']:5d} | "
          f"{fmt(d['vanilla']['strict']['f1']):7s} | {fmt(d['r0_locked']['strict']['f1']):6s} | "
          f"{fmt(d['r0_cal_f1_oof']['strict']['f1']):6s} | {fmt(d['r0_oracle']['metrics']['strict']['f1']):6s} | "
          f"{fmt(d['sm13_locked']['strict']['f1']):7s} | {fmt(d['sm13_cal_f1_oof']['strict']['f1']):8s} | "
          f"{fmt(d['sm13_oracle']['metrics']['strict']['f1']):7s} | {fmt(sm13_dev_decomp['neural_oracle']['strict']['f1']):8s} | "
          f"{fmt_ap(d['threshold_free']['r0_target_pr_auc']):6s} | {fmt_ap(d['threshold_free']['sm13_fused_max_target_pr_auc']):7s} | {fmt_ap(sm13_dev_decomp['neural_sum_pr_auc']):6s}")

    for label, obj in [
        ("L2A Held-6", held6_res),
        ("L2A Spontaneous", spont_res),
        ("EpaDB TEST", epadb_res),
        ("EpaDB (unmasked)", epadb_alt_res),
    ]:
        print(f"{label:18s} | {obj['n_utts']:4d}/{obj['vanilla']['n_gold_errors']:5d} | "
              f"{fmt(obj['vanilla']['strict']['f1']):7s} | {fmt(obj['r0_locked']['strict']['f1']):6s} | "
              f"{fmt(obj['r0_cal_transfer']['strict']['f1']):6s} | {fmt(obj['r0_oracle']['metrics']['strict']['f1']):6s} | "
              f"{fmt(obj['sm13_locked']['strict']['f1']):7s} | {fmt(obj['sm13_cal_transfer']['strict']['f1']):8s} | "
              f"{fmt(obj['sm13_oracle']['metrics']['strict']['f1']):7s} | {fmt(obj['sm13_neural_only_oracle']['metrics']['strict']['f1']):8s} | "
              f"{fmt_ap(obj['threshold_free']['r0_target_pr_auc']):6s} | {fmt_ap(obj['threshold_free']['sm13_fused_max_target_pr_auc']):7s} | {fmt_ap(obj['threshold_free']['sm13_neural_target_pr_auc']):6s}")

    s = saa27_summary
    print(f"{'SAA-27 (SAA-Cal)':18s} | {s['n_utts']:4d}/{s['vanilla']['n_gold_errors']:5d} | "
          f"{fmt(s['vanilla']['strict']['f1']):7s} | {fmt(s['r0_locked']['strict']['f1']):6s} | "
          f"{fmt(s['r0_cal_saa18']['strict']['f1']):6s} | {fmt(s['r0_oracle']['metrics']['strict']['f1']):6s} | "
          f"{fmt(s['sm13_locked']['strict']['f1']):7s} | {fmt(s['sm13_cal_saa18']['strict']['f1']):8s} | "
          f"{fmt(s['sm13_oracle']['metrics']['strict']['f1']):7s} | {fmt(sm13_saa_decomp['neural_oracle']['strict']['f1']):8s} | "
          f"{fmt_ap(s['threshold_free']['r0_target_pr_auc']):6s} | {fmt_ap(s['threshold_free']['sm13_fused_max_target_pr_auc']):7s} | {fmt_ap(sm13_saa_decomp['neural_sum_pr_auc']):6s}")
    print(f"{'SAA-27 (DEV-Cal)':18s} | {s['n_utts']:4d}/{s['vanilla']['n_gold_errors']:5d} | "
          f"{fmt(s['vanilla']['strict']['f1']):7s} | {fmt(s['r0_locked']['strict']['f1']):6s} | "
          f"{fmt(s['r0_cal_transfer_dev18']['strict']['f1']):6s} | {fmt(s['r0_oracle']['metrics']['strict']['f1']):6s} | "
          f"{fmt(s['sm13_locked']['strict']['f1']):7s} | {fmt(s['sm13_cal_transfer_dev18']['strict']['f1']):8s} | "
          f"{fmt(s['sm13_oracle']['metrics']['strict']['f1']):7s} | {fmt(sm13_saa_decomp['neural_oracle']['strict']['f1']):8s} | "
          f"{fmt_ap(s['threshold_free']['r0_target_pr_auc']):6s} | {fmt_ap(s['threshold_free']['sm13_fused_max_target_pr_auc']):7s} | {fmt_ap(sm13_saa_decomp['neural_sum_pr_auc']):6s}")

    print("\n=== TIMIT CORE NATIVE L1 FALSE-ALARM CONTROL (192 utts, 6,374 target phones) ===")
    for k in ["vanilla", "r0_locked", "r0_cal_transfer", "sm13_locked", "sm13_cal_transfer"]:
        print(f"  {k:20s}: {timit_fa_summary[k]}")

    print("\n=== SM13 E4 BUG PROOF (NEURAL vs E4 vs FUSED) ===")
    print("  DEV-18-180:", {k: v for k, v in sm13_dev_decomp.items() if not isinstance(v, dict)})
    print("  DEV-18-180 by membership:", json.dumps(sm13_dev_decomp["by_sm13_membership"], indent=2))
    print("  SAA-27:", {k: v for k, v in sm13_saa_decomp.items() if not isinstance(v, dict)})
    print("  SM13 learned weights:", sm13_weights)

    print("\n=== KOEL BASE UNMAPPED TOKEN HEMORRHAGE ===")
    for name, obj in [("held6", held6_res), ("spontaneous", spont_res), ("epadb_test", epadb_res), ("timit_core", timit_native)]:
        ms = obj["meta_stats"]
        print(f"  {name:14s}: unmapped={ms['total_unmapped_koel_tokens']}/{ms['total_emitted_koel_tokens']} ({ms['unmapped_rate_pct']}%) | utts={ms['utts_with_unmapped']}/{obj['n_utts']} | top={ms['top_unmapped_tokens'][:6]}")
        print(f"    ? abstentions on valid targets: {obj['vanilla_q_abstention']}")


if __name__ == "__main__":
    main()
