"""Task 1: Package & verify Cal-locked 3-seed Head Ensemble (O5) + Gap Platt (O4) on S0_clean.

Strictly read-only with respect to stage0_s0_clean checkpoints and BACKBONE_REP_CAMPAIGN_20261122_v1.
Locks all ensemble thresholds & Gap Platt parameters strictly on Cal (clean_11) with tight L1 guard,
seals Eval predictions to Parquet BEFORE joining historical Eval gold, and measures GTX 1650 head
latency overhead vs single-seed S0_clean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import average_precision_score
import torch

from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import (
    BASE_KOEL_CACHE,
    load_splits_no_eval_gold,
    pool_koel_iso,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import save_json, sha_file
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import infer_sf_logits
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import seal_predictions
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.single_factor_models import (
    FACTOR_ARMS,
    SingleFactorV2,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import domain

from .calibration import (
    OrthoPolicy,
    apply_ortho_policy,
    calibrate_ortho_policy,
    ensemble_predictions,
)

S0_CLEAN_DIR = Path(
    "research/BACKBONE_REP_CAMPAIGN_20261122_v1/runs/full_clean11_v1/stage0_s0_clean"
)
EVAL_GOLD_PATH = Path(
    "research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/v2_prompt_eos_repair_v1/splits/eval_gold.jsonl"
)
SEEDS: Tuple[int, ...] = (20261121, 20261122, 20261123)
COHORTS: Tuple[str, ...] = (
    "held6",
    "epadb_test",
    "saa27",
    "spontaneous",
    "timit_core",
    "timit_core_corrupt_ctrl",
)
HELD6_SPEAKERS = {"ABA", "ASI", "BWC", "EBVS", "HQTV", "LXC"}
VALID_GOLD = {"C", "S", "O", "N", "I"}


def _norm_from_ckpt(ckpt_norm: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for k, v in ckpt_norm.items():
        m, s = v
        out[k] = (
            m.cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m, dtype=np.float32),
            s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s, dtype=np.float32),
        )
    return out


def _load_gold_df() -> pd.DataFrame:
    rows = []
    with EVAL_GOLD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c, uid = r["cohort"], r["uid"]
            for i, g in enumerate(r["gold_targets"]):
                rows.append((c, uid, "target", i, g))
            for i, g in enumerate(r["gold_gaps"]):
                rows.append((c, uid, "gap", i, g))
    return pd.DataFrame(rows, columns=["cohort", "uid", "domain", "index", "gold"])


def _build_eval_records(
    eval_prompts: Sequence[Dict[str, Any]],
    t_out: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    t_off, g_off = 0, 0
    for r in eval_prompts:
        L = len(r["canonical_phones"])
        c, uid, spk = r["cohort"], r["uid"], r["speaker"]
        for i in range(L):
            idx = t_off + i
            records.append({
                "cohort": c,
                "uid": uid,
                "speaker": spk,
                "domain": "target",
                "index": i,
                "canonical_phone": r["canonical_phones"][i],
                "score": float(t_out["target_score"][idx]),
                "call": str(t_out["target_call"][idx]),
                "p_s": float(t_out["p_s"][idx]),
                "p_o": float(t_out["p_o"][idx]),
            })
        for i in range(L + 1):
            idx = g_off + i
            records.append({
                "cohort": c,
                "uid": uid,
                "speaker": spk,
                "domain": "gap",
                "index": i,
                "canonical_phone": None,
                "score": float(t_out["gap_score"][idx]),
                "call": str(t_out["gap_call"][idx]),
                "p_s": None,
                "p_o": None,
            })
        t_off += L
        g_off += L + 1
    assert t_off == len(t_out["target_score"])
    assert g_off == len(t_out["gap_score"])
    return records


def score_eval_dir(pred_dir: Path, gold_df: pd.DataFrame) -> Dict[str, Any]:
    parts = [pd.read_parquet(pred_dir / f"{c}.parquet") for c in COHORTS]
    df = pd.concat(parts, ignore_index=True).merge(
        gold_df, on=["cohort", "uid", "domain", "index"], how="inner"
    )
    out: Dict[str, Any] = {}
    for name in ("held6", "epadb_test", "saa27", "spont_h6", "timit_core", "ctrl"):
        if name == "spont_h6":
            sub = df[
                (df["cohort"] == "spontaneous")
                & (df["speaker"].str.upper().isin(HELD6_SPEAKERS))
                & (df["gold"].isin(VALID_GOLD))
            ]
        elif name == "ctrl":
            sub = df[(df["cohort"] == "timit_core_corrupt_ctrl") & (df["gold"].isin(VALID_GOLD))]
            ge = (sub["gold"] != "C") & (sub["gold"] != "N")
            pe = (sub["call"] != "C") & (sub["call"] != "N")
            out["ctrl_rec"] = round(100.0 * float(pe[ge].mean()), 2)
            continue
        elif name == "timit_core":
            sub = df[df["cohort"] == "timit_core"]
            t_fa = int(((sub["domain"] == "target") & (sub["call"] != "C")).sum())
            g_fa = int(((sub["domain"] == "gap") & (sub["call"] != "N")).sum())
            out["timit_core"] = {
                "fa_per_utt": round((t_fa + g_fa) / 192.0, 3),
                "target_fa": t_fa,
                "gap_fa": g_fa,
            }
            continue
        else:
            sub = df[(df["cohort"] == name) & (df["gold"].isin(VALID_GOLD))]

        g, p = sub["gold"].values, sub["call"].values
        ge = (g != "C") & (g != "N")
        pe = (p != "C") & (p != "N")
        tp = int(np.sum(ge & (p == g)))
        fp = int(np.sum(pe & (p != g)))
        fn = int(np.sum(ge & (p != g)))
        f1 = 200.0 * tp / max(1, 2 * tp + fp + fn)

        cls_f1 = {}
        for cls in ("S", "O", "I"):
            tk = int(np.sum((g == cls) & (p == cls)))
            fk = int(np.sum((g != cls) & (p == cls)))
            nk = int(np.sum((g == cls) & (p != cls)))
            cls_f1[cls] = round(200.0 * tk / max(1, 2 * tk + fk + nk), 2)

        s2o = int(np.sum((g == "S") & (p == "O")))
        o2s = int(np.sum((g == "O") & (p == "S")))
        ts = sub[sub["domain"] == "target"]
        gs = sub[sub["domain"] == "gap"]
        out[name] = {
            "strict_f1": round(f1, 2),
            "target_ap": round(float(average_precision_score((ts["gold"].values != "C").astype(int), ts["score"].values)), 4),
            "gap_ap": round(float(average_precision_score((gs["gold"].values == "I").astype(int), gs["score"].values)), 4),
            "S_f1": cls_f1["S"],
            "O_f1": cls_f1["O"],
            "I_f1": cls_f1["I"],
            "s_to_o": s2o,
            "o_to_s": o2s,
        }
    return out


def _measure_head_latency(
    models: Sequence[SingleFactorV2],
    rows: Sequence[Dict[str, Any]],
    feats: Sequence[Dict[str, np.ndarray]],
    norm: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> Dict[str, float]:
    from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import collate_sf

    # Warmup + 100 Held-6 utterances at batch=1
    h6_idx = [i for i, r in enumerate(rows) if r["cohort"] == "held6"][:105]
    samples = [
        collate_sf([rows[i]], [feats[i]], norm, device, labels=False)[0]
        for i in h6_idx
    ]
    with torch.inference_mode():
        for inp in samples[:5]:
            for m in models:
                _ = m(**inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_single, t_ens3 = [], []
        for inp in samples[5:]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = models[0](**inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_single.append(1000.0 * (time.perf_counter() - t0))

            t1 = time.perf_counter()
            for m in models:
                _ = m(**inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_ens3.append(1000.0 * (time.perf_counter() - t1))

    return {
        "device": str(device),
        "n_utts": len(t_single),
        "single_head_p50_ms": round(float(np.median(t_single)), 2),
        "single_head_p95_ms": round(float(np.percentile(t_single, 95)), 2),
        "ens3_head_p50_ms": round(float(np.median(t_ens3)), 2),
        "ens3_head_p95_ms": round(float(np.percentile(t_ens3, 95)), 2),
        "overhead_p50_ms": round(float(np.median(t_ens3) - np.median(t_single)), 2),
    }


def package_s0_clean_ensemble(
    out_dir: Path,
    *,
    device_str: str = "cuda",
    max_l1_fa_per_1k: float = 1.0,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    t0 = time.perf_counter()

    _, cal_rows, eval_prompts = load_splits_no_eval_gold("clean_11")
    xc = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in cal_rows]
    xe = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in eval_prompts]

    l2_idx = [i for i, r in enumerate(cal_rows) if domain(r) != "l1"]
    l1_idx = [i for i, r in enumerate(cal_rows) if domain(r) == "l1"]
    l2_t_gold = np.concatenate([np.array(cal_rows[i]["gold_targets"]) for i in l2_idx])
    l2_g_gold = np.concatenate([np.array(cal_rows[i]["gold_gaps"]) for i in l2_idx])

    cfg = FACTOR_ARMS["iso_no_hwin"]
    models: List[SingleFactorV2] = []
    ckpt_records = []
    cal_t_l2, cal_g_l2, cal_t_l1, cal_g_l1 = [], [], [], []
    eval_t_all, eval_g_all = [], []
    shared_norm = None

    for s in SEEDS:
        cp = S0_CLEAN_DIR / f"seed_{s}/control.pt"
        ckpt = torch.load(cp, map_location="cpu", weights_only=False)
        norm = _norm_from_ckpt(ckpt["norm"])
        shared_norm = norm
        model = SingleFactorV2(cfg).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
        ckpt_records.append({
            "seed": s,
            "path": str(cp),
            "sha256": sha_file(cp),
            "single_policy": ckpt["policy"],
        })

        cal_lg = infer_sf_logits(model, cal_rows, xc, norm, dev, batch_size=32)
        cal_t_l2.append(np.concatenate([cal_lg[i].target for i in l2_idx], axis=0))
        cal_g_l2.append(np.concatenate([cal_lg[i].gap for i in l2_idx], axis=0))
        cal_t_l1.append(np.concatenate([cal_lg[i].target for i in l1_idx], axis=0))
        cal_g_l1.append(np.concatenate([cal_lg[i].gap for i in l1_idx], axis=0))

        ev_lg = infer_sf_logits(model, eval_prompts, xe, norm, dev, batch_size=32)
        eval_t_all.append(np.concatenate([lg.target for lg in ev_lg], axis=0))
        eval_g_all.append(np.concatenate([lg.gap for lg in ev_lg], axis=0))

    # Cal-only comparison of prob_mean vs logit_mean (both with O4 Gap Platt + tight L1 guard)
    cal_candidates = {}
    for mode in ("prob_mean", "logit_mean"):
        et_l2, eg_l2, is_p = ensemble_predictions(cal_t_l2, cal_g_l2, mode=mode, is_probs=False)
        et_l1, eg_l1, _ = ensemble_predictions(cal_t_l1, cal_g_l1, mode=mode, is_probs=False)
        pol, extra = calibrate_ortho_policy(
            et_l2, l2_t_gold, eg_l2, l2_g_gold, et_l1, eg_l1,
            is_probs=is_p,
            tune_rho_o=False,
            fit_gap_temp=True,
            known_gap_pos_weight=8.0,
            max_l1_fa_per_1k=max_l1_fa_per_1k,
            ensemble_mode=mode,
            n_seeds=len(SEEDS),
        )
        # Also compute Cal tAP / gAP
        mt = np.isin(l2_t_gold, ["C", "S", "O"])
        mg = np.isin(l2_g_gold, ["N", "I"])
        t_p = et_l2 if is_p else softmax(et_l2, axis=-1)
        g_p = apply_ortho_policy(et_l2[:1], eg_l2, pol, is_probs=is_p)["gap_score"]
        cal_tap = float(average_precision_score((l2_t_gold[mt] != "C").astype(int), t_p[mt, 1] + t_p[mt, 2]))
        cal_gap = float(average_precision_score((l2_g_gold[mg] == "I").astype(int), g_p[mg]))
        cal_candidates[mode] = {
            "policy": pol,
            "extra": extra,
            "is_probs": is_p,
            "cal_t_ap": round(cal_tap, 5),
            "cal_g_ap": round(cal_gap, 5),
        }

    # Select winning ensemble mode strictly by Cal strict F1 (tie-break: lower Cal L1 FA)
    best_mode = max(
        cal_candidates.keys(),
        key=lambda m: (
            cal_candidates[m]["policy"].cal_guard_met,
            cal_candidates[m]["policy"].cal_l2_strict_f1,
            -cal_candidates[m]["policy"].cal_l1_fa_per_1k,
        ),
    )
    chosen = cal_candidates[best_mode]
    pol: OrthoPolicy = chosen["policy"]

    # Measure GTX 1650 head-only latency overhead
    latency = _measure_head_latency(models, eval_prompts, xe, shared_norm, dev)

    # Save ENSEMBLE_CARD.json BEFORE sealing Eval predictions
    card = {
        "name": "S0_clean_ens3_o5_o4",
        "split": "clean_11",
        "selected_mode_by_cal": best_mode,
        "max_l1_fa_per_1k_guard": max_l1_fa_per_1k,
        "source_checkpoints": ckpt_records,
        "policy": pol.to_dict(),
        "gap_calibration": chosen["extra"]["gap_calibration"],
        "cal_candidates": {
            m: {
                "policy": d["policy"].to_dict(),
                "cal_t_ap": d["cal_t_ap"],
                "cal_g_ap": d["cal_g_ap"],
                "gap_calibration": d["extra"]["gap_calibration"],
            }
            for m, d in cal_candidates.items()
        },
        "head_latency": latency,
    }
    save_json(out_dir / "ENSEMBLE_CARD.json", card)
    card_sha = sha_file(out_dir / "ENSEMBLE_CARD.json")

    # Apply locked policy to Eval & SEAL BEFORE opening gold
    ev_t, ev_g, ev_is_p = ensemble_predictions(eval_t_all, eval_g_all, mode=best_mode, is_probs=False)
    ev_out = apply_ortho_policy(ev_t, ev_g, pol, is_probs=ev_is_p)
    records = _build_eval_records(eval_prompts, ev_out)
    seal = seal_predictions(out_dir, records, card_sha)

    # Post-seal single Eval pass + comparison against the 3 individual S0_clean seeds
    gold_df = _load_gold_df()
    ens_eval = score_eval_dir(out_dir / "eval_predictions", gold_df)
    single_evals = {
        str(s): score_eval_dir(S0_CLEAN_DIR / f"seed_{s}/eval_predictions", gold_df)
        for s in SEEDS
    }
    single_mean = {}
    for c in ("held6", "epadb_test", "saa27", "spont_h6"):
        single_mean[c] = {
            k: round(float(np.mean([single_evals[str(s)][c][k] for s in SEEDS])), 4 if "ap" in k else 2)
            for k in ("strict_f1", "target_ap", "gap_ap", "S_f1", "O_f1", "I_f1")
        }
    single_mean["timit_core"] = {
        "fa_per_utt": round(float(np.mean([single_evals[str(s)]["timit_core"]["fa_per_utt"] for s in SEEDS])), 3),
    }
    single_mean["ctrl_rec"] = round(float(np.mean([single_evals[str(s)]["ctrl_rec"] for s in SEEDS])), 2)

    results = {
        "status": "COMPLETE_VERIFIED_ENSEMBLE",
        "wall_s": round(time.perf_counter() - t0, 1),
        "card_sha256": card_sha,
        "seal_sha256": sha_file(out_dir / "PREDICTIONS_SEAL.json"),
        "selected_mode_by_cal": best_mode,
        "policy": pol.to_dict(),
        "gap_calibration": chosen["extra"]["gap_calibration"],
        "head_latency": latency,
        "s0_clean_single_seed_mean": single_mean,
        "s0_clean_single_seeds": single_evals,
        "s0_clean_ens3_o5_o4": ens_eval,
        "seal_files": seal["files"],
    }
    save_json(out_dir / "RESULTS.json", results)

    print(f"=== TASK 1 COMPLETE: S0_clean 3-Seed Ensemble (O5+O4) | mode={best_mode} | wall={results['wall_s']}s ===")
    print(f"  Cal policy: tau_t={pol.tau_t} tau_g={pol.tau_g} (T={pol.gap_temp:.3f}, b={pol.gap_bias:.3f}) | "
          f"Cal F1={pol.cal_l2_strict_f1:.2f}% | Cal L1={pol.cal_l1_fa_per_1k:.2f}/1k (guard<={max_l1_fa_per_1k})")
    print(f"  Head latency ({latency['device']}): single P50={latency['single_head_p50_ms']}ms -> ens3 P50={latency['ens3_head_p50_ms']}ms (+{latency['overhead_p50_ms']}ms)")
    print(f"\n  {'Cohort':12s} | {'1-seed TB (tAP / gAP / I)':28s} | {'Ens3 O5+O4 (tAP / gAP / I)':28s} | Delta F1")
    for c in ("held6", "epadb_test", "saa27", "spont_h6"):
        m1, me = single_mean[c], ens_eval[c]
        print(f"  {c:12s} | {m1['strict_f1']:5.2f}% ({m1['target_ap']:.4f}/{m1['gap_ap']:.4f}/{m1['I_f1']:4.1f}) | "
              f"{me['strict_f1']:5.2f}% ({me['target_ap']:.4f}/{me['gap_ap']:.4f}/{me['I_f1']:4.1f}) | {me['strict_f1']-m1['strict_f1']:+.2f} pp")
    print(f"  {'timit_core':12s} | {single_mean['timit_core']['fa_per_utt']:.3f} FA/utt                  | "
          f"{ens_eval['timit_core']['fa_per_utt']:.3f} FA/utt ({ens_eval['timit_core']['target_fa']}t+{ens_eval['timit_core']['gap_fa']}g)       | "
          f"{ens_eval['timit_core']['fa_per_utt']-single_mean['timit_core']['fa_per_utt']:+.3f}")
    print(f"  {'ctrl_rec':12s} | {single_mean['ctrl_rec']:.2f}%                        | "
          f"{ens_eval['ctrl_rec']:.2f}%                        | {ens_eval['ctrl_rec']-single_mean['ctrl_rec']:+.2f} pp")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research/ORTHO_IMPROVEMENTS_20261122_v1/runs/s0_clean_ens3_o5_o4"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-l1-fa", type=float, default=1.0)
    args = ap.parse_args()
    package_s0_clean_ensemble(args.out, device_str=args.device, max_l1_fa_per_1k=args.max_l1_fa)


if __name__ == "__main__":
    main()
