"""Read-only CPU replay & independent recount audit of fixed vs fix12_arch.

Answers Sol 6.2's Step 1 review items without any training or Eval threshold tuning:
  1. Independent recount (no calls to scoring.score_decisions) of sealed Parquets.
  2. Fair operating-point comparisons locked strictly on Calibrate:
       - native Cal-locked thresholds
       - common fixed thresholds (tau_t=0.43, tau_g=0.775)
       - Calibrate L2 recall-matched thresholds (matching fixed's Cal L2 target & gap recall)
     Separates target vs gap, S vs O vs I, S->O / O->S, and unseen Spontaneous Held-6 slice.
  3. High-can_lp regime (> -0.2) comparison of BOTH TP and FN (plus TN and FP).
  4. Held-6 S_same intersection with can_lp > -0.2, TP vs FN breakdown, and raw TextGrid audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    HELD6_GOLD,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    PHONE_TO_ID,
    normalize_canonical_phone,
)
from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    dp_align_csoi_realized,
)

from .data import (
    HERE,
    ROOT,
    admitted,
    read_frames,
    read_json,
    read_jsonl,
    save_json,
    sha_file,
)
from .features import Spec, pool_features
from .repair_model import LengthAwareV2
from .repaired_v2_arch import CodeFixConfig, RepairedCodeFixV2, pool_features_codefix
from .run_code_fixes_ablation import infer_codefix_logits
from .scoring import KEY, gold_frame, keyed_merge, read_predictions
from .training import domain, infer_logits, probabilities

DEFAULT_RUN = HERE / "runs/campaign_d5f8e89"
DEFAULT_REPAIR = HERE / "runs/v2_prompt_eos_repair_v1"
DEFAULT_CODEFIX = HERE / "runs/code_fixes_123_20261121_v1"
HELD6_SPEAKERS = frozenset({"ABA", "ASI", "BWC", "EBVS", "HQTV", "LXC"})


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "p": p, "r": r, "f1": f1}


def independent_count(
    t_gold: np.ndarray,
    t_call: np.ndarray,
    g_gold: np.ndarray,
    g_call: np.ndarray,
) -> dict:
    """Explicit standalone counter (independent of scoring.score_decisions)."""
    tv = t_gold != "unknown"
    gv = g_gold != "unknown"
    tg, tc = t_gold[tv], t_call[tv]
    gg, gc = g_gold[gv], g_call[gv]

    s_tp = int(((tg == "S") & (tc == "S")).sum())
    s_fp = int(((tg != "S") & (tc == "S")).sum())
    s_fn = int(((tg == "S") & (tc != "S")).sum())

    o_tp = int(((tg == "O") & (tc == "O")).sum())
    o_fp = int(((tg != "O") & (tc == "O")).sum())
    o_fn = int(((tg == "O") & (tc != "O")).sum())

    i_tp = int(((gg == "I") & (gc == "I")).sum())
    i_fp = int(((gg == "N") & (gc == "I")).sum())
    i_fn = int(((gg == "I") & (gc == "N")).sum())

    s_to_o = int(((tg == "S") & (tc == "O")).sum())
    o_to_s = int(((tg == "O") & (tc == "S")).sum())

    strict_tp = s_tp + o_tp + i_tp
    strict_fp = s_fp + o_fp + i_fp
    strict_fn = s_fn + o_fn + i_fn

    t_err_g = (tg == "S") | (tg == "O")
    t_err_c = (tc == "S") | (tc == "O")
    tb_tp = int((t_err_g & t_err_c).sum())
    tb_fp = int(((~t_err_g) & t_err_c).sum())
    tb_fn = int((t_err_g & (~t_err_c)).sum())

    bin_tp = tb_tp + i_tp
    bin_fp = tb_fp + i_fp
    bin_fn = tb_fn + i_fn

    return {
        "strict": _prf(strict_tp, strict_fp, strict_fn),
        "binary": _prf(bin_tp, bin_fp, bin_fn),
        "target_binary": _prf(tb_tp, tb_fp, tb_fn),
        "S": _prf(s_tp, s_fp, s_fn),
        "O": _prf(o_tp, o_fp, o_fn),
        "I": _prf(i_tp, i_fp, i_fn),
        "s_to_o": s_to_o,
        "o_to_s": o_to_s,
    }


def apply_policy_df(df: pd.DataFrame, tau_t: float, tau_g: float) -> pd.DataFrame:
    out = df.copy()
    is_t = out["domain"].eq("target")
    t_call = np.where(
        out["score"] >= tau_t, np.where(out["p_o"] > out["p_s"], "O", "S"), "C"
    )
    g_call = np.where(out["score"] >= tau_g, "I", "N")
    out["call"] = np.where(is_t, t_call, g_call)
    return out


def summarize_cohorts(df: pd.DataFrame) -> dict:
    res = {}
    for cohort, sub in df.groupby("cohort", sort=False):
        if cohort == "timit_core_corrupt_ctrl":
            t = sub[sub["domain"] == "target"]
            y = t["gold_label"].eq("S").to_numpy()
            flags = t["call"].isin(["S", "O"]).to_numpy()
            res[cohort] = {
                "detected": int((flags & y).sum()),
                "n_corrupt": int(y.sum()),
                "recall": float(flags[y].mean()),
                "unchanged_alert_rate": float(flags[~y].mean()),
            }
            continue
        t = sub[sub["domain"] == "target"]
        g = sub[sub["domain"] == "gap"]
        cnt = independent_count(
            t["gold_label"].to_numpy(),
            t["call"].to_numpy(),
            g["gold_label"].to_numpy(),
            g["call"].to_numpy(),
        )
        n_u = int(sub["uid"].nunique())
        cnt["n_utts"] = n_u
        if cohort == "timit_core":
            fa_t = int(t["call"].isin(["S", "O"]).sum())
            fa_g = int(g["call"].eq("I").sum())
            cnt.update(
                fa_total=fa_t + fa_g,
                fa_target=fa_t,
                fa_gap=fa_g,
                fa_per_utt=(fa_t + fa_g) / n_u,
                target_fa_per_1k=fa_t * 1000.0 / len(t),
                gap_fa_per_1k=fa_g * 1000.0 / len(g),
            )
        if cohort == "spontaneous":
            u = sub[sub["speaker"].isin(HELD6_SPEAKERS)]
            ut = u[u["domain"] == "target"]
            ug = u[u["domain"] == "gap"]
            cnt["held6_only"] = independent_count(
                ut["gold_label"].to_numpy(),
                ut["call"].to_numpy(),
                ug["gold_label"].to_numpy(),
                ug["call"].to_numpy(),
            )
            cnt["held6_only"]["n_utts"] = int(u["uid"].nunique())
        res[cohort] = cnt
    return res


def cal_scores_for_model(cache, cal_rows, ckpt_path: Path, kind: str):
    """Run deterministic single-utterance inference on Calibrate (CPU)."""
    d = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    norm = {k: tuple(t.numpy() for t in v) for k, v in d["norm"].items()}
    policy = d["policy"]
    if kind == "fixed":
        model = LengthAwareV2().eval()
        model.load_state_dict(d["state_dict"], strict=True)
        feats = [
            pool_features(*read_frames(cache, r), r["canonical_phones"], Spec(), None)
            for r in cal_rows
        ]
        logits = infer_logits(model, cal_rows, feats, norm, torch.device("cpu"), 1)
    else:
        cfg = CodeFixConfig(**d["config"])
        model = RepairedCodeFixV2(d_model=cfg.d_model, dropout=cfg.dropout).eval()
        model.load_state_dict(d["state_dict"], strict=True)
        feats = [
            pool_features_codefix(*read_frames(cache, r), r["canonical_phones"])
            for r in cal_rows
        ]
        logits = infer_codefix_logits(model, cal_rows, feats, norm, torch.device("cpu"), 1)

    l2 = {"t_sc": [], "t_po": [], "t_ps": [], "t_g": [], "g_sc": [], "g_g": []}
    l1 = {"t_sc": [], "g_sc": []}
    for r, lg in zip(cal_rows, logits, strict=True):
        tp, gp = probabilities(lg, r, policy)
        sc = tp[:, 1] + tp[:, 2]
        if domain(r) == "l1":
            l1["t_sc"].extend(sc.tolist())
            l1["g_sc"].extend(gp[:, 1].tolist())
        else:
            l2["t_sc"].extend(sc.tolist())
            l2["t_ps"].extend(tp[:, 1].tolist())
            l2["t_po"].extend(tp[:, 2].tolist())
            l2["t_g"].extend(r["gold_targets"])
            l2["g_sc"].extend(gp[:, 1].tolist())
            l2["g_g"].extend(r["gold_gaps"])
    return (
        {k: np.asarray(v) for k, v in l2.items()},
        {k: np.asarray(v) for k, v in l1.items()},
        policy,
    )


def match_cal_recall(cal_l2_cand, target_recall_ref: float, gap_recall_ref: float):
    """Pick exact quantile threshold on Cal L2 positives to match fixed's Cal L2 recall."""
    t_pos = cal_l2_cand["t_sc"][np.isin(cal_l2_cand["t_g"], ["S", "O"])]
    g_pos = cal_l2_cand["g_sc"][cal_l2_cand["g_g"] == "I"]
    # Threshold such that fraction >= tau matches ref recall
    tau_t = float(np.quantile(t_pos, max(0.0, min(1.0, 1.0 - target_recall_ref))))
    tau_g = float(np.quantile(g_pos, max(0.0, min(1.0, 1.0 - gap_recall_ref))))
    return tau_t, tau_g


def parse_textgrid_phone_intervals(tg_path: Path) -> list[tuple[float, float, str]]:
    """Parse the 'phones' interval tier from a Praat TextGrid."""
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'name\s*=\s*"phones"(.*?)(?:item\s*\[\d+\]:|\Z)', text, re.S)
    if not m:
        return []
    block = m.group(1)
    out = []
    for im in re.finditer(
        r'xmin\s*=\s*([0-9.]+)\s+xmax\s*=\s*([0-9.]+)\s+text\s*=\s*"([^"]*)"', block
    ):
        out.append((float(im.group(1)), float(im.group(2)), im.group(3).strip()))
    return out


def analyze_can_lp_and_s_same(cache, eval_prompts, gold_df, frames_by_arm):
    """Compare BOTH TP and FN (and TN vs FP) across can_lp regimes, plus Held-6 S_same audit."""
    # Precompute can_lp_vit_max for all eval cohorts
    can_lp_rows = []
    for r in eval_prompts:
        if r["cohort"] == "timit_core_corrupt_ctrl":
            continue
        h, lp = read_frames(cache, r)
        clp = pool_features(h, lp, r["canonical_phones"], Spec(), None)[
            "target_scalars"
        ][:, 7]
        for i, v in enumerate(clp):
            can_lp_rows.append((r["cohort"], r["uid"], "target", i, float(v)))
    clp_df = pd.DataFrame(can_lp_rows, columns=KEY + ["can_lp"])

    # Tag S_same vs S_diff on Held-6 and inspect raw L2-ARCTIC TextGrid strings
    s_same_meta = {}
    tg_tag_counts = Counter()
    s_same_samples = []
    for r in read_jsonl(HELD6_GOLD):
        can = [normalize_canonical_phone(p) for p in r["canonical_phones"]]
        real_raw = [
            normalize_canonical_phone(p)
            if p not in {"", "SIL", "sp", "spn", "sil"}
            else ""
            for p in r["realized_phones"]
        ]
        _, _, rids = dp_align_csoi_realized(can, [p for p in real_raw if p])
        tg_intervals = None
        for i, y in enumerate(r["gold_targets"]):
            if y != "S":
                continue
            is_same = rids[i] == PHONE_TO_ID[can[i]]
            s_same_meta[(r["uid"], i)] = is_same
            if is_same:
                # Inspect raw TextGrid if available
                wav_p = Path(r["audio_path"])
                if not wav_p.is_absolute():
                    wav_p = ROOT / wav_p
                tg_p = wav_p.parent.parent / "annotation" / (wav_p.stem + ".TextGrid")
                if tg_intervals is None and tg_p.is_file():
                    tg_intervals = parse_textgrid_phone_intervals(tg_p)
                raw_tag = None
                t_span = None
                if tg_intervals:
                    # Find matching error-tagged interval (contains ',')
                    err_ints = [
                        (s0, e0, txt)
                        for s0, e0, txt in tg_intervals
                        if "," in txt and txt.split(",")[0].strip().rstrip("012").upper() == can[i]
                    ]
                    if err_ints:
                        # Count error code types in raw TextGrid
                        for _, _, txt in err_ints:
                            parts = [x.strip() for x in txt.split(",")]
                            if len(parts) >= 3:
                                tag_key = f"{parts[2]}:{'*' in parts[1]}"
                            else:
                                tag_key = "other"
                            tg_tag_counts[tag_key] += 1
                        s0, e0, raw_tag = err_ints[0]
                        t_span = [round(s0, 3), round(e0, 3)]
                s_same_samples.append(
                    {
                        "uid": r["uid"],
                        "speaker": str(r["speaker_id"]).upper(),
                        "index": i,
                        "canonical_phone": can[i],
                        "audio_path": str(wav_p.relative_to(ROOT))
                        if wav_p.is_relative_to(ROOT)
                        else str(wav_p),
                        "raw_textgrid_example": raw_tag,
                        "textgrid_span_s": t_span,
                    }
                )

    regimes = {}
    s_same_breakdown = {}
    for arm_name, df in frames_by_arm.items():
        t = df[df["domain"] == "target"].merge(clp_df, on=KEY, how="inner")
        regimes[arm_name] = {}
        for cohort in ("held6", "epadb_test", "saa27", "spontaneous", "spontaneous_held6_only"):
            if cohort == "spontaneous_held6_only":
                sub = t[(t["cohort"] == "spontaneous") & (t["speaker"].isin(HELD6_SPEAKERS))]
            else:
                sub = t[t["cohort"] == cohort]
            sub = sub[sub["gold_label"] != "unknown"]
            out_c = {}
            for rname, mask in [
                ("high_can_lp_gt_neg02", sub["can_lp"] > -0.2),
                ("border_neg2_to_neg02", (sub["can_lp"] <= -0.2) & (sub["can_lp"] >= -2.0)),
                ("low_can_lp_lt_neg2", sub["can_lp"] < -2.0),
            ]:
                x = sub[mask]
                g_err = x["gold_label"].isin(["S", "O"])
                p_err = x["call"].isin(["S", "O"])
                tp = int((g_err & p_err).sum())
                fn = int((g_err & (~p_err)).sum())
                fp = int(((~g_err) & p_err).sum())
                tn = int(((~g_err) & (~p_err)).sum())
                out_c[rname] = {
                    "n_slots": int(len(x)),
                    "gold_errors": tp + fn,
                    "tp_binary": tp,
                    "fn_binary": fn,
                    "fp_binary": fp,
                    "tn_binary": tn,
                    "recall_in_slice": tp / (tp + fn) if (tp + fn) else None,
                    "precision_in_slice": tp / (tp + fp) if (tp + fp) else None,
                }
            regimes[arm_name][cohort] = out_c

        # Held-6 S_same vs S_diff (both overall and intersected with can_lp > -0.2)
        h6 = t[(t["cohort"] == "held6") & (t["gold_label"] == "S")].copy()
        h6["is_s_same"] = [
            s_same_meta.get((u, int(i)), False)
            for u, i in zip(h6["uid"], h6["index"], strict=True)
        ]
        sb = {}
        for lbl, m_s in [("S_same", h6["is_s_same"]), ("S_diff", ~h6["is_s_same"])]:
            sub = h6[m_s]
            hi = sub["can_lp"] > -0.2
            p_err = sub["call"].isin(["S", "O"])
            p_s = sub["call"].eq("S")
            sb[lbl] = {
                "total": int(len(sub)),
                "can_lp_gt_neg02": int(hi.sum()),
                "frac_can_lp_gt_neg02": float(hi.mean()) if len(sub) else 0.0,
                "tp_binary": int(p_err.sum()),
                "tp_strict_s": int(p_s.sum()),
                "fn_binary": int((~p_err).sum()),
                "binary_recall": float(p_err.mean()) if len(sub) else 0.0,
                "in_can_lp_gt_neg02": {
                    "n": int(hi.sum()),
                    "tp_binary": int((hi & p_err).sum()),
                    "fn_binary": int((hi & (~p_err)).sum()),
                    "binary_recall": float(p_err[hi].mean()) if hi.any() else None,
                },
            }
        s_same_breakdown[arm_name] = sb

    # Deterministic 20-sample blind-check roster of Held-6 S_same cases
    rng = np.random.default_rng(20261121)
    idx_pick = sorted(
        rng.choice(len(s_same_samples), size=min(20, len(s_same_samples)), replace=False).tolist()
    )
    return {
        "can_lp_tp_fn_fp_tn_regimes": regimes,
        "held6_s_same_vs_s_diff": s_same_breakdown,
        "held6_s_same_raw_textgrid_tag_counts": dict(tg_tag_counts),
        "held6_s_same_blind_check_sample_20": [s_same_samples[i] for i in idx_pick],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--repair", type=Path, default=DEFAULT_REPAIR)
    p.add_argument("--codefix", type=Path, default=DEFAULT_CODEFIX)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CODEFIX / "REPLAY_AUDIT.json",
    )
    a = p.parse_args()
    t0 = time.perf_counter()

    cache = a.run / "frames"
    splits = a.repair / "splits"
    cal_rows = admitted(splits, "cal.jsonl")
    ev_prompts = admitted(splits, "eval_prompts.jsonl")
    gold_df = gold_frame(admitted(splits, "eval_gold.jsonl"))

    fixed_eval_dir = a.repair / "matched_seed20261121_e14_b16/eval_fixed"
    fix12_eval_dir = a.codefix / "eval_fix12_arch"
    fixed_ckpt = a.repair / "matched_seed20261121_e14_b16/fixed/control.pt"
    fix12_ckpt = a.codefix / "fix12_arch/control.pt"

    df_fixed = keyed_merge(read_predictions(fixed_eval_dir), gold_df)
    df_fix12 = keyed_merge(read_predictions(fix12_eval_dir), gold_df)

    # 1. Independent recount parity check against PAIRED_CODE_FIXES_RESULTS.json
    saved = read_json(a.codefix / "PAIRED_CODE_FIXES_RESULTS.json")
    native_fixed = summarize_cohorts(df_fixed)
    native_fix12 = summarize_cohorts(df_fix12)
    parity = {}
    for arm, rec in [("fixed", native_fixed), ("fix12_arch", native_fix12)]:
        ok = True
        for c in ("held6", "spontaneous", "epadb_test", "saa27"):
            for k in ("tp", "fp", "fn"):
                if rec[c]["strict"][k] != saved["full_eval"][arm][c]["strict"][k]:
                    ok = False
        if rec["timit_core"]["fa_total"] != saved["full_eval"][arm]["timit_core"]["fa_total"]:
            ok = False
        parity[arm] = ok
    if not all(parity.values()):
        raise RuntimeError(f"Independent recount mismatch: {parity}")

    # 2. Calibrate recall-matched thresholds (strictly on Calibrate L2!)
    cal_l2_fix, cal_l1_fix, pol_fix = cal_scores_for_model(cache, cal_rows, fixed_ckpt, "fixed")
    cal_l2_f12, cal_l1_f12, pol_f12 = cal_scores_for_model(cache, cal_rows, fix12_ckpt, "fix12")

    fix_t_pos = cal_l2_fix["t_sc"][np.isin(cal_l2_fix["t_g"], ["S", "O"])]
    fix_g_pos = cal_l2_fix["g_sc"][cal_l2_fix["g_g"] == "I"]
    ref_t_rec = float((fix_t_pos >= pol_fix["tau_t"]).mean())
    ref_g_rec = float((fix_g_pos >= pol_fix["tau_g"]).mean())
    rm_tau_t, rm_tau_g = match_cal_recall(cal_l2_f12, ref_t_rec, ref_g_rec)

    # Calibrate diagnostics at native, common, and recall-matched operating points
    def cal_diag(cl2, cl1, tt, tg):
        t_pos = cl2["t_sc"][np.isin(cl2["t_g"], ["S", "O"])]
        t_neg = cl2["t_sc"][cl2["t_g"] == "C"]
        g_pos = cl2["g_sc"][cl2["g_g"] == "I"]
        g_neg = cl2["g_sc"][cl2["g_g"] == "N"]
        return {
            "tau_t": round(tt, 6),
            "tau_g": round(tg, 6),
            "cal_l2_target_recall": float((t_pos >= tt).mean()),
            "cal_l2_target_fp_rate": float((t_neg >= tt).mean()),
            "cal_l2_gap_recall": float((g_pos >= tg).mean()),
            "cal_l2_gap_fp_rate": float((g_neg >= tg).mean()),
            "cal_l1_target_fa_per_1k": float((cl1["t_sc"] >= tt).mean() * 1000.0),
            "cal_l1_gap_fa_per_1k": float((cl1["g_sc"] >= tg).mean() * 1000.0),
        }

    common_fix12 = summarize_cohorts(
        apply_policy_df(df_fix12, pol_fix["tau_t"], pol_fix["tau_g"])
    )
    rm_fix12 = summarize_cohorts(apply_policy_df(df_fix12, rm_tau_t, rm_tau_g))

    # 3. High-can_lp TP vs FN & Held-6 S_same audit (at native locked calls)
    regime_audit = analyze_can_lp_and_s_same(
        cache, ev_prompts, gold_df, {"fixed": df_fixed, "fix12_arch": df_fix12}
    )

    result = {
        "protocol": "CPU read-only replay over sealed Parquets; thresholds locked strictly on Calibrate",
        "source_sha256": sha_file(__file__),
        "independent_recount_parity": parity,
        "calibrate_operating_points": {
            "fixed_native": cal_diag(cal_l2_fix, cal_l1_fix, pol_fix["tau_t"], pol_fix["tau_g"]),
            "fix12_native": cal_diag(cal_l2_f12, cal_l1_f12, pol_f12["tau_t"], pol_f12["tau_g"]),
            "fix12_common_fixed_tau": cal_diag(cal_l2_f12, cal_l1_f12, pol_fix["tau_t"], pol_fix["tau_g"]),
            "fix12_cal_recall_matched": cal_diag(cal_l2_f12, cal_l1_f12, rm_tau_t, rm_tau_g),
        },
        "eval_operating_points": {
            "fixed_native": native_fixed,
            "fix12_native": native_fix12,
            "fix12_common_fixed_tau": common_fix12,
            "fix12_cal_recall_matched": rm_fix12,
        },
        **regime_audit,
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    save_json(a.output, result)
    print(f"Saved {a.output} ({result['wall_s']}s) | parity={parity}", flush=True)


if __name__ == "__main__":
    main()
