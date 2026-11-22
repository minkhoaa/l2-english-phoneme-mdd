"""Read-only CPU audit for ORTHO_IMPROVEMENTS_20261122_v1 (zero GPU, zero Eval gold).

Verifies on real admitted clean_11 Fit + S0_clean Cal arrays:
  1. Lever O1: Frequency-matched TIMIT Fit corruption eliminates Zipfian rare-phone flood.
  2. Lever O2: Fit-only SourcePhonePriorTable quantifies cross-source prior collisions.
  3. Lever O4 + O5: Cal-only Gap Platt calibration & 3-seed Cal ensemble on S0_clean Cal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from .calibration import (
    calibrate_ortho_policy,
    compute_ece,
    fit_gap_platt_cal,
)
from .prior_debias import SourcePhonePriorTable
from .timit_freq_corrupt import (
    audit_timit_phone_rates,
    load_clean_timit_phones,
    rebuild_fit_with_freq_matched_timit,
)

CLEAN11_FIT = Path(
    "research/SYNTH_3DIR_ISO_V1_20261122/run_v2/proposed_custody_clean_baseline/fit.jsonl"
)
S0_CLEAN_DIR = Path(
    "research/BACKBONE_REP_CAMPAIGN_20261122_v1/runs/full_clean11_v1/stage0_s0_clean"
)


def run_cpu_audit() -> Dict[str, Any]:
    with CLEAN11_FIT.open("r", encoding="utf-8") as f:
        fit_rows = [json.loads(line) for line in f]

    # 1. Lever O1 audit on real clean_11 Fit TIMIT rows
    timit_uids = [r["uid"] for r in fit_rows if str(r.get("source", "")).startswith("timit")]
    clean_map = load_clean_timit_phones(timit_uids)
    before_o1 = audit_timit_phone_rates(fit_rows)
    rebuilt_fit, phone_probs = rebuild_fit_with_freq_matched_timit(
        fit_rows, clean_map, seed=20261122
    )
    after_o1 = audit_timit_phone_rates(rebuilt_fit)

    # Verify exact WAV & speaker invariance
    assert len(rebuilt_fit) == len(fit_rows)
    for a, b in zip(fit_rows, rebuilt_fit):
        assert a["uid"] == b["uid"]
        assert a["speaker"] == b["speaker"]
        assert a["audio_sha256"] == b["audio_sha256"]

    # 2. Lever O2 audit on real clean_11 Fit (and after O1)
    prior_before = SourcePhonePriorTable.from_fit_rows(fit_rows, shrinkage_alpha=20.0)
    prior_after_o1 = SourcePhonePriorTable.from_fit_rows(rebuilt_fit, shrinkage_alpha=20.0)

    # 3. Lever O4 & O5 audit strictly on S0_clean Cal arrays (if available; ZERO Eval touch!)
    s0_cal_audit: Dict[str, Any] = {}
    seeds = [20261121, 20261122, 20261123]
    cal_files = [S0_CLEAN_DIR / f"seed_{s}/cal_arrays.npz" for s in seeds]
    if all(p.exists() for p in cal_files):
        arrs = [np.load(p) for p in cal_files]
        l2_g_gold = arrs[0]["l2_g_g"]
        l2_t_gold = arrs[0]["l2_t_g"]
        mt = np.isin(l2_t_gold, ["C", "S", "O"])
        mg = np.isin(l2_g_gold, ["N", "I"])
        y_t = (l2_t_gold[mt] != "C").astype(int)
        y_g = (l2_g_gold[mg] == "I").astype(int)

        # Single-seed Gap Platt calibration check on Cal (using saved Cal gap probs)
        per_seed_gap_cal = {}
        t_sc_list, g_sc_list, l1_t_list, l1_g_list = [], [], [], []
        for s, a in zip(seeds, arrs):
            t_sc_list.append(a["l2_t_sc"])
            g_sc_list.append(a["l2_g_sc"])
            l1_t_list.append(a["l1_t_sc"])
            l1_g_list.append(a["l1_g_sc"])
            comb_g = np.concatenate([a["l2_g_sc"][mg], a["l1_g_sc"]], axis=0)
            comb_y = np.concatenate([l2_g_gold[mg], np.full(len(a["l1_g_sc"]), "N")], axis=0)
            per_seed_gap_cal[str(s)] = fit_gap_platt_cal(
                comb_g, comb_y, is_probs=True, known_pos_weight=8.0
            )

        ens_t = np.mean(t_sc_list, axis=0)
        ens_g = np.mean(g_sc_list, axis=0)
        ens_l1_t = np.mean(l1_t_list, axis=0)
        ens_l1_g = np.mean(l1_g_list, axis=0)

        comb_ens_g = np.concatenate([ens_g[mg], ens_l1_g], axis=0)
        comb_ens_y = np.concatenate([l2_g_gold[mg], np.full(len(ens_l1_g), "N")], axis=0)
        ens_gap_cal = fit_gap_platt_cal(
            comb_ens_g, comb_ens_y, is_probs=True, known_pos_weight=8.0
        )

        # Also check ECE before vs after on L2 Cal gaps
        s0_cal_audit = {
            "n_seeds": len(seeds),
            "single_seed_cal_tAP_mean": round(
                float(np.mean([average_precision_score(y_t, x[mt]) for x in t_sc_list])), 4
            ),
            "ens3_cal_tAP": round(float(average_precision_score(y_t, ens_t[mt])), 4),
            "single_seed_cal_gAP_mean": round(
                float(np.mean([average_precision_score(y_g, x[mg]) for x in g_sc_list])), 4
            ),
            "ens3_cal_gAP": round(float(average_precision_score(y_g, ens_g[mg])), 4),
            "per_seed_gap_platt_cal": per_seed_gap_cal,
            "ens3_gap_platt_cal": ens_gap_cal,
            "l2_cal_gap_ece_raw_ens3": round(compute_ece(y_g, ens_g[mg]), 6),
        }

        # Synthetic 3-class check of calibrate_ortho_policy API on binary-derived probs
        dummy_t_l2 = np.stack([1.0 - ens_t, ens_t, np.zeros_like(ens_t)], axis=-1)
        dummy_t_l1 = np.stack([1.0 - ens_l1_t, ens_l1_t, np.zeros_like(ens_l1_t)], axis=-1)
        pol_raw, _ = calibrate_ortho_policy(
            dummy_t_l2, l2_t_gold, ens_g, l2_g_gold, dummy_t_l1, ens_l1_g,
            is_probs=True, fit_gap_temp=False, ensemble_mode="prob_mean", n_seeds=3,
        )
        pol_cal, _ = calibrate_ortho_policy(
            dummy_t_l2, l2_t_gold, ens_g, l2_g_gold, dummy_t_l1, ens_l1_g,
            is_probs=True, fit_gap_temp=True, ensemble_mode="prob_mean", n_seeds=3,
        )
        s0_cal_audit["gap_tau_raw_vs_platt_cal"] = {
            "raw_tau_g": pol_raw.tau_g,
            "platt_tau_g": pol_cal.tau_g,
            "platt_temp": pol_cal.gap_temp,
            "platt_bias": pol_cal.gap_bias,
        }

    return {
        "o1_timit_before_uniform": {
            k: v for k, v in before_o1.items() if k != "per_phone"
        },
        "o1_timit_after_freq_matched": {
            k: v for k, v in after_o1.items() if k != "per_phone"
        },
        "o2_prior_collisions_before_o1": prior_before.audit_summary(),
        "o2_prior_collisions_after_o1": prior_after_o1.audit_summary(),
        "o4_o5_s0_clean_cal_audit": s0_cal_audit,
    }


def main() -> None:
    report = run_cpu_audit()
    out_p = Path("research/ORTHO_IMPROVEMENTS_20261122_v1/CPU_AUDIT_CLEAN11_FIT_CAL.json")
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved CPU audit to {out_p}")
    print("\n=== O1 TIMIT FIT BEFORE (uniform) vs AFTER (freq-matched) ===")
    b, a = report["o1_timit_before_uniform"], report["o1_timit_after_freq_matched"]
    print(f"  Overall target err: {b['overall_target_err_pct']}% -> {a['overall_target_err_pct']}% | "
          f"S: {b['overall_s_pct']}%->{a['overall_s_pct']}% | O: {b['overall_o_pct']}%->{a['overall_o_pct']}% | I: {b['overall_gap_i_pct']}%->{a['overall_gap_i_pct']}%")
    print(f"  Per-phone err rate std: {b['phone_err_rate_std_pct']}% -> {a['phone_err_rate_std_pct']}% | "
          f"min..max: {b['phone_err_rate_min_pct']}..{b['phone_err_rate_max_pct']}% -> {a['phone_err_rate_min_pct']}..{a['phone_err_rate_max_pct']}%")
    print("  Top 5 worst before:", [(p, d["err_rate_pct"]) for p, d in b["highest_8"][:5]])
    print("  Top 5 worst after :", [(p, d["err_rate_pct"]) for p, d in a["highest_8"][:5]])

    print("\n=== O2 TOP 5 CROSS-SOURCE PRIOR COLLISIONS (clean_11 Fit) ===")
    for row in report["o2_prior_collisions_after_o1"]["top_10_cross_source_prior_collisions"][:5]:
        print(f"  {row['phone']:3s}: L2A={row['l2arctic_err_pct']:5.1f}% vs EpaDB={row['epadb_err_pct']:5.1f}% "
              f"(ratio={row['l2_vs_epadb_ratio']:4.1f}x) | TIMIT={row['timit_err_pct']:4.1f}%")

    if report["o4_o5_s0_clean_cal_audit"]:
        s0 = report["o4_o5_s0_clean_cal_audit"]
        print("\n=== O4 + O5 ON S0_CLEAN CAL (clean_11, ZERO EVAL TOUCH) ===")
        print(f"  tAP: 1-seed {s0['single_seed_cal_tAP_mean']:.4f} -> 3-ens {s0['ens3_cal_tAP']:.4f}")
        print(f"  gAP: 1-seed {s0['single_seed_cal_gAP_mean']:.4f} -> 3-ens {s0['ens3_cal_gAP']:.4f}")
        gp = s0["ens3_gap_platt_cal"]
        print(f"  Gap Platt Cal (ens3): NLL {gp['cal_nll_before']:.5f} -> {gp['cal_nll_after']:.5f} | "
              f"ECE {gp['cal_ece_before']:.5f} -> {gp['cal_ece_after']:.5f} | "
              f"pred_I {100*gp['cal_pred_i_before']:.2f}% -> {100*gp['cal_pred_i_after']:.2f}% (emp {100*gp['cal_emp_i_rate']:.2f}%)")
        print(f"  Gap threshold shift: raw tau_g={s0['gap_tau_raw_vs_platt_cal']['raw_tau_g']} -> "
              f"Platt tau_g={s0['gap_tau_raw_vs_platt_cal']['platt_tau_g']} "
              f"(T={gp['gap_temp']:.3f}, b={gp['gap_bias']:.3f})")


if __name__ == "__main__":
    main()
