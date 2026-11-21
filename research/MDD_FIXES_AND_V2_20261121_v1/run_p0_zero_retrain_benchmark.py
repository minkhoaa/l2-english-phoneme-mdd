"""BEFORE vs AFTER P0 Zero-Retrain Benchmark using EXISTING KEPT checkpoints across all 5 cohorts.

Zero checkpoints are modified or retrained:
  - Koel Base:  OLD (40/75 IPA map) vs P0-Fixed (75/75 IPA map + allophone-aware DP)
  - R0:         OLD features vs P0-Fixed features (same untouched r0_base.pt / fold heads!)
  - SM13:       OLD broken E4 vs P0-Fixed E4 (same untouched seed101 checkpoints!)

Evaluates on: Held-6 (802), Spontaneous (129), EpaDB TEST (1,266), SAA-27 (27), TIMIT Core (192).
Compares directly against the saved BEFORE baseline in FULL_BENCHMARK_AND_DISSECTION.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs_p0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.CONTROLLED_PILOT_AB_20261118_v1 import run_pilot_ab as pilot
from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.analyze_and_dissect import (
    SM13_LOCKED_TAU_G,
    SM13_LOCKED_TAU_T,
    apply_r0_tau,
    apply_sm13_tau,
    oracle_best_f1_1d,
    oracle_best_f1_sm13,
    pr_roc,
    score_decisions,
)
from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    SAA27_GOLD,
    load_cohorts,
    load_r0_heads,
    sha256_file,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import normalize_canonical_phone
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.mdd_head import logits_to_probs_and_scores
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.KOEL_SM13_BENCHMARK_20261120_v1.infer_sm13 import SM13Seed101
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import extract_utterance_base_features_v2
from research.MDD_FIXES_AND_V2_20261121_v1.sm13_e4_fixed import predict_sm13_fixed

BEFORE_JSON = REPO_ROOT / "research/FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1/outputs/FULL_BENCHMARK_AND_DISSECTION.json"


def load_all_5_cohorts() -> Dict[str, List[Dict[str, Any]]]:
    cohorts = load_cohorts()
    saa_rows = []
    for line in SAA27_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ap = Path(r["audio_path"])
        if not ap.is_absolute():
            ap = REPO_ROOT / ap
        saa_rows.append({
            "cohort": "saa27",
            "uid": r["uid"],
            "speaker": str(r["speaker_id"]).upper(),
            "l1": str(r["l1"]),
            "audio_path": str(ap),
            "canonical_phones": [normalize_canonical_phone(p) for p in r["canonical_phones"]],
            "gold_targets": list(r["gold_targets"]),
            "gold_gaps": list(r["gold_gaps"]),
            "gold_targets_alt": list(r["gold_targets"]),
            "gold_gaps_alt": list(r["gold_gaps"]),
        })
    cohorts["saa27"] = saa_rows
    return cohorts


def run_koel_r0_p0_pass(cohorts: Dict[str, List[Dict[str, Any]]], device: torch.device) -> Dict[str, Dict[str, Any]]:
    print("\n[PASS 1/2] Frozen Base Koel + EXISTING R0 checkpoints with P0-Fixed v2 features...")
    koel_model = load_base_koel(device)
    koel_vocab = json.loads((pilot.WEIGHTS_DIR / "vocab.json").read_text(encoding="utf-8"))
    inverse_vocab = {int(idx): tok for tok, idx in koel_vocab.items()}
    r0_heads, spk_to_fold = load_r0_heads(device)

    out: Dict[str, Dict[str, Any]] = {}
    for cname, rows in cohorts.items():
        t0 = time.perf_counter()
        out[cname] = {}
        unmapped_tot = 0
        for idx_u, rec in enumerate(rows):
            wav_np, sr = sf.read(rec["audio_path"], dtype="float32")
            wav_t = torch.from_numpy(wav_np).unsqueeze(0)
            feats = extract_utterance_base_features_v2(
                koel_model, wav_t, rec["canonical_phones"], inverse_vocab, device
            )
            unmapped_tot += len(feats["unmapped_tokens"])

            r0_key = spk_to_fold[rec["speaker"]] if (cname == "spontaneous" and rec["speaker"] in spk_to_fold) else "r0_full_dev18"
            head, sc_mean, sc_std = r0_heads[r0_key]
            win_idx = feats["win_indices"]
            with torch.inference_mode():
                logits = head(
                    torch.from_numpy(feats["h18"][win_idx]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(feats["h24"][win_idx]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(feats["lp"][win_idx]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(feats["win_flags"]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(feats["h18_vit_mean"]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(feats["h24_vit_mean"]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(feats["lp_vit_mean"]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(feats["phone_ids"]).to(device=device, dtype=torch.long),
                    torch.from_numpy(feats["prev_phone_ids"]).to(device=device, dtype=torch.long),
                    torch.from_numpy(feats["next_phone_ids"]).to(device=device, dtype=torch.long),
                    torch.from_numpy((feats["scalars"] - sc_mean) / sc_std).to(device=device, dtype=torch.float32),
                )
                r0_out = logits_to_probs_and_scores(logits)

            out[cname][rec["uid"]] = {
                "greedy_call": feats["a_greedy_call"],
                "gap_call": feats["a_gap_call"],
                "r0_p_s": r0_out["p_s"].cpu().numpy().astype(np.float32),
                "r0_p_o": r0_out["p_o"].cpu().numpy().astype(np.float32),
                "r0_score": r0_out["score"].cpu().numpy().astype(np.float32),
            }
            if (idx_u + 1) % 300 == 0 or (idx_u + 1) == len(rows):
                print(f"  [Koel+R0 P0] {cname:12s} {idx_u + 1:4d}/{len(rows)} ({time.perf_counter() - t0:.1f}s) unmapped={unmapped_tot}")

    del koel_model, r0_heads
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run_sm13_p0_pass(cohorts: Dict[str, List[Dict[str, Any]]], device: str) -> Dict[str, Dict[str, Any]]:
    print("\n[PASS 2/2] EXISTING SM13 seed101 checkpoints with P0-Fixed E4...")
    sm13 = SM13Seed101(device=device)
    out: Dict[str, Dict[str, Any]] = {}
    for cname, rows in cohorts.items():
        t0 = time.perf_counter()
        out[cname] = {}
        for idx_u, rec in enumerate(rows):
            wav_np, _ = sf.read(rec["audio_path"], dtype="float32")
            out[cname][rec["uid"]] = predict_sm13_fixed(sm13, wav_np, rec["canonical_phones"], top_k=6, e4_weight=0.5)
            if (idx_u + 1) % 300 == 0 or (idx_u + 1) == len(rows):
                print(f"  [SM13 P0]    {cname:12s} {idx_u + 1:4d}/{len(rows)} ({time.perf_counter() - t0:.1f}s)")

    del sm13
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = json.loads(BEFORE_JSON.read_text(encoding="utf-8"))
    th = before["transferred_dev18_thresholds"]
    tau_r0_lk = th["r0_locked_mean"]
    tau_r0_cal = th["r0_cal_f1_mean"]
    tau_sm_cal_t = th["sm13_cal_f1_tau_t_mean"]
    tau_sm_cal_g = th["sm13_cal_f1_tau_g_mean"]

    t_all = time.perf_counter()
    cohorts = load_all_5_cohorts()
    koel_p0 = run_koel_r0_p0_pass(cohorts, torch.device(args.device))
    sm13_p0 = run_sm13_p0_pass(cohorts, args.device)

    summary: Dict[str, Any] = {"elapsed_s": round(time.perf_counter() - t_all, 1), "cohorts": {}}

    print("\n" + "=" * 122)
    print("BEFORE vs AFTER P0 ZERO-RETRAIN FIXES (EXISTING KEPT CHECKPOINTS — 0 WEIGHTS MODIFIED!)")
    print("=" * 122)

    for cname in ["held6", "spontaneous", "epadb_test", "saa27", "timit_core"]:
        rows = cohorts[cname]
        gt_l, gg_l, ph_l = [], [], []
        vt_l, vg_l = [], []
        r0_ps_l, r0_po_l, r0_sc_l = [], [], []
        sm_ps_l, sm_po_l, sm_e4_ps_l, sm_e4_po_l, sm_g_l = [], [], [], [], []
        for r in rows:
            uid = r["uid"]
            kr = koel_p0[cname][uid]
            sr = sm13_p0[cname][uid]
            gt_l.extend(r["gold_targets"])
            gg_l.extend(r["gold_gaps"])
            ph_l.extend(r["canonical_phones"])
            vt_l.extend(kr["greedy_call"])
            vg_l.extend(kr["gap_call"])
            r0_ps_l.append(kr["r0_p_s"])
            r0_po_l.append(kr["r0_p_o"])
            r0_sc_l.append(kr["r0_score"])
            sm_ps_l.append(sr["fused_fixed"][:, 1])
            sm_po_l.append(sr["fused_fixed"][:, 2])
            sm_e4_ps_l.append(sr["e4_fixed"][:, 1])
            sm_e4_po_l.append(sr["e4_fixed"][:, 2])
            sm_g_l.append(sr["gap"])

        gt = np.array(gt_l)
        gg = np.array(gg_l)
        ph = np.array(ph_l)
        vt = np.array(vt_l)
        vg = np.array(vg_l)
        r0_ps = np.concatenate(r0_ps_l)
        r0_po = np.concatenate(r0_po_l)
        r0_sc = np.concatenate(r0_sc_l)
        sm_ps = np.concatenate(sm_ps_l)
        sm_po = np.concatenate(sm_po_l)
        sm_e4_ps = np.concatenate(sm_e4_ps_l)
        sm_e4_po = np.concatenate(sm_e4_po_l)
        sm_g = np.concatenate(sm_g_l)

        # Save Parquet for reproducibility
        pq_path = OUT_DIR / f"{cname}_p0_slots.parquet"
        pq.write_table(
            pa.table({
                "gold_target": gt, "phone": ph, "vanilla_p0": vt,
                "r0_p_s": r0_ps, "r0_p_o": r0_po, "r0_score": r0_sc,
                "sm13_p0_p_s": sm_ps, "sm13_p0_p_o": sm_po,
                "sm13_e4_p0_p_s": sm_e4_ps, "sm13_e4_p0_p_o": sm_e4_po,
            }),
            pq_path,
            compression="zstd",
        )

        if cname == "timit_core":
            # Native L1 FA comparison!
            r0_lk_t = apply_r0_tau(r0_ps, r0_po, r0_sc, tau_r0_lk)
            r0_cal_t = apply_r0_tau(r0_ps, r0_po, r0_sc, tau_r0_cal)
            sm_lk_t, sm_lk_g = apply_sm13_tau(sm_ps, sm_po, sm_g, SM13_LOCKED_TAU_T, SM13_LOCKED_TAU_G)
            sm_cal_t, sm_cal_g = apply_sm13_tau(sm_ps, sm_po, sm_g, tau_sm_cal_t, tau_sm_cal_g, mode="sum")

            def fa_stats(pt, pg):
                s = int(np.sum(pt == "S"))
                o = int(np.sum(pt == "O"))
                i = int(np.sum(pg == "I"))
                return {"fa_total": s + o + i, "fa_S": s, "fa_O": o, "fa_I": i,
                        "fa_per_utt": round((s + o + i) / len(rows), 2),
                        "target_fa_per_1k": round(1000.0 * (s + o) / len(gt), 2)}

            tim_p0 = {
                "vanilla_p0": fa_stats(vt, vg),
                "r0_locked_p0": fa_stats(r0_lk_t, vg),
                "r0_cal_p0": fa_stats(r0_cal_t, vg),
                "sm13_locked_p0": fa_stats(sm_lk_t, sm_lk_g),
                "sm13_cal_p0": fa_stats(sm_cal_t, sm_cal_g),
            }
            summary["cohorts"]["timit_core"] = tim_p0
            b_tim = before["timit_native_fa_control"]
            print(f"\n--- TIMIT CORE NATIVE L1 FALSE-ALARM CONTROL (192 utts, 6,374 phones) ---")
            print(f"  Vanilla   : OLD {b_tim['vanilla']['fa_per_utt']:.2f} FA/u ({b_tim['vanilla']['target_fa_per_1k']:.1f}/1k)  -->  P0 {tim_p0['vanilla_p0']['fa_per_utt']:.2f} FA/u ({tim_p0['vanilla_p0']['target_fa_per_1k']:.1f}/1k)")
            print(f"  R0 Locked : OLD {b_tim['r0_locked']['fa_per_utt']:.2f} FA/u ({b_tim['r0_locked']['target_fa_per_1k']:.1f}/1k)  -->  P0 {tim_p0['r0_locked_p0']['fa_per_utt']:.2f} FA/u ({tim_p0['r0_locked_p0']['target_fa_per_1k']:.1f}/1k)")
            print(f"  R0 Cal-F1 : OLD {b_tim['r0_cal_transfer']['fa_per_utt']:.2f} FA/u ({b_tim['r0_cal_transfer']['target_fa_per_1k']:.1f}/1k)  -->  P0 {tim_p0['r0_cal_p0']['fa_per_utt']:.2f} FA/u ({tim_p0['r0_cal_p0']['target_fa_per_1k']:.1f}/1k)")
            print(f"  SM13 Cal  : OLD {b_tim['sm13_cal_transfer']['fa_per_utt']:.2f} FA/u ({b_tim['sm13_cal_transfer']['target_fa_per_1k']:.1f}/1k)  -->  P0 {tim_p0['sm13_cal_p0']['fa_per_utt']:.2f} FA/u ({tim_p0['sm13_cal_p0']['target_fa_per_1k']:.1f}/1k)")
            continue

        # L2 cohorts
        res_van = score_decisions(gt, vt, gg, vg)
        r0_cal_t = apply_r0_tau(r0_ps, r0_po, r0_sc, tau_r0_cal)
        res_r0_cal = score_decisions(gt, r0_cal_t, gg, vg)
        _, res_r0_or = oracle_best_f1_1d(gt, r0_ps, r0_po, r0_sc, gg, vg)

        sm_lk_t, sm_lk_g = apply_sm13_tau(sm_ps, sm_po, sm_g, SM13_LOCKED_TAU_T, SM13_LOCKED_TAU_G, mode="max")
        sm_cal_t, sm_cal_g = apply_sm13_tau(sm_ps, sm_po, sm_g, tau_sm_cal_t, tau_sm_cal_g, mode="sum")
        res_sm_lk = score_decisions(gt, sm_lk_t, gg, sm_lk_g)
        res_sm_cal = score_decisions(gt, sm_cal_t, gg, sm_cal_g)
        _, _, res_sm_or = oracle_best_f1_sm13(gt, sm_ps, sm_po, gg, sm_g, mode="sum")

        tv = gt != "unknown"
        yt = ((gt[tv] == "S") | (gt[tv] == "O")).astype(int)
        r0_ap, _ = pr_roc(yt, r0_sc[tv])
        sm_ap, _ = pr_roc(yt, (sm_ps + sm_po)[tv])
        e4_ap, _ = pr_roc(yt, (sm_e4_ps + sm_e4_po)[tv])

        # Check K and P precision on EpaDB specifically!
        extra = {}
        if cname == "epadb_test":
            for target_ph in ["K", "P", "AE"]:
                m = (ph == target_ph) & tv
                ge = (gt[m] == "S") | (gt[m] == "O")
                re = np.isin(r0_cal_t[m], ["S", "O"])
                tp = int((ge & re).sum()); fp = int(((~ge) & re).sum()); fn = int((ge & ~re).sum())
                extra[target_ph] = {"tp": tp, "fp": fp, "fn": fn, "p": round(100 * tp / max(1, tp + fp), 1), "r": round(100 * tp / max(1, tp + fn), 1)}

        b_c = before["cohorts"][cname]
        b_r0_key = "r0_cal_saa18" if cname == "saa27" else "r0_cal_transfer"
        b_sm_key = "sm13_cal_saa18" if cname == "saa27" else "sm13_cal_transfer"

        summary["cohorts"][cname] = {
            "vanilla_p0": res_van, "r0_cal_p0": res_r0_cal, "r0_oracle_p0": res_r0_or,
            "sm13_locked_p0": res_sm_lk, "sm13_cal_p0": res_sm_cal, "sm13_oracle_p0": res_sm_or,
            "r0_ap_p0": r0_ap, "sm13_ap_p0": sm_ap, "e4_ap_p0": e4_ap, "extra": extra,
        }

        print(f"\n--- {cname.upper()} ---")
        print(f"  Koel Base (Van) : OLD {b_c['vanilla']['strict']['f1']*100:5.2f}%  -->  P0 {res_van['strict']['f1']*100:5.2f}%  (Δ={100*(res_van['strict']['f1']-b_c['vanilla']['strict']['f1']):+5.2f} pp) | ?_cnt={int(np.sum(vt=='?'))}")
        print(f"  R0 (Cal-F1)     : OLD {b_c[b_r0_key]['strict']['f1']*100:5.2f}% (AP {b_c['threshold_free']['r0_target_pr_auc']:.4f})  -->  P0 {res_r0_cal['strict']['f1']*100:5.2f}% (AP {r0_ap:.4f}, Or {res_r0_or['strict']['f1']*100:5.2f}%)  (Δ={100*(res_r0_cal['strict']['f1']-b_c[b_r0_key]['strict']['f1']):+5.2f} pp)")
        print(f"  SM13 (Locked)   : OLD {b_c['sm13_locked']['strict']['f1']*100:5.2f}% (S={b_c['sm13_locked']['S']['f1']*100:4.1f}%)  -->  P0 {res_sm_lk['strict']['f1']*100:5.2f}% (S={res_sm_lk['S']['f1']*100:4.1f}%)")
        print(f"  SM13 (Cal / Or) : OLD {b_c[b_sm_key]['strict']['f1']*100:5.2f}% / {b_c['sm13_oracle']['metrics']['strict']['f1']*100:5.2f}% (S->O={b_c[b_sm_key]['s_to_o']})  -->  P0 {res_sm_cal['strict']['f1']*100:5.2f}% / {res_sm_or['strict']['f1']*100:5.2f}% (S->O={res_sm_cal['s_to_o']}, AP {b_c['threshold_free']['sm13_fused_max_target_pr_auc']:.4f}->{sm_ap:.4f}, E4_AP={e4_ap:.4f})")
        if extra:
            print(f"  EpaDB K/P/AE P0 : {extra}")

    out_json = OUT_DIR / "P0_BEFORE_AFTER_RESULTS.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved P0 results: {out_json} (elapsed {summary['elapsed_s']}s)")


if __name__ == "__main__":
    main()
