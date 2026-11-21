"""Eval-only verifier & head-to-head comparison for saved `seq_mdd_v2_antishortcut_seed20261121.pt`.

Computes full 5-cohort metrics + TIMIT Corruption Control + paired speaker bootstrap vs R0 & Koel Base
+ GPU latency (P50/P95 ms) and writes `outputs_seq_mdd_v2/SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs_seq_mdd_v2"
CKPT_PATH = HERE / "checkpoints/seq_mdd_v2_antishortcut_seed20261121.pt"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.analyze_and_dissect import (
    oracle_best_f1_sm13,
    paired_speaker_bootstrap,
    pr_roc,
    score_decisions,
)
from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    SAA27_GOLD,
    load_cohorts,
    sha256_file,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    normalize_canonical_phone,
)
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import SeqMDDv2, extract_seq_mdd_v2_features
from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    HELD6_SPEAKERS,
    apply_seq_thresholds,
    collate_batch,
)

TAU_T_LOCKED = 0.29
TAU_G_LOCKED = 0.825


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    koel = load_base_koel(dev)
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = SeqMDDv2(**ck["model_kwargs"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"], strict=True)
    t_mean, t_std, g_mean, g_std = ck["t_mean"], ck["t_std"], ck["g_mean"], ck["g_std"]

    cohorts = load_cohorts()
    saa_rows = []
    for line in SAA27_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ap = Path(r["audio_path"]) if Path(r["audio_path"]).is_absolute() else REPO_ROOT / r["audio_path"]
        saa_rows.append({
            "cohort": "saa27", "uid": r["uid"], "speaker": str(r["speaker_id"]).upper(),
            "audio_path": str(ap),
            "canonical_phones": [normalize_canonical_phone(p) for p in r["canonical_phones"]],
            "gold_targets": list(r["gold_targets"]), "gold_gaps": list(r["gold_gaps"]),
        })
    cohorts["saa27"] = saa_rows

    out: Dict[str, Any] = {
        "checkpoint": str(CKPT_PATH),
        "checkpoint_sha256": sha256_file(CKPT_PATH),
        "best_epoch": int(ck["best_epoch"]),
        "locked_thresholds": {"tau_t": TAU_T_LOCKED, "tau_g": TAU_G_LOCKED},
        "learned_target_layer_weights": [round(float(x), 4) for x in ck["target_layer_weights"]],
        "learned_gap_layer_weights": [round(float(x), 4) for x in ck["gap_layer_weights"]],
        "eval": {},
    }

    for cname in ["held6", "spontaneous", "epadb_test", "saa27", "timit_core"]:
        rows = cohorts[cname]
        ps_l, po_l, gi_l, gt_l, gg_l, sp_t_l, sp_g_l, ms_l = [], [], [], [], [], [], [], []
        for idx_u, r in enumerate(rows):
            wav_np, _ = sf.read(r["audio_path"], dtype="float32")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            feats = extract_seq_mdd_v2_features(koel, torch.from_numpy(wav_np).unsqueeze(0), r["canonical_phones"], dev)
            item = {"uid": r["uid"], "speaker": r["speaker"], "is_native_l1": False,
                    "gold_targets": r["gold_targets"], "gold_gaps": r["gold_gaps"], **feats}
            b = collate_batch([item], t_mean, t_std, g_mean, g_std, dev, with_labels=False)
            with torch.inference_mode():
                o = model(
                    h_vit=b["h_vit"], h_peak=b["h_peak"], lp_win=b["lp_win"], lp_vit=b["lp_vit"],
                    phone_ids=b["phone_ids"], prev_phone_ids=b["prev_phone_ids"], next_phone_ids=b["next_phone_ids"],
                    target_scalars=b["target_scalars"], h_gap=b["h_gap"], lp_gap_mean=b["lp_gap_mean"],
                    lp_gap_max=b["lp_gap_max"], gap_scalars=b["gap_scalars"], lengths=b["lengths"],
                )
                tp = F.softmax(o["target_logits"][0], dim=-1).cpu().numpy()
                gp = F.softmax(o["gap_logits"][0], dim=-1)[:, 1].cpu().numpy()
            torch.cuda.synchronize()
            ms_l.append((time.perf_counter() - t0) * 1000.0)

            ps_l.append(tp[:, 1]); po_l.append(tp[:, 2]); gi_l.append(gp)
            gt_l.extend(r["gold_targets"]); gg_l.extend(r["gold_gaps"])
            sp_t_l.extend([r["speaker"]] * len(r["gold_targets"]))
            sp_g_l.extend([r["speaker"]] * len(r["gold_gaps"]))

        ps, po, gi = np.concatenate(ps_l), np.concatenate(po_l), np.concatenate(gi_l)
        gt, gg = np.array(gt_l), np.array(gg_l)
        sp_t, sp_g = np.array(sp_t_l), np.array(sp_g_l)
        pt, pg = apply_seq_thresholds(ps, po, gi, TAU_T_LOCKED, TAU_G_LOCKED)
        ms_arr = np.array(ms_l)
        lat = {"p50_ms": round(float(np.median(ms_arr)), 1), "p95_ms": round(float(np.percentile(ms_arr, 95)), 1)}

        if cname == "timit_core":
            s, o_c, i_c = int(np.sum(pt == "S")), int(np.sum(pt == "O")), int(np.sum(pg == "I"))
            out["eval"]["timit_core"] = {
                "fa_total": s + o_c + i_c, "fa_S": s, "fa_O": o_c, "fa_I": i_c,
                "fa_per_utt": round((s + o_c + i_c) / len(rows), 2),
                "target_fa_per_1k": round(1000.0 * (s + o_c) / len(gt), 2),
                "gap_fa_per_1k": round(1000.0 * i_c / len(gg), 2),
                "latency": lat,
            }
            print(f"[timit_core] FA/utt={out['eval']['timit_core']['fa_per_utt']} ({out['eval']['timit_core']['target_fa_per_1k']}/1k) | lat={lat}")
            continue

        res = score_decisions(gt, pt, gg, pg)
        or_tt, or_tg, res_or = oracle_best_f1_sm13(gt, ps, po, gg, gi, mode="sum")
        tv, gv = gt != "unknown", gg != "unknown"
        t_ap, t_auc = pr_roc(((gt[tv] == "S") | (gt[tv] == "O")).astype(int), (ps + po)[tv])
        g_ap, g_auc = pr_roc((gg[gv] == "I").astype(int), gi[gv])

        # Paired speaker bootstrap vs old R0 Cal-F1 on held6/spontaneous/epadb_test!
        boot_vs_r0 = None
        old_pq = REPO_ROOT / f"research/FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1/outputs/{cname}_slots.parquet"
        if old_pq.exists():
            df_old = pq.read_table(old_pq).to_pandas()
            dt_o = df_old[df_old["domain"] == "target"]
            dg_o = df_old[df_old["domain"] == "gap"]
            r0_pt = np.where(dt_o["r0_score"].to_numpy() >= 0.3407, np.where(dt_o["r0_p_o"].to_numpy() > dt_o["r0_p_s"].to_numpy(), "O", "S"), "C")
            r0_pg = dg_o["vanilla_call"].to_numpy()
            boot_vs_r0 = paired_speaker_bootstrap(sp_t, gt, pt, r0_pt, sp_g, gg, pg, r0_pg)

        entry = {
            "cal_f1": res, "oracle": {"tau_t": or_tt, "tau_g": or_tg, "metrics": res_or},
            "target_pr_auc": t_ap, "target_roc_auc": t_auc, "gap_pr_auc": g_ap, "gap_roc_auc": g_auc,
            "latency": lat, "bootstrap_vs_r0_old": boot_vs_r0,
        }
        if cname == "spontaneous":
            mt = np.isin(sp_t, list(HELD6_SPEAKERS))
            mg = np.isin(sp_g, list(HELD6_SPEAKERS))
            entry["unseen_held6_slice"] = score_decisions(gt[mt], pt[mt], gg[mg], pg[mg])
        out["eval"][cname] = entry
        print(f"[{cname:11s}] μF1={res['strict']['f1']*100:.2f}% (Or {res_or['strict']['f1']*100:.2f}%) | S={res['S']['f1']*100:.1f} O={res['O']['f1']*100:.1f} I={res['I']['f1']*100:.1f} | tAP={t_ap:.4f} gAP={g_ap:.4f} | lat={lat} | vs_R0={boot_vs_r0}")

    # TIMIT Corruption Control Test (50 unseen native utts, SAME TIMIT mic!)
    rng = np.random.default_rng(42)
    c0, c1 = [], []
    for r in cohorts["timit_core"][:50]:
        wav_np, _ = sf.read(r["audio_path"], dtype="float32")
        can = list(r["canonical_phones"])
        corr_idx = set()
        corrupted = list(can)
        for i in range(1, len(can) - 1, 6):
            corrupted[i] = str(rng.choice([p for p in ARPA_39 if p != can[i]]))
            corr_idx.add(i)
        feats = extract_seq_mdd_v2_features(koel, torch.from_numpy(wav_np).unsqueeze(0), corrupted, dev)
        item = {"uid": r["uid"], "speaker": r["speaker"], "is_native_l1": False,
                "gold_targets": ["C"] * len(corrupted), "gold_gaps": ["N"] * (len(corrupted) + 1), **feats}
        b = collate_batch([item], t_mean, t_std, g_mean, g_std, dev, with_labels=False)
        with torch.inference_mode():
            o = model(
                h_vit=b["h_vit"], h_peak=b["h_peak"], lp_win=b["lp_win"], lp_vit=b["lp_vit"],
                phone_ids=b["phone_ids"], prev_phone_ids=b["prev_phone_ids"], next_phone_ids=b["next_phone_ids"],
                target_scalars=b["target_scalars"], h_gap=b["h_gap"], lp_gap_mean=b["lp_gap_mean"],
                lp_gap_max=b["lp_gap_max"], gap_scalars=b["gap_scalars"], lengths=b["lengths"],
            )
            sc = (F.softmax(o["target_logits"][0], dim=-1)[:, 1:].sum(dim=-1)).cpu().numpy()
        for i in range(len(corrupted)):
            (c1 if i in corr_idx else c0).append(float(sc[i]))
    c0a, c1a = np.array(c0), np.array(c1)
    y = np.concatenate([np.zeros_like(c0a), np.ones_like(c1a)])
    s = np.concatenate([c0a, c1a])
    out["timit_corruption_control"] = {
        "n_utts": 50, "n_clean": int(len(c0a)), "n_corrupt": int(len(c1a)),
        "clean_alert_rate_pct": round(float(100 * (c0a >= TAU_T_LOCKED).mean()), 2),
        "corrupt_recall_pct": round(float(100 * (c1a >= TAU_T_LOCKED).mean()), 2),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
    }
    print("[TIMIT Ctrl]", out["timit_corruption_control"])

    out_path = OUT_DIR / "SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
