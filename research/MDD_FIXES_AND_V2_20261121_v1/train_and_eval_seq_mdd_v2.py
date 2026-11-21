"""Train and evaluate P1 `SeqMDDv2` on Multi-Corpus L2 + L1 Native Anchor (keeping ALL old checkpoints untouched!).

Strict zero-leakage speaker-disjoint splits:
  - FIT (Train):
      * L2-ARCTIC DEV-18 (15 speakers, excluding ERMS/SVBI/YBAA)
      * EpaDB TRAIN (26 speakers, excluding 4 held-out EpaDB Cal speakers)
      * TIMIT TRAIN (1,000 utts from 400 native US L1 speakers, all-C/all-N anchor)
  - CALIBRATE (Speaker-disjoint model selection & threshold locking ONLY):
      * L2-ARCTIC DEV-18 Cal (3 speakers: ERMS, SVBI, YBAA)
      * EpaDB TRAIN Cal (4 speakers)
      * TIMIT TRAIN Cal (120 utts from 60 held-out native US L1 speakers)
  - EVAL (100% UNSEEN SPEAKERS & COHORTS — NEVER TOUCHED DURING FIT OR CALIBRATE):
      * L2-ARCTIC Held-6 (802 utts, 6 unseen speakers)
      * L2-ARCTIC Spontaneous (129 utts; reports strictly-unseen 24-utt Held-6 slice AND full 129)
      * EpaDB TEST (1,266 utts, 20 unseen Spanish L1 speakers)
      * SAA-27 / Stella (27 unseen speakers, long 22-49s passage)
      * TIMIT Core (192 utts, 24 unseen native US L1 speakers — Native L1 False-Alarm Control)

Saves new checkpoint to `checkpoints/seq_mdd_v2_seed20261121.pt` (never touches R0/SM13/Koel checkpoints).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs_seq_mdd_v2"
CKPT_DIR = HERE / "checkpoints"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.analyze_and_dissect import (
    oracle_best_f1_sm13,
    pr_roc,
    score_decisions,
)
from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    DEV18_GOLD,
    EPADB_REALIZED_MAP,
    INTAKE_MANIFEST,
    IPA_TO_ARPA39,
    SAA27_GOLD,
    load_cohorts,
    sha256_file,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    PHONE_TO_ID,
    normalize_canonical_phone,
)
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    CONTINUOUS_SCALAR_NAMES,
    GAP_SCALAR_NAMES,
    NUM_REALIZED_CLASSES,
    SeqMDDv2,
    compute_multitask_mdd_loss,
    extract_seq_mdd_v2_features,
)

DEV18_CAL_SPEAKERS = frozenset({"ERMS", "SVBI", "YBAA"})
HELD6_SPEAKERS = frozenset({"ABA", "ASI", "BWC", "EBVS", "HQTV", "LXC"})
SEED = 20261121
TARGET_LABEL_TO_ID = {"C": 0, "S": 1, "O": 2}
GAP_LABEL_TO_ID = {"N": 0, "I": 1}


def dp_align_csoi_realized(
    canonical: Sequence[str],
    realized: Sequence[str],
    *,
    allow_flap_td: bool = False,
    ambig_tokens: set[str] | None = None,
) -> Tuple[List[str], List[str], List[int]]:
    """Aligned (gold_targets, gold_gaps, realized_phone_ids) where 0..38=ARPA, 39=<DEL>, -100=ignore."""
    ambig = ambig_tokens or set()
    n, m = len(canonical), len(realized)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bp = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        dp[i, 0] = i
        bp[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = j
        bp[0, j] = 2

    for i in range(1, n + 1):
        ci = canonical[i - 1]
        for j in range(1, m + 1):
            rj = realized[j - 1]
            is_match = (ci == rj) or (allow_flap_td and rj == "<FLAP>" and ci in {"T", "D"})
            cost_ms = dp[i - 1, j - 1] + (0 if is_match else 1)
            cost_d = dp[i - 1, j] + 1
            cost_i = dp[i, j - 1] + 1
            best, op = cost_ms, 0
            if cost_d < best:
                best, op = cost_d, 1
            if cost_i < best:
                best, op = cost_i, 2
            dp[i, j], bp[i, j] = best, op

    targets = ["O"] * n
    gaps = ["N"] * (n + 1)
    real_ids = [-100] * n
    i, j = n, m
    while i > 0 or j > 0:
        op = int(bp[i, j])
        if op == 0:
            ci, rj = canonical[i - 1], realized[j - 1]
            if rj in ambig:
                targets[i - 1] = "unknown"
                real_ids[i - 1] = -100
            elif ci == rj or (allow_flap_td and rj == "<FLAP>" and ci in {"T", "D"}):
                targets[i - 1] = "C"
                real_ids[i - 1] = PHONE_TO_ID.get(ci, -100)
            else:
                targets[i - 1] = "S"
                real_ids[i - 1] = PHONE_TO_ID.get(rj, -100)
            i -= 1
            j -= 1
        elif op == 1:
            targets[i - 1] = "O"
            real_ids[i - 1] = 39  # <DEL>
            i -= 1
        else:
            rj = realized[j - 1]
            if rj in ambig:
                if gaps[i] == "N":
                    gaps[i] = "unknown"
            else:
                gaps[i] = "I"
            j -= 1
    return targets, gaps, real_ids


def build_datasets() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Build strictly speaker-disjoint Fit, Calibrate, and Eval manifests."""
    fit_rows: List[Dict[str, Any]] = []
    cal_rows: List[Dict[str, Any]] = []

    # 1. L2-ARCTIC DEV-18 (15 Fit speakers + 3 Cal speakers)
    for line in DEV18_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        spk = str(r["speaker_id"]).upper()
        ap = Path(r["audio_path"])
        if not ap.is_absolute():
            ap = REPO_ROOT / ap
        can = [normalize_canonical_phone(p) for p in r["canonical_phones"]]
        real_raw = [
            normalize_canonical_phone(p) if p not in {"", "SIL", "sp", "spn", "sil"} else ""
            for p in r["realized_phones"]
        ]
        real_clean = [p for p in real_raw if p]
        _, _, real_ids = dp_align_csoi_realized(can, real_clean)
        # Keep exact archived gold_targets and gold_gaps from DEV-18!
        gt = list(r["gold_targets"])
        gg = list(r["gold_gaps"])
        # Ensure real_ids respects gold O -> 39 and gold C -> canonical
        for idx_p, g_lbl in enumerate(gt):
            if g_lbl == "O":
                real_ids[idx_p] = 39
            elif g_lbl == "C":
                real_ids[idx_p] = PHONE_TO_ID[can[idx_p]]
        rec = {
            "source": "l2arctic_dev18",
            "uid": r["uid"],
            "speaker": spk,
            "audio_path": str(ap),
            "canonical_phones": can,
            "gold_targets": gt,
            "gold_gaps": gg,
            "realized_ids": real_ids,
            "is_native_l1": False,
        }
        if spk in DEV18_CAL_SPEAKERS:
            cal_rows.append(rec)
        else:
            fit_rows.append(rec)

    # 2. EpaDB TRAIN (30 speakers -> 26 Fit + 4 Cal, 100% disjoint from EpaDB TEST!)
    epadb_tr_by_spk: Dict[str, List[Dict[str, Any]]] = {}
    timit_tr_by_spk: Dict[str, List[Dict[str, Any]]] = {}
    with INTAKE_MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["corpus"] == "EpaDB" and r.get("split") == "TRAIN":
                epadb_tr_by_spk.setdefault(str(r["speaker_id"]), []).append(r)
            elif r["corpus"] == "TIMIT" and r.get("split") == "TRAIN":
                timit_tr_by_spk.setdefault(str(r["speaker_id"]).upper(), []).append(r)

    epadb_spks = sorted(epadb_tr_by_spk.keys())
    epadb_cal_spks = set(epadb_spks[-4:])  # 4 held-out EpaDB TRAIN speakers for Cal
    for spk in epadb_spks:
        for r in epadb_tr_by_spk[spk]:
            can = [IPA_TO_ARPA39[unicodedata.normalize("NFC", t)] for t in r["canonical_ipa_tokens"]]
            real_strict = [EPADB_REALIZED_MAP[unicodedata.normalize("NFC", t)] for t in r["realized_ipa_tokens"]]
            gt, gg, real_ids = dp_align_csoi_realized(
                can, real_strict, allow_flap_td=True, ambig_tokens={"<AMBIG_TH_H>", "<AMBIG_S_H>"}
            )
            rec = {
                "source": "epadb_train",
                "uid": r["uid"],
                "speaker": spk,
                "audio_path": str(REPO_ROOT / r["audio_path"]),
                "canonical_phones": can,
                "gold_targets": gt,
                "gold_gaps": gg,
                "realized_ids": real_ids,
                "is_native_l1": False,
            }
            if spk in epadb_cal_spks:
                cal_rows.append(rec)
            else:
                fit_rows.append(rec)

    # 3. TIMIT TRAIN L1 Native Anchor (462 native speakers -> 400 Fit [1,000 utts] + 60 Cal [120 utts])
    # CRITICAL ANTI-SHORTCUT DESIGN:
    #   On TIMIT FIT utterances, inject ~10% canonical-phone substitution swaps (label S), ~3% bogus
    #   extra canonical phones (label O), and ~3% canonical deletions (label I) inside the EXACT SAME
    #   TIMIT native recording! Makes any TIMIT mic/channel shortcut mathematically impossible while
    #   preserving ~87% real native C phones per utterance.
    timit_spks = sorted(timit_tr_by_spk.keys())
    rng = np.random.default_rng(SEED)
    timit_perm = rng.permutation(timit_spks).tolist()
    timit_fit_spks = timit_perm[:400]
    timit_cal_spks = timit_perm[400:460]

    for spk_list, target_list, per_spk_cap, max_total, apply_corruption in [
        (timit_fit_spks, fit_rows, 3, 1000, True),
        (timit_cal_spks, cal_rows, 2, 120, False),
    ]:
        added = 0
        for spk in spk_list:
            utts = sorted(timit_tr_by_spk[spk], key=lambda x: x["uid"])
            for r in utts[:per_spk_cap]:
                if added >= max_total:
                    break
                true_phones = [IPA_TO_ARPA39[unicodedata.normalize("NFC", t)] for t in r["realized_ipa_tokens"]]
                if len(true_phones) < 6:
                    continue
                if apply_corruption:
                    can_corr: List[str] = []
                    gt_corr: List[str] = []
                    real_corr: List[int] = []
                    gg_corr: List[str] = ["N"]
                    i_p = 0
                    while i_p < len(true_phones):
                        u = float(rng.random())
                        tp = true_phones[i_p]
                        if u < 0.03 and 0 < i_p < len(true_phones) - 1 and len(can_corr) > 0:
                            gg_corr[-1] = "I"
                            i_p += 1
                            continue
                        if u < 0.06:
                            bogus = str(rng.choice([p for p in ARPA_39 if p != tp]))
                            can_corr.append(bogus)
                            gt_corr.append("O")
                            real_corr.append(39)
                            gg_corr.append("N")
                        if float(rng.random()) < 0.10:
                            wrong = str(rng.choice([p for p in ARPA_39 if p != tp]))
                            can_corr.append(wrong)
                            gt_corr.append("S")
                            real_corr.append(PHONE_TO_ID[tp])
                            gg_corr.append("N")
                        else:
                            can_corr.append(tp)
                            gt_corr.append("C")
                            real_corr.append(PHONE_TO_ID[tp])
                            gg_corr.append("N")
                        i_p += 1
                    target_list.append({
                        "source": "timit_train_l1_antishortcut",
                        "uid": r["uid"],
                        "speaker": spk,
                        "audio_path": str(REPO_ROOT / r["audio_path"]),
                        "canonical_phones": can_corr,
                        "gold_targets": gt_corr,
                        "gold_gaps": gg_corr,
                        "realized_ids": real_corr,
                        "is_native_l1": False,
                    })
                else:
                    target_list.append({
                        "source": "timit_train_l1",
                        "uid": r["uid"],
                        "speaker": spk,
                        "audio_path": str(REPO_ROOT / r["audio_path"]),
                        "canonical_phones": true_phones,
                        "gold_targets": ["C"] * len(true_phones),
                        "gold_gaps": ["N"] * (len(true_phones) + 1),
                        "realized_ids": [PHONE_TO_ID[p] for p in true_phones],
                        "is_native_l1": True,
                    })
                added += 1
            if added >= max_total:
                break

    # 4. Held-out Eval cohorts (100% unseen speakers!)
    eval_cohorts = load_cohorts()
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
        })
    eval_cohorts["saa27"] = saa_rows

    # Leakage assertions!
    fit_spks = {r["speaker"] for r in fit_rows}
    cal_spks = {r["speaker"] for r in cal_rows}
    assert len(fit_spks & cal_spks) == 0, f"Fit/Cal speaker leakage: {fit_spks & cal_spks}"
    train_all_spks = fit_spks | cal_spks
    for cname in ["held6", "epadb_test", "saa27", "timit_core"]:
        ev_spks = {r["speaker"] for r in eval_cohorts[cname]}
        assert len(train_all_spks & ev_spks) == 0, f"Leakage into {cname}: {train_all_spks & ev_spks}"

    return fit_rows, cal_rows, eval_cohorts


def extract_feature_list(
    koel_model: torch.nn.Module,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    label: str,
) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    items: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows):
        wav_np, sr = sf.read(r["audio_path"], dtype="float32")
        if sr != 16000 or wav_np.ndim != 1:
            raise ValueError(f"Bad wav {r['uid']}")
        feats = extract_seq_mdd_v2_features(koel_model, torch.from_numpy(wav_np).unsqueeze(0), r["canonical_phones"], device)
        item: Dict[str, Any] = {
            "uid": r["uid"],
            "speaker": r["speaker"],
            "source": r.get("source", r.get("cohort", label)),
            "is_native_l1": bool(r.get("is_native_l1", False)),
            "gold_targets": r["gold_targets"],
            "gold_gaps": r["gold_gaps"],
            **feats,
        }
        if "realized_ids" in r:
            item["realized_ids"] = np.array(r["realized_ids"], dtype=np.int64)
        items.append(item)
        if (idx + 1) % 400 == 0 or (idx + 1) == len(rows):
            print(f"  [Extract {label:12s}] {idx + 1:4d}/{len(rows)} ({time.perf_counter() - t0:.1f}s)")
    return items


def collate_batch(
    batch: Sequence[Dict[str, Any]],
    t_mean: np.ndarray,
    t_std: np.ndarray,
    g_mean: np.ndarray,
    g_std: np.ndarray,
    device: torch.device,
    with_labels: bool = True,
) -> Dict[str, torch.Tensor]:
    B = len(batch)
    lengths = [int(x["h_vit"].shape[0]) for x in batch]
    max_L = max(lengths)
    n_layers = int(batch[0]["h_vit"].shape[1])
    h_dim = int(batch[0]["h_vit"].shape[2])
    W = int(batch[0]["lp_win"].shape[1])
    V = int(batch[0]["lp_win"].shape[2])

    h_vit = torch.zeros((B, max_L, n_layers, h_dim), dtype=torch.float32)
    h_peak = torch.zeros((B, max_L, n_layers, h_dim), dtype=torch.float32)
    lp_win = torch.zeros((B, max_L, W, V), dtype=torch.float32)
    lp_vit = torch.zeros((B, max_L, V), dtype=torch.float32)
    phone_ids = torch.zeros((B, max_L), dtype=torch.long)
    prev_ids = torch.zeros((B, max_L), dtype=torch.long)
    next_ids = torch.zeros((B, max_L), dtype=torch.long)
    t_sc = torch.zeros((B, max_L, len(CONTINUOUS_SCALAR_NAMES)), dtype=torch.float32)

    h_gap = torch.zeros((B, max_L + 1, n_layers, h_dim), dtype=torch.float32)
    lp_gm = torch.zeros((B, max_L + 1, V), dtype=torch.float32)
    lp_gx = torch.zeros((B, max_L + 1, V), dtype=torch.float32)
    g_sc = torch.zeros((B, max_L + 1, len(GAP_SCALAR_NAMES)), dtype=torch.float32)

    y_t = torch.full((B, max_L), -100, dtype=torch.long)
    y_g = torch.full((B, max_L + 1), -100, dtype=torch.long)
    y_r = torch.full((B, max_L), -100, dtype=torch.long)
    is_l1 = torch.zeros((B,), dtype=torch.bool)

    for b, x in enumerate(batch):
        L = lengths[b]
        h_vit[b, :L] = torch.from_numpy(x["h_vit"].astype(np.float32))
        h_peak[b, :L] = torch.from_numpy(x["h_peak"].astype(np.float32))
        lp_win[b, :L] = torch.from_numpy(x["lp_win"].astype(np.float32))
        lp_vit[b, :L] = torch.from_numpy(x["lp_vit"].astype(np.float32))
        phone_ids[b, :L] = torch.from_numpy(x["phone_ids"])
        prev_ids[b, :L] = torch.from_numpy(x["prev_phone_ids"])
        next_ids[b, :L] = torch.from_numpy(x["next_phone_ids"])
        t_sc[b, :L] = torch.from_numpy((x["target_scalars"] - t_mean) / t_std)

        h_gap[b, : L + 1] = torch.from_numpy(x["h_gap"].astype(np.float32))
        lp_gm[b, : L + 1] = torch.from_numpy(x["lp_gap_mean"].astype(np.float32))
        lp_gx[b, : L + 1] = torch.from_numpy(x["lp_gap_max"].astype(np.float32))
        g_sc[b, : L + 1] = torch.from_numpy((x["gap_scalars"] - g_mean) / g_std)
        is_l1[b] = x["is_native_l1"]

        if with_labels:
            y_t[b, :L] = torch.tensor([TARGET_LABEL_TO_ID.get(g, -100) for g in x["gold_targets"]], dtype=torch.long)
            y_g[b, : L + 1] = torch.tensor([GAP_LABEL_TO_ID.get(g, -100) for g in x["gold_gaps"]], dtype=torch.long)
            if "realized_ids" in x:
                y_r[b, :L] = torch.from_numpy(x["realized_ids"])

    out = {
        "h_vit": h_vit.to(device), "h_peak": h_peak.to(device),
        "lp_win": lp_win.to(device), "lp_vit": lp_vit.to(device),
        "phone_ids": phone_ids.to(device), "prev_phone_ids": prev_ids.to(device), "next_phone_ids": next_ids.to(device),
        "target_scalars": t_sc.to(device),
        "h_gap": h_gap.to(device), "lp_gap_mean": lp_gm.to(device), "lp_gap_max": lp_gx.to(device),
        "gap_scalars": g_sc.to(device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }
    if with_labels:
        out.update({"y_t": y_t.to(device), "y_g": y_g.to(device), "y_r": y_r.to(device), "is_l1": is_l1.to(device)})
    return out


@torch.inference_mode()
def predict_items(
    model: SeqMDDv2,
    items: Sequence[Dict[str, Any]],
    t_mean: np.ndarray,
    t_std: np.ndarray,
    g_mean: np.ndarray,
    g_std: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    model.eval()
    ps_l, po_l, gi_l, gt_l, gg_l, sp_t_l, sp_g_l = [], [], [], [], [], [], []
    for start in range(0, len(items), batch_size):
        sub = items[start : start + batch_size]
        b = collate_batch(sub, t_mean, t_std, g_mean, g_std, device, with_labels=False)
        out = model(
            h_vit=b["h_vit"], h_peak=b["h_peak"], lp_win=b["lp_win"], lp_vit=b["lp_vit"],
            phone_ids=b["phone_ids"], prev_phone_ids=b["prev_phone_ids"], next_phone_ids=b["next_phone_ids"],
            target_scalars=b["target_scalars"], h_gap=b["h_gap"], lp_gap_mean=b["lp_gap_mean"],
            lp_gap_max=b["lp_gap_max"], gap_scalars=b["gap_scalars"], lengths=b["lengths"],
        )
        tp = F.softmax(out["target_logits"], dim=-1).cpu().numpy()
        gp = F.softmax(out["gap_logits"], dim=-1)[..., 1].cpu().numpy()
        for bi, x in enumerate(sub):
            L = len(x["gold_targets"])
            ps_l.append(tp[bi, :L, 1])
            po_l.append(tp[bi, :L, 2])
            gi_l.append(gp[bi, : L + 1])
            gt_l.extend(x["gold_targets"])
            gg_l.extend(x["gold_gaps"])
            sp_t_l.extend([x["speaker"]] * L)
            sp_g_l.extend([x["speaker"]] * (L + 1))

    return {
        "p_s": np.concatenate(ps_l).astype(np.float32),
        "p_o": np.concatenate(po_l).astype(np.float32),
        "gap_i": np.concatenate(gi_l).astype(np.float32),
        "gold_t": np.array(gt_l),
        "gold_g": np.array(gg_l),
        "spk_t": np.array(sp_t_l),
        "spk_g": np.array(sp_g_l),
    }


def apply_seq_thresholds(
    p_s: np.ndarray, p_o: np.ndarray, gap_i: np.ndarray, tau_t: float, tau_g: float
) -> Tuple[np.ndarray, np.ndarray]:
    score = p_s + p_o
    pt = np.where(score >= tau_t, np.where(p_o > p_s, "O", "S"), "C")
    pg = np.where(gap_i >= tau_g, "I", "N")
    return pt, pg


def calibrate_dual_policies(
    cal_l2_pred: Dict[str, np.ndarray],
    cal_l1_pred: Dict[str, np.ndarray],
    l1_fa_per_1k_cap: float = 20.0,
) -> Dict[str, Any]:
    """Lock TWO thresholds strictly on Calibrate (0 Eval touch):
      1. Cal-F1:     max strict micro-F1 on L2 Calibrate (no cap)
      2. Cal-L1Safe: max strict micro-F1 on L2 Calibrate subject to TIMIT Cal target FA <= l1_fa_per_1k_cap
    """
    gt, gg = cal_l2_pred["gold_t"], cal_l2_pred["gold_g"]
    ps, po, gi = cal_l2_pred["p_s"], cal_l2_pred["p_o"], cal_l2_pred["gap_i"]
    sc = ps + po

    l1_sc = cal_l1_pred["p_s"] + cal_l1_pred["p_o"]
    l1_gi = cal_l1_pred["gap_i"]
    n_l1_t = len(l1_sc)

    t_grid = np.unique(np.linspace(0.15, 0.85, 71))
    g_grid = np.unique(np.linspace(0.20, 0.95, 31))

    # 1. Cal-F1 (coordinate search x2)
    best_f1, best_tt, best_tg = -1.0, 0.45, 0.65
    tg = 0.65
    for _ in range(2):
        for tt in t_grid:
            pt, pg = apply_seq_thresholds(ps, po, gi, float(tt), tg)
            f1 = score_decisions(gt, pt, gg, pg)["strict"]["f1"]
            if f1 > best_f1:
                best_f1, best_tt, best_tg = f1, float(tt), float(tg)
        for tg_c in g_grid:
            pt, pg = apply_seq_thresholds(ps, po, gi, best_tt, float(tg_c))
            f1 = score_decisions(gt, pt, gg, pg)["strict"]["f1"]
            if f1 > best_f1:
                best_f1, best_tg = f1, float(tg_c)
        tg = best_tg

    # 2. Cal-L1Safe: enforce TIMIT Cal target FA <= cap AND gap FA <= 5/1k
    valid_tt = [float(tt) for tt in t_grid if (1000.0 * np.sum(l1_sc >= tt) / n_l1_t) <= l1_fa_per_1k_cap]
    if not valid_tt:
        valid_tt = [float(t_grid[-1])]
    valid_tg = [float(tg_c) for tg_c in g_grid if (1000.0 * np.sum(l1_gi >= tg_c) / len(l1_gi)) <= 5.0]
    if not valid_tg:
        valid_tg = [float(g_grid[-1])]

    safe_f1, safe_tt, safe_tg = -1.0, valid_tt[0], valid_tg[0]
    tg = valid_tg[0]
    for _ in range(2):
        for tt in valid_tt:
            pt, pg = apply_seq_thresholds(ps, po, gi, tt, tg)
            f1 = score_decisions(gt, pt, gg, pg)["strict"]["f1"]
            if f1 > safe_f1:
                safe_f1, safe_tt, safe_tg = f1, tt, tg
        for tg_c in valid_tg:
            pt, pg = apply_seq_thresholds(ps, po, gi, safe_tt, tg_c)
            f1 = score_decisions(gt, pt, gg, pg)["strict"]["f1"]
            if f1 > safe_f1:
                safe_f1, safe_tg = f1, tg_c
        tg = safe_tg

    return {
        "cal_f1": {
            "tau_t": round(best_tt, 4), "tau_g": round(best_tg, 4),
            "cal_l2_strict_f1": round(best_f1 * 100, 2),
            "cal_l1_fa_per_1k": round(float(1000.0 * np.sum(l1_sc >= best_tt) / n_l1_t), 2),
        },
        "cal_l1safe": {
            "tau_t": round(safe_tt, 4), "tau_g": round(safe_tg, 4),
            "cal_l2_strict_f1": round(safe_f1 * 100, 2),
            "cal_l1_fa_per_1k": round(float(1000.0 * np.sum(l1_sc >= safe_tt) / n_l1_t), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_all = time.perf_counter()

    # 1. Build strictly speaker-disjoint multi-corpus + L1-anchored datasets
    fit_rows, cal_rows, eval_cohorts = build_datasets()
    print("=== Dataset Splits (Zero Speaker Leakage Verified) ===")
    for lbl, rlist in [("FIT", fit_rows), ("CALIBRATE", cal_rows)]:
        by_src = {}
        for r in rlist:
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        print(f"  {lbl:10s}: {len(rlist):4d} utts, {len({r['speaker'] for r in rlist}):3d} spks | {by_src}")
    for cname, rlist in eval_cohorts.items():
        print(f"  EVAL {cname:11s}: {len(rlist):4d} utts, {len({r['speaker'] for r in rlist}):2d} spks")

    # Build TIMIT Core Corruption Control Test (50 unseen native utts with 15% prompt swaps)
    # to PROVE genuine acoustic-phonetic matching on TIMIT (ZERO mic shortcut!).
    rng_ctrl = np.random.default_rng(SEED + 99)
    timit_ctrl_rows = []
    for r in eval_cohorts["timit_core"][:50]:
        can = list(r["canonical_phones"])
        corr_can = list(can)
        gt_ctrl = ["C"] * len(can)
        for i in range(1, len(can) - 1, 6):
            corr_can[i] = str(rng_ctrl.choice([p for p in ARPA_39 if p != can[i]]))
            gt_ctrl[i] = "S"
        timit_ctrl_rows.append({
            "cohort": "timit_core_corrupt_ctrl",
            "uid": r["uid"] + "_ctrl",
            "speaker": r["speaker"],
            "audio_path": r["audio_path"],
            "canonical_phones": corr_can,
            "gold_targets": gt_ctrl,
            "gold_gaps": ["N"] * (len(corr_can) + 1),
        })
    eval_cohorts["timit_core_corrupt_ctrl"] = timit_ctrl_rows

    # 2. Extract frozen Base Koel 6-layer compact utterance features ONCE, then free Base Koel VRAM!
    print("\n[STAGE 1/3] Extracting frozen Base Koel 6-layer utterance features...")
    koel = load_base_koel(device)
    fit_items = extract_feature_list(koel, fit_rows, device, "FIT")
    cal_items = extract_feature_list(koel, cal_rows, device, "CALIBRATE")
    eval_items: Dict[str, List[Dict[str, Any]]] = {}
    for cname in ["held6", "spontaneous", "epadb_test", "saa27", "timit_core", "timit_core_corrupt_ctrl"]:
        eval_items[cname] = extract_feature_list(koel, eval_cohorts[cname], device, cname)

    del koel
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Fit-only scalar standardization (zero Cal/Eval leakage)
    all_fit_ts = np.concatenate([x["target_scalars"] for x in fit_items], axis=0)
    all_fit_gs = np.concatenate([x["gap_scalars"] for x in fit_items], axis=0)
    t_mean = all_fit_ts.mean(axis=0).astype(np.float32)
    t_std = np.maximum(all_fit_ts.std(axis=0), 1e-5).astype(np.float32)
    g_mean = all_fit_gs.mean(axis=0).astype(np.float32)
    g_std = np.maximum(all_fit_gs.std(axis=0), 1e-5).astype(np.float32)

    cal_l2_items = [x for x in cal_items if not x["is_native_l1"]]
    cal_l1_items = [x for x in cal_items if x["is_native_l1"]]

    # 3. Train SeqMDDv2 (new checkpoint ONLY!)
    print("\n[STAGE 2/3] Training SeqMDDv2 (multi-layer + BiGRU + Target/Gap/Realized + L1 anchor)...")
    model = SeqMDDv2(d_model=192, dropout=0.30).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SeqMDDv2 trainable params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    steps_per_ep = math.ceil(len(fit_items) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_ep, eta_min=2e-5)
    rng = np.random.default_rng(SEED)

    best_score = -1.0
    best_state = None
    best_epoch = 0
    history = []

    for ep in range(1, args.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        order = rng.permutation(len(fit_items))
        ep_loss, ep_lt, ep_lg, ep_lr, n_b = 0.0, 0.0, 0.0, 0.0, 0

        for start in range(0, len(order), args.batch_size):
            batch_items = [fit_items[i] for i in order[start : start + args.batch_size]]
            b = collate_batch(batch_items, t_mean, t_std, g_mean, g_std, device, with_labels=True)
            opt.zero_grad()
            out = model(
                h_vit=b["h_vit"], h_peak=b["h_peak"], lp_win=b["lp_win"], lp_vit=b["lp_vit"],
                phone_ids=b["phone_ids"], prev_phone_ids=b["prev_phone_ids"], next_phone_ids=b["next_phone_ids"],
                target_scalars=b["target_scalars"], h_gap=b["h_gap"], lp_gap_mean=b["lp_gap_mean"],
                lp_gap_max=b["lp_gap_max"], gap_scalars=b["gap_scalars"], lengths=b["lengths"],
            )
            losses = compute_multitask_mdd_loss(
                out, b["y_t"], b["y_g"], b["y_r"], b["is_l1"],
                gap_pos_weight=8.0, lambda_gap=0.6, lambda_realized=0.25, native_fa_weight=1.5,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += float(losses["loss"].item())
            ep_lt += float(losses["loss_target"].item())
            ep_lg += float(losses["loss_gap"].item())
            ep_lr += float(losses["loss_realized"].item())
            n_b += 1

        # Evaluate strictly on Calibrate (0 Eval touch!)
        cal_l2_p = predict_items(model, cal_l2_items, t_mean, t_std, g_mean, g_std, device)
        cal_l1_p = predict_items(model, cal_l1_items, t_mean, t_std, g_mean, g_std, device)
        tv = cal_l2_p["gold_t"] != "unknown"
        gv = cal_l2_p["gold_g"] != "unknown"
        yt = ((cal_l2_p["gold_t"][tv] == "S") | (cal_l2_p["gold_t"][tv] == "O")).astype(int)
        yg = (cal_l2_p["gold_g"][gv] == "I").astype(int)
        t_ap, _ = pr_roc(yt, (cal_l2_p["p_s"] + cal_l2_p["p_o"])[tv])
        g_ap, _ = pr_roc(yg, cal_l2_p["gap_i"][gv])
        pt50, pg50 = apply_seq_thresholds(cal_l2_p["p_s"], cal_l2_p["p_o"], cal_l2_p["gap_i"], 0.45, 0.70)
        cal_f1_50 = score_decisions(cal_l2_p["gold_t"], pt50, cal_l2_p["gold_g"], pg50)["strict"]["f1"]
        l1_fa_50 = float(1000.0 * np.mean((cal_l1_p["p_s"] + cal_l1_p["p_o"]) >= 0.45))
        # Selection criterion: target AP + 0.15 * gap AP (threshold-free!)
        sel_score = float(t_ap) + 0.15 * float(g_ap)
        if sel_score > best_score:
            best_score = sel_score
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        rec_ep = {
            "epoch": ep, "loss": round(ep_loss / n_b, 4), "loss_t": round(ep_lt / n_b, 4),
            "loss_g": round(ep_lg / n_b, 4), "loss_r": round(ep_lr / n_b, 4),
            "cal_t_ap": round(float(t_ap), 4), "cal_g_ap": round(float(g_ap), 4),
            "cal_f1@0.45": round(cal_f1_50 * 100, 2), "cal_l1_fa1k@0.45": round(l1_fa_50, 1),
            "sec": round(time.perf_counter() - t_ep, 1),
        }
        history.append(rec_ep)
        print(f"  [Ep {ep:02d}/{args.epochs}] loss={rec_ep['loss']:.4f} (t={rec_ep['loss_t']:.3f} g={rec_ep['loss_g']:.3f} r={rec_ep['loss_r']:.3f}) | "
              f"Cal tAP={rec_ep['cal_t_ap']:.4f} gAP={rec_ep['cal_g_ap']:.4f} F1@0.45={rec_ep['cal_f1@0.45']:.2f}% L1_FA/1k={rec_ep['cal_l1_fa1k@0.45']:.1f} ({rec_ep['sec']:.1f}s)")

    # Restore best Calibrate epoch & save NEW checkpoint (anti-shortcut v2)!
    model.load_state_dict(best_state)
    model.eval()
    ckpt_path = CKPT_DIR / f"seq_mdd_v2_antishortcut_seed{SEED}.pt"
    torch.save({
        "arch": "SeqMDDv2",
        "seed": SEED,
        "best_epoch": best_epoch,
        "model_kwargs": {"d_model": 192, "dropout": 0.30},
        "state_dict": best_state,
        "t_mean": t_mean, "t_std": t_std,
        "g_mean": g_mean, "g_std": g_std,
        "target_layer_weights": torch.softmax(model.target_layer_logits.detach().cpu(), dim=0).tolist(),
        "gap_layer_weights": torch.softmax(model.gap_layer_logits.detach().cpu(), dim=0).tolist(),
    }, ckpt_path)
    print(f"\nSaved NEW checkpoint: {ckpt_path} (best_epoch={best_epoch}, sha256={sha256_file(ckpt_path)[:16]}...)")
    print("  Learned XLSR target layer weights (L12,15,18,20,22,24):",
          [round(x, 4) for x in torch.softmax(model.target_layer_logits.detach().cpu(), dim=0).tolist()])
    print("  Learned XLSR gap layer weights    (L12,15,18,20,22,24):",
          [round(x, 4) for x in torch.softmax(model.gap_layer_logits.detach().cpu(), dim=0).tolist()])

    # 4. Lock thresholds strictly on Calibrate, then evaluate ONCE on all 5 unseen Eval cohorts!
    print("\n[STAGE 3/3] Locking thresholds on Calibrate & Evaluating on all 5 UNSEEN Eval cohorts...")
    cal_l2_final = predict_items(model, cal_l2_items, t_mean, t_std, g_mean, g_std, device)
    cal_l1_final = predict_items(model, cal_l1_items, t_mean, t_std, g_mean, g_std, device)
    policies = calibrate_dual_policies(cal_l2_final, cal_l1_final, l1_fa_per_1k_cap=20.0)
    print("  Locked Calibrate policies:", json.dumps(policies, indent=2))

    tt_f1, tg_f1 = policies["cal_f1"]["tau_t"], policies["cal_f1"]["tau_g"]
    tt_safe, tg_safe = policies["cal_l1safe"]["tau_t"], policies["cal_l1safe"]["tau_g"]

    results: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "best_epoch": best_epoch,
        "n_params": n_params,
        "learned_target_layer_weights": [round(x, 4) for x in torch.softmax(model.target_layer_logits.detach().cpu(), dim=0).tolist()],
        "learned_gap_layer_weights": [round(x, 4) for x in torch.softmax(model.gap_layer_logits.detach().cpu(), dim=0).tolist()],
        "policies": policies,
        "history": history,
        "eval": {},
        "elapsed_s": round(time.perf_counter() - t_all, 1),
    }

    print("\n" + "=" * 124)
    print("SEQ_MDD_V2 (MULTI-CORPUS + ANTI-SHORTCUT L1-ANCHORED + GAP HEAD) RESULTS ON ALL UNSEEN EVAL COHORTS")
    print("=" * 124)

    for cname in ["held6", "spontaneous", "epadb_test", "saa27", "timit_core", "timit_core_corrupt_ctrl"]:
        pred = predict_items(model, eval_items[cname], t_mean, t_std, g_mean, g_std, device)
        gt, gg = pred["gold_t"], pred["gold_g"]
        ps, po, gi = pred["p_s"], pred["p_o"], pred["gap_i"]

        pt_f1, pg_f1 = apply_seq_thresholds(ps, po, gi, tt_f1, tg_f1)
        pt_safe, pg_safe = apply_seq_thresholds(ps, po, gi, tt_safe, tg_safe)

        if cname == "timit_core":
            n_u = len(eval_items[cname])
            def tim_fa(pt, pg):
                s, o, i = int(np.sum(pt == "S")), int(np.sum(pt == "O")), int(np.sum(pg == "I"))
                return {
                    "fa_total": s + o + i, "fa_S": s, "fa_O": o, "fa_I": i,
                    "fa_per_utt": round((s + o + i) / n_u, 2),
                    "target_fa_per_1k": round(1000.0 * (s + o) / len(gt), 2),
                    "gap_fa_per_1k": round(1000.0 * i / len(gg), 2),
                }
            res_tim = {"cal_f1": tim_fa(pt_f1, pg_f1), "cal_l1safe": tim_fa(pt_safe, pg_safe)}
            results["eval"]["timit_core"] = res_tim
            print(f"\n--- TIMIT CORE (192 unseen native US L1 utts, 6,374 phones) ---")
            print(f"  SeqMDDv2 Cal-F1     : {res_tim['cal_f1']['fa_per_utt']:.2f} FA/utt ({res_tim['cal_f1']['target_fa_per_1k']:.1f}/1k targets, {res_tim['cal_f1']['gap_fa_per_1k']:.1f}/1k gaps) | S={res_tim['cal_f1']['fa_S']} O={res_tim['cal_f1']['fa_O']} I={res_tim['cal_f1']['fa_I']}")
            print(f"  SeqMDDv2 Cal-L1Safe : {res_tim['cal_l1safe']['fa_per_utt']:.2f} FA/utt ({res_tim['cal_l1safe']['target_fa_per_1k']:.1f}/1k targets, {res_tim['cal_l1safe']['gap_fa_per_1k']:.1f}/1k gaps) | S={res_tim['cal_l1safe']['fa_S']} O={res_tim['cal_l1safe']['fa_O']} I={res_tim['cal_l1safe']['fa_I']}")
            continue

        res_f1 = score_decisions(gt, pt_f1, gg, pg_f1)
        res_safe = score_decisions(gt, pt_safe, gg, pg_safe)
        or_tt, or_tg, res_or = oracle_best_f1_sm13(gt, ps, po, gg, gi, mode="sum")

        tv, gv = gt != "unknown", gg != "unknown"
        yt = ((gt[tv] == "S") | (gt[tv] == "O")).astype(int)
        yg = (gg[gv] == "I").astype(int)
        t_ap, t_auc = pr_roc(yt, (ps + po)[tv])
        g_ap, g_auc = pr_roc(yg, gi[gv])

        c_out = {
            "cal_f1": res_f1, "cal_l1safe": res_safe,
            "oracle": {"tau_t": or_tt, "tau_g": or_tg, "metrics": res_or},
            "target_pr_auc": t_ap, "target_roc_auc": t_auc,
            "gap_pr_auc": g_ap, "gap_roc_auc": g_auc,
        }

        # For spontaneous, also report strictly unseen Held-6 speaker slice (24 utts)
        if cname == "spontaneous":
            mt = np.isin(pred["spk_t"], list(HELD6_SPEAKERS))
            mg = np.isin(pred["spk_g"], list(HELD6_SPEAKERS))
            c_out["unseen_held6_slice_cal_f1"] = score_decisions(gt[mt], pt_f1[mt], gg[mg], pg_f1[mg])
            c_out["unseen_held6_slice_cal_l1safe"] = score_decisions(gt[mt], pt_safe[mt], gg[mg], pg_safe[mg])

        results["eval"][cname] = c_out
        g_ap_str = f"{g_ap:.4f}" if g_ap is not None else "N/A"
        print(f"\n--- {cname.upper()} (tAP={t_ap:.4f} | gAP={g_ap_str}) ---")
        for lbl, m in [("Cal-F1    ", res_f1), ("Cal-L1Safe", res_safe), ("Oracle    ", res_or)]:
            st, bi, S, O, I = m["strict"], m["binary"], m["S"], m["O"], m["I"]
            print(f"  {lbl} | μF1={st['f1']*100:5.2f}% (P={st['p']*100:5.1f} R={st['r']*100:5.1f}) | Bin={bi['f1']*100:5.2f}% | "
                  f"S={S['f1']*100:5.2f}% O={O['f1']*100:5.2f}% I={I['f1']*100:5.2f}% (P={I['p']*100:4.1f}/R={I['r']*100:4.1f}) | "
                  f"FP/u={st['fp']/len(eval_items[cname]):5.2f} | S->O={m['s_to_o']:3d} O->S={m['o_to_s']:3d}")
        if cname == "spontaneous":
            u_m = c_out["unseen_held6_slice_cal_f1"]["strict"]
            u_s = c_out["unseen_held6_slice_cal_l1safe"]["strict"]
            print(f"  [Unseen Held-6 spks only (24 utts)] Cal-F1={u_m['f1']*100:.2f}% | Cal-L1Safe={u_s['f1']*100:.2f}%")

    out_json = OUT_DIR / "SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nDONE in {results['elapsed_s']:.1f}s! Saved: {out_json}")


if __name__ == "__main__":
    main()
