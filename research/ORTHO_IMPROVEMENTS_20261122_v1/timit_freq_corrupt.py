"""Lever O1: Frequency-matched in-utterance TIMIT anti-shortcut corruption.

Fixes the Zipfian distortion in train_and_eval_seq_mdd_v2.py:270-276 where sampling
replacement ('S') and bogus ('O') phones uniformly over ARPA_39 flooded rare phones
(ZH 95.6%, UH 74.0%, TH 70.6%, JH 70.1%, CH 67.5%, AW 64.2%, NG 55.1%, UW 21.6%)
while leaving frequent phones at 4-5%, teaching a spurious rare-phone-on-TIMIT prior.

Sampling replacement/bogus phones proportional to the Fit-only clean TIMIT unigram
distribution P_timit(phone) keeps every phone near the nominal ~13% corruption rate
while preserving the exact admitted WAVs, speakers, and overall S/O/I rates.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import unicodedata

import numpy as np

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    IPA_TO_ARPA39,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    PHONE_TO_ID,
)

OMISSION_REALIZED_ID = 39
DEFAULT_INTAKE_MANIFEST = Path(
    "data/processed/phonetic_intake_20261117T164741Z_v1/normalized_recognizer_manifest.jsonl"
)


def _row_rng(seed: int, uid: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).digest()
    row_seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(row_seed)


def load_clean_timit_phones(
    uids: Sequence[str],
    manifest_path: Path = DEFAULT_INTAKE_MANIFEST,
) -> Dict[str, List[str]]:
    """Load exact clean realized ARPAbet-39 phone sequences for admitted TIMIT Fit UIDs."""
    wanted = set(uids)
    out: Dict[str, List[str]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("uid")
            if uid in wanted:
                phones = [
                    IPA_TO_ARPA39[unicodedata.normalize("NFC", t)]
                    for t in r["realized_ipa_tokens"]
                ]
                out[uid] = phones
                if len(out) == len(wanted):
                    break
    missing = wanted - set(out.keys())
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} TIMIT UIDs in {manifest_path}: {sorted(missing)[:5]}"
        )
    return out


def compute_timit_phone_distribution(
    clean_sequences: Sequence[Sequence[str]],
    smoothing: float = 1.0,
) -> Dict[str, float]:
    """Compute smoothed Fit-only clean TIMIT unigram probabilities over ARPA_39."""
    counts = Counter()
    for seq in clean_sequences:
        for ph in seq:
            if ph in PHONE_TO_ID:
                counts[ph] += 1
    total = sum(counts[p] + smoothing for p in ARPA_39)
    return {p: float((counts[p] + smoothing) / total) for p in ARPA_39}


def _sample_other_phone(
    rng: np.random.Generator,
    exclude_phone: str,
    phone_probs: Mapping[str, float],
) -> str:
    cands = [p for p in ARPA_39 if p != exclude_phone]
    weights = np.array([max(1e-12, float(phone_probs[p])) for p in cands], dtype=np.float64)
    weights /= weights.sum()
    idx = int(rng.choice(len(cands), p=weights))
    return cands[idx]


def corrupt_timit_utterance_freq_matched(
    true_phones: Sequence[str],
    phone_probs: Mapping[str, float],
    *,
    seed: int,
    uid: str,
    p_ins: float = 0.03,
    p_omi: float = 0.03,
    p_sub: float = 0.10,
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Apply frequency-matched S/O/I prompt corruption on a single clean TIMIT utterance.

    Preserves the exact event semantics of train_and_eval_seq_mdd_v2.py while replacing
    uniform ARPA_39 draws with Fit-only P_timit(phone) draws.
    """
    if p_ins < 0 or p_omi < 0 or p_ins + p_omi >= 1.0 or not (0.0 <= p_sub <= 1.0):
        raise ValueError("Invalid corruption probabilities")
    rng = _row_rng(seed, uid)
    can_corr: List[str] = []
    gt_corr: List[str] = []
    real_corr: List[int] = []
    gg_corr: List[str] = ["N"]

    n = len(true_phones)
    i_p = 0
    while i_p < n:
        u = float(rng.random())
        tp = true_phones[i_p]
        if u < p_ins and 0 < i_p < n - 1 and len(can_corr) > 0:
            gg_corr[-1] = "I"
            i_p += 1
            continue
        if u < p_ins + p_omi:
            bogus = _sample_other_phone(rng, tp, phone_probs)
            can_corr.append(bogus)
            gt_corr.append("O")
            real_corr.append(OMISSION_REALIZED_ID)
            gg_corr.append("N")
        if float(rng.random()) < p_sub:
            wrong = _sample_other_phone(rng, tp, phone_probs)
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

    assert len(can_corr) == len(gt_corr) == len(real_corr)
    assert len(gg_corr) == len(can_corr) + 1
    return can_corr, gt_corr, gg_corr, real_corr


def rebuild_fit_with_freq_matched_timit(
    fit_rows: Sequence[Mapping[str, Any]],
    clean_timit_map: Mapping[str, Sequence[str]],
    *,
    seed: int = 20261122,
    p_ins: float = 0.03,
    p_omi: float = 0.03,
    p_sub: float = 0.10,
    smoothing: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Return a new Fit row list with only TIMIT anti-shortcut rows rebuilt via P_timit(phone).

    Non-TIMIT rows and all TIMIT WAV paths / SHA-256 / speaker IDs remain byte-identical.
    """
    timit_seqs = [
        clean_timit_map[r["uid"]]
        for r in fit_rows
        if str(r.get("source", "")).startswith("timit")
    ]
    if not timit_seqs:
        raise ValueError("No TIMIT rows found in fit_rows")
    phone_probs = compute_timit_phone_distribution(timit_seqs, smoothing=smoothing)

    rebuilt: List[Dict[str, Any]] = []
    for r in fit_rows:
        row = dict(r)
        if str(row.get("source", "")).startswith("timit"):
            true_phones = clean_timit_map[row["uid"]]
            can, gt, gg, real = corrupt_timit_utterance_freq_matched(
                true_phones,
                phone_probs,
                seed=seed,
                uid=str(row["uid"]),
                p_ins=p_ins,
                p_omi=p_omi,
                p_sub=p_sub,
            )
            row["canonical_phones"] = can
            row["gold_targets"] = gt
            row["gold_gaps"] = gg
            row["realized_ids"] = real
            row["source"] = "timit_train_l1_antishortcut"
        rebuilt.append(row)
    return rebuilt, phone_probs


def audit_timit_phone_rates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize per-phone target error rate uniformity across TIMIT Fit rows."""
    tot, err, cls_cnt = Counter(), Counter(), Counter()
    gap_tot, gap_ins = 0, 0
    for r in rows:
        if not str(r.get("source", "")).startswith("timit"):
            continue
        for ph, g in zip(r["canonical_phones"], r["gold_targets"]):
            if g not in ("C", "S", "O"):
                continue
            tot[ph] += 1
            cls_cnt[g] += 1
            if g != "C":
                err[ph] += 1
        for g in r["gold_gaps"]:
            if g in ("N", "I"):
                gap_tot += 1
                if g == "I":
                    gap_ins += 1

    per_phone = {
        ph: {
            "total": int(tot[ph]),
            "errors": int(err[ph]),
            "err_rate_pct": round(100.0 * err[ph] / max(1, tot[ph]), 2),
        }
        for ph in sorted(tot.keys())
    }
    rates = np.array([v["err_rate_pct"] for v in per_phone.values()], dtype=np.float64)
    worst = sorted(per_phone.items(), key=lambda kv: kv[1]["err_rate_pct"], reverse=True)[:8]
    lowest = sorted(per_phone.items(), key=lambda kv: kv[1]["err_rate_pct"])[:5]
    n_tot = max(1, sum(tot.values()))
    return {
        "n_phones_seen": len(per_phone),
        "overall_target_err_pct": round(100.0 * (cls_cnt["S"] + cls_cnt["O"]) / n_tot, 2),
        "overall_s_pct": round(100.0 * cls_cnt["S"] / n_tot, 2),
        "overall_o_pct": round(100.0 * cls_cnt["O"] / n_tot, 2),
        "overall_gap_i_pct": round(100.0 * gap_ins / max(1, gap_tot), 2),
        "phone_err_rate_mean_pct": round(float(rates.mean()), 2) if len(rates) else 0.0,
        "phone_err_rate_std_pct": round(float(rates.std()), 2) if len(rates) else 0.0,
        "phone_err_rate_min_pct": round(float(rates.min()), 2) if len(rates) else 0.0,
        "phone_err_rate_max_pct": round(float(rates.max()), 2) if len(rates) else 0.0,
        "highest_8": worst,
        "lowest_5": lowest,
        "per_phone": per_phone,
    }
