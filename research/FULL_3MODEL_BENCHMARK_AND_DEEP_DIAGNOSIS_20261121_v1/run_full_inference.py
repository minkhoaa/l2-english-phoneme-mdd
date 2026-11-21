"""Full 3-model benchmark inference (Koel Base, R0, SM13 formal seed 101) across all local cohorts.

Runs sequentially on GPU (single backbone resident at a time) for:
  1. L2-ARCTIC Held-6       (802 utts, 6 unseen R0 speakers; in SM13 train)
  2. L2-ARCTIC Spontaneous  (129 utts, 22 speakers; speaker-disjoint R0 matching)
  3. EpaDB TEST             (1,266 utts, 20 Spanish L1 speakers; unseen for R0 & SM13)
  4. TIMIT Core             (192 utts, 24 US Native L1 speakers; native false-alarm control)
  5. SAA-27 SM13 decomp     (27 utts; extracts neural vs E4 split to complement sealed SAA-27 run)
  6. DEV-18 SM13 decomp     (180 utts, 10/speaker across all 18 speakers; neural vs E4 diagnostic)

For every utterance, records:
  - Koel Base raw IPA tokens, unmapped tokens, greedy target/gap calls, alignment tier, Viterbi fallback
  - R0 target probabilities (p_c, p_s, p_o, score=p_s+p_o) + Base gap calls
  - SM13 formal seed 101 fused probabilities (0.5*neural + 0.5*e4), neural-only probabilities,
    E4-only probabilities, and gap_prob_i
  - Per-utterance GPU wall times
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.phoneme_review import ARPABET
from research.CONTROLLED_PILOT_AB_20261118_v1 import run_pilot_ab as pilot
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ALIGN_TIER_NAMES,
    extract_utterance_base_features,
    normalize_canonical_phone,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.mdd_head import (
    SmallMDDHead,
    logits_to_probs_and_scores,
)
from research.KOEL_DT_ACOUSTIC_CHECK_20261119_v1.extract_dt_features import load_canonical_ids
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.KOEL_SM13_BENCHMARK_20261120_v1.infer_sm13 import SM13Seed101
from research.KOEL_SM13_BENCHMARK_20261120_v1.seed101.align_centers import align_centers
from research.KOEL_V2_FILTER_BENCHMARK_20261119_v1.run_v2_filter_benchmark import FOLDS

HELD6_GOLD = REPO_ROOT / "research/KOEL_ABC_FROZEN_EVAL_20261119_v1/manifests/l2arctic_held6_gold.jsonl"
TIMIT_GOLD = REPO_ROOT / "research/KOEL_ABC_FROZEN_EVAL_20261119_v1/manifests/timit_core_gold.jsonl"
SAA27_GOLD = REPO_ROOT / "research/KOEL_MDD_BRANCH_ABLATION_20261119_v1/runs/saa_b_20261119/manifests/eval_gold.jsonl"
DEV18_GOLD = REPO_ROOT / "research/KOEL_MDD_BRANCH_ABLATION_20261119_v1/runs/saa_b_20261119/manifests/fit_gold.jsonl"
INTAKE_MANIFEST = REPO_ROOT / "data/processed/phonetic_intake_20261117T164741Z_v1/normalized_recognizer_manifest.jsonl"

R0_FULL_CKPT = REPO_ROOT / "research/KOEL_R0_FULL_CONFIG_BENCHMARK_20261120_v1/runs/full_config_20261120/checkpoints/saa27/r0_base.pt"
R0_FOLD_DIR = REPO_ROOT / "research/KOEL_TOP_LAYER_ABLATION_20261120_v1/runs/top_layer_20261120/checkpoints/R0"

# Standard IPA -> ARPAbet39 map for intake corpora (L2_ARCTIC_Spontaneous & EpaDB)
IPA_TO_ARPA39: Dict[str, str] = {
    "ɑ": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH", "ɔ": "AO",
    "aʊ": "AW", "aɪ": "AY", "b": "B", "tʃ": "CH", "d": "D",
    "ð": "DH", "ɛ": "EH", "ɝ": "ER", "ɚ": "ER", "eɪ": "EY",
    "f": "F", "ɡ": "G", "g": "G", "h": "HH", "ɪ": "IH",
    "i": "IY", "dʒ": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "ŋ": "NG", "oʊ": "OW", "ɔɪ": "OY", "p": "P",
    "ɹ": "R", "s": "S", "ʃ": "SH", "t": "T", "θ": "TH",
    "ʊ": "UH", "u": "UW", "v": "V", "w": "W", "j": "Y",
    "z": "Z", "ʒ": "ZH",
}

# EpaDB explicit allophone/accent handling:
# - kʰ -> K, pʰ -> P, ə -> AH (valid English allophones)
# - θʰ and sʰ are upstream-ambiguous merged symbols in KoelLabs/EpaDB -> marked UNKNOWN in strict eval
# - ɾ (flap) is accepted allophone of T/D, substitution on R/other
# - a, e, o, β, x, ɣ, r are genuine Spanish L2 non-English realizations -> distinct token -> S
EPADB_REALIZED_MAP: Dict[str, str] = {
    **IPA_TO_ARPA39,
    "kʰ": "K",
    "pʰ": "P",
    "ɾ": "<FLAP>",
    "θʰ": "<AMBIG_TH_H>",
    "sʰ": "<AMBIG_S_H>",
    "a": "<SPA_A>",
    "e": "<SPA_E>",
    "o": "<SPA_O>",
    "β": "<SPA_BETA>",
    "x": "<SPA_X>",
    "ɣ": "<SPA_GAMMA>",
    "r": "<SPA_TRILL>",
}

# Unambiguous sensitivity mapping for EpaDB if someone forces θʰ->TH and sʰ->SH
EPADB_REALIZED_ALT_MAP: Dict[str, str] = {
    **EPADB_REALIZED_MAP,
    "θʰ": "TH",
    "sʰ": "SH",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def dp_align_csoi(
    canonical: Sequence[str],
    realized: Sequence[str],
    *,
    allow_flap_td: bool = False,
    ambig_tokens: set[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """Deterministic Levenshtein alignment producing (gold_targets[L], gold_gaps[L+1]).

    Matches standard MDD convention:
      - Target: 'C' if match (or flap on T/D), 'unknown' if aligned to ambig_tokens, 'S' if sub, 'O' if del.
      - Gap[k]: 'I' if >=1 insertion before target k (or after last target for k=L), else 'N'.
    Tie-break preference matches standard DP (match/sub < del < ins).
    """
    ambig = ambig_tokens or set()
    n, m = len(canonical), len(realized)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bp = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=M/S, 1=D, 2=I
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
            best = cost_ms
            op = 0
            if cost_d < best:
                best = cost_d
                op = 1
            if cost_i < best:
                best = cost_i
                op = 2
            dp[i, j] = best
            bp[i, j] = op

    targets = ["O"] * n
    gaps = ["N"] * (n + 1)
    i, j = n, m
    while i > 0 or j > 0:
        op = int(bp[i, j])
        if op == 0:
            ci = canonical[i - 1]
            rj = realized[j - 1]
            if rj in ambig:
                targets[i - 1] = "unknown"
            elif ci == rj or (allow_flap_td and rj == "<FLAP>" and ci in {"T", "D"}):
                targets[i - 1] = "C"
            else:
                targets[i - 1] = "S"
            i -= 1
            j -= 1
        elif op == 1:
            targets[i - 1] = "O"
            i -= 1
        else:
            rj = realized[j - 1]
            if rj in ambig:
                if gaps[i] == "N":
                    gaps[i] = "unknown"
            else:
                gaps[i] = "I"
            j -= 1
    return targets, gaps


def load_cohorts() -> Dict[str, List[Dict[str, Any]]]:
    cohorts: Dict[str, List[Dict[str, Any]]] = {}

    # 1. Held-6 (exact pre-existing gold labels)
    held6_rows = []
    for line in HELD6_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ap = Path(r["audio_path"])
        if not ap.is_absolute():
            ap = REPO_ROOT / ap
        held6_rows.append({
            "cohort": "held6",
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
    cohorts["held6"] = held6_rows

    # 2. L2-ARCTIC Spontaneous & 3. EpaDB TEST from intake manifest
    spont_rows = []
    epadb_rows = []
    with INTAKE_MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = r["corpus"]
            if c == "L2_ARCTIC_Spontaneous":
                can_ipa = r["canonical_ipa_tokens"]
                real_ipa = r["realized_ipa_tokens"]
                can_arpa = [IPA_TO_ARPA39.get(unicodedata.normalize("NFC", t), "AA" if t == "a" else None) for t in can_ipa]
                real_arpa = [IPA_TO_ARPA39.get(unicodedata.normalize("NFC", t), "AA" if t == "a" else None) for t in real_ipa]
                if any(x is None for x in can_arpa) or any(x is None for x in real_arpa):
                    raise ValueError(f"Unmapped token in spontaneous {r['uid']}")
                gt, gg = dp_align_csoi(can_arpa, real_arpa)
                ap = REPO_ROOT / r["audio_path"]
                spont_rows.append({
                    "cohort": "spontaneous",
                    "uid": r["uid"],
                    "speaker": str(r["speaker_id"]).upper(),
                    "l1": str(r["native_language"]),
                    "audio_path": str(ap),
                    "canonical_phones": can_arpa,
                    "gold_targets": gt,
                    "gold_gaps": gg,
                    "gold_targets_alt": gt,
                    "gold_gaps_alt": gg,
                })
            elif c == "EpaDB" and r.get("split") == "TEST":
                can_ipa = r["canonical_ipa_tokens"]
                real_ipa = r["realized_ipa_tokens"]
                can_arpa = [IPA_TO_ARPA39[unicodedata.normalize("NFC", t)] for t in can_ipa]
                real_strict = [EPADB_REALIZED_MAP[unicodedata.normalize("NFC", t)] for t in real_ipa]
                real_alt = [EPADB_REALIZED_ALT_MAP[unicodedata.normalize("NFC", t)] for t in real_ipa]
                gt, gg = dp_align_csoi(
                    can_arpa, real_strict, allow_flap_td=True, ambig_tokens={"<AMBIG_TH_H>", "<AMBIG_S_H>"}
                )
                gt_alt, gg_alt = dp_align_csoi(can_arpa, real_alt, allow_flap_td=True)
                ap = REPO_ROOT / r["audio_path"]
                epadb_rows.append({
                    "cohort": "epadb_test",
                    "uid": r["uid"],
                    "speaker": str(r["speaker_id"]),
                    "l1": "Spanish",
                    "audio_path": str(ap),
                    "canonical_phones": can_arpa,
                    "gold_targets": gt,
                    "gold_gaps": gg,
                    "gold_targets_alt": gt_alt,
                    "gold_gaps_alt": gg_alt,
                })
    cohorts["spontaneous"] = spont_rows
    cohorts["epadb_test"] = epadb_rows

    # 4. TIMIT Core (Native US English L1 Control)
    timit_rows = []
    for line in TIMIT_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ap = Path(r["audio_path"])
        if not ap.is_absolute():
            ap = REPO_ROOT / ap
        can_arpa = [normalize_canonical_phone(p.upper()) for p in r["canonical39_ref"]]
        surf_arpa = [
            "<FLAP>" if p.upper() == "DX" else normalize_canonical_phone(p.upper())
            for p in r["surface39_ref"]
        ]
        # Primary gold: native speaker => all canonical phones are C, all gaps are N
        gt_native = ["C"] * len(can_arpa)
        gg_native = ["N"] * (len(can_arpa) + 1)
        # Alt gold: surface phonetic variation vs canonical pronunciation
        gt_surf, gg_surf = dp_align_csoi(can_arpa, surf_arpa, allow_flap_td=True)
        timit_rows.append({
            "cohort": "timit_core",
            "uid": r["uid"],
            "speaker": str(r["speaker_id"]).upper(),
            "l1": "English_L1_Native",
            "audio_path": str(ap),
            "canonical_phones": can_arpa,
            "gold_targets": gt_native,
            "gold_gaps": gg_native,
            "gold_targets_alt": gt_surf,
            "gold_gaps_alt": gg_surf,
        })
    cohorts["timit_core"] = timit_rows

    return cohorts


def load_sm13_decomp_cohorts() -> Dict[str, List[Dict[str, Any]]]:
    """Small diagnostic sets for SM13 neural-vs-E4 decomposition on SAA-27 (all 27) & DEV-18 (10/spk=180)."""
    out: Dict[str, List[Dict[str, Any]]] = {}

    saa_rows = []
    for line in SAA27_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ap = Path(r["audio_path"])
        if not ap.is_absolute():
            ap = REPO_ROOT / ap
        saa_rows.append({
            "cohort": "saa27_sm13_decomp",
            "uid": r["uid"],
            "speaker": str(r["speaker_id"]).upper(),
            "l1": str(r["l1"]),
            "audio_path": str(ap),
            "canonical_phones": [normalize_canonical_phone(p) for p in r["canonical_phones"]],
            "gold_targets": list(r["gold_targets"]),
            "gold_gaps": list(r["gold_gaps"]),
        })
    out["saa27_sm13_decomp"] = saa_rows

    # 10 deterministic utterances per speaker across all 18 DEV-18 speakers = 180 utts
    by_spk: Dict[str, List[Dict[str, Any]]] = {}
    for line in DEV18_GOLD.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        spk = str(r["speaker_id"]).upper()
        by_spk.setdefault(spk, []).append(r)
    dev_sub = []
    for spk in sorted(by_spk):
        rows_s = sorted(by_spk[spk], key=lambda x: x["uid"])
        step = max(1, len(rows_s) // 10)
        picked = rows_s[::step][:10]
        for r in picked:
            ap = Path(r["audio_path"])
            if not ap.is_absolute():
                ap = REPO_ROOT / ap
            dev_sub.append({
                "cohort": "dev18_sm13_decomp",
                "uid": r["uid"],
                "speaker": spk,
                "l1": str(r["l1"]),
                "audio_path": str(ap),
                "canonical_phones": [normalize_canonical_phone(p) for p in r["canonical_phones"]],
                "gold_targets": list(r["gold_targets"]),
                "gold_gaps": list(r["gold_gaps"]),
            })
    out["dev18_sm13_decomp"] = dev_sub
    return out


def load_r0_heads(device: torch.device) -> Tuple[Dict[str, Tuple[SmallMDDHead, np.ndarray, np.ndarray]], Dict[str, str]]:
    """Load full-DEV-18 R0 + all 6 DEV-18 OOF fold heads (tiny: ~3MB each)."""
    heads: Dict[str, Tuple[SmallMDDHead, np.ndarray, np.ndarray]] = {}

    ck = torch.load(R0_FULL_CKPT, map_location="cpu", weights_only=False)
    h_full = SmallMDDHead(**ck.get("model_kwargs", {"dropout": 0.35})).to(device).eval()
    h_full.load_state_dict(ck["state_dict"], strict=True)
    heads["r0_full_dev18"] = (
        h_full,
        np.asarray(ck["scalar_mean"], dtype=np.float32).reshape(-1),
        np.asarray(ck["scalar_std"], dtype=np.float32).reshape(-1),
    )

    spk_to_fold_key: Dict[str, str] = {}
    for fi, eval_spks in enumerate(FOLDS):
        fk = f"r0_fold_{fi}"
        fck = torch.load(R0_FOLD_DIR / f"seed_20261119_fold_{fi}.pt", map_location="cpu", weights_only=False)
        hf = SmallMDDHead(dropout=0.35).to(device).eval()
        hf.load_state_dict(fck["state_dict"], strict=True)
        sc_mean = fck["sc_mean"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        sc_std = fck["sc_std"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        heads[fk] = (hf, sc_mean, sc_std)
        for spk in eval_spks:
            spk_to_fold_key[spk.upper()] = fk

    return heads, spk_to_fold_key


def run_koel_and_r0_pass(
    cohorts: Dict[str, List[Dict[str, Any]]],
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    print("\n[PASS 1/2] Loading frozen Base Koel + R0 heads on", device)
    koel_model = load_base_koel(device)
    canonical_ids = load_canonical_ids()
    koel_vocab = json.loads((pilot.WEIGHTS_DIR / "vocab.json").read_text(encoding="utf-8"))
    inverse_vocab = {int(idx): tok for tok, idx in koel_vocab.items()}
    r0_heads, spk_to_fold = load_r0_heads(device)

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    utt_meta: Dict[str, List[Dict[str, Any]]] = {}

    for cname, rows in cohorts.items():
        t_c0 = time.perf_counter()
        results[cname] = {}
        utt_meta[cname] = []
        print(f"  [Koel+R0] Running {cname}: {len(rows)} utts...")
        for idx_u, rec in enumerate(rows):
            t0 = time.perf_counter()
            wav_np, sr = sf.read(rec["audio_path"], dtype="float32")
            if sr != 16000 or wav_np.ndim != 1:
                raise ValueError(f"Bad wav {rec['uid']}: sr={sr}, shape={wav_np.shape}")
            dur = float(len(wav_np) / sr)
            wav_t = torch.from_numpy(wav_np).unsqueeze(0)
            feats = extract_utterance_base_features(
                koel_model, wav_t, rec["canonical_phones"], inverse_vocab, canonical_ids, device
            )
            t1 = time.perf_counter()

            # Speaker-disjoint R0 checkpoint selection:
            # - For spontaneous: DEV-18 speakers use their held-out fold head; Held-6 speakers use r0_full_dev18
            # - For held6, epadb_test, timit_core: 100% unseen speakers -> r0_full_dev18
            if cname == "spontaneous" and rec["speaker"] in spk_to_fold:
                r0_key = spk_to_fold[rec["speaker"]]
            else:
                r0_key = "r0_full_dev18"
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
                out = logits_to_probs_and_scores(logits)
            t2 = time.perf_counter()

            # Track unmapped Koel tokens
            tokens = feats["tokens"]
            _, _, unmapped = pilot.normalize(tokens)

            results[cname][rec["uid"]] = {
                "greedy_call": feats["a_greedy_call"],
                "gap_call": feats["a_gap_call"],
                "align_tiers": [int(t) for t in feats["align_tiers"]],
                "vit_uniform_fallback": bool(feats["vit_uniform_fallback"]),
                "r0_ckpt": r0_key,
                "r0_p_c": out["p_c"].cpu().numpy().astype(np.float32),
                "r0_p_s": out["p_s"].cpu().numpy().astype(np.float32),
                "r0_p_o": out["p_o"].cpu().numpy().astype(np.float32),
                "r0_score": out["score"].cpu().numpy().astype(np.float32),
            }
            utt_meta[cname].append({
                "cohort": cname,
                "uid": rec["uid"],
                "speaker": rec["speaker"],
                "l1": rec["l1"],
                "duration_s": round(dur, 4),
                "n_targets": len(rec["canonical_phones"]),
                "n_frames": int(feats["n_frames"]),
                "r0_ckpt": r0_key,
                "vit_uniform_fallback": bool(feats["vit_uniform_fallback"]),
                "native_tokens": tokens,
                "unmapped_tokens": unmapped,
                "n_unmapped": len(unmapped),
                "koel_ms": round((t1 - t0) * 1000.0, 2),
                "r0_head_ms": round((t2 - t1) * 1000.0, 2),
            })
            if (idx_u + 1) % 200 == 0 or (idx_u + 1) == len(rows):
                print(f"    [{cname}] {idx_u + 1}/{len(rows)} ({time.perf_counter() - t_c0:.1f}s)")

    del koel_model, r0_heads
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, utt_meta


def predict_sm13_with_decomposition(
    sm13: SM13Seed101,
    samples: np.ndarray,
    canonical_phones: List[str],
) -> Dict[str, np.ndarray]:
    """Exact unmodified SM13Seed101 forward pass, also exposing neural and E4 probabilities."""
    samples = np.asarray(samples, dtype=np.float32)
    n = len(canonical_phones)
    with torch.inference_mode():
        waveform = torch.as_tensor(samples, device=sm13.device).unsqueeze(0)
        aligned_logits = sm13.aligner(waveform).logits[0].cpu().numpy()
        centers, frames = align_centers(aligned_logits, canonical_phones, sm13.tokenizer)
        model_input = sm13.extractor(samples, sampling_rate=16000, return_tensors="pt")
        hidden = sm13.backbone(
            model_input["input_values"].to(sm13.device), output_hidden_states=True
        ).hidden_states
        target_feat, gap_feat = sm13.engine.extract_features_from_hidden_states(hidden, centers, frames)
        ref_t = np.eye(39, dtype=np.float32)[[sm13.phone_to_ref[p] for p in canonical_phones]]
        ref_g = np.zeros((n + 1, 78), dtype=np.float32)
        ref_g[1:, :39] = ref_t
        ref_g[:-1, 39:] = ref_t
        neural = sm13.target.predict_proba(
            torch.as_tensor(target_feat, device=sm13.device),
            torch.as_tensor(ref_t, device=sm13.device),
        ).cpu().numpy().astype(np.float32)
        gap = sm13.gap.predict_proba(
            torch.as_tensor(gap_feat, device=sm13.device),
            torch.as_tensor(ref_g, device=sm13.device),
        )[:, 1].cpu().numpy().astype(np.float32)
        logp = sm13.ctc(hidden[6][0]).log_softmax(-1).cpu().numpy()

    e4, sllr, ollr = sm13.engine.compute_e4_likelihoods(
        logp, [sm13.engine.PHONE_TO_ID[p] for p in canonical_phones], sm13.kernel
    )
    fused = (0.5 * neural + 0.5 * e4).astype(np.float32)
    return {
        "fused": fused,
        "neural": neural,
        "e4": e4.astype(np.float32),
        "sllr": sllr.astype(np.float32),
        "ollr": ollr.astype(np.float32),
        "gap": gap,
    }


def run_sm13_pass(
    cohorts: Dict[str, List[Dict[str, Any]]],
    device: str,
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Dict[str, float]]]:
    print(f"\n[PASS 2/2] Loading SM13 formal seed 101 on {device}")
    sm13 = SM13Seed101(device=device)
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    timings: Dict[str, Dict[str, float]] = {}

    for cname, rows in cohorts.items():
        t_c0 = time.perf_counter()
        results[cname] = {}
        timings[cname] = {}
        print(f"  [SM13] Running {cname}: {len(rows)} utts...")
        for idx_u, rec in enumerate(rows):
            t0 = time.perf_counter()
            wav_np, sr = sf.read(rec["audio_path"], dtype="float32")
            if sr != 16000 or wav_np.ndim != 1:
                raise ValueError(f"Bad wav {rec['uid']}")
            out = predict_sm13_with_decomposition(sm13, wav_np, rec["canonical_phones"])
            t1 = time.perf_counter()
            results[cname][rec["uid"]] = out
            timings[cname][rec["uid"]] = round((t1 - t0) * 1000.0, 2)
            if (idx_u + 1) % 200 == 0 or (idx_u + 1) == len(rows):
                print(f"    [{cname}] {idx_u + 1}/{len(rows)} ({time.perf_counter() - t_c0:.1f}s)")

    del sm13
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return results, timings


def save_cohort_parquet(
    out_dir: Path,
    cname: str,
    rows: List[Dict[str, Any]],
    koel_res: Dict[str, Dict[str, Any]] | None,
    sm13_res: Dict[str, Dict[str, np.ndarray]],
) -> Path:
    cols: Dict[str, List[Any]] = {
        "cohort": [], "uid": [], "speaker": [], "l1": [], "domain": [], "index": [],
        "canonical_phone": [], "gold_label": [], "gold_label_alt": [],
        "vanilla_call": [], "align_tier": [], "vit_fallback": [], "r0_ckpt": [],
        "r0_p_c": [], "r0_p_s": [], "r0_p_o": [], "r0_score": [],
        "sm13_fused_p_c": [], "sm13_fused_p_s": [], "sm13_fused_p_o": [],
        "sm13_neural_p_c": [], "sm13_neural_p_s": [], "sm13_neural_p_o": [],
        "sm13_e4_p_c": [], "sm13_e4_p_s": [], "sm13_e4_p_o": [],
        "sm13_sllr": [], "sm13_ollr": [], "sm13_gap_prob_i": [],
    }
    for rec in rows:
        uid = rec["uid"]
        spk = rec["speaker"]
        l1 = rec["l1"]
        phones = rec["canonical_phones"]
        gt = rec["gold_targets"]
        gg = rec["gold_gaps"]
        gt_alt = rec.get("gold_targets_alt", gt)
        gg_alt = rec.get("gold_gaps_alt", gg)
        kr = koel_res[uid] if koel_res is not None else None
        sr = sm13_res[uid]

        n = len(phones)
        for i in range(n):
            cols["cohort"].append(cname)
            cols["uid"].append(uid)
            cols["speaker"].append(spk)
            cols["l1"].append(l1)
            cols["domain"].append("target")
            cols["index"].append(i)
            cols["canonical_phone"].append(phones[i])
            cols["gold_label"].append(gt[i])
            cols["gold_label_alt"].append(gt_alt[i])
            cols["vanilla_call"].append(kr["greedy_call"][i] if kr else "")
            cols["align_tier"].append(kr["align_tiers"][i] if kr else -1)
            cols["vit_fallback"].append(kr["vit_uniform_fallback"] if kr else False)
            cols["r0_ckpt"].append(kr["r0_ckpt"] if kr else "")
            cols["r0_p_c"].append(float(kr["r0_p_c"][i]) if kr else np.nan)
            cols["r0_p_s"].append(float(kr["r0_p_s"][i]) if kr else np.nan)
            cols["r0_p_o"].append(float(kr["r0_p_o"][i]) if kr else np.nan)
            cols["r0_score"].append(float(kr["r0_score"][i]) if kr else np.nan)
            cols["sm13_fused_p_c"].append(float(sr["fused"][i, 0]))
            cols["sm13_fused_p_s"].append(float(sr["fused"][i, 1]))
            cols["sm13_fused_p_o"].append(float(sr["fused"][i, 2]))
            cols["sm13_neural_p_c"].append(float(sr["neural"][i, 0]))
            cols["sm13_neural_p_s"].append(float(sr["neural"][i, 1]))
            cols["sm13_neural_p_o"].append(float(sr["neural"][i, 2]))
            cols["sm13_e4_p_c"].append(float(sr["e4"][i, 0]))
            cols["sm13_e4_p_s"].append(float(sr["e4"][i, 1]))
            cols["sm13_e4_p_o"].append(float(sr["e4"][i, 2]))
            cols["sm13_sllr"].append(float(sr["sllr"][i]))
            cols["sm13_ollr"].append(float(sr["ollr"][i]))
            cols["sm13_gap_prob_i"].append(np.nan)

        for g in range(n + 1):
            v_gap = kr["gap_call"][g] if kr else ""
            cols["cohort"].append(cname)
            cols["uid"].append(uid)
            cols["speaker"].append(spk)
            cols["l1"].append(l1)
            cols["domain"].append("gap")
            cols["index"].append(g)
            cols["canonical_phone"].append("<GAP>")
            cols["gold_label"].append(gg[g])
            cols["gold_label_alt"].append(gg_alt[g])
            cols["vanilla_call"].append(v_gap)
            cols["align_tier"].append(-1)
            cols["vit_fallback"].append(kr["vit_uniform_fallback"] if kr else False)
            cols["r0_ckpt"].append(kr["r0_ckpt"] if kr else "")
            cols["r0_p_c"].append(np.nan)
            cols["r0_p_s"].append(np.nan)
            cols["r0_p_o"].append(np.nan)
            cols["r0_score"].append((1.0 if v_gap == "I" else 0.0) if kr else np.nan)
            cols["sm13_fused_p_c"].append(np.nan)
            cols["sm13_fused_p_s"].append(np.nan)
            cols["sm13_fused_p_o"].append(np.nan)
            cols["sm13_neural_p_c"].append(np.nan)
            cols["sm13_neural_p_s"].append(np.nan)
            cols["sm13_neural_p_o"].append(np.nan)
            cols["sm13_e4_p_c"].append(np.nan)
            cols["sm13_e4_p_s"].append(np.nan)
            cols["sm13_e4_p_o"].append(np.nan)
            cols["sm13_sllr"].append(np.nan)
            cols["sm13_ollr"].append(np.nan)
            cols["sm13_gap_prob_i"].append(float(sr["gap"][g]))

    pq_path = out_dir / f"{cname}_slots.parquet"
    pq.write_table(pa.table(cols), pq_path, compression="zstd")
    return pq_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    t_all = time.perf_counter()
    cohorts = load_cohorts()
    sm13_decomp = load_sm13_decomp_cohorts()

    print("=== Cohort Summary ===")
    for k, v in {**cohorts, **sm13_decomp}.items():
        n_t = sum(len(r["canonical_phones"]) for r in v)
        print(f"  {k:20s}: {len(v):4d} utts, {len({r['speaker'] for r in v}):2d} spks, {n_t:6d} targets")

    # Pass 1: Koel Base + R0 on the 4 new cohorts
    koel_res, utt_meta = run_koel_and_r0_pass(cohorts, dev)

    # Pass 2: SM13 on the 4 new cohorts + 2 decomposition sets
    sm13_all_input = {**cohorts, **sm13_decomp}
    sm13_res, sm13_timings = run_sm13_pass(sm13_all_input, args.device)

    # Save outputs & SHA-256 receipt
    receipt: Dict[str, Any] = {
        "device": args.device,
        "elapsed_total_s": round(time.perf_counter() - t_all, 2),
        "files": {},
    }
    for cname, rows in cohorts.items():
        pq_path = save_cohort_parquet(out_dir, cname, rows, koel_res[cname], sm13_res[cname])
        meta_path = out_dir / f"{cname}_utterance_meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for m in utt_meta[cname]:
                m["sm13_ms"] = sm13_timings[cname][m["uid"]]
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        receipt["files"][pq_path.name] = {"sha256": sha256_file(pq_path), "bytes": pq_path.stat().st_size}
        receipt["files"][meta_path.name] = {"sha256": sha256_file(meta_path), "bytes": meta_path.stat().st_size}

    for cname, rows in sm13_decomp.items():
        pq_path = save_cohort_parquet(out_dir, cname, rows, None, sm13_res[cname])
        receipt["files"][pq_path.name] = {"sha256": sha256_file(pq_path), "bytes": pq_path.stat().st_size}

    receipt_path = out_dir / "INFERENCE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nDONE in {receipt['elapsed_total_s']:.1f}s! Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
