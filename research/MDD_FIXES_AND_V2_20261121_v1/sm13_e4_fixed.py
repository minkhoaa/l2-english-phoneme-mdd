"""Fixed SM13 E4 sequence likelihood scoring (zero retraining — keeps all seed101 checkpoints untouched!).

Root cause fixed (agent_a_sm13_engine.py:236, 281):
  OLD (BROKEN):
    sub_cands = [(phone_id + k) % 42 for k in range(1, 5)]   # 4 arbitrary alphabetical neighbors!
    max_sub = np.mean(sub_scores)                             # arithmetic mean -> massively negative!
    -> P_e4(O) > P_e4(S) on 99.54% of true S; caps P_fused(S) < 0.857 -> 0.00% S recall under fixed/0.2!

  NEW (FIXED):
    - For each canonical position i, selects the top-K (default K=6) highest-posterior acoustic
      competitor phones from log_probs in the local window around centers[i] (excluding canonical,
      BLANK_ID, SPN, ERR).
    - Computes max_sub = np.max(sub_scores) (the true max competitor substitution log-likelihood,
      matching the author's original variable name `max_sub`!).
    - Preserves exact SM13 seed101 checkpoints and C++ ctc_batch.so kernel untouched.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from research.KOEL_SM13_BENCHMARK_20261120_v1.infer_sm13 import SM13Seed101
from research.KOEL_SM13_BENCHMARK_20261120_v1.seed101.align_centers import align_centers

# Valid real speech phone IDs in SM13's 43-class LinearCTC (exclude SPN=32, ERR=13, BLANK=42)
EXCLUDED_COMPETITOR_IDS: frozenset[int] = frozenset({13, 32, 42})  # ERR, SPN, BLANK


def select_acoustic_competitors(
    log_probs: np.ndarray,
    centers: np.ndarray,
    n_align_frames: int,
    canonical_phone_ids: Sequence[int],
    *,
    top_k: int = 6,
    window_radius: int = 6,
) -> List[List[int]]:
    """Select top-K acoustically plausible competitor phone IDs per position from CTC log_probs."""
    T, C = log_probs.shape
    N = len(canonical_phone_ids)
    if N == 0:
        return []

    scaled_c = np.rint(centers * (T / max(1, n_align_frames))).astype(int)
    valid_ids = np.array([cid for cid in range(min(C, 42)) if cid not in EXCLUDED_COMPETITOR_IDS], dtype=np.int32)

    all_cands: List[List[int]] = []
    for i, can_id in enumerate(canonical_phone_ids):
        c = int(np.clip(scaled_c[i], 0, T - 1))
        lo = max(0, c - window_radius)
        hi = min(T, c + window_radius + 1)
        # Max frame log-prob in local window around position i
        win_max = log_probs[lo:hi, valid_ids].max(axis=0)
        # Exclude the canonical phone itself
        mask_other = valid_ids != can_id
        other_ids = valid_ids[mask_other]
        other_scores = win_max[mask_other]
        k = min(top_k, len(other_ids))
        top_idx = np.argpartition(other_scores, -k)[-k:]
        top_idx = top_idx[np.argsort(-other_scores[top_idx])]
        all_cands.append([int(other_ids[j]) for j in top_idx])
    return all_cands


def compute_e4_likelihoods_fixed(
    log_probs: np.ndarray,
    canonical_phone_ids: Sequence[int],
    centers: np.ndarray,
    n_align_frames: int,
    kernel_fn,
    *,
    blank_id: int = 42,
    top_k: int = 6,
    window_radius: int = 6,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed E4 sequence likelihood ratio computation using true acoustic top-K competitors & max()."""
    N = len(canonical_phone_ids)
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    import ctypes

    T = log_probs.shape[0]
    lp_contig = np.ascontiguousarray(log_probs, dtype=np.float64)
    cands_per_pos = select_acoustic_competitors(
        lp_contig, centers, n_align_frames, canonical_phone_ids, top_k=top_k, window_radius=window_radius
    )

    sequences: List[List[int]] = [list(canonical_phone_ids)]
    sub_map: List[List[int]] = []
    om_map: List[int] = []

    for i in range(N):
        sub_indices = []
        for cand in cands_per_pos[i]:
            hyp = list(canonical_phone_ids)
            hyp[i] = cand
            sub_indices.append(len(sequences))
            sequences.append(hyp)
        sub_map.append(sub_indices)

        hyp_del = list(canonical_phone_ids[:i]) + list(canonical_phone_ids[i + 1 :])
        om_map.append(len(sequences))
        sequences.append(hyp_del)

    flat_seqs: List[int] = []
    offsets: List[int] = [0]
    for seq in sequences:
        flat_seqs.extend(seq)
        offsets.append(len(flat_seqs))

    flat_arr = np.array(flat_seqs, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    scores_out = np.zeros(len(sequences), dtype=np.float64)

    kernel_fn(
        lp_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(T),
        ctypes.c_int(43),
        ctypes.c_int(blank_id),
        flat_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(sequences)),
        scores_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    can_score = scores_out[0]
    sllr_vals = np.zeros(N, dtype=np.float64)
    ollr_vals = np.zeros(N, dtype=np.float64)
    e4_simplex = np.zeros((N, 3), dtype=np.float32)

    inv_temp = 1.0 / max(temperature, 1e-6)
    for i in range(N):
        sub_scores = [scores_out[idx] for idx in sub_map[i]]
        # FIX: True max over acoustically plausible competitors (NOT np.mean of 4 alphabetical neighbors!)
        max_sub = float(np.max(sub_scores))
        sllr = max_sub - can_score
        del_score = scores_out[om_map[i]]
        ollr = del_score - can_score

        sllr_vals[i] = sllr
        ollr_vals[i] = ollr

        logits = np.array([0.0, sllr * inv_temp, ollr * inv_temp], dtype=np.float64)
        exp_l = np.exp(logits - np.max(logits))
        e4_simplex[i] = (exp_l / np.sum(exp_l)).astype(np.float32)

    return e4_simplex, sllr_vals, ollr_vals


def predict_sm13_fixed(
    sm13: SM13Seed101,
    samples: np.ndarray,
    canonical_phones: Sequence[str],
    *,
    top_k: int = 6,
    e4_weight: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Run SM13 using EXISTING untouched seed101 checkpoints with the fixed E4 algorithm."""
    samples = np.asarray(samples, dtype=np.float32)
    n = len(canonical_phones)
    with torch.inference_mode():
        waveform = torch.as_tensor(samples, device=sm13.device).unsqueeze(0)
        aligned_logits = sm13.aligner(waveform).logits[0].cpu().numpy()
        centers, frames = align_centers(aligned_logits, list(canonical_phones), sm13.tokenizer)
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

    can_ids = [sm13.engine.PHONE_TO_ID[p] for p in canonical_phones]
    e4_fixed, sllr, ollr = compute_e4_likelihoods_fixed(
        logp, can_ids, centers, frames, sm13.kernel, top_k=top_k
    )
    fused_fixed = ((1.0 - e4_weight) * neural + e4_weight * e4_fixed).astype(np.float32)
    return {
        "fused_fixed": fused_fixed,
        "neural": neural,
        "e4_fixed": e4_fixed,
        "sllr_fixed": sllr.astype(np.float32),
        "ollr_fixed": ollr.astype(np.float32),
        "gap": gap,
    }
