"""Allophone-aware, 100%-vocab-covered Base Koel feature extractor (P0 Zero-Retrain Fix).

Drop-in compatible with existing frozen R0 checkpoints (`r0_base.pt` and `seed_20261119_fold_*.pt`):
  1. Uses `koel_ipa_map_v2` covering all 75 real IPA tokens in KoelLabs/xlsr-english-01 (0% unmapped!).
  2. Uses `ALLOPHONE_TOKEN_IDS` in both CTC Viterbi alignment AND `can_lp` / `other_lp` / `frame_margin`
     (generalizing ER in {74, 70} to all legal English allophones: kʰ, pʰ, ɾ, ʔ, ɨ, ʉ, l̩, n̩, m̩, ɦ, ...).
  3. Eliminates spurious `align_tier_3` / `a_call_is_Q` triggers on valid English allophones while
     properly flagging genuine non-English L2 phones (a, e, o, β, x, ɣ, r) as S instead of '?'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    CTX_BOS_ID,
    CTX_EOS_ID,
    PHONE_TO_ID,
    SCALAR_FEATURE_NAMES,
    WINDOW_RADIUS,
    WINDOW_SIZE,
    normalize_canonical_phone,
)
from research.KOEL_DT_ACOUSTIC_CHECK_20261119_v1.extract_dt_features import emitted_segments
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import (
    ALLOPHONE_TOKEN_IDS,
    align_and_decide_v2,
    normalize_koel_tokens_v2,
)


def ctc_viterbi_align_allophone_aware(
    lp: np.ndarray,
    canonical_phones: Sequence[str],
) -> Tuple[List[Tuple[int, int, int]], bool]:
    """Forced CTC Viterbi alignment using ALLOPHONE_TOKEN_IDS for all 39 English phones."""
    T = int(lp.shape[0])
    L = len(canonical_phones)
    if L == 0:
        return [], False

    norm_phones = [normalize_canonical_phone(p) for p in canonical_phones]

    def _uniform_fallback() -> Tuple[List[Tuple[int, int, int]], bool]:
        spans: List[Tuple[int, int, int]] = []
        for i in range(L):
            c = min(T - 1, max(0, int(round((i + 0.5) * T / max(L, 1)))))
            spans.append((c, c, c))
        return spans, True

    if T < L:
        return _uniform_fallback()

    emit_pos = np.empty((T, L), dtype=np.float64)
    for i, p in enumerate(norm_phones):
        ids = ALLOPHONE_TOKEN_IDS[p]
        if len(ids) == 1:
            emit_pos[:, i] = lp[:, ids[0]]
        else:
            emit_pos[:, i] = np.max(lp[:, list(ids)], axis=1)

    S = 2 * L + 1
    state_emit = np.empty((T, S), dtype=np.float64)
    state_emit[:, 0::2] = lp[:, 0:1]
    state_emit[:, 1::2] = emit_pos

    can_skip2 = np.zeros(S - 2, dtype=bool)
    for i in range(1, L):
        s = 2 * i + 1
        if norm_phones[i] != norm_phones[i - 1]:
            can_skip2[s - 2] = True

    dp = np.full((T, S), -1e9, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int32)
    dp[0, 0] = state_emit[0, 0]
    if S > 1:
        dp[0, 1] = state_emit[0, 1]

    neg_inf = -1e9
    for t in range(1, T):
        prev = dp[t - 1]
        best_val = prev.copy()
        best_prev = np.arange(S, dtype=np.int32)

        cand1 = prev[:-1]
        mask1 = cand1 > best_val[1:]
        best_val[1:] = np.where(mask1, cand1, best_val[1:])
        best_prev[1:] = np.where(mask1, np.arange(S - 1, dtype=np.int32), best_prev[1:])

        cand2 = np.where(can_skip2, prev[:-2], neg_inf)
        mask2 = cand2 > best_val[2:]
        best_val[2:] = np.where(mask2, cand2, best_val[2:])
        best_prev[2:] = np.where(mask2, np.arange(S - 2, dtype=np.int32), best_prev[2:])

        dp[t] = best_val + state_emit[t]
        bp[t] = best_prev

    if dp[T - 1, S - 1] <= -1e8 and dp[T - 1, S - 2] <= -1e8:
        return _uniform_fallback()

    cur = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.empty(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = cur
        cur = int(bp[t, cur])

    spans: List[Tuple[int, int, int]] = []
    for i in range(L):
        state = 2 * i + 1
        frames = np.flatnonzero(path == state)
        if len(frames) == 0:
            return _uniform_fallback()
        vs = int(frames[0])
        ve = int(frames[-1])
        vp = vs + int(np.argmax(emit_pos[vs : ve + 1, i]))
        spans.append((vs, ve, vp))
    return spans, False


def extract_utterance_base_features_v2(
    model: torch.nn.Module,
    wav_tensor: torch.Tensor,
    canonical_phones: Sequence[str],
    inverse_vocab: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Run ONE frozen Base Koel forward pass and extract allophone-aware v2 features.

    Strictly drop-in compatible with existing frozen R0 (`SmallMDDHead`) checkpoints!
    """
    L = len(canonical_phones)
    if L == 0:
        raise ValueError("canonical_phones must be non-empty")

    norm_phones = [normalize_canonical_phone(p) for p in canonical_phones]

    x = wav_tensor.to(device)
    x = F.layer_norm(x, x.shape[1:])
    with torch.inference_mode():
        out = model(x, output_hidden_states=True)
        logits = out.logits[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite Base Koel logits")
        h18 = out.hidden_states[18][0]
        h24 = out.hidden_states[24][0]
        lp_gpu = F.log_softmax(logits, dim=-1)

    lp = lp_gpu.cpu().numpy().astype(np.float32)
    h18_np = h18.cpu().numpy().astype(np.float16)
    h24_np = h24.cpu().numpy().astype(np.float16)
    lp_fp16 = lp.astype(np.float16)
    T = int(lp.shape[0])

    # Greedy CTC collapse + v2 100%-vocab-covered normalization & allophone-aware DP alignment
    emitted = emitted_segments(lp)
    tokens = [inverse_vocab[e["token_id"]] for e in emitted]
    obs_v2, unk_v2, unmapped_v2, obs_to_seg = normalize_koel_tokens_v2(tokens)
    ev = align_and_decide_v2(norm_phones, obs_v2, unknown_indices=unk_v2)
    a_greedy_call = ev["target_calls"]
    a_gap_call = ev["gap_calls"]

    # Allophone-aware CTC Viterbi forced alignment
    vit_spans, vit_uniform_fallback = ctc_viterbi_align_allophone_aware(lp, norm_phones)

    probs = np.exp(lp)
    frame_entropy = -np.sum(probs * lp, axis=-1)
    frames_per_phone = max(float(T) / max(L, 1), 1e-3)

    phone_ids = np.array([PHONE_TO_ID[p] for p in norm_phones], dtype=np.int64)
    prev_phone_ids = np.array(
        [CTX_BOS_ID] + [PHONE_TO_ID[norm_phones[i - 1]] for i in range(1, L)],
        dtype=np.int64,
    )
    next_phone_ids = np.array(
        [PHONE_TO_ID[norm_phones[i + 1]] for i in range(L - 1)] + [CTX_EOS_ID],
        dtype=np.int64,
    )

    anchors: List[Tuple[int, int, int]] = []
    align_tiers: List[int] = []
    bnd_widths: List[int] = []

    for i, t_ev in enumerate(ev["targets"]):
        vs, ve, vp = vit_spans[i]
        status = t_ev["status"]
        obs_idx = t_ev["primary_obs_index"]

        if status in {"PHONE_MATCH_NOT_ACCEPTABILITY", "SUBSTITUTION_HYPOTHESIS"} and obs_idx is not None and obs_idx < len(obs_to_seg):
            seg = emitted[obs_to_seg[obs_idx]]
            a_s, a_e, a_p = int(seg["start"]), int(seg["end"]), int(seg["peak"])
            tier = 0  # clear_greedy
            bw = a_e - a_s + 1
        elif status == "ABSTAIN_MISSING_RECOGNIZED_PHONE":
            tier = 1  # omission_boundary
            a_s, a_e, a_p = vs, ve, vp
            bw = ve - vs + 1
        elif status == "AMBIGUOUS_ALIGNMENT":
            tier = 2  # ambiguous_alignment
            a_s, a_e, a_p = vs, ve, vp
            bw = ve - vs + 1
        else:
            tier = 3  # fallback
            a_s, a_e, a_p = vs, ve, vp
            bw = ve - vs + 1

        a_s = min(T - 1, max(0, a_s))
        a_e = min(T - 1, max(a_s, a_e))
        a_p = min(T - 1, max(0, a_p))
        anchors.append((a_s, a_e, a_p))
        align_tiers.append(tier)
        bnd_widths.append(bw)

    win_indices = np.empty((L, WINDOW_SIZE), dtype=np.int32)
    win_flags = np.empty((L, WINDOW_SIZE, 3), dtype=np.float32)
    h18_vit_mean = np.empty((L, h18_np.shape[1]), dtype=np.float16)
    h24_vit_mean = np.empty((L, h24_np.shape[1]), dtype=np.float16)
    lp_vit_mean = np.empty((L, lp.shape[1]), dtype=np.float16)
    scalars = np.empty((L, len(SCALAR_FEATURE_NAMES)), dtype=np.float32)

    offsets = np.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1, dtype=np.int32)
    all_vocab_ids = np.arange(lp.shape[1], dtype=np.int32)

    for i in range(L):
        a_s, a_e, a_p = anchors[i]
        vs, ve, vp = vit_spans[i]
        tier = align_tiers[i]
        bw = bnd_widths[i]
        t_ev = ev["targets"][i]
        ops = set(t_ev["ops"])
        ac = a_greedy_call[i]

        raw_t = a_p + offsets
        clamped_t = np.clip(raw_t, 0, T - 1)
        win_indices[i] = clamped_t
        win_flags[i, :, 0] = ((raw_t >= a_s) & (raw_t <= a_e)).astype(np.float32)
        win_flags[i, :, 1] = ((raw_t >= vs) & (raw_t <= ve)).astype(np.float32)
        win_flags[i, :, 2] = ((raw_t < 0) | (raw_t >= T)).astype(np.float32)

        h18_vit_mean[i] = h18_np[vs : ve + 1].astype(np.float32).mean(axis=0).astype(np.float16)
        h24_vit_mean[i] = h24_np[vs : ve + 1].astype(np.float32).mean(axis=0).astype(np.float16)
        lp_vit_mean[i] = lp[vs : ve + 1].mean(axis=0).astype(np.float16)

        prev_end = anchors[i - 1][1] if i > 0 else 0
        next_start = anchors[i + 1][0] if i + 1 < L else (T - 1)

        # FIX: Allophone-aware canonical log-prob (max over ALL valid allophone IDs for phone p!)
        p_norm = norm_phones[i]
        can_ids_tuple = ALLOPHONE_TOKEN_IDS[p_norm]
        can_ids_set = set(can_ids_tuple)
        if len(can_ids_tuple) == 1:
            can_lp_t = lp[:, can_ids_tuple[0]]
        else:
            can_lp_t = np.max(lp[:, list(can_ids_tuple)], axis=1)

        other_mask = np.array([c != 0 and c not in can_ids_set for c in all_vocab_ids], dtype=bool)
        other_lp_t = np.max(lp[:, other_mask], axis=1)

        w2_s = max(0, a_p - 2)
        w2_e = min(T - 1, a_p + 2)

        can_lp_peak = float(can_lp_t[a_p])
        can_lp_win_max = float(np.max(can_lp_t[w2_s : w2_e + 1]))
        can_lp_vit_max = float(np.max(can_lp_t[vs : ve + 1]))
        can_lp_vit_mean = float(np.mean(can_lp_t[vs : ve + 1]))
        blank_lp_peak = float(lp[a_p, 0])
        blank_lp_vit_mean = float(np.mean(lp[vs : ve + 1, 0]))
        other_lp_win_max = float(np.max(other_lp_t[w2_s : w2_e + 1]))
        other_lp_vit_max = float(np.max(other_lp_t[vs : ve + 1]))

        scalars[i] = np.array([
            1.0 if ac == "C" else 0.0,
            1.0 if ac == "S" else 0.0,
            1.0 if ac == "O" else 0.0,
            1.0 if ac == "?" else 0.0,
            1.0 if tier == 0 else 0.0,
            1.0 if tier == 1 else 0.0,
            1.0 if tier == 2 else 0.0,
            1.0 if tier == 3 else 0.0,
            float(max(1, len(ops))),
            1.0 if any(op in {"S", "O"} for op in ops) else 0.0,
            1.0 if "C" in ops else 0.0,
            1.0 if "O" in ops else 0.0,
            1.0 if "S" in ops else 0.0,
            float(np.log1p(a_e - a_s + 1)),
            float(np.log1p(ve - vs + 1)),
            float((ve - vs + 1) / frames_per_phone),
            float((a_s - prev_end) / frames_per_phone),
            float((next_start - a_e) / frames_per_phone),
            float(bw / frames_per_phone),
            float((a_p - vp) / frames_per_phone),
            1.0 if i == 0 else 0.0,
            1.0 if i == L - 1 else 0.0,
            float(i / max(L - 1, 1)),
            can_lp_peak,
            can_lp_win_max,
            can_lp_vit_max,
            can_lp_vit_mean,
            blank_lp_peak,
            blank_lp_vit_mean,
            other_lp_win_max,
            other_lp_win_max - can_lp_win_max,
            other_lp_vit_max - can_lp_vit_max,
            float(frame_entropy[a_p]),
            float(np.mean(frame_entropy[vs : ve + 1])),
        ], dtype=np.float32)

    return {
        "n_frames": T,
        "n_targets": L,
        "tokens": tokens,
        "unmapped_tokens": unmapped_v2,
        "a_greedy_call": a_greedy_call,
        "a_gap_call": a_gap_call,
        "h18": h18_np,
        "h24": h24_np,
        "lp": lp_fp16,
        "phone_ids": phone_ids,
        "prev_phone_ids": prev_phone_ids,
        "next_phone_ids": next_phone_ids,
        "align_tiers": np.array(align_tiers, dtype=np.int64),
        "vit_spans": np.array(vit_spans, dtype=np.int32),
        "vit_uniform_fallback": vit_uniform_fallback,
        "win_indices": win_indices,
        "win_flags": win_flags,
        "h18_vit_mean": h18_vit_mean,
        "h24_vit_mean": h24_vit_mean,
        "lp_vit_mean": lp_vit_mean,
        "scalars": scalars,
    }
