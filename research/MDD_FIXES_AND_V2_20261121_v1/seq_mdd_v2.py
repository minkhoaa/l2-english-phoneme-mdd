"""P1 Next-Generation Sequence Multi-Task MDD Architecture (`SeqMDDv2`) + Feature Extractor + Multi-Task Loss.

Addresses ALL architectural & training root causes diagnosed in FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1:
  1. Multi-layer XLSR Scalar Mixing (6 layers: L12, L15, L18, L20, L22, L24 — not just frozen H18/H24)
     + Utterance-level CMVN for speaker/channel normalization.
  2. ZERO discrete greedy-call shortcut features (`a_call_is_*` and `align_tier_*` REMOVED!). Uses only continuous
     allophone-aware CTC log-prob / margin / entropy / duration features + continuous acoustic hidden states.
  3. Utterance-level Sequence Encoder (2-layer BiGRU over full canonical phone sequence 0..L-1) so every target slot
     and every gap slot (0..L) sees neighboring phones, speaking rate, and speaker baseline accent.
  4. Joint 3-Head Multi-Task output in ONE frozen Base Koel forward pass:
       - Target Head (L, 3):          C / S / O
       - Gap Head (L+1, 2):           N / I   (fixes R0's complete blindness to Insertions!)
       - Realized-Phone Head (L, 40): auxiliary supervision (39 ARPAbet + <DEL>) using corpus realized phones!
  5. Multi-Corpus + L1 Native Anchor training loss (L2-ARCTIC + EpaDB TRAIN + TIMIT TRAIN all-C anchor)
     with gap class-weighting and Native-FA anchoring to crush native L1 false alarms.

NOTE: Saved to NEW checkpoint paths only; NEVER modifies any existing R0/SM13/Koel checkpoints.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    CTX_BOS_ID,
    CTX_EOS_ID,
    PHONE_TO_ID,
    WINDOW_RADIUS,
    WINDOW_SIZE,
    normalize_canonical_phone,
)
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import ALLOPHONE_TOKEN_IDS

# 6 evenly-spaced upper-half XLSR layers (covers acoustic-phonetic L12..L18 & lexical-phonetic L20..L24)
XLSR_SELECTED_LAYERS: Tuple[int, ...] = (12, 15, 18, 20, 22, 24)
NUM_XLSR_LAYERS: int = len(XLSR_SELECTED_LAYERS)

# Pure continuous acoustic/alignment scalar features (ZERO discrete a_call_is_* or align_tier_* shortcuts!)
CONTINUOUS_SCALAR_NAMES: Tuple[str, ...] = (
    "vit_dur_log",
    "norm_vit_dur",
    "gap_prev_norm",
    "gap_next_norm",
    "is_bos",
    "is_eos",
    "rel_pos",
    "can_lp_vit_max",
    "can_lp_vit_mean",
    "blank_lp_vit_mean",
    "other_lp_vit_max",
    "frame_margin_vit",
    "entropy_vit_mean",
)

# Pure continuous gap features for all L+1 gap slots (between target i-1 and target i)
GAP_SCALAR_NAMES: Tuple[str, ...] = (
    "gap_dur_log",
    "gap_dur_norm",
    "gap_other_nonblank_lp_max",   # max non-blank log-prob EXCLUDING left/right canonical allophones!
    "gap_any_nonblank_lp_max",
    "gap_blank_lp_mean",
    "gap_entropy_mean",
    "is_utt_start",
    "is_utt_end",
)

NUM_REALIZED_CLASSES = len(ARPA_39) + 1  # 39 ARPAbet phones + <DEL> (39)


def extract_seq_mdd_v2_features(
    koel_model: torch.nn.Module,
    wav_tensor: torch.Tensor,
    canonical_phones: Sequence[str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run ONE frozen Base Koel forward pass and extract compact SeqMDDv2 utterance features.

    Strictly label-free: uses ONLY waveform + canonical_phones + frozen Base Koel.
    """
    L = len(canonical_phones)
    if L == 0:
        raise ValueError("canonical_phones must be non-empty")
    norm_phones = [normalize_canonical_phone(p) for p in canonical_phones]

    x = wav_tensor.to(device)
    x = F.layer_norm(x, x.shape[1:])
    with torch.inference_mode():
        out = koel_model(x, output_hidden_states=True)
        logits = out.logits[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite Koel logits")
        lp = F.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float32)  # (T, 79)
        h_stack = torch.stack(
            [out.hidden_states[idx][0] for idx in XLSR_SELECTED_LAYERS], dim=1
        ).cpu().numpy().astype(np.float16)  # (T, 6, 1024)

    T = int(lp.shape[0])
    vit_spans, _ = ctc_viterbi_align_allophone_aware(lp, norm_phones)

    probs = np.exp(lp)
    frame_entropy = -np.sum(probs * lp, axis=-1)  # (T,)
    frames_per_phone = max(float(T) / max(L, 1), 1e-3)
    all_vocab_ids = np.arange(lp.shape[1], dtype=np.int32)
    offsets = np.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1, dtype=np.int32)

    phone_ids = np.array([PHONE_TO_ID[p] for p in norm_phones], dtype=np.int64)
    prev_phone_ids = np.array(
        [CTX_BOS_ID] + [PHONE_TO_ID[norm_phones[i - 1]] for i in range(1, L)], dtype=np.int64
    )
    next_phone_ids = np.array(
        [PHONE_TO_ID[norm_phones[i + 1]] for i in range(L - 1)] + [CTX_EOS_ID], dtype=np.int64
    )

    h_vit = np.empty((L, NUM_XLSR_LAYERS, 1024), dtype=np.float16)
    h_peak = np.empty((L, NUM_XLSR_LAYERS, 1024), dtype=np.float16)
    lp_win = np.empty((L, WINDOW_SIZE, 79), dtype=np.float16)
    lp_vit = np.empty((L, 79), dtype=np.float16)
    target_scalars = np.empty((L, len(CONTINUOUS_SCALAR_NAMES)), dtype=np.float32)

    for i in range(L):
        vs, ve, vp = vit_spans[i]
        h_vit[i] = h_stack[vs : ve + 1].astype(np.float32).mean(axis=0).astype(np.float16)
        h_peak[i] = h_stack[vp]
        clamped_t = np.clip(vp + offsets, 0, T - 1)
        lp_win[i] = lp[clamped_t].astype(np.float16)
        lp_vit[i] = lp[vs : ve + 1].mean(axis=0).astype(np.float16)

        can_ids = ALLOPHONE_TOKEN_IDS[norm_phones[i]]
        can_set = set(can_ids)
        can_lp_t = lp[:, can_ids[0]] if len(can_ids) == 1 else np.max(lp[:, list(can_ids)], axis=1)
        other_mask = np.array([c != 0 and c not in can_set for c in all_vocab_ids], dtype=bool)
        other_lp_t = np.max(lp[:, other_mask], axis=1)

        prev_ve = vit_spans[i - 1][1] if i > 0 else 0
        next_vs = vit_spans[i + 1][0] if i + 1 < L else (T - 1)

        can_vit_max = float(np.max(can_lp_t[vs : ve + 1]))
        can_vit_mean = float(np.mean(can_lp_t[vs : ve + 1]))
        blank_vit_mean = float(np.mean(lp[vs : ve + 1, 0]))
        other_vit_max = float(np.max(other_lp_t[vs : ve + 1]))

        target_scalars[i] = np.array([
            float(np.log1p(ve - vs + 1)),
            float((ve - vs + 1) / frames_per_phone),
            float((vs - prev_ve) / frames_per_phone),
            float((next_vs - ve) / frames_per_phone),
            1.0 if i == 0 else 0.0,
            1.0 if i == L - 1 else 0.0,
            float(i / max(L - 1, 1)),
            can_vit_max,
            can_vit_mean,
            blank_vit_mean,
            other_vit_max,
            other_vit_max - can_vit_max,
            float(np.mean(frame_entropy[vs : ve + 1])),
        ], dtype=np.float32)

    # Extract gap features for all L+1 gap slots (gap k lies between target k-1 and target k)
    h_gap = np.empty((L + 1, NUM_XLSR_LAYERS, 1024), dtype=np.float16)
    lp_gap_mean = np.empty((L + 1, 79), dtype=np.float16)
    lp_gap_max = np.empty((L + 1, 79), dtype=np.float16)
    gap_scalars = np.empty((L + 1, len(GAP_SCALAR_NAMES)), dtype=np.float32)

    for k in range(L + 1):
        # Raw inter-phone span between previous target's end and next target's start
        if k == 0:
            raw_s, raw_e = 0, vit_spans[0][0] - 1
        elif k == L:
            raw_s, raw_e = vit_spans[L - 1][1] + 1, T - 1
        else:
            raw_s, raw_e = vit_spans[k - 1][1] + 1, vit_spans[k][0] - 1

        raw_dur = max(0, raw_e - raw_s + 1)
        # Always provide a non-empty acoustic inspection span (at least the boundary 2-frame bridge)
        if raw_dur > 0:
            gs, ge = raw_s, raw_e
        else:
            mid = vit_spans[0][0] if k == 0 else (vit_spans[L - 1][1] if k == L else vit_spans[k - 1][1])
            gs = max(0, mid - 1)
            ge = min(T - 1, mid + 1)

        h_gap[k] = h_stack[gs : ge + 1].astype(np.float32).mean(axis=0).astype(np.float16)
        lp_span = lp[gs : ge + 1]  # (span, 79)
        lp_gap_mean[k] = lp_span.mean(axis=0).astype(np.float16)
        lp_gap_max[k] = lp_span.max(axis=0).astype(np.float16)

        # Exclude blank (0) and adjacent canonical allophones (and glottal onset 64 at k==0)
        excl = {0}
        if k > 0:
            excl.update(ALLOPHONE_TOKEN_IDS[norm_phones[k - 1]])
        if k < L:
            excl.update(ALLOPHONE_TOKEN_IDS[norm_phones[k]])
        if k == 0:
            excl.add(64)  # word-initial glottal onset ʔ
        other_g_mask = np.array([c not in excl for c in all_vocab_ids], dtype=bool)
        gap_other_nb_max = float(np.max(lp_span[:, other_g_mask]))
        gap_any_nb_max = float(np.max(lp_span[:, 1:]))
        gap_blank_mean = float(np.mean(lp_span[:, 0]))
        gap_ent_mean = float(np.mean(frame_entropy[gs : ge + 1]))

        gap_scalars[k] = np.array([
            float(np.log1p(raw_dur)),
            float(raw_dur / frames_per_phone),
            gap_other_nb_max,
            gap_any_nb_max,
            gap_blank_mean,
            gap_ent_mean,
            1.0 if k == 0 else 0.0,
            1.0 if k == L else 0.0,
        ], dtype=np.float32)

    return {
        "h_vit": h_vit,
        "h_peak": h_peak,
        "lp_win": lp_win,
        "lp_vit": lp_vit,
        "phone_ids": phone_ids,
        "prev_phone_ids": prev_phone_ids,
        "next_phone_ids": next_phone_ids,
        "target_scalars": target_scalars,
        "h_gap": h_gap,
        "lp_gap_mean": lp_gap_mean,
        "lp_gap_max": lp_gap_max,
        "gap_scalars": gap_scalars,
    }


class SeqMDDv2(nn.Module):
    """Utterance-level Multi-Layer Sequence MDD model with joint Target, Gap, and Realized-Phone heads."""

    def __init__(
        self,
        *,
        num_layers: int = NUM_XLSR_LAYERS,
        h_dim: int = 1024,
        vocab_dim: int = 79,
        target_scalar_dim: int = len(CONTINUOUS_SCALAR_NAMES),
        gap_scalar_dim: int = len(GAP_SCALAR_NAMES),
        d_model: int = 192,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        # 1. Differentiable scalar mixing over 6 XLSR layers (separate weights for target vs gap)
        self.target_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.target_layer_scale = nn.Parameter(torch.ones(1))
        self.gap_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gap_layer_scale = nn.Parameter(torch.ones(1))

        # 2. Phonetic canonical embeddings
        self.phone_emb = nn.Embedding(len(ARPA_39), 64)
        self.ctx_emb = nn.Embedding(CTX_EOS_ID + 1, 32)

        # 3. Acoustic & CTC projections
        self.proj_h_vit = nn.Sequential(nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_h_peak = nn.Sequential(nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_lp_win = nn.Linear(vocab_dim, 64)
        self.pos_emb = nn.Embedding(WINDOW_SIZE, 16)
        self.win_fuse = nn.Sequential(
            nn.Linear(64 + 16, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.query_proj = nn.Linear(128, d_model)
        self.local_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.10, batch_first=True)

        self.proj_lp_vit = nn.Sequential(nn.Linear(vocab_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.target_scalar_proj = nn.Sequential(
            nn.Linear(target_scalar_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
        )
        slot_in_dim = d_model * 3 + 64 + 128 + 48
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(slot_in_dim),
            nn.Dropout(dropout),
            nn.Linear(slot_in_dim, d_model),
            nn.GELU(),
        )

        # 4. Utterance-level Sequence Encoder (2-layer BiGRU over full canonical sequence 0..L-1)
        self.seq_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.seq_norm = nn.LayerNorm(d_model)

        # 5. Target Head (C/S/O) + Auxiliary Realized-Phone Head (39 + <DEL>)
        self.target_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 3),
        )
        self.realized_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_REALIZED_CLASSES),
        )

        # 6. Gap Head (N/I) combining gap acoustic span + lp_gap_mean/max + left/right sequence states
        self.gap_ac_proj = nn.Sequential(
            nn.Linear(h_dim + vocab_dim * 2 + gap_scalar_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gap_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),
        )
        self.bos_state = nn.Parameter(torch.zeros(1, 1, d_model))
        self.eos_state = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("pos_ids", torch.arange(WINDOW_SIZE, dtype=torch.long), persistent=False)

    def _mix_layers(
        self, h_layers: torch.Tensor, logits: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        w = torch.softmax(logits, dim=0)
        return torch.einsum("l,...ld->...d", w, h_layers) * scale

    def forward(
        self,
        *,
        h_vit: torch.Tensor,           # (B, L, num_layers, h_dim)
        h_peak: torch.Tensor,          # (B, L, num_layers, h_dim)
        lp_win: torch.Tensor,          # (B, L, W, 79)
        lp_vit: torch.Tensor,          # (B, L, 79)
        phone_ids: torch.Tensor,       # (B, L)
        prev_phone_ids: torch.Tensor,  # (B, L)
        next_phone_ids: torch.Tensor,  # (B, L)
        target_scalars: torch.Tensor,  # (B, L, 13)
        h_gap: torch.Tensor,           # (B, L+1, num_layers, h_dim)
        lp_gap_mean: torch.Tensor,     # (B, L+1, 79)
        lp_gap_max: torch.Tensor,      # (B, L+1, 79)
        gap_scalars: torch.Tensor,     # (B, L+1, 8)
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L, W, _ = lp_win.shape

        # Multi-layer scalar mixing + utterance-level masked CMVN (speaker/channel normalization)
        hv = self._mix_layers(h_vit, self.target_layer_logits, self.target_layer_scale)
        hp = self._mix_layers(h_peak, self.target_layer_logits, self.target_layer_scale)

        if lengths is not None:
            mask = (torch.arange(L, device=hv.device)[None, :] < lengths[:, None]).float().unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            utt_mean = (hv * mask).sum(dim=1, keepdim=True) / denom
            utt_var = (((hv - utt_mean) ** 2) * mask).sum(dim=1, keepdim=True) / denom
            utt_std = utt_var.sqrt().clamp(min=1e-4)
        else:
            utt_mean = hv.mean(dim=1, keepdim=True)
            utt_std = hv.std(dim=1, keepdim=True).clamp(min=1e-4)

        hv = (hv - utt_mean) / utt_std
        hp = (hp - utt_mean) / utt_std

        e_phone = torch.cat(
            [self.phone_emb(phone_ids), self.ctx_emb(prev_phone_ids), self.ctx_emb(next_phone_ids)],
            dim=-1,
        )  # (B, L, 128)

        pos = self.pos_emb(self.pos_ids).view(1, 1, W, 16).expand(B, L, -1, -1)
        f_win = self.win_fuse(torch.cat([self.proj_lp_win(lp_win), pos], dim=-1))
        q = self.query_proj(e_phone).reshape(B * L, 1, self.d_model)
        kv = f_win.reshape(B * L, W, self.d_model)
        h_attn, _ = self.local_attn(q, kv, kv, need_weights=False)
        h_attn = h_attn.reshape(B, L, self.d_model)

        slot_x = self.slot_proj(
            torch.cat(
                [
                    self.proj_h_vit(hv),
                    self.proj_h_peak(hp),
                    h_attn,
                    self.proj_lp_vit(lp_vit),
                    e_phone,
                    self.target_scalar_proj(target_scalars),
                ],
                dim=-1,
            )
        )

        # Utterance-level BiGRU sequence context
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                slot_x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            seq_out, _ = self.seq_encoder(packed)
            seq_out, _ = nn.utils.rnn.pad_packed_sequence(seq_out, batch_first=True, total_length=L)
        else:
            seq_out, _ = self.seq_encoder(slot_x)
        seq_h = self.seq_norm(seq_out + slot_x)

        target_logits = self.target_head(seq_h)
        realized_logits = self.realized_head(seq_h)

        # Gap Head over all L+1 gaps
        hg = self._mix_layers(h_gap, self.gap_layer_logits, self.gap_layer_scale)
        hg = (hg - utt_mean) / utt_std
        gap_ac = self.gap_ac_proj(torch.cat([hg, lp_gap_mean, lp_gap_max, gap_scalars], dim=-1))

        bos = self.bos_state.expand(B, 1, -1)
        eos = self.eos_state.expand(B, 1, -1)
        left_ctx = torch.cat([bos, seq_h], dim=1)
        right_ctx = torch.cat([seq_h, eos], dim=1)
        gap_logits = self.gap_head(torch.cat([left_ctx, gap_ac, right_ctx], dim=-1))

        return {
            "target_logits": target_logits,
            "gap_logits": gap_logits,
            "realized_logits": realized_logits,
        }


def compute_multitask_mdd_loss(
    outputs: Dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    gap_labels: torch.Tensor,
    realized_labels: Optional[torch.Tensor] = None,
    is_native_l1: Optional[torch.Tensor] = None,
    *,
    gap_pos_weight: float = 8.0,
    lambda_gap: float = 0.6,
    lambda_realized: float = 0.25,
    native_fa_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Joint multi-task loss with L1 Native False-Alarm anchoring and weighted Gap CE."""
    t_logits = outputs["target_logits"]
    g_logits = outputs["gap_logits"]

    t_ce = F.cross_entropy(t_logits.reshape(-1, 3), target_labels.reshape(-1), ignore_index=-100, reduction="none")
    t_ce = t_ce.view_as(target_labels)
    valid_t = target_labels != -100
    t_weights = torch.ones_like(t_ce)
    if is_native_l1 is not None:
        t_weights = torch.where(is_native_l1[:, None], torch.full_like(t_weights, native_fa_weight), t_weights)
    loss_target = (t_ce * t_weights * valid_t).sum() / valid_t.sum().clamp(min=1)

    gw = torch.tensor([1.0, gap_pos_weight], device=g_logits.device, dtype=g_logits.dtype)
    loss_gap = F.cross_entropy(g_logits.reshape(-1, 2), gap_labels.reshape(-1), weight=gw, ignore_index=-100)

    if realized_labels is not None and (realized_labels != -100).any():
        loss_real = F.cross_entropy(
            outputs["realized_logits"].reshape(-1, NUM_REALIZED_CLASSES),
            realized_labels.reshape(-1),
            ignore_index=-100,
        )
    else:
        loss_real = torch.zeros((), device=t_logits.device)

    total = loss_target + lambda_gap * loss_gap + lambda_realized * loss_real
    return {
        "loss": total,
        "loss_target": loss_target.detach(),
        "loss_gap": loss_gap.detach(),
        "loss_realized": loss_real.detach(),
    }
