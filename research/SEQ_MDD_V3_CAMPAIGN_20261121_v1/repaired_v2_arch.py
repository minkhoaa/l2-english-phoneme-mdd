"""Code fixes #1, #2, #3 on top of LengthAwareV2 (repaired prompt + EOS contract).

Fixes targeted from DEEP_FORENSIC_DIAGNOSIS.json (read-only over frozen Koel cache):
  Fix #1 (Scale, Norm, CMVN, Residual balance):
    - Per-layer LayerNorm over XLSR hidden states BEFORE softmax layer mixing
      (eliminates 3x L12-vs-L24 norm dominance).
    - Drop short-utterance peak CMVN and cross-layer CMVN on h_gap; use feature
      LayerNorm independently per branch (batch-invariant, length-invariant).
    - Map raw CTC log-probs (mean -17.2, L2 156.6) via [exp(lp), clip(lp,-10,0)/10]
      in [-1, 1] and normalize every branch BEFORE concatenation (fixes 50x-56x
      scale dominance in win_fuse and gap_ac_proj).
    - Add LayerNorm at the end of slot_proj and on BiGRU output before residual sum
      so slot_x no longer overwhelms BiGRU by 3x.
  Fix #2 (1-frame CTC spike -> 5-frame continuous acoustic window keeping h_peak):
    - Retain h_peak = h[vp] separately (never overwritten by mean pooling).
    - Extract 5 consecutive frames h_win = h[vp + (-2, -1, 0, +1, +2)] across all
      6 XLSR layers and fuse via phone-conditioned multi-head cross-attention
      (replaces redundant h_vit == h_peak).
  Fix #3 (Anti-modality-collapse + TIMIT training loss hygiene):
    - Slot-level modality dropout on target_scalars (and gap_scalars/lp_gap) during
      training only, forcing gradient flow into h_win, h_peak, lp_win, and BiGRU.
    - Apply native_fa_weight to clean 'C' slots of L1 TIMIT FIT rows (which had
      is_native_l1=False in archived manifests), and mask synthetic TIMIT Gap 'I'
      (61% of FIT 'I' labels) while keeping TIMIT Gap 'N' negatives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    CTX_BOS_ID,
    CTX_EOS_ID,
    PHONE_TO_ID,
    WINDOW_SIZE,
)
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import ALLOPHONE_TOKEN_IDS
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    CONTINUOUS_SCALAR_NAMES,
    GAP_SCALAR_NAMES,
    NUM_REALIZED_CLASSES,
    NUM_XLSR_LAYERS,
)

LP_OFFSETS = np.arange(-4, 5, dtype=np.int32)  # 9 frames (-80..+80 ms)
H_WIN_OFFSETS = np.arange(-2, 3, dtype=np.int32)  # 5 consecutive frames (-40..+40 ms)
H_WIN_SIZE = len(H_WIN_OFFSETS)


@dataclass(frozen=True)
class CodeFixConfig:
    mode: str = "fix123_full"  # fix12_arch | fix123_full
    d_model: int = 192
    dropout: float = 0.30
    scalar_drop_p: float = 0.35
    gap_lp_drop_p: float = 0.25
    fix_timit_loss: bool = True
    native_fa_weight: float = 1.5
    gap_pos_weight: float = 8.0
    lambda_gap: float = 0.6
    lambda_realized: float = 0.25

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def pool_features_codefix(h: np.ndarray, lp: np.ndarray, phones) -> Dict[str, np.ndarray]:
    """Label-free feature pooling keeping exact V2 Viterbi spans + 5-frame h_win."""
    T, L = len(lp), len(phones)
    if h.ndim != 3 or len(h) != T or not L or not T or lp.shape[1] != 79:
        raise ValueError("Frame/slot shape mismatch")
    spans, _ = ctc_viterbi_align_allophone_aware(lp, phones)
    s, e, peaks = np.asarray(spans).T
    norm = max(T / L, 1e-3)
    ids = [PHONE_TO_ID[p] for p in phones]

    t_lp_win = np.clip(peaks[:, None] + LP_OFFSETS[None, :], 0, T - 1)
    t_h_win = np.clip(peaks[:, None] + H_WIN_OFFSETS[None, :], 0, T - 1)

    h_peak = h[peaks]  # (L, 6, 1024) fp16 — exact CTC peak preserved
    h_win = h[t_h_win]  # (L, 5, 6, 1024) fp16 — 5 consecutive frames
    lp_win = lp[t_lp_win].astype(np.float16)
    lp_vit = np.stack([lp[a : b + 1].mean(0).astype(np.float16) for a, b in zip(s, e)])

    ent = -np.sum(np.exp(lp) * lp, axis=-1)
    tokens = np.arange(lp.shape[1])
    lp_by_phone = {}
    for p in set(phones):
        can = ALLOPHONE_TOKEN_IDS[p]
        other = (tokens != 0) & ~np.isin(tokens, can)
        lp_by_phone[p] = (
            lp[:, can[0]] if len(can) == 1 else lp[:, list(can)].max(1),
            lp[:, other].max(1),
        )

    target_sc = []
    for i, (vs, ve, _) in enumerate(spans):
        z = slice(vs, ve + 1)
        a, b = lp_by_phone[phones[i]]
        mx = a[z].max()
        omx = b[z].max()
        target_sc.append(
            [
                np.log1p(ve - vs + 1),
                (ve - vs + 1) / norm,
                (vs - (e[i - 1] if i else 0)) / norm,
                ((s[i + 1] if i + 1 < L else T - 1) - ve) / norm,
                float(not i),
                float(i == L - 1),
                i / max(L - 1, 1),
                mx,
                a[z].mean(),
                lp[z, 0].mean(),
                omx,
                omx - mx,
                ent[z].mean(),
            ]
        )

    hh, mean_g, mx_g, gsc = [], [], [], []
    for k in range(L + 1):
        a = int(e[k - 1] + 1) if k else 0
        b = int(s[k] - 1) if k < L else T - 1
        duration = max(0, b - a + 1)
        if duration > 0:
            gs, ge = a, b
        else:
            bridge = int(e[k - 1]) if k else int(s[0])
            gs, ge = max(0, bridge - 1), min(T - 1, bridge + 1)
        span = lp[gs : ge + 1]
        hh.append(h[gs : ge + 1].astype(np.float32).mean(0).astype(np.float16))
        mean_g.append(span.mean(0))
        mx_g.append(span.max(0))
        excl = {0}
        if k:
            excl.update(ALLOPHONE_TOKEN_IDS[phones[k - 1]])
        if k < L:
            excl.update(ALLOPHONE_TOKEN_IDS[phones[k]])
        if not k:
            excl.add(64)
        comp = ~np.isin(tokens, list(excl))
        gsc.append(
            [
                np.log1p(duration),
                duration / norm,
                float(span[:, comp].max()),
                float(span[:, 1:].max()),
                float(span[:, 0].mean()),
                float(ent[gs : ge + 1].mean()),
                float(k == 0),
                float(k == L),
            ]
        )

    return {
        "h_peak": h_peak,
        "h_win": h_win,
        "lp_win": lp_win,
        "lp_vit": lp_vit,
        "phone_ids": np.asarray(ids, dtype=np.int64),
        "prev_phone_ids": np.asarray([CTX_BOS_ID, *ids[:-1]], dtype=np.int64),
        "next_phone_ids": np.asarray([*ids[1:], CTX_EOS_ID], dtype=np.int64),
        "target_scalars": np.asarray(target_sc, dtype=np.float32),
        "h_gap": np.asarray(hh, dtype=np.float16),
        "lp_gap_mean": np.asarray(mean_g, dtype=np.float16),
        "lp_gap_max": np.asarray(mx_g, dtype=np.float16),
        "gap_scalars": np.asarray(gsc, dtype=np.float32),
    }


def scale_lp(lp: torch.Tensor) -> torch.Tensor:
    """Map raw log-probs (~[-30, 0]) into well-scaled [prob, clamped_lp/10] in [-1, 1]."""
    return torch.cat([lp.exp(), lp.clamp(min=-10.0, max=0.0) * 0.1], dim=-1)


class RepairedCodeFixV2(nn.Module):
    """Length-aware SeqMDDv2 with Fixes #1 (scale/norm/residual), #2 (5-frame h_win), #3 (modality drop)."""

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
        scalar_drop_p: float = 0.0,
        gap_lp_drop_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.d_model = d_model
        self.scalar_drop_p = float(scalar_drop_p)
        self.gap_lp_drop_p = float(gap_lp_drop_p)
        lp_feat_dim = vocab_dim * 2  # [prob, clipped_lp/10]

        # 1. Per-layer-normalized scalar mixing (separate for target vs gap)
        self.target_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.target_layer_scale = nn.Parameter(torch.ones(1))
        self.gap_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gap_layer_scale = nn.Parameter(torch.ones(1))
        self.target_h_norm = nn.LayerNorm(h_dim)
        self.gap_h_norm = nn.LayerNorm(h_dim)

        # 2. Phonetic context embeddings (normalized to match other branches)
        self.phone_emb = nn.Embedding(len(ARPA_39), 64)
        self.ctx_emb = nn.Embedding(CTX_EOS_ID + 1, 32)
        self.phone_ctx_norm = nn.LayerNorm(128)

        # 3a. Exact peak acoustic projection (preserved intact!)
        self.proj_h_peak = nn.Sequential(
            nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        # 3b. Fix #2: 5-frame continuous acoustic window (vp-2..vp+2) with cross-attention
        self.proj_h_win = nn.Sequential(
            nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.h_pos_emb = nn.Embedding(H_WIN_SIZE, d_model)
        self.h_win_query = nn.Linear(128, d_model)
        self.h_win_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=0.10, batch_first=True
        )
        self.h_win_out_norm = nn.LayerNorm(d_model)

        # 3c. Fix #1: Well-scaled 9-frame CTC posterior window (balanced with pos_emb!)
        self.proj_lp_win = nn.Sequential(
            nn.Linear(lp_feat_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.pos_emb = nn.Embedding(WINDOW_SIZE, 16)
        self.pos_emb_norm = nn.LayerNorm(16)
        self.win_fuse = nn.Sequential(
            nn.Linear(64 + 16, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.query_proj = nn.Linear(128, d_model)
        self.local_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=0.10, batch_first=True
        )
        self.lp_win_out_norm = nn.LayerNorm(d_model)

        self.proj_lp_vit = nn.Sequential(
            nn.Linear(lp_feat_dim, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.target_scalar_proj = nn.Sequential(
            nn.Linear(target_scalar_dim, 48), nn.LayerNorm(48), nn.GELU()
        )

        # Balanced slot projection WITH trailing LayerNorm so slot_x does not overwhelm BiGRU
        slot_in_dim = d_model * 3 + 64 + 128 + 48
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(slot_in_dim),
            nn.Dropout(dropout),
            nn.Linear(slot_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 4. Utterance-level BiGRU + balanced residual norm
        self.seq_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.gru_out_norm = nn.LayerNorm(d_model)
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

        # 6. Fix #1: Balanced Gap Head with separate LayerNorm per modality BEFORE concat
        self.gap_h_proj = nn.Sequential(
            nn.Linear(h_dim, 96), nn.LayerNorm(96), nn.GELU()
        )
        self.gap_lp_proj = nn.Sequential(
            nn.Linear(lp_feat_dim * 2, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.gap_sc_proj = nn.Sequential(
            nn.Linear(gap_scalar_dim, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.gap_ac_fuse = nn.Sequential(
            nn.Linear(96 + 64 + 32, d_model),
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
        self.register_buffer(
            "pos_ids", torch.arange(WINDOW_SIZE, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "h_pos_ids", torch.arange(H_WIN_SIZE, dtype=torch.long), persistent=False
        )

    def _mix_layers(
        self, h_layers: torch.Tensor, logits: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        # Per-layer feature normalization before softmax mixing so L24 (L2~640) does not
        # drown out L12 (L2~220). h_layers is frozen input, so computing its parameter-free
        # layer_norm under no_grad avoids storing a huge backward graph for (B,L,5,6,1024).
        if h_layers.requires_grad:
            h_normed = F.layer_norm(h_layers.float(), (self.h_dim,))
        else:
            with torch.no_grad():
                h_normed = F.layer_norm(h_layers.float(), (self.h_dim,))
        w = torch.softmax(logits, dim=0)
        return torch.einsum("l,...ld->...d", w, h_normed) * scale

    def _modality_drop(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if not self.training or p <= 0.0:
            return x
        keep = (
            torch.rand(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype) >= p
        ).to(x.dtype)
        return x * (keep / (1.0 - p))

    def forward(
        self,
        *,
        h_peak: torch.Tensor,          # (B, L, num_layers, h_dim)
        h_win: torch.Tensor,           # (B, L, 5, num_layers, h_dim)
        lp_win: torch.Tensor,          # (B, L, 9, 79)
        lp_vit: torch.Tensor,          # (B, L, 79)
        phone_ids: torch.Tensor,       # (B, L)
        prev_phone_ids: torch.Tensor,  # (B, L)
        next_phone_ids: torch.Tensor,  # (B, L)
        target_scalars: torch.Tensor,  # (B, L, 13)
        h_gap: torch.Tensor,           # (B, L+1, num_layers, h_dim)
        lp_gap_mean: torch.Tensor,     # (B, L+1, 79)
        lp_gap_max: torch.Tensor,      # (B, L+1, 79)
        gap_scalars: torch.Tensor,     # (B, L+1, 8)
        lengths: torch.Tensor,         # (B,)
    ) -> Dict[str, torch.Tensor]:
        B, L, W, _ = lp_win.shape

        # Fix #1: Feature-level LayerNorm (no short-utterance CMVN, no cross-layer CMVN)
        hp = self.target_h_norm(
            self._mix_layers(h_peak, self.target_layer_logits, self.target_layer_scale)
        )
        hw = self.target_h_norm(
            self._mix_layers(h_win, self.target_layer_logits, self.target_layer_scale)
        )

        e = self.phone_ctx_norm(
            torch.cat(
                [
                    self.phone_emb(phone_ids),
                    self.ctx_emb(prev_phone_ids),
                    self.ctx_emb(next_phone_ids),
                ],
                dim=-1,
            )
        )

        # Fix #2: 5-frame continuous acoustic attention (keeps exact h_peak separate!)
        hw_f = self.proj_h_win(hw) + self.h_pos_emb(self.h_pos_ids).view(
            1, 1, H_WIN_SIZE, self.d_model
        )
        qh = self.h_win_query(e).reshape(B * L, 1, self.d_model)
        kvh = hw_f.reshape(B * L, H_WIN_SIZE, self.d_model)
        h_win_attn, _ = self.h_win_attn(qh, kvh, kvh, need_weights=False)
        h_win_attn = self.h_win_out_norm(h_win_attn.reshape(B, L, self.d_model))

        # Fix #1: Well-scaled 9-frame CTC posterior window attention
        pos = self.pos_emb_norm(self.pos_emb(self.pos_ids)).view(1, 1, W, 16).expand(
            B, L, -1, -1
        )
        f_lp = self.win_fuse(
            torch.cat([self.proj_lp_win(scale_lp(lp_win)), pos], dim=-1)
        )
        ql = self.query_proj(e).reshape(B * L, 1, self.d_model)
        kvl = f_lp.reshape(B * L, W, self.d_model)
        lp_attn, _ = self.local_attn(ql, kvl, kvl, need_weights=False)
        lp_attn = self.lp_win_out_norm(lp_attn.reshape(B, L, self.d_model))

        # Fix #3: Modality dropout on target_scalars during training to prevent collapse
        t_sc = self._modality_drop(
            self.target_scalar_proj(target_scalars), self.scalar_drop_p
        )
        slot_x = self.slot_proj(
            torch.cat(
                [
                    self.proj_h_peak(hp),
                    h_win_attn,
                    lp_attn,
                    self.proj_lp_vit(scale_lp(lp_vit)),
                    e,
                    t_sc,
                ],
                dim=-1,
            )
        )

        # Balanced BiGRU residual
        packed = nn.utils.rnn.pack_padded_sequence(
            slot_x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        seq_out, _ = self.seq_encoder(packed)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(
            seq_out, batch_first=True, total_length=L
        )
        seq = self.seq_norm(self.gru_out_norm(seq_out) + slot_x)

        target_logits = self.target_head(seq)
        realized_logits = self.realized_head(seq)

        # Fix #1: Independent Gap LayerNorm + balanced per-modality projections
        hg = self.gap_h_norm(
            self._mix_layers(h_gap, self.gap_layer_logits, self.gap_layer_scale)
        )
        g_h = self.gap_h_proj(hg)
        g_lp = self._modality_drop(
            self.gap_lp_proj(
                torch.cat([scale_lp(lp_gap_mean), scale_lp(lp_gap_max)], dim=-1)
            ),
            self.gap_lp_drop_p,
        )
        g_sc = self._modality_drop(
            self.gap_sc_proj(gap_scalars), self.scalar_drop_p
        )
        gap_ac = self.gap_ac_fuse(torch.cat([g_h, g_lp, g_sc], dim=-1))

        # Preserve exact LengthAwareV2 per-utterance EOS contract
        left = torch.cat([self.bos_state.expand(B, 1, -1), seq], dim=1)
        eos_idx = lengths.view(B, 1, 1).expand(-1, -1, self.d_model)
        right = F.pad(seq, (0, 0, 0, 1)).scatter(
            1, eos_idx, self.eos_state.expand(B, 1, -1)
        )
        gap_logits = self.gap_head(torch.cat([left, gap_ac, right], dim=-1))

        return {
            "target_logits": target_logits,
            "gap_logits": gap_logits,
            "realized_logits": realized_logits,
        }


def compute_codefix_loss(
    outputs: Dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    gap_labels: torch.Tensor,
    realized_labels: Optional[torch.Tensor],
    is_l1_domain: torch.Tensor,
    cfg: CodeFixConfig,
) -> Dict[str, torch.Tensor]:
    """Multi-task loss with optional Fix #3 (TIMIT clean-C weighting + synthetic Gap-I mask)."""
    t_logits = outputs["target_logits"]
    g_logits = outputs["gap_logits"]

    t_ce = F.cross_entropy(
        t_logits.reshape(-1, 3),
        target_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(target_labels)
    valid_t = target_labels != -100
    t_w = torch.ones_like(t_ce)
    if cfg.fix_timit_loss:
        # Weight only genuine native clean 'C' slots on L1 domain; keep anti-shortcut S/O at 1.0
        clean_l1_c = is_l1_domain[:, None] & (target_labels == 0)
        t_w = torch.where(
            clean_l1_c, torch.full_like(t_w, cfg.native_fa_weight), t_w
        )
    loss_target = (t_ce * t_w * valid_t).sum() / valid_t.sum().clamp_min(1)

    g_y = gap_labels
    if cfg.fix_timit_loss:
        # Mask synthetic deleted-phone Gap 'I' on L1 TIMIT FIT; retain L1 Gap 'N' negatives
        g_y = torch.where(
            is_l1_domain[:, None] & (gap_labels == 1),
            torch.full_like(gap_labels, -100),
            gap_labels,
        )
    gw = torch.tensor(
        [1.0, cfg.gap_pos_weight], device=g_logits.device, dtype=g_logits.dtype
    )
    if (g_y != -100).any():
        loss_gap = F.cross_entropy(
            g_logits.reshape(-1, 2), g_y.reshape(-1), weight=gw, ignore_index=-100
        )
    else:
        loss_gap = g_logits.sum() * 0.0

    if realized_labels is not None and (realized_labels != -100).any():
        loss_real = F.cross_entropy(
            outputs["realized_logits"].reshape(-1, NUM_REALIZED_CLASSES),
            realized_labels.reshape(-1),
            ignore_index=-100,
        )
    else:
        loss_real = torch.zeros((), device=t_logits.device)

    total = (
        loss_target
        + cfg.lambda_gap * loss_gap
        + cfg.lambda_realized * loss_real
    )
    return {
        "loss": total,
        "loss_target": loss_target.detach(),
        "loss_gap": loss_gap.detach(),
        "loss_realized": loss_real.detach(),
    }
