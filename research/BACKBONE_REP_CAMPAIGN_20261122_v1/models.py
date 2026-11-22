"""Neural architectures for BACKBONE_REP_CAMPAIGN_20261122_v1 (Stages 2, 3, 4, 5).

All architectures keep Base Koel CTC 100% frozen (0.000% PER drift) and preserve exact
length-aware EOS handling and per-layer normalization from SingleFactorV2 (iso_no_hwin).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    CTX_EOS_ID,
    WINDOW_SIZE,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    CONTINUOUS_SCALAR_NAMES,
    GAP_SCALAR_NAMES,
    NUM_REALIZED_CLASSES,
    NUM_XLSR_LAYERS,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.repaired_v2_arch import scale_lp

from .articulatory import NUM_AF

FRAME_WIN_TARGET = 17  # +-8 frames (340 ms at 50 Hz)
FRAME_WIN_GAP = 17     # up to 17 frames across inter-peak gap


def _mix_layers(x: torch.Tensor, logits: torch.Tensor, scale: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Per-layer LayerNorm before learnable softmax scalar mixing.

    x: (..., L, D) -> returns (..., D) float32
    """
    w = torch.softmax(logits, dim=0)
    x_norm = norm(x.float())
    # Contract along second-to-last dim L
    return scale * torch.einsum("...ld,l->...d", x_norm, w)


# ---------------------------------------------------------------------------
# Stage 2: DualRepSingleFactorV2 (Koel + 1 or 2 Complementary SSL Backbones)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DualRepConfig:
    arm: str
    ext1_layers: int
    ext1_dim: int
    ext2_layers: int = 0
    ext2_dim: int = 0
    aux_scalar_dim: int = 0
    d_model: int = 192
    dropout: float = 0.30
    scalar_dropout_p: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DualRepSingleFactorV2(nn.Module):
    """SingleFactorV2 (iso_no_hwin) augmented with 1–2 complementary acoustic encoders."""

    def __init__(
        self,
        cfg: DualRepConfig,
        *,
        num_layers: int = NUM_XLSR_LAYERS,
        h_dim: int = 1024,
        vocab_dim: int = 79,
        target_scalar_dim: int = len(CONTINUOUS_SCALAR_NAMES),
        gap_scalar_dim: int = len(GAP_SCALAR_NAMES),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        dropout = cfg.dropout
        lp_feat_dim = vocab_dim * 2

        # 1. Koel layer mixing
        self.target_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.target_layer_scale = nn.Parameter(torch.ones(1))
        self.target_h_norm = nn.LayerNorm(h_dim)

        self.gap_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gap_layer_scale = nn.Parameter(torch.ones(1))
        self.gap_h_norm = nn.LayerNorm(h_dim)

        # 2. Phone context
        self.phone_emb = nn.Embedding(len(ARPA_39), 64)
        self.ctx_emb = nn.Embedding(CTX_EOS_ID + 1, 32)
        self.phone_ctx_norm = nn.LayerNorm(128)

        # 3. Koel target branches (identical to iso_no_hwin)
        self.proj_h_peak = nn.Sequential(nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_h_vit = nn.Sequential(nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_lp_win = nn.Sequential(nn.Linear(lp_feat_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.pos_emb = nn.Embedding(WINDOW_SIZE, 16)
        self.pos_emb_norm = nn.LayerNorm(16)
        self.win_fuse = nn.Sequential(
            nn.Linear(64 + 16, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.query_proj = nn.Linear(128, d_model)
        self.local_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.10, batch_first=True)
        self.lp_win_out_norm = nn.LayerNorm(d_model)
        self.proj_lp_vit = nn.Sequential(nn.Linear(lp_feat_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.target_scalar_proj = nn.Sequential(
            nn.Linear(target_scalar_dim, 48), nn.LayerNorm(48), nn.GELU()
        )

        # 4. External Backbone #1 (target + gap, clean gap_* prefix for Phase-2 freezing)
        self.ext1_target_logits = nn.Parameter(torch.zeros(cfg.ext1_layers))
        self.ext1_target_scale = nn.Parameter(torch.ones(1))
        self.ext1_target_norm = nn.LayerNorm(cfg.ext1_dim)
        self.proj_ext1_target = nn.Sequential(
            nn.Linear(cfg.ext1_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.gap_ext1_logits = nn.Parameter(torch.zeros(cfg.ext1_layers))
        self.gap_ext1_scale = nn.Parameter(torch.ones(1))
        self.gap_ext1_norm = nn.LayerNorm(cfg.ext1_dim)
        self.gap_ext1_proj = nn.Sequential(
            nn.Linear(cfg.ext1_dim * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )

        slot_in_dim = d_model * 3 + 64 + 128 + 48 + d_model
        gap_ext_dim = 96

        # 5. Optional External Backbone #2
        if cfg.ext2_layers > 0 and cfg.ext2_dim > 0:
            self.ext2_target_logits = nn.Parameter(torch.zeros(cfg.ext2_layers))
            self.ext2_target_scale = nn.Parameter(torch.ones(1))
            self.ext2_target_norm = nn.LayerNorm(cfg.ext2_dim)
            self.proj_ext2_target = nn.Sequential(
                nn.Linear(cfg.ext2_dim * 3, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
            )
            self.gap_ext2_logits = nn.Parameter(torch.zeros(cfg.ext2_layers))
            self.gap_ext2_scale = nn.Parameter(torch.ones(1))
            self.gap_ext2_norm = nn.LayerNorm(cfg.ext2_dim)
            self.gap_ext2_proj = nn.Sequential(
                nn.Linear(cfg.ext2_dim * 2, 64),
                nn.LayerNorm(64),
                nn.GELU(),
            )
            slot_in_dim += d_model // 2
            gap_ext_dim += 64
        else:
            self.proj_ext2_target = None
            self.gap_ext2_proj = None

        # 6. Optional Charsiu + Prototype scalars
        if cfg.aux_scalar_dim > 0:
            self.aux_scalar_proj = nn.Sequential(
                nn.Linear(cfg.aux_scalar_dim, 48), nn.LayerNorm(48), nn.GELU()
            )
            slot_in_dim += 48
        else:
            self.aux_scalar_proj = None

        self.slot_proj = nn.Sequential(
            nn.LayerNorm(slot_in_dim),
            nn.Dropout(dropout),
            nn.Linear(slot_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 7. Utterance BiGRU + balanced residual
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

        # 8. Target, Realized, and Auxiliary AF heads
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
        self.af_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_AF),
        )

        # 9. Gap head (all parameters begin with gap_ for Phase-2 isolation)
        self.gap_h_proj = nn.Sequential(nn.Linear(h_dim, 96), nn.LayerNorm(96), nn.GELU())
        self.gap_lp_proj = nn.Sequential(nn.Linear(lp_feat_dim * 2, 96), nn.LayerNorm(96), nn.GELU())
        self.gap_scalar_proj = nn.Sequential(nn.Linear(gap_scalar_dim, 48), nn.LayerNorm(48), nn.GELU())
        self.gap_eos_emb = nn.Parameter(torch.zeros(d_model))
        gap_in_dim = 96 + 96 + 48 + gap_ext_dim + d_model * 2
        self.gap_head = nn.Sequential(
            nn.LayerNorm(gap_in_dim),
            nn.Dropout(dropout),
            nn.Linear(gap_in_dim, 160),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(160, 2),
        )

    def _encode_lp(self, lp: torch.Tensor) -> torch.Tensor:
        return scale_lp(lp)

    def forward(
        self,
        *,
        h_peak: torch.Tensor,
        h_vit: torch.Tensor,
        lp_vit: torch.Tensor,
        lp_win: torch.Tensor,
        phone_ids: torch.Tensor,
        prev_phone_ids: torch.Tensor,
        next_phone_ids: torch.Tensor,
        target_scalars: torch.Tensor,
        h_gap: torch.Tensor,
        lp_gap_mean: torch.Tensor,
        lp_gap_max: torch.Tensor,
        gap_scalars: torch.Tensor,
        ext1_slot_mean: torch.Tensor,
        ext1_slot_peak: torch.Tensor,
        ext1_slot_std: torch.Tensor,
        ext1_gap_mean: torch.Tensor,
        ext1_gap_max: torch.Tensor,
        lengths: torch.Tensor,
        ext2_slot_mean: Optional[torch.Tensor] = None,
        ext2_slot_peak: Optional[torch.Tensor] = None,
        ext2_slot_std: Optional[torch.Tensor] = None,
        ext2_gap_mean: Optional[torch.Tensor] = None,
        ext2_gap_max: Optional[torch.Tensor] = None,
        aux_scalars: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        B, N = phone_ids.shape
        device = phone_ids.device

        # Koel target representations
        hp = _mix_layers(h_peak, self.target_layer_logits, self.target_layer_scale, self.target_h_norm)
        hv = _mix_layers(h_vit, self.target_layer_logits, self.target_layer_scale, self.target_h_norm)
        p_ctx = self.phone_ctx_norm(
            torch.cat(
                [self.phone_emb(phone_ids), self.ctx_emb(prev_phone_ids), self.ctx_emb(next_phone_ids)],
                dim=-1,
            )
        )

        # 9-frame CTC window attention
        pos = self.pos_emb_norm(self.pos_emb(torch.arange(WINDOW_SIZE, device=device)))
        pos = pos.view(1, 1, WINDOW_SIZE, -1).expand(B, N, -1, -1)
        lp_w_feat = self.proj_lp_win(self._encode_lp(lp_win))
        kv = self.win_fuse(torch.cat([lp_w_feat, pos], dim=-1)).reshape(B * N, WINDOW_SIZE, -1)
        q = self.query_proj(p_ctx).reshape(B * N, 1, -1)
        attn_out, _ = self.local_attn(q, kv, kv, need_weights=False)
        h_attn = self.lp_win_out_norm(attn_out.reshape(B, N, -1))

        sc_in = target_scalars.float()
        if self.training and self.cfg.scalar_dropout_p > 0.0:
            keep = (torch.rand((B, N, 1), device=device) >= self.cfg.scalar_dropout_p).float()
            sc_in = sc_in * keep / (1.0 - self.cfg.scalar_dropout_p)

        # External Backbone #1
        e1_m = _mix_layers(ext1_slot_mean, self.ext1_target_logits, self.ext1_target_scale, self.ext1_target_norm)
        e1_p = _mix_layers(ext1_slot_peak, self.ext1_target_logits, self.ext1_target_scale, self.ext1_target_norm)
        e1_s = _mix_layers(ext1_slot_std, self.ext1_target_logits, self.ext1_target_scale, self.ext1_target_norm)
        e1_feat = self.proj_ext1_target(torch.cat([e1_m, e1_p, e1_s], dim=-1))

        parts = [
            self.proj_h_peak(hp),
            self.proj_h_vit(hv),
            h_attn,
            self.proj_lp_vit(self._encode_lp(lp_vit)),
            p_ctx,
            self.target_scalar_proj(sc_in),
            e1_feat,
        ]
        if self.proj_ext2_target is not None:
            e2_m = _mix_layers(ext2_slot_mean, self.ext2_target_logits, self.ext2_target_scale, self.ext2_target_norm)
            e2_p = _mix_layers(ext2_slot_peak, self.ext2_target_logits, self.ext2_target_scale, self.ext2_target_norm)
            e2_s = _mix_layers(ext2_slot_std, self.ext2_target_logits, self.ext2_target_scale, self.ext2_target_norm)
            parts.append(self.proj_ext2_target(torch.cat([e2_m, e2_p, e2_s], dim=-1)))
        if self.aux_scalar_proj is not None:
            parts.append(self.aux_scalar_proj(aux_scalars.float()))

        slot_x = self.slot_proj(torch.cat(parts, dim=-1))

        # Length-packed BiGRU
        packed = nn.utils.rnn.pack_padded_sequence(
            slot_x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.seq_encoder(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=N)
        seq_h = self.seq_norm(slot_x + self.gru_out_norm(gru_out))

        target_logits = self.target_head(seq_h)
        realized_logits = self.realized_head(seq_h)
        af_logits = self.af_head(seq_h)

        # Length-aware Gap context (exact EOS per utterance length)
        eos = self.gap_eos_emb.view(1, 1, -1).expand(B, 1, -1)
        left_h = torch.cat([eos, seq_h], dim=1)
        right_h = torch.cat([seq_h, eos], dim=1)
        batch_ix = torch.arange(B, device=device)
        right_h = right_h.clone()
        right_h[batch_ix, lengths] = self.gap_eos_emb

        hg = _mix_layers(h_gap, self.gap_layer_logits, self.gap_layer_scale, self.gap_h_norm)
        g_e1_m = _mix_layers(ext1_gap_mean, self.gap_ext1_logits, self.gap_ext1_scale, self.gap_ext1_norm)
        g_e1_x = _mix_layers(ext1_gap_max, self.gap_ext1_logits, self.gap_ext1_scale, self.gap_ext1_norm)
        gap_parts = [
            self.gap_h_proj(hg),
            self.gap_lp_proj(torch.cat([self._encode_lp(lp_gap_mean), self._encode_lp(lp_gap_max)], dim=-1)),
            self.gap_scalar_proj(gap_scalars.float()),
            self.gap_ext1_proj(torch.cat([g_e1_m, g_e1_x], dim=-1)),
            left_h,
            right_h,
        ]
        if self.gap_ext2_proj is not None:
            g_e2_m = _mix_layers(ext2_gap_mean, self.gap_ext2_logits, self.gap_ext2_scale, self.gap_ext2_norm)
            g_e2_x = _mix_layers(ext2_gap_max, self.gap_ext2_logits, self.gap_ext2_scale, self.gap_ext2_norm)
            gap_parts.append(self.gap_ext2_proj(torch.cat([g_e2_m, g_e2_x], dim=-1)))

        gap_logits = self.gap_head(torch.cat(gap_parts, dim=-1))
        return {
            "target_logits": target_logits,
            "gap_logits": gap_logits,
            "realized_logits": realized_logits,
            "af_logits": af_logits,
        }


# ---------------------------------------------------------------------------
# Stage 3: FrameXAttnMDD (Monotonic Local Cross-Attention on 20ms Frame Windows)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameXAttnConfig:
    arm: str
    ext_layers: int
    ext_dim: int
    use_koel_frame_win: bool = False
    d_model: int = 192
    dropout: float = 0.30
    scalar_dropout_p: float = 0.0
    entropy_reg_weight: float = 0.02

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FrameXAttnMDD(nn.Module):
    """Canonical-to-Frame Monotonic Local Cross-Attention over full 20ms frame windows."""

    def __init__(
        self,
        cfg: FrameXAttnConfig,
        *,
        num_koel_layers: int = NUM_XLSR_LAYERS,
        koel_dim: int = 1024,
        vocab_dim: int = 79,
        target_scalar_dim: int = len(CONTINUOUS_SCALAR_NAMES),
        gap_scalar_dim: int = len(GAP_SCALAR_NAMES),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        dropout = cfg.dropout
        lp_feat_dim = vocab_dim * 2

        # Phone & Koel scalar/lp encoders
        self.phone_emb = nn.Embedding(len(ARPA_39), 64)
        self.ctx_emb = nn.Embedding(CTX_EOS_ID + 1, 32)
        self.phone_ctx_norm = nn.LayerNorm(128)
        self.proj_lp_vit = nn.Sequential(nn.Linear(lp_feat_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.target_scalar_proj = nn.Sequential(nn.Linear(target_scalar_dim, 48), nn.LayerNorm(48), nn.GELU())

        self.proj_lp_win = nn.Sequential(nn.Linear(lp_feat_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.lp_pos_emb = nn.Embedding(WINDOW_SIZE, 16)
        self.lp_win_fuse = nn.Sequential(nn.Linear(64 + 16, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.lp_q_proj = nn.Linear(128, d_model)
        self.lp_local_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.10, batch_first=True)
        self.lp_out_norm = nn.LayerNorm(d_model)

        # External frame window cross-attention (target)
        self.ext_t_logits = nn.Parameter(torch.zeros(cfg.ext_layers))
        self.ext_t_scale = nn.Parameter(torch.ones(1))
        self.ext_t_norm = nn.LayerNorm(cfg.ext_dim)
        self.ext_t_kv_proj = nn.Sequential(nn.Linear(cfg.ext_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.ext_t_pos = nn.Embedding(FRAME_WIN_TARGET, d_model)
        self.ext_t_q_proj = nn.Linear(128 + 48, d_model)
        self.ext_t_xattn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.0, batch_first=True)
        self.ext_t_out_norm = nn.LayerNorm(d_model)

        # Optional Koel frame window cross-attention (2 mid-layers L15, L18)
        if cfg.use_koel_frame_win:
            self.koel_fw_logits = nn.Parameter(torch.zeros(2))
            self.koel_fw_scale = nn.Parameter(torch.ones(1))
            self.koel_fw_norm = nn.LayerNorm(koel_dim)
            self.koel_fw_kv_proj = nn.Sequential(nn.Linear(koel_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.koel_fw_pos = nn.Embedding(FRAME_WIN_TARGET, d_model)
            self.koel_fw_q_proj = nn.Linear(128 + 48, d_model)
            self.koel_fw_xattn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.10, batch_first=True)
            self.koel_fw_out_norm = nn.LayerNorm(d_model)
            koel_slot_dim = d_model
        else:
            self.koel_fw_kv_proj = None
            self.target_layer_logits = nn.Parameter(torch.zeros(num_koel_layers))
            self.target_layer_scale = nn.Parameter(torch.ones(1))
            self.target_h_norm = nn.LayerNorm(koel_dim)
            self.proj_h_peak = nn.Sequential(nn.Linear(koel_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            koel_slot_dim = d_model

        slot_in = d_model + koel_slot_dim + d_model + 64 + 128 + 48
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(slot_in),
            nn.Dropout(dropout),
            nn.Linear(slot_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
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

        self.target_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 3),
        )
        self.realized_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, NUM_REALIZED_CLASSES))
        self.af_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, NUM_AF))

        # Gap Cross-Attention over full gap frame window
        self.gap_layer_logits = nn.Parameter(torch.zeros(num_koel_layers))
        self.gap_layer_scale = nn.Parameter(torch.ones(1))
        self.gap_h_norm = nn.LayerNorm(koel_dim)
        self.gap_h_proj = nn.Sequential(nn.Linear(koel_dim, 96), nn.LayerNorm(96), nn.GELU())
        self.gap_lp_proj = nn.Sequential(nn.Linear(lp_feat_dim * 2, 96), nn.LayerNorm(96), nn.GELU())
        self.gap_scalar_proj = nn.Sequential(nn.Linear(gap_scalar_dim, 48), nn.LayerNorm(48), nn.GELU())
        self.gap_eos_emb = nn.Parameter(torch.zeros(d_model))

        self.gap_ext_logits = nn.Parameter(torch.zeros(cfg.ext_layers))
        self.gap_ext_scale = nn.Parameter(torch.ones(1))
        self.gap_ext_norm = nn.LayerNorm(cfg.ext_dim)
        self.gap_ext_kv_proj = nn.Sequential(nn.Linear(cfg.ext_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.gap_ext_pos = nn.Embedding(FRAME_WIN_GAP, 128)
        self.gap_ext_q_proj = nn.Linear(d_model * 2 + 48, 128)
        self.gap_ext_xattn = nn.MultiheadAttention(128, num_heads=4, dropout=0.10, batch_first=True)
        self.gap_ext_out_norm = nn.LayerNorm(128)

        gap_in = 96 + 96 + 48 + 128 + d_model * 2
        self.gap_head = nn.Sequential(
            nn.LayerNorm(gap_in),
            nn.Dropout(dropout),
            nn.Linear(gap_in, 160),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(160, 2),
        )

    def forward(
        self,
        *,
        h_peak: torch.Tensor,
        lp_vit: torch.Tensor,
        lp_win: torch.Tensor,
        phone_ids: torch.Tensor,
        prev_phone_ids: torch.Tensor,
        next_phone_ids: torch.Tensor,
        target_scalars: torch.Tensor,
        h_gap: torch.Tensor,
        lp_gap_mean: torch.Tensor,
        lp_gap_max: torch.Tensor,
        gap_scalars: torch.Tensor,
        ext_t_win: torch.Tensor,       # (B, N, W_T, L, D)
        ext_t_mask: torch.Tensor,      # (B, N, W_T) bool (True = valid)
        ext_g_win: torch.Tensor,       # (B, N+1, W_G, L, D)
        ext_g_mask: torch.Tensor,      # (B, N+1, W_G) bool (True = valid)
        lengths: torch.Tensor,
        koel_t_win: Optional[torch.Tensor] = None,  # (B, N, W_T, 2, 1024)
        koel_t_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        B, N = phone_ids.shape
        device = phone_ids.device

        p_ctx = self.phone_ctx_norm(
            torch.cat(
                [self.phone_emb(phone_ids), self.ctx_emb(prev_phone_ids), self.ctx_emb(next_phone_ids)],
                dim=-1,
            )
        )
        sc_in = target_scalars.float()
        if self.training and self.cfg.scalar_dropout_p > 0.0:
            keep = (torch.rand((B, N, 1), device=device) >= self.cfg.scalar_dropout_p).float()
            sc_in = sc_in * keep / (1.0 - self.cfg.scalar_dropout_p)
        sc_feat = self.target_scalar_proj(sc_in)

        # LP window attention
        pos_lp = self.lp_pos_emb(torch.arange(WINDOW_SIZE, device=device)).view(1, 1, WINDOW_SIZE, -1).expand(B, N, -1, -1)
        kv_lp = self.lp_win_fuse(torch.cat([self.proj_lp_win(scale_lp(lp_win)), pos_lp], dim=-1)).reshape(B * N, WINDOW_SIZE, -1)
        q_lp = self.lp_q_proj(p_ctx).reshape(B * N, 1, -1)
        lp_attn, _ = self.lp_local_attn(q_lp, kv_lp, kv_lp, need_weights=False)
        lp_attn = self.lp_out_norm(lp_attn.reshape(B, N, -1))

        # External frame cross-attention (target)
        ext_m = _mix_layers(ext_t_win, self.ext_t_logits, self.ext_t_scale, self.ext_t_norm)  # (B, N, W_T, D)
        pos_t = self.ext_t_pos(torch.arange(FRAME_WIN_TARGET, device=device)).view(1, 1, FRAME_WIN_TARGET, -1)
        kv_t = (self.ext_t_kv_proj(ext_m) + pos_t).reshape(B * N, FRAME_WIN_TARGET, -1)
        q_t = self.ext_t_q_proj(torch.cat([p_ctx, sc_feat], dim=-1)).reshape(B * N, 1, -1)
        # Only padded rows may have zero valid frames; give those a dummy frame.
        safe_t_mask = ext_t_mask.clone()
        safe_t_mask[..., FRAME_WIN_TARGET // 2] |= ~safe_t_mask.any(dim=-1)
        kpm_t = (~safe_t_mask).reshape(B * N, FRAME_WIN_TARGET)
        xattn_out, attn_w = self.ext_t_xattn(q_t, kv_t, kv_t, key_padding_mask=kpm_t, need_weights=True)
        xattn_feat = self.ext_t_out_norm(xattn_out.reshape(B, N, -1))
        valid_slots = torch.arange(N, device=device)[None, :] < lengths[:, None]
        w = attn_w.reshape(B, N, FRAME_WIN_TARGET)
        entropy = -(w * w.clamp_min(1e-7).log()).sum(-1)
        attn_entropy = (entropy * valid_slots).sum() / valid_slots.sum().clamp_min(1)

        if self.koel_fw_kv_proj is not None:
            k_m = _mix_layers(koel_t_win, self.koel_fw_logits, self.koel_fw_scale, self.koel_fw_norm)
            pos_k = self.koel_fw_pos(torch.arange(FRAME_WIN_TARGET, device=device)).view(1, 1, FRAME_WIN_TARGET, -1)
            kv_k = (self.koel_fw_kv_proj(k_m) + pos_k).reshape(B * N, FRAME_WIN_TARGET, -1)
            q_k = self.koel_fw_q_proj(torch.cat([p_ctx, sc_feat], dim=-1)).reshape(B * N, 1, -1)
            safe_k_mask = koel_t_mask.clone()
            safe_k_mask[..., FRAME_WIN_TARGET // 2] |= ~safe_k_mask.any(dim=-1)
            kpm_k = (~safe_k_mask).reshape(B * N, FRAME_WIN_TARGET)
            k_out, _ = self.koel_fw_xattn(q_k, kv_k, kv_k, key_padding_mask=kpm_k, need_weights=False)
            koel_feat = self.koel_fw_out_norm(k_out.reshape(B, N, -1))
        else:
            hp = _mix_layers(h_peak, self.target_layer_logits, self.target_layer_scale, self.target_h_norm)
            koel_feat = self.proj_h_peak(hp)

        slot_x = self.slot_proj(
            torch.cat(
                [xattn_feat, koel_feat, lp_attn, self.proj_lp_vit(scale_lp(lp_vit)), p_ctx, sc_feat],
                dim=-1,
            )
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            slot_x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.seq_encoder(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=N)
        seq_h = self.seq_norm(slot_x + self.gru_out_norm(gru_out))

        target_logits = self.target_head(seq_h)
        realized_logits = self.realized_head(seq_h)
        af_logits = self.af_head(seq_h)

        # Gap Cross-Attention
        eos = self.gap_eos_emb.view(1, 1, -1).expand(B, 1, -1)
        left_h = torch.cat([eos, seq_h], dim=1)
        right_h = torch.cat([seq_h, eos], dim=1).clone()
        right_h[torch.arange(B, device=device), lengths] = self.gap_eos_emb

        g_sc_feat = self.gap_scalar_proj(gap_scalars.float())
        hg = _mix_layers(h_gap, self.gap_layer_logits, self.gap_layer_scale, self.gap_h_norm)
        ext_g = _mix_layers(ext_g_win, self.gap_ext_logits, self.gap_ext_scale, self.gap_ext_norm)
        pos_g = self.gap_ext_pos(torch.arange(FRAME_WIN_GAP, device=device)).view(1, 1, FRAME_WIN_GAP, -1)
        kv_g = (self.gap_ext_kv_proj(ext_g) + pos_g).reshape(B * (N + 1), FRAME_WIN_GAP, -1)
        q_g = self.gap_ext_q_proj(torch.cat([left_h, right_h, g_sc_feat], dim=-1)).reshape(B * (N + 1), 1, -1)
        safe_g_mask = ext_g_mask.clone()
        safe_g_mask[..., 0] |= ~safe_g_mask.any(dim=-1)
        kpm_g = (~safe_g_mask).reshape(B * (N + 1), FRAME_WIN_GAP)
        gx_out, _ = self.gap_ext_xattn(q_g, kv_g, kv_g, key_padding_mask=kpm_g, need_weights=False)
        gx_feat = self.gap_ext_out_norm(gx_out.reshape(B, N + 1, -1))

        gap_logits = self.gap_head(
            torch.cat(
                [
                    self.gap_h_proj(hg),
                    self.gap_lp_proj(torch.cat([scale_lp(lp_gap_mean), scale_lp(lp_gap_max)], dim=-1)),
                    g_sc_feat,
                    gx_feat,
                    left_h,
                    right_h,
                ],
                dim=-1,
            )
        )
        return {
            "target_logits": target_logits,
            "gap_logits": gap_logits,
            "realized_logits": realized_logits,
            "af_logits": af_logits,
            "attn_entropy": attn_entropy,
        }


# ---------------------------------------------------------------------------
# Stage 4: Native PyTorch LoRA on WavLM (Base+ or Large) + Multi-Task AF
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper over a frozen nn.Linear layer (zero external peft dependency)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / float(r)
        self.drop = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.drop(x), self.lora_a), self.lora_b) * self.scaling
        return out + delta


def inject_wavlm_lora(
    wavlm_model: nn.Module,
    *,
    top_k_layers: int = 6,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> int:
    """Freeze entire WavLM and inject LoRA into q/k/v/out_proj of top_k_layers."""
    for p in wavlm_model.parameters():
        p.requires_grad = False
    layers = wavlm_model.encoder.layers
    start_idx = max(0, len(layers) - top_k_layers)
    replaced = 0
    for idx in range(start_idx, len(layers)):
        attn = layers[idx].attention
        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            orig = getattr(attn, proj_name)
            if isinstance(orig, nn.Linear):
                setattr(attn, proj_name, LoRALinear(orig, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


class WavLMLoRAMDD(nn.Module):
    """End-to-end differentiable WavLM-LoRA + Frozen-Koel DualRep MDD + 18-AF Multi-Task."""

    def __init__(
        self,
        wavlm_model: nn.Module,
        *,
        extract_layers: Sequence[int],
        wavlm_dim: int,
        d_model: int = 192,
        dropout: float = 0.30,
        scalar_dropout_p: float = 0.15,
    ) -> None:
        super().__init__()
        self.wavlm = wavlm_model
        self.extract_layers = tuple(extract_layers)
        self.wavlm_dim = wavlm_dim
        self.head = DualRepSingleFactorV2(
            DualRepConfig(
                arm="wavlm_lora_mdd",
                ext1_layers=len(self.extract_layers),
                ext1_dim=wavlm_dim,
                d_model=d_model,
                dropout=dropout,
                scalar_dropout_p=scalar_dropout_p,
            )
        )
        # Standalone warm-start linear probes on WavLM pooled slots (for Stage 4B multi-corpus warm-start)
        self.warm_layer_logits = nn.Parameter(torch.zeros(len(self.extract_layers)))
        self.warm_layer_scale = nn.Parameter(torch.ones(1))
        self.warm_norm = nn.LayerNorm(wavlm_dim)
        self.warm_phone_head = nn.Linear(wavlm_dim, NUM_REALIZED_CLASSES)
        self.warm_af_head = nn.Linear(wavlm_dim, NUM_AF + 1)  # final logit is blank gate

    def _encode_wav_layers(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """wav_tensor: (1, S) -> returns (T, L, D) float32 differentiable through LoRA."""
        out = self.wavlm(input_values=wav_tensor, output_hidden_states=True)
        hs = out.hidden_states
        return torch.stack([hs[i][0] for i in self.extract_layers], dim=1)  # (T, L, D)

    def pool_utterance_differentiable(
        self,
        frames: torch.Tensor,  # (T, L, D)
        starts: Sequence[int],
        ends: Sequence[int],
        peaks: Sequence[int],
        t_koel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_ext = frames.shape[0]
        scale = float(t_ext) / max(1.0, float(t_koel))
        n = len(peaks)
        s_mean_l, s_peak_l, s_std_l = [], [], []
        ext_peaks: List[int] = []
        for i in range(n):
            a = max(0, min(t_ext - 1, int(round(starts[i] * scale))))
            b = max(a + 1, min(t_ext, int(round(ends[i] * scale))))
            p = max(0, min(t_ext - 1, int(round(peaks[i] * scale))))
            ext_peaks.append(p)
            seg = frames[a:b]
            s_mean_l.append(seg.mean(dim=0))
            s_peak_l.append(frames[p])
            s_std_l.append(seg.std(dim=0, unbiased=False) if seg.shape[0] > 1 else torch.zeros_like(frames[p]))

        g_mean_l, g_max_l = [], []
        for g in range(n + 1):
            ga = 0 if g == 0 else ext_peaks[g - 1]
            gb = t_ext if g == n else ext_peaks[g] + 1
            if gb <= ga:
                gb = min(t_ext, ga + 1)
            gseg = frames[ga:gb]
            g_mean_l.append(gseg.mean(dim=0))
            g_max_l.append(gseg.max(dim=0).values)

        return (
            torch.stack(s_mean_l, dim=0),
            torch.stack(s_peak_l, dim=0),
            torch.stack(s_std_l, dim=0),
            torch.stack(g_mean_l, dim=0),
            torch.stack(g_max_l, dim=0),
        )

    def forward_single_utt(
        self,
        wav_tensor: torch.Tensor,  # (1, S)
        koel_inp: Dict[str, torch.Tensor],  # batch size 1
        starts: Sequence[int],
        ends: Sequence[int],
        peaks: Sequence[int],
        t_koel: int,
    ) -> Dict[str, torch.Tensor]:
        frames = self._encode_wav_layers(wav_tensor)
        sm, sp, ss, gm, gx = self.pool_utterance_differentiable(frames, starts, ends, peaks, t_koel)
        inp = dict(koel_inp)
        inp["ext1_slot_mean"] = sm.unsqueeze(0)
        inp["ext1_slot_peak"] = sp.unsqueeze(0)
        inp["ext1_slot_std"] = ss.unsqueeze(0)
        inp["ext1_gap_mean"] = gm.unsqueeze(0)
        inp["ext1_gap_max"] = gx.unsqueeze(0)
        return self.head(**inp)

    def forward_warm_single_utt(self, wav_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frame logits for phone and AF-bundle CTC, without pseudo time labels."""
        frames = self._encode_wav_layers(wav_tensor)
        h = _mix_layers(frames, self.warm_layer_logits, self.warm_layer_scale, self.warm_norm)
        return self.warm_phone_head(h), self.warm_af_head(h)
