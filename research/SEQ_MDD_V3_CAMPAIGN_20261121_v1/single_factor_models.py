"""Single-factor leave-one-out variants of RepairedCodeFixV2 for clean causal isolation.

Does NOT modify repaired_v2_arch.py (preserving code receipts of code_fixes_123_20261121_v1).
Each variant flips EXACTLY ONE architectural factor relative to `fix12_arch`:
  1. `iso_restore_cmvn`: restore length-masked utterance CMVN on target (hp, hw),
                         keeping independent gap_h_norm on h_gap (no cross-layer CMVN bug).
  2. `iso_no_clip_lp`  : replace [exp(lp), clip(lp,-10,0)/10] with LayerNorm(79) directly on
                         raw log-probs (preserves tail < -10 and avoids exp(mean(lp))).
  3. `iso_no_hwin`     : remove 5-frame h_win_attn and use h_vit + h_peak (isolates Fix #1 vs Fix #2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    CTX_EOS_ID,
    WINDOW_SIZE,
)
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    CONTINUOUS_SCALAR_NAMES,
    GAP_SCALAR_NAMES,
    NUM_REALIZED_CLASSES,
    NUM_XLSR_LAYERS,
)

from .repaired_v2_arch import H_WIN_SIZE, pool_features_codefix, scale_lp


@dataclass(frozen=True)
class SingleFactorConfig:
    arm: str
    use_utt_cmvn: bool = False
    raw_lp_layernorm: bool = False
    use_h_win: bool = True
    d_model: int = 192
    dropout: float = 0.30

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


FACTOR_ARMS: Dict[str, SingleFactorConfig] = {
    "iso_restore_cmvn": SingleFactorConfig(
        arm="iso_restore_cmvn", use_utt_cmvn=True, raw_lp_layernorm=False, use_h_win=True
    ),
    "iso_no_clip_lp": SingleFactorConfig(
        arm="iso_no_clip_lp", use_utt_cmvn=False, raw_lp_layernorm=True, use_h_win=True
    ),
    "iso_no_hwin": SingleFactorConfig(
        arm="iso_no_hwin", use_utt_cmvn=False, raw_lp_layernorm=False, use_h_win=False
    ),
}


def pool_features_with_hvit(h: np.ndarray, lp: np.ndarray, phones) -> Dict[str, np.ndarray]:
    """Same exact features as pool_features_codefix plus h_vit for iso_no_hwin."""
    feat = pool_features_codefix(h, lp, phones)
    spans, _ = ctc_viterbi_align_allophone_aware(lp, phones)
    s, e, _ = np.asarray(spans).T
    feat["h_vit"] = np.stack(
        [h[a : b + 1].astype(np.float32).mean(0).astype(np.float16) for a, b in zip(s, e)]
    )
    return feat


class SingleFactorV2(nn.Module):
    """Parameter-matched single-factor isolation model derived from RepairedCodeFixV2."""

    def __init__(
        self,
        cfg: SingleFactorConfig,
        *,
        num_layers: int = NUM_XLSR_LAYERS,
        h_dim: int = 1024,
        vocab_dim: int = 79,
        target_scalar_dim: int = len(CONTINUOUS_SCALAR_NAMES),
        gap_scalar_dim: int = len(GAP_SCALAR_NAMES),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.vocab_dim = vocab_dim
        self.d_model = cfg.d_model
        d_model = cfg.d_model
        dropout = cfg.dropout

        lp_feat_dim = vocab_dim if cfg.raw_lp_layernorm else vocab_dim * 2
        self.lp_in_norm = nn.LayerNorm(vocab_dim) if cfg.raw_lp_layernorm else None

        # 1. Layer mixing
        self.target_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.target_layer_scale = nn.Parameter(torch.ones(1))
        self.gap_layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gap_layer_scale = nn.Parameter(torch.ones(1))
        self.target_h_norm = nn.LayerNorm(h_dim)
        self.gap_h_norm = nn.LayerNorm(h_dim)

        # 2. Phone context
        self.phone_emb = nn.Embedding(len(ARPA_39), 64)
        self.ctx_emb = nn.Embedding(CTX_EOS_ID + 1, 32)
        self.phone_ctx_norm = nn.LayerNorm(128)

        # 3a. Peak acoustic projection
        self.proj_h_peak = nn.Sequential(
            nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        # 3b. Either 5-frame h_win attention OR V2-style h_vit projection
        if cfg.use_h_win:
            self.proj_h_win = nn.Sequential(
                nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
            )
            self.h_pos_emb = nn.Embedding(H_WIN_SIZE, d_model)
            self.h_win_query = nn.Linear(128, d_model)
            self.h_win_attn = nn.MultiheadAttention(
                d_model, num_heads=4, dropout=0.10, batch_first=True
            )
            self.h_win_out_norm = nn.LayerNorm(d_model)
            self.proj_h_vit = None
        else:
            self.proj_h_vit = nn.Sequential(
                nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
            )
            self.proj_h_win = None

        # 3c. 9-frame CTC posterior window
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

        slot_in_dim = d_model * 3 + 64 + 128 + 48
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(slot_in_dim),
            nn.Dropout(dropout),
            nn.Linear(slot_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 4. Utterance BiGRU + balanced residual
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

        # 5. Target & Realized heads
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

        # 6. Gap head (identical to RepairedCodeFixV2; uses same LP encoding as target)
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

    def _encode_lp(self, lp: torch.Tensor) -> torch.Tensor:
        if self.lp_in_norm is not None:
            return self.lp_in_norm(lp)
        return scale_lp(lp)

    def _mix_layers(
        self, h_layers: torch.Tensor, logits: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        if h_layers.requires_grad:
            h_normed = F.layer_norm(h_layers.float(), (self.h_dim,))
        else:
            with torch.no_grad():
                h_normed = F.layer_norm(h_layers.float(), (self.h_dim,))
        w = torch.softmax(logits, dim=0)
        return torch.einsum("l,...ld->...d", w, h_normed) * scale

    def encode_sequence(
        self,
        *,
        h_peak: torch.Tensor,
        h_win: torch.Tensor,
        h_vit: torch.Tensor,
        lp_win: torch.Tensor,
        lp_vit: torch.Tensor,
        phone_ids: torch.Tensor,
        prev_phone_ids: torch.Tensor,
        next_phone_ids: torch.Tensor,
        target_scalars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, L, W, _ = lp_win.shape
        hp_mix = self._mix_layers(h_peak, self.target_layer_logits, self.target_layer_scale)

        if self.cfg.use_utt_cmvn:
            # Length-masked utterance CMVN on target acoustic features (batch-invariant!)
            mask = (torch.arange(L, device=hp_mix.device)[None, :] < lengths[:, None]).unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            mu = (hp_mix * mask).sum(dim=1, keepdim=True) / denom
            var = ((hp_mix - mu).square() * mask).sum(dim=1, keepdim=True) / denom
            sd = var.clamp_min(1e-8).sqrt()
            hp = (hp_mix - mu) / sd
            if self.cfg.use_h_win:
                hw_mix = self._mix_layers(h_win, self.target_layer_logits, self.target_layer_scale)
                hw = (hw_mix - mu.unsqueeze(2)) / sd.unsqueeze(2)
                hv = None
            else:
                hv_mix = self._mix_layers(h_vit, self.target_layer_logits, self.target_layer_scale)
                hv = (hv_mix - mu) / sd
                hw = None
        else:
            hp = self.target_h_norm(hp_mix)
            if self.cfg.use_h_win:
                hw = self.target_h_norm(
                    self._mix_layers(h_win, self.target_layer_logits, self.target_layer_scale)
                )
                hv = None
            else:
                hv = self.target_h_norm(
                    self._mix_layers(h_vit, self.target_layer_logits, self.target_layer_scale)
                )
                hw = None

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

        if self.cfg.use_h_win:
            hw_f = self.proj_h_win(hw) + self.h_pos_emb(self.h_pos_ids).view(
                1, 1, H_WIN_SIZE, self.d_model
            )
            qh = self.h_win_query(e).reshape(B * L, 1, self.d_model)
            kvh = hw_f.reshape(B * L, H_WIN_SIZE, self.d_model)
            h_second, _ = self.h_win_attn(qh, kvh, kvh, need_weights=False)
            h_second = self.h_win_out_norm(h_second.reshape(B, L, self.d_model))
        else:
            h_second = self.proj_h_vit(hv)

        pos = self.pos_emb_norm(self.pos_emb(self.pos_ids)).view(1, 1, W, 16).expand(
            B, L, -1, -1
        )
        f_lp = self.win_fuse(
            torch.cat([self.proj_lp_win(self._encode_lp(lp_win)), pos], dim=-1)
        )
        ql = self.query_proj(e).reshape(B * L, 1, self.d_model)
        kvl = f_lp.reshape(B * L, W, self.d_model)
        lp_attn, _ = self.local_attn(ql, kvl, kvl, need_weights=False)
        lp_attn = self.lp_win_out_norm(lp_attn.reshape(B, L, self.d_model))

        slot_x = self.slot_proj(
            torch.cat(
                [
                    self.proj_h_peak(hp),
                    h_second,
                    lp_attn,
                    self.proj_lp_vit(self._encode_lp(lp_vit)),
                    e,
                    self.target_scalar_proj(target_scalars),
                ],
                dim=-1,
            )
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            slot_x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        seq_out, _ = self.seq_encoder(packed)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(
            seq_out, batch_first=True, total_length=L
        )
        return self.seq_norm(self.gru_out_norm(seq_out) + slot_x)

    def forward_gap_from_seq(
        self,
        *,
        seq: torch.Tensor,
        h_gap: torch.Tensor,
        lp_gap_mean: torch.Tensor,
        lp_gap_max: torch.Tensor,
        gap_scalars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B = seq.shape[0]
        hg = self.gap_h_norm(
            self._mix_layers(h_gap, self.gap_layer_logits, self.gap_layer_scale)
        )
        g_h = self.gap_h_proj(hg)
        g_lp = self.gap_lp_proj(
            torch.cat([self._encode_lp(lp_gap_mean), self._encode_lp(lp_gap_max)], dim=-1)
        )
        g_sc = self.gap_sc_proj(gap_scalars)
        gap_ac = self.gap_ac_fuse(torch.cat([g_h, g_lp, g_sc], dim=-1))

        left = torch.cat([self.bos_state.expand(B, 1, -1), seq], dim=1)
        eos_idx = lengths.view(B, 1, 1).expand(-1, -1, self.d_model)
        right = F.pad(seq, (0, 0, 0, 1)).scatter(
            1, eos_idx, self.eos_state.expand(B, 1, -1)
        )
        return self.gap_head(torch.cat([left, gap_ac, right], dim=-1))

    def forward(
        self,
        *,
        h_peak: torch.Tensor,
        h_win: torch.Tensor,
        h_vit: torch.Tensor,
        lp_win: torch.Tensor,
        lp_vit: torch.Tensor,
        phone_ids: torch.Tensor,
        prev_phone_ids: torch.Tensor,
        next_phone_ids: torch.Tensor,
        target_scalars: torch.Tensor,
        h_gap: torch.Tensor,
        lp_gap_mean: torch.Tensor,
        lp_gap_max: torch.Tensor,
        gap_scalars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        seq = self.encode_sequence(
            h_peak=h_peak,
            h_win=h_win,
            h_vit=h_vit,
            lp_win=lp_win,
            lp_vit=lp_vit,
            phone_ids=phone_ids,
            prev_phone_ids=prev_phone_ids,
            next_phone_ids=next_phone_ids,
            target_scalars=target_scalars,
            lengths=lengths,
        )
        gap_logits = self.forward_gap_from_seq(
            seq=seq,
            h_gap=h_gap,
            lp_gap_mean=lp_gap_mean,
            lp_gap_max=lp_gap_max,
            gap_scalars=gap_scalars,
            lengths=lengths,
        )
        return {
            "target_logits": self.target_head(seq),
            "gap_logits": gap_logits,
            "realized_logits": self.realized_head(seq),
        }
