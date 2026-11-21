"""Length-correct V2 control; archived V2 and WindowHead remain replayable."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import SeqMDDv2


class LengthAwareV2(SeqMDDv2):
    """Use the V2 recipe with the per-utterance EOS contract in the gap head.

    Parameter names and dimensions are unchanged. Keep the archived model for
    archival replay; use this class for new controls and regressions.
    """

    def forward(
        self,
        *,
        h_vit,
        h_peak,
        lp_win,
        lp_vit,
        phone_ids,
        prev_phone_ids,
        next_phone_ids,
        target_scalars,
        h_gap,
        lp_gap_mean,
        lp_gap_max,
        gap_scalars,
        lengths,
    ):
        b, l, w, _ = lp_win.shape
        hv = self._mix_layers(h_vit, self.target_layer_logits, self.target_layer_scale)
        hp = self._mix_layers(h_peak, self.target_layer_logits, self.target_layer_scale)
        mask = (
            torch.arange(l, device=hv.device)[None, :] < lengths[:, None]
        ).unsqueeze(-1)
        denom = mask.sum(1, keepdim=True).clamp_min(1)
        mean = (hv * mask).sum(1, keepdim=True) / denom
        var = ((hv - mean).square() * mask).sum(1, keepdim=True) / denom
        std = var.clamp_min(1e-8).sqrt()
        context = torch.cat(
            [
                self.phone_emb(phone_ids),
                self.ctx_emb(prev_phone_ids),
                self.ctx_emb(next_phone_ids),
            ],
            dim=-1,
        )
        pos = self.pos_emb(self.pos_ids).view(1, 1, w, 16).expand(b, l, -1, -1)
        window = self.win_fuse(torch.cat([self.proj_lp_win(lp_win), pos], dim=-1))
        kv = window.reshape(b * l, w, self.d_model)
        query = self.query_proj(context).reshape(b * l, 1, self.d_model)
        attn, _ = self.local_attn(query, kv, kv, need_weights=False)
        slot = self.slot_proj(
            torch.cat(
                [
                    self.proj_h_vit((hv - mean) / std),
                    self.proj_h_peak((hp - mean) / std),
                    attn.reshape(b, l, self.d_model),
                    self.proj_lp_vit(lp_vit),
                    context,
                    self.target_scalar_proj(target_scalars),
                ],
                dim=-1,
            )
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            slot, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        seq, _ = self.seq_encoder(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True, total_length=l)
        seq = self.seq_norm(seq + slot)
        target = self.target_head(seq)
        realized = self.realized_head(seq)
        gh = (
            self._mix_layers(h_gap, self.gap_layer_logits, self.gap_layer_scale) - mean
        ) / std
        acoustic = self.gap_ac_proj(
            torch.cat([gh, lp_gap_mean, lp_gap_max, gap_scalars], dim=-1)
        )
        left = torch.cat([self.bos_state.expand(b, 1, -1), seq], dim=1)
        # Gap k uses target k on its right, except at k=length[b] where it uses EOS.
        # Scatter replaces the residual padded state and routes its gradient to EOS.
        eos_idx = lengths.view(b, 1, 1).expand(-1, -1, self.d_model)
        right = F.pad(seq, (0, 0, 0, 1)).scatter(
            1, eos_idx, self.eos_state.expand(b, 1, -1)
        )
        gap = self.gap_head(torch.cat([left, acoustic, right], dim=-1))
        return {"target_logits": target, "gap_logits": gap, "realized_logits": realized}
