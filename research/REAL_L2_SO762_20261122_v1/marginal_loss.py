"""Exact hybrid 3-way + marginal/soft binary MDD loss for SpeechOcean762 TRAIN.

Key property:
  - On standard 3-way rows (L2-ARCTIC, EpaDB, TIMIT): bit-for-bit identical to
    compute_multitask_mdd_loss (C/S/O CE + weighted Gap CE + Realized-40 CE).
  - On SpeechOcean762 rows:
      * Slots with explicit typed gold (C=0, S=1, O=2) get standard 3-way CE.
      * Slots with binary-only or soft 5-expert error probability q_err in [0, 1]
        (where target_labels == -100 and soft_err_prob >= 0) get exact marginal
        cross-entropy:
            L = -(1 - q) * log P(C) - q * log(P(S) + P(O))
        which NEVER forces an arbitrary S vs O choice on conflated/borderline slots!
      * Gap and Realized losses remain masked (-100) on SO762 rows unless explicit
        realized phone ID is given.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.nn import functional as F

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import NUM_REALIZED_CLASSES


def compute_so762_hybrid_mdd_loss(
    outputs: Dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    gap_labels: torch.Tensor,
    realized_labels: Optional[torch.Tensor] = None,
    is_native_l1: Optional[torch.Tensor] = None,
    *,
    soft_err_prob: Optional[torch.Tensor] = None,
    slot_weight: Optional[torch.Tensor] = None,
    gap_pos_weight: float = 8.0,
    lambda_gap: float = 0.6,
    lambda_realized: float = 0.25,
    native_fa_weight: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """Unified multi-task MDD loss supporting exact marginal binary / soft targets."""
    t_logits = outputs["target_logits"]  # [B, L, 3]
    g_logits = outputs["gap_logits"]  # [B, L+1, 2]

    # 1. Standard 3-way CE on typed slots (target_labels in {0, 1, 2})
    t_ce = F.cross_entropy(
        t_logits.reshape(-1, 3),
        target_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(target_labels)
    valid_3way = target_labels != -100

    # 2. Marginal binary / soft CE on untyped error-probability slots (soft_err_prob >= 0)
    if soft_err_prob is not None:
        valid_bin = (soft_err_prob >= 0.0) & (~valid_3way)
        if valid_bin.any():
            log_z = torch.logsumexp(t_logits, dim=-1)  # [B, L]
            log_p_c = t_logits[..., 0] - log_z
            log_p_err = torch.logsumexp(t_logits[..., 1:3], dim=-1) - log_z
            q = soft_err_prob.clamp(0.0, 1.0).to(dtype=t_logits.dtype)
            bin_ce = -(1.0 - q) * log_p_c - q * log_p_err
        else:
            bin_ce = torch.zeros_like(t_ce)
    else:
        valid_bin = torch.zeros_like(valid_3way)
        bin_ce = torch.zeros_like(t_ce)

    combined_ce = torch.where(valid_3way, t_ce, torch.where(valid_bin, bin_ce, torch.zeros_like(t_ce)))
    valid_any = valid_3way | valid_bin

    t_weights = torch.ones_like(combined_ce)
    if is_native_l1 is not None:
        t_weights = torch.where(
            is_native_l1[:, None],
            torch.full_like(t_weights, native_fa_weight),
            t_weights,
        )
    if slot_weight is not None:
        t_weights = t_weights * slot_weight.to(dtype=t_weights.dtype)

    denom = valid_any.sum().clamp(min=1)
    loss_target = (combined_ce * t_weights * valid_any).sum() / denom

    # 3. Weighted Gap CE (safe if a batch happens to have all gap_labels == -100)
    if (gap_labels != -100).any():
        gw = torch.tensor([1.0, gap_pos_weight], device=g_logits.device, dtype=g_logits.dtype)
        loss_gap = F.cross_entropy(
            g_logits.reshape(-1, 2),
            gap_labels.reshape(-1),
            weight=gw,
            ignore_index=-100,
        )
    else:
        loss_gap = torch.zeros((), device=t_logits.device, dtype=t_logits.dtype)

    # 4. Realized-phone auxiliary CE
    if realized_labels is not None and (realized_labels != -100).any():
        loss_real = F.cross_entropy(
            outputs["realized_logits"].reshape(-1, NUM_REALIZED_CLASSES),
            realized_labels.reshape(-1),
            ignore_index=-100,
        )
    else:
        loss_real = torch.zeros((), device=t_logits.device, dtype=t_logits.dtype)

    total = loss_target + float(lambda_gap) * loss_gap + float(lambda_realized) * loss_real
    return {
        "loss": total,
        "loss_target": loss_target.detach(),
        "loss_gap": loss_gap.detach(),
        "loss_realized": loss_real.detach(),
    }
