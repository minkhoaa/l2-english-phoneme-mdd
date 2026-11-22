"""Levers O2 + O3 + O4: Drop-in orthogonal multi-task MDD loss.

Guarantees exact numerical parity with compute_multitask_mdd_loss when all orthogonal
levers are disabled (tau_la=0.0, lambda_so_cond=0.0, gap_pos_weight=8.0), while letting
each lever be ablated one variable at a time:
  - O2 (tau_la > 0): (source, phone) Logit-Adjusted Target Cross-Entropy.
  - O3 (lambda_so_cond > 0): Dedicated Conditional S-vs-O binary cross-entropy on
    true error slots (target_labels in {1, 2}), directly targeting the 3.4-12.2 pp
    Binary-to-Strict F1 gap caused by S<->O swaps.
  - O4 (gap_pos_weight): Configurable Gap positive class weight (or kept at 8.0 and
    analytically/Platt debiased on Cal in calibration.py).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.nn import functional as F

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import NUM_REALIZED_CLASSES
from .prior_debias import apply_logit_adjustment


def compute_ortho_mdd_loss(
    outputs: Dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    gap_labels: torch.Tensor,
    realized_labels: Optional[torch.Tensor] = None,
    is_native_l1: Optional[torch.Tensor] = None,
    *,
    source_ids: Optional[torch.Tensor] = None,
    phone_ids: Optional[torch.Tensor] = None,
    log_prior_table: Optional[torch.Tensor] = None,
    tau_la: float = 0.0,
    lambda_so_cond: float = 0.0,
    so_label_smoothing: float = 0.0,
    gap_pos_weight: float = 8.0,
    lambda_gap: float = 0.6,
    lambda_realized: float = 0.25,
    native_fa_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Unified multi-task MDD loss with strictly isolated orthogonal toggles."""
    t_logits = outputs["target_logits"]
    g_logits = outputs["gap_logits"]

    if tau_la != 0.0:
        if source_ids is None or phone_ids is None or log_prior_table is None:
            raise ValueError("tau_la != 0 requires source_ids, phone_ids, and log_prior_table")
        t_logits_ce = apply_logit_adjustment(
            t_logits, source_ids, phone_ids, log_prior_table, tau_la=tau_la
        )
    else:
        t_logits_ce = t_logits

    t_ce = F.cross_entropy(
        t_logits_ce.reshape(-1, 3),
        target_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(target_labels)
    valid_t = target_labels != -100
    t_weights = torch.ones_like(t_ce)
    if is_native_l1 is not None:
        t_weights = torch.where(
            is_native_l1[:, None],
            torch.full_like(t_weights, native_fa_weight),
            t_weights,
        )
    loss_target = (t_ce * t_weights * valid_t).sum() / valid_t.sum().clamp(min=1)

    # Lever O3: Conditional S vs O loss strictly on true error slots (1=S, 2=O)
    # Uses raw (or logit-adjusted conditional) difference: logit_O - logit_S.
    if lambda_so_cond > 0.0:
        err_mask = (target_labels == 1) | (target_labels == 2)
        if err_mask.any():
            logits_so = t_logits_ce[err_mask]  # [K, 3]
            so_diff = logits_so[:, 2] - logits_so[:, 1]  # positive => O, negative => S
            y_is_o = (target_labels[err_mask] == 2).to(dtype=so_diff.dtype)
            if so_label_smoothing > 0.0:
                eps = float(so_label_smoothing)
                y_is_o = y_is_o * (1.0 - eps) + 0.5 * eps
            loss_so_cond = F.binary_cross_entropy_with_logits(so_diff, y_is_o)
        else:
            loss_so_cond = torch.zeros((), device=t_logits.device, dtype=t_logits.dtype)
    else:
        loss_so_cond = torch.zeros((), device=t_logits.device, dtype=t_logits.dtype)

    gw = torch.tensor([1.0, gap_pos_weight], device=g_logits.device, dtype=g_logits.dtype)
    loss_gap = F.cross_entropy(
        g_logits.reshape(-1, 2),
        gap_labels.reshape(-1),
        weight=gw,
        ignore_index=-100,
    )

    if realized_labels is not None and (realized_labels != -100).any():
        loss_real = F.cross_entropy(
            outputs["realized_logits"].reshape(-1, NUM_REALIZED_CLASSES),
            realized_labels.reshape(-1),
            ignore_index=-100,
        )
    else:
        loss_real = torch.zeros((), device=t_logits.device, dtype=t_logits.dtype)

    total = (
        loss_target
        + float(lambda_so_cond) * loss_so_cond
        + float(lambda_gap) * loss_gap
        + float(lambda_realized) * loss_real
    )
    return {
        "loss": total,
        "loss_target": loss_target.detach(),
        "loss_so_cond": loss_so_cond.detach(),
        "loss_gap": loss_gap.detach(),
        "loss_realized": loss_real.detach(),
    }
