"""Lever O2 & O2b: Corpus-Conditioned Phone Prior Debiasing.

Addresses the strong corpus-by-phone prior collision observed in Fit:
  - DH: 3.3% error on EpaDB vs 57.2% on L2-ARCTIC (17x gap)
  - AH: 54.9% on EpaDB vs 8.8% on L2-ARCTIC (6.2x gap)
  - AO: 48.5% on EpaDB vs 6.4% on L2-ARCTIC (7.6x gap)
  - Z : 80.4% on EpaDB vs 48.6% on L2-ARCTIC

Implements:
  1. SourcePhonePriorTable (Fit-only): Empirical-Bayes shrunk P(C, S, O | source, phone)
     used for Menon et al. (ICLR 2021) Logit Adjustment during training:
         logits_train = logits_raw + tau_la * log P_fit(y | source, phone)
     so that the neural network learns pure acoustic likelihood ratios rather than
     memorizing P(error | corpus_mic, canonical_phone). At inference / Cal / Eval,
     only logits_raw (or neutral-prior adjusted logits) are used.
  2. apply_phone_id_dropout (O2b): optional stochastic masking of phone_id tokens during
     training to independently test regularizing phone-identity shortcuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    PHONE_TO_ID,
)

CANONICAL_SOURCES: Tuple[str, ...] = ("l2arctic", "epadb", "timit")
SOURCE_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(CANONICAL_SOURCES)}
LABEL_TO_IDX: Dict[str, int] = {"C": 0, "S": 1, "O": 2}
NUM_PHONES = len(ARPA_39)  # 39 valid ARPAbet phones; index 39/40 reserved for PAD/UNK


def canonical_source_group(source: str) -> str:
    s = str(source).lower()
    if s.startswith("epadb"):
        return "epadb"
    if s.startswith(("timit", "l1", "mocha", "librispeech")):
        return "timit"
    return "l2arctic"


def source_to_id(source: str) -> int:
    return SOURCE_TO_ID[canonical_source_group(source)]


@dataclass(frozen=True)
class SourcePhonePriorTable:
    """Fit-only Empirical-Bayes shrunk prior table P(C, S, O | source, phone)."""

    probs: np.ndarray  # shape [n_sources, NUM_PHONES + 2, 3], float32
    log_probs: np.ndarray  # shape [n_sources, NUM_PHONES + 2, 3], float32
    source_probs: np.ndarray  # shape [n_sources, 3], float32
    shrinkage_alpha: float
    min_prob: float
    counts: np.ndarray  # shape [n_sources, NUM_PHONES, 3], int64

    @classmethod
    def from_fit_rows(
        cls,
        fit_rows: Sequence[Mapping[str, Any]],
        *,
        shrinkage_alpha: float = 20.0,
        min_prob: float = 1e-4,
    ) -> "SourcePhonePriorTable":
        """Estimate Empirical-Bayes priors strictly from Fit rows (never Cal/Eval)."""
        if shrinkage_alpha < 0.0:
            raise ValueError("shrinkage_alpha must be >= 0")
        n_src = len(CANONICAL_SOURCES)
        counts = np.zeros((n_src, NUM_PHONES, 3), dtype=np.int64)
        for r in fit_rows:
            sid = source_to_id(str(r.get("source", "")))
            for ph, g in zip(r["canonical_phones"], r["gold_targets"]):
                if g not in LABEL_TO_IDX or ph not in PHONE_TO_ID:
                    continue
                counts[sid, PHONE_TO_ID[ph], LABEL_TO_IDX[g]] += 1

        # Source-level marginal prior P(y | source), smoothed slightly with uniform [1, 1, 1]
        src_totals = counts.sum(axis=1).astype(np.float64)  # [n_src, 3]
        src_probs = (src_totals + 1.0) / np.maximum(1.0, src_totals.sum(axis=1, keepdims=True) + 3.0)

        # Phone-conditioned prior with Empirical-Bayes shrinkage toward source marginal
        probs = np.zeros((n_src, NUM_PHONES + 2, 3), dtype=np.float64)
        for sid in range(n_src):
            for pid in range(NUM_PHONES):
                c_vec = counts[sid, pid].astype(np.float64)
                n_sp = c_vec.sum()
                p_sp = (c_vec + shrinkage_alpha * src_probs[sid]) / max(
                    1e-12, n_sp + shrinkage_alpha
                )
                p_sp = np.clip(p_sp, min_prob, 1.0)
                p_sp /= p_sp.sum()
                probs[sid, pid] = p_sp
            # Fallback rows for UNK/PAD indices (39, 40) use source marginal
            probs[sid, NUM_PHONES:] = src_probs[sid]

        log_probs = np.log(probs).astype(np.float32)
        return cls(
            probs=probs.astype(np.float32),
            log_probs=log_probs,
            source_probs=src_probs.astype(np.float32),
            shrinkage_alpha=float(shrinkage_alpha),
            min_prob=float(min_prob),
            counts=counts,
        )

    def to_log_prior_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.log_probs).to(device=device, dtype=torch.float32)

    def audit_summary(self) -> Dict[str, Any]:
        """Quantify cross-source prior ratios (e.g., max/min P(err | src, phone))."""
        p_err = 1.0 - self.probs[:, :NUM_PHONES, 0]  # [n_src, 39]
        contrasts = []
        for pid, ph in enumerate(ARPA_39):
            l2a_e = float(p_err[SOURCE_TO_ID["l2arctic"], pid])
            epa_e = float(p_err[SOURCE_TO_ID["epadb"], pid])
            tim_e = float(p_err[SOURCE_TO_ID["timit"], pid])
            ratio_l2_epa = max(l2a_e, epa_e) / max(1e-6, min(l2a_e, epa_e))
            contrasts.append(
                {
                    "phone": ph,
                    "l2arctic_err_pct": round(100.0 * l2a_e, 2),
                    "epadb_err_pct": round(100.0 * epa_e, 2),
                    "timit_err_pct": round(100.0 * tim_e, 2),
                    "l2_vs_epadb_ratio": round(ratio_l2_epa, 2),
                }
            )
        contrasts.sort(key=lambda x: x["l2_vs_epadb_ratio"], reverse=True)
        return {
            "shrinkage_alpha": self.shrinkage_alpha,
            "source_marginals": {
                s: {
                    "C": round(float(self.source_probs[i, 0]), 4),
                    "S": round(float(self.source_probs[i, 1]), 4),
                    "O": round(float(self.source_probs[i, 2]), 4),
                }
                for i, s in enumerate(CANONICAL_SOURCES)
            },
            "top_10_cross_source_prior_collisions": contrasts[:10],
        }


def apply_logit_adjustment(
    target_logits: torch.Tensor,
    source_ids: torch.Tensor,
    phone_ids: torch.Tensor,
    log_prior_table: torch.Tensor,
    *,
    tau_la: float = 1.0,
) -> torch.Tensor:
    """Add tau_la * log P_fit(y | source, phone) to target_logits during training only.

    Args:
        target_logits: [B, L, 3] raw model logits (C, S, O).
        source_ids: [B] integer source IDs in [0, n_sources).
        phone_ids: [B, L] integer canonical phone IDs in [0, NUM_PHONES + 1].
        log_prior_table: [n_sources, NUM_PHONES + 2, 3] from SourcePhonePriorTable.
        tau_la: adjustment strength (0.0 = exact no-op; 1.0 = full Bayes debiasing).
    """
    if tau_la == 0.0:
        return target_logits
    clamped_pids = phone_ids.clamp(min=0, max=log_prior_table.shape[1] - 1)
    src_idx = source_ids[:, None].expand_as(clamped_pids)
    log_pi = log_prior_table[src_idx, clamped_pids]  # [B, L, 3]
    return target_logits + float(tau_la) * log_pi.to(dtype=target_logits.dtype)


def apply_phone_id_dropout(
    phone_ids: torch.Tensor,
    *,
    p_drop: float,
    unk_id: int = NUM_PHONES,
    training: bool = True,
) -> torch.Tensor:
    """Stochastically mask canonical phone_ids during training (Lever O2b).

    Always returns phone_ids unchanged when training=False or p_drop <= 0.
    """
    if not training or p_drop <= 0.0:
        return phone_ids
    if p_drop >= 1.0:
        return torch.full_like(phone_ids, unk_id)
    mask = torch.rand(phone_ids.shape, device=phone_ids.device) < float(p_drop)
    return torch.where(mask, torch.full_like(phone_ids, unk_id), phone_ids)
