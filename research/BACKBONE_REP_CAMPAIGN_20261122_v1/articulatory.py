"""Deterministic 18-dimensional articulatory feature table for ARPAbet-39 phones.

Maps each canonical/realized ARPAbet-39 phone to binary phonological/articulatory
attributes (place, manner, voicing, vowel height/backness/rounding/diphthong/rhotic).
Used for Multi-Task Articulatory Supervision in Stage 4 to regularize backbone LoRA
against speaker memorization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import ARPA_39

AF_NAMES: tuple[str, ...] = (
    "is_vowel",
    "is_voiced",
    "manner_stop",
    "manner_fricative",
    "manner_affricate",
    "manner_nasal",
    "manner_liquid_glide",
    "place_labial",
    "place_dental_alveolar",
    "place_postalveolar_palatal",
    "place_velar_glottal",
    "vowel_high",
    "vowel_mid",
    "vowel_low",
    "vowel_front",
    "vowel_central",
    "vowel_back",
    "vowel_round_or_diphthong",
)
NUM_AF = len(AF_NAMES)

_AF_SPEC: dict[str, set[str]] = {
    # Vowels (15)
    "AA": {"is_vowel", "is_voiced", "vowel_low", "vowel_back"},
    "AE": {"is_vowel", "is_voiced", "vowel_low", "vowel_front"},
    "AH": {"is_vowel", "is_voiced", "vowel_mid", "vowel_central"},
    "AO": {"is_vowel", "is_voiced", "vowel_mid", "vowel_back", "vowel_round_or_diphthong"},
    "AW": {"is_vowel", "is_voiced", "vowel_low", "vowel_back", "vowel_round_or_diphthong"},
    "AY": {"is_vowel", "is_voiced", "vowel_low", "vowel_front", "vowel_round_or_diphthong"},
    "EH": {"is_vowel", "is_voiced", "vowel_mid", "vowel_front"},
    "ER": {"is_vowel", "is_voiced", "vowel_mid", "vowel_central", "manner_liquid_glide"},
    "EY": {"is_vowel", "is_voiced", "vowel_mid", "vowel_front", "vowel_round_or_diphthong"},
    "IH": {"is_vowel", "is_voiced", "vowel_high", "vowel_front"},
    "IY": {"is_vowel", "is_voiced", "vowel_high", "vowel_front"},
    "OW": {"is_vowel", "is_voiced", "vowel_mid", "vowel_back", "vowel_round_or_diphthong"},
    "OY": {"is_vowel", "is_voiced", "vowel_mid", "vowel_back", "vowel_round_or_diphthong"},
    "UH": {"is_vowel", "is_voiced", "vowel_high", "vowel_back", "vowel_round_or_diphthong"},
    "UW": {"is_vowel", "is_voiced", "vowel_high", "vowel_back", "vowel_round_or_diphthong"},
    # Stops (6)
    "P": {"manner_stop", "place_labial"},
    "B": {"is_voiced", "manner_stop", "place_labial"},
    "T": {"manner_stop", "place_dental_alveolar"},
    "D": {"is_voiced", "manner_stop", "place_dental_alveolar"},
    "K": {"manner_stop", "place_velar_glottal"},
    "G": {"is_voiced", "manner_stop", "place_velar_glottal"},
    # Affricates (2)
    "CH": {"manner_affricate", "place_postalveolar_palatal"},
    "JH": {"is_voiced", "manner_affricate", "place_postalveolar_palatal"},
    # Fricatives (9)
    "F": {"manner_fricative", "place_labial"},
    "V": {"is_voiced", "manner_fricative", "place_labial"},
    "TH": {"manner_fricative", "place_dental_alveolar"},
    "DH": {"is_voiced", "manner_fricative", "place_dental_alveolar"},
    "S": {"manner_fricative", "place_dental_alveolar"},
    "Z": {"is_voiced", "manner_fricative", "place_dental_alveolar"},
    "SH": {"manner_fricative", "place_postalveolar_palatal"},
    "ZH": {"is_voiced", "manner_fricative", "place_postalveolar_palatal"},
    "HH": {"manner_fricative", "place_velar_glottal"},
    # Nasals (3)
    "M": {"is_voiced", "manner_nasal", "place_labial"},
    "N": {"is_voiced", "manner_nasal", "place_dental_alveolar"},
    "NG": {"is_voiced", "manner_nasal", "place_velar_glottal"},
    # Liquids & Glides (4)
    "L": {"is_voiced", "manner_liquid_glide", "place_dental_alveolar"},
    "R": {"is_voiced", "manner_liquid_glide", "place_postalveolar_palatal"},
    "W": {"is_voiced", "manner_liquid_glide", "place_labial", "vowel_round_or_diphthong"},
    "Y": {"is_voiced", "manner_liquid_glide", "place_postalveolar_palatal"},
}


def build_af_matrix() -> np.ndarray:
    """Return float32 matrix of shape (40, NUM_AF); row 39 (omission/sil) is all zeros."""
    mat = np.zeros((len(ARPA_39) + 1, NUM_AF), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(AF_NAMES)}
    if set(_AF_SPEC.keys()) != set(ARPA_39):
        raise ValueError("Articulatory spec does not match ARPA_39 exactly")
    for idx, ph in enumerate(ARPA_39):
        feats = _AF_SPEC[ph]
        if not feats:
            raise ValueError(f"Empty articulatory spec for {ph}")
        for f in feats:
            mat[idx, name_to_idx[f]] = 1.0
    return mat


AF_MATRIX = build_af_matrix()


def articulatory_bce_loss(
    af_logits: torch.Tensor,
    realized_ids: torch.Tensor,
    af_table: torch.Tensor,
) -> torch.Tensor:
    """Multi-label BCE loss over valid non-omission realized phone slots (0 <= id < 39)."""
    valid = (realized_ids >= 0) & (realized_ids < len(ARPA_39))
    if not valid.any():
        return af_logits.sum() * 0.0
    targets = af_table[realized_ids[valid]]
    return F.binary_cross_entropy_with_logits(af_logits[valid], targets)
