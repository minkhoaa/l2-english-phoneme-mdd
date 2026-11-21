"""Unit tests for P0 zero-retrain fixes (koel_ipa_map_v2, sm13_e4_fixed) and P1 SeqMDDv2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.phoneme_review import ARPABET
from research.CONTROLLED_PILOT_AB_20261118_v1 import run_pilot_ab as pilot
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import (
    ALLOPHONE_TOKEN_IDS,
    SPECIAL_TOKENS,
    align_and_decide_v2,
    normalize_koel_tokens_v2,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    CONTINUOUS_SCALAR_NAMES,
    GAP_SCALAR_NAMES,
    SeqMDDv2,
    compute_multitask_mdd_loss,
)
from research.MDD_FIXES_AND_V2_20261121_v1.sm13_e4_fixed import (
    EXCLUDED_COMPETITOR_IDS,
    select_acoustic_competitors,
)


def test_koel_vocab_100pct_coverage():
    vocab = json.loads((pilot.WEIGHTS_DIR / "vocab.json").read_text(encoding="utf-8"))
    real_tokens = [tok for tok in vocab if tok not in SPECIAL_TOKENS]
    obs, unk, unmapped, obs_to_seg = normalize_koel_tokens_v2(real_tokens)
    assert len(unmapped) == 0, f"Unmapped Koel tokens remain: {unmapped}"
    assert len(unk) == 0
    assert len(obs) == len(obs_to_seg)


def test_allophone_token_ids_complete():
    assert set(ALLOPHONE_TOKEN_IDS.keys()) == set(ARPABET)
    for ph, ids in ALLOPHONE_TOKEN_IDS.items():
        assert len(ids) >= 1, f"Empty IDs for {ph}"
        for idx in ids:
            assert 1 <= idx <= 78, f"Invalid Koel token id {idx} for {ph}"


def test_allophone_and_l2_substitution_alignment():
    # Native allophones: kʰ->K, l̩->AH L, ɨ->AH, ʔ->T (and initial ʔ ignored as gap!)
    can = ["IH", "N", "K", "AE", "T", "S", "IH", "NG", "G", "AH", "L"]
    toks = ["ʔ", "ɪ", "n", "kʰ", "æ", "ʔ", "s", "ɪ", "ŋ", "ɡ", "l̩"]
    obs, unk, unmapped, _ = normalize_koel_tokens_v2(toks)
    assert not unmapped
    res = align_and_decide_v2(can, obs, unknown_indices=unk)
    assert res["target_calls"] == ["C"] * len(can)
    assert res["gap_calls"] == ["N"] * (len(can) + 1)

    # Genuine L2 substitutions (Spanish a->AE, β->V) must be flagged as S, NEVER '?'!
    can2 = ["V", "AE", "N"]
    toks2 = ["β", "a", "n"]
    obs2, unk2, unmapped2, _ = normalize_koel_tokens_v2(toks2)
    assert not unmapped2
    res2 = align_and_decide_v2(can2, obs2, unknown_indices=unk2)
    assert res2["target_calls"] == ["S", "S", "C"]

    # Flap ɾ is legal on T/D, but S on R!
    obs3, unk3, _, _ = normalize_koel_tokens_v2(["ɾ", "ɾ"])
    res3 = align_and_decide_v2(["T", "R"], obs3, unknown_indices=unk3)
    assert res3["target_calls"] == ["C", "S"]


def test_sm13_competitor_selection_picks_acoustic_peaks():
    T, C = 40, 43
    lp = np.full((T, C), -10.0, dtype=np.float64)
    # Position 0: canonical AA (0), actual acoustic peak at phone 30 ('S') and 15 ('F')
    lp[8:13, 30] = -0.1
    lp[8:13, 15] = -0.5
    lp[8:13, 42] = -0.01  # blank must be excluded
    lp[8:13, 32] = -0.02  # SPN must be excluded
    centers = np.array([10.0, 25.0], dtype=np.float32)
    cands = select_acoustic_competitors(lp, centers, T, [0, 10], top_k=4, window_radius=4)
    assert len(cands) == 2
    assert cands[0][0] == 30 and cands[0][1] == 15
    for c_list, can_id in zip(cands, [0, 10]):
        assert can_id not in c_list
        assert not (set(c_list) & EXCLUDED_COMPETITOR_IDS)


def test_seq_mdd_v2_forward_backward():
    B, L, W = 2, 12, 9
    model = SeqMDDv2(num_layers=6, h_dim=64, vocab_dim=79, d_model=64, dropout=0.1)
    out = model(
        h_vit=torch.randn(B, L, 6, 64),
        h_peak=torch.randn(B, L, 6, 64),
        lp_win=torch.randn(B, L, W, 79),
        lp_vit=torch.randn(B, L, 79),
        phone_ids=torch.randint(0, 39, (B, L)),
        prev_phone_ids=torch.randint(0, 41, (B, L)),
        next_phone_ids=torch.randint(0, 41, (B, L)),
        target_scalars=torch.randn(B, L, len(CONTINUOUS_SCALAR_NAMES)),
        h_gap=torch.randn(B, L + 1, 6, 64),
        lp_gap_mean=torch.randn(B, L + 1, 79),
        lp_gap_max=torch.randn(B, L + 1, 79),
        gap_scalars=torch.randn(B, L + 1, len(GAP_SCALAR_NAMES)),
        lengths=torch.tensor([L, L - 2]),
    )
    assert out["target_logits"].shape == (B, L, 3)
    assert out["gap_logits"].shape == (B, L + 1, 2)
    assert out["realized_logits"].shape == (B, L, 40)

    losses = compute_multitask_mdd_loss(
        out,
        target_labels=torch.randint(0, 3, (B, L)),
        gap_labels=torch.randint(0, 2, (B, L + 1)),
        realized_labels=torch.randint(0, 40, (B, L)),
        is_native_l1=torch.tensor([False, True]),
    )
    losses["loss"].backward()
    grad_ok = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert grad_ok == sum(1 for _ in model.parameters())
