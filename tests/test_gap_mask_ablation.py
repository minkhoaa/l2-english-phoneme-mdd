"""Unit tests for isolated TIMIT Gap-I mask ablation & independent replay counter."""

from __future__ import annotations

import numpy as np
import torch

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.repaired_v2_arch import RepairedCodeFixV2
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.replay_codefix_audit import (
    independent_count,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_gap_mask_ablation import (
    compute_isolated_gap_loss,
    gap_forward_from_seq,
    is_gap_param,
    precompute_frozen_seq,
    reset_gap_branch,
    split_param_shas,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_code_fixes_ablation import (
    collate_codefix,
)
from tests.test_seq_mdd_v2_code_fixes import _synth_row_and_feat


def test_independent_count_matches_known_confusion():
    tg = np.array(["C", "S", "O", "S", "O", "C", "unknown"])
    tc = np.array(["C", "S", "O", "O", "S", "S", "S"])
    gg = np.array(["N", "I", "N", "I", "unknown"])
    gc = np.array(["N", "I", "I", "N", "I"])
    c = independent_count(tg, tc, gg, gc)
    assert c["S"] == {"tp": 1, "fp": 2, "fn": 1, "p": 1 / 3, "r": 0.5, "f1": 0.4}
    assert c["O"] == {"tp": 1, "fp": 1, "fn": 1, "p": 0.5, "r": 0.5, "f1": 0.5}
    assert c["I"] == {"tp": 1, "fp": 1, "fn": 1, "p": 0.5, "r": 0.5, "f1": 0.5}
    assert c["s_to_o"] == 1 and c["o_to_s"] == 1
    assert c["strict"]["tp"] == 3 and c["strict"]["fp"] == 4 and c["strict"]["fn"] == 3
    assert c["binary"]["tp"] == 5 and c["binary"]["fp"] == 2 and c["binary"]["fn"] == 1


def test_gap_isolation_preserves_frozen_weights_and_exact_forward():
    rng = np.random.default_rng(19)
    r1, f1 = _synth_row_and_feat(rng, "u1", 5, "l2arctic_dev18")
    norm = {
        "target_scalars": (np.zeros(13, np.float32), np.ones(13, np.float32)),
        "gap_scalars": (np.zeros(8, np.float32), np.ones(8, np.float32)),
    }
    torch.manual_seed(1)
    model = RepairedCodeFixV2(scalar_drop_p=0.0, gap_lp_drop_p=0.0).eval()
    fr0, gp0 = split_param_shas(model.state_dict())
    reset_gap_branch(model, seed=99)
    fr1, gp1 = split_param_shas(model.state_dict())
    assert fr0 == fr1
    assert gp0 != gp1
    assert all(p.requires_grad == is_gap_param(n) for n, p in model.named_parameters())

    # Precomputed seq + gap_forward_from_seq must match full forward gap_logits exactly
    seqs = precompute_frozen_seq(model, [r1], [f1], norm, torch.device("cpu"))
    inp_full, _ = collate_codefix([r1], [f1], norm, torch.device("cpu"), labels=False)
    with torch.no_grad():
        out_full = model(**inp_full)["gap_logits"]
        out_iso = gap_forward_from_seq(
            model,
            torch.from_numpy(seqs[0]).unsqueeze(0),
            inp_full["h_gap"],
            inp_full["lp_gap_mean"],
            inp_full["lp_gap_max"],
            inp_full["gap_scalars"],
            inp_full["lengths"],
        )
    assert (out_full - out_iso).abs().max().item() < 1e-5


def test_isolated_gap_loss_masking_single_variable():
    logits = torch.zeros((2, 3, 2), requires_grad=True)
    y = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.long)
    is_l1 = torch.tensor([True, False], dtype=torch.bool)

    loss_mask = compute_isolated_gap_loss(logits, y, is_l1, mask_timit_gap_i=True)
    loss_mask.backward()
    # L1 I slot (0, 1) masked to 0 grad; L1 N (0, 0) and L2 I (1, 1) remain active
    assert logits.grad[0, 1].abs().sum().item() == 0.0
    assert logits.grad[0, 0].abs().sum().item() > 0.0
    assert logits.grad[1, 1].abs().sum().item() > 0.0

    logits2 = torch.zeros((2, 3, 2), requires_grad=True)
    loss_keep = compute_isolated_gap_loss(logits2, y, is_l1, mask_timit_gap_i=False)
    loss_keep.backward()
    assert logits2.grad[0, 1].abs().sum().item() > 0.0
