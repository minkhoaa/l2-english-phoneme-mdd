"""Unit tests for REAL_L2_SO762_20261122_v1 (CPU, zero GPU, zero Eval gold)."""

from __future__ import annotations

import torch

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    NUM_REALIZED_CLASSES,
    compute_multitask_mdd_loss,
)
from research.REAL_L2_SO762_20261122_v1.marginal_loss import compute_so762_hybrid_mdd_loss
from research.REAL_L2_SO762_20261122_v1.prepare_so762 import parse_so762_row


def test_hybrid_loss_exact_parity_with_baseline() -> None:
    torch.manual_seed(42)
    B, L = 4, 10
    outputs = {
        "target_logits": torch.randn(B, L, 3, requires_grad=True),
        "gap_logits": torch.randn(B, L + 1, 2, requires_grad=True),
        "realized_logits": torch.randn(B, L, NUM_REALIZED_CLASSES, requires_grad=True),
    }
    t_lab = torch.randint(0, 3, (B, L))
    t_lab[0, -2:] = -100
    g_lab = torch.randint(0, 2, (B, L + 1))
    r_lab = torch.randint(0, NUM_REALIZED_CLASSES, (B, L))
    is_l1 = torch.tensor([False, True, False, True])

    ref = compute_multitask_mdd_loss(
        outputs, t_lab, g_lab, r_lab, is_l1,
        gap_pos_weight=8.0, lambda_gap=0.6, lambda_realized=0.25, native_fa_weight=1.5,
    )
    got = compute_so762_hybrid_mdd_loss(
        outputs, t_lab, g_lab, r_lab, is_l1,
        soft_err_prob=None,
        gap_pos_weight=8.0, lambda_gap=0.6, lambda_realized=0.25, native_fa_weight=1.5,
    )
    assert torch.allclose(got["loss"], ref["loss"], atol=1e-6)
    assert torch.allclose(got["loss_target"], ref["loss_target"], atol=1e-6)


def test_marginal_binary_preserves_s_vs_o_ratio_gradient() -> None:
    # For a pure marginal binary error slot (q=1.0), dL/dz_S / dL/dz_O must equal P(S)/P(O)
    logits = torch.tensor([[[1.0, 0.5, -0.5]]], requires_grad=True)  # z_S - z_O = 1.0 => P(S)/P(O) = e^1
    outputs = {
        "target_logits": logits,
        "gap_logits": torch.zeros(1, 2, 2),
        "realized_logits": torch.zeros(1, 1, NUM_REALIZED_CLASSES),
    }
    t_lab = torch.tensor([[-100]])
    g_lab = torch.tensor([[-100, -100]])
    r_lab = torch.tensor([[-100]])
    soft_q = torch.tensor([[1.0]])

    res = compute_so762_hybrid_mdd_loss(outputs, t_lab, g_lab, r_lab, soft_err_prob=soft_q)
    res["loss"].backward()
    g = logits.grad[0, 0]
    # Pushing error up: g_C > 0, g_S < 0, g_O < 0, and g_S / g_O == exp(z_S - z_O)
    assert float(g[0]) > 0.0 and float(g[1]) < 0.0 and float(g[2]) < 0.0
    ratio = float(g[1] / g[2])
    assert abs(ratio - float(torch.exp(torch.tensor(1.0)))) < 1e-5


def test_parse_so762_row_typed_and_unk() -> None:
    raw = {
        "audio": {"bytes": b"fake-wav-bytes"},
        "speaker": "0042",
        "age": 24,
        "gender": "f",
        "text": "GREY THEME",
        "accuracy": 7.0,
        "words": [
            {
                "text": "GREY",
                "phones": ["G", "R", "EY0"],
                "phones-accuracy": [0.4, 0.4, 2.0],
                "mispronunciations": [
                    {"canonical-phone": "G", "index": 0, "pronounced-phone": "F"},
                    {"canonical-phone": "R", "index": 1, "pronounced-phone": "<unk>"},
                ],
            },
            {
                "text": "THEME",
                "phones": ["TH", "IY0", "M"],
                "phones-accuracy": [0.0, 2.0, 1.4],
                "mispronunciations": [
                    {"canonical-phone": "TH", "index": 0, "pronounced-phone": "<DEL>"},
                ],
            },
        ],
    }
    r = parse_so762_row(raw, 7)
    assert r["canonical_phones"] == ["G", "R", "EY", "TH", "IY", "M"]
    assert r["explicit_type"] == ["S", "UNK_ERR", "", "O", "", ""]
    assert r["explicit_real_id"][3] == 39  # <DEL>
    assert r["explicit_real_id"][1] == -100  # <unk>
    assert r["explicit_real_id"][2] >= 0  # score 2.0 => canonical realized
