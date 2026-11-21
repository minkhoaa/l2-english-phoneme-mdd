"""Unit tests for Code Fixes #1, #2, #3 (RepairedCodeFixV2 + loss + pooling)."""

from __future__ import annotations

import numpy as np
import torch

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.repaired_v2_arch import (
    CodeFixConfig,
    RepairedCodeFixV2,
    compute_codefix_loss,
    pool_features_codefix,
    scale_lp,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_code_fixes_ablation import (
    collate_codefix,
)


def _synth_row_and_feat(rng: np.random.Generator, uid: str, length: int, source: str):
    phones = ["M", "AY", "K", "AE", "T", "S", "IH", "T", "D", "R"][:length]
    T = max(length * 3 + 4, length + 2)
    h = rng.normal(0.0, 10.0, size=(T, 6, 1024)).astype(np.float16)
    # Give L24 3x larger norm than L12 (like real XLSR)
    h[:, 5, :] = (h[:, 5, :].astype(np.float32) * 3.0).astype(np.float16)
    raw = rng.normal(-15.0, 4.0, size=(T, 79)).astype(np.float32)
    lp = raw - np.log(np.exp(raw).sum(axis=-1, keepdims=True))
    feat = pool_features_codefix(h, lp, phones)
    row = {
        "uid": uid,
        "speaker": "SPK1",
        "source": source,
        "canonical_phones": phones,
        "gold_targets": ["C"] * (length - 1) + ["S"],
        "gold_gaps": ["N"] * length + ["I"],
        "realized_ids": [0] * length,
    }
    return row, feat


def test_batch_invariance_and_eos_routing():
    rng = np.random.default_rng(7)
    r1, f1 = _synth_row_and_feat(rng, "u1", 4, "l2arctic_dev18")
    r2, f2 = _synth_row_and_feat(rng, "u2", 9, "l2arctic_dev18")
    norm = {
        "target_scalars": (np.zeros(13, np.float32), np.ones(13, np.float32)),
        "gap_scalars": (np.zeros(8, np.float32), np.ones(8, np.float32)),
    }
    torch.manual_seed(11)
    model = RepairedCodeFixV2(scalar_drop_p=0.35, gap_lp_drop_p=0.25).eval()

    inp_b, _ = collate_codefix([r1, r2], [f1, f2], norm, torch.device("cpu"), labels=False)
    inp_1, _ = collate_codefix([r1], [f1], norm, torch.device("cpu"), labels=False)
    inp_2, _ = collate_codefix([r2], [f2], norm, torch.device("cpu"), labels=False)

    with torch.no_grad():
        out_b = model(**inp_b)
        out_1 = model(**inp_1)
        out_2 = model(**inp_2)

    assert (out_b["target_logits"][0, :4] - out_1["target_logits"][0]).abs().max().item() < 1e-5
    assert (out_b["gap_logits"][0, :5] - out_1["gap_logits"][0]).abs().max().item() < 1e-5
    assert (out_b["target_logits"][1, :9] - out_2["target_logits"][0]).abs().max().item() < 1e-5
    assert (out_b["gap_logits"][1, :10] - out_2["gap_logits"][0]).abs().max().item() < 1e-5

    # Verify EOS gradient flows on padded shorter utterance u1 (terminal gap index 4)
    model.train()
    inp_b2, _ = collate_codefix([r1, r2], [f1, f2], norm, torch.device("cpu"), labels=False)
    out_train = model(**inp_b2)
    out_train["gap_logits"][0, 4].sum().backward()
    assert model.eos_state.grad is not None
    assert model.eos_state.grad.abs().sum().item() > 0.0


def test_scale_lp_bounds_and_branch_balance():
    raw_lp = torch.tensor([[-28.5, -18.0, -5.0, -0.01]], dtype=torch.float32)
    sc = scale_lp(raw_lp)
    assert sc.shape == (1, 8)
    assert sc.min().item() >= -1.0 - 1e-6
    assert sc.max().item() <= 1.0 + 1e-6


def test_modality_dropout_active_only_in_train_and_routes_grad_to_h_win():
    rng = np.random.default_rng(13)
    r1, f1 = _synth_row_and_feat(rng, "u1", 6, "l2arctic_dev18")
    norm = {
        "target_scalars": (np.zeros(13, np.float32), np.ones(13, np.float32)),
        "gap_scalars": (np.zeros(8, np.float32), np.ones(8, np.float32)),
    }
    torch.manual_seed(17)
    model = RepairedCodeFixV2(scalar_drop_p=0.99, gap_lp_drop_p=0.99)
    inp, labels = collate_codefix([r1], [f1], norm, torch.device("cpu"), labels=True)

    # Eval must be deterministic even with scalar_drop_p=0.99
    model.eval()
    with torch.no_grad():
        a = model(**inp)["target_logits"]
        b = model(**inp)["target_logits"]
    assert torch.equal(a, b)

    # Train must route non-zero gradient to h_win_attn, proj_h_peak, and proj_lp_win
    model.train()
    cfg = CodeFixConfig(mode="fix123_full", scalar_drop_p=0.99, gap_lp_drop_p=0.99)
    loss = compute_codefix_loss(model(**inp), *labels, cfg)["loss"]
    loss.backward()
    assert model.proj_h_win[0].weight.grad.abs().sum().item() > 0.0
    assert model.proj_h_peak[0].weight.grad.abs().sum().item() > 0.0
    assert model.proj_lp_win[0].weight.grad.abs().sum().item() > 0.0


def test_timit_loss_weights_clean_c_only_and_masks_synthetic_gap_i():
    # Two utterances: idx 0 is L1 TIMIT, idx 1 is L2-ARCTIC
    t_logits = torch.zeros((2, 3, 3), requires_grad=True)
    g_logits = torch.zeros((2, 4, 2), requires_grad=True)
    r_logits = torch.zeros((2, 3, 40), requires_grad=True)
    # TIMIT has [C, S, C] and gaps [N, I, N, N]; L2 has [C, S, C] and gaps [N, I, N, N]
    y_t = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.long)
    y_g = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.long)
    y_r = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.long)
    is_l1 = torch.tensor([True, False], dtype=torch.bool)

    cfg = CodeFixConfig(mode="fix123_full", fix_timit_loss=True, native_fa_weight=1.5)
    out = {"target_logits": t_logits, "gap_logits": g_logits, "realized_logits": r_logits}
    loss = compute_codefix_loss(out, y_t, y_g, y_r, is_l1, cfg)["loss"]
    loss.backward()

    # L1 clean C slot (0, 0) must have 1.5x gradient of L2 clean C slot (1, 0)
    g_l1_c = t_logits.grad[0, 0].abs().sum().item()
    g_l2_c = t_logits.grad[1, 0].abs().sum().item()
    assert abs(g_l1_c / g_l2_c - 1.5) < 1e-5

    # L1 synthetic S slot (0, 1) must have 1.0x gradient (same as L2 S slot (1, 1))
    g_l1_s = t_logits.grad[0, 1].abs().sum().item()
    g_l2_s = t_logits.grad[1, 1].abs().sum().item()
    assert abs(g_l1_s / g_l2_s - 1.0) < 1e-5

    # L1 synthetic Gap I slot (0, 1) must be masked -> exact 0.0 gradient!
    assert g_logits.grad[0, 1].abs().sum().item() == 0.0
    # L1 Gap N slot (0, 0) and L2 Gap I slot (1, 1) must stay active (> 0)
    assert g_logits.grad[0, 0].abs().sum().item() > 0.0
    assert g_logits.grad[1, 1].abs().sum().item() > 0.0
