"""Unit tests for single-factor isolation variants of RepairedCodeFixV2."""

from __future__ import annotations

import numpy as np
import torch

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import (
    collate_sf,
    reset_sf_gap,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_gap_mask_ablation import (
    is_gap_param,
    split_param_shas,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.single_factor_models import (
    FACTOR_ARMS,
    SingleFactorV2,
    pool_features_with_hvit,
)


def _synth_sf_row(rng: np.random.Generator, uid: str, length: int):
    phones = ["M", "AY", "K", "AE", "T", "S", "IH", "T", "D", "R"][:length]
    T = max(length * 3 + 4, length + 2)
    h = rng.normal(0.0, 10.0, size=(T, 6, 1024)).astype(np.float16)
    raw = rng.normal(-15.0, 4.0, size=(T, 79)).astype(np.float32)
    lp = raw - np.log(np.exp(raw).sum(axis=-1, keepdims=True))
    feat = pool_features_with_hvit(h, lp, phones)
    row = {
        "uid": uid,
        "speaker": "SPK1",
        "source": "l2arctic_dev18",
        "canonical_phones": phones,
        "gold_targets": ["C"] * (length - 1) + ["S"],
        "gold_gaps": ["N"] * length + ["I"],
        "realized_ids": [0] * length,
    }
    return row, feat


def test_all_single_factor_arms_batch_invariant_and_eos_routed():
    rng = np.random.default_rng(29)
    r1, f1 = _synth_sf_row(rng, "u1", 4)
    r2, f2 = _synth_sf_row(rng, "u2", 9)
    norm = {
        "target_scalars": (np.zeros(13, np.float32), np.ones(13, np.float32)),
        "gap_scalars": (np.zeros(8, np.float32), np.ones(8, np.float32)),
    }
    for arm_name, cfg in FACTOR_ARMS.items():
        torch.manual_seed(31)
        model = SingleFactorV2(cfg).eval()
        inp_b, _ = collate_sf([r1, r2], [f1, f2], norm, torch.device("cpu"), labels=False)
        inp_1, _ = collate_sf([r1], [f1], norm, torch.device("cpu"), labels=False)
        inp_2, _ = collate_sf([r2], [f2], norm, torch.device("cpu"), labels=False)
        with torch.no_grad():
            ob = model(**inp_b)
            o1 = model(**inp_1)
            o2 = model(**inp_2)
        assert (ob["target_logits"][0, :4] - o1["target_logits"][0]).abs().max().item() < 1e-5, arm_name
        assert (ob["gap_logits"][0, :5] - o1["gap_logits"][0]).abs().max().item() < 1e-5, arm_name
        assert (ob["target_logits"][1, :9] - o2["target_logits"][0]).abs().max().item() < 1e-5, arm_name
        assert (ob["gap_logits"][1, :10] - o2["gap_logits"][0]).abs().max().item() < 1e-5, arm_name

        # Verify EOS gradient on shorter utterance & Phase-2 freeze contract
        model.train()
        ob2 = model(**inp_b)
        ob2["gap_logits"][0, 4].sum().backward()
        assert model.eos_state.grad is not None and model.eos_state.grad.abs().sum().item() > 0.0

        fr0, gp0 = split_param_shas(model.state_dict())
        reset_sf_gap(model, seed=123)
        fr1, gp1 = split_param_shas(model.state_dict())
        assert fr0 == fr1 and gp0 != gp1
        assert all(p.requires_grad == is_gap_param(n) for n, p in model.named_parameters())
