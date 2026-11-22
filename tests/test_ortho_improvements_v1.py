"""Unit tests for ORTHO_IMPROVEMENTS_20261122_v1 (CPU, zero GPU, zero Eval gold)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
import torch

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    PHONE_TO_ID,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    NUM_REALIZED_CLASSES,
    compute_multitask_mdd_loss,
)
from research.ORTHO_IMPROVEMENTS_20261122_v1.calibration import (
    apply_ortho_policy,
    calibrate_ortho_policy,
    ensemble_predictions,
    fit_gap_platt_cal,
)
from research.ORTHO_IMPROVEMENTS_20261122_v1.losses import compute_ortho_mdd_loss
from research.ORTHO_IMPROVEMENTS_20261122_v1.prior_debias import (
    NUM_PHONES,
    SourcePhonePriorTable,
    apply_logit_adjustment,
    apply_phone_id_dropout,
)
from research.ORTHO_IMPROVEMENTS_20261122_v1.timit_freq_corrupt import (
    audit_timit_phone_rates,
    compute_timit_phone_distribution,
    corrupt_timit_utterance_freq_matched,
    rebuild_fit_with_freq_matched_timit,
)


def test_loss_exact_parity_with_baseline_when_ortho_disabled() -> None:
    torch.manual_seed(7)
    B, L = 4, 12
    outputs = {
        "target_logits": torch.randn(B, L, 3, requires_grad=True),
        "gap_logits": torch.randn(B, L + 1, 2, requires_grad=True),
        "realized_logits": torch.randn(B, L, NUM_REALIZED_CLASSES, requires_grad=True),
    }
    t_lab = torch.randint(0, 3, (B, L))
    t_lab[0, -2:] = -100
    g_lab = torch.randint(0, 2, (B, L + 1))
    g_lab[0, -2:] = -100
    r_lab = torch.randint(0, NUM_REALIZED_CLASSES, (B, L))
    r_lab[0, -2:] = -100
    is_l1 = torch.tensor([False, True, False, True])

    ref = compute_multitask_mdd_loss(
        outputs, t_lab, g_lab, r_lab, is_l1,
        gap_pos_weight=8.0, lambda_gap=0.6, lambda_realized=0.25, native_fa_weight=2.0,
    )
    got = compute_ortho_mdd_loss(
        outputs, t_lab, g_lab, r_lab, is_l1,
        tau_la=0.0, lambda_so_cond=0.0,
        gap_pos_weight=8.0, lambda_gap=0.6, lambda_realized=0.25, native_fa_weight=2.0,
    )
    assert torch.allclose(got["loss"], ref["loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(got["loss_target"], ref["loss_target"], atol=1e-6)
    assert torch.allclose(got["loss_gap"], ref["loss_gap"], atol=1e-6)
    assert torch.allclose(got["loss_realized"], ref["loss_realized"], atol=1e-6)
    assert float(got["loss_so_cond"]) == 0.0


def test_timit_freq_matched_eliminates_zipf_rare_phone_flood() -> None:
    rng = np.random.default_rng(123)
    # Zipfian synthetic phone distribution: first 5 phones very frequent, last 5 very rare
    weights = np.array([1.0 / (k + 1) ** 1.3 for k in range(len(ARPA_39))], dtype=np.float64)
    weights /= weights.sum()

    clean_map = {}
    fit_rows = []
    for i in range(600):
        uid = f"timit_syn_{i:04d}"
        seq = [ARPA_39[int(idx)] for idx in rng.choice(len(ARPA_39), size=35, p=weights)]
        clean_map[uid] = seq
        fit_rows.append({
            "source": "timit_train_l1_antishortcut",
            "uid": uid,
            "speaker": f"spk_{i % 50}",
            "audio_sha256": f"sha_{i}",
            "canonical_phones": list(seq),
            "gold_targets": ["C"] * len(seq),
            "gold_gaps": ["N"] * (len(seq) + 1),
            "realized_ids": [PHONE_TO_ID[p] for p in seq],
        })

    rebuilt, probs = rebuild_fit_with_freq_matched_timit(fit_rows, clean_map, seed=99)
    assert abs(sum(probs.values()) - 1.0) < 1e-9

    # Check structural invariants on every rebuilt row
    for r in rebuilt:
        assert len(r["canonical_phones"]) == len(r["gold_targets"]) == len(r["realized_ids"])
        assert len(r["gold_gaps"]) == len(r["canonical_phones"]) + 1
        for ph, g, rid in zip(r["canonical_phones"], r["gold_targets"], r["realized_ids"]):
            if g == "C":
                assert rid == PHONE_TO_ID[ph]
            elif g == "S":
                assert rid != PHONE_TO_ID[ph]
            elif g == "O":
                assert rid == 39

    audit = audit_timit_phone_rates(rebuilt)
    # Overall ~12-13% error, and even rare phones should NOT explode to 70-95%
    assert 10.0 <= audit["overall_target_err_pct"] <= 16.0
    assert audit["phone_err_rate_max_pct"] < 35.0
    assert audit["phone_err_rate_std_pct"] < 6.0

    # Determinism check
    c1 = corrupt_timit_utterance_freq_matched(clean_map["timit_syn_0000"], probs, seed=42, uid="u0")
    c2 = corrupt_timit_utterance_freq_matched(clean_map["timit_syn_0000"], probs, seed=42, uid="u0")
    assert c1 == c2


def test_source_phone_prior_table_and_logit_adjustment() -> None:
    fit_rows = [
        # L2A: DH is mostly S
        {"source": "l2arctic_dev18", "canonical_phones": ["DH"] * 40 + ["T"] * 40,
         "gold_targets": ["S"] * 30 + ["C"] * 10 + ["C"] * 38 + ["S"] * 2},
        # EpaDB: DH is mostly C, AH is mostly S
        {"source": "epadb_train", "canonical_phones": ["DH"] * 40 + ["AH"] * 40,
         "gold_targets": ["C"] * 39 + ["S"] * 1 + ["S"] * 30 + ["C"] * 10},
        # TIMIT: uniform low error
        {"source": "timit_train_l1_antishortcut", "canonical_phones": ["DH"] * 40 + ["T"] * 40,
         "gold_targets": ["C"] * 36 + ["S"] * 4 + ["C"] * 36 + ["S"] * 4},
    ]
    tbl = SourcePhonePriorTable.from_fit_rows(fit_rows, shrinkage_alpha=10.0)
    assert tbl.probs.shape == (3, NUM_PHONES + 2, 3)
    assert np.all(np.isfinite(tbl.log_probs))
    assert np.allclose(tbl.probs.sum(axis=-1), 1.0, atol=1e-5)

    dh_id = PHONE_TO_ID["DH"]
    # L2A (source 0) must have much higher P(S|DH) than EpaDB (source 1)
    assert tbl.probs[0, dh_id, 1] > 4.0 * tbl.probs[1, dh_id, 1]

    # Logit adjustment adds tau_la * log_prior; tau_la=0 is exact identity
    logits = torch.zeros(2, 3, 3)
    src_ids = torch.tensor([0, 1])
    ph_ids = torch.tensor([[dh_id, PHONE_TO_ID["T"], NUM_PHONES], [dh_id, PHONE_TO_ID["AH"], 40]])
    log_tbl = tbl.to_log_prior_tensor(torch.device("cpu"))
    assert torch.equal(apply_logit_adjustment(logits, src_ids, ph_ids, log_tbl, tau_la=0.0), logits)
    adj = apply_logit_adjustment(logits, src_ids, ph_ids, log_tbl, tau_la=1.0)
    assert float(adj[0, 0, 1]) > float(adj[1, 0, 1])

    # Phone-ID dropout must be exact no-op when training=False
    assert torch.equal(apply_phone_id_dropout(ph_ids, p_drop=0.9, training=False), ph_ids)
    torch.manual_seed(0)
    dropped = apply_phone_id_dropout(torch.zeros(100, dtype=torch.long), p_drop=0.5, training=True)
    frac_unk = float((dropped == NUM_PHONES).float().mean())
    assert 0.30 < frac_unk < 0.70


def test_ortho_loss_cond_so_and_logit_adjustment_gradients() -> None:
    torch.manual_seed(11)
    B, L = 3, 8
    t_logits = torch.randn(B, L, 3, requires_grad=True)
    g_logits = torch.randn(B, L + 1, 2, requires_grad=True)
    r_logits = torch.randn(B, L, NUM_REALIZED_CLASSES, requires_grad=True)
    outputs = {"target_logits": t_logits, "gap_logits": g_logits, "realized_logits": r_logits}

    t_lab = torch.tensor([
        [0, 0, 1, 2, 0, 1, -100, -100],
        [0, 2, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    g_lab = torch.zeros(B, L + 1, dtype=torch.long)
    g_lab[0, 2] = 1
    r_lab = torch.zeros(B, L, dtype=torch.long)
    src_ids = torch.tensor([0, 1, 2])
    ph_ids = torch.randint(0, NUM_PHONES, (B, L))
    log_tbl = torch.log(torch.full((3, NUM_PHONES + 2, 3), 1.0 / 3.0))

    res = compute_ortho_mdd_loss(
        outputs, t_lab, g_lab, r_lab,
        source_ids=src_ids, phone_ids=ph_ids, log_prior_table=log_tbl,
        tau_la=0.8, lambda_so_cond=0.5,
    )
    assert float(res["loss_so_cond"]) > 0.0
    res["loss"].backward()
    assert t_logits.grad is not None and torch.isfinite(t_logits.grad).all()
    assert g_logits.grad is not None and torch.isfinite(g_logits.grad).all()

    # Batch with zero S/O slots must return finite zero loss_so_cond
    all_c = torch.zeros(B, L, dtype=torch.long)
    res_c = compute_ortho_mdd_loss(outputs, all_c, g_lab, lambda_so_cond=0.5)
    assert float(res_c["loss_so_cond"]) == 0.0
    assert torch.isfinite(res_c["loss"])


def test_gap_platt_calibration_removes_pos_weight_shift() -> None:
    rng = np.random.default_rng(2026)
    n = 4000
    # True underlying log-odds with ~2% base rate of I
    true_z = rng.normal(loc=-4.2, scale=2.2, size=n)
    y_bin = rng.binomial(1, expit(true_z))
    gold = np.where(y_bin == 1, "I", "N")

    # Simulate model trained with gap_pos_weight=8.0 -> logits shifted by +log(8.0) and slightly sharp
    raw_z = 1.25 * true_z + np.log(8.0)
    info = fit_gap_platt_cal(raw_z, gold, is_probs=False, known_pos_weight=8.0)

    assert info["cal_nll_after"] < info["cal_nll_before"]
    assert info["cal_ece_after"] < info["cal_ece_before"]
    assert abs(info["cal_pred_i_after"] - info["cal_emp_i_rate"]) < 0.01
    # Bias should move close to -log(8.0) ~= -2.079
    assert -2.6 < info["gap_bias"] < -1.5


def test_calibrate_ortho_policy_rho_o_and_3seed_ensemble() -> None:
    rng = np.random.default_rng(77)
    n_l2, m_l2, n_l1, m_l1 = 1500, 1550, 600, 620
    # True target labels: 78% C, 14% S, 8% O
    u = rng.random(n_l2)
    t_gold = np.where(u < 0.78, "C", np.where(u < 0.92, "S", "O"))
    g_gold = np.where(rng.random(m_l2) < 0.025, "I", "N")

    # Build 3 noisy seeds with systematic O over-prediction (logit_O biased +1.2)
    # and Gap logits biased by +log(8.0)
    base_t = np.zeros((n_l2, 3), dtype=np.float64)
    base_t[t_gold == "C"] = [2.2, -1.2, -1.5]
    base_t[t_gold == "S"] = [-1.0, 1.4, 2.2]  # deliberately biased toward O -> rho_o tuning should fix!
    base_t[t_gold == "O"] = [-1.2, -0.5, 3.0]

    base_g = np.where(g_gold == "I", 2.0, -3.5) + np.log(8.0)
    base_l1_t = np.tile(np.array([3.0, -1.5, -1.5]), (n_l1, 1))
    base_l1_g = np.full(m_l1, -4.5 + np.log(8.0))

    seed_t_l2, seed_g_l2, seed_t_l1, seed_g_l1 = [], [], [], []
    for k in range(3):
        seed_t_l2.append(base_t + rng.normal(0, 1.2, size=base_t.shape))
        seed_g_l2.append(base_g + rng.normal(0, 1.0, size=base_g.shape))
        seed_t_l1.append(base_l1_t + rng.normal(0, 0.8, size=base_l1_t.shape))
        seed_g_l1.append(base_l1_g + rng.normal(0, 0.8, size=base_l1_g.shape))

    ens_t_l2, ens_g_l2, is_p = ensemble_predictions(seed_t_l2, seed_g_l2, mode="logit_mean")
    ens_t_l1, ens_g_l1, _ = ensemble_predictions(seed_t_l1, seed_g_l1, mode="logit_mean")
    assert is_p is False

    pol_default, _ = calibrate_ortho_policy(
        ens_t_l2, t_gold, ens_g_l2, g_gold, ens_t_l1, ens_g_l1,
        is_probs=False, tune_rho_o=False, fit_gap_temp=False,
        ensemble_mode="logit_mean", n_seeds=3,
    )
    pol_tuned, _ = calibrate_ortho_policy(
        ens_t_l2, t_gold, ens_g_l2, g_gold, ens_t_l1, ens_g_l1,
        is_probs=False, tune_rho_o=True, fit_gap_temp=True,
        ensemble_mode="logit_mean", n_seeds=3,
    )
    # Tuning rho_o must beat fixed 0.50 on biased S->O logits, and Platt must pull tau_g down from 0.85+
    assert pol_tuned.cal_guard_met is True
    assert pol_tuned.rho_o > 0.50
    assert pol_tuned.cal_l2_strict_f1 > pol_default.cal_l2_strict_f1 + 3.0
    assert pol_tuned.tau_g < pol_default.tau_g

    # Single-seed average Cal F1 should trail 3-seed ensemble Cal F1
    single_f1s = []
    for k in range(3):
        ps, _ = calibrate_ortho_policy(
            seed_t_l2[k], t_gold, seed_g_l2[k], g_gold, seed_t_l1[k], seed_g_l1[k],
            is_probs=False, tune_rho_o=True, fit_gap_temp=True,
        )
        single_f1s.append(ps.cal_l2_strict_f1)
    assert pol_tuned.cal_l2_strict_f1 > float(np.mean(single_f1s))

    # apply_ortho_policy smoke check
    pred = apply_ortho_policy(ens_t_l2[:10], ens_g_l2[:10], pol_tuned, is_probs=False)
    assert pred["target_call"].shape == (10,)
    assert pred["gap_call"].shape == (10,)
    assert set(pred["target_call"]).issubset({"C", "S", "O"})
    assert set(pred["gap_call"]).issubset({"N", "I"})


def test_compute_timit_phone_distribution_rejects_unknown_gracefully() -> None:
    probs = compute_timit_phone_distribution([["T", "IY", "T", "UNK_NOISE"]], smoothing=1.0)
    assert len(probs) == len(ARPA_39)
    assert probs["T"] > probs["IY"] > probs["ZH"]
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        corrupt_timit_utterance_freq_matched(["T"], probs, seed=1, uid="x", p_ins=0.6, p_omi=0.5)
