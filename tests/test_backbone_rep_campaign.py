"""Unit tests for BACKBONE_REP_CAMPAIGN_20261122_v1 (CPU, zero network/Eval gold)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from research.BACKBONE_REP_CAMPAIGN_20261122_v1 import common
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.articulatory import (
    AF_MATRIX,
    NUM_AF,
    articulatory_bce_loss,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.backbones import (
    _viterbi_charsiu_spans,
    pool_slots_from_frames,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import (
    is_gap_param_name,
    open_arm_dir,
    param_group_sha,
    verify_environment_and_custody,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.models import (
    FRAME_WIN_GAP,
    FRAME_WIN_TARGET,
    DualRepConfig,
    DualRepSingleFactorV2,
    FrameXAttnConfig,
    FrameXAttnMDD,
    LoRALinear,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.stages_0_3 import probe_gate, xattn_inference_ranges
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.stages_4_6 import (
    AF_BUNDLES,
    PHONE_TO_BUNDLE,
    _verify_sealed_arm,
    adapter_self_attention,
    closest_recall_policy,
    warm_ctc_loss,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import save_json, save_jsonl, sha_file
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import KEY, scores_to_rows, seal_predictions


def _dummy_base_batch(B: int = 2, N: int = 7, lengths=(7, 5)):
    return {
        "h_peak": torch.randn(B, N, 6, 1024),
        "h_vit": torch.randn(B, N, 6, 1024),
        "lp_vit": torch.randn(B, N, 79),
        "lp_win": torch.randn(B, N, 9, 79),
        "phone_ids": torch.randint(0, 39, (B, N)),
        "prev_phone_ids": torch.randint(0, 41, (B, N)),
        "next_phone_ids": torch.randint(0, 41, (B, N)),
        "target_scalars": torch.randn(B, N, 13),
        "h_gap": torch.randn(B, N + 1, 6, 1024),
        "lp_gap_mean": torch.randn(B, N + 1, 79),
        "lp_gap_max": torch.randn(B, N + 1, 79),
        "gap_scalars": torch.randn(B, N + 1, 8),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def test_articulatory_matrix_and_loss():
    assert AF_MATRIX.shape == (40, NUM_AF)
    assert (AF_MATRIX[:39].sum(axis=1) >= 2.0).all()
    assert float(AF_MATRIX[39].sum()) == 0.0
    logits = torch.zeros(2, 5, NUM_AF, requires_grad=True)
    rids = torch.tensor([[0, 12, 39, -100, 5], [1, 2, 3, 4, 39]])
    loss = articulatory_bce_loss(logits, rids, torch.from_numpy(AF_MATRIX))
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_pool_slots_from_frames_shapes_and_finite():
    frames = np.random.randn(100, 6, 64).astype(np.float32)
    starts = np.array([0, 20, 50, 80], dtype=np.int32)
    ends = np.array([20, 50, 80, 100], dtype=np.int32)
    peaks = np.array([10, 35, 65, 90], dtype=np.int32)
    out = pool_slots_from_frames(frames, starts, ends, peaks, t_koel=100)
    assert out["slot_mean"].shape == (4, 6, 64)
    assert out["slot_peak"].shape == (4, 6, 64)
    assert out["slot_std"].shape == (4, 6, 64)
    assert out["gap_mean"].shape == (5, 6, 64)
    assert out["gap_max"].shape == (5, 6, 64)
    for arr in out.values():
        assert arr.dtype == np.float16 and np.isfinite(arr).all()


def test_dualrep_forward_and_phase2_isolation():
    cfg = DualRepConfig(
        arm="A4_test", ext1_layers=6, ext1_dim=128, ext2_layers=5, ext2_dim=96, aux_scalar_dim=11
    )
    model = DualRepSingleFactorV2(cfg)
    batch = _dummy_base_batch(B=2, N=7, lengths=(7, 5))
    batch.update(
        {
            "ext1_slot_mean": torch.randn(2, 7, 6, 128),
            "ext1_slot_peak": torch.randn(2, 7, 6, 128),
            "ext1_slot_std": torch.randn(2, 7, 6, 128),
            "ext1_gap_mean": torch.randn(2, 8, 6, 128),
            "ext1_gap_max": torch.randn(2, 8, 6, 128),
            "ext2_slot_mean": torch.randn(2, 7, 5, 96),
            "ext2_slot_peak": torch.randn(2, 7, 5, 96),
            "ext2_slot_std": torch.randn(2, 7, 5, 96),
            "ext2_gap_mean": torch.randn(2, 8, 5, 96),
            "ext2_gap_max": torch.randn(2, 8, 5, 96),
            "aux_scalars": torch.randn(2, 7, 11),
        }
    )
    out = model(**batch)
    assert out["target_logits"].shape == (2, 7, 3)
    assert out["gap_logits"].shape == (2, 8, 2)
    assert out["realized_logits"].shape == (2, 7, 40)
    assert out["af_logits"].shape == (2, 7, NUM_AF)

    # Verify Phase-2 gap parameter isolation
    sha_before = param_group_sha(model, lambda n: not is_gap_param_name(n))
    for name, p in model.named_parameters():
        p.requires_grad = is_gap_param_name(name)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss = model(**batch)["gap_logits"].pow(2).mean()
    loss.backward()
    opt.step()
    sha_after = param_group_sha(model, lambda n: not is_gap_param_name(n))
    assert sha_before == sha_after


def test_frame_xattn_forward_single_and_dual():
    for use_koel in (False, True):
        cfg = FrameXAttnConfig(arm="B_test", ext_layers=4, ext_dim=128, use_koel_frame_win=use_koel)
        model = FrameXAttnMDD(cfg)
        batch = _dummy_base_batch(B=2, N=6, lengths=(6, 4))
        batch.update(
            {
                "ext_t_win": torch.randn(2, 6, FRAME_WIN_TARGET, 4, 128),
                "ext_t_mask": torch.ones(2, 6, FRAME_WIN_TARGET, dtype=torch.bool),
                "ext_g_win": torch.randn(2, 7, FRAME_WIN_GAP, 4, 128),
                "ext_g_mask": torch.ones(2, 7, FRAME_WIN_GAP, dtype=torch.bool),
            }
        )
        if use_koel:
            batch["koel_t_win"] = torch.randn(2, 6, FRAME_WIN_TARGET, 2, 1024)
            batch["koel_t_mask"] = batch["ext_t_mask"].clone()
            batch["koel_t_mask"][1, 4:] = False
        out = model(**batch)
        assert out["target_logits"].shape == (2, 6, 3)
        assert out["gap_logits"].shape == (2, 7, 2)
        assert torch.isfinite(out["attn_entropy"]) and out["attn_entropy"].item() > 0.0


def test_lora_linear_zero_init_parity():
    base = torch.nn.Linear(32, 16)
    x = torch.randn(4, 32)
    y0 = base(x).detach()
    lora = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)
    y1 = lora(x).detach()
    assert torch.allclose(y0, y1, atol=1e-6)
    assert not any(p.requires_grad for p in lora.base.parameters())
    assert lora.lora_a.requires_grad and lora.lora_b.requires_grad


def test_custody_preflight_clean_11():
    pf = verify_environment_and_custody("clean_11")
    assert pf["test_sealed_opened"] is False
    assert pf["counts"]["fit"] == 3681
    assert pf["counts"]["cal"] == 635
    assert pf["counts"]["eval_prompts"] == 2446
    assert pf["historical_eval_rows_with_seen_speaker"] == {"spontaneous": 85}
    with pytest.raises(ValueError, match="only clean_11"):
        verify_environment_and_custody("legacy_15")


def test_xattn_inference_respects_padded_slots_and_keeps_order():
    lengths = (22, 38, 222, 33, 222, 71, 70, 71, 71, 6)
    rows = [{"canonical_phones": ["AA"] * n} for n in lengths]
    bounds = list(xattn_inference_ranges(rows, batch_size=4))
    assert [j for a, b in bounds for j in range(a, b)] == list(range(len(rows)))
    assert (2, 3) in bounds and (4, 5) in bounds
    assert all((b - a) * (max(lengths[a:b]) + 1) <= 256 for a, b in bounds)


def test_probe_gate_does_not_advance_a_ranked_but_untested_or_unsafe_backbone():
    base = dict(cal_l2_target_ap=0.75, cal_l2_blind_spot_ap=0.20, cal_l2_gap_ap=0.2,
                cal_l1_fa_at_r70=0.004, n_cal_blind_positives=100, n_cal_gap_positives=50)
    probes = {
        "koel_only": base,
        "koel+bad": dict(base, cal_l2_target_ap=0.77, cal_l2_blind_spot_ap=0.23, cal_l1_fa_at_r70=0.010),
        "koel+tie": dict(base),
        "koel+small_n": dict(base, cal_l2_gap_ap=0.27, n_cal_gap_positives=3),
        "koel+valid": dict(base, cal_l2_gap_ap=0.25),
    }
    gates = probe_gate(probes, ["bad", "tie", "small_n", "valid"])
    assert {k for k, v in gates.items() if v["passed"]} == {"valid"}


def test_recall_matching_keeps_ties_and_handles_zero_recall():
    scores = np.array([0.6, 0.9, 0.6, 0.6, 0.1])
    match = closest_recall_policy(scores, 2)
    assert match == {"tau": 0.9, "cal_called": 1, "ref_cal_called": 2, "cal_positive": 5}
    zero = closest_recall_policy(scores, 0)
    assert int((scores >= zero["tau"]).sum()) == 0


def test_charsiu_path_reaches_last_phone_despite_high_first_phone_score():
    logits = np.tile(np.array([[4.0, -15.0, -20.0]], dtype=np.float32), (12, 1))
    start, end, _, duration = _viterbi_charsiu_spans(logits, ["D", "K"], {"D": 0, "K": 1, "[SIL]": 2})
    assert start[0] == 0 and end[-1] == 12
    assert (duration > 0).all() and end[0] <= start[1]


def test_warm_ctc_backward_with_repeated_phones():
    phone = torch.zeros(12, 40, requires_grad=True)
    af = torch.zeros(12, NUM_AF + 1, requires_grad=True)
    labels = torch.tensor([1, 1, 4])
    loss = warm_ctc_loss(phone, af, labels, torch.from_numpy(AF_BUNDLES), torch.from_numpy(PHONE_TO_BUNDLE))
    assert torch.isfinite(loss)
    loss.backward()
    for gradient in (phone.grad, af.grad):
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


def test_wavlm_adapter_attention_zero_init_matches_stock_and_has_gradient():
    from transformers.models.wavlm.modeling_wavlm import WavLMAttention

    torch.manual_seed(11)
    module = WavLMAttention(embed_dim=16, num_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 8, 16)
    bias = torch.randn(4, 8, 8) / 10
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
    ref, _ = module.torch_multi_head_self_attention(x, mask, bias)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        setattr(module, name, LoRALinear(getattr(module, name), r=2, alpha=4, dropout=0))
    new, _ = adapter_self_attention(module, x, mask, bias)
    torch.testing.assert_close(ref, new, atol=1e-6, rtol=1e-5)
    new.square().mean().backward()
    grads = [p.grad for n, p in module.named_parameters() if n.endswith("lora_b")]
    assert len(grads) == 4 and all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in grads)


def test_restarting_arm_preserves_partial_artifacts(tmp_path):
    path = tmp_path / "arm"
    path.mkdir()
    (path / "receipt").write_text("old")
    open_arm_dir(path)
    assert path.is_dir() and not list(path.iterdir())
    archive = list((tmp_path / "_partial_attempts").glob("*/receipt"))
    assert len(archive) == 1 and archive[0].read_text() == "old"


def test_historical_gold_minimal_schema_and_key_whitelist(tmp_path, monkeypatch):
    p = tmp_path / "eval_gold.jsonl"
    prompt = {"cohort": "spontaneous", "uid": "ok", "speaker": "ABA", "canonical_phones": ["AH"]}
    gold = {"cohort": "spontaneous", "uid": "ok", "gold_targets": ["S"], "gold_gaps": ["N", "N"]}
    save_jsonl(p, [dict(gold, uid="not-admitted"), gold])
    monkeypatch.setattr(common, "LEGACY_15_SPLIT", tmp_path)
    assert common.load_eval_gold_stage6_only(sha_file(p), [prompt]) == [{**prompt, **gold}]
    with pytest.raises(RuntimeError, match="Protected speaker"):
        common.load_eval_gold_stage6_only(sha_file(p), [dict(prompt, speaker="ZHAA")])


def test_verifies_seal_and_refuses_mutated_parquet_before_gold(tmp_path):
    row = {"cohort": "held6", "uid": "ok", "speaker": "ABA", "canonical_phones": ["AH"]}
    policy = {"tau_t": 0.5, "tau_g": 0.7}
    ck = tmp_path / "control.pt"
    ck.write_bytes(b"checkpoint")
    cal = tmp_path / "cal_arrays.npz"
    np.savez(cal, scores=[0.1])
    card = tmp_path / "CHECKPOINT_CARD.json"
    save_json(card, {"policy": policy, "checkpoint_sha256": sha_file(ck), "cal_arrays_sha256": sha_file(cal)})
    records = list(scores_to_rows(row, np.array([[0.3, 0.2, 0.5]]), np.array([[1, 0], [0.1, 0.9]]), policy))
    seal_predictions(tmp_path, records, sha_file(card))
    expected = pd.DataFrame.from_records(records).set_index(KEY).sort_index()[["speaker", "canonical_phone"]]
    summary = {"checkpoint_sha256": sha_file(ck), "seal_sha256": sha_file(tmp_path / "PREDICTIONS_SEAL.json")}
    assert _verify_sealed_arm(tmp_path, expected, summary)["policy"] == policy
    with (tmp_path / "eval_predictions/held6.parquet").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="Predictions changed"):
        _verify_sealed_arm(tmp_path, expected, summary)
