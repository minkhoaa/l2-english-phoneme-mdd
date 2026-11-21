from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import sha_file
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import Spec
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.prepare_l1_canonical_anchor import (
    cmu_phones,
    corrupt_phones,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_exp5_joint_lora import (
    add_lora,
    differentiable_pool,
    from_wav_prefix,
    top_forward,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.stages import dev_oof_inputs, guard_split
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import (
    ARCHIVED_POLICY,
    Logits,
    probabilities,
    sample_probs,
)
from research.KOEL_V2_FILTER_BENCHMARK_20261119_v1.run_v2_filter_benchmark import FOLDS


def test_oof_twelve_three_three():
    data = [
        {"speaker": sp, "uid": sp, "audio_sha256": str(i)}
        for i, sp in enumerate(sum(FOLDS, []))
    ]
    u = []
    for f in range(6):
        a, b, c = dev_oof_inputs(data, f)
        guard_split(a, b, c)
        assert list(map(len, (a, b, c))) == [12, 3, 3]
        u.extend(r["uid"] for r in c)
    assert sorted(u) == sorted(r["uid"] for r in data)
    with pytest.raises(ValueError, match="Unexpected split exposure"):
        guard_split(a, b, b)


def test_in_vocabulary_and_prompt_edits():
    lex = {"say": ["s", "ey"], "a": ["ah"], "dark": ["d", "aa", "r", "k"]}
    p, oov = cmu_phones("Say a dark.", lex)
    assert not oov and p == ["S", "EY", "AH", "D", "AA", "R", "K"]
    assert cmu_phones("Say two now", lex)[1] == ["now", "two"]
    can, t, g, real = corrupt_phones(p, 21, "test")
    assert len(can) == len(t) == len(real) and len(g) == len(can) + 1
    assert any(y != "C" for y in t) or "I" in g
    assert (can, t, g, real) == corrupt_phones(p, 21, "test")


def test_corpus_weights_and_fusion():
    sources = (
        ["l2arctic_dev18"] * 9 + ["epadb_train"] * 2 + ["timit_train_l1_canon"] * 5
    )
    w = sample_probs([{"source": s} for s in sources])
    np.testing.assert_allclose(
        [w[:9].sum(), w[9:11].sum(), w[11:].sum()], [0.45, 0.30, 0.25]
    )
    a = Logits(
        np.array([[1.0, 2.0, 3.0]], np.float32),
        np.array([[0.1, 0.2], [0.3, 0.4]], np.float32),
        np.arange(40, dtype=np.float32)[None, :],
        np.array([2]),
    )
    p, g = probabilities(
        a, {"source": "epadb_train"}, dict(ARCHIVED_POLICY, alpha_realized=0.5)
    )
    assert p.shape == (1, 3) and g.shape == (2, 2) and np.isclose(p.sum(), 1)


@pytest.mark.parametrize(
    "spec", [Spec(), replace(Spec(), window="full", gap_core=True)]
)
def test_tiny_split_autograd_matches_full_eval_and_has_gradients(tmp_path: Path, spec):
    torch.manual_seed(7)
    c = Wav2Vec2Config(
        hidden_size=16,
        num_hidden_layers=24,
        num_attention_heads=4,
        intermediate_size=24,
        conv_dim=(8, 8, 8),
        conv_stride=(2, 2, 2),
        conv_kernel=(4, 4, 4),
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=True,
        vocab_size=79,
        apply_spec_augment=False,
    )
    c._attn_implementation = "eager"
    model = Wav2Vec2ForCTC(c).eval().requires_grad_(False)
    adapters = add_lora(model)
    assert len(adapters) == 32
    wav = tmp_path / "test.wav"
    sf.write(
        wav, np.random.default_rng(8).normal(0, 0.1, 480).astype(np.float32), 16000
    )
    row = {"audio_path": str(wav), "uid": "test", "audio_sha256": sha_file(wav)}
    from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import forward_frames

    device = torch.device("cpu")
    fixed, lp_ref = forward_frames(model, row, device)
    x, low = from_wav_prefix(model, row, device)
    assert not x.requires_grad
    h, lp = top_forward(model, x, low, checkpointed=True)
    np.testing.assert_array_equal(h.detach().numpy(), fixed.astype(np.float32))
    np.testing.assert_allclose(lp.detach().numpy(), lp_ref, atol=1e-6)
    static, ac = differentiable_pool(h, lp, ["AA", "D", "T", "L"], spec, None)
    for k, v in ac.items():
        np.testing.assert_allclose(v.detach().numpy(), static[k], atol=0.002)
    (
        ac["h_vit"].square().mean() + ac["h_gap"][0].square().sum() + lp[:, 2].mean()
    ).backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) for p in adapters.values()
    )
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
