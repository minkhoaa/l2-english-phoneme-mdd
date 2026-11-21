from pathlib import Path

import numpy as np
import pytest
import torch

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    SeqMDDv2,
    extract_seq_mdd_v2_features,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import (
    Spec,
    WindowHead,
    pool_features,
    voronoi_bounds,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import (
    KEY,
    gold_frame,
    keyed_merge,
    read_predictions,
    seal_predictions,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import collate, objective


def inputs(n=6):
    g = np.random.default_rng(17)
    h = g.normal(size=(24, 6, 12)).astype(np.float16)
    lp = (
        torch.as_tensor(g.normal(size=(24, 79)), dtype=torch.float32)
        .log_softmax(-1)
        .numpy()
    )
    can = ["AA", "AA", "T", "OW", "N", "L"][:n]
    return h, lp, can


class FakeKoel:
    def __init__(self, h, lp):
        from types import SimpleNamespace

        hs = [None] * 25
        for i, j in enumerate([12, 15, 18, 20, 22, 24]):
            hs[j] = torch.zeros((1, len(lp), 1024))
            hs[j][0, :, : h.shape[-1]] = torch.from_numpy(h[:, i].astype(np.float32))
        self.output = SimpleNamespace(
            hidden_states=hs, logits=torch.as_tensor(lp)[None, ...]
        )

    def __call__(self, x, output_hidden_states):
        return self.output


def test_v2_pooling_parity():
    h, lp, can = inputs()
    model = FakeKoel(h, lp)
    out = extract_seq_mdd_v2_features(
        model, torch.ones((1, 800)), can, torch.device("cpu")
    )
    lp = model.output.logits[0].log_softmax(-1).numpy()
    ref = np.pad(h, ((0, 0), (0, 0), (0, 1024 - h.shape[-1])))
    pooled = pool_features(ref, lp, can)
    for k, a in out.items():
        np.testing.assert_allclose(pooled[k], a, atol=1e-6, rtol=0, err_msg=k)


@pytest.mark.parametrize(
    "spec",
    [
        Spec(),
        Spec(window="mean"),
        Spec(window="full"),
        Spec(window="full", gap_core=True, focal=True),
    ],
)
def test_forward_gradient_and_singleton(spec):
    h, lp, can = inputs()
    rows = []
    feat = []
    for n in (1, len(can)):
        f = pool_features(h, lp, can[:n], spec)
        rows.append(
            {
                "canonical_phones": can[:n],
                "gold_targets": ["C"] * n,
                "gold_gaps": ["N"] * (n + 1),
                "realized_ids": f["phone_ids"],
            }
        )
        feat.append(f)
    head = WindowHead(spec, h_dim=h.shape[-1], d_model=24)
    x, labels = collate(rows, feat, {}, torch.device("cpu"), labels=True)
    y = head(**x)
    assert y["target_logits"].shape == (2, 6, 3)
    assert y["gap_logits"].shape == (2, 7, 2)
    loss = objective(y, labels, spec)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )
    assert head.target_head[-1].weight.grad.norm() > 0


def test_v2_forward_numerical_parity():
    h, lp, can = inputs()
    b, _ = collate(
        [{"canonical_phones": can}],
        [pool_features(h, lp, can)],
        {},
        torch.device("cpu"),
        labels=False,
    )
    old = SeqMDDv2(h_dim=12, d_model=24).eval()
    new = WindowHead(h_dim=12, d_model=24).eval()
    new.load_state_dict(old.state_dict())
    for k, y in old(**b).items():
        torch.testing.assert_close(new(**b)[k], y)


def test_voronoi_includes_peaks_even_with_uniform_fallback():
    assert voronoi_bounds([0, 4, 8], 10) == [(0, 3), (3, 7), (7, 10)]
    assert all(
        a <= p < b for (a, b), p in zip(voronoi_bounds([0, 0, 1, 1], 2), [0, 0, 1, 1])
    )


def test_seal_validation(tmp_path: Path):
    row = {
        "cohort": "held6",
        "uid": "x",
        "speaker": "A",
        "domain": "target",
        "index": 0,
        "canonical_phone": "AA",
        "score": 0.8,
        "call": "O",
    }
    seal = seal_predictions(tmp_path, [row], "a")
    pred = read_predictions(tmp_path)
    frame = gold_frame(
        [{"cohort": "held6", "uid": "x", "gold_targets": ["O"], "gold_gaps": []}]
    )
    assert keyed_merge(pred, frame).loc[0, "gold_label"] == "O"
    altered = frame.copy()
    altered.loc[0, "index"] = 2
    with pytest.raises(ValueError, match="Key mismatch"):
        keyed_merge(pred, altered)
    assert seal["key"] == KEY
    predfile = tmp_path / seal["directory"] / "held6.parquet"
    with predfile.open("ab") as f:
        f.write(b"altered")
    with pytest.raises(ValueError, match="Predictions changed"):
        read_predictions(tmp_path)
