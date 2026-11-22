"""Reference oracles on synthetic tensors; no data, certificates, GPU or sealed labels required."""

import numpy as np
import pytest
import torch

from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.frames.attention import (
    intervene,
    reconstruct,
)
from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.operating_point.frontier import (
    ceiling,
)


@pytest.mark.parametrize("key_weights", [True, False])
def test_masked_multhead_parity_and_content_preserving_interventions(key_weights):
    # Actual PyTorch MHA with distinct K/V and masks; distributions applied before joining heads.
    torch.manual_seed(17)
    m = torch.nn.MultiheadAttention(12, 3, dropout=0.3, batch_first=True).eval()
    q = torch.randn(4, 1, 12)
    k = torch.randn(4, 5, 12)
    v = torch.randn_like(k)
    mask = torch.tensor(
        [[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 0, 1, 1]],
        dtype=torch.bool,
    )

    def native(qq, mm):
        return m(qq, k, v, key_padding_mask=mm, need_weights=key_weights)

    with torch.no_grad():
        a, w = reconstruct(m, q, k, v, mask)
        b, bw = native(q, mask)
        torch.testing.assert_close(a, b)
        assert w.shape == (4, 3, 5)
        assert (w * mask[:, None]).count_nonzero() == 0
        if key_weights:
            torch.testing.assert_close(w.mean(1, keepdim=True), bw)
        # q_eff=0 gives equal logits for every valid key, regardless of k.
        assert torch.count_nonzero(m.in_proj_bias[:12]) == 0
        uniform, _ = native(torch.zeros_like(q), mask)
        centre_mask = torch.ones_like(mask)
        centre_mask[:, 2] = False
        center, _ = native(q, centre_mask)
        for mode, want in [("uniform", uniform), ("centre", center)]:
            with m.register_forward_hook(intervene(mode), with_kwargs=True):
                obs, _ = native(q, mask)
            torch.testing.assert_close(obs, want)
        torch.testing.assert_close(native(q, mask)[0], b)


def test_ratio_frontier_matches_direct_exhaustive_tie_preserving_search():
    rng = np.random.default_rng(48)
    for trial in range(80):
        nt = int(rng.integers(2, 18))
        ng = int(rng.integers(0, 13))
        t = rng.integers(0, 5, nt) / 4
        g = rng.integers(0, 4, ng) / 3
        yt = rng.integers(0, 3, nt)
        yt[0] = 2
        pred_type = rng.integers(1, 3, nt)
        yg = rng.random(ng) > 0.7
        pos = int((yt != 0).sum() + yg.sum())
        good = pred_type == yt
        best = 0.0
        for tt in np.r_[np.inf, np.unique(t)]:
            for tg in np.r_[np.inf, np.unique(g)]:
                pt, pg = t >= tt, g >= tg
                tp = (pt & good).sum() + (pg & yg).sum()
                best = max(best, 2 * tp / (pos + pt.sum() + pg.sum()))
        ans = ceiling(t, good, g, yg, pos)
        assert ans["strict_f1"] == pytest.approx(best)
        tp, fp, fn = ans["tp_fp_fn"]
        assert tp + fn == pos and best == pytest.approx(2 * tp / (2 * tp + fp + fn))
