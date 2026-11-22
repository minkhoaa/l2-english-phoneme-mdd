"""Synthetic-only fixtures for Cal recounts, conditional permutations, and frame timing."""

import numpy as np
import pandas as pd
import pytest

from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.metrics import (
    closest,
    geometry,
    paired,
    view,
)
from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.replay import permutations
from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.timing.pool import cut


def test_recount_masks_unknown_and_native_before_scoring():
    t = pd.DataFrame(
        dict(
            l2=[1, 1, 1, 1, 0],
            y=["S", "O", "C", "unknown", "C"],
            blind=[0, 1, 1, 1, 1],
            spk=["a", "a", "b", "b", "n"],
            src=["A", "A", "B", "B", "N"],
            ref_kind=["S_alt", None, None, None, None],
        )
    ).astype({"l2": bool, "blind": bool})
    g = pd.DataFrame(
        dict(
            l2=[1, 1, 1, 0],
            y=["I", "N", "unknown", "N"],
            spk=["a", "b", "b", "n"],
            src=["A", "B", "B", "N"],
        )
    ).astype({"l2": bool})
    pred = (
        np.array([[0.8, 0.1], [0.7, 0.1], [0, 0.1], [0.8, 0], [0.8, 0]]),
        np.array([0.1, 0.8, 0.9, 0.7]),
    )
    policy = dict(tau_t=0.5, tau_g=0.5)
    m = view(t, g, pred, policy)
    assert m["tp_fp_fn"] == [1, 2, 2] and m["strict_f1"] == pytest.approx(1 / 3)
    assert m["o_to_s"] == 1 and m["s_to_o"] == 0 and m["per_class"]["I"] == [0, 1, 1]
    assert m["l1"] == dict(target_alert=1, targets=1, gap_alert=1, gaps=1)
    p = paired(t, g, pred, policy, pred, policy)
    assert p["descriptive_speaker_bootstrap_delta_pp_95"] == [0, 0]
    assert p["matched_positive_counts"] == {
        "target_ref_arm": [2, 2],
        "gap_ref_arm": [0, 0],
    }
    assert p["swaps"]["O"]["lost"] == p["swaps"]["O"]["rescued"] == 0


def test_closest_never_splits_ties_including_zero_calls():
    sc = np.array([0.5, 0.5, 0.5, 0.9], dtype=np.float32)
    for k in range(len(sc) + 1):
        threshold, count = closest(sc, k)
        assert int((sc.astype("float64") >= threshold).sum()) == count
        assert abs(count - k) == min(abs(i - k) for i in [0, 1, 4])


def test_shuffle_preserves_context_and_singletons_without_fixed_points_in_repeated_groups():
    meta = pd.DataFrame(
        {
            "src": ["A"] * 5 + ["B"],
            "spk": ["a"] * 3 + ["b"] * 2 + ["a"],
            "context": ["DH", "DH", "T", "DH", "DH", "DH"],
        }
    )
    for p in permutations(meta).values():
        assert np.array_equal(meta.to_numpy(), meta.iloc[p].to_numpy())
        assert np.array_equal(p != np.arange(len(p)), [1, 1, 0, 1, 1, 0])


def test_mixed_object_annotation_hits_are_counts_not_logical_or():
    t = pd.DataFrame(
        dict(
            l2=[True] * 4,
            y=["C", "C", "C", "S"],
            blind=[False] * 4,
            one_frame=[True] * 4,
            equal_hidden=[True] * 4,
            ref_kind=[None] * 3 + ["S_alt"],
            can_lp=[-1.0] * 4,
            voronoi_iou=[0.8] * 4,
            charsiu_iou=[0.9] * 4,
        )
    )
    hits = pd.Series(
        [np.bool_(True), np.bool_(True), np.bool_(False), np.nan], dtype=object
    )
    t["peak_inside"] = hits
    t["peak_inside20"] = hits
    g = pd.DataFrame(dict(insertion_centres=[1], inside=[0], inside20=[1]))
    c = geometry(t, g)["annotated_cal_arctic"]["C"]
    assert c["peak_inside"] == c["inside_20ms"] == 2 and c["n"] == 3


def test_clock_half_open_and_short_reference_interval():
    centres = np.array([0.0125, 0.0325, 0.0525])
    assert cut(centres, 0.0125, 0.0525) == (0, 2, False)
    assert cut(centres, 0.033, 0.034) == (1, 2, True)
    assert cut(centres, 0.07, 0.09) == (2, 3, True)
