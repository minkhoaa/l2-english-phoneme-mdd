"""Synthetic treatment guard: no normalized-MDD, roster, or old-value drift."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.supervision.fit_only import (
    apply_changes,
    protected_fields_sha,
    records_sha,
)
from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.supervision.slices import (
    matched_ranks,
)


def test_conditional_positive_rank_ties_and_matched_controls():
    frame = pd.DataFrame(
        {
            "phone": ["S", "S", "S", "S", "L", "S"],
            "spk": ["a", "a", "a", "b", "a", "b"],
            "kind": ["C", "S_same", "C", "S_alt", "C", "C"],
            "blind": [True, True, False, True, True, False],
        },
        index=[2, 3, 4, 5, 6, 7],
    )
    score = np.array([0.0, 0.0, 0.5, 0.5, 0.1, 0.7, 0.8, 0.6])
    ranked = matched_ranks(frame, score)
    assert ranked["all"]["S_same"]["mean_rank_fraction"] == 0.75
    assert ranked["blind"]["S_same"]["mean_rank_fraction"] == 0.5
    assert ranked["all"]["S_alt"]["mean_rank_fraction"] == 1.0
    assert (
        ranked["blind"]["S_alt"]["eligible"] == 0
        and ranked["blind"]["S_alt"]["positive_count"] == 1
    )


def fixture():
    row = {
        "uid": "toy",
        "source": "l2arctic_dev18",
        "audio_sha256": "abc",
        "speaker": "X",
        "canonical_phones": ["T", "S", "L"],
        "gold_targets": ["S", "C", "S"],
        "gold_gaps": ["N", "I", "N", "N"],
        "realized_ids": [39, 5, 3],
    }
    edit = {
        "uid": "toy",
        "slot_index": 0,
        "old": 39,
        "new": 2,
        "wav_sha256": "abc",
        "canonical": "T",
    }
    return [row], edit


def test_copy_on_write_multiple_slots_and_immutable_input():
    fit, edit = fixture()
    before = deepcopy(fit)
    got = apply_changes(
        fit, [edit, {**edit, "slot_index": 2, "old": 3, "new": 5, "canonical": "L"}]
    )
    assert fit == before
    assert got[0]["realized_ids"] == [2, 5, 5]
    assert records_sha(fit) != records_sha(got) and protected_fields_sha(
        fit
    ) == protected_fields_sha(got)


@pytest.mark.parametrize(
    "alter",
    [
        {"uid": "not-fit"},
        {"wav_sha256": "changed"},
        {"old": 7},
        {"new": 39},
        {"new": -100},
        {"canonical": "K"},
        {"slot_index": -1},
        {"slot_index": 1, "canonical": "S", "old": 5},
    ],
)
def test_reject_stale_and_wrong_class(alter):
    fit, edit = fixture()
    with pytest.raises(ValueError):
        apply_changes(fit, [{**edit, **alter}])


def test_duplicate_change_and_nonarctic_rows_are_rejected():
    fit, edit = fixture()
    with pytest.raises(ValueError):
        apply_changes(fit, [edit, edit])
    with pytest.raises(ValueError):
        apply_changes([{**fit[0], "source": "not_arctic"}], [edit])
