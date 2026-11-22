"""CPU unit tests for deploy/koel_ens3_mdd (KoelEns3MDDEngine)."""

from __future__ import annotations

import numpy as np
import pytest

from deploy.koel_ens3_mdd.engine import DEFAULT_BUNDLE_PATH, KoelEns3MDDEngine


def test_engine_score_frames_contract_on_cpu() -> None:
    if not DEFAULT_BUNDLE_PATH.exists():
        pytest.skip("Bundle not yet built")
    eng = KoelEns3MDDEngine(DEFAULT_BUNDLE_PATH, device="cpu", load_koel=False)
    assert len(eng.heads) == 3
    rng = np.random.default_rng(7)
    T = 40
    h = rng.normal(0, 0.5, size=(T, 6, 1024)).astype(np.float16)
    lp = np.log(np.full((T, 79), 1.0 / 79.0, dtype=np.float32))
    phones = ["HH", "AH", "L", "OW"]

    res = eng.score_frames(h, lp, phones)
    assert len(res["targets"]) == 4
    assert len(res["gaps"]) == 5
    assert res["summary"]["n_phones"] == 4
    for t in res["targets"]:
        assert t["call"] in ("C", "S", "O")
        assert 0.0 <= t["error_prob"] <= 1.0
    for g in res["gaps"]:
        assert g["call"] in ("N", "I")
        assert 0.0 <= g["insertion_prob"] <= 1.0
