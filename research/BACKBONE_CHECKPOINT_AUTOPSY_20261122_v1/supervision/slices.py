"""Descriptive Cal-only rank of each S among correct tokens of the same speaker and canonical phone.

Use all eligible C as controls, weight each eligible S once, ties 1/2. 'blind' restricts BOTH the
positive and controls. Reused controls / few speakers: not independent pairs or a test of causality.
Also partition the three-S0 common binary target misses; never use these strata as training labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..inputs import (
    BASE_KOEL_CACHE,
    RUN,
    normalize_canonical_phone,
    read_json,
    roles_and_refs,
    sha_file,
)
from ..replay import ARMS
from ..frames.check import ARMS as B_ARMS
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import pool_koel_iso
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.stages_0_3 import CAN_LP_IDX


def matched_ranks(frame, score):
    out = {}
    for cond, subset in [("all", frame), ("blind", frame[frame.blind])]:
        sub = {}
        for key in ["S_same", "S_alt"]:
            parts = []
            by = {}
            for (spk, ph), grp in subset.groupby(["spk", "phone"], sort=True):
                neg = np.sort(score[grp.index[grp.kind == "C"]])
                pos = score[grp.index[grp.kind == key]]
                if not len(neg) or not len(pos):
                    continue
                v = (
                    np.searchsorted(neg, pos, side="left")
                    + np.searchsorted(neg, pos, side="right")
                ) / (2 * len(neg))
                parts.append(v)
                by.setdefault(spk, []).append(v)
            p = np.concatenate(parts) if parts else np.array([])
            sub[key] = {
                "eligible": len(p),
                "positive_count": int((subset.kind == key).sum()),
                "mean_rank_fraction": float(p.mean()) if len(p) else None,
                "by_speaker": {
                    k: float(np.concatenate(v).mean()) for k, v in by.items()
                },
            }
        out[cond] = sub
    return out


def run():
    roles, refs, h = roles_and_refs()
    records = []
    for r in roles["cal"]:
        if r["is_native_l1"]:
            continue
        assert not r["source"].startswith("timit")
        raw = refs.get(r["uid"])
        x = pool_koel_iso(BASE_KOEL_CACHE, r)["target_scalars"][:, CAN_LP_IDX]
        for i, (ph, y) in enumerate(
            zip(r["canonical_phones"], r["gold_targets"], strict=True)
        ):
            kind = y
            if y == "S" and raw:
                kind = (
                    "S_same"
                    if normalize_canonical_phone(
                        raw["target_phones"][i]["realized_phone"]
                    )
                    == ph
                    else "S_alt"
                )
            records.append(
                {
                    "y": y,
                    "spk": r["speaker"],
                    "source": r["source"],
                    "phone": ph,
                    "kind": kind,
                    "blind": bool(x[i] > -0.2),
                }
            )
    df = pd.DataFrame(records)
    sub = df[df.source == "l2arctic_dev18"]
    dirs = {
        **ARMS,
        **{name: RUN / "stage3_xattn" / arm for name, arm in B_ARMS.items()},
    }
    yes = {}
    sha = {}
    ranks = {}
    completed = read_json(RUN / "PIPELINE_STATE.json")["completed"]
    for name, arm in dirs.items():
        if name in B_ARMS:
            assert "stage3_" + B_ARMS[name] in completed
        card = read_json(arm / "CHECKPOINT_CARD.json")
        sha[name] = sha_file(arm / "cal_arrays.npz")
        assert sha[name] == card["cal_arrays_sha256"]
        with np.load(arm / "cal_arrays.npz", allow_pickle=False) as z:
            assert np.array_equal(df.y, z["l2_t_g"])
            sc = z["l2_t_sc"]
            assert np.isfinite(sc).all()
        yes[name] = sc >= card["policy"]["tau_t"]
        ranks[name] = matched_ranks(sub, sc)
    missed = df.y.isin(["S", "O"]).to_numpy() & ~np.stack(
        [v for k, v in yes.items() if k.startswith("S0_")]
    ).any(0)
    table = []
    for (src, kind), grp in df[missed].groupby(["source", "kind"]):
        ix = grp.index.to_numpy()
        table.append(
            {
                "source": src,
                "kind": kind,
                "n": len(grp),
                "blind": int(grp.blind.sum()),
                "rescued_by": {
                    k: int(yes[k][ix].sum()) for k in ["A1", "A3", "B1", "B2"]
                },
            }
        )
    assert sum(a["n"] for a in table) == 526 and sum(a["blind"] for a in table) == 397
    return {
        "scope": __doc__,
        "run_binding": sha_file(RUN / "RUN_BINDINGS.json"),
        "reference_digest": h,
        "code_sha": sha_file(__file__),
        "cal_arrays_sha": sha,
        "shared_misses": table,
        "conditional_positive_rank": ranks,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    dest = p.parse_args().out.resolve()
    if dest.exists() or dest.is_relative_to(RUN.parents[1]):
        raise FileExistsError("use a fresh path outside campaign")
    result = json.dumps(run(), indent=2, allow_nan=False) + "\n"
    with dest.open("x") as f:
        f.write(result)
    print(dest, flush=True)
