"""Exact two-threshold Cal ceiling for the saved scores and fixed target error types."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..frames.check import ARMS as FRAME_ARMS, load as load_frame
from ..inputs import BASE_KOEL_CACHE, RUN, read_json, roles_and_refs, sha_file, tables
from ..metrics import counts, f1
from ..replay import ARMS, Replay, cat, unpack
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import pool_koel_dataset
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.stages_0_3 import (
    _collate_xattn,
    xattn_inference_ranges,
)


def prefix(scores, correct):
    """Number of alerts and typed true positives at every attainable threshold."""
    assert len(correct) == len(scores) and np.isfinite(scores).all()
    order = np.argsort(-scores, kind="stable")
    ss = scores[order]
    ends = (
        np.flatnonzero(np.r_[ss[1:] != ss[:-1], True]) + 1
        if len(ss)
        else np.array([], dtype=int)
    )
    return np.r_[0, ends], np.r_[0, correct[order].cumsum(dtype=np.int64)[ends - 1]]


def ceiling(t_scores, t_correct, g_scores, g_correct, positives):
    """For a ratio n/d, max(2*TP*d - n*(P+alerts)) separates into two prefix maxima.

    Strict improvement until the integer residual is zero, hence an exact optimum on a finite set.
    """
    ts, tt = prefix(t_scores, t_correct)
    gs, gt = prefix(g_scores, g_correct)
    num, den = 0, 1
    if positives < 1:
        raise ValueError("positive gold required")
    for _ in range(len(ts) + len(gs) + 2):
        t = int(np.argmax(2 * tt * den - num * ts))
        g = int(np.argmax(2 * gt * den - num * gs))
        newnum = 2 * int(tt[t] + gt[g])
        newden = positives + int(ts[t] + gs[g])
        residual = newnum * den - num * newden
        if residual == 0:
            tp = newnum // 2
            return dict(
                strict_f1=newnum / newden,
                tp_fp_fn=[tp, int(ts[t] + gs[g]) - tp, positives - tp],
            )
        assert residual > 0
        num, den = newnum, newden
    raise RuntimeError("prefix optimum did not converge")


@torch.inference_mode()
def run(names):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise RuntimeError("CPU only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    roles, refs, _ = roles_and_refs()
    rows = roles["cal"]
    xf = pool_koel_dataset(BASE_KOEL_CACHE, rows)
    t, g = tables(rows, xf, refs)
    mt = (t.l2 & t.y.isin(["C", "S", "O"])).to_numpy()
    mg = (g.l2 & g.y.isin(["N", "I"])).to_numpy()
    yt = t.y[mt].to_numpy()
    yg = g.y[mg].to_numpy()
    pos = int(np.isin(yt, ["S", "O"]).sum() + (yg == "I").sum())
    results = {}
    for name in names:
        print("CPU threshold ceiling", name, flush=True)
        if name in ARMS:
            a = Replay(name)
            pred, _ = a.original(rows, xf)
            a.parity(t, g, pred)
            pol = a.policy
            card = read_json(ARMS[name] / "CHECKPOINT_CARD.json")
            path = ARMS[name]
        else:
            m, norm, pol, card, path = load_frame(name)
            parts = []
            for start, end in xattn_inference_ranges(rows, 4):
                rs = rows[start:end]
                inp, _ = _collate_xattn(
                    rs,
                    xf[start:end],
                    norm,
                    torch.device("cpu"),
                    frame_dir=RUN / "caches/frames_wavlm_base_plus",
                    include_koel_mid=m.cfg.use_koel_frame_win,
                    labels=False,
                )
                parts.append(unpack(m(**inp), [len(r["canonical_phones"]) for r in rs]))
            pred = cat(parts)
        # Use the original, SHA-bound GPU scalar scores at threshold breakpoints. CPU supplies the S/O type only.
        # Check the locked typed recount; report a second optimistic score on near-tied positive targets.
        with np.load(path / "cal_arrays.npz", allow_pickle=False) as z:
            assert np.array_equal(t.y[t.l2], z["l2_t_g"]) and np.array_equal(
                g.y[g.l2], z["l2_g_g"]
            )
            ts = z["l2_t_sc"][mt[t.l2]]
            gs = z["l2_g_sc"][mg[g.l2]]
        pp = pred[0][mt]
        assert (
            np.max(abs(ts - pp.sum(-1))) < 5e-5 and np.max(abs(gs - pred[1][mg])) < 5e-5
        )
        typed = np.where(pp[:, 1] > pp[:, 0], "O", "S")
        delta = np.abs(pp[:, 1] - pp[:, 0])
        candidates = np.isin(yt, ["S", "O"])  # only types on positive targets affect F1
        # Explicit bound on F1 if every numerically close positive were mis-typed.
        close = candidates & (delta < 2e-6)
        right = typed == yt
        locked = counts(
            yt, np.where(ts >= pol["tau_t"], typed, "C"), ["S", "O"]
        ) + counts(yg, np.where(gs >= pol["tau_g"], "I", "N"), ["I"])
        assert round(100 * f1(locked), 2) == card["cal_metrics"]["cal_l2_strict_f1"]
        best = ceiling(ts, right, gs, yg == "I", pos)
        upper = ceiling(ts, right | close, gs, yg == "I", pos)
        assert best["strict_f1"] >= f1(locked) - 1e-12
        results[name] = dict(
            ckpt_sha=card["checkpoint_sha256"],
            cal_arrays_sha=card["cal_arrays_sha256"],
            locked_f1=f1(locked),
            best_on_Cal=best,
            possible_tie_positive_count=int(close.sum()),
            optimistic_if_close_type_flips=upper["strict_f1"],
            headroom_pp=100 * (best["strict_f1"] - f1(locked)),
        )
    return dict(
        scope="optimistic same-Cal two-scalar-threshold envelope only; no output policy",
        binding_sha=sha_file(RUN / "RUN_BINDINGS.json"),
        code_sha=sha_file(__file__),
        cal_counts={"targets": int(mt.sum()), "gaps": int(mg.sum()), "positives": pos},
        results=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", required=True, choices=[*ARMS, *FRAME_ARMS])
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    dest = args.out.resolve()
    if dest.is_relative_to(RUN.parents[1]) or dest.exists():
        raise FileExistsError("use a fresh path outside campaign")
    out = json.dumps(run(args.arms), indent=2, allow_nan=False) + "\n"
    with dest.open("x") as f:
        f.write(out)
    print(dest, flush=True)
