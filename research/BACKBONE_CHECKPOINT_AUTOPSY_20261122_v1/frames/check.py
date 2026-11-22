"""Paired CPU checkpoint and frame-attention diagnostics. Never load Eval predictions/gold."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..inputs import (
    BASE_KOEL_CACHE,
    RUN,
    WEIGHTS_DIR,
    grid,
    koel_voronoi_spans,
    read_json,
    roles_and_refs,
    sha_file,
    tables,
)
from ..metrics import calls, closest, paired, view
from ..replay import Replay, SEEDS, cat, unpack
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import pool_koel_dataset
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.models import (
    FrameXAttnConfig,
    FrameXAttnMDD,
    FRAME_WIN_TARGET,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.stages_0_3 import (
    _collate_xattn,
    xattn_inference_ranges,
)
from .attention import intervene, reconstruct

ARMS = {"B1": "B1_xattn_top1", "B2": "B2_xattn_dual_frame"}


def load(name):
    arm = ARMS[name]
    path = RUN / "stage3_xattn" / arm
    assert "stage3_" + arm in read_json(RUN / "PIPELINE_STATE.json")["completed"]
    card, sm, seal = [
        read_json(path / (f + ".json"))
        for f in ["CHECKPOINT_CARD", "SUMMARY", "PREDICTIONS_SEAL"]
    ]
    assert (
        sha_file(path / "model.pt")
        == card["checkpoint_sha256"]
        == sm["checkpoint_sha256"]
    )
    assert sm["seal_sha256"] == sha_file(path / "PREDICTIONS_SEAL.json") and seal[
        "checkpoint_card_sha256"
    ] == sha_file(path / "CHECKPOINT_CARD.json")
    assert sha_file(path / "cal_arrays.npz") == card["cal_arrays_sha256"]
    ck = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    assert (
        ck["seed"] == card["seed"] == sm["seed"] == SEEDS[0]
        and ck["arm"] == card["arm"] == sm["arm"] == arm
    )
    assert (
        ck["best_epoch"]
        == card["best_epoch"]
        == max(sm["history"], key=lambda x: x["sel"])["epoch"]
    )
    assert (
        ck["config"] == sm["config"] and ck["policy"] == card["policy"] == sm["policy"]
    )
    assert ck["policy"]["alpha_realized"] == 0 and ck["policy"]["temperature"] == {
        "default": [1.0, 1.0]
    }
    m = FrameXAttnMDD(FrameXAttnConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"], strict=True)
    m.eval()
    norm = {
        k: tuple(np.asarray(a, np.float32) for a in v) for k, v in ck["norm"].items()
    }
    return m, norm, ck["policy"], card, path


def reference_masks(rows, refs, mask):
    # Same 20 ms grid verified against both frame tensors on every ARCTIC Fit/Cal WAV in timing/pool.py.
    width, stride = grid(WEIGHTS_DIR / "config.json")
    n = mask.shape[-1]
    assert n == FRAME_WIN_TARGET
    inside = np.zeros(mask.shape, dtype=bool)
    for j, r in enumerate(rows):
        if r["uid"] not in refs:
            continue
        _, _, peaks, T = koel_voronoi_spans(BASE_KOEL_CACHE, r)
        ix = peaks[:, None] + np.arange(n) - n // 2
        l = len(peaks)
        assert np.array_equal((ix >= 0) & (ix < T), mask[j, :l])
        times = (ix * stride + (width - 1) / 2) / 16000
        z = refs[r["uid"]]["target_phones"]
        a, b = (np.array([slot[k] for slot in z])[:, None] for k in ["xmin", "xmax"])
        inside[j, :l] = (times >= a) & (times <= b) & mask[j, :l]
    return inside


def observe(branch, lengths, inside, store, errors):
    def hook(module, args, kwargs, out):
        bad = kwargs["key_padding_mask"]
        z, w = reconstruct(module, *args, bad)
        err = float((out[0] - z).abs().max())
        mean = w.mean(1)
        if out[1] is not None:
            err = max(err, float((out[1][:, 0] - mean).abs().max()))
        assert err < 2e-5, (branch, err)
        errors[branch] = max(errors[branch], err)
        W = w.shape[-1]
        b = len(lengths)
        valid = (~bad).sum(-1)
        denom = valid.clamp_min(2).float().log()

        def H(d):
            return -(d * d.clamp_min(1e-30).log()).sum(-1)

        values = {
            "valid": valid,
            "Havg_norm": H(mean) / denom,
            "Hheads_norm": H(w).mean(1) / denom,
            "effective_frames_avg": H(mean).exp(),
        }
        if inside is not None:
            ins = torch.from_numpy(inside).reshape(-1, W)
            values.update(
                mass_inside=(mean * ins).sum(-1),
                uniform_mass_inside=ins.sum(-1) / valid,
            )
        for key, v in values.items():
            arr = v.reshape(b, -1).numpy()
            store[branch][key].append(
                np.concatenate(
                    [
                        a[: l + int(branch == "gap_ext_xattn")]
                        for a, l in zip(arr, lengths, strict=True)
                    ]
                )
            )

    return hook


def describe(vals):
    a = np.asarray(vals)
    return (
        {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a))}
        if len(a)
        else {"n": 0}
    )


def attention_summary(store, t, g):
    out = {}
    for name, raw in store.items():
        a = {k: np.concatenate(xs) for k, xs in raw.items()}
        tab = g if name == "gap_ext_xattn" else t
        m = (tab.l2 & tab.y.ne("unknown")).to_numpy()
        assert len(m) == len(a["valid"])
        mult = m & (a["valid"] > 1)
        out[name] = {
            "valid_frame_count": describe(a["valid"][m]),
            "multi_frame": {
                k: describe(a[k][mult])
                for k in ["Havg_norm", "Hheads_norm", "effective_frames_avg"]
            },
        }
        if "mass_inside" in a:
            by = {}
            vals = a["mass_inside"]
            unif = a["uniform_mass_inside"]
            for group, ix in [
                ("C", t.y == "C"),
                ("S_alt", t.ref_kind == "S_alt"),
                ("S_same", t.ref_kind == "S_same"),
            ]:
                sl = (t.src.eq("l2arctic_dev18") & ix).to_numpy()
                by[group] = {
                    "learned": describe(vals[sl]),
                    "uniform": describe(unif[sl]),
                    "mean_paired_delta": float((vals[sl] - unif[sl]).mean()),
                }
            out[name]["inside_annotation"] = by
    return out


def movement(t, g, consensus, decisions):
    tr, gr = decisions
    t0, g0 = consensus
    out = {}
    for name, y, yes, hit, prev in [
        ("target", t.y, t.l2 & t.y.isin(["S", "O"]), tr != "C", t0),
        ("blind", t.y, t.l2 & t.blind & t.y.isin(["S", "O"]), tr != "C", t0),
        ("gap", g.y, g.l2 & g.y.eq("I"), gr == "I", g0),
    ]:
        # `prev` records binary alerts, not whether S vs O was correct.
        miss = yes.to_numpy() & ~prev.any(0)
        detected = yes.to_numpy() & prev.all(0)
        typed = (gr if name == "gap" else tr) == y.to_numpy()
        out[name] = dict(
            S0_all_missed=int(miss.sum()),
            rescued_binary=int((miss & hit).sum()),
            rescued_typed=int((miss & typed).sum()),
            S0_all_detected=int(detected.sum()),
            new_binary_misses=int((detected & ~hit).sum()),
        )
    return out


@torch.inference_mode()
def run(name):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise RuntimeError("dedicated CPU replay only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, norm, pol, card, path = load(name)
    roles, refs, refsha = roles_and_refs()
    rows = roles["cal"]
    xf = pool_koel_dataset(BASE_KOEL_CACHE, rows)
    t, g = tables(rows, xf, refs)
    s0 = []
    for seed in SEEDS:
        r = Replay("S0_" + str(seed))
        pred, _ = r.original(rows, xf)
        r.parity(t, g, pred)
        s0.append((pred, r.policy, calls(pred, r.policy)))
    con = (
        np.stack([c[2] != "C" for _, _, c in s0]),
        np.stack([c[3] == "I" for _, _, c in s0]),
    )
    opts = {
        "target_uniform": ("ext_t_xattn", "uniform"),
        "target_centre": ("ext_t_xattn", "centre"),
        "gap_uniform": ("gap_ext_xattn", "uniform"),
    }
    if model.cfg.use_koel_frame_win:
        opts["koel_target_uniform"] = ("koel_fw_xattn", "uniform")
    store = defaultdict(lambda: defaultdict(list))
    err = defaultdict(float)
    parts = defaultdict(list)
    for i, end in xattn_inference_ranges(rows, 4):
        rs = rows[i:end]
        lens = [len(r["canonical_phones"]) for r in rs]
        inp, _ = _collate_xattn(
            rs,
            xf[i:end],
            norm,
            torch.device("cpu"),
            frame_dir=RUN / "caches/frames_wavlm_base_plus",
            include_koel_mid=model.cfg.use_koel_frame_win,
            labels=False,
        )
        ins = reference_masks(rs, refs, inp["ext_t_mask"].numpy())
        if model.cfg.use_koel_frame_win:
            assert torch.equal(inp["koel_t_mask"], inp["ext_t_mask"])
        with ExitStack() as ctx:
            for k in sorted(set(key for key, mode in opts.values())):
                ctx.enter_context(
                    model.get_submodule(k).register_forward_hook(
                        observe(
                            k, lens, None if k == "gap_ext_xattn" else ins, store, err
                        ),
                        with_kwargs=True,
                    )
                )
            parts["original"].append(unpack(model(**inp), lens))
        for mode, (k, h) in opts.items():
            with model.get_submodule(k).register_forward_hook(
                intervene(h), with_kwargs=True
            ):
                parts[mode].append(unpack(model(**inp), lens))
    preds = {k: cat(v) for k, v in parts.items()}
    base = preds["original"]
    parity = {}
    with np.load(path / "cal_arrays.npz", allow_pickle=False) as z:
        for k, tab, score in [("t", t, base[0].sum(1)), ("g", g, base[1])]:
            for domain, sl in [("l2", tab.l2), ("l1", ~tab.l2)]:
                field = f"{domain}_{k}_sc"
                v = z[field]
                assert len(v) == sl.sum()
                if domain == "l2":
                    assert np.array_equal(z[f"l2_{k}_g"], tab.y[sl])
                parity[field] = float(abs(score[sl] - v).max())
                assert parity[field] < 5e-5
    assert np.array_equal(preds["gap_uniform"][0], base[0])
    original = view(t, g, base, pol)
    assert (
        round(original["strict_f1"] * 100, 2) == card["cal_metrics"]["cal_l2_strict_f1"]
    )
    assert abs(original["blindAP"] - card["cal_metrics"]["cal_blind_spot_ap"]) < 1e-4
    ts, gs, tc, gc = calls(base, pol)
    refpred, refpol, refcalls = s0[0]
    mt = (t.l2 & t.y.isin(["S", "O"])).to_numpy()
    mg = (g.l2 & g.y.eq("I")).to_numpy()
    ut, _ = closest(ts[mt], int((refcalls[2][mt] != "C").sum()))
    ug, _ = closest(gs[mg], int((refcalls[3][mg] == "I").sum()))
    mc = calls(base, dict(tau_t=ut, tau_g=ug))
    vs = paired(t, g, refpred, refpol, base, pol)
    return dict(
        scope="Cal only. Fixed seed; no new training or operational policy.",
        run_binding_sha256=sha_file(RUN / "RUN_BINDINGS.json"),
        source_ref_sha256=refsha,
        code_sha256={
            p.name: sha_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        torch=str(torch.__version__),
        card=card,
        cal_parity_max_abs=parity,
        mha_parity_max_abs=err,
        original=original,
        vs_S0_same_seed=vs,
        common_misses={
            "own_policy": movement(t, g, con, (tc, gc)),
            "matched_S0_recall": movement(t, g, con, mc[2:]),
        },
        attention=attention_summary(store, t, g),
        interventions={
            k: view(t, g, p, pol, detailed=False)
            for k, p in preds.items()
            if k != "original"
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dest = args.out.resolve()
    if dest.is_relative_to(RUN.parents[1]) or dest.exists():
        raise FileExistsError("use a fresh path outside campaign")
    payload = json.dumps(run(args.arm), allow_nan=False, indent=2) + "\n"
    with dest.open("x") as f:
        f.write(payload)
    print(dest, flush=True)
