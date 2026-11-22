"""Run from repo root: python -m research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.audit --out NEW.json"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch
from .inputs import (
    roles_and_refs,
    profile,
    tables,
    RUN,
    read_json,
    sha_file,
    BASE_KOEL_CACHE,
)
from .replay import ARMS, Replay, permutations, perm_stats
from .metrics import geometry, paired, view, calls
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import pool_koel_dataset


def byhash(rows):
    return sorted(
        rows, key=lambda r: hashlib.sha256(("autopsy-v1:" + r["uid"]).encode()).digest()
    )


def audit():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise RuntimeError(
            "Run with CUDA_VISIBLE_DEVICES='' (CPU-only audit, never take the GPU)"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    roles, refs, ref_digest = roles_and_refs()
    fit, cal = roles["fit"], roles["cal"]
    pick = [
        r
        for src in sorted({r["source"] for r in fit})
        for r in byhash([r for r in fit if r["source"] == src])[:96]
    ]
    xcal = pool_koel_dataset(BASE_KOEL_CACHE, cal)
    xfit = pool_koel_dataset(BASE_KOEL_CACHE, pick)
    t, g = tables(cal, xcal, refs, timing=True)
    ft, fg = tables(pick, xfit, refs)
    plans = {"t": permutations(t), "g": permutations(g)}
    p = read_json(RUN / "stage1_probe/STAGE1_PROBE.json")
    bind = read_json(RUN / "RUN_BINDINGS.json")
    same_prompt = {}
    for src in ["l2arctic_dev18", "epadb_train"]:
        known = {tuple(r["canonical_phones"]) for r in fit if r["source"] == src}
        sub = [r for r in cal if r["source"] == src]
        same_prompt[src] = dict(
            cal_rows=len(sub),
            same_canonical_sequence_in_fit=sum(
                tuple(r["canonical_phones"]) in known for r in sub
            ),
        )
    selected = [
        i
        for src in sorted({r["source"] for r in pick})
        for i in [j for j, r in enumerate(pick) if r["source"] == src][:3]
    ]
    out = dict(
        scope="Post hoc CPU Fit/Cal audit only; fixed S0/A1/A3. No optimizer steps or Eval artifacts read beyond seal metadata.",
        utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        device="cpu",
        binding_sha256=sha_file(RUN / "RUN_BINDINGS.json"),
        run_git=bind["preflight"]["git_head"],
        source_ref_selected_digest=ref_digest,
        source_ref_counts=dict(
            Counter(r["source"] for r in fit + cal if r["uid"] in refs)
        ),
        fit_sample_uids_sha256=hashlib.sha256(
            "\n".join(r["uid"] for r in pick).encode()
        ).hexdigest(),
        audit_code_sha256={
            p.name: sha_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        fit_cal_same_prompt=same_prompt,
        profile={s: profile(rs, refs) for s, rs in roles.items()},
        geometry=geometry(t, g),
        permutation_coverage={
            "target": perm_stats(t, plans["t"]),
            "gap": perm_stats(g, plans["g"]),
        },
        stage1_probes=p["probes"],
        arms={},
        paired={},
    )
    preds = {}
    policies = {}
    orders = {}
    for name in ARMS:
        print("CPU replay", name, flush=True)
        a = Replay(name)
        policies[name] = a.policy
        orders[name] = a.order
        do_perm = name in ("A1", "A3", "S0_20261121")
        pred, bank = a.original(cal, xcal, capture=do_perm)
        pred_fit, _ = a.original(pick, xfit)
        pred_fit_metrics = view(ft, fg, pred_fit, a.policy)
        preds[name] = pred
        m = view(t, g, pred, a.policy)
        sm = a.summary
        h = sm["history"]
        assert max(h, key=lambda e: e["sel"])["epoch"] == sm["best_epoch"]
        assert (
            round(m["strict_f1"] * 100, 2) == sm["cal_metrics"]["cal_l2_strict_f1"]
        ), (m["strict_f1"], sm["cal_metrics"])
        assert (
            max(
                abs(m[k] - sm["cal_metrics"][ck])
                for k, ck in [
                    ("tAP", "cal_t_ap"),
                    ("gAP", "cal_g_ap"),
                    ("blindAP", "cal_blind_spot_ap"),
                ]
            )
            < 1e-4
        )
        rec = dict(
            weights=a.weights,
            policy=a.policy,
            replay_max_abs_delta=a.parity(t, g, pred),
            Cal=m,
            Fit_sample={
                k: pred_fit_metrics[k]
                for k in ["tAP", "gAP", "strict_f1", "blindAP", "by_source", "arctic_S"]
            },
            joint_history=h,
            gradients=a.gradient(
                [pick[i] for i in selected], [xfit[i] for i in selected]
            ),
        )
        if do_perm:
            rec["permutations"] = a.intervene(cal, xcal, t, g, bank, plans)
        out["arms"][name] = rec
        if name in ("A1", "A3"):
            b = "S0_20261121"
            assert orders[b] == orders[name]
            out["paired"][name] = paired(t, g, preds[b], policies[b], pred, a.policy)
    s0 = [k for k in ARMS if k.startswith("S0")]
    s_calls = [calls(preds[k], policies[k]) for k in s0]
    tmiss = np.stack([c[2] == "C" for c in s_calls])
    gmiss = np.stack([c[3] == "N" for c in s_calls])
    tm = t.l2 & t.y.isin(["S", "O"])
    gm = g.l2 & g.y.eq("I")
    out["three_seed_common_binary_FN"] = {
        "target": int(np.sum(tm & tmiss.all(0))),
        "blind": int(np.sum(tm & t.blind & tmiss.all(0))),
        "gap": int(np.sum(gm & gmiss.all(0))),
        "missed_by_any_target": int(np.sum(tm & tmiss.any(0))),
        "missed_by_any_gap": int(np.sum(gm & gmiss.any(0))),
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    path = p.parse_args().out.resolve()
    if path.is_relative_to(RUN.parents[1]) or path.exists():
        raise FileExistsError("Use a new output outside the campaign")
    payload = json.dumps(audit(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with path.open("x") as f:
        f.write(payload)
    print(path, flush=True)


if __name__ == "__main__":
    main()
