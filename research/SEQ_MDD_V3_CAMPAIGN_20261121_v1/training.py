"""Fit/calibrate on admitted data only; evaluate frozen heads with byte-sealed predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gc
import math
import resource
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score
import torch
from torch.nn import functional as F

from research.CONTROLLED_PILOT_AB_20261118_v1.run_pilot_ab import WEIGHTS_DIR
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    SeqMDDv2,
    compute_multitask_mdd_loss,
)
from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    calibrate_dual_policies,
    GAP_LABEL_TO_ID,
    TARGET_LABEL_TO_ID,
)
from .data import (
    COHORTS,
    GOAL_COMMIT,
    HERE,
    V2_CKPT,
    V2_SHA,
    admitted,
    enforce_vram,
    fill_cache,
    forward_frames,
    read_frames,
    read_json,
    save_json,
    sha_file,
    stable_sha,
    summary,
)
from .features import Spec, WindowHead, pool_features
from .scoring import brief, evaluate_seal, scores_to_rows, seal_predictions

ARCHIVED_POLICY = {
    "tau_t": 0.29,
    "tau_g": 0.825,
    "alpha_realized": 0.0,
    "temperature": {"default": [1.0, 1.0]},
}
SCALAR_KEYS = ("target_scalars", "gap_scalars")
CODE_FILES = (
    "data.py",
    "features.py",
    "scoring.py",
    "training.py",
    "ctc_delete.cpp",
    "run_exp1_window.py",
)


def domain(r):
    s = r.get("source", r.get("cohort", ""))
    if s.startswith(("timit", "l1", "mocha", "librispeech")):
        return "l1"
    if s.startswith("epadb"):
        return "epadb"
    if s.startswith("saa"):
        return "saa"
    return "l2arctic"


def release_cuda(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def features_from_cache(rows, spec, cache, kernel, tag):
    start = time.perf_counter()
    out = []
    for i, r in enumerate(rows, 1):
        h, lp = read_frames(cache, r)
        out.append(pool_features(h, lp, r["canonical_phones"], spec, kernel))
        if i % 800 == 0 or i == len(rows):
            print(
                f"POOL {tag} {i}/{len(rows)} ({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    return out


def norm_from_fit(features):
    stats = {}
    for key in SCALAR_KEYS:
        all_feats = np.concatenate([x[key] for x in features])
        stats[key] = (
            all_feats.mean(0).astype(np.float32),
            np.maximum(all_feats.std(0), 1e-5).astype(np.float32),
        )
    return stats


def collate(rows, feats, stats, device, *, labels):
    lengths = [len(r["canonical_phones"]) for r in rows]
    max_l = max(lengths)
    args = {}
    for name, arr in feats[0].items():
        if name in {"h_gap", "lp_gap_mean", "lp_gap_max", "gap_scalars"}:
            extra = 1
        else:
            extra = 0
        dtype = np.int64 if name.endswith("phone_ids") else np.float32
        data = np.zeros((len(rows), max_l + extra) + arr.shape[1:], dtype=dtype)
        for i, (item, L) in enumerate(zip(feats, lengths, strict=True)):
            v = item[name]
            if name in stats:
                m, s = stats[name]
                v = (v - m) / s
            data[i, : L + extra] = v
        args[name] = torch.from_numpy(data).to(device)
    args["lengths"] = torch.tensor(lengths, dtype=torch.long, device=device)
    if not labels:
        return args, None
    t, g, r = (np.full((len(rows), max_l + e), -100, dtype=np.int64) for e in (0, 1, 0))
    for i, (row, L) in enumerate(zip(rows, lengths, strict=True)):
        t[i, :L] = [TARGET_LABEL_TO_ID.get(a, -100) for a in row["gold_targets"]]
        g[i, : L + 1] = [GAP_LABEL_TO_ID.get(a, -100) for a in row["gold_gaps"]]
        if "realized_ids" in row:
            r[i, :L] = row["realized_ids"]
    l1 = np.array([x.get("is_native_l1", False) for x in rows])
    return args, tuple(torch.from_numpy(a).to(device) for a in (t, g, r, l1))


def objective(logits, labels, spec):
    t, g, r, l1 = labels
    tgt, gap, aux = (
        logits["target_logits"],
        logits["gap_logits"],
        logits["realized_logits"],
    )
    if not spec.focal and (g != -100).any():
        return compute_multitask_mdd_loss(
            logits,
            t,
            g,
            r,
            l1,
            gap_pos_weight=8.0,
            lambda_gap=0.6,
            lambda_realized=0.25,
            native_fa_weight=1.5,
        )["loss"]
    ce = F.cross_entropy(
        tgt.flatten(0, 1), t.flatten(), ignore_index=-100, reduction="none"
    ).reshape_as(t)
    tl = (ce * torch.where(l1[:, None], 1.5, 1.0)).sum() / (t != -100).sum().clamp_min(
        1
    )
    ok = g != -100
    if ok.any():
        q = gap[ok].log_softmax(-1).gather(-1, g[ok][:, None]).squeeze(-1)
        gl = (
            -torch.where(g[ok] == 1, 0.75, 0.25) * (1.0 - q.exp()).square() * q
        ).mean()
    else:
        gl = gap.sum() * 0.0
    rl = (
        F.cross_entropy(aux.flatten(0, 1), r.flatten(), ignore_index=-100)
        if (r != -100).any()
        else aux.sum() * 0.0
    )
    return tl + 0.6 * gl + 0.25 * rl


@dataclass
class Logits:
    target: np.ndarray
    gap: np.ndarray
    realized: np.ndarray
    can: np.ndarray


@torch.inference_mode()
def infer_logits(head, rows, feats, norm, device, batch_size):
    out = []
    head.eval()
    for i in range(0, len(rows), batch_size):
        rsub = rows[i : i + batch_size]
        inp, _ = collate(rsub, feats[i : i + batch_size], norm, device, labels=False)
        logits = head(**inp)
        t, g, r = (
            logits[k].cpu().numpy()
            for k in ("target_logits", "gap_logits", "realized_logits")
        )
        for j, row in enumerate(rsub):
            n = len(row["canonical_phones"])
            out.append(
                Logits(t[j, :n], g[j, : n + 1], r[j, :n], feats[i + j]["phone_ids"])
            )
    return out


def probabilities(logits, row, policy):
    temp = policy["temperature"]
    t, g = temp.get(domain(row), temp["default"])
    target = softmax(logits.target / np.float32(t), axis=-1)
    gap = softmax(logits.gap / np.float32(g), axis=-1)
    a = policy["alpha_realized"]
    if a:
        real = softmax(logits.realized, axis=-1)
        c = real[np.arange(len(real)), logits.can]
        o = real[:, -1]
        rf = np.stack((c, np.maximum(0.0, 1.0 - c - o), o), axis=-1)
        target = (1.0 - a) * target + a * rf
    return target, gap


def cal_arrays(rows, logits, policy):
    buckets = [
        {"p_s": [], "p_o": [], "gap_i": [], "gold_t": [], "gold_g": []}
        for _ in range(2)
    ]
    for row, l in zip(rows, logits, strict=True):
        target, gap = probabilities(l, row, policy)
        b = buckets[int(row.get("is_native_l1", False))]
        for k, v in [
            ("p_s", target[:, 1]),
            ("p_o", target[:, 2]),
            ("gap_i", gap[:, 1]),
            ("gold_t", row["gold_targets"]),
            ("gold_g", row["gold_gaps"]),
        ]:
            b[k].extend(v)
    return [{k: np.asarray(v) for k, v in b.items()} for b in buckets]


def cal_score(rows, logits):
    l2, _ = cal_arrays(rows, logits, ARCHIVED_POLICY)
    t = l2["gold_t"] != "unknown"
    g = l2["gold_g"] != "unknown"
    a = float(
        average_precision_score(
            np.isin(l2["gold_t"][t], ["S", "O"]), (l2["p_s"] + l2["p_o"])[t]
        )
    )
    b = float(average_precision_score(l2["gold_g"][g] == "I", l2["gap_i"][g]))
    return a + 0.15 * b, a, b


def _temp(x, y):
    mask = y >= 0
    x, y = x[mask], y[mask]
    if len(set(y)) < 2:
        return None

    def loss(logt):
        z = x / np.exp(logt)
        return float((logsumexp(z, axis=1) - z[np.arange(len(z)), y]).mean())

    r = minimize_scalar(loss, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
    if not r.success:
        raise ValueError("Temperature fit failed on Calibrate")
    return float(np.exp(r.x))


def fit_temperatures(rows, logits):
    def ts(ix):
        t = np.concatenate([logits[i].target for i in ix]).astype(np.float64)
        g = np.concatenate([logits[i].gap for i in ix]).astype(np.float64)
        yt = np.array(
            [
                TARGET_LABEL_TO_ID.get(y, -100)
                for i in ix
                for y in rows[i]["gold_targets"]
            ]
        )
        yg = np.array(
            [GAP_LABEL_TO_ID.get(y, -100) for i in ix for y in rows[i]["gold_gaps"]]
        )
        return [_temp(t, yt), _temp(g, yg)]

    idx = [i for i, r in enumerate(rows) if not r.get("is_native_l1", False)]
    base = [v if v is not None else 1.0 for v in ts(idx)]
    table = {"default": base}
    for name in sorted({domain(rows[i]) for i in idx}):
        z = ts([i for i in idx if domain(rows[i]) == name])
        table[name] = [b if a is None else a for a, b in zip(z, base, strict=True)]
    return table


def choose_policy(rows, logits, spec):
    temp = (
        fit_temperatures(rows, logits) if spec.temperature else {"default": [1.0, 1.0]}
    )
    best = None
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) if spec.fuse_realized else (0.0,):
        extra = {"temperature": temp, "alpha_realized": alpha}
        l2, l1 = cal_arrays(rows, logits, extra)
        if not len(l2["gold_t"]) or not len(l1["gold_t"]):
            raise ValueError("Need L2 and clean L1 Calibrate")
        policies = calibrate_dual_policies(l2, l1)
        trial = {**policies["cal_f1"], **extra}
        if best is None or trial["cal_l2_strict_f1"] > best["cal_l2_strict_f1"]:
            best = trial
    return best


def sample_probs(rows):
    w = {"l2arctic": 0.45, "epadb": 0.30, "l1": 0.25}
    dom = [domain(r) for r in rows]
    cnt = Counter(dom)
    if set(w) != set(cnt):
        raise ValueError("Unexpected Fit domains for balancing")
    return np.array([w[d] / cnt[d] for d in dom])


def code_hashes():
    return {f: sha_file(HERE / f) for f in CODE_FILES if (HERE / f).is_file()}


def train_arm(
    out, fit, cal, spec, run, device, seed, epochs, batch_size, *, kernel=None
):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cache = run / "frames"
    x = features_from_cache(fit, spec, cache, kernel, out.name + "/fit")
    c = features_from_cache(cal, spec, cache, kernel, out.name + "/cal")
    norm = norm_from_fit(x)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    head = WindowHead(spec).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=8e-4, weight_decay=0.02)
    steps = math.ceil(len(fit) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps * epochs, eta_min=2e-5
    )
    probs = sample_probs(fit) if spec.corpus_balance else None
    best = -float("inf")
    state = None
    hist = []
    for epoch in range(1, epochs + 1):
        t = time.perf_counter()
        head.train()
        order = (
            rng.choice(len(fit), size=len(fit), replace=True, p=probs)
            if probs is not None
            else rng.permutation(len(fit))
        )
        loss_sum = torch.zeros((), device=device)
        for i in range(0, len(fit), batch_size):
            batch = order[i : i + batch_size]
            inp, labels = collate(
                [fit[a] for a in batch],
                [x[a] for a in batch],
                norm,
                device,
                labels=True,
            )
            opt.zero_grad(set_to_none=True)
            loss = objective(head(**inp), labels, spec)
            loss.backward()
            nnorm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            if not torch.isfinite(nnorm).item():
                raise FloatingPointError(f"Invalid grad at {epoch=} {i=}")
            enforce_vram(device)
        sel, tap, gap = cal_score(
            cal, infer_logits(head, cal, c, norm, device, batch_size)
        )
        if sel > best:
            best = sel
            best_ep = epoch
            state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        row = {
            "epoch": epoch,
            "loss": float(loss_sum / steps),
            "cal_t_ap": tap,
            "cal_g_ap": gap,
            "wall_s": time.perf_counter() - t,
        }
        hist.append(row)
        print(
            f"EPOCH {out.name} {epoch}/{epochs} loss={row['loss']:.4f} cal AP={tap:.4f}/{gap:.4f} ({row['wall_s']:.1f}s)",
            flush=True,
        )
    head.load_state_dict(state)
    policy = choose_policy(
        cal, infer_logits(head, cal, c, norm, device, batch_size), spec
    )
    peak = enforce_vram(device)
    card = save_model(
        out,
        head,
        norm,
        spec,
        policy,
        seed,
        best_ep,
        extra={
            "optimizer": {
                "lr": 8e-4,
                "weight_decay": 0.02,
                "epochs": epochs,
                "batch_size": batch_size,
                "schedule": "cosine; min 2e-5",
            },
            "fit": summary(fit),
            "calibrate": summary(cal),
            "fit_rows_sha256": stable_sha(fit),
            "cal_rows_sha256": stable_sha(cal),
            "cal_epoch_selection": "target AP + 0.15 gap AP (Calibrate L2 only)",
            "peak_reserved_mib": peak,
            "runtime_s": time.perf_counter() - start,
            "ram_hwm_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "training_history": hist,
            "kernel_sha256": sha_file(kernel) if kernel and spec.ollr else None,
            "frame_card": read_json(cache / "FRAME_CARD.json"),
        },
    )
    del opt, sched, loss, head, x, c, inp, labels
    release_cuda(device)
    return card


def save_model(out, head, norm, spec, policy, seed, best_epoch, extra=None):
    cp = out / "checkpoint_best.pt"
    with cp.open("xb") as f:
        torch.save(
            {
                "spec": spec.as_dict(),
                "state_dict": head.state_dict(),
                "norm": {
                    k: tuple(torch.from_numpy(v) for v in tup)
                    for k, tup in norm.items()
                },
                "policy": policy,
            },
            f,
        )
    card = {
        "checkpoint": cp.name,
        "sha256": sha_file(cp),
        "goal_commit": GOAL_COMMIT,
        "spec": spec.as_dict(),
        "seed": seed,
        "best_epoch": best_epoch,
        "policy": policy,
        "locked_tau_t": policy["tau_t"],
        "locked_tau_g": policy["tau_g"],
        "learned_layer_weights": {
            k: getattr(head, k + "_layer_logits").detach().cpu().softmax(0).tolist()
            for k in ("target", "gap")
        },
        "code_sha256": code_hashes(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        **(extra or {}),
    }
    save_json(out / "CHECKPOINT_CARD.json", card)
    return card


def load_checkpoint(out, device):
    card = read_json(out / "CHECKPOINT_CARD.json")
    p = out / card["checkpoint"]
    if sha_file(p) != card["sha256"]:
        raise ValueError("Checkpoint changed")
    d = torch.load(p, map_location="cpu", weights_only=True)
    spec = Spec(**d["spec"])
    head = WindowHead(spec)
    head.load_state_dict(d["state_dict"])
    return (
        head.to(device).eval(),
        {k: tuple(x.numpy() for x in v) for k, v in d["norm"].items()},
        spec,
        card["policy"],
    )


def load_baseline(out, device):
    if sha_file(V2_CKPT) != V2_SHA:
        raise ValueError("Archived V2 changed")
    d = torch.load(V2_CKPT, map_location="cpu", weights_only=False)
    head = SeqMDDv2(**d["model_kwargs"])
    head.load_state_dict(d["state_dict"])
    card = {
        "checkpoint": str(V2_CKPT),
        "sha256": V2_SHA,
        "goal_commit": GOAL_COMMIT,
        "policy": ARCHIVED_POLICY,
        "code_sha256": code_hashes(),
    }
    save_json(out / "CHECKPOINT_CARD.json", card)
    return (
        head.to(device).eval(),
        {
            "target_scalars": (d["t_mean"], d["t_std"]),
            "gap_scalars": (d["g_mean"], d["g_std"]),
        },
        Spec(),
        ARCHIVED_POLICY,
    )


def pred_batches(head, norm, spec, policy, rows, cache, device, kernel):
    records = []
    for i, row in enumerate(rows, 1):
        f = [
            pool_features(
                *read_frames(cache, row), row["canonical_phones"], spec, kernel
            )
        ]
        l = infer_logits(head, [row], f, norm, device, 1)[0]
        records.extend(scores_to_rows(row, *probabilities(l, row, policy), policy))
        if i % 800 == 0 or i == len(rows):
            print(f"PREDICT {i}/{len(rows)}", flush=True)
    return records


def benchmark_live(
    head, norm, spec, policy, rows, device, kernel, *, full=False, backbone=None
):
    """Serial WAV-open to decision, unbatched, GPU synchronized. Fixed per-cohort subsample or full."""
    if device.type != "cuda":
        return {}
    if any(sha_file(WEIGHTS_DIR / k) != v for k, v in (backbone or {}).items()):
        raise ValueError("Frozen backbone files changed")
    torch.cuda.reset_peak_memory_stats(device)
    model = load_base_koel(device)
    out = {}
    for name in COHORTS:
        cohort = [r for r in rows if r["cohort"] == name]
        if not cohort:
            continue
        count = len(cohort) if full or name == "saa27" else min(36, len(cohort))
        test = [cohort[i] for i in np.linspace(0, len(cohort) - 1, count, dtype=int)]
        samples = []
        kernel_s = []
        for i, row in enumerate([*test[:2], *test]):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            h, lp = forward_frames(model, row, device, verify=False)
            f = [pool_features(h, lp, row["canonical_phones"], spec, kernel)]
            l = infer_logits(head, [row], f, norm, device, 1)[0]
            tuple(scores_to_rows(row, *probabilities(l, row, policy), policy))
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= 2:
                samples.append(elapsed)
                if spec.ollr:
                    from .features import deletion_feature

                    t1 = time.perf_counter()
                    deletion_feature(lp, row["canonical_phones"], kernel)
                    kernel_s.append((time.perf_counter() - t1) * 1000)
            enforce_vram(device)
        a = np.asarray(samples)
        out[name] = {
            "n": len(a),
            "full_cohort": len(a) == len(cohort),
            "p50_ms": float(np.median(a)),
            "p95_ms": float(np.quantile(a, 0.95)),
            "samples_ms": samples,
            "uids": [r["uid"] for r in test],
        }
        if kernel_s:
            out[name]["ollr_p50_ms"] = float(np.median(kernel_s))
            out[name]["ollr_p95_ms"] = float(np.quantile(kernel_s, 0.95))
    out["protocol"] = (
        "synchronized WAV→label-free calls, 2 warmups/cohort, serial offline local I/O"
    )
    out["peak_reserved_mib"] = enforce_vram(device)
    del model
    release_cuda(device)
    return out


def evaluate_arm(
    out,
    run,
    device,
    comparisons,
    *,
    kernel=None,
    baseline=False,
    rows=None,
    gold_rows=None,
    latency=False,
    full_latency=False,
):
    split = run / "splits"
    audit = read_json(split / "PREFLIGHT.json")
    if rows is None:
        rows = admitted(split, "eval_prompts.jsonl")
    head, norm, spec, policy = (load_baseline if baseline else load_checkpoint)(
        out, device
    )
    print(f"EVAL {out.name}, frozen policy {policy}", flush=True)
    perf = (
        benchmark_live(
            head,
            norm,
            spec,
            policy,
            rows,
            device,
            kernel,
            full=full_latency,
            backbone=audit["backbone"],
        )
        if latency
        else {}
    )
    records = pred_batches(
        head, norm, spec, policy, rows, run / "frames", device, kernel
    )
    seal_predictions(out, records, sha_file(out / "CHECKPOINT_CARD.json"))
    ev, boots = evaluate_seal(out, split, comparisons, gold_rows)
    ret = {
        "eval": ev,
        "bootstrap": boots,
        "latency": perf,
        "checkpoint_card_sha256": sha_file(out / "CHECKPOINT_CARD.json"),
    }
    save_json(out / "RESULTS.json", ret)
    print(out.name, brief(ev), flush=True)
    del head
    release_cuda(device)
    return ret


def replay_baseline(run, device, r0):
    out = run / "v2_keyed"
    out.mkdir()
    split = run / "splits"
    rows = admitted(split, "eval_prompts.jsonl")
    audit = read_json(split / "PREFLIGHT.json")
    save_json(
        out / "EVAL_FRAME_RECEIPT.json",
        fill_cache(rows, run / "frames", device, audit["backbone"]),
    )
    score = evaluate_arm(out, run, device, {"r0_old": r0}, baseline=True)
    legacy = read_json(
        V2_CKPT.parent.parent
        / "outputs_seq_mdd_v2/SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json"
    )
    for name in COHORTS:
        d = score["eval"][name]
        b = legacy["eval"][name]
        if name == "timit_core":
            ok = d["fa_total"] == b["fa_total"]
        else:
            ok = all(
                d["strict"][k] == b["cal_f1"]["strict"][k] for k in ("tp", "fp", "fn")
            )
        if not ok:
            raise ValueError(
                f"V2 frozen replay differs on {name}: {brief(score['eval'])}"
            )
    ctrl = score["eval"]["timit_core_corrupt_ctrl"]
    if (
        round(ctrl["recall"] * 100, 2)
        != legacy["timit_corruption_control"]["corrupt_recall_pct"]
    ):
        raise ValueError("V2 TIMIT corruption replay differs")
    save_json(
        out / "BASELINE_PARITY.json",
        {
            "status": "exact locked counts on all five cohorts and control recall",
            "legacy_sha256": sha_file(
                V2_CKPT.parent.parent
                / "outputs_seq_mdd_v2/SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json"
            ),
        },
    )
    return score
