"""Train or verify a versioned V2 control with repaired TIMIT prompts and EOS handling."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .data import (
    CONTROL,
    GOAL_COMMIT,
    SEED,
    V2_CKPT,
    V2_SHA,
    admitted,
    enforce_vram,
    read_json,
    save_json,
    sha_file,
)
from .features import Spec, WindowHead
from .repair_model import LengthAwareV2
from .repair_prompts import DEFAULT_RUN, DEFAULT_REPAIR, REPAIR_ID, ensure_separate
from .scoring import brief, evaluate_seal, seal_predictions
from .training import (
    ARCHIVED_POLICY,
    cal_score,
    choose_policy,
    collate,
    features_from_cache,
    infer_logits,
    norm_from_fit,
    objective,
    pred_batches,
)

V2_SPEC = Spec()
ARCHITECTURES = {"fixed": LengthAwareV2, "legacy-eos-control": WindowHead}


def code_receipt():
    root = Path(__file__).parent
    return {
        name: sha_file(root / name)
        for name in (
            "repair_model.py",
            "repair_prompts.py",
            "run_repaired_control.py",
            "features.py",
            "training.py",
        )
    }


def check_inputs(run, split):
    card = read_json(split / "PREFLIGHT.json")
    parent = run / "splits/PREFLIGHT.json"
    if card.get("repair_id") != REPAIR_ID or card[
        "parent_preflight_sha256"
    ] != sha_file(parent):
        raise ValueError("Repair split not bound to this campaign")
    old_hashes = read_json(parent)["manifest_sha256"]
    for name in ("fit.jsonl", "cal.jsonl", "eval_gold.jsonl"):
        if card["manifest_sha256"][name] != old_hashes[name]:
            raise ValueError(f"Baseline contract changed: {name}")
    frame_card = read_json(run / "frames/FRAME_CARD.json")
    if frame_card["backbone"] != card["backbone"]:
        raise ValueError("Wrong Koel frame cache")
    return sha_file(split / "PREFLIGHT.json")


def state_sha(state):
    h = hashlib.sha256()
    for name, p in sorted(state.items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(mode, seed):
    # Explicitly reset the stream after all feature precomputation. Both arms start
    # from the same parameter bytes and use the same batch permutation stream.
    torch.manual_seed(seed)
    return ARCHITECTURES[mode]()


def fit_control(run, split, out, device, *, mode, seed, epochs, batch_size):
    if min(epochs, batch_size) < 1:
        raise ValueError("Epochs and batch size must be positive")
    ensure_separate(run, out)
    split_sha = check_inputs(run, split)
    fit, cal = [admitted(split, f"{s}.jsonl") for s in ("fit", "cal")]
    out.mkdir(parents=True, exist_ok=False)
    cache = run / "frames"
    xf = features_from_cache(fit, V2_SPEC, cache, None, "fit/repair")
    xc = features_from_cache(cal, V2_SPEC, cache, None, "cal/repair")
    norm = norm_from_fit(xf)
    model = initialize(mode, seed).to(device)
    init_sha = state_sha(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    n_steps = math.ceil(len(fit) / batch_size)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * n_steps, eta_min=2e-5
    )
    rng, order_hash = np.random.default_rng(seed), hashlib.sha256()
    history, best, best_state = [], -math.inf, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fit))
        order_hash.update(order.astype("<i8").tobytes())
        total = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            inputs, labels = collate(
                [fit[i] for i in ix], [xf[i] for i in ix], norm, device, labels=True
            )
            opt.zero_grad(set_to_none=True)
            loss = objective(model(**inputs), labels, V2_SPEC)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss before optimizer step: epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            lr.step()
            total += loss.detach()
            enforce_vram(device)
        score, target_ap, gap_ap = cal_score(
            cal, infer_logits(model, cal, xc, norm, device, batch_size)
        )
        row = {
            "epoch": epoch,
            "mean_fit_loss": float(total / n_steps),
            "cal_selection_score": score,
            "cal_target_ap": target_ap,
            "cal_gap_ap": gap_ap,
        }
        history.append(row)
        print(row, flush=True)
        if score > best:
            best, best_epoch = score, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError("No valid Calibrate checkpoint")
    model.load_state_dict(best_state)
    policy = choose_policy(
        cal, infer_logits(model, cal, xc, norm, device, batch_size), V2_SPEC
    )
    cp = out / "control.pt"
    with cp.open("xb") as f:
        torch.save(
            {
                "repair_id": REPAIR_ID,
                "mode": mode,
                "state_dict": best_state,
                "norm": {
                    k: tuple(torch.from_numpy(x) for x in tup)
                    for k, tup in norm.items()
                },
                "policy": policy,
            },
            f,
        )
    card = {
        "repair_id": REPAIR_ID,
        "goal_commit": GOAL_COMMIT,
        "mode": mode,
        "checkpoint": cp.name,
        "sha256": sha_file(cp),
        "repair_preflight_sha256": split_sha,
        "init_state_sha256": init_sha,
        "train_order_sha256": order_hash.hexdigest(),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "batch_size": batch_size,
        "policy": policy,
        "training_history": history,
        "peak_reserved_mib": enforce_vram(device),
        "code_sha256": code_receipt(),
        "torch": str(torch.__version__),
        "device": str(device),
    }
    save_json(out / "CHECKPOINT_CARD.json", card)
    return card


def _load_control(control, split_sha, device):
    card = read_json(control / "CHECKPOINT_CARD.json")
    cp = control / card["checkpoint"]
    if (
        card.get("repair_id") != REPAIR_ID
        or card["repair_preflight_sha256"] != split_sha
        or sha_file(cp) != card["sha256"]
    ):
        raise ValueError("Not a verified checkpoint for these repaired splits")
    if card["code_sha256"] != code_receipt():
        raise ValueError("Control code changed since training")
    d = torch.load(cp, map_location="cpu", weights_only=True)
    if (
        d["policy"] != card["policy"]
        or d["repair_id"] != REPAIR_ID
        or d["mode"] != card["mode"]
    ):
        raise ValueError("Checkpoint and card disagree")
    model = ARCHITECTURES[d["mode"]]()
    model.load_state_dict(d["state_dict"], strict=True)
    return (
        model.to(device).eval(),
        {k: tuple(t.numpy() for t in tup) for k, tup in d["norm"].items()},
        d["policy"],
        card,
    )


def _score(run, split, out, rows, model, norm, policy, card, device):
    ensure_separate(run, out)
    out.mkdir(parents=True, exist_ok=False)
    marker = out / "CHECKPOINT_CARD.json"
    save_json(marker, card)
    rec = pred_batches(model, norm, V2_SPEC, policy, rows, run / "frames", device, None)
    seal_predictions(out, rec, sha_file(marker))
    # Gold is loaded only after the fixed checkpoint/policy and predicted calls are sealed.
    keys = {(r["cohort"], r["uid"]) for r in rows}
    gold = [
        r for r in admitted(split, "eval_gold.jsonl") if (r["cohort"], r["uid"]) in keys
    ]
    ev, _ = evaluate_seal(out, split, gold_rows=gold)
    result = {"scope": card["scope"], "eval": ev}
    save_json(out / "RESULTS.json", result)
    return result


def score_control(run, split, control, out, device):
    split_sha = check_inputs(run, split)
    model, norm, policy, card = _load_control(control, split_sha, device)
    rows = admitted(split, "eval_prompts.jsonl")
    return _score(
        run,
        split,
        out,
        rows,
        model,
        norm,
        policy,
        {
            **card,
            "scope": "trained_control_diagnostic",
            "training_card_sha256": sha_file(control / "CHECKPOINT_CARD.json"),
            "checkpoint": str((control / card["checkpoint"]).resolve()),
        },
        device,
    )


def verify_archived_v2(run, split, out):
    """Run the repair on archived V2 weights for regression only; no train/Cal search."""
    split_sha = check_inputs(run, split)
    if sha_file(V2_CKPT) != V2_SHA:
        raise ValueError("Archived V2 checkpoint changed")
    d = torch.load(V2_CKPT, map_location="cpu", weights_only=False)
    model = LengthAwareV2(**d["model_kwargs"]).eval()
    model.load_state_dict(d["state_dict"], strict=True)
    norm = {
        "target_scalars": (d["t_mean"], d["t_std"]),
        "gap_scalars": (d["g_mean"], d["g_std"]),
    }
    rows = [
        r
        for r in admitted(split, "eval_prompts.jsonl")
        if r["cohort"] in {"timit_core", CONTROL}
    ]
    card = {
        "scope": "frozen_cpu_regression_not_retrained",
        "repair_id": REPAIR_ID,
        "repair_preflight_sha256": split_sha,
        "checkpoint": str(V2_CKPT),
        "sha256": V2_SHA,
        "policy": ARCHIVED_POLICY,
        "code_sha256": code_receipt(),
    }
    return _score(
        run, split, out, rows, model, norm, ARCHIVED_POLICY, card, torch.device("cpu")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--splits", type=Path, default=DEFAULT_REPAIR / "splits")
    parser.add_argument(
        "--out", type=Path, required=True, help="Fresh output directory"
    )
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser(
        "verify",
        help="CPU regression with the immutable V2 checkpoint and locked policy",
    )
    fit = commands.add_parser("fit", help="Fit/Cal only; no Eval reads")
    fit.add_argument("--mode", choices=list(ARCHITECTURES), default="fixed")
    fit.add_argument("--seed", type=int, default=SEED)
    fit.add_argument("--epochs", type=int, default=14)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ev = commands.add_parser("score", help="Seal and score a previously locked control")
    ev.add_argument("--control", type=Path, required=True)
    ev.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = parser.parse_args()
    if a.cmd == "fit":
        card = fit_control(
            a.legacy_run,
            a.splits,
            a.out,
            torch.device(a.device),
            mode=a.mode,
            seed=a.seed,
            epochs=a.epochs,
            batch_size=a.batch_size,
        )
        print("Saved", card["checkpoint"], card["sha256"], flush=True)
    else:
        result = (
            verify_archived_v2(a.legacy_run, a.splits, a.out)
            if a.cmd == "verify"
            else score_control(
                a.legacy_run, a.splits, a.control, a.out, torch.device(a.device)
            )
        )
        print(brief(result["eval"]), flush=True)


if __name__ == "__main__":
    main()
