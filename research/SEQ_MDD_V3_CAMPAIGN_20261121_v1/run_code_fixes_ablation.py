"""Matched-control ablation for Code Fixes #1, #2, #3 on top of LengthAwareV2.

Runs two arms on the exact v2_prompt_eos_repair_v1 splits, identical batch order
and seed as `fixed` (LengthAwareV2), locking thresholds strictly on Calibrate
before sealing Eval predictions:
  - fix12_arch  : Fix #1 (scale/norm/no-cross-CMVN/BiGRU balance) + Fix #2 (5-frame h_win keeping h_peak),
                  keeping original fixed loss (scalar_drop_p=0.0, fix_timit_loss=False).
  - fix123_full : fix12_arch + Fix #3 (modality dropout on scalars/gap_lp + TIMIT clean-C weight
                  + mask synthetic TIMIT Gap 'I').
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    GAP_LABEL_TO_ID,
    TARGET_LABEL_TO_ID,
)

from .data import (
    COHORTS,
    GOAL_COMMIT,
    SEED,
    admitted,
    enforce_vram,
    read_frames,
    read_json,
    save_json,
    sha_file,
)
from .features import Spec
from .repair_prompts import DEFAULT_REPAIR, DEFAULT_RUN, REPAIR_ID, ensure_separate
from .repaired_v2_arch import (
    CodeFixConfig,
    RepairedCodeFixV2,
    compute_codefix_loss,
    pool_features_codefix,
)
from .run_repaired_control import check_inputs, state_sha
from .scoring import (
    brief,
    evaluate_seal,
    gold_frame,
    keyed_merge,
    read_predictions,
    scores_to_rows,
    seal_predictions,
)
from .training import (
    Logits,
    cal_score,
    choose_policy,
    domain,
    norm_from_fit,
    probabilities,
    release_cuda,
)

ARM_CONFIGS = {
    "fix12_arch": CodeFixConfig(
        mode="fix12_arch",
        scalar_drop_p=0.0,
        gap_lp_drop_p=0.0,
        fix_timit_loss=False,
    ),
    "fix123_full": CodeFixConfig(
        mode="fix123_full",
        scalar_drop_p=0.35,
        gap_lp_drop_p=0.25,
        fix_timit_loss=True,
    ),
}
GAP_KEYS = {"h_gap", "lp_gap_mean", "lp_gap_max", "gap_scalars"}


def code_receipt() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        name: sha_file(root / name)
        for name in (
            "repaired_v2_arch.py",
            "run_code_fixes_ablation.py",
            "repair_model.py",
            "repair_prompts.py",
            "training.py",
            "scoring.py",
        )
    }


def extract_codefix_features(rows, cache, tag):
    t0 = time.perf_counter()
    out = []
    for i, r in enumerate(rows, 1):
        h, lp = read_frames(cache, r)
        out.append(pool_features_codefix(h, lp, r["canonical_phones"]))
        if i % 800 == 0 or i == len(rows):
            print(
                f"POOL {tag} {i}/{len(rows)} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    return out


def collate_codefix(rows, feats, stats, device, *, labels: bool):
    lengths = [len(r["canonical_phones"]) for r in rows]
    max_l = max(lengths)
    args = {}
    for name, arr in feats[0].items():
        extra = 1 if name in GAP_KEYS else 0
        if name.endswith("phone_ids"):
            dtype = np.int64
        elif name in {"h_win", "h_peak", "h_gap"}:
            dtype = np.float16
        else:
            dtype = np.float32
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
    t, g, r_lbl = (
        np.full((len(rows), max_l + e), -100, dtype=np.int64) for e in (0, 1, 0)
    )
    for i, (row, L) in enumerate(zip(rows, lengths, strict=True)):
        t[i, :L] = [TARGET_LABEL_TO_ID.get(a, -100) for a in row["gold_targets"]]
        g[i, : L + 1] = [GAP_LABEL_TO_ID.get(a, -100) for a in row["gold_gaps"]]
        if "realized_ids" in row:
            r_lbl[i, :L] = row["realized_ids"]
    is_l1_dom = np.array([domain(row) == "l1" for row in rows], dtype=bool)
    return args, tuple(torch.from_numpy(a).to(device) for a in (t, g, r_lbl, is_l1_dom))


@torch.inference_mode()
def infer_codefix_logits(model, rows, feats, norm, device, batch_size):
    out = []
    model.eval()
    for i in range(0, len(rows), batch_size):
        rsub = rows[i : i + batch_size]
        inp, _ = collate_codefix(
            rsub, feats[i : i + batch_size], norm, device, labels=False
        )
        logits = model(**inp)
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


def init_codefix_model(cfg: CodeFixConfig, seed: int) -> RepairedCodeFixV2:
    torch.manual_seed(seed)
    return RepairedCodeFixV2(
        d_model=cfg.d_model,
        dropout=cfg.dropout,
        scalar_drop_p=cfg.scalar_drop_p,
        gap_lp_drop_p=cfg.gap_lp_drop_p,
    )


def fit_arm(
    run: Path,
    split: Path,
    out: Path,
    cfg: CodeFixConfig,
    fit_rows,
    cal_rows,
    xf,
    xc,
    norm,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    split_sha: str,
):
    ensure_separate(run, out)
    out.mkdir(parents=True, exist_ok=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model = init_codefix_model(cfg, seed).to(device)
    init_sha = state_sha(model.state_dict())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    n_steps = math.ceil(len(fit_rows) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * n_steps, eta_min=2e-5
    )
    rng = np.random.default_rng(seed)
    order_hash = hashlib.sha256()
    history, best, best_state, best_epoch = [], -math.inf, None, 0

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        order = rng.permutation(len(fit_rows))
        order_hash.update(order.astype("<i8").tobytes())
        total = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            inputs, labels = collate_codefix(
                [fit_rows[i] for i in ix],
                [xf[i] for i in ix],
                norm,
                device,
                labels=True,
            )
            opt.zero_grad(set_to_none=True)
            loss = compute_codefix_loss(model(**inputs), *labels, cfg)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {cfg.mode} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            sched.step()
            total += loss.detach()
            enforce_vram(device)
        score, target_ap, gap_ap = cal_score(
            cal_rows, infer_codefix_logits(model, cal_rows, xc, norm, device, batch_size)
        )
        row = {
            "epoch": epoch,
            "mean_fit_loss": float(total / n_steps),
            "cal_selection_score": score,
            "cal_target_ap": target_ap,
            "cal_gap_ap": gap_ap,
            "wall_s": round(time.perf_counter() - t_ep, 2),
        }
        history.append(row)
        print(f"[{cfg.mode}] {row}", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()
    policy = choose_policy(
        cal_rows,
        infer_codefix_logits(model, cal_rows, xc, norm, device, batch_size),
        Spec(),
    )
    cp = out / "control.pt"
    with cp.open("xb") as f:
        torch.save(
            {
                "repair_id": REPAIR_ID,
                "config": cfg.as_dict(),
                "state_dict": best_state,
                "norm": {
                    k: tuple(torch.from_numpy(x) for x in tup)
                    for k, tup in norm.items()
                },
                "policy": policy,
            },
            f,
        )
    peak = enforce_vram(device)
    card = {
        "repair_id": REPAIR_ID,
        "goal_commit": GOAL_COMMIT,
        "mode": cfg.mode,
        "config": cfg.as_dict(),
        "n_params": n_params,
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
        "learned_layer_weights": {
            "target": model.target_layer_logits.detach().cpu().softmax(0).tolist(),
            "gap": model.gap_layer_logits.detach().cpu().softmax(0).tolist(),
        },
        "training_history": history,
        "peak_reserved_mib": peak,
        "wall_s": round(time.perf_counter() - t0, 2),
        "code_sha256": code_receipt(),
        "torch": str(torch.__version__),
        "device": str(device),
    }
    save_json(out / "CHECKPOINT_CARD.json", card)
    del opt, sched, model
    release_cuda(device)
    return card


def score_arm(
    run: Path,
    split: Path,
    control_dir: Path,
    out: Path,
    device: torch.device,
    split_sha: str,
    comparisons: dict,
):
    ensure_separate(run, out)
    out.mkdir(parents=True, exist_ok=False)
    card = read_json(control_dir / "CHECKPOINT_CARD.json")
    cp = control_dir / card["checkpoint"]
    if sha_file(cp) != card["sha256"] or card["repair_preflight_sha256"] != split_sha:
        raise ValueError("Checkpoint verification failed")
    d = torch.load(cp, map_location="cpu", weights_only=True)
    cfg = CodeFixConfig(**d["config"])
    model = RepairedCodeFixV2(
        d_model=cfg.d_model,
        dropout=cfg.dropout,
        scalar_drop_p=cfg.scalar_drop_p,
        gap_lp_drop_p=cfg.gap_lp_drop_p,
    )
    model.load_state_dict(d["state_dict"], strict=True)
    model = model.to(device).eval()
    norm = {k: tuple(t.numpy() for t in tup) for k, tup in d["norm"].items()}
    policy = d["policy"]

    marker_card = {
        **card,
        "scope": "trained_codefix_diagnostic",
        "training_card_sha256": sha_file(control_dir / "CHECKPOINT_CARD.json"),
        "checkpoint": str(cp.resolve()),
    }
    marker = out / "CHECKPOINT_CARD.json"
    save_json(marker, marker_card)

    rows = admitted(split, "eval_prompts.jsonl")
    records = []
    cache = run / "frames"
    for i, row in enumerate(rows, 1):
        feat = [pool_features_codefix(*read_frames(cache, row), row["canonical_phones"])]
        logit = infer_codefix_logits(model, [row], feat, norm, device, 1)[0]
        records.extend(scores_to_rows(row, *probabilities(logit, row, policy), policy))
        if i % 800 == 0 or i == len(rows):
            print(f"PREDICT {cfg.mode} {i}/{len(rows)}", flush=True)
    seal_predictions(out, records, sha_file(marker))
    ev, boots = evaluate_seal(out, split, comparisons=comparisons)
    res = {
        "scope": marker_card["scope"],
        "eval": ev,
        "bootstrap": boots,
        "checkpoint_card_sha256": sha_file(marker),
    }
    save_json(out / "RESULTS.json", res)
    print(f"SCORED {cfg.mode}: {brief(ev)}", flush=True)
    del model
    release_cuda(device)
    return res


def run_ablation(
    run: Path,
    split: Path,
    fixed_eval_dir: Path,
    out: Path,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
):
    ensure_separate(run, out)
    split_sha = check_inputs(run, split)
    out.mkdir(parents=True, exist_ok=False)

    fit_rows = admitted(split, "fit.jsonl")
    cal_rows = admitted(split, "cal.jsonl")
    cache = run / "frames"
    xf = extract_codefix_features(fit_rows, cache, "fit/codefix")
    xc = extract_codefix_features(cal_rows, cache, "cal/codefix")
    norm = norm_from_fit(xf)

    cards = {}
    for arm_name, cfg in ARM_CONFIGS.items():
        cards[arm_name] = fit_arm(
            run,
            split,
            out / arm_name,
            cfg,
            fit_rows,
            cal_rows,
            xf,
            xc,
            norm,
            device,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            split_sha=split_sha,
        )

    del xf, xc
    gc.collect()

    # Verify matched init and batch order between fix12_arch and fix123_full
    assert (
        cards["fix12_arch"]["init_state_sha256"]
        == cards["fix123_full"]["init_state_sha256"]
    )
    assert (
        cards["fix12_arch"]["train_order_sha256"]
        == cards["fix123_full"]["train_order_sha256"]
    )
    fixed_card = read_json(fixed_eval_dir / "CHECKPOINT_CARD.json")
    assert (
        cards["fix12_arch"]["train_order_sha256"] == fixed_card["train_order_sha256"]
    ), "Batch order must match archived fixed baseline"

    save_json(
        out / "MATCH_RECEIPT.json",
        {
            "match": True,
            "repair_preflight_sha256": split_sha,
            "init_state_sha256_codefix_pair": cards["fix12_arch"]["init_state_sha256"],
            "train_order_sha256_matches_fixed": cards["fix12_arch"][
                "train_order_sha256"
            ],
            "fixed_eval_card_sha256": sha_file(fixed_eval_dir / "CHECKPOINT_CARD.json"),
            "policies_locked": {k: v["policy"] for k, v in cards.items()},
        },
    )

    fixed_preds = read_predictions(fixed_eval_dir)
    eval_results = {}
    for arm_name in ("fix12_arch", "fix123_full"):
        comps = {"fixed": fixed_preds}
        if arm_name == "fix123_full":
            comps["fix12_arch"] = read_predictions(out / "eval_fix12_arch")
        eval_results[arm_name] = score_arm(
            run,
            split,
            out / arm_name,
            out / f"eval_{arm_name}",
            device,
            split_sha,
            comps,
        )

    # Independent CPU recount from sealed Parquets
    gold_df = gold_frame(admitted(split, "eval_gold.jsonl"))
    fixed_ev, _ = evaluate_seal(fixed_eval_dir, split)
    recount_ok = True
    for arm_name in ("fix12_arch", "fix123_full"):
        ev_re, _ = evaluate_seal(out / f"eval_{arm_name}", split)
        if ev_re != eval_results[arm_name]["eval"]:
            recount_ok = False
        # Also verify exact slot alignment against fixed
        keyed_merge(read_predictions(out / f"eval_{arm_name}"), gold_df)

    paired = {
        "scope": "diagnostic_single_seed_code_fixes_1_2_3_on_repaired_v2_splits",
        "cpu_recount_match": recount_ok,
        "repair_preflight_sha256": split_sha,
        "train_order_sha256": cards["fix12_arch"]["train_order_sha256"],
        "heads": {
            "fixed": {
                "best_epoch": fixed_card["best_epoch"],
                "policy": fixed_card["policy"],
                "checkpoint_sha256": fixed_card["sha256"],
            },
            **{
                k: {
                    "best_epoch": c["best_epoch"],
                    "policy": c["policy"],
                    "checkpoint_sha256": c["sha256"],
                    "n_params": c["n_params"],
                    "learned_layer_weights": c["learned_layer_weights"],
                }
                for k, c in cards.items()
            },
        },
        "summary_strict_or_fa": {
            "fixed": brief(fixed_ev),
            "fix12_arch": brief(eval_results["fix12_arch"]["eval"]),
            "fix123_full": brief(eval_results["fix123_full"]["eval"]),
        },
        "target_and_gap_ap": {
            arm_name: {
                c: {
                    "target_ap": ev[c].get("target_ap"),
                    "gap_ap": ev[c].get("gap_ap"),
                    "i_f1": ev[c]["I"]["f1"] if "I" in ev[c] else None,
                    "s_to_o": ev[c].get("s_to_o"),
                    "o_to_s": ev[c].get("o_to_s"),
                }
                for c in COHORTS[:-1]
            }
            for arm_name, ev in [
                ("fixed", fixed_ev),
                ("fix12_arch", eval_results["fix12_arch"]["eval"]),
                ("fix123_full", eval_results["fix123_full"]["eval"]),
            ]
        },
        "bootstrap_vs_fixed": {
            "fix12_arch": eval_results["fix12_arch"]["bootstrap"]["fixed"],
            "fix123_full": eval_results["fix123_full"]["bootstrap"]["fixed"],
        },
        "bootstrap_fix123_vs_fix12": eval_results["fix123_full"]["bootstrap"][
            "fix12_arch"
        ],
        "full_eval": {
            "fixed": fixed_ev,
            "fix12_arch": eval_results["fix12_arch"]["eval"],
            "fix123_full": eval_results["fix123_full"]["eval"],
        },
    }
    save_json(out / "PAIRED_CODE_FIXES_RESULTS.json", paired)
    print("DONE:", paired["summary_strict_or_fa"], flush=True)
    return paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--splits", type=Path, default=DEFAULT_REPAIR / "splits")
    p.add_argument(
        "--fixed-eval",
        type=Path,
        default=DEFAULT_REPAIR / "matched_seed20261121_e14_b16/eval_fixed",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    run_ablation(
        a.legacy_run,
        a.splits,
        a.fixed_eval,
        a.out,
        torch.device(a.device),
        seed=a.seed,
        epochs=a.epochs,
        batch_size=a.batch_size,
    )


if __name__ == "__main__":
    main()
