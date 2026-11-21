"""Paired 3-seed confirmation: `iso_no_hwin` (pure Fix #1) vs `fixed` (LengthAwareV2).

Runs 3 paired seeds (20261121, 20261122, 20261123) on the exact v2_prompt_eos_repair_v1
splits with identical batch permutation order (`train_order_sha256`), identical optimizer
and V2 multi-task loss, Calibrate-only epoch & threshold locking, pre-gold Parquet seals,
standalone `independent_count` verification, Calibrate L2 recall-matched comparison,
and GTX 1650 FP32 latency measurement.
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

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import compute_multitask_mdd_loss

from .data import (
    COHORTS,
    CONTROL,
    GOAL_COMMIT,
    admitted,
    enforce_vram,
    read_frames,
    save_json,
    sha_file,
)
from .features import Spec, pool_features
from .repair_model import LengthAwareV2
from .repair_prompts import DEFAULT_REPAIR, DEFAULT_RUN, REPAIR_ID, ensure_separate
from .replay_codefix_audit import (
    apply_policy_df,
    match_cal_recall,
    summarize_cohorts,
)
from .run_repaired_control import check_inputs
from .scoring import (
    evaluate_seal,
    gold_frame,
    keyed_merge,
    read_predictions,
    scores_to_rows,
    seal_predictions,
)
from .single_factor_models import (
    FACTOR_ARMS,
    SingleFactorV2,
    pool_features_with_hvit,
)
from .run_single_factor_isolation import collate_sf, infer_sf_logits
from .training import (
    cal_score,
    choose_policy,
    collate,
    domain,
    infer_logits,
    norm_from_fit,
    probabilities,
    release_cuda,
)

SEEDS = (20261121, 20261122, 20261123)
CFG_FIX1 = FACTOR_ARMS["iso_no_hwin"]


def _cal_score_arrays(rows, logits, policy):
    l2 = {"t_sc": [], "t_g": [], "g_sc": [], "g_g": []}
    l1 = {"t_sc": [], "g_sc": []}
    for r, lg in zip(rows, logits, strict=True):
        tp, gp = probabilities(lg, r, policy)
        sc = tp[:, 1] + tp[:, 2]
        if domain(r) == "l1":
            l1["t_sc"].extend(sc.tolist())
            l1["g_sc"].extend(gp[:, 1].tolist())
        else:
            l2["t_sc"].extend(sc.tolist())
            l2["t_g"].extend(r["gold_targets"])
            l2["g_sc"].extend(gp[:, 1].tolist())
            l2["g_g"].extend(r["gold_gaps"])
    return (
        {k: np.asarray(v) for k, v in l2.items()},
        {k: np.asarray(v) for k, v in l1.items()},
    )


def _pool_no_hwin(h, lp, phones):
    """Same as pool_features_with_hvit but drops unused h_win to save RAM/VRAM."""
    f = pool_features_with_hvit(h, lp, phones)
    f["h_win"] = f["h_peak"][:, None, :, :]  # 1-frame stub (unused when use_h_win=False)
    return f


def fit_and_eval_one(
    arm: str,
    seed: int,
    out_dir: Path,
    split: Path,
    cache: Path,
    fit_rows,
    cal_rows,
    ev_rows,
    xf,
    xc,
    norm,
    gold_df,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    split_sha: str,
    comparisons: dict,
):
    out_dir.mkdir(parents=True, exist_ok=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    is_fix1 = arm == "iso_no_hwin"
    torch.manual_seed(seed)
    model = (SingleFactorV2(CFG_FIX1) if is_fix1 else LengthAwareV2()).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    n_steps = math.ceil(len(fit_rows) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * n_steps, eta_min=2e-5
    )
    rng = np.random.default_rng(seed)
    order_hash = hashlib.sha256()
    hist, best_sel, best_ep, best_state = [], -math.inf, 0, None

    collate_fn = collate_sf if is_fix1 else collate
    infer_fn = infer_sf_logits if is_fix1 else infer_logits

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fit_rows))
        order_hash.update(order.astype("<i8").tobytes())
        loss_sum = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            inp, labels = collate_fn(
                [fit_rows[i] for i in ix], [xf[i] for i in ix], norm, device, labels=True
            )
            opt.zero_grad(set_to_none=True)
            loss = compute_multitask_mdd_loss(
                model(**inp),
                *labels,
                gap_pos_weight=8.0,
                lambda_gap=0.6,
                lambda_realized=0.25,
                native_fa_weight=1.5,
            )["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            enforce_vram(device)
        sel, tap, gap = cal_score(
            cal_rows, infer_fn(model, cal_rows, xc, norm, device, batch_size)
        )
        hist.append(
            {
                "epoch": ep,
                "loss": float(loss_sum / n_steps),
                "cal_t_ap": tap,
                "cal_g_ap": gap,
                "sel": sel,
            }
        )
        if sel > best_sel:
            best_sel, best_ep = sel, ep
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()
    cal_logits = infer_fn(model, cal_rows, xc, norm, device, batch_size)
    policy = choose_policy(cal_rows, cal_logits, Spec())
    cal_l2, cal_l1 = _cal_score_arrays(cal_rows, cal_logits, policy)
    peak = enforce_vram(device)

    cp_path = out_dir / "control.pt"
    with cp_path.open("xb") as f:
        torch.save(
            {
                "repair_id": REPAIR_ID,
                "arm": arm,
                "seed": seed,
                "config": CFG_FIX1.as_dict() if is_fix1 else {"arm": "fixed"},
                "state_dict": best_state,
                "norm": {
                    k: tuple(torch.from_numpy(x) for x in tup) for k, tup in norm.items()
                },
                "policy": policy,
            },
            f,
        )
    card = {
        "repair_id": REPAIR_ID,
        "goal_commit": GOAL_COMMIT,
        "arm": arm,
        "seed": seed,
        "n_params": n_params,
        "best_epoch": best_ep,
        "policy": policy,
        "train_order_sha256": order_hash.hexdigest(),
        "repair_preflight_sha256": split_sha,
        "checkpoint_sha256": sha_file(cp_path),
        "peak_reserved_mib": peak,
        "wall_s": round(time.perf_counter() - t0, 2),
        "history": hist,
    }
    card_p = out_dir / "CHECKPOINT_CARD.json"
    save_json(card_p, card)

    # Single-utterance sealed Eval inference
    recs = []
    for row in ev_rows:
        h, lp = read_frames(cache, row)
        feat = (
            [_pool_no_hwin(h, lp, row["canonical_phones"])]
            if is_fix1
            else [pool_features(h, lp, row["canonical_phones"], Spec(), None)]
        )
        lg = infer_fn(model, [row], feat, norm, device, 1)[0]
        recs.extend(scores_to_rows(row, *probabilities(lg, row, policy), policy))
    seal_predictions(out_dir, recs, sha_file(card_p))
    ev, boots = evaluate_seal(out_dir, split, comparisons=comparisons)

    # Standalone independent recount parity verification
    df_merged = keyed_merge(read_predictions(out_dir), gold_df)
    ind = summarize_cohorts(df_merged)
    for c in COHORTS[:-1]:
        for k in ("tp", "fp", "fn"):
            assert ind[c]["strict"][k] == ev[c]["strict"][k], f"Recount mismatch {arm} s={seed} {c}"
    assert ind["timit_core"]["fa_total"] == ev["timit_core"]["fa_total"]

    print(
        f"[seed={seed} {arm:11s}] ep={best_ep:2d} pol={policy['tau_t']:.2f}/{policy['tau_g']:.3f} "
        f"H6={ev['held6']['strict']['f1']*100:.2f}%(tAP={ev['held6']['target_ap']:.3f}) "
        f"Sp={ev['spontaneous']['strict']['f1']*100:.2f}%(H6={ev['spontaneous']['held6_only']['strict']['f1']*100:.2f}%) "
        f"Ep={ev['epadb_test']['strict']['f1']*100:.2f}%(tAP={ev['epadb_test']['target_ap']:.3f}) "
        f"SA={ev['saa27']['strict']['f1']*100:.2f}%(tAP={ev['saa27']['target_ap']:.3f}) "
        f"TIMIT={ev['timit_core']['fa_per_utt']:.3f}({ev['timit_core']['fa_target']}+{ev['timit_core']['fa_gap']}) "
        f"Ctrl={ev[CONTROL]['recall']*100:.2f}% ({time.perf_counter()-t0:.1f}s)",
        flush=True,
    )

    del model, opt, sched
    release_cuda(device)
    return {
        "card": card,
        "eval": ev,
        "bootstrap": boots,
        "cal_l2": cal_l2,
        "cal_l1": cal_l1,
        "df_merged": df_merged,
    }


def benchmark_head_latency(
    fixed_ckpt: Path, fix1_ckpt: Path, ev_rows, cache: Path, device: torch.device
) -> dict:
    """Compare head-only + pooling GPU latency (fixed vs iso_no_hwin) on matched utterances."""
    if device.type != "cuda":
        return {}
    dfx = torch.load(fixed_ckpt, map_location="cpu", weights_only=True)
    m_fix = LengthAwareV2().to(device).eval()
    m_fix.load_state_dict(dfx["state_dict"], strict=True)
    n_fix = {k: tuple(t.numpy() for t in v) for k, v in dfx["norm"].items()}

    df1 = torch.load(fix1_ckpt, map_location="cpu", weights_only=True)
    m_f1 = SingleFactorV2(CFG_FIX1).to(device).eval()
    m_f1.load_state_dict(df1["state_dict"], strict=True)
    n_f1 = {k: tuple(t.numpy() for t in v) for k, v in df1["norm"].items()}

    out = {}
    for cohort in ("held6", "epadb_test", "saa27"):
        sub = [r for r in ev_rows if r["cohort"] == cohort]
        cnt = len(sub) if cohort == "saa27" else min(40, len(sub))
        pick = [sub[i] for i in np.linspace(0, len(sub) - 1, cnt, dtype=int)]
        preloaded = [(r, *read_frames(cache, r)) for r in pick]
        out[cohort] = {}
        for name, m, nm, is_f1 in [
            ("fixed", m_fix, n_fix, False),
            ("iso_no_hwin", m_f1, n_f1, True),
        ]:
            samples = []
            for idx, (r, h, lp) in enumerate([*preloaded[:2], *preloaded]):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                if is_f1:
                    f = [_pool_no_hwin(h, lp, r["canonical_phones"])]
                    infer_sf_logits(m, [r], f, nm, device, 1)
                else:
                    f = [pool_features(h, lp, r["canonical_phones"], Spec(), None)]
                    infer_logits(m, [r], f, nm, device, 1)
                torch.cuda.synchronize(device)
                if idx >= 2:
                    samples.append((time.perf_counter() - t0) * 1000.0)
            arr = np.asarray(samples)
            out[cohort][name] = {
                "n": len(arr),
                "p50_ms": float(np.median(arr)),
                "p95_ms": float(np.quantile(arr, 0.95)),
            }
    del m_fix, m_f1
    release_cuda(device)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--splits", type=Path, default=DEFAULT_REPAIR / "splits")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()

    ensure_separate(a.legacy_run, a.out)
    split_sha = check_inputs(a.legacy_run, a.splits)
    a.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    t_all = time.perf_counter()

    fit_rows = admitted(a.splits, "fit.jsonl")
    cal_rows = admitted(a.splits, "cal.jsonl")
    ev_rows = admitted(a.splits, "eval_prompts.jsonl")
    gold_df = gold_frame(admitted(a.splits, "eval_gold.jsonl"))
    cache = a.legacy_run / "frames"

    t0 = time.perf_counter()
    xf_fix = [pool_features(*read_frames(cache, r), r["canonical_phones"], Spec(), None) for r in fit_rows]
    xc_fix = [pool_features(*read_frames(cache, r), r["canonical_phones"], Spec(), None) for r in cal_rows]
    norm_fix = norm_from_fit(xf_fix)
    print(f"POOL fixed fit/cal ({time.perf_counter()-t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    xf_f1 = [_pool_no_hwin(*read_frames(cache, r), r["canonical_phones"]) for r in fit_rows]
    xc_f1 = [_pool_no_hwin(*read_frames(cache, r), r["canonical_phones"]) for r in cal_rows]
    norm_f1 = norm_from_fit(xf_f1)
    print(f"POOL iso_no_hwin fit/cal ({time.perf_counter()-t0:.1f}s)", flush=True)

    per_seed = {}
    for seed in SEEDS:
        r_fix = fit_and_eval_one(
            "fixed",
            seed,
            a.out / f"seed{seed}_fixed",
            a.splits,
            cache,
            fit_rows,
            cal_rows,
            ev_rows,
            xf_fix,
            xc_fix,
            norm_fix,
            gold_df,
            device,
            epochs=a.epochs,
            batch_size=a.batch_size,
            split_sha=split_sha,
            comparisons={},
        )
        r_f1 = fit_and_eval_one(
            "iso_no_hwin",
            seed,
            a.out / f"seed{seed}_iso_no_hwin",
            a.splits,
            cache,
            fit_rows,
            cal_rows,
            ev_rows,
            xf_f1,
            xc_f1,
            norm_f1,
            gold_df,
            device,
            epochs=a.epochs,
            batch_size=a.batch_size,
            split_sha=split_sha,
            comparisons={"fixed_same_seed": read_predictions(a.out / f"seed{seed}_fixed")},
        )
        assert (
            r_fix["card"]["train_order_sha256"] == r_f1["card"]["train_order_sha256"]
        ), "Batch order mismatch!"

        # Cal L2 recall-matched operating point (iso_no_hwin matching fixed's Cal L2 recall on same seed)
        pol_fix = r_fix["card"]["policy"]
        t_pos_ref = r_fix["cal_l2"]["t_sc"][np.isin(r_fix["cal_l2"]["t_g"], ["S", "O"])]
        g_pos_ref = r_fix["cal_l2"]["g_sc"][r_fix["cal_l2"]["g_g"] == "I"]
        ref_t_rec = float((t_pos_ref >= pol_fix["tau_t"]).mean())
        ref_g_rec = float((g_pos_ref >= pol_fix["tau_g"]).mean())
        rm_tt, rm_tg = match_cal_recall(r_f1["cal_l2"], ref_t_rec, ref_g_rec)
        rm_eval = summarize_cohorts(apply_policy_df(r_f1["df_merged"], rm_tt, rm_tg))

        per_seed[str(seed)] = {
            "train_order_sha256": r_fix["card"]["train_order_sha256"],
            "fixed": {
                "best_epoch": r_fix["card"]["best_epoch"],
                "policy": pol_fix,
                "checkpoint_sha256": r_fix["card"]["checkpoint_sha256"],
                "eval": r_fix["eval"],
            },
            "iso_no_hwin": {
                "best_epoch": r_f1["card"]["best_epoch"],
                "policy": r_f1["card"]["policy"],
                "checkpoint_sha256": r_f1["card"]["checkpoint_sha256"],
                "eval": r_f1["eval"],
                "bootstrap_vs_fixed": r_f1["bootstrap"]["fixed_same_seed"],
                "cal_recall_matched": {
                    "ref_cal_t_rec": ref_t_rec,
                    "ref_cal_g_rec": ref_g_rec,
                    "tau_t": round(rm_tt, 4),
                    "tau_g": round(rm_tg, 4),
                    "eval": rm_eval,
                },
            },
        }

    del xf_fix, xc_fix, xf_f1, xc_f1
    gc.collect()

    # Aggregate 3-seed summary (native & recall-matched)
    def agg_metric(getter):
        vf = [getter(per_seed[str(s)]["fixed"]["eval"]) for s in SEEDS]
        v1 = [getter(per_seed[str(s)]["iso_no_hwin"]["eval"]) for s in SEEDS]
        dl = [b - a for a, b in zip(vf, v1)]
        return {
            "fixed_seeds": vf,
            "fix1_seeds": v1,
            "fixed_mean_sd": [float(np.mean(vf)), float(np.std(vf))],
            "fix1_mean_sd": [float(np.mean(v1)), float(np.std(v1))],
            "paired_deltas": dl,
            "paired_delta_mean_sd": [float(np.mean(dl)), float(np.std(dl))],
            "wins_fix1_of_3": int(sum(x > 0 for x in dl)),
        }

    def agg_rm(getter):
        vf = [getter(per_seed[str(s)]["fixed"]["eval"]) for s in SEEDS]
        vr = [getter(per_seed[str(s)]["iso_no_hwin"]["cal_recall_matched"]["eval"]) for s in SEEDS]
        dl = [b - a for a, b in zip(vf, vr)]
        return {
            "fixed_mean_sd": [float(np.mean(vf)), float(np.std(vf))],
            "fix1_rm_mean_sd": [float(np.mean(vr)), float(np.std(vr))],
            "paired_deltas": dl,
            "paired_delta_mean": float(np.mean(dl)),
        }

    aggregate = {
        "strict_f1_pct": {
            "held6": agg_metric(lambda e: e["held6"]["strict"]["f1"] * 100.0),
            "spontaneous": agg_metric(lambda e: e["spontaneous"]["strict"]["f1"] * 100.0),
            "spontaneous_held6_only": agg_metric(
                lambda e: e["spontaneous"]["held6_only"]["strict"]["f1"] * 100.0
            ),
            "epadb_test": agg_metric(lambda e: e["epadb_test"]["strict"]["f1"] * 100.0),
            "saa27": agg_metric(lambda e: e["saa27"]["strict"]["f1"] * 100.0),
        },
        "target_ap": {
            c: agg_metric(lambda e, c=c: e[c]["target_ap"]) for c in COHORTS[:-1]
        },
        "gap_ap": {
            c: agg_metric(lambda e, c=c: e[c]["gap_ap"]) for c in COHORTS[:-1]
        },
        "i_f1_pct": {
            c: agg_metric(lambda e, c=c: e[c]["I"]["f1"] * 100.0) for c in COHORTS[:-1]
        },
        "timit_core": {
            "fa_per_utt": agg_metric(lambda e: e["timit_core"]["fa_per_utt"]),
            "fa_target": agg_metric(lambda e: e["timit_core"]["fa_target"]),
            "fa_gap": agg_metric(lambda e: e["timit_core"]["fa_gap"]),
            "control_recall_pct": agg_metric(lambda e: e[CONTROL]["recall"] * 100.0),
        },
        "cal_recall_matched": {
            "held6_f1_pct": agg_rm(lambda e: e["held6"]["strict"]["f1"] * 100.0),
            "spont_unseen_f1_pct": agg_rm(
                lambda e: e["spontaneous"]["held6_only"]["strict"]["f1"] * 100.0
            ),
            "epadb_f1_pct": agg_rm(lambda e: e["epadb_test"]["strict"]["f1"] * 100.0),
            "saa27_f1_pct": agg_rm(lambda e: e["saa27"]["strict"]["f1"] * 100.0),
            "timit_fa_per_utt": agg_rm(lambda e: e["timit_core"]["fa_per_utt"]),
            "timit_fa_target": agg_rm(lambda e: e["timit_core"]["fa_target"]),
            "timit_fa_gap": agg_rm(lambda e: e["timit_core"]["fa_gap"]),
        },
    }

    latency = benchmark_head_latency(
        a.out / f"seed{SEEDS[0]}_fixed/control.pt",
        a.out / f"seed{SEEDS[0]}_iso_no_hwin/control.pt",
        ev_rows,
        cache,
        device,
    )

    result = {
        "scope": "paired_3seed_confirmation_iso_no_hwin_vs_fixed_on_repaired_v2_splits",
        "independent_recount_verified": True,
        "repair_preflight_sha256": split_sha,
        "seeds": list(SEEDS),
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "aggregate_3seed": aggregate,
        "head_latency_ms": latency,
        "per_seed": per_seed,
        "wall_s": round(time.perf_counter() - t_all, 2),
    }
    save_json(a.out / "CONFIRM_3SEED_RESULTS.json", result)
    print(f"ALL 3-SEED CONFIRMATION DONE ({result['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
