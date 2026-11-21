"""Single-factor isolation runner on GPU (GTX 1650) with Phase-1 Joint + Phase-2 Frozen Gap.

Runs the 3 single-factor leave-one-out arms relative to `fix12_arch`:
  - iso_restore_cmvn : restore utterance CMVN on target only (keep independent gap_h_norm)
  - iso_no_clip_lp   : LayerNorm(79) on raw lp instead of [exp(lp), clip(lp,-10,0)/10]
  - iso_no_hwin      : drop 5-frame h_win_attn (use h_vit + h_peak), isolating Fix #1 vs Fix #2

For each arm:
  1. Phase 1: Train 14 epochs (same seed 20261121, batch 16, exact train_order_sha256 as fixed/fix12_arch,
              same standard V2 weighted CE loss, no modality dropout, select best_epoch on Cal).
  2. Phase 2: Freeze 100% of encoder + target head (0.0 diff verified), re-init gap branch with
              seed 20261121, train gap head for 14 epochs (keep TIMIT Gap 'I' anti-shortcut),
              select best gap epoch by cal_gap_ap.
  3. Lock thresholds strictly on Calibrate, seal Eval predictions in Parquet, verify exact
     target parity between Phase 1 and Phase 2, recount with standalone `independent_count`,
     and bootstrap against `fixed` and `fix12_arch`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import compute_multitask_mdd_loss
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
from .replay_codefix_audit import HELD6_SPEAKERS, independent_count
from .run_gap_mask_ablation import (
    cal_gap_arrays,
    compute_isolated_gap_loss,
    is_gap_param,
    select_cal_gap_policies,
    split_param_shas,
)
from .run_repaired_control import check_inputs
from .scoring import (
    brief,
    evaluate_seal,
    gold_frame,
    keyed_merge,
    read_predictions,
    scores_to_rows,
    seal_predictions,
)
from .single_factor_models import (
    FACTOR_ARMS,
    SingleFactorConfig,
    SingleFactorV2,
    pool_features_with_hvit,
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

DEFAULT_CODEFIX = DEFAULT_REPAIR.parent / "code_fixes_123_20261121_v1"
GAP_KEYS = {"h_gap", "lp_gap_mean", "lp_gap_max", "gap_scalars"}


def collate_sf(rows, feats, stats, device, *, labels: bool):
    lengths = [len(r["canonical_phones"]) for r in rows]
    max_l = max(lengths)
    args = {}
    for name, arr in feats[0].items():
        extra = 1 if name in GAP_KEYS else 0
        if name.endswith("phone_ids"):
            dtype = np.int64
        elif name in {"h_win", "h_peak", "h_vit", "h_gap"}:
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
    is_l1 = np.array([row.get("is_native_l1", False) for row in rows], dtype=bool)
    return args, tuple(torch.from_numpy(a).to(device) for a in (t, g, r_lbl, is_l1))


@torch.inference_mode()
def infer_sf_logits(model, rows, feats, norm, device, batch_size):
    out = []
    model.eval()
    for i in range(0, len(rows), batch_size):
        rsub = rows[i : i + batch_size]
        inp, _ = collate_sf(rsub, feats[i : i + batch_size], norm, device, labels=False)
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


@torch.inference_mode()
def precompute_sf_seq(model, rows, feats, norm, device, batch_size=16):
    model.eval()
    seqs = []
    for i in range(0, len(rows), batch_size):
        rsub = rows[i : i + batch_size]
        inp, _ = collate_sf(rsub, feats[i : i + batch_size], norm, device, labels=False)
        seq = model.encode_sequence(
            h_peak=inp["h_peak"],
            h_win=inp["h_win"],
            h_vit=inp["h_vit"],
            lp_win=inp["lp_win"],
            lp_vit=inp["lp_vit"],
            phone_ids=inp["phone_ids"],
            prev_phone_ids=inp["prev_phone_ids"],
            next_phone_ids=inp["next_phone_ids"],
            target_scalars=inp["target_scalars"],
            lengths=inp["lengths"],
        ).cpu().numpy()
        for j, r in enumerate(rsub):
            seqs.append(seq[j, : len(r["canonical_phones"])].astype(np.float32))
    return seqs


def collate_sf_gap(rows, feats, seqs, g_mean, g_std, device, *, labels: bool):
    lengths = [len(r["canonical_phones"]) for r in rows]
    max_l = max(lengths)
    B = len(rows)
    d_model = seqs[0].shape[-1]
    seq_t = np.zeros((B, max_l, d_model), dtype=np.float32)
    h_gap = np.zeros((B, max_l + 1, 6, 1024), dtype=np.float16)
    lp_gm = np.zeros((B, max_l + 1, 79), dtype=np.float32)
    lp_gx = np.zeros((B, max_l + 1, 79), dtype=np.float32)
    g_sc = np.zeros((B, max_l + 1, 8), dtype=np.float32)
    for i, (f, sq, L) in enumerate(zip(feats, seqs, lengths, strict=True)):
        seq_t[i, :L] = sq
        h_gap[i, : L + 1] = f["h_gap"]
        lp_gm[i, : L + 1] = f["lp_gap_mean"]
        lp_gx[i, : L + 1] = f["lp_gap_max"]
        g_sc[i, : L + 1] = (f["gap_scalars"] - g_mean) / g_std
    inp = {
        "seq": torch.from_numpy(seq_t).to(device),
        "h_gap": torch.from_numpy(h_gap).to(device),
        "lp_gap_mean": torch.from_numpy(lp_gm).to(device),
        "lp_gap_max": torch.from_numpy(lp_gx).to(device),
        "gap_scalars": torch.from_numpy(g_sc).to(device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }
    if not labels:
        return inp, None, None
    y_g = np.full((B, max_l + 1), -100, dtype=np.int64)
    for i, (r, L) in enumerate(zip(rows, lengths, strict=True)):
        y_g[i, : L + 1] = [GAP_LABEL_TO_ID.get(a, -100) for a in r["gold_gaps"]]
    is_l1 = np.array([domain(r) == "l1" for r in rows], dtype=bool)
    return inp, torch.from_numpy(y_g).to(device), torch.from_numpy(is_l1).to(device)


@torch.inference_mode()
def predict_sf_gap(model, rows, feats, seqs, g_mean, g_std, device, batch_size=16):
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        rsub = rows[i : i + batch_size]
        inp, _, _ = collate_sf_gap(
            rsub, feats[i : i + batch_size], seqs[i : i + batch_size], g_mean, g_std, device, labels=False
        )
        gp = F.softmax(model.forward_gap_from_seq(**inp), dim=-1)[..., 1].cpu().numpy()
        for j, r in enumerate(rsub):
            out.append(gp[j, : len(r["canonical_phones"]) + 1])
    return out


def reset_sf_gap(model: SingleFactorV2, seed: int) -> None:
    torch.manual_seed(seed)
    fresh = SingleFactorV2(model.cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if is_gap_param(name):
                param.copy_(fresh.state_dict()[name])
    for name, param in model.named_parameters():
        param.requires_grad_(is_gap_param(name))


def train_and_eval_arm(
    arm_name: str,
    cfg: SingleFactorConfig,
    run: Path,
    split: Path,
    out: Path,
    fit_rows,
    cal_rows,
    ev_rows,
    xf,
    xc,
    norm,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    split_sha: str,
    comparisons: dict,
    gold_df,
):
    arm_dir = out / arm_name
    arm_dir.mkdir(parents=True, exist_ok=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    # --- Phase 1: Joint training (exact same recipe as fix12_arch / fixed) ---
    torch.manual_seed(seed)
    model = SingleFactorV2(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    n_steps = math.ceil(len(fit_rows) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * n_steps, eta_min=2e-5
    )
    rng = np.random.default_rng(seed)
    order_hash = hashlib.sha256()
    hist_p1, best_sel, best_p1_ep, best_p1_state = [], -math.inf, 0, None

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fit_rows))
        order_hash.update(order.astype("<i8").tobytes())
        tot = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            inp, labels = collate_sf(
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
            tot += loss.detach()
            enforce_vram(device)
        sel, tap, gap = cal_score(
            cal_rows, infer_sf_logits(model, cal_rows, xc, norm, device, batch_size)
        )
        hist_p1.append(
            {
                "epoch": ep,
                "loss": float(tot / n_steps),
                "cal_t_ap": tap,
                "cal_g_ap": gap,
                "sel": sel,
            }
        )
        if sel > best_sel:
            best_sel, best_p1_ep = sel, ep
            best_p1_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_p1_state)
    model.eval()
    policy_p1 = choose_policy(
        cal_rows, infer_sf_logits(model, cal_rows, xc, norm, device, batch_size), Spec()
    )
    frozen_sha_p1, _ = split_param_shas(best_p1_state)
    print(
        f"[{arm_name}] Phase1 best_ep={best_p1_ep} cal_tAP={hist_p1[best_p1_ep-1]['cal_t_ap']:.4f} "
        f"cal_gAP={hist_p1[best_p1_ep-1]['cal_g_ap']:.4f} pol={policy_p1['tau_t']}/{policy_p1['tau_g']} "
        f"({time.perf_counter()-t0:.1f}s)",
        flush=True,
    )

    # --- Phase 2: Frozen encoder/target + isolated Gap head (keep TIMIT Gap 'I') ---
    peak_p1 = enforce_vram(device)
    del opt, sched
    release_cuda(device)
    seq_f = precompute_sf_seq(model, fit_rows, xf, norm, device, batch_size=batch_size)
    seq_c = precompute_sf_seq(model, cal_rows, xc, norm, device, batch_size=batch_size)
    del model
    release_cuda(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    g_mean, g_std = norm["gap_scalars"]

    model_p2 = SingleFactorV2(cfg)
    model_p2.load_state_dict(best_p1_state, strict=True)
    reset_sf_gap(model_p2, seed)
    model_p2 = model_p2.to(device)
    gap_params = [p for n, p in model_p2.named_parameters() if is_gap_param(n)]
    opt2 = torch.optim.AdamW(gap_params, lr=8e-4, weight_decay=0.02)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=epochs * n_steps, eta_min=2e-5
    )
    rng2 = np.random.default_rng(seed)
    best_gap_ap, best_p2_ep, best_p2_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        model_p2.train()
        order = rng2.permutation(len(fit_rows))
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            inp, y_g, is_l1 = collate_sf_gap(
                [fit_rows[i] for i in ix],
                [xf[i] for i in ix],
                [seq_f[i] for i in ix],
                g_mean,
                g_std,
                device,
                labels=True,
            )
            opt2.zero_grad(set_to_none=True)
            loss = compute_isolated_gap_loss(
                model_p2.forward_gap_from_seq(**inp),
                y_g,
                is_l1,
                mask_timit_gap_i=False,
                gap_pos_weight=8.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gap_params, 1.0, error_if_nonfinite=True)
            opt2.step()
            sched2.step()
            enforce_vram(device)
        cprobs = predict_sf_gap(model_p2, cal_rows, xc, seq_c, g_mean, g_std, device, batch_size)
        cy, cp, _ = cal_gap_arrays(cal_rows, cprobs)
        cap = float(average_precision_score(cy, cp))
        if cap > best_gap_ap:
            best_gap_ap, best_p2_ep = cap, ep
            best_p2_state = {
                k: v.detach().cpu().clone() for k, v in model_p2.state_dict().items()
            }

    frozen_sha_p2, _ = split_param_shas(best_p2_state)
    assert frozen_sha_p2 == frozen_sha_p1, "Frozen weights changed in Phase 2!"
    model_p2.load_state_dict(best_p2_state)
    model_p2.eval()
    cprobs = predict_sf_gap(model_p2, cal_rows, xc, seq_c, g_mean, g_std, device, batch_size)
    cy, cp, cl1 = cal_gap_arrays(cal_rows, cprobs)
    gap_pols = select_cal_gap_policies(cy, cp, cl1, l2_fp_cap=41)
    policy_p2 = {**policy_p1, "tau_g": gap_pols["cal_f1"]["tau_g"]}
    peak_p2 = enforce_vram(device)
    peak = max(peak_p1, peak_p2)
    print(
        f"[{arm_name}] Phase2 best_ep={best_p2_ep} cal_gAP={best_gap_ap:.4f} "
        f"pol={policy_p2['tau_t']}/{policy_p2['tau_g']} peak={peak:.1f} MiB (p1={peak_p1:.1f}, p2={peak_p2:.1f})",
        flush=True,
    )

    # Save both Phase 1 (joint) and Phase 2 (frozen-target + isolated gap) checkpoints & eval
    cache = run / "frames"
    arm_out = {"config": cfg.as_dict(), "n_params": n_params, "train_order_sha256": order_hash.hexdigest()}
    for tag, st, pol, bep in [
        ("phase1_joint", best_p1_state, policy_p1, best_p1_ep),
        ("phase2_2stage", best_p2_state, policy_p2, best_p2_ep),
    ]:
        sub = arm_dir / tag
        sub.mkdir(parents=True, exist_ok=False)
        cp_path = sub / "control.pt"
        with cp_path.open("xb") as f:
            torch.save(
                {
                    "repair_id": REPAIR_ID,
                    "config": cfg.as_dict(),
                    "phase": tag,
                    "state_dict": st,
                    "norm": {k: tuple(torch.from_numpy(x) for x in tup) for k, tup in norm.items()},
                    "policy": pol,
                },
                f,
            )
        card = {
            "repair_id": REPAIR_ID,
            "goal_commit": GOAL_COMMIT,
            "arm": arm_name,
            "phase": tag,
            "config": cfg.as_dict(),
            "n_params": n_params,
            "best_epoch": bep,
            "policy": pol,
            "frozen_params_sha256": frozen_sha_p1,
            "train_order_sha256": order_hash.hexdigest(),
            "repair_preflight_sha256": split_sha,
            "checkpoint_sha256": sha_file(cp_path),
            "peak_reserved_mib": peak,
        }
        card_p = sub / "CHECKPOINT_CARD.json"
        save_json(card_p, card)

        ev_model = SingleFactorV2(cfg)
        ev_model.load_state_dict(st, strict=True)
        ev_model = ev_model.to(device).eval()
        recs = []
        for row in ev_rows:
            feat = [pool_features_with_hvit(*read_frames(cache, row), row["canonical_phones"])]
            lg = infer_sf_logits(ev_model, [row], feat, norm, device, 1)[0]
            recs.extend(scores_to_rows(row, *probabilities(lg, row, pol), pol))
        seal_predictions(sub, recs, sha_file(card_p))
        ev, boots = evaluate_seal(sub, split, comparisons=comparisons)

        # Standalone independent recount parity check
        df_m = keyed_merge(read_predictions(sub), gold_df)
        for c in COHORTS[:-1]:
            g = df_m[df_m["cohort"] == c]
            gt = g[g["domain"] == "target"]
            gg = g[g["domain"] == "gap"]
            cnt = independent_count(
                gt["gold_label"].to_numpy(),
                gt["call"].to_numpy(),
                gg["gold_label"].to_numpy(),
                gg["call"].to_numpy(),
            )
            for k in ("tp", "fp", "fn"):
                assert cnt["strict"][k] == ev[c]["strict"][k], f"Recount mismatch {arm_name}/{tag}/{c}"
            if c == "spontaneous":
                gu = g[g["speaker"].isin(HELD6_SPEAKERS)]
                gtu = gu[gu["domain"] == "target"]
                ggu = gu[gu["domain"] == "gap"]
                ev[c]["held6_only_independent"] = independent_count(
                    gtu["gold_label"].to_numpy(),
                    gtu["call"].to_numpy(),
                    ggu["gold_label"].to_numpy(),
                    ggu["call"].to_numpy(),
                )
        save_json(sub / "RESULTS.json", {"eval": ev, "bootstrap": boots})
        arm_out[tag] = {
            "best_epoch": bep,
            "policy": pol,
            "brief": brief(ev),
            "eval": ev,
            "bootstrap": boots,
        }
        print(f"  [{arm_name}/{tag}] {brief(ev)}", flush=True)
        del ev_model
        release_cuda(device)

    # Verify exact Target score parity between Phase 1 and Phase 2 on all 2,466 Eval utterances!
    p1_t = read_predictions(arm_dir / "phase1_joint")
    p2_t = read_predictions(arm_dir / "phase2_2stage")
    p1_t = p1_t[p1_t["domain"] == "target"].sort_values(["cohort", "uid", "index"])
    p2_t = p2_t[p2_t["domain"] == "target"].sort_values(["cohort", "uid", "index"])
    max_diff = float(np.max(np.abs(p1_t["score"].to_numpy() - p2_t["score"].to_numpy())))
    assert max_diff < 1e-6, f"Phase 1/2 target drift: {max_diff}"
    arm_out["phase1_vs_phase2_target_max_abs_diff"] = max_diff
    arm_out["wall_s"] = round(time.perf_counter() - t0, 2)

    del model_p2, opt2, sched2, seq_f, seq_c
    release_cuda(device)
    return arm_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--splits", type=Path, default=DEFAULT_REPAIR / "splits")
    p.add_argument(
        "--fixed-eval",
        type=Path,
        default=DEFAULT_REPAIR / "matched_seed20261121_e14_b16/eval_fixed",
    )
    p.add_argument(
        "--fix12-eval",
        type=Path,
        default=DEFAULT_CODEFIX / "eval_fix12_arch",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=SEED)
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
    xf = [pool_features_with_hvit(*read_frames(cache, r), r["canonical_phones"]) for r in fit_rows]
    print(f"POOL fit {len(xf)} ({time.perf_counter()-t0:.1f}s)", flush=True)
    t0 = time.perf_counter()
    xc = [pool_features_with_hvit(*read_frames(cache, r), r["canonical_phones"]) for r in cal_rows]
    print(f"POOL cal {len(xc)} ({time.perf_counter()-t0:.1f}s)", flush=True)
    norm = norm_from_fit(xf)

    fixed_card = read_json(a.fixed_eval / "CHECKPOINT_CARD.json")
    comparisons = {
        "fixed": read_predictions(a.fixed_eval),
        "fix12_arch": read_predictions(a.fix12_eval),
    }

    results = {}
    for arm_name, cfg in FACTOR_ARMS.items():
        res = train_and_eval_arm(
            arm_name,
            cfg,
            a.legacy_run,
            a.splits,
            a.out,
            fit_rows,
            cal_rows,
            ev_rows,
            xf,
            xc,
            norm,
            device,
            seed=a.seed,
            epochs=a.epochs,
            batch_size=a.batch_size,
            split_sha=split_sha,
            comparisons=comparisons,
            gold_df=gold_df,
        )
        assert res["train_order_sha256"] == fixed_card["train_order_sha256"]
        results[arm_name] = res

    del xf, xc
    gc.collect()

    fixed_ev, _ = evaluate_seal(a.fixed_eval, a.splits)
    fix12_ev, _ = evaluate_seal(a.fix12_eval, a.splits)

    summary = {
        "scope": "single_factor_isolation_of_fix12_arch_phase1_and_phase2",
        "repair_preflight_sha256": split_sha,
        "train_order_sha256": fixed_card["train_order_sha256"],
        "references": {
            "fixed": brief(fixed_ev),
            "fix12_arch": brief(fix12_ev),
        },
        "arms": results,
        "wall_s": round(time.perf_counter() - t_all, 2),
    }
    save_json(a.out / "SINGLE_FACTOR_RESULTS.json", summary)
    print(f"ALL DONE ({summary['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
