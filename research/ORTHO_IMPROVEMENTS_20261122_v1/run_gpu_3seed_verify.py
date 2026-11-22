"""Task 2: Paired 3-seed GPU verification of FIT_ONLY_PATCH (F1) and O3_cond_so (0.25 / 0.50).

Strictly paired on clean_11 (3,681 Fit / 635 Cal / 2,446 Eval prompts; test_sealed untouched):
  - F0_control    : exact SingleFactorV2 (iso_no_hwin) baseline on clean_11 (3 seeds)
  - F1_fit_patch  : ONLY applies audited FIT_ONLY_PATCH.json (197 Fit realized_id DP fixes)
  - O3_cond_so_025: ONLY sets lambda_so_cond = 0.25 (unmodified Fit)
  - O3_cond_so_050: ONLY sets lambda_so_cond = 0.50 (unmodified Fit)

Every seed shares exact init_sha256 and train_order_sha256 across all 4 arms.
Epoch selection, single-seed thresholds, and 3-seed O5+O4 ensemble policies are locked
STRICTLY on Cal. All Eval predictions are sealed with SHA-256 BEFORE reading eval_gold.jsonl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch

from research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1.supervision.fit_only import load_fit
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import (
    BASE_KOEL_CACHE,
    load_splits_no_eval_gold,
    pool_koel_iso,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import CONTINUOUS_SCALAR_NAMES
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import save_json, sha_file
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import Spec
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import (
    collate_sf,
    infer_sf_logits,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import (
    scores_to_rows,
    seal_predictions,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.single_factor_models import (
    FACTOR_ARMS,
    SingleFactorV2,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import (
    cal_score,
    choose_policy,
    domain,
    norm_from_fit,
    probabilities,
)

from .calibration import (
    apply_ortho_policy,
    calibrate_ortho_policy,
    ensemble_predictions,
)
from .losses import compute_ortho_mdd_loss
from .package_s0_clean_ensemble import _build_eval_records

FIT_PATCH_PATH = Path(
    "research/BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1/supervision/FIT_ONLY_PATCH.json"
)
EVAL_GOLD_PATH = Path(
    "research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/v2_prompt_eos_repair_v1/splits/eval_gold.jsonl"
)
CAN_LP_IDX = CONTINUOUS_SCALAR_NAMES.index("can_lp_vit_max")
SEEDS: Tuple[int, ...] = (20261121, 20261122, 20261123)
COHORTS: Tuple[str, ...] = (
    "held6",
    "epadb_test",
    "saa27",
    "spontaneous",
    "timit_core",
    "timit_core_corrupt_ctrl",
)
HELD6_SPEAKERS = {"ABA", "ASI", "BWC", "EBVS", "HQTV", "LXC"}
VALID_GOLD = {"C", "S", "O", "N", "I"}


@dataclass(frozen=True)
class VerifyArm:
    name: str
    use_fit_patch: bool = False
    lambda_so_cond: float = 0.0


VERIFY_ARMS: Tuple[VerifyArm, ...] = (
    VerifyArm("F0_control"),
    VerifyArm("F1_fit_patch", use_fit_patch=True),
    VerifyArm("O3_cond_so_025", lambda_so_cond=0.25),
    VerifyArm("O3_cond_so_050", lambda_so_cond=0.50),
)


def state_sha(state_dict: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode("utf-8"))
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _load_gold_df() -> pd.DataFrame:
    rows = []
    with EVAL_GOLD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c, uid = r["cohort"], r["uid"]
            for i, g in enumerate(r["gold_targets"]):
                rows.append((c, uid, "target", i, g))
            for i, g in enumerate(r["gold_gaps"]):
                rows.append((c, uid, "gap", i, g))
    return pd.DataFrame(rows, columns=["cohort", "uid", "domain", "index", "gold"])


def _cal_metrics(cal_rows, cal_logits, policy, xc) -> Dict[str, float]:
    sel, tap, gap = cal_score(cal_rows, cal_logits)
    bs_y, bs_s = [], []
    for r, lg, kf in zip(cal_rows, cal_logits, xc, strict=True):
        if domain(r) == "l1":
            continue
        tp, _ = probabilities(lg, r, policy)
        err_sc = tp[:, 1] + tp[:, 2]
        can_lp = kf["target_scalars"][:, CAN_LP_IDX]
        for i, lbl in enumerate(r["gold_targets"]):
            if lbl in ("C", "S", "O") and can_lp[i] > -0.2:
                bs_y.append(0 if lbl == "C" else 1)
                bs_s.append(float(err_sc[i]))
    bs_ap = float(average_precision_score(bs_y, bs_s)) if bs_y and len(set(bs_y)) > 1 else 0.0
    return {
        "cal_sel": round(float(sel), 5),
        "cal_t_ap": round(float(tap), 5),
        "cal_g_ap": round(float(gap), 5),
        "cal_blind_ap": round(bs_ap, 5),
        "cal_l2_strict_f1": round(float(policy["cal_l2_strict_f1"]), 2),
        "cal_l1_fa_per_1k": round(float(policy["cal_l1_fa_per_1k"]), 2),
        "tau_t": float(policy["tau_t"]),
        "tau_g": float(policy["tau_g"]),
    }


def _cal_l2_target_recall(cal_rows, cal_logits, policy) -> float:
    hit, tot = 0, 0
    for r, lg in zip(cal_rows, cal_logits, strict=True):
        if domain(r) == "l1":
            continue
        tp, _ = probabilities(lg, r, policy)
        sc = tp[:, 1] + tp[:, 2]
        for i, g in enumerate(r["gold_targets"]):
            if g in ("S", "O"):
                tot += 1
                if sc[i] >= float(policy["tau_t"]):
                    hit += 1
    return float(hit / max(1, tot))


def _match_tau_t_on_cal(cal_rows, cal_logits, target_recall: float, policy) -> float:
    pos_scores = []
    for r, lg in zip(cal_rows, cal_logits, strict=True):
        if domain(r) == "l1":
            continue
        tp, _ = probabilities(lg, r, policy)
        sc = tp[:, 1] + tp[:, 2]
        for i, g in enumerate(r["gold_targets"]):
            if g in ("S", "O"):
                pos_scores.append(float(sc[i]))
    arr = np.sort(np.asarray(pos_scores, dtype=np.float64))[::-1]
    idx = min(len(arr) - 1, max(0, int(round(target_recall * len(arr))) - 1))
    return float(arr[idx])


def _score_sealed(
    pred_dir: Path,
    gold_df: pd.DataFrame,
    can_lp_map: Mapping[Tuple[str, str, int], float],
    *,
    override_tau_t: float | None = None,
) -> Dict[str, Any]:
    parts = [pd.read_parquet(pred_dir / f"{c}.parquet") for c in COHORTS]
    df = pd.concat(parts, ignore_index=True).merge(
        gold_df, on=["cohort", "uid", "domain", "index"], how="inner"
    )
    if override_tau_t is not None:
        is_t = df["domain"].values == "target"
        call = df["call"].values.copy()
        sc = df["score"].values
        p_s = df["p_s"].values
        p_o = df["p_o"].values
        call[is_t] = np.where(
            sc[is_t] >= override_tau_t,
            np.where(p_s[is_t] >= p_o[is_t], "S", "O"),
            "C",
        )
        df = df.assign(call=call)

    out: Dict[str, Any] = {}
    for name in ("held6", "epadb_test", "saa27", "spont_h6", "timit_core", "ctrl"):
        if name == "spont_h6":
            sub = df[
                (df["cohort"] == "spontaneous")
                & (df["speaker"].str.upper().isin(HELD6_SPEAKERS))
                & (df["gold"].isin(VALID_GOLD))
            ]
        elif name == "ctrl":
            sub = df[(df["cohort"] == "timit_core_corrupt_ctrl") & (df["gold"].isin(VALID_GOLD))]
            ge = (sub["gold"] != "C") & (sub["gold"] != "N")
            pe = (sub["call"] != "C") & (sub["call"] != "N")
            out["ctrl_rec"] = round(100.0 * float(pe[ge].mean()), 2)
            continue
        elif name == "timit_core":
            sub = df[df["cohort"] == "timit_core"]
            t_fa = int(((sub["domain"] == "target") & (sub["call"] != "C")).sum())
            g_fa = int(((sub["domain"] == "gap") & (sub["call"] != "N")).sum())
            out["timit_core"] = {
                "fa_per_utt": round((t_fa + g_fa) / 192.0, 3),
                "target_fa": t_fa,
                "gap_fa": g_fa,
            }
            continue
        else:
            sub = df[(df["cohort"] == name) & (df["gold"].isin(VALID_GOLD))]

        g, p = sub["gold"].values, sub["call"].values
        ge = (g != "C") & (g != "N")
        pe = (p != "C") & (p != "N")
        tp = int(np.sum(ge & (p == g)))
        fp = int(np.sum(pe & (p != g)))
        fn = int(np.sum(ge & (p != g)))
        f1 = 200.0 * tp / max(1, 2 * tp + fp + fn)
        s2o = int(np.sum((g == "S") & (p == "O")))
        o2s = int(np.sum((g == "O") & (p == "S")))
        ts = sub[sub["domain"] == "target"]
        gs = sub[sub["domain"] == "gap"]
        tap = float(average_precision_score((ts["gold"].values != "C").astype(int), ts["score"].values))
        gap = float(average_precision_score((gs["gold"].values == "I").astype(int), gs["score"].values))

        # Blind-spot AP (can_lp_vit_max > -0.2)
        clps = np.array([can_lp_map[(c, u, int(i))] for c, u, i in zip(ts["cohort"], ts["uid"], ts["index"])])
        bm = clps > -0.2
        by = (ts["gold"].values[bm] != "C").astype(int)
        bap = float(average_precision_score(by, ts["score"].values[bm])) if bm.sum() > 0 and len(set(by)) > 1 else 0.0

        # I F1
        ig, ip = (gs["gold"].values == "I"), (gs["call"].values == "I")
        itp, ifp, ifn = int((ig & ip).sum()), int(((~ig) & ip).sum()), int((ig & (~ip)).sum())
        if1 = 200.0 * itp / max(1, 2 * itp + ifp + ifn)

        out[name] = {
            "strict_f1": round(f1, 2),
            "target_ap": round(tap, 4),
            "gap_ap": round(gap, 4),
            "blind_ap": round(bap, 4),
            "I_f1": round(if1, 2),
            "so_swap": s2o + o2s,
            "s_to_o": s2o,
            "o_to_s": o2s,
        }
    return out


def run_verify(
    out_dir: Path,
    *,
    device_str: str = "cuda",
    epochs: int = 14,
    batch_size: int = 16,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[VerifyArm] = VERIFY_ARMS,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    t_all = time.perf_counter()

    print(f"=== TASK 2: PAIRED 3-SEED GPU VERIFICATION | clean_11 | device={dev} | epochs={epochs} ===", flush=True)
    fit_base, cal_rows, eval_prompts = load_splits_no_eval_gold("clean_11")
    fit_patched = load_fit(FIT_PATCH_PATH)

    # Confirm exact 197 realized_id edits & 0 other changes
    n_diff_slots = 0
    for a, b in zip(fit_base, fit_patched, strict=True):
        assert a["uid"] == b["uid"] and a["canonical_phones"] == b["canonical_phones"]
        assert a["gold_targets"] == b["gold_targets"] and a["gold_gaps"] == b["gold_gaps"]
        n_diff_slots += sum(x != y for x, y in zip(a["realized_ids"], b["realized_ids"], strict=True))
    assert n_diff_slots == 197, f"Expected 197 realized_id edits, got {n_diff_slots}"
    print(f"[FitPatch] Verified exact {n_diff_slots} realized_id edits via fit_only.load_fit (0 C/S/O/I changes)", flush=True)

    # Pool Koel features ONCE (100% identical between F0 and F1 since audio & canonical_phones are identical!)
    t_pool = time.perf_counter()
    xf = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in fit_base]
    xc = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in cal_rows]
    xe = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in eval_prompts]
    norm = norm_from_fit(xf)
    print(f"[Pool] Fit({len(fit_base)}) + Cal({len(cal_rows)}) + Eval({len(eval_prompts)}) in {time.perf_counter()-t_pool:.1f}s", flush=True)

    can_lp_map = {}
    for r, kf in zip(eval_prompts, xe, strict=True):
        clps = kf["target_scalars"][:, CAN_LP_IDX]
        for i, v in enumerate(clps):
            can_lp_map[(r["cohort"], r["uid"], i)] = float(v)

    l2_cal_idx = [i for i, r in enumerate(cal_rows) if domain(r) != "l1"]
    l1_cal_idx = [i for i, r in enumerate(cal_rows) if domain(r) == "l1"]
    l2_t_gold = np.concatenate([np.array(cal_rows[i]["gold_targets"]) for i in l2_cal_idx])
    l2_g_gold = np.concatenate([np.array(cal_rows[i]["gold_gaps"]) for i in l2_cal_idx])

    cfg = FACTOR_ARMS["iso_no_hwin"]
    # Pre-create identical initial weights per seed
    init_states = {}
    init_shas = {}
    for seed in seeds:
        torch.manual_seed(seed)
        m = SingleFactorV2(cfg)
        init_states[seed] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        init_shas[seed] = state_sha(init_states[seed])

    arm_seed_cal_logits: Dict[str, Dict[int, Any]] = {a.name: {} for a in arms}
    arm_seed_eval_logits: Dict[str, Dict[int, Any]] = {a.name: {} for a in arms}
    arm_seed_policies: Dict[str, Dict[int, Any]] = {a.name: {} for a in arms}
    arm_seed_meta: Dict[str, Dict[int, Any]] = {a.name: {} for a in arms}

    for arm in arms:
        fit_rows = fit_patched if arm.use_fit_patch else fit_base
        for seed in seeds:
            s_dir = out_dir / arm.name / f"seed_{seed}"
            s_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()

            if (s_dir / "model.pt").exists() and (s_dir / "PREDICTIONS_SEAL.json").exists() and (s_dir / "CHECKPOINT_CARD.json").exists():
                ckpt = torch.load(s_dir / "model.pt", map_location="cpu", weights_only=False)
                card = json.loads((s_dir / "CHECKPOINT_CARD.json").read_text(encoding="utf-8"))
                model = SingleFactorV2(cfg).to(dev)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                cal_lg = infer_sf_logits(model, cal_rows, xc, norm, dev, batch_size=32)
                ev_lg = infer_sf_logits(model, eval_prompts, xe, norm, dev, batch_size=32)
                arm_seed_cal_logits[arm.name][seed] = cal_lg
                arm_seed_eval_logits[arm.name][seed] = ev_lg
                arm_seed_policies[arm.name][seed] = ckpt["policy"]
                arm_seed_meta[arm.name][seed] = {**card, "pred_dir": str(s_dir / "eval_predictions"), "resumed": True}
                print(f"  [{arm.name:14s} seed={seed}] RESUMED ep={card['best_epoch']} Cal F1={card['cal']['cal_l2_strict_f1']:.2f}%", flush=True)
                continue

            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)

            torch.manual_seed(seed)
            model = SingleFactorV2(cfg).to(dev)
            model.load_state_dict(init_states[seed])
            opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
            n_steps = max(1, math.ceil(len(fit_rows) / batch_size))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs * n_steps, eta_min=2e-5
            )
            rng = np.random.default_rng(seed)
            order_hash = hashlib.sha256()

            hist, best_sel, best_ep, best_state = [], -math.inf, 0, None
            for ep in range(1, epochs + 1):
                model.train()
                order = rng.permutation(len(fit_rows))
                order_hash.update(order.astype("<i8").tobytes())
                loss_sum = torch.zeros((), device=dev)
                for start in range(0, len(order), batch_size):
                    ix = order[start : start + batch_size]
                    inp, labels = collate_sf(
                        [fit_rows[i] for i in ix], [xf[i] for i in ix], norm, dev, labels=True
                    )
                    opt.zero_grad(set_to_none=True)
                    loss = compute_ortho_mdd_loss(
                        model(**inp),
                        *labels,
                        lambda_so_cond=arm.lambda_so_cond,
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

                sel, tap, gap = cal_score(
                    cal_rows, infer_sf_logits(model, cal_rows, xc, norm, dev, batch_size=32)
                )
                hist.append({
                    "epoch": ep,
                    "loss": round(float(loss_sum / n_steps), 5),
                    "cal_t_ap": round(float(tap), 5),
                    "cal_g_ap": round(float(gap), 5),
                    "sel": round(float(sel), 5),
                })
                if sel > best_sel:
                    best_sel, best_ep = sel, ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()
            cal_lg = infer_sf_logits(model, cal_rows, xc, norm, dev, batch_size=32)
            policy = choose_policy(cal_rows, cal_lg, Spec())
            cal_m = _cal_metrics(cal_rows, cal_lg, policy, xc)
            ev_lg = infer_sf_logits(model, eval_prompts, xe, norm, dev, batch_size=32)
            peak_mib = round(torch.cuda.max_memory_reserved(dev) / (1024**2), 1) if dev.type == "cuda" else 0.0

            ckpt_p = s_dir / "model.pt"
            torch.save(
                {
                    "arm": arm.name,
                    "seed": seed,
                    "best_epoch": best_ep,
                    "init_sha256": init_shas[seed],
                    "train_order_sha256": order_hash.hexdigest(),
                    "state_dict": best_state,
                    "norm": {k: (v[0].tolist(), v[1].tolist()) for k, v in norm.items()},
                    "policy": policy,
                },
                ckpt_p,
            )
            card = {
                "arm": arm.name,
                "seed": seed,
                "best_epoch": best_ep,
                "init_sha256": init_shas[seed],
                "train_order_sha256": order_hash.hexdigest(),
                "checkpoint_sha256": sha_file(ckpt_p),
                "peak_mib": peak_mib,
                "wall_s": round(time.perf_counter() - t0, 1),
                "policy": policy,
                "cal": cal_m,
                "history": hist,
            }
            save_json(s_dir / "CHECKPOINT_CARD.json", card)
            card_sha = sha_file(s_dir / "CHECKPOINT_CARD.json")

            pred_records: List[Dict[str, Any]] = []
            for r, lg in zip(eval_prompts, ev_lg, strict=True):
                tp, gp = probabilities(lg, r, policy)
                pred_records.extend(scores_to_rows(r, tp, gp, policy))
            seal_predictions(s_dir, pred_records, card_sha)

            arm_seed_cal_logits[arm.name][seed] = cal_lg
            arm_seed_eval_logits[arm.name][seed] = ev_lg
            arm_seed_policies[arm.name][seed] = policy
            arm_seed_meta[arm.name][seed] = {**card, "pred_dir": str(s_dir / "eval_predictions")}
            print(
                f"  [{arm.name:14s} seed={seed}] ep={best_ep:2d} Cal F1={cal_m['cal_l2_strict_f1']:5.2f}% "
                f"tAP={cal_m['cal_t_ap']:.4f} gAP={cal_m['cal_g_ap']:.4f} L1={cal_m['cal_l1_fa_per_1k']:.2f}/1k "
                f"peak={peak_mib:.0f}MiB ({card['wall_s']}s)",
                flush=True,
            )

        # Build & seal 3-seed O5+O4 ensemble for this arm strictly on Cal BEFORE reading Eval gold
        ens_dir = out_dir / arm.name / "ens3_o5_o4"
        ens_dir.mkdir(parents=True, exist_ok=True)
        cal_t_l2 = [np.concatenate([arm_seed_cal_logits[arm.name][s][i].target for i in l2_cal_idx], axis=0) for s in seeds]
        cal_g_l2 = [np.concatenate([arm_seed_cal_logits[arm.name][s][i].gap for i in l2_cal_idx], axis=0) for s in seeds]
        cal_t_l1 = [np.concatenate([arm_seed_cal_logits[arm.name][s][i].target for i in l1_cal_idx], axis=0) for s in seeds]
        cal_g_l1 = [np.concatenate([arm_seed_cal_logits[arm.name][s][i].gap for i in l1_cal_idx], axis=0) for s in seeds]
        et_l2, eg_l2, is_p = ensemble_predictions(cal_t_l2, cal_g_l2, mode="prob_mean", is_probs=False)
        et_l1, eg_l1, _ = ensemble_predictions(cal_t_l1, cal_g_l1, mode="prob_mean", is_probs=False)
        ens_pol, ens_extra = calibrate_ortho_policy(
            et_l2, l2_t_gold, eg_l2, l2_g_gold, et_l1, eg_l1,
            is_probs=is_p, tune_rho_o=False, fit_gap_temp=True,
            known_gap_pos_weight=8.0, max_l1_fa_per_1k=1.0,
            ensemble_mode="prob_mean", n_seeds=len(seeds),
        )
        ens_card = {
            "arm": f"{arm.name}_ens3_o5_o4",
            "seeds": list(seeds),
            "policy": ens_pol.to_dict(),
            "gap_calibration": ens_extra["gap_calibration"],
        }
        save_json(ens_dir / "CHECKPOINT_CARD.json", ens_card)
        ev_t_list = [np.concatenate([lg.target for lg in arm_seed_eval_logits[arm.name][s]], axis=0) for s in seeds]
        ev_g_list = [np.concatenate([lg.gap for lg in arm_seed_eval_logits[arm.name][s]], axis=0) for s in seeds]
        ev_t, ev_g, ev_is_p = ensemble_predictions(ev_t_list, ev_g_list, mode="prob_mean", is_probs=False)
        ev_out = apply_ortho_policy(ev_t, ev_g, ens_pol, is_probs=ev_is_p)
        if not (ens_dir / "PREDICTIONS_SEAL.json").exists():
            seal_predictions(ens_dir, _build_eval_records(eval_prompts, ev_out), sha_file(ens_dir / "CHECKPOINT_CARD.json"))

    # Verify exact init_sha256 and train_order_sha256 parity across all 4 arms per seed
    for seed in seeds:
        inits = {arm_seed_meta[a.name][seed]["init_sha256"] for a in arms}
        orders = {arm_seed_meta[a.name][seed]["train_order_sha256"] for a in arms}
        assert len(inits) == 1, f"Init SHA mismatch on seed {seed}: {inits}"
        assert len(orders) == 1, f"Order SHA mismatch on seed {seed}: {orders}"
    print("[Parity] Verified bit-for-bit identical init_sha256 & train_order_sha256 across all 4 arms × 3 seeds!", flush=True)

    # Post-seal single Eval pass
    gold_df = _load_gold_df()
    f0_cal_recs = {
        s: _cal_l2_target_recall(cal_rows, arm_seed_cal_logits["F0_control"][s], arm_seed_policies["F0_control"][s])
        for s in seeds
    }

    results: Dict[str, Any] = {
        "split": "clean_11",
        "epochs": epochs,
        "seeds": list(seeds),
        "wall_total_s": round(time.perf_counter() - t_all, 1),
        "arms": {},
    }

    for arm in arms:
        per_seed = {}
        for s in seeds:
            pred_dir = Path(arm_seed_meta[arm.name][s]["pred_dir"])
            ev_nat = _score_sealed(pred_dir, gold_df, can_lp_map, override_tau_t=None)
            rm_tt = _match_tau_t_on_cal(cal_rows, arm_seed_cal_logits[arm.name][s], f0_cal_recs[s], arm_seed_policies[arm.name][s])
            ev_rm = _score_sealed(pred_dir, gold_df, can_lp_map, override_tau_t=rm_tt)
            per_seed[str(s)] = {
                "best_epoch": arm_seed_meta[arm.name][s]["best_epoch"],
                "cal": arm_seed_meta[arm.name][s]["cal"],
                "native": ev_nat,
                "rm_tau_t": round(rm_tt, 4),
                "recall_matched": ev_rm,
            }

        def agg(mode_key: str) -> Dict[str, Any]:
            summary = {}
            for c in ("held6", "epadb_test", "saa27", "spont_h6"):
                f1s = [per_seed[str(s)][mode_key][c]["strict_f1"] for s in seeds]
                taps = [per_seed[str(s)][mode_key][c]["target_ap"] for s in seeds]
                gaps = [per_seed[str(s)][mode_key][c]["gap_ap"] for s in seeds]
                baps = [per_seed[str(s)][mode_key][c]["blind_ap"] for s in seeds]
                sws = [per_seed[str(s)][mode_key][c]["so_swap"] for s in seeds]
                summary[c] = {
                    "f1_mean": round(float(np.mean(f1s)), 2),
                    "f1_sd": round(float(np.std(f1s)), 2),
                    "tAP_mean": round(float(np.mean(taps)), 4),
                    "gAP_mean": round(float(np.mean(gaps)), 4),
                    "bAP_mean": round(float(np.mean(baps)), 4),
                    "so_swap_mean": round(float(np.mean(sws)), 1),
                }
            fas = [per_seed[str(s)][mode_key]["timit_core"]["fa_per_utt"] for s in seeds]
            ctrls = [per_seed[str(s)][mode_key]["ctrl_rec"] for s in seeds]
            summary["timit_fa_mean"] = round(float(np.mean(fas)), 3)
            summary["timit_fa_sd"] = round(float(np.std(fas)), 3)
            summary["ctrl_mean"] = round(float(np.mean(ctrls)), 2)
            return summary

        ens_eval = _score_sealed(out_dir / arm.name / "ens3_o5_o4/eval_predictions", gold_df, can_lp_map)
        results["arms"][arm.name] = {
            "per_seed": per_seed,
            "mean_native": agg("native"),
            "mean_recall_matched": agg("recall_matched"),
            "ens3_o5_o4": ens_eval,
        }

    # Compute paired 3-seed deltas vs F0_control
    f0 = results["arms"]["F0_control"]
    paired_vs_f0 = {}
    for arm in arms:
        if arm.name == "F0_control":
            continue
        cur = results["arms"][arm.name]
        p_arm = {}
        for mode_key in ("native", "recall_matched"):
            pm = {}
            for c in ("held6", "epadb_test", "saa27", "spont_h6"):
                diffs = [
                    round(cur["per_seed"][str(s)][mode_key][c]["strict_f1"] - f0["per_seed"][str(s)][mode_key][c]["strict_f1"], 2)
                    for s in seeds
                ]
                pm[c] = {"diffs": diffs, "mean": round(float(np.mean(diffs)), 2), "wins": sum(d > 0 for d in diffs)}
            t_diffs = [
                round(cur["per_seed"][str(s)][mode_key]["timit_core"]["fa_per_utt"] - f0["per_seed"][str(s)][mode_key]["timit_core"]["fa_per_utt"], 3)
                for s in seeds
            ]
            pm["timit_fa"] = {"diffs": t_diffs, "mean": round(float(np.mean(t_diffs)), 3)}
            p_arm[mode_key] = pm
        paired_vs_f0[arm.name] = p_arm
    results["paired_vs_f0"] = paired_vs_f0

    save_json(out_dir / "GPU_3SEED_VERIFY_RESULTS.json", results)

    print("\n=== TASK 2 RESULTS: 3-SEED MEAN ± SD (Native Cal-locked) ===")
    print(f"  {'Arm':15s} | {'Held-6 (tAP, sw)':22s} | {'EpaDB (tAP, sw)':22s} | {'SAA-27 (tAP)':17s} | {'Spont H6':11s} | {'TIMIT/u':11s} | {'Ctrl':6s}")
    for arm in arms:
        m = results["arms"][arm.name]["mean_native"]
        print(
            f"  {arm.name:15s} | {m['held6']['f1_mean']:5.2f}±{m['held6']['f1_sd']:.2f} ({m['held6']['tAP_mean']:.3f},{m['held6']['so_swap_mean']:5.1f}) | "
            f"{m['epadb_test']['f1_mean']:5.2f}±{m['epadb_test']['f1_sd']:.2f} ({m['epadb_test']['tAP_mean']:.3f},{m['epadb_test']['so_swap_mean']:5.1f}) | "
            f"{m['saa27']['f1_mean']:5.2f}±{m['saa27']['f1_sd']:.2f} ({m['saa27']['tAP_mean']:.3f}) | "
            f"{m['spont_h6']['f1_mean']:5.2f}±{m['spont_h6']['f1_sd']:.2f} | "
            f"{m['timit_fa_mean']:.3f}±{m['timit_fa_sd']:.3f} | {m['ctrl_mean']:5.2f}%"
        )

    print("\n=== PAIRED 3-SEED DELTAS vs F0_control (Native / Recall-Matched) ===")
    for arm_name, pd_arm in paired_vs_f0.items():
        for mk in ("native", "recall_matched"):
            pm = pd_arm[mk]
            print(
                f"  {arm_name:15s} [{mk:14s}]: "
                f"H6 {pm['held6']['mean']:+.2f}pp({pm['held6']['wins']}/3 {pm['held6']['diffs']}) | "
                f"Epa {pm['epadb_test']['mean']:+.2f}pp({pm['epadb_test']['wins']}/3 {pm['epadb_test']['diffs']}) | "
                f"SAA {pm['saa27']['mean']:+.2f}pp({pm['saa27']['wins']}/3) | "
                f"SpH6 {pm['spont_h6']['mean']:+.2f}pp({pm['spont_h6']['wins']}/3) | "
                f"TIM {pm['timit_fa']['mean']:+.3f}/u"
            )

    print("\n=== 3-SEED ENSEMBLE (O5+O4, Cal L1 <= 1.0/1k) OF EACH ARM ===")
    for arm in arms:
        e = results["arms"][arm.name]["ens3_o5_o4"]
        print(
            f"  {arm.name:15s} ens3 | H6={e['held6']['strict_f1']:5.2f}% (tAP={e['held6']['target_ap']:.4f}, sw={e['held6']['so_swap']:3d}) | "
            f"Epa={e['epadb_test']['strict_f1']:5.2f}% (sw={e['epadb_test']['so_swap']:3d}) | "
            f"SAA={e['saa27']['strict_f1']:5.2f}% | SpH6={e['spont_h6']['strict_f1']:5.2f}% | "
            f"TIM={e['timit_core']['fa_per_utt']:.3f}/u | Ctrl={e['ctrl_rec']:.2f}%"
        )
    print(f"\nSaved Task 2 results to {out_dir / 'GPU_3SEED_VERIFY_RESULTS.json'} ({results['wall_total_s']}s)", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research/ORTHO_IMPROVEMENTS_20261122_v1/runs/gpu_3seed_verify_v1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()
    run_verify(args.out, device_str=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
