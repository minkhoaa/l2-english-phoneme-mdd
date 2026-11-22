"""Paired 3-seed GPU experiment: Real L2 Human-Expert Data Expansion (SpeechOcean762 TRAIN).

Compares on clean_11 (3,681 Fit / 635 Cal / 2,446 Eval; test_sealed untouched):
  - B0_clean11         : exact S0_clean baseline (37 L2 Fit speakers + 334 TIMIT)
  - B1_so762_strict_all: + 2,500 SO762 TRAIN utts (+125 real L2 spks), typed S/O +
                         exact marginal binary (score <= 1.4 -> Err, 2.0 -> C, 1.6-1.8 masked)
  - B2_so762_adult_only: same as B1 using adult SO762 speakers only (age >= 16, 1,340 utts)
  - B3_so762_soft_all  : + 2,500 SO762 TRAIN utts, typed S/O + 5-expert continuous soft target
                         q_err = 1.0 - score / 2.0 via exact marginal binary CE

Every arm is evaluated across 3 paired seeds (20261121..23) with identical init_sha256,
Cal-only epoch/policy selection, pre-gold Parquet seals, Native + Cal L2 Recall-Matched
evaluation, and Cal-locked O5+O4 3-seed Head Ensemble.
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
import torch

from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import (
    BASE_KOEL_CACHE,
    load_splits_no_eval_gold,
    pool_koel_iso,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import CONTINUOUS_SCALAR_NAMES
from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    GAP_LABEL_TO_ID,
    TARGET_LABEL_TO_ID,
)
from research.ORTHO_IMPROVEMENTS_20261122_v1.calibration import (
    apply_ortho_policy,
    calibrate_ortho_policy,
    ensemble_predictions,
)
from research.ORTHO_IMPROVEMENTS_20261122_v1.package_s0_clean_ensemble import (
    _build_eval_records,
)
from research.ORTHO_IMPROVEMENTS_20261122_v1.run_gpu_3seed_verify import (
    _cal_l2_target_recall,
    _cal_metrics,
    _load_gold_df,
    _match_tau_t_on_cal,
    _score_sealed,
    state_sha,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import save_json, sha_file
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import Spec
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import (
    GAP_KEYS,
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

from .marginal_loss import compute_so762_hybrid_mdd_loss

SO762_PREP_DIR = Path("research/REAL_L2_SO762_20261122_v1/runs/so762_prep")
CAN_LP_IDX = CONTINUOUS_SCALAR_NAMES.index("can_lp_vit_max")
SEEDS: Tuple[int, ...] = (20261121, 20261122, 20261123)


@dataclass(frozen=True)
class SO762Arm:
    name: str
    use_so762: bool = False
    adults_only: bool = False
    mode: str = "strict"  # strict | soft
    so762_weight: float = 0.5


ARMS: Tuple[SO762Arm, ...] = (
    SO762Arm("B0_clean11", use_so762=False),
    SO762Arm("B1_so762_strict_all", use_so762=True, adults_only=False, mode="strict", so762_weight=0.5),
    SO762Arm("B2_so762_adult_only", use_so762=True, adults_only=True, mode="strict", so762_weight=0.5),
    SO762Arm("B3_so762_soft_all", use_so762=True, adults_only=False, mode="soft", so762_weight=0.5),
)


def _prepare_so762_training_rows(
    raw_so_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    adults_only: bool,
    so762_weight: float,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Convert raw SO762 rows into hybrid 3-way + marginal/soft training rows + original indices."""
    out_rows: List[Dict[str, Any]] = []
    orig_indices: List[int] = []
    for idx, r in enumerate(raw_so_rows):
        if adults_only and int(r["age"]) < 16:
            continue
        L = len(r["canonical_phones"])
        t_ids = [-100] * L
        soft_q = [-1.0] * L
        real_ids = list(r["explicit_real_id"])
        for i in range(L):
            et = r["explicit_type"][i]
            sc = float(r["phone_scores"][i])
            if et == "S":
                t_ids[i] = 1
            elif et == "O":
                t_ids[i] = 2
            elif mode == "strict":
                if et == "UNK_ERR" or sc <= 1.4:
                    soft_q[i] = 1.0  # pure marginal binary Err (no forced S/O)
                elif sc >= 2.0:
                    t_ids[i] = 0  # clean C
                # else 1.6..1.8 masked (-100, soft_q=-1.0)
            elif mode == "soft":
                if et == "UNK_ERR":
                    soft_q[i] = max(0.5, 1.0 - sc / 2.0)
                else:
                    soft_q[i] = float(np.clip(1.0 - sc / 2.0, 0.0, 1.0))
            else:
                raise ValueError(f"Unknown mode: {mode}")

        out_rows.append({
            "source": "so762_train",
            "uid": r["uid"],
            "speaker": r["speaker"],
            "canonical_phones": r["canonical_phones"],
            "target_ids_direct": t_ids,
            "soft_err_prob": soft_q,
            "realized_ids": real_ids,
            "gap_masked": True,
            "is_native_l1": False,
            "slot_weight": float(so762_weight),
        })
        orig_indices.append(idx)
    return out_rows, orig_indices


def collate_hybrid(
    rows: Sequence[Mapping[str, Any]],
    feats: Sequence[Mapping[str, np.ndarray]],
    stats: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
):
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

    t = np.full((len(rows), max_l), -100, dtype=np.int64)
    g = np.full((len(rows), max_l + 1), -100, dtype=np.int64)
    r_lbl = np.full((len(rows), max_l), -100, dtype=np.int64)
    soft_q = np.full((len(rows), max_l), -1.0, dtype=np.float32)
    slot_w = np.ones((len(rows), max_l), dtype=np.float32)

    for i, (row, L) in enumerate(zip(rows, lengths, strict=True)):
        if "target_ids_direct" in row:
            t[i, :L] = row["target_ids_direct"]
            soft_q[i, :L] = row["soft_err_prob"]
            slot_w[i, :L] = float(row.get("slot_weight", 1.0))
        else:
            t[i, :L] = [TARGET_LABEL_TO_ID.get(a, -100) for a in row["gold_targets"]]
            g[i, : L + 1] = [GAP_LABEL_TO_ID.get(a, -100) for a in row["gold_gaps"]]
        if "realized_ids" in row:
            r_lbl[i, :L] = row["realized_ids"]
    is_l1 = np.array([row.get("is_native_l1", False) for row in rows], dtype=bool)

    return args, (
        torch.from_numpy(t).to(device),
        torch.from_numpy(g).to(device),
        torch.from_numpy(r_lbl).to(device),
        torch.from_numpy(is_l1).to(device),
        torch.from_numpy(soft_q).to(device),
        torch.from_numpy(slot_w).to(device),
    )


def run_experiment(
    out_dir: Path,
    *,
    prep_dir: Path = SO762_PREP_DIR,
    device_str: str = "cuda",
    epochs: int = 14,
    batch_size: int = 16,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[SO762Arm] = ARMS,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    t_all = time.perf_counter()

    print(f"=== REAL L2 EXPANSION (SpeechOcean762 TRAIN) | clean_11 | device={dev} | epochs={epochs} ===", flush=True)
    fit_base, cal_rows, eval_prompts = load_splits_no_eval_gold("clean_11")
    with (prep_dir / "so762_train_manifest.jsonl").open("r", encoding="utf-8") as f:
        so_raw = [json.loads(line) for line in f]
    so_cache = prep_dir / "koel_cache"

    t_pool = time.perf_counter()
    xf_base = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in fit_base]
    xf_so = [pool_koel_iso(so_cache, r) for r in so_raw]
    xc = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in cal_rows]
    xe = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in eval_prompts]
    # Keep exact clean_11 scalar normalization across all arms for clean comparability
    norm = norm_from_fit(xf_base)
    print(f"[Pool] clean11 Fit({len(fit_base)}) + SO762({len(so_raw)}) + Cal({len(cal_rows)}) + Eval({len(eval_prompts)}) in {time.perf_counter()-t_pool:.1f}s", flush=True)

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
    init_states, init_shas = {}, {}
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
        if arm.use_so762:
            so_rows, so_idx = _prepare_so762_training_rows(
                so_raw, mode=arm.mode, adults_only=arm.adults_only, so762_weight=arm.so762_weight
            )
            fit_rows = list(fit_base) + so_rows
            xf = list(xf_base) + [xf_so[i] for i in so_idx]
        else:
            fit_rows = list(fit_base)
            xf = list(xf_base)

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
                print(f"  [{arm.name:20s} seed={seed}] RESUMED ep={card['best_epoch']} Cal F1={card['cal']['cal_l2_strict_f1']:.2f}%", flush=True)
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
                    inp, (t_lab, g_lab, r_lab, is_l1, soft_q, slot_w) = collate_hybrid(
                        [fit_rows[i] for i in ix], [xf[i] for i in ix], norm, dev
                    )
                    opt.zero_grad(set_to_none=True)
                    loss = compute_so762_hybrid_mdd_loss(
                        model(**inp),
                        t_lab,
                        g_lab,
                        r_lab,
                        is_l1,
                        soft_err_prob=soft_q if arm.use_so762 else None,
                        slot_weight=slot_w if arm.use_so762 else None,
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
                "n_fit_utts": len(fit_rows),
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
                f"  [{arm.name:20s} seed={seed}] ep={best_ep:2d} Cal F1={cal_m['cal_l2_strict_f1']:5.2f}% "
                f"tAP={cal_m['cal_t_ap']:.4f} gAP={cal_m['cal_g_ap']:.4f} bAP={cal_m['cal_blind_ap']:.4f} "
                f"L1={cal_m['cal_l1_fa_per_1k']:.2f}/1k ({card['wall_s']}s)",
                flush=True,
            )

        # Build & seal Cal-locked O5+O4 3-seed ensemble for each arm BEFORE gold join
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

    # Post-seal single Eval pass
    gold_df = _load_gold_df()
    b0_cal_recs = {
        s: _cal_l2_target_recall(cal_rows, arm_seed_cal_logits["B0_clean11"][s], arm_seed_policies["B0_clean11"][s])
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
            rm_tt = _match_tau_t_on_cal(cal_rows, arm_seed_cal_logits[arm.name][s], b0_cal_recs[s], arm_seed_policies[arm.name][s])
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
                summary[c] = {
                    "f1_mean": round(float(np.mean(f1s)), 2),
                    "f1_sd": round(float(np.std(f1s)), 2),
                    "tAP_mean": round(float(np.mean(taps)), 4),
                    "gAP_mean": round(float(np.mean(gaps)), 4),
                    "bAP_mean": round(float(np.mean(baps)), 4),
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

    # Paired 3-seed deltas vs B0_clean11
    b0 = results["arms"]["B0_clean11"]
    paired_vs_b0 = {}
    for arm in arms:
        if arm.name == "B0_clean11":
            continue
        cur = results["arms"][arm.name]
        p_arm = {}
        for mk in ("native", "recall_matched"):
            pm = {}
            for c in ("held6", "epadb_test", "saa27", "spont_h6"):
                diffs = [
                    round(cur["per_seed"][str(s)][mk][c]["strict_f1"] - b0["per_seed"][str(s)][mk][c]["strict_f1"], 2)
                    for s in seeds
                ]
                tap_d = [
                    round(cur["per_seed"][str(s)][mk][c]["target_ap"] - b0["per_seed"][str(s)][mk][c]["target_ap"], 4)
                    for s in seeds
                ]
                pm[c] = {
                    "diffs": diffs,
                    "mean": round(float(np.mean(diffs)), 2),
                    "wins": sum(d > 0 for d in diffs),
                    "tAP_mean_delta": round(float(np.mean(tap_d)), 4),
                }
            t_diffs = [
                round(cur["per_seed"][str(s)][mk]["timit_core"]["fa_per_utt"] - b0["per_seed"][str(s)][mk]["timit_core"]["fa_per_utt"], 3)
                for s in seeds
            ]
            pm["timit_fa"] = {"diffs": t_diffs, "mean": round(float(np.mean(t_diffs)), 3)}
            p_arm[mk] = pm
        paired_vs_b0[arm.name] = p_arm
    results["paired_vs_b0"] = paired_vs_b0

    save_json(out_dir / "SO762_3SEED_RESULTS.json", results)

    print("\n=== SO762 3-SEED MEAN ± SD (Native Cal-locked) ===")
    print(f"  {'Arm':20s} | {'Held-6 (tAP/bAP)':22s} | {'EpaDB (tAP)':17s} | {'SAA-27 (tAP)':17s} | {'Spont H6':11s} | {'TIMIT/u':11s} | {'Ctrl':6s}")
    for arm in arms:
        m = results["arms"][arm.name]["mean_native"]
        print(
            f"  {arm.name:20s} | {m['held6']['f1_mean']:5.2f}±{m['held6']['f1_sd']:.2f} ({m['held6']['tAP_mean']:.3f}/{m['held6']['bAP_mean']:.3f}) | "
            f"{m['epadb_test']['f1_mean']:5.2f}±{m['epadb_test']['f1_sd']:.2f} ({m['epadb_test']['tAP_mean']:.3f}) | "
            f"{m['saa27']['f1_mean']:5.2f}±{m['saa27']['f1_sd']:.2f} ({m['saa27']['tAP_mean']:.3f}) | "
            f"{m['spont_h6']['f1_mean']:5.2f}±{m['spont_h6']['f1_sd']:.2f} | "
            f"{m['timit_fa_mean']:.3f}±{m['timit_fa_sd']:.3f} | {m['ctrl_mean']:5.2f}%"
        )

    print("\n=== PAIRED 3-SEED DELTAS vs B0_clean11 ===")
    for arm_name, pd_arm in paired_vs_b0.items():
        for mk in ("native", "recall_matched"):
            pm = pd_arm[mk]
            print(
                f"  {arm_name:20s} [{mk:14s}]: "
                f"H6 {pm['held6']['mean']:+.2f}pp({pm['held6']['wins']}/3, dAP={pm['held6']['tAP_mean_delta']:+.4f}) | "
                f"Epa {pm['epadb_test']['mean']:+.2f}pp({pm['epadb_test']['wins']}/3) | "
                f"SAA {pm['saa27']['mean']:+.2f}pp({pm['saa27']['wins']}/3, dAP={pm['saa27']['tAP_mean_delta']:+.4f}) | "
                f"SpH6 {pm['spont_h6']['mean']:+.2f}pp({pm['spont_h6']['wins']}/3) | "
                f"TIM {pm['timit_fa']['mean']:+.3f}/u"
            )

    print("\n=== 3-SEED ENSEMBLE (O5+O4, Cal L1 <= 1.0/1k) OF EACH ARM ===")
    for arm in arms:
        e = results["arms"][arm.name]["ens3_o5_o4"]
        print(
            f"  {arm.name:20s} ens3 | H6={e['held6']['strict_f1']:5.2f}% (tAP={e['held6']['target_ap']:.4f}, bAP={e['held6']['blind_ap']:.4f}) | "
            f"Epa={e['epadb_test']['strict_f1']:5.2f}% | SAA={e['saa27']['strict_f1']:5.2f}% (tAP={e['saa27']['target_ap']:.4f}) | "
            f"SpH6={e['spont_h6']['strict_f1']:5.2f}% | TIM={e['timit_core']['fa_per_utt']:.3f}/u | Ctrl={e['ctrl_rec']:.2f}%"
        )
    print(f"\nSaved full SO762 results to {out_dir / 'SO762_3SEED_RESULTS.json'} ({results['wall_total_s']}s)", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research/REAL_L2_SO762_20261122_v1/runs/so762_3seed_v1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()
    run_experiment(args.out, device_str=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
