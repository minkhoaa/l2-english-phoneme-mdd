"""Paired 1-variable CPU ablation for ORTHO_IMPROVEMENTS_20261122_v1 on clean_11.

Strictly zero GPU (CPU-only), zero Eval leakage before Parquet seal, and exact matched
initial weights + batch order across all arms:
  - O0_control    : exact SingleFactorV2 (iso_no_hwin) baseline on clean_11
  - O1_timit_freq : ONLY frequency-matched TIMIT Fit corruption (fixes Zipf bug)
  - O2_logit_adj  : ONLY (source, phone) Logit Adjustment (tau_la=0.5, alpha=20)
  - O2b_phone_drop: ONLY phone_id dropout (p=0.4 during train, off at eval)
  - O3_cond_so    : ONLY conditional S-vs-O BCE loss on true error slots (lambda=0.5)

Does NOT modify any file in BACKBONE_REP_CAMPAIGN_20261122_v1, GOAL.md, or archived
checkpoints, and NEVER opens test_sealed.jsonl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from .losses import compute_ortho_mdd_loss
from .prior_debias import (
    SourcePhonePriorTable,
    source_to_id,
)
from .timit_freq_corrupt import (
    load_clean_timit_phones,
    rebuild_fit_with_freq_matched_timit,
)

CAN_LP_IDX = CONTINUOUS_SCALAR_NAMES.index("can_lp_vit_max")
EVAL_GOLD_PATH = Path(
    "research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/v2_prompt_eos_repair_v1/splits/eval_gold.jsonl"
)
HELD6_SPEAKERS = {"ABA", "ASI", "BWC", "EBVS", "HQTV", "LXC"}
VALID_GOLD = {"C", "S", "O", "N", "I"}
COHORTS = ("held6", "epadb_test", "saa27", "spontaneous", "timit_core", "timit_core_corrupt_ctrl")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    use_o1_timit_freq: bool = False
    tau_la: float = 0.0
    phone_drop_p: float = 0.0
    lambda_so_cond: float = 0.0


ALL_ARMS: Tuple[ArmSpec, ...] = (
    ArmSpec("O0_control"),
    ArmSpec("O1_timit_freq", use_o1_timit_freq=True),
    ArmSpec("O2_logit_adj", tau_la=0.5),
    ArmSpec("O2b_phone_drop", phone_drop_p=0.4),
    ArmSpec("O3_cond_so", lambda_so_cond=0.5),
)


def state_sha(state_dict: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode("utf-8"))
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _cal_summary(
    cal_rows: Sequence[Mapping[str, Any]],
    cal_logits: Sequence[Any],
    policy: Mapping[str, Any],
    xc: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, float]:
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


def _load_eval_gold_df() -> pd.DataFrame:
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


def _score_sealed_eval(
    pred_dir: Path,
    gold_df: pd.DataFrame,
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
        call[is_t] = np.where(sc[is_t] >= override_tau_t, np.where(p_s[is_t] >= p_o[is_t], "S", "O"), "C")
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
            t_sub = sub[sub["domain"] == "target"]
            g_sub = sub[sub["domain"] == "gap"]
            t_fa = int((t_sub["call"] != "C").sum())
            g_fa = int((g_sub["call"] != "N").sum())
            uw_fa = int(((t_sub["canonical_phone"] == "UW") & (t_sub["call"] != "C")).sum())
            rare_fa = int(
                (t_sub["canonical_phone"].isin(["UW", "NG", "TH", "AW", "ZH", "UH", "JH", "CH"])
                 & (t_sub["call"] != "C")).sum()
            )
            out["timit_fa_utt"] = round((t_fa + g_fa) / 192.0, 3)
            out["timit_t_fa"] = t_fa
            out["timit_g_fa"] = g_fa
            out["timit_uw_fa"] = uw_fa
            out["timit_rare8_fa"] = rare_fa
            continue
        else:
            sub = df[(df["cohort"] == name) & (df["gold"].isin(VALID_GOLD))]

        g = sub["gold"].values
        p = sub["call"].values
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
        # Also measure DH/Z/AH/IH/AO/AE FPR on C
        c_sub = ts[ts["gold"] == "C"]
        key_fpr = {}
        for ph in ("DH", "Z", "AH", "AO"):
            ps = c_sub[c_sub["canonical_phone"] == ph]
            if len(ps) > 0:
                key_fpr[ph] = round(100.0 * float((ps["call"] != "C").mean()), 1)

        out[name] = {
            "f1": round(f1, 2),
            "tAP": round(tap, 4),
            "gAP": round(gap, 4),
            "so_swap": s2o + o2s,
            "s2o": s2o,
            "o2s": o2s,
            "key_c_fpr": key_fpr,
        }
    return out


def run_ablation(
    out_dir: Path,
    *,
    seed: int = 20261121,
    epochs: int = 8,
    lr_total_epochs: int = 14,
    batch_size: int = 16,
    arms: Sequence[ArmSpec] = ALL_ARMS,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    t_all = time.perf_counter()

    print(f"=== ORTHO_IMPROVEMENTS CPU ABLATION | split=clean_11 | seed={seed} | epochs={epochs} (cos T_max={lr_total_epochs}) ===", flush=True)
    fit_base, cal_rows, eval_prompts = load_splits_no_eval_gold("clean_11")

    # Build O1 rebuilt Fit rows (same 1000 TIMIT WAVs)
    timit_uids = [r["uid"] for r in fit_base if str(r.get("source", "")).startswith("timit")]
    clean_timit_map = load_clean_timit_phones(timit_uids)
    fit_o1, _ = rebuild_fit_with_freq_matched_timit(fit_base, clean_timit_map, seed=20261122)

    # Pool Koel features ONCE (reuse non-TIMIT features between base and O1!)
    t_pool = time.perf_counter()
    xf_base = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in fit_base]
    xf_o1 = []
    for i, r in enumerate(fit_o1):
        if str(r.get("source", "")).startswith("timit"):
            xf_o1.append(pool_koel_iso(BASE_KOEL_CACHE, r))
        else:
            xf_o1.append(xf_base[i])
    xc = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in cal_rows]
    xe = [pool_koel_iso(BASE_KOEL_CACHE, r) for r in eval_prompts]
    print(f"[Pool] Fit({len(fit_base)}) + O1_TIMIT(1000) + Cal({len(cal_rows)}) + Eval({len(eval_prompts)}) in {time.perf_counter()-t_pool:.1f}s", flush=True)

    norm_base = norm_from_fit(xf_base)
    norm_o1 = norm_from_fit(xf_o1)
    prior_base = SourcePhonePriorTable.from_fit_rows(fit_base, shrinkage_alpha=20.0).to_log_prior_tensor(dev)

    # Pre-generate identical initial state for exact parity across all arms
    cfg = FACTOR_ARMS["iso_no_hwin"]
    torch.manual_seed(seed)
    init_model = SingleFactorV2(cfg)
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
    init_sha = state_sha(init_state)
    print(f"[Init] SingleFactorV2 iso_no_hwin init_sha={init_sha[:16]}...", flush=True)

    arm_artifacts: Dict[str, Any] = {}
    arm_cal_logits: Dict[str, Any] = {}
    arm_policies: Dict[str, Any] = {}

    for spec in arms:
        t_arm = time.perf_counter()
        a_dir = out_dir / spec.name
        a_dir.mkdir(parents=True, exist_ok=True)
        fit_rows = fit_o1 if spec.use_o1_timit_freq else fit_base
        xf = xf_o1 if spec.use_o1_timit_freq else xf_base
        norm = norm_o1 if spec.use_o1_timit_freq else norm_base
        src_ids_all = np.array([source_to_id(r["source"]) for r in fit_rows], dtype=np.int64)

        if (a_dir / "model.pt").exists() and (a_dir / "PREDICTIONS_SEAL.json").exists() and (a_dir / "CHECKPOINT_CARD.json").exists():
            ckpt = torch.load(a_dir / "model.pt", map_location="cpu", weights_only=False)
            card = json.loads((a_dir / "CHECKPOINT_CARD.json").read_text(encoding="utf-8"))
            model = SingleFactorV2(cfg).to(dev)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            cal_logits = infer_sf_logits(model, cal_rows, xc, norm, dev, batch_size=32)
            policy = ckpt["policy"]
            arm_cal_logits[spec.name] = cal_logits
            arm_policies[spec.name] = policy
            arm_artifacts[spec.name] = {
                "spec": spec.__dict__,
                "best_epoch": int(ckpt["best_epoch"]),
                "init_sha256": str(ckpt["init_sha256"]),
                "train_order_sha256": str(ckpt["train_order_sha256"]),
                "checkpoint_sha256": str(card["checkpoint_sha256"]),
                "seal_sha256": sha_file(a_dir / "PREDICTIONS_SEAL.json"),
                "wall_s": 0.0,
                "cal": card["cal"],
                "history": [],
                "pred_dir": str(a_dir / "eval_predictions"),
                "resumed": True,
            }
            print(f"  ==> [{spec.name}] RESUMED from sealed artifact (ep={ckpt['best_epoch']}, Cal F1={card['cal']['cal_l2_strict_f1']:.2f}%)", flush=True)
            continue

        torch.manual_seed(seed)
        model = SingleFactorV2(cfg).to(dev)
        model.load_state_dict(init_state)
        hook_handle = None
        if spec.phone_drop_p > 0.0:
            p_drop = float(spec.phone_drop_p)

            def _drop_phone_emb_hook(module, _inp, out):
                if not module.training:
                    return out
                keep = (torch.rand(out.shape[:-1], device=out.device) >= p_drop).to(out.dtype)
                return out * keep.unsqueeze(-1)

            hook_handle = model.phone_emb.register_forward_hook(_drop_phone_emb_hook)

        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
        n_steps = max(1, math.ceil(len(fit_rows) / batch_size))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=lr_total_epochs * n_steps, eta_min=2e-5
        )
        rng = np.random.default_rng(seed)
        order_hash = hashlib.sha256()

        hist, best_sel, best_ep, best_state = [], -math.inf, 0, None
        for ep in range(1, epochs + 1):
            t_ep = time.perf_counter()
            model.train()
            order = rng.permutation(len(fit_rows))
            order_hash.update(order.astype("<i8").tobytes())
            loss_sum = 0.0
            for start in range(0, len(order), batch_size):
                ix = order[start : start + batch_size]
                inp, labels = collate_sf(
                    [fit_rows[i] for i in ix], [xf[i] for i in ix], norm, dev, labels=True
                )
                orig_pids = inp["phone_ids"]
                opt.zero_grad(set_to_none=True)
                outputs = model(**inp)
                b_src = torch.from_numpy(src_ids_all[ix]).to(dev)
                loss_d = compute_ortho_mdd_loss(
                    outputs,
                    *labels,
                    source_ids=b_src if spec.tau_la != 0.0 else None,
                    phone_ids=orig_pids if spec.tau_la != 0.0 else None,
                    log_prior_table=prior_base if spec.tau_la != 0.0 else None,
                    tau_la=spec.tau_la,
                    lambda_so_cond=spec.lambda_so_cond,
                    gap_pos_weight=8.0,
                    lambda_gap=0.6,
                    lambda_realized=0.25,
                    native_fa_weight=1.5,
                )
                loss = loss_d["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                sched.step()
                loss_sum += float(loss.detach())

            sel, tap, gap = cal_score(
                cal_rows, infer_sf_logits(model, cal_rows, xc, norm, dev, batch_size=32)
            )
            hist.append({
                "epoch": ep,
                "loss": round(loss_sum / n_steps, 5),
                "cal_t_ap": round(float(tap), 5),
                "cal_g_ap": round(float(gap), 5),
                "sel": round(float(sel), 5),
                "ep_s": round(time.perf_counter() - t_ep, 1),
            })
            print(
                f"  [{spec.name:15s}] ep {ep}/{epochs} loss={loss_sum/n_steps:.4f} "
                f"Cal tAP={tap:.4f} gAP={gap:.4f} sel={sel:.4f} ({time.perf_counter()-t_ep:.1f}s)",
                flush=True,
            )
            if sel > best_sel:
                best_sel, best_ep = sel, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Lock best Cal checkpoint & policy strictly on Cal
        if hook_handle is not None:
            hook_handle.remove()
        model.load_state_dict(best_state)
        model.eval()
        cal_logits = infer_sf_logits(model, cal_rows, xc, norm, dev, batch_size=32)
        policy = choose_policy(cal_rows, cal_logits, Spec())
        cal_m = _cal_summary(cal_rows, cal_logits, policy, xc)

        # Seal Eval predictions BEFORE reading gold
        eval_logits = infer_sf_logits(model, eval_prompts, xe, norm, dev, batch_size=32)
        ckpt_p = a_dir / "model.pt"
        torch.save(
            {
                "arm": spec.name,
                "seed": seed,
                "best_epoch": best_ep,
                "init_sha256": init_sha,
                "train_order_sha256": order_hash.hexdigest(),
                "state_dict": best_state,
                "policy": policy,
            },
            ckpt_p,
        )
        card = {
            "arm": spec.name,
            "seed": seed,
            "best_epoch": best_ep,
            "init_sha256": init_sha,
            "train_order_sha256": order_hash.hexdigest(),
            "checkpoint_sha256": sha_file(ckpt_p),
            "policy": policy,
            "cal": cal_m,
        }
        save_json(a_dir / "CHECKPOINT_CARD.json", card)
        card_sha = sha_file(a_dir / "CHECKPOINT_CARD.json")

        pred_records: List[Dict[str, Any]] = []
        for r, lg in zip(eval_prompts, eval_logits, strict=True):
            tp, gp = probabilities(lg, r, policy)
            pred_records.extend(scores_to_rows(r, tp, gp, policy))
        seal_predictions(a_dir, pred_records, card_sha)
        pred_dir = a_dir / "eval_predictions"

        arm_cal_logits[spec.name] = cal_logits
        arm_policies[spec.name] = policy
        arm_artifacts[spec.name] = {
            "spec": spec.__dict__,
            "best_epoch": best_ep,
            "init_sha256": init_sha,
            "train_order_sha256": order_hash.hexdigest(),
            "checkpoint_sha256": card["checkpoint_sha256"],
            "seal_sha256": sha_file(a_dir / "PREDICTIONS_SEAL.json"),
            "wall_s": round(time.perf_counter() - t_arm, 1),
            "cal": cal_m,
            "history": hist,
            "pred_dir": str(pred_dir),
        }
        print(
            f"  ==> [{spec.name}] DONE ep={best_ep} Cal F1={cal_m['cal_l2_strict_f1']:.2f}% "
            f"tAP={cal_m['cal_t_ap']:.4f} gAP={cal_m['cal_g_ap']:.4f} blindAP={cal_m['cal_blind_ap']:.4f} "
            f"L1={cal_m['cal_l1_fa_per_1k']:.2f}/1k ({arm_artifacts[spec.name]['wall_s']}s)",
            flush=True,
        )

    # Post-seal single Eval pass (Native + Cal L2 Recall-Matched to O0_control)
    print("\n=== POST-SEAL EVALUATION (Native Cal-locked + Cal L2 Recall-Matched to O0) ===", flush=True)
    gold_df = _load_eval_gold_df()
    o0_cal_rec = _cal_l2_target_recall(cal_rows, arm_cal_logits["O0_control"], arm_policies["O0_control"])

    results: Dict[str, Any] = {
        "split": "clean_11",
        "seed": seed,
        "epochs": epochs,
        "lr_total_epochs": lr_total_epochs,
        "init_sha256": init_sha,
        "o0_cal_l2_target_recall": round(o0_cal_rec, 5),
        "wall_total_s": round(time.perf_counter() - t_all, 1),
        "arms": {},
    }

    for spec in arms:
        art = arm_artifacts[spec.name]
        pred_dir = Path(art["pred_dir"])
        ev_native = _score_sealed_eval(pred_dir, gold_df, override_tau_t=None)
        rm_tau_t = _match_tau_t_on_cal(cal_rows, arm_cal_logits[spec.name], o0_cal_rec, arm_policies[spec.name])
        ev_rm = _score_sealed_eval(pred_dir, gold_df, override_tau_t=rm_tau_t)
        results["arms"][spec.name] = {
            **art,
            "eval_native": ev_native,
            "rm_tau_t": round(rm_tau_t, 4),
            "eval_cal_recall_matched": ev_rm,
        }

    save_json(out_dir / "CPU_ABLATION_RESULTS.json", results)

    # Pretty summary table
    print(f"\n  {'Arm':15s} | {'CalF1':5s} {'tAP':6s} {'gAP':6s} | {'Held6':6s}({'tAP':5s},{'sw':3s}) | {'EpaDB':6s}({'tAP':5s},{'sw':3s}) | {'SAA27':6s}({'tAP':5s}) | {'SpH6':5s} | {'TIMIT(UW/R8)':13s} | {'Ctrl':5s}")
    for spec in arms:
        d = results["arms"][spec.name]
        c = d["cal"]
        e = d["eval_native"]
        print(
            f"  {spec.name:15s} | {c['cal_l2_strict_f1']:5.2f} {c['cal_t_ap']:.4f} {c['cal_g_ap']:.4f} | "
            f"{e['held6']['f1']:6.2f}({e['held6']['tAP']:.3f},{e['held6']['so_swap']:3d}) | "
            f"{e['epadb_test']['f1']:6.2f}({e['epadb_test']['tAP']:.3f},{e['epadb_test']['so_swap']:3d}) | "
            f"{e['saa27']['f1']:6.2f}({e['saa27']['tAP']:.3f}) | {e['spont_h6']['f1']:5.2f} | "
            f"{e['timit_fa_utt']:5.3f}({e['timit_uw_fa']:2d}/{e['timit_rare8_fa']:2d}) | {e['ctrl_rec']:5.2f}"
        )

    print("\n  [Cal L2 Recall-Matched to O0]")
    for spec in arms:
        d = results["arms"][spec.name]
        e = d["eval_cal_recall_matched"]
        print(
            f"  {spec.name:15s} (tt={d['rm_tau_t']:.3f}) | "
            f"Held6={e['held6']['f1']:5.2f}% EpaDB={e['epadb_test']['f1']:5.2f}% "
            f"SAA27={e['saa27']['f1']:5.2f}% SpH6={e['spont_h6']['f1']:5.2f}% | "
            f"TIMIT={e['timit_fa_utt']:.3f}/u (UW={e['timit_uw_fa']}, R8={e['timit_rare8_fa']}) Ctrl={e['ctrl_rec']:.2f}%"
        )
    print(f"\nSaved full results to {out_dir / 'CPU_ABLATION_RESULTS.json'} (total {results['wall_total_s']}s)", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research/ORTHO_IMPROVEMENTS_20261122_v1/runs/cpu_ablation_s20261121"))
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    run_ablation(args.out, seed=args.seed, epochs=args.epochs)


if __name__ == "__main__":
    main()
