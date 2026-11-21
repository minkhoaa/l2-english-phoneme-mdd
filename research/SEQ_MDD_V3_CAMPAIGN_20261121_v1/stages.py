"""Sequential experiments 2–4; only diagnostic gate-passing changes advance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    SAA27_GOLD,
    normalize_canonical_phone,
)
from research.KOEL_V2_FILTER_BENCHMARK_20261119_v1.run_v2_filter_benchmark import FOLDS
from .data import (
    COHORTS,
    CONTROL,
    ROOT,
    SEED,
    admitted,
    bind_audio,
    fill_cache,
    identity,
    only_prompt,
    read_json,
    read_jsonl,
    save_json,
    save_jsonl,
    sha_file,
    split_overlap,
    summary,
)
from .features import Spec, build_kernel
from .prepare_l1_canonical_anchor import extended_anchors, load_anchor
from .scoring import (
    brief,
    evaluate_seal,
    load_old_dev_r0,
    load_old_r0,
    read_predictions,
    seal_predictions,
)
from .training import (
    choose_policy,
    evaluate_arm,
    features_from_cache,
    infer_logits,
    load_checkpoint,
    pred_batches,
    release_cuda,
    train_arm,
)

L2 = COHORTS[:-1]
ANCHOR = "manifests/l1_canonical_anchor.jsonl"


def inherited(run, through):
    path = run / "v2_keyed"
    for name in through:
        d = read_json(
            run / name / ("OUTCOME.json" if name == "exp2_l1_canon" else "RESULTS.json")
        )
        if d["status"] not in {"GO", "NO_GO"}:
            raise RuntimeError(f"Unfinished parent {name}: {d['status']}")
        if d["status"] == "GO":
            path = Path(d["promoted"])
    card = read_json(path / "CHECKPOINT_CARD.json")
    return path, Spec(**card.get("spec", {}))


def comparisons(run, parent):
    x = {"r0_old": load_old_r0(), "seq_mdd_v2": read_predictions(run / "v2_keyed")}
    if parent != run / "v2_keyed":
        x["parent"] = read_predictions(parent)
    return x


def training_rows(run, canonical=False):
    fit, cal = [admitted(run / "splits", f"{role}.jsonl") for role in ("fit", "cal")]
    if canonical:
        anc = load_anchor(run / ANCHOR)
        fit = [r for r in fit if not r["source"].startswith("timit")] + anc["fit"]
        cal = [r for r in cal if not r["source"].startswith("timit")] + anc["cal"]
    return fit, cal


def guard_split(fit, cal, eval_rows):
    for left, right in [(fit, cal), (fit, eval_rows), (cal, eval_rows)]:
        check = split_overlap(left, right)
        if any(check.values()):
            raise ValueError(f"Unexpected split exposure: {check}")


def write_result(path, ret):
    save_json(path / "RESULTS.json", ret)
    # One compact report per experiment as required by GOAL; numerical details stay as JSON.
    with (path / "REPORT_VI.md").open("x", encoding="utf-8") as f:
        f.write(f"# {ret['experiment']} — {ret['status']}\n\n")
        f.write(
            "Điểm khóa trên Calibrate; chấm chẩn đoán từ dự đoán đã niêm phong. Số chi tiết xem `RESULTS.json`.\n\n"
        )
        if ret.get("metrics_pct_or_fa"):
            for name, val in ret["metrics_pct_or_fa"].items():
                f.write(f"- {name}: `{val}`\n")
        for k, v in ret.get("gates", {}).items():
            f.write(f"- `{k}`: {'đạt' if v else 'không đạt'}\n")
        if ret.get("reason"):
            f.write(f"- {ret['reason']}\n")
    print(
        f"{ret['experiment']}: {ret['status']}; {ret.get('metrics_pct_or_fa', {})}",
        flush=True,
    )
    return ret


def run_exp2(run, device, seed=SEED, epochs=14, batch=16):
    parent, spec = inherited(run, ["exp1_window"])
    spec = replace(spec, l1_canon=True)
    out = run / "exp2_l1_canon"
    fit, cal = training_rows(run, True)
    ev = [
        r
        for r in admitted(run / "splits", "eval_prompts.jsonl")
        if r["cohort"] != "spontaneous"
    ]
    guard_split(fit, cal, ev)
    backbone = read_json(run / "splits/PREFLIGHT.json")["backbone"]
    receipt = fill_cache(fit + cal, run / "frames", device, backbone)
    train_arm(out, fit, cal, spec, run, device, seed, epochs, batch)
    save_json(out / "FRAME_RECEIPT.json", receipt)
    shutil.copyfile(
        (run / ANCHOR).parent / "l1_canonical_anchor_audit.json",
        out / "l1_canonical_anchor_audit.json",
    )
    r = evaluate_arm(out, run, device, comparisons(run, parent), latency=True)
    prev = read_json(parent / "RESULTS.json")["eval"]
    v2 = read_json(run / "v2_keyed/RESULTS.json")["eval"]
    e = r["eval"]
    t = e["timit_core"]
    ctrl = e[CONTROL]
    safety = {
        **{k: e[k]["strict"]["f1"] >= prev[k]["strict"]["f1"] - 0.004 for k in L2},
        "spontaneous_held6_safety": e["spontaneous"]["held6_only"]["strict"]["f1"]
        >= prev["spontaneous"]["held6_only"]["strict"]["f1"] - 0.004,
        "max_total_fa": t["fa_per_utt"] <= 0.65,
        "max_target_fa_per_1k": t["target_fa_per_1k"] <= 18,
        "aa_fa_minus_60pct": t["aa_fa"] <= 0.4 * v2["timit_core"]["aa_fa"],
        "control_recall": ctrl["recall"] >= 0.93,
        "control_ap": ctrl["ap"] >= 0.96,
    }
    # Arm RESULTS.json remains raw; stage outcome is preserved in OUTCOME.json and a tiny decision report.
    ret = {
        "experiment": "EXP-2",
        "status": "GO" if all(safety.values()) else "NO_GO",
        "gates": safety,
        "promoted": str(out) if all(safety.values()) else None,
        "metrics_pct_or_fa": brief(e),
        "parent": str(parent),
        "result": str(out / "RESULTS.json"),
    }
    save_json(out / "OUTCOME.json", ret)
    with (out / "REPORT_VI.md").open("x", encoding="utf-8") as f:
        f.write(
            f"# EXP-2 — {ret['status']}\n\nTIMIT TRAIN → CMUdict theo prompt ID; xem `OUTCOME.json` và `RESULTS.json`.\n"
        )
    print("EXP-2:", ret, flush=True)
    return ret


def run_exp3(run, device, seed=SEED, epochs=14, batch=16):
    parent, spec = inherited(run, ["exp1_window", "exp2_l1_canon"])
    out = run / "exp3_gap_so"
    out.mkdir()
    kernel = build_kernel(run / "ctc_delete.so")
    fit, cal = training_rows(run, spec.l1_canon)
    base = read_json(parent / "RESULTS.json")["eval"]
    arms = {
        "gap_core_focal": replace(spec, gap_core=True, focal=True),
        "ollr_realized": replace(spec, ollr=True, fuse_realized=True),
        "gapso_joint": replace(
            spec, gap_core=True, focal=True, ollr=True, fuse_realized=True
        ),
    }
    save_json(
        out / "DESIGN.json",
        {
            "arms": {a: s.as_dict() for a, s in arms.items()},
            "parent": str(parent),
            "promotion": "Only combined arm is gated; sub-arms are causal diagnostics for each named package.",
            "latency_budget": "P50: Held-6 and EpaDB short utterances <=145 ms; all Stella <=1600 ms; recorded P95 too.",
            "ollr_scope": "exact CTC for one canonical Koel token per ARPA phone; parity tested against original C++ kernel",
        },
    )
    details = {}
    for a, s in arms.items():
        path = out / a
        train_arm(path, fit, cal, s, run, device, seed, epochs, batch, kernel=kernel)
        r = evaluate_arm(
            path, run, device, comparisons(run, parent), kernel=kernel, latency=True
        )
        e = r["eval"]
        c = e[CONTROL]
        perf = r["latency"]
        idelta = {k: e[k]["I"]["f1"] - base[k]["I"]["f1"] for k in L2}
        strict_delta = {k: e[k]["strict"]["f1"] - base[k]["strict"]["f1"] for k in L2}
        interchange = lambda d, k: d[k]["s_to_o"] + d[k]["o_to_s"]
        swaps = {k: 1.0 - interchange(e, k) / max(interchange(base, k), 1) for k in L2}
        ok = {
            "mean_i_f1_plus_4_pp": float(np.mean(list(idelta.values()))) >= 0.04,
            "held6_i_f1_min_51": e["held6"]["I"]["f1"] >= 0.51,
            "epadb_i_f1_min_47": e["epadb_test"]["I"]["f1"] >= 0.47,
            "timit_gap_fa_max_15": e["timit_core"]["fa_gap"] <= 15,
            "held6_swaps_minus_15pct": swaps["held6"] >= 0.15,
            "spontaneous_swaps_minus_15pct": swaps["spontaneous"] >= 0.15,
            "mean_l2_strict_plus_point8_pp": float(np.mean(list(strict_delta.values())))
            >= 0.008,
            "timit_control": c["recall"] >= 0.90 and c["ap"] >= 0.95,
            "short_latency": bool(perf)
            and all(perf[k]["p50_ms"] <= 145 for k in ("held6", "epadb_test")),
            "stella_latency": bool(perf) and perf["saa27"]["p50_ms"] <= 1600,
        }
        if s.ollr:
            ok["ollr_short_p50_max_5_ms"] = bool(perf) and all(
                perf[k]["ollr_p50_ms"] < 5 for k in ("held6", "epadb_test")
            )
        details[a] = {
            "metrics_pct_or_fa": brief(e),
            "gates": ok,
            "insertion_f1_delta": idelta,
            "strict_f1_delta": strict_delta,
            "swaps_relative_reduction": swaps,
            "gap_ap": {k: e[k]["gap_ap"] for k in L2},
            "binary_strict_gap": {
                k: e[k]["binary"]["f1"] - e[k]["strict"]["f1"] for k in L2
            },
            "timit_gap_fa": e["timit_core"]["fa_gap"],
        }
    gates = details["gapso_joint"]["gates"]
    go = all(gates.values())
    return write_result(
        out,
        {
            "experiment": "EXP-3",
            "status": "GO" if go else "NO_GO",
            "gates": gates,
            "promoted": str(out / "gapso_joint") if go else None,
            "arms": details,
            "metrics_pct_or_fa": details["gapso_joint"]["metrics_pct_or_fa"],
        },
    )


def dev_oof_inputs(dev, idx):
    t = set(FOLDS[idx])
    c = set(FOLDS[(idx + 1) % len(FOLDS)])
    return (
        [r for r in dev if identity(r) not in t | c],
        [r for r in dev if identity(r) in c],
        [r for r in dev if identity(r) in t],
    )


def temp_diagnostic(
    out, head, norm, spec, policy, cal, rows, run, device, kernel, *, saa_cal
):
    """Separate sealed before-temperature and SAA18-transfer tracks. No Eval fit."""
    data = features_from_cache(cal, spec, run / "frames", kernel, "temp_cal")
    logits = infer_logits(head, cal, data, norm, device, 16)
    if any(r["source"].startswith("l2arctic") for r in cal):
        raise ValueError("4b Calibrate overlaps final DEV-18 Fit")
    raw = choose_policy(cal, logits, replace(spec, temperature=False))
    track = out / "before_temperature"
    track.mkdir()
    save_json(
        track / "CHECKPOINT_CARD.json",
        {
            "base_card_sha256": sha_file(out / "CHECKPOINT_CARD.json"),
            "policy": raw,
            "policy_source": "4b Calibrate; no temperature scaling",
        },
    )
    rec = pred_batches(head, norm, spec, raw, rows, run / "frames", device, kernel)
    seal_predictions(track, rec, sha_file(track / "CHECKPOINT_CARD.json"))
    unscaled, _ = evaluate_seal(track, run / "splits")
    save_json(track / "RESULTS.json", {"eval": unscaled})
    del data, logits
    ss = [row for row in rows if row["cohort"] == "saa27"]
    # Separate Calibrate-only SAA18 operating point; never used for the main model.
    cc = [r for r in cal if r["is_native_l1"]] + saa_cal
    data = features_from_cache(cc, spec, run / "frames", kernel, "saa18_cal")
    sc = choose_policy(cc, infer_logits(head, cc, data, norm, device, 16), spec)
    track = out / "saa18_sensitivity"
    track.mkdir()
    save_json(
        track / "CHECKPOINT_CARD.json",
        {
            "base_card_sha256": sha_file(out / "CHECKPOINT_CARD.json"),
            "policy": sc,
            "policy_source": "SAA18 + clean L1; not primary deployment thresholds",
        },
    )
    rec = pred_batches(head, norm, spec, sc, ss, run / "frames", device, kernel)
    seal_predictions(track, rec, sha_file(track / "CHECKPOINT_CARD.json"))
    gold = [
        r for r in admitted(run / "splits", "eval_gold.jsonl") if r["cohort"] == "saa27"
    ]
    sensitivity, _ = evaluate_seal(track, run / "splits", gold_rows=gold)
    save_json(track / "RESULTS.json", {"eval": sensitivity})
    return {k: unscaled[k]["target_ece"] for k in COHORTS}, sensitivity["saa27"][
        "strict"
    ]["f1"]


def run_exp4(run, device, seeds=(20261121, 20261122, 20261123), epochs=14, batch=16):
    parent, spec = inherited(run, ["exp1_window", "exp2_l1_canon", "exp3_gap_so"])
    spec = replace(spec, corpus_balance=True, temperature=True)
    out = run / "exp4_multiseed"
    out.mkdir()
    kernel = build_kernel(run / "ctc_delete.so") if spec.ollr else None
    fit, cal = training_rows(run, spec.l1_canon)
    dev = [r for r in fit + cal if r["source"] == "l2arctic_dev18"]
    f0 = [r for r in fit if r["source"] != "l2arctic_dev18"]
    c0 = [r for r in cal if r["source"] != "l2arctic_dev18"]
    novel = extended_anchors(run, [*f0, *dev], c0, spec.l1_canon)
    f1 = [*dev, *f0, *novel]
    eval_rows = admitted(run / "splits", "eval_prompts.jsonl")
    guard_split(f1, c0, [r for r in eval_rows if r["cohort"] != "spontaneous"])
    # Use only the already admitted SAA-18 development labels, not the SAA-27 Eval labels.
    s18 = []
    for r in read_jsonl(SAA27_GOLD.parent / "calibrate_gold.jsonl"):
        s18.append(
            {
                "cohort": "saa18",
                "source": "saa18",
                "uid": r["uid"],
                "speaker": str(r["speaker_id"]).upper(),
                "audio_path": str(ROOT / r["audio_path"]),
                "canonical_phones": [
                    normalize_canonical_phone(p) for p in r["canonical_phones"]
                ],
                "gold_targets": r["gold_targets"],
                "gold_gaps": r["gold_gaps"],
                "is_native_l1": False,
            }
        )
    bind_audio(s18, {})
    guard_split(f1, s18, [r for r in eval_rows if r["cohort"] != "spontaneous"])
    if any(split_overlap(c0, s18).values()):
        raise ValueError("SAA18 overlaps primary Calibrate")
    save_jsonl(out / "final_fit.jsonl", f1)
    save_jsonl(out / "final_cal.jsonl", c0)
    save_json(
        out / "DESIGN.json",
        {
            "parent": str(parent),
            "spec": spec.as_dict(),
            "fit": summary(f1),
            "cal": summary(c0),
            "fit_sha256": sha_file(out / "final_fit.jsonl"),
            "cal_sha256": sha_file(out / "final_cal.jsonl"),
            "additional_fit_speakers": len(novel),
            "additional_fit_reason": "Same speaker count (401) in both frozen 4b and LoRA 5; one previously unused TRAIN WAV/speaker. No addition to OOF.",
            "oof_folds": FOLDS,
            "oof_v2_note": "Archived V2 saw DEV-18 in Fit/Cal, so is not an OOF comparator there.",
            "temperatures": "CAL L2 global NLL fallback for unseen domains, separate per-domain temperature only with CAL support.",
            "speaker_seed_bootstrap": "Separate paired-speaker CIs per seed; SD across the three fixed seeds.",
        },
    )
    backbone = read_json(run / "splits/PREFLIGHT.json")["backbone"]
    save_json(
        out / "FRAME_RECEIPT.json",
        fill_cache(f1 + c0 + s18, run / "frames", device, backbone),
    )

    all_predictions = []
    all_gold = []
    oof_root = out / "oof"
    oof_root.mkdir()
    oof_masks = []
    for i in range(6):
        tr, cc, te = dev_oof_inputs(dev, i)
        ff = [*tr, *f0]
        ca = [*cc, *c0]
        guard_split(ff, ca, te)
        tagged = [dict(r, cohort="dev18_oof") for r in te]
        pp = [only_prompt(r) for r in tagged]
        gg = [
            {k: r[k] for k in ("cohort", "uid", "gold_targets", "gold_gaps")}
            for r in tagged
        ]
        path = oof_root / f"fold_{i}"
        c = train_arm(
            path, ff, ca, spec, run, device, seeds[0], epochs, batch, kernel=kernel
        )
        head, norm, spec, policy = load_checkpoint(path, device)
        pre = pred_batches(head, norm, spec, policy, pp, run / "frames", device, kernel)
        seal_predictions(path, pre, sha_file(path / "CHECKPOINT_CARD.json"))
        all_predictions.extend(pre)
        all_gold.extend(gg)
        del head
        release_cuda(device)
        oof_masks.append(
            {
                "fold": i,
                "fit": summary(ff),
                "cal": summary(ca),
                "eval": summary(te),
                "checkpoint_sha256": c["sha256"],
            }
        )
    save_json(oof_root / "CHECKPOINT_CARD.json", {"folds": oof_masks})
    seal_predictions(
        oof_root, all_predictions, sha_file(oof_root / "CHECKPOINT_CARD.json")
    )
    oof, ci = evaluate_seal(
        oof_root, run / "splits", {"r0_oof": load_old_dev_r0()}, all_gold
    )
    save_json(oof_root / "RESULTS.json", {"eval": oof, "bootstrap": ci})

    fit_summ = {}
    series = []
    parent_cmp = comparisons(run, parent)
    for seed in seeds:
        path = out / f"seed{seed}"
        card = train_arm(
            path, f1, c0, spec, run, device, seed, epochs, batch, kernel=kernel
        )
        # Cal locked in the arm checkpoint before making final main Eval decisions.
        r = evaluate_arm(path, run, device, parent_cmp, kernel=kernel, latency=True)
        e = r["eval"]
        series.append(e)
        h, n, s, p = load_checkpoint(path, device)
        pre_cal, saa_f1 = temp_diagnostic(
            path, h, n, s, p, c0, eval_rows, run, device, kernel, saa_cal=s18
        )
        del h
        release_cuda(device)
        fit_summ[str(seed)] = {
            "checkpoint_sha256": card["sha256"],
            "metrics_pct_or_fa": brief(e),
            "before_temperature_target_ece": pre_cal,
            "after_temperature_target_ece": {k: e[k]["target_ece"] for k in COHORTS},
            "saa18_cal_f1": saa_f1,
            "held6_aba_ebvs_fp": {
                x: e["held6"]["by_speaker"][x]["strict"]["fp"] for x in ("ABA", "EBVS")
            },
        }
    levels = {k: [e[k]["strict"]["f1"] * 100 for e in series] for k in L2}
    levels["timit_core"] = [e["timit_core"]["fa_per_utt"] for e in series]
    levels[CONTROL] = [e[CONTROL]["recall"] * 100 for e in series]
    summ = {
        k: {
            "mean": float(np.mean(v)),
            "sample_sd": float(np.std(v, ddof=1)),
            "min": min(v),
            "max": max(v),
        }
        for k, v in levels.items()
    }
    op = {"held6": 67.5, "spontaneous": 64.5, "epadb_test": 69.5, "saa27": 46.5}
    v2 = read_json(run / "v2_keyed/RESULTS.json")["eval"]
    r0 = {
        "held6": 66.746743,
        "spontaneous": 61.5456,
        "epadb_test": 58.946,
        "saa27": 43.9382,
    }
    ok = {
        k: summ[k]["mean"] >= lim
        and summ[k]["mean"] > max(r0[k], 100 * v2[k]["strict"]["f1"])
        for k, lim in op.items()
    }
    ok.update(
        timit_core=summ["timit_core"]["mean"] <= 0.6
        and summ["timit_core"]["mean"] < v2["timit_core"]["fa_per_utt"],
        three_seed_sd=all(summ[k]["sample_sd"] < 0.6 for k in L2),
        control_recall=summ[CONTROL]["min"] >= 94.0,
        control_ap=all(e[CONTROL]["ap"] >= 0.95 for e in series),
    )
    go = all(ok.values())
    return write_result(
        out,
        {
            "experiment": "EXP-4",
            "status": "GO" if go else "NO_GO",
            "gates": ok,
            "promoted": str(out / f"seed{seeds[0]}") if go else None,
            "frozen_lora_comparator": str(out / f"seed{seeds[0]}"),
            "oof": brief(oof),
            "oof_ci_vs_r0": ci,
            "seeds": fit_summ,
            "mean_sd": summ,
            "metrics_pct_or_fa": {k: round(v["mean"], 4) for k, v in summ.items()},
        },
    )
