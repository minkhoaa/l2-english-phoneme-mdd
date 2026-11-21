"""CPU recount and paired EOS-only comparison from independently sealed control arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import CONTROL, SEED, admitted, read_json, save_json, sha_file
from .repair_prompts import DEFAULT_REPAIR, DEFAULT_RUN, ensure_separate
from .scoring import (
    KEY,
    bootstrap,
    brief,
    evaluate_seal,
    gold_frame,
    keyed_merge,
    read_predictions,
    split_calibrated_summary,
)

MODES = ("legacy-eos-control", "fixed")


def compare_gap_slices(legacy, fixed):
    d = keyed_merge(
        legacy[KEY + ["speaker", "gold_label", "call"]],
        fixed[KEY + ["call"]].rename(columns={"call": "fixed_call"}),
    )
    g = d[d.domain.eq("gap") & d.cohort.ne(CONTROL)].copy()
    g["position"] = np.where(
        g["index"].eq(g.groupby(["cohort", "uid"])["index"].transform("max")),
        "eos",
        "other",
    )
    g = g[g.gold_label.ne("unknown")]
    result = {}
    for (name, position), x in g.groupby(["cohort", "position"], sort=True):
        stats = {}
        for arm, call in (("old", "call"), ("fixed", "fixed_call")):
            p = x[call].eq("I")
            y = x.gold_label.eq("I")
            stats[arm] = {
                "tp": int((p & y).sum()),
                "fp": int((p & ~y).sum()),
                "fn": int((~p & y).sum()),
            }
        result.setdefault(name, {})[position] = {
            "gaps": len(x),
            "changed_calls": int(x.call.ne(x.fixed_call).sum()),
            **stats,
        }
    return result


def gap_bootstrap(legacy, fixed):
    d = keyed_merge(
        legacy[KEY + ["speaker", "gold_label", "call"]],
        fixed[KEY + ["call"]].rename(columns={"call": "fixed_call"}),
    )
    d = d[
        d.domain.eq("gap")
        & d.gold_label.ne("unknown")
        & ~d.cohort.isin([CONTROL, "timit_core"])
    ]
    result = {}
    rng = np.random.default_rng(SEED)
    for name, group in d.groupby("cohort"):
        gold = group.gold_label.eq("I")
        old = group.call.eq("I")
        new = group.fixed_call.eq("I")
        n = group.assign(
            tp_old=gold & old, tp_new=gold & new, old=old, new=new, gold=gold
        )
        counts = (
            n.groupby("speaker")[["tp_old", "tp_new", "old", "new", "gold"]]
            .sum()
            .to_numpy()
        )
        take = rng.integers(len(counts), size=(5000, len(counts)))

        def diff(v):
            a, b, c, d, e = v.T
            return 200 * (b / np.maximum(1, d + e) - a / np.maximum(1, c + e))

        delta = diff(counts[take].sum(1))
        result[name] = {
            "delta_pp": float(diff(counts.sum(0, keepdims=True))[0]),
            "ci95_pp": np.quantile(delta, [0.025, 0.975]).tolist(),
            "n_speakers": len(counts),
        }
    return result


def replay_common_policy(frame, policy):
    """Use the preselected *legacy arm's CAL* point on both, never an Eval sweep."""
    out = frame.copy()
    t = out.domain.eq("target")
    out.loc[~t, "call"] = np.where(out.loc[~t, "score"] >= policy["tau_g"], "I", "N")
    vals = out.loc[t]
    out.loc[t, "call"] = np.where(
        vals.score >= policy["tau_t"], np.where(vals.p_o > vals.p_s, "O", "S"), "C"
    )
    return out


def compare(pair, split):
    matched = read_json(pair / "MATCH_RECEIPT.json")
    if matched["match"] is not True:
        raise ValueError("Matched training receipt required")
    scores, frames, cards = {}, {}, {}
    gold = gold_frame(admitted(split, "eval_gold.jsonl"))
    for mode in MODES:
        d = pair / mode
        c = read_json(d / "CHECKPOINT_CARD.json")
        if (
            c["mode"] != mode
            or any(c[k] != v for k, v in matched["matched"].items())
            or c["repair_preflight_sha256"] != sha_file(split / "PREFLIGHT.json")
        ):
            raise ValueError(f"Invalid training receipt for {mode}")
        if (
            c["policy"] != matched["policy_locked"][mode]
            or sha_file(d / c["checkpoint"]) != c["sha256"]
        ):
            raise ValueError(f"Checkpoint/policy not matched for {mode}")
        e = pair / f"eval_{mode}"
        ec = read_json(e / "CHECKPOINT_CARD.json")
        if (
            ec["training_card_sha256"] != sha_file(d / "CHECKPOINT_CARD.json")
            or ec["policy"] != c["policy"]
            or ec["sha256"] != c["sha256"]
        ):
            raise ValueError(f"Invalid evaluation receipt for {mode}")
        scores[mode], _ = evaluate_seal(e, split)
        if scores[mode] != read_json(e / "RESULTS.json")["eval"]:
            raise ValueError(f"CPU recount mismatch: {mode}")
        frames[mode] = keyed_merge(read_predictions(e), gold)
        cards[mode] = c
    old, fixed = (frames[m] for m in MODES)
    old_policy = cards[MODES[0]]["policy"]
    if any(
        c["policy"]["alpha_realized"]
        or c["policy"]["temperature"] != {"default": [1.0, 1.0]}
        for c in cards.values()
    ):
        raise ValueError("Common-threshold track requires raw probabilities")
    shared = replay_common_policy(fixed, old_policy)
    if not replay_common_policy(old, old_policy).call.eq(old.call).all():
        raise ValueError("Legacy arm seal does not mirror its CAL policy")
    return {
        "scope": "diagnostic_single_seed_eos_only_both_use_corrected_timit_inputs",
        "comparison_source_sha256": sha_file(__file__),
        "cpu_recount_match": True,
        "matched": matched["matched"],
        "heads": {
            m: {
                "policy": c["policy"],
                "best_epoch": c["best_epoch"],
                "checkpoint_sha256": c["sha256"],
            }
            for m, c in cards.items()
        },
        "primary": {
            "eval": scores,
            "strict_pct_or_fa": {m: brief(s) for m, s in scores.items()},
            "fixed_minus_old_strict_or_timit_fa": bootstrap(fixed, old),
            "fixed_minus_old_i_f1": gap_bootstrap(old, fixed),
            "gap_localization": compare_gap_slices(old, fixed),
        },
        "secondary_common_legacy_cal_thresholds": {
            "policy": old_policy,
            "fixed": split_calibrated_summary(shared),
            "fixed_minus_old_strict_or_timit_fa": bootstrap(shared, old),
            "fixed_minus_old_i_f1": gap_bootstrap(old, shared),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, default=DEFAULT_REPAIR / "splits")
    p.add_argument("--pair", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ensure_separate(DEFAULT_RUN, a.output)
    result = compare(a.pair, a.splits)
    save_json(a.output, result)
    print(result["primary"]["strict_pct_or_fa"], flush=True)
    print(a.output)


if __name__ == "__main__":
    main()
