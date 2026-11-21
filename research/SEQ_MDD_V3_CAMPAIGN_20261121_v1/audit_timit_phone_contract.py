"""Check the TIMIT model-input vocabulary independently of the PER reference vocabulary."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    TIMIT_GOLD,
)
from research.KOEL_ABC_FROZEN_EVAL_20261119_v1.build_manifests import (
    TIMIT_CANONICAL_OOV,
    load_cmudict,
)
from research.KOEL_FULL_CONFIG_BENCHMARK_20261120_v1.run_per_and_latency import (
    timit_recount,
)

from .audit_training_campaign import ARMS
from .data import (
    CONTROL,
    HERE,
    ROOT,
    admitted,
    read_json,
    read_jsonl,
    save_json,
    sha_file,
)
from .probe_training_contracts import DEVICE, load_arm
from .scoring import KEY, gold_frame, keyed_merge, read_predictions
from .training import features_from_cache, infer_logits, probabilities


def fcount(rows, logits, p, masks):
    count = Counter()
    for r, l, m in zip(rows, logits, masks, strict=True):
        target, gap = probabilities(l, r, p)
        flag = target[:, 1:].sum(1) >= p["tau_t"]
        s = target[:, 1] > target[:, 2]
        count.update(
            target_fa=int(flag.sum()),
            gap_fa=int((gap[:, 1] >= p["tau_g"]).sum()),
            changed_phone_fa=int((flag & m).sum()),
            unchanged_phone_fa=int((flag & ~m).sum()),
            changed_phone_s=int((flag & m & s).sum()),
        )
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run = a.run_dir.resolve()
    split = run / "splits"
    timit_gold_sha = read_json(split / "PREFLIGHT.json")["source_sha256"][
        str(TIMIT_GOLD.relative_to(ROOT))
    ]
    if sha_file(TIMIT_GOLD) != timit_gold_sha:
        raise ValueError("TIMIT gold changed")
    original = {r["uid"]: r for r in read_jsonl(TIMIT_GOLD)}
    rows = [
        r for r in admitted(split, "eval_prompts.jsonl") if r["cohort"] == "timit_core"
    ]
    from .prepare_l1_canonical_anchor import CMU

    expected = read_json(run / "manifests/l1_canonical_anchor_audit.json")[
        "cmudict_sha256"
    ]
    if sha_file(CMU) != expected:
        raise ValueError("CMUdict drift")
    cmu = load_cmudict(CMU)
    mapping = timit_recount.parse_map()
    restored = []
    pairs = Counter()
    masks = []
    affected_rows = 0
    for r in rows:
        src = []
        for w in original[r["uid"]]["words"]:
            src.extend(cmu[w] if w in cmu else TIMIT_CANONICAL_OOV[w])
        raw = [p.upper() for p in src if mapping[p] not in (None, "sil")]
        folded = r["canonical_phones"]
        if (
            list(map(str.upper, timit_recount.project_reference(src, mapping)))
            != folded
        ):
            raise ValueError("Reconstructed prompt does not match recorded fold")
        m = np.array(raw) != np.array(folded)
        masks.append(m)
        affected_rows += int(m.any())
        pairs.update(f"{b}_to_{a}" for a, b, k in zip(raw, folded, m, strict=True) if k)
        restored.append(dict(r, canonical_phones=raw))
    gold = gold_frame(admitted(split, "eval_gold.jsonl"))
    report = {}
    for name in ("v2", "canon"):
        model, norm, spec, policy = load_arm(run, name)
        sealed = read_predictions(run / ARMS[name])
        gold_t = keyed_merge(sealed, gold)
        so = gold_t[
            (gold_t.cohort == CONTROL)
            & (gold_t.domain == "target")
            & (gold_t.gold_label == "S")
        ]
        nt = (
            sealed[(sealed.cohort == "timit_core") & (sealed.domain == "target")]
            .set_index(KEY)
            .sort_index()
        )
        ct = Counter()
        for r, raw, m in zip(rows, restored, masks, strict=True):
            old, true = r["canonical_phones"], raw["canonical_phones"]
            t = nt.loc[("timit_core", r["uid"], "target"), :].sort_index()
            if t.canonical_phone.tolist() != old:
                raise ValueError("Key/phone mismatch in sealed scores")
            for b, c, is_edit, flag in zip(
                old, true, m, t.call.isin(["S", "O"]), strict=True
            ):
                if b == "AA":
                    ct[f"{c}_n"] += 1
                    ct[f"{c}_fa"] += int(flag)
                if is_edit:
                    ct[f"{b}_to_{c}_n"] += 1
                    ct[f"{b}_to_{c}_fa"] += int(flag)
        # Label-free frozen inference on exactly the same cached frame inputs; reuse the same locked policy.
        old_count = read_json(run / ARMS[name] / "RESULTS.json")["eval"]["timit_core"]
        orig = {
            "target_fa": old_count["fa_target"],
            "gap_fa": old_count["fa_gap"],
            "changed_phone_fa": sum(ct[f"{pair}_fa"] for pair in pairs),
        }
        ab = {}
        for k, tasks in [("folded_prompt", rows), ("unfolded_prompt", restored)]:
            ll = features_from_cache(tasks, spec, run / "frames", None, name + "_" + k)
            ab[k] = fcount(
                tasks, infer_logits(model, tasks, ll, norm, DEVICE, 1), policy, masks
            )
        report[name] = {
            "sealed_original": orig,
            "sealed_slices_aa": ct,
            "cpu_same_policy": ab,
            "timit_controls_gold_s_call_counts": {
                k: int(v) for k, v in so.call.value_counts().items()
            },
        }
    save_json(
        a.output,
        {
            "protocol": "no tuning, same cached frames, fixed models and locked thresholds; diagnostic only",
            "goal_commit": read_json(run / "CAMPAIGN_STATUS.json")["goal_commit"],
            "source_sha256": sha_file(__file__),
            "timit_gold_sha256": timit_gold_sha,
            "cmudict_sha256": expected,
            "n_utts": len(rows),
            "n_utts_with_affected_input": affected_rows,
            "folded_input_positions_by_pair": pairs,
            "arms": report,
        },
    )
    print(a.output)


if __name__ == "__main__":
    main()
