"""Versioned MDD prompt manifests; PER gold and historical manifests stay unchanged."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    TIMIT_GOLD,
)
from research.KOEL_ABC_FROZEN_EVAL_20261119_v1.build_manifests import (
    TIMIT_CANONICAL_OOV,
    load_cmudict,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    PHONE_TO_ID,
)

from .data import (
    HERE,
    ROOT,
    admitted,
    corruption_control,
    identity,
    only_prompt,
    read_json,
    read_jsonl,
    save_json,
    save_jsonl,
    sha_file,
)
from .prepare_l1_canonical_anchor import CMU

REPAIR_ID = "v2_prompt_eos_repair_v1"
DEFAULT_RUN = HERE / "runs/campaign_d5f8e89"
DEFAULT_REPAIR = HERE / "runs/v2_prompt_eos_repair_v1"
# Used only to verify the historical slot correspondence. Never used for model input.
LEGACY_FOLD = {"AO": "AA", "ZH": "SH"}


def mdd_prompt(words, dictionary, overrides):
    """Derive model tokens from original word prompts, never the PER projection."""
    phones = []
    for word in words:
        key = word.casefold()
        pronunciation = dictionary.get(key, overrides.get(key))
        if not pronunciation:
            raise ValueError(f"Missing TIMIT prompt pronunciation: {word!r}")
        for p in pronunciation:
            ph = p.upper().rstrip("012")
            if ph not in PHONE_TO_ID:
                raise ValueError(f"Unsupported MDD phone {p!r} for {word!r}")
            phones.append(ph)
    if not phones:
        raise ValueError("Empty TIMIT pronunciation")
    return phones


def repair_core(row, source, dictionary, overrides):
    if not (
        row["uid"] == source["uid"]
        and identity(row) == str(source["speaker_id"]).strip().upper()
        and row["audio_sha256"] == source["audio_sha256"]
    ):
        raise ValueError(f"TIMIT Core UID/speaker/WAV mismatch: {row['uid']}")
    phones = mdd_prompt(source["words"], dictionary, overrides)
    old = row["canonical_phones"]
    # Fail closed if the new input would silently invalidate old slot keys or native control lengths.
    if (
        list(map(str.upper, source["canonical39_ref"])) != old
        or [LEGACY_FOLD.get(p, p) for p in phones] != old
    ):
        raise ValueError(f"Unexpected legacy TIMIT slot correspondence: {row['uid']}")
    return dict(row, canonical_phones=phones)


def ensure_separate(run, out):
    run, out = run.resolve(), out.resolve()
    if out == run or run in out.parents or out in run.parents:
        raise ValueError(
            "Repair outputs must not be inside or replace the archived run"
        )


def _gold(row):
    return {k: row[k] for k in ("cohort", "uid", "gold_targets", "gold_gaps")}


def _key(row):
    return (row["cohort"], row["uid"])


def prepare(run, out):
    """Preserve Fit/Cal and every non-TIMIT input; write only a new manifest version."""
    ensure_separate(run, out)
    old = run / "splits"
    prior = read_json(old / "PREFLIGHT.json")
    if out.exists():
        raise FileExistsError(out)
    orig_sha = prior["source_sha256"][str(TIMIT_GOLD.relative_to(ROOT))]
    cmu_sha = read_json(run / "manifests/l1_canonical_anchor_audit.json")[
        "cmudict_sha256"
    ]
    if sha_file(TIMIT_GOLD) != orig_sha or sha_file(CMU) != cmu_sha:
        raise ValueError("TIMIT/CMUdict input differs from the archived evidence")
    source = {r["uid"]: r for r in read_jsonl(TIMIT_GOLD)}
    dictionary = load_cmudict(CMU)
    orig_prompt = admitted(old, "eval_prompts.jsonl")
    orig_gold = admitted(old, "eval_gold.jsonl")
    original_by_key = {_key(r): r for r in orig_prompt}
    gold_by_key = {_key(r): r for r in orig_gold}
    if (
        len(original_by_key) != len(orig_prompt)
        or len(gold_by_key) != len(orig_gold)
        or original_by_key.keys() != gold_by_key.keys()
    ):
        raise ValueError("Archived prompt/gold keys are not a bijection")
    core, counts = [], Counter()
    for r in orig_prompt:
        if r["cohort"] != "timit_core":
            continue
        row = repair_core(r, source[r["uid"]], dictionary, TIMIT_CANONICAL_OOV)
        counts.update(
            f"{a}_to_{b}"
            for a, b in zip(r["canonical_phones"], row["canonical_phones"], strict=True)
            if a != b
        )
        g = gold_by_key[_key(row)]
        if g["gold_targets"] != ["C"] * len(row["canonical_phones"]) or g[
            "gold_gaps"
        ] != ["N"] * (len(row["canonical_phones"]) + 1):
            raise ValueError("Unexpected label on clean TIMIT Core")
        core.append(row)
    if set(source) != {r["uid"] for r in core}:
        raise ValueError("TIMIT source/WAV roster mismatch")
    controls = corruption_control(core)
    new_rows = {_key(r): only_prompt(r) for r in [*core, *controls]}
    for r in controls:
        if gold_by_key[_key(r)] != _gold(r):
            raise ValueError("Control UID or perturbation positions changed")
    for key, r in new_rows.items():
        old_r = original_by_key[key]
        if {k: v for k, v in r.items() if k != "canonical_phones"} != {
            k: v for k, v in old_r.items() if k != "canonical_phones"
        }:
            raise ValueError(f"Non-prompt fields changed: {key}")
    prompts = [new_rows.get(_key(r), r) for r in orig_prompt]
    changed_utts = sum(
        a["canonical_phones"] != b["canonical_phones"]
        for a, b in zip(orig_prompt, prompts, strict=True)
        if a["cohort"] == "timit_core"
    )
    changes = {
        "clean_utterances": len(core),
        "clean_utterances_changed": changed_utts,
        "clean_slots_changed": dict(counts),
        "control_utterances": len(controls),
        "control_slots_edited_by_seed42": sum(
            r["gold_targets"].count("S") for r in controls
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    # Only prompts differ. Gold (including the corruption positions) and Fit/Cal bytes are unchanged.
    for name in ("fit.jsonl", "cal.jsonl", "eval_gold.jsonl"):
        payload = (old / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != prior["manifest_sha256"][name]:
            raise ValueError(f"Archived admitted source changed: {name}")
        with (out / name).open("xb") as f:
            f.write(payload)
    save_jsonl(out / "eval_prompts.jsonl", prompts)
    receipt = {
        **prior,
        "repair_id": REPAIR_ID,
        "parent_preflight_sha256": sha_file(old / "PREFLIGHT.json"),
        "manifest_sha256": {p.name: sha_file(p) for p in sorted(out.glob("*.jsonl"))},
        "cmudict_sha256": cmu_sha,
        "repair_prompts_source_sha256": sha_file(__file__),
        "timit_prompt_changes": changes,
        "note": "MDD prompt inventory preserves AO/ZH. PER gold untouched. Corrected control uses the same UIDs and slot positions, but the replacement phones can change.",
    }
    save_json(out / "PREFLIGHT.json", receipt)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, required=True, help="New split directory")
    args = parser.parse_args()
    print(prepare(args.legacy_run, args.out), flush=True)


if __name__ == "__main__":
    main()
