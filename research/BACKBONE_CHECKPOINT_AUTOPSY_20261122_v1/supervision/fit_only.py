"""Build and validate an in-memory Fit-only realized-ID intervention; never edit the bound split.

For clean_11 only. The builder does not read Cal, AUX.json, Eval or protected labels.
The only on-disk output is a guarded patch, never a replacement manifest.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from ..inputs import (
    PHONE_TO_ID,
    PROPOSED_CLEAN_SPLIT,
    PROTECTED_SEALED_SPEAKERS,
    RAW,
    RUN,
    SOURCES,
    TAG,
    normalize_canonical_phone,
    read_json,
    sha_file,
)

FIT_SHA = "ac4c0d02eb3757a3b94c24ca79a3030c26577e9091599333157de9be50515fb6"


# SHA of canonical jsonl for in-memory treatment: sorted object keys, ordered rows, UTF-8, newline.
def records_sha(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(
            (
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode()
        )
    return h.hexdigest()


def apply_changes(fit, edits):
    """Return detached edited ID lists; reject wrong slots, sources and stale old values."""
    by = {r["uid"]: i for i, r in enumerate(fit)}
    if len(by) != len(fit):
        raise ValueError("duplicate Fit uid")
    output = list(fit)
    seen = set()
    changed = set()
    for e in edits:
        uid, i = e["uid"], e["slot_index"]
        key = (uid, i)
        if uid not in by or key in seen:
            raise ValueError("unknown or duplicate patch key")
        r = fit[by[uid]]
        new, old = e["new"], e["old"]
        if (
            r["source"] != "l2arctic_dev18"
            or r["speaker"].upper() in PROTECTED_SEALED_SPEAKERS
            or r["audio_sha256"] != e["wav_sha256"]
            or not 0 <= i < len(r["canonical_phones"])
            or r["canonical_phones"][i] != e["canonical"]
            or r["gold_targets"][i] != "S"
            or r["realized_ids"][i] != old
            or not 0 <= new < 39
            or old == new
        ):
            raise ValueError("patch does not match immutable S slot")
        if uid not in changed:
            output[by[uid]] = {**r, "realized_ids": list(r["realized_ids"])}
            changed.add(uid)
        output[by[uid]]["realized_ids"][i] = new
        seen.add(key)
    return output


def protected_fields_sha(rows):
    return records_sha(
        [{k: v for k, v in r.items() if k != "realized_ids"} for r in rows]
    )


def load_fit(patch_path):
    """Future ablation loader: pinned `fit.jsonl` only; returns treatment rows, never writes them."""
    path = PROPOSED_CLEAN_SPLIT / "fit.jsonl"
    patch = read_json(patch_path)
    if (
        patch.get("role") != "fit"
        or sha_file(path) != FIT_SHA
        or patch["input_file_sha256"] != FIT_SHA
    ):
        raise ValueError("not the bound clean_11 Fit")
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    if (
        records_sha(rows) != patch["input_records_sha256"]
        or len(rows) != 3681
        or len(patch["changes"]) != 197
    ):
        raise ValueError("unexpected treatment roster")
    fixed = apply_changes(rows, patch["changes"])
    digest = protected_fields_sha(rows)
    if (
        records_sha(fixed) != patch["treatment_records_sha256"]
        or digest != protected_fields_sha(fixed)
        or digest != patch["untouched_fields_sha256"]
    ):
        raise ValueError("treatment digest mismatch")
    return fixed


def build():
    path = PROPOSED_CLEAN_SPLIT / "fit.jsonl"
    assert (
        sha_file(path)
        == FIT_SHA
        == read_json(RUN / "RUN_BINDINGS.json")["preflight"]["fit_sha256"]
    )
    with path.open() as f:
        fit = [json.loads(line) for line in f]
    assert len(fit) == 3681
    admitted = {r["uid"]: r for r in fit if r["source"] == "l2arctic_dev18"}
    assert len(admitted) == 1042 and all(
        r["speaker"].upper() not in PROTECTED_SEALED_SPEAKERS for r in fit
    )
    edits = []
    seen = {}
    source_files = {}
    # These are *legacy* file names, not clean_11 roles. Route only by the
    # SHA-pinned Fit UID allowlist before parsing any source annotation.
    for role in ("train", "dev_search", "dev_cal", "dev_replication"):
        n = 0
        with (RAW / (role + ".jsonl")).open() as f:
            for line in f:
                match = re.search(
                    r'"utterance_id"\s*:\s*"([A-Z]+_arctic_[ab]\d+)"',
                    line.split('"target_phones"', 1)[0],
                )
                if not match or match[1] not in admitted:
                    continue
                uid = match[1]
                assert uid not in seen
                seen[uid] = hashlib.sha256(line.encode()).hexdigest()
                r = admitted[uid]
                z = json.loads(line)
                n += 1
                assert (
                    z["speaker_id"] == r["speaker"]
                    and (SOURCES / z["audio_path"]).resolve()
                    == Path(r["audio_path"]).resolve()
                    and sha_file(r["audio_path"]) == r["audio_sha256"]
                )
                slots = z["target_phones"]
                assert len(slots) == len(r["gold_targets"]) == len(r["realized_ids"])
                for i, s in enumerate(slots):
                    ph = normalize_canonical_phone(s["target_phone"])
                    assert (
                        i == s["slot_idx"]
                        and ph == r["canonical_phones"][i]
                        and r["gold_targets"][i] == TAG[s["label"]]
                    )
                    if s["label"] != "SUBSTITUTION":
                        continue
                    old = r["realized_ids"][i]
                    new = PHONE_TO_ID[normalize_canonical_phone(s["realized_phone"])]
                    if old != new:
                        edits.append(
                            {
                                "uid": uid,
                                "slot_index": i,
                                "canonical": ph,
                                "wav_sha256": r["audio_sha256"],
                                "old": old,
                                "new": new,
                            }
                        )
        source_files[role] = n
    assert seen.keys() == admitted.keys()
    edits.sort(key=lambda e: (e["uid"], e["slot_index"]))
    assert len(edits) == 197
    old_digest = records_sha(fit)
    treatment = apply_changes(fit, edits)
    prot = protected_fields_sha(fit)
    assert protected_fields_sha(treatment) == prot and records_sha(fit) == old_digest
    assert (
        sum(
            a != b
            for r, s in zip(fit, treatment, strict=True)
            for a, b in zip(r["realized_ids"], s["realized_ids"], strict=True)
        )
        == 197
    )
    return {
        "role": "fit",
        "scope": "clean_11 isolated realized-ID ablation; not applied to campaign",
        "input_file_sha256": FIT_SHA,
        "input_records_sha256": old_digest,
        "treatment_records_sha256": records_sha(treatment),
        "untouched_fields_sha256": prot,
        "rows": len(fit),
        "reference_counts": source_files,
        "reference_digest": hashlib.sha256(
            json.dumps(seen, sort_keys=True).encode()
        ).hexdigest(),
        "code_sha256": sha_file(__file__),
        "changes": edits,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    dest = p.parse_args().out.resolve()
    if (
        dest.exists()
        or dest.is_relative_to(RUN.parents[1])
        or dest.is_relative_to(PROPOSED_CLEAN_SPLIT.resolve())
    ):
        raise FileExistsError("use a fresh location outside the split and campaign")
    patch = build()
    with dest.open("x") as f:
        json.dump(patch, f, indent=2, allow_nan=False)
        f.write("\n")
    load_fit(dest)
    print(
        f"Verified {len(patch['changes'])} Fit-only changes; no manifest written: {dest}"
    )
