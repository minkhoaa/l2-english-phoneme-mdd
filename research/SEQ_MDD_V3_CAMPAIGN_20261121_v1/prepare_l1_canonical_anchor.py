"""TIMIT TRAIN fallback: CTIMIT's published orthographic prompts → pinned CMUdict."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    INTAKE_MANIFEST,
    IPA_TO_ARPA39,
)
from research.KOEL_ABC_FROZEN_EVAL_20261119_v1.build_manifests import load_cmudict
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    PHONE_TO_ID,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    HERE,
    SEED,
    admitted,
    bind_audio,
    identity,
    read_json,
    read_jsonl,
    save_json,
    save_jsonl,
    sha_file,
    split_overlap,
    summary,
    validate_row,
)

CMU = ROOT / "research/PILOT3_PARALLEL_20261118_v1/cmudict.dict"
SOURCE = "https://catalog.ldc.upenn.edu/docs/LDC96S30/PROMPTS.TXT"
SOURCE_SHA = "af5f50cb34601a8cc74b855cb11d629e7ce48901ef8725860d14d5c6a03806b3"
PROMPT = re.compile(r"^(.+?)\s+\(([sS][aAiIxX]\d+)\)\s*$")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+")


def prompt_id(uid):
    return uid.rsplit("_", 1)[-1].upper()


def orthographic_prompts(path):
    if sha_file(path) != SOURCE_SHA:
        raise ValueError("Published prompt table digest mismatch")
    prompts = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        m = PROMPT.fullmatch(line)
        if m:
            uid = m[2].upper()
            if uid in prompts:
                raise ValueError(f"Duplicate TIMIT prompt {uid}")
            prompts[uid] = m[1]
    if len(prompts) != 2342:
        raise ValueError("Prompt table incomplete")
    return prompts


def cmu_phones(text, cmu):
    text = text.lower().replace("\u2019", "'")
    # Non-ASCII letters and digits must not be silently dropped or substituted.
    if any(c.isalpha() and not c.isascii() for c in text):
        return None, ["<non_ascii_word>"]
    words = WORD.findall(text)
    missing = sorted(set(words) - cmu.keys())
    if missing or not words:
        return None, missing or ["<no_words>"]
    ph = [p.upper() for w in words for p in cmu[w]]
    if len(ph) < 6 or not set(ph) <= set(ARPA_39):
        return None, ["<invalid_phones>"]
    return ph, []


def corrupt_phones(phones, seed, uid):
    local = int.from_bytes(hashlib.sha256(f"{seed}:{uid}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(local)
    can = []
    t = []
    g = ["N"]
    r = []
    for i, p in enumerate(phones):
        u = rng.random()
        if u < 0.03 and 0 < i < len(phones) - 1:
            g[-1] = "I"
            continue
        if 0.03 <= u < 0.06:
            can.append(str(rng.choice([x for x in ARPA_39 if x != p])))
            t.append("O")
            g.append("N")
            r.append(39)
        if rng.random() < 0.1:
            can.append(str(rng.choice([x for x in ARPA_39 if x != p])))
            t.append("S")
        else:
            can.append(p)
            t.append("C")
        g.append("N")
        r.append(PHONE_TO_ID[p])
    if all(a == "C" for a in t) and "I" not in g:
        mid = len(t) // 2
        can[mid] = str(rng.choice([x for x in ARPA_39 if x != can[mid]]))
        t[mid] = "S"
    return can, t, g, r


def anchor_row(r, phones, role, *, canonical=True):
    uid = r["uid"]
    source = "timit_train_l1_canon" if canonical else "timit_train_l1_antishortcut"
    if role == "fit":
        can, t, g, ri = corrupt_phones(phones, SEED, uid)
    else:
        can = list(phones)
        t = ["C"] * len(can)
        g = ["N"] * (len(can) + 1)
        ri = [PHONE_TO_ID[p] for p in can]
    result = {
        **r,
        "source": source,
        "canonical_phones": can,
        "gold_targets": t,
        "gold_gaps": g,
        "realized_ids": ri,
        "is_native_l1": role == "cal",
        "anchor_role": role,
        "anchor_prompt_id": prompt_id(uid),
    }
    validate_row(result)
    return result


def timit_intake(run):
    audit = read_json(run / "splits/PREFLIGHT.json")
    if (
        sha_file(INTAKE_MANIFEST)
        != audit["source_sha256"][str(INTAKE_MANIFEST.relative_to(ROOT))]
    ):
        raise ValueError("TIMIT intake changed since preflight")
    return [
        r
        for r in read_jsonl(INTAKE_MANIFEST)
        if r["corpus"] == "TIMIT" and r["split"] == "TRAIN"
    ]


def identity_from_intake(r):
    return {
        "uid": r["uid"],
        "speaker": str(r["speaker_id"]).upper(),
        "audio_path": str(ROOT / r["audio_path"]),
    }


def prepare(run, out, source=None):
    split = run / "splits"
    old_fit = [
        r for r in admitted(split, "fit.jsonl") if r["source"].startswith("timit")
    ]
    old_cal = [
        r for r in admitted(split, "cal.jsonl") if r["source"].startswith("timit")
    ]
    eval_rows = admitted(split, "eval_prompts.jsonl")
    core_ids = {prompt_id(r["uid"]) for r in eval_rows if r["cohort"] == "timit_core"}
    source = source or (run / "l1_sources/LDC_CTIMIT_PROMPTS.TXT")
    texts = orthographic_prompts(source)
    cmu = load_cmudict(CMU)
    raw = defaultdict(list)
    for rec in timit_intake(run):
        raw[str(rec["speaker_id"]).upper()].append(rec)
    seen = {r["uid"] for r in [*old_fit, *old_cal]}
    exclusions = []
    replacements = []
    out_rows = {"fit": [], "cal": []}

    def pron(uid):
        p = prompt_id(uid)
        if p in core_ids:
            return None, ["<core_prompt_excluded>"]
        if p not in texts:
            return None, ["<unbound_prompt>"]
        return cmu_phones(texts[p], cmu)

    for role, rows in [("fit", old_fit), ("cal", old_cal)]:
        for row in rows:
            phones, oov = pron(row["uid"])
            if not phones:
                exclusions.append({"uid": row["uid"], "reason": oov})
                for cand in sorted(raw[identity(row)], key=lambda r: r["uid"]):
                    if cand["uid"] in seen:
                        continue
                    phones, _ = pron(cand["uid"])
                    if phones:
                        replacements.append(
                            {
                                "from": row["uid"],
                                "to": cand["uid"],
                                "same_speaker": True,
                                "role": role,
                            }
                        )
                        row = identity_from_intake(cand)
                        seen.add(row["uid"])
                        break
                else:
                    raise ValueError(f"No in-vocabulary alternate for {row['uid']}")
            out_rows[role].append(anchor_row(row, phones, role))
    for role, rows in out_rows.items():
        bind_audio(rows, {})
    pairs = {
        role + "/eval": split_overlap(rows, eval_rows)
        for role, rows in out_rows.items()
    }
    pairs["fit/cal"] = split_overlap(out_rows["fit"], out_rows["cal"])
    if any(v for p in pairs.values() for v in p.values()):
        raise ValueError(f"L1 overlap: {pairs}")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out, [*out_rows["fit"], *out_rows["cal"]])
    ct = Counter(a for r in out_rows["fit"] for a in r["gold_targets"])
    ct.update(a for r in out_rows["fit"] for a in r["gold_gaps"])
    audit = {
        "source": SOURCE,
        "source_sha256": SOURCE_SHA,
        "cmudict_sha256": sha_file(CMU),
        "manifest_sha256": sha_file(out),
        "recipe": "TIMIT TRAIN fallback, CTIMIT prompt text matched by original SA/SX/SI ID (not surface IPA or per-utterance WRD).",
        "annotation_limit": "Canonical compatibility proxy, not manually verified pronunciation in these clips.",
        "fit": summary(out_rows["fit"]),
        "cal": summary(out_rows["cal"]),
        "oov_accepted": 0,
        "excluded": exclusions,
        "replacements": replacements,
        "fit_gold_counts": dict(ct),
        "overlap": pairs,
        "timit_core_prompt_overlap": 0,
        "mocha_excluded": "Existing intake marks the same normalized phonetic segmentation as both canonical and realized.",
    }
    save_json(out.parent / "l1_canonical_anchor_audit.json", audit)
    return out_rows


def load_anchor(path):
    audit = read_json(path.parent / "l1_canonical_anchor_audit.json")
    if sha_file(path) != audit["manifest_sha256"]:
        raise ValueError("Canonical anchor manifest changed")
    rows = read_jsonl(path)
    return {
        role: [r for r in rows if r["anchor_role"] == role] for role in ("fit", "cal")
    }


def extended_anchors(run, fit, cal, canonical, *, at_least=401):
    """Before the final matched 4b/5 comparison, add only previously unused TIMIT TRAIN speakers."""
    all_sp = {identity(r) for r in fit}
    if len(all_sp) >= at_least:
        return []
    ev = admitted(run / "splits", "eval_prompts.jsonl")
    blocked = all_sp | {identity(r) for r in [*cal, *ev]}
    core_ids = {prompt_id(r["uid"]) for r in ev if r["cohort"] == "timit_core"}
    texts = (
        orthographic_prompts(run / "l1_sources/LDC_CTIMIT_PROMPTS.TXT")
        if canonical
        else {}
    )
    cmu = load_cmudict(CMU) if canonical else {}
    out = []
    for r in sorted(
        timit_intake(run), key=lambda r: (str(r["speaker_id"]).upper(), r["uid"])
    ):
        sp = str(r["speaker_id"]).upper()
        pid = prompt_id(r["uid"])
        if sp in blocked or pid in core_ids:
            continue
        if canonical:
            phones, _ = cmu_phones(texts[pid], cmu) if pid in texts else (None, [])
        else:
            phones = [
                IPA_TO_ARPA39[unicodedata.normalize("NFC", t)]
                for t in r["realized_ipa_tokens"]
            ]
        if not phones or len(phones) < 6:
            continue
        out.append(
            anchor_row(identity_from_intake(r), phones, "fit", canonical=canonical)
        )
        blocked.add(sp)
        if len(all_sp) + len(out) == at_least:
            break
    bind_audio(out, {})
    if len(all_sp) + len(out) < at_least or any(
        v for v in split_overlap(out, [*fit, *cal, *ev]).values()
    ):
        raise ValueError("Cannot admit the required new TIMIT TRAIN speakers")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--prompt-table", type=Path)
    a = p.parse_args()
    result = prepare(a.run_dir, a.out_manifest, a.prompt_table)
    print({k: summary(v) for k, v in result.items()})


if __name__ == "__main__":
    main()
