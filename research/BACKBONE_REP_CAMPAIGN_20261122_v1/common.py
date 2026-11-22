"""Shared split loading, custody verification, state machine, and Koel pooling utilities."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    GOAL_COMMIT,
    ROOT,
    V2_CKPT,
    V2_SHA,
    frame_path,
    read_frames,
    read_json,
    read_jsonl,
    sha_file,
    validate_row,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.repair_prompts import DEFAULT_REPAIR
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.single_factor_models import (
    pool_features_with_hvit,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import (
    domain,
    probabilities,
)

CAMPAIGN_DIR = Path(__file__).resolve().parent
BASE_KOEL_CACHE = ROOT / "research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/campaign_d5f8e89/frames"
PROPOSED_CLEAN_SPLIT = ROOT / "research/SYNTH_3DIR_ISO_V1_20261122/run_v2/proposed_custody_clean_baseline"
LEGACY_15_SPLIT = DEFAULT_REPAIR / "splits"
CUSTODY_SPEC_YAML = ROOT.parent / "New-Method-Research/data/split_specs/L2_ARCTIC_SPLIT.yaml"
PROTECTED_SEALED_SPEAKERS = frozenset({"ZHAA", "TNI", "MBMPS", "TLV"})
ARCHIVED_ISO = DEFAULT_REPAIR.parents[1] / "checkpoints/seq_mdd_v2_1_fix1_iso_no_hwin_seed20261122.pt"
ARCHIVED_ISO_SHA = "91dc67fe396555933057f8ae4c52283682ee62e0f5002d90470607762bf13cd6"
SEEDS = (20261121, 20261122, 20261123)
MIN_DISK_FREE_GIB = 15.0


def atomic_save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def check_disk_guard(path: Path, min_gib: float = MIN_DISK_FREE_GIB) -> float:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    free_gib = usage.free / (1024**3)
    if free_gib < min_gib:
        raise RuntimeError(f"Disk guard triggered: {free_gib:.2f} GiB free < {min_gib:.2f} GiB")
    return free_gib


def load_state(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "PIPELINE_STATE.json"
    if p.exists():
        return read_json(p)
    return {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "completed": {}}


def mark_done(out_dir: Path, state: Dict[str, Any], key: str, payload: Dict[str, Any]) -> None:
    state["completed"][key] = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_save_json(out_dir / "PIPELINE_STATE.json", state)


def verify_environment_and_custody(split_mode: str) -> Dict[str, Any]:
    """Verify GOAL.md, archived V2 checkpoint, and split custody without opening test_sealed."""
    original_goal = subprocess.check_output(["git", "show", f"{GOAL_COMMIT}:GOAL.md"], cwd=ROOT)
    if (ROOT / "GOAL.md").read_bytes() != original_goal:
        raise RuntimeError("GOAL.md was modified")
    if sha_file(V2_CKPT) != V2_SHA or sha_file(ARCHIVED_ISO) != ARCHIVED_ISO_SHA:
        raise RuntimeError("Archived checkpoint SHA mismatch")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    if split_mode != "clean_11":
        raise ValueError(f"Unsupported split_mode: {split_mode}; only clean_11 was approved")
    split_dir = PROPOSED_CLEAN_SPLIT
    proposal = read_json(split_dir / "PROPOSED_BASELINE.json")
    spec_sha = sha_file(CUSTODY_SPEC_YAML)
    if spec_sha != proposal["custody_spec_sha256"]:
        raise RuntimeError("Public custody spec changed since the clean split was prepared")
    from yaml import safe_load

    roster = safe_load(CUSTODY_SPEC_YAML.read_text())["splits"]["TEST_SEALED"]["speakers"]
    if set(roster) != PROTECTED_SEALED_SPEAKERS:
        raise RuntimeError("Unexpected public custody roster")
    manifests = {name: sha_file(split_dir / name) for name in ("fit.jsonl", "cal.jsonl", "eval_prompts.jsonl")}
    if manifests != proposal["manifest_sha256"]:
        raise RuntimeError("Proposed clean manifests changed")
    parent = LEGACY_15_SPLIT / "PREFLIGHT.json"
    if sha_file(parent) != proposal["parent_preflight_sha256"]:
        raise RuntimeError("Repaired parent preflight changed")
    parent_hashes = read_json(parent)["manifest_sha256"]
    if parent_hashes["eval_gold.jsonl"] != proposal["parent_eval_gold_sha256_for_future_join"] or (
        sha_file(LEGACY_15_SPLIT / "eval_prompts.jsonl") != parent_hashes["eval_prompts.jsonl"]
    ):
        raise RuntimeError("Historical diagnostic prompt/gold binding mismatch")
    roles = {name: read_jsonl(split_dir / name) for name in manifests}
    fit_rows, cal_rows, eval_prompts = roles.values()
    parent_prompts = {(r["cohort"], r["uid"]): r for r in read_jsonl(LEGACY_15_SPLIT / "eval_prompts.jsonl")}
    if any(r != parent_prompts.get((r["cohort"], r["uid"])) for r in eval_prompts):
        raise RuntimeError("Clean diagnostic prompts are not an unchanged subset of parent")

    def speakers(rows):
        return {str(r["speaker"]).strip().upper() for r in rows}

    for name, rows in roles.items():
        if len(rows) != proposal["rows"][name] or len({r["uid"] for r in rows}) != len(rows):
            raise RuntimeError(f"Unexpected count or duplicate UID: {name}")
        if speakers(rows) & PROTECTED_SEALED_SPEAKERS:
            raise RuntimeError(f"Protected speakers in {name}")
    for a, b in ((fit_rows, cal_rows), (fit_rows, eval_prompts), (cal_rows, eval_prompts)):
        if any({r[k] for r in a} & {r[k] for r in b} for k in ("uid", "audio_sha256")):
            raise RuntimeError("Cross-role UID or WAV hash overlap")
    if speakers(fit_rows) & speakers(cal_rows):
        raise RuntimeError("Fit and Cal speakers overlap")
    seen = speakers(fit_rows) | speakers(cal_rows)
    seen_eval = Counter(r["cohort"] for r in eval_prompts if r["speaker"].strip().upper() in seen)
    if set(seen_eval) - {"spontaneous"}:
        raise RuntimeError(f"Unexpected Fit/Cal speaker exposure in diagnostic Eval: {seen_eval}")

    return {
        "git_head": git_head,
        "split_mode": split_mode,
        "split_dir": str(split_dir),
        "fit_sha256": manifests["fit.jsonl"],
        "cal_sha256": manifests["cal.jsonl"],
        "eval_prompts_sha256": manifests["eval_prompts.jsonl"],
        "custody_spec_sha256": spec_sha,
        "parent_eval_gold_sha256": proposal["parent_eval_gold_sha256_for_future_join"],
        "archived_iso_sha256": ARCHIVED_ISO_SHA,
        "archived_v2_sha256": V2_SHA,
        "historical_eval_rows_with_seen_speaker": dict(seen_eval),
        "counts": {
            "fit": len(fit_rows),
            "cal": len(cal_rows),
            "eval_prompts": len(eval_prompts),
        },
        "test_sealed_opened": False,
        "disk_free_gib": check_disk_guard(ROOT),
    }


def load_splits_no_eval_gold(
    split_mode: str,
    *,
    smoke_limits: Mapping[str, int] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load Fit, Cal, and label-free Eval prompts (NEVER touches eval_gold.jsonl)."""
    if split_mode != "clean_11":
        raise ValueError("Only clean_11 is admitted")
    split_dir = PROPOSED_CLEAN_SPLIT
    fit_rows = read_jsonl(split_dir / "fit.jsonl")
    cal_rows = read_jsonl(split_dir / "cal.jsonl")
    eval_prompts = read_jsonl(split_dir / "eval_prompts.jsonl")
    for r in eval_prompts:
        if "gold_targets" in r or "gold_gaps" in r:
            raise ValueError("eval_prompts.jsonl must remain label-free")

    if smoke_limits:
        # Stratified sample plus the longest real Fit/diagnostic Eval WAVs.
        import soundfile as sf
        longest_fit = max(fit_rows, key=lambda r: sf.info(r["audio_path"]).frames)
        fit_rows = [longest_fit] + _stratified_take(
            [r for r in fit_rows if r is not longest_fit], "source", smoke_limits["fit"] - 1
        )
        cal_rows = _stratified_take(cal_rows, "source", smoke_limits["cal"])
        long_row = max((r for r in eval_prompts if r["cohort"] == "saa27"), key=lambda r: sf.info(r["audio_path"]).frames)
        from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.replay_codefix_audit import HELD6_SPEAKERS

        sp_row = next(r for r in eval_prompts if r["cohort"] == "spontaneous" and r["speaker"] in HELD6_SPEAKERS)
        first = [long_row, sp_row]
        eval_prompts = first + _stratified_take(
            [r for r in eval_prompts if r not in first], "cohort", smoke_limits["eval"] - len(first)
        )
    return fit_rows, cal_rows, eval_prompts


def file_identities(rows):
    """Check admitted WAV bytes and digest the exact frozen Koel cache inputs."""
    verified = {}
    koels = {}
    for row in rows:
        path = Path(row["audio_path"]).resolve(strict=True)
        if any("test_sealed" in part.casefold() for part in path.parts):
            raise ValueError("Sealed path is not authorized")
        if path not in verified:
            verified[path] = sha_file(path)
        digest = row["audio_sha256"]
        if verified[path] != digest:
            raise RuntimeError(f"WAV SHA mismatch: {row['uid']}")
        if digest not in koels:
            koels[digest] = sha_file(frame_path(BASE_KOEL_CACHE, row))
    return {
        "unique_wav_paths_verified": len(verified),
        "unique_koel_cache_files": len(koels),
        "koel_cache_content_sha256": hashlib.sha256(json.dumps(koels, sort_keys=True).encode()).hexdigest(),
    }


def open_arm_dir(path: Path) -> None:
    """Retain partial attempts on resume rather than deleting their artifacts."""
    if path.exists():
        archive = path.parent / "_partial_attempts"
        archive.mkdir(exist_ok=True)
        path.rename(archive / f"{path.name}_{time.time_ns()}")
    path.mkdir(parents=True, exist_ok=False)


def load_eval_gold_stage6_only(
    expected_sha: str,
    eval_prompts: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Join only admitted diagnostic rows from the historical Eval gold, never the sealed test."""
    path = LEGACY_15_SPLIT / "eval_gold.jsonl"
    if sha_file(path) != expected_sha:
        raise RuntimeError("Historical diagnostic gold changed")
    keys = {(r["cohort"], r["uid"]) for r in eval_prompts}
    by_key = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["cohort"], r["uid"])
            if key in keys:
                if key in by_key:
                    raise ValueError(f"Duplicate diagnostic gold key: {key}")
                by_key[key] = r
    matched: List[Dict[str, Any]] = []
    for p in eval_prompts:
        k = (p["cohort"], p["uid"])
        if k not in by_key:
            raise KeyError(f"Missing eval gold for {k}")
        g = by_key[k]
        if str(p["speaker"]).strip().upper() in PROTECTED_SEALED_SPEAKERS:
            raise RuntimeError(f"Protected speaker in diagnostic Eval: {p['speaker']}")
        if set(g) != {"cohort", "uid", "gold_targets", "gold_gaps"}:
            raise ValueError(f"Unexpected historical diagnostic gold schema: {k}")
        joined = {**p, **g}
        validate_row(joined)
        matched.append(joined)
    return matched


def smoke_gold(rows):
    """Procedural labels solely to exercise seal/score; no historical Eval gold reads."""
    result = []
    for r in rows:
        n = len(r["canonical_phones"])
        targets = ["C"] * n
        gaps = ["N"] * (n + 1)
        if r["cohort"] != "timit_core":
            targets[n // 2] = "S"
        if not r["cohort"].startswith("timit"):
            targets[0] = "O"
            gaps[n // 2] = "I"
        result.append(dict(r, gold_targets=targets, gold_gaps=gaps))
    return result


def _stratified_take(rows: Sequence[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "?")), []).append(r)
    out: List[Dict[str, Any]] = []
    keys = sorted(groups.keys())
    idx = 0
    while len(out) < limit:
        added = 0
        for k in keys:
            if idx < len(groups[k]) and len(out) < limit:
                out.append(groups[k][idx])
                added += 1
        if added == 0:
            break
        idx += 1
    return out


def pool_koel_iso(koel_cache: Path, row: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Pool Koel features identically to iso_no_hwin (drops unused 5-frame h_win)."""
    h, lp = read_frames(koel_cache, row)
    f = pool_features_with_hvit(h, lp, row["canonical_phones"])
    f["h_win"] = f["h_peak"][:, None, :, :]
    return f


def pool_koel_dataset(
    koel_cache: Path, rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, np.ndarray]]:
    return [pool_koel_iso(koel_cache, r) for r in rows]


def compute_cal_arrays(rows, logits, policy):
    l2 = {"t_sc": [], "t_g": [], "g_sc": [], "g_g": []}
    l1 = {"t_sc": [], "g_sc": []}
    for r, lg in zip(rows, logits, strict=True):
        tp, gp = probabilities(lg, r, policy)
        sc = tp[:, 1] + tp[:, 2]
        if domain(r) == "l1":
            l1["t_sc"].extend(sc.tolist())
            l1["g_sc"].extend(gp[:, 1].tolist())
        else:
            l2["t_sc"].extend(sc.tolist())
            l2["t_g"].extend(r["gold_targets"])
            l2["g_sc"].extend(gp[:, 1].tolist())
            l2["g_g"].extend(r["gold_gaps"])
    return (
        {k: np.asarray(v) for k, v in l2.items()},
        {k: np.asarray(v) for k, v in l1.items()},
    )


def param_group_sha(model: torch.nn.Module, predicate) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if predicate(name):
            h.update(name.encode("utf-8"))
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def is_gap_param_name(name: str) -> bool:
    return name.startswith("gap_") or ".gap_" in name


def write_checkpoint_card(out_dir: Path, payload: Dict[str, Any]) -> str:
    card_path = out_dir / "CHECKPOINT_CARD.json"
    atomic_save_json(card_path, payload)
    return sha_file(card_path)
