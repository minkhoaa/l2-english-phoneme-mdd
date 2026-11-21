"""Print serial campaign progress; optionally verify and save its final artifact summary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from .data import (
    GOAL_COMMIT,
    HERE,
    V2_CKPT,
    V2_SHA,
    read_json,
    read_jsonl,
    save_json,
    sha_file,
)

STAGES = (
    ("exp1_window", "RESULTS.json"),
    ("exp2_l1_canon", "OUTCOME.json"),
    ("exp3_gap_so", "RESULTS.json"),
    ("exp4_multiseed", "RESULTS.json"),
    ("exp5_joint_lora", "RESULTS.json"),
)


def worker_running(run):
    p = run / "campaign_tail.pid"
    if not p.is_file():
        return False
    pid = p.read_text().strip()
    if not pid.isdecimal():
        raise ValueError("Invalid worker PID")
    try:
        cmd = (Path("/proc") / pid / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b"run_campaign_tail" in cmd


def progress(run):
    journal = run / "STAGES.jsonl"
    last = {}
    if journal.exists():
        for row in read_jsonl(journal):
            if row.get("experiment"):
                last[row["experiment"]] = row
    out = {}
    for idx, (name, file) in enumerate(STAGES, 1):
        stage = f"EXP-{idx}"
        target = run / name / file
        j = last.get(stage, {})
        if target.is_file():
            row = read_json(target)
            out[stage] = {
                "status": row["status"],
                "result": str(target.relative_to(run)),
                "sha256": sha_file(target),
                "metrics_pct_or_fa": row.get("metrics_pct_or_fa"),
                "gates": row.get("gates"),
                "reason": row.get("reason"),
            }
            if idx == 1:
                out[stage]["arms"] = row["arms"]
            if "mean_sd" in row:
                out[stage]["mean_sd"] = row["mean_sd"]
        else:
            state = j.get("status")
            out[stage] = {"status": state or "PENDING", "reason": j.get("conclusion")}
    return out


def verify(run):
    from .data import admitted
    from .scoring import evaluate_seal

    if sha_file(V2_CKPT) != V2_SHA:
        raise RuntimeError("Archived V2 checkpoint changed")
    audited = {}
    split = run / "splits"
    dev_gold = [
        dict(r, cohort="dev18_oof")
        for sp in ("fit", "cal")
        for r in admitted(split, sp + ".jsonl")
        if r["source"] == "l2arctic_dev18"
    ]
    for seal in sorted(run.glob("*/**/PREDICTIONS_SEAL.json")):
        if "campaign_tail_freeze" in seal.parts:
            continue
        path = seal.parent
        card = read_json(path / "CHECKPOINT_CARD.json")
        if (
            "checkpoint" in card
            and sha_file(path / card["checkpoint"]) != card["sha256"]
        ):
            raise RuntimeError(f"Checkpoint changed: {path}")
        if (
            "lora_adapter_sha256" in card
            and sha_file(path / "lora_adapter.pt") != card["lora_adapter_sha256"]
        ):
            raise RuntimeError(f"Adapter changed: {path}")
        if path.name.startswith("fold_"):
            continue  # All OOF fold rows are verified as one independent strict count below.
        gold = dev_gold if path.name == "oof" else None
        if path.name == "saa18_sensitivity":
            gold = [
                r for r in admitted(split, "eval_gold.jsonl") if r["cohort"] == "saa27"
            ]
        results = path / (
            "EVAL_METRICS.json"
            if (path / "EVAL_METRICS.json").is_file()
            else "RESULTS.json"
        )
        if not results.is_file():
            raise RuntimeError(f"Sealed arm lacks metrics: {path}")
        e, _ = evaluate_seal(path, split, gold_rows=gold)
        if e != read_json(results)["eval"]:
            raise RuntimeError(f"Sealed CPU recount mismatch: {path}")
        audited[str(path.relative_to(run))] = {
            "predictions_seal_sha256": sha_file(seal),
            "result_sha256": sha_file(results),
        }
    return audited


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    p.add_argument(
        "--finalize",
        action="store_true",
        help="Wait for the worker then CPU-recount and write CAMPAIGN_STATUS.json",
    )
    a = p.parse_args()
    run = a.run_dir.resolve()
    if a.finalize:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        while worker_running(run):
            time.sleep(15)
    stages = progress(run)
    alive = worker_running(run)
    statuses = {s["status"] for s in stages.values()}
    status = (
        "RUNNING"
        if alive
        else ("COMPLETE" if statuses <= {"GO", "NO_GO"} else "INCOMPLETE")
    )
    result = {"goal_commit": GOAL_COMMIT, "status": status, "stages": stages}
    if a.finalize:
        result["verified_arms"] = verify(run)
        save_json(run / "CAMPAIGN_STATUS.json", result)
    for k, v in stages.items():
        print(k, v["status"], v.get("result") or v.get("reason") or "")
    print("Campaign:", status)


if __name__ == "__main__":
    main()
