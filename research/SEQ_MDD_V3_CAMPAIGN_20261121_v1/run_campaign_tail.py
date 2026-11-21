"""Run EXP-2 through EXP-5 serially after the sealed EXP-1 decision (one local GPU)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .data import GOAL_COMMIT, HERE, ROOT, SEED, read_json, save_json, sha_file

TESTS = (
    "tests/test_mdd_fixes_v2.py",
    "tests/test_seq_mdd_v3.py",
    "tests/test_koel_ollr_and_gapcore.py",
    "tests/test_seq_mdd_v3_campaign.py",
)
STAGES = {
    1: "exp1_window",
    2: "exp2_l1_canon",
    3: "exp3_gap_so",
    4: "exp4_multiseed",
    5: "exp5_joint_lora",
}
OBJECTIVES = {
    1: "isolate target multi-frame features, then test the full window package",
    2: "measure public-orthography TIMIT native anchoring with in-utterance controls",
    3: "separate gap-core/focal from CTC deletion/realized fusion",
    4: "DEV18 speaker OOF and final three-seed multi-corpus weighting / temperature scaling",
    5: "paired frozen/upper-layer q-v LoRA with L1 CTC-KD and PER protection",
}


def journal(run, **entry):
    with (run / "STAGES.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"utc": datetime.now(timezone.utc).isoformat(), **entry},
                allow_nan=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def wait_for_exp1(run, pid):
    p = run / STAGES[1] / "RESULTS.json"
    print(f"Waiting for EXP-1 worker {pid} and result {p}", flush=True)
    while True:
        s = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        )
        alive = (
            s.returncode == 0
            and s.stdout.strip()
            and not s.stdout.strip().startswith("Z")
        )
        if not alive:
            if not p.exists():
                raise RuntimeError(
                    f"EXP-1 stopped before decision; inspect {run}/exp1_retry1.log"
                )
            return read_json(p)
        time.sleep(5)


def snapshot(run):
    dest = run / "campaign_tail_freeze"
    dest.mkdir()
    files = [p for p in HERE.iterdir() if p.suffix in {".py", ".cpp"}] + [
        ROOT / t for t in TESTS
    ]
    saved = {str(p.relative_to(ROOT)): sha_file(p) for p in files}
    for p in files:
        q = dest / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(p.read_bytes())
    receipt = {
        "goal_commit": GOAL_COMMIT,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "source_sha256": saved,
        "command": list(__import__("sys").argv),
    }
    save_json(dest / "CODE_RECEIPT.json", receipt)
    return saved


def tests(run, tag):
    py = ROOT / ".venv/bin/python"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with (run / f"{tag}_pytest.log").open("x", encoding="utf-8") as f:
        subprocess.run(
            [str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    n = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return n.stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    p.add_argument("--wait-exp1-pid", type=int, required=True)
    args = p.parse_args()
    run = args.run_dir.resolve()
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        p.error("Use offline mode")
    # Import after EXP-1 exits, avoiding a second concurrent CUDA context.
    with (run / "campaign_tail.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = snapshot(run)
        journal(
            run,
            status="WAITING_EXP1",
            setup={"goal_commit": GOAL_COMMIT, "pid": os.getpid()},
            next_steps="Wait for EXP-1 decision",
        )
        stage = 1
        try:
            v = wait_for_exp1(run, args.wait_exp1_pid)
            if v["status"] not in {"GO", "NO_GO"}:
                raise RuntimeError("EXP-1 gate missing")
            report = run / STAGES[1] / "REPORT_VI.md"
            with report.open("x", encoding="utf-8") as f:
                f.write(
                    f"# EXP-1 — {v['status']}\n\nCửa sổ đơn và biến thể đầy đủ; xem `RESULTS.json` và hai bảng đã niêm phong trong thư mục con.\n"
                )
            gpu = tests(run, "exp1")
            journal(
                run,
                experiment="EXP-1",
                objective=OBJECTIVES[1],
                setup={"epochs": 14, "seed": SEED, "batch": 16, "gpu_after": gpu},
                results=v,
                conclusion=v["status"],
                next_steps="EXP-2 starts from latest gate-passing parent",
            )
            import torch
            from .run_exp5_joint_lora import run_exp5
            from .stages import run_exp2, run_exp3, run_exp4

            device = torch.device("cuda")
            for stage, thunk in [
                (2, run_exp2),
                (3, run_exp3),
                (4, run_exp4),
                (5, run_exp5),
            ]:
                if any(sha_file(ROOT / k) != v for k, v in frozen.items()):
                    raise RuntimeError(
                        "Implementation changed after campaign source freeze; stop before training"
                    )
                journal(
                    run,
                    experiment=f"EXP-{stage}",
                    objective=OBJECTIVES[stage],
                    setup={
                        "goal_commit": GOAL_COMMIT,
                        "device": str(device),
                        "seed": SEED,
                    },
                    status="STARTED",
                )
                v = thunk(run, device)
                gpu = tests(run, f"exp{stage}")
                journal(
                    run,
                    experiment=f"EXP-{stage}",
                    objective=OBJECTIVES[stage],
                    setup={
                        "goal_commit": GOAL_COMMIT,
                        "code_receipt": "campaign_tail_freeze/CODE_RECEIPT.json",
                        "gpu_after": gpu,
                    },
                    results=v,
                    conclusion=v["status"],
                    next_steps=f"EXP-{stage + 1}"
                    if stage < 5
                    else "Independent VN confirmation remains separate",
                )
                if v["status"] == "BLOCKED":
                    break
        except Exception as exc:
            journal(
                run,
                experiment=f"EXP-{stage}",
                objective=OBJECTIVES[stage],
                status="FAILED",
                conclusion=f"{type(exc).__name__}: {exc}",
                next_steps="Inspect immutable logs and artifacts; do not overwrite",
            )
            raise


if __name__ == "__main__":
    main()
