"""EXP-1: isolate multi-frame pooling, then test the full window/attention package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    CONTROL,
    HERE,
    SEED,
    admitted,
    read_json,
    save_json,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import Spec
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.scoring import (
    brief,
    load_old_r0,
    read_predictions,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import (
    evaluate_arm,
    replay_baseline,
    train_arm,
)


def gates(ev, v2):
    c = ev[CONTROL]
    delta = {
        name: ev[name]["strict"]["f1"] - v2[name]["strict"]["f1"]
        for name in ("held6", "epadb_test")
    }
    return {
        "held6_plus_1_5_pp": delta["held6"] >= 0.015,
        "epadb_within_minus_0_5_pp": delta["epadb_test"] >= -0.005,
        "timit_max_1_70_fa_per_utt": ev["timit_core"]["fa_per_utt"] <= 1.70,
        "control_min_94_recall": c["recall"] >= 0.94,
        "control_min_0_95_ap": c["ap"] >= 0.95,
    }


def run_exp1(run, out, device, seed, epochs, batch_size, *, full_latency=False):
    if not (run / "splits/FIT_FRAME_RECEIPT.json").exists():
        raise RuntimeError("Finish FIT/Cal frame extraction first")
    fit = admitted(run / "splits", "fit.jsonl")
    cal = admitted(run / "splits", "cal.jsonl")
    r0 = load_old_r0()
    if not (run / "v2_keyed/BASELINE_PARITY.json").is_file():
        replay_baseline(run, device, r0)
    v2 = read_json(run / "v2_keyed/RESULTS.json")["eval"]
    compare = {"r0_old": r0, "seq_mdd_v2": read_predictions(run / "v2_keyed")}
    out.mkdir(parents=True, exist_ok=False)
    save_json(
        out / "DESIGN.json",
        {
            "arms": {
                "pool_only": Spec(window="mean").as_dict(),
                "full_window": Spec(window="full").as_dict(),
            },
            "promotion": "full_window only; pool_only isolates a feature change, not a second try at the promotion gate",
            "latency": "full if explicitly requested; otherwise 36 fixed short utterances/cohort and all SAA-27",
            "cal_selection": "V2 AP criterion followed by V2 Cal-F1 threshold search; no Eval tuning",
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )
    results = {}
    for name, window in [("pool_only", "mean"), ("full_window", "full")]:
        path = out / name
        train_arm(
            path, fit, cal, Spec(window=window), run, device, seed, epochs, batch_size
        )
        result = evaluate_arm(
            path,
            run,
            device,
            compare,
            latency=(name == "full_window"),
            full_latency=full_latency,
        )
        ev = result["eval"]
        results[name] = {
            "metrics_pct_or_fa": brief(ev),
            "gates": gates(ev, v2),
            "checkpoint": str(path),
            "checkpoint_sha256": read_json(path / "CHECKPOINT_CARD.json")["sha256"],
        }
    go = all(results["full_window"]["gates"].values())
    ret = {
        "status": "GO" if go else "NO_GO",
        "promoted": str(out / "full_window") if go else None,
        "baseline": brief(v2),
        "arms": results,
        "diagnostic_only": True,
    }
    save_json(out / "RESULTS.json", ret)
    print("EXP-1 decision:", ret, flush=True)
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full-latency", action="store_true")
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or min(args.epochs, args.batch_size) < 1:
        parser.error("Use offline mode and positive epochs/batch size")
    run_exp1(
        args.run_dir,
        args.out_dir,
        torch.device(args.device),
        args.seed,
        args.epochs,
        args.batch_size,
        full_latency=args.full_latency,
    )


if __name__ == "__main__":
    main()
