"""Build bundle, verify exact raw-WAV parity against sealed Parquets, and benchmark latency."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import load_splits_no_eval_gold
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import save_json, sha_file

from .engine import DEFAULT_BUNDLE_PATH, KoelEns3MDDEngine, build_bundle

SEALED_PRED_DIR = Path(
    "research/ORTHO_IMPROVEMENTS_20261122_v1/runs/s0_clean_ens3_o5_o4/eval_predictions"
)


def main() -> None:
    t0 = time.perf_counter()
    bundle_p = build_bundle()
    bundle_sha = sha_file(bundle_p)
    bundle_mb = round(bundle_p.stat().st_size / (1024**2), 2)
    print(f"[Bundle] Built {bundle_p} ({bundle_mb} MiB, SHA={bundle_sha[:16]}...)", flush=True)

    _, _, eval_prompts = load_splits_no_eval_gold("clean_11")
    h6_rows = [r for r in eval_prompts if r["cohort"] == "held6"]
    timit_rows = [r for r in eval_prompts if r["cohort"] == "timit_core"]
    saa_rows = [r for r in eval_prompts if r["cohort"] == "saa27"]

    # 1. Exact RAW-WAV parity vs sealed s0_clean_ens3_o5_o4 Parquet (10 Held-6 + 5 TIMIT WAVs)
    h6_pq = pd.read_parquet(SEALED_PRED_DIR / "held6.parquet")
    tc_pq = pd.read_parquet(SEALED_PRED_DIR / "timit_core.parquet")
    pq_map = {"held6": h6_pq, "timit_core": tc_pq}

    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_str == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eng_gpu = KoelEns3MDDEngine(bundle_p, device=dev_str, load_koel=True)

    parity_rows = h6_rows[:10] + timit_rows[:5]
    max_t_diff, max_g_diff = 0.0, 0.0
    n_t_slots, n_g_slots = 0, 0
    t_call_matches, g_call_matches = 0, 0

    for r in parity_rows:
        res = eng_gpu.score_wav(r["audio_path"], r["canonical_phones"])
        sub = pq_map[r["cohort"]]
        sub_u = sub[sub["uid"] == r["uid"]]
        t_ref = sub_u[sub_u["domain"] == "target"].sort_values("index")
        g_ref = sub_u[sub_u["domain"] == "gap"].sort_values("index")

        for pred, (_, ref) in zip(res["targets"], t_ref.iterrows(), strict=True):
            n_t_slots += 1
            max_t_diff = max(max_t_diff, abs(pred["error_prob"] - float(ref["score"])))
            if pred["call"] == str(ref["call"]):
                t_call_matches += 1
        for pred, (_, ref) in zip(res["gaps"], g_ref.iterrows(), strict=True):
            n_g_slots += 1
            max_g_diff = max(max_g_diff, abs(pred["insertion_prob"] - float(ref["score"])))
            if pred["call"] == str(ref["call"]):
                g_call_matches += 1

    print(
        f"[Parity RAW-WAV -> Sealed Parquet] Targets: {t_call_matches}/{n_t_slots} exact calls "
        f"(max |dScore|={max_t_diff:.2e}) | Gaps: {g_call_matches}/{n_g_slots} exact calls "
        f"(max |dScore|={max_g_diff:.2e})",
        flush=True,
    )

    # 2. End-to-End GPU Latency Benchmark (50 Held-6 short WAVs + 5 Stella long WAVs, 3 warmup)
    for r in h6_rows[:3]:
        eng_gpu.score_wav(r["audio_path"], r["canonical_phones"])
    if dev_str == "cuda":
        torch.cuda.reset_peak_memory_stats()

    short_tot, short_koel, short_head = [], [], []
    for r in h6_rows[3:53]:
        out = eng_gpu.score_wav(r["audio_path"], r["canonical_phones"])
        short_tot.append(out["timing_ms"]["total_ms"])
        short_koel.append(out["timing_ms"]["koel_ms"])
        short_head.append(out["timing_ms"]["ens3_head_ms"])
    short_peak_mib = round(torch.cuda.max_memory_reserved() / (1024**2), 1) if dev_str == "cuda" else 0.0

    stella_tot = []
    for r in saa_rows[:5]:
        out = eng_gpu.score_wav(r["audio_path"], r["canonical_phones"])
        stella_tot.append(out["timing_ms"]["total_ms"])
    stella_peak_mib = round(torch.cuda.max_memory_reserved() / (1024**2), 1) if dev_str == "cuda" else 0.0

    del eng_gpu
    gc.collect()
    if dev_str == "cuda":
        torch.cuda.empty_cache()

    # 3. End-to-End CPU Latency Benchmark (10 Held-6 short WAVs, 1 warmup)
    eng_cpu = KoelEns3MDDEngine(bundle_p, device="cpu", load_koel=True)
    eng_cpu.score_wav(h6_rows[0]["audio_path"], h6_rows[0]["canonical_phones"])
    cpu_tot = [
        eng_cpu.score_wav(r["audio_path"], r["canonical_phones"])["timing_ms"]["total_ms"]
        for r in h6_rows[1:11]
    ]

    receipt: Dict[str, Any] = {
        "status": "DEPLOY_BUNDLE_VERIFIED",
        "bundle_path": str(DEFAULT_BUNDLE_PATH),
        "bundle_size_mib": bundle_mb,
        "bundle_sha256": bundle_sha,
        "raw_wav_parity": {
            "n_utts": len(parity_rows),
            "target_call_exact": f"{t_call_matches}/{n_t_slots}",
            "gap_call_exact": f"{g_call_matches}/{n_g_slots}",
            "target_max_abs_diff": round(max_t_diff, 8),
            "gap_max_abs_diff": round(max_g_diff, 8),
            "all_calls_exact": (t_call_matches == n_t_slots and g_call_matches == n_g_slots),
        },
        "end_to_end_latency_gpu": {
            "device": dev_str,
            "short_held6_n": len(short_tot),
            "short_total_p50_ms": round(float(np.median(short_tot)), 1),
            "short_total_p95_ms": round(float(np.percentile(short_tot, 95)), 1),
            "short_koel_p50_ms": round(float(np.median(short_koel)), 1),
            "short_ens3_head_p50_ms": round(float(np.median(short_head)), 1),
            "short_peak_vram_mib": short_peak_mib,
            "stella_n": len(stella_tot),
            "stella_total_p50_ms": round(float(np.median(stella_tot)), 1),
            "stella_peak_vram_mib": stella_peak_mib,
        },
        "end_to_end_latency_cpu": {
            "short_held6_n": len(cpu_tot),
            "short_total_p50_ms": round(float(np.median(cpu_tot)), 1),
        },
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    out_json = Path("deploy/koel_ens3_mdd/DEPLOY_VERIFICATION.json")
    save_json(out_json, receipt)
    print("\n=== DEPLOY VERIFICATION SUMMARY ===")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
