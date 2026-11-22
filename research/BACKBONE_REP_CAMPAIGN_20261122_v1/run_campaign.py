"""Unified CLI driver for BACKBONE_REP_CAMPAIGN_20261122_v1 (Stages 0 -> 6).

Supports:
  --mode smoke : exercise all stages with procedural scoring labels, including longest WAVs.
  --mode full  : gated GPU campaign. Duration depends on measured throughput and passed arms.
  --resume     : validate frozen inputs and reuse completed stages/seeds without overwriting attempts.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import platform
import time
import traceback

import torch

from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import read_json, sha_file
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.training import norm_from_fit

from .backbones import VRAM_CAP_MIB, model_identities
from .common import (
    BASE_KOEL_CACHE,
    CAMPAIGN_DIR,
    ROOT,
    SEEDS,
    atomic_save_json,
    file_identities,
    load_splits_no_eval_gold,
    load_state,
    pool_koel_dataset,
    verify_environment_and_custody,
)
from .stages_0_3 import (
    CANDIDATE_BACKBONES,
    run_stage0_s0_clean,
    run_stage1_backbone_probe,
    run_stage2_dualrep,
    run_stage3_frame_xattn,
)
from .stages_4_6 import (
    run_stage4_wavlm_lora,
    run_stage5_confirm_and_distill,
    run_stage6_seal_recount_latency_report,
)


def run_all(
    out_dir: Path,
    *,
    mode: str = "full",
    split_mode: str = "clean_11",
    device_name: str = "cuda",
    resume: bool = False,
) -> dict:
    if out_dir.exists() and not resume:
        raise FileExistsError(f"Output directory already exists (use --resume): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume and not (out_dir / "RUN_BINDINGS.json").exists():
        raise RuntimeError("Cannot resume without this campaign's frozen run bindings; start a new output directory")
    with (out_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"An existing worker holds {lock.name}") from exc
        status_file = out_dir / "PROCESS_STATUS.json"
        if resume:
            binding = read_json(out_dir / "RUN_BINDINGS.json")
            if (binding["mode"] != mode or binding["preflight"]["split_mode"] != split_mode
                    or binding["preflight"]["environment"]["device"] != device_name):
                raise RuntimeError("Refusing to resume with different mode, split or device")
        if (out_dir / "RESULTS.json").exists() and status_file.exists():
            status = read_json(status_file)
            if status["status"].startswith("COMPLETE_"):
                if sha_file(out_dir / "RESULTS.json") != status["results_sha256"]:
                    raise RuntimeError("Completed result digest changed")
                return read_json(out_dir / "RESULTS.json")
        start = {"pid": os.getpid(), "mode": mode, "split": split_mode, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_save_json(status_file, {**start, "status": "RUNNING"})
        try:
            result = _run_campaign(out_dir, mode, split_mode, torch.device(device_name))
            atomic_save_json(status_file, {**start, "status": result["status"], "results_sha256": sha_file(out_dir / "RESULTS.json")})
            return result
        except BaseException as exc:
            atomic_save_json(status_file, {**start, "status": "FAILED", "error": str(exc), "traceback": traceback.format_exc()})
            raise


def _run_campaign(out_dir, mode, split_mode, device):
    from importlib.metadata import version

    t_all = time.perf_counter()
    state = load_state(out_dir)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
    print(f"=== BACKBONE_REP_CAMPAIGN_20261122_v1 | mode={mode} | split={split_mode} | device={device} ===", flush=True)
    preflight = verify_environment_and_custody(split_mode)
    print(f"[Preflight] {preflight['counts']} | disk_free={preflight['disk_free_gib']:.1f} GiB", flush=True)

    is_smoke = mode == "smoke"
    smoke_limits = {"fit": 36, "cal": 18, "eval": 24} if is_smoke else None
    fit_rows, cal_rows, eval_prompts = load_splits_no_eval_gold(split_mode, smoke_limits=smoke_limits)
    print(f"[Splits] fit={len(fit_rows)} cal={len(cal_rows)} eval_prompts={len(eval_prompts)}", flush=True)
    preflight["inputs"] = file_identities(fit_rows + cal_rows + eval_prompts)
    preflight["models"] = model_identities()
    preflight["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {k: version(k) for k in ("torch", "transformers", "numpy", "scikit-learn", "PyYAML", "soundfile", "pyarrow")},
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "vram_cap_mib_reserved": VRAM_CAP_MIB,
    }
    code_dirs = [CAMPAIGN_DIR, *(ROOT / "research" / n for n in (
        "SEQ_MDD_V3_CAMPAIGN_20261121_v1", "SYNTH_3DIR_ISO_V1_20261122", "MDD_FIXES_AND_V2_20261121_v1"
    ))]
    bindings = {
        "mode": mode,
        "preflight": {k: v for k, v in preflight.items() if k != "disk_free_gib"},
        "source_sha256": {str(p.relative_to(ROOT)): sha_file(p) for d in code_dirs for p in sorted(d.glob("*.py"))},
        "entrypoint": "research.BACKBONE_REP_CAMPAIGN_20261122_v1.run_campaign",
    }
    binding_file = out_dir / "RUN_BINDINGS.json"
    if binding_file.exists():
        if read_json(binding_file) != bindings:
            raise RuntimeError("Refusing resume: mode, source, environment or input identity changed")
    else:
        atomic_save_json(binding_file, bindings)
        atomic_save_json(out_dir / "PREFLIGHT.json", preflight)

    print("[Koel] Pooling immutable Base Koel features from BASE_KOEL_CACHE...", flush=True)
    t_k = time.perf_counter()
    xf = pool_koel_dataset(BASE_KOEL_CACHE, fit_rows)
    xc = pool_koel_dataset(BASE_KOEL_CACHE, cal_rows)
    xe = pool_koel_dataset(BASE_KOEL_CACHE, eval_prompts)
    norm = norm_from_fit(xf)
    print(f"[Koel] Pooled {len(xf)+len(xc)+len(xe)} utterances in {time.perf_counter()-t_k:.1f}s", flush=True)

    seeds = (SEEDS[0], SEEDS[1]) if is_smoke else SEEDS
    ep_s0 = 2 if is_smoke else 14
    ep_s2 = 2 if is_smoke else 14
    ep_p2 = 1 if is_smoke else 8
    ep_s3 = 2 if is_smoke else 12
    ep_s4 = 2 if is_smoke else 8
    warm_utts = 16 if is_smoke else 2400
    warm_ep = 1 if is_smoke else 2

    # Stage 0: Clean S0 Baseline
    s0 = run_stage0_s0_clean(
        out_dir, state, fit_rows, cal_rows, eval_prompts, xf, xc, xe, norm, device,
        seeds=seeds, epochs=ep_s0, batch_size=16,
    )

    # Stage 1: Multi-Backbone Fit+Cal Probe & Cal Kill-Gate
    s1 = run_stage1_backbone_probe(
        out_dir, state, fit_rows, cal_rows, eval_prompts, xf, xc, device,
        backbones=CANDIDATE_BACKBONES,
        max_fit_slots=4000 if is_smoke else 32000,
        smoke=is_smoke,
    )

    # Stage 2: DualRepSingleFactorV2 (4 arms + Phase-2 Frozen Gap)
    s2 = run_stage2_dualrep(
        out_dir, state, s1, fit_rows, cal_rows, eval_prompts, xf, xc, xe, norm, device,
        seed=seeds[0], epochs=ep_s2, phase2_epochs=ep_p2, batch_size=16,
    )

    # Stage 3: FrameXAttnMDD (Monotonic Local Cross-Attention)
    s3 = run_stage3_frame_xattn(
        out_dir, state, s1, fit_rows, cal_rows, eval_prompts, xf, xc, xe, norm, device,
        seed=seeds[0], epochs=ep_s3,
    )

    # Stage 4 is a separate controlled adaptation hypothesis; prefer Large as in
    # the approved design, independent of the ranking of frozen middle layers.
    s4 = run_stage4_wavlm_lora(
        out_dir, state, fit_rows, cal_rows, eval_prompts, xf, xc, xe, norm, device,
        preferred_backbone="wavlm_large",
        seed=seeds[0],
        epochs=ep_s4,
        warm_max_utts=warm_utts,
        warm_epochs=warm_ep,
    )

    # Stage 5: Cal-selected Top-2 Paired 3-Seed Confirmation + Fast Distillation
    s5 = run_stage5_confirm_and_distill(
        out_dir, state, s0, s1, s2, s3, s4, fit_rows, cal_rows, eval_prompts, xf, xc, xe, norm, device,
        seeds=seeds,
        dualrep_epochs=ep_s2,
        phase2_epochs=ep_p2,
        xattn_epochs=ep_s3,
        lora_epochs=ep_s4,
        warm_max_utts=warm_utts,
        warm_epochs=warm_ep,
        smoke=is_smoke,
    )

    # Stage 6: Pre-gold Seal Verification, CPU Independent Recount, Latency & Report
    results = run_stage6_seal_recount_latency_report(
        out_dir, state, preflight, s0, s1, s2, s3, s4, s5, eval_prompts, xe, device,
        seeds=seeds, smoke=is_smoke,
    )
    print(f"=== CAMPAIGN FINISHED in {(time.perf_counter() - t_all) / 60.0:.2f} min ===", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("research/BACKBONE_REP_CAMPAIGN_20261122_v1/runs/run_v1"),
    )
    p.add_argument("--mode", choices=["smoke", "full"], default="full")
    p.add_argument("--split-mode", choices=["clean_11"], default="clean_11")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run_all(
        args.out,
        mode=args.mode,
        split_mode=args.split_mode,
        device_name=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
