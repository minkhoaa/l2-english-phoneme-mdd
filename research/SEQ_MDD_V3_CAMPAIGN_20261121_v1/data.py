"""Local, allowlisted inputs and hashed single-backbone frame cache for SeqMDDv3."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GOAL_COMMIT = "d5f8e895a22cc28233f6c07bf6a35d018b054c3d"
LEGACY = ROOT / "research/MDD_FIXES_AND_V2_20261121_v1"
OLD = ROOT / "research/FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1/outputs"
V2_CKPT = LEGACY / "checkpoints/seq_mdd_v2_antishortcut_seed20261121.pt"
V2_SHA = "a9bb69ed5cdd49d6fa9d0b6d8c4305a43d9e9834c0e59ed64567f08c6c61cb65"
LAYERS = (12, 15, 18, 20, 22, 24)
COHORTS = ("held6", "spontaneous", "epadb_test", "saa27", "timit_core")
CONTROL = "timit_core_corrupt_ctrl"
SEED = 20261121
VRAM_MIB = 3500


def sha_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stable_sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def save_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(row):
    return str(row["speaker"]).strip().upper()


def summary(rows):
    return {
        "utterances": len(rows),
        "speakers": len({identity(r) for r in rows}),
        "sources": dict(Counter(r.get("source", r.get("cohort")) for r in rows)),
    }


def validate_row(row):
    n = len(row["canonical_phones"])
    if not n or len(row["gold_targets"]) != n or len(row["gold_gaps"]) != n + 1:
        raise ValueError(f"Slot length mismatch: {row['uid']}")
    if "realized_ids" in row and len(row["realized_ids"]) != n:
        raise ValueError(f"Realized slot length mismatch: {row['uid']}")
    if set(row["gold_targets"]) - {"C", "S", "O", "unknown"} or set(
        row["gold_gaps"]
    ) - {"N", "I", "unknown"}:
        raise ValueError(f"Unexpected gold label: {row['uid']}")


def bind_audio(rows, hashes):
    uid = set()
    for r in rows:
        validate_row(r)
        if r["uid"] in uid:
            raise ValueError(f"Duplicate UID within split: {r['uid']}")
        uid.add(r["uid"])
        p = Path(r["audio_path"]).resolve(strict=True)
        if p.suffix.lower() != ".wav" or not p.is_file():
            raise ValueError(f"Not an admitted WAV: {p}")
        if p not in hashes:
            hashes[p] = sha_file(p)
        r["audio_sha256"] = hashes[p]


def split_overlap(a, b):
    return {
        name: sorted(set(f(r) for r in a) & set(f(r) for r in b))
        for name, f in [
            ("speakers", identity),
            ("uids", lambda r: r["uid"]),
            ("audio_sha256", lambda r: r["audio_sha256"]),
        ]
    }


def only_prompt(r):
    return {
        k: r[k]
        for k in (
            "uid",
            "cohort",
            "speaker",
            "audio_path",
            "audio_sha256",
            "canonical_phones",
        )
    }


def corruption_control(rows):
    from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
        ARPA_39,
    )

    rng = np.random.default_rng(
        42
    )  # Replays the archived final V2 verifier, not its training driver's test.
    out = []
    for row in rows[:50]:
        can = list(row["canonical_phones"])
        target = ["C"] * len(can)
        for i in range(1, len(can) - 1, 6):
            can[i] = str(rng.choice([p for p in ARPA_39 if p != can[i]]))
            target[i] = "S"
        out.append(
            dict(
                row,
                uid=row["uid"] + "_ctrl",
                cohort=CONTROL,
                canonical_phones=can,
                gold_targets=target,
                gold_gaps=["N"] * (len(can) + 1),
            )
        )
    if sum(r["gold_targets"].count("S") for r in out) != 279 or len(out) != 50:
        raise ValueError("TIMIT control differs from the archived release")
    return out


def prepare(out):
    from research.CONTROLLED_PILOT_AB_20261118_v1.run_pilot_ab import WEIGHTS_DIR
    from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
        DEV18_GOLD,
        HELD6_GOLD,
        INTAKE_MANIFEST,
        SAA27_GOLD,
        TIMIT_GOLD,
    )
    from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
        build_datasets,
    )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    original_goal = subprocess.check_output(
        ["git", "show", f"{GOAL_COMMIT}:GOAL.md"], cwd=ROOT
    )
    if (ROOT / "GOAL.md").read_bytes() != original_goal or sha_file(V2_CKPT) != V2_SHA:
        raise RuntimeError("GOAL or archived V2 checkpoint mismatch")
    # The archived builder reads only these explicit sources. The separately sealed split is not used.
    allowed = [DEV18_GOLD, HELD6_GOLD, INTAKE_MANIFEST, SAA27_GOLD, TIMIT_GOLD]
    if any("test_sealed" in p.resolve().name.lower() for p in allowed):
        raise RuntimeError("Attempted sealed source in allowlist")
    fit, cal, ev = build_datasets()
    expected = {
        "held6": 802,
        "spontaneous": 129,
        "epadb_test": 1266,
        "saa27": 27,
        "timit_core": 192,
    }
    if (len(fit), len(cal)) != (4011, 635) or {
        k: len(v) for k, v in ev.items()
    } != expected:
        raise ValueError(
            f"Data recipe drift: FIT={len(fit)}, CAL={len(cal)}, EVAL={ {k: len(v) for k, v in ev.items()} }"
        )
    hashes = {}
    for rows in [fit, cal, *ev.values()]:
        bind_audio(rows, hashes)
    overlaps = {"fit/cal": split_overlap(fit, cal)}
    for name in COHORTS:
        for role, rows in [("fit", fit), ("cal", cal)]:
            overlaps[f"{role}/{name}"] = split_overlap(rows, ev[name])
    for name, overlap in overlaps.items():
        forbidden = {
            k: v
            for k, v in overlap.items()
            if v and not (name.endswith("/spontaneous") and k == "speakers")
        }
        if forbidden:
            raise ValueError(f"Forbidden overlap {name}: {forbidden}")
    ev[CONTROL] = corruption_control(ev["timit_core"])
    grouped = [r for name in (*COHORTS, CONTROL) for r in ev[name]]
    out.mkdir(parents=True, exist_ok=False)
    save_jsonl(out / "fit.jsonl", fit)
    save_jsonl(out / "cal.jsonl", cal)
    save_jsonl(out / "eval_prompts.jsonl", map(only_prompt, grouped))
    save_jsonl(
        out / "eval_gold.jsonl",
        (
            {
                "cohort": r["cohort"],
                "uid": r["uid"],
                "gold_targets": r["gold_targets"],
                "gold_gaps": r["gold_gaps"],
            }
            for r in grouped
        ),
    )
    web_weights = WEIGHTS_DIR / "model.safetensors"
    audit = {
        "git_head": commit,
        "goal_commit": GOAL_COMMIT,
        "goal_sha256": sha_file(ROOT / "GOAL.md"),
        "v2_sha256": V2_SHA,
        "fit": summary(fit),
        "cal": summary(cal),
        "eval": {k: summary(rows) for k, rows in ev.items()},
        "overlaps": overlaps,
        "control": {"seed": 42, "n_corrupt": 279, "n_unchanged": 1372},
        "source_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in allowed},
        "backbone": {
            name: sha_file(WEIGHTS_DIR / name)
            for name in (web_weights.name, "config.json", "vocab.json")
        },
        "manifest_sha256": {p.name: sha_file(p) for p in out.glob("*.jsonl")},
    }
    save_json(out / "PREFLIGHT.json", audit)
    print(
        json.dumps(
            {k: audit[k] for k in ("goal_commit", "fit", "cal", "eval", "overlaps")},
            indent=2,
        ),
        flush=True,
    )
    return audit


def admitted(out, name):
    p = out / name
    if sha_file(p) != read_json(out / "PREFLIGHT.json")["manifest_sha256"][name]:
        raise RuntimeError(f"Admitted input changed: {p}")
    return read_jsonl(p)


def frame_path(cache, row):
    h = row["audio_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", h):
        raise ValueError(f"Bad audio hash: {h!r}")
    return cache / h[:2] / (h + ".npz")


def read_frames(cache, row):
    with np.load(frame_path(cache, row), allow_pickle=False) as f:
        return f["hidden"], f["lp"]


def wav_input(row, device, *, verify=True):
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    if verify and sha_file(row["audio_path"]) != row["audio_sha256"]:
        raise RuntimeError(f"WAV changed: {row['uid']}")
    wav, rate = sf.read(row["audio_path"], dtype="float32")
    if rate != 16000 or wav.ndim != 1 or len(wav) < 400 or not np.isfinite(wav).all():
        raise ValueError(f"Bad rate/shape/content: {row['uid']}")
    x = torch.from_numpy(wav).unsqueeze(0).to(device)
    return F.layer_norm(x, x.shape[1:])


def enforce_vram(device, limit=VRAM_MIB):
    import torch

    if device.type != "cuda":
        return 0
    peak = torch.cuda.max_memory_reserved(device) / 2**20
    if peak > limit:
        raise RuntimeError(f"GPU budget exceeded: {peak:.1f} > {limit} MiB reserved")
    return peak


def forward_frames(model, row, device, *, verify=True):
    import torch

    if (
        device.type == "cuda"
        and Path(row["audio_path"]).stat().st_size > 16 * 16000 * 2
    ):
        # Stella's variable lengths otherwise leave decoder/attention allocator blocks reserved.
        torch.cuda.empty_cache()
    with torch.inference_mode():
        out = model(wav_input(row, device, verify=verify), output_hidden_states=True)
        lp = out.logits[0].log_softmax(dim=-1).cpu().numpy()
        h = (
            torch.stack([out.hidden_states[layer][0] for layer in LAYERS], dim=1)
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        if (
            lp.shape != (len(h), 79)
            or not np.isfinite(h).all()
            or not np.isfinite(lp).all()
        ):
            raise ValueError(f"Unexpected frame features: {row['uid']}")
    return h, lp


def fill_cache(rows, cache, device, backbone):
    import torch
    from research.CONTROLLED_PILOT_AB_20261118_v1.run_pilot_ab import WEIGHTS_DIR
    from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel

    if any(sha_file(WEIGHTS_DIR / name) != digest for name, digest in backbone.items()):
        raise RuntimeError("Backbone files changed since preflight")
    cache.mkdir(parents=True, exist_ok=True)
    binding = {
        "backbone": backbone,
        "layers": list(LAYERS),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda": torch.version.cuda,
        "dtype": "fp32 forward, float16 hidden storage, float32 lp",
        "code_sha256": sha_file(__file__),
    }
    card = cache / "FRAME_CARD.json"
    if card.exists():
        cached_binding = read_json(card)
        legacy_source = (
            "0b93d407d9bf24d064cace4f017a40d589689468d66268c1c7a57a41666e22d7"
        )
        if cached_binding not in (binding, {**binding, "code_sha256": legacy_source}):
            raise ValueError(f"Frame-cache identity mismatch: {cache}")
        # The legacy source differs only in allocator management, not fp32 frame content.
    else:
        save_json(card, binding)
    unique = {r["audio_sha256"]: r for r in rows}
    missing = [r for r in unique.values() if not frame_path(cache, r).is_file()]
    initial = len(missing)
    if not missing:
        print(f"Frame cache hit: {len(unique)} unique WAVs", flush=True)
        return {"new": 0, "unique": len(unique), "peak_reserved_mib": 0}
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = load_base_koel(device)
    for i, r in enumerate(missing, 1):
        h, lp = forward_frames(model, r, device)
        peak = enforce_vram(device)
        path = frame_path(cache, r)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        with temp.open("xb") as f:
            np.savez(f, hidden=h, lp=lp)
        if path.exists():
            raise RuntimeError(f"Concurrent output: {path}")
        temp.rename(path)
        if i % 200 == 0 or i == len(missing):
            print(
                f"FRAME {i}/{len(missing)} ({time.perf_counter() - start:.1f}s; peak {peak:.1f} MiB)",
                flush=True,
            )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "new": initial,
        "unique": len(unique),
        "peak_reserved_mib": peak,
        "extractor_sha256": binding["code_sha256"],
        "wall_s": round(time.perf_counter() - start, 3),
    }


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()
    if args.cache and os.environ.get("HF_HUB_OFFLINE") != "1":
        p.error("Set HF_HUB_OFFLINE=1")
    audit = prepare(args.out_dir)
    if args.cache:
        ret = fill_cache(
            admitted(args.out_dir, "fit.jsonl") + admitted(args.out_dir, "cal.jsonl"),
            args.cache,
            torch.device(args.device),
            audit["backbone"],
        )
        save_json(args.out_dir / "FIT_FRAME_RECEIPT.json", ret)


if __name__ == "__main__":
    main()
