"""Prepare SpeechOcean762 TRAIN manifest, verify zero leakage, extract frozen Base Koel cache,
and run the Pre-Train Acoustic Realism Audit (can_lp_vit_max vs clean_11 Fit L2).

Reads ONLY the official HuggingFace train parquet (NEVER touches test!).
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import (
    BASE_KOEL_CACHE,
    load_splits_no_eval_gold,
    pool_koel_iso,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    PHONE_TO_ID,
)
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import (
    CONTINUOUS_SCALAR_NAMES,
    XLSR_SELECTED_LAYERS,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import frame_path, save_json

SO762_TRAIN_PARQUET = Path(
    "/home/khoa/.cache/huggingface/hub/datasets--mispeech--speechocean762/"
    "snapshots/06385584fad212b26134c656fdd3ccf9f093f33e/data/train-00000-of-00001.parquet"
)
CAN_LP_IDX = CONTINUOUS_SCALAR_NAMES.index("can_lp_vit_max")
MARGIN_IDX = CONTINUOUS_SCALAR_NAMES.index("frame_margin_vit")
OMISSION_REALIZED_ID = 39


def _strip_phone(ph: str) -> str:
    return re.sub(r"[012\*]+$", "", str(ph).strip().upper())


def parse_so762_row(raw: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    """Parse one SpeechOcean762 TRAIN row into canonical_phones, 5-expert scores & misp tags."""
    audio_bytes = raw["audio"]["bytes"]
    audio_sha = hashlib.sha256(audio_bytes).hexdigest()
    spk = f"so762_spk_{str(raw['speaker']).strip()}"
    uid = f"so762_train_{idx:04d}_{spk}"

    can_phones: List[str] = []
    phone_scores: List[float] = []
    explicit_type: List[str] = []  # "", "S", "O", "UNK_ERR"
    explicit_real_id: List[int] = []  # -100 if unknown, 0..38 if S realized, 39 if O

    for w in raw["words"]:
        w_phones = w["phones"]
        w_scores = w["phones-accuracy"]
        misp_by_idx: Dict[int, str] = {}
        for m in w.get("mispronunciations") or []:
            m_idx = int(m["index"])
            misp_by_idx[m_idx] = str(m.get("pronounced-phone", "")).strip()

        for j, (ph_raw, sc) in enumerate(zip(w_phones, w_scores, strict=True)):
            ph = _strip_phone(ph_raw)
            if ph not in PHONE_TO_ID:
                raise ValueError(f"Unexpected phone {ph_raw!r} -> {ph!r} in row {idx}")
            can_phones.append(ph)
            phone_scores.append(round(float(sc), 4))

            if j in misp_by_idx:
                pron = misp_by_idx[j]
                if pron == "<DEL>":
                    explicit_type.append("O")
                    explicit_real_id.append(OMISSION_REALIZED_ID)
                elif pron == "<unk>" or pron == "":
                    explicit_type.append("UNK_ERR")
                    explicit_real_id.append(-100)
                else:
                    rph = _strip_phone(pron)
                    if rph in PHONE_TO_ID:
                        explicit_type.append("S")
                        explicit_real_id.append(PHONE_TO_ID[rph])
                    else:
                        explicit_type.append("UNK_ERR")
                        explicit_real_id.append(-100)
            else:
                explicit_type.append("")
                explicit_real_id.append(PHONE_TO_ID[ph] if float(sc) >= 2.0 else -100)

    return {
        "source": "so762_train",
        "uid": uid,
        "speaker": spk,
        "age": int(raw["age"]),
        "gender": str(raw["gender"]),
        "text": str(raw["text"]),
        "utt_accuracy": float(raw["accuracy"]),
        "audio_sha256": audio_sha,
        "canonical_phones": can_phones,
        "phone_scores": phone_scores,
        "explicit_type": explicit_type,
        "explicit_real_id": explicit_real_id,
        "is_native_l1": False,
    }


def _atomic_save_npz(path: Path, h: np.ndarray, lp: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.npz"
    np.savez(tmp, hidden=h, lp=lp)
    os.replace(tmp, path)


def prepare_and_audit(
    out_dir: Path,
    *,
    parquet_path: Path = SO762_TRAIN_PARQUET,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "koel_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Load clean_11 custody sets for zero-leakage check
    fit_rows, cal_rows, eval_prompts = load_splits_no_eval_gold("clean_11")
    protected_shas = {r["audio_sha256"] for r in (cal_rows + eval_prompts)}
    protected_spks = {str(r["speaker"]).upper() for r in (cal_rows + eval_prompts)}
    protected_uids = {str(r["uid"]) for r in (cal_rows + eval_prompts)}

    # 2. Parse SpeechOcean762 TRAIN parquet
    tbl = pq.read_table(parquet_path)
    so_rows: List[Dict[str, Any]] = []
    audio_blobs: List[bytes] = []
    for i in range(tbl.num_rows):
        raw = {c: tbl[c][i].as_py() for c in tbl.column_names}
        r = parse_so762_row(raw, i)
        assert r["audio_sha256"] not in protected_shas, f"SHA leak in {r['uid']}"
        assert r["speaker"].upper() not in protected_spks, f"Speaker leak in {r['uid']}"
        assert r["uid"] not in protected_uids, f"UID leak in {r['uid']}"
        so_rows.append(r)
        audio_blobs.append(raw["audio"]["bytes"])

    man_path = out_dir / "so762_train_manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for r in so_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Prepare] Parsed {len(so_rows)} SO762 TRAIN rows (0 Cal/Eval SHA/speaker/UID overlap)", flush=True)

    # 3. Extract frozen Base Koel frames (resumable atomic cache)
    missing_idx = [i for i, r in enumerate(so_rows) if not frame_path(cache_dir, r).exists()]
    print(f"[Koel] {len(so_rows) - len(missing_idx)} cached, {len(missing_idx)} to extract", flush=True)
    if missing_idx:
        dev = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
        koel = load_base_koel(dev)
        koel.eval()
        t_ex = time.perf_counter()
        with torch.inference_mode():
            for k, i in enumerate(missing_idx, 1):
                wav, sr = sf.read(io.BytesIO(audio_blobs[i]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                assert sr == 16000 and len(wav) >= 400 and np.isfinite(wav).all(), (
                    f"Bad wav on {so_rows[i]['uid']}"
                )
                x = torch.from_numpy(wav).unsqueeze(0).to(dev)
                x = torch.nn.functional.layer_norm(x, x.shape[1:])
                out = koel(x, output_hidden_states=True)
                lp = out.logits[0].log_softmax(dim=-1).cpu().numpy().astype(np.float16)
                h = (
                    torch.stack([out.hidden_states[layer][0] for layer in XLSR_SELECTED_LAYERS], dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                if lp.shape != (len(h), 79) or not np.isfinite(h).all() or not np.isfinite(lp).all():
                    raise ValueError(f"Non-finite Koel output on {so_rows[i]['uid']}")
                _atomic_save_npz(frame_path(cache_dir, so_rows[i]), h, lp)
                if k % 500 == 0 or k == len(missing_idx):
                    print(f"  [Koel SO762] {k}/{len(missing_idx)} ({time.perf_counter()-t_ex:.1f}s)", flush=True)
        del koel
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Pre-Train Acoustic Realism Audit (compare SO762 score buckets vs clean_11 Fit L2!)
    print("[Audit] Computing can_lp_vit_max & frame_margin realism vs clean_11 Fit L2...", flush=True)
    fit_l2_clps = {"C": [], "S": [], "O": []}
    for r in fit_rows:
        if str(r.get("source", "")).startswith("timit"):
            continue
        kf = pool_koel_iso(BASE_KOEL_CACHE, r)
        clps = kf["target_scalars"][:, CAN_LP_IDX]
        for g, v in zip(r["gold_targets"], clps):
            if g in fit_l2_clps:
                fit_l2_clps[g].append(float(v))

    so_buckets: Dict[str, List[float]] = {
        "score_2.0_C": [],
        "score_1.6_1.8": [],
        "score_1.0_1.4": [],
        "score_le_0.8": [],
        "explicit_S": [],
        "explicit_O": [],
        "explicit_UNK_ERR": [],
    }
    type_counts = Counter()
    for r in so_rows:
        kf = pool_koel_iso(cache_dir, r)
        clps = kf["target_scalars"][:, CAN_LP_IDX]
        for sc, et, v in zip(r["phone_scores"], r["explicit_type"], clps):
            fv = float(v)
            if et == "S":
                so_buckets["explicit_S"].append(fv)
                type_counts["explicit_S"] += 1
            elif et == "O":
                so_buckets["explicit_O"].append(fv)
                type_counts["explicit_O"] += 1
            elif et == "UNK_ERR":
                so_buckets["explicit_UNK_ERR"].append(fv)
                type_counts["explicit_UNK_ERR"] += 1
            if sc >= 2.0:
                so_buckets["score_2.0_C"].append(fv)
            elif sc >= 1.6:
                so_buckets["score_1.6_1.8"].append(fv)
            elif sc >= 1.0:
                so_buckets["score_1.0_1.4"].append(fv)
            else:
                so_buckets["score_le_0.8"].append(fv)

    def _stats(vals: Sequence[float]) -> Dict[str, Any]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(len(arr)),
            "p25": round(float(np.percentile(arr, 25)), 3),
            "median": round(float(np.median(arr)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
            "pct_gt_neg02": round(100.0 * float((arr > -0.2).mean()), 1),
        }

    report = {
        "n_utts": len(so_rows),
        "n_speakers": len({r["speaker"] for r in so_rows}),
        "n_adults_ge16": sum(r["age"] >= 16 for r in so_rows),
        "n_kids_lt16": sum(r["age"] < 16 for r in so_rows),
        "n_phones": sum(len(r["canonical_phones"]) for r in so_rows),
        "explicit_counts": dict(type_counts),
        "zero_leakage_verified": True,
        "wall_s": round(time.perf_counter() - t0, 1),
        "clean11_fit_l2_can_lp": {k: _stats(v) for k, v in fit_l2_clps.items()},
        "so762_train_can_lp": {k: _stats(v) for k, v in so_buckets.items()},
    }
    save_json(out_dir / "SO762_ACOUSTIC_AUDIT.json", report)
    print("\n=== PRE-TRAIN ACOUSTIC REALISM AUDIT (can_lp_vit_max) ===")
    print("  [clean_11 Fit L2 Reference]")
    for k, st in report["clean11_fit_l2_can_lp"].items():
        print(f"    {k:18s}: n={st['n']:5d} | p25/med/p75 = {st['p25']:6.2f} / {st['median']:6.2f} / {st['p75']:6.2f} | > -0.2: {st['pct_gt_neg02']:4.1f}%")
    print("  [SpeechOcean762 TRAIN Real L2]")
    for k, st in report["so762_train_can_lp"].items():
        print(f"    {k:18s}: n={st['n']:5d} | p25/med/p75 = {st['p25']:6.2f} / {st['median']:6.2f} / {st['p75']:6.2f} | > -0.2: {st['pct_gt_neg02']:4.1f}%")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research/REAL_L2_SO762_20261122_v1/runs/so762_prep"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    prepare_and_audit(args.out, device_str=args.device)


if __name__ == "__main__":
    main()
