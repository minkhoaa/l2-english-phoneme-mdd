"""Single-backbone-resident GPU feature & alignment extractors with atomic resumable caches.

Supports 4 complementary pretrained speech encoders (all 100% offline local snapshots):
  - wavlm_large     : microsoft/wavlm-large, layers (6, 9, 12, 15, 18, 21), dim 1024, tiled FP32
  - wavlm_base_plus : microsoft/wavlm-base-plus, layers (4, 6, 8, 10, 12), dim 768, tiled FP32
  - huper_mid       : local HuPER WavLM-Large phonetic checkpoint, layers (6, 9, 12, 15, 18, 21), dim 1024, tiled FP32
  - whisper_small   : openai/whisper-small encoder, layers (2, 4, 6, 8, 10, 12), dim 768, 30s chunked FP32

Plus:
  - charsiu_10ms    : local CharSiu 10ms Wav2Vec2 forced aligner -> real phone start/end durations
  - fit_prototypes  : Fit-only per-phone L1/L2 cosine prototype features (0 Cal/Eval leakage)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gc
import hashlib
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import ARPA_39
from research.KOEL_SM13_BENCHMARK_20261120_v1.infer_sm13 import FORMAL as SM13_FORMAL
from research.KOEL_TOP_LAYER_ABLATION_20261120_v1.run_machine_blind_listening import (
    HUPER_RECEIPT,
    HUPER_VENDOR,
    HUPER_WEIGHT,
)
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import (
    enforce_vram,
    frame_path,
    read_json,
    sha_file,
)
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.features import voronoi_bounds
from research.SYNTH_3DIR_ISO_V1_20261122.huper_attention import use_tiled_attention

from .common import check_disk_guard

PHONE_TO_ID = {p: i for i, p in enumerate(ARPA_39)}
VRAM_CAP_MIB = 3400
PINNED_REVISIONS = {
    "microsoft/wavlm-base-plus": "4c66d4806a428f2e922ccfa1a962776e232d487b",
    "microsoft/wavlm-large": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
}


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    kind: str  # "wavlm" | "huper" | "whisper"
    repo_or_path: str
    layers: Tuple[int, ...]
    hidden_dim: int
    frame_layers_stage3: Tuple[int, ...]


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "wavlm_large": BackboneSpec(
        name="wavlm_large",
        kind="wavlm",
        repo_or_path="microsoft/wavlm-large",
        layers=(6, 9, 12, 15, 18, 21),
        hidden_dim=1024,
        frame_layers_stage3=(6, 9, 12, 15),
    ),
    "wavlm_base_plus": BackboneSpec(
        name="wavlm_base_plus",
        kind="wavlm",
        repo_or_path="microsoft/wavlm-base-plus",
        layers=(4, 6, 8, 10, 12),
        hidden_dim=768,
        frame_layers_stage3=(4, 6, 8, 10),
    ),
    "huper_mid": BackboneSpec(
        name="huper_mid",
        kind="huper",
        repo_or_path=str(HUPER_WEIGHT),
        layers=(6, 9, 12, 15, 18, 21),
        hidden_dim=1024,
        frame_layers_stage3=(6, 9, 12, 15),
    ),
    "whisper_small": BackboneSpec(
        name="whisper_small",
        kind="whisper",
        repo_or_path="openai/whisper-small",
        layers=(2, 4, 6, 8, 10, 12),
        hidden_dim=768,
        frame_layers_stage3=(4, 6, 8, 10),
    ),
}


def release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slot_key(row: Mapping[str, Any]) -> str:
    """Unique deterministic key binding uid + audio_sha256 + canonical_phones."""
    raw = f"{row['uid']}|{row['audio_sha256']}|{' '.join(row['canonical_phones'])}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _slot_path(cache_dir: Path, row: Mapping[str, Any]) -> Path:
    k = _slot_key(row)
    return cache_dir / k[:2] / f"{k}.npz"


def _frame_path(cache_dir: Path, row: Mapping[str, Any]) -> Path:
    h = row["audio_sha256"]
    return cache_dir / h[:2] / f"{h}.npz"


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def koel_voronoi_spans(
    koel_cache: Path, row: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (starts, ends_exclusive, peaks, T_koel) for canonical_phones from immutable Koel cache."""
    return _koel_voronoi_spans(frame_path(koel_cache, row), tuple(row["canonical_phones"]))


@lru_cache(maxsize=7000)
def _koel_voronoi_spans(path, phones):
    with np.load(path, allow_pickle=False) as frames:
        lp = frames["lp"]
    t_koel = int(lp.shape[0])
    spans, _ = ctc_viterbi_align_allophone_aware(lp, list(phones))
    peaks = np.asarray([int(p) for _, _, p in spans], dtype=np.int32)
    vb = voronoi_bounds(peaks, t_koel)
    starts = np.asarray([int(a) for a, _ in vb], dtype=np.int32)
    ends = np.asarray([int(b) for _, b in vb], dtype=np.int32)
    return starts, ends, peaks, t_koel


def pool_slots_from_frames(
    frames: np.ndarray,  # (T_ext, L, D) float32
    starts: np.ndarray,
    ends: np.ndarray,
    peaks: np.ndarray,
    t_koel: int,
) -> Dict[str, np.ndarray]:
    """Pool (slot_mean, slot_peak, slot_std) [N, L, D] and (gap_mean, gap_max) [N+1, L, D] in FP16."""
    t_ext, n_layers, dim = frames.shape
    scale = float(t_ext) / max(1.0, float(t_koel))
    n = len(peaks)

    s_mean = np.empty((n, n_layers, dim), dtype=np.float16)
    s_peak = np.empty((n, n_layers, dim), dtype=np.float16)
    s_std = np.empty((n, n_layers, dim), dtype=np.float16)
    ext_peaks = np.empty(n, dtype=np.int32)

    for i in range(n):
        a = int(np.clip(round(starts[i] * scale), 0, t_ext - 1))
        b = int(np.clip(round(ends[i] * scale), a + 1, t_ext))
        p = int(np.clip(round(peaks[i] * scale), 0, t_ext - 1))
        ext_peaks[i] = p
        seg = frames[a:b]  # (W, L, D)
        s_mean[i] = seg.mean(axis=0).astype(np.float16)
        s_peak[i] = frames[p].astype(np.float16)
        s_std[i] = (seg.std(axis=0) if seg.shape[0] > 1 else np.zeros((n_layers, dim), np.float32)).astype(np.float16)

    g_mean = np.empty((n + 1, n_layers, dim), dtype=np.float16)
    g_max = np.empty((n + 1, n_layers, dim), dtype=np.float16)
    for g in range(n + 1):
        ga = 0 if g == 0 else int(ext_peaks[g - 1])
        gb = t_ext if g == n else int(ext_peaks[g]) + 1
        if gb <= ga:
            gb = min(t_ext, ga + 1)
        gseg = frames[ga:gb]
        g_mean[g] = gseg.mean(axis=0).astype(np.float16)
        g_max[g] = gseg.max(axis=0).astype(np.float16)

    return {
        "slot_mean": s_mean,
        "slot_peak": s_peak,
        "slot_std": s_std,
        "gap_mean": g_mean,
        "gap_max": g_max,
    }


def pinned_snapshot(repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    # The approved download omitted non-runtime files (README, training metadata).
    return Path(hf_hub_download(repo, "config.json", revision=PINNED_REVISIONS[repo], local_files_only=True)).parent


def model_identities() -> dict:
    files = {}
    for repo in PINNED_REVISIONS:
        snap = pinned_snapshot(repo)
        weights = snap / "model.safetensors"
        if not weights.is_file():
            weights = snap / "pytorch_model.bin"
        files[repo] = {p.name: sha_file(p) for p in (snap / "config.json", snap / "preprocessor_config.json", weights)}
    files["huper_mid"] = {
        "weight": sha_file(HUPER_WEIGHT),
        **{n: sha_file(HUPER_VENDOR / n) for n in ("config.json", "preprocessor_config.json")},
    }
    files["charsiu_10ms"] = {
        n: sha_file(SM13_FORMAL / n)
        for n in ("charsiu--en_w2v2_fc_10ms/config.json", "charsiu--en_w2v2_fc_10ms/pytorch_model.bin", "charsiu--tokenizer_en_cmu/vocab.json")
    }
    return {"revisions": PINNED_REVISIONS, "file_sha256": files}


def _load_backbone_model(spec: BackboneSpec, device: torch.device):
    if spec.kind == "wavlm":
        from transformers import AutoFeatureExtractor, WavLMModel

        snap = pinned_snapshot(spec.repo_or_path)
        fe = AutoFeatureExtractor.from_pretrained(snap, local_files_only=True)
        model = WavLMModel.from_pretrained(snap, local_files_only=True)
        model.eval().to(device=device, dtype=torch.float32)
        if device.type == "cuda":
            use_tiled_attention(model)
        return model, fe

    if spec.kind == "huper":
        from safetensors.torch import load_file
        from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMForCTC

        if sha_file(HUPER_WEIGHT) != read_json(HUPER_RECEIPT)["sha256"]:
            raise ValueError("HuPER checkpoint SHA mismatch")
        cfg = WavLMConfig.from_dict(read_json(HUPER_VENDOR / "config.json"))
        model = WavLMForCTC(cfg)
        model.load_state_dict(load_file(str(HUPER_WEIGHT), device="cpu"), strict=True)
        model.eval().to(device=device, dtype=torch.float32)
        if device.type == "cuda":
            use_tiled_attention(model)
        fe = Wav2Vec2FeatureExtractor.from_dict(read_json(HUPER_VENDOR / "preprocessor_config.json"))
        return model, fe

    if spec.kind == "whisper":
        from transformers import WhisperFeatureExtractor, WhisperModel

        snap = pinned_snapshot(spec.repo_or_path)
        fe = WhisperFeatureExtractor.from_pretrained(snap, local_files_only=True)
        full = WhisperModel.from_pretrained(snap, local_files_only=True, use_safetensors=True)
        encoder = full.encoder.eval().to(device=device, dtype=torch.float32)
        del full
        release_gpu()
        return encoder, fe

    raise ValueError(f"Unknown backbone kind: {spec.kind}")


@torch.inference_mode()
def forward_backbone_layers(
    spec: BackboneSpec,
    model: torch.nn.Module,
    fe: Any,
    wav: np.ndarray,
    layers: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    """Run one 16 kHz mono waveform through backbone and return (T_ext, len(layers), D) float32."""
    if spec.kind in {"wavlm", "huper"}:
        inp = fe(wav, sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(device=device, dtype=torch.float32) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states
        stacked = torch.stack([hs[i][0] for i in layers], dim=1)  # (T, L, D)
        arr = stacked.float().cpu().numpy()
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite output from {spec.name}")
        return arr

    if spec.kind == "whisper":
        # 50 Hz encoder; chunk into exact 30-second (480,000 sample) windows for Stella > 30s
        chunk_samples = 30 * 16000
        chunks_out: List[np.ndarray] = []
        n_total = len(wav)
        for start in range(0, max(1, n_total), chunk_samples):
            sub = wav[start : start + chunk_samples]
            n_frames = max(1, int(round(len(sub) / 320.0)))
            n_frames = min(n_frames, 1500)
            inp = fe(sub, sampling_rate=16000, return_tensors="pt")
            feats = inp["input_features"].to(device=device, dtype=torch.float32)
            out = model(input_features=feats, output_hidden_states=True)
            hs = out.hidden_states
            stacked = torch.stack([hs[i][0, :n_frames] for i in layers], dim=1)  # (n_frames, L, D)
            chunks_out.append(stacked.float().cpu().numpy())
        arr = np.concatenate(chunks_out, axis=0)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite output from {spec.name}")
        return arr

    raise ValueError(f"Unknown backbone kind: {spec.kind}")


def extract_slot_cache(
    backbone_name: str,
    rows: Sequence[Mapping[str, Any]],
    koel_cache: Path,
    out_cache_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Extract and atomically cache slot-pooled features for all missing rows."""
    spec = BACKBONE_SPECS[backbone_name]
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if not _slot_path(out_cache_dir, r).exists()]
    t0 = time.perf_counter()
    if not missing:
        return {
            "backbone": backbone_name,
            "total": len(rows),
            "extracted": 0,
            "reused": len(rows),
            "peak_mib": 0.0,
            "wall_s": 0.0,
        }

    # Group missing rows by audio_sha256 so shared audio (e.g. timit_core & ctrl) runs once!
    by_audio: Dict[str, List[Mapping[str, Any]]] = {}
    for r in missing:
        by_audio.setdefault(r["audio_sha256"], []).append(r)

    release_gpu()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, fe = _load_backbone_model(spec, device)
    peak = 0.0
    prev_long = False
    done_waves = 0
    done_rows = 0

    try:
        for group in by_audio.values():
            if done_waves % 100 == 0:
                check_disk_guard(out_cache_dir)
            wav_path = group[0]["audio_path"]
            wav, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000 or wav.ndim != 1 or not np.isfinite(wav).all():
                raise ValueError(f"Bad WAV {wav_path}: sr={sr}, shape={wav.shape}")
            is_long = len(wav) > 15 * 16000
            if device.type == "cuda" and (is_long or prev_long):
                torch.cuda.empty_cache()
            prev_long = is_long

            frames = forward_backbone_layers(spec, model, fe, wav, spec.layers, device)
            peak = max(peak, enforce_vram(device, limit=VRAM_CAP_MIB))

            for r in group:
                starts, ends, peaks, t_koel = koel_voronoi_spans(koel_cache, r)
                pooled = pool_slots_from_frames(frames, starts, ends, peaks, t_koel)
                _atomic_save_npz(_slot_path(out_cache_dir, r), **pooled)
                done_rows += 1

            done_waves += 1
            if done_waves == 1 or done_waves % 250 == 0 or done_waves == len(by_audio):
                print(
                    f"[{backbone_name}:slots] {done_waves}/{len(by_audio)} wavs "
                    f"({done_rows}/{len(missing)} rows) peak={peak:.1f} MiB "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
    finally:
        del model
        release_gpu()

    return {
        "backbone": backbone_name,
        "layers": list(spec.layers),
        "hidden_dim": spec.hidden_dim,
        "total": len(rows),
        "extracted": done_rows,
        "reused": len(rows) - len(missing),
        "unique_waves_run": done_waves,
        "peak_mib": peak,
        "wall_s": time.perf_counter() - t0,
    }


def extract_full_frame_cache(
    backbone_name: str,
    rows: Sequence[Mapping[str, Any]],
    out_cache_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Extract compact FP16 full-sequence 4-layer frames for Stage 3 Cross-Attention."""
    spec = BACKBONE_SPECS[backbone_name]
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    # Deduplicate by audio_sha256
    unique_waves: Dict[str, str] = {}
    for r in rows:
        if not _frame_path(out_cache_dir, r).exists():
            unique_waves[r["audio_sha256"]] = r["audio_path"]

    t0 = time.perf_counter()
    if not unique_waves:
        return {
            "backbone": backbone_name,
            "layers": list(spec.frame_layers_stage3),
            "extracted_waves": 0,
            "peak_mib": 0.0,
            "wall_s": 0.0,
        }

    release_gpu()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, fe = _load_backbone_model(spec, device)
    peak = 0.0
    prev_long = False
    done = 0

    try:
        for audio_sha, wav_path in unique_waves.items():
            if done % 100 == 0:
                check_disk_guard(out_cache_dir)
            wav, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000 or wav.ndim != 1 or not np.isfinite(wav).all():
                raise ValueError(f"Bad WAV {wav_path}")
            is_long = len(wav) > 15 * 16000
            if device.type == "cuda" and (is_long or prev_long):
                torch.cuda.empty_cache()
            prev_long = is_long

            frames = forward_backbone_layers(
                spec, model, fe, wav, spec.frame_layers_stage3, device
            )
            peak = max(peak, enforce_vram(device, limit=VRAM_CAP_MIB))
            _atomic_save_npz(
                out_cache_dir / audio_sha[:2] / f"{audio_sha}.npz",
                frames=frames.astype(np.float16),
            )
            done += 1
            if done == 1 or done % 250 == 0 or done == len(unique_waves):
                print(
                    f"[{backbone_name}:frames] {done}/{len(unique_waves)} wavs "
                    f"peak={peak:.1f} MiB ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
    finally:
        del model
        release_gpu()

    return {
        "backbone": backbone_name,
        "layers": list(spec.frame_layers_stage3),
        "hidden_dim": spec.hidden_dim,
        "extracted_waves": done,
        "peak_mib": peak,
        "wall_s": time.perf_counter() - t0,
    }


def read_slot_cache(cache_dir: Path, row: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    with np.load(_slot_path(cache_dir, row)) as z:
        return {k: z[k] for k in z.files}


def read_frame_cache(cache_dir: Path, row: Mapping[str, Any]) -> np.ndarray:
    with np.load(_frame_path(cache_dir, row)) as z:
        return z["frames"]  # (T, L, D) float16


# ---------------------------------------------------------------------------
# CharSiu 10ms Forced Aligner (real phone start/end boundaries)
# ---------------------------------------------------------------------------
def _viterbi_charsiu_spans(
    logits: np.ndarray, phones: Sequence[str], vocab: Mapping[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Model-estimated forced boundaries, confidence and duration on the 10 ms grid."""
    sil = int(vocab["[SIL]"])
    ids = [int(vocab[p]) for p in phones]
    lp = torch.log_softmax(torch.as_tensor(logits, dtype=torch.float32), dim=-1).numpy()
    t = lp.shape[0]
    n = len(ids)
    if not 0 < n <= t or not np.isfinite(lp).all():
        raise ValueError("Invalid CharSiu logits or more phones than frames")
    # SIL before, between and after the phones; starting in the first SIL must
    # not allow a path that has consumed no frames for the first phone.
    s_total = 2 * n + 1
    labels = np.full(s_total, sil, dtype=np.int64)
    labels[1::2] = ids
    dp = np.full((t, s_total), -np.inf, dtype=np.float32)
    back = np.zeros((t, s_total), dtype=np.int32)
    dp[0, :2] = lp[0, labels[:2]]

    # Precompute emit scores (t, s_total)
    emit = lp[:, labels]
    idx_all = np.arange(s_total, dtype=np.int32)
    next_phone = np.arange(3, s_total, 2, dtype=np.int32)
    for f in range(1, t):
        prev = dp[f - 1]
        best_val = prev.copy()
        best_idx = idx_all.copy()

        # transition s-1 -> s
        cand1 = prev[:-1]
        upd1 = cand1 > best_val[1:]
        best_val[1:] = np.where(upd1, cand1, best_val[1:])
        best_idx[1:] = np.where(upd1, idx_all[:-1], best_idx[1:])

        # Optional SIL between consecutive phones.
        if next_phone.size:
            cand2 = prev[next_phone - 2]
            upd2 = cand2 > best_val[next_phone]
            best_val[next_phone] = np.where(upd2, cand2, best_val[next_phone])
            best_idx[next_phone] = np.where(upd2, next_phone - 2, best_idx[next_phone])

        dp[f] = best_val + emit[f]
        back[f] = best_idx

    # A forced path must reach the last phone (optionally ending in silence).
    s = s_total - 2 + int(np.argmax(dp[-1, -2:]))
    path = np.empty(t, dtype=np.int32)
    for f in range(t - 1, -1, -1):
        path[f] = s
        s = int(back[f, s]) if f else s

    starts = np.zeros(n, dtype=np.float32)
    ends = np.zeros(n, dtype=np.float32)
    confs = np.zeros(n, dtype=np.float32)
    durs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ix = np.flatnonzero(path == 2 * i + 1)
        if ix.size:
            starts[i] = float(ix[0])
            ends[i] = float(ix[-1] + 1)
            confs[i] = float(lp[ix, ids[i]].mean())
            durs[i] = float(ix.size) * 0.01
        else:
            raise RuntimeError("CharSiu forced path skipped a target")
    return starts, ends, confs, durs


def extract_charsiu_bounds(
    rows: Sequence[Mapping[str, Any]],
    out_cache_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Extract hypothesized CharSiu boundaries; these are not human phone timings."""
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if not _slot_path(out_cache_dir, r).exists()]
    t0 = time.perf_counter()
    if not missing:
        return {"extracted": 0, "reused": len(rows), "peak_mib": 0.0, "wall_s": 0.0}

    from transformers import AutoTokenizer, Wav2Vec2ForCTC

    tok = AutoTokenizer.from_pretrained(SM13_FORMAL / "charsiu--tokenizer_en_cmu", local_files_only=True)
    vocab = tok.get_vocab()
    aligner = Wav2Vec2ForCTC.from_pretrained(
        SM13_FORMAL / "charsiu--en_w2v2_fc_10ms", local_files_only=True
    ).eval().to(device)

    by_audio: Dict[str, List[Mapping[str, Any]]] = {}
    for r in missing:
        by_audio.setdefault(r["audio_sha256"], []).append(r)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak = 0.0
    done_rows = 0
    done_waves = 0
    try:
        with torch.inference_mode():
            for group in by_audio.values():
                if done_waves % 100 == 0:
                    check_disk_guard(out_cache_dir)
                wav, sr = sf.read(group[0]["audio_path"], dtype="float32")
                if sr != 16000 or wav.ndim != 1 or not np.isfinite(wav).all():
                    raise ValueError(f"Bad CharSiu input: {group[0]['uid']}")
                if device.type == "cuda" and len(wav) > 15 * 16000:
                    torch.cuda.empty_cache()
                # CharSiu normalizes waveform zero-mean unit-var
                w = (wav - wav.mean()) / (wav.std() + 1e-6)
                t_in = torch.from_numpy(w).unsqueeze(0).to(device)
                logits = aligner(t_in).logits[0].float().cpu().numpy()
                peak = max(peak, enforce_vram(device, limit=VRAM_CAP_MIB))
                t_10ms = max(1, logits.shape[0])
                for r in group:
                    starts, ends, confs, durs = _viterbi_charsiu_spans(
                        logits, r["canonical_phones"], vocab
                    )
                    rel_pos = (0.5 * (starts + ends)) / float(t_10ms)
                    # Gap durations between phones (in seconds)
                    sil_prev = np.zeros(len(starts), dtype=np.float32)
                    sil_next = np.zeros(len(starts), dtype=np.float32)
                    if len(starts) > 1:
                        gaps_s = np.maximum(0.0, starts[1:] - ends[:-1]) * 0.01
                        sil_prev[1:] = gaps_s
                        sil_next[:-1] = gaps_s
                    # (N, 5): [log1p(dur*100), conf, sil_prev, sil_next, rel_pos]
                    feats = np.stack(
                        [
                            np.log1p(durs * 100.0),
                            np.clip(confs, -15.0, 0.0),
                            np.log1p(sil_prev * 100.0),
                            np.log1p(sil_next * 100.0),
                            rel_pos,
                        ],
                        axis=1,
                    ).astype(np.float32)
                    _atomic_save_npz(
                        _slot_path(out_cache_dir, r),
                        charsiu_feats=feats,
                        starts_10ms=starts,
                        ends_10ms=ends,
                        t_10ms=np.asarray([t_10ms], dtype=np.int32),
                    )
                    done_rows += 1
                done_waves += 1
                if done_waves == 1 or done_waves % 300 == 0 or done_waves == len(by_audio):
                    print(
                        f"[charsiu_10ms] {done_waves}/{len(by_audio)} wavs "
                        f"peak={peak:.1f} MiB ({time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )
    finally:
        del aligner
        release_gpu()

    return {
        "extracted": done_rows,
        "reused": len(rows) - len(missing),
        "peak_mib": peak,
        "wall_s": time.perf_counter() - t0,
    }


def read_charsiu_feats(cache_dir: Path, row: Mapping[str, Any]) -> np.ndarray:
    with np.load(_slot_path(cache_dir, row)) as z:
        return z["charsiu_feats"]  # (N, 5) float32


# ---------------------------------------------------------------------------
# Fit-only Per-Phone Cosine Prototypes (zero Cal/Eval leakage)
# ---------------------------------------------------------------------------
def fit_phone_prototypes(
    fit_rows: Sequence[Mapping[str, Any]],
    slot_cache_dir: Path,
    layer_idx: int = 2,
) -> Dict[str, np.ndarray]:
    """Fit L2-normalized per-phone prototypes strictly on Fit rows.

    Returns dict with:
      - proto_l1_c  : (39, D) clean L1 prototypes (from timit_train_l1_antishortcut C)
      - proto_l2_c  : (39, D) clean L2 prototypes (from L2 Fit C)
      - proto_l2_s  : (39, D) mispronounced L2 prototypes (from L2 Fit S)
    """
    sample = read_slot_cache(slot_cache_dir, fit_rows[0])["slot_mean"]
    dim = int(sample.shape[-1])
    sum_l1 = np.zeros((39, dim), dtype=np.float64)
    cnt_l1 = np.zeros(39, dtype=np.int64)
    sum_l2c = np.zeros((39, dim), dtype=np.float64)
    cnt_l2c = np.zeros(39, dtype=np.int64)
    sum_l2s = np.zeros((39, dim), dtype=np.float64)
    cnt_l2s = np.zeros(39, dtype=np.int64)

    for r in fit_rows:
        arr = read_slot_cache(slot_cache_dir, r)["slot_mean"][:, layer_idx, :].astype(np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        arr = arr / norms
        is_l1 = "timit" in str(r.get("source", "")).lower()
        for i, (ph, lbl) in enumerate(zip(r["canonical_phones"], r["gold_targets"], strict=True)):
            pid = PHONE_TO_ID[ph]
            if is_l1 and lbl == "C":
                sum_l1[pid] += arr[i]
                cnt_l1[pid] += 1
            elif not is_l1 and lbl == "C":
                sum_l2c[pid] += arr[i]
                cnt_l2c[pid] += 1
            elif not is_l1 and lbl == "S":
                sum_l2s[pid] += arr[i]
                cnt_l2s[pid] += 1

    def _norm_rows(s: np.ndarray, c: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros_like(s, dtype=np.float32)
        for i in range(39):
            if c[i] > 0:
                v = s[i] / c[i]
                out[i] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
            elif fallback is not None:
                out[i] = fallback[i]
        return out

    p_l1 = _norm_rows(sum_l1, cnt_l1)
    p_l2c = _norm_rows(sum_l2c, cnt_l2c, fallback=p_l1)
    p_l2s = _norm_rows(sum_l2s, cnt_l2s, fallback=p_l2c)
    return {
        "proto_l1_c": p_l1,
        "proto_l2_c": p_l2c,
        "proto_l2_s": p_l2s,
        "layer_idx": np.asarray([layer_idx], dtype=np.int32),
    }


def compute_prototype_features(
    row: Mapping[str, Any],
    slot_cache_dir: Path,
    prototypes: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Compute 6 prototype-cosine features per target slot: (N, 6) float32."""
    layer_idx = int(prototypes["layer_idx"][0])
    arr = read_slot_cache(slot_cache_dir, row)["slot_mean"][:, layer_idx, :].astype(np.float32)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)

    p_l1 = prototypes["proto_l1_c"]  # (39, D)
    p_l2c = prototypes["proto_l2_c"]
    p_l2s = prototypes["proto_l2_s"]

    pids = np.asarray([PHONE_TO_ID[ph] for ph in row["canonical_phones"]], dtype=np.int64)
    sim_l1_all = arr @ p_l1.T  # (N, 39)
    idx = np.arange(len(pids))
    cos_l1_can = sim_l1_all[idx, pids]
    sim_l1_masked = sim_l1_all.copy()
    sim_l1_masked[idx, pids] = -2.0
    cos_l1_best_other = sim_l1_masked.max(axis=1)
    cos_l1_margin = cos_l1_can - cos_l1_best_other

    cos_l2c_can = (arr * p_l2c[pids]).sum(axis=1)
    cos_l2s_can = (arr * p_l2s[pids]).sum(axis=1)
    cos_l2_diff = cos_l2c_can - cos_l2s_can

    return np.stack(
        [
            cos_l1_can,
            cos_l1_best_other,
            cos_l1_margin,
            cos_l2c_can,
            cos_l2s_can,
            cos_l2_diff,
        ],
        axis=1,
    ).astype(np.float32)
