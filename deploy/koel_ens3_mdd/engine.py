"""End-to-end WAV -> C/S/O/I Pronunciation MDD Inference Engine (S0_clean_ens3_o5_o4).

Single frozen Base Koel forward pass + 3-seed SingleFactorV2 (iso_no_hwin) Head Ensemble
with Cal-locked OrthoPolicy (prob_mean, tau_t=0.35, Gap Platt T=1.191 / b=-2.570, tau_g=0.38).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import expit, softmax
import soundfile as sf
import torch
import torch.nn.functional as F

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    ARPA_39,
    normalize_canonical_phone,
)
from research.KOEL_MDD_BRANCH_ABLATION_20261119_v1.models import load_base_koel
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import XLSR_SELECTED_LAYERS
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import sha_file
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.run_single_factor_isolation import GAP_KEYS
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.single_factor_models import (
    FACTOR_ARMS,
    SingleFactorV2,
    pool_features_with_hvit,
)

DEFAULT_BUNDLE_PATH = Path("deploy/koel_ens3_mdd/bundle_s0_clean_ens3_o5_o4.pt")
DEFAULT_ENSEMBLE_CARD = Path(
    "research/ORTHO_IMPROVEMENTS_20261122_v1/runs/s0_clean_ens3_o5_o4/ENSEMBLE_CARD.json"
)
REALIZED_VOCAB: Tuple[str, ...] = tuple(ARPA_39) + ("<DEL>",)


def build_bundle(
    ensemble_card_path: Path = DEFAULT_ENSEMBLE_CARD,
    out_path: Path = DEFAULT_BUNDLE_PATH,
) -> Path:
    """Pack the 3 verified S0_clean checkpoints + locked O5+O4 Cal policy into one .pt bundle."""
    card = json.loads(ensemble_card_path.read_text(encoding="utf-8"))
    state_dicts = []
    source_provenance = []
    norm_ref = None

    for item in card["source_checkpoints"]:
        cp = Path(item["path"])
        actual_sha = sha_file(cp)
        if actual_sha != item["sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch on {cp}: {actual_sha} != {item['sha256']}")
        ckpt = torch.load(cp, map_location="cpu", weights_only=False)
        state_dicts.append({k: v.detach().cpu() for k, v in ckpt["state_dict"].items()})
        source_provenance.append({"seed": int(item["seed"]), "path": str(cp), "sha256": actual_sha})
        if norm_ref is None:
            norm_ref = {
                k: (
                    np.asarray(v[0], dtype=np.float32).tolist(),
                    np.asarray(v[1], dtype=np.float32).tolist(),
                )
                for k, v in ckpt["norm"].items()
            }

    bundle = {
        "format_version": "koel_ens3_mdd_v1",
        "name": card["name"],
        "split": card["split"],
        "arch": "SingleFactorV2:iso_no_hwin",
        "arch_config": FACTOR_ARMS["iso_no_hwin"].as_dict(),
        "ensemble_card_sha256": sha_file(ensemble_card_path),
        "source_checkpoints": source_provenance,
        "norm": norm_ref,
        "policy": card["policy"],
        "gap_calibration": card["gap_calibration"],
        "state_dicts": state_dicts,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    return out_path


class KoelEns3MDDEngine:
    """Self-contained end-to-end pronunciation MDD inference engine (WAV -> C/S/O/I)."""

    def __init__(
        self,
        bundle_path: Path = DEFAULT_BUNDLE_PATH,
        *,
        device: str = "cuda",
        load_koel: bool = True,
    ) -> None:
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.bundle_path = Path(bundle_path)
        self.bundle = torch.load(self.bundle_path, map_location="cpu", weights_only=False)
        self.policy: Dict[str, Any] = dict(self.bundle["policy"])
        self.norm: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            k: (np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32))
            for k, v in self.bundle["norm"].items()
        }

        cfg = FACTOR_ARMS["iso_no_hwin"]
        self.heads: List[SingleFactorV2] = []
        for sd in self.bundle["state_dicts"]:
            m = SingleFactorV2(cfg).to(self.device)
            m.load_state_dict(sd)
            m.eval()
            self.heads.append(m)

        self.koel = load_base_koel(self.device).eval() if load_koel else None

    @staticmethod
    def _normalize_phones(canonical_phones: Sequence[str]) -> List[str]:
        out = [normalize_canonical_phone(p) for p in canonical_phones]
        if not out:
            raise ValueError("canonical_phones must be non-empty")
        return out

    def _collate_single(self, feat: Dict[str, np.ndarray], length: int) -> Dict[str, torch.Tensor]:
        args: Dict[str, torch.Tensor] = {}
        for name, arr in feat.items():
            extra = 1 if name in GAP_KEYS else 0
            if name.endswith("phone_ids"):
                dtype = np.int64
            elif name in {"h_win", "h_peak", "h_vit", "h_gap"}:
                dtype = np.float16
            else:
                dtype = np.float32
            v = arr
            if name in self.norm:
                m, s = self.norm[name]
                v = (v - m) / s
            data = np.zeros((1, length + extra) + arr.shape[1:], dtype=dtype)
            data[0, : length + extra] = v
            args[name] = torch.from_numpy(data).to(self.device)
        args["lengths"] = torch.tensor([length], dtype=torch.long, device=self.device)
        return args

    @torch.inference_mode()
    def score_frames(
        self,
        h: np.ndarray,
        lp: np.ndarray,
        canonical_phones: Sequence[str],
    ) -> Dict[str, Any]:
        """Score pre-extracted Koel frames (h: [T,6,1024], lp: [T,79]) with the 3-head ensemble."""
        phones = self._normalize_phones(canonical_phones)
        L = len(phones)

        t_pool0 = time.perf_counter()
        feat = pool_features_with_hvit(h, lp, phones)
        inp = self._collate_single(feat, L)
        pool_ms = 1000.0 * (time.perf_counter() - t_pool0)

        t_head0 = time.perf_counter()
        t_probs_list, g_probs_list, r_probs_list = [], [], []
        for head in self.heads:
            out = head(**inp)
            t_lg = out["target_logits"][0, :L].float().cpu().numpy()
            g_lg = out["gap_logits"][0, : L + 1].float().cpu().numpy()
            r_lg = out["realized_logits"][0, :L].float().cpu().numpy()
            t_probs_list.append(softmax(t_lg, axis=-1))
            g_probs_list.append(softmax(g_lg, axis=-1)[:, 1])
            r_probs_list.append(softmax(r_lg, axis=-1))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        head_ms = 1000.0 * (time.perf_counter() - t_head0)

        t_probs = np.mean(t_probs_list, axis=0)  # [L, 3]
        g_raw_prob = np.clip(np.mean(g_probs_list, axis=0), 1e-6, 1.0 - 1e-6)  # [L+1]
        r_probs = np.mean(r_probs_list, axis=0)  # [L, 40]

        # Apply locked Cal OrthoPolicy (tau_t, rho_o, Gap Platt T_gap/b_gap, tau_g)
        tau_t = float(self.policy["tau_t"])
        tau_g = float(self.policy["tau_g"])
        rho_o = float(self.policy.get("rho_o", 0.50))
        g_temp = float(self.policy.get("gap_temp", 1.0))
        g_bias = float(self.policy.get("gap_bias", 0.0))

        err_scores = t_probs[:, 1] + t_probs[:, 2]
        frac_o = t_probs[:, 2] / np.maximum(1e-12, err_scores)
        t_calls = np.where(err_scores >= tau_t, np.where(frac_o >= rho_o, "O", "S"), "C")

        g_z = np.log(g_raw_prob / (1.0 - g_raw_prob))
        g_cal_prob = expit((g_z + g_bias) / g_temp)
        g_calls = np.where(g_cal_prob >= tau_g, "I", "N")

        targets = []
        for i in range(L):
            r_top = int(np.argmax(r_probs[i]))
            targets.append({
                "index": i,
                "canonical_phone": phones[i],
                "call": str(t_calls[i]),
                "error_prob": round(float(err_scores[i]), 6),
                "p_s": round(float(t_probs[i, 1]), 6),
                "p_o": round(float(t_probs[i, 2]), 6),
                "realized_top1": REALIZED_VOCAB[r_top],
            })

        gaps = []
        for i in range(L + 1):
            left = phones[i - 1] if i > 0 else "<BOS>"
            right = phones[i] if i < L else "<EOS>"
            gaps.append({
                "index": i,
                "between": [left, right],
                "call": str(g_calls[i]),
                "insertion_prob": round(float(g_cal_prob[i]), 6),
            })

        n_sub = sum(1 for x in targets if x["call"] == "S")
        n_omi = sum(1 for x in targets if x["call"] == "O")
        n_ins = sum(1 for x in gaps if x["call"] == "I")
        n_err = n_sub + n_omi + n_ins
        return {
            "targets": targets,
            "gaps": gaps,
            "summary": {
                "n_phones": L,
                "n_errors": n_err,
                "n_sub": n_sub,
                "n_omi": n_omi,
                "n_ins": n_ins,
                "target_correct_pct": round(100.0 * (L - n_sub - n_omi) / max(1, L), 2),
            },
            "timing_ms": {
                "pool_ms": round(pool_ms, 2),
                "ens3_head_ms": round(head_ms, 2),
            },
        }

    @torch.inference_mode()
    def score_audio_array(
        self,
        wav_16k: np.ndarray,
        canonical_phones: Sequence[str],
    ) -> Dict[str, Any]:
        """Run full end-to-end Base Koel + 3-head ensemble on a 16 kHz mono float32 waveform."""
        if self.koel is None:
            raise RuntimeError("Koel backbone not loaded (pass load_koel=True)")
        wav = np.asarray(wav_16k, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if len(wav) < 400 or not np.isfinite(wav).all():
            raise ValueError("Invalid waveform: too short (<400 samples) or non-finite")

        t0 = time.perf_counter()
        x = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        x = F.layer_norm(x, x.shape[1:])
        out = self.koel(x, output_hidden_states=True)
        lp = out.logits[0].log_softmax(dim=-1).cpu().numpy()
        h = (
            torch.stack([out.hidden_states[layer][0] for layer in XLSR_SELECTED_LAYERS], dim=1)
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        koel_ms = 1000.0 * (time.perf_counter() - t0)

        res = self.score_frames(h, lp, canonical_phones)
        res["timing_ms"]["koel_ms"] = round(koel_ms, 2)
        res["timing_ms"]["total_ms"] = round(
            koel_ms + res["timing_ms"]["pool_ms"] + res["timing_ms"]["ens3_head_ms"], 2
        )
        res["duration_s"] = round(len(wav) / 16000.0, 3)
        return res

    def score_wav(
        self,
        wav_path: str | Path,
        canonical_phones: Sequence[str],
    ) -> Dict[str, Any]:
        """Load a 16 kHz WAV from disk and run full end-to-end MDD scoring."""
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected 16000 Hz WAV, got {sr} Hz: {wav_path}")
        return self.score_audio_array(wav, canonical_phones)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a 16kHz WAV with KoelEns3MDDEngine.")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--phones", type=str, required=True, help="Space-separated ARPAbet phones")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    engine = KoelEns3MDDEngine(args.bundle, device=args.device, load_koel=True)
    res = engine.score_wav(args.wav, args.phones.split())
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
