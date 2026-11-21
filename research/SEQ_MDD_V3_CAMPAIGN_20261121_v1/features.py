"""V2-compatible frame pooling and pre-registered target/gap ablations."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    CTX_BOS_ID,
    CTX_EOS_ID,
    PHONE_TO_ID,
)
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import ALLOPHONE_TOKEN_IDS
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import SeqMDDv2
from .data import HERE

OFFSETS = np.arange(-4, 5)
POINTS = np.array([0, 2, 4, 6, 8])
FOUR_LAYERS = [
    0,
    1,
    2,
    5,
]  # L12, L15, L18, L24 within the frozen six-layer frame tensor.
_KEEP_GAP = ("h_gap", "lp_gap_mean", "lp_gap_max", "gap_scalars")


@dataclass(frozen=True)
class Spec:
    window: str = "v2"  # v2 | mean (isolated window control) | full
    l1_canon: bool = False
    gap_core: bool = False
    focal: bool = False
    ollr: bool = False
    fuse_realized: bool = False
    corpus_balance: bool = False
    temperature: bool = False

    def as_dict(self):
        return asdict(self)


def build_kernel(path: Path) -> Path:
    if not path.is_file():
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O3",
                "-shared",
                "-fPIC",
                str(HERE / "ctc_delete.cpp"),
                "-o",
                str(path),
            ],
            check=True,
        )
    return path


@lru_cache(maxsize=8)
def _ctc_fn(path):
    f = ctypes.CDLL(str(Path(path).resolve(strict=True))).ctc_delete
    da = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    f.argtypes = (da, ctypes.c_int, ctypes.c_int, ia, ctypes.c_int, da)
    f.restype = None
    return f


def deletion_logprobs(lp, tokens, kernel):
    a = np.ascontiguousarray(lp, dtype=np.float64)
    ids = np.ascontiguousarray(tokens, dtype=np.int32)
    if (
        a.ndim != 2
        or not len(a)
        or ids.ndim != 1
        or not len(ids)
        or np.min(ids) < 1
        or np.max(ids) >= a.shape[1]
    ):
        raise ValueError("Invalid CTC frame matrix or canonical tokens")
    out = np.empty(len(ids) + 1, np.float64)
    _ctc_fn(kernel)(a, *a.shape, ids, len(ids), out)
    return out


def deletion_feature(lp, phones, kernel):
    # Exact for one fixed canonical Koel token per ARPA-39 phone (first entry of the pinned map),
    # not a sum over allophone transcripts. Use zero + availability mask for impossible parses.
    a = deletion_logprobs(lp, [ALLOPHONE_TOKEN_IDS[p][0] for p in phones], kernel)
    valid = np.isfinite(a[0]) & np.isfinite(a[1:])
    delta = np.zeros(len(phones), np.float32)
    if np.isfinite(a[0]):
        delta[valid] = np.clip(a[1:][valid] - a[0], -80, 80)
    return np.column_stack([delta, valid]).astype(np.float32)


def voronoi_bounds(peaks, T):
    p = np.asarray(peaks)
    if not len(p) or T < 1 or (np.diff(p) < 0).any() or p.min() < 0 or p.max() >= T:
        raise ValueError("CTC peaks must be monotone and inside the waveform")
    split = np.r_[0, (p[1:] + p[:-1]) // 2 + 1, T]
    lo = split[:-1]
    hi = split[1:]
    # Fallback alignment may repeat peaks. Retain at least that frame for each target.
    return list(zip(np.minimum(lo, p), np.maximum(hi, p + 1)))


def gap_indices(raw_s, raw_e, bridge, T, lp, competitor, core):
    if raw_e < raw_s:
        return (
            np.empty(0, dtype=int)
            if core
            else np.arange(max(0, bridge - 1), min(T, bridge + 2))
        )
    ids = np.arange(raw_s, raw_e + 1)
    if not core:
        return ids
    # Normalize for all eligible phones; pick the two frame-wise strongest competitors.
    scores = lp[ids][:, competitor].max(axis=1)
    best = np.argsort(-scores, kind="stable")[:2]
    return ids[best]


def pool_features(h, lp, phones, spec=Spec(), kernel=None):
    """Frozen arm: return compact float16 pooled states and float32 scalars, no labels."""
    T, L = len(lp), len(phones)
    if h.ndim != 3 or len(h) != T or not L or not T or lp.shape[1] != 79:
        raise ValueError("Frame/slot shape mismatch")
    spans, _ = ctc_viterbi_align_allophone_aware(lp, phones)
    s, e, peaks = np.asarray(spans).T
    norm = max(T / L, 1e-3)
    ids = [PHONE_TO_ID[p] for p in phones]
    t_win = np.clip(peaks[:, None] + OFFSETS, 0, T - 1)
    h_vit = np.stack(
        [
            h[a : b + 1].astype(np.float32).mean(0).astype(np.float16)
            for a, b in zip(s, e)
        ]
    )
    h_peak = h[peaks]
    lp_vit = np.stack([lp[a : b + 1].mean(0).astype(np.float16) for a, b in zip(s, e)])
    ent = -np.sum(np.exp(lp) * lp, axis=-1)
    lp_by_phone = {}
    tokens = np.arange(lp.shape[1])
    for p in set(phones):
        can = ALLOPHONE_TOKEN_IDS[p]
        other = (tokens != 0) & ~np.isin(tokens, can)
        lp_by_phone[p] = (
            lp[:, can[0]] if len(can) == 1 else lp[:, list(can)].max(1),
            lp[:, other].max(1),
        )
    target = []
    for i, (vs, ve, _) in enumerate(spans):
        z = slice(vs, ve + 1)
        a, b = lp_by_phone[phones[i]]
        mx = a[z].max()
        omx = b[z].max()
        target.append(
            [
                np.log1p(ve - vs + 1),
                (ve - vs + 1) / norm,
                (vs - (e[i - 1] if i else 0)) / norm,
                ((s[i + 1] if i + 1 < L else T - 1) - ve) / norm,
                float(not i),
                float(i == L - 1),
                i / max(L - 1, 1),
                mx,
                a[z].mean(),
                lp[z, 0].mean(),
                omx,
                omx - mx,
                ent[z].mean(),
            ]
        )
    target = np.array(target, np.float32)
    result = {}
    if spec.window == "mean":
        h_vit = h[t_win[:, POINTS]].astype(np.float32).mean(axis=1).astype(np.float16)
    elif spec.window == "full":
        body = voronoi_bounds(peaks, T)
        h_vit = np.stack(
            [
                h[a:b, FOUR_LAYERS].astype(np.float32).mean(0).astype(np.float16)
                for a, b in body
            ]
        )
        window = h[t_win[:, POINTS]][:, :, FOUR_LAYERS]
        h_peak = window.astype(np.float32).mean(axis=1).astype(np.float16)
        lp_vit = np.stack([lp[a:b].mean(0).astype(np.float16) for a, b in body])
        left = np.r_[0, peaks[:-1]]
        right = np.r_[peaks[1:], T - 1]
        target[:, :2] = np.column_stack(
            [np.log1p(right - left), (right - peaks) / norm]
        )
        result["h_window"] = window
    elif spec.window != "v2":
        raise ValueError(f"Unknown window {spec.window}")
    if spec.ollr:
        if kernel is None:
            raise ValueError("Need exact CTC deletion kernel")
        target = np.column_stack([target, deletion_feature(lp, phones, kernel)])
    result.update(
        h_vit=h_vit,
        h_peak=h_peak,
        lp_win=lp[t_win].astype(np.float16),
        lp_vit=lp_vit,
        phone_ids=np.array(ids),
        prev_phone_ids=np.array([CTX_BOS_ID, *ids[:-1]]),
        next_phone_ids=np.array([*ids[1:], CTX_EOS_ID]),
        target_scalars=target,
    )
    hh, mean, mx, sc = [], [], [], []
    for k in range(L + 1):
        a = e[k - 1] + 1 if k else 0
        b = s[k] - 1 if k < L else T - 1
        duration = max(0, b - a + 1)
        bridge = e[k - 1] if k else s[0]
        exclude = {0}
        if k:
            exclude.update(ALLOPHONE_TOKEN_IDS[phones[k - 1]])
        if k < L:
            exclude.update(ALLOPHONE_TOKEN_IDS[phones[k]])
        if not k:
            exclude.add(64)
        comp = ~np.isin(tokens, list(exclude))
        idx = gap_indices(a, b, bridge, T, lp, comp, spec.gap_core)
        if len(idx):
            span = lp[idx]
            hh.append(h[idx].astype(np.float32).mean(0).astype(np.float16))
            mean.append(span.mean(0))
            mx.append(span.max(0))
            scalars = [
                span[:, comp].max(),
                span[:, 1:].max(),
                span[:, 0].mean(),
                ent[idx].mean(),
            ]
        else:
            hh.append(np.zeros_like(h[0]))
            empty = np.r_[0.0, np.full(78, -25.0, np.float32)]
            mean.append(empty)
            mx.append(empty)
            scalars = [-25.0, -25.0, 0.0, 0.0]
        sc.append(
            [
                np.log1p(duration),
                duration / norm,
                *scalars,
                float(k == 0),
                float(k == L),
            ]
        )
    result.update(
        h_gap=np.asarray(hh),
        lp_gap_mean=np.asarray(mean, np.float16),
        lp_gap_max=np.asarray(mx, np.float16),
        gap_scalars=np.asarray(sc, np.float32),
    )
    return result


class WindowHead(SeqMDDv2):
    """Keep V2 head size and loss, adding acoustic attention only in the full window arm."""

    def __init__(self, spec=Spec(), h_dim=1024, d_model=192, dropout=0.30):
        super().__init__(
            h_dim=h_dim,
            d_model=d_model,
            dropout=dropout,
            target_scalar_dim=13 + 2 * spec.ollr,
        )
        self.spec = spec
        if spec.window == "full":
            self.target_layer_logits = nn.Parameter(torch.zeros(4))
            self.win_hidden = nn.Sequential(
                nn.Linear(h_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
            )
        self.register_buffer(
            "point_idx", torch.tensor(POINTS, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        *,
        h_vit,
        h_peak,
        lp_win,
        lp_vit,
        phone_ids,
        prev_phone_ids,
        next_phone_ids,
        target_scalars,
        h_gap,
        lp_gap_mean,
        lp_gap_max,
        gap_scalars,
        lengths,
        h_window=None,
    ):
        B, L, W, _ = lp_win.shape
        weights = self.target_layer_logits
        scale = self.target_layer_scale
        hv = self._mix_layers(h_vit, weights, scale)
        hp = self._mix_layers(h_peak, weights, scale)
        mask = (
            torch.arange(L, device=hv.device)[None, :] < lengths[:, None]
        ).unsqueeze(-1)
        den = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mu = (hv * mask).sum(dim=1, keepdim=True) / den
        var = ((hv - mu).square() * mask).sum(dim=1, keepdim=True) / den
        sd = var.clamp_min(
            1e-8
        ).sqrt()  # finite gradients even for singleton/zero-variance utterances
        e = torch.cat(
            [
                self.phone_emb(phone_ids),
                self.ctx_emb(prev_phone_ids),
                self.ctx_emb(next_phone_ids),
            ],
            -1,
        )
        pos = self.pos_emb(self.pos_ids).view(1, 1, W, 16).expand(B, L, -1, -1)
        f = self.win_fuse(torch.cat([self.proj_lp_win(lp_win), pos], -1))
        if h_window is not None:
            ha = self._mix_layers(h_window, weights, scale)
            ha = self.win_hidden((ha - mu.unsqueeze(2)) / sd.unsqueeze(2))
            f = torch.cat([f, f.index_select(2, self.point_idx) + ha], dim=2)
        q = self.query_proj(e).reshape(B * L, 1, self.d_model)
        kv = f.reshape(B * L, f.shape[2], self.d_model)
        attn, _ = self.local_attn(q, kv, kv, need_weights=False)
        slot = self.slot_proj(
            torch.cat(
                [
                    self.proj_h_vit((hv - mu) / sd),
                    self.proj_h_peak((hp - mu) / sd),
                    attn.reshape(B, L, self.d_model),
                    self.proj_lp_vit(lp_vit),
                    e,
                    self.target_scalar_proj(target_scalars),
                ],
                -1,
            )
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            slot, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.seq_encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
        out = self.seq_norm(out + slot)
        gh = (
            self._mix_layers(h_gap, self.gap_layer_logits, self.gap_layer_scale) - mu
        ) / sd
        ga = self.gap_ac_proj(torch.cat([gh, lp_gap_mean, lp_gap_max, gap_scalars], -1))
        left = torch.cat([self.bos_state.expand(B, 1, -1), out], dim=1)
        right = torch.cat(
            [out, self.eos_state.expand(B, 1, -1)], dim=1
        )  # V2 layout for like-for-like training.
        gap = self.gap_head(torch.cat([left, ga, right], dim=-1))
        return {
            "target_logits": self.target_head(out),
            "gap_logits": gap,
            "realized_logits": self.realized_head(out),
        }
