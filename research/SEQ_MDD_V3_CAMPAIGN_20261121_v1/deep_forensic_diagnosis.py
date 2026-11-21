"""Reproducible CPU-only deep forensic diagnosis of SeqMDDv2 / LengthAwareV2.

Read-only over cached frozen Koel frames, admitted splits, archived checkpoints,
and sealed predictions. Never modifies any historical artifact or touches test_sealed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    HELD6_GOLD,
)
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    PHONE_TO_ID,
    normalize_canonical_phone,
)
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import ALLOPHONE_TOKEN_IDS
from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    dp_align_csoi_realized,
)

from .data import (
    HERE,
    LAYERS,
    OLD,
    V2_CKPT,
    admitted,
    read_frames,
    read_jsonl,
    save_json,
    sha_file,
)
from .features import Spec, pool_features
from .repair_model import LengthAwareV2
from .training import collate, domain, features_from_cache

DEFAULT_RUN = HERE / "runs/campaign_d5f8e89"
DEFAULT_REPAIR = HERE / "runs/v2_prompt_eos_repair_v1"
DEFAULT_OUT = HERE / "runs/deep_forensic_20261121_v1/DEEP_FORENSIC_DIAGNOSIS.json"


def _attach_eval_gold(ev_p, ev_g_rows):
    ev_g = {(r["cohort"], r["uid"]): r for r in ev_g_rows}
    out = []
    for r in ev_p:
        g = ev_g[(r["cohort"], r["uid"])]
        out.append({**r, "gold_targets": g["gold_targets"], "gold_gaps": g["gold_gaps"]})
    return out


def probe_layer_stats(cache, rows):
    peak_norms = [[] for _ in LAYERS]
    nonpeak_norms = [[] for _ in LAYERS]
    means = [[] for _ in LAYERS]
    stds = [[] for _ in LAYERS]
    for r in rows[:150]:
        h, lp = read_frames(cache, r)
        h32 = h.astype(np.float32)
        spans, _ = ctc_viterbi_align_allophone_aware(lp, r["canonical_phones"])
        peaks = np.array([p for _, _, p in spans])
        is_peak = np.zeros(len(h32), dtype=bool)
        is_peak[peaks] = True
        for li in range(len(LAYERS)):
            hl = h32[:, li, :]
            peak_norms[li].append(float(np.linalg.norm(hl[is_peak], axis=-1).mean()))
            if (~is_peak).any():
                nonpeak_norms[li].append(
                    float(np.linalg.norm(hl[~is_peak], axis=-1).mean())
                )
            means[li].append(float(hl.mean()))
            stds[li].append(float(hl.std()))
    return {
        f"L{layer_id}": {
            "peak_l2": float(np.mean(peak_norms[li])),
            "nonpeak_l2": float(np.mean(nonpeak_norms[li])),
            "mean": float(np.mean(means[li])),
            "std": float(np.mean(stds[li])),
        }
        for li, layer_id in enumerate(LAYERS)
    }


def probe_zero_train_koel_ap(cache, groups):
    out = {}
    prompts = {}
    for name, rows in groups.items():
        yt, s_can, s_mar, yg, s_gap = [], [], [], [], []
        by_spk = defaultdict(lambda: ([], []))
        pset = set()
        for r in rows:
            pset.add(tuple(r["canonical_phones"]))
            h, lp = read_frames(cache, r)
            f = pool_features(h, lp, r["canonical_phones"], Spec(), None)
            ts = f["target_scalars"]
            gs = f["gap_scalars"]
            for i, y in enumerate(r["gold_targets"]):
                if y == "unknown":
                    continue
                lab = int(y in ("S", "O"))
                yt.append(lab)
                s_can.append(-float(ts[i, 7]))
                s_mar.append(float(ts[i, 11]))
                by_spk[r["speaker"]][0].append(lab)
                by_spk[r["speaker"]][1].append(-float(ts[i, 7]))
            for k, y in enumerate(r["gold_gaps"]):
                if y == "unknown":
                    continue
                yg.append(int(y == "I"))
                s_gap.append(float(gs[k, 2]))
        spk_aps = {
            spk: float(average_precision_score(a, b))
            for spk, (a, b) in sorted(by_spk.items())
            if 0 < sum(a) < len(a)
        }
        out[name] = {
            "utterances": len(rows),
            "speakers": len(by_spk),
            "unique_prompts": len(pset),
            "error_rate": float(np.mean(yt)) if yt else None,
            "raw_neg_can_lp_ap": float(average_precision_score(yt, s_can)),
            "raw_margin_ap": float(average_precision_score(yt, s_mar)),
            "raw_gap_other_nb_ap": float(average_precision_score(yg, s_gap))
            if sum(yg) > 0
            else None,
            "speaker_ap_min_med_max": [
                float(min(spk_aps.values())),
                float(np.median(list(spk_aps.values()))),
                float(max(spk_aps.values())),
            ]
            if spk_aps
            else [],
            "speaker_aps": spk_aps if len(spk_aps) <= 8 else {},
        }
        prompts[name] = pset
    out["prompt_overlap"] = {
        "epadb_fit_cal": [
            len(prompts["fit_epadb"] & prompts["cal_epadb"]),
            len(prompts["cal_epadb"]),
        ],
        "epadb_fit_test": [
            len(prompts["fit_epadb"] & prompts["eval_epadb_test"]),
            len(prompts["eval_epadb_test"]),
        ],
        "l2a_fit_cal": [
            len(prompts["fit_l2arctic"] & prompts["cal_l2arctic"]),
            len(prompts["cal_l2arctic"]),
        ],
        "l2a_fit_held6": [
            len(prompts["fit_l2arctic"] & prompts["eval_held6"]),
            len(prompts["eval_held6"]),
        ],
    }
    return out


def probe_model_internals_and_knockout(cache, m_fix, norm_fix, cohorts):
    # 1. Internal tensor magnitudes on first 250 Held-6 utterances
    h6_sub = cohorts["eval_held6"][:250]
    h6_feats = features_from_cache(h6_sub, Spec(), cache, None, "forensic_h6_sub")
    stats = defaultdict(list)

    def rec(k, t):
        v = t.detach().float().reshape(-1)
        stats[k].append(
            (
                float(v.mean().item()),
                float(v.std().item()),
                float(v.abs().max().item()),
                float(t.detach().float().norm(dim=-1).mean().item()),
            )
        )

    with torch.no_grad():
        for r, f in zip(h6_sub, h6_feats, strict=True):
            inp, _ = collate([r], [f], norm_fix, torch.device("cpu"), labels=False)
            B, L, W, _ = inp["lp_win"].shape
            hv_raw = m_fix._mix_layers(
                inp["h_vit"], m_fix.target_layer_logits, m_fix.target_layer_scale
            )
            hp_raw = m_fix._mix_layers(
                inp["h_peak"], m_fix.target_layer_logits, m_fix.target_layer_scale
            )
            hg_raw = m_fix._mix_layers(
                inp["h_gap"], m_fix.gap_layer_logits, m_fix.gap_layer_scale
            )
            mu = hv_raw.mean(1, keepdim=True)
            sd = hv_raw.var(1, unbiased=False, keepdim=True).clamp_min(1e-8).sqrt()
            hv, hp, hg = (hv_raw - mu) / sd, (hp_raw - mu) / sd, (hg_raw - mu) / sd
            rec("hv_minus_hp_abs", (hv - hp).abs())
            rec("hg_cross_cmvn", hg)
            pos = m_fix.pos_emb(m_fix.pos_ids).view(1, 1, W, 16).expand(B, L, -1, -1)
            p_lpw = m_fix.proj_lp_win(inp["lp_win"])
            rec("lp_win_raw", inp["lp_win"])
            rec("proj_lp_win", p_lpw)
            rec("pos_emb", pos)
            e = torch.cat(
                [
                    m_fix.phone_emb(inp["phone_ids"]),
                    m_fix.ctx_emb(inp["prev_phone_ids"]),
                    m_fix.ctx_emb(inp["next_phone_ids"]),
                ],
                -1,
            )
            fwin = m_fix.win_fuse(torch.cat([p_lpw, pos], -1))
            attn, _ = m_fix.local_attn(
                m_fix.query_proj(e).reshape(B * L, 1, m_fix.d_model),
                fwin.reshape(B * L, W, m_fix.d_model),
                fwin.reshape(B * L, W, m_fix.d_model),
                need_weights=False,
            )
            attn = attn.reshape(B, L, m_fix.d_model)
            phv = m_fix.proj_h_vit(hv)
            php = m_fix.proj_h_peak(hp)
            plpv = m_fix.proj_lp_vit(inp["lp_vit"])
            psc = m_fix.target_scalar_proj(inp["target_scalars"])
            rec("slot_in_h_attn", attn)
            rec("slot_in_proj_lp_vit", plpv)
            slot = m_fix.slot_proj(torch.cat([phv, php, attn, plpv, e, psc], -1))
            seq_out, _ = m_fix.seq_encoder(slot)
            rec("slot_x", slot)
            rec("bigru_seq_out", seq_out)
            W_g = m_fix.gap_ac_proj[0].weight
            rec("gap_lin_hg", F.linear(hg, W_g[:, :1024]))
            rec("gap_lin_lp_mean", F.linear(inp["lp_gap_mean"], W_g[:, 1024:1103]))
            rec("gap_lin_lp_max", F.linear(inp["lp_gap_max"], W_g[:, 1103:1182]))
            rec("gap_lin_scalars", F.linear(inp["gap_scalars"], W_g[:, 1182:]))

    magnitudes = {}
    for k, vals in stats.items():
        arr = np.asarray(vals)
        magnitudes[k] = {
            "mean": float(arr[:, 0].mean()),
            "std": float(arr[:, 1].mean()),
            "maxabs": float(arr[:, 2].max()),
            "l2": float(arr[:, 3].mean()),
        }

    # 2. Full-cohort component knockout (compute shared projections once per batch)
    knocks = (None, "h_vit+h_peak", "h_attn", "lp_vit", "e_phone", "scalars", "BiGRU")

    def run_all_knocks(rows, feats):
        yt, yg, pg_full = [], [], []
        pt_by = {k: [] for k in knocks}
        with torch.no_grad():
            for i in range(0, len(rows), 32):
                rsub, fsub = rows[i : i + 32], feats[i : i + 32]
                inp, _ = collate(rsub, fsub, norm_fix, torch.device("cpu"), labels=False)
                B, L, W, _ = inp["lp_win"].shape
                lengths = inp["lengths"]
                hv_raw = m_fix._mix_layers(
                    inp["h_vit"], m_fix.target_layer_logits, m_fix.target_layer_scale
                )
                hp_raw = m_fix._mix_layers(
                    inp["h_peak"], m_fix.target_layer_logits, m_fix.target_layer_scale
                )
                mask = (torch.arange(L)[None, :] < lengths[:, None]).unsqueeze(-1)
                den = mask.sum(1, keepdim=True).clamp_min(1)
                mu = (hv_raw * mask).sum(1, keepdim=True) / den
                sd = (
                    ((hv_raw - mu).square() * mask)
                    .sum(1, keepdim=True)
                    .div(den)
                    .clamp_min(1e-8)
                    .sqrt()
                )
                hv, hp = (hv_raw - mu) / sd, (hp_raw - mu) / sd
                e = torch.cat(
                    [
                        m_fix.phone_emb(inp["phone_ids"]),
                        m_fix.ctx_emb(inp["prev_phone_ids"]),
                        m_fix.ctx_emb(inp["next_phone_ids"]),
                    ],
                    -1,
                )
                pos = (
                    m_fix.pos_emb(m_fix.pos_ids)
                    .view(1, 1, W, 16)
                    .expand(B, L, -1, -1)
                )
                fwin = m_fix.win_fuse(
                    torch.cat([m_fix.proj_lp_win(inp["lp_win"]), pos], -1)
                )
                attn, _ = m_fix.local_attn(
                    m_fix.query_proj(e).reshape(B * L, 1, m_fix.d_model),
                    fwin.reshape(B * L, W, m_fix.d_model),
                    fwin.reshape(B * L, W, m_fix.d_model),
                    need_weights=False,
                )
                attn = attn.reshape(B, L, m_fix.d_model)
                phv = m_fix.proj_h_vit(hv)
                php = m_fix.proj_h_peak(hp)
                plpv = m_fix.proj_lp_vit(inp["lp_vit"])
                psc = m_fix.target_scalar_proj(inp["target_scalars"])

                # Full pass + BiGRU knockout reuse the same slot & seq_out
                slot_full = m_fix.slot_proj(
                    torch.cat([phv, php, attn, plpv, e, psc], -1)
                )
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    slot_full, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                seq_out_full, _ = m_fix.seq_encoder(packed)
                seq_out_full, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    seq_out_full, batch_first=True, total_length=L
                )
                seq_full = m_fix.seq_norm(seq_out_full + slot_full)
                tp_by = {
                    None: F.softmax(m_fix.target_head(seq_full), -1).numpy(),
                    "BiGRU": F.softmax(
                        m_fix.target_head(m_fix.seq_norm(slot_full)), -1
                    ).numpy(),
                }
                for kn in ("h_vit+h_peak", "h_attn", "lp_vit", "e_phone", "scalars"):
                    parts = [
                        torch.zeros_like(phv) if kn == "h_vit+h_peak" else phv,
                        torch.zeros_like(php) if kn == "h_vit+h_peak" else php,
                        torch.zeros_like(attn) if kn == "h_attn" else attn,
                        torch.zeros_like(plpv) if kn == "lp_vit" else plpv,
                        torch.zeros_like(e) if kn == "e_phone" else e,
                        torch.zeros_like(psc) if kn == "scalars" else psc,
                    ]
                    sl = m_fix.slot_proj(torch.cat(parts, -1))
                    pk = torch.nn.utils.rnn.pack_padded_sequence(
                        sl, lengths.cpu(), batch_first=True, enforce_sorted=False
                    )
                    so, _ = m_fix.seq_encoder(pk)
                    so, _ = torch.nn.utils.rnn.pad_packed_sequence(
                        so, batch_first=True, total_length=L
                    )
                    tp_by[kn] = F.softmax(
                        m_fix.target_head(m_fix.seq_norm(so + sl)), -1
                    ).numpy()

                hg = (
                    m_fix._mix_layers(
                        inp["h_gap"], m_fix.gap_layer_logits, m_fix.gap_layer_scale
                    )
                    - mu
                ) / sd
                ga = m_fix.gap_ac_proj(
                    torch.cat(
                        [hg, inp["lp_gap_mean"], inp["lp_gap_max"], inp["gap_scalars"]],
                        -1,
                    )
                )
                left = torch.cat([m_fix.bos_state.expand(B, 1, -1), seq_full], 1)
                eos_idx = lengths.view(B, 1, 1).expand(-1, -1, m_fix.d_model)
                right = F.pad(seq_full, (0, 0, 0, 1)).scatter(
                    1, eos_idx, m_fix.eos_state.expand(B, 1, -1)
                )
                gp = F.softmax(m_fix.gap_head(torch.cat([left, ga, right], -1)), -1)[
                    ..., 1
                ].numpy()
                for bi, r in enumerate(rsub):
                    for idx_p, g in enumerate(r["gold_targets"]):
                        if g != "unknown":
                            yt.append(int(g in ("S", "O")))
                            for kn in knocks:
                                tp = tp_by[kn]
                                pt_by[kn].append(
                                    float(tp[bi, idx_p, 1] + tp[bi, idx_p, 2])
                                )
                    for idx_g, g in enumerate(r["gold_gaps"]):
                        if g != "unknown":
                            yg.append(int(g == "I"))
                            pg_full.append(float(gp[bi, idx_g]))
        ft = float(average_precision_score(yt, pt_by[None]))
        fg = float(average_precision_score(yg, pg_full))
        return {
            "full_target_ap": ft,
            "full_gap_ap": fg,
            "delta_target_ap": {
                kn: float(average_precision_score(yt, pt_by[kn])) - ft
                for kn in knocks
                if kn is not None
            },
        }

    knockout = {}
    for cname, rows in cohorts.items():
        feats = features_from_cache(rows, Spec(), cache, None, f"ko_{cname}")
        knockout[cname] = run_all_knocks(rows, feats)
    return magnitudes, knockout


def probe_ridge_information_ceiling(cache, fit, ev_p):
    def build_bank(rows, max_rows=None):
        if max_rows:
            rows = rows[:max_rows]
        X_scal, X_ph, X_lpp = [], [], []
        X_h18, X_h18_cmvn, X_h18_mean5, X_h18_w3, X_h18_w5 = [], [], [], [], []
        Y = []
        tokens = np.arange(79)
        for r in rows:
            h, lp = read_frames(cache, r)
            T = len(lp)
            can = r["canonical_phones"]
            spans, _ = ctc_viterbi_align_allophone_aware(lp, can)
            peaks = np.array([x[2] for x in spans])
            ts = pool_features(h, lp, can, Spec(), None)["target_scalars"]
            h18 = h[:, 2, :].astype(np.float32)
            h18_p = h18[peaks]
            h18_cmvn = (h18_p - h18_p.mean(0, keepdims=True)) / np.maximum(
                h18_p.std(0, keepdims=True), 1e-4
            )
            probs = np.exp(lp)
            lp_clip = np.clip(lp, -10.0, 0.0) / 10.0
            for i, y in enumerate(r["gold_targets"]):
                if y == "unknown":
                    continue
                vp = int(peaks[i])
                Y.append(int(y in ("S", "O")))
                X_scal.append(ts[i])
                oh = np.zeros(39, dtype=np.float32)
                oh[PHONE_TO_ID[can[i]]] = 1.0
                X_ph.append(oh)
                can_ids = list(ALLOPHONE_TOKEN_IDS[can[i]])
                other_m = (tokens != 0) & (~np.isin(tokens, can_ids))
                loc = []
                for d in (-3, -2, -1, 0, 1, 2, 3):
                    tt = min(T - 1, max(0, vp + d))
                    loc.extend(
                        [
                            float(probs[tt, can_ids].sum()),
                            float(probs[tt, 0]),
                            float(probs[tt, other_m].max()),
                            float(lp_clip[tt, can_ids].max()),
                            float(lp_clip[tt, other_m].max()),
                        ]
                    )
                X_lpp.append(loc)
                X_h18.append(h18[vp])
                X_h18_cmvn.append(h18_cmvn[i])
                X_h18_mean5.append(
                    h18[np.clip(vp + np.array([-4, -2, 0, 2, 4]), 0, T - 1)].mean(0)
                )
                X_h18_w3.append(
                    np.concatenate(
                        [h18[max(0, vp - 1)], h18[vp], h18[min(T - 1, vp + 1)]]
                    )
                )
                X_h18_w5.append(
                    np.concatenate(
                        [
                            h18[max(0, vp - 2)],
                            h18[max(0, vp - 1)],
                            h18[vp],
                            h18[min(T - 1, vp + 1)],
                            h18[min(T - 1, vp + 2)],
                        ]
                    )
                )
        return {
            "y": np.asarray(Y, np.int32),
            "scal": np.asarray(X_scal, np.float32),
            "ph": np.asarray(X_ph, np.float32),
            "lpp": np.asarray(X_lpp, np.float32),
            "h18": np.asarray(X_h18, np.float32),
            "h18_cmvn": np.asarray(X_h18_cmvn, np.float32),
            "h18_mean5": np.asarray(X_h18_mean5, np.float32),
            "h18_w3": np.asarray(X_h18_w3, np.float32),
            "h18_w5": np.asarray(X_h18_w5, np.float32),
        }

    banks = {
        "fit_l2a": build_bank([r for r in fit if domain(r) == "l2arctic"]),
        "fit_epa": build_bank([r for r in fit if domain(r) == "epadb"]),
        "fit_tim": build_bank([r for r in fit if domain(r) == "l1"], max_rows=500),
        "held6": build_bank([r for r in ev_p if r["cohort"] == "held6"]),
        "epadb_test": build_bank([r for r in ev_p if r["cohort"] == "epadb_test"]),
        "saa27": build_bank([r for r in ev_p if r["cohort"] == "saa27"]),
        "spontaneous": build_bank([r for r in ev_p if r["cohort"] == "spontaneous"]),
    }

    def ridge(train_names, keys, alpha):
        Xtr = np.concatenate(
            [
                np.concatenate([banks[tn][k] for k in keys], axis=1)
                for tn in train_names
            ],
            axis=0,
        )
        ytr = np.concatenate([banks[tn]["y"] for tn in train_names], axis=0)
        mu = Xtr.mean(0)
        sd = np.maximum(Xtr.std(0), 1e-5)
        Xtr = (Xtr - mu) / sd
        d = Xtr.shape[1]
        w = np.linalg.solve(
            Xtr.T @ Xtr + alpha * np.eye(d, dtype=np.float32),
            Xtr.T @ ytr.astype(np.float32),
        )
        return {
            en: float(
                average_precision_score(
                    banks[en]["y"],
                    (
                        (
                            np.concatenate([banks[en][k] for k in keys], axis=1)
                            - mu
                        )
                        / sd
                    )
                    @ w,
                )
            )
            for en in ("held6", "epadb_test", "saa27", "spontaneous")
        }

    feature_probes = {
        "scal_ph": ridge(["fit_l2a", "fit_epa"], ["scal", "ph"], 100.0),
        "scal_ph_lpp": ridge(["fit_l2a", "fit_epa"], ["scal", "ph", "lpp"], 100.0),
        "only_h18_no_cmvn": ridge(["fit_l2a", "fit_epa"], ["h18", "ph"], 5000.0),
        "only_h18_with_cmvn": ridge(["fit_l2a", "fit_epa"], ["h18_cmvn", "ph"], 5000.0),
        "only_h18_mean5_exp1": ridge(["fit_l2a", "fit_epa"], ["h18_mean5", "ph"], 5000.0),
        "only_h18_w3_concat": ridge(["fit_l2a", "fit_epa"], ["h18_w3", "ph"], 15000.0),
        "scal_ph_lpp_h18_w5": ridge(
            ["fit_l2a", "fit_epa"], ["scal", "ph", "lpp", "h18_w5"], 25000.0
        ),
    }
    domain_probes = {
        "only_fit_l2a": ridge(["fit_l2a"], ["scal", "ph", "lpp", "h18_w5"], 25000.0),
        "only_fit_epa": ridge(["fit_epa"], ["scal", "ph", "lpp", "h18_w5"], 25000.0),
        "fit_l2a_plus_epa": ridge(
            ["fit_l2a", "fit_epa"], ["scal", "ph", "lpp", "h18_w5"], 25000.0
        ),
        "fit_l2a_plus_epa_plus_timit500": ridge(
            ["fit_l2a", "fit_epa", "fit_tim"],
            ["scal", "ph", "lpp", "h18_w5"],
            25000.0,
        ),
    }
    return {"feature_probes": feature_probes, "domain_probes": domain_probes}


def probe_r0_vs_fixed_and_regimes(cache, repair, ev_p, ev_g_rows):
    pred_fix_dir = repair / "matched_seed20261121_e14_b16/eval_fixed/eval_predictions"
    ev_g = {(r["cohort"], r["uid"]): r for r in ev_g_rows}

    # R0 vs Fixed on Held-6
    r0 = pq.read_table(OLD / "held6_slots.parquet").to_pandas()
    r0_t = r0[r0.domain == "target"][
        ["uid", "index", "speaker", "gold_label", "vanilla_call", "r0_score"]
    ]
    fix_t = pq.read_table(pred_fix_dir / "held6.parquet").to_pandas()
    fix_t = fix_t[fix_t.domain == "target"][["uid", "index", "score"]].rename(
        columns={"score": "fix_score"}
    )
    m = r0_t.merge(fix_t, on=["uid", "index"])
    y_bin = m.gold_label.isin(["S", "O"]).astype(int)
    r0_vs_fixed = {
        "overall_ap": {
            "r0": float(average_precision_score(y_bin, m.r0_score)),
            "fixed": float(average_precision_score(y_bin, m.fix_score)),
        },
        "by_speaker": {
            spk: {
                "r0": float(
                    average_precision_score(g.gold_label.isin(["S", "O"]), g.r0_score)
                ),
                "fixed": float(
                    average_precision_score(g.gold_label.isin(["S", "O"]), g.fix_score)
                ),
            }
            for spk, g in m.groupby("speaker")
        },
        "by_vanilla_call": {
            vc: {
                "n": int(len(g)),
                "errors": int(g.gold_label.isin(["S", "O"]).sum()),
                "r0": float(
                    average_precision_score(g.gold_label.isin(["S", "O"]), g.r0_score)
                ),
                "fixed": float(
                    average_precision_score(g.gold_label.isin(["S", "O"]), g.fix_score)
                ),
            }
            for vc, g in m.groupby("vanilla_call")
        },
    }

    # S_same keys on Held-6
    s_same = set()
    for r in read_jsonl(HELD6_GOLD):
        can = [normalize_canonical_phone(p) for p in r["canonical_phones"]]
        real_raw = [
            normalize_canonical_phone(p)
            if p not in {"", "SIL", "sp", "spn", "sil"}
            else ""
            for p in r["realized_phones"]
        ]
        _, _, rids = dp_align_csoi_realized(can, [p for p in real_raw if p])
        for i, y in enumerate(r["gold_targets"]):
            if y == "S" and rids[i] == PHONE_TO_ID[can[i]]:
                s_same.add((r["uid"], i))

    regimes = {}
    for cname in ("held6", "epadb_test", "saa27", "timit_core"):
        rows = [r for r in ev_p if r["cohort"] == cname]
        df = pq.read_table(pred_fix_dir / f"{cname}.parquet").to_pandas()
        df = df[df.domain == "target"]
        meta = []
        for r in rows:
            g = ev_g[(cname, r["uid"])]["gold_targets"]
            h, lp = read_frames(cache, r)
            clp = pool_features(h, lp, r["canonical_phones"], Spec(), None)[
                "target_scalars"
            ][:, 7]
            for i, y in enumerate(g):
                meta.append((r["uid"], i, y, float(clp[i]), (r["uid"], i) in s_same))
        j = df.merge(
            pd.DataFrame(meta, columns=["uid", "index", "gold", "can_lp", "is_s_same"]),
            on=["uid", "index"],
        )
        j = j[j.gold != "unknown"]

        def bucket(sub):
            return {
                "n": int(len(sub)),
                "koel_says_c_gt_neg02": int((sub.can_lp > -0.2).sum()),
                "borderline_neg2_to_neg02": int(
                    ((sub.can_lp <= -0.2) & (sub.can_lp >= -2.0)).sum()
                ),
                "koel_says_err_lt_neg2": int((sub.can_lp < -2.0).sum()),
            }

        if cname == "timit_core":
            regimes[cname] = {"target_fa": bucket(j[j.call.isin(["S", "O"])])}
        else:
            fn = j[j.gold.isin(["S", "O"]) & (j.call == "C")]
            fp = j[(j.gold == "C") & j.call.isin(["S", "O"])]
            rec = {"fn": bucket(fn), "fp": bucket(fp)}
            if cname == "held6":
                rec["fn"]["s_same_count"] = int(fn.is_s_same.sum())
            regimes[cname] = rec

    return r0_vs_fixed, regimes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--repair", type=Path, default=DEFAULT_REPAIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    t0 = time.perf_counter()

    cache = a.run / "frames"
    splits = a.repair / "splits"
    fit = admitted(splits, "fit.jsonl")
    cal = admitted(splits, "cal.jsonl")
    ev_g_rows = admitted(splits, "eval_gold.jsonl")
    ev_p = _attach_eval_gold(admitted(splits, "eval_prompts.jsonl"), ev_g_rows)

    d_fix = torch.load(
        a.repair / "matched_seed20261121_e14_b16/fixed/control.pt",
        map_location="cpu",
        weights_only=True,
    )
    m_fix = LengthAwareV2().eval()
    m_fix.load_state_dict(d_fix["state_dict"])
    norm_fix = {k: tuple(t.numpy() for t in v) for k, v in d_fix["norm"].items()}

    groups = {
        "fit_l2arctic": [r for r in fit if domain(r) == "l2arctic"],
        "cal_l2arctic": [r for r in cal if domain(r) == "l2arctic"],
        "fit_epadb": [r for r in fit if domain(r) == "epadb"],
        "cal_epadb": [r for r in cal if domain(r) == "epadb"],
        "fit_timit_corr": [r for r in fit if domain(r) == "l1"],
        "eval_held6": [r for r in ev_p if r["cohort"] == "held6"],
        "eval_epadb_test": [r for r in ev_p if r["cohort"] == "epadb_test"],
        "eval_saa27": [r for r in ev_p if r["cohort"] == "saa27"],
        "eval_spontaneous": [r for r in ev_p if r["cohort"] == "spontaneous"],
    }

    layer_stats = probe_layer_stats(cache, groups["eval_held6"])
    zero_train = probe_zero_train_koel_ap(cache, groups)
    ko_cohorts = {
        "fit_l2a_300": groups["fit_l2arctic"][:300],
        "fit_epadb_300": groups["fit_epadb"][:300],
        "cal_l2a": groups["cal_l2arctic"],
        "cal_epadb": groups["cal_epadb"],
        "eval_held6": groups["eval_held6"],
        "eval_epadb": groups["eval_epadb_test"],
        "eval_saa27": groups["eval_saa27"],
        "eval_spontaneous": groups["eval_spontaneous"],
    }
    magnitudes, knockout = probe_model_internals_and_knockout(
        cache, m_fix, norm_fix, ko_cohorts
    )
    ridge = probe_ridge_information_ceiling(cache, fit, ev_p)
    r0_vs_fixed, regimes = probe_r0_vs_fixed_and_regimes(cache, a.repair, ev_p, ev_g_rows)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": "CPU read-only forensic diagnosis; no training, no threshold search, no historical edits, test_sealed untouched",
        "source_sha256": sha_file(__file__),
        "repair_preflight_sha256": sha_file(splits / "PREFLIGHT.json"),
        "fixed_checkpoint_sha256": sha_file(
            a.repair / "matched_seed20261121_e14_b16/fixed/control.pt"
        ),
        "v2_checkpoint_sha256": sha_file(V2_CKPT),
        "layer_stats_held6_150": layer_stats,
        "zero_train_koel_ap": zero_train,
        "fixed_internal_magnitudes": magnitudes,
        "fixed_component_knockout": knockout,
        "ridge_information_ceiling": ridge,
        "r0_vs_fixed_held6": r0_vs_fixed,
        "fixed_can_lp_regimes": regimes,
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    save_json(a.output, result)
    print(f"Saved {a.output} ({result['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
