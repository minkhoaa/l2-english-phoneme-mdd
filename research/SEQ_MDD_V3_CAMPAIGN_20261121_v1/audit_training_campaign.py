"""Read-only forensic recount of the completed V3 campaign (no threshold search or training)."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    PHONE_TO_ID as PHONE_ID,
)
from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.run_full_inference import (
    INTAKE_MANIFEST,
    IPA_TO_ARPA39,
)
from research.MDD_FIXES_AND_V2_20261121_v1.feature_extractor_v2 import (
    ctc_viterbi_align_allophone_aware,
)
from research.MDD_FIXES_AND_V2_20261121_v1.koel_ipa_map_v2 import ALLOPHONE_TOKEN_IDS
from research.MDD_FIXES_AND_V2_20261121_v1.train_and_eval_seq_mdd_v2 import (
    dp_align_csoi_realized,
)

from .data import (
    COHORTS,
    CONTROL,
    HERE,
    OLD,
    ROOT,
    SEED,
    admitted,
    frame_path,
    read_json,
    read_jsonl,
    save_json,
    sha_file,
)
from .features import OFFSETS, POINTS, deletion_feature, gap_indices, voronoi_bounds
from .prepare_l1_canonical_anchor import (
    CMU,
    cmu_phones,
    orthographic_prompts,
    prompt_id,
)
from .scoring import KEY, gold_frame, keyed_merge, read_predictions, score_mask
from .training import domain, sample_probs

ARMS = {
    "v2": "v2_keyed",
    "pool": "exp1_window/pool_only",
    "window": "exp1_window/full_window",
    "canon": "exp2_l1_canon",
    "gap": "exp3_gap_so/gap_core_focal",
    "ollr": "exp3_gap_so/ollr_realized",
    "joint": "exp3_gap_so/gapso_joint",
    "seed21": "exp4_multiseed/seed20261121",
    "seed22": "exp4_multiseed/seed20261122",
    "seed23": "exp4_multiseed/seed20261123",
    "lora": "exp5_joint_lora",
}
L2 = COHORTS[:-1]


def q(values, probs=(0, 0.25, 0.5, 0.75, 0.95, 1)):
    return np.quantile(values, probs).tolist() if len(values) else []


def tags(rows):
    return Counter(x for r in rows for x in r["gold_targets"]), Counter(
        x for r in rows for x in r["gold_gaps"]
    )


def dataset_profile(rows):
    by = {}
    for d in sorted({domain(r) for r in rows}):
        rs = [r for r in rows if domain(r) == d]
        t, g = tags(rs)
        aux = Counter()
        top_s = Counter()
        for r in rs:
            for p, y, z in zip(
                r["canonical_phones"], r["gold_targets"], r["realized_ids"], strict=True
            ):
                au = (
                    "masked"
                    if z == -100
                    else "del"
                    if z == 39
                    else "canonical"
                    if z == PHONE_ID[p]
                    else "other"
                )
                aux[y + "_" + au] += 1
                if y == "S" and au in ("del", "canonical"):
                    top_s[(p, au)] += 1
        rec = dict(
            utts=len(rs),
            speakers=len({r["speaker"] for r in rs}),
            target=t,
            gap=g,
            native_l1_flag=sum(r["is_native_l1"] for r in rs),
            realized_cross_counts=aux,
            top_same_or_del_for_s=[[a, b, c] for (a, b), c in top_s.most_common(10)],
            target_per_utt=float(np.mean([len(r["gold_targets"]) for r in rs])),
            gap_per_utt=float(np.mean([len(r["gold_gaps"]) for r in rs])),
        )
        if d == "l1":
            rec["prompt_id_counts"] = dict(
                Counter(prompt_id(r["uid"]) for r in rs).most_common(15)
            )
            rec["unique_prompt_ids"] = len({prompt_id(r["uid"]) for r in rs})
            rec["rows_without_any_edit"] = sum(
                set(r["gold_targets"]) == {"C"} and set(r["gold_gaps"]) == {"N"}
                for r in rs
            )
        by[d] = rec
    return by


def cold_history(path):
    card = read_json(path / "CHECKPOINT_CARD.json")
    h = card.get("training_history", card.get("history"))
    if not h:
        return {"policy": card["policy"]}
    best_ep = card["best_epoch"]
    sel = lambda r: r["cal_t_ap"] + 0.15 * r["cal_g_ap"]
    assert max(h, key=sel)["epoch"] == best_ep
    return {
        "best_epoch": best_ep,
        "epochs": len(h),
        "policy": card["policy"],
        "fit": card.get("fit"),
        "loss_first_last": [h[0]["loss"], h[-1]["loss"]],
        "cal_selected": h[best_ep - 1],
        "cal_epoch_last": h[-1],
        "argmax_separate_target_ap": max(h, key=lambda x: x["cal_t_ap"])["epoch"],
        "argmax_separate_gap_ap": max(h, key=lambda x: x["cal_g_ap"])["epoch"],
    }


def score_breakdown(frame):
    ret = {}
    for c, sub in frame.groupby("cohort"):
        if c == CONTROL:
            continue
        if c == "timit_core":
            pr = int(sub.call.isin(["S", "O", "I"]).sum())
            tg = int(sub[sub.domain == "target"].call.isin(["S", "O"]).sum())
            ret[c] = {"fa_total": pr, "fa_target": tg, "fa_gap": pr - tg}
            continue
        t = sub[sub.domain == "target"]
        g = sub[sub.domain == "gap"]
        x = score_mask(sub)
        tv = t.gold_label.isin(["S", "O"])
        alerted = t.call.isin(["S", "O"])
        type_o = t.p_o > t.p_s
        conf = Counter(zip(t.gold_label, t.call))
        gr = Counter(zip(g.gold_label, g.call))
        z = (
            t.loc[tv, "p_o"] / (t.loc[tv, "p_s"] + t.loc[tv, "p_o"]).clip(lower=1e-10)
        ).to_numpy()
        ret[c] = {
            k: x[k]
            for k in (
                "strict",
                "binary",
                "S",
                "O",
                "I",
                "target_ap",
                "gap_ap",
                "target_ece",
                "s_to_o",
                "o_to_s",
            )
        }
        ret[c].update(
            target_aux_composition={
                "tp": conf["S", "S"] + conf["O", "O"],
                "fp_c": conf["C", "S"] + conf["C", "O"],
                "swaps": conf["S", "O"] + conf["O", "S"],
                "missed_calls": conf["S", "C"] + conf["O", "C"],
            },
            true_so_called=int((tv & alerted).sum()),
            wrong_type_rate_given_flag_and_gold_so=float(
                (type_o[alerted & tv] != t.gold_label[alerted & tv].eq("O")).mean()
            ),
            conditional_auc_o_vs_s=float(
                roc_auc_score(t.loc[tv, "gold_label"] == "O", z)
            ),
            gap_counts={
                "gold_i": gr["I", "I"] + gr["I", "N"],
                "tp": gr["I", "I"],
                "fp": gr["N", "I"],
                "fn": gr["I", "N"],
            },
        )
    return ret


def compare_type_on_common(base, cand):
    joined = keyed_merge(
        base[KEY + ["call", "gold_label"]],
        cand[KEY + ["call"]].rename(columns={"call": "candidate"}),
    )
    result = {}
    for c, x in joined[
        joined.cohort.isin(L2)
        & joined.domain.eq("target")
        & joined.gold_label.isin(["S", "O"])
    ].groupby("cohort"):
        both = x.call.isin(["S", "O"]) & x.candidate.isin(["S", "O"])
        result[c] = {
            "both_alerted": int(both.sum()),
            "base_swaps": int((both & x.call.ne(x.gold_label)).sum()),
            "candidate_swaps": int((both & x.candidate.ne(x.gold_label)).sum()),
            "base_only_alerted": int(
                (x.call.isin(["S", "O"]) & x.candidate.eq("C")).sum()
            ),
            "candidate_only_alerted": int(
                (x.candidate.isin(["S", "O"]) & x.call.eq("C")).sum()
            ),
        }
    return result


def snapshot_swaps(frames):
    # Keep already-locked calls; exchange whole domains, never retune thresholds here.
    out = {}
    cases = [
        ("v2", "v2"),
        ("v2", "gap"),
        ("gap", "v2"),
        ("gap", "gap"),
        ("v2", "joint"),
        ("joint", "v2"),
        ("joint", "joint"),
    ]
    for a, b in cases:
        t = frames[a][frames[a].domain.eq("target")]
        g = frames[b][frames[b].domain.eq("gap")]
        out[a + "_target__" + b + "_gap"] = {}
        for c in L2:
            res = score_mask(pd.concat([t[t.cohort.eq(c)], g[g.cohort.eq(c)]]))
            out[a + "_target__" + b + "_gap"][c] = res["strict"]["f1"]
    return out


def sampling_and_batches(fit):
    n = len(fit)
    probs = sample_probs(fit)
    rng = np.random.default_rng(SEED)
    order = rng.choice(n, size=n, replace=True, p=probs)
    out = {}
    for label, ids in [("epoch1", order), ("uniform_no_replacement", np.arange(n))]:
        subjects = {}
        for d in sorted({domain(r) for r in fit}):
            sample = [i for i in ids if domain(fit[i]) == d]
            r = [fit[i] for i in sample]
            t, g = tags(r)
            subjects[d] = dict(
                draws=len(sample),
                unique=len(set(sample)),
                valid_targets=sum(t[k] for k in "CSO"),
                target_errors=t["S"] + t["O"],
                valid_gaps=g["N"] + g["I"],
                insertions=g["I"],
            )
        out[label] = {"unique_utterances": len(set(ids)), "by_domain": subjects}
    lengths = np.asarray([len(r["canonical_phones"]) for r in fit])[order]
    out["expected_unique_per_epoch"] = float(np.sum(-np.expm1(n * np.log1p(-probs))))
    out["eos_uses_padding_epoch1"] = sum(
        int(np.sum(x < x.max())) for x in np.array_split(lengths, np.arange(16, n, 16))
    )
    return out


def anchor_audit(run, fit, cal, anchor, prompts):
    from research.KOEL_ABC_FROZEN_EVAL_20261119_v1.build_manifests import load_cmudict

    source = read_json(run / "splits/PREFLIGHT.json")["source_sha256"][
        str(INTAKE_MANIFEST.relative_to(ROOT))
    ]
    assert sha_file(INTAKE_MANIFEST) == source
    raw = {
        r["uid"]: r
        for r in read_jsonl(INTAKE_MANIFEST)
        if r["corpus"] == "TIMIT" and r["split"] == "TRAIN"
    }
    words = orthographic_prompts(run / "l1_sources/LDC_CTIMIT_PROMPTS.TXT")
    cmu = load_cmudict(CMU)
    out = {}
    for label, rows in [
        ("fit_legacy", fit),
        ("cal_legacy", cal),
        ("fit_canon", [r for r in anchor if r["anchor_role"] == "fit"]),
        ("cal_canon", [r for r in anchor if r["anchor_role"] == "cal"]),
    ]:
        aa = []
        by = Counter()
        total = Counter()
        for r in rows:
            if domain(r) != "l1":
                continue
            p = prompt_id(r["uid"])
            p0 = p[:2]
            surf = [
                IPA_TO_ARPA39[unicodedata.normalize("NFC", t)]
                for t in raw[r["uid"]]["realized_ipa_tokens"]
            ]
            canonical, _ = cmu_phones(words[p], cmu)
            if canonical is None:
                continue
            t, g, ri = dp_align_csoi_realized(canonical, surf)
            by[p] += 1
            total["rows"] += 1
            total["rows_any_proxy_mismatch"] += int(
                "I" in g or any(x != "C" for x in t)
            )
            total["canonical_phones"] += len(canonical)
            total["surface_phones"] += len(surf)
            total["S"] += t.count("S")
            total["O"] += t.count("O")
            total["I_gaps"] += g.count("I")
            total[p0 + "_rows"] += 1
            for x, y, z in zip(canonical, t, ri, strict=True):
                if x == "AA":
                    aa.append((y, int(z)))
        out[label] = {
            "prompt_examples_top": by.most_common(5),
            "cmu_vs_surface_proxy_not_error_gold": total,
            "aa_targets": len(aa),
            "aa_surface_disagree": sum(y != "C" for y, id_ in aa),
            "aa_to_ao_in_proxy": sum(id_ == PHONE_ID["AO"] for _, id_ in aa),
        }
    trows = [r for r in prompts if r["cohort"] == "timit_core"]
    crows = [r for r in cal if domain(r) == "l1"]
    out["timit_core_vs_legacy_cal_prompt_intersection"] = sorted(
        {prompt_id(r["uid"]) for r in trows} & {prompt_id(r["uid"]) for r in crows}
    )
    return out


def align_probes(run, prompts, gold, frames):
    kernel = run / "ctc_delete.so"
    card = read_json(run / ARMS["joint"] / "CHECKPOINT_CARD.json")
    if sha_file(kernel) != card["kernel_sha256"]:
        raise ValueError("CTC kernel changed")
    ev = {}
    for row in prompts:
        name = row["cohort"]
        if name == CONTROL:
            continue
        # Hash-keyed cached posterior, no new audio or GPU forward.
        with np.load(frame_path(run / "frames", row), allow_pickle=False) as a:
            lp = a["lp"]
        can = row["canonical_phones"]
        n = len(can)
        T = len(lp)
        spans, used_fallback = ctc_viterbi_align_allophone_aware(lp, can)
        s, e, p = np.asarray(spans).T
        t = np.clip(p[:, None] + OFFSETS[POINTS], 0, T - 1)
        lo, hi = np.asarray(voronoi_bounds(p, T)).T
        off = ((t < lo[:, None]) | (t >= hi[:, None])).sum(1)
        alt = []
        for i, (_, _, b) in enumerate(spans):
            ids = list(ALLOPHONE_TOKEN_IDS[can[i]])
            alt.append(int(lp[b, ids].argmax() != 0))
        # Need gold only for descriptive label slices; no policy construction here.
        ll = deletion_feature(lp, can, kernel)
        targets = [
            (
                name,
                row["uid"],
                "target",
                k,
                int(e[k] - s[k] + 1),
                int(off[k]),
                int(hi[k] - lo[k]),
                alt[k],
                int(len(set(t[k]))),
                float(ll[k, 0]),
                bool(ll[k, 1]),
                len(ALLOPHONE_TOKEN_IDS[can[k]]) > 1,
            )
            for k in range(n)
        ]
        gaps = []
        for k in range(n + 1):
            low = 0 if k == 0 else int(e[k - 1] + 1)
            high = T - 1 if k == n else int(s[k] - 1)
            exclude = {0}
            if k:
                exclude.update(ALLOPHONE_TOKEN_IDS[can[k - 1]])
            if k < n:
                exclude.update(ALLOPHONE_TOKEN_IDS[can[k]])
            if not k:
                exclude.add(64)
            other = ~np.isin(np.arange(79), list(exclude))
            ii = gap_indices(low, high, 0, T, lp, other, True)
            seam = (int(p[k - 1]) + int(p[k])) / 2 if 0 < k < n else np.nan
            spread = int(abs(ii[1] - ii[0])) if len(ii) == 2 else 0
            distance = (
                float(np.abs(ii - seam).mean())
                if len(ii) and np.isfinite(seam)
                else np.nan
            )
            maxscore = float(lp[ii][:, other].max()) if len(ii) else -25.0
            gaps.append(
                (
                    name,
                    row["uid"],
                    "gap",
                    k,
                    max(0, high - low + 1),
                    len(ii),
                    spread,
                    distance,
                    maxscore,
                )
            )
        v = ev.setdefault(name, [[], [], 0])
        v[0].extend(targets)
        v[1].extend(gaps)
        v[2] += int(used_fallback)
    output = {}
    x = frames["v2"][KEY + ["p_s", "p_o", "call"]]
    for c, (tt, gg, fb) in ev.items():
        t = pd.DataFrame(
            tt,
            columns=KEY
            + [
                "span",
                "outside",
                "voronoi",
                "alt",
                "unique_pts",
                "llr",
                "valid",
                "has_alt",
            ],
        )
        g = pd.DataFrame(
            gg, columns=KEY + ["span", "selected", "spread", "seam_dist", "log_other"]
        )
        t = t.merge(gold, on=KEY, how="left", validate="one_to_one").merge(
            x, on=KEY, validate="one_to_one"
        )
        g = g.merge(gold, on=KEY, how="left", validate="one_to_one")
        get = lambda t: {
            label: {
                "n": len(y),
                "span_one_frac": float(y["span"].eq(1).mean()),
                "window_pts_outside_voronoi_frac": float(y.outside.mean() / 5.0),
                "center_alt_preferred_frac": float(y.alt.mean()),
                "unique_five_pts_frac": float(y.unique_pts.eq(5).mean()),
                "voronoi_q": q(y.voronoi),
                "ollr_q": q(y.llr),
                "ollr_valid_frac": float(y.valid.mean()),
                "ollr_abs_clipped": int((y.llr.abs() >= 80).sum()),
            }
            for label, y in t.groupby("gold_label")
        }
        k = t[t.gold_label.isin(["S", "O"]) & t.valid]
        yy = k.gold_label.eq("O")
        sep = {}
        if yy.nunique() == 2:
            sep = {
                "n": len(k),
                "prevalence_o": float(yy.mean()),
                "auc_ollr": float(roc_auc_score(yy, k.llr)),
                "ap_ollr": float(average_precision_score(yy, k.llr)),
                "auc_v2_o_ratio": float(
                    roc_auc_score(yy, k.p_o / (k.p_s + k.p_o).clip(lower=1e-10))
                ),
            }
            m = t[t.gold_label.isin(list("CSO")) & t.valid]
            sep["ap_ollr_o_vs_all"] = float(
                average_precision_score(m.gold_label.eq("O"), m.llr)
            )
            sep["ap_v2_o_vs_all"] = float(
                average_precision_score(m.gold_label.eq("O"), m.p_o)
            )
        gi = {
            l: {
                "n": len(y),
                "span_q": q(y["span"]),
                "empty_frac": float(y.selected.eq(0).mean()),
                "top2_spread_q": q(y[y.selected.eq(2)].spread),
                "seam_dist_q": q(y.seam_dist.dropna()),
                "candidate_logprob_q": q(y.log_other),
            }
            for l, y in g.groupby("gold_label")
        }
        output[c] = {
            "fallback_utterances": fb,
            "by_target_label": get(t),
            "so_separation_ollr": sep,
            "raw_gap_by_label": gi,
        }
    return output


def native_overlap(frames):
    p = OLD / "timit_core_slots.parquet"
    assert (
        sha_file(p)
        == read_json(OLD / "INFERENCE_RECEIPT.json")["files"][p.name]["sha256"]
    )
    alt = pq.read_table(p, columns=KEY + ["gold_label_alt"]).to_pandas()
    out = {}
    for name in ("v2", "canon", "gap", "joint", "seed21", "lora"):
        base = frames[name]
        t = keyed_merge(base[base.cohort.eq("timit_core")], alt)
        d = t[t.domain.eq("target")].copy()
        d["alert"] = d.call.isin(["S", "O"])
        d["surface_diff"] = d.gold_label_alt.isin(["S", "O"])
        out[name] = {}
        for ph in ("all", "AA", "AH", "T", "D"):
            sub = d if ph == "all" else d[d.canonical_phone.eq(ph)]
            out[name][ph] = {
                str(flag): {"n": len(x), "fa": int(x.alert.sum())}
                for flag, x in sub.groupby("surface_diff")
            }
    return out


def per_shift(run):
    from research.KOEL_FULL_CONFIG_BENCHMARK_20261120_v1.run_per_and_latency import (
        EPADB_TEST_MANIFEST,
        TIMIT_CORE_MANIFEST,
        timit_recount,
    )

    root = run / ARMS["lora"]
    p = root / "CTC_PREDICTIONS.json"
    assert sha_file(p) == read_json(root / "CTC_SEAL.json")["sha256"]
    hyp = read_json(p)
    mapping = timit_recount.parse_map()
    out = {}
    for c, mp, ref in [
        ("timit_core", TIMIT_CORE_MANIFEST, "surface39_ref"),
        ("epadb_test", EPADB_TEST_MANIFEST, "realized_ipa_tokens"),
    ]:
        rows = read_jsonl(mp)
        meta = Counter()
        per_sp = {}
        delta_tokens = Counter()
        for r in rows:
            uid = r["uid"]
            sp = str(r.get("speaker", r.get("speaker_id")))
            before, after = hyp["base"][uid], hyp["lora"][uid]
            if before != after:
                meta["decoder_changed_utterances"] += 1
            delta_tokens.subtract(before)
            delta_tokens.update(after)
            if c == "timit_core":
                before = timit_recount.project_hypothesis(before, mapping, Counter())
                after = timit_recount.project_hypothesis(after, mapping, Counter())
            # Match the artifact's native79 reference field, not the MDD gold labels.
            gold = r[ref]
            aa = timit_recount.distance(gold, before)
            bb = timit_recount.distance(gold, after)
            v = per_sp.setdefault(sp, [0, 0, 0])
            v[0] += aa
            v[1] += bb
            v[2] += len(gold)
            meta["base_edits"] += aa
            meta["lora_edits"] += bb
            meta["ref_phones"] += len(gold)
            meta["worse" if bb > aa else "better" if bb < aa else "same"] += 1
        key = "timit_surface39" if c == "timit_core" else "epadb_native79"
        evidence = read_json(root / "PER_GUARD.json")
        assert (
            meta["base_edits"] == evidence["base"][key]["edits"]
            and meta["lora_edits"] == evidence["lora"][key]["edits"]
        ), (c, meta)
        out[c] = {
            "counts": meta,
            "speakers_worse": sum(a < b for a, b, n in per_sp.values()),
            "speakers_better": sum(a > b for a, b, n in per_sp.values()),
            "n_speakers": len(per_sp),
            "net_decoded_token_frequency_delta": sorted(
                delta_tokens.items(), key=lambda x: -abs(x[1])
            )[:12],
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HERE / "runs/campaign_d5f8e89")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run = a.run_dir.resolve()
    split = run / "splits"
    fit, cal, prompts, gold_rows = [
        admitted(split, fn + ".jsonl")
        for fn in ("fit", "cal", "eval_prompts", "eval_gold")
    ]
    gold = gold_frame(gold_rows)
    frames = {k: keyed_merge(read_predictions(run / d), gold) for k, d in ARMS.items()}
    before_temp = {}
    for s in ("seed21", "seed22", "seed23"):
        b = keyed_merge(read_predictions(run / ARMS[s] / "before_temperature"), gold)
        before_temp[s] = score_breakdown(b)
    anchor = read_jsonl(run / "manifests/l1_canonical_anchor.jsonl")
    assert (
        sha_file(run / "manifests/l1_canonical_anchor.jsonl")
        == read_json(run / "manifests/l1_canonical_anchor_audit.json")[
            "manifest_sha256"
        ]
    )
    fit4 = read_jsonl(run / "exp4_multiseed/final_fit.jsonl")
    cal4 = read_jsonl(run / "exp4_multiseed/final_cal.jsonl")
    ret = {
        "protocol": "CPU, immutable frames and sealed locked calls only; no threshold optimization",
        "source_sha256": sha_file(__file__),
        "campaign_sha256": sha_file(run / "CAMPAIGN_STATUS.json"),
        "versions": {
            p: version(p)
            for p in ("numpy", "torch", "transformers", "scikit-learn", "pyarrow")
        },
        "sets": {
            k: dataset_profile(v)
            for k, v in [
                ("fit", fit),
                ("cal", cal),
                ("exp4_fit", fit4),
                ("exp4_cal", cal4),
                ("anchor", anchor),
            ]
        },
        "training_curves": {k: cold_history(run / d) for k, d in ARMS.items()},
        "sample_exp4": sampling_and_batches(fit4),
        "locked": {k: score_breakdown(v) for k, v in frames.items()},
        "before_temperature_locked": before_temp,
        "locked_prediction_crossovers": snapshot_swaps(frames),
        "type_common": {
            k: compare_type_on_common(frames["v2"], frames[k])
            for k in ("gap", "ollr", "joint")
        },
        "native_surface_overlap": native_overlap(frames),
        "anchors": anchor_audit(run, fit, cal, anchor, prompts),
        "per_lora": per_shift(run),
    }
    print("SEAL, DATA, COUNTS, CTC PER OK", flush=True)
    ret["geometry"] = align_probes(run, prompts, gold, frames)
    save_json(a.output, ret)
    print(a.output)


if __name__ == "__main__":
    main()
