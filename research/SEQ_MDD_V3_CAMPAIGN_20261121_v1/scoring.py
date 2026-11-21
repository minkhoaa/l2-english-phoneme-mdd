"""Seal label-free predictions; join gold and comparisons by exact slot identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from research.FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1.analyze_and_dissect import (
    paired_speaker_bootstrap,
    score_decisions,
)
from .data import COHORTS, CONTROL, OLD, ROOT, admitted, read_json, save_json, sha_file

KEY = ["cohort", "uid", "domain", "index"]


def scores_to_rows(row, target, gap, policy):
    tc, gc = policy["tau_t"], policy["tau_g"]
    s, o = target[:, 1], target[:, 2]
    g = gap[:, 1]
    base = {"cohort": row["cohort"], "uid": row["uid"], "speaker": row["speaker"]}
    for i, (ph, si, oi) in enumerate(zip(row["canonical_phones"], s, o, strict=True)):
        sc = si + oi
        yield dict(
            base,
            domain="target",
            index=i,
            canonical_phone=ph,
            score=float(sc),
            call=("O" if oi > si else "S") if sc >= tc else "C",
            p_s=float(si),
            p_o=float(oi),
        )
    if len(g) != len(s) + 1:
        raise ValueError(f"Unexpected gap count on {row['uid']}")
    for i, p in enumerate(g):
        yield dict(
            base,
            domain="gap",
            index=i,
            canonical_phone=None,
            score=float(p),
            call="I" if p >= gc else "N",
            p_s=None,
            p_o=None,
        )


def seal_predictions(out, records, frozen_sha, *, name="eval_predictions"):
    pred_dir = out / name
    pred_dir.mkdir()
    grouped = {}
    for r in records:
        if "gold_label" in r or "gold" in r:
            raise ValueError("Gold must not enter prediction artifact")
        grouped.setdefault(r["cohort"], []).append(r)
    files = {}
    for cohort, rows in grouped.items():
        path = pred_dir / f"{cohort}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        files[path.name] = {"sha256": sha_file(path), "rows": len(rows)}
    seal = {
        "checkpoint_card_sha256": frozen_sha,
        "key": KEY,
        "directory": name,
        "files": files,
    }
    save_json(out / "PREDICTIONS_SEAL.json", seal)
    return seal


def read_predictions(out):
    seal = read_json(out / "PREDICTIONS_SEAL.json")
    card = out / "CHECKPOINT_CARD.json"
    if card.exists() and sha_file(card) != seal["checkpoint_card_sha256"]:
        raise ValueError("Seal/checkpoint-card mismatch")
    dfs = []
    for p, rec in seal["files"].items():
        path = out / seal["directory"] / p
        if sha_file(path) != rec["sha256"]:
            raise ValueError(f"Predictions changed: {path}")
        f = pq.read_table(path).to_pandas()
        if len(f) != rec["rows"]:
            raise ValueError("Sealed row count differs")
        dfs.append(f)
    df = pd.concat(dfs, ignore_index=True)
    if df.duplicated(KEY).any() or not np.isfinite(df.score).all():
        raise ValueError("Duplicate slot key or non-finite score")
    return df


def gold_frame(rows):
    vals = []
    for r in rows:
        for typ, col in [("target", "gold_targets"), ("gap", "gold_gaps")]:
            vals.extend(
                (r["cohort"], r["uid"], typ, i, g) for i, g in enumerate(r[col])
            )
    return pd.DataFrame(vals, columns=KEY + ["gold_label"])


def keyed_merge(a, b):
    df = a.merge(b, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not df["_merge"].eq("both").all():
        raise ValueError(f"Key mismatch: {df['_merge'].value_counts().to_dict()}")
    return df.drop(columns="_merge")


def ece(y, score, bins=10):
    if not len(y):
        return None
    b = np.clip(np.int32(score * bins), 0, bins - 1)
    count = np.bincount(b, minlength=bins)
    sum_y = np.bincount(b, weights=y, minlength=bins)
    sum_s = np.bincount(b, weights=score, minlength=bins)
    return float(np.sum(np.abs(sum_y - sum_s)) / count.sum())


def ap_auc(y, sc):
    if not len(y) or not 0 < y.sum() < len(y):
        return None, None
    return float(average_precision_score(y, sc)), float(roc_auc_score(y, sc))


def score_mask(frame):
    t = frame[frame.domain == "target"]
    g = frame[frame.domain == "gap"]
    details = score_decisions(
        t.gold_label.to_numpy(),
        t.call.to_numpy(),
        g.gold_label.to_numpy(),
        g.call.to_numpy(),
    )
    details.update(
        n_utts=int(frame.uid.nunique()),
        n_speakers=int(frame.speaker.nunique()),
        strict_fp_per_utt=details["strict"]["fp"] / frame.uid.nunique(),
    )
    for a, label in [(t, "target"), (g, "gap")]:
        a = a[a.gold_label != "unknown"]
        y = a.gold_label.isin(["S", "O", "I"]).to_numpy()
        p = a.score.to_numpy()
        ap, auc = ap_auc(y, p)
        details[f"{label}_ap"] = ap
        details[f"{label}_roc_auc"] = auc
        details[f"{label}_ece"] = ece(y, p)
    return details


def split_calibrated_summary(df):
    ev = {}
    for name, group in df.groupby("cohort", sort=False):
        t = group[group.domain == "target"]
        g = group[group.domain == "gap"]
        if name == "timit_core":
            ev[name] = score_mask(group)
            alarms = t.call.isin(["S", "O"])
            a = int(alarms.sum())
            b = int(g.call.eq("I").sum())
            ev[name].update(
                fa_total=a + b,
                fa_target=a,
                fa_gap=b,
                fa_per_utt=(a + b) / group.uid.nunique(),
                target_fa_per_1k=a * 1000 / len(t),
                gap_fa_per_1k=b * 1000 / len(g),
                aa_fa=int((alarms & t.canonical_phone.eq("AA")).sum()),
                by_phone_fa={
                    str(k): int(v)
                    for k, v in t[alarms].canonical_phone.value_counts().items()
                },
            )
        elif name == CONTROL:
            y = t.gold_label.eq("S").to_numpy()
            sc = t.score.to_numpy()
            flags = t.call.isin(["S", "O"]).to_numpy()
            ap, auc = ap_auc(y, sc)
            ev[name] = {
                "n_utts": int(group.uid.nunique()),
                "n_corrupt": int(y.sum()),
                "n_unchanged": int((~y).sum()),
                "detected": int((flags & y).sum()),
                "recall": float(flags[y].mean()),
                "unchanged_alert_rate": float(flags[~y].mean()),
                "ap": ap,
                "roc_auc": auc,
            }
        else:
            ev[name] = score_mask(group)
            if name == "spontaneous":
                ev[name]["held6_only"] = score_mask(
                    group[
                        group.speaker.isin({"ABA", "ASI", "BWC", "EBVS", "HQTV", "LXC"})
                    ]
                )
            if name == "held6":
                ev[name]["six_phones"] = {}
                for ph in ("AH", "D", "R", "IH", "T", "OW"):
                    m = t.canonical_phone.eq(ph)
                    res = score_decisions(
                        t.loc[m, "gold_label"].to_numpy(),
                        t.loc[m, "call"].to_numpy(),
                        np.array([], dtype=str),
                        np.array([], dtype=str),
                    )
                    ev[name]["six_phones"][ph] = {
                        k: res[k]
                        for k in ("S", "O", "binary", "strict", "s_to_o", "o_to_s")
                    }
                ev[name]["by_speaker"] = {
                    speaker: {
                        k: s
                        for k, s in score_mask(sub).items()
                        if k in ("strict", "binary")
                    }
                    for speaker, sub in group.groupby("speaker")
                }
    return ev


def native_boot(frame, seed=20261121, n=5000):
    def rate(col):
        return (
            frame[col]
            .isin(["S", "O", "I"])
            .astype(int)
            .groupby(frame.speaker)
            .sum()
            .sort_index()
            .to_numpy()
        )

    a, b = rate("call"), rate("baseline")
    n_u = frame.groupby("speaker").uid.nunique().sort_index().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(a), size=(n, len(a)))
    d = (a - b)[idx].sum(1) / n_u[idx].sum(1)
    return {
        "delta_fa_per_utt": float((a - b).sum() / n_u.sum()),
        "ci95": np.quantile(d, [0.025, 0.975]).tolist(),
    }


def bootstrap(frame, baseline):
    merged = keyed_merge(
        frame[frame.cohort.isin(baseline.cohort.unique())],
        baseline[KEY + ["call"]].rename(columns={"call": "baseline"}),
    )
    out = {}
    for name, g in merged.groupby("cohort", sort=False):
        if name == CONTROL:
            continue
        if name == "timit_core":
            out[name] = native_boot(g)
            continue
        groups = [("", g)]
        if name == "spontaneous":
            groups.append(
                (
                    "_held6_only",
                    g[g.speaker.isin({"ABA", "ASI", "BWC", "EBVS", "HQTV", "LXC"})],
                )
            )
        for suffix, x in groups:
            t = x[x.domain == "target"]
            s = x[x.domain == "gap"]
            out[name + suffix] = paired_speaker_bootstrap(
                t.speaker.to_numpy(),
                t.gold_label.to_numpy(),
                t.call.to_numpy(),
                t.baseline.to_numpy(),
                s.speaker.to_numpy(),
                s.gold_label.to_numpy(),
                s.call.to_numpy(),
                s.baseline.to_numpy(),
            )
    return out


def load_old_r0():
    receipt = read_json(OLD / "INFERENCE_RECEIPT.json")["files"]
    data = []
    for name in (c for c in COHORTS if c != "saa27"):
        path = OLD / f"{name}_slots.parquet"
        if sha_file(path) != receipt[path.name]["sha256"]:
            raise RuntimeError("R0 archived Parquet hash mismatch")
        f = pq.read_table(
            path, columns=KEY + ["vanilla_call", "r0_p_s", "r0_p_o", "r0_score"]
        ).to_pandas()
        f["call"] = np.where(
            f.domain.eq("gap"),
            f.vanilla_call,
            np.where(
                f.r0_score >= 0.340746, np.where(f.r0_p_o > f.r0_p_s, "O", "S"), "C"
            ),
        )
        data.append(f[KEY + ["call"]])
    saa = (
        ROOT
        / "research/KOEL_SM13_BENCHMARK_20261120_v1/runs/saa27_koel_vs_sm13_20261120"
    )
    path = saa / "saa27_all_systems_predictions.parquet"
    if (
        sha_file(path)
        != read_json(saa / "saa27_predictions_seal.json")["predictions_parquet_sha256"]
    ):
        raise RuntimeError("R0 archived SAA hash mismatch")
    f = pq.read_table(path, filters=[("variant", "=", "r0_base_cal_f1")]).to_pandas()
    f["cohort"] = "saa27"
    f["call"] = f.pred_call  # SAA-18 calibrated R0, the stronger reference in GOAL.md.
    data.append(f[KEY + ["call"]])
    return pd.concat(data, ignore_index=True)


def load_old_dev_r0():
    path = (
        ROOT
        / "research/KOEL_SM13_BENCHMARK_20261120_v1/runs/dev18_koel_vs_sm13_20261120"
    )
    p = path / "eval_predictions.parquet"
    if sha_file(p) != read_json(path / "PREDICTIONS_SEAL.json")["sha256"][p.name]:
        raise RuntimeError("R0 archived DEV-18 OOF hash mismatch")
    f = pq.read_table(p).to_pandas().rename(columns={"r0_cal_f1": "call"})
    f["cohort"] = "dev18_oof"
    return f[KEY + ["call"]]


def evaluate_seal(out, split_dir, comparisons=None, gold_rows=None):
    pred = read_predictions(out)
    if gold_rows is None:
        gold_rows = admitted(split_dir, "eval_gold.jsonl")
    matched = keyed_merge(pred, gold_frame(gold_rows))
    evals = split_calibrated_summary(matched)
    comparison = {
        label: bootstrap(matched, data) for label, data in (comparisons or {}).items()
    }
    return evals, comparison


def brief(score):
    return {
        name: (
            round(e["fa_per_utt"], 4)
            if name == "timit_core"
            else round(e["recall"] * 100, 4)
            if name == CONTROL
            else round(e["strict"]["f1"] * 100, 4)
        )
        for name, e in score.items()
    }
