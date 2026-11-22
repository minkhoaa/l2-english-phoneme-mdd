"""Allowlisted Fit/Cal supervision & frame geometry; never load any Eval gold."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import soundfile as sf

from research.BACKBONE_REP_CAMPAIGN_20261122_v1.articulatory import AF_MATRIX
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.backbones import (
    _slot_path,
    koel_voronoi_spans,
    SM13_FORMAL,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import (
    BASE_KOEL_CACHE,
    PROPOSED_CLEAN_SPLIT,
    PROTECTED_SEALED_SPEAKERS,
    ROOT,
)
from research.CONTROLLED_PILOT_AB_20261118_v1.run_pilot_ab import WEIGHTS_DIR
from research.KOEL_BASE_MDD_HEAD_DEV18_20261119_v1.base_feature_extractor import (
    PHONE_TO_ID,
    normalize_canonical_phone,
)
from research.MDD_FIXES_AND_V2_20261121_v1.seq_mdd_v2 import CONTINUOUS_SCALAR_NAMES
from research.SEQ_MDD_V3_CAMPAIGN_20261121_v1.data import read_json, sha_file

RUN = ROOT / "research/BACKBONE_REP_CAMPAIGN_20261122_v1/runs/full_clean11_v1"
SOURCES = ROOT.parent / "New-Method-Research"
RAW = SOURCES / "data/derived/l2arctic/manifests"
TAG = {"CORRECT": "C", "SUBSTITUTION": "S", "OMISSION": "O"}


def roles_and_refs():
    bind = read_json(RUN / "RUN_BINDINGS.json")
    pre = bind["preflight"]
    assert bind["mode"] == "full" and pre["split_mode"] == "clean_11"
    for name, digest in bind["source_sha256"].items():
        assert sha_file(ROOT / name) == digest, name
    roles = {}
    for role in ("fit", "cal"):
        path = PROPOSED_CLEAN_SPLIT / f"{role}.jsonl"
        assert sha_file(path) == pre[role + "_sha256"]
        roles[role] = [json.loads(line) for line in path.open()]
        assert len(roles[role]) == pre["counts"][role]
        assert (
            not {r["speaker"].upper() for r in roles[role]} & PROTECTED_SEALED_SPEAKERS
        )
    admitted = {
        r["uid"]: r
        for rows in roles.values()
        for r in rows
        if r["source"] == "l2arctic_dev18"
    }
    refs, digests = {}, {}
    for filename in ("train", "dev_search", "dev_cal", "dev_replication"):
        # Select by routing header *before* parsing any annotations. Only accepted rows are hashed/stored.
        with (RAW / (filename + ".jsonl")).open() as f:
            for line in f:
                m = re.search(
                    r'"utterance_id"\s*:\s*"([A-Z]+_arctic_[ab]\d+)"',
                    line.split('"target_phones"', 1)[0],
                )
                if not m or m[1] not in admitted:
                    continue
                uid = m[1]
                r = admitted[uid]
                z = json.loads(line)
                assert uid not in refs and z["speaker_id"] == r["speaker"]
                assert (SOURCES / z["audio_path"]).resolve() == Path(
                    r["audio_path"]
                ).resolve()
                assert sha_file(r["audio_path"]) == r["audio_sha256"]
                assert [
                    (
                        x["slot_idx"],
                        normalize_canonical_phone(x["target_phone"]),
                        TAG[x["label"]],
                    )
                    for x in z["target_phones"]
                ] == list(
                    zip(
                        range(len(r["canonical_phones"])),
                        r["canonical_phones"],
                        r["gold_targets"],
                        strict=True,
                    )
                )
                assert [
                    (x["gap_idx"], bool(x["has_insertion"])) for x in z["gaps"]
                ] == list(enumerate(v == "I" for v in r["gold_gaps"]))
                refs[uid] = z
                digests[uid] = hashlib.sha256(line.encode()).hexdigest()
    assert refs.keys() == admitted.keys()
    return (
        roles,
        refs,
        hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
    )


def profile(rows, refs):
    out = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        c, g, extra = Counter(), Counter(), Counter()
        for r in sub:
            c.update(r["gold_targets"])
            g.update(r["gold_gaps"])
            for i, (ph, y, rid) in enumerate(
                zip(
                    r["canonical_phones"],
                    r["gold_targets"],
                    r["realized_ids"],
                    strict=True,
                )
            ):
                if y != "S":
                    continue
                ci = PHONE_TO_ID[ph]
                extra[
                    "aux_"
                    + (
                        "masked"
                        if rid < 0
                        else "del"
                        if rid == 39
                        else "canonical"
                        if rid == ci
                        else "other"
                    )
                ] += 1
                if 0 <= rid < 39 and np.array_equal(AF_MATRIX[ci], AF_MATRIX[rid]):
                    extra["aux_has_same_AF_as_canonical"] += 1
                if r["uid"] in refs:
                    ref = PHONE_TO_ID[
                        normalize_canonical_phone(
                            refs[r["uid"]]["target_phones"][i]["realized_phone"]
                        )
                    ]
                    extra["ref_" + ("same" if ref == ci else "different")] += 1
                    if ref != rid:
                        extra["aux_ref_mismatch"] += 1
        out[src] = dict(
            rows=len(sub),
            speakers=len({r["speaker"] for r in sub}),
            targets=dict(c),
            gaps=dict(g),
            S_detail=dict(extra),
            native_flags=sum(r["is_native_l1"] for r in sub),
        )
    return out


def grid(config):
    cfg = read_json(config)
    width, stride = 1, 1
    for k, s in zip(cfg["conv_kernel"], cfg["conv_stride"], strict=True):
        width += (k - 1) * stride
        stride *= s
    return width, stride


def tables(rows, feats, refs, *, timing=False):
    kt, cs = (
        grid(WEIGHTS_DIR / "config.json"),
        grid(SM13_FORMAL / "charsiu--en_w2v2_fc_10ms/config.json"),
    )
    clp = CONTINUOUS_SCALAR_NAMES.index("can_lp_vit_max")
    t, g = [], []
    for r, f in zip(rows, feats, strict=True):
        ph = r["canonical_phones"]
        ref = refs.get(r["uid"])
        timed = timing and ref is not None
        base = dict(
            uid=r["uid"],
            src=r["source"],
            spk=r["speaker"],
            l2=not r["source"].startswith("timit"),
        )
        if timed:
            info = sf.info(r["audio_path"])
            assert info.samplerate == 16000
            lo, hi, peaks, T = koel_voronoi_spans(BASE_KOEL_CACHE, r)
            with np.load(
                _slot_path(RUN / "caches/charsiu_10ms", r), allow_pickle=False
            ) as z:
                st, en, Tc = z["starts_10ms"], z["ends_10ms"], z["t_10ms"].item()
            assert (info.frames - kt[0]) // kt[1] + 1 == T and (
                info.frames - cs[0]
            ) // cs[1] + 1 == Tc
            # Physical time at convolutional receptive-field centre; estimated cell edges halfway between centres.
            pk = (peaks * kt[1] + (kt[0] - 1) / 2) / 16000
            k0 = np.maximum(0, (lo * kt[1] + (kt[0] - 1 - kt[1]) / 2) / 16000)
            k1 = (hi * kt[1] + (kt[0] - 1 - kt[1]) / 2) / 16000
            c0 = np.maximum(0, (st * cs[1] + (cs[0] - 1 - cs[1]) / 2) / 16000)
            c1 = (en * cs[1] + (cs[0] - 1 - cs[1]) / 2) / 16000
        for i, (p, y, rid) in enumerate(
            zip(ph, r["gold_targets"], r["realized_ids"], strict=True)
        ):
            can_lp = float(f["target_scalars"][i, clp])
            d = dict(
                **base,
                index=i,
                context=p,
                y=y,
                blind=can_lp > -0.2,
                can_lp=can_lp,
                one_frame=bool(
                    np.isclose(f["target_scalars"][i, 0], np.log(2), atol=1e-4)
                ),
                equal_hidden=bool(np.array_equal(f["h_vit"][i], f["h_peak"][i])),
            )
            if ref and y == "S":
                raw = normalize_canonical_phone(
                    ref["target_phones"][i]["realized_phone"]
                )
                d.update(
                    ref_kind="S_same" if raw == p else "S_alt",
                    real=raw,
                    aux_mismatch=rid != PHONE_TO_ID[raw],
                )
            if timed and y in ("C", "S"):
                a, b = (ref["target_phones"][i][key] for key in ("xmin", "xmax"))
                d.update(
                    peak_inside=a <= pk[i] <= b,
                    peak_inside20=a - 0.02 <= pk[i] <= b + 0.02,
                    peak_dist_ms=1000 * max(0, a - pk[i], pk[i] - b),
                )
                for name, x0, x1 in [
                    ("voronoi", k0[i], k1[i]),
                    ("charsiu", c0[i], c1[i]),
                ]:
                    union = max(b, x1) - min(a, x0)
                    d[name + "_iou"] = (
                        max(0, min(b, x1) - max(a, x0)) / union if union > 0 else None
                    )
            t.append(d)
        for i, y in enumerate(r["gold_gaps"]):
            d = dict(
                **base,
                index=i,
                context=(ph[i - 1] if i else "^")
                + "/"
                + (ph[i] if i < len(ph) else "$"),
                y=y,
            )
            if timed and y == "I":
                ins = ref["gaps"][i]["insertions"]
                mid = np.asarray([(z["xmin"] + z["xmax"]) / 2 for z in ins])
                assert len(mid)
                a = pk[i - 1] if i else 0.0
                b = pk[i] if i < len(ph) else info.duration
                d.update(
                    insertion_centres=len(mid),
                    inside=int(np.sum((a <= mid) & (mid <= b))),
                    inside20=int(np.sum((a - 0.02 <= mid) & (mid <= b + 0.02))),
                )
            g.append(d)
    t, g = pd.DataFrame(t), pd.DataFrame(g)
    assert (
        not t.duplicated(["uid", "index"]).any()
        and not g.duplicated(["uid", "index"]).any()
    )
    return t, g
