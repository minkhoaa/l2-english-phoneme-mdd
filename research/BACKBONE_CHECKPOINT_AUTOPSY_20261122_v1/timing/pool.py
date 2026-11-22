"""CPU matched timing intervention on only admitted L2-ARCTIC Fit -> Cal C/S slots."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..inputs import (
    BASE_KOEL_CACHE,
    RUN,
    WEIGHTS_DIR,
    SM13_FORMAL,
    grid,
    roles_and_refs,
    sha_file,
    normalize_canonical_phone,
)
from ..metrics import ap
from ..replay import permutations
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.backbones import (
    BACKBONE_SPECS,
    _slot_path,
    koel_voronoi_spans,
    read_frame_cache,
)
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.common import pool_koel_iso
from research.BACKBONE_REP_CAMPAIGN_20261122_v1.stages_0_3 import CAN_LP_IDX


def cut(centres, lo, hi):
    """Include centres in [lo,hi), rather than including a frame twice on a shared boundary."""
    a, b = np.searchsorted(centres, [lo, hi], side="left")
    fallback = a == b
    if fallback:
        a = int(np.argmin(abs(centres - (lo + hi) / 2)))
        b = a + 1
    return int(a), int(b), bool(fallback)


def assemble(rows, refs):
    meta = []
    base = []
    banks = {k: [] for k in ["voronoi", "charsiu", "reference"]}
    h = sha256()
    fallback = Counter()
    layer = 8
    kt, cs = (
        grid(WEIGHTS_DIR / "config.json"),
        grid(SM13_FORMAL / "charsiu--en_w2v2_fc_10ms/config.json"),
    )
    spec = BACKBONE_SPECS["wavlm_base_plus"]
    mid = spec.frame_layers_stage3.index(layer)
    peak_diff = 0.0
    for ni, r in enumerate(rows):
        assert r["source"] == "l2arctic_dev18" and r["uid"] in refs
        raw = read_frame_cache(RUN / "caches/frames_wavlm_base_plus", r)
        frames = np.ascontiguousarray(raw[:, mid, :])
        h.update(r["uid"].encode() + b"\0")
        h.update(frames.tobytes())
        lo, hi, _, T = koel_voronoi_spans(BASE_KOEL_CACHE, r)
        assert len(frames) == T
        centres = (np.arange(T) * kt[1] + (kt[0] - 1) / 2) / 16000
        with np.load(
            _slot_path(RUN / "caches/charsiu_10ms", r), allow_pickle=False
        ) as z:
            c0 = (z["starts_10ms"] * cs[1] + (cs[0] - 1 - cs[1]) / 2) / 16000
            c1 = (z["ends_10ms"] * cs[1] + (cs[0] - 1 - cs[1]) / 2) / 16000
        k = pool_koel_iso(BASE_KOEL_CACHE, r)
        x = np.column_stack([k["target_scalars"], k["h_peak"][:, 2, :]]).astype(
            np.float32
        )
        # Both caches were extracted independently: verify same L8 & same Voronoi on a fixed subset.
        old = None
        if ni % 30 == 0:
            with np.load(
                _slot_path(RUN / "caches/slots_wavlm_base_plus", r), allow_pickle=False
            ) as z:
                old = z["slot_mean"][:, spec.layers.index(layer)].astype(np.float32)
        for j, y in enumerate(r["gold_targets"]):
            if y not in ("C", "S"):
                continue
            z = refs[r["uid"]]["target_phones"][j]
            ph = r["canonical_phones"][j]
            v = frames[lo[j] : hi[j]].mean(0, dtype=np.float32)
            if old is not None:
                peak_diff = max(peak_diff, float(abs(old[j] - v).max()))
            banks["voronoi"].append(v)
            base.append(x[j])
            meta.append(
                dict(
                    uid=r["uid"],
                    index=j,
                    context=ph,
                    src=r["source"],
                    spk=r["speaker"],
                    y=y,
                    blind=k["target_scalars"][j, CAN_LP_IDX] > -0.2,
                    same=y == "S"
                    and normalize_canonical_phone(z["realized_phone"]) == ph,
                )
            )
            for mode, a, b in [
                ("charsiu", c0[j], c1[j]),
                ("reference", z["xmin"], z["xmax"]),
            ]:
                l, e, f = cut(centres, a, b)
                fallback[mode] += int(f)
                banks[mode].append(frames[l:e].mean(0, dtype=np.float32))
    assert peak_diff < 0.01, ("unexpected layer/pooling cache drift", peak_diff)
    return (
        pd.DataFrame(meta),
        np.asarray(base, dtype=np.float32),
        {k: np.stack(v) for k, v in banks.items()},
        dict(
            utts=len(rows),
            slots=len(meta),
            S=sum(z["y"] == "S" for z in meta),
            frames_L8_sha=h.hexdigest(),
            checked_voronoi_max_abs=peak_diff,
            fallbacks=dict(fallback),
        ),
    )


def evaluate(meta, score):
    y = meta.y.eq("S").to_numpy()
    slices = {
        "all": np.ones(len(y), bool),
        "blind": meta.blind,
        "nonblind": ~meta.blind,
        "same_vs_C": meta.same | ~y,
        "alternate_vs_C": ~meta.same,
    }
    slices.update({s: meta.spk == s for s in sorted(meta.spk.unique())})
    return {
        name: dict(
            slots=int(m.sum()),
            pos=int(y[m].sum()),
            prevalence=float(y[m].mean()),
            ap=ap(y[m], score[m]),
        )
        for name, m in slices.items()
    }


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU only")
    roles, refs, refsha = roles_and_refs()
    data = {}
    receipt = {}
    for role, rs in roles.items():
        print("CPU window assembly", role, flush=True)
        sub = [r for r in rs if r["source"] == "l2arctic_dev18"]
        meta, x, z, receipt[role] = assemble(sub, refs)
        assert not meta.duplicated(["uid", "index"]).any()
        z["permuted"] = z["voronoi"][permutations(meta)[11]]
        data[role] = (meta, x, z)
    train, cal = data["fit"], data["cal"]
    assert set(train[0].spk).isdisjoint(cal[0].spk)
    out = {}
    for mode in ["koel_only", "voronoi", "charsiu", "reference", "permuted"]:
        print("CPU linear probe", mode, flush=True)

        def matrix(d):
            return (
                d[1].copy()
                if mode == "koel_only"
                else np.column_stack((d[1], d[2][mode]))
            )

        scaler = StandardScaler(copy=False)
        xx = scaler.fit_transform(matrix(train))
        clf = Ridge(alpha=1000.0, solver="cholesky", copy_X=False)
        clf.fit(xx, train[0].y.eq("S").to_numpy(np.float32))
        del xx
        score = clf.predict(scaler.transform(matrix(cal)))
        out[mode] = evaluate(cal[0], score)
    return dict(
        scope="Post hoc matched window probe, L2-ARCTIC C/S only; diagnostic reference timing, not deployable",
        run_binding_sha256=sha_file(RUN / "RUN_BINDINGS.json"),
        code_sha256=sha_file(__file__),
        reference_rows_sha256=refsha,
        sklearn=sklearn_version,
        layer=8,
        alpha=1000,
        receipt=receipt,
        cal=out,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    path = p.parse_args().out.resolve()
    if path.exists() or path.is_relative_to(RUN.parents[1]):
        raise FileExistsError("Use a fresh path outside the campaign")
    result = json.dumps(run(), indent=2, allow_nan=False) + "\n"
    with path.open("x") as f:
        f.write(result)
    print(path, flush=True)
