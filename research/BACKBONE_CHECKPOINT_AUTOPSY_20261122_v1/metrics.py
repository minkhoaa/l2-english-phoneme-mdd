"""Paired, descriptive metrics. No threshold fitting or model selection on Eval."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def ap(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None


def calls(pred, policy):
    ps, po = pred[0].T
    ts = (ps + po).astype(np.float64)
    gs = pred[1].astype(np.float64)
    return (
        ts,
        gs,
        np.where(ts >= policy["tau_t"], np.where(po > ps, "O", "S"), "C"),
        np.where(gs >= policy["tau_g"], "I", "N"),
    )


def counts(y, c, ks):
    valid = np.isin(y, ks)
    alert = np.isin(c, ks)
    tp = int((valid & (y == c)).sum())
    return np.array(
        [tp, int(alert.sum()) - tp, int(valid.sum()) - tp]
    )  # caller masks unknown FIRST


def f1(cc):
    tp, fp, fn = cc
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def view(t, g, pred, policy, *, detailed=True):
    ts, gs, tc, gc = calls(pred, policy)
    tm = t.l2.to_numpy() & np.isin(t.y, ["C", "S", "O"])
    gm = g.l2.to_numpy() & np.isin(g.y, ["N", "I"])
    yt = t.y.to_numpy()
    yg = g.y.to_numpy()
    pos = np.isin(yt, ["S", "O"])
    ins = yg == "I"
    blind = t.blind.to_numpy()

    def group(mt, mg):
        mt = tm & mt
        mg = gm & mg
        cc = counts(yt[mt], tc[mt], ["S", "O"]) + counts(yg[mg], gc[mg], ["I"])
        return dict(
            tp_fp_fn=cc.tolist(),
            strict_f1=f1(cc),
            tAP=ap(pos[mt], ts[mt]),
            gAP=ap(ins[mg], gs[mg]),
            blindAP=ap(pos[mt & blind], ts[mt & blind]),
            gold_SO=int(pos[mt].sum()),
            gold_I=int(ins[mg].sum()),
        )

    out = group(True, True)
    if not detailed:
        return out
    out["per_class"] = {k: counts(yt[tm], tc[tm], [k]).tolist() for k in ["S", "O"]}
    out["per_class"]["I"] = counts(yg[gm], gc[gm], ["I"]).tolist()
    out["s_to_o"] = int(np.sum(tm & (yt == "S") & (tc == "O")))
    out["o_to_s"] = int(np.sum(tm & (yt == "O") & (tc == "S")))
    out["oracle_only_perfect_gaps_fixed_target_f1"] = f1(
        counts(yt[tm], tc[tm], ["S", "O"]) + np.array([ins[gm].sum(), 0, 0])
    )
    out["by_speaker"] = {
        s: group(t.spk == s, g.spk == s) for s in sorted(t.spk[t.l2].unique())
    }
    out["by_source"] = {
        s: group(t.src == s, g.src == s) for s in sorted(t.src[t.l2].unique())
    }
    for name, sl in [("blind", tm & blind), ("outside_blind", tm & ~blind)]:
        tp = int(np.sum(sl & pos & (tc != "C")))
        fn = int(np.sum(sl & pos & (tc == "C")))
        fp = int(np.sum(sl & ~pos & (tc != "C")))
        out[name] = dict(
            n=int(sl.sum()),
            pos=tp + fn,
            binary_tp=tp,
            binary_fn=fn,
            binary_fp=fp,
            prevalence=float(np.mean(pos[sl])),
            AP=ap(pos[sl], ts[sl]),
        )
    out["l1"] = dict(
        target_alert=int(np.sum(~t.l2 & (tc != "C"))),
        targets=int(np.sum(~t.l2)),
        gap_alert=int(np.sum(~g.l2 & (gc == "I"))),
        gaps=int(np.sum(~g.l2)),
    )
    out["arctic_S"] = {}
    for k in ["S_same", "S_alt"]:
        m = t.ref_kind.eq(k).to_numpy() & tm
        out["arctic_S"][k] = dict(
            n=int(m.sum()),
            blind=int(np.sum(m & blind)),
            correct_S=int(np.sum(m & (tc == "S"))),
            called_C=int(np.sum(m & (tc == "C"))),
            called_O=int(np.sum(m & (tc == "O"))),
            median_score=float(np.median(ts[m])) if m.any() else None,
        )
    return out


def closest(scores, k):
    vals, n = np.unique(scores, return_counts=True)
    assert len(vals) > 0
    thresholds = np.r_[np.nextafter(float(vals[-1]), np.inf), vals[::-1]]
    actual = np.r_[0, n[::-1].cumsum()]
    i = int(np.argmin(abs(actual - k)))
    return float(thresholds[i]), int(actual[i])


def paired(t, g, ref, refpol, arm, armpol):
    ts, gs, tc, gc = calls(ref, refpol)
    us, vs, uc, vc = calls(arm, armpol)
    mt = t.l2.to_numpy() & np.isin(t.y, ["C", "S", "O"])
    mg = g.l2.to_numpy() & np.isin(g.y, ["N", "I"])
    yt = t.y.to_numpy()
    yg = g.y.to_numpy()
    p = mt & np.isin(yt, ["S", "O"])
    i = mg & (yg == "I")
    kt = int((tc[p] != "C").sum())
    kg = int((gc[i] == "I").sum())
    ut, at = closest(us[p], kt)
    ug, ag = closest(vs[i], kg)
    a, b = view(t, g, ref, refpol), view(t, g, arm, armpol)
    sp = sorted(a["by_speaker"])
    rng = np.random.default_rng(20261122)
    draws = np.concatenate(
        [
            rng.choice(
                [s for s in sp if s in set(t.spk[t.src == src])],
                (3000, t.spk[t.src == src].nunique()),
                replace=True,
            )
            for src in sorted(t.src[t.l2].unique())
        ],
        axis=1,
    )

    def pooled(m):
        vals = np.asarray([m["by_speaker"][s]["tp_fp_fn"] for s in sp])[
            np.searchsorted(sp, draws)
        ].sum(1)
        return (
            200 * vals[:, 0] / np.maximum(1, 2 * vals[:, 0] + vals[:, 1] + vals[:, 2])
        )

    ci = np.quantile(pooled(b) - pooled(a), [0.025, 0.975]).tolist()
    swap = {}
    for name, y, c, d, m, k in [
        ("S", yt, tc, uc, mt, "S"),
        ("O", yt, tc, uc, mt, "O"),
        ("I", yg, gc, vc, mg, "I"),
    ]:
        swap[name] = dict(
            rescued=int(np.sum(m & (y == k) & (c != k) & (d == k))),
            lost=int(np.sum(m & (y == k) & (c == k) & (d != k))),
            both_miss=int(np.sum(m & (y == k) & (c != k) & (d != k))),
            new_fp=int(np.sum(m & (y != k) & (c != k) & (d == k))),
            removed_fp=int(np.sum(m & (y != k) & (c == k) & (d != k))),
        )
    return dict(
        swaps=swap,
        locked_target_S0_gap_arm_f1=f1(
            counts(yt[mt], tc[mt], ["S", "O"]) + counts(yg[mg], vc[mg], ["I"])
        ),
        locked_target_arm_gap_S0_f1=f1(
            counts(yt[mt], uc[mt], ["S", "O"]) + counts(yg[mg], gc[mg], ["I"])
        ),
        strict_delta_pp=100 * (b["strict_f1"] - a["strict_f1"]),
        descriptive_speaker_bootstrap_delta_pp_95=ci,
        rank_corr_target=float(spearmanr(ts[mt], us[mt]).statistic),
        rank_corr_gap=float(spearmanr(gs[mg], vs[mg]).statistic),
        by_speaker_delta_pp={
            s: 100 * (b["by_speaker"][s]["strict_f1"] - a["by_speaker"][s]["strict_f1"])
            for s in sp
        },
        matched_positive_counts=dict(target_ref_arm=[kt, at], gap_ref_arm=[kg, ag]),
        matched_recall_diagnostic=view(
            t, g, arm, dict(tau_t=ut, tau_g=ug), detailed=False
        ),
    )


def geometry(t, g):
    m = t.l2.to_numpy() & np.isin(t.y, ["S", "O", "C"])
    out = dict(
        valid_l2=int(m.sum()),
        one_frame=int(t.one_frame[m].sum()),
        identical_hidden=int(t.equal_hidden[m].sum()),
    )
    out["annotated_cal_arctic"] = {}
    for name, ix in [
        ("C", t.y.eq("C")),
        ("S_alt", t.ref_kind.eq("S_alt")),
        ("S_same", t.ref_kind.eq("S_same")),
    ]:
        s = t[ix & t.peak_inside.notna()]
        out["annotated_cal_arctic"][name] = dict(
            n=len(s),
            # Mixed missing values + numpy.bool_ may produce object dtype;
            # summing objects can reduce by logical OR instead of integer addition.
            peak_inside=int(np.count_nonzero(s.peak_inside.to_numpy(dtype=bool))),
            inside_20ms=int(np.count_nonzero(s.peak_inside20.to_numpy(dtype=bool))),
            median_voronoi_iou=float(s.voronoi_iou.median()),
            median_charsiu_iou=float(s.charsiu_iou.median()),
            median_can_lp=float(s.can_lp.median()),
            n_blind=int(s.blind.sum()),
        )
    out["annotated_insertion_centres"] = {
        k: int(g[k].sum()) for k in ["insertion_centres", "inside", "inside20"]
    }
    return out
