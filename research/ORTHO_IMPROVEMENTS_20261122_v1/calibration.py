"""Levers O3 (rho_o), O4 (Gap Platt/Temperature), and O5 (Cal-Locked N-Seed Ensemble).

Zero Eval gold leakage: every calibration parameter (tau_t, tau_g, rho_o, gap_temp,
gap_bias) is estimated strictly on Calibrate arrays with deterministic tie breaking
and an explicit Cal L1 native FA guard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax

VALID_TARGET_GOLD = ("C", "S", "O")
VALID_GAP_GOLD = ("N", "I")


@dataclass(frozen=True)
class OrthoPolicy:
    """Frozen inference decision policy locked strictly on Calibrate."""

    tau_t: float
    tau_g: float
    rho_o: float = 0.50  # Call O iff p_o / (p_s + p_o) >= rho_o
    gap_temp: float = 1.0  # Calibrated gap logit scale: (logit_g + gap_bias) / gap_temp
    gap_bias: float = 0.0  # Includes optional -log(gap_pos_weight) or Platt intercept
    ensemble_mode: str = "single"  # single | prob_mean | logit_mean
    n_seeds: int = 1
    cal_l2_strict_f1: float = 0.0
    cal_l1_fa_per_1k: float = 0.0
    cal_guard_met: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gap_log_odds(gap_logits_or_probs: np.ndarray, *, is_probs: bool = False) -> np.ndarray:
    arr = np.asarray(gap_logits_or_probs, dtype=np.float64)
    if is_probs:
        clipped = np.clip(arr, 1e-6, 1.0 - 1e-6)
        return logit(clipped)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr[:, 1] - arr[:, 0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Unexpected gap array shape: {arr.shape}")


def _target_probs(target_logits_or_probs: np.ndarray, *, is_probs: bool = False) -> np.ndarray:
    arr = np.asarray(target_logits_or_probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise ValueError(f"Expected [N, 3] target array, got {arr.shape}")
    if is_probs:
        p = np.clip(arr, 0.0, 1.0)
        return p / np.maximum(1e-12, p.sum(axis=-1, keepdims=True))
    return softmax(arr, axis=-1)


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error on binary labels {0, 1}."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def binary_nll(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Numerically stable binary cross-entropy from log-odds."""
    y = np.asarray(y_true, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    # log(1 + exp(z)) - y*z
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_gap_platt_cal(
    gap_logits_or_probs: np.ndarray,
    gap_gold: np.ndarray,
    *,
    is_probs: bool = False,
    known_pos_weight: float = 8.0,
    fit_temperature: bool = True,
    fit_bias: bool = True,
) -> Dict[str, float]:
    """Fit Cal-only (gap_temp, gap_bias) minimizing NLL on valid Cal gap slots.

    Removes the analytical +log(gap_pos_weight) shift (+2.079 for w=8.0) that compresses
    Gap probabilities into the [0.875, 0.90] sigmoid tail, restoring smooth probability
    calibration and ~2.5x finer threshold resolution near p=0.50.
    """
    gold = np.asarray(gap_gold)
    valid = np.isin(gold, VALID_GAP_GOLD)
    if not np.any(valid):
        raise ValueError("No valid N/I gap labels in Calibrate")
    y = (gold[valid] == "I").astype(np.float64)
    raw_z = _gap_log_odds(gap_logits_or_probs[valid], is_probs=is_probs)

    init_bias = -float(np.log(max(1.0, known_pos_weight)))
    raw_nll = binary_nll(y, raw_z)
    raw_ece = compute_ece(y, expit(raw_z))

    if not fit_temperature and not fit_bias:
        cal_z = raw_z + init_bias
        return {
            "gap_temp": 1.0,
            "gap_bias": init_bias,
            "cal_nll_before": raw_nll,
            "cal_nll_after": binary_nll(y, cal_z),
            "cal_ece_before": raw_ece,
            "cal_ece_after": compute_ece(y, expit(cal_z)),
            "cal_emp_i_rate": float(y.mean()),
            "cal_pred_i_before": float(expit(raw_z).mean()),
            "cal_pred_i_after": float(expit(cal_z).mean()),
        }

    def obj(params: np.ndarray) -> float:
        log_t = float(params[0]) if fit_temperature else 0.0
        bias = float(params[1]) if fit_bias else init_bias
        temp = float(np.exp(np.clip(log_t, -2.0, 2.5)))
        z_cal = (raw_z + bias) / temp
        # Tiny L2 regularization on log_t to prevent degenerate scale on small Cal sets
        return binary_nll(y, z_cal) + 1e-4 * (log_t * log_t)

    res = minimize(
        obj,
        x0=np.array([0.0, init_bias], dtype=np.float64),
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 200},
    )
    opt_temp = float(np.exp(np.clip(res.x[0], -2.0, 2.5))) if fit_temperature else 1.0
    opt_bias = float(res.x[1]) if fit_bias else init_bias
    cal_z = (raw_z + opt_bias) / opt_temp
    cal_p = expit(cal_z)
    return {
        "gap_temp": round(opt_temp, 6),
        "gap_bias": round(opt_bias, 6),
        "cal_nll_before": round(raw_nll, 6),
        "cal_nll_after": round(binary_nll(y, cal_z), 6),
        "cal_ece_before": round(raw_ece, 6),
        "cal_ece_after": round(compute_ece(y, cal_p), 6),
        "cal_emp_i_rate": round(float(y.mean()), 6),
        "cal_pred_i_before": round(float(expit(raw_z).mean()), 6),
        "cal_pred_i_after": round(float(cal_p.mean()), 6),
    }


def ensemble_predictions(
    seed_targets: Sequence[np.ndarray],
    seed_gaps: Sequence[np.ndarray],
    *,
    mode: str = "prob_mean",
    is_probs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Combine N-seed head predictions on a shared backbone pass (Lever O5).

    Args:
        seed_targets: sequence of [N, 3] arrays (logits if is_probs=False else probs).
        seed_gaps: sequence of [M, 2] or [M] arrays (logits if is_probs=False else probs).
        mode: 'prob_mean' (average probabilities) or 'logit_mean' (average logits).
        is_probs: whether inputs are already probabilities.

    Returns:
        (ens_targets, ens_gaps, out_is_probs)
    """
    if len(seed_targets) == 0 or len(seed_targets) != len(seed_gaps):
        raise ValueError("seed_targets and seed_gaps must be non-empty and have equal length")
    if mode not in ("prob_mean", "logit_mean"):
        raise ValueError(f"Unsupported ensemble mode: {mode}")

    if mode == "logit_mean":
        if is_probs:
            raise ValueError("mode='logit_mean' requires raw logits (is_probs=False)")
        t_ens = np.mean([np.asarray(x, dtype=np.float64) for x in seed_targets], axis=0)
        g_ens = np.mean([np.asarray(x, dtype=np.float64) for x in seed_gaps], axis=0)
        return t_ens, g_ens, False

    t_probs = np.mean([_target_probs(x, is_probs=is_probs) for x in seed_targets], axis=0)
    g_probs = np.mean(
        [expit(_gap_log_odds(x, is_probs=is_probs)) for x in seed_gaps], axis=0
    )
    return t_probs, g_probs, True


def _strict_counts_target(
    gold: np.ndarray,
    t_probs: np.ndarray,
    tau_t: float,
    rho_o: float,
) -> Tuple[int, int, int]:
    err_score = t_probs[:, 1] + t_probs[:, 2]
    flagged = err_score >= tau_t
    frac_o = t_probs[:, 2] / np.maximum(1e-12, err_score)
    pred = np.where(flagged, np.where(frac_o >= rho_o, "O", "S"), "C")
    g_err = gold != "C"
    p_err = pred != "C"
    tp = int(np.sum(g_err & (pred == gold)))
    fp = int(np.sum(p_err & (pred != gold)))
    fn = int(np.sum(g_err & (pred != gold)))
    return tp, fp, fn


def _strict_counts_gap(
    gold: np.ndarray,
    g_probs: np.ndarray,
    tau_g: float,
) -> Tuple[int, int, int]:
    flagged = g_probs >= tau_g
    g_err = gold == "I"
    tp = int(np.sum(g_err & flagged))
    fp = int(np.sum((~g_err) & flagged))
    fn = int(np.sum(g_err & (~flagged)))
    return tp, fp, fn


def calibrate_ortho_policy(
    l2_target_arr: np.ndarray,
    l2_target_gold: np.ndarray,
    l2_gap_arr: np.ndarray,
    l2_gap_gold: np.ndarray,
    l1_target_arr: np.ndarray,
    l1_gap_arr: np.ndarray,
    *,
    is_probs: bool = False,
    tune_rho_o: bool = False,
    fit_gap_temp: bool = False,
    known_gap_pos_weight: float = 8.0,
    max_l1_fa_per_1k: float = 15.0,
    tau_t_grid: Sequence[float] | None = None,
    tau_g_grid: Sequence[float] | None = None,
    rho_o_grid: Sequence[float] | None = None,
    ensemble_mode: str = "single",
    n_seeds: int = 1,
) -> Tuple[OrthoPolicy, Dict[str, Any]]:
    """Lock (tau_t, tau_g, rho_o, gap_temp, gap_bias) strictly on Calibrate.

    Uses exact tie-safe maximization of Cal L2 strict micro-F1 (S/O/I) subject to
    Cal L1 total FA/1k <= max_l1_fa_per_1k.
    """
    t_gold_all = np.asarray(l2_target_gold)
    g_gold_all = np.asarray(l2_gap_gold)
    mt = np.isin(t_gold_all, VALID_TARGET_GOLD)
    mg = np.isin(g_gold_all, VALID_GAP_GOLD)
    t_gold = t_gold_all[mt]
    g_gold = g_gold_all[mg]

    t_probs_l2 = _target_probs(l2_target_arr[mt], is_probs=is_probs)
    t_probs_l1 = _target_probs(l1_target_arr, is_probs=is_probs)
    t_sc_l1 = t_probs_l1[:, 1] + t_probs_l1[:, 2]

    # Optional Lever O4: Cal-only Gap Platt/Temperature calibration
    gap_cal_info: Dict[str, float] = {"gap_temp": 1.0, "gap_bias": 0.0}
    if fit_gap_temp:
        # Fit on combined Cal L2 valid gaps + Cal L1 clean N gaps so L1 prior is respected
        l1_g_gold = np.full(len(l1_gap_arr), "N", dtype=object)
        comb_g_arr = np.concatenate([l2_gap_arr[mg], l1_gap_arr], axis=0)
        comb_g_gold = np.concatenate([g_gold, l1_g_gold], axis=0)
        gap_cal_info = fit_gap_platt_cal(
            comb_g_arr,
            comb_g_gold,
            is_probs=is_probs,
            known_pos_weight=known_gap_pos_weight,
            fit_temperature=True,
            fit_bias=True,
        )

    g_temp = float(gap_cal_info["gap_temp"])
    g_bias = float(gap_cal_info["gap_bias"])
    g_probs_l2 = expit((_gap_log_odds(l2_gap_arr[mg], is_probs=is_probs) + g_bias) / g_temp)
    g_probs_l1 = expit((_gap_log_odds(l1_gap_arr, is_probs=is_probs) + g_bias) / g_temp)

    if tau_t_grid is None:
        tau_t_grid = [round(x, 3) for x in np.linspace(0.15, 0.75, 61)]
    if tau_g_grid is None:
        if fit_gap_temp:
            tau_g_grid = [round(x, 3) for x in np.linspace(0.10, 0.90, 81)]
        else:
            tau_g_grid = [round(x, 3) for x in np.linspace(0.40, 0.975, 47)]
    if rho_o_grid is None:
        rho_o_grid = [round(x, 3) for x in np.linspace(0.30, 0.70, 17)] if tune_rho_o else [0.50]

    n_l1_slots = max(1, len(t_sc_l1) + len(g_probs_l1))

    # Precompute target & gap tables for fast exact Cartesian sweep (deterministic)
    t_table = []
    for tt in tau_t_grid:
        l1_t_fa = int(np.sum(t_sc_l1 >= tt))
        for rho in rho_o_grid:
            tp_t, fp_t, fn_t = _strict_counts_target(t_gold, t_probs_l2, float(tt), float(rho))
            t_table.append((float(tt), float(rho), tp_t, fp_t, fn_t, l1_t_fa))

    g_table = []
    for tg in tau_g_grid:
        l1_g_fa = int(np.sum(g_probs_l1 >= tg))
        tp_g, fp_g, fn_g = _strict_counts_gap(g_gold, g_probs_l2, float(tg))
        g_table.append((float(tg), tp_g, fp_g, fn_g, l1_g_fa))

    best_guarded = None
    best_fallback = None
    for tt, rho, tp_t, fp_t, fn_t, l1_t_fa in t_table:
        for tg, tp_g, fp_g, fn_g, l1_g_fa in g_table:
            tp = tp_t + tp_g
            fp = fp_t + fp_g
            fn = fn_t + fn_g
            f1 = 200.0 * tp / max(1, 2 * tp + fp + fn)
            l1_fa_1k = 1000.0 * (l1_t_fa + l1_g_fa) / n_l1_slots
            # Deterministic key: maximize F1 (rounded to 1e-6), then minimize L1 FA, then higher tau
            key = (round(f1, 6), -round(l1_fa_1k, 6), tt, tg, -abs(rho - 0.50))
            cand = (key, f1, l1_fa_1k, tt, tg, rho)
            if l1_fa_1k <= max_l1_fa_per_1k:
                if best_guarded is None or cand[0] > best_guarded[0]:
                    best_guarded = cand
            if best_fallback is None or cand[0] > best_fallback[0]:
                best_fallback = cand

    guard_met = best_guarded is not None
    chosen = best_guarded if guard_met else best_fallback
    assert chosen is not None
    _, best_f1, best_l1_fa, best_tt, best_tg, best_rho = chosen

    policy = OrthoPolicy(
        tau_t=best_tt,
        tau_g=best_tg,
        rho_o=best_rho,
        gap_temp=g_temp,
        gap_bias=g_bias,
        ensemble_mode=ensemble_mode,
        n_seeds=int(n_seeds),
        cal_l2_strict_f1=round(float(best_f1), 4),
        cal_l1_fa_per_1k=round(float(best_l1_fa), 4),
        cal_guard_met=guard_met,
    )
    return policy, {"gap_calibration": gap_cal_info}


def apply_ortho_policy(
    target_arr: np.ndarray,
    gap_arr: np.ndarray,
    policy: OrthoPolicy | Mapping[str, Any],
    *,
    is_probs: bool = False,
) -> Dict[str, np.ndarray]:
    """Apply a locked Calibrate OrthoPolicy to target and gap arrays (no gold required)."""
    pol = policy if isinstance(policy, OrthoPolicy) else OrthoPolicy(**dict(policy))
    t_probs = _target_probs(target_arr, is_probs=is_probs)
    t_score = t_probs[:, 1] + t_probs[:, 2]
    frac_o = t_probs[:, 2] / np.maximum(1e-12, t_score)
    t_call = np.where(t_score >= pol.tau_t, np.where(frac_o >= pol.rho_o, "O", "S"), "C")

    g_z = _gap_log_odds(gap_arr, is_probs=is_probs)
    g_score = expit((g_z + pol.gap_bias) / pol.gap_temp)
    g_call = np.where(g_score >= pol.tau_g, "I", "N")

    return {
        "target_score": t_score.astype(np.float64),
        "p_s": t_probs[:, 1].astype(np.float64),
        "p_o": t_probs[:, 2].astype(np.float64),
        "target_call": t_call,
        "gap_score": g_score.astype(np.float64),
        "gap_call": g_call,
    }
