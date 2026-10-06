"""Statistics for the evaluation: paired bootstrap CIs and length-controlled win-rate."""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(values: list[float], resamples: int = 10000, confidence: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling PROMPTS (each value is one prompt's
    outcome), so the interval reflects which prompts happened to be in the eval set."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(resamples, len(arr)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def length_controlled_win_rate(outcomes: list[float], len_a: list[int],
                               len_b: list[int]) -> float | None:
    """AlpacaEval-2-style length control: fit P(A wins) = σ(w0 + w1·Δlen) on decisive
    comparisons and report σ(w0), the predicted win-rate when both answers are equally
    long. Returns None when it cannot be estimated (too few or perfectly separated data)."""
    import statsmodels.api as sm

    rows = [(o, a, b) for o, a, b in zip(outcomes, len_a, len_b) if o in (0.0, 1.0)]
    if len(rows) < 10 or len({o for o, _, _ in rows}) < 2:
        return None
    y = np.array([o for o, _, _ in rows])
    diff = np.array([a - b for _, a, b in rows], dtype=float)
    scale = diff.std() or 1.0
    X = sm.add_constant(diff / scale)
    try:
        fit = sm.Logit(y, X).fit(disp=0)
    except Exception:  # noqa: BLE001 — perfect separation etc.
        return None
    return float(1 / (1 + math.exp(-fit.params[0])))
