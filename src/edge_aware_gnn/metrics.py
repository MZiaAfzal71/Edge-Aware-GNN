"""Regression metrics and confidence intervals with correct argument order."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return RMSE, MAE, and R2; all sklearn calls receive y_true first."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")
    if truth.size < 2 or not (np.isfinite(truth).all() and np.isfinite(pred).all()):
        raise ValueError("Metrics require at least two finite paired observations")
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "mae": float(mean_absolute_error(truth, pred)),
        "r2": float(r2_score(truth, pred)),
    }


def paired_bootstrap_delta(
    y_true: Iterable[float],
    pred_a: Iterable[float],
    pred_b: Iterable[float],
    *,
    metric: str = "mae",
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> dict[str, float]:
    """Bootstrap the paired error delta (model A minus model B)."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    a = np.asarray(pred_a, dtype=float).reshape(-1)
    b = np.asarray(pred_b, dtype=float).reshape(-1)
    if not (truth.shape == a.shape == b.shape):
        raise ValueError("Truth and both prediction vectors must have identical shapes")
    if metric not in {"mae", "rmse"}:
        raise ValueError("metric must be 'mae' or 'rmse'")

    def score(y: np.ndarray, p: np.ndarray) -> float:
        error = y - p
        return float(np.mean(np.abs(error))) if metric == "mae" else float(np.sqrt(np.mean(error**2)))

    observed = score(truth, a) - score(truth, b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.integers(0, truth.size, size=truth.size)
        deltas[i] = score(truth[sample], a[sample]) - score(truth[sample], b[sample])
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"delta": observed, "ci_low": float(low), "ci_high": float(high)}

