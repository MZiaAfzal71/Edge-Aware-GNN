"""Paired model-comparison procedures for repeated molecular benchmarks."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional sequence in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def pairwise_wilcoxon_holm(
    frame: pd.DataFrame,
    *,
    model_col: str = "model",
    value_col: str = "rmse",
    pair_cols: tuple[str, ...] = ("dataset", "split_strategy", "seed"),
) -> pd.DataFrame:
    """Compare models only after aligning the same dataset/split/seed rows."""
    wide = frame.pivot_table(index=list(pair_cols), columns=model_col, values=value_col, aggfunc="first")
    rows: list[dict[str, float | str | int]] = []
    for a, b in combinations(wide.columns, 2):
        paired = wide[[a, b]].dropna()
        if len(paired) < 2:
            continue
        delta = paired[a] - paired[b]
        try:
            statistic, p_value = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        rows.append(
            {
                "model_a": str(a),
                "model_b": str(b),
                "n_pairs": len(paired),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm_adjust(result["p_value"].tolist())
    return result


def corrected_resampled_ttest(
    differences: list[float] | np.ndarray,
    *,
    train_fraction: float,
    test_fraction: float,
) -> dict[str, float]:
    """Nadeau-Bengio correction for dependent repeated holdout/CV estimates."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two paired differences are required")
    if train_fraction <= 0 or test_fraction <= 0:
        raise ValueError("train_fraction and test_fraction must be positive")
    variance = values.var(ddof=1)
    correction = (1.0 / len(values)) + (test_fraction / train_fraction)
    standard_error = np.sqrt(correction * variance)
    if standard_error == 0:
        statistic = 0.0 if values.mean() == 0 else np.sign(values.mean()) * np.inf
    else:
        statistic = values.mean() / standard_error
    p_value = 0.0 if np.isinf(statistic) else 2.0 * student_t.sf(abs(statistic), df=len(values) - 1)
    return {
        "mean_delta": float(values.mean()),
        "standard_error": float(standard_error),
        "t_statistic": float(statistic),
        "degrees_of_freedom": float(len(values) - 1),
        "p_value": float(p_value),
    }

