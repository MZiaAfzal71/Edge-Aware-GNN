"""Paired model-comparison procedures for repeated molecular benchmarks."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from scipy.stats import t as student_t


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


def mean_confidence_interval(
    values: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Student-t confidence interval for the mean of seed-level values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("At least two finite values are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def matched_rank_biserial(differences: list[float] | np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size, positive when A exceeds B."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if values.ndim != 1 or len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values))
    denominator = float(ranks.sum())
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / denominator)


def specified_wilcoxon_holm(
    frame: pd.DataFrame,
    contrasts: list[dict[str, str]],
    *,
    model_col: str = "model",
    value_col: str = "rmse",
    pair_cols: tuple[str, ...] = ("seed",),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Evaluate an explicit family of aligned paired model contrasts."""
    wide = frame.pivot_table(index=list(pair_cols), columns=model_col, values=value_col, aggfunc="first")
    rows: list[dict[str, float | str | int]] = []
    for contrast in contrasts:
        model_a = contrast["model_a"]
        model_b = contrast["model_b"]
        if model_a not in wide or model_b not in wide:
            raise ValueError(f"Missing model for contrast {contrast['id']}: {model_a} vs {model_b}")
        paired = wide[[model_a, model_b]].dropna()
        if len(paired) < 2:
            raise ValueError(f"Contrast {contrast['id']} has fewer than two aligned pairs")
        delta = (paired[model_a] - paired[model_b]).to_numpy(dtype=float)
        if np.all(delta == 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                delta,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        ci_low, ci_high = mean_confidence_interval(delta, confidence=confidence)
        rows.append(
            {
                "contrast": contrast["id"],
                "question": contrast.get("question", contrast["id"]),
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": len(delta),
                "mean_a": float(paired[model_a].mean()),
                "mean_b": float(paired[model_b].mean()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "rank_biserial": matched_rank_biserial(delta),
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_value"].tolist())
    return result
