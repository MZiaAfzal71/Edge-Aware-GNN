import numpy as np
import pandas as pd

from edge_aware_gnn.statistics import (
    corrected_resampled_ttest,
    holm_adjust,
    pairwise_wilcoxon_holm,
)


def test_holm_adjust_is_bounded_and_not_smaller_than_raw():
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw)
    assert all(0 <= value <= 1 for value in adjusted)
    assert all(a >= p for a, p in zip(adjusted, raw))


def test_wilcoxon_pairs_by_split_identity():
    frame = pd.DataFrame(
        {
            "dataset": ["esol"] * 6,
            "split_strategy": ["scaffold"] * 6,
            "seed": [1, 2, 3, 1, 2, 3],
            "model": ["a", "a", "a", "b", "b", "b"],
            "rmse": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    result = pairwise_wilcoxon_holm(frame)
    assert result.loc[0, "n_pairs"] == 3
    assert np.isclose(result.loc[0, "mean_delta"], -0.3)
    assert "p_holm" in result


def test_corrected_resampled_ttest_returns_finite_probability():
    result = corrected_resampled_ttest([0.2, 0.1, 0.3, 0.2], train_fraction=0.8, test_fraction=0.2)
    assert 0 <= result["p_value"] <= 1

