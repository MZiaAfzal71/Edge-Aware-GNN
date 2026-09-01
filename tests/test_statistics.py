import numpy as np
import pandas as pd

from edge_aware_gnn.statistics import (
    corrected_resampled_ttest,
    holm_adjust,
    matched_rank_biserial,
    mean_confidence_interval,
    pairwise_wilcoxon_holm,
    specified_wilcoxon_holm,
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


def test_mean_confidence_interval_contains_sample_mean():
    values = [0.4, 0.5, 0.6, 0.7]
    low, high = mean_confidence_interval(values)
    assert low < np.mean(values) < high


def test_matched_rank_biserial_has_expected_direction():
    assert matched_rank_biserial([-0.4, -0.3, -0.2]) == -1.0
    assert matched_rank_biserial([0.4, 0.3, 0.2]) == 1.0


def test_specified_wilcoxon_uses_only_requested_contrasts():
    frame = pd.DataFrame(
        {
            "seed": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "model": ["a"] * 3 + ["b"] * 3 + ["unused"] * 3,
            "rmse": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.1, 0.1],
        }
    )
    contrasts = [{"id": "a_vs_b", "model_a": "a", "model_b": "b", "question": "test"}]
    result = specified_wilcoxon_holm(frame, contrasts)
    assert result["contrast"].tolist() == ["a_vs_b"]
    assert np.isclose(result.loc[0, "mean_delta"], -0.3)
    assert result.loc[0, "rank_biserial"] == -1.0
