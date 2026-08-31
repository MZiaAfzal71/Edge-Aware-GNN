import numpy as np
import pytest

from edge_aware_gnn.metrics import paired_bootstrap_delta, regression_metrics


def test_regression_metrics_use_truth_as_first_argument():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.0, 1.0, 2.0, 4.0])
    metrics = regression_metrics(truth, pred)
    assert np.isclose(metrics["rmse"], 0.5)
    assert np.isclose(metrics["mae"], 0.25)
    assert np.isclose(metrics["r2"], 0.8)


def test_regression_metrics_report_non_finite_predictions():
    with pytest.raises(FloatingPointError, match=r"y_true=0, y_pred=1, n=2"):
        regression_metrics([0.0, 1.0], [0.0, np.nan])


def test_paired_bootstrap_delta_prefers_lower_error_model():
    truth = np.arange(20.0)
    better = truth + 0.1
    worse = truth + 1.0
    result = paired_bootstrap_delta(truth, better, worse, n_bootstrap=200, seed=7)
    assert result["delta"] < 0
    assert result["ci_high"] < 0
