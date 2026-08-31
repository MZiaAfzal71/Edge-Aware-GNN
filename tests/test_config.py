from pathlib import Path

import pytest

from edge_aware_gnn.config import load_config, validate_config


def test_repository_benchmark_configuration_is_valid():
    config = load_config(Path(__file__).parents[1] / "configs" / "benchmark.yaml")
    assert config["datasets"] == ["esol", "freesolv", "lipophilicity"]
    assert config["split"]["fractions"] == [0.70, 0.10, 0.10, 0.10]


def test_invalid_split_sum_is_rejected():
    config = load_config(Path(__file__).parents[1] / "configs" / "benchmark.yaml")
    config["split"]["fractions"] = [0.7, 0.1, 0.1, 0.2]
    with pytest.raises(ValueError, match="sum"):
        validate_config(config)
