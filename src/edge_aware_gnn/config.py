"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "project": "edge-aware-moleculenet",
    "data_root": "data/raw",
    "output_root": "outputs",
    "datasets": ["esol", "freesolv", "lipophilicity"],
    "models": [
        {"name": "mean_baseline", "family": "mean", "features": "none"},
        {"name": "rf_rdkit", "family": "random_forest", "features": "rdkit"},
        {"name": "node_gnn", "family": "node_gnn", "features": "none"},
        {"name": "edge_gnn", "family": "edge_gnn", "features": "none"},
        {"name": "hybrid_gated", "family": "edge_gnn", "features": "rdkit", "fusion": "gated"},
    ],
    "split": {
        "strategies": ["random", "scaffold"],
        "seeds": [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035],
        "fractions": [0.70, 0.10, 0.10, 0.10],
    },
    "features": {
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "max_missing_fraction": 0.10,
        "variance_threshold": 0.0,
        "correlation_threshold": 0.95,
        "standardized_clip": 10.0,
    },
    "training": {
        "batch_size": 64,
        "hidden_dim": 128,
        "layers": 3,
        "dropout": 0.15,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "max_epochs": 250,
        "patience": 30,
        "num_workers": 0,
        "conformal_alpha": 0.10,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, merge defaults, and reject invalid experimental settings."""
    with Path(path).open(encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    cfg = _deep_update(deepcopy(DEFAULTS), supplied)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    allowed_datasets = {"esol", "freesolv", "lipophilicity"}
    unknown = set(cfg["datasets"]) - allowed_datasets
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")

    allowed_splits = {"random", "scaffold"}
    strategies = set(cfg["split"]["strategies"])
    if not strategies <= allowed_splits:
        raise ValueError(f"Unsupported split strategies: {sorted(strategies - allowed_splits)}")

    fractions = cfg["split"]["fractions"]
    if len(fractions) != 4 or any(float(x) <= 0 for x in fractions):
        raise ValueError("split.fractions must contain four positive values")
    if abs(sum(map(float, fractions)) - 1.0) > 1e-8:
        raise ValueError("split.fractions must sum to 1")

    names = [model["name"] for model in cfg["models"]]
    if len(names) != len(set(names)):
        raise ValueError("Every model must have a unique name")

    if float(cfg["features"]["standardized_clip"]) <= 0:
        raise ValueError("features.standardized_clip must be positive")


def describe_plan(config: dict[str, Any]) -> dict[str, Any]:
    runs = (
        len(config["datasets"])
        * len(config["models"])
        * len(config["split"]["strategies"])
        * len(config["split"]["seeds"])
    )
    return {
        "datasets": config["datasets"],
        "models": [model["name"] for model in config["models"]],
        "split_strategies": config["split"]["strategies"],
        "seeds": config["split"]["seeds"],
        "total_independent_runs": runs,
        "selection_partition": "validation",
        "uncertainty_calibration_partition": "calibration",
        "final_reporting_partition": "test",
    }
