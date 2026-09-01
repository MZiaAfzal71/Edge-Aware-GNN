"""Repository-structure and completed-result contract validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PARTITIONS = ("train", "validation", "calibration", "test")
REQUIRED_PATHS = (
    ".github/workflows/quality.yml",
    "ReadMe.md",
    "analysis/applicability_domain_summary.csv",
    "configs/benchmark.yaml",
    "configs/publication.yaml",
    "docs/PUBLICATION_ANALYSIS.md",
    "docs/RESEARCH_ROADMAP.md",
    "outputs/dataset_audit.csv",
    "outputs/metrics.csv",
    "outputs/software_manifest.json",
    "outputs/summary/aggregate_metrics.csv",
    "outputs/summary/pairwise_wilcoxon_holm.csv",
    "pyproject.toml",
    "src/edge_aware_gnn/cli.py",
    "src/edge_aware_gnn/publication.py",
    "src/edge_aware_gnn/statistics.py",
)
METRIC_COLUMNS = {
    "dataset",
    "split_strategy",
    "seed",
    "model",
    "partition",
    "git_commit",
    "rmse",
    "mae",
    "r2",
    "empirical_coverage",
    "mean_interval_width",
}
DOMAIN_COLUMNS = {
    "dataset",
    "split_strategy",
    "model",
    "similarity_bin",
    "n",
    "mean_similarity",
    "mae",
    "rmse",
    "empirical_coverage",
}


class RepositoryValidationError(RuntimeError):
    """Raised when one or more repository contract checks fail."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Repository validation failed:\n- " + "\n- ".join(issues))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def _tracked_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
    return [value.decode("utf-8") for value in result.stdout.split(b"\0") if value]


def _expected_run_keys(benchmark: dict[str, Any]) -> set[tuple[str, str, int, str]]:
    datasets = benchmark["datasets"]
    strategies = benchmark["split"]["strategies"]
    seeds = benchmark["split"]["seeds"]
    models = [item["name"] for item in benchmark["models"]]
    return {
        (str(dataset), str(strategy), int(seed), str(model))
        for dataset in datasets
        for strategy in strategies
        for seed in seeds
        for model in models
    }


def _prediction_run_keys(root: Path) -> set[tuple[str, str, int, str]]:
    keys: set[tuple[str, str, int, str]] = set()
    for path in (root / "outputs").glob("*/*/seed_*/*/predictions.csv"):
        relative = path.relative_to(root / "outputs")
        dataset, strategy, seed_name, model = relative.parts[:4]
        keys.add((dataset, strategy, int(seed_name.removeprefix("seed_")), model))
    return keys


def _check_tracked_layout(root: Path, issues: list[str]) -> list[str]:
    tracked = _tracked_files(root)
    for value in tracked:
        path = Path(value)
        if path.suffix == ".ipynb":
            issues.append(f"Notebook must not be tracked in CLI-only repository: {value}")
        if any(part.endswith(".egg-info") for part in path.parts):
            issues.append(f"Generated egg-info must not be tracked: {value}")
        if path.name == "predictions_with_applicability_domain.csv":
            issues.append(f"Large derived applicability-domain table must not be tracked: {value}")
        if path.parts and path.parts[0] == "publication_outputs":
            issues.append(f"Regenerable publication output must not be tracked: {value}")
    return tracked


def _check_metrics(
    root: Path,
    expected_runs: set[tuple[str, str, int, str]],
    issues: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    path = root / "outputs/metrics.csv"
    if not path.is_file():
        return pd.DataFrame(), []
    frame = pd.read_csv(path)
    missing = sorted(METRIC_COLUMNS - set(frame.columns))
    if missing:
        issues.append(f"outputs/metrics.csv is missing columns: {missing}")
        return frame, []
    key_columns = ["dataset", "split_strategy", "seed", "model", "partition"]
    if frame.duplicated(key_columns).any():
        issues.append("outputs/metrics.csv contains duplicate run/partition keys")
    expected_metric_rows = len(expected_runs) * len(PARTITIONS)
    if len(frame) != expected_metric_rows:
        issues.append(
            f"outputs/metrics.csv has {len(frame)} rows; expected {expected_metric_rows}"
        )
    actual_keys = {
        (row.dataset, row.split_strategy, int(row.seed), row.model)
        for row in frame.itertuples(index=False)
    }
    if actual_keys != expected_runs:
        issues.append(
            "Metric run matrix differs from benchmark configuration "
            f"(missing={len(expected_runs - actual_keys)}, extra={len(actual_keys - expected_runs)})"
        )
    partitions = set(frame["partition"].dropna().astype(str))
    if partitions != set(PARTITIONS):
        issues.append(f"Metric partitions are {sorted(partitions)}; expected {list(PARTITIONS)}")
    numeric = ["rmse", "mae", "r2", "empirical_coverage", "mean_interval_width"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        issues.append("outputs/metrics.csv contains non-finite reported metrics")
    commits = sorted(frame["git_commit"].dropna().astype(str).unique().tolist())
    if not commits:
        issues.append("outputs/metrics.csv contains no Git commit provenance")
    return frame, commits


def _check_run_files(
    root: Path,
    expected_runs: set[tuple[str, str, int, str]],
    issues: list[str],
) -> None:
    prediction_keys = _prediction_run_keys(root)
    if prediction_keys != expected_runs:
        issues.append(
            "Prediction-file run matrix differs from benchmark configuration "
            f"(missing={len(expected_runs - prediction_keys)}, "
            f"extra={len(prediction_keys - expected_runs)})"
        )
    for dataset, strategy, seed, model in sorted(expected_runs):
        directory = root / "outputs" / dataset / strategy / f"seed_{seed}" / model
        for name in ("predictions.csv", "run_metadata.json", "feature_names.txt"):
            path = directory / name
            if not path.is_file():
                issues.append(f"Missing run artifact: {path.relative_to(root)}")
        metadata_path = directory / "run_metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                issues.append(f"Invalid JSON: {metadata_path.relative_to(root)}")
            else:
                valid_records = (
                    isinstance(metadata, list)
                    and len(metadata) == len(PARTITIONS)
                    and all(isinstance(record, dict) for record in metadata)
                )
                if not valid_records:
                    issues.append(
                        "Run metadata must contain one object per partition: "
                        f"{metadata_path.relative_to(root)}"
                    )
                elif {str(record.get("partition")) for record in metadata} != set(PARTITIONS):
                    issues.append(
                        "Run metadata partition coverage is invalid: "
                        f"{metadata_path.relative_to(root)}"
                    )


def _check_domain_summary(
    root: Path,
    expected_runs: set[tuple[str, str, int, str]],
    issues: list[str],
) -> int:
    path = root / "analysis/applicability_domain_summary.csv"
    if not path.is_file():
        return 0
    frame = pd.read_csv(path)
    missing = sorted(DOMAIN_COLUMNS - set(frame.columns))
    if missing:
        issues.append(f"Applicability-domain summary is missing columns: {missing}")
        return len(frame)
    expected_conditions = {(dataset, strategy, model) for dataset, strategy, _, model in expected_runs}
    actual_conditions = set(
        frame[["dataset", "split_strategy", "model"]].itertuples(index=False, name=None)
    )
    if actual_conditions != expected_conditions:
        issues.append(
            "Applicability-domain model conditions are incomplete "
            f"(missing={len(expected_conditions - actual_conditions)}, "
            f"extra={len(actual_conditions - expected_conditions)})"
        )
    if (frame["n"] <= 0).any():
        issues.append("Applicability-domain summary contains non-positive bin counts")
    return len(frame)


def validate_repository(root: str | Path = ".") -> dict[str, Any]:
    """Validate the CLI-only layout and completed benchmark/publication inputs."""
    repository = Path(root).resolve()
    issues: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (repository / relative).is_file():
            issues.append(f"Missing required repository file: {relative}")

    tracked = _check_tracked_layout(repository, issues)
    benchmark_path = repository / "configs/benchmark.yaml"
    if not benchmark_path.is_file():
        raise RepositoryValidationError(issues)
    benchmark = _load_yaml(benchmark_path)
    expected_runs = _expected_run_keys(benchmark)

    metrics, commits = _check_metrics(repository, expected_runs, issues)
    _check_run_files(repository, expected_runs, issues)
    domain_rows = _check_domain_summary(repository, expected_runs, issues)

    audit_path = repository / "outputs/dataset_audit.csv"
    if audit_path.is_file():
        audit = pd.read_csv(audit_path)
        actual_datasets = set(audit.get("dataset", pd.Series(dtype=str)).astype(str))
        if actual_datasets != set(benchmark["datasets"]):
            issues.append("Dataset audit does not cover exactly the configured datasets")

    ignore_path = repository / ".gitignore"
    if ignore_path.is_file() and "publication_outputs/" not in ignore_path.read_text(encoding="utf-8"):
        issues.append(".gitignore must exclude regenerable publication_outputs/")

    if issues:
        raise RepositoryValidationError(issues)
    return {
        "status": "ok",
        "tracked_files": len(tracked),
        "datasets": list(benchmark["datasets"]),
        "models": [item["name"] for item in benchmark["models"]],
        "split_strategies": list(benchmark["split"]["strategies"]),
        "seeds": [int(value) for value in benchmark["split"]["seeds"]],
        "expected_runs": len(expected_runs),
        "metric_rows": len(metrics),
        "prediction_files": len(_prediction_run_keys(repository)),
        "applicability_domain_rows": domain_rows,
        "result_git_commits": commits,
    }
