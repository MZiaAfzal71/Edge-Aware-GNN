"""End-to-end benchmark runner with explicit prediction provenance."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from torch_geometric.loader import DataLoader

from .config import describe_plan
from .data import DATASET_REGISTRY, PARTITIONS, audit_as_dict, load_moleculenet, make_split
from .features import build_feature_transformer, make_graph_dataset
from .metrics import regression_metrics
from .models import EdgeAwareRegressor, count_parameters


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> TargetScaler:
        mean = float(np.mean(values))
        scale = float(np.std(values, ddof=0))
        return cls(mean=mean, scale=scale if scale > 0 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def software_manifest() -> dict[str, Any]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "PyYAML",
        "rdkit",
        "torch",
        "torch-geometric",
        "xgboost",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "packages": versions,
    }


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch).detach().cpu().numpy().reshape(-1))
        truth.append(batch.y.detach().cpu().numpy().reshape(-1))
    return np.concatenate(truth), np.concatenate(predictions)


def _train_gnn(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    options: dict[str, Any],
) -> tuple[nn.Module, int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
    )
    criterion = nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(options["max_epochs"]) + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch.y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        validation_truth, validation_pred = _predict(model, validation_loader, device)
        validation_loss = float(np.mean((validation_truth - validation_pred) ** 2))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(options["patience"]):
            break
    model.load_state_dict(best_state)
    return model, best_epoch, time.perf_counter() - started


def _result_rows(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    metadata: dict[str, Any],
    conformal_radius: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        mask = frame["partition"].to_numpy() == partition
        values = regression_metrics(frame.loc[mask, "target"], predictions[mask])
        truth = frame.loc[mask, "target"].to_numpy(dtype=float)
        prediction = predictions[mask]
        coverage = float(np.mean(np.abs(truth - prediction) <= conformal_radius))
        rows.append(
            {
                **metadata,
                "partition": partition,
                "n": int(mask.sum()),
                **values,
                "empirical_coverage": coverage,
                "mean_interval_width": 2.0 * conformal_radius,
            }
        )
    return rows


def run_one(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    model_spec: dict[str, Any],
    split_strategy: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Fit one model using validation only for selection and test only for reporting."""
    set_seed(seed)
    split = make_split(
        frame["smiles"],
        strategy=split_strategy,
        seed=seed,
        fractions=config["split"]["fractions"],
    )
    work = frame.reset_index(drop=True).copy()
    work["partition"] = split["partition"]
    work["scaffold"] = split["scaffold"]
    train_mask = work["partition"].eq("train").to_numpy()
    target_scaler = TargetScaler.fit(work.loc[train_mask, "target"].to_numpy())
    scaled_targets = target_scaler.transform(work["target"].to_numpy())

    feature_set = model_spec.get("features", "none")
    transformer = build_feature_transformer(feature_set, config["features"])
    transformer.fit(work.loc[train_mask, "smiles"])
    features = transformer.transform(work["smiles"])
    family = model_spec["family"]
    best_epoch = 0
    n_parameters = 0
    started = time.perf_counter()

    if family == "mean":
        predictions = np.full(len(work), target_scaler.mean, dtype=float)
        elapsed = time.perf_counter() - started
    elif family in {"random_forest", "xgboost"}:
        if features.shape[1] == 0:
            raise ValueError("Random Forest requires a non-empty feature set")
        parameters = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": seed,
            **model_spec.get("parameters", {}),
        }
        if family == "random_forest":
            model = RandomForestRegressor(**parameters)
        else:
            try:
                from xgboost import XGBRegressor
            except ImportError as error:
                raise ImportError("XGBoost baseline requires: pip install -e '.[boosting]'") from error
            parameters = {
                "n_estimators": 800,
                "learning_rate": 0.03,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": seed,
                **model_spec.get("parameters", {}),
            }
            model = XGBRegressor(**parameters)
        model.fit(features[train_mask], scaled_targets[train_mask])
        predictions = target_scaler.inverse_transform(model.predict(features))
        elapsed = time.perf_counter() - started
    elif family in {"node_gnn", "edge_gnn"}:
        device = _device()
        fusion = model_spec.get("fusion", "none")
        global_features = None if fusion == "none" else features
        if fusion != "none" and features.shape[1] == 0:
            raise ValueError("A fused GNN requires global features")
        datasets = {}
        loaders = {}
        generator = torch.Generator().manual_seed(seed)
        for partition in PARTITIONS:
            mask = work["partition"].eq(partition).to_numpy()
            datasets[partition] = make_graph_dataset(
                work.loc[mask].reset_index(drop=True),
                scaled_targets[mask],
                None if global_features is None else global_features[mask],
            )
            loaders[partition] = DataLoader(
                datasets[partition],
                batch_size=int(config["training"]["batch_size"]),
                shuffle=partition == "train",
                num_workers=int(config["training"]["num_workers"]),
                generator=generator if partition == "train" else None,
            )
        model = EdgeAwareRegressor(
            global_dim=0 if global_features is None else global_features.shape[1],
            fusion=fusion,
            hidden_dim=int(config["training"]["hidden_dim"]),
            layers=int(config["training"]["layers"]),
            dropout=float(config["training"]["dropout"]),
            edge_aware=family == "edge_gnn",
        ).to(device)
        n_parameters = count_parameters(model)
        model, best_epoch, elapsed = _train_gnn(
            model, loaders["train"], loaders["validation"], device=device, options=config["training"]
        )
        predictions = np.empty(len(work), dtype=float)
        for partition in PARTITIONS:
            mask = work["partition"].eq(partition).to_numpy()
            _, scaled_prediction = _predict(model, loaders[partition], device)
            predictions[mask] = target_scaler.inverse_transform(scaled_prediction)
    else:
        raise ValueError(f"Unsupported model family {family!r}")

    calibration_mask = work["partition"].eq("calibration").to_numpy()
    calibration_errors = np.abs(
        work.loc[calibration_mask, "target"].to_numpy() - predictions[calibration_mask]
    )
    alpha = float(config["training"]["conformal_alpha"])
    quantile_level = min(
        1.0,
        np.ceil((len(calibration_errors) + 1) * (1 - alpha)) / len(calibration_errors),
    )
    conformal_radius = float(np.quantile(calibration_errors, quantile_level, method="higher"))

    metadata = {
        "dataset": dataset_name,
        "property": DATASET_REGISTRY[dataset_name]["property"],
        "units": DATASET_REGISTRY[dataset_name]["units"],
        "split_strategy": split_strategy,
        "seed": seed,
        "model": model_spec["name"],
        "family": family,
        "features": feature_set,
        "fusion": model_spec.get("fusion", "none"),
        "best_epoch": best_epoch,
        "train_seconds": elapsed,
        "n_parameters": n_parameters,
        "git_commit": _git_commit(),
        "conformal_alpha": alpha,
        "conformal_radius": conformal_radius,
    }
    result_rows = _result_rows(
        work, predictions, metadata=metadata, conformal_radius=conformal_radius
    )
    prediction_frame = work[
        ["dataset", "molecule_id", "smiles", "property", "units", "partition", "scaffold", "target"]
    ].copy()
    prediction_frame = prediction_frame.rename(columns={"target": "y_true"})
    prediction_frame["y_pred"] = predictions
    prediction_frame["prediction_lower"] = predictions - conformal_radius
    prediction_frame["prediction_upper"] = predictions + conformal_radius
    for key in ("split_strategy", "seed", "model", "family", "features", "fusion", "git_commit"):
        prediction_frame[key] = metadata[key]
    prediction_frame.attrs["feature_names"] = transformer.feature_names_
    return result_rows, prediction_frame


def run_benchmark(config: dict[str, Any], *, dry_run: bool = False) -> pd.DataFrame:
    plan = describe_plan(config)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return pd.DataFrame()

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "software_manifest.json").write_text(
        json.dumps(software_manifest(), indent=2), encoding="utf-8"
    )
    all_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for dataset_name in config["datasets"]:
        frame, audit = load_moleculenet(dataset_name, config["data_root"])
        audits.append(audit_as_dict(audit))
        pd.DataFrame(audits).to_csv(output_root / "dataset_audit.csv", index=False)
        for split_strategy in config["split"]["strategies"]:
            for seed in config["split"]["seeds"]:
                for model_spec in config["models"]:
                    rows, predictions = run_one(
                        frame,
                        dataset_name=dataset_name,
                        model_spec=model_spec,
                        split_strategy=split_strategy,
                        seed=int(seed),
                        config=config,
                    )
                    all_rows.extend(rows)
                    run_dir = (
                        output_root
                        / dataset_name
                        / split_strategy
                        / f"seed_{seed}"
                        / model_spec["name"]
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    predictions.to_csv(run_dir / "predictions.csv", index=False)
                    (run_dir / "feature_names.txt").write_text(
                        "\n".join(predictions.attrs.get("feature_names", [])), encoding="utf-8"
                    )
                    (run_dir / "run_metadata.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )
                    pd.DataFrame(all_rows).to_csv(output_root / "metrics.csv", index=False)
    return pd.DataFrame(all_rows)
