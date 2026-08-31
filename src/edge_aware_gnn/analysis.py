"""Applicability-domain analysis from saved per-molecule predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def nearest_training_tanimoto(
    frame: pd.DataFrame,
    *,
    radius: int = 2,
    bits: int = 2048,
) -> pd.Series:
    """Maximum Morgan Tanimoto similarity to this run's training molecules."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    from .features import _molecule

    required = {"molecule_id", "smiles", "partition"}
    if not required <= set(frame.columns):
        raise ValueError(f"Prediction data must contain {sorted(required)}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fingerprints = {
        row.molecule_id: generator.GetFingerprint(_molecule(row.smiles))
        for row in frame[["molecule_id", "smiles"]].drop_duplicates().itertuples(index=False)
    }
    train_ids = frame.loc[frame["partition"] == "train", "molecule_id"].drop_duplicates().tolist()
    if not train_ids:
        raise ValueError("No training molecules found")
    train_fingerprints = [fingerprints[value] for value in train_ids]
    similarities: dict[str, float] = {}
    for molecule_id in frame["molecule_id"].drop_duplicates():
        if molecule_id in train_ids:
            similarities[molecule_id] = np.nan
        else:
            values = DataStructs.BulkTanimotoSimilarity(fingerprints[molecule_id], train_fingerprints)
            similarities[molecule_id] = float(max(values))
    return frame["molecule_id"].map(similarities)


def analyze_prediction_tree(
    root: str | Path,
    output: str | Path,
    *,
    radius: int = 2,
    bits: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add applicability-domain distances to every saved model prediction."""
    paths = sorted(Path(root).rglob("predictions.csv"))
    if not paths:
        raise FileNotFoundError(f"No predictions.csv files found below {root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["prediction_file"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "split_strategy", "seed"]
    enriched = []
    for _, subset in combined.groupby(keys, sort=False):
        molecule_table = subset.drop_duplicates("molecule_id")[
            ["molecule_id", "smiles", "partition"]
        ].copy()
        molecule_table["nearest_train_tanimoto"] = nearest_training_tanimoto(
            molecule_table, radius=radius, bits=bits
        )
        enriched.append(
            subset.merge(
                molecule_table[["molecule_id", "nearest_train_tanimoto"]],
                on="molecule_id",
                how="left",
                validate="many_to_one",
            )
        )
    result = pd.concat(enriched, ignore_index=True)
    result["absolute_error"] = np.abs(result["y_true"] - result["y_pred"])
    result["covered"] = (
        (result["y_true"] >= result["prediction_lower"])
        & (result["y_true"] <= result["prediction_upper"])
    )
    result["similarity_bin"] = pd.cut(
        result["nearest_train_tanimoto"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.000001],
        labels=["(0.0,0.2]", "(0.2,0.4]", "(0.4,0.6]", "(0.6,0.8]", "(0.8,1.0]"],
        include_lowest=True,
    )
    test = result[result["partition"] == "test"].copy()
    summary = (
        test.groupby(
            ["dataset", "split_strategy", "model", "similarity_bin"],
            observed=True,
        )
        .agg(
            n=("molecule_id", "size"),
            mean_similarity=("nearest_train_tanimoto", "mean"),
            mae=("absolute_error", "mean"),
            rmse=("absolute_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            empirical_coverage=("covered", "mean"),
        )
        .reset_index()
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path / "predictions_with_applicability_domain.csv", index=False)
    summary.to_csv(output_path / "applicability_domain_summary.csv", index=False)
    return result, summary

