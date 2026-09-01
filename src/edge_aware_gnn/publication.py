"""Deterministic manuscript tables and figures from completed benchmark outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t

from .statistics import (
    corrected_resampled_ttest,
    holm_adjust,
    mean_confidence_interval,
    specified_wilcoxon_holm,
)


METRICS = ("rmse", "mae", "r2")
LOWER_IS_BETTER = {"rmse": True, "mae": True, "r2": False}
REQUIRED_METRICS = {
    "dataset",
    "split_strategy",
    "seed",
    "model",
    "partition",
    "property",
    "units",
    "rmse",
    "mae",
    "r2",
    "empirical_coverage",
    "mean_interval_width",
}
REQUIRED_DOMAIN = {
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

MODEL_LABELS = {
    "mean_baseline": "Mean",
    "rf_rdkit": "RF–RDKit",
    "rf_rdkit_morgan": "RF–RDKit+Morgan",
    "xgb_rdkit_morgan": "XGB–RDKit+Morgan",
    "node_gnn": "Node GNN",
    "edge_gnn": "Edge GNN",
    "hybrid_concat": "Hybrid–concat",
    "hybrid_gated": "Hybrid–gated",
}

DATASET_LABELS = {
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipophilicity": "Lipophilicity",
}

# Okabe-Ito-derived, color-vision-deficiency-safe palette.
MODEL_COLORS = {
    "mean_baseline": "#999999",
    "rf_rdkit": "#E69F00",
    "rf_rdkit_morgan": "#F0E442",
    "xgb_rdkit_morgan": "#D55E00",
    "node_gnn": "#56B4E9",
    "edge_gnn": "#0072B2",
    "hybrid_concat": "#009E73",
    "hybrid_gated": "#CC79A7",
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    required = {
        "primary_metrics",
        "secondary_metrics",
        "confidence_level",
        "alpha",
        "train_fraction",
        "test_fraction",
        "target_coverage",
        "model_order",
        "contrasts",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Publication specification is missing keys: {missing}")
    metrics = [*spec["primary_metrics"], *spec["secondary_metrics"]]
    if set(metrics) != set(METRICS):
        raise ValueError(f"Publication metrics must be exactly {list(METRICS)}")
    contrast_ids = [item["id"] for item in spec["contrasts"]]
    if len(contrast_ids) != len(set(contrast_ids)):
        raise ValueError("Contrast identifiers must be unique")
    return spec


def _read_inputs(
    metrics_path: str | Path,
    audit_path: str | Path,
    domain_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    audit = pd.read_csv(audit_path)
    domain = pd.read_csv(domain_path)
    _require_columns(metrics, REQUIRED_METRICS, "Metrics table")
    _require_columns(domain, REQUIRED_DOMAIN, "Applicability-domain table")
    if "dataset" not in audit or "curated_rows" not in audit:
        raise ValueError("Dataset audit must contain dataset and curated_rows")
    test = metrics.loc[metrics["partition"] == "test"].copy()
    if test.empty:
        raise ValueError("Metrics table contains no test rows")
    key = ["dataset", "split_strategy", "seed", "model"]
    if test.duplicated(key).any():
        raise ValueError("Metrics table contains duplicate test run keys")
    numeric = [*METRICS, "empirical_coverage", "mean_interval_width"]
    if not np.isfinite(test[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Test metrics contain non-finite values")
    return test, audit, domain


def aggregate_performance(test: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """Summarize each dataset/split/model over aligned seed-level test metrics."""
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "split_strategy", "model"]
    for group_key, subset in test.groupby(keys, sort=True):
        row: dict[str, Any] = dict(zip(keys, group_key))
        row["n_seeds"] = int(subset["seed"].nunique())
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            ci_low, ci_high = mean_confidence_interval(values, confidence=confidence)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def best_models(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Return the best model for every dataset, split, and metric."""
    rows: list[dict[str, Any]] = []
    for (dataset, strategy), subset in aggregate.groupby(["dataset", "split_strategy"]):
        for metric in METRICS:
            column = f"{metric}_mean"
            index = subset[column].idxmin() if LOWER_IS_BETTER[metric] else subset[column].idxmax()
            winner = subset.loc[index]
            rows.append(
                {
                    "dataset": dataset,
                    "split_strategy": strategy,
                    "metric": metric,
                    "model": winner["model"],
                    "mean": float(winner[column]),
                    "ci_low": float(winner[f"{metric}_ci_low"]),
                    "ci_high": float(winner[f"{metric}_ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def scaffold_generalization_gap(test: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """Calculate paired scaffold-minus-random changes for every model and metric."""
    rows: list[dict[str, Any]] = []
    for (dataset, model), subset in test.groupby(["dataset", "model"]):
        for metric in METRICS:
            wide = subset.pivot_table(index="seed", columns="split_strategy", values=metric)
            if not {"random", "scaffold"} <= set(wide.columns):
                continue
            paired = wide[["random", "scaffold"]].dropna()
            delta = (paired["scaffold"] - paired["random"]).to_numpy(dtype=float)
            ci_low, ci_high = mean_confidence_interval(delta, confidence=confidence)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "n_pairs": len(delta),
                    "random_mean": float(paired["random"].mean()),
                    "scaffold_mean": float(paired["scaffold"].mean()),
                    "mean_gap": float(delta.mean()),
                    "gap_std": float(delta.std(ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def specified_contrasts(test: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Evaluate the configured model contrasts within each dataset/split/metric family."""
    rows: list[pd.DataFrame] = []
    confidence = float(spec["confidence_level"])
    alpha = float(spec["alpha"])
    train_fraction = float(spec["train_fraction"])
    test_fraction = float(spec["test_fraction"])
    metrics = [*spec["primary_metrics"], *spec["secondary_metrics"]]
    for (dataset, strategy), subset in test.groupby(["dataset", "split_strategy"]):
        for metric in metrics:
            result = specified_wilcoxon_holm(
                subset,
                spec["contrasts"],
                value_col=metric,
                pair_cols=("seed",),
                confidence=confidence,
            )
            corrected_p: list[float] = []
            corrected_low: list[float] = []
            corrected_high: list[float] = []
            for contrast in result.itertuples(index=False):
                wide = subset.pivot_table(index="seed", columns="model", values=metric)
                delta = (wide[contrast.model_a] - wide[contrast.model_b]).dropna().to_numpy()
                corrected = corrected_resampled_ttest(
                    delta,
                    train_fraction=train_fraction,
                    test_fraction=test_fraction,
                )
                critical = float(
                    student_t.ppf((1.0 + confidence) / 2.0, corrected["degrees_of_freedom"])
                )
                corrected_p.append(corrected["p_value"])
                corrected_low.append(corrected["mean_delta"] - critical * corrected["standard_error"])
                corrected_high.append(corrected["mean_delta"] + critical * corrected["standard_error"])
            result["p_corrected_t"] = corrected_p
            result["p_corrected_t_holm"] = holm_adjust(corrected_p)
            result["corrected_ci_low"] = corrected_low
            result["corrected_ci_high"] = corrected_high
            role = "primary" if metric in spec["primary_metrics"] else "secondary"
            result.insert(0, "metric_role", role)
            result.insert(0, "metric", metric)
            result.insert(0, "family_id", f"{dataset}:{strategy}:{metric}")
            result.insert(0, "split_strategy", strategy)
            result.insert(0, "dataset", dataset)
            result["favors"] = [
                _favored_model(metric, row.model_a, row.model_b, row.mean_delta)
                for row in result.itertuples(index=False)
            ]
            result["significant_holm"] = result["p_holm"] < alpha
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def paired_seed_differences(test: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Return the individual aligned-seed differences underlying each contrast."""
    rows: list[dict[str, Any]] = []
    metrics = [*spec["primary_metrics"], *spec["secondary_metrics"]]
    for (dataset, strategy), subset in test.groupby(["dataset", "split_strategy"]):
        for metric in metrics:
            wide = subset.pivot_table(index="seed", columns="model", values=metric)
            for contrast in spec["contrasts"]:
                paired = wide[[contrast["model_a"], contrast["model_b"]]].dropna()
                for seed, values in paired.iterrows():
                    rows.append(
                        {
                            "dataset": dataset,
                            "split_strategy": strategy,
                            "metric": metric,
                            "contrast": contrast["id"],
                            "seed": int(seed),
                            "model_a": contrast["model_a"],
                            "model_b": contrast["model_b"],
                            "value_a": float(values[contrast["model_a"]]),
                            "value_b": float(values[contrast["model_b"]]),
                            "difference_a_minus_b": float(
                                values[contrast["model_a"]] - values[contrast["model_b"]]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _favored_model(metric: str, model_a: str, model_b: str, delta: float) -> str:
    if np.isclose(delta, 0.0):
        return "tie"
    a_better = delta < 0 if LOWER_IS_BETTER[metric] else delta > 0
    return model_a if a_better else model_b


def interval_reliability(
    test: pd.DataFrame,
    *,
    confidence: float,
    target_coverage: float,
) -> pd.DataFrame:
    """Summarize conformal coverage and interval width over seeds."""
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "split_strategy", "model"]
    for group_key, subset in test.groupby(keys, sort=True):
        coverage = subset["empirical_coverage"].to_numpy(dtype=float)
        width = subset["mean_interval_width"].to_numpy(dtype=float)
        cov_low, cov_high = mean_confidence_interval(coverage, confidence=confidence)
        width_low, width_high = mean_confidence_interval(width, confidence=confidence)
        rows.append(
            {
                **dict(zip(keys, group_key)),
                "n_seeds": len(coverage),
                "target_coverage": target_coverage,
                "coverage_mean": float(coverage.mean()),
                "coverage_std": float(coverage.std(ddof=1)),
                "coverage_ci_low": cov_low,
                "coverage_ci_high": cov_high,
                "coverage_gap": float(coverage.mean() - target_coverage),
                "width_mean": float(width.mean()),
                "width_std": float(width.std(ddof=1)),
                "width_ci_low": width_low,
                "width_ci_high": width_high,
            }
        )
    return pd.DataFrame(rows)


def dataset_summary(test: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Join dataset curation counts to property and unit metadata."""
    metadata = test[["dataset", "property", "units"]].drop_duplicates()
    if metadata.duplicated("dataset").any():
        raise ValueError("Dataset has inconsistent property or unit metadata")
    return audit.merge(metadata, on="dataset", how="left", validate="one_to_one")


def feature_selection_tables(run_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize train-only feature retention from tracked feature-name files."""
    root = Path(run_root)
    paths = sorted(root.glob("*/*/seed_*/*/feature_names.txt"))
    records: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root)
        dataset, strategy, seed_name, model = relative.parts[:4]
        if model not in {"rf_rdkit", "rf_rdkit_morgan"}:
            continue
        features = [value.strip() for value in path.read_text(encoding="utf-8").splitlines() if value.strip()]
        rdkit_features = [value for value in features if not value.startswith("morgan_")]
        morgan_features = [value for value in features if value.startswith("morgan_")]
        seed = int(seed_name.removeprefix("seed_"))
        records.append(
            {
                "dataset": dataset,
                "split_strategy": strategy,
                "seed": seed,
                "model": model,
                "n_features": len(features),
                "n_rdkit_features": len(rdkit_features),
                "n_morgan_features": len(morgan_features),
            }
        )
        if model == "rf_rdkit":
            retained.extend(
                {
                    "dataset": dataset,
                    "split_strategy": strategy,
                    "seed": seed,
                    "feature": feature,
                }
                for feature in rdkit_features
            )
    runs = pd.DataFrame(records)
    if runs.empty:
        raise FileNotFoundError(f"No tree-model feature_names.txt files found below {root}")
    summary = (
        runs.groupby(["dataset", "split_strategy", "model"])[
            ["n_features", "n_rdkit_features", "n_morgan_features"]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(value).rstrip("_") if isinstance(value, tuple) else value
        for value in summary.columns
    ]

    retained_frame = pd.DataFrame(retained)
    run_counts = (
        runs.loc[runs["model"] == "rf_rdkit"]
        .groupby(["dataset", "split_strategy"])["seed"]
        .nunique()
        .rename("n_runs")
    )
    frequency = (
        retained_frame.groupby(["dataset", "split_strategy", "feature"])["seed"]
        .nunique()
        .rename("n_retained")
        .reset_index()
        .merge(run_counts.reset_index(), on=["dataset", "split_strategy"], validate="many_to_one")
    )
    frequency["retention_fraction"] = frequency["n_retained"] / frequency["n_runs"]
    return summary, frequency


def _latex_text(value: Any) -> str:
    text = str(value).replace("–", "--")
    for source, replacement in (("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
        text = text.replace(source, replacement)
    return text


def _write_main_latex(aggregate: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Dataset & Split & Model & RMSE & MAE & $R^2$ \\",
        r"\midrule",
    ]
    for row in aggregate.sort_values(["dataset", "split_strategy", "rmse_mean"]).itertuples():
        lines.append(
            f"{_latex_text(DATASET_LABELS.get(row.dataset, row.dataset))} & "
            f"{_latex_text(row.split_strategy.title())} & "
            f"{_latex_text(MODEL_LABELS.get(row.model, row.model))} & "
            f"{row.rmse_mean:.3f} $\\pm$ {row.rmse_std:.3f} & "
            f"{row.mae_mean:.3f} $\\pm$ {row.mae_std:.3f} & "
            f"{row.r2_mean:.3f} $\\pm$ {row.r2_std:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contrasts_latex(contrasts: pd.DataFrame, path: Path) -> None:
    primary = contrasts.loc[contrasts["metric"] == "rmse"].copy()
    lines = [
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Dataset & Split & Contrast & Favours & $\Delta$RMSE & 95\% CI & $p_{Holm}$ \\",
        r"\midrule",
    ]
    for row in primary.sort_values(["dataset", "split_strategy", "contrast"]).itertuples():
        interval = f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        lines.append(
            f"{_latex_text(DATASET_LABELS.get(row.dataset, row.dataset))} & "
            f"{_latex_text(row.split_strategy.title())} & {_latex_text(row.contrast)} & "
            f"{_latex_text(MODEL_LABELS.get(row.favors, row.favors))} & {row.mean_delta:.3f} & "
            f"{interval} & {row.p_holm:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_context() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Publication figures require matplotlib; install with "
            "python -m pip install -e '.[publication]'"
        ) from exc
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    return plt


def _save_figure(fig: Any, figures: Path, stem: str) -> None:
    from matplotlib.image import imread

    png_path = figures / f"{stem}.png"
    png_temporary = figures / f".{stem}.png.part"
    for attempt in range(2):
        with png_temporary.open("wb") as handle:
            fig.savefig(handle, format="png", bbox_inches="tight")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            imread(png_temporary)
        except Exception:
            if attempt == 1:
                raise
        else:
            png_temporary.replace(png_path)
            break

    pdf_path = figures / f"{stem}.pdf"
    pdf_temporary = figures / f".{stem}.pdf.part"
    with pdf_temporary.open("wb") as handle:
        fig.savefig(handle, format="pdf", bbox_inches="tight")
        handle.flush()
        os.fsync(handle.fileno())
    pdf_bytes = pdf_temporary.read_bytes()
    if not pdf_bytes.startswith(b"%PDF") or not pdf_bytes.rstrip().endswith(b"%%EOF"):
        raise OSError(f"Incomplete PDF figure generated: {pdf_temporary}")
    pdf_temporary.replace(pdf_path)


def _condition_axes(plt: Any, *, figsize: tuple[float, float] = (14, 8)) -> tuple[Any, Any]:
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    return fig, axes


def _plot_main_rmse(aggregate: pd.DataFrame, spec: dict[str, Any], figures: Path) -> None:
    plt = _plot_context()
    fig, axes = _condition_axes(plt)
    datasets = sorted(aggregate["dataset"].unique())
    plot_models = [model for model in spec["model_order"] if model != "mean_baseline"]
    for row_index, strategy in enumerate(("random", "scaffold")):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            subset = aggregate.loc[
                (aggregate["dataset"] == dataset)
                & (aggregate["split_strategy"] == strategy)
            ].set_index("model")
            subset = subset.loc[[model for model in plot_models if model in subset.index]]
            x = np.arange(len(subset))
            errors = np.vstack(
                [
                    subset["rmse_mean"] - subset["rmse_ci_low"],
                    subset["rmse_ci_high"] - subset["rmse_mean"],
                ]
            )
            axis.bar(
                x,
                subset["rmse_mean"],
                yerr=errors,
                capsize=2,
                color=[MODEL_COLORS[model] for model in subset.index],
                edgecolor="black",
                linewidth=0.4,
            )
            axis.set_xticks(x, [MODEL_LABELS[model] for model in subset.index], rotation=45, ha="right")
            axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {strategy}")
            axis.set_ylabel("Test RMSE")
            axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Molecular-property prediction across random and scaffold splits", y=1.01)
    fig.tight_layout()
    _save_figure(fig, figures, "main_rmse")
    plt.close(fig)


def _plot_scaffold_gap(gaps: pd.DataFrame, spec: dict[str, Any], figures: Path) -> None:
    plt = _plot_context()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), squeeze=False)
    rmse = gaps.loc[gaps["metric"] == "rmse"]
    plot_models = [model for model in spec["model_order"] if model != "mean_baseline"]
    for axis, dataset in zip(axes[0], sorted(rmse["dataset"].unique())):
        subset = rmse.loc[rmse["dataset"] == dataset].set_index("model")
        subset = subset.loc[[model for model in plot_models if model in subset.index]].iloc[::-1]
        y = np.arange(len(subset))
        errors = np.vstack(
            [subset["mean_gap"] - subset["ci_low"], subset["ci_high"] - subset["mean_gap"]]
        )
        axis.barh(
            y,
            subset["mean_gap"],
            xerr=errors,
            capsize=2,
            color=[MODEL_COLORS[model] for model in subset.index],
            edgecolor="black",
            linewidth=0.4,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, [MODEL_LABELS[model] for model in subset.index])
        axis.set_title(DATASET_LABELS.get(dataset, dataset))
        axis.set_xlabel("Scaffold − random RMSE")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Generalization penalty under scaffold shift", y=1.01)
    fig.tight_layout()
    _save_figure(fig, figures, "scaffold_generalization_gap")
    plt.close(fig)


def _plot_contrasts(
    contrasts: pd.DataFrame,
    seed_differences: pd.DataFrame,
    figures: Path,
) -> None:
    plt = _plot_context()
    fig, axes = _condition_axes(plt)
    primary = contrasts.loc[contrasts["metric"] == "rmse"]
    datasets = sorted(primary["dataset"].unique())
    for row_index, strategy in enumerate(("random", "scaffold")):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            subset = primary.loc[
                (primary["dataset"] == dataset)
                & (primary["split_strategy"] == strategy)
            ].iloc[::-1]
            y = np.arange(len(subset))
            for position, item in zip(y, subset.itertuples(index=False)):
                color = "#0072B2" if item.mean_delta < 0 else "#D55E00"
                seeds = seed_differences.loc[
                    (seed_differences["dataset"] == dataset)
                    & (seed_differences["split_strategy"] == strategy)
                    & (seed_differences["metric"] == "rmse")
                    & (seed_differences["contrast"] == item.contrast),
                    "difference_a_minus_b",
                ]
                offsets = np.linspace(-0.14, 0.14, len(seeds))
                axis.scatter(
                    seeds,
                    position + offsets,
                    s=9,
                    color="#666666",
                    alpha=0.45,
                    linewidth=0,
                    zorder=1,
                )
                axis.errorbar(
                    item.mean_delta,
                    position,
                    xerr=[[item.mean_delta - item.ci_low], [item.ci_high - item.mean_delta]],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    capsize=3,
                    zorder=2,
                )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_yticks(y, subset["contrast"])
            axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {strategy}")
            axis.set_xlabel("Paired RMSE difference (A − B)")
            axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Specified architecture and feature contrasts", y=1.01)
    fig.tight_layout()
    _save_figure(fig, figures, "specified_rmse_contrasts")
    plt.close(fig)


def _plot_applicability_domain(domain: pd.DataFrame, figures: Path) -> None:
    plt = _plot_context()
    fig, axes = _condition_axes(plt)
    datasets = sorted(domain["dataset"].unique())
    models = ["node_gnn", "edge_gnn", "hybrid_concat", "hybrid_gated"]
    for row_index, strategy in enumerate(("random", "scaffold")):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            subset = domain.loc[
                (domain["dataset"] == dataset)
                & (domain["split_strategy"] == strategy)
                & (domain["model"].isin(models))
            ]
            for model in models:
                model_rows = subset.loc[subset["model"] == model].sort_values("mean_similarity")
                if model_rows.empty:
                    continue
                axis.scatter(
                    model_rows["mean_similarity"],
                    model_rows["mae"],
                    s=np.clip(np.sqrt(model_rows["n"]) * 2.5, 8, 55),
                    color=MODEL_COLORS[model],
                    edgecolor="white",
                    linewidth=0.3,
                    label=MODEL_LABELS[model],
                    zorder=2,
                )
            axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {strategy}")
            axis.set_xlabel("Nearest-training Morgan Tanimoto")
            axis.set_ylabel("Test MAE")
            axis.set_xlim(0.1, 1.0)
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Applicability domain of graph architectures", y=1.08)
    fig.tight_layout()
    _save_figure(fig, figures, "applicability_domain")
    plt.close(fig)


def _plot_interval_reliability(reliability: pd.DataFrame, spec: dict[str, Any], figures: Path) -> None:
    plt = _plot_context()
    fig, axes = _condition_axes(plt)
    datasets = sorted(reliability["dataset"].unique())
    plot_models = [model for model in spec["model_order"] if model != "mean_baseline"]
    for row_index, strategy in enumerate(("random", "scaffold")):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            subset = reliability.loc[
                (reliability["dataset"] == dataset)
                & (reliability["split_strategy"] == strategy)
                & (reliability["model"].isin(plot_models))
            ]
            for item in subset.itertuples(index=False):
                axis.scatter(
                    item.width_mean,
                    item.coverage_mean,
                    s=35,
                    color=MODEL_COLORS[item.model],
                    edgecolor="black",
                    linewidth=0.4,
                    label=MODEL_LABELS[item.model],
                )
            axis.axhline(float(spec["target_coverage"]), color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {strategy}")
            axis.set_xlabel("Mean interval width")
            axis.set_ylabel("Empirical coverage")
            axis.set_ylim(0.85, 1.01)
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Prediction-interval efficiency and calibration", y=1.08)
    fig.tight_layout()
    _save_figure(fig, figures, "interval_reliability")
    plt.close(fig)


def _plot_feature_retention(frequency: pd.DataFrame, figures: Path) -> None:
    plt = _plot_context()
    pivot = frequency.pivot_table(
        index="feature",
        columns=["dataset", "split_strategy"],
        values="retention_fraction",
        fill_value=0.0,
    )
    columns = [
        (dataset, strategy)
        for dataset in ("esol", "freesolv", "lipophilicity")
        for strategy in ("random", "scaffold")
        if (dataset, strategy) in pivot.columns
    ]
    pivot = pivot.loc[:, columns]
    variability = pd.DataFrame(
        {
            "range": pivot.max(axis=1) - pivot.min(axis=1),
            "std": pivot.std(axis=1),
            "feature_name": pivot.index,
        },
        index=pivot.index,
    )
    top_features = variability.sort_values(
        ["range", "std", "feature_name"], ascending=[False, False, True]
    ).head(20).index
    matrix = pivot.loc[top_features].iloc[::-1]
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_yticks(np.arange(len(matrix)), matrix.index)
    axis.set_xticks(
        np.arange(len(matrix.columns)),
        [
            f"{DATASET_LABELS.get(dataset, dataset)}\n{strategy}"
            for dataset, strategy in matrix.columns
        ],
    )
    axis.set_title("Most variable train-only RDKit descriptor retention")
    axis.set_xlabel("Dataset and split strategy")
    axis.set_ylabel("Descriptor")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Fraction of seeds retaining descriptor")
    fig.tight_layout()
    _save_figure(fig, figures, "descriptor_retention_stability")
    plt.close(fig)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_collection(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def generate_publication_analysis(
    *,
    metrics_path: str | Path = "outputs/metrics.csv",
    audit_path: str | Path = "outputs/dataset_audit.csv",
    domain_path: str | Path = "analysis/applicability_domain_summary.csv",
    run_root: str | Path = "outputs",
    spec_path: str | Path = "configs/publication.yaml",
    output_path: str | Path = "publication_outputs",
    figures: bool = True,
) -> dict[str, Any]:
    """Generate all table- and figure-level manuscript evidence from tracked CSVs."""
    spec = _load_spec(spec_path)
    test, audit, domain = _read_inputs(metrics_path, audit_path, domain_path)
    output = Path(output_path)
    tables = output / "tables"
    figure_path = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    if figures:
        figure_path.mkdir(parents=True, exist_ok=True)

    feature_summary, feature_frequency = feature_selection_tables(run_root)

    products = {
        "dataset_summary": dataset_summary(test, audit),
        "main_performance": aggregate_performance(test, float(spec["confidence_level"])),
        "specified_contrasts": specified_contrasts(test, spec),
        "paired_seed_differences": paired_seed_differences(test, spec),
        "scaffold_generalization_gap": scaffold_generalization_gap(
            test, float(spec["confidence_level"])
        ),
        "interval_reliability": interval_reliability(
            test,
            confidence=float(spec["confidence_level"]),
            target_coverage=float(spec["target_coverage"]),
        ),
        "applicability_domain_summary": domain.sort_values(
            ["dataset", "split_strategy", "model", "mean_similarity"]
        ).reset_index(drop=True),
        "feature_engineering_summary": feature_summary,
        "descriptor_retention_frequency": feature_frequency,
    }
    products["best_models"] = best_models(products["main_performance"])
    for name, frame in products.items():
        frame.to_csv(tables / f"{name}.csv", index=False)

    _write_main_latex(products["main_performance"], tables / "main_performance.tex")
    _write_contrasts_latex(products["specified_contrasts"], tables / "specified_contrasts_rmse.tex")

    if figures:
        _plot_main_rmse(products["main_performance"], spec, figure_path)
        _plot_scaffold_gap(products["scaffold_generalization_gap"], spec, figure_path)
        _plot_contrasts(
            products["specified_contrasts"],
            products["paired_seed_differences"],
            figure_path,
        )
        _plot_applicability_domain(products["applicability_domain_summary"], figure_path)
        _plot_interval_reliability(products["interval_reliability"], spec, figure_path)
        _plot_feature_retention(products["descriptor_retention_frequency"], figure_path)
        for partial in figure_path.glob(".*.part"):
            partial.unlink()

    input_paths = [Path(metrics_path), Path(audit_path), Path(domain_path), Path(spec_path)]
    feature_paths = sorted(Path(run_root).glob("*/*/seed_*/rf_*/feature_names.txt"))
    generated = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.name != "analysis_manifest.json"
    )
    manifest = {
        "inputs": {
            **{str(path): _sha256(path) for path in input_paths},
            f"{run_root}/**/feature_names.txt": _sha256_collection(
                feature_paths, Path(run_root)
            ),
        },
        "test_rows": len(test),
        "datasets": sorted(test["dataset"].unique().tolist()),
        "split_strategies": sorted(test["split_strategy"].unique().tolist()),
        "models": list(spec["model_order"]),
        "seeds": sorted(int(value) for value in test["seed"].unique()),
        "figure_formats": ["pdf", "png"] if figures else [],
        "generated_files": [str(path.relative_to(output)) for path in generated],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
