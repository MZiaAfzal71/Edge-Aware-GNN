"""Command-line interface for planning, auditing, running, and summarizing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import describe_plan, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgegnn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="configs/benchmark.yaml")
        command.add_argument("--dataset", action="append", choices=["esol", "freesolv", "lipophilicity"])
        command.add_argument("--model", action="append")
        command.add_argument("--split", action="append", choices=["random", "scaffold"])
        command.add_argument("--seed", action="append", type=int)
    subparsers.choices["run"].add_argument("--dry-run", action="store_true")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", default="configs/benchmark.yaml")
    audit.add_argument("--dataset", action="append", choices=["esol", "freesolv", "lipophilicity"])

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--metrics", default="outputs/metrics.csv")
    summary.add_argument("--output", default="outputs/summary")

    domain = subparsers.add_parser("domain")
    domain.add_argument("--root", default="outputs")
    domain.add_argument("--output", default="outputs/analysis")
    domain.add_argument("--radius", type=int, default=2)
    domain.add_argument("--bits", type=int, default=2048)
    return parser


def _filter_config(config: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "dataset", None):
        config["datasets"] = args.dataset
    if getattr(args, "model", None):
        requested = set(args.model)
        config["models"] = [model for model in config["models"] if model["name"] in requested]
        missing = requested - {model["name"] for model in config["models"]}
        if missing:
            raise ValueError(f"Models not found in configuration: {sorted(missing)}")
    if getattr(args, "split", None):
        config["split"]["strategies"] = args.split
    if getattr(args, "seed", None):
        config["split"]["seeds"] = args.seed
    return config


def _summarize(metrics_path: str, output_path: str) -> None:
    from .statistics import pairwise_wilcoxon_holm

    frame = pd.read_csv(metrics_path)
    test = frame[frame["partition"] == "test"].copy()
    grouped = (
        test.groupby(["dataset", "split_strategy", "model"])[["rmse", "mae", "r2"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in grouped.columns]
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output / "aggregate_metrics.csv", index=False)

    comparisons = []
    for (dataset, strategy), subset in test.groupby(["dataset", "split_strategy"]):
        for metric in ("rmse", "mae", "r2"):
            result = pairwise_wilcoxon_holm(
                subset,
                value_col=metric,
                pair_cols=("seed",),
            )
            if not result.empty:
                result.insert(0, "metric", metric)
                result.insert(0, "split_strategy", strategy)
                result.insert(0, "dataset", dataset)
                comparisons.append(result)
    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(output / "pairwise_wilcoxon_holm.csv", index=False)
    print(grouped.to_string(index=False))


def main() -> None:
    args = _parser().parse_args()
    if args.command == "summarize":
        _summarize(args.metrics, args.output)
        return
    if args.command == "domain":
        from .analysis import analyze_prediction_tree

        _, summary = analyze_prediction_tree(
            args.root, args.output, radius=args.radius, bits=args.bits
        )
        print(summary.to_string(index=False))
        return

    config = _filter_config(load_config(args.config), args)
    if args.command == "plan":
        print(json.dumps(describe_plan(config), indent=2))
    elif args.command == "audit":
        from .data import audit_as_dict, load_moleculenet

        for dataset in config["datasets"]:
            _, audit = load_moleculenet(dataset, config["data_root"])
            print(json.dumps(audit_as_dict(audit), indent=2))
    elif args.command == "run":
        from .experiment import run_benchmark

        run_benchmark(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
