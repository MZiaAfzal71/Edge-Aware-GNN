from pathlib import Path

import pandas as pd
import yaml

from edge_aware_gnn.publication import generate_publication_analysis


MODELS = [
    "mean_baseline",
    "rf_rdkit",
    "rf_rdkit_morgan",
    "xgb_rdkit_morgan",
    "node_gnn",
    "edge_gnn",
    "hybrid_concat",
    "hybrid_gated",
]


def test_publication_tables_are_generated_from_compact_inputs(tmp_path: Path):
    rows = []
    for strategy_index, strategy in enumerate(("random", "scaffold")):
        for seed in (1, 2, 3):
            for model_index, model in enumerate(MODELS):
                rmse = 0.5 + model_index * 0.05 + strategy_index * 0.1 + seed * 0.001
                rows.append(
                    {
                        "dataset": "esol",
                        "property": "solubility",
                        "units": "logS",
                        "split_strategy": strategy,
                        "seed": seed,
                        "model": model,
                        "partition": "test",
                        "rmse": rmse,
                        "mae": rmse * 0.8,
                        "r2": 1.0 - rmse / 2.0,
                        "empirical_coverage": 0.9,
                        "mean_interval_width": 1.5 + model_index * 0.1,
                    }
                )
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics, index=False)

    audit = tmp_path / "audit.csv"
    pd.DataFrame(
        [{"dataset": "esol", "raw_rows": 10, "duplicate_rows": 0, "curated_rows": 10}]
    ).to_csv(audit, index=False)

    domain = tmp_path / "domain.csv"
    pd.DataFrame(
        [
            {
                "dataset": "esol",
                "split_strategy": "random",
                "model": "edge_gnn",
                "similarity_bin": "(0.2,0.4]",
                "n": 10,
                "mean_similarity": 0.3,
                "mae": 0.4,
                "rmse": 0.5,
                "empirical_coverage": 0.9,
            }
        ]
    ).to_csv(domain, index=False)

    spec = tmp_path / "publication.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "primary_metrics": ["rmse", "mae"],
                "secondary_metrics": ["r2"],
                "confidence_level": 0.95,
                "alpha": 0.05,
                "train_fraction": 0.7,
                "test_fraction": 0.1,
                "target_coverage": 0.9,
                "model_order": MODELS,
                "contrasts": [
                    {
                        "id": "edge_vs_node",
                        "model_a": "edge_gnn",
                        "model_b": "node_gnn",
                        "question": "edge test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run_root = tmp_path / "runs"
    for strategy in ("random", "scaffold"):
        for seed in (1, 2, 3):
            for model, names in (
                ("rf_rdkit", ["MolWt", "TPSA"]),
                ("rf_rdkit_morgan", ["MolWt", "TPSA", "morgan_1"]),
            ):
                path = run_root / "esol" / strategy / f"seed_{seed}" / model
                path.mkdir(parents=True, exist_ok=True)
                (path / "feature_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    output = tmp_path / "publication"
    manifest = generate_publication_analysis(
        metrics_path=metrics,
        audit_path=audit,
        domain_path=domain,
        run_root=run_root,
        spec_path=spec,
        output_path=output,
        figures=False,
    )
    assert manifest["test_rows"] == 48
    assert (output / "tables/main_performance.csv").is_file()
    assert (output / "tables/specified_contrasts.csv").is_file()
    assert (output / "tables/scaffold_generalization_gap.csv").is_file()
    assert (output / "tables/interval_reliability.csv").is_file()
    assert (output / "tables/feature_engineering_summary.csv").is_file()
    assert (output / "tables/descriptor_retention_frequency.csv").is_file()
    assert (output / "analysis_manifest.json").is_file()
