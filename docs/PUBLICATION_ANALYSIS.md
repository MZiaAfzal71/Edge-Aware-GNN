# Reproducible Publication Analysis

The publication layer is generated entirely from completed benchmark outputs. It does not import
the training stack, refit a model, download a dataset, or require a GPU.

## One-command workflow

From a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[publication]'
edgegnn publication
```

For tables without figures:

```bash
edgegnn publication --tables-only
```

Custom locations remain explicit:

```bash
edgegnn publication \
  --metrics outputs/metrics.csv \
  --dataset-audit outputs/dataset_audit.csv \
  --domain-summary analysis/applicability_domain_summary.csv \
  --run-root outputs \
  --spec configs/publication.yaml \
  --output publication_outputs
```

## Inputs

| Input | Purpose |
|---|---|
| `outputs/metrics.csv` | Seed-level performance, interval, runtime, and provenance records |
| `outputs/dataset_audit.csv` | Dataset curation counts and target distributions |
| `analysis/applicability_domain_summary.csv` | Error and coverage by nearest-training similarity |
| `outputs/.../feature_names.txt` | Train-only descriptor and fingerprint retention by run |
| `configs/publication.yaml` | Metrics, confidence level, model order, and specified contrasts |

The command rejects missing columns, duplicate test-run keys, non-finite test metrics, inconsistent
dataset metadata, or a contrast whose models are unavailable. SHA-256 hashes of every input are
recorded in `analysis_manifest.json`.

## Generated tables

| File below `publication_outputs/tables/` | Content |
|---|---|
| `dataset_summary.csv` | Curation audit joined to property and unit metadata |
| `main_performance.csv` | Mean, standard deviation, and 95% seed-level t interval |
| `best_models.csv` | Best model for every dataset, split, and metric |
| `specified_contrasts.csv` | Aligned-seed effects and inference for the configured comparisons |
| `paired_seed_differences.csv` | Every seed-level difference underlying the specified contrasts |
| `scaffold_generalization_gap.csv` | Paired scaffold-minus-random change by seed |
| `interval_reliability.csv` | Conformal coverage, width, confidence intervals, and coverage gap |
| `applicability_domain_summary.csv` | Canonically sorted chemical-similarity analysis |
| `feature_engineering_summary.csv` | Retained RDKit and Morgan feature counts by condition |
| `descriptor_retention_frequency.csv` | RDKit descriptor retention stability across seeds |
| `main_performance.tex` | LaTeX performance table using `booktabs` |
| `specified_contrasts_rmse.tex` | LaTeX table for the primary RMSE contrasts |

## Generated figures

Every figure is written as a 300-dpi PNG and a vector PDF:

- `main_rmse`: performance across the three datasets and two split regimes;
- `scaffold_generalization_gap`: paired scaffold-minus-random RMSE change;
- `specified_rmse_contrasts`: effect estimates and seed-level confidence intervals;
- `applicability_domain`: graph-model MAE versus nearest-training Tanimoto similarity;
- `interval_reliability`: empirical coverage versus mean prediction-interval width.
- `descriptor_retention_stability`: train-only descriptor-filter stability across seeds.

The plotted palette is color-vision-deficiency safe. Properties with different physical units are
shown in separate panels and are never pooled into one numerical performance test. In the
applicability-domain figure, marker area reflects the number of test observations in the similarity
bin so sparse bins are visually de-emphasized.

## Statistical interpretation

The configured contrasts mirror the architecture and feature-ablation design. They are described as
**specified design contrasts**, not as prospectively preregistered tests.

Within each dataset, split strategy, and metric, the command:

1. aligns models by the same ten split seeds;
2. defines the effect as `model_a - model_b`;
3. reports mean and median paired effects, a 95% t interval, and matched-pairs rank-biserial effect;
4. performs a two-sided Wilcoxon signed-rank test;
5. applies Holm correction across the contrasts listed in `configs/publication.yaml`;
6. reports the Nadeau-Bengio corrected resampled t-test as a dependence-aware sensitivity analysis.

For RMSE and MAE, a negative difference favours `model_a`; for R2, a positive difference favours
`model_a`. The `favors` column applies this rule automatically. RMSE and MAE are the co-primary
endpoints in the frozen research roadmap; R2 is secondary. The original all-model pairwise comparison
remains available as exploratory output in `outputs/summary/pairwise_wilcoxon_holm.csv`.

These seed-level tests do not turn overlapping resamples into independent experiments. Conclusions
should emphasize effect magnitude, consistency across properties and split regimes, and confidence
intervals rather than a significance threshold alone.

## Applicability-domain regeneration

The compact tracked summary is sufficient for publication figure generation. To reconstruct it from
the 480 tracked per-run prediction files:

```bash
edgegnn domain --root outputs --output outputs/analysis
```

The combined per-molecule applicability-domain CSV is intentionally not required by
`edgegnn publication`.
