# Edge-Aware Molecular Property Benchmark

This repository evaluates whether explicit bond-aware message passing and global molecular features improve **out-of-distribution molecular property prediction** across the physical-chemistry regression suite in MoleculeNet:

| Dataset | Predicted property | Approximate size | Unit | Primary split |
|---|---|---:|---|---|
| ESOL | Aqueous solubility | 1,128 | log10(mol/L) | Scaffold |
| FreeSolv | Hydration free energy | 642 | kcal/mol | Random, with scaffold stress test |
| Lipophilicity | Octanol/water distribution coefficient | 4,200 | logD at pH 7.4 | Scaffold |

The scope is a hypothesis-driven comparison across related but distinct physicochemical endpoints. Dataset names and loaders follow the [PyTorch Geometric MoleculeNet collection](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MoleculeNet.html). See the [Research Roadmap](docs/RESEARCH_ROADMAP.md) for the hypotheses, analyses, and manuscript outputs.

## What the revised benchmark tests

The model matrix separates representation and fusion effects:

| Model | Local representation | Global representation | Purpose |
|---|---|---|---|
| Training mean | None | None | Non-learning reference |
| Random Forest | None | RDKit descriptors | Strong small-data baseline |
| Random Forest | None | RDKit + Morgan | Descriptor/fingerprint baseline |
| XGBoost | None | RDKit + Morgan | Strong nonlinear baseline |
| Node-only GIN | Atom graph | None | Control for the value of explicit bond attributes |
| Edge-aware GINE | Atom + bond graph | None | Tests edge-aware message passing |
| Hybrid GINE, concatenation | Atom + bond graph | RDKit | Simple fusion control |
| Hybrid GINE, gated | Atom + bond graph | RDKit | Tests adaptive descriptor fusion |

Feature engineering is fitted on training molecules only:

- categorical atom embeddings for atomic number, degree, charge, hydrogen count, hybridization, chirality, valence, radicals, aromaticity, and ring membership;
- one-hot bond type and stereochemistry plus conjugation and ring indicators;
- RDKit descriptor missingness filtering, median imputation, variance filtering, high-correlation pruning, standardization, and bounded outlier handling;
- optional 2,048-bit radius-2 Morgan fingerprints;
- target normalization learned from the training partition only.

Every run has disjoint train, validation, calibration, and test partitions. Validation controls early stopping, calibration residuals determine split-conformal interval width, and the test partition is touched only for final evaluation.

## Repository layout

```text
configs/benchmark.yaml       Full multi-dataset experiment matrix
src/edge_aware_gnn/          Reusable data, feature, model, metric, and runner code
tests/                       Fast correctness tests
docs/                        Manuscript-oriented research roadmap
outputs/                     Completed benchmark metrics, predictions, and provenance
analysis/                    Compact applicability-domain summary
configs/publication.yaml     Auditable publication-analysis specification
```

## Installation

Python 3.10 or 3.11 is recommended. Create an isolated environment and install the full benchmark stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[boosting,dev]'
```

For CUDA, install the appropriate PyTorch build for the machine first, then run the final command.

## Commands

Inspect the planned experiment count without downloading data or importing the deep-learning stack:

```bash
edgegnn plan --config configs/benchmark.yaml
```

Download and audit all three datasets:

```bash
edgegnn audit --config configs/benchmark.yaml
```

Run a small smoke experiment first:

```bash
edgegnn run --config configs/benchmark.yaml \
  --dataset esol \
  --model rf_rdkit \
  --split random \
  --seed 2026
```

Run the complete benchmark:

```bash
edgegnn run --config configs/benchmark.yaml
```

Runs resume by default: a complete run directory with finite predictions is skipped, while an
interrupted or invalid run is recomputed. This makes it safe to repeat the command after a Kaggle
session interruption. To intentionally recompute the selected experiment matrix, add `--restart`.
Other result rows are preserved when selected runs are restarted.

Aggregate test metrics and create Holm-adjusted paired comparisons:

```bash
edgegnn summarize \
  --metrics outputs/metrics.csv \
  --output outputs/summary
```

Quantify error and interval coverage by nearest-training-set chemical similarity:

```bash
edgegnn domain \
  --root outputs \
  --output outputs/analysis
```

Generate all manuscript tables and figures from the tracked benchmark results:

```bash
python -m pip install -e '.[publication]'
edgegnn publication
```

The command reads `outputs/metrics.csv`, `outputs/dataset_audit.csv`,
`analysis/applicability_domain_summary.csv`, tracked feature-name files, and
`configs/publication.yaml`. It writes regenerable
CSV and LaTeX tables, paired statistical contrasts, and PDF/PNG figures below
`publication_outputs/`. No model fitting or GPU is required. Use `--tables-only` when matplotlib is
not installed. See [Publication Analysis](docs/PUBLICATION_ANALYSIS.md) for the complete output
contract and statistical sign conventions.

Run correctness tests:

```bash
pytest -q
```

## Output contract

`outputs/metrics.csv` contains one row per dataset, model, split strategy, seed, and partition. It records RMSE, MAE, R2, empirical interval coverage, interval width, runtime, parameter count, and the source Git commit. The completed benchmark tables and per-run predictions are tracked so a clone can reproduce the publication analysis without retraining.

`outputs/software_manifest.json` records the Python, platform, package, and Git versions used by the run.

Each run also writes:

```text
outputs/<dataset>/<strategy>/seed_<seed>/<model>/
  predictions.csv       molecule ID, SMILES, split, truth, prediction, interval
  run_metadata.json     metrics and provenance for the run
  feature_names.txt     exact post-filter global feature set
```

This long-form prediction output enables paired bootstrap analysis, applicability-domain plots, error-versus-similarity analysis, and independent verification of the reported tables.

## Reproducibility rules

- Never choose hyperparameters or epochs using the calibration or test partitions.
- Compare models on the same split seed and molecule IDs.
- Report mean and standard deviation across the ten predeclared seeds; do not treat dependent CV folds as independent molecules.
- Apply Holm correction within each predeclared family of pairwise tests.
- Call multiple trained models an ensemble only after their predictions are actually averaged.
- Keep random-split results as a secondary in-distribution reference; make scaffold results primary for ESOL and Lipophilicity.

## Recommended paper direction

The defensible research story is: **when do explicit bond attributes and global physicochemical features improve molecular prediction under scaffold shift?** The paper should emphasize cross-property consistency, ablations, uncertainty, and applicability domain rather than claiming a universally superior architecture from ESOL alone. Exact hypotheses and required tables are specified in [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md).

## Citation

A citation will be added after the revised multi-property study is completed. Until then, cite the repository URL and the exact Git commit used for a run.
