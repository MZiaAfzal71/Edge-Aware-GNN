# Research Roadmap: From ESOL Model Study to Multi-Property Evidence

## Proposed article question

**When do explicit bond attributes and global physicochemical descriptors improve molecular property prediction under scaffold shift?**

This question is stronger than “can a GNN predict solubility?” because it makes the contribution falsifiable across dataset size, property type, representation, and distribution shift.

Suggested working title:

> Edge-Aware Message Passing and Descriptor Fusion under Chemical Scaffold Shift: A Multi-Property Benchmark

## Primary hypotheses

1. Edge-aware GINE will outperform a parameter-matched node-only GIN more consistently under scaffold splitting than under random splitting.
2. Train-only RDKit descriptor fusion will provide its largest benefit on ESOL and FreeSolv, where data are scarce.
3. Gated fusion will improve or match simple concatenation only when the learned gate uses both modalities; gate saturation must therefore be measured.
4. Strong RDKit/Morgan tree baselines will remain competitive in-distribution, while graph/global hybrids will show smaller scaffold generalization gaps.
5. Validation-calibrated prediction intervals will under-cover more severely on scaffold tests than on random tests, quantifying reliability loss under chemical shift.

These hypotheses should be frozen before full training. If results contradict them, report the contradiction rather than changing the comparison after inspection.

## Dataset scope

The primary suite contains ESOL, FreeSolv, and Lipophilicity. They are all single-target physical-chemistry regression tasks from MoleculeNet, but represent solubility, hydration thermodynamics, and partitioning. This keeps the statistical design comparable while extending the chemical scope.

An optional second-stage external validation may add AqSolDB only after deduplication, unit harmonization, source tracking, and removal of overlap with ESOL. It should not be mixed into the primary MoleculeNet analysis without this curation.

## Required model matrix

### Primary baselines

- Mean predictor fitted on training targets.
- Random Forest with RDKit descriptors.
- Random Forest and XGBoost with RDKit plus Morgan fingerprints.
- Chemprop D-MPNN with fixed, documented settings.

### Architecture ablations

- Node-only GIN.
- Edge-aware GINE.
- GINE plus descriptors by concatenation.
- GINE plus descriptors by gated fusion.
- Edge-feature shuffle negative control: retain topology but permute bond attributes within each training graph.
- Descriptor-label shuffle negative control for the fusion branch.

The last two controls distinguish architectural benefit from extra parameters or unintended information pathways.

## Feature-engineering study

The main analysis should use the implemented categorical atom embeddings, explicit bond/stereo vector, and train-only RDKit pipeline. A compact secondary ablation can compare:

1. original raw numeric atom features versus categorical embeddings;
2. bond type only versus the full bond vector;
3. RDKit descriptors before versus after missingness/correlation pruning;
4. RDKit alone, Morgan alone, and RDKit plus Morgan for tree models;
5. full descriptor fusion versus a small interpretable physicochemical panel (molecular weight, logP, TPSA, H-bond donors/acceptors, rings, rotatable bonds, fraction Csp3).

Do not tune every feature family separately on the test set. Use the validation partition or nested training folds.

## Evaluation design

- Use the same ten predeclared seeds for every model. Five pairs cannot yield a two-sided exact Wilcoxon p-value below 0.05, so five seeds are inadequate for the planned paired test.
- Generate 70/10/10/10 train/validation/calibration/test partitions without target information.
- Use both random and Bemis-Murcko scaffold splits.
- Fit target scaling, descriptor imputation, pruning, and standardization on training molecules only.
- Use validation loss only for early stopping and the separate calibration residuals only for split-conformal intervals.
- Report the untouched test partition once per seed.
- Save molecule IDs, canonical SMILES, scaffolds, predictions, and intervals for every run.

For expensive Chemprop experiments, use the identical saved partitions instead of its independent default splitter.

## Statistical analysis

Primary endpoints are test RMSE and MAE; R2 is secondary. For each dataset and split strategy:

- report mean, standard deviation, and a seed-level confidence interval;
- compare predeclared model pairs using aligned seeds;
- apply Holm correction within each dataset/metric/split family;
- report effect size and confidence interval, not p-value alone;
- on an identical fixed test set, bootstrap paired per-molecule error differences;
- for repeated resampling estimates, use the Nadeau-Bengio corrected test and acknowledge residual dependence.

Avoid a global significance test that pools RMSE values across properties with different units.

## Applicability-domain and reliability analyses

For every test molecule, calculate maximum Morgan Tanimoto similarity to the training set. Plot absolute error and interval coverage against similarity bins. Also report:

- scaffold-seen versus scaffold-unseen status where meaningful;
- error by molecular-weight and heteroatom-count quartile;
- conformal empirical coverage and mean interval width;
- performance after excluding the lowest-similarity decile as a sensitivity analysis;
- gate-value distributions for the gated model, by dataset and split.

These analyses can produce a genuine chemical interpretation: not only which model wins, but where and why it fails.

## Minimum tables and figures for a new manuscript

1. Dataset curation, property, unit, size, and split statistics.
2. Main test performance across three datasets and two split regimes.
3. Architecture and feature ablation table.
4. Corrected pairwise comparisons with effect sizes.
5. Random-to-scaffold generalization-gap figure.
6. Error versus nearest-training similarity figure.
7. Prediction-interval coverage and width figure.
8. Gated-fusion behavior or descriptor-importance figure.

The old Wilcoxon heat map should be replaced by a compact effect-size plot or a table of predeclared comparisons; a dense all-pairs heat map encourages over-interpretation.

## Completion gates

The article should not be rewritten as final until all of the following are true:

- smoke runs pass on ESOL;
- dataset audits and duplicate handling are reviewed;
- every model writes test predictions for identical split IDs;
- metric tests pass and no historical result is mixed into new tables;
- all ten predeclared seeds complete for all primary models;
- failed runs and excluded molecules are accounted for;
- the main conclusions hold on at least two of the three datasets, or are reframed as property-dependent findings.
