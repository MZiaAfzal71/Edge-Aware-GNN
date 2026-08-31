# Validation Audit of the Original ESOL Study

This audit separates reusable work from results that must be regenerated. It applies to the historical notebooks in `Models/`, the CSV files in `Results/`, and the submitted ESOL manuscript.

## Findings that invalidate direct reuse of reported numbers

| Finding | Evidence in the historical implementation | Consequence | Revised rule |
|---|---|---|---|
| Test leakage in scaffold runs | Several notebooks select `split_col != "Train"`; Chemprop later concatenates validation and test targets | Test molecules influence early stopping and are reported as validation | Keep train, validation, calibration, and test separate; validation selects epochs, calibration sets intervals, and test reports |
| Reversed R2 arguments | Multiple calls use `r2_score(prediction, truth)` | Reported R2 is not the standard coefficient of determination | Always call `r2_score(y_true, y_pred)` through one tested metric function |
| Global target preprocessing | GNN notebooks calculate target mean and standard deviation before splitting | Holdout target distribution informs training transformation | Fit target scaling on training targets only |
| “Ensemble” is not an ensemble | Ten runs are scored separately; predictions are not averaged | The manuscript overstates ensemble evaluation | Use “independent seeded runs,” or save and average predictions before ensemble scoring |
| Unadjusted multiplicity | Pairwise Wilcoxon p-values are plotted directly | The text calls values adjusted without applying a correction | Predeclare families and apply Holm correction |
| Dependent resampling | Repeated K-fold scores are treated as ordinary independent pairs | Nominal p-values can be anti-conservative | Use aligned split-level comparisons and a corrected resampled test; use paired bootstrap on a fixed test set |
| Inconsistent model-selection design | Tree hyperparameters are tuned on the same repeated CV later summarized as performance | Selection can bias reported CV performance | Use nested tuning or freeze hyperparameters before a separate repeated evaluation |
| Duplicate molecules | Canonicalized ESOL contains repeated structures | A structure can cross a random split | Canonicalize and collapse duplicates before splitting |
| Notebook-only execution | Environment, configuration, predictions, and commit identity are not recorded together | Results are difficult to reproduce or audit | Use one configuration-driven CLI and write per-molecule predictions plus provenance |

## Work that remains scientifically useful

- The central comparison among edge-aware GNNs, descriptor fusion, tree ensembles, and Chemprop is worthwhile.
- Explicit bond type, conjugation, and ring information is chemically motivated.
- Descriptor fusion is a plausible small-data strategy.
- Random and scaffold evaluations answer different questions and should both remain.
- ESOL is useful as one benchmark, but not as the sole evidence for generality.

## Status of supplied figures

`gen_gap_cv_vs_scaffold.pdf` and `wilcoxon test.pdf` accurately visualize the historical CSV values, but those values inherit the design problems above. The figures should be treated as layout prototypes only. They must be regenerated from the revised `outputs/metrics.csv` and prediction files.

## Manuscript implication

The submitted manuscript should not receive a cosmetic revision around the old table. The title, abstract, methods, results, significance claims, figures, and conclusion all need to be rebuilt after the revised benchmark completes. The old ESOL-only text can supply background material, model motivation, and mathematical notation.
