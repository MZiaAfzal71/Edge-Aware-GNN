# Repository Recovery and Publication Contract

## Current history

The publication-analysis commit `a41bb6ae0fa87f1d8019e95b96177c43ef120098` is already an
ancestor of `main`. The following commit, `18082fe88cc3a7442bc5a809834f3ca656bbb871`, contains only
three Ruff formatting fixes. The publication work must therefore remain in the normal history; it
must not be re-created, cherry-picked, or force-pushed onto `main`.

## Protected project structure

The repository is intentionally CLI-only. Source modules, tests, benchmark and publication
configuration, compact analysis tables, and completed per-run results are tracked. Notebooks,
`*.egg-info`, `publication_outputs/`, and the derived
`predictions_with_applicability_domain.csv` file are not tracked.

The completed benchmark contract contains:

- three datasets: ESOL, FreeSolv, and Lipophilicity;
- two split strategies: random and scaffold;
- ten seeds: 2026 through 2035;
- eight model conditions;
- 480 runs and 1,920 partition-level metric rows.

Publication tables and figures are generated from tracked inputs. They do not require model
retraining or a GPU.

## Validation before a merge

Run these commands from the repository root:

```bash
python -m pip install -e '.[publication,dev]'
edgegnn validate --root .
edgegnn publication --tables-only --output /tmp/edgegnn-publication
pytest -q
```

`edgegnn validate` fails if required project files disappear, generated files are tracked, the
configured experiment matrix and recorded results diverge, run artifacts are missing, metadata is
invalid, or the applicability-domain and dataset audit tables are incomplete. GitHub Actions runs
the same structural and publication checks on every pull request.

## Safe recovery procedure

If a local checkout becomes disorganized, preserve any uncommitted work first, then recover through
a branch and pull request:

```bash
git status
git switch main
git pull --ff-only origin main
git switch -c recovery/<short-description>
edgegnn validate --root .
```

Do not force-push `main` and do not copy generated publication output into the repository. Make the
smallest necessary repair on the recovery branch, rerun the validation commands, and merge only
after the quality workflow succeeds.
