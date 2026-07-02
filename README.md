# regression_health, predicting next-day symptom severity

**Data source.** Flaredown Autoimmune Symptom Tracker, [Kaggle](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker?resource=download). Provenance and download steps live in `references/data_source.md`.

The project predicts next-day symptom severity for people with autoimmune
conditions such as rheumatoid arthritis, lupus, and Crohn's, using data they
already self-track. The inputs are recent symptoms, treatments, food, tags, and
weather. The aim is to support a shift from reactive toward preventive
self-management. A one-screen summary of every fixed decision is in
`reports/project_overview.md`.

**Target.** `target` is the next calendar day's mean symptom severity, taken from
Flaredown's `trackable_value`, an ordinal 0-4 scale, used as a regression target.

**Hypotheses**
- **H1.** A model using lagged symptom, treatment, and environmental features
  predicts next-day severity better than a naive "tomorrow equals today" baseline.
- **H2.** Recent symptom activity is a stronger predictor than environmental
  triggers, so the disease's own momentum matters more than the weather.

## Result (this build)
| Model | Test RMSE (2019 hold-out) |
|---|---|
| Naive baseline (today's severity) | 0.538 |
| Linear regression | **0.471** (−12% vs naive), H1 supported |
| HistGradientBoosting | 0.468 |

H2 probe (grouped-CV RMSE), symptom history **0.48** against environment-only
**0.83**, so H2 is supported and history dominates weather.

## Project layout
```
regression_health/
├── export.csv                 # raw Flaredown export (686 MB, 7.98M rows)
├── config/config.yml          # paths, random seed, run parameters (no hardcoded paths)
├── data/
│   ├── raw/                   # immutable source
│   ├── interim/               # daily_panel.csv.gz  (one row per user-day)
│   └── processed/             # train.csv.gz, test.csv.gz (leakage-safe split)
├── src/
│   ├── profile_data.py        # chunked data profiler  -> reports/profile_result.json
│   ├── build_panel.py         # long -> daily user-day panel (structural cleaning)
│   └── make_features.py       # target, split, CV, models, metrics
├── reports/
│   ├── project_overview.md    # one-screen project record + limitations
│   ├── data_profile_report.md # data exploration deliverable
│   ├── profile_result.json    # machine-readable profile
│   ├── model_metrics.json     # H1/H2 results
│   └── figures/               # charts, including residual diagnostics
├── references/data_dictionary.md
├── requirements.txt
└── notebooks/                 # interactive exploratory analysis
```

## How to run
Every path is read from `config/config.yml` and resolved relative to the
repository root, so the scripts run from a fresh clone without edits.
```bash
pip install -r requirements.txt
python src/profile_data.py      # 1. profile the raw export (~20s) -> reports/profile_result.json
python src/build_panel.py       # 2. structural cleaning + daily panel (~20s)
python src/make_features.py     # 3. target, temporal split, grouped CV, models, metrics (~10s)
jupyter lab notebooks/Viola_regression_capstone.ipynb   # full capstone analysis
```
Each script prints before-and-after counts for every transformation, so the run
is auditable end to end.

## Version control (git + DVC)
Run once on your machine.
```bash
bash init_repo.sh               # cleans any sandbox-built .git, inits git + DVC
```
The raw `export.csv` (686 MB) is tracked by **DVC**, not git, so git stores only
the pointer `export.csv.dvc` (md5 `0a99224b…`). Point the `storage` remote at
real storage and use `dvc push` and `dvc pull` for the bytes. Details are in
`references/data_source.md`.

## Methodology notes
- **Structural before statistical.** Duplicate removal, date validation, and the
  impossible-age fix happen in `build_panel.py` before the split. Imputation and
  scaling are fit on the training split only and applied to test through an
  sklearn `Pipeline`, so no test information leaks into training.
- **Temporal split** (train before 2019-01-01, test on or after) gives an honest
  simulation of predicting the future and respects the time-ordered data.
- **Grouped cross-validation** (`GroupKFold` on `user_id`) keeps any one user out
  of both train and validation folds, so the score is not inflated by memorising
  individual patients.
- **Naive baseline** is reported beside every model, so a "good RMSE" is always
  judged against "tomorrow equals today", which is what makes H1 testable.
- **Limitations** including the temporal-versus-grouped straddle and the deferred
  `trackable_name` standardisation are recorded in `reports/project_overview.md`.

## Submission

This project is submitted as a single compressed folder, `regression-project-viola.zip`.
The capstone notebook is `notebooks/Viola_regression_capstone.ipynb`, which holds the
problem framing, data understanding, modelling, evaluation, diagnostics, the
change-based analysis in Section 10, and the stakeholder impact report (Milestone 7)
in Section 11. The impact report is also kept as a standalone copy in
`notebooks/viola_impact_report.ipynb`.

The zip excludes the virtual environment (`.venv/`) and the 686 MB raw `export.csv`, which
is not redistributed; the derived `data/processed/` and `data/interim/` files are included
so both notebooks run, and `references/data_source.md` plus the in-notebook access section
explain how to fetch the raw file. To rebuild the zip from the parent directory:

```bash
zip -r -y regression-project-viola.zip regression_health \
  -x '*/.venv/*' '*/export.csv' '*/data/raw/*' '*/.git/*' \
     '*__pycache__*' '*.ipynb_checkpoints*' '*/.dvc/cache/*' '*/.dvc/tmp/*'
```
