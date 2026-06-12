# regression_health — predicting next-day symptom severity

**Data source:** Flaredown Autoimmune Symptom Tracker — [Kaggle](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker?resource=download) (provenance & download in `references/data_source.md`).

Predicting **next-day symptom severity** for people with autoimmune conditions
(rheumatoid arthritis, lupus, Crohn's, etc.) from data they already self-track:
recent symptoms, treatments, food, tags, and weather. The aim is to support a
shift from reactive to **preventive** self-management.

**Target:** `target` = the next calendar day's mean symptom severity (Flaredown's
`trackable_value`, ordinal 0–4) → a legitimate regression target.

**Hypotheses**
- **H1** — a model using lagged symptom, treatment and environmental features
  predicts next-day severity **better than a naive "tomorrow = today" baseline.**
- **H2** — recent symptom activity is a **stronger** predictor than environmental
  triggers (weather): the disease's own momentum matters more than the weather.

## Result (this build)
| Model | Test RMSE (2019 hold-out) |
|---|---|
| Naive baseline (today's severity) | 0.538 |
| Linear regression | **0.471** (−12.4%) → **H1 supported** |
| HistGradientBoosting | 0.468 |

H2 probe (grouped-CV RMSE): symptom history **0.48** vs environment-only **0.83**
→ **H2 supported** — history dominates weather.

## Project layout (per ToU "How to Organize Locally")
```
regression_health/
├── export.csv                 # raw Flaredown export (686 MB, 7.98M rows)
├── data/
│   ├── raw/                   # immutable source
│   ├── interim/               # daily_panel.csv.gz  (one row per user-day)
│   └── processed/             # train.csv.gz, test.csv.gz (leakage-safe split)
├── src/
│   ├── profile_data.py        # chunked data profiler  → reports/profile_result.json
│   ├── build_panel.py         # long → daily user-day panel
│   └── make_features.py       # target, split, CV, models, metrics
├── reports/
│   ├── data_profile_report.md # the /explore-data deliverable
│   ├── profile_result.json    # machine-readable profile
│   ├── model_metrics.json     # H1/H2 results
│   └── figures/               # charts
├── references/data_dictionary.md
├── requirements.txt
└── notebooks/                 # for exploratory analysis
```

## How to run
```bash
pip install -r requirements.txt
python src/profile_data.py      # 1. profile the raw export (~30s)
python src/build_panel.py       # 2. build the daily panel  (~30s)
python src/make_features.py     # 3. target + split + CV + models (~10s)
```

## Methodology notes (ToU "Common Mistakes to Avoid")
- **Split before preprocessing.** Imputation + scaling are fit on the training
  split only and applied to test via an sklearn `Pipeline` — no leakage.
- **Temporal split** (train < 2019-01-01, test ≥ 2019-01-01): honest simulation
  of predicting the future, and it respects the time-ordered nature of the data.
- **Grouped cross-validation** (`GroupKFold` on `user_id`): no user appears in
  both train and validation folds, so the score isn't inflated by memorizing
  individual patients.
- **Naive baseline** is reported alongside every model so "good RMSE" is always
  judged against "tomorrow = today", which is what makes H1 testable.
