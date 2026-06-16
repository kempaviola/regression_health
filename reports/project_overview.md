# Project overview

A single at-a-glance record of what this project is and which choices were fixed,
following the reusable template's project-overview format. It holds the whole
problem on one screen so the thread is not lost across the pipeline stages.

| Field | Entry |
| --- | --- |
| Project name | regression_health |
| Real-world problem | People with autoimmune conditions tend to manage flares reactively. An early signal of tomorrow's symptom severity would support a shift toward preventive self-management. |
| Research question | Can routinely self-tracked daily data predict next-day symptom severity better than assuming tomorrow equals today? |
| Working hypotheses | H1, lagged symptom and environmental features beat the naive persistence baseline. H2, recent symptom history predicts more strongly than weather. |
| Task type | Regression |
| Target variable | `target`, the next calendar day's mean symptom severity, derived from Flaredown's `trackable_value` on Symptom rows, ordinal on a 0 to 4 scale. |
| Grain | One row per (`user_id`, calendar day) in the daily panel. Modelling rows are the user-days that have both a recorded severity today and a known severity on the following calendar day. |
| Data source and licence | Flaredown Autoimmune Symptom Tracker, Kaggle. Provenance and download steps are in `references/data_source.md`. Use is governed by the dataset's Kaggle terms. |
| Independence structure | Both grouped and time-ordered. The same user appears on many days, and the days carry a calendar order. |
| Split strategy and reason | Temporal hold-out at 2019-01-01, train before and test on or after. Predicting the future for returning users is the production scenario, so a date cutoff simulates deployment honestly. |
| Cross-validation strategy and reason | `GroupKFold` on `user_id` inside the training set, so no user spans two folds and the validation score is not inflated by memorising individual patients. |
| Primary metric and baseline | RMSE on the 2019 hold-out, read against the naive persistence baseline of tomorrow equals today, with MAE reported alongside. |
| Known limitations and risks | See the limitations section below. |

## Limitations and stated tradeoffs

Following the template's rule that every tradeoff is recorded rather than hidden,
the main limits of this build are set out here.

**Users straddle the temporal split.** The data is both time-ordered and grouped,
but the train/test boundary honours only time, so a returning user can appear on
both sides of 2019-01-01. This is kept deliberately. Predicting next-day severity
for users already in the system is the real deployment scenario, so a temporal
cutoff is the honest simulation, and a fully grouped split would instead measure
generalisation to brand-new users, which is a different question. Within-training
cross-validation already uses `GroupKFold`, so the model-selection score is not
inflated by per-user memorisation. The straddle is noted so the test RMSE is read
as next-day performance for known users, not as cold-start performance.

**`trackable_name` is not yet standardised.** The free-text names carry
inconsistent casing and near-duplicates, for example "Vitamin D3" against
"Vitamin d". Standardising to a controlled vocabulary is deferred because the
current features use only per-type counts and severities, not the names
themselves. The item is recorded here rather than left as a silent omission, and
it is the natural first step before any per-name or per-condition cohort work.

**Weather is missing for most user-days.** The weather block is absent on roughly
two in five panel rows because only a minority of users log weather, and the five
weather columns go missing together. Median imputation fit on the training split
fills these, but the weak environment-only signal in H2 is partly a reflection of
this sparsity rather than purely of weather being uninformative.

**Lags follow recorded-day order, not the calendar.** Because check-ins are
sparse and irregular, `lag1` can reach back several calendar days rather than
exactly one. The target itself is strictly the next calendar day, so the features
and the target are on slightly different time bases. This is a tradeoff in favour
of keeping rows that would be dropped by a strict calendar-lag rule.

**The ordinal target is modelled as continuous.** Severity is an ordinal 0 to 4
scale, but it is treated as a real-valued regression target. RMSE and MAE are
therefore averages over a bounded ordinal range, and the predicted-versus-actual
plot shows the resulting banding.
