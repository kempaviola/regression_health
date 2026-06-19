# Data dictionary, Flaredown export

**Source:** Flaredown Autoimmune Symptom Tracker, [Kaggle](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker?resource=download) (see `references/data_source.md`)
**Source file:** `export.csv` (686 MB, 7,976,223 rows, 42,283 users)
**Format:** long "tidy event", one row per `(user, date, trackable)` check-in entry.
**Grain:** a single tracked item logged by one user on one day.
**Coverage:** 2012-05-18 → 2019-12-06 (68 active months; volume concentrates 2017–2019).

## Raw columns

| Column | Type | Description | Notes |
|---|---|---|---|
| `user_id` | string (hashed) | Anonymous user identifier | 0% null; FK / grouping key. 42,283 distinct. |
| `age` | int (dirty) | Self-reported age | 3.9% null; **contains garbage** (min −196,691, max 2,018: birth years and typos). Clip to [5,120]. |
| `sex` | category | female / male / other / doesnt_say | 1.7% null; 81% female (autoimmune skew). |
| `country` | category | ISO-2 country code | 3.7% null; 164 values; US 59%, GB 15%, AU/CA next. |
| `checkin_date` | date | Day of the check-in | 0% null; no future dates. |
| `trackable_id` | int | Internal id of the tracked item | 0% null; not used for modeling. |
| `trackable_type` | category | Kind of entry | Symptom, Weather, Condition, Treatment, Food, Tag, HBI. |
| `trackable_name` | string | Name of the tracked item | 0% null; free-text, high cardinality (8–9k distinct per type). |
| `trackable_value` | mixed | Value of the entry; **meaning depends on `trackable_type`** | 11.6% null (Tag/Food carry no value). |

## What `trackable_value` means per `trackable_type`

| type | rows | value meaning | range | notes |
|---|---|---|---|---|
| **Symptom** | 3,642,279 | severity **0–4** | 0–4 clean | **regression target source.** mean 1.44, median 1. |
| Weather | 1,393,806 | reading; `trackable_name` ∈ {temperature_max/min, humidity, pressure, precip_intensity, icon} | varies | long-format; `icon` is text. Pivot to wide before use. |
| Condition | 1,111,517 | condition severity 0–4 | 0–4 | mean 1.70. |
| Treatment | 901,820 | dose | mostly **text** ("200mg"); 90% non-numeric | count of treatments/day is the usable signal. |
| Food | 480,971 | n/a | null | name only; use daily count. |
| Tag | 445,669 | n/a | null | free-text lifestyle tags; use daily count. |
| HBI | 161 | Harvey-Bradshaw Index | 0–20 | negligible volume. |

## Known data-quality issues (see `reports/data_profile_report.md`)
1. `age` has impossible values (negatives, years > 120), 469 rows out of range → values outside [5,120] set to null in `build_panel.py`.
2. `trackable_value` is type-dependent and mixed numeric/text → never pool across types.
3. Weather is stored long with one variable per row → must pivot per `(user, date)`.
4. `trackable_name` is messy free-text (e.g. "Vitamin D3" vs "Vitamin d") → standardize before name-level analysis.
5. Sparse check-ins: users do not log every calendar day → "next-day" target is defined on the exact next calendar day and naturally drops gaps.
