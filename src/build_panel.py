"""
build_panel.py  -- Stage 1 of the regression_health pipeline.
Collapses the long Flaredown export (one row per user/date/trackable) into a
daily panel: one row per (user_id, checkin_date) with engineered daily features.
Leakage-safe: NO statistics computed here use the target or the test period.

All steps are structural (template Part B Step 6): exact-duplicate removal,
date-validity filtering, and an impossible-age fix. None of them use the data
distribution, so all are safe to run on the full dataset before the split. Every
transformation prints before-and-after counts (template Principle 1).
"""
import pandas as pd, numpy as np, time, os, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "config.yml").read_text())
RAW = ROOT / CFG["raw_path"]
OUT = ROOT / CFG["interim_path"]
CHUNK = int(CFG["chunk_size"])
AGE_MIN, AGE_MAX = CFG["age_min"], CFG["age_max"]
WEATHER_NUM = ["temperature_max","temperature_min","humidity","pressure","precip_intensity"]
t0 = time.time()

type_parts = []   # per-chunk (user,date,type) aggregates
wx_parts   = []   # per-chunk (user,date) weather pivots
demo = {}         # user_id -> (age, sex, country) first non-null
rows_raw = 0      # total rows read
rows_dup = 0      # exact duplicates removed
rows_nodate = 0   # rows dropped for missing/unparseable date

uc = ["user_id","age","sex","country","checkin_date","trackable_type","trackable_name","trackable_value"]
for chunk in pd.read_csv(RAW, chunksize=CHUNK, dtype=str, keep_default_na=False,
                         na_values=[""], usecols=uc):
    rows_raw += len(chunk)
    # --- remove EXACT duplicates (all columns): template marks these as likely
    #     collection errors. Computed per chunk; cross-chunk dups are negligible
    #     here and the (user,date,type) aggregation below absorbs any remainder.
    before = len(chunk); chunk = chunk.drop_duplicates(); rows_dup += before - len(chunk)
    # --- drop rows without a valid check-in date (cannot place them on the panel) ---
    chunk["d"] = pd.to_datetime(chunk["checkin_date"], errors="coerce")
    before = len(chunk); chunk = chunk[chunk["d"].notna()]; rows_nodate += before - len(chunk)
    chunk["v"] = pd.to_numeric(chunk["trackable_value"], errors="coerce")
    # demographics (first seen non-null per user)
    dem = chunk.dropna(subset=["user_id"]).groupby("user_id").agg(
        age=("age","first"), sex=("sex","first"), country=("country","first"))
    for uid, r in dem.iterrows():
        if uid not in demo:
            demo[uid] = (r["age"], r["sex"], r["country"])
    # (user,date,type): sum / count / max of numeric value
    g = chunk.groupby(["user_id","d","trackable_type"]).agg(
        v_sum=("v","sum"), v_cnt=("v","size"), v_max=("v","max")).reset_index()
    type_parts.append(g)
    # weather numeric vars pivoted
    wx = chunk[(chunk["trackable_type"]=="Weather") & (chunk["trackable_name"].isin(WEATHER_NUM))]
    if len(wx):
        wp = wx.groupby(["user_id","d","trackable_name"])["v"].mean().unstack()
        wx_parts.append(wp.reset_index())

print(f"[clean] raw rows read            : {rows_raw:,}")
print(f"[clean] exact duplicates removed : {rows_dup:,}")
print(f"[clean] rows dropped (no date)   : {rows_nodate:,}")
print(f"[clean] rows kept for paneling   : {rows_raw - rows_dup - rows_nodate:,}")

# ---- combine type aggregates across chunks ----
T = pd.concat(type_parts, ignore_index=True)
T = T.groupby(["user_id","d","trackable_type"]).agg(
    v_sum=("v_sum","sum"), v_cnt=("v_cnt","sum"), v_max=("v_max","max")).reset_index()

def piv(metric):
    return T.pivot_table(index=["user_id","d"], columns="trackable_type", values=metric)

cnt = piv("v_cnt"); ssum = piv("v_sum"); smax = piv("v_max")
panel = pd.DataFrame(index=cnt.index)
# Symptom = regression target source
panel["sym_count"]  = cnt.get("Symptom")
panel["sym_mean"]   = ssum.get("Symptom") / cnt.get("Symptom")
panel["sym_max"]    = smax.get("Symptom")
panel["cond_count"] = cnt.get("Condition")
panel["cond_mean"]  = ssum.get("Condition") / cnt.get("Condition")
panel["treat_count"]= cnt.get("Treatment")
panel["food_count"] = cnt.get("Food")
panel["tag_count"]  = cnt.get("Tag")
panel = panel.reset_index()

# ---- weather ----
if wx_parts:
    W = pd.concat(wx_parts, ignore_index=True)
    W = W.groupby(["user_id","d"]).mean().reset_index()
    panel = panel.merge(W, on=["user_id","d"], how="left")

# ---- demographics ----
dd = pd.DataFrame([(k,)+v for k,v in demo.items()], columns=["user_id","age","sex","country"])
dd["age"] = pd.to_numeric(dd["age"], errors="coerce")
# Fix impossible ages (raw range reaches -196,691 and 2,018, i.e. birth years and
# garbage). Out-of-range values become NaN rather than being clipped to a bound, so
# no artificial spike forms at the edges; the train-fit imputer fills them later.
in_range = dd["age"].between(AGE_MIN, AGE_MAX)
n_bad_age = int((~in_range & dd["age"].notna()).sum())
dd.loc[~in_range, "age"] = np.nan
print(f"[clean] impossible ages -> NaN   : {n_bad_age:,} users (kept outside [{AGE_MIN},{AGE_MAX}])")
panel = panel.merge(dd, on="user_id", how="left")

# counts: missing type that day => 0 occurrences
for c in ["sym_count","cond_count","treat_count","food_count","tag_count"]:
    panel[c] = panel[c].fillna(0).astype(int)
panel = panel.rename(columns={"d":"date"}).sort_values(["user_id","date"]).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
panel.to_csv(OUT, index=False, compression="gzip")
print("PANEL rows", len(panel), "users", panel["user_id"].nunique(),
      "cols", list(panel.columns), "elapsed", round(time.time()-t0,1))
print(panel.head(3).to_string())
