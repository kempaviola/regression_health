"""
build_panel.py  -- Stage 1 of the regression_health pipeline.
Collapses the long Flaredown export (one row per user/date/trackable) into a
daily panel: one row per (user_id, checkin_date) with engineered daily features.
Leakage-safe: NO statistics computed here use the target or the test period.
"""
import pandas as pd, numpy as np, time, os
RAW = "/sessions/dreamy-modest-euler/mnt/regression_health/export.csv"
OUT = "/sessions/dreamy-modest-euler/mnt/regression_health/data/interim/daily_panel.csv.gz"
CHUNK = 1_000_000
WEATHER_NUM = ["temperature_max","temperature_min","humidity","pressure","precip_intensity"]
t0 = time.time()

type_parts = []   # per-chunk (user,date,type) aggregates
wx_parts   = []   # per-chunk (user,date) weather pivots
demo = {}         # user_id -> (age, sex, country) first non-null

uc = ["user_id","age","sex","country","checkin_date","trackable_type","trackable_name","trackable_value"]
for chunk in pd.read_csv(RAW, chunksize=CHUNK, dtype=str, keep_default_na=False,
                         na_values=[""], usecols=uc):
    chunk["d"] = pd.to_datetime(chunk["checkin_date"], errors="coerce")
    chunk = chunk[chunk["d"].notna()]
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
