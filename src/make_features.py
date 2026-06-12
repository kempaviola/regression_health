"""
make_features.py -- Stage 2 of the regression_health pipeline.
Builds the next-day symptom-severity regression dataset from the daily panel,
applies a LEAKAGE-SAFE temporal split + preprocessing (fit on train only),
runs grouped cross-validation, and tests hypothesis H1 against a naive baseline.
Follows the ToU docs: 'Common Mistakes to Avoid', 'Why Split Data Properly',
'Random vs Stratified vs Temporal Splitting', 'Cross-Validation Setup'.
"""
import pandas as pd, numpy as np, json, time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE="/sessions/dreamy-modest-euler/mnt/regression_health"
PANEL=f"{BASE}/data/interim/daily_panel.csv.gz"
SPLIT_DATE=pd.Timestamp("2019-01-01")   # temporal hold-out
t0=time.time()

p=pd.read_csv(PANEL, parse_dates=["date"]).sort_values(["user_id","date"]).reset_index(drop=True)

# --- lagged symptom history (within user, recorded-day order) : H1 features ---
g=p.groupby("user_id")["sym_mean"]
p["lag1_sym"]=g.shift(1); p["lag2_sym"]=g.shift(2); p["lag3_sym"]=g.shift(3)
p["roll3_sym"]=g.shift(1).rolling(3,min_periods=1).mean().reset_index(level=0,drop=True)
p["dow"]=p["date"].dt.dayofweek

# --- TARGET: next calendar day's mean symptom severity ---
nxt=p[["user_id","date","sym_mean"]].copy()
nxt["date"]=nxt["date"]-pd.Timedelta(days=1)
nxt=nxt.rename(columns={"sym_mean":"target"})
p=p.merge(nxt,on=["user_id","date"],how="left")

# keep rows with a known next-day target AND some symptom history today
data=p[p["target"].notna() & p["sym_mean"].notna()].copy()

FEATURES=["sym_mean","lag1_sym","lag2_sym","lag3_sym","roll3_sym",
          "sym_count","cond_mean","treat_count","food_count","tag_count",
          "temperature_max","temperature_min","humidity","pressure","precip_intensity",
          "age","dow"]

# --- TEMPORAL split FIRST (before any fitting) : ToU 'split first' rule ---
train=data[data["date"]<SPLIT_DATE].copy()
test =data[data["date"]>=SPLIT_DATE].copy()
Xtr,ytr=train[FEATURES],train["target"]; Xte,yte=test[FEATURES],test["target"]

# --- Naive baseline: 'tomorrow == today' (persistence) ---
base_rmse=np.sqrt(mean_squared_error(yte, test["sym_mean"]))
base_mae =mean_absolute_error(yte, test["sym_mean"])

# --- Leakage-safe pipeline: impute+scale fit on TRAIN ONLY ---
def pipe(model): return Pipeline([("imp",SimpleImputer(strategy="median")),
                                  ("sc",StandardScaler()),("m",model)])
lin=pipe(LinearRegression())

# --- Grouped CV on TRAIN (no user in two folds) : ToU Group CV ---
gkf=GroupKFold(n_splits=5)
cv=cross_val_score(lin,Xtr,ytr,cv=gkf,groups=train["user_id"],
                   scoring="neg_root_mean_squared_error")
cv_rmse=(-cv).mean(); cv_std=(-cv).std()

# --- Fit on full train, evaluate on temporal test ---
lin.fit(Xtr,ytr); lin_rmse=np.sqrt(mean_squared_error(yte,lin.predict(Xte)))
lin_mae=mean_absolute_error(yte,lin.predict(Xte))
hgb=HistGradientBoostingRegressor(random_state=42,max_iter=200)
hgb.fit(Xtr.fillna(Xtr.median()),ytr)
hgb_rmse=np.sqrt(mean_squared_error(yte,hgb.predict(Xte.fillna(Xtr.median()))))

# --- H2 probe: symptom-history block vs environment block (train-CV RMSE) ---
HIST=["sym_mean","lag1_sym","lag2_sym","lag3_sym","roll3_sym"]
ENV =["temperature_max","temperature_min","humidity","pressure","precip_intensity"]
def block_rmse(cols):
    return (-cross_val_score(pipe(LinearRegression()),train[cols],ytr,cv=gkf,
            groups=train["user_id"],scoring="neg_root_mean_squared_error")).mean()
hist_rmse=block_rmse(HIST); env_rmse=block_rmse(ENV)

# --- save processed splits + metrics ---
keep=["user_id","date","target"]+FEATURES
train[keep].to_csv(f"{BASE}/data/processed/train.csv.gz",index=False,compression="gzip")
test[keep].to_csv(f"{BASE}/data/processed/test.csv.gz",index=False,compression="gzip")
metrics={
 "n_modeling_rows":len(data),"n_train":len(train),"n_test":len(test),
 "n_users_train":int(train["user_id"].nunique()),"n_users_test":int(test["user_id"].nunique()),
 "train_period":[str(train["date"].min().date()),str(train["date"].max().date())],
 "test_period":[str(test["date"].min().date()),str(test["date"].max().date())],
 "target_mean_train":round(float(ytr.mean()),3),"target_mean_test":round(float(yte.mean()),3),
 "naive_baseline_rmse":round(base_rmse,4),"naive_baseline_mae":round(base_mae,4),
 "linreg_cv_rmse_grouped":round(cv_rmse,4),"linreg_cv_rmse_std":round(cv_std,4),
 "linreg_test_rmse":round(lin_rmse,4),"linreg_test_mae":round(lin_mae,4),
 "hgb_test_rmse":round(hgb_rmse,4),
 "H1_beats_naive":bool(lin_rmse<base_rmse),
 "H1_improvement_pct":round((base_rmse-lin_rmse)/base_rmse*100,2),
 "H2_history_only_cv_rmse":round(hist_rmse,4),"H2_env_only_cv_rmse":round(env_rmse,4),
 "H2_history_stronger_than_env":bool(hist_rmse<env_rmse),
 "elapsed_sec":round(time.time()-t0,1)}
json.dump(metrics,open(f"{BASE}/reports/model_metrics.json","w"),indent=2)
print(json.dumps(metrics,indent=2))
