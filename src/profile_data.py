import pandas as pd, numpy as np, json, time, yaml
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/"config"/"config.yml").read_text())
PATH=ROOT/CFG["raw_path"]
OUT=ROOT/CFG["profile_json"]
TODAY=pd.Timestamp.now().normalize()   # flag any check-in dated after today
CHUNK=int(CFG["chunk_size"])
cols=["user_id","age","sex","country","checkin_date","trackable_id","trackable_type","trackable_name","trackable_value"]
t0=time.time()
total=0; nulls=Counter(); user_ids=set(); countries=set()
type_counts=Counter(); sex_counts=Counter(); country_counts=Counter()
names_by_type=defaultdict(Counter)
age_hist=Counter()
date_min=None; date_max=None; future=0; date_month=Counter()
sym_hist=Counter()                      # Symptom trackable_value histogram (target source)
# numeric tv aggregates per type
tv_agg=defaultdict(lambda:{"count":0,"min":np.inf,"max":-np.inf,"neg":0,"zero":0,"gt4":0,"sum":0.0,"hist":Counter()})
tv_nonnum=Counter()
for chunk in pd.read_csv(PATH,chunksize=CHUNK,dtype=str,keep_default_na=False,na_values=[""]):
    total+=len(chunk)
    for c in cols: nulls[c]+=int(chunk[c].isna().sum())
    user_ids.update(chunk["user_id"].dropna().unique())
    countries.update(chunk["country"].dropna().unique())
    sex_counts.update(chunk["sex"].dropna().value_counts().to_dict())
    country_counts.update(chunk["country"].dropna().value_counts().to_dict())
    type_counts.update(chunk["trackable_type"].dropna().value_counts().to_dict())
    ag=pd.to_numeric(chunk["age"],errors="coerce").dropna()
    age_hist.update(ag.astype(int).value_counts().to_dict())
    d=pd.to_datetime(chunk["checkin_date"],errors="coerce")
    if d.notna().any():
        dmin,dmax=d.min(),d.max()
        date_min=dmin if date_min is None else min(date_min,dmin)
        date_max=dmax if date_max is None else max(date_max,dmax)
        future+=int((d>TODAY).sum())
        date_month.update(d.dropna().dt.to_period("M").astype(str).value_counts().to_dict())
    # names per type (vectorized, capped)
    g=chunk.dropna(subset=["trackable_type"]).groupby("trackable_type")["trackable_name"]
    for t,vc in g.value_counts().groupby(level=0):
        c=names_by_type[t]
        if len(c)<8000:
            c.update(vc.droplevel(0).to_dict())
    # trackable_value
    tvn=pd.to_numeric(chunk["trackable_value"],errors="coerce")
    df=pd.DataFrame({"t":chunk["trackable_type"].fillna("<null>"),"raw":chunk["trackable_value"],"num":tvn})
    num=df[df["num"].notna()]
    for t,sub in num.groupby("t"):
        a=sub["num"].values; d2=tv_agg[t]
        d2["count"]+=len(a); d2["sum"]+=float(a.sum())
        d2["min"]=min(d2["min"],float(a.min())); d2["max"]=max(d2["max"],float(a.max()))
        d2["neg"]+=int((a<0).sum()); d2["zero"]+=int((a==0).sum()); d2["gt4"]+=int((a>4).sum())
        # integer-ish hist for low-range
        ai=a[(a>=-1)&(a<=10)]
        if len(ai): d2["hist"].update(pd.Series(ai).round(2).value_counts().to_dict())
    sym=num[num["t"]=="Symptom"]["num"]
    if len(sym): sym_hist.update(sym.round(2).value_counts().to_dict())
    nonnum=df[df["num"].isna() & df["raw"].notna()]
    tv_nonnum.update(nonnum["t"].value_counts().to_dict())

def pct_from_hist(hist):
    items=sorted(hist.items()); vals=np.array([k for k,_ in items],dtype=float); cnt=np.array([v for _,v in items],dtype=float)
    if cnt.sum()==0: return {}
    cum=np.cumsum(cnt); tot=cum[-1]
    def q(p):
        i=np.searchsorted(cum,p*tot); i=min(i,len(vals)-1); return float(vals[i])
    mean=float((vals*cnt).sum()/tot)
    return {"n":int(tot),"min":float(vals[0]),"p5":q(.05),"p25":q(.25),"median":q(.5),
            "mean":round(mean,3),"p75":q(.75),"p95":q(.95),"max":float(vals[-1])}

out={}
out["total_rows"]=total; out["n_users"]=len(user_ids)
out["nulls"]={c:nulls[c] for c in cols}
out["null_rate"]={c:round(nulls[c]/total*100,2) for c in cols}
out["type_counts"]=dict(type_counts.most_common())
out["sex_counts"]=dict(sex_counts.most_common())
out["country_top20"]=dict(country_counts.most_common(20)); out["n_countries"]=len(countries)
out["age_pct"]=pct_from_hist(age_hist)
out["age_gt120"]=sum(v for k,v in age_hist.items() if k>120)
out["age_neg"]=sum(v for k,v in age_hist.items() if k<0)
out["age_zero_or_1"]=sum(v for k,v in age_hist.items() if k<=1)
out["date_min"]=str(date_min); out["date_max"]=str(date_max); out["future_dates"]=future
out["date_by_month_first6"]=dict(sorted(date_month.items())[:6])
out["date_by_month_last6"]=dict(sorted(date_month.items())[-6:])
out["n_months"]=len(date_month)
out["symptom_value_hist"]={str(k):int(v) for k,v in sorted(sym_hist.items())}
out["symptom_value_pct"]=pct_from_hist(sym_hist)
tvn={}
for t,d in tv_agg.items():
    tvn[t]={"count":d["count"],"min":(None if d["min"]==np.inf else d["min"]),
            "max":(None if d["max"]==-np.inf else d["max"]),"neg":d["neg"],"zero":d["zero"],
            "gt4":d["gt4"],"mean":round(d["sum"]/d["count"],3) if d["count"] else None}
out["trackable_value_numeric_by_type"]=tvn
out["trackable_value_nonnumeric_counts"]=dict(tv_nonnum.most_common())
tn={}
for t in ["Condition","Symptom","Treatment","Tag","Food","Weather"]:
    if t in names_by_type: tn[t]=dict(names_by_type[t].most_common(15))
out["top_trackable_names"]=tn
out["n_trackable_names_by_type"]={t:len(c) for t,c in names_by_type.items()}
out["elapsed_sec"]=round(time.time()-t0,1)
OUT.parent.mkdir(parents=True,exist_ok=True)
json.dump(out,open(OUT,"w"),indent=2,default=str)
print("DONE",total,"rows",len(user_ids),"users",out["elapsed_sec"],"s ->",OUT)
