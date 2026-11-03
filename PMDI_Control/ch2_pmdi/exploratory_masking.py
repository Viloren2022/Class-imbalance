"""Exploratory MCAR counterfactual vs structured PMDI missingness; same labels and split."""
import sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
from ai4i_pmdi_experiment import make_pipeline,ORIG_NUM,PM_NUM,MODELS
P=BASE/'ch2_pmdi'
a=pd.read_csv(BASE/'ai4i2020.csv');p=pd.read_csv(BASE/'AI4I-PMDI.csv')
Xorig=a[['Type']+ORIG_NUM];y=a['Machine failure'].to_numpy();tr,te=train_test_split(np.arange(len(y)),stratify=y,random_state=42,test_size=.2)
rates=p[PM_NUM].isna().mean().to_numpy()
rows=[]
for seed in [42,123,2026]:
 rng=np.random.default_rng(seed)
 masked=Xorig.copy()
 # MCAR: generate masks with identical marginal Bernoulli probabilities as PMDI, independent of values and class
 for j,col in enumerate(ORIG_NUM):
    masked.loc[rng.random(len(masked)) < rates[j],col]=np.nan
 for m in MODELS:
    start=time.monotonic()
    mod=make_pipeline(y[tr],m,'baseline');mod.fit(masked.iloc[tr],y[tr]);sc=mod.predict_proba(masked.iloc[te])[:,1]
    ap=average_precision_score(y[te],sc)
    rows.append({'seed':seed,'model':m,'test_AP':ap,'seconds':round(time.monotonic()-start,2)})
    print('MCAR',seed,m,round(ap,5),flush=True)
pd.DataFrame(rows).to_csv(P/'MCAR_baseline_3seeds.csv',index=False)
print('FINISHED',flush=True)
