"""Paired stratified 1000 bootstrap for XGBoost ROS/SMOTE versus baseline, within PMDI."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
P=Path(__file__).resolve().parent
rows=[]
for protocol in ['matched_labels','native_diagnostic']:
  z=np.load(P/f'{protocol}_test_probabilities.npz');y=z['true'];a=z['xgb__baseline'];i0=np.where(y==0)[0];i1=np.where(y==1)[0]
  rng=np.random.default_rng(42)
  draws=[]
  for j in range(1000):
    idx=np.r_[rng.choice(i0,len(i0),replace=True),rng.choice(i1,len(i1),replace=True)]
    base=average_precision_score(y[idx],a[idx])
    draws.append((average_precision_score(y[idx],z['xgb__smote'][idx])-base, average_precision_score(y[idx],z['xgb__ros'][idx])-base))
  d=np.asarray(draws)
  for j,method in enumerate(['smote','ros']):
    point=average_precision_score(y,z[f'xgb__{method}'])-average_precision_score(y,a)
    rows.append({'protocol':protocol,'method':method,'point_delta_AP':point,'CI95_lower':np.quantile(d[:,j],.025),'CI95_upper':np.quantile(d[:,j],.975)})
pd.DataFrame(rows).to_csv(P/'within_PMDI_xgb_bootstrap_1000.csv',index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
