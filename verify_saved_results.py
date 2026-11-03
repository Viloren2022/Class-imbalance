"""Independent zero-training verification of saved unified results."""
from pathlib import Path
import argparse, csv
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix
p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results');a=p.parse_args()
r=a.results_dir
with (r/'test_and_cv_summary.csv').open(encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
x=np.load(r/'test_predictions.npz')
assert len(rows)==24, 'Expected exactly 24 configurations'
for row in rows:
 d,m,t=row['dataset'],row['model'],row['treatment'];y=x[f'{d}_y'].astype(int);s=x[f'{d}_{m}_{t}']
 ap=average_precision_score(y,s)
 assert abs(ap-float(row['test_AP']))<1e-10,(d,m,t,'AP',ap)
 for threshold,fp_col,fn_col in [(0.5,'fp_05','fn_05'),(float(row['cost_threshold_oof']),'fp_costopt','fn_costopt')]:
  tn,fp,fn,tp=confusion_matrix(y,s>=threshold,labels=[0,1]).ravel()
  assert fp==int(row[fp_col]) and fn==int(row[fn_col]),(d,m,t,'confusion')
  if fp_col=='fp_costopt':assert abs(10*fp+500*fn-float(row['cost_costopt']))<1e-8
print('OK: 24/24 AP, confusion matrices and cost checks passed')
