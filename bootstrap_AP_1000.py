"""Paired stratified percentile CIs for Average Precision, fixed test predictions.

Does not re-fit classifiers and is not the preregistered H1 confirmatory test.
"""
from pathlib import Path
import argparse,csv
import numpy as np
from sklearn.metrics import average_precision_score
p=argparse.ArgumentParser()
p.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results')
p.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'results')
p.add_argument('--replicates',type=int,default=1000)
p.add_argument('--seed',type=int,default=20261103)
a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
x=np.load(a.results_dir/'test_predictions.npz')
models=['logreg','rf','xgb'];tr=['baseline','weighting','ros','smote']
D=['AI4I','Scania']
pre={}
for d in D:
 y=x[f'{d}_y'].astype(int)
 for m in models:
  for t in tr:
   prob=x[f'{d}_{m}_{t}'];sort=np.argsort(-prob,kind='stable');sp=prob[sort]
   start=np.r_[0,np.flatnonzero(sp[:-1]!=sp[1:])+1]
   pre[d,m,t]=(sort,start,y[sort])
def weighted_ap(key,w):
 perm,start,labels=pre[key]; v=w[perm];positive=v*labels
 pos=np.add.reduceat(positive,start); tot=np.add.reduceat(v,start)
 prec=np.divide(np.cumsum(pos),np.cumsum(tot),out=np.zeros(len(pos)),where=np.cumsum(tot)>0)
 return float((prec*pos).sum()/positive.sum())
rng=np.random.default_rng(a.seed); series={ (d,m,t):[] for d in D for m in models for t in tr}
for d in D:
 y=x[f'{d}_y'].astype(int);ii={k:np.where(y==k)[0] for k in (0,1)}
 for b in range(a.replicates):
  chosen=np.concatenate([rng.choice(ii[k],size=len(ii[k]),replace=True) for k in (0,1)])
  counts=np.bincount(chosen,minlength=len(y))
  for m in models:
   for t in tr:series[d,m,t].append(weighted_ap((d,m,t),counts))
results=[];deltas=[]
for d in D:
 y=x[f'{d}_y'].astype(int)
 for m in models:
  ref=np.array(series[d,m,'baseline'])
  for t in tr:
   vals=np.array(series[d,m,t]);point=average_precision_score(y,x[f'{d}_{m}_{t}'])
   lo,hi=np.quantile(vals,[.025,.975])
   results.append({'dataset':d,'model':m,'treatment':t,'test_AP':point,'CI95_lower':lo,'CI95_upper':hi,'bootstrap_B':a.replicates,'bootstrap_seed':a.seed})
   if t!='baseline':
    diff=vals-ref;value=point-average_precision_score(y,x[f'{d}_{m}_baseline'])
    low,high=np.quantile(diff,[.025,.975])
    deltas.append({'dataset':d,'model':m,'treatment':t,'delta_AP':value,'CI95_lower':low,'CI95_upper':high,'bootstrap_B':a.replicates,'bootstrap_seed':a.seed})
def write(name,rows):
 with (a.out_dir/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(f'AP_test_CI_{a.replicates}.csv',results)
write(f'delta_AP_test_CI_{a.replicates}.csv',deltas)
print('Saved',len(results),'AP intervals and',len(deltas),'paired delta intervals')
