"""Stratified paired bootstrap B=5000 of *fixed* unified test predictions.
No refitting. Costs use thresholds chosen on training OOF in unified_24_cv.py.
"""
from pathlib import Path
import argparse
import numpy as np,pandas as pd,time,json
parser=argparse.ArgumentParser(description='Paired stratified bootstrap on fixed saved probabilities (24 configs).')
parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results')
parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'results'/'recomputed')
parser.add_argument('--replicates',type=int,default=5000)
parser.add_argument('--seed',type=int,default=20261103)
args=parser.parse_args()
ROOT=args.results_dir.resolve();OUT=args.out_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
pred=np.load(ROOT/'test_predictions.npz');summary=pd.read_csv(ROOT/'test_and_cv_summary.csv')
methods=['weighting','ros','smote'];models=['logreg','rf','xgb']; datasets=['AI4I','Scania'];
N=args.replicates;seed=args.seed
rng=np.random.default_rng(seed)
y={d:pred[f'{d}_y'].astype(int) for d in datasets}
positive={d:np.where(y[d]==1)[0] for d in datasets};negative={d:np.where(y[d]==0)[0] for d in datasets}
pre={}
for d in datasets:
 for m in models:
  for t in ['baseline',*methods]:
   scores=pred[f'{d}_{m}_{t}'];sort=np.argsort(-scores,kind='stable');sv=scores[sort]
   # aggregate equal thresholds exactly as AP integrates equal-score groups
   cuts=np.flatnonzero(np.r_[sv[1:]!=sv[:-1],True]); idx=np.r_[0,cuts[:-1]+1]
   pre[(d,m,t)]={'sort':sort,'cuts':cuts,'starts':idx,'y':y[d][sort]}

cost_cfg={}
sc=summary[summary.dataset=='Scania']
for r in sc.itertuples():
 scores=pred[f'Scania_{r.model}_{r.treatment}']; labels=(scores>=r.cost_threshold_oof).astype(int)
 cost_cfg[(r.model,r.treatment)]=10*(labels==1)*(y['Scania']==0)+500*(labels==0)*(y['Scania']==1)

def eval_ap(ds,m,t,w):
 inf=pre[(ds,m,t)];ws=w[inf['sort']]; wp=ws*inf['y'];
 # fast unique score block aggregation
 pos=np.add.reduceat(wp,inf['starts']);tot=np.add.reduceat(ws,inf['starts']);
 tp=np.cumsum(pos);predpos=np.cumsum(tot);prec=np.divide(tp,predpos,out=np.zeros_like(tp,dtype=float),where=predpos>0)
 npos=wp.sum(); return float(np.sum(prec*pos)/npos)

records=[];costrows=[];start=time.monotonic()
for b in range(N):
 counts={}
 for d in datasets:
  i1=rng.choice(positive[d],size=len(positive[d]),replace=True)
  i0=rng.choice(negative[d],size=len(negative[d]),replace=True)
  counts[d]=np.bincount(np.r_[i0,i1],minlength=len(y[d]))
 aps={(d,m,t):eval_ap(d,m,t,counts[d]) for d in datasets for m in models for t in ['baseline',*methods]}
 ss={};basecorr={}
 for t in methods:
  dscores=[]
  for m in models:
   a,bp=aps[('AI4I',m,'baseline')],aps[('Scania',m,'baseline')]
   ga=(aps[('AI4I',m,t)]-a)/(1-a)
   gs=(aps[('Scania',m,t)]-bp)/(1-bp)
   dscores.append(abs(ga-gs))
  ss[t]=float(np.mean(dscores))
 costs={(m,t):int(np.dot(counts['Scania'],cost_cfg[(m,t)])) for m in models for t in ['baseline',*methods]}
 row={'replicate':b+1,'S_weighting':ss['weighting'],'S_ros':ss['ros'],'S_smote':ss['smote'],'W_minus_ROS':ss['weighting']-ss['ros'],'W_minus_SMOTE':ss['weighting']-ss['smote']}
 for m in models:
  for t in methods:row[f'cost_delta_{m}_{t}']=costs[(m,t)]-costs[(m,'baseline')]
 row['cost_delta_rf_ros_minus_xgb_base']=costs[('rf','ros')]-costs[('xgb','baseline')]
 records.append(row)
 if (b+1)%500==0:print(b+1,round(time.monotonic()-start,1),flush=True)
boot=pd.DataFrame.from_records(records)
boot.to_csv(OUT/f'unified_bootstrap_{N}.csv',index=False)
# Point scores from exact sample indices
point={}
fullap={(d,m,t):eval_ap(d,m,t,np.ones(len(y[d]),dtype=int)) for d in datasets for m in models for t in ['baseline',*methods]}
for t in methods:
 dd=[]
 for m in models:
  ab, sb=fullap[('AI4I',m,'baseline')],fullap[('Scania',m,'baseline')]
  ai=(fullap[('AI4I',m,t)]-ab)/(1-ab);scg=(fullap[('Scania',m,t)]-sb)/(1-sb)
  dd.append(abs(ai-scg))
 point['S_'+t]=float(np.mean(dd))
point['W_minus_ROS']=point['S_weighting']-point['S_ros'];point['W_minus_SMOTE']=point['S_weighting']-point['S_smote']
for m in models:
 for t in methods:
  raw=sc.set_index(['model','treatment'])
  point[f'cost_delta_{m}_{t}']=float(raw.loc[(m,t),'cost_costopt']-raw.loc[(m,'baseline'),'cost_costopt'])
point['cost_delta_rf_ros_minus_xgb_base']=float(sc.set_index(['model','treatment']).loc[('rf','ros'),'cost_costopt']-sc.set_index(['model','treatment']).loc[('xgb','baseline'),'cost_costopt'])
intervals=[]
for name in boot.columns[1:]:
 arr=boot[name].to_numpy();lo,hi=np.quantile(arr,[.025,.975]);v=point[name]
 intervals.append({'metric':name,'point_estimate':v,'bootstrap_mean':arr.mean(),'CI95_lower':lo,'CI95_upper':hi,'one_sided_upper_97_5':hi,'fraction_negative':(arr<0).mean()})
pd.DataFrame(intervals).to_csv(OUT/f'unified_bootstrap_intervals_{N}.csv',index=False)
(OUT/'bootstrap_manifest.json').write_text(json.dumps({'bootstrap_count':N,'seed':seed,'stratified':True,'pairing':'same row multiplicities for 12 models within each dataset','models_fixed':True,'cost_threshold':'OOF training, no re-estimation on test bootstrap','note':'exploratory new unified series'},ensure_ascii=False,indent=2),encoding='utf8')
print('BOOTSTRAP FINISHED',round(time.monotonic()-start,1),flush=True)
