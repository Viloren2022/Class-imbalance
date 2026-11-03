"""Recompute Delta AP, normalized effects G, D and S from unified test summary."""
from pathlib import Path
import argparse,csv
p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results');p.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'results'/'recomputed')
a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
with (a.results_dir/'test_and_cv_summary.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
idx={(x['dataset'],x['model'],x['treatment']):float(x['test_AP']) for x in rr}
assert len(idx)==24,'Expected 24 unique pairs of dataset/model/treatment'
models=['logreg','rf','xgb'];methods=['weighting','ros','smote']
rows=[]
for model in models:
 for tr in methods:
  r={'model':model,'treatment':tr}
  for ds in ['AI4I','Scania']:
   b=idx[ds,model,'baseline'];value=idx[ds,model,tr];delta=value-b
   r.update({f'{ds}_AP':value,f'{ds}_baseline_AP':b,f'{ds}_delta_AP':delta,f'{ds}_G':delta/(1-b)})
  r['D_G']=abs(r['AI4I_G']-r['Scania_G'])
  r['D_absolute_delta_AP']=abs(r['AI4I_delta_AP']-r['Scania_delta_AP'])
  r['D_relative_delta_AP']=abs(r['AI4I_delta_AP']/r['AI4I_baseline_AP']-r['Scania_delta_AP']/r['Scania_baseline_AP'])
  rows.append(r)
summary=[]
for t in methods:
 q=[x for x in rows if x['treatment']==t]
 summary.append({'treatment':t,'S_G':sum(x['D_G'] for x in q)/3,'S_absolute_delta_AP':sum(x['D_absolute_delta_AP'] for x in q)/3,'S_relative_delta_AP':sum(x['D_relative_delta_AP'] for x in q)/3})
for filename,body in [('normalized_effects_G_D.csv',rows),('stability_summary_S.csv',summary)]:
 with (a.out_dir/filename).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(body[0]));w.writeheader();w.writerows(body)
print('Recomputed',len(rows),'G/D values and',len(summary),'S values')
