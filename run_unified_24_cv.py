"""Independent unified 24-config audit, new exploratory series, not registered original H1."""
from pathlib import Path
import sys,time,json,hashlib,argparse
import numpy as np,pandas as pd,sklearn,imblearn,xgboost
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0,str(Path(__file__).resolve().parent))
import chapter2_experiment as mod
parser=argparse.ArgumentParser(description='24 configurations, 5-fold CV and independent test. No test leakage.')
parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'results'/'recomputed')
parser.add_argument('--only-dataset',choices=['AI4I','Scania'])
parser.add_argument('--only-model',choices=['logreg','rf','xgb'])
parser.add_argument('--only-treatment',choices=['baseline','weighting','ros','smote'])
args=parser.parse_args()
OUT=args.out_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
DATA=args.data_dir.resolve()
datasets=mod.build_bundles(DATA)
if args.only_dataset:datasets={args.only_dataset:datasets[args.only_dataset]}
models=['logreg','rf','xgb'];treatments=['baseline','weighting','ros','smote']
def make(ds,model,method):
 p=mod.make_pipeline(ds,model,method)
 if model=='rf':p.set_params(model__n_estimators=50,model__max_samples=0.5,model__n_jobs=4)
 if model=='logreg':p.set_params(model__solver='lbfgs')
 return p
res=[];folds=[];probs={}
for ds in datasets.values():
 probs[f'{ds.name}_y']=ds.y_test.to_numpy(dtype=np.int8)
 for model in models:
  if args.only_model and model!=args.only_model:continue
  for t in treatments:
   if args.only_treatment and t!=args.only_treatment:continue
   start=time.monotonic();oof=np.full(len(ds.y_train),np.nan,dtype=float)
   cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
   thisfolds=[]
   for fold,(tr,va) in enumerate(cv.split(ds.X_train,ds.y_train),1):
    pipe=make(ds,model,t);pipe.fit(ds.X_train.iloc[tr],ds.y_train.iloc[tr]);scores=pipe.predict_proba(ds.X_train.iloc[va])[:,1];oof[va]=scores
    m=mod.binary_metrics(ds.y_train.iloc[va].to_numpy(),scores,.5)
    thisfolds.append({'dataset':ds.name,'model':model,'treatment':t,'fold':fold,**m})
   folds.extend(thisfolds)
   threshold,_=mod.select_cost_threshold(ds.y_train.to_numpy(),oof)
   pipe=make(ds,model,t);pipe.fit(ds.X_train,ds.y_train);score=pipe.predict_proba(ds.X_test)[:,1]
   test=mod.binary_metrics(ds.y_test.to_numpy(),score,.5);cost=mod.binary_metrics(ds.y_test.to_numpy(),score,threshold)
   r={'dataset':ds.name,'model':model,'treatment':t,'test_AP':test['AP'],'cv_AP_mean':float(np.mean([x['AP'] for x in thisfolds])),'cv_AP_sd':float(np.std([x['AP'] for x in thisfolds],ddof=1)),'precision_05':test['precision'],'recall_05':test['recall'],'f1_05':test['f1'],'fp_05':test['fp'],'fn_05':test['fn'],'cost_threshold_oof':threshold,'fp_costopt':cost['fp'],'fn_costopt':cost['fn'],'cost_costopt':cost['cost_10fp_500fn'],'elapsed_seconds':time.monotonic()-start}
   res.append(r);probs[f'{ds.name}_{model}_{t}']=score
   pd.DataFrame(res).to_csv(OUT/'summary_partial.csv',index=False)
   pd.DataFrame(folds).to_csv(OUT/'folds_partial.csv',index=False)
   np.savez_compressed(OUT/'probs_partial.npz',**probs)
   print(f'{len(res):02d}/24 {ds.name} {model} {t} AP={test["AP"]:.6f} CV={r["cv_AP_mean"]:.6f} cost={r["cost_costopt"]:.0f} sec={r["elapsed_seconds"]:.1f}',flush=True)
pd.DataFrame(res).to_csv(OUT/'test_and_cv_summary.csv',index=False)
pd.DataFrame(folds).to_csv(OUT/'cv_5fold_metrics.csv',index=False)
np.savez_compressed(OUT/'test_predictions.npz',**probs)
for f in ['summary_partial.csv','folds_partial.csv','probs_partial.npz']:(OUT/f).unlink(missing_ok=True)
files={'ai4i':DATA/'ai4i2020.csv','scan_train':DATA/'aps_failure_training_set.csv','scan_test':DATA/'aps_failure_test_set.csv','runner':Path(__file__),'pipeline':Path(mod.__file__)}
manifest={'sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()},'python':sys.version,'sklearn':sklearn.__version__,'imblearn':imblearn.__version__,'xgboost':xgboost.__version__,'pandas':pd.__version__,'numpy':np.__version__,'cv_folds':5,'random_state':42,'RF':'50 trees, max_samples=0.5, scaled features on both datasets','LR':'solver=lbfgs','XGBoost':'400 trees as chapter2_experiment.py','status':'Independent unified sensitivity series. NOT preregistered primary H1.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('FINISHED ALL 24',flush=True)
