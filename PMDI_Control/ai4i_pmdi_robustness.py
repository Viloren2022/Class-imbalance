"""Matched holdout pair-bootstrap and chronological split secondary assessment."""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from chapter2_experiment import build_bundles,make_pipeline as make_orig_pipeline,binary_metrics
from ai4i_pmdi_experiment import make_pipeline as make_pm_pipeline,ORIG_NUM,PM_NUM,MODELS,TREAT
P=BASE/'ch2_pmdi';P.mkdir(exist_ok=True)
a=pd.read_csv(BASE/'ai4i2020.csv');p=pd.read_csv(BASE/'AI4I-PMDI.csv')
Xorig=a[['Type']+ORIG_NUM]
Xpm=p[['Type']+PM_NUM].rename(columns=dict(zip(PM_NUM,ORIG_NUM)))
y=a['Machine failure'].values;yn=p.Diagnostic.ne('No failure').astype(int).values
idx=np.arange(len(y));train,test=train_test_split(idx,test_size=.2,stratify=y,random_state=42)
orig_bundle=build_bundles(BASE)['AI4I']
# Verify split consistency
assert np.all(orig_bundle.y_test.to_numpy()==y[test]);assert np.all(orig_bundle.X_test.to_numpy()==Xorig.iloc[test].to_numpy())
saved=np.load(P/'matched_labels_test_probabilities.npz')
assert np.all(saved['true']==y[test])
probs={}; summary=[]
for model in MODELS:
 for tr in TREAT:
    start=time.monotonic()
    pipe=make_orig_pipeline(orig_bundle,model,tr)
    pipe.fit(orig_bundle.X_train,orig_bundle.y_train)
    score=pipe.predict_proba(orig_bundle.X_test)[:,1]
    probs[f'{model}__{tr}']=score
    summ=average_precision_score(y[test],score)
    matched_ap=average_precision_score(y[test],saved[f'{model}__{tr}'])
    summary.append({'model':model,'treatment':tr,'original_ap_recomputed':summ,'pmdi_matched_ap':matched_ap,'difference_ap':matched_ap-summ,'seconds':round(time.monotonic()-start,2)})
    print('ORIGINAL',model,tr,round(summ,6),'PMDI',round(matched_ap,6),flush=True)
np.savez_compressed(P/'original_AI4I_test_probabilities.npz',true=y[test],UDI=test,**probs)
reference=pd.read_csv(BASE/'ch2_results_final/test_summary.csv').query("dataset=='AI4I'")
review=pd.DataFrame(summary).merge(reference[['model','treatment','test_AP']],on=['model','treatment'])
review['recomputed_error']=(review['original_ap_recomputed']-review['test_AP']).abs()
print('MAX REF ABS ERROR',review['recomputed_error'].max(),flush=True)
review.to_csv(P/'paired_AP_verified.csv',index=False)

# paired stratified resampling, same sample indices within all configurations
idx0=np.where(y[test]==0)[0];idx1=np.where(y[test]==1)[0]
rng=np.random.default_rng(20261102)
res=[]
for b in range(1000):
 picks=np.concatenate([rng.choice(idx0,len(idx0),replace=True),rng.choice(idx1,len(idx1),replace=True)])
 # same picks for all matched models and treatments
 for key in probs:
    ap_orig=average_precision_score(y[test][picks],probs[key][picks]);ap_pmdi=average_precision_score(y[test][picks],saved[key][picks])
    model,tr=key.split('__');res.append({'bootstrap':b,'model':model,'treatment':tr,'diff_pmdi_minus_original':ap_pmdi-ap_orig})
boot=pd.DataFrame(res);boot.to_csv(P/'paired_AP_bootstrap_1000.csv',index=False)
rows=[]
for (m,t),z in boot.groupby(['model','treatment'],sort=False):
 v=z.diff_pmdi_minus_original.to_numpy()
 point=review.set_index(['model','treatment']).loc[(m,t),'difference_ap']
 rows.append({'model':m,'treatment':t,'point_diff_AP':point,'boot_mean':v.mean(),'CI95_lower':np.quantile(v,.025),'CI95_upper':np.quantile(v,.975),'prob_diff_positive_boot':(v>0).mean()})
pd.DataFrame(rows).to_csv(P/'paired_AP_bootstrap_CI.csv',index=False)
print('BOOTSTRAP COMPLETE',flush=True)

# Chronological holdout - independent sensitivity run only, using native target and 6 core features
D=pd.to_datetime(p.Date,format='%d/%m/%Y %H:%M')
ix=np.argsort(D.to_numpy(),kind='stable');tridx=ix[:8000];teidx=ix[8000:]
rows=[]
for m in MODELS:
 pipe=make_pm_pipeline(yn[tridx],m,'baseline')
 start=time.monotonic();pipe.fit(Xpm.iloc[tridx],yn[tridx]);scores=pipe.predict_proba(Xpm.iloc[teidx])[:,1]
 mm=binary_metrics(yn[teidx],scores)
 rows.append({'model':m,'train_n':len(tridx),'train_positives':sum(yn[tridx]),'test_n':len(teidx),'test_positives':sum(yn[teidx]),'time_split_test_AP':mm['AP'],'test_f1_at_0_5':mm['f1'],'seconds':round(time.monotonic()-start,2)})
 print('CHRONO',m,round(mm['AP'],5),flush=True)
pd.DataFrame(rows).to_csv(P/'chronological_split_baseline.csv',index=False)
print('COMPLETE',flush=True)
