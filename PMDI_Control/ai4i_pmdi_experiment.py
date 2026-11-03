"""Supplemental AI4I-PMDI experiment. Run: python /mnt/data/ai4i_pmdi_experiment.py"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTENC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from chapter2_experiment import make_model, make_ai4i_preprocessor, binary_metrics

DIR=BASE/'ch2_pmdi';DIR.mkdir(exist_ok=True)
SEED=42
MODELS=['logreg','rf','xgb'];TREAT=['baseline','weighting','ros','smote']
ORIG_NUM=['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
PM_NUM=['Air temperature (K)','Process temperature (K)','Rotational speed (rpm)','Torque (Nm)','Tool wear (min)']

def make_pipeline(ytrain, model, tr):
    pre=ColumnTransformer([
        ('type',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),['Type']),
        ('num',ImbPipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),ORIG_NUM)
    ],remainder='drop',sparse_threshold=0,verbose_feature_names_out=False)
    _,post=make_ai4i_preprocessor()
    steps=[('pre',pre)]
    if tr=='ros':steps.append(('sampler',RandomOverSampler(random_state=SEED)))
    elif tr=='smote':steps.append(('sampler',SMOTENC(categorical_features=[0],random_state=SEED,k_neighbors=5)))
    steps.extend([('post',post),('model',make_model(model,tr,pd.Series(ytrain)))])
    return ImbPipeline(steps)

def evaluate_run(tag,X, y, train_idx, test_idx):
    Xt=X.iloc[train_idx].reset_index(drop=True); Xh=X.iloc[test_idx].reset_index(drop=True)
    yt=np.asarray(y)[train_idx]; yh=np.asarray(y)[test_idx]
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    fold_rows=[]; summary=[]; test_probs={}
    print('START',tag,'train',len(yt),'positives',sum(yt),'test',len(yh),'positives',sum(yh),flush=True)
    for m in MODELS:
      for t in TREAT:
        start=time.monotonic(); fold_aps=[]
        for k,(i,j) in enumerate(folds.split(Xt,yt),1):
            pipe=make_pipeline(yt,m,t)
            pipe.fit(Xt.iloc[i],yt[i]); p=pipe.predict_proba(Xt.iloc[j])[:,1]
            met=binary_metrics(yt[j],p)
            fold_rows.append({'protocol':tag,'model':m,'treatment':t,'fold':k,**met})
            fold_aps.append(met['AP'])
        pipe=make_pipeline(yt,m,t);pipe.fit(Xt,yt)
        p=pipe.predict_proba(Xh)[:,1];test_probs[f'{m}__{t}']=p
        mets=binary_metrics(yh,p)
        row={'protocol':tag,'model':m,'treatment':t,'cv_AP_mean':float(np.mean(fold_aps)),'cv_AP_sd':float(np.std(fold_aps,ddof=1)),
             'test_prevalence':float(np.mean(yh)),**{'test_'+key:value for key,value in mets.items()},'run_seconds':round(time.monotonic()-start,2)}
        summary.append(row)
        print(tag,m,t,'CV',round(row['cv_AP_mean'],4),'test_AP',round(mets['AP'],4),'secs',row['run_seconds'],flush=True)
        pd.DataFrame(summary).to_csv(DIR/f'{tag}_summary_partial.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(DIR/f'{tag}_fold_metrics.csv',index=False)
    sm_df=pd.DataFrame(summary)
    sm_df.to_csv(DIR/f'{tag}_summary.csv',index=False)
    np.savez_compressed(DIR/f'{tag}_test_probabilities.npz',true=yh,UDI=test_idx,**test_probs)
    return sm_df

if __name__=='__main__':
 p=pd.read_csv(BASE/'AI4I-PMDI.csv');a=pd.read_csv(BASE/'ai4i2020.csv')
 assert len(p)==len(a)==10000 and (p.UDI.to_numpy()==a.UDI.to_numpy()).all()
 assert (p['Product ID'].to_numpy()==a['Product ID'].to_numpy()).all()
 assert (p.Type.to_numpy()==a.Type.to_numpy()).all()
 for c1,c2 in zip(PM_NUM,ORIG_NUM):
    mask=p[c1].notna()
    assert np.all(p.loc[mask,c1].to_numpy()==a.loc[mask,c2].to_numpy()),c1
 X=p[['Type']+PM_NUM].rename(columns=dict(zip(PM_NUM,ORIG_NUM))).copy()
 native=p.Diagnostic.ne('No failure').astype(int).to_numpy()
 matched=a['Machine failure'].to_numpy()
 idx=np.arange(len(p));train,test=train_test_split(idx,stratify=matched,test_size=.2,random_state=SEED)
 dt=pd.to_datetime(p.Date,format='%d/%m/%Y %H:%M')
 audit={'observations':len(p),'source_AI4I_positives':int(sum(matched)),
 'PMDI_Diagnostic_positives':int(sum(native)),'discordant_label_count':int((native!=matched).sum()),
 'different_label_0_to_1':int(np.sum((matched==0)&(native==1))),
 'different_label_1_to_0':int(np.sum((matched==1)&(native==0))),
 'missing_numeric_cells':int(X[ORIG_NUM].isna().sum().sum()),
 'missing_numeric_share':float(X[ORIG_NUM].isna().mean().mean()),
 'date_min':dt.min().isoformat(),'date_max':dt.max().isoformat(),
 'unique_dates':int(dt.nunique()),'system_distinct':int(p.System.nunique()),
 'control_distinct':int(p.Control.nunique()),
 'matched_train_positives':int(sum(matched[train])),'matched_test_positives':int(sum(matched[test])),
 'native_train_positives':int(sum(native[train])),'native_test_positives':int(sum(native[test]))}
 (DIR/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 pd.DataFrame({'feature':ORIG_NUM,'missing_n':X[ORIG_NUM].isna().sum().to_numpy(),'missing_share':X[ORIG_NUM].isna().mean().to_numpy()}).to_csv(DIR/'missingness.csv',index=False)
 pd.crosstab(p.Diagnostic,a['Machine failure']).to_csv(DIR/'diagnostic_vs_original_target.csv')
 pd.DataFrame({'UDI':p.UDI,'Date':p.Date,'original_failure':matched,'pmdi_failure':native}).query('original_failure != pmdi_failure').to_csv(DIR/'label_disagreements.csv',index=False)
 s_matched=evaluate_run('matched_labels',X,matched,train,test)
 s_native=evaluate_run('native_diagnostic',X,native,train,test)
 s_original=pd.read_csv(BASE/'ch2_results_final/test_summary.csv').query("dataset=='AI4I'")[['model','treatment','test_AP','cv_AP_mean']].copy()
 s_original=s_original.rename(columns={'test_AP':'original_AI4I_test_AP','cv_AP_mean':'original_AI4I_cv_AP_mean'})
 cm=s_matched[['model','treatment','test_AP','cv_AP_mean']].rename(columns={'test_AP':'pmdi_matched_AP','cv_AP_mean':'pmdi_matched_cv_AP_mean'})
 cn=s_native[['model','treatment','test_AP','cv_AP_mean']].rename(columns={'test_AP':'pmdi_native_AP','cv_AP_mean':'pmdi_native_cv_AP_mean'})
 compare=s_original.merge(cm,on=['model','treatment']).merge(cn,on=['model','treatment'])
 compare['delta_missingness_matched']=compare.pmdi_matched_AP-compare.original_AI4I_test_AP
 compare['delta_label_effect_pmdi']=compare.pmdi_native_AP-compare.pmdi_matched_AP
 compare.to_csv(DIR/'comparison_original_vs_pmdi.csv',index=False)
 gains=[]
 for label,s in [('matched',s_matched),('native',s_native)]:
    for model in MODELS:
       z=s[s.model==model].set_index('treatment');base=z.loc['baseline','test_AP']
       for tr in TREAT[1:]:
          ap=z.loc[tr,'test_AP']
          gains.append({'protocol':label,'model':model,'treatment':tr,'baseline_AP':base,'treatment_AP':ap,'delta_AP':ap-base,'G':(ap-base)/(1-base)})
 pd.DataFrame(gains).to_csv(DIR/'pmdi_treatment_effects.csv',index=False)
 for protocol, cols in [('matched',('original_AI4I_test_AP','pmdi_matched_AP')),('native',('original_AI4I_test_AP','pmdi_native_AP'))]:
   f,ax=plt.subplots(figsize=(12,5));pos=np.arange(len(compare));width=.36
   ax.bar(pos-width/2,compare[cols[0]].values,width,label='Original AI4I')
   ax.bar(pos+width/2,compare[cols[1]].values,width,label='AI4I-PMDI')
   ax.set_xticks(pos, [m+'\n'+t for m,t in zip(compare.model,compare.treatment)])
   ax.tick_params(axis='x',labelsize=8)
   ax.set_ylim(0,1);ax.set_ylabel('Average Precision');ax.legend(loc='best');ax.set_title(f'AI4I versus AI4I-PMDI: {protocol} target')
   f.tight_layout();f.savefig(DIR/f'AP_comparison_{protocol}.png',dpi=180);plt.close(f)
 print('COMPLETE',flush=True)
