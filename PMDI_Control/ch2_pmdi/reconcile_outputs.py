import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
P=Path(__file__).resolve().parent
o=pd.read_csv(P/'paired_AP_verified.csv')
matched=pd.read_csv(P/'matched_labels_summary.csv')[['model','treatment','test_AP','cv_AP_mean']].rename(columns={'test_AP':'pmdi_matched_AP','cv_AP_mean':'pmdi_matched_cv_AP_mean'})
native=pd.read_csv(P/'native_diagnostic_summary.csv')[['model','treatment','test_AP','cv_AP_mean']].rename(columns={'test_AP':'pmdi_native_AP','cv_AP_mean':'pmdi_native_cv_AP_mean'})
reference=o[['model','treatment','original_ap_recomputed']].rename(columns={'original_ap_recomputed':'original_AI4I_test_AP'})
c=reference.merge(matched,on=['model','treatment']).merge(native,on=['model','treatment'])
c['delta_pmdi_matched_vs_original']=c.pmdi_matched_AP-c.original_AI4I_test_AP
c['delta_pmdi_native_vs_matched']=c.pmdi_native_AP-c.pmdi_matched_AP
c.to_csv(P/'comparison_original_vs_pmdi.csv',index=False)
ctrl=pd.read_csv(P.parent/'AI4I-PMDI.csv')
for col in ['Air temperature (K)','Process temperature (K)','Rotational speed (rpm)','Torque (Nm)','Tool wear (min)']:
 pass
ctrl.groupby('Control').agg({col:(lambda s:float(s.isna().mean())) for col in ['Air temperature (K)','Process temperature (K)','Rotational speed (rpm)','Torque (Nm)','Tool wear (min)']}).to_csv(P/'missingness_by_control.csv')
pd.crosstab(ctrl['Control'],ctrl['Diagnostic']).to_csv(P/'control_vs_diagnostic.csv')
dt=pd.to_datetime(ctrl.Date,format='%d/%m/%Y %H:%M')
ctrl.assign(year=dt.dt.year,failure=ctrl.Diagnostic.ne('No failure')).groupby('year').failure.agg(['count','sum','mean']).to_csv(P/'failure_prevalence_by_year.csv')
for proto in ['matched','native']:
  cmp='pmdi_'+proto+'_AP'
  fig,ax=plt.subplots(figsize=(12,5))
  pos=np.arange(len(c)); width=.36
  ax.bar(pos-width/2,c.original_AI4I_test_AP,width,label='Original AI4I')
  ax.bar(pos+width/2,c[cmp],width,label='AI4I-PMDI')
  ax.set_xticks(pos,[m+'\n'+t for m,t in zip(c.model,c.treatment)],fontsize=8)
  ax.set_ylim(0,1);ax.set_ylabel('Average Precision')
  ax.set_title('Identical hyperparameters / '+proto+' labels')
  ax.legend();fig.tight_layout();fig.savefig(P/f'AP_comparison_{proto}.png',dpi=200);plt.close(fig)
print(c[['model','treatment','original_AI4I_test_AP','pmdi_matched_AP','pmdi_native_AP']].round(4).to_string(index=False))
