"""Generate tabular description of train/test datasets from original UCI CSVs."""
from pathlib import Path
import argparse,csv
from chapter2_experiment import build_bundles
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data');p.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'results'/'recomputed');a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
sets=build_bundles(a.data_dir);rows=[]
for ds in sets.values():
 for name,X,y in [('train',ds.X_train,ds.y_train),('test',ds.X_test,ds.y_test)]:
  positives=int(y.sum())
  rows.append({'dataset':ds.name,'split':name,'rows':len(y),'features':X.shape[1],'positive':positives,'negative':len(y)-positives,'positive_rate':positives/len(y),'missing_cells':int(X.isna().sum().sum()),'missing_share':float(X.isna().mean().mean())})
with (a.out_dir/'dataset_overview.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Wrote dataset_overview.csv:',len(rows),'splits')
