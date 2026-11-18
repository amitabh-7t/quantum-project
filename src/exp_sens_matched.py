# Threshold sensitivity of the matched classical-vs-QSVM gap (10 seeds per configuration)
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *
from matched import run
CF=[('base',{}),('res-0.25',dict(thr_res=-0.25)),('res-0.45',dict(thr_res=-0.45)),('acdc0.80-1.05',dict(acdc_lo=0.80,acdc_hi=1.05)),
    ('acdc0.90-1.01',dict(acdc_lo=0.90,acdc_hi=1.01)),('z2.5',dict(zthr=2.5)),('z3.5',dict(zthr=3.5))]
def one(nm,kw,s):
    d=get(**kw); tr,te=split(d); r=run(d,FEATS,tr,te,seed=s,verbose=False)
    return dict(cfg=nm,seed=s,c=r['c']['f1'],q=r['q'][r['qbest']]['f1'])
res=Parallel(n_jobs=6)(delayed(one)(nm,kw,s) for nm,kw in CF for s in range(10))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df=pd.DataFrame(res); df['dF1']=df.c-df.q; df.to_csv(RESULTS_DIR / 'res_sens_matched.csv',index=False)
print(df.groupby('cfg',sort=False).agg(c=('c','mean'),q=('q','mean'),d=('dF1','mean'),dsd=('dF1','std'),wins=('dF1',lambda x:(x>0).sum())).round(3))
