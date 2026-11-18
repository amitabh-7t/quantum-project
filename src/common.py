import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from feats import build, FEATS
from data import ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, LABELED
SEED=42
NOLEAK=['dc_norm','p_exp','acdc','IRRADIATION','MODULE_TEMPERATURE','AMBIENT_TEMPERATURE']  # drops residual, zscore
RAW=['IRRADIATION','MODULE_TEMPERATURE','AMBIENT_TEMPERATURE','DC_POWER','AC_POWER','hour']
def get(**kw):
    d=build(**kw); d['hour']=d.DATE_TIME.dt.hour+d.DATE_TIME.dt.minute/60; d['day']=d.DATE_TIME.dt.strftime('%m-%d'); return d
def split(d,seed=SEED):
    return train_test_split(np.arange(len(d)),test_size=0.25,stratify=d.y.values,random_state=seed)
def metrics(y,pred,score):
    return dict(prec=precision_score(y,pred,zero_division=0),rec=recall_score(y,pred),f1=f1_score(y,pred),auc=roc_auc_score(y,score))
def matched(d,tr,te,seed=SEED,ntr=150,nte=80,frac=0.35):
    rng=np.random.default_rng(seed); y=d.y.values
    def pick(idx,n):
        a=idx[y[idx]==1]; b=idx[y[idx]==0]; na=int(round(n*frac))
        return np.concatenate([rng.choice(a,na,replace=False),rng.choice(b,n-na,replace=False)])
    return pick(tr,ntr),pick(te,nte)
