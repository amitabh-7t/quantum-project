# Repeated-subsample experiments: families, PCA sweep, no-leakage, chronological, cross-plant, noise.
import sys, json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *
from matched import run, prep, qsvm, csvm, cv_precomp
from qkernel import kernel_noisy, states, kernel_from_states
from sklearn.svm import SVC
SEEDS=list(range(20))
d=pd.read_pickle(LABELED)
def chrono(d):
    days=sorted(d.day.unique()); cut=days[25]  # first 25 of 34 days train
    return np.where(d.day<cut)[0], np.where(d.day>=cut)[0], days[:25][-1], cut
def splits(name):
    if name=='random': return split(d)
    if name=='chrono': a,b,_,_=chrono(d); return a,b
    if name=='p1to2': return np.where(d.plant==1)[0],np.where(d.plant==2)[0]
    if name=='p2to1': return np.where(d.plant==2)[0],np.where(d.plant==1)[0]
def one(seed,cfg):
    tr,te=splits(cfg['split'])
    feats={'all':FEATS,'noleak':NOLEAK,'raw':RAW}[cfg['feats']]
    r=run(d,feats,tr,te,seed=seed,ncomp=cfg.get('ncomp',4),Ls=cfg.get('Ls',(1,2,3)),kind=cfg.get('kind','zz'),verbose=False)
    out=dict(seed=seed,ev=r['ev'],qL=r['qbest'],q_f1=r['q'][r['qbest']]['f1'],q_auc=r['q'][r['qbest']]['auc'],c_f1=r['c']['f1'],c_auc=r['c']['auc'],
             qC=r['q'][r['qbest']]['C'],c_params=str(r['c']['params']),q_perL={L:r['q'][L]['f1'] for L in r['q']})
    if cfg.get('noise'):
        y=d.y.values; Xtr,Xte=r['f'](r['itr']),r['f'](r['ite']); ytr,yte=y[r['itr']],y[r['ite']]
        rng=np.random.default_rng(seed)
        S1,S2=states(Xtr,2),states(Xte,2); Ktr0,Kte0=kernel_from_states(S1,S1),kernel_from_states(S2,S1)
        for shots in (100,1000):
            Ktr=rng.binomial(shots,np.clip(Ktr0,0,1))/shots; Ktr=np.triu(Ktr,1); Ktr=Ktr+Ktr.T+np.eye(len(Ktr)); Kte=rng.binomial(shots,np.clip(Kte0,0,1))/shots
            cvf,C=cv_precomp(Ktr,ytr); m=SVC(kernel='precomputed',C=C,class_weight='balanced').fit(Ktr,ytr); s=m.decision_function(Kte)
            out[f'shots{shots}_f1']=f1_score(yte,(s>0).astype(int))
        for p in (0.001,0.01):
            Ktr=kernel_noisy(Xtr,Xtr,2,p); Kte=kernel_noisy(Xte,Xtr,2,p)
            cvf,C=cv_precomp(Ktr,ytr); m=SVC(kernel='precomputed',C=C,class_weight='balanced').fit(Ktr,ytr); s=m.decision_function(Kte)
            out[f'depol{p}_f1']=f1_score(yte,(s>0).astype(int)); out[f'depol{p}_kmean_offdiag']=float(Ktr[~np.eye(len(Ktr),dtype=bool)].mean())
        out['exact_L2_f1']=r['q'][2]['f1']; out['exact_kmean_offdiag']=float(Ktr0[~np.eye(len(Ktr0),dtype=bool)].mean())
    return out
CFGS={
 'base':dict(split='random',feats='all'),
 'zz_L1to4':dict(split='random',feats='all',Ls=(1,2,3,4)),
 'zonly':dict(split='random',feats='all',kind='zonly'),
 'iqp_allpairs':dict(split='random',feats='all',kind='iqp'),
 'pca2':dict(split='random',feats='all',ncomp=2),
 'pca6':dict(split='random',feats='all',ncomp=6),
 'pca8':dict(split='random',feats='all',ncomp=8),
 'noleak':dict(split='random',feats='noleak'),
 'raw':dict(split='random',feats='raw'),
 'chrono':dict(split='chrono',feats='all'),
 'chrono_noleak':dict(split='chrono',feats='noleak'),
 'p1to2':dict(split='p1to2',feats='all'),
 'p2to1':dict(split='p2to1',feats='all'),
 'noise':dict(split='random',feats='all',noise=True),
}
if __name__=='__main__':
    names=sys.argv[1:] or list(CFGS)
    for nm in names:
        cfg=CFGS[nm]; nj=2 if cfg.get('noise') else 4
        seeds=SEEDS[:10] if cfg.get('noise') else SEEDS
        res=Parallel(n_jobs=nj)(delayed(one)(s,cfg) for s in seeds)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df=pd.DataFrame(res); df.to_csv(RESULTS_DIR / f'res_seeds_{nm}.csv',index=False)
        dq=df.c_f1-df.q_f1
        print(nm, f"ev {df.ev.mean():.3f} | SVC F1 {df.c_f1.mean():.3f}±{df.c_f1.std():.3f} AUC {df.c_auc.mean():.3f} | QSVM F1 {df.q_f1.mean():.3f}±{df.q_f1.std():.3f} AUC {df.q_auc.mean():.3f} | dF1 {dq.mean():.3f}±{dq.std():.3f} wins {(dq>0).sum()}/{len(dq)} | L sel {df.qL.value_counts().to_dict()}",flush=True)
        if cfg.get('noise'): print('   ',df.filter(regex='f1|kmean').mean().round(3).to_dict(),flush=True)
