import numpy as np, time
from common import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from qkernel import kernel, states, kernel_from_states
CS=[0.1,0.5,1,2,5,10,20]; GS=['scale','auto',0.1,0.5,1,2]
def prep(d,feats,itr,ite,ncomp=4):
    X=d[feats].values
    sc=StandardScaler().fit(X[itr]); p=PCA(ncomp,random_state=SEED).fit(sc.transform(X[itr]))
    mm=MinMaxScaler((0,np.pi)).fit(p.transform(sc.transform(X[itr])))
    f=lambda idx: np.clip(mm.transform(p.transform(sc.transform(X[idx]))),0,np.pi)
    return f, p.explained_variance_ratio_.sum()
def cv_precomp(K,y,seed=SEED):
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); best=(-1,None)
    for C in CS:
        f=[]
        for a,b in skf.split(K,y):
            m=SVC(kernel='precomputed',C=C,class_weight='balanced').fit(K[np.ix_(a,a)],y[a]); f.append(f1_score(y[b],m.predict(K[np.ix_(b,a)])))
        if np.mean(f)>best[0]: best=(np.mean(f),C)
    return best
def qsvm(Xtr,ytr,Xte,yte,Ls=(1,2,3),kind='zz',Kfun=None):
    out={}; t0=time.time()
    for L in Ls:
        t=time.time()
        if Kfun is None:
            Str=states(Xtr,L,kind); Ste=states(Xte,L,kind); Ktr=kernel_from_states(Str,Str); Kte=kernel_from_states(Ste,Str)
        else: Ktr,Kte=Kfun(Xtr,Xtr,L),Kfun(Xte,Xtr,L)
        tk=time.time()-t
        cvf,C=cv_precomp(Ktr,ytr)
        m=SVC(kernel='precomputed',C=C,class_weight='balanced').fit(Ktr,ytr); s=m.decision_function(Kte)
        out[L]=dict(C=C,cv_f1=cvf,kernel_s=tk,**metrics(yte,(s>0).astype(int),s),score=s,pred=(s>0).astype(int))
    best=max(out,key=lambda L:out[L]['cv_f1']); return best,out
def csvm(Xtr,ytr,Xte,yte,seed=SEED):
    g=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),{'C':CS,'gamma':GS},scoring='f1',cv=StratifiedKFold(5,shuffle=True,random_state=seed)).fit(Xtr,ytr)
    s=g.best_estimator_.decision_function(Xte)
    return dict(params=g.best_params_,cv_f1=g.best_score_,**metrics(yte,(s>0).astype(int),s),score=s,pred=(s>0).astype(int))
def boot(y,pa,pb,B=5000,groups=None,seed=SEED):
    rng=np.random.default_rng(seed); n=len(y); diffs=[]
    if groups is not None: ug=np.unique(groups); gi={g:np.where(groups==g)[0] for g in ug}
    for _ in range(B):
        if groups is None: i=rng.integers(0,n,n)
        else: i=np.concatenate([gi[g] for g in rng.choice(ug,len(ug),replace=True)])
        diffs.append(f1_score(y[i],pa[i],zero_division=0)-f1_score(y[i],pb[i],zero_division=0))
    diffs=np.array(diffs); return dict(mean=diffs.mean(),lo=np.percentile(diffs,2.5),hi=np.percentile(diffs,97.5),p=(np.sum(diffs<=0)+1)/(B+1))
def run(d,feats,tr,te,seed=SEED,ncomp=4,Ls=(1,2,3),kind='zz',verbose=True):
    itr,ite=matched(d,tr,te,seed=seed); y=d.y.values
    f,ev=prep(d,feats,itr,ite,ncomp); Xtr,Xte=f(itr),f(ite)
    bl,q=qsvm(Xtr,y[itr],Xte,y[ite],Ls,kind); c=csvm(Xtr,y[itr],Xte,y[ite],seed)
    r=dict(itr=itr,ite=ite,ev=ev,qbest=bl,q=q,c=c,f=f)
    if verbose:
        print(f'seed {seed} ncomp {ncomp} var {ev:.3f} anomalies tr {y[itr].mean():.2f} te {y[ite].mean():.2f}')
        for L in q: print('  QSVM L',L,{k:(round(v,3) if isinstance(v,float) else v) for k,v in q[L].items() if k not in('score','pred')})
        print('  SVC',{k:(round(v,3) if isinstance(v,float) else v) for k,v in c.items() if k not in('score','pred')}, ' selected L',bl)
    return r
if __name__=='__main__':
    d=pd.read_pickle(LABELED); tr,te=split(d)
    r=run(d,FEATS,tr,te); y=d.y.values[r['ite']]
    b=boot(y,r['c']['pred'],r['q'][r['qbest']]['pred']); print('iid bootstrap',b)
    days=d.day.values[r['ite']]; print('day-block bootstrap',boot(y,r['c']['pred'],r['q'][r['qbest']]['pred'],groups=days))
    print('n distinct days in test80',len(np.unique(days)))
