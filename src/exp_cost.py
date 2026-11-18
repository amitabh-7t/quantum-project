# Compute cost: kernel construction (pairwise compute-uncompute circuits and statevector shortcut), training, inference, peak memory; scaling in n.
import json, time, tracemalloc, platform, subprocess, numpy as np, pandas as pd, sklearn, pennylane, scipy
from common import *
from matched import prep, cv_precomp, CS, GS
from qkernel import states, kernel_from_states, kernel_pairwise
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
d=pd.read_pickle(LABELED); tr,te=split(d); y=d.y.values
R=dict(hw=subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'",shell=True,capture_output=True,text=True).stdout.strip(),
       mem=subprocess.run("free -g | head -2",shell=True,capture_output=True,text=True).stdout.strip(),
       sw=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pennylane=pennylane.__version__,os=platform.platform()))
def tm(fn):
    tracemalloc.start(); t=time.perf_counter(); out=fn(); dt=time.perf_counter()-t; peak=tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return out,dt,peak/2**20
itr,ite=matched(d,tr,te); f,_=prep(d,FEATS,itr,ite); Xtr,Xte=f(itr),f(ite); ytr=y[itr]
L=2; C=5
# QSVM, pairwise circuits as in the original implementation
(Ktr_p),t_kp_tr,m_kp_tr=tm(lambda: kernel_pairwise(Xtr,Xtr,L,sym=True))
(Kte_p),t_kp_te,m_kp_te=tm(lambda: kernel_pairwise(Xte,Xtr,L))
(Ss),t_ks,m_ks=tm(lambda: (states(Xtr,L),states(Xte,L)))
Ktr=kernel_from_states(Ss[0],Ss[0]); Kte=kernel_from_states(Ss[1],Ss[0])
R['pairwise_vs_state_maxdiff']=float(max(np.abs(Ktr-Ktr_p).max(),np.abs(Kte-Kte_p).max()))
m,t_fit,m_fit=tm(lambda: SVC(kernel='precomputed',C=C,class_weight='balanced').fit(Ktr,ytr))
_,t_inf,_=tm(lambda: m.decision_function(Kte))
_,t_cv,_=tm(lambda: [cv_precomp(kernel_from_states(S,S),ytr) for S in [states(Xtr,l) for l in (1,2,3)]])
R['qsvm']=dict(kernel_pairwise_train_s=t_kp_tr,kernel_pairwise_test_s=t_kp_te,kernel_pairwise_peakMB=max(m_kp_tr,m_kp_te),
  n_circuits=150*151//2+80*150,kernel_statevector_s=t_ks,kernel_statevector_peakMB=m_ks,fit_s=t_fit,infer_s_excl_kernel=t_inf,
  per_sample_infer_ms_pairwise=1000*t_kp_te/80, model_selection_statevector_s=t_cv)
# classical
g,t_gs,m_gs=tm(lambda: GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),{'C':CS,'gamma':GS},scoring='f1',cv=StratifiedKFold(5,shuffle=True,random_state=SEED)).fit(Xtr,ytr))
mc,t_cf,m_cf=tm(lambda: SVC(kernel='rbf',class_weight='balanced',**g.best_params_).fit(Xtr,ytr))
_,t_ci,_=tm(lambda: mc.decision_function(Xte))
R['svc150']=dict(gridsearch_s=t_gs,gridsearch_peakMB=m_gs,fit_s=t_cf,fit_peakMB=m_cf,infer_s=t_ci,per_sample_infer_ms=1000*t_ci/80)
# full SVC on 45,681
X=StandardScaler().fit(d[FEATS].values[tr]); A=X.transform(d[FEATS].values[tr]); B=X.transform(d[FEATS].values[te])
mf,t_ff,m_ff=tm(lambda: SVC(kernel='rbf',class_weight='balanced').fit(A,y[tr]))
_,t_fi,_=tm(lambda: mf.decision_function(B))
R['svc_full']=dict(n_train=len(tr),fit_s=t_ff,fit_peakMB=m_ff,infer_s=t_fi,per_sample_infer_ms=1000*t_fi/len(te))
print(json.dumps(R,indent=1),flush=True)
# scaling in n (training-kernel construction + fit)
rng=np.random.default_rng(0); sc=[]
pool=f(tr)
for n in (50,100,200,400,800):
    idx=rng.choice(len(tr),n,replace=False); Xn=pool[idx]; yn=y[tr][idx]
    if yn.sum()<2: yn[:5]=1
    S,ts,ms=tm(lambda: states(Xn,L)); K=kernel_from_states(S,S)
    _,tf,_=tm(lambda: SVC(kernel='precomputed',C=C,class_weight='balanced').fit(K,yn))
    tp=None
    if n<=100: _,tp,_=tm(lambda: kernel_pairwise(Xn,Xn,L,sym=True))
    _,tc,mc_=tm(lambda: SVC(kernel='rbf',class_weight='balanced',gamma=2,C=0.5).fit(Xn,yn))
    sc.append(dict(n=n,q_statevector_s=ts,q_pairwise_s=tp,q_fit_s=tf,q_peakMB=ms,c_fit_s=tc,c_peakMB=mc_)); print(sc[-1],flush=True)
R['scaling']=sc
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
json.dump(R,open(RESULTS_DIR / 'results_cost.json','w'),indent=1)
print('DONE')
