# Full-data classical baselines, per-plant, permutation importance, common test sets, generalisation (full SVC),
# DC-correction ablation, threshold sensitivity.
import json, numpy as np, pandas as pd, time
from common import *
from classical_full import run_full
from matched import run, boot
from qkernel import states, kernel_from_states
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from exp_seeds import chrono
R={}
d=pd.read_pickle(LABELED); tr,te=split(d); y=d.y.values
# rule overlap & prevalence
R['prevalence']={p:dict(n=int((d.plant==p).sum()),anom=int(d.y[d.plant==p].sum()),rate=float(d.y[d.plant==p].mean()),
   r1=int(d.r1[d.plant==p].sum()),r2=int(d.r2[d.plant==p].sum()),r3=int(d.r3[d.plant==p].sum())) for p in (1,2)}
R['prevalence']['all']=dict(n=len(d),anom=int(d.y.sum()),rate=float(d.y.mean()))
print(R['prevalence'],flush=True)
# 1) full baselines, 3 feature sets
full={}
for nm,fs in [('all',FEATS),('noleak',NOLEAK),('raw',RAW)]:
    r=run_full(d,fs,tr,te); full[nm]={k:v for k,v in r.items() if not k.startswith('_')}
    print('full',nm,{k:{a:round(b,3) for a,b in v.items()} for k,v in full[nm].items()},flush=True)
    if nm=='all': svc,sc=r['_svc'],r['_sc']
R['full']=full
# per plant
s=svc.decision_function(sc.transform(d[FEATS].values[te]))
R['perplant']={p:dict(n=int((d.plant.values[te]==p).sum()),**metrics(y[te][d.plant.values[te]==p],(s>0).astype(int)[d.plant.values[te]==p],s[d.plant.values[te]==p])) for p in (1,2)}
print('perplant',R['perplant'],flush=True)
# day-block bootstrap CI for full SVC F1 (vs 0) - report CI only
# permutation importance
rng=np.random.default_rng(SEED); sub=rng.choice(te,2000,replace=False)
pi=permutation_importance(svc,sc.transform(d[FEATS].values[sub]),y[sub],scoring='f1',n_repeats=10,random_state=SEED)
R['permimp']={f:(float(m),float(sd)) for f,m,sd in zip(FEATS,pi.importances_mean,pi.importances_std)}
print('permimp',R['permimp'],flush=True)
# 2) matched headline seed 42 + common test sets
r=run(d,FEATS,tr,te); itr,ite=r['itr'],r['ite']; L=r['qbest']
R['matched42']=dict(ev=r['ev'],qL=L,q={k:{a:b for a,b in v.items() if a not in('score','pred')} for k,v in r['q'].items()},
                    c={a:b for a,b in r['c'].items() if a not in('score','pred')})
yq=y[ite]; pc,pq=r['c']['pred'],r['q'][L]['pred']
R['matched42']['cm_c']=pd.crosstab(yq,pc).values.tolist(); R['matched42']['cm_q']=pd.crosstab(yq,pq).values.tolist()
R['matched42']['boot_iid']=boot(yq,pc,pq); R['matched42']['boot_day']=boot(yq,pc,pq,groups=d.day.values[ite])
R['matched42']['boot_dayinv']=boot(yq,pc,pq,groups=(d.day.values[ite]+d.SOURCE_KEY.values[ite]))
R['matched42']['scores']=dict(y=yq.tolist(),c=r['c']['score'].tolist(),q=r['q'][L]['score'].tolist())
print('matched42',{k:v for k,v in R['matched42'].items() if k!='scores'},flush=True)
# full-data classical models on the same 80 matched test points (8 features)
fullres=run_full(d,FEATS,tr,ite)  # trained on full train split, tested on the 80
R['full_on80']={k:v for k,v in fullres.items() if not k.startswith('_')}
print('full models on matched 80',R['full_on80'],flush=True)
# common natural-prevalence test set: full test split (n=15,228) for QSVM(150) and matched SVC(150) and full models
from sklearn.model_selection import GridSearchCV
f=r['f']; Xtr=f(itr); t=time.time(); Ste=states(f(te),L); Str=states(Xtr,L); tq=time.time()-t
Ktr=kernel_from_states(Str,Str); Kte=kernel_from_states(Ste,Str)
mq=SVC(kernel='precomputed',C=r['q'][L]['C'],class_weight='balanced').fit(Ktr,y[itr]); sq=mq.decision_function(Kte)
mc=SVC(kernel='rbf',class_weight='balanced',**r['c']['params']).fit(Xtr,y[itr]); scc=mc.decision_function(f(te))
R['common_full']=dict(n=len(te),prev=float(y[te].mean()),qsvm=metrics(y[te],(sq>0).astype(int),sq),svc150=metrics(y[te],(scc>0).astype(int),scc),
     full=full['all'],state_time_s=tq)
R['common_full']['boot_day']=boot(y[te],(scc>0).astype(int),(sq>0).astype(int),B=2000,groups=d.day.values[te])
print('common full test',R['common_full'],flush=True)
# 3) generalisation for full SVC
gen={}
a,b,lastday,cut=chrono(d)
for nm,(A,B) in {'chrono':(a,b),'p1to2':(np.where(d.plant==1)[0],np.where(d.plant==2)[0]),'p2to1':(np.where(d.plant==2)[0],np.where(d.plant==1)[0])}.items():
    for fn,fs in [('all',FEATS),('noleak',NOLEAK),('raw',RAW)]:
        rr=run_full(d,fs,A,B,which=('if','svc')); gen[f'{nm}_{fn}']=dict(ntr=len(A),nte=len(B),prev_te=float(y[B].mean()),**{k:v for k,v in rr.items() if not k.startswith('_')})
        print('gen',nm,fn,{k:(round(v['f1'],3),round(v['auc'],3)) for k,v in rr.items() if not k.startswith('_')},flush=True)
R['gen']=gen; R['chrono_cut']=(lastday,cut)
# 4) DC correction ablation
dn=get(dc_fix=False); 
R['dc_nofix']={p:dict(rate=float(dn.y[dn.plant==p].mean()),r2=float(dn.r2[dn.plant==p].mean()),med_acdc=float(dn.acdc[dn.plant==p].median())) for p in (1,2)}
R['dc_fix_medacdc']={p:float(d.acdc[d.plant==p].median()) for p in (1,2)}
# which features change: compare
R['dc_feat_change']={f:float(np.abs(dn[f].values-d[f].values).max()) for f in FEATS} if len(dn)==len(d) else 'len differ'
t2,e2=split(dn); rr=run_full(dn,FEATS,t2,e2,which=('svc',)); R['dc_nofix']['svc']=rr['SVC']
print('dc nofix',R['dc_nofix'],R['dc_feat_change'],flush=True)
# 5) threshold sensitivity
sens=[]
for nm,kw in [('base',{}),('res-0.25',dict(thr_res=-0.25)),('res-0.45',dict(thr_res=-0.45)),('acdc0.80-1.05',dict(acdc_lo=0.80,acdc_hi=1.05)),
              ('acdc0.90-1.01',dict(acdc_lo=0.90,acdc_hi=1.01)),('z2.5',dict(zthr=2.5)),('z3.5',dict(zthr=3.5))]:
    ds=get(**kw); a2,b2=split(ds); rr=run_full(ds,FEATS,a2,b2,which=('svc',)); rn=run_full(ds,NOLEAK,a2,b2,which=('svc',))
    sens.append(dict(cfg=nm,rate=float(ds.y.mean()),anom=int(ds.y.sum()),svc_f1=rr['SVC']['f1'],svc_auc=rr['SVC']['auc'],noleak_f1=rn['SVC']['f1']))
    print('sens',sens[-1],flush=True)
R['sens']=sens
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
json.dump(R,open(RESULTS_DIR / 'results_full.json','w'),indent=1,default=lambda o: o.item() if hasattr(o,'item') else str(o))
print('DONE')
