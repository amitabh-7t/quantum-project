import sys, json, time
from common import *
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest
def run_full(d, feats, tr, te, which=('if','ocsvm','svc'), log=None):
    X=d[feats].values; y=d.y.values
    sc=StandardScaler().fit(X[tr]); Xtr,Xte=sc.transform(X[tr]),sc.transform(X[te])
    res={}
    if 'if' in which:
        m=IsolationForest(n_estimators=200,contamination=y[tr].mean(),random_state=SEED).fit(Xtr)
        s=-m.score_samples(Xte); res['IF']=metrics(y[te],(m.predict(Xte)==-1).astype(int),s)
    if 'ocsvm' in which:
        m=OneClassSVM(kernel='rbf',gamma='scale',nu=y[tr].mean()).fit(Xtr[y[tr]==0])
        s=-m.decision_function(Xte); res['OCSVM']=metrics(y[te],(m.predict(Xte)==-1).astype(int),s)
    if 'svc' in which:
        t=time.time(); m=SVC(kernel='rbf',class_weight='balanced',gamma='scale',C=1.0,cache_size=500).fit(Xtr,y[tr]); tt=time.time()-t
        t=time.time(); s=m.decision_function(Xte); ti=time.time()-t
        res['SVC']=metrics(y[te],(s>0).astype(int),s); res['SVC']['train_s']=tt; res['SVC']['infer_s']=ti
        res['_svc']=m; res['_sc']=sc
    return res
if __name__=='__main__':
    d=pd.read_pickle(LABELED); tr,te=split(d)
    print(len(tr),len(te))
    r=run_full(d,FEATS,tr,te)
    for k,v in r.items():
        if not k.startswith('_'): print(k,{a:round(b,3) for a,b in v.items()})
