import json, ast, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from common import *
from classical_full import run_full
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'font.family':'serif'})
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
O=str(FIGURES_DIR)+'/'
R=json.load(open(RESULTS_DIR / 'results_full.json'))
# Fig 1 pipeline
fig,ax=plt.subplots(figsize=(7.2,2.5)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(-0.5,4)
def box(x,y,w,h,t,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',fc=fc,ec='0.3',lw=1)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7.6)
top=[('SCADA logs\n2 plants, 44 inverters\n15-min, 34 days','0.95'),('Daylight filter,\nPlant 1 DC/10,\n8 features','#e8f0fb'),('Weak labels:\nresidual, AC/DC,\ntrailing |z| > 3','#e8f0fb'),('Splits: random,\nchronological,\ncross-plant','#e8f0fb')]
for i,(t,c) in enumerate(top):
    box(0.1+i*2.55,2.6,2.15,1.25,t,c)
    if i<3: ax.annotate('',xy=(0.1+(i+1)*2.55-0.07,3.22),xytext=(0.1+i*2.55+2.22,3.22),arrowprops=dict(arrowstyle='->',lw=1.2))
box(0.3,0.15,4.2,1.6,'Classical branch\nIsolation Forest, One-Class SVM,\nRBF-SVM (5-fold CV over C, gamma)','#fcefe3')
box(5.4,0.15,4.4,1.6,'Quantum branch (150/80 subsample)\nPCA to 4 components, angles in [0, pi]\nZZ / IQP / Z-only maps, L = 1-4 (CV)','#e9f5ea')
ax.annotate('',xy=(2.4,1.85),xytext=(8.9,2.5),arrowprops=dict(arrowstyle='->',lw=1.0,color='0.4'))
ax.annotate('',xy=(7.6,1.85),xytext=(8.9,2.5),arrowprops=dict(arrowstyle='->',lw=1.0,color='0.4'))
ax.text(5.0,-0.25,'Evaluation: P / R / F1 / ROC-AUC, 20 repeated subsamples, day-block bootstrap, compute cost',ha='center',fontsize=8,style='italic')
fig.savefig(O+'fig_pipeline.pdf',bbox_inches='tight'); plt.close()
# Fig 2 depth / feature-map sweep (20 seeds)
fig,ax=plt.subplots(figsize=(3.45,2.3))
for nm,lab,mk,ls in [('zz_L1to4','ZZ ring (paper map)','o','-'),('iqp_allpairs','IQP all-pairs','s','--'),('zonly','Z-only (no entanglement)','^',':')]:
    df=pd.read_csv(RESULTS_DIR / f'res_seeds_{nm}.csv'); q=pd.DataFrame([ast.literal_eval(x) for x in df.q_perL]); Ls=[int(c) for c in q.columns]
    ax.errorbar(Ls,q.mean(),yerr=q.std(),marker=mk,ls=ls,capsize=2,label=lab)
c=pd.read_csv(RESULTS_DIR / 'res_seeds_base.csv').c_f1; ax.axhline(c.mean(),color='k',lw=0.8); ax.text(2.0,c.mean()+0.015,'RBF-SVM (CV-tuned)',fontsize=7.5)
ax.set_xlabel('Feature-map layers L'); ax.set_ylabel('Test F1 (matched set, n = 80)'); ax.set_xticks([1,2,3,4]); ax.set_ylim(0.4,1.0); ax.legend(loc='lower left',frameon=False)
ax.grid(alpha=0.3); fig.savefig(O+'fig_layer_sweep.pdf',bbox_inches='tight'); plt.close()
# Fig 3 confusion matrices seed 42
m=R['matched42']; fig,axs=plt.subplots(1,2,figsize=(3.6,1.9))
for a,cm,t in zip(axs,[m['cm_c'],m['cm_q']],['RBF-SVM','QSVM (ZZ, L = 2)']):
    cm=np.array(cm); a.imshow(cm,cmap='Blues',vmin=0,vmax=55)
    for i in range(2):
        for j in range(2): a.text(j,i,cm[i,j],ha='center',va='center',fontsize=10,color='white' if cm[i,j]>30 else 'black')
    a.set_xticks([0,1],['Normal','Anom.'],fontsize=7.5); a.set_yticks([0,1],['Normal','Anom.'],fontsize=7.5); a.set_title(t,fontsize=9); a.set_xlabel('Predicted label')
axs[0].set_ylabel('True label'); fig.tight_layout(); fig.savefig(O+'fig_confusion.pdf',bbox_inches='tight'); plt.close()
# Fig 4 ROC
d=pd.read_pickle(LABELED); tr,te=split(d); y=d.y.values; X=d[FEATS].values; sc=StandardScaler().fit(X[tr]); A,B=sc.transform(X[tr]),sc.transform(X[te])
s_if=-IsolationForest(n_estimators=200,contamination=y[tr].mean(),random_state=SEED).fit(A).score_samples(B)
s_oc=-OneClassSVM(kernel='rbf',gamma='scale',nu=y[tr].mean()).fit(A[y[tr]==0]).decision_function(B)
s_sv=SVC(kernel='rbf',class_weight='balanced').fit(A,y[tr]).decision_function(B)
fig,axs=plt.subplots(1,2,figsize=(7.0,2.6))
for s,lab,ls in [(s_if,'Isolation Forest','-.'),(s_oc,'One-Class SVM',':'),(s_sv,'SVC-RBF (supervised)','-')]:
    f,t,_=roc_curve(y[te],s); axs[0].plot(f,t,ls=ls,lw=1.6,label=f'{lab}, AUC = {roc_auc_score(y[te],s):.3f}')
sm=m['scores']; ym=np.array(sm['y'])
for s,lab,ls,col in [(sm['c'],'RBF-SVM (CV-tuned)','-','C0'),(sm['q'],'QSVM (ZZ, L = 2)','--','C3')]:
    f,t,_=roc_curve(ym,s); axs[1].plot(f,t,ls=ls,lw=1.8,color=col,label=f'{lab}, AUC = {roc_auc_score(ym,s):.3f}')
for a,tt in zip(axs,['(a) Full test set, n = 15,228','(b) Matched test set, n = 80']):
    a.plot([0,1],[0,1],color='0.6',lw=0.8,ls=(0,(1,3)),label='Chance'); a.set_xlabel('False positive rate'); a.set_ylabel('True positive rate'); a.set_title(tt,fontsize=9); a.legend(loc='lower right',frameon=True); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(O+'fig_roc.pdf',bbox_inches='tight'); plt.close()
# Fig 5 permutation importance
pi=R['permimp']; names={'dc_norm':'Normalized DC power','p_exp':'Expected power','residual':'Performance residual*','acdc':'AC/DC ratio*','zscore':'Rolling z-score*','IRRADIATION':'Irradiation','MODULE_TEMPERATURE':'Module temperature','AMBIENT_TEMPERATURE':'Ambient temperature'}
k=sorted(pi,key=lambda x:pi[x][0]); fig,ax=plt.subplots(figsize=(3.45,2.2))
ax.barh([names[x] for x in k],[pi[x][0] for x in k],xerr=[pi[x][1] for x in k],color=['C3' if '*' in names[x] else 'C0' for x in k],capsize=2)
ax.set_xlabel('Drop in F1 when feature is permuted'); ax.grid(axis='x',alpha=0.3); fig.savefig(O+'fig_feature_importance.pdf',bbox_inches='tight'); plt.close()
print('ok')
