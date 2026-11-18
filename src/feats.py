import numpy as np, pandas as pd
from data import load
FEATS=['dc_norm','p_exp','residual','acdc','zscore','IRRADIATION','MODULE_TEMPERATURE','AMBIENT_TEMPERATURE']
def mm(s): 
    r=s.max()-s.min(); return (s-s.min())/(r if r>0 else 1)
def build(dc_fix=True, thr_res=-0.35, acdc_lo=0.85, acdc_hi=1.02, zthr=3.0, center=False, variant="A", minp=4, ddof=0, shift=0):
    out=[]
    for p in (1,2):
        d=load(p); d=d[(d.IRRADIATION>0.05)&(d.DC_POWER>0)].copy()
        if p==1 and dc_fix: d['DC_POWER']=d.DC_POWER/10
        d=d.sort_values(['SOURCE_KEY','DATE_TIME'])
        d['acdc']=d.AC_POWER/d.DC_POWER
        raw=d.IRRADIATION*(1-0.004*(d.MODULE_TEMPERATURE-25))
        d['_raw']=raw
        g=d.groupby('SOURCE_KEY')
        d['dc_norm']=g.DC_POWER.transform(mm)
        if variant=='A': d['p_exp']=g._raw.transform(mm)
        elif variant=='C':
            d['p_exp']=d._raw*g.apply(lambda s:(s.dc_norm/s._raw).median()).reindex(d.SOURCE_KEY).values
        elif variant=='D':
            d['p_exp']=d._raw/d._raw.max()
        elif variant=='E':
            d['dc_norm']=mm(d.DC_POWER); d['p_exp']=mm(d._raw)
        elif variant=='F':
            d['dc_norm']=d.DC_POWER/d.DC_POWER.max(); d['p_exp']=d._raw/d._raw.max()
        elif variant=='G':
            d['dc_norm']=g.DC_POWER.transform(lambda s:s/s.max()); d['p_exp']=g._raw.transform(lambda s:s/s.max())
        elif variant=='H':
            d['dc_norm']=mm(d.DC_POWER); d['p_exp']=d._raw/d._raw.max()
        else:  # c fitted per inverter by least squares on normalized dc
            d['p_exp']=0.
            for k,ix in g.groups.items():
                x=d.loc[ix,'_raw']; y=d.loc[ix,'dc_norm']; c=(x*y).sum()/(x*x).sum(); d.loc[ix,'p_exp']=c*x
        d['residual']=d.dc_norm-d.p_exp
        g=d.groupby('SOURCE_KEY')
        rm=g.residual.transform(lambda s:s.shift(shift).rolling(16,min_periods=minp,center=center).mean())
        rs=g.residual.transform(lambda s:s.shift(shift).rolling(16,min_periods=minp,center=center).std(ddof=ddof))
        d['zscore']=(d.residual-rm)/rs
        d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=['zscore'])
        r1=d.residual<thr_res; r2=(d.acdc<acdc_lo)|(d.acdc>acdc_hi); r3=d.zscore.abs()>zthr
        d['r1'],d['r2'],d['r3']=r1,r2,r3
        d['y']=(r1|r2|r3).astype(int)
        out.append(d.drop(columns='_raw'))
    return pd.concat(out,ignore_index=True)
if __name__=='__main__':
    for v in 'AB':
        d=build(variant=v)
        for p in (1,2):
            s=d[d.plant==p]; print(v,p,len(s),s.y.sum(),s.r1.sum(),s.r2.sum(),s.r3.sum())
