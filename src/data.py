import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'
LABELED = DATA_DIR / 'labeled.pkl'
def load(plant, dc_fix=True):
    g=pd.read_csv(DATA_DIR / f'Plant_{plant}_Generation_Data.csv'); w=pd.read_csv(DATA_DIR / f'Plant_{plant}_Weather_Sensor_Data.csv')
    g['DATE_TIME']=pd.to_datetime(g.DATE_TIME, dayfirst=(plant==1), format='%d-%m-%Y %H:%M' if plant==1 else '%Y-%m-%d %H:%M:%S')
    w['DATE_TIME']=pd.to_datetime(w.DATE_TIME)
    m=g.merge(w[['DATE_TIME','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']],on='DATE_TIME',how='inner')
    m['plant']=plant
    return m
if __name__=='__main__':
    for p in (1,2):
        m=load(p); d=m[m.IRRADIATION>0.05]
        r=(d.AC_POWER/d.DC_POWER.replace(0,np.nan))
        print(p,len(m),len(d),d.SOURCE_KEY.nunique(),d.DATE_TIME.dt.date.nunique(),'median AC/DC',r.median())
