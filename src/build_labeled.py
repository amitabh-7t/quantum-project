# Build data/labeled.pkl from the Kaggle CSVs in data/.
# This is the cached table used by the experiment scripts (features, weak labels, hour, day).
from common import get, LABELED
LABELED.parent.mkdir(parents=True, exist_ok=True)
d = get()
d.to_pickle(LABELED)
print(f'wrote {LABELED} rows={len(d)} anomalies={int(d.y.sum())}')
