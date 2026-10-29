"""Skin planes of base / walls / flanges in the part frame (histogram peaks)."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *
d = np.load(SCR/'a3c'/'pf.npz'); X = d['X']
def peaks(vals, lo, hi, bw=0.05):
    h, ed = np.histogram(vals, bins=np.arange(lo, hi + bw, bw))
    c = 0.5 * (ed[1:] + ed[:-1])
    idx = [i for i in range(1, len(h) - 1) if h[i] >= h[i-1] and h[i] >= h[i+1] and h[i] > 0.15 * h.max()]
    return [(round(c[i], 3), int(h[i])) for i in idx]
# base: interior region away from walls and holes
m = (np.abs(X[:, 0]) < 80) & (np.abs(X[:, 1]) < 70) & (np.abs(X[:, 2]) < 4)
print('base e3 peaks', peaks(X[m, 2], -3, 3))
for nm, cond in [('+e2 wall', X[:, 1] > 80), ('-e2 wall', X[:, 1] < -80)]:
    m = cond & (X[:, 2] > 12) & (X[:, 2] < 32) & (np.abs(X[:, 0]) < 70)
    print(nm, 'e2 peaks', peaks(X[m, 1], -96 if X[m,1].mean()<0 else 86, -86 if X[m,1].mean()<0 else 96))
    print('   e1 extent of wall at mid height (1%,99%)', np.round(np.percentile(X[m, 0], [0, 0.5, 99.5, 100]), 2))
# flanges/tabs at the top
for nm, cond in [('+e2 top', X[:, 1] > 70), ('-e2 top', X[:, 1] < -70)]:
    m = cond & (X[:, 2] > 38)
    print(nm, 'e3 peaks', peaks(X[m, 2], 38, 46))
# end lips
for nm, cond in [('e1+ lip', X[:, 0] > 95), ('e1- lip', X[:, 0] < -95)]:
    m = cond & (X[:, 2] > 3)
    print(nm, 'n', m.sum(), 'e1 peaks', peaks(X[m, 0], -112 if 'e1-' in nm else 95, -95 if 'e1-' in nm else 112),
          'e2 range', np.round(np.percentile(X[m, 1], [0, 100]), 2), 'e3 max', np.round(X[m, 2].max(), 2))
