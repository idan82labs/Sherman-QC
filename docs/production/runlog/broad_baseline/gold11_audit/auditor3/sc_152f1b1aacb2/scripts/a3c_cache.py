"""Cache raw mesh + part-frame vertices for auditor-3 analysis (read-only on scan/gold)."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *
V, F = load_mesh()
R, o = part_frame(V)
X = to_part(V, R, o)
G = load_gold()
np.savez(SCR/'a3c'/'pf.npz', V=V, F=F, X=X, R=R, o=o)
print('V', V.shape, 'F', F.shape)
print('R rows (e1,e2,e3) =\n', np.round(R, 5), '\norigin', np.round(o, 4))
print('part-frame bbox', np.round(X.min(0), 2), np.round(X.max(0), 2))
for bid, s, e in G:
    a, b = to_part(s, R, o), to_part(e, R, o)
    d = b - a; L = np.linalg.norm(d)
    print(bid, 'start', np.round(a, 2), 'end', np.round(b, 2), 'len %.1f' % L, 'dir', np.round(d / L, 3))
