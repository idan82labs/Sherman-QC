"""Vectorised mesh-plane sections in the part frame + 90-deg corner analysis (auditor 3)."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *

_d = None
def data():
    global _d
    if _d is None:
        z = np.load(SCR/'a3c'/'pf.npz'); _d = dict(X=z['X'], F=z['F'])
    return _d

class Slicer:
    """Precompute per-triangle min/max along an axis for fast slicing by axis-aligned planes."""
    def __init__(self, axis, box=None):
        d = data(); X, F = d['X'], d['F']
        if box is not None:
            c = X[F].mean(1)
            m = np.ones(len(F), bool)
            for ax, (lo, hi) in box.items():
                m &= (c[:, ax] > lo) & (c[:, ax] < hi)
            F = F[m]
        self.X, self.F, self.axis = X, F, axis
        dd = X[F][:, :, axis]
        self.lo, self.hi = dd.min(1), dd.max(1)
        self.dd = dd
    def section(self, s):
        """Return intersection points (k,3) of plane x[axis]=s with mesh (segment endpoints, 2 per tri)."""
        m = (self.lo < s) & (self.hi > s)
        T = self.X[self.F[m]]; dd = self.dd[m] - s
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da, db = dd[:, a], dd[:, b]
            c = (da > 0) != (db > 0)
            t = da[c] / (da[c] - db[c])
            pts.append(T[c, a] + t[:, None] * (T[c, b] - T[c, a]))
        return np.concatenate(pts) if pts else np.zeros((0, 3))

def fit_line2(P):
    """Total least squares 2D line: returns point c and unit direction u."""
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0]

def intersect2(c1, u1, c2, u2):
    A = np.array([u1, -u2]).T
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * u1

def fit_circle(P):
    A = np.c_[2 * P, np.ones(len(P))]; b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]; r = np.sqrt(sol[2] + c @ c)
    return c, r
