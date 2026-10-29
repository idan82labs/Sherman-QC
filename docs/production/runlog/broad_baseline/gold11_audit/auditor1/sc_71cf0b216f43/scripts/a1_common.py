"""Auditor-1 common: scan in part frame + gold lines in part frame."""
import numpy as np
from a1_load import load, gold_lines, OUT
FR = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_71cf/frame.npz"

def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def get():
    V, N, F = load()
    fr = np.load(FR); R, c = fr["R"], fr["c"]
    P = (V - c) @ R.T
    Nn = N @ R.T
    # face normals in frame, assign area-weighted vertex normals for robustness
    a, b, cc = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    fn = np.cross(b - a, cc - a)
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    nrm = np.linalg.norm(vn, axis=1); nrm[nrm == 0] = 1
    vn /= nrm[:, None]
    G = [(bid, (s - c) @ R.T, (e - c) @ R.T) for bid, s, e in gold_lines()]
    return P, vn, F, G, R, c

def to_raw(p, R, c):
    return np.asarray(p) @ R + c
