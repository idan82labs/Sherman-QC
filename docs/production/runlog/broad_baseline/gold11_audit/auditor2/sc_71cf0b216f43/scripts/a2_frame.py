import numpy as np
from a2_load import *

def part_frame(V):
    # PCA of the vertices: PC1 long, PC2 short, PC3 normal
    c = V.mean(0)
    X = V - c
    w, U = np.linalg.eigh(np.cov(X[::7].T))
    U = U[:, ::-1]
    if np.linalg.det(U) < 0: U[:, 2] *= -1
    return c, U  # columns e1,e2,e3

def to_frame(P, c, U):
    return (np.atleast_2d(P) - c) @ U

def refined_frame(V, N):
    """Frame aligned to the part's own planes: e3 = face normal, e1 = long walls' in-plane direction."""
    c, U = part_frame(V)
    P = to_frame(V, c, U); Nn = N @ U
    # face normal: average of (sign-aligned) normals with |n3|>0.97
    m = np.abs(Nn[:, 2]) > 0.97
    n3 = (Nn[m] * np.sign(Nn[m, 2])[:, None]).mean(0); n3 /= np.linalg.norm(n3)
    # long walls (normal ~ PC2): fit normal direction
    m2 = np.abs(Nn[:, 1]) > 0.95
    n2 = (Nn[m2] * np.sign(Nn[m2, 1])[:, None]).mean(0)
    n2 -= n3 * (n2 @ n3); n2 /= np.linalg.norm(n2)
    n1 = np.cross(n2, n3)
    R = np.stack([n1, n2, n3], 1)  # in PCA coords
    U2 = U @ R
    return c, U2
