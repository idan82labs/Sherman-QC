"""Auditor-1: part frame (x = long axis, y = short axis, z = flange normal) from area-weighted face normals."""
import numpy as np
from a1_load import load, gold_lines

def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def part_frame(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a); ar = np.linalg.norm(fn, axis=1) / 2
    fnu = fn / (2 * ar)[:, None]
    seeds = {"Z": np.array([-0.2575, 0.3321, -0.9074]), "Y": np.array([0.7585, -0.5098, -0.4059]),
             "X": np.array([0.5982, 0.7919, 0.1229])}
    out = {}
    for k, ax in seeds.items():
        for _ in range(3):
            cs = fnu @ ax; m = np.abs(cs) > np.cos(np.radians(10)); sg = np.sign(cs[m])[:, None]
            A = fnu[m] * sg * np.sqrt(ar[m])[:, None]
            ev, evec = np.linalg.eigh(A.T @ A); ax2 = evec[:, -1]; ax = ax2 * np.sign(ax2 @ ax)
        out[k] = ax
    Z = unit(out["Z"]); Y = unit(out["Y"] - (out["Y"] @ Z) * Z); X = np.cross(Y, Z)
    if X @ out["X"] < 0: X = -X; Y = -Y  # keep right-handed with X ~ long axis
    # re-check right-handedness
    R = np.array([X, Y, Z])
    if np.linalg.det(R) < 0: Z = -Z; R = np.array([X, Y, Z])
    c = V.mean(0)
    return R, c, out

if __name__ == "__main__":
    V, N, F = load()
    R, c, raw = part_frame(V, F)
    print("R", R, "det", np.linalg.det(R), "c", c)
    for k in "XYZ": print(k, "raw", raw[k])
    print("angle raw X vs frame X %.3f deg" % np.degrees(np.arccos(abs(raw['X'] @ R[0]))))
    P = (V - c) @ R.T
    print("extent", P.min(0), P.max(0))
    for bid, s, e in gold_lines():
        ps, pe = (s - c) @ R.T, (e - c) @ R.T
        d = unit(pe - ps)
        print(bid, np.round(ps, 2), np.round(pe, 2), "dir", np.round(d, 4))
    np.savez("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_71cf/frame.npz", R=R, c=c)
