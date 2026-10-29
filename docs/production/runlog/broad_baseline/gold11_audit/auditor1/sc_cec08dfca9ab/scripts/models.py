"""Auditor-1: which displacement model explains gold vs apex? translation / similarity / per-axis scale / constant concave offset."""
import json
import numpy as np
from scipy.optimize import least_squares
from a1frame import *
xyz, nrm, faces, R, c, G = load()
d = json.load(open(W + "fold_extent_all.json"))
# correspondences: on fold stations within the gold span, gold point (part) and apex point (part) and bisector (part, 3D)
from corner import corner_at, PLATES, gp
pairs = []
for name in d:
    a, b = gp[name]; L = np.linalg.norm(b - a); u = (b - a) / L
    ca = np.array(d[name]["apex_line_point_part"]); ua = np.array(d[name]["apex_line_dir_part"])
    rows = [r for r in d[name]["rows"] if r["ok"] and r.get("apex_dev", 9) <= 0.4 and 0 <= r.get("gold_t", -1) <= 1]
    for r in rows[::10]:
        q = ca + r["s"] * ua
        rr = corner_at(q, ua, *PLATES[name], expect=np.zeros(2), max_from_expect=3.0)
        if rr.get("status") != "corner":
            continue
        bis3 = rr["bis"][0] * rr["e1"] + rr["bis"][1] * rr["e2"]
        gq = a + (r["gold_t"] * L) * u
        pairs.append((name, gq, rr["apex3"], bis3, ua))
names = [p[0] for p in pairs]
Gp = np.array([p[1] for p in pairs]); Ap = np.array([p[2] for p in pairs]); B3 = np.array([p[3] for p in pairs]); U = np.array([p[4] for p in pairs])
print("n pairs", len(pairs), {n: names.count(n) for n in sorted(set(names))})

def perp_res(Gt):
    # residual = component of (transformed gold - apex) perpendicular to fold direction (along-fold slip is free)
    dvec = Gt - Ap
    return dvec - (np.sum(dvec * U, 1))[:, None] * U

def report(tag, Gt):
    r = np.linalg.norm(perp_res(Gt), axis=1)
    per = {n: round(float(np.median(r[[i for i, x in enumerate(names) if x == n]])), 2) for n in sorted(set(names))}
    print(f"{tag:<40} rms {np.sqrt(np.mean(r**2)):.2f}  max {r.max():.2f}  per-line median {per}")
    return r

report("identity (as labelled)", Gp)
# translation
f = lambda p: perp_res(Gp + p[:3]).ravel()
t = least_squares(f, np.zeros(3)).x; report(f"translation {np.round(t,2)}", Gp + t)
# similarity (scale s, rotation vec w, translation t) about centroid
cg = Gp.mean(0)
def sim(p, X):
    s, w, t = p[0], p[1:4], p[4:7]
    th = np.linalg.norm(w)
    if th < 1e-12: Rm = np.eye(3)
    else:
        k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return (s * (X - cg) @ Rm.T) + cg + t
ps = least_squares(lambda p: perp_res(sim(p, Gp)).ravel(), np.r_[1, 0, 0, 0, 0, 0, 0]).x
report(f"similarity s={ps[0]:.4f} rot={np.degrees(np.linalg.norm(ps[1:4])):.2f}deg", sim(ps, Gp))
# per-axis scale (part frame) + translation
pa = least_squares(lambda p: perp_res((Gp - cg) * p[:3] + cg + p[3:]).ravel(), np.r_[1, 1, 1, 0, 0, 0]).x
report(f"axis scale {np.round(pa[:3],4)} + t", (Gp - cg) * pa[:3] + cg + pa[3:])
# constant concave offset k along each bisector: gold = apex + k*bis  -> estimate k, residual
k = float(np.mean(np.sum((Gp - Ap) * B3, 1)))
report(f"constant concave offset k={k:.2f} mm", Gp - k * B3)
for n in sorted(set(names)):
    idx = [i for i, x in enumerate(names) if x == n]
    kn = np.sum((Gp[idx] - Ap[idx]) * B3[idx], 1)
    print(f"   {n}: concave offset along bisector mean {kn.mean():.2f} sd {kn.std():.2f}")
