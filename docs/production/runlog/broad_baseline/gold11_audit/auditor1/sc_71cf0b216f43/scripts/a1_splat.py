"""Auditor-1: shaded point-splat renders of the part-frame scan (Lambert shading from mesh normals)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def view_basis(az, el):
    az, el = np.radians(az), np.radians(el)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # direction from target to eye
    up0 = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([0, 1.0, 0])
    r = np.cross(up0, d); r /= np.linalg.norm(r)
    u = np.cross(d, r)
    return r, u, d

def splat(ax, P, VN, az, el, sel=None, s=0.4, light=None, gold=None, extra=None, title=None, lim=None):
    r, u, d = view_basis(az, el)
    if sel is not None:
        P, VN = P[sel], VN[sel]
    x, y, z = P @ r, P @ u, P @ d
    L = d if light is None else light / np.linalg.norm(light)
    sh = np.abs(VN @ L)  # two-sided
    facing = (VN @ d) > 0
    col = np.where(facing, 0.25 + 0.7 * sh, 0.15 + 0.35 * sh)
    o = np.argsort(z)
    ax.scatter(x[o], y[o], c=col[o], s=s, cmap="gray", vmin=0, vmax=1, linewidths=0, rasterized=True)
    if gold:
        for name, a, b, colr in gold:
            e = np.array([a, b])
            ax.plot(e @ r, e @ u, color=colr, lw=1.6, zorder=5)
            m = 0.5 * (e[0] + e[1])
            ax.text(m @ r, m @ u, name, color=colr, fontsize=8, zorder=6)
    if extra:
        for name, pts, colr in extra:
            pts = np.atleast_2d(pts)
            ax.plot(pts @ r, pts @ u, color=colr, lw=1.2, ls="--", zorder=5)
    ax.set_aspect("equal")
    if lim:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    if title:
        ax.set_title(title, fontsize=9)
    return r, u, d
