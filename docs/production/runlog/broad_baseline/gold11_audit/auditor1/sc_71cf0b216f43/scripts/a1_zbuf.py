"""Auditor-1: z-buffered, normal-shaded point renders (opaque surface look) of the part-frame scan."""
import numpy as np

def view_basis(az, el):
    az, el = np.radians(az), np.radians(el)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # target -> eye
    up0 = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([0, 1.0, 0])
    r = np.cross(up0, d); r /= np.linalg.norm(r)
    u = np.cross(d, r)
    return r, u, d

def render(P, VN, az, el, px=0.15, splat=2, sel=None, lights=None, bbox=None):
    """Return (img RGB float, extent, (r,u,d)). Pixel size px mm; each point drawn as splat x splat block."""
    r, u, d = view_basis(az, el)
    if sel is not None:
        P, VN = P[sel], VN[sel]
    x, y, z = P @ r, P @ u, P @ d
    if bbox is None:
        x0, x1, y0, y1 = x.min() - 1, x.max() + 1, y.min() - 1, y.max() + 1
    else:
        x0, x1, y0, y1 = bbox
    W = int((x1 - x0) / px) + 1; Hh = int((y1 - y0) / px) + 1
    zb = np.full((Hh, W), -np.inf); ib = np.full((Hh, W), -1, np.int64)
    ix = ((x - x0) / px).astype(np.int64); iy = ((y1 - y) / px).astype(np.int64)
    order = np.argsort(z)  # far -> near; later writes win
    h = splat // 2
    for dx in range(-h, splat - h):
        for dy in range(-h, splat - h):
            jx = ix[order] + dx; jy = iy[order] + dy
            ok = (jx >= 0) & (jx < W) & (jy >= 0) & (jy < Hh)
            jx, jy, oo = jx[ok], jy[ok], order[ok]
            zz = z[oo]
            better = zz > zb[jy, jx]
            # with sorted order, assignment keeps the last (nearest) duplicate
            zb[jy[better], jx[better]] = zz[better]
            ib[jy[better], jx[better]] = oo[better]
    if lights is None:
        lights = [d + 0.5 * u + 0.3 * r]
    img = np.ones((Hh, W, 3))
    m = ib >= 0
    n = VN[ib[m]]
    n = n * np.sign((n @ d))[:, None]  # face the viewer
    sh = np.zeros(len(n))
    for L in lights:
        L = L / np.linalg.norm(L)
        sh += np.clip(n @ L, 0, 1)
    sh = 0.15 + 0.85 * sh / len(lights)
    img[m] = np.c_[sh, sh, sh]
    return img, (x0, x1, y0, y1), (r, u, d)

def project(pts, basis):
    r, u, d = basis
    pts = np.atleast_2d(pts)
    return pts @ r, pts @ u
