"""Small geometry kernels for roll measurement: circle fits, line fits, cylinder fits (no part-specific logic)."""
from __future__ import annotations

import math

import numpy as np


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def frame(u):
    """Right-handed orthonormal (e1, e2) perpendicular to u (same convention as z3 slicer)."""
    u = unit(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(u, a))
    e2 = np.cross(u, e1)
    return e1, e2


def fit_circle_kasa(P):
    P = np.asarray(P, float)
    m = P.mean(0)
    Q = P - m
    A = np.c_[2 * Q, np.ones(len(Q))]
    b = (Q ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 1e-12))
    return c + m, r


def fit_circle(P, iters: int = 30, w=None):
    """Geometric (orthogonal-distance) circle fit, Gauss-Newton from the Kasa start.
    Returns centre (2,), radius, rms residual."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return None, float("nan"), float("nan")
    c, r = fit_circle_kasa(P)
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    for _ in range(iters):
        d = P - c
        rho = np.linalg.norm(d, axis=1)
        rho = np.maximum(rho, 1e-12)
        res = rho - r
        J = np.c_[-d / rho[:, None], -np.ones(len(P))]
        step, *_ = np.linalg.lstsq(J * sw[:, None], -res * sw, rcond=None)
        c = c + step[:2]
        r = r + step[2]
        if np.abs(step).max() < 1e-7 * max(1.0, abs(r)):
            break
    res = np.linalg.norm(P - c, axis=1) - r
    return c, float(abs(r)), float(math.sqrt(np.mean(res ** 2)))


def fit_concentric(Pa, Pb):
    """Two concentric circles (shared centre) fitted jointly: returns centre, ra, rb, rms."""
    ca, ra, _ = fit_circle(Pa)
    cb, rb, _ = fit_circle(Pb)
    c = 0.5 * (ca + cb)
    P = np.vstack([Pa, Pb])
    lab = np.r_[np.zeros(len(Pa)), np.ones(len(Pb))]
    for _ in range(30):
        d = P - c
        rho = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        rr = np.where(lab == 0, ra, rb)
        res = rho - rr
        J = np.c_[-d / rho[:, None], -(lab == 0).astype(float), -(lab == 1).astype(float)]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step[:2]
        ra += step[2]
        rb += step[3]
        if np.abs(step).max() < 1e-8:
            break
    rho = np.linalg.norm(P - c, axis=1)
    res = rho - np.where(lab == 0, ra, rb)
    return c, float(ra), float(rb), float(math.sqrt(np.mean(res ** 2)))


def fit_line3(X, w=None):
    """TLS 3-D line: point, unit direction, rms orthogonal residual."""
    X = np.asarray(X, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    c = (X * w[:, None]).sum(0) / w.sum()
    Y = (X - c) * np.sqrt(w)[:, None]
    _, s, vt = np.linalg.svd(Y, full_matrices=False)
    d = vt[0]
    r = X - c - np.outer((X - c) @ d, d)
    return c, d, float(math.sqrt(np.mean((r ** 2).sum(1))))


def angle_of(v2):
    return np.arctan2(v2[..., 1], v2[..., 0])


def angular_extent(phi):
    """Smallest arc (start, sweep) in radians that contains all angles phi (handles wrap)."""
    a = np.sort(np.mod(phi, 2 * math.pi))
    if len(a) == 0:
        return 0.0, 0.0
    gaps = np.diff(np.r_[a, a[0] + 2 * math.pi])
    k = int(np.argmax(gaps))
    start = a[(k + 1) % len(a)]
    sweep = 2 * math.pi - gaps[k]
    return float(start), float(sweep)


def wrap_from(phi, start):
    return np.mod(np.asarray(phi) - start, 2 * math.pi)


def cylinder_residual(P, p0, u, R):
    d = P - p0
    r = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
    return r - R


def fit_cylinder(P, p0, u, R, iters: int = 40):
    """Nonlinear least-squares cylinder (axis point p0, direction u, radius R) by Gauss-Newton on a
    4-parameter axis (two angles, two offsets in the axis-normal plane) + R. Returns p0, u, R, rms."""
    P = np.asarray(P, float)
    p0 = np.asarray(p0, float).copy()
    u = unit(u)
    for _ in range(iters):
        e1, e2 = frame(u)
        x = np.zeros(5)

        def res(x):
            uu = unit(u + x[0] * e1 + x[1] * e2)
            pp = p0 + x[2] * e1 + x[3] * e2
            return cylinder_residual(P, pp, uu, R + x[4])

        r0 = res(x)
        J = np.empty((len(P), 5))
        h = np.array([1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
        for k in range(5):
            xx = x.copy()
            xx[k] = h[k]
            J[:, k] = (res(xx) - r0) / h[k]
        step, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        u = unit(u + step[0] * e1 + step[1] * e2)
        p0 = p0 + step[2] * e1 + step[3] * e2
        R = R + step[4]
        if np.abs(step[:2]).max() < 1e-9 and np.abs(step[2:]).max() < 1e-6:
            break
    r = cylinder_residual(P, p0, u, R)
    return p0, u, float(R), float(math.sqrt(np.mean(r ** 2)))
