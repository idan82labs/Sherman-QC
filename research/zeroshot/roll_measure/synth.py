"""Synthetic rolled sheet-metal scans with known truth (cylinder / cone, tangent flats, flat spot, holes).

The part: inner skin radius R(z) = R0 + taper * (z - L/2) about the z axis, sweep theta from angle 0,
tangent flats of length f1 (before angle 0) and f2 (after theta), outer skin at R(z) + t, sheet-edge walls
joining the skins along the two profile ends, outward winding, Gaussian noise along the normal, then a
random rigid pose. Sampled as a triangle grid at the given spacing (a structured-light scan's mesh
spacing on the real parts is 0.2-0.5 mm; coarser grids are used for big parts to bound memory).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from research.zeroshot.roll_measure import geom as G


@dataclass
class SynthSpec:
    R: float = 100.0            # inner radius at mid-length
    t: float = 2.0
    sweep_deg: float = 90.0
    L: float = 200.0
    f1: float = 30.0
    f2: float = 30.0
    taper: float = 0.0          # dR/dz (mm per mm)
    flat_center_frac: float | None = None   # flat spot centre as a fraction of the sweep
    flat_width_deg: float = 20.0
    flat_depth_mm: float = 0.0
    flat_z: tuple[float, float] = (0.0, 1.0)  # fraction of the length
    noise_mm: float = 0.05
    spacing_mm: float = 0.6
    skins: str = "both"         # both / inner / outer
    hole_frac: tuple[float, float] | None = None  # angular band (fraction of sweep) missing on every skin
    seed: int = 0


def _profile(spec: SynthSpec, r_of_z, z, du, offset):
    """Points of one skin on a (profile, z) grid. offset 0 = inner skin, t = outer skin."""
    th = math.radians(spec.sweep_deg)
    Rm = spec.R + offset
    arc_len = Rm * th
    tot = spec.f1 + arc_len + spec.f2
    n = max(4, int(math.ceil(tot / du)) + 1)
    p = np.linspace(-spec.f1, arc_len + spec.f2, n)
    P = np.zeros((n, len(z), 3))
    for j, zz in enumerate(z):
        r = r_of_z(zz) + offset
        for i, pp in enumerate(p):
            if pp < 0:
                x, y = r, pp
            elif pp > arc_len:
                s = pp - arc_len
                x, y = r * math.cos(th) - s * math.sin(th), r * math.sin(th) + s * math.cos(th)
            else:
                phi = pp / Rm * 1.0
                rr = r
                if spec.flat_center_frac is not None and spec.flat_depth_mm > 0:
                    zf = (zz / spec.L)
                    if spec.flat_z[0] <= zf <= spec.flat_z[1]:
                        c = spec.flat_center_frac * th
                        w = math.radians(spec.flat_width_deg)
                        if abs(phi - c) < w / 2:
                            rr = r - spec.flat_depth_mm * math.cos(math.pi * (phi - c) / w) ** 2
                x, y = rr * math.cos(phi), rr * math.sin(phi)
            P[i, j] = (x, y, zz)
    return P, p, arc_len


def _grid_faces(ni, nj, base, flip):
    i, j = np.meshgrid(np.arange(ni - 1), np.arange(nj - 1), indexing="ij")
    a = base + i * nj + j
    b = a + nj
    c = a + 1
    d = b + 1
    F = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return F[:, ::-1] if flip else F


def make(spec: SynthSpec):
    rng = np.random.default_rng(spec.seed)
    du = spec.spacing_mm
    nz = max(4, int(math.ceil(spec.L / du)) + 1)
    z = np.linspace(0, spec.L, nz)

    def r_of_z(zz):
        return spec.R + spec.taper * (zz - spec.L / 2)

    Vs, Fs, keep_masks = [], [], []
    base = 0
    skins = {"both": ["inner", "outer"], "inner": ["inner"], "outer": ["outer"]}[spec.skins]
    grids = {}
    for sk in skins:
        off = 0.0 if sk == "inner" else spec.t
        P, p, arc_len = _profile(spec, r_of_z, z, du, off)
        ni, nj = P.shape[:2]
        F = _grid_faces(ni, nj, base, flip=False)
        V = P.reshape(-1, 3)
        # orient: inner skin's outward normal points to the axis (-radial), outer skin's away from it
        Fl = F - base
        cr = np.cross(V[Fl[:, 1]] - V[Fl[:, 0]], V[Fl[:, 2]] - V[Fl[:, 0]])
        cen = V[Fl].mean(1)
        rad = cen.copy()
        rad[:, 2] = 0
        want = -1 if sk == "inner" else 1
        if np.sum(np.sum(cr * rad, 1) * want) < 0:
            F = F[:, ::-1]
        # hole: drop faces in an angular band of the arc
        if spec.hole_frac is not None:
            th = math.radians(spec.sweep_deg)
            ang = np.arctan2(cen[:, 1], cen[:, 0])
            h0, h1 = spec.hole_frac[0] * th, spec.hole_frac[1] * th
            F = F[~((ang > h0) & (ang < h1) & (np.hypot(cen[:, 0], cen[:, 1]) > 0))]
        grids[sk] = (base, ni, nj)
        Vs.append(V)
        Fs.append(F)
        base += len(V)
    V = np.vstack(Vs)
    F = np.vstack(Fs)
    # sheet-edge walls along the profile ends (connect inner and outer end rows)
    if len(skins) == 2:
        bi, ni_i, nj_ = grids["inner"]
        bo, ni_o, _ = grids["outer"]
        walls = []
        for ii, io in ((0, 0), (ni_i - 1, ni_o - 1)):
            a = bi + ii * nj_ + np.arange(nj_ - 1)
            b = bo + io * nj_ + np.arange(nj_ - 1)
            Wf = np.concatenate([np.stack([a, b, b + 1], 1), np.stack([a, b + 1, a + 1], 1)])
            # orient away from the sheet middle
            cen = V[Wf].mean(1)
            cr = np.cross(V[Wf[:, 1]] - V[Wf[:, 0]], V[Wf[:, 2]] - V[Wf[:, 0]])
            mid_dir = cen - 0.5 * (V[bi + ii * nj_ + nj_ // 2] + V[bo + io * nj_ + nj_ // 2])
            # the wall's outward direction = profile direction away from the sheet (end tangent)
            if ii == 0:
                tdir = np.array([0.0, -1.0, 0.0])
            else:
                th = math.radians(spec.sweep_deg)
                tdir = np.array([-math.sin(th), math.cos(th), 0.0])
            if np.sum(cr @ tdir) < 0:
                Wf = Wf[:, ::-1]
            walls.append(Wf)
        F = np.vstack([F] + walls)
    # noise along vertex normals
    from research.zeroshot.z3_slicing import slicer as SL
    m0 = SL.mesh_from_arrays(V, F)
    V = V + rng.normal(0, spec.noise_mm, len(V))[:, None] * m0.vn
    # random rigid pose
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, zq = q
    Rm = np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)],
                   [2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)],
                   [2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)]])
    tr = rng.uniform(-200, 200, 3)
    Vw = V @ Rm.T + tr
    th = math.radians(spec.sweep_deg)
    truth = {
        "R_in": spec.R, "R_out": spec.R + spec.t, "t": spec.t, "sweep_deg": spec.sweep_deg,
        "taper_mm_per_m": spec.taper * 1000.0, "chord_in": 2 * spec.R * math.sin(th / 2),
        "axis_dir": (Rm @ np.array([0, 0, 1.0])).tolist(),
        "axis_point": (Rm @ np.array([0, 0, spec.L / 2]) + tr).tolist(),
        "runout_start_in": [(Rm @ np.array([r_of_z(zz), 0, zz]) + tr).tolist() for zz in (0, spec.L)],
        "runout_end_in": [(Rm @ np.array([r_of_z(zz) * math.cos(th), r_of_z(zz) * math.sin(th), zz]) + tr).tolist()
                          for zz in (0, spec.L)],
        "flat": None if not spec.flat_depth_mm else {"center_deg": spec.flat_center_frac * spec.sweep_deg,
                                                     "width_deg": spec.flat_width_deg, "depth_mm": spec.flat_depth_mm},
        "coverage": 1.0 if spec.hole_frac is None else 1 - (spec.hole_frac[1] - spec.hole_frac[0]),
        "spec": asdict(spec),
    }
    return SL.mesh_from_arrays(Vw, F), truth
