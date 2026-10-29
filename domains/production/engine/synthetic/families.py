"""The seven named E2 part families and their seeded parameter samplers.

Families (plan §4 E2, amendment 5: "name the seven families"):

=========================  ======  ==============================================
Family                     Bends   Shape
=========================  ======  ==============================================
``L_BRACKET``              1       two legs; datum = larger leg
``U_CHANNEL``              2       web + two flanges, same direction; optional
                                   symmetric variant (equal flanges, symmetric
                                   holes) for the symmetry gate
``Z_STEP``                 2       flange / riser / flange, opposite directions
``HAT``                    4       brim / wall / top / wall / brim
``CHANNEL_WITH_RETURNS``   4       C-channel with inward returns (lips)
``TWO_DIRECTION_TAB``      3       base with flanges about non-parallel axes
                                   (corner relief) and a narrow tab on one flange
``PANEL_WITH_HOLES_NOTCHES``  3    large datum panel, hole pattern, edge notch,
                                   partial-span and perpendicular flanges, a
                                   notched flange edge
=========================  ======  ==============================================

Conditioning. Every flange side and backgauge run is laid out so that its far
boundary has one dominant level (>= 80 % of the stations under both the E1 and
the E2 station rules, :data:`~.analytic.WELL_DEFINED_LEVEL_FRACTION`), or is
clearly mixed (then the recipe rule makes it NOT_INSPECTED): partial-span
flanges and tabs cover either well under 20 % or well over 80 % of the edge a
neighbouring bend marches across, and notches stay out of the corridors whose
far boundary they would split near 50/50.

Every family honours a :class:`FeatureRequest` so the case planner can force
the amendment-5 coverage per family: acute bends (included < 90°), ``Ri >= 4t``,
a flange ``<= 12 mm`` and a flange ``>= 300 mm``. Profiles are asymmetric
(and the hole pattern chiral) unless ``symmetric`` is requested, so that
mirrored-part injections are detectable and bend identities unambiguous.

Builders are pure functions of a ``numpy.random.Generator``; infeasible draws
(a straight portion shorter than ``max(2t, 3 mm)``) are redrawn from the same
stream, so the result stays deterministic per seed.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import (
    DOWN,
    FREE,
    UP,
    BendDesign,
    GeometryError,
    Hole,
    PanelDesign,
    PartDesign,
    neutral_length,
    outside_setback,
)

__all__ = [
    "FAMILIES",
    "FeatureRequest",
    "build_family",
    "random_placement",
    "ProfileSegment",
    "build_profile",
    "DEFAULT_K_FACTOR",
    "SHORT_FLANGE_MAX_MM",
    "LONG_FLANGE_MIN_MM",
]

DEFAULT_K_FACTOR = 0.33
SHORT_FLANGE_MAX_MM = 12.0
LONG_FLANGE_MIN_MM = 300.0
_THICKNESSES = (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0)
_THICKNESS_WEIGHTS = (0.06, 0.14, 0.12, 0.18, 0.2, 0.12, 0.12, 0.06)
_MAX_ATTEMPTS = 200


@dataclass(frozen=True)
class FeatureRequest:
    """Geometric features a case must contain (amendment 5 coverage)."""

    acute: bool = False
    large_radius: bool = False
    short_flange: bool = False
    long_flange: bool = False
    symmetric: bool = False

    def tags(self) -> list[str]:
        return [name for name in ("acute", "large_radius", "short_flange", "long_flange", "symmetric") if getattr(self, name)]


# ---------------------------------------------------------------------------
# shared draws
# ---------------------------------------------------------------------------


def _thickness(rng: np.random.Generator) -> float:
    return float(rng.choice(_THICKNESSES, p=_THICKNESS_WEIGHTS))


def _radius(rng: np.random.Generator, t: float, large: bool) -> float:
    if large:
        return round(t * float(rng.uniform(4.0, 6.0)), 1)
    return max(0.5, round(t * float(rng.uniform(0.7, 2.0)), 1))


def _sweep(rng: np.random.Generator, acute: bool) -> float:
    """Nominal sweep (deg). Acute = included in [40, 85]; else 90 (60 %) or included in [95, 150]."""
    if acute:
        return float(180 - int(rng.integers(40, 86)))
    if rng.random() < 0.6:
        return 90.0
    return float(180 - int(rng.integers(95, 151)))


def _min_straight(t: float) -> float:
    return max(2.0 * t, 3.0)


def random_placement(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation plus a translation within +-400 mm (CAD placement)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    # re-orthonormalise (exactly proper)
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = rng.uniform(-400.0, 400.0, size=3)
    return m


# ---------------------------------------------------------------------------
# single-direction profile builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileSegment:
    """One panel of a profile part (bends parallel to the flat Y axis).

    ``oml_length``: OML length of the panel along the profile (end panels: sharp
    to free edge; interior panels: sharp to sharp). ``holes`` are fractional
    positions ``(fx, fy, radius)`` inside the panel's flat rectangle. ``notches``
    are ``(edge, f0, f1, depth)`` with edge ``"y0"``/``"y1"`` (side edges along
    the profile) or ``"far"`` (the free end edge of an end panel), fractions of
    that edge's length."""

    key: str
    oml_length: float
    role: str = ""
    holes: tuple[tuple[float, float, float], ...] = ()
    notches: tuple[tuple[str, float, float, float], ...] = ()


def build_profile(
    *,
    family: str,
    thickness: float,
    width: float,
    segments: Sequence[ProfileSegment],
    sweeps: Sequence[float],
    directions: Sequence[str],
    radii: Sequence[float],
    sequence: Sequence[int],
    groups: Sequence[Sequence[int]] = (),
    k_factor: float = DEFAULT_K_FACTOR,
    params: dict | None = None,
) -> PartDesign:
    """Build a profile part: panels ``segments[0..n]`` along +X joined by ``n`` bends
    parallel to +Y spanning the full ``width``. ``sequence``/``groups`` use bend
    indices 0..n-1. Raises :class:`GeometryError` when a straight portion is
    shorter than ``max(2t, 3 mm)``."""
    n = len(sweeps)
    if len(segments) != n + 1 or len(directions) != n or len(radii) != n:
        raise GeometryError("profile: segments must be one more than bends")
    t = thickness
    ossb = [outside_setback(sweeps[i], radii[i], t) for i in range(n)]
    ba = [neutral_length(sweeps[i], radii[i], t, k_factor) for i in range(n)]
    straight = []
    for i, seg in enumerate(segments):
        s = seg.oml_length
        if i > 0:
            s -= ossb[i - 1]
        if i < n:
            s -= ossb[i]
        if s < _min_straight(t):
            raise GeometryError(f"profile: straight portion of {seg.key} is {s:.3f} mm")
        straight.append(s)
    bend_keys = [f"BND{i + 1}" for i in range(n)]
    xs = [0.0]
    panels: list[PanelDesign] = []
    bends: list[BendDesign] = []
    x = 0.0
    for i, seg in enumerate(segments):
        xa, xb = x, x + straight[i]
        left = bend_keys[i - 1] if i > 0 else FREE
        right = bend_keys[i] if i < n else FREE
        outline, tags = _rect_with_notches(xa, xb, width, left, right, seg.notches, end_panel=(i == 0 or i == n), first=(i == 0))
        holes = tuple(Hole((xa + fx * (xb - xa), fy * width), r) for fx, fy, r in seg.holes)
        panels.append(PanelDesign(seg.key, outline, tags, holes, seg.role))
        if i < n:
            bends.append(
                BendDesign(
                    key=bend_keys[i],
                    panel_a=seg.key,
                    panel_b=segments[i + 1].key,
                    line_point=(xb + ba[i] / 2.0, 0.0),
                    line_dir=(0.0, 1.0),
                    normal_ab=(1.0, 0.0),
                    extent=(0.0, width),
                    direction=directions[i],
                    inner_radius=radii[i],
                    sweep_deg=sweeps[i],
                )
            )
            x = xb + ba[i]
            xs.append(x)
    design = PartDesign(
        family=family,
        thickness=t,
        k_factor=k_factor,
        panels=tuple(panels),
        bends=tuple(bends),
        sequence=tuple(bend_keys[i] for i in sequence),
        groups=tuple(tuple(bend_keys[i] for i in g) for g in groups),
        params=dict(params or {}),
    )
    design.validate()
    return design


def _rect_with_notches(
    xa: float,
    xb: float,
    w: float,
    left: str,
    right: str,
    notches: Sequence[tuple[str, float, float, float]],
    *,
    end_panel: bool,
    first: bool,
) -> tuple[tuple[tuple[float, float], ...], tuple[str, ...]]:
    """CCW rectangle ``[xa, xb] x [0, w]`` with rectangular notches cut into free edges.

    Edge order: bottom (``y = 0``, +X), right (``x = xb``, +Y), top (``y = w``, -X),
    left (``x = xa``, -Y). ``"far"`` notches go into the free end edge of an end
    panel (left edge of the first panel, right edge of the last). Each emitted
    vertex carries the tag of the edge that *starts* at it."""
    by_edge: dict[str, list[tuple[float, float, float]]] = {"y0": [], "y1": [], "far": []}
    for edge, f0, f1, depth in notches:
        if edge not in by_edge or not 0.0 < f0 < f1 < 1.0 or depth <= 0:
            raise GeometryError(f"invalid notch {(edge, f0, f1, depth)}")
        by_edge[edge].append((f0, f1, depth))
    far_left = end_panel and first and left == FREE
    far_right = end_panel and not first and right == FREE
    if by_edge["far"] and not (far_left or far_right):
        raise GeometryError("far-edge notches need a free end edge")
    pts: list[tuple[float, float]] = []
    tags: list[str] = []

    def emit(p: tuple[float, float], tag: str) -> None:
        pts.append(p)
        tags.append(tag)

    length = xb - xa
    emit((xa, 0.0), FREE)
    for f0, f1, depth in sorted(by_edge["y0"]):
        a, b = xa + f0 * length, xa + f1 * length
        emit((a, 0.0), FREE)
        emit((a, depth), FREE)
        emit((b, depth), FREE)
        emit((b, 0.0), FREE)
    if far_right:
        emit((xb, 0.0), FREE)
        for f0, f1, depth in sorted(by_edge["far"]):
            a, b = f0 * w, f1 * w
            emit((xb, a), FREE)
            emit((xb - depth, a), FREE)
            emit((xb - depth, b), FREE)
            emit((xb, b), FREE)
    else:
        emit((xb, 0.0), right)
    emit((xb, w), FREE)
    for f0, f1, depth in sorted(by_edge["y1"], reverse=True):
        a, b = xa + f1 * length, xa + f0 * length
        emit((a, w), FREE)
        emit((a, w - depth), FREE)
        emit((b, w - depth), FREE)
        emit((b, w), FREE)
    if far_left:
        emit((xa, w), FREE)
        for f0, f1, depth in sorted(by_edge["far"], reverse=True):
            a, b = f1 * w, f0 * w
            emit((xa, a), FREE)
            emit((xa + depth, a), FREE)
            emit((xa + depth, b), FREE)
            emit((xa, b), FREE)
    else:
        emit((xa, w), left)
    return _close_polygon(pts, tags)


def _close_polygon(pts: list[tuple[float, float]], tags: list[str]) -> tuple[tuple[tuple[float, float], ...], tuple[str, ...]]:
    """Drop consecutive duplicate vertices (keeping the later edge's tag)."""
    if len(tags) != len(pts):
        raise GeometryError("internal: polygon tags/points mismatch")
    out_pts: list[tuple[float, float]] = []
    out_tags: list[str] = []
    for p, tg in zip(pts, tags):
        if out_pts and abs(out_pts[-1][0] - p[0]) < 1e-12 and abs(out_pts[-1][1] - p[1]) < 1e-12:
            out_tags[-1] = tg
            continue
        out_pts.append((float(p[0]), float(p[1])))
        out_tags.append(tg)
    return tuple(out_pts), tuple(out_tags)


# ---------------------------------------------------------------------------
# hole helpers
# ---------------------------------------------------------------------------


def _chiral_holes(rng: np.random.Generator, count: int, rmax: float, symmetric: bool = False) -> tuple[tuple[float, float, float], ...]:
    """Fractional hole positions. Chiral (no mirror/rotation symmetry) by default;
    ``symmetric`` places pairs symmetric under a 180° rotation about the centre."""
    holes: list[tuple[float, float, float]] = []
    if count <= 0:
        return ()
    if symmetric:
        r = round(float(rng.uniform(2.0, rmax)), 1)
        fx, fy = float(rng.uniform(0.25, 0.4)), float(rng.uniform(0.25, 0.4))
        return ((fx, fy, r), (1 - fx, 1 - fy, r))
    tries = 0
    while len(holes) < count and tries < 200:
        tries += 1
        fx, fy = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.18, 0.82))
        r = round(float(rng.uniform(2.0, rmax)), 1)
        if any(abs(fx - h[0]) < 0.18 and abs(fy - h[1]) < 0.18 for h in holes):
            continue
        # avoid accidental symmetry about the panel centre lines
        if abs(fx - 0.5) < 0.04 or abs(fy - 0.5) < 0.04:
            continue
        holes.append((round(fx, 4), round(fy, 4), r))
    if len(holes) >= 2 and len({h[2] for h in holes}) == 1:
        h0 = holes[0]
        holes[0] = (h0[0], h0[1], round(h0[2] + 1.0, 1))
    return tuple(holes)


def _hole_radius_cap(dims: Sequence[float]) -> float:
    return max(2.0, min(6.0, min(dims) / 7.0))


def _pick_groups(rng: np.random.Generator, n: int, sequence: Sequence[int]) -> list[list[int]]:
    """Random מהלך groups: contiguous pairs of the sequence, chosen with probability 0.5."""
    groups: list[list[int]] = []
    i = 0
    while i < n - 1:
        if rng.random() < 0.5:
            groups.append([sequence[i], sequence[i + 1]])
            i += 2
        else:
            i += 1
    return groups


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------


def _flange_len(rng: np.random.Generator, lo: float = 20.0, hi: float = 110.0) -> float:
    return round(float(rng.uniform(lo, hi)), 1)


def _l_bracket(rng: np.random.Generator, req: FeatureRequest) -> PartDesign:
    t = _thickness(rng)
    ri = _radius(rng, t, req.large_radius)
    sweep = _sweep(rng, req.acute)
    la = round(float(rng.uniform(LONG_FLANGE_MIN_MM, 360.0)), 1) if req.long_flange else _flange_len(rng, 45, 130)
    if req.short_flange:
        lb = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1)
    else:
        lb = _flange_len(rng, 18, max(20.0, 0.75 * la))
    width = round(float(rng.uniform(40, 200)), 1)
    cap_a, cap_b = _hole_radius_cap((la, width)), _hole_radius_cap((lb, width))
    segs = [
        ProfileSegment("PA", la, "leg_a", _chiral_holes(rng, int(rng.integers(2, 4)), cap_a)),
        ProfileSegment("PB", lb, "leg_b", _chiral_holes(rng, 1 if lb > 30 else 0, cap_b)),
    ]
    direction = UP if rng.random() < 0.5 else DOWN
    return build_profile(
        family="L_BRACKET", thickness=t, width=width, segments=segs, sweeps=[sweep], directions=[direction],
        radii=[ri], sequence=[0], params={"oml_lengths": [la, lb], "width": width},
    )


def _u_channel(rng: np.random.Generator, req: FeatureRequest) -> PartDesign:
    t = _thickness(rng)
    ri = _radius(rng, t, req.large_radius)
    if req.symmetric:
        s1 = 90.0 if not req.acute else _sweep(rng, True)
        s2 = s1
    else:
        s1 = _sweep(rng, req.acute)
        s2 = _sweep(rng, False)
    web = round(float(rng.uniform(LONG_FLANGE_MIN_MM, 360.0)), 1) if req.long_flange else _flange_len(rng, 60, 180)
    if req.symmetric:
        l1 = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1) if req.short_flange else _flange_len(rng, 20, 0.7 * web)
        l2 = l1
    else:
        l1 = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1) if req.short_flange else _flange_len(rng, 18, 0.75 * web)
        l2 = _flange_len(rng, 18, 0.75 * web)
        if abs(l1 - l2) < 0.15 * max(l1, l2):
            l2 = round(min(0.8 * web, max(l1, l2) * 1.35), 1)
    width = round(float(rng.uniform(50, 220)), 1)
    cap_w = _hole_radius_cap((web, width))
    web_holes = _chiral_holes(rng, int(rng.integers(2, 5)), cap_w, symmetric=req.symmetric)
    if req.symmetric:
        f_holes: tuple = ()
        g_holes: tuple = ()
    else:
        f_holes = _chiral_holes(rng, 1 if l1 > 35 else 0, _hole_radius_cap((l1, width)))
        g_holes = _chiral_holes(rng, 1 if l2 > 35 else 0, _hole_radius_cap((l2, width)))
    segs = [
        ProfileSegment("PF1", l1, "flange", f_holes),
        ProfileSegment("PWEB", web, "web", web_holes),
        ProfileSegment("PF2", l2, "flange", g_holes),
    ]
    seq = [0, 1] if rng.random() < 0.5 else [1, 0]
    groups = [seq] if rng.random() < 0.4 else []
    return build_profile(
        family="U_CHANNEL", thickness=t, width=width, segments=segs, sweeps=[s1, s2], directions=[UP, UP],
        radii=[ri, ri], sequence=seq, groups=groups,
        params={"oml_lengths": [l1, web, l2], "width": width, "symmetric": req.symmetric},
    )


def _z_step(rng: np.random.Generator, req: FeatureRequest) -> PartDesign:
    t = _thickness(rng)
    ri = _radius(rng, t, req.large_radius)
    s1 = _sweep(rng, req.acute)
    s2 = s1 if rng.random() < 0.6 else _sweep(rng, False)
    l1 = round(float(rng.uniform(LONG_FLANGE_MIN_MM, 350.0)), 1) if req.long_flange else _flange_len(rng, 50, 150)
    riser = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1) if req.short_flange else _flange_len(rng, 15, 60)
    l2 = _flange_len(rng, 20, 0.75 * l1)
    width = round(float(rng.uniform(40, 200)), 1)
    segs = [
        ProfileSegment("PLOW", l1, "flange", _chiral_holes(rng, int(rng.integers(2, 4)), _hole_radius_cap((l1, width)))),
        ProfileSegment("PRISER", riser, "riser"),
        ProfileSegment("PHIGH", l2, "flange", _chiral_holes(rng, 1 if l2 > 35 else 0, _hole_radius_cap((l2, width)))),
    ]
    d1 = UP if rng.random() < 0.5 else DOWN
    d2 = DOWN if d1 == UP else UP
    seq = [0, 1] if rng.random() < 0.5 else [1, 0]
    return build_profile(
        family="Z_STEP", thickness=t, width=width, segments=segs, sweeps=[s1, s2], directions=[d1, d2],
        radii=[ri, ri], sequence=seq, groups=[],
        params={"oml_lengths": [l1, riser, l2], "width": width},
    )


def _hat(rng: np.random.Generator, req: FeatureRequest) -> PartDesign:
    t = _thickness(rng)
    ri = _radius(rng, t, req.large_radius)
    top = round(float(rng.uniform(LONG_FLANGE_MIN_MM, 350.0)), 1) if req.long_flange else _flange_len(rng, 60, 160)
    h1 = _flange_len(rng, 20, 0.6 * top)
    h2 = h1 if rng.random() < 0.3 else _flange_len(rng, 20, 0.6 * top)
    b1 = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1) if req.short_flange else _flange_len(rng, 15, 50)
    b2 = _flange_len(rng, 15, 55)
    if abs(b1 - b2) < 3.0:
        b2 = round(b1 + 6.0, 1)
    wall_sweep = 90.0 if rng.random() < 0.7 else float(180 - int(rng.integers(95, 121)))
    brim_sweep = wall_sweep
    if req.acute:
        brim_sweep = _sweep(rng, True)
    width = round(float(rng.uniform(50, 200)), 1)
    segs = [
        ProfileSegment("PBRIM1", b1, "brim"),
        ProfileSegment("PWALL1", h1, "wall", _chiral_holes(rng, 1 if h1 > 40 else 0, _hole_radius_cap((h1, width)))),
        ProfileSegment("PTOP", top, "top", _chiral_holes(rng, int(rng.integers(2, 4)), _hole_radius_cap((top, width)))),
        ProfileSegment("PWALL2", h2, "wall"),
        ProfileSegment("PBRIM2", b2, "brim", _chiral_holes(rng, 1 if b2 > 35 else 0, _hole_radius_cap((b2, width)))),
    ]
    order = [[0, 3, 1, 2], [3, 0, 2, 1], [0, 1, 2, 3], [1, 2, 0, 3]][int(rng.integers(0, 4))]
    groups = _pick_groups(rng, 4, order)
    return build_profile(
        family="HAT", thickness=t, width=width, segments=segs, sweeps=[brim_sweep, wall_sweep, wall_sweep, brim_sweep],
        directions=[UP, DOWN, DOWN, UP], radii=[ri] * 4, sequence=order, groups=groups,
        params={"oml_lengths": [b1, h1, top, h2, b2], "width": width},
    )


def _channel_with_returns(rng: np.random.Generator, req: FeatureRequest) -> PartDesign:
    t = _thickness(rng)
    ri = _radius(rng, t, req.large_radius)
    web = round(float(rng.uniform(LONG_FLANGE_MIN_MM, 350.0)), 1) if req.long_flange else _flange_len(rng, 70, 180)
    h1 = _flange_len(rng, 25, 0.55 * web)
    h2 = _flange_len(rng, 25, 0.55 * web)
    if abs(h1 - h2) < 4.0:
        h2 = round(h1 + 8.0, 1)
    r1 = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1) if req.short_flange else _flange_len(rng, 12, min(30.0, 0.35 * web))
    r2 = _flange_len(rng, 12, min(32.0, 0.35 * web))
    ret_sweep1 = _sweep(rng, req.acute) if req.acute else 90.0
    ret_sweep2 = 90.0 if rng.random() < 0.6 else ret_sweep1
    width = round(float(rng.uniform(50, 220)), 1)
    segs = [
        ProfileSegment("PRET1", r1, "return"),
        ProfileSegment("PFL1", h1, "flange", _chiral_holes(rng, 1 if h1 > 40 else 0, _hole_radius_cap((h1, width)))),
        ProfileSegment("PWEB", web, "web", _chiral_holes(rng, int(rng.integers(2, 4)), _hole_radius_cap((web, width)))),
        ProfileSegment("PFL2", h2, "flange"),
        ProfileSegment("PRET2", r2, "return"),
    ]
    order = [[0, 3, 1, 2], [3, 0, 2, 1], [0, 3, 2, 1]][int(rng.integers(0, 3))]
    groups = _pick_groups(rng, 4, order)
    return build_profile(
        family="CHANNEL_WITH_RETURNS", thickness=t, width=width, segments=segs,
        sweeps=[ret_sweep1, 90.0, 90.0, ret_sweep2], directions=[UP, UP, UP, UP], radii=[ri] * 4,
        sequence=order, groups=groups, params={"oml_lengths": [r1, h1, web, h2, r2], "width": width},
    )


def _two_direction_tab(rng: np.random.Generator, req: FeatureRequest) -> PartDesign:
    """Base (datum) with F1 on its +X edge (axis // Y), F2 on its +Y edge (axis // X,
    corner relief) and a tab T on F2's free edge (axis // X)."""
    t = _thickness(rng)
    ri = _radius(rng, t, req.large_radius)
    k = DEFAULT_K_FACTOR
    s1 = _sweep(rng, False) if not req.acute else 90.0
    s2 = 90.0
    st = _sweep(rng, req.acute) if req.acute else float(rng.choice([90.0, 45.0, 60.0]))
    wb = round(float(rng.uniform(LONG_FLANGE_MIN_MM, 340.0)), 1) if req.long_flange else _flange_len(rng, 90, 200)
    db = _flange_len(rng, 80, 170)
    h1 = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1) if req.short_flange else _flange_len(rng, 20, min(70.0, 0.55 * db))
    h2 = _flange_len(rng, 25, min(70.0, 0.5 * wb))
    tl = _flange_len(rng, 10, min(25.0, 0.5 * h2))
    o1, o2, ot = (outside_setback(s, ri, t) for s in (s1, s2, st))
    ba1, ba2, bat = (neutral_length(s, ri, t, k) for s in (s1, s2, st))
    xmax, ymax = wb - o1, db - o2
    relief = round(max(2.0 * t, ri + t) * float(rng.uniform(1.0, 1.6)), 1)
    xr, yr = xmax - relief, ymax - relief
    f1_s = h1 - o1
    f2_s = h2 - o2 - ot
    t_s = tl - ot
    ms = _min_straight(t)
    if min(f1_s, f2_s, t_s) < ms or min(xr, yr) < 30:
        raise GeometryError("two-direction tab: infeasible straight portions")
    # base polygon with corner relief at (+X, +Y)
    base_pts = ((0.0, 0.0), (xmax, 0.0), (xmax, yr), (xr, yr), (xr, ymax), (0.0, ymax))
    base_tags = (FREE, "BNDF1", FREE, FREE, "BNDF2", FREE)
    holes_f = _chiral_holes(rng, int(rng.integers(3, 6)), _hole_radius_cap((xmax, ymax)))
    base_holes = tuple(Hole((fx * xr, fy * yr), r) for fx, fy, r in holes_f)
    # F1 panel (x beyond the strip), y in [0, yr]
    x1a = xmax + ba1
    f1 = PanelDesign("PF1", ((x1a, 0.0), (x1a + f1_s, 0.0), (x1a + f1_s, yr), (x1a, yr)), (FREE, FREE, FREE, "BNDF1"), (), "flange")
    # F2 panel (y beyond the strip), x in [0, xr], a narrow tab on its far edge: the
    # tab covers <= 15 % of F2's edge, so F2's far boundary (and a backgauge across
    # the unformed tab) keeps one dominant level
    y2a = ymax + ba2
    y2b = y2a + f2_s
    tab_w = max(10.0, float(rng.uniform(0.08, 0.13)) * xr)
    if tab_w > 0.15 * xr:
        raise GeometryError("two-direction tab: F2 too short for a narrow tab")
    ta = round(float(rng.uniform(0.3, 0.55)) * xr, 3)
    tb = round(ta + tab_w, 3)
    f2_pts = ((0.0, y2a), (xr, y2a), (xr, y2b), (tb, y2b), (ta, y2b), (0.0, y2b))
    f2_tags = ("BNDF2", FREE, FREE, "BNDT", FREE, FREE)
    f2_holes = ()
    if f2_s > 30:
        f2_holes = (Hole((0.5 * (ta + tb) + 0.07 * xr, y2a + 0.45 * f2_s), round(min(4.0, 0.15 * f2_s), 1)),)
    f2 = PanelDesign("PF2", f2_pts, f2_tags, f2_holes, "flange")
    y3a = y2b + bat
    tab = PanelDesign("PTAB", ((ta, y3a), (tb, y3a), (tb, y3a + t_s), (ta, y3a + t_s)), ("BNDT", FREE, FREE, FREE), (), "tab")
    base = PanelDesign("PBASE", base_pts, base_tags, base_holes, "base")
    bends = (
        BendDesign("BNDF1", "PBASE", "PF1", (xmax + ba1 / 2.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, yr), UP, ri, s1),
        BendDesign("BNDF2", "PBASE", "PF2", (0.0, ymax + ba2 / 2.0), (1.0, 0.0), (0.0, 1.0), (0.0, xr), UP, ri, s2),
        BendDesign("BNDT", "PF2", "PTAB", (0.0, y2b + bat / 2.0), (1.0, 0.0), (0.0, 1.0), (ta, tb), UP if rng.random() < 0.5 else DOWN, ri, st),
    )
    order = [["BNDT", "BNDF2", "BNDF1"], ["BNDF1", "BNDT", "BNDF2"], ["BNDF2", "BNDT", "BNDF1"]][int(rng.integers(0, 3))]
    groups: list[list[str]] = []
    design = PartDesign(
        family="TWO_DIRECTION_TAB", thickness=t, k_factor=k, panels=(base, f1, f2, tab), bends=bends,
        sequence=tuple(order), groups=tuple(tuple(g) for g in groups),
        params={"oml": {"base_x": wb, "base_y": db, "f1": h1, "f2": h2, "tab": tl}, "relief": relief},
    )
    design.validate()
    return design


def _panel_with_holes_notches(rng: np.random.Generator, req: FeatureRequest) -> PartDesign:
    """Large datum panel with a hole grid and a top-edge notch. F1 on +X spans
    86-94 % of the base edge (so the +X march of BNDF2 meets one dominant far
    boundary: BNDF1, or F1's free edge while BNDF1 is flat), F2 is full-span on
    -X (DOWN), F3 is partial-span on -Y (perpendicular axis) with a notch in its
    free far edge. The base notch lies outside F3's span (the +Y march of BNDF3
    sees only the straight top edge; the X marches pass over it, since the far
    boundary is the farthest crossing)."""
    t = _thickness(rng)
    ri = _radius(rng, t, req.large_radius)
    k = DEFAULT_K_FACTOR
    s1 = _sweep(rng, req.acute)
    s2 = 90.0
    s3 = 90.0 if rng.random() < 0.7 else float(180 - int(rng.integers(100, 140)))
    wb = round(float(rng.uniform(LONG_FLANGE_MIN_MM, 380.0)), 1) if req.long_flange else _flange_len(rng, 150, 280)
    db = _flange_len(rng, 120, 240)
    h1 = _flange_len(rng, 25, 70)
    h2 = round(float(rng.uniform(8.0, SHORT_FLANGE_MAX_MM)), 1) if req.short_flange else _flange_len(rng, 15, 45)
    h3 = _flange_len(rng, 20, 55)
    o1, o2, o3 = (outside_setback(s, ri, t) for s in (s1, s2, s3))
    ba1, ba2, ba3 = (neutral_length(s, ri, t, k) for s in (s1, s2, s3))
    ms = _min_straight(t)
    x0, x1 = 0.0, wb - o1 - o2  # base spans [x0, x1] (F2 sharp at -X side, F1 at +X side)
    y0, y1 = 0.0, db - o3
    f1s, f2s, f3s = h1 - o1, h2 - o2, h3 - o3
    if min(f1s, f2s, f3s) < ms:
        raise GeometryError("panel family: infeasible straight portions")
    ya, yb = round(float(rng.uniform(0.03, 0.06)) * y1, 3), round(float(rng.uniform(0.93, 0.97)) * y1, 3)
    relief3 = max(2.0 * t, ri + t) + 2.0
    xa3, xb3 = round(relief3 + float(rng.uniform(0.05, 0.2)) * x1, 3), round(float(rng.uniform(0.55, 0.72)) * x1, 3)
    # top-edge notch on the base (y = y1), right of F3's span, away from the corner
    margin = max(4.0, 0.04 * x1)
    room = (x1 - max(8.0, 0.06 * x1)) - (xb3 + margin)
    width = min(float(rng.uniform(0.06, 0.1)) * x1, room)
    if width < 0.04 * x1:
        raise GeometryError("panel family: no room for the base notch outside F3's span")
    nx0 = round(xb3 + margin, 3)
    nx1 = round(nx0 + width, 3)
    ndepth = round(float(rng.uniform(6.0, 18.0)), 2)
    base_pts = (
        (x0, y0),
        (xa3, y0),
        (xb3, y0),
        (x1, y0),
        (x1, ya),
        (x1, yb),
        (x1, y1),
        (nx1, y1),
        (nx1, y1 - ndepth),
        (nx0, y1 - ndepth),
        (nx0, y1),
        (x0, y1),
    )
    base_tags = (FREE, "BNDF3", FREE, FREE, "BNDF1", FREE, FREE, FREE, FREE, FREE, FREE, "BNDF2")
    # hole grid (chiral: one extra hole off-grid, different radii)
    holes: list[Hole] = []
    nxh, nyh = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    rad = round(float(rng.uniform(2.5, 5.0)), 1)
    for i in range(nxh):
        for j in range(nyh):
            cx = (0.18 + 0.64 * (i + 0.5) / nxh) * x1
            cy = (0.2 + 0.5 * (j + 0.5) / nyh) * y1
            holes.append(Hole((round(cx, 3), round(cy, 3)), rad))
    holes.append(Hole((round(0.83 * x1, 3), round(0.14 * y1 + 6, 3)), round(rad + 1.5, 1)))
    base = PanelDesign("PBASE", base_pts, base_tags, tuple(holes), "base")
    # F1: +X, span [ya, yb] (86-94 % of the base edge)
    xf1 = x1 + ba1
    span1 = yb - ya
    f1_pts = ((xf1, ya), (xf1 + f1s, ya), (xf1 + f1s, yb), (xf1, yb))
    f1_tags = (FREE, FREE, FREE, "BNDF1")
    f1_holes = (Hole((xf1 + 0.55 * f1s, ya + 0.78 * span1), round(min(3.5, 0.18 * f1s), 1)),) if f1s > 20 else ()
    f1 = PanelDesign("PF1", f1_pts, f1_tags, f1_holes, "flange")
    # F2: -X, full span, DOWN
    xf2 = x0 - ba2
    f2 = PanelDesign("PF2", ((xf2 - f2s, y0), (xf2, y0), (xf2, y1), (xf2 - f2s, y1)), (FREE, "BNDF2", FREE, FREE), (), "flange")
    # F3: -Y, partial span [xa3, xb3], with a notch (8-12 % of the span) in its free far edge
    yf3 = y0 - ba3
    ybot = yf3 - f3s
    span3 = xb3 - xa3
    n0 = round(xa3 + float(rng.uniform(0.35, 0.45)) * span3, 3)
    n1 = round(n0 + float(rng.uniform(0.08, 0.12)) * span3, 3)
    nd = round(min(0.4 * f3s, float(rng.uniform(3.0, 8.0))), 2)
    f3_pts = ((xa3, ybot), (n0, ybot), (n0, ybot + nd), (n1, ybot + nd), (n1, ybot), (xb3, ybot), (xb3, yf3), (xa3, yf3))
    f3_tags = (FREE, FREE, FREE, FREE, FREE, FREE, "BNDF3", FREE)
    f3 = PanelDesign("PF3", f3_pts, f3_tags, (), "flange")
    bends = (
        BendDesign("BNDF1", "PBASE", "PF1", (x1 + ba1 / 2.0, 0.0), (0.0, 1.0), (1.0, 0.0), (ya, yb), UP, ri, s1),
        BendDesign("BNDF2", "PBASE", "PF2", (x0 - ba2 / 2.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (y0, y1), DOWN, ri, s2),
        BendDesign("BNDF3", "PBASE", "PF3", (0.0, y0 - ba3 / 2.0), (1.0, 0.0), (0.0, -1.0), (xa3, xb3), UP if rng.random() < 0.6 else DOWN, ri, s3),
    )
    order = list(rng.permutation(["BNDF1", "BNDF2", "BNDF3"]))
    groups = [order[:2]] if rng.random() < 0.4 else []
    design = PartDesign(
        family="PANEL_WITH_HOLES_NOTCHES", thickness=t, k_factor=k, panels=(base, f1, f2, f3), bends=bends,
        sequence=tuple(order), groups=tuple(tuple(g) for g in groups),
        params={"oml": {"base_x": wb, "base_y": db, "f1": h1, "f2": h2, "f3": h3}},
    )
    design.validate()
    return design


_BUILDERS: dict[str, Callable[[np.random.Generator, FeatureRequest], PartDesign]] = {
    "L_BRACKET": _l_bracket,
    "U_CHANNEL": _u_channel,
    "Z_STEP": _z_step,
    "HAT": _hat,
    "CHANNEL_WITH_RETURNS": _channel_with_returns,
    "TWO_DIRECTION_TAB": _two_direction_tab,
    "PANEL_WITH_HOLES_NOTCHES": _panel_with_holes_notches,
}

#: The seven named families, in canonical order.
FAMILIES: tuple[str, ...] = tuple(_BUILDERS)


def build_family(family: str, rng: np.random.Generator, req: FeatureRequest = FeatureRequest()) -> PartDesign:
    """Draw a feasible part of ``family`` honouring ``req`` (deterministic per rng state)."""
    if family not in _BUILDERS:
        raise KeyError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")
    if req.symmetric and family != "U_CHANNEL":
        raise GeometryError("only U_CHANNEL has a symmetric variant")
    last: Exception | None = None
    for _ in range(_MAX_ATTEMPTS):
        try:
            design = _BUILDERS[family](rng, req)
        except GeometryError as exc:
            last = exc
            continue
        if _satisfies(design, req):
            return design
        last = GeometryError("feature request not satisfied")
    raise GeometryError(f"{family}: no feasible draw after {_MAX_ATTEMPTS} attempts ({last})")


def _satisfies(design: PartDesign, req: FeatureRequest) -> bool:
    """Check the requested features on the *design* parameters (flange lengths are
    re-verified analytically by the truth builder)."""
    t = design.thickness
    if req.acute and not any(b.included_deg < 90.0 for b in design.bends):
        return False
    if req.large_radius and not any(b.inner_radius >= 4.0 * t - 1e-9 for b in design.bends):
        return False
    return True
