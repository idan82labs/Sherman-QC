"""Exact characteristic values computed from a :class:`~.geometry.FoldedPart`.

These functions implement the ENGINE_IO §1 conventions on the analytic model,
so the synthetic truth is a function of the part parameters only:

- **OML flange length** of bend *b* on side *S*: distance, in *S*'s plane and
  perpendicular to the bend, from *b*'s OML virtual sharp to the far boundary
  of *S* (a free cut edge, or the OML virtual sharp of the adjacent formed
  bend). Evaluated per *station* along the bend's axial corridor (the bend
  span trimmed by ``min(2 mm, 5 %)`` at both ends). At each station the far
  boundary is the **farthest crossing** of the station line with the
  material (the contract's per-bin maximum extent: a notch or hole the line
  passes over does not end the flange); the crossed boundary segment says
  whether it is a free edge or a bend. The value is the median over the
  stations of the majority boundary kind, with min/max and per-kind groups.
- **Backgauge length** of a just-formed bend: the same measurement, continued
  across still-flat (unformed) bends to the far boundary of the flat island.
- **Bend-line position**: signed offset of the actual OML sharp from the
  nominal stage model's OML sharp, perpendicular to the bend axis at the
  nominal mid-span, projected on side A's outward direction.

**Well-definedness.** Station values are grouped into *levels* (sorted values
chained within ``LEVEL_TOL_MM`` = 0.05 mm, inside one boundary kind, the same
grouping E1 uses). A far boundary is ``well_defined`` when its largest level
holds at least ``WELL_DEFINED_LEVEL_FRACTION`` (80 %) of the valid stations:
then the median is that level under any reasonable station placement (E1
samples 21 stations with a ``min(0.5 mm, 2 %)`` trim, E2 41 with
``min(2 mm, 5 %)``). A stepped edge or a mixed boundary near a 50/50 split is
not: its median flips with station placement, for the nominal and for a scan
alike, so truth flags it and the recipe never makes it REQUIRED.

All station work happens in the shared flat frame (every panel polygon is
stored there); unformed bends do not move anything, so a flat island is
coplanar in that frame.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .geometry import FREE, FoldedPart, GeometryError

__all__ = [
    "FarBoundary",
    "far_boundary",
    "corridor_stations",
    "bend_line_offset",
    "flat_island",
    "LEVEL_TOL_MM",
    "WELL_DEFINED_LEVEL_FRACTION",
    "E2_STATION_RULE",
    "E1_STATION_RULE",
]

_GOLDEN_JITTER = 1e-6 * (math.sqrt(5.0) - 1.0) / 2.0
#: Station values within this distance (chained) form one level (E1 uses the same).
LEVEL_TOL_MM = 0.05
#: A far boundary is well defined when its largest level holds this share of the stations.
WELL_DEFINED_LEVEL_FRACTION = 0.8
#: Station rules: (stations, trim cap mm, trim fraction of the span).
E2_STATION_RULE = (41, 2.0, 0.05)
#: E1's compile rule (``flange_stations = 21``, trim ``min(0.5, 2 %)``), for cross-checks only.
E1_STATION_RULE = (21, 0.5, 0.02)


@dataclass(frozen=True)
class FarBoundary:
    """Result of marching from a formed bend's OML sharp across one side."""

    value_mm: float | None
    min_mm: float | None
    max_mm: float | None
    boundary_kind: str  # FREE_EDGE | ADJACENT_BEND | ADJACENT_BEND_NOT_FORMED | UNDEFINED
    adjacent_bend: str | None
    stations: int
    valid_stations: int
    kind_fraction: float
    island_panels: tuple[str, ...]
    crossed_bends: tuple[str, ...]
    #: Per boundary kind: ``(kind, adjacent_bend, fraction, median, min, max)``,
    #: most frequent first. The primary values above are the first group's.
    groups: tuple[tuple[str, str | None, float, float, float, float], ...] = ()
    #: Median station-line parameter (from the bend line, along the side's
    #: outward direction) where the material ends (the far wall or the far
    #: bend's tangent); used for edge-coverage observability.
    exit_lambda_mm: float | None = None
    #: Levels (chained within ``LEVEL_TOL_MM`` inside one boundary kind):
    #: ``(kind, adjacent_bend, fraction of valid stations, median, min, max)``,
    #: largest first.
    levels: tuple[tuple[str, str | None, float, float, float, float], ...] = ()

    @property
    def constant_along_bend(self) -> bool:
        return self.min_mm is not None and self.max_mm is not None and (self.max_mm - self.min_mm) < 1e-6

    @property
    def dominant_level_fraction(self) -> float:
        return self.levels[0][2] if self.levels else 0.0

    @property
    def dominant_level_mm(self) -> float | None:
        return self.levels[0][3] if self.levels else None

    @property
    def well_defined(self) -> bool:
        """True when one level holds >= 80 % of the stations of a FREE_EDGE or
        ADJACENT_BEND boundary (see the module docstring)."""
        return (
            self.value_mm is not None
            and self.boundary_kind in ("FREE_EDGE", "ADJACENT_BEND")
            and self.dominant_level_fraction >= WELL_DEFINED_LEVEL_FRACTION - 1e-12
        )

    @property
    def ill_defined_reason(self) -> str | None:
        if self.well_defined:
            return None
        if self.value_mm is None or self.boundary_kind not in ("FREE_EDGE", "ADJACENT_BEND"):
            return "NO_FAR_BOUNDARY"
        if self.kind_fraction < WELL_DEFINED_LEVEL_FRACTION:
            return "MIXED_FAR_BOUNDARY"
        return "STEPPED_FAR_BOUNDARY"


def corridor_stations(extent: tuple[float, float], n: int = 41, *, trim_mm: float = 2.0, trim_fraction: float = 0.05) -> np.ndarray:
    """Station positions along the bend line (``line_dir`` coordinate): the bend
    span trimmed by ``min(trim_mm, trim_fraction * span)`` at both ends, ``n``
    cell centres, with a sub-micron irrational jitter so stations never sit
    exactly on a vertex."""
    e0, e1 = extent
    span = e1 - e0
    tau = min(trim_mm, trim_fraction * span)
    a, b = e0 + tau, e1 - tau
    j = (np.arange(n) + 0.5) / n
    return a + j * (b - a) + _GOLDEN_JITTER


def _levels(values: list[float], kinds: list[tuple[str, str | None]]) -> tuple[tuple[str, str | None, float, float, float, float], ...]:
    """Chain sorted station values within ``LEVEL_TOL_MM`` inside each boundary kind."""
    total = len(values)
    out = []
    for kk in sorted(set(kinds), key=lambda x: (x[0], x[1] or "")):
        vals = sorted(v for v, k in zip(values, kinds) if k == kk)
        chain = [vals[0]]
        for v in vals[1:]:
            if v - chain[-1] <= LEVEL_TOL_MM:
                chain.append(v)
            else:
                out.append((kk[0], kk[1], len(chain) / total, float(np.median(chain)), chain[0], chain[-1]))
                chain = [v]
        out.append((kk[0], kk[1], len(chain) / total, float(np.median(chain)), chain[0], chain[-1]))
    out.sort(key=lambda lv: (-lv[2], lv[0], lv[1] or "", lv[3]))
    return tuple(out)


def flat_island(fp: FoldedPart, bend: str, side_panel: str) -> tuple[list[str], list[str]]:
    """Panels reachable from ``side_panel`` through *unformed* bends (never crossing
    ``bend`` itself), and the unformed bends crossed."""
    panels = [side_panel]
    crossed: list[str] = []
    seen = {side_panel}
    queue = [side_panel]
    while queue:
        cur = queue.pop(0)
        for b in fp.design.bends:
            if b.key == bend or fp.bend_geo[b.key].formed:
                continue
            if cur not in (b.panel_a, b.panel_b):
                continue
            other = b.panel_b if cur == b.panel_a else b.panel_a
            if other in seen:
                continue
            seen.add(other)
            panels.append(other)
            crossed.append(b.key)
            queue.append(other)
    return panels, crossed


def _line_intervals(
    pts: np.ndarray, tags: tuple[str, ...], q0: np.ndarray, u: np.ndarray
) -> list[tuple[float, float, str, str]]:
    """Inside-intervals of the line ``q0 + lam*u`` with a simple polygon (half-open
    crossing rule). Returns ``(lam_in, lam_out, tag_in, tag_out)`` sorted."""
    perp = np.array([-u[1], u[0]])
    rel = pts - q0
    w = rel @ perp
    lam = rel @ u
    n = len(pts)
    hits: list[tuple[float, str]] = []
    for i in range(n):
        j = (i + 1) % n
        if (w[i] >= 0.0) != (w[j] >= 0.0):
            s = w[i] / (w[i] - w[j])
            hits.append((float(lam[i] + s * (lam[j] - lam[i])), tags[i]))
    hits.sort(key=lambda h: h[0])
    out = []
    for k in range(0, len(hits) - 1, 2):
        out.append((hits[k][0], hits[k + 1][0], hits[k][1], hits[k + 1][1]))
    return out


def _sharp_lambda(fp: FoldedPart, other_bend: str, panel: str, q0: np.ndarray, u: np.ndarray) -> float | None:
    """Station-line parameter where it crosses ``other_bend``'s OML sharp line (in the
    flat frame of ``panel``, which is attached to ``other_bend``)."""
    g = fp.bend_geo[other_bend]
    o = g.outward(panel)
    base = g.line_point + g.sigma * o
    denom = u[0] * g.d[1] - u[1] * g.d[0]
    if abs(denom) < 1e-12:
        return None
    rel = base - q0
    return float((rel[0] * g.d[1] - rel[1] * g.d[0]) / denom)


def far_boundary(
    fp: FoldedPart,
    bend: str,
    side_panel: str,
    *,
    across_unformed: bool,
    station_rule: tuple[int, float, float] = E2_STATION_RULE,
) -> FarBoundary:
    """March from ``bend``'s OML virtual sharp into ``side_panel`` (see module doc).

    ``across_unformed=False`` gives the flange length (the side panel's own
    boundary); ``True`` gives the backgauge length (the farthest material of
    the flat island reached through unformed bends). ``bend`` must be formed.
    ``station_rule`` = ``(stations, trim cap mm, trim fraction)``."""
    g = fp.bend_geo[bend]
    if not g.formed:
        raise GeometryError(f"bend {bend} is not formed; it has no OML virtual sharp")
    if side_panel not in (g.parent, g.child):
        raise GeometryError(f"{side_panel} is not adjacent to {bend}")
    u = g.outward(side_panel)
    if across_unformed:
        island, crossed = flat_island(fp, bend, side_panel)
    else:
        island, crossed = [side_panel], []
    n_stations, trim_mm, trim_fraction = station_rule
    stations = corridor_stations(g.extent, n_stations, trim_mm=trim_mm, trim_fraction=trim_fraction)
    lam_start = g.ba_actual / 2.0 + 1e-6
    lengths: list[float] = []
    kinds: list[tuple[str, str | None]] = []
    exits: list[float] = []
    for v in stations:
        q0 = g.line_point + v * g.d
        # the farthest crossing beyond the bend: per-bin maximum extent of the material
        hit: tuple[float, str, str] | None = None
        for pk in island:
            for (_a, b, _ta, tb) in _line_intervals(fp.outline[pk], fp.tags[pk], q0, u):
                if b > lam_start and (hit is None or b > hit[0]):
                    hit = (b, tb, pk)
        if hit is None:
            continue
        end, exit_tag, exit_panel = hit
        exits.append(end)
        if exit_tag == FREE:
            lengths.append(end - g.sigma)
            kinds.append(("FREE_EDGE", None))
            continue
        if exit_tag == bend:
            continue
        og = fp.bend_geo[exit_tag]
        if og.formed:
            lam = _sharp_lambda(fp, exit_tag, exit_panel, q0, u)
            if lam is None:
                continue
            lengths.append(lam - g.sigma)
            kinds.append(("ADJACENT_BEND", exit_tag))
        else:
            # flange-mode stop at an unformed neighbour: distance to its strip centre
            lam = _sharp_lambda(fp, exit_tag, exit_panel, q0, u)
            if lam is None:
                continue
            lengths.append(lam - g.sigma)
            kinds.append(("ADJACENT_BEND_NOT_FORMED", exit_tag))
    if not lengths:
        return FarBoundary(None, None, None, "UNDEFINED", None, len(stations), 0, 0.0, tuple(island), tuple(crossed))
    counts = Counter(kinds)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1] or ""))
    arr = np.asarray(lengths)
    groups = []
    for (kind_g, adj_g), count in ordered:
        sel = np.asarray([kk == (kind_g, adj_g) for kk in kinds])
        vals = arr[sel]
        groups.append((kind_g, adj_g, count / len(lengths), float(np.median(vals)), float(vals.min()), float(vals.max())))
    kind, adj, frac, med, vmin, vmax = groups[0]
    return FarBoundary(
        value_mm=med,
        min_mm=vmin,
        max_mm=vmax,
        boundary_kind=kind,
        adjacent_bend=adj,
        stations=len(stations),
        valid_stations=len(lengths),
        kind_fraction=frac,
        island_panels=tuple(island),
        crossed_bends=tuple(crossed),
        groups=tuple(groups),
        exit_lambda_mm=float(np.median(exits)) if exits else None,
        levels=_levels(lengths, kinds),
    )


def bend_line_offset(actual: FoldedPart, nominal: FoldedPart, bend: str, side_a_panel: str) -> dict[str, object]:
    """Signed bend-line position of ``bend`` (ENGINE_IO §1): the actual OML sharp's
    offset from the nominal (stage-model) OML sharp, perpendicular to the nominal
    axis at its mid-span, along side A's nominal outward direction. Both parts
    must have the bend formed and share the CAD frame (same datum pose)."""
    a_line = actual.bend_line_cad(bend)
    n_line = nominal.bend_line_cad(bend)
    dn = n_line[1] - n_line[0]
    dn = dn / np.linalg.norm(dn)
    mid = n_line.mean(axis=0)
    da = a_line[1] - a_line[0]
    denom = float(da @ dn)
    if abs(denom) < 1e-12:
        raise GeometryError(f"bend {bend}: actual and nominal lines are perpendicular")
    s = float((mid - a_line[0]) @ dn) / denom
    q = a_line[0] + s * da
    off = q - mid
    out_a = nominal.outward_cad(bend, side_a_panel)
    normal_a = nominal.panel_normal_skin0(side_a_panel)
    cosang = abs(float(da @ dn)) / float(np.linalg.norm(da))
    return {
        "value_mm": float(off @ out_a),
        "normal_component_mm": float(off @ normal_a),
        "offset_vector_mm": [float(x) for x in off],
        "line_angle_deg": math.degrees(math.acos(min(1.0, cosang))),
    }
