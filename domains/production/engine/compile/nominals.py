"""Nominal flange lengths and backgauge lengths (OUTSIDE_MOLD_LINE convention, ENGINE_IO §1).

**Flange length** of bend *b*, side *S* (panel Q): the distance from *b*'s OML
virtual sharp to the far boundary of Q, measured in Q's OML plane for *b* (the
skin on *b*'s convex side), perpendicular to the bend direction (along Q's
outward direction ``w``), inside *b*'s axial corridor. It is evaluated at
``flange_stations`` section stations spread over the corridor: at each station
the line ``sharp(u) + s * w`` is intersected with Q's skin boundary loops and the
farthest crossing is the far boundary at that station; its boundary segment
says what the boundary is:

- a cut edge -> FREE_EDGE, length = that crossing's ``s``;
- the tangent line of another bend *b'* -> ADJACENT_BEND, length at that
  station = where the station line meets *b'*'s OML virtual sharp (projected
  into the station plane): OML sharp to OML sharp. For *b'* parallel to *b*
  every station gives the same value; for a non-parallel *b'* (an angled or
  trapezoidal panel) the values vary along the corridor;
- an unsupported feature -> no nominal (UNSUPPORTED_FEATURE).

The boundary (kind and, for ADJACENT_BEND, the bend) must hold >= 60 % of the
valid stations. A FREE_EDGE nominal is then the dominant *level* of the edge
(station values grouped within 0.05 mm, the level holding >= 60 % of the
stations), or the median when the edge varies continuously (a taper). When at
least 30 % of the FREE stations end on a far-edge segment parallel to the bend
(within 5 deg: the straight far edge or a notch bottom), only those stations
count, so corner rounds and chamfers at the corridor ends do not move it. A stepped
edge without a dominant level, or a boundary without a 60 % class, has no
nominal (NOMINAL_UNAVAILABLE): a near 50/50 split would make the median flip
with station placement, for the nominal and for the scan alike. A notch or a
short tab covering a minority of the corridor never moves the nominal; the
spread is in diagnostics. ADJACENT_BEND nominals are the median of the
per-station crossings (exact for parallel bends).

**Backgauge length** at stage S_k: the same measurement from the OML sharp of
the just-formed bend *k*, across the material that is still flat at S_k (the
side panel plus every panel and flat strip reachable through unformed bends),
to the far boundary (a cut edge or the OML sharp of an already formed bend),
at the S_k pose. It is attached to the side that leads toward the unformed
bends (more unformed bends wins, then the longer developed run, then side A).
Unlike a final-pose flange length it depends on K through every flat strip it
crosses (``L_n = sweep (Ri + K t)``); the compiler records the crossed bends and
``d nominal / d K`` in diagnostics so the K uncertainty can be propagated.

Every nominal also reports the **plurality** far-boundary class of its
stations (FREE or a given adjacent bend; features excluded), which names the
side's boundary when no nominal can be given (``FlangeSideV1.boundary_kind``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from domains.production.engine.compile.flange_graph import LABEL_BEND, LABEL_FEATURE, LABEL_FREE
from domains.production.engine.compile.geometry import apply4, apply_dir4, station_crossings, unit
from domains.production.engine.compile.part_model import ModelBend, PartModel

__all__ = [
    "FlangeNominal",
    "BackgaugeNominal",
    "flange_length_nominal",
    "backgauge_nominal",
    "far_boundary_stations",
]
#: A far-boundary class (FREE / a given adjacent bend / a feature), and a free-edge
#: level, must hold at least this fraction of the valid stations: a near 50/50
#: split would make the median flip with station placement (nominal *and* scan).
_MAJORITY = 0.6
#: Station values within this distance are one free-edge level (steps vs continuous).
_LEVEL_TOL_MM = 0.05
#: A free-edge station is "clean" when its far boundary segment is parallel to the bend.
_PARALLEL_SEGMENT_DEG = 5.0
_COS_PARALLEL_SEGMENT = float(np.cos(np.radians(_PARALLEL_SEGMENT_DEG)))
#: Clean stations are used alone when they are at least this fraction of the FREE stations.
_CLEAN_FRACTION = 0.3


@dataclass
class FlangeNominal:
    value: float | None
    boundary_kind: str | None
    adjacent_bend_key: int | None
    unavailable_reason: str | None
    detail: str
    stations_valid: int = 0
    spread_mm: float | None = None
    station_values: list[float] = field(default_factory=list)
    #: graph feature index that is the far boundary (UNSUPPORTED_FEATURE), when any
    feature_ref: int | None = None
    #: most frequent station class among FREE / ADJ (features excluded): ("FREE", None) or
    #: ("ADJ", graph bend key); None when no station ends on either
    plurality: tuple[str, int | None] | None = None


@dataclass
class BackgaugeNominal:
    side_panel: int
    nominal: FlangeNominal
    unformed_in_region: int
    #: unformed graph bends whose flat strips belong to the measured region
    crossed: list[int] = field(default_factory=list)


@dataclass
class _Poly:
    points: np.ndarray  # (K,3) closed polygon at the evaluation pose
    labels: np.ndarray  # (K,) label of segment i -> i+1
    refs: np.ndarray


def far_boundary_stations(polys: list[_Poly], origin: np.ndarray, a_dir: np.ndarray, w_dir: np.ndarray,
                          length: float, stations: int) -> list[tuple[float, float, int, int, bool] | None]:
    """Per station ``(u, far_s, label, ref, parallel)`` or ``None`` when no material crosses it.

    ``parallel`` is True when the far boundary segment runs parallel to the bend
    (within ``_PARALLEL_SEGMENT_DEG``): a straight far edge or a notch bottom,
    as opposed to a corner round, a chamfer or a slanted edge.
    """
    starts, ends, labels, refs = [], [], [], []
    for poly in polys:
        p = poly.points - origin
        uw = np.stack([p @ a_dir, p @ w_dir], axis=1)
        starts.append(uw)
        ends.append(np.roll(uw, -1, axis=0))
        labels.append(poly.labels)
        refs.append(poly.refs)
    if not starts:
        return [None] * stations
    s0, s1 = np.concatenate(starts), np.concatenate(ends)
    lab, ref = np.concatenate(labels), np.concatenate(refs)
    eps = min(0.5, 0.02 * length)
    out: list[tuple[float, float, int, int] | None] = []
    for i in range(stations):
        u = eps + (length - 2 * eps) * (i + 0.5) / stations
        w, idx = station_crossings(s0, s1, u)
        if len(w) == 0:
            out.append(None)
            continue
        j = int(np.argmax(w))
        k = idx[j]
        seg = s1[k] - s0[k]
        seg_len = float(np.hypot(seg[0], seg[1]))
        parallel = seg_len > 1e-12 and abs(seg[0]) / seg_len >= _COS_PARALLEL_SEGMENT
        out.append((u, float(w[j]), int(lab[k]), int(ref[k]), bool(parallel)))
    return out


def _side_polys(model: PartModel, panel_key: int, side_index: int, transform: np.ndarray | None = None) -> list[_Poly]:
    side = model.graph.panels[panel_key].sides[side_index]
    polys = []
    for loop in side.loops:
        pts = loop.points if transform is None else apply4(transform, loop.points)
        polys.append(_Poly(pts, loop.labels, loop.refs))
    return polys


def _classify(model: PartModel, bend_key: int, results, formed: set[int] | None
              ) -> tuple[str | None, int | None, list[int], int, str]:
    """Majority far-boundary class over stations: (kind, ref, station indices, valid count, detail).

    kind is FREE, ADJ (the tangent line of another formed bend, parallel or not),
    FEATURE, MIXED (no class at >= ``_MAJORITY`` = 60 % of the valid stations) or None.
    """
    counts: dict[tuple[str, int], list[int]] = {}
    valid = 0
    for i, r in enumerate(results):
        if r is None:
            continue
        _, _, label, ref, _ = r
        if label == LABEL_FREE:
            key = ("FREE", -1)
        elif label == LABEL_FEATURE:
            key = ("FEATURE", ref)
        elif label == LABEL_BEND:
            if ref == bend_key:
                continue
            if formed is not None and ref not in formed:
                continue  # a strip seam of an unformed bend inside the flat region
            key = ("ADJ", ref)
        else:
            continue
        valid += 1
        counts.setdefault(key, []).append(i)
    if not valid:
        return None, None, [], 0, "no material crosses the bend corridor"
    (kind, ref), idx = max(sorted(counts.items()), key=lambda kv: len(kv[1]))
    detail = ", ".join(f"{k}{'' if r < 0 else ':' + str(r)}={len(v)}" for (k, r), v in sorted(counts.items()))
    if len(idx) < _MAJORITY * valid:
        return "MIXED", None, [], valid, f"no far-boundary majority ({detail})"
    return kind, (ref if kind in ("ADJ", "FEATURE") else None), idx, valid, detail


def _plurality(results, bend_key: int, formed: set[int] | None) -> tuple[str, int | None] | None:
    """Most frequent FREE / ADJ station class (ties: ADJ first, then the lower bend key)."""
    counts: dict[tuple[str, int], int] = {}
    for r in results:
        if r is None:
            continue
        _, _, label, ref, _ = r
        if label == LABEL_FREE:
            key = ("FREE", -1)
        elif label == LABEL_BEND and ref != bend_key and (formed is None or ref in formed):
            key = ("ADJ", int(ref))
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    (kind, ref), _ = max(sorted(counts.items()), key=lambda kv: kv[1])
    return kind, (ref if kind == "ADJ" else None)


def _dominant_level(values: np.ndarray) -> tuple[float | None, str]:
    """Robust free-edge nominal: the dominant level of a stepped edge, else the median.

    Station values are grouped into levels (``_LEVEL_TOL_MM``). A level holding
    >= ``_MAJORITY`` of the stations gives the nominal (its median). An edge
    that varies continuously (about one level per station: a tapered flange)
    gives the median, which is stable. A stepped edge whose levels split the
    stations without a dominant one has no well-defined nominal (``None``).
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    levels: list[list[float]] = [[float(v[0])]]
    for x in v[1:]:
        if x - levels[-1][-1] <= _LEVEL_TOL_MM:
            levels[-1].append(float(x))
        else:
            levels.append([float(x)])
    top = max(levels, key=len)
    frac = len(top) / len(v)
    if frac >= _MAJORITY:
        return float(np.median(top)), f"level {np.median(top):.3f} mm at {frac:.0%} of stations"
    if len(levels) >= 0.5 * len(v):
        return float(np.median(v)), f"continuous edge, median of {len(v)} stations"
    summary = ", ".join(f"{np.median(lv):.3f} mm x{len(lv)}" for lv in sorted(levels, key=len, reverse=True)[:4])
    return None, summary


def flange_length_nominal(model: PartModel, bend: ModelBend, panel_key: int, stations: int) -> FlangeNominal:
    """Final-pose OML flange length of ``bend`` on ``panel_key`` (module docstring)."""
    side_index = bend.skin_side_index(panel_key, convex=True, graph=model.graph)
    polys = _side_polys(model, panel_key, side_index)
    origin, _ = bend.oml_sharp()
    w_dir = bend.outward_for(panel_key)
    results = far_boundary_stations(polys, origin, bend.axis_dir, w_dir, bend.length, stations)
    kind, ref, idx, valid, detail = _classify(model, bend.key, results, None)
    nominal = _finish(model, kind, ref, idx, valid, detail, origin, bend.axis_dir, w_dir, results, None)
    nominal.plurality = _plurality(results, bend.key, None)
    return nominal


def _sharp_crossing(origin: np.ndarray, a_dir: np.ndarray, w_dir: np.ndarray, u: float,
                    sharp: tuple[np.ndarray, np.ndarray]) -> float | None:
    """``s`` where the station line ``origin + u a + s w`` meets the projection of an OML sharp line."""
    q0, q1 = sharp
    d = q1 - q0
    du = float(d @ a_dir)
    if abs(du) < 0.1 * float(np.linalg.norm(d)):
        return None  # the far bend runs along the station direction: not a far boundary
    tau = (u - float((q0 - origin) @ a_dir)) / du
    q = q0 + tau * d
    return float((q - origin) @ w_dir)


def _finish(model: PartModel, kind, ref, idx, valid, detail, origin, a_dir, w_dir, results,
            sharp_of) -> FlangeNominal:
    station_values = [r[1] for r in results if r is not None]
    if kind == "FREE":
        values = np.asarray([results[i][1] for i in idx])
        clean = np.asarray([results[i][1] for i in idx if results[i][4]])
        # stations whose far edge runs parallel to the bend: the straight far edge (and notch
        # bottoms); corner rounds and chamfers at the corridor ends are not the flange length
        if len(clean) >= max(3, int(np.ceil(_CLEAN_FRACTION * len(values)))):
            level_value, level_detail = _dominant_level(clean)
            level_detail += f" ({len(clean)}/{len(values)} stations with a far edge parallel to the bend)"
        else:
            level_value, level_detail = _dominant_level(values)
        if level_value is None:
            return FlangeNominal(None, None, None, "NOMINAL_UNAVAILABLE",
                                 f"far free edge is stepped without a dominant level ({level_detail})", valid,
                                 float(values.max() - values.min()), station_values)
        return FlangeNominal(level_value, "FREE_EDGE", None, None, f"{detail}; {level_detail}", valid,
                             float(values.max() - values.min()), station_values)
    if kind == "ADJ":
        sharp = sharp_of(ref) if sharp_of else model.bends[ref].oml_sharp()
        values = [_sharp_crossing(origin, a_dir, w_dir, results[i][0], sharp) for i in idx]
        if any(v is None for v in values):
            return FlangeNominal(None, None, None, "NOMINAL_UNAVAILABLE",
                                 f"far bend runs along the flange direction ({detail})", valid)
        arr = np.asarray(values, dtype=np.float64)
        value = float(np.median(arr))
        if not value > 0:
            return FlangeNominal(None, None, None, "NOMINAL_UNAVAILABLE",
                                 f"adjacent bend sharp lies behind the bend ({value:.3f} mm)", valid)
        return FlangeNominal(value, "ADJACENT_BEND", ref, None, detail, valid, float(arr.max() - arr.min()),
                             station_values)
    if kind == "FEATURE":
        return FlangeNominal(None, None, None, "UNSUPPORTED_FEATURE", f"far boundary is a feature ({detail})", valid,
                             feature_ref=ref)
    return FlangeNominal(None, None, None, "NOMINAL_UNAVAILABLE", detail, valid)


def _flat_region(model: PartModel, start: int, blocked_bend: int, unformed: set[int]) -> tuple[list[int], list[int]]:
    """Panels reachable from ``start`` through unformed bends (not crossing ``blocked_bend``)."""
    adj = model.graph.adjacency()
    seen, stack, crossed = {start}, [start], []
    while stack:
        cur = stack.pop()
        for bk, other in adj[cur]:
            if bk == blocked_bend or bk not in unformed or other in seen:
                continue
            seen.add(other)
            crossed.append(bk)
            stack.append(other)
    return sorted(seen), sorted(set(crossed))


def backgauge_nominal(model: PartModel, bend: ModelBend, side_panels: tuple[int, int], formed: set[int],
                      transforms: dict[int, np.ndarray], neutral: dict[int, float], stations: int) -> BackgaugeNominal:
    """Backgauge nominal of the just-formed ``bend`` at the stage whose formed set is ``formed``."""
    unformed = set(model.bends) - formed
    skin = bend.oml_skin
    best = None
    for order, panel_key in enumerate(side_panels):
        region, crossed = _flat_region(model, panel_key, bend.key, unformed)
        t_side = transforms[panel_key]
        origin = apply4(transforms[bend.parent], bend.oml_sharp()[0])
        a_dir = unit(apply_dir4(transforms[bend.parent], bend.axis_dir))
        w_dir = unit(apply_dir4(t_side, bend.outward_for(panel_key)))
        polys: list[_Poly] = []
        for x in region:
            side_index = model.panels[x].skin_side.get(skin)
            if side_index is None:
                continue
            polys.extend(_side_polys(model, x, side_index, transforms[x]))
        for bk in crossed:
            polys.append(_strip_poly(model.bends[bk], skin, transforms[model.bends[bk].parent], neutral[bk]))
        results = far_boundary_stations(polys, origin, a_dir, w_dir, bend.length, stations)

        def sharp_of(ref: int):
            b2 = model.bends[ref]
            a, b = b2.oml_sharp()
            t = transforms[b2.parent]
            return apply4(t, a), apply4(t, b)

        kind, ref, idx, valid, detail = _classify(model, bend.key, results, formed)
        nominal = _finish(model, kind, ref, idx, valid, detail, origin, a_dir, w_dir, results, sharp_of)
        nominal.plurality = _plurality(results, bend.key, formed)
        developed = nominal.value if nominal.value is not None else -1.0
        score = (len(crossed), developed, -order)
        if best is None or score > best[0]:
            best = (score, BackgaugeNominal(panel_key, nominal, len(crossed), list(crossed)))
    return best[1]


def _strip_poly(bend: ModelBend, skin: int, parent_t: np.ndarray, neutral_length: float) -> _Poly:
    """Flat strip of an unformed bend on ``skin``, at the stage pose of its parent."""
    loci = bend.loci()
    tangent = loci["outer_tangent_parent"] if skin == bend.oml_skin else loci["inner_tangent_parent"]
    a, b = tangent
    shift = -bend.p_out * neutral_length
    pts = np.array([a, b, b + shift, a + shift])
    labels = np.array([LABEL_BEND, LABEL_FREE, LABEL_BEND, LABEL_FREE], dtype=np.int64)
    refs = np.array([bend.key, -1, bend.key, -1], dtype=np.int64)
    return _Poly(apply4(parent_t, pts), labels, refs)
