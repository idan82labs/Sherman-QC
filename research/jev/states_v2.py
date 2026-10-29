"""Typed, coordinate-free Jev states for bend candidates (v2 input designs).

The TypeSafe docs recommend named JSON fields with units and short meanings
(primitives/advanced: ``{"name", "type", "unit", "description"}``), only the
fields the question needs (jev-1.13 jaggedness #5: large state full of
irrelevant detail), and conversion of numbers into named buckets in code
(jaggedness #2: numeric representations). Three designs:

``typed``          each measurement is ``{"value", "unit", "meaning"}``, grouped by
                   what it describes (cross-section, fold angle, flanges, apex,
                   support, line); ``state.surroundings`` holds neighbour context.
``typed_reading``  ``typed`` plus a ``reading``: the value in words, from fixed,
                   physically motivated bands (not fitted to labels).
``composite``      the ``typed_reading`` state, asked the primary Noul plus four
                   atomic Nouls (``questions_v2.composite_v2``) combined in code.

Fail-closed rules (:func:`check_state_v2`): keys from a fixed vocabulary and
never an identity/geometry token; numbers finite and rounded to 3 significant
digits; every string must be one this module authored (meanings, units,
readings), so no free text from the data can reach the state; no lists of
numbers. Coordinates, part ids, filenames and SKUs therefore cannot appear.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from research.jev.state_v1 import DENY_TOKENS, GEOMETRY_FORBIDDEN_KEYS, INFERENCE_FORBIDDEN_KEYS

__all__ = ["DESIGNS", "STATE_SCHEMA_V2", "build_state_v2", "check_state_v2", "StateV2Error", "MEASURES",
           "record_id_v2"]

STATE_SCHEMA_V2 = "jev_bend_candidate_state_v2"
DESIGNS = ("typed", "typed_reading", "composite")


class StateV2Error(ValueError):
    pass


def _r3(v: float) -> float:
    if v == 0:
        return 0.0
    return float(f"{v:.3g}")


# (group, key, source, unit, meaning). source: ("feat"|"ctx"|"pool", field)
MEASURES: tuple[tuple[str, str, tuple[str, str], str, str], ...] = (
    ("cross_section", "two_flange_slices", ("feat", "two_flange_slice_fraction"), "fraction of slices",
     "Share of cross-section slices along the line that show two separate flanges meeting at the line."),
    ("cross_section", "one_flange_slices", ("feat", "one_flange_slice_fraction"), "fraction of slices",
     "Share of slices with material on one side of the line only, as at a cut edge."),
    ("cross_section", "closed_ring_slices", ("feat", "enclosed_slice_fraction"), "fraction of slices",
     "Share of slices where material surrounds the line on all sides, as inside a hole wall or a tube."),
    ("cross_section", "many_flange_slices", ("feat", "multi_flange_slice_fraction"), "fraction of slices",
     "Share of slices with three or more separate pieces of material around the line."),
    ("cross_section", "empty_slices", ("feat", "empty_slice_fraction"), "fraction of slices",
     "Share of slices with almost no scan points around the line."),
    ("fold_angle", "surface_normal_switch", ("pool", "pool_two_sided_transition_score"), "score from 0 to 1",
     "How clearly the scanned surface normals switch between two distinct flat directions across the line: "
     "1 is a clean two-sided fold, 0 is no switch."),
    ("fold_angle", "normal_switch_angle", ("pool", "pool_two_sided_normal_angle_deg"), "deg",
     "Angle between the two surface-normal directions on either side of the line."),
    ("fold_angle", "bend_angle", ("feat", "median_bend_angle_deg"), "deg",
     "How far the sheet turns at the line: 0 is flat, 90 is a right-angle fold (median over two-flange slices)."),
    ("fold_angle", "angle_stability", ("feat", "included_angle_stable_fraction"), "fraction of slices",
     "Share of two-flange slices whose angle is within 5 deg of the median angle."),
    ("fold_angle", "angle_spread", ("feat", "included_angle_iqr_deg"), "deg",
     "Interquartile range of the angle along the line; small means the angle is constant."),
    ("flanges", "flatter_flange_fit_residual", ("feat", "median_flatter_flange_fit_rms_mm"), "mm",
     "RMS distance of the flatter flange's points from a straight line in the slice; small means flat."),
    ("flanges", "rougher_flange_fit_residual", ("feat", "median_rougher_flange_fit_rms_mm"), "mm",
     "RMS distance of the rougher flange's points from a straight line in the slice; small means flat."),
    ("flanges", "shorter_flange_width", ("feat", "median_shorter_flange_width_mm"), "mm",
     "Visible width of the narrower flange next to the line (measured up to 40 mm)."),
    ("flanges", "longer_flange_width", ("feat", "median_longer_flange_width_mm"), "mm",
     "Visible width of the wider flange next to the line (measured up to 40 mm)."),
    ("flanges", "flange_balance", ("feat", "median_flange_balance_ratio"), "ratio",
     "Points on the smaller flange divided by points on the larger flange (1 = equal)."),
    ("apex", "corner_gap", ("feat", "median_corner_gap_mm"), "mm",
     "Gap between the sharp corner where the two flange planes would meet and the nearest scanned surface; "
     "a rounded formed bend leaves a gap of about its outer radius, a sharp cut edge leaves almost none."),
    ("apex", "apex_side", ("feat", "median_apex_side_mm"), "mm",
     "Where the material lies relative to the line, towards the inside of the fold angle: positive means "
     "the line is on the convex outer skin, about minus one sheet thickness means the concave inner skin."),
    ("apex", "line_to_corner", ("feat", "median_line_to_corner_mm"), "mm",
     "Distance from the line to the sharp corner where the two flange planes would meet; a line on the "
     "apex of a formed bend sits within a few mm of it."),
    ("apex", "bend_radius_estimate", ("pool", "pool_radius_mm"), "mm",
     "Bend radius from cylinder fits to the scan near the line, when available."),
    ("support", "line_support", ("feat", "near_support_fraction"), "fraction of line",
     "Share of the line that has scan points directly on it."),
    ("support", "points_near_line_distance", ("pool", "pool_median_line_distance_mm"), "mm",
     "Median distance of the scan points in a narrow corridor around the line from the line itself."),
    ("support", "corridor_coverage", ("pool", "pool_finite_support_fraction"), "fraction of line",
     "Share of the line's length with scan points in the corridor around it."),
    ("support", "distance_to_surface", ("feat", "median_surface_distance_mm"), "mm",
     "Median distance from the line to the nearest scanned point."),
    ("line", "length", ("feat", "line_length_mm"), "mm", "Length of the candidate line."),
    ("surroundings", "near_identical_proposals", ("ctx", "near_duplicate_count"), "count",
     "Other candidate proposals lying almost on this line (parallel, within 4 mm, overlapping): independent "
     "detections of the same feature."),
    ("surroundings", "parallel_lines_within_6mm", ("ctx", "parallel_within_6mm_count"), "count",
     "Other candidate lines parallel to this one and 0.4 to 6 mm away (possible other skin of the same fold)."),
    ("surroundings", "nearest_parallel_offset", ("ctx", "nearest_parallel_offset_mm"), "mm",
     "Distance to the closest such parallel candidate."),
    ("surroundings", "parallel_lines_6_to_40mm", ("ctx", "parallel_6_to_40mm_count"), "count",
     "Other parallel candidates 6 to 40 mm away (for example the other fold of a joggle, or a hem)."),
    ("surroundings", "collinear_partners", ("ctx", "collinear_partner_count"), "count",
     "Other candidates on the same straight carrier line, separated by a gap of up to 80 mm "
     "(as when a window or relief interrupts one fold)."),
    ("surroundings", "nearest_collinear_gap", ("ctx", "nearest_collinear_gap_mm"), "mm",
     "Gap to the nearest collinear partner."),
)


def _band(v: float, edges: tuple[float, ...], words: tuple[str, ...]) -> str:
    for e, w in zip(edges, words):
        if v < e:
            return w
    return words[-1]


_FRACTION = ((0.05, 0.3, 0.6, 0.9), ("none or almost none", "a few", "about half", "most", "nearly all"))
READINGS: dict[str, tuple[tuple[float, ...], tuple[str, ...]]] = {
    "two_flange_slices": _FRACTION, "one_flange_slices": _FRACTION, "closed_ring_slices": _FRACTION,
    "many_flange_slices": _FRACTION, "empty_slices": _FRACTION, "angle_stability": _FRACTION,
    "line_support": _FRACTION,
    "surface_normal_switch": ((0.3, 0.6, 0.8), ("no switch", "weak switch", "clear switch", "very clear switch")),
    "normal_switch_angle": ((8, 15, 35, 75, 105), ("almost flat", "gentle kink", "small formed bend",
                                                   "moderate bend", "about a right angle", "sharp fold")),
    "points_near_line_distance": ((0.5, 1.5, 3.0), ("on the line", "close", "spread out", "far from the line")),
    "corridor_coverage": _FRACTION,
    "line_to_corner": ((1.5, 4.0, 10.0), ("at the corner", "near the corner", "away from the corner",
                                          "far from the corner")),
    "near_identical_proposals": ((1, 3, 8), ("none", "a few", "several", "many")),
    "bend_angle": ((8, 15, 35, 75, 105), ("almost flat", "gentle kink", "small formed bend",
                                          "moderate bend", "about a right angle", "sharp fold")),
    "angle_spread": ((3, 8, 20), ("constant", "nearly constant", "varies", "varies strongly")),
    "flatter_flange_fit_residual": ((0.25, 0.6, 1.2), ("flat", "nearly flat", "uneven", "rough or curved")),
    "rougher_flange_fit_residual": ((0.25, 0.6, 1.2), ("flat", "nearly flat", "uneven", "rough or curved")),
    "shorter_flange_width": ((5, 12, 25, 39), ("very narrow", "narrow", "medium", "wide", "wider than 40 mm")),
    "longer_flange_width": ((5, 12, 25, 39), ("very narrow", "narrow", "medium", "wide", "wider than 40 mm")),
    "flange_balance": ((0.2, 0.5, 0.8), ("very unequal", "unequal", "fairly equal", "equal")),
    "corner_gap": ((0.3, 1.0, 3.0), ("sharp corner", "slightly rounded", "rounded", "large gap")),
    "apex_side": ((-0.5, 0.3), ("concave inner side", "on the surface", "convex outer side")),
    "bend_radius_estimate": ((1.0, 3.0, 8.0), ("tight", "typical tooling radius", "large", "very large")),
    "distance_to_surface": ((0.3, 1.0, 3.0), ("on the surface", "close", "off the surface", "far off")),
    "length": ((15, 40, 120), ("very short", "short", "medium", "long")),
    "parallel_lines_within_6mm": ((1, 2), ("none", "one", "several")),
    "parallel_lines_6_to_40mm": ((1, 2), ("none", "one", "several")),
    "collinear_partners": ((1, 2), ("none", "one", "several")),
    "nearest_parallel_offset": ((1.0, 2.5, 6.1), ("under 1 mm", "about one sheet thickness", "a few mm")),
    "nearest_collinear_gap": ((10, 40, 81), ("small gap", "medium gap", "large gap")),
}

_GROUP_MEANING = {
    "cross_section": "What a slice cut across the line shows around it.",
    "fold_angle": "The angle between the two flanges, where two flanges are seen.",
    "flanges": "The two pieces of sheet on either side of the line.",
    "apex": "Where the line sits on the fold.",
    "support": "How well the scan supports the line.",
    "line": "The candidate line itself.",
    "surroundings": "Other candidate lines found on the same scan.",
}

_MISSING = "not measured"


def _authored_strings() -> frozenset[str]:
    out = {STATE_SCHEMA_V2, _MISSING}
    for g, k, _s, unit, meaning in MEASURES:
        out |= {unit, meaning}
    for _edges, words in READINGS.values():
        out |= set(words)
    out |= set(_GROUP_MEANING.values())
    return frozenset(out)


AUTHORED = _authored_strings()
_KEYS = frozenset({"schema", "candidate", "surroundings", "value", "unit", "meaning", "reading",
                   "about_this_group"} | {m[0] for m in MEASURES} | {m[1] for m in MEASURES})


def _value(row: Mapping[str, Any], source: tuple[str, str]) -> float | None:
    block = row.get(source[0]) or {}
    v = block.get(source[1])
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return None
    return float(v)


def build_state_v2(row: Mapping[str, Any], design: str) -> dict[str, Any]:
    if design not in DESIGNS:
        raise KeyError(design)
    readings = design in ("typed_reading", "composite")
    cand: dict[str, dict[str, Any]] = {}
    sur: dict[str, Any] = {"about_this_group": _GROUP_MEANING["surroundings"]}
    for group, key, source, unit, meaning in MEASURES:
        v = _value(row, source)
        if v is None and source[0] == "ctx" and key.endswith(("offset", "gap")):
            continue  # no such neighbour: the count already says "none"
        entry: dict[str, Any] = {"value": _r3(v) if v is not None else _MISSING, "unit": unit, "meaning": meaning}
        if readings and v is not None:
            edges, words = READINGS[key]
            entry["reading"] = _band(v, edges, words)
        if group == "surroundings":
            sur[key] = entry
        else:
            cand.setdefault(group, {"about_this_group": _GROUP_MEANING[group]})[key] = entry
    state = {"schema": STATE_SCHEMA_V2, "candidate": cand, "surroundings": sur}
    check_state_v2(state)
    return state


def _walk(node: Any, path: str) -> None:
    if isinstance(node, Mapping):
        for k, v in node.items():
            if not isinstance(k, str) or k not in _KEYS:
                raise StateV2Error(f"{path}.{k}: key not in the v2 vocabulary")
            if k in GEOMETRY_FORBIDDEN_KEYS or k in INFERENCE_FORBIDDEN_KEYS or any(
                    t in DENY_TOKENS for t in k.split("_")):
                raise StateV2Error(f"{path}.{k}: denied key")
            _walk(v, f"{path}.{k}")
        return
    if isinstance(node, bool) or node is None:
        raise StateV2Error(f"{path}: booleans/nulls are not used in v2 states")
    if isinstance(node, (int, float)):
        if not math.isfinite(float(node)) or _r3(float(node)) != node:
            raise StateV2Error(f"{path}: number not finite / not rounded to 3 significant digits")
        return
    if isinstance(node, str):
        if node not in AUTHORED:
            raise StateV2Error(f"{path}: string is not authored by states_v2 (data text cannot enter a state)")
        return
    raise StateV2Error(f"{path}: unsupported type {type(node).__name__}")


def check_state_v2(state: Any) -> None:
    if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA_V2:
        raise StateV2Error("not a v2 state")
    if set(state) != {"schema", "candidate", "surroundings"}:
        raise StateV2Error(f"unexpected top-level keys {sorted(state)}")
    _walk(state, "state")


def record_id_v2(row: Mapping[str, Any]) -> str:
    key = f"{row['scan_sha256']}|{row['set']}|{row['cand']}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def state_sha(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
