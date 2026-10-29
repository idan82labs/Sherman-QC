"""Coordinate-free, identity-free bend-candidate states for Jev (C10, v1).

A *state* is the JSON object sent to Jev as the System One ``state`` field. It
describes one bend candidate with named, unit-suffixed scalars and short enum
strings only. :func:`build_state` turns a :class:`CandidateRecord` into a state
and **fails closed** (:class:`StateDenyError`) on anything that could carry
identity, location or truth:

Keys
    - the legacy ``INFERENCE_FORBIDDEN_KEYS`` (``scripts/build_f1_scan_only_
      candidate_pool.py``), matched exactly at every depth;
    - geometry keys (anchors, line endpoints, axis directions, polylines,
      centroids) matched exactly;
    - any key with an identity token: ids, uids, sku, part, job, recipe, scan,
      file, path, timestamp, date, serial, station, hash/sha, name, label, case,
      seed, split, family, operator, truth, gold, manual, review.
    Every numeric key must end with a unit suffix (:data:`UNIT_SUFFIXES`), every
    boolean key must start with ``is_``/``has_`` or end with ``_missing``, every
    string key must end with a category suffix (:data:`CATEGORY_SUFFIXES`).

Values
    - lists are rejected outright (so no 3-vector coordinate can appear);
    - strings must be short enum tokens: ``[A-Za-z][A-Za-z0-9_]{0,63}`` with no run
      of four or more digits, which rejects hashes, uuids, timestamps, SKUs and
      filenames;
    - finite numbers are rounded to 4 significant digits; NaN/inf become ``null``
      plus a sibling ``<key>_missing: true``;
    - nesting is at most :data:`MAX_DEPTH` levels.

The record id (``sha256(source_sha256 + "|" + candidate_key)[:24]``) is kept
outside the state. The CAD-assisted relation block (``cad_relation``) and the
part-level context block (``context``) are separate top-level keys so the CAD-free
and context-free ablations are a key drop (:func:`ablate`).

The JSON Schema ``schemas/jev_bend_candidate_state_v1.schema.json`` states the
same rules declaratively; a test checks that both agree.

This module also holds the line-geometry helpers that turn candidate lines into
coordinate-free descriptors (length, neighbour summary, relation to a reference
set). Coordinates are consumed here and never leave.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "STATE_SCHEMA",
    "CANDIDATE_SOURCES",
    "INFERENCE_FORBIDDEN_KEYS",
    "GEOMETRY_FORBIDDEN_KEYS",
    "DENY_TOKENS",
    "UNIT_SUFFIXES",
    "CATEGORY_SUFFIXES",
    "MAX_DEPTH",
    "TOP_LEVEL_BLOCKS",
    "StateDenyError",
    "CandidateRecord",
    "Line",
    "build_state",
    "record_id_for",
    "round_sig",
    "check_state",
    "ablate",
    "line_from_endpoints",
    "line_length_mm",
    "line_pair_metrics",
    "segment_distance_mm",
    "neighbour_summary",
    "relation_to_reference",
    "enum_token",
]

STATE_SCHEMA = "jev_bend_candidate_state_v1"

#: Allowed ``candidate_source`` values (a property of the exporter, not of a part).
CANDIDATE_SOURCES = frozenset(
    {"CAD_ONBOARDING", "E1_FLANGE_GRAPH", "SCAN_E3", "SCAN_LEGACY_REPORT", "SCAN_F1_POOL"}
)

#: Exact copy of ``scripts/build_f1_scan_only_candidate_pool.py`` INFERENCE_FORBIDDEN_KEYS
#: (not imported: research code must not pull the legacy scripts tree in).
INFERENCE_FORBIDDEN_KEYS = frozenset(
    {
        "point_indices",
        "source_points",
        "points",
        "bbox_points",
        "filename",
        "part_id",
        "expected_count",
        "manual_truth",
        "cad_metadata",
        "split_group_id",
    }
)

#: Exact keys that carry coordinates or directions in any source format.
GEOMETRY_FORBIDDEN_KEYS = frozenset(
    {
        "anchor",
        "anchors",
        "axis_direction",
        "bend_line_direction",
        "line_start",
        "line_end",
        "line_start_mm",
        "line_end_mm",
        "bend_line_start",
        "bend_line_end",
        "cad_line_start",
        "cad_line_end",
        "detected_line_start",
        "detected_line_end",
        "span_endpoints",
        "polyline_mm",
        "nominal_polyline_mm",
        "measured_polyline_mm",
        "centroid",
        "centroid_mm",
        "point_mm",
        "direction",
        "normal",
        "transform",
        "raw_to_frame",
    }
)

#: Identity / truth tokens. A key is denied when any ``_``-separated token is here.
DENY_TOKENS = frozenset(
    {
        "id", "ids", "uid", "uids", "sku", "skus", "part", "parts", "job", "jobs",
        "recipe", "recipes", "scan", "scans", "file", "files", "filename", "filenames",
        "path", "paths", "timestamp", "timestamps", "time", "date", "at", "serial",
        "station", "stations", "sha", "sha256", "hash", "name", "names", "label",
        "labels", "case", "cases", "seed", "seeds", "split", "splits", "family",
        "families", "operator", "truth", "gold", "manual", "review", "reviews",
        "reviewed", "anchor", "anchors", "endpoint", "endpoints", "coordinate",
        "coordinates", "xyz", "centroid", "polyline", "uuid",
    }
)

#: A numeric key must end with one of these (longest match wins; order is irrelevant).
UNIT_SUFFIXES = (
    "_mm", "_mm2", "_deg", "_count", "_fraction", "_ratio", "_score", "_pct",
    "_per_mm", "_eigenvalue", "_quantile", "_probability", "_log1p",
)

#: A string key must end with one of these.
CATEGORY_SUFFIXES = (
    "_kind", "_state", "_source", "_method", "_mode", "_status", "_convention",
    "_form", "_direction_sign", "_semantics", "_reason", "_family_kind",
)

#: Top-level keys a state may carry (beyond ``schema``).
TOP_LEVEL_BLOCKS = ("candidate", "neighbours", "context", "cad_relation")

MAX_DEPTH = 8

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_ENUM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_DIGIT_RUN_RE = re.compile(r"\d{4,}")


class StateDenyError(ValueError):
    """A state (or a record headed for one) breaks the deny-list or the format."""


# --------------------------------------------------------------------------
# Numbers and enum strings
# --------------------------------------------------------------------------


def round_sig(value: float | int, digits: int = 4) -> float | int:
    """Round to ``digits`` significant digits; ints stay ints, ``-0.0`` becomes ``0.0``."""
    if isinstance(value, bool):
        raise TypeError("bool is not a number here")
    if isinstance(value, int):
        if value == 0:
            return 0
        return int(float(f"{value:.{digits}g}"))
    if not math.isfinite(value):
        raise ValueError("non-finite value cannot be rounded")
    if value == 0.0:
        return 0.0
    rounded = float(f"{value:.{digits}g}")
    return 0.0 if rounded == 0.0 else rounded


def enum_token(value: Any) -> str | None:
    """Normalise a source enum to a state token, or ``None`` when it is not a safe token."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"[\s\-./:]+", "_", text)
    if not _ENUM_RE.fullmatch(text) or _DIGIT_RUN_RE.search(text):
        return None
    return text


# --------------------------------------------------------------------------
# Deny-list check
# --------------------------------------------------------------------------


def _key_problem(key: str) -> str | None:
    if key in INFERENCE_FORBIDDEN_KEYS:
        return "INFERENCE_FORBIDDEN_KEY"
    if key in GEOMETRY_FORBIDDEN_KEYS:
        return "GEOMETRY_KEY"
    if not _KEY_RE.fullmatch(key):
        return "KEY_FORMAT"
    if any(token in DENY_TOKENS for token in key.split("_")):
        return "IDENTITY_TOKEN"
    return None


def _has_suffix(key: str, suffixes: Sequence[str]) -> bool:
    return any(key.endswith(s) for s in suffixes)


def _check_leaf(key: str, value: Any, path: str, siblings: Mapping[str, Any]) -> None:
    if isinstance(value, bool):
        if not (key.startswith(("is_", "has_")) or key.endswith("_missing")):
            raise StateDenyError(f"{path}: boolean key must start with is_/has_ or end with _missing")
        if key.endswith("_missing") and value is not True:
            raise StateDenyError(f"{path}: a _missing flag is only ever true")
        return
    if value is None:
        if siblings.get(f"{key}_missing") is not True:
            raise StateDenyError(f"{path}: null needs a sibling '{key}_missing': true")
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise StateDenyError(f"{path}: non-finite number")
        if not _has_suffix(key, UNIT_SUFFIXES):
            raise StateDenyError(f"{path}: numeric key needs a unit suffix {UNIT_SUFFIXES}")
        if round_sig(value) != value:
            raise StateDenyError(f"{path}: number is not rounded to 4 significant digits")
        return
    if isinstance(value, str):
        if not _has_suffix(key, CATEGORY_SUFFIXES) and key != "schema":
            raise StateDenyError(f"{path}: string key needs a category suffix {CATEGORY_SUFFIXES}")
        if key == "schema":
            if value != STATE_SCHEMA:
                raise StateDenyError(f"{path}: schema must be {STATE_SCHEMA}")
            return
        if not _ENUM_RE.fullmatch(value) or _DIGIT_RUN_RE.search(value):
            raise StateDenyError(f"{path}: string value is not a short enum token")
        return
    if isinstance(value, (list, tuple)):
        raise StateDenyError(f"{path}: lists are not allowed in a state (named scalars only)")
    raise StateDenyError(f"{path}: unsupported value type {type(value).__name__}")


def _check_node(node: Mapping[str, Any], path: str, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise StateDenyError(f"{path}: nesting deeper than {MAX_DEPTH}")
    for key, value in node.items():
        if not isinstance(key, str):
            raise StateDenyError(f"{path}: non-string key")
        child = f"{path}.{key}"
        problem = _key_problem(key) if not (depth == 1 and key == "schema") else None
        if problem:
            raise StateDenyError(f"{child}: denied key ({problem})")
        if isinstance(value, Mapping):
            if key.endswith("_missing"):
                raise StateDenyError(f"{child}: a _missing flag must be a boolean")
            _check_node(value, child, depth + 1)
        else:
            _check_leaf(key, value, child, node)


def check_state(state: Any) -> None:
    """Raise :class:`StateDenyError` unless ``state`` is a valid v1 state."""
    if not isinstance(state, Mapping):
        raise StateDenyError("state must be an object")
    if state.get("schema") != STATE_SCHEMA:
        raise StateDenyError(f"state.schema must be {STATE_SCHEMA}")
    if state.get("candidate_source") not in CANDIDATE_SOURCES:
        raise StateDenyError("state.candidate_source is missing or unknown")
    if not isinstance(state.get("candidate"), Mapping) or not state["candidate"]:
        raise StateDenyError("state.candidate must be a non-empty object")
    allowed_top = {"schema", "candidate_source", "units_convention", *TOP_LEVEL_BLOCKS}
    extra = sorted(set(state) - allowed_top)
    if extra:
        raise StateDenyError(f"unexpected top-level keys: {extra}")
    if state.get("units_convention") != "MM_DEG":
        raise StateDenyError("state.units_convention must be MM_DEG")
    for block in TOP_LEVEL_BLOCKS:
        if block in state and not isinstance(state[block], Mapping):
            raise StateDenyError(f"state.{block} must be an object")
    _check_node(state, "state", 1)


# --------------------------------------------------------------------------
# Records -> states
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate as an exporter sees it, before it becomes a state.

    ``candidate_key`` and ``source_sha256`` identify it and stay out of the state.
    The four blocks hold *raw* scalars; :func:`build_state` normalises them.
    """

    source_sha256: str
    candidate_key: str
    candidate_source: str
    candidate: Mapping[str, Any]
    neighbours: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    cad_relation: Mapping[str, Any] | None = None


def record_id_for(source_sha256: str, candidate_key: str) -> str:
    """``sha256(source_sha256 + "|" + candidate_key)[:24]``."""
    if not source_sha256 or not candidate_key:
        raise ValueError("record id needs a source sha and a candidate key")
    return hashlib.sha256(f"{source_sha256}|{candidate_key}".encode("utf-8")).hexdigest()[:24]


def _normalise_block(block: Mapping[str, Any], path: str, depth: int) -> dict[str, Any]:
    if depth > MAX_DEPTH:
        raise StateDenyError(f"{path}: nesting deeper than {MAX_DEPTH}")
    out: dict[str, Any] = {}
    for key in sorted(block):
        value = block[key]
        if isinstance(value, Mapping):
            nested = _normalise_block(value, f"{path}.{key}", depth + 1)
            if nested:
                out[key] = nested
            continue
        if isinstance(value, bool) or value is None or isinstance(value, str):
            if value is None:
                out[key] = None
                out[f"{key}_missing"] = True
            else:
                out[key] = value
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
                out[f"{key}_missing"] = True
            else:
                out[key] = round_sig(value)
            continue
        # numpy scalars
        item = getattr(value, "item", None)
        if callable(item) and not isinstance(value, (list, tuple)):
            scalar = item()
            if isinstance(scalar, (int, float, bool)):
                out.update(_normalise_block({key: scalar}, path, depth))
                continue
        raise StateDenyError(f"{path}.{key}: unsupported value type {type(value).__name__}")
    return out


def build_state(record: CandidateRecord) -> tuple[str, dict[str, Any]]:
    """Return ``(record_id, state)`` for ``record``; raise :class:`StateDenyError` on any leak."""
    if record.candidate_source not in CANDIDATE_SOURCES:
        raise StateDenyError(f"unknown candidate_source {record.candidate_source!r}")
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "candidate_source": record.candidate_source,
        "units_convention": "MM_DEG",
        "candidate": _normalise_block(record.candidate, "state.candidate", 2),
    }
    if record.neighbours:
        state["neighbours"] = _normalise_block(record.neighbours, "state.neighbours", 2)
    if record.context:
        state["context"] = _normalise_block(record.context, "state.context", 2)
    if record.cad_relation:
        state["cad_relation"] = _normalise_block(record.cad_relation, "state.cad_relation", 2)
    for block in TOP_LEVEL_BLOCKS[1:]:
        if block in state and not state[block]:
            del state[block]
    check_state(state)
    return record_id_for(record.source_sha256, record.candidate_key), state


def ablate(state: Mapping[str, Any], *, cad_free: bool = False, context_free: bool = False) -> dict[str, Any]:
    """Copy of ``state`` without the CAD-assisted and/or the part-context block."""
    out = dict(state)
    if cad_free:
        out.pop("cad_relation", None)
    if context_free:
        out.pop("context", None)
    return out


# --------------------------------------------------------------------------
# Line geometry -> coordinate-free descriptors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    """A finite segment (mm). Used inside exporters only; never serialised."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def midpoint(self) -> tuple[float, float, float]:
        return tuple((a + b) / 2.0 for a, b in zip(self.start, self.end))  # type: ignore[return-value]

    @property
    def axis(self) -> tuple[float, float, float]:
        length = self.length
        return tuple((b - a) / length for a, b in zip(self.start, self.end))  # type: ignore[return-value]


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out  # type: ignore[return-value]


def line_from_endpoints(start: Any, end: Any, *, min_length_mm: float = 1e-6) -> Line | None:
    """A :class:`Line` from two 3-vectors, or ``None`` when either is unusable or degenerate."""
    a, b = _vec3(start), _vec3(end)
    if a is None or b is None or math.dist(a, b) <= min_length_mm:
        return None
    return Line(a, b)


def line_from_polyline(polyline: Any) -> Line | None:
    """The chord from the first to the last vertex of a polyline."""
    if not isinstance(polyline, (list, tuple)) or len(polyline) < 2:
        return None
    return line_from_endpoints(polyline[0], polyline[-1])


def line_length_mm(line: Line | None) -> float | None:
    return None if line is None else line.length


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def line_pair_metrics(reference: Line, other: Line) -> dict[str, float]:
    """Axis angle (deg), symmetric transverse offset (mm) and overlap fractions.

    Same definitions as ``scripts/evaluate_f1_against_manual_gold.py`` ``pair_metrics``
    (re-implemented, not imported).
    """
    ra, oa = reference.axis, other.axis
    dot = max(-1.0, min(1.0, _dot(ra, oa)))
    angle = math.degrees(math.acos(abs(dot)))
    if dot < 0:
        oa = tuple(-v for v in oa)
    delta = tuple(o - r for o, r in zip(other.midpoint, reference.midpoint))

    def _perp(axis: Sequence[float]) -> float:
        along = _dot(delta, axis)
        return math.sqrt(max(0.0, _dot(delta, delta) - along * along))

    transverse = max(_perp(ra), _perp(oa))
    ref_interval = sorted(_dot(tuple(p - m for p, m in zip(pt, reference.midpoint)), ra)
                          for pt in (reference.start, reference.end))
    oth_interval = sorted(_dot(tuple(p - m for p, m in zip(pt, reference.midpoint)), ra)
                          for pt in (other.start, other.end))
    overlap = max(0.0, min(ref_interval[1], oth_interval[1]) - max(ref_interval[0], oth_interval[0]))
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": transverse,
        "reference_overlap_fraction": overlap / max(reference.length, 1e-9),
        "other_overlap_fraction": overlap / max(other.length, 1e-9),
    }


def segment_distance_mm(a: Line, b: Line) -> float:
    """True closest distance between two finite 3D segments (mm).

    Closest points of two segments (Ericson, *Real-Time Collision Detection*, 5.1.9),
    so a long line that crosses a reference is at distance 0 and a collinear line far
    along the axis is as far away as its nearest endpoint.
    """
    d1 = tuple(e - s for s, e in zip(a.start, a.end))
    d2 = tuple(e - s for s, e in zip(b.start, b.end))
    r = tuple(p - q for p, q in zip(a.start, b.start))
    aa, ee, ff = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    eps = 1e-12
    if aa <= eps and ee <= eps:
        return math.dist(a.start, b.start)
    if aa <= eps:
        s, t = 0.0, max(0.0, min(1.0, ff / ee))
    else:
        c = _dot(d1, r)
        if ee <= eps:
            s, t = max(0.0, min(1.0, -c / aa)), 0.0
        else:
            bb = _dot(d1, d2)
            denom = aa * ee - bb * bb
            s = max(0.0, min(1.0, (bb * ff - c * ee) / denom)) if denom > eps else 0.0
            t = (bb * s + ff) / ee
            if t < 0.0:
                s, t = max(0.0, min(1.0, -c / aa)), 0.0
            elif t > 1.0:
                s, t = max(0.0, min(1.0, (bb - c) / aa)), 1.0
    p1 = tuple(p + s * d for p, d in zip(a.start, d1))
    p2 = tuple(q + t * d for q, d in zip(b.start, d2))
    return math.dist(p1, p2)


def neighbour_summary(index: int, lines: Sequence[Line | None]) -> dict[str, Any]:
    """Coordinate-free summary of the other candidate lines around ``lines[index]``.

    Counts of parallel neighbours (within 2 deg and 5 deg) at transverse offsets of
    up to 5 mm and 25 mm with any overlap, the nearest parallel neighbour's offset
    and the candidate's length relative to the longest and median candidate.
    """
    me = lines[index]
    others = [line for j, line in enumerate(lines) if j != index and line is not None]
    out: dict[str, Any] = {"other_candidates_count": len(others)}
    if me is None:
        return out
    lengths = sorted(line.length for line in lines if line is not None)
    longest = lengths[-1]
    median = lengths[len(lengths) // 2] if len(lengths) % 2 else 0.5 * (lengths[len(lengths) // 2 - 1] + lengths[len(lengths) // 2])
    out["length_to_longest_ratio"] = me.length / longest if longest > 0 else None
    out["length_to_median_ratio"] = me.length / median if median > 0 else None
    near_parallel: list[float] = []
    counts = {"parallel_2deg_within_5mm_count": 0, "parallel_2deg_within_25mm_count": 0,
              "parallel_5deg_within_5mm_count": 0, "parallel_5deg_within_25mm_count": 0,
              "parallel_2deg_any_offset_count": 0}
    for other in others:
        m = line_pair_metrics(me, other)
        overlapping = m["reference_overlap_fraction"] > 0.0
        if m["axis_angle_deg"] <= 2.0:
            counts["parallel_2deg_any_offset_count"] += 1
            near_parallel.append(m["transverse_offset_mm"])
        for limit_deg in (2, 5):
            if m["axis_angle_deg"] > limit_deg or not overlapping:
                continue
            for limit_mm in (5, 25):
                if m["transverse_offset_mm"] <= limit_mm:
                    counts[f"parallel_{limit_deg}deg_within_{limit_mm}mm_count"] += 1
    out.update(counts)
    out["nearest_parallel_offset_mm"] = min(near_parallel) if near_parallel else None
    return out


def relation_to_reference(line: Line | None, references: Iterable[tuple[Line, Mapping[str, Any]]]) -> dict[str, Any]:
    """Relation of ``line`` to the nearest reference line (e.g. the nearest recipe bend).

    The nearest reference minimises ``transverse_offset_mm + 0.5 mm per degree``
    of axis angle. Extra numeric attributes of that reference (already
    unit-suffixed, e.g. ``nominal_sweep_deg``) are copied with a ``nearest_`` prefix.
    """
    refs = [(ref, attrs) for ref, attrs in references if ref is not None]
    out: dict[str, Any] = {"reference_bends_count": len(refs)}
    if line is None or not refs:
        out["nearest_offset_mm"] = None
        out["nearest_axis_angle_deg"] = None
        return out
    scored = []
    for ref, attrs in refs:
        m = line_pair_metrics(ref, line)
        scored.append((m["transverse_offset_mm"] + 0.5 * m["axis_angle_deg"], m, ref, attrs))
    scored.sort(key=lambda item: item[0])
    _, best, ref, attrs = scored[0]
    out["nearest_offset_mm"] = best["transverse_offset_mm"]
    out["nearest_axis_angle_deg"] = best["axis_angle_deg"]
    out["nearest_overlap_fraction"] = best["reference_overlap_fraction"]
    out["nearest_length_ratio"] = line.length / ref.length if ref.length > 0 else None
    for key, value in sorted(attrs.items()):
        out[f"nearest_{key}"] = value
    # Within 10 mm means the segments themselves come that close (true segment distance)
    # with roughly parallel axes; a collinear line far along the axis is not "within".
    out["is_within_10mm_of_reference"] = any(
        m["axis_angle_deg"] <= 10.0 and segment_distance_mm(line, ref) <= 10.0 for _, m, ref, _ in scored
    )
    return out
