"""Source-bound bijection from native folded CAD markers to existing vendor IDs.

This is nominal CAD identity, not scan correspondence or manufacturing truth.
No closest-available match: every marker requires exactly one admissible source
candidate and every candidate must be used exactly once.
"""
from __future__ import annotations

import math

MAX_BENDS = 32
MAX_COORDINATE_MM = 1_000_000
ENDPOINT_TOL_MM = 1e-7


class ManufacturingBridgeRefused(ValueError):
    pass


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ManufacturingBridgeRefused("INVALID_CAD_MARKER_POINT")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in value):
        raise ManufacturingBridgeRefused("INVALID_CAD_MARKER_POINT")
    try:
        result = tuple(float(v) for v in value)
    except OverflowError as exc:
        raise ManufacturingBridgeRefused("INVALID_CAD_MARKER_POINT") from exc
    if any(not math.isfinite(v) or abs(v) > MAX_COORDINATE_MM for v in result):
        raise ManufacturingBridgeRefused("INVALID_CAD_MARKER_POINT")
    return result


def _line(row):
    start, end = _point(row.get("line_start")), _point(row.get("line_end"))
    if math.dist(start, end) <= ENDPOINT_TOL_MM:
        raise ManufacturingBridgeRefused("DEGENERATE_CAD_MARKER")
    return start, end


def bind_vendor_bend_ids(markers, candidates, source_sha256):
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ManufacturingBridgeRefused("INVALID_SOURCE_HASH")
    if (not isinstance(markers, list) or not isinstance(candidates, list)
            or not 0 < len(markers) == len(candidates) <= MAX_BENDS):
        raise ManufacturingBridgeRefused("INCOMPLETE_NATIVE_VENDOR_BEND_SET")
    native_ids, vendor_ids, native_lines, vendor_lines = [], [], [], []
    for rows, key, ids, lines in ((markers, "pair_id", native_ids, native_lines),
                                (candidates, "feature_id", vendor_ids, vendor_lines)):
        for row in rows:
            if not isinstance(row, dict) or row.get("source_sha256") != source_sha256:
                raise ManufacturingBridgeRefused("BEND_SOURCE_HASH_MISMATCH")
            value = row.get(key)
            if not isinstance(value, str) or not value or len(value) > 200 or value in ids:
                raise ManufacturingBridgeRefused("INVALID_OR_DUPLICATE_BEND_ID")
            ids.append(value); lines.append(_line(row))
    matches = []
    for native in native_lines:
        admitted = []
        for index, vendor in enumerate(vendor_lines):
            error = min(max(math.dist(native[i], vendor[i]) for i in (0, 1)),
                        max(math.dist(native[i], vendor[1-i]) for i in (0, 1)))
            if error <= ENDPOINT_TOL_MM:
                admitted.append((index, error))
        if len(admitted) != 1:
            raise ManufacturingBridgeRefused("MISSING_OR_AMBIGUOUS_EXACT_CAD_MARKER")
        matches.append(admitted[0])
    if len({index for index, _ in matches}) != len(candidates):
        raise ManufacturingBridgeRefused("REUSED_VENDOR_BEND_ID")
    return {"schema": "SHERMAN_NATIVE_VENDOR_BEND_ID_BRIDGE_V1", "source_sha256": source_sha256,
            "nominal_cad_identity_only": True, "manufacturing_ready": False,
            "all_native_and_vendor_bends_bijective": True, "endpoint_tolerance_mm": ENDPOINT_TOL_MM,
            "assignments": [{"pair_id": native_ids[i], "feature_id": vendor_ids[index],
                             "endpoint_error_mm": error} for i, (index, error) in enumerate(matches)]}
