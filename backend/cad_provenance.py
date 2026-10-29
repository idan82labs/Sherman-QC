"""Bounded native-CAD bookkeeping and pre-contraction support handoff.

No CAD kernel, filesystem input, scan evidence or research engine is invoked.
Captured native incidence is import-local, not persistent or manufacturing-event
identity. Flange support is CONTEXT; this module never nominates it as bend roots.
"""
from __future__ import annotations

import math
import re

import numpy as np


MAX_FACES = 4096
MAX_EDGE_USES = 32768
MAX_RAW_VERTICES = 200000
MAX_CANDIDATES = 4096
MAX_SUPPORT_REFERENCES = 4000000
MAX_DETECTION_PASSES = 3


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**53


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is not None


def unavailable_native_topology(reason):
    return {
        "schema_version": 1, "kind": "CAD_NATIVE_TOPOLOGY_V1", "status": "UNAVAILABLE",
        "face_records": [], "owner_records": [], "reasons": [reason],
        "identity_scope": "IMPORT_LOCAL_NATIVE_OCCURRENCES_NOT_PERSISTENT_IDS",
        "native_geometry_is_verified": False, "event_identity_is_verified": False,
        "source_identity_is_verified": False, "observed_state": "UNKNOWN",
    }


def _sanitize_native_topology(native, tessellation):
    """Copy bounded incidence only when it covers the exact face-range ledger.

    This checks bookkeeping, NOT the native kernel, source bytes, physical units,
    manifold validity or manufacturing semantics. Extra authority flags are lost.
    """
    bad = unavailable_native_topology("MISSING_OR_INCONSISTENT_NATIVE_INCIDENCE")
    if not isinstance(native, dict) or not isinstance(tessellation, dict):
        return bad
    if (type(native.get("schema_version")) is not int or native["schema_version"] != 1
            or native.get("kind") != "CAD_NATIVE_TOPOLOGY_V1"
            or native.get("status") != "CAPTURED_IMPORT_LOCAL_INCIDENCE"
            or tessellation.get("mapping_status") != "FACE_OCCURRENCE_RANGES"):
        return bad
    faces, owners = native.get("face_records"), native.get("owner_records")
    ranges, count = tessellation.get("face_records"), tessellation.get("source_face_count")
    if (not isinstance(faces, list) or not isinstance(owners, list)
            or not isinstance(ranges, list) or not _integer(count)
            or len(faces) != len(ranges) or len(faces) != count or count > MAX_FACES
            or len(owners) > MAX_FACES):
        return bad
    owner_records, owner_ids, paths = [], set(), set()
    for owner in owners:
        if not isinstance(owner, dict):
            return bad
        identity, path, kind = owner.get("owner_id"), owner.get("occurrence_path"), owner.get("kind")
        if (not _identifier(identity) or identity in owner_ids or kind not in ("SOLID", "SHELL", "FACE")
                or not isinstance(path, list) or len(path) > 64 or not all(_integer(v) for v in path)
                or tuple(path) in paths):
            return bad
        owner_ids.add(identity)
        paths.add(tuple(path))
        owner_records.append(dict(owner_id=identity, kind=kind, occurrence_path=list(path)))
    records, edge_owners, edge_degeneracy, used_owners = [], {}, {}, set()
    uses = 0
    for index, (face, mesh_range) in enumerate(zip(faces, ranges)):
        if not isinstance(face, dict) or not isinstance(mesh_range, dict):
            return bad
        owner, edges = face.get("owner_id"), face.get("edge_uses")
        if (not _integer(face.get("source_face_index")) or face["source_face_index"] != index
                or not _integer(mesh_range.get("source_face_index")) or mesh_range["source_face_index"] != index
                or face.get("face_id") != f"face:{index}"
                or not isinstance(owner, str) or owner not in owner_ids or not isinstance(edges, list)):
            return bad
        uses += len(edges)
        if uses > MAX_EDGE_USES:
            return bad
        edge_records = []
        for edge in edges:
            if not isinstance(edge, dict):
                return bad
            identity, orientation, degenerate = edge.get("edge_id"), edge.get("orientation"), edge.get("degenerate")
            if (not _identifier(identity) or orientation not in ("FORWARD", "REVERSED", "INTERNAL", "EXTERNAL")
                    or type(degenerate) is not bool or edge_owners.get(identity, owner) != owner
                    or edge_degeneracy.get(identity, degenerate) != degenerate):
                return bad
            edge_owners[identity], edge_degeneracy[identity] = owner, degenerate
            edge_records.append(dict(edge_id=identity, orientation=orientation, degenerate=degenerate))
        used_owners.add(owner)
        records.append(dict(source_face_index=index, face_id=f"face:{index}", owner_id=owner, edge_uses=edge_records))
    if used_owners != owner_ids:
        return bad
    return {
        "schema_version": 1, "kind": "CAD_NATIVE_TOPOLOGY_V1",
        "status": "CAPTURED_IMPORT_LOCAL_INCIDENCE", "face_records": records,
        "owner_records": owner_records, "reasons": [],
        "identity_scope": "IMPORT_LOCAL_NATIVE_OCCURRENCES_NOT_PERSISTENT_IDS",
        "native_geometry_is_verified": False, "event_identity_is_verified": False,
        "source_identity_is_verified": False, "observed_state": "UNKNOWN",
    }


def sanitize_native_topology(native, tessellation):
    """Malformed foreign metadata cannot interrupt the geometry fallback."""
    try:
        return _sanitize_native_topology(native, tessellation)
    except (TypeError, ValueError, OverflowError, IndexError, KeyError, AttributeError):
        return unavailable_native_topology("MISSING_OR_INCONSISTENT_NATIVE_INCIDENCE")


def _rows(value, raw_count):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.ndim != 1
            or not 1 <= len(value) <= MAX_RAW_VERTICES):
        raise ValueError("invalid support membership")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           or not 0 <= int(v) < raw_count for v in value):
        raise ValueError("invalid source vertex")
    rows = np.asarray(value, dtype=np.int64)
    if len(rows) > 1 and not np.all(rows[1:] > rows[:-1]):
        raise ValueError("support indices must be sorted and unique")
    return rows


def _endpoint(value):
    if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3
            or isinstance(value, np.ndarray) and value.ndim != 1
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
                   or isinstance(v, (int, np.integer)) and abs(int(v)) > 2**53 for v in value)):
        raise ValueError("invalid finite endpoint")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite endpoint")
    return result


def _validate_support_transport(ledger, raw_count):
    """Check partitions and bind each source segment to its detection pass."""
    def rows(value, *, empty=False):
        if empty and isinstance(value, (list, tuple)) and not value:
            return np.empty(0, dtype=np.int64)
        return _rows(value, raw_count)

    def add(mask, members):
        if np.any(mask[members]):
            raise ValueError("reused row in source partition")
        mask[members] = True

    groups = ledger.get("processed_source_vertex_indices")
    if not isinstance(groups, (list, tuple)) or len(groups) > raw_count:
        raise ValueError("invalid preprocessing groups")
    kept = np.zeros(raw_count, dtype=bool)
    parsed_groups = []
    for group in groups:
        members = rows(group)
        add(kept, members)
        parsed_groups.append(members)
    lengths = np.asarray([len(group) for group in parsed_groups], dtype=np.int64)
    ordered_rows = np.concatenate(parsed_groups) if parsed_groups else np.empty(0, dtype=np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]] if parsed_groups else np.empty(0, dtype=np.int64)
    complete = kept.copy()
    add(complete, rows(ledger.get("preprocessing_rejected_source_vertex_indices"), empty=True))
    if not complete.all():
        raise ValueError("preprocessing does not account for all original rows")
    passes = ledger.get("segmentation_passes")
    if not isinstance(passes, list) or not 1 <= len(passes) <= MAX_DETECTION_PASSES:
        raise ValueError("invalid segmentation passes")
    segments = {}
    for pass_index, detection in enumerate(passes):
        if (not isinstance(detection, dict) or not _integer(detection.get("detection_pass"))
                or detection["detection_pass"] != pass_index):
            raise ValueError("invalid detection pass identity")
        partition = np.zeros(raw_count, dtype=bool)
        categories = np.full(raw_count, -1, dtype=np.int16)
        source_segments = detection.get("segments")
        if not isinstance(source_segments, (list, tuple)) or len(source_segments) > 50:
            raise ValueError("invalid segmentation ledger")
        for index, segment in enumerate(source_segments):
            expected_id = f"pass:{pass_index}:plane:{index}"
            if not isinstance(segment, dict) or segment.get("segment_id") != expected_id:
                raise ValueError("invalid original plane identity")
            members = rows(segment.get("source_vertex_indices"))
            add(partition, members)
            categories[members] = index
            segments[(pass_index, expected_id)] = members
        for category, key in enumerate(("rejected_source_vertex_indices", "unassigned_source_vertex_indices"), start=50):
            members = rows(detection.get(key), empty=True)
            add(partition, members)
            categories[members] = category
        if not np.array_equal(partition, kept):
            raise ValueError("segmentation does not partition retained raw rows")
        if not np.array_equal(categories[ordered_rows], np.repeat(categories[ordered_rows[starts]], lengths)):
            raise ValueError("one processed voxel cannot be split across segmentation categories")
    return segments


def _build_cad_provenance_handoff(provenance, selected_specs):
    """Recover original pre-dedup candidates, mapped to native CONTEXT faces.

    The complete primary extractor ledger survives alternative strategy selection.
    A selected legacy/modified spec without its own exact ledger remains unknown.
    No geometric nearest-neighbor attribution or grouping authorization is added.
    """
    result = {
        "schema_version": 1, "kind": "CAD_PRECONTRACTION_PROVENANCE_HANDOFF_V1",
        "status": "UNAVAILABLE", "reasons": [], "candidate_records": [],
        "selected_spec_lineage_complete": False, "selected_spec_memberships": [],
        "native_topology_status": "UNKNOWN", "root_role_status": "NOT_NOMINATED",
        "grouping_eligible": False, "grouping_inputs": None,
        "observed_state": "UNKNOWN", "event_identity_is_verified": False,
        "source_identity_is_verified": False, "physical_geometry_is_verified": False,
        "identity_scope": "IMPORT_INVOCATION_LOCAL_PRECONTRACTION_CANDIDATES",
    }

    def unavailable(reason):
        result.update(status="UNAVAILABLE", reasons=[reason], candidate_records=[],
                      selected_spec_memberships=[], selected_spec_lineage_complete=False)
        return result

    if not isinstance(provenance, dict):
        return unavailable("MISSING_CAD_PROVENANCE")
    native = sanitize_native_topology(provenance.get("native_topology"), provenance)
    result["native_topology_status"] = native["status"]
    if native["status"] != "CAPTURED_IMPORT_LOCAL_INCIDENCE":
        return unavailable("NATIVE_INCIDENCE_UNAVAILABLE")
    ledger = provenance.get("support_lineage")
    raw_count = provenance.get("vertex_count")
    if (not isinstance(ledger, dict) or ledger.get("kind") != "CAD_SUPPORT_LINEAGE_V1"
            or type(ledger.get("schema_version")) is not int or ledger["schema_version"] != 1
            or ledger.get("status") != "COMPLETE" or not _integer(raw_count)
            or not 0 < raw_count <= MAX_RAW_VERTICES or not _integer(ledger.get("raw_vertex_count"))
            or ledger["raw_vertex_count"] != raw_count):
        return unavailable("SUPPORT_LINEAGE_UNAVAILABLE")
    candidates = ledger.get("candidate_records")
    if not isinstance(candidates, list) or not 0 < len(candidates) <= MAX_CANDIDATES:
        return unavailable("NO_COMPLETE_BOUNDED_CANDIDATE_LEDGER")
    # Independently validate range coverage before index lookup. Empty face
    # blocks are legal; searchsorted(side='right') skips them correctly.
    stops, start = [], 0
    for mesh_range in provenance["face_records"]:
        span = mesh_range.get("vertex_range")
        if (not isinstance(span, list) or len(span) != 2 or not all(_integer(v) for v in span)
                or span[0] != start or not start <= span[1] <= raw_count):
            return unavailable("INVALID_VERTEX_FACE_RANGES")
        start = span[1]
        stops.append(start)
    if start != raw_count:
        return unavailable("INCOMPLETE_VERTEX_FACE_RANGES")
    stops = np.asarray(stops, dtype=np.int64)
    ids, original_records, references = set(), {}, 0
    try:
        source_segments = _validate_support_transport(ledger, raw_count)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("invalid candidate")
            identity, detection_pass = candidate.get("candidate_id"), candidate.get("detection_pass")
            if not _identifier(identity) or identity in ids or not _integer(detection_pass):
                raise ValueError("invalid candidate identity")
            endpoints = [_endpoint(candidate.get(key)) for key in ("bend_line_start", "bend_line_end")]
            with np.errstate(over="ignore", invalid="ignore"):
                length = math.hypot(*(endpoints[1] - endpoints[0]))
            if not math.isfinite(length) or length < np.finfo(float).tiny:
                raise ValueError("invalid finite segment")
            flanges = []
            for number in (1, 2):
                rows = _rows(candidate.get(f"flange{number}_source_vertex_indices"), raw_count)
                references += len(rows)
                pieces = candidate.get(f"flange{number}_segments")
                if not isinstance(pieces, list) or not 1 <= len(pieces) <= 50:
                    raise ValueError("missing pre-coplanar support segments")
                piece_ids, piece_rows, segments = set(), [], []
                for piece in pieces:
                    if not isinstance(piece, dict) or not _identifier(piece.get("segment_id")) or piece["segment_id"] in piece_ids:
                        raise ValueError("invalid source segment")
                    members = _rows(piece.get("source_vertex_indices"), raw_count)
                    source_members = source_segments.get((detection_pass, piece["segment_id"]))
                    if source_members is None or not np.array_equal(members, source_members):
                        raise ValueError("candidate does not reference its original source segment")
                    references += len(members)
                    if references > MAX_SUPPORT_REFERENCES:
                        raise ValueError("support budget exceeded")
                    piece_ids.add(piece["segment_id"])
                    piece_rows.append(members)
                    faces = sorted(set(int(v) for v in np.searchsorted(stops, members, side="right")))
                    segments.append(dict(segment_id=piece["segment_id"], source_vertex_indices=members.tolist(),
                                         context_face_ids=[f"face:{index}" for index in faces]))
                if not np.array_equal(np.unique(np.concatenate(piece_rows)), rows):
                    raise ValueError("source segment union disagrees")
                faces = sorted(set(int(v) for v in np.searchsorted(stops, rows, side="right")))
                flanges.append(dict(source_vertex_indices=rows.tolist(), segments=segments,
                                    context_face_ids=[f"face:{index}" for index in faces],
                                    owner_ids=sorted({native["face_records"][index]["owner_id"] for index in faces})))
            ids.add(identity)
            original_records[identity] = candidate
            result["candidate_records"].append(dict(
                candidate_id=identity, detection_pass=detection_pass,
                bend_line_start=endpoints[0].tolist(), bend_line_end=endpoints[1].tolist(),
                flanges=flanges, root_face_ids=[], root_role_status="NOT_NOMINATED"))
    except (ValueError, TypeError, OverflowError, IndexError):
        return unavailable("INVALID_OR_OVER_BUDGET_CANDIDATE_LINEAGE")
    if not isinstance(selected_specs, (list, tuple)) or len(selected_specs) > MAX_CANDIDATES:
        return unavailable("INVALID_SELECTED_SPEC_LEDGER")
    selected_ids, complete, memberships = set(), bool(selected_specs), []
    for index, spec in enumerate(selected_specs):
        local = getattr(spec, "cad_provenance", None)
        local_candidates = local.get("candidate_records") if isinstance(local, dict) else None
        valid = (isinstance(local, dict) and local.get("status") == "COMPLETE"
                 and type(local.get("schema_version")) is int and local["schema_version"] == 1
                 and local.get("kind") == "CAD_SUPPORT_LINEAGE_V1"
                 and _integer(local.get("raw_vertex_count")) and local["raw_vertex_count"] == raw_count
                 and isinstance(local_candidates, list) and 0 < len(local_candidates) <= MAX_CANDIDATES)
        members = []
        if valid:
            for record in local_candidates:
                identity = record.get("candidate_id") if isinstance(record, dict) else None
                # Equal IDs without equal original finite geometry/membership
                # are not the same record. These serialized records are JSON-like.
                try:
                    equal = (record == original_records[identity]) if _identifier(identity) and identity in ids else False
                except (ValueError, TypeError, OverflowError):
                    equal = False
                if (not _identifier(identity) or identity not in ids or identity in selected_ids
                        or type(equal) is not bool or not equal):
                    valid = False
                    break
                members.append(identity)
                selected_ids.add(identity)
        complete = complete and valid
        memberships.append(dict(selected_spec_index=index, status="CAPTURED" if valid else "UNAVAILABLE",
                                candidate_ids=members if valid else []))
    # Missing selected attribution does not destroy the independently retained
    # primary ledger, but cannot leave a successful selected attribution prefix.
    result.update(status="CAPTURED_SUPPORT_CONTEXT", reasons=["ROOT_ROLE_NOMINATION_REQUIRED"],
                  selected_spec_lineage_complete=complete,
                  selected_spec_memberships=memberships if complete else [])
    if not complete:
        result["reasons"].append("SELECTED_STRATEGY_LINEAGE_UNAVAILABLE")
    return result


def build_cad_provenance_handoff(provenance, selected_specs):
    """Return no partial handoff when a malformed sidecar cannot be inspected."""
    try:
        return _build_cad_provenance_handoff(provenance, selected_specs)
    except (TypeError, ValueError, OverflowError, IndexError, KeyError, AttributeError, FloatingPointError):
        result = _build_cad_provenance_handoff(None, [])
        result["reasons"] = ["MALFORMED_PROVENANCE_METADATA"]
        return result


def cad_provenance_summary(provenance):
    """Compact export; do not serialize the raw support ledger into reports."""
    handoff = provenance.get("handoff", {}) if isinstance(provenance, dict) else {}
    return {
        "status": handoff.get("status", "UNAVAILABLE"),
        "native_topology_status": handoff.get("native_topology_status", "UNKNOWN"),
        "precontraction_candidate_count": len(handoff.get("candidate_records", [])),
        "selected_spec_lineage_complete": handoff.get("selected_spec_lineage_complete") is True,
        "root_role_status": "NOT_NOMINATED", "grouping_eligible": False,
        "observed_state": "UNKNOWN", "reasons": list(handoff.get("reasons", [])),
    }
