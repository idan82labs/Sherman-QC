"""Conditional CAD root grouping BEFORE geometry-only candidate contraction.

Native face/edge ownership is supplied explicitly; tessellated vertex indices or
coordinate welding never stand in for it. No scan observations or physical event
counts are produced. This pure research audit does not invoke a CAD kernel.
"""
from __future__ import annotations

import math
import re

import numpy as np


MAX_CANDIDATES = 256
MAX_FACES = 4096
MAX_EDGE_INCIDENCES = 32768
MAX_CANDIDATE_FACE_MEMBERSHIPS = 16384
MAX_PAIR_EVALUATIONS = 32640
_EPS = float(np.finfo(float).eps)
_TINY = float(np.finfo(float).tiny)


def _id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ValueError(f"{name} must be a bounded explicit identifier")
    return value


def _number(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _vector(value):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.ndim != 1 or len(value) != 3):
        raise ValueError("axis endpoints require three explicit coordinates")
    if any(isinstance(v, (bool, np.bool_))
           or not isinstance(v, (int, float, np.integer, np.floating)) for v in value):
        raise ValueError("axis endpoint coordinates must be real")
    if any(isinstance(v, (int, np.integer)) and abs(int(v)) > 2**53 for v in value):
        raise ValueError("axis endpoint integer precision is unsupported")
    try:
        p = np.array(value, dtype=float, copy=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError("axis endpoints must be representable") from exc
    if not np.isfinite(p).all():
        raise ValueError("axis endpoints must be finite")
    return p


def _id_list(value, name, maximum, *, empty=False):
    if not isinstance(value, (list, tuple)) or not (0 if empty else 1) <= len(value) <= maximum:
        raise ValueError(f"{name} must be an explicit bounded identifier list")
    ids = [_id(item, name) for item in value]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} cannot repeat identities")
    return sorted(ids)


def _components(nodes, adjacency):
    remaining, result = set(nodes), []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        found, stack = {seed}, [seed]
        while stack:
            current = stack.pop()
            neighbors = remaining & adjacency.get(current, set())
            remaining.difference_update(neighbors)
            found.update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        result.append(sorted(found))
    return result


def _pair_geometry(first, second):
    """Whole finite-segment lateral compatibility, with a separate axial gap.

    Endpoint distances are to the other supporting INFINITE line. The maximum
    bounds every point on a segment by convexity. The axial projected-interval
    gap is the maximum in both axis directions, not Euclidean closest-segment
    distance. It is used only on direct topology links, never as an all-pairs
    connected-strip test.
    """
    p, q, a, b = first["points"], second["points"], first["direction"], second["direction"]
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = np.asarray([p[0]-q[0], p[1]-q[0], q[0]-p[0], q[1]-p[0]])
        lateral = np.asarray([np.cross(offsets[0], b), np.cross(offsets[1], b),
                              np.cross(offsets[2], a), np.cross(offsets[3], a)])
        separation = max(math.hypot(*v) for v in lateral)
        angle = math.atan2(math.hypot(*np.cross(a, b)), abs(float(a@b)))
        # Check projected interval gaps in BOTH directions; keep original spans
        # in the output rather than replacing them with one enclosing segment.
        projected_q = [float((v-p[0])@a) for v in q]
        projected_p = [float((v-q[0])@b) for v in p]
        gap_first = max(0., min(projected_q)-first["length"], -max(projected_q))
        gap_second = max(0., min(projected_p)-second["length"], -max(projected_p))
        gap = max(gap_first, gap_second)
        magnitude = max(float(np.max(np.abs(p))), float(np.max(np.abs(q))))
        extent = max(first["length"], second["length"], *(math.hypot(*v) for v in offsets))
        guard = 512*_EPS*(magnitude+extent)
    if (not all(math.isfinite(v) for v in (separation, angle, gap, guard))
            or guard < _TINY):
        return None
    return dict(axis_angle=angle, axis_separation=separation, segment_gap=gap,
                distance_allowance=guard, angle_allowance=512*_EPS)


def group_cad_root_candidates(
    candidates, topology, *, unit, frame_id, max_axis_angle,
    max_axis_separation, max_segment_gap, max_pair_evaluations,
):
    """Partition only under a declared COMPLETE local native-incidence contract.

    Topology completeness and source identity are stipulations, not verified CAD
    provenance. No path through an un-nominated face establishes root identity.
    Any unresolved/incompatible component clears all grouping conclusions while
    retaining every immutable pre-merge candidate and its own finite segment.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be m or mm")
    frame_id = _id(frame_id, "frame_id")
    angle_limit = _number(max_axis_angle, "max_axis_angle")
    if angle_limit >= math.pi/2:
        raise ValueError("max_axis_angle must be less than pi/2 radians")
    separation_limit = _number(max_axis_separation, "max_axis_separation")
    gap_limit = _number(max_segment_gap, "max_segment_gap")
    if (isinstance(max_pair_evaluations, (bool, np.bool_))
            or not isinstance(max_pair_evaluations, (int, np.integer))
            or not 1 <= max_pair_evaluations <= MAX_PAIR_EVALUATIONS):
        raise ValueError("max_pair_evaluations must be an explicit supported integer")
    pair_cap = int(max_pair_evaluations)
    if not isinstance(candidates, (list, tuple)) or len(candidates) > MAX_CANDIDATES:
        raise ValueError("candidates must be an explicit bounded list")
    parsed, memberships = {}, 0
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
                "candidate_id", "owner_id", "face_ids", "axis_start", "axis_end"}:
            raise ValueError("candidate requires exactly identity, owner, faces and finite axis endpoints")
        name = _id(candidate["candidate_id"], "candidate_id")
        if name in parsed:
            raise ValueError("candidate identities must be unique")
        owner = _id(candidate["owner_id"], "owner_id") if candidate["owner_id"] is not None else None
        faces = _id_list(candidate["face_ids"], "face_ids", MAX_FACES)
        memberships += len(faces)
        if memberships > MAX_CANDIDATE_FACE_MEMBERSHIPS:
            raise ValueError("candidate face membership budget exceeded")
        p = np.asarray([_vector(candidate["axis_start"]), _vector(candidate["axis_end"])])
        with np.errstate(over="ignore", invalid="ignore"):
            v = p[1]-p[0]
            length = math.hypot(*v)
        if not math.isfinite(length) or length < _TINY:
            raise ValueError("axis endpoints must have positive normal representable separation")
        parsed[name] = dict(candidate_id=name, owner_id=owner, face_ids=faces,
                            points=p, direction=v/length, length=length)
    if not isinstance(topology, dict) or set(topology) != {"source_id", "status", "faces"}:
        raise ValueError("topology requires exactly source_id, status and faces")
    source_id = _id(topology["source_id"], "source_id")
    topology_status = topology["status"]
    if not isinstance(topology_status, str) or topology_status not in ("COMPLETE", "UNKNOWN"):
        raise ValueError("topology status must explicitly be COMPLETE or UNKNOWN")
    raw_faces = topology["faces"]
    if not isinstance(raw_faces, (list, tuple)) or len(raw_faces) > MAX_FACES:
        raise ValueError("topology faces must be an explicit bounded list")
    face_records, edges, incidences = {}, {}, 0
    for face in raw_faces:
        if not isinstance(face, dict) or set(face) != {"face_id", "owner_id", "edge_ids"}:
            raise ValueError("each source face requires exactly face_id, owner_id and edge_ids")
        name, owner = _id(face["face_id"], "face_id"), _id(face["owner_id"], "owner_id")
        if name in face_records:
            raise ValueError("source face identities must be unique")
        edge_ids = _id_list(face["edge_ids"], "edge_ids", MAX_EDGE_INCIDENCES, empty=True)
        incidences += len(edge_ids)
        if incidences > MAX_EDGE_INCIDENCES:
            raise ValueError("source edge incidence budget exceeded")
        face_records[name] = dict(face_id=name, owner_id=owner, edge_ids=edge_ids)
        for edge in edge_ids:
            edges.setdefault((owner, edge), []).append(name)
    records = [dict(candidate_id=c["candidate_id"], owner_id=c["owner_id"], face_ids=list(c["face_ids"]),
                    axis_start=c["points"][0].tolist(), axis_end=c["points"][1].tolist())
               for _, c in sorted(parsed.items())]
    result = {
        "schema_version": 1, "kind": "CAD_LOCAL_ROOT_GROUPING_AUDIT_V1", "status": "ABSTAINED",
        "production_disposition": "NOT_AUTHORIZED", "unit": unit, "frame_id": frame_id,
        "source_id": source_id, "input_candidates": records, "groups": [], "candidate_to_group": {},
        "group_hypothesis_count": None, "physical_bend_count": None, "observed_state": "UNKNOWN",
        "physical_geometry_is_verified": False, "event_identity_is_verified": False,
        "search_complete_within_declared_policy": False,
        "policy": {
            "max_axis_angle_radians": angle_limit, "max_axis_separation": separation_limit,
            "max_segment_gap": gap_limit, "max_pair_evaluations": pair_cap,
            "max_candidates": MAX_CANDIDATES, "max_faces": MAX_FACES,
            "max_edge_incidences": MAX_EDGE_INCIDENCES,
            "max_candidate_face_memberships": MAX_CANDIDATE_FACE_MEMBERSHIPS,
            "topology_status": topology_status,
            "topology_scope": "DECLARED_NATIVE_FACE_EDGE_IDENTITY_NOT_MESH_VERTEX_CONNECTIVITY",
            "connectivity_scope": "GRAPH_INDUCED_ONLY_BY_NOMINATED_ROOT_FACES_WITHIN_OWNER",
            "geometry_scope": "ALL_PAIRS_AXIS_CHECKS_FINITE_GAP_ONLY_ON_DIRECT_TOPOLOGY_LINKS",
            "segment_gap_scope": "MAX_AXIAL_PROJECTED_INTERVAL_GAP_IN_BOTH_DIRECTIONS_NOT_EUCLIDEAN_DISTANCE",
            "geometry_rule": "REQUIRE_THRESHOLD_MARGIN_EXCEED_ENGINEERING_ALLOWANCE",
            "no_convex_hull_span_or_geometry_average": True,
            "any_unresolved_component_clears_all_groups": True,
        },
        "diagnostics": {
            "candidate_count": len(parsed), "source_face_count": len(face_records),
            "edge_incidence_count": incidences, "candidate_face_memberships": memberships,
            "root_component_count": None, "pair_evaluation_count": 0,
            "max_distance_allowance": 0., "unresolved_candidate_ids": [],
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION", "TOPOLOGY_COMPLETENESS_AND_IDENTITY_STIPULATED_NOT_VERIFIED",
            "CAD_EXPECTATIONS_NOT_SCAN_OBSERVATIONS", "NO_REGISTRATION_OR_PHYSICAL_LOCALIZATION",
            "LOCAL_CONNECTED_SUPPORT_NOT_MANUFACTURING_EVENT_AUTHORITY",
            "NATIVE_EDGE_NONDEGENERACY_FACE_TRIMMING_AND_ORIENTED_MANIFOLD_NOT_VERIFIED",
            "CROSS_FACE_INNER_OUTER_PAIRING_AND_THICKNESS_UNESTABLISHED",
            "FINITE_SEGMENTS_RETAINED_NOT_CERTIFIED_PHYSICAL_MARKERS", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        ], "reasons": [],
    }

    def unresolved(reason, names):
        result["reasons"] = [reason]
        result["diagnostics"]["unresolved_candidate_ids"] = sorted(names)
        return result

    if topology_status != "COMPLETE":
        return unresolved("TOPOLOGY_COMPLETENESS_UNKNOWN", parsed)
    if any(len(owners) > 2 for owners in edges.values()):
        return unresolved("NONMANIFOLD_NATIVE_EDGE_INCIDENCE", parsed)
    unknown = [name for name, c in parsed.items() if c["owner_id"] is None
               or any(face not in face_records or face_records[face]["owner_id"] != c["owner_id"]
                      for face in c["face_ids"])]
    if unknown:
        return unresolved("MISSING_OR_INCONSISTENT_FACE_OWNERSHIP", unknown)
    nominated = {face for c in parsed.values() for face in c["face_ids"]}
    adjacency = {face: set() for face in nominated}
    for incident in edges.values():
        if len(incident) == 2 and set(incident) <= nominated:
            first, second = incident
            adjacency[first].add(second)
            adjacency[second].add(first)
    broken = [name for name, c in parsed.items() if len(_components(c["face_ids"], adjacency)) != 1]
    if broken:
        return unresolved("CANDIDATE_LOCAL_SUPPORT_DISCONNECTED", broken)
    roots = _components(nominated, adjacency)
    root_index = {face: i for i, component in enumerate(roots) for face in component}
    groups = [[] for _ in roots]
    for name, c in sorted(parsed.items()):
        groups[root_index[c["face_ids"][0]]].append(name)
    groups.sort(key=lambda names: tuple(names))
    result["diagnostics"]["root_component_count"] = len(groups)
    required = sum(len(group)*(len(group)-1)//2 for group in groups)
    if required > pair_cap:
        return unresolved("PAIR_EVALUATION_BUDGET_EXCEEDED", parsed)
    for group in groups:
        for i, first_id in enumerate(group):
            first = parsed[first_id]
            for second_id in group[i+1:]:
                second = parsed[second_id]
                metrics = _pair_geometry(first, second)
                result["diagnostics"]["pair_evaluation_count"] += 1
                if metrics is None:
                    return unresolved("NUMERICAL_RANGE_UNSUPPORTED", group)
                guard = metrics["distance_allowance"]
                result["diagnostics"]["max_distance_allowance"] = max(
                    result["diagnostics"]["max_distance_allowance"], guard)
                if (metrics["axis_angle"]+metrics["angle_allowance"] >= angle_limit
                        or metrics["axis_separation"]+guard >= separation_limit):
                    return unresolved("ROOT_COMPONENT_AXIS_COMPATIBILITY_UNRESOLVED", group)
                direct = (not set(first["face_ids"]).isdisjoint(second["face_ids"])
                          or any(adjacency[face] & set(second["face_ids"]) for face in first["face_ids"]))
                if direct and metrics["segment_gap"]+guard >= gap_limit:
                    return unresolved("DIRECT_ROOT_LINK_FINITE_GAP_UNRESOLVED", group)
    by_id = {record["candidate_id"]: record for record in records}
    for index, names in enumerate(groups):
        group_id = f"cad_root_group_{index:04d}"
        result["groups"].append({
            "group_id": group_id, "candidate_ids": names,
            "owner_id": parsed[names[0]]["owner_id"],
            "face_ids": sorted({face for name in names for face in parsed[name]["face_ids"]}),
            "member_segments": [dict(candidate_id=name, axis_start=by_id[name]["axis_start"],
                                     axis_end=by_id[name]["axis_end"]) for name in names],
            "observed_state": "UNKNOWN", "event_identity_is_verified": False,
            "scope": "CONNECTED_CAD_ROOT_GROUPING_HYPOTHESIS_NOT_PHYSICAL_OR_OBSERVED_EVENT",
        })
        result["candidate_to_group"].update({name: group_id for name in names})
    result.update(status="CAD_ROOT_GROUPING_HYPOTHESES", group_hypothesis_count=len(groups),
                  search_complete_within_declared_policy=True)
    if not records:
        result.update(status="ABSTAINED", reasons=["NO_CAD_CANDIDATES"])
    return result
