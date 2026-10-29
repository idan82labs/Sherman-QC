"""Research-only V40 parent-bridge certificate for obtuse folds.

V40 addresses the projection degeneracy where both tails of an obtuse fold
land on the same V34 transverse half-axis.  It restores only candidates whose
source mesh topology independently shows one or two connected skins, each
carrying broad axial support for both proposed parent planes and reaching the
candidate ridge.  Decisions are restored before ordinary NMS.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from backend.f1_final_line_accounting_v39 import select_unified_v39
from backend.f1_ordered_projective_gauss_transport import (
    ACTION_LEGACY,
    ACTION_RESCUE,
    ordinary_nms,
    parse_candidates,
)
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "V40_OBTUSE_PARENT_BRIDGE_V1"
PRIMITIVE = "CONNECTED_SOURCE_SKIN_PARENT_BRIDGE_CERTIFICATE_V1"
MIN_OBTUSE_TURN_DEGREES = 90.0
MAX_OBTUSE_TURN_DEGREES = 170.0
MAX_BRIDGING_SKINS = 2
NORMAL_ALIGNMENT_COSINE = math.cos(math.radians(15.0))
MIN_PARENT_AREA_SCALE2 = 4.0
MIN_PARENT_AXIAL_COVERAGE = 0.5
MAX_RIDGE_DISTANCE_SCALE = 2.0


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def parent_bridge_certificate(raw_source: dict[str, Any], raw_candidate: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(raw_source.get("positions", []), dtype=np.float64)
    faces = np.asarray(raw_source.get("triangles", []), dtype=np.int64)
    count = len(positions)
    parent = list(range(count))
    size = [1] * count

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for face in faces:
        a, b, c = (int(value) for value in face)
        union(a, b)
        union(a, c)

    n_a = np.asarray(raw_candidate["parent_A"]["unit_normal_xyz"], dtype=np.float64)
    n_b = np.asarray(raw_candidate["parent_B"]["unit_normal_xyz"], dtype=np.float64)
    n_a /= np.linalg.norm(n_a)
    n_b /= np.linalg.norm(n_b)
    p0 = np.asarray(raw_candidate["render_line_start"], dtype=np.float64)
    p1 = np.asarray(raw_candidate["render_line_end"], dtype=np.float64)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    axis /= length
    scale = float(raw_candidate["local_scale_s"])
    area_floor = MIN_PARENT_AREA_SCALE2 * scale * scale
    component_rows: dict[int, dict[str, Any]] = {}

    for face in faces:
        a, b, c = (int(value) for value in face)
        triangle = positions[[a, b, c]]
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        doubled_area = float(np.linalg.norm(cross))
        if doubled_area <= 0.0:
            continue
        normal = cross / doubled_area
        area = 0.5 * doubled_area
        root = find(a)
        row = component_rows.setdefault(
            root,
            {"area_a": 0.0, "area_b": 0.0, "u_a": [], "u_b": [], "minimum_ridge_distance": math.inf},
        )
        radial = triangle - p0 - np.outer((triangle - p0) @ axis, axis)
        row["minimum_ridge_distance"] = min(row["minimum_ridge_distance"], float(np.min(np.linalg.norm(radial, axis=1))))
        axial_values = ((triangle - p0) @ axis).tolist()
        if abs(float(np.dot(normal, n_a))) >= NORMAL_ALIGNMENT_COSINE:
            row["area_a"] += area
            row["u_a"].extend(axial_values)
        if abs(float(np.dot(normal, n_b))) >= NORMAL_ALIGNMENT_COSINE:
            row["area_b"] += area
            row["u_b"].extend(axial_values)

    details = []
    bridging = 0
    for root in sorted(component_rows):
        row = component_rows[root]
        span_a = max(row["u_a"]) - min(row["u_a"]) if row["u_a"] else 0.0
        span_b = max(row["u_b"]) - min(row["u_b"]) if row["u_b"] else 0.0
        passed = bool(
            row["area_a"] >= area_floor
            and row["area_b"] >= area_floor
            and span_a >= MIN_PARENT_AXIAL_COVERAGE * length
            and span_b >= MIN_PARENT_AXIAL_COVERAGE * length
            and row["minimum_ridge_distance"] <= MAX_RIDGE_DISTANCE_SCALE * scale
        )
        bridging += int(passed)
        details.append(
            {
                "area_a": float(row["area_a"]),
                "area_b": float(row["area_b"]),
                "axial_span_a": float(span_a),
                "axial_span_b": float(span_b),
                "minimum_ridge_distance": float(row["minimum_ridge_distance"]),
                "bridges_parents": passed,
            }
        )
    turn = abs(float(raw_candidate["signed_turn_alpha"]))
    component_count = len(component_rows)
    passed = bool(
        MIN_OBTUSE_TURN_DEGREES < turn <= MAX_OBTUSE_TURN_DEGREES
        and 1 <= bridging <= MAX_BRIDGING_SKINS
        and component_count == bridging
    )
    return {
        "passed": passed,
        "signed_turn_degrees": turn,
        "source_component_count": component_count,
        "bridging_skin_count": bridging,
        "maximum_bridging_skins": MAX_BRIDGING_SKINS,
        "component_details": details,
    }


def select_unified_v40(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    base = select_unified_v39(raw_source, raw_candidates, config_sha256=config_sha256, worker_count=worker_count)
    candidates = parse_candidates(raw_candidates)
    decisions = normalize_numpy_scalars([dict(value) for value in base["decisions"]])
    visuals = [dict(value) for value in base["_ephemeral"]["candidate_visuals"]]
    restored_ids: set[int] = set()
    for candidate, raw_candidate, decision, visual in zip(candidates, raw_candidates, decisions, visuals):
        predicates = list(decision.get("predicate_passes", []))
        only_tail_transition_failed = bool(
            len(predicates) == 7
            and predicates[1]
            and predicates[2]
            and not predicates[3]
            and not predicates[4]
            and predicates[5]
            and predicates[6]
        )
        certificate = parent_bridge_certificate(raw_source, raw_candidate)
        certificate["only_tail_and_transition_failed"] = only_tail_transition_failed
        certificate["passed"] = bool(certificate["passed"] and only_tail_transition_failed)
        decision["v40_obtuse_parent_bridge_certificate"] = certificate
        visual["v40_obtuse_parent_bridge_certificate"] = certificate
        if decision["action"] == "NO_RESCUE" and certificate["passed"]:
            predicates[3] = True
            predicates[4] = True
            predicates.append(True)
            decision["predicate_passes"] = predicates
            decision["action"] = ACTION_RESCUE
            decision["reason"] = "OPGT_PASS"
            decision["v40_restored_obtuse_fold"] = True
            visual["first_failing_predicate"] = "OPGT_PASS"
            visual["v40_restored_obtuse_fold"] = True
            restored_ids.add(int(decision["candidate_id"]))

    pairs = list(zip(candidates, decisions))
    legacy = [pair for pair in pairs if pair[1]["action"] == ACTION_LEGACY]
    additions = [pair for pair in pairs if pair[1]["action"] == ACTION_RESCUE]
    ordered = sorted(legacy, key=lambda row: (row[0].frozen_order, row[0].candidate_id))
    ordered += sorted(additions, key=lambda row: row[0].candidate_id)
    accepted, suppressed = ordinary_nms(ordered)
    for decision in decisions:
        decision["nms_reason"] = "ORDINARY_NMS_SUPPRESSED" if int(decision["candidate_id"]) in suppressed else ""
    final_lines = [
        {
            "candidate_id": candidate.candidate_id,
            "line_start": candidate.raw["render_line_start"],
            "line_end": candidate.raw["render_line_end"],
            "score": candidate.raw.get("score"),
            "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v40_obtuse_parent_bridge_rescue",
        }
        for candidate, decision in accepted
    ]
    decisions = normalize_numpy_scalars(decisions)
    final_lines = normalize_numpy_scalars(final_lines)
    rescue_only = sum(value["source_lane"] != "legacy" for value in final_lines)
    cache = _canonical({"profile": PROFILE, "source_sha256": base["source_sha256"], "config_sha256": config_sha256, "decisions": decisions}) + b"\n"
    return {
        **{key: value for key, value in base.items() if key != "_ephemeral"},
        "schema_version": 1,
        "kind": "sherman_qc_f1_v40_obtuse_parent_bridge_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "v39_retained": True,
        "decisions": decisions,
        "final_lines": final_lines,
        "pre_nms_rescue_count": sum(value["action"] == ACTION_RESCUE for value in decisions),
        "post_nms_rescue_only_count": rescue_only,
        "final_line_count": len(final_lines),
        "rescue_count": len(final_lines),
        "v40_restored_count": len(restored_ids),
        "cache_bytes": len(cache),
        "cache_sha256": hashlib.sha256(cache).hexdigest(),
        "_ephemeral": {"candidate_visuals": visuals, "fixed_cache": cache, "base_result": base},
    }
