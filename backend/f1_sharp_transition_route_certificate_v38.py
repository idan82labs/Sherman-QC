"""Research-only V38 sharp-transition route certificate.

V38 retains V37 and admits one narrow source-only correction for a sharp,
well-supported fold that misses V34's mass-weighted transition threshold.  A
candidate may be restored only when every other frozen predicate passes, the
observed transition spans at least the grid-equivalent V34 width, the direct
projective phase is monotone, and the source mesh has at most two connected
surface routes.  Ordinary NMS is then rerun over the resulting pre-NMS stream.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from backend.f1_axial_causal_precedence_v37 import select_unified_v37
from backend.f1_observed_projective_phase_reversal_v35 import (
    direct_transition_phase_certificate,
)
from backend.f1_ordered_projective_gauss_transport import (
    ACTION_LEGACY,
    ACTION_RESCUE,
    TRANSITION_MASS_FRACTION,
    TRANSVERSE_STRIPS,
    ordinary_nms,
    parse_candidates,
)
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "V38_SHARP_TRANSITION_ROUTE_CERTIFICATE_V1"
PRIMITIVE = "SOURCE_MESH_ROUTE_MULTIPLICITY_SHARP_TRANSITION_V1"
SHARP_TRANSITION_MIN_STRIPS = int(math.ceil(TRANSVERSE_STRIPS * TRANSITION_MASS_FRACTION))
MAX_SOURCE_ROUTE_COMPONENTS = 2


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def mesh_component_count(raw_source: dict[str, Any]) -> int:
    positions = raw_source.get("positions", [])
    triangles = raw_source.get("triangles", [])
    parent = list(range(len(positions)))
    size = [1] * len(positions)
    used: set[int] = set()

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

    for triangle in triangles:
        if len(triangle) != 3:
            continue
        a, b, c = (int(value) for value in triangle)
        if min(a, b, c) < 0 or max(a, b, c) >= len(positions):
            continue
        used.update((a, b, c))
        union(a, b)
        union(a, c)
    return len({find(value) for value in used})


def select_unified_v38(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    base = select_unified_v37(
        raw_source,
        raw_candidates,
        config_sha256=config_sha256,
        worker_count=worker_count,
    )
    candidates = parse_candidates(raw_candidates)
    decisions = normalize_numpy_scalars([dict(value) for value in base["decisions"]])
    visuals = [dict(value) for value in base["_ephemeral"]["candidate_visuals"]]
    component_count = mesh_component_count(raw_source)
    restored_ids: set[int] = set()

    for decision, visual in zip(decisions, visuals):
        predicates = list(decision.get("predicate_passes", []))
        transition_count = len(visual.get("transition_strips", []))
        phase_certificate = direct_transition_phase_certificate(visual)
        only_transition_failed = (
            len(predicates) == 7
            and predicates[1]
            and predicates[2]
            and predicates[3]
            and not predicates[4]
            and predicates[5]
            and predicates[6]
        )
        certificate = {
            "only_transition_failed": bool(only_transition_failed),
            "transition_strip_count": int(transition_count),
            "minimum_transition_strip_count": SHARP_TRANSITION_MIN_STRIPS,
            "source_mesh_component_count": int(component_count),
            "maximum_source_route_components": MAX_SOURCE_ROUTE_COMPONENTS,
            "direct_phase_passed": bool(phase_certificate["passed"]),
            "passed": bool(
                only_transition_failed
                and transition_count >= SHARP_TRANSITION_MIN_STRIPS
                and 0 < component_count <= MAX_SOURCE_ROUTE_COMPONENTS
                and phase_certificate["passed"]
            ),
        }
        decision["v38_sharp_transition_route_certificate"] = certificate
        visual["v38_sharp_transition_route_certificate"] = certificate
        visual["v38_direct_phase_certificate"] = phase_certificate
        if certificate["passed"]:
            predicates[4] = True
            predicates.append(True)
            decision["predicate_passes"] = predicates
            decision["action"] = ACTION_RESCUE
            decision["reason"] = "OPGT_PASS"
            decision["v38_restored_sharp_transition"] = True
            visual["first_failing_predicate"] = "OPGT_PASS"
            visual["v38_restored_sharp_transition"] = True
            restored_ids.add(int(decision["candidate_id"]))

    pairs = list(zip(candidates, decisions))
    legacy = [pair for pair in pairs if pair[1]["action"] == ACTION_LEGACY]
    additions = [pair for pair in pairs if pair[1]["action"] == ACTION_RESCUE]
    ordered = sorted(legacy, key=lambda row: (row[0].frozen_order, row[0].candidate_id))
    ordered += sorted(additions, key=lambda row: row[0].candidate_id)
    accepted, suppressed = ordinary_nms(ordered)
    for decision in decisions:
        decision["nms_reason"] = (
            "ORDINARY_NMS_SUPPRESSED"
            if int(decision["candidate_id"]) in suppressed
            else ""
        )
    final_lines = [
        {
            "candidate_id": candidate.candidate_id,
            "line_start": candidate.raw["render_line_start"],
            "line_end": candidate.raw["render_line_end"],
            "score": candidate.raw.get("score"),
            "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v38_sharp_transition_rescue",
        }
        for candidate, decision in accepted
    ]
    normalized_decisions = normalize_numpy_scalars(decisions)
    cache_payload = normalize_numpy_scalars(
        {
            "profile": PROFILE,
            "source_sha256": base["source_sha256"],
            "config_sha256": config_sha256,
            "decisions": normalized_decisions,
        }
    )
    cache = _canonical(cache_payload) + b"\n"
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v38_sharp_transition_route_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
        "filename_identity_used": False,
        "labels_used": False,
        "cad_used": False,
        "v37_retained": True,
        "decisions": normalized_decisions,
        "final_lines": normalize_numpy_scalars(final_lines),
        "pre_nms_rescue_count": sum(value["action"] == ACTION_RESCUE for value in normalized_decisions),
        "rescue_count": sum(value["source_lane"] != "legacy" for value in final_lines),
        "v38_restored_count": len(restored_ids),
        "source_mesh_component_count": int(component_count),
        "ordinary_nms_call_count": 1,
        "candidate_added_after_nms": False,
        "cache_bytes": len(cache),
        "cache_sha256": hashlib.sha256(cache).hexdigest(),
        "source_sha256": base["source_sha256"],
        "_ephemeral": {
            "candidate_visuals": visuals,
            "fixed_cache": cache,
            "base_result": base,
        },
    }
