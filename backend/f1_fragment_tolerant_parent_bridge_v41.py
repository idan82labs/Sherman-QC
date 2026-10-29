"""Research-only V41 fragment-tolerant parent-bridge certificate.

V41 handles true folds whose two dominant source skins remain coherent while
small disconnected scan fragments inflate the raw mesh-component count.  It
restores only candidates for which every frozen predicate except transverse
tail/transition evidence passes, exactly two dominant components independently
bridge both proposed parents, and all remaining aligned fragment area is at
most one percent of the source-aligned area.  Restoration occurs before the
ordinary frozen NMS pass.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from backend.f1_obtuse_parent_bridge_v40 import select_unified_v40
from backend.f1_ordered_projective_gauss_transport import (
    ACTION_LEGACY,
    ACTION_RESCUE,
    ordinary_nms,
    parse_candidates,
)
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "V41_FRAGMENT_TOLERANT_PARENT_BRIDGE_V1"
PRIMITIVE = "DOMINANT_TWO_SKIN_FRAGMENT_TOLERANT_PARENT_BRIDGE_V1"
MIN_TURN_DEGREES = 45.0
MAX_TURN_DEGREES = 170.0
REQUIRED_BRIDGING_SKINS = 2
MAX_RESIDUAL_ALIGNED_AREA_RATIO = 0.01


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def fragment_tolerant_certificate(decision: dict[str, Any]) -> dict[str, Any]:
    predicates = list(decision.get("predicate_passes", []))[:7]
    allowed_failure_shape = bool(
        len(predicates) == 7
        and predicates[0]
        and predicates[1]
        and predicates[2]
        and predicates[5]
        and predicates[6]
        and not predicates[4]
    )
    # Explicitly exclude any failure outside the tail-separation/transition pair.
    failed_indices = [index for index, passed in enumerate(predicates) if not passed]
    allowed_failure_shape = allowed_failure_shape and set(failed_indices).issubset({3, 4})

    parent = decision.get("v40_obtuse_parent_bridge_certificate", {})
    details = list(parent.get("component_details", []))
    aligned_total = sum(float(row.get("area_a", 0.0)) + float(row.get("area_b", 0.0)) for row in details)
    residual_aligned = sum(
        float(row.get("area_a", 0.0)) + float(row.get("area_b", 0.0))
        for row in details
        if not row.get("bridges_parents", False)
    )
    residual_ratio = residual_aligned / aligned_total if aligned_total > 0.0 else 1.0
    turn = float(parent.get("signed_turn_degrees", 0.0))
    v38 = decision.get("v38_sharp_transition_route_certificate", {})
    passed = bool(
        allowed_failure_shape
        and MIN_TURN_DEGREES <= turn <= MAX_TURN_DEGREES
        and int(parent.get("bridging_skin_count", 0)) == REQUIRED_BRIDGING_SKINS
        and int(parent.get("source_component_count", 0)) > REQUIRED_BRIDGING_SKINS
        and residual_ratio <= MAX_RESIDUAL_ALIGNED_AREA_RATIO
        and bool(v38.get("direct_phase_passed", False))
    )
    return {
        "passed": passed,
        "allowed_failed_predicate_indices": [3, 4],
        "observed_failed_predicate_indices": failed_indices,
        "signed_turn_degrees": turn,
        "source_component_count": int(parent.get("source_component_count", 0)),
        "bridging_skin_count": int(parent.get("bridging_skin_count", 0)),
        "required_bridging_skin_count": REQUIRED_BRIDGING_SKINS,
        "aligned_area_total": aligned_total,
        "residual_aligned_area": residual_aligned,
        "residual_aligned_area_ratio": residual_ratio,
        "maximum_residual_aligned_area_ratio": MAX_RESIDUAL_ALIGNED_AREA_RATIO,
        "direct_phase_passed": bool(v38.get("direct_phase_passed", False)),
    }


def select_unified_v41(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    base = select_unified_v40(raw_source, raw_candidates, config_sha256=config_sha256, worker_count=worker_count)
    candidates = parse_candidates(raw_candidates)
    decisions = normalize_numpy_scalars([dict(value) for value in base["decisions"]])
    visuals = [dict(value) for value in base["_ephemeral"]["candidate_visuals"]]
    restored_ids: set[int] = set()
    for decision, visual in zip(decisions, visuals):
        certificate = fragment_tolerant_certificate(decision)
        decision["v41_fragment_tolerant_parent_bridge_certificate"] = certificate
        visual["v41_fragment_tolerant_parent_bridge_certificate"] = certificate
        if decision["action"] == "NO_RESCUE" and certificate["passed"]:
            predicates = list(decision.get("predicate_passes", []))
            predicates[3] = True
            predicates[4] = True
            predicates.append(True)
            decision["predicate_passes"] = predicates
            decision["action"] = ACTION_RESCUE
            decision["reason"] = "OPGT_PASS"
            decision["v41_restored_fragmented_fold"] = True
            visual["first_failing_predicate"] = "OPGT_PASS"
            visual["v41_restored_fragmented_fold"] = True
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
            "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v41_fragment_tolerant_parent_bridge_rescue",
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
        "kind": "sherman_qc_f1_v41_fragment_tolerant_parent_bridge_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "v40_retained": True,
        "decisions": decisions,
        "final_lines": final_lines,
        "pre_nms_rescue_count": sum(value["action"] == ACTION_RESCUE for value in decisions),
        "post_nms_rescue_only_count": rescue_only,
        "final_line_count": len(final_lines),
        "rescue_count": len(final_lines),
        "v41_restored_count": len(restored_ids),
        "cache_bytes": len(cache),
        "cache_sha256": hashlib.sha256(cache).hexdigest(),
        "_ephemeral": {"candidate_visuals": visuals, "fixed_cache": cache, "base_result": base},
    }
