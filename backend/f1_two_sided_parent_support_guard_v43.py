"""Research-only V43 guard against open-rim and thickness-edge aliases."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from backend.f1_ordered_projective_gauss_transport import (
    ACTION_LEGACY,
    ACTION_RESCUE,
    ordinary_nms,
    parse_candidates,
)
from backend.f1_single_shell_parent_bridge_v42 import select_unified_v42
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "V43_TWO_SIDED_PARENT_SUPPORT_GUARD_V1"
PRIMITIVE = "MINIMUM_QUARTER_SUPPORT_ON_BOTH_TRANSVERSE_SIDES_V1"
MIN_TWO_SIDED_SUPPORT_FRACTION = 0.25


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def support_guard(raw_candidate: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    value = raw_candidate.get("two_sided_min_support_fraction")
    fraction = None if value is None else float(value)
    applies = bool(decision.get("v42_restored_single_or_double_shell_fold"))
    passed = bool(not applies or (fraction is not None and fraction >= MIN_TWO_SIDED_SUPPORT_FRACTION))
    return {
        "passed": passed,
        "applies_to_v42_restoration": applies,
        "two_sided_min_support_fraction": fraction,
        "minimum_two_sided_support_fraction": MIN_TWO_SIDED_SUPPORT_FRACTION,
    }


def select_unified_v43(
    measurement_source: dict[str, Any],
    topology_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    base = select_unified_v42(
        measurement_source, topology_source, raw_candidates,
        config_sha256=config_sha256, worker_count=worker_count,
    )
    candidates = parse_candidates(raw_candidates)
    decisions = normalize_numpy_scalars([dict(value) for value in base["decisions"]])
    visuals = [dict(value) for value in base["_ephemeral"]["candidate_visuals"]]
    vetoed: set[int] = set()
    for raw_candidate, decision, visual in zip(raw_candidates, decisions, visuals):
        certificate = support_guard(raw_candidate, decision)
        decision["v43_two_sided_parent_support_guard"] = certificate
        visual["v43_two_sided_parent_support_guard"] = certificate
        if decision["action"] == ACTION_RESCUE and not certificate["passed"]:
            observed = list(
                (decision.get("v42_single_shell_parent_bridge_certificate") or {}).get(
                    "observed_failed_predicate_indices", []
                )
            )
            predicates = list(decision.get("predicate_passes", []))[:7]
            for index in observed:
                if 0 <= int(index) < len(predicates):
                    predicates[int(index)] = False
            decision["predicate_passes"] = predicates
            decision["action"] = "NO_RESCUE"
            decision["reason"] = (
                "TAIL_NORMALS_NOT_SEPARATED" if 3 in observed
                else "INSUFFICIENT_TRANSITION_MASS"
            )
            decision["v43_vetoed_open_rim_or_thickness_edge"] = True
            visual["first_failing_predicate"] = "TWO_SIDED_PARENT_SUPPORT"
            visual["v43_vetoed_open_rim_or_thickness_edge"] = True
            vetoed.add(int(decision["candidate_id"]))

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
            "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v43_two_sided_parent_support_rescue",
        }
        for candidate, decision in accepted
    ]
    decisions = normalize_numpy_scalars(decisions)
    final_lines = normalize_numpy_scalars(final_lines)
    rescue_only = sum(value["source_lane"] != "legacy" for value in final_lines)
    cache = _canonical({"profile": PROFILE, "measurement_source_sha256": base["source_sha256"], "config_sha256": config_sha256, "decisions": decisions}) + b"\n"
    return {
        **{key: value for key, value in base.items() if key != "_ephemeral"},
        "schema_version": 1,
        "kind": "sherman_qc_f1_v43_two_sided_parent_support_guard_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "v42_retained": True,
        "decisions": decisions,
        "final_lines": final_lines,
        "pre_nms_rescue_count": sum(value["action"] == ACTION_RESCUE for value in decisions),
        "post_nms_rescue_only_count": rescue_only,
        "final_line_count": len(final_lines),
        "rescue_count": len(final_lines),
        "v43_vetoed_count": len(vetoed),
        "cache_bytes": len(cache),
        "cache_sha256": hashlib.sha256(cache).hexdigest(),
        "_ephemeral": {"candidate_visuals": visuals, "fixed_cache": cache, "base_result": base},
    }
