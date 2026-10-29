"""Research-only V35 direct observed-projective-phase reversal veto.

V35 retains the frozen V34 OPGT proposal path, then applies one independent
source-only veto to accepted rescue candidates. The veto reads the observed
per-strip projective phase before the V34 monotone fit and rejects a candidate
when meaningful adjacent phase increments change sign inside its transition.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from backend.f1_ordered_projective_gauss_transport import (
    ACTION_LEGACY,
    ACTION_RESCUE,
    PHASE_STEP,
    select_unified_v34,
)


PROFILE = "V35_OBSERVED_PROJECTIVE_PHASE_REVERSAL_VETO_V1"
PRIMITIVE = "DIRECT_TRANSITION_PHASE_DIRECTION_CERTIFICATE_V1"
REASON_REVERSAL = "OBSERVED_PROJECTIVE_PHASE_REVERSAL"
ACTION_NONE = "NO_RESCUE"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def projective_delta(left: float, right: float) -> float:
    """Shortest signed projective angle from left to right, in radians."""

    difference = right - left
    return 0.5 * math.atan2(math.sin(2.0 * difference), math.cos(2.0 * difference))


def direct_transition_phase_certificate(visual: dict[str, Any]) -> dict[str, Any]:
    phases = visual.get("free_phases", [])
    retained = [int(value) for value in visual.get("phase_path_strips", [])]
    transition = {int(value) for value in visual.get("transition_strips", [])}
    meaningful_floor = 0.5 * PHASE_STEP
    steps: list[dict[str, Any]] = []
    signs: list[int] = []
    for left, right in zip(retained, retained[1:]):
        if right != left + 1 or (left not in transition and right not in transition):
            continue
        left_phase = phases[left]
        right_phase = phases[right]
        if left_phase is None or right_phase is None:
            continue
        delta = projective_delta(float(left_phase), float(right_phase))
        sign = 1 if delta >= meaningful_floor else -1 if delta <= -meaningful_floor else 0
        steps.append(
            {
                "left_strip": left,
                "right_strip": right,
                "delta_radians": delta,
                "meaningful_sign": sign,
            }
        )
        if sign:
            signs.append(sign)
    reversals = sum(current != previous for previous, current in zip(signs, signs[1:]))
    return {
        "passed": not (1 in signs and -1 in signs),
        "meaningful_floor_radians": meaningful_floor,
        "meaningful_step_count": len(signs),
        "positive_step_count": signs.count(1),
        "negative_step_count": signs.count(-1),
        "direction_reversal_count": reversals,
        "steps": steps,
    }


def select_unified_v35(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    base = select_unified_v34(
        raw_source,
        raw_candidates,
        config_sha256=config_sha256,
        worker_count=worker_count,
    )
    decisions = [dict(value) for value in base["decisions"]]
    visuals = [dict(value) for value in base["_ephemeral"]["candidate_visuals"]]
    certificates: list[dict[str, Any] | None] = []
    vetoed_ids: set[int] = set()
    for decision, visual in zip(decisions, visuals):
        certificate = None
        if decision["action"] == ACTION_RESCUE and visual.get("kind") == "opgt":
            certificate = direct_transition_phase_certificate(visual)
            decision["predicate_passes"] = list(decision["predicate_passes"]) + [
                bool(certificate["passed"])
            ]
            decision["v35_direct_phase_certificate"] = {
                key: value for key, value in certificate.items() if key != "steps"
            }
            if not certificate["passed"]:
                decision["action"] = ACTION_NONE
                decision["reason"] = REASON_REVERSAL
                vetoed_ids.add(int(decision["candidate_id"]))
            visual["v35_direct_phase_certificate"] = certificate
        certificates.append(certificate)

    final_lines = [
        value for value in base["final_lines"] if int(value["candidate_id"]) not in vetoed_ids
    ]
    cache = _canonical(
        {
            "profile": PROFILE,
            "source_sha256": base["source_sha256"],
            "config_sha256": config_sha256,
            "decisions": decisions,
        }
    ) + b"\n"
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v35_observed_phase_veto_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "research_only": True,
        "source_only": True,
        "rescue_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
        "filename_identity_used": False,
        "labels_used": False,
        "cad_used": False,
        "v34_proposal_path_retained": True,
        "post_nms_veto_only": True,
        "candidate_added_after_nms": False,
        "decisions": decisions,
        "final_lines": final_lines,
        "pre_veto_rescue_count": int(base["rescue_count"]),
        "veto_count": len(vetoed_ids),
        "pre_nms_rescue_count": sum(
            value["action"] == ACTION_RESCUE for value in decisions
        ),
        "rescue_count": sum(
            value["source_lane"] == "v34_opgt_rescue" for value in final_lines
        ),
        "ordinary_nms_call_count": int(base["ordinary_nms_call_count"]),
        "cache_bytes": len(cache),
        "cache_sha256": hashlib.sha256(cache).hexdigest(),
        "source_sha256": base["source_sha256"],
        "_ephemeral": {
            "candidate_visuals": visuals,
            "fixed_cache": cache,
            "base_result": base,
            "certificates": certificates,
        },
    }
