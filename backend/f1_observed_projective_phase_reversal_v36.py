"""Research-only V36 observed-projective-phase reversal veto.

V36 is the serialization-complete successor to terminal V35. The geometric
rule is unchanged: retain the frozen V34 proposal path and veto only accepted
rescues whose directly observed transition phase changes direction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from backend.f1_observed_projective_phase_reversal_v35 import (
    ACTION_NONE,
    PRIMITIVE,
    REASON_REVERSAL,
    direct_transition_phase_certificate,
)
from backend.f1_ordered_projective_gauss_transport import (
    ACTION_RESCUE,
    select_unified_v34,
)
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "V36_OBSERVED_PROJECTIVE_PHASE_REVERSAL_VETO_V1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def select_unified_v36(
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
    decisions = normalize_numpy_scalars([dict(value) for value in base["decisions"]])
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
            decision["v36_direct_phase_certificate"] = {
                key: value for key, value in certificate.items() if key != "steps"
            }
            if not certificate["passed"]:
                decision["action"] = ACTION_NONE
                decision["reason"] = REASON_REVERSAL
                vetoed_ids.add(int(decision["candidate_id"]))
            visual["v36_direct_phase_certificate"] = certificate
        certificates.append(certificate)

    final_lines = [
        value for value in base["final_lines"] if int(value["candidate_id"]) not in vetoed_ids
    ]
    cache_payload = normalize_numpy_scalars(
        {
            "profile": PROFILE,
            "source_sha256": base["source_sha256"],
            "config_sha256": config_sha256,
            "decisions": decisions,
        }
    )
    cache = _canonical(cache_payload) + b"\n"
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v36_observed_phase_veto_research_result",
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
