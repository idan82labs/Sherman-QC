"""Research-only V37 causal precedence over retained V36 decisions.

V37 never changes an action, candidate, or final line. When a measured,
supported candidate fails the frozen axial-coherence predicate, it exposes that
failure as the causal reason even if an earlier pooled-tail predicate also
fails. This makes the certificate agree with the source-observed axial drift.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from backend.f1_observed_projective_phase_reversal_v36 import select_unified_v36
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "V37_AXIAL_CAUSAL_PRECEDENCE_V1"
PRIMITIVE = "COUNT_PRESERVING_AXIAL_FAILURE_PRECEDENCE_V1"
AXIAL_REASON = "AXIAL_INCOHERENCE"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def select_unified_v37(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    worker_count: int = 1,
) -> dict[str, Any]:
    base = select_unified_v36(
        raw_source,
        raw_candidates,
        config_sha256=config_sha256,
        worker_count=worker_count,
    )
    decisions = normalize_numpy_scalars([dict(value) for value in base["decisions"]])
    visuals = [dict(value) for value in base["_ephemeral"]["candidate_visuals"]]
    precedence_count = 0
    for decision, visual in zip(decisions, visuals):
        predicates = decision.get("predicate_passes", [])
        measured_supported = len(predicates) >= 6 and predicates[1] is True
        axial_failed = measured_supported and predicates[5] is False
        if decision["action"] == "NO_RESCUE" and axial_failed:
            decision["reason"] = AXIAL_REASON
            decision["v37_causal_precedence"] = AXIAL_REASON
            visual["first_failing_predicate"] = AXIAL_REASON
            visual["v37_causal_precedence"] = AXIAL_REASON
            precedence_count += 1

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
        "kind": "sherman_qc_f1_v37_axial_causal_precedence_research_result",
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
        "v36_actions_retained": True,
        "count_preserving": True,
        "decisions": decisions,
        "final_lines": list(base["final_lines"]),
        "pre_nms_rescue_count": int(base["pre_nms_rescue_count"]),
        "rescue_count": int(base["rescue_count"]),
        "veto_count": int(base["veto_count"]),
        "causal_precedence_count": precedence_count,
        "ordinary_nms_call_count": int(base["ordinary_nms_call_count"]),
        "candidate_added_after_nms": False,
        "cache_bytes": len(cache),
        "cache_sha256": hashlib.sha256(cache).hexdigest(),
        "source_sha256": base["source_sha256"],
        "_ephemeral": {
            "candidate_visuals": visuals,
            "fixed_cache": cache,
            "base_result": base,
            "certificates": base["_ephemeral"]["certificates"],
        },
    }
