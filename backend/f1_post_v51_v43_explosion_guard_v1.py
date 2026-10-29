"""Source-only guard for multi-addition V43 explosions after frozen F1 V51.

The frozen V43 stage can occasionally append several geometrically related
lines at once.  This development successor falls back to the already-produced
V16 prediction set when V43 reports more than one post-merge addition.  It does
not inspect labels, expected counts, part identity, CAD, drawings, or filenames.

The module deliberately lives outside the frozen V51 chain.  The cutoff was
selected after inspecting development errors, so results are development
evidence only until a genuinely prospective cohort is scored.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROFILE = "F1_POST_V51_V43_EXPLOSION_GUARD_V1"
MAX_SAFE_V43_POST_MERGE_ADDITIONS = 1
RESULT_KIND = "sherman_qc_f1_post_v51_v43_explosion_guard_result_v1"
FAIL_CLOSED_KIND = "sherman_qc_v51_fail_closed_case_abstention"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _prediction_rows(payload: Mapping[str, Any], *, name: str) -> list[dict[str, Any]]:
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError(f"{name} predictions must be a list")
    result: list[dict[str, Any]] = []
    for ordinal, value in enumerate(rows):
        if not isinstance(value, dict):
            raise ValueError(f"{name} prediction {ordinal} must be an object")
        start = value.get("line_start")
        end = value.get("line_end")
        if not (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) == len(end) == 3
        ):
            raise ValueError(f"{name} prediction {ordinal} has invalid endpoints")
        coordinates = [float(item) for item in (*start, *end)]
        if not all(math.isfinite(item) for item in coordinates):
            raise ValueError(f"{name} prediction {ordinal} is non-finite")
        if math.dist(coordinates[:3], coordinates[3:]) <= 1e-9:
            raise ValueError(f"{name} prediction {ordinal} is degenerate")
        result.append(deepcopy(value))
    return result


def _validate_source_only(payload: Mapping[str, Any], *, name: str) -> None:
    if payload.get("source_only") is not True or payload.get("research_only") is not True:
        raise ValueError(f"{name} is not source-only research output")
    for field in (
        "labels_used",
        "expected_count_used",
        "part_identity_used",
        "cad_used",
        "drawing_used",
    ):
        if payload.get(field) not in (None, False):
            raise ValueError(f"{name} reports forbidden inference input: {field}")


def _sha_payload(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _science_projection(
    payload: Mapping[str, Any] | None,
    *,
    include_v43_accounting: bool = False,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    projection: dict[str, Any] = {
        "kind": payload.get("kind"),
        "profile": payload.get("profile"),
        "predictions": payload.get("predictions"),
        "source_only": payload.get("source_only"),
        "research_only": payload.get("research_only"),
    }
    if include_v43_accounting:
        projection.update(
            {
                "retained_v16_prediction_count": payload.get(
                    "retained_v16_prediction_count"
                ),
                "v43_post_merge_addition_count": payload.get(
                    "v43_post_merge_addition_count"
                ),
                "candidate_count": payload.get("candidate_count"),
            }
        )
    return projection


def compose_v43_explosion_guard(
    v16_result: Mapping[str, Any] | None,
    v43_result: Mapping[str, Any] | None,
    v51_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic guarded result without consulting truth.

    Failed V51 cases are preserved as empty abstentions.  Successful cases
    require all three frozen stage payloads so the guard cannot silently infer
    an addition count from missing evidence.
    """

    _validate_source_only(v51_result, name="V51")
    v51_predictions = _prediction_rows(v51_result, name="V51")
    if v51_result.get("kind") == FAIL_CLOSED_KIND:
        if v51_predictions:
            raise ValueError("fail-closed V51 abstention must not contain predictions")
        if v16_result is not None or v43_result is not None:
            raise ValueError("fail-closed V51 abstention must not carry partial stage inputs")
        selected: list[dict[str, Any]] = []
        guard_triggered = False
        selection = "PRESERVE_FAIL_CLOSED_V51_ABSTENTION"
        addition_count: int | None = None
        v16_count: int | None = None
    else:
        if v16_result is None or v43_result is None:
            raise ValueError("successful V51 input requires V16 and V43 stage payloads")
        _validate_source_only(v16_result, name="V16")
        _validate_source_only(v43_result, name="V43")
        v16_predictions = _prediction_rows(v16_result, name="V16")
        _prediction_rows(v43_result, name="V43")
        raw_count = v43_result.get("v43_post_merge_addition_count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError("V43 post-merge addition count must be a nonnegative integer")
        addition_count = int(raw_count)
        retained_v16 = v43_result.get("retained_v16_prediction_count")
        if retained_v16 != len(v16_predictions):
            raise ValueError("V43 retained-V16 count does not match the supplied V16 result")
        v43_candidate_count = v43_result.get("candidate_count")
        if v43_candidate_count != len(v16_predictions) + addition_count:
            raise ValueError("V43 candidate count is inconsistent with its merge accounting")
        guard_triggered = addition_count > MAX_SAFE_V43_POST_MERGE_ADDITIONS
        selected = deepcopy(v16_predictions if guard_triggered else v51_predictions)
        selection = "FALL_BACK_TO_V16" if guard_triggered else "RETAIN_V51"
        v16_count = len(v16_predictions)

    audit = {
        "guard_triggered": guard_triggered,
        "selection": selection,
        "maximum_safe_v43_post_merge_additions": MAX_SAFE_V43_POST_MERGE_ADDITIONS,
        "v43_post_merge_addition_count": addition_count,
        "v16_prediction_count": v16_count,
        "v51_prediction_count": len(v51_predictions),
        "output_prediction_count": len(selected),
    }
    hash_payload = {
        "profile": PROFILE,
        "audit": audit,
        "predictions": selected,
        # Exclude elapsed/runtime metadata so independently executed primary and
        # replay stages yield identical successor science bytes.
        "v16_science_sha256": _sha_payload(_science_projection(v16_result)),
        "v43_science_sha256": _sha_payload(
            _science_projection(v43_result, include_v43_accounting=True)
        ),
        "v51_science_sha256": _sha_payload(_science_projection(v51_result)),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "profile": PROFILE,
        "predictions": selected,
        "candidate_count": len(selected),
        "guard_audit": audit,
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "part_identity_used": False,
        "cad_used": False,
        "drawing_used": False,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "development_tuned_cutoff": True,
        "result_sha256": _sha_payload(hash_payload),
    }
