from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.cibec_v1.contract import (
    CibecValidationError,
    canonical_json_bytes,
    canonicalize_scan_truth,
    validate_cad_correspondence,
    validate_charter,
)


HEX = "a" * 64


def _truth(events):
    return {
        "schema": "SCAN_BEND_EVENTS_V1", "ontology_sha256": HEX, "renderer_sha256": HEX,
        "case_id": "b" * 32, "scan_sha256": HEX, "units": "mm", "coordinate_frame": "RAW_SCAN",
        "events": events, "case_status": "RESOLVED",
    }


def _event(start, end, code="STANDARD"):
    return {"class_code": code, "line_start_mm": start, "line_end_mm": end,
            "visibility": "FULL", "render_admissible": True,
            "verified_view_ids": ["V2", "V1", "V1"], "adjudication_rule_id": "RULE_01"}


def test_unsigned_charter_is_an_explicit_hard_gate() -> None:
    path = Path("research/cibec_v1/CIBEC_V1_CHARTER.yaml")
    with pytest.raises(CibecValidationError, match="not frozen"):
        validate_charter(json.loads(path.read_text()))


def test_endpoint_and_record_permutation_are_byte_identical() -> None:
    first = canonicalize_scan_truth(_truth([
        _event([20.0, 5.0, 0.0], [10.0, 5.0, 0.0], "B"),
        _event([0.0, 0.0, 0.0], [12.0, 0.0, 0.0], "A"),
    ]))
    second = canonicalize_scan_truth(_truth([
        _event([12.0, 0.0, 0.0], [0.0, 0.0, 0.0], "A"),
        _event([10.0, 5.0, 0.0], [20.0, 5.0, 0.0], "B"),
    ]))
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert [item["event_id"] for item in first["events"]] == ["E0001", "E0002"]
    assert first["events"][0]["verified_view_ids"] == ["V1", "V2"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coordinates_are_rejected(bad: float) -> None:
    with pytest.raises(CibecValidationError, match="finite"):
        canonicalize_scan_truth(_truth([_event([0.0, 0.0, 0.0], [bad, 0.0, 0.0])]))


def test_short_line_non_mm_and_source_points_are_rejected() -> None:
    with pytest.raises(CibecValidationError, match="at least 8.0"):
        canonicalize_scan_truth(_truth([_event([0, 0, 0], [7.999999, 0, 0])]))
    payload = _truth([_event([0, 0, 0], [8, 0, 0])]); payload["units"] = "inch"
    with pytest.raises(CibecValidationError, match="millimetres"):
        canonicalize_scan_truth(payload)
    payload = _truth([_event([0, 0, 0], [8, 0, 0])]); payload["source_points"] = []
    with pytest.raises(CibecValidationError, match="prohibited"):
        canonicalize_scan_truth(payload)


def test_decimal_minimum_span_boundary_survives_canonicalization() -> None:
    payload = _truth([_event([-8.513, -34.94, -33.027], [-8.513, -26.94, -33.027])])
    result = canonicalize_scan_truth(payload)
    assert result["events"][0]["length_mm"] == 8.0
    short = _truth([_event([-8.513, -34.94, -33.027], [-8.513, -26.940001, -33.027])])
    with pytest.raises(CibecValidationError, match="at least 8.0"):
        canonicalize_scan_truth(short)


def test_cad_face_permutation_is_canonical_and_duplicate_ownership_fails() -> None:
    base = {"schema": "CAD_EVENT_CORRESPONDENCE_V1", "primary_truth_sha256": HEX,
            "step_sha256": HEX, "overlay_transform_sha256": HEX, "overlay_status": "TRUSTED",
            "events": [{"event_id": "E0001", "step_face_indices": [18, 12, 14],
                        "v9_primitive_ids": ["B0005", "B0004"], "relation": "MANY_TO_ONE",
                        "mapping_status": "RESOLVED"}], "cad_only_unobserved": []}
    result = validate_cad_correspondence(base)
    assert result["events"][0]["step_face_indices"] == [12, 14, 18]
    assert result["events"][0]["v9_primitive_ids"] == ["B0004", "B0005"]
    base["cad_only_unobserved"] = [{"step_face_indices": [18], "reason_code": "UNOBSERVED"}]
    with pytest.raises(CibecValidationError, match="duplicate STEP-face"):
        validate_cad_correspondence(base)


def test_abstain_preserves_empty_correspondence() -> None:
    payload = {"schema": "CAD_EVENT_CORRESPONDENCE_V1", "primary_truth_sha256": HEX,
               "step_sha256": HEX, "overlay_transform_sha256": HEX, "overlay_status": "ABSTAIN",
               "reason": "OVERLAY_UNAVAILABLE_OR_UNTRUSTED", "events": [], "cad_only_unobserved": []}
    assert validate_cad_correspondence(payload)["overlay_status"] == "ABSTAIN"
