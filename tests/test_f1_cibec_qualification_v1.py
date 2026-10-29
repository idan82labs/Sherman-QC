from __future__ import annotations

import copy

from research.f1_cibec_qualification_v1 import score


def annotation(case_index: int, *, y: float = 0.0, code: str = "STANDARD", status: str = "RESOLVED", extra: bool = False) -> dict:
    events = [{
        "class_code": code,
        "line_start_mm": [0.0, y, 0.0],
        "line_end_mm": [20.0, y, 0.0],
        "visibility": "FULL",
        "render_admissible": True,
        "verified_view_ids": ["V1", "V2"],
        "adjudication_rule_id": "RULE_01",
    }]
    if extra:
        events.append({**events[0], "line_start_mm": [0.0, 10.0, 0.0], "line_end_mm": [20.0, 10.0, 0.0]})
    return {
        "schema": "SCAN_BEND_EVENTS_V1",
        "ontology_sha256": "a" * 64,
        "renderer_sha256": "b" * 64,
        "case_id": f"{case_index + 1:032x}",
        "scan_sha256": f"{case_index + 100:064x}",
        "units": "mm",
        "coordinate_frame": "RAW_SCAN",
        "events": events,
        "case_status": status,
    }


def perfect_payload() -> dict:
    cases = []
    for index in range(13):
        first = annotation(index)
        cases.append({
            "annotation_a": first,
            "annotation_b": copy.deepcopy(first),
            "adjudicated": copy.deepcopy(first),
            "native_visual_passed": True,
            "native_visual_audit_sha256": f"{index + 500:064x}",
            "charter_rule_justification_sha256": f"{index + 800:064x}" if index < 3 else None,
            "annotation_a_replay_identical": True,
            "annotation_b_replay_identical": True,
            "adjudicated_replay_identical": True,
            "export_latency_seconds": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            "peak_rss_bytes": 100_000_000,
            "jobs_attempted": 6,
            "jobs_succeeded": 6,
        })
    return {
        "schema": "CIBEC_V1_PHYSICAL_ANNOTATION_QUALIFICATION_INPUT",
        "hash_manifest_verified": True,
        "leakage_audit_prohibited_field_or_asset_count": 0,
        "cases": cases,
    }


def test_perfect_13_case_qualification_passes() -> None:
    result = score(perfect_payload())
    assert result["status"] == "PASS"
    assert result["metrics"]["exact_event_count_cases"] == 13
    assert result["metrics"]["inter_annotator_f1"] == 1.0
    assert result["failed_gates"] == []


def test_exact_count_allows_one_case_but_not_two() -> None:
    one = perfect_payload()
    one["cases"][0]["annotation_b"] = annotation(0, extra=True)
    result = score(one)
    assert result["metrics"]["exact_event_count_cases"] == 12
    assert result["gates"]["exact_event_count"] is True
    two = copy.deepcopy(one)
    two["cases"][1]["annotation_b"] = annotation(1, extra=True)
    result = score(two)
    assert result["gates"]["exact_event_count"] is False
    assert result["status"] == "FAIL"


def test_class_visual_latency_and_job_failures_are_independent() -> None:
    value = perfect_payload()
    value["cases"][0]["annotation_b"] = annotation(0, code="HEM")
    value["cases"][1]["annotation_b"] = annotation(1, code="HEM")
    value["cases"][1]["native_visual_passed"] = False
    value["cases"][2]["export_latency_seconds"] = [0.1, 5.01, 0.1, 0.1, 0.1, 0.1]
    value["cases"][4]["export_latency_seconds"] = [0.1, 5.02, 0.1, 0.1, 0.1, 0.1]
    value["cases"][5]["export_latency_seconds"] = [0.1, 5.03, 0.1, 0.1, 0.1, 0.1]
    value["cases"][6]["export_latency_seconds"] = [0.1, 5.04, 0.1, 0.1, 0.1, 0.1]
    value["cases"][3]["jobs_succeeded"] = 5
    result = score(value)
    assert result["gates"]["class_code"] is False
    assert result["gates"]["native_visual"] is False
    assert result["gates"]["export_latency"] is False
    assert result["gates"]["job_success"] is False


def test_replay_memory_manifest_leakage_and_justification_gates_are_independent() -> None:
    value = perfect_payload()
    value["cases"][0]["annotation_a_replay_identical"] = False
    value["cases"][1]["peak_rss_bytes"] = 2 * 1024**3
    value["cases"][2]["charter_rule_justification_sha256"] = None
    value["hash_manifest_verified"] = False
    value["leakage_audit_prohibited_field_or_asset_count"] = 1
    result = score(value)
    assert result["gates"]["deterministic_replay"] is False
    assert result["gates"]["memory"] is False
    assert result["gates"]["three_required_written_justifications"] is False
    assert result["gates"]["hash_manifest"] is False
    assert result["gates"]["leakage_audit"] is False


def test_duplicate_final_event_identity_is_rejected() -> None:
    value = perfect_payload()
    duplicate = copy.deepcopy(value["cases"][0]["adjudicated"]["events"][0])
    duplicate["line_start_mm"] = [0.0, 1.0, 0.0]
    duplicate["line_end_mm"] = [20.0, 1.0, 0.0]
    value["cases"][0]["adjudicated"]["events"].append(duplicate)
    result = score(value)
    assert result["metrics"]["final_duplicate_event_identities"] == 1
    assert result["gates"]["no_duplicate_event_identities"] is False
