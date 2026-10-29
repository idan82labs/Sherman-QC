from __future__ import annotations

import copy

import pytest

from research.f1_custody_v1 import CustodyError, allocate


def payload(case_count: int = 30, family_count: int = 15) -> dict:
    return {
        "schema": "F1_PHYSICAL_CUSTODY_INPUT_V1",
        "cohort": "UNTOUCHED_QUALIFICATION",
        "charter_sha256": "a" * 64,
        "blinding_salt_sha256": "f" * 64,
        "roles": {
            "custodian": "custodian",
            "adjudicator": "adjudicator",
            "annotators": ["annotator_a", "annotator_b", "annotator_c"],
            "model_implementers": ["implementer"],
        },
        "cases": [
            {"scan_sha256": f"{index + 1:064x}", "morphology_group_sha256": f"{(index % family_count) + 1000:064x}"}
            for index in range(case_count)
        ],
    }


def test_deterministic_balanced_blind_assignment() -> None:
    first = allocate(payload())
    second_payload = payload()
    second_payload["cases"].reverse()
    second = allocate(second_payload)
    assert first == second
    public = first["public_assignment"]
    assert public["status"] == "READY_FOR_BLIND_ANNOTATION"
    assert public["case_count"] == 30
    assert public["morphology_family_count"] == 15
    assert max(public["annotator_loads"].values()) - min(public["annotator_loads"].values()) <= 1
    assert all(len(row["annotators"]) == 2 for row in public["assignments"])
    assert all("scan_sha256" not in row for row in public["assignments"])
    assert all("morphology_group_sha256" not in row for row in public["assignments"])
    assert all(set(row) == {"case_id", "scan_sha256", "morphology_group_sha256"} for row in first["custodian_only_case_map"]["cases"])


@pytest.mark.parametrize("forbidden", ["scan_path", "filename", "expected_count", "prediction", "gold"])
def test_rejects_forbidden_case_fields(forbidden: str) -> None:
    value = payload(1, 1)
    value["cases"][0][forbidden] = "leak"
    with pytest.raises(CustodyError, match="forbidden"):
        allocate(value)


def test_rejects_role_overlap_consumed_cohort_and_duplicate_scan() -> None:
    value = payload(2, 1)
    value["roles"]["model_implementers"] = ["annotator_a"]
    with pytest.raises(CustodyError, match="may not hold"):
        allocate(value)
    value = payload(2, 1)
    value["cohort"] = "DEVELOPMENT"
    with pytest.raises(CustodyError, match="UNTOUCHED"):
        allocate(value)
    value = payload(2, 1)
    value["cases"][1]["scan_sha256"] = value["cases"][0]["scan_sha256"]
    with pytest.raises(CustodyError, match="duplicate"):
        allocate(value)


def test_assignment_hash_changes_if_public_contract_changes() -> None:
    first = allocate(payload())
    changed = payload()
    changed["charter_sha256"] = "b" * 64
    second = allocate(changed)
    assert first["custodian_only_case_map"]["public_assignment_sha256"] != second["custodian_only_case_map"]["public_assignment_sha256"]


def test_blinding_salt_changes_opaque_ids_and_is_never_public() -> None:
    first = allocate(payload())
    changed = payload()
    changed["blinding_salt_sha256"] = "e" * 64
    second = allocate(changed)
    first_ids = [row["case_id"] for row in first["public_assignment"]["assignments"]]
    second_ids = [row["case_id"] for row in second["public_assignment"]["assignments"]]
    assert first_ids != second_ids
    public_text = str(first["public_assignment"])
    assert payload()["blinding_salt_sha256"] not in public_text
    assert all(row["morphology_group_sha256"] not in public_text for row in payload()["cases"])


def test_insufficient_inventory_abstains_before_case_mapping() -> None:
    result = allocate(payload(29, 15))
    assert result["public_assignment"]["status"] == "CORPUS_CAPACITY_ABSTAIN"
    assert result["public_assignment"]["geometry_opened"] == 0
    assert result["public_assignment"]["assignments"] == []
    assert result["custodian_only_case_map"]["cases"] == []
    result = allocate(payload(30, 14))
    assert result["public_assignment"]["status"] == "CORPUS_CAPACITY_ABSTAIN"
