"""In-memory report oracles; no raw inputs, artifacts, models or server imports."""
from copy import deepcopy
from itertools import permutations
import json

import pytest
from pydantic import ValidationError

from contracts.ui.bend_results import BendDecisionContract

from domains.bend.application.finalize_bend_report import finalize_bend_report, format_bend_decision_table
from domains.bend.mappers.ui_contracts import map_bend_report_to_contract


def row(identity, state="UNKNOWN", **fields):
    return {"bend_id": identity, "observability_state": state, "status": "PASS", **fields}


def decision(rows, **fields):
    return finalize_bend_report({"matches": rows, **fields})["decision_contract"]


def test_counts_require_explicit_observation_and_ignore_legacy_and_model_totals():
    rows = [row("A", "FORMED"), row("B", "OBSERVED_NOT_FORMED"), row("C", "PARTIALLY_OBSERVED")]
    expected = {"formed": 1, "unformed": 1, "unknown": 1, "total": 3, "partially_observed": 1}
    for count in [-20, 0, 2, 999]:
        d = decision(rows, summary={"completed_bends": count, "structured_completed_bends": count, "expected_bends": count})
        assert d["observed_counts"] == expected
        assert d["production_disposition"] == "NOT_AUTHORIZED"


@pytest.mark.parametrize("status", ["PASS", "FAIL", "WARNING", "NOT_DETECTED"])
def test_unknown_is_not_resolved_by_tolerance_status_or_completion(status):
    d = decision([row("A", status=status, physical_completion_state="FORMED")])
    assert d["observed_counts"]["unknown"] == 1
    assert d["bends"][0]["state"] == "UNKNOWN"


def test_partial_visibility_and_partial_formation_are_different_unknowns():
    d = decision([
        row("A", "FORMED", observability_detail_state="PARTIALLY_OBSERVED"),
        row("B", "FORMED", physical_completion_state="PARTIALLY_FORMED"),
    ])
    assert d["observed_counts"]["unknown"] == 2
    assert d["observed_counts"]["partially_observed"] == 1
    assert "INCOMPLETE_OBSERVATION" in d["bends"][0]["reasons"]
    assert "PHYSICAL_STAGE_NOT_QUALIFIED" in d["bends"][1]["reasons"]


@pytest.mark.parametrize("fields", [
    {"observability_detail_state": "UNOBSERVED"},
    {"observability_detail_state": "OBSERVED_NOT_FORMED"},
    {"observability_detail_state": "UNFORMED"},
    {"physical_completion_state": "UNFORMED"},
])
def test_conflicting_evidence_abstains(fields):
    assert decision([row("A", "FORMED", **fields)])["observed_counts"]["unknown"] == 1


def test_registration_abstention_suppresses_even_explicit_states():
    d = decision([row("A", "FORMED"), row("B", "UNFORMED")], alignment={"alignment_abstained": True})
    assert d["registration"] == "ABSTAINED"
    assert d["observed_counts"]["unknown"] == 2
    assert {item["state"] for item in d["bends"]} == {"UNKNOWN"}


@pytest.mark.parametrize("alignment", [None, {}, {"alignment_abstained": False}, {"alignment_confidence": 1.0}, {"registration_trusted": True}])
def test_no_registration_qualification_is_invented_from_flags_or_confidence(alignment):
    assert decision([row("A", "FORMED")], alignment=alignment)["registration"] == "UNVERIFIED"


def test_countable_events_exclude_process_features_without_hiding_their_rows():
    d = decision([row("A", "FORMED"), row("P", "FORMED", countable_in_regression=False),
                  row("Q", "UNFORMED", cad_bend={"countable_in_regression": False})])
    assert len(d["bends"]) == 3
    assert d["observed_counts"] == {"formed": 1, "unformed": 0, "unknown": 0, "total": 1, "partially_observed": 0}


def test_null_top_level_countability_falls_back_to_cad_like_ui_and_export():
    d = decision([
        row("P", "FORMED", countable_in_regression=None, cad_bend={"countable_in_regression": False}),
        row("A", "FORMED", countable_in_regression=None),
        row("B", "UNFORMED", countable_in_regression=True, cad_bend={"countable_in_regression": False}),
    ])
    assert len(d["bends"]) == 3
    assert d["observed_counts"] == {"formed": 1, "unformed": 1, "unknown": 0, "total": 2, "partially_observed": 0}


@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}])
@pytest.mark.parametrize("nested", [False, True])
def test_malformed_effective_countability_retains_unknown_target(value, nested):
    fields = {"countable_in_regression": value}
    if nested:
        fields = {"countable_in_regression": None, "cad_bend": fields}
    d = decision([row("A", "FORMED", **fields)])
    assert d["observed_counts"] == {"formed": 0, "unformed": 0, "unknown": 1, "total": 1, "partially_observed": 0}
    assert "MALFORMED_COUNTABILITY_EVIDENCE" in d["bends"][0]["reasons"]


@pytest.mark.parametrize("alignment", [True, [], "unverified",
    {"alignment_abstained": 0}, {"alignment_abstained": 1},
    {"alignment_abstained": "true"}, {"alignment_abstained": "false"},
    {"alignment_abstained": []}, {"alignment_abstained": {}}])
def test_malformed_registration_evidence_abstains_instead_of_preserving_known_states(alignment):
    d = decision([row("A", "FORMED"), row("B", "UNFORMED")], alignment=alignment)
    assert d["registration"] == "ABSTAINED"
    assert d["observed_counts"]["unknown"] == 2
    assert "MALFORMED_REGISTRATION_EVIDENCE" in d["reasons"]
    assert BendDecisionContract(**d).registration == "ABSTAINED"


def test_duplicate_identity_does_not_increase_evidence_or_depend_on_row_order():
    rows = [row("A", "FORMED"), row("A", "UNFORMED", physical_completion_state="PARTIALLY_FORMED"), row("B", "UNFORMED")]
    variants = [decision(list(order)) for order in permutations(rows)]
    assert all(d == variants[0] for d in variants)
    assert variants[0]["observed_counts"] == {"formed": 0, "unformed": 1, "unknown": 1, "total": 2, "partially_observed": 0}
    assert "DUPLICATE_BEND_ID" in variants[0]["bends"][0]["reasons"]


def test_finalization_is_nonmutating_idempotent_and_json_mapper_preserves_evidence():
    report = {"summary": {"total_bends": 2, "completed_bends": 999},
              "matches": [row("B"), row("A", "FORMED")]}
    saved = deepcopy(report)
    finalized = finalize_bend_report(report, render_artifacts_available=True)
    assert report == saved
    assert finalized["summary"] == saved["summary"]
    assert finalize_bend_report(finalized, render_artifacts_available=True) == finalized
    roundtrip = json.loads(json.dumps(finalized, allow_nan=False))
    contract = map_bend_report_to_contract("TEST", roundtrip)
    assert contract.decision_contract.model_dump() == finalized["decision_contract"]
    assert finalized["decision_contract"]["render_evidence"] == "UNVERIFIED"


def test_missing_artifacts_are_not_qualified_by_display_geometry_flags():
    d = decision([row("A", "FORMED", display_geometry_valid=True)])
    assert d["render_evidence"] == "UNAVAILABLE"
    assert d["production_disposition"] == "NOT_AUTHORIZED"


def test_missing_identity_cannot_contribute_positive_evidence():
    d = decision([{"observability_state": "FORMED", "status": "PASS"}])
    assert d["observed_counts"]["formed"] == 0
    assert "EVENT_ID_UNAVAILABLE" in d["reasons"]


def test_missing_match_records_do_not_inherit_summary_counts():
    d = finalize_bend_report({"summary": {"completed_bends": 10}})["decision_contract"]
    assert d["observed_counts"]["total"] == 0
    assert "MATCH_RECORDS_UNAVAILABLE" in d["reasons"]


def test_legacy_mapper_does_not_manufacture_new_evidence_contract():
    assert map_bend_report_to_contract("TEST", {"summary": {}, "matches": []}).decision_contract is None


def test_text_export_uses_same_evidence_not_legacy_completion_claim():
    report = finalize_bend_report({"matches": [row("A"), row("B", "UNFORMED")], "summary": {"completed_bends": 999}})
    text = format_bend_decision_table(report)
    assert "Observed formed: 0/2" in text
    assert "Observed unformed: 1; Unknown: 1" in text
    assert "999" not in text
    assert "NOT QUALIFIED FOR PRODUCTION ACCEPT/REJECT" in text
    assert "A | UNKNOWN" in text


@pytest.mark.parametrize("bad_counts", [
    {"formed": True}, {"formed": "1"}, {"formed": -1},
    {"total": 2}, {"partially_observed": 1},
])
def test_typed_contract_rejects_inconsistent_or_coerced_counts(bad_counts):
    d = decision([row("A", "FORMED")])
    d["observed_counts"].update(bad_counts)
    with pytest.raises(ValidationError):
        BendDecisionContract(**d)


def test_typed_contract_rejects_abstained_positive_observation():
    d = decision([row("A", "FORMED")])
    d["registration"] = "ABSTAINED"
    with pytest.raises(ValidationError):
        BendDecisionContract(**d)


@pytest.mark.parametrize("field", ["observability_state", "observability_detail_state", "physical_completion_state"])
@pytest.mark.parametrize("value", [7, True, [], {}, "garbage"])
def test_malformed_state_evidence_never_releases_a_positive_observation(field, value):
    match = row("A", "FORMED")
    match[field] = value
    d = decision([match])
    assert d["observed_counts"]["unknown"] == 1
    assert "MALFORMED_STATE_EVIDENCE" in d["bends"][0]["reasons"]


@pytest.mark.parametrize("value", [True, 1.0, "1", 2, None])
def test_typed_schema_version_is_explicit_and_not_coerced(value):
    d = decision([row("A", "FORMED")])
    d["schema_version"] = value
    with pytest.raises(ValidationError):
        BendDecisionContract(**d)


def test_typed_contract_rejects_missing_version_and_contradictory_detail():
    d = decision([row("A", "FORMED")])
    del d["schema_version"]
    with pytest.raises(ValidationError):
        BendDecisionContract(**d)
    d["schema_version"] = 1
    d["bends"][0]["observation_detail"] = "UNOBSERVED"
    with pytest.raises(ValidationError):
        BendDecisionContract(**d)
