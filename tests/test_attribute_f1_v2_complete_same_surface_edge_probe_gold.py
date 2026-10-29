from __future__ import annotations

from copy import deepcopy

from scripts.attribute_f1_v2_complete_same_surface_edge_probe_gold import (
    BUNDLE_KIND,
    BUNDLE_PROFILE,
    KIND,
    LOCK_ATTESTATION_KIND,
    LOCK_CRITERION_KEYS,
    POSTLOCK_ROW_FIELDS,
    _sha256_json,
    _validate_bundle_lock,
    _validate_synthetic_resource_reference,
    attribute_gold,
)


SHA = "2" * 64


def _lock() -> dict:
    record = {
        "kind": LOCK_ATTESTATION_KIND,
        "criteria": {key: True for key in LOCK_CRITERION_KEYS},
        "ready_for_gold": True,
    }
    record["lock_attestation_sha256"] = _sha256_json(record)
    return record


def _novelty(*, dependent: bool, equivalent: bool) -> dict:
    equivalents = ["CONTROL_1"] if equivalent else []
    record = {
        "case_token": "C001",
        "replacement_certificate_id": "VEC_1",
        "edge_semantics_dependent": dependent,
        "causal_added_same_surface_edge_ids_sha256": SHA,
        "control_equivalent_certificate_ids": equivalents,
        "control_equivalent_certificate_ids_sha256": _sha256_json(equivalents),
        "novel_edge_attributable": dependent and not equivalent,
    }
    record["certificate_novelty_sha256"] = _sha256_json(record)
    return record


def _bundle(novelty: dict) -> dict:
    lock = _lock()
    return {
        "schema_version": 1,
        "kind": BUNDLE_KIND,
        "profile": BUNDLE_PROFILE,
        "locked": True,
        "ready_for_gold": True,
        "case_count": 1,
        "successful_case_count": 1,
        "synthetic_resource_attestation_sha256": SHA,
        "lock_attestation": lock,
        "lock_attestation_sha256": lock["lock_attestation_sha256"],
        "cases": [{
            "case_token": "C001",
            "scan_sha256": SHA,
            "a0_prediction_sha256": SHA,
            "certificate_novelty_table": [novelty],
        }],
    }


def _probe(*, dependent: bool) -> dict:
    return {
        "component_edge_attribution_table": [{
            "local_transition_component_id": "VEC_1",
            "edge_semantics_dependent": dependent,
            "causal_added_same_surface_edge_ids_sha256": SHA,
            "geometry_certificate_valid": True,
        }],
        "local_plane_pair_records": [{
            "components": [{
                "local_transition_component_id": "VEC_1",
                "span_endpoints": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
                "geometry_certificate_valid": True,
                "geometry_gate_depth": 13,
            }],
        }],
    }


def _truth_inputs() -> tuple[dict, dict, dict, dict]:
    prior = {"rows": [{
        "case_token": "C001", "gold_id": "G1", "first_failed_stage": 1,
    }]}
    split = {"cases": [{"case_token": "C001", "split_group_id": "FAMILY_A"}]}
    gold = {"cases": [{
        "case_token": "C001",
        "source_scan_sha256": SHA,
        "manual_bends": [{
            "bend_id": "G1",
            "span_endpoints": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        }],
    }]}
    predictions = {"cases": [{
        "case_token": "C001",
        "prediction_owned_regions_sha256": SHA,
        "a0_predictions": [],
    }]}
    return prior, split, gold, predictions


def test_bundle_lock_validation_rejects_nonexact_criteria() -> None:
    bundle = _bundle(_novelty(dependent=True, equivalent=False))
    lock = bundle["lock_attestation"]
    expected = lock["lock_attestation_sha256"]
    _validate_bundle_lock(
        bundle,
        actual_bundle_sha256=SHA,
        expected_bundle_sha256=SHA,
        expected_lock_attestation_sha256=expected,
        expected_case_tokens=("C001",),
    )

    changed = deepcopy(bundle)
    changed["lock_attestation"]["criteria"]["extra"] = True
    try:
        _validate_bundle_lock(
            changed,
            actual_bundle_sha256=SHA,
            expected_bundle_sha256=SHA,
            expected_lock_attestation_sha256=expected,
            expected_case_tokens=("C001",),
        )
    except ValueError as exc:
        assert "exact 39" in str(exc)
    else:
        raise AssertionError("extra lock criterion was accepted")


def test_postlock_refuses_synthetic_attestation_not_locked_by_bundle(
    tmp_path,
) -> None:
    path = tmp_path / "synthetic.json"
    path.write_text("{}", encoding="utf-8")
    bundle = _bundle(_novelty(dependent=True, equivalent=False))
    try:
        _validate_synthetic_resource_reference(bundle=bundle, attestation_path=path)
    except ValueError as exc:
        assert "differs from locked bundle" in str(exc)
    else:
        raise AssertionError("unlocked synthetic resource evidence was accepted")


def test_causal_novel_certificate_recovers_primary_row_exactly() -> None:
    novelty = _novelty(dependent=True, equivalent=False)
    prior, split, gold, predictions = _truth_inputs()
    result = attribute_gold(
        bundle=_bundle(novelty),
        probes={"C001": _probe(dependent=True)},
        prior_attribution=prior,
        grouped_split=split,
        sealed_gold=gold,
        a0_predictions=predictions,
        source_hashes={"replacement_bundle_sha256": SHA},
        expected_primary_count=1,
        enforce_frozen_false_count=False,
    )
    assert result["kind"] == KIND
    assert result["summary"]["edge_attributable_recovered_stage1_count"] == 1
    assert result["summary"]["replacement_recovered_stage1_count"] == 1
    assert set(result["rows"][0]) == POSTLOCK_ROW_FIELDS
    assert result["rows"][0]["reason_code"] == "VALID_NOVEL_EDGE_ATTRIBUTABLE_CERTIFICATE"
    assert result["rows"][0]["recovered"] is True


def test_noncausal_matching_certificate_is_excluded_from_numerator() -> None:
    novelty = _novelty(dependent=False, equivalent=False)
    prior, split, gold, predictions = _truth_inputs()
    result = attribute_gold(
        bundle=_bundle(novelty),
        probes={"C001": _probe(dependent=False)},
        prior_attribution=prior,
        grouped_split=split,
        sealed_gold=gold,
        a0_predictions=predictions,
        source_hashes={},
        expected_primary_count=1,
        enforce_frozen_false_count=False,
    )
    assert result["summary"]["replacement_recovered_stage1_count"] == 1
    assert result["summary"]["edge_attributable_recovered_stage1_count"] == 0
    assert result["rows"][0]["recovered"] is False
    assert result["rows"][0]["reason_code"] == "MATCHED_CERTIFICATE_NOT_EDGE_DEPENDENT"


def test_control_equivalent_matching_certificate_is_excluded() -> None:
    novelty = _novelty(dependent=True, equivalent=True)
    prior, split, gold, predictions = _truth_inputs()
    result = attribute_gold(
        bundle=_bundle(novelty),
        probes={"C001": _probe(dependent=True)},
        prior_attribution=prior,
        grouped_split=split,
        sealed_gold=gold,
        a0_predictions=predictions,
        source_hashes={},
        expected_primary_count=1,
        enforce_frozen_false_count=False,
    )
    assert result["summary"]["edge_attributable_recovered_stage1_count"] == 0
    assert result["summary"]["control_equivalent_recovery_count"] == 1
    assert result["rows"][0]["reason_code"] == "MATCHED_CONTROL_EQUIVALENT_CERTIFICATE"
