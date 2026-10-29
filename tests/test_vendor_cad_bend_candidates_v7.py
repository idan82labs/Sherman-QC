import copy
import pytest
from tests.test_vendor_cad_bend_candidates_v4 import feature as frame_feature
from backend.vendor_cad_bend_candidates_v7 import axis_rows_v7, extraction_to_candidates
from backend.vendor_unfold_contract_v2 import PreviewContractError


def feature():
    axis = frame_feature()
    axis["domain_receipt"].update(certificate_schema_version=6,
        native_domain_certificates=[{"face_index": i, "status": "PROVED", "type_proven": True,
            "kind": "NATIVE_PCURVE_RECTANGLE_SEGMENT_UNION", "wire_count": 1,
            "declared_tolerance_mm": 1e-7, "tolerance_mm": 1e-7, "maximum_residual_mm": 0.,
            "bounds_metric_mm": [0., 2., 0., 30.], "simple_connected_outer_loop": True,
            "complete_side_coverage": True, "certificate_schema_version": 6,
            "error_budget": {"status": "PROVED", "source_tolerance_mm": 1e-7,
                "endpoint_side_join_mm": 0., "maximum_coefficient_corridor_mm": 0.,
                "curve_to_endpoint_chord_mm": 0., "adaptor_bounds_alignment_mm": 0.,
                "total_bound_mm": 0., "composition": "SIDE_JOIN_PLUS_2_CORRIDOR_PLUS_ADAPTOR_ALIGNMENT"}}
            for i in [2, 5]])
    return axis


def test_exact_certificate_custody_and_no_input_mutation():
    axis = feature(); old = copy.deepcopy(axis)
    row = axis_rows_v7([axis])[0]
    assert row["v6_provenance"]["extractor_schema_version"] == 6
    assert row["v6_provenance"]["native_domain_certificates"] == axis["domain_receipt"]["native_domain_certificates"]
    assert "v4_provenance" not in row and axis == old


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "refused", "tolerance", "residual", "bounds", "hole", "loop"])
def test_bad_or_incomplete_native_proof_refused(mutation):
    axis = feature(); domain = axis["domain_receipt"]; proof = domain["native_domain_certificates"][0]
    if mutation == "missing": domain["native_domain_certificates"].pop()
    elif mutation == "duplicate": domain["native_domain_certificates"][1]["face_index"] = 2
    elif mutation == "refused": proof["status"] = "REFUSED"
    elif mutation == "tolerance": proof["declared_tolerance_mm"] = 1.
    elif mutation == "residual": proof["maximum_residual_mm"] = float("nan")
    elif mutation == "bounds": proof["bounds_metric_mm"][0] = True
    elif mutation == "hole": proof["wire_count"] = 2
    else: proof["simple_connected_outer_loop"] = False
    with pytest.raises(PreviewContractError): axis_rows_v7([axis])


def test_wrong_extraction_version_is_not_relabelled():
    payload = {"kind": "sherman_qc_step_topology_bend_axes", "schema_version": 4, "bend_axes": [feature()]}
    with pytest.raises(PreviewContractError): extraction_to_candidates(payload)
    payload["schema_version"] = 6
    assert len(extraction_to_candidates(payload)["candidates"]) == 1


@pytest.mark.parametrize("field,value", [("certificate_schema_version", 6.0),
    ("wire_count", True), ("wire_count", 1.0), ("face_index", 2.0), ("face_index", True)])
def test_float_and_bool_are_not_native_integer_identity(field, value):
    axis = feature()
    if field == "certificate_schema_version":
        axis["domain_receipt"][field] = value
    else:
        axis["domain_receipt"]["native_domain_certificates"][0][field] = value
    with pytest.raises(PreviewContractError):
        axis_rows_v7([axis])


def test_float_extraction_schema_refused():
    with pytest.raises(PreviewContractError):
        extraction_to_candidates({"kind": "sherman_qc_step_topology_bend_axes",
                                  "schema_version": 6.0, "bend_axes": [feature()]})


@pytest.mark.parametrize("mutation", ["missing", "sum", "chord", "source", "nonfinite", "schema"])
def test_budget_cannot_hide_individually_small_errors(mutation):
    axis = feature(); proof = axis["domain_receipt"]["native_domain_certificates"][0]
    budget = proof["error_budget"]
    if mutation == "missing": del proof["error_budget"]
    elif mutation == "sum": budget["endpoint_side_join_mm"] = 5e-8
    elif mutation == "chord": budget["maximum_coefficient_corridor_mm"] = 1e-8
    elif mutation == "source": budget["source_tolerance_mm"] = 1e-6
    elif mutation == "nonfinite": budget["total_bound_mm"] = float("nan")
    else: proof["certificate_schema_version"] = 6.0
    with pytest.raises(PreviewContractError): axis_rows_v7([axis])
