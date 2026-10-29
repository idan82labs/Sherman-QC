"""Authored transport adversaries; no native imports, catalog or physical CAD."""
import copy
import json

import pytest

from backend.vendor_unfold_contract_v2 import (PreviewContractError, REQUIRED_PROOFS,
    canonical_candidates, strict_json, unavailable, validate_preview)

HASH = "a" * 64
CANDIDATES = [{"feature_id": "B1", "line_start": [0, 0, 0], "line_end": [10, 0, 0]}]


def receipt():
    vertices = [0,0,0,10,0,0,0,10,0,10,10,0,20,0,0,20,10,0]
    diag = {key: True for key in REQUIRED_PROOFS}
    diag.update(mechanism_version=3, candidate_count=1, native_face_count=6,
        selected_source_face_indices=[0, 1, 2], selected_cylinder_face_indices=[1],
        root_source_face_index=0, selected_skin_rule="LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN",
        direction_status="UNKNOWN_NATIVE_NORMAL_SIGN_NOT_AUTHORITY",
        limitations=["VISUAL_ONLY_SELECTED_CAD_SKIN", "NOT_NEUTRAL_AXIS_OR_K_FACTOR_BLANK", "UNITS_ASSUMED_NOT_VERIFIED"])
    result = unavailable("placeholder", HASH)
    result.update(status="AVAILABLE", reason=None, vertices=vertices, triangles=[0,1,2,1,3,2,1,4,3,4,5,3],
        bend_lines=[{"feature_id": "B1", "line_start": [0, 0, 0], "line_end": [5, 5, 0],
                     "bend_direction": "unknown", "sweep_deg": 90}],
        flange_surfaces=[{"feature_id": "B1", "vertices": list(vertices), "triangles": [0, 1, 2]}], diagnostics=diag)
    return result


def test_valid_receipt_and_empty_abstention():
    assert validate_preview(receipt(), HASH, CANDIDATES)["status"] == "AVAILABLE"
    assert validate_preview(unavailable("SYNTHETIC_UNSUPPORTED", HASH), HASH, CANDIDATES)["vertices"] == []


def test_public_v2_gate_remains_v3_only_after_common_factoring():
    payload = receipt()
    payload["diagnostics"]["mechanism_version"] = 4
    with pytest.raises(PreviewContractError,match="MISSING_NATIVE_COVERAGE_PROOF"):
        validate_preview(payload,HASH,CANDIDATES)
    assert payload["diagnostics"]["mechanism_version"] == 4


@pytest.mark.parametrize("start,end",[([0,-.001,0],[1e-6,-.001,0]),
                                     ([-1e-6,0,0],[0,0,0])])
def test_gap_budget_cannot_admit_a_wholly_unsupported_short_marker(start,end):
    payload = receipt()
    payload["bend_lines"][0].update(line_start=start,line_end=end)
    with pytest.raises(PreviewContractError,match="FLAT_BEND_LINE_OUTSIDE_BODY"):
        validate_preview(payload,HASH,CANDIDATES)


@pytest.mark.parametrize("version",[True,3.,5,None,"4"])
def test_private_common_boundary_rejects_untyped_or_unsupported_versions(version):
    from backend.vendor_unfold_contract_v2 import _validate_preview
    with pytest.raises(PreviewContractError,match="UNSUPPORTED_COMMON_PREVIEW_MECHANISM"):
        _validate_preview(receipt(),HASH,CANDIDATES,mechanism_version=version)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '['*1100+'0'+']'*1100])
def test_strict_json_rejects_nonfinite_and_duplicate(raw):
    with pytest.raises(PreviewContractError):
        strict_json(raw)


@pytest.mark.parametrize("change", [
    lambda p: p.update(status=[]), lambda p: p.update(extra=True),
    lambda p: p.update(source_sha256="b"*64), lambda p: p.update(model="MANUFACTURING"),
    lambda p: p.update(units_verified=True), lambda p: p.update(manufacturing_ready=True),
    lambda p: p["diagnostics"].update(nested={"cut_release_allowed":True}),
    lambda p: p["diagnostics"].update(nested={"units_verified":True}),
    lambda p: p["diagnostics"].update(candidate_count=True),
    lambda p: p["diagnostics"].update(mechanism_version=True),
    lambda p: p["diagnostics"].update(selected_source_face_indices=[[0],1,2]),
    lambda p: p["diagnostics"].update(selected_source_face_indices=[0,0,2]),
    lambda p: p["diagnostics"].update(selected_cylinder_face_indices=[0],root_source_face_index=0),
    lambda p: p["diagnostics"].update(selected_skin_rule="arbitrary"),
    lambda p: p["diagnostics"].update(limitations=[]),
    lambda p: p.update(vertices=[0,0,0,10,0,0,0,10,0.1]),
    lambda p: p.update(triangles=[0,1,True]), lambda p: p.update(triangles=[0,1,6]),
    lambda p: p.update(triangles=[0,1,1]),
    lambda p: p["bend_lines"][0].update(feature_id="foreign"),
    lambda p: p["bend_lines"][0].update(line_start=[100,100,0]),
    lambda p: p["bend_lines"][0].update(sweep_deg=176),
    lambda p: p["bend_lines"][0].update(bend_direction="up"),
    lambda p: p.update(flange_surfaces=[]),
    lambda p: p["flange_surfaces"][0].update(vertices=[0,0,0,20,0,0,0,20,0]),
])
def test_malformed_receipts_refuse(change):
    payload = receipt()
    change(payload)
    with pytest.raises(PreviewContractError):
        validate_preview(payload, HASH, CANDIDATES)


@pytest.mark.parametrize("proof", REQUIRED_PROOFS)
def test_each_native_coverage_proof_is_required(proof):
    payload = receipt()
    del payload["diagnostics"][proof]
    with pytest.raises(PreviewContractError):
        validate_preview(payload, HASH, CANDIDATES)


@pytest.mark.parametrize("candidates", [None, [], [None], [{"feature_id":"", "line_start":[0,0,0], "line_end":[1,0,0]}],
    [{"feature_id":"B1", "line_start":[0,0,0], "line_end":[0,0,0]}],
    [{"feature_id":"B1", "line_start":[True,0,0], "line_end":[1,0,0]}], CANDIDATES+CANDIDATES])
def test_bad_candidate_inputs_refuse(candidates):
    with pytest.raises(PreviewContractError):
        canonical_candidates(candidates)


def test_huge_integer_is_typed_refusal_not_overflow():
    with pytest.raises(PreviewContractError):
        canonical_candidates([{"feature_id":"B1","line_start":[10**400,0,0],"line_end":[1,0,0]}])
    payload=receipt()
    payload["bend_lines"][0]["sweep_deg"]=10**400
    with pytest.raises(PreviewContractError):
        validate_preview(payload,HASH,CANDIDATES)


def test_identifier_limit_applies_to_actual_identity_not_stripped_length():
    with pytest.raises(PreviewContractError):
        canonical_candidates([{**CANDIDATES[0],"feature_id":" "*101+"B1"}])


def test_no_geometry_on_unavailable():
    payload = unavailable("SYNTHETIC_UNSUPPORTED", HASH)
    payload["vertices"] = [0]
    with pytest.raises(PreviewContractError):
        validate_preview(payload, HASH, CANDIDATES)


def test_float32_collapse_is_refused():
    payload = receipt()
    payload["vertices"] = [100000,100000,0,100000.001,100000,0,100000,100000.001,0]
    payload["triangles"] = [0,1,2]
    with pytest.raises(PreviewContractError, match="FLOAT32"):
        validate_preview(payload, HASH, CANDIDATES)


def test_overlay_cannot_fill_hole_using_only_existing_vertices():
    # Authored ring made of eight triangles, absent central square.
    payload = receipt()
    payload["vertices"] = [0,0,0,4,0,0,4,4,0,0,4,0,1,1,0,3,1,0,3,3,0,1,3,0]
    payload["triangles"] = [0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]
    payload["bend_lines"][0].update(line_start=[0,0,0],line_end=[4,0,0])
    payload["flange_surfaces"][0].update(vertices=payload["vertices"],triangles=[4,5,6])
    with pytest.raises(PreviewContractError, match="SUBSET"):
        validate_preview(payload, HASH, CANDIDATES)


def test_complete_marker_crossing_off_midpoint_hole_is_refused():
    payload=receipt()
    payload["vertices"]=[0,0,0,10,0,0,10,10,0,0,10,0,2,4,0,3,4,0,3,6,0,2,6,0]
    payload["triangles"]=[0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]
    payload["flange_surfaces"][0].update(vertices=payload["vertices"],triangles=[0,1,5])
    payload["bend_lines"][0].update(line_start=[0,5,0],line_end=[10,5,0])
    with pytest.raises(PreviewContractError,match="OUTSIDE_BODY"):
        validate_preview(payload,HASH,CANDIDATES)


def test_double_supported_marker_outside_float32_body_is_refused():
    payload=receipt()
    origin=100000.
    # Three adjacent slanted planar bands; every Float32 triangle stays valid.
    points=[]
    for x in [0.,2.,4.]:
        points.extend([origin+x,origin+x+.004,0.,origin+x,origin+x+2.,0.])
    payload["vertices"]=points
    payload["triangles"]=[0,2,3,0,3,1,2,4,5,2,5,3]
    payload["flange_surfaces"][0].update(vertices=points,triangles=[0,2,3])
    payload["bend_lines"][0].update(line_start=[origin+.004,origin+.00800001,0.],
        line_end=[origin+3.996,origin+4.00000001,0.])
    with pytest.raises(PreviewContractError,match="OUTSIDE_BODY"):
        validate_preview(payload,HASH,CANDIDATES)


def test_unordered_float32_triangles_must_be_unique_after_conversion():
    payload=receipt()
    payload["vertices"]=[100000.001,100001,0,100000.002,100000,0,100001,100000,0,
        100000.002,100001,0,100000.001,100000,0,100001.001,100000,0,
        100003,100000,0,100004,100000,0,100003,100001,0]
    payload["triangles"]=[0,1,2,3,4,5,6,7,8]
    payload["flange_surfaces"][0].update(vertices=payload["vertices"],triangles=[0,1,2])
    payload["bend_lines"][0].update(line_start=[100003.1,100000.1,0],line_end=[100003.2,100000.2,0])
    with pytest.raises(PreviewContractError,match="FLOAT32_DUPLICATE"):
        validate_preview(payload,HASH,CANDIDATES)


def test_candidate_canonicalization_ignores_only_authoring_metadata():
    annotated = copy.deepcopy(CANDIDATES)
    annotated[0].update(label="operator label", note="review", review_status="confirmed")
    assert canonical_candidates(annotated) == CANDIDATES
    assert strict_json(json.dumps(receipt())) == receipt()
