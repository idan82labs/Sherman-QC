"""Full V4 receipt tests from pure authored charts; no OCP, CAD or file jobs.

These contracts exercise typed transport, deterministic chart geometry and
diagnostic composition. They do not qualify native STEP support or shop use.
Run only after the private V2 common-validator interface is reviewed/held.
"""
import copy
import math

import numpy as np
import pytest

from backend import vendor_unfold_contract_v2 as common
from backend import vendor_unfold_contract_v3 as contract
from backend import vendor_unfold_finite_width_contract_v4 as addon
from backend import vendor_unfold_preview as legacy_engine
from backend import vendor_unfold_preview_v4 as engine
from tests.test_vendor_unfold_preview_v4 import HASH, chain, posed, quarter


def developed(case=None):
    faces, axes, candidates = copy.deepcopy(quarter() if case is None else case)
    payload = engine._develop_faces(faces, axes, candidates, HASH)
    # This authors only the new transport declaration on an actual pure V4
    # mechanism receipt; it never rewrites a V3 mechanism to resemble V4.
    payload["diagnostics"].update(transport_version=3,
        candidate_binding=common.canonical_candidates(candidates),
        v4_engine_source_sha256=addon.V4_ENGINE_SHA256)
    return payload, candidates


def with_hole():
    faces, axes, candidates = quarter()
    points = np.array([[4,-30,0], [4,0,0], [4,0,12], [4,-30,12],
                       [4,-20,3], [4,-15,3], [4,-15,8], [4,-20,8]], dtype=float)
    triangles = np.array([[0,1,5], [0,5,4], [1,2,6], [1,6,5],
                          [2,3,7], [2,7,6], [3,0,4], [3,4,7]], dtype=np.int32)
    for index in (0, 3):
        faces[index].vertices = points + [index / 3, 0, 0]
        faces[index].triangles = triangles.copy()
    return faces, axes, candidates


def refuses(payload, candidates, reason=".", expected_hash=HASH):
    before = copy.deepcopy((payload, candidates))
    with pytest.raises(common.PreviewContractError, match=reason):
        contract.validate_preview(payload, expected_hash, candidates)
    assert (payload, candidates) == before


@pytest.mark.parametrize("case", [quarter(), chain(), posed(quarter()), posed(quarter(), mirrored=True), with_hole()])
def test_pure_quarter_chain_pose_mirror_and_hole_pass_full_composition_without_mutation(case):
    payload, candidates = developed(case)
    before = copy.deepcopy((payload, candidates))
    assert contract.validate_preview(payload, HASH, candidates) is payload
    assert (payload, candidates) == before
    assert payload["diagnostics"]["mechanism_version"] == 4
    assert payload["diagnostics"]["transport_version"] == 3
    assert payload["units"] == "assumed_mm"
    assert payload["units_verified"] is False and payload["manufacturing_ready"] is False


def test_full_common_gate_runs_before_addon_and_returns_original(monkeypatch):
    payload, candidates = developed()
    events = []
    actual_common = contract._validate_common_preview
    actual_addon = contract.validate_finite_width_diagnostics
    def checked_common(receipt, expected, rows, *, mechanism_version):
        events.append(("common", mechanism_version))
        return actual_common(receipt, expected, rows, mechanism_version=mechanism_version)
    def checked_addon(receipt, rows):
        events.append(("addon", receipt["diagnostics"]["mechanism_version"]))
        return actual_addon(receipt, rows)
    monkeypatch.setattr(contract, "_validate_common_preview", checked_common)
    monkeypatch.setattr(contract, "validate_finite_width_diagnostics", checked_addon)
    assert contract.validate_preview(payload, HASH, candidates) is payload
    assert events == [("common", 4), ("addon", 4)]


def test_addon_alone_cannot_admit_missing_common_body_source_or_markers():
    from tests.test_vendor_unfold_finite_width_contract_v4 import authored

    payload, candidates = authored()
    assert addon.validate_finite_width_diagnostics(payload, candidates) is None
    refuses(payload, candidates, "INVALID_PREVIEW_ENVELOPE")


def test_public_v2_contract_stays_strict_mechanism_three():
    payload, candidates = developed()
    with pytest.raises(common.PreviewContractError):
        common.validate_preview(payload, HASH, candidates)


def test_v3_core_is_never_implicitly_or_explicitly_relabelled_as_v4():
    case = quarter()
    payload = legacy_engine._develop_faces(*copy.deepcopy(case), HASH)
    candidates = case[2]
    payload["diagnostics"]["transport_version"] = 3
    refuses(payload, candidates)
    assert payload["diagnostics"]["mechanism_version"] == 3
    payload["diagnostics"]["mechanism_version"] = 4  # Adversarial label only, no V4 proofs.
    refuses(payload, candidates, "FINITE_WIDTH|UNREVIEWED")


@pytest.mark.parametrize("version", [None, 2, 4, 3.0, True, "3", []])
def test_transport_three_is_required_as_a_typed_integer(version):
    payload, candidates = developed()
    payload["diagnostics"]["transport_version"] = version
    refuses(payload, candidates, "TRANSPORT")


@pytest.mark.parametrize("version", [None, 3, 5, 4.0, True, "4", []])
def test_mechanism_four_is_required_as_a_typed_integer(version):
    payload, candidates = developed()
    payload["diagnostics"]["mechanism_version"] = version
    refuses(payload, candidates)


@pytest.mark.parametrize("change", ["missing_proof", "reset_shared", "global_drift", "pair_interleave", "tree", "foreign_id"])
def test_multi_bend_chain_addon_cannot_be_bypassed_by_common_mesh_validity(change):
    payload, candidates = developed(chain())
    diag = payload["diagnostics"]
    first, second = diag["finite_width_proofs"]
    if change == "missing_proof":
        diag["finite_width_proofs"].pop()
    elif change == "reset_shared":
        second["parent"]["shared_work"] = first["parent"]["shared_work"].copy()
    elif change == "global_drift":
        diag["finite_width_work"]["triangles"] += 1
    elif change == "pair_interleave":
        first["child"]["shared_work"], second["parent"]["shared_work"] = second["parent"]["shared_work"], first["child"]["shared_work"]
    elif change == "tree":
        second["parent_source_face_index"] = second["child_source_face_index"]
    else:
        second["feature_id"] = "foreign"
    refuses(payload, candidates, "FINITE_WIDTH")


@pytest.mark.parametrize("field", addon.OBLIGATIONS)
def test_every_common_and_finite_width_obligation_is_mandatory(field):
    payload, candidates = developed()
    payload["diagnostics"][field] = False
    refuses(payload, candidates)


@pytest.mark.parametrize("change", ["missing", "overlap", "foreign_triangle", "body_copy"])
def test_overlays_remain_exact_disjoint_proper_body_subsets(change):
    payload, candidates = developed(chain())
    surfaces = payload["flange_surfaces"]
    if change == "missing":
        surfaces.pop()
    elif change == "overlap":
        surfaces[1].update(vertices=copy.deepcopy(surfaces[0]["vertices"]), triangles=copy.deepcopy(surfaces[0]["triangles"]))
    elif change == "foreign_triangle":
        surfaces[0].update(vertices=[0,0,0,1,0,0,0,1,0], triangles=[0,1,2])
    else:
        surfaces[0].update(vertices=copy.deepcopy(payload["vertices"]), triangles=copy.deepcopy(payload["triangles"]))
    refuses(payload, candidates)


def test_retained_hole_cannot_be_filled_by_overlay():
    payload, candidates = developed(with_hole())
    payload["flange_surfaces"][0].update(vertices=payload["vertices"][:24], triangles=[4,5,6])
    refuses(payload, candidates, "SUBSET")


def test_marker_must_be_supported_along_entire_span_not_cross_retained_hole():
    payload, candidates = developed(with_hole())
    payload["bend_lines"][0].update(line_start=[0,-12.5,0], line_end=[12,-12.5,0])
    refuses(payload, candidates, "OUTSIDE_BODY")


@pytest.mark.parametrize("change", ["outside", "nonplanar", "degenerate", "foreign_feature", "direction", "sweep"])
def test_markers_remain_finite_planar_supported_and_unknown_direction(change):
    payload, candidates = developed()
    line = payload["bend_lines"][0]
    if change == "outside":
        line.update(line_start=[1000,1000,0], line_end=[1001,1000,0])
    elif change == "nonplanar":
        line["line_start"][2] = .1
    elif change == "degenerate":
        line["line_end"] = line["line_start"].copy()
    elif change == "foreign_feature":
        line["feature_id"] = "unknown"
    elif change == "direction":
        line["bend_direction"] = "up"
    else:
        line["sweep_deg"] = 176
    refuses(payload, candidates)


@pytest.mark.parametrize("change", ["payload_hash", "expected_hash", "candidate_binding", "runtime_declaration", "helper_declaration"])
def test_source_and_candidate_binding_remain_mandatory(change):
    payload, candidates = developed()
    expected = HASH
    if change == "payload_hash":
        payload["source_sha256"] = "b"*64
    elif change == "expected_hash":
        expected = "b"*64
    elif change == "candidate_binding":
        payload["diagnostics"]["candidate_binding"][0]["line_end"][0] += 1
    elif change == "runtime_declaration":
        payload["diagnostics"]["v4_engine_source_sha256"] = "b"*64
    else:
        payload["diagnostics"]["finite_width_source_sha256"] = "b"*64
    refuses(payload, candidates, expected_hash=expected)


@pytest.mark.parametrize("field", ["manufacturing_ready", "units_verified", "cut_release_allowed", "physical_calibration_verified"])
def test_nested_or_envelope_approval_claims_remain_false(field):
    payload, candidates = developed()
    payload["diagnostics"]["unearned_approval"] = {field: True}
    refuses(payload, candidates, "CONTRADICTORY_RELEASE_CLAIM")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "0"])
def test_nonfinite_or_untyped_body_coordinates_are_refused(value):
    payload, candidates = developed()
    payload["vertices"][0] = value
    # NaN cannot be equality-compared in the unchanged-input helper.
    with pytest.raises(common.PreviewContractError):
        contract.validate_preview(payload, HASH, candidates)


@pytest.mark.parametrize("source_hash", [None, HASH])
def test_typed_empty_unavailable_needs_no_successful_width_proofs(source_hash, monkeypatch):
    payload = contract.unavailable("SYNTHETIC_ABSTENTION", source_hash)
    candidates = quarter()[2]
    before = copy.deepcopy(payload)
    monkeypatch.setattr(contract, "validate_finite_width_diagnostics", lambda *args: pytest.fail("Unavailable must not require success proofs"))
    assert contract.validate_preview(payload, HASH, candidates) is payload
    assert payload == before and payload["source_sha256"] == source_hash
    assert all(payload[key] == [] for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"))
    assert payload["diagnostics"] == {"transport_version": 3, "mechanism_version": 4, "visual_only": True}


@pytest.mark.parametrize("reason", [None, True, "", "   ", "x"*201, []])
def test_unavailable_factory_requires_nonblank_bounded_reason(reason):
    with pytest.raises(common.PreviewContractError, match="INVALID_UNAVAILABLE_REASON"):
        contract.unavailable(reason)


@pytest.mark.parametrize("source_hash", ["", "a"*63, "a"*65, "A"*64, "g"*64, True, []])
def test_unavailable_factory_requires_null_or_exact_lowercase_hash(source_hash):
    with pytest.raises(common.PreviewContractError, match="INVALID_SOURCE_HASH"):
        contract.unavailable("SYNTHETIC_ABSTENTION", source_hash)


def test_reason_maximum_is_allowed_and_factories_do_not_share_mutable_geometry():
    first = contract.unavailable("x"*200)
    second = contract.unavailable("SYNTHETIC_ABSTENTION")
    first["vertices"].append(0)
    assert second["vertices"] == []


@pytest.mark.parametrize("change", ["source", "geometry", "transport", "mechanism", "reason", "release"])
def test_unavailable_keeps_full_common_empty_source_claim_and_version_contract(change):
    payload, candidates = contract.unavailable("SYNTHETIC_ABSTENTION", HASH), quarter()[2]
    if change == "source":
        payload["source_sha256"] = "b"*64
    elif change == "geometry":
        payload["vertices"] = [0]
    elif change == "transport":
        payload["diagnostics"]["transport_version"] = 2
    elif change == "mechanism":
        payload["diagnostics"]["mechanism_version"] = 3
    elif change == "reason":
        payload["reason"] = "x"*201
    else:
        payload["manufacturing_ready"] = True
    refuses(payload, candidates)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'])
def test_strict_json_reuse_never_admits_duplicate_or_nonfinite_fields(raw):
    assert contract.strict_json is common.strict_json
    assert contract.canonical_candidates is common.canonical_candidates
    with pytest.raises(common.PreviewContractError):
        contract.strict_json(raw)
