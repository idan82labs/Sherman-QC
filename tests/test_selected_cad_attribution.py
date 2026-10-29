"""Isolated source contracts for selected-CAD mesh-surface attribution.

Read only the named pipeline source file and compile a closed whitelist of pure
helper definitions. Never import the legacy pipeline, Open3D, CAD kernels, app
services or runtime initialization. All candidate objects are invented in memory.
Local extractor surface IDs are neither native face identity nor scan evidence.
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


PIPELINE_SOURCE = Path(__file__).resolve().parents[1]/"backend"/"bend_inspection_pipeline.py"
HELPER_NAMES = frozenset({
    "_selected_cad_attribution_vector",
    "_selected_cad_extracted_identity",
    "_cad_spec_to_measurement_dict",
    "_selected_cad_attribution_geometry",
    "_selected_cad_attribution_pairs",
    "_match_selected_spec_to_extracted_bend",
    "_build_selected_measurement_dicts",
})


@pytest.fixture(scope="module")
def helpers():
    tree = ast.parse(PIPELINE_SOURCE.read_text(encoding="utf-8"), filename=str(PIPELINE_SOURCE))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES]
    assert {node.name for node in definitions} == HELPER_NAMES
    assert len(definitions) == len(HELPER_NAMES)
    assert all(not node.decorator_list for node in definitions)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                   for function in definitions for node in ast.walk(function))
    isolated = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                          +definitions, type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = dict(np=np, copy=copy, math=math, Any=Any, Dict=Dict, List=List,
                     Optional=Optional, Tuple=Tuple, BendSpecification=SimpleNamespace)
    exec(compile(isolated, str(PIPELINE_SOURCE), "exec"), namespace)
    return namespace


def selected(identifier="selected_A", *, center=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
             length=10.0, angle=90.0):
    center, direction = np.asarray(center, dtype=float), np.asarray(direction, dtype=float)
    direction = direction/np.linalg.norm(direction)
    return SimpleNamespace(
        bend_id=identifier, bend_line_start=center-0.5*length*direction,
        bend_line_end=center+0.5*length*direction, bend_line_direction=direction.copy(),
        target_angle=angle, flange1_normal=np.array([0.0, 1.0, 0.0]),
        flange2_normal=np.array([0.0, 0.0, 1.0]), bend_form="FOLDED",
        feature_type="DISCRETE_BEND", countable_in_regression=True,
    )


class Extracted(SimpleNamespace):
    def to_dict(self):
        raise AssertionError("attribution must whitelist IDs, never copy extracted payloads")


def extracted(identifier="extracted_A", *, center=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
              length=10.0, angle=90.0, surface_ids=(11, 12)):
    center, direction = np.asarray(center, dtype=float), np.asarray(direction, dtype=float)
    direction = direction/np.linalg.norm(direction)
    return Extracted(
        bend_id=identifier, bend_center=center.copy(), bend_direction=direction.copy(),
        bend_line_start=center-0.5*length*direction, bend_line_end=center+0.5*length*direction,
        bend_angle=angle, surface1_normal=np.array([0.0, 1.0, 0.0]),
        surface2_normal=np.array([0.0, 0.0, 1.0]), surface1_id=surface_ids[0], surface2_id=surface_ids[1],
        confidence=1.0, status="PASS", success=True, observed_state="FORMED",
        native_topology_verified=True, physical_geometry_is_verified=True,
        native_face_id="invented_untrusted_native_claim",
    )


def assert_selected_geometry(payload, spec):
    assert payload["bend_id"] == payload["bend_name"] == spec.bend_id
    assert payload["bend_angle"] == spec.target_angle
    for output, source in (("bend_line_start", "bend_line_start"), ("bend_line_end", "bend_line_end"),
                           ("bend_direction", "bend_line_direction"), ("surface1_normal", "flange1_normal"),
                           ("surface2_normal", "flange2_normal")):
        if getattr(spec, source) is not None:
            np.testing.assert_array_equal(payload[output], getattr(spec, source))
    np.testing.assert_array_equal(payload["bend_center"], 0.5*(spec.bend_line_start+spec.bend_line_end))
    assert payload["bend_form"] == spec.bend_form
    assert payload["feature_type"] == spec.feature_type
    assert payload["countable_in_regression"] is bool(spec.countable_in_regression)
    assert payload["selected_baseline_direct"] is True
    assert not {"confidence", "status", "success", "observed_state", "native_face_id",
                "native_topology_verified", "physical_geometry_is_verified"} & set(payload)


def assert_attribution(payload, *, status, reason, index=None):
    attribution = payload["mesh_surface_attribution"]
    assert attribution["status"] == status
    assert attribution["reason"] == reason
    assert attribution["scope"] == "EXTRACTOR_LOCAL_MESH_SURFACE_IDS_NOT_NATIVE_FACE_IDENTITY"
    assert attribution["native_topology_verified"] is False
    assert attribution["observed_state"] == "UNKNOWN"
    assert attribution["extracted_index"] == index
    if status == "UNATTRIBUTED":
        assert "geometry" not in attribution
        assert payload["surface1_id"] < 0 and payload["surface2_id"] < 0
        assert payload["surface1_id"] != payload["surface2_id"]
    else:
        assert status == "GEOMETRY_COMPATIBLE_UNIQUE"
        assert payload["surface1_id"] >= 0 and payload["surface2_id"] >= 0
        assert payload["surface1_id"] != payload["surface2_id"]
        geometry = attribution["geometry"]
        assert geometry["surface_ids"] == [payload["surface1_id"], payload["surface2_id"]]
        assert all(np.isfinite(geometry[key]) for key in (
            "center_distance_mm", "endpoint_distance_mm", "bend_angle_gap_degrees",
            "direction_alignment", "numerical_distance_allowance_mm"))


def test_exact_unique_match_consumes_the_callers_initially_empty_used_set_once(helpers):
    spec, bend, used = selected(), extracted(), set()
    match = helpers["_match_selected_spec_to_extracted_bend"]
    assert match(spec, [bend], used_indices=used) is bend
    assert used == {0}
    assert match(spec, [bend], used_indices=used) is None
    assert used == {0}


def test_preused_indices_are_respected_without_replacing_or_clearing_the_caller_set(helpers):
    spec = selected()
    bends, used = [extracted("first"), extracted("second", center=(200.0, 0.0, 0.0))], {1}
    match = helpers["_match_selected_spec_to_extracted_bend"]
    assert match(spec, bends, used_indices=used) is bends[0]
    assert used == {0, 1}
    assert match(spec, bends, used_indices=used) is None
    assert used == {0, 1}


def test_single_helper_does_not_choose_the_closer_of_two_admissible_bends(helpers):
    used = set()
    bends = [extracted("near", center=(0.1, 0.0, 0.0)), extracted("also_near", center=(1.0, 0.0, 0.0))]
    assert helpers["_match_selected_spec_to_extracted_bend"](selected(), bends, used_indices=used) is None
    assert used == set()


def test_preused_alternative_cannot_rescue_an_ambiguous_single_match(helpers):
    bends = [extracted("first"), extracted("second", center=(1.0, 0.0, 0.0))]
    used = {1}
    assert helpers["_match_selected_spec_to_extracted_bend"](selected(), bends, used_indices=used) is None
    assert used == {1}


@pytest.mark.parametrize("case", ["million_mm", "distance_equality", "endpoint_equality", "angle_equality", "orthogonal_axis"])
def test_fixed_distance_angle_and_direction_gates_reject_without_consuming_any_index(helpers, case):
    spec = selected()
    if case == "million_mm":
        bend = extracted(center=(1_000_000.0, 0.0, 0.0))
    elif case == "distance_equality":
        bend = extracted(center=(45.0, 0.0, 0.0))
    elif case == "endpoint_equality":
        # Centers agree, but corresponding endpoints are exactly 45 mm apart.
        bend = extracted(length=100.0)
    elif case == "angle_equality":
        bend = extracted(angle=125.0)
    else:
        bend = extracted(direction=(0.0, 1.0, 0.0))
    assert helpers["_selected_cad_attribution_geometry"](spec, bend) is None
    used = {19}
    assert helpers["_match_selected_spec_to_extracted_bend"](spec, [bend], used_indices=used) is None
    assert used == {19}


@pytest.mark.parametrize("alignment,admitted", [(0.969, False), (0.971, True)])
def test_unoriented_direction_gate_has_declared_threshold_not_nearest_score(helpers, alignment, admitted):
    direction = (alignment, math.sqrt(1.0-alignment**2), 0.0)
    result = helpers["_selected_cad_attribution_geometry"](selected(), extracted(direction=direction))
    assert (result is not None) is admitted


def test_reversed_finite_endpoints_and_direction_are_the_same_unoriented_axis(helpers):
    spec, bend = selected(), extracted(direction=(-1.0, 0.0, 0.0))
    geometry = helpers["_selected_cad_attribution_geometry"](spec, bend)
    assert geometry is not None
    assert geometry["direction_alignment"] == 1.0
    assert geometry["endpoint_distance_mm"] == 0.0
    payloads, count = helpers["_build_selected_measurement_dicts"]([spec], [bend])
    assert count == 1
    assert_selected_geometry(payloads[0], spec)


@pytest.mark.parametrize("swap", [False, True])
def test_unique_normal_correspondence_reorders_only_local_surface_ids(helpers, swap):
    spec, bend = selected(), extracted(center=(0.25, 0.0, 0.0), angle=95.0)
    if swap:
        bend.surface1_normal, bend.surface2_normal = bend.surface2_normal, bend.surface1_normal
    pristine = copy.deepcopy(spec)
    payloads, count = helpers["_build_selected_measurement_dicts"]([spec], [bend])
    assert count == 1
    payload = payloads[0]
    assert_selected_geometry(payload, pristine)
    assert [payload["surface1_id"], payload["surface2_id"]] == ([12, 11] if swap else [11, 12])
    assert_attribution(payload, status="GEOMETRY_COMPATIBLE_UNIQUE",
                       reason="UNIQUE_ADMISSIBLE_MESH_SURFACE_PAIR", index=0)
    assert payload["mesh_surface_attribution"]["geometry"]["center_distance_mm"] == 0.25
    assert payload["mesh_surface_attribution"]["geometry"]["bend_angle_gap_degrees"] == 5.0


def test_parallel_flange_normals_cannot_create_an_arbitrary_direct_or_swapped_correspondence(helpers):
    spec, bend = selected(), extracted()
    spec.flange2_normal = spec.flange1_normal.copy()
    bend.surface2_normal = bend.surface1_normal.copy()
    assert helpers["_selected_cad_attribution_geometry"](spec, bend) is None
    payloads, count = helpers["_build_selected_measurement_dicts"]([spec], [bend])
    assert count == 0
    assert_selected_geometry(payloads[0], spec)
    assert_attribution(payloads[0], status="UNATTRIBUTED", reason="NO_ADMISSIBLE_EXTRACTED_BEND")


def test_matching_normals_with_an_inconsistent_flange_intersection_axis_are_rejected(helpers):
    spec, bend = selected(), extracted()
    # The direct surface correspondence is unique, but these flange normals
    # intersect along Y, perpendicular to the explicitly declared finite X axis.
    spec.flange1_normal = bend.surface1_normal = np.array([1.0, 0.0, 0.0])
    assert helpers["_selected_cad_attribution_geometry"](spec, bend) is None


@pytest.mark.parametrize("value", [[0.0, True, 0.0], [2**53+1, 0.0, 0.0]])
def test_vector_validation_rejects_mixed_entries_before_numpy_coercion(helpers, value):
    assert helpers["_selected_cad_attribution_vector"](value) is None
    bend = extracted()
    bend.surface1_normal = value
    assert helpers["_selected_cad_attribution_geometry"](selected(), bend) is None


@pytest.mark.parametrize("identifier", [None, "", True, -1, 2**53+1, "invalid identity", "x"*129])
def test_extracted_identity_must_be_explicit_bounded_and_unambiguous(helpers, identifier):
    bend = extracted(identifier)
    assert helpers["_selected_cad_extracted_identity"](bend) is None
    assert helpers["_selected_cad_attribution_geometry"](selected(), bend) is None


def test_integer_and_string_identity_namespaces_are_distinct(helpers):
    assert helpers["_selected_cad_extracted_identity"](extracted(1)) == ("integer", 1)
    assert helpers["_selected_cad_extracted_identity"](extracted("1")) == ("string", "1")


@pytest.mark.parametrize("side,field,value", [
    ("selected", "bend_line_start", [math.nan, 0.0, 0.0]),
    ("selected", "bend_line_end", [-5.0, 0.0, 0.0]),
    ("selected", "bend_line_direction", [0.0, 0.0, 0.0]),
    ("selected", "bend_line_direction", [0.0, 1.0, 0.0]),
    ("selected", "target_angle", -1.0), ("selected", "target_angle", 181.0),
    ("selected", "target_angle", True), ("selected", "flange1_normal", None),
    ("selected", "flange1_normal", [0.0, 0.0, 0.0]),
    ("selected", "flange1_normal", [0.0, math.inf, 0.0]),
    ("extracted", "bend_center", [math.nan, 0.0, 0.0]),
    ("extracted", "bend_center", [0.0, 0.0]),
    ("extracted", "bend_center", [1.0, 0.0, 0.0]),
    ("extracted", "bend_line_start", [2**53+1, 0, 0]),
    ("extracted", "bend_line_end", [-5.0, 0.0, 0.0]),
    ("extracted", "bend_direction", [0.0, 0.0, 0.0]),
    ("extracted", "bend_direction", [0.0, 1.0, 0.0]),
    ("extracted", "bend_angle", math.nan), ("extracted", "bend_angle", 181.0),
    ("extracted", "surface1_normal", [False, True, False]),
    ("extracted", "surface2_normal", [0.0, 0.0, 0.0]),
    ("extracted", "surface1_id", -1), ("extracted", "surface1_id", True),
    ("extracted", "surface1_id", 1.5), ("extracted", "surface1_id", 2**53+1),
    ("extracted", "surface2_id", 11),
])
def test_invalid_geometry_or_local_surface_identity_never_borrows_target_defaults(helpers, side, field, value):
    spec, bend = selected(), extracted()
    setattr(spec if side == "selected" else bend, field, value)
    assert helpers["_selected_cad_attribution_geometry"](spec, bend) is None
    used = set()
    assert helpers["_match_selected_spec_to_extracted_bend"](spec, [bend], used_indices=used) is None
    assert not used


@pytest.mark.parametrize("field", ["bend_center", "bend_direction", "bend_angle", "bend_line_start",
                                    "surface1_normal", "surface2_id"])
def test_missing_extracted_fields_are_not_synthesized_from_selected_cad(helpers, field):
    bend = extracted()
    delattr(bend, field)
    assert helpers["_selected_cad_attribution_geometry"](selected(), bend) is None


def test_subnormal_segments_and_overflowing_finite_vector_norms_are_not_attributable(helpers):
    eta = float(np.nextafter(0.0, 1.0))
    bend = extracted()
    bend.bend_line_start = np.array([1.0, 0.0, 0.0])
    bend.bend_line_end = np.array([1.0, 2*eta, 3*eta])
    assert helpers["_selected_cad_attribution_geometry"](selected(), bend) is None
    bend = extracted()
    bend.surface1_normal = np.array([1.7e308, 1.7e308, 1.7e308])
    assert helpers["_selected_cad_attribution_geometry"](selected(), bend) is None


@pytest.mark.parametrize("two_extracted", [False, True])
def test_many_to_one_and_many_to_many_batch_ambiguities_are_not_greedily_assigned(helpers, two_extracted):
    specs = [selected("selected_A"), selected("selected_B", center=(1.0, 0.0, 0.0))]
    bends = [extracted("extracted_A")]
    if two_extracted:
        bends.append(extracted("extracted_B", center=(1.0, 0.0, 0.0), surface_ids=(21, 22)))
    pairs, reasons = helpers["_selected_cad_attribution_pairs"](specs, bends)
    assert pairs == {}
    assert reasons == ["AMBIGUOUS_MESH_SURFACE_ATTRIBUTION"]*2
    payloads, count = helpers["_build_selected_measurement_dicts"](specs, bends)
    assert count == 0 and len(payloads) == 2
    for payload, spec in zip(payloads, specs):
        assert_selected_geometry(payload, spec)
        assert_attribution(payload, status="UNATTRIBUTED", reason="AMBIGUOUS_MESH_SURFACE_ATTRIBUTION")
    ids = [payload[key] for payload in payloads for key in ("surface1_id", "surface2_id")]
    assert len(ids) == len(set(ids)) == 4


def test_batch_does_not_rescue_conflicts_after_assignment_but_keeps_an_independent_singleton(helpers):
    specs = [selected("A", center=(0.0, 0.0, 0.0)), selected("B", center=(40.0, 0.0, 0.0)),
             selected("C", center=(200.0, 0.0, 0.0))]
    bends = [extracted("X", center=(0.0, 0.0, 0.0), surface_ids=(11, 12)),
             extracted("Y", center=(80.0, 0.0, 0.0), surface_ids=(21, 22)),
             extracted("Z", center=(200.0, 0.0, 0.0), surface_ids=(31, 32))]
    pairs, reasons = helpers["_selected_cad_attribution_pairs"](specs, bends)
    # A-X, B-X, B-Y is a conflict component. Removing A-X to rescue B-Y is
    # expressly disallowed; only the disconnected C-Z singleton is accepted.
    assert set(pairs) == {2} and pairs[2][0] == 2
    assert reasons == ["AMBIGUOUS_MESH_SURFACE_ATTRIBUTION"]*2+["UNIQUE_ADMISSIBLE_MESH_SURFACE_PAIR"]
    payloads, count = helpers["_build_selected_measurement_dicts"](specs, bends)
    assert count == 1
    for payload, spec in zip(payloads, specs):
        assert_selected_geometry(payload, spec)
    assert_attribution(payloads[2], status="GEOMETRY_COMPATIBLE_UNIQUE",
                       reason="UNIQUE_ADMISSIBLE_MESH_SURFACE_PAIR", index=2)


def test_unique_disjoint_pairings_are_invariant_to_selected_and_extracted_order(helpers):
    specs = [selected("A"), selected("B", center=(200.0, 0.0, 0.0))]
    bends = [extracted("X", surface_ids=(11, 12)),
             extracted("Y", center=(200.0, 0.0, 0.0), surface_ids=(21, 22))]
    expected = {"A": (11, 12), "B": (21, 22)}
    for selected_order, extracted_order in ((specs, bends), (specs[::-1], bends),
                                           (specs, bends[::-1]), (specs[::-1], bends[::-1])):
        payloads, count = helpers["_build_selected_measurement_dicts"](selected_order, extracted_order)
        assert count == 2
        assert {payload["bend_id"]: (payload["surface1_id"], payload["surface2_id"])
                for payload in payloads} == expected
        for payload, spec in zip(payloads, selected_order):
            assert_selected_geometry(payload, spec)
            index = next(i for i, bend in enumerate(extracted_order)
                         if bend.surface1_id == expected[spec.bend_id][0])
            assert_attribution(payload, status="GEOMETRY_COMPATIBLE_UNIQUE",
                               reason="UNIQUE_ADMISSIBLE_MESH_SURFACE_PAIR", index=index)


def test_empty_extraction_keeps_valid_selected_geometry_and_neutral_fallback_provenance(helpers):
    specs = [selected("A"), selected("B", center=(200.0, 0.0, 0.0))]
    payloads, count = helpers["_build_selected_measurement_dicts"](specs, [])
    assert count == 0 and len(payloads) == 2
    for payload, spec in zip(payloads, specs):
        assert_selected_geometry(payload, spec)
        assert_attribution(payload, status="UNATTRIBUTED", reason="NO_ADMISSIBLE_EXTRACTED_BEND")
    ids = [payload[key] for payload in payloads for key in ("surface1_id", "surface2_id")]
    assert len(ids) == len(set(ids)) == 4


def test_missing_extracted_normal_keeps_selected_normal_without_borrowing_it_for_attribution(helpers):
    spec, bend = selected(), extracted()
    bend.surface1_normal = None
    payloads, count = helpers["_build_selected_measurement_dicts"]([spec], [bend])
    assert count == 0
    assert_selected_geometry(payloads[0], spec)
    assert_attribution(payloads[0], status="UNATTRIBUTED", reason="NO_ADMISSIBLE_EXTRACTED_BEND")


@pytest.mark.parametrize("field,value", [
    ("flange1_normal", None), ("flange2_normal", [0.0, 0.0, 0.0]),
    ("bend_line_start", [math.nan, 0.0, 0.0]),
    ("bend_line_end", [-5.0, 0.0, 0.0]),
    ("bend_line_direction", [0.0, 1.0, 0.0]),
    ("target_angle", True), ("target_angle", math.inf),
])
def test_entire_selected_batch_is_validated_before_attribution_or_default_payloads(helpers, monkeypatch, field, value):
    specs = [selected("valid_first"), selected("invalid_later")]
    setattr(specs[1], field, value)

    def forbidden_pairs(*args):
        raise AssertionError("all selected geometry must be validated before attribution")

    monkeypatch.setitem(helpers, "_selected_cad_attribution_pairs", forbidden_pairs)
    with pytest.raises(ValueError):
        helpers["_build_selected_measurement_dicts"](specs, [extracted()])


@pytest.mark.parametrize("matched", [False, True])
def test_selected_payload_geometry_is_a_snapshot_not_an_alias(helpers, matched):
    spec = selected()
    pristine = copy.deepcopy(spec)
    payloads, count = helpers["_build_selected_measurement_dicts"]([spec], [extracted()] if matched else [])
    assert count == int(matched)
    payload = payloads[0]
    for output, source in (("bend_line_start", "bend_line_start"), ("bend_line_end", "bend_line_end"),
                           ("bend_direction", "bend_line_direction"), ("surface1_normal", "flange1_normal"),
                           ("surface2_normal", "flange2_normal")):
        assert not np.shares_memory(payload[output], getattr(spec, source))
        getattr(spec, source)[:] = 999.0
    assert_selected_geometry(payload, pristine)
    assert_attribution(payload,
                       status="GEOMETRY_COMPATIBLE_UNIQUE" if matched else "UNATTRIBUTED",
                       reason="UNIQUE_ADMISSIBLE_MESH_SURFACE_PAIR" if matched else "NO_ADMISSIBLE_EXTRACTED_BEND",
                       index=0 if matched else None)


def test_duplicate_extracted_identity_blocks_geometrically_disjoint_rows_but_not_an_independent_identity(helpers):
    specs = [selected("A"), selected("B", center=(200.0, 0.0, 0.0)),
             selected("C", center=(400.0, 0.0, 0.0))]
    bends = [extracted("duplicate", surface_ids=(11, 12)),
             extracted("duplicate", center=(200.0, 0.0, 0.0), surface_ids=(21, 22)),
             extracted("independent", center=(400.0, 0.0, 0.0), surface_ids=(31, 32))]
    for spec, bend in zip(specs, bends):
        assert helpers["_selected_cad_attribution_geometry"](spec, bend) is not None
    pairs, reasons = helpers["_selected_cad_attribution_pairs"](specs, bends)
    assert set(pairs) == {2} and pairs[2][0] == 2
    assert reasons == ["DUPLICATE_EXTRACTED_BEND_IDENTITY"]*2+["UNIQUE_ADMISSIBLE_MESH_SURFACE_PAIR"]
    payloads, count = helpers["_build_selected_measurement_dicts"](specs, bends)
    assert count == 1
    for index, (payload, spec) in enumerate(zip(payloads, specs)):
        assert_selected_geometry(payload, spec)
        assert_attribution(payload, status="GEOMETRY_COMPATIBLE_UNIQUE" if index == 2 else "UNATTRIBUTED",
                           reason=reasons[index], index=2 if index == 2 else None)
    # The far duplicate has no admissible geometric edge to A, but still blocks
    # A's otherwise unique source identity, including in the one-spec helper.
    used = {1}
    assert helpers["_match_selected_spec_to_extracted_bend"](specs[0], bends, used_indices=used) is None
    assert used == {1}


@pytest.mark.parametrize("oversized_side", ["selected", "extracted"])
def test_pair_budget_preflight_preserves_all_selected_fallbacks_without_geometry_work(helpers, oversized_side, monkeypatch):
    specs = [selected(f"S{index}") for index in range(257 if oversized_side == "selected" else 1)]
    bends = [extracted(f"E{index}") for index in range(257 if oversized_side == "extracted" else 1)]

    def forbidden_geometry(*args):
        raise AssertionError("pair cap must be checked before geometry evaluation")

    monkeypatch.setitem(helpers, "_selected_cad_attribution_geometry", forbidden_geometry)
    pairs, reasons = helpers["_selected_cad_attribution_pairs"](specs, bends)
    assert pairs == {} and reasons == ["ATTRIBUTION_PAIR_BUDGET_EXCEEDED"]*len(specs)
    payloads, count = helpers["_build_selected_measurement_dicts"](specs, bends)
    assert count == 0 and len(payloads) == len(specs)
    for payload, spec in zip(payloads, specs):
        assert_selected_geometry(payload, spec)
        assert_attribution(payload, status="UNATTRIBUTED", reason="ATTRIBUTION_PAIR_BUDGET_EXCEEDED")


def test_empty_selected_batch_cannot_become_an_extracted_or_observed_bend_count(helpers):
    payloads, count = helpers["_build_selected_measurement_dicts"]([], [extracted()])
    assert payloads == [] and count == 0
