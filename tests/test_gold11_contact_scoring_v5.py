"""V3 contact projection tests: the V2 invented-payload matrix upgraded to the V9 grammar (cover V3,
context V2, separately bound layer policy, per-carrier declared material-side identity), plus
identity-specific checks. The matcher/thresholds are the unchanged V2 text.

Prospective invented-only tests; no inference, native, raw or artifact access.

Workflow-shaped dictionaries below are structural fixtures, not claims that any
engine geometry fit or raw-support proof occurred. Only the supplied JSON bytes
are hashed. The eleven generic cases, six groups and seventy-one straight-line
references are generated here and are not a physical cohort or control replay.
"""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scoring():
    # No scripts/__init__ or backend package initialization. Only this pure
    # module and its three named legacy helper dependencies are imported.
    name = "_invented_gold11_contact_v2"
    package = ModuleType(name)
    package.__path__ = [str(ROOT / "scripts")]
    prior = {key: value for key, value in sys.modules.items() if key == name or key.startswith(name + ".")}
    sys.modules[name] = package
    try:
        yield importlib.import_module(name + ".gold11_contact_scoring_v5")
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        sys.modules.update(prior)


def _configuration():
    # Independent literal descriptor, rather than copying the adapter's caps.
    return dict(mode="composite_v1", seeded_budgets=dict(max_seeds=32,
        max_seed_selection_triangle_visits=64_000_000, max_proposal_raw_row_visits=32_000_000,
        max_proposal_triangle_visits=64_000_000, max_proposal_edge_visits=96_000_000,
        max_full_support_row_tests=32_000_000, max_triangle_memberships=20_000_000,
        max_raw_memberships=32_000_000, max_fit_evaluations=32,
        max_retained_array_bytes=268_435_456, max_metadata_bytes=67_108_864,
        max_working_array_bytes=1_073_741_824), composite_budgets=dict(max_qualified_planes=65_552,
        max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
        max_full_support_row_tests=64_000_000, max_raw_predicate_visits=48_000_000,
        max_triangle_predicate_visits=160_000_000, max_edge_predicate_visits=192_000_000,
        max_fit_evaluations=65_584, max_retained_array_bytes=536_870_912,
        max_metadata_bytes=67_108_864))


def _identity(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _instance(identity, index):
    x = index * 20
    return dict(bend_id=identity, feature_id=identity,
        line_start=[x, 0, 0], line_end=[x, 0, 10],
        axis_support_line_start=[x, -5, 0], axis_support_line_end=[x, -5, 10],
        line_convention="MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR", unit="mm",
        geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        endpoint_semantics="SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
        observation_status="SAMPLE_SUPPORTED_CONDITIONAL", continuous_surface_coverage="NOT_ESTABLISHED",
        physical_trim_endpoints=None, manufacturing_event_identity="UNKNOWN",
        **IDENTITY_ELIGIBLE,
        reference_carrier_compatibility="UNVERIFIED_NO_GOLD_IN_CARRIER_SELECTION",
        # Four axial stations and nineteen angular stations permit a 90-degree
        # sector with 5-degree gaps. Adjacent slabs share their boundary station.
        axial_support_length_mm=10, unique_arc_support_count=76,
        unique_flange_support_counts=[16, 16], max_axial_sample_gap_mm=10 / 3,
        local_coverage=[dict(slab_index=i, arc_points=38, flange_points=[8, 8],
            arc_sweep_deg=90., max_angular_gap_deg=5.) for i in range(3)])


def _fit_policy():
    return dict(max_fit_residual_mm=.05, max_tangency_error_mm=.1, min_arc_sweep_deg=20.,
        max_circle_condition=1000., min_plane_spread_ratio=.01, max_normal_deviation_deg=15., fit_model="free_circle")


def _layer_policy():
    return dict(enabled=True, layer_split_gap_mm=.3, min_layer_rows=12, max_layers_reported=8,
        max_layer_normal_angle_deg=3., max_layer_gap_mm=6., min_projected_interval_overlap=.5, max_layer_rms_mm=.1, layer_mode_bin_mm=.05, layer_membership_band_mm=.1, max_bridging_row_fraction=.35, layer_refit_iterations=3, layer_mode_ambiguity_fraction=.25)


# The two-layer selected status is a side hypothesis, never eligible: no present status is.
IDENTITY_ELIGIBLE = dict(material_side_status="TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED",
    inner_face_claim_eligible=False, carrier_role="GENERIC_CONDITIONAL_CONTACT_LINE_SURFACE_IDENTITY_UNRESOLVED_NOT_INNER_ARC")
IDENTITY_UNRESOLVED = dict(material_side_status="SINGLE_LAYER_IDENTITY_UNRESOLVED",
    inner_face_claim_eligible=False, carrier_role="GENERIC_CONDITIONAL_CONTACT_LINE_SURFACE_IDENTITY_UNRESOLVED_NOT_INNER_ARC")


def _workflow(count=1, *, mode="v7", finite=True, label="invented"):
    groups, instances, source_records, plans = [], [], [], []
    for index in range(count):
        identity, member = _identity(label + "-group-" + str(index)), _identity(label + "-member-" + str(index))
        instance = _instance(identity, index)
        # Numerically self-consistent serialization of a radius-5 generator
        # parallel to the canonical +Z axis. Residuals and conditioning are
        # explicit invented reports, not a claim that a raw fit was executed.
        quantity = dict(fit_model="free_circle", status="CONDITIONAL_SCAN_POINT_ESTIMATE",
            geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM", radius_uncertainty_interval_mm=None,
            sweep_uncertainty_interval_deg=None, physical_trim_endpoints=None, datum_relative_deviation=None,
            **IDENTITY_ELIGIBLE,
            sweep_sign_convention="CANONICAL_RAW_AXIS_AND_RAW_ROLE_ORDER",
            measured_radius_mm=5., measured_sweep_magnitude_deg=90., measured_sweep_deg=90.,
            measured_axis_direction=[0., 0., 1.], measured_axis_point=[index * 20., -5., 5.],
            circle_max_residual_mm=0., flange_max_residual_mm=[0., 0.],
            measured_tangency_residual_mm=[0., 0.], free_circle_jacobian_condition=25.,
            free_circle_min_singular_per_sqrt_n=.055, tangency_check_role="FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK")
        shared = dict(group_id=identity, member_record_ids=[member], cover_record_id=member,
            member_flange_permutations={member: [0, 1]})
        groups.append(dict(shared, status="MEASURED_CONDITIONAL", reasons=[], measurements=quantity, fit_policy=_fit_policy(),
            layer_policy=_layer_policy(), **IDENTITY_ELIGIBLE,
            finite_support=dict(status="SUPPORTED" if finite else "UNAVAILABLE",
                reasons=[] if finite else ["INVENTED_FINITE_SUPPORT_REFUSAL"], instance=instance if finite else None)))
        plans.append(dict(copy.deepcopy(shared), status="COVERED", reasons=[]))
        source_records.append(dict(record_id=member, fit_policy=_fit_policy(), layer_policy=_layer_policy(), **IDENTITY_ELIGIBLE))
        if finite:
            instances.append(copy.deepcopy(instance))
    status = "PARTIAL" if count else "UNAVAILABLE"
    measured = dict(schema="CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V5", status=status,
        computation_status="COMPLETE", predictions=copy.deepcopy(instances), support_groups=groups,
        conditional_support_group_count=len(groups), support_cover_plan=dict(status="COMPLETE", groups=plans),
        source_context_result=dict(schema="CAD_GUIDED_GENERATED_CONTEXT_MEASUREMENTS_V4",
            kernel_implementation="CAD_GUIDED_MEASUREMENT_V3_TWO_MODE_LAYER_SEPARATION_AND_DIAGNOSTICS",
            layer_policy=_layer_policy(), computation_status="COMPLETE", support_records=source_records,
            context_windows=[dict(context_index=0) for _ in groups], predictions=[]),
        observed_bend_count=None, observed_bend_count_status="MANUFACTURING_EVENT_IDENTITY_UNKNOWN",
        production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED_NO_PROPAGATED_MEASUREMENT_INTERVAL",
        geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        endpoint_semantics="SAMPLED_WINDOW_SUPPORT_NOT_PHYSICAL_TRIM",
        context_quantifier="EXISTS_COMPLETED_EXISTING_COVER_CONTEXT",
        joint_configuration_status="NOT_ESTABLISHED", continuous_pose_family_coverage="NOT_ESTABLISHED",
        registration_status="NOT_ACCEPTED_CONTEXT_PROPOSALS_ONLY",
        source_context_result_role="RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS",
        aggregation_policy="NARY_EXISTING_EXACT_RAW_SUPPORT_COVER",
        member_geometry_policy="COMMON_COVER_MODEL_WITNESS_NOT_INDEPENDENT_MEMBER_REFIT_CLAIM",
        finite_support_policy="COVER_ONLY_MEMBER_FINITE_REFUSAL_DOES_NOT_REMOVE_MEMBER",
        prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS")
    result = dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status=status, phase="COMPLETE",
        reasons=[] if count else ["INVENTED_NO_MEASUREMENT"], predictions=copy.deepcopy(instances),
        measurements=measured, connected_measurement_contexts=True,
        prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS",
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE", policies={},
        registration=dict(status="CONDITIONAL_MEASUREMENT_CONTEXTS", transform_scan_to_cad=None,
            evidence=dict(candidate_evaluation_complete=True, accepted_registration=False, joint_configuration_status="NOT_ESTABLISHED",
                measurement_context_indices=[0], initialization={})),
        support_cover_publication_validation=dict(cover_plan_revalidated=True, cover_refit_repeated=False))
    if mode == "composite_v1":
        for parent in _configuration_parents(result):
            parent["scan_plane_configuration"] = _configuration()
    return result


def _configuration_parents(payload):
    return [payload["policies"], payload["registration"]["evidence"],
            payload["registration"]["evidence"]["initialization"]]


def _instances(payload, index=-1):
    return [payload["predictions"][index], payload["measurements"]["predictions"][index],
        payload["measurements"]["support_groups"][index]["finite_support"]["instance"]]


def _wire(payload, **kwargs):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, **kwargs).encode("utf-8")


def _project(scoring, payload, *, mode="v7", **kwargs):
    blob = _wire(payload)
    return scoring.project_gold11_contact_workflow_v5(blob, expected_result_sha256=hashlib.sha256(blob).hexdigest(),
        expected_scan_plane_mode=mode,
        expected_scan_plane_configuration=_configuration() if mode == "composite_v1" else None, **kwargs)


def _refusal(phase="ADMISSION"):
    return dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status="UNAVAILABLE", phase=phase,
        reasons=["INVENTED_EXPLICIT_REFUSAL"], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE")


@pytest.fixture
def paired(scoring):
    cases = ["toy-case-" + str(i).zfill(2) for i in range(11)]
    families = {case: "toy-family-" + str(i % 6) for i, case in enumerate(cases)}
    references, control, challenger = {}, {}, {}
    for index, case in enumerate(cases):
        count = 7 if index < 5 else 6
        references[case] = [dict(bend_id="reference-" + str(j),
            line_start=[j * 20, 0, 0], line_end=[j * 20, 0, 10]) for j in range(count)]
        control[case] = _project(scoring, _workflow(1, label=case))
        challenger[case] = _project(scoring, _workflow(count, mode="composite_v1", label=case), mode="composite_v1")
    return dict(reference_records_by_case=references, control_results_by_case=control,
        challenger_results_by_case=challenger, case_family_by_id=families,
        scan_plane_requests_by_arm=dict(control=dict(mode="v7", configuration=None),
            challenger=dict(mode="composite_v1", configuration=_configuration())))


def test_exact_legacy_function_bindings_and_descriptor_literals(scoring):
    assert scoring.match_lines is scoring._legacy.match_lines
    assert scoring.Thresholds is scoring._legacy.Thresholds
    assert scoring.score_localized_cases is scoring._legacy.score_localized_cases
    assert _configuration()["seeded_budgets"] == scoring.SEEDED_BUDGET_MAXIMA
    assert _configuration()["composite_budgets"] == scoring.COMPOSITE_BUDGET_MAXIMA


@pytest.mark.parametrize("mode", ["v7", "composite_v1"])
def test_literal_contact_projection_and_exact_byte_binding(scoring, mode):
    payload = _workflow(2, mode=mode)
    wire = _wire(payload)
    result = _project(scoring, payload, mode=mode)
    assert result["disposition"] == "SCORED_CONDITIONAL"
    assert result["result_binding"]["supplied_sha256"] == hashlib.sha256(wire).hexdigest()
    assert result["result_binding"]["supplied_bytes"] == len(wire)
    assert result["result_binding"]["matched"] is True
    assert "NOT_EXTERNAL_OR_RAW_ATTESTATION" in result["result_binding"]["authority"]
    for original, projected in zip(payload["predictions"], result["predictions"]):
        assert projected["line_start"] == original["line_start"]
        assert projected["line_end"] == original["line_end"]
        assert all(type(v) is int for v in projected["line_start"])
        assert projected["line_convention"] == "MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR"
        assert projected["geometry_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
        assert "reference_frame" not in projected
        assert not {"axis_support_line_start", "axis_support_line_end", "local_coverage"}.intersection(projected)
        assert projected["physical_trim_endpoints"] is None
        assert projected["manufacturing_event_identity"] == "UNKNOWN"
    assert result["diagnostics"]["emitted_carrier_count"] == 2
    assert result["observed_bend_count"] is None
    assert result["production_use_allowed"] is False
    returned = result["returned_scan_plane_configurations"]
    assert all(row["present"] is (mode == "composite_v1") for row in returned.values())


@pytest.mark.parametrize("location,operation", [(0, "missing"), (0, "different"),
    (1, "missing"), (1, "different"), (2, "missing"), (2, "different")])
def test_completed_composite_requires_all_three_exact_configurations(scoring, location, operation):
    payload = _workflow(mode="composite_v1")
    parent = _configuration_parents(payload)[location]
    if operation == "missing":
        del parent["scan_plane_configuration"]
    else:
        parent["scan_plane_configuration"]["seeded_budgets"]["max_seeds"] = 31
    result = _project(scoring, payload, mode="composite_v1")
    assert result["disposition"] == "MALFORMED_RESULT"
    assert result["predictions"] == []
    assert result["returned_scan_plane_configurations"] is None


@pytest.mark.parametrize("location", [0, 1, 2])
def test_v7_rejects_even_null_returned_composite_fields(scoring, location):
    payload = _workflow()
    _configuration_parents(payload)[location]["scan_plane_configuration"] = None
    assert _project(scoring, payload)["disposition"] == "MALFORMED_RESULT"


@pytest.mark.parametrize("problem", ["unknown-mode", "v7-extra", "composite-missing", "bool-budget", "over-budget", "extra-budget"])
def test_invalid_caller_request_raises_before_case_attribution(scoring, problem):
    mode, configuration = "composite_v1", _configuration()
    if problem == "unknown-mode":
        mode = "v8"
    elif problem == "v7-extra":
        mode = "v7"
    elif problem == "composite-missing":
        configuration = None
    elif problem == "bool-budget":
        configuration["seeded_budgets"]["max_seeds"] = True
    elif problem == "over-budget":
        configuration["seeded_budgets"]["max_seeds"] = 33
    else:
        configuration["seeded_budgets"]["unexpected"] = 0
    with pytest.raises(ValueError):
        scoring.project_gold11_contact_workflow_v5(b"{}", expected_result_sha256=_identity("{}"),
            expected_scan_plane_mode=mode, expected_scan_plane_configuration=configuration)


@pytest.mark.parametrize("problem", ["duplicate-key", "NaN", "Infinity", "float-overflow", "bad-utf8", "truncated", "array-root", "depth"])
def test_all_result_json_including_unused_later_values_is_strict(scoring, problem):
    blob = _wire(_workflow(2))
    if problem == "duplicate-key":
        blob = blob[:-1] + b',"unused":{"duplicate":0,"duplicate":1}}'
    elif problem in ("NaN", "Infinity", "float-overflow"):
        value = {"NaN": b"NaN", "Infinity": b"Infinity", "float-overflow": b"1e999"}[problem]
        blob = blob[:-1] + b',"unused_later_value":' + value + b"}"
    elif problem == "bad-utf8":
        blob = b"\xff"
    elif problem == "truncated":
        blob = blob[:-1]
    elif problem == "array-root":
        blob = b"[]"
    else:
        blob = blob[:-1] + b',"unused_later_value":' + b"[" * 66 + b"0" + b"]" * 66 + b"}"
    result = scoring.project_gold11_contact_workflow_v5(blob,
        expected_result_sha256=hashlib.sha256(blob).hexdigest(), expected_scan_plane_mode="v7")
    assert result["disposition"] == "MALFORMED_RESULT"
    assert result["predictions"] == []
    assert all(v is None for v in result["diagnostics"].values())


@pytest.mark.parametrize("field,value", [
    ("line_convention", "MEASURED_INNER_ARC_MID_ANGLE_GENERATOR"),
    ("geometry_frame", "CAD_FRAME"), ("unit", "inch"),
    ("material_side_status", "VERIFIED_INNER"), ("physical_trim_endpoints", []),
    ("manufacturing_event_identity", "KNOWN"), ("continuous_surface_coverage", "ESTABLISHED"),
    ("reference_carrier_compatibility", "VERIFIED"), ("line_start", [True, 0, 0]),
    ("line_end", [20, 0, 0]), ("unique_arc_support_count", False),
    ("span_endpoints", [[20, 0, 0], [20, 0, 10]]), ("anchor", [20, 0, 0]),
    ("axis_direction", [0, 0, 1]), ("span_mm", [0, 10]),
])
def test_malformed_later_carrier_invalidates_whole_case_even_if_all_three_copies_agree(scoring, field, value):
    payload = _workflow(2)
    for instance in _instances(payload):
        instance[field] = copy.deepcopy(value)
    result = _project(scoring, payload)
    assert result["disposition"] == "MALFORMED_RESULT"
    assert result["predictions"] == []
    assert result["diagnostics"]["emitted_carrier_count"] is None


@pytest.mark.parametrize("problem", ["top-missing", "v2-missing", "top-reordered", "duplicate-group", "false-permutation", "validation-absent", "known-count", "discarded-source-prefix"])
def test_whole_v2_projection_and_structural_denominators(scoring, problem):
    payload = _workflow(2)
    measured = payload["measurements"]
    if problem == "top-missing":
        payload["predictions"].pop()
    elif problem == "v2-missing":
        measured["predictions"].pop()
    elif problem == "top-reordered":
        payload["predictions"].reverse()
    elif problem == "duplicate-group":
        measured["support_groups"][1] = copy.deepcopy(measured["support_groups"][0])
        measured["support_cover_plan"]["groups"][1] = copy.deepcopy(measured["support_cover_plan"]["groups"][0])
    elif problem == "false-permutation":
        group = measured["support_groups"][-1]
        member = group["member_record_ids"][0]
        group["member_flange_permutations"][member] = [False, True]
        measured["support_cover_plan"]["groups"][-1]["member_flange_permutations"][member] = [False, True]
    elif problem == "validation-absent":
        del payload["support_cover_publication_validation"]
    elif problem == "discarded-source-prefix":
        measured["source_context_result"]["discarded_measurement_prefix"] = True
    else:
        payload["observed_bend_count"] = 2
    assert _project(scoring, payload)["disposition"] == "MALFORMED_RESULT"


def test_nested_v1_carriers_are_evidence_only_and_not_compact_output(scoring):
    payload = _workflow()
    nested = _instance(_identity("additional-nested-evidence"), 9)
    payload["measurements"]["source_context_result"]["predictions"] = [nested]
    result = _project(scoring, payload)
    assert result["disposition"] == "SCORED_CONDITIONAL"
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["bend_id"] != nested["bend_id"]
    assert "source_context_result" not in result


@pytest.mark.parametrize("phase", ["ADMISSION", "REPORT_PUBLICATION", "MEASUREMENT_REFUSAL"])
def test_explicit_refusal_before_configuration_is_not_malformed(scoring, phase):
    result = _project(scoring, _refusal(phase), mode="composite_v1")
    assert result["disposition"] == "REFUSED"
    assert result["predictions"] == []
    assert all(row == dict(present=False, configuration=None)
        for row in result["returned_scan_plane_configurations"].values())
    assert all(v is None for v in result["diagnostics"].values())


def test_complete_measurements_without_finite_carrier_are_explicit(scoring):
    result = _project(scoring, _workflow(2, finite=False))
    assert result["disposition"] == "COMPLETE_NO_LOCALIZED_CARRIERS"
    assert result["reported_workflow_status"] == "PARTIAL"
    assert result["predictions"] == []
    assert result["diagnostics"]["measured_group_count"] == result["diagnostics"]["finite_refused_group_count"] == 2
    assert result["diagnostics"]["emitted_carrier_count"] == 0


def test_complete_zero_measurement_refusal_has_known_structural_zero_counts(scoring):
    result = _project(scoring, _workflow(0))
    assert result["disposition"] == "REFUSED"
    assert result["reported_workflow_phase"] == "COMPLETE"
    assert result["diagnostics"]["support_group_count"] == result["diagnostics"]["emitted_carrier_count"] == 0
    assert result["diagnostics"]["registration_context_count"] == 1


@pytest.mark.parametrize("problem", ["missing-radius", "missing-axis", "missing-residual", "missing-fit-policy",
    "bool-radius", "string-radius", "sweep-contradiction", "zero-axis", "nonunit-axis", "negative-canonical-axis",
    "circle-residual", "flange-residual", "tangency-residual", "condition", "zero-singular-value",
    "tangency-role", "witness-policy-mismatch"])
def test_later_quantity_requires_typed_numeric_v2_serialization(scoring, problem):
    payload = _workflow(2)
    group = payload["measurements"]["support_groups"][-1]
    quantity = group["measurements"]
    if problem.startswith("missing-"):
        key = {"missing-radius": "measured_radius_mm", "missing-axis": "measured_axis_direction",
            "missing-residual": "circle_max_residual_mm"}.get(problem)
        if key is None:
            del group["fit_policy"]
        else:
            del quantity[key]
    elif problem == "bool-radius":
        quantity["measured_radius_mm"] = True
    elif problem == "string-radius":
        quantity["measured_radius_mm"] = "5"
    elif problem == "sweep-contradiction":
        quantity["measured_sweep_deg"] = 89.
    elif problem == "zero-axis":
        quantity["measured_axis_direction"] = [0., 0., 0.]
    elif problem == "nonunit-axis":
        quantity["measured_axis_direction"] = [0., 0., 2.]
    elif problem == "negative-canonical-axis":
        quantity["measured_axis_direction"] = [0., 0., -1.]
    elif problem == "circle-residual":
        quantity["circle_max_residual_mm"] = .051
    elif problem == "flange-residual":
        quantity["flange_max_residual_mm"] = [0., -.001]
    elif problem == "tangency-residual":
        quantity["measured_tangency_residual_mm"] = [0., .101]
    elif problem == "condition":
        quantity["free_circle_jacobian_condition"] = 1001.
    elif problem == "zero-singular-value":
        quantity["free_circle_min_singular_per_sqrt_n"] = 0.
    elif problem == "tangency-role":
        quantity["tangency_check_role"] = "CALIBRATED_ACCURACY"
    else:
        group["fit_policy"]["max_fit_residual_mm"] = .06
    result = _project(scoring, payload)
    assert result["disposition"] == "MALFORMED_RESULT"
    assert result["predictions"] == []
    assert all(value is None for value in result["diagnostics"].values())


@pytest.mark.parametrize("problem", ["degenerate-axis", "reported-length", "axis-length", "nonconstant-offset",
    "wrong-radius-offset", "axial-offset", "axis-off-center", "reported-radius", "reversed-carrier"])
def test_self_agreeing_later_copies_cannot_hide_carrier_axis_radius_contradictions(scoring, problem):
    payload = _workflow(2)
    quantity = payload["measurements"]["support_groups"][-1]["measurements"]
    for instance in _instances(payload):
        if problem == "degenerate-axis":
            instance["axis_support_line_end"] = list(instance["axis_support_line_start"])
        elif problem == "reported-length":
            instance["axial_support_length_mm"] = 9.
        elif problem == "axis-length":
            instance["axis_support_line_end"][2] = 9.
        elif problem == "nonconstant-offset":
            instance["line_end"][1] += .01
        elif problem == "wrong-radius-offset":
            instance["line_start"][1] += .01
            instance["line_end"][1] += .01
        elif problem == "axial-offset":
            instance["line_start"][2] += .01
            instance["line_end"][2] += .01
        elif problem == "axis-off-center":
            for key in ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end"):
                instance[key][0] += .01
        elif problem == "reversed-carrier":
            instance["line_start"], instance["line_end"] = instance["line_end"], instance["line_start"]
    if problem == "reported-radius":
        quantity["measured_radius_mm"] = 5.01
    result = _project(scoring, payload)
    assert result["disposition"] == "MALFORMED_RESULT"
    assert result["predictions"] == []
    assert all(value is None for value in result["diagnostics"].values())


@pytest.mark.parametrize("factor,expected", [(.25, "SCORED_CONDITIONAL"), (4., "MALFORMED_RESULT")])
def test_existing_v2_serialization_guard_is_not_replaced_by_a_physical_tolerance(scoring, factor, expected):
    payload = _workflow()
    # Here V2's serialization scale is exactly 10, determined by the endpoints.
    guard = 512 * (2.**-52) * 10
    for instance in _instances(payload):
        instance["line_start"][1] += factor * guard
        instance["line_end"][1] += factor * guard
    result = _project(scoring, payload)
    assert result["disposition"] == expected
    if expected == "SCORED_CONDITIONAL":
        assert result["predictions"][0]["line_start"][1] == factor * guard
    else:
        assert result["predictions"] == []


def test_json_structure_checks_disclose_external_parser_resource_requirement(scoring):
    assert "node/depth checks follow" in scoring.__doc__
    assert "not preallocation, RAM or CPU" in scoring.__doc__
    assert "externally enforced" in scoring.__doc__


@pytest.mark.parametrize("problem", ["result-bytes", "compact-bytes", "groups", "predictions"])
def test_declared_budgets_refuse_without_successful_prefix(scoring, problem):
    payload = _workflow(32 if problem == "compact-bytes" else 1)
    kwargs = {}
    if problem == "result-bytes":
        kwargs["max_result_bytes"] = len(_wire(payload)) - 1
    elif problem == "compact-bytes":
        kwargs["max_projection_bytes"] = 8192
    elif problem == "groups":
        payload["measurements"]["support_groups"] *= 513
    else:
        payload["predictions"] *= 513
    result = _project(scoring, payload, **kwargs)
    assert result["disposition"] == "MALFORMED_RESULT"
    assert result["predictions"] == []
    assert len(_wire(result)) <= result["limits"]["projection_bytes"]


def test_hash_binds_exact_bytes_not_reserialized_equivalent_json(scoring):
    payload = _workflow()
    compact = _wire(payload)
    spaced = json.dumps(payload, indent=1).encode("utf-8")
    mismatch = scoring.project_gold11_contact_workflow_v5(spaced,
        expected_result_sha256=hashlib.sha256(compact).hexdigest(), expected_scan_plane_mode="v7")
    assert mismatch["disposition"] == "MALFORMED_RESULT"
    assert mismatch["result_binding"]["matched"] is False
    matched = scoring.project_gold11_contact_workflow_v5(spaced,
        expected_result_sha256=hashlib.sha256(spaced).hexdigest(), expected_scan_plane_mode="v7")
    assert matched["disposition"] == "SCORED_CONDITIONAL"
    assert matched["predictions"] == _project(scoring, payload)["predictions"]


@pytest.mark.parametrize("bad_sha", [None, "0" * 63, "A" * 64, True])
def test_invalid_caller_sha_is_not_an_engine_failure(scoring, bad_sha):
    with pytest.raises(ValueError, match="INVALID_CALLER_RESULT_SHA256"):
        scoring.project_gold11_contact_workflow_v5(b"{}", expected_result_sha256=bad_sha, expected_scan_plane_mode="v7")


def test_no_refusal_or_incomplete_status_can_keep_predictions(scoring):
    payload = _workflow()
    payload["status"], payload["phase"] = "UNAVAILABLE", "NATIVE_FEATURES"
    result = _project(scoring, payload)
    assert result["disposition"] == "MALFORMED_RESULT" and result["predictions"] == []


def test_unchanged_matcher_and_metric_differential_for_all_twenty_two_invented_cases(scoring, paired):
    report = scoring.score_gold11_contact_workflows_v5(**paired)
    assert (report["case_count"], report["family_count"], report["reference_line_count"]) == (11, 6, 71)
    for arm in ("control", "challenger"):
        matched_rows = []
        for case, projection in paired[arm + "_results_by_case"].items():
            direct = [{key: row[key] for key in ("bend_id", "line_start", "line_end")}
                for row in projection["predictions"]]
            expected = scoring._legacy.match_lines(paired["reference_records_by_case"][case], direct,
                scoring._legacy.Thresholds(5., 4., .60, .60))
            actual = report[arm]["case_dispositions"][case]
            assert {key: actual[key] for key in expected} == expected
            matched_rows.append(dict(expected, case_id=case, family_id=paired["case_family_by_id"][case]))
        assert report[arm]["geometry"] == scoring._legacy._summary(matched_rows)
        assert report[arm]["physical_observed_bend_count"]["known_case_count"] == 0
        assert report[arm]["physical_observed_bend_count"]["unknown_case_count"] == 11
    assert report["control"]["geometry"]["true_positive"] == 11
    assert report["challenger"]["geometry"]["true_positive"] == 71
    assert report["challenger"]["geometry"]["false_negative"] == 0
    assert report["observed_bend_count"] is None
    assert report["strict_physical_engine_targets_met"] is False
    assert report["historical_score_beaten_claim_allowed"] is False


@pytest.mark.parametrize("start,end", [
    ([0, 4, 0], [0, 4, 10]), ([0, 4.000001, 0], [0, 4.000001, 10]),
    ([0, 0, 4], [0, 0, 14]), ([0, 0, 4.000001], [0, 0, 14.000001]),
])
def test_endpoint_boundaries_use_self_consistent_serialization_and_unchanged_matcher(scoring, paired, start, end):
    case = sorted(paired["case_family_by_id"])[0]
    payload = _workflow(1, mode="composite_v1", label=case)
    for instance in _instances(payload):
        instance["line_start"], instance["line_end"] = list(start), list(end)
        instance["axis_support_line_start"] = [start[0], start[1] - 5, start[2]]
        instance["axis_support_line_end"] = [end[0], end[1] - 5, end[2]]
    payload["measurements"]["support_groups"][0]["measurements"]["measured_axis_point"] = [
        start[0], start[1] - 5, (start[2] + end[2]) / 2]
    projection = _project(scoring, payload, mode="composite_v1")
    assert projection["disposition"] == "SCORED_CONDITIONAL"
    paired["challenger_results_by_case"][case] = projection
    report = scoring.score_gold11_contact_workflows_v5(**paired)
    expected = scoring.match_lines(paired["reference_records_by_case"][case],
        [dict(bend_id=projection["predictions"][0]["bend_id"], line_start=start, line_end=end)],
        scoring.Thresholds(5., 4., .60, .60))
    actual = report["challenger"]["case_dispositions"][case]
    assert {key: actual[key] for key in expected} == expected


def test_reference_endpoint_reversal_keeps_unchanged_undirected_matching(scoring, paired):
    case = sorted(paired["case_family_by_id"])[0]
    before = scoring.score_gold11_contact_workflows_v5(**paired)
    reference = paired["reference_records_by_case"][case][0]
    reference["line_start"], reference["line_end"] = reference["line_end"], reference["line_start"]
    after = scoring.score_gold11_contact_workflows_v5(**paired)
    for arm in ("control", "challenger"):
        assert before[arm]["geometry"] == after[arm]["geometry"]
        assert before[arm]["case_dispositions"][case] == after[arm]["case_dispositions"][case]


def test_arm_exchange_swaps_ledgers_and_negates_every_defined_delta(scoring, paired):
    before = scoring.score_gold11_contact_workflows_v5(**paired)
    paired["control_results_by_case"], paired["challenger_results_by_case"] = (
        paired["challenger_results_by_case"], paired["control_results_by_case"])
    requests = paired["scan_plane_requests_by_arm"]
    requests["control"], requests["challenger"] = requests["challenger"], requests["control"]
    after = scoring.score_gold11_contact_workflows_v5(**paired)
    assert after["control"] == before["challenger"]
    assert after["challenger"] == before["control"]
    for key, value in before["paired_deltas"].items():
        assert after["paired_deltas"][key] == (None if value is None else -value)
    for family, rows in before["family_paired_deltas"].items():
        for key, value in rows.items():
            assert after["family_paired_deltas"][family][key] == (None if value is None else -value)


@pytest.mark.parametrize("problem", ["later-row", "duplicate-id", "binding-authority", "hash-contradiction", "physical-count", "count-contradiction", "embedded-identity", "wrong-arm"])
def test_compact_tampering_retains_whole_case_reference_false_negatives(scoring, paired, problem):
    case = sorted(paired["case_family_by_id"])[0]
    projection = paired["challenger_results_by_case"][case]
    if problem == "later-row":
        projection["predictions"][-1]["line_start"] = [float("nan"), 0, 0]
    elif problem == "duplicate-id":
        projection["predictions"][-1]["bend_id"] = projection["predictions"][0]["bend_id"]
        projection["predictions"][-1]["feature_id"] = projection["predictions"][0]["feature_id"]
    elif problem == "binding-authority":
        projection["result_binding"]["authority"] = "RAW_SUPPORT_VERIFIED"
    elif problem == "hash-contradiction":
        projection["result_binding"]["supplied_sha256"] = "0" * 64
    elif problem == "physical-count":
        projection["observed_bend_count"] = 7
    elif problem == "count-contradiction":
        projection["diagnostics"]["finite_supported_group_count"] = 1
    elif problem == "embedded-identity":
        projection["embedded_case_id"] = "different-toy-case"
    else:
        paired["challenger_results_by_case"][case] = paired["control_results_by_case"][case]
    report = scoring.score_gold11_contact_workflows_v5(**paired)
    row = report["challenger"]["case_dispositions"][case]
    assert row["disposition"] == "MALFORMED_RESULT"
    assert row["prediction_count"] == row["true_positive"] == row["false_positive"] == 0
    assert row["false_negative"] == row["gold_count"] == 7
    assert report["challenger"]["geometry"]["reference_line_count"] == 71
    assert report["challenger"]["geometry"]["case_count"] == 11


@pytest.mark.parametrize("kind", ["complete-empty", "refused", "malformed"])
def test_all_zero_dispositions_preserve_denominators_and_undefined_metrics(scoring, paired, kind):
    for case in paired["challenger_results_by_case"]:
        if kind == "complete-empty":
            projection = _project(scoring, _workflow(mode="composite_v1", finite=False, label=case), mode="composite_v1")
        elif kind == "refused":
            projection = _project(scoring, _refusal(), mode="composite_v1")
        else:
            projection = _project(scoring, dict(schema="WRONG"), mode="composite_v1")
        paired["challenger_results_by_case"][case] = projection
    report = scoring.score_gold11_contact_workflows_v5(**paired)
    geometry = report["challenger"]["geometry"]
    assert geometry["reference_line_count"] == geometry["false_negative"] == 71
    assert geometry["true_positive"] == geometry["false_positive"] == geometry["emitted_support_line_count"] == 0
    assert geometry["precision"] is geometry["f1"] is None
    assert geometry["recall"] == 0.
    assert report["paired_deltas"]["f1"] is None
    key = {"complete-empty": "complete_no_localized_carrier_case_count", "refused": "refused_case_count",
        "malformed": "malformed_case_count"}[kind]
    assert report["challenger"][key] == 11


@pytest.mark.parametrize("problem", ["missing-result-case", "missing-reference-case", "wrong-family-count", "seventy-lines", "duplicate-reference", "same-mode"])
def test_complete_map_reference_or_request_errors_raise_not_drop_cases(scoring, paired, problem):
    first = sorted(paired["case_family_by_id"])[0]
    if problem == "missing-result-case":
        paired["challenger_results_by_case"].pop(first)
    elif problem == "missing-reference-case":
        paired["reference_records_by_case"].pop(first)
    elif problem == "wrong-family-count":
        paired["case_family_by_id"] = {case: "one-family" for case in paired["case_family_by_id"]}
    elif problem == "seventy-lines":
        paired["reference_records_by_case"][first].pop()
    elif problem == "duplicate-reference":
        paired["reference_records_by_case"][first][-1]["bend_id"] = "reference-0"
    else:
        paired["scan_plane_requests_by_arm"]["challenger"] = dict(mode="v7", configuration=None)
    with pytest.raises(ValueError):
        scoring.score_gold11_contact_workflows_v5(**paired)


@pytest.mark.parametrize("phase", ["SUPERVISOR", "GLOBAL_TERMINALIZATION"])
def test_absent_or_failed_worker_supervisor_envelope_is_refused_without_result_attestation(scoring, paired, phase):
    case = sorted(paired["case_family_by_id"])[0]
    paired["challenger_results_by_case"][case] = dict(status="UNAVAILABLE", phase=phase,
        reasons=["INVENTED_WORKER_FAILURE"], predictions=[], observed_bend_count=None, production_use_allowed=False)
    report = scoring.score_gold11_contact_workflows_v5(**paired)
    row = report["challenger"]["case_dispositions"][case]
    assert row["disposition"] == "REFUSED"
    assert row["reason"] == "EXPLICIT_SUPERVISOR_UNAVAILABLE"
    assert row["prediction_count"] == 0 and row["false_negative"] == 7
    assert row["result_binding"] is None and row["diagnostics"] is None
    assert row["reported_workflow_phase"] == phase


@pytest.mark.parametrize("change", ["carrier", "count", "non-supervisor-phase"])
def test_malformed_supervisor_refusal_is_not_an_admitted_empty_case(scoring, paired, change):
    case = sorted(paired["case_family_by_id"])[0]
    row = dict(status="UNAVAILABLE", phase="SUPERVISOR", reasons=["INVENTED_WORKER_FAILURE"],
        predictions=[], observed_bend_count=None, production_use_allowed=False)
    if change == "carrier":
        row["predictions"] = [dict(bend_id="not-a-refusal")]
    elif change == "count":
        row["observed_bend_count"] = 0
    else:
        row["phase"] = "COMPLETE"
    paired["challenger_results_by_case"][case] = row
    result = scoring.score_gold11_contact_workflows_v5(**paired)["challenger"]["case_dispositions"][case]
    assert result["disposition"] == "MALFORMED_RESULT"
    assert result["prediction_count"] == 0 and result["false_negative"] == 7


# ---- serialized layer-policy domain mirrors the kernel constructor (Codex consumer-chain P2) ----

def _every_layer_policy(node, out=None):
    """Every serialized layer-policy copy in a payload (source result, records, cover groups, nested)."""
    out = [] if out is None else out
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "layer_policy" and isinstance(value, dict):
                out.append(value)
            else:
                _every_layer_policy(value, out)
    elif isinstance(node, list):
        for value in node:
            _every_layer_policy(value, out)
    return out


@pytest.mark.parametrize("field,value", [
    ("layer_refit_iterations", 0), ("layer_refit_iterations", 9), ("layer_refit_iterations", 2.0),
    ("max_bridging_row_fraction", .9), ("max_bridging_row_fraction", 0.), ("max_bridging_row_fraction", .5000001),
    ("min_layer_rows", 1), ("min_layer_rows", 11), ("min_layer_rows", 100_001), ("max_layers_reported", 1), ("max_layers_reported", 65),
    ("max_layer_normal_angle_deg", 45.5), ("max_layer_normal_angle_deg", 0.), ("min_projected_interval_overlap", 1.5),
    ("layer_mode_ambiguity_fraction", 0.), ("layer_mode_ambiguity_fraction", 1.0001), ("max_layer_rms_mm", 100.5),
    ("layer_mode_bin_mm", .11), ("layer_mode_bin_mm", .1), ("layer_membership_band_mm", .16), ("max_layer_gap_mm", .3),
    ("max_layer_gap_mm", .29), ("layer_split_gap_mm", .19), ("layer_split_gap_mm", 0.), ("enabled", 1), ("layer_split_gap_mm", "0.3"),
])
def test_layer_policy_outside_the_kernel_constructor_domain_is_malformed_even_when_every_copy_agrees(scoring, field, value):
    payload = _workflow(2)
    copies = _every_layer_policy(payload)
    assert len(copies) >= 5                                                  # source result + records + groups, both cases
    for copy_ in copies:
        copy_[field] = value
    result = _project(scoring, payload)
    assert result["disposition"] == "MALFORMED_RESULT" and result["predictions"] == []
    assert "LAYER_POLICY" in result["reason"]


@pytest.mark.parametrize("field,value", [
    ("layer_refit_iterations", 1), ("layer_refit_iterations", 8), ("max_bridging_row_fraction", .5), ("min_layer_rows", 12),
    ("max_layers_reported", 2), ("max_layers_reported", 64), ("layer_mode_bin_mm", .09), ("layer_membership_band_mm", .15),
    ("layer_split_gap_mm", .29), ("max_layer_gap_mm", .31),
    ("max_layer_normal_angle_deg", 45.), ("layer_mode_ambiguity_fraction", 1.), ("min_projected_interval_overlap", 1.),
])
def test_layer_policy_on_the_kernel_constructor_boundary_is_accepted_when_every_copy_agrees(scoring, field, value):
    payload = _workflow(2)
    for copy_ in _every_layer_policy(payload):
        copy_[field] = value
    result = _project(scoring, payload)
    assert result["disposition"] == "SCORED_CONDITIONAL", result.get("reason")
    assert len(result["predictions"]) == 2


def test_serialized_layer_policy_domain_is_bound_to_the_kernel_constructor(scoring):
    """The inert mirror equals the kernel's own domain: every boundary in the mirror is accepted by the kernel
    constructor and every value just outside it is rejected by both. The scorer still imports no engine module."""
    import importlib.util as _util
    spec = _util.spec_from_file_location("bound_kernel_v3_for_policy_domain", ROOT/"backend"/"cad_guided_measurement_v3.py")
    kernel = _util.module_from_spec(spec); sys.modules[spec.name] = kernel; spec.loader.exec_module(kernel)
    assert not any(name.startswith("backend") for name in getattr(scoring, "__dict__", {}) if hasattr(getattr(scoring, name), "__file__"))
    base = _layer_policy()
    coupled = ("layer_split_gap_mm", "layer_mode_bin_mm", "layer_membership_band_mm", "max_layer_gap_mm")
    for key, low, high in scoring.LAYER_POLICY_SCALAR_DOMAIN:
        for value, accepted in ((high, True), (low, False), (high + 1e-6, False)):
            candidate = dict(base, **{key: value})
            if key in coupled:
                accepted = None                                          # cross-field coupled: agreement only, exact cases below
            kernel_ok = True
            try:
                kernel.LayerSelectionPolicyV2(**candidate)
            except ValueError:
                kernel_ok = False
            mirror_ok = True
            try:
                scoring._layer_policy(candidate, "X")
            except ValueError:
                mirror_ok = False
            assert kernel_ok is mirror_ok and (accepted is None or kernel_ok is accepted), (key, value)
    for key, low, high in scoring.LAYER_POLICY_INTEGER_DOMAIN:
        for value, accepted in ((low, True), (high, True), (low - 1, False), (high + 1, False)):
            candidate = dict(base, **{key: value})
            kernel_ok = mirror_ok = True
            try:
                kernel.LayerSelectionPolicyV2(**candidate)
            except ValueError:
                kernel_ok = False
            try:
                scoring._layer_policy(candidate, "X")
            except ValueError:
                mirror_ok = False
            assert kernel_ok is mirror_ok is accepted, (key, value)
    # Cross-field bounds are evaluated in the same binary64 arithmetic as the kernel (0.1 > 0.3/3 there), so the
    # mirror agrees with the kernel on the exact boundary too; the accepted/rejected expectations use safe margins.
    for candidate, accepted in ((dict(base, layer_mode_bin_mm=.09), True), (dict(base, layer_mode_bin_mm=.1001), False),
                                (dict(base, layer_mode_bin_mm=.1), False),
                                (dict(base, layer_membership_band_mm=.15), True), (dict(base, layer_membership_band_mm=.1501), False),
                                (dict(base, max_layer_gap_mm=.31), True), (dict(base, max_layer_gap_mm=.3), False)):
        kernel_ok = mirror_ok = True
        try:
            kernel.LayerSelectionPolicyV2(**candidate)
        except ValueError:
            kernel_ok = False
        try:
            scoring._layer_policy(candidate, "X")
        except ValueError:
            mirror_ok = False
        assert kernel_ok is mirror_ok is accepted, candidate
