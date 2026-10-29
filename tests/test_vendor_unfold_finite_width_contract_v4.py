"""Authored diagnostic rows only; never CAD/mesh/native/file experiments."""
import copy
import math

import pytest

from backend import vendor_unfold_finite_width_contract_v4 as contract


def authored(count=1, branch=False):
    candidates = [{"feature_id": f"native-{i}", "line_start": [0, 0, i], "line_end": [0, 12, i]}
                  for i in range(count)]
    cumulative = {key: 0 for key in contract.WORK_CAPS}
    def certificate(side):
        work = {"triangles": 2, "projected_vertices": 4, "clip_edge_tests": 12}
        for key in cumulative: cumulative[key] += work[key]
        return {"status": "CERTIFIED", "model": contract.MODEL, "side": side,
            "geometry_mutated": False, "manufacturing_ready": False, "exact_binary_clipping": True,
            "tolerance_mm": 1e-5, "axial_bounds": [0., 12.],
            "expanded_axial_bounds": [-1e-5, 12.+1e-5], "intersecting_triangles": 2,
            "max_oriented_side_distance_mm": 0., "charged_uncertainty_mm": 0.,
            "work": work, "shared_work": cumulative.copy(),
            "required_external_gates": sorted(contract.EXTERNAL_GATES)}
    proofs = [{"feature_id": row["feature_id"], "cylinder_source_face_index": 2*i+1,
               "parent_source_face_index": 0 if branch else 2*i, "child_source_face_index": 2*i+2,
               "parent": certificate("parent"), "child": certificate("child")}
              for i, row in enumerate(candidates)]
    payload = {"status": "AVAILABLE", "diagnostics": {
        "mechanism_version": 4, "finite_width_model": contract.MODEL,
        "legacy_engine_source_sha256": contract.LEGACY_ENGINE_SHA256,
        "finite_width_source_sha256": contract.FINITE_WIDTH_SHA256,
        "limitations": sorted(contract.LIMITATIONS),
        "candidate_count": count, "native_face_count": 2*(2*count+1),
        "selected_source_face_indices": list(range(2*count+1)),
        "selected_cylinder_face_indices": list(range(1, 2*count, 2)),
        "root_source_face_index": 0, "selected_skin_rule": "LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN",
        "finite_width_proofs": proofs, "finite_width_work": cumulative.copy(),
        **{key: True for key in contract.OBLIGATIONS}}}
    return payload, candidates


def check(payload, candidates):
    return contract.validate_finite_width_diagnostics(payload, candidates)


def refuses(payload, candidates, reason="."):
    with pytest.raises(contract.FiniteWidthContractError, match=reason):
        check(payload, candidates)


@pytest.mark.parametrize("count,branch", [(1, False), (3, False), (3, True), (32, True)])
def test_authored_complete_ledgers_pass_without_mutation(count, branch):
    payload, candidates = authored(count, branch)
    before = copy.deepcopy((payload, candidates))
    assert check(payload, candidates) is None
    assert (payload, candidates) == before


def test_feature_sort_and_candidate_order_do_not_replace_budget_chronology():
    payload, candidates = authored(3)
    payload["diagnostics"]["finite_width_proofs"].reverse()
    candidates.reverse()
    assert check(payload, candidates) is None


def test_addon_is_not_a_body_marker_or_available_admission_contract():
    payload, candidates = authored()
    payload.update({"vertices": ["invalid body"], "triangles": [True], "bend_lines": None,
                    "model": "NOT_A_COMMON_PREVIEW", "source_sha256": "unbound"})
    # Intentional: these unrelated fields MUST be rejected by the mandatory
    # common/source/transport contract, not silently certified by this add-on.
    assert check(payload, candidates) is None
    assert payload["model"] == "NOT_A_COMMON_PREVIEW"


def test_optional_v4_runtime_declaration_is_checked_not_invented():
    payload, candidates = authored()
    assert "v4_engine_source_sha256" not in payload["diagnostics"]
    assert check(payload, candidates) is None
    payload["diagnostics"]["v4_engine_source_sha256"] = contract.V4_ENGINE_SHA256
    assert check(payload, candidates) is None
    payload["diagnostics"]["v4_engine_source_sha256"] = "b"*64
    refuses(payload, candidates, "UNREVIEWED_V4")


@pytest.mark.parametrize("field,value", [
    ("mechanism_version", 3), ("mechanism_version", 4.), ("mechanism_version", True),
    ("finite_width_model", "CONTINUOUS_CAD_PROOF"), ("finite_width_model", []),
    ("legacy_engine_source_sha256", "b"*64), ("finite_width_source_sha256", "b"*64),
    ("candidate_count", True), ("candidate_count", 2), ("native_face_count", 5),
    ("native_face_count", 1025), ("native_face_count", True),
    ("root_source_face_index", 1), ("root_source_face_index", True),
    ("selected_skin_rule", "BEST_PARTIAL_SKIN"), ("limitations", []), ("limitations", [None]),
])
def test_top_level_typed_mechanism_ledger_refusals(field, value):
    payload, candidates = authored()
    payload["diagnostics"][field] = value
    refuses(payload, candidates)


@pytest.mark.parametrize("field", contract.OBLIGATIONS)
@pytest.mark.parametrize("value", [False, 1, None])
def test_obligations_are_true_booleans_not_truthy_claims(field, value):
    payload, candidates = authored()
    payload["diagnostics"][field] = value
    refuses(payload, candidates, "OBLIGATION")


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_feature", "foreign_feature",
                                     "duplicate_cylinder", "foreign_cylinder", "bool_cylinder"])
def test_feature_cylinder_exact_bijection(change):
    payload, candidates = authored(2)
    proofs = payload["diagnostics"]["finite_width_proofs"]
    if change == "missing": proofs.pop()
    elif change == "extra": proofs.append(copy.deepcopy(proofs[0]))
    elif change == "duplicate_feature": proofs[1]["feature_id"] = proofs[0]["feature_id"]
    elif change == "foreign_feature": proofs[1]["feature_id"] = "unknown"
    elif change == "duplicate_cylinder": proofs[1]["cylinder_source_face_index"] = 1
    elif change == "foreign_cylinder": proofs[1]["cylinder_source_face_index"] = 9
    else: proofs[0]["cylinder_source_face_index"] = True
    refuses(payload, candidates)


@pytest.mark.parametrize("change", ["same_node", "child_root", "duplicate_child", "cylinder_parent",
                                     "unknown_parent", "cycle", "bool_parent", "disconnected_chronology"])
def test_plane_tree_and_transform_dependency_refusals(change):
    payload, candidates = authored(3)
    proofs = payload["diagnostics"]["finite_width_proofs"]
    if change == "same_node": proofs[0]["child_source_face_index"] = 0
    elif change == "child_root": proofs[1]["child_source_face_index"] = 0
    elif change == "duplicate_child": proofs[1]["child_source_face_index"] = 2
    elif change == "cylinder_parent": proofs[0]["parent_source_face_index"] = 1
    elif change == "unknown_parent": proofs[0]["parent_source_face_index"] = 30
    elif change == "bool_parent": proofs[0]["parent_source_face_index"] = False
    elif change == "cycle":
        proofs[0]["parent_source_face_index"] = 6
        proofs[2]["parent_source_face_index"] = 2
    else:
        # A complete rooted tree, but the first transform depends on plane4
        # which has not yet been developed at that cumulative-budget position.
        proofs[0]["parent_source_face_index"] = 4
        proofs[1]["parent_source_face_index"] = 0
    refuses(payload, candidates)


@pytest.mark.parametrize("field,value", [
    ("status", "PARTIAL"), ("model", "MANUFACTURING_PROOF"), ("side", "child"),
    ("geometry_mutated", True), ("geometry_mutated", 0), ("manufacturing_ready", True),
    ("manufacturing_ready", 0), ("exact_binary_clipping", 1),
    ("tolerance_mm", 1.01e-5), ("tolerance_mm", True),
    ("intersecting_triangles", 0), ("intersecting_triangles", True), ("intersecting_triangles", 3),
    ("required_external_gates", []), ("required_external_gates", [None]*3),
    ("max_oriented_side_distance_mm", 1.01e-5), ("charged_uncertainty_mm", -1e-9),
    ("charged_uncertainty_mm", 1e-5),
])
def test_closed_typed_certificate_refusals(field, value):
    payload, candidates = authored()
    payload["diagnostics"]["finite_width_proofs"][0]["parent"][field] = value
    refuses(payload, candidates)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 10**400, "0", None, True])
def test_nonfinite_and_untyped_diagnostic_numbers_refuse_without_overflow(value):
    payload, candidates = authored()
    payload["diagnostics"]["finite_width_proofs"][0]["parent"]["max_oriented_side_distance_mm"] = value
    refuses(payload, candidates)


@pytest.mark.parametrize("change", ["reversed", "degenerate", "short", "wrong_expansion", "huge", "side_mismatch"])
def test_axial_span_and_unchanged_expansion(change):
    payload, candidates = authored()
    parent = payload["diagnostics"]["finite_width_proofs"][0]["parent"]
    if change == "reversed": parent["axial_bounds"] = [12, 0]
    elif change == "degenerate": parent["axial_bounds"] = [0, 0]
    elif change == "short": parent["axial_bounds"] = [0, 1e-6]
    elif change == "wrong_expansion": parent["expanded_axial_bounds"][0] = -2e-5
    elif change == "huge": parent["axial_bounds"] = [1e12, 1e12+12]
    else:
        parent["axial_bounds"] = [1., 13.]
        parent["expanded_axial_bounds"] = [1.-1e-5, 13.+1e-5]
    refuses(payload, candidates)


def test_shared_side_error_budget_is_not_tolerance_per_component():
    payload, candidates = authored()
    parent = payload["diagnostics"]["finite_width_proofs"][0]["parent"]
    parent["max_oriented_side_distance_mm"] = 9e-6
    parent["charged_uncertainty_mm"] = 1.4e-6
    refuses(payload, candidates, "SIDE_BOUND")
    parent["max_oriented_side_distance_mm"] = 8e-6
    assert check(payload, candidates) is None


def test_only_report_rounding_ulps_not_a_new_geometry_tolerance():
    payload, candidates = authored()
    parent = payload["diagnostics"]["finite_width_proofs"][0]["parent"]
    parent["charged_uncertainty_mm"] = 1e-6
    parent["max_oriented_side_distance_mm"] = math.nextafter(9e-6, math.inf)
    assert check(payload, candidates) is None
    parent["max_oriented_side_distance_mm"] = 9e-6+1e-12
    refuses(payload, candidates, "SIDE_BOUND")


@pytest.mark.parametrize("change", ["local_zero", "local_bool", "local_cap", "vertices_cap", "edge_low", "edge_high",
                                     "cumulative_reset", "cumulative_equal", "one_counter_drift", "global_drift",
                                     "global_cap", "extra_key", "pair_interleave"])
def test_exact_shared_work_chain_and_fixed_caps(change):
    payload, candidates = authored(2)
    diag = payload["diagnostics"]
    a, b = diag["finite_width_proofs"]
    cert = a["parent"]
    if change == "local_zero": cert["work"]["triangles"] = 0
    elif change == "local_bool": cert["work"]["triangles"] = True
    elif change == "local_cap": cert["work"]["triangles"] = 200001
    elif change == "vertices_cap": cert["work"]["projected_vertices"] = 600001
    elif change == "edge_low": cert["work"]["clip_edge_tests"] = 5
    elif change == "edge_high": cert["work"]["clip_edge_tests"] = 15
    elif change == "cumulative_reset": b["parent"]["shared_work"] = cert["shared_work"].copy()
    elif change == "cumulative_equal": a["child"]["shared_work"] = cert["shared_work"].copy()
    elif change == "one_counter_drift": b["parent"]["shared_work"]["projected_vertices"] += 1
    elif change == "global_drift": diag["finite_width_work"]["triangles"] += 1
    elif change == "global_cap": diag["finite_width_work"]["triangles"] = 1000001
    elif change == "extra_key": cert["work"]["unmetered"] = 1
    else:
        # Preserve a valid global counter chain but interleave parent A/B then
        # child A/B, which the reviewed producer never emits.
        a["child"]["shared_work"], b["parent"]["shared_work"] = b["parent"]["shared_work"], a["child"]["shared_work"]
    refuses(payload, candidates)


@pytest.mark.parametrize("value", [None, [], "payload", {}, {"diagnostics": None}])
def test_malformed_payload_refuses_typed(value):
    _, candidates = authored()
    refuses(value, candidates)


@pytest.mark.parametrize("value", [None, [], {}, [None], [{"feature_id": "bad"}]])
def test_malformed_candidates_refuse_typed(value):
    payload, _ = authored()
    refuses(payload, value)


@pytest.mark.parametrize("change", ["row_extra", "cert_extra", "cert_missing", "duplicate_gate", "duplicate_selected", "duplicate_cylinder"])
def test_closed_rows_certificates_gates_and_source_tables(change):
    payload, candidates = authored()
    diag = payload["diagnostics"]
    row = diag["finite_width_proofs"][0]
    if change == "row_extra": row["partial_result"] = True
    elif change == "cert_extra": row["parent"]["release_allowed"] = False
    elif change == "cert_missing": del row["parent"]["exact_binary_clipping"]
    elif change == "duplicate_gate": row["parent"]["required_external_gates"] = [sorted(contract.EXTERNAL_GATES)[0]]*3
    elif change == "duplicate_selected": diag["selected_source_face_indices"] = [0, 0, 2]
    else: diag["selected_cylinder_face_indices"] = [True]
    refuses(payload, candidates)
