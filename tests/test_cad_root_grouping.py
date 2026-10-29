"""Invented CAD-root grouping contracts; no native or physical inputs.

Face and edge identities are stipulated source metadata in these tests, not
reconstructed from mesh vertex indices or inferred by spatial coincidence.
Only nominated root faces may connect candidates. Grouping is an expected-CAD
hypothesis, not observed bend detection, physical count, thickness or forming
state. The tests supply no raw data, runtime imports or protected artifacts.
"""
from copy import deepcopy
import inspect
import json
import math

import numpy as np
import pytest

from research.geometry.cad_root_grouping import group_cad_root_candidates


def candidate(identifier, faces, *, owner="solid_A", start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 1.0)):
    return dict(candidate_id=identifier, owner_id=owner, face_ids=list(faces),
                axis_start=list(start), axis_end=list(end))


def face(identifier, edges, *, owner="solid_A"):
    return dict(face_id=identifier, owner_id=owner, edge_ids=list(edges))


def topology(faces, *, status="COMPLETE"):
    return dict(source_id="invented_native_identity_source", status=status, faces=list(faces))


def chain_fixture(count=3):
    """Native-edge chain with one nominated face and finite segment per root."""
    faces = [face(f"face_{index}", [f"edge_{index}", f"edge_{index+1}"]) for index in range(count)]
    candidates = [candidate(f"candidate_{index}", [f"face_{index}"],
                            start=(0.0, 0.0, float(index)), end=(0.0, 0.0, float(index+1)))
                  for index in range(count)]
    return candidates, topology(faces)


def group(candidates, source_topology, **changes):
    policy = dict(unit="mm", frame_id="invented_CAD_frame", max_axis_angle=0.15,
                  max_axis_separation=0.1, max_segment_gap=0.2, max_pair_evaluations=32640)
    policy.update(changes)
    return group_cad_root_candidates(candidates, source_topology, **policy)


def expected_frozen_candidates(candidates):
    return [dict(candidate_id=item["candidate_id"], owner_id=item["owner_id"],
                 face_ids=sorted(item["face_ids"]),
                 axis_start=np.asarray(item["axis_start"], dtype=float).tolist(),
                 axis_end=np.asarray(item["axis_end"], dtype=float).tolist())
            for item in sorted(candidates, key=lambda item: item["candidate_id"])]


def assert_scope(result, candidates):
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["kind"] == "CAD_LOCAL_ROOT_GROUPING_AUDIT_V1"
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["observed_state"] == "UNKNOWN"
    assert result["physical_geometry_is_verified"] is False
    assert result["event_identity_is_verified"] is False
    assert result["input_candidates"] == expected_frozen_candidates(candidates)
    policy = result["policy"]
    assert policy["topology_scope"] == "DECLARED_NATIVE_FACE_EDGE_IDENTITY_NOT_MESH_VERTEX_CONNECTIVITY"
    assert policy["connectivity_scope"] == "GRAPH_INDUCED_ONLY_BY_NOMINATED_ROOT_FACES_WITHIN_OWNER"
    assert policy["geometry_scope"] == "ALL_PAIRS_AXIS_CHECKS_FINITE_GAP_ONLY_ON_DIRECT_TOPOLOGY_LINKS"
    assert policy["segment_gap_scope"] == "MAX_AXIAL_PROJECTED_INTERVAL_GAP_IN_BOTH_DIRECTIONS_NOT_EUCLIDEAN_DISTANCE"
    assert policy["no_convex_hull_span_or_geometry_average"] is True
    assert policy["any_unresolved_component_clears_all_groups"] is True
    assert policy["max_candidates"] == 256 and policy["max_faces"] == 4096
    assert policy["max_edge_incidences"] == 32768
    assert policy["max_candidate_face_memberships"] == 16384
    for limitation in (
        "TOPOLOGY_COMPLETENESS_AND_IDENTITY_STIPULATED_NOT_VERIFIED",
        "CAD_EXPECTATIONS_NOT_SCAN_OBSERVATIONS", "NO_REGISTRATION_OR_PHYSICAL_LOCALIZATION",
        "LOCAL_CONNECTED_SUPPORT_NOT_MANUFACTURING_EVENT_AUTHORITY",
        "CROSS_FACE_INNER_OUTER_PAIRING_AND_THICKNESS_UNESTABLISHED",
        "FINITE_SEGMENTS_RETAINED_NOT_CERTIFIED_PHYSICAL_MARKERS",
    ):
        assert limitation in result["limitations"]
    by_id = {item["candidate_id"]: item for item in expected_frozen_candidates(candidates)}
    for root in result["groups"]:
        assert root["observed_state"] == "UNKNOWN"
        assert root["event_identity_is_verified"] is False
        assert root["scope"] == "CONNECTED_CAD_ROOT_GROUPING_HYPOTHESIS_NOT_PHYSICAL_OR_OBSERVED_EVENT"
        assert root["member_segments"] == [
            dict(candidate_id=name, axis_start=by_id[name]["axis_start"], axis_end=by_id[name]["axis_end"])
            for name in root["candidate_ids"]]
        assert root["face_ids"] == sorted({identity for name in root["candidate_ids"]
                                           for identity in by_id[name]["face_ids"]})
        assert {by_id[name]["owner_id"] for name in root["candidate_ids"]} == {root["owner_id"]}
        assert not {"axis_start", "axis_end", "marker_segment", "thickness"} & set(root)


def assert_complete(result, candidates, expected_memberships):
    assert_scope(result, candidates)
    assert result["search_complete_within_declared_policy"] is True
    assert result["status"] == "CAD_ROOT_GROUPING_HYPOTHESES"
    memberships = {frozenset(root["candidate_ids"]) for root in result["groups"]}
    assert memberships == {frozenset(names) for names in expected_memberships}
    assert result["group_hypothesis_count"] == len(expected_memberships)
    assert len(result["candidate_to_group"]) == len(candidates)
    assert set(result["candidate_to_group"]) == {item["candidate_id"] for item in candidates}
    for root in result["groups"]:
        assert all(result["candidate_to_group"][name] == root["group_id"] for name in root["candidate_ids"])
    assert result["reasons"] == []


def assert_unresolved(result, candidates, reason):
    assert_scope(result, candidates)
    assert result["status"] == "ABSTAINED"
    assert not result["search_complete_within_declared_policy"]
    assert result["groups"] == [] and result["candidate_to_group"] == {}
    assert result["group_hypothesis_count"] is None
    assert result["reasons"] == [reason]


def test_api_requires_explicit_frame_unit_geometry_policy_and_pair_budget():
    signature = inspect.signature(group_cad_root_candidates)
    assert set(signature.parameters) == {
        "candidates", "topology", "unit", "frame_id", "max_axis_angle",
        "max_axis_separation", "max_segment_gap", "max_pair_evaluations",
    }
    assert all(value.default is inspect.Parameter.empty for value in signature.parameters.values())
    assert all(value.kind is inspect.Parameter.KEYWORD_ONLY
               for name, value in signature.parameters.items() if name not in ("candidates", "topology"))


def test_native_root_chain_retains_every_original_finite_segment_without_enclosing_marker():
    candidates, source = chain_fixture()
    original = deepcopy(candidates)
    result = group(candidates, source)
    assert_complete(result, candidates, [["candidate_0", "candidate_1", "candidate_2"]])
    assert candidates == original
    assert result["diagnostics"]["pair_evaluation_count"] == 3
    # The first and third spans have a one-unit axial gap, greater than 0.2.
    # That is allowed: they are not direct native-face links. The adjacent
    # pairs touch, and all three supporting lines are mutually compatible.
    assert candidates[2]["axis_start"][2]-candidates[0]["axis_end"][2] == 1.0
    assert len(result["groups"][0]["member_segments"]) == 3


def test_remote_unnominated_flange_does_not_bridge_two_roots_in_the_same_owner():
    _, source = chain_fixture()
    candidates = [candidate("left", ["face_0"]), candidate("right", ["face_2"])]
    result = group(candidates, source)
    assert_complete(result, candidates, [["left"], ["right"]])
    assert result["diagnostics"]["root_component_count"] == 2
    assert result["diagnostics"]["pair_evaluation_count"] == 0
    assert all("face_1" not in root["face_ids"] for root in result["groups"])


def test_corner_only_geometric_contact_cannot_invent_a_shared_native_edge():
    candidates = [candidate("one", ["first"]),
                  candidate("two", ["second"], start=(0.0, 0.0, 1.0), end=(0.0, 0.0, 2.0))]
    source = topology([face("first", ["native_edge_A"]), face("second", ["native_edge_B"])])
    # Axis endpoints meet, but no shared edge identity is supplied. This test
    # does not claim a verified CAD vertex from that coordinate coincidence.
    assert candidates[0]["axis_end"] == candidates[1]["axis_start"]
    assert_complete(group(candidates, source), candidates, [["one"], ["two"]])


def test_coincident_owners_with_reused_edge_names_are_never_cross_owner_roots():
    candidates = [candidate("A", ["face_A"], owner="owner_A"),
                  candidate("B", ["face_B"], owner="owner_B")]
    source = topology([face("face_A", ["edge_0"], owner="owner_A"),
                       face("face_B", ["edge_0"], owner="owner_B")])
    assert_complete(group(candidates, source), candidates, [["A"], ["B"]])


def test_nonmanifold_edge_incidence_is_counted_within_owner_not_globally():
    candidates = [candidate("A0", ["A0"], owner="owner_A"),
                  candidate("A1", ["A1"], owner="owner_A"),
                  candidate("B0", ["B0"], owner="owner_B")]
    source = topology([face(item["candidate_id"], ["same_edge_name"], owner=item["owner_id"])
                       for item in candidates])
    assert_complete(group(candidates, source), candidates, [["A0", "A1"], ["B0"]])


@pytest.mark.parametrize("nominate_third", [False, True])
def test_nonmanifold_native_incidence_abstains_even_when_extra_face_is_not_nominated(nominate_third):
    source = topology([face(f"face_{index}", ["shared_edge"]) for index in range(3)])
    candidates = [candidate(f"candidate_{index}", [f"face_{index}"])
                  for index in range(3 if nominate_third else 2)]
    result = group(candidates, source)
    assert_unresolved(result, candidates, "NONMANIFOLD_NATIVE_EDGE_INCIDENCE")
    assert result["diagnostics"]["pair_evaluation_count"] == 0


def test_each_candidates_own_face_set_must_connect_without_borrowing_another_candidate():
    _, source = chain_fixture()
    candidates = [candidate("broken", ["face_0", "face_2"]), candidate("bridge", ["face_1"])]
    result = group(candidates, source)
    assert_unresolved(result, candidates, "CANDIDATE_LOCAL_SUPPORT_DISCONNECTED")
    assert result["diagnostics"]["unresolved_candidate_ids"] == ["broken"]
    assert result["diagnostics"]["pair_evaluation_count"] == 0


def test_connected_multiface_candidate_and_overlapping_candidate_preserve_all_identities():
    _, source = chain_fixture()
    candidates = [candidate("multi", ["face_1", "face_0"]), candidate("overlap", ["face_1"])]
    assert_complete(group(candidates, source), candidates, [["multi", "overlap"]])


@pytest.mark.parametrize("kind", ["angle", "parallel_offset"])
def test_neighbor_compatible_chains_still_require_all_pairs_no_greedy_split_or_merge(kind):
    candidates, source = chain_fixture()
    for index, item in enumerate(candidates):
        if kind == "angle":
            angle = 0.1*index
            item["axis_start"] = [0.0, 0.0, 0.0]
            item["axis_end"] = [math.sin(angle), 0.0, math.cos(angle)]
        else:
            item["axis_start"] = [0.075*index, 0.0, 0.0]
            item["axis_end"] = [0.075*index, 0.0, 1.0]
    changes = dict(max_axis_separation=1.0 if kind == "angle" else 0.1)
    for pair in (candidates[:2], candidates[1:]):
        assert_complete(group(pair, source, **changes), pair, [[item["candidate_id"] for item in pair]])
    result = group(candidates, source, **changes)
    assert_unresolved(result, candidates, "ROOT_COMPONENT_AXIS_COMPATIBILITY_UNRESOLVED")
    assert result["diagnostics"]["unresolved_candidate_ids"] == [item["candidate_id"] for item in candidates]
    assert result["diagnostics"]["pair_evaluation_count"] == 2


@pytest.mark.parametrize("long_first", [False, True])
def test_endpoint_separation_is_checked_against_both_supporting_lines(long_first):
    candidates, source = chain_fixture(2)
    lengths = (100.0, 1.0) if long_first else (1.0, 100.0)
    angles = (0.0, 0.01)
    for item, length, angle in zip(candidates, lengths, angles):
        item["axis_start"] = [0.0, 0.0, 0.0]
        item["axis_end"] = [length*math.sin(angle), 0.0, length*math.cos(angle)]
    # Starts coincide and angle 0.01 is inside 0.15, but one distant endpoint
    # is nearly one unit off the other line. Looking only at starts, or at only
    # the short segment's endpoints, would wrongly accept one orientation.
    result = group(candidates, source)
    assert_unresolved(result, candidates, "ROOT_COMPONENT_AXIS_COMPATIBILITY_UNRESOLVED")


@pytest.mark.parametrize("overlap_faces", [False, True])
def test_direct_native_links_need_small_finite_axial_gap_even_for_identical_infinite_lines(overlap_faces):
    candidates, source = chain_fixture(2)
    candidates[1]["axis_start"] = [0.0, 0.0, 3.0]
    candidates[1]["axis_end"] = [0.0, 0.0, 4.0]
    if overlap_faces:
        candidates[1]["face_ids"] = ["face_0"]
    result = group(candidates, source)
    assert_unresolved(result, candidates, "DIRECT_ROOT_LINK_FINITE_GAP_UNRESOLVED")


def test_later_incompatible_component_clears_an_earlier_compatible_group():
    candidates = [candidate("a0", ["fa0"]), candidate("a1", ["fa1"]),
                  candidate("z0", ["fz0"]), candidate("z1", ["fz1"], end=(1.0, 0.0, 0.0))]
    source = topology([face("fa0", ["edge_a"]), face("fa1", ["edge_a"]),
                       face("fz0", ["edge_z"]), face("fz1", ["edge_z"])])
    result = group(candidates, source)
    assert_unresolved(result, candidates, "ROOT_COMPONENT_AXIS_COMPATIBILITY_UNRESOLVED")
    assert result["diagnostics"]["pair_evaluation_count"] == 2
    assert result["diagnostics"]["unresolved_candidate_ids"] == ["z0", "z1"]


def test_grouped_inner_outer_named_faces_do_not_establish_pairing_thickness_or_observed_bends():
    candidates = [candidate("inner_surface_candidate", ["inner", "side"]),
                  candidate("outer_surface_candidate", ["outer"])]
    source = topology([face("inner", ["inner_to_side"]),
                       face("side", ["inner_to_side", "side_to_outer"]),
                       face("outer", ["side_to_outer"])])
    result = group(candidates, source)
    assert_complete(result, candidates, [["inner_surface_candidate", "outer_surface_candidate"]])
    assert result["group_hypothesis_count"] == 1
    assert result["physical_bend_count"] is None
    assert result["observed_state"] == "UNKNOWN"


def test_unknown_native_completeness_keeps_candidates_but_clears_every_group():
    candidates, source = chain_fixture()
    source["status"] = "UNKNOWN"
    result = group(candidates, source)
    assert_unresolved(result, candidates, "TOPOLOGY_COMPLETENESS_UNKNOWN")
    assert result["diagnostics"]["unresolved_candidate_ids"] == [item["candidate_id"] for item in candidates]
    assert result["diagnostics"]["pair_evaluation_count"] == 0


@pytest.mark.parametrize("problem", ["missing_face", "mismatched_owner", "unknown_owner"])
def test_missing_or_inconsistent_owner_identity_cannot_be_repaired_by_geometry(problem):
    candidates, source = chain_fixture(2)
    if problem == "missing_face":
        source["faces"] = source["faces"][:1]
    elif problem == "mismatched_owner":
        source["faces"][1]["owner_id"] = "another_owner"
    else:
        candidates[1]["owner_id"] = None
    result = group(candidates, source)
    assert_unresolved(result, candidates, "MISSING_OR_INCONSISTENT_FACE_OWNERSHIP")
    assert result["diagnostics"]["unresolved_candidate_ids"] == ["candidate_1"]


def test_input_permutations_preserve_complete_canonical_output():
    _, source = chain_fixture()
    candidates = [candidate("z", ["face_0", "face_1"], end=(0.0, 0.0, 2.0)),
                  candidate("a", ["face_2"], start=(0.0, 0.0, 2.0), end=(0.0, 0.0, 3.0))]
    baseline = group(candidates, source)
    changed_candidates, changed_source = deepcopy(candidates[::-1]), deepcopy(source)
    for item in changed_candidates:
        item["face_ids"].reverse()
    changed_source["faces"].reverse()
    for item in changed_source["faces"]:
        item["edge_ids"].reverse()
    assert group(changed_candidates, changed_source) == baseline
    assert_complete(baseline, candidates, [["a", "z"]])


def test_axis_sign_reversal_preserves_group_identity_and_keeps_reversed_original_member_segment():
    candidates, source = chain_fixture()
    baseline = group(candidates, source)
    changed = deepcopy(candidates)
    changed[1]["axis_start"], changed[1]["axis_end"] = changed[1]["axis_end"], changed[1]["axis_start"]
    result = group(changed, source)
    assert_complete(result, changed, [["candidate_0", "candidate_1", "candidate_2"]])
    assert result["candidate_to_group"] == baseline["candidate_to_group"]
    assert result["groups"][0]["member_segments"][1]["axis_start"] == [0.0, 0.0, 2.0]
    assert result["groups"][0]["member_segments"][1]["axis_end"] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("scale,unit", [(1.0, "mm"), (0.001, "m")])
def test_proper_rigid_motion_and_units_preserve_memberships_with_each_finite_segment_transformed(scale, unit):
    candidates, source = chain_fixture()
    baseline = group(candidates, source)
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    translation = np.array([7.0, -11.0, 13.0])
    assert np.linalg.det(rotation) == 1.0
    changed = deepcopy(candidates)
    for item in changed:
        for key in ("axis_start", "axis_end"):
            item[key] = ((rotation@item[key]+translation)*scale).tolist()
    result = group(changed, source, unit=unit, frame_id="moved_CAD_frame",
                   max_axis_separation=0.1*scale, max_segment_gap=0.2*scale)
    assert_complete(result, changed, [["candidate_0", "candidate_1", "candidate_2"]])
    assert result["candidate_to_group"] == baseline["candidate_to_group"]
    assert result["unit"] == unit and result["frame_id"] == "moved_CAD_frame"


def test_retained_candidate_and_member_snapshots_do_not_alias_mutable_input_records():
    candidates, source = chain_fixture(2)
    expected = deepcopy(candidates)
    result = group(candidates, source)
    candidates[0]["axis_start"][0] = 1000.0
    candidates[1]["face_ids"].append("unverified_extra_face")
    source["faces"][0]["edge_ids"].clear()
    assert_complete(result, expected, [["candidate_0", "candidate_1"]])


def test_pair_budget_preflights_the_whole_component_before_any_pair_evaluation():
    candidates, source = chain_fixture()
    result = group(candidates, source, max_pair_evaluations=2)
    assert_unresolved(result, candidates, "PAIR_EVALUATION_BUDGET_EXCEEDED")
    assert result["diagnostics"]["pair_evaluation_count"] == 0


def test_pair_budget_is_aggregate_across_disconnected_components_not_reset_per_group():
    candidates = [candidate(name, [name]) for name in ("a0", "a1", "b0", "b1")]
    source = topology([face(name, ["edge_"+name[0]]) for name in ("a0", "a1", "b0", "b1")])
    result = group(candidates, source, max_pair_evaluations=1)
    assert_unresolved(result, candidates, "PAIR_EVALUATION_BUDGET_EXCEEDED")
    assert result["diagnostics"]["root_component_count"] == 2
    assert result["diagnostics"]["pair_evaluation_count"] == 0


@pytest.mark.parametrize("numeric_range", ["subnormal_allowance", "overflow"])
def test_unsupported_pair_arithmetic_preserves_candidates_but_clears_groups(numeric_range):
    candidates, source = chain_fixture(2)
    if numeric_range == "subnormal_allowance":
        # Direction normalization itself is supported at the smallest normal
        # segment length, but the later relative comparison allowance is not.
        eta = float(np.finfo(float).tiny)
        for item in candidates:
            item["axis_start"] = [0.0, 0.0, 0.0]
            item["axis_end"] = [0.0, 0.0, eta]
    else:
        for item, position in zip(candidates, (-1e308, 1e308)):
            item["axis_start"] = [position, 0.0, 0.0]
            item["axis_end"] = [position, 0.0, 1.0]
    result = group(candidates, source)
    assert_unresolved(result, candidates, "NUMERICAL_RANGE_UNSUPPORTED")
    assert result["diagnostics"]["pair_evaluation_count"] == 1


def test_subnormal_segment_length_is_rejected_before_imprecise_direction_normalization():
    eta = float(np.nextafter(0.0, 1.0))
    candidates = [candidate("subnormal", ["face_0"], start=(1.0, 0.0, 0.0), end=(1.0, 2*eta, 3*eta))]
    with pytest.raises(ValueError):
        group(candidates, topology([face("face_0", [])]))


@pytest.mark.parametrize("threshold", ["angle", "separation", "gap"])
def test_exact_engineering_allowance_equality_abstains_instead_of_accepting_zero_margin(threshold):
    candidates, source = chain_fixture(2)
    # Identical unit segments: angle, lateral distance and projected gap are
    # exactly zero. Magnitude=extent=1 gives a distance allowance of 1024u.
    for item in candidates:
        item["axis_start"] = [0.0, 0.0, 0.0]
        item["axis_end"] = [0.0, 0.0, 1.0]
    epsilon = float(np.finfo(float).eps)
    changes = ({"max_axis_angle": 512*epsilon} if threshold == "angle" else
               {"max_axis_separation": 1024*epsilon} if threshold == "separation" else
               {"max_segment_gap": 1024*epsilon})
    result = group(candidates, source, **changes)
    reason = ("DIRECT_ROOT_LINK_FINITE_GAP_UNRESOLVED" if threshold == "gap" else
              "ROOT_COMPONENT_AXIS_COMPATIBILITY_UNRESOLVED")
    assert_unresolved(result, candidates, reason)


def test_complete_empty_candidate_input_does_not_establish_zero_physical_bends():
    result = group([], topology([]))
    assert_scope(result, [])
    assert result["status"] == "ABSTAINED" and result["reasons"] == ["NO_CAD_CANDIDATES"]
    assert result["search_complete_within_declared_policy"] is True
    assert result["group_hypothesis_count"] == 0
    assert result["physical_bend_count"] is None
    assert result["groups"] == [] and result["candidate_to_group"] == {}


def test_exact_candidate_count_cap_allows_256_disconnected_singletons_without_pair_work():
    candidates = [candidate(f"c_{index:03d}", [f"f_{index:03d}"]) for index in range(256)]
    source = topology([face(f"f_{index:03d}", []) for index in range(256)])
    result = group(candidates, source, max_pair_evaluations=1)
    assert_complete(result, candidates, [[item["candidate_id"]] for item in candidates])
    assert result["diagnostics"]["candidate_count"] == 256
    assert result["diagnostics"]["pair_evaluation_count"] == 0


@pytest.mark.parametrize("budget", ["candidates", "faces", "edge_incidences", "candidate_face_memberships"])
def test_each_fixed_raw_identity_budget_is_enforced_without_truncating_records(budget):
    candidates, source = chain_fixture(1)
    if budget == "candidates":
        candidates = [candidate(f"candidate_{index}", ["face_0"]) for index in range(257)]
    elif budget == "faces":
        source = topology([face(f"face_{index}", []) for index in range(4097)])
    elif budget == "edge_incidences":
        source = topology([face("face_0", [f"edge_a_{index}" for index in range(16384)]),
                           face("face_1", [f"edge_b_{index}" for index in range(16385)])])
    else:
        _, source = chain_fixture(256)
        all_faces = [f"face_{index}" for index in range(256)]
        candidates = [candidate(f"candidate_{index}", all_faces) for index in range(64)]
        candidates.append(candidate("candidate_extra", ["face_0"]))
        assert sum(len(item["face_ids"]) for item in candidates) == 16385
    with pytest.raises(ValueError):
        group(candidates, source)


@pytest.mark.parametrize("field,value", [
    ("unit", "cm"), ("unit", True), ("frame_id", ""), ("frame_id", "mesh vertex IDs"),
    ("max_axis_angle", 0.0), ("max_axis_angle", math.pi/2), ("max_axis_angle", math.nan),
    ("max_axis_angle", True), ("max_axis_separation", 0.0), ("max_axis_separation", -1.0),
    ("max_axis_separation", math.inf), ("max_segment_gap", 0.0), ("max_segment_gap", True),
    ("max_pair_evaluations", 0), ("max_pair_evaluations", 32641),
    ("max_pair_evaluations", 1.5), ("max_pair_evaluations", True),
])
def test_invalid_policies_are_rejected_even_with_no_candidates(field, value):
    with pytest.raises(ValueError):
        group([], topology([]), **{field: value})


@pytest.mark.parametrize("field,value", [
    ("candidate_id", ""), ("candidate_id", "a"*129), ("owner_id", ""),
    ("face_ids", []), ("face_ids", ["face_0", "face_0"]), ("face_ids", [1]),
    ("axis_start", [0.0, 0.0]), ("axis_start", [False, 0.0, 0.0]),
    ("axis_start", [math.nan, 0.0, 0.0]), ("axis_start", [2**53+1, 0, 0]),
    ("axis_end", [0.0, 0.0, 0.0]), ("axis_end", [math.inf, 0.0, 0.0]),
])
def test_malformed_candidates_are_not_silently_repaired(field, value):
    candidates, source = chain_fixture(1)
    candidates[0][field] = value
    with pytest.raises(ValueError):
        group(candidates, source)


@pytest.mark.parametrize("mutation", [
    "candidate_extra", "candidate_missing", "duplicate_candidate", "topology_extra", "topology_missing",
    "bad_topology_status", "bad_source_id", "face_extra_vertex_ids", "face_missing_edges",
    "duplicate_face", "duplicate_edge_id", "bad_face_owner", "bad_edge_id",
])
def test_exact_identity_schemas_do_not_accept_mesh_connectivity_or_inferred_provenance_fallbacks(mutation):
    candidates, source = chain_fixture(1)
    if mutation == "candidate_extra":
        candidates[0]["confidence"] = 1.0
    elif mutation == "candidate_missing":
        del candidates[0]["axis_end"]
    elif mutation == "duplicate_candidate":
        candidates.append(deepcopy(candidates[0]))
    elif mutation == "topology_extra":
        source["native_source_verified"] = True
    elif mutation == "topology_missing":
        del source["source_id"]
    elif mutation == "bad_topology_status":
        source["status"] = "PARTIAL"
    elif mutation == "bad_source_id":
        source["source_id"] = ""
    elif mutation == "face_extra_vertex_ids":
        source["faces"][0]["mesh_vertex_indices"] = [0, 1, 2]
    elif mutation == "face_missing_edges":
        del source["faces"][0]["edge_ids"]
    elif mutation == "duplicate_face":
        source["faces"].append(deepcopy(source["faces"][0]))
    elif mutation == "duplicate_edge_id":
        source["faces"][0]["edge_ids"] = ["edge_0", "edge_0"]
    elif mutation == "bad_face_owner":
        source["faces"][0]["owner_id"] = None
    else:
        source["faces"][0]["edge_ids"] = [12]
    with pytest.raises(ValueError):
        group(candidates, source)
