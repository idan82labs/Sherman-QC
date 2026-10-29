"""Fixed paired progress check on invented CAD topology, not physical accuracy.

Frozen before first execution: four constructions, three proper rigid poses,
12 cases and 24 distinct candidate identities; 120 s / 512 MiB for the matrix.
The baseline is an explicit geometry-only connected-component ABLATION using
the same finite-axis/gap limits. It is not runtime replay, EXP10, an incumbent
detector benchmark, or a prospective physical comparison. Native identity and
topology completeness are stipulated synthetic inputs, never kernel-verified.
"""
import copy
import json
import math
import resource
import sys
import time

import numpy as np

from research.geometry.cad_root_grouping import group_cad_root_candidates


FAMILIES = (
    "separate_owners_collinear",
    "same_owner_global_flange_connection_only",
    "coincident_geometry_separate_owners",
    "connected_local_root_fragments",
)
POLICY = dict(unit="mm", frame_id="declared-cad-frame", max_axis_angle=0.02,
              max_axis_separation=0.25, max_segment_gap=2.0,
              max_pair_evaluations=100)
WALL_TARGET_SECONDS = 120.0
RSS_TARGET_BYTES = 512 * 1024**2


def poses():
    def rotation(axis, angle):
        u = np.asarray(axis, dtype=float)
        u /= np.linalg.norm(u)
        x, y, z = u
        skew = np.asarray([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        return np.eye(3)*math.cos(angle)+(1-math.cos(angle))*np.outer(u, u)+math.sin(angle)*skew

    return (
        (np.eye(3), np.zeros(3)),
        (rotation([0., 0., 1.], math.pi/3), np.asarray([7., -4., 2.])),
        (rotation([1., 2., 3.], 1.1), np.asarray([-13., 11., 19.])),
    )


def partition(groups):
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def fixture(family, case_number, pose):
    """Oracle partition is returned separately and never enters an engine call."""
    rotation, translation = pose
    identifiers = [f"c{2*case_number:03d}", f"c{2*case_number+1:03d}"]
    owners = ["owner0", "owner1"] if family in (
        "separate_owners_collinear", "coincident_geometry_separate_owners"
    ) else ["owner0", "owner0"]
    spans = [(0., 10.), (11., 21.)]
    if family == "coincident_geometry_separate_owners":
        spans = [(0., 10.), (0., 10.)]
    if family == "connected_local_root_fragments":
        spans = [(0., 10.), (10., 20.)]
    candidates = []
    for i, (start, end) in enumerate(spans):
        candidates.append(dict(
            candidate_id=identifiers[i], owner_id=owners[i], face_ids=[f"f{i}"],
            axis_start=(rotation@np.asarray([start, 0., 0.])+translation).tolist(),
            axis_end=(rotation@np.asarray([end, 0., 0.])+translation).tolist(),
        ))
    faces = [dict(face_id="f0", owner_id=owners[0], edge_ids=["e0"]),
             dict(face_id="f1", owner_id=owners[1], edge_ids=["e1"])]
    if family == "same_owner_global_flange_connection_only":
        # This third face connects the WHOLE source body, but no candidate
        # nominates it as local forming/root support. It must not bridge tabs.
        faces.append(dict(face_id="f2", owner_id="owner0", edge_ids=["e0", "e1"]))
    if family == "connected_local_root_fragments":
        faces[0]["edge_ids"] = ["joint"]
        faces[1]["edge_ids"] = ["joint"]
    topology = dict(source_id=f"native-fixture-{case_number:02d}", status="COMPLETE", faces=faces)
    oracle = partition([identifiers]) if family == "connected_local_root_fragments" else partition(
        [[identifiers[0]], [identifiers[1]]])
    return candidates, topology, oracle


def geometry_compatible(first, second):
    """Independent arithmetic for the declared geometry-only ablation.

    The fixtures are far from thresholds; this is not a floating-point boundary
    oracle. No production helper, grouping output or truth partition is read.
    """
    p = np.asarray([first["axis_start"], first["axis_end"]], dtype=float)
    q = np.asarray([second["axis_start"], second["axis_end"]], dtype=float)
    lengths = [float(np.linalg.norm(p[1]-p[0])), float(np.linalg.norm(q[1]-q[0]))]
    a, b = (p[1]-p[0])/lengths[0], (q[1]-q[0])/lengths[1]
    angle = math.acos(float(np.clip(abs(a@b), 0., 1.)))
    offsets = [p[0]-q[0], p[1]-q[0], q[0]-p[0], q[1]-p[0]]
    separation = max(float(np.linalg.norm(np.cross(offset, direction)))
                     for offset, direction in zip(offsets, (b, b, a, a)))
    gaps = []
    for direction in (a, b):
        # Use one origin for BOTH projected intervals, in each direction.
        projections_p = (p-p[0])@direction
        projections_q = (q-p[0])@direction
        gaps.append(max(0., float(np.min(projections_q)-np.max(projections_p)),
                        float(np.min(projections_p)-np.max(projections_q))))
    magnitude = max(float(np.max(np.abs(p))), float(np.max(np.abs(q))))
    extent = max(*lengths, *(float(np.linalg.norm(v)) for v in offsets))
    guard = 512*float(np.finfo(float).eps)*(magnitude+extent)
    return (angle+512*float(np.finfo(float).eps) < POLICY["max_axis_angle"]
            and separation+guard < POLICY["max_axis_separation"]
            and max(gaps)+guard < POLICY["max_segment_gap"])


def geometry_only_ablation(candidates, topology):
    """Both lanes receive the same native records; this ablation ignores them."""
    del topology
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    names = sorted(by_id)
    adjacency = {name: set() for name in names}
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if geometry_compatible(by_id[first], by_id[second]):
                adjacency[first].add(second)
                adjacency[second].add(first)
    remaining = set(names)
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group, todo = {seed}, [seed]
        while todo:
            current = todo.pop()
            found = remaining & adjacency[current]
            remaining.difference_update(found)
            group.update(found)
            todo.extend(sorted(found))
        groups.append(group)
    return partition(groups)


def scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["observed_state"] == "UNKNOWN"
    assert result["physical_geometry_is_verified"] is False
    assert result["event_identity_is_verified"] is False
    assert result["policy"]["no_convex_hull_span_or_geometry_average"] is True
    assert result["policy"]["any_unresolved_component_clears_all_groups"] is True
    assert result["policy"]["connectivity_scope"] == "GRAPH_INDUCED_ONLY_BY_NOMINATED_ROOT_FACES_WITHIN_OWNER"
    for group in result["groups"]:
        assert group["observed_state"] == "UNKNOWN"
        assert group["event_identity_is_verified"] is False


def preserved_candidates(result, candidates):
    assert result["input_candidates"] == sorted(candidates, key=lambda c: c["candidate_id"])
    grouped_ids = [name for group in result["groups"] for name in group["candidate_ids"]]
    names = sorted(c["candidate_id"] for c in candidates)
    assert sorted(grouped_ids) == names
    assert len(grouped_ids) == len(set(grouped_ids))
    assert sorted(result["candidate_to_group"]) == names
    original_segments = {c["candidate_id"]: (c["axis_start"], c["axis_end"]) for c in candidates}
    returned_segments = {segment["candidate_id"]: (segment["axis_start"], segment["axis_end"])
                         for group in result["groups"] for segment in group["member_segments"]}
    assert returned_segments == original_segments


def test_fixed_twelve_case_paired_partition_progress_and_candidate_conservation():
    started = time.perf_counter()
    comparisons = []
    all_identifiers = []
    for family_number, family in enumerate(FAMILIES):
        for pose_number, pose in enumerate(poses()):
            rotation, _ = pose
            assert np.allclose(rotation.T@rotation, np.eye(3), rtol=0., atol=1e-14)
            assert abs(float(np.linalg.det(rotation))-1.) < 1e-14
            case_number = 3*family_number+pose_number
            candidates, topology, oracle = fixture(family, case_number, pose)
            pristine_candidates, pristine_topology = copy.deepcopy(candidates), copy.deepcopy(topology)
            baseline = geometry_only_ablation(candidates, topology)
            result = group_cad_root_candidates(candidates, topology, **POLICY)
            scope(result)
            assert candidates == pristine_candidates and topology == pristine_topology
            assert result["status"] == "CAD_ROOT_GROUPING_HYPOTHESES"
            assert result["search_complete_within_declared_policy"]
            assert result["reasons"] == []
            actual = partition(group["candidate_ids"] for group in result["groups"])
            assert actual == oracle
            assert result["group_hypothesis_count"] == len(oracle)
            preserved_candidates(result, candidates)
            names = sorted(c["candidate_id"] for c in candidates)
            assert sorted(name for group in baseline for name in group) == names
            assert len(baseline) == 1  # Frozen predicted contraction in all four constructions.
            assert (baseline == oracle) is (family == "connected_local_root_fragments")
            all_identifiers.extend(names)
            comparisons.append(dict(
                family=family, pose=pose_number, candidate_ids=names,
                oracle_partition=oracle, geometry_only_partition=baseline,
                root_grouping_partition=actual, baseline_exact_partition=baseline == oracle,
                root_grouping_exact_partition=actual == oracle,
                source_candidates_preserved=True,
            ))
    assert len(comparisons) == 12
    assert len(all_identifiers) == len(set(all_identifiers)) == 24
    assert sum(c["baseline_exact_partition"] for c in comparisons) == 3
    assert sum(c["root_grouping_exact_partition"] for c in comparisons) == 12
    for family in FAMILIES:
        rows = [c for c in comparisons if c["family"] == family]
        assert len(rows) == 3
        assert all(c["root_grouping_exact_partition"] for c in rows)
        assert sum(c["baseline_exact_partition"] for c in rows) == (3 if family == FAMILIES[-1] else 0)
    # Lifetime high-water RSS is not a per-call delta or native-kernel benchmark.
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(raw_rss if sys.platform == "darwin" else raw_rss*1024)
    elapsed = time.perf_counter()-started
    assert elapsed <= WALL_TARGET_SECONDS
    assert peak_bytes <= RSS_TARGET_BYTES
    receipt = dict(
        comparison="FIXED_GEOMETRY_ONLY_GROUPING_ABLATION_NOT_RUNTIME_REPLAY_OR_EXP10",
        valid_declared_topology_cases=12, input_candidate_identities=24,
        geometry_only_exact_partitions=3, root_grouping_exact_partitions=12,
        disconnected_contractions_prevented=9, connected_control_partitions_preserved=3,
        physical_accuracy_established=False, topology_provenance_verified=False,
        matrix_wall_seconds=elapsed, process_lifetime_peak_rss_bytes=peak_bytes,
        wall_target_seconds=WALL_TARGET_SECONDS, rss_target_bytes=RSS_TARGET_BYTES,
        comparisons=comparisons,
    )
    print("CAD_ROOT_GROUPING_PROGRESS_RECEIPT " + json.dumps(receipt, sort_keys=True, allow_nan=False))


def test_unknown_topology_preserves_inputs_without_asserting_a_partition():
    candidates, topology, _ = fixture(FAMILIES[-1], 20, poses()[0])
    topology["status"] = "UNKNOWN"
    result = group_cad_root_candidates(candidates, topology, **POLICY)
    scope(result)
    assert result["input_candidates"] == candidates
    assert result["status"] == "ABSTAINED"
    assert result["reasons"] == ["TOPOLOGY_COMPLETENESS_UNKNOWN"]
    assert not result["search_complete_within_declared_policy"]
    assert result["groups"] == []
    assert result["candidate_to_group"] == {}
    assert result["group_hypothesis_count"] is None


def test_geometric_chaining_cannot_authorize_an_incompatible_native_root_group():
    candidates = [dict(candidate_id=f"chain{i}", owner_id="owner0", face_ids=[f"f{i}"],
                       axis_start=[0., 0.2*i, 0.], axis_end=[10., 0.2*i, 0.]) for i in range(3)]
    topology = dict(source_id="native-chain", status="COMPLETE", faces=[
        dict(face_id="f0", owner_id="owner0", edge_ids=["e01"]),
        dict(face_id="f1", owner_id="owner0", edge_ids=["e01", "e12"]),
        dict(face_id="f2", owner_id="owner0", edge_ids=["e12"]),
    ])
    assert geometry_compatible(candidates[0], candidates[1])
    assert geometry_compatible(candidates[1], candidates[2])
    assert not geometry_compatible(candidates[0], candidates[2])
    assert geometry_only_ablation(candidates, topology) == (("chain0", "chain1", "chain2"),)
    result = group_cad_root_candidates(candidates, topology, **POLICY)
    scope(result)
    assert result["input_candidates"] == candidates
    assert result["status"] == "ABSTAINED"
    assert result["reasons"] == ["ROOT_COMPONENT_AXIS_COMPATIBILITY_UNRESOLVED"]
    assert not result["search_complete_within_declared_policy"]
    assert result["groups"] == []
    assert result["candidate_to_group"] == {}
    assert result["group_hypothesis_count"] is None
