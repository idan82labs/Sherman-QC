"""Bounded fixed-seed mesh regions with unchanged all-row V5 qualification.

Unqualified research candidate. Triangle normals and slabs are proposal
heuristics, not uncertainty enclosures. No CAD, IO, registration, physical
surface identities or count inference. Original raw triangle unions survive.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import cad_guided_noisy_mesh_planes_v5 as _v5
from . import cad_guided_noisy_mesh_planes_v7 as _codec


@dataclass(frozen=True)
class SeededPlaneBudgetsV1:
    max_seeds: int = 32
    max_seed_selection_triangle_visits: int = 64_000_000
    max_proposal_raw_row_visits: int = 32_000_000
    max_proposal_triangle_visits: int = 64_000_000
    max_proposal_edge_visits: int = 96_000_000
    max_full_support_row_tests: int = 32_000_000
    max_triangle_memberships: int = 20_000_000
    max_raw_memberships: int = 32_000_000
    max_fit_evaluations: int = 32
    max_retained_array_bytes: int = 268_435_456
    max_metadata_bytes: int = 67_108_864
    max_working_array_bytes: int = 1_073_741_824


def _configuration(budgets):
    _v5._require(type(budgets) is SeededPlaneBudgetsV1, "INVALID_SEEDED_PLANE_BUDGET_TYPE")
    cap, maximum = asdict(budgets), asdict(SeededPlaneBudgetsV1())
    _v5._require(all(type(v) is int and 0 <= v <= maximum[k] for k, v in cap.items()),
                 "INVALID_SEEDED_PLANE_BUDGET")
    return cap


def _charge(stats, cap, key, count):
    _v5._charge(stats, cap, key, int(count), "max_" + key)


def _normals(points, triangles, chunk_rows):
    normals, centers, areas = np.empty((len(triangles), 3)), np.empty((len(triangles), 3)), np.empty(len(triangles))
    for start in range(0, len(triangles), chunk_rows):
        stop = min(len(triangles), start + chunk_rows)
        xyz = points[triangles[start:stop]]
        a, b = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
        cross = np.cross(a, b)
        lengths = np.linalg.norm(cross, axis=1)
        _v5._require(np.isfinite(lengths).all() and np.all(lengths > 0.), "UNRESOLVED_SEED_NORMAL")
        normals[start:stop] = cross / lengths[:, None]
        centers[start:stop] = xyz[:, 0] + (a + b) / 3.
        areas[start:stop] = lengths / 2.
    _v5._require(np.isfinite(normals).all() and np.isfinite(centers).all(), "NONFINITE_SEED_GEOMETRY")
    return normals, centers, areas


def _schedule(normals, centers, areas, cap, stats):
    """Freeze before region tests; alternate unoriented normal and spatial spread.

    Largest area first, then novelty primary on odd slots and centroid spread
    primary on even slots. Area and original triangle ID break ties. Parallel
    planes at distinct offsets are not collapsed. Finite/order dependent;
    no row-permutation invariance or physical coverage guarantee is claimed.
    """
    count = min(cap["max_seeds"], len(normals))
    _v5._require(count * len(normals) <= cap["max_seed_selection_triangle_visits"],
                 "SEED_SELECTION_PREFLIGHT_BUDGET_EXCEEDED")
    ids = np.arange(len(normals))
    chosen, available = [], np.ones(len(normals), dtype=bool)
    novelty, spread = np.ones(len(normals)), np.full(len(normals), np.inf)
    scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), np.finfo(float).tiny)
    for slot in range(count):
        _charge(stats, cap, "seed_selection_triangle_visits", len(normals))
        pool = ids[available]
        priorities = (areas,) if slot == 0 else (
            (novelty, spread, areas) if slot % 2 else (spread, novelty, areas))
        # Linear stable lexicographic maximum: no per-slot O(M log M) sort.
        for values in priorities:
            pool = pool[values[pool] == np.max(values[pool])]
        seed = int(pool[0])
        chosen.append(seed)
        available[seed] = False
        novelty = np.minimum(novelty, np.maximum(0., 1. - np.abs(normals @ normals[seed])))
        spread = np.minimum(spread, np.linalg.norm((centers - centers[seed]) / scale, axis=1))
    return chosen


def _region(points, triangles, epsilon, normals, centers, seed, left, right, gate, cap, stats):
    """One fixed-plane slab and one seed-connected component; never refit/grow."""
    n, m = len(points), len(triangles)
    _charge(stats, cap, "proposal_raw_row_visits", n)
    normal, anchor = normals[seed], centers[seed]
    seed_error = float(np.max(epsilon[triangles[seed]]))
    delta = points - anchor
    distance = np.abs(delta @ normal)
    guard = 512 * np.finfo(float).eps * (
        float(np.max(np.abs(points))) + float(np.max(np.abs(anchor)))
        + float(np.max(np.linalg.norm(delta, axis=1))))
    _v5._require(np.isfinite(distance).all() and math.isfinite(guard), "NONFINITE_PROPOSAL_SLAB")
    # Seed-plane orientation error is deliberately NOT claimed to be covered.
    # This is an engineering proposal mask. Qualification uses original epsilon.
    allowed = distance <= epsilon + seed_error + guard
    _charge(stats, cap, "proposal_triangle_visits", m)
    dot = np.abs(normals @ normal)
    eligible = np.all(allowed[triangles], axis=1) & (
        dot >= math.cos(gate["max_normal_angle_radians"]) - 512 * np.finfo(float).eps)
    _v5._require(bool(eligible[seed]), "NUMERICAL_SEED_SELF_EXCLUSION")
    _charge(stats, cap, "proposal_edge_visits", len(left))
    edge = eligible[left] & eligible[right]
    graph = coo_matrix((np.ones(int(np.count_nonzero(edge)), dtype=np.uint8),
                        (left[edge], right[edge])), shape=(m, m)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    rows = np.flatnonzero(eligible & (labels == labels[seed]))
    return rows, dict(seed_triangle_row=seed, normal=normal.tolist(), anchor_mm=anchor.tolist(),
        seed_max_position_error_mm=seed_error, arithmetic_guard_mm=guard,
        compatible_triangle_count=int(np.count_nonzero(eligible)),
        seed_connected_triangle_count=len(rows), normal_is_uncertainty_bound=False,
        threshold_scope="FIXED_SEED_PROPOSAL_ONLY_NOT_POSITION_UNCERTAINTY_ENCLOSURE")


def extract_seeded_plane_proposals_v1(points, triangle_raw_rows, *, position_error_bounds_mm,
        qualification_policy=_v5.NoisyMeshPlanePolicyV5(), budgets=SeededPlaneBudgetsV1()):
    """Independent additive lane; not yet joined to V7 or any runtime adapter."""
    result = dict(schema_version="seeded_plane_proposals_v1", status="UNAVAILABLE", reasons=[],
        unit="mm", coordinate_frame="RAW_SCAN_UNREGISTERED", uncertainty_status="STIPULATED_UNCALIBRATED",
        production_use_allowed=False, accepted_registration=False, physical_plane_count=None,
        observed_bend_count=None, search_exhaustive=False, raw_coordinate_welding=False,
        residual_trimming=False, finite_support_representation="ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON",
        scheduled_seed_triangle_rows=[], seeds=[], regions=[], proposals=[], input_digests={},
        statistics=dict(seed_selection_triangle_visits=0, proposal_raw_row_visits=0,
            proposal_triangle_visits=0, proposal_edge_visits=0, full_support_row_tests=0,
            triangle_memberships=0, raw_memberships=0, fit_evaluations=0,
            retained_array_bytes=0, metadata_bytes=0, metadata_expanded_bytes=0,
            metadata_node_visits=0, metadata_array_bytes_hashed=0,
            selected_policy_evaluation_complete=False))
    stats, work = result["statistics"], _codec._Work()
    codec = _codec._Codec(work)
    try:
        cap = _configuration(budgets)
        gate, input_cap = _v5._configuration(qualification_policy, _v5.NoisyMeshPlaneBudgetsV5())
        result.update(budgets=cap, qualification_policy=gate,
            seed_schedule_policy="LARGEST_AREA_THEN_ALTERNATING_NORMAL_NOVELTY_CENTROID_SPREAD",
            qualification="UNCHANGED_V5_ALL_RAW_ID_FIXED_NORMAL_WITNESS_AND_CONDITIONAL_CONE",
            work_counter_scope="LOGICAL_SUPPORT_EVALUATIONS_NOT_EVERY_ARRAY_PASS",
            preparation_scope="V5_FIXED_INPUT_CAPS_BEFORE_WORKING_RESERVATION_AND_SEED_WORK",
            proposal_slab_policy="EPSILON_PER_ROW_PLUS_MAX_SEED_EPSILON_PLUS_ARITHMETIC_GUARD",
            proposal_normal_policy="OBSERVED_TRIANGLE_NORMALS_WITH_QUALIFICATION_ANGLE_AS_HEURISTIC")
        p, t, epsilon = _v5._inputs(points, triangle_raw_rows, position_error_bounds_mm, input_cap)
        n, m = len(p), len(t)
        stats.update(input_raw_rows=n, input_triangles=m,
            preparation_validated_raw_rows=n, preparation_validated_triangles=m,
            preparation_triangle_normals=0, preparation_adjacency_edge_records=0)
        # Conservative declared dense-array/CSR scratch reservation, not an OS
        # RSS certificate (Python/SciPy temporaries remain supervisor-bounded).
        working = 160*n + 320*m + cap["max_retained_array_bytes"]
        _v5._require(working <= cap["max_working_array_bytes"], "WORKING_ARRAY_RESERVATION_EXCEEDED")
        stats["working_array_reservation_bytes"] = working
        result["input_digests"] = dict(points_sha256=_v5._digest(p), triangle_raw_rows_sha256=_v5._digest(t),
            position_error_bounds_mm_sha256=_v5._digest(epsilon))
        count = min(cap["max_seeds"], m)
        for field, visits in (("proposal_raw_row_visits", count*n),
                              ("proposal_triangle_visits", count*m),
                              ("seed_selection_triangle_visits", count*m)):
            _v5._require(visits <= cap["max_" + field], field.upper() + "_PREFLIGHT_EXCEEDED")
        graph, left, right = _v5._adjacency(t, n, stats)
        stats["preparation_adjacency_edge_records"] = 3*m
        del graph
        _v5._require(count*len(left) <= cap["max_proposal_edge_visits"], "PROPOSAL_EDGE_PREFLIGHT_EXCEEDED")
        normals, centers, areas = _normals(p, t, input_cap["chunk_rows"])
        stats["preparation_triangle_normals"] = m
        schedule = _schedule(normals, centers, areas, cap, stats)
        result["scheduled_seed_triangle_rows"] = schedule
        result["unscheduled_seed_triangle_count"] = m - len(schedule)
        referenced = np.zeros(n, dtype=bool)
        referenced[t.reshape(-1)] = True
        unreferenced = _v5._retained(np.flatnonzero(~referenced), stats, cap)
        result["raw_input_accounting"] = dict(referenced_raw_row_count=int(np.count_nonzero(referenced)),
            unreferenced_raw_vertex_rows=unreferenced, unreferenced_disposition="RETAINED_NOT_TRIANGLE_SUPPORT")
        seen = {}
        for seed in schedule:
            rows, proposal = _region(p, t, epsilon, normals, centers, seed, left, right, gate, cap, stats)
            digest = _v5._digest(np.asarray(rows, dtype="<i8"))
            duplicate = next((i for i in seen.get(digest, [])
                              if np.array_equal(result["regions"][i]["triangle_rows"], rows)), None)
            if duplicate is not None:
                proposal.update(disposition="EXACT_SUPPORT_DUPLICATE", region_id=duplicate)
            else:
                _charge(stats, cap, "triangle_memberships", len(rows))
                rows = _v5._retained(rows, stats, cap)
                raw = _v5._support_rows(t, rows, stats, cap)
                analysis = _v5._fit(p, epsilon, raw, gate, cap, stats)
                identifier = len(result["regions"])
                record = dict(region_id=identifier, first_seed_triangle_row=seed, triangle_rows=rows,
                    raw_vertex_rows=raw, triangle_rows_sha256=digest,
                    raw_vertex_rows_sha256=_v5._digest(raw), analysis=analysis)
                _codec._append(result["regions"], record, stats, cap, codec)
                seen.setdefault(digest, []).append(identifier)
                proposal.update(disposition=analysis["status"], region_id=identifier)
                if analysis["status"] == "PROPOSED":
                    _codec._append(result["proposals"], dict(kind="SEEDED_REGION", record_id=identifier),
                                   stats, cap, codec)
            _codec._append(result["seeds"], proposal, stats, cap, codec)
        stats.update(selected_policy_evaluation_complete=True, unique_region_count=len(result["regions"]),
            completed_seed_count=len(result["seeds"]), qualified_region_count=len(result["proposals"]))
        result.update(status="COMPLETE", reasons=[] if result["proposals"] else ["NO_QUALIFIED_SEEDED_REGION"])
        _codec._finalize(result, cap, work)
    except Exception as error:
        stats.update(selected_policy_evaluation_complete=False,
            discarded_seed_records=len(result["seeds"]), discarded_region_records=len(result["regions"]),
            discarded_proposals=len(result["proposals"]), metadata_node_visits=work.visits,
            metadata_array_bytes_hashed=work.hash_bytes,
            evidence_scope="ATTEMPTED_WORK_ONLY_ALL_PROPOSALS_INVALIDATED")
        result.update(status="UNAVAILABLE", reasons=[str(error) or type(error).__name__],
            seeds=[], regions=[], proposals=[], raw_input_accounting=None)
    return result
