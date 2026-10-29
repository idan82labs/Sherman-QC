"""Bounded declared-plane correspondence search; candidates, NOT registration.

All supports and normal cones are unverified declarations. A predeclared bank,
scan-triplet and stratified CAD-pair/assignment schedule is explicitly incomplete.
Finite raw support verification belongs to the downstream owned-array wrapper.
Only standard-library modules and NumPy are imported; no IO/native/app startup.
"""

from dataclasses import dataclass, fields
import hashlib
import itertools
import json
import math

import numpy as np


@dataclass(frozen=True)
class PlaneEquationV5:
    """Unit n with n dot (x-anchor_mm)=offset_from_anchor_mm, in declared mm."""
    plane_id: int
    normal: tuple
    anchor_mm: tuple
    offset_from_anchor_mm: float
    normal_angle_bound_radians: float = 0.


@dataclass(frozen=True)
class ObservedPlaneV5:
    equation: PlaneEquationV5
    support_count: int
    support_rms_mm: float
    support_centroid_mm: tuple
    proposal_kind: str = "DECLARED"


@dataclass(frozen=True)
class PlaneSearchPolicyV5:
    min_scan_support_count: int = 3
    max_scan_support_rms_mm: float = .1
    min_normal_condition_ratio: float = .02
    max_normal_alignment_extra_radians: float = .02
    max_intersection_backward_residual_mm: float = 1e-8
    pair_angle_numeric_guard_radians: float = 1e-10


@dataclass(frozen=True)
class PlaneSearchBudgetsV5:
    max_scan_planes: int = 65_552
    max_cad_planes: int = 2048
    max_bank_planes: int = 128
    max_scan_triplets: int = 256
    max_scan_partner_tests: int = 1_048_576
    max_cad_pair_records: int = 2_096_128
    max_pair_join_tests: int = 32_000_000
    max_cad_pair_samples_per_triplet: int = 12
    max_unsigned_per_scan_triplet: int = 4
    max_compatible_rows_per_triplet: int = 49_104
    max_unsigned_assignments: int = 1024
    max_signed_assignments: int = 8192
    max_report_bytes: int = 67_108_864
    chunk_pair_rows: int = 65_536


class _Unavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unavailable(reason)


def _real(value, *, low=0., high=1e6):
    _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
             "INVALID_DECLARED_PLANE_SCALAR")
    return float(value)


def _vector(value, *, maximum):
    _require(type(value) is tuple and len(value) == 3, "INVALID_DECLARED_PLANE_VECTOR")
    return tuple(_real(item, low=-maximum, high=maximum) for item in value)


def _equation(value):
    _require(type(value) is PlaneEquationV5, "INVALID_PLANE_EQUATION_TYPE")
    identifier = value.plane_id
    _require(type(identifier) is int and 0 <= identifier < 2**63, "INVALID_PLANE_ID")
    normal = _vector(value.normal, maximum=1.+1e-8)
    anchor = _vector(value.anchor_mm, maximum=1e6)
    offset = _real(value.offset_from_anchor_mm, low=-1e6)
    alpha = _real(value.normal_angle_bound_radians, high=math.pi/4.)
    length = math.sqrt(math.fsum(item*item for item in normal))
    _require(abs(length-1.) <= 1e-8, "DECLARED_PLANE_NORMAL_NOT_UNIT")
    # Normalize BOTH coefficients of the anchored equation. The anchor is not
    # a matching centroid and need not itself be on the represented plane.
    return dict(plane_id=identifier, normal=tuple(item/length for item in normal),
        anchor_mm=anchor, offset_from_anchor_mm=offset/length,
        normal_angle_bound_radians=alpha)


def _observed(value):
    _require(type(value) is ObservedPlaneV5, "INVALID_OBSERVED_PLANE_TYPE")
    equation = _equation(value.equation)
    count = value.support_count
    _require(type(count) is int and 0 <= count <= 1_000_000, "INVALID_DECLARED_SUPPORT_COUNT")
    kind = value.proposal_kind
    _require(type(kind) is str and kind in ("CORE", "CLOSURE", "DECLARED"), "INVALID_PROPOSAL_KIND")
    return dict(equation=equation, support_count=count,
        support_rms_mm=_real(value.support_rms_mm, high=1000.),
        support_centroid_mm=_vector(value.support_centroid_mm, maximum=1e6), proposal_kind=kind)


def _configuration(policy, budgets):
    _require(type(policy) is PlaneSearchPolicyV5, "INVALID_PLANE_SEARCH_POLICY_TYPE")
    _require(type(budgets) is PlaneSearchBudgetsV5, "INVALID_PLANE_SEARCH_BUDGET_TYPE")
    gate = {item.name: getattr(policy, item.name) for item in fields(policy)}
    _require(type(gate["min_scan_support_count"]) is int
             and 3 <= gate["min_scan_support_count"] <= 1_000_000, "INVALID_SUPPORT_POLICY")
    for name, maximum in (("max_scan_support_rms_mm", 1000.),
                          ("min_normal_condition_ratio", 1.),
                          ("max_normal_alignment_extra_radians", .25),
                          ("max_intersection_backward_residual_mm", .01),
                          ("pair_angle_numeric_guard_radians", 1e-6)):
        gate[name] = _real(gate[name], high=maximum)
        _require(gate[name] > 0., "INVALID_POSITIVE_SEARCH_POLICY")
    maximum = PlaneSearchBudgetsV5()
    cap = {item.name: getattr(budgets, item.name) for item in fields(budgets)}
    _require(all(type(value) is int and 0 <= value <= getattr(maximum, key)
                 for key, value in cap.items()) and cap["chunk_pair_rows"] > 0,
             "INVALID_PLANE_SEARCH_BUDGET")
    return gate, cap


def _base():
    return dict(schema_version="plane_search_v5", status="UNAVAILABLE", reasons=[],
        scope="DECLARED_PLANE_NONEXHAUSTIVE_INITIALIZATION_ONLY",
        support_provenance="UNVERIFIED_DECLARATIONS", raw_support_provenance_verified=False,
        normal_cone_status="STIPULATED_UNCALIBRATED", translation_uncertainty="UNAVAILABLE_NO_OFFSET_ERROR_MODEL",
        finite_support_match_verified=False, accepted_registration=False, production_use_allowed=False,
        observed_bend_count=None, physical_plane_count=None, search_exhaustive=False,
        global_pose_uniqueness="NOT_ESTABLISHED", selected_policy_evaluation_complete=False,
        scan_planes=[], cad_planes=[], scan_bank=[], anchor_schedules=[], scan_triplets=[],
        unsigned_assignments=[], hypotheses=[], input_digests={},
        statistics=dict(input_scan_planes=0, input_cad_planes=0, eligible_scan_planes=0,
            bank_plane_count=0, unscheduled_eligible_scan_planes=0, scan_partner_tests=0,
            scan_triplets_scheduled=0, scan_triplets_evaluated=0, cad_pair_records=0,
            pair_join_tests=0, indexed_pair_window_occurrences=0, scheduled_pair_occurrences=0,
            unscheduled_index_window_occurrences=0, tested_incompatible_pair_occurrences=0,
            compatible_unsigned_occurrences=0, unscheduled_compatible_unsigned_occurrences=0,
            unsigned_assignments_scheduled=0, signed_assignments_planned=0,
            signed_assignments_attempted=0, signed_assignments_evaluated=0,
            proposed_hypotheses=0, rejected_hypotheses=0,
            metadata_admission_bytes=0))


def _charge(result, cap, field, count, maximum):
    _require(type(count) is int and count >= 0, "INVALID_SEARCH_WORK_CHARGE")
    stats = result["statistics"]
    _require(stats[field]+count <= cap[maximum], maximum.upper()+"_EXCEEDED")
    stats[field] += count


def _append(result, cap, target, record):
    size = len(json.dumps(record, allow_nan=False, separators=(",", ":")).encode())
    _charge(result, cap, "metadata_admission_bytes", size, "max_report_bytes")
    target.append(record)


def _fail(result, reason):
    stats = result["statistics"]
    stats["proposals_invalidated_on_failure"] = stats["proposed_hypotheses"]
    stats["proposed_hypotheses"] = 0
    for row in result["hypotheses"]:
        row["status_before_failure"] = row["status"]
        row.update(status="INVALIDATED_BY_FATAL_SEARCH_FAILURE", rotation=None, translation_mm=None)
    for row in result["scan_triplets"]:
        if row["status"] == "EVALUATING":
            row["status_before_failure"] = row["status"]
            row["status"] = "INTERRUPTED_BY_FATAL_SEARCH_FAILURE"
    result.update(status="UNAVAILABLE", reasons=[str(reason) or type(reason).__name__],
                  selected_policy_evaluation_complete=False, usable_hypotheses=False,
                  evidence_scope="ATTEMPTED_WORK_ONLY_ALL_USABLE_POSES_INVALIDATED")
    return result


def _publish_cad_usage(result, usage):
    """Export charged participation even when later numerical work failed."""
    for index, record in enumerate(result["cad_planes"]):
        record.update(scheduled_pair_occurrences=int(usage[index, 0]),
            third_plane_test_occurrences=int(usage[index, 1]), scheduled_unsigned_occurrences=int(usage[index, 2]),
            disposition="USED_IN_SCHEDULED_UNSIGNED_ASSIGNMENT" if usage[index, 2] else
                "PAIR_JOIN_REACHED_NOT_IN_SCHEDULED_UNSIGNED_ASSIGNMENT" if np.any(usage[index]) else
                "NOT_REACHED_BY_DECLARED_SCHEDULE")


def _report(result, maximum):
    try:
        _require(len(json.dumps(result, allow_nan=False, separators=(",", ":")).encode()) <= maximum,
                 "SEARCH_REPORT_BUDGET_EXCEEDED")
        return result
    except Exception:
        return dict(schema_version="plane_search_v5", status="UNAVAILABLE",
            reasons=["SEARCH_REPORT_UNAVAILABLE_NO_PREFIX"], hypotheses=[], usable_hypotheses=False,
            selected_policy_evaluation_complete=False, search_exhaustive=False,
            accepted_registration=False, production_use_allowed=False, observed_bend_count=None,
            discarded_hypotheses=len(result.get("hypotheses", [])),
            discarded_scan_records=len(result.get("scan_planes", [])),
            discarded_cad_records=len(result.get("cad_planes", [])),
            discarded_unsigned_assignments=len(result.get("unsigned_assignments", [])),
            attempted_statistics=dict(result["statistics"]),
            evidence_scope="NO_PARTIAL_REPORT_OR_POSE_PUBLISHED")


def _strata(length, maximum):
    """Fixed rank strata including endpoints; never an exhausted first prefix."""
    count = min(length, maximum)
    if count == 0:
        return []
    if count == 1:
        return [length//2]
    return [index*(length-1)//(count-1) for index in range(count)]


def _bank(scan, maximum):
    if not scan or maximum == 0:
        return []
    normals = np.asarray([row["equation"]["normal"] for row in scan])
    centers = np.asarray([row["support_centroid_mm"] for row in scan])
    alphas = np.asarray([row["equation"]["normal_angle_bound_radians"] for row in scan])
    ids = np.asarray([row["equation"]["plane_id"] for row in scan], dtype=np.int64)
    kinds = sorted({row["proposal_kind"] for row in scan})
    kind_rows = {kind: np.asarray([row["proposal_kind"] == kind for row in scan]) for kind in kinds}
    available = np.ones(len(scan), dtype=bool)
    novelty, spread = np.ones(len(scan)), np.full(len(scan), np.inf)
    scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), np.finfo(float).tiny)
    selected = []
    for slot in range(min(len(scan), maximum)):
        pool = np.flatnonzero(available & kind_rows[kinds[slot % len(kinds)]])
        if not len(pool):
            pool = np.flatnonzero(available)
        order = np.lexsort((ids[pool], alphas[pool], -spread[pool], -novelty[pool]))
        chosen = int(pool[order[0]])
        selected.append(chosen)
        available[chosen] = False
        novelty = np.minimum(novelty, 1.-np.abs(normals@normals[chosen]))
        spread = np.minimum(spread, np.linalg.norm(centers-centers[chosen], axis=1)/scale)
    return selected


def _angle(first, second):
    return np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1), np.abs(np.sum(first*second, axis=-1)))


def _interval(angle, first_alpha, second_alpha, guard):
    radius = first_alpha+second_alpha+guard
    return np.maximum(0., angle-radius), np.minimum(math.pi/2., angle+radius)


def _overlap(low, high, other_low, other_high):
    return (low <= other_high) & (other_low <= high)


def _triplet(equations, gate):
    normals = np.asarray([row["normal"] for row in equations])
    singular = np.linalg.svd(normals, compute_uv=False)
    _require(np.isfinite(singular).all() and singular[0] > 0., "NONFINITE_TRIPLET_SYSTEM")
    ratio = float(singular[-1]/singular[0])
    eta = math.sqrt(math.fsum((2*math.sin(row["normal_angle_bound_radians"]/2.))**2 for row in equations))
    result = dict(status="REJECTED", reasons=[], normal_condition_ratio=ratio,
        declared_normal_perturbation_frobenius_bound=eta,
        normal_independence_margin=float(singular[-1])-eta, intersection_mm=None,
        backward_residual_max_mm=None)
    if ratio < gate["min_normal_condition_ratio"]:
        result["reasons"] = ["NOMINAL_TRIPLET_ILL_CONDITIONED"]
        return result
    if singular[-1] <= eta:
        result["reasons"] = ["INDEPENDENCE_UNRESOLVED_WITH_DECLARED_NORMAL_CONES"]
        return result
    anchor = np.asarray(equations[0]["anchor_mm"])
    rhs = np.asarray([row["offset_from_anchor_mm"]+np.dot(row["normal"], np.asarray(row["anchor_mm"])-anchor)
                      for row in equations])
    local = np.linalg.solve(normals, rhs)
    intersection = anchor+local
    residuals = np.asarray([np.dot(row["normal"], intersection-np.asarray(row["anchor_mm"]))
                            -row["offset_from_anchor_mm"] for row in equations])
    _require(np.isfinite(intersection).all() and np.isfinite(residuals).all(), "NONFINITE_TRIPLET_INTERSECTION")
    backward = float(np.max(np.abs(residuals)))
    result.update(intersection_mm=intersection.tolist(), backward_residual_max_mm=backward)
    if backward > gate["max_intersection_backward_residual_mm"]:
        result["reasons"] = ["INTERSECTION_BACKWARD_RESIDUAL_EXCEEDS_GATE"]
        return result
    result.update(status="AVAILABLE", reasons=[])
    return result


def _scan_triplets(bank, result, gate, cap):
    count = len(bank)
    if count < 3:
        return []
    planned = count*math.comb(count-1, 2)
    _require(planned <= cap["max_scan_partner_tests"], "SCAN_PARTNER_PREFLIGHT_BUDGET_EXCEEDED")
    _require(min(math.comb(count, 3), 2*count) <= cap["max_scan_triplets"], "SCAN_TRIPLET_PREFLIGHT_BUDGET_EXCEEDED")
    normals = np.asarray([row["equation"]["normal"] for row in bank])
    alpha = np.asarray([row["equation"]["normal_angle_bound_radians"] for row in bank])
    selected, lookup = [], {}
    for anchor in range(count):
        others = [index for index in range(count) if index != anchor]
        pairs = np.asarray(list(itertools.combinations(others, 2)), dtype=np.int64)
        _charge(result, cap, "scan_partner_tests", len(pairs), "max_scan_partner_tests")
        determinant = np.abs(np.cross(normals[pairs[:, 0]], normals[pairs[:, 1]])@normals[anchor])
        widths = alpha[pairs[:, 0]]+alpha[pairs[:, 1]]+alpha[anchor]
        order = np.lexsort((pairs[:, 1], pairs[:, 0], widths, -determinant))
        ids = []
        for candidate in order[:min(2, len(order))]:
            key = tuple(sorted((anchor, int(pairs[candidate, 0]), int(pairs[candidate, 1]))))
            if key not in lookup:
                index = len(selected)
                _require(index < cap["max_scan_triplets"], "SCAN_TRIPLET_BUDGET_EXCEEDED")
                lookup[key] = index
                equations = [bank[item]["equation"] for item in key]
                analysis = _triplet(equations, gate)
                record = dict(scan_triplet_id=index, scan_plane_ids=[row["plane_id"] for row in equations],
                    origin_anchor_plane_ids=[], priority_abs_normal_determinant=float(determinant[candidate]),
                    analysis=analysis, status="NOT_EVALUATED", cad_pair_records=[],
                    indexed_pair_window_count=0, scheduled_pair_window_positions=[],
                    compatible_unsigned_count=0, scheduled_unsigned_ids=[],
                    unscheduled_compatible_unsigned_count=0)
                _append(result, cap, result["scan_triplets"], record)
                selected.append((key, equations, analysis))
            index = lookup[key]
            result["scan_triplets"][index]["origin_anchor_plane_ids"].append(bank[anchor]["equation"]["plane_id"])
            ids.append(index)
        _append(result, cap, result["anchor_schedules"], dict(scan_plane_id=bank[anchor]["equation"]["plane_id"],
            partner_pair_count=len(pairs), scheduled_scan_triplet_ids=ids,
            unscheduled_partner_pair_count=len(pairs)-len(ids)))
    result["statistics"]["scan_triplets_scheduled"] = len(selected)
    return selected


def _pair_table(cad, cap, result):
    count = len(cad)
    planned = math.comb(count, 2)
    _require(planned <= cap["max_cad_pair_records"], "CAD_PAIR_INDEX_PREFLIGHT_BUDGET_EXCEEDED")
    left, right = np.empty(planned, dtype=np.int32), np.empty(planned, dtype=np.int32)
    angles, radii = np.empty(planned), np.empty(planned)
    normals = np.asarray([row["normal"] for row in cad])
    alpha = np.asarray([row["normal_angle_bound_radians"] for row in cad])
    cursor = 0
    for first in range(count-1):
        length = count-first-1
        left[cursor:cursor+length] = first
        right[cursor:cursor+length] = np.arange(first+1, count)
        cursor += length
    for start in range(0, planned, cap["chunk_pair_rows"]):
        stop = min(planned, start+cap["chunk_pair_rows"])
        a, b = left[start:stop], right[start:stop]
        angles[start:stop] = _angle(normals[a], normals[b])
        radii[start:stop] = alpha[a]+alpha[b]
    _require(np.isfinite(angles).all() and np.isfinite(radii).all(), "NONFINITE_CAD_PAIR_INDEX")
    order = np.argsort(angles, kind="stable")
    result["statistics"]["cad_pair_records"] = planned
    return dict(left=left, right=right, angles=angles, radii=radii, order=order,
                sorted_angles=angles[order], max_radius=float(np.max(radii)) if planned else 0.)


def _pair_indices(first, second, count):
    low, high = np.minimum(first, second), np.maximum(first, second)
    return low*(2*count-low-1)//2+(high-low-1)


def _bits(mask):
    return np.packbits(mask, bitorder="little").tobytes().hex()


def _join(equations, cad, table, row, result, gate, cap, cad_usage):
    n = len(cad)
    guard = gate["pair_angle_numeric_guard_radians"]
    normals = np.asarray([item["normal"] for item in equations])
    alpha = np.asarray([item["normal_angle_bound_radians"] for item in equations])
    scan_angles = [float(_angle(normals[a], normals[b])) for a, b in ((0, 1), (0, 2), (1, 2))]
    intervals = [_interval(angle, alpha[a], alpha[b], guard)
                 for angle, (a, b) in zip(scan_angles, ((0, 1), (0, 2), (1, 2)))]
    lower, upper = intervals[0]
    start = int(np.searchsorted(table["sorted_angles"], lower-table["max_radius"]-guard, side="left"))
    stop = int(np.searchsorted(table["sorted_angles"], upper+table["max_radius"]+guard, side="right"))
    window = stop-start
    positions = _strata(window, cap["max_cad_pair_samples_per_triplet"])
    maximum_compatible = len(positions)*2*max(0, n-2)
    _require(maximum_compatible <= cap["max_compatible_rows_per_triplet"], "CAD_JOIN_BUFFER_PREFLIGHT_BUDGET_EXCEEDED")
    planned_tests = len(positions)*(1+4*max(0, n-2))
    _require(result["statistics"]["pair_join_tests"]+planned_tests <= cap["max_pair_join_tests"],
             "CAD_JOIN_PREFLIGHT_BUDGET_EXCEEDED")
    compatible = np.empty((maximum_compatible, 3), dtype=np.int32)
    used = 0
    stats = result["statistics"]
    stats["indexed_pair_window_occurrences"] += window
    stats["scheduled_pair_occurrences"] += len(positions)
    stats["unscheduled_index_window_occurrences"] += window-len(positions)
    row.update(indexed_pair_window_count=window, scheduled_pair_window_positions=positions,
        scan_pair_angles_radians=scan_angles,
        scan_pair_intervals_radians=[[float(a), float(b)] for a, b in intervals],
        index_window_scope="CONSERVATIVE_ANGLE_WINDOW_NOT_ALL_PAIRS_VERIFIED_COMPATIBLE")
    for position in positions:
        pair_index = int(table["order"][start+position])
        a, b = int(table["left"][pair_index]), int(table["right"][pair_index])
        cad_usage[a, 0] += 1
        cad_usage[b, 0] += 1
        angle, radius = float(table["angles"][pair_index]), float(table["radii"][pair_index])
        cad_low, cad_high = max(0., angle-radius-guard), min(math.pi/2., angle+radius+guard)
        _charge(result, cap, "pair_join_tests", 1, "max_pair_join_tests")
        passed = bool(_overlap(lower, upper, cad_low, cad_high))
        record = dict(cad_pair_ids=[cad[a]["plane_id"], cad[b]["plane_id"]],
            index_window_position=position, cad_angle_radians=angle,
            cad_angle_interval_radians=[cad_low, cad_high],
            status="PAIR_INTERVAL_COMPATIBLE" if passed else "PAIR_INTERVAL_INCOMPATIBLE",
            orientations=[])
        _append(result, cap, row["cad_pair_records"], record)
        if not passed:
            stats["tested_incompatible_pair_occurrences"] += 1
            continue
        for first, second in ((a, b), (b, a)):
            third = np.asarray([index for index in range(n) if index not in (first, second)], dtype=np.int64)
            _charge(result, cap, "pair_join_tests", 2*len(third), "max_pair_join_tests")
            cad_usage[third, 1] += 1
            ia, ib = _pair_indices(first, third, n), _pair_indices(second, third, n)
            a_low = np.maximum(0., table["angles"][ia]-table["radii"][ia]-guard)
            a_high = np.minimum(math.pi/2., table["angles"][ia]+table["radii"][ia]+guard)
            b_low = np.maximum(0., table["angles"][ib]-table["radii"][ib]-guard)
            b_high = np.minimum(math.pi/2., table["angles"][ib]+table["radii"][ib]+guard)
            reject_a = ~_overlap(*intervals[1], a_low, a_high)
            reject_b = ~_overlap(*intervals[2], b_low, b_high)
            keep = ~(reject_a | reject_b)
            selected = third[keep]
            _require(used+len(selected) <= maximum_compatible, "CAD_JOIN_BUFFER_ACCOUNTING_FAILURE")
            compatible[used:used+len(selected)] = np.column_stack((np.full(len(selected), first),
                np.full(len(selected), second), selected))
            used += len(selected)
            orientation = dict(assigned_first_two_cad_ids=[cad[first]["plane_id"], cad[second]["plane_id"]],
                third_plane_order="ALL_CAD_IDS_ASCENDING_EXCLUDING_ASSIGNED_PAIR",
                tested_third_count=len(third), compatible_third_count=len(selected),
                rejected_02_packbits_hex=_bits(reject_a), rejected_12_packbits_hex=_bits(reject_b),
                mask_bit_order="LITTLE_PADDING_BITS_IGNORED_AFTER_TESTED_COUNT")
            _append(result, cap, record["orientations"], orientation)
    row["compatible_unsigned_count"] = used
    stats["compatible_unsigned_occurrences"] += used
    selected_positions = _strata(used, cap["max_unsigned_per_scan_triplet"])
    row["scheduled_compatible_assignment_positions"] = selected_positions
    row["unscheduled_compatible_unsigned_count"] = used-len(selected_positions)
    stats["unscheduled_compatible_unsigned_occurrences"] += used-len(selected_positions)
    return compatible[selected_positions].copy()


def _signed(source, target, source_analysis, target_analysis, unsigned_id, result, gate, cap):
    ns = np.asarray([row["normal"] for row in source])
    nc = np.asarray([row["normal"] for row in target])
    allowed = np.asarray([a["normal_angle_bound_radians"]+b["normal_angle_bound_radians"]
                          +gate["max_normal_alignment_extra_radians"] for a, b in zip(source, target)])
    for signs_tuple in itertools.product((-1, 1), repeat=3):
        _charge(result, cap, "signed_assignments_attempted", 1, "max_signed_assignments")
        signs = np.asarray(signs_tuple)
        record = dict(hypothesis_index=len(result["hypotheses"]), unsigned_assignment_id=unsigned_id,
            status="EVALUATING", reasons=[], normal_signs=list(signs_tuple),
            rotation=None, translation_mm=None,
            correspondences=[dict(scan_plane_id=a["plane_id"], cad_plane_id=b["plane_id"], normal_sign=sign)
                             for a, b, sign in zip(source, target, signs_tuple)])
        _append(result, cap, result["hypotheses"], record)
        if source_analysis["status"] != "AVAILABLE" or target_analysis["status"] != "AVAILABLE":
            record["status"] = "REJECTED"
            record["reasons"] = ["TRIPLET_UNAVAILABLE", *source_analysis["reasons"], *target_analysis["reasons"]]
        else:
            signed_target = signs[:, None]*nc
            left, _, right = np.linalg.svd(ns.T@signed_target)
            correction = np.eye(3)
            correction[-1, -1] = 1. if np.linalg.det(right.T@left.T) >= 0. else -1.
            rotation = right.T@correction@left.T
            mapped = ns@rotation.T
            angles = np.arctan2(np.linalg.norm(np.cross(mapped, signed_target), axis=1),
                                np.sum(mapped*signed_target, axis=1))
            _require(np.isfinite(rotation).all() and np.isfinite(angles).all()
                     and np.allclose(rotation.T@rotation, np.eye(3), rtol=0., atol=1e-10)
                     and abs(float(np.linalg.det(rotation))-1.) <= 1e-10,
                     "NUMERICALLY_INVALID_PROPER_NORMAL_ALIGNMENT")
            record.update(normal_alignment_residuals_radians=angles.tolist(),
                          normal_alignment_allowances_radians=allowed.tolist())
            if np.any(angles > allowed):
                record["status"] = "REJECTED"
                record["reasons"] = ["NOMINAL_WAHBA_WITNESS_OUTSIDE_DECLARED_GATE"]
            else:
                translation = np.asarray(target_analysis["intersection_mm"])-rotation@np.asarray(source_analysis["intersection_mm"])
                _require(np.isfinite(translation).all(), "NONFINITE_INTRINSIC_INTERSECTION_POSE")
                record.update(status="PROPOSED", rotation=rotation.tolist(), translation_mm=translation.tolist(),
                    source_intersection_mm=source_analysis["intersection_mm"],
                    cad_intersection_mm=target_analysis["intersection_mm"],
                    translation_rule="CAD_INTERSECTION_MINUS_R_SCAN_INTERSECTION_NOT_PARTIAL_CENTROIDS")
        key = "proposed_hypotheses" if record["status"] == "PROPOSED" else "rejected_hypotheses"
        result["statistics"][key] += 1
        result["statistics"]["signed_assignments_evaluated"] += 1


def propose_plane_search_v5(scan_planes, cad_planes, *, unit_assumption,
        policy=PlaneSearchPolicyV5(), budgets=PlaneSearchBudgetsV5()):
    """Return every scheduled signed candidate/refusal and explicit omissions.

    Exactly typed immutable tuples are snapshotted/revalidated, not trusted as
    raw lineage. Policy quotas are incomplete search; unexpected hard-cap or
    numerical failures invalidate all usable poses. There is no physical IO.
    """
    result = _base()
    maximum_report = PlaneSearchBudgetsV5().max_report_bytes
    usage = None
    try:
        gate, cap = _configuration(policy, budgets)
        maximum_report = cap["max_report_bytes"]
        _require(type(unit_assumption) is str and 1 <= len(unit_assumption) <= 128 and unit_assumption.strip(),
                 "EXPLICIT_MM_UNIT_ASSUMPTION_REQUIRED")
        result.update(unit="mm", unit_assumption=unit_assumption, policy=gate, budgets=cap,
            bank_policy="PROPOSAL_KIND_ROUND_ROBIN_NORMAL_NOVELTY_CENTROID_SPREAD",
            scan_triplet_policy="TWO_MAX_ABS_NORMAL_DETERMINANT_PARTNER_PAIRS_PER_BANK_ANCHOR",
            cad_pair_policy="STRATIFIED_INDEX_WINDOW_POSITIONS_BOTH_ORIENTATIONS_ALL_THIRD_PLANES",
            unsigned_policy="STRATIFIED_COMPATIBLE_ASSIGNMENT_POSITIONS_PER_SCAN_TRIPLET",
            identity_ties="SORTED_IMPORT_LOCAL_IDS_NO_CASE_IDENTITY_FEATURES_NOT_ID_RENUMBERING_INVARIANT")
        _require(type(scan_planes) is tuple and type(cad_planes) is tuple, "IMMUTABLE_TYPED_PLANE_TABLES_REQUIRED")
        result["statistics"].update(input_scan_planes=len(scan_planes), input_cad_planes=len(cad_planes))
        _require(len(scan_planes) <= cap["max_scan_planes"] and len(cad_planes) <= cap["max_cad_planes"],
                 "PLANE_TABLE_PREFLIGHT_BUDGET_EXCEEDED")
        scan = sorted((_observed(row) for row in scan_planes), key=lambda row: row["equation"]["plane_id"])
        cad = sorted((_equation(row) for row in cad_planes), key=lambda row: row["plane_id"])
        _require(len({row["equation"]["plane_id"] for row in scan}) == len(scan)
                 and len({row["plane_id"] for row in cad}) == len(cad), "DUPLICATE_PLANE_ID_WITHIN_FRAME")
        for name, values in (("scan_table_sha256", scan), ("cad_table_sha256", cad)):
            result["input_digests"][name] = hashlib.sha256(json.dumps(values, sort_keys=True,
                allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        eligible, eligible_records = [], []
        for row in scan:
            reasons = []
            if row["support_count"] < gate["min_scan_support_count"]:
                reasons.append("INSUFFICIENT_DECLARED_SCAN_SUPPORT")
            if row["support_rms_mm"] > gate["max_scan_support_rms_mm"]:
                reasons.append("DECLARED_SCAN_RMS_EXCEEDS_GATE")
            record = dict(row, disposition="EXCLUDED_BY_DECLARED_SUPPORT_POLICY" if reasons
                else "NOT_SCHEDULED_IN_SCAN_BANK", reasons=reasons, bank_index=None)
            _append(result, cap, result["scan_planes"], record)
            if not reasons:
                eligible.append(row)
                eligible_records.append(record)
        for row in cad:
            _append(result, cap, result["cad_planes"], dict(row, disposition="NOT_REACHED_BY_DECLARED_SCHEDULE",
                scheduled_pair_occurrences=0, third_plane_test_occurrences=0, scheduled_unsigned_occurrences=0))
        selection = _bank(eligible, cap["max_bank_planes"])
        bank = [eligible[index] for index in selection]
        for bank_index, source_index in enumerate(selection):
            eligible_records[source_index].update(disposition="SCHEDULED_IN_SCAN_BANK", bank_index=bank_index)
        result["scan_bank"] = [row["equation"]["plane_id"] for row in bank]
        result["statistics"].update(eligible_scan_planes=len(eligible), bank_plane_count=len(bank),
                                   unscheduled_eligible_scan_planes=len(eligible)-len(bank))
        if len(bank) < 3 or len(cad) < 3:
            result.update(status="COMPLETE", reasons=["FEWER_THAN_THREE_PLANES_IN_DECLARED_SEARCH"],
                          selected_policy_evaluation_complete=True, usable_hypotheses=False)
            return _report(result, maximum_report)
        triplets = _scan_triplets(bank, result, gate, cap)
        # Worst-case scheduled assignment and pair-index bounds precede their
        # allocations/loops. A smaller hard budget does not select a good prefix.
        unsigned_maximum = len(triplets)*cap["max_unsigned_per_scan_triplet"]
        _require(unsigned_maximum <= cap["max_unsigned_assignments"]
                 and 8*unsigned_maximum <= cap["max_signed_assignments"], "ASSIGNMENT_PREFLIGHT_BUDGET_EXCEEDED")
        pair_maximum = math.comb(len(cad), 2)
        _require(pair_maximum <= cap["max_cad_pair_records"], "CAD_PAIR_INDEX_PREFLIGHT_BUDGET_EXCEEDED")
        per_triplet_pairs = min(pair_maximum, cap["max_cad_pair_samples_per_triplet"])
        worst_join = len(triplets)*per_triplet_pairs*(1+4*(len(cad)-2))
        _require(worst_join <= cap["max_pair_join_tests"], "PAIR_JOIN_WORST_CASE_PREFLIGHT_BUDGET_EXCEEDED")
        _require(2*per_triplet_pairs*(len(cad)-2) <= cap["max_compatible_rows_per_triplet"],
                 "CAD_JOIN_BUFFER_PREFLIGHT_BUDGET_EXCEEDED")
        table = _pair_table(cad, cap, result)
        usage = np.zeros((len(cad), 3), dtype=np.int64)
        for index, (_, source, analysis) in enumerate(triplets):
            row = result["scan_triplets"][index]
            if analysis["status"] != "AVAILABLE":
                row["status"] = "SCAN_TRIPLET_UNAVAILABLE_NO_CAD_JOIN"
                result["statistics"]["scan_triplets_evaluated"] += 1
                continue
            row["status"] = "EVALUATING"
            assignments = _join(source, cad, table, row, result, gate, cap, usage)
            for indices in assignments:
                target = [cad[int(item)] for item in indices]
                target_analysis = _triplet(target, gate)
                unsigned_id = len(result["unsigned_assignments"])
                _charge(result, cap, "unsigned_assignments_scheduled", 1, "max_unsigned_assignments")
                result["statistics"]["signed_assignments_planned"] += 8
                usage[indices, 2] += 1
                record = dict(unsigned_assignment_id=unsigned_id, scan_triplet_id=index,
                    scan_plane_ids=[item["plane_id"] for item in source],
                    cad_plane_ids=[item["plane_id"] for item in target], target_analysis=target_analysis)
                _append(result, cap, result["unsigned_assignments"], record)
                row["scheduled_unsigned_ids"].append(unsigned_id)
                _signed(source, target, analysis, target_analysis, unsigned_id, result, gate, cap)
            row["status"] = "DECLARED_PAIR_JOIN_AND_ASSIGNMENT_SCHEDULE_COMPLETE"
            result["statistics"]["scan_triplets_evaluated"] += 1
        stats = result["statistics"]
        _require(stats["signed_assignments_evaluated"] == stats["signed_assignments_planned"]
                 and stats["scan_triplets_evaluated"] == stats["scan_triplets_scheduled"],
                 "SEARCH_COMPLETION_ACCOUNTING_FAILURE")
        result.update(status="COMPLETE", reasons=[] if stats["proposed_hypotheses"] else
            ["NO_POSE_PROPOSAL_WITHIN_DECLARED_SEARCH"], selected_policy_evaluation_complete=True,
            usable_hypotheses=stats["proposed_hypotheses"] > 0)
    except Exception as error:
        _fail(result, error)
    if usage is not None:
        try:
            _publish_cad_usage(result, usage)
        except Exception as error:
            _fail(result, error)
    return _report(result, maximum_report)
