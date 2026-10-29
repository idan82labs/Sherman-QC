"""Bounded supplied-plane pose proposals; NOT accepted registration.

Only NumPy and the standard library are imported. No extraction, raw input,
finite-support matching, refinement, pose selection or app initialization runs.
Support fields are caller declarations, not verified raw-row provenance.

For each independent triplet, enumerate every CAD bijection and normal sign.
Fit proper R with R*n_scan ~= sign*n_CAD, then map the triplets' intrinsic
intersections: t=x_CAD-R*x_scan. This remains origin-equivariant for imperfect
normal congruence. The familiar N_CAD*t=d_CAD-sign*d_scan is equivalent only
for exact normal alignment. No centroid fills an unconstrained direction.
"""

from dataclasses import dataclass, fields
import itertools
import math

import numpy as np


_HARD_CAPS = dict(max_scan_planes=12, max_cad_planes=24,
                  max_signed_assignments=100_000, max_hypotheses=4096)
_REAL = (int, float)


def _real(value, *, low=0., high=1e6, positive=False):
    if (type(value) not in _REAL or not math.isfinite(value)
            or value > high or (value <= low if positive else value < low)):
        raise ValueError("INVALID_PLANE_SCALAR")
    return float(value)


@dataclass(frozen=True)
class PlaneEquationV4:
    """An unoriented plane n dot x=d in the caller's stated millimetre frame."""
    plane_id: int
    normal: tuple
    offset_mm: float

    def __post_init__(self):
        _equation_snapshot(self)


@dataclass(frozen=True)
class ObservedPlaneV4:
    equation: PlaneEquationV4
    support_count: int
    support_rms_mm: float
    support_extent_mm: tuple

    def __post_init__(self):
        _observed_snapshot(self)


@dataclass(frozen=True)
class PlaneHypothesisPolicyV4:
    min_scan_support_count: int = 12
    max_scan_support_rms_mm: float = .1
    min_scan_second_extent_mm: float = .25
    min_normal_condition_ratio: float = .02
    max_normal_residual_radians: float = .02
    max_intersection_backward_residual_mm: float = 1e-8

    def __post_init__(self):
        _policy_snapshot(self)


@dataclass(frozen=True)
class PlaneHypothesisBudgetsV4:
    max_scan_planes: int = 12
    max_cad_planes: int = 24
    max_signed_assignments: int = 100_000
    max_hypotheses: int = 4096

    def __post_init__(self):
        _budget_snapshot(self)


def _equation_snapshot(equation):
    if type(equation) is not PlaneEquationV4:
        raise ValueError("INVALID_PLANE_EQUATION_TYPE")
    plane_id, normal, offset = equation.plane_id, equation.normal, equation.offset_mm
    if type(plane_id) is not int or not 0 <= plane_id < 2**63:
        raise ValueError("INVALID_PLANE_ID")
    if type(normal) is not tuple or len(normal) != 3:
        raise ValueError("INVALID_PLANE_NORMAL")
    normal = tuple(_real(value, low=-1.-1e-8, high=1.+1e-8) for value in normal)
    length = math.sqrt(math.fsum(value*value for value in normal))
    if abs(length-1.) > 1e-8:
        raise ValueError("PLANE_NORMAL_NOT_UNIT")
    offset = _real(offset, low=-1e6)
    # Rescale BOTH coefficients, preserving the supplied plane's geometry.
    return dict(plane_id=plane_id, normal=tuple(value/length for value in normal), offset_mm=offset/length)


def _observed_snapshot(observed):
    if type(observed) is not ObservedPlaneV4:
        raise ValueError("INVALID_OBSERVED_PLANE_TYPE")
    equation = _equation_snapshot(observed.equation)
    count, rms, extent = observed.support_count, observed.support_rms_mm, observed.support_extent_mm
    if type(count) is not int or not 0 <= count < 2**63:
        raise ValueError("INVALID_PLANE_SUPPORT_COUNT")
    if type(extent) is not tuple or len(extent) != 2:
        raise ValueError("INVALID_PLANE_SUPPORT_EXTENT")
    extent = tuple(_real(value) for value in extent)
    if extent[0] < extent[1]:
        raise ValueError("PLANE_SUPPORT_EXTENTS_NOT_DESCENDING")
    return dict(equation=equation, support_count=count,
                support_rms_mm=_real(rms), support_extent_mm=extent)


def _policy_snapshot(policy):
    if type(policy) is not PlaneHypothesisPolicyV4:
        raise ValueError("INVALID_PLANE_HYPOTHESIS_POLICY")
    result = {field.name: getattr(policy, field.name) for field in fields(policy)}
    count = result["min_scan_support_count"]
    if type(count) is not int or not 3 <= count <= 1_000_000:
        raise ValueError("INVALID_PLANE_SUPPORT_POLICY")
    for name, maximum in (("max_scan_support_rms_mm", 1e3),
                          ("min_scan_second_extent_mm", 1e6),
                          ("min_normal_condition_ratio", 1.),
                          ("max_normal_residual_radians", .25),
                          ("max_intersection_backward_residual_mm", .01)):
        result[name] = _real(result[name], high=maximum, positive=True)
    return result


def _budget_snapshot(budgets):
    if type(budgets) is not PlaneHypothesisBudgetsV4:
        raise ValueError("INVALID_PLANE_HYPOTHESIS_BUDGETS")
    result = {name: getattr(budgets, name) for name in _HARD_CAPS}
    if any(type(value) is not int or not 0 <= value <= _HARD_CAPS[name]
           for name, value in result.items()):
        raise ValueError("INVALID_PLANE_HYPOTHESIS_BUDGETS")
    return result


def _triplet(equations, policy, role):
    normals = np.asarray([row["normal"] for row in equations], dtype=float)
    offsets = np.asarray([row["offset_mm"] for row in equations], dtype=float)
    singular = np.linalg.svd(normals, compute_uv=False)
    if not np.isfinite(singular).all() or singular[0] <= 0.:
        raise ValueError("NONFINITE_PLANE_TRIPLET_SYSTEM")
    ratio = float(singular[-1]/singular[0])
    if ratio < policy["min_normal_condition_ratio"]:
        return dict(reason="ILL_CONDITIONED_"+role+"_TRIPLET")
    intersection = np.linalg.solve(normals, offsets)
    residual = float(np.max(np.abs(normals@intersection-offsets)))
    if not np.isfinite(intersection).all() or not math.isfinite(residual):
        raise ValueError("NONFINITE_PLANE_INTERSECTION")
    if residual > policy["max_intersection_backward_residual_mm"]:
        return dict(reason="INTERSECTION_SOLVE_RESIDUAL_EXCEEDS_GATE")
    return dict(reason=None, normals=normals, offsets=offsets, intersection=intersection,
                condition_ratio=ratio, backward_residual_mm=residual,
                plane_ids=tuple(row["plane_id"] for row in equations))


def propose_plane_hypotheses_v4(scan_planes, cad_planes, *,
        policy=PlaneHypothesisPolicyV4(), budgets=PlaneHypothesisBudgetsV4()):
    """Return all engineering-admissible proposals or clear all on fatal failure.

    Collections must be exact tuples of the frozen types above; nested scalar
    tuples are copied and revalidated, including bypass-mutated dataclasses.
    IDs are unique within each collection but need not match between frames.
    All repeated/coplanar/inner-outer records with DISTINCT IDs remain distinct.
    Caps are checked before numeric input work and before triplet construction.
    A COMPLETE empty set is a completed bounded search, not a missing feature.
    """
    statistics = dict(input_scan_planes=0, input_cad_planes=0, eligible_scan_planes=0,
                      planned_signed_assignments=0, signed_assignments_accounted=0,
                      normal_alignments_attempted=0, qualified_proposal_assignments=0,
                      hypotheses_discarded_on_failure=0, rejection_counts={})
    result = dict(schema_version="plane_hypotheses_v4", status="UNSUPPORTED", reason=None,
                  scope="SUPPLIED_INFINITE_PLANE_TRIPLET_INITIALIZATION_ONLY",
                  search_complete=False, candidates_only=True, accepted_registration=False,
                  registration_status="NOT_EVALUATED", production_use_allowed=False,
                  support_provenance="UNVERIFIED_DECLARATIONS", support_raw_row_provenance_verified=False,
                  finite_support_match_verified=False, direct_offset_diagnostic_is_origin_invariant=False,
                  hypotheses=[], excluded_scan_planes=[], statistics=statistics)

    def refuse(reason):
        statistics["hypotheses_discarded_on_failure"] = len(result["hypotheses"])
        result["hypotheses"] = []
        result["status"], result["reason"], result["search_complete"] = "UNSUPPORTED", reason, False
        return result

    def reject(reason, count):
        counts = statistics["rejection_counts"]
        counts[reason] = counts.get(reason, 0)+count
        statistics["signed_assignments_accounted"] += count

    try:
        cap = _budget_snapshot(budgets)
        gate = _policy_snapshot(policy)
        result.update(policy=gate.copy(), budgets=cap.copy())
        if type(scan_planes) is not tuple or type(cad_planes) is not tuple:
            return refuse("PLANE_COLLECTIONS_MUST_BE_IMMUTABLE_TUPLES")
        statistics.update(input_scan_planes=len(scan_planes), input_cad_planes=len(cad_planes))
        if len(scan_planes) > cap["max_scan_planes"] or len(cad_planes) > cap["max_cad_planes"]:
            return refuse("PLANE_INPUT_BUDGET_EXCEEDED")
        scan = sorted((_observed_snapshot(row) for row in scan_planes), key=lambda row: row["equation"]["plane_id"])
        cad = sorted((_equation_snapshot(row) for row in cad_planes), key=lambda row: row["plane_id"])
        scan_ids = [row["equation"]["plane_id"] for row in scan]
        if len(set(scan_ids)) != len(scan) or len({row["plane_id"] for row in cad}) != len(cad):
            return refuse("DUPLICATE_PLANE_ID_WITHIN_FRAME")
        qualified = []
        for row in scan:
            reasons = []
            if row["support_count"] < gate["min_scan_support_count"]:
                reasons.append("INSUFFICIENT_DECLARED_PLANE_SUPPORT")
            if row["support_rms_mm"] > gate["max_scan_support_rms_mm"]:
                reasons.append("DECLARED_PLANE_RMS_EXCEEDS_GATE")
            if row["support_extent_mm"][1] < gate["min_scan_second_extent_mm"]:
                reasons.append("INSUFFICIENT_DECLARED_SECOND_SUPPORT_EXTENT")
            if reasons:
                result["excluded_scan_planes"].append(dict(plane_id=row["equation"]["plane_id"], reasons=reasons))
            else:
                qualified.append(row["equation"])
        result["scan_support_summaries"] = [dict(
            equation=dict(plane_id=row["equation"]["plane_id"], normal=list(row["equation"]["normal"]),
                          offset_mm=row["equation"]["offset_mm"]),
            support_count=row["support_count"], support_rms_mm=row["support_rms_mm"],
            support_extent_mm=list(row["support_extent_mm"])) for row in scan]
        statistics["eligible_scan_planes"] = len(qualified)
        if len(qualified) < 3 or len(cad) < 3:
            return refuse("FEWER_THAN_THREE_ADMISSIBLE_PLANES")
        planned = math.comb(len(qualified), 3)*math.comb(len(cad), 3)*48
        statistics["planned_signed_assignments"] = planned
        if planned > cap["max_signed_assignments"]:
            return refuse("PLANE_ASSIGNMENT_BUDGET_EXCEEDED")
        scans = [_triplet(rows, gate, "SCAN") for rows in itertools.combinations(qualified, 3)]
        cads = [_triplet(rows, gate, "CAD") for rows in itertools.combinations(cad, 3)]
        for source in scans:
            for target in cads:
                reason = source["reason"] or target["reason"]
                if reason:
                    reject(reason, 48)
                    continue
                for permutation in itertools.permutations(range(3)):
                    order = list(permutation)
                    nc, dc = target["normals"][order], target["offsets"][order]
                    for signs_tuple in itertools.product((-1, 1), repeat=3):
                        signs = np.asarray(signs_tuple)
                        statistics["normal_alignments_attempted"] += 1
                        signed_target = signs[:, None]*nc
                        left, _, right = np.linalg.svd(source["normals"].T@signed_target)
                        correction = np.eye(3)
                        correction[-1, -1] = 1. if np.linalg.det(right.T@left.T) >= 0. else -1.
                        rotation = right.T@correction@left.T
                        mapped = source["normals"]@rotation.T
                        cross = np.cross(mapped, signed_target)
                        angles = np.arctan2(np.linalg.norm(cross, axis=1), np.sum(mapped*signed_target, axis=1))
                        if (not np.isfinite(rotation).all() or not np.isfinite(angles).all()
                                or not np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-10)
                                or abs(float(np.linalg.det(rotation))-1.) > 1e-10):
                            return refuse("NUMERICALLY_INVALID_PROPER_NORMAL_ALIGNMENT")
                        normal_error = float(np.max(angles))
                        if normal_error > gate["max_normal_residual_radians"]:
                            reject("NORMAL_ALIGNMENT_RESIDUAL_EXCEEDS_GATE", 1)
                            continue
                        translation = target["intersection"]-rotation@source["intersection"]
                        direct_error = float(np.max(np.abs(nc@translation+signs*source["offsets"]-dc)))
                        if not np.isfinite(translation).all() or not math.isfinite(direct_error):
                            return refuse("NONFINITE_PLANE_HYPOTHESIS")
                        statistics["signed_assignments_accounted"] += 1
                        statistics["qualified_proposal_assignments"] += 1
                        if len(result["hypotheses"]) >= cap["max_hypotheses"]:
                            return refuse("PLANE_HYPOTHESIS_BUDGET_EXCEEDED")
                        result["hypotheses"].append(dict(
                            hypothesis_index=len(result["hypotheses"]),
                            rotation=rotation.tolist(), translation_mm=translation.tolist(),
                            correspondences=[dict(scan_plane_id=source["plane_ids"][i],
                                cad_plane_id=target["plane_ids"][permutation[i]], normal_sign=signs_tuple[i])
                                for i in range(3)],
                            scan_intersection_mm=source["intersection"].tolist(),
                            cad_intersection_mm=target["intersection"].tolist(),
                            scan_normal_condition_ratio=source["condition_ratio"],
                            cad_normal_condition_ratio=target["condition_ratio"],
                            normal_max_residual_radians=normal_error,
                            intersection_backward_residual_max_mm=max(source["backward_residual_mm"], target["backward_residual_mm"]),
                            direct_offset_discrepancy_max_mm=direct_error))
        result.update(status="COMPLETE", search_complete=True,
                      reason="PLANE_HYPOTHESES_REQUIRE_FINITE_VALIDATION" if result["hypotheses"] else "NO_ADMISSIBLE_PLANE_HYPOTHESES")
        return result
    except (ValueError, TypeError, AttributeError, OverflowError, FloatingPointError,
            MemoryError, np.linalg.LinAlgError) as error:
        return refuse("PLANE_HYPOTHESIS_FAILURE:"+str(error))
