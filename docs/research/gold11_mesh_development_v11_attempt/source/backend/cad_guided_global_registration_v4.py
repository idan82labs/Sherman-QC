"""Bound-aware consumers of the separately versioned finite-triangle V4 primitive.

V4 is the V3 consumer text bound to cad_guided_triangle_surface_v4: the query
diagnostics gain the declared nearest-vertex lookup work and the per-row
interval-width uncertainty counter, the near-tie storage maximum is 256, and a
complete query no longer requires zero width rejections (the V4 primitive marks
rows uncertain instead of refusing). Policy/objective/consumer strings are
retained verbatim as wire contracts for the unchanged measurement consumer; the
separate REGISTRATION_IMPLEMENTATION identity names this V4 binding.

No PCA/public workflow entrypoint: the versioned plane initializer owns starts,
input custody and publication. V1 policy numbers and V2 Kabsch/Huber proposal
math remain unchanged. The proposal objective uses MATERIALIZED fixed targets;
it is not a certificate of decreasing exact surface distance. Final forward
support/residual decisions consume the local interval bounds instead.

Intervals concern represented, transformed binary64 points and triangle hulls,
under the primitive's arithmetic model. They do not certify transformation
rounding, normal/rank estimates, reverse sample coverage, native CAD, physical
uncertainty, acquisition side or global pose uniqueness. No product route changes.
"""

from dataclasses import asdict, dataclass, fields
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import cad_guided_global_registration_v1 as _v1
from . import cad_guided_global_registration_v2 as _v2
from . import cad_guided_triangle_surface_v4 as _surface


DISTANCE_MODEL = "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
SUPPORT_POLICY = "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
SURROGATE = "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"
CONSUMER_COMPLETE = "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"  # retained wire contract
REGISTRATION_IMPLEMENTATION = "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4"
_EXECUTION_MAXIMA = (
    ("max_query_calls", 4096), ("max_query_rows", 64_000_000),
    ("max_node_visits", 2_000_000_000), ("max_triangle_pairs", 2_000_000_000),
    ("max_certificate_pairs", 8_000_000_000), ("max_normal_pairs", 500_000_000),
    ("max_work_units", 20_000_000_000),
)
_RESOURCE_KEYS = ("node_visits", "triangle_pairs", "certificate_pairs", "normal_pairs", "work_units")
_STAT_KEYS = (
    "node_visits", "triangle_box_pairs", "triangle_pairs", "certificate_pairs", "normal_pairs",
    "work_units", "passes_completed", "maximum_pair_batch", "maximum_certificate_batch",
    "tie_batches_completed", "maximum_tie_storage_rows", "interval_width_uncertain_pairs",
    "certificate_failures", "proof_pruned_triangle_pairs", "vertex_lookup_rows",
)
_FLOAT_ROWS = (
    "distances", "numerical_distance_guard", "distance_lower", "distance_upper",
    "triangle_distance_lower", "triangle_distance_upper", "closest_point_error",
    "closest_point_restoration_error",
)
_BOOL_ROWS = (
    "normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous",
    "normal_ambiguity_uncertain", "closest_point_ambiguity_uncertain", "region_uncertain",
)
_QUERY_METADATA = dict(
    status="COMPLETE", reason="CONDITIONAL_LOCAL_FINITE_TRIANGLE_QUERY_COMPLETE",
    distance_model=DISTANCE_MODEL,
    closest_point_error_scope="SELECTED_TRIANGLE_OPTIMIZER_NOT_GLOBAL_UNIQUE_WITNESS",
    normal_region_status="ENGINEERING_NOT_INTERVAL_CERTIFIED",
    tie_semantics="POSSIBLE_COMPETITORS_AT_FIXED_GLOBAL_UPPER_PLUS_TAU",
    initial_upper_bound=_surface.INITIAL_UPPER_BOUND,
    interval_width_policy=_surface.INTERVAL_WIDTH_POLICY,
)


@dataclass(frozen=True)
class SurfaceRegistrationBudgetsV4:
    max_query_calls: int = 4096
    max_query_rows: int = 64_000_000
    max_node_visits: int = 2_000_000_000
    max_triangle_pairs: int = 2_000_000_000
    max_certificate_pairs: int = 8_000_000_000
    max_normal_pairs: int = 500_000_000
    max_work_units: int = 20_000_000_000


class _SurfaceFailure(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _SurfaceFailure(reason)


def _up(value):
    return np.nextafter(value, np.inf)


def _down(value):
    return np.nextafter(value, -np.inf)


_norm = _v2._norm


def _validated_execution_budgets(budgets):
    _require(type(budgets) is SurfaceRegistrationBudgetsV4, "INVALID_SURFACE_REGISTRATION_BUDGETS")
    values = {name: getattr(budgets, name) for name, _ in _EXECUTION_MAXIMA}
    _require(all(type(values[name]) is int and 0 <= values[name] <= maximum
                 for name, maximum in _EXECUTION_MAXIMA), "INVALID_SURFACE_REGISTRATION_BUDGETS")
    # Immutable literal maxima are independent of caller-bypass-mutated instances.
    return SurfaceRegistrationBudgetsV4(**values)


def _policy_snapshot(policy):
    _require(type(policy) is _v1.WholePartRegistrationPolicyV1, "INVALID_WHOLE_PART_REGISTRATION_POLICY")
    values = {field.name: getattr(policy, field.name) for field in fields(policy)}
    _require(all(type(value) in (int, float) for value in values.values()), "INVALID_WHOLE_PART_REGISTRATION_POLICY")
    return _v1.WholePartRegistrationPolicyV1(**values)


def _diagnostic_snapshot(result, *, complete):
    _require(type(result) is dict and type(result.get("diagnostics")) is dict,
             "MALFORMED_V3_QUERY_DIAGNOSTICS")
    raw = result["diagnostics"]
    _require(len(raw) == len(_STAT_KEYS) and all(type(key) is str for key in raw)
             and set(raw) == set(_STAT_KEYS), "MALFORMED_V3_QUERY_DIAGNOSTICS")
    values = {key: raw[key] for key in _STAT_KEYS}
    _require(all(type(value) is int and 0 <= value <= 2**53 for value in values.values()),
             "MALFORMED_V3_QUERY_DIAGNOSTICS")
    _require(values["work_units"] == values["node_visits"] + values["triangle_box_pairs"]
        + 8*values["triangle_pairs"] + values["certificate_pairs"] + 3*values["normal_pairs"]
        + _surface.VERTEX_LOOKUP_WORK_UNITS*values["vertex_lookup_rows"],
        "INCONSISTENT_V3_QUERY_WORK_ACCOUNTING")
    _require(values["passes_completed"] <= 2 and values["certificate_pairs"] <= 4*values["triangle_pairs"]
        and values["maximum_pair_batch"] <= 65536 and values["maximum_certificate_batch"] <= 65536
        and values["maximum_tie_storage_rows"] <= 65536,
        "INCONSISTENT_V3_QUERY_PAIR_ACCOUNTING")
    if complete:
        # V4: wide competitive intervals mark rows uncertain; they are counted,
        # not a completeness failure. Certificate work must still be complete.
        _require(values["passes_completed"] == 2
            and values["certificate_failures"] == 0
            and values["certificate_pairs"] == 4*values["triangle_pairs"],
            "INCOMPLETE_V3_QUERY_CERTIFICATE_WORK")
    return values


def _validate_query_result(query, expected_rows, *, index=None):
    """Validate the entire V3 result contract; optional index binds row identities.

This is structural/in-process validation, not independent re-proving of interval
arithmetic or native provenance. No legacy/missing uncertainty field defaults to
False. An unavailable or malformed result raises and must invalidate its caller.
"""
    _require(type(expected_rows) is int and 0 <= expected_rows <= 200000,
             "INVALID_V3_QUERY_EXPECTED_ROWS")
    expected = set(_surface.RESULT_ARRAY_KEYS) | set(_QUERY_METADATA) | {"diagnostics"}
    _require(type(query) is dict and len(query) == len(expected)
        and all(type(key) is str for key in query) and set(query) == expected,
        "MALFORMED_V3_QUERY_RESULT_SCHEMA")
    _require(all(type(query[key]) is str and query[key] == value for key, value in _QUERY_METADATA.items()),
             "UNAVAILABLE_OR_INCOMPATIBLE_V3_QUERY_RESULT")
    for key in _FLOAT_ROWS + ("closest_points", "oriented_normals"):
        value = query[key]
        shape = (expected_rows, 3) if key in ("closest_points", "oriented_normals") else (expected_rows,)
        _require(type(value) is np.ndarray and value.dtype == np.dtype(np.float64)
            and value.shape == shape and np.isfinite(value).all(), "MALFORMED_V3_QUERY_FLOAT_ARRAY")
        if key in _FLOAT_ROWS:
            _require(np.all(value >= 0), "NEGATIVE_V3_QUERY_DISTANCE_OR_ERROR")
    for key in _BOOL_ROWS:
        value = query[key]
        _require(type(value) is np.ndarray and value.dtype == np.dtype(bool)
                 and value.shape == (expected_rows,), "MALFORMED_V3_QUERY_BOOLEAN_ARRAY")
    for key in ("triangle_indices", "near_tie_counts"):
        value = query[key]
        _require(type(value) is np.ndarray and value.dtype == np.dtype(np.int64)
                 and value.shape == (expected_rows,), "MALFORMED_V3_QUERY_INDEX_ARRAY")
    _require(np.all(query["triangle_indices"] >= 0) and np.all(query["triangle_indices"] < 500000)
        and np.all(query["near_tie_counts"] >= 1) and np.all(query["near_tie_counts"] <= _surface.MAX_TIES_PER_QUERY),
        "INVALID_V3_QUERY_IDENTITIES_OR_TIE_COUNTS")
    regions = query["closest_regions"]
    _require(type(regions) is np.ndarray and regions.dtype == np.dtype("U6")
        and regions.shape == (expected_rows,) and np.isin(regions, ("FACE", "EDGE", "VERTEX")).all(),
        "MALFORMED_V3_QUERY_REGION_ARRAY")
    identities = query["source_face_ids"]
    _require(type(identities) is tuple and len(identities) == expected_rows,
             "MALFORMED_V3_QUERY_FACE_IDENTITIES")
    for identity in identities:
        _require((type(identity) is int and 0 <= identity <= 2**53)
            or (type(identity) is str and 0 < len(identity) <= 128
                and all(32 <= ord(char) <= 126 for char in identity)), "MALFORMED_V3_QUERY_FACE_IDENTITIES")
    lower, upper = query["distance_lower"], query["distance_upper"]
    _require(np.all(lower <= query["triangle_distance_lower"])
        and np.all(query["triangle_distance_lower"] <= upper)
        and np.array_equal(upper, query["triangle_distance_upper"])
        and np.all(query["closest_point_restoration_error"] <= query["closest_point_error"]),
        "INCONSISTENT_V3_QUERY_BOUNDS")
    guard, nominal = query["numerical_distance_guard"], query["distances"]
    _require(np.all(_down(nominal-guard) <= lower) and np.all(_up(nominal+guard) >= upper)
        and np.all(np.abs(_norm(query["oriented_normals"])-1.) <= 1e-6),
        "INCONSISTENT_V3_QUERY_GUARD_OR_NORMAL")
    stats = _diagnostic_snapshot(query, complete=True)
    _require((expected_rows == 0 and stats["triangle_pairs"] == 0 and stats["tie_batches_completed"] == 0)
        or (expected_rows > 0 and stats["triangle_pairs"] >= 2*expected_rows
            and stats["tie_batches_completed"] > 0), "INCOMPLETE_V3_QUERY_ROW_ACCOUNTING")
    # V4 consistency: every counted wide competitive pair marked its row with
    # BOTH uncertainty flags and occupied one stored tie slot, so a positive
    # counter requires jointly uncertain rows and cannot exceed their summed
    # near-tie counts. This is a necessary check, not proof of which pair was wide.
    jointly = query["normal_ambiguity_uncertain"] & query["closest_point_ambiguity_uncertain"]
    _require(stats["interval_width_uncertain_pairs"] == 0
        or (bool(np.any(jointly))
            and stats["interval_width_uncertain_pairs"] <= int(np.sum(query["near_tie_counts"][jointly]))),
        "WIDE_INTERVAL_COUNTER_WITHOUT_JOINTLY_UNCERTAIN_ROWS")
    if index is not None:
        _require(type(index) is _surface._TriangleSurfaceIndex, "INVALID_V3_QUERY_BINDING_INDEX")
        ids = query["triangle_indices"]
        _require(np.all(ids < len(index.triangles)) and identities == tuple(
            index.source_face_ids[int(value)] for value in ids), "V3_QUERY_SOURCE_IDENTITY_MISMATCH")
        _require(np.array_equal(query["oriented_normals"], index.normals[ids]), "V3_QUERY_NORMAL_BINDING_MISMATCH")
    return query


class _Queries:
    def __init__(self, index, budgets, surface_budgets):
        _require(type(index) is _surface._TriangleSurfaceIndex, "INVALID_V3_TRIANGLE_INDEX")
        self.index = index
        self.budgets = _validated_execution_budgets(budgets)
        self.surface_budgets = _surface._budgets(surface_budgets)
        self.statistics = dict(query_calls=0, query_rows=0, node_visits=0, triangle_box_pairs=0,
            triangle_pairs=0, certificate_pairs=0, normal_pairs=0, work_units=0)

    def __call__(self, points):
        _require(type(points) is np.ndarray and points.ndim == 2 and points.shape[1:] == (3,),
                 "INVALID_REGISTRATION_QUERY_ROWS")
        rows = len(points)
        _require(self.statistics["query_calls"] < self.budgets.max_query_calls
            and self.statistics["query_rows"]+rows <= self.budgets.max_query_rows,
            "REGISTRATION_QUERY_BUDGET_EXCEEDED")
        self.statistics["query_calls"] += 1
        self.statistics["query_rows"] += rows
        available = asdict(self.surface_budgets)
        for key in _RESOURCE_KEYS:
            available["max_"+key] = min(available["max_"+key],
                getattr(self.budgets, "max_"+key)-self.statistics[key])
        answer = _surface.query_closest_surface(self.index, points,
            budgets=_surface.TriangleSurfaceBudgetsV4(**available))
        # Charge valid attempted telemetry even when the primitive wholly refuses.
        _require(type(answer) is dict, "MALFORMED_V3_QUERY_RESULT_SCHEMA")
        stats = _diagnostic_snapshot(answer, complete=answer.get("status") == "COMPLETE")
        for key in _RESOURCE_KEYS:
            _require(stats[key] <= available["max_"+key], "V3_QUERY_EXCEEDED_ALLOCATED_BUDGET")
        for key in _RESOURCE_KEYS + ("triangle_box_pairs",):
            self.statistics[key] += stats[key]
        if answer.get("status") != "COMPLETE":
            reason = answer.get("reason")
            _require(answer.get("status") == "UNAVAILABLE" and type(reason) is str
                and 0 < len(reason) <= 180 and reason.isupper(), "MALFORMED_V3_QUERY_REFUSAL")
            raise _SurfaceFailure("FINITE_SURFACE_QUERY_UNAVAILABLE:"+reason)
        return _validate_query_result(answer, rows, index=self.index)


def _support_masks(query, gate):
    """Distance-only trichotomy, independent of ambiguity and region labels."""
    _require(type(gate) in (int, float) and math.isfinite(gate) and 0 <= gate <= 1e100,
             "INVALID_FINITE_SUPPORT_GATE")
    _require(type(query) is dict and type(query.get("distances")) is np.ndarray,
             "MALFORMED_V3_SUPPORT_QUERY")
    _validate_query_result(query, len(query["distances"]))
    inside, outside = query["distance_upper"] <= gate, query["distance_lower"] > gate
    return dict(definite_support=inside, definite_outside=outside, unresolved=~(inside | outside))


def _unambiguous(query):
    # Face-tag differences alone are not a geometric conflict. Boundary regions
    # remain usable for finite distance but not conservative tangent-normal rank.
    return ~(query["normal_ambiguous"] | query["closest_point_ambiguous"]
        | query["normal_ambiguity_uncertain"] | query["closest_point_ambiguity_uncertain"])


def _chosen(query, gate, fraction):
    _require(type(fraction) in (int, float) and math.isfinite(fraction) and .5 <= fraction <= 1.,
             "INVALID_REGISTRATION_TRIM_FRACTION")
    masks = _support_masks(query, gate)
    distances = np.where(masks["definite_support"] & _unambiguous(query), query["distance_upper"], np.inf)
    # Same stable global trim/minimum36 rule, now on the declared upper bound.
    return _v1._trim_indices(distances, gate, fraction)


def _possible_mask(query, masks):
    definite_conflict = ((query["normal_ambiguous"] & ~query["normal_ambiguity_uncertain"])
        | (query["closest_point_ambiguous"] & ~query["closest_point_ambiguity_uncertain"]))
    return ~masks["definite_outside"] & ~definite_conflict


def _sum_bounds(lower, upper):
    """Pairwise outward nonnegative sum, O(N) work and O(N) storage, no NxN."""
    low, high = lower.copy(), upper.copy()
    while len(low) > 1:
        pairs = len(low)//2
        next_low = np.maximum(0., _down(low[:2*pairs:2]+low[1:2*pairs:2]))
        next_high = _up(high[:2*pairs:2]+high[1:2*pairs:2])
        if len(low) % 2:
            next_low = np.concatenate((next_low, low[-1:]))
            next_high = np.concatenate((next_high, high[-1:]))
        low, high = next_low, next_high
    return (float(low[0]), float(high[0])) if len(low) else (0., 0.)


def _p95_endpoint(values, *, upper):
    # Preserve NumPy's default linear quantile and represented q=.95. Exact
    # integer indexing avoids a rounded rank changing the selected order stats.
    ordered = np.sort(values)
    numerator, denominator = (.95).as_integer_ratio()
    index, remainder = divmod((len(ordered)-1)*numerator, denominator)
    if not remainder:
        return float(ordered[index])
    weight = remainder/denominator
    a, b = ordered[index], ordered[index+1]
    if upper:
        difference = _up(b-a)
        value = _up(a+_up(difference*min(1., float(_up(weight)))))
    else:
        difference = max(0., float(_down(b-a)))
        value = max(0., float(_down(a+_down(difference*max(0., float(_down(weight)))))))
    return float(value)


def _metric_bounds(lower, upper):
    if not len(lower):
        return dict(p95_lower=None, p95_upper=None, rms_lower=None, rms_upper=None)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        sums = _sum_bounds(np.maximum(0., _down(lower*lower)), _up(upper*upper))
        low_mean = max(0., float(_down(sums[0]/len(lower))))
        high_mean = _up(sums[1]/len(upper))
        result = dict(p95_lower=_p95_endpoint(lower, upper=False), p95_upper=_p95_endpoint(upper, upper=True),
            rms_lower=max(0., float(_down(np.sqrt(low_mean)))), rms_upper=float(_up(np.sqrt(high_mean))))
    _require(all(math.isfinite(value) and value >= 0 for value in result.values()),
             "UNSUPPORTED_V3_RESIDUAL_SUMMARY_ARITHMETIC")
    return result


def _error_summary(query):
    return dict(maximum_closest_point_error_mm=float(np.max(query["closest_point_error"])) if len(query["distances"]) else 0.,
        maximum_closest_point_restoration_error_mm=float(np.max(query["closest_point_restoration_error"]))
            if len(query["distances"]) else 0.,
        maximum_distance_interval_width_mm=float(np.max(_up(query["distance_upper"]-query["distance_lower"])))
            if len(query["distances"]) else 0.)


def _fraction_bound(count, total, *, upper):
    value = count/total
    numerator, denominator = value.as_integer_ratio()
    comparison = numerator*total-count*denominator
    if (upper and comparison < 0) or (not upper and comparison > 0):
        value = float(_up(value) if upper else _down(value))
    return min(1., max(0., value))


def _refine(scan, query_surface, r, t, *, diagonal, policy, coarse):
    """Unchanged V2 materialized-target optimizer; bound-aware support selection.

World-point residuals below are deliberately the SAME fixed materialized-q
surrogate before/after a step. They never replace V3's local distance intervals
in support or final acceptance, and are not certified exact-surface objectives.
"""
    policy = _policy_snapshot(policy)
    _require(type(coarse) is bool and type(diagonal) in (int, float)
        and math.isfinite(diagonal) and diagonal >= 0, "INVALID_REFINEMENT_CONFIGURATION")
    scan = _v1._positions(scan, 200000)
    count = policy.coarse_iterations if coarse else policy.refine_iterations
    gate = (max(policy.max_correspondence_distance_mm, policy.coarse_distance_fraction*diagonal)
            if coarse else policy.max_correspondence_distance_mm)
    last = {}
    for iteration in range(count):
        transformed = scan@r.T+t
        query = query_surface(transformed)
        masks = _support_masks(query, gate)
        chosen = _chosen(query, gate, policy.trim_fraction)
        safe = _unambiguous(query)
        last = dict(admissible_rows=int(np.count_nonzero(masks["definite_support"] & safe)),
            retained_rows=len(chosen), ambiguous_rows=int(np.count_nonzero(~safe)),
            boundary_rows=int(np.count_nonzero(query["closest_regions"] != "FACE")),
            threshold_unresolved_rows=int(np.count_nonzero(masks["unresolved"])),
            normal_uncertain_rows=int(np.count_nonzero(query["normal_ambiguity_uncertain"])),
            point_uncertain_rows=int(np.count_nonzero(query["closest_point_ambiguity_uncertain"])),
            region_uncertain_rows=int(np.count_nonzero(query["region_uncertain"])),
            numerical_support_policy=SUPPORT_POLICY, optimizer_objective=SURROGATE,
            distance_model=DISTANCE_MODEL, **_error_summary(query))
        if len(chosen) < 36:
            return r, t, iteration, "INSUFFICIENT_REFINEMENT_SUPPORT", last
        p, q = transformed[chosen], query["closest_points"][chosen]
        distances = _norm(p-q)
        weight = (np.ones(len(chosen)) if coarse else
                  np.minimum(1., policy.huber_delta_mm/np.maximum(distances, 1e-15)))
        increment, center, displacement = _v2._weighted_rigid_increment(p, q, weight)
        if coarse:
            nr = increment@r
            nt = increment@(t-center)+center+displacement
        else:
            omega = Rotation.from_matrix(increment).as_rotvec()
            factor = min(1., policy.max_rotation_step_radians/max(float(_norm(omega)), 1e-15),
                         policy.max_translation_step_mm/max(float(_norm(displacement)), 1e-15))
            previous = _v2._huber_loss(distances, policy.huber_delta_mm)
            nr, nt, accepted = r, t, False
            for backtrack in range(8):
                fraction = factor*.5**backtrack
                partial = _v1._rodrigues(fraction*omega)
                trial_r = partial@r
                trial_t = partial@(t-center)+center+fraction*displacement
                trial_distance = _norm(scan[chosen]@trial_r.T+trial_t-q)
                loss = _v2._huber_loss(trial_distance, policy.huber_delta_mm)
                if loss <= previous+1e-14:
                    nr, nt, accepted = trial_r, trial_t, True
                    last.update(fixed_set_huber_before=previous, fixed_set_huber_after=loss,
                        update_center=center.tolist(), update_fraction=fraction,
                        applied_rotation_radians=fraction*float(_norm(omega)),
                        applied_center_displacement_mm=fraction*float(_norm(displacement)))
                    break
            if not accepted:
                return r, t, iteration+1, "NO_IMPROVING_FIXED_CORRESPONDENCE_STEP", last
        motion = float(np.max(_norm(scan@(nr-r).T+nt-t)))
        r, t = nr, nt
        if motion <= 1e-8:
            return r, t, iteration+1, "STATIONARY", last
    return r, t, count, "ITERATION_LIMIT", last


def _diagnose(scan, cad, refine, query_surface, r, t, policy):
    """Completed fixed-subset gates plus conservative possible-pose evidence.

P95/RMS upper bounds concern ONLY definite, uncertainty-excluded support. For
possible eligibility, padding that subset's lower distances with zeros for every
additional possible row gives optimistic lower bounds, not invented observations.
Neither summary bounds residuals for every possible varying true-inlier set.
The selected support set changes relative to V2; no nested acceptance claim.
"""
    policy = _policy_snapshot(policy)
    scan, cad, refine = (_v1._positions(value, 200000) for value in (scan, cad, refine))
    points = scan@r.T+t
    query = query_surface(points)
    masks = _support_masks(query, policy.max_correspondence_distance_mm)
    safe = _unambiguous(query)
    support = masks["definite_support"] & safe
    possible = _possible_mask(query, masks)
    interior = support & (query["closest_regions"] == "FACE") & ~query["region_uncertain"]
    backward, _ = cKDTree(points).query(cad, k=1, workers=1)
    _require(np.isfinite(backward).all(), "UNSUPPORTED_REVERSE_SAMPLE_DISTANCE")
    reverse = float(np.mean(backward <= policy.max_correspondence_distance_mm))
    residual = np.einsum("ij,ij->i", points-query["closest_points"], query["oriented_normals"])
    ratio, singular = _v2._rank(points[interior], query["oriented_normals"][interior])
    fit_points = refine@r.T+t
    fit_query = query_surface(fit_points)
    retained = _chosen(fit_query, policy.max_correspondence_distance_mm, policy.trim_fraction)
    informative = retained[(fit_query["closest_regions"][retained] == "FACE")
                           & ~fit_query["region_uncertain"][retained]]
    fit_ratio, fit_singular = _v2._rank(fit_points[informative], fit_query["oriented_normals"][informative])
    # Actual optimizer weights use materialized-q distances, not local distance
    # midpoints. Keep the weighted rank tied to that declared proposal objective.
    fit_surrogate = _norm(fit_points[informative]-fit_query["closest_points"][informative])
    weight = np.minimum(1., policy.huber_delta_mm/np.maximum(fit_surrogate, 1e-15))
    weighted_ratio, weighted_singular = _v2._rank(fit_points[informative],
        fit_query["oriented_normals"][informative], weight)
    count, possible_count = int(np.count_nonzero(support)), int(np.count_nonzero(possible))
    forward_lower = _fraction_bound(count, len(scan), upper=False)
    forward_upper = _fraction_bound(possible_count, len(scan), upper=True)
    metrics = _metric_bounds(query["distance_lower"][support], query["distance_upper"][support])
    all_metrics = _metric_bounds(query["distance_lower"], query["distance_upper"])
    padded = np.concatenate((query["distance_lower"][support], np.zeros(possible_count-count)))
    possible_metrics = _metric_bounds(padded, padded)
    observable = min(ratio, fit_ratio, weighted_ratio) >= policy.min_observability_ratio
    rank_may_change = bool(np.any(possible & (masks["unresolved"] | query["normal_ambiguity_uncertain"]
        | query["closest_point_ambiguity_uncertain"] | query["region_uncertain"])))
    fit_masks = _support_masks(fit_query, policy.max_correspondence_distance_mm)
    fit_possible = _possible_mask(fit_query, fit_masks)
    rank_may_change |= bool(np.any(fit_possible & (fit_masks["unresolved"]
        | fit_query["normal_ambiguity_uncertain"] | fit_query["closest_point_ambiguity_uncertain"]
        | fit_query["region_uncertain"])))
    reasons = []
    if count < 36 or forward_lower < policy.min_scan_coverage or reverse < policy.min_cad_coverage:
        reasons.append("INSUFFICIENT_FINITE_SURFACE_SUPPORT")
    if metrics["p95_upper"] is None or metrics["p95_upper"] > policy.max_inlier_plane_p95_mm:
        reasons.append("SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE")
    if not observable:
        reasons.append("LOCAL_POSE_UNOBSERVABLE")
    possible_reasons = []
    if possible_count < 36 or forward_upper < policy.min_scan_coverage or reverse < policy.min_cad_coverage:
        possible_reasons.append("INSUFFICIENT_POSSIBLE_FINITE_SURFACE_SUPPORT")
    if possible_metrics["p95_lower"] is None or possible_metrics["p95_lower"] > policy.max_inlier_plane_p95_mm:
        possible_reasons.append("POSSIBLE_RESIDUAL_LOWER_BOUND_OUTSIDE_GATE")
    if not observable and not rank_may_change:
        possible_reasons.append("LOCAL_POSE_UNOBSERVABLE_WITHOUT_UNRESOLVED_SUPPORT")
    _require(np.isfinite(residual).all() and all(math.isfinite(value) for value in (ratio, fit_ratio, weighted_ratio)),
             "UNSUPPORTED_V3_DIAGNOSTIC_ARITHMETIC")
    result = dict(scan_coverage=forward_lower, scan_coverage_lower=forward_lower,
        scan_coverage_upper=forward_upper, cad_coverage=reverse,
        geometric_scan_coverage=_fraction_bound(int(np.count_nonzero(masks["definite_support"])), len(scan), upper=False),
        inlier_count=count, possible_inlier_count=possible_count,
        inlier_surface_p95_mm=metrics["p95_upper"], inlier_surface_rms_mm=metrics["rms_upper"],
        inlier_surface_p95_lower_mm=metrics["p95_lower"], inlier_surface_p95_upper_mm=metrics["p95_upper"],
        inlier_surface_rms_lower_mm=metrics["rms_lower"], inlier_surface_rms_upper_mm=metrics["rms_upper"],
        possible_inlier_surface_p95_lower_mm=possible_metrics["p95_lower"],
        possible_inlier_surface_rms_lower_mm=possible_metrics["rms_lower"],
        possible_eligible=not possible_reasons, possible_eligibility_reasons=possible_reasons,
        fixed_residual_denominator=count, possible_residual_denominator=possible_count,
        residual_interval_scope="FIXED_DEFINITE_UNAMBIGUOUS_SUPPORT_ONLY",
        possible_residual_scope="OPTIMISTIC_ZERO_PADDED_LOWER_BOUND_NOT_OBSERVATIONS_OR_ACCEPTANCE",
        inlier_plane_p95_mm=float(np.quantile(np.abs(residual[support]), .95)) if count else None,
        inlier_plane_rms_mm=float(np.sqrt(np.mean(residual[support]**2))) if count else None,
        tangent_residual_status="MATERIALIZED_POINT_AND_ENGINEERING_NORMAL_DIAGNOSTIC_ONLY",
        all_scan_point_p95_mm=all_metrics["p95_upper"], all_scan_point_p95_lower_mm=all_metrics["p95_lower"],
        all_scan_point_p95_upper_mm=all_metrics["p95_upper"],
        all_scan_point_max_mm=float(np.max(query["distance_upper"])),
        all_cad_point_p95_mm=float(np.quantile(backward, .95)), all_cad_point_max_mm=float(np.max(backward)),
        reverse_coverage_status="UNCHANGED_NOMINAL_CAD_SAMPLE_TO_SCAN_ROWS_NOT_INTERVAL_CERTIFIED",
        ambiguous_scan_rows=int(np.count_nonzero(~safe)),
        boundary_scan_rows=int(np.count_nonzero(query["closest_regions"] != "FACE")),
        threshold_unresolved_scan_rows=int(np.count_nonzero(masks["unresolved"])),
        definite_outside_scan_rows=int(np.count_nonzero(masks["definite_outside"])),
        normal_uncertain_scan_rows=int(np.count_nonzero(query["normal_ambiguity_uncertain"])),
        point_uncertain_scan_rows=int(np.count_nonzero(query["closest_point_ambiguity_uncertain"])),
        region_uncertain_scan_rows=int(np.count_nonzero(query["region_uncertain"])),
        diagnostic_scan_count=len(scan), diagnostic_cad_count=len(cad), optimizer_scan_count=len(refine),
        informative_scan_count=int(np.count_nonzero(interior)), informative_fraction_of_scan=float(np.mean(interior)),
        fit_subset_count=len(retained), informative_fit_subset_count=len(informative),
        fit_fraction_of_optimizer_scan=len(retained)/len(refine),
        informative_fit_fraction_of_retained=len(informative)/len(retained) if len(retained) else 0.,
        fit_fraction_of_scan=len(retained)/len(scan),
        observability_ratio=ratio, observability_singular_values=singular,
        fit_observability_ratio=fit_ratio, fit_observability_singular_values=fit_singular,
        weighted_fit_observability_ratio=weighted_ratio, weighted_fit_observability_singular_values=weighted_singular,
        observability_status="ENGINEERING_FACE_NORMAL_RANK_EXCLUDES_UNCERTAIN_REGIONS_NOT_INTERVAL_CERTIFIED",
        possible_observability_unresolved=rank_may_change,
        reasons=reasons, numerical_consumer_status=CONSUMER_COMPLETE,
        numerical_support_policy=SUPPORT_POLICY, optimizer_objective=SURROGATE,
        distance_model=DISTANCE_MODEL, **_error_summary(query))
    return result


def _candidate_pose(row):
    r, t = np.asarray(row["rotation"], dtype=np.float64), np.asarray(row["translation"], dtype=np.float64)
    _require(r.shape == (3, 3) and t.shape == (3,) and np.isfinite(r).all() and np.isfinite(t).all()
        and np.max(np.abs(r.T@r-np.eye(3))) <= 1e-8 and abs(float(np.linalg.det(r))-1.) <= 1e-8,
        "INVALID_COMPLETED_CANDIDATE_POSE")
    return r, t


def _select(result, refine, policy):
    """Choose a definitely eligible pose after accounting for possible competitors.

Candidate class distances retain V2's engineering represented-point action and
numeric threshold; no interval/global uniqueness is claimed. Residual/coverage
overlap is handled conservatively, including potentially eligible rejected rows.
Every selected-policy candidate must have completed its V3 consumer diagnostics.
"""
    _require(type(result) is dict and type(result.get("evidence")) is dict,
             "INVALID_V3_SELECTION_RESULT")
    result.update(status="UNAVAILABLE", transform_scan_to_cad=None)
    evidence = result["evidence"]
    evidence.pop("selected_candidate_index", None)
    policy = _policy_snapshot(policy)
    refine = _v1._positions(refine, 20000)
    candidates = evidence.get("candidates")
    _require(type(candidates) is list and len(candidates) <= 48
        and evidence.get("candidate_search_complete") is True, "INCOMPLETE_SELECTED_CANDIDATE_EVALUATION")
    actions = {}
    for index, row in enumerate(candidates):
        _require(type(row) is dict and type(row.get("index")) is int and row["index"] == index
            and row.get("execution_status") == "ATTEMPTED"
            and row.get("numerical_consumer_status") == CONSUMER_COMPLETE
            and row.get("numerical_support_policy") == SUPPORT_POLICY
            and row.get("optimizer_objective") == SURROGATE and row.get("distance_model") == DISTANCE_MODEL,
            "INCOMPLETE_SELECTED_CANDIDATE_EVALUATION")
        _require(type(row.get("reasons")) is list and all(type(value) is str for value in row["reasons"])
            and "CANDIDATE_NUMERICAL_FAILURE" not in row["reasons"]
            and type(row.get("possible_eligible")) is bool,
            "INVALID_COMPLETED_CANDIDATE_DISPOSITION")
        for key in ("scan_coverage", "scan_coverage_lower", "scan_coverage_upper", "cad_coverage"):
            _require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and 0 <= row[key] <= 1,
                     "INVALID_COMPLETED_CANDIDATE_COVERAGE")
        _require(row["scan_coverage"] == row["scan_coverage_lower"] <= row["scan_coverage_upper"],
                 "INCONSISTENT_COMPLETED_CANDIDATE_COVERAGE")
        for key in ("inlier_surface_rms_lower_mm", "inlier_surface_rms_upper_mm",
                    "possible_inlier_surface_rms_lower_mm"):
            value = row.get(key)
            _require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
                     "INVALID_COMPLETED_CANDIDATE_RESIDUAL")
        if not row["reasons"]:
            _require(row["possible_eligible"] and row["inlier_surface_rms_upper_mm"] is not None
                and row["inlier_surface_rms_lower_mm"] is not None
                and row.get("inlier_surface_rms_mm") == row["inlier_surface_rms_upper_mm"]
                and row["inlier_surface_rms_lower_mm"] <= row["inlier_surface_rms_upper_mm"],
                "INCONSISTENT_DEFINITE_CANDIDATE_ELIGIBILITY")
        if row["possible_eligible"]:
            _require(row["possible_inlier_surface_rms_lower_mm"] is not None,
                     "MISSING_POSSIBLE_CANDIDATE_RESIDUAL")
        r, t = _candidate_pose(row)
        actions[index] = refine@r.T+t
        _require(np.isfinite(actions[index]).all(), "NONFINITE_CANDIDATE_POSE_ACTION")
    eligible = [row for row in candidates if not row["reasons"]]
    possible = [row for row in candidates if row["possible_eligible"]]
    evidence.update(eligible_candidate_count=len(eligible), possible_eligible_candidate_count=len(possible),
        candidate_comparison="POSSIBLE_COVERAGE_UPPER_AND_RESIDUAL_LOWER_VS_DEFINITE_WINNER",
        pose_class_status="UNCHANGED_V2_ENGINEERING_ACTION_THRESHOLD_NOT_CERTIFIED_GLOBAL_UNIQUENESS")
    if not eligible:
        result["reasons"] = ["NO_DEFINITELY_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
        return
    eligible.sort(key=lambda row: (-row["scan_coverage_lower"], row["inlier_surface_rms_upper_mm"],
                                   -row["cad_coverage"], row["index"]))
    classes = []
    for row in eligible:
        if not any(float(np.max(_norm(actions[row["index"]]-actions[prior["index"]])))
                   <= policy.pose_class_distance_mm for prior in classes):
            classes.append(row)
    best = classes[0]
    competing = []
    for row in possible:
        if row["index"] == best["index"]:
            continue
        same_class = float(np.max(_norm(actions[row["index"]]-actions[best["index"]]))) <= policy.pose_class_distance_mm
        if (not same_class and row["scan_coverage_upper"] >= _down(best["scan_coverage_lower"]-policy.ambiguity_coverage_margin)
                and row["cad_coverage"] >= _down(best["cad_coverage"]-policy.ambiguity_coverage_margin)
                and row["possible_inlier_surface_rms_lower_mm"]
                    <= _up(best["inlier_surface_rms_upper_mm"]+policy.ambiguity_plane_rms_margin_mm)):
            competing.append(row)
    competing_classes = []
    for row in competing:
        if not any(float(np.max(_norm(actions[row["index"]]-actions[prior["index"]])))
                   <= policy.pose_class_distance_mm for prior in competing_classes):
            competing_classes.append(row)
    evidence.update(distinct_eligible_pose_class_count=len(classes),
        comparable_distinct_pose_count=len(competing_classes), possible_comparable_candidate_count=len(competing),
        possible_comparable_candidate_indices=[row["index"] for row in competing],
        best_generated_candidate_index=best["index"])
    if competing:
        result["reasons"] = ["AMBIGUOUS_POSSIBLE_GENERATED_WHOLE_PART_POSES"]
        return
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = best["rotation"], best["translation"]
    result.update(status="CONDITIONAL_BEST_FIT", transform_scan_to_cad=transform.tolist(), reasons=[])
    evidence["selected_candidate_index"] = best["index"]
