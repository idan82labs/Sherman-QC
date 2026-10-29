"""V9 bound-aware finite-face initialization and registration, opt-in only.

Preserve V8 noisy extraction, plane search, all-generated screening and 48-start
schedule. V3 finite-surface intervals supply definite distance support; unresolved
distance, normal or point evidence cannot grant support. Registration refinement
remains a declared materialized-candidate engineering surrogate, not certified
pose optimization or metrology. Original V8 and all terminal attempts are frozen.
"""

from dataclasses import asdict, dataclass

import numpy as np

from . import cad_guided_plane_initialization_v4 as _old
from . import cad_guided_plane_initialization_v8 as _previous
from . import cad_guided_global_registration_v3 as _v3
from . import cad_guided_triangle_surface_v3 as _surface
from . import cad_guided_noisy_mesh_planes_v5 as _noisy
from . import cad_guided_noisy_mesh_planes_v7 as _compact
from .cad_guided_report_envelope_v6 import bound_initialization_report_v6 as _bounded_report
from . import cad_guided_plane_search_v6 as _search


_v1, _patch = _old._v1, _old._patch
_require, _digest, _pose = _old._require, _old._digest, _old._pose


_LIMITS = dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
    max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
    max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
    chunk_rows=65536, max_node_visits=512_000_000,
    max_triangle_pairs=256_000_000, max_certificate_pairs=1_024_000_000, max_normal_pairs=32_000_000,
    max_work_units=4_000_000_000, max_face_build_nodes=131071,
    max_face_build_triangle_visits=10_000_000, max_refined_start_groups=48,
    max_report_bytes=100_663_296)


@dataclass(frozen=True)
class PlaneInitializationBudgetsV9:
    max_hypothesis_records: int = 8192
    max_prefilter_unsigned_assignments: int = 1024
    max_cached_cad_faces: int = 2048
    max_indexed_face_triangles: int = 500_000
    max_prefilter_query_calls: int = 32768
    max_prefilter_query_rows: int = 64_000_000
    chunk_rows: int = 65536
    max_node_visits: int = 512_000_000
    max_triangle_pairs: int = 256_000_000
    max_certificate_pairs: int = 1_024_000_000
    max_normal_pairs: int = 32_000_000
    max_work_units: int = 4_000_000_000
    max_face_build_nodes: int = 131071
    max_face_build_triangle_visits: int = 10_000_000
    max_refined_start_groups: int = 48
    max_report_bytes: int = 100_663_296


def _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
        cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets):
    _require(type(budgets) is PlaneInitializationBudgetsV9, "INVALID_V9_INITIALIZATION_BUDGET_TYPE")
    cap = {key: getattr(budgets, key) for key in _LIMITS}
    _require(all(type(value) is int and 0 <= value <= _LIMITS[key] for key, value in cap.items())
        and cap["chunk_rows"] > 0, "INVALID_V9_INITIALIZATION_BUDGET")
    ng, nb = _noisy._configuration(noisy_policy, noisy_budgets)
    cg, cb = _patch._configuration(cad_patch_policy, cad_patch_budgets)
    sg, sb = _search._configuration(search_policy, search_budgets)
    surface = _surface._budgets(surface_budgets)
    return dict(policy=_old._registration_policy(registration_policy),
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(**ng), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(**nb),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(**cg), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(**cb),
        search_policy=_search.PlaneSearchPolicyV6(**sg), search_budgets=_search.PlaneSearchBudgetsV6(**sb),
        budgets=PlaneInitializationBudgetsV9(**cap),
        surface_budgets=_surface.TriangleSurfaceBudgets(**asdict(surface)))


def _base():
    result = _old._base()
    result.update(schema_version="noisy_plane_initialization_prefilter_v9", search_exhaustive=False,
        search_scope="ALL_GENERATED_PROPOSED_SIGNS_SCREENED_UPSTREAM_SEARCH_AND_REFINEMENT_NOT_EXHAUSTIVE",
        noise_bound_status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        global_pose_uniqueness="NOT_ESTABLISHED", selected_unsigned_assignment_ids=[],
        prefilter_schedule="ALL_GENERATED_PROPOSED_SIGNS_HARD_UNSIGNED_BOUND_NO_RANK_OMISSION",
        refinement_schedule="48_STRATIFIED_EXACT_PASSING_START_GROUPS",
        all_passing_start_groups=[], selected_policy_evaluation_complete=False,
        numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION",
        numerical_consumer_complete=False)
    result["statistics"].update(certificate_pairs=0, triangle_box_pairs=0,
        proposed_hypothesis_count=0, scheduled_hypothesis_count=0,
        omitted_proposed_hypothesis_count=0, search_rejected_hypothesis_count=0,
        eligible_unsigned_assignment_count=0, scheduled_unsigned_assignment_count=0,
        omitted_unsigned_assignment_count=0, exact_passing_start_group_count=0,
        omitted_passing_start_group_count=0)
    return result


# Pure owned-array preparation/binding are unchanged frozen mechanisms.
_prepare = _previous._prepare
_bound_scan = _previous._bound_scan


class _FaceQueries:
    """Owned finite-face indices with cumulative V3 certificate/work budgets."""
    def __init__(self, context, face_map, budget, surface_budget, result):
        self.indices = {}
        self.budget, self.surface_budget = budget, surface_budget
        self.statistics = result["statistics"]
        _require(len(face_map) <= budget.max_cached_cad_faces, "CAD_FACE_CACHE_BUDGET_EXCEEDED")
        planned = sum(len(patch["triangle_rows"]) for patch in face_map.values())
        _require(planned <= budget.max_indexed_face_triangles, "CAD_FACE_TRIANGLE_BUDGET_EXCEEDED")
        for face, patch in sorted(face_map.items()):
            values = asdict(surface_budget)
            for name in ("nodes", "triangle_visits"):
                key = "build_"+name
                values["max_"+key] = min(values["max_"+key],
                    getattr(budget, "max_face_"+key)-self.statistics["face_"+key])
            rows = patch["triangle_rows"]
            built = _surface.build_triangle_surface(context["cad"][rows], context["faces"][rows],
                budgets=_surface.TriangleSurfaceBudgets(**values))
            result["face_index_evidence"].append(dict(cad_plane_id=face,
                triangle_rows_sha256=patch["triangle_rows_sha256"], triangle_count=len(rows),
                result={key: value for key, value in built.items() if key != "index"}))
            for name in ("build_nodes", "build_triangle_visits"):
                self.statistics["face_"+name] += built["diagnostics"].get(name, 0)
            _require(built["status"] == "COMPLETE", "ASSIGNED_FACE_SURFACE_UNAVAILABLE")
            self.indices[face] = built["index"]
            self.statistics["indexed_face_count"] += 1
            self.statistics["indexed_face_triangles"] += len(rows)

    def __call__(self, face, points):
        stats, budget = self.statistics, self.budget
        _require(stats["query_calls"] < budget.max_prefilter_query_calls
            and stats["query_rows"]+len(points) <= budget.max_prefilter_query_rows,
            "PREFILTER_QUERY_BUDGET_EXCEEDED")
        values = asdict(self.surface_budget)
        charged = ("node_visits", "triangle_pairs", "certificate_pairs", "normal_pairs", "work_units")
        for name in charged:
            values["max_"+name] = min(values["max_"+name], getattr(budget, "max_"+name)-stats[name])
        stats["query_calls"] += 1
        stats["query_rows"] += len(points)
        query = _surface.query_closest_surface(self.indices[face], points,
            budgets=_surface.TriangleSurfaceBudgets(**values))
        _require(type(query) is dict, "INVALID_V3_FACE_QUERY_RESULT")
        telemetry = _v3._diagnostic_snapshot(query, complete=query.get("status") == "COMPLETE")
        for name in charged:
            _require(telemetry[name] <= values["max_"+name], "V3_FACE_QUERY_EXCEEDED_ALLOCATED_BUDGET")
        for name in (*charged, "triangle_box_pairs"):
            stats[name] += telemetry[name]
        _require(query["status"] == "COMPLETE", "ASSIGNED_FACE_QUERY_UNAVAILABLE:"+str(query["reason"]))
        return _v3._validate_query_result(query, len(points), index=self.indices[face])


def _prefilter(context, result, configuration):
    budget, policy, surface = configuration["budgets"], configuration["policy"], configuration["surface_budgets"]
    stats, hypotheses = result["statistics"], context["hypotheses"]
    _require(len(hypotheses) <= budget.max_hypothesis_records, "HYPOTHESIS_RECORD_BUDGET_EXCEEDED")
    eligible = sorted({row["unsigned_assignment_id"] for row in hypotheses if row["status"] == "PROPOSED"})
    selected = set(eligible)
    result["selected_unsigned_assignment_ids"] = sorted(selected)
    stats.update(hypothesis_count=len(hypotheses), eligible_unsigned_assignment_count=len(eligible),
        scheduled_unsigned_assignment_count=len(selected), omitted_unsigned_assignment_count=len(eligible)-len(selected))
    scans = context["scan_supports"]
    faces = {row["plane_id"]: row for row in context["cad_patches"]["patches"]}
    chunk = min(budget.chunk_rows, surface.max_query_rows)
    _require(chunk > 0, "PREFILTER_QUERY_ROW_CAP_ZERO")
    used_faces, checked_scans, checked_faces, scheduled = {}, set(), set(), []
    # All completed-search identities enter the ledger before any per-support
    # validation can fail. A malformed selected support must not erase later
    # known hypotheses that were never attempted.
    for hypothesis in hypotheses:
        proposed = hypothesis["status"] == "PROPOSED"
        chosen = proposed and hypothesis["unsigned_assignment_id"] in selected
        row = dict(hypothesis, prefilter_status="NOT_ATTEMPTED" if chosen else
            "NOT_SCHEDULED_BY_DECLARED_UNSIGNED_QUOTA" if proposed else "SEARCH_NOMINAL_WITNESS_REFUSAL",
            prefilter_reasons=[], assigned_support=[], prefilter_scheduled=chosen)
        result["hypotheses"].append(row)
        stats["proposed_hypothesis_count" if proposed else "search_rejected_hypothesis_count"] += 1
        stats["scheduled_hypothesis_count"] += int(chosen)
        stats["omitted_proposed_hypothesis_count"] += int(proposed and not chosen)
    # Retain the whole completed-search ledger before a smaller hard bound can
    # refuse. This bound is NOT a selection quota: no _strata/min/prefix subset
    # may acquire finite support evidence while other proposed signs are omitted.
    _require(len(eligible) <= budget.max_prefilter_unsigned_assignments,
             "ALL_PROPOSED_UNSIGNED_PREFLIGHT_BUDGET_EXCEEDED")
    for index, row in enumerate(result["hypotheses"]):
        _require(row["hypothesis_index"] == index and row["status"] in ("PROPOSED", "REJECTED"),
                 "INVALID_COMPLETED_SEARCH_HYPOTHESIS")
        chosen = row["prefilter_scheduled"]
        if not chosen:
            continue
        scheduled.append(row)
        _pose(row)
        matches = row["correspondences"]
        _require(len(matches) == 3 and len({item["scan_plane_id"] for item in matches}) == 3
            and len({item["cad_plane_id"] for item in matches}) == 3, "INVALID_SEARCH_TRIPLET_BINDING")
        for match in matches:
            scan_id, face = match["scan_plane_id"], match["cad_plane_id"]
            _require(scan_id in scans and face in faces and type(match["normal_sign"]) is int
                and match["normal_sign"] in (-1, 1), "UNKNOWN_NOISY_PLANE_SUPPORT")
            patch = scans[scan_id]
            if scan_id not in checked_scans:
                _bound_scan(context, patch)
                checked_scans.add(scan_id)
            if face not in checked_faces:
                _old._bound_support(context, cad_patch=faces[face])
                checked_faces.add(face)
            count = len(patch["raw_vertex_rows"])
            stats["planned_query_rows"] += count
            stats["planned_query_calls"] += (count+chunk-1)//chunk
            used_faces[face] = faces[face]
            row["assigned_support"].append(dict(match, proposal_kind=patch["proposal_kind"],
                source_record_id=patch["source_record_id"], support_count=count,
                raw_vertex_rows_sha256=patch["raw_vertex_rows_sha256"],
                scan_triangle_rows_sha256=patch["triangle_rows_sha256"],
                cad_triangle_rows_sha256=faces[face]["triangle_rows_sha256"],
                query_rows=0, attempted_query_rows=0, finite_unambiguous_inlier_count=0,
                coverage=None, ambiguous_rows=0, boundary_rows=0,
                definite_distance_inlier_rows=0, definite_distance_outside_rows=0,
                threshold_unresolved_rows=0, uncertain_region_rows=0,
                maximum_distance_interval_width_mm=0., maximum_optimizer_error_mm=0.,
                maximum_restoration_error_mm=0., status="NOT_ATTEMPTED", reasons=[]))
    _require(stats["planned_query_rows"] <= budget.max_prefilter_query_rows
        and stats["planned_query_calls"] <= budget.max_prefilter_query_calls,
        "PREFILTER_PREFLIGHT_BUDGET_EXCEEDED")
    query = _FaceQueries(context, used_faces, budget, surface, result)
    groups, lookup = result["all_passing_start_groups"], {}
    for row in scheduled:
        rotation, translation = _pose(row)
        row["prefilter_status"] = "INCOMPLETE"
        for support in row["assigned_support"]:
            raw = scans[support["scan_plane_id"]]["raw_vertex_rows"]
            support["status"] = "INCOMPLETE"
            for start in range(0, len(raw), chunk):
                points = context["scan"][raw[start:start+chunk]]@rotation.T+translation
                support["attempted_query_rows"] += len(points)
                answer = query(support["cad_plane_id"], points)
                unambiguous = _v3._unambiguous(answer)
                support["query_rows"] += len(points)
                masks = _v3._support_masks(answer, policy.max_correspondence_distance_mm)
                support["finite_unambiguous_inlier_count"] += int(np.count_nonzero(
                    unambiguous & masks["definite_support"]))
                for key, name in (("definite_support", "definite_distance_inlier_rows"),
                                  ("definite_outside", "definite_distance_outside_rows"),
                                  ("unresolved", "threshold_unresolved_rows")):
                    support[name] += int(np.count_nonzero(masks[key]))
                support["uncertain_region_rows"] += int(np.count_nonzero(answer["region_uncertain"]))
                for name, values in (
                    ("maximum_distance_interval_width_mm", np.nextafter(
                        answer["distance_upper"]-answer["distance_lower"], np.inf)),
                    ("maximum_optimizer_error_mm", answer["closest_point_error"]),
                    ("maximum_restoration_error_mm", answer["closest_point_restoration_error"])):
                    support[name] = max(support[name], float(np.max(values)))
                support["ambiguous_rows"] += int(np.count_nonzero(~unambiguous))
                support["boundary_rows"] += int(np.count_nonzero(answer["closest_regions"] != "FACE"))
            # A nearest-rounded fraction can cross a represented policy value
            # (for example 5/6). Only a lower enclosure can establish support.
            support["coverage"] = _v3._fraction_bound(
                support["finite_unambiguous_inlier_count"], support["support_count"], upper=False)
            passed = support["coverage"] >= policy.min_scan_coverage
            support.update(status="PASSED" if passed else "REJECTED",
                reasons=[] if passed else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"])
        passed = all(support["status"] == "PASSED" for support in row["assigned_support"])
        row.update(prefilter_status="PASSED" if passed else "REJECTED",
            prefilter_reasons=[] if passed else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
        stats["hypotheses_evaluated"] += 1
        stats["passing_hypothesis_count" if passed else "rejected_hypothesis_count"] += 1
        if passed:
            key = rotation.tobytes()+translation.tobytes()
            if key not in lookup:
                lookup[key] = len(groups)
                groups.append(dict(exact_group_index=len(groups), initialized_rotation=rotation.tolist(),
                    initialized_translation_mm=translation.tolist(), hypothesis_indices=[],
                    refinement_disposition="NOT_SCHEDULED_BY_DECLARED_GROUP_QUOTA"))
            group = groups[lookup[key]]
            row["exact_start_group_index"] = group["exact_group_index"]
            group["hypothesis_indices"].append(row["hypothesis_index"])
    indices = _search._strata(len(groups), budget.max_refined_start_groups)
    for index, group_index in enumerate(indices):
        groups[group_index]["refinement_disposition"] = "SCHEDULED_FOR_V3_REFINEMENT"
        result["start_groups"].append(dict(groups[group_index], index=index))
    stats.update(exact_passing_start_group_count=len(groups), surviving_start_group_count=len(indices),
                 omitted_passing_start_group_count=len(groups)-len(indices))
    _require(stats["hypotheses_evaluated"] == stats["scheduled_hypothesis_count"],
             "PREFILTER_SELECTED_DENOMINATOR_INCOMPLETE")
    result.update(status="COMPLETE", search_complete=True, selected_policy_evaluation_complete=True,
        numerical_consumer_complete=True,
        reasons=[] if indices else ["NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"])


def _initialize(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
                cad_source_triangle_ids, epsilon, configuration, unit_assumption):
    result, context = _base(), None
    result.update(budgets=asdict(configuration["budgets"]), registration_policy=asdict(configuration["policy"]),
        surface_budgets=asdict(configuration["surface_budgets"]))
    try:
        context = _prepare(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
                           cad_source_triangle_ids, epsilon, configuration, result, unit_assumption)
        _prefilter(context, result, configuration)
    except Exception as error:
        _old._fail_prefilter(result, error)
        result["selected_policy_evaluation_complete"] = False
        result["numerical_consumer_complete"] = False
        result["statistics"].update(exact_passing_start_group_count=len(result["all_passing_start_groups"]),
            exact_passing_start_group_count_scope="ATTEMPTED_PREFILTER_BEFORE_FATAL_NOT_SURVIVING",
            omitted_passing_start_group_count=None)
        for group in result["all_passing_start_groups"]:
            group.update(refinement_disposition="INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE",
                attempted_initialized_rotation=group["initialized_rotation"],
                attempted_initialized_translation_mm=group["initialized_translation_mm"],
                initialized_rotation=None, initialized_translation_mm=None)
        context = None
    result = _bounded_report(result, configuration["budgets"].max_report_bytes)
    return result, context if result["status"] == "COMPLETE" else None


def prefilter_plane_initializations_v9(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm, registration_policy,
        unit_assumption, noisy_policy=_noisy.NoisyMeshPlanePolicyV5(),
        noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(), cad_patch_policy=_patch.CoplanarPatchPolicyV4(),
        cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(), search_policy=_search.PlaneSearchPolicyV6(),
        search_budgets=_search.PlaneSearchBudgetsV6(), budgets=PlaneInitializationBudgetsV9(),
        surface_budgets=None):
    """Owned-array scheduled finite-face evidence; no accepted registration."""
    result, maximum = _base(), _LIMITS["max_report_bytes"]
    try:
        configuration = _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
            cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets)
        maximum = configuration["budgets"].max_report_bytes
        _require(type(unit_assumption) is str and unit_assumption == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                 "EXPLICIT_PROVISIONAL_UNIT_ASSUMPTION_REQUIRED")
        result, _ = _initialize(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
            cad_source_triangle_ids, position_error_bounds_mm, configuration, unit_assumption)
    except Exception as error:
        _old._fail_prefilter(result, error)
        result["selected_policy_evaluation_complete"] = False
        result["numerical_consumer_complete"] = False
    return _bounded_report(result, maximum)


_registration_rows = _previous._registration_rows


def register_whole_part_planes_v9(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm,
        cad_initialization_points, registration_policy, unit_assumption, registration_raw_rows=None,
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(),
        search_policy=_search.PlaneSearchPolicyV6(), search_budgets=_search.PlaneSearchBudgetsV6(),
        budgets=PlaneInitializationBudgetsV9(), surface_budgets=None, registration_budgets=None):
    """Explicit error-bound declaration and owned supports, no native or file IO.

All original assigned support rows reach finite-face checks. The explicit global
registration selection still has V2's <=200k limit and is separately disclosed.
The returned fit never establishes global pose uniqueness or a stable datum.
"""
    result = _old._registration_result(unit_assumption)
    result["schema_version"] = "noisy_plane_initialized_finite_registration_v9"
    evidence, query = result["evidence"], None
    evidence.update(schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V9",
        candidate_search_status="ALL_GENERATED_PROPOSED_SIGNS_FINITE_SCREENED_NOT_GLOBAL_SEARCH",
        search_exhaustive=False, global_pose_uniqueness="NOT_ESTABLISHED",
        noise_bound_status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION",
        numerical_consumer_complete=False,
        optimizer_objective="MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS",
        distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
        observability_basis="UNAMBIGUOUS_NONUNCERTAIN_FACE_INTERIORS_ACTUAL_TRIM_ENGINEERING_NORMAL_RANK",
        residual_gate="FINITE_DISTANCE_P95_UPPER_USING_UNCHANGED_LEGACY_NUMERIC_THRESHOLD",
        reverse_coverage_status="ENGINEERING_CAD_SAMPLE_TO_SELECTED_SCAN_ROWS_NOT_CERTIFIED_SURFACE_COVERAGE")
    maximum = _LIMITS["max_report_bytes"]
    try:
        configuration = _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
            cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets)
        maximum = configuration["budgets"].max_report_bytes
        execution = _v3._validated_execution_budgets(_v3.SurfaceRegistrationBudgetsV3()
            if registration_budgets is None else registration_budgets)
        execution = _v3.SurfaceRegistrationBudgetsV3(**asdict(execution))
        _require(type(unit_assumption) is str and unit_assumption == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                 "EXPLICIT_PROVISIONAL_UNIT_ASSUMPTION_REQUIRED")
        full_scan = _old._array(scan_points, (3,), "fiu", configuration["noisy_budgets"].max_raw_rows,
                               "<f8", "SCAN_POINTS")
        scan = _registration_rows(dict(scan=full_scan), registration_raw_rows, evidence)
        cad = _v1._canonical_scan(cad_initialization_points)
        evidence["cad_initialization_points_sha256"] = _digest(_patch._immutable(cad, "<f8"))
        evidence["cad_sample_provenance"] = "CALLER_MUST_BIND_TO_TRIANGLES_NOT_AUTHENTICATED_BY_ARRAY_API"
        initialization, context = _initialize(full_scan, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
            cad_source_triangle_ids, position_error_bounds_mm, configuration, unit_assumption)
        evidence["initialization"] = initialization
        _require(initialization["status"] == "COMPLETE" and initialization["selected_policy_evaluation_complete"]
            and initialization["numerical_consumer_complete"] is True,
                 "NOISY_PLANE_INITIALIZATION_UNAVAILABLE")
        _require(bool(initialization["start_groups"]), "NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS")
        candidates = evidence["candidates"]
        for group in initialization["start_groups"]:
            candidates.append(dict(group, execution_status="NOT_ATTEMPTED", reasons=["NOT_ATTEMPTED_YET"]))
        policy, surface = configuration["policy"], configuration["surface_budgets"]
        evidence.update(policy=asdict(policy), surface_query_budgets=asdict(surface),
            registration_budgets=asdict(execution), input_digests=dict(initialization["input_digests"]))
        coarse = _v1._spatial_sample(scan, policy.coarse_points)
        refine = _v1._spatial_sample(scan, policy.refine_points)
        built = _surface.build_triangle_surface(context["cad"], context["faces"], budgets=surface)
        evidence["surface_index"] = {key: value for key, value in built.items() if key != "index"}
        _require(built["status"] == "COMPLETE", "FINITE_TRIANGLE_SURFACE_UNAVAILABLE")
        query = _v3._Queries(built["index"], execution, surface)
        diagonal = float(_v3._norm(np.ptp(built["index"].triangles.reshape(-1, 3), axis=0)))
        evidence.update(scan_unique_rows=len(scan), cad_unique_rows=len(cad),
            coarse_scan_rows=len(coarse), refined_scan_rows=len(refine))
        for row in candidates:
            row.update(execution_status="ATTEMPTED", reasons=[])
            rotation, translation = np.asarray(row["initialized_rotation"]), np.asarray(row["initialized_translation_mm"])
            try:
                rotation, translation, nc, cr, cd = _v3._refine(coarse, query, rotation, translation,
                    diagonal=diagonal, policy=policy, coarse=True)
                rotation, translation, nf, fr, fd = _v3._refine(refine, query, rotation, translation,
                    diagonal=diagonal, policy=policy, coarse=False)
                _pose(dict(rotation=rotation, translation_mm=translation))
                row.update(_v3._diagnose(scan, cad, refine, query, rotation, translation, policy))
                row.update(coarse_iterations=nc, refine_iterations=nf, coarse_convergence_state=cr,
                    refine_convergence_state=fr, coarse_last_step=cd, refine_last_step=fd,
                    rotation=rotation.tolist(), translation=translation.tolist())
            except _v3._SurfaceFailure as error:
                row["reasons"] = [str(error)]
                raise
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                row["reasons"] = ["CANDIDATE_NUMERICAL_FAILURE"]
        evidence["candidate_search_complete"] = True
        evidence["numerical_consumer_complete"] = True
        _v3._select(result, refine, policy)
    except Exception as error:
        result.update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=[_old._reason(error)])
        evidence.pop("selected_candidate_index", None)
        evidence["numerical_consumer_complete"] = False
        evidence.update(candidate_search_complete=False, evidence_scope="ATTEMPTED_WORK_ONLY_NO_ACCEPTED_REGISTRATION")
        for row in evidence["candidates"]:
            if row["execution_status"] == "NOT_ATTEMPTED":
                row["reasons"] = ["NOT_ATTEMPTED_AFTER_FATAL_FAILURE"]
            elif not row.get("reasons"):
                row["reasons"] = ["INVALIDATED_BY_FATAL_REGISTRATION_FAILURE"]
    finally:
        evidence["candidate_count"] = len(evidence["candidates"])
        evidence["attempted_candidate_count"] = sum(row["execution_status"] == "ATTEMPTED" for row in evidence["candidates"])
        evidence["candidate_numerical_failure_count"] = sum("CANDIDATE_NUMERICAL_FAILURE" in row["reasons"]
            for row in evidence["candidates"])
        evidence["candidate_evaluation_complete"] = (evidence["candidate_search_complete"]
            and evidence["attempted_candidate_count"] == evidence["candidate_count"]
            and evidence["candidate_numerical_failure_count"] == 0)
        evidence["numerical_consumer_complete"] = bool(evidence["candidate_evaluation_complete"]
            and all(row.get("numerical_consumer_status") == "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
                    for row in evidence["candidates"]))
        if query is not None:
            evidence["query_resources"] = dict(query.statistics)
    if result["status"] == "CONDITIONAL_BEST_FIT" and not (
            evidence["candidate_evaluation_complete"] and evidence["numerical_consumer_complete"]):
        result.update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=["INCOMPLETE_SELECTED_CANDIDATE_EVALUATION"])
        evidence.pop("selected_candidate_index", None)
    return _bounded_report(result, maximum)
