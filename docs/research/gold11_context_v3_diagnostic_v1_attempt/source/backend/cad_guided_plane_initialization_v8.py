"""V8 screens all proposed signs in the already-bounded generated search.

This additive path remains opt-in, nonexhaustive and uncalibrated. It removes
V7's outer32-unsigned rank omission. A hard1024 eligible-ID bound refuses the
whole obligation instead of selecting a prefix; existing finite-work caps apply
to every proposed sign. V6 inner search,48-start refinement, all finite-support/
numerical/registration gates, V7 compact extraction and V6 refusal helper remain.
No supplied support dictionaries, CAD measurements, pose or Gold enters inference.
"""

from dataclasses import asdict, dataclass

import numpy as np

from . import cad_guided_plane_initialization_v4 as _old
from . import cad_guided_noisy_mesh_planes_v5 as _noisy
from . import cad_guided_noisy_mesh_planes_v7 as _compact
from .cad_guided_report_envelope_v6 import bound_initialization_report_v6 as _bounded_report
from . import cad_guided_plane_search_v6 as _search


_v1, _v2, _patch, _surface = _old._v1, _old._v2, _old._patch, _old._surface
_require, _digest, _pose = _old._require, _old._digest, _old._pose


_LIMITS = dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
    max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
    max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
    chunk_rows=65536, max_node_visits=512_000_000,
    max_triangle_pairs=256_000_000, max_normal_pairs=32_000_000,
    max_work_units=4_000_000_000, max_face_build_nodes=131071,
    max_face_build_triangle_visits=10_000_000, max_refined_start_groups=48,
    max_report_bytes=100_663_296)


@dataclass(frozen=True)
class PlaneInitializationBudgetsV8:
    max_hypothesis_records: int = 8192
    max_prefilter_unsigned_assignments: int = 1024
    max_cached_cad_faces: int = 2048
    max_indexed_face_triangles: int = 500_000
    max_prefilter_query_calls: int = 32768
    max_prefilter_query_rows: int = 64_000_000
    chunk_rows: int = 65536
    max_node_visits: int = 512_000_000
    max_triangle_pairs: int = 256_000_000
    max_normal_pairs: int = 32_000_000
    max_work_units: int = 4_000_000_000
    max_face_build_nodes: int = 131071
    max_face_build_triangle_visits: int = 10_000_000
    max_refined_start_groups: int = 48
    max_report_bytes: int = 100_663_296


def _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
        cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets):
    _require(type(budgets) is PlaneInitializationBudgetsV8, "INVALID_V8_INITIALIZATION_BUDGET_TYPE")
    cap = {key: getattr(budgets, key) for key in _LIMITS}
    _require(all(type(value) is int and 0 <= value <= _LIMITS[key] for key, value in cap.items())
        and cap["chunk_rows"] > 0, "INVALID_V8_INITIALIZATION_BUDGET")
    ng, nb = _noisy._configuration(noisy_policy, noisy_budgets)
    cg, cb = _patch._configuration(cad_patch_policy, cad_patch_budgets)
    sg, sb = _search._configuration(search_policy, search_budgets)
    surface = _surface._budgets(surface_budgets)
    return dict(policy=_old._registration_policy(registration_policy),
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(**ng), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(**nb),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(**cg), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(**cb),
        search_policy=_search.PlaneSearchPolicyV6(**sg), search_budgets=_search.PlaneSearchBudgetsV6(**sb),
        budgets=PlaneInitializationBudgetsV8(**cap),
        surface_budgets=_surface.TriangleSurfaceBudgets(**asdict(surface)))


def _base():
    result = _old._base()
    result.update(schema_version="noisy_plane_initialization_prefilter_v8", search_exhaustive=False,
        search_scope="ALL_GENERATED_PROPOSED_SIGNS_SCREENED_UPSTREAM_SEARCH_AND_REFINEMENT_NOT_EXHAUSTIVE",
        noise_bound_status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        global_pose_uniqueness="NOT_ESTABLISHED", selected_unsigned_assignment_ids=[],
        prefilter_schedule="ALL_GENERATED_PROPOSED_SIGNS_HARD_UNSIGNED_BOUND_NO_RANK_OMISSION",
        refinement_schedule="48_STRATIFIED_EXACT_PASSING_START_GROUPS",
        all_passing_start_groups=[], selected_policy_evaluation_complete=False)
    result["statistics"].update(proposed_hypothesis_count=0, scheduled_hypothesis_count=0,
        omitted_proposed_hypothesis_count=0, search_rejected_hypothesis_count=0,
        eligible_unsigned_assignment_count=0, scheduled_unsigned_assignment_count=0,
        omitted_unsigned_assignment_count=0, exact_passing_start_group_count=0,
        omitted_passing_start_group_count=0)
    return result


def _prepare(scan_points, raw_triangles, cad_triangles, cad_face_ids, cad_source_triangle_ids,
             epsilon, configuration, result, unit_assumption):
    scan, raw, errors = _noisy._inputs(scan_points, raw_triangles, epsilon,
                                      asdict(configuration["noisy_budgets"]))
    maximum = min(configuration["cad_patch_budgets"].max_triangles,
                  configuration["budgets"].max_indexed_face_triangles)
    cad = _old._array(cad_triangles, (3, 3), "fiu", maximum, "<f8", "CAD_TRIANGLES")
    faces = _old._array(cad_face_ids, (), "iu", len(cad), "<i8", "CAD_FACE_IDS")
    local = _old._array(cad_source_triangle_ids, (), "iu", len(cad), "<i8", "CAD_SOURCE_TRIANGLE_IDS")
    _require(len(cad) == len(faces) == len(local) and np.all(faces <= 2**53), "INVALID_CAD_SOURCE_BINDING")
    result["input_digests"] = dict(scan_points_sha256=_digest(scan),
        scan_triangle_raw_rows_sha256=_digest(raw), position_error_bounds_mm_sha256=_digest(errors),
        cad_triangles_sha256=_digest(cad), cad_face_ids_sha256=_digest(faces),
        cad_source_triangle_ids_sha256=_digest(local))
    result["input_counts"] = dict(scan_raw_rows=len(scan), scan_triangles=len(raw), cad_triangles=len(cad))
    # Only the later V12 API can supply this validated explicit successor mode.
    # Older callers keep the exact V7 producer and original proposal namespace.
    mode = configuration.get("scan_plane_mode", "v7")
    _require(type(mode) is str and mode in ("v7", "composite_v1"), "INVALID_SCAN_PLANE_MODE")
    if mode == "composite_v1":
        from .cad_guided_scan_plane_preparation_v1 import prepare_composite_scan_planes_v1
        patches = prepare_composite_scan_planes_v1(scan, raw, position_error_bounds_mm=errors,
            noisy_policy=configuration["noisy_policy"], noisy_budgets=configuration["noisy_budgets"],
            seeded_budgets=configuration["seeded_plane_budgets"],
            budgets=configuration["composite_plane_budgets"])
    else:
        patches = _compact.extract_noisy_scan_plane_proposals_v7(scan, raw,
            position_error_bounds_mm=errors, policy=configuration["noisy_policy"], budgets=configuration["noisy_budgets"])
    result["patch_evidence"]["scan"] = _compact.encode_metadata_v7(patches)
    if mode == "composite_v1":
        _require(patches["schema_version"] == "composite_scan_planes_v1"
            and patches["status"] == "COMPLETE"
            and patches["statistics"]["selected_policy_evaluation_complete"] is True,
            "COMPOSITE_SCAN_PLANE_PROPOSALS_UNAVAILABLE")
        _require(type(patches["lanes"]) is dict and tuple(patches["lanes"]) == ("v7", "seeded_v1")
            and all(patches["lanes"][lane]["status"] == "COMPLETE" for lane in ("v7", "seeded_v1")),
            "INCOMPLETE_COMPOSITE_PREPARATION_LANES")
        expected_origins = [(lane, ref["kind"], ref["record_id"])
            for lane in ("v7", "seeded_v1") for ref in patches["lanes"][lane]["proposals"]]
        _require(type(patches["planes"]) is list
            and [(row["lane"], row["kind"], row["record_id"]) for row in patches["planes"]] == expected_origins
            and type(patches["statistics"]["qualified_plane_count"]) is int
            and patches["statistics"]["qualified_plane_count"] == len(expected_origins),
            "INCOMPLETE_COMPOSITE_PRESEARCH_ORIGIN_TABLE")
    else:
        _require(patches["status"] == "COMPLETE" and patches["statistics"]["core_evaluation_complete"]
            and patches["statistics"]["scheduled_closure_evaluation_complete"], "NOISY_SCAN_PLANE_PROPOSALS_UNAVAILABLE")
    _require(patches["input_digests"] == dict(points_sha256=_digest(scan),
        triangle_raw_rows_sha256=_digest(raw), position_error_bounds_mm_sha256=_digest(errors)),
        "NOISY_SCAN_PROPOSAL_INPUT_BINDING_MISMATCH")
    cad_patches = _patch.fit_cad_face_planes_v4(cad, source_face_indices=faces, source_triangle_indices=local,
        policy=configuration["cad_patch_policy"], budgets=configuration["cad_patch_budgets"])
    result["patch_evidence"]["cad"] = _old._compact_patches(cad_patches)
    _require(cad_patches["status"] == "AVAILABLE", "CAD_COMPLETE_FACE_PROPOSALS_UNAVAILABLE")
    supports, observed = {}, []
    proposals = patches["planes"] if mode == "composite_v1" else patches["proposals"]
    for index, proposal in enumerate(proposals):
        kind, identifier = proposal["kind"], proposal["record_id"]
        if mode == "composite_v1":
            lane = proposal["lane"]
            _require(type(proposal["plane_id"]) is int and proposal["plane_id"] == index
                and type(identifier) is int and identifier >= 0
                and ((lane == "v7" and kind in ("CORE", "CLOSURE"))
                     or (lane == "seeded_v1" and kind == "SEEDED_REGION")),
                "INVALID_OWNED_COMPOSITE_ORIGIN")
            table = ("cores" if kind == "CORE" else "closures") if lane == "v7" else "regions"
            record = patches["lanes"][lane][table][identifier]
            id_key = {"CORE": "core_id", "CLOSURE": "closure_id", "SEEDED_REGION": "region_id"}[kind]
            _require(type(record[id_key]) is int and record[id_key] == identifier
                and all(record[key] == proposal[key] for key in
                    ("triangle_rows_sha256", "raw_vertex_rows_sha256")), "STALE_OWNED_COMPOSITE_ORIGIN")
            proposal_kind, source_id = lane + ":" + kind, index
            search_kind = kind if lane == "v7" else "DECLARED"
        else:
            _require(kind in ("CORE", "CLOSURE") and type(identifier) is int and identifier >= 0,
                     "INVALID_OWNED_NOISY_PROPOSAL_REFERENCE")
            record = patches["cores" if kind == "CORE" else "closures"][identifier]
            _require(record["core_id" if kind == "CORE" else "closure_id"] == identifier,
                     "STALE_OWNED_NOISY_PROPOSAL_REFERENCE")
            proposal_kind, source_id, search_kind = kind, identifier, kind
        analysis = record["analysis"]
        _require(analysis["status"] == "PROPOSED", "UNQUALIFIED_OWNED_NOISY_PROPOSAL")
        supports[index] = dict(record, plane_id=index, proposal_kind=proposal_kind, source_record_id=source_id)
        observed.append(_search.ObservedPlaneV6(_search.PlaneEquationV6(index,
            tuple(analysis["normal"]), tuple(analysis["witness_anchor_mm"]),
            analysis["witness_offset_from_anchor_mm"], analysis["normal_angle_bound_radians"]),
            len(record["raw_vertex_rows"]), analysis["raw_support_rms_mm"],
            tuple(analysis["observed_centroid_mm"]), search_kind))
    nominal = tuple(_search.PlaneEquationV6(row["plane_id"], tuple(row["normal"]),
        tuple(row["area_centroid_mm"]), 0., 0.) for row in cad_patches["patches"])
    search = _search.propose_plane_search_v6(tuple(observed), nominal, unit_assumption=unit_assumption,
        policy=configuration["search_policy"], budgets=configuration["search_budgets"])
    result["hypothesis_evidence"] = {key: value for key, value in search.items() if key != "hypotheses"}
    if search["status"] != "COMPLETE":
        result["hypothesis_evidence"]["invalidated_hypotheses"] = search.get("hypotheses", [])
    _require(search["status"] == "COMPLETE" and search["selected_policy_evaluation_complete"],
             "BOUNDED_PLANE_SEARCH_UNAVAILABLE")
    return dict(scan=scan, raw_triangles=raw, errors=errors, cad=cad, faces=faces, local=local,
        scan_patches=patches, scan_supports=supports, cad_patches=cad_patches,
        hypotheses=search["hypotheses"])


def _bound_scan(context, patch):
    rows, raw = patch["triangle_rows"], patch["raw_vertex_rows"]
    _require(type(rows) is np.ndarray and rows.ndim == 1 and rows.dtype.kind == "i"
        and len(rows) > 0 and np.all(rows >= 0) and np.all(rows < len(context["raw_triangles"]))
        and np.array_equal(rows, np.unique(rows)), "INVALID_NOISY_TRIANGLE_SUPPORT")
    _require(type(raw) is np.ndarray and raw.ndim == 1 and raw.dtype.kind == "i"
        and np.array_equal(raw, np.unique(context["raw_triangles"][rows]))
        and len(raw) == patch["analysis"]["raw_support_count"]
        and _digest(rows) == patch["triangle_rows_sha256"] and _digest(raw) == patch["raw_vertex_rows_sha256"],
        "STALE_OR_INCOMPLETE_NOISY_RAW_SUPPORT")


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
                coverage=None, ambiguous_rows=0, boundary_rows=0, status="NOT_ATTEMPTED", reasons=[]))
    _require(stats["planned_query_rows"] <= budget.max_prefilter_query_rows
        and stats["planned_query_calls"] <= budget.max_prefilter_query_calls,
        "PREFILTER_PREFLIGHT_BUDGET_EXCEEDED")
    query = _old._FaceQueries(context, used_faces, budget, surface, result)
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
                unambiguous = _v2._unambiguous(answer)
                support["query_rows"] += len(points)
                support["finite_unambiguous_inlier_count"] += int(np.count_nonzero(unambiguous
                    & (answer["distances"] <= policy.max_correspondence_distance_mm)))
                support["ambiguous_rows"] += int(np.count_nonzero(~unambiguous))
                support["boundary_rows"] += int(np.count_nonzero(answer["closest_regions"] != "FACE"))
            support["coverage"] = support["finite_unambiguous_inlier_count"]/support["support_count"]
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
        groups[group_index]["refinement_disposition"] = "SCHEDULED_FOR_V2_REFINEMENT"
        result["start_groups"].append(dict(groups[group_index], index=index))
    stats.update(exact_passing_start_group_count=len(groups), surviving_start_group_count=len(indices),
                 omitted_passing_start_group_count=len(groups)-len(indices))
    _require(stats["hypotheses_evaluated"] == stats["scheduled_hypothesis_count"],
             "PREFILTER_SELECTED_DENOMINATOR_INCOMPLETE")
    result.update(status="COMPLETE", search_complete=True, selected_policy_evaluation_complete=True,
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


def prefilter_plane_initializations_v8(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm, registration_policy,
        unit_assumption, noisy_policy=_noisy.NoisyMeshPlanePolicyV5(),
        noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(), cad_patch_policy=_patch.CoplanarPatchPolicyV4(),
        cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(), search_policy=_search.PlaneSearchPolicyV6(),
        search_budgets=_search.PlaneSearchBudgetsV6(), budgets=PlaneInitializationBudgetsV8(),
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
    return _bounded_report(result, maximum)


def _registration_rows(context, supplied, evidence):
    if supplied is None:
        return _old._registration_rows(context, None, evidence)
    rows = _old._array(supplied, (), "iu", 200000, "<i8", "REGISTRATION_RAW_ROWS")
    # Early validation of the caller array is not authority over the copy.
    # Revalidate owned IDs BEFORE indexing, including the NumPy-negative alias.
    _require(rows.ndim == 1 and 36 <= len(rows) <= 200000
        and np.all(rows >= 0) and np.all(rows < len(context["scan"]))
        and len(np.unique(rows)) == len(rows), "INVALID_OWNED_REGISTRATION_RAW_ROW_SELECTION")
    points = context["scan"][rows]
    evidence.update(coverage_basis="EXPLICIT_SELECTED_UNIQUE_SCAN_ROWS_TO_TRIANGLES_AND_CAD_SAMPLE_ROWS_TO_SELECTED_SCAN_ROWS",
        registration_raw_rows=dict(selection="EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING", count=len(rows), sha256=_digest(rows)),
        prefilter_full_scan_raw_rows=len(context["scan"]), registration_input_rows=len(points),
        registration_input_points_sha256=_digest(_patch._immutable(points, "<f8")))
    return _v1._canonical_scan(points)


def register_whole_part_planes_v8(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm,
        cad_initialization_points, registration_policy, unit_assumption, registration_raw_rows=None,
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(),
        search_policy=_search.PlaneSearchPolicyV6(), search_budgets=_search.PlaneSearchBudgetsV6(),
        budgets=PlaneInitializationBudgetsV8(), surface_budgets=None, registration_budgets=None):
    """Explicit error-bound declaration and owned supports, no native or file IO.

All original assigned support rows reach finite-face checks. The explicit global
registration selection still has V2's <=200k limit and is separately disclosed.
The returned fit never establishes global pose uniqueness or a stable datum.
"""
    result = _old._registration_result(unit_assumption)
    result["schema_version"] = "noisy_plane_initialized_finite_registration_v8"
    evidence, query = result["evidence"], None
    evidence.update(schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V8",
        candidate_search_status="ALL_GENERATED_PROPOSED_SIGNS_FINITE_SCREENED_NOT_GLOBAL_SEARCH",
        search_exhaustive=False, global_pose_uniqueness="NOT_ESTABLISHED",
        noise_bound_status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION")
    maximum = _LIMITS["max_report_bytes"]
    try:
        configuration = _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
            cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets)
        maximum = configuration["budgets"].max_report_bytes
        execution = _v2._validated_execution_budgets(_v2.SurfaceRegistrationBudgetsV2()
            if registration_budgets is None else registration_budgets)
        execution = _v2.SurfaceRegistrationBudgetsV2(**asdict(execution))
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
        _require(initialization["status"] == "COMPLETE" and initialization["selected_policy_evaluation_complete"],
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
        query = _v2._Queries(built["index"], execution, surface)
        diagonal = float(_v2._norm(np.ptp(built["index"].triangles.reshape(-1, 3), axis=0)))
        evidence.update(scan_unique_rows=len(scan), cad_unique_rows=len(cad),
            coarse_scan_rows=len(coarse), refined_scan_rows=len(refine))
        for row in candidates:
            row.update(execution_status="ATTEMPTED", reasons=[])
            rotation, translation = np.asarray(row["initialized_rotation"]), np.asarray(row["initialized_translation_mm"])
            try:
                rotation, translation, nc, cr, cd = _v2._refine(coarse, query, rotation, translation,
                    diagonal=diagonal, policy=policy, coarse=True)
                rotation, translation, nf, fr, fd = _v2._refine(refine, query, rotation, translation,
                    diagonal=diagonal, policy=policy, coarse=False)
                _pose(dict(rotation=rotation, translation_mm=translation))
                row.update(_v2._diagnose(scan, cad, refine, query, rotation, translation, policy))
                row.update(coarse_iterations=nc, refine_iterations=nf, coarse_convergence_state=cr,
                    refine_convergence_state=fr, coarse_last_step=cd, refine_last_step=fd,
                    rotation=rotation.tolist(), translation=translation.tolist())
            except _v2._SurfaceFailure as error:
                row["reasons"] = [str(error)]
                raise
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                row["reasons"] = ["CANDIDATE_NUMERICAL_FAILURE"]
        evidence["candidate_search_complete"] = True
        _old._v2_select(result, refine, policy)
    except Exception as error:
        result.update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=[_old._reason(error)])
        evidence.pop("selected_candidate_index", None)
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
        if query is not None:
            evidence["query_resources"] = dict(query.statistics)
    if result["status"] == "CONDITIONAL_BEST_FIT" and not evidence["candidate_evaluation_complete"]:
        result.update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=["INCOMPLETE_SELECTED_CANDIDATE_EVALUATION"])
        evidence.pop("selected_candidate_index", None)
    return _bounded_report(result, maximum)
