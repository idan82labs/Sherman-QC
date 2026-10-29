"""Unqualified raw-scan measurements proposed by finite CAD pose contexts.

No context is asserted to be the true pose. CAD bounds row-selection searches;
fixed raw support roles then determine one canonical scan-frame fit. All CAD
associations and per-context fits survive. A collection of conditional support
records is not a jointly established configuration or a manufacturing count.
"""

from dataclasses import asdict, replace
import base64
import hashlib
import json
import math

import numpy as np

from .cad_guided_measurement import CADGuidedFeature, _measure, _plane, _circle, _Unavailable, _unit
from .cad_guided_local_measurement_v1 import _crop_rows
from .cad_guided_instances import SupportPolicy, build_finite_support_instance
from .cad_guided_mesh_offline_v2 import _rigid_transform
from . import cad_guided_stage_policies_v1 as _stages
from . import cad_guided_global_registration_v3 as _v3
from . import cad_guided_context_surface_roles_v1 as _roles


MAX_RAW_ROWS = 1_000_000
MAX_LOCAL_ROWS = 200_000
MAX_CONTEXT_WINDOWS = 4096
MAX_TOTAL_LOCAL_ROWS = 4_000_000
MAX_CROP_ROW_VISITS = 64_000_000
MAX_SUPPORT_RECORDS = 512
MAX_REPORT_BYTES = 64 * 1024**2
MAX_OWNERSHIP_PAIRS = 4_000_000
MAX_ROLE_MEMBERSHIP_VISITS = 64_000_000
MAX_ROLE_INCIDENCE_VISITS = 64_000_000
MAX_ROLE_CACHE_ENTRIES = 128
FIT_FIELDS = ("max_fit_residual_mm", "max_tangency_error_mm", "min_arc_sweep_deg",
              "max_circle_condition", "min_plane_spread_ratio", "max_normal_deviation_deg", "fit_model")


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _fit_policy(feature):
    return {key: getattr(feature, key) for key in FIT_FIELDS}


def _canonical_direction(vector):
    result = _unit(vector)
    if result[int(np.argmax(np.abs(result)))] < 0:
        result = -result
    return result


def _raw_feature(feature, rotation, translation):
    # p_cad = R p_scan + t. No best-fit/verified-datum assertion is made.
    return replace(feature, origin=tuple(rotation.T@(np.asarray(feature.origin)-translation)),
        axis=tuple(rotation.T@np.asarray(feature.axis)),
        start_radial=tuple(rotation.T@np.asarray(feature.start_radial)))


def _canonical_fit(points, arc_ids, flange_ids, feature, instance_policy, record_id):
    """Refit FIXED raw roles, with no nominal center/radius/axis in the answer.

    Role selection remains CAD-context-conditioned. Canonical signs are raw-frame
    serialization conventions, not material normals or manufacturing bend sense.
    Fits/residuals are point estimates, not bounded-error inverse certificates.
    """
    roles = [np.asarray(arc_ids, dtype=np.int64),
             *[np.asarray(ids, dtype=np.int64) for ids in flange_ids]]
    _require(all(12 <= len(ids) <= MAX_LOCAL_ROWS for ids in roles), "INVALID_CANONICAL_SUPPORT_SIZE")
    _require(len(np.unique(np.concatenate(roles))) == sum(map(len, roles)), "OVERLAPPING_RAW_SUPPORT_ROLES")
    subsets = [points[ids] for ids in roles]
    _require(all(len(np.unique(subset, axis=0)) == len(subset) for subset in subsets),
             "DUPLICATE_POSITION_IN_CANONICAL_SUPPORT")
    _require(len(np.unique(np.vstack(subsets), axis=0)) == sum(map(len, subsets)),
             "OVERLAPPING_CANONICAL_SUPPORT_POSITIONS")
    planes = []
    for subset in subsets[1:]:
        _, _, vh = np.linalg.svd(subset-subset.mean(axis=0), full_matrices=False)
        reference = _canonical_direction(vh[-1])
        planes.append(_plane(subset, reference, feature))
    axis = _canonical_direction(np.cross(planes[0][0], planes[1][0]))
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = _unit(np.cross(axis, basis))
    v = np.cross(axis, u)
    origin = subsets[0].mean(axis=0)
    q = np.column_stack(((subsets[0]-origin)@u, (subsets[0]-origin)@v))
    center2, radius, residual, condition, min_singular = _circle(q, feature)
    center = origin+center2[0]*u+center2[1]*v
    radials, tangency = [], []
    for normal, plane_center, _ in planes:
        radial = normal-(normal@axis)*axis
        radial = _unit(radial)
        if float(radial@(plane_center-center)) < 0:
            radial = -radial
        radials.append(radial)
        tangency.append(float(radial@(plane_center-center)-radius))
    if max(map(abs, tangency)) > feature.max_tangency_error_mm:
        raise _Unavailable("CANONICAL_CIRCLE_NOT_TANGENT_TO_SCAN_FLANGES")
    sweep = math.atan2(float(axis@np.cross(*radials)), float(radials[0]@radials[1]))
    if not 20 <= abs(math.degrees(sweep)) <= 160:
        raise _Unavailable("CANONICAL_SWEEP_OUTSIDE_MODEL")
    sign = math.copysign(1., sweep)
    radial_points = subsets[0]-center
    angles = sign*np.arctan2(radial_points@np.cross(axis, radials[0]), radial_points@radials[0])
    if np.any(angles < -1e-8) or np.any(angles > abs(sweep)+1e-8):
        raise _Unavailable("CANONICAL_ARC_OUTSIDE_MEASURED_CONTACT_SECTOR")
    if math.degrees(float(np.ptp(angles))) < feature.min_arc_sweep_deg:
        raise _Unavailable("CANONICAL_ARC_ANGULAR_COVERAGE")
    measured = dict(measured_radius_mm=radius, measured_sweep_magnitude_deg=abs(math.degrees(sweep)),
        measured_sweep_deg=math.degrees(sweep), sweep_sign_convention="CANONICAL_RAW_AXIS_AND_RAW_ROLE_ORDER",
        measured_axis_direction=axis.tolist(), measured_axis_point=center.tolist(),
        circle_max_residual_mm=residual, flange_max_residual_mm=[row[2] for row in planes],
        measured_tangency_residual_mm=tangency, free_circle_jacobian_condition=condition,
        free_circle_min_singular_per_sqrt_n=min_singular, fit_model="free_circle",
        tangency_check_role="FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK",
        geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        status="CONDITIONAL_SCAN_POINT_ESTIMATE", radius_uncertainty_interval_mm=None,
        sweep_uncertainty_interval_deg=None, physical_trim_endpoints=None,
        datum_relative_deviation=None, material_side_status="CONTEXT_ASSUMED_NOT_VERIFIED")
    # The old finite-support constructor is used on the same fixed raw roles.
    # Compact locally only to its array cap; original raw IDs remain in the record.
    selected = np.unique(np.concatenate(roles))
    _require(len(selected) <= MAX_LOCAL_ROWS, "CANONICAL_LOCAL_POINT_BUDGET")
    local_roles = [np.searchsorted(selected, ids) for ids in roles]
    finite = build_finite_support_instance(points[selected], measured,
        arc_row_indices=local_roles[0], flange_row_indices=local_roles[1:],
        feature_id=record_id, policy=instance_policy)
    if finite["status"] == "SUPPORTED":
        carrier_offset = radius*_unit(radials[0]+radials[1])
        instance = dict(finite["instance"])
        instance["axis_support_line_start"] = instance["line_start"]
        instance["axis_support_line_end"] = instance["line_end"]
        for key in ("line_start", "line_end"):
            instance[key] = (np.asarray(instance[key])+carrier_offset).tolist()
        instance.update(line_convention="MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR",
            geometry_frame=measured["geometry_frame"],
            reference_carrier_compatibility="UNVERIFIED_NO_GOLD_IN_CARRIER_SELECTION",
            material_side_status="CONTEXT_ASSUMED_NOT_VERIFIED")
        finite = dict(finite, instance=instance)
    return measured, finite


def _base():
    return dict(schema="CAD_GUIDED_GENERATED_CONTEXT_MEASUREMENTS_V1", status="UNAVAILABLE",
        reasons=[], support_records=[], context_windows=[], predictions=[],
        observed_bend_count=None, observed_bend_count_status="MANUFACTURING_EVENT_IDENTITY_UNKNOWN",
        context_quantifier="EXISTS_COMPLETED_GENERATED_CONTEXT",
        joint_configuration_status="NOT_ESTABLISHED", continuous_pose_family_coverage="NOT_ESTABLISHED",
        registration_status="NOT_ACCEPTED_CONTEXT_PROPOSALS_ONLY", production_use_allowed=False,
        conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED_NO_PROPAGATED_MEASUREMENT_INTERVAL",
        geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        endpoint_semantics="SAMPLED_WINDOW_SUPPORT_NOT_PHYSICAL_TRIM")


def inspect_generated_context_windows_v1(raw_points, features, *, registration, scan_sha256, instance_policy,
                                         surface_roles=None):
    """Use only completed V12 contexts; no labels/truth/datum/nominal answer input.

    This array API checks in-process coordinate digests, not raw-file authority.
    The byte adapter must authenticate scan_sha256 and the registration inputs.
    Late malformed data or aggregate budget failures invalidate the whole output.
    Expected per-window unsupported geometry remains in the complete ledger.
    """
    result = _base()
    try:
        points = np.asarray(raw_points)
        _require(points.ndim == 2 and points.shape[1] == 3 and points.dtype.kind in "fiu"
            and 1 <= len(points) <= MAX_RAW_ROWS and np.isfinite(points).all()
            and np.max(np.abs(points)) <= 1e6, "INVALID_RAW_CONTEXT_POINTS")
        points = np.frombuffer(np.asarray(points, dtype="<f8").tobytes(), dtype="<f8").reshape(-1, 3)
        _require(points.shape[1:] == (3,) and 1 <= len(points) <= MAX_RAW_ROWS
            and np.isfinite(points).all() and np.max(np.abs(points)) <= 1e6,
            "INVALID_OWNED_RAW_CONTEXT_POINTS")
        _require(type(scan_sha256) is str and len(scan_sha256) == 64
            and all(c in "0123456789abcdef" for c in scan_sha256), "INVALID_SCAN_DIGEST_DECLARATION")
        _require(type(instance_policy) is SupportPolicy, "INVALID_CONTEXT_INSTANCE_POLICY")
        instance_policy = SupportPolicy(**asdict(instance_policy))
        _require(type(features) in (tuple, list) and 1 <= len(features) <= MAX_SUPPORT_RECORDS
            and all(type(f) is CADGuidedFeature and f.fit_model == "free_circle" for f in features)
            and len({f.feature_id for f in features}) == len(features), "FREE_CIRCLE_FEATURES_REQUIRED")
        features = tuple(CADGuidedFeature(**asdict(f)) for f in features)
        _require(type(registration) is dict
            and registration.get("schema_version") == "noisy_plane_initialized_finite_registration_v12"
            and registration.get("status") == "CONDITIONAL_MEASUREMENT_CONTEXTS"
            and registration.get("transform_scan_to_cad") is None, "COMPLETED_V12_CONTEXTS_REQUIRED")
        evidence = registration["evidence"]
        _require(evidence["candidate_evaluation_complete"] is True
            and evidence["numerical_consumer_complete"] is True
            and evidence["accepted_registration"] is False
            and evidence["connected_measurement_contexts"] is True
            and evidence["stage_policies"] == _stages.ablation_stage_policies_v1(connected=True)
            and evidence["input_digests"]["scan_points_sha256"] == hashlib.sha256(points.tobytes()).hexdigest(),
            "INCOMPLETE_OR_UNBOUND_CONTEXT_EVIDENCE")
        _require(type(surface_roles) is _roles.PreparedContextSurfaceRolesV1
            and surface_roles.points_sha256 == evidence["input_digests"]["scan_points_sha256"]
            and surface_roles.triangle_raw_rows_sha256 == evidence["input_digests"]["scan_triangle_raw_rows_sha256"]
            and surface_roles.position_error_bounds_mm_sha256 == evidence["input_digests"]["position_error_bounds_mm_sha256"]
            and surface_roles.metadata_sha256 == evidence["surface_role_preparation"]["metadata_sha256"]
            and surface_roles.raw_row_count == len(points), "MISSING_OR_UNBOUND_OWNED_SURFACE_ROLES")
        candidates = evidence["candidates"]
        indices = evidence["measurement_context_indices"]
        _require(type(candidates) is list and 1 <= len(candidates) <= 48
            and type(indices) is list and indices == [i for i, row in enumerate(candidates)
                if row["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"]
            and indices and all(type(i) is int for i in indices)
            and all(type(row["index"]) is int and row["index"] == i
                    and row["execution_status"] == "ATTEMPTED"
                    and row["numerical_consumer_status"] == _v3.CONSUMER_COMPLETE
                    and row["numerical_support_policy"] == _v3.SUPPORT_POLICY
                    and row["distance_model"] == _v3.DISTANCE_MODEL
                    and row["optimizer_objective"] == _v3.SURROGATE
                    and row["conditional_context_support"]["proximity_policy"] == _stages.PROXIMITY
                    and row["conditional_context_support"]["accepted_registration"] is False
                    for i, row in enumerate(candidates)), "INVALID_CONTEXT_DENOMINATOR")
        # Snapshot strict numeric poses before coercion or fitting. The digest
        # binds each context's final finite query to this exact represented pose.
        poses = {}
        for i, candidate in enumerate(candidates):
            r, t = np.asarray(candidate["rotation"]), np.asarray(candidate["translation"])
            _require(r.shape == (3, 3) and t.shape == (3,) and r.dtype.kind in "fiu"
                and t.dtype.kind in "fiu", "INVALID_CONTEXT_POSE_ARRAYS")
            r = np.frombuffer(np.asarray(r, dtype="<f8").tobytes(), dtype="<f8").reshape(3, 3)
            t = np.frombuffer(np.asarray(t, dtype="<f8").tobytes(), dtype="<f8")
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = r, t
            _rigid_transform(matrix)
            _require(candidate["conditional_context_support"]["pose_sha256"] == hashlib.sha256(
                np.concatenate((r.ravel(), t)).astype("<f8").tobytes()).hexdigest(),
                "CONTEXT_QUERY_POSE_BINDING_MISMATCH")
            poses[i] = (r, t, matrix.tolist())
        _require(len(indices)*len(features) <= MAX_CONTEXT_WINDOWS, "CONTEXT_WINDOW_PREFLIGHT_BUDGET")
        _require(len(indices)*len(features)*len(points) <= MAX_CROP_ROW_VISITS,
                 "FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET")
        indices = tuple(indices)
        records, lookup, total, role_cache = result["support_records"], {}, 0, {}
        result.update(scan_sha256=scan_sha256, raw_points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                      surface_role_source_binding=dict(points_sha256=surface_roles.points_sha256,
                          triangle_raw_rows_sha256=surface_roles.triangle_raw_rows_sha256,
                          position_error_bounds_mm_sha256=surface_roles.position_error_bounds_mm_sha256,
                          encoded_patch_metadata_sha256=surface_roles.metadata_sha256),
                      instance_policy=asdict(instance_policy),
                      attempted_crop_row_visits=0, completed_crop_row_visits=0,
                      canonical_refit_row_visits=0,
                      role_membership_visits=0, role_incidence_corner_visits=0,
                      surface_role_filters=[],
                      attempted_context_count=len(indices), planned_window_count=len(indices)*len(features))
        for index in indices:
            rotation, translation, matrix = poses[index]
            for feature in features:
                raw_feature = _raw_feature(feature, rotation, translation)
                axis = _canonical_direction(np.asarray(raw_feature.axis, dtype=np.float64))
                axis_key = axis.tobytes()
                if axis_key not in role_cache:
                    _require(len(role_cache) < MAX_ROLE_CACHE_ENTRIES, "CONTEXT_ROLE_CACHE_BUDGET")
                    try:
                        filtered = _roles.select_context_measurement_rows_v1(surface_roles, axis,
                            max_triangle_membership_visits=min(20_000_000,
                                MAX_ROLE_MEMBERSHIP_VISITS-result["role_membership_visits"]),
                            max_incidence_corner_visits=min(6_000_000,
                                MAX_ROLE_INCIDENCE_VISITS-result["role_incidence_corner_visits"]))
                    except _roles.ContextSurfaceRolesUnavailableV1 as error:
                        result["failed_role_selection_work"] = error.statistics
                        raise
                    metadata = filtered["metadata"]
                    result["role_membership_visits"] += metadata["attempted_triangle_membership_visits"]
                    result["role_incidence_corner_visits"] += metadata["attempted_incidence_corner_visits"]
                    exclusion = np.zeros(len(points), dtype=np.uint8)
                    exclusion[filtered["excluded_raw_rows"]] = 1
                    publication = dict(metadata,
                        excluded_raw_row_bitmap=dict(encoding="BASE64_PACKBITS_LITTLE_ZERO_PADDED",
                            bit_count=len(points), data=base64.b64encode(np.packbits(exclusion, bitorder="little").tobytes()).decode()))
                    role_cache[axis_key] = (filtered["keep_raw_rows"], len(result["surface_role_filters"]))
                    result["surface_role_filters"].append(publication)
                kept, filter_index = role_cache[axis_key]
                result["attempted_crop_row_visits"] += len(points)
                selected = _crop_rows(points, raw_feature)
                result["completed_crop_row_visits"] += len(points)
                before_role_filter = len(selected)
                selected = np.intersect1d(selected, kept, assume_unique=True)
                total += len(selected)
                _require(total <= MAX_TOTAL_LOCAL_ROWS, "AGGREGATE_CONTEXT_ROW_BUDGET")
                window = dict(context_index=index, feature_id=feature.feature_id, status="UNAVAILABLE",
                    reasons=[], local_input_rows=len(selected), support_record_id=None,
                    surface_role_filter_index=filter_index,
                    pre_role_filter_roi_rows=before_role_filter,
                    excluded_cap_like_roi_rows=before_role_filter-len(selected))
                result["context_windows"].append(window)
                if not 1 <= len(selected) <= MAX_LOCAL_ROWS:
                    window["reasons"] = ["LOCAL_SUPPORT_UNAVAILABLE_OR_OVER_BUDGET"]
                    continue
                try:
                    measured = _measure(points[selected], raw_feature)
                    support = measured.pop("_instance_support")
                    measured.pop("support_row_indices")
                except _Unavailable as error:
                    window["reasons"] = [str(error)]
                    continue
                arc = tuple(int(i) for i in selected[np.asarray(support["arc_row_indices"], dtype=np.int64)])
                ordered_flange = tuple(tuple(int(i) for i in selected[np.asarray(ids, dtype=np.int64)])
                                       for ids in support["flange_row_indices"])
                flanges = tuple(sorted(ordered_flange))
                key = _json_digest(dict(scan_sha256=scan_sha256, arc_rows=arc, flange_rows=flanges,
                    fit_policy=_fit_policy(feature), instance_policy=asdict(instance_policy),
                    surface_assumption="SINGLE_INNER_FACE_UNVERIFIED",
                    canonical_frame="RAW_AXIS_MAX_COMPONENT_POSITIVE_AND_RAW_ROLE_ORDER"))
                if key not in lookup:
                    _require(len(records) < MAX_SUPPORT_RECORDS, "SUPPORT_RECORD_BUDGET")
                    lookup[key] = len(records)
                    records.append(dict(record_id=key, status="PENDING_CANONICAL_REFIT", reasons=[],
                        raw_support=dict(arc_rows=list(arc), flange_rows=[list(ids) for ids in flanges]),
                        fit_policy=_fit_policy(feature), associations=[], measurements=None, finite_support=None))
                record = records[lookup[key]]
                _require(record["raw_support"] == dict(arc_rows=list(arc), flange_rows=[list(ids) for ids in flanges])
                    and record["fit_policy"] == _fit_policy(feature), "CONTRADICTORY_SUPPORT_EQUIVALENCE_DIGEST")
                record["associations"].append(dict(context_index=index, feature_id=feature.feature_id,
                    nominal_source=feature.nominal_source, nominal_radius_mm=feature.nominal_radius_mm,
                    nominal_sweep_deg=feature.nominal_sweep_deg,
                    ordered_flange_raw_rows=[list(ids) for ids in ordered_flange],
                    surface_role_filter_index=filter_index,
                    per_context_measurements=measured,
                    transform_scan_to_cad=matrix, accepted_registration=False))
                window.update(status="MEASURED_CONTEXT", support_record_id=key)
        # Canonical refits occur once per fixed-role class. No first/best fit is
        # selected from the per-context estimates. Those estimates remain intact.
        feature_by_id = {f.feature_id: f for f in features}
        for record in records:
            feature = feature_by_id[record["associations"][0]["feature_id"]]
            _require(_fit_policy(feature) == record["fit_policy"], "CANONICAL_FIT_POLICY_BINDING")
            try:
                result["canonical_refit_row_visits"] += len(record["raw_support"]["arc_rows"])+sum(
                    len(ids) for ids in record["raw_support"]["flange_rows"])
                _require(result["canonical_refit_row_visits"] <= MAX_TOTAL_LOCAL_ROWS,
                         "CANONICAL_REFIT_ROW_BUDGET")
                measured, finite = _canonical_fit(points, record["raw_support"]["arc_rows"],
                    record["raw_support"]["flange_rows"], feature, instance_policy, record["record_id"])
                record.update(status="MEASURED_CONDITIONAL", measurements=measured, finite_support=finite)
                for association in record["associations"]:
                    association["canonical_radius_deviation_mm"] = measured["measured_radius_mm"]-association["nominal_radius_mm"]
                    association["canonical_sweep_magnitude_deviation_deg"] = (
                        measured["measured_sweep_magnitude_deg"]-abs(association["nominal_sweep_deg"]))
            except _Unavailable as error:
                record.update(status="UNAVAILABLE", reasons=[str(error)])
        result["ownership_work"] = dict(row_visits=0, competing_claim_pairs=0, invalidated_records=0)
        _invalidate_overlapping_records(records, points, telemetry=result["ownership_work"])
        result["predictions"] = [record["finite_support"]["instance"] for record in records
            if record["status"] == "MEASURED_CONDITIONAL" and record["finite_support"]
            and record["finite_support"]["status"] == "SUPPORTED"]
        result.update(status="PARTIAL" if any(row["status"] == "MEASURED_CONDITIONAL" for row in records) else "UNAVAILABLE",
            computation_status="COMPLETE", total_local_input_rows=total,
            conditional_support_record_count=len(records),
            prediction_collection_status="CONDITIONAL_SUPPORT_RECORDS_NOT_JOINT_BEND_INSTANCES",
            reasons=[] if any(row["status"] == "MEASURED_CONDITIONAL" for row in records)
                else ["NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS"])
        size = 0
        for piece in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(result):
            size += len(piece.encode())
            _require(size <= MAX_REPORT_BYTES, "CONTEXT_REPORT_BUDGET")
        return result
    except Exception as error:
        refused = _base()
        refused.update(reasons=["CONTEXT_MEASUREMENT_FAILURE:"+type(error).__name__+":"+str(error)[:180]],
            attempted_context_windows=len(result["context_windows"]),
            attempted_support_records=len(result["support_records"]),
            discarded_measurement_prefix=True)
        refused["attempted_work"] = {key: result.get(key, 0) for key in (
            "attempted_crop_row_visits", "completed_crop_row_visits", "canonical_refit_row_visits",
            "role_membership_visits", "role_incidence_corner_visits")}
        refused["attempted_work"]["failed_role_selection"] = result.get("failed_role_selection_work")
        refused["attempted_work"]["ownership"] = result.get("ownership_work",
            dict(row_visits=0, competing_claim_pairs=0, invalidated_records=0))
        return refused


def _invalidate_overlapping_records(records, points, *, telemetry=None):
    """All classes involved in arc/arc or arc/flange reuse lose publication.

    Exact position keys additionally catch duplicate raw IDs at the same point.
    Shared flange-only evidence is allowed, but no joint-event claim follows.
    """
    claims, conflicts, pairs, visits = {}, set(), 0, 0
    if telemetry is None:
        telemetry = dict(row_visits=0, competing_claim_pairs=0, invalidated_records=0)
    for index, record in enumerate(records):
        roles = [("arc", record["raw_support"]["arc_rows"]),
                 *[("flange", ids) for ids in record["raw_support"]["flange_rows"]]]
        for role, ids in roles:
            for row in ids:
                visits += 1
                telemetry["row_visits"] = visits
                _require(visits <= MAX_TOTAL_LOCAL_ROWS, "CONTEXT_OWNERSHIP_ROW_BUDGET")
                key = tuple(float(value) for value in points[row])
                for prior, old_role in claims.get(key, ()):
                    pairs += 1
                    telemetry["competing_claim_pairs"] = pairs
                    _require(pairs <= MAX_OWNERSHIP_PAIRS, "CONTEXT_OWNERSHIP_PAIR_BUDGET")
                    if prior != index and (role == "arc" or old_role == "arc"):
                        conflicts.update((index, prior))
                claims.setdefault(key, set()).add((index, role))
    for index in conflicts:
        records[index].update(status="UNAVAILABLE", measurements=None, finite_support=None,
            reasons=["AMBIGUOUS_SHARED_RAW_OR_DUPLICATED_POSITION_SUPPORT"])
    telemetry["invalidated_records"] = len(conflicts)
    return telemetry


def validate_context_measurement_publication_v1(result, *, raw_count, scan_sha256, raw_points_sha256):
    """Uniform fail-closed boundary for malformed publication objects."""
    try:
        return _validate_context_measurement_publication_v1(result, raw_count=raw_count,
            scan_sha256=scan_sha256, raw_points_sha256=raw_points_sha256)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("MALFORMED_CONTEXT_PUBLICATION:"+type(error).__name__) from error


def _validate_context_measurement_publication_v1(result, *, raw_count, scan_sha256, raw_points_sha256):
    """Defensive byte-workflow publication boundary, not a second geometric fit."""
    _require(type(result) is dict and result.get("schema") == "CAD_GUIDED_GENERATED_CONTEXT_MEASUREMENTS_V1"
        and result.get("status") in ("PARTIAL", "UNAVAILABLE")
        and result.get("observed_bend_count") is None
        and result.get("production_use_allowed") is False
        and result.get("conformance") == "INDETERMINATE"
        and result.get("joint_configuration_status") == "NOT_ESTABLISHED"
        and result.get("continuous_pose_family_coverage") == "NOT_ESTABLISHED"
        and result.get("registration_status") == "NOT_ACCEPTED_CONTEXT_PROPOSALS_ONLY"
        and result.get("context_quantifier") == "EXISTS_COMPLETED_GENERATED_CONTEXT"
        and result.get("geometry_frame") == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        "INVALID_CONDITIONAL_MEASUREMENT_PUBLICATION")
    records, predictions = result["support_records"], result["predictions"]
    _require(type(records) is list and len(records) <= MAX_SUPPORT_RECORDS
        and type(predictions) is list and len(predictions) <= MAX_SUPPORT_RECORDS,
        "CONDITIONAL_PUBLICATION_RECORD_BUDGET")
    if result.get("discarded_measurement_prefix") is True:
        _require(result["status"] == "UNAVAILABLE" and records == [] and predictions == []
            and result.get("context_windows") == [], "SCIENTIFIC_PREFIX_AFTER_CONTEXT_FAILURE")
        return
    _require(result.get("computation_status") == "COMPLETE"
        and result.get("scan_sha256") == scan_sha256
        and result.get("raw_points_sha256") == raw_points_sha256,
        "CONTEXT_PUBLICATION_SOURCE_BINDING")
    filters = _validate_published_role_filters(result, raw_count, raw_points_sha256)
    windows = result["context_windows"]
    _require(type(windows) is list and len(windows) <= MAX_CONTEXT_WINDOWS
        and type(result["planned_window_count"]) is int
        and len(windows) == result["planned_window_count"], "INVALID_CONTEXT_WINDOW_DENOMINATOR")
    window_by_key, matched_windows = {}, set()
    for window in windows:
        context, feature, fi = window["context_index"], window["feature_id"], window["surface_role_filter_index"]
        _require(type(context) is int and 0 <= context < 48 and type(feature) is str
            and 1 <= len(feature) <= 200 and type(fi) is int and 0 <= fi < len(filters),
            "INVALID_CONTEXT_WINDOW_FILTER_BINDING")
        key = (context, feature)
        _require(key not in window_by_key, "DUPLICATE_CONTEXT_WINDOW")
        window_by_key[key] = window
        counts = [window[name] for name in ("local_input_rows", "pre_role_filter_roi_rows", "excluded_cap_like_roi_rows")]
        _require(all(type(value) is int and 0 <= value <= raw_count for value in counts)
            and counts[0]+counts[2] == counts[1], "INVALID_CONTEXT_WINDOW_ROLE_COUNTS")
        _require((window["status"] == "UNAVAILABLE" and window["support_record_id"] is None)
            or (window["status"] == "MEASURED_CONTEXT" and type(window["support_record_id"]) is str),
            "INVALID_CONTEXT_WINDOW_DISPOSITION")
    _require({row["surface_role_filter_index"] for row in windows} == set(range(len(filters))),
             "UNREFERENCED_PUBLISHED_ROLE_FILTER")
    expected, seen, arc_owners, flange_owners = [], set(), set(), set()
    associated_row_visits = 0
    for record in records:
        identity = record["record_id"]
        _require(type(identity) is str and len(identity) == 64
            and all(c in "0123456789abcdef" for c in identity) and identity not in seen,
            "INVALID_CANONICAL_RECORD_IDENTITY")
        seen.add(identity)
        raw = record["raw_support"]
        _require(type(raw["flange_rows"]) is list and len(raw["flange_rows"]) == 2,
                 "INVALID_PUBLISHED_SUPPORT_ROLES")
        ids = [raw["arc_rows"], *raw["flange_rows"]]
        _require(all(type(rows) is list and 12 <= len(rows) <= MAX_LOCAL_ROWS
            and all(type(i) is int and 0 <= i < raw_count for i in rows)
            and rows == sorted(set(rows)) for rows in ids), "INVALID_PUBLISHED_RAW_ROWS")
        arc, flange = set(ids[0]), set(ids[1]+ids[2])
        associations = record["associations"]
        _require(type(associations) is list and 1 <= len(associations) <= MAX_CONTEXT_WINDOWS,
                 "INVALID_CONTEXT_ASSOCIATIONS")
        for association in associations:
            context, feature = association["context_index"], association["feature_id"]
            fi = association["surface_role_filter_index"]
            _require(type(context) is int and type(feature) is str and type(fi) is int
                and 0 <= fi < len(filters), "INVALID_ASSOCIATION_FILTER_INDEX")
            key = (context, feature)
            _require(key in window_by_key and key not in matched_windows,
                     "INVALID_ASSOCIATION_WINDOW_BIJECTION")
            window = window_by_key[key]
            _require(window["status"] == "MEASURED_CONTEXT" and window["support_record_id"] == identity
                and window["surface_role_filter_index"] == fi and association["accepted_registration"] is False
                and sorted(association["ordered_flange_raw_rows"]) == sorted(raw["flange_rows"]),
                "ASSOCIATION_WINDOW_ROLE_BINDING_MISMATCH")
            associated_row_visits += sum(map(len, ids))
            _require(associated_row_visits <= MAX_TOTAL_LOCAL_ROWS, "PUBLICATION_ASSOCIATED_SUPPORT_WORK_BUDGET")
            _require(not any(np.any(filters[fi][rows]) for rows in ids),
                     "PUBLISHED_SUPPORT_EXCLUDED_BY_ASSOCIATED_ROLE_FILTER")
            matched_windows.add(key)
        _require(len(flange) == len(ids[1])+len(ids[2]) and not arc & flange,
                 "OVERLAPPING_PUBLISHED_RAW_ROLES")
        if record["status"] != "MEASURED_CONDITIONAL":
            _require(record["status"] == "UNAVAILABLE" and record["measurements"] is None
                and record["finite_support"] is None, "INVALID_UNAVAILABLE_CANONICAL_RECORD")
            continue
        _require(len(flange) == len(ids[1])+len(ids[2]) and not arc & flange
            and not arc & (arc_owners | flange_owners) and not flange & arc_owners,
            "REUSED_PUBLISHED_CANONICAL_SUPPORT")
        arc_owners.update(arc)
        flange_owners.update(flange)
        measured = record["measurements"]
        _require(measured["fit_model"] == "free_circle"
            and measured["status"] == "CONDITIONAL_SCAN_POINT_ESTIMATE"
            and measured["radius_uncertainty_interval_mm"] is None
            and measured["sweep_uncertainty_interval_deg"] is None
            and measured["physical_trim_endpoints"] is None and measured["datum_relative_deviation"] is None
            and measured["geometry_frame"] == result["geometry_frame"], "INVALID_CANONICAL_QUANTITY_CLAIMS")
        _require(type(measured["measured_radius_mm"]) in (int, float)
            and math.isfinite(measured["measured_radius_mm"])
            and 1e-8 <= measured["measured_radius_mm"] <= 1e6
            and type(measured["measured_sweep_magnitude_deg"]) in (int, float)
            and math.isfinite(measured["measured_sweep_magnitude_deg"])
            and 20 <= measured["measured_sweep_magnitude_deg"] <= 160
            and type(measured["measured_sweep_deg"]) in (int, float)
            and abs(measured["measured_sweep_deg"]) == measured["measured_sweep_magnitude_deg"],
            "INVALID_PUBLISHED_CANONICAL_RADIUS_OR_SWEEP")
        finite = record["finite_support"]
        _require(type(finite) is dict and finite["status"] in ("SUPPORTED", "UNAVAILABLE"),
                 "INVALID_CANONICAL_FINITE_SUPPORT")
        if finite["status"] != "SUPPORTED":
            _require(finite["instance"] is None, "INSTANCE_AFTER_FINITE_REFUSAL")
            continue
        instance = finite["instance"]
        _require(instance["feature_id"] == identity and instance["bend_id"] == identity
            and instance["line_convention"] == "MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR"
            and instance["geometry_frame"] == result["geometry_frame"]
            and instance["physical_trim_endpoints"] is None
            and instance["manufacturing_event_identity"] == "UNKNOWN"
            and instance["endpoint_semantics"] == "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
            "INVALID_CONTEXT_CARRIER_SEMANTICS")
        for key in ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end"):
            vector = np.asarray(instance[key])
            _require(vector.shape == (3,) and vector.dtype.kind in "fiu"
                and np.isfinite(vector).all() and np.max(np.abs(vector)) <= 1e6,
                "INVALID_PUBLISHED_CARRIER_ENDPOINT")
        _require(instance["line_start"] != instance["line_end"], "DEGENERATE_PUBLISHED_CARRIER")
        expected.append(instance)
    _require(predictions == expected and (result["status"] != "UNAVAILABLE" or not expected),
             "PREDICTIONS_NOT_EXACT_CANONICAL_RECORD_PROJECTION")
    _require((result["status"] == "PARTIAL") == any(
        record["status"] == "MEASURED_CONDITIONAL" for record in records),
        "CONTEXT_STATUS_NOT_EXACT_MEASUREMENT_DISPOSITION")
    _require(matched_windows == {key for key, row in window_by_key.items() if row["status"] == "MEASURED_CONTEXT"},
             "UNBOUND_MEASURED_CONTEXT_WINDOW")


def _validate_published_role_filters(result, raw_count, raw_points_sha256):
    """Validate a bounded lossless publication, not repeat cap classification."""
    _require(type(raw_count) is int and 1 <= raw_count <= MAX_RAW_ROWS, "INVALID_PUBLICATION_RAW_COUNT")
    binding = result["surface_role_source_binding"]
    keys = ("points_sha256", "triangle_raw_rows_sha256", "position_error_bounds_mm_sha256",
            "encoded_patch_metadata_sha256")
    _require(type(binding) is dict and set(binding) == set(keys)
        and all(type(binding[key]) is str and len(binding[key]) == 64
            and all(c in "0123456789abcdef" for c in binding[key]) for key in keys)
        and binding["points_sha256"] == raw_points_sha256, "INVALID_ROLE_SOURCE_PUBLICATION_BINDING")
    filters = result["surface_role_filters"]
    _require(type(filters) is list and 1 <= len(filters) <= MAX_ROLE_CACHE_ENTRIES
        and len(filters)*raw_count <= MAX_CROP_ROW_VISITS, "ROLE_BITMAP_PUBLICATION_WORK_BUDGET")
    masks = []
    for row in filters:
        _require(type(row) is dict and all(row[key] == binding[key] for key in keys)
            and row["schema_version"] == "context_surface_roles_v1"
            and row["coordinate_frame"] == "RAW_SCAN_UNREGISTERED"
            and row["scope"] == "CONDITIONAL_PATCH_NORMAL_ROLE_NOT_PHYSICAL_CAP_CERTIFICATE"
            and row["role_policy"] == "CAP_AND_NO_COMPETING_NONCAP_OR_UNRESOLVED_ALL_INCIDENT_TRIANGLES"
            and row["numeric_status"] == "ENGINEERING_NORMALIZATION_AND_ANGULAR_INWARD_RESERVE_NOT_CERTIFIED"
            and row["selected_proposal_evaluation_complete"] is True
            and row["production_use_allowed"] is False and row["observed_bend_count"] is None,
            "INVALID_ROLE_FILTER_PUBLICATION_CLAIMS")
        bitmap = row["excluded_raw_row_bitmap"]
        _require(type(bitmap) is dict and bitmap["encoding"] == "BASE64_PACKBITS_LITTLE_ZERO_PADDED"
            and type(bitmap["bit_count"]) is int and bitmap["bit_count"] == raw_count
            and type(bitmap["data"]) is str and len(bitmap["data"]) == 4*((((raw_count+7)//8)+2)//3),
            "INVALID_ROLE_EXCLUSION_BITMAP_ENCODING")
        packed = base64.b64decode(bitmap["data"], validate=True)
        _require(len(packed) == (raw_count+7)//8 and base64.b64encode(packed).decode() == bitmap["data"]
            and (raw_count % 8 == 0 or packed[-1] >> (raw_count % 8) == 0),
            "INVALID_ROLE_EXCLUSION_BITMAP_PADDING")
        excluded = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little", count=raw_count).astype(bool)
        count = int(np.count_nonzero(excluded))
        _require(all(type(row[key]) is int for key in ("raw_row_count", "kept_row_count", "excluded_row_count"))
            and row["raw_row_count"] == raw_count and row["excluded_row_count"] == count
            and row["kept_row_count"] == raw_count-count, "ROLE_BITMAP_COUNT_MISMATCH")
        for key, mask in (("keep_raw_rows_sha256", ~excluded), ("excluded_raw_rows_sha256", excluded)):
            digest = hashlib.sha256(np.asarray(np.flatnonzero(mask), dtype="<i8").tobytes()).hexdigest()
            _require(row[key] == digest, "ROLE_BITMAP_RAW_ROW_DIGEST_MISMATCH")
        masks.append(excluded)
    return masks
