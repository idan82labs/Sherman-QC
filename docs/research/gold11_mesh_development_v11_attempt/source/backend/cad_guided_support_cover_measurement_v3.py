"""V3 of the support-cover measurement: V2 text bound to context measurement V2 / kernel V2
(layer selection, diagnostics, unresolved single-layer identity). V2 stays frozen.

Conditional measurements from one existing exact raw-support cover.

On nonfatal paths the frozen V1 context result remains intact as evidence;
fatal clearing retains only its completed digest/count receipt. Only this module's
new cover projection is publishable by its consumer; nested/member predictions
must never be concatenated. A cover is a sample-model witness, not proof of a
unique physical surface, manufacturing event, datum or joint pose.
"""

from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from . import cad_guided_context_measurement_v2 as _v1
from . import cad_guided_exact_support_cover_v1 as _cover


MAX_COVER_REFIT_ROW_VISITS = 4_000_000
MAX_REPORT_BYTES = 128*1024**2


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _same(left, right):
    """Exact JSON-tree equality without Python's bool/int equivalence."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (len(left) == len(right) and all(type(key) is str for key in left)
            and left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) in (str, int, float, bool, type(None)) and left == right


def _scalar(value, lower, upper, reason):
    _require(type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper, reason)
    return float(value)


def _vector(value, reason):
    _require(type(value) is list and len(value) == 3, reason)
    return np.array([_scalar(item, -1e6, 1e6, reason) for item in value])


def _digest(value):
    digest, size = hashlib.sha256(), 0
    for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).iterencode(value):
        encoded = fragment.encode()
        size += len(encoded)
        _require(size <= _v1.MAX_REPORT_BYTES, "SUPPORT_COVER_SOURCE_REPORT_BUDGET")
        digest.update(encoded)
    return digest.hexdigest()


def _owned_points(points):
    _require(type(points) is np.ndarray and points.ndim == 2 and points.shape[1] == 3
        and points.dtype.kind in "fiu" and 1 <= len(points) <= _v1.MAX_RAW_ROWS
        and np.isfinite(points).all() and np.max(np.abs(points)) <= 1e6,
        "INVALID_SUPPORT_COVER_POINTS")
    owned = np.frombuffer(np.asarray(points, dtype="<f8").tobytes(), dtype="<f8").reshape(-1, 3)
    _require(np.isfinite(owned).all() and np.max(np.abs(owned)) <= 1e6,
             "INVALID_OWNED_SUPPORT_COVER_POINTS")
    return owned


def _base():
    result = _v1._base()
    result.pop("support_records")
    result.pop("context_windows")
    result.update(schema="CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V3",
        context_quantifier="EXISTS_COMPLETED_EXISTING_COVER_CONTEXT",
        source_context_result=None, source_context_result_sha256=None,
        source_context_result_role="RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS",
        support_cover_plan=None, support_groups=[], cover_refit_row_visits=0,
        aggregation_policy="NARY_EXISTING_EXACT_RAW_SUPPORT_COVER",
        member_geometry_policy="COMMON_COVER_MODEL_WITNESS_NOT_INDEPENDENT_MEMBER_REFIT_CLAIM",
        finite_support_policy="COVER_ONLY_MEMBER_FINITE_REFUSAL_DOES_NOT_REMOVE_MEMBER",
        prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS")
    return result


def _bounded(result):
    size = 0
    for fragment in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(result):
        size += len(fragment.encode())
        _require(size <= MAX_REPORT_BYTES, "SUPPORT_COVER_REPORT_BUDGET")
    return result


def _associations(plan_group, records, measured):
    rows = []
    for identity in plan_group["member_record_ids"]:
        for index, original in enumerate(records[identity]["associations"]):
            _require(type(original["context_index"]) is int and 0 <= original["context_index"] < 48
                and all(type(original[key]) is str and original[key].strip()
                    and len(original[key]) <= 512 for key in ("feature_id", "nominal_source")),
                "INVALID_SUPPORT_COVER_NOMINAL_ASSOCIATION")
            _require(_scalar(original["nominal_radius_mm"], 0., 1e6,
                "INVALID_SUPPORT_COVER_NOMINAL_RADIUS") > 0., "ZERO_SUPPORT_COVER_NOMINAL_RADIUS")
            _require(20. <= abs(_scalar(original["nominal_sweep_deg"], -160., 160.,
                "INVALID_SUPPORT_COVER_NOMINAL_SWEEP")), "INVALID_SUPPORT_COVER_NOMINAL_SWEEP")
            row = dict(member_record_id=identity, member_association_index=index,
                context_index=original["context_index"], feature_id=original["feature_id"],
                nominal_source=original["nominal_source"], nominal_radius_mm=original["nominal_radius_mm"],
                nominal_sweep_deg=original["nominal_sweep_deg"],
                source_support_role="ORIGINAL_MEMBER_SUPPORT_NOT_FULL_COVER_UNDER_EVERY_CONTEXT",
                is_full_cover_witness=identity == plan_group["cover_record_id"],
                accepted_registration=False)
            if measured is not None:
                eligible = measured["inner_face_claim_eligible"]
                row.update(canonical_radius_deviation_mm=(measured["measured_radius_mm"]-original["nominal_radius_mm"]
                                                          if eligible else None),
                    canonical_sweep_magnitude_deviation_deg=measured["measured_sweep_magnitude_deg"]-abs(original["nominal_sweep_deg"]),
                    inner_face_claim_eligible=eligible)
            rows.append(row)
    return rows


def inspect_support_cover_context_windows_v3(raw_points, features, *, registration, scan_sha256,
                                             instance_policy, surface_roles=None, layer_policy=None):
    """Recompute only a cover's scan geometry; never revive serialized answers.

    Every member remains in its maximal ownership component, irrespective of
    old fitting/finite status. Only the full cover needs a fresh geometry fit;
    subset inclusion gives its common sample-model witness to the members.
    Cover finite-support gates alone govern carrier eligibility. No member
    finite refusal is removed or relabeled as independently successful.
    """
    result = _base()
    try:
        points = _owned_points(raw_points)
        _require(type(features) in (list, tuple) and 1 <= len(features) <= 64
            and all(type(feature) is _v1.CADGuidedFeature for feature in features),
            "INVALID_SUPPORT_COVER_FEATURES")
        features = tuple(_v1.CADGuidedFeature(**asdict(feature)) for feature in features)
        _require(type(instance_policy) is _v1.SupportPolicy, "INVALID_SUPPORT_COVER_INSTANCE_POLICY")
        instance_policy = _v1.SupportPolicy(**asdict(instance_policy))
        point_sha = hashlib.sha256(points.tobytes()).hexdigest()
        original = _v1.inspect_generated_context_windows_v2(points, features,
            registration=registration, scan_sha256=scan_sha256,
            instance_policy=instance_policy, surface_roles=surface_roles, layer_policy=layer_policy)
        _v1.validate_context_measurement_publication_v2(original, raw_count=len(points),
            scan_sha256=scan_sha256, raw_points_sha256=point_sha)
        result.update(source_context_result=original, source_context_result_sha256=_digest(original),
            scan_sha256=scan_sha256, raw_points_sha256=point_sha, instance_policy=asdict(instance_policy))
        if original.get("computation_status") != "COMPLETE" or original.get("discarded_measurement_prefix") is True:
            result.update(computation_status="SOURCE_CONTEXT_REFUSED", reasons=["COMPLETE_CONTEXT_EVIDENCE_UNAVAILABLE"])
            return _bounded(result)
        plan = _cover.plan_exact_support_covers_v1(points, original["support_records"])
        result["support_cover_plan"] = plan
        if plan["status"] != "COMPLETE":
            result.update(computation_status="SUPPORT_COVER_PLAN_REFUSED", reasons=["COMPLETE_SUPPORT_COVER_PLAN_UNAVAILABLE"])
            return _bounded(result)
        records = {row["record_id"]: row for row in original["support_records"]}
        feature_by_id = {feature.feature_id: feature for feature in features}
        for proposal in plan["groups"]:
            group = dict(group_id=proposal["group_id"], status="UNAVAILABLE", reasons=list(proposal["reasons"]),
                member_record_ids=list(proposal["member_record_ids"]), cover_record_id=proposal["cover_record_id"],
                member_flange_permutations=dict(proposal["member_flange_permutations"]),
                raw_support=None, fit_policy=None, layer_policy=None, material_side_status=None,
                inner_face_claim_eligible=None, carrier_role=None, measurements=None, finite_support=None,
                associations=_associations(proposal, records, None))
            result["support_groups"].append(group)
            if proposal["status"] != "COVERED":
                continue
            witness = records[proposal["cover_record_id"]]
            raw = witness["raw_support"]
            group["raw_support"] = dict(arc_rows=list(raw["arc_rows"]),
                flange_rows=[list(rows) for rows in raw["flange_rows"]])
            group["fit_policy"] = dict(witness["fit_policy"])
            group["layer_policy"] = dict(witness["layer_policy"])
            side = _v1._identity_fields(witness["material_side_status"])
            _v1._bind_identity(witness, side)
            group.update(side)
            feature = feature_by_id[witness["associations"][0]["feature_id"]]
            _require(_v1._fit_policy(feature) == witness["fit_policy"]
                and witness["layer_policy"] == original["layer_policy"], "SUPPORT_COVER_FIT_POLICY_BINDING")
            result["cover_refit_row_visits"] += len(raw["arc_rows"])+sum(map(len, raw["flange_rows"]))
            _require(result["cover_refit_row_visits"] <= MAX_COVER_REFIT_ROW_VISITS,
                     "SUPPORT_COVER_REFIT_ROW_BUDGET")
            try:
                measured, finite = _v1._canonical_fit(points, raw["arc_rows"], raw["flange_rows"],
                    feature, instance_policy, group["group_id"],
                    material_side_status=witness["material_side_status"])
                _v1._bind_identity(measured, side)
                if finite["status"] == "SUPPORTED":
                    _v1._bind_identity(finite["instance"], side)
                group.update(status="MEASURED_CONDITIONAL", reasons=[], measurements=measured,
                    finite_support=finite, associations=_associations(proposal, records, measured))
            except (_v1._Unavailable, ValueError) as error:
                # A set-level cover with unsupported/duplicate/singular sample
                # geometry remains one whole refused component, never subsets.
                group["reasons"] = ["COVER_GEOMETRY_UNAVAILABLE:"+type(error).__name__+":"+str(error)[:160]]
        result["predictions"] = [row["finite_support"]["instance"] for row in result["support_groups"]
            if row["status"] == "MEASURED_CONDITIONAL" and row["finite_support"]["status"] == "SUPPORTED"]
        result.update(status="PARTIAL" if any(row["status"] == "MEASURED_CONDITIONAL" for row in result["support_groups"])
            else "UNAVAILABLE", computation_status="COMPLETE",
            conditional_support_group_count=len(result["support_groups"]),
            reasons=[] if any(row["status"] == "MEASURED_CONDITIONAL" for row in result["support_groups"])
            else ["NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS"])
        return _bounded(result)
    except Exception as error:
        refused = _base()
        refused.update(reasons=["SUPPORT_COVER_MEASUREMENT_FAILURE:"+type(error).__name__+":"+str(error)[:180]],
            discarded_measurement_prefix=True,
            discarded_source_context_sha256=result["source_context_result_sha256"],
            discarded_source_record_count=(len(result["source_context_result"].get("support_records", []))
                if result["source_context_result"] is not None else 0),
            attempted_work=dict(cover_refit_row_visits=result["cover_refit_row_visits"],
                attempted_groups=len(result["support_groups"]),
                cover_plan_work=(result["support_cover_plan"] or {}).get("work")))
        return refused


def validate_support_cover_publication_v3(result, *, raw_points, scan_sha256):
    """Revalidate source/cover set proof and exact output projection, not refit."""
    try:
        return _validate_publication(result, raw_points=raw_points, scan_sha256=scan_sha256)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("MALFORMED_SUPPORT_COVER_PUBLICATION:"+type(error).__name__) from error


def _validate_publication(result, *, raw_points, scan_sha256):
    _require(type(result) is dict, "INVALID_SUPPORT_COVER_PUBLICATION")
    _bounded(result)
    for key, expected in _base().items():
        if key in ("schema", "context_quantifier", "source_context_result_role", "aggregation_policy",
                   "member_geometry_policy", "finite_support_policy", "prediction_collection_status",
                   "joint_configuration_status", "continuous_pose_family_coverage", "registration_status",
                   "production_use_allowed", "observed_bend_count", "observed_bend_count_status",
                   "uncertainty_status", "endpoint_semantics", "conformance", "geometry_frame"):
            _require(type(result[key]) is type(expected) and result[key] == expected,
                     "INVALID_SUPPORT_COVER_PUBLICATION_SEMANTICS")
    _require(result["status"] in ("PARTIAL", "UNAVAILABLE") and type(result["support_groups"]) is list
        and type(result["predictions"]) is list and len(result["predictions"]) <= _cover.MAX_RECORDS
        and len(result["support_groups"]) <= _cover.MAX_RECORDS,
        "INVALID_SUPPORT_COVER_OUTPUT_DENOMINATOR")
    if result.get("discarded_measurement_prefix") is True:
        _require(result["status"] == "UNAVAILABLE" and result["support_groups"] == result["predictions"] == []
            and result["source_context_result"] is None and result["support_cover_plan"] is None,
            "SUPPORT_COVER_PREFIX_AFTER_FATAL_FAILURE")
        digest = result["discarded_source_context_sha256"]
        _require(digest is None or type(digest) is str and len(digest) == 64
            and all(value in "0123456789abcdef" for value in digest), "INVALID_DISCARDED_SUPPORT_COVER_SOURCE_DIGEST")
        _require(type(result["discarded_source_record_count"]) is int
            and 0 <= result["discarded_source_record_count"] <= _cover.MAX_RECORDS,
            "INVALID_DISCARDED_SUPPORT_COVER_SOURCE_COUNT")
        return dict(cover_plan_revalidated=False, fatal_refusal=True)
    points = _owned_points(raw_points)
    point_sha = hashlib.sha256(points.tobytes()).hexdigest()
    _require(result["raw_points_sha256"] == point_sha and result["scan_sha256"] == scan_sha256,
             "SUPPORT_COVER_PUBLICATION_SOURCE_MISMATCH")
    original = result["source_context_result"]
    _v1.validate_context_measurement_publication_v2(original, raw_count=len(points),
        scan_sha256=scan_sha256, raw_points_sha256=point_sha)
    _require(result["source_context_result_sha256"] == _digest(original), "SUPPORT_COVER_NESTED_EVIDENCE_DIGEST")
    if original.get("computation_status") != "COMPLETE" or original.get("discarded_measurement_prefix") is True:
        _require(result["computation_status"] == "SOURCE_CONTEXT_REFUSED" and result["status"] == "UNAVAILABLE"
            and result["support_groups"] == result["predictions"] == [] and result["support_cover_plan"] is None,
            "SUPPORT_COVER_RECOVERY_FROM_INCOMPLETE_SOURCE")
        return dict(cover_plan_revalidated=False, source_refusal=True)
    _require(_same(result["instance_policy"], original["instance_policy"]),
             "SUPPORT_COVER_INSTANCE_POLICY_CHANGED")
    _require(type(result["instance_policy"]) is dict and len(result["instance_policy"]) == 4,
             "INVALID_SUPPORT_COVER_INSTANCE_POLICY")
    _v1.SupportPolicy(**result["instance_policy"])
    plan = _cover.plan_exact_support_covers_v1(points, original["support_records"])
    _require(_same(result["support_cover_plan"], plan), "SUPPORT_COVER_PLAN_NOT_EXACT_RECOMPUTATION")
    if plan["status"] != "COMPLETE":
        _require(result["computation_status"] == "SUPPORT_COVER_PLAN_REFUSED" and result["status"] == "UNAVAILABLE"
            and result["support_groups"] == result["predictions"] == [], "SUPPORT_COVER_OUTPUT_AFTER_PLAN_REFUSAL")
        return dict(cover_plan_revalidated=True, cover_plan_work=plan["work"], plan_refusal=True)
    _require(result["computation_status"] == "COMPLETE"
        and len(result["support_groups"]) == len(plan["groups"])
        and type(result["conditional_support_group_count"]) is int
        and result["conditional_support_group_count"] == len(plan["groups"]), "SUPPORT_COVER_GROUP_DENOMINATOR")
    records = {row["record_id"]: row for row in original["support_records"]}
    expected_predictions, refit_rows, publication_row_visits = [], 0, 0
    for group, proposal in zip(result["support_groups"], plan["groups"]):
        _require(all(_same(group[key], proposal[key]) for key in
            ("group_id", "member_record_ids", "cover_record_id", "member_flange_permutations")),
            "SUPPORT_COVER_GROUP_PROOF_BINDING")
        measured = group["measurements"]
        _require(_same(group["associations"], _associations(proposal, records, measured)),
                 "SUPPORT_COVER_MEMBER_ASSOCIATION_PROJECTION")
        if proposal["status"] != "COVERED":
            _require(group["status"] == "UNAVAILABLE" and measured is None and group["finite_support"] is None
                and group["raw_support"] is None and group["fit_policy"] is None and group["layer_policy"] is None
                and group["material_side_status"] is None and group["inner_face_claim_eligible"] is None
                and group["carrier_role"] is None
                and group["reasons"] == proposal["reasons"], "UNCOVERED_GROUP_HAS_SCIENTIFIC_OUTPUT")
            continue
        witness = records[proposal["cover_record_id"]]
        _require(_same(group["raw_support"], witness["raw_support"])
            and _same(group["fit_policy"], witness["fit_policy"])
            and _same(group["layer_policy"], witness["layer_policy"])
            and _same(group["layer_policy"], original["layer_policy"]),
                 "SUPPORT_COVER_WITNESS_ROWS_OR_POLICY_CHANGED")
        side = _v1._identity_fields(witness["material_side_status"])
        _v1._bind_identity(witness, side)
        _v1._bind_identity(group, side)
        refit_rows += len(witness["raw_support"]["arc_rows"])+sum(map(len, witness["raw_support"]["flange_rows"]))
        _require(refit_rows <= MAX_COVER_REFIT_ROW_VISITS, "SUPPORT_COVER_PUBLICATION_REFIT_BUDGET")
        if group["status"] == "UNAVAILABLE":
            _require(measured is None and group["finite_support"] is None and group["reasons"],
                     "REFUSED_COVER_HAS_SCIENTIFIC_OUTPUT")
            continue
        _require(group["status"] == "MEASURED_CONDITIONAL" and group["reasons"] == [],
                 "INVALID_SUPPORT_COVER_MEASUREMENT_STATUS")
        _require(measured["fit_model"] == "free_circle" and measured["status"] == "CONDITIONAL_SCAN_POINT_ESTIMATE"
            and measured["geometry_frame"] == result["geometry_frame"]
            and measured["radius_uncertainty_interval_mm"] is None and measured["sweep_uncertainty_interval_deg"] is None
            and measured["physical_trim_endpoints"] is None and measured["datum_relative_deviation"] is None
            and measured["sweep_sign_convention"] == "CANONICAL_RAW_AXIS_AND_RAW_ROLE_ORDER"
            and measured["material_side_status"] == side["material_side_status"],
            "INVALID_SUPPORT_COVER_QUANTITY_CLAIMS")
        _v1._bind_identity(measured, side)
        radius, sweep, signed = (measured[name] for name in
            ("measured_radius_mm", "measured_sweep_magnitude_deg", "measured_sweep_deg"))
        _require(type(radius) in (float, int) and math.isfinite(radius) and 1e-8 <= radius <= 1e6
            and type(sweep) in (float, int) and math.isfinite(sweep) and 20 <= sweep <= 160
            and type(signed) in (float, int) and abs(signed) == sweep, "INVALID_SUPPORT_COVER_QUANTITIES")
        policy = group["fit_policy"]
        _scalar(measured["circle_max_residual_mm"], 0., policy["max_fit_residual_mm"],
                "SUPPORT_COVER_CIRCLE_RESIDUAL_CONTRADICTS_POLICY")
        _require(type(measured["flange_max_residual_mm"]) is list and len(measured["flange_max_residual_mm"]) == 2
            and type(measured["measured_tangency_residual_mm"]) is list
            and len(measured["measured_tangency_residual_mm"]) == 2, "INVALID_SUPPORT_COVER_RESIDUAL_VECTORS")
        for value in measured["flange_max_residual_mm"]:
            _scalar(value, 0., policy["max_fit_residual_mm"], "SUPPORT_COVER_FLANGE_RESIDUAL_CONTRADICTS_POLICY")
        for value in measured["measured_tangency_residual_mm"]:
            _scalar(value, -policy["max_tangency_error_mm"], policy["max_tangency_error_mm"],
                    "SUPPORT_COVER_TANGENCY_CONTRADICTS_POLICY")
        _scalar(measured["free_circle_jacobian_condition"], 1., policy["max_circle_condition"],
                "SUPPORT_COVER_CIRCLE_CONDITION_CONTRADICTS_POLICY")
        _require(_scalar(measured["free_circle_min_singular_per_sqrt_n"], 0., 2.,
            "INVALID_SUPPORT_COVER_CIRCLE_SINGULAR_VALUE") > 0., "ZERO_SUPPORT_COVER_CIRCLE_SINGULAR_VALUE")
        _require(measured["tangency_check_role"] == "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK",
                 "INVALID_SUPPORT_COVER_TANGENCY_CLAIM")
        axis = _vector(measured["measured_axis_direction"], "INVALID_SUPPORT_COVER_AXIS")
        center = _vector(measured["measured_axis_point"], "INVALID_SUPPORT_COVER_AXIS_POINT")
        _require(abs(float(np.linalg.norm(axis))-1.) <= 1e-10 and axis[int(np.argmax(np.abs(axis)))] >= 0,
                 "SUPPORT_COVER_AXIS_NOT_CANONICAL_UNIT_VECTOR")
        finite = group["finite_support"]
        _require(type(finite) is dict and finite["status"] in ("SUPPORTED", "UNAVAILABLE"), "INVALID_SUPPORT_COVER_FINITE_STATUS")
        _require(type(finite["reasons"]) is list and len(finite["reasons"]) <= 16
            and all(type(reason) is str and 1 <= len(reason) <= 512 for reason in finite["reasons"]),
            "INVALID_SUPPORT_COVER_FINITE_REASONS")
        if finite["status"] == "UNAVAILABLE":
            _require(finite["instance"] is None and finite["reasons"], "SUPPORT_COVER_INSTANCE_AFTER_FINITE_REFUSAL")
            continue
        _require(finite["reasons"] == [], "SUPPORTED_COVER_HAS_FINITE_REFUSAL")
        instance = finite["instance"]
        _require(instance["feature_id"] == group["group_id"] and instance["bend_id"] == group["group_id"]
            and instance["line_convention"] == "MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR"
            and instance["geometry_frame"] == result["geometry_frame"] and instance["physical_trim_endpoints"] is None
            and instance["manufacturing_event_identity"] == "UNKNOWN"
            and instance["continuous_surface_coverage"] == "NOT_ESTABLISHED"
            and instance["observation_status"] == "SAMPLE_SUPPORTED_CONDITIONAL"
            and instance["unit"] == "mm"
            and instance["material_side_status"] == side["material_side_status"]
            and instance["inner_face_claim_eligible"] is side["inner_face_claim_eligible"]
            and instance["carrier_role"] == side["carrier_role"]
            and instance["reference_carrier_compatibility"] == "UNVERIFIED_NO_GOLD_IN_CARRIER_SELECTION"
            and instance["endpoint_semantics"] == "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM", "INVALID_SUPPORT_COVER_CARRIER_CLAIMS")
        start, end, axis_start, axis_end = [_vector(instance[key], "INVALID_SUPPORT_COVER_CARRIER_ENDPOINT")
            for key in ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end")]
        _require(not np.array_equal(start, end) and not np.array_equal(axis_start, axis_end),
                 "DEGENERATE_SUPPORT_COVER_CARRIER_OR_AXIS")
        # Serialization consistency only, not a measurement error allowance.
        guard = 512*np.finfo(float).eps*max(1., radius,
            *(float(np.max(np.abs(value))) for value in (center, start, end, axis_start, axis_end)))
        length = _scalar(instance["axial_support_length_mm"], 0., 4e6, "INVALID_SUPPORT_COVER_LENGTH")
        _require(length > 0. and np.linalg.norm(axis_end-axis_start-length*axis) <= guard
            and np.linalg.norm(end-start-length*axis) <= 2*guard,
            "SUPPORT_COVER_CARRIER_LENGTH_OR_DIRECTION_CONTRADICTION")
        for endpoint in (axis_start, axis_end):
            delta = endpoint-center
            _require(np.linalg.norm(delta-float(delta@axis)*axis) <= 2*guard,
                     "SUPPORT_COVER_ENDPOINT_OFF_REPORTED_AXIS")
        offset = start-axis_start
        _require(np.linalg.norm((end-axis_end)-offset) <= 2*guard and abs(float(offset@axis)) <= guard
            and abs(float(np.linalg.norm(offset))-radius) <= guard,
            "SUPPORT_COVER_CARRIER_RADIUS_OR_OFFSET_CONTRADICTION")
        roles = [witness["raw_support"]["arc_rows"], *witness["raw_support"]["flange_rows"]]
        counts = list(map(len, roles))
        _require(_same(instance["unique_arc_support_count"], counts[0])
            and _same(instance["unique_flange_support_counts"], counts[1:]),
            "SUPPORT_COVER_FINITE_TOTAL_COUNT_MISMATCH")
        publication_row_visits += sum(counts)
        _require(publication_row_visits <= MAX_COVER_REFIT_ROW_VISITS,
                 "SUPPORT_COVER_PUBLICATION_ROW_BUDGET")
        magnitude = max(float(np.max(np.abs(points[rows]))) for rows in roles)
        # Exactly the constructor's arithmetic guard, not the carrier check's
        # serialization reserve or a new physical/noise tolerance.
        finite_guard = 128*np.finfo(float).eps*max(1., magnitude, float(np.max(np.abs(center))), radius)
        angular_guard = math.degrees(finite_guard/max(radius, 1e-8))
        support_policy = result["instance_policy"]
        _require(length > finite_guard
            and finite_guard < 1e-4*min(radius, support_policy["max_axial_gap_mm"]),
            "SUPPORT_COVER_FINITE_NUMERIC_RESOLUTION_CONTRADICTION")
        maximum_gap = _scalar(instance["max_axial_sample_gap_mm"], 0.,
            min(length, support_policy["max_axial_gap_mm"]+finite_guard), "INVALID_SUPPORT_COVER_AXIAL_GAP")
        _require(maximum_gap > 0., "ZERO_SUPPORT_COVER_AXIAL_GAP")
        slabs = max(1, math.ceil(length/support_policy["max_axial_gap_mm"]))
        coverage = instance["local_coverage"]
        _require(slabs <= 2048 and type(coverage) is list and len(coverage) == slabs,
                 "SUPPORT_COVER_FINITE_SLAB_DENOMINATOR")
        for index, slab in enumerate(coverage):
            _require(type(slab) is dict and type(slab["slab_index"]) is int and slab["slab_index"] == index
                and type(slab["flange_points"]) is list and len(slab["flange_points"]) == 2,
                "INVALID_SUPPORT_COVER_SLAB_SCHEMA")
            _require(all(type(value) is int and support_policy["min_local_points"] <= value <= total
                for value, total in zip([slab["arc_points"], *slab["flange_points"]], counts)),
                "INVALID_SUPPORT_COVER_SLAB_COUNTS")
            sweep_at_slab = _scalar(slab["arc_sweep_deg"], 0., 180.+angular_guard,
                                   "INVALID_SUPPORT_COVER_SLAB_SWEEP")
            gap_at_slab = _scalar(slab["max_angular_gap_deg"], 0., sweep_at_slab,
                                 "INVALID_SUPPORT_COVER_SLAB_GAP")
            _require(gap_at_slab > 0. and sweep_at_slab+angular_guard >= support_policy["min_arc_sweep_deg"]
                and gap_at_slab <= support_policy["max_angular_gap_deg"]+angular_guard,
                "SUPPORT_COVER_SLAB_CONTRADICTS_POLICY")
        expected_predictions.append(instance)
    _require(type(result["cover_refit_row_visits"]) is int and result["cover_refit_row_visits"] == refit_rows,
             "SUPPORT_COVER_REFIT_LEDGER_MISMATCH")
    _require(_same(result["predictions"], expected_predictions), "SUPPORT_COVER_NOT_EXACT_PREDICTION_PROJECTION")
    _require((result["status"] == "PARTIAL") == any(row["status"] == "MEASURED_CONDITIONAL" for row in result["support_groups"]),
             "SUPPORT_COVER_STATUS_NOT_MEASUREMENT_DISPOSITION")
    return dict(cover_plan_revalidated=True, cover_plan_work=plan["work"], cover_refit_repeated=False,
        publication_geometry_row_visits=publication_row_visits)
