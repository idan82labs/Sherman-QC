"""Measure finite candidate windows using full original-row scan support.

The old narrow kernel is unchanged. A conservative crop admits every point its
flange/arc searches could use, without downsampling. Oversized LOCAL supports
abstain. Cross-window support ownership is checked globally in original raw-row
space, including across the old kernel's 16-feature batch boundary.

Native windows are not manufacturing events; a supported window is not a known
observed bend count. All outputs remain conditional on registration, single-side
acquisition identity, units and the kernel's declared surface model.
"""

import hashlib
import math

import numpy as np

if __package__:
    from .cad_guided_measurement import CADGuidedFeature, _measure, _Unavailable
    from .cad_guided_instances import SupportPolicy, build_finite_support_instance
else:
    from cad_guided_measurement import CADGuidedFeature, _measure, _Unavailable
    from cad_guided_instances import SupportPolicy, build_finite_support_instance


MAX_RAW_ROWS = 1_000_000
MAX_LOCAL_ROWS = 200_000
MAX_WINDOWS = 512
MAX_TOTAL_LOCAL_ROWS = 4_000_000


def _immutable_rows(rows):
    return np.frombuffer(np.asarray(rows, dtype="<i8").tobytes(), dtype="<i8")


def _crop_rows(points, feature):
    """Superset of the kernel's CAD-guided masks, with a numerical-only guard.

    Arc distance is bounded above but deliberately not below, matching the
    original kernel. A measured answer is never used to tighten the input crop.
    """
    origin, axis, n0 = map(np.asarray, (feature.origin, feature.axis, feature.start_radial))
    k0 = np.cross(axis, n0)
    phi = math.radians(feature.nominal_sweep_deg)
    sign = math.copysign(1., phi)
    n1 = math.cos(phi)*n0 + math.sin(phi)*k0
    radials = (n0, n1)
    rays = (-sign*k0, sign*np.cross(axis, n1))
    selected = []
    guard = 256*np.finfo(float).eps*max(1., float(np.max(np.abs(points))),
                                       float(np.max(np.abs(origin))))
    # Bound working arrays, even when the original scan has one million rows.
    for start in range(0, len(points), 65536):
        block = points[start:start+65536]
        local = block-origin
        z = local @ axis
        in_span = (z >= feature.span_mm[0]-guard) & (z <= feature.span_mm[1]+guard)
        radial_distance = np.hypot.reduce(local-z[:, None]*axis, axis=1)
        keep = radial_distance <= feature.nominal_radius_mm+feature.search_margin_mm+guard
        for radial, ray in zip(radials, rays):
            delta = local-feature.nominal_radius_mm*radial
            along = delta @ ray
            keep |= ((along >= feature.flange_window_mm[0]-guard)
                     & (along <= feature.flange_window_mm[1]+guard)
                     & (np.abs(delta @ radial) <= feature.search_margin_mm+guard))
        selected.append(np.flatnonzero(in_span & keep)+start)
    return np.concatenate(selected)


def _surface_carrier(measurement, local_points, support):
    """Measured inner-arc mid-angle generator, not a CAD axis/nominal radius.

    Fixed geometric convention, never selected against references. The existing
    physical Gold carrier is unresolved, so bridge compatibility is unverified.
    A fresh TLS plane from the same scan flange support determines radial start;
    its sign is oriented using the fitted circle center, not CAD nominal values.
    """
    center = np.asarray(measurement["measured_axis_point"])
    axis = np.asarray(measurement["measured_axis_direction"])
    flange = local_points[support["flange_row_indices"][0]]
    mean = np.mean(flange, axis=0)
    _, _, vh = np.linalg.svd(flange-mean, full_matrices=False)
    normal = vh[-1]
    # Inner contact radial points from fitted center toward the flange plane.
    if float(normal @ (mean-center)) < 0:
        normal = -normal
    radial = normal - (normal @ axis)*axis
    length = float(np.linalg.norm(radial))
    if length <= 1e-12:
        return None
    radial /= length
    phi = math.radians(measurement["measured_sweep_deg"])/2
    mid = math.cos(phi)*radial + math.sin(phi)*np.cross(axis, radial)
    return measurement["measured_radius_mm"]*mid


def inspect_full_raw_windows_v1(aligned_raw_points, features, *, instance_policy,
                                scan_sha256, source_raw_row_indices=None,
                                registration_status="UNTRUSTED"):
    """No thinning, nominal copying, hidden support reuse or count substitution.

    source_raw_row_indices may identify a datum-excluded ordered subsequence of
    the validated raw cloud. Those IDs, never local crop offsets, bind ownership.
    The digest is a binding declaration supplied by the byte-validating adapter,
    not authentication implemented by this array-only function.
    """
    result = dict(schema="CAD_GUIDED_FULL_RAW_WINDOW_MEASUREMENTS_V1", status="UNAVAILABLE",
                  features=[], finite_support_instances=[], observed_support_instance_count=0,
                  observed_bend_count=None, observed_bend_count_status="MANUFACTURING_EVENT_IDENTITY_UNKNOWN",
                  production_use_allowed=False, conformance="INDETERMINATE",
                  uncertainty_status="UNCALIBRATED", reasons=[],
                  measurement_cloud_contract="ALL_ORIGINAL_ROWS_IN_EACH_SEARCH_ROI_NO_THINNING",
                  line_convention="MEASURED_INNER_ARC_MID_ANGLE_GENERATOR",
                  reference_carrier_compatibility="UNVERIFIED_NO_GOLD_IN_CARRIER_SELECTION",
                  endpoint_semantics="SAMPLED_WINDOW_SUPPORT_NOT_PHYSICAL_TRIM",
                  registration_status=registration_status if type(registration_status) is str else "INVALID",
                  surface_side_identity="CALLER_ASSUMED_NOT_VERIFIED",
                  datum_relative_deviation_status="PROVISIONAL_NOT_STABLE_DATUM_METROLOGY")
    try:
        points = np.asarray(aligned_raw_points)
    except (ValueError, TypeError, OverflowError):
        result["reasons"] = ["INVALID_FULL_RAW_SCAN"]
        return result
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fiu"
            or not 1 <= len(points) <= MAX_RAW_ROWS or not np.isfinite(points).all()
            or np.max(np.abs(points.astype(np.float64))) > 1e6):
        result["reasons"] = ["INVALID_FULL_RAW_SCAN"]
        return result
    points = points.astype(np.float64, copy=False)
    if (type(scan_sha256) is not str or len(scan_sha256) != 64
            or any(c not in "0123456789abcdef" for c in scan_sha256)):
        result["reasons"] = ["BYTE_BOUND_SCAN_DIGEST_REQUIRED"]
        return result
    if type(registration_status) is not str or registration_status != "CONDITIONAL_BEST_FIT":
        result["reasons"] = ["EXPLICIT_PROVISIONAL_REGISTRATION_REQUIRED"]
        return result
    if (type(features) not in (tuple, list) or not 1 <= len(features) <= MAX_WINDOWS
            or not all(type(f) is CADGuidedFeature for f in features)
            or len({f.feature_id for f in features}) != len(features)
            or type(instance_policy) is not SupportPolicy):
        result["reasons"] = ["INVALID_OR_OVER_BUDGET_FEATURE_POLICY"]
        return result
    if source_raw_row_indices is None:
        raw_ids = np.arange(len(points), dtype=np.int64)
    else:
        try:
            original = np.asarray(source_raw_row_indices)
        except (ValueError, TypeError, OverflowError):
            result["reasons"] = ["INVALID_OR_REUSED_RAW_ROW_IDENTITIES"]
            return result
        if (original.ndim != 1 or len(original) != len(points) or original.dtype.kind not in "iu"
                or np.any(original < 0) or np.any(original >= MAX_RAW_ROWS)
                or len(np.unique(original)) != len(original)):
            result["reasons"] = ["INVALID_OR_REUSED_RAW_ROW_IDENTITIES"]
            return result
        raw_ids = original.astype(np.int64)
    result["scan_sha256"] = scan_sha256
    claimed, competing = {}, set()
    total_local = 0
    for index, feature in enumerate(features):
        selected = _crop_rows(points, feature)
        total_local += len(selected)
        row = dict(feature_id=feature.feature_id, status="UNAVAILABLE", measurements=None,
                   reasons=[], raw_support=None, finite_support=None, local_input_rows=len(selected),
                   conformance="INDETERMINATE", nominal_radius_mm=feature.nominal_radius_mm,
                   nominal_sweep_deg=feature.nominal_sweep_deg, nominal_source=feature.nominal_source)
        result["features"].append(row)
        if not 1 <= len(selected) <= MAX_LOCAL_ROWS or total_local > MAX_TOTAL_LOCAL_ROWS:
            row["reasons"] = ["LOCAL_OR_TOTAL_SUPPORT_BUDGET_NO_THINNING"]
            continue
        local = points[selected]
        try:
            measured = _measure(local, feature)
            support = measured.pop("_instance_support")
            measured.pop("support_row_indices")
            arc = raw_ids[selected[np.asarray(support["arc_row_indices"], dtype=np.int64)]]
            flanges = [raw_ids[selected[np.asarray(ids, dtype=np.int64)]]
                       for ids in support["flange_row_indices"]]
            row["raw_support"] = dict(arc_rows=_immutable_rows(arc),
                                       flange_rows=tuple(_immutable_rows(f) for f in flanges))
            for role, ids in [("arc", arc), *[("flange", f) for f in flanges]]:
                for raw in ids:
                    key = int(raw)
                    for other, old_role in claimed.get(key, ()):
                        if role == "arc" or old_role == "arc":
                            competing.update((index, other))
                    claimed.setdefault(key, []).append((index, role))
            finite = build_finite_support_instance(local, measured, **support,
                                                   feature_id=feature.feature_id, policy=instance_policy)
            offset = _surface_carrier(measured, local, support)
            if finite["status"] == "SUPPORTED" and offset is not None:
                instance = dict(finite["instance"])
                instance["axis_support_line_start"] = instance["line_start"]
                instance["axis_support_line_end"] = instance["line_end"]
                for endpoint in ("line_start", "line_end"):
                    instance[endpoint] = (np.asarray(instance[endpoint])+offset).tolist()
                instance.update(line_convention=result["line_convention"],
                                reference_carrier_compatibility=result["reference_carrier_compatibility"])
                finite = dict(finite, instance=instance)
            elif offset is None:
                finite = dict(status="UNAVAILABLE", instance=None, reasons=["MEASURED_CARRIER_UNAVAILABLE"])
            row.update(status="MEASURED_CONDITIONAL", measurements=measured, finite_support=finite)
        except (_Unavailable, np.linalg.LinAlgError, FloatingPointError) as error:
            row["reasons"] = [str(error) or type(error).__name__]
    for index in competing:
        result["features"][index].update(status="UNAVAILABLE", measurements=None,
            finite_support=dict(status="UNAVAILABLE", instance=None,
                                reasons=["AMBIGUOUS_SHARED_RAW_FEATURE_SUPPORT"]),
            reasons=["AMBIGUOUS_SHARED_RAW_FEATURE_SUPPORT"])
    instances = [row["finite_support"]["instance"] for row in result["features"]
                 if row["status"] == "MEASURED_CONDITIONAL" and row["finite_support"]
                 and row["finite_support"]["status"] == "SUPPORTED"]
    measured_count = sum(row["status"] == "MEASURED_CONDITIONAL" for row in result["features"])
    result.update(status="PARTIAL" if measured_count else "UNAVAILABLE",
                  computation_status="COMPLETE", expected_window_count=len(features),
                  measured_conditional_window_count=measured_count,
                  finite_support_instances=instances, observed_support_instance_count=len(instances),
                  total_local_input_rows=total_local, raw_input_rows=len(points),
                  raw_row_identity_sha256=hashlib.sha256(np.asarray(raw_ids, dtype="<i8").tobytes()).hexdigest())
    return result
