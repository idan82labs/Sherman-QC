"""Bounded CAD-guided *offline* scan measurements, not qualified metrology.

Only positions enter fitting. A declared, already-aligned CAD inner-face feature
supplies finite search windows and normal orientation, never the fitted radius.
The supported model is one circular extrusion with two tangent planar flanges.
Registration, feature identity and acquisition side are caller declarations.
No native CAD, file, model, application, random sampling or outlier trimming.

Geometric circle objective: sum((norm(q_i-c)-R)**2); the algebraic solution is
only an initializer. Short-arc conditioning uses J, not J.T@J. See NIST Hopp
(1994), NISTIR 5501, and Witzgall/Cheok/Kearsley (2006), Recovering Circles and
Spheres from Point Data. Residuals/conditioning are NOT calibrated uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError


MAX_POINTS = 200_000
MAX_FEATURES = 16
MAX_FIT_EVALUATIONS = 80
MAX_FLAT_HULL_VERTICES = 2048


@dataclass(frozen=True)
class CADGuidedFeature:
    feature_id: str
    nominal_radius_mm: float
    nominal_sweep_deg: float
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    start_radial: tuple[float, float, float]
    span_mm: tuple[float, float]
    flange_window_mm: tuple[float, float]
    search_margin_mm: float
    max_fit_residual_mm: float
    plane_exclusion_mm: float
    max_tangency_error_mm: float
    nominal_source: str
    unit: str = "mm"
    surface: str = "inner"
    arc_guard_deg: float = 8.0
    min_arc_sweep_deg: float = 20.0
    max_normal_deviation_deg: float = 15.0
    max_axis_deviation_deg: float = 10.0
    max_circle_condition: float = 1000.0
    min_plane_spread_ratio: float = 0.01
    min_axial_span_fraction: float = 0.5
    fit_model: str = "free_circle"

    def __post_init__(self):
        if (not isinstance(self.feature_id, str) or not self.feature_id.strip()
                or not isinstance(self.nominal_source, str) or not self.nominal_source.strip()):
            raise ValueError("feature/source identity must be explicit nonempty strings")
        if self.unit != "mm" or self.surface != "inner":
            raise ValueError("only explicit millimetre single-inner-face input is supported")
        if self.fit_model not in ("free_circle", "measured_flange_tangent"):
            raise ValueError("unsupported circle fit model")
        for name, size in (("origin", 3), ("axis", 3), ("start_radial", 3),
                           ("span_mm", 2), ("flange_window_mm", 2)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        for name in ("nominal_radius_mm", "nominal_sweep_deg", "search_margin_mm",
                     "max_fit_residual_mm", "plane_exclusion_mm", "max_tangency_error_mm",
                     "arc_guard_deg", "min_arc_sweep_deg", "max_normal_deviation_deg",
                     "max_axis_deviation_deg", "max_circle_condition", "min_plane_spread_ratio",
                     "min_axial_span_fraction"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, float(value))
        a, n = np.asarray(self.axis), np.asarray(self.start_radial)
        if not np.allclose([a @ a, n @ n, a @ n], [1, 1, 0], atol=1e-12, rtol=0):
            raise ValueError("axis and start radial must be orthonormal")
        if (not 20 <= abs(self.nominal_sweep_deg) <= 160
                or self.nominal_radius_mm <= 0
                or not self.span_mm[0] < self.span_mm[1]
                or not 0 < self.flange_window_mm[0] < self.flange_window_mm[1]
                or not 0 < self.max_fit_residual_mm < self.plane_exclusion_mm < self.search_margin_mm
                or self.max_tangency_error_mm <= 0
                or not 0 < self.arc_guard_deg < abs(self.nominal_sweep_deg) / 2
                or not 0 < self.min_arc_sweep_deg < 180
                or not 0 < self.max_normal_deviation_deg < 45
                or not 0 < self.max_axis_deviation_deg < 45
                or not 1 < self.max_circle_condition <= 1e6
                or not 0 < self.min_plane_spread_ratio < 1
                or not 0 < self.min_axial_span_fraction <= 1):
            raise ValueError("invalid or unsupported feature/search policy")
        # Bound arithmetic before products/squares; this is an engineering scope,
        # not a universal CAD size or precision claim.
        lengths = (*self.origin, *self.span_mm, *self.flange_window_mm,
                   self.nominal_radius_mm, self.search_margin_mm)
        if max(abs(x) for x in lengths) > 1e6 or min(
                self.nominal_radius_mm, self.max_fit_residual_mm) < 1e-8:
            raise ValueError("unsupported numerical scale")


class _Unavailable(Exception):
    pass


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-12:
        raise _Unavailable("DEGENERATE_DIRECTION")
    return vector / length


def _plane(points, reference, feature):
    if len(points) < 12:
        raise _Unavailable("INSUFFICIENT_FLANGE_SUPPORT")
    center = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= 1e-12 or singular[1] / singular[0] < feature.min_plane_spread_ratio:
        raise _Unavailable("FLANGE_SUPPORT_RANK_DEFICIENT")
    if (singular[1] <= 10 * feature.max_fit_residual_mm * math.sqrt(len(points))
            or singular[2] / singular[1] >= 0.2):
        raise _Unavailable("FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY")
    normal = vh[-1]
    if normal @ reference < 0:
        normal = -normal
    if normal @ reference < math.cos(math.radians(feature.max_normal_deviation_deg)):
        raise _Unavailable("FLANGE_ORIENTATION_OUTSIDE_SEARCH_SCOPE")
    residual = np.abs((points - center) @ normal)
    if float(residual.max()) > feature.max_fit_residual_mm:
        raise _Unavailable("FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL")
    return normal, center, float(residual.max())


def circle_residual_jacobian(parameters, points):
    """Geometric radial residual/Jacobian in normalized cross-section units."""
    center, radius = parameters[:2], parameters[2]
    delta = points - center
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1e-14):
        raise _Unavailable("CIRCLE_CENTER_ON_SAMPLE")
    jacobian = np.column_stack((-delta / distance[:, None], -np.ones(len(points))))
    return distance - radius, jacobian


def _flat_strip_width(points):
    """Minimum enclosing 2-D strip width, bounded hull-edge enumeration.

    A convex polygon attains minimum width normal to one of its edges. This
    frees BOTH line direction and offset, unlike a max-residual TLS test.
    No Qhull joggling/point perturbation. Ordinary floating arithmetic is not
    an interval-arithmetic or calibrated error certificate.
    """
    try:
        hull = points[ConvexHull(points).vertices]
    except QhullError as error:
        raise _Unavailable("FLAT_HULL_UNAVAILABLE") from error
    if len(hull) > MAX_FLAT_HULL_VERTICES:
        raise _Unavailable("FLAT_WITNESS_BUDGET")
    width = math.inf
    for p, q in zip(hull, np.roll(hull, -1, axis=0)):
        edge = q - p
        # A near-duplicate projected observation can create a very short hull
        # edge. Its length is not a physical-direction identifiability test.
        length = math.hypot(float(edge[0]), float(edge[1]))
        if length == 0 or not math.isfinite(length):
            raise _Unavailable("FLAT_HULL_DEGENERATE_EDGE")
        normal = np.array([-edge[1], edge[0]]) / length
        projected = hull @ normal
        width = min(width, float(np.ptp(projected)))
    return width


def _circle(points, feature):
    if len(points) < 12:
        raise _Unavailable("INSUFFICIENT_ARC_SUPPORT")
    mean = points.mean(axis=0)
    centered = points - mean
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale <= 1e-10 or singular[-1] <= 1e-12:
        raise _Unavailable("ARC_SUPPORT_RANK_DEFICIENT")
    # A free-offset straight generator plane is a competing explanation at the
    # same declared residual limit. Include an arithmetic grey zone in abstention.
    guard = 128 * np.finfo(float).eps * max(1., float(np.max(np.abs(centered))))
    if _flat_strip_width(centered) / 2 <= feature.max_fit_residual_mm + guard:
        raise _Unavailable("ARC_COMPATIBLE_WITH_FLAT")
    normalized = centered / scale
    design = np.column_stack((2 * normalized, np.ones(len(points))))
    seed, _, rank, _ = np.linalg.lstsq(design, np.sum(normalized**2, axis=1), rcond=None)
    radius_sq = float(seed[2] + seed[:2] @ seed[:2])
    if rank != 3 or radius_sq <= 0 or not math.isfinite(radius_sq):
        raise _Unavailable("CIRCLE_INITIALIZER_UNAVAILABLE")
    initial = np.r_[seed[:2], math.sqrt(radius_sq)]
    result = least_squares(
        lambda p: circle_residual_jacobian(p, normalized)[0], initial,
        jac=lambda p: circle_residual_jacobian(p, normalized)[1],
        bounds=([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf]),
        max_nfev=MAX_FIT_EVALUATIONS, ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise _Unavailable("CIRCLE_SOLVER_INCOMPLETE")
    residual, jacobian = circle_residual_jacobian(result.x, normalized)
    s = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(s[0] / s[-1]) if s[-1] > 0 else math.inf
    if not math.isfinite(condition) or condition > feature.max_circle_condition:
        raise _Unavailable("ARC_ILL_CONDITIONED")
    max_residual = float(np.abs(residual).max() * scale)
    if max_residual > feature.max_fit_residual_mm:
        raise _Unavailable("ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL")
    return (mean + result.x[:2] * scale, float(result.x[2] * scale),
            max_residual, condition, float(s[-1] / math.sqrt(len(points))))


def tangent_residual_jacobian(radius, points, h, g):
    """Residual and Nx1 derivative for c(R)=h+R*g, using measured planes only."""
    radius = float(np.asarray(radius).reshape(-1)[0])
    delta = points - h - radius * g
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1e-14):
        raise _Unavailable("TANGENT_CENTER_ON_SAMPLE")
    derivative = -(delta / distance[:, None]) @ g - 1.
    return distance - radius, derivative[:, None]


def _tangent_circle(points, plane_matrix, plane_offsets, sweep_deg, seed_radius, feature):
    """New named hypothesis: scan-plane tangency constrains center, not CAD.

    A*c=b+R*1 for INNER normals, hence c=h+R*g. Each point must also lie
    inside the measured contact sector; its two linear inequalities restrict R
    before optimization, excluding the wrong-contact-sector second-circle branch.
    No measurement uncertainty follows from treating the fitted planes as fixed.
    """
    h = np.linalg.solve(plane_matrix, plane_offsets)
    g = np.linalg.solve(plane_matrix, np.ones(2))
    d = points - h
    phi = math.radians(sweep_deg)
    sign = math.copysign(1., phi)
    sector_normals = (np.array([0., sign]),
                      sign * np.array([math.sin(phi), -math.cos(phi)]))
    lower, upper = 1e-12 * seed_radius, math.inf
    for normal in sector_normals:
        alpha = d @ normal
        beta = float(g @ normal)
        if abs(beta) <= 1e-12:
            if np.any(alpha < 0):
                raise _Unavailable("TANGENT_CONTACT_SECTOR_UNAVAILABLE")
        elif beta > 0:
            upper = min(upper, float(np.min(alpha / beta)))
        else:
            lower = max(lower, float(np.max(alpha / beta)))
    if upper <= lower or lower <= 0 or not math.isfinite(lower):
        raise _Unavailable("TANGENT_CONTACT_SECTOR_UNAVAILABLE")
    scale = seed_radius  # Measured free-circle initializer, never nominal CAD R.
    lower /= scale
    upper /= scale
    initial = max(1., np.nextafter(lower, math.inf))
    if math.isfinite(upper):
        initial = min(initial, np.nextafter(upper, -math.inf))
    if not lower < initial < upper:
        raise _Unavailable("TANGENT_CONTACT_SECTOR_NUMERICALLY_UNRESOLVED")
    normalized, h_normalized = points / scale, h / scale
    result = least_squares(
        lambda r: tangent_residual_jacobian(r, normalized, h_normalized, g)[0], [initial],
        jac=lambda r: tangent_residual_jacobian(r, normalized, h_normalized, g)[1],
        bounds=([lower], [upper]), max_nfev=MAX_FIT_EVALUATIONS,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise _Unavailable("TANGENT_SOLVER_INCOMPLETE")
    residual, jacobian = tangent_residual_jacobian(result.x, normalized, h_normalized, g)
    rms_derivative = float(np.linalg.norm(jacobian) / math.sqrt(len(points)))
    if rms_derivative < 1 / feature.max_circle_condition:
        raise _Unavailable("TANGENT_RADIUS_WEAKLY_IDENTIFIED")
    max_residual = float(np.abs(residual).max() * scale)
    if max_residual > feature.max_fit_residual_mm:
        raise _Unavailable("ARC_NOT_TANGENT_CIRCLE_AT_DECLARED_RESIDUAL")
    radius = float(result.x[0] * scale)
    return h + radius * g, radius, max_residual, rms_derivative


def _measure(points, feature):
    origin, a0, n0 = map(np.asarray, (feature.origin, feature.axis, feature.start_radial))
    k0 = np.cross(a0, n0)
    phi = math.radians(feature.nominal_sweep_deg)
    sign = math.copysign(1.0, phi)
    n1 = math.cos(phi) * n0 + math.sin(phi) * k0
    radials = (n0, n1)
    rays = (-sign * k0, sign * np.cross(a0, n1))
    local = points - origin
    axial = local @ a0
    numeric_guard = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(points))),
                                                float(np.max(np.abs(origin))))
    in_span = ((axial >= feature.span_mm[0] - numeric_guard)
               & (axial <= feature.span_mm[1] + numeric_guard))
    masks = []
    for radial, ray in zip(radials, rays):
        from_contact = local - feature.nominal_radius_mm * radial
        along = from_contact @ ray
        masks.append(in_span & (along >= feature.flange_window_mm[0])
                     & (along <= feature.flange_window_mm[1])
                     & (np.abs(from_contact @ radial) <= feature.search_margin_mm))
    if np.any(masks[0] & masks[1]):
        raise _Unavailable("AMBIGUOUS_FLANGE_WINDOWS")
    fitted = [_plane(points[mask], -radial, feature) for mask, radial in zip(masks, radials)]
    normals = [item[0] for item in fitted]
    axis = _unit(np.cross(*normals))
    if axis @ a0 < 0:
        axis = -axis
    if axis @ a0 < math.cos(math.radians(feature.max_axis_deviation_deg)):
        raise _Unavailable("MEASURED_AXIS_OUTSIDE_SEARCH_SCOPE")
    sweep = math.degrees(math.atan2(float(axis @ np.cross(*normals)),
                                    float(normals[0] @ normals[1])))
    if math.copysign(1., sweep) != sign or not 20 <= abs(sweep) <= 160:
        raise _Unavailable("MEASURED_SWEEP_OUTSIDE_MODEL_SCOPE")

    angle = sign * np.arctan2(local @ k0, local @ n0)
    radial_distance = np.linalg.norm(local - axial[:, None] * a0, axis=1)
    guard = math.radians(feature.arc_guard_deg)
    arc_mask = (in_span & (angle >= guard) & (angle <= abs(phi) - guard)
                & (radial_distance <= feature.nominal_radius_mm + feature.search_margin_mm))
    for normal, center, _ in fitted:
        arc_mask &= np.abs((points - center) @ normal) > feature.plane_exclusion_mm
    arc = points[arc_mask]
    if len(arc) < 12:
        raise _Unavailable("INSUFFICIENT_ARC_SUPPORT")
    # Derive the projection direction from scan planes, not CAD. Center is free.
    u = _unit(-normals[0])
    v = np.cross(axis, u)
    q = np.column_stack(((arc - origin) @ u, (arc - origin) @ v))
    center2, radius, residual, condition, min_singular = _circle(q, feature)
    free_radius = radius
    free_center3 = origin + center2[0] * u + center2[1] * v
    free_tangency = [float(normal @ (center - free_center3) + radius)
                     for normal, center, _ in fitted]
    tangent_derivative = None
    if feature.fit_model == "measured_flange_tangent":
        matrix = np.array([[normal @ u, normal @ v] for normal in normals])
        offsets = np.array([normal @ (center - origin) for normal, center, _ in fitted])
        center2, radius, residual, tangent_derivative = _tangent_circle(
            q, matrix, offsets, sweep, radius, feature)
    center3 = origin + center2[0] * u + center2[1] * v
    # Both are INNER outward normals: d - n.C = -R. Absolute distance alone
    # would also admit a circle tangent on the wrong side of a flange.
    tangency = [float(normal @ (center - center3) + radius)
                for normal, center, _ in fitted]
    if max(abs(value) for value in tangency) > feature.max_tangency_error_mm:
        raise _Unavailable("CIRCLE_NOT_SAME_INNER_FACE_TANGENT")
    measured_angle = sign * np.arctan2((q - center2)[:, 1], (q - center2)[:, 0])
    if np.any(measured_angle < -1e-8) or np.any(measured_angle > math.radians(abs(sweep)) + 1e-8):
        raise _Unavailable("ARC_OUTSIDE_MEASURED_CONTACT_SECTOR")
    sweep_coverage = math.degrees(float(np.ptp(measured_angle)))
    if sweep_coverage < feature.min_arc_sweep_deg:
        raise _Unavailable("INSUFFICIENT_ARC_ANGULAR_COVERAGE")
    sampled_axial = (arc - origin) @ axis
    axial_stations = np.unique(sampled_axial)
    if len(axial_stations) < 2:
        raise _Unavailable("INSUFFICIENT_DISTINCT_AXIAL_STATIONS")
    axial_span = [float(sampled_axial.min()), float(sampled_axial.max())]
    if (axial_span[1] - axial_span[0]) < feature.min_axial_span_fraction * (
            feature.span_mm[1] - feature.span_mm[0]):
        raise _Unavailable("INSUFFICIENT_AXIAL_SAMPLE_SPAN")
    return {
        "fit_model": feature.fit_model,
        "free_circle_initializer_radius_mm": free_radius,
        "free_circle_tangency_residual_mm": free_tangency,
        "tangent_radius_rms_derivative": tangent_derivative,
        "measured_sweep_deg": sweep,
        "sweep_deviation_deg": sweep - feature.nominal_sweep_deg,
        "measured_opening_deg": 180. - abs(sweep),
        "measured_radius_mm": radius,
        "radius_deviation_mm": radius - feature.nominal_radius_mm,
        "measured_axis_direction": axis.tolist(),
        "measured_axis_point": center3.tolist(),
        "sampled_axial_interval_mm": axial_span,
        "max_axial_sample_gap_mm": float(np.diff(axial_stations).max()),
        "max_angular_sample_gap_deg": math.degrees(float(np.diff(np.unique(measured_angle)).max())),
        "continuous_surface_coverage": "NOT_ESTABLISHED",
        "physical_trim_endpoints": None,
        "physical_extent_deviation_mm": None,
        "flange_support_counts": [int(mask.sum()) for mask in masks],
        "arc_support_count": len(arc),
        "flange_max_residual_mm": [item[2] for item in fitted],
        "circle_max_residual_mm": residual,
        "inner_tangency_residual_mm": tangency,
        "tangency_check_role": ("ENFORCED_CONSTRAINT_NOT_INDEPENDENT_VALIDATION"
                                if feature.fit_model == "measured_flange_tangent"
                                else "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK"),
        "free_circle_jacobian_condition": condition,
        "free_circle_min_singular_per_sqrt_n": min_singular,
        "sampled_arc_span_deg": sweep_coverage,
        "support_row_indices": np.flatnonzero(arc_mask | masks[0] | masks[1]).tolist(),
        "_instance_support": {
            "arc_row_indices": np.flatnonzero(arc_mask).tolist(),
            "flange_row_indices": [np.flatnonzero(mask).tolist() for mask in masks],
        },
    }


def inspect_cad_guided_scan(points, features, *, registration_status="UNTRUSTED", instance_policy=None):
    """Measure declared CAD features from an already-aligned single-side scan.

    This explicit offline entry does not register, identify physical events or
    qualify tolerances. Every feature either retains the full conditional result
    or reports UNAVAILABLE, never a prefix measurement or copied nominal value.
    Overlapping support between features clears all competing measurements.
    Explicit finite-instance mode distinguishes exclusive curved evidence from
    reusable planar context; it never assigns physical trim or event identity.
    """
    if not isinstance(features, (tuple, list)) or not 1 <= len(features) <= MAX_FEATURES:
        raise ValueError("require 1..16 explicitly declared CAD features")
    if not all(isinstance(f, CADGuidedFeature) for f in features):
        raise ValueError("features must be CADGuidedFeature values")
    if len({f.feature_id for f in features}) != len(features):
        raise ValueError("duplicate feature identity")
    invalid = None
    if (not isinstance(points, np.ndarray) or points.ndim != 2
            or points.shape[1] != 3 or points.dtype.kind not in "fiu"):
        invalid = "INVALID_POINT_ARRAY"
    elif not 1 <= len(points) <= MAX_POINTS:
        invalid = "POINT_BUDGET_OR_EMPTY_INPUT"
    if invalid is None:
        points = np.array(points, dtype=np.float64, copy=True)
        if not np.isfinite(points).all() or float(np.max(np.abs(points))) > 1e6:
            invalid = "UNSUPPORTED_POINT_NUMERICS"
    if type(registration_status) is not str or registration_status != "DECLARED_ALIGNED":
        invalid = "REGISTRATION_NOT_DECLARED_ALIGNED"
    rows = []
    instance_builder = None
    if instance_policy is not None:
        if __package__:
            from .cad_guided_instances import build_finite_support_instance, SupportPolicy
        else:
            from cad_guided_instances import build_finite_support_instance, SupportPolicy
        if not isinstance(instance_policy, SupportPolicy):
            raise ValueError("instance_policy must be a validated SupportPolicy")
        instance_builder = build_finite_support_instance
    for feature in features:
        row = {"feature_id": feature.feature_id, "status": "UNAVAILABLE",
               "nominal_radius_mm": feature.nominal_radius_mm,
               "nominal_sweep_deg": feature.nominal_sweep_deg,
               "nominal_source": feature.nominal_source,
               "unit": "mm", "surface": "inner", "measurements": None,
               "uncertainty_status": "UNCALIBRATED", "conformance": "INDETERMINATE",
               "physical_completion_state": "UNKNOWN", "reasons": []}
        try:
            if invalid:
                raise _Unavailable(invalid)
            row["measurements"] = _measure(points, feature)
            row["status"] = "MEASURED_CONDITIONAL"
        except (_Unavailable, np.linalg.LinAlgError, FloatingPointError) as error:
            row["reasons"] = [str(error) or type(error).__name__]
        rows.append(row)
    claimed = {}
    competing = set()
    private_support = {}
    for index, row in enumerate(rows):
        if row["measurements"] is not None:
            support = row["measurements"].pop("_instance_support")
            private_support[index] = support
            arc_rows = set(support["arc_row_indices"])
            for raw in row["measurements"].pop("support_row_indices"):
                if raw in claimed:
                    for other, was_arc in claimed[raw]:
                        if instance_builder is None or was_arc or raw in arc_rows:
                            competing.update((index, other))
                claimed.setdefault(raw, []).append((index, raw in arc_rows))
    for index in competing:
        rows[index].update(status="UNAVAILABLE", measurements=None,
                           reasons=["AMBIGUOUS_SHARED_FEATURE_SUPPORT"])
    payload = {"schema_version": 1, "scope": "CAD_GUIDED_OFFLINE_SINGLE_INNER_FACE",
            "status": "COMPLETE" if invalid is None else "UNAVAILABLE",
            "registration_status": registration_status if isinstance(registration_status, str) else "INVALID",
            "feature_identity_status": "DECLARED_NOT_AUTOMATICALLY_VERIFIED",
            "features": rows, "expected_feature_count": len(features),
            "measured_conditional_feature_count": sum(r["status"] == "MEASURED_CONDITIONAL" for r in rows),
            "observed_bend_count": None, "conformance": "INDETERMINATE",
            "production_use_allowed": False}
    if instance_builder is not None:
        instances = []
        for index, row in enumerate(rows):
            if row["measurements"] is None:
                result = {"status": "UNAVAILABLE", "instance": None,
                          "reasons": ["MEASUREMENT_UNAVAILABLE", *row["reasons"]]}
            else:
                result = instance_builder(
                    points, row["measurements"], **private_support[index],
                    feature_id=row["feature_id"], policy=instance_policy)
            row["finite_support"] = result
            if result["status"] == "SUPPORTED":
                instances.append(result["instance"])
        payload.update(
            finite_support_instances=instances,
            observed_support_instance_count=len(instances),
            instance_extraction_status=("COMPLETE" if len(instances) == len(features)
                                        else "PARTIAL" if instances else "UNAVAILABLE"),
            line_convention="CYLINDER_CENTER_AXIS_SUPPORT",
            endpoint_semantics="SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
            reference_frame="CAD_REFERENCE", unit="mm",
        )
    return payload
