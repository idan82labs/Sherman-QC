"""Explicit R0 engineering control; not the historical Gold11 incumbent.

Policies are copied unchanged from the September 9 invented-file rehearsal.
They are not scanner-calibrated, corpus-tuned or manufacturing-approved.
No files, environment variables, model pointers or labels are read here.
"""


CONTROL_ID = "CAD_DATUM_REGISTERED_R0_20260909"


def control_specification():
    """Return fresh plain data, including the inherited feature defaults."""
    return {
        "control_id": CONTROL_ID,
        "scope": "SOURCE_CONFIG_FROZEN_DEVELOPMENT_CONTROL",
        "historical_gold11_comparable": False,
        "physical_score": None,
        "physical_access_authorized": False,
        "production_use_allowed": False,
        "entrypoint": "backend.cad_guided_offline.inspect_registered_native",
        "input_entrypoint": "backend.cad_guided_case_input.load_registered_case",
        "unit_status": "DECLARED_MM",
        "reference_frame": "CAD_REFERENCE",
        "line_convention": "CYLINDER_CENTER_AXIS_SUPPORT",
        "endpoint_semantics": "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
        "datum_selection": "EXPLICIT_STABLE_REGION_SELECTION_NOT_AUTOMATIC",
        "preprocessing": "NONE_RAW_ROW_ORDER_PRESERVED",
        "registration": {
            "max_iterations": 40, "max_correspondence_distance_mm": .30,
            "max_p95_residual_mm": .10, "max_residual_mm": .50,
            "min_scan_coverage": .95, "min_cad_coverage": .30,
            "min_observability_ratio": .01, "pose_class_distance_mm": .05,
            "ambiguity_residual_margin_mm": .01, "ambiguity_coverage_margin": .02,
            "normal_consistency_ratio": .05,
        },
        "measurement": {
            "flange_window_mm": [2.5, 12.5], "search_margin_mm": 3.,
            "max_fit_residual_mm": .005, "plane_exclusion_mm": .075,
            "max_tangency_error_mm": .1, "fit_model": "measured_flange_tangent",
        },
        "inherited_feature_defaults": {
            "arc_guard_deg": 8., "min_arc_sweep_deg": 20.,
            "max_normal_deviation_deg": 15., "max_axis_deviation_deg": 10.,
            "max_circle_condition": 1000., "min_plane_spread_ratio": .01,
            "min_axial_span_fraction": .5,
        },
        "support": {
            "max_axial_gap_mm": 4., "max_angular_gap_deg": 2.,
            "min_arc_sweep_deg": 20., "min_local_points": 12,
        },
        "scoring": {
            "angle_deg": 5., "transverse_offset_mm": 4.,
            "gold_span_overlap": .60, "prediction_span_overlap": .60,
        },
        "input_limits": {
            "max_scan_rows": 200000, "max_datum_rows": 5000,
            "max_ply_bytes": 67108864, "max_cad_bytes": 8388608,
            "max_datum_bytes": 2097152,
        },
        "native_loader": {
            "binding": "ocp", "allow_ocp": True, "require_occ_absent": True,
            "linear_deflection": .25, "angular_deflection": .5,
            "retain_native_shape": True, "meshing_parallel": True,
            "native_thread_count": "UNVERIFIED",
            "kernel_unit_resources": "DEFAULTS_NOT_PHYSICALLY_VERIFIED",
        },
        "failure_handling": "RETAIN_ALL_EXPECTED_CASES_ABSTENTION_IS_NOT_DEFECT",
        "promotion": "NOT_AUTHORIZED_BY_THIS_CONTROL",
    }


def construct_policies():
    """Construct the actual dataclasses and reject implicit-default drift."""
    from dataclasses import fields
    from .cad_guided_registration import RegistrationPolicy
    from .cad_guided_native_features import NativeGuidedPolicy
    from .cad_guided_measurement import CADGuidedFeature
    from .cad_guided_instances import SupportPolicy
    from .cad_guided_evaluation import Thresholds

    spec = control_specification()
    defaults = {field.name: field.default for field in fields(CADGuidedFeature)}
    if any(defaults[name] != value for name, value in spec["inherited_feature_defaults"].items()):
        raise ValueError("CONTROL_INHERITED_DEFAULT_DRIFT")
    return {
        "registration_policy": RegistrationPolicy(**spec["registration"]),
        "measurement_policy": NativeGuidedPolicy(**spec["measurement"]),
        "instance_policy": SupportPolicy(**spec["support"]),
        "thresholds": Thresholds(**spec["scoring"]),
    }
