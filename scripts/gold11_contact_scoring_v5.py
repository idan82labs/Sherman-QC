"""V5 contact scoring: the V4 text recognizing support cover V5 / context V4 (measurement kernel V3 two-mode
layer separation: the thirteen-field layer policy record and the two new unresolved material-side statuses,
neither inner-face eligible). Matcher, thresholds, denominator and failure arithmetic unchanged. V4 stays frozen.

Original V4 header follows.
V4 contact scoring: the V3 text recognizing support cover V4 / context V3 (initializer-V16 explicit
finite-subset registration contract, validated here too). Matcher, thresholds, denominator and failure
arithmetic unchanged. V3 stays frozen.

Original V3 header: V3 of the Gold11 contact projection: V2 text bound to the V3 support-cover / V2 context /
kernel-V2 evidence (separately bound layer policy, per-carrier declared material-side identity,
carrier roles, eligibility counts). The matcher, thresholds, denominators and the legacy scorer
imports are the V2 text and stay unchanged; the score is explicitly a generic conditional
contact-generator localization, never an inner-arc or manufacturing-event metric.

Bounded, source-only V13 contact-carrier projection and paired diagnostics.

No file, native, raw-scan, geometry-fit, metadata-codec or authorization access.
The caller supplies the exact separately retained RESULT.json bytes and its
expected SHA256. V4's worker writes a bare workflow JSON object, not a wrapper.
Matching and metric aggregation reuse the unchanged legacy pure functions.

This is a new development adapter, not a repair of scoring V1. Contact-sector
generators are not relabeled as inner arcs. Structural checks and a supplied-byte
hash do not attest raw support, material side, registration, historical custody,
physical trim, manufacturing identity or a calibrated measurement interval.

The input byte cap precedes JSON decoding, but node/depth checks follow the
standard JSON parser's allocation. They are not preallocation, RAM or CPU
limits. The caller must isolate decoding/projection with externally enforced
process-tree memory and CPU/wall budgets; this module is not a streaming parser.
"""

import hashlib
import json
import math

from . import gold11_mesh_scoring_v1 as _legacy
from .evaluate_f1_against_manual_gold import Thresholds, match_lines
from .f1_physical_evaluation_contract_v2 import score_localized_cases


PROJECTION_SCHEMA = "GOLD11_CONTACT_WORKFLOW_PROJECTION_V5"
SCORING_SCHEMA = "GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V5"
WORKFLOW_SCHEMA = "CAD_GUIDED_MESH_OFFLINE_V13"
MEASUREMENT_SCHEMA = "CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V5"
SOURCE_CONTEXT_SCHEMA = "CAD_GUIDED_GENERATED_CONTEXT_MEASUREMENTS_V4"
SUBSET_PUBLICATION_CONTRACT = "EVALUATED_ALLOWANCE_SUBSET_EXPLICIT_DISJOINT_DISPOSITION_LEDGER_V1"
SUBSET_LEDGER_SCHEMA = "candidate_disposition_ledger_v1"
_LEDGER_KEYS = ("completed", "failed", "expired", "omitted", "not_attempted")
KERNEL_IMPLEMENTATION = "CAD_GUIDED_MEASUREMENT_V3_TWO_MODE_LAYER_SEPARATION_AND_DIAGNOSTICS"
SCORING_SCOPE = "GENERIC_CONDITIONAL_CONTACT_GENERATOR_LOCALIZATION_NOT_INNER_ARC_OR_MANUFACTURING_EVENT"
MATERIAL_SIDE_STATUSES = ("CONTEXT_ASSUMED_NOT_VERIFIED", "SINGLE_LAYER_IDENTITY_UNRESOLVED",
    "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED", "MIXED_LAYER_EVIDENCE_IDENTITY_UNRESOLVED",
    "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED", "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED",
    "TWO_MODES_MEMBERSHIP_UNRESOLVED", "PROJECTION_PREMISE_UNRESOLVED")
# Mirrors kernel V2: presently EMPTY. No material-side status is verified side evidence, so no
# carrier is inner-face eligible; the two-layer selection is retained as a side hypothesis only.
INNER_FACE_ELIGIBLE_STATUSES = ()
INNER_FACE_CARRIER_ROLE = "INNER_FACE_CONDITIONAL_CONTACT_LINE_UNVERIFIED"
GENERIC_CARRIER_ROLE = "GENERIC_CONDITIONAL_CONTACT_LINE_SURFACE_IDENTITY_UNRESOLVED_NOT_INNER_ARC"
LAYER_POLICY_FIELDS = ("enabled", "layer_split_gap_mm", "min_layer_rows", "max_layers_reported",
    "max_layer_normal_angle_deg", "max_layer_gap_mm", "min_projected_interval_overlap", "max_layer_rms_mm",
    "layer_mode_bin_mm", "layer_membership_band_mm", "max_bridging_row_fraction", "layer_refit_iterations",
    "layer_mode_ambiguity_fraction")
RAW_FRAME = "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
UNITS = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
CARRIER = "MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR"
COLLECTION = "CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS"
COUNT_SEMANTICS = "EMITTED_SUPPORT_LINES_NOT_PHYSICAL_OBSERVED_BENDS"
AUTHORITY = "STRUCTURAL_PROJECTION_NOT_RAW_SUPPORT_OR_EXTERNAL_CUSTODY_ATTESTATION"
MAX_RESULT_BYTES = 128 * 1024**2
MAX_PROJECTION_BYTES = 8 * 1024**2
MIN_PROJECTION_BYTES = 8192
MAX_SCORE_BYTES = 32 * 1024**2
MAX_GROUPS = 512
MAX_PREDICTIONS = 512
MAX_JSON_NODES = 8_000_000
MAX_JSON_DEPTH = 64

# Versioned descriptor grammar, not imports of engine constructors. The runner
# separately freezes the actual engine sources and requested budget values.
SEEDED_BUDGET_MAXIMA = dict(max_seeds=32, max_seed_selection_triangle_visits=64_000_000,
    max_proposal_raw_row_visits=32_000_000, max_proposal_triangle_visits=64_000_000,
    max_proposal_edge_visits=96_000_000, max_full_support_row_tests=32_000_000,
    max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
    max_fit_evaluations=32, max_retained_array_bytes=268_435_456,
    max_metadata_bytes=67_108_864, max_working_array_bytes=1_073_741_824)
COMPOSITE_BUDGET_MAXIMA = dict(max_qualified_planes=65_552,
    max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
    max_full_support_row_tests=64_000_000, max_raw_predicate_visits=48_000_000,
    max_triangle_predicate_visits=160_000_000, max_edge_predicate_visits=192_000_000,
    max_fit_evaluations=65_584, max_retained_array_bytes=536_870_912,
    max_metadata_bytes=67_108_864)

_TOP_CLAIMS = dict(observed_bend_count=None, production_use_allowed=False,
    conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED",
    prediction_frame=RAW_FRAME, units_status=UNITS,
    reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
_MEASUREMENT_CLAIMS = dict(schema=MEASUREMENT_SCHEMA, observed_bend_count=None,
    observed_bend_count_status="MANUFACTURING_EVENT_IDENTITY_UNKNOWN",
    production_use_allowed=False, conformance="INDETERMINATE",
    uncertainty_status="UNCALIBRATED_NO_PROPAGATED_MEASUREMENT_INTERVAL",
    geometry_frame=RAW_FRAME, endpoint_semantics="SAMPLED_WINDOW_SUPPORT_NOT_PHYSICAL_TRIM",
    context_quantifier="EXISTS_COMPLETED_EXISTING_COVER_CONTEXT",
    joint_configuration_status="NOT_ESTABLISHED", continuous_pose_family_coverage="NOT_ESTABLISHED",
    registration_status="NOT_ACCEPTED_CONTEXT_PROPOSALS_ONLY",
    source_context_result_role="RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS",
    aggregation_policy="NARY_EXISTING_EXACT_RAW_SUPPORT_COVER",
    member_geometry_policy="COMMON_COVER_MODEL_WITNESS_NOT_INDEPENDENT_MEMBER_REFIT_CLAIM",
    finite_support_policy="COVER_ONLY_MEMBER_FINITE_REFUSAL_DOES_NOT_REMOVE_MEMBER",
    prediction_collection_status=COLLECTION)
_QUANTITY_CLAIMS = dict(fit_model="free_circle", status="CONDITIONAL_SCAN_POINT_ESTIMATE",
    geometry_frame=RAW_FRAME, radius_uncertainty_interval_mm=None,
    sweep_uncertainty_interval_deg=None, physical_trim_endpoints=None,
    datum_relative_deviation=None,
    sweep_sign_convention="CANONICAL_RAW_AXIS_AND_RAW_ROLE_ORDER")
_IDENTITY_KEYS = ("material_side_status", "inner_face_claim_eligible", "carrier_role")
_QUANTITY_KEYS = {*_QUANTITY_CLAIMS, *_IDENTITY_KEYS, "measured_radius_mm", "measured_sweep_magnitude_deg",
    "measured_sweep_deg", "measured_axis_direction", "measured_axis_point", "circle_max_residual_mm",
    "flange_max_residual_mm", "measured_tangency_residual_mm", "free_circle_jacobian_condition",
    "free_circle_min_singular_per_sqrt_n", "tangency_check_role"}
_FIT_POLICY_KEYS = {"max_fit_residual_mm", "max_tangency_error_mm", "min_arc_sweep_deg",
    "max_circle_condition", "min_plane_spread_ratio", "max_normal_deviation_deg", "fit_model"}
_LINE_CLAIMS = dict(line_convention=CARRIER, unit="mm", geometry_frame=RAW_FRAME,
    endpoint_semantics="SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
    observation_status="SAMPLE_SUPPORTED_CONDITIONAL", continuous_surface_coverage="NOT_ESTABLISHED",
    physical_trim_endpoints=None, manufacturing_event_identity="UNKNOWN",
    reference_carrier_compatibility="UNVERIFIED_NO_GOLD_IN_CARRIER_SELECTION")
_COMPACT_LINE_KEYS = {"bend_id", "feature_id", "line_start", "line_end", *_LINE_CLAIMS, *_IDENTITY_KEYS}
_FULL_LINE_KEYS = _COMPACT_LINE_KEYS | {"axis_support_line_start", "axis_support_line_end",
    "axial_support_length_mm", "unique_arc_support_count", "unique_flange_support_counts",
    "max_axial_sample_gap_mm", "local_coverage"}
_COUNT_KEYS = ("support_group_count", "covered_group_count", "refused_group_count",
    "measured_group_count", "finite_supported_group_count", "finite_refused_group_count",
    "emitted_carrier_count", "inner_face_eligible_carrier_count", "identity_unresolved_carrier_count", "subset_registration_count",
    "source_support_record_count", "source_context_window_count",
    "registration_context_count")
_DISPOSITIONS = {"SCORED_CONDITIONAL", "COMPLETE_NO_LOCALIZED_CARRIERS", "REFUSED", "MALFORMED_RESULT"}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _same(left, right):
    """Type-strict JSON equality; bool/int or int/float aliases are not custody."""
    pending = [iter(((left, right),))]
    visits = 0
    while pending:
        try:
            a, b = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        visits += 1
        _require(visits <= MAX_JSON_NODES, "STRUCTURAL_EQUALITY_NODE_BUDGET")
        if type(a) is not type(b):
            return False
        if type(a) is dict:
            if a.keys() != b.keys():
                return False
            pending.append(iter(zip(a.values(), map(b.__getitem__, a))))
        elif type(a) is list:
            if len(a) != len(b):
                return False
            pending.append(iter(zip(a, b)))
        elif type(a) not in (str, int, float, bool, type(None)) or a != b:
            return False
    return True


def _json_tree(value, *, maximum_nodes=MAX_JSON_NODES):
    # Iterators avoid a second large list of every child. This also bounds
    # cycles/deep native dictionaries supplied directly to the paired scorer.
    pending = [(iter((value,)), 0)]
    visits = 0
    while pending:
        iterator, depth = pending[-1]
        try:
            item = next(iterator)
        except StopIteration:
            pending.pop()
            continue
        visits += 1
        _require(visits <= maximum_nodes and depth <= MAX_JSON_DEPTH, "JSON_STRUCTURE_BUDGET")
        if type(item) is dict:
            _require(all(type(key) is str for key in item), "JSON_STRING_KEYS_REQUIRED")
            pending.append((iter(item.values()), depth + 1))
        elif type(item) is list:
            pending.append((iter(item), depth + 1))
        else:
            _require(type(item) in (str, int, float, bool, type(None)), "JSON_VALUE_TYPE")
            if type(item) is float:
                _require(math.isfinite(item), "NONFINITE_JSON_NUMBER")
            if type(item) is int:
                _require(abs(item) <= 2**63 - 1, "JSON_INTEGER_RANGE")
            if type(item) is str:
                _require(len(item) <= MAX_RESULT_BYTES, "JSON_STRING_BUDGET")


def _encoded_size(value, cap):
    size = 0
    for piece in json.JSONEncoder(sort_keys=True, allow_nan=False,
            ensure_ascii=True, separators=(",", ":")).iterencode(value):
        size += len(piece.encode("ascii"))
        _require(size <= cap, "COMPACT_PROJECTION_BYTE_BUDGET")
    return size


def _decode(blob):
    # Byte admission has already happened. json.loads allocates before the
    # iterative node/depth check below; external parser resource limits remain
    # mandatory, including when malformed input causes an early parser error.
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("NONFINITE_JSON_NUMBER")

    def integer(value):
        _require(len(value.lstrip("-")) <= 19, "JSON_INTEGER_RANGE")
        result = int(value)
        _require(abs(result) <= 2**63 - 1, "JSON_INTEGER_RANGE")
        return result

    value = json.loads(blob.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=constant, parse_int=integer)
    _json_tree(value)
    return value


def _claims(value, expected, *, required=True):
    _require(type(value) is dict, "CLAIM_OBJECT_REQUIRED")
    for key, answer in expected.items():
        if required or key in value:
            _require(key in value and _same(value[key], answer), "CONDITIONAL_CLAIM_CONFLICT:" + key)


def _identity(row, reason):
    """Declared, unverified material-side identity; eligibility is a function of the status."""
    status = row.get("material_side_status")
    _require(type(status) is str and status in MATERIAL_SIDE_STATUSES, reason+":STATUS")
    eligible = status in INNER_FACE_ELIGIBLE_STATUSES
    _require(row.get("inner_face_claim_eligible") is eligible, reason+":ELIGIBILITY")
    _require(row.get("carrier_role") == (INNER_FACE_CARRIER_ROLE if eligible else GENERIC_CARRIER_ROLE), reason+":ROLE")
    return status, eligible


# Inert mirror of kernel V3 `LayerSelectionPolicyV2.__post_init__` (ranges are (low, high]; cross-field
# constraints below). The scorer imports no engine module, so the canonical domain is restated here and
# bound by test against the kernel's own validator; a serialized policy the kernel constructor would
# reject is malformed evidence, whether or not every serialized copy agrees.
LAYER_POLICY_SCALAR_DOMAIN = (("layer_split_gap_mm", 0., 100.), ("max_layer_normal_angle_deg", 0., 45.),
    ("max_layer_gap_mm", 0., 1000.), ("min_projected_interval_overlap", 0., 1.), ("max_layer_rms_mm", 0., 100.),
    ("layer_mode_bin_mm", 0., 100.), ("layer_membership_band_mm", 0., 100.), ("max_bridging_row_fraction", 0., .5),
    ("layer_mode_ambiguity_fraction", 0., 1.))
LAYER_POLICY_INTEGER_DOMAIN = (("min_layer_rows", 12, 100_000), ("max_layers_reported", 2, 64), ("layer_refit_iterations", 1, 8))
LAYER_POLICY_CROSS_FIELD_RULES = ("max_layer_gap_mm > layer_split_gap_mm", "layer_mode_bin_mm <= layer_split_gap_mm / 3",
                                  "layer_membership_band_mm <= layer_split_gap_mm / 2")


def _layer_policy(value, reason):
    _require(type(value) is dict and set(value) == set(LAYER_POLICY_FIELDS) and type(value["enabled"]) is bool
        and all(type(value[key]) is int and low <= value[key] <= high for key, low, high in LAYER_POLICY_INTEGER_DOMAIN)
        and all(type(value[key]) in (int, float) and math.isfinite(value[key]) and low < float(value[key]) <= high
                for key, low, high in LAYER_POLICY_SCALAR_DOMAIN), reason)
    _require(float(value["max_layer_gap_mm"]) > float(value["layer_split_gap_mm"])
        and float(value["layer_mode_bin_mm"]) <= float(value["layer_split_gap_mm"]) / 3.
        and float(value["layer_membership_band_mm"]) <= float(value["layer_split_gap_mm"]) / 2., reason + ":CROSS_FIELD")
    return value


def _reasons(value):
    _require(type(value) is list and len(value) <= 64 and all(type(row) is str
        and 1 <= len(row) <= 512 for row in value), "INVALID_REASONS")
    return list(value)


def _request(mode, configuration):
    _require(type(mode) is str and mode in ("v7", "composite_v1"), "INVALID_REQUESTED_SCAN_PLANE_MODE")
    if mode == "v7":
        _require(configuration is None, "V7_REQUEST_REQUIRES_ABSENT_COMPOSITE_CONFIGURATION")
        return dict(mode=mode, configuration=None)
    _require(type(configuration) is dict and set(configuration) ==
        {"mode", "seeded_budgets", "composite_budgets"}
        and type(configuration["mode"]) is str and configuration["mode"] == "composite_v1",
        "INVALID_REQUESTED_COMPOSITE_CONFIGURATION")
    copied = dict(mode="composite_v1")
    for key, maximum in (("seeded_budgets", SEEDED_BUDGET_MAXIMA),
                         ("composite_budgets", COMPOSITE_BUDGET_MAXIMA)):
        row = configuration[key]
        _require(type(row) is dict and set(row) == set(maximum)
            and all(type(v) is int and 0 <= v <= maximum[k] for k, v in row.items()),
            "INVALID_REQUESTED_COMPOSITE_BUDGETS")
        copied[key] = dict(row)
    return dict(mode=mode, configuration=copied)


def _returned_configurations(payload, request, *, completed):
    policies, registration = payload.get("policies"), payload.get("registration")
    for row in (policies, registration):
        _require(row is None or type(row) is dict, "INVALID_CONFIGURATION_PARENT")
    evidence = registration.get("evidence") if type(registration) is dict else None
    _require(evidence is None or type(evidence) is dict, "INVALID_CONFIGURATION_PARENT")
    initialization = evidence.get("initialization") if type(evidence) is dict else None
    _require(initialization is None or type(initialization) is dict, "INVALID_CONFIGURATION_PARENT")
    result = {}
    for name, parent in (("policies", policies), ("registration", evidence), ("initialization", initialization)):
        present = type(parent) is dict and "scan_plane_configuration" in parent
        if request["mode"] == "v7":
            _require(not present, "UNREQUESTED_RETURNED_COMPOSITE_CONFIGURATION")
        elif present:
            actual = _request("composite_v1", parent["scan_plane_configuration"])
            _require(_same(actual, request), "REQUESTED_RETURNED_CONFIGURATION_MISMATCH")
        elif completed:
            raise ValueError("MISSING_COMPLETED_COMPOSITE_CONFIGURATION:" + name)
        result[name] = dict(present=present,
            configuration=parent["scan_plane_configuration"] if present else None)
    return result


def _number(value, lower, upper, reason):
    _require(type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper, reason)
    return float(value)


def _quantity(row, policy):
    """Check V3's reported numeric serialization, never refit or attest it."""
    _claims(row, _QUANTITY_CLAIMS)
    _require(set(row) == _QUANTITY_KEYS, "INVALID_CONTACT_QUANTITY_FIELDS")
    _identity(row, "INVALID_CONTACT_QUANTITY_IDENTITY")
    _require(type(policy) is dict and set(policy) == _FIT_POLICY_KEYS
        and type(policy["fit_model"]) is str and policy["fit_model"] == "free_circle",
        "INVALID_CONTACT_FIT_POLICY_FIELDS")
    # These are serialized kernel policy ranges, not added measurement gates.
    _number(policy["max_fit_residual_mm"], 1e-8, 1e6, "INVALID_CONTACT_FIT_RESIDUAL_POLICY")
    _require(_number(policy["max_tangency_error_mm"], 0., 1e12, "INVALID_CONTACT_TANGENCY_POLICY") > 0.,
             "INVALID_CONTACT_TANGENCY_POLICY")
    _require(0. < _number(policy["min_arc_sweep_deg"], 0., 180., "INVALID_CONTACT_SWEEP_POLICY") < 180.,
             "INVALID_CONTACT_SWEEP_POLICY")
    _require(_number(policy["max_circle_condition"], 1., 1e6, "INVALID_CONTACT_CONDITION_POLICY") > 1.,
             "INVALID_CONTACT_CONDITION_POLICY")
    _require(0. < _number(policy["min_plane_spread_ratio"], 0., 1., "INVALID_CONTACT_SPREAD_POLICY") < 1.,
             "INVALID_CONTACT_SPREAD_POLICY")
    _require(0. < _number(policy["max_normal_deviation_deg"], 0., 45., "INVALID_CONTACT_NORMAL_POLICY") < 45.,
             "INVALID_CONTACT_NORMAL_POLICY")
    radius = _number(row["measured_radius_mm"], 1e-8, 1e6, "INVALID_CONTACT_RADIUS")
    sweep = _number(row["measured_sweep_magnitude_deg"], 20., 160., "INVALID_CONTACT_SWEEP_MAGNITUDE")
    signed = _number(row["measured_sweep_deg"], -160., 160., "INVALID_CONTACT_SIGNED_SWEEP")
    _require(abs(signed) == sweep, "CONTACT_SWEEP_MAGNITUDE_CONTRADICTION")
    _number(row["circle_max_residual_mm"], 0., policy["max_fit_residual_mm"], "CONTACT_CIRCLE_RESIDUAL_POLICY_CONTRADICTION")
    for key, lower, upper in (
            ("flange_max_residual_mm", 0., policy["max_fit_residual_mm"]),
            ("measured_tangency_residual_mm", -policy["max_tangency_error_mm"], policy["max_tangency_error_mm"])):
        values = row[key]
        _require(type(values) is list and len(values) == 2, "INVALID_CONTACT_RESIDUAL_VECTOR")
        for value in values:
            _number(value, lower, upper, "CONTACT_RESIDUAL_POLICY_CONTRADICTION")
    _number(row["free_circle_jacobian_condition"], 1., policy["max_circle_condition"], "CONTACT_CONDITION_POLICY_CONTRADICTION")
    _require(_number(row["free_circle_min_singular_per_sqrt_n"], 0., 2., "INVALID_CONTACT_MINIMUM_SINGULAR_VALUE") > 0.,
             "ZERO_CONTACT_MINIMUM_SINGULAR_VALUE")
    _require(row["tangency_check_role"] == "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK", "INVALID_CONTACT_TANGENCY_CLAIM")
    axis, center = _legacy._point(row["measured_axis_direction"]), _legacy._point(row["measured_axis_point"])
    _require(abs(math.hypot(*axis) - 1.) <= 1e-10
        and axis[max(range(3), key=lambda index: abs(axis[index]))] >= 0., "CONTACT_AXIS_NOT_CANONICAL_UNIT_VECTOR")
    return radius, axis, center


def _carrier_serialization(row, quantity):
    """V2's constant-radius generator checks; no physical tolerance is added."""
    _require(quantity is not None, "CONTACT_QUANTITY_REQUIRED_FOR_CARRIER")
    radius, axis, center = quantity
    start, end, axis_start, axis_end = [_legacy._point(row[key]) for key in
        ("line_start", "line_end", "axis_support_line_start", "axis_support_line_end")]
    _require(start != end and axis_start != axis_end, "DEGENERATE_CONTACT_CARRIER_OR_AXIS")
    # Match the existing V2 serialization guard and axis tolerance. These are
    # floating-point consistency checks, not raw residual/error propagation or
    # directed-rounded numerical certificates. No endpoint is altered.
    guard = 512 * (2.**-52) * max(1., radius,
        *(abs(value) for vector in (center, start, end, axis_start, axis_end) for value in vector))
    length = _number(row["axial_support_length_mm"], 0., 4e6, "INVALID_CONTACT_CARRIER_LENGTH")
    _require(length > 0. and math.hypot(*(b - a - length * n for a, b, n in zip(axis_start, axis_end, axis))) <= guard
        and math.hypot(*(b - a - length * n for a, b, n in zip(start, end, axis))) <= 2 * guard,
        "CONTACT_CARRIER_LENGTH_OR_DIRECTION_CONTRADICTION")
    for endpoint in (axis_start, axis_end):
        delta = [value - origin for value, origin in zip(endpoint, center)]
        axial = sum(value * normal for value, normal in zip(delta, axis))
        _require(math.hypot(*(value - axial * normal for value, normal in zip(delta, axis))) <= 2 * guard,
                 "CONTACT_AXIS_ENDPOINT_OFF_REPORTED_AXIS")
    offset = [value - origin for value, origin in zip(start, axis_start)]
    _require(math.hypot(*(b - a - d for a, b, d in zip(axis_end, end, offset))) <= 2 * guard
        and abs(sum(value * normal for value, normal in zip(offset, axis))) <= guard
        and abs(math.hypot(*offset) - radius) <= guard, "CONTACT_CARRIER_RADIUS_OR_OFFSET_CONTRADICTION")


def _line(row, *, full, quantity=None):
    _require(type(row) is dict, "INVALID_CONTACT_LINE_OBJECT")
    _require(not any(key in row for key in ("span_endpoints", "anchor", "axis_direction", "span_mm")),
             "ALTERNATE_LINE_REPRESENTATION_NOT_ADMITTED")
    _require(set(row) == (_FULL_LINE_KEYS if full else _COMPACT_LINE_KEYS), "INVALID_CONTACT_LINE_SCHEMA")
    _claims(row, _LINE_CLAIMS)
    _identity(row, "INVALID_CONTACT_LINE_IDENTITY")
    _require(_hex(row["bend_id"]) and row["feature_id"] == row["bend_id"], "INVALID_CONTACT_GROUP_ID")
    start, end = _legacy._point(row["line_start"]), _legacy._point(row["line_end"])
    _require(math.hypot(*(b - a for a, b in zip(start, end))) > 1e-9, "DEGENERATE_CONTACT_LINE")
    if full:
        _carrier_serialization(row, quantity)
        counts = [row["unique_arc_support_count"], *row["unique_flange_support_counts"]] \
            if type(row["unique_flange_support_counts"]) is list else []
        _require(len(counts) == 3 and all(type(v) is int and 12 <= v <= 200_000 for v in counts),
                 "INVALID_CONTACT_SUPPORT_COUNTS")
        for key in ("axial_support_length_mm", "max_axial_sample_gap_mm"):
            _require(type(row[key]) in (int, float) and 0 < row[key] <= 4e6, "INVALID_CONTACT_SUPPORT_LENGTH")
        _require(row["max_axial_sample_gap_mm"] <= row["axial_support_length_mm"], "INVALID_CONTACT_SUPPORT_GAP")
        coverage = row["local_coverage"]
        _require(type(coverage) is list and 1 <= len(coverage) <= 2048, "INVALID_CONTACT_COVERAGE_CONTAINER")
        for index, slab in enumerate(coverage):
            _require(type(slab) is dict and set(slab) == {"slab_index", "arc_points", "flange_points",
                "arc_sweep_deg", "max_angular_gap_deg"} and type(slab["slab_index"]) is int
                and slab["slab_index"] == index, "INVALID_CONTACT_COVERAGE_ROW")
            local = [slab["arc_points"], *slab["flange_points"]] if type(slab["flange_points"]) is list else []
            _require(len(local) == 3 and all(type(v) is int and 3 <= v <= total
                for v, total in zip(local, counts)), "INVALID_CONTACT_COVERAGE_COUNTS")
            _require(type(slab["arc_sweep_deg"]) in (int, float)
                and type(slab["max_angular_gap_deg"]) in (int, float)
                and 0 < slab["max_angular_gap_deg"] <= slab["arc_sweep_deg"] <= 360.,
                "INVALID_CONTACT_COVERAGE_ANGLES")
    # Preserve original numeric coordinates. Validation's float conversions are
    # not emitted, and the diagnostic center-axis alternatives are never used.
    return {key: list(row[key]) if key in ("line_start", "line_end") else row[key]
            for key in sorted(_COMPACT_LINE_KEYS)}


def _complete_measurements(payload):
    measured = payload["measurements"]
    _claims(measured, _MEASUREMENT_CLAIMS)
    _require(measured.get("computation_status") == "COMPLETE"
        and measured.get("discarded_measurement_prefix", False) is False
        and measured.get("status") == payload["status"]
        and payload.get("prediction_collection_status") == COLLECTION, "INVALID_COMPLETE_MEASUREMENT_STATUS")
    receipt = payload.get("support_cover_publication_validation")
    _require(type(receipt) is dict and receipt.get("cover_plan_revalidated") is True
        and receipt.get("cover_refit_repeated") is False
        and not any(receipt.get(key, False) is not False for key in
            ("fatal_refusal", "source_refusal", "plan_refusal")), "MISSING_REPORTED_V13_PUBLICATION_VALIDATION")
    groups = measured.get("support_groups")
    _require(type(groups) is list and len(groups) <= MAX_GROUPS
        and type(measured.get("conditional_support_group_count")) is int
        and measured["conditional_support_group_count"] == len(groups), "INVALID_SUPPORT_GROUP_DENOMINATOR")
    original, plan = measured.get("source_context_result"), measured.get("support_cover_plan")
    _require(type(original) is dict and original.get("schema") == SOURCE_CONTEXT_SCHEMA
        and original.get("kernel_implementation") == KERNEL_IMPLEMENTATION
        and original.get("computation_status") == "COMPLETE"
        and original.get("discarded_measurement_prefix", False) is False
        and type(original.get("support_records")) is list and len(original["support_records"]) <= MAX_GROUPS
        and type(original.get("context_windows")) is list and len(original["context_windows"]) <= 4096,
        "INVALID_RETAINED_SOURCE_CONTEXT_STRUCTURE")
    _require(type(plan) is dict and plan.get("status") == "COMPLETE"
        and type(plan.get("groups")) is list and len(plan["groups"]) == len(groups),
        "INVALID_REPORTED_COVER_PLAN_STRUCTURE")
    source_ids = [row.get("record_id") if type(row) is dict else None for row in original["support_records"]]
    _require(all(_hex(v) for v in source_ids) and len(set(source_ids)) == len(source_ids),
             "INVALID_RETAINED_SOURCE_RECORD_IDENTITIES")
    source_by_id = dict(zip(source_ids, original["support_records"]))
    source_layer_policy = _layer_policy(original.get("layer_policy"), "INVALID_SOURCE_LAYER_POLICY")
    for row in original["support_records"]:
        _require(type(row) is dict and _same(row.get("layer_policy"), source_layer_policy), "SOURCE_RECORD_LAYER_POLICY_MISMATCH")
        _identity(row, "INVALID_SOURCE_RECORD_IDENTITY")
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    counts.update(support_group_count=len(groups), source_support_record_count=len(original["support_records"]),
        source_context_window_count=len(original["context_windows"]))
    eligible, compact, identities, member_ids = [], [], set(), set()
    for group, proposal in zip(groups, plan["groups"]):
        _require(type(group) is dict and type(proposal) is dict and _hex(group.get("group_id"))
            and group["group_id"] not in identities, "INVALID_OR_DUPLICATE_SUPPORT_GROUP_ID")
        identities.add(group["group_id"])
        for key in ("group_id", "member_record_ids", "cover_record_id", "member_flange_permutations"):
            _require(key in group and key in proposal and _same(group[key], proposal[key]),
                     "GROUP_PLAN_STRUCTURAL_PROJECTION_MISMATCH")
        _require(type(group["member_record_ids"]) is list and 1 <= len(group["member_record_ids"]) <= MAX_GROUPS
            and all(_hex(v) for v in group["member_record_ids"])
            and len(set(group["member_record_ids"])) == len(group["member_record_ids"]), "INVALID_GROUP_MEMBERS")
        _require(not member_ids.intersection(group["member_record_ids"]), "REUSED_MEMBER_ACROSS_SUPPORT_GROUPS")
        member_ids.update(group["member_record_ids"])
        _require(proposal.get("status") in ("COVERED", "UNAVAILABLE"), "INVALID_COVER_PLAN_GROUP_STATUS")
        covered = proposal["status"] == "COVERED"
        counts["covered_group_count"] += int(covered)
        if covered:
            _require(group["cover_record_id"] in group["member_record_ids"], "INVALID_EXISTING_COVER_ID")
            witness = source_by_id.get(group["cover_record_id"])
            _require(type(witness) is dict and "fit_policy" in group and "fit_policy" in witness
                and _same(group["fit_policy"], witness["fit_policy"]), "COVER_FIT_POLICY_WITNESS_MISMATCH")
            _require(_same(group.get("layer_policy"), witness["layer_policy"])
                and _same(group["layer_policy"], source_layer_policy), "COVER_LAYER_POLICY_WITNESS_MISMATCH")
            witness_identity = _identity(witness, "INVALID_COVER_WITNESS_IDENTITY")
            _require(all(group.get(key) == witness.get(key) for key in _IDENTITY_KEYS), "COVER_GROUP_IDENTITY_WITNESS_MISMATCH")
            permutations = group["member_flange_permutations"]
            _require(type(permutations) is dict and set(permutations) == set(group["member_record_ids"])
                and all(type(v) is list and len(v) == 2 and all(type(i) is int for i in v)
                    and v in ([0, 1], [1, 0]) for v in permutations.values()), "INVALID_GROUP_ROLE_PERMUTATIONS")
        else:
            _require(group["cover_record_id"] is None and group["member_flange_permutations"] == {},
                     "UNCOVERED_GROUP_HAS_COVER_WITNESS")
        if group.get("status") == "UNAVAILABLE":
            _require(group.get("measurements") is None and group.get("finite_support") is None
                and bool(_reasons(group.get("reasons"))), "REFUSED_GROUP_HAS_MEASUREMENT_OR_CARRIER")
            counts["refused_group_count"] += 1
            continue
        _require(covered and group.get("status") == "MEASURED_CONDITIONAL"
            and _reasons(group.get("reasons")) == [], "INVALID_MEASURED_GROUP_STATUS")
        quantity = _quantity(group.get("measurements"), group.get("fit_policy"))
        _require(all(group["measurements"].get(key) == witness.get(key) for key in _IDENTITY_KEYS),
                 "MEASUREMENT_IDENTITY_WITNESS_MISMATCH")
        counts["measured_group_count"] += 1
        finite = group.get("finite_support")
        _require(type(finite) is dict and finite.get("status") in ("SUPPORTED", "UNAVAILABLE"),
                 "INVALID_FINITE_SUPPORT_STATUS")
        if finite["status"] == "UNAVAILABLE":
            _require(finite.get("instance") is None and bool(_reasons(finite.get("reasons"))),
                     "FINITE_REFUSAL_HAS_CARRIER")
            counts["finite_refused_group_count"] += 1
        else:
            _require(_reasons(finite.get("reasons")) == [], "SUPPORTED_FINITE_HAS_REFUSAL")
            instance = finite.get("instance")
            projected = _line(instance, full=True, quantity=quantity)
            _require(instance["bend_id"] == group["group_id"], "FINITE_CARRIER_GROUP_ID_MISMATCH")
            _require(all(instance.get(key) == witness.get(key) for key in _IDENTITY_KEYS),
                     "CARRIER_IDENTITY_WITNESS_MISMATCH")
            eligible.append(instance)
            compact.append(projected)
            counts["finite_supported_group_count"] += 1
            counts["inner_face_eligible_carrier_count" if witness_identity[1] else "identity_unresolved_carrier_count"] += 1
    for rows in (measured.get("predictions"), payload["predictions"]):
        _require(type(rows) is list and len(rows) <= MAX_PREDICTIONS and _same(rows, eligible),
                 "TOP_V2_ELIGIBLE_PREDICTION_PROJECTION_MISMATCH")
    _require(member_ids == set(source_ids), "SUPPORT_GROUP_SOURCE_RECORD_DENOMINATOR")
    _require((payload["status"] == "PARTIAL") == (counts["measured_group_count"] > 0),
             "MEASURED_STATUS_COUNT_CONTRADICTION")
    registration = payload.get("registration")
    _require(type(registration) is dict and registration.get("status") == "CONDITIONAL_MEASUREMENT_CONTEXTS"
        and registration.get("transform_scan_to_cad", False) is None, "INVALID_CONDITIONAL_REGISTRATION_CLAIM")
    evidence = registration.get("evidence")
    _require(type(evidence) is dict and evidence.get("accepted_registration") is False
        and evidence.get("joint_configuration_status") == "NOT_ESTABLISHED", "INVALID_CONTEXT_AUTHORITY_CLAIM")
    indices = evidence.get("measurement_context_indices")
    _require(type(indices) is list and 1 <= len(indices) <= 48
        and all(type(v) is int and 0 <= v < 48 for v in indices) and len(set(indices)) == len(indices),
        "INVALID_REGISTRATION_CONTEXT_COUNT")
    # V4: the initializer-V16 finite-subset contract is validated explicitly (strict whole-set
    # completeness OR the declared disjoint disposition ledger); the altered meaning never passes silently.
    _require(_registration_evaluation_acceptable(evidence), "INVALID_REGISTRATION_EVALUATION_CONTRACT")
    # 1 when the registration published an evaluated allowance subset (expired/omitted alternatives unknown),
    # 0 for strict whole-set completeness; an integer count so the unchanged aggregation grammar applies.
    counts["subset_registration_count"] = 0 if evidence.get("candidate_evaluation_complete") is True else 1
    counts.update(registration_context_count=len(indices), emitted_carrier_count=len(eligible))
    return compact, counts


def _registration_evaluation_acceptable(evidence):
    """Same rule as context measurement V3 (kept textually independent so this scorer binds its own
    grammar): strict `candidate_evaluation_complete` True, OR False with the explicit V16 contract."""
    if type(evidence) is not dict:
        return False
    if evidence.get("candidate_evaluation_complete") is True:
        return True
    if evidence.get("candidate_evaluation_complete") is not False:
        return False
    ledger = evidence.get("candidate_disposition_ledger")
    candidates = evidence.get("candidates")
    if (evidence.get("publication_contract") != SUBSET_PUBLICATION_CONTRACT
            or evidence.get("evaluated_subset_complete") is not True
            or evidence.get("candidate_omission_status") != "ALLOWANCE_OMISSIONS_RETAINED_UNKNOWN"
            or type(ledger) is not dict or ledger.get("schema") != SUBSET_LEDGER_SCHEMA or type(candidates) is not list
            or not all(type(ledger.get(key)) is list and all(type(i) is int for i in ledger[key]) for key in _LEDGER_KEYS)):
        return False
    members = [index for key in _LEDGER_KEYS for index in ledger[key]]
    by_index = {}
    for row in candidates:
        if type(row) is not dict or type(row.get("index")) is not int or row["index"] in by_index:
            return False
        by_index[row["index"]] = row
    if (sorted(members) != sorted(by_index) or len(set(members)) != len(members)
            or ledger.get("selected_count") != len(candidates)
            or ledger["failed"] or ledger["not_attempted"] or not (ledger["expired"] or ledger["omitted"])):
        return False
    contexts = evidence.get("measurement_context_indices")
    if type(contexts) is not list or not set(contexts) <= set(ledger["completed"]):
        return False
    expected = dict(completed="ATTEMPTED", expired="EXPIRED", omitted="OMITTED")
    return all(by_index[index].get("execution_status") == status
               for key, status in expected.items() for index in ledger[key])


def _base(request, binding, limits):
    return dict(schema=PROJECTION_SCHEMA, disposition="MALFORMED_RESULT", reason=None,
        authority=AUTHORITY, result_binding=binding, requested_scan_planes=request,
        returned_scan_plane_configurations=None, reported_workflow_status=None,
        reported_workflow_phase=None, reported_workflow_reasons=[], embedded_case_id=None,
        embedded_family_id=None, diagnostics=dict.fromkeys(_COUNT_KEYS), predictions=[],
        limits=limits, observed_bend_count=None, production_use_allowed=False,
        conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED",
        count_metric_semantics=COUNT_SEMANTICS, reference_carrier_compatibility="UNVERIFIED_NOT_ASSERTED_BY_SCORING")


def project_gold11_contact_workflow_v5(result_bytes, *, expected_result_sha256,
        expected_scan_plane_mode, expected_scan_plane_configuration=None,
        max_result_bytes=MAX_RESULT_BYTES, max_projection_bytes=MAX_PROJECTION_BYTES):
    """Validate one supplied result before retaining any successful projection.

    Malformed result bytes, binding mismatch or malformed later records yield
    one whole-case zero-prediction disposition. Invalid caller declarations
    raise ValueError, rather than being misattributed to inference. Refusals can
    legitimately precede returned configuration evidence. A completed composite
    result must carry identical requested configuration at all three locations.
    """
    _require(_hex(expected_result_sha256), "INVALID_CALLER_RESULT_SHA256")
    request = _request(expected_scan_plane_mode, expected_scan_plane_configuration)
    _require(type(max_result_bytes) is int and 1 <= max_result_bytes <= MAX_RESULT_BYTES
        and type(max_projection_bytes) is int and MIN_PROJECTION_BYTES <= max_projection_bytes <= MAX_PROJECTION_BYTES,
        "INVALID_PROJECTION_BYTE_LIMITS")
    limits = dict(result_bytes=max_result_bytes, projection_bytes=max_projection_bytes,
        support_groups=MAX_GROUPS, predictions=MAX_PREDICTIONS, json_nodes=MAX_JSON_NODES, json_depth=MAX_JSON_DEPTH)
    binding = dict(expected_sha256=expected_result_sha256, supplied_sha256=None,
        supplied_bytes=len(result_bytes) if type(result_bytes) is bytes else None, matched=False,
        authority="SUPPLIED_BYTES_VS_CALLER_SHA256_ONLY_NOT_EXTERNAL_OR_RAW_ATTESTATION",
        separate_result_retention="CALLER_RESPONSIBILITY_NOT_CHECKED_HERE")
    output = _base(request, binding, limits)
    try:
        _require(type(result_bytes) is bytes and 1 <= len(result_bytes) <= max_result_bytes,
                 "RESULT_BYTE_TYPE_OR_BUDGET")
        binding["supplied_sha256"] = hashlib.sha256(result_bytes).hexdigest()
        binding["matched"] = binding["supplied_sha256"] == expected_result_sha256
        _require(binding["matched"], "CALLER_RESULT_SHA256_MISMATCH")
        payload = _decode(result_bytes)
        _require(type(payload) is dict and payload.get("schema") == WORKFLOW_SCHEMA, "V13_WORKFLOW_SCHEMA_REQUIRED")
        _require(payload.get("status") in ("PARTIAL", "UNAVAILABLE")
            and type(payload.get("phase")) is str and 1 <= len(payload["phase"]) <= 128,
            "INVALID_V13_STATUS_OR_PHASE")
        output.update(reported_workflow_status=payload["status"], reported_workflow_phase=payload["phase"],
            reported_workflow_reasons=_reasons(payload.get("reasons")))
        for key in ("case_id", "family_id"):
            if key in payload:
                _require(_legacy._identifier(payload[key]), "INVALID_EMBEDDED_IDENTITY")
                output["embedded_" + key] = payload[key]
        completed = payload["phase"] == "COMPLETE"
        _claims(payload, _TOP_CLAIMS, required=completed or payload["status"] == "PARTIAL")
        for key in ("observed_bend_count", "production_use_allowed", "conformance"):
            _claims(payload, {key: _TOP_CLAIMS[key]})
        rows = payload.get("predictions")
        _require(type(rows) is list and len(rows) <= MAX_PREDICTIONS, "INVALID_V13_PREDICTION_DENOMINATOR")
        configurations = _returned_configurations(payload, request, completed=completed)
        if completed:
            _require(payload.get("connected_measurement_contexts") is True, "CONNECTED_V13_REQUIRED")
            predictions, counts = _complete_measurements(payload)
            disposition = ("SCORED_CONDITIONAL" if predictions else "COMPLETE_NO_LOCALIZED_CARRIERS") \
                if payload["status"] == "PARTIAL" else "REFUSED"
        else:
            _require(payload["status"] == "UNAVAILABLE" and rows == [], "NONCOMPLETE_RESULT_HAS_PREDICTIONS")
            measurements = payload.get("measurements")
            if measurements is not None:
                _claims(measurements, _MEASUREMENT_CLAIMS)
                _require(measurements.get("status") == "UNAVAILABLE"
                    and measurements.get("predictions") == [] and measurements.get("support_groups") == []
                    and measurements.get("computation_status") != "COMPLETE", "REFUSED_RESULT_HAS_MEASUREMENT_PREFIX")
            predictions, counts, disposition = [], dict.fromkeys(_COUNT_KEYS), "REFUSED"
        output.update(disposition=disposition,
            reason="EXPLICIT_WORKFLOW_UNAVAILABLE" if disposition == "REFUSED" else None,
            returned_scan_plane_configurations=configurations, predictions=predictions, diagnostics=counts)
        _encoded_size(output, max_projection_bytes)
        return output
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError) as error:
        # No diagnostic count/record prefix or partially admitted configuration
        # survives failure. The byte binding and bounded reported phase remain.
        refused = _base(request, binding, limits)
        refused.update(reason=str(error)[:256] or type(error).__name__,
            reported_workflow_status=output["reported_workflow_status"],
            reported_workflow_phase=output["reported_workflow_phase"])
        _encoded_size(refused, max_projection_bytes)
        return refused


def _projected_case(payload, case_id, family_id, request):
    try:
        _json_tree(payload, maximum_nodes=200_000)
        _encoded_size(payload, MAX_PROJECTION_BYTES)
        # An absent/failed worker has no returned V13 bytes to project. The
        # unchanged runner emits this exact small envelope instead. Admit only
        # its refusal, without manufacturing a hash or configuration receipt.
        if type(payload) is dict and set(payload) == {"status", "phase", "reasons", "predictions",
                "observed_bend_count", "production_use_allowed"} and payload.get("status") == "UNAVAILABLE":
            _require(payload["phase"] in ("SUPERVISOR", "GLOBAL_TERMINALIZATION")
                and payload["predictions"] == [] and type(payload["predictions"]) is list
                and payload["observed_bend_count"] is None and payload["production_use_allowed"] is False
                and bool(_reasons(payload["reasons"])), "INVALID_SUPERVISOR_REFUSAL")
            return [], "REFUSED", "EXPLICIT_SUPERVISOR_UNAVAILABLE", dict(
                reported_workflow_phase=payload["phase"], diagnostics=None, result_binding=None)
        _require(type(payload) is dict and payload.get("schema") == PROJECTION_SCHEMA
            and payload.get("authority") == AUTHORITY, "CONTACT_PROJECTION_REQUIRED")
        _require(set(payload) == set(_base(None, None, None)), "INVALID_CONTACT_PROJECTION_SCHEMA")
        _require(_same(payload.get("requested_scan_planes"), request), "PROJECTION_ARM_REQUEST_MISMATCH")
        for key, expected in (("embedded_case_id", case_id), ("embedded_family_id", family_id)):
            _require(payload.get(key) is None or payload[key] == expected, "EMBEDDED_CASE_IDENTITY_CONFLICT")
        _claims(payload, dict(observed_bend_count=None, production_use_allowed=False,
            conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED",
            count_metric_semantics=COUNT_SEMANTICS,
            reference_carrier_compatibility="UNVERIFIED_NOT_ASSERTED_BY_SCORING"))
        disposition = payload.get("disposition")
        _require(type(disposition) is str and disposition in _DISPOSITIONS, "INVALID_PROJECTION_DISPOSITION")
        _require(payload["reported_workflow_status"] is None or payload["reported_workflow_status"] in ("PARTIAL", "UNAVAILABLE"),
                 "INVALID_PROJECTED_WORKFLOW_STATUS")
        _require(payload["reported_workflow_phase"] is None or type(payload["reported_workflow_phase"]) is str
            and 1 <= len(payload["reported_workflow_phase"]) <= 128, "INVALID_PROJECTED_PHASE")
        _reasons(payload["reported_workflow_reasons"])
        _require(payload["reason"] is None or type(payload["reason"]) is str and 1 <= len(payload["reason"]) <= 256,
                 "INVALID_PROJECTED_REASON")
        limits = payload["limits"]
        _require(type(limits) is dict and set(limits) == {"result_bytes", "projection_bytes", "support_groups",
            "predictions", "json_nodes", "json_depth"} and all(type(v) is int for v in limits.values())
            and 1 <= limits["result_bytes"] <= MAX_RESULT_BYTES
            and MIN_PROJECTION_BYTES <= limits["projection_bytes"] <= MAX_PROJECTION_BYTES
            and limits["support_groups"] == MAX_GROUPS and limits["predictions"] == MAX_PREDICTIONS
            and limits["json_nodes"] == MAX_JSON_NODES and limits["json_depth"] == MAX_JSON_DEPTH,
            "INVALID_PROJECTED_LIMITS")
        _encoded_size(payload, limits["projection_bytes"])
        rows, binding = payload.get("predictions"), payload.get("result_binding")
        _require(type(rows) is list and len(rows) <= MAX_PREDICTIONS, "INVALID_COMPACT_PREDICTION_DENOMINATOR")
        _require(type(binding) is dict and set(binding) == {"expected_sha256", "supplied_sha256", "supplied_bytes",
            "matched", "authority", "separate_result_retention"} and _hex(binding.get("expected_sha256"))
            and (binding["supplied_sha256"] is None or _hex(binding["supplied_sha256"]))
            and (binding["supplied_bytes"] is None or type(binding["supplied_bytes"]) is int and binding["supplied_bytes"] >= 0)
            and type(binding.get("matched")) is bool
            and binding["authority"] == "SUPPLIED_BYTES_VS_CALLER_SHA256_ONLY_NOT_EXTERNAL_OR_RAW_ATTESTATION"
            and binding["separate_result_retention"] == "CALLER_RESPONSIBILITY_NOT_CHECKED_HERE",
            "INVALID_PROJECTION_RESULT_BINDING")
        _require(binding["matched"] is (binding["supplied_sha256"] == binding["expected_sha256"]),
                 "CONTRADICTORY_PROJECTED_HASH_BINDING")
        if disposition == "MALFORMED_RESULT":
            _require(rows == [] and type(payload.get("reason")) is str and payload["reason"],
                     "MALFORMED_PROJECTION_HAS_PREFIX")
            _require(payload["diagnostics"] == dict.fromkeys(_COUNT_KEYS)
                and payload["returned_scan_plane_configurations"] is None, "MALFORMED_PROJECTION_HAS_DIAGNOSTIC_PREFIX")
            return [], disposition, payload["reason"], payload
        _require(binding["matched"] is True and binding.get("supplied_sha256") == binding["expected_sha256"]
            and type(binding.get("supplied_bytes")) is int and 1 <= binding["supplied_bytes"] <= limits["result_bytes"],
            "UNBOUND_ADMITTED_PROJECTION")
        configurations = payload.get("returned_scan_plane_configurations")
        _require(type(configurations) is dict and set(configurations) == {"policies", "registration", "initialization"},
                 "INVALID_RETURNED_CONFIGURATION_PROJECTION")
        for row in configurations.values():
            _require(type(row) is dict and set(row) == {"present", "configuration"} and type(row["present"]) is bool,
                     "INVALID_RETURNED_CONFIGURATION_PROJECTION")
            if request["mode"] == "v7":
                _require(row == dict(present=False, configuration=None), "UNREQUESTED_PROJECTED_CONFIGURATION")
            elif row["present"]:
                _require(_same(row["configuration"], request["configuration"]), "PROJECTED_CONFIGURATION_MISMATCH")
            else:
                _require(row["configuration"] is None and payload.get("reported_workflow_phase") != "COMPLETE",
                         "MISSING_COMPLETE_PROJECTED_CONFIGURATION")
        normalized, seen = [], set()
        for row in rows:
            compact = _line(row, full=False)
            _require(compact["bend_id"] not in seen, "DUPLICATE_CONTACT_ID")
            seen.add(compact["bend_id"])
            normalized.append({key: compact[key] for key in ("bend_id", "line_start", "line_end")})
            _identity(compact, "INVALID_COMPACT_CARRIER_IDENTITY")
        counts = payload.get("diagnostics")
        _require(type(counts) is dict and set(counts) == set(_COUNT_KEYS)
            and all(v is None or type(v) is int and 0 <= v <= (
                4096 if key == "source_context_window_count" else 48 if key == "registration_context_count" else MAX_GROUPS)
                for key, v in counts.items()), "INVALID_PROJECTION_COUNTS")
        if disposition == "SCORED_CONDITIONAL":
            _require(payload.get("reported_workflow_status") == "PARTIAL"
                and payload.get("reported_workflow_phase") == "COMPLETE" and bool(normalized)
                and payload["reason"] is None, "INVALID_LOCALIZED_PROJECTION")
        elif disposition == "COMPLETE_NO_LOCALIZED_CARRIERS":
            _require(payload.get("reported_workflow_status") == "PARTIAL"
                and payload.get("reported_workflow_phase") == "COMPLETE" and normalized == []
                and type(counts["measured_group_count"]) is int and counts["measured_group_count"] > 0,
                "INVALID_COMPLETE_EMPTY_PROJECTION")
            _require(payload["reason"] is None, "COMPLETE_EMPTY_HAS_REFUSAL_REASON")
        else:
            _require(payload.get("reported_workflow_status") == "UNAVAILABLE" and normalized == [],
                     "REFUSED_PROJECTION_HAS_CARRIER")
        if payload.get("reported_workflow_phase") == "COMPLETE":
            _require(all(type(v) is int for v in counts.values())
                and counts["emitted_carrier_count"] == counts["finite_supported_group_count"] == len(normalized)
                and counts["inner_face_eligible_carrier_count"]+counts["identity_unresolved_carrier_count"] == counts["emitted_carrier_count"]
                and counts["measured_group_count"] == counts["finite_supported_group_count"] + counts["finite_refused_group_count"]
                and counts["support_group_count"] == counts["measured_group_count"] + counts["refused_group_count"]
                and counts["measured_group_count"] <= counts["covered_group_count"] <= counts["support_group_count"] <= MAX_GROUPS
                and counts["source_support_record_count"] >= counts["support_group_count"]
                and 1 <= counts["registration_context_count"] <= 48, "PROJECTION_COUNT_CONTRADICTION")
        else:
            _require(counts == dict.fromkeys(_COUNT_KEYS), "NONCOMPLETE_PROJECTION_HAS_MEASUREMENT_COUNTS")
        return normalized, disposition, payload.get("reason"), payload
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError) as error:
        return [], "MALFORMED_RESULT", str(error)[:256] or type(error).__name__, None


def _lane(results, references, families, request):
    rows, dispositions = [], {}
    thresholds = Thresholds(angle_deg=5., transverse_offset_mm=4.,
        gold_span_overlap=.60, prediction_span_overlap=.60)
    for case_id in sorted(families):
        predictions, disposition, reason, projection = _projected_case(results[case_id], case_id, families[case_id], request)
        matched = match_lines(references[case_id], predictions, thresholds)
        row = dict(matched, case_id=case_id, family_id=families[case_id])
        rows.append(row)
        singleton = score_localized_cases([row])
        dispositions[case_id] = dict(disposition=disposition, reason=reason, family_id=families[case_id],
            **matched, **{key: singleton[key] for key in ("precision", "recall", "f1")},
            exact_emitted_support_count=singleton["exact_count_cases"] == 1,
            count_metric_semantics=COUNT_SEMANTICS, observed_bend_count=None,
            reported_workflow_phase=projection["reported_workflow_phase"] if projection else None,
            diagnostics=projection["diagnostics"] if projection else None,
            result_binding=projection["result_binding"] if projection else None)
    families_out = {family: _legacy._summary([row for row in rows if row["family_id"] == family])
        for family in sorted(set(families.values()))}
    return dict(geometry=_legacy._summary(rows), families=families_out,
        minimum_family_recall=min(row["recall"] for row in families_out.values()), case_dispositions=dispositions,
        refused_case_count=sum(row["disposition"] == "REFUSED" for row in dispositions.values()),
        malformed_case_count=sum(row["disposition"] == "MALFORMED_RESULT" for row in dispositions.values()),
        complete_no_localized_carrier_case_count=sum(row["disposition"] == "COMPLETE_NO_LOCALIZED_CARRIERS" for row in dispositions.values()),
        physical_observed_bend_count=dict(known_case_count=0, unknown_case_count=11, unknown_rate=1.,
            strict_count_target_met=False, status="UNMET_NO_PHYSICAL_OBSERVED_COUNTS"))


def score_gold11_contact_workflows_v5(*, reference_records_by_case, control_results_by_case,
        challenger_results_by_case, case_family_by_id, scan_plane_requests_by_arm):
    """Score exactly eleven/six/seventy-one supplied development records.

    Complete-map/reference/caller-request failures raise. Invalid individual
    projections keep their case with zero predictions and every reference FN.
    The runner's exact small supervisor UNAVAILABLE envelope is also admitted
    as a refusal without any returned-result-byte or configuration attestation.
    Unexpected matcher errors propagate; they are not invented engine failures.
    Request roles are explicit so an arm-exchange test can swap both declaration
    and results. The eventual frozen runner determines control/challenger roles.
    """
    _require(type(case_family_by_id) is dict and len(case_family_by_id) == 11
        and all(_legacy._identifier(case) and _legacy._identifier(family)
            for case, family in case_family_by_id.items())
        and len(set(case_family_by_id.values())) == 6, "REQUIRE_ELEVEN_CASES_SIX_FAMILIES")
    expected = set(case_family_by_id)
    for row in (reference_records_by_case, control_results_by_case, challenger_results_by_case):
        _require(type(row) is dict and set(row) == expected, "CASE_SET_MISMATCH")
    _require(type(scan_plane_requests_by_arm) is dict and set(scan_plane_requests_by_arm) == {"control", "challenger"},
             "EXPLICIT_PAIRED_SCAN_PLANE_REQUESTS_REQUIRED")
    requests = {}
    for arm, row in scan_plane_requests_by_arm.items():
        _require(type(row) is dict and set(row) == {"mode", "configuration"}, "INVALID_ARM_REQUEST")
        requests[arm] = _request(row["mode"], row["configuration"])
    _require({row["mode"] for row in requests.values()} == {"v7", "composite_v1"}, "REQUIRE_V7_COMPOSITE_PAIR")
    references = {case: _legacy._lines(reference_records_by_case[case], prediction=False) for case in sorted(expected)}
    _require(sum(map(len, references.values())) == 71, "REQUIRE_SEVENTY_ONE_REFERENCE_LINES")
    control = _lane(control_results_by_case, references, case_family_by_id, requests["control"])
    challenger = _lane(challenger_results_by_case, references, case_family_by_id, requests["challenger"])
    keys = ("true_positive", "false_positive", "false_negative", "precision", "recall", "f1",
        "exact_emitted_support_count_accuracy", "exact_localized_target_set_accuracy")
    result = dict(schema=SCORING_SCHEMA, status="COMPUTED_DEVELOPMENT_DIAGNOSTIC",
        case_count=11, family_count=6, reference_line_count=71,
        scan_plane_requests_by_arm=requests, control=control, challenger=challenger,
        paired_deltas={**{key: _legacy._delta(control["geometry"][key], challenger["geometry"][key]) for key in keys},
            "minimum_family_recall": _legacy._delta(control["minimum_family_recall"], challenger["minimum_family_recall"])},
        family_paired_deltas={family: {key: _legacy._delta(control["families"][family][key], challenger["families"][family][key])
            for key in keys} for family in control["families"]},
        matching=dict(axis_angle_deg_max=5., transverse_offset_mm_max=4., gold_span_overlap_min=.60,
            prediction_span_overlap_min=.60, method="UNCHANGED_LEGACY_ONE_TO_ONE_MINIMUM_COST"),
        comparison_scope="NEW_V13_PROVISIONAL_COMMON_MM_RAW_CONTACT_ENDPOINT_GEOMETRY_ONLY",
        prediction_carrier=CARRIER, prediction_geometry_frame=RAW_FRAME,
        material_side_status="PER_CARRIER_DECLARED_UNVERIFIED_SEE_CASE_DISPOSITIONS", physical_trim_endpoints=None,
        scoring_scope=SCORING_SCOPE, inner_face_specific_claims="NONE_UNRESOLVED_CARRIERS_SCORED_AS_GENERIC_CONTACT_LINES_ONLY",
        carrier_identity_policy="ELIGIBILITY_IS_A_DECLARED_FUNCTION_OF_MATERIAL_SIDE_STATUS_NEVER_A_DEFAULT",
        reference_carrier_compatibility="UNVERIFIED_NOT_ASSERTED_BY_SCORING",
        reference_frame_compatibility="PROVISIONAL_ASSUMPTION_NOT_VERIFIED",
        scoring_operations="DIRECT_ENDPOINT_MATCHING_ONLY_NO_TRANSFORM_OR_UNIT_ESTIMATION",
        projection_authority=AUTHORITY, cohort_authentication="CALLER_FROZEN_CASE_IDS_AND_REFERENCE_BYTES_REQUIRED",
        caller_inference_gold_isolation="REQUIRES_INDEPENDENT_PROTOCOL_AND_SOURCE_AUDIT",
        failure_policy="WHOLE_CASE_ZERO_PREDICTIONS_ALL_REFERENCE_FNS_NO_PREFIX_OR_DROPPED_CASE",
        complete_empty_policy="VALID_MEASUREMENTS_WITHOUT_FINITE_CARRIER_RETAIN_ALL_REFERENCE_FNS",
        prediction_safety_cap_per_case=MAX_PREDICTIONS, observed_bend_count=None,
        strict_physical_engine_targets_met=False, per_bend_measurement_accuracy="NOT_EVALUATED_NO_MEASUREMENT_TRUTH",
        historical_score_comparability="UNESTABLISHED", historical_score_beaten_claim_allowed=False,
        production_use_allowed=False, floor_release_allowed=False, conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED")
    _encoded_size(result, MAX_SCORE_BYTES)
    return result
