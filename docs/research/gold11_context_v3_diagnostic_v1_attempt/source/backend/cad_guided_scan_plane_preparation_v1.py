"""Unqualified owned composite scan-plane proposal manifest, no CAD or IO.

Preserves truthful V7 and seeded schemas; no synthetic V7 core records, support
merging or nominal geometry. All qualified proposals retain original origins.
Connected only through the explicit composite_v1 initializer/workflow option;
not the default path and not connected-run, native or physical qualification.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from . import cad_guided_noisy_mesh_planes_v5 as _v5
from . import cad_guided_noisy_mesh_planes_v7 as _v7
from . import cad_guided_seeded_plane_proposals_v1 as _seeded


@dataclass(frozen=True)
class CompositeScanPlaneBudgetsV1:
    max_qualified_planes: int = 65_552
    max_triangle_memberships: int = 20_000_000
    max_raw_memberships: int = 32_000_000
    max_full_support_row_tests: int = 64_000_000
    max_raw_predicate_visits: int = 48_000_000
    max_triangle_predicate_visits: int = 160_000_000
    max_edge_predicate_visits: int = 192_000_000
    max_fit_evaluations: int = 65_584
    max_retained_array_bytes: int = 536_870_912
    max_metadata_bytes: int = 67_108_864


def _configuration(budgets):
    _v5._require(type(budgets) is CompositeScanPlaneBudgetsV1, "INVALID_COMPOSITE_BUDGET_TYPE")
    cap, maximum = asdict(budgets), asdict(CompositeScanPlaneBudgetsV1())
    _v5._require(all(type(v) is int and 0 <= v <= maximum[k] for k, v in cap.items()),
                 "INVALID_COMPOSITE_BUDGET")
    return cap


def _aggregate(v7, seeded, cap):
    a, b = v7["statistics"], seeded["statistics"]
    totals = {k: a[k] + b[k] for k in (
        "triangle_memberships", "raw_memberships", "full_support_row_tests",
        "fit_evaluations", "retained_array_bytes")}
    totals.update(raw_predicate_visits=a["cone_row_tests"] + b["proposal_raw_row_visits"],
        triangle_predicate_visits=a["closure_triangle_tests"] + b["proposal_triangle_visits"]
            + b["seed_selection_triangle_visits"],
        edge_predicate_visits=a["closure_edge_tests"] + b["proposal_edge_visits"])
    for field, value in totals.items():
        _v5._require(type(value) is int and value >= 0 and value <= cap["max_" + field],
                     "COMPOSITE_" + field.upper() + "_EXCEEDED")
    return totals


def _lane_digest(value, cap, work):
    envelope, _, _ = _v7._envelope(value, work, cap["max_metadata_bytes"])
    digest, count = hashlib.sha256(), 0
    for fragment in json.JSONEncoder(sort_keys=True, ensure_ascii=True,
            allow_nan=False, separators=(",", ":")).iterencode(envelope):
        blob = fragment.encode("utf-8")
        count += len(blob)
        _v5._require(count <= cap["max_metadata_bytes"], "COMPOSITE_LANE_METADATA_BUDGET_EXCEEDED")
        digest.update(blob)
    return digest.hexdigest()


def _lane_budgets(noisy_budgets, seeded_budgets, cap, completed=None):
    """Deterministic residual allowances, frozen before each lane dispatch.

    No schedule-size refill/reduction after observing a fit. If the fixed seed
    schedule cannot fit these residual allowances, its ordinary preflight fails.
    """
    if completed is None:
        values = asdict(noisy_budgets)
        for key in ("triangle_memberships", "raw_memberships", "full_support_row_tests",
                    "fit_evaluations", "retained_array_bytes", "metadata_bytes"):
            values["max_" + key] = min(values["max_" + key], cap["max_" + key])
        for local, shared in (("cone_row_tests", "raw_predicate_visits"),
                              ("closure_triangle_tests", "triangle_predicate_visits"),
                              ("closure_edge_tests", "edge_predicate_visits")):
            values["max_" + local] = min(values["max_" + local], cap["max_" + shared])
        return _v5.NoisyMeshPlaneBudgetsV5(**values)
    # Validate caller policy before reading it or deriving a new allowance.
    values = _seeded._configuration(seeded_budgets)
    prior = completed["statistics"]
    for key in ("triangle_memberships", "raw_memberships", "full_support_row_tests",
                "fit_evaluations", "retained_array_bytes", "metadata_bytes"):
        remaining = cap["max_" + key] - prior[key]
        _v5._require(remaining >= 0, "COMPOSITE_NEGATIVE_REMAINING_" + key.upper())
        values["max_" + key] = min(values["max_" + key], remaining)
    values["max_proposal_raw_row_visits"] = min(values["max_proposal_raw_row_visits"],
        cap["max_raw_predicate_visits"] - prior["cone_row_tests"])
    values["max_proposal_edge_visits"] = min(values["max_proposal_edge_visits"],
        cap["max_edge_predicate_visits"] - prior["closure_edge_tests"])
    # Selection and proposal each evaluate the same M triangles for every seed.
    half = (cap["max_triangle_predicate_visits"] - prior["closure_triangle_tests"]) // 2
    for key in ("max_seed_selection_triangle_visits", "max_proposal_triangle_visits"):
        values[key] = min(values[key], half)
    return _seeded.SeededPlaneBudgetsV1(**values)


def prepare_composite_scan_planes_v1(points, triangle_raw_rows, *, position_error_bounds_mm,
        noisy_policy=_v5.NoisyMeshPlanePolicyV5(), noisy_budgets=_v5.NoisyMeshPlaneBudgetsV5(),
        seeded_budgets=_seeded.SeededPlaneBudgetsV1(), budgets=CompositeScanPlaneBudgetsV1()):
    """Both lanes complete or no composite publication; no failure fallback.

    Lane allowances are restricted by shared caps before dispatch; the seeded
    lane receives their deterministic remainder. Metadata's composite envelope
    is additionally bounded before publication. No failed-lane fallback exists.
    """
    result = dict(schema_version="composite_scan_planes_v1", status="UNAVAILABLE", reasons=[],
        unit="mm", coordinate_frame="RAW_SCAN_UNREGISTERED", raw_coordinate_welding=False,
        residual_trimming=False, production_use_allowed=False, observed_bend_count=None,
        physical_plane_count=None, accepted_registration=False, search_exhaustive=False,
        input_digests={}, lanes={}, planes=[],
        lane_execution=dict(v7="NOT_ATTEMPTED", seeded_v1="NOT_ATTEMPTED"),
        statistics=dict(selected_policy_evaluation_complete=False,
            metadata_bytes=0, metadata_expanded_bytes=0, metadata_node_visits=0,
            metadata_array_bytes_hashed=0))
    work = _v7._Work()
    try:
        cap = _configuration(budgets)
        result["budgets"] = cap
        gate, input_cap = _v5._configuration(noisy_policy, noisy_budgets)
        seeded_budgets = _seeded.SeededPlaneBudgetsV1(**_seeded._configuration(seeded_budgets))
        result["requested_lane_budgets"] = dict(v7=dict(input_cap), seeded_v1=asdict(seeded_budgets))
        # One caller snapshot before either lane. Each lane owns and validates
        # its internal copy; caller mutation cannot split the pair's input.
        p, t, epsilon = _v5._inputs(points, triangle_raw_rows, position_error_bounds_mm, input_cap)
        binding = dict(points_sha256=_v5._digest(p), triangle_raw_rows_sha256=_v5._digest(t),
                       position_error_bounds_mm_sha256=_v5._digest(epsilon))
        result["input_digests"] = binding
        first_budget = _lane_budgets(_v5.NoisyMeshPlaneBudgetsV5(**input_cap), seeded_budgets, cap)
        result["effective_lane_budgets"] = dict(v7=asdict(first_budget))
        result["lane_execution"]["v7"] = "ATTEMPTED"
        first = _v7.extract_noisy_scan_plane_proposals_v7(p, t, position_error_bounds_mm=epsilon,
            policy=_v5.NoisyMeshPlanePolicyV5(**gate), budgets=first_budget)
        result["lanes"]["v7"] = first
        result["lane_execution"]["v7"] = first["status"]
        _v5._require(first["status"] == "COMPLETE", "COMPOSITE_V7_LANE_UNAVAILABLE")
        second_budget = _lane_budgets(noisy_budgets, seeded_budgets, cap, completed=first)
        result["effective_lane_budgets"]["seeded_v1"] = asdict(second_budget)
        result["lane_execution"]["seeded_v1"] = "ATTEMPTED"
        second = _seeded.extract_seeded_plane_proposals_v1(p, t, position_error_bounds_mm=epsilon,
            qualification_policy=_v5.NoisyMeshPlanePolicyV5(**gate), budgets=second_budget)
        result["lanes"]["seeded_v1"] = second
        result["lane_execution"]["seeded_v1"] = second["status"]
        _v5._require(second["status"] == "COMPLETE", "COMPOSITE_SEEDED_LANE_UNAVAILABLE")
        for lane in (first, second):
            _v5._require(lane["input_digests"] == binding, "COMPOSITE_LANE_INPUT_MISMATCH")
        result["statistics"].update(_aggregate(first, second, cap))
        origins = [("v7", ref) for ref in first["proposals"]] + [
            ("seeded_v1", ref) for ref in second["proposals"]]
        _v5._require(len(origins) <= cap["max_qualified_planes"], "COMPOSITE_PLANE_COUNT_EXCEEDED")
        for plane_id, (lane, ref) in enumerate(origins):
            kind, identifier = ref["kind"], ref["record_id"]
            table = first["cores" if kind == "CORE" else "closures"] if lane == "v7" else second["regions"]
            record = table[identifier]
            _v5._require(record["analysis"]["status"] == "PROPOSED", "COMPOSITE_UNQUALIFIED_REFERENCE")
            result["planes"].append(dict(plane_id=plane_id, lane=lane, kind=kind, record_id=identifier,
                triangle_rows_sha256=record["triangle_rows_sha256"],
                raw_vertex_rows_sha256=record["raw_vertex_rows_sha256"]))
        qualified_triangle_memberships = qualified_raw_memberships = 0
        for lane, ref in origins:
            record = (first["cores" if ref["kind"] == "CORE" else "closures"][ref["record_id"]]
                      if lane == "v7" else second["regions"][ref["record_id"]])
            qualified_triangle_memberships += len(record["triangle_rows"])
            qualified_raw_memberships += len(record["raw_vertex_rows"])
        result["lane_metadata_sha256"] = {name: _lane_digest(value, cap, work)
            for name, value in result["lanes"].items()}
        result["statistics"].update(qualified_plane_count=len(origins),
            qualified_triangle_memberships=qualified_triangle_memberships,
            qualified_raw_memberships=qualified_raw_memberships,
            selected_policy_evaluation_complete=True, input_raw_rows=len(p), input_triangles=len(t))
        result.update(status="COMPLETE", reasons=[] if origins else ["NO_QUALIFIED_COMPOSITE_SCAN_PLANES"],
            support_id_namespace="COMPOSITE_PLANE_ID_WITH_EXPLICIT_LANE_KIND_RECORD_ORIGIN",
            aggregate_work_scope="SUM_OF_LANE_LOGICAL_EVALUATIONS_NOT_PRIMITIVE_CPU_OPERATIONS")
        _v7._finalize(result, cap, work)
    except Exception as error:
        result["statistics"].update(selected_policy_evaluation_complete=False,
            discarded_plane_records=len(result["planes"]), discarded_lane_records=len(result["lanes"]),
            lane_attempts={key: dict(status=value.get("status"), reasons=value.get("reasons"),
                statistics=value.get("statistics")) for key, value in result["lanes"].items()},
            metadata_node_visits=work.visits, metadata_array_bytes_hashed=work.hash_bytes,
            evidence_scope="ATTEMPTED_WORK_ONLY_COMPOSITE_INVALIDATED_WHOLE")
        result.update(status="UNAVAILABLE", reasons=[str(error) or type(error).__name__], lanes={}, planes=[])
    return result
