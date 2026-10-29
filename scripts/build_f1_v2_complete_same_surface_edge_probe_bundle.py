#!/usr/bin/env python3
"""Build the compact, non-embedding lock bundle for V2 complete edges."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.attribute_f1_local_plane_fragment_probe_gold import (  # noqa: E402
    Thresholds,
    _endpoints,
    _line,
    is_match,
    pair_metrics,
)
from scripts.run_f1_v2_complete_same_surface_edge_probe_batch import (  # noqa: E402
    BATCH_KIND,
    BATCH_ROW_FIELDS,
    BATCH_SCHEMA_VERSION,
    EXPECTED_CASE_TOKENS,
    FROZEN_LPF_V2_CONTROL_MANIFEST_SHA256,
    PEAK_RSS_LIMIT_BYTES,
    PREREGISTERED_INFERENCE_MANIFEST_SHA256,
    PREREGISTERED_V2_POOL_MANIFEST_SHA256,
)
from scripts.run_f1_v2_complete_same_surface_edge_probe_replays import (  # noqa: E402
    validate_replay_attestation,
)


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "f1_v2_complete_same_surface_edge_probe_bundle_v1"
LOCK_ATTESTATION_KIND = (
    "f1_v2_complete_same_surface_edge_probe_lock_attestation_v1"
)
PRIMITIVE_PROFILE = "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "f1_v2_complete_same_surface_edge_probe_v1"
CONTROL_CENSUS = {
    "local_plane_fragment_count": 13_413,
    "local_plane_orphan_count": 39,
    "local_plane_pair_count": 17_034,
    "local_transition_component_count": 35_813,
    "geometry_certificate_count": 31,
    "zero_span_component_count": 15_609,
}
CONTROL_CERTIFICATE_DISTRIBUTION = {
    "C017": 2, "C018": 3, "C019": 1, "C022": 11,
    "C032": 1, "C035": 8, "C036": 5,
}
COUNTERFACTUAL_PROFILE = "DELETE_ADDED_SAME_SURFACE_EDGES_RESTORES_LPF_V2_V1"
COUNTERFACTUAL_FIELDS = {
    "counterfactual_profile",
    "restored_inherited_edge_table_sha256",
    "frozen_control_probe_sha256",
    "added_edge_count_deleted",
    "valid",
}
SYNTHETIC_RESOURCE_SCHEMA_VERSION = 1
SYNTHETIC_RESOURCE_KIND = (
    "f1_v2_complete_same_surface_edge_probe_synthetic_worst_cap_resource_v1"
)
SYNTHETIC_RESOURCE_PROFILE = (
    "V2_COMPLETE_EDGE_ABSTRACT_26_REGULAR_CAP_ENVELOPE_V1"
)
SYNTHETIC_RESOURCE_TOP_FIELDS = {
    "schema_version",
    "kind",
    "profile",
    "resource_profile",
    "resource_envelope_only",
    "physical_voxel_realizable",
    "resource_limits",
    "runs",
    "semantic_hashes_identical",
    "all_runs_below_16_gib",
    "attestation_sha256",
}
SYNTHETIC_RESOURCE_RUN_FIELDS = {
    "run_index",
    "status",
    "parent_count",
    "mode_count",
    "modes_per_parent",
    "adjacent_parent_pair_count",
    "eligible_mode_pair_evaluation_count",
    "complete_same_surface_edge_count",
    "inherited_cross_mode_edge_count",
    "effective_mode_edge_count",
    "complete_same_surface_edge_table_sha256",
    "inherited_cross_mode_edge_table_sha256",
    "effective_mode_edge_semantic_sha256",
    "canonical_resource_semantic_sha256",
    "wall_seconds",
    "peak_rss_bytes",
}

LOCK_CRITERION_KEYS = (
    "profile_identity_valid",
    "engineering_profile_hashes_recomputed",
    "exact_nested_schemas_valid",
    "case_count_is_37",
    "successful_case_count_is_37",
    "canonical_rebuilds_identical",
    "scan_hashes_unchanged",
    "candidate_pool_file_hashes_unchanged",
    "a0_prediction_hashes_unchanged",
    "input_table_hashes_unchanged",
    "input_projection_attestations_valid",
    "inherited_v2_caps_valid",
    "no_structural_cap_breach",
    "forbidden_data_absent",
    "source_points_not_consumed",
    "source_points_not_serialized",
    "research_only",
    "inference_safe",
    "prediction_mutated_false",
    "birth_columns_disabled",
    "ownership_not_consumed",
    "solver_not_consumed",
    "truth_not_consumed",
    "designated_replay_attestation_valid",
    "all_measured_peak_rss_below_16_gib",
    "frozen_source_hashes_unchanged",
    "frozen_lpf_control_probe_hashes_reproduced",
    "frozen_lpf_control_census_reproduced",
    "complete_same_surface_edges_recomputed",
    "current_same_surface_edges_exact_subset",
    "current_same_surface_records_byte_identical",
    "inherited_cross_mode_edges_byte_identical",
    "added_edges_only_connect_distinct_neighbor_parents",
    "no_neighbor_degree_truncation",
    "edge_deletion_counterfactual_valid",
    "one_parent_components_remain_zero_span",
    "intervention_nonempty",
    "two_sided_single_parent_support_extended",
    "novel_edge_attributable_certificate_exists",
)

CERTIFICATE_NOVELTY_FIELDS = {
    "case_token",
    "replacement_certificate_id",
    "edge_semantics_dependent",
    "causal_added_same_surface_edge_ids_sha256",
    "control_equivalent_certificate_ids",
    "control_equivalent_certificate_ids_sha256",
    "novel_edge_attributable",
    "certificate_novelty_sha256",
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_BUNDLE_WORKERS = 18


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _record_hash(record: Mapping[str, Any], field: str) -> str:
    return _sha_json({key: value for key, value in record.items() if key != field})


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _index_cases(
    payload: Mapping[str, Any], label: str, *, exact_tokens: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    raw = payload.get("cases")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(item.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        result[token] = dict(item)
    if exact_tokens is not None and set(result) != exact_tokens:
        raise ValueError(f"{label} case set mismatch")
    return result


def build_lock_attestation(criteria: Mapping[str, Any]) -> dict[str, Any]:
    exact = set(criteria) == set(LOCK_CRITERION_KEYS)
    normalized = {
        key: bool(exact and criteria.get(key) is True) for key in LOCK_CRITERION_KEYS
    }
    record = {
        "kind": LOCK_ATTESTATION_KIND,
        "criteria": normalized,
        "ready_for_gold": all(normalized.values()),
    }
    record["lock_attestation_sha256"] = _record_hash(
        record, "lock_attestation_sha256"
    )
    return record


def validate_lock_attestation(record: Mapping[str, Any]) -> None:
    if set(record) != {
        "kind", "criteria", "ready_for_gold", "lock_attestation_sha256"
    }:
        raise ValueError("lock attestation schema mismatch")
    if record.get("kind") != LOCK_ATTESTATION_KIND:
        raise ValueError("lock attestation kind mismatch")
    criteria = record.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(LOCK_CRITERION_KEYS):
        raise ValueError("lock attestation must contain exact 39 criteria")
    if any(value is not True and value is not False for value in criteria.values()):
        raise ValueError("lock attestation criteria must be Boolean")
    ready = all(criteria[key] is True for key in LOCK_CRITERION_KEYS)
    if record.get("ready_for_gold") is not ready:
        raise ValueError("lock ready_for_gold mismatch")
    if _require_sha(record.get("lock_attestation_sha256"), "lock SHA") != _record_hash(
        record, "lock_attestation_sha256"
    ):
        raise ValueError("lock attestation SHA mismatch")


def validate_synthetic_resource_attestation(
    attestation: Mapping[str, Any],
) -> int:
    """Validate the exact three-process professor worst-cap evidence."""

    if set(attestation) != SYNTHETIC_RESOURCE_TOP_FIELDS:
        raise ValueError("synthetic resource attestation top-level schema mismatch")
    if (
        attestation.get("schema_version") != SYNTHETIC_RESOURCE_SCHEMA_VERSION
        or attestation.get("kind") != SYNTHETIC_RESOURCE_KIND
        or attestation.get("profile") != PRIMITIVE_PROFILE
        or attestation.get("resource_profile") != SYNTHETIC_RESOURCE_PROFILE
    ):
        raise ValueError("synthetic resource attestation identity mismatch")
    if (
        attestation.get("resource_envelope_only") is not True
        or attestation.get("physical_voxel_realizable") is not False
    ):
        raise ValueError(
            "synthetic worst-cap evidence must identify the non-geometric abstract envelope"
        )
    limits = attestation.get("resource_limits")
    if not isinstance(limits, Mapping):
        raise ValueError("synthetic resource_limits must be an object")
    expected_limit_values = {
        "parent_cap": 4_000,
        "mode_cap": 16_000,
        "modes_per_parent_cap": 4,
        "adjacent_parent_pair_cap": 52_000,
        "complete_edge_evaluation_cap": 832_000,
        "complete_same_surface_edge_cap": 832_000,
        "inherited_cross_mode_edge_cap": 24_000,
        "effective_mode_edge_cap": 856_000,
        "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
    }
    if set(limits) != set(expected_limit_values):
        raise ValueError("synthetic resource_limits schema mismatch")
    for field, expected in expected_limit_values.items():
        if int(limits.get(field) or -1) != expected:
            raise ValueError(f"synthetic resource limit mismatch: {field}")
    raw_runs = attestation.get("runs")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise ValueError("synthetic resource runs must be a sequence")
    runs = [dict(run) for run in raw_runs if isinstance(run, Mapping)]
    if len(runs) != 3 or [int(run.get("run_index") or -1) for run in runs] != [1, 2, 3]:
        raise ValueError("synthetic resource requires exactly isolated runs 1,2,3")
    peaks: list[int] = []
    semantic_fields = (
        "complete_same_surface_edge_table_sha256",
        "inherited_cross_mode_edge_table_sha256",
        "effective_mode_edge_semantic_sha256",
        "canonical_resource_semantic_sha256",
    )
    semantic_hashes: dict[str, set[str]] = {field: set() for field in semantic_fields}
    for run in runs:
        if set(run) != SYNTHETIC_RESOURCE_RUN_FIELDS:
            raise ValueError("synthetic resource run schema mismatch")
        if run.get("status") != "SUCCESS":
            raise ValueError("synthetic resource run did not succeed")
        exact_counts = {
            "parent_count": 4_000,
            "mode_count": 16_000,
            "modes_per_parent": 4,
            "adjacent_parent_pair_count": 52_000,
            "eligible_mode_pair_evaluation_count": 832_000,
            "complete_same_surface_edge_count": 832_000,
        }
        for field, expected in exact_counts.items():
            if int(run.get(field) or -1) != expected:
                raise ValueError(f"synthetic resource run requires {field}={expected}")
        inherited_rungs = int(run.get("inherited_cross_mode_edge_count", -1))
        if inherited_rungs < 0 or inherited_rungs > 24_000:
            raise ValueError("synthetic resource inherited local-rung bound breached")
        if int(run.get("effective_mode_edge_count") or -1) != 832_000 + inherited_rungs:
            raise ValueError("synthetic effective-edge count mismatch")
        for field in semantic_fields:
            semantic_hashes[field].add(
                _require_sha(run.get(field), f"synthetic {field}")
            )
        peak = int(run.get("peak_rss_bytes") or -1)
        if peak < 0 or peak >= PEAK_RSS_LIMIT_BYTES:
            raise ValueError("synthetic run RSS reaches/exceeds 16 GiB")
        if float(run.get("wall_seconds") or -1.0) < 0.0:
            raise ValueError("synthetic run wall time is invalid")
        peaks.append(peak)
    if any(len(values) != 1 for values in semantic_hashes.values()):
        raise ValueError("synthetic resource semantic hashes are not identical")
    if attestation.get("semantic_hashes_identical") is not True:
        raise ValueError("synthetic resource semantic-hash attestation is false")
    maximum_peak = max(peaks)
    if attestation.get("all_runs_below_16_gib") is not True:
        raise ValueError("synthetic resource RSS attestation is false")
    if _require_sha(
        attestation.get("attestation_sha256"), "synthetic attestation SHA"
    ) != _record_hash(attestation, "attestation_sha256"):
        raise ValueError("synthetic resource attestation SHA mismatch")
    return maximum_peak


def _valid_components(probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    attributions = {
        str(row.get("local_transition_component_id") or ""): dict(row)
        for row in probe.get("component_edge_attribution_table") or ()
        if isinstance(row, Mapping)
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in probe.get("local_plane_pair_records") or ():
        if not isinstance(pair, Mapping):
            raise ValueError("local plane-pair rows must be objects")
        for raw in pair.get("components") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("local component rows must be objects")
            component = deepcopy(dict(raw))
            identifier = str(component.get("local_transition_component_id") or "")
            if not identifier or identifier in seen:
                raise ValueError("local component IDs must be nonempty and unique")
            seen.add(identifier)
            if component.get("geometry_certificate_valid") is True:
                if attributions and identifier not in attributions:
                    raise ValueError("valid replacement certificate lacks causal attribution")
                component.update(attributions.get(identifier) or {})
                component["certificate_id"] = identifier
                component["_line"] = _line(component, identifier=identifier)
                result.append(component)
    return sorted(result, key=lambda row: str(row["certificate_id"]))


def build_certificate_novelty_table(
    *, case_token: str, replacement_probe: Mapping[str, Any],
    frozen_control_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare every valid replacement certificate to same-case frozen controls."""

    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    replacement = _valid_components(replacement_probe)
    control = _valid_components(frozen_control_probe)
    rows: list[dict[str, Any]] = []
    for certificate in replacement:
        equivalents: list[str] = []
        for candidate in control:
            metrics = pair_metrics(
                _endpoints(certificate["_line"]), _endpoints(candidate["_line"])
            )
            if is_match(metrics, thresholds):
                equivalents.append(str(candidate["certificate_id"]))
        equivalents.sort()
        dependent = certificate.get("edge_semantics_dependent") is True
        row = {
            "case_token": case_token,
            "replacement_certificate_id": str(certificate["certificate_id"]),
            "edge_semantics_dependent": dependent,
            "causal_added_same_surface_edge_ids_sha256": _require_sha(
                certificate.get("causal_added_same_surface_edge_ids_sha256"),
                f"{case_token} causal edge IDs SHA",
            ),
            "control_equivalent_certificate_ids": equivalents,
            "control_equivalent_certificate_ids_sha256": _sha_json(equivalents),
            "novel_edge_attributable": bool(dependent and not equivalents),
        }
        row["certificate_novelty_sha256"] = _record_hash(
            row, "certificate_novelty_sha256"
        )
        if set(row) != CERTIFICATE_NOVELTY_FIELDS:
            raise AssertionError("certificate novelty schema drift")
        rows.append(row)
    return rows


def _serialized_control_component_has_zero_span(
    component: Mapping[str, Any],
) -> bool:
    """Classify the frozen LPF component's canonical serialized endpoints.

    LPF component records store their line geometry under
    ``intersection_span_endpoints``.  The generic post-lock ``_endpoints``
    helper does not read that field directly; it expects the record to have
    first passed through ``_line``.  Zero-span records cannot pass through
    ``_line`` by design, so using that helper here silently skipped the entire
    frozen zero-span population.
    """

    raw = component.get("intersection_span_endpoints")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 2
    ):
        raise ValueError("frozen control component lacks two intersection endpoints")
    endpoints: list[tuple[float, float, float]] = []
    for point in raw:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 3
        ):
            raise ValueError("frozen control component endpoint must be a finite 3-vector")
        vector = tuple(float(value) for value in point)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("frozen control component endpoint must be a finite 3-vector")
        endpoints.append(vector)
    squared_span = sum(
        (endpoints[1][axis] - endpoints[0][axis]) ** 2 for axis in range(3)
    )
    return squared_span <= 1e-18


def _control_census(probes: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    census = Counter()
    distribution: dict[str, int] = {}
    for token, probe in sorted(probes.items()):
        fragments = probe.get("local_plane_fragment_table") or ()
        orphans = probe.get("local_plane_orphan_table") or ()
        pairs = probe.get("local_plane_pair_records") or ()
        components = [
            component
            for pair in pairs if isinstance(pair, Mapping)
            for component in pair.get("components") or () if isinstance(component, Mapping)
        ]
        certificates = [c for c in components if c.get("geometry_certificate_valid") is True]
        zero_span = sum(
            _serialized_control_component_has_zero_span(component)
            for component in components
        )
        census.update({
            "local_plane_fragment_count": len(fragments),
            "local_plane_orphan_count": len(orphans),
            "local_plane_pair_count": len(pairs),
            "local_transition_component_count": len(components),
            "geometry_certificate_count": len(certificates),
            "zero_span_component_count": zero_span,
        })
        if certificates:
            distribution[token] = len(certificates)
    return dict(census), distribution


def _core_api() -> tuple[Any, Any, Any, Any, Any, Any]:
    from backend.f1_v2_complete_same_surface_edge_probe import (
        build_v2_complete_same_surface_edge_probe,
        canonical_json_bytes,
        project_edge_probe_input,
        regenerate_frozen_control_probe,
        validate_v2_complete_same_surface_edge_probe_payload,
        verify_edge_deletion_counterfactual,
    )
    return (
        build_v2_complete_same_surface_edge_probe,
        canonical_json_bytes,
        project_edge_probe_input,
        validate_v2_complete_same_surface_edge_probe_payload,
        regenerate_frozen_control_probe,
        verify_edge_deletion_counterfactual,
    )


def _rebuild_case_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Perform one read-only deterministic rebuild in an isolated process."""

    token = str(task["case_token"])
    replacement_path = Path(str(task["replacement_path"])).resolve()
    control_path = Path(str(task["control_path"])).resolve()
    pool_path = Path(str(task["pool_path"])).resolve()
    for label, path in (
        ("replacement", replacement_path),
        ("control", control_path),
        ("pool", pool_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{token}: missing {label}: {path}")
    expected_hashes = {
        "replacement_probe_sha256": replacement_path,
        "frozen_control_probe_sha256": control_path,
        "candidate_pool_sha256": pool_path,
    }
    for field, path in expected_hashes.items():
        if sha256_file(path) != _require_sha(task[field], f"{token}.{field}"):
            raise ValueError(f"{token}: {field} changed before rebuild")

    (
        build_probe,
        canonical_bytes,
        project_input,
        validate_payload,
        regenerate_control,
        verify_counterfactual,
    ) = _core_api()
    projected = project_input(_read(pool_path))
    payload = _read(replacement_path)
    validate_payload(payload, projected_input=projected)
    rebuilt = build_probe(
        projected,
        scan_sha256=str(task["scan_sha256"]),
        candidate_pool_sha256=str(task["candidate_pool_sha256"]),
        a0_prediction_sha256=str(task["a0_prediction_sha256"]),
        frozen_control_probe_sha256=str(task["frozen_control_probe_sha256"]),
    )
    validate_payload(rebuilt, projected_input=projected)
    rebuilt_bytes = canonical_bytes(rebuilt)
    regenerated_control = regenerate_control(
        projected,
        scan_sha256=str(task["scan_sha256"]),
        candidate_pool_sha256=str(task["candidate_pool_sha256"]),
        a0_prediction_sha256=str(task["a0_prediction_sha256"]),
    )
    regenerated_control_bytes = canonical_bytes(regenerated_control)
    counterfactual = verify_counterfactual(
        projected,
        payload,
        scan_sha256=str(task["scan_sha256"]),
        candidate_pool_sha256=str(task["candidate_pool_sha256"]),
        a0_prediction_sha256=str(task["a0_prediction_sha256"]),
        expected_control_probe_sha256=str(task["frozen_control_probe_sha256"]),
    )
    return {
        "case_token": token,
        "rebuilt_probe_sha256": hashlib.sha256(rebuilt_bytes).hexdigest(),
        "rebuilt_probe_bytes": len(rebuilt_bytes),
        "regenerated_control_probe_sha256": hashlib.sha256(
            regenerated_control_bytes
        ).hexdigest(),
        "regenerated_control_probe_bytes": len(regenerated_control_bytes),
        "edge_deletion_counterfactual": counterfactual,
    }


def _execute_case_rebuilds(
    tasks: Sequence[Mapping[str, Any]], *, workers: int,
) -> list[dict[str, Any]]:
    """Execute bounded rebuilds and return canonical case-token order."""

    worker_count = int(workers)
    if worker_count < 1 or worker_count > MAX_BUNDLE_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_BUNDLE_WORKERS}")
    ordered_tasks = sorted(
        (dict(task) for task in tasks), key=lambda task: str(task["case_token"])
    )
    if worker_count == 1:
        results = [_rebuild_case_worker(task) for task in ordered_tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves input order even when completion order differs.
            results = list(executor.map(_rebuild_case_worker, ordered_tasks))
    results.sort(key=lambda result: str(result["case_token"]))
    if [result["case_token"] for result in results] != [
        task["case_token"] for task in ordered_tasks
    ]:
        raise ValueError("parallel rebuild result case set/order mismatch")
    return results

def _validate_manifest_identity(manifest: Mapping[str, Any], tokens: set[str]) -> dict[str, dict[str, Any]]:
    if (
        manifest.get("schema_version") != BATCH_SCHEMA_VERSION
        or manifest.get("kind") != BATCH_KIND
        or manifest.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("ordinary replacement batch identity mismatch")
    if int(manifest.get("case_count") or -1) != len(tokens) or int(
        manifest.get("successful_case_count") or -1
    ) != len(tokens):
        raise ValueError("ordinary replacement batch count mismatch")
    rows = _index_cases(manifest, "ordinary replacement batch", exact_tokens=tokens)
    for token, row in rows.items():
        if set(row) != set(BATCH_ROW_FIELDS) or row.get("status") != "SUCCESS":
            raise ValueError(f"{token}: ordinary replacement row schema/status mismatch")
    return rows


def build_bundle(
    *,
    probe_manifest_path: Path,
    probe_root: Path,
    replay_attestation_path: Path,
    synthetic_resource_attestation_path: Path,
    inference_manifest_path: Path,
    candidate_pool_manifest_path: Path,
    a0_prediction_root: Path,
    frozen_control_manifest_path: Path,
    expected_case_tokens: Sequence[str] = tuple(sorted(EXPECTED_CASE_TOKENS)),
    expected_control_census: Mapping[str, int] = CONTROL_CENSUS,
    expected_control_distribution: Mapping[str, int] = CONTROL_CERTIFICATE_DISTRIBUTION,
    workers: int = 1,
) -> dict[str, Any]:
    """Validate replacement/control lanes and emit only compact attestations."""

    tokens = set(expected_case_tokens)
    manifest = _read(probe_manifest_path.resolve())
    rows = _validate_manifest_identity(manifest, tokens)
    replay = _read(replay_attestation_path.resolve())
    validate_replay_attestation(replay, batch_manifest=manifest)
    if _require_sha(replay.get("probe_manifest_sha256"), "replay batch SHA") != sha256_file(
        probe_manifest_path.resolve()
    ):
        raise ValueError("replay references a different ordinary batch")
    synthetic_resource_attestation_path = synthetic_resource_attestation_path.resolve()
    synthetic_resource = _read(synthetic_resource_attestation_path)
    synthetic_maximum_peak_rss = validate_synthetic_resource_attestation(
        synthetic_resource
    )

    source_paths = {
        "inference_manifest_sha256": inference_manifest_path.resolve(),
        "candidate_pool_manifest_sha256": candidate_pool_manifest_path.resolve(),
        "frozen_control_manifest_sha256": frozen_control_manifest_path.resolve(),
    }
    expected_source_hashes = {
        "inference_manifest_sha256": PREREGISTERED_INFERENCE_MANIFEST_SHA256,
        "candidate_pool_manifest_sha256": PREREGISTERED_V2_POOL_MANIFEST_SHA256,
        "frozen_control_manifest_sha256": FROZEN_LPF_V2_CONTROL_MANIFEST_SHA256,
    }
    for field, path in source_paths.items():
        if sha256_file(path) != expected_source_hashes[field] or manifest.get(field) != expected_source_hashes[field]:
            raise ValueError(f"{field} differs from frozen source")
    inference_rows = _index_cases(_read(inference_manifest_path), "inference manifest", exact_tokens=tokens)
    pool_rows = _index_cases(_read(candidate_pool_manifest_path), "candidate-pool manifest", exact_tokens=tokens)
    control_manifest = _read(frozen_control_manifest_path)
    control_rows = _index_cases(control_manifest, "frozen LPF control", exact_tokens=tokens)

    rebuild_tasks = []
    for token in sorted(tokens):
        row = rows[token]
        rebuild_tasks.append({
            "case_token": token,
            "replacement_path": str((probe_root.resolve() / f"{token}.json")),
            "control_path": str(
                Path(str(control_rows[token].get("probe_path") or "")).resolve()
            ),
            "pool_path": str(
                Path(str(pool_rows[token].get("path") or "")).resolve()
            ),
            "replacement_probe_sha256": str(row["replacement_probe_sha256"]),
            "frozen_control_probe_sha256": str(row["frozen_control_probe_sha256"]),
            "candidate_pool_sha256": str(row["candidate_pool_sha256"]),
            "scan_sha256": str(row["scan_sha256"]),
            "a0_prediction_sha256": str(row["a0_prediction_sha256"]),
        })
    rebuild_results = {
        str(result["case_token"]): result
        for result in _execute_case_rebuilds(rebuild_tasks, workers=workers)
    }

    (
        _build_probe,
        canonical_bytes,
        project_input,
        validate_payload,
        _regenerate_control,
        _verify_counterfactual,
    ) = _core_api()
    control_probes: dict[str, dict[str, Any]] = {}
    replacement_payloads: dict[str, dict[str, Any]] = {}
    compact_cases: list[dict[str, Any]] = []
    maximum_peak_rss = 0
    aggregate_added_edges = 0
    aggregate_two_sided_extensions = 0
    all_counterfactuals = True
    all_one_parent_zero = True

    for token in sorted(tokens):
        row = rows[token]
        replacement_path = (probe_root.resolve() / f"{token}.json")
        control_path = Path(str(control_rows[token].get("probe_path") or "")).resolve()
        pool_path = Path(str(pool_rows[token].get("path") or "")).resolve()
        scan_path = Path(str(inference_rows[token].get("source_scan_path") or "")).resolve()
        prediction_path = a0_prediction_root.resolve() / "cases" / token / "prediction_owned_regions.json"
        for label, path in (("replacement", replacement_path), ("control", control_path), ("pool", pool_path), ("scan", scan_path), ("A0 prediction", prediction_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        for field, path in (("replacement_probe_sha256", replacement_path), ("frozen_control_probe_sha256", control_path), ("candidate_pool_sha256", pool_path), ("scan_sha256", scan_path), ("a0_prediction_sha256", prediction_path)):
            if sha256_file(path) != _require_sha(row.get(field), f"{token}.{field}"):
                raise ValueError(f"{token}: {field} changed")

        payload = _read(replacement_path)
        projected = project_input(_read(pool_path))
        validate_payload(payload, projected_input=projected)
        if replacement_path.read_bytes() != canonical_bytes(payload):
            raise ValueError(f"{token}: replacement payload is not canonical")
        if payload.get("schema_version") != PROBE_SCHEMA_VERSION or payload.get("kind") != PROBE_KIND or payload.get("profile") != PRIMITIVE_PROFILE:
            raise ValueError(f"{token}: replacement payload identity mismatch")
        rebuild_result = rebuild_results[token]
        if (
            str(rebuild_result["rebuilt_probe_sha256"])
            != str(row["replacement_probe_sha256"])
            or int(rebuild_result["rebuilt_probe_bytes"])
            != len(canonical_bytes(payload))
        ):
            raise ValueError(f"{token}: deterministic replacement rebuild mismatch")
        control = _read(control_path)
        if control_path.read_bytes() != canonical_bytes(control):
            raise ValueError(f"{token}: frozen control payload is not canonical")
        if (
            str(rebuild_result["regenerated_control_probe_sha256"])
            != str(row["frozen_control_probe_sha256"])
            or int(rebuild_result["regenerated_control_probe_bytes"])
            != len(canonical_bytes(control))
        ):
            raise ValueError(f"{token}: frozen LPF V2 control did not reproduce")
        counterfactual = dict(rebuild_result["edge_deletion_counterfactual"])
        if (
            set(counterfactual) != COUNTERFACTUAL_FIELDS
            or counterfactual.get("counterfactual_profile") != COUNTERFACTUAL_PROFILE
            or counterfactual.get("valid") is not True
            or int(counterfactual.get("added_edge_count_deleted", -1))
            != int(payload.get("case_counters", {}).get("added_same_surface_edge_count", -2))
        ):
            raise ValueError(f"{token}: edge-deletion counterfactual schema/evidence mismatch")
        control_probes[token] = control
        replacement_payloads[token] = payload

        counters = dict(payload.get("case_counters") or {})
        aggregate_added_edges += int(counters.get("added_same_surface_edge_count") or 0)
        aggregate_two_sided_extensions += int(counters.get("two_sided_one_parent_support_extended_count") or 0)
        all_one_parent_zero &= int(counters.get("one_parent_finite_span_count") or 0) == 0
        all_counterfactuals &= counterfactual.get("valid") is True
        attestation = dict(payload.get("edge_semantics_attestation") or {})
        for field in ("input_table_hashes", "replacement_edge_table_hashes", "output_table_hashes", "case_counters"):
            if dict(row.get(field) or {}) != dict(payload.get(field) or {}):
                raise ValueError(f"{token}: batch/payload {field} mismatch")
        for field, nested in (
            ("input_contract_attestation_sha256", payload.get("input_contract_attestation")),
            ("edge_semantics_attestation_sha256", payload.get("edge_semantics_attestation")),
        ):
            if not isinstance(nested, Mapping) or _require_sha(row.get(field), f"{token}.{field}") != _require_sha(nested.get(field), f"{token}.payload.{field}"):
                raise ValueError(f"{token}: {field} mismatch")
        novelty = build_certificate_novelty_table(
            case_token=token, replacement_probe=payload, frozen_control_probe=control
        )
        peak = int(row.get("peak_rss_bytes") or -1)
        if peak < 0 or peak >= PEAK_RSS_LIMIT_BYTES:
            raise ValueError(f"{token}: ordinary RSS reaches/exceeds 16 GiB")
        if float(row.get("wall_seconds") or -1.0) < 0.0:
            raise ValueError(f"{token}: ordinary wall time is invalid")
        if int(row.get("input_json_bytes") or -1) != pool_path.stat().st_size:
            raise ValueError(f"{token}: ordinary input byte measurement mismatch")
        if int(row.get("serialized_output_bytes") or -1) != replacement_path.stat().st_size:
            raise ValueError(f"{token}: ordinary output byte measurement mismatch")
        maximum_peak_rss = max(maximum_peak_rss, peak)
        compact_cases.append({
            "case_token": token,
            "status": "SUCCESS",
            "replacement_probe_sha256": str(row["replacement_probe_sha256"]),
            "frozen_control_probe_sha256": str(row["frozen_control_probe_sha256"]),
            "scan_sha256": str(row["scan_sha256"]),
            "candidate_pool_sha256": str(row["candidate_pool_sha256"]),
            "a0_prediction_sha256": str(row["a0_prediction_sha256"]),
            "input_table_hashes": deepcopy(row["input_table_hashes"]),
            "replacement_edge_table_hashes": deepcopy(row["replacement_edge_table_hashes"]),
            "output_table_hashes": deepcopy(row["output_table_hashes"]),
            "input_contract_attestation_sha256": str(row["input_contract_attestation_sha256"]),
            "edge_semantics_attestation_sha256": str(row["edge_semantics_attestation_sha256"]),
            "engineering_profile_sha256": str(row["engineering_profile_sha256"]),
            "inference_semantic_sha256": str(row["inference_semantic_sha256"]),
            "case_counters": deepcopy(row["case_counters"]),
            "wall_seconds": float(row["wall_seconds"]),
            "peak_rss_bytes": peak,
            "input_json_bytes": int(row["input_json_bytes"]),
            "serialized_output_bytes": int(row["serialized_output_bytes"]),
            "certificate_novelty_table_sha256": _sha_json(novelty),
            "certificate_novelty_table": novelty,
            "edge_deletion_counterfactual_sha256": _sha_json(counterfactual),
            "edge_deletion_counterfactual": deepcopy(counterfactual),
        })

    census, distribution = _control_census(control_probes)
    novel_exists = any(
        row["novel_edge_attributable"] is True
        for case in compact_cases for row in case["certificate_novelty_table"]
    )
    replay_max = max(
        int(case.get("maximum_peak_rss_bytes") or 0)
        for case in replay.get("cases") or ()
    )
    maximum_peak_rss = max(
        maximum_peak_rss, replay_max, synthetic_maximum_peak_rss
    )
    semantics = [dict(payload.get("edge_semantics_attestation") or {}) for payload in replacement_payloads.values()]
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    criteria.update({
        "frozen_lpf_control_census_reproduced": census == dict(expected_control_census) and distribution == dict(expected_control_distribution),
        "current_same_surface_edges_exact_subset": all(row.get("all_current_same_surface_edges_reproduced") is True for row in semantics),
        "current_same_surface_records_byte_identical": all(row.get("current_same_surface_records_byte_identical") is True for row in semantics),
        "inherited_cross_mode_edges_byte_identical": all(row.get("cross_mode_edges_byte_identical") is True for row in semantics),
        "added_edges_only_connect_distinct_neighbor_parents": all(row.get("all_added_edges_connect_distinct_parents") is True and row.get("all_added_edges_connect_26_neighbor_voxels") is True for row in semantics),
        "no_neighbor_degree_truncation": all(row.get("no_neighbor_degree_truncation") is True for row in semantics),
        "edge_deletion_counterfactual_valid": all_counterfactuals,
        "one_parent_components_remain_zero_span": all_one_parent_zero,
        "intervention_nonempty": aggregate_added_edges > 0,
        "two_sided_single_parent_support_extended": aggregate_two_sided_extensions > 0,
        "novel_edge_attributable_certificate_exists": novel_exists,
        "all_measured_peak_rss_below_16_gib": maximum_peak_rss < PEAK_RSS_LIMIT_BYTES,
    })
    lock = build_lock_attestation(criteria)
    validate_lock_attestation(lock)
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "profile": PRIMITIVE_PROFILE,
        "locked": lock["ready_for_gold"],
        "ready_for_gold": lock["ready_for_gold"],
        "research_only": True,
        "inference_safe": True,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "embedded_probe_payloads": False,
        "case_count": len(tokens),
        "successful_case_count": len(tokens),
        **expected_source_hashes,
        "probe_manifest_sha256": sha256_file(probe_manifest_path.resolve()),
        "replay_attestation_sha256": sha256_file(replay_attestation_path.resolve()),
        "synthetic_resource_attestation_sha256": sha256_file(
            synthetic_resource_attestation_path
        ),
        "probe_hash_table_sha256": _sha_json([[row["case_token"], row["replacement_probe_sha256"]] for row in compact_cases]),
        "certificate_novelty_table_sha256": _sha_json([[row["case_token"], row["certificate_novelty_table_sha256"]] for row in compact_cases]),
        "frozen_control_census": census,
        "frozen_control_certificate_distribution": distribution,
        "aggregate_added_same_surface_edge_count": aggregate_added_edges,
        "aggregate_two_sided_single_parent_support_extended_count": aggregate_two_sided_extensions,
        "maximum_peak_rss_bytes": maximum_peak_rss,
        "lock_attestation": lock,
        "lock_attestation_sha256": lock["lock_attestation_sha256"],
        "cases": compact_cases,
    }


def canonical_bundle_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(payload) + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--replay-attestation", required=True, type=Path)
    parser.add_argument(
        "--synthetic-resource-attestation", required=True, type=Path
    )
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument(
        "--workers", type=int, default=1,
        help=f"bounded isolated per-case rebuild workers (1-{MAX_BUNDLE_WORKERS})",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = build_bundle(
        probe_manifest_path=args.probe_manifest,
        probe_root=args.probe_root,
        replay_attestation_path=args.replay_attestation,
        synthetic_resource_attestation_path=args.synthetic_resource_attestation,
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        frozen_control_manifest_path=args.frozen_control_manifest,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bundle_bytes(bundle))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "bundle_sha256": sha256_file(args.output.resolve()),
        "lock_attestation_sha256": bundle["lock_attestation_sha256"],
        "ready_for_gold": bundle["ready_for_gold"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
