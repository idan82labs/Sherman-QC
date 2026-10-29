#!/usr/bin/env python3
"""Strict post-lock gold attribution for the V2 complete-edge research lane.

The compact bundle, its 39-key lock, every separately stored probe, the replay
attestation, and all frozen source byte hashes are verified before a truth-
bearing JSON file is parsed.  This command is research-only and never mutates
F1 predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
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
    _exact_line_matching,
    _integer_geometric_cost,
    _line,
    is_match,
    pair_metrics,
)
from scripts.build_f1_v2_complete_same_surface_edge_probe_bundle import (  # noqa: E402
    validate_synthetic_resource_attestation,
)


SCHEMA_VERSION = 1
KIND = "postlock_gold_v2_complete_same_surface_edge_attribution_v1"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "f1_v2_complete_same_surface_edge_probe_bundle_v1"
BUNDLE_PROFILE = "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "f1_v2_complete_same_surface_edge_probe_v1"
LOCK_ATTESTATION_KIND = (
    "f1_v2_complete_same_surface_edge_probe_lock_attestation_v1"
)
MATCHING_PROFILE = "V2_COMPLETE_EDGE_POSTLOCK_EXACT_MATCH_V1"

FROZEN_GROUPED_SPLIT_SHA256 = (
    "dbf1cb1f115780a546a72c296a962c37fdcd49602f8489eb4c5b0c9721c9cbfc"
)
FROZEN_PRIOR_ATTRIBUTION_SHA256 = (
    "dda757ce04962c8159ff218924b358e9526d7e56bd13e1845a47c1e61b738438"
)
FROZEN_GOLD_SHA256 = (
    "276fffd42db78629fe3930545f5a1e046fa45e64ea1eb9dc83bcc5916cca6f8b"
)
FROZEN_A0_PREDICTION_SHA256 = (
    "d543b970f3e2830f2bc2e976f490ffb7169eba9f3cef59c32323e1dfd7592fd8"
)
FROZEN_TRANSITION_CENSUS_SHA256 = (
    "dd0007286a2182cdfe1bf0c18219b54fb75226db5543b67caf20a3747ce52ac6"
)

EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
PRIMARY_STAGE1_COUNT = 67
FROZEN_A0_FALSE_PREDICTION_COUNT = 33
CONTROL_VALID_CERTIFICATE_COUNT = 31
MIN_RECOVERED_STAGE1_OBJECTS = 14
MIN_INDEPENDENT_GROUPS = 8
MAX_A0_FALSE_PREDICTIONS_CERTIFIED = 8

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

POSTLOCK_ROW_FIELDS = {
    "case_token",
    "split_group_id",
    "gold_id",
    "primary_population",
    "recovered",
    "reason_code",
    "replacement_certificate_id",
    "selected_component_id",
    "selected_geometry_gate_depth",
    "edge_semantics_dependent",
    "novel_edge_attributable",
    "causal_added_same_surface_edge_ids_sha256",
    "control_equivalent_certificate_ids",
    "integer_geometric_cost",
}

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
COUNTERFACTUAL_PROFILE = "DELETE_ADDED_SAME_SURFACE_EDGES_RESTORES_LPF_V2_V1"
COUNTERFACTUAL_FIELDS = {
    "counterfactual_profile",
    "restored_inherited_edge_table_sha256",
    "frozen_control_probe_sha256",
    "added_edge_count_deleted",
    "valid",
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(value: Any, field: str) -> str:
    result = str(value or "").lower()
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return result


def _index_cases(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        result[token] = dict(raw)
    return result


def _validate_bundle_lock(
    bundle: Mapping[str, Any],
    *,
    actual_bundle_sha256: str,
    expected_bundle_sha256: str,
    expected_lock_attestation_sha256: str,
    expected_case_tokens: Sequence[str] = EXPECTED_CASE_TOKENS,
) -> dict[str, bool]:
    """Validate external bundle hash, exact identity, and exact 39-key lock."""

    if _require_hash(actual_bundle_sha256, "bundle SHA") != _require_hash(
        expected_bundle_sha256, "expected bundle SHA"
    ):
        raise ValueError("replacement bundle SHA-256 mismatch")
    if (
        bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or bundle.get("kind") != BUNDLE_KIND
        or bundle.get("profile") != BUNDLE_PROFILE
    ):
        raise ValueError("replacement bundle identity mismatch")
    if bundle.get("locked") is not True or bundle.get("ready_for_gold") is not True:
        raise ValueError("replacement bundle is not ready for gold")
    rows = _index_cases(bundle, "replacement bundle")
    expected_tokens = set(expected_case_tokens)
    if set(rows) != expected_tokens:
        raise ValueError("replacement bundle case set mismatch")
    if int(bundle.get("case_count") or -1) != len(expected_tokens):
        raise ValueError("replacement bundle case count mismatch")
    if int(bundle.get("successful_case_count") or -1) != len(expected_tokens):
        raise ValueError("replacement bundle successful-case count mismatch")
    _require_hash(
        bundle.get("synthetic_resource_attestation_sha256"),
        "bundle synthetic resource attestation SHA",
    )
    lock = bundle.get("lock_attestation")
    if not isinstance(lock, Mapping) or set(lock) != {
        "kind", "criteria", "ready_for_gold", "lock_attestation_sha256"
    }:
        raise ValueError("replacement lock-attestation schema mismatch")
    if lock.get("kind") != LOCK_ATTESTATION_KIND:
        raise ValueError("replacement lock-attestation kind mismatch")
    criteria = lock.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(LOCK_CRITERION_KEYS):
        raise ValueError("replacement lock must contain the exact 39 criteria")
    if any(criteria[key] is not True for key in LOCK_CRITERION_KEYS):
        raise ValueError("replacement lock contains a failed criterion")
    if lock.get("ready_for_gold") is not True:
        raise ValueError("replacement lock does not authorize gold")
    lock_hash = _require_hash(lock.get("lock_attestation_sha256"), "lock SHA")
    recomputed = _sha256_json(
        {key: value for key, value in lock.items() if key != "lock_attestation_sha256"}
    )
    if lock_hash != recomputed:
        raise ValueError("replacement lock SHA does not recompute")
    if lock_hash != _require_hash(
        expected_lock_attestation_sha256, "expected replacement lock SHA"
    ):
        raise ValueError("replacement lock SHA differs from expected lock")
    if _require_hash(bundle.get("lock_attestation_sha256"), "bundle lock mirror") != lock_hash:
        raise ValueError("replacement bundle lock mirror mismatch")
    return {key: True for key in LOCK_CRITERION_KEYS}


def _core_api() -> tuple[Any, Any]:
    from backend.f1_v2_complete_same_surface_edge_probe import (
        canonical_json_bytes,
        validate_v2_complete_same_surface_edge_probe_payload,
    )

    return canonical_json_bytes, validate_v2_complete_same_surface_edge_probe_payload


def _validate_locked_probes(
    *,
    bundle: Mapping[str, Any],
    probe_root: Path,
    expected_case_tokens: Sequence[str] = EXPECTED_CASE_TOKENS,
) -> dict[str, dict[str, Any]]:
    """Validate all external probe files without consulting truth-bearing data."""

    canonical_bytes, validate_payload = _core_api()
    bundle_rows = _index_cases(bundle, "replacement bundle")
    probes: dict[str, dict[str, Any]] = {}
    observed_probe_hashes: list[list[str]] = []
    observed_novelty_hashes: list[list[str]] = []
    for token in sorted(expected_case_tokens):
        path = (probe_root / f"{token}.json").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{token}: missing locked replacement probe")
        row = bundle_rows[token]
        if _sha256(path) != _require_hash(
            row.get("replacement_probe_sha256"),
            f"{token}.replacement_probe_sha256",
        ):
            raise ValueError(f"{token}: locked replacement probe SHA changed")
        observed_probe_hashes.append([token, str(row["replacement_probe_sha256"])])
        payload = _read(path)
        validate_payload(payload)
        if path.read_bytes() != canonical_bytes(payload):
            raise ValueError(f"{token}: replacement probe is not canonical")
        if (
            payload.get("schema_version") != PROBE_SCHEMA_VERSION
            or payload.get("kind") != PROBE_KIND
            or payload.get("profile") != BUNDLE_PROFILE
        ):
            raise ValueError(f"{token}: replacement probe identity mismatch")
        for field in (
            "scan_sha256", "candidate_pool_sha256", "a0_prediction_sha256",
            "frozen_control_probe_sha256", "engineering_profile_sha256",
            "inference_semantic_sha256",
        ):
            if _require_hash(payload.get(field), f"{token}.probe.{field}") != _require_hash(
                row.get(field), f"{token}.bundle.{field}"
            ):
                raise ValueError(f"{token}: bundle/probe {field} mismatch")
        for field in (
            "input_table_hashes", "replacement_edge_table_hashes",
            "output_table_hashes", "case_counters",
        ):
            if dict(row.get(field) or {}) != dict(payload.get(field) or {}):
                raise ValueError(f"{token}: bundle/probe {field} mismatch")
        for field, nested_field in (
            ("input_contract_attestation_sha256", "input_contract_attestation"),
            ("edge_semantics_attestation_sha256", "edge_semantics_attestation"),
        ):
            nested = payload.get(nested_field)
            if not isinstance(nested, Mapping) or _require_hash(
                row.get(field), f"{token}.bundle.{field}"
            ) != _require_hash(nested.get(field), f"{token}.probe.{field}"):
                raise ValueError(f"{token}: bundle/probe {field} mismatch")
        novelty_rows = row.get("certificate_novelty_table")
        if not isinstance(novelty_rows, Sequence) or isinstance(novelty_rows, (str, bytes)):
            raise ValueError(f"{token}: compact novelty table must be a sequence")
        novelty_sha = _sha256_json(list(novelty_rows))
        if novelty_sha != _require_hash(
            row.get("certificate_novelty_table_sha256"),
            f"{token}.certificate_novelty_table_sha256",
        ):
            raise ValueError(f"{token}: compact novelty table SHA mismatch")
        observed_novelty_hashes.append([token, novelty_sha])
        counterfactual = row.get("edge_deletion_counterfactual")
        if (
            not isinstance(counterfactual, Mapping)
            or set(counterfactual) != COUNTERFACTUAL_FIELDS
            or counterfactual.get("counterfactual_profile") != COUNTERFACTUAL_PROFILE
            or counterfactual.get("valid") is not True
            or _sha256_json(counterfactual) != _require_hash(
            row.get("edge_deletion_counterfactual_sha256"),
            f"{token}.edge_deletion_counterfactual_sha256",
            )
        ):
            raise ValueError(f"{token}: compact counterfactual attestation mismatch")
        probes[token] = payload
    if _require_hash(bundle.get("probe_hash_table_sha256"), "probe hash table SHA") != _sha256_json(observed_probe_hashes):
        raise ValueError("bundle probe hash table SHA mismatch")
    if _require_hash(bundle.get("certificate_novelty_table_sha256"), "novelty hash table SHA") != _sha256_json(observed_novelty_hashes):
        raise ValueError("bundle novelty hash table SHA mismatch")
    return probes


def _validate_synthetic_resource_reference(
    *, bundle: Mapping[str, Any], attestation_path: Path
) -> str:
    """Verify the external worst-cap artifact against its locked bundle SHA."""

    path = attestation_path.resolve()
    actual = _sha256(path)
    if actual != _require_hash(
        bundle.get("synthetic_resource_attestation_sha256"),
        "bundle synthetic resource attestation SHA",
    ):
        raise ValueError("synthetic resource attestation SHA differs from locked bundle")
    validate_synthetic_resource_attestation(_read(path))
    return actual


def _component_records(
    probe: Mapping[str, Any], novelty_rows: Sequence[Mapping[str, Any]], *, case_token: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    novelty: dict[str, dict[str, Any]] = {}
    for raw in novelty_rows:
        if not isinstance(raw, Mapping) or set(raw) != CERTIFICATE_NOVELTY_FIELDS:
            raise ValueError("certificate novelty row schema mismatch")
        row = dict(raw)
        identifier = str(row.get("replacement_certificate_id") or "")
        equivalents = row.get("control_equivalent_certificate_ids")
        if (
            not identifier
            or identifier in novelty
            or not isinstance(equivalents, Sequence)
            or isinstance(equivalents, (str, bytes))
            or list(equivalents) != sorted(set(str(value) for value in equivalents))
        ):
            raise ValueError("certificate novelty IDs/equivalents are invalid")
        if _require_hash(
            row.get("control_equivalent_certificate_ids_sha256"),
            "control-equivalent IDs SHA",
        ) != _sha256_json(list(equivalents)):
            raise ValueError("control-equivalent IDs SHA mismatch")
        if _require_hash(
            row.get("certificate_novelty_sha256"), "certificate novelty SHA"
        ) != _sha256_json(
            {key: value for key, value in row.items() if key != "certificate_novelty_sha256"}
        ):
            raise ValueError("certificate novelty SHA mismatch")
        expected_novel = bool(
            row.get("edge_semantics_dependent") is True and not equivalents
        )
        if row.get("novel_edge_attributable") is not expected_novel:
            raise ValueError("certificate novelty causal predicate mismatch")
        novelty[identifier] = row
    if "" in novelty or len(novelty) != len(novelty_rows):
        raise ValueError("certificate novelty IDs must be nonempty and unique")
    attributions = {
        str(row.get("local_transition_component_id") or ""): dict(row)
        for row in probe.get("component_edge_attribution_table") or ()
        if isinstance(row, Mapping)
    }
    components: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_pair in probe.get("local_plane_pair_records") or ():
        if not isinstance(raw_pair, Mapping):
            raise ValueError("local plane-pair rows must be objects")
        for raw in raw_pair.get("components") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("local component rows must be objects")
            component = deepcopy(dict(raw))
            identifier = str(component.get("local_transition_component_id") or "")
            if not identifier or identifier in seen or identifier not in attributions:
                raise ValueError("component IDs must be unique and attributed")
            seen.add(identifier)
            component.update(attributions[identifier])
            try:
                component["_line"] = _line(component, identifier=identifier)
            except ValueError as exc:
                if "line span must be nonzero" not in str(exc):
                    raise
                component["_line"] = None
            if component.get("geometry_certificate_valid") is True:
                if component["_line"] is None or identifier not in novelty:
                    raise ValueError("valid certificate lacks line or novelty record")
                nrow = novelty[identifier]
                if nrow.get("case_token") != case_token:
                    raise ValueError("certificate novelty case token mismatch")
                if nrow.get("edge_semantics_dependent") is not (
                    component.get("edge_semantics_dependent") is True
                ):
                    raise ValueError("certificate novelty causal flag mismatch")
                if _require_hash(
                    nrow.get("causal_added_same_surface_edge_ids_sha256"),
                    "novelty causal edge IDs SHA",
                ) != _require_hash(
                    component.get("causal_added_same_surface_edge_ids_sha256"),
                    "component causal edge IDs SHA",
                ):
                    raise ValueError("certificate novelty causal edge hash mismatch")
                component.update(nrow)
                valid.append(
                    {
                        **dict(component["_line"]),
                        "certificate_id": identifier,
                        **nrow,
                    }
                )
            components.append(component)
    if set(novelty) != {row["certificate_id"] for row in valid}:
        raise ValueError("novelty table must cover exactly all valid certificates")
    return components, valid


def _first_geometry_failure(
    gold: Mapping[str, Any], components: Sequence[Mapping[str, Any]], thresholds: Thresholds
) -> tuple[str, Mapping[str, Any] | None, int | None]:
    gold_line = _line(gold, identifier=str(gold.get("bend_id") or "gold"))
    candidates: list[tuple[int, int, str, Mapping[str, Any]]] = []
    for component in components:
        line = component.get("_line")
        if not isinstance(line, Mapping):
            continue
        metrics = pair_metrics(_endpoints(gold_line), _endpoints(line))
        if is_match(metrics, thresholds):
            candidates.append(
                (
                    -int(component.get("geometry_gate_depth") or 0),
                    _integer_geometric_cost(metrics),
                    str(component.get("local_transition_component_id") or ""),
                    component,
                )
            )
    if not candidates:
        return "NO_MATCHING_REPLACEMENT_COMPONENT", None, None
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    selected = candidates[0]
    component = selected[3]
    if component.get("geometry_certificate_valid") is True:
        if component.get("edge_semantics_dependent") is not True:
            reason = "MATCHED_CERTIFICATE_NOT_EDGE_DEPENDENT"
        elif component.get("control_equivalent_certificate_ids"):
            reason = "MATCHED_CONTROL_EQUIVALENT_CERTIFICATE"
        else:
            # A novel valid certificate may already be assigned to a different
            # gold row by the exact one-to-one solution.  It cannot be reused.
            reason = "NO_MATCHING_REPLACEMENT_COMPONENT"
    else:
        reason = str(component.get("first_rejection_code") or "NO_MATCHING_REPLACEMENT_COMPONENT")
    return reason, component, selected[1]


def _make_row(
    *, token: str, group_id: str, gold_id: str, recovered: bool,
    reason: str, certificate: Mapping[str, Any] | None, cost: int | None,
) -> dict[str, Any]:
    identifier = None if certificate is None else str(
        certificate.get("certificate_id")
        or certificate.get("local_transition_component_id")
        or ""
    )
    row = {
        "case_token": token,
        "split_group_id": group_id,
        "gold_id": gold_id,
        "primary_population": True,
        "recovered": recovered,
        "reason_code": reason,
        "replacement_certificate_id": identifier if recovered else None,
        "selected_component_id": identifier,
        "selected_geometry_gate_depth": None if certificate is None else int(
            certificate.get("geometry_gate_depth") or 0
        ),
        "edge_semantics_dependent": None if certificate is None else (
            certificate.get("edge_semantics_dependent") is True
        ),
        "novel_edge_attributable": None if certificate is None else (
            certificate.get("novel_edge_attributable") is True
        ),
        "causal_added_same_surface_edge_ids_sha256": None if certificate is None else (
            certificate.get("causal_added_same_surface_edge_ids_sha256")
        ),
        "control_equivalent_certificate_ids": [] if certificate is None else list(
            certificate.get("control_equivalent_certificate_ids") or ()
        ),
        "integer_geometric_cost": cost,
    }
    if set(row) != POSTLOCK_ROW_FIELDS:
        raise AssertionError("post-lock row schema drift")
    return row


def attribute_gold(
    *,
    bundle: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
    prior_attribution: Mapping[str, Any],
    grouped_split: Mapping[str, Any],
    sealed_gold: Mapping[str, Any],
    a0_predictions: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    expected_primary_count: int = PRIMARY_STAGE1_COUNT,
    enforce_frozen_false_count: bool = True,
) -> dict[str, Any]:
    """Perform exact causal attribution after the caller has opened the lock."""

    bundle_cases = _index_cases(bundle, "replacement bundle")
    split_cases = _index_cases(grouped_split, "grouped split")
    gold_cases = _index_cases(sealed_gold, "sealed gold")
    prediction_cases = _index_cases(a0_predictions, "frozen A0 prediction")
    tokens = set(bundle_cases)
    if tokens != set(probes) or tokens != set(split_cases) or tokens != set(gold_cases) or tokens != set(prediction_cases):
        raise ValueError("all post-lock case sets must match")
    prior_rows = prior_attribution.get("rows")
    if not isinstance(prior_rows, Sequence) or isinstance(prior_rows, (str, bytes)):
        raise ValueError("prior attribution rows must be a sequence")
    primary: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in prior_rows:
        if not isinstance(raw, Mapping) or int(raw.get("first_failed_stage") or 0) != 1:
            continue
        key = (str(raw.get("case_token") or ""), str(raw.get("gold_id") or ""))
        if not all(key) or key in primary:
            raise ValueError("frozen primary row keys must be nonempty and unique")
        primary[key] = dict(raw)
    if len(primary) != expected_primary_count:
        raise ValueError(f"frozen stage-1 population must contain {expected_primary_count} rows")

    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    rows: list[dict[str, Any]] = []
    novel_matches: list[dict[str, Any]] = []
    all_replacement_matches: list[dict[str, Any]] = []
    false_matches: list[dict[str, Any]] = []
    recovered_groups: set[str] = set()
    valid_count = novel_count = observed_false_count = 0
    control_equivalent_recovery_count = 0

    for token in sorted(tokens):
        bundle_row = bundle_cases[token]
        novelty_rows = bundle_row.get("certificate_novelty_table") or ()
        if not isinstance(novelty_rows, Sequence) or isinstance(novelty_rows, (str, bytes)):
            raise ValueError(f"{token}: certificate novelty table must be a sequence")
        components, valid_certificates = _component_records(
            probes[token], novelty_rows, case_token=token
        )
        novel_certificates = [
            row for row in valid_certificates if row.get("novel_edge_attributable") is True
        ]
        valid_count += len(valid_certificates)
        novel_count += len(novel_certificates)

        split = split_cases[token]
        group_id = str(split.get("split_group_id") or split.get("part_key") or split.get("part_family") or "")
        if not group_id:
            raise ValueError(f"{token}: missing independent split group")
        prediction_file_hash = _require_hash(
            prediction_cases[token].get("prediction_owned_regions_sha256"),
            f"{token}.prediction_owned_regions_sha256",
        )
        if prediction_file_hash != _require_hash(
            bundle_row.get("a0_prediction_sha256"),
            f"{token}.bundle.a0_prediction_sha256",
        ):
            raise ValueError(f"{token}: frozen A0 prediction hash changed")
        if _require_hash(
            gold_cases[token].get("source_scan_sha256"),
            f"{token}.gold.source_scan_sha256",
        ) != _require_hash(bundle_row.get("scan_sha256"), f"{token}.bundle.scan_sha256"):
            raise ValueError(f"{token}: sealed-gold scan hash mismatch")
        gold_records = [dict(row) for row in gold_cases[token].get("manual_bends") or () if isinstance(row, Mapping)]
        gold_by_id = {str(row.get("bend_id") or ""): row for row in gold_records}
        predictions = [dict(row) for row in prediction_cases[token].get("a0_predictions") or () if isinstance(row, Mapping)]
        a0_gold = _exact_line_matching(
            gold_records, predictions, left_id_field="bend_id", right_id_field="bend_id",
            thresholds=thresholds,
        )
        false_ids = set(a0_gold["unmatched_right_ids"])
        frozen_false = [row for row in predictions if str(row.get("bend_id")) in false_ids]
        observed_false_count += len(frozen_false)

        case_primary_rows = [row for (case_token, _), row in primary.items() if case_token == token]
        case_primary_gold: list[dict[str, Any]] = []
        for prior_row in sorted(case_primary_rows, key=lambda row: str(row.get("gold_id"))):
            gold_id = str(prior_row.get("gold_id") or "")
            if gold_id not in gold_by_id or gold_id not in set(a0_gold["unmatched_left_ids"]):
                raise ValueError(f"{token}: frozen primary gold row changed")
            case_primary_gold.append(gold_by_id[gold_id])

        causal = _exact_line_matching(
            case_primary_gold, novel_certificates,
            left_id_field="bend_id", right_id_field="certificate_id", thresholds=thresholds,
        )
        all_result = _exact_line_matching(
            case_primary_gold, valid_certificates,
            left_id_field="bend_id", right_id_field="certificate_id", thresholds=thresholds,
        )
        causal_by_gold = {str(row["left_id"]): row for row in causal["matches"]}
        valid_by_id = {str(row["certificate_id"]): row for row in valid_certificates}
        for match in causal["matches"]:
            cert = valid_by_id[str(match["right_id"])]
            recovered_groups.add(group_id)
            novel_matches.append({
                "case_token": token, "split_group_id": group_id,
                "gold_id": match["left_id"], "replacement_certificate_id": match["right_id"],
                "integer_geometric_cost": match["integer_geometric_cost"],
            })
        for match in all_result["matches"]:
            cert = valid_by_id[str(match["right_id"])]
            if cert.get("control_equivalent_certificate_ids"):
                control_equivalent_recovery_count += 1
            all_replacement_matches.append({
                "case_token": token, "gold_id": match["left_id"],
                "replacement_certificate_id": match["right_id"],
                "integer_geometric_cost": match["integer_geometric_cost"],
            })

        false_result = _exact_line_matching(
            frozen_false, valid_certificates,
            left_id_field="bend_id", right_id_field="certificate_id", thresholds=thresholds,
        )
        false_matches.extend({
            "case_token": token, "a0_false_prediction_id": match["left_id"],
            "replacement_certificate_id": match["right_id"],
            "integer_geometric_cost": match["integer_geometric_cost"],
        } for match in false_result["matches"])

        for gold in sorted(case_primary_gold, key=lambda row: str(row.get("bend_id"))):
            gold_id = str(gold["bend_id"])
            match = causal_by_gold.get(gold_id)
            if match is not None:
                cert = valid_by_id[str(match["right_id"])]
                rows.append(_make_row(
                    token=token, group_id=group_id, gold_id=gold_id, recovered=True,
                    reason="VALID_NOVEL_EDGE_ATTRIBUTABLE_CERTIFICATE", certificate=cert,
                    cost=int(match["integer_geometric_cost"]),
                ))
            else:
                reason, selected, cost = _first_geometry_failure(gold, components, thresholds)
                rows.append(_make_row(
                    token=token, group_id=group_id, gold_id=gold_id, recovered=False,
                    reason=reason, certificate=selected, cost=cost,
                ))

    if enforce_frozen_false_count and observed_false_count != FROZEN_A0_FALSE_PREDICTION_COUNT:
        raise ValueError("frozen A0 false-prediction count changed")
    rows.sort(key=lambda row: (row["case_token"], row["gold_id"]))
    novel_matches.sort(key=lambda row: (row["case_token"], row["gold_id"]))
    false_matches.sort(key=lambda row: (row["case_token"], row["a0_false_prediction_id"]))
    failure_counts = Counter(row["reason_code"] for row in rows if not row["recovered"])
    lock_ready = bundle.get("ready_for_gold") is True
    go = bool(
        lock_ready
        and len(novel_matches) >= MIN_RECOVERED_STAGE1_OBJECTS
        and len(recovered_groups) >= MIN_INDEPENDENT_GROUPS
        and len(false_matches) <= MAX_A0_FALSE_PREDICTIONS_CERTIFIED
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "research_only": True,
        "postlock": True,
        "primary_population": "frozen_transition_stage_1_only",
        "matching_profile": MATCHING_PROFILE,
        **dict(source_hashes),
        "line_matching_contract": {
            "axis_angle_deg": 5.0,
            "transverse_offset_mm": 4.0,
            "gold_span_overlap": 0.60,
            "certificate_span_overlap": 0.60,
            "stages": ["maximum_cardinality", "minimum_total_integer_geometric_cost", "lexicographically_smallest_ordered_assignment"],
            "solver": {"engine": "OR_TOOLS_CP_SAT", "num_search_workers": 1, "random_seed": 0, "search_branching": "FIXED_SEARCH", "wall_clock_cutoff": None},
        },
        "summary": {
            "primary_stage1_count": len(primary),
            "replacement_recovered_stage1_count": len(all_replacement_matches),
            "edge_attributable_recovered_stage1_count": len(novel_matches),
            "control_recovered_stage1_count": 0,
            "independent_edge_attributable_group_count": len(recovered_groups),
            "replacement_valid_certificate_count": valid_count,
            "control_valid_certificate_count": CONTROL_VALID_CERTIFICATE_COUNT,
            "novel_edge_attributable_certificate_count": novel_count,
            "control_equivalent_recovery_count": control_equivalent_recovery_count,
            "observed_frozen_a0_false_prediction_count": observed_false_count,
            "a0_false_predictions_certified": len(false_matches),
            "new_failure_reason_counts": dict(sorted(failure_counts.items())),
        },
        "professor_probe_gate": {
            "thresholds": {"minimum_edge_attributable_recoveries": 14, "minimum_independent_groups": 8, "maximum_a0_false_predictions_certified": 8},
            "criteria": {
                "lock_ready_for_gold": lock_ready,
                "edge_attributable_recoveries_at_least_14": len(novel_matches) >= 14,
                "independent_groups_at_least_8": len(recovered_groups) >= 8,
                "a0_false_predictions_certified_at_most_8": len(false_matches) <= 8,
            },
            "passed": go,
            "decision": "GO" if go else "STOP",
            "authorization": "PREPARE_BOUNDED_PROFESSOR_RESEARCH_PACKET_ONLY" if go else "EDGE_HYPOTHESIS_FALSIFIED_OR_THRESHOLD_NOT_MET",
        },
        "replacement_recovery_matches": all_replacement_matches,
        "edge_attributable_recovery_matches": novel_matches,
        "a0_false_certificate_matches": false_matches,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-lock-attestation-sha256", required=True)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--replay-attestation", required=True, type=Path)
    parser.add_argument(
        "--synthetic-resource-attestation", required=True, type=Path
    )
    parser.add_argument("--prior-attribution", required=True, type=Path)
    parser.add_argument("--grouped-split", required=True, type=Path)
    parser.add_argument("--sealed-gold", required=True, type=Path)
    parser.add_argument("--a0-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # Mandatory opening order.  The four truth-bearing files are not parsed
    # until bundle, lock, all 37 probes, replay, and frozen byte hashes pass.
    bundle_path = args.bundle.resolve()
    bundle = _read(bundle_path)
    bundle_sha = _sha256(bundle_path)
    _validate_bundle_lock(
        bundle,
        actual_bundle_sha256=bundle_sha,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_lock_attestation_sha256=args.expected_lock_attestation_sha256,
    )
    probes = _validate_locked_probes(bundle=bundle, probe_root=args.probe_root.resolve())
    replay_sha = _sha256(args.replay_attestation.resolve())
    if replay_sha != _require_hash(bundle.get("replay_attestation_sha256"), "replay attestation SHA"):
        raise ValueError("replay-attestation SHA differs from locked bundle")
    synthetic_resource_sha = _validate_synthetic_resource_reference(
        bundle=bundle,
        attestation_path=args.synthetic_resource_attestation,
    )

    protected = {
        "prior_attribution_sha256": (args.prior_attribution.resolve(), FROZEN_PRIOR_ATTRIBUTION_SHA256),
        "grouped_split_sha256": (args.grouped_split.resolve(), FROZEN_GROUPED_SPLIT_SHA256),
        "a0_prediction_sha256": (args.a0_predictions.resolve(), FROZEN_A0_PREDICTION_SHA256),
        "gold_sha256": (args.sealed_gold.resolve(), FROZEN_GOLD_SHA256),
    }
    source_hashes = {
        "replacement_bundle_sha256": bundle_sha,
        "lock_attestation_sha256": str(bundle["lock_attestation_sha256"]),
        "replay_attestation_sha256": replay_sha,
        "synthetic_resource_attestation_sha256": synthetic_resource_sha,
    }
    # Hash non-gold sources first and the sealed gold last.  No JSON is opened.
    for field in ("prior_attribution_sha256", "grouped_split_sha256", "a0_prediction_sha256", "gold_sha256"):
        path, expected = protected[field]
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{field} differs from preregistered frozen value")
        source_hashes[field] = actual

    prior = _read(protected["prior_attribution_sha256"][0])
    if (
        prior.get("kind") != "postlock_gold_transition_failure_attribution"
        or _require_hash(prior.get("diagnostic_census_sha256"), "prior census SHA") != FROZEN_TRANSITION_CENSUS_SHA256
    ):
        raise ValueError("prior attribution identity changed")
    for field, expected in (
        ("prediction_sha256", FROZEN_A0_PREDICTION_SHA256),
        ("gold_sha256", FROZEN_GOLD_SHA256),
        ("grouped_split_sha256", FROZEN_GROUPED_SPLIT_SHA256),
    ):
        if _require_hash(prior.get(field), f"prior.{field}") != expected:
            raise ValueError(f"prior attribution {field} changed")
    split = _read(protected["grouped_split_sha256"][0])
    predictions = _read(protected["a0_prediction_sha256"][0])
    gold = _read(protected["gold_sha256"][0])
    result = attribute_gold(
        bundle=bundle, probes=probes, prior_attribution=prior,
        grouped_split=split, sealed_gold=gold, a0_predictions=predictions,
        source_hashes=source_hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "decision": result["professor_probe_gate"]["decision"],
        **result["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
