#!/usr/bin/env python3
"""Recompute the exact gold-free causal funnel for complete-edge V2 probes.

This audit consumes only the locked inference bundle, frozen candidate pools,
frozen LPF controls, and separately stored replacement probes.  It reconstructs
full component memberships instead of relying on bounded witness arrays and
never opens gold, ownership, labels, or prediction geometry.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import f1_local_plane_fragment_probe as lpf  # noqa: E402
from backend import f1_v2_complete_same_surface_edge_probe as edge  # noqa: E402
from scripts.build_f1_v2_complete_same_surface_edge_probe_bundle import (  # noqa: E402
    BUNDLE_KIND,
    BUNDLE_SCHEMA_VERSION,
    PRIMITIVE_PROFILE,
)


SCHEMA_VERSION = 1
KIND = "f1_v2_complete_edge_gold_free_causal_funnel_v1"
PROFILE = "V2_COMPLETE_EDGE_CAUSAL_FUNNEL_AUDIT_V1"
EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
EXPECTED_EXTENSION_COUNT = 320
MAX_WORKERS = 18
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

ROW_FIELDS = {
    "case_token",
    "replacement_component_id",
    "extended_frozen_target_set_count",
    "extended_frozen_target_sets_sha256",
    "support_parent_count",
    "support_mode_count",
    "causal_added_same_surface_edge_count",
    "causal_added_same_surface_edge_ids_sha256",
    "geometry_gate_depth",
    "first_rejection_code",
    "geometry_certificate_valid",
    "control_equivalent_certificate_ids",
    "causal_funnel_row_sha256",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _record_hash(record: Mapping[str, Any], field: str) -> str:
    return _sha_json({key: value for key, value in record.items() if key != field})


def _index_cases(
    payload: Mapping[str, Any], label: str, *, expected_tokens: Sequence[str]
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
    if set(result) != set(expected_tokens):
        raise ValueError(f"{label} case set mismatch")
    return result


def _frozen_targets(control: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for pair in control.get("local_plane_pair_records") or ():
        if not isinstance(pair, Mapping):
            raise ValueError("frozen control plane-pair rows must be objects")
        for component in pair.get("components") or ():
            if not isinstance(component, Mapping):
                raise ValueError("frozen control component rows must be objects")
            endpoints = component.get("intersection_span_endpoints")
            if (
                int(component.get("support_parent_count") or 0) == 1
                and int(component.get("support_mode_count") or 0) > 1
                and int(component.get("left_side_bin_count") or 0) > 0
                and int(component.get("right_side_bin_count") or 0) > 0
                and int(component.get("transition_rung_bin_count") or 0) > 0
                and isinstance(endpoints, Sequence)
                and len(endpoints) == 2
                and endpoints[0] == endpoints[1]
            ):
                if component.get("witness_ids_truncated") is True:
                    raise ValueError("frozen one-parent target witness is truncated")
                modes = tuple(sorted(str(value) for value in component.get("witness_mode_ids") or ()))
                if len(modes) != int(component["support_mode_count"]):
                    raise ValueError("frozen target mode witness is incomplete")
                target = {
                    "frozen_control_component_id": str(
                        component["local_transition_component_id"]
                    ),
                    "member_mode_ids_sha256": _sha_json(list(modes)),
                    "mode_ids": modes,
                }
                targets.append(target)
    return sorted(
        targets,
        key=lambda row: (
            row["frozen_control_component_id"], row["member_mode_ids_sha256"]
        ),
    )


def _full_component_memberships(
    projected_input: Mapping[str, Any], replacement_probe: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    """Re-enumerate full mode memberships and prove IDs match the locked probe."""

    validated = edge._validate_projected_input(projected_input)
    complete, _current_same, _current_cross, _added, effective, _adjacent, _evaluations = (
        edge._edge_tables(projected_input, validated)
    )
    del complete
    parents = validated["parent_by_id"]
    modes = validated["mode_by_id"]
    planes = validated["planes"]
    spacing = float(validated["spacing"])
    plane_tolerance = min(3.0, max(1.25, 1.5 * spacing))
    line_tolerance = min(3.0, max(1.5, 1.25 * spacing))
    counters: Counter[str] = Counter()
    fragments, _orphans, fragment_of_mode, _orphan_modes = edge._profile_fragments(
        parents, modes, effective, planes, plane_tolerance, counters
    )
    fragment_by_id = {
        fragment.record["local_plane_fragment_id"]: fragment
        for fragment in fragments
    }
    pair_seeds: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    same_by_fragment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cross_by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in effective:
        left_mode, right_mode = map(str, record["mode_ids"])
        if left_mode not in fragment_of_mode or right_mode not in fragment_of_mode:
            continue
        left_fragment = fragment_of_mode[left_mode]
        right_fragment = fragment_of_mode[right_mode]
        if record["edge_kind"] == "SAME_SURFACE" and left_fragment == right_fragment:
            same_by_fragment[left_fragment].append(record)
        elif record["edge_kind"] == "CROSS_MODE_LOCAL" and left_fragment != right_fragment:
            cross_by_pair[tuple(sorted((left_fragment, right_fragment)))].append(record)
    for record in sorted(
        (row for row in effective if row["edge_kind"] == "CROSS_MODE_LOCAL"),
        key=lambda row: row["mode_edge_id"],
    ):
        left_mode, right_mode = map(str, record["mode_ids"])
        if left_mode not in fragment_of_mode or right_mode not in fragment_of_mode:
            continue
        pair = tuple(
            sorted((fragment_of_mode[left_mode], fragment_of_mode[right_mode]))
        )
        if pair[0] != pair[1]:
            pair_seeds[pair].append(record)

    memberships: dict[str, tuple[str, ...]] = {}
    for pair_ids in sorted(pair_seeds):
        left, right = (fragment_by_id[identifier] for identifier in pair_ids)
        pair_id = "VEP_" + edge._sha([list(pair_ids)])[:16]
        try:
            anchor, axis, _angle = lpf._plane_intersection(
                left.normal, left.offset, right.normal, right.offset
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        transverse = {
            parent_id
            for parent_id, parent in parents.items()
            if lpf._line_distance(
                lpf._vector(parent["centroid"], "parent centroid", allow_zero=True),
                anchor,
                axis,
            )
            <= line_tolerance + 1e-12
        }
        left_modes = {
            mode_id
            for mode_id in left.mode_ids
            if str(modes[mode_id]["parent_atom_id"]) in transverse
        }
        right_modes = {
            mode_id
            for mode_id in right.mode_ids
            if str(modes[mode_id]["parent_atom_id"]) in transverse
        }
        supported = left_modes | right_modes
        restricted: list[Mapping[str, Any]] = []
        candidate_edges = [
            *same_by_fragment.get(pair_ids[0], ()),
            *same_by_fragment.get(pair_ids[1], ()),
            *cross_by_pair.get(pair_ids, ()),
        ]
        for record in candidate_edges:
            mode_ids = tuple(map(str, record["mode_ids"]))
            if not set(mode_ids) <= supported:
                continue
            if record["edge_kind"] == "SAME_SURFACE":
                fragment_id = fragment_of_mode[mode_ids[0]]
                if (
                    fragment_id != fragment_of_mode[mode_ids[1]]
                    or fragment_id not in pair_ids
                ):
                    continue
            elif record["edge_kind"] == "CROSS_MODE_LOCAL":
                if {
                    fragment_of_mode[mode_ids[0]], fragment_of_mode[mode_ids[1]]
                } != set(pair_ids):
                    continue
            else:
                continue
            restricted.append(record)
        for component_modes in lpf._connected_components(supported, restricted):
            full_modes = tuple(sorted(map(str, component_modes)))
            parent_ids = tuple(
                sorted({str(modes[mode_id]["parent_atom_id"]) for mode_id in full_modes})
            )
            component_id = "VEC_" + edge._sha(
                [pair_id, list(parent_ids), list(full_modes)]
            )[:16]
            if component_id in memberships:
                raise ValueError("replacement component ID is not unique")
            memberships[component_id] = full_modes

    locked_ids = {
        str(component["local_transition_component_id"])
        for pair in replacement_probe.get("local_plane_pair_records") or ()
        for component in pair.get("components") or ()
    }
    if set(memberships) != locked_ids:
        raise ValueError("recomputed full component membership does not match locked probe")
    return memberships


def _replacement_components(
    probe: Mapping[str, Any], bundle_case: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    attributions = {
        str(row["local_transition_component_id"]): dict(row)
        for row in probe.get("component_edge_attribution_table") or ()
    }
    novelty = {
        str(row["replacement_certificate_id"]): dict(row)
        for row in bundle_case.get("certificate_novelty_table") or ()
    }
    result: dict[str, dict[str, Any]] = {}
    for pair in probe.get("local_plane_pair_records") or ():
        for raw in pair.get("components") or ():
            component = dict(raw)
            identifier = str(component.get("local_transition_component_id") or "")
            if not identifier or identifier in result or identifier not in attributions:
                raise ValueError("replacement component/attribution identity mismatch")
            component["_attribution"] = attributions[identifier]
            component["_novelty"] = novelty.get(identifier)
            result[identifier] = component
    if set(attributions) != set(result):
        raise ValueError("component attribution table does not cover all components")
    valid_ids = {
        identifier
        for identifier, component in result.items()
        if component.get("geometry_certificate_valid") is True
    }
    if set(novelty) != valid_ids:
        raise ValueError("certificate novelty table does not cover valid components")
    return result


def _audit_case(
    *,
    token: str,
    bundle_case: Mapping[str, Any],
    probe_path: Path,
    pool_path: Path,
    control_path: Path,
) -> dict[str, Any]:
    if _sha_file(probe_path) != _require_sha(
        bundle_case.get("replacement_probe_sha256"),
        f"{token}.replacement_probe_sha256",
    ):
        raise ValueError(f"{token}: replacement probe SHA changed")
    probe = _read(probe_path)
    edge.validate_v2_complete_same_surface_edge_probe_payload(probe)
    if _sha_file(pool_path) != _require_sha(
        bundle_case.get("candidate_pool_sha256"),
        f"{token}.candidate_pool_sha256",
    ):
        raise ValueError(f"{token}: candidate-pool SHA changed")
    projected = edge.project_edge_probe_input(_read(pool_path))
    edge.validate_v2_complete_same_surface_edge_probe_payload(
        probe, projected_input=projected
    )
    if _sha_file(control_path) != _require_sha(
        bundle_case.get("frozen_control_probe_sha256"),
        f"{token}.frozen_control_probe_sha256",
    ):
        raise ValueError(f"{token}: frozen control SHA changed")
    targets = _frozen_targets(_read(control_path))
    memberships = _full_component_memberships(projected, probe)
    components = _replacement_components(probe, bundle_case)
    causal_components = [
        component
        for component in components.values()
        if component["_attribution"].get("edge_semantics_dependent") is True
    ]
    causal_component_outcomes: Counter[str] = Counter()
    for component in causal_components:
        novelty = component.get("_novelty") or {}
        if component.get("geometry_certificate_valid") is True:
            outcome = (
                "VALID_CONTROL_EQUIVALENT_CERTIFICATE"
                if novelty.get("control_equivalent_certificate_ids")
                else "VALID_NOVEL_CERTIFICATE"
            )
        else:
            outcome = str(component.get("first_rejection_code") or "")
        if not outcome:
            raise ValueError("causal component lacks a funnel outcome")
        causal_component_outcomes[outcome] += 1
    edge_dependent_valid = sum(
        component.get("geometry_certificate_valid") is True
        for component in causal_components
    )
    control_equivalent_edge_dependent_valid = sum(
        component.get("geometry_certificate_valid") is True
        and bool(
            (component.get("_novelty") or {}).get(
                "control_equivalent_certificate_ids"
            )
        )
        for component in causal_components
    )
    rows: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for component_id in sorted(components):
        component = components[component_id]
        attribution = component["_attribution"]
        full_modes = set(memberships[component_id])
        matched_targets = [
            target for target in targets if set(target["mode_ids"]) <= full_modes
        ]
        if (
            int(component.get("support_parent_count") or 0) < 2
            or attribution.get("edge_semantics_dependent") is not True
            or not matched_targets
        ):
            continue
        target_commitments = sorted(
            [
                target["frozen_control_component_id"],
                target["member_mode_ids_sha256"],
            ]
            for target in matched_targets
        )
        novelty = component.get("_novelty") or {}
        reason = (
            None
            if component.get("geometry_certificate_valid") is True
            else str(component.get("first_rejection_code") or "")
        )
        if component.get("geometry_certificate_valid") is not True and not reason:
            raise ValueError("invalid extended component lacks first rejection code")
        row = {
            "case_token": token,
            "replacement_component_id": component_id,
            "extended_frozen_target_set_count": len(matched_targets),
            "extended_frozen_target_sets_sha256": _sha_json(target_commitments),
            "support_parent_count": int(component["support_parent_count"]),
            "support_mode_count": int(component["support_mode_count"]),
            "causal_added_same_surface_edge_count": int(
                attribution["causal_added_same_surface_edge_count"]
            ),
            "causal_added_same_surface_edge_ids_sha256": _require_sha(
                attribution["causal_added_same_surface_edge_ids_sha256"],
                f"{token}.{component_id}.causal edge SHA",
            ),
            "geometry_gate_depth": int(component.get("geometry_gate_depth") or 0),
            "first_rejection_code": reason,
            "geometry_certificate_valid": component.get("geometry_certificate_valid") is True,
            "control_equivalent_certificate_ids": list(
                novelty.get("control_equivalent_certificate_ids") or ()
            ),
        }
        row["causal_funnel_row_sha256"] = _record_hash(
            row, "causal_funnel_row_sha256"
        )
        if set(row) != ROW_FIELDS:
            raise AssertionError("causal funnel row schema drift")
        rows.append(row)
        rejections[
            "VALID_CONTROL_EQUIVALENT_CERTIFICATE"
            if row["geometry_certificate_valid"]
            and row["control_equivalent_certificate_ids"]
            else (
                "VALID_NOVEL_CERTIFICATE"
                if row["geometry_certificate_valid"]
                else str(row["first_rejection_code"])
            )
        ] += 1
    expected_extensions = int(
        probe.get("case_counters", {}).get(
            "two_sided_one_parent_support_extended_count", -1
        )
    )
    if len(rows) != expected_extensions:
        raise ValueError(
            f"{token}: recomputed {len(rows)} extensions != locked {expected_extensions}"
        )
    valid_count = sum(row["geometry_certificate_valid"] for row in rows)
    equivalent_valid = sum(
        row["geometry_certificate_valid"]
        and bool(row["control_equivalent_certificate_ids"])
        for row in rows
    )
    return {
        "case_token": token,
        "frozen_target_count": len(targets),
        "edge_dependent_fragment_count": sum(
            row.get("edge_semantics_dependent") is True
            for row in probe.get("fragment_edge_attribution_table") or ()
        ),
        "accepted_added_union_edge_count": sum(
            int(row.get("accepted_added_same_surface_union_edge_count") or 0)
            for row in probe.get("fragment_edge_attribution_table") or ()
        ),
        "edge_dependent_component_count": len(causal_components),
        "edge_dependent_valid_certificate_count": edge_dependent_valid,
        "control_equivalent_edge_dependent_valid_certificate_count": control_equivalent_edge_dependent_valid,
        "causal_component_first_outcome_counts": dict(
            sorted(causal_component_outcomes.items())
        ),
        "rows": rows,
        "extended_valid_certificate_count": valid_count,
        "extended_control_equivalent_valid_certificate_count": equivalent_valid,
        "first_outcome_counts": dict(sorted(rejections.items())),
    }


def _audit_case_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return _audit_case(
        token=str(task["token"]),
        bundle_case=dict(task["bundle_case"]),
        probe_path=Path(str(task["probe_path"])),
        pool_path=Path(str(task["pool_path"])),
        control_path=Path(str(task["control_path"])),
    )


def build_report(
    *,
    replacement_bundle_path: Path,
    expected_bundle_sha256: str,
    probe_root: Path,
    candidate_pool_manifest_path: Path,
    frozen_control_manifest_path: Path,
    expected_case_tokens: Sequence[str] = EXPECTED_CASE_TOKENS,
    expected_extension_count: int = EXPECTED_EXTENSION_COUNT,
    workers: int = 8,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    bundle_path = replacement_bundle_path.resolve()
    actual_bundle_sha = _sha_file(bundle_path)
    if actual_bundle_sha != _require_sha(expected_bundle_sha256, "expected bundle SHA"):
        raise ValueError("replacement bundle SHA differs from expected lock input")
    bundle = _read(bundle_path)
    if (
        bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or bundle.get("kind") != BUNDLE_KIND
        or bundle.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("replacement bundle identity mismatch")
    if bundle.get("truth_inputs_consumed") is not False:
        raise ValueError("causal funnel requires an inference-only bundle")
    bundle_cases = _index_cases(
        bundle, "replacement bundle", expected_tokens=expected_case_tokens
    )
    pool_manifest_path = candidate_pool_manifest_path.resolve()
    control_manifest_path = frozen_control_manifest_path.resolve()
    pool_rows = _index_cases(
        _read(pool_manifest_path),
        "candidate-pool manifest",
        expected_tokens=expected_case_tokens,
    )
    control_rows = _index_cases(
        _read(control_manifest_path),
        "frozen-control manifest",
        expected_tokens=expected_case_tokens,
    )

    rows: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    aggregate_rejections: Counter[str] = Counter()
    aggregate_target_count = 0
    aggregate_valid = 0
    aggregate_control_equivalent_valid = 0
    aggregate_edge_dependent_fragments = 0
    aggregate_accepted_added_union_edges = 0
    aggregate_edge_dependent_components = 0
    aggregate_edge_dependent_valid = 0
    aggregate_control_equivalent_edge_dependent_valid = 0
    aggregate_causal_component_outcomes: Counter[str] = Counter()
    tasks = [
        {
            "token": token,
            "bundle_case": bundle_cases[token],
            "probe_path": str(probe_root.resolve() / f"{token}.json"),
            "pool_path": str(
                Path(str(pool_rows[token].get("path") or "")).resolve()
            ),
            "control_path": str(
                Path(str(control_rows[token].get("probe_path") or "")).resolve()
            ),
        }
        for token in sorted(expected_case_tokens)
    ]
    if workers == 1:
        results = [_audit_case_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, max_tasks_per_child=1
        ) as executor:
            results = list(executor.map(_audit_case_task, tasks))
    for result in results:
        token = str(result["case_token"])
        case_rows = list(result["rows"])
        rows.extend(case_rows)
        aggregate_target_count += int(result["frozen_target_count"])
        aggregate_rejections.update(result["first_outcome_counts"])
        aggregate_valid += int(result["extended_valid_certificate_count"])
        aggregate_control_equivalent_valid += int(
            result["extended_control_equivalent_valid_certificate_count"]
        )
        aggregate_edge_dependent_fragments += int(
            result["edge_dependent_fragment_count"]
        )
        aggregate_accepted_added_union_edges += int(
            result["accepted_added_union_edge_count"]
        )
        aggregate_edge_dependent_components += int(
            result["edge_dependent_component_count"]
        )
        aggregate_edge_dependent_valid += int(
            result["edge_dependent_valid_certificate_count"]
        )
        aggregate_control_equivalent_edge_dependent_valid += int(
            result["control_equivalent_edge_dependent_valid_certificate_count"]
        )
        aggregate_causal_component_outcomes.update(
            result["causal_component_first_outcome_counts"]
        )
        per_case.append(
            {
                "case_token": token,
                "frozen_target_count": int(result["frozen_target_count"]),
                "edge_dependent_fragment_count": int(
                    result["edge_dependent_fragment_count"]
                ),
                "accepted_added_union_edge_count": int(
                    result["accepted_added_union_edge_count"]
                ),
                "edge_dependent_component_count": int(
                    result["edge_dependent_component_count"]
                ),
                "edge_dependent_valid_certificate_count": int(
                    result["edge_dependent_valid_certificate_count"]
                ),
                "control_equivalent_edge_dependent_valid_certificate_count": int(
                    result[
                        "control_equivalent_edge_dependent_valid_certificate_count"
                    ]
                ),
                "causal_component_first_outcome_counts": deepcopy(
                    result["causal_component_first_outcome_counts"]
                ),
                "extended_component_count": len(case_rows),
                "extended_valid_certificate_count": int(
                    result["extended_valid_certificate_count"]
                ),
                "extended_control_equivalent_valid_certificate_count": int(
                    result["extended_control_equivalent_valid_certificate_count"]
                ),
                "first_outcome_counts": deepcopy(result["first_outcome_counts"]),
                "case_rows_sha256": _sha_json(case_rows),
            }
        )
    if len(rows) != expected_extension_count:
        raise ValueError(
            f"aggregate extension count {len(rows)} != expected {expected_extension_count}"
        )
    if len(rows) != int(
        bundle.get("aggregate_two_sided_single_parent_support_extended_count", -1)
    ):
        raise ValueError("causal report extension count differs from locked bundle")
    rows.sort(key=lambda row: (row["case_token"], row["replacement_component_id"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "profile": PROFILE,
        "research_only": True,
        "inference_safe": True,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "replacement_bundle_sha256": actual_bundle_sha,
        "candidate_pool_manifest_sha256": _sha_file(pool_manifest_path),
        "frozen_control_manifest_sha256": _sha_file(control_manifest_path),
        "case_count": len(expected_case_tokens),
        "aggregate": {
            "added_same_surface_edge_count": int(
                bundle.get("aggregate_added_same_surface_edge_count", -1)
            ),
            "accepted_added_union_edge_count": aggregate_accepted_added_union_edges,
            "edge_dependent_fragment_count": aggregate_edge_dependent_fragments,
            "edge_dependent_component_count": aggregate_edge_dependent_components,
            "edge_dependent_valid_certificate_count": aggregate_edge_dependent_valid,
            "control_equivalent_edge_dependent_valid_certificate_count": aggregate_control_equivalent_edge_dependent_valid,
            "novel_edge_dependent_valid_certificate_count": aggregate_edge_dependent_valid
            - aggregate_control_equivalent_edge_dependent_valid,
            "causal_component_first_outcome_counts": dict(
                sorted(aggregate_causal_component_outcomes.items())
            ),
            "frozen_two_sided_one_parent_target_count": aggregate_target_count,
            "extended_component_count": len(rows),
            "extended_valid_certificate_count": aggregate_valid,
            "extended_control_equivalent_valid_certificate_count": aggregate_control_equivalent_valid,
            "extended_novel_valid_certificate_count": aggregate_valid
            - aggregate_control_equivalent_valid,
            "first_outcome_counts": dict(sorted(aggregate_rejections.items())),
        },
        "per_case": per_case,
        "rows_sha256": _sha_json(rows),
        "rows": rows,
    }
    report["report_sha256"] = _record_hash(report, "report_sha256")
    return report


def canonical_report_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(payload) + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement-bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = build_report(
        replacement_bundle_path=args.replacement_bundle,
        expected_bundle_sha256=args.expected_bundle_sha256,
        probe_root=args.probe_root,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        frozen_control_manifest_path=args.frozen_control_manifest,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_report_bytes(report))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": _sha_file(args.output.resolve()),
                **report["aggregate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
