#!/usr/bin/env python3
"""Validate and lock the 37-case LPF-M1-V2 inference bundle."""

from __future__ import annotations

import argparse
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

from scripts.diagnose_f1_local_plane_fragment_probe import (  # noqa: E402
    PROFILE,
    PROBE_KIND,
    PROBE_SCHEMA_VERSION,
    project_lpf_input,
    sha256_file,
)
from scripts.run_f1_local_plane_fragment_probe_batch import (  # noqa: E402
    BATCH_KIND,
    BATCH_SCHEMA_VERSION,
    MAX_PEAK_RSS_BYTES,
    PREREGISTERED_INFERENCE_MANIFEST_SHA256,
    PREREGISTERED_V2_POOL_MANIFEST_SHA256,
)
from scripts.run_f1_local_plane_fragment_probe_replays import (  # noqa: E402
    EXPECTED_CASES as EXPECTED_REPLAY_CASES,
    KIND as REPLAY_ATTESTATION_KIND,
    SELECTION_REASONS,
    select_designated_cases,
    validate_replay_attestation,
)


BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KIND = "f1_local_plane_fragment_probe_bundle_v2"
LOCK_ATTESTATION_KIND = "f1_local_plane_fragment_probe_lock_attestation_v2"
EXPECTED_CASE_TOKENS = frozenset(f"C{index:03d}" for index in range(1, 38))
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
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _core_api() -> tuple[Any, Any, Any]:
    from backend.f1_local_plane_fragment_probe import (
        build_local_plane_fragment_probe,
        canonical_json_bytes,
        validate_local_plane_fragment_payload,
    )

    return build_local_plane_fragment_probe, canonical_json_bytes, validate_local_plane_fragment_payload


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _sha_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _record_hash(record: Mapping[str, Any], hash_field: str) -> str:
    return _sha_json({key: value for key, value in record.items() if key != hash_field})


def _rows(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    raw = payload.get("cases")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} rows must be objects")
        token = str(item.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} tokens must be nonempty and unique")
        result[token] = dict(item)
    return result


def build_lock_attestation(criteria: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {key: criteria.get(key) is True for key in LOCK_CRITERION_KEYS}
    if set(criteria) != set(LOCK_CRITERION_KEYS):
        # Missing and extra criteria both prevent a lock.
        normalized = {
            key: (criteria.get(key) is True and set(criteria) == set(LOCK_CRITERION_KEYS))
            for key in LOCK_CRITERION_KEYS
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
        raise ValueError("lock attestation criterion schema mismatch")
    if any(value is not True and value is not False for value in criteria.values()):
        raise ValueError("lock attestation criteria must be Boolean")
    ready = all(criteria[key] is True for key in LOCK_CRITERION_KEYS)
    if record.get("ready_for_gold") is not ready:
        raise ValueError("lock attestation ready_for_gold mismatch")
    if _require_sha(record.get("lock_attestation_sha256"), "lock attestation") != _record_hash(
        record, "lock_attestation_sha256"
    ):
        raise ValueError("lock attestation hash mismatch")


def _projected_rebuild(
    pool_path: Path,
    *,
    scan_sha256: str,
    candidate_pool_sha256: str,
    a0_prediction_sha256: str,
) -> dict[str, Any]:
    build_probe, _canonical, _validate = _core_api()
    return build_probe(
        project_lpf_input(_read(pool_path)),
        scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
    )


def _selection_manifest_from_batch(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cases = []
    for token in sorted(rows):
        counters = dict(rows[token].get("case_counters") or {})
        cases.append(
            {
                "case_token": token,
                "working_point_count": int(counters.get("inherited_working_point_count") or -1),
                "transition_mode_count": int(counters.get("input_mode_count") or -1),
                "transition_mode_edge_count": int(
                    counters.get("input_same_surface_edge_count") or 0
                )
                + int(counters.get("input_cross_mode_edge_count") or 0),
            }
        )
    return {"cases": cases}


def build_local_plane_fragment_probe_bundle(
    *,
    probe_manifest_path: Path,
    replay_attestation_path: Path,
) -> dict[str, Any]:
    probe_manifest_path = probe_manifest_path.resolve()
    replay_attestation_path = replay_attestation_path.resolve()
    manifest = _read(probe_manifest_path)
    replay = _read(replay_attestation_path)
    if (
        manifest.get("schema_version") != BATCH_SCHEMA_VERSION
        or manifest.get("kind") != BATCH_KIND
        or manifest.get("profile") != PROFILE
    ):
        raise ValueError("V2 batch manifest identity mismatch")
    if _require_sha(manifest.get("inference_manifest_sha256"), "inference manifest") != PREREGISTERED_INFERENCE_MANIFEST_SHA256:
        raise ValueError("batch is not tied to frozen Gold37 inference manifest")
    if _require_sha(manifest.get("candidate_pool_manifest_sha256"), "pool manifest") != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("batch is not tied to frozen V2 pool manifest")
    rows = _rows(manifest, "V2 batch manifest")
    if set(rows) != EXPECTED_CASE_TOKENS:
        raise ValueError("V2 batch manifest must contain exact C001-C037 set")
    if int(manifest.get("case_count") or -1) != 37 or int(
        manifest.get("successful_case_count") or -1
    ) != 37:
        raise ValueError("V2 batch requires 37/37 successful jobs")
    expected_batch_flags = {
        "research_only": True,
        "inference_safe": True,
        "prediction_changed": False,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "isolated_process_per_case": True,
    }
    for field, expected in expected_batch_flags.items():
        if manifest.get(field) is not expected:
            raise ValueError(f"V2 batch safety flag mismatch: {field}")
    selection_manifest = _selection_manifest_from_batch(rows)
    if select_designated_cases(selection_manifest) != SELECTION_REASONS:
        raise ValueError("batch counters do not reproduce designated replay selection")
    validate_replay_attestation(
        replay, batch_rows=rows, pool_manifest=selection_manifest
    )
    if _require_sha(replay.get("probe_manifest_sha256"), "replay batch manifest") != sha256_file(
        probe_manifest_path
    ):
        raise ValueError("replay attestation references a different batch manifest")
    if _require_sha(replay.get("candidate_pool_manifest_sha256"), "replay pool manifest") != PREREGISTERED_V2_POOL_MANIFEST_SHA256:
        raise ValueError("replay attestation references a different pool manifest")

    _build, canonical_bytes, validate_payload = _core_api()
    cases: list[dict[str, Any]] = []
    maximum_peak_rss = 0
    for token in sorted(EXPECTED_CASE_TOKENS):
        row = rows[token]
        if row.get("status") != "SUCCESS":
            raise ValueError(f"{token}: ordinary batch job did not succeed")
        probe_path = Path(str(row.get("probe_path") or "")).resolve()
        scan_path = Path(str(row.get("scan_path") or "")).resolve()
        pool_path = Path(str(row.get("candidate_pool_path") or "")).resolve()
        prediction_path = Path(str(row.get("a0_prediction_path") or "")).resolve()
        for label, path in (
            ("probe", probe_path), ("scan", scan_path),
            ("candidate pool", pool_path), ("A0 prediction", prediction_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        probe_sha = sha256_file(probe_path)
        scan_sha = sha256_file(scan_path)
        pool_sha = sha256_file(pool_path)
        prediction_sha = sha256_file(prediction_path)
        for field, actual in (
            ("probe_sha256", probe_sha), ("scan_sha256", scan_sha),
            ("candidate_pool_sha256", pool_sha),
            ("a0_prediction_sha256", prediction_sha),
        ):
            if _require_sha(row.get(field), f"{token} {field}") != actual:
                raise ValueError(f"{token}: {field} changed")
        projected_input = project_lpf_input(_read(pool_path))
        payload = _read(probe_path)
        validate_payload(payload, projected_input=projected_input)
        if (
            payload.get("schema_version") != PROBE_SCHEMA_VERSION
            or payload.get("kind") != PROBE_KIND
            or payload.get("profile") != PROFILE
        ):
            raise ValueError(f"{token}: per-case V2 identity mismatch")
        if probe_path.read_bytes() != canonical_bytes(payload):
            raise ValueError(f"{token}: per-case output is not canonical")
        rebuilt = _projected_rebuild(
            pool_path,
            scan_sha256=scan_sha,
            candidate_pool_sha256=pool_sha,
            a0_prediction_sha256=prediction_sha,
        )
        validate_payload(rebuilt, projected_input=projected_input)
        if canonical_bytes(rebuilt) != canonical_bytes(payload):
            raise ValueError(f"{token}: deterministic projected rebuild mismatch")
        input_attestation = dict(payload.get("input_contract_attestation") or {})
        input_attestation_sha = _require_sha(
            input_attestation.get("input_contract_attestation_sha256"),
            f"{token} input contract",
        )
        for field, value in (
            ("input_contract_attestation_sha256", input_attestation_sha),
            ("inference_semantic_sha256", payload.get("inference_semantic_sha256")),
            ("engineering_profile_sha256", payload.get("engineering_profile_sha256")),
        ):
            if _require_sha(row.get(field), f"{token} row {field}") != _require_sha(
                value, f"{token} payload {field}"
            ):
                raise ValueError(f"{token}: {field} mismatch")
        for field in ("input_table_hashes", "output_table_hashes"):
            if dict(row.get(field) or {}) != dict(payload.get(field) or {}):
                raise ValueError(f"{token}: {field} mismatch")
        peak = int(row.get("peak_rss_bytes") or -1)
        if peak < 0 or peak >= MAX_PEAK_RSS_BYTES:
            raise ValueError(f"{token}: ordinary job RSS reaches/exceeds 16 GiB")
        if float(row.get("wall_seconds") or -1.0) < 0.0:
            raise ValueError(f"{token}: ordinary job wall time is invalid")
        if int(row.get("input_json_bytes") or -1) != pool_path.stat().st_size:
            raise ValueError(f"{token}: ordinary input byte measurement mismatch")
        if int(row.get("serialized_output_bytes") or -1) != probe_path.stat().st_size:
            raise ValueError(f"{token}: ordinary output byte measurement mismatch")
        maximum_peak_rss = max(maximum_peak_rss, peak)
        cases.append(
            {
                "case_token": token,
                "status": "SUCCESS",
                "probe_sha256": probe_sha,
                "scan_sha256": scan_sha,
                "candidate_pool_sha256": pool_sha,
                "a0_prediction_sha256": prediction_sha,
                "probe": deepcopy(payload),
            }
        )
    replay_max = max(
        int(case["maximum_peak_rss_bytes"]) for case in replay["cases"]
    )
    maximum_peak_rss = max(maximum_peak_rss, replay_max)
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    lock = build_lock_attestation(criteria)
    validate_lock_attestation(lock)
    ready = lock["ready_for_gold"] is True
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "profile": PROFILE,
        "locked": ready,
        "ready_for_gold": ready,
        "research_only": True,
        "inference_safe": True,
        "prediction_changed": False,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "case_count": 37,
        "successful_case_count": 37,
        "inference_manifest_sha256": PREREGISTERED_INFERENCE_MANIFEST_SHA256,
        "candidate_pool_manifest_sha256": PREREGISTERED_V2_POOL_MANIFEST_SHA256,
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "replay_attestation_sha256": sha256_file(replay_attestation_path),
        "probe_hash_table_sha256": _sha_json(
            [[case["case_token"], case["probe_sha256"]] for case in cases]
        ),
        "inference_semantic_hash_table_sha256": _sha_json(
            [[case["case_token"], case["probe"]["inference_semantic_sha256"]] for case in cases]
        ),
        "engineering_profile_hash_table_sha256": _sha_json(
            [[case["case_token"], case["probe"]["engineering_profile_sha256"]] for case in cases]
        ),
        "input_contract_attestation_hash_table_sha256": _sha_json(
            [[case["case_token"], case["probe"]["input_contract_attestation"]["input_contract_attestation_sha256"]] for case in cases]
        ),
        "a0_prediction_hash_table_sha256": _sha_json(
            [[case["case_token"], case["a0_prediction_sha256"]] for case in cases]
        ),
        "maximum_peak_rss_bytes": maximum_peak_rss,
        "lock_attestation": lock,
        "lock_attestation_sha256": lock["lock_attestation_sha256"],
        "cases": cases,
    }
    return bundle


def canonical_bundle_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--replay-attestation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = build_local_plane_fragment_probe_bundle(
        probe_manifest_path=args.probe_manifest,
        replay_attestation_path=args.replay_attestation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bundle_bytes(bundle))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "bundle_sha256": sha256_file(args.output),
        "lock_attestation_sha256": bundle["lock_attestation_sha256"],
        "ready_for_gold": bundle["ready_for_gold"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
