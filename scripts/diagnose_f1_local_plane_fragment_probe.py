#!/usr/bin/env python3
"""Build one inference-safe LOCAL_PLANE_FRAGMENT_PROBE_V2 artifact.

Only the frozen candidate pool is parsed.  The scan and frozen A0 prediction
are opened as byte streams solely to compute their SHA-256 digests.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROBE_SCHEMA_VERSION = 2
PROBE_KIND = "f1_local_plane_fragment_probe_v2"
PROFILE = "LOCAL_PLANE_FRAGMENT_PROBE_V2"
INPUT_PROJECTION_PROFILE = "LPF_INPUT_PROJECTION_V2"
ALLOWED_POOL_PROJECTION_FIELDS = (
    "support_representation_profile",
    "source_point_serialization",
    "support_parent_table",
    "support_mode_table",
    "support_mode_edges",
    "plane_table",
    "transition_parent_table_sha256",
    "transition_mode_table_sha256",
    "transition_mode_edge_table_sha256",
    "transition_local_spacing_mm",
    "transition_atom_caps",
    "working_point_budget",
    "transient_sampled_point_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_candidate_pool(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("candidate pool must be a JSON object")
    return dict(payload)


def project_lpf_input(raw_pool: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy the exact Ruling-B inference allowlist in canonical order."""

    missing = [field for field in ALLOWED_POOL_PROJECTION_FIELDS if field not in raw_pool]
    if missing:
        raise ValueError(f"LPF input projection missing fields: {missing}")
    projection = {
        field: deepcopy(raw_pool[field]) for field in ALLOWED_POOL_PROJECTION_FIELDS
    }
    if tuple(projection) != ALLOWED_POOL_PROJECTION_FIELDS:
        raise AssertionError("LPF input projection schema/order mismatch")
    return projection


def _core_api() -> tuple[Any, Any, Any]:
    try:
        from backend.f1_local_plane_fragment_probe import (
            build_local_plane_fragment_probe,
            canonical_json_bytes,
            validate_local_plane_fragment_payload,
        )
    except (ImportError, ModuleNotFoundError) as exc:  # explicit integration failure
        raise RuntimeError(
            "LOCAL_PLANE_FRAGMENT_PROBE_V2 backend core is unavailable"
        ) from exc
    return (
        build_local_plane_fragment_probe,
        canonical_json_bytes,
        validate_local_plane_fragment_payload,
    )


def build_local_plane_fragment_probe_artifact(
    *,
    scan_path: Path,
    candidate_pool_path: Path,
    a0_prediction_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Return the payload and canonical bytes without parsing scan/A0 JSON."""

    scan_path = scan_path.resolve()
    candidate_pool_path = candidate_pool_path.resolve()
    a0_prediction_path = a0_prediction_path.resolve()
    scan_sha256 = sha256_file(scan_path)
    candidate_pool_sha256 = sha256_file(candidate_pool_path)
    a0_prediction_sha256 = sha256_file(a0_prediction_path)
    candidate_pool = project_lpf_input(_read_candidate_pool(candidate_pool_path))
    build_probe, canonical_bytes, validate_payload = _core_api()
    payload = build_probe(
        candidate_pool,
        scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("LPF backend core must return a mapping payload")
    result = dict(payload)
    validate_payload(result)
    if (
        result.get("schema_version") != PROBE_SCHEMA_VERSION
        or result.get("kind") != PROBE_KIND
        or result.get("profile") != PROFILE
    ):
        raise ValueError("LPF backend returned a non-V2 probe identity")
    attestation = result.get("input_contract_attestation")
    if not isinstance(attestation, Mapping) or attestation.get(
        "projection_profile"
    ) != INPUT_PROJECTION_PROFILE:
        raise ValueError("LPF backend returned an invalid input projection attestation")
    encoded = canonical_bytes(result)
    if not isinstance(encoded, bytes):
        raise TypeError("canonical_json_bytes() must return bytes")
    return result, encoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--a0-prediction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _payload, encoded = build_local_plane_fragment_probe_artifact(
        scan_path=args.scan,
        candidate_pool_path=args.candidate_pool,
        a0_prediction_path=args.a0_prediction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
