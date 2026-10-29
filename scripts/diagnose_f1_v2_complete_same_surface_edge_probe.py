#!/usr/bin/env python3
"""Build one inference-safe V2 complete-SAME_SURFACE edge probe artifact.

Only the frozen candidate-pool JSON is parsed.  The scan and frozen A0
prediction are opened as byte streams solely to compute SHA-256 digests.
"""

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


PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "f1_v2_complete_same_surface_edge_probe_v1"
PRIMITIVE_PROFILE = "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
INPUT_PROJECTION_PROFILE = "V2_COMPLETE_EDGE_INPUT_PROJECTION_V1"
ALLOWED_EDGE_PROBE_INPUT_FIELDS = (
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
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    result = str(value or "").lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _read_candidate_pool(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("candidate pool must be a JSON object")
    return dict(payload)


def project_edge_probe_input(raw_pool: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy exactly the preregistered edge-probe input allowlist."""

    missing = [
        field for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS if field not in raw_pool
    ]
    if missing:
        raise ValueError(f"edge-probe input projection missing fields: {missing}")
    projected = {
        field: deepcopy(raw_pool[field]) for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS
    }
    if tuple(projected) != ALLOWED_EDGE_PROBE_INPUT_FIELDS:
        raise AssertionError("edge-probe projection order mismatch")
    return projected


def _control_api() -> tuple[Any, Any, Any]:
    from backend.f1_local_plane_fragment_probe import (
        build_local_plane_fragment_probe,
        canonical_json_bytes,
        validate_local_plane_fragment_payload,
    )

    return (
        build_local_plane_fragment_probe,
        canonical_json_bytes,
        validate_local_plane_fragment_payload,
    )


def _replacement_api() -> tuple[Any, Any, Any]:
    try:
        from backend.f1_v2_complete_same_surface_edge_probe import (
            build_v2_complete_same_surface_edge_probe,
            canonical_json_bytes,
            validate_v2_complete_same_surface_edge_probe_payload,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1 backend core is unavailable"
        ) from exc
    return (
        build_v2_complete_same_surface_edge_probe,
        canonical_json_bytes,
        validate_v2_complete_same_surface_edge_probe_payload,
    )


def build_v2_complete_same_surface_edge_probe_artifact(
    *,
    scan_path: Path,
    candidate_pool_path: Path,
    a0_prediction_path: Path,
    expected_control_probe_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Regenerate the frozen control and return canonical replacement bytes."""

    scan_path = scan_path.resolve()
    candidate_pool_path = candidate_pool_path.resolve()
    a0_prediction_path = a0_prediction_path.resolve()
    expected_control_sha = _require_sha256(
        expected_control_probe_sha256, label="expected control probe SHA-256"
    )
    scan_sha = sha256_file(scan_path)
    candidate_pool_sha = sha256_file(candidate_pool_path)
    a0_prediction_sha = sha256_file(a0_prediction_path)
    projected = project_edge_probe_input(_read_candidate_pool(candidate_pool_path))

    control_build, control_canonical, control_validate = _control_api()
    control = control_build(
        deepcopy(projected),
        scan_sha256=scan_sha,
        candidate_pool_sha256=candidate_pool_sha,
        a0_prediction_sha256=a0_prediction_sha,
    )
    control_validate(control, projected_input=projected)
    actual_control_sha = hashlib.sha256(control_canonical(control)).hexdigest()
    if actual_control_sha != expected_control_sha:
        raise ValueError(
            "regenerated frozen LPF V2 control hash differs from expected control"
        )

    build_probe, canonical_bytes, validate_payload = _replacement_api()
    payload = build_probe(
        deepcopy(projected),
        scan_sha256=scan_sha,
        candidate_pool_sha256=candidate_pool_sha,
        a0_prediction_sha256=a0_prediction_sha,
        frozen_control_probe_sha256=actual_control_sha,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("replacement core must return a mapping payload")
    result = dict(payload)
    validate_payload(result, projected_input=projected)
    if (
        result.get("schema_version") != PROBE_SCHEMA_VERSION
        or result.get("kind") != PROBE_KIND
        or result.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("replacement core returned a non-edge-probe identity")
    if _require_sha256(
        result.get("frozen_control_probe_sha256"), label="payload control probe"
    ) != expected_control_sha:
        raise ValueError("replacement payload references a different frozen control")
    encoded = canonical_bytes(result)
    if not isinstance(encoded, bytes):
        raise TypeError("canonical_json_bytes() must return bytes")
    return result, encoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--a0-prediction", required=True, type=Path)
    parser.add_argument("--expected-control-probe-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _payload, encoded = build_v2_complete_same_surface_edge_probe_artifact(
        scan_path=args.scan,
        candidate_pool_path=args.candidate_pool,
        a0_prediction_path=args.a0_prediction,
        expected_control_probe_sha256=args.expected_control_probe_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "replacement_probe_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
