#!/usr/bin/env python3
"""Build one inference-safe V2 cross-parent opposite-fragment rung probe."""

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
PROBE_KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_v1"
PRIMITIVE_PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CASE_RE = re.compile(r"^C[0-9]{3}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    result = str(value or "").lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _read_candidate_pool(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("candidate pool must be a JSON object")
    return dict(payload)


def _control_api() -> tuple[Any, Any, Any]:
    from backend.f1_local_plane_fragment_probe import (
        build_local_plane_fragment_probe,
        canonical_json_bytes,
        validate_local_plane_fragment_payload,
    )

    return build_local_plane_fragment_probe, canonical_json_bytes, validate_local_plane_fragment_payload


def _baseline_api() -> tuple[Any, Any, Any]:
    from backend.f1_v2_complete_same_surface_edge_probe import (
        build_v2_complete_same_surface_edge_probe,
        canonical_json_bytes,
        validate_v2_complete_same_surface_edge_probe_payload,
    )

    return (
        build_v2_complete_same_surface_edge_probe,
        canonical_json_bytes,
        validate_v2_complete_same_surface_edge_probe_payload,
    )


def _replacement_api() -> tuple[Any, Any, Any, Any]:
    try:
        from backend.f1_v2_cross_parent_opposite_fragment_rung_probe import (
            build_v2_cross_parent_opposite_fragment_rung_probe,
            canonical_json_bytes,
            project_cross_parent_rung_input,
            validate_v2_cross_parent_opposite_fragment_rung_probe_payload,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V1 core is unavailable"
        ) from exc
    return (
        build_v2_cross_parent_opposite_fragment_rung_probe,
        canonical_json_bytes,
        project_cross_parent_rung_input,
        validate_v2_cross_parent_opposite_fragment_rung_probe_payload,
    )


def _opaque_case_token(candidate_pool_path: Path) -> str:
    token = candidate_pool_path.stem
    if not _CASE_RE.fullmatch(token):
        raise ValueError(
            "candidate-pool basename must be the opaque Gold37 token C001-C037"
        )
    return token


def build_cross_parent_rung_probe_artifact(
    *,
    scan_path: Path,
    candidate_pool_path: Path,
    a0_prediction_path: Path,
    expected_frozen_control_probe_sha256: str,
    expected_complete_edge_baseline_probe_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Regenerate both baselines before building one replacement payload."""

    scan_path = scan_path.resolve()
    candidate_pool_path = candidate_pool_path.resolve()
    a0_prediction_path = a0_prediction_path.resolve()
    case_token = _opaque_case_token(candidate_pool_path)
    expected_control_sha = _require_sha(
        expected_frozen_control_probe_sha256, label="expected frozen-control probe"
    )
    expected_baseline_sha = _require_sha(
        expected_complete_edge_baseline_probe_sha256,
        label="expected complete-edge baseline probe",
    )
    scan_sha = sha256_file(scan_path)
    pool_sha = sha256_file(candidate_pool_path)
    a0_sha = sha256_file(a0_prediction_path)
    raw_pool = _read_candidate_pool(candidate_pool_path)
    build_probe, canonical_bytes, project_input, validate_payload = _replacement_api()
    projected = project_input(raw_pool)

    control_build, control_canonical, control_validate = _control_api()
    control = control_build(
        deepcopy(projected),
        scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha,
    )
    control_validate(control, projected_input=projected)
    actual_control_sha = hashlib.sha256(control_canonical(control)).hexdigest()
    if actual_control_sha != expected_control_sha:
        raise ValueError("regenerated LPF V2 control differs from expected hash")

    baseline_build, baseline_canonical, baseline_validate = _baseline_api()
    baseline = baseline_build(
        deepcopy(projected),
        scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha,
        frozen_control_probe_sha256=actual_control_sha,
    )
    baseline_validate(baseline, projected_input=projected)
    actual_baseline_sha = hashlib.sha256(baseline_canonical(baseline)).hexdigest()
    if actual_baseline_sha != expected_baseline_sha:
        raise ValueError("regenerated complete-edge baseline differs from expected hash")

    payload = build_probe(
        deepcopy(projected),
        case_token=case_token,
        scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha,
        frozen_control_probe_sha256=actual_control_sha,
        complete_edge_baseline_probe_sha256=actual_baseline_sha,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("cross-parent rung core must return a mapping payload")
    result = dict(payload)
    validate_payload(result, projected_input=projected)
    if (
        result.get("schema_version") != PROBE_SCHEMA_VERSION
        or result.get("kind") != PROBE_KIND
        or result.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("cross-parent rung core returned the wrong identity")
    if result.get("gold_open_authorized") is not False:
        raise ValueError("cross-parent rung payload must keep gold closed")
    for field, expected in (
        ("frozen_control_probe_sha256", expected_control_sha),
        ("complete_edge_baseline_probe_sha256", expected_baseline_sha),
    ):
        if _require_sha(result.get(field), label=field) != expected:
            raise ValueError(f"replacement payload {field} mismatch")
    encoded = canonical_bytes(result)
    if not isinstance(encoded, bytes):
        raise TypeError("canonical_json_bytes() must return bytes")
    return result, encoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--a0-prediction", required=True, type=Path)
    parser.add_argument("--expected-frozen-control-probe-sha256", required=True)
    parser.add_argument("--expected-complete-edge-baseline-probe-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _payload, encoded = build_cross_parent_rung_probe_artifact(
        scan_path=args.scan,
        candidate_pool_path=args.candidate_pool,
        a0_prediction_path=args.a0_prediction,
        expected_frozen_control_probe_sha256=args.expected_frozen_control_probe_sha256,
        expected_complete_edge_baseline_probe_sha256=(
            args.expected_complete_edge_baseline_probe_sha256
        ),
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
