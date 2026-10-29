#!/usr/bin/env python3
"""Seal the V57 raw-range capability-not-opened STOP before inference.

This utility is deliberately not a detector.  It reads only the frozen corpus
manifest, the frozen corpus generator source, and the preserved professor
ruling.  It never imports or invokes V57 geometry, V56, truth, family maps, or
scoring code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


CAPABILITY_TOKEN = "OPEN_CALIBRATED_RAW_RANGE_VIEW_BUNDLE_V1_ONCE"
PRECOMMIT_TOKEN = "V57_CMTCC_MECHANISM_LOCKED"
TERMINAL_TOKEN = "V57_CMTCC_TERMINAL_RETIRE"
STOP_STATUS = "V57_CMTCC_CAPABILITY_NOT_OPENED_STOP"
EXPECTED_CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))
QUALIFYING_SUFFIXES = {".bag", ".mcap", ".oni", ".h5", ".hdf5", ".npz", ".npy"}
QUALIFYING_NAME_TOKENS = (
    "depth",
    "validity",
    "range_frame",
    "range-frame",
    "intrinsic",
    "extrinsic",
    "camera_pose",
    "camera-pose",
    "acquisition_manifest",
    "acquisition-manifest",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _qualifying_candidates(corpus_root: Path) -> list[str]:
    rows: list[str] = []
    excluded_roots = {"truth_cases", "truth_overlays"}
    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file() or any(part in excluded_roots for part in path.parts):
            continue
        relative = path.relative_to(corpus_root)
        lowered = relative.as_posix().lower()
        if path.suffix.lower() in QUALIFYING_SUFFIXES or any(token in lowered for token in QUALIFYING_NAME_TOKENS):
            rows.append(relative.as_posix())
    return rows


def build_stop(
    *,
    corpus_root: Path,
    generator_path: Path,
    ruling_path: Path,
    metadata_read_disclosure: bool,
) -> dict[str, Any]:
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_rows = list(manifest.get("cases") or ())
    case_ids = tuple(str(row.get("case_id")) for row in case_rows)
    if case_ids != EXPECTED_CASE_IDS:
        raise ValueError("locked corpus case order is not S001-S030")
    if int(manifest.get("case_count", -1)) != 30:
        raise ValueError("locked corpus case count is not 30")

    generator_source = generator_path.read_text(encoding="utf-8")
    required_direct_mesh_markers = (
        "def build_case(",
        "o3d.geometry.TriangleMesh(",
        "o3d.io.write_triangle_mesh(",
    )
    missing_markers = [marker for marker in required_direct_mesh_markers if marker not in generator_source]
    if missing_markers:
        raise ValueError(f"generator direct-mesh markers missing: {missing_markers}")
    forbidden_acquisition_markers = (
        "depth_array",
        "validity_mask",
        "camera_intrinsic",
        "camera_extrinsic",
        "raycast",
    )
    present_acquisition_markers = [marker for marker in forbidden_acquisition_markers if marker in generator_source]
    if present_acquisition_markers:
        raise ValueError(f"generator unexpectedly contains acquisition-frame markers: {present_acquisition_markers}")

    candidates = _qualifying_candidates(corpus_root)
    case_bundle_presence = {case_id: False for case_id in EXPECTED_CASE_IDS}
    for candidate in candidates:
        for case_id in EXPECTED_CASE_IDS:
            if case_id.lower() in candidate.lower():
                case_bundle_presence[case_id] = True
    available = [case_id for case_id, present in case_bundle_presence.items() if present]
    missing = [case_id for case_id, present in case_bundle_presence.items() if not present]
    if available:
        raise ValueError(f"candidate capability files require manual custody adjudication: {available}")

    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v57_capability_stop",
        "status": STOP_STATUS,
        "capability_opening_token": CAPABILITY_TOKEN,
        "precommit_token": PRECOMMIT_TOKEN,
        "terminal_token": TERMINAL_TOKEN,
        "architecture": "V57_CALIBRATED_MULTIVIEW_TURN_CHANGEPOINT_CONSENSUS_V1",
        "reason": "AUTHENTIC_CALIBRATED_RAW_RANGE_VIEW_BUNDLE_UNAVAILABLE_FOR_LOCKED_S001_S030",
        "capability_opened": False,
        "qualifying_bundle_candidate_count": len(candidates),
        "qualifying_bundle_candidates": candidates,
        "available_case_count": len(available),
        "available_case_ids": available,
        "missing_case_count": len(missing),
        "missing_case_ids": missing,
        "corpus_generator_mode": "DIRECT_TRIANGLE_MESH_CONSTRUCTION_WITHOUT_ORGANIZED_RANGE_FRAMES",
        "corpus_generator_direct_mesh_markers": list(required_direct_mesh_markers),
        "corpus_generator_acquisition_markers_present": present_acquisition_markers,
        "corpus_manifest_sha256": _sha256(manifest_path),
        "corpus_generator_sha256": _sha256(generator_path),
        "professor_ruling_sha256": _sha256(ruling_path),
        "detector_implemented": False,
        "detector_invocation_count": 0,
        "source_free_fixture_invocation_count": 0,
        "sealed_child_invocation_count": 0,
        "prediction_count": 0,
        "score_consumed": False,
        "visual_gate_consumed": False,
        "truth_opened_for_v57": False,
        "v56_outputs_opened": False,
        "mesh_rendering_substitute_used": False,
        "mesh_only_fallback_used": False,
        "rerender_or_new_view_generation_used": False,
        "operator_disclosure": {
            "generation_metadata_read_after_ruling": bool(metadata_read_disclosure),
            "generation_metadata_contains_forbidden_family_and_bend_count_fields": bool(metadata_read_disclosure),
            "read_used_by_detector_or_fixture": False,
            "detector_or_fixture_existed_at_time_of_read": False,
            "disposition": "ADDITIONAL_PROCESS_BREACH_REINFORCES_TERMINAL_STOP",
        },
        "no_rescue_boundary": {
            "retro_render_fused_mesh_as_raw_views": False,
            "invent_camera_poses": False,
            "replace_locked_scans": False,
            "create_v57b_same_mechanism": False,
            "run_partial_corpus": False,
        },
        "next_authority_required": "NEW_PROSPECTIVELY_ACQUIRED_CORPUS_WITH_AUTHENTIC_RAW_RANGE_BUNDLES_REQUIRES_FRESH_USER_CUSTODY_AND_PROFESSOR_PROTOCOL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--ruling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disclose-generation-metadata-read", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"terminal artifact already exists: {args.output}")
    payload = build_stop(
        corpus_root=args.corpus_root.resolve(),
        generator_path=args.generator.resolve(),
        ruling_path=args.ruling.resolve(),
        metadata_read_disclosure=args.disclose_generation_metadata_read,
    )
    _atomic_write_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "missing_case_count": payload["missing_case_count"],
        "detector_invocation_count": payload["detector_invocation_count"],
        "truth_opened_for_v57": payload["truth_opened_for_v57"],
    }, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
