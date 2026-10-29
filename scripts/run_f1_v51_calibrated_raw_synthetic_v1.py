#!/usr/bin/env python3
"""Run the frozen V51 mesh-only chain on opaque calibrated synthetic V1 sources.

The organized frames are intentionally not passed to this baseline.  This
runner changes only the corpus token/namespace and event sink around the
already-frozen per-case V51 execution function.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_f1_v51_new_untouched_corpus_blind_once as frozen


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY"
CASE_IDS = tuple(f"O{index:03d}" for index in range(1, 31))
SANDBOX = ROOT / "research/f1_v51_synthetic_stress_detector.sb"
RETAINED_RUN = frozen._run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_frozen(event_path: Path) -> None:
    frozen.TOKEN = TOKEN
    frozen.CASE_IDS = CASE_IDS
    frozen.SANDBOX = SANDBOX
    frozen.THREAD_ENV = {**frozen.THREAD_ENV, "PYTHONPATH": str(ROOT)}

    def run_stage(stage: str, command: list[str], case_dir: Path) -> dict[str, Any]:
        if stage == "v16":
            command = list(command)
            command[1] = str(ROOT / "scripts/select_f1_mesh_route_v16_empty_safe.py")
        elif stage == "v43":
            command = list(command)
            command[1] = str(ROOT / "scripts/run_f1_v43_pointcloud_safe.py")
        return RETAINED_RUN(stage, command, case_dir)

    frozen._run = run_stage

    def append_event(payload: dict[str, Any]) -> None:
        record = {
            "token": TOKEN,
            "event_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "research_only": True,
            "confirmatory_eligible": False,
            **payload,
        }
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("ab", buffering=0) as handle:
            handle.write(encoded)
            os.fsync(handle.fileno())

    frozen._append_event = append_event


def _validate_public_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("token") != TOKEN or payload.get("schema_version") != 1:
        raise ValueError("public manifest token/schema mismatch")
    if payload.get("confirmatory_eligible") is not False:
        raise ValueError("corpus must remain non-confirmatory")
    rows = list(payload.get("cases") or [])
    if [str(row.get("opaque_case_id")) for row in rows] != list(CASE_IDS):
        raise ValueError("public manifest must contain O001..O030 in exact order")
    cases: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["opaque_case_id"])
        bundle = Path(str(row["bundle_path"])).resolve()
        scan = bundle / "source.ply"
        if not scan.is_file() or _sha(scan) != row.get("source_sha256"):
            raise ValueError(f"{case_id}: source hash mismatch")
        manifest = bundle / "manifest.json"
        if not manifest.is_file() or _sha(manifest) != row.get("manifest_sha256"):
            raise ValueError(f"{case_id}: bundle manifest hash mismatch")
        cases.append({"case_id": case_id, "scan_path": str(scan), "scan_sha256": row["source_sha256"]})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic engineering token required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("synthetic execution output root is not empty")

    cases = _validate_public_manifest(args.public_manifest.resolve())
    _configure_frozen(args.output_root.resolve() / "invocation_events.jsonl")
    records: list[dict[str, Any]] = []
    for pass_name in ("primary", "replay"):
        for case in cases:
            records.append(
                frozen._run_case(
                    case,
                    args.output_root.resolve(),
                    pass_name,
                    args.v9_model.resolve(),
                    args.v4_model.resolve(),
                )
            )

    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    identity = {
        case_id: primary[case_id]["semantic_result_sha256"] == replay[case_id]["semantic_result_sha256"]
        for case_id in CASE_IDS
    }
    if not all(identity.values()):
        raise RuntimeError("synthetic primary/replay semantic bytes differ")
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_calibrated_raw_synthetic_v1_execution",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "truth_opened_by_detector": False,
        "organized_frames_opened_by_baseline": False,
        "selected_case_ids": list(CASE_IDS),
        "passes": ["primary", "replay"],
        "semantic_replay_identity": identity,
        "records": records,
    }
    frozen._write_json(args.output_root / "execution_manifest.json", manifest, compact=True)
    print(json.dumps({"case_count": 30, "replay_identity": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
