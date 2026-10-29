#!/usr/bin/env python3
"""Run the frozen V51 chain on quarantined non-confirmatory synthetic scans.

This is an engineering stress lane, never evidence for the untouched-corpus
generalization claim. It deliberately reuses the frozen per-case execution
function while swapping only the corpus token, case namespace, event sink, and
sandbox. Truth and generator metadata are denied inside every detector stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_f1_v51_new_untouched_corpus_blind_once as frozen


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
ALL_CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))
SANDBOX = ROOT / "research/f1_v51_synthetic_stress_detector.sb"
RETAINED_RUN = frozen._run


def _configure_frozen(event_path: Path) -> None:
    frozen.TOKEN = TOKEN
    frozen.CASE_IDS = ALL_CASE_IDS
    frozen.SANDBOX = SANDBOX
    frozen.THREAD_ENV = {**frozen.THREAD_ENV, "PYTHONPATH": str(ROOT)}

    def run_stage(stage: str, command: list[str], case_dir: Path) -> dict[str, Any]:
        if stage == "v16":
            command = list(command)
            command[1] = str(ROOT / "scripts/select_f1_mesh_route_v16_empty_safe.py")
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


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return ALL_CASE_IDS
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("case IDs must be a non-empty unique comma-separated list")
    unknown = sorted(set(values).difference(ALL_CASE_IDS))
    if unknown:
        raise ValueError(f"unknown synthetic case IDs: {unknown}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-ids", default="all")
    parser.add_argument("--passes", choices=("primary", "primary,replay"), default="primary,replay")
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic engineering token required")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("synthetic execution output root is not empty")
    selected_ids = _parse_case_ids(args.case_ids)
    _configure_frozen(output_root / "invocation_events.jsonl")
    all_cases = frozen.validate_opaque_manifest(args.manifest.resolve())
    selected = [row for row in all_cases if str(row["case_id"]) in selected_ids]
    if [str(row["case_id"]) for row in selected] != [case_id for case_id in ALL_CASE_IDS if case_id in selected_ids]:
        raise ValueError("selected cases must retain manifest order")

    pass_names = tuple(args.passes.split(","))
    records: list[dict[str, Any]] = []
    for pass_name in pass_names:
        for case in selected:
            records.append(
                frozen._run_case(
                    case,
                    output_root,
                    pass_name,
                    args.v9_model.resolve(),
                    args.v4_model.resolve(),
                )
            )

    identity: dict[str, bool] = {}
    if "replay" in pass_names:
        primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
        replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
        identity = {
            case_id: primary[case_id]["semantic_result_sha256"] == replay[case_id]["semantic_result_sha256"]
            for case_id in selected_ids
        }
        if not all(identity.values()):
            raise RuntimeError("synthetic primary/replay semantic bytes differ")

    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_synthetic_stress_execution",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "truth_opened_by_detector": False,
        "scoring_opened_by_detector": False,
        "selected_case_ids": list(selected_ids),
        "passes": list(pass_names),
        "semantic_replay_identity": identity,
        "records": records,
    }
    frozen._write_json(output_root / "execution_manifest.json", manifest, compact=True)
    print(
        json.dumps(
            {
                "case_count": len(selected_ids),
                "passes": list(pass_names),
                "replay_identity": all(identity.values()) if identity else None,
                "manifest": str(output_root / "execution_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
