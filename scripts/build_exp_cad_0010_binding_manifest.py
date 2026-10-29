#!/usr/bin/env python3
"""Build the truth-denying, hash-bound case manifest for EXP_CAD_0010."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compile_cad_bend_template_v1 import _atomic


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def build_manifest(
    *,
    slice_path: Path,
    parent_summary_path: Path,
    parent_cases_root: Path,
    ontology_path: Path,
    acceptance_paths: list[Path],
) -> dict[str, Any]:
    source_slice = _load(slice_path)
    parent_summary = _load(parent_summary_path)
    ontology = _load(ontology_path)
    entries = source_slice.get("entries") or []
    if len(entries) != 13:
        raise ValueError("exactly 13 source entries required")
    if len(parent_summary.get("results") or []) != 13:
        raise ValueError("exactly 13 parent results required")
    entry_by_sha = {str(row["scan_sha256"]): row for row in entries}
    if len(entry_by_sha) != 13:
        raise ValueError("scan SHA values must be unique")
    ontology_events: dict[str, list[dict[str, Any]]] = {}
    for event in ontology.get("events") or []:
        ontology_events.setdefault(str(event["case_id"]), []).append(event)
    records = []
    all_bound_refs: list[str] = []
    for ordinal in range(1, 14):
        parent_case_id = f"C{ordinal:03d}"
        parent_path = parent_cases_root / parent_case_id / "semantic_result.json"
        parent = _load(parent_path)
        scan_sha = str(parent["scan_sha256"])
        entry = entry_by_sha.get(scan_sha)
        if entry is None:
            raise ValueError(f"{parent_case_id}: parent scan SHA absent from source slice")
        if str(parent.get("step_sha256")) != str(entry["cad_sha256"]):
            raise ValueError(f"{parent_case_id}: STEP SHA mismatch")
        parent_events = sorted(
            str(row["event_id"])
            for row in parent.get("records") or []
            if row.get("accepted") is True
        )
        mapped = sorted(ontology_events.get(parent_case_id, []), key=lambda row: str(row["event_id"]))
        mapped_ids = [str(row["event_id"]) for row in mapped]
        if mapped_ids != parent_events:
            raise ValueError(
                f"{parent_case_id}: accepted parent events do not exactly match ontology; "
                f"parent={parent_events} ontology={mapped_ids}"
            )
        source_path = Path(str(entry["scan_path"]))
        if source_path.name != Path(str(parent_summary["results"][ordinal - 1]["scan_path"])).name:
            raise ValueError(f"{parent_case_id}: parent-summary basename mismatch")
        event_bindings = []
        for row in mapped:
            ref = str(row["event_ref"])
            all_bound_refs.append(ref)
            event_bindings.append({
                "event_ref": ref,
                "event_id": str(row["event_id"]),
                "status": str(row["proposed_status"]),
                "line_convention": row.get("proposed_line_convention"),
                "physical_event_group_id": row.get("physical_event_group_id"),
                "canonical_line_owner_ref": row.get("canonical_line_owner_ref"),
                "emission_role": row.get("emission_role"),
            })
        records.append({
            "case_token": str(entry["case_token"]),
            "scan_sha256": scan_sha,
            "scan_path": str(source_path.resolve()),
            "scan_basename_audit": source_path.name,
            "parent_case_id": parent_case_id,
            "parent_semantic_path": str(parent_path.resolve()),
            "parent_semantic_sha256": _sha(parent_path),
            "parent_status": str(parent["status"]),
            "step_sha256": str(parent["step_sha256"]),
            "accepted_parent_event_ids": parent_events,
            "event_bindings": event_bindings,
            "ontology_coverage": "EXACT" if mapped else "EXACT_EMPTY_PARENT_ABSTENTION",
        })
    acceptance_hashes = {path.stem: _sha(path) for path in acceptance_paths}
    decision_counts = ontology.get("proposal_summary") or {}
    if sorted(all_bound_refs) != sorted(str(row["event_ref"]) for row in ontology.get("events") or []):
        raise ValueError("bound ontology union differs from V2 ontology")
    if decision_counts.get("event_count") != 63 or decision_counts.get("physical_event_group_count") != 53:
        raise ValueError("unexpected V2 ontology dimensions")
    return {
        "schema": "SHERMAN_QC_EXP_CAD_0010_CASE_BINDING_MANIFEST_V1",
        "status": "LOCKED",
        "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL",
        "generalization_evidence": False,
        "promotion_eligible": False,
        "ai_adjudicated_case_mapping_used": True,
        "gold_or_expected_counts_present": False,
        "selection_key": "SCAN_SHA256_ONLY",
        "filenames_are_audit_only": True,
        "frozen_inputs": {
            "source_slice_sha256": _sha(slice_path),
            "parent_summary_sha256": _sha(parent_summary_path),
            "ontology_v2_sha256": _sha(ontology_path),
            "acceptance_sha256": acceptance_hashes,
        },
        "ontology_summary": {
            "event_count": 63,
            "qc_bend_count": 60,
            "not_qc_bend_count": 3,
            "physical_event_group_count": 53,
        },
        "case_count": len(records),
        "records": records,
        "integrity": {
            "all_scan_sha_unique": True,
            "all_step_sha_match_slice": True,
            "all_parent_event_sets_exactly_bound": True,
            "empty_parent_cases_fail_closed": True,
            "bound_event_union_complete": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--parent-cases-root", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        slice_path=args.slice,
        parent_summary_path=args.parent_summary,
        parent_cases_root=args.parent_cases_root,
        ontology_path=args.ontology,
        acceptance_paths=args.acceptance,
    )
    _atomic(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "cases": payload["case_count"],
        "events": payload["ontology_summary"]["event_count"],
        "groups": payload["ontology_summary"]["physical_event_group_count"],
        "output": str(args.output.resolve()),
        "sha256": _sha(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
