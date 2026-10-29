#!/usr/bin/env python3
"""Apply a completed Codex multiview bend-label adjudication ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
DEFAULT_LEDGER = (
    ROOT / "data/bend_corpus/annotations/codex_manual_multiview_adjudication_20260803.json"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def apply(catalog_path: Path, ledger_path: Path) -> dict[str, Any]:
    catalog = _read(catalog_path)
    ledger = _read(ledger_path)
    pending = {
        str(case["case_id"]): case
        for case in catalog.get("cases") or []
        if not case.get("promotion_eligible")
    }
    decisions = {str(case["case_id"]): case for case in ledger.get("cases") or []}
    if set(decisions) != set(pending):
        raise ValueError(
            f"ledger coverage mismatch: missing={sorted(set(pending)-set(decisions))}, "
            f"extra={sorted(set(decisions)-set(pending))}"
        )

    evidence: dict[str, Any] = {}
    manifests: set[Path] = set()
    total_bends = 0
    for case_id in sorted(pending):
        case = pending[case_id]
        decision = decisions[case_id]
        if decision.get("decision") != "accept_all_present_bends":
            raise ValueError(f"{case_id}: unsupported decision")
        label_path = Path(str(case["source_labels_path"])).resolve()
        labels = _read(label_path)
        bends = list(labels.get("manual_bends") or [])
        expected = int(decision["expected_bend_count"])
        if len(bends) != expected or int(case["bend_count"]) != expected:
            raise ValueError(f"{case_id}: expected {expected}, found {len(bends)}")
        artifacts = []
        for key in ("overlay", "raw"):
            path = (ROOT / str(decision[key])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            width, height = _png_dimensions(path)
            if width < 3000 or height < 1800:
                raise ValueError(f"{case_id}: {key} resolution is only {width}x{height}")
            artifacts.append(
                {
                    "kind": key,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "width": width,
                    "height": height,
                }
            )
        for bend in bends:
            bend["review_decision"] = "accepted_codex_manual_multiview"
            bend["review_reason"] = str(decision["notes"])
        labels["manual_bends"] = bends
        labels["review_status"] = "codex_manual_multiview_complete"
        labels["authoritative_ground_truth"] = False
        labels["manual_visual_adjudication"] = {
            "reviewer": ledger["reviewer"],
            "reviewed_at": ledger["reviewed_at"],
            "ledger_path": str(ledger_path.resolve()),
            "ledger_sha256": _sha256(ledger_path),
            "decision": decision["decision"],
            "bend_ids": [str(bend["bend_id"]) for bend in bends],
            "notes": str(decision["notes"]),
            "visual_artifacts": artifacts,
        }
        _write(label_path, labels)
        manifest_raw = case.get("source_manifest_path")
        if manifest_raw:
            manifests.add(Path(str(manifest_raw)).resolve())
        total_bends += len(bends)
        evidence[case_id] = {
            "label_path": str(label_path),
            "label_sha256": _sha256(label_path),
            "bend_count": len(bends),
            "visual_artifacts": artifacts,
        }

    for manifest_path in sorted(manifests):
        manifest = _read(manifest_path)
        manifest["review_status"] = "codex_manual_multiview_complete"
        manifest["authoritative_ground_truth"] = False
        manifest["manual_visual_adjudication"] = {
            "reviewer": ledger["reviewer"],
            "reviewed_at": ledger["reviewed_at"],
            "ledger_path": str(ledger_path.resolve()),
            "ledger_sha256": _sha256(ledger_path),
        }
        _write(manifest_path, manifest)

    resolved = {
        "schema_version": 1,
        "kind": "sherman_qc_codex_manual_multiview_adjudication_resolved",
        "reviewer": ledger["reviewer"],
        "reviewed_at": ledger["reviewed_at"],
        "enterprise_promotion_eligible": False,
        "case_count": len(evidence),
        "bend_count": total_bends,
        "cases": evidence,
    }
    output = ledger_path.with_name("codex_manual_multiview_adjudication_resolved_20260803.json")
    _write(output, resolved)
    return {"resolved": str(output), "case_count": len(evidence), "bend_count": total_bends}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    print(json.dumps(apply(args.catalog.resolve(), args.ledger.resolve()), indent=2))


if __name__ == "__main__":
    main()
