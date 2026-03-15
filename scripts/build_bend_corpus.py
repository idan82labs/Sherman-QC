#!/usr/bin/env python3
"""
Build an ordered bend-inspection corpus from Downloads.

Creates a dedicated corpus folder with:
- global inventory JSON/Markdown
- per-part folders
- symlinked source assets grouped into cad/scans/drawings/specs
- metadata + label templates for later refinement

The corpus is intended for:
- detector improvement
- regression selection
- partial/full/bad-part benchmarking
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
BACKEND_DIR = ROOT / "backend"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from bend_improvement_loop import PartDataset, ScanFile, discover_parts  # noqa: E402
from bend_inspection_pipeline import load_expected_bend_overrides  # noqa: E402


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "file"


def _unlink_or_remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _unlink_or_remove(dst)
    os.symlink(src, dst)


def _scan_kind(scan: ScanFile) -> str:
    name = scan.path.stem.lower().replace(" ", "_")
    if "before" in name:
        return "pre_bend"
    if scan.state == "partial":
        return "partial"
    return "full_or_final"


def _part_readiness(dataset: PartDataset, expected_override: Optional[int]) -> Dict[str, Any]:
    partials = [s for s in dataset.scan_files if s.state == "partial"]
    fulls = [s for s in dataset.scan_files if s.state == "full"]
    has_specs = bool(dataset.spec_files)
    has_drawings = bool(dataset.drawing_files)
    known_expected = expected_override is not None

    ready_regression = bool(dataset.cad_files and dataset.scan_files)
    ready_partial = ready_regression and bool(partials)
    ready_full = ready_regression and bool(fulls)
    needs_confirmation = not known_expected

    reasons: List[str] = []
    if not dataset.cad_files:
        reasons.append("missing CAD")
    if not dataset.scan_files:
        reasons.append("missing scans")
    if not has_drawings:
        reasons.append("no drawing linked")
    if not has_specs:
        reasons.append("no spec sheet linked")
    if not known_expected:
        reasons.append("unknown expected bend count")

    return {
        "ready_regression": ready_regression,
        "ready_partial_regression": ready_partial,
        "ready_full_regression": ready_full,
        "has_drawings": has_drawings,
        "has_specs": has_specs,
        "known_expected_bends": known_expected,
        "needs_confirmation": needs_confirmation,
        "reasons": reasons,
    }


def _load_normalized_expected_overrides() -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for raw_key, value in load_expected_bend_overrides().items():
        part_key = _safe_part_key(raw_key)
        if part_key and value > 0:
            merged[part_key] = int(value)

    for extra_path in sorted((ROOT / "output").glob("*expected_override*.json")):
        try:
            raw = json.loads(extra_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for row in raw:
            if not isinstance(row, dict):
                continue
            part = row.get("part")
            expected = row.get("drawing_expected_bends")
            part_key = _safe_part_key(str(part or ""))
            try:
                expected_int = int(expected)
            except Exception:
                continue
            if part_key and expected_int > 0:
                merged[part_key] = expected_int
    return merged


def _safe_part_key(raw: str) -> Optional[str]:
    matches = re.findall(r"\d{6,}", raw or "")
    if not matches:
        return None
    return max(matches, key=len)


def _load_seed_labels() -> Dict[str, Any]:
    seed_path = ROOT / "data" / "bend_corpus_seed_labels.json"
    if not seed_path.exists():
        return {}
    try:
        raw = json.loads(seed_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _apply_seed_scan_overrides(dataset: PartDataset, seed_labels: Dict[str, Any]) -> PartDataset:
    entry = seed_labels.get(dataset.part_key)
    if not isinstance(entry, dict):
        return dataset
    scan_overrides = entry.get("scan_overrides")
    if not isinstance(scan_overrides, dict):
        return dataset

    updated_scans: List[ScanFile] = []
    for scan in dataset.scan_files:
        override = scan_overrides.get(scan.path.name)
        if not isinstance(override, dict):
            updated_scans.append(scan)
            continue
        state = str(override.get("state", scan.state))
        sequence = int(override.get("sequence", scan.sequence))
        sequence_confident = bool(override.get("sequence_confident", scan.sequence_confident))
        updated_scans.append(
            ScanFile(
                path=scan.path,
                state=state,
                sequence=sequence,
                sequence_confident=sequence_confident,
                mtime=scan.mtime,
            )
        )
    return PartDataset(
        part_key=dataset.part_key,
        cad_files=dataset.cad_files,
        scan_files=updated_scans,
        drawing_files=dataset.drawing_files,
        spec_files=dataset.spec_files,
    )


def _build_part_record(dataset: PartDataset, expected_override: Optional[int], seed_labels: Dict[str, Any]) -> Dict[str, Any]:
    readiness = _part_readiness(dataset, expected_override)
    partials = [s for s in dataset.scan_files if s.state == "partial"]
    fulls = [s for s in dataset.scan_files if s.state == "full"]
    seed_entry = seed_labels.get(dataset.part_key) if isinstance(seed_labels.get(dataset.part_key), dict) else {}
    return {
        "part_key": dataset.part_key,
        "expected_bends_override": expected_override,
        "seed_notes": seed_entry.get("notes") if isinstance(seed_entry, dict) else None,
        "cad_count": len(dataset.cad_files),
        "scan_count": len(dataset.scan_files),
        "partial_scan_count": len(partials),
        "full_scan_count": len(fulls),
        "drawing_count": len(dataset.drawing_files),
        "spec_count": len(dataset.spec_files),
        "readiness": readiness,
        "cad_files": [
            {
                "name": p.name,
                "path": str(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "size_bytes": p.stat().st_size,
            }
            for p in dataset.cad_files
        ],
        "scan_files": [
            {
                "name": scan.path.name,
                "path": str(scan.path),
                "state": scan.state,
                "kind": _scan_kind(scan),
                "sequence": scan.sequence,
                "sequence_confident": scan.sequence_confident,
                "mtime": datetime.fromtimestamp(scan.path.stat().st_mtime).isoformat(),
                "size_bytes": scan.path.stat().st_size,
            }
            for scan in dataset.scan_files
        ],
        "drawing_files": [
            {
                "name": p.name,
                "path": str(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "size_bytes": p.stat().st_size,
            }
            for p in dataset.drawing_files
        ],
        "spec_files": [
            {
                "name": p.name,
                "path": str(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "size_bytes": p.stat().st_size,
            }
            for p in dataset.spec_files
        ],
    }


def _write_labels_template(part_dir: Path, part_record: Dict[str, Any]) -> None:
    template = {
        "part_id": part_record["part_key"],
        "expected_total_bends": part_record["expected_bends_override"],
        "notes": part_record.get("seed_notes") or "",
        "scans": [
            {
                "filename": scan["name"],
                "state": scan["state"],
                "completed_bends": None,
                "completed_bend_ids": [],
                "all_completed_in_spec": None,
                "comments": "",
            }
            for scan in part_record["scan_files"]
        ],
    }
    (part_dir / "labels.template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _materialize_part_folder(part_dir: Path, part_record: Dict[str, Any]) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("cad", "scans", "drawings", "specs"):
        (part_dir / sub).mkdir(exist_ok=True)

    for idx, item in enumerate(part_record["cad_files"], start=1):
        src = Path(item["path"])
        dst = part_dir / "cad" / f"{idx:02d}_{_safe_name(src.name)}"
        _ensure_symlink(src, dst)

    for idx, item in enumerate(part_record["scan_files"], start=1):
        src = Path(item["path"])
        prefix = f"{idx:02d}_{item['state']}_{item['sequence']:02d}"
        dst = part_dir / "scans" / f"{prefix}_{_safe_name(src.name)}"
        _ensure_symlink(src, dst)

    for idx, item in enumerate(part_record["drawing_files"], start=1):
        src = Path(item["path"])
        dst = part_dir / "drawings" / f"{idx:02d}_{_safe_name(src.name)}"
        _ensure_symlink(src, dst)

    for idx, item in enumerate(part_record["spec_files"], start=1):
        src = Path(item["path"])
        dst = part_dir / "specs" / f"{idx:02d}_{_safe_name(src.name)}"
        _ensure_symlink(src, dst)

    (part_dir / "metadata.json").write_text(
        json.dumps(part_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_labels_template(part_dir, part_record)


def _write_summary_md(out_dir: Path, inventory: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Bend Corpus Inventory")
    lines.append("")
    lines.append(f"Generated: `{inventory['generated_at']}`")
    lines.append(f"Source: `{inventory['source_dir']}`")
    lines.append(f"Corpus root: `{inventory['corpus_root']}`")
    lines.append("")
    counts = inventory["counts"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Parts discovered: `{counts['parts']}`")
    lines.append(f"- CAD files linked: `{counts['cad_files']}`")
    lines.append(f"- Scan files linked: `{counts['scan_files']}`")
    lines.append(f"- Drawings linked: `{counts['drawing_files']}`")
    lines.append(f"- Specs linked: `{counts['spec_files']}`")
    lines.append(f"- Ready regression parts: `{counts['ready_regression_parts']}`")
    lines.append(f"- Ready partial-regression parts: `{counts['ready_partial_parts']}`")
    lines.append(f"- Ready full-regression parts: `{counts['ready_full_parts']}`")
    lines.append(f"- Parts needing bend-count confirmation: `{counts['needs_confirmation_parts']}`")
    lines.append("")
    lines.append("## Parts")
    lines.append("")
    lines.append("| Part | CAD | Scans | Partial | Full | Drawings | Specs | Expected bends | Ready | Needs confirmation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for part in inventory["parts"]:
        ready = "yes" if part["readiness"]["ready_regression"] else "no"
        needs = "yes" if part["readiness"]["needs_confirmation"] else "no"
        expected = part["expected_bends_override"] if part["expected_bends_override"] is not None else "-"
        lines.append(
            f"| `{part['part_key']}` | {part['cad_count']} | {part['scan_count']} | "
            f"{part['partial_scan_count']} | {part['full_scan_count']} | {part['drawing_count']} | "
            f"{part['spec_count']} | {expected} | {ready} | {needs} |"
        )
    lines.append("")
    lines.append("## Immediate Use")
    lines.append("")
    for part in inventory["parts"]:
        readiness = part["readiness"]
        if readiness["ready_regression"]:
            lines.append(
                f"- `{part['part_key']}`: regression-ready; "
                f"partials={part['partial_scan_count']}, fulls={part['full_scan_count']}, "
                f"drawings={part['drawing_count']}, specs={part['spec_count']}, "
                f"expected_bends={part['expected_bends_override'] if part['expected_bends_override'] is not None else 'unknown'}"
            )
    lines.append("")
    lines.append("## Confirmation Targets")
    lines.append("")
    for part in inventory["parts"]:
        readiness = part["readiness"]
        if readiness["needs_confirmation"]:
            lines.append(
                f"- `{part['part_key']}`: {', '.join(readiness['reasons']) or 'needs manual review'}"
            )
    (out_dir / "dataset_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_operational_views(out_dir: Path, inventory: Dict[str, Any]) -> None:
    ready_regression = [
        {
            "part_key": part["part_key"],
            "expected_bends_override": part["expected_bends_override"],
            "partial_scan_count": part["partial_scan_count"],
            "full_scan_count": part["full_scan_count"],
            "drawing_count": part["drawing_count"],
            "spec_count": part["spec_count"],
            "metadata_path": str(out_dir / "parts" / part["part_key"] / "metadata.json"),
            "labels_template_path": str(out_dir / "parts" / part["part_key"] / "labels.template.json"),
        }
        for part in inventory["parts"]
        if part["readiness"]["ready_regression"]
    ]
    confirmation_queue = [
        {
            "part_key": part["part_key"],
            "expected_bends_override": part["expected_bends_override"],
            "reasons": part["readiness"]["reasons"],
            "metadata_path": str(out_dir / "parts" / part["part_key"] / "metadata.json"),
        }
        for part in inventory["parts"]
        if part["readiness"]["needs_confirmation"]
    ]
    (out_dir / "ready_regression.json").write_text(
        json.dumps(ready_regression, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "confirmation_queue.json").write_text(
        json.dumps(confirmation_queue, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# Confirmation Queue", ""]
    for item in confirmation_queue:
        lines.append(f"- `{item['part_key']}`: {', '.join(item['reasons'])}")
    (out_dir / "confirmation_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_corpus(downloads_dir: Path, out_dir: Path) -> Dict[str, Any]:
    datasets = discover_parts(downloads_dir)
    overrides = _load_normalized_expected_overrides()
    seed_labels = _load_seed_labels()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)

    parts: List[Dict[str, Any]] = []
    for dataset in datasets:
        dataset = _apply_seed_scan_overrides(dataset, seed_labels)
        expected_override = overrides.get(dataset.part_key)
        if dataset.part_key in seed_labels and isinstance(seed_labels[dataset.part_key], dict):
            seed_expected = seed_labels[dataset.part_key].get("expected_total_bends")
            if isinstance(seed_expected, int) and seed_expected > 0:
                expected_override = seed_expected
        part_record = _build_part_record(dataset, expected_override, seed_labels)
        _materialize_part_folder(parts_dir / dataset.part_key, part_record)
        parts.append(part_record)

    inventory = {
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(downloads_dir),
        "corpus_root": str(out_dir),
        "counts": {
            "parts": len(parts),
            "cad_files": sum(p["cad_count"] for p in parts),
            "scan_files": sum(p["scan_count"] for p in parts),
            "drawing_files": sum(p["drawing_count"] for p in parts),
            "spec_files": sum(p["spec_count"] for p in parts),
            "ready_regression_parts": sum(1 for p in parts if p["readiness"]["ready_regression"]),
            "ready_partial_parts": sum(1 for p in parts if p["readiness"]["ready_partial_regression"]),
            "ready_full_parts": sum(1 for p in parts if p["readiness"]["ready_full_regression"]),
            "needs_confirmation_parts": sum(1 for p in parts if p["readiness"]["needs_confirmation"]),
        },
        "parts": parts,
    }

    (out_dir / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary_md(out_dir, inventory)
    _write_operational_views(out_dir, inventory)
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ordered bend corpus from Downloads")
    parser.add_argument(
        "--downloads",
        default=str(Path.home() / "Downloads"),
        help="Source directory to inventory",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "bend_corpus"),
        help="Output corpus directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_corpus(Path(args.downloads), Path(args.out))
    print(json.dumps({
        "corpus_root": inventory["corpus_root"],
        "parts": inventory["counts"]["parts"],
        "ready_regression_parts": inventory["counts"]["ready_regression_parts"],
        "needs_confirmation_parts": inventory["counts"]["needs_confirmation_parts"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
