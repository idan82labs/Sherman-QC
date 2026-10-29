#!/usr/bin/env python3
"""Build the truth-free post-V51-protocol-invalidation Pro packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output/professor_package/Sherman_QC_F1_Post_V51_Protocol_Invalidation_20260814_v1"
DEFAULT_ARCHIVE = DEFAULT_OUTPUT_DIR.with_suffix(".zip")
DEFAULT_AUDIT = ROOT / "output/research/f1-post-v51-protocol-invalidation-professor-20260814/package_integrity_audit.json"

FILES = {
    "00_GPT_PRO_PROMPT.md": "docs/research/2026-08-14_f1_post_v51_protocol_invalidation_professor_packet.md",
    "01_PROTOCOL_INVALIDATION.md": "docs/research/2026-08-14_f1_v51_gold37_blind_once_protocol_invalidation.md",
    "02_INVOCATION_EVENTS.jsonl": "output/research/f1-v51-gold37-blind-once-20260814/invocation_events.jsonl",
    "03_V51_AUTHORIZING_RULING_VERBATIM.md": "docs/research/2026-08-14_gpt56_sol_pro_v51_transfer_ruling_verbatim.md",
    "04_V51_ORIGINAL_PROMPT.md": "docs/research/2026-08-14_f1_v51_post_prospective_professor_packet.md",
    "05_V46_V51_PROSPECTIVE_RECORD.md": "docs/research/2026-08-14_f1_v46_v51_prospective_action_cycle.md",
    "06_AGGREGATE_READINESS_ONLY.json": "output/research/f1-v51-post-prospective-professor-20260814/locked_gold37_readiness.json",
    "source/scripts/select_f1_mesh_route_v16.py": "scripts/select_f1_mesh_route_v16.py",
    "source/backend/f1_two_sided_parent_support_guard_v43.py": "backend/f1_two_sided_parent_support_guard_v43.py",
    "source/backend/f1_v43_fragment_stitch_v44.py": "backend/f1_v43_fragment_stitch_v44.py",
    "source/backend/f1_longest_incumbent_ownership_v50.py": "backend/f1_longest_incumbent_ownership_v50.py",
    "source/backend/f1_route_independent_elongated_lane_v47.py": "backend/f1_route_independent_elongated_lane_v47.py",
    "source/backend/f1_nested_partial_duplicate_nms_v48.py": "backend/f1_nested_partial_duplicate_nms_v48.py",
    "source/backend/f1_short_span_cross_source_duplicate_nms_v49.py": "backend/f1_short_span_cross_source_duplicate_nms_v49.py",
    "source/backend/f1_single_axis_parallel_return_recovery_v51.py": "backend/f1_single_axis_parallel_return_recovery_v51.py",
    "source/scripts/evaluate_f1_against_manual_gold.py": "scripts/evaluate_f1_against_manual_gold.py",
    "tests/test_f1_single_axis_parallel_return_recovery_v51.py": "tests/test_f1_single_axis_parallel_return_recovery_v51.py",
    "visuals/P06_V51_overlay.png": "output/research/f1-v49-prospective-50535000-04-v1/visuals/P06_v51_overlay.png",
    "visuals/P07_V51_overlay.png": "output/research/f1-v51-prospective-47555020-1-v1/visuals/P07_v51_overlay.png",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_package(*, output_dir: Path, archive_path: Path, audit_path: Path | None) -> dict[str, object]:
    missing = [name for name in FILES.values() if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing packet sources: {missing}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for packet_name, source_name in FILES.items():
        source = ROOT / source_name
        target = output_dir / packet_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": packet_name, "sha256": _sha(target), "size_bytes": target.stat().st_size, "source": source_name})
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_post_v51_protocol_invalidation_professor_packet_manifest",
        "research_only": True,
        "gold_truth_included": False,
        "gold_case_mapping_included": False,
        "per_case_gold_score_included": False,
        "raw_scans_included": False,
        "v51_gold37_rerun_requested": False,
        "file_count_excluding_manifest": len(rows),
        "files": rows,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(_canonical(manifest), encoding="utf-8")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(output_dir).as_posix())
            info.date_time = (2026, 8, 14, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    result = {
        **{key: value for key, value in manifest.items() if key != "files"},
        "packet_directory": str(output_dir),
        "archive_path": str(archive_path),
        "archive_sha256": _sha(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "manifest_sha256": _sha(manifest_path),
    }
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(_canonical(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    print(_canonical(build_package(output_dir=args.output_dir.resolve(), archive_path=args.archive.resolve(), audit_path=args.audit.resolve())), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
