#!/usr/bin/env python3
"""Build a deterministic, prediction-hidden multiview review pack.

The input contract deliberately permits only scan paths and their frozen
SHA-256 identities.  Case names, predictions, sender counts, CAD, drawings,
labels, and corpus membership are neither accepted nor propagated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

try:
    from scripts.render_scan_multiview_audit import render_scan_multiview
except ModuleNotFoundError:  # Direct ``python scripts/...py`` invocation.
    from render_scan_multiview_audit import render_scan_multiview


SCHEMA = "shermanqc.opaque_inference_manifest.v1"
OUTPUT_SCHEMA = "shermanqc.prediction_hidden_multiview_render_pack.v1"
CASE_KEYS = frozenset({"scan_path", "sha256"})
TOP_LEVEL_KEYS = frozenset({"schema", "cohort_date", "cases"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def validate_opaque_manifest(
    path: Path,
    *,
    expected_count: int = 28,
    expected_cohort_date: str = "2026-08-30",
) -> tuple[Mapping[str, Any], list[tuple[Path, str]]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_KEYS:
        raise ValueError("opaque manifest has unexpected top-level fields")
    if payload["schema"] != SCHEMA:
        raise ValueError("opaque manifest schema mismatch")
    if payload["cohort_date"] != expected_cohort_date:
        raise ValueError("opaque manifest cohort date mismatch")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise ValueError(f"opaque manifest must contain exactly {expected_count} cases")

    validated: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for index, row in enumerate(cases, start=1):
        if not isinstance(row, dict) or set(row) != CASE_KEYS:
            raise ValueError(f"case {index} must contain scan_path and sha256 only")
        scan_path = Path(row["scan_path"]).expanduser().resolve()
        declared = str(row["sha256"]).lower()
        if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
            raise ValueError(f"case {index} has invalid sha256")
        if scan_path.suffix.lower() != ".ply":
            raise ValueError(f"case {index} is not a PLY scan")
        if scan_path in seen_paths or declared in seen_hashes:
            raise ValueError("opaque manifest contains a duplicate scan identity")
        seen_paths.add(scan_path)
        seen_hashes.add(declared)
        validated.append((scan_path, declared))
    return payload, validated


def build_render_pack(
    manifest_path: Path,
    output_root: Path,
    *,
    expected_count: int = 28,
    expected_cohort_date: str = "2026-08-30",
    max_points: int = 260_000,
    renderer: Callable[..., Path] = render_scan_multiview,
) -> tuple[Path, str]:
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix or overwrite render pack: {output_root}")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    _, cases = validate_opaque_manifest(
        manifest_path,
        expected_count=expected_count,
        expected_cohort_date=expected_cohort_date,
    )
    source_manifest_sha256 = sha256_file(manifest_path)

    # Hash-lock every source before rendering any case.
    for index, (scan_path, declared) in enumerate(cases, start=1):
        if not scan_path.is_file():
            raise FileNotFoundError(f"case {index} scan is missing: {scan_path}")
        actual = sha256_file(scan_path)
        if actual != declared:
            raise ValueError(f"case {index} scan hash drift: {actual} != {declared}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        rendered = []
        for index, (scan_path, scan_sha256) in enumerate(cases, start=1):
            opaque_id = f"O{index:03d}"
            relative = Path(opaque_id) / "multiview.png"
            destination = staging / relative
            renderer(
                scan_path,
                destination,
                max_points=max_points,
                display_name=opaque_id,
                write_frame=False,
            )
            if not destination.is_file():
                raise RuntimeError(f"renderer did not create {relative}")
            rendered.append(
                {
                    "opaque_case_id": opaque_id,
                    "source_scan_sha256": scan_sha256,
                    "png_path": relative.as_posix(),
                    "png_sha256": sha256_file(destination),
                    "png_size_bytes": destination.stat().st_size,
                }
            )

        renderer_source = Path(__file__).with_name("render_scan_multiview_audit.py")
        output_manifest = {
            "schema": OUTPUT_SCHEMA,
            "cohort_date": expected_cohort_date,
            "status": "prediction_hidden_raw_geometry_multiview_pack",
            "source_manifest_sha256": source_manifest_sha256,
            "case_count": len(rendered),
            "review_boundary": {
                "predictions_included": False,
                "sender_counts_included": False,
                "labels_included": False,
                "cad_included": False,
                "drawings_included": False,
                "source_filenames_rendered": False,
            },
            "renderer": {
                "implementation": "scripts/render_scan_multiview_audit.py",
                "implementation_sha256": sha256_file(renderer_source),
                "max_points": max_points,
                "frame_sidecars_written": False,
            },
            "cases": rendered,
        }
        manifest_output = staging / "render_pack_manifest.json"
        manifest_output.write_bytes(canonical_json_bytes(output_manifest))
        manifest_sha256 = sha256_file(manifest_output)
        (staging / "render_pack_manifest.sha256").write_text(
            f"{manifest_sha256}  render_pack_manifest.json\n", encoding="ascii"
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / "render_pack_manifest.json", manifest_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opaque-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=28)
    parser.add_argument("--cohort-date", default="2026-08-30")
    parser.add_argument("--max-points", type=int, default=260_000)
    args = parser.parse_args()
    manifest, digest = build_render_pack(
        args.opaque_manifest,
        args.output_root,
        expected_count=args.expected_count,
        expected_cohort_date=args.cohort_date,
        max_points=args.max_points,
    )
    print(json.dumps({"manifest": str(manifest), "manifest_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
