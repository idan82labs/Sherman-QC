#!/usr/bin/env python3
"""Build and render an opaque 15-family physical development lane.

All sources are development-consumed.  This lane is never confirmatory and
cannot produce GPI-A/B.  Evaluator identity stays outside the public manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_source_only_prediction_audit import render


TOKEN = "PHYSICAL15_DEVELOPMENT_CONSUMED_V1"
TARGET = ROOT / "data/bend_corpus/development/2026-08-22_physical15_v1"

# opaque_id, evaluator family, source relative path, frozen source SHA-256
SOURCES = (
    ("D001", "49191023", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191023_03.ply", "28a25c37a928f36856d281ecef0d9653331ce961cb2abe548ad3eb566950ed6b"),
    ("D002", "24694000", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/24694000_02.ply", "8b72c3ff233f679b7262f182d32721c86eabbeae872b80ce709a688a5cd37e42"),
    ("D003", "49191054", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191054_02.ply", "24c57f2660b3016486b985e16c413f5b6e64559cb4e06c2895b7e36fd5e59296"),
    ("D004", "48340008", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/48340008_02.ply", "521e2ee69b21ab45d7a2c7a1c494dce47087df32cdae77f9eef0e8820f21840c"),
    ("D005", "49191032", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191032_00.ply", "891a8e4a2ee1adcfe40e5ef46a722cdb3196fb4a65b563cde1513dbd926ca9cf"),
    ("D006", "49191061", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191061_00.ply", "e0285cea8c56602614f276afb21bfb5b5d2b02726c065732d0d9ee09fe9e6d7f"),
    ("D007", "49191038", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191038_02.ply", "d741e1d515bfcf04729fe361ac6fd6127b9f8490f95214a7a23ed0ebdef3c174"),
    ("D008", "48333018", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/48333018_00.ply", "7d28946d9f49bfe38f7a3e9f5ba4e26e7a07d4c07bdf3c90e780c578e3a85897"),
    ("D009", "49191037", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191037_01.ply", "9b38b1167834c37e3af11cbbe08b18f9d8b584bcb478958cb0c7153cc739dadf"),
    ("D010", "48333006", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/48333006_01.ply", "b7bd53831ee277e0b4b57a88567f3c5fb2b56b68b7154c01ffe8d0cb318a574f"),
    ("D011", "48340007", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/48340007_01.ply", "c1d112196ae59f697dfad63aaa18f63380715766c52998a7e15e83bafc705a10"),
    ("D012", "49191030", "data/bend_corpus/incoming/2026-07-16-shaked-count-backed-ply/49191030_01.ply", "e218b16ed0701591db7f96bf206fc7c6012692a76609e12c05a79e132f5aa1e3"),
    ("D013", "50757000_02", "data/bend_corpus/incoming/2026-08-04_shaked_scan_quality_parts/parts/50757000_02/scans/50757000_02_V.ply", "777c64c88967868678d9b488bf89ca51d9f94c1accb1eb3a561c66115ce7c5f2"),
    ("D014", "45796000_A", "data/bend_corpus/incoming/2026-08-04_shaked_scan_quality_parts/parts/45796000_A/scans/45796000_A_FULL_SCAN.ply", "1268c72dcc3049dc18be947aec969fde5e9e28bccc47f5c380cbe852698d448f"),
    ("D015", "48811000_A", "data/bend_corpus/incoming/2026-08-04_shaked_scan_quality_parts/parts/48811000_A/scans/48811000_A.ply", "ff1667a2a80aa9a7a91fa0fa0ffd59bbe3ecb43c3609ca85e17a78c62c57727c"),
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _contact_sheet(paths: list[Path], output: Path) -> None:
    tiles = []
    for path in paths:
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((1500, 875), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1520, 930), "white")
            canvas.paste(tile, ((1520 - tile.width) // 2, 42))
            ImageDraw.Draw(canvas).text((24, 14), path.stem, fill="black")
            tiles.append(canvas)
    sheet = Image.new("RGB", (1520, 930 * len(tiles)), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, 930 * index))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def main() -> int:
    if TARGET.exists() and any(TARGET.iterdir()):
        raise FileExistsError(f"development lane already exists: {TARGET}")
    public_cases = []
    evaluator_cases = []
    render_paths: list[Path] = []
    empty_prediction = TARGET / "preexecution_visual/empty_predictions.json"
    _write_json(empty_prediction, {"source_only": True, "research_only": True, "predictions": []})

    for opaque_id, family_id, relative, expected_sha in SOURCES:
        source = ROOT / relative
        if not source.is_file() or _sha(source) != expected_sha:
            raise ValueError(f"source mismatch: {relative}")
        bundle = TARGET / "opaque" / opaque_id
        opaque_source = bundle / "source.ply"
        _link_or_copy(source, opaque_source)
        if _sha(opaque_source) != expected_sha:
            raise ValueError(f"opaque link mismatch: {opaque_id}")
        bundle_manifest = {
            "schema_version": 1,
            "token": TOKEN,
            "opaque_case_id": opaque_id,
            "source_only": True,
            "research_only": True,
            "confirmatory_eligible": False,
            "source_sha256": expected_sha,
        }
        _write_json(bundle / "manifest.json", bundle_manifest)
        public_cases.append({
            "opaque_case_id": opaque_id,
            "bundle_path": str(bundle),
            "source_sha256": expected_sha,
            "manifest_sha256": _sha(bundle / "manifest.json"),
        })
        evaluator_cases.append({
            "opaque_case_id": opaque_id,
            "family_id": family_id,
            "source_relative": relative,
            "source_sha256": expected_sha,
            "truth_status": "UNANNOTATED_SINGLE_REVIEWER_DEVELOPMENT_PENDING",
        })
        render_path = TARGET / "preexecution_visual" / f"{opaque_id}.png"
        render(geometry=opaque_source, prediction_path=empty_prediction, output=render_path, max_points=150_000)
        render_paths.append(render_path)

    public_manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_physical15_development_consumed_v1",
        "token": TOKEN,
        "source_only": True,
        "research_only": True,
        "confirmatory_eligible": False,
        "evidence_class": "development_consumed_physical",
        "cases": public_cases,
    }
    evaluator_manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_physical15_development_evaluator_mapping",
        "token": TOKEN,
        "family_count": 15,
        "case_count": 15,
        "independent_line_truth": False,
        "cases": evaluator_cases,
    }
    _write_json(TARGET / "public_manifest.json", public_manifest)
    _write_json(TARGET / "evaluator_only_manifest.json", evaluator_manifest)
    sheets = []
    for first in range(0, len(render_paths), 3):
        sheet = TARGET / "preexecution_visual" / f"contact_sheet_{first // 3 + 1:02d}.png"
        _contact_sheet(render_paths[first:first + 3], sheet)
        sheets.append({"path": str(sheet.relative_to(TARGET)), "sha256": _sha(sheet)})
    _write_json(TARGET / "preexecution_visual_manifest.json", {
        "schema_version": 1,
        "kind": "sherman_qc_f1_physical15_raw_visual_preexecution_manifest",
        "token": TOKEN,
        "case_ids": [row[0] for row in SOURCES],
        "individual_renders": [
            {"case_id": path.stem, "path": str(path.relative_to(TARGET)), "sha256": _sha(path)}
            for path in render_paths
        ],
        "contact_sheets": sheets,
        "visual_ruling": "PENDING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "detector_executed": False,
    })
    print(json.dumps({"target": str(TARGET), "cases": len(SOURCES), "contact_sheets": len(sheets)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
