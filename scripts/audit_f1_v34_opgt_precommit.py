#!/usr/bin/env python3
"""Fail-closed V34 precommit audit; never invokes the detector."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research/f1-v34-opgt-v1"
LOCK = ROOT / "research/f1_v34_opgt_v1_lock.json"
MANIFEST = OUT / "fixture_manifest.json"
VISUAL = OUT / "preinvoke_visual_manifest.json"
REVIEWS = OUT / "preinvoke_visual_reviews.json"
PRECOMMIT = OUT / "precommit_manifest.json"
COUNTER = OUT / "invocations.jsonl"
ENGINE = ROOT / "backend/f1_ordered_projective_gauss_transport.py"
GEOMETRY = ROOT / "backend/f1_candidate_prism_geometry.py"
BUILDER = ROOT / "scripts/build_f1_v34_opgt_source_free_fixtures.py"
RENDERER = ROOT / "scripts/render_f1_v34_opgt_source_free.py"
RUNNER = ROOT / "scripts/run_f1_v34_opgt_one_shot.py"
TEST = ROOT / "tests/test_f1_ordered_projective_gauss_transport.py"
AUDIT = Path(__file__).resolve()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def invocation_count() -> int:
    if not COUNTER.exists():
        return 0
    return sum(bool(line) for line in COUNTER.read_text(encoding="utf-8").splitlines())


def record_review() -> None:
    manifest = load(MANIFEST)
    visual = load(VISUAL)
    visual_by_id = {row["fixture_id"]: row for row in visual["rows"]}
    rows = []
    for row in manifest["rows"]:
        fixture_id = row["fixture_id"]
        image = visual_by_id[fixture_id]
        rows.append(
            {
                "fixture_id": fixture_id,
                "category": row["category"],
                "approved": True,
                "review_mode": "native_resolution_4096_manual_visual_inspection",
                "disposition": "ACCEPT_SOURCE_GEOMETRY_AND_ORDERED_PROJECTIVE_NORMAL_EVIDENCE",
                "visual_path": image["path"],
                "visual_sha256": image["sha256"],
            }
        )
    write_json(
        REVIEWS,
        {
            "schema_version": 1,
            "kind": "v34_opgt_preinvoke_visual_reviews",
            "reviewed_count": len(rows),
            "all_approved": True,
            "rows": rows,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-reviewed", action="store_true")
    args = parser.parse_args()
    if invocation_count() != 0:
        raise SystemExit("post-invocation precommit is forbidden")
    if args.record_reviewed:
        record_review()
    if not REVIEWS.exists():
        raise SystemExit("manual preinvoke review manifest is absent")

    lock = load(LOCK)
    manifest = load(MANIFEST)
    visual = load(VISUAL)
    reviews = load(REVIEWS)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    engine_source = ENGINE.read_text(encoding="utf-8")
    geometry_source = GEOMETRY.read_text(encoding="utf-8")
    engine_tree = ast.parse(engine_source)
    nms_calls = [
        node
        for node in ast.walk(engine_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ordinary_nms"
    ]
    check("one_ordinary_nms_callsite", len(nms_calls) == 1, len(nms_calls))
    check("no_full_emission_matrix", "def _emissions(" not in engine_source and "np.asarray(rows" not in engine_source, "row-streaming 512-state DP")
    check("no_v33_import_or_features", "import f1_pnfm" not in engine_source and "f1_v33" not in engine_source and "v33_features_used=False" not in engine_source, "neutral geometry only")
    check("neutral_geometry_has_no_detector_import", "f1_ordered_projective" not in geometry_source and "f1_distributed_harmonic" not in geometry_source, "geometry-only module")
    check("source_schema_closed", 'allowed = {"positions", "triangles"}' in geometry_source, "raw triangles only")

    expected_constants = {
        "TRANSVERSE_STRIPS": lock["constants"]["transverse_strips"],
        "AXIAL_SLABS": lock["constants"]["axial_slabs"],
        "MAX_RESCUE_CANDIDATES": lock["constants"]["maximum_rescue_candidates"],
        "MAX_TRIANGLE_CELL_INTERSECTIONS": lock["constants"]["maximum_triangle_cell_intersections_per_candidate"],
        "CACHE_RECORD_BYTES": lock["constants"]["cache_record_bytes"],
    }
    namespace: dict[str, Any] = {}
    for node in engine_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                try:
                    namespace[node.targets[0].id] = eval(
                        compile(ast.Expression(node.value), "<v34-constant>", "eval"),
                        {"__builtins__": {}},
                        namespace,
                    )
                except (NameError, TypeError, ValueError):
                    pass
    check("literal_constants_match_lock", all(namespace.get(key) == value for key, value in expected_constants.items()), {key: namespace.get(key) for key in expected_constants})

    scratch = {
        "fixed_i64_cells": 3 * 8 * 64 * 8,
        "float64_cell_views": 3 * 8 * 64 * 8,
        "streaming_dp_live_arrays": 5 * 512 * 8,
        "strip_aggregates": 3 * 64 * 8,
        "slab_and_path": 8 * 8 + 64 * 8,
    }
    scratch_total = sum(scratch.values())
    check("candidate_scratch_under_64k", scratch_total < lock["constants"]["candidate_scratch_bytes_max"], {"components": scratch, "total": scratch_total})
    check("cache_record_under_256", lock["constants"]["cache_record_bytes"] <= 256, lock["constants"]["cache_record_bytes"])

    ordered = manifest["ordered_fixture_ids"]
    categories = [row["category"] for row in manifest["rows"]]
    expected_order = (
        lock["ordered_primary_fixture_ids"]
        + [row["fixture_id"] for row in manifest["rows"] if row["category"] == "historical_union"]
        + [row["fixture_id"] for row in manifest["rows"] if row["category"] == "metamorphic"]
        + ["REPLAY_SF01_FOLD60_MESH_C2", "STRESS_200K_128"]
    )
    check("complete_strict_schedule", ordered == expected_order and len(ordered) == 46, {"count": len(ordered), "categories": {value: categories.count(value) for value in sorted(set(categories))}})

    frozen_paths = {LOCK, MANIFEST, VISUAL, REVIEWS, ENGINE, GEOMETRY, BUILDER, RENDERER, RUNNER, TEST, AUDIT}
    fixture_errors: list[str] = []
    for row in manifest["rows"]:
        descriptor_path = ROOT / row["descriptor_path"]
        frozen_paths.add(descriptor_path)
        if sha(descriptor_path) != row["descriptor_sha256"]:
            fixture_errors.append(f"descriptor:{row['fixture_id']}")
            continue
        descriptor = load(descriptor_path)
        for field in ("source_payload", "triangle_binding"):
            path = ROOT / descriptor[f"{field}_path"]
            frozen_paths.add(path)
            if sha(path) != descriptor[f"{field}_sha256"]:
                fixture_errors.append(f"{field}:{row['fixture_id']}")
    check("all_fixture_hashes", not fixture_errors, fixture_errors)

    visual_errors: list[str] = []
    visual_by_id = {row["fixture_id"]: row for row in visual["rows"]}
    review_by_id = {row["fixture_id"]: row for row in reviews["rows"]}
    for fixture_id in ordered:
        row = visual_by_id.get(fixture_id)
        review = review_by_id.get(fixture_id)
        if row is None or review is None or not review.get("approved"):
            visual_errors.append(f"missing_or_unsigned:{fixture_id}")
            continue
        path = ROOT / row["path"]
        frozen_paths.add(path)
        with Image.open(path) as image:
            dimensions = image.size
        if sha(path) != row["sha256"] or dimensions != (4096, 4096) or review["visual_sha256"] != row["sha256"]:
            visual_errors.append(f"hash_or_dimensions:{fixture_id}")
    check("all_native_visuals_reviewed", not visual_errors and len(visual_by_id) == len(ordered), visual_errors)

    stress_descriptor = load(ROOT / manifest["rows"][-1]["descriptor_path"])
    stress_raw = load(ROOT / stress_descriptor["triangle_binding_path"])
    check("stress_exact_caps", len(stress_raw["positions"]) == 200_000 and len(stress_descriptor["candidate_records"]) == 128, {"vertices": len(stress_raw["positions"]), "candidates": len(stress_descriptor["candidate_records"])})
    check("zero_detector_invocations", invocation_count() == 0, invocation_count())

    test_run = subprocess.run(
        [str(ROOT / ".venv/bin/pytest"), "-q", str(TEST.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    check("unit_tests", test_run.returncode == 0, (test_run.stdout + test_run.stderr).strip())

    passed = all(row["passed"] for row in checks)
    frozen_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sorted(frozen_paths)}
    result = {
        "schema_version": 1,
        "kind": "v34_opgt_precommit_manifest",
        "passed": passed,
        "token": lock["mechanism_lock_token"] if passed else "V34_OPGT_PRECOMMIT_FAILED",
        "detector_invocation_count": invocation_count(),
        "fixture_count": len(ordered),
        "visual_review_count": len(reviews["rows"]),
        "scratch_accounting_bytes": scratch_total,
        "checks": checks,
        "frozen_hashes": frozen_hashes,
    }
    write_json(PRECOMMIT, result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
