"""Code fingerprints for zs_lane_v1: every source file the lane executes (its own files, the reused Z3 / Z2 / D1
modules, the harness guard and the eval_lines writer), the model files, and the scorer files.

The fingerprint is sha256 over (relative path, file sha256) pairs in sorted order. ``check_loaded`` asserts that
every repo module loaded in the process is covered, so an unlisted dependency cannot slip past the freeze.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from research.zeroshot.zs_lane_v1.core import REPO

LANE_FILES = [
    "research/zeroshot/__init__.py",
    "research/zeroshot/zs_lane_v1/__init__.py",
    "research/zeroshot/zs_lane_v1/core.py",
    "research/zeroshot/zs_lane_v1/lane.py",
    "research/zeroshot/zs_lane_v1/fingerprint.py",
    "research/zeroshot/z3_slicing/__init__.py",
    "research/zeroshot/z3_slicing/slicer.py",
    "research/zeroshot/z3_slicing/selector.py",
    "research/zeroshot/z3_slicing/pooled.py",
    "research/zeroshot/z3_slicing/union.py",
    "research/zeroshot/z3_slicing/custody.py",
    "research/zeroshot/z3_slicing/zscore.py",
    "research/zeroshot/z3_slicing/gate.py",
    "research/zeroshot/z3_slicing/score_run.py",
    "research/zeroshot/z2_self_recipe/__init__.py",
    "research/zeroshot/z2_self_recipe/run.py",
    "research/zeroshot/z2_self_recipe/segment.py",
    "research/zeroshot/z2_self_recipe/graph.py",
    "research/zeroshot/z2_self_recipe/common.py",
    "research/zeroshot/d1_two_skin/__init__.py",
    "research/zeroshot/d1_two_skin/two_skin.py",
    "research/zeroshot/d1_two_skin/custody.py",
    "research/zeroshot/common/__init__.py",
    "research/zeroshot/common/gold_guard.py",
    "research/bench_labeling/custody.py",
    "research/broad_baseline/bb_common.py",
    "research/broad_baseline/score.py",
    "research/broad_baseline/score_broad.py",
    "scripts/evaluate_f1_against_manual_gold.py",
    # E3 metrology used by Z2 (frozen engine code, imported read-only) and package inits
    "contracts/__init__.py",
    "contracts/production/__init__.py",
    "contracts/production/canonical.py",
    "contracts/production/characteristics.py",
    "contracts/production/common.py",
    "contracts/production/compiled_part.py",
    "contracts/production/consistency.py",
    "contracts/production/inspection_result.py",
    "contracts/production/layout.py",
    "contracts/production/line_job.py",
    "contracts/production/part_recipe.py",
    "contracts/production/states.py",
    "domains/__init__.py",
    "domains/production/__init__.py",
    "domains/production/engine/__init__.py",
    "domains/production/engine/inspect/__init__.py",
    "domains/production/engine/inspect/articulated.py",
    "domains/production/engine/inspect/cloud.py",
    "domains/production/engine/inspect/config.py",
    "domains/production/engine/inspect/decode.py",
    "domains/production/engine/inspect/engine.py",
    "domains/production/engine/inspect/geom.py",
    "domains/production/engine/inspect/metrology.py",
    "domains/production/engine/inspect/outputs.py",
    "domains/production/engine/inspect/part.py",
    "domains/production/engine/inspect/registration.py",
    "domains/production/engine/inspect/result.py",
    "domains/production/engine/inspect/scan_io.py",
    "research/__init__.py",
    "research/bench_labeling/__init__.py",
    "research/broad_baseline/gold11_cases.py",
]
SCORER_FILES = [
    "research/broad_baseline/score.py",
    "research/broad_baseline/score_broad.py",
    "research/broad_baseline/bb_common.py",
    "scripts/evaluate_f1_against_manual_gold.py",
    "research/zeroshot/zs_lane_v1/scoring.py",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_hashes(files: list[str]) -> dict[str, str]:
    return {f: sha256_file(REPO / f) for f in sorted(files)}


def combined(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(k.encode())
        h.update(b"\0")
        h.update(hashes[k].encode())
        h.update(b"\n")
    return h.hexdigest()


def lane_fingerprint() -> dict:
    fh = file_hashes(LANE_FILES)
    return {"files": fh, "sha256": combined(fh)}


def scorer_fingerprint() -> dict:
    fh = file_hashes(SCORER_FILES)
    return {"files": fh, "sha256": combined(fh)}


def loaded_repo_files() -> list[str]:
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(REPO)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in (".venv", ".venv-f1-structured"):
            continue
        out.add(rel.as_posix())
    return sorted(out)


def check_loaded(extra_allowed: tuple[str, ...] = ()) -> list[str]:
    """Repo source files loaded in this process that the lane fingerprint does not cover."""
    cover = set(LANE_FILES) | set(SCORER_FILES) | set(extra_allowed)
    return [f for f in loaded_repo_files() if f not in cover]
