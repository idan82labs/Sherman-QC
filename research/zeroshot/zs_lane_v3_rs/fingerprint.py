"""Code fingerprint for zs_lane_v3_rs: the Rust crate (Cargo.toml, Cargo.lock, src/*.rs, the golden self-test set
golden/*) and the release binary itself, the exported GBM trees, this adapter, and the frozen zs_lane_v3 revision-2 file set (config_v3.json code
fingerprint ab31d061...; the zs_lane_v3 files are hashed from the byte copies in rev2_snapshot/ because the live
package now holds revision 3), which the adapter's point-cloud fallback executes and which holds the guard / custody
modules. sha256 over (relative path, file sha256) pairs in sorted order, as zs_lane_v3.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from research.zeroshot.zs_lane_v3_rs import rs
from research.zeroshot.zs_lane_v3_rs.rs import REPO

# the frozen zs_lane_v3 revision-2 pin list (config_v3.json, code fingerprint ab31d061...)
V3_REV2 = [
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
    "research/bench_labeling/custody.py",
    "research/broad_baseline/bb_common.py",
    "research/broad_baseline/gold11_cases.py",
    "research/broad_baseline/score.py",
    "research/broad_baseline/score_broad.py",
    "research/zeroshot/__init__.py",
    "research/zeroshot/common/__init__.py",
    "research/zeroshot/common/gold_guard.py",
    "research/zeroshot/d1_two_skin/__init__.py",
    "research/zeroshot/d1_two_skin/custody.py",
    "research/zeroshot/d1_two_skin/two_skin.py",
    "research/zeroshot/z2_self_recipe/__init__.py",
    "research/zeroshot/z2_self_recipe/common.py",
    "research/zeroshot/z2_self_recipe/graph.py",
    "research/zeroshot/z2_self_recipe/run.py",
    "research/zeroshot/z2_self_recipe/segment.py",
    "research/zeroshot/z3_slicing/__init__.py",
    "research/zeroshot/z3_slicing/custody.py",
    "research/zeroshot/z3_slicing/gate.py",
    "research/zeroshot/z3_slicing/pooled.py",
    "research/zeroshot/z3_slicing/score_run.py",
    "research/zeroshot/z3_slicing/selector.py",
    "research/zeroshot/z3_slicing/slicer.py",
    "research/zeroshot/z3_slicing/union.py",
    "research/zeroshot/z3_slicing/zscore.py",
    "research/zeroshot/zs_lane_v2/__init__.py",
    "research/zeroshot/zs_lane_v2/core.py",
    "research/zeroshot/zs_lane_v2/evidence.py",
    "research/zeroshot/zs_lane_v2/postproc.py",
    "research/zeroshot/zs_lane_v2/z3s.py",
    "research/zeroshot/zs_lane_v3/__init__.py",
    "research/zeroshot/zs_lane_v3/budget.py",
    "research/zeroshot/zs_lane_v3/core.py",
    "research/zeroshot/zs_lane_v3/fingerprint.py",
    "research/zeroshot/zs_lane_v3/fixes.py",
    "research/zeroshot/zs_lane_v3/lane.py",
    "research/zeroshot/zs_lane_v3/z3_merge.py",
    "scripts/evaluate_f1_against_manual_gold.py",
]
ADAPTER = [
    "research/zeroshot/zs_lane_v3_rs/__init__.py",
    "research/zeroshot/zs_lane_v3_rs/core.py",
    "research/zeroshot/zs_lane_v3_rs/lane.py",
    "research/zeroshot/zs_lane_v3_rs/rs.py",
    "research/zeroshot/zs_lane_v3_rs/fingerprint.py",
    "research/zeroshot/zs_lane_v3_rs/rev2ref.py",
    "research/zeroshot/zs_lane_v3_rs/rev2_snapshot/config_v3.json",
    "research/zeroshot/zs_lane_v3_rs/models/zs_lane_v3_zs_gbm.gbm.json",
    "research/zeroshot/zs_lane_v3_rs/models/zs_lane_v3_z3_gbm.gbm.json",
]
CRATE = "research/zeroshot/zs_rust"
SCORER_FILES = [
    "research/broad_baseline/score.py",
    "research/broad_baseline/score_broad.py",
    "research/broad_baseline/bb_common.py",
    "scripts/evaluate_f1_against_manual_gold.py",
    "research/zeroshot/zs_lane_v2/scoring.py",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rust_files() -> list[str]:
    """crate sources + the golden self-test set (golden.json is compiled into the binary; the meshes are its input)"""
    src = sorted(p.relative_to(REPO).as_posix() for p in (REPO / CRATE / "src").glob("*.rs"))
    golden = sorted(p.relative_to(REPO).as_posix() for p in (REPO / CRATE / "golden").glob("*") if p.is_file())
    return [f"{CRATE}/Cargo.toml", f"{CRATE}/Cargo.lock", *src, *golden]


TARGETS = {  # built binaries of the portable crate (sha256 recorded per target; outputs are identical)
    "aarch64-apple-darwin": ("target/release/zs_rust", "generic"),
    "x86_64-apple-darwin": ("target/x86_64-apple-darwin/release/zs_rust", "generic"),
    "x86_64-apple-darwin+x86-64-v2": ("target/x86_64-v2/x86_64-apple-darwin/release/zs_rust", "x86-64-v2"),
    "x86_64-apple-darwin+x86-64-v3": ("target/x86_64-v3/x86_64-apple-darwin/release/zs_rust", "x86-64-v3"),
    "x86_64-pc-windows-msvc+x86-64-v2": ("target/x86_64-pc-windows-msvc/release/zs_rust.exe", "x86-64-v2"),
}


def target_binaries() -> dict:
    out = {}
    for t, (rel, cpu) in TARGETS.items():
        p = REPO / CRATE / rel
        out[t] = {"path": f"{CRATE}/{rel}", "target_cpu": cpu,
                  "sha256": sha256_file(p) if p.is_file() else None, "bytes": p.stat().st_size if p.is_file() else None}
    return out


def lane_files() -> list[str]:
    return sorted(set(V3_REV2) | set(ADAPTER) | set(rust_files()))


def combined(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(k.encode())
        h.update(b"\0")
        h.update(hashes[k].encode())
        h.update(b"\n")
    return h.hexdigest()


def lane_fingerprint(revision: int = 2) -> dict:
    """revision 2: the zs_lane_v3 files are hashed from their revision-2 copies (rev2_snapshot/); revision 3: from
    the live package (frozen revision 3, fingerprint dc23a905...; only budget.py / core.py differ). The binary runs
    the budget of the revision its config names (budget.cell_mm present = revision 3)."""
    from research.zeroshot.zs_lane_v3_rs import rev2ref
    res = rev2ref.resolve if revision == 2 else (lambda f: REPO / f)
    fh = {f: sha256_file(res(f)) for f in lane_files()}
    b = rs.binary()
    fh["BINARY:zs_rust"] = sha256_file(b) if b.is_file() else "MISSING"
    return {"files": fh, "sha256": combined(fh), "binary_sha256": fh["BINARY:zs_rust"],
            "cargo_lock_sha256": fh[f"{CRATE}/Cargo.lock"]}


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
    cover = set(lane_files()) | set(SCORER_FILES) | set(extra_allowed)
    return [f for f in loaded_repo_files() if f not in cover]
