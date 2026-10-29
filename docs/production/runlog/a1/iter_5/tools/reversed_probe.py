"""CAD-stage virtual scan with chosen bends formed the WRONG WAY (A1 iter 5 diagnostic; CAD only, no physical scan).

Like springback_probe.py (E3's fold kinematics chained from the datum, E2's scanner), but every bend in --reverse is
folded to -frac * its nominal sweep (the other side of its parent's plane), every other bend of S_k to its nominal
sweep, every unformed bend flat. The reversed child turns about the CAD axis, so its hinge differs from a real
reversed fold by about 2 Ri + t; the flange itself lies on the wrong side, which is what is tested.
Reports, per reversed bend, E3's state and angle decision: a PASS there is a false PASS.

usage: reversed_probe.py COMPILED_DIR RECIPE OUT_DIR STAGE --reverse B1[,B2] [--frac 1.0] [--seed 5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stage_probe import CARD, read_ply_mesh, write_ply_points  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("compiled", type=Path)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("stage", type=int)
    ap.add_argument("--reverse", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--plan", default="FULL")
    ap.add_argument("--voxel", type=float, default=0.57)
    ap.add_argument("--engine-root", type=Path, default=None, help="tree holding the domains package E3 runs from")
    ap.add_argument("--keep-scan", action="store_true")
    a = ap.parse_args()
    a.compiled, a.recipe, a.out = a.compiled.resolve(), a.recipe.resolve(), a.out.resolve()
    from scipy.spatial import cKDTree

    from contracts.production.compiled_part import CompiledPartV1
    from domains.production.engine.inspect.articulated import fold_transform
    from domains.production.engine.inspect.geom import apply
    from domains.production.engine.inspect.part import PartView
    from domains.production.engine.synthetic.brep import TessellatedSurface
    from domains.production.engine.synthetic.scanner import SCAN_PLANS, ScannerModel, random_pose, scan_surface

    raw = json.loads((a.compiled / "compiled_part.json").read_text())
    compiled = CompiledPartV1.model_validate(raw)
    part = PartView(compiled, a.compiled)
    stage = raw["stages"][a.stage]
    formed = set(stage["formed_bend_uids"])
    rev = set(a.reverse.split(","))
    rng = np.random.default_rng(np.random.SeedSequence([a.seed, a.stage, 999]))
    A = {part.datum: np.eye(4)}
    for j in part.bfs:
        b = part.bends[j]
        if b.label in rev:
            th = -a.frac * b.sweep
        else:
            th = b.sweep if b.uid in formed else 0.0
        A[b.child] = A[b.parent] @ fold_transform(b, th)
    Tk = {i: np.asarray(stage["panel_transforms"][p.panel_id], dtype=np.float64) for i, p in enumerate(part.panels)}
    for i in range(len(part.panels)):
        A.setdefault(i, Tk[i])
    V, F = read_ply_mesh(a.compiled / stage["display_mesh"]["rel_path"])
    npz = np.load(a.compiled / stage["sample_cloud"]["rel_path"])
    kind, sidx = npz["segment_kind"], npz["segment_index"]
    _, nn = cKDTree(npz["points"].astype(np.float64)).query(V)
    vk, vi = kind[nn], sidx[nn]
    bend_parent = np.array([b.parent for b in part.bends], dtype=np.int64)
    panel_of = vi.astype(np.int64).copy()
    isb = vk == 1
    panel_of[isb] = bend_parent[np.clip(vi[isb], 0, len(bend_parent) - 1)]
    Vp = np.empty_like(V)
    for p in np.unique(panel_of):
        sel = panel_of == p
        M = A[int(p)] @ np.linalg.inv(Tk[int(p)])
        Vp[sel] = apply(M, V[sel])
    a.out.mkdir(parents=True, exist_ok=True)
    scan_path = a.out / "scan.ply"
    if not scan_path.exists():
        z = np.zeros(len(F), dtype=np.uint8)
        surf = TessellatedSurface(Vp, F, z, np.zeros(len(F), dtype=np.int16), z.copy())
        pose = random_pose(rng)
        res = scan_surface(surf.transformed(pose), ScannerModel.from_json(CARD), SCAN_PLANS[a.plan], rng,
                           spacing_mm=a.voxel, output="points", ray_threads=1)
        write_ply_points(scan_path, res.points)
    sha = hashlib.sha256(scan_path.read_bytes()).hexdigest()
    e3 = a.out / "e3"
    if e3.exists():
        shutil.rmtree(e3)
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(a.compiled), "--recipe",
           str(a.recipe), "--scan", str(scan_path), "--out", str(e3), "--scan-sha256", sha]
    env, cwd = None, None
    if a.engine_root is not None:
        import os
        repo = Path.cwd()
        env = dict(os.environ, PYTHONPATH=f"{a.engine_root.resolve()}:{repo}:{repo / 'backend'}")
        cwd = str(a.engine_root.resolve())
    t1 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    t_e3 = time.time() - t1
    if r.returncode != 0 or not (e3 / "result.json").exists():
        print("ERROR", (r.stdout + r.stderr)[-800:])
        return 0
    out = json.loads((e3 / "result.json").read_text())
    print("SUMMARY", json.dumps({"sku": raw["recipe"]["recipe_id"], "stage": stage["stage_model_id"],
                                 "reversed": sorted(rev), "frac": a.frac,
                                 "validity": out["inspection_validity"]["state"],
                                 "validity_reasons": out["inspection_validity"].get("reasons"),
                                 "t_e3_s": round(t_e3, 1)}))
    for b in out["bends"]:
        chs = {c["kind"][:5] + (c.get("side") or ""): (c["decision"], None if c.get("deviation") is None else round(c["deviation"], 2))
               for c in b["characteristics"]}
        mark = "REVERSED" if b["label"] in rev else ""
        print(f"  {b['label']:4s} {b['state']:18s} {mark:8s} {chs} ev={b['state_evidence'].get('formed_fraction')}")
    print("  engine", out["engine"]["engine_version"], "notes", {b["label"]: b["notes"] for b in out["bends"] if b["notes"]})
    for name in ("registered_scan.ply", "deviations.f32", "segment_labels.u16"):
        p = e3 / name
        if p.exists():
            p.unlink()
    if not a.keep_scan:
        scan_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
