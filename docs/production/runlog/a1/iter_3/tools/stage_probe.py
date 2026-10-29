"""CAD-stage virtual scan probe (A1 iter 3; CAD only, no physical scan is read).

For a compiled A1 part and a compiled stage S_k, raycast the stage's display mesh (E1's
outward-oriented tessellation, posed by a random rigid motion) with E2's virtual scanner
(the amendment-4b model card), run the E3 CLI on the virtual scan and compare with the
stage's truth: bend states (formed iff in S_k's formed set) and, since the geometry is the
nominal stage model, every MEASURED deviation should be ~0 (any FAIL is a false FAIL).

usage: stage_probe.py COMPILED_DIR RECIPE OUT_DIR STAGE_INDEX [--plan FULL] [--seed N]
                      [--voxel MM] [--config CFG.json] [--keep-scan]
Prints one JSON summary line (prefix SUMMARY) and writes OUT_DIR/summary.json.
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

CARD = Path("docs/production/runlog/scanner_model_card.json")


def read_ply_mesh(path: Path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    nv = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in header if l.startswith("element face")).split()[-1])
    v = np.frombuffer(data, dtype="<f4", count=nv * 3, offset=end).reshape(nv, 3).astype(np.float64)
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    f = np.frombuffer(data, dtype=fdt, count=nf, offset=end + nv * 12)
    return v, f["i"].astype(np.int64)


def write_ply_points(path: Path, pts: np.ndarray) -> None:
    v = np.ascontiguousarray(pts, dtype="<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n").encode()
    path.write_bytes(header + v.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("compiled", type=Path)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("stage", type=int)
    ap.add_argument("--plan", default="FULL")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--voxel", type=float, default=0.57)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--keep-scan", action="store_true")
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--engine-root", type=Path, default=None, help="tree holding the domains package E3 runs from")
    a = ap.parse_args()
    a.compiled, a.recipe, a.out = a.compiled.resolve(), a.recipe.resolve(), a.out.resolve()
    if a.config is not None:
        a.config = a.config.resolve()

    from domains.production.engine.synthetic.brep import TessellatedSurface
    from domains.production.engine.synthetic.scanner import SCAN_PLANS, ScannerModel, random_pose, scan_surface

    cp = json.loads((a.compiled / "compiled_part.json").read_text())
    stage = cp["stages"][a.stage]
    assert stage["stage_index"] == a.stage
    a.out.mkdir(parents=True, exist_ok=True)
    scan_path = a.out / "scan.ply"
    t0 = time.time()
    if a.rescan or not scan_path.exists():
        v, f = read_ply_mesh(a.compiled / stage["display_mesh"]["rel_path"])
        z = np.zeros(len(f), dtype=np.uint8)
        surf = TessellatedSurface(v, f, z, np.zeros(len(f), dtype=np.int16), z.copy())
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, a.stage, 4242]))
        pose = random_pose(rng)
        surf_raw = surf.transformed(pose)
        model = ScannerModel.from_json(CARD)
        res = scan_surface(surf_raw, model, SCAN_PLANS[a.plan], rng, spacing_mm=a.voxel, output="points", ray_threads=1)
        write_ply_points(scan_path, res.points)
        np.save(a.out / "pose.npy", pose)
        scan_stats = {"points": int(len(res.points)), "nn_spacing_mm": res.stats.get("nn_spacing_mm")}
        (a.out / "scan_stats.json").write_text(json.dumps(scan_stats))
    scan_stats = json.loads((a.out / "scan_stats.json").read_text())
    t_scan = time.time() - t0
    sha = hashlib.sha256(scan_path.read_bytes()).hexdigest()
    e3 = a.out / "e3"
    if e3.exists():
        shutil.rmtree(e3)
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(a.compiled), "--recipe", str(a.recipe),
           "--scan", str(scan_path), "--out", str(e3), "--scan-sha256", sha]
    if a.config:
        cmd += ["--config", str(a.config)]
    t1 = time.time()
    env = None
    cwd = None
    if a.engine_root is not None:
        import os
        repo = Path.cwd()
        env = dict(os.environ, PYTHONPATH=f"{a.engine_root}:{repo}:{repo / 'backend'}")
        cwd = str(a.engine_root)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    t_e3 = time.time() - t1
    summ: dict = {"sku": cp["recipe"]["recipe_id"], "stage": stage["stage_model_id"], "plan": a.plan, "seed": a.seed,
                  "points": scan_stats["points"], "rc": r.returncode, "t_scan_s": round(t_scan, 1), "t_e3_s": round(t_e3, 1)}
    if r.returncode != 0 or not (e3 / "result.json").exists():
        summ["error"] = (r.stdout + r.stderr)[-800:]
        (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
        print("SUMMARY", json.dumps(summ))
        return 0
    res = json.loads((e3 / "result.json").read_text())
    formed_truth = set(stage["formed_bend_uids"])
    summ["validity"] = res["inspection_validity"]["state"]
    summ["config_sha"] = res["engine"]["config_sha256"][:8]
    summ["stage_decoded"] = res["stage_decision"].get("stage_model_id")
    summ["stage_detail"] = res["stage_decision"].get("detail")
    states, wrong, fails, reviews, unknowns, big = [], [], [], [], [], []
    for b in res["bends"]:
        truth = "FORMED" if b["bend_uid"] in formed_truth else "NOT_YET_FORMED"
        st = b["state"]
        ok = (st == truth) or (truth == "FORMED" and st == "PARTIALLY_FORMED" and False)
        states.append(f"{b['label']}:{st[:3]}{'' if st == truth else '!' + truth[:3]}")
        if st != truth:
            ev = b.get("state_evidence") or {}
            wrong.append({"bend": b["label"], "state": st, "truth": truth, "reasons": b.get("state_reasons"),
                          "ff": ev.get("formed_fraction"), "meas_deg": ev.get("measured_included_deg"),
                          "nom_deg": ev.get("nominal_included_deg")})
        for ch in b["characteristics"]:
            k = f"{b['label']}/{ch['kind'][:4]}{ch.get('side') or ''}"
            if ch["decision"] == "FAIL":
                fails.append({"c": k, "dev": ch.get("deviation"), "U": ch.get("uncertainty")})
            elif ch["decision"] == "REVIEW":
                reviews.append({"c": k, "dev": ch.get("deviation"), "U": ch.get("uncertainty")})
            elif ch["decision"] == "UNKNOWN" and truth == "FORMED":
                unknowns.append({"c": k, "reason": ch.get("reason")})
            if ch.get("deviation") is not None and ch.get("uncertainty") is not None and abs(ch["deviation"]) > ch["uncertainty"]:
                big.append({"c": k, "dev": round(ch["deviation"], 3), "U": round(ch["uncertainty"], 3)})
    summ.update({"states": " ".join(states), "wrong_states": wrong, "fails": fails, "reviews": reviews,
                 "unknown_on_formed": unknowns, "dev_beyond_U": big,
                 "warnings": [w for w in res.get("warnings", []) if not str(w).startswith("SCAN_")][:12]})
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    if not a.keep_scan:
        for name in ("registered_scan.ply", "deviations.f32", "segment_labels.u16"):
            p = e3 / name
            if p.exists():
                p.unlink()
    print("SUMMARY", json.dumps({k: summ[k] for k in ("sku", "stage", "validity", "stage_decoded", "states", "t_scan_s", "t_e3_s", "points", "config_sha")}))
    print("   wrong", json.dumps(wrong)[:600])
    print("   fails", json.dumps(fails)[:600])
    print("   beyondU", json.dumps(big)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
