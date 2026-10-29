"""CAD-stage virtual scan with springback (A1 iter 3; CAD only, no physical scan read).

Like stage_probe.py, but every formed bend of S_k is bent to its nominal sweep + delta (delta uniform in
+-MAXDEG, seeded), every unformed bend gets a small kink (+-KINK deg), using E3's fold kinematics
(articulated.fold_transform, the E2 fold model) chained from the datum. Mesh vertices are labelled by
their nearest S_k sample (panel or bend); bend-arc vertices follow their parent panel (a sub-0.1 mm
artefact at the tangent line). With MAXDEG = KINK = 0 the mesh must reproduce S_k (self-check printed).

usage: springback_probe.py COMPILED_DIR RECIPE OUT_DIR STAGE [--maxdeg 2] [--kink 1] [--seed 11] [--plan FULL]
                           [--engine-root DIR]
Truth per bend: state (formed in S_k), included-angle deviation = -delta (formed bends).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    ap.add_argument("--maxdeg", type=float, default=2.0)
    ap.add_argument("--kink", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--plan", default="FULL")
    ap.add_argument("--voxel", type=float, default=0.57)
    ap.add_argument("--engine-root", type=Path, default=None)
    ap.add_argument("--rescan", action="store_true")
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
    rng = np.random.default_rng(np.random.SeedSequence([a.seed, a.stage, 777]))
    delta = {}
    for b in part.bends:
        if b.uid in formed:
            delta[b.index] = float(rng.uniform(-a.maxdeg, a.maxdeg))
        else:
            delta[b.index] = float(rng.uniform(-a.kink, a.kink))
    # perturbed panel poses (final pose -> perturbed), chained from the datum
    A = {part.datum: np.eye(4)}
    for j in part.bfs:
        b = part.bends[j]
        th = (b.sweep if b.uid in formed else 0.0) + np.radians(delta[j])
        A[b.child] = A[b.parent] @ fold_transform(b, th)
    Tk = {i: np.asarray(stage["panel_transforms"][p.panel_id], dtype=np.float64) for i, p in enumerate(part.panels)}
    for i in range(len(part.panels)):
        A.setdefault(i, Tk[i])  # a rigid follower keeps its stage pose
    V, F = read_ply_mesh(a.compiled / stage["display_mesh"]["rel_path"])
    npz = np.load(a.compiled / stage["sample_cloud"]["rel_path"])
    kind, sidx = npz["segment_kind"], npz["segment_index"]
    _, nn = cKDTree(npz["points"].astype(np.float64)).query(V)
    vk, vi = kind[nn], sidx[nn]
    bend_parent = np.array([b.parent for b in part.bends], dtype=np.int64)
    panel_of = vi.astype(np.int64).copy()
    isb = vk == 1
    panel_of[isb] = bend_parent[np.clip(vi[isb], 0, len(bend_parent) - 1)]

    def mapped(Amap):
        out = np.empty_like(V)
        for p in np.unique(panel_of):
            sel = panel_of == p
            M = Amap[int(p)] @ np.linalg.inv(Tk[int(p)])
            out[sel] = apply(M, V[sel])
        return out

    # self-check: the unperturbed chain reproduces S_k
    A0 = {part.datum: np.eye(4)}
    for j in part.bfs:
        b = part.bends[j]
        A0[b.child] = A0[b.parent] @ fold_transform(b, b.sweep if b.uid in formed else 0.0)
    for i in range(len(part.panels)):
        A0.setdefault(i, Tk[i])
    selfcheck = float(np.max(np.linalg.norm(mapped(A0) - V, axis=1)))
    Vp = mapped(A)
    a.out.mkdir(parents=True, exist_ok=True)
    scan_path = a.out / "scan.ply"
    t0 = time.time()
    if a.rescan or not scan_path.exists():
        z = np.zeros(len(F), dtype=np.uint8)
        surf = TessellatedSurface(Vp, F, z, np.zeros(len(F), dtype=np.int16), z.copy())
        pose = random_pose(rng)
        res = scan_surface(surf.transformed(pose), ScannerModel.from_json(CARD), SCAN_PLANS[a.plan], rng,
                           spacing_mm=a.voxel, output="points", ray_threads=1)
        write_ply_points(scan_path, res.points)
        np.save(a.out / "pose.npy", pose)
        (a.out / "truth.json").write_text(json.dumps({"delta_deg": {part.bends[j].label: d for j, d in delta.items()},
                                                      "formed": sorted(part.bends[part.bend_index[u]].label for u in formed),
                                                      "selfcheck_mm": selfcheck, "points": int(len(res.points))}))
    t_scan = time.time() - t0
    sha = hashlib.sha256(scan_path.read_bytes()).hexdigest()
    e3 = a.out / "e3"
    if e3.exists():
        shutil.rmtree(e3)
    cmd = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(a.compiled), "--recipe",
           str(a.recipe), "--scan", str(scan_path), "--out", str(e3), "--scan-sha256", sha]
    env, cwd = None, None
    if a.engine_root is not None:
        repo = Path.cwd()
        env = dict(os.environ, PYTHONPATH=f"{a.engine_root}:{repo}:{repo / 'backend'}")
        cwd = str(a.engine_root)
    t1 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    t_e3 = time.time() - t1
    truth = json.loads((a.out / "truth.json").read_text())
    summ = {"sku": raw["recipe"]["recipe_id"], "stage": stage["stage_model_id"], "selfcheck_mm": round(truth["selfcheck_mm"], 4),
            "rc": r.returncode, "t_e3_s": round(t_e3, 1), "t_scan_s": round(t_scan, 1)}
    if r.returncode != 0 or not (e3 / "result.json").exists():
        summ["error"] = (r.stdout + r.stderr)[-600:]
        (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
        print("SUMMARY", json.dumps(summ))
        return 0
    res = json.loads((e3 / "result.json").read_text())
    summ["validity"] = res["inspection_validity"]["state"]
    summ["config_sha"] = res["engine"]["config_sha256"][:8]
    states, wrong, false_fail, false_pass, ang = [], [], [], [], []
    for b in res["bends"]:
        lab = b["label"]
        truth_state = "FORMED" if lab in truth["formed"] else "NOT_YET_FORMED"
        st = b["state"]
        states.append(f"{lab}:{st[:3]}{'' if st == truth_state else '!' + truth_state[:3]}")
        if st != truth_state:
            wrong.append({"bend": lab, "state": st, "truth": truth_state, "delta": round(truth["delta_deg"][lab], 2)})
        for ch in b["characteristics"]:
            if ch["kind"] != "INCLUDED_ANGLE" or truth_state != "FORMED":
                continue
            tdev = -truth["delta_deg"][lab]
            tol = ch["tolerance"]
            out_tol = tdev > tol["plus"] or tdev < -tol["minus"]
            if ch["decision"] == "FAIL" and not out_tol:
                false_fail.append({"bend": lab, "dev": ch.get("deviation"), "true": round(tdev, 3), "U": ch.get("uncertainty")})
            if ch["decision"] == "PASS" and out_tol:
                false_pass.append({"bend": lab, "dev": ch.get("deviation"), "true": round(tdev, 3), "U": ch.get("uncertainty")})
            if ch.get("deviation") is not None:
                err = ch["deviation"] - tdev
                ang.append({"bend": lab, "err": round(err, 3), "U": round(ch["uncertainty"], 3), "inU": abs(err) <= ch["uncertainty"]})
    summ.update({"stage_decoded": res["stage_decision"].get("stage_model_id"), "states": " ".join(states),
                 "wrong_states": wrong, "angle_false_fail": false_fail, "angle_false_pass": false_pass,
                 "angle_errors": ang,
                 "fails_other": [f"{b['label']}/{ch['kind'][:4]}{ch.get('side') or ''}:{round(ch['deviation'], 2)}"
                                 for b in res["bends"] for ch in b["characteristics"]
                                 if ch["decision"] == "FAIL" and ch["kind"] != "INCLUDED_ANGLE"]})
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    n_in = sum(1 for x in ang if x["inU"])
    print("SUMMARY", json.dumps({k: summ[k] for k in ("sku", "stage", "validity", "stage_decoded", "states", "selfcheck_mm", "t_e3_s", "config_sha")}))
    print("   wrong", json.dumps(wrong)[:500])
    print(f"   angles measured {len(ang)}, within U {n_in}; false FAIL {false_fail}; false PASS {false_pass}")
    print("   other FAILs", summ["fails_other"][:8])
    for name in ("registered_scan.ply", "deviations.f32", "segment_labels.u16"):
        p = e3 / name
        if p.exists():
            p.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
