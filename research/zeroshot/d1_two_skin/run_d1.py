#!/usr/bin/env python3
"""Per-scan D1 job: thickness, skin labels, gold-locus profiles, lane-line profiles + outer-apex snaps.

usage: run_d1.py --case-id sc_... --out OUT_DIR [--write-mid PLY_DIR]
Heavy (loads the mesh, ray-casts every vertex): run through sherman_slot.sh.
Reads only GOLD_SET_V1 case lists, the gold files they name (through the frozen score.load_gold),
the broad_baseline lane predictions, the scan mesh and (for validation) the STEP file.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/broad_baseline"))
from research.zeroshot.d1_two_skin import custody, two_skin as ts  # noqa: E402

RL = REPO / "docs/production/runlog"
CASE_LIST = RL / "gold_set_v1/case_list_gold_v1.json"
SCORED_LIST = RL / "gold_set_v1/rescore/case_list_gold_v1__scored.json"
FRAME_RECORDS = RL / "broad_baseline/broad_frame_records.json"
RUNS = RL / "broad_baseline/runs/best_f1"
LANES = {"a0": "predictions_a0.json", "best_f1": "predictions_best_f1.json"}


def case_entries(case_id: str) -> tuple[dict, dict | None]:
    base = [c for c in json.loads(CASE_LIST.read_text())["cases"] if c["case_id"] == case_id]
    if len(base) != 1:
        raise SystemExit(f"{case_id} not in GOLD_SET_V1")
    base = base[0]
    pred_id = base.get("prediction_case_id") or case_id
    scored = [c for c in json.loads(SCORED_LIST.read_text())["cases"] if c["case_id"] == pred_id]
    return base, (scored[0] if scored else None)


def _profile_dict(p: ts.BendProfile) -> dict:
    return p.to_dict()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--write-mid", type=Path, help="write the mid-surface point cloud PLY here")
    ap.add_argument("--no-cad", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    base, scored = case_entries(args.case_id)
    inv = custody.inventory_custody()
    scan = custody.check_scan(base["case_id"], base["scan_path"], inv)
    if scored is not None and scored["case_id"] != base["case_id"]:
        # STL twin: same mesh (INVENTORY marks it DUPLICATE_MESH_TWIN); only its prediction JSON is read
        custody.check_path(scored["scan_path"])
    rec: dict = {"schema": "sherman.zeroshot.d1.scan.v1", "case_id": base["case_id"],
                 "prediction_case_id": scored["case_id"] if scored else None,
                 "family_id": base["family_id"], "in_gold11": base.get("in_gold11"),
                 "custody": inv[base["case_id"]]["custody"], "scan_path": str(scan),
                 "has_step": bool(base.get("cad_path")) and str(base.get("cad_path")).lower().endswith((".stp", ".step"))}
    mesh = ts.load_mesh(str(scan))
    scene = ts._scene(mesh)
    rec["mesh"] = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                   "file_normal_agreement": mesh.file_normal_agreement}
    te = ts.estimate_thickness(mesh, scene=scene)
    rec["thickness"] = te.to_dict()
    rec["timing_s"] = {"thickness": round(time.time() - t0, 2)}
    t = te.t_mm
    if t is None:
        rec["status"] = "NO_THICKNESS"
    else:
        lab = ts.label_skins(mesh, t, scene=scene)
        rec["skins"] = lab.summary()
        rec["timing_s"]["labels"] = round(time.time() - t0, 2)
        if args.write_mid:
            mp, mn = ts.mid_surface(mesh, lab, t)
            args.write_mid.mkdir(parents=True, exist_ok=True)
            import open3d as o3d
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mp))
            pc.normals = o3d.utility.Vector3dVector(mn)
            outp = args.write_mid / f"mid_{base['case_id']}.ply"
            o3d.io.write_point_cloud(str(outp), pc, write_ascii=False)
            rec["mid_surface"] = {"path": str(outp), "points": int(len(mp))}
        P, N = mesh.vertices, mesh.vertex_normals
        # gold loci
        import score as frozen_score  # research/broad_baseline/score.py (frozen loader)
        frames = json.loads(FRAME_RECORDS.read_text())
        gold_spec = (scored or base)["gold"]
        try:
            gold = frozen_score.load_gold(gold_spec, frame_records=frames)
            rec["gold"] = {"kind": gold_spec["kind"], "n": len(gold["lines"]), "lines": []}
            for g in gold["lines"]:
                s, e = np.asarray(g["line_start"], float), np.asarray(g["line_end"], float)
                prof = ts.bend_profile(P, N, s, e, t)
                rec["gold"]["lines"].append({"bend_id": g["bend_id"], "start": s.tolist(), "end": e.tolist(),
                                             "profile": _profile_dict(prof)})
        except frozen_score.Excluded as err:
            rec["gold"] = {"excluded": err.reason}
        # lane predictions + snaps
        rec["lanes"] = {}
        if scored is not None:
            for lane, fname in LANES.items():
                pp = RUNS / scored["case_id"] / fname
                if not pp.is_file():
                    continue
                pay = json.loads(pp.read_text())
                rows = []
                for r in pay.get("owned_bend_regions") or []:
                    if r.get("line_start") is None:
                        continue
                    s, e = np.asarray(r["line_start"], float), np.asarray(r["line_end"], float)
                    prof = ts.bend_profile(P, N, s, e, t)
                    rows.append({"bend_id": r["bend_id"], "start": s.tolist(), "end": e.tolist(),
                                 "origin": r.get("origin"), "profile": _profile_dict(prof),
                                 "outer_apex_3d": None if prof.outer_apex_2d is None else prof.to3d(prof.outer_apex_2d).tolist(),
                                 "outer_sharp_3d": None if prof.outer_sharp_2d is None else prof.to3d(prof.outer_sharp_2d).tolist()})
                rec["lanes"][lane] = {"path": str(pp.relative_to(REPO)), "scan": pay.get("scan"), "lines": rows}
        rec["status"] = "OK"
    if not args.no_cad and rec["has_step"]:
        from research.zeroshot.d1_two_skin.cad_thickness import cad_thickness
        cad = custody.check_path(base["cad_path"])
        try:
            rec["cad_thickness"] = cad_thickness(str(cad))
        except Exception as err:  # noqa: BLE001
            rec["cad_thickness"] = {"error": f"{type(err).__name__}: {err}"}
    rec["timing_s"]["total"] = round(time.time() - t0, 2)
    rec["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{base['case_id']}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({"case": base["case_id"], "status": rec["status"], "t": t,
                      "cad_t": (rec.get("cad_thickness") or {}).get("t_mm"), "s": rec["timing_s"]["total"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
