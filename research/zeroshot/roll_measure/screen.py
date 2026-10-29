"""Roll screen over the usable DEV scans (custody DEV_CONSUMED, usable) of the programme INVENTORY.json.
Full detector + measurement (no overlay data); scans are full resolution (a 1.7 M-vertex scan takes < 10 s).
Reads only the inventory's scan paths / part ids / CAD flags; no gold lines.

usage: .venv/bin/python -B -m research.zeroshot.roll_measure.screen --out DIR [--part i/n]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.zeroshot.roll_measure import run as RUN, measure as M  # noqa: E402

PROG = REPO / "docs/production/runlog/zeroshot_2026-10-25"
FORBIDDEN = ("HOLDOUT", "holdout", "38173000_01", "30820000_A01", "EHO4", "Eyal50", "eyal50", "SEALED", "sealed",
             "TEST-B", "testb", "DEV-CHECK", "FRESH")


def dev_list():
    inv = json.loads((PROG / "INVENTORY.json").read_text())
    out = []
    for r in inv["scans"]:
        if r.get("custody") != "DEV_CONSUMED" or not r.get("usable"):
            continue
        p = os.path.expanduser(r["path"])
        if any(b in p for b in FORBIDDEN):
            continue
        out.append({"scan_id": r["scan_id"], "part_id": r["part_id"], "path": p,
                    "has_step": bool((r.get("cad") or {}).get("has_step")),
                    "step_path": (r.get("cad") or {}).get("step_path"),
                    "cad_compile": (r.get("cad") or {}).get("compile_status")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    i, n = map(int, a.part.split("/"))
    for s in dev_list()[i::n]:
        f = out / f"{s['scan_id']}.json"
        if f.exists():
            continue
        t0 = time.time()
        try:
            mesh = RUN.load(s["path"], None)
            res, plot = RUN.analyse(mesh, M.RollParams(), log=lambda *x: None, overlay=False)
            regs = []
            for r in res["regions"]:
                m = r["measure"]
                regs.append({"R_inner": m["R_inner"]["mean"], "R_outer": m["R_outer"]["mean"],
                             "sweep_deg": m["sweep_deg"]["mean"], "length_mm": m["length_mm"],
                             "n_stations": m["n_stations"], "skins": m["skins"], "fit_rms_mm": m["fit_rms_mm"]["median"],
                             "axis": m["axis"]["direction"], "runout_types": [m["runout_lines"][k]["type_majority"] for k in ("start", "end")]})
            rec = s | {"n_regions": len(regs), "regions": regs, "t_d1": res["thickness_d1"]["t_mm"],
                       "n_vertices": int(len(mesh.V)), "wall_s": round(time.time() - t0, 1)}
            sub = mesh.V[np.random.default_rng(0).choice(len(mesh.V), min(len(mesh.V), 40000), replace=False)]
            (out / f"{s['scan_id']}.pkl").write_bytes(pickle.dumps({"regions": plot, "cloud": sub, "res": res}))
        except Exception as e:  # noqa: BLE001
            rec = s | {"error": repr(e), "wall_s": round(time.time() - t0, 1)}
        f.write_text(json.dumps(rec, indent=1, default=float))
        print(f"{s['scan_id']} {s['part_id']:10s} regions={rec.get('n_regions')} "
              f"{[(round(x['R_inner'] or x['R_outer'], 1), round(x['sweep_deg'], 1), round(x['length_mm'])) for x in rec.get('regions', [])]} "
              f"{rec['wall_s']}s {rec.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
