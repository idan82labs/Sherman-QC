"""Run the CAD roll solver on STEP files (heavy: run through /private/tmp/claude-501/sherman_slot.sh).

usage: .venv/bin/python -B -m research.cad_roll_solver.run --out DIR [--name NAME] STEP [STEP ...]
       .venv/bin/python -B -m research.cad_roll_solver.run --out DIR --list STEPS.json   ({name: path})

Writes DIR/<name>.cad_roll_nominal.json and DIR/<name>.render.pkl (mesh + overlay data for render.py),
and prints one summary line per file. Reads only the STEP files given.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.cad_roll_solver.solver import solve_step  # noqa: E402


def _mesh(model):
    vs, ts, off = [], [], 0
    for f in model.faces:
        if len(f.mesh_triangles):
            vs.append(f.mesh_vertices)
            ts.append(f.mesh_triangles + off)
            off += len(f.mesh_vertices)
    return (np.concatenate(vs) if vs else np.zeros((0, 3))), (np.concatenate(ts) if ts else np.zeros((0, 3), int))


def run_one(path: Path, name: str, out: Path) -> dict:
    t0 = time.time()
    doc, render, model = solve_step(path)
    doc["source"]["name"] = name
    doc["runtime_s"] = round(time.time() - t0, 2)
    (out / f"{name}.cad_roll_nominal.json").write_text(json.dumps(doc, indent=1, default=float))
    v, t = _mesh(model)
    render.update({"vertices": v.astype(np.float32), "triangles": t.astype(np.int32), "doc": doc})
    with open(out / f"{name}.render.pkl", "wb") as fh:
        pickle.dump(render, fh)
    s = doc["summary"]
    return {"name": name, "file": path.name, "rolls": s["rolls"], "junction_bends": s["junction_bends"],
            "e1_bends": s["e1_bends"], "solver_bends": s["solver_bends"], "checks_pass": s["all_checks_pass"],
            "e1_error": doc["e1"]["flange_graph_error"] or doc["e1"]["part_model_error"],
            "unclassified": len(doc["unclassified_paired_features"]),
            "unpaired_large": len(doc["unpaired_large_curved_skins"]),
            "rolls_brief": [(r["id"], r["r_inner_mm"], r["r_outer_mm"], r["sweep_deg"], r["axial_length_mm"],
                             [ro["class"] for ro in r["runouts"]]) for r in doc["rolls"]],
            "s": doc["runtime_s"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name")
    ap.add_argument("--list")
    ap.add_argument("steps", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(a.list).read_text()) if a.list else {}
    for s in a.steps:
        jobs[a.name or Path(s).stem] = s
    rows = []
    for name, p in jobs.items():
        try:
            row = run_one(Path(p).expanduser(), name, out)
        except Exception as exc:  # recorded, never hidden
            row = {"name": name, "file": Path(p).name, "error": f"{type(exc).__name__}: {exc}"[:300]}
        row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    (out / "run_summary.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
