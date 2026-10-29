"""Ablation: D1's gated outer-apex snap (research/zeroshot/d1_two_skin, read-only) as a final placement step.

For every Z2 line: D1 bend_profile on the mesh vertices around the line, then snap_line_to_outer_apex with the
D1 gate shift <= t / sin(theta/2) + 1 mm. Writes runs/<tag>_d1snap/<sid>/eval_lines.json.
usage: .venv/bin/python -m research.zeroshot.z2_self_recipe.snap_d1 --tag it07
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from research.zeroshot.z2_self_recipe import common as C
from research.zeroshot.d1_two_skin import two_skin as D1
import bb_common as bb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    x = ap.parse_args(argv)
    inv = C.inventory()
    src = C.OUT / "runs" / x.tag
    dst = C.OUT / "runs" / f"{x.tag}_d1snap"
    moved = []
    for sid in C.cohort():
        p = src / sid / "eval_lines.json"
        if not p.is_file():
            continue
        pay = json.loads(p.read_text())
        lines = pay.get("owned_bend_regions") or []
        d = dst / sid
        d.mkdir(parents=True, exist_ok=True)
        if lines:
            try:
                mesh = D1.load_mesh(str(C.scan_path(sid, inv)))
            except Exception:  # noqa: BLE001  (point clouds: no mesh -> unchanged)
                mesh = None
            diag = json.loads((src / sid / "diag.json").read_text())
            t = float(diag["t_mm"])
            out = []
            for ln in lines:
                ln = dict(ln)
                if mesh is not None:
                    th = float(ln.get("included_angle_deg") or 90.0)
                    prof = D1.bend_profile(mesh.vertices, mesh.vertex_normals, np.asarray(ln["line_start"]),
                                           np.asarray(ln["line_end"]), t)
                    gate = t / max(math.sin(math.radians(th) / 2.0), 1e-3) + 1.0
                    a, b, info = D1.snap_line_to_outer_apex(ln["line_start"], ln["line_end"], prof, gate)
                    if info.get("moved"):
                        moved.append(info["shift_mm"])
                    ln["line_start"], ln["line_end"] = a.tolist(), b.tolist()
                    ln["d1_snap"] = info
                out.append(ln)
            pay["owned_bend_regions"] = out
        pay["lane"] = "z2_self_recipe+d1snap"
        bb.write_json(d / "eval_lines.json", pay)
        if (src / sid / "diag.json").is_file():
            (d / "diag.json").write_text((src / sid / "diag.json").read_text())
    print("moved", len(moved), "median shift", float(np.median(moved)) if moved else None)


if __name__ == "__main__":
    main()
