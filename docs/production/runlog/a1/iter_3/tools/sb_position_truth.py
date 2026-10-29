"""True bend-line position deviations of a springback probe (CAD-only), vs E3's measured ones.

The probe bent every formed bend to sweep + delta (and kinked unformed ones) by E3's fold kinematics chained
from the datum (springback_probe.py; deltas in truth.json). The true OML virtual sharp of a formed bend is the
intersection of its perturbed parent and child OML planes. Its deviation is taken as E3 defines it: offset from
the stage's nominal OML sharp, perpendicular to the nominal line, along side A's stage-pose outward direction.
usage: sb_position_truth.py COMPILED_DIR PROBE_DIR [PROBE_DIR ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.articulated import fold_transform
from domains.production.engine.inspect.geom import apply, apply_dir
from domains.production.engine.inspect.part import PartView

cdir = Path(sys.argv[1])
compiled = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_text())
raw = json.loads((cdir / "compiled_part.json").read_text())
part = PartView(compiled, cdir)
for pdir in map(Path, sys.argv[2:]):
    truth = json.loads((pdir / "truth.json").read_text())
    k = int(pdir.name[1:])
    stage = raw["stages"][k]
    formed = set(stage["formed_bend_uids"])
    deltas = {part.bends[j].index: truth["delta_deg"][part.bends[j].label] for j in range(part.n)}
    A = {part.datum: np.eye(4)}
    for j in part.bfs:
        b = part.bends[j]
        th = (b.sweep if b.uid in formed else 0.0) + np.radians(deltas[j])
        A[b.child] = A[b.parent] @ fold_transform(b, th)
    Tk = {i: np.asarray(stage["panel_transforms"][p.panel_id], dtype=np.float64) for i, p in enumerate(part.panels)}
    res = json.loads((pdir / "e3" / "result.json").read_text())
    meas = {}
    for rb in res["bends"]:
        for ch in rb["characteristics"]:
            if ch["kind"] == "BEND_LINE_POSITION" and ch.get("deviation") is not None:
                meas[rb["label"]] = (ch["deviation"], ch["uncertainty"], ch["decision"], ch["tolerance"])
    out = []
    for b in part.bends:
        if b.uid not in formed or b.label not in meas:
            continue
        def plane(panel, skin):
            pv = part.panels[panel]
            n0 = apply_dir(A[panel], pv.n0[None])[0]
            p0 = apply(A[panel], pv.p0[None])[0]
            if skin == 0:
                return n0, float(n0 @ p0)
            return -n0, float(-(n0 @ p0) + part.t)
        n1, d1 = plane(b.parent, b.oml_skin)
        n2, d2 = plane(b.child, b.oml_skin)
        dirv = np.cross(n1, n2)
        dirv /= np.linalg.norm(dirv)
        M = np.vstack([n1, n2, dirv])
        q0 = np.linalg.solve(M, np.array([d1, d2, 0.0]))
        nl = stage["bend_lines"][b.uid]
        s0, s1 = np.asarray(nl["start_mm"]), np.asarray(nl["end_mm"])
        mid = 0.5 * (s0 + s1)
        e = (s1 - s0) / np.linalg.norm(s1 - s0)
        q = q0 + float((mid - q0) @ dirv) * dirv  # true sharp point nearest the nominal midpoint
        dvec = q - mid
        dvec = dvec - float(dvec @ e) * e
        side_a = next(s for s in b.model.sides if s.side == "A")
        pa = part.panel_index[side_a.panel_id]
        out_a = Tk[pa][:3, :3] @ np.asarray(side_a.outward_dir, dtype=np.float64)
        dev_true = float(dvec @ out_a)
        dm, um, dec, tol = meas[b.label]
        out_tol = dev_true > tol["plus"] or dev_true < -tol["minus"]
        flag = ""
        if dec == "FAIL" and not out_tol:
            flag = "FALSE_FAIL"
        if dec == "PASS" and out_tol:
            flag = "FALSE_PASS"
        out.append(f"{b.label}: meas {dm:+.3f} U {um:.3f} {dec} | true {dev_true:+.3f} | err {dm - dev_true:+.3f} {'in U' if abs(dm - dev_true) <= um else 'OUT of U'} {flag}")
    print(f"== {pdir.parent.name} {pdir.name}")
    for o in out:
        print("   " + o)
