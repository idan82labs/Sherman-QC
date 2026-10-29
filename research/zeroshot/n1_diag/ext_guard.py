"""Flank-presence guard for the EXT prototype (DEV only): for every line the EXT rule would extend, re-run the frozen
Z3 detect on the scan (same candidates as the DEV dump), take the candidate's flank directions and measure, with the
frozen slicer.flank_continuity, the fraction of 1 mm stations of each extension piece where BOTH plates are present.
usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.n1_diag.ext_guard OUT.json [shard i/n]"""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from research.zeroshot.n1_diag import dev_data as D, proto as PR
from research.zeroshot.z3_slicing import slicer as SL


def guard_pieces(mesh, tree, t, cand, line_ext, line_orig):
    a = np.asarray(cand["start"], float); b = np.asarray(cand["end"], float)
    d = (b - a) / np.linalg.norm(b - a)
    da, db = np.asarray(cand["da"], float), np.asarray(cand["db"], float)
    s_o = sorted([float((np.asarray(p) - a) @ d) for p in (line_orig["start"], line_orig["end"])])
    s_e = sorted([float((np.asarray(p) - a) @ d) for p in (line_ext["start"], line_ext["end"])])
    reach = min(20.0, max(3.0, float(cand.get("flank_min", 3.0))))
    out = []
    for s0, s1 in ((s_e[0], s_o[0]), (s_o[1], s_e[1])):
        if s1 - s0 > 1.0:
            fa, fb = SL.flank_continuity(mesh, tree, a, d, da, db, s0, s1, t, reach)
            out.append({"s0": s0, "s1": s1, "fa": fa, "fb": fb})
    return out


def main(out, shard="0/1"):
    from research.zeroshot.common import gold_guard
    sp, V = D.load_v1()
    v2l = json.loads((D.PROG / "v2/oof/OOF_base__final_lines.json").read_text())
    i, n = (int(x) for x in shard.split("/"))
    res = {}
    for j, s in enumerate(sorted(V)):
        if j % n != i:
            continue
        v = V[s]
        todo = []
        for bname, base in (("v1", PR.base_v1(v)), ("v2", PR.base_v2(v2l.get(s, [])))):
            ex = PR.ext(base, v, r=1.3, dedupe=False)
            for lo, le in zip(base, ex):
                if le["ext"]:
                    todo.append((bname, lo, le))
        if not todo:
            continue
        gold_guard.require_runnable(sp[s]["scan_path"].replace("~", str(__import__("pathlib").Path.home()), 1))
        mesh = SL.load_mesh(str(__import__("pathlib").Path(sp[s]["scan_path"]).expanduser()))
        det = SL.detect(mesh)
        tree = cKDTree(mesh.V)
        t = det["t_mm"]
        for bname, lo, le in todo:
            # the Z3 candidate of the line: largest has_z3 part, matched to detect() by raw endpoints
            parts = [k for k in lo["parts"] if v["C"][k]["has_z3"]]
            if not parts:
                continue
            k = max(parts, key=lambda k: np.linalg.norm(np.subtract(v["C"][k]["end"], v["C"][k]["start"])))
            c = v["C"][k]
            m = [x for x in det["candidates"] if np.allclose(x["start"], c["start"], atol=1e-6) and np.allclose(x["end"], c["end"], atol=1e-6)]
            if not m:
                res[f"{bname}|{s}|{','.join(map(str, lo['parts']))}"] = {"error": "NO_Z3_MATCH"}
                continue
            res[f"{bname}|{s}|{','.join(map(str, lo['parts']))}"] = {"pieces": guard_pieces(mesh, tree, t, m[0], le, lo)}
        print(s, len(todo), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
