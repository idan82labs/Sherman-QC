"""Springback probe summary per part with angle AND bend-line position truth (CAD-only).

usage: sb_summary.py A1_ITER3_DIR TAG
For every probe sbprobes/TAG/<sku>/S<k>: bend states vs truth; angles: |measured - true| within U, false FAIL /
false PASS; positions: the true OML virtual sharp of the perturbed parent/child planes (sb_position_truth.py's
definition), within U, false FAIL / PASS; flange FAILs are listed (no flange truth is computed).
"""
import json
import sys
from pathlib import Path

import numpy as np

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.articulated import fold_transform
from domains.production.engine.inspect.geom import apply, apply_dir
from domains.production.engine.inspect.part import PartView

M, TAG = Path(sys.argv[1]), sys.argv[2]
FAM = {"DEV-EA3E4820DF": "10839000", "DEV-6E71609F74": "23553000", "DEV-4D512B5C56": "38172000",
       "DEV-BDE878D2AF": "11979003", "DEV-AAB8C67C31": "28870000", "DEV-1F88D309A7": "15651004"}
rows, notes = {}, []
for sku, fam in FAM.items():
    pdirs = sorted((M / "sbprobes" / TAG / sku).glob("S*"), key=lambda p: int(p.name[1:]))
    pdirs = [p for p in pdirs if (p / "e3" / "result.json").exists() and (p / "truth.json").exists()]
    if not pdirs:
        continue
    cdir = M / "phys/compiled/dev" / sku
    raw = json.loads((cdir / "compiled_part.json").read_text())
    part = PartView(CompiledPartV1.model_validate(raw), cdir)
    r = rows.setdefault(fam, {"probes": 0, "bends": 0, "wrong": 0, "probes_wrong": 0, "ang_n": 0, "ang_inu": 0,
                              "ang_ff": 0, "ang_fp": 0, "pos_n": 0, "pos_inu": 0, "pos_fail_true": 0, "pos_ff": 0,
                              "pos_fp": 0, "flange_fail": 0})
    for pdir in pdirs:
        k = int(pdir.name[1:])
        stage = raw["stages"][k]
        formed = set(stage["formed_bend_uids"])
        truth = json.loads((pdir / "truth.json").read_text())
        res = json.loads((pdir / "e3" / "result.json").read_text())
        delta = {j: truth["delta_deg"][part.bends[j].label] for j in range(part.n)}
        A = {part.datum: np.eye(4)}
        for j in part.bfs:
            b = part.bends[j]
            A[b.child] = A[b.parent] @ fold_transform(b, (b.sweep if b.uid in formed else 0.0) + np.radians(delta[j]))
        Tk = {i: np.asarray(stage["panel_transforms"][p.panel_id], dtype=np.float64) for i, p in enumerate(part.panels)}
        r["probes"] += 1
        wrong = 0
        for rb in res["bends"]:
            b = part.bends[part.bend_index[rb["bend_uid"]]]
            tstate = "FORMED" if b.uid in formed else "NOT_YET_FORMED"
            r["bends"] += 1
            if rb["state"] != tstate:
                wrong += 1
            if tstate != "FORMED":
                continue
            for ch in rb["characteristics"]:
                tol = ch.get("tolerance") or {}
                if ch["kind"] == "INCLUDED_ANGLE" and ch.get("deviation") is not None:
                    tdev = -delta[b.index]
                    r["ang_n"] += 1
                    r["ang_inu"] += int(abs(ch["deviation"] - tdev) <= ch["uncertainty"])
                    out = tdev > tol["plus"] or tdev < -tol["minus"]
                    r["ang_ff"] += int(ch["decision"] == "FAIL" and not out)
                    r["ang_fp"] += int(ch["decision"] == "PASS" and out)
                elif ch["kind"] == "BEND_LINE_POSITION" and ch.get("deviation") is not None:
                    def plane(panel, skin):
                        pv = part.panels[panel]
                        n0 = apply_dir(A[panel], pv.n0[None])[0]
                        p0 = apply(A[panel], pv.p0[None])[0]
                        return (n0, float(n0 @ p0)) if skin == 0 else (-n0, float(-(n0 @ p0) + part.t))
                    n1, d1 = plane(b.parent, b.oml_skin)
                    n2, d2 = plane(b.child, b.oml_skin)
                    dv = np.cross(n1, n2)
                    dv /= np.linalg.norm(dv)
                    q0 = np.linalg.solve(np.vstack([n1, n2, dv]), np.array([d1, d2, 0.0]))
                    nl = stage["bend_lines"][b.uid]
                    s0, s1 = np.asarray(nl["start_mm"]), np.asarray(nl["end_mm"])
                    mid, e = 0.5 * (s0 + s1), (s1 - s0) / np.linalg.norm(s1 - s0)
                    q = q0 + float((mid - q0) @ dv) * dv
                    dvec = (q - mid) - float((q - mid) @ e) * e
                    sa = next(s for s in b.model.sides if s.side == "A")
                    tdev = float(dvec @ (Tk[part.panel_index[sa.panel_id]][:3, :3] @ np.asarray(sa.outward_dir, float)))
                    out = tdev > tol["plus"] or tdev < -tol["minus"]
                    r["pos_n"] += 1
                    r["pos_inu"] += int(abs(ch["deviation"] - tdev) <= ch["uncertainty"])
                    r["pos_fail_true"] += int(ch["decision"] == "FAIL" and out)
                    if ch["decision"] == "FAIL" and not out:
                        r["pos_ff"] += 1
                        notes.append(f"{fam} S{k} {b.label} position FALSE FAIL: meas {ch['deviation']:+.3f} true {tdev:+.3f}")
                    if ch["decision"] == "PASS" and out:
                        r["pos_fp"] += 1
                        notes.append(f"{fam} S{k} {b.label} position FALSE PASS: meas {ch['deviation']:+.3f} true {tdev:+.3f}")
                elif ch["kind"] == "FLANGE_LENGTH" and ch["decision"] == "FAIL":
                    r["flange_fail"] += 1
                    notes.append(f"{fam} S{k} {b.label} flange {ch.get('side')} FAIL {ch['deviation']:+.2f} (no flange truth)")
        r["wrong"] += wrong
        r["probes_wrong"] += int(wrong > 0)
        if wrong:
            notes.append(f"{fam} S{k}: {wrong} wrong states ({res['stage_decision'].get('detail')})")
print("| Family | Probes | Bend states | Probes with a wrong state | Wrong bend states | Angles within U | False angle FAIL / PASS | Positions within U | Position FAIL (all true) | False position FAIL / PASS | Flange FAILs (no truth) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
t = {}
for fam, r in rows.items():
    print(f"| {fam} | {r['probes']} | {r['bends']} | {r['probes_wrong']} | {r['wrong']} | {r['ang_inu']}/{r['ang_n']} | {r['ang_ff']} / {r['ang_fp']} | "
          f"{r['pos_inu']}/{r['pos_n']} | {r['pos_fail_true']} | {r['pos_ff']} / {r['pos_fp']} | {r['flange_fail']} |")
    for kk, v in r.items():
        t[kk] = t.get(kk, 0) + v
if t:
    print(f"| **All** | {t['probes']} | {t['bends']} | {t['probes_wrong']} | {t['wrong']} | {t['ang_inu']}/{t['ang_n']} | {t['ang_ff']} / {t['ang_fp']} | "
          f"{t['pos_inu']}/{t['pos_n']} | {t['pos_fail_true']} | {t['pos_ff']} / {t['pos_fp']} | {t['flange_fail']} |")
print()
for n in notes:
    print("-", n)
