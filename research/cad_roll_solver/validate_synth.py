"""Exact-recovery check of the roll solver on the synthetic STEP cases (synth.py).

usage: .venv/bin/python -B -m research.cad_roll_solver.validate_synth --dir DIR [--json OUT.json]
Reads DIR/truth.json and DIR/<case>.step (written by synth.py); prints and writes per-case errors.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.cad_roll_solver.solver import solve_step  # noqa: E402

TOL_MM, TOL_DEG = 1e-4, 1e-3


def _ang(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def _line_dist(p, d, q):
    p, d, q = (np.asarray(x, float) for x in (p, d, q))
    d = d / np.linalg.norm(d)
    v = q - p
    return float(np.linalg.norm(v - (v @ d) * d))


def compare(doc: dict, truth: dict) -> dict:
    exp = truth["expected"]
    s = doc["summary"]
    res = {"counts": {}, "rolls": [], "pass": True}
    for k in ("rolls", "junction_bends", "e1_bends", "solver_bends", "cones"):
        if k in exp:
            ok = s[k] == exp[k]
            res["counts"][k] = {"got": s[k], "want": exp[k], "ok": ok}
            res["pass"] &= ok
    if "ends" in exp and doc["rolls"]:
        got = sorted(ro["end_type"] for ro in doc["rolls"][0]["runouts"])
        ok = got == sorted(exp["ends"])
        res["counts"]["ends"] = {"got": got, "want": sorted(exp["ends"]), "ok": ok}
        res["pass"] &= ok
    if exp.get("closed"):
        ok = bool(doc["rolls"]) and doc["rolls"][0]["closed"]
        res["counts"]["closed"] = {"got": ok, "want": True, "ok": ok}
        res["pass"] &= ok
    res["all_checks_pass"] = s["all_checks_pass"]
    res["pass"] &= s["all_checks_pass"]
    for tr in truth["rolls"]:
        # the solver roll whose axis line passes through the truth centre
        best = min(doc["rolls"], key=lambda r: _line_dist(r["axis"]["point_mm"], r["axis"]["direction"], tr["centre"]),
                   default=None)
        if best is None:
            res["rolls"].append({"found": False})
            res["pass"] = False
            continue
        e = {"found": True, "id": best["id"],
             "axis_angle_deg": _ang(best["axis"]["direction"], tr["axis"]),
             "axis_offset_mm": _line_dist(best["axis"]["point_mm"], best["axis"]["direction"], tr["centre"]),
             "ri_err_mm": best["r_inner_mm"] - tr["ri"], "ro_err_mm": best["r_outer_mm"] - tr["ro"],
             "t_err_mm": best["thickness_mm"] - tr.get("t", tr["ro"] - tr["ri"]),
             "sweep_err_deg": best["sweep_deg"] - tr["sweep_deg"],
             "length_err_mm": best["axial_length_mm"] - tr["length"],
             "taper_err_mm_per_m": best["taper"]["mm_per_m"] - tr["taper_mm_per_m"],
             "chord_err_mm": best["chord_inner_mm"] - 2 * tr["ri"] * math.sin(math.radians(min(tr["sweep_deg"], 360)) / 2),
             "is_cone": best["is_cone"]}
        if tr.get("radials"):
            # run-out lines: each truth end radial must have one solver end on both skins at R from the axis
            c = np.asarray(tr["centre"], float)
            ax = np.asarray(tr["axis"], float)
            worst_pos, worst_ang = 0.0, 0.0
            for rad in tr["radials"]:
                rad = np.asarray(rad, float)
                cands = []
                for ro in best["runouts"]:
                    for skin, r_t in (("inner", tr["ri"]), ("outer", tr["ro"])):
                        mid = 0.5 * (np.asarray(ro[skin]["start_mm"]) + np.asarray(ro[skin]["end_mm"]))
                        v = mid - c
                        v = v - (v @ ax) * ax
                        ang = math.degrees(math.acos(float(np.clip(v @ rad / np.linalg.norm(v), -1, 1))))
                        cands.append((ang, skin, abs(np.linalg.norm(v) - r_t), ro))
                for skin in ("inner", "outer"):
                    m = min((x for x in cands if x[1] == skin), key=lambda x: x[0])
                    worst_ang, worst_pos = max(worst_ang, m[0]), max(worst_pos, m[2])
            e["runout_angle_err_deg"] = worst_ang
            e["runout_radius_err_mm"] = worst_pos
            e["runout_lengths_mm"] = sorted({round(ro[s]["length_mm"], 6) for ro in best["runouts"] for s in ("inner", "outer")})
        e["neighbours"] = best["neighbours"]
        e["runout_classes"] = [ro["class"] for ro in best["runouts"]]
        errs_mm = [abs(e[k]) for k in ("axis_offset_mm", "ri_err_mm", "ro_err_mm", "t_err_mm", "length_err_mm",
                                        "chord_err_mm") ] + [abs(e.get("runout_radius_err_mm", 0.0))]
        errs_deg = [abs(e[k]) for k in ("axis_angle_deg", "sweep_err_deg")] + [abs(e.get("runout_angle_err_deg", 0.0))]
        e["max_err_mm"], e["max_err_deg"] = max(errs_mm), max(errs_deg)
        e["exact"] = e["max_err_mm"] <= TOL_MM and e["max_err_deg"] <= TOL_DEG and abs(e["taper_err_mm_per_m"]) <= 1e-3
        res["pass"] &= e["exact"]
        res["rolls"].append(e)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    d = Path(a.dir)
    truth = json.loads((d / "truth.json").read_text())
    out = {}
    for name, tr in truth.items():
        doc, _, _ = solve_step(d / f"{name}.step")
        r = compare(doc, tr)
        out[name] = r
        brief = [(x.get("id"), round(x.get("max_err_mm", -1), 9), round(x.get("max_err_deg", -1), 9)) for x in r["rolls"]]
        print(f"{name:28s} pass={r['pass']} counts={ {k: (v['got'], v['want']) for k, v in r['counts'].items()} } "
              f"rolls={brief}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    print("ALL PASS" if all(v["pass"] for v in out.values()) else "SOME FAIL")


if __name__ == "__main__":
    main()
