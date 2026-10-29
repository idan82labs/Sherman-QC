"""Scan roll measurement vs CAD roll nominal (cad_roll_nominal.json from research/cad_roll_solver).

No registration: intrinsic quantities only (radii, thickness, sweep, opening, axial length, taper), which do not
depend on the part's pose. For the overlays the CAD roll is ALIGNED ON THE FITTED AXIS: CAD axis := scan axis,
CAD axial centre := scan region centre, CAD angular middle := scan angular middle. The overlay therefore shows
shape differences (R, sweep, length) and hides any position/orientation offset of the roll on the part.

usage: .venv/bin/python -B -m research.zeroshot.roll_measure.compare SCAN_RESULT.json CAD_NOMINAL.json OUT.json [--region K]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math
from pathlib import Path

import numpy as np

# synthetic method floor (bias +- sd, grid of R 30-300 / sweep 30-180 at the real-scan noise); filled from the
# synthetic table by the report builder, used to widen the station-bootstrap uncertainty into a total one
DEFAULT_FLOOR = {"R": 0.08, "sweep_deg": 0.4, "taper_mm_per_m": 1.0, "opening_mm": 0.5, "length_mm": 2 * 2.0}


def match_roll(cad: dict, m: dict) -> dict | None:
    rolls = cad.get("rolls") or []
    if not rolls:
        return None
    Rs = m["R_inner"]["mean"] if m["R_inner"]["mean"] is not None else m["R_outer"]["mean"] - (cad["rolls"][0]["thickness_mm"] or 0)

    def score(r):
        return abs(r["r_inner_mm"] - Rs) / max(r["r_inner_mm"], 1.0) + abs(r["sweep_deg"] - m["sweep_deg"]["mean"]) / 360.0 \
            + abs(r["axial_length_mm"] - m["length_mm"]) / max(r["axial_length_mm"], 1.0)
    return min(rolls, key=score)


def _tot(u, floor):
    return None if u is None else float(math.hypot(u, floor))


def compare(m: dict, cad: dict, floor: dict | None = None) -> dict:
    floor = floor or DEFAULT_FLOOR
    r = match_roll(cad, m)
    if r is None:
        return {"status": "NO_CAD_ROLL", "note": "the CAD nominal lists no roll"}
    sw = m.get("sweep_clean_deg") or m["sweep_deg"]
    rows = {}

    def add(name, meas, u, nom, fl, note=""):
        if meas is None or nom is None:
            rows[name] = {"measured": meas, "nominal": nom, "delta": None, "note": note or "not measurable"}
            return
        rows[name] = {"measured": float(meas), "nominal": float(nom), "delta": float(meas - nom),
                      "u_station": u, "u_total": _tot(u, fl), "note": note}

    add("r_inner_mm", m["R_inner"]["mean"], m["R_inner"].get("u_boot"), r["r_inner_mm"], floor["R"],
        "" if m["R_inner"]["source"] == "measured" else m["R_inner"]["source"])
    add("r_outer_mm", m["R_outer"]["mean"], m["R_outer"].get("u_boot"), r["r_outer_mm"], floor["R"],
        "" if m["R_outer"]["source"] == "measured" else m["R_outer"]["source"])
    add("thickness_mm", m["thickness"]["t_used_mm"], None, r["thickness_mm"], 0.02, m["thickness"]["t_used_source"])
    add("sweep_deg", sw["mean"], sw.get("u_boot"), r["sweep_deg"], floor["sweep_deg"],
        "scan: curved sweep between the detected run-outs (stations with both run-outs on their lines)")
    ext = m.get("extent_to_edges_deg")
    if ext:
        add("extent_to_edges_deg", ext["mean"], ext.get("u_boot"), r["sweep_deg"], floor["sweep_deg"],
            "scan: sweep + straight unrolled tips to a sheet edge; compare when the CAD arc runs to the edge")
    ch = m["chord_runout_mm"]
    nom_ch = r["chord_inner_mm"] if m["measured_skin_for_angles"] == "inner" else r["chord_outer_mm"]
    add("opening_chord_mm", ch["mean"], ch.get("u_boot"), nom_ch, floor["opening_mm"],
        f"chord between the run-outs on the {m['measured_skin_for_angles']} skin")
    add("axial_length_mm", m["length_mm"], None, r["axial_length_mm"], floor["length_mm"],
        "scan length = span of the measured stations (excludes up to ~1 station step at each end)")
    tp = m["taper_mm_per_m"]
    add("taper_abs_mm_per_m", abs(tp["value"]), tp.get("u_boot"), abs(r["taper"]["mm_per_m"]), floor["taper_mm_per_m"],
        "magnitude only (sign depends on the axis orientation)")
    return {"status": "COMPARED", "cad_roll_id": r["id"], "cad_is_cone": r["is_cone"],
            "cad_end_types": [ro.get("end_type") for ro in r.get("runouts", [])],
            "registration": "none: intrinsic quantities; overlays aligned on the fitted axis",
            "rows": rows}


def cad_overlay(cad: dict, m: dict):
    """CAD drawing data in the scan region's axis frame (aligned on the fitted axis, see module doc)."""
    r = match_roll(cad, m)
    if r is None:
        return None, "CAD: no roll in cad_roll_nominal.json"
    ps = m["per_station"]
    psi = np.array([q["psi_deg"] for q in ps])
    mid = float(np.median(psi.mean(1)))
    half = r["sweep_deg"] / 2
    # feature anchor: if the CAD roll has exactly one tangent run-out (into a bend / flange) and the scan has
    # exactly one non-edge end, put the CAD tangent end on the scan's non-edge end; otherwise centre the arcs
    cad_tan = [ro["which"] for ro in r.get("runouts", []) if ro.get("end_type") == "TANGENT_RUNOUT"]
    scan_types = {k: m["runout_lines"][k]["type_majority"] for k in ("start", "end")}
    non_edge = [k for k, v in scan_types.items() if v != "edge"]
    anchor = "angular middle"
    if len(cad_tan) == 1 and len(non_edge) == 1:
        if non_edge[0] == "start":
            mid = float(np.median(psi[:, 0])) + half
        else:
            mid = float(np.median(psi[:, 1])) - half
        anchor = f"CAD tangent run-out on the scan's {non_edge[0]} run-out ({scan_types[non_edge[0]]})"
    L = r["axial_length_mm"]
    ri, ro = r["r_inner_mm"], r["r_outer_mm"]
    k = r["taper"]["dr_dz"]
    zs = np.linspace(-L / 2, L / 2, 7)
    ph = np.radians(np.linspace(mid - half, mid + half, 120))
    wire = [np.c_[(ri + k * z) * np.cos(ph), (ri + k * z) * np.sin(ph), np.full(len(ph), z)] for z in zs]
    ros = []
    for key, a in (("start", mid - half), ("end", mid + half)):
        a = math.radians(a)
        ros.append((key, np.array([[(ri - k * L / 2) * math.cos(a), (ri - k * L / 2) * math.sin(a), -L / 2],
                                   [(ri + k * L / 2) * math.cos(a), (ri + k * L / 2) * math.sin(a), L / 2]])))
    skin_off = {-1: ri, 1: ro}
    out = {"_circles_axis": [(0.0, 0.0, ri, "R_in"), (0.0, 0.0, ro, "R_out")], "_psi": [mid - half, mid + half],
           "_wire_axis": wire, "_runouts_axis": ros,
           "_dev_fn": lambda s, phi, rho, skin: rho - (skin_off[skin] + k * s)}
    note = (f"CAD {r['id']} (teal): R_in {ri:.2f} / R_out {ro:.2f}, sweep {r['sweep_deg']:.1f}°, L {L:.1f}; "
            f"NOT registered: aligned on the fitted axis, axial centre and {anchor}")
    out["_anchor"] = anchor
    return out, note


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    cad = json.loads(Path(sys.argv[2]).read_text())
    k = int(sys.argv[sys.argv.index("--region") + 1]) if "--region" in sys.argv else 0
    out = compare(res["regions"][k]["measure"], cad)
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    for n, row in out.get("rows", {}).items():
        print(n, row)


if __name__ == "__main__":
    main()
