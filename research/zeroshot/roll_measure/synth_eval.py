"""Synthetic accuracy of the roll measurement: known cylinders / cones / flat spots / single skin / holes.

usage: .venv/bin/python -B -m research.zeroshot.roll_measure.synth_eval --out OUT.json [--part i/n] [--noise MM]
Each case: synth.make -> run.analyse (the same detector + measurement as the real scans) -> errors vs truth.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.zeroshot.roll_measure import synth, run as RUN, measure as M  # noqa: E402

NOISE = 0.07   # median per-station circle-fit rms on the four real rolled scans (0.012-0.091 mm)


def cases(noise=NOISE):
    out = []
    seed = 0
    for R in (30, 60, 120, 200, 300):
        for sw in (30, 60, 90, 135, 180):
            seed += 1
            out.append({"tag": f"cyl_R{R}_sw{sw}", "kind": "grid",
                        "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=2.0, L=150, f1=30, f2=30, noise_mm=noise,
                                                spacing_mm=0.6 if R * math.radians(sw) < 700 else 0.8, seed=seed)})
    for R, sw, tp in ((60, 90, 0.02), (120, 90, 0.02), (200, 135, 0.01), (300, 135, 0.03)):
        seed += 1
        out.append({"tag": f"cone_R{R}_sw{sw}_tp{int(tp * 1000)}", "kind": "taper",
                    "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=2.0, L=200, taper=tp, noise_mm=noise, seed=seed)})
    for R, sw, dep, frac in ((60, 135, 1.0, 0.25), (120, 90, 1.0, 0.3), (120, 90, 0.5, 0.3), (300, 135, 1.5, 0.2),
                             (200, 90, 1.0, 0.1)):
        seed += 1
        out.append({"tag": f"flat_R{R}_sw{sw}_d{dep}", "kind": "flat",
                    "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=2.0, L=150, flat_center_frac=frac, flat_width_deg=20,
                                            flat_depth_mm=dep, flat_z=(0.0, 0.5), noise_mm=noise, seed=seed)})
    for sk in ("inner", "outer"):
        for R, sw in ((60, 90), (200, 135)):
            seed += 1
            out.append({"tag": f"one_{sk}_R{R}_sw{sw}", "kind": "single_skin",
                        "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=2.0, L=150, skins=sk, noise_mm=noise, seed=seed)})
    for R, sw, h in ((120, 135, (0.4, 0.55)), (60, 90, (0.3, 0.5))):
        seed += 1
        out.append({"tag": f"hole_R{R}_sw{sw}", "kind": "coverage",
                    "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=2.0, L=150, hole_frac=h, noise_mm=noise, seed=seed)})
    for R, sw in ((60, 90), (200, 135)):
        seed += 1
        out.append({"tag": f"noisy_R{R}_sw{sw}", "kind": "noise_x2.5",
                    "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=2.0, L=150, noise_mm=2.5 * noise, seed=seed)})
    for R, sw, t in ((40, 90, 4.0), (150, 120, 0.6)):
        seed += 1
        out.append({"tag": f"thick_R{R}_t{t}", "kind": "thickness",
                    "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=t, L=150, noise_mm=noise, seed=seed)})
    for R, sw, f1, f2 in ((37, 155, 0.0, 0.0), (60, 135, 0.0, 30.0), (200, 90, 0.0, 30.0), (30, 155, 0.0, 12.0)):
        seed += 1
        out.append({"tag": f"edge_R{R}_sw{sw}_f{int(f1)}-{int(f2)}", "kind": "free_edge",
                    "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=1.5, L=75, f1=f1, f2=f2, noise_mm=noise, seed=seed)})
    for R, sw, f in ((60, 90, 10.0), (200, 90, 10.0), (60, 90, 5.0)):
        seed += 1
        out.append({"tag": f"shortflat_R{R}_sw{sw}_f{int(f)}", "kind": "short_flat",
                    "spec": synth.SynthSpec(R=R, sweep_deg=sw, t=2.0, L=100, f1=f, f2=f, noise_mm=noise, seed=seed)})
    return out


def _line_dist(p, a, b):
    a, b, p = map(np.asarray, (a, b, p))
    d = (b - a) / np.linalg.norm(b - a)
    v = p - a
    return float(np.linalg.norm(v - (v @ d) * d))


def evaluate(case):
    spec = case["spec"]
    mesh, tr = synth.make(spec)
    t0 = time.time()
    res, _ = RUN.analyse(mesh, M.RollParams(), log=lambda *a: None)
    row = {"tag": case["tag"], "kind": case["kind"], "truth": {k: tr[k] for k in ("R_in", "R_out", "sweep_deg",
                                                                                   "taper_mm_per_m", "chord_in", "coverage", "flat")},
           "n_regions": res["n_regions"], "wall_s": round(time.time() - t0, 1)}
    regs = [r for r in res["regions"] if "measure" in r]
    if not regs:
        row["detected"] = False
        return row
    # the region nearest the truth radius
    m = min((r["measure"] for r in regs), key=lambda m: abs((m["R_inner"]["mean"] or m["R_outer"]["mean"] or 0) - tr["R_in"]))
    row["detected"] = True
    ad = np.array(m["axis"]["direction"])
    ta = np.array(tr["axis_dir"])
    row["axis_angle_err_deg"] = float(np.degrees(np.arccos(min(1.0, abs(ad @ ta)))))
    row["axis_pos_err_mm"] = _line_dist(m["axis"]["point_mid"], tr["axis_point"], np.array(tr["axis_point"]) + ta)
    sw = (m.get("sweep_clean_deg") or m["sweep_deg"])
    sgn = 1.0 if ad @ ta > 0 else -1.0     # taper / twist are signed along the reported axis direction
    nz = lambda v: np.nan if v is None else v
    err = {"R_in": nz(m["R_inner"]["mean"]) - tr["R_in"], "R_out": nz(m["R_outer"]["mean"]) - tr["R_out"],
           "sweep_deg": sw["mean"] - tr["sweep_deg"], "taper_mm_per_m": sgn * m["taper_mm_per_m"]["value"] - tr["taper_mm_per_m"],
           "chord_in": (m["chord_runout_mm"]["mean"] - tr["chord_in"]) if m["measured_skin_for_angles"] == "inner" else None,
           "twist_deg_per_m": (m["twist_deg_per_m"] or {}).get("value"),
           "straightness_rms_mm": m["straightness"]["centre_rms_mm"],
           "coverage": m["coverage"]["arc_fraction"]["mean"] - tr["coverage"],
           "t": (m["thickness"]["t_used_mm"] or np.nan) - spec.t}
    if spec.taper:
        # cone: the full-arc mean radius is at the middle station, truth R at mid-length
        pass
    unc = {"R_in": m["R_inner"]["u_boot"], "sweep_deg": sw["u_boot"], "taper_mm_per_m": m["taper_mm_per_m"]["u_boot"],
           "axis_dir_deg": m["axis"]["u_dir_deg"]}
    # run-out position: distance of the truth run-out lines (inner skin) to the measured primary lines
    ro = []
    for key in ("runout_start_in", "runout_end_in"):
        a, b = tr[key]
        best = None
        for mk in ("start", "end"):
            L = m["runout_lines"][mk]
            dd = 0.5 * (_line_dist(L["start"], a, b) + _line_dist(L["end"], a, b))
            if best is None or dd < best[0]:
                best = (dd, L["u_pos_mm"], L["type_majority"])
        ro.append(best)
    err["runout_pos_mm"] = float(np.mean([x[0] for x in ro]))
    unc["runout_pos_mm"] = float(np.mean([x[1] or np.nan for x in ro]))
    row["runout_types"] = [x[2] for x in ro]
    if m["measured_skin_for_angles"] == "outer":
        # the run-out points are on the outer skin: radial offset t from the inner truth line is expected
        err["runout_pos_mm"] = float(np.mean([abs(x[0] - spec.t) for x in ro]))
    row["err"] = err
    row["unc"] = unc
    row["fit_rms_mm"] = m["fit_rms_mm"]["median"]
    row["skins"] = m["skins"]
    row["flat_spots"] = m["flat_spots"]
    row["zone_radius"] = {k: (v["mean"] if v else None) for k, v in m["zone_radius"].items()}
    if tr["flat"]:
        f = tr["flat"]
        # the arc's start is CCW about the reported axis; if the axis is reported flipped the angle is mirrored
        swm = sw["mean"]
        hits = [z for z in m["flat_spots"] if any(z["angle_from_start_deg"][0] - 5 <= c <= z["angle_from_start_deg"][1] + 5
                                                 for c in (f["center_deg"], swm - f["center_deg"]))]
        row["flat_detected"] = bool(hits)
        if hits:
            z = max(hits, key=lambda z: z["max_depth_mm"])
            row["flat_depth_err_mm"] = z["max_depth_mm"] - f["depth_mm"]
            zc = 0.5 * sum(z["angle_from_start_deg"])
            row["flat_center_err_deg"] = min(zc - f["center_deg"], zc - (swm - f["center_deg"]), key=abs)
            row["flat_length_mm"] = z["length_mm"]
    else:
        row["false_flat_spots"] = len(m["flat_spots"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--noise", type=float, default=NOISE)
    a = ap.parse_args()
    i, n = map(int, a.part.split("/"))
    cs = cases(a.noise)[i::n]
    rows = []
    for c in cs:
        t0 = time.time()
        try:
            r = evaluate(c)
        except Exception as e:  # noqa: BLE001
            r = {"tag": c["tag"], "kind": c["kind"], "error": repr(e)}
        rows.append(r)
        e = r.get("err", {})
        print(f"{c['tag']:28s} det={r.get('detected')} R_in {e.get('R_in', float('nan')):+.3f} sweep {e.get('sweep_deg', float('nan')):+.2f} "
              f"taper {e.get('taper_mm_per_m', float('nan')):+.2f} ro {e.get('runout_pos_mm', float('nan')):.2f} "
              f"ax {r.get('axis_angle_err_deg', float('nan')):.3f} {time.time() - t0:.0f}s {r.get('error', '')}", flush=True)
        Path(a.out).write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
