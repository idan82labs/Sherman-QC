"""Run the V10 STEP bend-axis extractor on one STEP file (read-only) and write
its result plus a lenient coaxial-pair census to an output JSON.

Usage: python -m research.broad_baseline.cad_crosscheck.run_one_step <step> <out.json>
"""
import json, math, sys, time
from pathlib import Path
import numpy as np


def _axis_key(row):
    a = np.asarray(row["angular_frame"]["axis"], float); a /= np.linalg.norm(a)
    p = np.asarray(row["line_start"], float)
    c = p - a * np.dot(p, a)  # axis line point closest to origin
    return a, c


def lenient_census(rows, ang_tol=1e-3, pos_tol=0.05):
    """Group partial cylindrical faces by coaxial carrier (any radius).
    A group is a bend candidate if it holds >=2 distinct radii whose
    separation is 0.2..20 mm and the union angular sweep is < 350 deg."""
    groups = []
    for r in rows:
        a, c = _axis_key(r)
        for g in groups:
            ga, gc = g["a"], g["c"]
            if np.linalg.norm(np.cross(a, ga)) < ang_tol:
                d = c - gc; d = d - ga * np.dot(d, ga)
                if np.linalg.norm(d) < pos_tol:
                    g["rows"].append(r); break
        else:
            groups.append({"a": a, "c": c, "rows": [r]})
    # split each coaxial group into axially-connected sub-groups (collinear
    # but separate tabs are separate bends)
    split = []
    for g in groups:
        a = g["a"]; subs = []
        for r in sorted(g["rows"], key=lambda r: min(np.dot(np.asarray(r["line_start"]), a), np.dot(np.asarray(r["line_end"]), a))):
            lo, hi = sorted([float(np.dot(np.asarray(r["line_start"]), a)), float(np.dot(np.asarray(r["line_end"]), a))])
            if subs and lo <= subs[-1]["hi"] + 0.5:
                subs[-1]["rows"].append(r); subs[-1]["hi"] = max(subs[-1]["hi"], hi)
            else:
                subs.append({"a": a, "c": g["c"], "rows": [r], "hi": hi})
        split.extend(subs)
    groups = split
    out = []
    for g in groups:
        radii = sorted({round(r["radius_mm"], 4) for r in g["rows"]})
        by_r = {}
        for r in g["rows"]:
            by_r.setdefault(round(r["radius_mm"], 4), 0.0)
            by_r[round(r["radius_mm"], 4)] += r["angular_sweep_deg"]
        max_sweep = max(by_r.values())
        sep = (radii[-1] - radii[0]) if len(radii) > 1 else 0.0
        span = max(r["axial_span_mm"] for r in g["rows"])
        kind = "BEND_CANDIDATE" if (len(radii) >= 2 and 0.2 <= sep <= 20 and max_sweep < 350) else (
            "HOLE_OR_FULL" if max_sweep >= 350 else "SINGLE_SKIN_OR_OTHER")
        r0 = min(g["rows"], key=lambda r: r["radius_mm"])
        iv = sorted(r0["angular_intervals"])
        if len(iv) == 2 and iv[0][0] <= 1e-9 and iv[1][1] >= 2 * math.pi - 1e-9:
            st, sw = iv[1][0], (2 * math.pi - iv[1][0]) + iv[0][1]
        else:
            st, sw = iv[0][0], iv[-1][1] - iv[0][0]
        th = st + sw / 2
        fa = r0["angular_frame"]; e1, e2 = np.asarray(fa["x_axis"]), np.asarray(fa["y_axis"])
        rmid = (radii[0] + radii[-1]) / 2
        off = rmid * (math.cos(th) * e1 + math.sin(th) * e2)
        lo_r = min(g["rows"], key=lambda r: min(np.dot(r["line_start"], g["a"]), np.dot(r["line_end"], g["a"])))
        a_ = g["a"]; ends = []
        for r in g["rows"]:
            ends += [np.asarray(r["line_start"]), np.asarray(r["line_end"])]
        ends.sort(key=lambda p: float(np.dot(p, a_)))
        canon = [(ends[0] + off).tolist(), (ends[-1] + off).tolist()]
        out.append({"canonical_line": canon, "face_indices": sorted(r["face_index"] for r in g["rows"]),
                    "radii_mm": radii, "sweep_by_radius_deg": {str(k): round(v, 3) for k, v in by_r.items()},
                    "max_axial_span_mm": round(span, 3), "radius_separation_mm": round(sep, 4), "kind": kind,
                    "axis": [round(x, 5) for x in g["a"].tolist()]})
    return out


def main():
    step, out = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.extract_step_topology_bend_axes_v10 import extract_step_bend_axes
    t0 = time.time()
    rec = {"step_path": str(step), "status": "OK"}
    try:
        res = extract_step_bend_axes(step)
    except Exception as exc:  # report, do not hide
        rec.update(status="EXTRACTOR_EXCEPTION", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        res = None
    rec["elapsed_s"] = round(time.time() - t0, 3)
    if res is not None:
        axes = res.get("bend_axes", [])
        rec["v10_bend_count"] = len(axes)
        rec["v10_bend_axes"] = [{"bend_id": a["bend_id"], "angular_sweeps_deg": a.get("angular_sweeps_deg"),
                                 "radii_mm": a.get("radii_mm"), "axial_span_mm": a.get("axial_span_mm"),
                                 "line_start": a.get("canonical_line_start"), "line_end": a.get("canonical_line_end")}
                                for a in axes]
        rec["v10_rejected_groups"] = [{"face_indices": g.get("face_indices"), "reason": g.get("rejection_reason")}
                                      for g in res.get("rejected_groups", [])]
        rec["extension_status"] = res.get("extension_status")
        rec["admission"] = (res.get("contract") or {}).get("admission_status")
        rows = res.get("partial_cylindrical_faces", [])
        rec["n_partial_cylindrical_faces"] = len(rows)
        rec["n_surface_diagnostics"] = len(res.get("surface_diagnostics", []))
        diag_types = {}
        for d in res.get("surface_diagnostics", []):
            k = d.get("surface_type") or d.get("status"); diag_types[k] = diag_types.get(k, 0) + 1
        rec["surface_type_census"] = diag_types
        cen = lenient_census(rows) if rows else []
        rec["lenient_coaxial_groups"] = cen
        rec["lenient_bend_count"] = sum(g["kind"] == "BEND_CANDIDATE" for g in cen)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("status", "v10_bend_count", "lenient_bend_count", "elapsed_s", "admission", "extension_status")}))


if __name__ == "__main__":
    main()
