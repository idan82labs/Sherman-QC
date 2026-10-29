"""One STEP -> V10 bend count + same-two-plates collinear merge count (read-only).
Usage: python step_one.py <step> <out.json>"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/broad_baseline/cad_crosscheck"))


def main():
    step, out = Path(sys.argv[1]), Path(sys.argv[2])
    from scripts.extract_step_topology_bend_axes_v10 import extract_step_bend_axes
    from run_one_step import lenient_census
    rec = {"step_path": str(step), "status": "OK"}
    t0 = time.time()
    try:
        res = extract_step_bend_axes(step)
    except Exception as exc:
        rec.update(status="EXTRACTOR_EXCEPTION", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        rec["elapsed_s"] = round(time.time() - t0, 3)
        out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec)); return
    rec["elapsed_s"] = round(time.time() - t0, 3)
    ax = res.get("bend_axes", [])
    rows = {r["face_index"]: r for r in res.get("partial_cylindrical_faces", [])}
    rec["v10_bend_count"] = len(ax)
    rec["extension_status"] = res.get("extension_status")
    rec["v10_bend_axes"] = [{"bend_id": a["bend_id"], "angular_sweeps_deg": a.get("angular_sweeps_deg"),
                             "radii_mm": a.get("radii_mm"), "axial_span_mm": a.get("axial_span_mm"),
                             "line_start": a.get("canonical_line_start"), "line_end": a.get("canonical_line_end")}
                            for a in ax]
    rec["v10_rejected_groups"] = [{"face_indices": g.get("face_indices"), "reason": g.get("rejection_reason")}
                                  for g in res.get("rejected_groups", [])]
    plates = []
    for a in ax:
        ps = set()
        for f in a["face_indices"]:
            for rail in rows.get(f, {}).get("tangent_rails", []):
                ps.add(rail["plane_face_index"])
        plates.append(ps)
    n = len(ax); parent = list(range(n))
    def find(i):
        while parent[i] != i: i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            s1, e1 = np.array(ax[i]["canonical_line_start"]), np.array(ax[i]["canonical_line_end"])
            s2, e2 = np.array(ax[j]["canonical_line_start"]), np.array(ax[j]["canonical_line_end"])
            d1 = (e1 - s1) / np.linalg.norm(e1 - s1); d2 = (e2 - s2) / np.linalg.norm(e2 - s2)
            if abs(d1 @ d2) < np.cos(np.radians(2)): continue
            off = max(np.linalg.norm((p - s1) - d1 * ((p - s1) @ d1)) for p in (s2, e2))
            if off < 1.0 and len(plates[i] & plates[j]) >= 3:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(f"C{i+1}")
    rec["bend_lines_same_plates_merged"] = len(groups)
    rec["merged_groups"] = [g for g in groups.values() if len(g) > 1]
    cen = lenient_census(list(rows.values())) if rows else []
    rec["lenient_bend_count"] = sum(g["kind"] == "BEND_CANDIDATE" for g in cen)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("status", "v10_bend_count", "bend_lines_same_plates_merged", "lenient_bend_count", "elapsed_s", "extension_status")}))


if __name__ == "__main__":
    main()
