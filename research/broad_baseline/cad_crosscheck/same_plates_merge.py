"""CAD bend-LINE count: merge V10 bend segments that are collinear AND fold the
same two plates (share >= 3 of their adjacent native plane faces, i.e. one
fold interrupted by reliefs/holes). Separate tabs on a common line share only
the base plate (2 faces) and stay separate.
Usage: same_plates_merge.py  (iterates the raw dir; re-runs V10 read-only)"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.extract_step_topology_bend_axes_v10 import extract_step_bend_axes  # noqa
O = ROOT / "docs/production/runlog/broad_baseline"
res = {}
for raw in sorted(O.glob("cad_crosscheck_raw/[0-9]*.json")):
    R = json.loads(raw.read_text())
    out = extract_step_bend_axes(Path(R["step_path"]))
    rows = {r["face_index"]: r for r in out["partial_cylindrical_faces"]}
    ax = out["bend_axes"]
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
    merged = [g for g in groups.values() if len(g) > 1]
    res[raw.stem] = {"v10_segments": n, "bend_lines_same_plates_merged": len(groups), "merged_groups": merged,
                     "plate_faces_per_bend": {f"C{i+1}": sorted(p) for i, p in enumerate(plates)}}
    print(raw.stem, n, len(groups), merged)
(O / "cad_crosscheck_match/_same_plates_merge.json").write_text(json.dumps(res, indent=1))
