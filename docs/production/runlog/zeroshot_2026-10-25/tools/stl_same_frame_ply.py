"""Export an STL scan as a same-frame binary PLY for bench_labeling write-label (the studio indexes .ply only).
Vertices are exactly the ones research.bench_labeling.scanio.load_scan() reads from the STL (open3d read +
remove_duplicated_vertices, no transform), so write-label's vertex-set coordinate check passes (max dist 0).
usage: .venv/bin/python tools/stl_same_frame_ply.py <scan.stl> <out.ply>"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from research.bench_labeling.scanio import load_scan  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if src.suffix.lower() != ".stl" or out.suffix.lower() != ".ply":
    sys.exit("usage: stl_same_frame_ply.py <scan.stl> <out.ply>")
mesh = o3d.io.read_triangle_mesh(str(src))
mesh.remove_duplicated_vertices()
mesh.vertex_normals = o3d.utility.Vector3dVector()  # let loaders recompute from winding, as for the STL
out.parent.mkdir(parents=True, exist_ok=True)
o3d.io.write_triangle_mesh(str(out), mesh, write_ascii=False, compressed=False, write_vertex_normals=False)
a, b = load_scan(src), load_scan(out)
from scipy.spatial import cKDTree  # noqa: E402
d, _ = cKDTree(a.points).query(b.points, k=1)
print(json.dumps({"stl": str(src), "ply": str(out), "stl_sha256": a.sha256, "ply_sha256": b.sha256,
                  "n_stl": len(a.points), "n_ply": len(b.points), "faces": [a.n_faces, b.n_faces],
                  "max_vertex_dist_mm": float(d.max()), "same_frame": len(a.points) == len(b.points) and float(d.max()) <= 1e-4}))
