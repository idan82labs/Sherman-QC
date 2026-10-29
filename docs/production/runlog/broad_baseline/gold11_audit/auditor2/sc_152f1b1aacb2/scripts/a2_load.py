#!/usr/bin/env python3
"""Auditor 2 (sc_152f1b1aacb2): load raw scan mesh (vertices, normals, faces) + gold, cache to scratch npz. Read-only."""
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import open3d as o3d
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0_SCAN2 1.ply"
OUT = sys.argv[1]
m = o3d.io.read_triangle_mesh(SCAN)
V = np.asarray(m.vertices, float)
F = np.asarray(m.triangles, np.int64)
m.compute_vertex_normals()
N = np.asarray(m.vertex_normals, float)
cat = json.load(open("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"))
case = [c for c in cat["cases"] if c["case_id"] == "corpus/08_40500000"][0]
G = np.array([[b["line_start"], b["line_end"]] for b in case["bends"]], float)
ids = np.array([b["bend_id"] for b in case["bends"]])
np.savez(OUT, V=V, F=F, N=N, G=G, ids=ids)
print(V.shape, F.shape, V.min(0), V.max(0))
