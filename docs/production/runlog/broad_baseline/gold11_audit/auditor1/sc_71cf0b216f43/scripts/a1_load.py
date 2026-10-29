"""Auditor-1 helper: read the raw Geomagic PLY (vertices + normals + faces) read-only, cache as npy in scratch."""
import sys, os, json
import numpy as np
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01_full_scan.ply"
CACHE = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_71cf"
GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
OUT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_71cf0b216f43"

def load():
    vp = os.path.join(CACHE, "v.npy"); np_ = os.path.join(CACHE, "n.npy"); fp = os.path.join(CACHE, "f.npy")
    if os.path.exists(vp) and os.path.exists(np_) and os.path.exists(fp):
        return np.load(vp), np.load(np_), np.load(fp)
    with open(SCAN, "rb") as fh:
        head = fh.read(4096)
        k = head.index(b"end_header")
        k = head.index(b"\n", k) + 1  # header uses CRLF line endings
        fh.seek(k)
        lines = head[:k].decode().splitlines()
        nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
        nf = int([l for l in lines if l.startswith("element face")][0].split()[-1])
        vd = np.frombuffer(fh.read(nv * 24), dtype="<f4").reshape(nv, 6).astype(np.float64)
        rest = fh.read()
    # faces: uchar count + 3 int32, assume all triangles; verify
    fdt = np.dtype([("c", "u1"), ("i", "<i4", (3,))])
    fa = np.frombuffer(rest[: nf * 13], dtype=fdt)
    assert (fa["c"] == 3).all(), "non-triangle faces"
    F = fa["i"].astype(np.int64)
    V, N = vd[:, :3], vd[:, 3:]
    np.save(vp, V); np.save(np_, N); np.save(fp, F)
    return V, N, F

def gold_lines():
    d = json.load(open(GOLD))
    row = [c for c in d["cases"] if c["case_id"] == "corpus/01_10839000"][0]
    return [(b["bend_id"], np.array(b["line_start"], float), np.array(b["line_end"], float)) for b in row["bends"]]

if __name__ == "__main__":
    V, N, F = load()
    print(V.shape, N.shape, F.shape, V.min(0), V.max(0))
    nn = np.linalg.norm(N, axis=1); print("normal norms", nn.min(), nn.max())
