"""Auditor-1 (sc_cec08dfca9ab) loader: raw Geomagic PLY -> npz cache in a private scratch dir. Read-only on inputs."""
import numpy as np
W = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_cec08dfca9ab_k7v2/"
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_2.ply"

def load_ply(path):
    with open(path, "rb") as f:
        hdr = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            hdr.append(line)
            if line == "end_header":
                break
        nv = nf = 0
        for h in hdr:
            if h.startswith("element vertex"): nv = int(h.split()[-1])
            if h.startswith("element face"): nf = int(h.split()[-1])
        v = np.frombuffer(f.read(nv * 24), dtype="<f4").reshape(nv, 6).astype(np.float64)
        fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        fr = np.frombuffer(f.read(nf * 13), dtype=fdt)
        assert np.all(fr["n"] == 3)
        return v[:, :3], v[:, 3:], fr["i"].astype(np.int64)

xyz, nrm, faces = load_ply(SCAN)
print("verts", xyz.shape, "faces", faces.shape, "bbox", xyz.min(0).round(2), xyz.max(0).round(2))
np.savez(W + "cec08_scan.npz", xyz=xyz, nrm=nrm, faces=faces)
