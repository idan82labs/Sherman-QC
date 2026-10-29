"""mirror_scan.py SRC DST AXIS(x|y|z): the scan mirrored in one axis (coordinate and normal negated, winding reversed).
PLY: reuses the E2E helper's reader/writer (scripts/production/hotfolder_e2e_support.py). Binary STL: facet normals and
vertices mirrored, vertices 2 and 3 swapped. DST keeps the source extension."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject/scripts/production")
from hotfolder_e2e_support import _read_ply, _write_ply  # noqa: E402
src, dst, ax = sys.argv[1], sys.argv[2], sys.argv[3]
i = "xyz".index(ax)
if src.lower().endswith(".stl"):
    raw = Path(src).read_bytes()
    n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    assert 84 + 50 * n == len(raw), "binary STL expected"
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    f = np.frombuffer(raw, dt, n, 84).copy()
    f["n"][:, i] *= -1; f["v"][:, :, i] *= -1
    f["v"] = f["v"][:, [0, 2, 1], :]
    Path(dst).write_bytes(raw[:84] + f.tobytes())
    print(json.dumps({"facets": n, "axis": ax})); sys.exit(0)
header, v, f, _ = _read_ply(Path(src))
v[ax] = -v[ax]
if "n" + ax in v.dtype.names:
    v["n" + ax] = -v["n" + ax]
if f is not None:
    f["v"] = f["v"][:, ::-1].copy()
_write_ply(Path(dst), header, v, f, f"e3-mirror research: mirrored in {ax}")
print(json.dumps({"vertices": int(len(v)), "faces": 0 if f is None else int(len(f)), "axis": ax}))
