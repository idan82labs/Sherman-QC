"""Load the raw scan once (read-only) and cache it in the part frame (scratchpad npz)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from research.bench_labeling.scanio import load_scan

A = Path(__file__).resolve().parent
out = Path(sys.argv[1])
fr = json.loads((A / "part_frame.json").read_text())
R = np.stack([fr["ex"], fr["ey"], fr["ez"]]); o = np.array(fr["origin"])
scan = load_scan("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01_full_scan.ply")
print(scan.sha256, len(scan.points), scan.kind, scan.oriented)
Q = (scan.points - o) @ R.T
NQ = scan.normals @ R.T
np.savez(out, Q=Q.astype(np.float64), NQ=NQ.astype(np.float32), R=R, o=o)
print("saved", out)
