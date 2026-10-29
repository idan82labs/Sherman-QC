"""DIAGNOSTIC (consumed 49207000): raw-scan cross-sections perpendicular to gold B3, with the CAD final-stage section
mapped by the E3 pose and by the ICP pose, the gold lines and the engine's measured B2 apex."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = Path("/Users/idantal/ShermanQC/FullProject"); sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.inspect.scan_io import read_scan
SP = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
c = {x["case"]: x for x in json.loads(Path(sys.argv[1]).read_text())}["fr_6fa053514df8"]; gold = {g["bend_id"]: g for g in c["gold"]}
res = json.loads((SP / "fixes3/posthoc3/fr_6fa053514df8/engine/result.json").read_text())
cp = json.loads((SP / "fixes/compile_after1/DEV-7052AE8973/compiled/compiled_part.json").read_text())
raw = read_scan(str(REPO / "data/bend_corpus/incoming_codex_fetch_2026-10-17/20260901_e8bfd4f6/49207000_B.ply")).points
cad = np.load(SP / "fixes/compile_after1/DEV-7052AE8973/compiled" / cp["stages"][-1]["sample_cloud"]["rel_path"])["points"]
T = np.asarray(res["inspection_validity"]["registration"]["raw_to_frame"], float); Ti = np.linalg.inv(T)
z = np.load(SP / "fixes2/icp49207.npz"); R, t = z["R"], z["t"]
cad_e3 = cad @ Ti[:3, :3].T + Ti[:3, 3]; cad_icp = cad @ R.T + t
g = gold["B3"]; a = np.array(g["line_start"]); e = np.array(g["line_end"]); d = (e - a) / np.linalg.norm(e - a)
u1 = np.cross(d, [0, 0, 1.0]); u1 /= np.linalg.norm(u1); u2 = np.cross(d, u1)
def sec(P, s, half=0.6, r=60):
    q = P - (a + d * s); m = (np.abs(q @ d) < half); q = q[m]; uv = np.stack([q @ u1, q @ u2], 1)
    return uv[np.linalg.norm(uv, axis=1) < r]
def pt(p0, p1, s):
    p0, p1 = np.asarray(p0), np.asarray(p1); dd = p1 - p0; f = ((a + d * s - p0) @ d) / (dd @ d); q = p0 + f * dd - (a + d * s)
    return q @ u1, q @ u2
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
dg = next(b for b in res["bends"] if b["label"] == "B2")["diagnostics"]
ea0 = (Ti @ np.array([dg["outer_apex_start_x"], dg["outer_apex_start_y"], dg["outer_apex_start_z"], 1]))[:3]
ea1 = (Ti @ np.array([dg["outer_apex_end_x"], dg["outer_apex_end_y"], dg["outer_apex_end_z"], 1]))[:3]
for ax, f in zip(axs, (0.25, 0.5, 0.75)):
    s = f * np.linalg.norm(e - a)
    S = sec(raw, s); ax.scatter(S[:, 0], S[:, 1], s=1, c="k", label="scan")
    C1 = sec(cad_e3, s); ax.scatter(C1[:, 0], C1[:, 1], s=1, c="tab:red", alpha=0.4, label="CAD via E3 pose")
    C2 = sec(cad_icp, s); ax.scatter(C2[:, 0], C2[:, 1], s=1, c="tab:blue", alpha=0.4, label="CAD via ICP pose")
    for k, col in (("B3", "green"), ("B10", "orange")):
        try:
            x, y = pt(gold[k]["line_start"], gold[k]["line_end"], s); ax.plot(x, y, "o", ms=12, mfc="none", mec=col, mew=2, label=f"gold {k}")
        except Exception: pass
    x, y = pt(ea0, ea1, s); ax.plot(x, y, "x", ms=12, c="m", mew=3, label="engine B2 apex")
    ax.set_aspect("equal"); ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.grid(alpha=0.3); ax.set_title(f"section at {f:.2f} of gold B3"); ax.legend(fontsize=7, markerscale=3)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=80)
