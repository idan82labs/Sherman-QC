"""Gold11 relabel sc_07062eee17aa: per-station outer-apex check for a straight fold line (auditor-3 frame; light, read-only).

usage: apex_stations.py FOLD LINES_JSON[:BEND_ID] [...]
FOLD in {flange_-Y, flange_+Y, tab_A, tab_B}. Prints per-line stats of the distance from the
line's station point to the locally measured outer apex (local skin-line fits, every 2 mm)."""
import json, sys
import numpy as np
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply"


def _load_vertices(path):
    with open(path, "rb") as fh:
        head = fh.read(8192)
        k = head.index(b"end_header")
        k = head.index(b"\n", k) + 1
        lines = head[:k].decode("ascii", "replace").splitlines()
        props = [ln for ln in lines if ln.startswith("property") and "list" not in ln]
        nv = int([ln for ln in lines if ln.startswith("element vertex")][0].split()[-1])
        fh.seek(k)
        v = np.frombuffer(fh.read(nv * 4 * len(props)), dtype="<f4").reshape(nv, len(props))
    return v[:, :3].astype(np.float64)


P = _load_vertices(SCAN)
F = json.load(open("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_07062eee17aa/a3x_frame.json"))
O, X, Y, Z = (np.array(F[k]) for k in ("O", "X", "Y", "Z"))
R = np.stack([X, Y, Z])
L = (P - O) @ R.T
# fold configs in the a3 frame: corner (y0,z0), web side sign (+/-Y), leg direction (+/-Z), outer normals
CFG = {
    "flange_-Y": dict(y0=-114.3, web=+1, leg=-1, x=(-147.0, 159.0), web_outer=+1),   # flange goes -Z; outer = web top (+Z) & flange -Y face
    "flange_+Y": dict(y0=110.7, web=-1, leg=-1, x=(-148.0, 158.0), web_outer=+1),
    "tab_A": dict(y0=79.0, web=+1, leg=+1, x=(-2.0, 76.5), web_outer=-1),             # tab goes +Z; outer = web bottom (-Z) & window face (-Y)
    "tab_B": dict(y0=-94.0, web=-1, leg=+1, x=(-1.5, 77.0), web_outer=-1),
}


def fit_line2(pts):
    m = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - m)
    return m, vt[0]


def station_apex(cfg, x0):
    sl = L[np.abs(L[:, 0] - x0) < 0.5]
    q = sl[:, 1:3]
    y0 = cfg["y0"]
    # web outer skin points: 5..10 mm from the corner on the web side, outer skin = the face on web_outer side
    dy = (q[:, 0] - y0) * cfg["web"]
    web = q[(dy > 5) & (dy < 11) & (np.abs(q[:, 1]) < 3)]
    if len(web) < 10:
        return None
    zmed = np.median(web[:, 1])
    web_o = web[(web[:, 1] - zmed) * cfg["web_outer"] > 0]
    # leg outer face: 4..9 mm along the leg, outer = side away from the web
    dz = q[:, 1] * cfg["leg"]
    leg = q[(dz > 4) & (dz < 9) & (np.abs(q[:, 0] - y0) < 3)]
    if len(leg) < 10 or len(web_o) < 5:
        return None
    ymed = np.median(leg[:, 0])
    leg_o = leg[(leg[:, 0] - ymed) * (-cfg["web"]) > 0]
    if len(leg_o) < 5:
        return None
    m1, d1 = fit_line2(web_o)
    m2, d2 = fit_line2(leg_o)
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    s, _ = np.linalg.solve(A, m2 - m1)
    sharp = m1 + s * d1
    # inward directions: along web toward web side, along leg toward leg side
    u1 = d1 if (d1[0] * cfg["web"]) > 0 else -d1
    u2 = d2 if (d2[1] * cfg["leg"]) > 0 else -d2
    bis = (u1 + u2) / np.linalg.norm(u1 + u2)
    rel = q - sharp
    h = rel @ bis
    lat = np.abs(rel @ np.array([-bis[1], bis[0]]))
    cand = (lat < 0.35) & (h > -0.5) & (h < 4.0)
    if cand.sum() < 2:
        return None
    hv = np.sort(h[cand])[:3].mean()
    return sharp + hv * bis, sharp, bis


def line_point(seg, x0):
    a, b = seg
    t = (x0 - a[0]) / (b[0] - a[0])
    p = a + t * (b - a)
    return p[1:3], (0.0 <= t <= 1.0)


fold = sys.argv[1]
cfg = CFG[fold]
lines = []
for spec in sys.argv[2:]:
    if spec.startswith("raw:"):
        name, coords = spec[4:].split("=")
        v = [float(c) for c in coords.split(",")]
        seg = np.array([v[:3], v[3:]])
    else:
        path, bid = spec.split(":")
        d = json.load(open(path))
        b = [b for b in d["bends"] if b["bend_id"] == bid][0]
        seg = np.array([b["label_line"]["start"], b["label_line"]["end"]])
        name = path.split("/")[-2] + ":" + bid
    lines.append((name, (seg - O) @ R.T))
stations = []
for x0 in np.arange(cfg["x"][0], cfg["x"][1] + 0.01, 2.0):
    r = station_apex(cfg, x0)
    if r is not None:
        stations.append((x0, r[0]))
print(fold, "stations with a measured apex:", len(stations))
apex = np.array([s[1] for s in stations])
xs = np.array([s[0] for s in stations])
# best straight line through measured apex points (in YZ as functions of X)
cy = np.polyfit(xs, apex[:, 0], 1); cz = np.polyfit(xs, apex[:, 1], 1)
fitd = np.hypot(apex[:, 0] - np.polyval(cy, xs), apex[:, 1] - np.polyval(cz, xs))
print(f"  best-fit straight line through apex stations: dist median {np.median(fitd):.3f} p90 {np.percentile(fitd,90):.3f} max {fitd.max():.3f}")
print(f"  apex bow (Z of apex) range {apex[:,1].min():.2f}..{apex[:,1].max():.2f}; Y range {apex[:,0].min():.2f}..{apex[:,0].max():.2f}")
for name, seg in lines:
    ds, sg = [], []
    for x0, ap in stations:
        p, inside = line_point(seg, x0)
        if not inside:
            continue
        ds.append(np.linalg.norm(p - ap))
    ds = np.array(ds)
    print(f"  {name:40s} n={len(ds):3d} dist-to-apex median {np.median(ds):.3f} p90 {np.percentile(ds,90):.3f} max {ds.max():.3f}")
