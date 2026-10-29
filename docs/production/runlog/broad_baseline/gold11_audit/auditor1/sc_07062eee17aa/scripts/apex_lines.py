"""Fit 3D apex lines (raw scan frame) per fold from the dense apex track; check segment collinearity;
score every gold line against every apex line with the frozen matcher pair_metrics/is_match."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
sys.path.insert(0, ROOT)
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds
T = json.load(open(OUT + '/a1_apex_track.json'))
EXT = json.load(open(OUT + '/a1_fold_extents.json'))
c = np.array(T['frame']['centroid']); R = np.array(T['frame']['R'])
def to_raw(p_pc): return c + np.asarray(p_pc) @ R
def fit3(pts):
    m = pts.mean(0); u, s, vt = np.linalg.svd(pts - m); d = vt[0]
    if d[0] < 0: d = -d
    return m, d
lines = {}
for fold, rows in T['folds'].items():
    runs = EXT[fold]
    pts = []; seg = []
    for r in rows:
        if not r.get('bend'): continue
        x = r['x0']
        k = [i for i, (a, b) in enumerate(runs) if a + 1.0 <= x <= b - 1.0]
        if not k: continue
        pts.append([x, r['apex'][0], r['apex'][1]]); seg.append(k[0])
    pts = np.array(pts); seg = np.array(seg)
    raw = to_raw(pts)
    m, d = fit3(raw)
    resid = np.linalg.norm((raw - m) - np.outer((raw - m) @ d, d), axis=1)
    # endpoints at fold extremes
    lo = min(a for a, b in runs); hi = max(b for a, b in runs)
    def at_pc1(x):
        # point on fitted line whose PC1 coordinate equals x
        mp = (m - c) @ R.T; dp = d @ R.T
        t = (x - mp[0]) / dp[0]
        return m + t * d
    s, e = at_pc1(lo), at_pc1(hi)
    lines[fold] = dict(line_start=s.tolist(), line_end=e.tolist(), length=float(np.linalg.norm(e - s)),
                       resid_med=float(np.median(resid)), resid_max=float(resid.max()), n=int(len(raw)), runs=runs)
    print(f'{fold}: n={len(raw)} span PC1 [{lo},{hi}] len {np.linalg.norm(e-s):.1f} mm  apex-line residual med {np.median(resid):.2f} max {resid.max():.2f}')
    if len(set(seg)) > 1:
        for k in sorted(set(seg)):
            mk, dk = fit3(raw[seg == k])
            ang = np.degrees(np.arccos(min(1, abs(dk @ d))))
            off = np.linalg.norm((mk - m) - ((mk - m) @ d) * d)
            print(f'   segment {k} PC1 {runs[k]}: n={int((seg==k).sum())} angle to whole-fold line {ang:.2f} deg, centroid offset from it {off:.2f} mm')
        mA, dA = fit3(raw[seg == 0]); mB, dB = fit3(raw[seg == 1])
        ang = np.degrees(np.arccos(min(1, abs(dA @ dB))))
        offA = np.linalg.norm((mA - mB) - ((mA - mB) @ dB) * dB)
        print(f'   segment A vs B: angle {ang:.2f} deg, A centroid off B line {offA:.2f} mm')
json.dump(lines, open(OUT + '/a1_apex_lines_raw.json', 'w'), indent=1)
G = load_gold(); th = Thresholds()
print('\nfrozen matcher (5 deg / 4 mm / 60%/60%): gold vs auditor apex lines')
for k, (s, e) in G.items():
    for fold, L in lines.items():
        pm = pair_metrics(np.array([s, e]), np.array([L['line_start'], L['line_end']]))
        if pm['transverse_offset_mm'] > 40: continue
        print(f"  {k} vs {fold:9s}: ang {pm['axis_angle_deg']:.2f}  transv {pm['transverse_offset_mm']:.2f}  gold_ov {pm['gold_span_overlap']:.2f}  pred_ov {pm['prediction_span_overlap']:.2f}  -> {'MATCH' if is_match(pm, th) else 'no'}")
