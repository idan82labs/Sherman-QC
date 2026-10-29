"""List bend cylinders (R ~ 1-2.5 mm) in the part's CAD STEP, with axis and face extent (read-only)."""
import re, numpy as np, json, sys
STEP = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0 1.stp"
txt = open(STEP, errors='ignore').read()
txt = txt[txt.index('DATA;'):]
ents = {}
for m in re.finditer(r'#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|ENDSEC)', txt, re.S):
    ents[int(m.group(1))] = m.group(2).replace('\n', '')
def refs(s): return [int(x) for x in re.findall(r'#(\d+)', s)]
def nums(s): return [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?', s)]
def point(i):
    s = ents[i]; inner = s[s.index("',") + 2:]; return np.array(nums(inner)[:3])
def direction(i):
    s = ents[i]; inner = s[s.index("',") + 2:]; return np.array(nums(inner)[:3])
def placement(i):
    r = refs(ents[i]); return point(r[0]), direction(r[1])
# face -> surface; collect vertex points of each face's edge loops
def face_vertices(fid):
    pts = []
    stack = refs(ents[fid])[:-1]  # bounds + surface(last)
    seen = set()
    while stack:
        k = stack.pop()
        if k in seen or k not in ents: continue
        seen.add(k); s = ents[k]
        if s.startswith('VERTEX_POINT'):
            pts.append(point(refs(s)[0])); continue
        if s.startswith(('FACE_OUTER_BOUND', 'FACE_BOUND', 'EDGE_LOOP', 'ORIENTED_EDGE', 'EDGE_CURVE')):
            rr = refs(s)
            if s.startswith('EDGE_CURVE'): rr = rr[:2]
            stack.extend(rr)
    return np.array(pts)
out = []
for fid, s in ents.items():
    if not s.startswith('ADVANCED_FACE'): continue
    sid = refs(s)[-1]; ss = ents.get(sid, '')
    if not ss.startswith('CYLINDRICAL_SURFACE'): continue
    r = nums(ss.split(',')[-1])[0]
    if not (0.5 < r < 2.6): continue
    p, a = placement(refs(ss)[0]); a = a / np.linalg.norm(a)
    V = face_vertices(fid)
    if len(V) == 0: continue
    t = (V - p) @ a
    out.append(dict(face=fid, r=round(r, 4), p=p.tolist(), a=a.tolist(), tmin=float(t.min()), tmax=float(t.max()), span=float(t.max() - t.min()),
                    centre=(p + a * 0.5 * (t.min() + t.max())).tolist()))
# group coaxial faces (bend = inner+outer cylinders, possibly split by windows)
groups = []
for f in sorted(out, key=lambda f: -f['span']):
    a = np.array(f['a']); p = np.array(f['p'])
    placed = False
    for g in groups:
        ga = np.array(g['a']); gp = np.array(g['p'])
        if abs(abs(a @ ga) - 1) < 1e-4:
            dv = p - gp; perp = dv - (dv @ ga) * ga
            if np.linalg.norm(perp) < 0.05:
                t = (np.array([f['tmin'], f['tmax']]) + (p - gp) @ ga * np.sign(a @ ga)) if False else None
                g['faces'].append(f); placed = True; break
    if not placed:
        groups.append(dict(a=f['a'], p=f['p'], faces=[f]))
for i, g in enumerate(groups):
    ga = np.array(g['a']); gp = np.array(g['p'])
    ivs = []
    for f in g['faces']:
        s0 = (np.array(f['p']) - gp) @ ga
        sg = np.sign(np.array(f['a']) @ ga)
        lo, hi = sorted([s0 + sg * f['tmin'], s0 + sg * f['tmax']])
        ivs.append((round(lo, 2), round(hi, 2), f['r']))
    lo = min(v[0] for v in ivs); hi = max(v[1] for v in ivs)
    g['interval'] = [lo, hi]; g['ivs'] = sorted(ivs)
    print(f'bend-group {i}: axis {np.round(ga,4)} point {np.round(gp,2)} span {hi-lo:.1f} mm  radii {sorted(set(v[2] for v in ivs))}  pieces {sorted(ivs)}')
json.dump([dict(a=g['a'], p=g['p'], interval=g['interval'], ivs=g['ivs']) for g in groups],
          open('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3c/cad_bends.json', 'w'), indent=1)
