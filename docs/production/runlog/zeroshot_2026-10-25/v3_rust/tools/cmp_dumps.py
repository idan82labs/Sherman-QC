"""Compare a zs_lane_v3_rs DEV dump with the frozen zs_lane_v3 revision-1 DEV dump, section by section
(Z3 candidates with every pooled field and the D1 snap, pooled X, Z2 lines, v2 evidence: sweep, splits, pairs).
Bitwise: JSON values must be equal; floats are reported with their max |difference| when not.
usage: .venv/bin/python -B docs/.../v3_rust/tools/cmp_dumps.py MINE_DIR REF_DIR --out JSON"""
import argparse, json, sys
from pathlib import Path

def walk_diff(a, b, path, out, tol_abs=0.0):
    """Collect (path, kind, magnitude) differences between JSON values a and b."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "missing_key", None)); continue
            walk_diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "len", (len(a), len(b)))); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk_diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b and not (a is None and isinstance(b, float) and b != b) and not (b is None and isinstance(a, float) and a != a):
            out.append((path, "value", (a, b)))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b and not (a != a and b != b):
            d = abs(float(a) - float(b))
            out.append((path, "float", d))
    elif a != b:
        out.append((path, "value", (a, b)))


SECTIONS = ("candidates", "X", "z2_lines", "evidence")


def compare(mine, ref):
    res = {}
    for s in SECTIONS:
        diffs = []
        walk_diff(mine.get(s), ref.get(s), s, diffs)
        fl = [d for d in diffs if d[1] == "float"]
        other = [d for d in diffs if d[1] != "float"]
        res[s] = {"n_diff": len(diffs), "n_float": len(fl), "max_abs": max([d[2] for d in fl], default=0.0),
                  "structural": [f"{p} {k} {v}" for p, k, v in other[:10]], "n_structural": len(other),
                  "worst": sorted([(d[2], d[0]) for d in fl], reverse=True)[:5]}
    return res


ap = argparse.ArgumentParser(); ap.add_argument("mine"); ap.add_argument("ref"); ap.add_argument("--out", default="")
a = ap.parse_args()
res = {"scans": 0, "identical": 0, "missing": [], "diff": {}, "max_abs": {}}
for q in sorted(Path(a.ref).glob("*.json")):
    p = Path(a.mine) / q.name
    if not p.is_file():
        res["missing"].append(q.stem); continue
    mine, ref = json.loads(p.read_text()), json.loads(q.read_text())
    res["scans"] += 1
    cmp = compare(mine, ref)
    for k, v in cmp.items():
        res["max_abs"][k] = max(res["max_abs"].get(k, 0.0), v["max_abs"])
    if all(v["n_diff"] == 0 for v in cmp.values()):
        res["identical"] += 1
    else:
        res["diff"][q.stem] = {k: v for k, v in cmp.items() if v["n_diff"]}
print(json.dumps({k: v for k, v in res.items() if k != "diff"}, indent=1))
for s, d in res["diff"].items():
    print(s, {k: (v["n_float"], v["max_abs"], v["n_structural"], v["structural"][:2], v["worst"][:2]) for k, v in d.items()})
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
