"""A1 iter 7: never fit the reference parabola on fewer than 10 points (a degenerate robust-line rejection)."""
from pathlib import Path
p = Path("/Users/idantal/ShermanQC/FullProject/domains/production/engine/inspect/metrology.py")
s = p.read_text()
old = """            kept = line[2]
            sags.append(_sagitta(sg[ref][kept], dg[ref][kept]))"""
new = """            kept = line[2] if int(line[2].sum()) >= 10 else np.ones(int(ref.sum()), dtype=bool)
            sags.append(_sagitta(sg[ref][kept], dg[ref][kept]))"""
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("patched")
