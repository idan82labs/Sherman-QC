"""compare chars of record.json files: cmp.py REC_A REC_B [REC_C ...] -- prints rows where decision or value differs"""
import json, sys
recs = [json.load(open(p)) for p in sys.argv[1:]]
def key(c): return (c[0], c[2], c[3])
maps = [{key(c): c for c in r.get("chars", [])} for r in recs]
print("summaries:", [r.get("summary") for r in recs], "walls:", [r.get("wall_s") for r in recs])
keys = list(maps[0].keys())
for k in keys:
    row = [m.get(k) for m in maps]
    decs = [x[5] if x else None for x in row]
    vals = [x[6] if x else None for x in row]
    diff = len(set(decs)) > 1 or any(v is not None and vals[0] is not None and abs(v - vals[0]) > 0.05 for v in vals)
    if diff or "-a" in sys.argv[:0]:
        print(f"{k[0]:>4} {k[1]:<18} {str(k[2]):<4} " + " | ".join(f"{d or '-':<7} {('%.3f' % v) if v is not None else '   -   '} U{('%.3f' % x[7]) if x and x[7] else '-'}" for d, v, x in zip(decs, vals, row)))
