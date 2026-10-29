"""Markdown tables for CHANGE.md: nominal base vs final per part, and springback final per part.
usage: sweep_md.py A1_ITER3_DIR"""
import json
import sys
from pathlib import Path

M = Path(sys.argv[1])
FAM = {"DEV-EA3E4820DF": "10839000", "DEV-6E71609F74": "23553000", "DEV-4D512B5C56": "38172000",
       "DEV-BDE878D2AF": "11979003", "DEV-AAB8C67C31": "28870000", "DEV-1F88D309A7": "15651004"}


def load(root):
    out = {}
    for f in sorted(root.glob("*/S*/summary.json")):
        out[(f.parent.parent.name, int(f.parent.name[1:]))] = json.loads(f.read_text())
    return out


def nb(d):
    return len(d.get("states", "").split()) if d.get("states") else 0


def wrong(d):
    return len(d.get("wrong_states", [])) if "error" not in d else nb(d)


A, B = load(M / "probes/base"), load(M / "probes/final")
keys = sorted(k for k in set(A) & set(B) if k[0] in FAM)
rows = {}
for k in keys:
    a, b = A[k], B[k]
    r = rows.setdefault(k[0], {"p": 0, "bends": 0, "pw": [0, 0], "w": [0, 0], "ff": [0, 0], "amb": [0, 0], "inv": [0, 0]})
    r["p"] += 1
    r["bends"] += nb(a)
    for i, d in enumerate((a, b)):
        w = wrong(d)
        r["w"][i] += w
        r["pw"][i] += int(w > 0)
        r["ff"][i] += len(d.get("fails", []))
        r["amb"][i] += sum(1 for u in d.get("unknown_on_formed", []) if u.get("reason") == "AMBIGUOUS_SEGMENTATION")
        r["inv"][i] += int(d.get("validity") != "VALID")
print("| Family (recipe) | Probes (stages) | Bend states | Probes with a wrong state | Wrong bend states | False FAILs | AMBIGUOUS_SEGMENTATION on formed bends | INVALID |")
print("|---|---|---|---|---|---|---|---|")
tot = {"p": 0, "bends": 0, "pw": [0, 0], "w": [0, 0], "ff": [0, 0], "amb": [0, 0], "inv": [0, 0]}
for sku in FAM:
    if sku not in rows:
        continue
    r = rows[sku]
    print(f"| {FAM[sku]} | {r['p']} | {r['bends']} | {r['pw'][0]} → **{r['pw'][1]}** | {r['w'][0]} → **{r['w'][1]}** | "
          f"{r['ff'][0]} → **{r['ff'][1]}** | {r['amb'][0]} → **{r['amb'][1]}** | {r['inv'][0]} → **{r['inv'][1]}** |")
    for kk in ("p", "bends"):
        tot[kk] += r[kk]
    for kk in ("pw", "w", "ff", "amb", "inv"):
        tot[kk] = [tot[kk][0] + r[kk][0], tot[kk][1] + r[kk][1]]
print(f"| **All** | {tot['p']} | {tot['bends']} | {tot['pw'][0]} → **{tot['pw'][1]}** | {tot['w'][0]} → **{tot['w'][1]}** | "
      f"{tot['ff'][0]} → **{tot['ff'][1]}** | {tot['amb'][0]} → **{tot['amb'][1]}** | {tot['inv'][0]} → **{tot['inv'][1]}** |")
print()
remaining = [f"{FAM[k[0]]} S{k[1]}: {B[k].get('states')}" for k in keys if wrong(B[k]) or B[k].get("fails")]
print("final engine, probes still wrong or failing:", remaining or "none")
print()
S = load(M / "sbprobes/final")
srows = {}
for k, d in S.items():
    if k[0] not in FAM:
        continue
    r = srows.setdefault(k[0], {"p": 0, "bends": 0, "pw": 0, "w": 0, "ang": 0, "inu": 0, "ff": 0, "fp": 0, "of": 0})
    r["p"] += 1
    r["bends"] += nb(d)
    w = wrong(d)
    r["w"] += w
    r["pw"] += int(w > 0)
    ang = d.get("angle_errors", [])
    r["ang"] += len(ang)
    r["inu"] += sum(1 for a in ang if a["inU"])
    r["ff"] += len(d.get("angle_false_fail", []))
    r["fp"] += len(d.get("angle_false_pass", []))
    r["of"] += len(d.get("fails_other", []))
print("| Family (recipe) | Probes | Bend states | Probes with a wrong state | Wrong bend states | Angles measured / within U | False angle FAIL | False angle PASS | Other FAILs |")
print("|---|---|---|---|---|---|---|---|---|")
t = {"p": 0, "bends": 0, "pw": 0, "w": 0, "ang": 0, "inu": 0, "ff": 0, "fp": 0, "of": 0}
for sku in FAM:
    if sku not in srows:
        continue
    r = srows[sku]
    print(f"| {FAM[sku]} | {r['p']} | {r['bends']} | {r['pw']} | {r['w']} | {r['ang']} / {r['inu']} | {r['ff']} | {r['fp']} | {r['of']} |")
    for kk in t:
        t[kk] += r[kk]
print(f"| **All** | {t['p']} | {t['bends']} | {t['pw']} | {t['w']} | {t['ang']} / {t['inu']} | {t['ff']} | {t['fp']} | {t['of']} |")
print()
print("springback, probes wrong or with FAILs:",
      [f"{FAM[k[0]]} S{k[1]}: wrong={[w['bend'] for w in d.get('wrong_states', [])]} ff={d.get('angle_false_fail')} "
       f"fp={d.get('angle_false_pass')} other={d.get('fails_other')}" for k, d in sorted(S.items())
       if k[0] in FAM and (wrong(d) or d.get('angle_false_fail') or d.get('angle_false_pass') or d.get('fails_other'))])
