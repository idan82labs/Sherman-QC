"""Per-characteristic moves between two physical runs (iteration 2's measured run vs this one).

usage: phys_moves.py ROOT_A ROOT_B [THRESH]
Each root holds physical/E3/<case>/engine/result.json. Prints decision, state and summary changes, the number of
measured characteristics in both runs, and every one whose deviation moved by more than THRESH (default 0.05).
"""
import json
import sys
from pathlib import Path

A, B = Path(sys.argv[1]) / "physical/E3", Path(sys.argv[2]) / "physical/E3"
th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
cases = sorted(p.name for p in B.iterdir() if (p / "engine/result.json").exists())
n_meas = moved = 0
changes, big = [], []
for c in cases:
    fa, fb = A / c / "engine/result.json", B / c / "engine/result.json"
    if not fa.exists():
        changes.append(f"{c}: missing in A")
        continue
    a, b = json.loads(fa.read_text()), json.loads(fb.read_text())
    sa = {k: v for k, v in (a.get("summary") or {}).items() if k != "worst"}
    sb = {k: v for k, v in (b.get("summary") or {}).items() if k != "worst"}
    if sa != sb:
        changes.append(f"{c}: summary {sa} -> {sb}")
    wa = ((a.get("summary") or {}).get("worst") or {}).get("characteristic_id")
    wb = ((b.get("summary") or {}).get("worst") or {}).get("characteristic_id")
    if wa != wb:
        changes.append(f"{c}: worst characteristic {wa} -> {wb} (report detail, no decision)")
    ba = {r["bend_uid"]: r for r in a["bends"]}
    for rb in b["bends"]:
        ra = ba.get(rb["bend_uid"])
        if ra is None:
            changes.append(f"{c} {rb['label']}: only in B")
            continue
        if ra["state"] != rb["state"]:
            changes.append(f"{c} {rb['label']}: state {ra['state']} -> {rb['state']}")
        ca = {ch["characteristic_id"]: ch for ch in ra["characteristics"]}
        for ch in rb["characteristics"]:
            x = ca.get(ch["characteristic_id"])
            if x is None:
                changes.append(f"{c} {rb['label']} {ch['kind']}: only in B")
                continue
            if x.get("decision") != ch.get("decision"):
                changes.append(f"{c} {rb['label']} {ch['kind']} {ch.get('side') or ''}: decision {x.get('decision')} -> {ch.get('decision')}")
            dx, dy = x.get("deviation"), ch.get("deviation")
            if dx is None or dy is None:
                if (dx is None) != (dy is None):
                    changes.append(f"{c} {rb['label']} {ch['kind']} {ch.get('side') or ''}: measured {dx} -> {dy}")
                continue
            n_meas += 1
            if abs(dy - dx) > th:
                moved += 1
                big.append(f"{c} {rb['label']} {ch['kind']} {ch.get('side') or ''}: {dx:+.3f} -> {dy:+.3f} "
                           f"(U {x.get('uncertainty')} -> {ch.get('uncertainty')}, {ch.get('decision')})")
print(f"cases {len(cases)}; measured in both {n_meas}; moved > {th}: {moved}; within: {n_meas - moved}")
print("changes:", changes or "none")
for s in big:
    print("  ", s)
