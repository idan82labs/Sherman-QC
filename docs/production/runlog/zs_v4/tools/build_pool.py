"""zs_lane_v4 development pool (metadata only; no geometry, no engine output).

Pool (freeze-ledger NOTE seq 48): every gold-labelled real scan = the coverage-run scans with frozen gold (DEV 125,
TEST-B 61, N1 4; gold located exactly as tools/score_coverage.py --allow-sealed does, seq 47) + the 27 FE1 scored
scans (gold = adjudicated root ai_bend_lines_final_eval_2026_10_28_v1, GOLD_ROOT_MANIFEST, sha-checked).
Family (bootstrap / LOFO unit) = part_id[:5] (FE1 PREREG convention); a scan whose part id is unknown is its own family.
Part ids are used ONLY for grouping; never as a feature.
Writes docs/production/runlog/zs_v4/POOL.json.  usage (repo root): .venv/bin/python -B <this>
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v1 import scoring as SC  # noqa: E402

COV = REPO / "docs/production/runlog/zs_coverage_2026-10-28"
FE1 = REPO / "docs/production/runlog/final_eval_2026-10-28_heldout"
OUT = REPO / "docs/production/runlog/zs_v4/POOL.json"


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fam5(part_id: str | None, scan_id: str) -> str:
    if not part_id or str(part_id).upper().startswith("UNKNOWN"):
        return f"UNKNOWN:{scan_id}"
    return str(part_id)[:5]


def main() -> None:
    cov = {r.get("scan_id"): r for r in json.loads((COV / "COV_INVENTORY.json").read_text())["rows"]}
    man = json.loads((COV / "RUN_MANIFEST.json").read_text())["scans"]
    led = SC._ledger_frozen()
    inv = SC._inv()
    dev = SC.case_specs("dev")
    rows = []
    for s in man:
        sid = s["scan_id"]
        if sid in dev:
            g = dev[sid]["gold"]; origin = "DEV"; lc = dev[sid].get("label_complete", True)
        elif s["split"] == "TEST-B":
            g = {"kind": "ai_studio", "path": led[sid]["label_path"]}; origin = "TEST-B"; lc = True
        elif s["split"] == "N1":
            g = {"kind": "ai_studio", "path": led[sid]["label_path"]}; origin = "N1"; lc = True
        else:
            continue                                    # no frozen gold (30820000_A01 X/V demo scans)
        gd = json.loads((REPO / g["path"]).read_text())
        pid = s.get("part_id") or (inv.get(sid) or {}).get("part_id")
        runrec = json.loads((COV / "runs" / f"{sid}.json").read_text())
        rows.append({"scan_id": sid, "case_id": sid, "sha256": s["sha256"], "path": s["path"],
                     "face_count": runrec.get("face_count_header"), "origin": origin,
                     "sub": (cov.get(sid) or {}).get("sub"), "part_id": pid, "family5": fam5(pid, sid),
                     "gold": g, "label_complete": bool(lc), "zero_bend": bool(gd.get("no_bends_present")),
                     "n_gold": len(gd.get("bends") or []), "part_class": None,
                     "pinned_outputs": {"zs_lane_v1": f"docs/production/runlog/zs_coverage_2026-10-28/runs/{sid}.json#zs_lane_v1",
                                        "zs_lane_v3": f"docs/production/runlog/zs_coverage_2026-10-28/runs/{sid}.json#zs_lane_v3"}})
    gm = json.loads((FE1 / "adjudication/GOLD_ROOT_MANIFEST.json").read_text())["labels"]
    sm = {s["scan_id"]: s for s in json.loads((FE1 / "SET_MANIFEST.json").read_text())["scored"]}
    scans = {}
    for ln in (FE1 / "runs/SCANS.tsv").read_text().splitlines()[1:]:
        gid, scan_id, staged, sh, fc, _ = ln.split("\t")
        scans[gid] = (scan_id, staged, sh, int(fc))
    for gid, m in sorted(gm.items()):
        assert sha(REPO / m["label_path"]) == m["label_sha256"], gid
        scan_id, staged, sh, fc = scans[gid]
        assert scan_id == m["scan_id"]
        s = sm[scan_id]
        adj = json.loads((FE1 / "adjudication" / gid / "adjudication.json").read_text())
        gd = json.loads((REPO / m["label_path"]).read_text())
        rows.append({"scan_id": scan_id, "case_id": gid, "sha256": sh, "path": staged, "face_count": fc,
                     "origin": "FE1", "sub": s["subset"], "part_id": s["part_id"],
                     "family5": fam5(s["part_id"], scan_id), "gold": {"kind": "ai_studio", "path": m["label_path"]},
                     "label_complete": bool(adj.get("label_complete", True)),
                     "zero_bend": bool(gd.get("no_bends_present")) or bool(adj.get("zero_bend")),
                     "n_gold": len(gd.get("bends") or []), "part_class": adj.get("part_class"),
                     "pinned_outputs": {"zs_lane_v1": f"docs/production/runlog/final_eval_2026-10-28_heldout/runs/zs_lane_v1/{gid}.json",
                                        "zs_lane_v3": f"docs/production/runlog/final_eval_2026-10-28_heldout/runs/zs_lane_v3/{gid}.json"}})
    ids = [r["scan_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate scan in pool"
    shas = [r["sha256"] for r in rows]
    assert len(shas) == len(set(shas)), "duplicate bytes in pool"
    doc = {"schema": "sherman.zs_v4.pool.v1", "created": "2026-10-29",
           "authority": "freeze-ledger NOTE seq 48 (owner: all gold-labelled real scans = v4 DEVELOPMENT pool); seq 47 (sealed gold readable post-hoc)",
           "family_rule": "family5 = part_id[:5]; unknown part id -> own family 'UNKNOWN:<scan_id>'; grouping only, never a feature",
           "counts": {"scans": len(rows), "by_origin": dict(Counter(r["origin"] for r in rows)),
                      "line_scans": sum(1 for r in rows if not r["zero_bend"]),
                      "zero_bend_controls": sum(1 for r in rows if r["zero_bend"]),
                      "families5": len(set(r["family5"] for r in rows)),
                      "gold_lines": sum(r["n_gold"] for r in rows if not r["zero_bend"])},
           "scans": rows}
    OUT.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["counts"], indent=1))


if __name__ == "__main__":
    main()
