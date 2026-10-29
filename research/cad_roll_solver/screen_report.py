"""Screen + regression report over solver outputs and unchanged-E1 compiles (no geometry, JSON only).

usage: python -B -m research.cad_roll_solver.screen_report --solver DIR --e1 DIR --skus SKUS.json --names NAMES.json
                --out REPORT.json
- solver DIR: <name>.cad_roll_nominal.json from run.py
- e1 DIR: compile_some.py output (<sku>/compiled/compiled_part.json, compile_some.json)
Regression rule (per file without rolls): solver bends == E1 flange-graph bends (ids), == the compiled
part's bends (graph_bend_id), zero rolls, zero junction bends, and every E1 bend's own solver cross-check
(Ri, t, sweep) agrees to 1e-4 mm / 1e-3 deg where the solver could pair its skins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("solver", "e1", "skus", "names", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    skus = json.loads(Path(a.skus).read_text())
    names = json.loads(Path(a.names).read_text())
    sku_of = {p: s for s, p in skus.items()}
    comp = {r["sku"]: r for r in json.loads((Path(a.e1) / "compile_some.json").read_text())}
    rows = []
    for name, path in names.items():
        doc = json.loads((Path(a.solver) / f"{name}.cad_roll_nominal.json").read_text())
        sku = sku_of[path]
        c = comp.get(sku, {})
        cp = Path(a.e1) / sku / "compiled" / "compiled_part.json"
        compiled = []
        if cp.is_file():
            cj = json.loads(cp.read_text())
            compiled = sorted(b["graph_bend_id"] for b in cj["bends"])
        e1_ids = sorted(b["graph_bend_id"] for b in doc["e1_bends"])
        solver_ids = sorted(b["id"] for b in doc["solver_bends"])
        xc = [b["solver_cross_check"] for b in doc["e1_bends"]]
        paired = [x for x in xc if x.get("paired")]
        worst = max([max(abs(x["ri_diff_mm"]), abs(x["t_diff_mm"])) for x in paired] or [0.0])
        worst_sw = max([abs(x["sweep_diff_deg"]) for x in paired] or [0.0])
        s = doc["summary"]
        row = {"name": name, "sku": sku, "rolls": s["rolls"], "junction_bends": s["junction_bends"],
               "e1_compile": c.get("status"), "e1_compile_reason": c.get("reason"),
               "e1_graph_error": doc["e1"]["flange_graph_error"] or doc["e1"]["part_model_error"],
               "e1_graph_bends": e1_ids, "e1_compiled_bends": compiled, "solver_bends": solver_ids,
               "e1_bends_paired_by_solver": f"{len(paired)}/{len(xc)}",
               "cross_check_worst_mm": worst, "cross_check_worst_deg": worst_sw,
               "large_radius_faces": len(doc["large_radius_curved_faces"]),
               "large_radius_not_roll": [(f["face"], f["radius_mm"], f["r_over_t"], f["e1_role"])
                                         for f in doc["large_radius_curved_faces"] if f["solver"] == "not a roll skin"],
               "unclassified_paired": len(doc["unclassified_paired_features"]),
               "checks_pass": s["all_checks_pass"],
               "roll_faces_overlap_e1_bend_faces": s["roll_faces_overlap_e1_bend_faces"]}
        if s["rolls"] == 0:
            row["regression"] = {
                "solver_eq_e1_graph": solver_ids == e1_ids,
                "solver_eq_e1_compiled": (solver_ids == compiled) if compiled else None,
                "zero_rolls": True, "zero_junction_bends": s["junction_bends"] == 0,
                "cross_check_exact": worst <= 1e-4 and worst_sw <= 1e-3}
            row["regression"]["pass"] = all(v for v in row["regression"].values() if v is not None)
        rows.append(row)
    reg = [r for r in rows if "regression" in r]
    summary = {"files": len(rows), "with_rolls": [r["name"] for r in rows if r["rolls"]],
               "regression_files": len(reg), "regression_pass": sum(r["regression"]["pass"] for r in reg),
               "regression_with_compiled_compare": sum(r["regression"]["solver_eq_e1_compiled"] is not None for r in reg),
               "e1_bends_total": sum(len(r["e1_graph_bends"]) for r in reg),
               "e1_bends_paired_by_solver": sum(int(r["e1_bends_paired_by_solver"].split("/")[0]) for r in reg)}
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r["name"], "rolls", r["rolls"], "jb", r["junction_bends"], "E1", r["e1_compile"], len(r["e1_graph_bends"]),
              "compiled", len(r["e1_compiled_bends"]), "solver", len(r["solver_bends"]), "paired", r["e1_bends_paired_by_solver"],
              "xc", r["cross_check_worst_mm"], r["cross_check_worst_deg"], "bigR-not-roll", len(r["large_radius_not_roll"]),
              "REG", r.get("regression", {}).get("pass"))


if __name__ == "__main__":
    main()
