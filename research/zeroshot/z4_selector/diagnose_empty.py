"""Why does A0 emit nothing on DEV scans? Reads only the stored A0 run logs/records (no labels besides
gold_count from the phase-1 per-case score rows). Writes z4/a0_empty_diagnosis.json."""
import json, re
from collections import Counter
from pathlib import Path
from research.zeroshot.z4_selector.custody import RUNLOG, OUT, dev_inventory

RUNS = RUNLOG.parents[0] / "broad_baseline/runs/best_f1"


def last_json(path: Path):
    if not path.is_file():
        return None
    txt = path.read_text(errors="replace")
    i = txt.rfind("\n{")
    for start in ([i + 1] if i >= 0 else []) + [0]:
        try:
            return json.loads(txt[start:])
        except Exception:
            pass
    for line in reversed(txt.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def main():
    inv = dev_inventory()
    rows = json.loads((RUNLOG / "baseline/scores_dev_per_case.json").read_text())["a0"]
    out = {}
    for sid, r in rows.items():
        if sid not in inv or r.get("prediction_count", 0):
            continue
        pid = inv[sid].get("a0_prediction_case_id") or sid
        d = RUNS / pid if (RUNS / pid).is_dir() else RUNLOG / "baseline/a0_runs" / sid
        rec = json.loads((d / "run_record.json").read_text())
        steps = {s["name"]: s for s in rec["steps"]}
        pool = last_json(d / "logs/a0_pool.stdout.log") or {}
        tu = last_json(d / "logs/a0_typed_union.stdout.log") or {}
        tu_step = steps.get("a0_typed_union", {})
        if tu_step.get("timed_out") or (tu_step and tu_step.get("status") != "OK"):
            cause = "TYPED_UNION_TIMEOUT" if tu_step.get("timed_out") else "TYPED_UNION_FAILED"
        elif not pool:
            cause = "POOL_MISSING"
        elif (pool.get("candidate_count") or 0) + (pool.get("line_object_count") or 0) == 0:
            cause = "POOL_EMPTY"
        elif tu.get("counted_bend_component_count") == 0:
            cause = "POOL_NONEMPTY_SOLVER_SELECTED_ZERO"
        else:
            cause = "OTHER"
        out[sid] = {"disposition": r["disposition"], "gold_count": r.get("gold_count"), "cause": cause,
                    "pool_candidates": pool.get("candidate_count"), "pool_line_objects": pool.get("line_object_count"),
                    "pool_plane_points": pool.get("plane_point_count"),
                    "typed_assignment_count": tu.get("typed_assignment_count"),
                    "solver_status": tu.get("solver_status"),
                    "typed_union_wall_s": tu_step.get("wall_s"), "typed_union_timed_out": tu_step.get("timed_out")}
    summ = Counter(v["cause"] for v in out.values())
    (OUT / "a0_empty_diagnosis.json").write_text(json.dumps({"summary": summ, "cases": out}, indent=1))
    print(summ)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
