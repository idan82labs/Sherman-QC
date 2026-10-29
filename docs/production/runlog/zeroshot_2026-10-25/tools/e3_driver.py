"""E3 (e3-1.12.0) baseline on DEV gold scans with a compiling STEP. No labels read here.
Per case: run final_eval_2026-10-19/fixes/tools/run_e3.py under one geometry slot, then write the engine's measured
OUTER_APEX lines in the raw scan frame as an eval-lines file (predictions_e3.json) for research/broad_baseline/score.py.
Skips the 5 DEV-CHECK cases that have never had engine output (kept for a possible one-time CAD-path evaluation).
usage: e3_driver.py   (reads INVENTORY.json, SPLIT_FROZEN.json, compile dir)"""
import json, os, subprocess, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
H = Path(__file__).resolve().parents[1]
W = Path("/private/tmp/claude-501/sherman_zeroshot_work")
COMP = W / "compile"
RUN_E3 = REPO / "docs/production/runlog/final_eval_2026-10-19/fixes/tools/run_e3.py"
NR = {"sc_eaa89c6d1f1f", "sc_152f1b1aacb2", "sc_c623d2e45bff", "sc_9ef2cf4e52cb", "sc_7a1773eca9fa"}
inv = {r["scan_id"]: r for r in json.loads((H / "INVENTORY.json").read_text())["scans"]}
dev = json.loads((H / "SPLIT_FROZEN.json").read_text())["dev"]["scans"]
out_root = H / "baseline/e3_runs"; out_root.mkdir(parents=True, exist_ok=True)
log = open(H / "baseline/e3_driver.jsonl", "a")
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1"}
sys.path[:0] = [str(REPO), str(REPO / "backend")]
for sid in dev:
    r = inv[sid]; sku = r["cad"].get("sku")
    if not sku or sid in NR:
        continue
    o = out_root / sid; o.mkdir(exist_ok=True)
    pred = o / "predictions_e3.json"
    if pred.exists():
        continue
    cdir, rec = COMP / sku / "compiled", COMP / sku / "dev.recipe.json"
    row = {"case": sid, "sku": sku}
    if not (cdir / "compiled_part.json").is_file():
        row.update(status="NOT_COMPILED")
        body = {"schema": "sherman.broad_baseline.eval_lines.v1", "lane": "e3", "status": "NOT_COMPILED", "frame": "scan_raw", "owned_bend_regions": []}
    else:
        work = W / "e3" / sid
        p = subprocess.run(["/private/tmp/claude-501/sherman_slot.sh", str(REPO / ".venv/bin/python"), str(RUN_E3),
                            os.path.expanduser(r["path"]), str(cdir), str(rec), str(work)],
                           cwd=REPO, env=env, capture_output=True, text=True)
        head = json.loads(p.stdout.splitlines()[0]) if p.stdout.strip().startswith("{") else {"err": p.stderr[-400:]}
        row.update({k: head.get(k) for k in ("rc", "wall", "rss_gb", "state", "reasons", "err")})
        lines, status = [], "ENGINE_FAILED"
        q = subprocess.run([str(REPO / ".venv/bin/python"), "-c", (
            "import sys,json;from pathlib import Path;sys.path[:0]=[sys.argv[4],sys.argv[4]+'/backend'];"
            "from domains.production.engine.eval.engine_output import load_compiled,load_engine_output,load_recipe;"
            "from domains.production.engine.eval.line_metric import engine_lines;"
            "c=load_compiled(Path(sys.argv[1]));r=load_recipe(Path(sys.argv[2]));o=load_engine_output(Path(sys.argv[3]),compiled=c,recipe=r);"
            "print(json.dumps({'ok':o.ok,'status':str(o.status),'lines':engine_lines(o.result,c,locus='OUTER_APEX',frame='raw') if o.ok else []},default=str))"),
            str(cdir / "compiled_part.json"), str(rec), str(work / "engine"), str(REPO)], cwd=REPO, env=env, capture_output=True, text=True)
        try:
            d = json.loads(q.stdout.strip().splitlines()[-1]); lines = d["lines"]
            status = "OK" if d["ok"] else "ENGINE_OUTPUT_NOT_USABLE:" + d["status"]
        except Exception as exc:
            status = "LINE_EXTRACT_FAILED:" + (q.stderr[-300:] or str(exc))
        row["status"] = status; row["lines"] = len(lines)
        body = {"schema": "sherman.broad_baseline.eval_lines.v1", "lane": "e3", "engine": "e3-1.12.0", "status": status,
                "inspection_state": row.get("state"), "reasons": row.get("reasons"), "frame": "scan_raw",
                "scan": {"path": os.path.expanduser(r["path"])}, "line_count": len(lines),
                "owned_bend_regions": [{"bend_id": l.get("bend_id") or l.get("label") or f"E{i}", "line_start": l["line_start"],
                                        "line_end": l["line_end"]} for i, l in enumerate(lines, 1)]}
    pred.write_text(json.dumps(body, indent=1, default=str))
    log.write(json.dumps(row, default=str) + "\n"); log.flush(); print(json.dumps(row, default=str), flush=True)
