"""FRESH (owner-authorized one-time run): one step. No freeze/custody code is modified or patched.
usage: fresh_step.py build STEP_PATH WORK | compile SKU WORK | e3 SCAN_ID SCAN_PATH SKU WORK
build   = E1-DEV-POLICY-1 via E1's own dev_recipe API (dev_candidates + dev_recipe, FINAL, opaque name/SKU),
          the same rule as the freeze's dev variant; no hand edits.
compile = freeze recipes.compile_recipe (E1 CLI, 300 s, 4 GiB).
e3      = the E3 command line under the frozen standard run conditions (G1 run_conditions), opaque read-only inputs."""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze import physical, recipes  # noqa: E402
from domains.production.engine.eval.freeze.common import dev_sku_for, sha256_file  # noqa: E402
from domains.production.engine.eval.runner import RunnerConfig, engine_env  # noqa: E402


def stage(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
        dst.unlink()
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


cmd = sys.argv[1]
if cmd == "build":
    step, work = Path(sys.argv[2]).expanduser(), Path(sys.argv[3])
    from domains.production.engine.compile import dev_recipe as dr
    sha = sha256_file(step)
    sku = dev_sku_for(sha)
    staged = work / "inputs" / f"cad_{sha[:12]}.step"
    stage(step, staged)
    out = work / "recipes" / sku
    out.mkdir(parents=True, exist_ok=True)
    summ = {"sku": sku, "cad_sha256": sha, "staged_cad": str(staged)}
    try:
        cands, report = dr.dev_candidates(staged)
    except Exception as exc:
        cands, report = [], {"error": f"{type(exc).__name__}: {exc}"[:300]}
    summ["dev_policy_report"] = report
    if not cands:
        summ.update(status="NO_RECIPE", reason="NO_CANDIDATE_BENDS")
    else:
        try:
            rec = dr.dev_recipe(staged, cands, sku=sku, inspection_stage="FINAL")
        except Exception as exc:
            summ.update(status="NO_RECIPE", reason=f"RECIPE_BUILD_REFUSED: {type(exc).__name__}: {exc}"[:400])
        else:
            p = out / "dev.recipe.json"
            summ.update(status="RECIPE_BUILT", recipe_path=str(p), recipe_file_sha256=recipes._write_recipe(p, rec),
                        **recipes._recipe_facts(rec))
    (out / "build.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ.get(k) for k in ("sku", "status", "reason", "bends")}))
elif cmd == "compile":
    sku, work = sys.argv[2], Path(sys.argv[3])
    b = json.loads((work / "recipes" / sku / "build.json").read_text())
    env = engine_env(RunnerConfig(tier_dir=work, out_dir=work))
    rec = recipes.compile_recipe(Path(b["recipe_path"]), Path(b["staged_cad"]), work / "compiled" / "dev" / sku, env=env)
    (work / "recipes" / sku / "compile.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("status", "reason", "bends", "wall_s")}))
elif cmd == "e3":
    from domains.production.engine.eval.engine_output import load_compiled, load_engine_output, load_recipe
    from domains.production.engine.eval.procmon import run_monitored
    sid, scan, sku, work = sys.argv[2], Path(sys.argv[3]).expanduser(), sys.argv[4], Path(sys.argv[5])
    RC = physical._RC
    cid = "fr_" + sid[3:]
    ws = work / "physical" / "E3" / cid
    if ws.exists():
        for p in ws.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        shutil.rmtree(ws)
    scan_sha = sha256_file(scan)
    scan_in, recipe_in, out = ws / "input" / f"scan{scan.suffix.lower()}", ws / "input" / "recipe.json", ws / "engine"
    stage(scan, scan_in)
    stage(work / "recipes" / sku / "dev.recipe.json", recipe_in)
    recipe = load_recipe(recipe_in)
    cp = work / "compiled" / "dev" / sku / "compiled_part.json"
    compiled = load_compiled(cp)
    argv = [sys.executable, *RC["engine_argv_tail"], "--compiled", str(cp), "--recipe", str(recipe_in), "--scan", str(scan_in),
            "--out", str(out), "--scan-sha256", scan_sha, "--recipe-sha256", recipe.recipe_sha256,
            "--time-budget-s", f"{RC['engine_time_budget_s']:g}", "--compiled-sha256", compiled.compiled_part_sha256]
    env = engine_env(RunnerConfig(tier_dir=ws, out_dir=ws, repo_root=REPO))
    proc = run_monitored(argv, cwd=REPO, env=env, timeout_s=float(RC["engine_timeout_s"]),
                         rss_limit_bytes=int(RC["engine_rss_limit_bytes"]), stderr_path=ws / "engine.stderr.log",
                         stdout_path=ws / "engine.stdout.log")
    status, detail = physical._classify(proc, out)
    if status == "OK":
        eo = load_engine_output(out, compiled=compiled, recipe=recipe)
        if eo.status != "OK":
            status, detail = eo.status, (eo.error or ",".join(eo.problems))[:500]
        elif eo.result.scan.sha256 != scan_sha:
            status, detail = "RESULT_INCONSISTENT", "SCAN_SHA_MISMATCH"
    if status == "OK" and sha256_file(scan_in) != scan_sha:
        status, detail = "RESULT_INCONSISTENT", "ENGINE_MODIFIED_INPUT:scan"
    rec = {"case_ref": cid, "scan_id": sid, "sku": sku, "scan_sha256": scan_sha, "status": status, "detail": detail,
           "wall_s": round(proc.wall_s, 3), "peak_rss_bytes": proc.peak_rss_bytes, "engine_dir": str(out),
           "compiled_part_sha256": compiled.compiled_part_sha256, "conditions": {k: RC[k] for k in
           ("engine_argv_tail", "engine_timeout_s", "engine_rss_limit_bytes", "engine_time_budget_s")}}
    (ws / "record.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec[k] for k in ("case_ref", "status", "detail", "wall_s")}))
