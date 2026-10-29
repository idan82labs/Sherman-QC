"""[e3-1.13 candidate copy: REPO = E3C_REPO (the candidate shadow tree), optional E3C_CONFIG]
One monitored E3 or zs (v3 rev2 pin) run on one scan, for DENSE_READINESS. Measurement only.

usage: run_step.py e3 SCAN COMPILED_DIR RECIPE OUT THREADS [TIMEOUT_S]
       run_step.py zs SCAN COMPILED_JSON OUT THREADS [TIMEOUT_S]
E3: the frozen CLI (python -m domains.production.engine.inspect) with the eval runner's env, every thread variable
set to THREADS, RSS limit 32 GiB, --time-budget-s = TIMEOUT_S - 10 (diagnostic: lets a dense run finish so the
output can be compared; the production limits are 180 s / 170 s).
zs: python -m domains.production.zeroshot.lane_cli with SHERMAN_ZS_LANE_PIN=zs_lane_v3 (pin-verified, fail
closed), on OUT/input/scan.ply (a symlink to SCAN). Writes OUT/record.json.
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(os.environ.get("E3C_REPO", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3pose/repo"))
CONFIG = os.environ.get("E3C_CONFIG")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import load_compiled, load_recipe  # noqa: E402
from domains.production.engine.eval.procmon import run_monitored  # noqa: E402
from domains.production.engine.eval.runner import RunnerConfig, engine_env  # noqa: E402

RSS = 32 * 1024**3
TV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
VOLATILE = {"created_at", "duration_s", "peak_rss_bytes", "context"}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def e3(scan, cdir, recipe_p, out, threads, timeout):
    scan, cdir, recipe_p, out = Path(scan), Path(cdir), Path(recipe_p), Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    compiled = load_compiled(cdir / "compiled_part.json")
    recipe = load_recipe(recipe_p)
    argv = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(cdir / "compiled_part.json"),
            "--recipe", str(recipe_p), "--scan", str(scan), "--out", str(out / "engine"),
            "--scan-sha256", sha_file(scan), "--recipe-sha256", recipe.recipe_sha256,
            "--time-budget-s", f"{timeout - 10:g}", "--compiled-sha256", compiled.compiled_part_sha256] + (["--config", CONFIG] if CONFIG else [])
    env = engine_env(RunnerConfig(tier_dir=out, out_dir=out, repo_root=REPO))
    for v in TV:
        env[v] = str(threads)
    proc = run_monitored(argv, cwd=REPO, env=env, timeout_s=float(timeout), rss_limit_bytes=RSS,
                         stderr_path=out / "stderr.log", stdout_path=out / "stdout.log")
    rec = {"mode": "e3", "scan": str(scan), "threads": threads, "rc": proc.returncode, "wall_s": round(proc.wall_s, 2),
           "peak_rss_bytes": proc.peak_rss_bytes, "timed_out": proc.timed_out, "rss_exceeded": proc.rss_exceeded}
    log = out / "engine" / "engine.log"
    stages = []
    if log.exists():
        for line in log.read_text().splitlines():
            m = re.match(r"\[\s*([\d.]+)s\] (\w[\w_]*)$", line)
            if m:
                stages.append((m.group(2), float(m.group(1))))
        m = [re.match(r"\[\s*([\d.]+)s\]", l) for l in log.read_text().splitlines()]
        last = max((float(x.group(1)) for x in m if x), default=None)
        rec["engine_log_last_s"] = last
    rec["stage_starts"] = stages
    rp = out / "engine" / "result.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        iv = r.get("inspection_validity") or {}
        rec["state"] = iv.get("state")
        rec["summary"] = (r.get("summary") or {}).get("state")
        rec["stage_status"] = (r.get("stage_decision") or {}).get("status")
        chars = []
        for b in r.get("bends", []):
            for c in b.get("characteristics", []):
                chars.append([b.get("label"), b.get("state"), c.get("kind"), c.get("side"), c.get("state"),
                              c.get("decision"), c.get("actual"), c.get("uncertainty"), c.get("reason"), c.get("characteristic_id"),
                              list((c.get("evidence") or {}).get("notes") or [])])
        for c in r.get("general_measurements", []) or []:
            chars.append(["GEN", None, c.get("kind"), c.get("side"), c.get("state"), c.get("decision"), c.get("actual"),
                          c.get("uncertainty"), c.get("reason")])
        rec["chars"] = chars
        rec["result_stripped_sha256"] = hashlib.sha256(
            json.dumps(strip(r), sort_keys=True).encode()).hexdigest()
        rec["files_sha256"] = {p.name: sha_file(p) for p in sorted((out / "engine").iterdir())
                               if p.is_file() and p.name not in ("result.json", "engine.log")}
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("mode", "threads", "rc", "wall_s", "peak_rss_bytes", "state", "summary")}),
          flush=True)


def zs(scan, compiled_json, out, threads, timeout):
    scan, out = Path(scan).resolve(), Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "input").mkdir(parents=True)
    os.symlink(scan, out / "input" / "scan.ply")
    env = {k: v for k, v in os.environ.items()}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{REPO}:{REPO / 'backend'}",
                "SHERMAN_ZS_LANE_PIN": "zs_lane_v3", "PYTHONNOUSERSITE": "1"})
    for v in TV:
        env[v] = str(threads)
    argv = [sys.executable, "-B", "-m", "domains.production.zeroshot.lane_cli", "--scan", str(out / "input" / "scan.ply"),
            "--compiled", str(compiled_json), "--out", str(out / "zs_lines.json")]
    proc = run_monitored(argv, cwd=REPO, env=env, timeout_s=float(timeout), rss_limit_bytes=RSS,
                         stderr_path=out / "stderr.log", stdout_path=out / "stdout.log")
    rec = {"mode": "zs", "scan": str(scan), "threads": threads, "rc": proc.returncode, "wall_s": round(proc.wall_s, 2),
           "peak_rss_bytes": proc.peak_rss_bytes, "timed_out": proc.timed_out, "rss_exceeded": proc.rss_exceeded}
    zp = out / "zs_lines.json"
    if zp.exists():
        d = json.loads(zp.read_text())
        rec.update(status=d.get("status"), lane=d.get("lane"), code_fingerprint=d.get("code_fingerprint"),
                   stages=d.get("stages"), n_lines=len(d.get("lines") or []), lane_wall_s=d.get("wall_s"),
                   errors=d.get("errors"), pin_mismatch=d.get("pin_mismatch"), lines=d.get("lines"),
                   lines_sha256=hashlib.sha256(json.dumps(d.get("lines"), sort_keys=True).encode()).hexdigest())
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("mode", "threads", "rc", "wall_s", "peak_rss_bytes", "status", "n_lines")}),
          flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "e3":
        e3(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]),
           int(sys.argv[7]) if len(sys.argv) > 7 else 1800)
    else:
        zs(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 1800)
