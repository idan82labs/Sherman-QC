"""One monitored E3 run (production limits 180 s / budget 170 s, RSS cap 16 GiB) from REPO=$E3M_REPO (default: the
e3-mirror scratch copy; pass the live repo to run live). E3M_* env vars pass through; E3M_LOG = OUT/e3m.jsonl.
usage: run_e3m.py SCAN COMPILED_DIR RECIPE OUT"""
import hashlib, json, os, re, shutil, sys
from pathlib import Path
REPO = Path(os.environ.get("E3M_REPO", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3mirror/repo"))
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import load_compiled, load_recipe  # noqa: E402
from domains.production.engine.eval.procmon import run_monitored  # noqa: E402
from domains.production.engine.eval.runner import RunnerConfig, engine_env  # noqa: E402
RSS = 16 * 1024**3
TV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


scan, cdir, recipe_p, out = map(Path, sys.argv[1:5])
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
compiled = load_compiled(cdir / "compiled_part.json"); recipe = load_recipe(recipe_p)
argv = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(cdir / "compiled_part.json"),
        "--recipe", str(recipe_p), "--scan", str(scan), "--out", str(out / "engine"), "--scan-sha256", sha_file(scan),
        "--recipe-sha256", recipe.recipe_sha256, "--time-budget-s", "170", "--compiled-sha256", compiled.compiled_part_sha256]
env = engine_env(RunnerConfig(tier_dir=out, out_dir=out, repo_root=REPO))
for v in TV: env[v] = "1"
for k, v in os.environ.items():
    if k.startswith("E3M_") and k != "E3M_REPO": env[k] = v
env["E3M_LOG"] = str(out / "e3m.jsonl"); env["E3M_DIAG"] = "1"
proc = run_monitored(argv, cwd=REPO, env=env, timeout_s=180.0, rss_limit_bytes=RSS,
                     stderr_path=out / "stderr.log", stdout_path=out / "stdout.log")
rec = {"repo": str(REPO), "env": {k: v for k, v in env.items() if k.startswith("E3M_")}, "scan": str(scan), "rc": proc.returncode,
       "wall_s": round(proc.wall_s, 2), "peak_rss_bytes": proc.peak_rss_bytes, "timed_out": proc.timed_out,
       "rss_exceeded": proc.rss_exceeded}
rp = out / "engine" / "result.json"
if rp.exists():
    r = json.loads(rp.read_text())
    iv = r.get("inspection_validity") or {}
    rec["state"] = iv.get("state"); rec["reasons"] = iv.get("reasons")
    rec["summary"] = (r.get("summary") or {}).get("state")
    rec["engine_version"] = r.get("engine_version") or (r.get("engine") or {}).get("version")
    chars = []
    for b in r.get("bends", []):
        for c in b.get("characteristics", []):
            chars.append([b.get("label"), b.get("state"), c.get("kind"), c.get("side"), c.get("state"), c.get("decision")])
    for c in r.get("general_measurements", []) or []:
        chars.append(["GEN", None, c.get("kind"), c.get("side"), c.get("state"), c.get("decision")])
    rec["chars"] = chars
    rec["n_pass"] = sum(1 for c in chars if c[5] == "PASS"); rec["n_fail"] = sum(1 for c in chars if c[5] == "FAIL")
    rec["bends_measured"] = sum(1 for b in r.get("bends", []) if any(c.get("state") == "MEASURED" for c in b.get("characteristics", [])))
    rec["n_bends"] = len(r.get("bends", []))
log = out / "engine" / "engine.log"
if log.exists():
    rec["notes"] = [l for l in log.read_text().splitlines() if re.search(r"best datum|mirror|pose margin|pose well|symmetric|twin hole", l)][:40]
e3m = out / "e3m.jsonl"
if e3m.exists(): rec["e3m"] = [json.loads(l) for l in e3m.read_text().splitlines() if l.strip()]
for f in ("registered_scan.ply", "deviations.f32", "segment_labels.u16"):
    try: os.remove(out / "engine" / f)
    except FileNotFoundError: pass
shutil.rmtree(out / "engine" / "display", ignore_errors=True)
(out / "record.json").write_text(json.dumps(rec, indent=1, default=str))
print(json.dumps({k: rec.get(k) for k in ("rc", "wall_s", "peak_rss_bytes", "state", "summary", "n_pass", "reasons")}), flush=True)
