"""One monitored E3 run (production limits: 180 s timeout, 170 s budget, RSS cap 16 GiB).
usage: run_c.py SCAN COMPILED_DIR RECIPE MAP|- OUT   (REPO = $E3C_REPO, default the candidate tree work/repo_c;
MAP '-' = no --chirality-map argument, as for the live engine)."""
import hashlib, json, os, re, shutil, sys
from pathlib import Path
W = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work")
REPO = Path(os.environ.get("E3C_REPO", str(W / "repo_c")))
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import load_compiled, load_recipe  # noqa: E402
from domains.production.engine.eval.procmon import run_monitored  # noqa: E402
from domains.production.engine.eval.runner import RunnerConfig, engine_env  # noqa: E402
TV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


scan, cdir, recipe_p, mp, out = sys.argv[1:6]
scan, cdir, recipe_p, out = (Path(x).resolve() for x in (scan, cdir, recipe_p, out))
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
compiled = load_compiled(cdir / "compiled_part.json"); recipe = load_recipe(recipe_p)
argv = [sys.executable, "-m", "domains.production.engine.inspect", "--compiled", str(cdir / "compiled_part.json"),
        "--recipe", str(recipe_p), "--scan", str(scan), "--out", str(out / "engine"), "--scan-sha256", sha_file(scan),
        "--recipe-sha256", recipe.recipe_sha256, "--time-budget-s", "170", "--compiled-sha256", compiled.compiled_part_sha256]
if mp != "-":
    argv += ["--chirality-map", mp]
env = engine_env(RunnerConfig(tier_dir=out, out_dir=out, repo_root=REPO))
for v in TV: env[v] = "1"
proc = run_monitored(argv, cwd=REPO, env=env, timeout_s=180.0, rss_limit_bytes=16 * 1024**3,
                     stderr_path=out / "stderr.log", stdout_path=out / "stdout.log")
rec = {"repo": str(REPO), "scan": str(scan), "scan_sha256": sha_file(scan), "compiled_part_sha256": compiled.compiled_part_sha256,
       "map": mp, "rc": proc.returncode, "wall_s": round(proc.wall_s, 2),
       "peak_rss_bytes": proc.peak_rss_bytes, "timed_out": proc.timed_out, "rss_exceeded": proc.rss_exceeded}
rp = out / "engine" / "result.json"
if rp.exists():
    r = json.loads(rp.read_text())
    iv = r.get("inspection_validity") or {}
    rec["state"] = iv.get("state"); rec["reasons"] = iv.get("reasons")
    rec["summary"] = (r.get("summary") or {}).get("state")
    rec["engine_version"] = (r.get("engine") or {}).get("engine_version")
    chars = []
    for b in r.get("bends", []):
        for c in b.get("characteristics", []):
            chars.append([b.get("label"), b.get("state"), c.get("kind"), c.get("side"), c.get("state"), c.get("decision"),
                          "LEFT_RIGHT_UNCONFIRMED" in ((c.get("evidence") or {}).get("notes") or [])])
    for g in r.get("general_measurements", []) or []:
        c = g.get("characteristic") or g
        chars.append(["GEN", None, c.get("kind"), c.get("side"), c.get("state"), c.get("decision"),
                      "LEFT_RIGHT_UNCONFIRMED" in ((c.get("evidence") or {}).get("notes") or [])])
    rec["chars"] = chars
    rec["n_pass"] = sum(1 for c in chars if c[5] == "PASS"); rec["n_fail"] = sum(1 for c in chars if c[5] == "FAIL")
    rec["n_review"] = sum(1 for c in chars if c[5] == "REVIEW"); rec["n_lr_withheld"] = sum(1 for c in chars if c[6])
    rec["warnings"] = [w for w in r.get("warnings", []) if "LEFT_RIGHT" in w]
cc = out / "engine" / "chirality_check.json"
if cc.exists():
    c = json.loads(cc.read_text())
    rec["chirality"] = {k: c.get(k) for k in ("decision", "reason", "class", "stage_model_id", "pass_before_check", "map_sha256")}
    rec["chirality"]["verdicts"] = [h["verdict"] for h in c.get("hypotheses", [])]
    rec["chirality"]["features"] = [[h["hypothesis"], f["feature"], f["state"], f["proper_material"], f["mirror_material"]]
                                    for h in c.get("hypotheses", []) for f in h["features"] if f["state"] != "UNOBSERVED"][:60]
log = out / "engine" / "engine.log"
if log.exists():
    rec["notes"] = [l for l in log.read_text().splitlines() if re.search(r"chirality|achiral|mirror|best datum|validity", l)][:30]
shutil.rmtree(out / "engine", ignore_errors=True)  # bulky; result facts are in this record
(out / "record.json").write_text(json.dumps(rec, indent=1, default=str))
print(json.dumps({k: rec.get(k) for k in ("rc", "wall_s", "state", "summary", "n_pass", "n_lr_withheld", "reasons")} |
                 {"chir": (rec.get("chirality") or {}).get("decision")}), flush=True)
