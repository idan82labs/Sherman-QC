"""Option 4 DEV runs: E3 base / add / only on every CAD-path cohort scan (guard first, one geometry slot each).
usage: seeded_driver.py COHORT.json SEEDS_DIR OUT_DIR MODE [scan ...]"""
import json, os, subprocess, sys, hashlib
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from research.zeroshot.common.gold_guard import require_runnable
from domains.production.engine.eval.engine_output import load_compiled, load_recipe
from domains.production.engine.eval.procmon import run_monitored
from domains.production.engine.eval.runner import RunnerConfig, engine_env
W = Path("/private/tmp/claude-501/sherman_zeroshot_work")
HERE = Path(__file__).resolve().parent


def run_one(row, seeds_dir, out_root, mode):
    s = row["scan"]
    require_runnable(s, cad_path=True)
    out = out_root / mode / s
    if (out / "summary.json").is_file():
        return json.loads((out / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    cdir = W / "compile" / row["sku"] / "compiled"; rp = W / "compile" / row["sku"] / "dev.recipe.json"
    comp, rec = load_compiled(cdir / "compiled_part.json"), load_recipe(rp)
    scan = Path(os.path.expanduser(row["scan_path"]))
    sha = hashlib.sha256(scan.read_bytes()).hexdigest()
    eng = out / "engine"
    if eng.exists():
        import shutil; shutil.rmtree(eng)
    argv = [str(REPO / ".venv/bin/python"), "-B", str(HERE / "e3_seeded.py"), "--seeds", str(seeds_dir / f"{s}.seeds.json"),
            "--mode", mode, "--", "--compiled", str(cdir / "compiled_part.json"), "--recipe", str(rp), "--scan", str(scan),
            "--out", str(eng), "--scan-sha256", sha, "--recipe-sha256", rec.recipe_sha256, "--time-budget-s", "170",
            "--compiled-sha256", comp.compiled_part_sha256]
    env = engine_env(RunnerConfig(tier_dir=out, out_dir=out, repo_root=REPO))
    env["OMP_NUM_THREADS"] = "1"
    p = run_monitored(argv, cwd=REPO, env=env, timeout_s=180.0, rss_limit_bytes=4 * 2**30,
                      stderr_path=out / "stderr.log", stdout_path=out / "stdout.log")
    r = json.loads((eng / "result.json").read_text()) if (eng / "result.json").is_file() else {}
    iv = r.get("inspection_validity") or {}
    summ = {"scan": s, "mode": mode, "rc": p.returncode, "wall": round(p.wall_s, 2), "rss_gb": round((p.peak_rss_bytes or 0) / 2**30, 2),
            "state": iv.get("state"), "reasons": iv.get("reasons")}
    (out / "summary.json").write_text(json.dumps(summ))
    return summ


def main():
    cohort, seeds_dir, out_root, mode = sys.argv[1:5]
    only = set(sys.argv[5:])
    rows = [r for r in json.loads(Path(cohort).read_text())["rows"] if not only or r["scan"] in only]
    for r in rows:
        print(json.dumps(run_one(r, Path(seeds_dir), Path(out_root), mode)), flush=True)


if __name__ == "__main__":
    main()
