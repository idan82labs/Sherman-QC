"""e3-1.13 candidate sweep: the survey protocol (native, 5 seeded 95% subsamples 9501-9505, 0.40 mm thinning) through the
candidate CLI (run_cand.py via sherman_slot.sh, one job per slot). usage:
  sweep.py MANIFEST OUTROOT CONFIG|- VARIANTS(comma|all) WORKER NWORKERS   (skips variants whose record.json exists)"""
import json, os, subprocess, sys, shutil, time
from pathlib import Path
S = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
DR = S / "dense_rel/tools"
PY = "/Users/idantal/ShermanQC/FullProject/.venv/bin/python"
RUN = str(S / "e3c/tools/run_cand.py")
SLOT = "/private/tmp/claude-501/sherman_slot.sh"
SEEDS = [9501, 9502, 9503, 9504, 9505]
ALL = [("native", None)] + [(f"s95_{s}", ("sub", 0.95, s)) for s in SEEDS] + [("thin040", ("thin", 0.40))]
man = json.load(open(sys.argv[1])); root = Path(sys.argv[2]); cfg = sys.argv[3]
want = None if sys.argv[4] == "all" else set(sys.argv[4].split(","))
wi, nw = int(sys.argv[5]), int(sys.argv[6])
jobs = sorted(man["unique_jobs"], key=lambda e: (e["corpus"] != "G1", e["id"]))
jobs = [j for i, j in enumerate(jobs) if i % nw == wi]
log = open(root / f"sweep_w{wi}.log", "a")
env = dict(os.environ)
if cfg != "-": env["E3C_CONFIG"] = cfg
for j in jobs:
    jd = root / "runs" / j["key"]; jd.mkdir(parents=True, exist_ok=True)
    (jd / "job.json").write_text(json.dumps(j, indent=1))
    tmp = root / "tmp" / j["key"]; tmp.mkdir(parents=True, exist_ok=True)
    for name, spec in ALL:
        if want and name not in want: continue
        out = jd / name
        if (out / "record.json").exists(): continue
        t0 = time.time()
        if spec is None:
            scan = j["scan"]; gen = None
        else:
            scan = str(tmp / f"{name}.ply")
            argv = [PY, str(DR / "subsample.py"), j["scan"], scan, str(spec[1]), str(spec[2])] if spec[0] == "sub" else \
                   [PY, str(DR / "thin_exp.py"), j["scan"], scan, str(spec[1])]
            g = subprocess.run(argv, capture_output=True, text=True)
            gen = g.stdout.strip().splitlines()[-1] if g.returncode == 0 and g.stdout.strip() else ("ERR " + g.stderr[-300:])
        p = subprocess.run([SLOT, PY, RUN, "e3", scan, j["compiled_dir"], j["recipe"], str(out), "1", "180"],
                           capture_output=True, text=True, env=env)
        if spec is not None and Path(scan).exists(): os.remove(scan)
        rp = out / "record.json"
        if rp.exists():
            rec = json.loads(rp.read_text()); rec["gen"] = gen; rec["variant"] = name
            rc = out / "engine" / "resample_check.json"
            if rc.exists(): rec["resample_check"] = json.loads(rc.read_text())
            rp.write_text(json.dumps(rec, indent=1, default=str))
        for f in ("registered_scan.ply", "deviations.f32", "segment_labels.u16", "stage_mesh.ply", "final_mesh.ply"):
            for q in (out / "engine").glob(f):
                q.unlink()
        shutil.rmtree(out / "engine" / "display", ignore_errors=True)
        row = {"key": j["key"], "corpus": j["corpus"], "id": j["id"], "variant": name, "rc": p.returncode,
               "wall": round(time.time() - t0, 1), "out": p.stdout.strip()[-200:], "gen": gen}
        log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
print("DONE", flush=True)
