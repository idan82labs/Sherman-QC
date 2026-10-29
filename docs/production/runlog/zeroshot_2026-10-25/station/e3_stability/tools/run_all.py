"""Run every unique manifest job through the frozen E3 CLI (dense_rel's run_step.py via sherman_slot.sh, 1 job at a time):
native, K=5 seeded 95% subsamples (dense_rel subsample.py), one 0.40 mm thinning (dense_rel thin_exp.py).
usage: run_all.py MANIFEST [--only KEY,...]   (skips variants whose record.json exists)"""
import json, os, subprocess, sys, shutil, time
from pathlib import Path
R = Path(__file__).resolve().parents[1]
DR = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/dense_rel/tools")
PY = "/Users/idantal/ShermanQC/FullProject/.venv/bin/python"
RUN_STEP = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/zeroshot_2026-10-25/station/tools/run_step.py"
SLOT = "/private/tmp/claude-501/sherman_slot.sh"
TO = int(os.environ.get("TO", "180"))
SEEDS = [9501, 9502, 9503, 9504, 9505]
VARIANTS = [("native", None)] + [(f"s95_{s}", ("sub", 0.95, s)) for s in SEEDS] + [("thin040", ("thin", 0.40))]
KEEP_HEAVY = os.environ.get("KEEP_HEAVY") == "1"
man = json.load(open(sys.argv[1]))
only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
jobs = sorted(man["unique_jobs"], key=lambda e: (e["corpus"] != "G1", e["id"]))
log = open(R / "run_all.log", "a")
for j in jobs:
    if only and j["key"] not in only: continue
    jd = R / "runs" / j["key"]; jd.mkdir(parents=True, exist_ok=True)
    (jd / "job.json").write_text(json.dumps(j, indent=1))
    tmp = R / "tmp" / j["key"]; tmp.mkdir(parents=True, exist_ok=True)
    for name, spec in VARIANTS:
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
        p = subprocess.run([SLOT, PY, RUN_STEP, "e3", scan, j["compiled_dir"], j["recipe"], str(out), "1", str(TO)],
                           capture_output=True, text=True)
        if spec is not None and Path(scan).exists(): os.remove(scan)
        rp = out / "record.json"
        if rp.exists():
            rec = json.loads(rp.read_text()); rec["gen"] = gen; rec["variant"] = name; rp.write_text(json.dumps(rec, indent=1, default=str))
        if not KEEP_HEAVY and name != "native":
            for f in ("registered_scan.ply", "deviations.f32", "segment_labels.u16"):
                try: os.remove(out / "engine" / f)
                except FileNotFoundError: pass
            shutil.rmtree(out / "engine" / "display", ignore_errors=True)
        row = {"key": j["key"], "corpus": j["corpus"], "id": j["id"], "variant": name, "rc": p.returncode,
               "wall": round(time.time() - t0, 1), "out": p.stdout.strip()[-300:], "gen": gen}
        log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
print("DONE", flush=True)
