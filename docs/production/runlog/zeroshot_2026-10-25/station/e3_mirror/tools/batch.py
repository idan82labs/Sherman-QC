"""batch.py JOBS.jsonl RUNS_DIR [--workers 2]: each job {id, scan, cdir, recipe, mirror (None|x|y|z), env{}} runs once
through sherman_slot.sh (OMP_NUM_THREADS=1) at nice 10; a mirrored scan is written to RUNS_DIR/tmp and removed after.
Skips jobs whose record.json exists. Appends a row per job to RUNS_DIR/batch.log."""
import json, os, subprocess, sys, time, threading, queue
from pathlib import Path
T = Path(__file__).resolve().parent
PY = "/Users/idantal/ShermanQC/FullProject/.venv/bin/python"
SLOT = "/private/tmp/claude-501/sherman_slot.sh"
jobs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
runs = Path(sys.argv[2]).resolve(); (runs / "tmp").mkdir(parents=True, exist_ok=True)
W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
q = queue.Queue()
for j in jobs: q.put(j)
lock = threading.Lock(); log = open(runs / "batch.log", "a")


def work():
    while True:
        try: j = q.get_nowait()
        except queue.Empty: return
        out = runs / j["id"]
        if (out / "record.json").exists(): continue
        t0 = time.time(); scan = j["scan"]
        if j.get("mirror"):
            scan = str(runs / "tmp" / f"{j['id']}.{j['scan'].rsplit('.', 1)[-1].lower()}")
            g = subprocess.run(["nice", "-n", "10", PY, str(T / "mirror_scan.py"), j["scan"], scan, j["mirror"]],
                               capture_output=True, text=True)
            if g.returncode != 0:
                with lock:
                    log.write(json.dumps({"id": j["id"], "rc": -1, "err": "MIRROR_FAILED " + g.stderr[-300:]}) + "\n"); log.flush()
                continue
        env = dict(os.environ); env.update(j.get("env", {}))
        p = subprocess.run(["nice", "-n", "10", SLOT, PY, str(T / "run_e3m.py"), scan, j["cdir"], j["recipe"], str(out)],
                           capture_output=True, text=True, env=env)
        if j.get("mirror") and os.path.exists(scan): os.remove(scan)
        row = {"id": j["id"], "rc": p.returncode, "wall": round(time.time() - t0, 1), "out": p.stdout.strip()[-400:],
               "err": p.stderr.strip()[-300:] if p.returncode else ""}
        with lock:
            log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)


ths = [threading.Thread(target=work) for _ in range(W)]
for t in ths: t.start()
for t in ths: t.join()
print("DONE", flush=True)
