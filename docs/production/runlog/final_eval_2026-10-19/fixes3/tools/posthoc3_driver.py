"""DIAGNOSTIC - consumed sets, not a held-out score. Re-runs E3 (current engine) on the consumed cases of CASES.json,
one case per hold of the shared geometry lock. Never a RUN, never a new evaluation.
usage: posthoc3_driver.py CASES.json OUT_ROOT"""
import json, os, subprocess, sys, time
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
RUN_E3 = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/fixes/run_e3.py"
LOCK = "/private/tmp/claude-501/sherman_geom.lock"
cases = json.loads(Path(sys.argv[1]).read_text()); root = Path(sys.argv[2]); root.mkdir(parents=True, exist_ok=True)
log = open(root / "posthoc3.jsonl", "a")
for c in cases:
    while True:
        try:
            os.mkdir(LOCK); break
        except FileExistsError:
            time.sleep(3)
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "OMP_NUM_THREADS": "1"}
        r = subprocess.run([sys.executable, RUN_E3, c["scan"], c["compiled"], c["recipe"], str(root / c["case"])],
                           cwd=REPO, env=env, capture_output=True, text=True)
        head = json.loads(r.stdout.splitlines()[0]) if r.stdout else {"err": r.stderr[-500:]}
    finally:
        os.rmdir(LOCK)
    row = {"case": c["case"], "set": c["set"], **{k: head.get(k) for k in ("rc", "wall", "rss_gb", "state", "reasons", "err")}}
    log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
