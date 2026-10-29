"""pool.py JOBS.tsv [--workers 2] [--env K=V]: each TSV line = name, scan, compiled_dir, recipe, map|-, out.
Runs tools/run_c.py through tools/slot.sh; skips jobs whose out/record.json exists. Prints one line per job."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
W = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work"
args = sys.argv[1:]; workers = 2; env = dict(os.environ)
jobs_path = args[0]
for i, a in enumerate(args):
    if a == "--workers": workers = int(args[i + 1])
    if a == "--env": k, v = args[i + 1].split("=", 1); env[k] = v
jobs = [l.rstrip("\n").split("\t") for l in open(jobs_path) if l.strip()]
def run(j):
    name, scan, cdir, recipe, mp, out = j
    if os.path.exists(os.path.join(out, "record.json")):
        return f"{name} skip"
    p = subprocess.run([f"{W}/tools/slot.sh", "/Users/idantal/ShermanQC/FullProject/.venv/bin/python", f"{W}/tools/run_c.py",
                        scan, cdir, recipe, mp, out], capture_output=True, text=True, env=env)
    return f"{name} {(p.stdout.strip().splitlines() or [''])[-1]} {p.stderr.strip()[-200:] if p.returncode else ''}"
with ThreadPoolExecutor(workers) as ex:
    for line in ex.map(run, jobs):
        print(line, flush=True)
print("POOL_DONE", flush=True)
