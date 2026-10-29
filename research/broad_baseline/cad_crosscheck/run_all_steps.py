"""Serially run run_one_step.py on every STEP of a BENCHMARK_V2 CONSUMED part with
a benchmark scan. One process at a time, RSS gate < 9 GiB, 600 s timeout,
/usr/bin/time -l for wall time and peak RSS."""
import json, os, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "docs/production/runlog/benchmark_v2/BENCHMARK_V2.json"
OUT = ROOT / "docs/production/runlog/broad_baseline/cad_crosscheck_raw"
PY = ROOT / ".venv/bin/python"
FORBIDDEN = ("incoming_codex_fetch_2026-10-17", "HOLDOUT", "holdout", "sealed", "EHO4")


def python_rss_gib():
    out = subprocess.run(["ps", "-axo", "rss,comm"], capture_output=True, text=True).stdout
    return sum(int(l.split()[0]) for l in out.splitlines()[1:] if "python" in l.lower()) / 1048576


def main():
    d = json.loads(BENCH.read_text())
    jobs = []
    for pid, part in d["parts"].items():
        if part["split"] != "CONSUMED" or part["n_benchmark_scans"] == 0:
            continue
        for f in part["cad"]["files"]:
            if f["format"].lower() in ("stp", "step"):
                path = os.path.expanduser(f["path"])
                if any(t in path for t in FORBIDDEN):
                    print("SKIP forbidden", path); continue
                jobs.append((pid, path, f["sha256"]))
    summary = []
    for pid, path, sha in jobs:
        while python_rss_gib() >= 9.0:
            time.sleep(10)
        out = OUT / f"{pid}.json"
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        t0 = time.time()
        try:
            p = subprocess.run(["/usr/bin/time", "-l", str(PY), str(Path(__file__).with_name("run_one_step.py")), path, str(out)],
                               capture_output=True, text=True, timeout=600, env=env, cwd=str(ROOT))
            rss = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
            row = {"part_id": pid, "step_path": path, "step_sha256": sha, "returncode": p.returncode,
                   "wall_s": round(time.time() - t0, 2), "peak_rss_bytes": int(rss.group(1)) if rss else None,
                   "stdout": p.stdout.strip()[-400:], "stderr_tail": "\n".join(l for l in p.stderr.splitlines() if "Error" in l or "Traceback" in l)[-600:]}
        except subprocess.TimeoutExpired:
            row = {"part_id": pid, "step_path": path, "returncode": None, "status": "TIMEOUT_600S"}
        print(json.dumps(row)); summary.append(row)
    (OUT / "_run_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
