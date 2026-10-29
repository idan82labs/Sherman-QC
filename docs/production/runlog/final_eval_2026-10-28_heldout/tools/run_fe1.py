"""FE1 one-time engine run driver (final_eval_2026-10-28_heldout).

Each arm runs exactly once per scored scan, on the staged guard-clean copy
docs/production/runlog/fe1_stage_2026-10-28/scans/<gid>.ply, through
  zeroshot_2026-10-25/tools/guarded_slot.sh <staged scan> -- perl alarm(1800) -- cov_worker.py <lane> ...
(gold guard -> one geometry slot: OMP 1, nice 15, process-tree RSS watchdog 16 GiB, 32 GiB for > 3M faces).
The worker is zs_coverage_2026-10-28/tools/cov_worker.py, unchanged (sha bound at freeze seq 41): blind no-CAD config,
lane_cli.run_lane (zs_lane_v1) / lane_cli.run_lane_v3 (zs_lane_v3 rev3) after PIN.verify == [] plus exact pin-sha and
code-fingerprint checks; n_bends 999, t_mm = Z2 self-recipe estimate (v3 reuses the v1 job's value on the same bytes),
max_families 20.

Never re-runs: an arm x scan with an output file, or with a START marker but no output (interrupted driver), is
never launched again; the latter is recorded as a failure (DRIVER_INTERRUPTED). Failures are empty predictions.
usage (repo root): .venv/bin/python -B docs/production/runlog/final_eval_2026-10-28_heldout/tools/run_fe1.py
"""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
D = REPO / "docs/production/runlog/final_eval_2026-10-28_heldout"
GUARD = REPO / "docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh"
WORKER = REPO / "docs/production/runlog/zs_coverage_2026-10-28/tools/cov_worker.py"
PY = REPO / ".venv/bin/python"
ARMS = ["zs_lane_v1", "zs_lane_v3"]          # v1 first: v3 reuses its t estimate (seq 41 config)
JOB_LIMIT_S = 1800                           # per job, inside the slot (perl alarm)
OUTER_LIMIT_S = 7200                         # safety net incl. slot wait (the slot is shared one-at-a-time)


def save(path, doc):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, path)


def main():
    rows = [r.rstrip("\n").split("\t") for r in (D / "runs/SCANS.tsv").read_text().splitlines()[1:] if r.strip()]
    for gid, scan_id, path, sha, faces, rss_gb in rows:
        for arm in ARMS:
            out = D / "runs" / arm / f"{gid}.json"
            start = D / "runs" / arm / f".{gid}.started"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue
            if start.exists():
                save(out, {"gid": gid, "arm": arm, "status": "DRIVER_INTERRUPTED", "lines": [], "n_lines": 0,
                           "errors": {"driver": "START marker without output; never re-run"}})
                continue
            t_prev = None
            if arm == "zs_lane_v3":
                v1 = json.loads((D / "runs/zs_lane_v1" / f"{gid}.json").read_text())
                if v1.get("status") == "OK" and v1.get("t_mm_scan") is not None:
                    t_prev = str(v1["t_mm_scan"])
            env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
            env["SHERMAN_RSS_LIMIT_GB"] = rss_gb
            cmd = ["bash", str(GUARD), path, "--", "perl", "-e", f"alarm {JOB_LIMIT_S}; exec @ARGV or die",
                   str(PY), "-B", str(WORKER), arm, path, sha, str(out)] + ([t_prev] if t_prev else [])
            start.write_text(json.dumps({"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cmd": cmd}))
            t0 = time.time()
            try:
                cp = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=OUTER_LIMIT_S)
                rc, tail = cp.returncode, (cp.stdout + cp.stderr)[-1500:]
            except subprocess.TimeoutExpired as e:
                rc, tail = "OUTER_TIMEOUT", str(e)[-300:]
                subprocess.run(["pkill", "-f", str(out)])
            outer = round(time.time() - t0, 1)
            if out.exists():
                w = json.loads(out.read_text())
            else:
                st = {3: "GUARD_REFUSED", 137: "KILLED_RSS_WATCHDOG", 142: f"TIMEOUT_{JOB_LIMIT_S}s",
                      -14: f"TIMEOUT_{JOB_LIMIT_S}s", "OUTER_TIMEOUT": f"OUTER_TIMEOUT_{OUTER_LIMIT_S}s"}.get(rc, "NO_OUTPUT")
                w = {"status": st, "lines": [], "n_lines": 0, "errors": {"driver": tail}, "wall_s": None, "peak_rss_gib": None}
            w.update({"gid": gid, "arm": arm, "scan_id": scan_id, "staged_scan": path, "face_count_header": int(faces),
                      "rss_limit_gib": int(rss_gb), "job_limit_s": JOB_LIMIT_S, "driver_rc": rc,
                      "driver_tail": tail[-600:], "outer_wall_s_incl_slot_wait": outer,
                      "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            save(out, w)
            print(time.strftime("%H:%M:%S"), gid, arm, w["status"], w.get("n_lines"), w.get("wall_s"),
                  w.get("peak_rss_gib"), f"outer {outer}s", flush=True)


if __name__ == "__main__":
    main()
