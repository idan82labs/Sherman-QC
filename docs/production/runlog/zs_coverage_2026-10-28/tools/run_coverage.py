"""Coverage driver: every RUN_MANIFEST scan x {zs_lane_v1, zs_lane_v3}, one job at a time through sherman_slot.sh.
Durable + resumable: runs/<scan_id>.json holds both lanes; a lane with a recorded outcome is never re-run.
"""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
D = REPO / "docs/production/runlog/zs_coverage_2026-10-28"
SLOT = REPO / "docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh"
PY = REPO / ".venv/bin/python"
WORKER = D / "tools/cov_worker.py"
LANES = ["zs_lane_v1", "zs_lane_v3"]
TIMEOUT = int(os.environ.get("COV_TIMEOUT_S", "1800"))
ORDER = {"DEV": 0, "N1": 1, "OTHER_CONSUMED": 2, "TEST-B": 3}
REUSE = {("f81ebf011f827339a934b62385dd0300463f0e7385c7ee5d2573d293b9127500", "zs_lane_v1"):
         Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/blind_30820000/out/blind.json")}


def faces(p: Path):
    try:
        with open(p, "rb") as f:
            head = f.read(4096)
        if p.suffix.lower() == ".ply":
            for ln in head.split(b"end_header")[0].splitlines():
                w = ln.split()
                if len(w) == 3 and w[0] == b"element" and w[1] == b"face":
                    return int(w[2])
        if p.suffix.lower() == ".stl" and not head[:5].lower().startswith(b"solid"):
            return (p.stat().st_size - 84) // 50
    except OSError:
        pass
    return None


def save(path, doc):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, path)


def main():
    man = json.loads((D / "RUN_MANIFEST.json").read_text())
    scans = sorted(man["scans"], key=lambda s: (ORDER[s["split"]], s["scan_id"]))
    only = set(sys.argv[1:])
    (D / "runs").mkdir(exist_ok=True); (D / "work").mkdir(exist_ok=True)
    for s in scans:
        if only and s["scan_id"] not in only and s["split"] not in only:
            continue
        rp = D / "runs" / f"{s['scan_id']}.json"
        rec = json.loads(rp.read_text()) if rp.exists() else {"scan_id": s["scan_id"], "sha256": s["sha256"],
                                                              "part_id": s["part_id"], "split": s["split"], "lanes": {}}
        nf = faces(Path(s["path"]))
        rec["face_count_header"] = nf
        for lane in LANES:
            if lane in rec["lanes"]:
                continue
            wp = D / "work" / f"{s['scan_id']}__{lane}.json"
            src = REUSE.get((s["sha256"], lane))
            if src is not None and src.exists():
                b = json.loads(src.read_text())
                assert b.get("pin_fp") == "c712c15c0843d89db98eca5d8b757e75976f884520e5543933acd4fabb64555b"
                rec["lanes"][lane] = {"status": b["status"], "n_lines": len(b["lines"]), "lines": b["lines"],
                                      "t_mm_scan": b.get("t_mm_scan"), "stages": b.get("stages"), "errors": b.get("errors"),
                                      "code_fingerprint": b["pin_fp"], "pin_sha256": "0087fb23 (by fingerprint)",
                                      "wall_s": None, "peak_rss_gib": None, "reused": str(src),
                                      "reuse_note": "owner blind demo 2026-10-28 (freeze seq 34): identical pinned lane and blind config (n_bends 999, z2 t, max_families 20)",
                                      "driver_rc": None}
                save(rp, rec); continue
            t_prev = None
            v1 = rec["lanes"].get("zs_lane_v1") or {}
            if lane == "zs_lane_v3" and v1.get("t_mm_scan") is not None:
                t_prev = str(v1["t_mm_scan"])
            env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
            if nf and nf > 3_000_000:
                env["SHERMAN_RSS_LIMIT_GB"] = "32"
            else:
                env.pop("SHERMAN_RSS_LIMIT_GB", None)
            cmd = ["bash", str(SLOT), str(PY), "-B", str(WORKER), lane, s["path"], s["sha256"], str(wp)] + ([t_prev] if t_prev else [])
            t0 = time.time()
            try:
                cp = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=TIMEOUT)
                rc, tail = cp.returncode, (cp.stdout + cp.stderr)[-800:]
            except subprocess.TimeoutExpired as e:
                rc, tail = "TIMEOUT", str(e)[-300:]
                subprocess.run(["pkill", "-f", str(wp)])
            outer = round(time.time() - t0, 1)
            if wp.exists():
                w = json.loads(wp.read_text())
            else:
                w = {"status": {137: "KILLED_RSS_WATCHDOG", "TIMEOUT": f"TIMEOUT_{TIMEOUT}s"}.get(rc, "NO_OUTPUT"), "lines": [],
                     "n_lines": 0, "errors": {"driver": tail}}
            w["driver_rc"] = rc; w["outer_wall_s_incl_slot_wait"] = outer
            w["rss_limit_gib"] = int(env.get("SHERMAN_RSS_LIMIT_GB", "16"))
            rec["lanes"][lane] = w
            save(rp, rec)
            print(time.strftime("%H:%M:%S"), s["split"], s["scan_id"], s["part_id"], lane, w["status"], w.get("n_lines"),
                  w.get("wall_s"), w.get("peak_rss_gib"), flush=True)


if __name__ == "__main__":
    main()
