"""Serially run align_and_match.py for every AI-labelled scan of a STEP part."""
import json, os, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "docs/production/runlog/benchmark_v2"
O = ROOT / "docs/production/runlog/broad_baseline"
PY = ROOT / ".venv/bin/python"
sys.path.insert(0, str(Path(__file__).parent))
from run_all_steps import python_rss_gib  # noqa


def lines_dir(a):
    ld = (a.get("reverify_2026_10_17") or {}).get("lines_dir") or a["disk_at_manifest_time"]["label_lines_dir"]
    return Path(ld) if ld.startswith("/") else ROOT / ld


def main():
    only = set(sys.argv[1:])
    d = json.loads((B / "BENCHMARK_V2.json").read_text())
    summ = []
    for raw in sorted((O / "cad_crosscheck_raw").glob("[0-9]*.json")):
        pid = raw.stem; R = json.loads(raw.read_text()); step = R["step_path"]
        for s in d["parts"][pid]["scans"]:
            a = s.get("ai_label_run")
            if s.get("role") != "benchmark_scan" or not a or (only and s["scan_id"] not in only):
                continue
            sid = s["scan_id"]; ld = lines_dir(a)
            L = json.loads((ld / "lines.json").read_text())
            if not [b for b in L.get("bends", []) if b.get("status") == "ok"]:
                # zero-bend labels still get registered + CAD bends measured
                pass
            prep = Path(L.get("prep_dir") or "")
            if not (prep / "points_ds.npz").exists():
                for cand in (B / "label_prep_r3" / sid, B / "label_prep" / sid):
                    if (cand / "points_ds.npz").exists(): prep = cand; break
            while python_rss_gib() >= 9.0:
                time.sleep(10)
            t0 = time.time()
            try:
                p = subprocess.run(["/usr/bin/time", "-l", str(PY), str(Path(__file__).with_name("align_and_match.py")), step, str(raw), str(prep),
                                    str(ld / "lines.json"), str(O / "cad_crosscheck_match" / f"{sid}.json"), str(O / "cad_crosscheck_renders" / f"{sid}_match.png")],
                                   capture_output=True, text=True, timeout=600, env={**os.environ, "OMP_NUM_THREADS": "1"}, cwd=str(ROOT))
                rss = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
                row = {"scan_id": sid, "part_id": pid, "rc": p.returncode, "wall_s": round(time.time() - t0, 1),
                       "peak_rss_bytes": int(rss.group(1)) if rss else None, "prep": str(prep), "lines": str(ld / "lines.json"),
                       "out": p.stdout.strip()[-300:], "err": "\n".join(p.stderr.splitlines()[-3:]) if p.returncode else ""}
            except subprocess.TimeoutExpired:
                row = {"scan_id": sid, "part_id": pid, "rc": None, "status": "TIMEOUT_600S"}
            print(json.dumps(row), flush=True); summ.append(row)
    (O / "cad_crosscheck_match" / ("_run_summary.json" if not only else "_run_summary_partial.json")).write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
