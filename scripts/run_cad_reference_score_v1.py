"""Bounded derived-only scoring of frozen CAD technical references."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))


def run(output, frozen_sha):
    from scripts.evaluate_vendor_cad_upload_unfold_v2 import _ps_rows, _tree, _kill_seen_groups
    from scripts.score_cad_bend_references_v1 import read
    frozen, _ = read(output / "FROZEN.json", frozen_sha)
    if (output / "EXECUTION.json").exists():
        raise ValueError("TERMINAL_OUTPUT_REUSE_REFUSED")
    receipt_dir = output.with_name(output.name + "-supervision")
    receipt_dir.mkdir(exist_ok=False)
    def write(name, value):
        with (receipt_dir / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
    command = [sys.executable, "scripts/score_cad_bend_references_v1.py", "--execute",
               "--out", str(output), "--frozen-sha256", frozen_sha]
    pins = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "scripts/run_cad_reference_score_v1.py", "scripts/score_cad_bend_references_v1.py",
        "backend/cad_bend_reference_scoring_v1.py", "scripts/evaluate_vendor_cad_upload_unfold_v2.py")}
    write("FROZEN.json", {"command": command, "source_pins": pins, "seconds": 30,
                        "rss_kib": 1024**2, "comparison_frozen_sha256": frozen_sha,
                        "raw_cad_access": False, "retry": False})
    process, groups, reason, peak, cleaned = None, set(), None, 0, False
    started = time.monotonic()
    try:
        with (receipt_dir / "stdout.log").open("xb") as stdout, (receipt_dir / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                    "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            groups.add(process.pid)
            while process.poll() is None:
                rows = _ps_rows()
                owned = _tree(rows, process.pid)
                groups.update(rows[p][1] for p in owned)
                rss = sum(row[2] for row in rows.values() if row[1] in groups)
                peak = max(peak, rss)
                if time.monotonic()-started > 30 or rss > 1024**2:
                    reason = "RESOURCE_BUDGET"
                    break
                time.sleep(.05)
    except Exception as exc:
        reason = "SUPERVISION_FAILURE:" + type(exc).__name__
    finally:
        try:
            cleaned = _kill_seen_groups(groups) if groups else process is None
            if process is not None:
                process.wait(timeout=3)
        except Exception:
            cleaned = False
        terminal = {"outer_exit": None if process is None else process.returncode,
                    "cleanup_known": cleaned, "reason": reason,
                    "elapsed_seconds": time.monotonic()-started,
                    "peak_sampled_rss_kib": peak}
        write("TERMINAL.json", terminal)
    print(json.dumps(terminal))
    return 0 if process is not None and process.returncode == 0 and cleaned and reason is None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frozen-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.out.absolute(), args.frozen_sha256))

