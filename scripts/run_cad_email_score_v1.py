"""One-shot, bounded derived-only email-transfer score supervisor."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import score_cad_email_transfer_v1 as scoring


def accounting(folder):
    size = logs = 0
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("OUTPUT_SYMLINK")
        if path.is_file():
            count = path.stat().st_size
            size += count
            if path.name in ("stdout.log", "stderr.log"):
                logs += count
    return {"output_bytes": size, "log_bytes": logs,
            "reason": "LOG_BUDGET" if logs > 1_000_000 else
                      "OUTPUT_BUDGET" if size+8192 > 40_000_000 else None}


def command(args):
    result = [sys.executable, str(ROOT/"scripts/score_cad_email_transfer_v1.py")]
    for name in ("plan", "plan_sha256", "reference", "terminal", "terminal_sha256", "post", "post_sha256",
                 "extraction", "extraction_sha256"):
        value = getattr(args, name, None)
        if value is not None:
            result.extend(["--"+name.replace("_", "-"), str(value)])
    return result+["--output", str(args.out/"SCORE.json")]


def run(args):
    from scripts.evaluate_vendor_cad_upload_unfold_v2 import _ps_rows, _tree, _kill_seen_groups
    plan, _ = scoring.pinned(args.plan, args.plan_sha256)
    if plan.get("source_pins") != scoring.source_pins() or plan.get("limits") != scoring.LIMITS:
        raise ValueError("SCORER_PLAN_CHANGED")
    pre, _ = scoring.pinned(args.pre, args.pre_sha256)
    if (pre.get("decision") != "GREEN" or pre.get("independent_non_author") is not True
            or pre.get("scorer_plan_sha256") != args.plan_sha256):
        raise ValueError("INDEPENDENT_SCORE_PRE_REQUIRED")
    args.out.mkdir(exist_ok=False)
    cmd = command(args)
    scoring.write(args.out/"EXECUTION.json", {"command": cmd, "source_pins": scoring.source_pins(),
                  "plan_sha256": args.plan_sha256, "pre_sha256": args.pre_sha256,
                  "limits": {**scoring.LIMITS, "log_bytes": 1_000_000}, "retry": False,
                  "raw_cad_access": False})
    process, groups, reason, peak, cleaned = None, set(), None, 0, False
    started = time.monotonic()
    try:
        with open(args.out/"stdout.log", "xb") as stdout, open(args.out/"stderr.log", "xb") as stderr:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                     "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            groups.add(process.pid)
            while process.poll() is None:
                rows = _ps_rows()
                groups.update(rows[p][1] for p in _tree(rows, process.pid))
                rss = sum(row[2] for row in rows.values() if row[1] in groups)
                peak = max(peak, rss)
                reason = ("TIME_BUDGET" if time.monotonic()-started >= 30 else
                          "RSS_BUDGET" if rss > 1024**2 else accounting(args.out)["reason"])
                if reason:
                    break
                time.sleep(.05)
    except Exception as exc:
        reason = "SUPERVISION_FAILURE:"+type(exc).__name__
    finally:
        try:
            cleaned = _kill_seen_groups(groups) if groups else process is None
            if process is not None:
                process.wait(timeout=3)
        except Exception:
            cleaned = False
        try:
            final = accounting(args.out)
            reason = reason or final["reason"]
        except Exception:
            final = None
            reason = reason or "FINAL_ACCOUNTING_FAILURE"
        terminal = {"outer_exit": None if process is None else process.returncode,
                    "cleanup_known": cleaned, "reason": reason, "accounting": final,
                    "elapsed_seconds": time.monotonic()-started, "peak_sampled_rss_kib": peak,
                    "sampled_not_continuous_hwm": True, "terminal_reserve_bytes": 8192}
        scoring.write(args.out/"TERMINAL.json", terminal)
    return 0 if process is not None and process.returncode == 0 and cleaned and not reason else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "reference", "terminal", "post", "extraction", "pre", "out"):
        p.add_argument("--"+name, type=Path, required=name != "extraction")
        if name not in ("out", "reference"):
            p.add_argument("--"+name+"-sha256", required=name != "extraction")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
