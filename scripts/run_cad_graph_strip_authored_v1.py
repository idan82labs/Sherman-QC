"""Serial authored-only native qualification, with immutable attempt receipts."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
TESTS = ["tests/test_cad_finite_trimmed_cylinder_domain_v1.py",
         "tests/test_cad_finite_trimmed_cylinder_qualified_joins_v1.py",
         "tests/test_extract_step_topology_bend_axes_v8.py",
         "tests/test_extract_step_topology_bend_axes_v8_native.py",
         "tests/test_evaluate_vendor_cad_corpus_v7.py"]


def run(out, reason):
    from backend import vendor_unfold_worker_v4 as shared
    from scripts.evaluate_vendor_cad_corpus_v7 import source_pins
    pins = source_pins()
    pins[str(Path(__file__).absolute())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    def write(name, value):
        with (out / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
    write("FROZEN.json", {"source_pins": pins, "tests": TESTS, "seconds": 60,
                         "rss_kib": 4194304, "raw_corpus_access": False,
                         "successor_reason": reason})
    fd = os.open(shared.ADMISSION_LOCK_PATH, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+b") as lock:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("ADMISSION_OWNERSHIP")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shared._check_admission_record(lock)
        shared._record(lock, {"state": "ADMITTING", "pgid": None})
        started, process, peak, failure, cleaned = time.monotonic(), None, 0, None, False
        try:
            env = {**os.environ, "SHERMAN_AUTHORED_NATIVE_PHASE": "1",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "VECLIB_MAXIMUM_THREADS": "1"}
            with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
                process = subprocess.Popen([sys.executable, "-m", "pytest", "-c", "/dev/null",
                    "--noconftest", "-p", "no:cacheprovider", "-q", *TESTS],
                    cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                shared._record(lock, {"state": "RUNNING", "pgid": process.pid})
                while process.poll() is None:
                    rss = sum(r for _, r in shared._group_rows(process.pid)); peak = max(peak, rss)
                    if rss > 4194304 or time.monotonic() - started > 60:
                        failure = "RESOURCE_BUDGET"; break
                    time.sleep(.05)
        except Exception as exc:
            failure = "SUPERVISION_FAILURE:"+type(exc).__name__+":"+str(exc)[:180]
        finally:
            if process is not None:
                try:
                    shared._stop_group(process)
                    cleaned = not shared._group_rows(process.pid)
                except Exception as exc:
                    failure = (failure or "")+";CLEANUP_CHECK_FAILURE:"+type(exc).__name__
            if cleaned:
                shared._record(lock, {"state": "QUIESCENT", "pgid": None})
            receipt = {"outer_exit": None if process is None else process.returncode,
                       "cleanup_known": cleaned, "reason": failure,
                       "elapsed_seconds": time.monotonic() - started,
                       "peak_sampled_rss_kib": peak}
            write("TERMINAL.json", receipt)
    print(json.dumps(receipt))
    print((out / "stdout.log").read_text())
    return 0 if process.returncode == 0 and cleaned and failure is None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.attempt, args.reason))
