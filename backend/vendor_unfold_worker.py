"""Serial, time/RSS-bounded isolation for optional vendor CAD development."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from backend.vendor_unfold_preview import unavailable_unfold_preview

MAX_SECONDS = 60.0
MAX_RSS_KIB = 4 * 1024 * 1024
MAX_RESULT_BYTES = 40_000_000
MAX_REQUEST_BYTES = 100_000
_slot = threading.Lock()


def build_unfold_preview_bounded(cad_path, expected_hash, candidates):
    """Optional visual work cannot hold a request indefinitely or exhaust RAM."""
    if not _slot.acquire(blocking=False):
        return unavailable_unfold_preview("UNFOLD_WORKER_BUSY")
    try:
        with tempfile.TemporaryDirectory(prefix="sherman-unfold-worker-") as folder:
            result_path = Path(folder) / "result.json"
            payload = json.dumps({"cad_path": str(cad_path), "expected_hash": expected_hash,
                                  "candidates": candidates, "result_path": str(result_path)}, allow_nan=False)
            if len(payload.encode("utf-8")) > MAX_REQUEST_BYTES:
                return unavailable_unfold_preview("UNFOLD_REQUEST_BUDGET_EXCEEDED")
            request_path = Path(folder) / "request.json"
            request_path.write_text(payload)
            root = Path(__file__).resolve().parents[1]
            start = time.monotonic()
            process = subprocess.Popen([sys.executable, "-m", "backend.vendor_unfold_worker", "--worker", str(request_path)],
                cwd=root, env={**os.environ, "OMP_NUM_THREADS": "1"}, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                while process.poll() is None:
                    if time.monotonic() - start >= MAX_SECONDS:
                        process.kill(); process.wait()
                        return unavailable_unfold_preview("UNFOLD_TIME_BUDGET_EXCEEDED")
                    usage = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                        capture_output=True, text=True, timeout=2)
                    if (usage.returncode != 0 or not usage.stdout.strip()) and process.poll() is None:
                        process.kill(); process.wait()
                        return unavailable_unfold_preview("UNFOLD_RSS_MONITOR_UNAVAILABLE")
                    if usage.returncode == 0 and usage.stdout.strip() and int(usage.stdout.strip()) > MAX_RSS_KIB:
                        process.kill(); process.wait()
                        return unavailable_unfold_preview("UNFOLD_RSS_BUDGET_EXCEEDED")
                    time.sleep(0.1)
                if process.returncode != 0 or not result_path.is_file():
                    return unavailable_unfold_preview("UNFOLD_WORKER_FAILED")
                with result_path.open("rb") as stream:
                    raw = stream.read(MAX_RESULT_BYTES + 1)
                if len(raw) > MAX_RESULT_BYTES:
                    return unavailable_unfold_preview("UNFOLD_RESULT_BUDGET_EXCEEDED")
                return json.loads(raw)
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        result = unavailable_unfold_preview("UNFOLD_WORKER_UNAVAILABLE")
        result["diagnostics"]["error_type"] = type(exc).__name__
        return result
    finally:
        _slot.release()


def _main(request_path):
    from backend.vendor_unfold_preview import build_unfold_preview
    with Path(request_path).open("rb") as stream:
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Worker request exceeds fixed bound")
    payload = json.loads(raw)
    result = build_unfold_preview(payload["cad_path"], payload["expected_hash"], payload["candidates"])
    Path(payload["result_path"]).write_text(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    _main(sys.argv[2])
