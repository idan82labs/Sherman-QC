"""One terminal, bounded diagnostic of the approved 37361005 CAD only.

This is not a preview retry. The frozen V2 kernel is read unchanged; diagnostic
output must contain no flat geometry. No catalog, scan or reference input exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.vendor_unfold_worker import MAX_REQUEST_BYTES, MAX_RESULT_BYTES, MAX_RSS_KIB, MAX_SECONDS

PART = "37361005"
CAD_SHA256 = "81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e"
PINNED_SOURCE = {
    "backend/vendor_unfold_preview.py": "f316f56530e2fd6e44f21aa28a9871f43c795714c4b15b652139ee2cfdca461a",
    "backend/vendor_unfold_worker.py": "51b3c249fc0bb0dc889a5ae4d98c6a394672ae869182f6213599b67fe17d6ff1",
    "scripts/extract_step_topology_bend_axes.py": "f089f71466e0f9bba47139c06efdf8bf651e7707fb05c78ca2195f06146b1314",
}


def bounded_diagnostic(cad_path, expected_hash, candidates):
    """Same frozen worker bounds; a fresh child runs the diagnostic-only seam."""
    with tempfile.TemporaryDirectory(prefix="sherman-unfold-diagnostic-") as directory:
        result_path = Path(directory) / "result.json"
        payload = json.dumps({"cad_path": str(cad_path), "expected_hash": expected_hash,
                              "candidates": candidates, "result_path": str(result_path)}, allow_nan=False)
        if len(payload.encode()) > MAX_REQUEST_BYTES:
            return {"status": "DIAGNOSTIC_UNAVAILABLE", "reason": "REQUEST_BUDGET_EXCEEDED"}
        request_path = Path(directory) / "request.json"
        request_path.write_text(payload)
        start = time.monotonic()
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
            cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1"}, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                if time.monotonic() - start >= MAX_SECONDS:
                    return {"status": "DIAGNOSTIC_UNAVAILABLE", "reason": "TIME_BUDGET_EXCEEDED"}
                usage = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                    capture_output=True, text=True, timeout=2)
                if (usage.returncode != 0 or not usage.stdout.strip()) and process.poll() is None:
                    return {"status": "DIAGNOSTIC_UNAVAILABLE", "reason": "RSS_MONITOR_UNAVAILABLE"}
                if usage.returncode == 0 and usage.stdout.strip() and int(usage.stdout.strip()) > MAX_RSS_KIB:
                    return {"status": "DIAGNOSTIC_UNAVAILABLE", "reason": "RSS_BUDGET_EXCEEDED"}
                time.sleep(0.1)
            if process.returncode != 0 or not result_path.is_file():
                return {"status": "DIAGNOSTIC_UNAVAILABLE", "reason": "WORKER_FAILED"}
            with result_path.open("rb") as stream:
                raw = stream.read(MAX_RESULT_BYTES + 1)
            if len(raw) > MAX_RESULT_BYTES:
                return {"status": "DIAGNOSTIC_UNAVAILABLE", "reason": "RESULT_BUDGET_EXCEEDED"}
            result = json.loads(raw)
            if not isinstance(result, dict) or result.get("status") == "AVAILABLE" or any(
                result.get(key) for key in ("vertices", "triangles", "bend_lines", "flange_surfaces")
            ):
                return {"status": "DIAGNOSTIC_UNAVAILABLE", "reason": "INVALID_DIAGNOSTIC_ENVELOPE"}
            return result
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


def worker(request_path):
    from backend.vendor_unfold_diagnostic import diagnose_unfold_preview
    with Path(request_path).open("rb") as stream:
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Bounded request required")
    request = json.loads(raw)
    result = diagnose_unfold_preview(request["cad_path"], request["expected_hash"], request["candidates"])
    Path(request["result_path"]).write_text(json.dumps(result, allow_nan=False, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    args = parser.parse_args()
    bindings = {**PINNED_SOURCE, "backend/vendor_unfold_diagnostic.py": args.diagnostic_sha256,
                "scripts/diagnose_vendor_unfold_case.py": args.runner_sha256}
    for name, expected in bindings.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Audited diagnostic source binding failed")
    report_path = ROOT / "artifacts/vendor-unfold-preview/37361005-diagnostic-v1.json"
    fixture = json.loads((ROOT / f"sites/vendor-tagging-demo/public/vendor-corpus/{PART}.json").read_text())
    if fixture.get("part_number") != PART or fixture.get("cad_sha256") != CAD_SHA256:
        raise ValueError("Pinned fixture identity mismatch")
    report = {"protocol": "VENDOR_UNFOLD_37361005_DIAGNOSTIC_V1", "status": "RUNNING", "part": PART,
        "cad_sha256": CAD_SHA256, "source_bindings": bindings,
        "limits": {"wall_seconds": MAX_SECONDS, "rss_kib": MAX_RSS_KIB, "result_bytes": MAX_RESULT_BYTES},
        "hypothesis": "Localize frozen V2 wall span, slab, opposite plane or native translated-curve refusal",
        "acceptance_changed": False, "manufacturing_ready": False, "result": None}
    # Exclusive creation prevents reopening either a completed or interrupted attempt.
    with report_path.open("x") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"))
    try:
        report["result"] = bounded_diagnostic(ROOT.parent / "source-assets/02_CAD_QC_3D/37361005_01.stp",
                                              CAD_SHA256, fixture["candidates"])
        report["status"] = "COMPLETE"
        print(json.dumps(report["result"], allow_nan=False), flush=True)
    except BaseException as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report_path.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
