"""Frozen serial synthetic campaign; never opens corpus or physical CADs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / ".tools/freecad-1.1.3/runtime/FreeCAD.app/Contents/Resources"
UPSTREAM = ROOT / ".tools/sheetmetal-3141e205"
MAX_SECONDS = 120.0
MAX_TREE_RSS_KIB = 4 * 1024 * 1024
MAX_LOG_BYTES = 10_000_000
MAX_RESULT_BYTES = 10_000_000
LIMITS = {"wall_seconds": MAX_SECONDS, "tree_rss_kib": MAX_TREE_RSS_KIB,
          "combined_log_bytes": MAX_LOG_BYTES, "result_bytes": MAX_RESULT_BYTES, "omp_threads": 1}
RUNTIME_IDENTITY = {
    "freecad_version": "1.1.3", "freecad_commit": "145529fe741292ff0b3977a01195bf0247425794",
    "official_dmg_sha256": "f5c0ece7cd7c932466d6effadc0fc6e179b0538a9d9a6a77a6769eae3af2667c",
    "sheetmetal_commit": "3141e205ea340d68e69ff0e257a44e27d9bcb591", "networkx_version": "3.4.2"}
SOURCE_FILES = (
    "backend/vendor_manufacturing_freecad.py", "tests/test_vendor_manufacturing_freecad_native.py",
    "scripts/run_vendor_manufacturing_synthetic.py", "backend/vendor_manufacturing_profile.py",
    "backend/vendor_manufacturing_input.py", "backend/vendor_manufacturing_step_units.py",
    ".tools/sheetmetal-3141e205/SheetMetalNewUnfolder.py",
    ".tools/sheetmetal-3141e205/SheetMetalTools.py", ".tools/sheetmetal-3141e205/SheetMetalText.py",
    ".tools/freecad-1.1.3/runtime/FreeCAD.app/Contents/Resources/bin/python",
    ".tools/freecad-1.1.3/runtime/FreeCAD.app/Contents/Resources/lib/FreeCAD.so",
    ".tools/freecad-1.1.3/runtime/FreeCAD.app/Contents/Resources/lib/Part.so",
    ".tools/freecad-1.1.3/runtime/FreeCAD.app/Contents/Resources/lib/TechDraw.so",
    ".tools/freecad-1.1.3/runtime/FreeCAD.app/Contents/Resources/lib/libFreeCADApp.dylib",
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources():
    files = set(SOURCE_FILES)
    files.update(str(path.relative_to(ROOT)) for path in
                 (ROOT / ".tools/freecad-1.1.3/python-deps/networkx").rglob("*.py"))
    files.update(str(path.relative_to(ROOT)) for path in (RESOURCES / "Mod/Draft").rglob("*.py"))
    if len(files) <= len(SOURCE_FILES) or len(files) > 1500:
        raise ValueError("NETWORKX_SOURCE_SET_MISSING_OR_EXCESSIVE")
    return {name: _hash(ROOT / name) for name in sorted(files)}


def _group_rows(pgid):
    result = subprocess.run(["ps", "-axo", "pid=,pgid=,rss=,stat="], capture_output=True,
                            text=True, timeout=2, check=True)
    rows = []
    for line in result.stdout.splitlines():
        values = line.split()
        if values and int(values[1]) == pgid and not values[3].startswith("Z"):
            rows.append((int(values[0]), int(values[2])))
    return rows


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)
    deadline = time.monotonic() + 3
    while _group_rows(process.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError("PROCESS_GROUP_NOT_QUIESCENT")
        time.sleep(.05)


def _save(record, ledger):
    record.seek(0)
    record.write(json.dumps(ledger, indent=2, allow_nan=False))
    record.truncate(); record.flush(); os.fsync(record.fileno())


def _reject_json_constant(value):
    raise ValueError("NONFINITE_JSON_CONSTANT: " + value)


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("NONFINITE_JSON_FLOAT")
    return result


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _load_json(raw):
    return json.loads(raw, parse_constant=_reject_json_constant, parse_float=_finite_json_float,
                      object_pairs_hook=_unique_json_object)


def _validate_suite_result(result):
    if not isinstance(result, dict):
        raise ValueError("INVALID_RESULT_ENVELOPE")
    cases = result.get("cases")
    if (result.get("status") != "SYNTHETIC_SUITE_PASSED" or type(result.get("passed")) is not int
            or result["passed"] <= 0 or not isinstance(cases, list) or len(cases) != result["passed"]
            or not all(isinstance(name, str) and 0 < len(name) <= 2000 for name in cases)
            or len(set(cases)) != len(cases) or type(result.get("failed")) is not int
            or result["failed"] != 0 or result.get("failed_cases") != []
            or result.get("physical_inputs_read") is not False
            or result.get("manufacturing_release_evidence") is not False
            or not isinstance(result.get("freecad_version"), list)
            or result["freecad_version"][:3] != ["1", "1", "3"]
            or len(result["freecad_version"]) < 8
            or result["freecad_version"][7] != RUNTIME_IDENTITY["freecad_commit"]
            or result.get("upstream_commit") != RUNTIME_IDENTITY["sheetmetal_commit"]):
        raise ValueError("SYNTHETIC_RESULT_NOT_COMPLETE_PASSED_DRAFT")


def execute(record_path, frozen):
    if (not isinstance(frozen, dict) or set(frozen) != {"schema", "limits", "source_sha256", "runtime_identity"}
            or frozen.get("schema") != "SHERMAN_MANUFACTURING_SYNTHETIC_FREEZE_V1"
            or frozen.get("limits") != LIMITS or frozen.get("runtime_identity") != RUNTIME_IDENTITY):
        raise ValueError("FROZEN_MANIFEST_SCHEMA_OR_LIMITS_MISMATCH")
    if frozen.get("source_sha256") != _sources():
        raise ValueError("FROZEN_SOURCE_HASH_MISMATCH")
    with open("/tmp/sherman-manufacturing-unfold.lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with record_path.open("x", encoding="utf-8") as record:
            ledger = {"schema": "SHERMAN_MANUFACTURING_SYNTHETIC_CAMPAIGN_V1", "status": "RUNNING",
                "started_at": datetime.now(timezone.utc).isoformat(), "source_sha256": frozen["source_sha256"],
                "runtime_identity": frozen["runtime_identity"], "inputs": "independently_constructed_synthetic_only",
                "limits": LIMITS, "manufacturing_ready": False, "cut_release_allowed": False,
                "returncode": None, "stop_reason": None, "peak_tree_rss_kib": 0}
            _save(record, ledger)
            process = None
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix="sherman-manufacturing-synthetic-") as folder:
                    folder = Path(folder)
                    result_path = folder / "result.json"
                    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "TMPDIR": str(folder),
                        "HOME": str(folder), "FREECAD_USER_HOME": str(folder),
                        "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": os.pathsep.join(map(str, (RESOURCES / "lib", RESOURCES / "Mod/Draft",
                            ROOT / ".tools/freecad-1.1.3/python-deps", UPSTREAM, ROOT)))}
                    try:
                        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
                            process = subprocess.Popen([str(RESOURCES / "bin/python"),
                                str(ROOT / "tests/test_vendor_manufacturing_freecad_native.py"), "--result", str(result_path)],
                                cwd=ROOT, env=env, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                start_new_session=True)
                            while process.poll() is None:
                                if time.monotonic() - started > MAX_SECONDS:
                                    raise RuntimeError("WALL_BUDGET_EXCEEDED")
                                rows = _group_rows(process.pid)
                                if not rows and process.poll() is None:
                                    raise RuntimeError("RSS_MONITOR_UNAVAILABLE")
                                ledger["peak_tree_rss_kib"] = max(ledger["peak_tree_rss_kib"], sum(rss for _, rss in rows))
                                if ledger["peak_tree_rss_kib"] > MAX_TREE_RSS_KIB:
                                    raise RuntimeError("RSS_BUDGET_EXCEEDED")
                                if stdout.tell() + stderr.tell() > MAX_LOG_BYTES:
                                    raise RuntimeError("LOG_BUDGET_EXCEEDED")
                                time.sleep(.1)
                            if any(pid != process.pid for pid, _ in _group_rows(process.pid)):
                                raise RuntimeError("UNEXPECTED_LIVE_DESCENDANTS")
                    finally:
                        if process is not None:
                            _stop_group(process)
                            ledger["returncode"] = process.returncode
                        remaining = MAX_LOG_BYTES
                        sizes = {name: (folder / f"{name}.log").stat().st_size
                                 if (folder / f"{name}.log").exists() else 0 for name in ("stdout", "stderr")}
                        ledger["observed_log_bytes"] = sizes
                        for name in ("stdout", "stderr"):
                            path = folder / f"{name}.log"
                            if path.is_file():
                                with path.open("rb") as stream:
                                    raw = stream.read(remaining)
                                remaining -= len(raw)
                                ledger[name] = raw.decode("utf-8", errors="replace")
                        if sum(sizes.values()) > MAX_LOG_BYTES:
                            ledger["stop_reason"] = "LOG_BUDGET_EXCEEDED"
                    if result_path.is_file():
                        with result_path.open("rb") as stream:
                            raw = stream.read(MAX_RESULT_BYTES + 1)
                        if len(raw) > MAX_RESULT_BYTES:
                            raise RuntimeError("RESULT_BUDGET_EXCEEDED")
                        result = _load_json(raw)
                        if not isinstance(result, dict):
                            raise ValueError("INVALID_RESULT_ENVELOPE")
                        ledger["result"] = result
                    if ledger["returncode"] != 0 or "result" not in ledger:
                        raise RuntimeError("NATIVE_CAMPAIGN_FAILED_OR_RESULT_MISSING")
                    _validate_suite_result(ledger["result"])
                    ledger["status"] = "COMPLETE" if not ledger["stop_reason"] else "FAILED"
            except BaseException as error:
                ledger.update(status="FAILED", error_type=type(error).__name__, error=str(error)[:1000])
                if ledger["stop_reason"] is None:
                    ledger["stop_reason"] = str(error)[:1000] or type(error).__name__
            finally:
                ledger.update(elapsed_seconds=time.monotonic() - started,
                              finished_at=datetime.now(timezone.utc).isoformat())
                if ledger["elapsed_seconds"] > MAX_SECONDS:
                    ledger.update(status="FAILED", stop_reason="WALL_BUDGET_EXCEEDED")
                _save(record, ledger)
            print(json.dumps({key: ledger[key] for key in ("status", "returncode", "stop_reason",
                "elapsed_seconds", "peak_tree_rss_kib")}))
            return 0 if ledger["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.freeze_manifest is not None:
        if args.frozen_manifest or args.record:
            parser.error("freeze and execute are separate operations")
        frozen = {"schema": "SHERMAN_MANUFACTURING_SYNTHETIC_FREEZE_V1", "limits": LIMITS,
            "source_sha256": _sources(), "runtime_identity": RUNTIME_IDENTITY}
        with args.freeze_manifest.open("x", encoding="utf-8") as target:
            json.dump(frozen, target, indent=2, allow_nan=False)
        print(json.dumps({"manifest": str(args.freeze_manifest), "sha256": _hash(args.freeze_manifest),
                          "file_count": len(frozen["source_sha256"])}))
        return 0
    if args.frozen_manifest is None or args.record is None:
        parser.error("execute requires --frozen-manifest and a new --record")
    return execute(args.record, _load_json(args.frozen_manifest.read_bytes()))


if __name__ == "__main__":
    raise SystemExit(main())
