"""Frozen neutral CAD reference evidence campaign; no extraction predictions."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
RAW_ROOT = ROOT.parent / "source-assets/02_CAD_QC_3D"
INVENTORY = ROOT / "artifacts/vendor-cad-corpus-inventory-v1-2026-09-30-attempt1/FILES.jsonl"
PROTOCOL = ROOT / "docs/operations/cad-bend-reference-policy-v1-2026-09-30.md"
PROFILE = "NEUTRAL_CAD_REFERENCE_EVIDENCE_V1"
LIMITS = {"case_seconds": 60, "campaign_seconds": 1230, "rss_kib": 4*1024**2,
          "source_bytes": 5_000_000, "artifact_bytes": 40_000_000, "log_bytes": 1_000_000}
NONCE_ENV = "SHERMAN_REFERENCE_V1_NATIVE_NONCE"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum=4_000_000):
    with open(path, "rb") as stream:
        raw = stream.read(maximum+1)
    if not 0 < len(raw) <= maximum:
        raise ValueError("READ_BUDGET")
    return raw


def write(path, value):
    raw = value if isinstance(value, bytes) else encode(value)
    if len(raw) > LIMITS["artifact_bytes"]:
        raise ValueError("ARTIFACT_BUDGET")
    with open(path, "xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def budget_accounting(folder):
    """Source, logs and evidence all count toward the case aggregate."""
    log_bytes = sum(p.stat().st_size for p in (folder / "stdout.log", folder / "stderr.log") if p.exists())
    artifact_bytes = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    reason = ("CASE_LOG_BUDGET" if log_bytes > LIMITS["log_bytes"] else
              "CASE_ARTIFACT_BUDGET" if artifact_bytes > LIMITS["artifact_bytes"] else None)
    return {"log_bytes": log_bytes, "artifact_bytes": artifact_bytes, "reason": reason}


def bindings(raw):
    unique, metadata = {}, []
    for line in raw.splitlines():
        row = json.loads(line)
        path = row["relative_path"]
        parts = Path(path).parts
        if Path(path).is_absolute() or not parts or any(p in {"..", "."} for p in parts):
            raise ValueError("UNSAFE_INVENTORY_PATH")
        ext = row.get("extension", "").lower()
        reserved = any(any(t in p.casefold() for t in ("scan", "incoming", "eho", "holdout10")) for p in parts)
        if ext in {".step", ".stp"} and not reserved and row.get("disposition") == "supported_format":
            digest = row.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("INVALID_CAD_BINDING")
            size = row.get("file_bytes")
            if type(size) is not int or not 0 < size <= LIMITS["source_bytes"]:
                raise ValueError("SOURCE_BUDGET")
            item = unique.setdefault(digest, {"sha256": digest, "file_bytes": size, "aliases": []})
            if item["file_bytes"] != size or path in item["aliases"]:
                raise ValueError("INCONSISTENT_ALIAS")
            item["aliases"].append(path)
        else:
            metadata.append({"relative_path": path, "extension": ext,
                             "disposition": "UNSUPPORTED_DWG" if ext == ".dwg" else "METADATA_ONLY_NO_READ"})
    cases = []
    for digest, item in sorted(unique.items()):
        item["aliases"].sort()
        cases.append({**item, "relative_path": item["aliases"][0]})
    if len(cases) != 20 or sum(len(c["aliases"]) for c in cases) != 23:
        raise ValueError("EXPECTED_20_UNIQUE_23_STEP_BINDINGS")
    return cases, metadata


def source_pins():
    from backend import vendor_unfold_worker_v4 as worker
    pins = worker._source_pins()
    for relative in (
        "scripts/evaluate_cad_reference_evidence_v1.py",
        "scripts/run_cad_reference_authored_v1.py",
        "scripts/build_cad_bend_reference_evidence_v1.py",
        "scripts/evaluate_vendor_cad_upload_unfold_v2.py",
        "scripts/inventory_vendor_cad_corpus_v2.py",
        "tests/test_build_cad_bend_reference_evidence_v1.py",
        "backend/cad_bend_reference_scoring_v1.py",
        "tests/test_cad_bend_reference_scoring_v1.py",
        "tests/test_evaluate_cad_reference_evidence_v1.py",
        str(PROTOCOL.relative_to(ROOT)),
    ):
        path = ROOT / relative
        pins[str(path.absolute())] = sha(path.read_bytes())
    return dict(sorted(pins.items()))


def freeze(attempt, inventory_sha256):
    raw = read(INVENTORY)
    if sha(raw) != inventory_sha256:
        raise ValueError("INVENTORY_PIN_MISMATCH")
    cases, metadata = bindings(raw)
    pins = source_pins()
    attempt.mkdir(mode=0o700, parents=False, exist_ok=False)
    from scripts.evaluate_vendor_cad_upload_unfold_v2 import _archive_sources
    archive = _archive_sources(attempt, pins)
    write(attempt / "INVENTORY_RELEASED.jsonl", raw)
    frozen = {"profile": PROFILE, "source_pins": pins, **archive, "limits": LIMITS,
              "inventory_sha256": inventory_sha256, "cases": cases, "metadata_only": metadata,
              "raw_root": str(RAW_ROOT), "accuracy": "UNSCORED", "truth": "NONE",
              "qualification": "DIAGNOSTIC_NOT_HTTP_NOT_MANUFACTURING"}
    write(attempt / "FROZEN.json", frozen)
    return frozen


def verify(attempt, frozen_sha256):
    raw = read(attempt / "FROZEN.json")
    if sha(raw) != frozen_sha256:
        raise ValueError("FREEZE_PIN_MISMATCH")
    frozen = json.loads(raw)
    if frozen["profile"] != PROFILE or frozen["limits"] != LIMITS or frozen["raw_root"] != str(RAW_ROOT):
        raise ValueError("FROZEN_PROFILE_MISMATCH")
    if source_pins() != frozen["source_pins"]:
        raise ValueError("SOURCE_CHANGED")
    if sha(read(attempt / "SOURCE_MANIFEST.json")) != frozen["source_manifest_sha256"]:
        raise ValueError("SOURCE_MANIFEST_CHANGED")
    archive_hash = hashlib.sha256()
    with open(attempt / "SOURCE.tar", "rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            archive_hash.update(block)
    if archive_hash.hexdigest() != frozen["source_archive_sha256"]:
        raise ValueError("SOURCE_ARCHIVE_CHANGED")
    inventory_raw = read(attempt / "INVENTORY_RELEASED.jsonl")
    if sha(inventory_raw) != frozen["inventory_sha256"]:
        raise ValueError("RELEASED_INVENTORY_CHANGED")
    cases, metadata = bindings(inventory_raw)
    if cases != frozen["cases"] or metadata != frozen["metadata_only"]:
        raise ValueError("BINDINGS_CHANGED")
    return frozen


def snapshot(case, folder):
    from scripts.inventory_vendor_cad_corpus_v2 import open_root
    fd = open_root(str(RAW_ROOT))
    try:
        components = Path(case["relative_path"]).parts
        for component in components[:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        source = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(source)
            if not stat.S_ISREG(before.st_mode) or before.st_size != case["file_bytes"]:
                raise ValueError("SOURCE_SIZE_OR_TYPE_CHANGED")
            chunks, size = [], 0
            while True:
                block = os.read(source, min(1024**2, LIMITS["source_bytes"]+1-size))
                if not block:
                    break
                chunks.append(block)
                size += len(block)
                if size > LIMITS["source_bytes"]:
                    raise ValueError("SOURCE_BUDGET")
            after = os.fstat(source)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("SOURCE_MUTATED")
            raw = b"".join(chunks)
            if len(raw) != case["file_bytes"] or sha(raw) != case["sha256"]:
                raise ValueError("SOURCE_HASH_CHANGED")
            write(folder / "source.step", raw)
        finally:
            os.close(source)
    finally:
        os.close(fd)


def diagnostic(folder, case):
    from scripts.build_cad_bend_reference_evidence_v1 import collect_evidence
    result = collect_evidence(folder / "source.step")
    write(folder / "EVIDENCE.json", result)
    write(folder / "RESULT.json", {
        "source_sha256": case["sha256"], "status": "EVIDENCE_COLLECTED",
        "truth": "NOT_YET_REVIEWED", "accuracy": "UNSCORED"
    })


def child(attempt, index, frozen_sha256):
    frozen = verify(attempt, frozen_sha256)
    if type(index) is not int or not 0 <= index < len(frozen["cases"]):
        raise ValueError("INVALID_CASE_INDEX")
    folder = attempt / f"case-{index+1:02d}"
    launch = json.loads(read(folder / "CASE_LAUNCH.json"))
    nonce = os.environ.get(NONCE_ENV, "")
    if not nonce or sha(nonce.encode()) != launch["nonce_sha256"] or launch["frozen_sha256"] != frozen_sha256 or launch["case_index"] != index or launch["parent_pid"] != os.getppid():
        raise ValueError("NATIVE_CHILD_UNAUTHORIZED")
    write(folder / "CASE_CLAIM.json", {"pid": os.getpid(), "index": index, "source_sha256": frozen["cases"][index]["sha256"]})
    snapshot(frozen["cases"][index], folder)
    diagnostic(folder, frozen["cases"][index])


def run_case(attempt, index, pin):
    from scripts.evaluate_vendor_cad_upload_unfold_v2 import _ps_rows, _tree, _kill_seen_groups
    from backend import vendor_unfold_worker_v4 as shared
    folder = attempt / f"case-{index+1:02d}"
    nonce = secrets.token_hex(32)
    write(folder / "CASE_LAUNCH.json", {"case_index": index, "frozen_sha256": pin,
          "nonce_sha256": sha(nonce.encode()), "parent_pid": os.getpid()})
    env = {**os.environ, NONCE_ENV: nonce, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    fd = os.open(shared.ADMISSION_LOCK_PATH, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    groups, process, reason, peak = set(), None, None, 0
    with os.fdopen(fd, "r+b") as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("ADMISSION_OWNERSHIP")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shared._check_admission_record(lock)
        started = time.monotonic()
        shared._record(lock, {"state": "ADMITTING", "pgid": None})
        try:
            with open(folder / "stdout.log", "xb") as stdout, open(folder / "stderr.log", "xb") as stderr:
                process = subprocess.Popen([sys.executable, str(Path(__file__).absolute()), "--native-child", str(index),
                  "--attempt", str(attempt), "--frozen-sha256", pin], stdout=stdout, stderr=stderr,
                  cwd=ROOT, env=env, start_new_session=True)
                groups.add(process.pid)
                shared._record(lock, {"state": "RUNNING", "pgid": process.pid})
                while process.poll() is None:
                    rows = _ps_rows()
                    owned = _tree(rows, process.pid)
                    groups.update(rows[p][1] for p in owned)
                    rss = sum(value[2] for value in rows.values() if value[1] in groups)
                    peak = max(peak, rss)
                    if time.monotonic()-started >= LIMITS["case_seconds"]:
                        reason = "CASE_TIME_BUDGET"
                    elif rss > LIMITS["rss_kib"]:
                        reason = "CASE_RSS_BUDGET"
                    else:
                        reason = budget_accounting(folder)["reason"]
                    if reason:
                        break
                    time.sleep(.05)
        except Exception as exc:
            reason = f"OUTER_MONITOR_FAILURE:{type(exc).__name__}"
        finally:
            try:
                cleaned = _kill_seen_groups(groups) if groups else process is None
            except Exception:
                cleaned = False
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    cleaned = False
            if cleaned:
                shared._record(lock, {"state": "QUIESCENT", "pgid": None})
            accounting = budget_accounting(folder)
            log_bytes, artifact_bytes = accounting["log_bytes"], accounting["artifact_bytes"]
            if reason is None and accounting["reason"]:
                reason = "FINAL_" + accounting["reason"]
            receipt = {"case_index": index, "outer_exit": None if process is None else process.returncode,
              "reason": reason, "cleanup_known": cleaned, "elapsed_seconds": time.monotonic()-started,
              "log_bytes": log_bytes, "artifact_bytes": artifact_bytes,
              "peak_sampled_rss_kib": peak, "sampled_not_continuous_hwm": True}
            # Reserve the bounded terminal record before certifying aggregate storage.
            if artifact_bytes + 4096 > LIMITS["artifact_bytes"] and receipt["reason"] is None:
                receipt["reason"] = "FINAL_CASE_ARTIFACT_RECEIPT_RESERVE"
            receipt["terminal_receipt_reserve_bytes"] = 4096
            if len(encode(receipt)) > 4096:
                raise ValueError("TERMINAL_RECEIPT_BUDGET")
            write(folder / "CASE_TERMINAL.json", receipt)
    return receipt


def execute(attempt, pin, pre_path, pre_sha256):
    frozen = verify(attempt, pin)
    raw = read(pre_path)
    pre = json.loads(raw)
    if sha(raw) != pre_sha256 or pre.get("decision") != "GREEN" or pre.get("frozen_sha256") != pin or pre.get("independent_non_author") is not True:
        raise ValueError("INDEPENDENT_PRE_REQUIRED")
    write(attempt / "EXECUTION.json", {"frozen_sha256": pin, "pre_sha256": pre_sha256, "pid": os.getpid()})
    deadline, receipts = time.monotonic()+LIMITS["campaign_seconds"], []
    halt = None
    try:
        for index in range(len(frozen["cases"])):
            folder = attempt / f"case-{index+1:02d}"
            folder.mkdir(exist_ok=False)
            if halt or time.monotonic()+LIMITS["case_seconds"] > deadline:
                row = {"case_index": index, "status": "NOT_EXECUTED", "reason": halt or "CAMPAIGN_BUDGET"}
                write(folder / "CASE_TERMINAL.json", row)
            else:
                try:
                    row = run_case(attempt, index, pin)
                except Exception as exc:
                    row = {"case_index": index, "status": "NOT_EXECUTED_OR_UNKNOWN", "cleanup_known": False,
                           "reason": f"ADMISSION_OR_SUPERVISION_FAILURE:{type(exc).__name__}"}
                    if not (folder / "CASE_TERMINAL.json").exists():
                        write(folder / "CASE_TERMINAL.json", row)
                if not row["cleanup_known"]:
                    halt = "CLEANUP_UNKNOWN"
            receipts.append(row)
    except Exception as exc:
        halt = f"CAMPAIGN_FAILURE:{type(exc).__name__}:{exc}"[:500]
    finally:
        failures = [r["case_index"] for r in receipts if r.get("outer_exit") != 0 or r.get("reason") or not r.get("cleanup_known")]
        write(attempt / "CAMPAIGN_TERMINAL.json", {"profile": PROFILE, "receipts": receipts,
          "halt_reason": halt, "accuracy": "UNSCORED", "unique_denominator": len(frozen["cases"]),
          "alias_denominator": 23, "failed_case_indices": failures,
          "successful_case_count": len(receipts)-len(failures),
          "status": "COMPLETE" if len(receipts)==20 and not halt and not failures else "COMPLETE_WITH_FAILURES" if len(receipts)==20 and not halt else "INCOMPLETE"})
    return 0 if len(receipts)==20 and not halt and not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--native-child", type=int)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--pre", type=Path)
    parser.add_argument("--pre-sha256")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.attempt, args.inventory_sha256)
        return 0
    if args.native_child is not None:
        child(args.attempt, args.native_child, args.frozen_sha256)
        return 0
    return execute(args.attempt, args.frozen_sha256, args.pre, args.pre_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
