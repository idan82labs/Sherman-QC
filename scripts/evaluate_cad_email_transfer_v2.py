"""One selected CAD transfer; native modes require frozen independent gates.

Importing this module performs no input access and imports no native runtime.
Neutral and extraction are separate one-shot jobs; extraction requires a pinned
neutral POST and independently frozen technical reference. No unfolding occurs.
"""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILE = "ONE_CAD_EMAIL_TRANSFER_V2_UNCHANGED_V10"
INPUT_NAME = "MEC003744_02.step"
INPUT_SHA256 = "0199d1289e9ff67daabb7b25f9d69de30805b2e81ad43006bb7b916f573eae25"
INPUT_BYTES = 657890
LIMITS = {"case_seconds": 60, "rss_kib": 4*1024**2, "source_bytes": 5_000_000,
          "artifact_bytes": 40_000_000, "log_bytes": 1_000_000, "frozen_archive_bytes": 300_000_000}
NONCE_ENV = "SHERMAN_EMAIL_TRANSFER_V2_NATIVE_NONCE"
INCUMBENT = {"scripts/extract_step_topology_bend_axes_v10.py":
             "ab4feed332f19e232156c4f50d1ec47a18bc13220e8ea9f5e4e741b52c933dc1",
             "backend/cad_finite_native_curve_graph_domain_v3.py":
             "4483165d2c9b2034fb6e30a96e8974931c0e671f4d5b16f74a6539149c33ae41"}


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum=5_000_000):
    with open(path, "rb") as stream:
        raw = stream.read(maximum+1)
    if len(raw) > maximum:
        raise ValueError("READ_BUDGET")
    return raw


def write(path, value):
    raw = value if isinstance(value, bytes) else encode(value)
    if len(raw) > LIMITS["artifact_bytes"]:
        raise ValueError("WRITE_BUDGET")
    with open(path, "xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_input_receipt(receipt):
    if (receipt.get("filename"), receipt.get("sha256"), receipt.get("file_bytes")) != (
            INPUT_NAME, INPUT_SHA256, INPUT_BYTES):
        raise ValueError("SELECTED_INPUT_BINDING")
    hashes = receipt.get("approved_corpus_sha256")
    if not isinstance(hashes, list) or len(hashes) != 20 or any(
            not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
            for x in hashes) or len(set(hashes)) != 20:
        raise ValueError("APPROVED_20_HASH_BINDINGS_REQUIRED")
    if INPUT_SHA256 in hashes:
        raise ValueError("DUPLICATE_SELECTED_INPUT_NO_REPLACEMENT")
    return receipt


def validate_pre(pre, pin, mode):
    if (pre.get("decision") != "GREEN" or pre.get("independent_non_author") is not True
            or pre.get("frozen_sha256") != pin or pre.get("mode") != mode):
        raise ValueError("INDEPENDENT_MODE_PRE_REQUIRED")


def source_pins():
    from scripts.evaluate_vendor_cad_corpus_v9 import source_pins as inherited
    pins = inherited()
    names = ["scripts/evaluate_cad_email_transfer_v2.py",
             "tests/test_evaluate_cad_email_transfer_v2.py",
             "scripts/build_cad_bend_reference_evidence_v1.py",
             "docs/operations/cad-email-transfer-v1-2026-10-01.md",
             "docs/operations/cad-email-transfer-v2-2026-10-01.md",
             "scripts/run_cad_email_render_v1.py",
             "scripts/render_cad_email_transfer_v1.py",
             "tests/test_render_cad_email_transfer_v1.py"]
    for name in names:
        path = ROOT / name
        pins[str(path)] = sha(path.read_bytes())
    for name, expected in INCUMBENT.items():
        if pins.get(str(ROOT/name)) != expected:
            raise ValueError("INCUMBENT_CHANGED")
    return dict(sorted(pins.items()))


def selected_bytes(attempt):
    # No-follow both containing directory and leaf; no arbitrary source argument.
    fd = os.open(attempt / "input", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        source = os.open(INPUT_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(source)
            if not stat.S_ISREG(before.st_mode) or before.st_size != INPUT_BYTES:
                raise ValueError("INPUT_SIZE_OR_TYPE")
            chunks, size = [], 0
            while size <= LIMITS["source_bytes"]:
                block = os.read(source, min(1024**2, LIMITS["source_bytes"]+1-size))
                if not block:
                    break
                chunks.append(block)
                size += len(block)
            raw = b"".join(chunks)
            after = os.fstat(source)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("INPUT_MUTATED")
            if len(raw) != INPUT_BYTES or sha(raw) != INPUT_SHA256:
                raise ValueError("INPUT_PIN_MISMATCH")
            return raw
        finally:
            os.close(source)
    finally:
        os.close(fd)


def freeze(attempt, receipt_path, receipt_pin):
    raw = read(receipt_path)
    if sha(raw) != receipt_pin:
        raise ValueError("INPUT_RECEIPT_PIN")
    validate_input_receipt(json.loads(raw))
    selected_bytes(attempt)
    pins = source_pins()
    from scripts.evaluate_vendor_cad_upload_unfold_v2 import _archive_sources
    archive = _archive_sources(attempt, pins)
    write(attempt / "INPUT_RECEIPT_RELEASED.json", raw)
    frozen = {"profile": PROFILE, "limits": LIMITS, "source_pins": pins, **archive,
              "input_receipt_sha256": receipt_pin, "input_sha256": INPUT_SHA256,
              "input_bytes": INPUT_BYTES, "accuracy": "UNSCORED", "unfolding": "NOT_EXECUTED"}
    write(attempt / "FROZEN.json", frozen)
    if accounting(attempt)["reason"]:
        raise ValueError("FREEZE_STORAGE_BUDGET")
    return frozen


def verify(attempt, pin):
    raw = read(attempt / "FROZEN.json")
    if sha(raw) != pin:
        raise ValueError("FREEZE_PIN")
    frozen = json.loads(raw)
    if (frozen.get("profile") != PROFILE or frozen.get("limits") != LIMITS
            or frozen.get("input_sha256") != INPUT_SHA256 or frozen.get("input_bytes") != INPUT_BYTES
            or frozen.get("source_pins") != source_pins()):
        raise ValueError("FROZEN_SOURCE_OR_PROFILE_CHANGED")
    for name, key in (("SOURCE_MANIFEST.json", "source_manifest_sha256"),
                      ("SOURCE.tar", "source_archive_sha256"),
                      ("INPUT_RECEIPT_RELEASED.json", "input_receipt_sha256")):
        if name == "SOURCE.tar":
            digest, size = hashlib.sha256(), 0
            with open(attempt/name, "rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    size += len(block)
                    if size+8192 > LIMITS["frozen_archive_bytes"]:
                        raise ValueError("FROZEN_ARCHIVE_BUDGET")
                    digest.update(block)
            actual = digest.hexdigest()
        else:
            actual = sha(read(attempt/name, LIMITS["artifact_bytes"]))
        if actual != frozen[key]:
            raise ValueError("FROZEN_ARTIFACT_CHANGED")
    validate_input_receipt(json.loads(read(attempt/"INPUT_RECEIPT_RELEASED.json")))
    return frozen


FROZEN_ARCHIVE_FILES = frozenset(("SOURCE.tar", "SOURCE_MANIFEST.json", "FROZEN.json"))


def storage_reason(output_bytes, archive_bytes, log_bytes):
    if archive_bytes+8192 > LIMITS["frozen_archive_bytes"]:
        return "FROZEN_ARCHIVE_BUDGET"
    if log_bytes > LIMITS["log_bytes"]:
        return "AGGREGATE_LOG_BUDGET"
    if output_bytes+8192 > LIMITS["artifact_bytes"]:
        return "AGGREGATE_ARTIFACT_BUDGET"
    return None


def accounting(attempt):
    # Engineering successor: full pinned runtime archive is not CAD output.
    # Only these exact root-level files receive the separate fixed allowance.
    # V1's prelaunch failure is retained; no started run gets relaxed budgets.
    total = archive = logs = 0
    for path in attempt.rglob("*"):
        if path.is_symlink():
            raise ValueError("ARTIFACT_SYMLINK")
        if path.is_file():
            size = path.stat().st_size
            if path.parent == attempt and path.name in FROZEN_ARCHIVE_FILES:
                archive += size
            else:
                total += size
            if path.name in ("stdout.log", "stderr.log"):
                logs += size
    return {"artifact_bytes": total, "frozen_archive_bytes": archive,
            "log_bytes": logs, "reason": storage_reason(total, archive, logs)}


def extraction_gate(attempt, pin, post_path, post_pin, reference_path, reference_pin):
    raw = read(post_path)
    post = json.loads(raw)
    if sha(raw) != post_pin or post.get("decision") != "GREEN" or post.get("independent_non_author") is not True or post.get("frozen_sha256") != pin or post.get("mode") != "neutral":
        raise ValueError("INDEPENDENT_NEUTRAL_POST_REQUIRED")
    terminal = json.loads(read(attempt/"neutral"/"CASE_TERMINAL.json"))
    if terminal.get("outer_exit") != 0 or terminal.get("reason") or terminal.get("cleanup_known") is not True:
        raise ValueError("NEUTRAL_NOT_SUCCESSFULLY_TERMINAL")
    raw = read(reference_path)
    reference = json.loads(raw)
    if sha(raw) != reference_pin or reference.get("input_sha256") != INPUT_SHA256 or reference.get("evidence_sha256") != sha(read(attempt/"neutral"/"EVIDENCE.json", LIMITS["artifact_bytes"])) or reference.get("independent_non_author") is not True or reference.get("reference_frozen") is not True:
        raise ValueError("INDEPENDENT_REFERENCE_FREEZE_REQUIRED")
    validate_reference_labels(reference)


def validate_reference_labels(reference):
    """Require explicit technical labels; a freeze flag is not reference truth."""
    regions = reference.get("reference_bends")
    complete = reference.get("feature_complete")
    status = reference.get("reference_status")
    limitations = reference.get("limitations")
    if (not isinstance(regions, list) or type(complete) is not bool
            or status != ("complete" if complete else "partial")
            or not isinstance(limitations, list) or not limitations
            or any(not isinstance(x, str) or not x.strip() for x in limitations)):
        raise ValueError("EXPLICIT_REFERENCE_LABELS_AND_COMPLETENESS_REQUIRED")
    census = reference.get("face_ids")
    if (not isinstance(census, list) or not census or any(type(x) is not int or x < 1 for x in census)
            or len(set(census)) != len(census)):
        raise ValueError("REFERENCE_FACE_CENSUS_REQUIRED")
    owned, ids = set(), set()
    for region in regions:
        if not isinstance(region, dict) or not isinstance(region.get("bend_id"), str) or not region["bend_id"] or region["bend_id"] in ids:
            raise ValueError("REFERENCE_REGION_ID")
        ids.add(region["bend_id"])
        faces = []
        for key in ("inner_face_ids", "outer_face_ids"):
            group = region.get(key)
            if not isinstance(group, list) or not group or any(type(x) is not int or x not in census for x in group):
                raise ValueError("REFERENCE_REGION_FACES")
            faces.extend(group)
        if len(set(faces)) != len(faces) or owned.intersection(faces):
            raise ValueError("REFERENCE_REGION_OWNERSHIP")
        owned.update(faces)
    groups = [owned]
    for key in ("unknown_face_ids", "verified_nonbend_face_ids"):
        faces = reference.get(key)
        if not isinstance(faces, list) or any(type(x) is not int or x not in census for x in faces) or len(set(faces)) != len(faces):
            raise ValueError("REFERENCE_PARTITION_REQUIRED")
        groups.append(set(faces))
    if any(groups[i].intersection(groups[j]) for i in range(3) for j in range(i+1, 3)) or set().union(*groups) != set(census):
        raise ValueError("REFERENCE_FULL_PARTITION_REQUIRED")
    if bool(groups[1]) == complete:
        raise ValueError("REFERENCE_UNKNOWN_COMPLETENESS_CONFLICT")


def child(attempt, mode, pin):
    verify(attempt, pin)
    folder = attempt/mode
    launch = json.loads(read(folder/"CASE_LAUNCH.json"))
    nonce = os.environ.get(NONCE_ENV, "")
    if not nonce or sha(nonce.encode()) != launch["nonce_sha256"] or launch["parent_pid"] != os.getppid() or launch["mode"] != mode or launch["frozen_sha256"] != pin:
        raise ValueError("NATIVE_CHILD_UNAUTHORIZED")
    write(folder/"CASE_CLAIM.json", {"pid": os.getpid(), "mode": mode})
    write(folder/"source.step", selected_bytes(attempt))
    if mode == "neutral":
        from scripts.build_cad_bend_reference_evidence_v1 import collect_evidence
        write(folder/"EVIDENCE.json", collect_evidence(folder/"source.step"))
    elif mode == "extraction":
        from scripts.extract_step_topology_bend_axes_v10 import extract_step_bend_axes
        result = extract_step_bend_axes(folder/"source.step")
        result["input_sha256"] = INPUT_SHA256
        write(folder/"EXTRACTION.json", result)
    else:
        raise ValueError("INVALID_MODE")
    write(folder/"RESULT.json", {"status": "COMPLETE", "mode": mode, "accuracy": "UNSCORED"})


def run_job(attempt, mode, pin):
    from scripts.evaluate_vendor_cad_upload_unfold_v2 import _ps_rows, _tree, _kill_seen_groups
    from backend import vendor_unfold_worker_v4 as shared
    folder = attempt/mode
    nonce = secrets.token_hex(32)
    write(folder/"CASE_LAUNCH.json", {"mode": mode, "frozen_sha256": pin,
          "nonce_sha256": sha(nonce.encode()), "parent_pid": os.getpid()})
    env = {**os.environ, NONCE_ENV: nonce, "PYTHONDONTWRITEBYTECODE": "1",
           **{name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
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
            with open(folder/"stdout.log", "xb") as out, open(folder/"stderr.log", "xb") as err:
                process = subprocess.Popen([sys.executable, str(Path(__file__).absolute()), "--native-child", mode,
                    "--attempt", str(attempt), "--frozen-sha256", pin], stdout=out, stderr=err,
                    cwd=ROOT, env=env, start_new_session=True)
                groups.add(process.pid)
                shared._record(lock, {"state": "RUNNING", "pgid": process.pid})
                while process.poll() is None:
                    rows = _ps_rows()
                    groups.update(rows[p][1] for p in _tree(rows, process.pid))
                    rss = sum(row[2] for row in rows.values() if row[1] in groups)
                    peak = max(peak, rss)
                    reason = ("CASE_TIME_BUDGET" if time.monotonic()-started >= LIMITS["case_seconds"]
                              else "CASE_RSS_BUDGET" if rss > LIMITS["rss_kib"] else accounting(attempt)["reason"])
                    if reason:
                        break
                    time.sleep(.05)
        except Exception as exc:
            reason = "OUTER_MONITOR_FAILURE:"+type(exc).__name__
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
            try:
                counts = accounting(attempt)
            except Exception:
                counts = {"reason": "ACCOUNTING_FAILURE"}
            receipt = {"mode": mode, "outer_exit": None if process is None else process.returncode,
                       "reason": reason or counts["reason"], "cleanup_known": cleaned,
                       "elapsed_seconds": time.monotonic()-started, "peak_sampled_rss_kib": peak,
                       "sampled_not_continuous_hwm": True, "accounting": counts}
            write(folder/"CASE_TERMINAL.json", receipt)
    return receipt


def execute(attempt, mode, pin, pre_path, pre_pin, post_path=None, post_pin=None,
            reference_path=None, reference_pin=None):
    if mode not in ("neutral", "extraction"):
        raise ValueError("INVALID_MODE")
    verify(attempt, pin)
    raw = read(pre_path)
    if sha(raw) != pre_pin:
        raise ValueError("PRE_PIN")
    validate_pre(json.loads(raw), pin, mode)
    if mode == "extraction":
        extraction_gate(attempt, pin, post_path, post_pin, reference_path, reference_pin)
    elif (attempt/"extraction").exists():
        raise ValueError("NEUTRAL_MUST_PRECEDE_EXTRACTION")
    if accounting(attempt)["reason"]:
        raise ValueError("PRELAUNCH_AGGREGATE_BUDGET")
    folder = attempt/mode
    folder.mkdir(exist_ok=False)
    write(folder/"EXECUTION.json", {"mode": mode, "frozen_sha256": pin, "pre_sha256": pre_pin,
          "neutral_post_sha256": post_pin, "reference_sha256": reference_pin, "pid": os.getpid()})
    try:
        receipt = run_job(attempt, mode, pin)
    except Exception as exc:
        receipt = {"mode": mode, "outer_exit": None, "cleanup_known": False,
                   "reason": "ADMISSION_OR_SUPERVISION_FAILURE:"+type(exc).__name__}
        if not (folder/"CASE_TERMINAL.json").exists():
            write(folder/"CASE_TERMINAL.json", receipt)
    return 0 if receipt.get("outer_exit") == 0 and not receipt.get("reason") and receipt.get("cleanup_known") else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("--input-receipt-sha256")
    parser.add_argument("--mode", choices=("neutral", "extraction"))
    parser.add_argument("--native-child", choices=("neutral", "extraction"))
    parser.add_argument("--frozen-sha256")
    for name in ("pre", "neutral-post", "reference"):
        parser.add_argument("--"+name, type=Path)
        parser.add_argument("--"+name+"-sha256")
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.freeze:
        freeze(args.attempt, args.input_receipt, args.input_receipt_sha256)
        return 0
    if args.native_child:
        child(args.attempt, args.native_child, args.frozen_sha256)
        return 0
    return execute(args.attempt, args.mode, args.frozen_sha256, args.pre, args.pre_sha256,
                   args.neutral_post, args.neutral_post_sha256, args.reference, args.reference_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
