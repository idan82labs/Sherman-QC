"""Fixed, read-only consumed Gold11 admission review; never a benchmark.

Only --invented-check or --review-approved is accepted. The latter requires the
independent source/protocol gate documented in the protocol; a CLI flag is not
authority. The fixed attempt directory is exclusive and never retried.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import runpy
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(r'''[{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp","cad_sha256":"ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37","case_token":"CG08506AE97D","part_key":"10839000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_2.ply","scan_sha256":"cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b","reference_id":"corpus/03_10839000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.stp","cad_sha256":"d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5","case_token":"CG191DEC7F81","part_key":"49001000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.ply","scan_sha256":"7a1773eca9fab616afef498a9f9831e67eacb17fcc0d170607490a9e9db37020","reference_id":"corpus/10_49001000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01.stp","cad_sha256":"4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f","case_token":"CG1E0BBA2933","part_key":"38172000","scan_path":"/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_X.ply","scan_sha256":"4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9","reference_id":"shaked_xv/C05"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.stp","cad_sha256":"81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e","case_token":"CG4BDD1371B6","part_key":"37361005","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.ply","scan_sha256":"d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038","reference_id":"corpus/05_37361005"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.stp","cad_sha256":"3cdfcba8fbea3735e3c9b3610b69ccab2bc8bf681919ef2c86839693ef54479f","case_token":"CG4FAF273093","part_key":"49024000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply","scan_sha256":"70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade","reference_id":"corpus/11_49024000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0 1.stp","cad_sha256":"f9b8965c0a172756d6cf69110f3336b9ecaa0dcb14fe3848b9d3187a885d7f8f","case_token":"CG64C412CC73","part_key":"40500000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0_SCAN2 1.ply","scan_sha256":"152f1b1aacb20920a601f7a8197d4c89a111688325bcf557a6a731641f306614","reference_id":"corpus/08_40500000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01.stp","cad_sha256":"4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f","case_token":"CG65CD575B1C","part_key":"38172000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply","scan_sha256":"07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880","reference_id":"corpus/06_38172000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp","cad_sha256":"ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37","case_token":"CG84C8FA12D2","part_key":"10839000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01_full_scan.ply","scan_sha256":"71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215","reference_id":"corpus/01_10839000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp","cad_sha256":"ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37","case_token":"CGE4B0549E41","part_key":"10839000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_1.ply","scan_sha256":"08686dc1d5d4c89725393907082e8cc4e4c0f5a21427697760ff44d29c7939fc","reference_id":"corpus/02_10839000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01.stp","cad_sha256":"4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f","case_token":"CGEBA9A869AC","part_key":"38172000","scan_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_2 1.ply","scan_sha256":"f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63","reference_id":"corpus/07_38172000"},{"cad_path":"/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01.stp","cad_sha256":"4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f","case_token":"CGF41F49066A","part_key":"38172000","scan_path":"/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply","scan_sha256":"2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba","reference_id":"shaked_xv/C06"}]''')
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
PROTOCOL = "docs/research/2026-09-09_gold11_input_review_protocol.md"
SOURCES = ("scripts/review_gold11_inputs_v1.py", "scripts/gold11_admission_geometry.py",
           "scripts/gold11_cad_admission.py", PROTOCOL)
ATTEMPT = ROOT / "docs/research/gold11_input_review_v1_attempt"
WALL = 300
CPU = 240
RSS = 2 * 1024**3
OUTPUT = 1024**2


def bindings():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES}


def bounded_read(path, cap, expected=None):
    """No final symlink; size/hash and descriptor consistency bind used bytes."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= cap:
            raise ValueError("NONREGULAR_OR_BYTE_BUDGET")
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(1024**2, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > cap:
                raise ValueError("BYTE_BUDGET")
        after = os.fstat(fd)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in keys):
            raise ValueError("SOURCE_CHANGED_DURING_READ")
        blob = b"".join(chunks)
        digest = hashlib.sha256(blob).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("DECLARED_HASH_MISMATCH")
        return blob, {"sha256": digest, "bytes": len(blob),
                      "snapshot_consistent": True}
    finally:
        os.close(fd)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("NONFINITE_JSON")


DECLARATION_KEYS = ("reference_frame", "coordinate_frame", "frame", "unit", "units",
                    "line_convention", "carrier_convention", "endpoint_convention",
                    "unit_status", "units_status", "frame_status")


def declarations(record):
    """Preserve bounded explicit metadata, without supplying missing semantics."""
    result = {}
    for key in DECLARATION_KEYS:
        if key in record:
            value = record[key]
            encoded = json.dumps(value, sort_keys=True, allow_nan=False)
            result[key] = (value if len(encoded) <= 1024 else
                           {"status": "DECLARATION_VALUE_BUDGET", "encoded_characters": len(encoded)})
    return result


def declaration_differences(case_declarations, line_declarations):
    return [{"key": key, "case_value": case_declarations[key], "line_value": value,
             "status": "EXPLICIT_DIFFERENCE_REQUIRES_REVIEW_NO_ALIAS_ASSUMED"}
            for key, value in line_declarations.items()
            if key in case_declarations and value != case_declarations[key]]


def load_selected_references():
    blob, binding = bounded_read(CATALOG, 8 * 1024**2)
    payload = json.loads(blob, object_pairs_hook=unique_object,
                         parse_constant=reject_constant)
    by_sha = {case["scan_sha256"]: case for case in CASES}
    selected = {}
    # The container is necessary to select approved rows. Never inspect or emit
    # unselected geometry, nor follow its source labels or other linked paths.
    for row in payload["cases"]:
        case = by_sha.get(row.get("scan_sha256"))
        if case is None:
            continue
        key = case["case_token"]
        if key in selected or row.get("case_id") != case["reference_id"]:
            raise ValueError("REFERENCE_IDENTITY_OR_DUPLICATE")
        if row.get("scan_path") != case["scan_path"] or str(row.get("part_id")) != case["part_key"]:
            raise ValueError("REFERENCE_SOURCE_IDENTITY_MISMATCH")
        if not isinstance(row.get("bends"), list) or len(row["bends"]) > 64:
            raise ValueError("REFERENCE_SCHEMA_OR_BUDGET")
        selected[key] = row
    if set(selected) != {case["case_token"] for case in CASES}:
        raise ValueError("REFERENCE_CASE_DENOMINATOR")
    if sum(len(row["bends"]) for row in selected.values()) != 71:
        raise ValueError("REFERENCE_LINE_DENOMINATOR")
    binding["selected_geometry_record_sha256"] = hashlib.sha256(json.dumps(
        selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    binding["historical_reference_byte_equality"] = "UNESTABLISHED"
    return selected, binding


def stats(points):
    import numpy as np
    center = points.mean(axis=0)
    delta = points - center
    values = np.linalg.eigvalsh(delta.T @ delta / len(points))
    return {"aabb_min": points.min(axis=0).tolist(), "aabb_max": points.max(axis=0).tolist(),
            "centroid": center.tolist(),
            "principal_rms_lengths": np.sqrt(np.maximum(values[::-1], 0)).tolist(),
            "unit": "RAW_SOURCE_COORDINATE_UNIT_NOT_CALIBRATED"}


def review_reference_rows(reference, parse_reference):
    """Review every selected line even when its raw PLY is unavailable."""
    import numpy as np
    rows = []
    case_declarations = declarations(reference)
    for index, bend in enumerate(reference["bends"]):
        item = {"reference_index": index, "bend_id": bend.get("bend_id"),
                "line_declarations": declarations(bend)}
        item["case_line_declaration_differences"] = declaration_differences(
            case_declarations, item["line_declarations"])
        try:
            normalized = parse_reference(bend)
            endpoints = normalized.pop("endpoints")
            item.update(representation=normalized, unchanged_endpoints=endpoints.tolist(),
                        span_raw_units=float(np.linalg.norm(endpoints[1] - endpoints[0])),
                        status="DESCRIPTIVE_ONLY_CARRIER_UNRESOLVED")
        except (ValueError, KeyError, TypeError) as exc:
            item.update(status="REFERENCE_REPRESENTATION_UNAVAILABLE",
                        error_type=type(exc).__name__, reason=str(exc)[:160])
        rows.append(item)
    return rows


def review_case(case, reference, parse_ply, parse_reference):
    import numpy as np
    from scipy.spatial import cKDTree
    row = {"case_id": case["case_token"], "family_id": case["part_key"],
           "raw_scan_to_cad_transform": None, "datum_status": "DATUM_CONTRACT_ABSENT",
           "scan_unit_status": "UNVERIFIED_EXPORT_UNIT",
           "case_declarations": declarations(reference),
           "reference_count": len(reference["bends"]),
           "references": review_reference_rows(reference, parse_reference)}
    try:
        blob, binding = bounded_read(case["scan_path"], 64 * 1024**2, case["scan_sha256"])
        row["scan_binding"] = binding
        parsed = parse_ply(blob)
        points = parsed.pop("points")
        row["ply"] = parsed
        del blob
        if points is None:
            row["status"] = "READER_OR_BUDGET_UNSUPPORTED"
            return row
        row["raw_point_count"] = len(points)
        row["scan_geometry"] = stats(points)
        row["r0_row_cap_compatible"] = len(points) <= 200000
        row["r0_vertex_only_schema_compatible"] = (
            len(parsed["header"]["elements"]) == 1)
        row["r0_header_and_row_limits_compatible"] = (
            row["r0_row_cap_compatible"] and row["r0_vertex_only_schema_compatible"])
        row["r0_actual_loader_executed"] = False
        tree = cKDTree(points)
        for item in row["references"]:
            if "unchanged_endpoints" not in item:
                continue
            endpoints = np.asarray(item["unchanged_endpoints"])
            line = endpoints[0] + np.linspace(0., 1., 201)[:, None] * (endpoints[1] - endpoints[0])
            distances, _ = tree.query(line, k=1, workers=1)
            item["nearest_raw_vertex_distance"] = {
                "min": float(distances.min()), "median": float(np.median(distances)),
                "p95": float(np.quantile(distances, .95)), "max": float(distances.max()),
                "endpoint_distances": [float(distances[0]), float(distances[-1])],
                "sample_count": 201, "threshold_or_score_applied": False}
        row["status"] = "REVIEWED_NOT_REGISTERED"
    except Exception as exc:
        row["status"] = "UNAVAILABLE"
        row["error_type"] = type(exc).__name__
        row["reason"] = str(exc)[:160]
    return row


def install_guard(cache, mode, site):
    root = str(ROOT)
    cache_roots = (os.path.abspath(cache), str(Path(cache).resolve()))
    allowed = {str(ROOT / name) for name in SOURCES}
    if mode == "review-approved":
        allowed.add(str(CATALOG))
        allowed.update(case["scan_path"] for case in CASES)
        allowed.update(case["cad_path"] for case in CASES)
    def in_cache(path):
        return any(path == item or path.startswith(item + os.sep) for item in cache_roots)
    def audit(event, args):
        if event.startswith("socket.") or event in (
                "subprocess.Popen", "os.system", "os.fork", "os.forkpty",
                "os.posix_spawn", "os.exec", "os.setpgid", "os.setsid"):
            raise RuntimeError("WORKER_PROCESS_OR_NETWORK_FORBIDDEN")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0]))
            mode_string, flags = args[1:3]
            writing = (isinstance(mode_string, str) and any(x in mode_string for x in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not in_cache(path) and path != "/dev/null":
                raise RuntimeError("WORKER_WRITE_FORBIDDEN")
            workspace = str(ROOT.parent)
            if path.startswith(workspace + os.sep) and not (
                    path.startswith(site + os.sep) or path in allowed or in_cache(path)):
                raise RuntimeError("UNNAMED_WORKSPACE_READ_FORBIDDEN")
            if not in_cache(path) and any(x in (".env", ".aws", ".ssh", ".codex") for x in Path(path).parts):
                raise RuntimeError("PRIVATE_CONFIGURATION_FORBIDDEN")
        if event in ("os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0]))
            if path.startswith(str(ROOT.parent)) and not path.startswith(site):
                raise RuntimeError("WORKSPACE_ENUMERATION_FORBIDDEN")
    sys.addaudithook(audit)


def invented_check(cache, parse_ply, parse_reference, inspect_step):
    import numpy as np
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    points = parse_ply(b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 2 3\n")["points"]
    assert points.shape == (2, 3) and np.array_equal(points[1], [1, 2, 3])
    ref = parse_reference({"line_start": [0, 0, 0], "line_end": [1, 0, 0]})
    assert np.array_equal(ref["endpoints"], [[0, 0, 0], [1, 0, 0]])
    snapshot = Path(cache) / "invented_box.step"
    writer = STEPControl_Writer()
    assert writer.Transfer(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(snapshot)) == IFSelect_RetDone
    result = inspect_step(snapshot)
    print(json.dumps({"invented_native_inventory": result}, sort_keys=True), flush=True)
    assert result["status"] == "AVAILABLE_REVIEW_INVENTORY"
    assert result["topology_occurrence_counts"]["FACE"] == 6
    assert np.allclose(np.array(result["bounding_box"]["maximum"]) - result["bounding_box"]["minimum"],
                       [10, 20, 30], atol=1e-6, rtol=0)
    return {"status": "PASS", "physical_inputs_opened": False}


def worker(cache, mode):
    resource.setrlimit(resource.RLIMIT_CPU, (CPU, CPU))
    sys.dont_write_bytecode = True
    site = str(Path(sys.executable).absolute().parent.parent / "lib/python3.11/site-packages")
    sys.path.insert(0, site)
    install_guard(cache, mode, site)
    helper = runpy.run_path(str(ROOT / "scripts/gold11_admission_geometry.py"))
    native = runpy.run_path(str(ROOT / "scripts/gold11_cad_admission.py"))
    start = time.monotonic()
    if mode == "invented-check":
        result = invented_check(cache, helper["parse_diagnostic_ply"],
                                helper["parse_reference_endpoints"], native["inspect_step_snapshot"])
    else:
        references, reference_binding = load_selected_references()
        print(json.dumps({"reference_binding": reference_binding}, sort_keys=True), flush=True)
        cad_results = {}
        for case in CASES:
            key = case["cad_sha256"]
            if key in cad_results:
                continue
            item = {"family_id": case["part_key"], "cad_sha256": key}
            try:
                blob, binding = bounded_read(case["cad_path"], 8 * 1024**2, key)
                snapshot = Path(cache) / (key + ".step")
                with snapshot.open("xb") as handle:
                    handle.write(blob)
                del blob
                item["binding"] = binding
                item["native"] = native["inspect_step_snapshot"](snapshot)
                item["status"] = item["native"]["status"]
            except Exception as exc:
                item.update(status="UNAVAILABLE", error_type=type(exc).__name__, reason=str(exc)[:160])
            cad_results[key] = item
            print(json.dumps({"cad_review": item}, sort_keys=True, allow_nan=False), flush=True)
        rows = []
        for case in CASES:
            row = review_case(case, references[case["case_token"]],
                              helper["parse_diagnostic_ply"], helper["parse_reference_endpoints"])
            row["cad_review_status"] = cad_results[case["cad_sha256"]]["status"]
            rows.append(row)
            print(json.dumps({"case_review": row}, sort_keys=True, allow_nan=False), flush=True)
        # Re-read the small container only: bind absence of reference edits during
        # this review. Do not double-read raw data or claim historical equality.
        _, final_binding = bounded_read(CATALOG, 8 * 1024**2, reference_binding["sha256"])
        result = {"status": "REVIEW_COMPLETED_NOT_BENCHMARKED",
                  "requested_cases": len(CASES), "reported_cases": len(rows),
                  "selected_reference_lines": sum(row["reference_count"] for row in rows),
                  "cad_files": len(cad_results), "reference_container_unchanged": True,
                  "engine_executed": False, "registration_fitted": False,
                  "score_computed": False, "production_use_allowed": False}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps({"worker_result": result, "worker_wall_seconds": time.monotonic() - start,
                      "worker_cpu_seconds": usage.ru_utime + usage.ru_stime,
                      "worker_peak_rss_bytes": usage.ru_maxrss}, sort_keys=True), flush=True)


def parent(mode):
    before = bindings()
    start = time.monotonic()
    if mode == "review-approved":
        # A prior started/failed/completed attempt is never read or overwritten.
        ATTEMPT.mkdir(mode=0o700)
        with (ATTEMPT / "STARTED.json").open("x") as handle:
            json.dump({"status": "STARTED_TERMINAL_NO_RETRY", "source_sha256": before,
                       "mode": mode, "budgets": {"wall": WALL, "cpu": CPU, "rss": RSS, "output": OUTPUT}}, handle, sort_keys=True)
    output, total, peak, stop = [], 0, 0, None
    with tempfile.TemporaryDirectory(prefix="sherman-gold11-review-") as cache:
        env = {key: "1" for key in ("PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        env.update(HOME=cache, TMPDIR=cache, XDG_CACHE_HOME=cache, MPLCONFIGDIR=cache,
                   PATH="/usr/bin:/bin", LANG="C.UTF-8")
        child = subprocess.Popen([sys.executable, "-I", "-S", str(Path(__file__).resolve()),
                                  "--worker", cache, mode], cwd=cache, env=env,
                                 start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                for key, _ in selector.select(.1):
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = max(0, OUTPUT - total)
                    output.append(block[:remaining])
                    sys.stdout.buffer.write(block[:remaining])
                    sys.stdout.flush()
                    total += len(block)
                    if total > OUTPUT:
                        stop = "OUTPUT_BUDGET"
                        break
                if not stop:
                    try:
                        sample = subprocess.run(["/bin/ps", "-o", "rss=", "-g", str(child.pid)],
                                                capture_output=True, text=True, timeout=2)
                        if sample.returncode == 0:
                            peak = max(peak, sum(int(value) * 1024 for value in sample.stdout.split()))
                        elif child.poll() is None:
                            stop = "RSS_MONITOR_UNAVAILABLE"
                    except Exception:
                        stop = "RSS_MONITOR_UNAVAILABLE"
                if time.monotonic() - start > WALL:
                    stop = "WALL_BUDGET"
                if peak > RSS:
                    stop = "RSS_BUDGET"
                if stop:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
            child.wait(timeout=5)
        finally:
            selector.close()
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    after = bindings()
    if after != before:
        stop = "SOURCE_CHANGED_DURING_EXECUTION"
    receipt = {"parent_wall_seconds": time.monotonic() - start, "exit_code": child.returncode,
               "stop_reason": stop, "sampled_process_group_peak_rss_bytes": peak,
               "output_bytes_seen": total, "source_sha256": before,
               "source_binding_unchanged": before == after, "mode": mode}
    if mode == "review-approved":
        reported = {}
        for line in b"".join(output).splitlines():
            try:
                payload = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if isinstance(payload, dict) and "case_review" in payload:
                row = payload["case_review"]
                reported[row["case_id"]] = row["status"]
        receipt["all_case_dispositions"] = [{
            "case_id": case["case_token"],
            "status": reported.get(case["case_token"], "UNAVAILABLE_REVIEW_INTERRUPTED_OR_REFERENCE_GATE"),
        } for case in CASES]
    if mode == "review-approved":
        with (ATTEMPT / "OUTPUT.txt").open("xb") as handle:
            handle.write(b"".join(output))
        with (ATTEMPT / "COMPLETED.json").open("x") as handle:
            json.dump(receipt, handle, sort_keys=True)
    print(json.dumps({"parent_receipt": receipt}, sort_keys=True), flush=True)
    return int(stop is not None or child.returncode != 0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker" and sys.argv[3] in ("invented-check", "review-approved"):
        worker(sys.argv[2], sys.argv[3])
    elif sys.argv[1:] in (["--invented-check"], ["--review-approved"]):
        raise SystemExit(parent(sys.argv[1][2:]))
    else:
        raise SystemExit("Only fixed --invented-check or independently gated --review-approved")
