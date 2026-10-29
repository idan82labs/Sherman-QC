"""Explicit offline, frozen, bounded STEP -> compensated draft -> analytic DXF.

Not imported by the legacy server. Every job requires an exact input/profile
freeze and an exclusive terminal record. All outputs are drafts, never cutting
release. No corpus lookup, scan, Gold, DB, GUI or preferences are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

from backend.vendor_manufacturing_profile import CompensationProfile, manufacturing_release_status
from backend.vendor_manufacturing_input import MAX_STEP_BYTES
from scripts import run_vendor_manufacturing_synthetic as safety

ROOT = safety.ROOT
SCHEMA = "SHERMAN_MANUFACTURING_FILE_FREEZE_V1"
NEW_SOURCES = ("backend/vendor_manufacturing_job.py", "backend/vendor_manufacturing_bridge.py",
    "backend/vendor_manufacturing_serialization.py", "backend/vendor_manufacturing_dxf.py",
    "scripts/extract_step_topology_bend_axes.py", ".venv/bin/python", ".venv/pyvenv.cfg")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_pins():
    pins = safety._sources()
    paths = set(NEW_SOURCES)
    packages = ROOT / ".venv/lib/python3.11/site-packages"
    for name in ("OCP", "numpy"):
        folder = packages / name
        found = {str(path.relative_to(ROOT)) for path in folder.rglob("*")
                 if path.is_file() and path.suffix in {".py", ".so", ".dylib"}}
        if not found:
            raise ValueError("FILE_WORKER_DEPENDENCY_MISSING:" + name)
        paths.update(found)
    if len(paths) > 2000:
        raise ValueError("FILE_WORKER_SOURCE_BUDGET_EXCEEDED")
    pins.update({name: safety._hash(ROOT / name) for name in sorted(paths)})
    return pins


def _inputs(raw, expected_sha256, profile, root_face_index, candidates):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_STEP_BYTES:
        raise ValueError("STEP_BYTE_BUDGET_EXCEEDED")
    if type(expected_sha256) is not str or _digest(raw) != expected_sha256:
        raise ValueError("CAD_BYTES_HASH_MISMATCH")
    validated = CompensationProfile.from_dict(profile)
    if root_face_index is not None and (type(root_face_index) is not int or not 0 <= root_face_index < 1024):
        raise ValueError("INVALID_EXPLICIT_ROOT_FACE")
    if candidates is not None:
        from backend.vendor_manufacturing_bridge import bind_vendor_bend_ids
        # Pure admission only. Native correspondence is performed inside the job.
        fields = {"feature_id", "line_start", "line_end", "source_sha256"}
        if (type(candidates) is not list or not 0 < len(candidates) <= 32
                or not all(type(row) is dict and set(row) == fields for row in candidates)):
            raise ValueError("INVALID_VENDOR_CANDIDATES")
        markers = [{**row, "pair_id": row.get("feature_id")} for row in candidates]
        bind_vendor_bend_ids(markers, candidates, expected_sha256)
    candidate_bytes = json.dumps(candidates, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(candidate_bytes) > 100_000:
        raise ValueError("CANDIDATE_BYTE_BUDGET_EXCEEDED")
    return {"source_sha256": expected_sha256, "source_bytes": len(raw),
            "profile_sha256": validated.sha256, "root_face_index": root_face_index,
            "root_rule": "EXPLICIT_NATIVE_FACE" if root_face_index is not None else "CAD_ONLY_LARGEST_PLANAR_FACE",
            "vendor_candidates_sha256": _digest(candidate_bytes)}, validated


def freeze_file_job(raw, expected_sha256, profile, *, root_face_index=None, candidates=None):
    inputs, _ = _inputs(raw, expected_sha256, profile, root_face_index, candidates)
    return {"schema": SCHEMA, "limits": safety.LIMITS, "runtime_identity": safety.RUNTIME_IDENTITY,
            "source_sha256": source_pins(), "inputs": inputs}


def _read_bounded(path, maximum):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("FILE_BYTE_BUDGET_EXCEEDED")
    return raw


def _environment(folder, *, native=False):
    paths = (safety.RESOURCES / "lib", safety.RESOURCES / "Mod/Draft",
             ROOT / ".tools/freecad-1.1.3/python-deps", safety.UPSTREAM, ROOT) if native else (ROOT,)
    return {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(folder), "TMPDIR": str(folder),
            "FREECAD_USER_HOME": str(folder), "PYTHONPATH": os.pathsep.join(map(str, paths)),
            "PYTHONPYCACHEPREFIX": str(folder / "pycache"),
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def _validate_result(result, inputs):
    if (type(result) is not dict or result.get("source_sha256") != inputs["source_sha256"]
            or result.get("profile_sha256") != inputs["profile_sha256"]
            or result.get("manufacturing_ready") is not False
            or result.get("cut_release_allowed") is not False
            or result.get("status") not in {"DRAFT_COMPUTED", "UNAVAILABLE"}):
        raise ValueError("INVALID_DRAFT_RESULT_ENVELOPE")
    if result["status"] == "UNAVAILABLE":
        allowed = {"schema", "status", "source_sha256", "profile_sha256", "manufacturing_ready",
                   "cut_release_allowed", "draft", "reason"}
        if (set(result) - allowed or result.get("draft") is not None
                or type(result.get("reason")) is not str or not 0 < len(result["reason"]) <= 1000):
            raise ValueError("INVALID_UNAVAILABLE_RESULT")
        return
    draft = result.get("draft")
    nested = ("source_unit_proof", "bend_id_bridge", "dxf_roundtrip", "cut_dxf", "bend_guide_dxf", "release_gates")
    if type(draft) is not dict or not all(type(result.get(key)) is dict for key in nested):
        raise ValueError("INCOMPLETE_DRAFT_WORKFLOW_RESULT")
    units, bridge = result["source_unit_proof"], result["bend_id_bridge"]
    if (draft.get("source_sha256") != inputs["source_sha256"]
            or draft.get("profile_sha256") != inputs["profile_sha256"]
            or draft.get("manufacturing_ready") is not False or draft.get("release_allowed") is not False
            or draft.get("units_verified") is not True or draft.get("units") != "mm"
            or draft.get("source_units_are_not_physical_calibration") is not True
            or draft.get("root_selection_is_not_shop_datum_or_bend_order") is not True
            or units.get("source_sha256") != inputs["source_sha256"]
            or units.get("units") != "mm" or units.get("source_mm_per_coordinate") != 1.0
            or units.get("units_verified_from_native_step") is not True
            or units.get("all_shape_representation_length_units_verified") is not True
            or units.get("physical_calibration_verified") is not False
            or bridge.get("source_sha256") != inputs["source_sha256"]
            or bridge.get("all_native_and_vendor_bends_bijective") is not True
            or bridge.get("manufacturing_ready") is not False
            or bridge.get("nominal_cad_identity_only") is not True
            or result["dxf_roundtrip"].get("passed") is not True
            or result["dxf_roundtrip"].get("independent_importer_evidence") is not False
            or result["dxf_roundtrip"].get("physical_dimensions_verified") is not False
            or result["dxf_roundtrip"].get("cut_file_has_no_bend_entities") is not True
            or result["release_gates"].get("manufacturing_ready") is not False
            or result["release_gates"].get("cut_release_allowed") is not False
            or result["release_gates"].get("profile_sha256") != inputs["profile_sha256"]):
        raise ValueError("INCOMPLETE_DRAFT_WORKFLOW_RESULT")
    from backend.vendor_manufacturing_serialization import REQUIRED_PROOFS, COVERAGE_GROUPS
    proof = draft.get("coverage_proof")
    if (type(proof) is not dict or not all(proof.get(key) is True for key in REQUIRED_PROOFS)
            or type(proof.get("native_face_count")) is not int or not 0 < proof["native_face_count"] <= 1024):
        raise ValueError("INCOMPLETE_NATIVE_COVERAGE_RECEIPT")
    indices = []
    for key in COVERAGE_GROUPS:
        rows = proof.get(key)
        if type(rows) is not list or not all(type(index) is int for index in rows):
            raise ValueError("INVALID_NATIVE_COVERAGE_RECEIPT")
        indices.extend(rows)
    if len(indices) != proof["native_face_count"] or set(indices) != set(range(proof["native_face_count"])):
        raise ValueError("INCOMPLETE_NATIVE_COVERAGE_RECEIPT")
    keys = ("bend_rows", "bend_row_provenance", "folded_bend_markers")
    if (not all(type(draft.get(key)) is list for key in keys)
            or not 0 < len(draft["bend_rows"]) <= 32
            or not all(len(draft[key]) == len(draft["bend_rows"]) for key in keys)
            or proof.get("native_bend_count") != len(draft["bend_rows"])):
        raise ValueError("INCOMPLETE_BEND_RECEIPT")
    assignments = bridge.get("assignments")
    if type(assignments) is not list or len(assignments) != len(draft["bend_rows"]):
        raise ValueError("INCOMPLETE_BRIDGE_RECEIPT")
    pairs, features = set(), set()
    for index, assignment in enumerate(assignments):
        if type(assignment) is not dict:
            raise ValueError("INVALID_BRIDGE_RECEIPT")
        pair, feature, error = assignment.get("pair_id"), assignment.get("feature_id"), assignment.get("endpoint_error_mm")
        if (type(pair) is not str or type(feature) is not str or pair in pairs or feature in features
                or type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1e-7
                or not all(type(draft[key][index]) is dict and draft[key][index].get("pair_id") == pair
                           and draft[key][index].get("feature_id") == feature
                           for key in ("bend_row_provenance", "folded_bend_markers"))):
            raise ValueError("INVALID_BRIDGE_RECEIPT")
        pairs.add(pair); features.add(feature)
    from backend.vendor_manufacturing_dxf import build_analytic_dxf, read_analytic_dxf
    for key, rows in (("cut_dxf", draft.get("cut_rows")),
                      ("bend_guide_dxf", draft.get("cut_rows", []) + draft["bend_rows"])):
        dxf = result[key]
        if (dxf.get("source_sha256") != inputs["source_sha256"] or dxf.get("profile_sha256") != inputs["profile_sha256"]
                or dxf.get("units") != "mm" or dxf.get("units_verified") is not True
                or dxf.get("roundtrip_checked") is not True or dxf.get("release_allowed") is not False
                or dxf.get("manufacturing_ready") is not False or type(dxf.get("text")) is not str):
            raise ValueError("INCOMPLETE_DXF_RECEIPT")
        decoded = read_analytic_dxf(dxf["text"])
        if key == "cut_dxf" and any(row["layer"] not in {"CUT_OUTER", "CUT_INNER"} for row in decoded["rows"]):
            raise ValueError("BEND_ENTITIES_IN_CUT_FILE")
        if build_analytic_dxf(rows, units="mm")["text"] != dxf["text"]:
            raise ValueError("DXF_NATIVE_ROWS_RECEIPT_MISMATCH")


def _supervise_command(command, folder, started, ledger, env):
    """Internal shared cap enforcement; caller owns lock, freeze and record."""
    process = None
    try:
        with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True)
            while process.poll() is None:
                if time.monotonic() - started > safety.MAX_SECONDS:
                    raise RuntimeError("WALL_BUDGET_EXCEEDED")
                rows = safety._group_rows(process.pid)
                if not rows and process.poll() is None:
                    raise RuntimeError("RSS_MONITOR_UNAVAILABLE")
                ledger["peak_tree_rss_kib"] = max(ledger["peak_tree_rss_kib"], sum(rss for _, rss in rows))
                if ledger["peak_tree_rss_kib"] > safety.MAX_TREE_RSS_KIB:
                    raise RuntimeError("RSS_BUDGET_EXCEEDED")
                if stdout.tell() + stderr.tell() > safety.MAX_LOG_BYTES:
                    raise RuntimeError("LOG_BUDGET_EXCEEDED")
                time.sleep(.1)
            if any(pid != process.pid for pid, _ in safety._group_rows(process.pid)):
                raise RuntimeError("UNEXPECTED_LIVE_DESCENDANTS")
    finally:
        if process is not None:
            safety._stop_group(process)
            ledger["returncode"] = process.returncode
        sizes, remaining = {}, safety.MAX_LOG_BYTES
        for name in ("stdout", "stderr"):
            path = folder / f"{name}.log"
            sizes[name] = path.stat().st_size if path.exists() else 0
            if path.exists():
                with path.open("rb") as stream:
                    output = stream.read(remaining)
                remaining -= len(output)
                ledger[name] = output.decode("utf-8", errors="replace")
        ledger["observed_log_bytes"] = sizes
        if sum(sizes.values()) > safety.MAX_LOG_BYTES:
            raise RuntimeError("LOG_BUDGET_EXCEEDED")


def run_file_job(raw, expected_sha256, profile, frozen, record_path, *, root_face_index=None, candidates=None):
    """Run one frozen job once; return durable draft/refusal/failure ledger."""
    if (type(frozen) is not dict or set(frozen) != {"schema", "limits", "runtime_identity", "source_sha256", "inputs"}
            or frozen.get("schema") != SCHEMA or frozen.get("limits") != safety.LIMITS
            or frozen.get("runtime_identity") != safety.RUNTIME_IDENTITY
            or frozen.get("source_sha256") != source_pins()):
        raise ValueError("FILE_JOB_FREEZE_MISMATCH")
    inputs, validated = _inputs(raw, expected_sha256, profile, root_face_index, candidates)
    if frozen.get("inputs") != inputs:
        raise ValueError("FILE_JOB_INPUT_FREEZE_MISMATCH")
    request = {"inputs": inputs, "profile": {"schema": "SHERMAN_COMPENSATION_PROFILE_V1", **asdict(validated)},
               "candidates": candidates}
    request_bytes = json.dumps(request, allow_nan=False).encode()
    if len(request_bytes) > 200_000:
        raise ValueError("REQUEST_BYTE_BUDGET_EXCEEDED")
    with open("/tmp/sherman-manufacturing-unfold.lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Path(record_path).open("x", encoding="utf-8") as record:
            ledger = {"schema": "SHERMAN_MANUFACTURING_FILE_JOB_V1", "status": "RUNNING", "freeze": frozen,
                      "started_at": datetime.now(timezone.utc).isoformat(), "peak_tree_rss_kib": 0,
                      "manufacturing_ready": False, "cut_release_allowed": False, "returncode": None}
            safety._save(record, ledger)
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix="sherman-manufacturing-file-") as temp:
                    folder = Path(temp)
                    (folder / "source.step").write_bytes(raw)
                    (folder / "request.json").write_bytes(request_bytes)
                    _supervise_command([str(ROOT / ".venv/bin/python"), "-m", "backend.vendor_manufacturing_job",
                        "--worker", str(folder)], folder, started, ledger, _environment(folder))
                    result = None
                    if (folder / "result.json").is_file():
                        result = safety._load_json(_read_bounded(folder / "result.json", safety.MAX_RESULT_BYTES))
                        ledger["result"] = result
                    if ledger["returncode"] != 0:
                        reason = result.get("reason") if type(result) is dict else None
                        raise RuntimeError("FILE_WORKER_FAILED:" + (reason or str(ledger["returncode"])))
                    if result is None:
                        raise RuntimeError("FILE_WORKER_RESULT_MISSING")
                    _validate_result(result, inputs)
                    if source_pins() != frozen["source_sha256"]:
                        raise RuntimeError("SOURCE_CHANGED_DURING_FILE_JOB")
                    ledger["status"] = "COMPLETE_DRAFT" if result["status"] == "DRAFT_COMPUTED" else "UNAVAILABLE"
            except BaseException as error:
                ledger.update(status="FAILED", error_type=type(error).__name__, reason=str(error)[:1000])
            finally:
                ledger.update(elapsed_seconds=time.monotonic() - started, finished_at=datetime.now(timezone.utc).isoformat())
                if ledger["elapsed_seconds"] > safety.MAX_SECONDS:
                    ledger.update(status="FAILED", reason="WALL_BUDGET_EXCEEDED")
                safety._save(record, ledger)
            return ledger


def _native(folder, request):
    from backend.vendor_manufacturing_freecad import compensated_unfold
    from backend.vendor_manufacturing_serialization import serialize_compensated_draft
    # The embedded application must exist before Part's native module is loaded.
    # The lazy adapter initializes it later, which is too late for this import.
    import FreeCAD as App
    version = App.Version()
    if version[:3] != ["1", "1", "3"] or len(version) < 8 or version[7] != safety.RUNTIME_IDENTITY["freecad_commit"]:
        raise RuntimeError("PINNED_NATIVE_APPLICATION_NOT_INITIALIZED")
    import Part
    raw = _read_bounded(folder / "source.step", MAX_STEP_BYTES)
    profile = CompensationProfile.from_dict(request["profile"])
    if _digest(raw) != request["inputs"]["source_sha256"] or profile.sha256 != request["inputs"]["profile_sha256"]:
        raise ValueError("PRIVATE_SNAPSHOT_HASH_MISMATCH")
    shape = Part.read(str(folder / "source.step"))
    root = request["inputs"]["root_face_index"]
    if root is None:
        choices = [(face.Area, -index, index) for index, face in enumerate(shape.Faces)
                   if face.Surface.TypeId == "Part::GeomPlane"]
        if not choices:
            raise ValueError("PLANAR_ROOT_UNAVAILABLE")
        root = max(choices)[2]
    try:
        result = compensated_unfold(shape, root, profile.thickness_mm, profile.k_factor_ansi)
    except ValueError as error:
        raise ValueError("NATIVE_UNFOLD_REFUSED:" + str(error)) from error
    for bend in result["bend_ledger"]:
        allowance = profile.allowance_mm(bend["inner_radius_mm"], math.degrees(bend["sweep_rad"]),
            bend["thickness_mm"], material=profile.material, process=profile.process, tooling=profile.tooling)
        if abs(allowance - bend["bend_allowance_mm"]) > 1e-7:
            raise ValueError("PROFILE_ALLOWANCE_MISMATCH")
    serialized = serialize_compensated_draft(shape, result, request["inputs"]["source_sha256"], profile.sha256)
    serialized["root_selection_rule"] = request["inputs"]["root_rule"]
    serialized["root_selection_is_not_shop_datum_or_bend_order"] = True
    return serialized


def _worker(folder, request):
    from backend.vendor_manufacturing_input import audit_step_bytes
    from backend.vendor_manufacturing_bridge import bind_vendor_bend_ids
    from backend.vendor_manufacturing_dxf import build_analytic_dxf, read_analytic_dxf
    raw = _read_bounded(folder / "source.step", MAX_STEP_BYTES)
    inputs, profile = _inputs(raw, request["inputs"]["source_sha256"], request["profile"],
                               request["inputs"]["root_face_index"], request["candidates"])
    if inputs != request["inputs"]:
        raise ValueError("PRIVATE_REQUEST_HASH_MISMATCH")
    units = audit_step_bytes(raw, inputs["source_sha256"])
    # Grandchild inherits the supervised process group; it cannot escape its caps.
    subprocess.run([str(safety.RESOURCES / "bin/python"), "-X", "faulthandler", "-m", "backend.vendor_manufacturing_job",
                    "--native", str(folder)], cwd=ROOT, env=_environment(folder, native=True),
                   stdin=subprocess.DEVNULL, check=True)
    native = safety._load_json(_read_bounded(folder / "native.json", safety.MAX_RESULT_BYTES))
    if native.get("status") == "UNAVAILABLE":
        raise ValueError(native["reason"])
    if native.get("status") == "FAILED":
        raise RuntimeError("NATIVE_WORKER_FAILED:" + native.get("reason", "UNKNOWN"))
    markers = [{**row, "source_sha256": inputs["source_sha256"]} for row in native["folded_bend_markers"]]
    candidates = request["candidates"]
    if candidates is None:
        from scripts.extract_step_topology_bend_axes import extract_step_bend_axes
        axes = extract_step_bend_axes(folder / "source.step")["bend_axes"]
        candidates = [{"feature_id": row["bend_id"], "line_start": row["canonical_line_start"],
                       "line_end": row["canonical_line_end"], "source_sha256": inputs["source_sha256"]} for row in axes]
    bridge = bind_vendor_bend_ids(markers, candidates, inputs["source_sha256"])
    mapping = {row["pair_id"]: row["feature_id"] for row in bridge["assignments"]}
    for key in ("bend_row_provenance", "folded_bend_markers"):
        for row in native[key]:
            row.update(feature_id=mapping[row["pair_id"]], vendor_feature_id_mapping="SOURCE_BOUND_BIJECTION")
    native.update(units="mm", units_verified=True, source_units_are_not_physical_calibration=True,
                  vendor_feature_id_mapping="SOURCE_BOUND_BIJECTION")
    outputs = {}
    for name, rows in (("cut_dxf", native["cut_rows"]),
                       ("bend_guide_dxf", native["cut_rows"] + native["bend_rows"])):
        dxf = build_analytic_dxf(rows, units="mm")
        decoded = read_analytic_dxf(dxf["text"])
        # Canonical re-encoding supplements native coverage; it is not an
        # independent geometric/importer oracle or shop dimensional evidence.
        encoded_again = build_analytic_dxf(decoded["rows"], units=decoded["units"])
        if encoded_again["text"] != dxf["text"]:
            raise ValueError("DXF_ROUNDTRIP_NOT_IDEMPOTENT")
        if abs(dxf["analytic_cut_area_mm2"] - native["flat_bounds"]["area_mm2"]) > max(
                dxf["analytic_cut_area_numerical_tolerance_mm2"], native["coverage_proof"]["area_numerical_tolerance_mm2"]):
            raise ValueError("DXF_NATIVE_CUT_AREA_MISMATCH")
        dxf.update(units_verified=True, source_sha256=inputs["source_sha256"], profile_sha256=profile.sha256,
                   unit_proof_basis="BYTE_BOUND_NATIVE_STEP_MM_AND_UNSCALED_ANALYTIC_ROWS")
        outputs[name] = dxf
    return {"schema": "SHERMAN_MANUFACTURING_FILE_RESULT_V1", "status": "DRAFT_COMPUTED",
            "source_sha256": inputs["source_sha256"], "profile_sha256": profile.sha256,
            "manufacturing_ready": False, "cut_release_allowed": False,
            "source_unit_proof": units, "bend_id_bridge": bridge, "draft": native,
            **outputs, "dxf_roundtrip": {"passed": True, "method": "STRICT_ANALYTIC_PARSE_CANONICAL_REENCODE",
                "independent_importer_evidence": False, "cut_file_has_no_bend_entities": True,
                "physical_dimensions_verified": False}, "release_gates": manufacturing_release_status(profile)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", type=Path)
    mode.add_argument("--native", type=Path)
    args = parser.parse_args()
    folder = args.worker or args.native
    request = safety._load_json(_read_bounded(folder / "request.json", 200_000))
    exit_code = 0
    try:
        result = _native(folder, request) if args.native else _worker(folder, request)
    except ValueError as error:
        result = {"schema": "SHERMAN_MANUFACTURING_FILE_RESULT_V1", "status": "UNAVAILABLE",
                  "source_sha256": request["inputs"]["source_sha256"], "profile_sha256": request["inputs"]["profile_sha256"],
                  "manufacturing_ready": False, "cut_release_allowed": False, "draft": None, "reason": str(error)[:1000]}
    except Exception as error:
        exit_code = 1
        reason = ("NATIVE_SUBPROCESS_EXIT:" + str(error.returncode)) if isinstance(error, subprocess.CalledProcessError) else str(error)
        result = {"schema": "SHERMAN_MANUFACTURING_FILE_RESULT_V1", "status": "FAILED",
                  "source_sha256": request["inputs"]["source_sha256"], "profile_sha256": request["inputs"]["profile_sha256"],
                  "manufacturing_ready": False, "cut_release_allowed": False, "draft": None,
                  "reason": reason[:1000] or type(error).__name__, "error_type": type(error).__name__}
    rendered = json.dumps(result, allow_nan=False)
    if len(rendered.encode()) > safety.MAX_RESULT_BYTES:
        raise ValueError("RESULT_BYTE_BUDGET_EXCEEDED")
    (folder / ("native.json" if args.native else "result.json")).write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
