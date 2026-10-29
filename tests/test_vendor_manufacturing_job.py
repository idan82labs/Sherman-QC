"""Pure admission/failure transport; no STEP parser or FreeCAD jobs."""
import hashlib
import ast
import inspect
import json
from pathlib import Path

import pytest

from backend import vendor_manufacturing_job as job

RAW = b"synthetic placeholder: no native parsing in these tests"
HASH = hashlib.sha256(RAW).hexdigest()


def profile():
    return dict(schema="SHERMAN_COMPENSATION_PROFILE_V1", profile_id="synthetic", revision="1",
        material="synthetic", process="synthetic", tooling="synthetic", thickness_mm=1.5,
        k_factor_ansi=.4, inner_radius_min_mm=.1, inner_radius_max_mm=100,
        sweep_min_deg=1, sweep_max_deg=175, evidence_sha256=None, purpose="synthetic_development")


@pytest.fixture
def pins(monkeypatch):
    monkeypatch.setattr(job, "source_pins", lambda: {"fake": "a" * 64})


def test_freeze_binds_every_declaration(pins):
    base = job.freeze_file_job(RAW, HASH, profile())
    assert base["inputs"]["source_sha256"] == HASH
    assert base["inputs"]["root_rule"] == "CAD_ONLY_LARGEST_PLANAR_FACE"
    changed = profile(); changed["k_factor_ansi"] = .5
    assert job.freeze_file_job(RAW, HASH, changed) != base
    assert job.freeze_file_job(RAW, HASH, profile(), root_face_index=0) != base


@pytest.mark.parametrize("value", [True, -1, 1024, "0"])
def test_invalid_root_refused_before_launch(pins, value):
    with pytest.raises(ValueError, match="EXPLICIT_ROOT"):
        job.freeze_file_job(RAW, HASH, profile(), root_face_index=value)


def test_bad_hash_before_native_parser(pins):
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        job.freeze_file_job(RAW, "a"*64, profile())


@pytest.mark.parametrize("candidate", [None, {}, {"feature_id": "B1", "line_start": [0,0,0],
    "line_end": [1,0,0], "source_sha256": HASH, "unused_blob": "not serialized"}])
def test_strict_candidate_schema(pins, candidate):
    with pytest.raises(ValueError):
        job.freeze_file_job(RAW, HASH, profile(), candidates=[candidate])


def test_pin_mismatch_before_admission_or_reservation(tmp_path, pins, monkeypatch):
    frozen = job.freeze_file_job(RAW, HASH, profile())
    frozen["source_sha256"]["fake"] = "b"*64
    monkeypatch.setattr(job, "_inputs", lambda *args: pytest.fail("admission happened before source freeze check"))
    path = tmp_path / "record.json"
    with pytest.raises(ValueError, match="FREEZE_MISMATCH"):
        job.run_file_job(RAW, HASH, profile(), frozen, path)
    assert not path.exists()


def test_popen_failure_terminal_record_exclusive(tmp_path, pins, monkeypatch):
    frozen = job.freeze_file_job(RAW, HASH, profile())
    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("authored")))
    path = tmp_path / "record.json"
    result = job.run_file_job(RAW, HASH, profile(), frozen, path)
    assert result["status"] == "FAILED" and result["error_type"] == "OSError"
    assert json.loads(path.read_text())["status"] == "FAILED"
    with pytest.raises(FileExistsError):
        job.run_file_job(RAW, HASH, profile(), frozen, path)


def test_refusal_kept_as_unavailable_not_success(tmp_path, pins, monkeypatch):
    frozen = job.freeze_file_job(RAW, HASH, profile())
    monkeypatch.setattr(job.safety, "_group_rows", lambda pid: [])
    monkeypatch.setattr(job.safety, "_stop_group", lambda process: None)
    class Process:
        pid, returncode = 123, 0
        def __init__(self, args, **kwargs):
            folder = Path(args[-1])
            (folder / "result.json").write_text(json.dumps({"status": "UNAVAILABLE", "draft": None,
                "source_sha256": HASH, "profile_sha256": frozen["inputs"]["profile_sha256"],
                "manufacturing_ready": False, "cut_release_allowed": False, "reason": "UNSUPPORTED_MODEL"}))
        def poll(self): return 0
    monkeypatch.setattr(job.subprocess, "Popen", Process)
    result = job.run_file_job(RAW, HASH, profile(), frozen, tmp_path / "record.json")
    assert result["status"] == "UNAVAILABLE" and result["result"]["draft"] is None
    assert not result["cut_release_allowed"]


def test_minimal_child_environment_excludes_inherited_loader(tmp_path):
    env = job._environment(tmp_path, native=True)
    assert "DYLD_LIBRARY_PATH" not in env and "PYTHONHOME" not in env
    assert env["PYTHONPYCACHEPREFIX"].startswith(str(tmp_path))


@pytest.mark.parametrize("result", [{}, {"status": "DRAFT_COMPUTED", "manufacturing_ready": True},
    {"status": "UNAVAILABLE", "source_sha256": HASH, "profile_sha256": "b"*64,
     "manufacturing_ready": False, "cut_release_allowed": False, "reason": "x", "draft": {}}])
def test_bad_result_cannot_be_complete(result):
    with pytest.raises(ValueError):
        job._validate_result(result, {"source_sha256": HASH, "profile_sha256": "b"*64})


def test_unavailable_cannot_transport_partial_cut_geometry():
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        job._validate_result({"status": "UNAVAILABLE", "source_sha256": HASH, "profile_sha256": "b"*64,
            "manufacturing_ready": False, "cut_release_allowed": False, "draft": None,
            "reason": "x", "cut_dxf": "partial"}, {"source_sha256": HASH, "profile_sha256": "b"*64})


def test_false_unit_proof_cannot_be_a_complete_draft():
    draft = {"source_sha256": HASH, "profile_sha256": "b"*64, "units": "mm", "units_verified": True,
             "manufacturing_ready": False, "release_allowed": False}
    result = {"status": "DRAFT_COMPUTED", "source_sha256": HASH, "profile_sha256": "b"*64,
              "manufacturing_ready": False, "cut_release_allowed": False, "draft": draft,
              "source_unit_proof": {"source_sha256": HASH, "units_verified_from_native_step": False},
              "bend_id_bridge": {"all_native_and_vendor_bends_bijective": True},
              "cut_dxf": {"text": "not DXF"}, "bend_guide_dxf": {"text": "not DXF"},
              "dxf_roundtrip": {"passed": True}}
    with pytest.raises(ValueError, match="INCOMPLETE"):
        job._validate_result(result, {"source_sha256": HASH, "profile_sha256": "b"*64})


def test_final_elapsed_cap_even_after_fast_leader_exit(tmp_path, pins, monkeypatch):
    frozen = job.freeze_file_job(RAW, HASH, profile())
    monkeypatch.setattr(job.safety, "_group_rows", lambda pid: [])
    monkeypatch.setattr(job.safety, "_stop_group", lambda process: None)
    ticks = iter([0, job.safety.MAX_SECONDS + 1])
    monkeypatch.setattr(job.time, "monotonic", lambda: next(ticks))
    class Process:
        pid, returncode = 123, 0
        def __init__(self, args, **kwargs):
            (Path(args[-1]) / "result.json").write_text(json.dumps({"status": "UNAVAILABLE", "draft": None,
                "source_sha256": HASH, "profile_sha256": frozen["inputs"]["profile_sha256"],
                "manufacturing_ready": False, "cut_release_allowed": False, "reason": "UNSUPPORTED_MODEL"}))
        def poll(self): return 0
    monkeypatch.setattr(job.subprocess, "Popen", Process)
    result = job.run_file_job(RAW, HASH, profile(), frozen, tmp_path / "late.json")
    assert result["status"] == "FAILED" and result["reason"] == "WALL_BUDGET_EXCEEDED"


def valid_pure_receipt():
    from backend.vendor_manufacturing_serialization import REQUIRED_PROOFS
    from backend.vendor_manufacturing_dxf import build_analytic_dxf
    cut = [{"type": "LINE", "layer": "CUT_OUTER", "start": a, "end": b} for a, b in
           [([0,0],[10,0]), ([10,0],[10,20]), ([10,20],[0,20]), ([0,20],[0,0])]]
    bends = [{"type": "LINE", "layer": "BEND_UP", "start": [0,10], "end": [10,10]}]
    proof = {key: True for key in REQUIRED_PROOFS}
    proof.update(native_face_count=4, native_bend_count=1, selected_source_face_indices=[0,1],
                 opposite_source_face_indices=[2,3], cap_source_face_indices=[], hole_wall_source_face_indices=[])
    row = {"pair_id": "P1", "feature_id": "B1"}
    draft = dict(source_sha256=HASH, profile_sha256="b"*64, manufacturing_ready=False, release_allowed=False,
        units="mm", units_verified=True, source_units_are_not_physical_calibration=True,
        root_selection_is_not_shop_datum_or_bend_order=True, coverage_proof=proof, cut_rows=cut,
        bend_rows=bends, bend_row_provenance=[row.copy()], folded_bend_markers=[row.copy()])
    result = dict(status="DRAFT_COMPUTED", source_sha256=HASH, profile_sha256="b"*64,
        manufacturing_ready=False, cut_release_allowed=False, draft=draft,
        source_unit_proof=dict(source_sha256=HASH, units="mm", source_mm_per_coordinate=1.0,
            units_verified_from_native_step=True, all_shape_representation_length_units_verified=True,
            physical_calibration_verified=False),
        bend_id_bridge=dict(source_sha256=HASH, all_native_and_vendor_bends_bijective=True,
            nominal_cad_identity_only=True, manufacturing_ready=False,
            assignments=[{**row, "endpoint_error_mm": 0}]),
        dxf_roundtrip=dict(passed=True, independent_importer_evidence=False, physical_dimensions_verified=False,
            cut_file_has_no_bend_entities=True),
        release_gates=dict(manufacturing_ready=False, cut_release_allowed=False, profile_sha256="b"*64))
    for key, rows in (("cut_dxf", cut), ("bend_guide_dxf", cut+bends)):
        dxf = build_analytic_dxf(rows, units="mm")
        dxf.update(source_sha256=HASH, profile_sha256="b"*64, units_verified=True)
        result[key] = dxf
    return result


def test_complete_pure_receipt_is_draft_only():
    job._validate_result(valid_pure_receipt(), {"source_sha256": HASH, "profile_sha256": "b"*64})


@pytest.mark.parametrize("parent,field,value", [
    ("dxf_roundtrip", "independent_importer_evidence", True),
    ("dxf_roundtrip", "physical_dimensions_verified", True),
    ("release_gates", "manufacturing_ready", True),
    ("release_gates", "cut_release_allowed", True),
    ("draft", "source_units_are_not_physical_calibration", False),
    ("draft", "root_selection_is_not_shop_datum_or_bend_order", False),
])
def test_contradictory_nested_qualification_refused(parent, field, value):
    result = valid_pure_receipt()
    result[parent][field] = value
    with pytest.raises(ValueError, match="INCOMPLETE"):
        job._validate_result(result, {"source_sha256": HASH, "profile_sha256": "b"*64})


def test_application_initialization_precedes_part_import_source_contract():
    tree = ast.parse(inspect.getsource(job._native))
    ordered = [alias.name for node in tree.body[0].body if isinstance(node, ast.Import) for alias in node.names]
    assert ordered.index("FreeCAD") < ordered.index("Part")


def test_native_failure_code_survives_missing_success_result(tmp_path, pins, monkeypatch):
    frozen = job.freeze_file_job(RAW, HASH, profile())
    monkeypatch.setattr(job.safety, "_group_rows", lambda pid: [])
    monkeypatch.setattr(job.safety, "_stop_group", lambda process: None)
    class Process:
        pid, returncode = 123, 1
        def __init__(self, args, **kwargs):
            (Path(args[-1]) / "result.json").write_text(json.dumps({"status": "FAILED", "reason": "NATIVE_SUBPROCESS_EXIT:-11"}))
        def poll(self): return 1
    monkeypatch.setattr(job.subprocess, "Popen", Process)
    result = job.run_file_job(RAW, HASH, profile(), frozen, tmp_path / "crash.json")
    assert result["status"] == "FAILED" and result["reason"] == "FILE_WORKER_FAILED:NATIVE_SUBPROCESS_EXIT:-11"
    assert result["result"]["status"] == "FAILED"
