"""Pure diagnostic transport/graph/admission checks; no physical input."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import diagnose_vendor_unfold_geometry_v1 as diagnostic


def test_components_retains_disconnected_and_cycle():
    link=SimpleNamespace(cylinder=7,feature_id="A",planes=(0,1))
    other=SimpleNamespace(cylinder=8,feature_id="B",planes=(2,3))
    got=diagnostic._components({0:[link],1:[link],2:[other],3:[other]})
    assert got==[{"planes":[0,1],"cylinders":[7],"features":["A"],"tree":True},
                 {"planes":[2,3],"cylinders":[8],"features":["B"],"tree":True}]


def test_components_cycle_is_not_relabelled_tree():
    a=SimpleNamespace(cylinder=10,feature_id="A",planes=(0,1))
    b=SimpleNamespace(cylinder=11,feature_id="B",planes=(1,2))
    c=SimpleNamespace(cylinder=12,feature_id="C",planes=(2,0))
    got=diagnostic._components({0:[a,c],1:[a,b],2:[b,c]})
    assert got[0]["tree"] is False
    assert got[0]["features"]==["A","B","C"]


def test_plain_preserves_triangle_and_coefficient_values():
    import numpy as np
    value={"array":np.array([[1.,2.,3.]]),"i":np.int64(3)}
    assert diagnostic._plain(value)=={"array":[[1.,2.,3.]],"i":3}


def test_admission_refuses_nonquiescent_without_geometry(tmp_path,monkeypatch):
    from backend import vendor_unfold_worker_v3 as shared
    path=tmp_path/"admission"
    path.write_text(json.dumps({"state":"RUNNING","pgid":999999}))
    monkeypatch.setattr(shared,"ADMISSION_LOCK_PATH",path)
    monkeypatch.setattr(diagnostic,"_diagnose_native",lambda *_:pytest.fail("native work admitted"))
    with pytest.raises(ValueError,match="ADMISSION_NOT_QUIESCENT"):
        diagnostic.diagnose_case(tmp_path,b"unused","synthetic","0"*64)
    assert json.loads(path.read_text())=={"state":"RUNNING","pgid":999999}


def test_admission_records_own_group_and_cleans_normal_exception(tmp_path,monkeypatch):
    from backend import vendor_unfold_worker_v3 as shared
    path=tmp_path/"admission"
    path.write_text(json.dumps({"state":"QUIESCENT","pgid":None}))
    monkeypatch.setattr(shared,"ADMISSION_LOCK_PATH",path)
    def fail(*_):
        assert json.loads(path.read_text())=={"state":"RUNNING","pgid":os.getpgrp()}
        raise ValueError("synthetic geometry refused")
    monkeypatch.setattr(diagnostic,"_diagnose_native",fail)
    with pytest.raises(ValueError,match="synthetic geometry refused"):
        diagnostic.diagnose_case(tmp_path,b"unused","synthetic","0"*64)
    assert json.loads(path.read_text())=={"state":"QUIESCENT","pgid":None}


def test_diagnostic_completed_never_claims_preview(tmp_path,monkeypatch):
    from backend import vendor_unfold_worker_v3 as shared
    path=tmp_path/"admission"
    path.write_text(json.dumps({"state":"QUIESCENT","pgid":None}))
    monkeypatch.setattr(shared,"ADMISSION_LOCK_PATH",path)
    monkeypatch.setattr(diagnostic,"_diagnose_native",lambda *_:{"import_status":"UNAVAILABLE"})
    outcome=diagnostic.diagnose_case(tmp_path,b"unused","synthetic","0"*64)
    assert outcome["diagnostic_completed"] is True and outcome["preview_claim"] is False
    assert outcome["flat_status"] is None
    assert (tmp_path/"GEOMETRY_DIAGNOSTIC.json").is_file()


def test_freeze_is_exclusive_and_never_reads_cad(tmp_path,monkeypatch):
    transport=diagnostic.transport
    monkeypatch.setattr(transport,"_source_pins",lambda:{"synthetic":"0"*64})
    monkeypatch.setattr(transport,"_archive_sources",lambda *_:{"source_archive_sha256":"1"*64,
        "source_manifest_sha256":"2"*64})
    root=tmp_path/"freeze"
    frozen=diagnostic.freeze(root)
    assert frozen["profile"]==diagnostic.PROFILE
    assert "DIAGNOSTIC" in frozen["route_method"]
    assert len(list(root.glob("case-*/RESERVED.json")))==5
    assert frozen["score_rule"].startswith("NO_SUPPORT_SCORE")
    with pytest.raises(FileExistsError):
        diagnostic.freeze(root)


def test_candidate_refusal_preserves_computed_ledger(monkeypatch):
    from backend import vendor_unfold_preview as geometry
    from backend import vendor_native_step_upload_v1 as native
    from backend.vendor_unfold_contract_v2 import PreviewContractError
    monkeypatch.setattr(diagnostic,"_native_surface_ledger",
        lambda _,rows:rows.append({"source_face_index":0,"proof":"retained"}))
    monkeypatch.setattr(geometry,"_read_native",lambda _: ([],[{"axis":"retained"}]))
    monkeypatch.setattr(native,"_axis_rows",lambda *_: (_ for _ in ()).throw(
        PreviewContractError("SPLIT_OR_AMBIGUOUS_BEND_SKINS")))
    got=diagnostic._diagnose_native(Path("synthetic-unopened.step"),"0"*64)
    assert got["surface_ledger"]==[{"source_face_index":0,"proof":"retained"}]
    assert got["axes"]==[{"axis":"retained"}]
    assert got["candidate_reason"]=="SPLIT_OR_AMBIGUOUS_BEND_SKINS"
    assert got["flat_status"]=="NOT_RUN_CANDIDATE_REFUSAL"


def test_configuration_binds_child_runner_and_original_source(monkeypatch):
    transport=diagnostic.transport
    for name in ("PROFILE","PRE_PROFILE","PROTOCOL","__file__","OWNED_SOURCE_FILES","_actual_upload_result"):
        monkeypatch.setattr(transport,name,getattr(transport,name))
    diagnostic.configure_transport()
    assert Path(transport.__file__).resolve()==Path(diagnostic.__file__).resolve()
    assert transport.PROFILE==diagnostic.PROFILE
    assert transport._actual_upload_result is diagnostic.diagnose_case
    assert "scripts/evaluate_vendor_cad_upload_unfold_v1.py" in transport.OWNED_SOURCE_FILES
    assert "scripts/diagnose_vendor_unfold_geometry_v1.py" in transport.OWNED_SOURCE_FILES
    completed=subprocess.run([sys.executable,diagnostic.__file__,"--help"],
        capture_output=True,text=True,timeout=10,check=True)
    assert "--case-worker" in completed.stdout
