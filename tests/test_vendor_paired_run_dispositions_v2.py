"""Both-arm durable failure ledger; authored JSON only, no native/data access."""
import json
import pytest
from scripts import evaluate_vendor_cad_upload_unfold_v2 as run


def safe_row(**kwargs):
    return {"part":"AUTHORED","expected_sha256":"a"*64,"status":"UNAVAILABLE",
        "reason":"WHOLE_CASE_WALL_BUDGET_EXCEEDED","process_tree_quiescent":True,
        "native_admission_quiescent":True,"monitor_failed":False,**kwargs}


def test_completed_control_and_killed_challenger_both_retained(tmp_path):
    run._write_result(tmp_path/"CONTROL_ARM_RESULT.json",{"status":"AVAILABLE","candidate_count":3})
    run._write_result(tmp_path/"CHALLENGER_ARM_STARTED.json",{"state":"STARTED"})
    r=run._paired_terminal(tmp_path,safe_row())
    assert r["arms"]["control"]["status"]=="AVAILABLE"
    assert r["arms"]["challenger"]["status"]=="UNAVAILABLE"
    assert r["arms"]["challenger"]["reason"]=="WHOLE_CASE_WALL_BUDGET_EXCEEDED"


def test_failed_control_marks_unstarted_challenger_without_implying_retry(tmp_path):
    run._write_result(tmp_path/"CONTROL_ARM_STARTED.json",{"state":"STARTED"})
    r=run._paired_terminal(tmp_path,safe_row(reason="CONTROL_INTEGRITY_FAILURE"))
    assert r["arms"]["control"]["status"]=="UNAVAILABLE"
    assert r["arms"]["challenger"]["status"]=="NOT_EXECUTED"


def test_unsafe_writer_payloads_are_never_read(tmp_path):
    (tmp_path/"CONTROL_ARM_RESULT.json").write_bytes(b"not JSON: MUST NOT READ")
    r=run._paired_terminal(tmp_path,safe_row(process_tree_quiescent=False))
    assert not r["writers_safe_to_read"]
    assert all(a["status"]=="UNOBSERVED" for a in r["arms"].values())


def test_parent_rejects_partial_available_challenger_and_is_exclusive(tmp_path):
    run._write_result(tmp_path/"CHALLENGER_ARM_RESULT.json",{"status":"AVAILABLE"})
    r=run._paired_terminal(tmp_path,safe_row())
    assert r["arms"]["challenger"]["status"]=="UNAVAILABLE"
    assert json.loads((tmp_path/"PAIRED_TERMINAL.json").read_bytes())==r
    with pytest.raises(FileExistsError):run._paired_terminal(tmp_path,safe_row())


def test_both_arms_use_same_claimed_bytes_and_keep_routine_abstentions(tmp_path,monkeypatch):
    calls=[]
    def one(folder,raw,part,sha):
        calls.append((run.RUN_ARM,raw,part,sha))
        return {"status":"UNAVAILABLE","reason":"AUTHORED_ABSTENTION","candidate_count":0}
    monkeypatch.setattr(run,"_one_upload_result",one)
    result=run._actual_upload_result(tmp_path,b"AUTHORED STEP PLACEHOLDER","AUTHORED","a"*64)
    assert [r[0] for r in calls]==["control","challenger"]
    assert calls[0][1:]==calls[1][1:]
    assert result["paired_comparison"] is True
    for arm in ("CONTROL","CHALLENGER"):
        assert (tmp_path/f"{arm}_ARM_STARTED.json").is_file()
        assert json.loads((tmp_path/f"{arm}_ARM_RESULT.json").read_bytes())["reason"]=="AUTHORED_ABSTENTION"


@pytest.mark.parametrize("raw",[b"",b'{"status":',b"[]",b'{"status":"NOT_A_STATUS"}'])
def test_killed_partial_arm_result_does_not_break_five_case_finalization(tmp_path,raw):
    (tmp_path/"CHALLENGER_ARM_RESULT.json").write_bytes(raw)
    result=run._paired_terminal(tmp_path,safe_row())
    arm=result["arms"]["challenger"]
    assert arm["status"]=="UNAVAILABLE" and arm["reason"]=="WHOLE_CASE_WALL_BUDGET_EXCEEDED"
    assert arm["retained_error_type"] in {"ValueError","JSONDecodeError"}
    assert (tmp_path/"PAIRED_TERMINAL.json").is_file()


def test_oversize_arm_result_is_retained_as_unavailable(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"MAX_RESULT_BYTES",1024)
    (tmp_path/"CHALLENGER_ARM_RESULT.json").write_bytes(b"x"*1025)
    result=run._paired_terminal(tmp_path,safe_row())
    assert result["arms"]["challenger"]["status"]=="UNAVAILABLE"


def test_unknown_upload_selector_never_falls_back_to_native_or_mesh(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from backend.part_catalog import PartCatalogManager
    catalog=PartCatalogManager(str(tmp_path/"authored.sqlite"))
    part=catalog.create_part("AUTHORED")
    route=run._route_isolated(catalog,tmp_path)
    route.__globals__["os"]=SimpleNamespace(environ={"SHERMAN_VENDOR_CAD_RUNTIME":"MISSPELLED"})
    class File:
        filename="authored.step"
        async def read(self,_):return b"AUTHORED PLACEHOLDER"
    response=asyncio.run(route(part.id,File(),True))
    assert response["status"]=="CAD_ANALYSIS_FAILED_NOT_PUBLISHED"
    assert "VENDOR_CAD_RUNTIME_UNRECOGNIZED" in response["bend_analysis_error"]
    assert catalog.get_part(part.id).cad_file_path is None
