"""Pure source-only corpus runner contracts; no raw/native imports."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

SOURCE = Path(__file__).parents[1] / "scripts/evaluate_vendor_cad_corpus_v9.py"
spec = importlib.util.spec_from_file_location("corpus_runner_v9", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def inventory_rows():
    rows = [{"relative_path": f"part-{i}.stp", "extension": ".stp", "file_bytes": 10,
             "sha256": f"{i:064x}", "disposition": "supported_format"} for i in range(20)]
    rows += [{**rows[i], "relative_path": f"alias-{i}.STEP", "extension": ".step"} for i in range(3)]
    return rows


def raw(rows):
    return b"\n".join(runner.encode(r) for r in rows)


def test_bindings_keep_aliases_and_never_bind_mesh():
    rows = inventory_rows() + [{"relative_path": "SCAN.ply", "extension": ".ply", "sha256": "f"*64},
                               {"relative_path": "part.dwg", "extension": ".dwg"}]
    cases, metadata = runner.bindings(raw(rows))
    assert len(cases) == 20 and sum(len(c["aliases"]) for c in cases) == 23
    assert {r["disposition"] for r in metadata} == {"METADATA_ONLY_NO_READ", "UNSUPPORTED_DWG"}
    assert all(Path(c["relative_path"]).suffix.lower() in {".step", ".stp"} for c in cases)


@pytest.mark.parametrize("mutation", ["../escape.stp", "/absolute.stp", "SCAN_part.stp", "HOLDOUT10/part.stp"])
def test_unsafe_or_reserved_step_not_accepted(mutation):
    rows = inventory_rows()
    rows[19]["relative_path"] = mutation
    with pytest.raises(ValueError):
        runner.bindings(raw(rows))


def test_size_and_hash_refused():
    rows = inventory_rows()
    rows[0]["file_bytes"] = runner.LIMITS["source_bytes"]+1
    with pytest.raises(ValueError, match="SOURCE_BUDGET"):
        runner.bindings(raw(rows))
    rows[0]["file_bytes"] = 10
    rows[0]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="INVALID_CAD_BINDING"):
        runner.bindings(raw(rows))


def test_exclusive_writes_and_artifact_cap(tmp_path, monkeypatch):
    runner.write(tmp_path / "one.json", {"a": 1})
    with pytest.raises(FileExistsError):
        runner.write(tmp_path / "one.json", {})
    monkeypatch.setitem(runner.LIMITS, "artifact_bytes", 1)
    with pytest.raises(ValueError, match="ARTIFACT_BUDGET"):
        runner.write(tmp_path / "large", b"12")


def test_extraction_survives_folded_and_candidate_refusal(tmp_path, monkeypatch):
    extraction = {"bend_axes": [{"axis": "synthetic", "domain_receipt":{"kind":"CERTIFIED_RECTANGULAR_NATIVE_PCURVE_DOMAIN","certificate_schema_version":7}}], "rejected_groups": [{"reason": "synthetic"}]}
    extractor = types.ModuleType("scripts.extract_step_topology_bend_axes_v10")
    extractor.extract_step_bend_axes = lambda _: extraction
    upload = types.ModuleType("backend.vendor_native_step_upload_v2")
    def refuse(*_):
        raise ValueError("synthetic refusal")
    adapter = types.ModuleType("backend.vendor_cad_bend_candidates_v8")
    adapter.axis_rows_v8 = refuse
    inspector = types.ModuleType("scripts.inspect_vendor_cad_geometry_v1")
    inspector.inspect_geometry = lambda _: {"synthetic": True}
    upload._folded_ply = refuse
    mesh = types.ModuleType("backend.vendor_native_folded_mesh_v2")
    mesh.read_folded_native = refuse
    for name, module in [(extractor.__name__, extractor), (upload.__name__, upload), (mesh.__name__, mesh),
                         (adapter.__name__, adapter), (inspector.__name__, inspector)]:
        monkeypatch.setitem(sys.modules, name, module)
    runner.diagnostic(tmp_path, {"sha256": "a"*64})
    assert json.loads((tmp_path / "EXTRACTION.json").read_text()) == extraction
    result = json.loads((tmp_path / "RESULT.json").read_text())
    assert result["extraction_status"] == "COMPLETE" and result["axis_count"] == 1
    assert result["folded_status"] == result["flat_status"] == "REFUSED"
    assert result["accuracy"] == "UNSCORED"


def test_claim_precedes_snapshot(tmp_path, monkeypatch):
    case = {"sha256": "a"*64}
    frozen = {"cases": [case]}
    monkeypatch.setattr(runner, "verify", lambda *_: frozen)
    folder = tmp_path / "case-01"
    folder.mkdir()
    nonce = "synthetic-nonce"
    monkeypatch.setenv(runner.NONCE_ENV, nonce)
    runner.write(folder / "CASE_LAUNCH.json", {"nonce_sha256": runner.sha(nonce.encode()),
                 "frozen_sha256": "pin", "case_index": 0, "parent_pid": runner.os.getppid()})
    def snapshot(case, directory):
        assert (directory / "CASE_CLAIM.json").exists()
    monkeypatch.setattr(runner, "snapshot", snapshot)
    monkeypatch.setattr(runner, "diagnostic", lambda *_: None)
    runner.child(tmp_path, 0, "pin")
    with pytest.raises(ValueError, match="INVALID_CASE_INDEX"):
        runner.child(tmp_path, -1, "pin")


def test_unknown_cleanup_retains_whole_denominator(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    pre_path = tmp_path / "pre.json"
    runner.write(pre_path, {"decision": "GREEN", "independent_non_author": True, "frozen_sha256": "pin"})
    monkeypatch.setattr(runner, "verify", lambda *_: {"cases": [{} for _ in range(20)]})
    calls = []
    def unknown(*args):
        calls.append(args[1])
        return {"case_index": args[1], "cleanup_known": False, "outer_exit": None}
    monkeypatch.setattr(runner, "run_case", unknown)
    assert runner.execute(attempt, "pin", pre_path, runner.sha(pre_path.read_bytes())) == 1
    terminal = json.loads((attempt / "CAMPAIGN_TERMINAL.json").read_text())
    assert calls == [0] and len(terminal["receipts"]) == 20
    assert terminal["halt_reason"] == "CLEANUP_UNKNOWN"
    assert all(r["status"] == "NOT_EXECUTED" for r in terminal["receipts"][1:])
    with pytest.raises(FileExistsError):
        runner.execute(attempt, "pin", pre_path, runner.sha(pre_path.read_bytes()))


def test_failed_case_cannot_be_reported_as_successful_campaign(tmp_path,monkeypatch):
    attempt=tmp_path/"attempt";attempt.mkdir()
    pre=tmp_path/"pre.json"
    runner.write(pre,{"decision":"GREEN","independent_non_author":True,"frozen_sha256":"pin"})
    monkeypatch.setattr(runner,"verify",lambda *_:{"cases":[{} for _ in range(20)]})
    monkeypatch.setattr(runner,"run_case",lambda _,i,p:{"case_index":i,"cleanup_known":True,"outer_exit":1 if i==3 else 0,"reason":None})
    assert runner.execute(attempt,"pin",pre,runner.sha(pre.read_bytes()))==1
    terminal=json.loads((attempt/"CAMPAIGN_TERMINAL.json").read_text())
    assert terminal["status"]=="COMPLETE_WITH_FAILURES" and len(terminal["receipts"])==20
