"""Exercise response integration without importing live legacy catalog code."""
import ast
import asyncio
import copy
import re
from pathlib import Path
from types import SimpleNamespace

from backend.vendor_unfold_preview import unavailable_unfold_preview
from backend.vendor_unfold_contract_v3 import unavailable as unavailable_v4
from backend.vendor_unfold_contract_v4 import unavailable as unavailable_v5

SOURCE = Path(__file__).resolve().parents[1] / "apps/api/services/vendor_setup_service.py"


def endpoint(cad_path, controlled=False, changed=False, crashed=False, runtime=None, changed_draft=False):
    tree = ast.parse(SOURCE.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "get_vendor_setup")
    helpers = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_natural_label_key"]
    part = SimpleNamespace(cad_file_path=cad_path, cad_file_hash="a" * 64, preview_mesh_path="preview.ply", to_dict=lambda: {"id": "synthetic"})
    spec = SimpleNamespace(id="A", bend_id="B1", bend_line_start=[0, 0, 0], bend_line_end=[10, 0, 0], target_angle=90, target_radius=2)
    generations=iter(["generation", "changed" if changed else "generation"])
    draft = {"cad_file_hash": "a" * 64, "candidate_generation": "generation", "marks": [{"feature_id": "A"}]} if changed_draft else None
    catalog = SimpleNamespace(get_part=lambda _: part, get_bend_specs_for_part=lambda _: [spec], candidate_generation=lambda _: next(generations), get_vendor_setup_draft=lambda _: draft)
    calls = []
    async def threadpool(function, *args):
        calls.append((function, args))
        return function(*args)
    def unfold(path, expected_hash, candidates):
        assert path == cad_path and expected_hash == "a" * 64
        assert candidates[0]["feature_id"] == "A"
        assert candidates[0]["line_start"] == [0, 0, 0]
        if changed_draft:
            part.cad_file_hash = "b" * 64
            draft["cad_file_hash"] = "b" * 64
            draft["candidate_generation"] = "new-generation"
            draft["marks"][0]["feature_id"] = "NEW"
        if crashed:
            raise RuntimeError("authored worker failure")
        return unavailable_unfold_preview("SYNTHETIC_UNSUPPORTED")
    def unfold_v3(path,expected_hash,candidates):
        unfold(path,expected_hash,candidates)
        return unavailable_v4("SYNTHETIC_UNSUPPORTED")
    def unfold_v4(path,expected_hash,candidates):
        unfold(path,expected_hash,candidates)
        return unavailable_v5("SYNTHETIC_UNSUPPORTED")
    selected_runtime = runtime if runtime is not None else "v2-development" if controlled else None
    scope = {"copy": copy, "re": re, "legacy": SimpleNamespace(get_catalog=lambda: catalog), "_line": lambda raw: raw,
        "os":SimpleNamespace(environ={"SHERMAN_VENDOR_UNFOLD_RUNTIME":selected_runtime} if selected_runtime else {}),
        "run_in_threadpool": threadpool, "build_flange_previews": lambda *a: [],
        "FlangePreviewUnavailable": ValueError, "build_unfold_preview_bounded": unfold, "build_unfold_preview_v2_bounded":unfold,
        "build_unfold_preview_v3_bounded":unfold_v3,
        "build_unfold_preview_v4_bounded":unfold_v4,
        "unavailable_unfold_preview_v5":unavailable_v5,
        "unavailable_unfold_preview_v4":unavailable_v4,
        "unavailable_unfold_preview": unavailable_unfold_preview}
    exec(compile(ast.Module(body=[*helpers, node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return asyncio.run(scope["get_vendor_setup"]("synthetic")), calls


def test_response_preserves_candidates_when_unfolding_unavailable():
    response, calls = endpoint("synthetic.step")
    assert len(calls) == 2
    assert len(response["candidates"]) == 1
    assert response["unfold_preview"]["status"] == "UNAVAILABLE"
    assert response["unfold_preview"]["reason"] == "SYNTHETIC_UNSUPPORTED"
    assert response["unfold_preview"]["source_sha256"] is None


def test_absent_cad_does_not_launch_native_worker():
    response, calls = endpoint(None)
    assert calls == []
    assert response["unfold_preview"]["reason"] == "CAD_SOURCE_UNAVAILABLE"


def test_controlled_path_never_runs_unbounded_folded_flange_aid():
    response,calls=endpoint("synthetic.step",controlled=True)
    assert len(calls)==1 and response["flange_surfaces"]==[]
    assert response["flange_surface_status"]=="CONTROLLED_PATH_FOLDED_FLANGE_NOT_QUALIFIED"
    assert len(response["candidates"])==1


def test_changed_candidates_fail_closed_without_discarding_candidate_snapshot():
    response,calls=endpoint("synthetic.step",controlled=True,changed=True)
    assert len(calls)==1 and len(response["candidates"])==1
    assert response["candidate_generation"]=="generation"
    assert response["unfold_preview"]["reason"]=="CAD_CANDIDATES_CHANGED_DURING_PREVIEW"
    assert response["unfold_preview"]["vertices"]==[]


def test_optional_worker_failure_keeps_catalog_response():
    response,calls=endpoint("synthetic.step",controlled=True,crashed=True)
    assert len(calls)==1 and len(response["candidates"])==1
    assert response["unfold_preview"]["reason"]=="UNFOLD_WORKER_UNAVAILABLE"


def test_v2_import_is_explicit_and_does_not_replace_frozen_v1():
    imports={node.module:node.names[0].asname for node in ast.parse(SOURCE.read_text()).body if isinstance(node,ast.ImportFrom)}
    assert imports["backend.vendor_unfold_worker_v2"]=="build_unfold_preview_v2_bounded"
    assert "backend.vendor_unfold_worker" in imports


def test_v3_routes_only_new_worker_and_typed_v4_refusal():
    response,calls=endpoint("synthetic.step",runtime="v3-development")
    assert len(calls)==1 and calls[0][0].__name__=="unfold_v3"
    assert response["unfold_preview"]["diagnostics"]["mechanism_version"]==4
    assert response["unfold_preview"]["diagnostics"]["transport_version"]==3
    assert response["flange_surfaces"]==[] and len(response["candidates"])==1


def test_v3_absent_crashed_and_changed_paths_never_relabel_as_v3_core():
    for path,changed,crashed,reason in [(None,False,False,"CAD_SOURCE_UNAVAILABLE"),
            ("synthetic.step",False,True,"UNFOLD_WORKER_UNAVAILABLE"),
            ("synthetic.step",True,False,"CAD_CANDIDATES_CHANGED_DURING_PREVIEW")]:
        response,_=endpoint(path,changed=changed,crashed=crashed,runtime="v3-development")
        preview=response["unfold_preview"]
        assert preview["status"]=="UNAVAILABLE" and preview["reason"]==reason
        assert preview["diagnostics"]["mechanism_version"]==4 and preview["vertices"]==[]


def test_unknown_configured_profile_cannot_fall_back_to_unbounded_legacy_work():
    response,calls=endpoint("synthetic.step",runtime="misspelled-profile")
    assert calls==[] and response["flange_surfaces"]==[]
    assert response["unfold_preview"]["reason"]=="UNFOLD_RUNTIME_UNRECOGNIZED"
    assert len(response["candidates"])==1


def test_concurrent_draft_change_cannot_mix_new_plan_with_old_cad_snapshot():
    response,calls=endpoint("synthetic.step",changed=True,changed_draft=True,runtime="v3-development")
    assert len(calls)==1
    assert response["unfold_preview"]["reason"]=="CAD_CANDIDATES_CHANGED_DURING_PREVIEW"
    assert response["candidate_generation"]=="generation"
    assert response["candidates"][0]["feature_id"]=="A"
    assert response["draft"]=={"cad_file_hash":"a"*64,"candidate_generation":"generation","marks":[{"feature_id":"A"}]}


def test_v4_controlled_path_and_all_failure_paths_keep_typed_v5_boundary():
    for path,changed,crashed,reason in [("synthetic.step",False,False,"SYNTHETIC_UNSUPPORTED"),
        (None,False,False,"CAD_SOURCE_UNAVAILABLE"),("synthetic.step",True,False,"CAD_CANDIDATES_CHANGED_DURING_PREVIEW"),
        ("synthetic.step",False,True,"UNFOLD_WORKER_UNAVAILABLE")]:
        response,calls=endpoint(path,changed=changed,crashed=crashed,runtime="v4-development")
        p=response["unfold_preview"]
        assert p["status"]=="UNAVAILABLE" and p["reason"]==reason
        assert p["diagnostics"]["mechanism_version"]==5 and p["diagnostics"]["transport_version"]==4
        assert response["flange_surfaces"]==[] and len(calls)==(1 if path else 0)
