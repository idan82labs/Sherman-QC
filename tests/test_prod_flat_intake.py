"""C05 flat intake: upload/sniff/store by sha, E1 parse + binding proposal, operator decision, release.

E2 synthetic hat (real STEP + its flat DXF); E1 runs in supervised
subprocesses (``flat_intake_cli`` and the compile). A flat STEP plate is made
with E1's ``testparts.write_flat_step`` from the DXF outline.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from apps.production_line.recipes.flat_intake import (
    FlatIntakeError,
    FlatIntakeService,
    FlatTool,
    binding_basis_sha256,
    binding_marks,
    detect_flat_format,
)
from apps.production_line.recipes.release_service import ReleaseGateError
from apps.production_line.recipes.synthetic_onboarding import release_request
from tests.test_prod_recipe_routes import api, h  # noqa: F401 (fixture)
from tests.test_prod_release_service import (  # noqa: F401 (fixture)
    ADMIN,
    REPO,
    NeverCompile,
    do_release,
    hat_case,
    make_env,
    refused,
    resave_draft,
)

assert api and hat_case

TOOL = FlatTool(source_root=REPO, python_exe=Path(sys.executable))
OPERATOR = {"kind": "user", "user_id": "2", "username": "otto", "role": "operator"}


class GateProbe:
    """Records what the gate would compile, then stops (no E1 run)."""

    class Stop(Exception):
        pass

    def __init__(self) -> None:
        self.recipe = None
        self.flat_path = None

    def compile(self, recipe, *, cad_path, flat_path, staging_root):
        self.recipe, self.flat_path = recipe, flat_path
        raise self.Stop()


def service(env, **kw) -> FlatIntakeService:
    return FlatIntakeService(catalog=env.port, store=env.store, data_root=env.root, tool=kw.pop("tool", TOOL), **kw)


def upload(env, data: bytes, name: str = "flat.dxf", **kw):
    return service(env, **kw).ingest_stream(env.ob.part_id, io.BytesIO(data), name, OPERATOR)


def flats_dir_names(env) -> list[str]:
    d = env.root / "recipes" / "flats"
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


# ---------------------------------------------------------------------------- sniffing


def test_detect_flat_format() -> None:
    assert detect_flat_format("a.DXF", b"  0\r\nSECTION\r\n  2\r\nHEADER\r\n") == ("DXF", ".dxf")
    assert detect_flat_format("a.dxf", b"\xef\xbb\xbf0\nSECTION\n") == ("DXF", ".dxf")
    assert detect_flat_format("a.dxf", b"AutoCAD Binary DXF\r\n\x1a\x00") == ("DXF", ".dxf")
    assert detect_flat_format("a.stp", b"ISO-10303-21;\nHEADER;") == ("STEP", ".stp")
    assert detect_flat_format("a.step", b"ISO-10303-21;") == ("STEP", ".step")
    for name, head in (("a.dxf", b"ISO-10303-21;"), ("a.step", b"0\nSECTION\n"), ("a.dwg", b"AC1032"),
                       ("a.dxf", b"%PDF-1.4"), ("noext", b"0\nSECTION\n"), ("a.dxf", b"0\nENTITIES\n")):
        with pytest.raises(FlatIntakeError) as err:
            detect_flat_format(name, head)
        assert err.value.codes == ("FLAT_FORMAT_UNSUPPORTED",), name


def test_binding_basis_follows_marks_not_order(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    draft = env.catalog.get_vendor_setup_draft(env.ob.part_id)
    cad = env.catalog.get_part(env.ob.part_id).cad_file_hash
    marks = binding_marks(draft, cad)
    assert [m["feature_id"] for m in marks] == env.ob.sequence
    reordered = {**draft, "sequence": list(reversed(draft["sequence"]))}
    assert binding_basis_sha256(cad, binding_marks(reordered, cad)) == binding_basis_sha256(cad, marks)
    fewer = {**draft, "sequence": draft["sequence"][1:]}
    assert binding_basis_sha256(cad, binding_marks(fewer, cad)) != binding_basis_sha256(cad, marks)
    assert binding_basis_sha256("0" * 64, marks) != binding_basis_sha256(cad, marks)


# ---------------------------------------------------------------------------- upload


def test_upload_parse_and_refusals(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    data = (hat_case / "flat.dxf").read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    rec = upload(env, data)
    assert rec.state == "UPLOADED" and rec.format == "DXF" and rec.flat_sha256 == sha and rec.revision == 1
    assert rec.stored_rel_path == f"recipes/flats/{sha}.dxf" and (env.root / rec.stored_rel_path).read_bytes() == data
    geo = rec.parse["geometry"]
    assert geo["units"] == "mm" and len(geo["outer"]) >= 4 and [b["line_id"] for b in geo["bend_lines"]] == [
        "L1", "L2", "L3", "L4"]
    assert rec.uploaded_by == {"id": "2", "kind": "user", "role": "operator"}
    # Same bytes again: idempotent storage, a new revision of the part's flat row.
    assert upload(env, data).revision == 2 and flats_dir_names(env) == [f"{sha}.dxf"]
    # Refusals leave nothing behind (no incoming temp files, the attached flat untouched).
    for payload, name, code, kw in (
        (b"%PDF-1.4 not a dxf", "flat.dxf", "FLAT_FORMAT_UNSUPPORTED", {}),
        (b"", "flat.dxf", "FLAT_EMPTY", {}),
        (data, "flat.dxf", "FLAT_TOO_LARGE", {"max_bytes": 1000}),
        (data, "flat.pdf", "FLAT_FORMAT_UNSUPPORTED", {}),
    ):
        with pytest.raises(FlatIntakeError) as err:
            upload(env, payload, name, **kw)
        assert err.value.codes == (code,)
        assert flats_dir_names(env) == [f"{sha}.dxf"]
    assert env.store.get_part_flat(env.ob.part_id).revision == 2
    with pytest.raises(FlatIntakeError) as err:
        service(env).ingest_stream("nope", io.BytesIO(data), "f.dxf", OPERATOR)
    assert err.value.codes == ("PART_NOT_FOUND",) and err.value.http_status == 404
    # A DXF-shaped file E1 cannot read is stored but PARSE_FAILED; it blocks proposal and release.
    rec = upload(env, b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n")
    assert rec.state == "PARSE_FAILED" and rec.parse["status"] == "PARSE_FAILED" and rec.parse["reason"]
    with pytest.raises(FlatIntakeError) as err:
        service(env).propose(env.ob.part_id)
    assert err.value.codes == ("FLAT_PARSE_FAILED",)
    assert "FLAT_PARSE_FAILED" in refused(env).codes


# ---------------------------------------------------------------------------- propose / decide / release


def test_propose_confirm_and_release_bound_flat(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    svc = service(env)
    rec = upload(env, (hat_case / "flat.dxf").read_bytes())
    # Undecided flat: the gate refuses before any compile.
    assert refused(env).codes == ("FLAT_BINDING_UNCONFIRMED",)
    with pytest.raises(FlatIntakeError) as err:
        svc.propose(env.ob.part_id, expected_revision=rec.revision + 5)
    assert err.value.codes == ("FLAT_REVISION_CHANGED",) and err.value.http_status == 409
    rec = svc.propose(env.ob.part_id, expected_revision=rec.revision)
    assert rec.state == "PROPOSED" and rec.proposal["status"] == "PROPOSED"
    prop = rec.proposal["proposal"]
    cad = env.catalog.get_part(env.ob.part_id).cad_file_hash
    uids = {m["bend_uid"] for m in binding_marks(env.catalog.get_vendor_setup_draft(env.ob.part_id), cad)}
    assert set(prop["bend_line_map"].values()) == uids and set(prop["bend_line_map"]) == {"L1", "L2", "L3", "L4"}
    assert prop["fit_choice"]["ambiguous"] is False and set(rec.proposal["bend_labels"]) == uids
    assert refused(env).codes == ("FLAT_BINDING_UNCONFIRMED",)  # a proposal is not a confirmation

    # Invalid operator maps are refused, the stored proposal untouched.
    good = dict(prop["bend_line_map"])
    first_line, first_uid = sorted(good.items())[0]
    for bad in ({"L9": first_uid}, {first_line: "bu_" + "0" * 20}, {"L1": first_uid, "L2": first_uid}, {}):
        with pytest.raises(FlatIntakeError) as err:
            svc.set_binding(env.ob.part_id, expected_revision=rec.revision, state="OPERATOR_CONFIRMED",
                            principal=OPERATOR, bend_line_map=bad)
        assert err.value.codes == ("FLAT_BINDING_INVALID",), bad
    with pytest.raises(FlatIntakeError) as err:
        svc.set_binding(env.ob.part_id, expected_revision=rec.revision, state="AUTO_PROPOSED", principal=OPERATOR)
    assert err.value.codes == ("FLAT_BINDING_INVALID",)
    with pytest.raises(FlatIntakeError) as err:
        svc.set_binding(env.ob.part_id, expected_revision=rec.revision - 1, state="OPERATOR_CONFIRMED",
                        principal=OPERATOR)
    assert err.value.codes == ("FLAT_REVISION_CHANGED",)

    rec = svc.set_binding(env.ob.part_id, expected_revision=rec.revision, state="OPERATOR_CONFIRMED", principal=OPERATOR)
    assert rec.state == "CONFIRMED" and rec.binding["method"] == "GRAPH_MATCH_RIGID_FIT"
    assert rec.binding["bend_line_map"] == good and rec.binding["confirmed_by"]["username"] == "otto"
    out = do_release(env)
    flat = out.recipe.flat
    assert flat.format == "DXF" and flat.binding.state == "OPERATOR_CONFIRMED" and flat.binding.bend_line_map == good
    assert flat.binding.confirmed_by.username == "otto" and flat.stored_rel_path == rec.stored_rel_path
    compiled = json.loads((env.root / out.compiled.stored_rel_path).read_text())
    assert compiled["flat"]["binding_state"] == "OPERATOR_CONFIRMED" and compiled["flat_sha256"] == flat.sha256
    assert compiled["k_factor"]["source"] == "BOUND_FLAT"
    assert compiled["flat"]["developed_length_check"]["status"] == "OK"
    assert env.store.get_benchmark_reference(out.recipe.recipe_id) is not None

    # Marks change after the confirmation (a bend removed): the binding is stale, refused before compiling.
    def drop_last(d):
        victim = d["sequence"][-1]
        d["sequence"] = d["sequence"][:-1]
        for m in d["marks"]:
            if m["feature_id"] == victim:
                m.update(status="removed", issue="not_a_bend", review_status="pending")
        d["machine_moves"] = [[f for f in mv if f != victim] for mv in d["machine_moves"]]
        d["machine_moves"] = [mv for mv in d["machine_moves"] if mv]
        d["machine_moves_explicit"] = bool(d["machine_moves"])
    saved = resave_draft(env, drop_last)
    req = release_request(env.ob)
    req["expected_draft_revision"] = saved["draft_revision"]
    req["bend_overrides"] = {k: v for k, v in req["bend_overrides"].items() if k in saved["sequence"]}
    assert "FLAT_BINDING_STALE" in refused(env, req).codes


def test_operator_edited_map_is_checked_by_e1(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    svc = service(env)
    rec = svc.propose(env.ob.part_id, upload(env, (hat_case / "flat.dxf").read_bytes()).revision)
    good = rec.proposal["proposal"]["bend_line_map"]
    swapped = dict(good)
    swapped["L1"], swapped["L2"] = good["L2"], good["L1"]
    rec = svc.set_binding(env.ob.part_id, expected_revision=rec.revision, state="OPERATOR_CONFIRMED",
                          principal=OPERATOR, bend_line_map=swapped)
    assert rec.binding["method"] == "MANUAL"
    with pytest.raises(ReleaseGateError) as err:
        do_release(env)
    assert {"COMPILE_BLOCKED", "FLAT_BINDING_INCONSISTENT"} <= set(err.value.codes)
    assert env.store.list_all() == []

    # The onboarding checklist blocks on that BLOCKED compile: same recipe content as a release now.
    from apps.production_line.recipes.onboarding import checklist

    req = release_request(env.ob)
    req.pop("expected_draft_revision")
    env.store.put_release_draft(env.ob.part_id, {**req, "expected_draft_revision": 0}, 0)

    def compile_step(doc):
        return next(i for i in doc["checklist"] if i["key"] == "compile")

    doc = checklist(env.ob.part_id, catalog=env.port, recipe_store=env.store, data_root=env.root)
    assert doc["ready_to_release"] is False and doc["last_compile"]["blocking"] is True
    assert "FLAT_BINDING_INCONSISTENT" in compile_step(doc)["codes"] and doc["onboarding_state"] == "NEEDS_BEND_ORDER"
    # The operator re-confirms the flat with the proposed map (no Vendor Setup edit): the old block no
    # longer describes the release, the part is releasable again, and the release succeeds.
    rec = svc.set_binding(env.ob.part_id, expected_revision=rec.revision, state="OPERATOR_CONFIRMED",
                          principal=OPERATOR, bend_line_map=good)
    doc = checklist(env.ob.part_id, catalog=env.port, recipe_store=env.store, data_root=env.root)
    assert compile_step(doc)["ok"] and compile_step(doc)["codes"] == [] and doc["last_compile"]["blocking"] is False
    assert doc["ready_to_release"] is True and doc["onboarding_state"] == "READY_TO_RELEASE", doc["blockers"]
    assert do_release(env).recipe.flat.binding.bend_line_map == good


def test_declined_and_dev_auto_and_tampered(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    env = make_env(tmp_path, hat_case)
    svc = service(env)
    rec = svc.propose(env.ob.part_id, upload(env, (hat_case / "flat.dxf").read_bytes()).revision)
    # DEV_AUTO_CONFIRMED (E4 dev recipes only) may release the un-reviewed proposal as AUTO_PROPOSED.
    probe = GateProbe()
    with pytest.raises(GateProbe.Stop):
        do_release(env, {**release_request(env.ob), "confirmation": "DEV_AUTO_CONFIRMED"}, compiler=probe)
    assert probe.recipe.flat.binding.state == "AUTO_PROPOSED" and probe.recipe.confirmation == "DEV_AUTO_CONFIRMED"
    assert probe.flat_path == env.root / rec.stored_rel_path
    # Declined: released unbound (the flat is kept and shown; E1 takes no K from it).
    rec = svc.set_binding(env.ob.part_id, expected_revision=rec.revision, state="UNBOUND", principal=OPERATOR)
    assert rec.state == "DECLINED" and rec.binding["decided_by"]["user_id"] == "2"
    out = do_release(env)
    assert out.recipe.flat.binding.state == "UNBOUND"
    compiled = json.loads((env.root / out.compiled.stored_rel_path).read_text())
    assert compiled["flat"]["binding_state"] == "UNBOUND" and compiled["k_factor"]["source"] == "SHOP_DEFAULT"
    assert "FLAT_NOT_BOUND" in out.compile_report["warning_codes"]
    # Tampered or missing stored bytes are refused at the next release.
    stored = env.root / rec.stored_rel_path
    stored.write_bytes(stored.read_bytes() + b"999\nx\n")
    req = {**release_request(env.ob), "release_note": "second"}
    assert refused(env, req).codes == ("FLAT_HASH_MISMATCH",)
    stored.unlink()
    assert refused(env, req).codes == ("FLAT_MISSING",)
    # Detaching needs the flat's revision (a stale tab must not delete what it has not seen).
    with pytest.raises(FlatIntakeError) as err:
        svc.delete(env.ob.part_id, None)
    assert err.value.codes == ("FLAT_REVISION_REQUIRED",) and err.value.http_status == 422
    current = env.store.get_part_flat(env.ob.part_id)
    with pytest.raises(FlatIntakeError) as err:
        svc.delete(env.ob.part_id, current.revision - 1)
    assert err.value.codes == ("FLAT_REVISION_CHANGED",)
    # Detaching the flat makes the part releasable without one.
    assert svc.delete(env.ob.part_id, current.revision) is True and env.store.get_part_flat(env.ob.part_id) is None
    probe = GateProbe()
    with pytest.raises(GateProbe.Stop):
        do_release(env, req, compiler=probe)
    assert probe.recipe.flat is None and probe.flat_path is None


def test_flat_step_intake(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    from domains.production.engine.compile import testparts as tp
    from domains.production.engine.compile.flat import parse_flat

    env = make_env(tmp_path, hat_case)
    parsed = parse_flat(hat_case / "flat.dxf")
    thickness = json.loads((hat_case / "recipe_draft.json").read_text())["summary"]["thickness_mm"]
    step = tp.write_flat_step(tmp_path / "plate.step", parsed.outer, thickness)
    svc = service(env)
    rec = upload(env, step.read_bytes(), "plate.step")
    assert rec.format == "STEP" and rec.state == "UPLOADED" and rec.stored_rel_path.endswith(".step")
    assert rec.parse["geometry"]["bend_lines"] == []  # a flat STEP carries no bend annotations
    rec = svc.propose(env.ob.part_id, rec.revision)
    lines = [ln["line_id"] for ln in rec.proposal["proposal"]["lines"]]
    assert lines and all(lid.startswith("I") for lid in lines)  # bend lines inferred geometrically by E1
    rec = svc.set_binding(env.ob.part_id, expected_revision=rec.revision, state="OPERATOR_CONFIRMED", principal=OPERATOR)
    probe = GateProbe()
    with pytest.raises(GateProbe.Stop):
        do_release(env, compiler=probe)
    assert probe.recipe.flat.format == "STEP" and set(probe.recipe.flat.binding.bend_line_map) <= set(lines)




def test_tool_timeout_is_explicit(tmp_path: Path, hat_case: Path) -> None:  # noqa: F811
    engine = tmp_path / "engine" / "apps" / "production_line" / "recipes"
    engine.mkdir(parents=True)
    for pkg in (tmp_path / "engine" / "apps", tmp_path / "engine" / "apps" / "production_line", engine):
        (pkg / "__init__.py").write_text("")
    (engine / "flat_intake_cli.py").write_text("import time\ntime.sleep(30)\n")
    env = make_env(tmp_path, hat_case)
    slow = FlatTool(source_root=tmp_path / "engine", python_exe=Path(sys.executable), parse_timeout_s=1.0,
                    kill_grace_s=0.5)
    with pytest.raises(FlatIntakeError) as err:
        upload(env, (hat_case / "flat.dxf").read_bytes(), tool=slow)
    assert err.value.codes == ("FLAT_TOOL_TIMEOUT",) and err.value.http_status == 504
    assert env.store.get_part_flat(env.ob.part_id) is None


# ---------------------------------------------------------------------------- HTTP


def test_flat_routes(api) -> None:  # noqa: F811
    pid = api.ob.part_id
    data = (Path(api.ob.case_dir) / "flat.dxf").read_bytes()
    assert api.client.get(f"/api/production/parts/{pid}/flat", headers=h("tok-viewer")).status_code == 404
    r = api.client.post(f"/api/production/parts/{pid}/flat", files={"file": ("flat.dxf", data)}, headers=h("tok-operator"))
    assert r.status_code == 201, r.text
    assert r.json()["state"] == "UPLOADED" and len(r.json()["parse"]["geometry"]["bend_lines"]) == 4
    r = api.client.post(f"/api/production/parts/{pid}/flat", files={"file": ("flat.dxf", b"%PDF")},
                        headers=h("tok-operator"))
    assert r.status_code == 422 and r.json()["detail"]["codes"] == ["FLAT_FORMAT_UNSUPPORTED"]
    r = api.client.post(f"/api/production/parts/{pid}/flat/proposal?expected_revision=1", headers=h("tok-operator"))
    assert r.status_code == 200 and r.json()["state"] == "PROPOSED"
    assert r.json()["review"]["ambiguous"] is False and r.json()["review"]["needs_operator_review"] is False
    rev = r.json()["revision"]
    r = api.client.put(f"/api/production/parts/{pid}/flat/binding",
                       json={"expected_revision": rev, "state": "OPERATOR_CONFIRMED"}, headers=h("tok-operator"))
    assert r.status_code == 200 and r.json()["state"] == "CONFIRMED"
    r = api.client.put(f"/api/production/parts/{pid}/flat/binding",
                       json={"expected_revision": rev, "state": "UNBOUND"}, headers=h("tok-operator"))
    assert r.status_code == 409 and r.json()["detail"]["codes"] == ["FLAT_REVISION_CHANGED"]
    r = api.client.put(f"/api/production/parts/{pid}/flat/binding", json={"state": "UNBOUND"}, headers=h("tok-operator"))
    assert r.status_code == 422
    doc = api.client.get(f"/api/production/parts/{pid}/onboarding", headers=h("tok-viewer")).json()
    assert doc["flat"]["state"] == "CONFIRMED" and doc["onboarding_state"] in ("NEEDS_TOLERANCES", "READY_TO_RELEASE")
    rel = api.client.post(f"/api/production/parts/{pid}/recipe/release", json=release_request(api.ob),
                          headers=h("tok-admin"))
    assert rel.status_code == 201 and rel.json()["recipe"]["flat"]["binding"]["state"] == "OPERATOR_CONFIRMED"
    r = api.client.delete(f"/api/production/parts/{pid}/flat?expected_revision=1", headers=h("tok-operator"))
    assert r.status_code == 409
    r = api.client.delete(f"/api/production/parts/{pid}/flat", headers=h("tok-operator"))
    assert r.status_code == 422 and r.json()["detail"]["codes"] == ["FLAT_REVISION_REQUIRED"]
    r = api.client.delete(f"/api/production/parts/{pid}/flat?expected_revision={rev + 1}", headers=h("tok-operator"))
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert api.client.delete(f"/api/production/parts/{pid}/flat?expected_revision={rev + 1}",
                             headers=h("tok-operator")).status_code == 404
    # A re-upload continues the revision counter: the deleted flat's revisions never match the new one.
    r = api.client.post(f"/api/production/parts/{pid}/flat", files={"file": ("flat.dxf", data)}, headers=h("tok-operator"))
    assert r.status_code == 201 and r.json()["revision"] == rev + 2
    for stale in range(1, rev + 2):
        r = api.client.put(f"/api/production/parts/{pid}/flat/binding",
                           json={"expected_revision": stale, "state": "UNBOUND"}, headers=h("tok-operator"))
        assert r.status_code == 409, stale
