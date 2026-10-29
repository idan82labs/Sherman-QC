"""SolidWorks STEP intake (apps.production_line.recipes.cad_import): validate + idempotent draft import.

Synthetic only: an L bracket and a U channel built with OCP (domains.production.engine.compile.testparts),
a plain box (no bends), an inch-unit STEP, a corrupt and an empty file. Temporary catalog and recipe store;
no scans, no repository data, no legacy server.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.production_line.recipes import cad_import as ci

REPO = Path(__file__).resolve().parents[1]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _box_shape():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    return BRepPrimAPI_MakeBox(50.0, 40.0, 30.0).Shape()


def _write_inch_step(shape, path: Path) -> Path:
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Controller, STEPControl_Writer

    STEPControl_Controller.Init_s()
    assert Interface_Static.SetCVal_s("write.step.unit", "INCH")
    try:
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        assert int(w.Write(str(path))) == 1
    finally:
        Interface_Static.SetCVal_s("write.step.unit", "MM")
    return path


@pytest.fixture(scope="module")
def parts(tmp_path_factory) -> SimpleNamespace:
    from domains.production.engine.compile.testparts import l_part, u_part, write_step

    out = tmp_path_factory.mktemp("returned")
    (out / "sub").mkdir()
    l_path = write_step(l_part()[0], out / "44211000_A.step")
    u_path = write_step(u_part()[0], out / "sub" / "38172000_01.step")
    box = write_step(_box_shape(), out / "50000000_B.step")
    badname = write_step(l_part(a=55.0)[0], out / "bracket final.step")
    inch = _write_inch_step(l_part(a=45.0)[0], out / "47000000_C.step")
    corrupt = out / "49000000_A.step"
    corrupt.write_text("ISO-10303-21;\nHEADER;\nthis is not a STEP body\n", encoding="ascii")
    empty = out / "49100000_A.step"
    empty.write_bytes(b"")
    return SimpleNamespace(root=out, l=l_path, u=u_path, box=box, badname=badname, inch=inch, corrupt=corrupt,
                           empty=empty)


def _ps_manifest(parts: SimpleNamespace, *, tamper: bool = False) -> Path:
    rows = []
    for p, src in ((parts.l, "44211000_A.SLDPRT"), (parts.u, "sub\\38172000_01.SLDPRT"),
                   (parts.badname, "bracket final.SLDPRT")):
        rel = str(p.relative_to(parts.root)).replace("/", "\\")
        sha = _sha(p)
        if tamper and p == parts.l:
            sha = "0" * 64
        rows.append({"source_rel": src, "source_name": Path(src.replace("\\", "/")).name, "configuration": "Default",
                     "output_rel": rel, "step_sha256": sha, "status": "OK", "error": "", "sw_revision": "33.1.0"})
    rows.append({"source_rel": "asm.SLDASM", "source_name": "asm.SLDASM", "status": "UNSUPPORTED_TYPE",
                 "error": ".sldasm is not exported"})
    rows.append({"source_rel": "new.SLDPRT", "source_name": "new.SLDPRT", "status": "FAILED_OPEN",
                 "error": "swFutureVersion (8192)"})
    name = "manifest_tampered.json" if tamper else "manifest.json"
    path = parts.root / name
    # PowerShell 5.1 writes UTF-8 with a BOM
    path.write_text(json.dumps({"tool": "sldprt_to_step 1.0", "files": rows}), encoding="utf-8-sig")
    return path


# ------------------------------------------------------------------ SKU from filename
@pytest.mark.parametrize("name,sku,rev", [
    ("44211000_A.step", "44211000_A", "A"),
    ("30820000_A01.STEP", "30820000_A01", "A01"),
    ("38172000_01__Default.step", "38172000_01", "01"),
    ("49467010_A-RFQ.step", "49467010_A", "A"),
    ("44513027_01A.step", "44513027_01A", "01A"),
    ("2927000_E2.SLDPRT", "2927000_E2", "E2"),
    ("MEC003744_02.step", "MEC003744_02", "02"),
])
def test_sku_parsed(name, sku, rev):
    r = ci.part_sku_from_step_name(name)
    assert r["status"] == "OK" and r["sku"] == sku and r["revision"] == rev


@pytest.mark.parametrize("name,reason", [
    ("bracket final.step", "NAME_NOT_BASE_REV"),
    ("41880000.step", "NO_REVISION"),
    ("47555020_1.step", "REVISION_SINGLE_DIGIT"),
    ("50278001_A_1.step", "NAME_NOT_BASE_REV"),
    ("PS SP 05041143 SUPPORT.step", "NAME_NOT_BASE_REV"),
    (".step", "EMPTY_NAME"),
])
def test_sku_unparseable_is_reported_not_guessed(name, reason):
    r = ci.part_sku_from_step_name(name)
    assert r["status"] == "UNPARSEABLE" and r["sku"] is None and r["reason"] == reason


def test_configuration_suffix_kept():
    assert ci.part_sku_from_step_name("44211000_A__Flat Pattern.step")["configuration"] == "Flat Pattern"


# ------------------------------------------------------------------ STEP validation
def test_l_bracket_and_u_channel_valid(parts):
    for p, bends in ((parts.l, 1), (parts.u, 2)):
        r = ci.validate_step(p, _sha(p))
        assert r["verdict"] == "VALID", r
        assert r["sha_check"] == "MATCH" and r["units"] == "mm"
        assert r["brep"]["solid_count"] == 1
        assert r["sheet_metal"]["constant"] and abs(r["sheet_metal"]["thickness_mm"] - 2.0) < 1e-3
        assert r["compile"]["outcome"] == "FLANGE_TREE_OK" and r["compile"]["graph_bends"] == bends


def test_sha_mismatch_rejected(parts):
    r = ci.validate_step(parts.l, "f" * 64)
    assert r["verdict"] == "INVALID" and r["codes"] == ["STEP_SHA_MISMATCH"]


def test_corrupt_and_empty(parts):
    assert ci.validate_step(parts.empty)["codes"] == ["STEP_EMPTY"]
    r = ci.validate_step(parts.corrupt)
    assert r["verdict"] == "INVALID" and r["codes"] and r["codes"][0].startswith("STEP_")
    assert ci.validate_step(parts.root / "missing.step")["codes"] == ["STEP_FILE_MISSING"]


def test_box_records_compile_blocker(parts):
    r = ci.validate_step(parts.box)
    assert r["verdict"] == "INVALID"
    assert r["compile"]["outcome"] in {"DATUM_PANEL_UNDETERMINED", "FLANGE_GRAPH_NOT_A_TREE", "THICKNESS_INCONSISTENT"}
    assert r["compile"]["outcome"] in r["codes"]


def test_inch_units_flagged(parts):
    assert ci.step_length_unit(parts.inch) == "inch"
    r = ci.validate_step(parts.inch)
    assert "STEP_UNITS_NOT_MM" in r["codes"]


# ------------------------------------------------------------------ manifests
def test_powershell_manifest_rows(parts):
    rows = ci.load_manifest(_ps_manifest(parts))
    assert len(rows) == 5
    u = next(r for r in rows if r["source_name"] == "38172000_01.SLDPRT")
    assert Path(u["step_path"]) == parts.u and u["manifest_step_sha256"] == _sha(parts.u)
    rep = ci.validate_rows(rows, isolate=False)
    by = {r["source_name"]: r for r in rep["rows"]}
    assert by["asm.SLDASM"]["validation"]["codes"] == ["CONVERSION_FAILED"]
    assert by["new.SLDPRT"]["validation"]["codes"] == ["CONVERSION_FAILED"]
    assert by["44211000_A.SLDPRT"]["validation"]["verdict"] == "VALID"
    assert by["bracket final.SLDPRT"]["sku"]["status"] == "UNPARSEABLE"
    assert rep["counts"]["valid"] == 3 and rep["counts"]["rows"] == 5


def test_vba_manifest_csv(parts):
    path = parts.root / "manifest_vba.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_rel", "source_name", "source_size", "source_modified", "configuration", "output_rel",
                    "step_size", "status", "error", "warnings", "sw_revision", "finished"])
        w.writerow(["44211000_A.SLDPRT", "44211000_A.SLDPRT", "123", "2026-10-28 10:00:00", "Default",
                    "44211000_A.step", parts.l.stat().st_size, "OK", "", "", "33.1.0", "2026-10-28 10:01:00"])
    rows = ci.load_manifest(path)
    assert rows[0]["manifest_step_sha256"] is None
    r = ci.validate_rows(rows, isolate=False)["rows"][0]
    assert r["validation"]["verdict"] == "VALID" and r["validation"]["sha_check"] == "NO_MANIFEST_SHA"


def test_step_dir_rows(parts):
    names = {Path(r["step_rel"]).name for r in ci.rows_from_step_dir(parts.root)}
    assert {"44211000_A.step", "38172000_01.step", "bracket final.step"} <= names


def test_dry_run_manifest_refused(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"dry_run": True, "files": []}))
    with pytest.raises(ValueError):
        ci.load_manifest(p)


# ------------------------------------------------------------------ import
def _env(tmp_path: Path) -> SimpleNamespace:
    from part_catalog import PartCatalogManager

    from apps.api.services.production_recipe_service import RecipeContext
    from apps.production_line.recipes.catalog_port import CatalogManagerPort
    from apps.production_line.recipes.store import RecipeStore

    manager = PartCatalogManager(str(tmp_path / "catalog.sqlite"))
    root = tmp_path / "root"
    store = RecipeStore(root / "db" / "recipes.sqlite3")
    ctx = RecipeContext(catalog=CatalogManagerPort(manager), store=store, data_root=root)
    return SimpleNamespace(manager=manager, store=store, ctx=ctx, uploads=tmp_path / "uploads")


def _import(env, report):
    return ci.import_validated(report, ctx=env.ctx, manager=env.manager, upload_dir=env.uploads, manifest_label="m")


def test_import_creates_drafts_and_is_idempotent(parts, tmp_path):
    env = _env(tmp_path)
    report = ci.validate_rows(ci.load_manifest(_ps_manifest(parts)), isolate=False)
    first = _import(env, report)
    assert first["counts"] == {"IMPORTED": 2, "NOT_VALID": 2, "SKU_UNPARSEABLE": 1}
    by = {r["sku"]: r for r in first["results"] if r["outcome"] == "IMPORTED"}
    for sku, path in (("44211000_A", parts.l), ("38172000_01", parts.u)):
        part = env.manager.get_part(by[sku]["part_id"])
        assert part.part_number == sku and part.cad_file_hash == _sha(path)
        assert Path(part.cad_file_path).is_file() and _sha(Path(part.cad_file_path)) == _sha(path)
        assert env.store.sku_owner(sku) == part.id
        assert env.store.get_active_for_part(part.id) is None  # draft: nothing released
        assert ci.IMPORT_MARKER in (part.notes or "")
    n_parts = len(env.manager.list_parts(limit=1000))
    n_files = len(list((env.uploads / "parts_cad").iterdir()))
    second = _import(env, report)
    assert second["counts"] == {"ALREADY_IMPORTED": 2, "NOT_VALID": 2, "SKU_UNPARSEABLE": 1}
    assert len(env.manager.list_parts(limit=1000)) == n_parts
    assert len(list((env.uploads / "parts_cad").iterdir())) == n_files


def test_sha_mismatch_not_imported(parts, tmp_path):
    env = _env(tmp_path)
    report = ci.validate_rows(ci.load_manifest(_ps_manifest(parts, tamper=True)), isolate=False)
    res = {r["source_name"]: r for r in _import(env, report)["results"]}
    assert res["44211000_A.SLDPRT"]["outcome"] == "NOT_VALID"
    assert "STEP_SHA_MISMATCH" in res["44211000_A.SLDPRT"]["detail"]
    assert env.store.sku_owner("44211000_A") is None


def test_existing_sku_left_alone_and_step_on_other_part(parts, tmp_path):
    env = _env(tmp_path)
    from apps.api.services.production_recipe_service import RecipeService

    RecipeService(env.ctx).create_part({"part_number": "44211000_A", "part_name": "made by hand"})
    report = ci.validate_rows(ci.rows_from_step_dir(parts.root), isolate=False)
    res = {r["sku"]: r for r in _import(env, report)["results"] if r["sku"]}
    assert res["44211000_A"]["outcome"] == "SKU_EXISTS"
    assert env.manager.get_part(res["44211000_A"]["part_id"]).cad_file_hash is None
    # the same STEP bytes under a second name is another part's CAD already
    dup = tmp_path / "dup"
    dup.mkdir()
    (dup / "38172000_01.step").write_bytes(parts.u.read_bytes())
    (dup / "38999000_02.step").write_bytes(parts.u.read_bytes())
    out = _import(env, ci.validate_rows(ci.rows_from_step_dir(dup), isolate=False))
    assert sorted(r["outcome"] for r in out["results"]) == ["ALREADY_IMPORTED", "STEP_ON_OTHER_PART"]


def test_interrupted_import_is_finished(parts, tmp_path):
    env = _env(tmp_path)
    from apps.api.services.production_recipe_service import RecipeService

    RecipeService(env.ctx).create_part({"part_number": "44211000_A", "notes": f"{ci.IMPORT_MARKER} step_sha256=x"})
    rows = [r for r in ci.load_manifest(_ps_manifest(parts)) if r["source_name"] == "44211000_A.SLDPRT"]
    out = _import(env, ci.validate_rows(rows, isolate=False))
    assert out["counts"] == {"CAD_ATTACHED_RETRY": 1}
    assert _import(env, ci.validate_rows(rows, isolate=False))["counts"] == {"ALREADY_IMPORTED": 1}


def test_two_configurations_same_sku_not_guessed(parts, tmp_path):
    env = _env(tmp_path)
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "44211000_A__Default.step").write_bytes(parts.l.read_bytes())
    from domains.production.engine.compile.testparts import l_part, write_step

    write_step(l_part(a=70.0)[0], d / "44211000_A__Long.step")
    out = _import(env, ci.validate_rows(ci.rows_from_step_dir(d), isolate=False))
    assert out["counts"] == {"DUPLICATE_SKU_IN_BATCH": 2}
    assert env.store.sku_owner("44211000_A") is None


# ------------------------------------------------------------------ CLI (isolated child per file)
def test_cli_validate_and_import(parts, tmp_path):
    env_vars = dict(os.environ, OMP_NUM_THREADS="1")
    manifest = _ps_manifest(parts)
    out = tmp_path / "validation.json"
    p = subprocess.run([sys.executable, "-m", "apps.production_line.recipes.cad_import", "validate",
                        "--manifest", str(manifest), "--timeout-s", "120", "--out", str(out)],
                       cwd=REPO, env=env_vars, capture_output=True, text=True, timeout=600)
    assert p.returncode == 1, p.stderr  # two conversion failures in the manifest
    doc = json.loads(out.read_text())
    assert doc["counts"]["valid"] == 3
    imp = tmp_path / "import.json"
    p = subprocess.run([sys.executable, "-m", "apps.production_line.recipes.cad_import", "import",
                        "--manifest", str(manifest), "--data-root", str(tmp_path / "root"),
                        "--catalog-db", str(tmp_path / "catalog.sqlite"), "--upload-dir", str(tmp_path / "uploads"),
                        "--out", str(imp)], cwd=REPO, env=env_vars, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    assert json.loads(imp.read_text())["import"]["counts"]["IMPORTED"] == 2
    p = subprocess.run([sys.executable, "-m", "apps.production_line.recipes.cad_import", "validate",
                        "--manifest", str(tmp_path / "nope.json"), "--out", str(out)],
                       cwd=REPO, env=env_vars, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2
