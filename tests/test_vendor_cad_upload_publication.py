"""Synthetic upload-publication checks; never import the live API/server."""

import ast
import asyncio
import hashlib
import logging
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Optional
import uuid

import aiofiles
import pytest
from fastapi import HTTPException

from backend.part_catalog import PartCatalogManager


ROOT = Path(__file__).resolve().parents[1]


def bend(label="B1", *, end=(10., 0., 0.)):
    return SimpleNamespace(
        bend_id=label, target_angle=90., target_radius=2.,
        tolerance_angle=1., tolerance_radius=.5,
        bend_line_start=(0., 0., 0.), bend_line_end=end,
    )


def committed(catalog, part_id, marker="a", bends=None):
    reservation = catalog.begin_cad_upload(part_id)
    assert catalog.publish_cad_upload(
        part_id, reservation, f"/{marker}.step", marker * 64,
        f"/{marker}.ply", [bend()] if bends is None else bends, analyzed=True,
    )


def state(catalog, part_id):
    part = catalog.get_part(part_id)
    return (part.cad_file_path, part.cad_file_hash, part.cad_uploaded_at,
            part.preview_mesh_path, part.bend_count,
            [(row.id, row.bend_id, row.bend_line_start, row.bend_line_end)
             for row in catalog.get_bend_specs_for_part(part_id)])


def test_failed_analysis_and_cancel_keep_complete_previous_publication(tmp_path):
    catalog = PartCatalogManager(str(tmp_path / "parts.sqlite"))
    part = catalog.create_part("SYN-UPLOAD")
    committed(catalog, part.id)
    before = state(catalog, part.id)
    failed = catalog.begin_cad_upload(part.id)
    assert state(catalog, part.id) == before
    assert catalog.cancel_cad_upload(part.id, failed)
    assert state(catalog, part.id) == before


def test_newer_reservation_supersedes_older_even_when_cad_hash_would_match(tmp_path):
    catalog = PartCatalogManager(str(tmp_path / "parts.sqlite"))
    part = catalog.create_part("SYN-OVERLAP")
    committed(catalog, part.id)
    before = state(catalog, part.id)
    older = catalog.begin_cad_upload(part.id)
    newer = catalog.begin_cad_upload(part.id)
    assert not catalog.publish_cad_upload(part.id, older, "/old.step", "b" * 64,
                                          "/old.ply", [bend("OLD")], analyzed=True)
    assert not catalog.cancel_cad_upload(part.id, older)
    assert state(catalog, part.id) == before
    assert catalog.publish_cad_upload(part.id, newer, "/new.step", "b" * 64,
                                      "/new.ply", [bend("NEW")], analyzed=True)
    after = state(catalog, part.id)
    assert after[:2] == ("/new.step", "b" * 64)
    assert after[3:5] == ("/new.ply", 1)
    assert [row[1] for row in after[5]] == ["NEW"]


def test_intervening_candidate_edit_refuses_upload_without_mixed_publication(tmp_path):
    catalog = PartCatalogManager(str(tmp_path / "parts.sqlite"))
    part = catalog.create_part("SYN-EDIT")
    committed(catalog, part.id)
    pending = catalog.begin_cad_upload(part.id)
    catalog.add_bend_spec(part.id, "USER-EDIT", 85., bend_line_start=[0, 1, 0],
                          bend_line_end=[10, 1, 0])
    edited = state(catalog, part.id)
    assert not catalog.publish_cad_upload(part.id, pending, "/new.step", "b" * 64,
                                          "/new.ply", [bend("NEW")], analyzed=True)
    assert catalog.cancel_cad_upload(part.id, pending)
    assert state(catalog, part.id) == edited


def test_invalid_bends_roll_back_and_unanalyzed_commit_clears_old_rows_atomically(tmp_path):
    catalog = PartCatalogManager(str(tmp_path / "parts.sqlite"))
    part = catalog.create_part("SYN-INVALID")
    committed(catalog, part.id)
    before = state(catalog, part.id)
    pending = catalog.begin_cad_upload(part.id)
    with pytest.raises(ValueError, match="invalid coordinates"):
        catalog.publish_cad_upload(part.id, pending, "/bad.step", "b" * 64,
                                   "/bad.ply", [bend("BAD", end=(float("nan"), 0., 0.))],
                                   analyzed=True)
    assert state(catalog, part.id) == before
    invalid_nominal = bend("BAD-NOMINAL")
    invalid_nominal.target_angle = float("nan")
    with pytest.raises(ValueError, match="invalid nominal"):
        catalog.publish_cad_upload(part.id, pending, "/bad-angle.step", "b" * 64,
                                   "/bad-angle.ply", [invalid_nominal], analyzed=True)
    assert state(catalog, part.id) == before
    assert catalog.publish_cad_upload(part.id, pending, "/unanalyzed.step", "c" * 64,
                                      None, [], analyzed=False)
    after = state(catalog, part.id)
    assert after[0:2] == ("/unanalyzed.step", "c" * 64)
    assert after[3:] == (None, None, [])


def test_route_uses_reservation_and_single_atomic_publish_not_early_destructive_write():
    source = ROOT / "backend/_server_legacy_impl.py"
    tree = ast.parse(source.read_text())
    endpoint = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "upload_part_cad")
    calls = [node for node in ast.walk(endpoint) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "catalog"]
    names = [node.func.attr for node in calls]
    assert names.count("begin_cad_upload") == 1
    assert names.count("publish_cad_upload") == 1
    assert names.count("cancel_cad_upload") == 1
    assert "update_part" not in names
    assert "delete_bend_specs_for_part" not in names
    assert "replace_bend_specs_for_part" not in names
    begin = next(node for node in calls if node.func.attr == "begin_cad_upload")
    publish = next(node for node in calls if node.func.attr == "publish_cad_upload")
    assert begin.lineno < publish.lineno


def test_preview_route_rejects_different_cad_snapshot_and_keeps_legacy_query_optional(tmp_path):
    source = ROOT / "backend/_server_legacy_impl.py"
    tree = ast.parse(source.read_text())
    endpoint = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "download_part_preview")
    endpoint.decorator_list = []
    preview_root = tmp_path / "parts_previews"
    preview_root.mkdir()
    preview = preview_root / "owned.ply"
    preview.write_bytes(b"synthetic PLY")
    part = SimpleNamespace(id="synthetic", cad_file_hash="a" * 64,
                           preview_mesh_path=str(preview))
    scope = {"Optional": Optional, "Path": Path, "UPLOAD_DIR": tmp_path,
             "HTTPException": HTTPException,
             "get_catalog": lambda: SimpleNamespace(get_part=lambda _: part),
             "FileResponse": lambda path, **kwargs: (path, kwargs)}
    exec(compile(ast.Module(body=[endpoint], type_ignores=[]), str(source), "exec"), scope)
    download = scope["download_part_preview"]
    assert asyncio.run(download(part.id, "a" * 64))[0] == preview
    assert asyncio.run(download(part.id))[0] == preview
    with pytest.raises(HTTPException) as error:
        asyncio.run(download(part.id, "b" * 64))
    assert error.value.status_code == 409
    part.cad_file_hash = "b" * 64
    with pytest.raises(HTTPException) as error:
        asyncio.run(download(part.id, "a" * 64))
    assert error.value.status_code == 409


@pytest.mark.parametrize("runtime",[None,"v2-development"])
def test_step_upload_uses_native_source_and_retains_previous_on_native_refusal(tmp_path, monkeypatch, runtime):
    """Execute only the route AST with mocked native producer and temporary DB."""
    source = ROOT / "backend/_server_legacy_impl.py"
    tree = ast.parse(source.read_text())
    endpoint = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "upload_part_cad")
    endpoint.decorator_list = []
    catalog = PartCatalogManager(str(tmp_path / "parts.sqlite"))
    part = catalog.create_part("SYN-NATIVE")
    committed(catalog, part.id)
    calls = []
    newer_reservations = []
    module = ModuleType("backend.vendor_native_step_upload_v2" if runtime else "backend.vendor_native_step_upload_v1")
    folded = b"ply\nsynthetic-native-folded"

    def native(path, expected_hash, tolerance_angle):
        calls.append((path, expected_hash, tolerance_angle))
        if len(calls) == 2:
            return [], None, {"status": "UNAVAILABLE", "reason": "SYNTHETIC_UNSUPPORTED"}
        if len(calls) == 3:
            return [bend("NATIVE-B1")], None, {"status": "AVAILABLE", "source_sha256": expected_hash,
                "folded_mesh_sha256": hashlib.sha256(folded).hexdigest()}
        if len(calls) == 4:
            return [bend("NATIVE-B1")], folded, {"status": "AVAILABLE", "source_sha256": expected_hash,
                "folded_mesh_sha256": "0" * 64}
        if len(calls) == 5:
            return [bend("NATIVE-B1")], folded, {"status": "AVAILABLE", "source_sha256": "0" * 64,
                "folded_mesh_sha256": hashlib.sha256(folded).hexdigest()}
        if len(calls) == 6:
            newer_reservations.append(catalog.begin_cad_upload(part.id))
        row = bend("NATIVE-B1")
        row.to_dict = lambda: {"bend_id": "NATIVE-B1"}
        return [row], folded, {"status": "AVAILABLE", "source_sha256": expected_hash,
            "folded_mesh_sha256": hashlib.sha256(folded).hexdigest()}

    module.preprocess_vendor_step = native
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("STEP upload must not run the legacy mesh extractor")

    scope = {"UploadFile": object, "File": lambda *args, **kwargs: None,
             "Form": lambda *args, **kwargs: None, "Path": Path, "uuid": uuid,
             "MAX_MESH_FILE_SIZE": 100_000_000, "HTTPException": HTTPException,
             "aiofiles": aiofiles, "UPLOAD_DIR": tmp_path, "get_catalog": lambda: catalog,
             "asyncio": asyncio, "logger": logging.getLogger(__name__),
             "load_bend_runtime_config": forbidden, "load_cad_geometry": forbidden,
             "extract_cad_bend_specs": forbidden}
    scope["os"]=SimpleNamespace(environ={"SHERMAN_VENDOR_CAD_RUNTIME":runtime} if runtime else {})
    exec(compile(ast.Module(body=[endpoint], type_ignores=[]), str(source), "exec"), scope)
    upload = scope["upload_part_cad"]

    class FileLike:
        filename = "operator.step"

        async def read(self, _limit):
            return b"SYNTHETIC STEP BYTES"

    first = asyncio.run(upload(part.id, FileLike(), True))
    assert first["status"] == "CAD_PROCESSED"
    assert first["bends_detected"] == 1 and first["preview_available"]
    assert calls[0][1:] == (hashlib.sha256(b"SYNTHETIC STEP BYTES").hexdigest(), 1.0)
    assert [row.bend_id for row in catalog.get_bend_specs_for_part(part.id)] == ["NATIVE-B1"]
    preview = Path(catalog.get_part(part.id).preview_mesh_path)
    assert preview.read_bytes() == folded
    before = state(catalog, part.id)
    files_before = set((tmp_path / "parts_cad").iterdir())

    refused = asyncio.run(upload(part.id, FileLike(), True))
    assert refused["status"] == "CAD_ANALYSIS_FAILED_NOT_PUBLISHED"
    assert "SYNTHETIC_UNSUPPORTED" in refused["bend_analysis_error"]
    assert state(catalog, part.id) == before
    assert set((tmp_path / "parts_cad").iterdir()) == files_before
    preview_files_before = set((tmp_path / "parts_previews").iterdir())
    for invalid in ("PREVIEW", "PREVIEW", "SOURCE"):
        refused_binding = asyncio.run(upload(part.id, FileLike(), True))
        assert refused_binding["status"] == "CAD_ANALYSIS_FAILED_NOT_PUBLISHED"
        assert invalid in refused_binding["bend_analysis_error"]
        assert state(catalog, part.id) == before
        assert set((tmp_path / "parts_cad").iterdir()) == files_before
        assert set((tmp_path / "parts_previews").iterdir()) == preview_files_before
    superseded = asyncio.run(upload(part.id, FileLike(), True))
    assert superseded["status"] == "CAD_UPLOAD_SUPERSEDED_NOT_PUBLISHED"
    assert state(catalog, part.id) == before
    assert set((tmp_path / "parts_cad").iterdir()) == files_before
    assert set((tmp_path / "parts_previews").iterdir()) == preview_files_before
    assert catalog.cancel_cad_upload(part.id, newer_reservations[0])
