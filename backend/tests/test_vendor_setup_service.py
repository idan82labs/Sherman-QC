"""Synthetic service contracts with a stub legacy context; no server import."""
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException

from part_catalog import PartCatalogManager


SERVICE_PATH = Path(__file__).resolve().parents[2] / "apps/api/services/vendor_setup_service.py"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    catalog = PartCatalogManager(str(tmp_path / "catalog.sqlite"))
    part = catalog.create_part("SYN-SVC", revision="A")
    catalog.update_part(part.id, cad_file_hash="cad-hash", cad_uploaded_at="upload-1")
    catalog.add_bend_spec(part.id, "B1", 92.0, target_radius=2.5,
                          bend_line_start=[0, 0, 0], bend_line_end=[1, 0, 0])
    catalog.add_bend_spec(part.id, "B2", 88.0, target_radius=3.0,
                          bend_line_start=[0, 1, 0], bend_line_end=[1, 1, 0])
    stub = ModuleType("apps.api.legacy_context")
    stub.legacy = SimpleNamespace(get_catalog=lambda: catalog)
    with monkeypatch.context() as patcher:
        patcher.setitem(sys.modules, "apps.api.legacy_context", stub)
        spec = importlib.util.spec_from_file_location("_vendor_setup_service_test", SERVICE_PATH)
        module = importlib.util.module_from_spec(spec)
        patcher.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield catalog, part.id, module


def request(catalog, part_id, service):
    response = asyncio.run(service.get_vendor_setup(part_id))
    part = response["part"]
    return service.VendorSetupDraft(
        cad_file_hash=part["cad_file_hash"], cad_uploaded_at=part["cad_uploaded_at"],
        part_revision=part["revision"], drawing_file_hash=part["drawing_file_hash"],
        candidate_generation=response["candidate_generation"],
        marks=response["candidates"],
        sequence=[candidate["feature_id"] for candidate in response["candidates"]],
        operator_sequence_confirmed=True,
    )


def test_complete_candidate_round_trip_and_cas(setup):
    catalog, part_id, service = setup
    payload = request(catalog, part_id, service)
    assert payload.marks[0].target_angle == 92.0
    payload.marks[0].review_status = "confirmed"
    first = asyncio.run(service.save_vendor_setup(part_id, payload))["draft"]
    assert first["draft_revision"] == 1
    assert first["marks"][0]["review_status"] == "confirmed"
    assert first["operator_sequence_confirmed"] is True
    assert catalog.get_vendor_setup_draft(part_id)["is_stale"] is False
    with pytest.raises(HTTPException) as error:
        asyncio.run(service.save_vendor_setup(part_id, payload))
    assert error.value.status_code == 409


def test_initial_catalog_order_is_not_operator_confirmation(setup):
    catalog, part_id, service = setup
    payload = request(catalog, part_id, service)
    payload.operator_sequence_confirmed = False
    with pytest.raises(HTTPException, match="Operator must confirm bending order"):
        asyncio.run(service.save_vendor_setup(part_id, payload))


def test_candidate_omission_and_mutation_rejected(setup):
    catalog, part_id, service = setup
    payload = request(catalog, part_id, service)
    payload.marks.pop()
    with pytest.raises(HTTPException, match="Every CAD candidate"):
        asyncio.run(service.save_vendor_setup(part_id, payload))
    payload = request(catalog, part_id, service)
    payload.marks[0].line_end = [3, 0, 0]
    with pytest.raises(HTTPException, match="immutable"):
        asyncio.run(service.save_vendor_setup(part_id, payload))


def test_sequence_and_manual_geometry_rejected(setup):
    catalog, part_id, service = setup
    payload = request(catalog, part_id, service)
    payload.sequence = [payload.sequence[0], payload.sequence[0]]
    with pytest.raises(HTTPException, match="Sequence"):
        asyncio.run(service.save_vendor_setup(part_id, payload))
    payload = request(catalog, part_id, service)
    payload.sequence.append("not-a-candidate")
    with pytest.raises(HTTPException, match="Sequence"):
        asyncio.run(service.save_vendor_setup(part_id, payload))
    payload = request(catalog, part_id, service)
    manual = service.BendMark(feature_id="manual:new", label="M1", source="manual",
                              issue="missing", line_start=[1, 0, 0], line_end=[1, 0, 0])
    payload.marks.append(manual)
    payload.sequence.append(manual.feature_id)
    with pytest.raises(HTTPException, match="must differ"):
        asyncio.run(service.save_vendor_setup(part_id, payload))
    manual.line_end = [float("nan"), 0, 0]
    with pytest.raises(HTTPException, match="finite"):
        asyncio.run(service.save_vendor_setup(part_id, payload))


def test_linked_location_correction(setup):
    catalog, part_id, service = setup
    payload = request(catalog, part_id, service)
    original = payload.marks[0]
    original.status = "removed"
    original.issue = "wrong_location"
    correction = service.BendMark(
        feature_id="manual:corrected", label="M1", source="manual",
        issue="wrong_location", corrects_feature_id=original.feature_id,
        line_start=[0, 0.1, 0], line_end=[1, 0.1, 0],
    )
    payload.marks.append(correction)
    payload.sequence = [payload.sequence[1], correction.feature_id]
    saved = asyncio.run(service.save_vendor_setup(part_id, payload))["draft"]
    assert saved["marks"][-1]["corrects_feature_id"] == original.feature_id


def test_review_confirmation_requires_active_located_mark(setup):
    catalog, part_id, service = setup
    payload = request(catalog, part_id, service)
    payload.marks[0].status = "removed"
    payload.marks[0].issue = "not_a_bend"
    payload.marks[0].review_status = "confirmed"
    payload.sequence.pop(0)
    with pytest.raises(HTTPException, match="cannot remain review-confirmed"):
        asyncio.run(service.save_vendor_setup(part_id, payload))

    unlocated = catalog.add_bend_spec(part_id, "B3", 90.0)
    payload = request(catalog, part_id, service)
    mark = next(item for item in payload.marks if item.feature_id == unlocated.id)
    mark.review_status = "confirmed"
    with pytest.raises(HTTPException, match="Only located"):
        asyncio.run(service.save_vendor_setup(part_id, payload))


def test_flange_preview_is_optional_and_does_not_change_candidates(setup, monkeypatch):
    catalog, part_id, service = setup
    catalog.update_part(part_id, cad_file_path="synthetic.stp", preview_mesh_path="synthetic.ply")
    captured = []
    def preview(path, digest, candidates):
        captured.append((path, digest, len(candidates)))
        return [{"feature_id": candidates[0]["feature_id"], "vertices": [0, 0, 0, 1, 0, 0, 0, 1, 0],
                 "triangles": [0, 1, 2], "source_face_indices": [1, 2],
                 "association": "CAD_ADJACENT_PLANAR_FLANGE_SKINS"}]
    monkeypatch.setattr(service, "build_flange_previews", preview)
    response = asyncio.run(service.get_vendor_setup(part_id))
    assert captured == [("synthetic.stp", "cad-hash", 2)]
    assert response["flange_surface_status"] == "AVAILABLE"
    assert len(response["candidates"]) == 2
    assert response["flange_surfaces"][0]["feature_id"] == response["candidates"][0]["feature_id"]

    def unavailable(*_args):
        raise service.FlangePreviewUnavailable("CAD_HASH_MISMATCH")
    monkeypatch.setattr(service, "build_flange_previews", unavailable)
    response = asyncio.run(service.get_vendor_setup(part_id))
    assert response["flange_surfaces"] == []
    assert response["flange_surface_status"] == "CAD_HASH_MISMATCH"
    assert len(response["candidates"]) == 2
