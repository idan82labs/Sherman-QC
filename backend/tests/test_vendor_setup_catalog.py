"""Isolated synthetic persistence checks; never import the server or product DB."""
from types import SimpleNamespace

import pytest

from part_catalog import PartCatalogManager


def bend(bend_id, start=(0, 0, 0), end=(1, 0, 0)):
    return SimpleNamespace(
        bend_id=bend_id, bend_line_start=start, bend_line_end=end,
        target_angle=90.0, target_radius=1.0,
        tolerance_angle=1.0, tolerance_radius=0.5,
    )


def draft(part, generation, revision=0):
    return {
        "schema_version": 1, "cad_file_hash": part.cad_file_hash,
        "drawing_file_hash": part.drawing_file_hash,
        "part_revision": part.revision, "candidate_generation": generation,
        "expected_draft_revision": revision,
        "marks": [], "sequence": [], "completed_steps": 0,
        "matching_search": {"search_radius_mm": None, "search_angle_deg": None},
    }


def test_draft_binding_and_optimistic_revision(tmp_path):
    catalog = PartCatalogManager(str(tmp_path / "catalog.sqlite"))
    part = catalog.create_part("SYN-001", revision="A")
    catalog.update_part(part.id, cad_file_hash="cad-a", drawing_file_hash="drawing-a")
    catalog.replace_bend_specs_for_part(part.id, [bend("B1")])
    part = catalog.get_part(part.id)
    first = catalog.save_vendor_setup_draft(part.id, draft(part, catalog.candidate_generation(part.id)))
    assert first["draft_revision"] == 1
    assert catalog.get_vendor_setup_draft(part.id)["is_stale"] is False
    with pytest.raises(ValueError, match="another session"):
        catalog.save_vendor_setup_draft(part.id, draft(part, catalog.candidate_generation(part.id)))
    second = catalog.save_vendor_setup_draft(part.id, draft(part, catalog.candidate_generation(part.id), 1))
    assert second["draft_revision"] == 2
    catalog.update_part(part.id, drawing_file_hash="drawing-b")
    assert catalog.get_vendor_setup_draft(part.id)["is_stale"] is True
    with pytest.raises(ValueError, match="drawing"):
        catalog.save_vendor_setup_draft(part.id, draft(part, catalog.candidate_generation(part.id), 2))
    catalog.update_part(part.id, drawing_file_hash="drawing-a", revision="B")
    assert catalog.get_vendor_setup_draft(part.id)["is_stale"] is True
    catalog.update_part(part.id, revision="A", cad_file_hash="cad-b")
    assert catalog.get_vendor_setup_draft(part.id)["is_stale"] is True


def test_candidate_generation_and_atomic_replacement(tmp_path):
    catalog = PartCatalogManager(str(tmp_path / "catalog.sqlite"))
    part = catalog.create_part("SYN-002")
    catalog.update_part(part.id, cad_file_hash="cad-a")
    catalog.replace_bend_specs_for_part(part.id, [bend("B1")])
    before = catalog.candidate_generation(part.id)
    saved = catalog.save_vendor_setup_draft(part.id, draft(catalog.get_part(part.id), before))
    assert saved["is_stale"] is False
    with pytest.raises(ValueError, match="invalid coordinates"):
        catalog.replace_bend_specs_for_part(part.id, [bend("B2"), bend("B3", end=(float("nan"), 0, 0))])
    assert [spec.bend_id for spec in catalog.get_bend_specs_for_part(part.id)] == ["B1"]
    assert catalog.candidate_generation(part.id) == before
    catalog.replace_bend_specs_for_part(part.id, [bend("B1")])
    assert catalog.candidate_generation(part.id) != before
    assert catalog.get_vendor_setup_draft(part.id)["is_stale"] is True
    with pytest.raises(ValueError, match="candidates changed"):
        catalog.save_vendor_setup_draft(part.id, draft(catalog.get_part(part.id), before, 1))
    catalog.update_part(part.id, cad_uploaded_at="new-upload-at")
    assert catalog.get_vendor_setup_draft(part.id)["is_stale"] is True
