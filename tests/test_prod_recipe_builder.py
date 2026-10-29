"""PartRecipeV1 builder (C01 builder.py): hash stability, exact refusal codes, conventions.

Synthetic dicts shaped like backend/tests/test_vendor_setup_catalog.py; no catalog DB, no server.
"""
from __future__ import annotations

import copy
import json
import random

import pytest

from contracts.production.canonical import canonical_json_bytes
from contracts.production.part_recipe import (
    RECIPE_BUILD_REASON_CODES,
    CadRef,
    DrawingRef,
    FlatBindingV1,
    FlatRef,
    PartRecipeV1,
    ReleasedBy,
    ReleaseRequestV1,
)
from domains.production.recipe.bend_identity import bend_uid
from domains.production.recipe.builder import RecipeBuildError, build_part_recipe

CAD_SHA = "a" * 64
DRAWING_SHA = "b" * 64
FLAT_SHA = "e" * 64
GEN = "c" * 64
LINES = [
    ([0.0, 0.0, 0.0], [0.0, 100.0, 0.0]),
    ([20.0, 0.0, 30.0], [20.0, 100.0, 30.0]),
    ([60.0, 0.0, 30.0], [60.0, 100.0, 30.0]),
]


def make_inputs(*, sweeps=(90.0, 90.0, 45.0)):
    specs, marks = [], []
    for i, ((start, end), sweep) in enumerate(zip(LINES, sweeps), start=1):
        fid = f"spec-{i}"
        specs.append({"id": fid, "part_id": "p1", "bend_id": f"B{i}", "target_angle": sweep, "target_radius": 2.0,
                      "tolerance_angle": None, "tolerance_radius": None,
                      "bend_line_start": json.dumps(start), "bend_line_end": json.dumps(end), "notes": None})
        marks.append({"feature_id": fid, "label": f"B{i}", "source": "cad_candidate", "status": "active",
                      "review_status": "confirmed", "issue": None, "corrects_feature_id": None, "note": "",
                      "line_start": start, "line_end": end, "target_angle": sweep, "target_radius": 2.0})
    draft = {
        "schema_version": 1, "cad_file_hash": CAD_SHA, "drawing_file_hash": DRAWING_SHA, "part_revision": "A",
        "candidate_generation": GEN, "expected_draft_revision": 1, "marks": marks,
        "sequence": [m["feature_id"] for m in marks], "machine_moves": [["spec-2", "spec-3"]],
        "machine_moves_explicit": True, "general_measurements": [], "operator_sequence_confirmed": True,
        "completed_steps": 0, "matching_search": {"search_radius_mm": None, "search_angle_deg": None},
        "draft_revision": 2, "is_stale": False, "updated_at": "2026-10-17T05:30:00",
    }
    release = {
        "expected_draft_revision": 2,
        "default_tolerances": {"angle_deg": {"plus": 1.0, "minus": 1.0}, "flange_mm": {"plus": 0.5, "minus": 0.5},
                               "source": "SHOP_STANDARD"},
    }
    return {"part": {"id": "p1", "part_number": "syn 001", "part_name": "Synthetic", "revision": "A"},
            "spec_rows": specs, "draft": draft, "release": release}


def build(inputs=None, **overrides):
    inputs = inputs or make_inputs()
    kwargs = dict(
        part=inputs["part"], spec_rows=inputs["spec_rows"], draft=inputs["draft"], release=inputs["release"],
        cad=CadRef(sha256=CAD_SHA, stored_rel_path=f"recipes/cad/{CAD_SHA}.stp", candidate_generation=GEN,
                   candidate_angle_convention="NATIVE_SWEEP"),
        drawing=DrawingRef(sha256=DRAWING_SHA, stored_rel_path=f"recipes/drawings/{DRAWING_SHA}.pdf"),
        version=1, supersedes_version=None,
        released_by={"user_id": "u1", "username": "admin", "role": "admin"},
        released_at="2026-10-17T06:00:00Z",
    )
    kwargs.update(overrides)
    return build_part_recipe(**kwargs)


def codes_of(**kwargs) -> tuple[str, ...]:
    with pytest.raises(RecipeBuildError) as exc:
        build(**kwargs)
    assert set(exc.value.codes) <= RECIPE_BUILD_REASON_CODES
    return exc.value.codes


def _shuffle(obj, rng):
    if isinstance(obj, dict):
        items = list(obj.items())
        rng.shuffle(items)
        return {k: _shuffle(v, rng) for k, v in items}
    if isinstance(obj, list):
        return [_shuffle(v, rng) for v in obj]
    return obj


def test_basic_recipe_shape():
    recipe = build()
    assert recipe.sku == "SYN-001" and recipe.sku_display == "syn 001" and recipe.recipe_id == "SYN-001@v1"
    assert [b.label for b in recipe.bends] == ["B1", "B2", "B3"]
    assert [b.cad_label for b in recipe.bends] == ["B1", "B2", "B3"]
    assert recipe.bends[0].bend_uid == bend_uid(CAD_SHA, "cad_candidate", LINES[0][0], LINES[0][1], 2.0)
    assert recipe.machine_moves == ((recipe.bends[1].bend_uid, recipe.bends[2].bend_uid),)
    assert [b.machine_move_index for b in recipe.bends] == [None, 1, 1]
    # FINAL is the default; IN_PROCESS is an explicit opt-in.
    assert recipe.inspection_stage == "FINAL" and recipe.flat is None
    assert recipe.source_draft.draft_revision == 2
    assert PartRecipeV1.model_validate(json.loads(canonical_json_bytes(recipe))) == recipe


def test_hashes_are_independent_of_dict_key_order():
    first = build()
    rng = random.Random(7)
    for _ in range(3):
        shuffled = _shuffle(copy.deepcopy(make_inputs()), rng)
        again = build(shuffled)
        assert (again.content_sha256, again.recipe_sha256) == (first.content_sha256, first.recipe_sha256)


def test_release_bookkeeping_changes_only_recipe_sha():
    base = build()
    later = build(released_at="2026-10-18T09:00:00Z")
    v2 = build(version=2, supersedes_version=1)
    other_user = build(released_by=ReleasedBy(user_id="u2", username="eng", role="admin"))
    for other in (later, v2, other_user):
        assert other.content_sha256 == base.content_sha256
        assert other.recipe_sha256 != base.recipe_sha256
    assert v2.recipe_id == "SYN-001@v2"


def test_one_tolerance_change_changes_content_sha():
    inputs = make_inputs()
    inputs["release"]["default_tolerances"]["angle_deg"]["plus"] = 1.5
    assert build(inputs).content_sha256 != build().content_sha256
    inputs = make_inputs()
    inputs["release"]["bend_overrides"] = {"spec-2": {"flange_a_mm": {"plus": 0.3, "minus": 0.3}}}
    recipe = build(inputs)
    assert recipe.content_sha256 != build().content_sha256
    assert recipe.resolved_tolerance(recipe.bends[1].bend_uid, "flange_a_mm").plus == 0.3
    assert recipe.resolved_tolerance(recipe.bends[0].bend_uid, "flange_a_mm").plus == 0.5


def test_native_sweep_convention():
    recipe = build()
    assert [b.nominal_angle.included_deg for b in recipe.bends] == [90.0, 90.0, 135.0]
    assert [b.nominal_angle.sweep_deg for b in recipe.bends] == [90.0, 90.0, 45.0]
    assert {b.nominal_angle.source for b in recipe.bends} == {"CAD_NATIVE_SWEEP"}
    assert not any(b.nominal_angle.convention_verified for b in recipe.bends)


def test_mesh_included_convention():
    recipe = build(cad=CadRef(sha256=CAD_SHA, stored_rel_path=f"recipes/cad/{CAD_SHA}.step",
                              candidate_generation=GEN, candidate_angle_convention="MESH_INCLUDED_FOLDED"))
    assert [b.nominal_angle.included_deg for b in recipe.bends] == [90.0, 90.0, 45.0]
    assert {b.nominal_angle.source for b in recipe.bends} == {"CAD_MESH_FOLDED"}


def test_stale_draft_code():
    inputs = make_inputs()
    inputs["draft"]["is_stale"] = True
    assert codes_of(**inputs) == ("DRAFT_STALE",)
    inputs = make_inputs()
    inputs["spec_rows"][0]["bend_line_end"] = json.dumps([0.0, 99.0, 0.0])  # CAD republished
    assert codes_of(**inputs) == ("DRAFT_STALE",)


def test_unconfirmed_order_code():
    inputs = make_inputs()
    inputs["draft"]["operator_sequence_confirmed"] = False
    assert codes_of(**inputs) == ("SEQUENCE_NOT_CONFIRMED",)


def test_unconfirmed_active_mark_code():
    inputs = make_inputs()
    inputs["draft"]["marks"][1]["review_status"] = "pending"
    with pytest.raises(RecipeBuildError) as exc:
        build(inputs)
    assert exc.value.codes == ("UNCONFIRMED_ACTIVE_MARKS",)
    assert exc.value.details["UNCONFIRMED_ACTIVE_MARKS"] == ("spec-2",)


def test_missing_drawing_code():
    assert codes_of(drawing=None) == ("DRAWING_MISSING",)


def test_missing_and_invalid_tolerance_codes():
    inputs = make_inputs()
    del inputs["release"]["default_tolerances"]
    assert codes_of(**inputs) == ("TOLERANCES_MISSING",)
    inputs = make_inputs()
    inputs["release"]["default_tolerances"]["flange_mm"] = {"plus": 0.0, "minus": 0.0}
    assert codes_of(**inputs) == ("TOLERANCE_INVALID",)


def _with_manual(angle=None):
    inputs = make_inputs()
    inputs["draft"]["marks"].append({
        "feature_id": "manual:1", "label": "M1", "source": "manual", "status": "active",
        "review_status": "confirmed", "issue": "missing", "corrects_feature_id": None, "note": "",
        "line_start": [90.0, 0.0, 0.0], "line_end": [90.0, 100.0, 0.0], "target_angle": None,
        "target_radius": None})
    inputs["draft"]["sequence"].append("manual:1")
    if angle is not None:
        inputs["release"]["bend_overrides"] = {"manual:1": {"manual_included_angle_deg": angle}}
    return inputs


def test_manual_mark_without_angle_code():
    assert codes_of(**_with_manual()) == ("MANUAL_MARK_MISSING_NOMINAL_ANGLE",)


def test_manual_mark_with_angle():
    recipe = build(_with_manual(angle=120.0))
    manual = recipe.bends[-1]
    assert (manual.source, manual.nominal_angle.source, manual.nominal_angle.included_deg) == (
        "manual", "MANUAL_ENTRY", 120.0)
    assert manual.nominal_angle.sweep_deg == 60.0


def test_every_applicable_code_is_collected():
    inputs = _with_manual()
    inputs["draft"]["operator_sequence_confirmed"] = False
    inputs["draft"]["marks"][0]["review_status"] = "pending"
    inputs["draft"]["is_stale"] = True
    del inputs["release"]["default_tolerances"]
    inputs["release"]["bend_overrides"] = {"ghost": {"angle_deg": {"plus": 1, "minus": 1}}}
    assert codes_of(drawing=None, **inputs) == (
        "DRAFT_STALE", "DRAWING_MISSING", "MANUAL_MARK_MISSING_NOMINAL_ANGLE", "OVERRIDE_UNKNOWN_BEND",
        "SEQUENCE_NOT_CONFIRMED", "TOLERANCES_MISSING", "UNCONFIRMED_ACTIVE_MARKS")


def test_missing_draft_still_reports_other_codes():
    inputs = make_inputs()
    inputs["draft"] = None
    assert codes_of(drawing=None, **inputs) == ("DRAFT_MISSING", "DRAWING_MISSING")


def test_removed_marks_go_to_removed_candidates():
    inputs = make_inputs()
    mark = inputs["draft"]["marks"][2]
    mark.update(status="removed", review_status="pending", issue="not_a_bend", note="cut edge")
    inputs["draft"]["sequence"].remove("spec-3")
    inputs["draft"]["machine_moves"] = [["spec-2"]]
    recipe = build(inputs)
    assert [b.source_feature_id for b in recipe.bends] == ["spec-1", "spec-2"]
    assert [(r.feature_id, r.issue, r.note) for r in recipe.removed_candidates] == [("spec-3", "not_a_bend", "cut edge")]
    # A removed mark may stay unconfirmed: it is not an active mark.
    assert recipe.bends[1].machine_move_index == 1


def test_revision_and_binding_codes():
    assert codes_of(release={**make_inputs()["release"], "expected_draft_revision": 1}) == ("DRAFT_REVISION_CHANGED",)
    inputs = make_inputs()
    inputs["draft"]["cad_file_hash"] = "f" * 64
    assert codes_of(**inputs) == ("CAD_HASH_MISMATCH",)
    inputs = make_inputs()
    inputs["draft"]["drawing_file_hash"] = "f" * 64
    assert codes_of(**inputs) == ("DRAWING_HASH_MISMATCH",)
    inputs = make_inputs()
    inputs["draft"]["candidate_generation"] = "9" * 64
    assert codes_of(**inputs) == ("DRAFT_STALE",)
    inputs = make_inputs()
    inputs["part"]["part_number"] = "bad/sku"
    assert codes_of(**inputs) == ("SKU_INVALID",)


def test_sequence_and_moves_are_validated():
    inputs = make_inputs()
    inputs["draft"]["sequence"] = ["spec-1", "spec-2"]
    assert "DRAFT_INVALID" in codes_of(**inputs)
    inputs = make_inputs()
    inputs["draft"]["machine_moves"] = [["spec-1", "spec-3"]]
    assert codes_of(**inputs) == ("DRAFT_INVALID",)
    inputs = make_inputs()
    inputs["draft"]["machine_moves_explicit"] = False  # not operator-authored -> ignored
    recipe = build(inputs)
    assert recipe.machine_moves == () and all(b.machine_move_index is None for b in recipe.bends)
    inputs = make_inputs()
    inputs["draft"]["sequence"] = ["spec-3", "spec-1", "spec-2"]
    inputs["draft"]["machine_moves"] = []
    recipe = build(inputs)
    assert [b.source_feature_id for b in recipe.bends] == ["spec-3", "spec-1", "spec-2"]
    assert [b.label for b in recipe.bends] == ["B1", "B2", "B3"] and recipe.bends[0].cad_label == "B3"


def test_no_active_bends_and_unlocated_mark():
    inputs = make_inputs()
    for mark in inputs["draft"]["marks"]:
        mark.update(status="removed", review_status="pending", issue="not_a_bend")
    inputs["draft"]["sequence"] = []
    inputs["draft"]["machine_moves"] = []
    assert codes_of(**inputs) == ("NO_ACTIVE_BENDS",)
    inputs = _with_manual(angle=90.0)
    inputs["draft"]["marks"][-1]["line_end"] = None
    assert "MARK_WITHOUT_LINE" in codes_of(**inputs)


def test_duplicate_identity_is_refused():
    inputs = _with_manual(angle=90.0)
    inputs["draft"]["marks"][-1].update(line_start=[0.0, 0.0, 0.0], line_end=[0.0, 100.0, 0.0], source="manual")
    inputs["draft"]["marks"].append(dict(inputs["draft"]["marks"][-1], feature_id="manual:2"))
    inputs["draft"]["sequence"].append("manual:2")
    inputs["release"]["bend_overrides"]["manual:2"] = {"manual_included_angle_deg": 90.0}
    assert codes_of(**inputs) == ("DUPLICATE_BEND_IDENTITY",)


def test_general_measurements_are_mapped():
    inputs = make_inputs()
    inputs["draft"]["general_measurements"] = [
        {"id": "gm1", "label": "B1 to B2", "kind": "bend_pair", "bend_ids": ["spec-1", "spec-2"],
         "point_start": [0.0, 50.0, 0.0], "point_end": [20.0, 50.0, 30.0],
         "cad_distance": 36.05551275463989, "target_mm": 36.0, "note": ""},
        {"id": "gm2", "label": "width", "kind": "points", "bend_ids": [],
         "point_start": [0.0, 0.0, 0.0], "point_end": [0.0, 100.0, 0.0], "cad_distance": 100.0, "target_mm": None},
    ]
    inputs["release"]["measurement_overrides"] = {"gm2": {"tolerance": {"plus": 0.2, "minus": 0.2}, "required": False}}
    recipe = build(inputs)
    gm1, gm2 = recipe.general_measurements
    assert gm1.bend_uids == (recipe.bends[0].bend_uid, recipe.bends[1].bend_uid)
    assert (gm1.nominal_mm, gm1.nominal_source) == (36.0, "OPERATOR_TARGET")
    assert (gm2.nominal_mm, gm2.nominal_source, gm2.required, gm2.tolerance.plus) == (100.0, "CAD_DISTANCE", False, 0.2)
    inputs["release"]["measurement_overrides"]["ghost"] = {"required": True}
    assert codes_of(**inputs) == ("OVERRIDE_UNKNOWN_MEASUREMENT",)
    inputs = make_inputs()
    inputs["draft"]["general_measurements"] = [
        {"id": "gm1", "label": "bad", "kind": "bend_pair", "bend_ids": ["spec-1", "nope"],
         "point_start": [0.0, 0.0, 0.0], "point_end": [1.0, 0.0, 0.0], "cad_distance": 1.0}]
    assert codes_of(**inputs) == ("MEASUREMENT_INVALID",)


def test_flat_reference_and_binding():
    recipe = build()
    flat = FlatRef(sha256=FLAT_SHA, stored_rel_path=f"recipes/flats/{FLAT_SHA}.dxf", format="DXF",
                   binding=FlatBindingV1(state="AUTO_PROPOSED", method="GRAPH_MATCH_RIGID_FIT",
                                         bend_line_map={"L1": recipe.bends[0].bend_uid}))
    with_flat = build(flat=flat)
    assert with_flat.flat == flat and with_flat.content_sha256 != recipe.content_sha256
    bad = FlatRef(sha256=FLAT_SHA, stored_rel_path=f"recipes/flats/{FLAT_SHA}.dxf", format="DXF",
                  binding=FlatBindingV1(state="AUTO_PROPOSED", method="MANUAL", bend_line_map={"L1": "bu_" + "0" * 20}))
    assert codes_of(flat=bad) == ("FLAT_BINDING_UNKNOWN_BEND",)
    with pytest.raises(ValueError):
        FlatBindingV1(state="OPERATOR_CONFIRMED", method="MANUAL", bend_line_map={"L1": "bu_" + "0" * 20})


def test_release_request_model_is_accepted_directly():
    inputs = make_inputs()
    inputs["release"] = ReleaseRequestV1.model_validate({**inputs["release"], "inspection_stage": "FINAL",
                                                         "confirmation": "DEV_AUTO_CONFIRMED", "thickness_mm": 2.0})
    recipe = build(inputs)
    assert (recipe.inspection_stage, recipe.confirmation, recipe.thickness_mm) == ("FINAL", "DEV_AUTO_CONFIRMED", 2.0)


def test_unknown_codes_cannot_be_raised():
    with pytest.raises(ValueError):
        RecipeBuildError(["NOT_A_CODE"])
    with pytest.raises(ValueError):
        RecipeBuildError([])


def test_flat_bend_is_refused_at_the_release_gate():
    # A NATIVE_SWEEP of 0 is flat (included 180): refused here, not late in the E1 compile.
    assert codes_of(inputs=make_inputs(sweeps=(90.0, 0.0, 45.0))) == ("DRAFT_INVALID",)
    # MESH_INCLUDED_FOLDED with included 180 is flat too.
    inputs = make_inputs()
    for mark in inputs["draft"]["marks"]:
        mark["target_angle"] = 180.0
    for row in inputs["spec_rows"]:
        row["target_angle"] = 180.0
    mesh_cad = CadRef(sha256=CAD_SHA, stored_rel_path=f"recipes/cad/{CAD_SHA}.stp", candidate_generation=GEN,
                      candidate_angle_convention="MESH_INCLUDED_FOLDED")
    assert codes_of(inputs=inputs, cad=mesh_cad) == ("DRAFT_INVALID",)
    # A manual override of 180 is rejected by the request model itself.
    with pytest.raises(ValueError):
        ReleaseRequestV1.model_validate({**make_inputs()["release"],
                                         "bend_overrides": {"spec-1": {"manual_included_angle_deg": 180.0}}})


def test_release_request_defaults_to_final():
    assert ReleaseRequestV1.model_validate(make_inputs()["release"]).inspection_stage == "FINAL"


def test_manual_flat_angle_is_refused_by_the_builder():
    # 180 = flat: the release request is invalid, so the override is never applied.
    assert codes_of(**_with_manual(angle=180.0)) == ("DRAFT_INVALID", "MANUAL_MARK_MISSING_NOMINAL_ANGLE")
    assert build(_with_manual(angle=179.5)).bends[-1].nominal_angle.included_deg == 179.5
