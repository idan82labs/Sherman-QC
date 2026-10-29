"""Production contract checks (C01): examples, honesty validators, pure derivations.

Synthetic in-test data only; never imports the server, the legacy worker or real data.
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.production.canonical import CanonicalJsonError, canonical_json_bytes, sha256_hex, strip_keys
from contracts.production.characteristics import (
    UNKNOWN_REASONS,
    PLAN_UNKNOWN_REASONS,
    exceedance_ratio,
    guard_band_decision,
    make_characteristic_id,
)
from contracts.production.common import Tolerance
from contracts.production.compiled_part import (
    COMPILE_BLOCKER_CODES,
    CompiledPartV1,
    assign_sides,
    check_compiled_against_recipe,
)
from contracts.production.events import EVENT_KINDS, LineEventV1
from contracts.production.examples.generate_examples import EXAMPLE_FILES, build_examples, render
from contracts.production.inspection_result import (
    ShopfloorInspectionResultV1,
    StationFeedV1,
    derive_stage_decision,
    derive_summary,
)
from contracts.production.layout import ARTIFACT_ALLOWLIST, ENGINE_OUTPUT_FILES, workspace_paths
from contracts.production.line_job import FAILURE_CODES, LineJobV1, ScanFileV1
from contracts.production.part_recipe import PartRecipeV1
from contracts.production.schema_export import SCHEMA_MODELS, export_schemas
from contracts.production.vocabulary import DISPLAY_TOKENS, LEGACY_TO_PRODUCTION, to_production_term

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "contracts" / "production" / "examples"
MODEL_BY_EXAMPLE = {
    "part_recipe_v1.example.json": PartRecipeV1,
    "compiled_part_v1.example.json": CompiledPartV1,
    "shopfloor_result_v1.example.json": ShopfloorInspectionResultV1,
    "shopfloor_result_v1.in_process.example.json": ShopfloorInspectionResultV1,
    "shopfloor_result_v1.incomplete.example.json": ShopfloorInspectionResultV1,
    "shopfloor_result_v1.invalid.example.json": ShopfloorInspectionResultV1,
    "station_feed_v1.example.json": StationFeedV1,
    "line_job_v1.example.json": LineJobV1,
}


def load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built():
    return build_examples()


# --------------------------------------------------------------------------
# Examples and schema export
# --------------------------------------------------------------------------


def test_every_example_file_validates_against_its_model():
    on_disk = sorted(p.name for p in EXAMPLES.glob("*.json"))
    assert on_disk == sorted(EXAMPLE_FILES) == sorted(MODEL_BY_EXAMPLE)
    for name, model in MODEL_BY_EXAMPLE.items():
        model.model_validate(load(name))


def test_committed_examples_match_the_generator(built):
    for name, model in built.items():
        assert (EXAMPLES / name).read_text(encoding="utf-8") == render(model), f"{name} drifted; regenerate"


def test_example_summary_states_cover_the_spec(built):
    states = {name: m.summary.state for name, m in built.items() if isinstance(m, ShopfloorInspectionResultV1)}
    assert states == {
        "shopfloor_result_v1.example.json": "OUT_OF_TOLERANCE_FOUND",
        # every measured value passes, but B3/B4 are not formed yet: never green
        "shopfloor_result_v1.in_process.example.json": "STAGE_IN_TOLERANCE",
        "shopfloor_result_v1.incomplete.example.json": "INCOMPLETE",
        "shopfloor_result_v1.invalid.example.json": "INVALID_INSPECTION",
    }
    main = built["shopfloor_result_v1.example.json"]
    decisions = {c.decision for row in main.bends for c in row.characteristics}
    assert {"PASS", "FAIL", "REVIEW", "UNKNOWN"} <= decisions
    reasons = {c.reason for row in main.bends for c in row.characteristics if c.state == "NOT_MEASURED"}
    assert "EDGE_NOT_SEEN" in reasons
    invalid = built["shopfloor_result_v1.invalid.example.json"]
    assert all(c.state == "NOT_MEASURED" for row in invalid.bends for c in row.characteristics)
    in_process = built["shopfloor_result_v1.in_process.example.json"]
    assert [r.state for r in in_process.bends] == ["FORMED", "FORMED", "NOT_YET_FORMED", "NOT_YET_FORMED"]
    assert any(c.kind == "BACKGAUGE_LENGTH" and c.state == "MEASURED" for c in in_process.bends[1].characteristics)


def test_examples_are_cross_consistent(built):
    recipe = built["part_recipe_v1.example.json"]
    compiled = built["compiled_part_v1.example.json"]
    assert check_compiled_against_recipe(compiled, recipe) == []
    result = built["shopfloor_result_v1.example.json"]
    assert result.engine.compiled_part_sha256 == compiled.compiled_part_sha256
    assert result.recipe.recipe_sha256 == recipe.recipe_sha256
    assert [r.bend_uid for r in result.bends] == [b.bend_uid for b in recipe.bends]
    nominal_ids = {c.characteristic_id for c in compiled.characteristics}
    for row in result.bends:
        assert {c.characteristic_id for c in row.characteristics} <= nominal_ids
    assert not compiled.stages[1].symmetry.is_symmetric  # asymmetric hat
    assert compiled.stages[0].symmetry.is_symmetric  # flat blank registers onto itself


def test_schema_export_writes_every_document_schema(tmp_path):
    written = export_schemas(tmp_path / "schemas")
    assert sorted(p.name for p in written) == sorted(f"{n}.schema.json" for n in SCHEMA_MODELS)
    assert len(written) == 9
    for path in written:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["type"] == "object" and "properties" in schema
    recipe_schema = json.loads((tmp_path / "schemas" / "part_recipe_v1.schema.json").read_text())
    assert "schema" in recipe_schema["properties"]


def test_schema_export_refuses_repo_data_dirs():
    for name in ("data", "output", "outputs", "artifacts"):
        with pytest.raises(SystemExit):
            export_schemas(REPO / name / "prod-schemas-should-not-exist")
        assert not (REPO / name / "prod-schemas-should-not-exist").exists()


def test_contract_imports_stay_pure():
    code = (
        "import sys, contracts.production.inspection_result, contracts.production.part_recipe, "
        "contracts.production.compiled_part, domains.production.recipe.builder, domains.production.recipe.sku; "
        "bad=[m for m in sys.modules if m.startswith(('_server_legacy_impl','apps.api','open3d','fastapi','numpy'))]; "
        "assert not bad, bad"
    )
    env = {"PYTHONPATH": f"{REPO}:{REPO / 'backend'}", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------
# Canonical JSON
# --------------------------------------------------------------------------


def test_canonical_json_is_order_independent_and_normalised():
    a = {"b": 1.0000004, "a": [-0.0, 2], "c": {"y": None, "x": "é"}}
    b = {"c": {"x": "é", "y": None}, "a": [0.0, 2], "b": 1.0}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert canonical_json_bytes(a) == '{"a":[0.0,2],"b":1.0,"c":{"x":"é","y":null}}'.encode()
    assert sha256_hex(a) == sha256_hex(b)
    assert sha256_hex({"v": 1.000001}) != sha256_hex({"v": 1.0})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite(bad):
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes({"x": [1, bad]})


def test_canonical_json_rejects_sets_and_non_string_keys():
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes({"x": {1, 2}})
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes({1: "a"})
    assert strip_keys({"a": 1, "b": 2}, ["b", "zz"]) == {"a": 1}


# --------------------------------------------------------------------------
# Guard band
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dev", "u", "plus", "minus", "expected"),
    [
        (0.0, 0.1, 0.5, 0.5, "PASS"),
        (0.4, 0.1, 0.5, 0.5, "PASS"),  # |d| + U == tol is PASS
        (0.45, 0.1, 0.5, 0.5, "REVIEW"),
        (0.6, 0.1, 0.5, 0.5, "REVIEW"),  # |d| - U == tol is not FAIL
        (0.61, 0.1, 0.5, 0.5, "FAIL"),
        (-0.61, 0.1, 0.5, 0.5, "FAIL"),
        (-0.31, 0.1, 0.5, 0.2, "FAIL"),  # asymmetric minus side
        (-0.3, 0.1, 0.5, 0.2, "REVIEW"),  # d + U == -minus is not FAIL
        (0.3, 0.1, 0.5, 0.2, "PASS"),
        (0.05, 0.1, 0.5, 0.0, "REVIEW"),  # unilateral band, U crosses the closed side
    ],
)
def test_guard_band_truth_table(dev, u, plus, minus, expected):
    assert guard_band_decision(dev, u, plus, minus) == expected


def test_guard_band_requires_positive_uncertainty_and_valid_band():
    with pytest.raises(ValueError):
        guard_band_decision(0.0, 0.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        guard_band_decision(0.0, 0.1, 0.0, 0.0)
    assert exceedance_ratio(0.4, 0.1, 0.5, 0.5) == pytest.approx(1.0)
    assert exceedance_ratio(-0.1, 0.1, 0.5, 0.0) is None


def test_tolerance_needs_positive_width():
    with pytest.raises(ValidationError):
        Tolerance(plus=0.0, minus=0.0)
    with pytest.raises(ValidationError):
        Tolerance(plus=-0.1, minus=0.5)
    assert Tolerance(plus=0.5, minus=0.0).plus == 0.5


def test_unknown_reason_vocabulary_contains_the_plan_list():
    assert PLAN_UNKNOWN_REASONS <= UNKNOWN_REASONS
    assert make_characteristic_id("BACKGAUGE_LENGTH", bend_uid="bu_" + "0" * 20, stage_index=2).endswith("/S2")
    with pytest.raises(ValueError):
        make_characteristic_id("FLANGE_LENGTH", bend_uid="bu_" + "0" * 20)


# --------------------------------------------------------------------------
# Result honesty validators (mutate a valid example, expect rejection)
# --------------------------------------------------------------------------


def _bend_chars(doc, row=0):
    return doc["bends"][row]["characteristics"]


def _reject(doc, match=None):
    with pytest.raises(ValidationError, match=match):
        ShopfloorInspectionResultV1.model_validate(doc)


def test_all_measured_in_tolerance_is_rejected_while_a_required_value_is_unknown():
    doc = load("shopfloor_result_v1.incomplete.example.json")
    ShopfloorInspectionResultV1.model_validate(doc)
    doc["summary"]["state"] = "ALL_MEASURED_IN_TOLERANCE"
    _reject(doc, "derive_summary")


def test_pass_or_fail_is_rejected_when_validity_is_not_valid():
    doc = load("shopfloor_result_v1.example.json")
    doc["inspection_validity"]["state"] = "UNTRUSTED_REGISTRATION"
    doc["inspection_validity"]["reasons"] = ["DATUM_FITNESS_BELOW_POLICY"]
    _reject(doc, "invalid inspection")


def test_measured_characteristic_is_rejected_in_an_invalid_result():
    doc = load("shopfloor_result_v1.invalid.example.json")
    valid = load("shopfloor_result_v1.example.json")
    doc["bends"][0]["characteristics"][2] = valid["bends"][0]["characteristics"][2]
    _reject(doc)


def test_not_measured_with_an_actual_value_is_rejected():
    doc = load("shopfloor_result_v1.example.json")
    char = _bend_chars(doc, 3)[0]
    assert char["state"] == "NOT_MEASURED"
    char["actual"] = char["expected"]  # nominal copied into actual
    _reject(doc, "must not carry actual")


def test_not_measured_without_reason_is_rejected():
    doc = load("shopfloor_result_v1.example.json")
    _bend_chars(doc, 3)[0]["reason"] = None
    _reject(doc, "UNKNOWN reason")


def test_decision_must_follow_the_guard_band():
    doc = load("shopfloor_result_v1.example.json")
    angle = _bend_chars(doc, 1)[2]
    assert angle["decision"] == "FAIL"
    angle["decision"] = "PASS"
    _reject(doc, "guard band")
    doc = load("shopfloor_result_v1.example.json")
    review = _bend_chars(doc, 2)[0]
    assert review["decision"] == "REVIEW"
    review["decision"] = "PASS"
    _reject(doc, "guard band")


def test_measured_needs_positive_uncertainty_and_consistent_deviation():
    doc = load("shopfloor_result_v1.example.json")
    _bend_chars(doc)[0]["uncertainty"] = 0.0
    _reject(doc)
    doc = load("shopfloor_result_v1.example.json")
    _bend_chars(doc)[0]["deviation"] += 0.3
    _reject(doc, "deviation")


def test_required_angle_cannot_be_hidden_from_the_summary():
    doc = load("shopfloor_result_v1.example.json")
    _bend_chars(doc, 1)[2]["required"] = False
    _reject(doc, "required")


def test_not_applicable_is_limited_by_kind():
    doc = load("shopfloor_result_v1.example.json")
    angle = _bend_chars(doc)[2]
    for key in ("actual", "deviation", "uncertainty", "stations"):
        angle[key] = None
    angle.update(state="NOT_APPLICABLE", decision="NOT_APPLICABLE", reason="NOT_INSPECTED_BY_RECIPE", required=False)
    _reject(doc, "cannot be NOT_APPLICABLE")


def test_authority_literals_are_fixed():
    doc = load("shopfloor_result_v1.example.json")
    doc["authority"]["mode"] = "AUTONOMOUS"
    _reject(doc)
    doc = load("shopfloor_result_v1.example.json")
    doc["authority"]["production_disposition"] = "AUTHORIZED"
    _reject(doc)
    doc = load("shopfloor_result_v1.example.json")
    doc["inspection_validity"]["metrology_qualified"] = True
    _reject(doc)


def test_rows_must_be_sorted_and_unique():
    doc = load("shopfloor_result_v1.example.json")
    doc["bends"][0], doc["bends"][1] = doc["bends"][1], doc["bends"][0]
    _reject(doc, "sorted")
    doc = load("shopfloor_result_v1.example.json")
    doc["bends"][1]["sequence_index"] = 1
    _reject(doc)


def test_not_yet_formed_bend_is_never_a_defect():
    doc = load("shopfloor_result_v1.in_process.example.json")
    measured = load("shopfloor_result_v1.in_process.example.json")["bends"][0]["characteristics"][2]
    row = doc["bends"][2]
    assert row["state"] == "NOT_YET_FORMED"
    fake = copy.deepcopy(measured)
    fake.update(characteristic_id=row["characteristics"][2]["characteristic_id"], bend_uid=row["bend_uid"])
    fake.update(actual=180.0, deviation=90.0, decision="FAIL")
    row["characteristics"][2] = fake
    _reject(doc, "NOT_YET_FORMED")


def test_bend_not_yet_formed_reason_only_on_unformed_bends():
    doc = load("shopfloor_result_v1.example.json")
    char = _bend_chars(doc, 3)[0]
    char.update(state="NOT_APPLICABLE", decision="NOT_APPLICABLE", reason="BEND_NOT_YET_FORMED", required=False)
    _reject(doc, "BEND_NOT_YET_FORMED")


def test_unknown_bend_cannot_carry_measurements():
    doc = load("shopfloor_result_v1.example.json")
    doc["bends"][0]["state"] = "UNKNOWN"
    doc["bends"][0]["state_reasons"] = ["OCCLUDED"]
    _reject(doc, "UNKNOWN bend")


def test_measured_polyline_exists_exactly_when_position_is_measured():
    doc = load("shopfloor_result_v1.example.json")
    doc["bends"][0]["display"]["measured_polyline_mm"] = None
    _reject(doc, "measured_polyline_mm")
    doc = load("shopfloor_result_v1.in_process.example.json")
    doc["bends"][2]["display"]["measured_polyline_mm"] = doc["bends"][2]["display"]["nominal_polyline_mm"]
    _reject(doc, "measured_polyline_mm")


def test_stage_decision_must_equal_the_derivation():
    doc = load("shopfloor_result_v1.in_process.example.json")
    doc["stage_decision"]["status"] = "STAGE_MISMATCH"
    _reject(doc, "stage_decision")
    doc = load("shopfloor_result_v1.in_process.example.json")
    doc["stage_decision"]["declared_stage"] = 3  # declared != inferred -> derivation says STAGE_MISMATCH
    _reject(doc, "stage_decision")


def test_ambiguous_pose_measures_nothing():
    doc = load("shopfloor_result_v1.example.json")
    doc["inspection_validity"]["registration"]["pose_ambiguous"] = True
    _reject(doc, "ambiguous pose")


def test_degenerate_in_plane_pose_withholds_position_and_heatmap():
    doc = load("shopfloor_result_v1.example.json")
    doc["inspection_validity"]["registration"]["in_plane_degenerate"] = True
    _reject(doc, "degenerate")


def test_invalid_result_publishes_no_heatmap():
    doc = load("shopfloor_result_v1.invalid.example.json")
    doc["viewer_assets"].update(registered_scan="registered_scan.ply", point_count=10, deviations="deviations.f32",
                                deviation_reference_stage="S4", heatmap_unavailable_reason=None)
    _reject(doc, "heatmap")


def test_unplanned_crease_forces_review(built):
    main = built["shopfloor_result_v1.in_process.example.json"]
    doc = main.model_dump(mode="json")
    doc["unplanned_creases"] = [{"crease_id": "X1", "reason": "EXTRA_BEND_SUSPECTED",
                                 "polyline_mm": [[0, 0, 0], [0, 10, 0]]}]
    _reject(doc, "derive_summary")
    doc["summary"]["state"] = "REVIEW_REQUIRED"
    doc["summary"]["unplanned_creases"] = 1
    doc["summary"]["reasons"] = ["UNPLANNED_CREASES"]
    assert ShopfloorInspectionResultV1.model_validate(doc).summary.state == "REVIEW_REQUIRED"


# --------------------------------------------------------------------------
# derive_summary / derive_stage_decision truth tables
# --------------------------------------------------------------------------


def test_derive_summary_truth_table(built):
    main = built["shopfloor_result_v1.example.json"]
    in_process = built["shopfloor_result_v1.in_process.example.json"]
    incomplete = built["shopfloor_result_v1.incomplete.example.json"]
    assert derive_summary(validity_state="UNTRUSTED_REGISTRATION", bends=main.bends).state == "INVALID_INSPECTION"
    assert derive_summary(validity_state="VALID", bends=main.bends).state == "OUT_OF_TOLERANCE_FOUND"
    assert derive_summary(validity_state="VALID", bends=in_process.bends).state == "STAGE_IN_TOLERANCE"
    assert "PART_NOT_COMPLETE" in derive_summary(validity_state="VALID", bends=in_process.bends).reasons
    assert derive_summary(validity_state="VALID", bends=in_process.bends, stage_decision="REVIEW",
                          stage_status="STAGE_MISMATCH").state == "REVIEW_REQUIRED"
    assert derive_summary(validity_state="VALID", bends=in_process.bends, unplanned_creases=1).state == "REVIEW_REQUIRED"
    assert derive_summary(validity_state="VALID", bends=incomplete.bends[:3]).state == "ALL_MEASURED_IN_TOLERANCE"
    assert derive_summary(validity_state="VALID", bends=incomplete.bends).state == "INCOMPLETE"
    assert derive_summary(validity_state="VALID", bends=in_process.bends, stage_decision="UNKNOWN").state == "INCOMPLETE"
    # REVIEW without FAIL: drop the FAIL bend (B2) from the main result
    no_fail = [b for b in main.bends if b.label != "B2"]
    assert derive_summary(validity_state="VALID", bends=no_fail).state == "REVIEW_REQUIRED"
    worst = derive_summary(validity_state="VALID", bends=main.bends).worst
    assert worst is not None and worst.kind == "INCLUDED_ANGLE" and worst.decision == "FAIL"


class _R:
    def __init__(self, uid, seq, state, move=None):
        self.bend_uid, self.sequence_index, self.state, self.machine_move_index = uid, seq, state, move


def _stage(states, moves=None, *, stage="IN_PROCESS", declared=None, pose=False, validity="VALID"):
    moves = moves or [None] * len(states)
    rows = [_R(f"b{i}", i + 1, s, m) for i, (s, m) in enumerate(zip(states, moves))]
    return derive_stage_decision(inspection_stage=stage, declared_stage=declared, bends=rows,
                                 pose_ambiguous=pose, validity_state=validity)


def test_stage_decision_prefix_is_consistent():
    d = _stage(["FORMED", "FORMED", "NOT_YET_FORMED", "NOT_YET_FORMED"])
    assert (d["status"], d["decision"], d["inferred_stage"], d["part_complete"]) == ("CONSISTENT", "PASS", 2, False)
    assert _stage(["NOT_YET_FORMED"] * 3)["inferred_stage"] == 0  # blank = stage 0


def test_stage_decision_allows_free_order_inside_a_machine_move():
    states = ["FORMED", "NOT_YET_FORMED", "FORMED", "NOT_YET_FORMED"]
    assert _stage(states, [None, 1, 1, None])["status"] == "CONSISTENT"
    assert _stage(states)["status"] == "OUT_OF_ORDER_SUSPECTED"


def test_stage_decision_flags():
    assert _stage(["FORMED", "UNEXPECTED_FORMED"])["status"] == "OUT_OF_ORDER_SUSPECTED"
    assert _stage(["FORMED", "NOT_YET_FORMED"], declared=2)["status"] == "STAGE_MISMATCH"
    assert _stage(["FORMED", "NOT_YET_FORMED"], declared=1)["status"] == "CONSISTENT"
    final = _stage(["FORMED", "NOT_YET_FORMED"], stage="FINAL")
    assert (final["status"], final["decision"]) == ("FINAL_INCOMPLETE", "REVIEW")
    unknown = _stage(["FORMED", "UNKNOWN"], declared=2)
    assert (unknown["status"], unknown["decision"], unknown["inferred_stage"]) == ("UNDETERMINED", "UNKNOWN", None)
    partial = _stage(["PARTIALLY_FORMED", "NOT_YET_FORMED"], declared=0)
    assert partial["status"] == "PARTIAL_BEND_FOUND" and "STAGE_MISMATCH" in partial["flags"]
    assert _stage(["UNKNOWN"], pose=True)["status"] == "AMBIGUOUS_POSE"
    assert _stage(["UNKNOWN"], validity="TIMEOUT")["status"] == "NOT_EVALUATED"


# --------------------------------------------------------------------------
# Recipe, compiled part, jobs, events, layout, vocabulary
# --------------------------------------------------------------------------


def test_recipe_hash_tampering_is_rejected():
    doc = load("part_recipe_v1.example.json")
    PartRecipeV1.model_validate(doc)
    doc["bends"][0]["nominal_angle"]["included_deg"] = 91.0
    doc["bends"][0]["nominal_angle"]["sweep_deg"] = 89.0
    with pytest.raises(ValidationError, match="content_sha256"):
        PartRecipeV1.model_validate(doc)
    doc = load("part_recipe_v1.example.json")
    doc["released_at"] = "2026-10-18T06:00:00Z"
    with pytest.raises(ValidationError, match="recipe_sha256"):
        PartRecipeV1.model_validate(doc)


def test_recipe_structural_rules():
    doc = load("part_recipe_v1.example.json")
    fields = {k: v for k, v in doc.items() if k not in ("content_sha256", "recipe_sha256")}
    PartRecipeV1.finalize(**fields)
    bad = copy.deepcopy(fields)
    bad["recipe_id"] = "SYN-HAT-1@v2"
    with pytest.raises(ValidationError, match="recipe_id"):
        PartRecipeV1.finalize(**bad)
    bad = copy.deepcopy(fields)
    bad["machine_moves"] = [[bad["bends"][1]["bend_uid"], bad["bends"][3]["bend_uid"]]]
    with pytest.raises(ValidationError, match="contiguous"):
        PartRecipeV1.finalize(**bad)
    bad = copy.deepcopy(fields)
    bad["bends"][2]["machine_move_index"] = None
    with pytest.raises(ValidationError, match="machine_move_index"):
        PartRecipeV1.finalize(**bad)
    bad = copy.deepcopy(fields)
    bad["bends"][1]["sequence_index"] = 5
    with pytest.raises(ValidationError, match="sequence_index"):
        PartRecipeV1.finalize(**bad)
    bad = copy.deepcopy(fields)
    bad["cad"]["stored_rel_path"] = "recipes/cad/other.step"
    with pytest.raises(ValidationError):
        PartRecipeV1.finalize(**bad)
    bad = copy.deepcopy(fields)
    bad["released_at"] = "2026-10-17T08:00:00+02:00"
    with pytest.raises(ValidationError, match="UTC"):
        PartRecipeV1.finalize(**bad)


def test_compiled_part_rules(built):
    doc = load("compiled_part_v1.example.json")
    CompiledPartV1.model_validate(doc)
    tampered = copy.deepcopy(doc)
    tampered["characteristics"][0]["nominal"] += 1.0
    with pytest.raises(ValidationError, match="compiled_part_sha256"):
        CompiledPartV1.model_validate(tampered)
    fields = {k: v for k, v in doc.items() if k != "compiled_part_sha256"}
    blocked = copy.deepcopy(fields)
    blocked["status"] = "BLOCKED"
    with pytest.raises(ValidationError, match="BLOCKER"):
        CompiledPartV1.finalize(**blocked)
    blocked["issues"].append({"code": "UNMARKED_BEND", "severity": "BLOCKER", "graph_bend_id": "G9"})
    part = CompiledPartV1.finalize(**blocked)
    assert part.blocker_codes == ("UNMARKED_BEND",)
    wrong_severity = copy.deepcopy(fields)
    wrong_severity["issues"].append({"code": "MARK_NOT_IN_FLANGE_GRAPH", "severity": "WARNING",
                                     "bend_uid": fields["bends"][0]["bend_uid"]})
    with pytest.raises(ValidationError, match="severity"):
        CompiledPartV1.finalize(**wrong_severity)
    cyclic = copy.deepcopy(fields)
    cyclic["bends"][0]["child_panel_id"] = "P2"  # datum cannot be a child
    with pytest.raises(ValidationError):
        CompiledPartV1.finalize(**cyclic)
    swapped = copy.deepcopy(fields)
    swapped["bends"][0]["sides"] = [dict(s) for s in reversed(swapped["bends"][0]["sides"])]
    swapped["bends"][0]["sides"][0]["side"], swapped["bends"][0]["sides"][1]["side"] = "A", "B"
    with pytest.raises(ValidationError, match="naming rule"):
        CompiledPartV1.finalize(**swapped)
    missing_stage = copy.deepcopy(fields)
    missing_stage["stages"] = missing_stage["stages"][:-1]
    with pytest.raises(ValidationError, match="prefix stage"):
        CompiledPartV1.finalize(**missing_stage)
    lying_symmetry = copy.deepcopy(fields)
    lying_symmetry["stages"][2]["symmetry"]["is_symmetric"] = True
    with pytest.raises(ValidationError, match="is_symmetric"):
        CompiledPartV1.finalize(**lying_symmetry)
    assert "MARK_NOT_IN_FLANGE_GRAPH" in COMPILE_BLOCKER_CODES and "UNMARKED_BEND" in COMPILE_BLOCKER_CODES


def test_compiled_part_must_match_its_recipe(built):
    recipe = built["part_recipe_v1.example.json"]
    compiled = built["compiled_part_v1.example.json"]
    data = compiled.model_dump(mode="json")
    data["recipe"]["recipe_sha256"] = "f" * 64
    data.pop("compiled_part_sha256")
    other = CompiledPartV1.finalize(**data)
    assert "RECIPE_REF_MISMATCH" in check_compiled_against_recipe(other, recipe)


def test_side_naming_rule():
    assert assign_sides((1, 0, 0), (0, 0, 1)) == (0, 1)
    assert assign_sides((0, 0, 1), (1, 0, 0)) == (1, 0)
    assert assign_sides((0, 1, 0), (0, -1, 0)) == (0, 1)  # tie on X broken by +Y
    assert assign_sides((0, 0, -1), (0, 0, 1)) == (1, 0)  # tie on X, Y broken by +Z
    with pytest.raises(ValueError):
        assign_sides((1, 0, 0), (1, 0, 0))


def test_line_job_and_scan_file_rules():
    doc = load("line_job_v1.example.json")
    LineJobV1.model_validate(doc)
    bad = dict(doc, execution_status="failed")
    with pytest.raises(ValidationError, match="failure"):
        LineJobV1.model_validate(bad)
    bad = dict(doc, workspace_rel="jobs/other")
    with pytest.raises(ValidationError, match="workspace_rel"):
        LineJobV1.model_validate(bad)
    sha = "a" * 64
    scan = {
        "scan_file_id": "scan_" + sha[:24], "sha256": sha, "original_name": "X_A.ply", "size_bytes": 10,
        "station_id": "cell-1", "received_at": "2026-10-17T07:00:00Z",
        "stored_rel_path": f"archive/scans/2026/10/{sha}.ply",
        "parse": {"status": "REVISION_MISMATCH", "sku": "X", "revision": "A", "expected_revision": "B",
                  "revision_status": "MISMATCH"},
        "match": {"status": "REVISION_MISMATCH", "reason": "file revision A, recipe B"},
        "ingest_state": "quarantined", "quarantine_reason": "REVISION_MISMATCH",
    }
    ScanFileV1.model_validate(scan)
    with pytest.raises(ValidationError, match="quarantine_reason"):
        ScanFileV1.model_validate(dict(scan, quarantine_reason=None))
    with pytest.raises(ValidationError, match="scan_file_id"):
        ScanFileV1.model_validate(dict(scan, scan_file_id="scan_" + "b" * 24))
    with pytest.raises(ValidationError, match="MATCHED"):
        ScanFileV1.model_validate(dict(scan, ingest_state="enqueued", quarantine_reason=None))
    assert {"WORKER_TIMEOUT", "COMPILED_PART_HASH_MISMATCH", "ENGINE_OUTPUT_INVALID"} <= FAILURE_CODES


def test_events_and_layout():
    LineEventV1(id=1, ts="2026-10-17T07:00:00Z", kind="RECIPE_COMPILE_BLOCKED", payload={"codes": ["UNMARKED_BEND"]})
    with pytest.raises(ValidationError):
        LineEventV1(id=1, ts="2026-10-17T07:00:00Z", kind="NOT_A_KIND")
    assert "JOB_PROGRESS" in EVENT_KINDS
    job_id = "line_" + "0" * 32
    paths = workspace_paths("/data/jobs", job_id)
    assert str(paths.engine_file("result")) == f"/data/jobs/{job_id}/engine/result.json"
    assert str(paths.scan(".PLY")) == f"/data/jobs/{job_id}/input/scan.ply"
    assert str(paths.artifact("report.pdf")) == f"/data/jobs/{job_id}/result/report.pdf"
    with pytest.raises(KeyError):
        paths.artifact("recipe.json")
    with pytest.raises(ValueError):
        workspace_paths("/data/jobs", "../etc")
    assert ENGINE_OUTPUT_FILES["deviations"] == "deviations.f32"
    assert all(v.startswith("result/") for v in ARTIFACT_ALLOWLIST.values())


def test_vocabulary_map_and_review_is_amber():
    assert LEGACY_TO_PRODUCTION["IN"] == "PASS" and LEGACY_TO_PRODUCTION["OUT"] == "FAIL"
    assert LEGACY_TO_PRODUCTION["NOT_MEASURED"] == "UNKNOWN"
    assert LEGACY_TO_PRODUCTION["UNFORMED"] == "NOT_YET_FORMED"
    assert DISPLAY_TOKENS["REVIEW"]["color"] == "#F59E0B" and DISPLAY_TOKENS["REVIEW"]["tone"] == "review"
    for term in ("UNKNOWN", "NOT_YET_FORMED", "NOT_APPLICABLE"):
        assert DISPLAY_TOKENS[term]["color"] not in (DISPLAY_TOKENS["PASS"]["color"], DISPLAY_TOKENS["FAIL"]["color"])
    assert to_production_term("PASS") == "PASS"
    with pytest.raises(ValueError):
        to_production_term("WARNING")
    with pytest.raises(TypeError):
        LEGACY_TO_PRODUCTION["IN"] = "FAIL"  # type: ignore[index]


# --------------------------------------------------------------------------
# Canonical round trip and producer factories
# --------------------------------------------------------------------------


def test_every_example_survives_a_canonical_json_round_trip(built):
    for name, model in built.items():
        again = MODEL_BY_EXAMPLE[name].model_validate(json.loads(canonical_json_bytes(model)))
        assert canonical_json_bytes(again) == canonical_json_bytes(model), name


def test_factory_rounds_before_deciding_so_boundaries_round_trip(built):
    from contracts.production.inspection_result import CharacteristicResultV1

    compiled = built["compiled_part_v1.example.json"]
    nom = next(c for c in compiled.characteristics if c.kind == "FLANGE_LENGTH")
    tol = nom.tolerance.plus
    # d + U lands within 1e-7 of the limit: decided on the rounded values, stable after canonical JSON.
    row = CharacteristicResultV1.measured(nom, actual=nom.nominal + tol - 0.1 + 4e-7, uncertainty=0.1000001,
                                          expected_source="COMPILED_NOMINAL", required=True)
    again = CharacteristicResultV1.model_validate(json.loads(canonical_json_bytes(row)))
    assert again == row and row.decision in ("PASS", "REVIEW")
    assert row.deviation == round(row.actual - row.expected, 6)


def test_factory_unknown_without_nominal_or_tolerance(built):
    from contracts.production.inspection_result import CharacteristicResultV1

    compiled = built["compiled_part_v1.example.json"]
    pos = next(c for c in compiled.characteristics if c.kind == "BEND_LINE_POSITION")
    no_tol = pos.model_copy(update={"tolerance": None})
    row = CharacteristicResultV1.measured(no_tol, actual=0.2, uncertainty=0.1,
                                          expected_source="NOMINAL_LOCUS_REFERENCE", required=False)
    assert (row.decision, row.reason, row.required) == ("UNKNOWN", "NO_TOLERANCE", False)
    with pytest.raises(ValidationError, match="required exactly when"):
        CharacteristicResultV1.measured(no_tol, actual=0.2, uncertainty=0.1,
                                        expected_source="NOMINAL_LOCUS_REFERENCE", required=True)
    no_nominal = no_tol.model_copy(update={"nominal": None, "unavailable_reason": "NOMINAL_UNAVAILABLE"})
    row = CharacteristicResultV1.measured(no_nominal, actual=0.2, uncertainty=0.1,
                                          expected_source="NOMINAL_LOCUS_REFERENCE", required=False)
    assert (row.decision, row.reason, row.expected, row.deviation) == ("UNKNOWN", "NOMINAL_UNAVAILABLE", None, None)
    with pytest.raises(ValidationError):
        CharacteristicResultV1.measured(no_tol, actual=0.2, uncertainty=1e-8,
                                        expected_source="NOMINAL_LOCUS_REFERENCE", required=False)


# --------------------------------------------------------------------------
# Review fixes: summary honesty, rigid registration, stage-model ids,
# degenerate pose, cross-document consistency
# --------------------------------------------------------------------------

from contracts.production import consistency  # noqa: E402
from contracts.production.examples import generate_examples as gen  # noqa: E402
from contracts.production.inspection_result import (  # noqa: E402
    CharacteristicResultV1,
    InspectionValidityV1,
    RegistrationV1,
    ViewerAssetsV1,
    check_rigid_transform,
)
from contracts.production.part_recipe import FlangeInspection, FlatBindingV1, FlatRef  # noqa: E402
from contracts.production.states import SUMMARY_STATES  # noqa: E402

check_result = consistency.check_result_against_compiled


@pytest.fixture(scope="module")
def ctx(built):
    return built["part_recipe_v1.example.json"], built["compiled_part_v1.example.json"]


def _nyf_rows(compiled, labels, stage_id):
    return [
        gen._row(compiled, label, "NOT_YET_FORMED",
                 [gen._char(compiled, i, state="NOT_APPLICABLE", reason="BEND_NOT_YET_FORMED")
                  for i in gen._ids(compiled, label)], measured_included=179.8, stage_id=stage_id)
        for label in labels
    ]


def _gm_na(compiled, recipe):
    return gen._gm_row(compiled, recipe, state="NOT_APPLICABLE", reason="NOT_APPLICABLE_AT_STAGE")


def _final_pass_rows(built):
    """B1..B3 FORMED and all PASS (the incomplete example's first three rows)."""
    rows = list(built["shopfloor_result_v1.incomplete.example.json"].bends[:3])
    assert all(c.decision == "PASS" for r in rows for c in r.characteristics)
    return rows


def test_nothing_measured_is_never_in_tolerance(ctx):
    """Every bend NOT_YET_FORMED and nothing measured -> INCOMPLETE, never green."""
    recipe, compiled = ctx
    rows = _nyf_rows(compiled, gen._LABELS, "S0")
    result = gen._result(recipe, compiled, rows=rows, stage_model_id="S0", gms=[_gm_na(compiled, recipe)])
    assert result.stage_decision.status == "CONSISTENT" and result.stage_decision.inferred_stage == 0
    assert result.summary.characteristics_required == 0
    assert (result.summary.state, result.summary.reasons) == ("INCOMPLETE", ("NO_REQUIRED_MEASURED",))
    doc = result.model_dump(mode="json")
    doc["summary"].update(state="ALL_MEASURED_IN_TOLERANCE", reasons=[])
    _reject(doc, "derive_summary")
    assert derive_summary(validity_state="VALID", bends=rows).state == "INCOMPLETE"


@pytest.mark.parametrize(("stage", "expected"), [("FINAL", "REVIEW_REQUIRED"), ("IN_PROCESS", "STAGE_IN_TOLERANCE")])
def test_missing_last_bend_is_never_all_measured_in_tolerance(built, ctx, stage, expected):
    """Amendment 5: a finished-looking part missing its last bend is never green."""
    recipe, compiled = ctx
    rows = _final_pass_rows(built) + _nyf_rows(compiled, ["B4"], "S3")
    staged = recipe.model_copy(update={"inspection_stage": stage})
    result = gen._result(staged, compiled, rows=rows, stage_model_id="S3", gms=[_gm_na(compiled, recipe)])
    assert result.summary.state == expected
    assert result.summary.required_fail == result.summary.required_review == result.summary.required_unknown == 0
    if stage == "FINAL":
        assert result.stage_decision.status == "FINAL_INCOMPLETE"
    doc = result.model_dump(mode="json")
    doc["summary"].update(state="ALL_MEASURED_IN_TOLERANCE", reasons=[])
    _reject(doc, "derive_summary")
    # The result belongs to a recipe with the other inspection_stage: the cross-check sees it.
    assert ("INSPECTION_STAGE_MISMATCH" in check_result(result, compiled, recipe)) == (stage != recipe.inspection_stage)


def test_all_measured_in_tolerance_needs_every_bend_formed(built):
    rows = _final_pass_rows(built)
    assert derive_summary(validity_state="VALID", bends=rows).state == "ALL_MEASURED_IN_TOLERANCE"
    nyf = built["shopfloor_result_v1.in_process.example.json"].bends[3]
    assert derive_summary(validity_state="VALID", bends=rows + [nyf]).state == "STAGE_IN_TOLERANCE"
    assert "STAGE_IN_TOLERANCE" in SUMMARY_STATES


def _registration(doc):
    return doc["inspection_validity"]["registration"]


@pytest.mark.parametrize("mutation", ["scale", "mirror", "shear", "last_row"])
def test_raw_to_frame_must_be_a_proper_rigid_transform(mutation):
    doc = load("shopfloor_result_v1.example.json")
    m = [list(row) for row in _registration(doc)["raw_to_frame"]]
    if mutation == "scale":
        m = [[v * 1.02 if j < 3 and i < 3 else v for j, v in enumerate(row)] for i, row in enumerate(m)]
    elif mutation == "mirror":
        m[2][2] = -m[2][2]
    elif mutation == "shear":
        m[0][2] = 0.01
    else:
        m[3] = [0.0, 0.0, 0.001, 1.0]
    _registration(doc)["raw_to_frame"] = m
    _reject(doc, "raw_to_frame")


def test_rounded_rotation_is_still_rigid():
    c, s = round(math.cos(0.3), 6), round(math.sin(0.3), 6)
    check_rigid_transform(((c, -s, 0.0, 1.0), (s, c, 0.0, 2.0), (0.0, 0.0, 1.0, 3.0), (0.0, 0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="rigid"):
        check_rigid_transform(((c * 1.001, -s, 0.0, 1.0), (s, c, 0.0, 2.0), (0.0, 0.0, 1.0, 3.0), (0.0, 0.0, 0.0, 1.0)))


def test_stage_model_id_must_match_the_decode():
    doc = load("shopfloor_result_v1.example.json")
    assert doc["stage_decision"]["stage_model_id"] == "S4" and doc["stage_decision"]["inferred_stage"] == 4
    doc["stage_decision"]["stage_model_id"] = "S0"
    _reject(doc, "stage_model_id")
    doc = load("shopfloor_result_v1.example.json")
    doc["stage_decision"]["stage_model_id"] = "S4-swap"  # tagged model on an order-consistent decode
    _reject(doc, "prefix stage model")
    doc = load("shopfloor_result_v1.example.json")
    doc["viewer_assets"]["deviation_reference_stage"] = "S3"
    _reject(doc, "heatmap reference stage")
    doc = load("shopfloor_result_v1.example.json")
    doc["stage_decision"]["stage_model_id"] = 4  # an int cannot name an out-of-order model
    _reject(doc)
    doc = load("shopfloor_result_v1.invalid.example.json")
    doc["stage_decision"]["stage_model_id"] = "S4"
    _reject(doc, "decodes no stage model")


def _degenerate_result(ctx, position_reason):
    recipe, compiled = ctx
    nom = {c.characteristic_id: c.nominal for c in compiled.characteristics}
    rows = []
    for label in gen._LABELS:
        ids = gen._ids(compiled, label)
        rows.append(gen._row(compiled, label, "FORMED", [
            gen._char(compiled, ids[0], actual=nom[ids[0]] + 0.05, u=0.09),
            gen._char(compiled, ids[1], actual=nom[ids[1]] - 0.05, u=0.09),
            gen._char(compiled, ids[2], actual=90.1, u=0.15),
            gen._char(compiled, ids[3], state="NOT_MEASURED", reason=position_reason)], measured_included=90.1))
    validity = InspectionValidityV1(
        state="VALID",
        registration=RegistrationV1(method="DATUM_V1", raw_to_frame=gen.RAW_TO_FRAME, in_plane_degenerate=True))
    viewer = ViewerAssetsV1(registered_scan="registered_scan.ply", point_count=10, heatmap_band_mm=0.5,
                            heatmap_unavailable_reason="REGISTRATION_UNTRUSTED")
    return gen._result(recipe, compiled, rows=rows, validity=validity, viewer=viewer,
                       gms=[gen._gm_row(compiled, recipe, actual=86.1, u=0.15)])


def test_degenerate_in_plane_pose_reports_position_as_registration_untrusted(ctx):
    result = _degenerate_result(ctx, "REGISTRATION_UNTRUSTED")
    assert all(r.characteristics[2].state == "MEASURED" for r in result.bends)  # angle does not need the pose
    assert result.summary.state == "INCOMPLETE"  # required positions are UNKNOWN
    with pytest.raises(ValidationError, match="REGISTRATION_UNTRUSTED"):
        _degenerate_result(ctx, "OCCLUDED")


# -- mutants that survived the first review --------------------------------


def test_invalid_result_cannot_carry_a_measured_general_measurement():
    doc = load("shopfloor_result_v1.invalid.example.json")
    doc["general_measurements"][0] = load("shopfloor_result_v1.example.json")["general_measurements"][0]
    assert doc["general_measurements"][0]["characteristic"]["state"] == "MEASURED"
    _reject(doc, "invalid inspection has no MEASURED")


def test_partially_formed_bend_cannot_pass_its_angle():
    doc = load("shopfloor_result_v1.example.json")
    assert doc["bends"][0]["characteristics"][2]["decision"] == "PASS"
    doc["bends"][0]["state"] = "PARTIALLY_FORMED"
    _reject(doc, "PARTIALLY_FORMED")


def test_not_applicable_characteristic_is_never_required():
    doc = load("shopfloor_result_v1.in_process.example.json")
    char = doc["bends"][2]["characteristics"][0]
    assert char["state"] == "NOT_APPLICABLE"
    char["required"] = True
    _reject(doc, "never required")


def test_flat_sha_mismatch_is_detected(ctx):
    recipe, compiled = ctx
    sha = "e" * 64
    flat = FlatRef(sha256=sha, stored_rel_path=f"recipes/flats/{sha}.dxf", format="DXF",
                   binding=FlatBindingV1(state="UNBOUND"))
    with_flat = recipe.model_copy(update={"flat": flat})
    assert "FLAT_SHA_MISMATCH" in check_compiled_against_recipe(compiled, with_flat)
    assert check_compiled_against_recipe(compiled, recipe) == []


# -- compiled nominals vs recipe policy ------------------------------------


def _refinalize(compiled, mutate):
    data = compiled.model_dump(mode="json")
    data.pop("compiled_part_sha256")
    mutate(data)
    return CompiledPartV1.finalize(**data)


def test_compiled_tolerance_and_requirement_must_follow_the_recipe(ctx):
    recipe, compiled = ctx

    def widen(data):
        data["characteristics"][0]["tolerance"]["plus"] = 5.0

    assert check_compiled_against_recipe(_refinalize(compiled, widen), recipe) == ["TOLERANCE_MISMATCH"]
    bend = recipe.bends[0].model_copy(update={"flange_inspection": FlangeInspection(side_a="NOT_INSPECTED")})
    not_inspected = recipe.model_copy(update={"bends": (bend, *recipe.bends[1:])})
    assert check_compiled_against_recipe(compiled, not_inspected) == ["REQUIREMENT_MISMATCH"]


@pytest.mark.parametrize(
    ("index", "update", "match"),
    [
        (0, {"tolerance": None}, "REQUIRED FLANGE_LENGTH nominal carries"),  # FL A of B1
        (2, {"recipe_requirement": "NOT_INSPECTED"}, "recipe_requirement must be"),  # angle of B1
        (3, {"tolerance": None}, "REQUIRED exactly when"),  # position of B1
    ],
)
def test_nominal_requirement_vocabulary(index, update, match):
    doc = load("compiled_part_v1.example.json")
    doc["characteristics"][index].update(update)
    with pytest.raises(ValidationError, match=match):
        CompiledPartV1.model_validate(doc, context={"_skip_hash_check": True})


def test_compiled_part_size_caps_keep_segment_labels_unique():
    doc = load("compiled_part_v1.example.json")
    doc["panels"] = doc["panels"] * 200  # 1000 panels
    with pytest.raises(ValidationError, match="at most 999"):
        CompiledPartV1.model_validate(doc, context={"_skip_hash_check": True})


# -- result vs CompiledPart vs recipe --------------------------------------


def test_examples_pass_the_cross_document_check(built, ctx):
    recipe, compiled = ctx
    for name, model in built.items():
        if isinstance(model, ShopfloorInspectionResultV1):
            assert check_result(model, compiled, recipe) == [], name
    assert set(check_result(built["shopfloor_result_v1.example.json"], compiled, recipe)) <= (
        consistency.RESULT_CONSISTENCY_CODES)


def test_cross_check_catches_a_dropped_bend_and_measurement(built, ctx):
    recipe, compiled = ctx
    main = built["shopfloor_result_v1.example.json"]
    # The result validator alone cannot know the recipe had a general measurement.
    no_gm = gen._result(recipe, compiled, rows=list(main.bends), gms=[])
    assert check_result(no_gm, compiled, recipe) == ["GENERAL_MEASUREMENT_MISMATCH"]
    # A relabelled bend, and (structure only, bypassing the model) a dropped bend.
    relabelled = main.bends[3].model_copy(update={"label": "B9"})
    assert check_result(main.model_copy(update={"bends": (*main.bends[:3], relabelled)}), compiled, recipe) == [
        "BEND_INVENTORY_MISMATCH"]
    assert "BEND_INVENTORY_MISMATCH" in check_result(main.model_copy(update={"bends": main.bends[:3]}), compiled, recipe)


def test_cross_check_catches_a_hidden_required_flange(built, ctx):
    recipe, compiled = ctx
    main = built["shopfloor_result_v1.example.json"]
    b2 = main.bends[1]
    nom = compiled.nominal(b2.characteristics[0].characteristic_id)
    hidden = CharacteristicResultV1.not_applicable(nom, reason="NOT_INSPECTED_BY_RECIPE",
                                                   expected_source="COMPILED_NOMINAL")
    row = b2.model_copy(update={"characteristics": (hidden, *b2.characteristics[1:])})
    rows = [main.bends[0], row, *main.bends[2:]]
    result = gen._result(recipe, compiled, rows=rows, gms=list(main.general_measurements))
    assert check_result(result, compiled, recipe) == ["REQUIREMENT_MISMATCH"]


@pytest.mark.parametrize(
    ("update", "code"),
    [({"nominal": 31.0}, "EXPECTED_MISMATCH"),
     ({"tolerance": {"plus": 3.0, "minus": 3.0, "origin": "RECIPE_DEFAULT", "standard": "SHOP_STANDARD"}},
      "TOLERANCE_MISMATCH")],
)
def test_cross_check_catches_a_drifted_nominal_or_tolerance(built, ctx, update, code):
    recipe, compiled = ctx
    main = built["shopfloor_result_v1.example.json"]
    b1 = main.bends[0]
    nom = compiled.nominal(b1.characteristics[0].characteristic_id)
    fake_nominal = nom.model_validate({**nom.model_dump(mode="json"), **update})
    drifted = CharacteristicResultV1.measured(fake_nominal, actual=nom.nominal + 0.08, uncertainty=0.09,
                                              expected_source="COMPILED_NOMINAL")
    row = b1.model_copy(update={"characteristics": (drifted, *b1.characteristics[1:])})
    result = gen._result(recipe, compiled, rows=[row, *main.bends[1:]], gms=list(main.general_measurements))
    assert check_result(result, compiled, recipe) == [code]


def test_cross_check_requires_the_mid_sequence_backgauge(built, ctx):
    recipe, compiled = ctx
    ip = built["shopfloor_result_v1.in_process.example.json"]
    b2 = ip.bends[1]
    assert b2.characteristics[4].kind == "BACKGAUGE_LENGTH"
    no_bg = b2.model_copy(update={"characteristics": b2.characteristics[:4]})
    result = gen._result(recipe, compiled, rows=[ip.bends[0], no_bg, *ip.bends[2:]], declared=2,
                         stage_model_id="S2", gms=list(ip.general_measurements))
    assert check_result(result, compiled, recipe) == ["BACKGAUGE_MISSING"]


def test_cross_check_catches_a_wrong_stage_model_and_references(built, ctx):
    recipe, compiled = ctx
    inc = built["shopfloor_result_v1.incomplete.example.json"]
    assert inc.stage_decision.inferred_stage is None  # B4 UNKNOWN: any k is allowed by the model
    wrong = gen._result(recipe, compiled, rows=list(inc.bends), stage_model_id="S2",
                        gms=list(inc.general_measurements))
    assert check_result(wrong, compiled, recipe) == ["STAGE_MODEL_MISMATCH"]  # S2 leaves formed B3 unformed
    other = _refinalize(compiled, lambda d: d["diagnostics"].update(x=1) or d["panels"][0].update(area_mm2=1801.0))
    assert "COMPILED_PART_REF_MISMATCH" in check_result(inc, other, recipe)


def test_inspect_job_pins_its_compiled_part():
    doc = load("line_job_v1.example.json")
    doc["compiled_part_sha256"] = None
    with pytest.raises(ValidationError, match="compiled_part_sha256"):
        LineJobV1.model_validate(doc)


def test_display_tokens_cover_summary_states_and_label_unknown_neutrally():
    assert SUMMARY_STATES <= set(DISPLAY_TOKENS)
    green = DISPLAY_TOKENS["PASS"]["color"]
    assert [s for s in SUMMARY_STATES if DISPLAY_TOKENS[s]["color"] == green] == ["ALL_MEASURED_IN_TOLERANCE"]
    assert DISPLAY_TOKENS["STAGE_IN_TOLERANCE"]["tone"] not in ("pass", "fail")
    assert DISPLAY_TOKENS["UNKNOWN"]["en"] == "Undetermined"


# -- second review: stage applicability, pose ambiguity, summary precedence --


def _complete_rows(compiled):
    """B1..B4 FORMED, every characteristic MEASURED and PASS (a finished part)."""
    nom = {c.characteristic_id: c.nominal for c in compiled.characteristics}
    rows = []
    for label in gen._LABELS:
        ids = gen._ids(compiled, label)
        rows.append(gen._row(compiled, label, "FORMED", [
            gen._char(compiled, ids[0], actual=nom[ids[0]] + 0.05, u=0.09),
            gen._char(compiled, ids[1], actual=nom[ids[1]] - 0.05, u=0.09),
            gen._char(compiled, ids[2], actual=90.1, u=0.15),
            gen._char(compiled, ids[3], actual=0.04, u=0.12)], measured_included=90.1))
    return rows


def _gm_char(compiled, recipe):
    gm = recipe.general_measurements[0]
    return compiled.nominal(make_characteristic_id("GENERAL_MEASUREMENT", measurement_id=gm.measurement_id)), gm


def _gm_row_from(gm, char, *, measured):
    from contracts.production.inspection_result import GeneralMeasurementRowV1

    return GeneralMeasurementRowV1(
        measurement_id=gm.measurement_id, label=gm.label, characteristic=char,
        nominal_start_mm=gm.point_start_mm, nominal_end_mm=gm.point_end_mm,
        measured_start_mm=gm.point_start_mm if measured else None,
        measured_end_mm=gm.point_end_mm if measured else None)


def test_finished_part_cannot_skip_a_required_general_measurement(ctx):
    recipe, compiled = ctx
    rows = _complete_rows(compiled)
    ok = gen._result(recipe, compiled, rows=rows, gms=[gen._gm_row(compiled, recipe, actual=86.1, u=0.15)])
    assert ok.summary.state == "ALL_MEASURED_IN_TOLERANCE"
    assert ok.general_measurements[0].characteristic.required
    assert check_result(ok, compiled, recipe) == []
    # The reviewer's repro: every bend formed and PASS, the required gm "not applicable at this stage".
    with pytest.raises(ValidationError, match="NOT_APPLICABLE_AT_STAGE needs a NOT_YET_FORMED bend"):
        gen._result(recipe, compiled, rows=rows, gms=[_gm_na(compiled, recipe)])
    # Same for a backgauge on a finished part.
    nominal = next(c for c in compiled.characteristics if c.kind == "BACKGAUGE_LENGTH")
    row_i = next(i for i, r in enumerate(rows) if r.bend_uid == nominal.bend_uid)
    bg = CharacteristicResultV1.not_applicable(nominal, reason="NOT_APPLICABLE_AT_STAGE",
                                               expected_source="STAGE_MODEL_NOMINAL")
    rows_bg = list(rows)
    rows_bg[row_i] = rows[row_i].model_copy(update={"characteristics": (*rows[row_i].characteristics, bg)})
    doc = ok.model_dump(mode="json")
    doc["bends"][row_i] = rows_bg[row_i].model_dump(mode="json")
    _reject(doc, "NOT_APPLICABLE_AT_STAGE needs a NOT_YET_FORMED bend")


def _with_gm(recipe, **update):
    gm = recipe.general_measurements[0].model_copy(update=update)
    return recipe.model_copy(update={"general_measurements": (gm,)})


def test_cross_check_limits_not_applicable_at_stage_to_unformed_bends(built, ctx):
    recipe, compiled = ctx
    ip = built["shopfloor_result_v1.in_process.example.json"]  # B1, B2 formed; B3, B4 not yet
    assert ip.general_measurements[0].characteristic.reason == "NOT_APPLICABLE_AT_STAGE"
    uid = {r.label: r.bend_uid for r in ip.bends}
    assert check_result(ip, compiled, recipe) == []  # `points` with some bend unformed
    formed_pair = _with_gm(recipe, kind="bend_pair", bend_uids=(uid["B1"], uid["B2"]))
    assert check_result(ip, compiled, formed_pair) == ["STAGE_APPLICABILITY_MISMATCH"]
    open_pair = _with_gm(recipe, kind="bend_pair", bend_uids=(uid["B2"], uid["B3"]))
    assert check_result(ip, compiled, open_pair) == []


def test_cross_check_rejects_a_skipped_backgauge_on_a_prefix_decode(built, ctx):
    recipe, compiled = ctx
    ip = built["shopfloor_result_v1.in_process.example.json"]
    b2 = ip.bends[1]
    assert ip.stage_decision.stage_model_id == "S2" and b2.characteristics[4].state == "MEASURED"
    skipped = CharacteristicResultV1.not_applicable(compiled.nominal(b2.characteristics[4].characteristic_id),
                                                    reason="NOT_APPLICABLE_AT_STAGE",
                                                    expected_source="STAGE_MODEL_NOMINAL")
    row = b2.model_copy(update={"characteristics": (*b2.characteristics[:4], skipped)})
    result = gen._result(recipe, compiled, rows=[ip.bends[0], row, *ip.bends[2:]], declared=2,
                         stage_model_id="S2", gms=list(ip.general_measurements))
    assert check_result(result, compiled, recipe) == ["STAGE_APPLICABILITY_MISMATCH"]


def test_required_general_measurement_reported_optional_is_caught(ctx):
    """Mutant (a): only the cross-check's required_for guard catches this."""
    recipe, compiled = ctx
    nom, gm = _gm_char(compiled, recipe)
    assert CharacteristicResultV1.required_for(nom)
    fail = CharacteristicResultV1.measured(nom, actual=nom.nominal + 5.0, uncertainty=0.15,
                                           expected_source="CAD_DISTANCE", required=False)
    assert fail.decision == "FAIL"
    result = gen._result(recipe, compiled, rows=_complete_rows(compiled),
                         gms=[_gm_row_from(gm, fail, measured=True)])
    # Hidden from the state, but never silent: the summary says an optional value failed ...
    assert result.summary.state == "ALL_MEASURED_IN_TOLERANCE"
    assert result.summary.reasons == ("OPTIONAL_FAIL",)
    # ... and the cross-check knows the recipe requires it.
    assert check_result(result, compiled, recipe) == ["REQUIREMENT_MISMATCH"]
    doc = result.model_dump(mode="json")
    doc["summary"]["reasons"] = []
    _reject(doc, "derive_summary")


def test_optional_review_is_named_in_the_summary(ctx):
    recipe, compiled = ctx
    optional = _with_gm(recipe, required=False)
    nom, gm = _gm_char(compiled, recipe)
    tol = nom.tolerance
    review = CharacteristicResultV1.measured(nom, actual=nom.nominal + tol.plus, uncertainty=0.15,
                                             expected_source="CAD_DISTANCE", required=False)
    assert review.decision == "REVIEW"
    result = gen._result(optional, compiled, rows=_complete_rows(compiled),
                         gms=[_gm_row_from(gm, review, measured=True)])
    assert (result.summary.state, result.summary.reasons) == ("ALL_MEASURED_IN_TOLERANCE", ("OPTIONAL_REVIEW",))


def test_required_general_measurement_needs_a_tolerance():
    doc = load("shopfloor_result_v1.incomplete.example.json")
    char = doc["general_measurements"][0]["characteristic"]
    assert char["state"] == "NOT_MEASURED" and char["required"]
    char["tolerance"] = None
    _reject(doc, "required GENERAL_MEASUREMENT is toleranced")


def _ambiguous_result(ctx, *, gm_measured=False, heatmap=False):
    recipe, compiled = ctx
    rows = [gen._row(compiled, label, "UNKNOWN",
                     [gen._char(compiled, i, state="NOT_MEASURED", reason="AMBIGUOUS_POSE")
                      for i in gen._ids(compiled, label)], reasons=("AMBIGUOUS_POSE",))
            for label in gen._LABELS]
    validity = InspectionValidityV1(
        state="VALID", registration=RegistrationV1(method="DATUM_V1", raw_to_frame=gen.RAW_TO_FRAME,
                                                   pose_ambiguous=True))
    if heatmap:
        viewer = ViewerAssetsV1(registered_scan="registered_scan.ply", point_count=10, deviations="deviations.f32",
                                deviation_reference_stage="S4", heatmap_band_mm=0.5)
    else:
        viewer = ViewerAssetsV1(registered_scan="registered_scan.ply", point_count=10, heatmap_band_mm=0.5,
                                heatmap_unavailable_reason="AMBIGUOUS_POSE")
    gm = (gen._gm_row(compiled, recipe, actual=86.1, u=0.15) if gm_measured
          else gen._gm_row(compiled, recipe, state="NOT_MEASURED", reason="AMBIGUOUS_POSE"))
    return gen._result(recipe, compiled, rows=rows, validity=validity, viewer=viewer,
                       stage_model_id="S4" if heatmap else None, gms=[gm])


def test_ambiguous_pose_cannot_measure_a_general_measurement(ctx):
    """Mutant (b): bends are all UNKNOWN, so only the 'measures nothing' rule can fire."""
    ok = _ambiguous_result(ctx)
    assert (ok.stage_decision.status, ok.summary.state) == ("AMBIGUOUS_POSE", "INCOMPLETE")
    with pytest.raises(ValidationError, match="an ambiguous pose measures nothing"):
        _ambiguous_result(ctx, gm_measured=True)


def test_ambiguous_pose_publishes_no_heatmap(ctx):
    with pytest.raises(ValidationError, match="an ambiguous pose publishes no heatmap"):
        _ambiguous_result(ctx, heatmap=True)


def test_invalid_result_heatmap_rule_fires_on_its_own():
    """Mutant (c): stage_model_id and the heatmap reference agree, so only the heatmap rule fires."""
    doc = load("shopfloor_result_v1.invalid.example.json")
    doc["stage_decision"]["stage_model_id"] = "S4"
    doc["viewer_assets"].update(registered_scan="registered_scan.ply", point_count=10, deviations="deviations.f32",
                                deviation_reference_stage="S4", heatmap_unavailable_reason=None)
    _reject(doc, "an invalid inspection publishes no heatmap")


def test_degenerate_pose_heatmap_rule_fires_on_its_own(ctx):
    """Mutant (d): positions are already UNKNOWN(REGISTRATION_UNTRUSTED), only the heatmap is wrong."""
    doc = _degenerate_result(ctx, "REGISTRATION_UNTRUSTED").model_dump(mode="json")
    doc["viewer_assets"].update(deviations="deviations.f32", deviation_reference_stage="S4",
                                heatmap_unavailable_reason=None)
    _reject(doc, "a degenerate in-plane pose publishes no heatmap")


def test_non_valid_inspection_must_state_reasons():
    """Mutant (e)."""
    doc = load("shopfloor_result_v1.invalid.example.json")
    doc["inspection_validity"]["reasons"] = []
    _reject(doc, "must state reasons")


def test_review_evidence_outranks_an_undetermined_stage():
    partial = _stage(["PARTIALLY_FORMED", "FORMED", "UNKNOWN"])
    assert (partial["status"], partial["decision"]) == ("PARTIAL_BEND_FOUND", "REVIEW")
    assert partial["flags"] == ("PARTIAL_BEND_FOUND", "UNDETERMINED") and partial["inferred_stage"] is None
    final = _stage(["FORMED", "NOT_YET_FORMED", "UNKNOWN"], stage="FINAL")
    assert final["flags"] == ("FINAL_INCOMPLETE", "UNDETERMINED")  # FINAL scan missing B2, B3 unknown
    assert final["decision"] == "REVIEW"
    assert _stage(["FORMED", "UNKNOWN"])["status"] == "UNDETERMINED"  # nothing to review: still UNKNOWN


def test_partial_bend_next_to_an_unknown_bend_needs_review(ctx):
    """Reviewer repro: B1 PARTIALLY_FORMED (angle NOT_MEASURED) and B4 UNKNOWN was INCOMPLETE."""
    recipe, compiled = ctx
    rows = _complete_rows(compiled)
    b1 = gen._ids(compiled, "B1")
    rows[0] = gen._row(compiled, "B1", "PARTIALLY_FORMED", [
        *rows[0].characteristics[:2], gen._char(compiled, b1[2], state="NOT_MEASURED", reason="OCCLUDED"),
        rows[0].characteristics[3]], measured_included=120.0)
    rows[3] = gen._row(compiled, "B4", "UNKNOWN", [
        gen._char(compiled, i, state="NOT_MEASURED", reason="OCCLUDED") for i in gen._ids(compiled, "B4")],
        reasons=("OCCLUDED",))
    viewer = ViewerAssetsV1(registered_scan="registered_scan.ply", point_count=10, heatmap_band_mm=0.5,
                            heatmap_unavailable_reason="STAGE_MODEL_UNDETERMINED")
    result = gen._result(recipe, compiled, rows=rows, stage_model_id=None, viewer=viewer,
                         gms=[gen._gm_row(compiled, recipe, actual=86.1, u=0.15)])
    assert result.stage_decision.flags == ("PARTIAL_BEND_FOUND", "UNDETERMINED")
    assert result.summary.state == "REVIEW_REQUIRED"
    assert result.summary.reasons == ("REQUIRED_UNKNOWN", "STAGE:PARTIAL_BEND_FOUND")
