"""Synthetic source-isolated reference contracts; never import the live service."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "apps/api/services/vendor_setup_service.py"


class ValidationError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@pytest.fixture(scope="module")
def validate():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_validate_draft")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [function],
                        type_ignores=[])
    scope = {"HTTPException": ValidationError, "math": math}
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope["_validate_draft"]


@pytest.fixture(scope="module")
def models():
    # Only Pydantic models are evaluated: no legacy catalog or geometry imports.
    from pydantic import BaseModel, Field
    from typing import Literal, Optional

    names = {"BendMark", "MatchingSearchSettings", "GeneralMeasurement", "VendorSetupDraft"}
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    module = ast.Module(body=classes, type_ignores=[])
    scope = {"BaseModel": BaseModel, "Field": Field, "Literal": Literal, "Optional": Optional}
    exec(compile(module, str(SOURCE), "exec", dont_inherit=True), scope)
    return scope


def reference(**overrides):
    fields = dict(id="reference:1", label="Edge reference", kind="points", bend_ids=[],
                  point_start=[0.0, 0.0, 0.0], point_end=[3.0, 4.0, 0.0],
                  cad_distance=5.0, target_mm=None, note="Operator-authored CAD reference")
    fields.update(overrides)
    return SimpleNamespace(**fields)


def example(measurements=None):
    marks = [SimpleNamespace(feature_id=feature_id, label=label, source="cad_candidate",
                             status="active", review_status="pending", issue=None,
                             corrects_feature_id=None, line_start=[0.0, float(index), 0.0],
                             line_end=[1.0, float(index), 0.0], target_angle=90.0,
                             target_radius=None)
             for index, (feature_id, label) in enumerate((("A", "B1"), ("B", "B2")))]
    draft = SimpleNamespace(marks=marks, sequence=["A", "B"], machine_moves=[],
                            operator_sequence_confirmed=True, completed_steps=0,
                            general_measurements=[] if measurements is None else measurements)
    return draft, {mark.feature_id: mark for mark in marks}


def assert_rejected(validate, measurements, message):
    draft, candidates = example(measurements)
    with pytest.raises(ValidationError, match=message) as error:
        validate(draft, candidates)
    assert error.value.status_code == 422


@pytest.mark.parametrize("kind,bend_ids", [("points", []), ("bend_pair", ["A", "B"])])
@pytest.mark.parametrize("target", [None, 4.75])
def test_reference_modes_and_optional_target_are_accepted(validate, kind, bend_ids, target):
    # Pair anchors deliberately do not equal bend endpoints or midpoints.
    draft, candidates = example([reference(kind=kind, bend_ids=bend_ids, target_mm=target)])
    validate(draft, candidates)


def test_old_source_isolated_draft_without_measurements_still_validates(validate):
    draft, candidates = example()
    del draft.general_measurements
    validate(draft, candidates)


def test_round_off_in_computed_reference_distance_is_accepted(validate):
    draft, candidates = example([reference(point_end=[1.0, 1.0, 0.0],
                                           cad_distance=math.sqrt(2.0) + 1e-12)])
    validate(draft, candidates)


@pytest.mark.parametrize("endpoint", ["point_start", "point_end"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_nonfinite_anchor_coordinates_are_rejected(validate, endpoint, axis, nonfinite):
    coordinates = [0.0, 0.0, 0.0]
    coordinates[axis] = nonfinite
    assert_rejected(validate, [reference(**{endpoint: coordinates})], "finite 3D coordinates")


@pytest.mark.parametrize("bad_anchor", [None, [], [0.0, 0.0], [0.0] * 4, [0.0, "bad", 0.0]])
def test_anchors_need_exactly_three_numeric_values(validate, bad_anchor):
    assert_rejected(validate, [reference(point_start=bad_anchor)], "finite 3D coordinates")


def test_coincident_anchors_are_rejected(validate):
    assert_rejected(validate, [reference(point_end=[0.0, 0.0, 0.0])], "anchors must differ")


@pytest.mark.parametrize("distance", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_invalid_cad_distance_is_rejected(validate, distance):
    assert_rejected(validate, [reference(cad_distance=distance)], "finite and positive")


def test_distance_overflow_with_finite_anchors_is_rejected(validate):
    assert_rejected(validate, [reference(point_start=[-1e308, 0.0, 0.0],
                                        point_end=[1e308, 0.0, 0.0])], "finite and positive")


@pytest.mark.parametrize("distance", [4.0, 5.001, 10.0])
def test_distance_must_match_euclidean_anchors(validate, distance):
    assert_rejected(validate, [reference(cad_distance=distance)], "match the straight-line anchors")


def test_short_distance_mismatch_is_not_hidden_by_absolute_allowance(validate):
    assert_rejected(validate, [reference(point_end=[1e-8, 0.0, 0.0], cad_distance=2e-8)],
                    "match the straight-line anchors")


@pytest.mark.parametrize("target", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_target_must_be_positive_finite_or_null(validate, target):
    assert_rejected(validate, [reference(target_mm=target)], "target must be finite and positive")


def test_measurement_count_boundary(validate):
    measurements = [reference(id=f"reference:{index}") for index in range(100)]
    draft, candidates = example(measurements)
    validate(draft, candidates)
    assert_rejected(validate, measurements + [reference(id="reference:100")], "At most 100")


def test_duplicate_measurement_ids_are_rejected(validate):
    assert_rejected(validate, [reference(), reference()], "Measurement IDs must be unique")


@pytest.mark.parametrize("field,value,message", [
    ("id", "", "Measurement IDs must be nonempty"),
    ("id", "   ", "Measurement IDs must be nonempty"),
    ("id", "a" * 101, "Measurement IDs must be nonempty"),
    ("label", "", "Measurement labels must be nonempty"),
    ("label", "   ", "Measurement labels must be nonempty"),
    ("label", "a" * 101, "Measurement labels must be nonempty"),
    ("note", "a" * 501, "Measurement notes must be at most 500"),
])
def test_measurement_text_boundaries(validate, field, value, message):
    assert_rejected(validate, [reference(**{field: value})], message)


@pytest.mark.parametrize("bend_ids", [[], ["A"], ["A", "A"], ["A", "B", "C"]])
def test_pair_needs_exactly_two_distinct_bend_ids(validate, bend_ids):
    assert_rejected(validate, [reference(kind="bend_pair", bend_ids=bend_ids)], "two distinct bend IDs")


def test_unknown_pair_reference_is_rejected(validate):
    assert_rejected(validate, [reference(kind="bend_pair", bend_ids=["A", "missing"])],
                    "reference active located bends")


@pytest.mark.parametrize("stale_state", ["removed", "unlocated"])
def test_pair_reference_cannot_survive_removed_or_unlocated_bend(validate, stale_state):
    draft, candidates = example([reference(kind="bend_pair", bend_ids=["A", "B"])])
    linked = candidates["B"]
    if stale_state == "removed":
        linked.status = "removed"
        linked.issue = "not_a_bend"
        draft.sequence = ["A"]
    else:
        linked.line_start = linked.line_end = None
    with pytest.raises(ValidationError, match="reference active located bends") as error:
        validate(draft, candidates)
    assert error.value.status_code == 422


def test_points_cannot_carry_bend_references(validate):
    assert_rejected(validate, [reference(bend_ids=["A"])], "cannot reference bend IDs")


def test_unknown_kind_is_rejected(validate):
    assert_rejected(validate, [reference(kind="surface")], "Unknown general measurement kind")


def test_schema_defaults_remain_backward_compatible(models):
    draft = models["VendorSetupDraft"](cad_file_hash="synthetic", candidate_generation="a" * 64)
    assert draft.general_measurements == []
    assert draft.model_dump()["general_measurements"] == []
    measurement = models["GeneralMeasurement"](**vars(reference()))
    assert measurement.model_dump()["target_mm"] is None


@pytest.mark.parametrize("changes", [
    {"id": ""}, {"id": "a" * 101}, {"label": ""}, {"label": "a" * 101},
    {"kind": "surface"}, {"bend_ids": ["A", "B", "C"]},
    {"point_start": [0.0, 0.0]}, {"point_end": [0.0] * 4},
    {"cad_distance": math.nan}, {"cad_distance": math.inf}, {"cad_distance": 0.0},
    {"target_mm": math.nan}, {"target_mm": math.inf}, {"target_mm": 0.0},
    {"note": "a" * 501},
])
def test_schema_rejects_invalid_field_boundaries(models, changes):
    from pydantic import ValidationError as ModelValidationError

    with pytest.raises(ModelValidationError):
        models["GeneralMeasurement"](**vars(reference(**changes)))


def test_schema_rejects_more_than_100_measurements(models):
    from pydantic import ValidationError as ModelValidationError

    with pytest.raises(ModelValidationError):
        models["VendorSetupDraft"](cad_file_hash="synthetic", candidate_generation="a" * 64,
                                   general_measurements=[vars(reference())] * 101)
