"""Source-isolated move validation; importing the live catalog service is unsafe here."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "apps/api/services/vendor_setup_service.py"


class ValidationError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def validator():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_draft")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [function], type_ignores=[])
    scope = {"HTTPException": ValidationError, "math": __import__("math")}
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope["_validate_draft"]


def example(moves):
    marks = [SimpleNamespace(feature_id=feature_id, label=label, source="cad_candidate",
                             status="active", review_status="pending", issue=None,
                             corrects_feature_id=None, line_start=[0.0, float(index), 0.0],
                             line_end=[1.0, float(index), 0.0], target_angle=90.0,
                             target_radius=None)
             for index, (feature_id, label) in enumerate((("A", "B1"), ("B", "B2"), ("C", "B3")))]
    draft = SimpleNamespace(marks=marks, sequence=["A", "B", "C"], machine_moves=moves,
                            operator_sequence_confirmed=True, completed_steps=0)
    return draft, {mark.feature_id: mark for mark in marks}


@pytest.mark.parametrize("moves", [[], [["A"]], [["B"]], [["A", "B"]], [["B", "C"]], [["A"], ["C"]]])
def test_machine_moves_are_optional_ordered_contiguous_subsets(moves):
    draft, candidates = example(moves)
    validator()(draft, candidates)


@pytest.mark.parametrize("moves", [[["A", "C"]], [["C"], ["A"]], [["A"], [], ["C"]], [["A", "B"], ["B", "C"]], [["unknown"]]])
def test_machine_moves_cannot_reorder_overlap_be_noncontiguous_or_empty(moves):
    draft, candidates = example(moves)
    with pytest.raises(ValidationError, match="Machine moves must be ordered contiguous bend groups") as error:
        validator()(draft, candidates)
    assert error.value.status_code == 422
