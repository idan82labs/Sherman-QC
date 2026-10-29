from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_f1_explicit_parent_ownership_lofo_v1.py"
SPEC = importlib.util.spec_from_file_location("ownership_lofo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _line(identifier, y, ownership):
    return {
        "bend_id": identifier,
        "candidate_id": identifier,
        "line_start": [0.0, y, 0.0],
        "line_end": [20.0, y, 0.0],
        "source_hypothesis_id": identifier,
        "selection_route": "test",
        "ownership": ownership,
    }


def test_additive_selection_never_deletes_parent_and_claims_one_per_key(monkeypatch):
    parent = _line("parent", 0.0, None)
    candidates = [_line("parent", 0.0, None), _line("a", 10.0, None), _line("b", 20.0, None)]
    case = {
        "baseline_predictions": [parent],
        "baseline_by_source": {"parent": parent},
        "candidates": candidates,
        "ownership": [
            {"keys": ["FLANGE:F1|F2"], "unique_key": "FLANGE:F1|F2", "group_size": 1},
            {"keys": ["FLANGE:F3|F4"], "unique_key": "FLANGE:F3|F4", "group_size": 2},
            {"keys": ["FLANGE:F3|F4"], "unique_key": "FLANGE:F3|F4", "group_size": 2},
        ],
    }
    selected = MODULE._select(case, np.asarray([0.99, 0.90, 0.80]), 0.5)
    assert [row["bend_id"] for row in selected] == ["a", "parent"]
    assert any(row["bend_id"] == "parent" for row in selected)


def test_ambiguous_or_unowned_candidates_abstain():
    case = {
        "baseline_predictions": [], "baseline_by_source": {},
        "candidates": [_line("a", 0.0, None), _line("b", 10.0, None)],
        "ownership": [
            {"keys": [], "unique_key": None, "group_size": 0},
            {"keys": ["PLANE:A|B", "SURFACE:A|B"], "unique_key": None, "group_size": 0},
        ],
    }
    assert MODULE._select(case, np.asarray([0.99, 0.99]), 0.5) == []

