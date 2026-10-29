from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from backend.f1_near_coincident_mesh_duplicate_pruning import (
    prune_near_coincident_mesh_duplicates,
)


ROOT = Path(__file__).resolve().parents[1]


def _line(
    candidate_id: str,
    y: float,
    *,
    surfaces: tuple[int, int] = (1, 2),
    source: str = "scan_mesh_adjacency",
    transition: float = 0.8,
    span: float = 60.0,
):
    return {
        "candidate_id": candidate_id,
        "candidate_source": source,
        "surface1_id": surfaces[0],
        "surface2_id": surfaces[1],
        "line_start": [0.0, y, 0.0],
        "line_end": [span, y, 0.0],
        "two_sided_transition_score": transition,
        "selection_probability": transition,
    }


def test_prunes_boundary_duplicate_and_keeps_stronger_transition() -> None:
    weaker = _line("weaker", 4.0016, surfaces=(1, 3), transition=0.7)
    stronger = _line("stronger", 0.0, surfaces=(1, 2), transition=0.9)

    result = prune_near_coincident_mesh_duplicates([weaker, stronger])

    assert [item["candidate_id"] for item in result["selected"]] == ["stronger"]
    assert result["pruned_candidate_ids"] == ["weaker"]
    assert result["conflicts"][0]["transverse_offset_mm"] < 4.01


def test_preserves_distinct_or_unsupported_parallel_lines() -> None:
    witness = _line("witness", 0.0, surfaces=(1, 2), transition=0.9)
    candidates = [
        _line("outside_margin", 4.02, surfaces=(1, 3)),
        _line("unshared_surface", 4.0, surfaces=(4, 5)),
        _line("other_source", 4.0, surfaces=(1, 3), source="cross_source_line_object"),
        _line("different_span", 4.0, surfaces=(1, 3), span=50.0),
    ]

    for candidate in candidates:
        result = prune_near_coincident_mesh_duplicates([candidate, witness])
        assert result["pruned_candidate_ids"] == []


def test_is_deterministic_order_preserving_and_does_not_mutate_inputs() -> None:
    selected = [
        _line("unrelated", 20.0, surfaces=(8, 9)),
        _line("weaker", 4.0016, surfaces=(1, 3), transition=0.7),
        _line("stronger", 0.0, surfaces=(1, 2), transition=0.9),
    ]
    before = deepcopy(selected)

    first = prune_near_coincident_mesh_duplicates(selected)
    second = prune_near_coincident_mesh_duplicates(selected)

    assert first == second
    assert selected == before
    assert [item["candidate_id"] for item in first["selected"]] == [
        "unrelated",
        "stronger",
    ]


def test_is_inert_on_frozen_gold27_v21_outputs() -> None:
    prediction_root = ROOT / (
        "output/research/f1-gold27-20260803/geometry_gate_v21/cases"
    )
    if not prediction_root.exists():
        return

    for path in sorted(prediction_root.glob("*/prediction_owned_regions.json")):
        selected = json.loads(path.read_text())["owned_bend_regions"]
        result = prune_near_coincident_mesh_duplicates(selected)
        assert result["pruned_candidate_ids"] == []
