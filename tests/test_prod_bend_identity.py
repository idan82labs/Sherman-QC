"""Stable bend identity and remapping (C01 bend_identity.py). Synthetic geometry only."""
from __future__ import annotations

import re

import pytest

from domains.production.recipe.bend_identity import bend_uid, bend_uid_for_mark, remap_bends

CAD = "c" * 64
OTHER_CAD = "d" * 64


def test_uid_format_and_stability():
    uid = bend_uid(CAD, "cad_candidate", (0.0, 0.0, 0.0), (100.0, 0.0, 0.0), 2.0)
    assert re.fullmatch(r"bu_[0-9a-f]{20}", uid)
    assert bend_uid(CAD, "cad_candidate", (100.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2.0) == uid  # endpoint swap
    jitter = bend_uid(CAD, "cad_candidate", (1e-4, -1e-4, 0.0), (100.0001, 0.0, -1e-4), 2.0001)
    assert jitter == uid
    assert bend_uid(CAD, "cad_candidate", (-0.0, 0.0, 0.0), (100.0, 0.0, 0.0), 2.0) == uid  # -0.0 folded


def test_uid_changes_with_identity_inputs():
    base = bend_uid(CAD, "cad_candidate", (0, 0, 0), (100, 0, 0), 2.0)
    assert bend_uid(OTHER_CAD, "cad_candidate", (0, 0, 0), (100, 0, 0), 2.0) != base
    assert bend_uid(CAD, "manual", (0, 0, 0), (100, 0, 0), 2.0) != base
    assert bend_uid(CAD, "cad_candidate", (0, 0, 0), (100, 0, 0), None) != base
    assert bend_uid(CAD, "cad_candidate", (0, 0.002, 0), (100, 0, 0), 2.0) != base


@pytest.mark.parametrize(
    "args",
    [
        ("C" * 64, "cad_candidate", (0, 0, 0), (1, 0, 0), None),
        (CAD, "other", (0, 0, 0), (1, 0, 0), None),
        (CAD, "manual", (0, 0, 0), (0.0001, 0, 0), None),
        (CAD, "manual", (0, 0), (1, 0, 0), None),
        (CAD, "manual", (0, 0, float("nan")), (1, 0, 0), None),
    ],
)
def test_uid_rejects_bad_inputs(args):
    with pytest.raises(ValueError):
        bend_uid(*args)


def test_uid_for_mark_uses_mark_fields():
    mark = {"source": "manual", "line_start": [0, 0, 0], "line_end": [0, 50, 0], "target_radius": None}
    assert bend_uid_for_mark(CAD, mark) == bend_uid(CAD, "manual", (0, 0, 0), (0, 50, 0), None)
    with pytest.raises(ValueError):
        bend_uid_for_mark(CAD, {"source": "manual", "line_start": None, "line_end": None})


def _bends(lines):
    return [
        {"bend_uid": bend_uid(CAD, "cad_candidate", s, e, 2.0), "line_start_mm": s, "line_end_mm": e}
        for s, e in lines
    ]


def _marks(lines, prefix="f"):
    return [{"feature_id": f"{prefix}{i}", "line_start": list(s), "line_end": list(e)} for i, (s, e) in enumerate(lines)]


HAT = [((0, 0, 0), (0, 100, 0)), ((20, 0, 30), (20, 100, 30)), ((60, 0, 30), (60, 100, 30)), ((90, 0, 0), (90, 100, 0))]


def test_retagging_identical_cad_matches_everything():
    result = remap_bends(_bends(HAT), _marks(list(reversed(HAT))))
    assert len(result.matched) == 4 and not result.new and not result.removed and not result.ambiguous
    by_uid = {m.previous_uid: m.feature_id for m in result.matched}
    prev = _bends(HAT)
    assert by_uid[prev[0]["bend_uid"]] == "f3"  # reversed order still maps by geometry
    assert all(m.cost == 0 for m in result.matched)


def test_moved_bend_is_removed_plus_new():
    moved = list(HAT)
    moved[1] = ((25, 0, 30), (25, 100, 30))  # 5 mm away: infeasible
    result = remap_bends(_bends(HAT), _marks(moved))
    assert len(result.matched) == 3
    assert result.removed == (_bends(HAT)[1]["bend_uid"],)
    assert result.new == ("f1",)
    assert not result.ambiguous


def test_small_shift_within_tolerance_still_matches():
    shifted = list(HAT)
    shifted[2] = ((60.3, 0, 30), (60.3, 100.2, 30))
    result = remap_bends(_bends(HAT), _marks(shifted))
    assert len(result.matched) == 4


def test_symmetric_parallel_bends_with_midway_candidate_are_ambiguous_not_assigned():
    prev = _bends([((0, 0, 0), (0, 100, 0)), ((0.6, 0, 0), (0.6, 100, 0))])
    marks = _marks([((0.3, 0, 0), (0.3, 100, 0))])
    result = remap_bends(prev, marks)
    assert result.matched == () and result.new == () and result.removed == ()
    assert len(result.ambiguous) == 1
    group = result.ambiguous[0]
    assert set(group.previous_uids) == {p["bend_uid"] for p in prev} and group.feature_ids == ("f0",)


def test_clear_match_survives_next_to_an_ambiguity():
    prev = _bends([((0, 0, 0), (0, 100, 0)), ((0.6, 0, 0), (0.6, 100, 0)), ((50, 0, 0), (50, 100, 0))])
    marks = _marks([((0.3, 0, 0), (0.3, 100, 0)), ((50, 0, 0), (50, 100, 0))])
    result = remap_bends(prev, marks)
    assert [(m.previous_uid, m.feature_id) for m in result.matched] == [(prev[2]["bend_uid"], "f1")]
    assert len(result.ambiguous) == 1 and result.ambiguous[0].feature_ids == ("f0",)


def test_remap_rejects_bad_input():
    with pytest.raises(ValueError):
        remap_bends(_bends(HAT), _marks(HAT), ambiguity_ratio=0)
    with pytest.raises(ValueError):
        remap_bends([{"bend_uid": "x", "line_start_mm": (0, 0, 0), "line_end_mm": (0, 0, 0)}], [])
    with pytest.raises(ValueError):
        remap_bends(_bends(HAT), [{"line_start": [0, 0, 0], "line_end": [1, 0, 0]}])
