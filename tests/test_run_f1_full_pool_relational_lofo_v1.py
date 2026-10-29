from __future__ import annotations

from scripts.run_f1_full_pool_relational_lofo_v1 import _duplicate, _select


def _candidate(name,offset):
    return {"bend_id":name,"candidate_id":name,"line_start":[0,offset,0],"line_end":[10,offset,0]}


def test_duplicate_rule_matches_frozen_geometry_contract():
    assert _duplicate(_candidate("a",0),_candidate("b",3.9))
    assert not _duplicate(_candidate("a",0),_candidate("b",4.1))


def test_selection_is_probability_ordered_and_duplicate_suppressed():
    case={"candidates":[_candidate("a",0),_candidate("b",1),_candidate("c",8)],"baseline_by_source":{}}
    selected=_select(case,[0.8,0.9,0.7],0.5)
    assert [row["bend_id"] for row in selected]==["b","c"]
