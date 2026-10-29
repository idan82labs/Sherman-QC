"""Scoring definition V2 (owner decision 2026-10-19, option 1): withheld bend-line position is n/a and the bend's
identity is checked by its mapped CAD nominal line. Report-only scoring code: scripts/production/per_bend_scoring_v2.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from domains.production.engine.eval.line_metric import match_lines

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("per_bend_scoring_v2", _REPO / "scripts/production/per_bend_scoring_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


def _ch(kind, state="MEASURED", decision="PASS", reason=None, side=None):
    return {"kind": kind, "state": state, "decision": decision, "reason": reason, "side": side}


def _bend(uid, *, withheld=False, published=True, pos_reason=None):
    pos = (_ch("BEND_LINE_POSITION", "NOT_MEASURED", "UNKNOWN", "REGISTRATION_UNTRUSTED") if withheld else
           _ch("BEND_LINE_POSITION", "MEASURED" if published else "NOT_MEASURED", "PASS" if published else "UNKNOWN",
               None if published else pos_reason))
    return {"bend_uid": uid, "label": uid, "state": "FORMED",
            "characteristics": [_ch("FLANGE_LENGTH", side="A"), _ch("FLANGE_LENGTH", side="B"), _ch("INCLUDED_ANGLE"), pos]}


# Two parallel gold lines 40 mm apart (raw frame, 100 mm long).
GOLD = [{"bend_id": "G1", "line_start": [0, 0, 0], "line_end": [100, 0, 0]},
        {"bend_id": "G2", "line_start": [0, 40, 0], "line_end": [100, 40, 0]}]


def _score(result, published_preds, nominal):
    """Run the V2 pipeline exactly as the runners do: published lines matched first (frozen matcher), then the
    withheld bends' mapped nominal lines against the remaining gold."""
    m = match_lines(GOLD, published_preds)
    pub_ids = {x["prediction_id"] for x in m["matches"]}
    gold_used = {x["gold_id"] for x in m["matches"]}
    published = {p["bend_id"] for p in published_preds}
    withheld = [b["bend_uid"] for b in result["bends"] if b["bend_uid"] not in published and v2.position_withheld(b)]
    ident = v2.identity_by_nominal(withheld, nominal, GOLD, gold_used, match_lines)

    def status_of(uid):
        return v2.line_status(uid, criterion_applied=True, published=uid in published, published_match_ids=pub_ids,
                              withheld=uid in ident, identity_ok=(ident.get(uid) or {}).get("identity_ok"))

    rows = v2.per_bend_rows("c", "f", result, lambda uid: "FORMED", status_of, "TEST")
    return {r["bend_uid"]: r for r in rows}, ident


def test_withheld_position_with_correct_identity_is_not_a_false_pass():
    res = {"bends": [_bend("B1", withheld=True)]}
    nominal = {"B1": ([1.0, 1.5, 0.0], [99.0, 1.2, 0.3])}  # 1.5 mm off G1: same bend
    rows, ident = _score(res, [], nominal)
    assert ident["B1"]["identity_ok"] and ident["B1"]["gold_id"] == "G1"
    r = rows["B1"]
    assert r["line_status"] == v2.NA_WITHHELD_IDENTITY_OK and r["line_ok"] is None
    assert r["false_pass_proxy"] is False and r["correct_riser_aware"] is True


def test_withheld_position_with_wrong_identity_is_a_false_pass():
    res = {"bends": [_bend("B1", withheld=True)]}
    nominal = {"B1": ([0.0, 20.0, 0.0], [100.0, 20.0, 0.0])}  # half-way between G1 and G2: matches no gold line
    rows, ident = _score(res, [], nominal)
    assert ident["B1"]["identity_ok"] is False
    r = rows["B1"]
    assert r["line_status"] == v2.WITHHELD_IDENTITY_WRONG and r["line_ok"] is False
    assert r["false_pass_proxy"] is True and r["correct_riser_aware"] is False


def test_withheld_without_nominal_line_falls_back_to_wrong():
    rows, ident = _score({"bends": [_bend("B1", withheld=True)]}, [], {})
    assert ident["B1"]["why"] == "NO_NOMINAL_LINE"
    assert rows["B1"]["false_pass_proxy"] is True


def test_identity_cannot_reuse_a_gold_line_claimed_by_a_published_line():
    # B2 publishes a line on G1; withheld B1's nominal line also sits on G1 -> G1 is taken, G2 is 40 mm away -> wrong.
    res = {"bends": [_bend("B1", withheld=True), _bend("B2")]}
    pub = [{"bend_id": "B2", "line_start": [0, 0.5, 0], "line_end": [100, 0.5, 0]}]
    rows, _ = _score(res, pub, {"B1": ([0, 1.0, 0], [100, 1.0, 0])})
    assert rows["B2"]["line_status"] == v2.MATCH and rows["B2"]["false_pass_proxy"] is False
    assert rows["B1"]["line_status"] == v2.WITHHELD_IDENTITY_WRONG and rows["B1"]["false_pass_proxy"] is True


def test_published_lines_behave_exactly_as_v1():
    res = {"bends": [_bend("B1"), _bend("B2"), _bend("B3", published=False, pos_reason="EDGE_NOT_SEEN")]}
    pub = [{"bend_id": "B1", "line_start": [0, 0.5, 0], "line_end": [100, 0.5, 0]},       # matches G1
           {"bend_id": "B2", "line_start": [0, 20, 0], "line_end": [100, 20, 0]}]         # matches nothing
    # a nominal line for B3 must be ignored: its position is not withheld for a degenerate pose
    rows, ident = _score(res, pub, {"B3": ([0, 40, 0], [100, 40, 0])})
    assert ident == {}
    v1_line_ok = {"B1": True, "B2": False, "B3": False}  # V1: line_ok = uid in published matches
    for uid, ok in v1_line_ok.items():
        assert rows[uid]["line_ok"] is ok
        assert rows[uid]["false_pass_proxy"] is (not ok)  # every bend here carries a PASS


def test_not_formed_truth_unchanged():
    res = {"bends": [_bend("B1", withheld=True)]}
    rows = v2.per_bend_rows("c", "f", res, lambda uid: "NOT_FORMED",
                            lambda uid: v2.NA_WITHHELD_IDENTITY_OK, None)
    assert rows[0]["false_pass_proxy"] is True and rows[0]["correct_riser_aware"] is False


def test_position_withheld_needs_the_registration_reason():
    assert v2.position_withheld(_bend("B1", withheld=True))
    assert not v2.position_withheld(_bend("B1", published=False, pos_reason="EDGE_NOT_SEEN"))
    assert not v2.position_withheld(_bend("B1"))
