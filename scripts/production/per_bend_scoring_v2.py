"""Per-bend correctness given CAD and the false-PASS proxy, scoring definition V2 (report-only).

Amendment of the frozen V1 definition (``docs/production/runlog/a1/iter_7/measure/tools/per_bend_all.py`` and
``final_eval_2026-10-19/tools/pb_common.py``; both stay untouched and runnable). Owner decision 2026-10-19
("option 1"), recorded in ``docs/production/runlog/freeze/LEDGER.json`` as a NOTE / AMENDMENT.

What changes against V1, and only this:

* V1 sets ``line_ok = False`` for every bend that published no line. When the datum leaves an in-plane direction
  free, E3 withholds the bend-line position (``BEND_LINE_POSITION`` = ``NOT_MEASURED(REGISTRATION_UNTRUSTED)``),
  so V1 counted every such bend with a PASS characteristic as a false PASS.
* V2: a withheld position is *not applicable*. The bend's identity is checked instead: the recipe's CAD nominal
  bend line (the E1 ``outer_apex`` locus of the decoded stage), mapped into the raw scan frame through the E3
  registration, is matched to the gold lines left unmatched by the published lines, with the frozen matcher
  (5 deg / 4 mm / 0.6 overlap, Hungarian, OUTER_APEX locus, raw frame).
    - identity matched  -> ``line_ok = None`` (n/a): no line miss, no false PASS;
    - identity not matched (or no nominal line) -> ``line_ok = False``: the bend counts as wrong, and a PASS on it
      is still a false PASS (fallback).
* A bend that published a line is scored exactly as in V1 (its match is computed first, on the published lines
  alone, with the frozen matcher), so published-line behaviour is unchanged by construction.
* Expected NOT_FORMED bends are unchanged (a PASS there is a false PASS).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence

DEFINITION = {
    "id": "PER_BEND_GIVEN_CAD_V2",
    "supersedes": "PER_BEND_GIVEN_CAD_V1 (a1/iter_7 per_bend_all.py; final_eval tools/pb_common.py) - kept runnable",
    "amendment": "position withheld (BEND_LINE_POSITION NOT_MEASURED/REGISTRATION_UNTRUSTED) -> line criterion n/a; "
                 "identity by the CAD nominal outer-apex line mapped to raw through the E3 registration, matched to "
                 "the gold lines left unmatched by published lines with the frozen matcher; no match -> wrong bend "
                 "(PASS on it stays a false PASS); published lines unchanged",
    "owner_decision": "2026-10-19 option 1",
}

FORMED = {"FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"}
WITHHELD_STATE, WITHHELD_REASON = "NOT_MEASURED", "REGISTRATION_UNTRUSTED"

# line status values
MATCH = "MATCH"                                   # published line matches gold (V1 behaviour)
MISS = "MISS"                                     # no matching line (V1 behaviour)
NA_WITHHELD_IDENTITY_OK = "NA_POSITION_WITHHELD_IDENTITY_OK"
WITHHELD_IDENTITY_WRONG = "POSITION_WITHHELD_IDENTITY_WRONG"
NOT_APPLIED = "NOT_APPLIED"                       # no gold for this case: criterion not applied (V1 behaviour)


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def position_char(bend: Any) -> Any:
    for ch in _get(bend, "characteristics") or []:
        if _get(ch, "kind") == "BEND_LINE_POSITION":
            return ch
    return None


def position_withheld(bend: Any) -> bool:
    """True when E3 withheld the bend-line position because the pose leaves an in-plane direction free."""
    ch = position_char(bend)
    return ch is not None and _get(ch, "state") == WITHHELD_STATE and _get(ch, "reason") == WITHHELD_REASON


def identity_by_nominal(withheld_uids: Iterable[str], nominal_lines: Mapping[str, Any],
                        gold_lines: Sequence[Mapping[str, Any]], matched_gold_ids: Iterable[str],
                        match_fn: Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Mapping[str, Any]]
                        ) -> dict[str, dict[str, Any]]:
    """``{uid: {"identity_ok": bool, "gold_id": str|None, "metrics": {...}|None}}`` for every withheld bend.

    ``nominal_lines[uid]`` is ``(start, end)`` in the raw scan frame (or missing -> identity not verifiable -> wrong).
    Only gold lines not already matched by a published line are eligible (one-to-one), and the matching is one
    Hungarian assignment of the frozen matcher ``match_fn(gold, predictions)``."""
    used = {str(g) for g in matched_gold_ids}
    remaining = [g for g in gold_lines if str(g["bend_id"]) not in used]
    uids = [str(u) for u in withheld_uids]
    preds = [{"bend_id": u, "line_start": [float(x) for x in nominal_lines[u][0]],
              "line_end": [float(x) for x in nominal_lines[u][1]]} for u in uids if nominal_lines.get(u) is not None]
    out = {u: {"identity_ok": False, "gold_id": None, "metrics": None,
               "why": "NO_NOMINAL_LINE" if nominal_lines.get(u) is None else "NO_GOLD_MATCH"} for u in uids}
    if not preds or not remaining:
        return out
    m = match_fn(remaining, preds)
    for x in m.get("matches") or []:
        u = str(x["prediction_id"])
        out[u] = {"identity_ok": True, "gold_id": str(x["gold_id"]), "why": "MATCHED",
                  "metrics": {k: x.get(k) for k in ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap",
                                                     "prediction_span_overlap") if k in x}}
    return out


def line_status(uid: str, *, criterion_applied: bool, published: bool, published_match_ids: Iterable[str],
                withheld: bool, identity_ok: bool | None) -> str:
    if not criterion_applied:
        return NOT_APPLIED
    if published:
        return MATCH if uid in set(published_match_ids) else MISS
    if withheld:
        return NA_WITHHELD_IDENTITY_OK if identity_ok else WITHHELD_IDENTITY_WRONG
    return MISS


def line_ok_of(status: str) -> bool | None:
    return {MATCH: True, MISS: False, WITHHELD_IDENTITY_WRONG: False, NA_WITHHELD_IDENTITY_OK: None,
            NOT_APPLIED: None}[status]


def per_bend_rows(cid: str, family: str, result: Mapping[str, Any], truth_of: Callable[[str], str | None],
                  status_of: Callable[[str], str], line_src: str | None) -> list[dict[str, Any]]:
    """V2 rows for one engine-OK case (same row shape as V1 ``pb_common.per_bend_rows`` plus ``line_status``).
    ``status_of(uid)`` returns a :func:`line_status` value."""
    rows = []
    for b in result["bends"]:
        uid = b["bend_uid"]
        t = truth_of(uid)
        if t is None or t == "NOT_DETERMINABLE":
            continue
        chs = b["characteristics"]
        ang = [c for c in chs if c["kind"] == "INCLUDED_ANGLE"]
        fl = [c for c in chs if c["kind"] == "FLANGE_LENGTH"]
        decs = [c.get("decision") for c in chs]
        status = status_of(uid)
        line_ok = line_ok_of(status)
        st = b["state"]
        if t == "FORMED":
            ang_ok = bool(ang) and ang[0].get("state") == "MEASURED"
            fl_strict = all(c.get("state") == "MEASURED" for c in fl)
            fl_riser = all(c.get("state") == "MEASURED" or c.get("reason") == "UNSUPPORTED_FEATURE" for c in fl)
            base = st in FORMED and ang_ok and line_ok is not False
            ok_s, ok_r = base and fl_strict, base and fl_riser
        else:
            ok_s = ok_r = st == "NOT_YET_FORMED"
        fp = "PASS" in decs and (t == "NOT_FORMED" or line_ok is False)
        rows.append({"case": cid, "family": family, "bend_uid": uid, "label": b["label"], "truth": t, "state": st,
                     "line_status": status, "line_ok": line_ok, "line_src": line_src, "correct_strict": ok_s,
                     "correct_riser_aware": ok_r, "false_pass_proxy": fp, "any_pass": "PASS" in decs,
                     "decisions": dict(Counter(decs))})
    return rows
