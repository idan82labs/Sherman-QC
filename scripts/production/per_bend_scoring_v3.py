"""Per-bend correctness given CAD and the false-PASS proxy, scoring definition V3 (report-only).

Amendment of V2 (``scripts/production/per_bend_scoring_v2.py``, kept untouched and runnable). Owner decision
2026-10-19 (FIXES2.md section 4 item 4), recorded in ``docs/production/runlog/freeze/LEDGER.json`` as a
NOTE / AMENDMENT together with the contract amendment ENGINE_IO-A1 (``contracts/production/ENGINE_IO.md`` 7.1).

What changes against V2, and only this:

* A FLANGE_LENGTH side that is ``NOT_APPLICABLE(NOT_APPLICABLE_AT_STAGE)`` (its adjacent bend is NOT_YET_FORMED,
  so its length is undefined at that stage; the contract validator enforces that) counts as satisfied, in both the
  strict and the riser-aware flange criterion. V2 required every flange side MEASURED.
* Everything else (line status, identity by the mapped CAD nominal line, the false-PASS proxy, NOT_FORMED truth)
  is V2, imported from the V2 module.
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

_spec = importlib.util.spec_from_file_location("per_bend_scoring_v2", Path(__file__).with_name("per_bend_scoring_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

DEFINITION = {
    "id": "PER_BEND_GIVEN_CAD_V3",
    "supersedes": "PER_BEND_GIVEN_CAD_V2 (scripts/production/per_bend_scoring_v2.py) - kept runnable",
    "amendment": "FLANGE_LENGTH NOT_APPLICABLE(NOT_APPLICABLE_AT_STAGE) (adjacent bend NOT_YET_FORMED, contract "
                 "amendment ENGINE_IO-A1) satisfies the flange criterion (strict and riser-aware); all else V2",
    "owner_decision": "2026-10-19 FIXES2.md section 4 item 4",
}

FORMED = v2.FORMED
# re-exported V2 pieces (unchanged)
position_withheld = v2.position_withheld
identity_by_nominal = v2.identity_by_nominal
line_status = v2.line_status
line_ok_of = v2.line_ok_of
MATCH, MISS, NOT_APPLIED = v2.MATCH, v2.MISS, v2.NOT_APPLIED
NA_WITHHELD_IDENTITY_OK, WITHHELD_IDENTITY_WRONG = v2.NA_WITHHELD_IDENTITY_OK, v2.WITHHELD_IDENTITY_WRONG


def flange_na_at_stage(c: Mapping[str, Any]) -> bool:
    return c.get("state") == "NOT_APPLICABLE" and c.get("reason") == "NOT_APPLICABLE_AT_STAGE"


def per_bend_rows(cid: str, family: str, result: Mapping[str, Any], truth_of: Callable[[str], str | None],
                  status_of: Callable[[str], str], line_src: str | None) -> list[dict[str, Any]]:
    """V3 rows: V2 rows with the flange criterion amended (module docstring)."""
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
            fl_strict = all(c.get("state") == "MEASURED" or flange_na_at_stage(c) for c in fl)
            fl_riser = all(c.get("state") == "MEASURED" or flange_na_at_stage(c)
                           or c.get("reason") == "UNSUPPORTED_FEATURE" for c in fl)
            base = st in FORMED and ang_ok and line_ok is not False
            ok_s, ok_r = base and fl_strict, base and fl_riser
        else:
            ok_s = ok_r = st == "NOT_YET_FORMED"
        fp = "PASS" in decs and (t == "NOT_FORMED" or line_ok is False)
        rows.append({"case": cid, "family": family, "bend_uid": uid, "label": b["label"], "truth": t, "state": st,
                     "line_status": status, "line_ok": line_ok, "line_src": line_src, "correct_strict": ok_s,
                     "correct_riser_aware": ok_r, "false_pass_proxy": fp, "any_pass": "PASS" in decs,
                     "flange_na_at_stage": sum(flange_na_at_stage(c) for c in fl),
                     "decisions": dict(Counter(decs))})
    return rows
