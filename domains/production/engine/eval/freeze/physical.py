"""Physical development runner (E3 arm) and the frozen G2 / G3 evaluator (plan §5 G2, G3; §10 amendments 4d-f, 6, 7).

Written and frozen before A1 starts, so the scoring rules of the physical gates
cannot be shaped by the accuracy loop (eval code changed after the freeze makes
every scorecard non-qualifying, :mod:`..freeze`). Numbers come from
``G2_PHYSICAL_THRESHOLDS``, ``G2_LINE_THRESHOLDS`` and ``G3_THRESHOLDS``
(:mod:`..thresholds`); nothing is inline.

**E3 arm** (:func:`run_e3_case`): the engine command line exactly as the line and
the E4a runner call it (``python -m domains.production.engine.inspect``, 180 s
hard stop, 4 GiB, ``--time-budget-s 170``, the three sha256 arguments), in a
workspace named by the opaque case reference with read-only staged copies
``input/scan.<ext>`` and ``input/recipe.json`` (amendment 4: no file name, part
number, stage key or count reaches E3). The output is validated with the model
and ``check_result_against_compiled``; any problem is an engine failure that
stays in every denominator. :func:`extract_e3` turns a valid result into one row
per recipe bend.

**G2 evaluator** (:func:`score_g2`), on one frozen case document (the A1 lists or
the sealed DEV-CHECK document):

- *(4d) compile rate* (:data:`COMPILE_GATE_RULE`, decided before A1 and recorded
  in the ledger): COMPILED share of the frozen DEV_AUTO_CONFIRMED recipes
  (``E1-DEV-POLICY-1``, the recipes the evaluation uses) over every authorized
  unique STEP and over the plan §9 source-tree STEPs; both must reach
  ``compile_rate_min``. The strict single-source figures (V10 marks only, which
  E1's ENGINE_COMPILE.md §10 calls "the amendment-4d figure", and native marks
  only) are reported next to it as diagnostics, never gated.
- *(b) coverage*, over every G2_COMPLETE case (compile and engine failures
  included): MEASURED INCLUDED_ANGLE share of the planned bends and MEASURED
  FLANGE_LENGTH share of the planned flange sides (two per bend, both REQUIRED).
  The only exclusion is a characteristic E1's frozen per-feature rule declared
  ``UNSUPPORTED_FEATURE`` (amendment 4d); ``NOMINAL_UNAVAILABLE`` stays in.
- *(c) angle on known-good parts*: |actual - nominal| MAE of every MEASURED angle
  (<= ``mae_deg_max``) and, on the bends **both** arms measure, the engine MAE
  no worse than the legacy arm's (legacy |measured - E1 nominal|, the nominal folded like
  the worker's angles); engine
  angle coverage >= legacy coverage on the known-good planned bends. Per part,
  with a histogram.
- *(d) flange length on known-good parts*: P95 |actual - nominal| of every
  MEASURED flange side, nothing excluded; every |delta| above
  ``large_deviation_mm`` is listed for a zoomed crop.
- *(e) runtime and failures*: P95 engine wall time over every G2 case run
  (failures included at their wall time); crashes = every run status other than
  OK and the explicit input refusal ``ENGINE_EXIT_2`` (a hard-stop kill, an RSS
  kill, exit 3, an invalid or inconsistent result).
- *(f) repeatability*: per evaluable repeat set, every pair of its scans; per
  bend measured in both, |delta angle|, per side measured in both, |delta flange|;
  pooled P95 <= the limits and below the legacy arm's P95 (legacy angle, and the
  legacy tip-to-tip edge distance as its flange proxy). A legacy arm that
  produced no repeat pair at all has no repeatability to beat (PASS, recorded);
  likewise (c) "no worse than legacy" holds when the legacy arm measured no
  known-good bend while the engine measured some.
- *(a) line metric*: :func:`..line_metric.score_line_metric` on the frozen Gold
  cases with the engine's declared-locus lines (raw frame) and the legacy arm's
  scan-fit lines.

**Visual truth table** (amendment 7, :func:`load_truth_table`): rows ``(case_ref,
bend_uid, reviewer_1, reviewer_2, tiebreak, final)`` with labels FORMED /
NOT_FORMED / NOT_DETERMINABLE; ``final`` must equal the agreeing reviewers or the
tie-break. A ``provisional_complete`` G2 case (corpus stage keys say partial)
stays in G2 only when every bend's final label is FORMED; otherwise it is
scored with the staged cases. Without a frozen table the scorecard is
``final = false``.

**Amendment-7 order** ("built before any engine output is seen"), enforced
twice on the **truth-relevant** cases (:func:`truth_relevant`: the G3 and BROAD
staged scans and the provisional-complete G2 scans, i.e. every case whose
per-bend truth a score reads): :func:`truth_table_gate` is checked by
``prefreeze run-physical`` before any E3 run on such a case (a frozen table that
covers every template row of those cases and whose sha256 a FREEZE entry of the
ledger records), and :func:`truth_table_order` makes the scorer reject a table
frozen, or recorded in the ledger, after the earliest E3 record of a
truth-relevant case of the document (G3 NOT_EVALUABLE, ``final = false``). The
``STATE_CHECK`` rows of the other complete scans are a completeness check that no
score reads.

**G3 evaluator** (:func:`score_g3`) on the staged cases: truth FORMED matches an
engine FORMED / PARTIALLY_FORMED / UNEXPECTED_FORMED state, truth NOT_FORMED
matches NOT_YET_FORMED, an engine UNKNOWN matches only NOT_DETERMINABLE; an
engine claim on a NOT_DETERMINABLE bend is listed as an unverified claim (no
truth exists to score it). Gates: state match rate, no FAIL on a truly unformed
bend, the blank decoded as stage 0 with every bend NOT_YET_FORMED, the engine's
formed count equal to ``expected_completed`` where it exists (scoring only; the
owner's count refers to the owner's bend inventory, so it is mapped onto the
recipe only when unambiguous: equal inventories, nothing formed or everything
formed -- otherwise it is listed as not comparable, :func:`_expected_formed`), and
G2(b) coverage on the truly formed bends (G2(c)/(d) on those of known-good
families; an unformed bend's "deviation" is its whole sweep and is never scored).

:func:`legacy_baseline` gives the legacy arm's own numbers on a case document
(G2a line metric, coverage, known-good angle) without any engine.

**G2(a) dual report** (owner direction 2026-10-17, :func:`_line_metric_dual`): the
same line metric against the Gold11 audit relabel (``ai_bend_lines_gold11_relabel_v1``,
re-read at scoring time with its snapshot sha256 compared with the freeze's), next
to the human Gold11 on all cases and on exactly the cases the relabel covers.
Report-only: the human Gold11 stays the gate of record until the owner signs off.

**BROAD report** (owner directions 2026-10-17, :func:`score_broad`, report-only):
on the BROAD cases (every CONSUMED scan of ``case_list_broad_v3.json`` whose STEP
compiles), the family-macro line-F1 (:func:`broad_line_scores`: mean over families
of the family-micro F1 of the complete-label line cases, the broad baseline's
definition; a ``with_controls`` variant adds zero-bend-control lines as false
positives) of the engine and of the broad baseline's reference lanes (the shipped
runtime and the old best research lane, read from the frozen
``broad_reference_lanes_*.json``) on the same frozen gold, against the owner's 0.80
aim; plus per-bend checks: G2(b) coverage on the complete BROAD scans, G2(e)
runtime and failures, G2(c)/(d) on known-good families, the G3 state rules on the
staged BROAD scans (frozen visual truth table), and the engine's bend-state and
characteristic-decision counts.

**Recipe ceiling** (:func:`recipe_line_ceiling`, report-only, BROAD and G2(a)): a
recipe-guided engine publishes at most one line per recipe bend, so a case with
``G`` gold lines and ``R`` planned bends has at most ``min(G, R)`` true positives.
The best line-F1 any such engine can reach on the frozen gold (no false
positive) is reported per case, per family and family-macro next to the 0.80
aim, so a gold that caps a family below the aim is visible before any run.
"""
from __future__ import annotations

import itertools
import math
import os
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from contracts.production.states import FORMED_LIKE_STATES

from ..engine_output import load_compiled, load_engine_output, load_recipe, row_characteristic
from ..procmon import run_monitored
from ..runner import RunnerConfig, _classify, engine_env
from ..thresholds import G1_THRESHOLDS, G2_LINE_THRESHOLDS, G2_PHYSICAL_THRESHOLDS, G3_THRESHOLDS, thresholds_sha256
from .common import REPO_ROOT, custody_check, now_utc, read_json, sha256_file, write_json

__all__ = [
    "PHYSICAL_RECORD_SCHEMA",
    "SCORECARD_SCHEMA",
    "TRUTH_TABLE_SCHEMA",
    "TRUTH_LABELS",
    "COMPILE_GATE_RULE",
    "TRUTH_RELEVANT_LISTS",
    "compile_rate_block",
    "compile_gate_fields",
    "run_e3_case",
    "extract_e3",
    "load_truth_table",
    "truth_relevant",
    "ledger_record_of",
    "truth_table_gate",
    "truth_table_order",
    "recipe_line_ceiling",
    "planned_population",
    "score_g2",
    "score_g3",
    "score_physical",
    "score_broad",
    "broad_line_scores",
    "BROAD_OWNER_TARGET",
    "legacy_baseline",
]

PHYSICAL_RECORD_SCHEMA = "sherman_e4b_physical_run_v1"
SCORECARD_SCHEMA = "sherman_e4b_physical_scorecard_v1"
TRUTH_TABLE_SCHEMA = "sherman_e4b_g3_truth_table_v1"
TRUTH_LABELS = ("FORMED", "NOT_FORMED", "NOT_DETERMINABLE")
_RC = G1_THRESHOLDS["run_conditions"]
#: Engine statuses that are an explicit refusal, not a crash (e.g. COMPILE_BLOCKED, SCAN_MISSING).
_EXPLICIT_REFUSALS = ("ENGINE_EXIT_2",)
#: Amendment 4d compile-rate gate, decided before A1 (the ledger held no RUN or WAIVER entry) and recorded
#: in the freeze ledger. The floor (``compile_rate_min``, unchanged) applies to the recipes it names.
COMPILE_GATE_RULE: dict[str, Any] = {
    "rule_version": "E4B-4D-GATE-2",
    "gated_variant": "dev",
    "gated_denominators": ["rates (every authorized unique STEP)", "rates_source_assets (plan §9 source-tree STEPs)"],
    "diagnostic_variants": ["v10_strict", "native_strict"],
    "basis": "plan §10 amendment 4d: 'The DEV_AUTO_CONFIRMED recipes: every V10/native bend, both flange sides "
             "REQUIRED, exclusions only through a pre-written per-feature UNSUPPORTED rule. Compile failures stay in "
             "the denominators, and at least 80% of authorized STEPs must compile.' The floor applies to the frozen "
             "DEV_AUTO_CONFIRMED recipes (E1-DEV-POLICY-1: the V10/native union, pieces of one fold merged, "
             "rule-declared features excluded), i.e. the recipes the evaluation runs on. The strict single-source "
             "V10 figure is called 'the amendment-4d figure' only in E1's ENGINE_COMPILE.md §10; it is reported next "
             "to the gate as a diagnostic",
    "decided": "2026-10-17, E4b freeze fix pass, before A1; flagged for owner confirmation before the anchor is "
               "published",
    "alternative_not_taken": "gate the strict V10 figure as well (the first pass): strict V10 is 20/26 = 0.769 on "
                             "the widened denominator, so G2 would be FAIL by construction on A1 and on DEV-CHECK "
                             "whatever the engine does",
}
#: Lists whose cases carry scored per-bend truth (with ``provisional_complete`` G2 cases): :func:`truth_relevant`.
TRUTH_RELEVANT_LISTS = ("G3_STAGED", "BROAD_STAGED")


def _p95(values: Sequence[float]) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95)) if len(values) else None


def _gate(value: float | None, *, max_: float | None = None, min_: float | None = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NOT_EVALUABLE"
    if max_ is not None and value > max_:
        return "FAIL"
    if min_ is not None and value < min_:
        return "FAIL"
    return "PASS"


def _verdict(gates: Mapping[str, str]) -> str:
    vals = list(gates.values())
    if any(v == "FAIL" for v in vals):
        return "FAIL"
    if vals and all(v == "PASS" for v in vals):
        return "PASS"
    return "NOT_EVALUABLE"


# --------------------------------------------------------------------------------------------
# E3 arm
# --------------------------------------------------------------------------------------------


def _stage_readonly(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
        dst.unlink()
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _char_doc(c: Any) -> dict[str, Any] | None:
    if c is None:
        return None
    return {"characteristic_id": c.characteristic_id, "state": c.state, "decision": c.decision, "reason": c.reason,
            "required": c.required, "expected": c.expected, "actual": c.actual, "deviation": c.deviation,
            "uncertainty": c.uncertainty, "side": c.side}


def extract_e3(result: Any, compiled: Any) -> dict[str, Any]:
    """Case summary and one row per recipe bend of a valid E3 result."""
    rows = []
    for row in result.bends:
        backgauge = [c for c in row.characteristics if c.kind == "BACKGAUGE_LENGTH"]
        rows.append({
            "bend_uid": row.bend_uid, "sequence_index": row.sequence_index, "label": row.label, "state": row.state,
            "state_reasons": list(row.state_reasons),
            "measured_included_deg_evidence": row.state_evidence.measured_included_deg,
            "angle": _char_doc(row_characteristic(row, "INCLUDED_ANGLE")),
            "flange": {"A": _char_doc(row_characteristic(row, "FLANGE_LENGTH", "A")),
                       "B": _char_doc(row_characteristic(row, "FLANGE_LENGTH", "B"))},
            "position": _char_doc(row_characteristic(row, "BEND_LINE_POSITION")),
            "backgauge": [_char_doc(c) for c in backgauge],
            "fail_any": any(c.decision == "FAIL" for c in row.characteristics),
            "line_published": row.display.measured_polyline_mm is not None,
        })
    sd = result.stage_decision
    return {"validity": result.inspection_validity.state, "validity_reasons": list(result.inspection_validity.reasons),
            "summary": result.summary.state, "engine_version": result.engine.engine_version,
            "stage": {"inferred_stage": sd.inferred_stage, "stage_model_id": sd.stage_model_id, "status": sd.status,
                      "decision": sd.decision, "formed": list(sd.formed_bend_uids),
                      "not_yet_formed": list(sd.not_yet_formed_bend_uids), "unknown": list(sd.unknown_bend_uids)},
            "pose_ambiguous": bool(result.inspection_validity.registration.pose_ambiguous),
            "bends": rows}


def run_e3_case(case: Mapping[str, Any], scan_path: Path, recipe_path: Path, compiled_dir: Path, work_root: Path, *,
                repo_root: Path = REPO_ROOT, engine_config: Path | None = None) -> dict[str, Any]:
    """One E3 run on a physical case under the standard run conditions (module docstring)."""
    scan_src = custody_check(scan_path, "physical E3 scan")
    ws = Path(work_root) / "physical" / "E3" / case["case_ref"]
    if ws.exists():
        for p in ws.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        shutil.rmtree(ws)
    scan_in = ws / "input" / f"scan{scan_src.suffix.lower()}"
    recipe_in = ws / "input" / "recipe.json"
    _stage_readonly(scan_src, scan_in)
    _stage_readonly(recipe_path, recipe_in)
    out = ws / "engine"
    recipe = load_recipe(recipe_in)
    record: dict[str, Any] = {"schema": PHYSICAL_RECORD_SCHEMA, "arm": "E3", "case_ref": case["case_ref"],
                              "recorded_at": now_utc(), "scan_sha256": case["scan_sha256"],
                              "recipe_sha256": recipe.recipe_sha256}
    cp = compiled_dir / "compiled_part.json"
    compiled = load_compiled(cp) if cp.is_file() else None
    argv = [sys.executable, *_RC["engine_argv_tail"], "--compiled", str(cp), "--recipe", str(recipe_in),
            "--scan", str(scan_in), "--out", str(out), "--scan-sha256", case["scan_sha256"],
            "--recipe-sha256", recipe.recipe_sha256, "--time-budget-s", f"{_RC['engine_time_budget_s']:g}"]
    if compiled is not None:
        argv += ["--compiled-sha256", compiled.compiled_part_sha256]
    if engine_config is not None:
        argv += ["--config", str(engine_config)]
    env = engine_env(RunnerConfig(tier_dir=ws, out_dir=ws, repo_root=repo_root))
    proc = run_monitored(argv, cwd=repo_root, env=env, timeout_s=float(_RC["engine_timeout_s"]),
                         rss_limit_bytes=int(_RC["engine_rss_limit_bytes"]), stderr_path=ws / "engine.stderr.log",
                         stdout_path=ws / "engine.stdout.log")
    status, detail = _classify(proc, out)
    extraction = None
    if status == "OK":
        eo = load_engine_output(out, compiled=compiled, recipe=recipe) if compiled is not None else None
        if eo is None:
            status, detail = "RESULT_INCONSISTENT", "NO_COMPILED_PART"
        elif eo.status != "OK":
            status, detail = eo.status, (eo.error or ",".join(eo.problems))[:500]
        elif eo.result.scan.sha256 != case["scan_sha256"]:
            status, detail = "RESULT_INCONSISTENT", "SCAN_SHA_MISMATCH"
        else:
            extraction = extract_e3(eo.result, compiled)
            record["result_sha256"] = eo.result_sha256
    modified = [n for p, d, n in ((scan_in, case["scan_sha256"], "scan"),) if sha256_file(p) != d]
    if modified and status == "OK":
        status, detail, extraction = "RESULT_INCONSISTENT", "ENGINE_MODIFIED_INPUT:scan", None
    record.update({"status": status, "detail": detail, "wall_s": round(proc.wall_s, 3),
                   "peak_rss_bytes": proc.peak_rss_bytes, "engine_dir": str(out),
                   "compiled_part_sha256": None if compiled is None else compiled.compiled_part_sha256,
                   "conditions": {"engine_argv_tail": list(_RC["engine_argv_tail"]),
                                  "engine_timeout_s": _RC["engine_timeout_s"],
                                  "engine_rss_limit_bytes": _RC["engine_rss_limit_bytes"],
                                  "engine_time_budget_s": _RC["engine_time_budget_s"],
                                  "engine_config_sha256": None if engine_config is None else sha256_file(engine_config)},
                   "extraction": extraction})
    write_json(ws / "record.json", record)
    return record


# --------------------------------------------------------------------------------------------
# truth table and populations
# --------------------------------------------------------------------------------------------


def load_truth_table(path: str | os.PathLike | None) -> dict[tuple[str, str], str] | None:
    """``(case_ref, bend_uid) -> final label`` of a frozen visual truth table (module docstring)."""
    if path is None:
        return None
    doc = read_json(path)
    if doc.get("schema") != TRUTH_TABLE_SCHEMA:
        raise ValueError(f"{path}: not a {TRUTH_TABLE_SCHEMA} document")
    if not doc.get("frozen_at"):
        raise ValueError(f"{path}: the truth table is not frozen (frozen_at missing)")
    out: dict[tuple[str, str], str] = {}
    for r in doc.get("rows") or []:
        a, b, t, final = r.get("reviewer_1"), r.get("reviewer_2"), r.get("tiebreak"), r.get("final")
        for v in (a, b, final):
            if v not in TRUTH_LABELS:
                raise ValueError(f"{path}: label {v!r} not in {TRUTH_LABELS}")
        expected = a if a == b else t
        if expected not in TRUTH_LABELS or final != expected:
            raise ValueError(f"{path}: {r.get('case_ref')}/{r.get('bend_uid')}: final must equal the agreeing "
                             "reviewers or the tie-break")
        key = (str(r["case_ref"]), str(r["bend_uid"]))
        if key in out:
            raise ValueError(f"{path}: duplicate row {key}")
        out[key] = final
    return out


def truth_relevant(case: Mapping[str, Any]) -> bool:
    """Whether a score reads this case's per-bend visual truth: a G3 or BROAD staged scan (state
    rules) or a ``provisional_complete`` G2 scan (it stays in G2 only if every bend is FORMED)."""
    return bool(case.get("provisional_complete")) or any(
        str(lab).split(":")[0] in TRUTH_RELEVANT_LISTS for lab in case.get("lists") or [])


def _ts(value: Any) -> datetime | None:
    """An ISO-8601 UTC time (``...Z``); naive times are taken as UTC."""
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def ledger_record_of(ledger: str | os.PathLike, sha256: str) -> dict[str, Any] | None:
    """The first FREEZE entry of a verifying ledger that records a file with ``sha256`` (a ``files`` or
    ``case_lists`` item, or a sha-only ``devcheck_sealed`` item; ``prefreeze record-truth-table``
    writes them), or None."""
    from .. import freeze as e4a_ledger

    p = Path(ledger)
    if not p.is_file() or not e4a_ledger.verify_ledger(p)["ok"]:
        return None
    for e in e4a_ledger.read_ledger(p):
        if e.get("kind") != "FREEZE":
            continue
        for group in ("files", "case_lists", "devcheck_sealed"):
            for name, d in ((e.get("items") or {}).get(group) or {}).items():
                if isinstance(d, Mapping) and d.get("sha256") == sha256:
                    return {"entry_id": e.get("entry_id"), "recorded_at": e.get("recorded_at"),
                            "item": f"{group}:{name}"}
    return None


def truth_table_gate(path: str | os.PathLike | None, required: Mapping[str, Sequence[str]], *,
                     ledger: str | os.PathLike | None = None, now: str | None = None) -> dict[str, Any]:
    """May E3 run on cases with these truth-table rows (``case_ref -> bend uids``; ``prefreeze
    run-physical`` passes the template rows of its truth-relevant cases, :func:`truth_relevant`)?
    Amendment 7: only once the table that covers every one of those rows is frozen and (with
    ``ledger``) recorded in a FREEZE entry of the ledger. Returns ``{"ok", "reason", ...}``; an
    unfrozen or invalid table raises ``ValueError`` (:func:`load_truth_table`)."""
    need = {str(c): sorted(set(u)) for c, u in required.items() if u}
    out: dict[str, Any] = {"cases_with_truth_rows": len(need), "truth_table": None if path is None else str(path)}
    if not need:
        return {**out, "ok": True, "reason": None}
    if path is None:
        return {**out, "ok": False,
                "reason": "NO_FROZEN_TRUTH_TABLE: amendment 7 requires the per-bend visual truth table of these "
                          "scans to be filled by two reviewers plus a tie-break, frozen and recorded in the ledger "
                          "before any engine output on them exists"}
    table = load_truth_table(path)
    missing = sum(1 for c, uids in need.items() for u in uids if (c, u) not in table)
    if missing:
        return {**out, "ok": False, "reason": f"TRUTH_TABLE_INCOMPLETE: {missing} row(s) of these cases missing"}
    doc = read_json(path)
    frozen = _ts(doc.get("frozen_at"))
    now_t = _ts(now) if now is not None else datetime.now(timezone.utc)
    if frozen is None or now_t is None or frozen > now_t:
        return {**out, "ok": False, "reason": "TRUTH_TABLE_FROZEN_AT_INVALID (unparseable or in the future)"}
    out.update(truth_table_sha256=sha256_file(path), frozen_at=doc.get("frozen_at"))
    if ledger is not None:
        rec = ledger_record_of(ledger, out["truth_table_sha256"])
        if rec is None:
            return {**out, "ok": False,
                    "reason": "TRUTH_TABLE_NOT_RECORDED_IN_LEDGER: record its sha256 first (python -m "
                              "domains.production.engine.eval.freeze.prefreeze record-truth-table --truth-table PATH)"}
        out["ledger_record"] = rec
    return {**out, "ok": True, "reason": None}


def truth_table_order(path: str | os.PathLike | None, e3_records: Mapping[str, Mapping[str, Any]], *,
                      ledger: str | os.PathLike | None = None, relevant: set[str] | None = None) -> dict[str, Any]:
    """Scorer side of the amendment-7 order: ``status`` ACCEPTED, MISSING or REJECTED_* (a table frozen,
    or recorded in the ledger, after the earliest E3 record of a truth-relevant case, or never
    recorded). ``relevant`` (case refs, :func:`truth_relevant`) restricts the E3 records compared;
    None compares every record."""
    if path is None:
        return {"status": "MISSING"}
    doc = read_json(path)
    out: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "frozen_at": doc.get("frozen_at"),
                           "compared_e3_records": "truth-relevant cases" if relevant is not None else "every case"}
    times = [t for t in (_ts(r.get("recorded_at")) for ref, r in e3_records.items()
                         if relevant is None or ref in relevant) if t is not None]
    earliest = min(times) if times else None
    out["earliest_e3_record_at"] = None if earliest is None else earliest.strftime("%Y-%m-%dT%H:%M:%SZ")
    frozen = _ts(doc.get("frozen_at"))
    if frozen is None:
        return {**out, "status": "REJECTED_FROZEN_AT_INVALID"}
    if earliest is not None and frozen > earliest:
        return {**out, "status": "REJECTED_FROZEN_AFTER_E3_OUTPUT"}
    if ledger is None:
        return {**out, "status": "ACCEPTED", "ledger_record": "NOT_CHECKED"}
    rec = ledger_record_of(ledger, out["sha256"])
    out["ledger_record"] = rec
    if rec is None:
        return {**out, "status": "REJECTED_NOT_RECORDED_IN_LEDGER"}
    rec_t = _ts(rec.get("recorded_at"))
    if earliest is not None and (rec_t is None or rec_t > earliest):
        return {**out, "status": "REJECTED_RECORDED_AFTER_E3_OUTPUT"}
    return {**out, "status": "ACCEPTED"}


def recipe_line_ceiling(gold_cases: Sequence[Mapping[str, Any]], pops: Mapping[str, Any], *,
                        target: float | None = None, only_ids: set[str] | None = None) -> dict[str, Any]:
    """The best line-F1 any recipe-guided engine can reach on frozen gold (module docstring): per case
    ``min(G, R)`` true positives (``R`` = planned bends of the compiled part, 0 without one) and no false
    positive. Zero-bend controls and recall-only (partial-label) cases are left out, like the
    family-macro F1; ``only_ids`` restricts the population to the cases the metric scored."""
    capped, fams = [], {}
    tp_all = g_all = 0
    for g in gold_cases:
        if g.get("zero_bend_control") or not g.get("label_complete", True):
            continue
        if only_ids is not None and g["case_id"] not in only_ids:
            continue
        n_gold = len(g.get("gold_lines") or [])
        if n_gold == 0:
            continue
        pop = pops.get(g["case_id"]) or {}
        n_rec = len(pop.get("bends") or []) if pop.get("source") == "COMPILED" else 0
        tp = min(n_gold, n_rec)
        f = fams.setdefault(str(g.get("family")), [0, 0])
        f[0] += tp
        f[1] += n_gold
        tp_all += tp
        g_all += n_gold
        if tp < n_gold:
            capped.append({"case_id": g["case_id"], "family": g.get("family"), "gold_lines": n_gold,
                           "recipe_bends": n_rec, "recall_ceiling": tp / n_gold, "f1_ceiling": 2.0 * tp / (tp + n_gold),
                           "flags": [x for x in g.get("flags") or [] if x in (
                               "GOLD_LINES_EXCEED_RECIPE_BENDS", "RECIPE_EXCLUDED_FEATURE", "GOLD_CONVENTION_CONFLICT")]})
    per_family = {f: {"true_positive_max": tp, "gold_lines": n, "recall_ceiling": tp / n if n else None,
                      "f1_ceiling": 2.0 * tp / (tp + n) if tp + n else 0.0} for f, (tp, n) in sorted(fams.items())}
    macro = float(np.mean([v["f1_ceiling"] for v in per_family.values()])) if per_family else None
    out: dict[str, Any] = {
        "definition": "best line-F1 of any recipe-guided engine on this frozen gold: at most one line per recipe "
                      "bend, so min(gold lines, recipe bends) true positives per case and no false positive",
        "capped_cases": capped, "per_family": per_family, "family_macro_f1_ceiling": macro,
        "micro_f1_ceiling": (2.0 * tp_all / (tp_all + g_all)) if g_all else None}
    if target is not None:
        out["target"] = target
        out["families_capped_below_target"] = sorted(f for f, v in per_family.items() if v["f1_ceiling"] < target)
        out["target_reachable"] = None if macro is None else bool(macro >= target)
    return out


def planned_population(recipe_path: Path | None, compiled_dir: Path | None) -> dict[str, Any]:
    """Planned bends and flange sides of one case and E1's rule-declared exclusions."""
    bends: list[dict[str, Any]] = []
    excluded: set[str] = set()
    if compiled_dir is not None and (compiled_dir / "compiled_part.json").is_file():
        cp = load_compiled(compiled_dir / "compiled_part.json")
        excluded = {c.characteristic_id for c in cp.characteristics if c.unavailable_reason == "UNSUPPORTED_FEATURE"}
        ids = {(c.kind, c.bend_uid, c.side): c.characteristic_id for c in cp.characteristics}
        for b in cp.bends:
            bends.append({"bend_uid": b.bend_uid, "sequence_index": b.sequence_index,
                          "angle_id": ids.get(("INCLUDED_ANGLE", b.bend_uid, None)),
                          "side_ids": {s: ids.get(("FLANGE_LENGTH", b.bend_uid, s)) for s in ("A", "B")},
                          "nominal_included_deg": float(b.nominal_included_deg)})
        if cp.status == "COMPILED":
            return {"source": "COMPILED", "bends": bends, "rule_excluded_ids": sorted(excluded)}
    if recipe_path is not None and Path(recipe_path).is_file():
        recipe = load_recipe(recipe_path)
        bends = [{"bend_uid": b.bend_uid, "sequence_index": b.sequence_index, "angle_id": None,
                  "side_ids": {"A": None, "B": None}, "nominal_included_deg": None} for b in recipe.bends]
        return {"source": "RECIPE_ONLY_COMPILE_NOT_USABLE", "bends": bends, "rule_excluded_ids": []}
    return {"source": "NO_RECIPE", "bends": [], "rule_excluded_ids": []}


def _rows_by_uid(record: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    ext = (record or {}).get("extraction") or {}
    if (record or {}).get("status") != "OK" or not ext:
        return {}
    return {r["bend_uid"]: r for r in ext.get("bends") or []}


def _restrict(record: Mapping[str, Any], uids: set[str]) -> dict[str, Any]:
    """A copy of an arm record whose extraction keeps only the bends in ``uids``."""
    ext = dict(record.get("extraction") or {})
    ext["bends"] = [r for r in ext.get("bends") or [] if r["bend_uid"] in uids]
    return {**record, "extraction": ext}


def _measured(c: Mapping[str, Any] | None) -> bool:
    return bool(c) and c.get("state") == "MEASURED" and c.get("actual") is not None


# --------------------------------------------------------------------------------------------
# G2
# --------------------------------------------------------------------------------------------


def _coverage(cases: Sequence[Mapping[str, Any]], e3: Mapping[str, Any], pops: Mapping[str, Any],
              only: Mapping[str, set[str]] | None = None) -> dict[str, Any]:
    """G2(b) shares; ``only`` restricts a case to some bend uids (truly formed bends of staged scans)."""
    a_n = a_m = s_n = s_m = a_x = s_x = 0
    per_case = []
    for c in cases:
        pop = pops[c["case_ref"]]
        rows = _rows_by_uid(e3.get(c["case_ref"]))
        keep = None if only is None else only.get(c["case_ref"], set())
        ca_n = ca_m = cs_n = cs_m = 0
        for b in pop["bends"]:
            if keep is not None and b["bend_uid"] not in keep:
                continue
            row = rows.get(b["bend_uid"])
            if b["angle_id"] in pop["rule_excluded_ids"]:
                a_x += 1
            else:
                ca_n += 1
                ca_m += int(_measured((row or {}).get("angle")))
            for s in ("A", "B"):
                if b["side_ids"][s] in pop["rule_excluded_ids"]:
                    s_x += 1
                    continue
                cs_n += 1
                cs_m += int(_measured(((row or {}).get("flange") or {}).get(s)))
        a_n, a_m, s_n, s_m = a_n + ca_n, a_m + ca_m, s_n + cs_n, s_m + cs_m
        per_case.append({"case_ref": c["case_ref"], "planned_bends": ca_n, "angle_measured": ca_m,
                         "planned_sides": cs_n, "sides_measured": cs_m, "population_source": pop["source"],
                         "engine_status": (e3.get(c["case_ref"]) or {}).get("status", "NOT_RUN")})
    th = G2_PHYSICAL_THRESHOLDS["coverage"]
    ar = a_m / a_n if a_n else None
    sr = s_m / s_n if s_n else None
    return {"angle": {"planned": a_n, "measured": a_m, "rate": ar, "rule_excluded": a_x,
                      "gate": _gate(ar, min_=th["bend_angle_measured_min"])},
            "flange_sides": {"planned": s_n, "measured": s_m, "rate": sr, "rule_excluded": s_x,
                             "gate": _gate(sr, min_=th["flange_side_length_measured_min"])},
            "per_case": per_case}


def _hist(values: Sequence[float], edges: Sequence[float]) -> list[dict[str, Any]]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append({"lo": lo, "hi": None if math.isinf(hi) else hi,  # None = open upper bin (JSON has no inf)
                    "count": int(sum(1 for v in values if lo <= v < hi))})
    return out


def _angle_known_good(cases, e3, legacy, pops) -> dict[str, Any]:
    th = G2_PHYSICAL_THRESHOLDS["angle_nominal_relative"]
    all_e3, common_e3, common_lg = [], [], []
    e3_planned = e3_meas = lg_meas = 0
    per_part: dict[str, dict[str, Any]] = {}
    for c in cases:
        rows = _rows_by_uid(e3.get(c["case_ref"]))
        lrows = {r["bend_uid"]: r for r in ((legacy.get(c["case_ref"]) or {}).get("bends") or [])}
        part = per_part.setdefault(c["family"], {"bends": 0, "e3_measured": 0, "legacy_measured": 0, "e3_abs": [],
                                                 "common_e3_abs": [], "common_legacy_abs": []})
        for b in pops[c["case_ref"]]["bends"]:
            if b["angle_id"] in pops[c["case_ref"]]["rule_excluded_ids"]:
                continue
            e3_planned += 1
            part["bends"] += 1
            a = (rows.get(b["bend_uid"]) or {}).get("angle")
            lr = lrows.get(b["bend_uid"]) or {}
            lg_ok = bool((lr.get("angle") or {}).get("measured")) and (lr.get("angle") or {}).get(
                "legacy_delta_vs_e1_deg") is not None
            if _measured(a) and a.get("deviation") is not None:
                e3_meas += 1
                part["e3_measured"] += 1
                all_e3.append(abs(float(a["deviation"])))
                part["e3_abs"].append(abs(float(a["deviation"])))
            if lg_ok:
                lg_meas += 1
                part["legacy_measured"] += 1
            if _measured(a) and a.get("deviation") is not None and lg_ok:
                common_e3.append(abs(float(a["deviation"])))
                common_lg.append(abs(float(lr["angle"]["legacy_delta_vs_e1_deg"])))
                part["common_e3_abs"].append(common_e3[-1])
                part["common_legacy_abs"].append(common_lg[-1])
    mae = float(np.mean(all_e3)) if all_e3 else None
    mae_c = float(np.mean(common_e3)) if common_e3 else None
    mae_l = float(np.mean(common_lg)) if common_lg else None
    cov_e = e3_meas / e3_planned if e3_planned else None
    cov_l = lg_meas / e3_planned if e3_planned else None
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf")]
    parts = {f: {"bends": p["bends"], "e3_measured": p["e3_measured"], "legacy_measured": p["legacy_measured"],
                 "e3_mae_deg": float(np.mean(p["e3_abs"])) if p["e3_abs"] else None,
                 "common_bends": len(p["common_e3_abs"]),
                 "common_e3_mae_deg": float(np.mean(p["common_e3_abs"])) if p["common_e3_abs"] else None,
                 "common_legacy_mae_deg": float(np.mean(p["common_legacy_abs"])) if p["common_legacy_abs"] else None,
                 "histogram_e3_abs_deg": _hist(p["e3_abs"], edges)} for f, p in sorted(per_part.items())}
    gates = {"mae_all_measured": _gate(mae, max_=th["mae_deg_max"]),
             "mae_common": _gate(mae_c, max_=th["mae_deg_max"]),
             "no_worse_than_legacy": ("PASS" if lg_meas == 0 and e3_meas > 0 else
                                      "NOT_EVALUABLE" if mae_c is None or mae_l is None
                                      else "PASS" if mae_c <= mae_l else "FAIL"),
             "coverage_at_least_legacy": ("NOT_EVALUABLE" if cov_e is None or cov_l is None
                                          else "PASS" if cov_e >= cov_l else "FAIL")}
    return {"planned_bends": e3_planned, "e3_measured": e3_meas, "legacy_measured": lg_meas,
            "e3_mae_deg": mae, "common_bends": len(common_e3), "common_e3_mae_deg": mae_c,
            "common_legacy_mae_deg": mae_l, "e3_coverage": cov_e, "legacy_coverage": cov_l,
            "legacy_delta_definition": "legacy measured_angle - E1 included nominal folded like the worker "
                                       "(max(theta, 180 - theta)); both arms against the same nominal",
            "per_part": parts, "gates": gates, "gate": _verdict(gates)}


def _flange_known_good(cases, e3) -> dict[str, Any]:
    th = G2_PHYSICAL_THRESHOLDS["flange_nominal_relative"]
    vals, large = [], []
    for c in cases:
        for uid, r in _rows_by_uid(e3.get(c["case_ref"])).items():
            for s in ("A", "B"):
                f = (r.get("flange") or {}).get(s)
                if _measured(f) and f.get("deviation") is not None:
                    d = abs(float(f["deviation"]))
                    vals.append(d)
                    if d > th["large_deviation_mm"]:
                        large.append({"case_ref": c["case_ref"], "bend_uid": uid, "side": s,
                                      "deviation_mm": float(f["deviation"]), "evidence": "zoomed section crop required"})
    p95 = _p95(vals)
    return {"measured_sides": len(vals), "p95_abs_mm": p95, "max_abs_mm": max(vals) if vals else None,
            "exclusions": "none (amendment 6d)", "large_deviations": large,
            "gate": _gate(p95, max_=th["p95_abs_mm_max"])}


def _runtime(cases, e3) -> dict[str, Any]:
    th = G2_PHYSICAL_THRESHOLDS["runtime"]
    walls, crashes, explicit, not_run = [], [], [], []
    for c in cases:
        r = e3.get(c["case_ref"])
        if r is None:
            not_run.append(c["case_ref"])
            continue
        walls.append(float(r["wall_s"]))
        if r["status"] in _EXPLICIT_REFUSALS:
            explicit.append({"case_ref": c["case_ref"], "status": r["status"], "detail": r.get("detail")})
        elif r["status"] != "OK":
            crashes.append({"case_ref": c["case_ref"], "status": r["status"], "detail": r.get("detail")})
    p95 = _p95(walls)
    gates = {"p95_wall": _gate(p95, max_=th["p95_wall_s_max"]),
             "crashes": "NOT_EVALUABLE" if not walls else ("PASS" if len(crashes) <= th["crashes_max"] else "FAIL"),
             "all_cases_run": "PASS" if not not_run else "FAIL"}
    return {"runs": len(walls), "p95_wall_s": p95, "max_wall_s": max(walls) if walls else None,
            "legacy_control_p95_s": th["legacy_control_p95_s"], "crashes": crashes, "explicit_refusals": explicit,
            "not_run": not_run, "gates": gates, "gate": _verdict(gates)}


def _pairs_delta(arm_rows: Sequence[Mapping[str, dict[str, Any]]], arm: str) -> tuple[list[float], list[float]]:
    da, dl = [], []
    for r1, r2 in itertools.combinations(arm_rows, 2):
        for uid in sorted(set(r1) & set(r2)):
            a, b = r1[uid], r2[uid]
            if arm == "E3":
                if _measured(a.get("angle")) and _measured(b.get("angle")):
                    da.append(abs(float(a["angle"]["actual"]) - float(b["angle"]["actual"])))
                for s in ("A", "B"):
                    fa, fb = (a.get("flange") or {}).get(s), (b.get("flange") or {}).get(s)
                    if _measured(fa) and _measured(fb):
                        dl.append(abs(float(fa["actual"]) - float(fb["actual"])))
            else:
                ma, mb = (a.get("angle") or {}).get("legacy_measured_deg"), (b.get("angle") or {}).get(
                    "legacy_measured_deg")
                if ma is not None and mb is not None:
                    da.append(abs(float(ma) - float(mb)))
                ea, eb = (a.get("edge_distance_proxy") or {}).get("measured_mm"), (b.get("edge_distance_proxy") or {}).get(
                    "measured_mm")
                if ea is not None and eb is not None:
                    dl.append(abs(float(ea) - float(eb)))
    return da, dl


def _repeatability(sets, e3, legacy) -> dict[str, Any]:
    th = G2_PHYSICAL_THRESHOLDS["repeatability"]
    e_a, e_l, l_a, l_l = [], [], [], []
    per_set = []
    for s in sets:
        if s["status"] != "EVALUABLE":
            per_set.append({"family": s["family"], "status": s["status"], "reason": s.get("reason")})
            continue
        e_rows = [_rows_by_uid(e3.get(ref)) for ref in s["case_refs"]]
        l_rows = [{r["bend_uid"]: r for r in ((legacy.get(ref) or {}).get("bends") or [])} for ref in s["case_refs"]]
        da, dl = _pairs_delta(e_rows, "E3")
        la, ll = _pairs_delta(l_rows, "LEGACY")
        e_a += da
        e_l += dl
        l_a += la
        l_l += ll
        per_set.append({"family": s["family"], "kind": s.get("kind"), "status": "EVALUATED",
                        "pairs": len(s["case_refs"]) * (len(s["case_refs"]) - 1) // 2,
                        "e3_angle_pairs": len(da), "e3_flange_pairs": len(dl), "legacy_angle_pairs": len(la),
                        "legacy_edge_pairs": len(ll), "e3_p95_angle_deg": _p95(da), "e3_p95_flange_mm": _p95(dl),
                        "legacy_p95_angle_deg": _p95(la), "legacy_p95_edge_proxy_mm": _p95(ll)})
    pe_a, pe_l, pl_a, pl_l = _p95(e_a), _p95(e_l), _p95(l_a), _p95(l_l)

    def better(e: float | None, lg: float | None) -> str:
        if e is None:
            return "NOT_EVALUABLE"
        if lg is None:  # the legacy arm produced no repeat pair at all: it has no repeatability to beat
            return "PASS"
        return "PASS" if e < lg else "FAIL"

    gates = {"p95_angle": _gate(pe_a, max_=th["p95_abs_between_repeats_deg_max"]),
             "p95_flange": _gate(pe_l, max_=th["p95_abs_between_repeats_mm_max"]),
             "angle_better_than_legacy": better(pe_a, pl_a),
             "flange_better_than_legacy_edge_proxy": better(pe_l, pl_l)}
    return {"per_set": per_set, "e3_p95_angle_deg": pe_a, "e3_p95_flange_mm": pe_l, "legacy_p95_angle_deg": pl_a,
            "legacy_p95_edge_proxy_mm": pl_l,
            "legacy_flange_note": "legacy has no OML flange length; its tip-to-tip edge distance is the proxy",
            "gates": gates, "gate": _verdict(gates)}


def _gold_case_objects(gold_cases: Sequence[Mapping[str, Any]]) -> list[Any]:
    from ..line_metric import GoldCase

    return [GoldCase(case_id=g["case_id"], gold_lines=tuple(g["gold_lines"]), frame=g["frame"],
                     raw_to_reference=None if g.get("raw_to_reference") is None else np.asarray(g["raw_to_reference"]),
                     residual_mm=g.get("residual_mm"), transform_method=g.get("transform_method")) for g in gold_cases]


def _engine_line_predictions(case_ids: Sequence[str], e3_records: Mapping[str, Any],
                             compiled_dirs: Mapping[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """Declared-locus (OUTER_APEX) engine lines in the raw-scan frame of every valid E3 record."""
    from ..engine_output import load_engine_output as _leo
    from ..line_metric import engine_lines

    preds: dict[str, list[dict[str, Any]]] = {}
    for cid in case_ids:
        r = e3_records.get(cid)
        cdir = compiled_dirs.get(cid)
        if r is None or r.get("status") != "OK" or cdir is None:
            continue
        compiled = load_compiled(Path(cdir) / "compiled_part.json")
        eo = _leo(r["engine_dir"])
        if eo.result is None:
            continue
        preds[cid] = engine_lines(eo.result, compiled, locus="OUTER_APEX", frame="raw")
    return preds


def _line_metric(gold_cases, e3_records, compiled_dirs, legacy) -> dict[str, Any]:
    from ..line_metric import score_line_metric

    cases = _gold_case_objects(gold_cases)
    preds = _engine_line_predictions([g["case_id"] for g in gold_cases], e3_records, compiled_dirs)
    lg = {g["case_id"]: list((legacy.get(g["case_id"]) or {}).get("line_predictions_raw") or []) for g in gold_cases}
    report = score_line_metric(cases, preds, legacy_predictions=lg)
    report["note"] = G2_LINE_THRESHOLDS["label"]
    return report


def _line_metric_dual(cases_doc: Mapping[str, Any], e3: Mapping[str, Any], compiled_dirs: Mapping[str, Path],
                      legacy: Mapping[str, Any], human: Mapping[str, Any]) -> dict[str, Any] | None:
    """G2(a) against the Gold11 audit relabel, next to the human Gold11 (owner direction 2026-10-17:
    report both until the owner signs off; never replace silently). Report-only: the human set is
    the gate of record. Also scores the human set on exactly the cases the relabel covers."""
    relabel = list(cases_doc.get("relabel_gold_cases") or [])
    if not relabel:
        return None
    covered = {g["case_id"] for g in relabel}
    gold = [g for g in cases_doc.get("gold_cases") or [] if g["case_id"] in covered]
    rel = _line_metric(relabel, e3, compiled_dirs, legacy)
    hum = _line_metric(gold, e3, compiled_dirs, legacy) if gold else None
    keys = ("f1", "precision", "recall", "true_positive", "false_positive", "false_negative", "exact_part_rate",
            "case_count")
    at_freeze, now = cases_doc.get("relabel_snapshot_sha256"), cases_doc.get("relabel_snapshot_sha256_now")
    others_then, others_now = cases_doc.get("relabel_other_files"), cases_doc.get("relabel_other_files_now")
    changed = None if now is None else (now != at_freeze or (others_now is not None and others_now != others_then))
    return {"gate_of_record": "HUMAN_GOLD11", "relabel_status": cases_doc.get("relabel_snapshot_status"),
            "relabel_snapshot_scope": cases_doc.get("relabel_snapshot_scope", "every relabel file"),
            "relabel_snapshot_sha256_at_freeze": at_freeze, "relabel_snapshot_sha256_now": now,
            "relabel_other_files_at_freeze": others_then, "relabel_other_files_now": others_now,
            "relabel_changed_since_freeze": changed,
            "cases_covered_by_relabel": sorted(covered),
            "relabel": {k: (rel.get("engine") or {}).get(k) for k in keys} | {"verdict": rel.get("verdict")},
            "human_on_the_same_cases": None if hum is None else
            {k: (hum.get("engine") or {}).get(k) for k in keys} | {"verdict": hum.get("verdict")},
            "human_all_cases": {k: (human.get("engine") or {}).get(k) for k in keys} | {"verdict": human.get("verdict")},
            "relabel_report": rel,
            "note": "report-only; the relabel is AI-adjudicated (claude-vision-ai-v1) and pending the owner's sign-off"}


def _compile_rate(compile_report: Mapping[str, Any]) -> dict[str, Any]:
    """The amendment-4d block (:data:`COMPILE_GATE_RULE`): the frozen DEV recipes gated on both denominators,
    the strict single-source figures reported as diagnostics (``meets_floor``, never a gate)."""
    floor = G2_PHYSICAL_THRESHOLDS["compile_rate_min"]
    gated = COMPILE_GATE_RULE["gated_variant"]

    def part(rates: Mapping[str, Any]) -> dict[str, Any]:
        block = {gated: {**(rates.get(gated) or {}), "gate": _gate((rates.get(gated) or {}).get("rate"), min_=floor)}}
        block["diagnostics"] = {v: {**(rates.get(v) or {}), "meets_floor": None if (rates.get(v) or {}).get("rate")
                                    is None else bool(rates[v]["rate"] >= floor)}
                                for v in COMPILE_GATE_RULE["diagnostic_variants"]}
        return block

    out: dict[str, Any] = {"rule": dict(COMPILE_GATE_RULE), "floor": floor, **part(compile_report.get("rates") or {})}
    sub = {gated: out[gated]["gate"]}
    src = compile_report.get("rates_source_assets") or {}
    if src:
        out["source_assets"] = part(src)
        sub[f"source_assets_{gated}"] = out["source_assets"][gated]["gate"]
    out["gate"] = _verdict(sub)
    return out


def compile_rate_block(compile_report: Mapping[str, Any]) -> dict[str, Any]:
    """The amendment-4d block G2 reports (:data:`COMPILE_GATE_RULE`; see :func:`_compile_rate`)."""
    return _compile_rate(compile_report)


def compile_gate_fields(rates: Mapping[str, Any], rates_source_assets: Mapping[str, Any]) -> dict[str, Any]:
    """Gate and floor fields of a compile-report document, under :data:`COMPILE_GATE_RULE`: the gated
    variant's gate on both denominators, and per variant whether its rate meets the floor (the strict
    single-source figures are diagnostics, never a gate)."""
    floor = G2_PHYSICAL_THRESHOLDS["compile_rate_min"]
    blk = _compile_rate({"rates": rates, "rates_source_assets": rates_source_assets})
    gated = COMPILE_GATE_RULE["gated_variant"]

    def meets(r: Mapping[str, Any]) -> dict[str, bool | None]:
        return {v: (None if (d or {}).get("rate") is None else bool(d["rate"] >= floor)) for v, d in sorted(r.items())}

    return {"gate_rule": dict(COMPILE_GATE_RULE),
            "gate": {"rule_version": COMPILE_GATE_RULE["rule_version"], "variant": gated,
                     "rates": blk[gated]["gate"],
                     "rates_source_assets": (blk.get("source_assets") or {}).get(gated, {}).get("gate", "NOT_EVALUABLE"),
                     "verdict": blk["gate"]},
            "meets_floor": meets(rates), "meets_floor_source_assets": meets(rates_source_assets),
            "figures": {"gated": {"variant": gated, "label": "the frozen DEV_AUTO_CONFIRMED recipes (E1-DEV-POLICY-1: "
                                                             "the V10/native union the evaluation runs on)"},
                        "diagnostics": [{"variant": "v10_strict",
                                         "label": "strict V10, every bend as reported (E1 ENGINE_COMPILE.md §10 "
                                                  "calls it 'the amendment-4d figure'); reported, not gated"},
                                        {"variant": "native_strict",
                                         "label": "native extractor only, every bend as reported; reported, not gated"}]}}


def score_g2(cases_doc: Mapping[str, Any], *, e3: Mapping[str, Any], legacy: Mapping[str, Any],
             pops: Mapping[str, Any], compile_report: Mapping[str, Any] | None,
             truth: Mapping[tuple[str, str], str] | None, compiled_dirs: Mapping[str, Path]) -> dict[str, Any]:
    """The G2 report on one frozen case document (module docstring)."""
    cases = cases_doc["cases"]
    g2, moved = [], []
    for c in cases:
        if "G2_COMPLETE" not in c["lists"]:
            continue
        if c.get("provisional_complete") and truth is not None:
            labels = [truth.get((c["case_ref"], b["bend_uid"])) for b in pops[c["case_ref"]]["bends"]]
            if any(lab != "FORMED" for lab in labels):
                moved.append(c["case_ref"])
                continue
        g2.append(c)
    kg_fams = {f for v in (cases_doc.get("known_good_families") or {}).values() for f in v}
    known_good = [c for c in g2 if c["family"] in kg_fams]
    out: dict[str, Any] = {"population": {"g2_cases": len(g2), "known_good_cases": len(known_good),
                                          "moved_to_staged_by_truth_table": moved}}
    if compile_report is not None:
        out["compile_rate"] = _compile_rate(compile_report)
    out["coverage"] = _coverage(g2, e3, pops)
    out["angle_known_good"] = _angle_known_good(known_good, e3, legacy, pops)
    out["flange_known_good"] = _flange_known_good(known_good, e3)
    out["runtime"] = _runtime(g2, e3)
    out["repeatability"] = _repeatability(cases_doc.get("repeatability_sets") or [], e3, legacy)
    gold = cases_doc.get("gold_cases") or []
    out["line_metric"] = _line_metric(gold, e3, compiled_dirs, legacy) if gold else {"verdict": "NOT_EVALUABLE",
                                                                                      "reason": "no Gold case"}
    if gold:  # report-only: what the frozen Gold allows a recipe-guided engine at best
        out["line_metric_recipe_ceiling"] = recipe_line_ceiling(gold, pops, target=G2_LINE_THRESHOLDS["f1_min"])
    dual = _line_metric_dual(cases_doc, e3, compiled_dirs, legacy, out["line_metric"]) if gold else None
    if dual is not None:
        out["line_metric_dual_report"] = dual
    gates = {
        "compile_rate": (out.get("compile_rate") or {}).get("gate", "NOT_EVALUABLE"),
        "a_line_metric": out["line_metric"].get("verdict", "NOT_EVALUABLE"),
        "b_angle_coverage": out["coverage"]["angle"]["gate"],
        "b_flange_coverage": out["coverage"]["flange_sides"]["gate"],
        "c_angle_known_good": out["angle_known_good"]["gate"],
        "d_flange_known_good": out["flange_known_good"]["gate"],
        "e_runtime_and_failures": out["runtime"]["gate"],
        "f_repeatability": out["repeatability"]["gate"],
    }
    out["gates"] = gates
    out["verdict"] = _verdict(gates)
    return out


# --------------------------------------------------------------------------------------------
# G3
# --------------------------------------------------------------------------------------------


def _expected_formed(case: Mapping[str, Any], planned: int) -> tuple[int | None, str | None]:
    """The formed count ``expected_completed`` implies for the recipe's ``planned`` bends.

    ``expected_completed`` counts bends in the owner's inventory (``expected_total_bends``), which can
    differ from the recipe's (E1's per-feature rule excludes rolled/hem features). The comparison is
    made only when the mapping is unambiguous: equal inventories, nothing formed, or everything formed."""
    keys = case.get("stage_keys") or {}
    exp, total = keys.get("expected_completed"), keys.get("expected_total_bends")
    if exp is None:
        return None, None
    if case.get("stage_class") == "BLANK_STAGE0" or exp == 0:
        return 0, "nothing formed"
    if total is None or int(total) == planned:
        return int(exp), "owner and recipe inventories agree"
    if int(exp) == int(total):
        return planned, "every bend formed"
    return None, (f"NOT_COMPARABLE: owner inventory {total} bends, recipe plans {planned}; "
                  f"expected_completed {exp} cannot be mapped onto the recipe")


def _match(truth: str, engine: str | None) -> str:
    if engine is None:
        return "MISMATCH"
    if truth == "FORMED":
        return "MATCH" if engine in FORMED_LIKE_STATES else "MISMATCH"
    if truth == "NOT_FORMED":
        return "MATCH" if engine == "NOT_YET_FORMED" else "MISMATCH"
    return "MATCH" if engine == "UNKNOWN" else "UNVERIFIED_CLAIM"


def score_g3(cases_doc: Mapping[str, Any], *, e3: Mapping[str, Any], legacy: Mapping[str, Any],
             pops: Mapping[str, Any], truth: Mapping[tuple[str, str], str] | None,
             moved: Sequence[str] = ()) -> dict[str, Any]:
    """The G3 report on the staged cases of one frozen case document (module docstring)."""
    staged = [c for c in cases_doc["cases"] if "G3_STAGED" in c["lists"] or c["case_ref"] in set(moved)]
    if truth is None:
        return {"population": len(staged), "truth_table": "MISSING", "verdict": "NOT_EVALUABLE",
                "reason": "the frozen visual truth table (amendment 7) does not exist yet"}
    rows, unverified, defects, missing_truth = [], [], [], []
    match = total = 0
    formed_only: dict[str, set[str]] = {}
    blank, counts, not_comparable = [], [], []
    for c in staged:
        erows = _rows_by_uid(e3.get(c["case_ref"]))
        status = (e3.get(c["case_ref"]) or {}).get("status", "NOT_RUN")
        formed_engine = 0
        formed_truth = 0
        for b in pops[c["case_ref"]]["bends"]:
            t = truth.get((c["case_ref"], b["bend_uid"]))
            if t is None:
                missing_truth.append({"case_ref": c["case_ref"], "bend_uid": b["bend_uid"]})
                continue
            er = erows.get(b["bend_uid"])
            state = None if er is None else er["state"]
            m = _match(t, state)
            if state in FORMED_LIKE_STATES:
                formed_engine += 1
            if t == "FORMED":
                formed_truth += 1
                formed_only.setdefault(c["case_ref"], set()).add(b["bend_uid"])
            if m == "UNVERIFIED_CLAIM":
                unverified.append({"case_ref": c["case_ref"], "bend_uid": b["bend_uid"], "engine_state": state})
            else:
                total += 1
                match += int(m == "MATCH")
            if t == "NOT_FORMED" and er is not None and er.get("fail_any"):
                defects.append({"case_ref": c["case_ref"], "bend_uid": b["bend_uid"]})
            rows.append({"case_ref": c["case_ref"], "bend_uid": b["bend_uid"], "truth": t, "engine_state": state,
                         "engine_status": status, "result": m})
        if c.get("stage_class") == "BLANK_STAGE0":
            ext = (e3.get(c["case_ref"]) or {}).get("extraction") or {}
            ok = (status == "OK" and (ext.get("stage") or {}).get("inferred_stage") == 0
                  and all(r["state"] == "NOT_YET_FORMED" for r in erows.values()) and len(erows) > 0)
            blank.append({"case_ref": c["case_ref"], "decoded_stage_0_all_not_yet_formed": ok,
                          "inferred_stage": (ext.get("stage") or {}).get("inferred_stage")})
        target, why = _expected_formed(c, len(pops[c["case_ref"]]["bends"]))
        if target is not None:
            counts.append({"case_ref": c["case_ref"], "expected_formed": target, "basis": why,
                           "engine_formed": formed_engine, "truth_formed": formed_truth,
                           "engine_equals": formed_engine == target, "truth_equals": formed_truth == target})
        elif why:
            not_comparable.append({"case_ref": c["case_ref"], "reason": why})
    th = G3_THRESHOLDS
    rate = match / total if total else None
    kg_fams = {f for v in (cases_doc.get("known_good_families") or {}).values() for f in v}
    formed_cov = _coverage(staged, e3, pops, only=formed_only)
    kg_staged = [c for c in staged if c["family"] in kg_fams]
    gates = {
        "state_match_rate": _gate(rate, min_=th["state_match_rate_min"]),
        "truth_complete": "PASS" if not missing_truth else "FAIL",
        "unformed_reported_as_defect": "PASS" if len(defects) <= th["unformed_reported_as_defect_max"] else "FAIL",
        "blank_stage0": ("NOT_EVALUABLE" if not blank else "PASS" if all(b["decoded_stage_0_all_not_yet_formed"]
                                                                        for b in blank) else "FAIL"),
        "formed_count_equals_expected_completed": ("NOT_EVALUABLE" if not counts else "PASS" if all(
            x["engine_equals"] for x in counts) else "FAIL"),
        "formed_bends_angle_coverage": formed_cov["angle"]["gate"],
        "formed_bends_flange_coverage": formed_cov["flange_sides"]["gate"],
    }
    out = {"population": len(staged), "truth_table": "FROZEN", "bends_scored": total, "matches": match,
           "state_match_rate": rate, "rows": rows, "unverified_claims": unverified,
           "unformed_reported_as_defect": defects, "missing_truth": missing_truth, "blank": blank,
           "formed_counts": counts, "formed_counts_not_comparable": not_comparable,
           "formed_bends_coverage": formed_cov,
           "truth_table_count_consistency": [x for x in counts if not x["truth_equals"]],
           "gates": gates}
    if kg_staged:
        keep = {c["case_ref"]: formed_only.get(c["case_ref"], set()) for c in kg_staged}
        kg_pops = {k: {**pops[k], "bends": [b for b in pops[k]["bends"] if b["bend_uid"] in v]}
                   for k, v in keep.items()}
        kg_e3 = {k: _restrict(e3[k], v) for k, v in keep.items() if k in e3}
        ang = _angle_known_good(kg_staged, kg_e3, legacy, kg_pops)
        fl = _flange_known_good(kg_staged, kg_e3)
        out["formed_bends_known_good_angle"] = ang
        out["formed_bends_known_good_flange"] = fl
        gates["formed_bends_known_good_angle_mae"] = ang["gates"]["mae_all_measured"]
        gates["formed_bends_known_good_flange_p95"] = fl["gate"]
    out["verdict"] = _verdict(gates)
    return out


def legacy_baseline(cases_doc: Mapping[str, Any], *, legacy: Mapping[str, Any], pops: Mapping[str, Any]
                    ) -> dict[str, Any]:
    """The legacy arm's own numbers on one frozen case document (no engine involved): the G2a line
    metric against the frozen Gold cases, G2(b) coverage and the known-good angle deviation."""
    g2 = [c for c in cases_doc["cases"] if "G2_COMPLETE" in c["lists"]]
    kg_fams = {f for v in (cases_doc.get("known_good_families") or {}).values() for f in v}
    planned = measured = 0
    deltas = []
    for c in g2:
        rows = {r["bend_uid"]: r for r in ((legacy.get(c["case_ref"]) or {}).get("bends") or [])}
        for b in pops[c["case_ref"]]["bends"]:
            planned += 1
            a = (rows.get(b["bend_uid"]) or {}).get("angle") or {}
            if a.get("measured"):
                measured += 1
                if c["family"] in kg_fams and a.get("legacy_delta_vs_e1_deg") is not None:
                    deltas.append(abs(float(a["legacy_delta_vs_e1_deg"])))
    out: dict[str, Any] = {"g2_planned_bends": planned, "g2_angle_measured": measured,
                           "g2_angle_coverage": measured / planned if planned else None,
                           "known_good_angle_measured": len(deltas),
                           "known_good_mae_vs_e1_deg": float(np.mean(deltas)) if deltas else None}
    gold = cases_doc.get("gold_cases") or []
    if gold:
        lm = _line_metric(gold, {}, {}, legacy).get("legacy") or {}
        out["line_metric"] = {k: lm.get(k) for k in ("case_count", "true_positive", "false_positive", "false_negative",
                                                      "precision", "recall", "f1", "exact_part_count",
                                                      "exact_part_rate")}
    return out


# --------------------------------------------------------------------------------------------
# BROAD (owner directions 2026-10-17: family-macro line-F1 + per-bend checks; report-only)
# --------------------------------------------------------------------------------------------

#: The owner's aim for wave 4 (2026-10-17), recorded verbatim in the freeze ledger; not a plan §5 gate.
BROAD_OWNER_TARGET = {"family_macro_line_f1_min": 0.80,
                      "text": "aim for line-F1 >= 0.80 with NO part-specific optimization -- generality over the "
                              "parts in the mix; blind/broad line-F1 is reported family-macro"}


def _prf3(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def broad_line_scores(gold_cases: Sequence[Mapping[str, Any]], predictions: Mapping[str, Sequence[Mapping[str, Any]]]
                      ) -> dict[str, Any]:
    """Family-macro line-F1 of one arm on the frozen BROAD gold (raw frame, the frozen 5 deg / 4 mm /
    0.6 matcher). A case with gold but no prediction entry is a miss (all its gold lines are FN).

    - line cases with complete labels: TP/FP/FN; recall-only (``label_complete`` false): TP/FN only;
    - zero-bend controls: every predicted line is a false positive (reported; counted in the
      ``with_controls`` variant only, like the broad baseline);
    - ``family_macro_f1``: mean over families of the family-micro F1 of its complete-label line cases
      (the broad baseline's definition); ``family_macro_f1_with_controls`` adds each family's
      zero-bend-control lines as false positives."""
    from ..line_metric import match_lines, prepare_gold

    usable, excluded = prepare_gold(_gold_case_objects(gold_cases))
    meta = {g["case_id"]: g for g in gold_cases}
    rows = []
    for cid in sorted(usable):
        g = meta[cid]
        preds = list(predictions.get(cid) or [])
        if g.get("zero_bend_control"):
            rows.append({"case_id": cid, "family": g["family"], "disposition": "ZERO_BEND_CONTROL",
                         "prediction_count": len(preds), "false_positive_lines": len(preds)})
            continue
        m = match_lines(usable[cid], preds)
        row = {"case_id": cid, "family": g["family"], "tier": g.get("tier"), "gold_count": m["gold_count"],
               "prediction_count": m["prediction_count"], "true_positive": m["true_positive"],
               "false_negative": m["false_negative"], "exact": m["exact"],
               "unmatched_gold_ids": m["unmatched_gold_ids"]}
        if g.get("label_complete", True):
            row.update(disposition="SCORED_COMPLETE_LABEL", false_positive=m["false_positive"])
        else:
            row.update(disposition="SCORED_RECALL_ONLY_PARTIAL_LABEL", false_positive=None,
                       unscored_unmatched_predictions=m["false_positive"], exact=None)
        rows.append(row)
    complete = [r for r in rows if r["disposition"] == "SCORED_COMPLETE_LABEL"]
    controls = [r for r in rows if r["disposition"] == "ZERO_BEND_CONTROL"]
    tp = sum(r["true_positive"] for r in complete)
    fp = sum(r["false_positive"] for r in complete)
    fn = sum(r["false_negative"] for r in complete)
    fams: dict[str, list[int]] = {}
    for r in complete:
        f = fams.setdefault(r["family"], [0, 0, 0, 0])
        f[0] += r["true_positive"]
        f[1] += r["false_positive"]
        f[2] += r["false_negative"]
    per_family = {f: {"tp": v[0], "fp": v[1], "fn": v[2], **_prf3(v[0], v[1], v[2])} for f, v in sorted(fams.items())}
    with_ctrl = {f: list(v) for f, v in fams.items()}
    for r in controls:
        if r["family"] in with_ctrl:
            with_ctrl[r["family"]][1] += r["false_positive_lines"]
    macro = float(np.mean([v["f1"] for v in per_family.values()])) if per_family else None
    macro_c = float(np.mean([_prf3(v[0], v[1], v[2])["f1"] for v in with_ctrl.values()])) if with_ctrl else None
    micro = _prf3(tp, fp, fn) if complete else {"precision": None, "recall": None, "f1": None}
    return {"cases": rows, "excluded_before_scoring": excluded, "line_cases": len(complete),
            "families": len(per_family), "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "micro": micro, "family_macro_f1": macro, "family_macro_f1_with_controls": macro_c,
            "family_macro_precision": float(np.mean([v["precision"] for v in per_family.values()])) if per_family
            else None,
            "family_macro_recall": float(np.mean([v["recall"] for v in per_family.values()])) if per_family else None,
            "exact_part_rate": (sum(1 for r in complete if r["exact"]) / len(complete)) if complete else None,
            "zero_bend_controls": {"cases": len(controls),
                                   "false_positive_lines": sum(r["false_positive_lines"] for r in controls)},
            "per_family": per_family}


def score_broad(cases_doc: Mapping[str, Any], *, e3: Mapping[str, Any], legacy: Mapping[str, Any],
                pops: Mapping[str, Any], truth: Mapping[tuple[str, str], str] | None,
                compiled_dirs: Mapping[str, Path], reference_lanes: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
    """BROAD report (report-only; owner directions 2026-10-17): family-macro line-F1 of the engine and of
    the broad baseline's reference lanes on the same frozen gold, and the per-bend checks (coverage,
    runtime and failures, known-good nominal-relative angle / flange, staged per-bend state against
    the frozen visual truth table, decision counts)."""
    broad = [c for c in cases_doc["cases"] if "BROAD" in c["lists"]]
    if not broad:
        return {"verdict": "NOT_EVALUABLE", "reason": "no BROAD case"}
    gold = [g for g in cases_doc.get("broad_gold_cases") or [] if g["case_id"] in {c["case_ref"] for c in broad}]
    preds = _engine_line_predictions([g["case_id"] for g in gold], e3, compiled_dirs)
    engine = broad_line_scores(gold, preds)
    refs = {}
    for lane, per_case in ((reference_lanes or {}).get("lanes") or {}).items():
        refs[lane] = broad_line_scores(gold, {k: v.get("lines") or [] for k, v in per_case.items()})
        refs[lane]["conditions"] = ((reference_lanes or {}).get("meta") or {}).get(lane, {}).get("conditions")
    tgt = BROAD_OWNER_TARGET["family_macro_line_f1_min"]
    scored_ids = {r["case_id"] for r in engine["cases"] if r.get("disposition") == "SCORED_COMPLETE_LABEL"}
    ceiling = recipe_line_ceiling(gold, pops, target=tgt, only_ids=scored_ids)
    complete, staged = [], []
    for c in broad:
        if c["stage_class"] in ("STAGED_PARTIAL", "BLANK_STAGE0"):
            staged.append(c)
            continue
        if c.get("provisional_complete") and truth is not None:
            if any(truth.get((c["case_ref"], b["bend_uid"])) != "FORMED" for b in pops[c["case_ref"]]["bends"]):
                staged.append(c)
                continue
        complete.append(c)
    kg_fams = {f for v in (cases_doc.get("known_good_families") or {}).values() for f in v}
    known_good = [c for c in complete if c["family"] in kg_fams]
    decisions: dict[str, int] = {}
    states: dict[str, int] = {}
    for c in broad:
        for r in _rows_by_uid(e3.get(c["case_ref"])).values():
            states[r["state"]] = states.get(r["state"], 0) + 1
            for ch in [r.get("angle"), *(r.get("flange") or {}).values(), r.get("position"), *(r.get("backgauge") or [])]:
                if ch:
                    decisions[str(ch.get("decision"))] = decisions.get(str(ch.get("decision")), 0) + 1
    staged_doc = {"cases": [dict(c, lists=[*c["lists"], "G3_STAGED"]) for c in staged],
                  "known_good_families": cases_doc.get("known_good_families")}
    out: dict[str, Any] = {
        "population": {"broad_cases": len(broad), "families": len({c["family"] for c in broad}),
                       "gold_cases": len(gold), "complete_cases": len(complete), "staged_cases": len(staged)},
        "label": "consumed development (engineering ledger); recipe-guided engine vs blind reference lanes, not "
                 "like-for-like; report-only (owner target, not a plan §5 gate)",
        "owner_target": dict(BROAD_OWNER_TARGET),
        "line_metric": {"engine": engine, "reference_lanes": refs,
                        "target_status": ("NOT_EVALUABLE" if engine["family_macro_f1"] is None else
                                          "MET" if engine["family_macro_f1"] >= tgt else "NOT_MET"),
                        "recipe_ceiling": ceiling},
        "coverage": _coverage(complete, e3, pops),
        "angle_known_good": _angle_known_good(known_good, e3, legacy, pops),
        "flange_known_good": _flange_known_good(known_good, e3),
        "runtime": _runtime(broad, e3),
        "staged_states": score_g3(staged_doc, e3=e3, legacy=legacy, pops=pops, truth=truth) if staged else
        {"verdict": "NOT_EVALUABLE", "reason": "no staged BROAD case"},
        "engine_bend_states": dict(sorted(states.items())),
        "characteristic_decisions": dict(sorted(decisions.items())),
    }
    out["verdict"] = "REPORT_ONLY"
    return out


def score_physical(cases_doc: Mapping[str, Any], *, e3_dir: Path | None, legacy_dir: Path | None,
                   recipes_dir: Path, compiled_root: Path, compile_report: Mapping[str, Any] | None = None,
                   truth_table: Path | None = None, relabel_live: bool = True, ledger: Path | None = None,
                   ledger_anchor: str | None = None) -> dict[str, Any]:
    """Load the arm records of one frozen case document and score G2, G3 and the BROAD report.

    ``ledger`` enables the ledger checks: the truth table must be recorded in it before the earliest E3
    record (:func:`truth_table_order`) and the frozen eval/contracts code and the anchor are compared
    (``ledger_check``); ``qualifying`` needs both and a final scorecard."""
    pops, compiled_dirs, e3, legacy = {}, {}, {}, {}
    for c in cases_doc["cases"]:
        cdir = compiled_root / c["recipe_sku"]
        compiled_dirs[c["case_ref"]] = cdir
        pops[c["case_ref"]] = planned_population(recipes_dir / f"{c['recipe_sku']}.recipe.json", cdir)
        if e3_dir is not None and (e3_dir / c["case_ref"] / "record.json").is_file():
            e3[c["case_ref"]] = read_json(e3_dir / c["case_ref"] / "record.json")
        if legacy_dir is not None and (legacy_dir / c["case_ref"] / "legacy_case.json").is_file():
            legacy[c["case_ref"]] = read_json(legacy_dir / c["case_ref"] / "legacy_case.json")
    if truth_table is not None:
        load_truth_table(truth_table)  # an unfrozen or invalid table is an input error (raises)
    relevant = {c["case_ref"] for c in cases_doc["cases"] if truth_relevant(c)}
    order = truth_table_order(truth_table, e3, ledger=ledger, relevant=relevant)
    truth = load_truth_table(truth_table) if order["status"] == "ACCEPTED" else None
    if relabel_live and cases_doc.get("gold_cases"):
        # the Gold11 audit relabel is another workflow's product (in progress at the freeze): the dual
        # report reads it as it stands now and records whether it changed since the freeze; only this
        # document's own relabel files are opened (the others are counted by name)
        from .gold import relabel_gold_cases

        files = cases_doc.get("relabel_files")
        live, snap = relabel_gold_cases(cases_doc.get("gold_cases") or [],
                                        only_files=None if files is None else [f["file"] for f in files])
        cases_doc = dict(cases_doc, relabel_gold_cases=live, relabel_snapshot_status=snap["status"],
                         relabel_snapshot_sha256_now=snap["snapshot_sha256"],
                         relabel_other_files_now=snap.get("other_files"))
    g2 = score_g2(cases_doc, e3=e3, legacy=legacy, pops=pops, compile_report=compile_report, truth=truth,
                  compiled_dirs=compiled_dirs)
    g3 = score_g3(cases_doc, e3=e3, legacy=legacy, pops=pops, truth=truth,
                  moved=g2["population"]["moved_to_staged_by_truth_table"])
    lanes = None
    lanes_path = cases_doc.get("broad_reference_lanes_path")
    if lanes_path and (REPO_ROOT / lanes_path).is_file():
        lanes = read_json(REPO_ROOT / lanes_path)
    broad = score_broad(cases_doc, e3=e3, legacy=legacy, pops=pops, truth=truth, compiled_dirs=compiled_dirs,
                        reference_lanes=lanes)
    if order["status"].startswith("REJECTED"):
        g3 = {**g3, "truth_table": order["status"],
              "reason": "the visual truth table was rejected by the amendment-7 order check (truth_table.status)"}
    verdict = _verdict({"G2": g2["verdict"], "G3": g3["verdict"]})
    ledger_check = None
    if ledger is not None:
        from .. import freeze as e4a_ledger

        chk = e4a_ledger.check_against_ledger(
            ledger, code_fingerprints=e4a_ledger.current_code_fingerprints(REPO_ROOT, e4a_ledger.FROZEN_CODE),
            anchor_sha256=ledger_anchor)
        ledger_check = {"path": str(ledger), "ok": chk["ok"], "mismatches": chk["mismatches"],
                        "waived": chk["waived"], "anchor_checked": chk["anchor_checked"], "entry_id": chk["entry_id"]}
    final = truth is not None
    return {"schema": SCORECARD_SCHEMA, "created_at": now_utc(),
            "label": G2_PHYSICAL_THRESHOLDS["label"],
            "thresholds": {name: thresholds_sha256(doc) for name, doc in (
                ("G2_PHYSICAL", G2_PHYSICAL_THRESHOLDS), ("G2_LINE", G2_LINE_THRESHOLDS), ("G3", G3_THRESHOLDS))},
            "case_document": {"schema": cases_doc.get("schema"), "cases": len(cases_doc["cases"])},
            "arms": {"E3_records": len(e3), "LEGACY_records": len(legacy)},
            "truth_table": order, "final": final, "ledger_check": ledger_check,
            "qualifying": bool(final and ledger_check and ledger_check["ok"] and ledger_check["anchor_checked"]),
            "G2": g2, "G3": g3, "BROAD": broad, "verdict": verdict,
            "verdict_scope": "plan §5 G2 + G3 (BROAD is report-only)"}
