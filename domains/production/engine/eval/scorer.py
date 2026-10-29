"""G1 scorer: per (scan, bend_uid, characteristic) rows and the synthetic scorecard.

Inputs per case: the E2 truth (:mod:`.truth`), what the runner observed
(:class:`EngineRun`), the validated E3 result (or None) and the case's E1
CompiledPart (or None). :func:`score_case` turns them into three row kinds:

- :class:`CharRow` -- one per truth characteristic (backgauge resolved to the
  side E1 attached it to), with the engine's state/decision/value next to the
  truth and the derived error, ``|error| <= U``, false PASS / false FAIL flags;
- :class:`BendRow` -- one per recipe bend: truth vs engine bend state, a
  truth-frame check of the measured line (on its own bend / on another bend =
  confident wrong ID / on no bend) and the bend's *identity* verdict, which
  also uses the registration (:func:`pose_check`), so a confident identity
  flip is caught even when no line is published;
- :class:`CaseRow` -- run outcome, runtime, validity, summary, stage decode, pose.

:func:`build_scorecard` aggregates rows into the plan §5 G1 / amendment 5
gates (thresholds in :data:`.thresholds.G1_THRESHOLDS`), with family-cluster
bootstrap CIs and breakdowns. Honesty rules the scorer enforces:

- An engine failure (non-zero exit, timeout, RSS kill, invalid or inconsistent
  output) is a failure, never a skipped case: its characteristics stay in every
  denominator as not measured, and its values are never scored.
- Only scorable truth is scored (applicable, defined, well defined); a nominal
  is never used as truth.
- False PASS / FAIL are judged on E2 truth (``truth_in_tolerance``), not on the
  engine's U (amendment 5). A PASS is also false on E2's never-PASS list
  (wrong / mirrored parts, out-of-tolerance backgauge sides) and on a truly
  NOT_YET_FORMED bend (its true angle deviation is the whole sweep).
- 'Symmetric' (bend identity ambiguous) is decided by E1's compile-time
  symmetry test of the stage model whose formed set is the truth formed set
  (``ambiguous_bend_groups`` non-empty; amendment 8: AMBIGUOUS is justified
  only by E1's test). Only when no such stage model exists (out-of-order
  formed sets) does the E2 truth flag decide, and the source is recorded.
  An AMBIGUOUS or UNKNOWN bend state on a truth-unambiguous case is an error.
- A gate whose population is empty is NOT_EVALUABLE, never PASS. When even the
  *truth* population is empty in the scored tier (no degraded plans in the
  regression tier, say), the gate is also flagged ``not_applicable_to_tier``;
  ``regression_verdict`` (A1's per-change signal) ignores only those.
- A scorecard is *qualifying* only for the full tier on QUAL seeds with a
  complete amendment-4b MODEL_CARD scanner model, standard run conditions, no
  oracle results, a passing suite composition and an anchored, matching
  freeze ledger. Anything else is labelled DEV / NOT QUALIFYING, whatever its
  numbers.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

from contracts.production.compiled_part import CompiledPartV1, StageModelV1
from contracts.production.inspection_result import CharacteristicResultV1, ShopfloorInspectionResultV1, iter_characteristics

from . import stats
from .engine_output import apply4, decoded_stage, frame_to_raw, stage_reference_points
from .line_metric import pair_metrics
from .thresholds import G1_THRESHOLDS, thresholds_sha256
from .truth import FORMED_LIKE, TruthCase, TruthCharacteristic, scorable

__all__ = [
    "EVAL_VERSION",
    "SCORECARD_SCHEMA",
    "RUN_STATUSES",
    "ENGINE_FAILURE_STATUSES",
    "EngineRun",
    "CharRow",
    "BendRow",
    "CaseRow",
    "CaseScore",
    "score_case",
    "build_scorecard",
    "ambiguity_of",
    "pose_check",
    "scanner_model_gaps",
    "run_condition_reasons",
    "g1_verdict",
    "ORACLE_ENGINE_PREFIX",
]

EVAL_VERSION = "e4a-eval-1.1.0"
SCORECARD_SCHEMA = "sherman_e4_g1_scorecard_v1"
#: Engine versions starting with this prefix come from the harness's truth oracle.
ORACLE_ENGINE_PREFIX = "e4-oracle"

RUN_STATUSES = frozenset({
    "OK",
    "COMPILE_BLOCKED",
    "COMPILE_FAILED",
    "ENGINE_EXIT_2",
    "ENGINE_EXIT_3",
    "ENGINE_EXIT_OTHER",
    "ENGINE_CRASH",
    "ENGINE_TIMEOUT",
    "ENGINE_RSS_LIMIT",
    "RESULT_MISSING",
    "RESULT_INVALID",
    "RESULT_INCONSISTENT",
    "NOT_RUN",
})
#: Statuses that mean the engine itself failed on a compiled part.
ENGINE_FAILURE_STATUSES = frozenset(RUN_STATUSES - {"OK", "COMPILE_BLOCKED", "COMPILE_FAILED", "NOT_RUN"})
_DEGRADED_PLANS = ("TOP_ONLY", "SPARSE", "SINGLE")
#: Bend identity verdicts that count as a confident wrong identity.
_WRONG_IDENTITY = ("WRONG_ID", "WRONG_POSE")


@dataclass(frozen=True)
class EngineRun:
    """What the runner observed for one case."""

    status: str
    detail: str = ""
    wall_s: float | None = None
    peak_rss_bytes: int | None = None
    engine_version: str | None = None

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unknown run status {self.status!r}")


@dataclass(frozen=True)
class CharRow:
    case_id: str
    family: str
    scenario: str
    scan_plan: str
    inspection_stage: str
    native_density: bool
    complete: bool
    full_visibility: bool
    defined_part: bool
    symmetric: bool
    characteristic_id: str
    kind: str
    unit: str
    bend_uid: str | None
    side: str | None
    stage_index: int | None
    truth_bend_state: str | None
    # truth
    scorable: bool
    applicable: bool
    truth: float | None
    nominal: float | None
    true_deviation: float | None
    truth_in_tolerance: bool | None
    tol_plus: float | None
    tol_minus: float | None
    well_defined: bool | None
    truth_undefined_reason: str | None
    must_be_unknown: bool
    never_pass: bool
    never_fail: bool
    ill_defined: bool
    # engine
    engine_state: str  # MEASURED | NOT_MEASURED | NOT_APPLICABLE | ABSENT
    decision: str | None
    reason: str | None
    expected: float | None
    actual: float | None
    deviation: float | None
    uncertainty: float | None
    required: bool | None
    # derived
    error: float | None
    within_u: bool | None
    false_pass: bool
    false_fail: bool
    #: §5 wording: PASS while |true dev| >= tol + U (informational next to the amendment-5 rule).
    false_pass_tol_plus_u: bool
    #: AMBIGUOUS_POSE on an E1-symmetric case: the honest answer, excused from coverage.
    ambiguity_excused: bool
    nominal_disagreement: float | None
    #: Why a PASS is false: OUT_OF_TOLERANCE | NEVER_PASS_LIST | NOT_YET_FORMED (None otherwise).
    false_pass_kind: str | None = None


@dataclass(frozen=True)
class BendRow:
    case_id: str
    family: str
    scenario: str
    scan_plan: str
    complete: bool
    defined_part: bool
    symmetric: bool
    native_density: bool
    bend_uid: str
    sequence_index: int
    truth_state: str | None
    engine_state: str  # contract state or ABSENT
    state_reasons: tuple[str, ...]
    pose_ambiguous: bool
    #: Binary formed/unformed agreement; an engine UNKNOWN/ABSENT is wrong. None when truth is undefined.
    binary_correct: bool | None
    exact_correct: bool | None
    #: ON_OWN_BEND | WRONG_ID | MISPLACED | NO_LINE | NOT_EVALUATED
    line_check: str
    line_offset_mm: float | None
    line_angle_deg: float | None
    wrong_id_target: str | None
    #: The bend row has at least one MEASURED characteristic.
    measured_any: bool = False
    #: Case pose verdict (:func:`pose_check`) repeated per bend.
    pose_check: str = "NOT_EVALUATED"
    #: RIGHT_ID | WRONG_ID | WRONG_POSE | NO_MEASUREMENT | NOT_EVALUATED (lines and registration combined).
    identity: str = "NOT_EVALUATED"
    #: The physical bend whose values this row carries when the identity is wrong (if known).
    identity_target: str | None = None


@dataclass(frozen=True)
class CaseRow:
    case_id: str
    family: str
    scenario: str
    scan_plan: str
    scan_output: str | None
    inspection_stage: str
    density_policy: str
    point_count: int
    complete: bool
    defined_part: bool
    mirrored: bool
    wrong_sku: bool
    mirror_distinguishable_by_rms: bool | None
    symmetric: bool
    ambiguity_source: str
    symmetric_e2: bool
    run_status: str
    run_detail: str
    wall_s: float | None
    peak_rss_bytes: int | None
    engine_version: str | None
    oracle: bool
    validity_state: str | None
    summary_state: str | None
    summary_never: tuple[str, ...]
    summary_never_violation: bool
    pose_ambiguous: bool
    stage_model_id: str | None
    expected_stage_model_id: str | None
    stage_model_correct: bool | None
    inferred_stage: int | None
    truth_formed_count: int | None
    stage_status: str | None
    stage_flags: tuple[str, ...]
    unplanned_creases: int | None
    truth_extra_bends: int
    unmatched_engine_characteristics: tuple[str, ...]
    unmatched_engine_decided: int
    #: PASS decisions on engine characteristics without a truth counterpart.
    unmatched_engine_pass: int = 0
    #: POSE_OK | POSE_SYMMETRIC_FLIP | POSE_WRONG | NOT_EVALUATED (:func:`pose_check`).
    pose_check: str = "NOT_EVALUATED"
    pose_rotation_deg: float | None = None
    pose_rms_mm: float | None = None
    #: ``<stage_model_id>#<index> <kind> rms=<E1 rms>`` of the matched E1 self-map, if any.
    pose_candidate: str | None = None


_TUPLE_FIELDS = {"summary_never", "stage_flags", "unmatched_engine_characteristics", "state_reasons"}


@dataclass
class CaseScore:
    case: CaseRow
    chars: list[CharRow] = field(default_factory=list)
    bends: list[BendRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"case": asdict(self.case), "chars": [asdict(r) for r in self.chars],
                "bends": [asdict(r) for r in self.bends]}

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "CaseScore":
        def build(kind: type, d: Mapping[str, Any]) -> Any:
            names = {f.name for f in fields(kind)}
            unknown = set(d) - names
            if unknown:
                raise ValueError(f"{kind.__name__}: unknown fields {sorted(unknown)} (scored by another eval version)")
            out = {k: (tuple(v) if k in _TUPLE_FIELDS and v is not None else v) for k, v in d.items()}
            return kind(**out)

        case = build(CaseRow, doc["case"])
        chars = [build(CharRow, r) for r in doc.get("chars", ())]
        bends = [build(BendRow, r) for r in doc.get("bends", ())]
        return cls(case, chars, bends)


# ---------------------------------------------------------------------------
# per-case scoring
# ---------------------------------------------------------------------------


def _truth_stage(tc: TruthCase, compiled: CompiledPartV1 | None) -> StageModelV1 | None:
    """E1's stage model whose formed set equals the truth formed set (None if none)."""
    if compiled is None or compiled.status != "COMPILED" or tc.formed_bend_uids is None:
        return None
    formed = set(tc.formed_bend_uids)
    return next((s for s in compiled.stages if set(s.formed_bend_uids) == formed), None)


def ambiguity_of(tc: TruthCase, compiled: CompiledPartV1 | None) -> tuple[bool, str, bool]:
    """``(symmetric, source, e1_e2_disagree)`` for a case (module docstring rule)."""
    if not tc.defined_part:
        return False, "UNDEFINED_PART", False
    stage = _truth_stage(tc, compiled)
    if stage is not None:
        e1 = bool(stage.symmetry.ambiguous_bend_groups)
        return e1, f"E1:{stage.stage_model_id}", e1 != tc.symmetric_e2
    return tc.symmetric_e2, "E2_TRUTH_FALLBACK", False


def _engine_chars(result: ShopfloorInspectionResultV1 | None) -> dict[str, CharacteristicResultV1]:
    if result is None:
        return {}
    return {c.characteristic_id: c for c in iter_characteristics(result.bends, result.general_measurements)}


def _backgauge_side(cid: str, truth: TruthCharacteristic, engine: CharacteristicResultV1 | None,
                    compiled: CompiledPartV1 | None) -> str | None:
    if engine is not None and engine.side is not None:
        return engine.side
    if compiled is not None:
        for nom in compiled.characteristics:
            if nom.characteristic_id == cid and nom.side is not None:
                return nom.side
    if len(truth.per_side) == 1:
        return next(iter(truth.per_side))
    return None


def _limit_on_side(dev: float, tol: tuple[float, float]) -> float:
    plus, minus = tol
    return plus if dev >= 0 else minus


def _char_row(tc: TruthCase, t: TruthCharacteristic, eng: CharacteristicResultV1 | None, *,
              symmetric: bool) -> CharRow:
    bend_state = None
    if t.bend_uid is not None:
        try:
            bend_state = tc.bend(t.bend_uid).state
        except KeyError:
            bend_state = None
    ok_truth = scorable(t)
    state = eng.state if eng is not None else "ABSENT"
    decision = eng.decision if eng is not None else None
    actual = eng.actual if eng is not None and eng.state == "MEASURED" else None
    u = eng.uncertainty if eng is not None and eng.state == "MEASURED" else None
    error = within = None
    if ok_truth and actual is not None and t.truth is not None:
        error = float(actual) - float(t.truth)
        within = bool(u is not None and abs(error) <= float(u) + 1e-9)
    tol = t.tolerance
    never_pass = tc.never_pass_key(t)
    never_fail = t.characteristic_id in tc.never_fail
    fp_kind = None
    if decision == "PASS":
        if ok_truth and t.truth_in_tolerance is False:
            fp_kind = "OUT_OF_TOLERANCE"
        elif never_pass:
            fp_kind = "NEVER_PASS_LIST"
        elif never_fail:
            fp_kind = "NOT_YET_FORMED"  # a PASS on a flat bend: the true deviation is the whole sweep
    false_fail = bool(ok_truth and decision == "FAIL" and t.truth_in_tolerance is True)
    fp_s5 = False
    if ok_truth and decision == "PASS" and t.true_deviation is not None and tol is not None and u is not None:
        fp_s5 = abs(t.true_deviation) >= _limit_on_side(t.true_deviation, tol) + float(u)
    measured_dev = eng.deviation if eng is not None and eng.state == "MEASURED" else None
    nominal_dis = None
    if eng is not None and eng.expected is not None and t.nominal is not None:
        nominal_dis = float(eng.expected) - float(t.nominal)
    excused = bool(symmetric and state == "NOT_MEASURED" and eng is not None and eng.reason == "AMBIGUOUS_POSE")
    return CharRow(
        case_id=tc.case_id, family=tc.family, scenario=tc.scenario, scan_plan=tc.scan_plan,
        inspection_stage=tc.inspection_stage, native_density=tc.native_density, complete=tc.complete,
        full_visibility=tc.full_visibility, defined_part=tc.defined_part, symmetric=symmetric,
        characteristic_id=t.characteristic_id, kind=t.kind, unit=t.unit, bend_uid=t.bend_uid, side=t.side,
        stage_index=t.stage_index, truth_bend_state=bend_state,
        scorable=ok_truth, applicable=t.applicable, truth=t.truth, nominal=t.nominal, true_deviation=t.true_deviation,
        truth_in_tolerance=t.truth_in_tolerance, tol_plus=None if tol is None else tol[0],
        tol_minus=None if tol is None else tol[1], well_defined=t.well_defined,
        truth_undefined_reason=t.truth_undefined_reason,
        must_be_unknown=t.characteristic_id in tc.must_be_unknown, never_pass=never_pass,
        never_fail=never_fail, ill_defined=tc.ill_defined_key(t),
        engine_state=state, decision=decision, reason=eng.reason if eng is not None else None,
        expected=eng.expected if eng is not None else None, actual=actual, deviation=measured_dev, uncertainty=u,
        required=eng.required if eng is not None else None,
        error=error, within_u=within, false_pass=fp_kind is not None, false_fail=false_fail,
        false_pass_tol_plus_u=fp_s5, ambiguity_excused=excused, nominal_disagreement=nominal_dis,
        false_pass_kind=fp_kind,
    )


def _line_check(tc: TruthCase, result: ShopfloorInspectionResultV1, row: Any, cfg: Mapping[str, Any]
                ) -> tuple[str, float | None, float | None, str | None]:
    line = row.display.measured_polyline_mm
    if line is None:
        return "NO_LINE", None, None, None
    if not tc.defined_part or tc.raw_to_cad is None:
        return "NOT_EVALUATED", None, None, None
    try:
        raw = frame_to_raw(result, np.asarray([line[0], line[-1]], dtype=np.float64))
    except ValueError:
        return "NOT_EVALUATED", None, None, None
    pts = raw @ tc.raw_to_cad[:3, :3].T + tc.raw_to_cad[:3, 3]

    def check(bend: Any) -> tuple[bool, float, float]:
        m = pair_metrics(bend.actual_oml, pts)
        limit = float(cfg["offset_floor_mm"])
        if cfg.get("offset_adds_ri_plus_t", True):
            limit = max(limit, bend.inner_radius_mm + bend.thickness_mm)
        ok = m["axis_angle_deg"] <= float(cfg["angle_deg_max"]) and m["transverse_offset_mm"] <= limit
        return ok, m["transverse_offset_mm"], m["axis_angle_deg"]

    own = tc.bend(row.bend_uid)
    own_res = check(own) if own.actual_oml is not None else None
    if own_res is not None and own_res[0]:
        return "ON_OWN_BEND", own_res[1], own_res[2], None
    best: tuple[float, str, float] | None = None
    for other in tc.bends:
        if other.bend_uid == row.bend_uid or other.actual_oml is None:
            continue
        ok, off, ang = check(other)
        if ok and (best is None or off < best[0]):
            best = (off, other.bend_uid, ang)
    if best is not None:
        return "WRONG_ID", best[0], best[2], best[1]
    if own_res is None:
        return "MISPLACED", None, None, None
    return "MISPLACED", own_res[1], own_res[2], None


def _rotation_deg(m: np.ndarray) -> float:
    """Rotation angle of the orthonormalised 3x3 part of ``m`` (degrees)."""
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64)[:3, :3])
    r = u @ vt
    return math.degrees(math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def pose_check(tc: TruthCase, result: ShopfloorInspectionResultV1 | None, compiled: CompiledPartV1 | None,
               cfg: Mapping[str, Any] = G1_THRESHOLDS["pose_check"]) -> dict[str, Any]:
    """Identity of the engine's registration against the E2 registration truth.

    ``D = raw_to_frame . diag(unit_factor) . inv(truth.raw_to_cad)`` maps the true
    part (cad_model_mm) to where the engine believes it is. Verdicts:
    ``POSE_OK`` (D is about the identity), ``POSE_SYMMETRIC_FLIP`` (D matches an
    E1 ROTATION self-map of the stage model; ``permutation`` says which bend lands
    where), ``POSE_WRONG`` (anything else) or ``NOT_EVALUATED`` (no result, an
    undefined part, no transform, no registration truth or no stage model)."""
    out: dict[str, Any] = {"verdict": "NOT_EVALUATED", "rotation_deg": None, "rms_mm": None, "candidate": None,
                           "permutation": None}
    if result is None or compiled is None or compiled.status != "COMPILED" or not tc.defined_part:
        return out
    reg = result.inspection_validity.registration
    if reg.raw_to_frame is None or tc.raw_to_cad is None:
        return out
    stage = _truth_stage(tc, compiled) or decoded_stage(result, compiled)
    pts = stage_reference_points(compiled, stage)
    if len(pts) < 3:
        return out
    scale = np.diag([float(reg.unit_factor)] * 3 + [1.0])
    d = np.asarray(reg.raw_to_frame, dtype=np.float64).reshape(4, 4) @ scale @ np.linalg.inv(tc.raw_to_cad)
    moved = apply4(d, pts)
    rot = _rotation_deg(d)
    rms = float(np.sqrt(np.mean(np.sum((moved - pts) ** 2, axis=1))))
    out.update(rotation_deg=round(rot, 4), rms_mm=round(rms, 4))
    if rot <= float(cfg["rotation_deg_max"]) and rms <= float(cfg["rms_displacement_mm_max"]):
        out["verdict"] = "POSE_OK"
        return out
    best: tuple[float, int, Any] | None = None
    for i, cand in enumerate(stage.symmetry.candidates):
        if cand.kind not in cfg["candidate_kinds"]:
            continue
        c = np.asarray(cand.transform, dtype=np.float64).reshape(4, 4)
        rot_c = _rotation_deg(d[:3, :3] @ c[:3, :3].T)
        rms_c = float(np.sqrt(np.mean(np.sum((moved - apply4(c, pts)) ** 2, axis=1))))
        if rot_c <= float(cfg["candidate_rotation_deg_max"]) and rms_c <= float(cfg["candidate_rms_mm_max"]):
            if best is None or rms_c < best[0]:
                best = (rms_c, i, cand)
    if best is not None:
        _, i, cand = best
        out.update(verdict="POSE_SYMMETRIC_FLIP", permutation=dict(cand.bend_permutation),
                   candidate=f"{stage.stage_model_id}#{i} {cand.kind} rms={cand.rms_mm:.3f}")
        return out
    out["verdict"] = "POSE_WRONG"
    return out


def _identity(line_check: str, line_target: str | None, pose: Mapping[str, Any], uid: str,
              measured_any: bool) -> tuple[str, str | None]:
    """Combine the truth-frame line check with the registration identity check."""
    if line_check == "WRONG_ID":
        return "WRONG_ID", line_target
    if not measured_any:
        return "NO_MEASUREMENT", None
    verdict = pose["verdict"]
    if verdict == "POSE_SYMMETRIC_FLIP":
        perm = pose.get("permutation") or {}
        if uid not in perm:
            return "WRONG_POSE", None
        inverse = {v: k for k, v in perm.items()}
        source = inverse.get(uid, uid)
        return ("RIGHT_ID", None) if source == uid else ("WRONG_ID", source)
    if verdict == "POSE_WRONG":
        return "WRONG_POSE", None
    if verdict == "POSE_OK" or line_check == "ON_OWN_BEND":
        return "RIGHT_ID", None
    return "NOT_EVALUATED", None


def score_case(
    tc: TruthCase,
    run: EngineRun,
    result: ShopfloorInspectionResultV1 | None,
    compiled: CompiledPartV1 | None,
    *,
    thresholds: Mapping[str, Any] = G1_THRESHOLDS,
) -> CaseScore:
    """Rows for one case (module docstring). ``result`` is used only when
    ``run.status == "OK"``; anything else is an engine or compile failure."""
    use = result if run.status == "OK" else None
    symmetric, source, _disagree = ambiguity_of(tc, compiled)
    engine = _engine_chars(use)
    reg = use.inspection_validity.registration if use is not None else None
    pose_ambiguous = bool(reg.pose_ambiguous) if reg is not None else False
    pose = pose_check(tc, use, compiled, thresholds["pose_check"])
    rows: list[CharRow] = []
    seen: set[str] = set()
    for cid, t in tc.characteristics.items():
        eng = engine.get(cid)
        if t.kind == "BACKGAUGE_LENGTH":
            t = t.for_side(_backgauge_side(cid, t, eng, compiled))
        rows.append(_char_row(tc, t, eng, symmetric=symmetric))
        seen.add(cid)
    unmatched = sorted(set(engine) - seen)
    unmatched_decided = sum(1 for cid in unmatched if engine[cid].decision in ("PASS", "FAIL"))
    unmatched_pass = sum(1 for cid in unmatched if engine[cid].decision == "PASS")

    bend_rows: list[BendRow] = []
    eng_rows = {r.bend_uid: r for r in use.bends} if use is not None else {}
    for tb in tc.bends:
        er = eng_rows.get(tb.bend_uid)
        estate = er.state if er is not None else "ABSENT"
        if tb.state is None:
            binary = exact = None
        else:
            truth_formed = tb.state in FORMED_LIKE
            if estate in ("UNKNOWN", "ABSENT"):
                binary = False
            else:
                binary = (estate in FORMED_LIKE) == truth_formed
            exact = estate == tb.state
        if er is not None and use is not None:
            lc, off, ang, target = _line_check(tc, use, er, thresholds["line_check"])
        else:
            lc, off, ang, target = "NO_LINE", None, None, None
        measured_any = bool(er is not None and any(c.state == "MEASURED" for c in er.characteristics))
        if not tc.defined_part:
            identity, identity_target = "NOT_EVALUATED", None
        else:
            identity, identity_target = _identity(lc, target, pose, tb.bend_uid, measured_any)
        bend_rows.append(BendRow(
            case_id=tc.case_id, family=tc.family, scenario=tc.scenario, scan_plan=tc.scan_plan, complete=tc.complete,
            defined_part=tc.defined_part, symmetric=symmetric, native_density=tc.native_density,
            bend_uid=tb.bend_uid, sequence_index=tb.sequence_index, truth_state=tb.state, engine_state=estate,
            state_reasons=tuple(er.state_reasons) if er is not None else (), pose_ambiguous=pose_ambiguous,
            binary_correct=binary, exact_correct=exact, line_check=lc, line_offset_mm=off, line_angle_deg=ang,
            wrong_id_target=target, measured_any=measured_any, pose_check=pose["verdict"], identity=identity,
            identity_target=identity_target,
        ))

    sd = use.stage_decision if use is not None else None
    summary = use.summary.state if use is not None else None
    stage_ok = None
    if sd is not None and tc.expected_stage_model_id is not None:
        stage_ok = sd.stage_model_id == tc.expected_stage_model_id
    case = CaseRow(
        case_id=tc.case_id, family=tc.family, scenario=tc.scenario, scan_plan=tc.scan_plan, scan_output=tc.scan_output,
        inspection_stage=tc.inspection_stage, density_policy=tc.density_policy, point_count=tc.point_count,
        complete=tc.complete, defined_part=tc.defined_part, mirrored=tc.mirrored, wrong_sku=tc.wrong_sku,
        mirror_distinguishable_by_rms=tc.mirror_distinguishable_by_rms, symmetric=symmetric, ambiguity_source=source,
        symmetric_e2=tc.symmetric_e2, run_status=run.status, run_detail=run.detail[:500], wall_s=run.wall_s,
        peak_rss_bytes=run.peak_rss_bytes, engine_version=run.engine_version or (
            use.engine.engine_version if use is not None else None),
        oracle=bool((run.engine_version or (use.engine.engine_version if use is not None else "") or "")
                    .startswith(ORACLE_ENGINE_PREFIX)),
        validity_state=use.inspection_validity.state if use is not None else None,
        summary_state=summary, summary_never=tuple(sorted(tc.summary_never)),
        summary_never_violation=bool(summary is not None and summary in tc.summary_never),
        pose_ambiguous=pose_ambiguous, stage_model_id=sd.stage_model_id if sd is not None else None,
        expected_stage_model_id=tc.expected_stage_model_id, stage_model_correct=stage_ok,
        inferred_stage=sd.inferred_stage if sd is not None else None, truth_formed_count=tc.formed_count,
        stage_status=sd.status if sd is not None else None, stage_flags=tuple(sd.flags) if sd is not None else (),
        unplanned_creases=len(use.unplanned_creases) if use is not None else None,
        truth_extra_bends=tc.extra_bend_count, unmatched_engine_characteristics=tuple(unmatched),
        unmatched_engine_decided=unmatched_decided, unmatched_engine_pass=unmatched_pass,
        pose_check=pose["verdict"], pose_rotation_deg=pose["rotation_deg"], pose_rms_mm=pose["rms_mm"],
        pose_candidate=pose["candidate"],
    )
    return CaseScore(case, rows, bend_rows)


# ---------------------------------------------------------------------------
# qualification helpers
# ---------------------------------------------------------------------------


def scanner_model_gaps(manifest: Mapping[str, Any] | None,
                       thresholds: Mapping[str, Any] = G1_THRESHOLDS) -> dict[str, Any]:
    """Amendment-4b completeness of the tier's scanner model.

    Returns ``{"provenance", "field_status", "unmeasured_required_fields"}``: the
    required fields (``thresholds["scanner_model"]``) whose status in the card's
    machine-readable FIELD_STATUS (:func:`.scanner_card.field_status`) is not a
    measured status. A card without FIELD_STATUS counts every field as unmeasured."""
    from .scanner_card import field_status

    model = dict((manifest or {}).get("scanner_model") or {})
    cfg = thresholds["scanner_model"]
    required = [f for group in cfg["amendment_4b_quantities"].values() for f in group]
    status = field_status(model) if model else None
    ok = set(cfg["measured_statuses"])
    gaps = [f for f in required if (status or {}).get(f) not in ok]
    return {"provenance": model.get("provenance"), "field_status": status, "unmeasured_required_fields": gaps}


def run_condition_reasons(conditions: Mapping[str, Any] | None,
                          thresholds: Mapping[str, Any] = G1_THRESHOLDS) -> list[str]:
    """Non-qualifying reasons for run conditions that differ from the standard ones
    (``thresholds["run_conditions"]``); unrecorded conditions never qualify."""
    if not conditions:
        return ["RUN_CONDITIONS_NOT_RECORDED"]
    std = thresholds["run_conditions"]
    reasons = []
    limits = (("engine_timeout_s", float), ("engine_rss_limit_bytes", int), ("engine_time_budget_s", float))
    if any(conditions.get(k) is None or typ(conditions[k]) != typ(std[k]) for k, typ in limits):
        reasons.append("NON_STANDARD_LIMITS")
    if list(conditions.get("engine_argv_tail") or ()) != list(std["engine_argv_tail"]):
        reasons.append("NON_STANDARD_ENGINE_CMD")
    if conditions.get("engine_extra_env_keys"):
        reasons.append("NON_STANDARD_ENGINE_ENV")
    return reasons


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _gate(gid: str, description: str, population: str, statistic: str, value: Any, threshold: Any,
          comparator: str, n: int, *, verdict: str | None = None, ci: Mapping[str, Any] | None = None,
          details: Mapping[str, Any] | None = None, informational: bool = False,
          truth_n: int | None = None) -> dict[str, Any]:
    if verdict is None:
        if value is None or n == 0:
            verdict = "NOT_EVALUABLE"
        else:
            ops = {"<=": lambda a, b: a <= b + 1e-12, ">=": lambda a, b: a >= b - 1e-12, "==": lambda a, b: a == b,
                   "in": lambda a, b: b[0] - 1e-12 <= a <= b[1] + 1e-12}
            verdict = "PASS" if ops[comparator](value, threshold) else "FAIL"
    return {"id": gid, "description": description, "population": population, "statistic": statistic,
            "value": _clean(value), "threshold": threshold, "comparator": comparator, "n": n, "verdict": verdict,
            "ci": dict(ci) if ci else None, "details": dict(details) if details else {},
            "informational": informational, "truth_population_n": truth_n,
            "not_applicable_to_tier": bool(verdict == "NOT_EVALUABLE" and truth_n == 0)}


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return stats.round_or_none(value, 6)
    return value


def _by_family(rows: Sequence[Any], fn) -> dict[str, Any]:
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r.family, []).append(r)
    return {k: fn(v) for k, v in out.items()}


def _boot(rows: Sequence[Any], stat, cfg: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.family, []).append(r)
    return stats.cluster_bootstrap(
        groups, lambda parts: stat([r for part in parts for r in part]),
        resamples=int(cfg["resamples"]), seed=int(cfg["seed"]), ci=float(cfg["ci"]))


def _rate_of(rows: Sequence[Any], pred) -> float | None:
    return stats.rate(sum(1 for r in rows if pred(r)), len(rows))


def build_scorecard(
    scores: Sequence[CaseScore],
    *,
    tier: str,
    manifest: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] = G1_THRESHOLDS,
    provenance: Mapping[str, Any] | None = None,
    ledger_check: Mapping[str, Any] | None = None,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """The tier scorecard (module docstring). ``tier`` is the E2 tier name
    (``regression`` / ``full`` / ``runtime`` / ``smoke``). ``provenance`` should
    carry ``run_conditions`` (:func:`.runner.run_conditions`); ``ledger_check``
    is :func:`.freeze.check_against_ledger` output."""
    th = thresholds
    boot_cfg = th["bootstrap"]
    cases = [s.case for s in scores]
    chars = [r for s in scores for r in s.chars]
    bends = [r for s in scores for r in s.bends]
    native_cases = [c for c in cases if c.density_policy == "NATIVE_SPACING"]
    accuracy_tier = bool(native_cases)
    gates: list[dict[str, Any]] = []

    def boot(rows, stat):
        return _boot(rows, stat, boot_cfg) if bootstrap else None

    native_chars = [r for r in chars if r.native_density]
    scor = [r for r in native_chars if r.scorable]
    meas = [r for r in scor if r.engine_state == "MEASURED"]

    # -- accuracy (complete parts, full visibility) --------------------------------
    for kind, lim in th["accuracy_complete_full_visibility"].items():
        truth_rows = [r for r in scor if r.kind == kind and r.complete and r.full_visibility]
        rows = [r for r in meas if r.kind == kind and r.complete and r.full_visibility and r.error is not None]
        errs = [r.error for r in rows]
        pop = f"MEASURED scorable {kind} on complete parts, FULL plan, native density"
        if lim.get("abs_bias_max") is not None:
            bias = stats.mean(errs)
            gates.append(_gate(
                f"G1.ACC.{kind}.BIAS", f"|bias| of {kind} error (actual - truth)", pop, "|mean(error)|",
                None if bias is None else abs(bias), lim["abs_bias_max"], "<=", len(rows),
                ci=boot(rows, lambda rs: stats.mean([x.error for x in rs])), truth_n=len(truth_rows),
                details={"signed_bias": stats.round_or_none(bias), "mae": stats.round_or_none(stats.mae(errs)),
                         "unit": lim["unit"],
                         "by_family": _by_family(rows, lambda rs: stats.round_or_none(stats.mean([x.error for x in rs])))}))
        gates.append(_gate(
            f"G1.ACC.{kind}.P95", f"P95 |error| of {kind}", pop, "p95(|error|)", stats.p95_abs(errs),
            lim["p95_abs_error_max"], "<=", len(rows), truth_n=len(truth_rows),
            ci=boot(rows, lambda rs: stats.p95_abs([x.error for x in rs])),
            details={"mae": stats.round_or_none(stats.mae(errs)), "max_abs": stats.round_or_none(
                max((abs(e) for e in errs), default=None) if errs else None), "unit": lim["unit"],
                     "by_family": _by_family(rows, lambda rs: stats.round_or_none(stats.p95_abs([x.error for x in rs])))}))

    # -- coverage ------------------------------------------------------------------
    cov_rows = [r for r in scor if r.full_visibility and not r.ambiguity_excused]
    cov_raw = [r for r in scor if r.full_visibility]
    gates.append(_gate(
        "G1.COVERAGE", "share of scorable characteristics MEASURED (realistic occlusion)",
        "scorable characteristics of defined parts, FULL plan, native density; failed runs included as not measured; "
        "AMBIGUOUS_POSE on E1-symmetric cases excused", "measured / scorable",
        _rate_of(cov_rows, lambda r: r.engine_state == "MEASURED"), th["coverage_min"], ">=", len(cov_rows),
        ci=boot(cov_rows, lambda rs: _rate_of(rs, lambda r: r.engine_state == "MEASURED")), truth_n=len(cov_raw),
        details={"raw_rate_without_excuse": stats.round_or_none(_rate_of(cov_raw, lambda r: r.engine_state == "MEASURED")),
                 "excused_ambiguous": sum(1 for r in cov_raw if r.ambiguity_excused),
                 "by_plan": {p: stats.round_or_none(_rate_of(
                     [r for r in scor if r.scan_plan == p and not r.ambiguity_excused],
                     lambda r: r.engine_state == "MEASURED"))
                     for p in sorted({r.scan_plan for r in chars})},
                 "by_kind": {k: stats.round_or_none(_rate_of([r for r in cov_rows if r.kind == k],
                                                             lambda r: r.engine_state == "MEASURED"))
                             for k in sorted({r.kind for r in cov_rows})},
                 "not_measured_reasons": dict(Counter(r.reason or r.engine_state for r in cov_rows
                                                      if r.engine_state != "MEASURED").most_common())}))

    # -- false PASS / false FAIL ------------------------------------------------------
    fp = [r for r in native_chars if r.false_pass]
    gates.append(_gate(
        "G1.FALSE_PASS", "PASS while the truth says it must not pass (E2 truth, never the engine's U)",
        "every characteristic, native density, every plan: scorable and out of tolerance, on E2's never-PASS list, "
        "or on a truly NOT_YET_FORMED bend", "count", len(fp), th["false_pass_max_count"], "<=",
        len(native_chars), truth_n=len(native_chars),
        details={"examples": [f"{r.case_id}:{r.characteristic_id}:{r.false_pass_kind}" for r in fp[:20]],
                 "by_kind": dict(Counter(r.false_pass_kind for r in fp).most_common()),
                 "s5_tol_plus_u_count": sum(1 for r in native_chars if r.false_pass_tol_plus_u)}))
    ff_truth = [r for r in scor if r.truth_in_tolerance is True]
    ff_pop = [r for r in meas if r.truth_in_tolerance is True and r.decision in ("PASS", "FAIL", "REVIEW")]
    gates.append(_gate(
        "G1.FALSE_FAIL", "FAIL while the true deviation is within tolerance",
        "MEASURED scorable characteristics with truth in tolerance, native density", "false FAIL / population",
        _rate_of(ff_pop, lambda r: r.false_fail), th["false_fail_max_rate"], "<=", len(ff_pop), truth_n=len(ff_truth),
        details={"count": sum(1 for r in ff_pop if r.false_fail),
                 "examples": [f"{r.case_id}:{r.characteristic_id}" for r in ff_pop if r.false_fail][:20]}))

    # -- U calibration and median U ---------------------------------------------------
    uc = [r for r in meas if r.within_u is not None]
    gates.append(_gate(
        "G1.U_CALIBRATION", "|actual - truth| <= U", "MEASURED scorable characteristics, native density",
        "share within U", _rate_of(uc, lambda r: r.within_u), th["u_calibration_min_rate"], ">=", len(uc),
        ci=boot(uc, lambda rs: _rate_of(rs, lambda r: r.within_u)), truth_n=len(scor),
        details={"by_kind": {k: stats.round_or_none(_rate_of([r for r in uc if r.kind == k], lambda r: r.within_u))
                             for k in sorted({r.kind for r in uc})}}))
    for unit, lim in th["median_u_full_visibility_max"].items():
        rows = [r for r in meas if r.full_visibility and r.unit == unit and r.uncertainty is not None]
        gates.append(_gate(
            f"G1.MEDIAN_U.{unit}", f"median U ({unit}) at full visibility", f"MEASURED scorable {unit} characteristics, "
            "FULL plan, native density", "median(U)", stats.median([r.uncertainty for r in rows]), lim, "<=", len(rows),
            truth_n=sum(1 for r in scor if r.full_visibility and r.unit == unit),
            details={"by_kind": {k: stats.round_or_none(stats.median([r.uncertainty for r in rows if r.kind == k]))
                                 for k in sorted({r.kind for r in rows})}}))

    # -- decisiveness ------------------------------------------------------------------
    dc = th["decisiveness"]

    def far_or_near(r: CharRow) -> bool:
        if r.true_deviation is None or r.tol_plus is None or r.tol_minus is None:
            return False
        lim = _limit_on_side(r.true_deviation, (r.tol_plus, r.tol_minus))
        if lim <= 0:
            return False
        ratio = abs(r.true_deviation) / lim
        return ratio <= dc["near_fraction"] or ratio >= dc["far_fraction"]

    dec = [r for r in meas if r.decision in ("PASS", "FAIL", "REVIEW") and far_or_near(r)]
    gates.append(_gate(
        "G1.DECISIVENESS", "PASS or FAIL (not REVIEW) when |true dev| <= 0.5 tol or >= 1.5 tol",
        "MEASURED scorable characteristics with a guard-band decision, native density", "decided share",
        _rate_of(dec, lambda r: r.decision in ("PASS", "FAIL")), dc["min_rate"], ">=", len(dec),
        truth_n=sum(1 for r in scor if far_or_near(r)),
        ci=boot(dec, lambda rs: _rate_of(rs, lambda r: r.decision in ("PASS", "FAIL")))))

    # -- slope -------------------------------------------------------------------------
    sc = th["slope"]

    def is_large(r: CharRow) -> bool:
        return (r.true_deviation is not None and r.tol_plus is not None and r.tol_minus is not None
                and abs(r.true_deviation) >= sc["large_fraction_of_tol"] * _limit_on_side(
                    r.true_deviation, (r.tol_plus, r.tol_minus)))

    for kind in sc["kinds"]:
        truth_rows = [r for r in scor if r.kind == kind and r.true_deviation is not None]
        truth_ok = len(truth_rows) >= sc["min_points"] and sum(1 for r in truth_rows if is_large(r)) >= sc["min_large_points"]
        rows = [r for r in meas if r.kind == kind and r.true_deviation is not None and r.deviation is not None]
        large = [r for r in rows if is_large(r)]
        fit = stats.ols_slope([r.true_deviation for r in rows], [r.deviation for r in rows])
        enough = len(rows) >= sc["min_points"] and len(large) >= sc["min_large_points"]
        gates.append(_gate(
            f"G1.SLOPE.{kind}", f"slope of measured vs true deviation ({kind})",
            f"MEASURED scorable {kind}, native density", "OLS slope", fit[0] if fit else None,
            [sc["min"], sc["max"]], "in", len(rows), verdict=None if enough else "NOT_EVALUABLE",
            truth_n=len(truth_rows) if truth_ok else 0,
            ci=boot(rows, lambda rs: (stats.ols_slope([x.true_deviation for x in rs], [x.deviation for x in rs]) or (None,))[0]),
            details={"intercept": stats.round_or_none(fit[1]) if fit else None, "large_points": len(large),
                     "truth_large_points": sum(1 for r in truth_rows if is_large(r))},
            informational=kind == "BACKGAUGE_LENGTH"))

    # -- bend states and identities ------------------------------------------------------
    st_truth = [b for b in bends if b.native_density and b.defined_part and not b.symmetric and not b.complete
                and b.truth_state is not None]
    st_rows = [b for b in st_truth if b.binary_correct is not None]
    gates.append(_gate(
        "G1.STAGE_STATE", "per-bend formed/unformed state correct (UNKNOWN/AMBIGUOUS = error)",
        "bends of truth-unambiguous incomplete-part cases, native density; failed runs count as wrong",
        "correct / bends", _rate_of(st_rows, lambda b: b.binary_correct), th["stage_state_accuracy_min"], ">=",
        len(st_rows), ci=boot(st_rows, lambda rs: _rate_of(rs, lambda b: b.binary_correct)), truth_n=len(st_truth),
        details={"all_unambiguous_cases_rate": stats.round_or_none(_rate_of(
            [b for b in bends if b.native_density and b.defined_part and not b.symmetric and b.binary_correct is not None],
            lambda b: b.binary_correct)),
                 "exact_vocabulary_rate": stats.round_or_none(_rate_of(st_rows, lambda b: b.exact_correct)),
                 "unknown_or_ambiguous_errors": sum(1 for b in st_rows if b.engine_state in ("UNKNOWN", "ABSENT")),
                 "confusion": dict(Counter(f"{b.truth_state}->{b.engine_state}" for b in st_rows).most_common())}))
    sym_bends = [b for b in bends if b.native_density and b.symmetric]
    wrong_sym = [b for b in sym_bends if b.identity in _WRONG_IDENTITY]
    undetermined = [b for b in sym_bends if b.identity == "NOT_EVALUATED" and b.measured_any]
    if wrong_sym:
        sym_verdict = "FAIL"
    elif not sym_bends or undetermined:
        sym_verdict = "NOT_EVALUABLE"  # measured bends without any identity evidence are never a PASS
    else:
        sym_verdict = "PASS"
    gates.append(_gate(
        "G1.SYMMETRIC_NO_WRONG_ID", "symmetric cases yield AMBIGUOUS or the right ids, never a confident wrong id",
        "bends of E1-symmetric cases, native density; identity from the truth-frame line check and the "
        "registration (pose check), so withheld lines cannot hide a flip", "confident wrong ids",
        len(wrong_sym), 0, "<=", len(sym_bends), verdict=sym_verdict, truth_n=len(sym_bends),
        details={"symmetric_cases": len({b.case_id for b in sym_bends}),
                 "ambiguous_pose_cases": len({b.case_id for b in sym_bends if b.pose_ambiguous}),
                 "identity": dict(Counter(b.identity for b in sym_bends).most_common()),
                 "pose_checks": dict(Counter(b.pose_check for b in sym_bends).most_common()),
                 "measured_without_identity_evidence": len(undetermined),
                 "examples": [f"{b.case_id}:{b.bend_uid}:{b.identity}->{b.identity_target}" for b in wrong_sym[:20]]}))
    all_line = [b for b in bends if b.native_density and b.line_check in ("ON_OWN_BEND", "WRONG_ID", "MISPLACED")]
    gates.append(_gate(
        "G1.LINE_ON_OWN_BEND", "measured lines land on their own bend in the truth frame (automated LANDS/TAG_OK)",
        "bends with a measured line, native density", "share on own bend",
        _rate_of(all_line, lambda b: b.line_check == "ON_OWN_BEND"), 1.0, ">=", len(all_line), informational=True,
        details={"wrong_id": sum(1 for b in all_line if b.line_check == "WRONG_ID"),
                 "misplaced": sum(1 for b in all_line if b.line_check == "MISPLACED")}))
    posed = [c for c in native_cases if c.defined_part and c.pose_check != "NOT_EVALUATED"]
    gates.append(_gate(
        "G1.POSE_CHECK", "the engine registration identifies the part (POSE_OK) on defined cases",
        "defined cases with a result and registration truth, native density", "share POSE_OK",
        _rate_of(posed, lambda c: c.pose_check == "POSE_OK"), 1.0, ">=", len(posed), informational=True,
        details={"verdicts": dict(Counter(c.pose_check for c in posed).most_common()),
                 "not_ok": [f"{c.case_id}:{c.pose_check}:{c.pose_candidate or ''}"
                            f":rot={c.pose_rotation_deg}:rms={c.pose_rms_mm}" for c in posed
                            if c.pose_check != "POSE_OK"][:20]}))
    amb_err = [c for c in native_cases if c.pose_ambiguous and c.defined_part and not c.symmetric]
    gates.append(_gate(
        "G1.AMBIGUOUS_ON_UNAMBIGUOUS", "AMBIGUOUS_POSE on truth-unambiguous cases (counted as state errors)",
        "defined cases that are not E1-symmetric", "cases", len(amb_err), 0, "<=",
        sum(1 for c in native_cases if c.defined_part and not c.symmetric), informational=True,
        details={"examples": [c.case_id for c in amb_err[:20]]}))

    # -- backgauge detection -----------------------------------------------------------------
    bd = th["backgauge_detection"]
    bg = [r for r in scor if r.kind == "BACKGAUGE_LENGTH"
          and r.true_deviation is not None and abs(r.true_deviation) >= bd["min_abs_dev_mm"]]
    gates.append(_gate(
        "G1.BACKGAUGE_DETECTION", f"injected backgauge errors >= {bd['min_abs_dev_mm']} mm decided FAIL or REVIEW",
        "scorable backgauge characteristics (E1 side), native density", "detected share",
        _rate_of(bg, lambda r: r.decision in ("FAIL", "REVIEW")), bd["min_rate"], ">=", len(bg), truth_n=len(bg),
        details={"fail_share": stats.round_or_none(_rate_of(bg, lambda r: r.decision == "FAIL")),
                 "missed": [f"{r.case_id}:{r.characteristic_id}:{r.decision or r.engine_state}"
                            for r in bg if r.decision not in ("FAIL", "REVIEW")][:20]}))

    # -- hard never-rules ------------------------------------------------------------------------
    nyf = [r for r in chars if r.never_fail]
    nyf_bad = [r for r in nyf if r.decision == "FAIL"]
    gates.append(_gate(
        "G1.NOT_YET_FORMED_NEVER_FAIL", "no NOT_YET_FORMED bend characteristic is ever FAIL",
        "characteristics of truly NOT_YET_FORMED bends, every case", "FAIL count", len(nyf_bad), 0, "<=", len(nyf),
        verdict=None if nyf else "NOT_EVALUABLE", truth_n=len(nyf),
        details={"examples": [f"{r.case_id}:{r.characteristic_id}" for r in nyf_bad[:20]],
                 "decided_pass_counted_as_false_pass": sum(1 for r in nyf if r.decision == "PASS")}))
    wm = [r for r in chars if not r.defined_part]
    wm_pass = [r for r in wm if r.decision == "PASS"]
    wm_extra = sum(c.unmatched_engine_pass for c in cases if not c.defined_part)
    gates.append(_gate(
        "G1.WRONG_OR_MIRRORED_NEVER_PASS", "wrong-SKU and mirrored parts never PASS",
        "every characteristic of wrong-SKU / mirrored cases (with or without a truth counterpart)", "PASS count",
        len(wm_pass) + wm_extra, 0, "<=", len(wm), verdict=None if wm else "NOT_EVALUABLE", truth_n=len(wm),
        details={"examples": [f"{r.case_id}:{r.characteristic_id}" for r in wm_pass[:20]],
                 "pass_without_truth_counterpart": wm_extra,
                 "validity_states": dict(Counter(f"{c.scenario}:{c.validity_state}" for c in cases
                                                 if not c.defined_part).most_common()),
                 "mirrored_by_rms_distinguishability": dict(Counter(
                     f"distinguishable={c.mirror_distinguishable_by_rms}:{c.validity_state}" for c in cases if c.mirrored
                 ).most_common())}))
    sn = [c for c in cases if c.summary_never_violation]
    gates.append(_gate(
        "G1.SUMMARY_NEVER", "no forbidden summary (missing/extra bend in FINAL never ALL_MEASURED_IN_TOLERANCE, ...)",
        "every case with a result", "violations", len(sn), 0, "<=", sum(1 for c in cases if c.summary_state is not None),
        verdict=None if any(c.summary_state is not None for c in cases) else "NOT_EVALUABLE", truth_n=len(cases),
        details={"examples": [f"{c.case_id}:{c.summary_state}" for c in sn[:20]],
                 "missing_extra_final_cases": sum(1 for c in cases if c.scenario in ("MISSING_BEND", "EXTRA_BEND"))}))
    mbu = [r for r in chars if r.must_be_unknown]
    mbu_bad = [r for r in mbu if r.engine_state == "MEASURED"]
    gates.append(_gate(
        "G1.MUST_BE_UNKNOWN", "characteristics without defined truth are never measured or decided",
        "truth must-be-unknown characteristics", "MEASURED count", len(mbu_bad), 0, "<=", len(mbu),
        verdict=None if mbu else "NOT_EVALUABLE", truth_n=len(mbu),
        details={"decided": sum(1 for r in mbu_bad if r.decision in ("PASS", "FAIL")),
                 "examples": [f"{r.case_id}:{r.characteristic_id}:{r.decision}" for r in mbu_bad[:20]]}))

    # -- robustness ------------------------------------------------------------------------------
    failures = [c for c in cases if c.run_status in ENGINE_FAILURE_STATUSES]
    gates.append(_gate(
        "G1.ENGINE_FAILURES", "every engine outcome is a valid result (UNKNOWN/INVALID, never a crash)",
        "every compiled case", "engine failures", len(failures), 0, "<=",
        sum(1 for c in cases if c.run_status not in ("COMPILE_BLOCKED", "COMPILE_FAILED", "NOT_RUN")),
        truth_n=len(cases),
        details={"by_status": dict(Counter(c.run_status for c in failures).most_common()),
                 "examples": [f"{c.case_id}:{c.run_status}:{c.run_detail[:120]}" for c in failures[:20]],
                 "compile_failures": [f"{c.case_id}:{c.run_status}" for c in cases
                                      if c.run_status in ("COMPILE_BLOCKED", "COMPILE_FAILED")][:50],
                 "not_run": sum(1 for c in cases if c.run_status == "NOT_RUN")}))
    deg_cases = [c for c in native_cases if c.scan_plan in _DEGRADED_PLANS]
    deg_fail = [c for c in deg_cases if c.run_status in ENGINE_FAILURE_STATUSES]
    deg_meas = [r for r in meas if r.scan_plan in _DEGRADED_PLANS and r.within_u is not None]
    deg_u = _rate_of(deg_meas, lambda r: r.within_u)
    deg_fp = [r for r in native_chars if r.scan_plan in _DEGRADED_PLANS and r.false_pass]
    if not deg_cases:
        deg_verdict = "NOT_EVALUABLE"
    else:
        ok = not deg_fail and not deg_fp and (deg_u is None or deg_u >= th["u_calibration_min_rate"] - 1e-12)
        deg_verdict = "PASS" if ok else "FAIL"
    gates.append(_gate(
        "G1.OCCLUSION_RAISES_UNKNOWN", "more occlusion raises UNKNOWN, not error",
        "cases scanned with TOP_ONLY / SPARSE / SINGLE plans, native density",
        "failures, false PASS and U calibration on degraded plans",
        {"engine_failures": len(deg_fail), "false_pass": len(deg_fp), "u_calibration": stats.round_or_none(deg_u)},
        {"engine_failures": 0, "false_pass": 0, "u_calibration_min": th["u_calibration_min_rate"]}, "all",
        len(deg_cases), verdict=deg_verdict, truth_n=len(deg_cases),
        details={"coverage_by_plan": {p: stats.round_or_none(_rate_of(
            [r for r in scor if r.scan_plan == p and not r.ambiguity_excused],
            lambda r: r.engine_state == "MEASURED")) for p in ("FULL",) + _DEGRADED_PLANS},
                 "note": "coverage per plan is reported, not gated (cases differ between plans)"}))

    # -- runtime -----------------------------------------------------------------------------------
    rt = th["runtime"]
    timed = [c for c in cases if c.wall_s is not None and c.run_status not in ("COMPILE_BLOCKED", "COMPILE_FAILED")]
    runtime_tier = tier == "runtime"
    big = [c for c in timed if c.point_count >= rt["min_points"]]
    p95_wall = stats.percentile([c.wall_s for c in (big if runtime_tier else timed)], 95.0)
    rss = [c.peak_rss_bytes for c in (big if runtime_tier else timed) if c.peak_rss_bytes is not None]
    rt_ok = runtime_tier and bool(big) and len(big) == len(timed)
    gates.append(_gate(
        "G1.RUNTIME.P95", f"runtime P95 <= {rt['p95_wall_s_max']} s at ~1M points", "runtime-tier cases (>= "
        f"{rt['min_points']} points)", "p95(wall s)", p95_wall, rt["p95_wall_s_max"], "<=", len(big if runtime_tier else timed),
        verdict=None if rt_ok else "NOT_EVALUABLE", informational=not runtime_tier,
        details={"max_wall_s": stats.round_or_none(max((c.wall_s for c in timed), default=None) if timed else None),
                 "cases_below_min_points": sum(1 for c in timed if c.point_count < rt["min_points"])}))
    gates.append(_gate(
        "G1.RUNTIME.RSS", "peak RSS <= 4 GiB at ~1M points", "runtime-tier cases", "max(peak RSS bytes)",
        max(rss) if rss else None, rt["rss_max_bytes"], "<=", len(rss),
        verdict=None if (rt_ok and rss) else "NOT_EVALUABLE", informational=not runtime_tier))

    # -- tier budget (amendment 3: the regression tier runs on every A1 change, <= 20 min) -------------
    budget = th.get("tier_budget_s", {}).get(tier)
    engine_total = round(sum(c.wall_s or 0.0 for c in cases), 1)
    gates.append(_gate(
        "G1.TIER_BUDGET", "tier wall-time budget (engine runs; compiles are cached across A1 iterations)",
        f"every case of the {tier} tier", "sum of engine wall s", engine_total, budget, "<=", len(cases),
        verdict=None if budget is not None else "NOT_EVALUABLE", informational=True))

    # -- suite composition (full tier) -----------------------------------------------------------------
    st = th["suite"]
    fams = sorted({c.family for c in cases})
    coverage = (manifest or {}).get("coverage") or {}
    feature_gaps = {f: [k for k in st["required_family_features"] if not (coverage.get(f) or {}).get(k)] for f in fams}
    feature_gaps = {k: v for k, v in feature_gaps.items() if v}
    suite_ok = len(cases) >= st["min_cases"] and len(fams) >= st["min_families"] and not feature_gaps and bool(coverage)
    gates.append(_gate(
        "G1.SUITE", f">= {st['min_cases']} cases across >= {st['min_families']} families with every amendment-5 feature",
        "the scored tier", "cases / families / feature gaps",
        {"cases": len(cases), "families": len(fams), "feature_gaps": feature_gaps},
        {"min_cases": st["min_cases"], "min_families": st["min_families"]}, "all", len(cases),
        verdict="PASS" if suite_ok else "FAIL", informational=tier != st["qualifying_tier"],
        details={"families": fams}))

    # -- applicability by tier ---------------------------------------------------------------------------
    for g in gates:
        if not accuracy_tier and g["id"] not in ("G1.RUNTIME.P95", "G1.RUNTIME.RSS", "G1.ENGINE_FAILURES"):
            g["informational"] = True
            g["details"]["not_applicable"] = "accuracy is never scored on a point-count override tier"
        if runtime_tier and g["id"] in ("G1.RUNTIME.P95", "G1.RUNTIME.RSS"):
            g["informational"] = False

    # -- qualification -------------------------------------------------------------------------------------
    prov = dict(provenance or {})
    seedset = (manifest or {}).get("seedset")
    scanner = (manifest or {}).get("scanner_model") or {}
    oracle = any(c.oracle for c in cases)
    ledger = dict(ledger_check) if ledger_check else None
    reasons = []
    if tier != st["qualifying_tier"] and tier != "runtime":
        reasons.append(f"TIER_{tier.upper()}")
    if seedset != st["qualifying_seedset"]:
        reasons.append(f"SEEDSET_{seedset}")
    if scanner.get("provenance") != st["qualifying_scanner_provenance"]:
        reasons.append(f"SCANNER_{scanner.get('provenance')}")
    scan_gaps = scanner_model_gaps(manifest, th)
    waiver = None
    if scanner.get("provenance") == st["qualifying_scanner_provenance"] and scan_gaps["unmeasured_required_fields"]:
        model_sha = (manifest or {}).get("scanner_model_sha256")
        for w in (ledger or {}).get("waivers") or ():
            if (w.get("scope") == "SCANNER_MODEL_PARTIAL" and w.get("scanner_model_sha256") == model_sha
                    and set(scan_gaps["unmeasured_required_fields"]) <= set(w.get("fields") or ())):
                waiver = w
        if waiver is None:
            reasons.append("SCANNER_MODEL_PARTIAL")
    reasons += run_condition_reasons(prov.get("run_conditions"), th)
    if oracle:
        reasons.append("ORACLE_RESULTS")
    if ledger is None:
        reasons.append("NO_FREEZE_LEDGER")
    else:
        if not ledger.get("ok"):
            reasons.append("FREEZE_LEDGER_MISMATCH")
        if not ledger.get("anchor_checked"):
            reasons.append("LEDGER_NOT_ANCHORED")
        if ledger.get("qual_seeds_reused"):
            reasons.append("QUAL_SEEDS_REUSED")
    if tier == st["qualifying_tier"] and not suite_ok:
        reasons.append("SUITE_INCOMPLETE")
    binding = [g for g in gates if not g["informational"]]
    if any(g["verdict"] == "FAIL" for g in binding):
        verdict = "FAIL"
    elif all(g["verdict"] == "PASS" for g in binding) and binding:
        verdict = "PASS" if not reasons else "PASS_NOT_QUALIFYING"
    else:
        verdict = "INCOMPLETE"
    applicable = [g for g in binding if not g["not_applicable_to_tier"]]
    if any(g["verdict"] == "FAIL" for g in applicable):
        regression_verdict = "FAIL"
    elif applicable and all(g["verdict"] == "PASS" for g in applicable):
        regression_verdict = "PASS"
    else:
        regression_verdict = "INCOMPLETE"
    return {
        "schema": SCORECARD_SCHEMA,
        "eval_version": EVAL_VERSION,
        "tier": tier,
        "seedset": seedset,
        "qualifying": not reasons,
        "not_qualifying_reasons": reasons,
        "label": "SYNTHETIC ENGINEERING EVIDENCE (E2 suite); never physical or unseen-family accuracy"
                 + ("" if not reasons else " -- DEV / NOT QUALIFYING")
                 + ("" if waiver is None else f" -- scanner model partial, owner waiver {waiver.get('entry_id')}"),
        "verdict": verdict,
        "regression_verdict": regression_verdict,
        "failed_gates": [g["id"] for g in binding if g["verdict"] == "FAIL"],
        "not_evaluable_gates": [g["id"] for g in binding if g["verdict"] == "NOT_EVALUABLE"],
        "not_applicable_to_tier_gates": [g["id"] for g in binding if g["not_applicable_to_tier"]],
        "thresholds": {"version": th["version"], "sha256": thresholds_sha256(dict(th))},
        "provenance": {
            **prov,
            "generator_version": (manifest or {}).get("generator_version"),
            "scanner_model_sha256": (manifest or {}).get("scanner_model_sha256"),
            "scanner_model_provenance": scanner.get("provenance"),
            "scanner_model_unmeasured_required_fields": scan_gaps["unmeasured_required_fields"],
            "scanner_model_field_status": scan_gaps["field_status"],
            "scanner_model_waiver": waiver,
            "engine_versions": sorted({c.engine_version for c in cases if c.engine_version}),
            "oracle_results": oracle,
            "ledger_check": ledger,
        },
        "counts": {
            "cases": len(cases),
            "families": len(fams),
            "characteristics": len(chars),
            "scorable": sum(1 for r in chars if r.scorable),
            "measured_scorable": sum(1 for r in chars if r.scorable and r.engine_state == "MEASURED"),
            "bends": len(bends),
            "run_status": dict(Counter(c.run_status for c in cases).most_common()),
            "scenarios": dict(Counter(c.scenario for c in cases).most_common()),
            "scan_plans": dict(Counter(c.scan_plan for c in cases).most_common()),
            "summary_states": dict(Counter(f"{c.scenario}:{c.summary_state}" for c in cases).most_common()),
            "symmetric_cases": sum(1 for c in cases if c.symmetric),
            "symmetry_source": dict(Counter(c.ambiguity_source.split(":")[0] for c in cases).most_common()),
            "e1_e2_symmetry_disagreements": sorted(c.case_id for c in cases
                                                   if c.ambiguity_source.startswith("E1") and c.symmetric != c.symmetric_e2),
            "nominal_disagreements_gt_1e-3": sum(1 for r in chars if r.nominal_disagreement is not None
                                                 and abs(r.nominal_disagreement) > 1e-3),
            "unmatched_engine_characteristics_decided": sum(c.unmatched_engine_decided for c in cases),
            "stage_model_correct": dict(Counter(str(c.stage_model_correct) for c in cases).most_common()),
            "pose_check": dict(Counter(c.pose_check for c in cases).most_common()),
        },
        "gates": gates,
    }


def g1_verdict(full: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Overall G1 from the full-tier and runtime-tier scorecards: PASS only when
    both are PASS and qualifying, with one engine identity, one engine config and
    one freeze ledger; FAIL when any binding gate failed; otherwise INCOMPLETE
    (with the reasons)."""
    reasons: list[str] = []
    for name, sc in (("full", full), ("runtime", runtime)):
        if sc.get("tier") != name:
            reasons.append(f"{name.upper()}_SCORECARD_HAS_TIER_{sc.get('tier')}")
        if not sc.get("qualifying"):
            reasons.extend(f"{name.upper()}:{r}" for r in sc.get("not_qualifying_reasons", ()))
    ident = [((sc.get("provenance") or {}).get("fingerprints") or {}).get("engine") for sc in (full, runtime)]
    cfgs = [(sc.get("provenance") or {}).get("engine_config_sha256") for sc in (full, runtime)]
    if ident[0] != ident[1] or cfgs[0] != cfgs[1] or ident[0] is None:
        reasons.append("ENGINE_IDENTITY_DIFFERS_BETWEEN_TIERS")  # amendment 4: one engine config passes every gate
    ledgers = [((sc.get("provenance") or {}).get("ledger_check") or {}).get("first_entry_sha256") for sc in (full, runtime)]
    if ledgers[0] != ledgers[1] or ledgers[0] is None:
        reasons.append("LEDGER_DIFFERS_BETWEEN_TIERS")
    failed = [g["id"] for sc in (full, runtime) for g in sc.get("gates", ()) if not g["informational"]
              and g["verdict"] == "FAIL"]
    pending = [g["id"] for sc in (full, runtime) for g in sc.get("gates", ()) if not g["informational"]
               and g["verdict"] != "PASS"]
    if failed:
        verdict = "FAIL"
    elif pending or reasons:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "failed_gates": failed, "pending_gates": sorted(set(pending) - set(failed)),
            "not_qualifying_reasons": reasons}
