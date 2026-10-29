"""Per-bend benchmark evaluator: reported values against independent truth.

Two sources feed one record shape (:class:`Observation`, one per truth quantity):

- :func:`evaluate` scores a ``ShopfloorInspectionResultV1`` (or a failed job,
  ``result=None``) against a registry row
  (:class:`~domains.production.benchmark.case_registry.BenchmarkCaseV1`):
  metrology-measured, synthetic or operator-verdict truth.
- :func:`from_e4_score` adapts an E4 synthetic ``CaseScore``
  (:mod:`domains.production.engine.eval.scorer`). E4 owns the scorer and the G1
  gate scorecards (plan §10 amendment 2), so this module only re-labels E4's
  scorable rows. It never re-derives truth, the scoring rule or a gate.

:func:`aggregate` turns case evaluations into a report with **explicit
denominators**: every rate is ``{"numerator", "denominator", "value"}`` and
``value`` is ``None`` when the denominator is 0, never 0 or 1. It has an
overall block, a per-SKU-family breakdown and the worst family per metric.

Vocabulary (contracts/production/ENGINE_IO.md §1.2):

- Reported class of a characteristic:
  - ``IN`` (decision PASS) and ``OUT`` (FAIL) are the decided classes.
  - ``REVIEW`` is measured but inside the guard band.
  - ``UNDECIDED`` is measured without a nominal or tolerance.
  - ``NOT_MEASURED`` covers state NOT_MEASURED, a characteristic missing from the
    result, and every characteristic of a failed job.
  - ``NOT_APPLICABLE`` is the state NOT_APPLICABLE.
- Truth class (registry data only, never the engine's result):
  - ``IN`` / ``OUT`` come from the truth value against the registry row's
    recipe snapshot (``nominal``, ``tol_plus``, ``tol_minus``), using the truth
    instrument's uncertainty as a guard band, or from an operator verdict.
  - ``INDETERMINATE`` means the truth lies within its own uncertainty of a limit.
  - ``UNKNOWN`` means the snapshot states no tolerance for the quantity.

  Because the truth class never reads the result, it is identical for a
  successful run, a failed job and a characteristic the engine dropped: an
  engine failure can only add NOT_MEASURED rows to the denominators, never
  remove a truly-OUT part or characteristic from them, and an engine-reported
  tolerance cannot redefine truth. The result's expected value is compared with
  the snapshot only as the ``nominal_disagreement`` diagnostic.

Errors are ``reported actual - truth``. The headline error statistics count only
decided characteristics (IN or OUT), as the survey spec requires. The same
statistics over every MEASURED value (REVIEW and UNDECIDED included) are
reported next to them as ``errors_all_measured``, so the decided-only view
cannot hide a biased REVIEW band.

Part level:

- A part is ``CONFORMING`` when every registered quantity is truth IN.
- It is ``NONCONFORMING`` when any quantity is truth OUT, or (E4) when the part is
  mirrored, the wrong SKU, or on E2's never-green list.
- Otherwise it is ``UNDETERMINED``.

The summary states are compared against that: false OUT on conforming parts,
missed OUT on nonconforming parts, and the critical false accept
(ALL_MEASURED_IN_TOLERANCE on a nonconforming part).

Statistics come from E4's :mod:`domains.production.engine.eval.stats` (linear
interpolation percentiles; ``None`` on an empty population).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from contracts.production.characteristics import make_characteristic_id
from contracts.production.inspection_result import ShopfloorInspectionResultV1, iter_characteristics
from domains.production.engine.eval import stats

from .case_registry import QUANTITY_KEYS, BenchmarkCaseV1, TruthQuantityV1

__all__ = [
    "EVALUATOR_VERSION",
    "REPORTED_CLASSES",
    "TRUTH_CLASSES",
    "PART_TRUTH_CLASSES",
    "Observation",
    "truth_class",
    "CaseEvaluation",
    "evaluate",
    "from_e4_score",
    "aggregate",
    "ratio",
]

EVALUATOR_VERSION = "c12-benchmark-eval-1.1.0"
REPORTED_CLASSES: tuple[str, ...] = ("IN", "OUT", "REVIEW", "UNDECIDED", "NOT_MEASURED", "NOT_APPLICABLE")
TRUTH_CLASSES: tuple[str, ...] = ("IN", "OUT", "INDETERMINATE", "UNKNOWN")
PART_TRUTH_CLASSES: tuple[str, ...] = ("CONFORMING", "NONCONFORMING", "UNDETERMINED")
#: Part summary column used for jobs that produced no result.
RUN_FAILED = "RUN_FAILED"
_DECISION_TO_CLASS = {"PASS": "IN", "FAIL": "OUT", "REVIEW": "REVIEW", "UNKNOWN": "UNDECIDED"}
_KIND_ORDER = ("FLANGE_LENGTH", "INCLUDED_ANGLE", "BEND_LINE_POSITION", "BACKGAUGE_LENGTH", "GENERAL_MEASUREMENT")


@dataclass(frozen=True)
class Observation:
    """One truth quantity next to what the line reported for it."""

    case_id: str
    sku_family: str
    split_group: str
    characteristic_id: str
    kind: str
    side: Optional[str]
    unit: str
    bend_uid: Optional[str]
    truth_value: Optional[float]
    truth_uncertainty: Optional[float]
    truth_class: str
    reported_class: str
    reported_state: str  # MEASURED | NOT_MEASURED | NOT_APPLICABLE | ABSENT | RUN_FAILED
    reason: Optional[str]
    required: Optional[bool]
    expected: Optional[float]
    actual: Optional[float]
    uncertainty: Optional[float]
    #: ``actual - truth`` for decided (IN / OUT) characteristics only.
    error: Optional[float]
    #: ``actual - truth`` for every MEASURED characteristic.
    error_measured: Optional[float]
    #: Result expected minus the truth's nominal snapshot, when both exist (diagnostic).
    nominal_disagreement: Optional[float] = None


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    sku_family: str
    split_group: str
    truth_source: str
    run_status: str  # OK or the job failure / run status code
    summary_state: Optional[str]
    validity_state: Optional[str]
    part_truth: str
    observations: tuple[Observation, ...]
    duration_s: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["observations"] = [asdict(o) for o in self.observations]
        doc["extras"] = dict(self.extras)
        return doc


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """An explicit-denominator rate; ``value`` is ``None`` for an empty denominator."""
    return {"numerator": int(numerator), "denominator": int(denominator), "value": stats.rate(numerator, denominator)}


# ---------------------------------------------------------------------------
# registry truth
# ---------------------------------------------------------------------------


def truth_class(q: TruthQuantityV1) -> str:
    """Truth class of one registry quantity, from registry data alone (module docstring)."""
    if q.verdict is not None:
        return q.verdict
    if q.value is None or q.nominal is None or q.tol_plus is None or q.tol_minus is None:
        return "UNKNOWN"
    dev = float(q.value) - float(q.nominal)
    u = float(q.uncertainty or 0.0)
    plus, minus = q.tol_plus, q.tol_minus
    if -minus + u <= dev <= plus - u:
        return "IN"
    if dev > plus + u or dev < -minus - u:
        return "OUT"
    return "INDETERMINATE"


def _reported(char: Any) -> tuple[str, str, Optional[str]]:
    """(reported_class, reported_state, reason) for a contract characteristic or None."""
    if char is None:
        return "NOT_MEASURED", "ABSENT", "ABSENT_FROM_RESULT"
    if char.state == "NOT_APPLICABLE":
        return "NOT_APPLICABLE", "NOT_APPLICABLE", char.reason
    if char.state == "NOT_MEASURED":
        return "NOT_MEASURED", "NOT_MEASURED", char.reason
    return _DECISION_TO_CLASS[char.decision], "MEASURED", char.reason


def _part_truth(observations: Sequence[Observation], *, forced_nonconforming: bool = False) -> str:
    classes = [o.truth_class for o in observations]
    if forced_nonconforming or "OUT" in classes:
        return "NONCONFORMING"
    if classes and all(c == "IN" for c in classes):
        return "CONFORMING"
    return "UNDETERMINED"


def _as_result(result: Any) -> Optional[ShopfloorInspectionResultV1]:
    if result is None or isinstance(result, ShopfloorInspectionResultV1):
        return result
    return ShopfloorInspectionResultV1.model_validate(result)


def evaluate(
    result: ShopfloorInspectionResultV1 | Mapping[str, Any] | None,
    truth: BenchmarkCaseV1 | Mapping[str, Any],
    *,
    run_status: str = "OK",
    duration_s: Optional[float] = None,
    peak_rss_bytes: Optional[int] = None,
) -> CaseEvaluation:
    """Score one inspection against its registry truth (module docstring).

    ``result=None`` (or ``run_status != "OK"``) is a failed job: every truth
    quantity stays in the denominators as NOT_MEASURED. A result for another scan
    or recipe than the registry row names raises ``ValueError``, because scoring it would
    be silently wrong.
    """
    row = truth if isinstance(truth, BenchmarkCaseV1) else BenchmarkCaseV1.model_validate(truth)
    res = _as_result(result) if run_status == "OK" else None
    if run_status == "OK" and res is None:
        run_status = "RESULT_MISSING"
    if res is not None:
        if res.scan.sha256 != row.scan_sha256:
            raise ValueError(f"{row.case_id}: result scan sha256 differs from the registry row")
        if res.recipe.recipe_sha256 != row.recipe_sha256:
            raise ValueError(f"{row.case_id}: result recipe sha256 differs from the registry row")
    chars = {c.characteristic_id: c for c in iter_characteristics(res.bends, res.general_measurements)} if res else {}
    obs: list[Observation] = []
    for bend_uid in sorted(row.truth):
        for key in sorted(row.truth[bend_uid]):
            q = row.truth[bend_uid][key]
            kind, side, unit = QUANTITY_KEYS[key]
            cid = make_characteristic_id(kind, bend_uid=bend_uid, side=side)
            char = chars.get(cid)
            expected = char.expected if char is not None else None
            if res is None:
                rclass, rstate, reason = "NOT_MEASURED", "RUN_FAILED", run_status
            else:
                rclass, rstate, reason = _reported(char)
            actual = char.actual if char is not None and char.state == "MEASURED" else None
            err_m = None if actual is None or q.value is None else float(actual) - float(q.value)
            nominal_dis = None
            if expected is not None and q.nominal is not None:
                nominal_dis = float(expected) - float(q.nominal)
            obs.append(Observation(
                case_id=row.case_id, sku_family=row.sku_family, split_group=row.split_group, characteristic_id=cid,
                kind=kind, side=side, unit=unit, bend_uid=bend_uid, truth_value=q.value,
                truth_uncertainty=q.uncertainty, truth_class=truth_class(q), reported_class=rclass,
                reported_state=rstate, reason=reason, required=char.required if char is not None else None,
                expected=expected, actual=actual, uncertainty=char.uncertainty if char is not None else None,
                error=err_m if rclass in ("IN", "OUT") else None, error_measured=err_m,
                nominal_disagreement=nominal_dis,
            ))
    return CaseEvaluation(
        case_id=row.case_id, sku_family=row.sku_family, split_group=row.split_group, truth_source=row.truth_source,
        run_status=run_status, summary_state=res.summary.state if res is not None else None,
        validity_state=res.inspection_validity.state if res is not None else None, part_truth=_part_truth(obs),
        observations=tuple(obs), duration_s=duration_s, peak_rss_bytes=peak_rss_bytes,
    )


# ---------------------------------------------------------------------------
# E4 adapter (synthetic)
# ---------------------------------------------------------------------------


def from_e4_score(
    score: Any,
    *,
    sku_family: Optional[str] = None,
    split_group: Optional[str] = None,
    duration_s: Optional[float] = None,
    peak_rss_bytes: Optional[int] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> CaseEvaluation:
    """Re-label an E4 ``CaseScore`` (scorer rows) as a :class:`CaseEvaluation`.

    Only E4-scorable rows are kept (applicable, truth defined, well defined; the
    E2 scoring rule), and their truth class is E2's ``truth_in_tolerance``. An
    E4 run failure (``run_status != OK``) leaves every row NOT_MEASURED.
    """
    case = score.case
    family = sku_family or case.family
    group = split_group or case.family
    obs: list[Observation] = []
    for r in score.chars:
        if not r.scorable:
            continue
        if case.run_status != "OK":
            rclass, rstate, reason = "NOT_MEASURED", "RUN_FAILED", case.run_status
        elif r.engine_state == "ABSENT":
            rclass, rstate, reason = "NOT_MEASURED", "ABSENT", "ABSENT_FROM_RESULT"
        elif r.engine_state == "NOT_APPLICABLE":
            rclass, rstate, reason = "NOT_APPLICABLE", "NOT_APPLICABLE", r.reason
        elif r.engine_state == "NOT_MEASURED":
            rclass, rstate, reason = "NOT_MEASURED", "NOT_MEASURED", r.reason
        else:
            rclass, rstate, reason = _DECISION_TO_CLASS[r.decision], "MEASURED", r.reason
        tclass = "UNKNOWN" if r.truth_in_tolerance is None else ("IN" if r.truth_in_tolerance else "OUT")
        obs.append(Observation(
            case_id=r.case_id, sku_family=family, split_group=group, characteristic_id=r.characteristic_id,
            kind=r.kind, side=r.side, unit=r.unit, bend_uid=r.bend_uid, truth_value=r.truth, truth_uncertainty=None,
            truth_class=tclass, reported_class=rclass, reported_state=rstate, reason=reason, required=r.required,
            expected=r.expected, actual=r.actual, uncertainty=r.uncertainty,
            error=r.error if rclass in ("IN", "OUT") else None, error_measured=r.error,
            nominal_disagreement=r.nominal_disagreement,
        ))
    forced = bool(case.mirrored or case.wrong_sku or "ALL_MEASURED_IN_TOLERANCE" in (case.summary_never or ()))
    part = _part_truth(obs, forced_nonconforming=forced)
    if part == "CONFORMING" and not case.complete:
        part = "UNDETERMINED"  # an unfinished part is never 'conforming' as a whole
    return CaseEvaluation(
        case_id=case.case_id, sku_family=family, split_group=group, truth_source="synthetic",
        run_status=case.run_status, summary_state=case.summary_state, validity_state=case.validity_state,
        part_truth=part, observations=tuple(obs),
        duration_s=duration_s if duration_s is not None else case.wall_s,
        peak_rss_bytes=peak_rss_bytes if peak_rss_bytes is not None else case.peak_rss_bytes,
        extras=dict(extras or {}),
    )


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _error_stats(values: Sequence[Optional[float]]) -> dict[str, Any]:
    arr = stats.finite(values)
    return {
        "n": int(arr.size),
        "bias": stats.round_or_none(stats.mean(values)),
        "mae": stats.round_or_none(stats.mae(values)),
        "p95_abs": stats.round_or_none(stats.p95_abs(values)),
        "max_abs": stats.round_or_none(float(abs(arr).max()) if arr.size else None),
    }


def _kinds(observations: Sequence[Observation]) -> list[str]:
    present = {o.kind for o in observations}
    return [k for k in _KIND_ORDER if k in present] + sorted(present - set(_KIND_ORDER))


def _block(cases: Sequence[CaseEvaluation]) -> dict[str, Any]:
    obs = [o for c in cases for o in c.observations]
    per_kind: dict[str, Any] = {}
    for kind in _kinds(obs):
        rows = [o for o in obs if o.kind == kind]
        per_kind[kind] = {
            "unit": rows[0].unit,
            "errors_decided": _error_stats([o.error for o in rows]),
            "errors_all_measured": _error_stats([o.error_measured for o in rows]),
            "not_measured": ratio(sum(o.reported_class == "NOT_MEASURED" for o in rows),
                                  sum(o.reported_class != "NOT_APPLICABLE" for o in rows)),
        }
    confusion = {t: {r: 0 for r in REPORTED_CLASSES} for t in TRUTH_CLASSES}
    for o in obs:
        confusion[o.truth_class][o.reported_class] += 1
    truth_in = sum(confusion["IN"].values())
    truth_out = sum(confusion["OUT"].values())
    char_rates = {
        "false_out_on_truth_in": ratio(confusion["IN"]["OUT"], truth_in),
        "missed_out_on_truth_out": ratio(truth_out - confusion["OUT"]["OUT"], truth_out),
        "false_in_on_truth_out": ratio(confusion["OUT"]["IN"], truth_out),
        "decided_on_truth_in_or_out": ratio(
            sum(confusion[t]["IN"] + confusion[t]["OUT"] for t in ("IN", "OUT")), truth_in + truth_out),
    }
    part_conf: dict[str, Counter[str]] = {p: Counter() for p in PART_TRUTH_CLASSES}
    for c in cases:
        part_conf[c.part_truth][c.summary_state if c.summary_state is not None else RUN_FAILED] += 1
    conforming = sum(part_conf["CONFORMING"].values())
    nonconforming = sum(part_conf["NONCONFORMING"].values())
    part_rates = {
        "false_out_on_conforming": ratio(part_conf["CONFORMING"]["OUT_OF_TOLERANCE_FOUND"], conforming),
        "missed_out_on_nonconforming": ratio(nonconforming - part_conf["NONCONFORMING"]["OUT_OF_TOLERANCE_FOUND"],
                                             nonconforming),
        "false_accept_on_nonconforming": ratio(part_conf["NONCONFORMING"]["ALL_MEASURED_IN_TOLERANCE"], nonconforming),
    }
    calib = [o for o in obs if o.error_measured is not None and o.uncertainty is not None]
    within = sum(1 for o in calib
                 if abs(o.error_measured) <= ((o.uncertainty or 0.0) ** 2 + (o.truth_uncertainty or 0.0) ** 2) ** 0.5 + 1e-9)
    durations = [c.duration_s for c in cases]
    rss = [c.peak_rss_bytes for c in cases if c.peak_rss_bytes is not None]
    return {
        "cases": len(cases),
        "characteristics": len(obs),
        "run_status": dict(sorted(Counter(c.run_status for c in cases).items())),
        "per_kind": per_kind,
        "not_measured": ratio(sum(o.reported_class == "NOT_MEASURED" for o in obs),
                              sum(o.reported_class != "NOT_APPLICABLE" for o in obs)),
        "not_applicable_count": sum(o.reported_class == "NOT_APPLICABLE" for o in obs),
        "characteristic_confusion": confusion,
        "characteristic_rates": char_rates,
        #: |actual - truth| <= sqrt(U^2 + U_truth^2) over MEASURED characteristics with a truth value.
        "u_calibration": ratio(within, len(calib)),
        "part_confusion": {p: dict(sorted(v.items())) for p, v in part_conf.items()},
        "part_rates": part_rates,
        "runtime": {
            "duration_s": {"n": len(stats.finite(durations)), "p50": stats.round_or_none(stats.percentile(durations, 50)),
                           "p95": stats.round_or_none(stats.percentile(durations, 95)),
                           "max": stats.round_or_none(stats.percentile(durations, 100))},
            "peak_rss_bytes_max": max(rss) if rss else None,
        },
    }


#: (metric path, label) pairs used to name the worst family; higher is worse for each.
WORST_FAMILY_METRICS: tuple[tuple[str, str], ...] = (
    ("per_kind.FLANGE_LENGTH.errors_decided.mae", "flange length MAE (decided)"),
    ("per_kind.INCLUDED_ANGLE.errors_decided.mae", "angle MAE (decided)"),
    ("per_kind.BEND_LINE_POSITION.errors_decided.mae", "bend-line position MAE (decided)"),
    ("not_measured.value", "NOT_MEASURED rate"),
    ("characteristic_rates.false_in_on_truth_out.value", "false IN on truth OUT"),
    ("characteristic_rates.missed_out_on_truth_out.value", "missed OUT on truth OUT"),
    ("part_rates.false_accept_on_nonconforming.value", "false accept on nonconforming parts"),
)


def _dig(doc: Mapping[str, Any], path: str) -> Any:
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def aggregate(cases: Iterable[CaseEvaluation]) -> dict[str, Any]:
    """Overall + per-SKU-family blocks and the worst family per metric (module docstring)."""
    items = list(cases)
    by_family: dict[str, list[CaseEvaluation]] = defaultdict(list)
    for c in items:
        by_family[c.sku_family].append(c)
    families = {fam: _block(rows) for fam, rows in sorted(by_family.items())}
    worst: dict[str, Any] = {}
    for path, label in WORST_FAMILY_METRICS:
        candidates = [(fam, _dig(block, path)) for fam, block in families.items()]
        candidates = [(fam, v) for fam, v in candidates if v is not None]
        if not candidates:
            worst[path] = {"label": label, "family": None, "value": None}
            continue
        fam, value = sorted(candidates, key=lambda fv: (-fv[1], fv[0]))[0]
        if len(candidates) > 1 and all(v == value for _f, v in candidates):
            # every family scores the same: naming one would invent a "worst"
            worst[path] = {"label": label, "family": None, "value": value, "families_evaluable": len(candidates),
                           "note": "all evaluable families are equal"}
            continue
        worst[path] = {"label": label, "family": fam, "value": value, "families_evaluable": len(candidates)}
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "overall": _block(items),
        "by_family": families,
        "worst_family": worst,
    }
