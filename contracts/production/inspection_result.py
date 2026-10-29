"""ShopfloorInspectionResultV1: the per-scan output of the production engine (E3).

One result per scan. It reports, per planned bend in sequence order, the
bend state (FORMED / NOT_YET_FORMED / PARTIALLY_FORMED / UNEXPECTED_FORMED /
UNKNOWN) and the characteristic values (flange length A/B first, then the
included angle, the bend-line position and, mid-sequence, the backgauge
length) with expected / actual / deviation / uncertainty U / tolerance /
decision / state / reason. It also carries the general measurements, a stage
decision, validity, and viewer assets, all in one declared frame
(``cad_model_mm``).

Honesty rules enforced by validators (plan §3.2 step 6 and AGENTS.md):

- Execution status, validity, bend state and per-characteristic decision are
  separate fields. ``authority`` is fixed to OPERATOR_ASSIST /
  NOT_AUTHORIZED.
- MEASURED needs an actual value and a strictly positive U; when a nominal and
  a tolerance exist the decision must equal the guard-band rule
  (:func:`contracts.production.characteristics.guard_band_decision`).
- NOT_MEASURED carries no actual value, decides UNKNOWN and states a reason;
  UNKNOWN never becomes PASS or FAIL.
- A NOT_YET_FORMED bend is never a defect: its bend characteristics are
  NOT_APPLICABLE (BEND_NOT_YET_FORMED).
- An invalid inspection has no MEASURED characteristic and no known bend state.
- NOT_APPLICABLE_AT_STAGE needs some NOT_YET_FORMED bend, so a finished part
  cannot skip a required value. On a FLANGE_LENGTH (amendment ENGINE_IO-A1,
  2026-10-19) it needs the flange's own adjacent bend to be NOT_YET_FORMED.
- An ambiguous pose measures nothing and, like an invalid inspection or a
  degenerate in-plane pose, publishes no heatmap. Amendment ENGINE_IO-A2
  (2026-10-19): with ``registration.identity_ambiguous`` it may publish the
  readings identical under every tied symmetric twin, never a bend-line
  position, a heatmap or a stage model.
- ``stage_decision`` and ``summary`` must equal the pure derivations
  :func:`derive_stage_decision` and :func:`derive_summary`. Nothing measured
  is never "in tolerance", and a part with an unformed bend is never
  ALL_MEASURED_IN_TOLERANCE (at best STAGE_IN_TOLERANCE).
- ``registration.raw_to_frame`` is a proper rigid transform (no scale, no
  mirror).

Cross-document checks (result vs CompiledPart vs recipe) live in
:mod:`contracts.production.consistency`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from contracts.production.characteristics import (
    BEND_CHARACTERISTIC_ORDER,
    NOT_APPLICABLE_REASONS,
    UNKNOWN_REASONS,
    BoundaryKind,
    CharacteristicKind,
    Decision,
    ExpectedSource,
    FlangeSide,
    MeasurementState,
    exceedance_ratio,
    guard_band_decision,
    make_characteristic_id,
    unit_for_kind,
)
from contracts.production.common import (
    FRAME_CAD_MODEL,
    CFloat,
    CNonNegFloat,
    ContractModel,
    CPosFloat,
    FiniteFloat,
    Mat4,
    NonNegFloat,
    PosFloat,
    RelPath,
    Sha256Hex,
    ToleranceRef,
    UtcIsoTimestamp,
    Vec3,
    round6,
)
from contracts.production.compiled_part import STAGE_MODEL_ID_PATTERN
from contracts.production.line_job import LineJobSummaryV1
from contracts.production.states import (
    BEND_STATES,
    FORMED_LIKE_STATES,
    STAGE_STATUSES,
    SUMMARY_STATES,
    VALIDITY_STATES,
    BendState,
    StageStatus,
    SummaryState,
    ValidityState,
)
from contracts.production.part_recipe import BEND_UID_PATTERN, RECIPE_ID_PATTERN, SKU_PATTERN, InspectionStage

__all__ = [
    "SHOPFLOOR_RESULT_SCHEMA",
    "STATION_FEED_SCHEMA",
    "BEND_GEOMETRY_SCHEMA",
    "BendState",
    "BEND_STATES",
    "FORMED_LIKE_STATES",
    "ValidityState",
    "VALIDITY_STATES",
    "SummaryState",
    "SUMMARY_STATES",
    "StageStatus",
    "STAGE_STATUSES",
    "StageDecisionValue",
    "DeclaredStageSource",
    "AUTHORITY_MODE",
    "PRODUCTION_DISPOSITION",
    "StationStatsV1",
    "CharacteristicEvidenceV1",
    "CharacteristicResultV1",
    "BendStateEvidenceV1",
    "BendDisplayV1",
    "BendRowV1",
    "GeneralMeasurementRowV1",
    "UnplannedCreaseV1",
    "RegistrationV1",
    "CoverageV1",
    "InspectionValidityV1",
    "StageDecisionV1",
    "WorstCharacteristicV1",
    "SummaryV1",
    "ViewerAssetsV1",
    "ArtifactsV1",
    "EngineInfoV1",
    "ScanInfoV1",
    "ResultContextV1",
    "RecipeRefV1",
    "AuthorityV1",
    "ShopfloorInspectionResultV1",
    "BendGeometryEntryV1",
    "BendGeometryFileV1",
    "StationIssueV1",
    "StationLatestCompletedV1",
    "StationActiveV1",
    "StationFeedV1",
    "derive_stage_decision",
    "derive_summary",
    "required_characteristics",
    "iter_characteristics",
    "RIGID_TOLERANCE",
    "check_rigid_transform",
    "stage_index_of_model_id",
]

SHOPFLOOR_RESULT_SCHEMA = "sherman_shopfloor_inspection_result_v1"
STATION_FEED_SCHEMA = "sherman_station_feed_v1"
BEND_GEOMETRY_SCHEMA = "sherman_bend_geometry_v1"

AUTHORITY_MODE = "OPERATOR_ASSIST"
PRODUCTION_DISPOSITION = "NOT_AUTHORIZED"

BendUid = Annotated[str, Field(pattern=BEND_UID_PATTERN)]
StageModelId = Annotated[str, Field(pattern=STAGE_MODEL_ID_PATTERN)]

#: Orthonormality tolerance for ``raw_to_frame``. Canonical JSON keeps 6
#: decimals, so a true rotation can drift by a few 1e-6 after rounding; any
#: real scale (>= 1e-4 relative) or shear is far above this.
RIGID_TOLERANCE = 2e-5


def check_rigid_transform(m: Sequence[Sequence[float]], *, name: str = "transform") -> None:
    """Raise ``ValueError`` unless ``m`` (4x4, row-major) is a proper rigid
    transform: last row ``[0, 0, 0, 1]``, ``R^T R = I`` and ``det(R) = +1``."""
    if [float(v) for v in m[3]] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    r = [[float(m[i][j]) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > RIGID_TOLERANCE:
                raise ValueError(f"{name} must be rigid (no scale or shear): R^T R != I")
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if det <= 0:
        raise ValueError(f"{name} must be a proper rotation (det = +1); a mirror is MIRRORED_PART_SUSPECTED")


def stage_index_of_model_id(stage_model_id: str) -> int:
    """``"S3"`` / ``"S3-swap"`` -> 3."""
    return int(stage_model_id[1:].split("-", 1)[0])


StageDecisionValue = Literal["PASS", "REVIEW", "UNKNOWN"]
DeclaredStageSource = Literal["SIDECAR", "OPERATOR", "API"]

#: Status precedence (first match wins) and the decision each status implies.
_STAGE_PRECEDENCE: tuple[tuple[str, str], ...] = (
    ("NOT_EVALUATED", "UNKNOWN"),
    ("AMBIGUOUS_POSE", "UNKNOWN"),
    # REVIEW evidence outranks an unknown bend elsewhere (as REVIEW outranks
    # UNKNOWN for characteristics), so UNDETERMINED never hides it.
    ("PARTIAL_BEND_FOUND", "REVIEW"),
    ("OUT_OF_ORDER_SUSPECTED", "REVIEW"),
    ("STAGE_MISMATCH", "REVIEW"),
    ("FINAL_INCOMPLETE", "REVIEW"),
    ("UNDETERMINED", "UNKNOWN"),
    ("CONSISTENT", "PASS"),
)


# --------------------------------------------------------------------------
# Characteristics
# --------------------------------------------------------------------------


class StationStatsV1(ContractModel):
    """Per-station aggregation along the visible bend length (5-11 stations)."""

    count: int = Field(ge=1)
    median: CFloat
    min: CFloat
    max: CFloat

    @model_validator(mode="after")
    def _ordered(self) -> "StationStatsV1":
        if not (self.min <= self.median <= self.max):
            raise ValueError("station stats must satisfy min <= median <= max")
        return self


class CharacteristicEvidenceV1(ContractModel):
    method: str = Field(min_length=1, max_length=96)
    support_points: int | None = Field(default=None, ge=0)
    coverage_ratio: Annotated[float, Field(ge=0, le=1)] | None = None
    fit_rms_mm: NonNegFloat | None = None
    notes: tuple[str, ...] = ()


class CharacteristicResultV1(ContractModel):
    characteristic_id: str = Field(min_length=1, max_length=200)
    kind: CharacteristicKind
    bend_uid: BendUid | None = None
    side: FlangeSide | None = None
    stage_index: int | None = Field(default=None, ge=0)
    measurement_id: str | None = Field(default=None, min_length=1, max_length=100)
    unit: Literal["mm", "deg"]
    #: Counted for the part summary. NOT_APPLICABLE characteristics are never required.
    required: bool
    state: MeasurementState
    decision: Decision
    #: UNKNOWN reason (NOT_MEASURED / unresolvable MEASURED) or NOT_APPLICABLE reason.
    reason: str | None = None
    expected: CFloat | None = None
    expected_source: ExpectedSource | None = None
    actual: CFloat | None = None
    #: actual - expected, rounded to 6 decimals.
    deviation: CFloat | None = None
    #: Expanded uncertainty U (same unit), strictly positive when MEASURED.
    uncertainty: CPosFloat | None = None
    tolerance: ToleranceRef | None = None
    boundary_kind: BoundaryKind | None = None
    adjacent_bend_uid: BendUid | None = None
    stations: StationStatsV1 | None = None
    evidence: CharacteristicEvidenceV1 | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CharacteristicResultV1":
        expected_id = make_characteristic_id(
            self.kind,
            bend_uid=self.bend_uid,
            side=self.side,
            stage_index=self.stage_index,
            measurement_id=self.measurement_id,
        )
        if self.characteristic_id != expected_id:
            raise ValueError(f"characteristic_id must be {expected_id!r}")
        if self.unit != unit_for_kind(self.kind):
            raise ValueError(f"{self.kind} must use unit {unit_for_kind(self.kind)}")
        if self.side is not None and self.kind not in ("FLANGE_LENGTH", "BACKGAUGE_LENGTH"):
            raise ValueError("only flange/backgauge characteristics carry a side")
        if self.boundary_kind is not None and self.kind not in ("FLANGE_LENGTH", "BACKGAUGE_LENGTH"):
            raise ValueError("boundary_kind applies to flange/backgauge lengths only")
        if self.adjacent_bend_uid is not None and self.boundary_kind != "ADJACENT_BEND":
            raise ValueError("adjacent_bend_uid requires boundary_kind ADJACENT_BEND")
        if (self.expected is None) != (self.expected_source is None):
            raise ValueError("expected and expected_source go together")

        if self.state == "MEASURED":
            if self.actual is None or self.uncertainty is None:
                raise ValueError("MEASURED requires actual and a strictly positive uncertainty")
            if self.expected is None:
                if self.deviation is not None:
                    raise ValueError("deviation requires an expected value")
            else:
                if self.deviation is None or not math.isclose(
                    self.deviation, self.actual - self.expected, abs_tol=2e-6
                ):
                    raise ValueError("deviation must equal actual - expected")
            if self.expected is not None and self.tolerance is not None:
                derived = guard_band_decision(
                    self.deviation, self.uncertainty, self.tolerance.plus, self.tolerance.minus  # type: ignore[arg-type]
                )
                if self.decision != derived:
                    raise ValueError(f"decision must follow the guard band ({derived})")
                if self.reason is not None:
                    raise ValueError("a guard-band decision carries no reason")
            else:
                needed = "NOMINAL_UNAVAILABLE" if self.expected is None else "NO_TOLERANCE"
                if self.decision != "UNKNOWN" or self.reason != needed:
                    raise ValueError(f"a measured value without nominal/tolerance decides UNKNOWN ({needed})")
        else:
            if any(v is not None for v in (self.actual, self.deviation, self.uncertainty, self.stations)):
                raise ValueError(f"{self.state} must not carry actual/deviation/uncertainty/stations")
            if self.state == "NOT_MEASURED":
                if self.decision != "UNKNOWN":
                    raise ValueError("NOT_MEASURED decides UNKNOWN")
                if self.reason not in UNKNOWN_REASONS:
                    raise ValueError("NOT_MEASURED needs an UNKNOWN reason")
            else:  # NOT_APPLICABLE
                if self.decision != "NOT_APPLICABLE":
                    raise ValueError("NOT_APPLICABLE decides NOT_APPLICABLE")
                if self.reason not in _NA_REASONS_BY_KIND[self.kind]:
                    raise ValueError(f"{self.kind} cannot be NOT_APPLICABLE for reason {self.reason!r}")
                if self.required:
                    raise ValueError("a NOT_APPLICABLE characteristic is never required")
        if self.state != "NOT_APPLICABLE":
            # Requiredness is fixed by the contract, not chosen by the engine,
            # so a value can never be hidden from the summary.
            if self.kind in ("FLANGE_LENGTH", "INCLUDED_ANGLE"):
                if not self.required or self.tolerance is None:
                    raise ValueError(f"an applicable {self.kind} is required and toleranced")
            elif self.kind in ("BEND_LINE_POSITION", "BACKGAUGE_LENGTH"):
                if self.required != (self.tolerance is not None):
                    raise ValueError(f"{self.kind} is required exactly when a tolerance exists")
            elif self.required and self.tolerance is None:  # GENERAL_MEASUREMENT
                raise ValueError("a required GENERAL_MEASUREMENT is toleranced")
        return self


    # -- factories (the only recommended way for producers to build rows) --

    @staticmethod
    def required_for(nominal: Any) -> bool:
        """Requiredness of an applicable row built from a CompiledPart nominal:
        the recipe requires it and a tolerance is resolved."""
        return nominal.recipe_requirement == "REQUIRED" and nominal.tolerance is not None

    @classmethod
    def _identity(cls, nominal: Any) -> dict[str, Any]:
        """Identity fields copied from a CompiledPart ``CharacteristicNominalV1``."""
        return {
            "characteristic_id": nominal.characteristic_id,
            "kind": nominal.kind,
            "bend_uid": nominal.bend_uid,
            "side": nominal.side,
            "stage_index": nominal.stage_index,
            "measurement_id": nominal.measurement_id,
            "unit": nominal.unit,
            "boundary_kind": nominal.boundary_kind,
            "adjacent_bend_uid": nominal.adjacent_bend_uid,
            "tolerance": nominal.tolerance,
        }

    @classmethod
    def measured(
        cls,
        nominal: Any,
        *,
        actual: float,
        uncertainty: float,
        expected_source: str,
        required: bool | None = None,
        stations: "StationStatsV1 | None" = None,
        evidence: "CharacteristicEvidenceV1 | None" = None,
    ) -> "CharacteristicResultV1":
        """A MEASURED row: rounds inputs canonically, then derives deviation and
        the guard-band decision (or UNKNOWN NOMINAL_UNAVAILABLE / NO_TOLERANCE).
        ``required`` defaults to :meth:`required_for`."""
        ident = cls._identity(nominal)
        required = cls.required_for(nominal) if required is None else required
        expected = None if nominal.nominal is None else round6(nominal.nominal)
        actual_r, u_r = round6(actual), round6(uncertainty)
        deviation = None if expected is None else round6(actual_r - expected)
        tol = ident["tolerance"]
        if expected is None:
            decision, reason = "UNKNOWN", "NOMINAL_UNAVAILABLE"
        elif tol is None:
            decision, reason = "UNKNOWN", "NO_TOLERANCE"
        else:
            decision, reason = guard_band_decision(deviation, u_r, tol.plus, tol.minus), None  # type: ignore[arg-type]
        return cls(
            **ident,
            required=required,
            state="MEASURED",
            decision=decision,
            reason=reason,
            expected=expected,
            expected_source=None if expected is None else expected_source,
            actual=actual_r,
            deviation=deviation,
            uncertainty=u_r,
            stations=stations,
            evidence=evidence,
        )

    @classmethod
    def not_measured(
        cls, nominal: Any, *, reason: str, expected_source: str, required: bool | None = None,
        evidence: "CharacteristicEvidenceV1 | None" = None,
    ) -> "CharacteristicResultV1":
        """A NOT_MEASURED row (decision UNKNOWN). The nominal is shown as expected only.
        ``required`` defaults to :meth:`required_for`."""
        ident = cls._identity(nominal)
        required = cls.required_for(nominal) if required is None else required
        expected = None if nominal.nominal is None else round6(nominal.nominal)
        return cls(
            **ident, required=required, state="NOT_MEASURED", decision="UNKNOWN", reason=reason,
            expected=expected, expected_source=None if expected is None else expected_source, evidence=evidence,
        )

    @classmethod
    def not_applicable(cls, nominal: Any, *, reason: str, expected_source: str) -> "CharacteristicResultV1":
        ident = cls._identity(nominal)
        expected = None if nominal.nominal is None else round6(nominal.nominal)
        return cls(
            **ident, required=False, state="NOT_APPLICABLE", decision="NOT_APPLICABLE", reason=reason,
            expected=expected, expected_source=None if expected is None else expected_source,
        )


#: NOT_APPLICABLE is only legitimate for these (kind, reason) pairs.
_NA_REASONS_BY_KIND: dict[str, frozenset[str]] = {
    # NOT_APPLICABLE_AT_STAGE on a flange: contract amendment ENGINE_IO-A1 (owner decision 2026-10-19, ENGINE_IO.md
    # section 7.1): a flange bounded by an adjacent bend that is NOT_YET_FORMED has no defined length at this stage.
    "FLANGE_LENGTH": frozenset({"BEND_NOT_YET_FORMED", "NOT_INSPECTED_BY_RECIPE", "NOT_APPLICABLE_AT_STAGE"}),
    "INCLUDED_ANGLE": frozenset({"BEND_NOT_YET_FORMED"}),
    "BEND_LINE_POSITION": frozenset({"BEND_NOT_YET_FORMED"}),
    "BACKGAUGE_LENGTH": frozenset({"BEND_NOT_YET_FORMED", "NOT_APPLICABLE_AT_STAGE"}),
    "GENERAL_MEASUREMENT": frozenset({"NOT_INSPECTED_BY_RECIPE", "NOT_APPLICABLE_AT_STAGE"}),
}
if not all(v <= NOT_APPLICABLE_REASONS for v in _NA_REASONS_BY_KIND.values()):  # pragma: no cover
    raise RuntimeError("NOT_APPLICABLE reason table drifted from the vocabulary")


# --------------------------------------------------------------------------
# Bend rows
# --------------------------------------------------------------------------


class BendStateEvidenceV1(ContractModel):
    nominal_included_deg: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)]
    measured_included_deg: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)] | None = None
    #: Fraction of the nominal sweep realised (0 = flat, 1 = nominal).
    formed_fraction: FiniteFloat | None = None
    parent_support_points: int | None = Field(default=None, ge=0)
    child_support_points: int | None = Field(default=None, ge=0)
    coverage_ratio: Annotated[float, Field(ge=0, le=1)] | None = None
    #: Order/moves prior used by the decoder (soft prior, never decisive alone).
    prior_formed_probability: Annotated[float, Field(ge=0, le=1)] | None = None


class BendDisplayV1(ContractModel):
    """Viewer geometry in ``cad_model_mm``.

    ``measured_polyline_mm`` exists exactly when BEND_LINE_POSITION is MEASURED
    (the solid OML line fitted from scan planes). ``nominal_polyline_mm`` is the
    nominal OML locus of the stage model the scan was decoded against.
    """

    nominal_polyline_mm: tuple[Vec3, ...] = Field(min_length=2)
    measured_polyline_mm: tuple[Vec3, ...] | None = Field(default=None, min_length=2)
    label_anchor_mm: Vec3
    side_a_anchor_mm: Vec3 | None = None
    side_b_anchor_mm: Vec3 | None = None


class BendRowV1(ContractModel):
    bend_uid: BendUid
    label: str = Field(min_length=1, max_length=32)
    sequence_index: int = Field(ge=1)
    machine_move_index: int | None = Field(default=None, ge=1)
    state: BendState
    state_reasons: tuple[str, ...] = ()
    state_evidence: BendStateEvidenceV1
    #: Fixed order: FLANGE_LENGTH A, FLANGE_LENGTH B, INCLUDED_ANGLE,
    #: BEND_LINE_POSITION, then optionally one BACKGAUGE_LENGTH.
    characteristics: tuple[CharacteristicResultV1, ...]
    display: BendDisplayV1
    diagnostics: dict[str, FiniteFloat] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> "BendRowV1":
        chars = self.characteristics
        if len(chars) not in (4, 5):
            raise ValueError("a bend row has 4 characteristics plus an optional backgauge")
        for char, (kind, side) in zip(chars, BEND_CHARACTERISTIC_ORDER):
            if char.kind != kind or char.side != side:
                raise ValueError("bend characteristics must follow FLANGE A, FLANGE B, ANGLE, POSITION")
        if len(chars) == 5 and chars[4].kind != "BACKGAUGE_LENGTH":
            raise ValueError("the fifth bend characteristic must be BACKGAUGE_LENGTH")
        if any(c.bend_uid != self.bend_uid for c in chars):
            raise ValueError("characteristics must belong to their bend row")

        for char in chars:
            if char.reason == "BEND_NOT_YET_FORMED" and self.state != "NOT_YET_FORMED":
                raise ValueError("BEND_NOT_YET_FORMED is only valid on a NOT_YET_FORMED bend")
        if self.state == "NOT_YET_FORMED":
            for char in chars:
                if char.state != "NOT_APPLICABLE":
                    raise ValueError("a NOT_YET_FORMED bend has only NOT_APPLICABLE characteristics")
            for char in chars[2:4]:
                if char.reason != "BEND_NOT_YET_FORMED":
                    raise ValueError("angle/position of a NOT_YET_FORMED bend are BEND_NOT_YET_FORMED")
        elif self.state == "UNKNOWN":
            if not self.state_reasons or not set(self.state_reasons) <= UNKNOWN_REASONS:
                raise ValueError("an UNKNOWN bend needs UNKNOWN reasons")
            if any(c.state == "MEASURED" for c in chars):
                raise ValueError("an UNKNOWN bend has no MEASURED characteristic")
        elif self.state == "PARTIALLY_FORMED":
            if chars[2].decision == "PASS":
                raise ValueError("a PARTIALLY_FORMED bend cannot have a PASS angle")

        position_measured = chars[3].state == "MEASURED"
        if position_measured != (self.display.measured_polyline_mm is not None):
            raise ValueError("measured_polyline_mm exists exactly when BEND_LINE_POSITION is MEASURED")
        return self


class GeneralMeasurementRowV1(ContractModel):
    measurement_id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=100)
    characteristic: CharacteristicResultV1
    nominal_start_mm: Vec3
    nominal_end_mm: Vec3
    measured_start_mm: Vec3 | None = None
    measured_end_mm: Vec3 | None = None

    @model_validator(mode="after")
    def _validate(self) -> "GeneralMeasurementRowV1":
        c = self.characteristic
        if c.kind != "GENERAL_MEASUREMENT" or c.measurement_id != self.measurement_id:
            raise ValueError("general measurement row must wrap its GENERAL_MEASUREMENT characteristic")
        measured = self.measured_start_mm is not None and self.measured_end_mm is not None
        if (self.measured_start_mm is None) != (self.measured_end_mm is None) or measured != (c.state == "MEASURED"):
            raise ValueError("measured anchors exist exactly when the measurement is MEASURED")
        return self


class UnplannedCreaseV1(ContractModel):
    """A crease the recipe does not plan (extra bend / damage). Always forces REVIEW."""

    crease_id: str = Field(min_length=1, max_length=64)
    reason: Literal["EXTRA_BEND_SUSPECTED", "UNPLANNED_CREASE"]
    polyline_mm: tuple[Vec3, ...] = Field(min_length=2)
    included_deg: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)] | None = None
    note: str = Field(default="", max_length=500)


# --------------------------------------------------------------------------
# Validity, stage decision, summary
# --------------------------------------------------------------------------


class RegistrationV1(ContractModel):
    method: str = Field(min_length=1, max_length=96)
    frame: Literal["cad_model_mm"] = FRAME_CAD_MODEL  # type: ignore[assignment]
    #: Rigid transform applied to raw scan points (after unit scaling) to reach the frame.
    raw_to_frame: Mat4 | None = None
    #: Explicit unit factor applied to raw coordinates. No continuous auto-scale.
    unit_factor: Literal[1.0, 25.4, 1000.0] = 1.0
    datum_fitness: Annotated[float, Field(ge=0, le=1)] | None = None
    rmse_mm: NonNegFloat | None = None
    pose_candidates_considered: int | None = Field(default=None, ge=0)
    #: Score margin between the best and the runner-up pose (incl. symmetric/mirror).
    pose_margin: NonNegFloat | None = None
    pose_ambiguous: bool = False
    #: Amendment ENGINE_IO-A2 (owner decision 2026-10-19): certified symmetric twin poses tie and the
    #: result publishes only the readings identical under every tied twin (same nominal and tolerance of
    #: the matched characteristic, measured values within U, same decision); everything else is UNKNOWN.
    #: Requires ``pose_ambiguous``. Bend identities stay ambiguous: no position, heatmap or stage model.
    identity_ambiguous: bool = False
    #: In-plane pose degenerate: only bend-line position and heatmap become untrusted.
    in_plane_degenerate: bool = False
    abstained: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "RegistrationV1":
        if self.abstained and self.raw_to_frame is not None:
            raise ValueError("an abstained registration publishes no transform")
        if self.identity_ambiguous and not self.pose_ambiguous:
            raise ValueError("identity_ambiguous needs pose_ambiguous")
        if self.raw_to_frame is not None:
            check_rigid_transform(self.raw_to_frame, name="raw_to_frame")
        return self


class CoverageV1(ContractModel):
    surface_coverage_ratio: Annotated[float, Field(ge=0, le=1)] | None = None
    datum_coverage_ratio: Annotated[float, Field(ge=0, le=1)] | None = None
    scan_point_count: int | None = Field(default=None, ge=0)


class InspectionValidityV1(ContractModel):
    state: ValidityState
    reasons: tuple[str, ...] = ()
    metrology_qualified: Literal[False] = False
    registration: RegistrationV1
    coverage: CoverageV1 = Field(default_factory=CoverageV1)

    @model_validator(mode="after")
    def _validate(self) -> "InspectionValidityV1":
        if self.state != "VALID" and not self.reasons:
            raise ValueError("a non-VALID inspection must state reasons")
        if self.state == "VALID" and (self.registration.abstained or self.registration.raw_to_frame is None):
            raise ValueError("a VALID inspection needs a non-abstained registration with a transform")
        return self


class StageDecisionV1(ContractModel):
    inspection_stage: InspectionStage
    declared_stage: int | None = Field(default=None, ge=0)
    declared_stage_source: DeclaredStageSource | None = None
    bends_total: int = Field(ge=0)
    #: Number of formed-like bends when no bend state is UNKNOWN, else null.
    inferred_stage: int | None = Field(default=None, ge=0)
    formed_bend_uids: tuple[str, ...] = ()
    not_yet_formed_bend_uids: tuple[str, ...] = ()
    unknown_bend_uids: tuple[str, ...] = ()
    status: StageStatus
    #: Every applicable status, in precedence order (status == flags[0]).
    flags: tuple[StageStatus, ...]
    decision: StageDecisionValue
    part_complete: bool | None = None
    #: CompiledPart stage model (``S<k>`` or ``S<k>-<tag>``) the scan was decoded
    #: against; used for the heatmap, nominal polylines and backgauge nominals.
    #: Null for an invalid inspection. When ``inferred_stage`` is known its k
    #: equals ``inferred_stage``; without OUT_OF_ORDER_SUSPECTED it is the
    #: prefix model ``S<inferred_stage>``.
    stage_model_id: StageModelId | None = None
    detail: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _validate(self) -> "StageDecisionV1":
        if (self.declared_stage is None) != (self.declared_stage_source is None):
            raise ValueError("declared_stage and declared_stage_source go together")
        if self.declared_stage is not None and self.declared_stage > self.bends_total:
            raise ValueError("declared_stage cannot exceed bends_total")
        if self.stage_model_id is not None and self.inferred_stage is not None:
            if stage_index_of_model_id(self.stage_model_id) != self.inferred_stage:
                raise ValueError("stage_model_id must name a stage with inferred_stage formed bends")
            if "OUT_OF_ORDER_SUSPECTED" not in self.flags and self.stage_model_id != f"S{self.inferred_stage}":
                raise ValueError("an order-consistent decode uses the prefix stage model S<inferred_stage>")
        return self


class WorstCharacteristicV1(ContractModel):
    characteristic_id: str
    bend_uid: str | None = None
    kind: CharacteristicKind
    deviation: CFloat
    uncertainty: CPosFloat
    decision: Decision
    #: (|d| + U) / tolerance limit on the deviation's side; null when that limit is 0.
    exceedance_ratio: CNonNegFloat | None = None


class SummaryV1(ContractModel):
    state: SummaryState
    bends_total: int = Field(ge=0)
    bends_formed: int = Field(ge=0)
    bends_not_yet_formed: int = Field(ge=0)
    bends_unknown: int = Field(ge=0)
    characteristics_required: int = Field(ge=0)
    required_pass: int = Field(ge=0)
    required_fail: int = Field(ge=0)
    required_review: int = Field(ge=0)
    required_unknown: int = Field(ge=0)
    unplanned_creases: int = Field(ge=0)
    worst: WorstCharacteristicV1 | None = None
    reasons: tuple[str, ...] = ()


def iter_characteristics(
    bends: Sequence[BendRowV1], general_measurements: Sequence[GeneralMeasurementRowV1] = ()
) -> list[CharacteristicResultV1]:
    """All characteristics in display order: bends by sequence, then general measurements."""
    out: list[CharacteristicResultV1] = []
    for row in bends:
        out.extend(row.characteristics)
    out.extend(gm.characteristic for gm in general_measurements)
    return out


def required_characteristics(
    bends: Sequence[BendRowV1], general_measurements: Sequence[GeneralMeasurementRowV1] = ()
) -> list[CharacteristicResultV1]:
    return [c for c in iter_characteristics(bends, general_measurements) if c.required]


def derive_stage_decision(
    *,
    inspection_stage: str,
    declared_stage: int | None,
    bends: Sequence[Any],
    pose_ambiguous: bool,
    validity_state: str = "VALID",
) -> dict[str, Any]:
    """Pure stage decision from per-bend states (sequence-ordered rows).

    ``bends`` items need ``bend_uid``, ``sequence_index``, ``machine_move_index``
    and ``state``. Rules:

    - The *rank* of a bend is its sequence index, except that all bends of one
      machine move share the rank of the move's first bend (order inside a
      מהלך is free). The formed set is *order-consistent* iff no not-yet-formed
      bend has a lower rank than a formed one, and no bend is UNEXPECTED_FORMED.
    - ``inferred_stage`` = number of formed-like bends, or ``None`` if any bend
      is UNKNOWN.
    - Flags, in precedence order: NOT_EVALUATED (validity != VALID),
      AMBIGUOUS_POSE, PARTIAL_BEND_FOUND, OUT_OF_ORDER_SUSPECTED,
      STAGE_MISMATCH (declared != inferred), FINAL_INCOMPLETE (FINAL and some
      bend not yet formed), UNDETERMINED (any UNKNOWN), else CONSISTENT.
      ``status`` is the first flag; its decision is PASS (CONSISTENT), REVIEW
      or UNKNOWN. UNDETERMINED ranks below the REVIEW flags so that an unknown
      bend never hides review evidence elsewhere.

    A declared stage is never trusted or overridden silently: disagreement is
    STAGE_MISMATCH and goes to review.
    """
    rows = sorted(bends, key=lambda r: r.sequence_index)
    total = len(rows)
    formed = tuple(r.bend_uid for r in rows if r.state in FORMED_LIKE_STATES)
    not_formed = tuple(r.bend_uid for r in rows if r.state == "NOT_YET_FORMED")
    unknown = tuple(r.bend_uid for r in rows if r.state == "UNKNOWN")

    first_index_of_move: dict[int, int] = {}
    for r in rows:
        if r.machine_move_index is not None:
            first_index_of_move.setdefault(r.machine_move_index, r.sequence_index)

    def rank(r: Any) -> int:
        if r.machine_move_index is not None:
            return first_index_of_move[r.machine_move_index]
        return r.sequence_index

    formed_ranks = [rank(r) for r in rows if r.state in FORMED_LIKE_STATES]
    unformed_ranks = [rank(r) for r in rows if r.state == "NOT_YET_FORMED"]
    order_consistent = not (formed_ranks and unformed_ranks and min(unformed_ranks) < max(formed_ranks))
    if any(r.state == "UNEXPECTED_FORMED" for r in rows):
        order_consistent = False

    inferred = None if unknown else len(formed)
    flags: list[str] = []
    if validity_state != "VALID":
        flags.append("NOT_EVALUATED")
    if pose_ambiguous:
        flags.append("AMBIGUOUS_POSE")
    if unknown:
        flags.append("UNDETERMINED")
    if any(r.state == "PARTIALLY_FORMED" for r in rows):
        flags.append("PARTIAL_BEND_FOUND")
    if not order_consistent:
        flags.append("OUT_OF_ORDER_SUSPECTED")
    if declared_stage is not None and inferred is not None and declared_stage != inferred:
        flags.append("STAGE_MISMATCH")
    if inspection_stage == "FINAL" and not_formed:
        flags.append("FINAL_INCOMPLETE")
    if not flags:
        flags.append("CONSISTENT")
    order = [name for name, _ in _STAGE_PRECEDENCE]
    flags.sort(key=order.index)
    status = flags[0]
    decision = dict(_STAGE_PRECEDENCE)[status]
    if unknown or validity_state != "VALID":
        part_complete = None
    else:
        part_complete = not not_formed and not any(r.state == "PARTIALLY_FORMED" for r in rows)
    return {
        "bends_total": total,
        "inferred_stage": inferred,
        "formed_bend_uids": formed,
        "not_yet_formed_bend_uids": not_formed,
        "unknown_bend_uids": unknown,
        "status": status,
        "flags": tuple(flags),
        "decision": decision,
        "part_complete": part_complete,
    }


#: Bend states that count as "formed as planned" for a complete part.
_COMPLETE_BEND_STATES = frozenset({"FORMED", "UNEXPECTED_FORMED"})


def derive_summary(
    *,
    validity_state: str,
    bends: Sequence[BendRowV1],
    general_measurements: Sequence[GeneralMeasurementRowV1] = (),
    stage_decision: str = "PASS",
    stage_status: str = "CONSISTENT",
    unplanned_creases: int = 0,
) -> SummaryV1:
    """Pure part summary (mirrored by the station UI).

    Precedence (first match wins):

    1. validity != VALID -> INVALID_INSPECTION
    2. any required FAIL -> OUT_OF_TOLERANCE_FOUND
    3. any required REVIEW, a REVIEW stage decision, or any unplanned crease
       -> REVIEW_REQUIRED
    4. any required UNKNOWN, or an UNKNOWN stage decision -> INCOMPLETE
    5. no required characteristic at all (nothing was measured, e.g. every
       bend NOT_YET_FORMED) -> INCOMPLETE (reason NO_REQUIRED_MEASURED)
    6. some planned bend is not FORMED / UNEXPECTED_FORMED (the part is not
       finished; only reachable under IN_PROCESS, FINAL is already REVIEW via
       FINAL_INCOMPLETE) -> STAGE_IN_TOLERANCE (reason PART_NOT_COMPLETE),
       never shown green
    7. otherwise ALL_MEASURED_IN_TOLERANCE: every planned bend is formed and
       every required characteristic is MEASURED and PASS

    Optional characteristics never change the state, but on a valid result an
    optional FAIL / REVIEW adds the reason ``OPTIONAL_FAIL`` / ``OPTIONAL_REVIEW``
    so a red or amber row is never shown under an unexplained banner.
    """
    required = required_characteristics(bends, general_measurements)
    counts = {d: sum(1 for c in required if c.decision == d) for d in ("PASS", "FAIL", "REVIEW", "UNKNOWN")}
    reasons: list[str] = []
    if validity_state != "VALID":
        state = "INVALID_INSPECTION"
        reasons.append(f"VALIDITY:{validity_state}")
    elif counts["FAIL"]:
        state = "OUT_OF_TOLERANCE_FOUND"
    elif counts["REVIEW"] or stage_decision == "REVIEW" or unplanned_creases:
        state = "REVIEW_REQUIRED"
    elif counts["UNKNOWN"] or stage_decision == "UNKNOWN":
        state = "INCOMPLETE"
    elif not required:
        state = "INCOMPLETE"
        reasons.append("NO_REQUIRED_MEASURED")
    elif not all(r.state in _COMPLETE_BEND_STATES for r in bends):
        state = "STAGE_IN_TOLERANCE"
        reasons.append("PART_NOT_COMPLETE")
    else:
        state = "ALL_MEASURED_IN_TOLERANCE"
    if validity_state == "VALID":
        if counts["FAIL"]:
            reasons.append("REQUIRED_FAIL")
        if counts["REVIEW"]:
            reasons.append("REQUIRED_REVIEW")
        if counts["UNKNOWN"]:
            reasons.append("REQUIRED_UNKNOWN")
        if stage_decision != "PASS":
            reasons.append(f"STAGE:{stage_status}")
        if unplanned_creases:
            reasons.append("UNPLANNED_CREASES")
        optional = [c for c in iter_characteristics(bends, general_measurements) if not c.required]
        if any(c.decision == "FAIL" for c in optional):
            reasons.append("OPTIONAL_FAIL")
        if any(c.decision == "REVIEW" for c in optional):
            reasons.append("OPTIONAL_REVIEW")

    worst: WorstCharacteristicV1 | None = None
    worst_key: float | None = None
    if validity_state == "VALID":
        for c in required:
            if c.state != "MEASURED" or c.deviation is None or c.tolerance is None or c.uncertainty is None:
                continue
            ratio = exceedance_ratio(c.deviation, c.uncertainty, c.tolerance.plus, c.tolerance.minus)
            key = math.inf if ratio is None else ratio
            if worst_key is None or key > worst_key:
                worst_key = key
                worst = WorstCharacteristicV1(
                    characteristic_id=c.characteristic_id,
                    bend_uid=c.bend_uid,
                    kind=c.kind,
                    deviation=c.deviation,
                    uncertainty=c.uncertainty,
                    decision=c.decision,
                    exceedance_ratio=None if ratio is None else round6(ratio),
                )
    return SummaryV1(
        state=state,  # type: ignore[arg-type]
        bends_total=len(bends),
        bends_formed=sum(1 for r in bends if r.state in FORMED_LIKE_STATES),
        bends_not_yet_formed=sum(1 for r in bends if r.state == "NOT_YET_FORMED"),
        bends_unknown=sum(1 for r in bends if r.state == "UNKNOWN"),
        characteristics_required=len(required),
        required_pass=counts["PASS"],
        required_fail=counts["FAIL"],
        required_review=counts["REVIEW"],
        required_unknown=counts["UNKNOWN"],
        unplanned_creases=unplanned_creases,
        worst=worst,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------
# Result document
# --------------------------------------------------------------------------


class ViewerAssetsV1(ContractModel):
    """Files written next to the result (paths relative to the result's directory)."""

    frame: Literal["cad_model_mm"] = FRAME_CAD_MODEL  # type: ignore[assignment]
    registered_scan: RelPath | None = None
    point_count: int | None = Field(default=None, ge=0)
    #: float32 little-endian, index-aligned with registered_scan vertices, NaN = no value.
    deviations: RelPath | None = None
    deviation_encoding: Literal["float32_le_nan_missing"] = "float32_le_nan_missing"
    deviation_sign: Literal["SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL"] = "SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL"
    #: Stage model id of the heatmap reference; equals ``stage_decision.stage_model_id``.
    deviation_reference_stage: StageModelId | None = None
    heatmap_band_mm: PosFloat
    heatmap_unavailable_reason: str | None = Field(default=None, max_length=200)
    #: uint16 little-endian per-point segment labels (optional).
    segment_labels: RelPath | None = None
    stage_mesh: RelPath | None = None
    final_mesh: RelPath | None = None
    bend_geometry: RelPath | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ViewerAssetsV1":
        if self.deviations is not None:
            if self.registered_scan is None or self.point_count is None or self.deviation_reference_stage is None:
                raise ValueError("deviations need registered_scan, point_count and deviation_reference_stage")
            if self.heatmap_unavailable_reason is not None:
                raise ValueError("a published heatmap has no unavailable reason")
        elif not self.heatmap_unavailable_reason:
            raise ValueError("without deviations, heatmap_unavailable_reason is required")
        if self.segment_labels is not None and self.registered_scan is None:
            raise ValueError("segment_labels need registered_scan")
        return self


class ArtifactsV1(ContractModel):
    pdf: RelPath | None = None
    overview_png: RelPath | None = None


class EngineInfoV1(ContractModel):
    engine_name: Literal["sherman_production_engine"] = "sherman_production_engine"
    engine_version: str = Field(min_length=1, max_length=64)
    config_sha256: Sha256Hex | None = None
    bundle_id: str | None = Field(default=None, max_length=128)
    compiled_part_sha256: Sha256Hex
    algorithm_versions: dict[str, str] = Field(default_factory=dict)
    duration_s: NonNegFloat | None = None
    peak_rss_bytes: int | None = Field(default=None, ge=0)


class ScanInfoV1(ContractModel):
    sha256: Sha256Hex
    format: Literal["PLY_ASCII", "PLY_BINARY_LE", "PLY_BINARY_BE", "STL_BINARY", "STL_ASCII"]
    vertex_count: int = Field(ge=0)
    face_count: int = Field(ge=0)


class ResultContextV1(ContractModel):
    """Job identity, added after the engine (ENGINE_IO §4, §5).

    E3 has no ``--context`` option and writes this block empty (plan §10
    amendment 4: no filename tokens or metadata reach the engine). C06 (or the
    runner) fills it from ``input/context.json`` when it writes
    ``result/shopfloor_result.json``. No validator and no measurement reads it.
    """

    job_id: str | None = Field(default=None, max_length=64)
    station_id: str | None = Field(default=None, max_length=64)
    scan_file_name: str | None = Field(default=None, max_length=255)
    serial: str | None = Field(default=None, max_length=64)
    scan_timestamp: str | None = Field(default=None, max_length=40)
    received_at: UtcIsoTimestamp | None = None
    #: Display label per bend: the label the operator tagged in the studio (the
    #: recipe's ``cad_label``). ``BendRowV1.label`` is the recipe's sequence label
    #: (``B<sequence_index>``); the station and the PDF show this one instead so a
    #: bend has the same name in the studio, on the station and in the report.
    #: Display only: no validator, measurement or bend identity reads it. Left out
    #: of the JSON when unset, so results without it serialise byte-identically.
    bend_display_labels: dict[BendUid, Annotated[str, Field(min_length=1, max_length=100)]] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )


class RecipeRefV1(ContractModel):
    recipe_id: Annotated[str, Field(pattern=RECIPE_ID_PATTERN)]
    version: int = Field(ge=1)
    recipe_sha256: Sha256Hex
    content_sha256: Sha256Hex


class AuthorityV1(ContractModel):
    mode: Literal["OPERATOR_ASSIST"] = "OPERATOR_ASSIST"
    production_disposition: Literal["NOT_AUTHORIZED"] = "NOT_AUTHORIZED"
    note: str = Field(
        default="Operator assist only; not an automatic accept/reject.", max_length=500
    )


class ShopfloorInspectionResultV1(ContractModel):
    schema_: Literal["sherman_shopfloor_inspection_result_v1"] = Field(
        default=SHOPFLOOR_RESULT_SCHEMA, alias="schema"
    )
    created_at: UtcIsoTimestamp
    context: ResultContextV1 = Field(default_factory=ResultContextV1)
    sku: Annotated[str, Field(pattern=SKU_PATTERN)]
    part_name: str | None = Field(default=None, max_length=255)
    recipe: RecipeRefV1
    scan: ScanInfoV1
    engine: EngineInfoV1
    authority: AuthorityV1 = Field(default_factory=AuthorityV1)
    inspection_validity: InspectionValidityV1
    stage_decision: StageDecisionV1
    summary: SummaryV1
    bends: tuple[BendRowV1, ...]
    general_measurements: tuple[GeneralMeasurementRowV1, ...] = ()
    unplanned_creases: tuple[UnplannedCreaseV1, ...] = ()
    viewer_assets: ViewerAssetsV1
    artifacts: ArtifactsV1 = Field(default_factory=ArtifactsV1)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> "ShopfloorInspectionResultV1":
        if self.recipe.recipe_id != f"{self.sku}@v{self.recipe.version}":
            raise ValueError("recipe.recipe_id must be '<sku>@v<version>'")
        indices = [row.sequence_index for row in self.bends]
        if indices != list(range(1, len(self.bends) + 1)):
            raise ValueError("bend rows must be sorted by a contiguous sequence_index 1..N")
        uids = [row.bend_uid for row in self.bends]
        if len(set(uids)) != len(uids):
            raise ValueError("bend rows must not repeat a bend_uid")
        gm_ids = [gm.measurement_id for gm in self.general_measurements]
        if len(set(gm_ids)) != len(gm_ids):
            raise ValueError("general measurement rows must be unique")
        crease_ids = [c.crease_id for c in self.unplanned_creases]
        if len(set(crease_ids)) != len(crease_ids):
            raise ValueError("unplanned crease ids must be unique")
        for row in self.bends:
            for char in row.characteristics:
                if char.adjacent_bend_uid is not None and char.adjacent_bend_uid not in uids:
                    raise ValueError("adjacent_bend_uid must name a bend of this result")

        validity = self.inspection_validity
        registration = validity.registration
        all_chars = iter_characteristics(self.bends, self.general_measurements)
        if any(c.reason == "NOT_APPLICABLE_AT_STAGE" for c in all_chars) and not any(
            row.state == "NOT_YET_FORMED" for row in self.bends
        ):
            # A stage can only make a value inapplicable while something is still unformed;
            # otherwise a finished part could skip a required value and still go green.
            raise ValueError("NOT_APPLICABLE_AT_STAGE needs a NOT_YET_FORMED bend")
        state_of = {row.bend_uid: row.state for row in self.bends}
        for row in self.bends:
            for char in row.characteristics:
                if char.kind == "FLANGE_LENGTH" and char.reason == "NOT_APPLICABLE_AT_STAGE" and (
                    char.adjacent_bend_uid is None or state_of.get(char.adjacent_bend_uid) != "NOT_YET_FORMED"
                ):
                    # amendment ENGINE_IO-A1: only the stage (an unformed bounding bend) can make a flange
                    # length undefined; an unknown or unmeasured neighbour stays UNKNOWN(ADJACENT_BEND_NOT_MEASURED)
                    raise ValueError("a FLANGE_LENGTH is NOT_APPLICABLE_AT_STAGE only when its adjacent bend is "
                                     "NOT_YET_FORMED")
        if validity.state != "VALID":
            if any(c.state == "MEASURED" for c in all_chars):
                raise ValueError("an invalid inspection has no MEASURED characteristic")
            if any(row.state != "UNKNOWN" for row in self.bends):
                raise ValueError("an invalid inspection cannot claim a bend state")
            if self.viewer_assets.deviations is not None:
                raise ValueError("an invalid inspection publishes no heatmap")
            if self.stage_decision.stage_model_id is not None:
                raise ValueError("an invalid inspection decodes no stage model")
        if self.viewer_assets.deviations is not None and (
            self.stage_decision.stage_model_id is None
            or self.viewer_assets.deviation_reference_stage != self.stage_decision.stage_model_id
        ):
            raise ValueError("the heatmap reference stage must equal stage_decision.stage_model_id")
        if registration.pose_ambiguous and not registration.identity_ambiguous:
            # Identities cannot be assigned: formed bends stay UNKNOWN(AMBIGUOUS_POSE);
            # NOT_YET_FORMED is allowed only because every pose hypothesis agrees on it.
            if any(c.state == "MEASURED" for c in all_chars):
                raise ValueError("an ambiguous pose measures nothing")
            for row in self.bends:
                if row.state == "NOT_YET_FORMED":
                    continue
                if row.state != "UNKNOWN" or "AMBIGUOUS_POSE" not in row.state_reasons:
                    raise ValueError("an ambiguous pose leaves bends UNKNOWN (AMBIGUOUS_POSE) or NOT_YET_FORMED")
            if self.viewer_assets.deviations is not None:
                raise ValueError("an ambiguous pose publishes no heatmap")
        if registration.identity_ambiguous:
            # Amendment ENGINE_IO-A2: only readings identical under every tied twin are published (the engine's
            # agreement test); a bend whose state differs between the twins stays UNKNOWN(AMBIGUOUS_POSE). The
            # bend lines' places depend on the identity, so no position, heatmap or stage model is published.
            for row in self.bends:
                if row.state == "UNKNOWN" and "AMBIGUOUS_POSE" not in row.state_reasons:
                    raise ValueError("an identity-ambiguous result leaves unknown bends UNKNOWN (AMBIGUOUS_POSE)")
                if row.characteristics[3].state == "MEASURED":
                    raise ValueError("an identity-ambiguous result measures no bend-line position")
            if self.viewer_assets.deviations is not None:
                raise ValueError("an ambiguous pose publishes no heatmap")
            if self.stage_decision.stage_model_id is not None:
                raise ValueError("an identity-ambiguous result decodes no stage model")
        if registration.in_plane_degenerate:
            # Only position and heatmap are withheld; on a bend whose state is
            # known the position is UNKNOWN(REGISTRATION_UNTRUSTED).
            for row in self.bends:
                pos = row.characteristics[3]
                if pos.state == "MEASURED":
                    raise ValueError("a degenerate in-plane pose cannot measure bend-line position")
                if row.state in ("NOT_YET_FORMED", "UNKNOWN"):
                    continue
                if pos.state != "NOT_MEASURED" or pos.reason != "REGISTRATION_UNTRUSTED":
                    raise ValueError(
                        "a degenerate in-plane pose reports bend-line position as UNKNOWN(REGISTRATION_UNTRUSTED)"
                    )
            if self.viewer_assets.deviations is not None:
                raise ValueError("a degenerate in-plane pose publishes no heatmap")

        derived_stage = derive_stage_decision(
            inspection_stage=self.stage_decision.inspection_stage,
            declared_stage=self.stage_decision.declared_stage,
            bends=self.bends,
            pose_ambiguous=registration.pose_ambiguous,
            validity_state=validity.state,
        )
        for key, value in derived_stage.items():
            if getattr(self.stage_decision, key) != value:
                raise ValueError(f"stage_decision.{key} must equal the derived value {value!r}")

        expected_summary = derive_summary(
            validity_state=validity.state,
            bends=self.bends,
            general_measurements=self.general_measurements,
            stage_decision=self.stage_decision.decision,
            stage_status=self.stage_decision.status,
            unplanned_creases=len(self.unplanned_creases),
        )
        if self.summary != expected_summary:
            raise ValueError(
                f"summary must equal derive_summary (expected state {expected_summary.state})"
            )
        return self


class BendGeometryEntryV1(ContractModel):
    bend_uid: BendUid
    label: str
    sequence_index: int = Field(ge=1)
    state: BendState
    nominal_polyline_mm: tuple[Vec3, ...] = Field(min_length=2)
    measured_polyline_mm: tuple[Vec3, ...] | None = Field(default=None, min_length=2)
    label_anchor_mm: Vec3
    side_a_anchor_mm: Vec3 | None = None
    side_b_anchor_mm: Vec3 | None = None


class BendGeometryFileV1(ContractModel):
    """``bend_geometry.json``: bend polylines + label anchors for the viewer."""

    schema_: Literal["sherman_bend_geometry_v1"] = Field(default=BEND_GEOMETRY_SCHEMA, alias="schema")
    frame: Literal["cad_model_mm"] = FRAME_CAD_MODEL  # type: ignore[assignment]
    bends: tuple[BendGeometryEntryV1, ...]
    unplanned_creases: tuple[UnplannedCreaseV1, ...] = ()

    @classmethod
    def from_result(cls, result: ShopfloorInspectionResultV1) -> "BendGeometryFileV1":
        return cls(
            bends=tuple(
                BendGeometryEntryV1(
                    bend_uid=row.bend_uid,
                    label=row.label,
                    sequence_index=row.sequence_index,
                    state=row.state,
                    nominal_polyline_mm=row.display.nominal_polyline_mm,
                    measured_polyline_mm=row.display.measured_polyline_mm,
                    label_anchor_mm=row.display.label_anchor_mm,
                    side_a_anchor_mm=row.display.side_a_anchor_mm,
                    side_b_anchor_mm=row.display.side_b_anchor_mm,
                )
                for row in result.bends
            ),
            unplanned_creases=result.unplanned_creases,
        )


# --------------------------------------------------------------------------
# Station feed
# --------------------------------------------------------------------------


class StationLatestCompletedV1(ContractModel):
    job: LineJobSummaryV1
    result: ShopfloorInspectionResultV1
    #: served artifact name -> URL (/api/line/jobs/{job_id}/artifacts/{name})
    artifacts: dict[str, str] = Field(default_factory=dict)


class StationActiveV1(ContractModel):
    job: LineJobSummaryV1


class StationIssueV1(ContractModel):
    kind: Literal["QUARANTINE", "JOB_FAILED"]
    at: UtcIsoTimestamp
    file_name: str | None = Field(default=None, max_length=255)
    reason_code: str = Field(min_length=1, max_length=64)
    message: str = Field(default="", max_length=1000)


class StationFeedV1(ContractModel):
    schema_: Literal["sherman_station_feed_v1"] = Field(default=STATION_FEED_SCHEMA, alias="schema")
    station_id: str = Field(min_length=1, max_length=64)
    etag: str = Field(min_length=1, max_length=128)
    server_time: UtcIsoTimestamp
    latest_completed: StationLatestCompletedV1 | None = None
    active: StationActiveV1 | None = None
    latest_issue: StationIssueV1 | None = None
    recent: tuple[LineJobSummaryV1, ...] = Field(default=(), max_length=10)
