"""PartRecipeV1: the immutable, hash-bound, versioned onboarding output.

A recipe is released once per (SKU, version). It freezes *what* to inspect:
the folded STEP CAD, the drawing, an optional flat (DXF or flat STEP), the
operator-confirmed bend inventory with stable ``bend_uid`` identities, the
bending order (``sequence_index``), the מהלך (machine-move) groups, general
measurements, tolerances and the inspection-stage policy. The per-scan engine
only answers *how far* each planned characteristic deviates.

Conventions (plan §3.1 and CONTRACTS §0):

- Angles are INCLUDED angles in degrees (180 = flat). Native STEP candidates
  store the sweep, so ``included = 180 - sweep``.
- Flange length is OUTSIDE_MOLD_LINE (OML virtual sharp to the far boundary).
- Bend-line position is the OML virtual sharp in the CAD model frame.
- ``machine_move_index`` is 1-based and indexes ``machine_moves``.

Hashes:

- ``content_sha256`` covers every field except the release bookkeeping
  (``recipe_id``, ``recipe_version``, ``supersedes_version``, ``released_at``,
  ``released_by``) and the two hashes. Two releases with identical inspection
  content share it.
- ``recipe_sha256`` covers every field except itself.

Both are recomputed by a model validator; a mismatch is rejected. Use
:meth:`PartRecipeV1.finalize` to build a recipe with correct hashes.
"""
from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationInfo, model_validator

from contracts.production.canonical import sha256_hex, strip_keys
from contracts.production.common import (
    ContractModel,
    PosFloat,
    RelPath,
    Sha256Hex,
    Tolerance,
    ToleranceSource,
    UtcIsoTimestamp,
    Vec3,
)

__all__ = [
    "PART_RECIPE_SCHEMA",
    "SKU_PATTERN",
    "BEND_UID_PATTERN",
    "RECIPE_ID_PATTERN",
    "RECIPE_BUILD_REASON_CODES",
    "RecipeBuildCode",
    "INSPECTION_STAGES",
    "InspectionStage",
    "RecipeConfirmation",
    "Units",
    "CadRef",
    "DrawingRef",
    "FlatBindingV1",
    "FlatRef",
    "ReleasedBy",
    "SourceDraftRef",
    "DefaultTolerances",
    "RegistrationPolicy",
    "CoveragePolicy",
    "AngleNominal",
    "BendTolerances",
    "FlangeInspection",
    "RecipeBendV1",
    "RecipeGeneralMeasurementV1",
    "RemovedCandidate",
    "PartRecipeV1",
    "BendOverrideV1",
    "MeasurementOverrideV1",
    "ReleaseRequestV1",
    "make_recipe_id",
    "RECIPE_CONTENT_EXCLUDED_KEYS",
]

PART_RECIPE_SCHEMA = "sherman_part_recipe_v1"
SKU_PATTERN = r"^[A-Z0-9][A-Z0-9._-]{0,63}$"
BEND_UID_PATTERN = r"^bu_[0-9a-f]{20}$"
RECIPE_ID_PATTERN = r"^[A-Z0-9][A-Z0-9._-]{0,63}@v[1-9][0-9]*$"

RecipeBuildCode = Literal[
    "DRAFT_MISSING",
    "DRAFT_STALE",
    "DRAFT_INVALID",
    "DRAFT_REVISION_CHANGED",
    "SEQUENCE_NOT_CONFIRMED",
    "NO_ACTIVE_BENDS",
    "UNCONFIRMED_ACTIVE_MARKS",
    "MARK_WITHOUT_LINE",
    "MANUAL_MARK_MISSING_NOMINAL_ANGLE",
    "DUPLICATE_BEND_IDENTITY",
    "CAD_MISSING",
    "CAD_HASH_MISMATCH",
    "DRAWING_MISSING",
    "DRAWING_HASH_MISMATCH",
    "FLAT_HASH_MISMATCH",
    "FLAT_BINDING_UNKNOWN_BEND",
    "SKU_INVALID",
    "SKU_COLLISION",
    "TOLERANCES_MISSING",
    "TOLERANCE_INVALID",
    "OVERRIDE_UNKNOWN_BEND",
    "OVERRIDE_UNKNOWN_MEASUREMENT",
    "MEASUREMENT_INVALID",
    "CONCURRENT_RELEASE",
    "RELEASE_DRAFT_REVISION_CHANGED",
]
#: Every reason code :class:`domains.production.recipe.builder.RecipeBuildError`
#: (and the C05 release gate) may emit. C11's parity test imports this.
RECIPE_BUILD_REASON_CODES: frozenset[str] = frozenset(RecipeBuildCode.__args__)  # type: ignore[attr-defined]

InspectionStage = Literal["FINAL", "IN_PROCESS"]
INSPECTION_STAGES: frozenset[str] = frozenset(InspectionStage.__args__)  # type: ignore[attr-defined]
RecipeConfirmation = Literal["OPERATOR_CONFIRMED", "DEV_AUTO_CONFIRMED"]

BendUid = Annotated[str, Field(pattern=BEND_UID_PATTERN)]
FeatureId = Annotated[str, Field(min_length=1, max_length=100)]

#: Fields excluded from ``content_sha256`` (release bookkeeping + hashes).
RECIPE_CONTENT_EXCLUDED_KEYS: tuple[str, ...] = (
    "recipe_id",
    "recipe_version",
    "supersedes_version",
    "released_at",
    "released_by",
    "content_sha256",
    "recipe_sha256",
)


def make_recipe_id(sku: str, version: int) -> str:
    """``"<SKU>@v<version>"``."""
    return f"{sku}@v{int(version)}"


def _ext_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name[name.rfind("."):].lower() if "." in name else ""


class Units(ContractModel):
    length: Literal["mm"] = "mm"
    angle: Literal["deg"] = "deg"
    #: True only when the CAD units were verified as mm (not merely assumed).
    cad_units_verified: bool


class CadRef(ContractModel):
    """Folded CAD, frozen by content address at release. Production requires STEP."""

    sha256: Sha256Hex
    original_filename: str | None = Field(default=None, max_length=255)
    stored_rel_path: RelPath
    format: Literal["STEP"] = "STEP"
    #: Catalog candidate generation (backend/part_catalog.py:813-818) the draft was bound to.
    candidate_generation: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    candidate_angle_convention: Literal["NATIVE_SWEEP", "MESH_INCLUDED_FOLDED"]

    @model_validator(mode="after")
    def _path_is_content_addressed(self) -> "CadRef":
        ext = _ext_of(self.stored_rel_path)
        if ext not in (".step", ".stp") or self.stored_rel_path != f"recipes/cad/{self.sha256}{ext}":
            raise ValueError("cad.stored_rel_path must be recipes/cad/<sha256>.step|.stp")
        return self


class DrawingRef(ContractModel):
    sha256: Sha256Hex
    original_filename: str | None = Field(default=None, max_length=255)
    stored_rel_path: RelPath
    format: Literal["PDF", "PNG", "JPEG", "TIFF", "DXF", "DWG"] = "PDF"

    @model_validator(mode="after")
    def _path_is_content_addressed(self) -> "DrawingRef":
        ext = _ext_of(self.stored_rel_path)
        if not ext or self.stored_rel_path != f"recipes/drawings/{self.sha256}{ext}":
            raise ValueError("drawing.stored_rel_path must be recipes/drawings/<sha256><ext>")
        return self


class ReleasedBy(ContractModel):
    user_id: str = Field(min_length=1, max_length=128)
    username: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1, max_length=64)


class FlatBindingV1(ContractModel):
    """How the flat's bend lines map to recipe ``bend_uid`` values.

    ``UNBOUND``: the flat is stored but no bend line is bound (the map is empty).
    ``AUTO_PROPOSED``: a panel-tree graph match + 2D rigid/mirror fit proposed
    the map; not yet confirmed. ``OPERATOR_CONFIRMED``: confirmed in the wizard.
    """

    state: Literal["UNBOUND", "AUTO_PROPOSED", "OPERATOR_CONFIRMED"]
    #: flat bend-line id (as emitted by the E1 flat parser) -> recipe bend_uid.
    bend_line_map: dict[Annotated[str, Field(min_length=1, max_length=100)], BendUid] = Field(
        default_factory=dict
    )
    method: Literal["GRAPH_MATCH_RIGID_FIT", "MANUAL"] | None = None
    mirrored: bool | None = None
    confirmed_by: ReleasedBy | None = None
    confirmed_at: UtcIsoTimestamp | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "FlatBindingV1":
        if self.state == "UNBOUND":
            if self.bend_line_map or self.method is not None:
                raise ValueError("an UNBOUND flat carries no bend-line map or method")
        elif not self.bend_line_map or self.method is None:
            raise ValueError("a bound flat needs a non-empty bend-line map and a method")
        values = list(self.bend_line_map.values())
        if len(values) != len(set(values)):
            raise ValueError("each bend_uid may be bound to at most one flat bend line")
        if (self.state == "OPERATOR_CONFIRMED") != (self.confirmed_by is not None and self.confirmed_at is not None):
            raise ValueError("confirmed_by/confirmed_at are required exactly when OPERATOR_CONFIRMED")
        return self


class FlatRef(ContractModel):
    """Optional unfolded part: DXF (bend lines by layer/linetype/annotation) or flat STEP."""

    sha256: Sha256Hex
    original_filename: str | None = Field(default=None, max_length=255)
    stored_rel_path: RelPath
    format: Literal["DXF", "STEP"]
    binding: FlatBindingV1

    @model_validator(mode="after")
    def _path_is_content_addressed(self) -> "FlatRef":
        ext = _ext_of(self.stored_rel_path)
        allowed = (".dxf",) if self.format == "DXF" else (".step", ".stp")
        if ext not in allowed or self.stored_rel_path != f"recipes/flats/{self.sha256}{ext}":
            raise ValueError("flat.stored_rel_path must be recipes/flats/<sha256><ext matching format>")
        return self


class SourceDraftRef(ContractModel):
    draft_revision: int = Field(ge=1)
    draft_sha256: Sha256Hex
    vendor_schema_version: int = Field(ge=1)


class DefaultTolerances(ContractModel):
    angle_deg: Tolerance
    flange_mm: Tolerance
    bend_line_position_mm: Tolerance | None = None
    #: Mid-sequence backgauge length; falls back to ``flange_mm`` when null.
    backgauge_mm: Tolerance | None = None
    #: General measurements; falls back to ``flange_mm`` when null.
    general_mm: Tolerance | None = None
    #: Half-width of the signed heatmap colour band (not a conformance tolerance).
    surface_profile_mm: PosFloat = 0.5
    source: ToleranceSource
    note: str = Field(default="", max_length=500)


class RegistrationPolicy(ContractModel):
    """Datum-registration acceptance limits used by the per-scan engine."""

    #: Minimum inlier fraction of the datum island after fine registration.
    min_datum_fitness: Annotated[float, Field(gt=0, le=1)] = 0.55
    max_rmse_mm: PosFloat | None = None


class CoveragePolicy(ContractModel):
    min_surface_coverage: Annotated[float, Field(ge=0, le=1)] = 0.25
    min_side_support_points: int = Field(default=200, ge=1)
    min_edge_coverage: Annotated[float, Field(ge=0, le=1)] = 0.6


class AngleNominal(ContractModel):
    """Nominal INCLUDED angle and, when known, the sweep (= 180 - included).

    A planned bend is never flat: ``0 < included < 180`` and ``0 < sweep < 180``
    (the same bounds as ``CompiledBendV1``), so a flat "bend" is refused at the
    release gate instead of failing later in the E1 compile.
    """

    included_deg: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)]
    sweep_deg: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)] | None = None
    source: Literal["CAD_NATIVE_SWEEP", "CAD_MESH_FOLDED", "MANUAL_ENTRY", "COMPILED_FLANGE_GRAPH"]
    convention_verified: bool

    @model_validator(mode="after")
    def _sweep_consistent(self) -> "AngleNominal":
        if self.sweep_deg is not None and not math.isclose(
            self.included_deg, 180.0 - self.sweep_deg, abs_tol=1e-6
        ):
            raise ValueError("included_deg must equal 180 - sweep_deg")
        return self


class BendTolerances(ContractModel):
    """Per-bend overrides; ``None`` means the recipe default applies."""

    angle_deg: Tolerance | None = None
    flange_a_mm: Tolerance | None = None
    flange_b_mm: Tolerance | None = None
    bend_line_position_mm: Tolerance | None = None
    backgauge_mm: Tolerance | None = None


FlangeSideRequirement = Literal["REQUIRED", "NOT_INSPECTED"]


class FlangeInspection(ContractModel):
    side_a: FlangeSideRequirement = "REQUIRED"
    side_b: FlangeSideRequirement = "REQUIRED"


class RecipeBendV1(ContractModel):
    bend_uid: BendUid
    sequence_index: int = Field(ge=1)
    label: str = Field(min_length=1, max_length=32)
    cad_label: str = Field(min_length=1, max_length=100)
    source: Literal["cad_candidate", "manual"]
    source_feature_id: FeatureId
    corrects_feature_id: FeatureId | None = None
    machine_move_index: int | None = Field(default=None, ge=1)
    #: Operator-confirmed bend line from the draft (CAD model frame, mm). This is
    #: the *identity* line; the compiled OML locus lives in the CompiledPart.
    line_start_mm: Vec3
    line_end_mm: Vec3
    nominal_angle: AngleNominal
    nominal_inner_radius_mm: PosFloat | None = None
    tolerances: BendTolerances = Field(default_factory=BendTolerances)
    flange_inspection: FlangeInspection = Field(default_factory=FlangeInspection)
    extent_semantics: Literal["UNKNOWN", "FULL", "INTERIOR_CORE"] = "UNKNOWN"

    @model_validator(mode="after")
    def _line_nondegenerate(self) -> "RecipeBendV1":
        if math.dist(self.line_start_mm, self.line_end_mm) <= 1e-9:
            raise ValueError("bend line endpoints must differ")
        return self


class RecipeGeneralMeasurementV1(ContractModel):
    """Operator-tagged straight-line characteristic (Vendor Setup general measurement).

    ``nominal_mm`` is the operator target when given, else the CAD anchor distance.
    """

    measurement_id: Annotated[str, Field(min_length=1, max_length=100)]
    label: str = Field(min_length=1, max_length=100)
    kind: Literal["bend_pair", "points"]
    bend_uids: tuple[BendUid, ...] = ()
    point_start_mm: Vec3
    point_end_mm: Vec3
    cad_distance_mm: PosFloat
    nominal_mm: PosFloat
    nominal_source: Literal["OPERATOR_TARGET", "CAD_DISTANCE"]
    tolerance: Tolerance | None = None
    required: bool = True
    note: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _shape(self) -> "RecipeGeneralMeasurementV1":
        if self.kind == "bend_pair":
            if len(self.bend_uids) != 2 or len(set(self.bend_uids)) != 2:
                raise ValueError("bend_pair measurements need two distinct bend_uids")
        elif self.bend_uids:
            raise ValueError("point measurements cannot reference bends")
        if not math.isclose(
            math.dist(self.point_start_mm, self.point_end_mm), self.cad_distance_mm, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("cad_distance_mm must equal the anchor distance")
        if self.nominal_source == "CAD_DISTANCE" and not math.isclose(
            self.nominal_mm, self.cad_distance_mm, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("a CAD_DISTANCE nominal must equal cad_distance_mm")
        return self


class RemovedCandidate(ContractModel):
    feature_id: FeatureId
    label: str = Field(min_length=1, max_length=100)
    issue: Literal["missing", "wrong_location", "not_a_bend"]
    note: str = Field(default="", max_length=500)


class PartRecipeV1(ContractModel):
    schema_: Literal["sherman_part_recipe_v1"] = Field(default=PART_RECIPE_SCHEMA, alias="schema")
    recipe_id: Annotated[str, Field(pattern=RECIPE_ID_PATTERN)]
    sku: Annotated[str, Field(pattern=SKU_PATTERN)]
    sku_display: str = Field(min_length=1, max_length=128)
    part_id: str = Field(min_length=1, max_length=128)
    part_name: str | None = Field(default=None, max_length=255)
    part_revision: str | None = Field(default=None, max_length=32)
    material: str | None = Field(default=None, max_length=128)
    thickness_mm: PosFloat | None = None
    recipe_version: int = Field(ge=1)
    supersedes_version: int | None = Field(default=None, ge=1)
    released_at: UtcIsoTimestamp
    released_by: ReleasedBy
    confirmation: RecipeConfirmation = "OPERATOR_CONFIRMED"
    units: Units
    cad: CadRef
    drawing: DrawingRef
    flat: FlatRef | None = None
    source_draft: SourceDraftRef
    flange_length_convention: Literal["OUTSIDE_MOLD_LINE"] = "OUTSIDE_MOLD_LINE"
    angle_convention: Literal["INCLUDED_DEG"] = "INCLUDED_DEG"
    bend_line_convention: Literal["OML_VIRTUAL_SHARP"] = "OML_VIRTUAL_SHARP"
    #: Default expectation for scans of this recipe. IN_PROCESS: the engine infers
    #: the stage per bend; FINAL: every planned bend is expected formed.
    inspection_stage: InspectionStage
    default_tolerances: DefaultTolerances
    registration_policy: RegistrationPolicy = Field(default_factory=RegistrationPolicy)
    coverage_policy: CoveragePolicy = Field(default_factory=CoveragePolicy)
    bends: tuple[RecipeBendV1, ...] = Field(min_length=1)
    #: Ordered contiguous groups of the sequence (מהלך). Bends outside a group
    #: keep their order but carry no machine-phase claim.
    machine_moves: tuple[tuple[BendUid, ...], ...] = ()
    general_measurements: tuple[RecipeGeneralMeasurementV1, ...] = ()
    removed_candidates: tuple[RemovedCandidate, ...] = ()
    release_note: str = Field(default="", max_length=2000)
    content_sha256: Sha256Hex
    recipe_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate(self, info: ValidationInfo) -> "PartRecipeV1":
        if self.recipe_id != make_recipe_id(self.sku, self.recipe_version):
            raise ValueError("recipe_id must be '<sku>@v<recipe_version>'")
        if self.supersedes_version is not None and self.supersedes_version >= self.recipe_version:
            raise ValueError("supersedes_version must be lower than recipe_version")
        indices = [bend.sequence_index for bend in self.bends]
        if indices != list(range(1, len(self.bends) + 1)):
            raise ValueError("bends must be sorted by a contiguous sequence_index 1..N")
        uids = [bend.bend_uid for bend in self.bends]
        if len(set(uids)) != len(uids):
            raise ValueError("bend_uid values must be unique")
        labels = [bend.label for bend in self.bends]
        if len(set(labels)) != len(labels):
            raise ValueError("bend labels must be unique")
        _check_machine_moves(uids, self.machine_moves, {b.bend_uid: b.machine_move_index for b in self.bends})
        uid_set = set(uids)
        gm_ids = [gm.measurement_id for gm in self.general_measurements]
        if len(set(gm_ids)) != len(gm_ids):
            raise ValueError("general measurement ids must be unique")
        for gm in self.general_measurements:
            if not set(gm.bend_uids) <= uid_set:
                raise ValueError("general measurement references an unknown bend_uid")
        if self.flat is not None and not set(self.flat.binding.bend_line_map.values()) <= uid_set:
            raise ValueError("flat binding references an unknown bend_uid")
        removed = [rc.feature_id for rc in self.removed_candidates]
        if len(set(removed)) != len(removed) or set(removed) & {b.source_feature_id for b in self.bends}:
            raise ValueError("removed candidates must be unique and disjoint from active bends")
        if not (info.context or {}).get("_skip_hash_check"):
            content, recipe = self.compute_hashes()
            if self.content_sha256 != content:
                raise ValueError("content_sha256 does not match the recipe content")
            if self.recipe_sha256 != recipe:
                raise ValueError("recipe_sha256 does not match the recipe")
        return self

    # -- hashing -----------------------------------------------------------

    def compute_hashes(self) -> tuple[str, str]:
        """Return ``(content_sha256, recipe_sha256)`` recomputed from the fields."""
        data = self.model_dump(mode="json")
        content = sha256_hex(strip_keys(data, RECIPE_CONTENT_EXCLUDED_KEYS))
        recipe = sha256_hex(strip_keys({**data, "content_sha256": content}, ("recipe_sha256",)))
        return content, recipe

    @classmethod
    def finalize(cls, **fields: Any) -> "PartRecipeV1":
        """Validate ``fields`` (without hashes) and return the recipe with both hashes set."""
        placeholder = "0" * 64
        draft = cls.model_validate(
            {**fields, "content_sha256": placeholder, "recipe_sha256": placeholder},
            context={"_skip_hash_check": True},
        )
        content, recipe = draft.compute_hashes()
        data = draft.model_dump(mode="json")
        data.update(content_sha256=content, recipe_sha256=recipe)
        return cls.model_validate(data)

    # -- convenience -------------------------------------------------------

    def bend(self, bend_uid: str) -> RecipeBendV1:
        for bend in self.bends:
            if bend.bend_uid == bend_uid:
                return bend
        raise KeyError(bend_uid)

    def resolved_tolerance(self, bend_uid: str, kind: str) -> Tolerance | None:
        """Per-bend override or recipe default for ``kind`` in
        {angle_deg, flange_a_mm, flange_b_mm, bend_line_position_mm, backgauge_mm}."""
        bend = self.bend(bend_uid)
        override = getattr(bend.tolerances, kind)
        if override is not None:
            return override
        defaults = self.default_tolerances
        if kind == "angle_deg":
            return defaults.angle_deg
        if kind in ("flange_a_mm", "flange_b_mm"):
            return defaults.flange_mm
        if kind == "bend_line_position_mm":
            return defaults.bend_line_position_mm
        if kind == "backgauge_mm":
            return defaults.backgauge_mm or defaults.flange_mm
        raise KeyError(kind)

    def general_measurement(self, measurement_id: str) -> RecipeGeneralMeasurementV1:
        for gm in self.general_measurements:
            if gm.measurement_id == measurement_id:
                return gm
        raise KeyError(measurement_id)

    def resolved_measurement_tolerance(self, measurement_id: str) -> Tolerance:
        """Measurement override, else ``general_mm``, else ``flange_mm``."""
        gm = self.general_measurement(measurement_id)
        defaults = self.default_tolerances
        return gm.tolerance or defaults.general_mm or defaults.flange_mm


def _check_machine_moves(
    sequence: list[str],
    moves: tuple[tuple[str, ...], ...],
    move_index_by_uid: dict[str, int | None],
) -> None:
    """Moves are non-empty, ordered, contiguous, disjoint slices of the sequence
    (same rule as apps/api/services/vendor_setup_service.py:190-200), and each
    bend's ``machine_move_index`` names its group (1-based) or is ``None``."""
    expected_index: dict[str, int | None] = {uid: None for uid in sequence}
    cursor = 0
    for group_number, move in enumerate(moves, start=1):
        if not move:
            raise ValueError("machine moves must be non-empty")
        try:
            start = sequence.index(move[0], cursor)
        except ValueError:
            raise ValueError("machine moves must be ordered contiguous bend groups") from None
        if tuple(sequence[start:start + len(move)]) != tuple(move):
            raise ValueError("machine moves must be ordered contiguous bend groups")
        for uid in move:
            expected_index[uid] = group_number
        cursor = start + len(move)
    if expected_index != move_index_by_uid:
        raise ValueError("bend machine_move_index must match its machine_moves group")


class BendOverrideV1(ContractModel):
    angle_deg: Tolerance | None = None
    flange_a_mm: Tolerance | None = None
    flange_b_mm: Tolerance | None = None
    bend_line_position_mm: Tolerance | None = None
    backgauge_mm: Tolerance | None = None
    flange_inspection: FlangeInspection | None = None
    #: Required for manual marks (no CAD nominal exists); included angle,
    #: strictly between 0 and 180 (180 would be flat, i.e. no bend).
    manual_included_angle_deg: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)] | None = None


class MeasurementOverrideV1(ContractModel):
    tolerance: Tolerance | None = None
    required: bool | None = None


class ReleaseRequestV1(ContractModel):
    """Release parameters (also persisted as the per-part release draft)."""

    expected_draft_revision: int = Field(ge=0)
    expected_release_draft_revision: int | None = Field(default=None, ge=0)
    #: ``None`` is a persisted-but-incomplete draft; the builder reports TOLERANCES_MISSING.
    default_tolerances: DefaultTolerances | None = None
    #: Keyed by Vendor Setup draft ``feature_id``.
    bend_overrides: dict[FeatureId, BendOverrideV1] = Field(default_factory=dict)
    #: Keyed by general measurement id.
    measurement_overrides: dict[FeatureId, MeasurementOverrideV1] = Field(default_factory=dict)
    thickness_mm: PosFloat | None = None
    #: FINAL by default: every planned bend is expected formed, so a missing
    #: bend is FINAL_INCOMPLETE (review). IN_PROCESS is an explicit operator
    #: opt-in for staged inspection; even then a part with an unformed bend is
    #: never ALL_MEASURED_IN_TOLERANCE (it is STAGE_IN_TOLERANCE at best).
    inspection_stage: InspectionStage = "FINAL"
    confirmation: RecipeConfirmation = "OPERATOR_CONFIRMED"
    release_note: str = Field(default="", max_length=2000)
