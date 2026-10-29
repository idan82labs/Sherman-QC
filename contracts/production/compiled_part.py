"""CompiledPartV1: the onboarding-compile artifact (E1) consumed by the per-scan engine (E3).

Produced once per recipe release by ``python -m domains.production.engine.compile``
(see contracts/production/ENGINE_IO.md). All CAD work is precomputed here; the
per-scan engine never re-reads the STEP file.

Contents:

- **Flange graph**: planar panels (outer/inner skin planes, thickness), bends
  (inner radius ``Ri``, thickness ``t``, sweep, axis, parent and child panel),
  forming a tree rooted at the datum panel (the panel that never moves in the
  sequence; the largest when several qualify).
- **Loci** per bend: the cylinder axis *and* the surface loci (inner/outer
  tangent lines on both panels, IML/OML virtual sharps, inner/outer apex).
  The cylinder axis is never published as the bend line; the bend line is the
  OML virtual sharp.
- **Nominals per characteristic** with explicit conventions and resolved
  tolerances, keyed by :func:`contracts.production.characteristics.make_characteristic_id`.
- **Stage models** ``S_k``: for every sequence prefix ``k = 0..N`` (and
  optionally extra out-of-order stages), panel transforms from the final pose,
  stage bend lines, a face-labelled 0.5 mm sample cloud (``.npz``) and a
  display mesh (``.ply``), plus the stage's **symmetry test** (the stage model
  registered onto itself; symmetric iff a non-identity candidate reaches
  ``<= 0.3 mm`` RMS).
- **Reconciliation** of the flange graph with the recipe's confirmed marks.
  Blockers (e.g. MARK_NOT_IN_FLANGE_GRAPH, UNMARKED_BEND) make the compile
  ``BLOCKED``; the release gate refuses it and the wizard shows the codes.

Frame: ``cad_model_mm`` (STEP model coordinates, mm). Hash:
``compiled_part_sha256`` covers every field except itself, ``compiled_at`` and
``diagnostics``, so identical inputs give an identical hash.
"""
from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationInfo, model_validator

from contracts.production.canonical import sha256_hex, strip_keys
from contracts.production.characteristics import (
    BEND_CHARACTERISTIC_ORDER,
    BoundaryKind,
    CharacteristicKind,
    FlangeSide,
    make_characteristic_id,
    unit_for_kind,
)
from contracts.production.common import (
    FRAME_CAD_MODEL,
    ContractModel,
    FiniteFloat,
    Mat4,
    NonNegFloat,
    PosFloat,
    RelPath,
    Sha256Hex,
    Tolerance,
    ToleranceRef,
    UtcIsoTimestamp,
    Vec3,
)
from contracts.production.part_recipe import BEND_UID_PATTERN, RECIPE_ID_PATTERN, PartRecipeV1

__all__ = [
    "COMPILED_PART_SCHEMA",
    "COMPILED_PART_HASH_EXCLUDED_KEYS",
    "STAGE_MODEL_ID_PATTERN",
    "MAX_PANELS",
    "MAX_BENDS",
    "SYMMETRY_RMS_THRESHOLD_MM",
    "SAMPLE_CLOUD_SPACING_MM",
    "SAMPLE_CLOUD_ARRAYS",
    "CompileIssueCode",
    "COMPILE_ISSUE_CODES",
    "COMPILE_BLOCKER_CODES",
    "COMPILE_WARNING_CODES",
    "UnsupportedFeatureKind",
    "UNSUPPORTED_FEATURE_KINDS",
    "LineLocusV1",
    "AxisV1",
    "PlaneV1",
    "FreeEdgeV1",
    "PanelV1",
    "BendLociV1",
    "FlangeSideV1",
    "CompiledBendV1",
    "GraphBendV1",
    "CharacteristicNominalV1",
    "SampleCloudRefV1",
    "MeshRefV1",
    "SymmetryCandidateV1",
    "SymmetryTestV1",
    "StageModelV1",
    "ThicknessV1",
    "KFactorV1",
    "DevelopedLengthCheckV1",
    "FlatCompileV1",
    "UnsupportedFeatureV1",
    "CompileIssueV1",
    "CompilerInfoV1",
    "CompiledRecipeRefV1",
    "CompiledPartV1",
    "CompiledPartRefV1",
    "assign_sides",
    "prefix_stage_model_id",
    "check_compiled_against_recipe",
    "recipe_policy_for",
]

COMPILED_PART_SCHEMA = "sherman_compiled_part_v1"
COMPILED_PART_HASH_EXCLUDED_KEYS: tuple[str, ...] = ("compiled_part_sha256", "compiled_at", "diagnostics")
SYMMETRY_RMS_THRESHOLD_MM = 0.3
SAMPLE_CLOUD_SPACING_MM = 0.5
#: Arrays inside every stage ``.npz`` (np.load(..., allow_pickle=False)):
#: points float32 (M,3) mm in cad_model_mm at the stage pose; normals float32
#: (M,3) unit, pointing out of the material; segment_kind uint8 (M,)
#: 0=panel 1=bend 2=edge wall; segment_index int32 (M,) index into
#: ``panels`` (kind 0/2) or ``bends`` (kind 1); skin uint8 (M,) 0=skin 0,
#: 1=skin 1, 2=edge wall (see :class:`PanelV1` for the skin convention).
SAMPLE_CLOUD_ARRAYS: tuple[str, ...] = ("points", "normals", "segment_kind", "segment_index", "skin")

#: ``S<k>`` (sequence prefix with k bends formed) or ``S<k>-<tag>`` (out of order).
STAGE_MODEL_ID_PATTERN = r"^S[0-9]{1,4}(-[a-z0-9]{1,16})?$"
#: Upper bounds that keep the ``segment_labels.u16`` encoding collision-free
#: (``1 + i`` for panels, ``1001 + j`` for bends; ENGINE_IO §4.1).
MAX_PANELS = 999
MAX_BENDS = 999

BendUid = Annotated[str, Field(pattern=BEND_UID_PATTERN)]
PanelId = Annotated[str, Field(pattern=r"^P[0-9]{1,4}$")]
GraphBendId = Annotated[str, Field(pattern=r"^G[0-9]{1,4}$")]
StageModelId = Annotated[str, Field(pattern=STAGE_MODEL_ID_PATTERN)]

CompileIssueCode = Literal[
    # blockers
    "MARK_NOT_IN_FLANGE_GRAPH",  # a recipe bend matches no flange-graph bend
    "UNMARKED_BEND",  # a flange-graph bend matches no recipe bend
    "MARK_AMBIGUOUS",  # a recipe bend matches several graph bends (or vice versa)
    "DATUM_PANEL_UNDETERMINED",
    "FLANGE_GRAPH_NOT_A_TREE",
    "CAD_READ_FAILED",
    "CAD_HASH_MISMATCH",
    "RECIPE_INVALID",
    "THICKNESS_INCONSISTENT",
    "FLAT_PARSE_FAILED",
    "FLAT_BINDING_INCONSISTENT",
    "FLAT_DEVELOPED_LENGTH_MISMATCH",
    "STAGE_MODEL_FAILED",
    # warnings
    "UNSUPPORTED_FEATURE",
    "NOMINAL_UNAVAILABLE",
    "K_FACTOR_DEFAULTED",
    "ANGLE_CONVENTION_DISAGREES_WITH_RECIPE",
    "FLAT_NOT_BOUND",
]
COMPILE_ISSUE_CODES: frozenset[str] = frozenset(CompileIssueCode.__args__)  # type: ignore[attr-defined]
COMPILE_WARNING_CODES: frozenset[str] = frozenset(
    {"UNSUPPORTED_FEATURE", "NOMINAL_UNAVAILABLE", "K_FACTOR_DEFAULTED", "ANGLE_CONVENTION_DISAGREES_WITH_RECIPE", "FLAT_NOT_BOUND"}
)
COMPILE_BLOCKER_CODES: frozenset[str] = COMPILE_ISSUE_CODES - COMPILE_WARNING_CODES

UnsupportedFeatureKind = Literal["ROLLED", "HEM", "SPLINE_WALL", "STEPPED_BOSS", "OTHER"]
UNSUPPORTED_FEATURE_KINDS: frozenset[str] = frozenset(UnsupportedFeatureKind.__args__)  # type: ignore[attr-defined]


def _unit(v: tuple[float, float, float], name: str) -> None:
    norm = math.sqrt(sum(c * c for c in v))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"{name} must be a unit vector (|v| = {norm:.6f})")


def assign_sides(outward_1: Vec3, outward_2: Vec3) -> tuple[int, int]:
    """Side naming rule: side A is the flange whose in-plane outward unit vector
    has the larger dot product with +X; ties (|delta| < 1e-6) are broken by +Y,
    then +Z. Returns ``(index_of_A, index_of_B)`` into ``(outward_1, outward_2)``.
    Raises ``ValueError`` when the two vectors tie on all three axes."""
    for axis in range(3):
        delta = outward_1[axis] - outward_2[axis]
        if abs(delta) >= 1e-6:
            return (0, 1) if delta > 0 else (1, 0)
    raise ValueError("flange outward directions are identical; sides are undefined")


def prefix_stage_model_id(k: int) -> str:
    """Id of the sequence-prefix stage model with the first ``k`` bends formed."""
    return f"S{int(k)}"


class LineLocusV1(ContractModel):
    start_mm: Vec3
    end_mm: Vec3

    @model_validator(mode="after")
    def _nondegenerate(self) -> "LineLocusV1":
        if math.dist(self.start_mm, self.end_mm) <= 1e-9:
            raise ValueError("a line locus needs distinct endpoints")
        return self


class AxisV1(ContractModel):
    point_mm: Vec3
    #: Unit direction. Rotating the child subtree by +sweep about this axis
    #: (right-hand rule) takes it from flat to formed.
    direction: Vec3

    @model_validator(mode="after")
    def _unit_direction(self) -> "AxisV1":
        _unit(self.direction, "axis.direction")
        return self


class PlaneV1(ContractModel):
    point_mm: Vec3
    #: Unit normal pointing out of the material on this skin.
    normal: Vec3

    @model_validator(mode="after")
    def _unit_normal(self) -> "PlaneV1":
        _unit(self.normal, "plane.normal")
        return self


class FreeEdgeV1(ContractModel):
    edge_id: str = Field(min_length=1, max_length=32)
    polyline_mm: tuple[Vec3, ...] = Field(min_length=2)
    #: True when the edge is a cut boundary (never a bend line); usable for
    #: stage-independent coarse registration.
    registration_usable: bool = True


class PanelV1(ContractModel):
    """A planar flange panel at the final pose.

    Skins: *skin 0* of the datum panel is the face whose outward normal has the
    larger dot product with +Z (ties: +Y, then +X); skin 0 of every other panel
    is the face continuous with the datum's skin 0 across the bends. Which skin
    is the outside of a given bend (its OML side) is ``CompiledBendV1.oml_skin``.
    """

    panel_id: PanelId
    skin0_plane: PlaneV1
    skin1_plane: PlaneV1 | None = None
    #: Measured skin separation where both skins exist.
    thickness_mm: PosFloat | None = None
    area_mm2: PosFloat
    centroid_mm: Vec3
    #: Closed skin-0 boundary polygon (first point not repeated).
    outer_boundary_mm: tuple[Vec3, ...] = Field(min_length=3)
    free_edges: tuple[FreeEdgeV1, ...] = ()
    holes_mm: tuple[tuple[Vec3, ...], ...] = ()
    is_datum: bool = False
    supported: bool = True
    unsupported_feature: UnsupportedFeatureKind | None = None

    @model_validator(mode="after")
    def _validate(self) -> "PanelV1":
        if self.supported == (self.unsupported_feature is not None):
            raise ValueError("unsupported_feature is set exactly when the panel is unsupported")
        return self


class BendLociV1(ContractModel):
    """All loci of one bend at the *final* (fully formed) pose."""

    #: Centre of curvature. Diagnostic and stage-model hinge only; never the bend line.
    cylinder_axis: LineLocusV1
    inner_tangent_parent: LineLocusV1
    inner_tangent_child: LineLocusV1
    outer_tangent_parent: LineLocusV1
    outer_tangent_child: LineLocusV1
    iml_virtual_sharp: LineLocusV1
    #: The published bend line (bend-line position characteristic).
    oml_virtual_sharp: LineLocusV1
    inner_apex: LineLocusV1
    #: Visible outer-surface apex (mid-sweep). Declared G2 line-localization locus.
    outer_apex: LineLocusV1


class FlangeSideV1(ContractModel):
    side: FlangeSide
    panel_id: PanelId
    #: In-plane unit vector perpendicular to the bend axis, pointing away from the bend.
    outward_dir: Vec3
    boundary_kind: BoundaryKind
    adjacent_bend_uid: BendUid | None = None

    @model_validator(mode="after")
    def _validate(self) -> "FlangeSideV1":
        _unit(self.outward_dir, "outward_dir")
        if (self.boundary_kind == "ADJACENT_BEND") != (self.adjacent_bend_uid is not None):
            raise ValueError("adjacent_bend_uid is set exactly for ADJACENT_BEND boundaries")
        return self


class CompiledBendV1(ContractModel):
    bend_uid: BendUid
    sequence_index: int = Field(ge=1)
    label: str = Field(min_length=1, max_length=32)
    machine_move_index: int | None = Field(default=None, ge=1)
    graph_bend_id: GraphBendId
    parent_panel_id: PanelId
    child_panel_id: PanelId
    inner_radius_mm: PosFloat
    thickness_mm: PosFloat
    sweep_deg: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)]
    nominal_included_deg: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)]
    #: Relative to the datum panel's skin-0 normal at the final pose; None when undefined.
    direction: Literal["UP", "DOWN"] | None = None
    #: Skin on the convex (outside) side of this bend; its virtual sharp is the OML.
    oml_skin: Literal[0, 1]
    axis: AxisV1
    loci: BendLociV1
    bend_length_mm: PosFloat
    #: Developed (neutral) length of the bend region: sweep_rad * (Ri + K * t).
    neutral_length_mm: PosFloat
    sides: tuple[FlangeSideV1, FlangeSideV1]
    supported: bool = True
    unsupported_feature: UnsupportedFeatureKind | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CompiledBendV1":
        if not math.isclose(self.nominal_included_deg, 180.0 - self.sweep_deg, abs_tol=1e-6):
            raise ValueError("nominal_included_deg must equal 180 - sweep_deg")
        if self.parent_panel_id == self.child_panel_id:
            raise ValueError("a bend joins two different panels")
        if [s.side for s in self.sides] != ["A", "B"]:
            raise ValueError("sides must be (A, B)")
        if {s.panel_id for s in self.sides} != {self.parent_panel_id, self.child_panel_id}:
            raise ValueError("sides must be the parent and child panels")
        a, b = self.sides
        if assign_sides(a.outward_dir, b.outward_dir) != (0, 1):
            raise ValueError("side A must follow the +X/+Y/+Z naming rule")
        if self.supported == (self.unsupported_feature is not None):
            raise ValueError("unsupported_feature is set exactly when the bend is unsupported")
        return self


class GraphBendV1(ContractModel):
    """A flange-graph bend with no recipe mark (shown in the wizard as UNMARKED_BEND)."""

    graph_bend_id: GraphBendId
    oml_virtual_sharp: LineLocusV1
    sweep_deg: Annotated[float, Field(gt=0, lt=180, allow_inf_nan=False)]
    inner_radius_mm: PosFloat


class CharacteristicNominalV1(ContractModel):
    characteristic_id: str = Field(min_length=1, max_length=200)
    kind: CharacteristicKind
    bend_uid: BendUid | None = None
    side: FlangeSide | None = None
    stage_index: int | None = Field(default=None, ge=0)
    measurement_id: str | None = Field(default=None, min_length=1, max_length=100)
    unit: Literal["mm", "deg"]
    nominal: FiniteFloat | None = None
    unavailable_reason: Literal["NOMINAL_UNAVAILABLE", "UNSUPPORTED_FEATURE"] | None = None
    #: e.g. OML_SHARP_TO_FREE_EDGE, OML_SHARP_TO_OML_SHARP, INCLUDED_DEG,
    #: OML_SHARP_OFFSET_ALONG_SIDE_A, OML_SHARP_TO_FAR_BOUNDARY_AT_STAGE, CAD_STRAIGHT_LINE.
    convention: str = Field(min_length=1, max_length=64)
    boundary_kind: BoundaryKind | None = None
    adjacent_bend_uid: BendUid | None = None
    tolerance: ToleranceRef | None = None
    recipe_requirement: Literal["REQUIRED", "NOT_INSPECTED", "OPTIONAL"]

    @model_validator(mode="after")
    def _validate(self) -> "CharacteristicNominalV1":
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
        if (self.nominal is None) == (self.unavailable_reason is None):
            raise ValueError("exactly one of nominal / unavailable_reason is set")
        if self.side is not None and self.kind not in ("FLANGE_LENGTH", "BACKGAUGE_LENGTH"):
            raise ValueError("only flange/backgauge nominals carry a side")
        if (self.boundary_kind == "ADJACENT_BEND") != (self.adjacent_bend_uid is not None):
            raise ValueError("adjacent_bend_uid is set exactly for ADJACENT_BEND boundaries")
        # Requirement vocabulary per kind (mirrors recipe_policy_for), so E3's
        # row factories can never build a row the result validator rejects.
        allowed = _REQUIREMENTS_BY_KIND[self.kind]
        if self.recipe_requirement not in allowed:
            raise ValueError(f"{self.kind} recipe_requirement must be one of {sorted(allowed)}")
        if self.kind in ("FLANGE_LENGTH", "INCLUDED_ANGLE") and self.recipe_requirement == "REQUIRED":
            if self.tolerance is None:
                raise ValueError(f"a REQUIRED {self.kind} nominal carries its resolved tolerance")
        if self.kind in ("BEND_LINE_POSITION", "BACKGAUGE_LENGTH", "GENERAL_MEASUREMENT"):
            if self.kind != "GENERAL_MEASUREMENT" and (self.recipe_requirement == "REQUIRED") != (
                self.tolerance is not None
            ):
                raise ValueError(f"{self.kind} is REQUIRED exactly when a tolerance is resolved (else OPTIONAL)")
            if self.kind == "GENERAL_MEASUREMENT" and self.tolerance is None:
                raise ValueError("a general measurement nominal carries its resolved tolerance")
        return self


#: Allowed ``recipe_requirement`` values per characteristic kind.
_REQUIREMENTS_BY_KIND: dict[str, frozenset[str]] = {
    "FLANGE_LENGTH": frozenset({"REQUIRED", "NOT_INSPECTED"}),
    "INCLUDED_ANGLE": frozenset({"REQUIRED"}),
    "BEND_LINE_POSITION": frozenset({"REQUIRED", "OPTIONAL"}),
    "BACKGAUGE_LENGTH": frozenset({"REQUIRED", "OPTIONAL"}),
    "GENERAL_MEASUREMENT": frozenset({"REQUIRED", "OPTIONAL"}),
}


class SampleCloudRefV1(ContractModel):
    #: Relative to the directory holding compiled_part.json.
    rel_path: RelPath
    sha256: Sha256Hex
    point_count: int = Field(ge=1)
    spacing_mm: PosFloat = SAMPLE_CLOUD_SPACING_MM
    arrays: tuple[str, ...] = SAMPLE_CLOUD_ARRAYS

    @model_validator(mode="after")
    def _validate(self) -> "SampleCloudRefV1":
        if not self.rel_path.endswith(".npz") or tuple(self.arrays) != SAMPLE_CLOUD_ARRAYS:
            raise ValueError("sample clouds are .npz files with the declared arrays")
        return self


class MeshRefV1(ContractModel):
    #: Binary little-endian PLY, float32 vertices, int32 triangle faces. Relative path.
    rel_path: RelPath
    sha256: Sha256Hex
    vertex_count: int = Field(ge=3)
    face_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate(self) -> "MeshRefV1":
        if not self.rel_path.endswith(".ply"):
            raise ValueError("display meshes are .ply files")
        return self


class SymmetryCandidateV1(ContractModel):
    kind: Literal["ROTATION", "MIRROR"]
    #: Non-identity self-map of the stage model (cad_model_mm -> cad_model_mm).
    transform: Mat4
    rms_mm: NonNegFloat
    #: bend_uid -> bend_uid it lands on under ``transform`` (formed bends only).
    bend_permutation: dict[BendUid, BendUid] = Field(default_factory=dict)


class SymmetryTestV1(ContractModel):
    """Stage model registered onto itself (plan §10 amendment 5).

    ``is_symmetric`` iff some non-identity candidate reaches ``rms_mm <=
    threshold_rms_mm``. ``ambiguous_bend_groups`` are the orbits (size >= 2)
    of bends under those candidates; only these may be reported AMBIGUOUS.
    MIRROR candidates matter only for mirrored-part detection; a physical
    part cannot be its own mirror image unless the CAD is mirror-symmetric.
    """

    method: str = Field(min_length=1, max_length=96)
    threshold_rms_mm: PosFloat = SYMMETRY_RMS_THRESHOLD_MM
    candidates: tuple[SymmetryCandidateV1, ...] = ()
    is_symmetric: bool
    ambiguous_bend_groups: tuple[tuple[BendUid, ...], ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> "SymmetryTestV1":
        hits = [c for c in self.candidates if c.rms_mm <= self.threshold_rms_mm + 1e-12]
        if self.is_symmetric != bool(hits):
            raise ValueError("is_symmetric must equal 'some candidate is within threshold_rms_mm'")
        if self.ambiguous_bend_groups and not self.is_symmetric:
            raise ValueError("only a symmetric stage has ambiguous bend groups")
        seen: set[str] = set()
        for group in self.ambiguous_bend_groups:
            if len(group) < 2 or seen & set(group) or len(set(group)) != len(group):
                raise ValueError("ambiguous bend groups are disjoint orbits of size >= 2")
            seen |= set(group)
        return self


class StageModelV1(ContractModel):
    stage_model_id: StageModelId
    #: Number of formed bends.
    stage_index: int = Field(ge=0)
    kind: Literal["SEQUENCE_PREFIX", "OUT_OF_ORDER"]
    formed_bend_uids: tuple[BendUid, ...]
    method: Literal["METHOD_A_ROTATE_BACK", "METHOD_B_FOLD_FLAT"]
    k_factor_used: Annotated[float, Field(gt=0, lt=1)]
    #: Final-pose -> stage-pose rigid transform per panel (datum panel = identity).
    #: For each not-yet-formed bend on the path from the datum: rotate back by
    #: the sweep about the bend axis, then translate by the neutral strip length
    #: away from the parent panel (Method A).
    panel_transforms: dict[PanelId, Mat4]
    #: Bend line per bend at this stage's pose: the OML virtual sharp for formed
    #: bends; for unformed bends, the centre line of the flat strip on the bend's
    #: OML skin.
    bend_lines: dict[BendUid, LineLocusV1]
    sample_cloud: SampleCloudRefV1
    display_mesh: MeshRefV1
    symmetry: SymmetryTestV1

    @model_validator(mode="after")
    def _validate(self) -> "StageModelV1":
        if len(self.formed_bend_uids) != self.stage_index or len(set(self.formed_bend_uids)) != self.stage_index:
            raise ValueError("stage_index equals the number of distinct formed bends")
        if self.kind == "SEQUENCE_PREFIX" and self.stage_model_id != prefix_stage_model_id(self.stage_index):
            raise ValueError("prefix stage models are named S<k>")
        if self.kind == "OUT_OF_ORDER" and "-" not in self.stage_model_id:
            raise ValueError("out-of-order stage models are named S<k>-<tag>")
        return self


class ThicknessV1(ContractModel):
    t_mm: PosFloat
    source: Literal["MEASURED_BOTH_SKINS", "RECIPE"]
    #: max - min of per-panel measured thickness (0 when a single value).
    spread_mm: NonNegFloat = 0.0


class KFactorV1(ContractModel):
    value: Annotated[float, Field(gt=0, lt=1)]
    source: Literal["BOUND_FLAT", "SHOP_DEFAULT"]
    #: Standard uncertainty of K; carried into stage-model nominals' U.
    uncertainty: NonNegFloat


class DevelopedLengthCheckV1(ContractModel):
    """Flat developed lengths vs folded-CAD lengths under K (QA check)."""

    status: Literal["OK", "MISMATCH", "NOT_EVALUATED"]
    threshold_mm: PosFloat = 0.5
    max_abs_delta_mm: NonNegFloat | None = None
    detail: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _validate(self) -> "DevelopedLengthCheckV1":
        if self.status == "NOT_EVALUATED":
            return self
        if self.max_abs_delta_mm is None:
            raise ValueError("an evaluated developed-length check reports max_abs_delta_mm")
        if (self.status == "MISMATCH") != (self.max_abs_delta_mm > self.threshold_mm):
            raise ValueError("MISMATCH iff max_abs_delta_mm > threshold_mm")
        return self


class FlatCompileV1(ContractModel):
    provided: bool
    format: Literal["DXF", "STEP"] | None = None
    sha256: Sha256Hex | None = None
    parse_status: Literal["NOT_PROVIDED", "PARSED", "PARSE_FAILED"]
    bend_lines_found: int | None = Field(default=None, ge=0)
    #: Binding state as released in the recipe (UNBOUND / AUTO_PROPOSED / OPERATOR_CONFIRMED).
    binding_state: Literal["UNBOUND", "AUTO_PROPOSED", "OPERATOR_CONFIRMED"] | None = None
    developed_length_check: DevelopedLengthCheckV1 = Field(
        default_factory=lambda: DevelopedLengthCheckV1(status="NOT_EVALUATED")
    )

    @model_validator(mode="after")
    def _validate(self) -> "FlatCompileV1":
        if self.provided != (self.parse_status != "NOT_PROVIDED"):
            raise ValueError("parse_status is NOT_PROVIDED exactly when no flat was provided")
        if self.provided and (self.format is None or self.sha256 is None or self.binding_state is None):
            raise ValueError("a provided flat names its format, sha256 and binding state")
        return self


class UnsupportedFeatureV1(ContractModel):
    feature_kind: UnsupportedFeatureKind
    panel_id: PanelId | None = None
    bend_uid: BendUid | None = None
    affected_characteristic_ids: tuple[str, ...] = ()
    note: str = Field(default="", max_length=500)


class CompileIssueV1(ContractModel):
    code: CompileIssueCode
    severity: Literal["BLOCKER", "WARNING"]
    bend_uid: BendUid | None = None
    graph_bend_id: GraphBendId | None = None
    detail: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _validate(self) -> "CompileIssueV1":
        expected = "WARNING" if self.code in COMPILE_WARNING_CODES else "BLOCKER"
        if self.severity != expected:
            raise ValueError(f"{self.code} has severity {expected}")
        if self.code == "MARK_NOT_IN_FLANGE_GRAPH" and self.bend_uid is None:
            raise ValueError("MARK_NOT_IN_FLANGE_GRAPH names the recipe bend_uid")
        if self.code == "UNMARKED_BEND" and self.graph_bend_id is None:
            raise ValueError("UNMARKED_BEND names the graph_bend_id")
        return self


class CompilerInfoV1(ContractModel):
    name: Literal["sherman_production_compile"] = "sherman_production_compile"
    version: str = Field(min_length=1, max_length=64)
    config_sha256: Sha256Hex | None = None


class CompiledRecipeRefV1(ContractModel):
    recipe_id: Annotated[str, Field(pattern=RECIPE_ID_PATTERN)]
    version: int = Field(ge=1)
    recipe_sha256: Sha256Hex
    content_sha256: Sha256Hex


class CompiledPartV1(ContractModel):
    schema_: Literal["sherman_compiled_part_v1"] = Field(default=COMPILED_PART_SCHEMA, alias="schema")
    status: Literal["COMPILED", "BLOCKED"]
    recipe: CompiledRecipeRefV1
    cad_sha256: Sha256Hex
    flat_sha256: Sha256Hex | None = None
    compiler: CompilerInfoV1
    compiled_at: UtcIsoTimestamp
    frame: Literal["cad_model_mm"] = FRAME_CAD_MODEL  # type: ignore[assignment]
    units: Literal["mm_deg"] = "mm_deg"
    angle_convention: Literal["INCLUDED_DEG"] = "INCLUDED_DEG"
    flange_length_convention: Literal["OUTSIDE_MOLD_LINE"] = "OUTSIDE_MOLD_LINE"
    bend_line_convention: Literal["OML_VIRTUAL_SHARP"] = "OML_VIRTUAL_SHARP"
    thickness: ThicknessV1 | None = None
    k_factor: KFactorV1 | None = None
    datum_panel_id: PanelId | None = None
    panels: tuple[PanelV1, ...] = Field(default=(), max_length=MAX_PANELS)
    #: Recipe bends (recipe bend_uid, sorted by sequence_index) bound to the graph.
    bends: tuple[CompiledBendV1, ...] = Field(default=(), max_length=MAX_BENDS)
    unmarked_graph_bends: tuple[GraphBendV1, ...] = ()
    characteristics: tuple[CharacteristicNominalV1, ...] = ()
    stages: tuple[StageModelV1, ...] = ()
    flat: FlatCompileV1 = Field(default_factory=lambda: FlatCompileV1(provided=False, parse_status="NOT_PROVIDED"))
    unsupported_features: tuple[UnsupportedFeatureV1, ...] = ()
    issues: tuple[CompileIssueV1, ...] = ()
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    compiled_part_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate(self, info: ValidationInfo) -> "CompiledPartV1":
        blockers = [i for i in self.issues if i.severity == "BLOCKER"]
        if (self.status == "BLOCKED") != bool(blockers):
            raise ValueError("status is BLOCKED exactly when a BLOCKER issue exists")
        if self.flat.provided != (self.flat_sha256 is not None) or (
            self.flat_sha256 is not None and self.flat.sha256 != self.flat_sha256
        ):
            raise ValueError("flat_sha256 must match the flat compile record")

        panel_ids = [p.panel_id for p in self.panels]
        if len(set(panel_ids)) != len(panel_ids):
            raise ValueError("panel ids must be unique")
        uids = [b.bend_uid for b in self.bends]
        if len(set(uids)) != len(uids):
            raise ValueError("bend_uid values must be unique")
        if [b.sequence_index for b in self.bends] != list(range(1, len(self.bends) + 1)):
            raise ValueError("bends must be sorted by a contiguous sequence_index 1..N")
        graph_ids = [b.graph_bend_id for b in self.bends] + [g.graph_bend_id for g in self.unmarked_graph_bends]
        if len(set(graph_ids)) != len(graph_ids):
            raise ValueError("graph_bend_id values must be unique")
        char_ids = [c.characteristic_id for c in self.characteristics]
        if len(set(char_ids)) != len(char_ids):
            raise ValueError("characteristic ids must be unique")
        uid_set = set(uids)
        for c in self.characteristics:
            if c.bend_uid is not None and c.bend_uid not in uid_set:
                raise ValueError("characteristic references an unknown bend_uid")

        if self.status == "COMPILED":
            self._validate_compiled(panel_ids, uids, set(char_ids))

        if not (info.context or {}).get("_skip_hash_check"):
            if self.compiled_part_sha256 != self.compute_hash():
                raise ValueError("compiled_part_sha256 does not match the content")
        return self

    def _validate_compiled(self, panel_ids: list[str], uids: list[str], char_ids: set[str]) -> None:
        if self.thickness is None or self.k_factor is None or self.datum_panel_id is None:
            raise ValueError("a COMPILED part has thickness, k_factor and datum_panel_id")
        if not self.bends:
            raise ValueError("a COMPILED part has at least one bend")
        datums = [p.panel_id for p in self.panels if p.is_datum]
        if datums != [self.datum_panel_id]:
            raise ValueError("exactly one panel is the datum and it is datum_panel_id")
        panel_set = set(panel_ids)
        children: dict[str, str] = {}
        for bend in self.bends:
            if bend.parent_panel_id not in panel_set or bend.child_panel_id not in panel_set:
                raise ValueError("bends must join known panels")
            if bend.child_panel_id in children or bend.child_panel_id == self.datum_panel_id:
                raise ValueError("flange graph must be a tree rooted at the datum (one parent per panel)")
            children[bend.child_panel_id] = bend.parent_panel_id
            for side in bend.sides:
                if side.adjacent_bend_uid is not None and side.adjacent_bend_uid not in uids:
                    raise ValueError("adjacent_bend_uid must name a compiled bend")
        for panel in children:
            seen = {panel}
            cursor = panel
            while cursor != self.datum_panel_id:
                cursor = children.get(cursor, "")
                if not cursor or cursor in seen:
                    raise ValueError("flange graph must be a tree rooted at the datum")
                seen.add(cursor)
        for bend in self.bends:
            for kind, side in BEND_CHARACTERISTIC_ORDER:
                cid = make_characteristic_id(kind, bend_uid=bend.bend_uid, side=side)
                if cid not in char_ids:
                    raise ValueError(f"missing nominal {cid}")
        n = len(self.bends)
        prefix = {s.stage_index: s for s in self.stages if s.kind == "SEQUENCE_PREFIX"}
        if sorted(prefix) != list(range(n + 1)) or len(prefix) != sum(
            1 for s in self.stages if s.kind == "SEQUENCE_PREFIX"
        ):
            raise ValueError("a COMPILED part has exactly one prefix stage model per k = 0..N")
        for k, stage in prefix.items():
            if tuple(stage.formed_bend_uids) != tuple(uids[:k]):
                raise ValueError(f"prefix stage S{k} must form the first {k} bends")
        stage_ids = [s.stage_model_id for s in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("stage_model_id values must be unique")
        for stage in self.stages:
            if not set(stage.formed_bend_uids) <= set(uids):
                raise ValueError("stage models form only compiled bends")
            if set(stage.panel_transforms) != panel_set or set(stage.bend_lines) != set(uids):
                raise ValueError("stage models give a transform per panel and a line per bend")

    # -- hashing / helpers ---------------------------------------------------

    def compute_hash(self) -> str:
        return sha256_hex(strip_keys(self.model_dump(mode="json"), COMPILED_PART_HASH_EXCLUDED_KEYS))

    @classmethod
    def finalize(cls, **fields: Any) -> "CompiledPartV1":
        """Validate ``fields`` (without the hash) and return the part with its hash set."""
        draft = cls.model_validate(
            {**fields, "compiled_part_sha256": "0" * 64}, context={"_skip_hash_check": True}
        )
        data = draft.model_dump(mode="json")
        data["compiled_part_sha256"] = draft.compute_hash()
        return cls.model_validate(data)

    @property
    def blocker_codes(self) -> tuple[str, ...]:
        return tuple(sorted({i.code for i in self.issues if i.severity == "BLOCKER"}))

    def stage(self, stage_model_id: str) -> StageModelV1:
        for stage in self.stages:
            if stage.stage_model_id == stage_model_id:
                return stage
        raise KeyError(stage_model_id)

    def nominal(self, characteristic_id: str) -> CharacteristicNominalV1:
        for c in self.characteristics:
            if c.characteristic_id == characteristic_id:
                return c
        raise KeyError(characteristic_id)


class CompiledPartRefV1(ContractModel):
    """Where a release stores its compile (data root relative)."""

    compiled_part_sha256: Sha256Hex
    recipe_sha256: Sha256Hex
    status: Literal["COMPILED", "BLOCKED"]
    compiler_version: str = Field(min_length=1, max_length=64)
    stored_rel_path: RelPath

    @model_validator(mode="after")
    def _validate(self) -> "CompiledPartRefV1":
        if self.stored_rel_path != f"recipes/compiled/{self.recipe_sha256}/compiled_part.json":
            raise ValueError("stored_rel_path must be recipes/compiled/<recipe_sha256>/compiled_part.json")
        return self


def recipe_policy_for(recipe: PartRecipeV1, nominal: CharacteristicNominalV1) -> tuple[Tolerance | None, str]:
    """The tolerance band and ``recipe_requirement`` the recipe implies for ``nominal``.

    - FLANGE_LENGTH: ``flange_a_mm`` / ``flange_b_mm`` (override, else default);
      REQUIRED or NOT_INSPECTED from ``flange_inspection``.
    - INCLUDED_ANGLE: ``angle_deg``; always REQUIRED.
    - BEND_LINE_POSITION / BACKGAUGE_LENGTH: their tolerance; REQUIRED when a
      tolerance resolves, else OPTIONAL.
    - GENERAL_MEASUREMENT: :meth:`PartRecipeV1.resolved_measurement_tolerance`;
      REQUIRED or OPTIONAL from ``required``.

    Raises ``KeyError`` when the nominal names an unknown bend or measurement.
    """
    kind = nominal.kind
    if kind == "GENERAL_MEASUREMENT":
        gm = recipe.general_measurement(nominal.measurement_id or "")
        return recipe.resolved_measurement_tolerance(gm.measurement_id), ("REQUIRED" if gm.required else "OPTIONAL")
    bend = recipe.bend(nominal.bend_uid or "")
    if kind == "FLANGE_LENGTH":
        key, inspection = (
            ("flange_a_mm", bend.flange_inspection.side_a)
            if nominal.side == "A"
            else ("flange_b_mm", bend.flange_inspection.side_b)
        )
        return recipe.resolved_tolerance(bend.bend_uid, key), inspection
    if kind == "INCLUDED_ANGLE":
        return recipe.resolved_tolerance(bend.bend_uid, "angle_deg"), "REQUIRED"
    key = "bend_line_position_mm" if kind == "BEND_LINE_POSITION" else "backgauge_mm"
    tol = recipe.resolved_tolerance(bend.bend_uid, key)
    return tol, ("REQUIRED" if tol is not None else "OPTIONAL")


def _same_band(a: Tolerance | None, b: Tolerance | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a.plus, b.plus, abs_tol=1e-9) and math.isclose(a.minus, b.minus, abs_tol=1e-9)


def check_compiled_against_recipe(compiled: CompiledPartV1, recipe: PartRecipeV1) -> list[str]:
    """Cross-document consistency problems (empty list = consistent).

    Codes: RECIPE_REF_MISMATCH, CAD_SHA_MISMATCH, FLAT_SHA_MISMATCH and, for a
    COMPILED part, BEND_INVENTORY_MISMATCH, GENERAL_MEASUREMENT_MISMATCH,
    NOMINAL_UNKNOWN_REFERENCE (a nominal names a bend/measurement the recipe
    lacks), TOLERANCE_MISMATCH (a nominal's band differs from
    :func:`recipe_policy_for`) and REQUIREMENT_MISMATCH (its
    ``recipe_requirement`` differs). Sorted, de-duplicated.

    E3 refuses to run on a mismatch (exit 2, COMPILED_PART_HASH_MISMATCH /
    RECIPE_SHA_MISMATCH); the C05 release gate refuses to store one.
    """
    problems: list[str] = []
    ref = compiled.recipe
    if (ref.recipe_id, ref.version, ref.recipe_sha256, ref.content_sha256) != (
        recipe.recipe_id,
        recipe.recipe_version,
        recipe.recipe_sha256,
        recipe.content_sha256,
    ):
        problems.append("RECIPE_REF_MISMATCH")
    if compiled.cad_sha256 != recipe.cad.sha256:
        problems.append("CAD_SHA_MISMATCH")
    if compiled.flat_sha256 != (recipe.flat.sha256 if recipe.flat else None):
        problems.append("FLAT_SHA_MISMATCH")
    if compiled.status == "COMPILED":
        recipe_rows = [(b.bend_uid, b.sequence_index, b.label, b.machine_move_index) for b in recipe.bends]
        compiled_rows = [(b.bend_uid, b.sequence_index, b.label, b.machine_move_index) for b in compiled.bends]
        if recipe_rows != compiled_rows:
            problems.append("BEND_INVENTORY_MISMATCH")
        gm_expected = {f"gm/{gm.measurement_id}" for gm in recipe.general_measurements}
        gm_present = {c.characteristic_id for c in compiled.characteristics if c.kind == "GENERAL_MEASUREMENT"}
        if gm_expected != gm_present:
            problems.append("GENERAL_MEASUREMENT_MISMATCH")
        for nominal in compiled.characteristics:
            try:
                tol, requirement = recipe_policy_for(recipe, nominal)
            except KeyError:
                problems.append("NOMINAL_UNKNOWN_REFERENCE")
                continue
            if not _same_band(nominal.tolerance, tol):
                problems.append("TOLERANCE_MISMATCH")
            if nominal.recipe_requirement != requirement:
                problems.append("REQUIREMENT_MISMATCH")
    return sorted(set(problems))
