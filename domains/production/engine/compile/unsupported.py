"""Per-feature UNSUPPORTED rule of the onboarding compile (plan §3.1, §10 amendment 4d).

Frozen 2026-10-17, before the compile was evaluated on any real STEP file. The
thresholds below are deliberately *not* part of :class:`CompileConfig`: an
exclusion from inspection may only come from this pre-written rule, never from
tuning against a part. Changing a value here requires a new
``UNSUPPORTED_RULE_VERSION`` and a ledger note.

Version 2 (2026-10-17, independent review, before A1) changes no threshold. It
makes the implementation follow the rule text below: the thin-wall constants
live here with a fixed sampling allowance (they had been inline, scaled by the
configurable sample spacing); a panel feature's coverage is its one-sided
footprint (it had been the area of every feature face, both skins and walls);
a lone cylinder joins the skin-connected cluster it belongs to before the
cluster is classified (it had become a ROLLED record of its own even inside a
formed feature); a teardrop (reflex sweep) is a HEM by its true sweep (it had
read as OTHER through the plane-only sweep); and a flange side whose far
boundary is an unsupported bend, marked or not, is UNSUPPORTED_FEATURE.
ENGINE_COMPILE.md §9 logs the effect on the corpus.

Every face of the solid gets exactly one role: panel skin, bend skin, wall
(thickness / cut-edge face) or part of an unsupported **feature** (a connected
set of skin faces). Unsupported geometry is declared per feature, never per
part. The rule, evaluated in order (first match wins):

``HEM``
    A flange-graph bend (coaxial inner/outer cylinders tangent to two panels)
    whose sweep is at least ``HEM_MIN_SWEEP_DEG`` (included angle <= 10 deg):
    open hem (180 deg), closed hem (180 deg, tiny Ri) or teardrop (reflex
    sweep above 180 deg, read from the cylinder faces' angular span).
``ROLLED``
    (a) A flange-graph bend whose inner radius exceeds ``ROLLED_RI_OVER_T`` x t
    (a rolled / bump-formed radius rather than a press-brake bend), or
    (b) a skin-connected cluster made only of cylindrical skin faces tangent to
    a panel without a coaxial partner on the other skin or without a tangent
    panel at the other end (curl, open roll). A lone cylinder inside a larger
    cluster takes that cluster's class (for example STEPPED_BOSS).
``SPLINE_WALL``
    A skin face (smooth-connected to a panel or bend skin) whose surface is
    free-form: B-spline, Bezier, offset, extrusion or revolution surface not
    recognised as a plane. Free-form *thickness walls* (cut outline faces)
    are supported and are not features.
``STEPPED_BOSS``
    A skin-connected cluster that contains toroidal, conical or spherical
    faces, or planar faces that belong to no panel (emboss, dimple, louver,
    extruded hole, stepped boss).
``OTHER``
    Anything else the flange graph cannot explain: a bend whose cylinders do
    not meet exactly two tangent ends, inconsistent tangency between a bend and
    its panels, cylinder-to-cylinder forming edges, an inner radius below
    ``MIN_REPRESENTABLE_RI_MM``, or wall faces that are not thin: more than
    ``THIN_WALL_FAR_FRACTION`` of a wall face's samples farther than
    ``t + THIN_WALL_ALLOWANCE_MM`` from every skin (the allowance is the
    sampling slack of the 0.5 mm contract grid, fixed, not the configured spacing).

Effect of a feature (never more than the feature):

- a *bend* feature (HEM, ROLLED (a), OTHER on a bend) makes that bend
  ``supported = False``; its FLANGE_LENGTH A/B, INCLUDED_ANGLE,
  BEND_LINE_POSITION and its BACKGAUGE_LENGTH nominal carry
  ``unavailable_reason = UNSUPPORTED_FEATURE``, and so does every flange or
  backgauge side of another bend whose far boundary is that bend (an
  unsupported bend's OML sharp is no measurable reference). An unmarked
  unsupported bend is not an UNMARKED_BEND blocker (it can never be a recipe
  bend): it stays rigid (formed) in every stage model and is listed as a
  feature. Mark it in the wizard when it is formed mid-sequence: a marked
  unsupported bend is staged like any recipe bend, with every characteristic
  UNKNOWN(UNSUPPORTED_FEATURE).
- a *panel* feature (ROLLED (b), SPLINE_WALL, STEPPED_BOSS, OTHER on a panel)
  leaves the panel supported unless the features' one-sided footprint covers
  more than ``PANEL_FEATURE_DOMINANT_FRACTION`` of the panel (planar skin plus
  footprint). A flange side whose far boundary is a feature at the majority of
  section stations gets ``unavailable_reason = UNSUPPORTED_FEATURE``. An
  unsupported panel is a wizard and display flag only: its characteristics
  stay available, because E3 fits planes on the panel's remaining planar skin
  samples (feature faces are never panel samples) and feature far boundaries
  are already handled per side.
- feature faces are labelled as edge-wall points (``segment_kind`` 2,
  ``skin`` 2) in the stage sample clouds so they never enter a skin plane fit.
- each feature adds one ``UNSUPPORTED_FEATURE`` WARNING issue; the release is
  allowed (ENGINE_IO §2.1). E3 reports the affected characteristics as
  UNKNOWN(UNSUPPORTED_FEATURE).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "UNSUPPORTED_RULE_VERSION",
    "HEM_MIN_SWEEP_DEG",
    "ROLLED_RI_OVER_T",
    "MIN_REPRESENTABLE_RI_MM",
    "PANEL_FEATURE_DOMINANT_FRACTION",
    "TANGENCY_TOL_MM",
    "SWEEP_CONSISTENCY_TOL_DEG",
    "THIN_WALL_FAR_FRACTION",
    "THIN_WALL_ALLOWANCE_MM",
    "wall_is_thin",
    "panel_feature_dominates",
    "FREEFORM_SURFACES",
    "FORMED_FEATURE_SURFACES",
    "FeatureRecord",
    "classify_bend",
    "classify_skin_cluster",
]

UNSUPPORTED_RULE_VERSION = "E1-UNSUPPORTED-RULE-2 (2026-10-17; rule 1 thresholds, review fixes)"

#: Sweep at or above which a bend is a hem (included angle <= 10 deg).
HEM_MIN_SWEEP_DEG = 170.0
#: Inner radius above this multiple of the thickness is a rolled radius.
ROLLED_RI_OVER_T = 10.0
#: CompiledBendV1 needs Ri > 0; a CAD inner radius below this cannot be represented.
MIN_REPRESENTABLE_RI_MM = 0.01
#: A panel feature covering more than this fraction of the panel makes the panel unsupported.
PANEL_FEATURE_DOMINANT_FRACTION = 0.5
#: Bend/panel tangency and plane-vs-cylinder sweep consistency (CAD numerical tolerances).
TANGENCY_TOL_MM = 0.02
SWEEP_CONSISTENCY_TOL_DEG = 1.0
#: A wall face is not thin when more than this fraction of its samples lie farther
#: than ``t + THIN_WALL_ALLOWANCE_MM`` from every skin sample.
THIN_WALL_FAR_FRACTION = 0.2
#: Sampling slack of the thin-wall test: two spacings of the 0.5 mm contract grid
#: (rule 1 used two *configured* spacings; the value is the same at the default).
THIN_WALL_ALLOWANCE_MM = 1.0

#: OCC surface families treated as free-form skins (SPLINE_WALL).
FREEFORM_SURFACES = frozenset({"bspline", "bezier", "offset", "extrusion", "revolution", "other"})
#: OCC surface families of formed features (STEPPED_BOSS).
FORMED_FEATURE_SURFACES = frozenset({"torus", "cone", "sphere"})


@dataclass
class FeatureRecord:
    """One unsupported feature found by the flange-graph builder (internal)."""

    kind: str  # ROLLED | HEM | SPLINE_WALL | STEPPED_BOSS | OTHER
    reason: str
    face_indices: tuple[int, ...] = ()
    #: internal panel key the feature is attached to (panel features) or None
    panel_key: int | None = None
    #: internal graph-bend key (bend features) or None
    bend_key: int | None = None
    area_mm2: float = 0.0
    #: one-sided footprint on the owning panel's plane (mm2), for the dominance test
    footprint_mm2: float = 0.0
    notes: list[str] = field(default_factory=list)


def wall_is_thin(far_fraction: float) -> bool:
    """Thin-wall test on the fraction of a wall face's samples beyond ``t + THIN_WALL_ALLOWANCE_MM``."""
    return far_fraction <= THIN_WALL_FAR_FRACTION


def panel_feature_dominates(feature_footprint_mm2: float, panel_skin_area_mm2: float) -> bool:
    """True when the features' one-sided footprint covers more than the dominant fraction of the panel."""
    total = feature_footprint_mm2 + panel_skin_area_mm2
    return total > 0 and feature_footprint_mm2 > PANEL_FEATURE_DOMINANT_FRACTION * total


def classify_bend(
    *,
    sweep_deg: float,
    inner_radius_mm: float,
    thickness_mm: float,
    tangency_error_mm: float,
    sweep_mismatch_deg: float,
) -> tuple[str | None, str]:
    """Rule for a flange-graph bend. Returns ``(feature_kind | None, reason)``."""
    if sweep_deg >= HEM_MIN_SWEEP_DEG:
        return "HEM", f"sweep {sweep_deg:.2f} deg >= {HEM_MIN_SWEEP_DEG:g} deg"
    if inner_radius_mm < MIN_REPRESENTABLE_RI_MM:
        return "OTHER", f"inner radius {inner_radius_mm:.4f} mm below {MIN_REPRESENTABLE_RI_MM} mm"
    if thickness_mm > 0 and inner_radius_mm > ROLLED_RI_OVER_T * thickness_mm:
        return "ROLLED", f"inner radius {inner_radius_mm:.3f} mm > {ROLLED_RI_OVER_T:g} x t ({thickness_mm:.3f} mm)"
    if tangency_error_mm > TANGENCY_TOL_MM:
        return "OTHER", f"bend/panel tangency error {tangency_error_mm:.4f} mm > {TANGENCY_TOL_MM} mm"
    if sweep_mismatch_deg > SWEEP_CONSISTENCY_TOL_DEG:
        return "OTHER", f"cylinder sweep differs from panel dihedral by {sweep_mismatch_deg:.3f} deg"
    return None, ""


def classify_skin_cluster(surface_families: set[str], *, has_unpaired_cylinder: bool, has_offset_plane: bool) -> tuple[str, str]:
    """Rule for a skin-connected cluster of faces that belongs to no panel or bend."""
    if surface_families & FREEFORM_SURFACES:
        return "SPLINE_WALL", "free-form skin surface (" + ", ".join(sorted(surface_families & FREEFORM_SURFACES)) + ")"
    if surface_families & FORMED_FEATURE_SURFACES or has_offset_plane:
        found = sorted(surface_families & FORMED_FEATURE_SURFACES) + (["offset plane"] if has_offset_plane else [])
        return "STEPPED_BOSS", "formed feature surfaces (" + ", ".join(found) + ")"
    if has_unpaired_cylinder:
        return "ROLLED", "cylindrical skin without a coaxial partner skin or second tangent panel"
    return "OTHER", "skin faces not explained by the flange graph"
