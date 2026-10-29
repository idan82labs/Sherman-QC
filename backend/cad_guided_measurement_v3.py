"""V3 of the bounded CAD-guided offline scan measurement kernel: the V2 text with ONE
changed mechanism, the material-side layer separation.

V10 evidence (terminal attempt: 43 physical windows = 86 before/after flange diagnostic
pairs, 43 first-flange plane attempts, all refused). The retained diagnostics are
CONSISTENT WITH two-population crops (two parallel planar populations 1.33-1.84 mm apart in
the accepted pairs) plus bridging rows continuous in the material-side coordinate; this is
an inferred mixture HYPOTHESIS, not a proved physical sheet identity or thickness. Under it
the V2 empty-gap partition (largest gap > `layer_split_gap_mm`) left 34 first-flange fits
unresolved as a single population (32 with no qualifying gap; 2 with a singleton-tail gap
failing the 12-row minimum), 2 pairs incompatible, and, where a pair was selected (7
first-flange, 17 second-flange fits), kept every row below the split threshold in the lower
group, which then failed the 0.03 mm maximum residual. Thirteen selected second-flange populations met the declared
residual (rms 0.005-0.011 mm); their physical identity remains unverified.

V3 layer separation (declared, deterministic, geometric; fit quality never selects):
* the material-side coordinate along the crop TLS normal oriented to +radial, as V2;
* TWO-MODE detection on that coordinate: fixed-width bins (`layer_mode_bin_mm`), the
  tallest local maximum and the tallest other local maximum whose centre separation lies
  in [`layer_split_gap_mm`, `max_layer_gap_mm`] AND whose band mass is at least
  `layer_mode_ambiguity_fraction` of the primary's (a displaced dent or a sparse tail is
  not a population: single population, all rows kept, the unchanged gate then refuses a
  dented flange rather than trimming it); a third qualifying mode separated from both is
  AMBIGUOUS (all rows kept); no second mode is a single population (all rows kept);
* BAND membership (MEMBERSHIP_RULE): after each of at most `layer_refit_iterations` TLS
  refits of the two groups, a row belongs to the skin whose fitted plane is nearest (equal
  distances -> plane index 0, the primary-mode plane) when that distance is within
  `layer_membership_band_mm` (inclusive); a row outside both bands that lies STRICTLY
  BETWEEN the two planes is an UNASSIGNED between-skin row (published as `bridging_rows`,
  never measured, NOT a verified cut-edge or bridge identity); a row outside both bands that
  is NOT between the planes stays in its nearest group (never discarded): in the SELECTED
  population it meets the unchanged gates, in the OPPOSITE population only the pair
  planarity checks (that surface is never measured; its conformance is unknown); membership must
  reach a FIXED POINT within the declared refits (otherwise the separation abstains; no
  unbudgeted refit); an unassigned fraction above `max_bridging_row_fraction` or a group
  below `min_layer_rows` refuses the separation (all rows kept, identity unresolved); the
  band must exceed the feature's residual policy (joint contract, enforced at measurement);
  mode band masses are EXACT inclusive coordinate-band row counts, and a third population
  needs band mass >= `min_layer_rows` (population support, not a single-bin peak);
* the UNCHANGED V2 pair compatibility and the UNCHANGED smaller-coordinate choice; the kept
  skin then meets the UNCHANGED flange gate forms (`_plane`: spread, thickness ratio,
  orientation, max residual <= the declared residual policy). The band is declared wider
  than the residual policy so the gate is never satisfied by construction: rows between
  the policy and the band stay in the group and are reported by the gate, never trimmed.
  The maximum-residual gate applies only AFTER membership, so a measured flange is a
  SKIN-MEMBERSHIP SUBSET whose excluded rows (opposite skin, unassigned between-skin rows)
  are published per flange in `diagnostics["flange_support"]`; it is never a whole-crop
  planarity certificate or a conformance claim: a dent of the SELECTED skin toward the air
  side stays in its group and is refused by the unchanged gate; a deviation of the OPPOSITE
  skin (beyond it) stays in that group and faces only the pair planarity checks (its
  conformance is unknown, it is never measured); a between-skin displacement is an explicit,
  counted, unverified exclusion (never silently absorbed as a verified cut edge), bounded by
  `max_bridging_row_fraction`.
* `INNER_FACE_ELIGIBLE_STATUSES` stays EMPTY: every measured window is a generic
  conditional contact line with an unverified side hypothesis.

Original V2 header: V1 text plus per-window diagnostics and a declared material-side layer
selection.

Unchanged from V1: the feature model (one circular extrusion with two tangent
planar flanges, inner face), every sample-model gate (`_plane`, `_circle`,
tangency, sweep, condition, coverage) and the free-circle objective. NEW and
separately declared:

* Sign conventions. `radial` is the nominal unit vector from the bend axis to
  the inner flange contact; the sheet material lies on the +radial side, so a
  second (outer) skin sits about +t further along +radial. `_plane` is called
  with reference `-radial`, so every FITTED flange normal points toward the
  axis (air side): against it the outer skin has signed distance about -t and
  the inner arc has signed distance R*(1-cos theta) >= 0.
* Layer selection uses the crop's own TLS normal oriented toward +radial (the
  material-side-positive coordinate), never the fitted axis-pointing normal:
  rows are split at the largest gap of that coordinate when it exceeds
  `layer_split_gap_mm` with at least `min_layer_rows` on both sides, and the
  layer with the SMALLER coordinate (nearest the axis) is kept as the inner
  skin candidate. A single detected layer is kept unchanged and published as
  SINGLE_LAYER_IDENTITY_UNRESOLVED; it is never labelled inner.
* Arc rows must lie strictly on the axis side of BOTH fitted flange planes:
  signed distance against each fitted normal > plane_exclusion_mm (V1 kept
  |signed| > plane_exclusion_mm; the material side is now dropped as well).
* Diagnostics are computed for both flange sides and both weak-gate branches
  without raising, before and after selection, and are attached to the
  measurement or to the `_Unavailable` exception (`error.diagnostics`).

Nothing here consults CAD nominal values for the answer, Gold, or case
identity; a layer is never chosen because it fits better. Residuals and
conditioning remain uncalibrated engineering quantities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError


MAX_POINTS = 200_000
MAX_FEATURES = 16
MAX_FIT_EVALUATIONS = 80
MAX_FLAT_HULL_VERTICES = 2048
MATERIAL_SIDE_CONVENTION = "MATERIAL_ON_POSITIVE_NOMINAL_RADIAL_FITTED_NORMALS_POINT_TO_AXIS"
LAYER_SELECTION_IMPLEMENTATION = "TWO_MODE_BAND_MEMBERSHIP_ON_MATERIAL_SIDE_COORDINATE_KEEP_SMALLER_COMPATIBLE_PAIR_V2"
MAX_MODE_BINS = 4096
ARC_SIDE_FILTER_IMPLEMENTATION = "AXIS_SIDE_OF_BOTH_FITTED_FLANGE_PLANES_BEYOND_PLANE_EXCLUSION_V1"
# No presently defined material-side status carries VERIFIED side evidence: the two-layer
# selection is a scan-derived side HYPOTHESIS (smaller material-side coordinate of a
# compatible parallel pair), not an isolation proof; disabling the selection supplies no
# evidence either. Every status is therefore ineligible for inner-face-specific claims
# (no nominal inner-radius deviation, no inner-face carrier role); the generic measured
# contact geometry and the side-hypothesis metadata (status, layer selection, filter
# diagnostics) are retained. A future true eligibility path needs separately bound
# material-side evidence and its own review; it is not a per-status default here.
INNER_FACE_ELIGIBLE_STATUSES = ()
INNER_FACE_ELIGIBILITY_SCOPE = ("NO_PRESENT_STATUS_IS_INNER_FACE_ELIGIBLE_ALL_MATERIAL_SIDE_STATUSES_ARE_"
                                "UNVERIFIED_SIDE_HYPOTHESES_RETAINED_AS_METADATA")
MATERIAL_SIDE_STATUSES = ("CONTEXT_ASSUMED_NOT_VERIFIED", "SINGLE_LAYER_IDENTITY_UNRESOLVED",
    "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED", "MIXED_LAYER_EVIDENCE_IDENTITY_UNRESOLVED",
    "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED", "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED",
    "TWO_MODES_MEMBERSHIP_UNRESOLVED",   # V3: bridging fraction above the bound or a group below min rows
    "PROJECTION_PREMISE_UNRESOLVED")     # V3: a refitted skin normal departs from the crop TLS normal beyond the declared angle
INNER_FACE_CARRIER_ROLE = "INNER_FACE_CONDITIONAL_CONTACT_LINE_UNVERIFIED"
GENERIC_CARRIER_ROLE = "GENERIC_CONDITIONAL_CONTACT_LINE_SURFACE_IDENTITY_UNRESOLVED_NOT_INNER_ARC"


@dataclass(frozen=True)
class LayerSelectionPolicyV2:
    """Declared geometric row-selection policy (V1 fields unchanged in meaning, plus the V3 two-mode
    band membership fields); enabled=False reproduces V1 crops."""
    enabled: bool = True
    layer_split_gap_mm: float = 0.3
    min_layer_rows: int = 12
    max_layers_reported: int = 8
    max_layer_normal_angle_deg: float = 3.0
    max_layer_gap_mm: float = 6.0
    min_projected_interval_overlap: float = 0.5
    max_layer_rms_mm: float = 0.1
    layer_mode_bin_mm: float = 0.05          # histogram bin of the material-side coordinate (<= gap/3)
    layer_membership_band_mm: float = 0.1    # skin membership band around each fitted skin plane (<= gap/2)
    max_bridging_row_fraction: float = 0.35  # rows outside both bands; above this the separation is refused
    layer_refit_iterations: int = 6          # bounded membership/plane refits; membership must reach a fixed point within them
    layer_mode_ambiguity_fraction: float = 0.25  # minimum band mass of a mode, relative to the primary's, to count as a population

    def __post_init__(self):
        if (type(self.enabled) is not bool or type(self.min_layer_rows) is not int
                or type(self.max_layers_reported) is not int or type(self.layer_refit_iterations) is not int):
            raise ValueError("invalid layer selection policy types")
        for name, low, high in (("layer_split_gap_mm", 0., 100.), ("max_layer_normal_angle_deg", 0., 45.),
                                ("max_layer_gap_mm", 0., 1000.), ("min_projected_interval_overlap", 0., 1.),
                                ("max_layer_rms_mm", 0., 100.), ("layer_mode_bin_mm", 0., 100.),
                                ("layer_membership_band_mm", 0., 100.), ("max_bridging_row_fraction", 0., .5),
                                ("layer_mode_ambiguity_fraction", 0., 1.)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or not low < float(value) <= high:
                raise ValueError("invalid "+name)
            object.__setattr__(self, name, float(value))
        if self.max_layer_gap_mm <= self.layer_split_gap_mm:
            raise ValueError("max_layer_gap_mm must exceed layer_split_gap_mm")
        if not 12 <= self.min_layer_rows <= 100_000 or not 2 <= self.max_layers_reported <= 64:
            raise ValueError("invalid layer selection row/report limits")
        if self.layer_mode_bin_mm > self.layer_split_gap_mm / 3.:
            raise ValueError("layer_mode_bin_mm must not exceed a third of layer_split_gap_mm")
        if self.layer_membership_band_mm > self.layer_split_gap_mm / 2.:
            raise ValueError("layer_membership_band_mm must not exceed half of layer_split_gap_mm")
        if not 1 <= self.layer_refit_iterations <= 8:
            raise ValueError("invalid layer_refit_iterations")



@dataclass(frozen=True)
class CADGuidedFeature:
    feature_id: str
    nominal_radius_mm: float
    nominal_sweep_deg: float
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    start_radial: tuple[float, float, float]
    span_mm: tuple[float, float]
    flange_window_mm: tuple[float, float]
    search_margin_mm: float
    max_fit_residual_mm: float
    plane_exclusion_mm: float
    max_tangency_error_mm: float
    nominal_source: str
    unit: str = "mm"
    surface: str = "inner"
    arc_guard_deg: float = 8.0
    min_arc_sweep_deg: float = 20.0
    max_normal_deviation_deg: float = 15.0
    max_axis_deviation_deg: float = 10.0
    max_circle_condition: float = 1000.0
    min_plane_spread_ratio: float = 0.01
    min_axial_span_fraction: float = 0.5
    fit_model: str = "free_circle"

    def __post_init__(self):
        if (not isinstance(self.feature_id, str) or not self.feature_id.strip()
                or not isinstance(self.nominal_source, str) or not self.nominal_source.strip()):
            raise ValueError("feature/source identity must be explicit nonempty strings")
        if self.unit != "mm" or self.surface != "inner":
            raise ValueError("only explicit millimetre single-inner-face input is supported")
        if self.fit_model not in ("free_circle", "measured_flange_tangent"):
            raise ValueError("unsupported circle fit model")
        for name, size in (("origin", 3), ("axis", 3), ("start_radial", 3),
                           ("span_mm", 2), ("flange_window_mm", 2)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        for name in ("nominal_radius_mm", "nominal_sweep_deg", "search_margin_mm",
                     "max_fit_residual_mm", "plane_exclusion_mm", "max_tangency_error_mm",
                     "arc_guard_deg", "min_arc_sweep_deg", "max_normal_deviation_deg",
                     "max_axis_deviation_deg", "max_circle_condition", "min_plane_spread_ratio",
                     "min_axial_span_fraction"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, float(value))
        a, n = np.asarray(self.axis), np.asarray(self.start_radial)
        if not np.allclose([a @ a, n @ n, a @ n], [1, 1, 0], atol=1e-12, rtol=0):
            raise ValueError("axis and start radial must be orthonormal")
        if (not 20 <= abs(self.nominal_sweep_deg) <= 160
                or self.nominal_radius_mm <= 0
                or not self.span_mm[0] < self.span_mm[1]
                or not 0 < self.flange_window_mm[0] < self.flange_window_mm[1]
                or not 0 < self.max_fit_residual_mm < self.plane_exclusion_mm < self.search_margin_mm
                or self.max_tangency_error_mm <= 0
                or not 0 < self.arc_guard_deg < abs(self.nominal_sweep_deg) / 2
                or not 0 < self.min_arc_sweep_deg < 180
                or not 0 < self.max_normal_deviation_deg < 45
                or not 0 < self.max_axis_deviation_deg < 45
                or not 1 < self.max_circle_condition <= 1e6
                or not 0 < self.min_plane_spread_ratio < 1
                or not 0 < self.min_axial_span_fraction <= 1):
            raise ValueError("invalid or unsupported feature/search policy")
        # Bound arithmetic before products/squares; this is an engineering scope,
        # not a universal CAD size or precision claim.
        lengths = (*self.origin, *self.span_mm, *self.flange_window_mm,
                   self.nominal_radius_mm, self.search_margin_mm)
        if max(abs(x) for x in lengths) > 1e6 or min(
                self.nominal_radius_mm, self.max_fit_residual_mm) < 1e-8:
            raise ValueError("unsupported numerical scale")


class _Unavailable(Exception):
    pass


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-12:
        raise _Unavailable("DEGENERATE_DIRECTION")
    return vector / length


def _finite_list(values):
    return [float(v) if math.isfinite(float(v)) else None for v in values]


def _spectrum(points):
    """Population RMS spectrum s_i/sqrt(N) and the TLS normal of centered rows (no gates)."""
    center = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    root = math.sqrt(len(points))
    return center, singular, vh, [float(v) / root for v in singular]


def _flange_diagnostics(points, reference, feature, *, role, ray=None):
    """Both weak-gate branches and the residual gate, evaluated without raising.

    Extents: `axial_extent_mm` along the declared feature axis, `flange_ray_extent_mm`
    along the declared flange ray of this side (the real ray, when supplied), and the
    principal in-plane extents of the centered rows; none is a coverage certificate.
    """
    out = dict(role=role, rows=int(len(points)), status="NOT_EVALUATED", singular_over_sqrt_n=None,
        rank_ratio=None, spread_gate_passed=None, thickness_ratio_gate_passed=None,
        reference_angle_deg=None, max_residual_mm=None, rms_residual_mm=None,
        residual_gate_passed=None, axial_extent_mm=None, flange_ray_extent_mm=None,
        principal_extent_mm=None, second_principal_extent_mm=None,
        normal_extent_mm=None, first_failure=None)
    if len(points) < 12:
        out.update(status="INSUFFICIENT_ROWS", first_failure="INSUFFICIENT_FLANGE_SUPPORT")
        return out
    try:
        center, singular, vh, alpha = _spectrum(points)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        out.update(status="SPECTRUM_UNAVAILABLE", first_failure=type(error).__name__)
        return out
    out["singular_over_sqrt_n"] = alpha
    root = math.sqrt(len(points))
    if singular[0] <= 1e-12:
        out.update(status="RANK_DEFICIENT", first_failure="FLANGE_SUPPORT_RANK_DEFICIENT")
        return out
    out["rank_ratio"] = float(singular[1] / singular[0])
    spread = bool(singular[1] > 10 * feature.max_fit_residual_mm * root)
    ratio = bool(singular[2] / singular[1] < 0.2) if singular[1] > 0 else False
    out.update(spread_gate_passed=spread, thickness_ratio_gate_passed=ratio,
               spread_gate_minor_rms_mm=float(singular[1] / root),
               spread_gate_threshold_mm=float(10 * feature.max_fit_residual_mm),
               thickness_ratio=float(singular[2] / singular[1]) if singular[1] > 0 else None)
    normal = vh[-1]
    if normal @ reference < 0:
        normal = -normal
    cosine = float(np.clip(normal @ reference, -1., 1.))
    out["reference_angle_deg"] = math.degrees(math.acos(cosine))
    residual = (points - center) @ normal
    out.update(max_residual_mm=float(np.abs(residual).max()), rms_residual_mm=float(np.sqrt(np.mean(residual**2))),
               residual_gate_passed=bool(float(np.abs(residual).max()) <= feature.max_fit_residual_mm))
    axis = np.asarray(feature.axis)
    out.update(axial_extent_mm=float(np.ptp((points - center) @ axis)),
               principal_extent_mm=float(np.ptp((points - center) @ vh[0])),
               second_principal_extent_mm=float(np.ptp((points - center) @ vh[1])),
               flange_ray_extent_mm=None if ray is None else float(np.ptp((points - center) @ np.asarray(ray))),
               normal_extent_mm=float(np.ptp(residual)))
    if out["rank_ratio"] < feature.min_plane_spread_ratio:
        out.update(status="RANK_DEFICIENT", first_failure="FLANGE_SUPPORT_RANK_DEFICIENT")
    elif not (spread and ratio):
        out.update(status="WEAK", first_failure="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY")
    elif cosine < math.cos(math.radians(feature.max_normal_deviation_deg)):
        out.update(status="ORIENTATION", first_failure="FLANGE_ORIENTATION_OUTSIDE_SEARCH_SCOPE")
    elif not out["residual_gate_passed"]:
        out.update(status="NOT_PLANAR", first_failure="FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL")
    else:
        out.update(status="PASS", first_failure=None)
    return out


def _two_modes(coordinate, policy):
    """Deterministic two-population detection on the 1-D material-side coordinate.

    Fixed-width bins over [min, max] (`layer_mode_bin_mm`, widened only when the span would need more
    than MAX_MODE_BINS bins, and then published); a local maximum is a bin whose count exceeds the
    previous bin's and is at least the next bin's (the first bin of a plateau); the primary mode is the
    tallest local maximum (ties: the lower bin); the secondary mode is the tallest other local maximum
    whose centre separation from the primary lies in [layer_split_gap_mm, max_layer_gap_mm]. Every local
    maximum also carries its band MASS: the EXACT number of rows whose coordinate lies within
    +-layer_membership_band_mm (inclusive) of the bin CENTRE, counted on the sorted coordinate with two
    binary searches per maximum (bounded: one sort plus O(log n) per local maximum; never a quantized bin
    window). A further local maximum separated from BOTH modes by at least the gap floor whose band mass
    is at least `min_layer_rows` (band-population support, not a single-bin peak count) AND reaches
    `layer_mode_ambiguity_fraction` of the smaller skin's band mass is a third population (ambiguous);
    sparse between-skin rows never reach that mass. Bin counts and masses are reported; nothing here fits a
    plane or consults fit quality.
    """
    lo, hi = float(coordinate.min()), float(coordinate.max())
    span = max(0., hi - lo)
    width = float(policy.layer_mode_bin_mm)
    bins = int(math.floor(span / width)) + 1
    widened = False
    if bins > MAX_MODE_BINS:
        width, bins, widened = span / (MAX_MODE_BINS - 1), MAX_MODE_BINS, True
    index = np.minimum(np.floor((coordinate - lo) / width).astype(np.int64), bins - 1)
    counts = np.bincount(index, minlength=bins)
    centres = lo + (np.arange(bins) + 0.5) * width
    maxima = [i for i in range(bins) if counts[i] > 0
              and (i == 0 or counts[i] > counts[i - 1]) and (i == bins - 1 or counts[i] >= counts[i + 1])]
    ranked = sorted(maxima, key=lambda i: (-int(counts[i]), i))
    ordered = np.sort(coordinate)
    band = float(policy.layer_membership_band_mm)
    def mass(i):                                   # exact inclusive coordinate band around the bin centre
        centre = float(centres[i])
        return int(np.searchsorted(ordered, centre + band, side="right")
                   - np.searchsorted(ordered, centre - band, side="left"))
    record = dict(bin_width_mm=width, bin_count=bins, bins_widened=widened, coordinate_min_mm=lo, coordinate_max_mm=hi,
                  mass_semantics=("EXACT_ROW_COUNT_WITHIN_INCLUSIVE_MEMBERSHIP_BAND_OF_BIN_CENTRE_NOT_A_BIN_WINDOW_"
                                  "INTERVAL_BOUNDS_CENTRE_MINUS_BAND_AND_CENTRE_PLUS_BAND_EVALUATED_IN_BINARY64"),
                  mass_band_mm=band, ambiguity_fraction=float(policy.layer_mode_ambiguity_fraction),
                  third_population_support="BAND_MASS_AT_LEAST_MIN_LAYER_ROWS_NOT_SINGLE_BIN_PEAK",
                  local_maxima=[dict(centre_mm=float(centres[i]), rows=int(counts[i]), band_mass=mass(i))
                                for i in ranked[:policy.max_layers_reported]],
                  local_maximum_count=len(maxima), primary=None, secondary=None, ambiguous_third=None)
    if not ranked:
        return None, None, record
    primary = ranked[0]
    record["primary"] = dict(centre_mm=float(centres[primary]), rows=int(counts[primary]), band_mass=mass(primary))
    secondary = None
    population_floor = policy.layer_mode_ambiguity_fraction * mass(primary)
    for i in ranked[1:]:
        separation = abs(float(centres[i] - centres[primary]))
        if policy.layer_split_gap_mm <= separation <= policy.max_layer_gap_mm and mass(i) >= population_floor:
            secondary = i
            break
    record["population_mass_floor"] = float(population_floor)
    if secondary is None:
        return float(centres[primary]), None, record
    record["secondary"] = dict(centre_mm=float(centres[secondary]), rows=int(counts[secondary]), band_mass=mass(secondary),
                               separation_mm=abs(float(centres[secondary] - centres[primary])))
    threshold = policy.layer_mode_ambiguity_fraction * min(mass(primary), mass(secondary))
    for i in ranked:
        if i in (primary, secondary) or mass(i) < policy.min_layer_rows or mass(i) < threshold:
            continue
        if (abs(float(centres[i] - centres[primary])) >= policy.layer_split_gap_mm
                and abs(float(centres[i] - centres[secondary])) >= policy.layer_split_gap_mm):
            record["ambiguous_third"] = dict(centre_mm=float(centres[i]), rows=int(counts[i]), band_mass=mass(i),
                                             mass_threshold=float(threshold))
            break
    return float(centres[primary]), float(centres[secondary]), record


MEMBERSHIP_RULE = dict(
    nearest_plane="ARGMIN_OF_ABSOLUTE_POINT_TO_PLANE_DISTANCES_OVER_THE_TWO_PUBLISHED_GROUP_PLANES",
    tie="EQUAL_DISTANCES_ASSIGN_PLANE_INDEX_0_THE_PRIMARY_MODE_PLANE",
    member="NEAREST_PLANE_WHEN_ITS_DISTANCE_IS_WITHIN_THE_INCLUSIVE_MEMBERSHIP_BAND",
    unassigned="OUTSIDE_BOTH_BANDS_AND_STRICTLY_BETWEEN_THE_TWO_PLANES_SIGNED_DISTANCES_OF_OPPOSITE_SIGN",
    outside_sandwich=("OUTSIDE_BOTH_BANDS_AND_NOT_BETWEEN_THE_PLANES_ASSIGNED_TO_THE_NEAREST_PLANE_"
                      "IN_THE_SELECTED_POPULATION_THEY_MEET_THE_UNCHANGED_GATES_IN_THE_OPPOSITE_POPULATION_ONLY_THE_PAIR_PLANARITY_CHECKS"),
    opposite_surface="OPPOSITE_POPULATION_IS_NEVER_MEASURED_ITS_CONFORMANCE_IS_UNKNOWN",
    unassigned_identity="UNASSIGNED_BETWEEN_SKIN_ROWS_NOT_VERIFIED_CUT_EDGE_OR_BRIDGE_IDENTITY")


def _band_membership(points, planes, band):
    """Labels over the crop rows for two co-oriented planes (MEMBERSHIP_RULE): k = nearest plane k (ties -> 0)
    when that distance is within `band` (inclusive); -1 (unassigned between-skin row) when the row is outside
    both bands AND strictly between the two planes; a row outside both bands that is NOT between the planes
    (air side of the lower skin, far side of the upper skin) belongs to its nearest plane, so a displaced or
    foreign population outside the sandwich is never discarded: in the SELECTED (measured) population it meets
    the unchanged flange gates; in the OPPOSITE population it is subject only to the pair-compatibility planarity
    checks (RMS, rank) and that surface's conformance stays unknown. Returns (labels, absolute distances, between
    mask, within-band mask)."""
    signed = np.stack([(points - center) @ normal for center, normal in planes], axis=1)
    distances = np.abs(signed)
    nearest = np.argmin(distances, axis=1)
    within = distances[np.arange(len(points)), nearest] <= band
    between = (signed[:, 0] * signed[:, 1]) < 0.
    return np.where(within | ~between, nearest, -1), distances, between, within


def _group_plane(points):
    center, singular, vh, alpha = _spectrum(points)
    normal = vh[-1]
    rms = float(singular[2] / math.sqrt(len(points)))
    rank_ratio = float(singular[1] / singular[0]) if singular[0] > 0 else 0.
    return center, normal, rms, rank_ratio, vh


def _interval_overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    smaller = min(a[1] - a[0], b[1] - b[0])
    if smaller <= 0.:
        return 0.
    return max(0., hi - lo) / smaller


def _layer_pair_compatibility(points, lower, radial, feature, policy):
    """Scan-derived checks before two groups may be called a parallel layer pair.

    Each group must be a planar population in its own right (rank ratio, RMS), the two
    fitted normals must be parallel within the declared angle, the offset between the
    groups along their mean normal must lie within the declared gap window, and the
    two groups' projected bounding intervals (on the lower group's first two TLS
    directions) must overlap by the declared fraction of the shorter interval. That
    interval overlap is a bounding-box proxy, not an area or material-footprint
    overlap: a small island inside a hole of an annular lower group scores 1. A
    tilted single plane sampled in two separated bands fails the interval proxy; a
    contaminant exceeding the declared RMS/rank bounds fails the planarity check
    (arbitrarily shallow curvature can pass any positive RMS bound). The TLS in-plane
    basis is unique only when the two in-plane singular values differ (a lower
    `rank_ratio` near 1 means a non-unique, basis-dependent proxy). Nothing here
    consults the fit quality of the eventual selection.
    """
    record = dict(status="INCOMPATIBLE", reasons=[], lower_rows=int(lower.sum()), upper_rows=int((~lower).sum()))
    groups = []
    for name, mask in (("lower", lower), ("upper", ~lower)):
        subset = points[mask]
        center, normal, rms, rank_ratio, vh = _group_plane(subset)
        if normal @ radial < 0:
            normal = -normal
        groups.append((name, center, normal, rms, rank_ratio, vh, subset))
        record[name+"_rms_mm"] = rms
        record[name+"_rank_ratio"] = rank_ratio
        if rank_ratio < feature.min_plane_spread_ratio:
            record["reasons"].append(name.upper()+"_GROUP_RANK_DEFICIENT")
        if rms > policy.max_layer_rms_mm:
            record["reasons"].append(name.upper()+"_GROUP_NOT_PLANAR_WITHIN_LAYER_RMS")
    (_, c0, n0, _, _, vh0, s0), (_, c1, n1, _, _, _, s1) = groups
    cosine = float(np.clip(n0 @ n1, -1., 1.))
    angle = math.degrees(math.acos(cosine))
    record["normal_angle_deg"] = angle
    if angle > policy.max_layer_normal_angle_deg:
        record["reasons"].append("GROUP_NORMALS_NOT_PARALLEL")
    mean_normal = n0 + n1
    if float(np.linalg.norm(mean_normal)) < 1e-12:
        record["reasons"].append("GROUP_NORMALS_DEGENERATE")
        mean_normal = n0
    mean_normal = mean_normal / float(np.linalg.norm(mean_normal))
    offset = float(mean_normal @ (c1 - c0))
    record["offset_along_mean_normal_mm"] = offset
    if not policy.layer_split_gap_mm <= offset <= policy.max_layer_gap_mm:
        record["reasons"].append("GROUP_OFFSET_OUTSIDE_DECLARED_LAYER_GAP_WINDOW")
    overlaps, intervals = [], []
    for axis_vector in (vh0[0], vh0[1]):
        a = s0 @ axis_vector
        b = s1 @ axis_vector
        lower_interval, upper_interval = (float(a.min()), float(a.max())), (float(b.min()), float(b.max()))
        intervals.append(dict(lower=list(lower_interval), upper=list(upper_interval)))
        overlaps.append(_interval_overlap(lower_interval, upper_interval))
    record["projected_bounding_interval_overlap"] = overlaps
    record["projected_bounding_intervals_mm"] = intervals
    record["projection_basis"] = [vh0[0].tolist(), vh0[1].tolist()]
    record["overlap_scope"] = ("BOUNDING_INTERVAL_OVERLAP_FRACTION_OF_SHORTER_INTERVAL_ON_LOWER_GROUP_TLS_BASIS_"
                               "NOT_AREA_OR_MATERIAL_FOOTPRINT_BASIS_NOT_UNIQUE_WHEN_LOWER_RANK_RATIO_NEAR_1")
    if min(overlaps) < policy.min_projected_interval_overlap:
        record["reasons"].append("GROUP_PROJECTED_BOUNDING_INTERVALS_DO_NOT_OVERLAP")
    if not record["reasons"]:
        record["status"] = "COMPATIBLE_PARALLEL_LAYER_PAIR_UNVERIFIED_MATERIAL"
    return record


def _select_flange_layer(points, radial, feature, policy, role):
    """Return (kept_mask_over_points, record).

    Selection is geometric only: two-mode detection on the material-side coordinate, band membership
    around each skin's fitted plane with bounded refits (rows outside both bands are bridging rows), the
    unchanged scan-derived pair compatibility, and the unchanged smaller-coordinate choice. Only a
    compatible pair selects (the smaller-coordinate group, nearest the axis); every other outcome keeps all
    rows and publishes an unresolved identity. Fit quality never selects; the residual gate is applied
    afterwards, unchanged, to the retained rows.
    """
    record = dict(role=role, implementation=LAYER_SELECTION_IMPLEMENTATION,
        material_side_convention=MATERIAL_SIDE_CONVENTION, input_rows=int(len(points)),
        enabled=bool(policy.enabled), qualifying_groups_detected=None, modes=None,
        membership_band_mm=float(policy.layer_membership_band_mm),
        band_exceeds_declared_residual=bool(policy.layer_membership_band_mm > feature.max_fit_residual_mm),
        refit_iterations_used=0, refit_converged=None, group_planes=None, group_rows=None,
        bridging_rows=None, bridging_fraction=None, max_bridging_row_fraction=float(policy.max_bridging_row_fraction),
        sort_normal=None, sort_center_mm=None,
        kept_rule="ALL_ROWS", kept_rows=int(len(points)), dropped_rows=0, dropped_other_skin_rows=0,
        dropped_bridging_rows=0, kept_layer="ALL_ROWS", layer_means_mm=None, layer_rows=None,
        pair_compatibility=None, identity="NOT_EVALUATED",
        projection="CROP_TLS_NORMAL_ORIENTED_TO_POSITIVE_NOMINAL_RADIAL",
        summary_scope=("BOUNDED_DIAGNOSTIC_SUMMARY_LOCAL_MAXIMA_TRUNCATED_TO_MAX_LAYERS_REPORTED_"
                       "PARTITION_RECONSTRUCTIBLE_FROM_PUBLISHED_GROUP_PLANES_AND_BAND_ON_RETAINED_ROWS"))
    keep = np.ones(len(points), dtype=bool)
    if not policy.enabled or len(points) < 2:
        record["identity"] = "LAYER_SELECTION_DISABLED" if not policy.enabled else "INSUFFICIENT_ROWS"
        return keep, record
    center, singular, vh, _ = _spectrum(points)
    normal = vh[-1]
    if singular[0] <= 1e-12 or not np.isfinite(normal).all():
        record["identity"] = "SPECTRUM_UNAVAILABLE"
        return keep, record
    if normal @ radial < 0:
        normal = -normal
    coordinate = (points - center) @ normal
    record.update(sort_normal=normal.tolist(), sort_center_mm=center.tolist())
    primary, secondary, modes = _two_modes(coordinate, policy)
    record["modes"] = modes
    if primary is None or secondary is None:
        record.update(qualifying_groups_detected=1, identity="SINGLE_LAYER_IDENTITY_UNRESOLVED",
                      layer_means_mm=[float(coordinate.mean())], layer_rows=[int(len(points))])
        return keep, record
    if modes.get("ambiguous_third") is not None:
        record.update(qualifying_groups_detected=3, identity="AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED")
        return keep, record
    record["qualifying_groups_detected"] = 2
    # Band membership around the two skin planes, starting from the mode centres on the crop normal and
    # refit a bounded number of times; rows outside both bands are bridging rows.
    planes = [(center + primary * normal, normal), (center + secondary * normal, normal)]
    labels, _, between, within = _band_membership(points, planes, policy.layer_membership_band_mm)
    converged, entries, completed, starved = False, 0, 0, False
    for _ in range(policy.layer_refit_iterations):
        entries += 1                                   # loop entries (attempted refits)
        if any(int((labels == k).sum()) < 3 for k in (0, 1)):
            starved = True
            break
        refitted = []
        for k in (0, 1):
            group_center, group_normal, _, _, _ = _group_plane(points[labels == k])
            if group_normal @ normal < 0:
                group_normal = -group_normal
            refitted.append((group_center, group_normal))
        planes = refitted
        completed += 1                                 # both group planes refitted and membership recomputed
        new_labels, _, between, within = _band_membership(points, planes, policy.layer_membership_band_mm)
        if np.array_equal(new_labels, labels):
            converged = True
            break
        labels = new_labels
    record.update(refit_iterations_used=completed, refit_loop_entries=entries, refit_converged=converged,
                  refit_accounting="REFIT_ITERATIONS_USED_COUNTS_COMPLETED_TWO_PLANE_TLS_REFITS_WITH_MEMBERSHIP_RECOMPUTED_"
                                   "REFIT_LOOP_ENTRIES_COUNTS_ATTEMPTS_INCLUDING_A_STARVED_ENTRY")
    # Projection premise: the material-side coordinate was taken along the crop TLS normal; if a refitted skin
    # normal departs from it beyond the declared parallelism angle the premise is not validated and the
    # selection abstains (all rows kept, identity unresolved).
    tilt = [math.degrees(math.acos(min(1., abs(float(n @ normal))))) for _, n in planes]
    record["crop_normal_to_group_normal_deg"] = tilt
    group_rows = [int((labels == k).sum()) for k in (0, 1)]
    bridging = int((labels == -1).sum())
    outside = int((~within & ~between).sum())          # outside both bands and not between: kept in the nearest group
    record.update(group_rows=group_rows, bridging_rows=bridging, bridging_fraction=bridging / float(len(points)),
        outside_sandwich_member_rows=outside, membership_rule=dict(MEMBERSHIP_RULE),
        group_planes=[dict(center_mm=[float(v) for v in c], normal=[float(v) for v in n],
                           coordinate_mm=float((c - center) @ normal)) for c, n in planes], unresolved_reason=None)
    # A fixed point is required: at convergence the published planes are fitted from exactly the final labels and
    # the final labels are exactly their band membership, so the premise check, the pair compatibility and the
    # unchanged gate all see the same partition. Exhausting the declared refits without a fixed point abstains;
    # no extra unbudgeted refit is performed.
    if not converged:
        record.update(identity="TWO_MODES_MEMBERSHIP_UNRESOLVED",
                      unresolved_reason=("GROUP_BELOW_THREE_ROWS_DURING_REFIT" if starved
                                         else "MEMBERSHIP_NOT_A_FIXED_POINT_WITHIN_DECLARED_REFITS"))
        return keep, record
    if min(group_rows) < policy.min_layer_rows or record["bridging_fraction"] > policy.max_bridging_row_fraction:
        record.update(identity="TWO_MODES_MEMBERSHIP_UNRESOLVED",
                      unresolved_reason=("GROUP_BELOW_MIN_LAYER_ROWS" if min(group_rows) < policy.min_layer_rows
                                         else "BRIDGING_FRACTION_ABOVE_DECLARED_MAXIMUM"))
        return keep, record
    if max(tilt) > policy.max_layer_normal_angle_deg:
        record.update(identity="PROJECTION_PREMISE_UNRESOLVED",
                      unresolved_reason="REFITTED_SKIN_NORMAL_DEPARTS_FROM_CROP_TLS_NORMAL_BEYOND_DECLARED_ANGLE")
        return keep, record
    # Smaller material-side coordinate (nearest the axis) is the lower group; the unchanged pair
    # compatibility runs on the member rows only (bridging rows belong to neither skin).
    order = sorted((0, 1), key=lambda k: record["group_planes"][k]["coordinate_mm"])
    lower_k, upper_k = order
    members = labels >= 0
    member_points = points[members]
    lower = labels[members] == lower_k
    means = [float(coordinate[labels == lower_k].mean()), float(coordinate[labels == upper_k].mean())]
    compatibility = _layer_pair_compatibility(member_points, lower, radial, feature, policy)
    record.update(layer_means_mm=means, layer_rows=[int(lower.sum()), int((~lower).sum())],
        observed_layer_gap_mm=float(means[1] - means[0]), pair_compatibility=compatibility)
    if compatibility["status"] != "COMPATIBLE_PARALLEL_LAYER_PAIR_UNVERIFIED_MATERIAL":
        record["identity"] = "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED"
        return keep, record
    keep = labels == lower_k
    record.update(kept_rows=int(keep.sum()), dropped_rows=int((~keep).sum()),
        dropped_other_skin_rows=int((labels == upper_k).sum()), dropped_bridging_rows=bridging,
        kept_layer="SMALLER_MATERIAL_SIDE_COORDINATE",
        kept_rule=("ROW_KEPT_IFF_ITS_LABEL_UNDER_MEMBERSHIP_RULE_IS_THE_SMALLER_COORDINATE_GROUP_PLANE_"
                   "NEAREST_PLANE_TIES_TO_INDEX_0_WITHIN_BAND_OR_OUTSIDE_THE_SANDWICH_AFTER_THE_FIXED_POINT"),
        identity="TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED")
    return keep, record


def _plane(points, reference, feature):
    if len(points) < 12:
        raise _Unavailable("INSUFFICIENT_FLANGE_SUPPORT")
    center = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= 1e-12 or singular[1] / singular[0] < feature.min_plane_spread_ratio:
        raise _Unavailable("FLANGE_SUPPORT_RANK_DEFICIENT")
    if (singular[1] <= 10 * feature.max_fit_residual_mm * math.sqrt(len(points))
            or singular[2] / singular[1] >= 0.2):
        raise _Unavailable("FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY")
    normal = vh[-1]
    if normal @ reference < 0:
        normal = -normal
    if normal @ reference < math.cos(math.radians(feature.max_normal_deviation_deg)):
        raise _Unavailable("FLANGE_ORIENTATION_OUTSIDE_SEARCH_SCOPE")
    residual = np.abs((points - center) @ normal)
    if float(residual.max()) > feature.max_fit_residual_mm:
        raise _Unavailable("FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL")
    return normal, center, float(residual.max())


def circle_residual_jacobian(parameters, points):
    """Geometric radial residual/Jacobian in normalized cross-section units."""
    center, radius = parameters[:2], parameters[2]
    delta = points - center
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1e-14):
        raise _Unavailable("CIRCLE_CENTER_ON_SAMPLE")
    jacobian = np.column_stack((-delta / distance[:, None], -np.ones(len(points))))
    return distance - radius, jacobian


def _flat_strip_width(points):
    """Minimum enclosing 2-D strip width, bounded hull-edge enumeration.

    A convex polygon attains minimum width normal to one of its edges. This
    frees BOTH line direction and offset, unlike a max-residual TLS test.
    No Qhull joggling/point perturbation. Ordinary floating arithmetic is not
    an interval-arithmetic or calibrated error certificate.
    """
    try:
        hull = points[ConvexHull(points).vertices]
    except QhullError as error:
        raise _Unavailable("FLAT_HULL_UNAVAILABLE") from error
    if len(hull) > MAX_FLAT_HULL_VERTICES:
        raise _Unavailable("FLAT_WITNESS_BUDGET")
    width = math.inf
    for p, q in zip(hull, np.roll(hull, -1, axis=0)):
        edge = q - p
        # A near-duplicate projected observation can create a very short hull
        # edge. Its length is not a physical-direction identifiability test.
        length = math.hypot(float(edge[0]), float(edge[1]))
        if length == 0 or not math.isfinite(length):
            raise _Unavailable("FLAT_HULL_DEGENERATE_EDGE")
        normal = np.array([-edge[1], edge[0]]) / length
        projected = hull @ normal
        width = min(width, float(np.ptp(projected)))
    return width


def _circle(points, feature):
    if len(points) < 12:
        raise _Unavailable("INSUFFICIENT_ARC_SUPPORT")
    mean = points.mean(axis=0)
    centered = points - mean
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale <= 1e-10 or singular[-1] <= 1e-12:
        raise _Unavailable("ARC_SUPPORT_RANK_DEFICIENT")
    # A free-offset straight generator plane is a competing explanation at the
    # same declared residual limit. Include an arithmetic grey zone in abstention.
    guard = 128 * np.finfo(float).eps * max(1., float(np.max(np.abs(centered))))
    if _flat_strip_width(centered) / 2 <= feature.max_fit_residual_mm + guard:
        raise _Unavailable("ARC_COMPATIBLE_WITH_FLAT")
    normalized = centered / scale
    design = np.column_stack((2 * normalized, np.ones(len(points))))
    seed, _, rank, _ = np.linalg.lstsq(design, np.sum(normalized**2, axis=1), rcond=None)
    radius_sq = float(seed[2] + seed[:2] @ seed[:2])
    if rank != 3 or radius_sq <= 0 or not math.isfinite(radius_sq):
        raise _Unavailable("CIRCLE_INITIALIZER_UNAVAILABLE")
    initial = np.r_[seed[:2], math.sqrt(radius_sq)]
    result = least_squares(
        lambda p: circle_residual_jacobian(p, normalized)[0], initial,
        jac=lambda p: circle_residual_jacobian(p, normalized)[1],
        bounds=([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf]),
        max_nfev=MAX_FIT_EVALUATIONS, ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise _Unavailable("CIRCLE_SOLVER_INCOMPLETE")
    residual, jacobian = circle_residual_jacobian(result.x, normalized)
    s = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(s[0] / s[-1]) if s[-1] > 0 else math.inf
    if not math.isfinite(condition) or condition > feature.max_circle_condition:
        raise _Unavailable("ARC_ILL_CONDITIONED")
    max_residual = float(np.abs(residual).max() * scale)
    if max_residual > feature.max_fit_residual_mm:
        raise _Unavailable("ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL")
    return (mean + result.x[:2] * scale, float(result.x[2] * scale),
            max_residual, condition, float(s[-1] / math.sqrt(len(points))))


def tangent_residual_jacobian(radius, points, h, g):
    """Residual and Nx1 derivative for c(R)=h+R*g, using measured planes only."""
    radius = float(np.asarray(radius).reshape(-1)[0])
    delta = points - h - radius * g
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1e-14):
        raise _Unavailable("TANGENT_CENTER_ON_SAMPLE")
    derivative = -(delta / distance[:, None]) @ g - 1.
    return distance - radius, derivative[:, None]


def _tangent_circle(points, plane_matrix, plane_offsets, sweep_deg, seed_radius, feature):
    """New named hypothesis: scan-plane tangency constrains center, not CAD.

    A*c=b+R*1 for INNER normals, hence c=h+R*g. Each point must also lie
    inside the measured contact sector; its two linear inequalities restrict R
    before optimization, excluding the wrong-contact-sector second-circle branch.
    No measurement uncertainty follows from treating the fitted planes as fixed.
    """
    h = np.linalg.solve(plane_matrix, plane_offsets)
    g = np.linalg.solve(plane_matrix, np.ones(2))
    d = points - h
    phi = math.radians(sweep_deg)
    sign = math.copysign(1., phi)
    sector_normals = (np.array([0., sign]),
                      sign * np.array([math.sin(phi), -math.cos(phi)]))
    lower, upper = 1e-12 * seed_radius, math.inf
    for normal in sector_normals:
        alpha = d @ normal
        beta = float(g @ normal)
        if abs(beta) <= 1e-12:
            if np.any(alpha < 0):
                raise _Unavailable("TANGENT_CONTACT_SECTOR_UNAVAILABLE")
        elif beta > 0:
            upper = min(upper, float(np.min(alpha / beta)))
        else:
            lower = max(lower, float(np.max(alpha / beta)))
    if upper <= lower or lower <= 0 or not math.isfinite(lower):
        raise _Unavailable("TANGENT_CONTACT_SECTOR_UNAVAILABLE")
    scale = seed_radius  # Measured free-circle initializer, never nominal CAD R.
    lower /= scale
    upper /= scale
    initial = max(1., np.nextafter(lower, math.inf))
    if math.isfinite(upper):
        initial = min(initial, np.nextafter(upper, -math.inf))
    if not lower < initial < upper:
        raise _Unavailable("TANGENT_CONTACT_SECTOR_NUMERICALLY_UNRESOLVED")
    normalized, h_normalized = points / scale, h / scale
    result = least_squares(
        lambda r: tangent_residual_jacobian(r, normalized, h_normalized, g)[0], [initial],
        jac=lambda r: tangent_residual_jacobian(r, normalized, h_normalized, g)[1],
        bounds=([lower], [upper]), max_nfev=MAX_FIT_EVALUATIONS,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise _Unavailable("TANGENT_SOLVER_INCOMPLETE")
    residual, jacobian = tangent_residual_jacobian(result.x, normalized, h_normalized, g)
    rms_derivative = float(np.linalg.norm(jacobian) / math.sqrt(len(points)))
    if rms_derivative < 1 / feature.max_circle_condition:
        raise _Unavailable("TANGENT_RADIUS_WEAKLY_IDENTIFIED")
    max_residual = float(np.abs(residual).max() * scale)
    if max_residual > feature.max_fit_residual_mm:
        raise _Unavailable("ARC_NOT_TANGENT_CIRCLE_AT_DECLARED_RESIDUAL")
    radius = float(result.x[0] * scale)
    return h + radius * g, radius, max_residual, rms_derivative


def _measure(points, feature, layer_policy=None):
    if layer_policy is None:
        layer_policy = LayerSelectionPolicyV2(enabled=False)
    if type(layer_policy) is not LayerSelectionPolicyV2:
        raise ValueError("layer_policy must be a LayerSelectionPolicyV2")
    if layer_policy.enabled and not layer_policy.layer_membership_band_mm > feature.max_fit_residual_mm:
        # Joint feature/policy contract: the membership band must exceed the residual policy, otherwise the
        # unchanged maximum-residual gate would be satisfied by construction for the retained rows.
        raise ValueError("LAYER_MEMBERSHIP_BAND_MUST_EXCEED_DECLARED_RESIDUAL_POLICY")
    origin, a0, n0 = map(np.asarray, (feature.origin, feature.axis, feature.start_radial))
    k0 = np.cross(a0, n0)
    phi = math.radians(feature.nominal_sweep_deg)
    sign = math.copysign(1.0, phi)
    n1 = math.cos(phi) * n0 + math.sin(phi) * k0
    radials = (n0, n1)
    rays = (-sign * k0, sign * np.cross(a0, n1))
    local = points - origin
    axial = local @ a0
    numeric_guard = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(points))),
                                                float(np.max(np.abs(origin))))
    in_span = ((axial >= feature.span_mm[0] - numeric_guard)
               & (axial <= feature.span_mm[1] + numeric_guard))
    masks = []
    for radial, ray in zip(radials, rays):
        from_contact = local - feature.nominal_radius_mm * radial
        along = from_contact @ ray
        masks.append(in_span & (along >= feature.flange_window_mm[0])
                     & (along <= feature.flange_window_mm[1])
                     & (np.abs(from_contact @ radial) <= feature.search_margin_mm))
    if np.any(masks[0] & masks[1]):
        raise _Unavailable("AMBIGUOUS_FLANGE_WINDOWS")
    # Diagnostics for BOTH flanges and both weak-gate branches, before and after
    # the declared layer selection, computed without raising. The legacy first
    # failure (flange 0 then flange 1) remains the primary reason.
    diagnostics = dict(material_side_convention=MATERIAL_SIDE_CONVENTION,
        layer_policy=asdict(layer_policy), flanges=[], arc_material_side_filter=None,
        phases_completed=[], failure_phase=None,
        phase_scope="PHASES_LISTED_IN_ORDER_A_MISSING_PHASE_WAS_NOT_EVALUATED_NOT_SILENTLY_PASSED")
    selected_masks = []
    try:
        for role, (mask, radial, ray) in enumerate(zip(masks, radials, rays)):
            crop = points[mask]
            before = _flange_diagnostics(crop, -radial, feature, role=role, ray=ray)
            keep, selection = _select_flange_layer(crop, radial, feature, layer_policy, role)
            kept_mask = np.zeros(len(points), dtype=bool)
            kept_mask[np.flatnonzero(mask)[keep]] = True
            after = _flange_diagnostics(points[kept_mask], -radial, feature, role=role, ray=ray)
            diagnostics["flanges"].append(dict(role=role, crop_rows=int(mask.sum()),
                before_selection=before, layer_selection=selection, after_selection=after))
            selected_masks.append(kept_mask)
    except _NUMERICAL as error:
        raise _attach(error, diagnostics, "FLANGE_DIAGNOSTICS_OR_LAYER_SELECTION_ROLE_%d" % len(diagnostics["flanges"]))
    diagnostics["phases_completed"].append("FLANGE_DIAGNOSTICS_AND_LAYER_SELECTION")
    diagnostics["flange_support"] = dict(
        semantics=("SKIN_MEMBERSHIP_SUBSET_WHEN_TWO_POPULATIONS_ARE_SEPARATED_EXCLUDED_ROWS_PUBLISHED_PER_FLANGE_"
                   "NOT_A_WHOLE_CROP_PLANARITY_CERTIFICATE_RESIDUAL_GATE_APPLIES_AFTER_MEMBERSHIP_"
                   "EXCLUDED_BETWEEN_SKIN_ROWS_ARE_UNASSIGNED_NOT_VERIFIED_CUT_EDGE_IDENTITY_COVERAGE_NOT_ESTABLISHED"),
        per_role=[dict(role=row["role"], crop_rows=row["crop_rows"], kept_rows=row["layer_selection"]["kept_rows"],
                       excluded_opposite_skin_rows=row["layer_selection"].get("dropped_other_skin_rows", 0),
                       excluded_bridging_rows=row["layer_selection"].get("dropped_bridging_rows", 0),
                       outside_sandwich_member_rows=row["layer_selection"].get("outside_sandwich_member_rows", 0),
                       identity=row["layer_selection"]["identity"]) for row in diagnostics["flanges"]])
    masks = selected_masks
    fitted = []
    try:
        for mask, radial in zip(masks, radials):
            fitted.append(_plane(points[mask], -radial, feature))
    except (_Unavailable, *_NUMERICAL) as error:
        raise _attach(error, diagnostics, "PLANE_FIT_ROLE_%d" % len(fitted))
    diagnostics["phases_completed"].append("PLANE_FITS")
    normals = [item[0] for item in fitted]
    try:
        axis = _unit(np.cross(*normals))
        if axis @ a0 < 0:
            axis = -axis
        if axis @ a0 < math.cos(math.radians(feature.max_axis_deviation_deg)):
            raise _Unavailable("MEASURED_AXIS_OUTSIDE_SEARCH_SCOPE")
        sweep = math.degrees(math.atan2(float(axis @ np.cross(*normals)),
                                        float(normals[0] @ normals[1])))
        if math.copysign(1., sweep) != sign or not 20 <= abs(sweep) <= 160:
            raise _Unavailable("MEASURED_SWEEP_OUTSIDE_MODEL_SCOPE")
    except (_Unavailable, *_NUMERICAL) as error:
        raise _attach(error, diagnostics, "AXIS_AND_SWEEP")
    diagnostics["phases_completed"].append("AXIS_AND_SWEEP")

    angle = sign * np.arctan2(local @ k0, local @ n0)
    radial_distance = np.linalg.norm(local - axial[:, None] * a0, axis=1)
    guard = math.radians(feature.arc_guard_deg)
    arc_mask = (in_span & (angle >= guard) & (angle <= abs(phi) - guard)
                & (radial_distance <= feature.nominal_radius_mm + feature.search_margin_mm))
    # Fitted normals point toward the axis (reference -radial). V1 excluded the
    # band |signed| <= plane_exclusion. With layer selection enabled V2 keeps
    # rows strictly on the axis side of BOTH fitted planes (signed > exclusion),
    # i.e. it additionally drops the material side; disabled mode is the V1
    # control and keeps the absolute band rule. Inner-arc rows have signed
    # distance R*(1-cos theta) against these axis-pointing normals.
    candidate_rows = int(arc_mask.sum())
    band_dropped, material_dropped = 0, 0
    for normal, center, _ in fitted:
        signed = (points - center) @ normal
        band = arc_mask & (np.abs(signed) <= feature.plane_exclusion_mm)
        band_dropped += int(band.sum())
        if layer_policy.enabled:
            material = arc_mask & (signed < -feature.plane_exclusion_mm)
            material_dropped += int(material.sum())
            arc_mask &= signed > feature.plane_exclusion_mm
        else:
            arc_mask &= np.abs(signed) > feature.plane_exclusion_mm
    gaps = [row["layer_selection"].get("observed_layer_gap_mm") for row in diagnostics["flanges"]
            if row["layer_selection"].get("identity") == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"]
    heuristic = dict(certificate=False,
        scope="IDEALIZED_NOMINAL_MODEL_HEURISTIC_NOT_AN_OBSERVED_EXCLUSION_CERTIFICATE",
        assumptions=["exact common center and axis", "parallel-offset tangent flanges",
                     "circular skins at the nominal radius and nominal radius plus the observed flange gap",
                     "exact inner-plane selection", "no fitted-plane or model error bound"],
        observed_flange_layer_gaps_mm=gaps, bound_deg=None, status="NOMINAL_HEURISTIC_NOT_EVALUATED_NO_TWO_LAYER_GAP")
    if len(gaps) == 2 and min(gaps) > 0.:
        ratio = (feature.nominal_radius_mm - feature.plane_exclusion_mm) / (feature.nominal_radius_mm + min(gaps))
        if -1. <= ratio <= 1.:
            heuristic["bound_deg"] = 2. * math.degrees(math.acos(ratio))
            heuristic["status"] = ("NOMINAL_HEURISTIC_SWEEP_BELOW_BOUND" if abs(feature.nominal_sweep_deg) < heuristic["bound_deg"]
                                   else "NOMINAL_HEURISTIC_SWEEP_ABOVE_BOUND")
        else:
            heuristic["status"] = "NOMINAL_HEURISTIC_OUT_OF_DOMAIN"
    diagnostics["arc_material_side_filter"] = dict(implementation=ARC_SIDE_FILTER_IMPLEMENTATION,
        enabled=bool(layer_policy.enabled), rule=("SIGNED_GREATER_THAN_PLANE_EXCLUSION_BOTH_PLANES"
            if layer_policy.enabled else "V1_ABSOLUTE_BAND_EXCLUSION"),
        candidate_rows=candidate_rows, plane_band_dropped_rows=band_dropped,
        material_side_dropped_rows=material_dropped, kept_rows=int(arc_mask.sum()),
        outer_arc_isolation_evidence="UNCHANGED_CIRCLE_RESIDUAL_AND_TANGENCY_GATES_AGAINST_SELECTED_PLANES_ONLY",
        outer_arc_nominal_heuristic=heuristic)
    diagnostics["phases_completed"].append("ARC_MATERIAL_SIDE_FILTER")
    arc = points[arc_mask]
    if len(arc) < 12:
        raise _attach(_Unavailable("INSUFFICIENT_ARC_SUPPORT"), diagnostics, "ARC_SUPPORT_COUNT")
    try:
        measured = _measure_arc(points, feature, diagnostics, arc, arc_mask, masks, fitted, normals, axis, sweep,
                                sign, phi, origin, a0, n0, k0)
    except (_Unavailable, *_NUMERICAL) as error:
        raise _attach(error, diagnostics, "ARC_FIT_AND_TANGENCY")
    diagnostics["phases_completed"].append("ARC_FIT_AND_TANGENCY")
    return measured


_NUMERICAL = (np.linalg.LinAlgError, FloatingPointError)


def _attach(error, diagnostics, phase):
    """Retain the already-computed bounded diagnostic prefix on any refusal or numerical failure.

    The prefix is attempted evidence with an explicit failure phase; it is never a surviving
    partial measurement. An error that already carries diagnostics (raised deeper) keeps them.
    """
    if not hasattr(error, "diagnostics"):
        error.diagnostics = diagnostics
    if diagnostics.get("failure_phase") is None:
        diagnostics["failure_phase"] = phase
    return error


def _measure_arc(points, feature, diagnostics, arc, arc_mask, masks, fitted, normals, axis, sweep, sign, phi,
                 origin, a0, n0, k0):
    identities = [row["layer_selection"]["identity"] for row in diagnostics["flanges"]]
    if all(identity == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED" for identity in identities):
        material_side_status = "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    elif any(identity == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED" for identity in identities):
        material_side_status = "MIXED_LAYER_EVIDENCE_IDENTITY_UNRESOLVED"
    elif all(identity == "LAYER_SELECTION_DISABLED" for identity in identities):
        material_side_status = "CONTEXT_ASSUMED_NOT_VERIFIED"
    elif any(identity == "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED" for identity in identities):
        material_side_status = "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED"
    elif any(identity == "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED" for identity in identities):
        material_side_status = "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED"
    else:
        material_side_status = "SINGLE_LAYER_IDENTITY_UNRESOLVED"
    # Connected eligibility is a function of the status through INNER_FACE_ELIGIBLE_STATUSES,
    # which is presently EMPTY: a compatible two-layer selection on both flanges is retained
    # as the side hypothesis `TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED`, but it
    # is not side-isolation evidence, and disabling the selection (the V1 control) supplies
    # none either. Every fit is therefore a generic conditional contact line: no inner
    # radius deviation, no inner-face role, no conformance or count.
    inner_face_claim_eligible = material_side_status in INNER_FACE_ELIGIBLE_STATUSES
    carrier_role = INNER_FACE_CARRIER_ROLE if inner_face_claim_eligible else GENERIC_CARRIER_ROLE
    # Derive the projection direction from scan planes, not CAD. Center is free.
    u = _unit(-normals[0])
    v = np.cross(axis, u)
    q = np.column_stack(((arc - origin) @ u, (arc - origin) @ v))
    center2, radius, residual, condition, min_singular = _circle(q, feature)
    free_radius = radius
    free_center3 = origin + center2[0] * u + center2[1] * v
    free_tangency = [float(normal @ (center - free_center3) + radius)
                     for normal, center, _ in fitted]
    tangent_derivative = None
    if feature.fit_model == "measured_flange_tangent":
        matrix = np.array([[normal @ u, normal @ v] for normal in normals])
        offsets = np.array([normal @ (center - origin) for normal, center, _ in fitted])
        center2, radius, residual, tangent_derivative = _tangent_circle(
            q, matrix, offsets, sweep, radius, feature)
    center3 = origin + center2[0] * u + center2[1] * v
    # Both are INNER outward normals: d - n.C = -R. Absolute distance alone
    # would also admit a circle tangent on the wrong side of a flange.
    tangency = [float(normal @ (center - center3) + radius)
                for normal, center, _ in fitted]
    if max(abs(value) for value in tangency) > feature.max_tangency_error_mm:
        raise _Unavailable("CIRCLE_NOT_SAME_INNER_FACE_TANGENT")
    measured_angle = sign * np.arctan2((q - center2)[:, 1], (q - center2)[:, 0])
    if np.any(measured_angle < -1e-8) or np.any(measured_angle > math.radians(abs(sweep)) + 1e-8):
        raise _Unavailable("ARC_OUTSIDE_MEASURED_CONTACT_SECTOR")
    sweep_coverage = math.degrees(float(np.ptp(measured_angle)))
    if sweep_coverage < feature.min_arc_sweep_deg:
        raise _Unavailable("INSUFFICIENT_ARC_ANGULAR_COVERAGE")
    sampled_axial = (arc - origin) @ axis
    axial_stations = np.unique(sampled_axial)
    if len(axial_stations) < 2:
        raise _Unavailable("INSUFFICIENT_DISTINCT_AXIAL_STATIONS")
    axial_span = [float(sampled_axial.min()), float(sampled_axial.max())]
    if (axial_span[1] - axial_span[0]) < feature.min_axial_span_fraction * (
            feature.span_mm[1] - feature.span_mm[0]):
        raise _Unavailable("INSUFFICIENT_AXIAL_SAMPLE_SPAN")
    return {
        "fit_model": feature.fit_model,
        "free_circle_initializer_radius_mm": free_radius,
        "free_circle_tangency_residual_mm": free_tangency,
        "tangent_radius_rms_derivative": tangent_derivative,
        "measured_sweep_deg": sweep,
        "sweep_deviation_deg": sweep - feature.nominal_sweep_deg,
        "measured_opening_deg": 180. - abs(sweep),
        "measured_radius_mm": radius,
        "radius_identity": ("INNER_FACE_RADIUS_UNVERIFIED" if inner_face_claim_eligible
                            else "OBSERVED_SURFACE_RADIUS_NOT_INNER_RADIUS"),
        "radius_deviation_mm": (radius - feature.nominal_radius_mm) if inner_face_claim_eligible else None,
        "measured_axis_direction": axis.tolist(),
        "measured_axis_point": center3.tolist(),
        "sampled_axial_interval_mm": axial_span,
        "max_axial_sample_gap_mm": float(np.diff(axial_stations).max()),
        "max_angular_sample_gap_deg": math.degrees(float(np.diff(np.unique(measured_angle)).max())),
        "continuous_surface_coverage": "NOT_ESTABLISHED",
        "physical_trim_endpoints": None,
        "physical_extent_deviation_mm": None,
        "flange_support_counts": [int(mask.sum()) for mask in masks],
        "arc_support_count": len(arc),
        "flange_max_residual_mm": [item[2] for item in fitted],
        "circle_max_residual_mm": residual,
        "inner_tangency_residual_mm": tangency,
        "tangency_check_role": ("ENFORCED_CONSTRAINT_NOT_INDEPENDENT_VALIDATION"
                                if feature.fit_model == "measured_flange_tangent"
                                else "FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK"),
        "free_circle_jacobian_condition": condition,
        "free_circle_min_singular_per_sqrt_n": min_singular,
        "sampled_arc_span_deg": sweep_coverage,
        "material_side_status": material_side_status,
        "inner_face_claim_eligible": inner_face_claim_eligible,
        "carrier_role": carrier_role,
        "material_side_convention": MATERIAL_SIDE_CONVENTION,
        "layer_selection_implementation": LAYER_SELECTION_IMPLEMENTATION,
        "arc_side_filter_implementation": ARC_SIDE_FILTER_IMPLEMENTATION,
        "diagnostics": diagnostics,
        "support_row_indices": np.flatnonzero(arc_mask | masks[0] | masks[1]).tolist(),
        "_instance_support": {
            "arc_row_indices": np.flatnonzero(arc_mask).tolist(),
            "flange_row_indices": [np.flatnonzero(mask).tolist() for mask in masks],
        },
    }


def inspect_cad_guided_scan(points, features, *, registration_status="UNTRUSTED", instance_policy=None,
                            layer_policy=None):
    """Measure declared CAD features from an already-aligned single-side scan.

    This explicit offline entry does not register, identify physical events or
    qualify tolerances. Every feature either retains the full conditional result
    or reports UNAVAILABLE, never a prefix measurement or copied nominal value.
    Overlapping support between features clears all competing measurements.
    Explicit finite-instance mode distinguishes exclusive curved evidence from
    reusable planar context; it never assigns physical trim or event identity.
    """
    if not isinstance(features, (tuple, list)) or not 1 <= len(features) <= MAX_FEATURES:
        raise ValueError("require 1..16 explicitly declared CAD features")
    if not all(isinstance(f, CADGuidedFeature) for f in features):
        raise ValueError("features must be CADGuidedFeature values")
    if len({f.feature_id for f in features}) != len(features):
        raise ValueError("duplicate feature identity")
    invalid = None
    if (not isinstance(points, np.ndarray) or points.ndim != 2
            or points.shape[1] != 3 or points.dtype.kind not in "fiu"):
        invalid = "INVALID_POINT_ARRAY"
    elif not 1 <= len(points) <= MAX_POINTS:
        invalid = "POINT_BUDGET_OR_EMPTY_INPUT"
    if invalid is None:
        points = np.array(points, dtype=np.float64, copy=True)
        if not np.isfinite(points).all() or float(np.max(np.abs(points))) > 1e6:
            invalid = "UNSUPPORTED_POINT_NUMERICS"
    if type(registration_status) is not str or registration_status != "DECLARED_ALIGNED":
        invalid = "REGISTRATION_NOT_DECLARED_ALIGNED"
    rows = []
    instance_builder = None
    if instance_policy is not None:
        if __package__:
            from .cad_guided_instances import build_finite_support_instance, SupportPolicy
        else:
            from cad_guided_instances import build_finite_support_instance, SupportPolicy
        if not isinstance(instance_policy, SupportPolicy):
            raise ValueError("instance_policy must be a validated SupportPolicy")
        instance_builder = build_finite_support_instance
    for feature in features:
        row = {"feature_id": feature.feature_id, "status": "UNAVAILABLE",
               "nominal_radius_mm": feature.nominal_radius_mm,
               "nominal_sweep_deg": feature.nominal_sweep_deg,
               "nominal_source": feature.nominal_source,
               "unit": "mm", "surface": "inner", "measurements": None,
               "uncertainty_status": "UNCALIBRATED", "conformance": "INDETERMINATE",
               "physical_completion_state": "UNKNOWN", "reasons": []}
        try:
            if invalid:
                raise _Unavailable(invalid)
            row["measurements"] = _measure(points, feature, layer_policy)
            row["status"] = "MEASURED_CONDITIONAL"
            row["material_side_status"] = row["measurements"]["material_side_status"]
            row["inner_face_claim_eligible"] = row["measurements"]["inner_face_claim_eligible"]
            row["carrier_role"] = row["measurements"]["carrier_role"]
        except (_Unavailable, np.linalg.LinAlgError, FloatingPointError) as error:
            row["reasons"] = [str(error) or type(error).__name__]
            row["diagnostics"] = getattr(error, "diagnostics", None)
        rows.append(row)
    claimed = {}
    competing = set()
    private_support = {}
    for index, row in enumerate(rows):
        if row["measurements"] is not None:
            support = row["measurements"].pop("_instance_support")
            private_support[index] = support
            arc_rows = set(support["arc_row_indices"])
            for raw in row["measurements"].pop("support_row_indices"):
                if raw in claimed:
                    for other, was_arc in claimed[raw]:
                        if instance_builder is None or was_arc or raw in arc_rows:
                            competing.update((index, other))
                claimed.setdefault(raw, []).append((index, raw in arc_rows))
    for index in competing:
        attempted = rows[index]["measurements"].get("diagnostics") if rows[index]["measurements"] else None
        rows[index].update(status="UNAVAILABLE", measurements=None,
                           reasons=["AMBIGUOUS_SHARED_FEATURE_SUPPORT"], attempted_diagnostics=attempted,
                           attempted_diagnostics_scope=("DIAGNOSTICS_OF_A_MEASUREMENT_INVALIDATED_BY_SHARED_SUPPORT_"
                                                        "ATTEMPTED_EVIDENCE_NOT_A_SURVIVING_MEASUREMENT"))
    payload = {"schema_version": 1, "scope": "CAD_GUIDED_OFFLINE_SINGLE_INNER_FACE",
            "status": "COMPLETE" if invalid is None else "UNAVAILABLE",
            "registration_status": registration_status if isinstance(registration_status, str) else "INVALID",
            "feature_identity_status": "DECLARED_NOT_AUTOMATICALLY_VERIFIED",
            "features": rows, "expected_feature_count": len(features),
            "measured_conditional_feature_count": sum(r["status"] == "MEASURED_CONDITIONAL" for r in rows),
            "observed_bend_count": None, "conformance": "INDETERMINATE",
            "production_use_allowed": False}
    if instance_builder is not None:
        instances = []
        for index, row in enumerate(rows):
            if row["measurements"] is None:
                result = {"status": "UNAVAILABLE", "instance": None,
                          "reasons": ["MEASUREMENT_UNAVAILABLE", *row["reasons"]]}
            else:
                result = instance_builder(
                    points, row["measurements"], **private_support[index],
                    feature_id=row["feature_id"], policy=instance_policy)
            if result["status"] == "SUPPORTED":
                result["instance"].update(material_side_status=row["material_side_status"],
                    inner_face_claim_eligible=row["inner_face_claim_eligible"], carrier_role=row["carrier_role"])
            row["finite_support"] = result
            if result["status"] == "SUPPORTED":
                instances.append(result["instance"])
        payload.update(
            finite_support_instances=instances,
            observed_support_instance_count=len(instances),
            instance_extraction_status=("COMPLETE" if len(instances) == len(features)
                                        else "PARTIAL" if instances else "UNAVAILABLE"),
            line_convention="CYLINDER_CENTER_AXIS_SUPPORT",
            endpoint_semantics="SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
            reference_frame="CAD_REFERENCE", unit="mm",
        )
    return payload
