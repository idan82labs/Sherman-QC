Decision

OBSERVED. TRANSITION_MODE_PROBE_V1 is a formal STOP. Across Gold37 it produced 938 global plane pairs, 10,457 multimodal-parent records, 1,994 transition components, two geometry-valid certificates, zero recovered previously unrepresented bends, zero recovered family groups, and zero of the 33 frozen A0 false predictions passing the certificate. The post-lock split is 67 misses at stage 1, three at stage 4, and 70 at stage 7. Neither the C010 nor the C033 certificate corresponds to recovered gold. No birth was emitted and no prediction changed. 

01_DIAGNOSTIC_STOP_REPORT

 

12_C010_CERTIFICATE_CENSUS

 

13_C033_CERTIFICATE_CENSUS

 

10_POSTLOCK_GOLD_ATTRIBUTION

OBSERVED. The current probe is conditioned on the existing global plane table. The pool first calls extract_deterministic_planes(), then forms the plane lattice, and only afterward enriches those merged planes with V2 mode memberships. Each global plane receives one normal, one offset, one 1–99% projected extent, and one synthetic support-component ID. A mode is attached only when its normal is within 15° and its centroid is within the existing scale-dependent plane distance. The diagnostic then enumerates pairs from that table and asks whether V2 modes support their intersection. It never asks whether the V2 graph itself supports a different, local pair of planes. 

04_build_f1_scan_only_candidate…

 

04_build_f1_scan_only_candidate…

 

05_f1_structured_birth_diagnost…

INFERENCE. Select option 1: missing or fragmented incident-plane extraction/coverage as the next test.

INFERENCE. The exact tranche should be:

LOCAL_PLANE_FRAGMENT_PROBE_V1 — construct bounded local plane fragments from the already frozen TRANSITION_ATOM_V2 mode graph, pair only fragments joined by an already serialized CROSS_MODE_LOCAL edge, and apply the unchanged hard transition geometry certificate.

INFERENCE. This is smaller than another spatial representation because the V2 parent, mode, and edge tables remain byte-identical. It changes only how incident planes are represented for a read-only diagnostic.

UNKNOWN. The 67 stage-1 rows do not prove that global plane extraction is the root cause. They prove only that no matching certificate could be formed from the existing global plane-pair enumeration. The proposed probe is the smallest experiment that distinguishes that explanation from failure of the V2 primitive: if local planes constructed from the frozen V2 graph still fail, the plane-table-only hypothesis is falsified.

1. Exact hypothesis and falsifier

INFERENCE. The preregistered hypothesis is:

H-LPF: For a material subset of the 67 stage-1 misses, the frozen V2 modes and SAME_SURFACE/CROSS_MODE_LOCAL edges already contain two locally coherent panel-side surfaces and a transition run, but one or both surfaces are absent, merged incorrectly, offset incorrectly, or represented too globally in the current plane table.

INFERENCE. The counterfactual is evaluated while holding all of the following fixed:

support_parent_table bytes
support_mode_table bytes
support_mode_edges bytes
transition table hashes
working-cloud contract
A0 prediction bytes
all current hard geometry thresholds
the frozen line-matching contract

INFERENCE. H-LPF is falsified if LOCAL_PLANE_FRAGMENT_PROBE_V1 fails any preregistered GO condition. There is no second configuration, no threshold relaxation, and no case-specific retry.

INFERENCE. The probe must not read the typed graph, A0 ownership, the A0 exact solution, object costs, residual costs, duplicate factors, or solver state. Those inputs are irrelevant to a plane-only counterfactual and would confound it with the prohibited ownership and birth tranches.

INFERENCE. A passing result would establish only that a local plane representation deserves a later professor review. It would not establish bend semantics, justify emitted columns, or authorize A1 integration.

2. Smallest new representation
2.1 Inputs

INFERENCE. The per-case probe consumes only:

support_parent_table
support_mode_table
support_mode_edges
plane_table
transition_parent_table_sha256
transition_mode_table_sha256
transition_mode_edge_table_sha256
transition_local_spacing_mm
transition_atom_caps
working_point_budget
transient_sampled_point_count

INFERENCE. The scan and frozen A0 prediction are supplied to the command only so their files can be hashed. The scan is not parsed and the A0 prediction is not read as geometry.

INFERENCE. The existing V2 constraints remain unchanged: at most 200,000 working points, 4,000 parents, 16,000 mode children, four children per parent, 20,000 parent-plus-mode records, and 200,000 mode edges. 

manifest

2.2 Top-level payload

INFERENCE. The exact per-case payload is:

Python
Run
{
    "schema_version": 1,
    "kind": "f1_local_plane_fragment_probe_v1",
    "profile": "LOCAL_PLANE_FRAGMENT_PROBE_V1",

    "research_only": True,
    "inference_safe": True,
    "prediction_mutated": False,
    "birth_columns_enabled": False,
    "ownership_consumed": False,
    "solver_consumed": False,
    "truth_inputs_consumed": False,
    "source_points_consumed": False,
    "source_points_serialized": False,

    "scan_sha256": str,
    "candidate_pool_sha256": str,
    "a0_prediction_sha256": str,

    "input_table_hashes": {
        "parent_table_sha256": str,
        "mode_table_sha256": str,
        "mode_edge_table_sha256": str,
        "global_plane_table_sha256": str,
    },

    "output_table_hashes": {
        "local_plane_fragment_table_sha256": str,
        "local_plane_pair_records_sha256": str,
    },

    "engineering_profile_sha256": str,
    "thresholds": {...},
    "hard_caps": {...},
    "case_counters": {...},

    "local_plane_fragment_table": [...],
    "local_plane_pair_records": [...],
}

INFERENCE. engineering_profile_sha256 hashes the profile name, all formulas and fixed numerical constants, all cap values, and the first-rejection ordering. A numerical or ordering change therefore creates a different experiment rather than silently modifying V1.

2.3 Local plane fragment record

INFERENCE. Every final fragment has exactly these fields:

Python
Run
@dataclass(frozen=True)
class LocalPlaneFragmentRecord:
    local_plane_fragment_id: str

    normal_projective: tuple[float, float, float]
    offset: float
    centroid: tuple[float, float, float]
    tangent_basis: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    centroid_extent_uv: tuple[
        tuple[float, float],
        tuple[float, float],
    ]

    mode_count: int
    parent_count: int
    point_count_sum: int
    normalized_parent_mass_sum: int
    same_surface_edge_count: int

    normal_diameter_deg: float
    max_mode_to_fragment_normal_deg: float
    surface_fit_p90_mm: float

    mode_overflow_parent_count: int
    transition_certificate_allowed: bool

    member_mode_ids_sha256: str
    member_parent_ids_sha256: str
    witness_mode_ids: tuple[str, ...]       # at most 8
    witness_parent_ids: tuple[str, ...]     # at most 8
    witness_ids_truncated: bool

    global_plane_coverage: GlobalPlaneCoverageRecord
    fragment_sha256: str

INFERENCE. The global comparison subrecord is:

Python
Run
@dataclass(frozen=True)
class GlobalPlaneCoverageRecord:
    coverage_state: str
    # NONE, PARTIAL, or FULL

    best_global_plane_id: str | None
    best_mode_match_count: int
    best_mode_coverage_fraction: float
    best_parent_match_count: int
    best_parent_coverage_fraction: float
    best_sum_dimensionless_residual: float | None

    full_coverage_plane_ids: tuple[str, ...]     # at most 4
    partial_coverage_plane_ids: tuple[str, ...]  # at most 4
    match_ids_truncated: bool

INFERENCE. Full member ID arrays are never serialized. The complete sorted sets enter their SHA-256 hashes and canonical IDs; only the first eight canonical witnesses are retained for audit.

2.4 Local plane-pair and component records

INFERENCE. Each local pair record contains:

Python
Run
@dataclass(frozen=True)
class LocalPlanePairRecord:
    local_plane_pair_id: str
    fragment_ids: tuple[str, str]

    plane_angle_deg: float
    intersection_line_hash: str
    intersection_anchor: tuple[float, float, float]
    intersection_axis: tuple[float, float, float]

    transverse_parent_count: int
    left_mode_count: int
    right_mode_count: int
    cross_mode_rung_count: int
    component_count: int

    global_pair_comparison: GlobalPairComparisonRecord
    pair_rejection_counts: dict[str, int]
    components: tuple[LocalTransitionComponentRecord, ...]

    pair_sha256: str

INFERENCE. The global pair comparison contains:

Python
Run
@dataclass(frozen=True)
class GlobalPairComparisonRecord:
    state: str
    # NO_DISTINCT_FULL_GLOBAL_PAIR
    # or FULL_DISTINCT_GLOBAL_PAIR

    global_plane_ids: tuple[str, ...]  # empty or exactly two
    total_dimensionless_residual: float | None

INFERENCE. Each component has:

Python
Run
@dataclass(frozen=True)
class LocalTransitionComponentRecord:
    local_transition_component_id: str

    support_parent_count: int
    support_mode_count: int
    normalized_parent_mass: int

    support_parent_ids_sha256: str
    support_mode_ids_sha256: str
    witness_parent_ids: tuple[str, ...]  # at most 8
    witness_mode_ids: tuple[str, ...]    # at most 8
    witness_ids_truncated: bool

    intersection_span_endpoints: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    occupied_bin_count: int
    left_side_bin_count: int
    right_side_bin_count: int
    transition_rung_bin_count: int
    longest_consecutive_occupied_run: int
    maximum_empty_bin_run: int
    internal_branch_count: int

    tls_axis_angle_deg: float
    line_p90_mm: float
    certified_three_bin_window_count: int
    mode_overflow_parent_count: int

    geometry_gate_depth: int
    geometry_gates: dict[str, bool]
    first_rejection_code: str | None
    geometry_certificate_valid: bool

    global_pair_line_state: str
    # UNAVAILABLE, EQUIVALENT, or NON_EQUIVALENT

    global_pair_axis_angle_deg: float | None
    global_pair_transverse_offset_mm: float | None

    component_sha256: str
3. Deterministic construction
3.1 Fixed engineering profile

INFERENCE. No existing geometric threshold is relaxed. The profile reuses the current V2 and transition-certificate values:

Python
Run
LOCAL_PLANE_FRAGMENT_PROFILE = "LOCAL_PLANE_FRAGMENT_PROBE_V1"

PLANE_NORMAL_DIAMETER_DEG = 15.0

plane_fit_tolerance_mm = min(
    3.0,
    max(1.25, 1.5 * local_spacing_mm),
)

PLANE_PAIR_MIN_ANGLE_DEG = 20.0
PLANE_PAIR_MAX_PROJECTIVE_ANGLE_DEG = 90.0

line_tube_tolerance_mm = min(
    3.0,
    max(1.5, 1.25 * local_spacing_mm),
)

axial_bin_width_mm = max(
    3.0,
    2.0 * local_spacing_mm,
)

MINIMUM_PARENT_COUNT = 3
MINIMUM_AXIAL_SPAN_BINS = 2
MINIMUM_OCCUPIED_BINS = 3
MINIMUM_SIDE_BINS = 3
MINIMUM_TRANSITION_RUNG_BINS = 2
MINIMUM_CONSECUTIVE_RUN = 3
MAXIMUM_EMPTY_BIN_RUN = 1
MAXIMUM_INTERNAL_BRANCH_COUNT = 0
MAXIMUM_TLS_AXIS_ANGLE_DEG = 4.0

OBSERVED. These are the same parent, span, bin, side, rung, gap, branch, TLS, line-residual, overflow, and three-bin-window conditions already used by the frozen transition-mode diagnostic. 

05_f1_structured_birth_diagnost…

 

09_PROBE_CENSUS_BUNDLE

3.2 Canonical fitting helpers

INFERENCE. Plane normals are projective. Flipping every input normal must leave every output byte unchanged.

Python
Run
def canonical_projective_normal(value: Array3) -> Array3:
    normal = unit(value)
    for component in normal:
        if abs(component) > 1e-12:
            if component < 0.0:
                normal = -normal
            break
    normal[abs(normal) <= 1e-12] = 0.0
    return normal

INFERENCE. Fragment normals use the existing point-count-weighted projective moment:

Python
Run
def fit_fragment_normal(modes: Sequence[Mode]) -> Array3:
    moment = zeros((3, 3))
    for mode in sorted(modes, key=lambda m: m.mode_atom_id):
        normal = unit(mode.normal_projective)
        weight = max(1, mode.point_count)
        moment += weight * outer(normal, normal)

    eigenvalues, eigenvectors = eigh(moment)
    return canonical_projective_normal(eigenvectors[:, -1])

INFERENCE. The offset is the deterministic weighted lower median of −n⋅c
m
	​

. Sorting is by (offset_value, mode_atom_id). The target cumulative weight is ⌈W/2⌉.

INFERENCE. For mode m, the conservative fragment-plane residual is:

e
m
	​

=∣n
F
	​

⋅c
m
	​

+d
F
	​

∣+surface_fit_p90_mm(m)

INFERENCE. surface_fit_p90_mm for the fragment is the deterministic weighted 90th percentile of e
m
	​

, using point count as the integer weight and (residual, mode_atom_id) as the ordering.

INFERENCE. normal_diameter_deg is the maximum projective angle over every pair of member modes. A fragment is geometrically coherent only when:

normal_diameter_deg <= 15 degrees
surface_fit_p90_mm <= plane_fit_tolerance_mm
3.3 Local fragment partition

INFERENCE. The fragment partition uses only the already serialized SAME_SURFACE edges. It does not invent an edge from centroid proximity and does not repair missing connectivity.

Python
Run
def build_local_plane_fragments(
    modes: Sequence[Mode],
    edges: Sequence[ModeEdge],
    parents: Sequence[Parent],
    *,
    local_spacing_mm: float,
) -> list[LocalPlaneFragment]:
    same_surface_edges = sorted(
        (
            edge for edge in edges
            if edge.edge_kind == "SAME_SURFACE"
        ),
        key=lambda edge: (
            round(edge.normal_residual, 12),
            round(edge.plane_residual, 12),
            round(edge.spatial_residual, 12),
            edge.mode_edge_id,
        ),
    )

    # One deterministic union-find component per mode initially.
    components = {
        mode.mode_atom_id: Component({mode.mode_atom_id})
        for mode in sorted(modes, key=lambda item: item.mode_atom_id)
    }

    for edge in same_surface_edges:
        left_root = find(edge.mode_ids[0])
        right_root = find(edge.mode_ids[1])

        if left_root == right_root:
            continue

        proposed_mode_ids = sorted(
            components[left_root].mode_ids
            | components[right_root].mode_ids
        )

        fit = fit_local_plane(proposed_mode_ids)

        if fit.normal_diameter_deg > 15.0:
            counters["plane_union_reject_normal_diameter_count"] += 1
            continue

        if fit.surface_fit_p90_mm > plane_fit_tolerance_mm:
            counters["plane_union_reject_fit_count"] += 1
            continue

        canonical_root = min(left_root, right_root)
        union_into(canonical_root, proposed_mode_ids)
        counters["plane_union_accept_count"] += 1

    fragments = []
    for mode_ids in canonical_final_components():
        fragments.append(build_fragment_record(mode_ids))

    return sorted(
        fragments,
        key=lambda fragment: fragment.local_plane_fragment_id,
    )

INFERENCE. Each mode belongs to exactly one final fragment. A fragment is therefore a partition of existing evidence, not an additional copy of that evidence.

INFERENCE. Singleton fragments are retained. No new minimum-fragment-size threshold is introduced. A singleton cannot become a hard bend certificate unless the unchanged three-parent and three-bin conditions are subsequently met.

INFERENCE. A fragment containing any parent with mode_overflow=true or transition_certificate_allowed=false is serialized for diagnosis but has transition_certificate_allowed=false.

3.4 Canonical IDs and hashes

INFERENCE. IDs are based on full canonical membership, not geometry rounding alone:

Python
Run
local_plane_fragment_id = (
    "LPF_"
    + sha256_json([
        "LOCAL_PLANE_FRAGMENT_PROBE_V1",
        sorted(full_member_mode_ids),
    ])[:16]
)

local_plane_pair_id = (
    "LPP_"
    + sha256_json([
        sorted((left_fragment_id, right_fragment_id)),
    ])[:16]
)

local_transition_component_id = (
    "LTC_"
    + sha256_json([
        local_plane_pair_id,
        sorted(full_support_parent_ids),
        sorted(full_support_mode_ids),
    ])[:16]
)

INFERENCE. Fragment and component record hashes are computed after rounding:

normals, axes, tangent vectors: 12 decimal places
centroids, offsets, endpoints, extents: 6 decimal places
dimensionless residuals: 8 decimal places

INFERENCE. The table hashes use canonical JSON with sort_keys=True, compact separators, ASCII encoding, and records sorted by canonical ID.

3.5 Local pair enumeration

INFERENCE. Two fragments become a pair only if at least one existing CROSS_MODE_LOCAL edge has one endpoint in each fragment. All other fragment combinations are prohibited.

Python
Run
cross_edges = [
    edge for edge in edges
    if edge.edge_kind == "CROSS_MODE_LOCAL"
]

pair_to_cross_edges = defaultdict(list)

for edge in sorted(cross_edges, key=lambda item: item.mode_edge_id):
    left_fragment = fragment_of_mode[edge.mode_ids[0]]
    right_fragment = fragment_of_mode[edge.mode_ids[1]]

    if left_fragment == right_fragment:
        continue

    pair = tuple(sorted((left_fragment, right_fragment)))
    pair_to_cross_edges[pair].append(edge)

INFERENCE. The pair normal angle must lie in the existing projective interval [20
∘
,90
∘
]. There is no pair-angle sweep.

INFERENCE. For an eligible pair:

solve the intersection of the two fitted local planes;

retain parents whose existing parent centroid lies within the unchanged line tube;

retain left and right fragment modes belonging to those parents;

retain only existing SAME_SURFACE edges internal to either fragment and existing CROSS_MODE_LOCAL edges joining the two fragments;

compute connected components in that restricted graph.

INFERENCE. This restriction is the key difference from the failed probe. It does not allow every mode compatible with two global planes into one component. It preserves the spatially connected local plane patches from which the pair was fitted.

Python
Run
def build_local_pair_components(pair, fragments, modes, parents, edges):
    left, right = fragments[pair[0]], fragments[pair[1]]
    anchor, axis, angle = intersect_planes(left, right)

    transverse_parent_ids = {
        parent.parent_atom_id
        for parent in parents
        if line_distance(parent.centroid, anchor, axis)
        <= line_tube_tolerance_mm
    }

    left_modes = {
        mode_id
        for mode_id in left.full_mode_ids
        if parent_of_mode[mode_id] in transverse_parent_ids
    }
    right_modes = {
        mode_id
        for mode_id in right.full_mode_ids
        if parent_of_mode[mode_id] in transverse_parent_ids
    }

    supported_modes = left_modes | right_modes

    restricted_edges = [
        edge
        for edge in edges
        if set(edge.mode_ids) <= supported_modes
        and (
            (
                edge.edge_kind == "SAME_SURFACE"
                and both_modes_belong_to_same_pair_fragment(edge)
            )
            or (
                edge.edge_kind == "CROSS_MODE_LOCAL"
                and edge_joins_left_and_right(edge)
            )
        )
    ]

    return connected_components(supported_modes, restricted_edges)
3.6 Component geometry

INFERENCE. Component calculations are identical to the current probe wherever the representation has not changed:

physical support unit: unique parent
axial projection origin: component minimum projection
bin width: max(3 mm, 2s)
owned/occupied bin: at least one component parent
left/right bin: parent contains a component mode from that fragment
rung bin: parent participates in a retained cross-mode edge
TLS points: parent centroids
TLS weights: normalized parent mass
line residual: P90 parent-centroid distance to TLS line
branch count: non-end axial bins with degree greater than two

INFERENCE. The hard gates and their first-failure order are:

Python
Run
REJECTION_ORDER = (
    ("minimum_parent_count", "COMPONENT_TOO_SMALL"),
    ("minimum_axial_span", "INSUFFICIENT_AXIAL_SPAN"),
    ("minimum_occupied_bins", "INSUFFICIENT_OCCUPIED_BINS"),
    ("minimum_left_side_bins", "INSUFFICIENT_LEFT_SIDE_BINS"),
    ("minimum_right_side_bins", "INSUFFICIENT_RIGHT_SIDE_BINS"),
    ("minimum_transition_rung_bins", "INSUFFICIENT_TRANSITION_RUNG_BINS"),
    ("minimum_consecutive_occupied_run", "INSUFFICIENT_CONSECUTIVE_BIN_RUN"),
    ("maximum_empty_bin_run", "EXCESSIVE_EMPTY_BIN_RUN"),
    ("no_internal_branching", "INTERNAL_BRANCHING"),
    ("tls_axis", "TLS_AXIS_MISMATCH"),
    ("line_residual", "LINE_RESIDUAL_EXCEEDED"),
    (
        "transition_certificate_complete",
        "MODE_OVERFLOW_CERTIFICATE_DISABLED",
    ),
    (
        "certified_three_bin_window",
        "NO_CERTIFIED_THREE_BIN_WINDOW",
    ),
)

INFERENCE. A valid certificate is exactly:

Python
Run
geometry_certificate_valid = all(geometry_gates.values())

INFERENCE. Ownership availability, residual windows, duplicates, object cost, reduced cost, and MAP status are not present in this profile.

4. Boundedness and inference-safe census
4.1 Hard per-case caps

INFERENCE. The exact caps are:

Python
Run
MAX_INPUT_WORKING_POINTS = 200_000

MAX_TRANSITION_PARENTS = 4_000
MAX_TRANSITION_MODES = 16_000
MAX_TRANSITION_MODE_EDGES = 200_000
MAX_MODES_PER_PARENT = 4

MAX_LOCAL_PLANE_FRAGMENTS = 16_000
MAX_LOCAL_PLANE_PAIRS = 32_768
MAX_LOCAL_TRANSITION_COMPONENTS = 65_536
MAX_COMPONENTS_PER_LOCAL_PAIR = 4_096

MAX_FRAGMENT_WITNESS_MODE_IDS = 8
MAX_FRAGMENT_WITNESS_PARENT_IDS = 8
MAX_COMPONENT_WITNESS_MODE_IDS = 8
MAX_COMPONENT_WITNESS_PARENT_IDS = 8
MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT = 4

INFERENCE. A structural cap breach fails the case. No pair, fragment, or component is retained by “best score” when a cap is exceeded.

INFERENCE. Only witness arrays may be truncated. Every such record must set its explicit truncation Boolean. The complete member set must still enter the record hash.

4.2 Exact case counters

INFERENCE. case_counters contains:

Python
Run
{
    # Frozen inputs
    "input_parent_count": int,
    "input_mode_count": int,
    "input_same_surface_edge_count": int,
    "input_cross_mode_edge_count": int,
    "input_global_plane_count": int,
    "inherited_working_point_count": int,

    # Local plane partition
    "plane_union_attempt_count": int,
    "plane_union_accept_count": int,
    "plane_union_reject_normal_diameter_count": int,
    "plane_union_reject_fit_count": int,
    "local_plane_fragment_count": int,
    "singleton_fragment_count": int,
    "overflow_ineligible_fragment_count": int,

    # Global-plane comparison
    "fragment_no_global_match_count": int,
    "fragment_partial_global_match_count": int,
    "fragment_full_global_match_count": int,

    # Pair and component construction
    "cross_mode_seed_edge_count": int,
    "local_plane_pair_count": int,
    "angle_eligible_local_pair_count": int,
    "local_transition_component_count": int,
    "geometry_certificate_count": int,

    # Pair-level failure
    "pair_no_transverse_parent_count": int,
    "pair_missing_left_mode_count": int,
    "pair_missing_right_mode_count": int,
    "pair_no_cross_mode_rung_count": int,

    # Global-pair interpretation
    "certificate_no_distinct_full_global_pair_count": int,
    "certificate_global_pair_line_non_equivalent_count": int,
    "certificate_global_pair_line_equivalent_count": int,

    # Ordered component funnel
    "components_after_minimum_parent_count": int,
    "components_after_minimum_axial_span": int,
    "components_after_minimum_occupied_bins": int,
    "components_after_minimum_left_side_bins": int,
    "components_after_minimum_right_side_bins": int,
    "components_after_minimum_transition_rung_bins": int,
    "components_after_minimum_consecutive_run": int,
    "components_after_maximum_empty_run": int,
    "components_after_no_internal_branching": int,
    "components_after_tls_axis": int,
    "components_after_line_residual": int,
    "components_after_transition_certificate_complete": int,
    "components_after_certified_three_bin_window": int,

    "first_rejection_counts": dict[str, int],
}
4.3 Forbidden data

INFERENCE. The serializer recursively rejects these keys, case-insensitively:

point_indices
source_indices
source_point_indices
vertex_indices
points
source_points
point_coordinates
coordinates
bbox_points

gold
manual_truth
expected_count
requested_count
cad
drawing
filename
scan_path
part_id
part_key
family_id
split_group_id

atom_owners
parent_owners
mode_side_assignments
solver
objective
reduced_cost
birth_columns

INFERENCE. A case_token may appear in the external batch manifest used to associate output files with evaluation cases. It must not appear in the per-case inference payload or influence any hash, ordering, fit, pair, or certificate.

INFERENCE. The new diagnostic command accepts no --gold, --cad, --graph, --a0-solution, --expected-count, or --family argument.

5. Global-plane comparison

INFERENCE. The probe must show whether a recovered local certificate genuinely adds a missing plane representation or merely reconstructs a pair that the current global table already contains.

INFERENCE. For each local fragment and every global plane, recompute the current mode-membership rule exactly:

r
n
	​

(m,g)=
15
∘
∠
proj
	​

(n
m
	​

,n
g
	​

)
	​

r
d
	​

(m,g)=
min(3,max(1.25,1.5s))
∣n
g
	​

⋅c
m
	​

+d
g
	​

∣
	​

mode matches global plane iff r_n <= 1 and r_d <= 1

INFERENCE. The best global plane is selected by:

Python
Run
(
    -matching_mode_count,
    sum_of_matching_mode_dimensionless_residuals,
    global_plane_id,
)

INFERENCE. Coverage states are exact rather than thresholded:

NONE     best match count == 0
PARTIAL  0 < best match count < fragment mode count
FULL     best match count == fragment mode count

INFERENCE. A local pair has a FULL_DISTINCT_GLOBAL_PAIR only when its two fragments can be assigned to two distinct global planes with full coverage. Among multiple assignments, select the one minimizing total dimensionless residual and then the sorted plane-ID tuple.

INFERENCE. For each component with a full global pair, compare the global and local infinite intersection lines using the frozen 5° axis and 4 mm transverse tolerances:

EQUIVALENT      axis angle <= 5° and symmetric offset <= 4 mm
NON_EQUIVALENT  otherwise

INFERENCE. These comparison fields are diagnostic only. They never affect local pair construction or hard certificate validity.

6. Exact post-lock procedure
6.1 Freeze boundary

INFERENCE. The execution order is mandatory:

verify every candidate-pool table hash;

hash the scan without parsing it;

hash the frozen A0 prediction without reading its lines;

generate all 37 inference-safe payloads;

create a manifest containing every input and output SHA-256;

create and hash the census bundle;

complete replay and memory checks;

verify that all A0 prediction hashes still match;

only then open the frozen gold attribution, grouped split, and frozen A0 false-prediction set.

INFERENCE. A post-lock script must refuse to run when the bundle hash, grouped-split hash, prior attribution hash, or A0 prediction hash differs from the preregistered values.

6.2 Gold population

INFERENCE. The primary population is exactly the 67 rows already frozen as first_failed_stage == 1. The three stage-4 misses and 70 stage-7 misses may be reported as non-primary diagnostics but cannot count toward the GO threshold. The frozen post-lock artifact must remain the source of this partition. 

10_POSTLOCK_GOLD_ATTRIBUTION

INFERENCE. Family grouping uses the already frozen split_group_id and grouped_split_sha256 only after lock. It is never copied into an inference payload.

6.3 One-to-one recovery matching

INFERENCE. Build a bipartite graph, separately per case:

left vertices:  frozen stage-1 gold lines
right vertices: geometry-valid local certificates

INFERENCE. An edge exists only when the frozen line contract passes:

axis angle <= 5°
symmetric transverse offset <= 4 mm
gold span overlap >= 0.60
certificate span overlap >= 0.60

INFERENCE. Solve matching exactly in three stages:

maximize match cardinality;

among maximum-cardinality matchings, minimize total integer geometric cost;

among equal-cost matchings, choose the lexicographically smallest ordered (gold_id, certificate_id) assignment.

INFERENCE. The integer edge cost is:

10
6
(
5
∘
θ
	​

+
4 mm
d
⊥
	​

	​

+1−o
gold
	​

+1−o
certificate
	​

)

rounded to the nearest integer. The tie ordinal is applied only after the primary geometric cost.

6.4 New failure attribution

INFERENCE. A stage-1 gold line not matched to a valid certificate is compared against all frozen local component spans, including invalid components.

INFERENCE. When several components match the line contract, select by:

Python
Run
(
    -geometry_gate_depth,  # furthest-progressed component first
    integer_geometric_match_cost,
    local_transition_component_id,
)

INFERENCE. Its new post-lock reason is then:

NO_MATCHING_LOCAL_COMPONENT
or the selected component's first_rejection_code

INFERENCE. This creates a new, falsifiable distribution such as:

no local plane pair
component too small
one side missing
rungs missing
branching
line fit
overflow
valid local certificate

without allowing gold to affect construction.

6.5 Frozen A0 false predictions

INFERENCE. Independently build a second bipartite matching:

left vertices:  the frozen 33 A0 false-prediction segments
right vertices: geometry-valid local certificates

INFERENCE. Use the same line contract and exact three-stage matching. The false-certificate metric is the number of distinct frozen A0 false predictions matched, not the number of certificate records.

INFERENCE. A certificate matched to a stage-1 gold line is not exempt from this test. If it also geometrically certifies a frozen A0 false line in the independent matching, that false prediction counts against the ceiling.

6.6 Post-lock cause categories

INFERENCE. Each recovered gold object receives exactly one cause category:

LOCAL_FRAGMENT_NO_GLOBAL_PLANE
    At least one local fragment has global coverage state NONE.

LOCAL_FRAGMENT_PARTIAL_GLOBAL_COVERAGE
    No fragment is NONE, but at least one is PARTIAL.

FULL_GLOBAL_PAIR_LINE_NON_EQUIVALENT
    Both fragments have a distinct full global pair, but its line differs
    from the local line under 5° / 4 mm.

FULL_GLOBAL_PAIR_LINE_EQUIVALENT_RUN_ABSENT
    A full equivalent global pair exists, but the global-pair probe did not
    expose the matching valid run.

INFERENCE. These categories determine what a passing packet says about the plane representation. They do not alter the GO arithmetic.

7. Preregistered GO and STOP
GO

INFERENCE. LOCAL_PLANE_FRAGMENT_PROBE_V1 passes only when all conditions hold:

Recovered previously unrepresented stage-1 gold objects >= 14

Independent frozen split groups containing at least one recovery >= 8

Frozen A0 false predictions passing a valid certificate <= 8
    # floor(0.25 * 33); no rounding upward to 9

Successful inference-safe jobs = 37 / 37

Prediction changes = 0
Birth columns emitted = 0
Ownership inputs consumed = 0
Solver inputs consumed = 0

All parent/mode/edge input hashes unchanged
All A0 prediction hashes unchanged

No structural cap breach
No forbidden serialized field
No source point or point-index serialization

C001, the maximum-mode case, and the maximum-edge case each replay
three times with byte-identical output

Every measured probe process remains below 16 GiB peak RSS

OBSERVED. The 14-object, eight-group, and 25%-of-33 false-certificate boundaries are the already frozen professor gate; the failed V1 probe met only the false-certificate condition. They should be preserved rather than relaxed. 

10_POSTLOCK_GOLD_ATTRIBUTION

STOP

INFERENCE. The experiment is a formal STOP when any GO condition fails.

INFERENCE. In particular, the plane-table-only hypothesis is falsified when:

recovered stage-1 objects < 14
or recovered groups < 8
or matched frozen A0 false predictions > 8

INFERENCE. A STOP result must not be followed by changing the 15° plane diameter, plane-distance formula, 20° pair angle, tube size, bin width, three-bin minima, or line contract within this tranche.

UNKNOWN. The 33 frozen A0 false predictions do not constitute an exhaustive annotation of every possible non-bend local certificate. Certificates matching neither gold nor an A0 false prediction remain semantically unadjudicated. This is why even a GO permits only another professor packet, not ownership integration or runtime use.

What a pass permits

INFERENCE. A pass permits exactly one next action:

Submit a frozen LOCAL_PLANE_FRAGMENT_PROBE_V1 GO professor packet asking whether a research-only local-plane representation arm is justified.

INFERENCE. That packet must contain the complete manifest, table hashes, per-family recovery list, false-certificate matches, new first-failure distribution, global-plane cause categories, memory profiles, and replay hashes.

INFERENCE. A pass does not itself authorize A1 parent/child ownership, any birth family, learned potentials, or runtime/main promotion.

8. Tests
8.1 Unit fixtures

INFERENCE. Implement these hand-auditable fixtures:

Fixture	Required result
three_bin_wedge_without_global_planes	Two local fragments, one valid local certificate, global-pair state unavailable
fragmented_global_panel	Spatially disconnected coplanar mode chains remain separate local fragments
flat_panel_free_edge	One local fragment, no cross-mode pair, zero certificates
same_parent_same_normal_children	Separate same-normal children do not manufacture a transition pair
parallel_double_fold	Two disconnected local transition components, never one merged run
normal_drift_chain	Aggregate 15° diameter prevents chain-drift from becoming one plane
short_return_two_bins	Local pair exists but fails the unchanged three-bin certificate
missing_one_side	Pair fails with the exact missing-side rejection
mode_overflow_transition	Geometry may be present, but the hard certificate is disabled
full_global_pair_equivalent	Global comparison reports full distinct/equivalent
full_global_pair_wrong_offset	Global comparison reports full distinct/non-equivalent
normal_sign_flip	Serialized payload is byte-identical
mode_and_edge_permutation	Serialized payload is byte-identical
8.2 Property tests

INFERENCE. The following invariants must hold over generated compact fixtures:

every input mode belongs to exactly one local fragment

every multi-mode fragment is connected through input SAME_SURFACE edges

every fragment has normal_diameter_deg <= 15

every fragment has surface_fit_p90_mm <= plane_fit_tolerance_mm

every local pair is witnessed by at least one input CROSS_MODE_LOCAL edge

no local pair contains the same fragment twice

every component mode belongs to one of its pair's two fragments

every component edge is an existing input edge

a parent contributes once to parent count, mass, and axial occupancy

a parent with multiple children cannot multiply physical mass

every valid certificate passes every geometry gate

every invalid component has exactly one first rejection code

canonical IDs are invariant to input record ordering

canonical bytes are invariant to normal sign flips

global-plane comparison cannot affect certificate validity

no gold, family, filename, count, ownership, solver, or birth field can enter
an inference hash
8.3 Schema tests

INFERENCE. Schema tests must verify:

all required top-level fields exist
all profile and enum values are exact
all SHA-256 fields are lowercase 64-character hex
every table hash recomputes exactly
all IDs are unique
all references resolve
all witness arrays respect caps
every truncated witness array sets its flag
structural tables are never silently truncated
forbidden-key rejection is recursive
NaN, Infinity, negative counts, and malformed vectors fail closed
8.4 Post-lock tests

INFERENCE. Post-lock tests must verify:

the inference CLI has no truth-bearing option

gold attribution fails before bundle lock

a bundle-hash mismatch fails closed

only frozen stage-1 rows count as recovered

maximum-cardinality matching precedes cost minimization

matching tie resolution is deterministic

the same certificate cannot recover two gold lines in one matching

the same certificate cannot certify two A0 false lines in one matching

family counts use the frozen grouped split only

14 / 8 / 8 passes
13 / 8 / 8 stops
14 / 7 / 8 stops
14 / 8 / 9 stops
8.5 Memory and replay

INFERENCE. Measure at least:

C001                  inherited 200,000-point-bound case
maximum-mode case     largest support_mode_table in the frozen manifest
maximum-edge case     largest support_mode_edges table in the frozen manifest

INFERENCE. Each receives three isolated runs. Require identical canonical JSON and SHA-256 across all runs.

INFERENCE. Record:

wall_seconds
peak_rss_bytes
input JSON bytes
fragment count
pair count
component count
certificate count
serialized output bytes

INFERENCE. Peak RSS at or above 16 GiB is a hard failure.

9. Ordered patch plan
Frozen existing files

INFERENCE. backend/patch_graph_latent_decomposition.py receives no production change. Its V2 parent, mode, edge construction, IDs, caps, and serialized table hashes are the frozen input being tested.

INFERENCE. scripts/build_f1_scan_only_candidate_pool.py receives no production change. The existing schema-3 candidate pools and global plane tables are reused byte-for-byte.

INFERENCE. backend/f1_structured_birth_diagnostics.py receives no production change. The new experiment must not be implemented inside a birth module or call ownership, MDL, or duplicate logic.

INFERENCE. scripts/diagnose_f1_transition_mode_probe.py receives no production change. Its formal STOP outputs remain reproducible historical evidence.

New files

INFERENCE. Add files in this order:

backend/f1_local_plane_fragment_probe.py

Implements table validation, canonical projective fitting, deterministic fragment partitioning, local pair enumeration, restricted component construction, global-plane comparison, hard geometry gates, counters, IDs, hashes, and compact serialization.

scripts/diagnose_f1_local_plane_fragment_probe.py

Accepts only:

--scan
--candidate-pool
--a0-prediction
--output

It hashes --scan and --a0-prediction, parses only the candidate pool, and writes one canonical per-case payload.

scripts/run_f1_local_plane_fragment_probe_batch.py

Runs the 37 cases in isolated processes, records input/output hashes and process status, and fails the batch on any case failure or cap breach.

scripts/build_f1_local_plane_fragment_probe_bundle.py

Validates all 37 payloads, recomputes their table hashes, verifies prediction immutability, and creates the locked inference-safe bundle.

scripts/attribute_f1_local_plane_fragment_probe_gold.py

Opens gold only after bundle lock, performs the exact one-to-one stage-1 and A0-false matching, assigns new failure codes and global-plane cause categories, and evaluates the preregistered gate.

tests/test_f1_local_plane_fragment_probe.py

Contains the unit fixtures, property tests, canonicalization tests, cap tests, and forbidden-data tests.

tests/test_build_f1_local_plane_fragment_probe_bundle.py

Covers manifest completeness, hash validation, immutability, missing-case failure, and replay aggregation.

tests/test_attribute_f1_local_plane_fragment_probe_gold.py

Covers lock ordering, exact matching, stage-1 restriction, family grouping, false-certificate arithmetic, and GO/STOP boundaries.

tests/test_f1_local_plane_fragment_probe_memory.py

Verifies bounded record growth and validates the emitted memory-profile schema. The full case measurements remain separate process-level evidence rather than ordinary unit-test assertions.

Mandatory stop milestone

INFERENCE. The single milestone is:

LPF-M1 — local plane fragments, local fragment-pair components, inference-safe census, bundle locking, and post-lock matching are complete; all existing prediction, ownership, birth, learning, and runtime paths remain untouched.

INFERENCE. At LPF-M1:

run all focused tests;

complete the three-case memory and replay suite;

run all 37 Gold37 inference-safe cases exactly once;

freeze the per-case outputs, manifest, and bundle hashes;

open gold;

produce the post-lock recovery and false-certificate evaluation;

apply the exact 14-object, eight-group, and eight-false-prediction gate;

stop all implementation work.

INFERENCE. Only an LPF-M1 GO result permits the next professor packet. An LPF-M1 STOP terminates this plane-extraction tranche without threshold rescue. Runtime/main promotion remains prohibited in either outcome.
