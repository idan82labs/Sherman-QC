Disposition

OBSERVED. The package itself freezes Gold37 A0, names transition-side atomization as the next research decision, and explicitly leaves runtime promotion unauthorized. The operational solver claims are credible for the encoded model, but the Gold37 line result remains 42 TP, 33 FP, and 140 FN, with only 1/37 exact. Of the 140 misses, 70 have no incidence-complete bend candidate and 70 have a represented candidate that loses selection. BIRTH_V1 produces zero columns before the MDL test, not because its reduced-cost bound is too expensive. 

manifest

 

01_GOLD37_CHECKPOINT

 

05_BIRTH_V1_PRECHECK

A code-level correction is also required before V2 work: source coordinates are not cached, but the stronger claim that atom point indices remain transient is false. SurfaceAtom serializes point_indices, and the research cache writes asdict(atom_graph) while only rejecting a top-level bbox_points field. The compact candidate pool correctly strips the indices later, but research_inputs.json can contain them. 

13_patch_graph_latent_decomposi…

 

13_patch_graph_latent_decomposi…

 

13_patch_graph_latent_decomposi…

 

14_build_f1_scan_only_candidate…

INFERENCE. The smallest defensible next representation is a physical coarse parent voxel with multiple projective-normal mode children, supplemented by bounded fine-cell topology used during construction. Panel-interior, transition-strip, and exposed-boundary status should be derived certificates, not mutually exclusive atom kinds.

UNKNOWN. It is not yet established that preserving local normal modes will recover enough of the 70 representation misses. A diagnostic-only probe below makes that choice falsifiable before the complete A1 solver patch.

1. Localize the zero-birth failure before redesign
1.1 Mutually exclusive attribution rule

OBSERVED. The existing stage census records aggregate birth rejection, but it cannot attribute a specific gold miss because it contains no gold-to-plane-pair, gold-to-atom, or gold-to-component links. It does establish that the dominant current failure is connectivity after aggressive gates: 2,960 normal-compatible trials become 390 atom-axis-compatible trials, forming 312 components, of which 311 fail the six-atom minimum. The only remaining component fails occupied-bin coverage. 

07_BIRTH_V1_STAGE_CENSUS

Use the following partition before assigning stages:

Python
Run
def attribute_gold_miss(
    gold: GoldLine,
    *,
    frozen_paths: Sequence[PathRecord],
    frozen_solution: SolutionRecord,
    frozen_census: TransitionDiagnosticCensus,
) -> FailureAttribution:
    # This is a partitioning gate, not an eighth failure stage.
    matching_paths = match_under_frozen_line_contract(
        gold,
        [
            path for path in frozen_paths
            if path.incidence_complete
            and path.label_eligibility["BEND_STRAIGHT_COUNTED"]
        ],
    )

    if matching_paths:
        # The object was represented. Birth diagnosis is not applicable.
        return stage_7_map_rejection(gold, matching_paths, frozen_solution)

    # Only candidate-unrepresented misses enter the birth cascade.
    return first_failed_birth_stage(gold, frozen_census)

This should place the currently reported 70 represented-but-unselected gold objects in stage 7 and the remaining 70 representation misses in stages 1–6, subject to exact matching under the already frozen line contract.

For a future emitted birth column, stages 1–6 are evaluated first. If all pass, the column is added, remains admissible, and MAP leaves it residual, it is then stage 7.

1.2 Inference-safe census

Create this artifact before opening gold:

JSON
{
  "schema_version": 1,
  "kind": "f1_transition_diagnostic_census_v1",
  "inference_safe": true,
  "scan_sha256": "...",
  "candidate_pool_sha256": "...",
  "graph_sha256": "...",
  "a0_solution_sha256": "...",
  "atomization_profiles": {
    "prediction": "SURFACE_ATOM_V1",
    "diagnostic_probe": "TRANSITION_MODE_PROBE_V1"
  },
  "hard_caps": {},
  "case_counters": {},
  "plane_pair_records": [],
  "multimodal_parent_records": [],
  "transition_component_records": [],
  "birth_eligibility_records": []
}

Hard caps:

Python
Run
MAX_DIAGNOSTIC_PLANE_PAIRS = 512
MAX_MULTIMODAL_PARENT_RECORDS = 4_000
MAX_DIAGNOSTIC_COMPONENTS = 4_096
MAX_COMPONENT_OWNER_BLOCKERS = 16
MAX_MATCHING_PATH_IDS = 32
MAX_NORMAL_MODES_PER_PARENT = 4

A cap breach must fail the diagnostic job. Do not silently retain the “best” records, because that would make post-lock attribution depend on a ranking rule.

Exact inference-safe counters
Stage	Counters recorded before gold
1	plane_count, plane_pair_count, angle_eligible_pair_count, locally_supported_pair_count, plane_support_component_count
2	Per pair: tube_parent_count, left_mode_count, right_mode_count, two_mode_rung_count, left_side_bin_count, right_side_bin_count
3	raw_multimodal_parent_count, v1_merged_parent_count, per parent up to four mode normals, mode support fractions, pairwise mode angles, V1 mean-normal residuals
4	transition_edge_count, component_count, largest_component_parent_count, largest_component_mode_count, occupied bins, side bins, transition-rung bins, maximum empty run, branch count
5	The same component statistics restricted to A0-residual physical parents; non-residual blocker histogram by semantic label and path ID
6	Every geometry gate as a Boolean, first rejection code, all dimensionless residuals, duplicate match IDs, saved residual mass upper bound, object cost, birth cost, and reduced-cost lower bound
7	Already available inference-side: path label, owned mass, owned bins, side bins, transition bins, conflicting paths/factors, primary energy and second semantic energy

A bounded plane-pair record should be:

JSON
{
  "plane_pair_id": "PP_<hash>",
  "plane_ids": ["PLANE:P1", "PLANE:P2"],
  "plane_angle_deg": 90.0,
  "intersection_line_hash": "...",
  "intersection_axis": [0, 0, 1],
  "transverse_parent_count": 0,
  "left_mode_count": 0,
  "right_mode_count": 0,
  "two_mode_rung_count": 0,
  "v1_merged_parent_count": 0,
  "component_count": 0,
  "largest_component": {
    "parent_count": 0,
    "mode_count": 0,
    "occupied_bin_count": 0,
    "left_side_bin_count": 0,
    "right_side_bin_count": 0,
    "transition_rung_bin_count": 0,
    "maximum_empty_bin_run": 0,
    "internal_branch_count": 0,
    "normalized_parent_mass": 0,
    "residual_parent_mass": 0
  },
  "rejection_counts": {}
}

A multimodal-parent record may serialize mode centroids and normals because those are bounded compact statistics. It may not serialize point coordinates or indices:

JSON
{
  "parent_atom_id": "A123",
  "voxel_key": [1, 2, 3],
  "raw_valid_mode_count": 2,
  "mode_overflow": false,
  "mode_summaries": [
    {
      "normal": [0, 0, 1],
      "centroid": [10, 20, 30],
      "point_count": 14,
      "occupied_fine_cell_count": 3,
      "support_fraction": 0.52
    }
  ],
  "maximum_mode_separation_deg": 87.2,
  "v1_mean_normal": [0.70, 0, 0.71],
  "v1_compatible_plane_mode_count": 0
}

For stage 5, ownership is evaluated over minimal certified axial windows, not by a residual percentage:

Python
Run
full_windows = enumerate_certified_windows(component)
residual_windows = [
    window for window in full_windows
    if all(a0_owner[parent_id] == "RESIDUAL" for parent_id in window.parent_ids)
]

if full_windows and not residual_windows:
    first_failure = 5

This prevents a large component with 80% residual support from being called residual-eligible when every valid three- or six-bin window contains a parent already consumed by an A0 object.

1.3 Post-lock gold overlay

Only after the prediction and inference-safe census hashes are frozen, generate:

JSON
{
  "schema_version": 1,
  "kind": "postlock_gold_transition_failure_attribution",
  "prediction_sha256": "...",
  "diagnostic_census_sha256": "...",
  "gold_sha256": "...",
  "rows": [
    {
      "gold_id": "M01",
      "first_failed_stage": 3,
      "reason_code": "TWO_RAW_MODES_COLLAPSED_TO_ONE_V1_ATOM",
      "matching_path_ids": [],
      "matching_plane_pair_ids": ["PP_..."],
      "matching_parent_ids": ["A123"],
      "matching_component_ids": [],
      "bounded_counters": {}
    }
  ]
}

The exact stage rules are:

No incident plane mode: no canonical plane pair has an intersection matching the gold line under the frozen line contract and local plane-mode support reaches that line.

No transition-side atoms: a matching plane pair exists, but the transient projective-normal probe finds no supported mode for one or both sides.

Modes merged: both supported raw modes exist in one parent or a local cross-parent rung, but V1 presents one mean-normal atom or one exclusive side.

Connectivity broken: supported side modes exist, but no atom/mode component forms the required axial run.

Support non-residual: a full run exists, but no corresponding residual-only window exists after A0 ownership.

Birth rejection: residual support exists, but an explicit geometry, duplicate, MDL, or column-cap condition rejects it.

MAP rejection: an incidence-complete bend-eligible existing or birth path matches the gold object but is not counted.

Gold may be used only to:

match a line to already emitted paths, plane pairs, parents, and components;

choose the line-local diagnostic record;

report the first failure and confusion category.

Gold must not alter mode clustering, plane fitting, graph edges, residual ownership, eligibility thresholds, proposal ordering, or solver energy.

INFERENCE. This cascade will distinguish whether zero birth is primarily an atom-mode loss, a plane-table loss, a graph-connectivity loss, or a residual-ownership loss. Those mechanisms currently remain conflated.

UNKNOWN. The supplied artifacts do not establish the per-gold distribution across stages 1–6. That distribution must be measured before choosing the second V2 birth family.

2. Specify TRANSITION_ATOM_V2
2.1 Chosen representation

Use:

One physical parent per coarse voxel, zero to four projective-normal mode children per parent, and bounded same-surface/cross-mode edges.

Do not create separate physical atoms for “panel interior,” “transition strip,” and “exposed boundary.” A panel mode can be both a surface patch and a transition boundary; those semantic roles should be derived from topology. A full serialized fine-resolution surfel graph is not yet justified.

OBSERVED. V1 averages all normals within a voxel and derives one axis from the largest point-covariance eigenvector. Its area proxy is point_count * voxel_size², which is sampling-density dependent. Adjacency is based on neighboring occupied voxels without surface-mode typing. 

13_patch_graph_latent_decomposi…

2.2 Data structures
Python
Run
@dataclass(frozen=True)
class TransitionParentAtom:
    parent_atom_id: str
    voxel_key: tuple[int, int, int]
    voxel_level: int
    centroid: tuple[float, float, float]
    local_spacing_mm: float
    voxel_size_mm: float
    point_count: int
    area_proxy_mm2: float
    normalized_support_mass: int
    mode_child_ids: tuple[str, ...]
    valid_mode_count: int
    mode_overflow: bool
    unmodeled_point_count: int
    boundary_sector_mask: int
    boundary_coverage: float
    largest_empty_sector_run: int


@dataclass(frozen=True)
class SurfaceModeChild:
    mode_atom_id: str
    parent_atom_id: str
    centroid: tuple[float, float, float]
    normal_projective: tuple[float, float, float]
    point_count: int
    support_fraction: float
    occupied_fine_cell_count: int
    normal_dispersion_deg: float
    planarity: float
    linearity: float
    axis_direction: tuple[float, float, float]
    axis_reliable: bool
    surface_fit_p90_mm: float
    boundary_sector_mask: int
    boundary_coverage: float
    largest_empty_sector_run: int
    boundary_strength: float
    fine_occupancy_hash: str


@dataclass(frozen=True)
class SurfaceModeEdge:
    mode_edge_id: str
    mode_ids: tuple[str, str]
    edge_kind: str  # SAME_SURFACE or CROSS_MODE_LOCAL
    spatial_residual: float
    normal_residual: float
    plane_residual: float


@dataclass(frozen=True)
class TransitionAtomGraphV2:
    profile: str
    parents: tuple[TransitionParentAtom, ...]
    modes: tuple[SurfaceModeChild, ...]
    mode_edges: tuple[SurfaceModeEdge, ...]
    local_spacing_mm: float
    parent_voxel_size_mm: float
    fine_cell_size_mm: float
    diagnostics: dict[str, Any]
2.3 Projective normal handling

Input normal signs must not split a surface:

Python
Run
def projective_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = abs(float(np.dot(unit(a), unit(b))))
    return degrees(acos(clip(cosine, -1.0, 1.0)))


def projective_mean(normals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    moment = sum(
        weight * np.outer(normal, normal)
        for normal, weight in zip(normals, weights)
    )
    _, vectors = np.linalg.eigh(moment)
    result = vectors[:, -1]

    # Canonical serialization sign.
    for value in result:
        if abs(float(value)) > 1e-12:
            if value < 0:
                result = -result
            break
    result[np.abs(result) < 1e-12] = 0.0
    return result

When a plane normal is canonicalized, its offset must be negated at the same time when the normal flips.

2.4 Deterministic builder pseudocode
Python
Run
TRANSITION_ATOM_PROFILE = "TRANSITION_ATOM_V2"

MAX_PARENT_ATOMS = 4_000
MAX_MODE_CHILDREN_PER_PARENT = 4
MAX_MODE_CHILDREN = 16_000
MAX_COMPACT_SUPPORT_RECORDS = 20_000
MAX_MODE_EDGES = 200_000
MAX_PLANE_MEMBERSHIPS_PER_MODE = 3

MODE_SEED_ANGLE_DEG = 12.0
SAME_SURFACE_ANGLE_DEG = 15.0
MIN_TRANSITION_MODE_ANGLE_DEG = 20.0
MIN_MODE_POINTS = 4
MIN_MODE_FINE_CELLS = 2
MAX_SAME_SURFACE_NEIGHBORS = 8
BOUNDARY_SECTOR_COUNT = 8


def build_transition_atom_graph_v2(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    local_spacing_mm: float,
) -> TransitionAtomGraphV2:
    assert len(points) == len(normals)
    assert len(points) <= 200_000

    base_voxel = max(3.0 * local_spacing_mm, 2.5)

    # A level increases voxel volume by exactly 2x.
    level = 0
    while True:
        voxel_size = base_voxel * (2.0 ** (level / 3.0))
        parent_buckets = bucket_by_voxel(points, voxel_size)

        if len(parent_buckets) <= MAX_PARENT_ATOMS:
            break

        level += 1
        if level > 12:
            raise ValueError("unable to satisfy V2 parent cap")

    fine_size = max(local_spacing_mm, voxel_size / 3.0)

    parents = []
    modes = []
    transient_point_partition = {}

    for voxel_key, point_ids in sorted(parent_buckets.items()):
        # Canonical order makes floating reductions and first-fit clustering
        # invariant to source PLY order.
        point_ids = canonical_point_order(
            point_ids,
            points,
            normals,
            xyz_quantum_mm=0.001,
            normal_quantum=1e-6,
        )

        micro_surfels = []

        for fine_key, fine_ids in sorted(
            bucket_subset_by_voxel(point_ids, points, fine_size).items()
        ):
            normal_groups = deterministic_complete_link_groups(
                fine_ids,
                normals,
                projective_angle_limit_deg=MODE_SEED_ANGLE_DEG,
            )

            for group in normal_groups:
                micro_surfels.append(
                    build_micro_surfel(
                        group,
                        points,
                        normals,
                        fine_key=fine_key,
                    )
                )

        micro_edges = []
        tau_surface = min(
            3.0,
            max(1.25, 1.5 * local_spacing_mm),
        )

        for left, right in neighboring_micro_surfel_pairs(micro_surfels):
            angle = projective_angle_deg(left.normal, right.normal)
            plane_gap = symmetric_surface_plane_gap(left, right)

            if (
                angle <= SAME_SURFACE_ANGLE_DEG
                and plane_gap <= tau_surface
            ):
                micro_edges.append((left.micro_id, right.micro_id))

        components = connected_components(micro_surfels, micro_edges)

        valid = [
            component
            for component in components
            if component.point_count >= MIN_MODE_POINTS
            and component.occupied_fine_cell_count >= MIN_MODE_FINE_CELLS
        ]

        valid.sort(
            key=lambda c: (
                -c.point_count,
                quantized_projective_normal(c.normal),
                quantized_centroid(c.centroid),
                c.fine_occupancy_hash,
            )
        )

        overflow = len(valid) > MAX_MODE_CHILDREN_PER_PARENT
        retained = valid[:MAX_MODE_CHILDREN_PER_PARENT]

        parent_id = stable_parent_id(
            profile=TRANSITION_ATOM_PROFILE,
            voxel_level=level,
            voxel_key=voxel_key,
        )

        child_ids = []
        for component in retained:
            mode_id = stable_mode_id(
                parent_id=parent_id,
                normal=component.normal,
                centroid=component.centroid,
                fine_occupancy_hash=component.fine_occupancy_hash,
            )
            child_ids.append(mode_id)

            modes.append(
                serialize_mode_summary(
                    mode_id,
                    parent_id,
                    component,
                )
            )

            for point_id in component.point_ids:
                if point_id in transient_point_partition:
                    raise AssertionError("point assigned to multiple V2 modes")
                transient_point_partition[point_id] = mode_id

        area_proxy = min(
            6.0 * voxel_size * voxel_size,
            fine_size * fine_size
            * sum(mode.occupied_fine_cell_count for mode in retained),
        )

        parents.append(
            TransitionParentAtom(
                parent_atom_id=parent_id,
                voxel_key=voxel_key,
                voxel_level=level,
                centroid=stable_mean(points[point_ids]),
                local_spacing_mm=local_spacing_mm,
                voxel_size_mm=voxel_size,
                point_count=len(point_ids),
                area_proxy_mm2=area_proxy,
                normalized_support_mass=0,  # filled case-relatively later
                mode_child_ids=tuple(sorted(child_ids)),
                valid_mode_count=len(retained),
                mode_overflow=overflow,
                unmodeled_point_count=(
                    len(point_ids)
                    - sum(mode.point_count for mode in retained)
                ),
                boundary_sector_mask=0,
                boundary_coverage=0.0,
                largest_empty_sector_run=0,
            )
        )

    if len(parents) + len(modes) > MAX_COMPACT_SUPPORT_RECORDS:
        raise AssertionError("V2 compact support cap violated")

    same_surface_edges = build_mutual_same_surface_edges(
        parents=parents,
        modes=modes,
        parent_neighbor_rule="26_NEIGHBOR",
        angle_limit_deg=SAME_SURFACE_ANGLE_DEG,
        surface_gap_mm=tau_surface,
        maximum_neighbors=MAX_SAME_SURFACE_NEIGHBORS,
    )

    cross_mode_edges = []
    for parent in parents:
        children = modes_for_parent(parent, modes)
        for left, right in combinations(children, 2):
            angle = projective_angle_deg(
                left.normal_projective,
                right.normal_projective,
            )
            if MIN_TRANSITION_MODE_ANGLE_DEG <= angle <= 90.0:
                cross_mode_edges.append(
                    build_cross_mode_edge(left, right)
                )

    edges = canonical_deduplicate_edges(
        [*same_surface_edges, *cross_mode_edges]
    )
    if len(edges) > MAX_MODE_EDGES:
        raise AssertionError("V2 edge cap violated")

    modes = attach_boundary_sector_evidence(
        modes,
        same_surface_edges,
        sector_count=BOUNDARY_SECTOR_COUNT,
    )
    parents = aggregate_parent_boundary_evidence(parents, modes)

    masses = normalized_parent_masses(
        parents,
        minimum=1,
        maximum=4,
    )
    parents = [
        replace(parent, normalized_support_mass=masses[parent.parent_atom_id])
        for parent in parents
    ]

    return TransitionAtomGraphV2(
        profile=TRANSITION_ATOM_PROFILE,
        parents=tuple(sorted(parents, key=lambda p: p.parent_atom_id)),
        modes=tuple(sorted(modes, key=lambda m: m.mode_atom_id)),
        mode_edges=tuple(sorted(edges, key=lambda e: e.mode_edge_id)),
        local_spacing_mm=local_spacing_mm,
        parent_voxel_size_mm=voxel_size,
        fine_cell_size_mm=fine_size,
        diagnostics=build_v2_diagnostics(parents, modes, edges),
    )

deterministic_complete_link_groups() should use sorted input and add a normal to the first existing group only when its projective angle to every group member is at most 12°. A final canonical merge pass may merge groups only when the combined complete-link diameter remains at most 15°.

2.5 Boundary evidence

For each child, construct a deterministic tangent basis from its projective normal. Project vectors to same-surface neighboring children into eight angular sectors. Store:

boundary_sector_mask
boundary_coverage = occupied_sector_count / 8
largest_empty_sector_run
boundary_strength = largest_empty_sector_run / 8

A free edge and one side of a fold can both have an open half-neighborhood. The distinction is that a fold has a compatible second mode and cross-mode transition rung; a free edge does not.

2.6 Compact serializer
Python
Run
def compact_transition_graph_v2(
    graph: TransitionAtomGraphV2,
) -> dict[str, Any]:
    payload = {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "support_parent_table": [
            parent_to_compact_dict(parent)
            for parent in graph.parents
        ],
        "support_mode_table": [
            mode_to_compact_dict(mode)
            for mode in graph.modes
        ],
        "support_mode_edges": [
            edge_to_compact_dict(edge)
            for edge in graph.mode_edges
        ],
        "transition_atom_caps": {
            "input_point_cap": 200_000,
            "parent_cap": 4_000,
            "mode_cap": 16_000,
            "parent_plus_mode_cap": 20_000,
            "mode_edge_cap": 200_000,
            "mode_children_per_parent_cap": 4
        }
    }

    reject_recursive_keys(
        payload,
        forbidden={
            "point_indices",
            "source_points",
            "points",
            "bbox_points",
            "vertex_indices",
        },
    )
    return payload

Parent IDs:

P_<sha256(profile, voxel_level, voxel_key)[:16]>

Mode IDs:

M_<sha256(
    parent_id,
    normal quantized to 1e-6,
    centroid quantized to 0.01 mm,
    fine_occupancy_hash
)[:16]>

Mode-edge IDs hash the sorted mode IDs and edge kind.

Children do not multiply physical evidence because:

each transient point belongs to at most one child;

only parents own normalized support mass;

children are side/topology witnesses, not independently ownable objects;

overlapping paths compete for the same parents;

a parent contributes one axial-bin occupancy regardless of child count.

UNKNOWN. A cap-overflow parent cannot be trusted as a complete transition certificate. It should remain usable as generic support but must set:

mode_overflow = true
transition_certificate_allowed = false
3. Make panel-side ownership coherent
3.1 Schema

Each child may have zero to three candidate memberships:

JSON
{
  "mode_atom_id": "M_...",
  "plane_side_memberships": [
    {
      "canonical_plane_id": "PLANE:P1",
      "normal_residual": 0.31,
      "plane_distance_residual": 0.42,
      "support_component_id": "PSC_...",
      "eligible": true
    }
  ]
}

A path stores physical parent incidence separately from mode-side evidence:

JSON
{
  "support_parent_ids": ["P_1", "P_2"],
  "support_parent_bins": {
    "P_1": "BIN_000001"
  },
  "support_side_mode_candidates": {
    "P_1": {
      "PLANE:P1": ["M_1"],
      "PLANE:P2": ["M_2"]
    }
  },
  "transition_rung_ids": ["TR_1"],
  "transition_parent_ids": ["P_1"],
  "boundary_parent_ids": [],
  "incidence_complete": true,
  "incidence_certificate": {
    "tolerance_profile": "INCIDENCE_V2",
    "parent_table_sha256": "...",
    "mode_table_sha256": "...",
    "plane_table_sha256": "...",
    "max_dimensionless_residuals": {}
  }
}

A transition rung may be:

two distinct mode children in one parent; or

two compatible mode children in adjacent parents connected by a local cross-mode edge.

The latter is necessary because two panel skins near a sharp transition will not always land in the same coarse voxel.

3.2 Ownership variables

Let P
i
	​

 be the physical parents incident to path i, M(p) its child modes, and F
i
	​

 its candidate panel modes.

Parent ownership:

O
p,i
	​

∈{0,1},O
p,R
	​

∈{0,1}
i:p∈P
i
	​

∑
	​

O
p,i
	​

+O
p,R
	​

=1

Mode-side assignment:

Q
m,i,f
	​

∈{0,1}
Q
m,i,f
	​

≤O
parent(m),i
	​

Q
m,i,f
	​

≤membership(m,f)
f
∑
	​

Q
m,i,f
	​

≤O
parent(m),i
	​


Therefore:

a parent has exactly one physical owner;

multiple child modes may coexist;

one child may be compatible with several planes but is selected for at most one side of its owner;

two distinct children may represent the two sides of one bend.

The children are not mutually exclusive existential hypotheses. Their transient point partitions are mutually disjoint. Their selected side assignments are exclusive.

3.3 Transition witnesses

For a canonical rung r=(m,n) supporting distinct planes f

=g:

W
r,i,f,g
	​

=Q
m,i,f
	​

∧Q
n,i,g
	​


Linearization:

Python
Run
model.Add(w <= q_m_f)
model.Add(w <= q_n_g)
model.Add(w >= q_m_f + q_n_g - 1)

For a cross-parent rung, also require both parents owned:

Python
Run
model.Add(w <= owner[parent_m, path])
model.Add(w <= owner[parent_n, path])

Axial occupancy:

Python
Run
parent_bin[path, bin] = OR(
    owner[parent, path]
    for parent in parents_in_bin
)

side_bin[path, plane, bin] = OR(
    side_assignment[mode, path, plane]
    for mode whose parent is in bin
)

transition_bin[path, plane_f, plane_g, bin] = OR(
    rung_witness[rung, path, plane_f, plane_g]
    for rung in bin
)

Parent mass:

Python
Run
owned_mass[path] = sum(
    parent.normalized_support_mass * owner[parent, path]
    for parent in support_parents[path]
)

No child mass appears in this expression.

3.4 Label constraints
Counted bend

For fixed incident pair f,g:

Python
Run
z_bend = label[path, BEND]

model.Add(sum(parent_bin[path, b] for b in bins) >= min_bins_bend).OnlyEnforceIf(z_bend)
model.Add(owned_mass[path] >= min_mass_bend).OnlyEnforceIf(z_bend)

model.Add(
    sum(side_bin[path, f, b] for b in bins) >= min_side_bins_bend
).OnlyEnforceIf(z_bend)

model.Add(
    sum(side_bin[path, g, b] for b in bins) >= min_side_bins_bend
).OnlyEnforceIf(z_bend)

# A1 engineering-fixed transition certificate.
model.Add(
    sum(transition_bin[path, f, g, b] for b in bins) >= 1
).OnlyEnforceIf(z_bend)

For A1, retain the existing generic and side minima rather than sweeping them. Change only how side evidence is represented and counted.

Perimeter/free edge

A perimeter object has one supported panel mode, persistent boundary evidence, and no two-mode transition rung:

Python
Run
model.Add(selected_panel_mode_count[path] == 1).OnlyEnforceIf(z_perimeter)

model.Add(
    sum(boundary_bin[path, b] for b in bins) >= min_boundary_bins
).OnlyEnforceIf(z_perimeter)

model.Add(
    sum(transition_bin_any_pair[path, b] for b in bins) == 0
).OnlyEnforceIf(z_perimeter)

A path with two incident plane modes should not be relabeled as a perimeter merely because one side lost ownership. It needs an explicit one-panel boundary interpretation in the graph.

Corner

For A1, preserve the current endpoint semantics and add coherent side evidence:

Python
Run
model.Add(endpoint[path, 0, MULTIWAY_JUNCTION] == 1).OnlyEnforceIf(z_corner)
model.Add(endpoint[path, 1, MULTIWAY_JUNCTION] == 1).OnlyEnforceIf(z_corner)

model.Add(two_distinct_plane_modes[path] == 1).OnlyEnforceIf(z_corner)

Corner versus bend is then decided by endpoint topology and structured relations, not by support length.

Junction

A junction should own a NODE-geometry support object, not share the same parents among several bend paths:

Python
Run
model.Add(incident_plane_mode_count[node] >= 3).OnlyEnforceIf(z_junction)
# or a certified transition-graph branch:
model.Add(local_transition_degree[node] >= 3).OnlyEnforceIf(z_junction)

Adjacent bend paths terminate at the junction node through endpoint relations. The junction object owns the central parents. This preserves exclusive physical ownership.

Residual
Python
Run
model.AddExactlyOne([
    owner[parent, path] for path in compatible_paths[parent]
] + [residual_owner[parent]])

Residual parents have no mode-side assignments and incur residual cost once.

For the present line-only tranche:

HOLE_NONCOUNTED  — eligible only for LOOP geometry
CURVE_NONCOUNTED — eligible only for POLYLINE geometry
JUNCTION         — eligible only with a node/branch certificate
4. Redesign the first birth slice only as far as evidence permits
4.1 V1 contradictions

OBSERVED. The current birth implementation combines assumptions that are inconsistent with the V1 atom schema:

The parent voxel is at least 3s wide, but the birth transverse tube is at most 1.25s and capped at 3 mm. A parent centroid can therefore miss the transition even when its underlying points cross it.

The atom normal is a mean over potentially two modes, but birth requires it to be within 15° of one plane.

Every surviving atom is exclusively assigned to the nearer plane, so one atom can never witness both physical side modes.

The atom-axis gate compares the plane-intersection axis with the largest covariance eigenvector of a voxel. On a roughly planar or cubic patch that in-plane eigenvector is not a stable bend-axis estimate. The census shows this gate removes 2,570 of the 2,960 normal survivors.

Connectivity uses untyped voxel-neighbor edges rather than two continuing panel-side chains.

Six atoms and six bins are sampling-scale surrogates, not a minimal transition-topology certificate.

Duplicate suppression examines geometric overlap with existing paths without requiring the existing path to be incidence-certified, bend-eligible, or supported by the same physical parents. 

15_f1_structured_residual_birth

 

15_f1_structured_residual_birth

 

15_f1_structured_residual_birth

 

07_BIRTH_V1_STAGE_CENSUS

There are also two implementation blockers that would prevent a V1 birth from becoming selectable even if one reached the proposal list:

the emitted incidence_certificate lacks the atom-table hash and max_dimensionless_residuals required by _incidence_is_certified();

it emits no support_atom_sides, while compact V1 atoms contain no incident plane IDs for the solver’s fallback side map.

The solver will consequently fix such a birth inactive or make the bend side constraint infeasible. 

15_f1_structured_residual_birth

 

17_solve_f1_structured_union_cp…

 

17_solve_f1_structured_union_cp…

 

17_solve_f1_structured_union_cp…

4.2 First justified family: BIRTH_V2A_PLANE_PAIR_TRANSITION

This first family remains restricted to a globally extracted plane pair. It does not attempt missing-plane, one-sided, curved, or generic-line recovery.

Eligibility:

1. Every support parent is residual in the current exact solution.
2. Two canonical plane modes exist.
3. Their projective normal angle is in [20°, 90°].
4. Each selected child passes:
     normal residual          <= 1
     point-to-plane residual  <= 1
     plane-support-component membership is true
5. The two sides use distinct child modes.
6. At least one cross-mode rung exists.
7. The rung graph forms a nonbranching axial lane.
8. Repeated parallel lanes are split before proposal construction.
9. The proposal does not duplicate a supported existing bend path.

Dimensionless memberships:

r
n
	​

(m,f)=
15
∘
∠
proj
	​

(n
m
	​

,n
f
	​

)
	​

r
d
	​

(m,f)=
min(3 mm,max(1.25 mm,1.5s))
∣n
f
	​

⋅c
m
	​

+d
f
	​

∣
	​


No atom-axis gate is used. The plane-pair cross product supplies the expected axis.

Axial requirements:

Python
Run
bin_width = max(3.0, 2.0 * spacing)

occupied_bins >= 3
left_side_bins >= 3
right_side_bins >= 3
transition_rung_bins >= 2
longest_consecutive_occupied_run >= 3
maximum_empty_bin_run <= 1
span_mm >= 2.0 * bin_width
internal_axial_branch_degree <= 2
tls_axis_angle_to_plane_intersection <= 4 degrees
p90_line_residual <= transverse_tolerance

Three bins are the minimum proof-friendly line run: two endpoint bins and one interior bin. This replaces the V1 six-atom surrogate. It is fixed before the Gold37 A1/A2 runs.

Repeated parallel support is separated by connected components in the two-sided ladder graph. Do not merge components merely because their axes are parallel. A lane contains paired side chains:

f_mode(p0) — f_mode(p1) — f_mode(p2)
    |            |            |
g_mode(p0) — g_mode(p1) — g_mode(p2)

The vertical edges are cross-mode rungs; horizontal edges are same-surface continuation.

Duplicate suppression requires all of:

existing path incidence_complete
existing path bend-eligible
axis angle <= 4 degrees
transverse offset <= max(3 mm, 2*s)
shorter-span overlap >= 0.75
support-parent Jaccard >= 0.60

An unsupported residual path is not allowed to suppress a birth.

4.3 Safe lower bound

Let P
k
	​

 be the support parents of birth k, all currently residual, and q
p
	​

 the normalized physical parent mass.

S
k
	​

=C
residual
	​

p∈P
k
	​

∑
	​

q
p
	​


Before learned weights:

LB(k)=C
object
	​

+C
birth,k
	​

−S
k
	​


This is safe when:

birth support is restricted to currently residual parents;

fit and conflict costs are nonnegative;

no negative factor or continuation reward is attached to the birth;

parent mass is counted once;

the graph freezes the pre-birth object and residual costs.

Python
Run
saved_residual_upper_bound = residual_cost * sum(
    parent_mass[parent_id]
    for parent_id in proposal.support_parent_ids
)

lower_bound = (
    object_cost
    + proposal.birth_cost_int
    - saved_residual_upper_bound
)

if lower_bound >= 0:
    prune_safely()
else:
    add_column_for_exact_decision()

The existing solver already places object and residual costs in the primary objective and freezes the profile when birth columns are added. 

17_solve_f1_structured_union_cp…

 

15_f1_structured_residual_birth

When learned negative unaries or structured rewards are introduced, this bound must either include their proven minimum contribution or be disabled.

4.4 Ordered later families
Order	Family	Preconditions	Falsification test
1	V2A_PLANE_PAIR_TRANSITION	Two global plane modes and two-sided ladder	On deterministic candidate-deletion cases with a valid global pair: recover ≥50%, birth precision ≥0.70, false births ≤0.25/scan
2	V2B_LOCAL_PLANE_PAIR	Stage 1 dominates and local mode chains can fit two planes	Among stage-1 misses, recover ≥30% across at least five held-out families; false local pairs ≤0.25/scan
3	V2C_PARALLEL_LANE_SPLIT	V2A components contain multiple transversely disjoint ladders	Remove one represented sibling from annotated parallel sets; recover ≥50% without increasing duplicate births above 0.1/scan
4	V2D_ENDPOINT_SHORT_RETURN	Stage 4/6 is dominated by topologically annotated returns of two or three bins	Recover ≥40% of deleted short returns; false endpoint spurs ≤0.2/scan
—	One-sided occlusion	Requires a visibility or scanner-view certificate	Do not implement from the current PLY representation

V2B fits local planes from two same-surface child chains rather than proposing an unconstrained line. V2D must be anchored to an existing path endpoint or certified junction node. Neither belongs in V2A.

UNKNOWN. A one-sided true bend and a one-sided free edge can be locally indistinguishable without viewpoint or observability information. A one-sided birth family is therefore not justified by the current artifacts.

5. Separate true folds from outlines at representation level
Material transition certificate

Require all of:

two distinct projective-normal modes
distinct canonical incident plane modes
cross-mode rung between the side modes
same-surface continuation on both sides
paired axial run over multiple bins
plane-pair axis compatibility
no mode-overflow parent used as the sole certificate

This is stronger than “two normals occur near the line.” It requires a local two-sided surface ladder.

Exposed perimeter/free-edge certificate

Require:

one coherent plane-side mode
persistent tangent-sector boundary void
same-surface continuation along the edge
no compatible second mode chain
no cross-mode transition rung

A single isolated mode with missing neighbors is not sufficient; the boundary structure must persist along the path.

Duplicate candidate certificate

Two candidate representations are hard duplicates when they explain the same physical support:

support-parent Jaccard >= 0.60
same canonical side-mode pair
line-equivalent geometry
same transition-rung component

They then compete for identical parent ownership or are collapsed to representatives of one object.

Two geometrically parallel candidates with disjoint parent ladders are not duplicates.

Corner or junction certificate

Use local transition-graph topology:

endpoint transition degree >= 3
or at least three canonical plane modes meet
or multiple certified bend ladders terminate at one node

A non-counted corner trace additionally needs the current multiway endpoint conditions. A junction object owns the central support; neighboring bend paths terminate at it.

The contact sheets are consistent with the need for these local distinctions: several cases place F1 paths on outer frames while omitting interior or repeated parallel gold lines, and the curved C023 case has no adequate straight-line representation. 

09_AUTHORITATIVE_CONTACT_1

 

10_AUTHORITATIVE_CONTACT_2

Distinctions that remain UNKNOWN

The present PLY evidence cannot reliably distinguish:

an occluded one-sided fold from a free edge;

a true material edge from a scan-coverage truncation;

a formed fold from a weld or seam with identical local geometry;

disjoint inner/outer skin traces from two intentionally distinct parallel bends without thickness or solid-side evidence;

manufacturing-counted versus non-counted semantics at every complex corner;

a maximal counted bend on strongly rolled geometry under the current line-only ontology.

Those cases should remain residual or review-required. The manual PLY audit already found 0/13 visually acceptable, with outline aliases, duplicates, and missed bends dominating; numerical count agreement was explicitly insufficient. 

13_manual_visual_audit

6. Family-held-out preregistered experiment
6.1 Evaluation-only grouping

Create a separate evaluator manifest:

JSON
{
  "scan_id": "C001",
  "split_group_id": "G001",
  "physical_specimen_id": "...",
  "part_lineage_id": "...",
  "drawing_revision_lineage_id": "...",
  "tooling_family_id": "...",
  "acquisition_series_id": "...",
  "scan_sha256": "..."
}

Union scans into one split group when metadata establishes any of:

identical scan bytes
same physical specimen
full/partial/repeat acquisitions of that specimen
same part and revision lineage
same drawing/CAD/tooling geometry lineage
left/right or mirrored variants explicitly declared as one topology family

When relation metadata is ambiguous, merge conservatively rather than risk splitting a near duplicate across folds.

split_group_id and all lineage fields must be absent from candidate pools, graphs, solver inputs, caches, and features. Add a test that injecting or changing them leaves the graph bytes unchanged.

6.2 Arms
A0

Use the frozen V1 output exactly as recorded. Do not regenerate it with later dependencies.

A1

Change atomization and panel-side representation only:

same 200,000-point working cloud
same plane extractor
same candidate geometries and source lanes
same path set
same labels
same fixed MDL costs
same continuations and topology factors
birth disabled
no learned potential
V2 parent ownership and child side witnesses

To isolate the change, continue building the legacy V1 atom graph for upstream decomposition and candidate generators. Build V2 in parallel and use it only in the structured pool, graph, incidence, and exact ownership stages.

A2

A1 plus only:

BIRTH_V2A_PLANE_PAIR_TRANSITION

No V2B, short-return, one-sided, or learned family is included.

6.3 Required annotations and metrics

The experiment needs structured labels for false outputs, not only bend truth lines. At minimum, every A0/A1/A2 matched or unmatched prediction must be adjudicated as:

BEND
PERIMETER_OR_FREE_EDGE
CORNER
JUNCTION
DUPLICATE_TRACE
CURVE_OR_ROLLED
UNOBSERVABLE
OTHER_RESIDUAL

Report per split group and as pooled totals:

candidate recall before MAP
candidate precision before MAP
represented-but-unselected gold rate
line TP / FP / FN / precision / recall / F1
exact counted-object count
outline/free-edge counted-FP rate
duplicate counted-FP rate
stage 1–7 miss distribution
parent ownership accuracy
side-mode and transition-rung recall
V2 mode-overflow rate
birth eligibility, precision, and recovery
deterministic replay rate
runtime and peak RSS

Use the split group as the sampling unit:

Python
Run
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0

For each bootstrap replicate, sample groups with replacement and recompute both family-macro and pooled metrics. Report 95% paired percentile intervals. Also report an exact paired sign test over per-group F1 changes. Do not bootstrap scans or candidates as independent observations.

6.4 Preregistered A1 criteria
GO only when all hold
supported candidate recall >= 0.720
95% paired lower bound for candidate-recall change > 0

family-macro line F1 improves by >= 0.050
95% paired lower bound for family-macro F1 change > 0

counted outline/free-edge FPs fall by >= 50%
pooled bend recall is at least 0.211
stage-3 plus stage-4 misses fall by >= 30%

100% exact-job success
100% selected replay byte identity
no compact cap violation
all measured jobs < 16 GiB peak RSS
STOP when any hold
candidate-recall gain < 0.050
outline/free-edge FP reduction < 30%
bend recall < 0.201
family-macro F1 gain < 0.020
stage-3 plus stage-4 misses fall by < 15%
mode overflow or incomplete V2 certification occurs in > 5% of scans

A1 failing these criteria falsifies the chosen parent/mode representation. Do not compensate with learned selection weights.

6.5 Preregistered A2 criteria
GO only when all hold
>= 50% recovery among A1 misses that possess a valid V2A certificate
birth precision >= 0.70
false births <= 0.25 per scan
family-macro F1 improves by >= 0.040 over A1
pooled precision loses no more than 0.030
at least ceil(2 * G / 3) held-out groups improve or tie
100% deterministic replay

For 27 independent groups, ceil(2G/3) is 18.

STOP when any hold
V2A eligibility covers < 10% of candidate-unrepresented A1 misses
eligible-object recovery < 30%
birth precision < 0.50
false births > 0.50 per scan
outline/free-edge counted FPs increase by > 10%
6.6 Learning threshold

Do not fit low-capacity potentials unless every inner training split contains:

at least 20 independent family groups
at least 8 families containing counted bends and explicit perimeter/corner contrasts

A class-specific coefficient additionally requires that class in at least eight independent training families. Otherwise tie it to its ontology group or leave it engineering-fixed. The prior sign-constrained, low-capacity feature plan remains appropriate only after this representation gate passes. 

02_PRIOR_PROFESSOR_RESPONSE

The current Gold37 corpus is now a development corpus: its sheets and failure statistics have influenced this V2 design. Preserve its A0 hashes and use it for preregistered engineering ablations, but do not call it independent confirmation.

After A1/A2 and any later learning are frozen, collect at least:

30 new scans
15 untouched families
preferably >= 2 scans per family

Freeze code, dependency, engineering-profile, and prediction-command hashes before opening the confirmation truth.

7. Smallest code tranche
7.1 Ordered file plan
1. 13_patch_graph_latent_decomposition.py

Patch first:

a. Add TransitionParentAtom, SurfaceModeChild,
   SurfaceModeEdge, TransitionAtomGraphV2.

b. Add build_transition_atom_graph_v2().

c. Keep build_surface_atom_graph() unchanged for upstream A1 generators.

d. Build V2 from the same bounded working cloud in parallel.

e. Add diagnostic-only TRANSITION_MODE_PROBE_V1/V2 census.

f. Replace asdict(atom_graph) cache serialization with an explicit
   compact serializer that omits point_indices.

g. Make _surface_atom_graph_from_dict() tolerate absent point_indices,
   or rebuild the legacy atom graph on a cache hit if a downstream
   legacy function still requires them.

h. Bump F1_RESEARCH_INPUT_SEMANTICS_VERSION and cache_schema_version.

New transition_atom_graph_v2 fields:

profile
parent_voxel_size_mm
fine_cell_size_mm
parent_table_sha256
mode_table_sha256
mode_edge_table_sha256
parents
modes
mode_edges
caps
diagnostics
2. 14_build_f1_scan_only_candidate_pool.py

Move the research pool to schema version 3 and add:

support_representation_profile
support_parent_table
support_mode_table
support_mode_edges
plane_mode_memberships
transition_atom_caps

Expand each plane record with bounded scan-derived support:

canonical_plane_id
normal
offset
support_count
merged_plane_ids
tangent_basis
extent_uv
fit_p90_mm
support_component_ids
support_parent_ids
support_mode_ids

Keep V1 tables during A1 for regression comparison, but make the structured graph explicitly select TRANSITION_ATOM_V2.

Add recursive schema rejection for:

point_indices
source_points
points
bbox_points
filename
part_id
expected_count
manual_truth
cad_metadata
split_group_id
3. 15_f1_structured_residual_birth.py

First tranche:

a. Leave birth selection disabled.
b. Add generate_transition_birth_census_v2a().
c. Emit exact rejection codes and lower-bound terms.
d. Add no columns during A1.

Only after A1 GO:

e. Add generate_residual_birth_columns_v2a().
f. Replace atom IDs with physical parent IDs.
g. Remove the V1 atom-axis gate.
h. Build components from two-sided ladder edges.
i. Require valid INCIDENCE_V2 certificates.
j. Suppress duplicates only through certified shared-parent support.

New birth fields:

birth_profile
support_parent_ids
support_parent_bins
support_side_mode_ids
transition_rung_ids
transition_bin_ids
normalized_parent_mass
object_cost_int
birth_cost_int
residual_cost_per_mass_int
saved_residual_cost_upper_bound_int
lower_bound_int
lower_bound_cost_source

New rejection codes:

NO_GLOBAL_PLANE_PAIR
NO_LEFT_SIDE_MODE
NO_RIGHT_SIDE_MODE
NO_TRANSITION_RUNG
NO_THREE_BIN_RUN
AXIAL_BRANCH
LINE_FIT_RESIDUAL
PLANE_AXIS_RESIDUAL
NON_RESIDUAL_PARENT
CERTIFIED_EXISTING_DUPLICATE
NONNEGATIVE_LOWER_BOUND
COLUMN_CAP
4. 16_f1_structured_union_graph.py

Extend the strict whitelist and normalization for V2.

Replace V1-only _atom_side() with mode-membership construction. complete_all_path_incidence() becomes profile-dispatched:

Python
Run
if profile == "SURFACE_ATOM_V1":
    return complete_incidence_v1(...)
if profile == "TRANSITION_ATOM_V2":
    return complete_incidence_v2(...)
raise ValueError(...)

New path fields:

support_parent_ids
support_parent_bins
support_side_mode_candidates
transition_rung_ids
transition_parent_ids
boundary_parent_ids
incidence_complete
incidence_origin
incidence_certificate

A V2 stable path ID hashes the physical parent-support signature, not child IDs. Splitting one parent into two valid normal modes must not manufacture a distinct physical path identity.

5. 17_solve_f1_structured_union_cpsat.py

Add profile-specific support variables while preserving the existing exact objective and tie stages:

parent ownership O[p,i]
residual parent owner O[p,R]
mode-side assignment Q[m,i,f]
rung witness W[r,i,f,g]
parent-bin occupancy
side-bin occupancy
transition-bin occupancy
boundary-bin occupancy

Add _incidence_v2_is_certified() requiring:

tolerance_profile == INCIDENCE_V2
parent_table_sha256
mode_table_sha256
plane_table_sha256
complete dimensionless residual set
no truncated parent incidence
no mode-overflow parent used as sole transition certificate

Output:

parent_owners
mode_side_assignments
owned_parent_mass_by_path
owned_parent_bins_by_path
owned_side_bins_by_path
owned_transition_bins_by_path
owned_boundary_bins_by_path

Parent ownership variables, not child witnesses, enter residual cost and the owned-mass tie stage.

7.2 New invariants
input_point_count_le_200000
compact_parent_count_le_4000
compact_mode_count_le_16000
compact_parent_plus_mode_count_le_20000
compact_mode_edge_count_le_200000
mode_children_per_parent_le_4
mode_memberships_per_child_le_3

source_point_coordinates_not_serialized
source_point_indices_not_serialized
transient_point_partition_is_disjoint
parent_mass_counted_exactly_once

projective_normal_sign_invariant
canonical_ids_input_order_invariant
mode_edges_reference_existing_modes
plane_memberships_reference_existing_planes

bend_side_witnesses_use_distinct_modes
transition_rungs_use_distinct_plane_modes
perimeter_has_no_transition_rung
junction_central_parent_has_one_owner

incidence_hashes_match_support_tables
exact_incidence_not_truncated
mode_overflow_disables_hard_transition_certificate
gold_fields_absent_from_inference_graph
split_group_fields_absent_from_inference_graph
birth_columns_disabled_for_A1
7.3 Unit fixtures

Implement compact synthetic fixtures:

wedge_single_parent
    Two planes enter one coarse voxel; two modes must survive.

flat_panel_free_edge
    One mode plus persistent boundary void; no bend certificate.

normal_sign_flip
    Randomly flip all point-normal signs; graph bytes remain identical.

parallel_double_fold
    Two disjoint, parallel two-sided ladders remain separate.

trihedral_junction
    Three modes and branch degree >= 3; junction, not three duplicate bends.

short_return_three_bins
    Minimal V2A run remains representable.

one_sided_occlusion
    One mode only; no counted bend or V2A birth.

mode_overflow_five_surfaces
    Overflow is explicit and cannot hard-certify a bend.

duplicate_candidates_same_parents
    At most one object owns the shared physical support.
7.4 Property and regression tests
input point permutation leaves V2 JSON byte-identical
normal sign flips leave V2 JSON byte-identical
mode child ordering leaves physical path ID unchanged
one point cannot enter two mode children
two modes in one parent do not double parent mass
one child cannot satisfy both bend sides
two distinct children can satisfy both bend sides
cross-parent transition rung is accepted
same-normal parallel surfaces are not same-surface-linked across a gap
parallel ladders are not hard duplicates
overflow parent cannot be the only transition witness
V2 birth certificate is accepted by the exact solver
V1-style incomplete birth certificate is rejected
safe lower bound never prunes an improving exhaustive fixture
no JSON output recursively contains point_indices
all compact caps are enforced without silent truncation
metadata mutations leave inference graph hash unchanged
three complete runs are byte-identical

Memory regressions should measure:

V2 builder peak RSS
candidate-pool peak RSS
graph builder peak RSS
exact solver peak RSS
serialized pool size
parent/mode/edge counts

The existing 16 GiB limit remains a hard failure, not a target to tune toward.

7.5 Diagnostic probe before full A1

The evidence is sufficient to choose mode children as the smallest falsifiable design, but not sufficient to assume they solve the problem. Run a diagnostic-only mode probe with no prediction changes.

Proceed with the complete A1 solver patch only when post-lock analysis shows:

at least 14 of the 70 candidate-unrepresented gold objects
gain a valid two-mode connected run,

those objects occur across at least 8 independent family groups,

and no more than 25% of the 33 A0 false predictions
pass the same hard transition certificate.

Failure means the next work should target plane extraction or a genuinely finer spatial representation, not complete the parent/mode ownership implementation.

7.6 Mandatory stop milestone

The implementation milestone is:

M1: V2 parent/mode construction, compact serialization, V2 incidence, and exact parent/mode ownership are complete; births and learned potentials remain disabled.

At M1:

all fixtures and property tests pass;

cache and pool contain no point indices or source points;

three replays of every smoke case are byte-identical;

measured jobs remain below 16 GiB;

run the full Gold37 A1 batch exactly once;

generate the inference-safe census;

freeze all outputs;

generate post-lock stage attribution;

apply the A1 GO/STOP criteria.

Do not add BIRTH_V2A, fit a potential, alter a threshold, or implement another birth family until that decision is recorded.

Runtime/main promotion remains prohibited.
