1. Exclusive ownership with zero-incidence paths

OBSERVED. The formulation is exact only for the explicitly serialized incidence relation, not for the stronger claim that every selected semantic object owns scan support. 15_solve_f1_structured_union_cpsat.py:235–240 deliberately permits a non-residual path with no incidence, because the minimum-ownership constraint is created only for paths present in owned_by_path. The solution reports three such selected paths: LP_a8702afb6ca4526d, LP_b8af5a56cc3a8d78, and LP_cc110572e7f31617. It also reports 1,294 serialized atoms but the graph diagnostic shows only 780 atom IDs incident to any path; 514 atoms therefore have no possible non-residual owner. 

04_exact_cpsat_solution

 

15_solve_f1_structured_union_cp…

 

03_typed_union_graph

Answer. It is mathematically honest only under the narrow output qualifier:

exclusive_sparse_support_ownership_scope =
    "serialized_path_incidence_only"

It is not yet an honest implementation of the prior formulation “every non-residual object must own class-specific scan support.” A better name for the present property is:

exclusive_ownership_over_partial_incidence

The smallest scan-only repair is not a new support detector. It is deterministic incidence completion from the already serialized atoms, followed by a hard prohibition on unsupported semantic selections.

Minimal completion

Use two stages, applied uniformly to every path.

Stage A: propagate existing incidence across equivalent geometry

A path with no atom IDs may inherit an atom ID only from a raw candidate or latent path that:

already carries that atom ID;

is geometrically duplicate-equivalent;

passes an atom-to-target-path verification.

Let s be the median positive local_spacing_mm in the case. For source path j and target path i:

r
θ
	​

(i,j)=
4
∘
∠(u
i
	​

,u
j
	​

)
	​

r
⊥
	​

(i,j)=
max(3 mm,2s)
d
sym-line
	​

(i,j)
	​

r
ov
	​

(i,j)=
0.25
0.75−overlap
min
	​

(i,j)
	​


The paths are incidence-equivalent only when:

r_theta <= 1
r_perp  <= 1
shorter_span_overlap >= 0.75

These are scanner/geometry tolerances, frozen before family validation. They are not chosen from 457 truth. In the current 457 pool, this propagation route is available because the three unsupported selected paths have geometrically equivalent atom-bearing bend-hypothesis candidates; the completion should nevertheless be written generically and tested across all families.

Stage B: verify or complete directly from compact atoms

For target segment p
0
	​

,p
1
	​

, unit axis u, length L, and atom a with centroid c
a
	​

, normal n
a
	​

, optional local axis v
a
	​

, and spacing s
a
	​

:

t
a
	​

=(c
a
	​

−p
0
	​

)⋅u
d
⊥,a
	​

=∥c
a
	​

−(p
0
	​

+clip(t
a
	​

,0,L)u)∥
d
end,a
	​

=max(0,−t
a
	​

,t
a
	​

−L)

Use:

τ
⊥,a
	​

=min(3 mm,max(1.5 mm,1.25s
a
	​

))
τ
end,a
	​

=min(6 mm,max(3 mm,2s
a
	​

))
r
⊥,a
	​

=d
⊥,a
	​

/τ
⊥,a
	​

r
end,a
	​

=d
end,a
	​

/τ
end,a
	​

r
normal,a
	​

=
20
∘
arcsin(∣n
a
	​

⋅u∣)
	​


When axis_direction is valid:

r
axis,a
	​

=
15
∘
∠(v
a
	​

,u)
	​


An atom is geometrically incident when:

r_perp   <= 1
r_end    <= 1
r_normal <= 1
and (atom axis unavailable or r_axis <= 1)

The inherited IDs from Stage A must pass this same test; inheritance alone is not sufficient.

Build the induced subgraph from support_atom_edges. Project eligible atoms onto u with bin width:

w=max(3 mm,2s)

Retain only connected support components containing at least two occupied axial bins and a run of at least two consecutive bins. This is the generic pre-learning minimum; the class-specific minima replace it in Question 4.

No new atoms, normals, points, panel IDs, or support edges are created. The procedure only completes the incidence relation between existing compact atoms and existing paths.

Required graph fields
Python
Run
path["support_atom_ids"] = completed_ids
path["support_bin_ids"] = completed_bin_ids
path["incidence_complete"] = True
path["incidence_origin"] = {
    "upstream_verified": [...],
    "equivalent_path_propagation": [...],
    "compact_atom_completion": [...],
}
path["incidence_certificate"] = {
    "tolerance_profile": "INCIDENCE_V1",
    "max_dimensionless_residuals": {
        "perpendicular": ...,
        "endpoint": ...,
        "normal": ...,
        "axis": ...,
    },
}

The tolerance profile and atom-table hash must enter graph_sha256.

Solver constraint

Replace the present one-atom conditional with:

Python
Run
non_residual_i = sum(
    z[i, c] for c in LABELS if c != RESIDUAL
)

if not path["incidence_complete"]:
    model.Add(non_residual_i == 0)
else:
    model.Add(
        sum(bin_owned[i, b] for b in path["support_bin_ids"])
        >= GENERIC_MIN_BINS * non_residual_i
    )

And reject any exact solve where:

Python
Run
path["support_atom_ids_truncated"] is True

unless incidence was recomputed from the complete serialized atom table. Because the table cap is 20,000 and the graph path cap is 32,768, the graph layer itself has no reason to retain a truncated exact-incidence relation.

INFERENCE. This is the smallest defensible repair because it preserves the existing atoms, graph, candidate pool, and solver. It does not use the 457 K(2,4) topology, expected count, or post-lock labels.

UNKNOWN. Compact-atom proximity proves that support exists near a path; it does not by itself prove that the support is a bend rather than an outline. That distinction requires class-conditioned side ownership and the outline/perimeter ablation in Question 6.

2. Deterministic continuation proposals

OBSERVED. The solver correctly accepts continuation records, enforces the same primary_panel_pair, restricts each endpoint to at most one selected continuation, and requires the proposal graph itself to be a forest. But the graph contains no continuation proposals, and _continuation_records() validates no geometric certificate beyond panel-pair compatibility. 

03_typed_union_graph

 

15_solve_f1_structured_union_cp…

Proposal algorithm

Only paths satisfying all of these are considered:

incidence_complete == true
support_atom_ids is nonempty
primary_panel_pair is a valid scan-derived pair
straight-line eligibility is true
not a hard duplicate of another fragment

Let s again be the case median atom spacing. Freeze:

TAU_AXIS_DEG       = 4.0
TAU_CHORD_DEG      = 6.0
TAU_PERP_MM        = max(3.0, 2*s), capped at 6.0 mm
TAU_GAP_MM         = max(6.0, 4*s), capped at 12.0 mm
AXIAL_BIN_MM       = max(3.0, 2*s)
MAX_EMPTY_BINS     = 1
MAX_OVERLAP_MM     = max(2*AXIAL_BIN_MM, 0.10*min(L_i, L_j))

The caps are engineering tolerances and must be frozen before the outer family folds.

Step 1: orientation families

Within each identical panel pair, build complete-link orientation families under:

∠(u
i
	​

,u
j
	​

)≤4
∘

For each family, compute a canonical signed axis:

Python
Run
u_family = normalize(weighted_sum(canonical_sign(u_i)))

where the weight is owned support mass, not span or source count.

The family sign is selected by the first nonzero component in x,y,z, so coordinate transforms and endpoint reversal do not alter ordering.

Step 2: one-dimensional intervals

For every fragment:

Python
Run
lo_i = min(dot(endpoint, u_family))
hi_i = max(dot(endpoint, u_family))

Sort by:

(
    primary_panel_pair,
    quantized u_family,
    lo_i,
    hi_i,
    latent_path_id
)
Step 3: pair eligibility

For an end-to-start pair i→j, with lo_j >= lo_i, define:

r
θ
	​

=
4
∘
∠(u
i
	​

,u
j
	​

)
	​

r
⊥
	​

=
τ
⊥
	​

max(d(mid(i),ℓ
j
	​

),d(mid(j),ℓ
i
	​

))
	​

g
ij
	​

=max(0,lo
j
	​

−hi
i
	​

)
r
gap
	​

=g
ij
	​

/τ
gap
	​


Let q
i
	​

 be the relevant endpoint of i, q
j
	​

 the relevant endpoint of j:

r
chord
	​

=
6
∘
∠(q
j
	​

−q
i
	​

,u
family
	​

)
	​


Let C
ij
	​

 be the union of the two completed incidence sets. Refit a weighted TLS line and compute:

r
fit
	​

=
τ
⊥
	​

P
90
	​

(d(c
a
	​

,ℓ
TLS
	​

),a∈C
ij
	​

)
	​


Let:

o
ij
	​

=max(0,min(hi
i
	​

,hi
j
	​

)−max(lo
i
	​

,lo
j
	​

))
r
overlap
	​

=
max(2w,0.1min(L
i
	​

,L
j
	​

))
o
ij
	​

	​


The proposal is rejected when r_overlap > 1. This is the main safeguard against joining distinct parallel bends that overlap along their lengths.

Step 4: support-bridge certificate

Restrict the atom graph to atoms that:

pass the completed incidence tube for the refitted line;

project between hi_i - w and lo_j + w;

are compatible with the same panel pair.

Require a graph path from an atom in the last two occupied bins of i to an atom in the first two occupied bins of j.

After converting the bridge atoms to axial bins, require:

maximum consecutive empty bins <= 1

This bridge is a certificate, not a score. Without it, do not propose a continuation in the first implementation. That deliberately favors precision over recovery under heavy occlusion.

Step 5: mutual-nearest construction

For each right endpoint of i, choose the eligible j with the smallest canonical tuple:

(
    r_gap,
    r_perp,
    r_theta,
    r_fit,
    r_chord,
    latent_path_id_j
)

For each left endpoint of j, independently choose its best predecessor. Emit i -> j only when the choice is mutual.

Store:

JSON
{
  "continuation_id": "CONT_<geometry hash>",
  "path_ids": ["LP_i", "LP_j"],
  "endpoint_ids": [1, 0],
  "dimensionless_residuals": {
    "axis": 0.0,
    "perpendicular": 0.0,
    "gap": 0.0,
    "chord": 0.0,
    "refit": 0.0,
    "overlap": 0.0
  },
  "engineering_tolerance_profile": "CONTINUATION_V1",
  "bridge_atom_ids": [],
  "bridge_certificate_sha256": "..."
}
Acyclicity argument

Every emitted edge increases the scalar coordinate along the canonical family axis. Mutual-nearest selection gives each left endpoint at most one predecessor and each right endpoint at most one successor. Therefore each proposal component is an ordered chain. A directed cycle would require strictly increasing projection around the cycle and returning to its starting projection, which is impossible.

This is stronger and easier to audit than generating a dense cyclic proposal graph and relying on cycle cuts.

Solver validation patch

Extend _continuation_records():

Python
Run
required = {
    "axis",
    "perpendicular",
    "gap",
    "chord",
    "refit",
    "overlap",
}
residuals = raw["dimensionless_residuals"]

if set(residuals) != required:
    raise ValueError("incomplete continuation residual certificate")

if any(not math.isfinite(float(residuals[k])) for k in required):
    raise ValueError("non-finite continuation residual")

if any(float(residuals[k]) > 1.0 for k in required):
    raise ValueError("continuation outside frozen engineering tolerance")

if not raw.get("bridge_certificate_sha256"):
    raise ValueError("continuation lacks scan-support certificate")

INFERENCE. Fragmented bends should join when they are end-to-end fragments of the same supported track. Overlapping parallel bends, duplicated skins, and frame outlines should not join because they fail the overlap or bridge requirements.

UNKNOWN. A true bend split by a large occlusion may have no atom bridge and will remain fragmented. That is an explicit high-precision limitation of the first proposal generator; it should be measured before permitting a visibility-based exception.

3. First residual-birth column-generation slice

OBSERVED. A meaningful reduced cost cannot be computed under the current primary objective. primary_energy contains only topology-factor activation. Residual ownership is maximized in a later tie stage, not penalized in the primary energy, while birth variables and birth costs do not exist. 

15_solve_f1_structured_union_cp…

 

15_solve_f1_structured_union_cp…

The first implementation therefore needs a small, engineering-fixed MDL layer before it adds birth columns.

Fixed pre-learning MDL terms

Normalize atom area to bounded dimensionless support mass:

q
a
	​

=clip(round
median
a
′
	​

area
a
′
	​

area
a
	​

	​

,1,4)

Use integer microbits:

Python
Run
C_RESIDUAL = ENERGY_SCALE       # one bit per normalized support unit

For N existing paths and K deterministic birth proposals:

C
OBJECT
	​

=SCALE(1+⌈log
2
	​

(N+K+1)⌉)

For the first birth slice, only BEND_STRAIGHT_COUNTED births are generated, so no learned class prior is required. The extra 1 is the generic object-existence code; the proposal index encodes its deterministic geometry.

The primary energy gains:

Python
Run
primary_energy += sum(
    C_OBJECT * z[i, c]
    for i in paths
    for c in LABELS
    if c != RESIDUAL
)

primary_energy += sum(
    C_RESIDUAL * atom_mass[a] * o_residual[a]
    for a in atoms
)

All fixed geometric fit costs are nonnegative. Therefore they can be omitted from an optimistic lower bound without making the bound unsafe.

Birth eligibility

Generate births only from atoms owned by residual in the current optimum.

For every unordered pair of serialized planes f,g:

Require plane-normal angle:

20
∘
≤∠(n
f
	​

,n
g
	​

)≤160
∘

Form the scan-only intersection axis:

u
fg
	​

=
∥n
f
	​

×n
g
	​

∥
n
f
	​

×n
g
	​

	​


Select residual atoms compatible with the intersection tube using the same INCIDENCE_V1 residuals from Question 1.

Assign each atom to side f or g by the smaller combination of:

unoriented normal angle to the plane normal;

distance to the plane.

Require side-normal residual:

15
∘
∠(n
a
	​

,n
f
	​

)
	​

≤1or
15
∘
∠(n
a
	​

,n
g
	​

)
	​

≤1

Split the induced residual atom graph into connected components.

For each component, project onto u
fg
	​

, use:

birth_bin_mm = max(3 mm, 2*s)

and require all of:

occupied bins total             >= 6
bins containing f-side support  >= 3
bins containing g-side support  >= 3
bins containing both sides      >= 3
maximum empty run in hull       <= 1
span                            >= max(12 mm, 6*s)
p90 TLS line residual           <= tau_perp
axis delta from plane crossing  <= 4 degrees

Reject a proposal when an existing latent path satisfies:

axis angle               <= 4 degrees
transverse offset        <= max(3 mm, 2*s)
shorter-span overlap     >= 0.75

That is a duplicate-completion case, not a missing-object birth.

Reject components whose axial-bin adjacency branches internally. Endpoint branching is allowed only in the first and last bin; otherwise the proposal is a junction candidate, not a straight bend birth.

Require two independent scan modalities:

the two plane-normal support modes;

connected atom-graph support across the axial run.

A high scalar bend score alone is not eligible.

Reduced-cost lower bound

Let R
⋆
 be the residual atoms in the current optimum and I
k
	​

 the eligible incidence of birth proposal k. Define:

S
k
	​

=C
RESIDUAL
	​

a∈I
k
	​

∩R
⋆
∑
	​

q
a
	​


Before learned continuation rewards exist, the most optimistic energy change is:

LB(k)=C
OBJECT
	​

−S
k
	​


The actual change satisfies:

ΔE(k)≥LB(k)

because:

the proposal cannot save more residual cost than S
k
	​

;

fit, class, conflict, and ownership costs are nonnegative;

continuation contributes no negative primary reward in this first slice.

Therefore:

LB(k) >= 0  => safely do not add the column
LB(k) < 0   => add the column; CP-SAT still decides whether to select it

This is an optimistic integer-energy bound, not an LP dual claim.

Deterministic iteration
Python
Run
TOTAL_BIRTH_COLUMN_CAP = 64

while True:
    solution = exact_solve(graph)

    proposals = generate_births_from_current_residual_atoms(
        solution,
        canonical_plane_pair_order=True,
    )

    proposals.sort(
        key=lambda k: (
            k.lower_bound_int,
            tuple(k.panel_pair),
            tuple(k.quantized_axis),
            tuple(k.quantized_midpoint),
            k.span_mm,
            k.proposal_hash,
        )
    )

    negative = [k for k in proposals if k.lower_bound_int < 0]

    if not negative:
        break

    available = TOTAL_BIRTH_COLUMN_CAP - total_added
    if available <= 0:
        break

    graph.add_birth_columns(negative[:available])

Every birth variable must bound all of its semantic, ownership, and continuation variables:

Python
Run
z_birth[k, BEND] <= b[k]
owner[a, k]      <= b[k]
continuation[e]  <= b[k]

There is no requested count, no missing-object oracle, and no “add one because the count is low” branch.

INFERENCE. This slice should recover planar, straight, two-sided folds that are present in the compact residual atom graph but absent from all candidate lanes.

UNKNOWN. It will not recover a bend when plane extraction failed, when support is one-sided through occlusion, or when the physical object is rolled/nonplanar. Those require later birth families and cannot be inferred from the current 27-family evidence.

4. Annotation and nested family-held-out learning

OBSERVED. The fresh PLY-only slice had 13/13 successful jobs but zero manually acceptable outputs. The dominant failures were outlines/free edges, duplicates, floating spans, and missed visible bends. The CAD+PLY slice reached only 3/12 exact. The enterprise promotion report remains blocked on semantic accuracy, truth-set size, holdout size, and family coverage. 

11_blind_ply13_summary

 

12_blind_cadply12_summary

 

13_manual_visual_audit

 

13b_enterprise_promotion_gates

Minimum annotation contract

One record per scan:

JSON
{
  "family_id": "opaque-family-group",
  "scan_id": "opaque-scan-id",
  "visibility": {
    "scan_limited": false,
    "unobservable_regions": []
  },
  "objects": [],
  "atom_ownership": [],
  "candidate_links": []
}

Each object must contain:

JSON
{
  "object_id": "GT_O1",
  "semantic_label": "BEND_STRAIGHT_COUNTED",
  "counted": true,
  "geometry_kind": "LINE",
  "geometry": {
    "span_endpoints": [[0, 0, 0], [0, 0, 0]]
  },
  "incident_panel_ids": ["P1", "P2"],
  "endpoint_types": ["ORDINARY_END", "ORDINARY_END"],
  "continuation_component_id": "GT_COMPONENT_1",
  "visibility_state": "DIRECTLY_OBSERVED",
  "manufacturing_subtype": "UNKNOWN",
  "adjudication": {
    "reviewers": 2,
    "agreement": "AGREED",
    "confidence": "HIGH"
  }
}

geometry_kind must support:

LINE       — straight bend/perimeter/corner fragment
POLYLINE   — curve
LOOP       — hole
NODE       — junction-dominated support

Each compact atom must be assigned:

JSON
{
  "atom_id": "A123",
  "owner_object_id": "GT_O1",
  "side": "P1",
  "state": "OWNED"
}

Permitted states:

OWNED
RESIDUAL
AMBIGUOUS
JUNCTION_SHARED
UNOBSERVABLE

Atom ownership can be proposed automatically from the annotated object region, but ownership conflicts, non-counted objects, and all birth/miss cases require human review.

Each truth object also needs candidate coverage:

JSON
{
  "object_id": "GT_O1",
  "matching_candidate_ids": [],
  "candidate_recall_state": "NO_CANDIDATE",
  "best_candidate_source_groups": []
}

This separates selector failures from birth failures.

Nested leave-one-family-out plan

For outer family F
o
	​

:

Hold out all scans and object instances from F
o
	​

.

Use the other 26 families for training.

Within those 26, run inner leave-one-family-out model selection.

Freeze all selected parameters.

Evaluate once on F
o
	​

.

Aggregate metrics by family, not by candidate or atom.

The fresh 13-scan set should not silently become training data. Once fully typed, it must be declared either:

additional development families, losing locked-holdout status; or

a locked external confirmation set.

It cannot serve both roles.

Class-specific ownership minima

Select each minimum from a preregistered small grid:

candidate grid in occupied axial bins: {2, 3, 4, 6}

Parameters:

m_BEND
m_CORNER
m_PERIMETER
m_HOLE
m_JUNCTION
m_CURVE

When a class appears in fewer than five training families, it does not receive an independent parameter. Tie it to an ontology group:

two-panel path: BEND, CORNER, CURVE
one-boundary path: PERIMETER, HOLE
junction support: JUNCTION

The inner objective should maximize family-macro semantic object F1 subject to a bend-recall floor, rather than maximizing candidate-level accuracy.

Low-capacity potentials

Use a sign-constrained linear or monotone additive model. Start with these normalized feature groups:

1. completed owned-bin count
2. completed owned mass
3. two-sided balance
4. one-sided fraction
5. line-fit residual
6. finite-run fraction
7. endpoint/junction degree
8. loop-closure residual
9. tangent/curvature variation
10. independent source-group count

Keep at most 12 effective slope parameters after grouping, plus label intercepts. Do not create candidate-source-by-label interactions with 27 families.

A suitable initial energy is:

U(i,c)=β
c
	​

+
g
∑
	​

w
c,g
	​

ϕ
i,g
	​


with strong L
2
	​

 or group regularization and sign constraints such as:

BEND:
  two-sided balance       cannot hurt
  one-sided fraction      cannot help
  line residual           cannot help

PERIMETER:
  one-sided fraction      cannot hurt
  two-sided balance       cannot help

HOLE:
  loop closure            cannot hurt

JUNCTION:
  endpoint degree         cannot hurt

CURVE:
  tangent variation       cannot hurt
  straight-line fit       cannot help

Every family contributes equal total training weight regardless of how many candidates or atoms it contains.

Only after the unary model is stable should no more than four or five global structured weights be fit:

continuation
duplicate conflict
residual ownership
birth code adjustment
generic corner/chord relation

The present K(2,4) factor cannot support a learned weight: it was designed with 457 visible and activates on none of the Gold-27 development scans. Its weight should remain zero outside the explicit 457 regression until other families contain independently annotated instances of the same relation.

Engineering-fixed parameters

These must not be selected by inner cross-validation:

200,000-point working-set contract
deterministic sampling and atomization
atom adjacency construction
plane extraction
INCIDENCE_V1 tolerances
CONTINUATION_V1 tolerances
birth eligibility geometry
candidate and birth budgets
ontology and geometry kinds
integer energy scale
solver version and CP-SAT settings
canonical ordering
evaluation line contract
family partition
feature definitions and sign constraints

Learnable:

class ownership minima from the tiny grid
low-capacity unary coefficients
small number of structured weights
cross-fitted ambiguity-margin calibration

Report per outer fold:

candidate recall before selection
birth recall
line precision/recall/F1
exact counted-object count
seven-class confusion
outline/free-edge counted-FP rate
duplicate rate
ownership accuracy
deterministic replay rate

UNKNOWN. The number of independently represented families for perimeter, hole, junction, and curve is not established in the supplied package. Some classes may need to remain engineering-only or tied until more families are annotated.

5. Code audit
5.1 No direct truth leakage found, but the boundary is too permissive

OBSERVED. build_structured_union_graph() checks the declared inference flags and does not use case_id or scan_path in feature construction. The metadata-mutation test reverses candidates, changes those fields, and reverses endpoints while requiring identical graph output. That is a useful leakage regression. 

18_test_f1_structured_union_gra…

However, the candidate-pool builder writes case_id, scan_path, and coordinate_frame into every output at 16_build_f1_scan_only_candidate_pool.py:849–851. The graph currently ignores them, but future code can accidentally consume them. 

16_build_f1_scan_only_candidate…

Patch.

Construct an explicit sanitized view at the graph boundary:

Python
Run
ALLOWED_POOL_FIELDS = {
    "schema_version",
    "algorithm",
    "inference_contract",
    "source_point_serialization",
    "support_atom_table",
    "support_atom_edges",
    "plane_table",
    "candidates",
    "line_objects",
}

def strict_geometry_view(pool: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(pool) - ALLOWED_POOL_FIELDS
    # Metadata may exist in storage, but never enters the graph object.
    return {key: pool[key] for key in ALLOWED_POOL_FIELDS if key in pool}

Use a nested whitelist for candidate and line-object fields. Do not rely solely on self-reported booleans.

5.2 The strict global 200k-point bound is not currently demonstrated

OBSERVED. 02_candidate_pool_v2.json records:

transient_sampled_point_count = 268623

_load_geometry_views() loads and sorts the complete cloud. Individual cylindrical and trimming routines impose their own 200k budgets, but the builder itself still holds more than 200k points, and the scan-mesh lane reads the complete mesh. 

02_candidate_pool_v2

 

16_build_f1_scan_only_candidate…

Patch.

Immediately after finite filtering and before sorting:

Python
Run
points, normals = deterministic_geometry_sample(
    points,
    normals,
    max_points=200_000,
    key="quantized_xyz_normal",
)

Derive the plane cloud, finite-support tree, local-support features, and cylindrical generators from this single bounded cloud.

When mesh adjacency uses more than 200,000 scan vertices, either:

deterministically decimate it to the same cap; or

document that the 200k contract excludes mesh vertices.

The current evidence supports the per-generator interpretation, not a global in-memory point bound.

5.3 Path identity is source-dependent

_stable_path_id() at 14:312–323 hashes independent_source_groups. The same physical geometry can therefore receive a different latent ID when a generator is enabled or disabled.

Patch.

Python
Run
canonical = {
    "geometry_kind": path["geometry_kind"],
    "axis_direction": quantize(path["axis_direction"]),
    "midpoint": quantize(path["midpoint"]),
    "span_mm": quantize(path["span_mm"]),
    "incident_panel_pairs": path["incident_panel_pairs"],
    "completed_support_signature": hash_sorted(path["support_atom_ids"]),
}

Keep source groups as evidence, never identity.

5.4 Panel namespaces are mixed

_incident_panel_pairs() at 14:203–222 places incident_plane_ids, incident_flange_ids, and parsed plane-lattice IDs into one set. An F1 and an RP1 are not interchangeable panel identities.

Patch.

JSON
{
  "incident_plane_pairs": [["PLANE:RP1", "PLANE:RP5"]],
  "incident_flange_pairs": [["FLANGE:F1", "FLANGE:F5"]],
  "primary_panel_pair_namespace": "PLANE"
}

Cross-namespace equivalence must be a separately certified relation, not string co-membership.

Also remove _member_plane_pair()’s PLI_<left>_<right> parser as an inference fallback. It fails when IDs contain underscores and makes identity depend on naming syntax. Require the structured incident_plane_ids field.

5.5 Longest-member geometry is selected before optimization

At 14:380–418, the graph chooses the longest eligible member geometry as the latent representative. At 14:570–584, _best_paths_by_panel_pair() chooses exactly one path per panel pair using source count, transition score, span, and ID.

This is pre-solver selection. It contradicts joint lane fusion and can prefer a zero-incidence path over a support-backed equivalent. 

14_f1_structured_union_graph

 

14_f1_structured_union_graph

Patch.

Do not collapse all alternatives to one path per panel pair. Retain:

latent object hypothesis
  -> alternative geometry representatives
  -> shared duplicate/equivalence group

Introduce one object-activation variable and one representative-choice variable:

Python
Run
sum(rep[i, r] for r in representatives[i]) == active[i]

Alternatively, retain paths separately with a hard duplicate constraint and let ownership choose one.

5.6 The K(2,4) factor is 457-designed logic

apply_relational_chord_corner_factor() at 14:607–722:

explicitly describes a “plug-box factor”;

defaults to four common neighbors;

applies a fixed 2.5 span ratio;

chooses only the highest-scoring hypothesis;

directly writes eight paths as bends and internal paths as corners.

The test suite then hard-codes the exact 457 RP pairs and post-lock 8/0/0 outcome. That is a valid designed regression but not evidence of a generic structured inference rule. 

14_f1_structured_union_graph

 

18_test_f1_structured_union_gra…

 

05_postlock_step_evaluation

Patch.

Rename the current function:

Python
Run
build_biclique_chord_regression_factors()

It should emit all hypotheses as relational factors without mutating path labels:

JSON
{
  "factor_id": "...",
  "factor_kind": "BICLIQUE_CHORD",
  "label_implications": [
    {"path_id": "...", "label": "BEND_STRAIGHT_COUNTED"},
    {"path_id": "...", "label": "CORNER_NONCOUNTED"}
  ],
  "feature_vector": {
    "common_neighbor_count": 4,
    "topology_coverage": 0.6667,
    "span_ratio": 2.9352
  },
  "weight_status": "DISABLED_PENDING_FAMILY_EVIDENCE"
}

Keep the exact 457 test under a path such as:

tests/regression/45796000/

Do not include it in generalization acceptance.

5.7 The solver does not operationalize the seven-label ontology

At 15:192–204, every path not referenced by a factor is forced to residual. Factor members can only be the factor-specified bend or corner, plus residual. Consequently:

PERIMETER_NONCOUNTED
HOLE_NONCOUNTED
JUNCTION_NONCOUNTED
CURVE_NONCOUNTED

can never be selected.

Further, 14 canonicalizes every candidate to one straight segment. A hole loop and a curve are not representable by the current path geometry.

Patch.

The graph must provide:

JSON
{
  "geometry_kind": "LINE",
  "label_eligibility": {
    "BEND_STRAIGHT_COUNTED": true,
    "CORNER_NONCOUNTED": true,
    "PERIMETER_NONCOUNTED": true,
    "HOLE_NONCOUNTED": false,
    "JUNCTION_NONCOUNTED": false,
    "CURVE_NONCOUNTED": false,
    "RESIDUAL_NONCOUNTED": true
  }
}

The solver should enforce eligibility, not factor membership:

Python
Run
for label in LABELS:
    if not path["label_eligibility"][label]:
        model.Add(z[path_id, label] == 0)

Do not claim hole or curve classification until loop/polyline proposals exist.

5.8 Endpoint types are not solver variables

endpoint_types is static metadata written by the 457 chord function at 14:705–709. CP-SAT contains no endpoint-type variable and does not enforce that a corner has two multiway endpoints.

Patch.

Python
Run
j[path_id, endpoint, node_type] ∈ {0,1}
model.AddExactlyOne(j[path_id, endpoint, t] for t in NODE_TYPES)

model.Add(
    j[path_id, 0, MULTIWAY] == 1
).OnlyEnforceIf(z[path_id, CORNER])

model.Add(
    j[path_id, 1, MULTIWAY] == 1
).OnlyEnforceIf(z[path_id, CORNER])

Continuation incidence must also imply an endpoint type compatible with continuation.

5.9 Global AtMostOne over topology factors is invalid

At 15:189–190, at most one topology factor may activate anywhere in the scan. A part can contain multiple disjoint corner systems, holes, or local structured regions.

Patch.

Remove the global constraint. Generate pairwise conflict constraints only when two factors:

assign contradictory labels to the same path;

demand the same exclusive support atom;

or assert incompatible node types.

Python
Run
for h1, h2 in conflicting_factor_pairs:
    model.Add(h[h1] + h[h2] <= 1)

Disjoint factors may activate together.

5.10 Ownership is too weak and ignores atom mass

At 15:206–240:

zero-incidence paths can be selected;

one atom satisfies a supported path;

atom weight and area_proxy_mm2 are ignored;

axial distribution is ignored;

class-specific minima are absent;

support_atom_ids_truncated is ignored.

The unit test verifies only that one atom cannot support two simultaneously required objects; it does not test spatially distributed ownership. 

15_solve_f1_structured_union_cp…

 

17_test_f1_structured_union_cps…

Apply Question 1 and add:

Python
Run
model.Add(
    sum(bin_active[i, b] for b in bins_i)
    >= class_min_bins[c]
).OnlyEnforceIf(z[i, c])

model.Add(
    sum(atom_mass[a] * owner[a, i] for a in incidence_i)
    >= class_min_mass[c]
).OnlyEnforceIf(z[i, c])

For a bend, add side minima:

Python
Run
model.Add(
    sum(side_bin_active[i, f, b] for b in bins_i)
    >= min_side_bins[BEND]
).OnlyEnforceIf(z[i, BEND])

model.Add(
    sum(side_bin_active[i, g, b] for b in bins_i)
    >= min_side_bins[BEND]
).OnlyEnforceIf(z[i, BEND])
5.11 Duplicate and shared-boundary constraints are not implemented

The graph appends strings such as:

duplicate_of:<path_id>

but the solver ignores relations. There is no generic duplicate variable, no panel-boundary arc ownership, and no shared-flange constraint beyond compact atom ownership.

Patch.

Python
Run
for i, j in hard_duplicate_pairs:
    model.Add(z[i, BEND] + z[j, BEND] <= 1)

for arc in panel_boundary_arcs:
    model.AddExactlyOne(
        arc_owner[arc, object_id]
        for object_id in compatible_objects[arc]
        + [RESIDUAL_OWNER]
    )

Inner/outer skin traces certified as the same physical bend must be representatives of one object, not two duplicate objects.

5.12 The primary objective largely encodes the 457 answer upstream

At 15:259–263, primary energy is only:

Python
Run
sum(factor_energy[f] * factor_var[f])

The one 457 factor has score 4.892046..., so its integer energy is -4,892,046. The reported second semantic energy is zero and the margin is exactly 4,892,046. The factor already contains the eight counted IDs and four corner IDs. Thus CP-SAT proves whether that preconstructed factor is feasible and preferable to factor-off; it does not independently infer the 8/4 assignment from unaries and ownership. 

04_exact_cpsat_solution

 

15_solve_f1_structured_union_cp…

The exact solver claim is technically true for the encoded restricted model. The output should say:

JSON
{
  "exact_restricted_model_optimality_proven": true,
  "complete_structured_model_implemented": false,
  "primary_objective_terms": [
    "relational_factor_activation"
  ]
}

Do not expose energy_margin_int as a calibrated confidence margin.

5.13 The final tie is deterministic in replay, but not a proved lexicographic tie

After fixing primary energy, the code maximizes the number of owned atom variables and then the number of continuations. It then calls:

Python
Run
model.Minimize(0)

and relies on fixed search for the final assignment. Ownership variables are not all included in the explicit decision strategy. Pinned replay is empirically deterministic, but this is not the stated sequential lexicographic proof. 

15_solve_f1_structured_union_cp…

Replace the last stage with exact feasibility fixing:

Python
Run
def fix_bool_lexicographically(
    model: cp_model.CpModel,
    ordered_variables: list[cp_model.IntVar],
    *,
    prefer: int,
) -> None:
    for variable in ordered_variables:
        trial = model.Clone()
        trial_var = trial.get_bool_var_from_proto_index(variable.Index())
        trial.Add(trial_var == prefer)

        status = solve_zero_objective(trial)

        if status == cp_model.OPTIMAL:
            model.Add(variable == prefer)
        elif status == cp_model.INFEASIBLE:
            model.Add(variable == 1 - prefer)
        else:
            raise RuntimeError("lex stage did not prove feasibility")

Apply in order:

1. canonical path labels
2. factor activations
3. continuation variables
4. birth variables
5. atom owners

For labels, test labels in a frozen ordinal order rather than fixing individual one-hot booleans arbitrarily.

Also replace:

Python
Run
owned_support_mass = sum(ownership_vars.values())

with normalized atom mass or owned-bin mass.

5.14 Continuation component counting needs endpoint-aware validation in both layers

The CP-SAT worker correctly indexes endpoint incidence by (path_id, endpoint_id). But count_bend_components() in 14:735–789 ignores endpoint IDs for mapping-form continuation records and tracks degree per path, not per endpoint.

Patch.

Python
Run
endpoint_degree: dict[tuple[str, int], int] = defaultdict(int)

left_endpoint, right_endpoint = continuation["endpoint_ids"]

endpoint_degree[(left, left_endpoint)] += 1
endpoint_degree[(right, right_endpoint)] += 1

if endpoint_degree[(left, left_endpoint)] > 1:
    raise ValueError(...)

Use one shared continuation validator in graph generation, solver input validation, and count reconstruction.

5.15 Missing validation and tests

Add tests for:

zero-incidence non-residual path is infeasible
completed incidence is invariant to source order
truncated incidence rejects exact solve
two disjoint factors may activate
contradictory factors conflict
perimeter/hole/junction/curve eligibility is real
weighted ownership differs from atom-count ownership
class-specific axial minima are enforced
invalid continuation residual certificate is rejected
parallel overlapping bends never receive continuation
exact lex output is invariant to variable creation order
birth lower bound never prunes an improving exact column
metadata whitelist excludes case_id and scan_path
global working point set is <= 200000

The current tests are good deterministic regressions, but the 457-specific pair test and STEP exactness test must not be counted as generalization evidence. 

18_test_f1_structured_union_gra…

Consolidated replacement flow
Python
Run
pool = strict_geometry_view(raw_pool)

paths = build_all_latent_paths_without_pair_preselection(pool)

paths = complete_all_path_incidence(
    paths=paths,
    raw_candidates=pool["candidates"],
    atoms=pool["support_atom_table"],
    edges=pool["support_atom_edges"],
    profile=INCIDENCE_V1,
)

label_eligibility = build_engineering_label_eligibility(paths)

duplicate_relations = build_duplicate_relations(paths)

continuations = generate_proof_friendly_continuations(
    paths,
    profile=CONTINUATION_V1,
)

topology_factors = generate_relational_factor_proposals(
    paths,
    enabled_factor_families=FROZEN_FACTOR_SET,
)

graph = StructuredGraph(
    paths=paths,
    label_eligibility=label_eligibility,
    duplicate_relations=duplicate_relations,
    continuations=continuations,
    topology_factors=topology_factors,
)

solution = exact_map_with_proved_lex_ties(graph)

graph = residual_birth_column_generation(
    graph,
    solution,
    profile=BIRTH_V1,
    total_column_cap=64,
)
6. Next two falsifiable ablations
Ablation A: outline/free-edge rejection

Target. The repeated manual failure in the blind slice: long exterior outlines and free edges selected as bends. The audit found zero visually acceptable scans even when raw or render-admissible counts numerically matched. 

13_manual_visual_audit

Arms

Use identical candidate pools, completed incidence, continuation settings, and solver. Disable the 457-specific chord factor in all arms.

A0 — completed incidence, no class-conditioned side ownership
A1 — A0 + bend two-sided ownership minima
A2 — A1 + explicit PERIMETER_NONCOUNTED unary/eligibility

A bend must own support attributable to both incident panel modes across its axial run. A perimeter may own one-sided boundary support. No length threshold may distinguish them.

Evaluation

Run nested family-held-out evaluation over all fully annotated families. Primary metrics:

counted outline/free-edge FP per scan
family-macro bend precision
family-macro bend recall
exact object count
perimeter-versus-bend confusion

After all parameters are frozen, run the existing 13-scan visual set once as an external regression.

GO criteria

Proceed to the next research stage only when all are satisfied:

>= 50% relative reduction in counted outline/free-edge FPs versus A0
<= 5 percentage-point absolute loss in bend recall
95% family-bootstrap lower bound for precision change > 0
no increase in hole-boundary counted FPs
100% deterministic replay
STOP criteria

Stop the approach and revisit atom/panel-side construction when any occurs:

outline reduction < 30%
bend recall loss > 5 percentage points
gains appear only on 457 or one family
the rule behaves as a span/length threshold
the external 13-scan set remains dominated by the same outline aliases

Falsification. If two-sided ownership does not separate visible folds from outlines without losing bends, the compact atoms or plane-side assignments are not semantically adequate.

Ablation B: missing-bend birth under family holdout

Target. Determine whether the residual-birth slice recovers a real object when every original candidate lane missed it.

Deterministic deletion protocol

Within each outer held-out family:

Select one directly observed truth bend per eligible scan using a hash of:

scan geometry hash;

canonical truth geometry.

Remove every raw candidate and line-object member matching that truth bend from all sources.

Retain the scan-derived atom table, atom edges, and plane table.

Verify candidate recall for the deleted object is exactly zero before solving.

Run:

B0: no residual birth;

B1: BIRTH_V1 column generation.

Run the undeleted scan through both arms as the false-birth control.

The deletion code may use truth only to construct the research perturbation. The inference process receives only the modified scan-only graph and is never told that one object was removed.

Metrics
deleted-object recovery rate under the frozen line contract
false births per undeleted control scan
birth precision
natural-miss recovery on objects already lacking candidates
exact count change
residual atoms reclaimed per accepted birth
deterministic replay
GO criteria
deleted-object recovery >= 50%
false births <= 0.25 per undeleted scan
family-macro bend precision loss <= 3 percentage points
at least 18 of 27 outer families improve or tie
100% deterministic replay
no accepted birth has a surviving matching source candidate
STOP criteria
recovery < 30%
false births > 0.5 per scan
births predominantly trace perimeter/hole support
recovery requires expected count or deletion identity
recovery occurs only when a sibling matching candidate remains
negative reduced-cost columns repeatedly fail to improve exact primary energy

Falsification. If the birth rule cannot recover deliberately removed bends while the residual atoms and planes remain, then the residual support representation does not contain enough object-forming information. Increasing model capacity would not be justified until that representation problem is resolved.

Final disposition: the 457 result should remain a designed regression test. The operational determinism and memory work are valuable, but the current exact model does not yet establish general scan-only semantic ownership. Runtime promotion remains prohibited.
