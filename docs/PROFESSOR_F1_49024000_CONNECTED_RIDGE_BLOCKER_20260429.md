# Professor Handoff: F1 Connected-Ridge Blocker on `49024000`

Date: 2026-04-29

## Short Version

We need guidance on a specific mathematical blocker in the offline F1 patch-graph decomposition lane.

The case is:

```text
scan: 49024000_A.ply
manual truth: 3 conventional bends
current raw F1: exact 4 owned bend regions
manual region review: only OB2 is a valid conventional bend
problem: two conventional bends are missing, while three raw F1 regions are false/fragment support
```

We tried the natural deterministic recovery sequence:

1. split existing owned bend regions;
2. mine current overcomplete interface-birth candidates;
3. propose new residual/flange-interface support;
4. sub-cluster the residual/interface support;
5. trace local curvature/normal ridges inside that support.

All five tests failed the visual promotion gate. The strongest signal now is one connected U/corner-shaped ridge/patch, not two separated missing bend supports.

The question is:

> How should we formulate the next object model or optimization so a connected transition/corner ridge can be decomposed into the correct number of conventional bend objects, without overcounting arbitrary fragments?

This seems to be the first case where local geometry is real but the current support-object ontology is too weak.

## Current Project Context

We have an offline-only F1 research lane. Runtime zero-shot remains unchanged.

The F1 lane currently uses:

- surface atoms as ownership units;
- flange hypotheses;
- bend hypotheses;
- residual/outlier labels;
- candidate-based MAP decomposition;
- object-level admissibility;
- duplicate/fragment suppression;
- scan-family routing;
- 1080p visual validation gates before promotion.

The conceptual model is:

```text
visible scan -> atom graph -> flange/bend/residual ownership -> connected admissible bend supports -> observed bend count
```

The current accepted/runtime path is still conservative. We are not asking whether to promote F1 yet. We are asking how to fix the next F1 formulation wall.

## Case Study: `49024000`

Manual review from Idan:

```text
49024000 is a 3-bend part.
Raw F1 OB2 is OK.
The other raw owned regions are wrong.
```

Current raw F1 output:

```json
{
  "exact_bend_count": 4,
  "bend_count_range": [4, 4],
  "owned_bend_region_count": 4
}
```

Manual region labels:

```text
OB1: rejected, large false support / fragment
OB2: accepted conventional bend
OB3: rejected small false support near OB2
OB4: rejected small cross-pair fragment
```

Label-capacity summary:

```json
{
  "truth_conventional_bends": 3,
  "valid_conventional_region_count": 1,
  "valid_conventional_region_deficit": 2,
  "valid_counted_feature_deficit": 2,
  "owned_region_capacity_deficit": 0
}
```

This distinction matters:

- raw F1 has enough regions numerically: 4;
- but only one region is a valid conventional bend;
- therefore the problem is not "count fewer raw regions";
- the problem is "replace false support with two missing true supports."

## Reproducible Artifacts

Main review folder:

```text
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427
```

Core case files:

```text
49024000_feature_region_labels.json
49024000_feature_region_summary.json
49024000_split_candidate_diagnostics.json
49024000_residual_interface_support_candidates.json
49024000_residual_interface_subclusters.json
49024000_local_ridge_candidates.json
49024000_residual_interface_visual_check_summary.md
49024000_local_ridge_visual_check_summary.md
```

Visual overlays:

```text
49024000_residual_interface_visual_check/residual_interface_candidates_overview_1080p.png
49024000_residual_interface_subcluster_visual_check/residual_interface_subclusters_overview_1080p.png
49024000_local_ridge_visual_check/local_ridge_candidates_overview_1080p.png
```

Current decomposition JSON:

```text
data/bend_corpus/evaluation/f1_visual_qa_after_debug_alias_patch_20260426/known/01_49024000_49024000_A/decomposition_result.json
```

Relevant diagnostic scripts:

```text
scripts/analyze_f1_owned_region_split_candidates.py
scripts/analyze_f1_exact_blocker_candidates.py
scripts/analyze_f1_residual_interface_support_candidates.py
scripts/cluster_f1_residual_interface_candidates.py
scripts/trace_f1_local_ridge_candidates.py
```

Core F1 solver:

```text
backend/patch_graph_latent_decomposition.py
```

## Diagnostic Sequence and Results

### 1. Existing Owned-Region Split Diagnostic

Question tested:

> Is one of the current raw F1 owned regions actually two bends merged together?

Result:

```json
{
  "status": "missing_region_recovery_needed",
  "capacity_deficit": 2,
  "owned_region_capacity_deficit": 0,
  "valid_conventional_region_deficit": 2,
  "split_candidate_count": 0
}
```

Interpretation:

- No existing owned region has a strong balanced along-axis gap.
- `OB1` is broad but not splitable by the current along-axis gap test.
- `OB2` is the accepted bend and should not be split.
- `OB3` and `OB4` are too small/fragmentary.

Decision:

```text
Reject "split the current OB set" as the fix.
```

### 2. Existing Overcomplete Interface-Birth Candidate Diagnostic

Question tested:

> Does the current overcomplete birth pool already contain the two missing supports?

We compared the visual-QA raw result against the previous `cap_6` overcomplete run.

Result:

```text
selection_status: no_admissible_candidate
corridor_coverage_status: no_corridor_gap
boundary_residual_ownership_status: no_uncovered_corridor
family_residual_repair_eligible: false
```

Interpretation:

- The current overcomplete candidate pool does not contain a clean missing-bend candidate.
- Loosening birth caps previously caused counts to rise to 5, 6, 8, then 15-16 instead of converging to 3.

Decision:

```text
Reject "relax current birth caps" as the fix.
```

### 3. Residual / Flange-Interface Support Proposal Diagnostic

Question tested:

> If we ignore current bend hypotheses and mine residual/flange-interface atoms, do the two missing supports appear?

Result:

```json
{
  "status": "ambiguous_overlapping_candidate_pool",
  "recovery_deficit": 2,
  "candidate_count": 237,
  "admissible_candidate_count": 4,
  "admissible_conflict_count": 6,
  "admissible_conflict_components": [
    ["RIC1", "RIC2", "RIC3", "RIC4"]
  ]
}
```

Top admissible candidates:

```text
RIC1 pair=[F11,F17] atoms=172 score=0.612449
RIC2 pair=[F1,F14]  atoms=191 score=0.597588
RIC3 pair=[F1,F17]  atoms=143 score=0.587421
RIC4 pair=[F11,F14] atoms=207 score=0.561758
```

Visual inspection:

- The candidates are not two separated supports.
- They are four competing descriptions of the same broad corner/transition zone.
- They overlap heavily in atom support and centroid location.

Decision:

```text
Reject residual/interface candidates as direct birth objects.
```

### 4. Residual / Interface Sub-Clustering

Question tested:

> If the broad residual/interface support contains two bends, can simple geometric sub-clustering separate them?

Result:

```json
{
  "status": "partial_subcluster_signal",
  "candidate_atom_count": 319,
  "connected_component_count": 1,
  "cluster_count": 1,
  "plausible_cluster_count": 1
}
```

Visual inspection:

- Still one broad support patch.
- Not two separated bend lines.

Decision:

```text
Reject simple sub-clustering as the fix.
```

### 5. Local Curvature / Normal Ridge Tracing

Question tested:

> If broad interface support is too permissive, can local curvature/normal discontinuity isolate real bend lines?

Result:

```json
{
  "status": "partial_ridge_signal",
  "domain_atom_count": 319,
  "ridge_atom_count": 118,
  "candidate_count": 1,
  "admissible_candidate_count": 1,
  "ridge_label_mix": {
    "B1": 2,
    "F11": 2,
    "F14": 2,
    "F17": 1,
    "residual": 111
  }
}
```

The single ridge candidate:

```text
RIDGE1 atoms=118
span=36.467861 mm
mean_ridge_score=0.802522
mean_curvature_signal=0.902221
mean_normal_contrast=0.828651
```

Visual inspection:

- This is sharper than the broad residual/interface patch.
- But it is still one connected U/corner-shaped ridge.
- It does not separate into two missing conventional bend lines.

Decision:

```text
Reject local ridge tracing alone as the fix.
```

## The Wall We Hit

The deterministic evidence now says:

```text
There is real local bend/transition signal in the missing-support area,
but our current object proposals collapse it into one connected support object.
```

Manual truth says there are 3 conventional bends total. Current accepted F1 support gives us only one valid bend (`OB2`). The two missing conventional bends are not recovered by:

- splitting current F1 owned regions;
- selecting from current overcomplete birth hypotheses;
- broad residual/flange-interface support mining;
- geometric sub-clustering of that support;
- local curvature/normal ridge tracing.

The hardest part is not finding "something bend-like." We find bend-like support. The hard part is decomposing a connected corner/transition ridge into the right number of conventional bend objects.

This is exactly where our current representation may be insufficient:

```text
Owned support component = one bend object
```

That assumption breaks if a connected ridge/corner support corresponds to multiple conventional bend objects whose supports touch or nearly touch.

## Current Hypotheses

### Hypothesis A: We Need Explicit Interface-Line Instances

Maybe each conventional bend should be represented by a latent interface line between two flange components, and support atoms should be assigned to interface-line instances, not only to broad bend labels.

The current residual/interface method scores support against flange pairs, but it does not create competing line instances along the connected ridge.

Possible formulation:

```text
For each flange pair (f_i, f_j), infer one or more interface-line instances l_k.
Each l_k owns a connected or nearly connected subset of transition atoms.
Bend count is number of admissible line instances, not number of connected ridge components.
```

Question:

> Is this the right next object type, or should the primary object still be a support patch?

### Hypothesis B: We Need Junction-Aware Decomposition

The visual ridge looks like a U/corner shape. Conventional bends may meet at a junction or tight corner. The correct count may require splitting a connected ridge at junctions, endpoints, tangent changes, or incidence changes.

Possible formulation:

```text
Build a ridge graph from high-curvature/normal-contrast atoms.
Fit local tangent directions.
Detect junctions / high tangent-turn locations.
Segment ridge graph into path primitives.
Assign each path primitive to one bend object.
```

Question:

> Should this be treated as graph path decomposition of a ridge skeleton, with bend objects as path segments?

### Hypothesis C: We Need Flange-Incidence Changes Along the Ridge

The broad support may be one connected atom component because the atom graph joins adjacent transition zones, but the true conventional bends may have different incident flange pairs along the path.

Possible formulation:

```text
For each ridge atom, compute posterior over incident flange pair.
Split connected ridge where the best incident pair changes or where pair confidence drops.
```

Question:

> Should support splitting be driven primarily by incident flange-pair changes rather than geometric gaps?

### Hypothesis D: Manual "Conventional Bend" Semantics Need a Feature-Class Layer

The current ridge might include conventional bend plus corner/raised/form transition material. Manual truth excludes those from conventional count.

Possible formulation:

```text
Classify ridge segments as:
- conventional straight bend;
- corner transition / junction;
- raised/form feature;
- scan artifact / fragment.
Count only conventional straight bend segments.
```

Question:

> Do we need this class layer before count recovery is mathematically well-posed?

## What We Need From You

Please do not answer only at the level of "use graph cuts" or "use better clustering." We need a concrete next formulation that can be mapped into the current F1 atom graph.

The highest-return questions:

### 1. What is the correct latent object here?

When a connected high-curvature ridge may contain multiple conventional bends, should the countable object be:

```text
A. connected bend support component;
B. interface-line instance between flange components;
C. path segment in a ridge skeleton;
D. flange-incidence transition object;
E. something else?
```

Which object has the best identifiability properties from scan-only geometry?

### 2. How should we split a connected ridge into conventional bend objects?

Given a connected ridge atom set `R`, what are the right split criteria?

Candidate criteria:

```text
- tangent direction changes along ridge skeleton;
- incident flange-pair changes;
- endpoints / junction degree;
- local normal-arc discontinuity;
- support width changes;
- MDL/BIC gain from 2 path instances vs 1 path instance;
- flange-contact boundary changes;
- minimum straight-span support.
```

Which of these should be primary? Which should be secondary?

### 3. What should the model-selection test be?

For this case, we need to compare:

```text
one connected ridge object
vs
two conventional bend objects plus one junction/corner residual
vs
three or more over-fragmented bend objects
```

What should the score look like?

One possible objective:

```text
E =
  data fit to local normal/curvature ridge
  + line/path fit residual
  + flange-incidence consistency
  + junction penalty
  + bend label cost
  + residual/junction label cost
  + duplicate/overlap penalty
```

But we need guidance on the correct terms and which terms must dominate.

### 4. Should junction/corner support be its own label?

Right now, if a connected ridge includes a corner/junction, the solver can only:

```text
count it as a bend,
assign it to flange,
or throw it into residual.
```

Should F1 add an explicit label type:

```text
junction_transition / corner_transition
```

so conventional bend line instances can terminate at it without merging into one support?

### 5. Should conventional-bend count require straightness?

Manual truth here is conventional bends, not every formed/curved/ridged transition.

Should a countable conventional bend satisfy:

```text
- enough straight span;
- roughly constant axis direction;
- two dominant incident flange components;
- no junction branch inside the support;
- local path curvature below threshold along the bend axis?
```

How would you define this robustly on an atom graph?

### 6. Is this a case where the current atom graph resolution is too coarse?

The atom graph may be joining nearby transition supports into one connected ridge.

Should the next test be:

```text
rebuild atoms at finer resolution near high curvature / high normal contrast,
then rerun ridge/path decomposition?
```

Or is the issue not atom resolution but missing object-level topology?

### 7. What is the minimal experiment you would run next?

Please propose the smallest next implementation step.

Examples:

```text
A. ridge skeleton + path segmentation;
B. per-atom incident flange-pair posterior + split on pair changes;
C. explicit junction label and line-instance model selection;
D. finer atom graph around high-curvature ridge;
E. local hand-built MAP on this one case to prove the formulation.
```

We want the experiment that best answers whether exact recovery is possible without over-fragmenting other cases.

## The Concrete Challenge for You

If you were solving only this case from our payload, what would you do?

Inputs you have:

```text
atom graph:
  atom centroid
  atom normal
  curvature proxy
  local axis proxy
  weight / area
  adjacency

flange hypotheses:
  plane normal
  plane offset
  support atoms
  centroid

manual labels:
  OB2 is valid
  OB1/OB3/OB4 are invalid
  truth conventional bend count is 3

diagnostics:
  residual/interface support = one overlapping component
  sub-clustering = one broad component
  local ridge = one U/corner-shaped ridge
```

Output needed:

```text
two additional conventional bend objects,
or a principled explanation that the two objects are not identifiable from this scan without stronger priors/manual semantics.
```

Please answer in terms of:

```text
variables,
candidate objects,
energy terms,
split/merge/add/remove moves,
and concrete acceptance/rejection tests.
```

## My Current Read

I think the next version should stop treating connected ridge support as the countable object. A countable conventional bend should probably be a path-like/interface-line object with:

```text
support atoms,
axis/path,
incident flange-pair field,
endpoints,
junction contacts,
straightness score,
and class = conventional bend / junction / form feature / residual.
```

The likely missing operation is:

```text
connected ridge component -> skeleton/path graph -> path segment model selection
```

But I want your guidance before implementing this because the wrong formulation can easily create beautiful over-segmented lines that count too many bends.

## What I Will Send With This Note

Suggested reading order:

1. Read this note first.
2. Open the three 1080p overlays:
   - residual/interface candidate overview;
   - residual/interface subcluster overview;
   - local ridge candidate overview.
3. Inspect:
   - `49024000_feature_region_labels.json`
   - `49024000_local_ridge_candidates.json`
   - `49024000_residual_interface_support_candidates.json`
4. If code-level details are needed, inspect:
   - `scripts/trace_f1_local_ridge_candidates.py`
   - `scripts/analyze_f1_residual_interface_support_candidates.py`
   - `backend/patch_graph_latent_decomposition.py`

The goal is not to tune thresholds. The goal is to decide the next mathematical object model.
