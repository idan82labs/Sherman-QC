# Patch-Graph F1 Professor Handoff

## Purpose

This note captures the current state of the offline F1 patch-graph latent decomposition lane after the first professor-guided refactor toward:

- fixed atom graph and candidate pools
- inner solve as unary + Potts-like smoothness + label costs
- outer object penalties for incidence, fragmentation, duplicate families, and support mass
- count from admissible connected bend support components

The runtime zero-shot engine is unchanged. This is an offline research lane only.

## Best Review Subjects

### 1. `10839000`

Why this case matters:

- current validated control remains strong:
  - exact `10`
  - range `9-10`
- F1 is now more deterministic but still wrong:
  - current cached F1 result: exact `7`
  - range `[7, 7]`

Why it is useful:

- this is not a “noisy uncertainty” case anymore
- the professor-guided refactor made the solution family collapse to one stable answer
- that answer is still undercounting by `3`
- so this is now the cleanest test of whether the inner ownership/object score is still too biased toward merging or under-activating bends

Current read:

- the missing issue is no longer broad perturbation instability
- it looks like the model is confidently preferring too few bend objects
- local repair moves alone have not fixed it

### 2. `49204020_04`

Why this case matters:

- validated zero-shot control remains a hard `11-12` case
- the typed F1 lane can already hit exact `12` in MAP on this scan

Why it is useful:

- this is the opposite of `10839000`
- it suggests the atom graph and candidate pool already contain enough information for exact observed count on at least one hard case
- so it is a good “positive” companion to the `10839000` failure

Current read:

- the formulation direction is probably right
- but the object score / support assignment still fails on some repeated-near-90 anchor geometries

## What Changed Recently

The latest F1 refactor did three main things:

1. Removed typed adjacency from the inner atom-label cost.
   - inner solve is now closer to unary + Potts smoothness

2. Moved more structure into explicit outer penalties.
   - incidence
   - fragmentation
   - duplicate-family penalty
   - support-mass penalty

3. Added fast cached offline inputs.
   - repeat runs now reuse:
     - atom graph
     - zero-shot control result
     - normalized flange hypotheses
     - normalized bend hypotheses

This made iteration much faster and cleaner, and reduced `10839000` from unstable `[4, 7]` to deterministic `[7, 7]`.

## Most Relevant Files

### Core offline prototype

- `backend/patch_graph_latent_decomposition.py`

What to review there:

- `run_patch_graph_latent_decomposition(...)`
  - stage timings
  - cached fixed-input reuse

- `solve_typed_latent_decomposition(...)`
  - inner/outer split

- `_label_cost(...)`
  - current inner solve score

- `_total_energy(...)`
  - current outer object penalties

- `_activate_interface_bends(...)`
  - current birth / split-like repair logic

- `_bend_component_admissibility(...)`
  - current count gate

### Offline runner

- `scripts/run_patch_graph_f1_prototype.py`

What to review there:

- single-part execution
- cached input reuse
- no-render iteration path

### Focused tests

- `tests/test_patch_graph_latent_decomposition.py`

What to review there:

- canonical bend-family normalization
- admissibility expectations
- interface birth / split-like behavior
- outer duplicate/fragmentation penalty coverage

## Current Evidence on `10839000`

Latest cached F1 result:

- exact count: `7`
- count range: `[7, 7]`

Interpretation:

- the new formulation is more stable
- but still structurally undercounts
- this suggests the next missing piece is probably not another local threshold
- it is more likely a deeper ownership/model-competition issue in the inner solve or in explicit split/merge model selection

## Question for Professor

Given these two contrasting subjects:

- `10839000`: stable but confidently wrong undercount
- `49204020_04`: typed MAP can already hit exact `12`

what is the next highest-leverage mathematical change?

The most likely options now seem to be:

1. stronger split/merge model selection between competing bend families
2. richer object-level competition between flange and bend ownership on overlapping support
3. a better local approximation to decomposition comparison, rather than only one MAP pass plus lightweight repair moves

## Current Recommendation

Do not spend more time on:

- detector/preprocess tuning
- viewer work
- broad selector heuristics

The next serious work should stay inside the typed decomposition lane.
