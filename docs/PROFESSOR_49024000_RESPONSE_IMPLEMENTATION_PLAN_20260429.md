# Professor Response Implementation Plan: `49024000` Junction-Aware Bend Arrangement

Date: 2026-04-29

## Decision

The next F1 prototype should not make a support patch or connected ridge the countable object.

The countable object becomes:

```text
maximal flange-pair-conditioned conventional bend-line instance
```

inside a larger uncounted object:

```text
connected ridge/corner support
  -> bend-complex / junction-transition object
    -> countable conventional bend-line instances
    -> uncounted junction/corner-transition atoms
    -> residual/form/uncertain atoms
```

## Why

For `49024000`, the existing evidence is decisive:

- manual truth is `3` conventional bends;
- only raw `OB2` is accepted;
- current raw F1 has four owned regions but only one valid conventional region;
- split diagnostics find no splitable current owned region;
- residual/interface support exists but collapses into one overlapping component;
- sub-clustering still gives one component;
- local ridge tracing sharpens the support but still gives one connected U/corner-shaped ridge.

The current ontology:

```text
connected support component = countable bend
```

is the wrong abstraction for this case.

## Minimal Next Experiment

Build an offline diagnostic-only local MAP/MDL prototype for the single connected ridge complex in `49024000`.

### Inputs

Use:

```text
decomposition_result.json
49024000_feature_region_labels.json
49024000_local_ridge_candidates.json
```

Local domain:

```text
R = atoms from RIDGE1
locked accepted bend = OB2 / F11-F14
manual rejected OB1/OB3/OB4 = evidence only, not trusted bend objects
nearby candidate flanges = F1, F11, F14, F17 plus top contacted/nearby flanges
```

### Candidate Objects

Generate candidate bend-line instances:

```text
b_k = (flange_pair, path_atoms, fitted_line, axis, endpoints, score)
```

For each plausible flange pair `(f_a, f_b)`:

```text
t_ab = normalize(n_a x n_b)
```

Score atoms by:

```text
normal-arc compatibility with (n_a, n_b)
local tangent / ridge axis compatibility with t_ab
distance to candidate line/path
ridge score
two-sided flange contact
support span / mass
```

### Junction Label

Add explicit uncounted label:

```text
junction_transition
```

This label allows conventional bend-line instances to terminate at a shared corner without forcing a merge or creating extra counted fragments.

### Model Hypotheses

Compare local arrangements:

```text
H0: one connected ridge / residual-junction only
H1: accepted OB2 only + junction/residual
H2: accepted OB2 + one new conventional bend line + junction
H3: accepted OB2 + two new conventional bend lines + compact junction
H4: accepted OB2 + three or more line candidates / overfragmentation
```

The expected useful hypothesis, if identifiable:

```text
keep:    F11-F14 as OB2
add:     F11-F17 if it has stable arm support
add:     F1-F14 if it has stable arm support
reject:  F1-F17 if it is only a diagonal/chord explanation
label:   central U/corner as junction_transition
```

Do not hard-code those pair choices. Use them as the first audit expectation.

## Acceptance Gate

This stays diagnostic-only until the visual evidence passes.

Pass condition:

```text
1080p overlay shows:
  - OB2 kept;
  - two additional separated bend-line instances;
  - compact uncounted junction/corner support;
  - low overlap between the new bend-line supports;
  - no counted diagonal/chord candidate;
  - no stealing from accepted OB2.
```

Fail condition:

```text
The local model still returns:
  - one broad U/corner patch;
  - arbitrary 2-vs-3+ segmentation;
  - counted chord/diagonal candidate;
  - small endpoint fragments;
  - or no stable pair-specific line arms.
```

## Implementation Order

1. Add `scripts/solve_f1_junction_bend_arrangement.py`.
2. Load the local ridge payload and candidate flanges.
3. Compute per-atom flange-pair scores for the ridge domain.
4. Generate maximal pair-specific line/path candidates.
5. Add a small local MDL selector over candidate subsets.
6. Penalize:
   - duplicate overlap;
   - same-pair collinear duplicates;
   - short stubs;
   - chord candidates explained by a sparse flange path;
   - internal high-turn/junction atoms inside a conventional bend line.
7. Emit:
   - arrangement JSON;
   - 1080p overlay;
   - visual verdict markdown.

## Promotion Rule

No product/runtime promotion from this tranche.

Even if `49024000` looks good, the next step is to run the same local arrangement diagnostic on:

```text
48991006
10839000 partial/full raised-feature cases
47959001 compact-neighbor case
ysherman / 49204020_04 repeated-near-90 cases
```

Only after visual gates pass across these families should the object model move into F1 solver logic.
