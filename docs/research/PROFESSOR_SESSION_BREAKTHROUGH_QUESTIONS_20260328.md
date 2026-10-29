# Professor Session: Breakthrough Questions For Bend Detection, Pinpointing, Rolled Sections, Alignment, And Partial-State Overlay

## Purpose
These are the highest-leverage technical questions still blocking major accuracy gains.

They are written to force concrete decisions, not open-ended discussion.
For each question, I included:
- why it matters
- what decision it unlocks
- the exact answer format we want back

The ideal outcome from the session is not "interesting guidance".
It is a set of rules we can convert directly into:
- model structure
- target definitions
- alignment policy
- labeling policy
- benchmark slices

---

## Executive view: the core unknowns
The project is no longer blocked by repo structure.
It is blocked by 5 technical unknowns:

1. What exactly counts as a bend target versus a process-form feature.
2. What the canonical datum/alignment strategy should be for finished and partially completed parts.
3. How rolled sections should be parameterized and validated.
4. How to compare scan vs CAD locally when the part is only partially formed.
5. What minimum truth labels would most efficiently improve the system.

---

## 1. Canonical target definition
### Question
What is the canonical definition of a "bend" for this system?

We need a rule that decides, for every geometric change in the CAD/drawing/process:
- `countable discrete bend`
- `rolled/process feature`
- `non-bend transition / geometry detail`

### Why this matters
Right now many downstream failures are really target-definition failures:
- rolled sections counted as bends
- shell transitions mistaken for flange bends
- repeated small features competing with true bend targets

### Decision this unlocks
- corpus labeling rules
- UI numbering rules
- per-family detector routing
- MAE interpretation

### Expected answer format
```yaml
canonical_bend_definition:
  countable_discrete_bend: "..."
  rolled_process_feature: "..."
  non_bend_transition: "..."
  count_by:
    - final geometry
    - manufacturing operation
    - drawing callout
  preferred_choice: "pick one"
  exceptions:
    - family: "..."
      rule: "..."
```

### Hard follow-up
If manufacturing sequence and final geometry disagree, which one wins for model truth?

---

## 2. Gold-standard alignment hierarchy
### Question
What should be the canonical alignment hierarchy for scan-to-CAD comparison?

Specifically:
- should drilled holes / slots / punched datums outrank surface ICP?
- should broad flat regions outrank holes when holes are few?
- should the alignment strategy change between:
  - full finished parts
  - partially bent parts
  - rolled/hybrid parts

### Why this matters
The current system can align well globally but still compare the wrong local bend frame.
If we get the datum hierarchy right, a lot of "complex bend math" becomes simpler and more stable.

### Decision this unlocks
- whether to build a hole-first or surface-first registration stack
- whether to use unchanged flat regions as stage anchors on partial parts
- whether local bend metrology should inherit a dedicated datum frame

### Expected answer format
```yaml
alignment_hierarchy:
  full_finished:
    primary: "..."
    secondary: "..."
    tertiary: "..."
  partial_stage:
    primary: "..."
    secondary: "..."
    tertiary: "..."
  rolled_hybrid:
    primary: "..."
    secondary: "..."
    tertiary: "..."
  note_on_holes: "when holes are authoritative vs not"
  note_on_flat_regions: "when large unchanged flats should dominate"
```

### Hard follow-up
If you had to ban one alignment primitive from the system entirely, which one would you ban first:
- global ICP
- hole-based alignment
- plane-based alignment
- feature-to-feature local registration

That answer tells us what is actually carrying the system.

---

## 3. Hole/slot overlays as precision anchors
### Question
Can drilled holes / slots / punched openings be treated as the primary local anchor set for 1:1 overlay accuracy on finished parts?

And if yes:
- what minimum hole quality is required?
- how many features are enough?
- what geometry is acceptable: circles only, slots too, countersinks too?

### Why this matters
If holes are stable and precise enough, we can simplify local bend pinpointing:
- fit the part by hole datums first
- only then measure bend deltas
- reduce reliance on more fragile global curvature matching

### Decision this unlocks
- whether to build a hole-first alignment mode
- whether to generate local per-bend frames from nearby holes/slots
- whether final-part metrology can become simpler and more robust

### Expected answer format
```yaml
hole_anchor_policy:
  use_as_primary_for_finished_parts: true_or_false
  allowed_feature_types:
    - circle
    - slot
    - countersink
    - other
  minimum_anchor_count: N
  minimum_spatial_spread_rule: "..."
  known_failure_modes:
    - "..."
  confidence: high_medium_low
```

### Hard follow-up
If hole-based registration and surface-based registration disagree by a meaningful amount, which one should we trust, and why?

---

## 4. Partial-stage invariant geometry
### Question
For partially completed parts, what geometry should be treated as invariant and safe to align on across stages?

Examples:
- big flat plates that do not change shape yet
- hole patterns in still-flat regions
- reference edges far from future bend lines
- already completed bends

### Why this matters
A major missing capability is robust overlay of partially completed parts where most of the sheet is still geometrically similar between stages.
If we identify invariant regions correctly, stage-to-stage comparison becomes much easier.

### Decision this unlocks
- partial-part registration masks
- unchanged-region alignment policy
- stage progression priors
- better observability reasoning

### Expected answer format
```yaml
partial_stage_invariants:
  trusted_region_types:
    - "..."
  avoid_region_types:
    - "..."
  can_use_completed_bends_as_datums: true_or_false
  can_use_unbent_large_flats_as_primary_datums: true_or_false
  stage_to_stage_rule: "..."
```

### Hard follow-up
If only one large flat region remains unchanged, is that enough to define a trustworthy local comparison frame, or do we still need discrete features like holes?

---

## 5. Rolled sections: representation
### Question
How should a rolled section be represented mathematically and semantically in the system?

Choices include:
- one manufacturing step only
- one geometric feature with radius + arc extent + centerline
- several local sections sampled along the roll
- a profile family rather than a discrete bend

### Why this matters
Rolled sections are currently the hardest family to model cleanly.
Treating them as discrete bends is wrong, but treating them as generic surfaces loses process meaning.

### Decision this unlocks
- whether rolled sections get their own validator path
- whether rolled sections should be parameterized by section slices
- whether rolled sections should ever enter bend-count MAE

### Expected answer format
```yaml
rolled_section_representation:
  semantic_type: "manufacturing_step | geometric_feature | both"
  countable_in_bend_progress: true_or_false
  primary_parameters:
    - radius
    - arc_extent
    - centerline
    - local_section_shape
    - other
  comparison_method_preference:
    - local cross-sections
    - full surface residuals
    - centerline fit
    - hybrid
```

### Hard follow-up
If we had to validate a rolled section with only one number and one plot, what should they be?

---

## 6. Rolled sections: tolerance semantics
### Question
What is the correct tolerance semantics for rolled sections?

Do we validate:
- local radius error
- centerline deviation
- arc extent error
- cross-sectional profile mismatch
- global heatmap only
- some combination

### Why this matters
The answer determines whether rolled validation can be simple and robust or stays vague.

### Decision this unlocks
- rolled-feature pass/fail logic
- what the UI should show for rolled areas
- whether heatmap is enough or only supportive

### Expected answer format
```yaml
rolled_tolerance_semantics:
  primary_metric: "..."
  secondary_metric: "..."
  when_global_heatmap_is_enough: "..."
  when_local_section_validation_is_required: "..."
  pass_fail_rule: "..."
```

---

## 7. Bend angle truth source
### Question
When the drawing, process sheet, and final CAD imply slightly different bend angles, what is the correct truth source for expected bend angle?

Possible truth sources:
- final CAD included angle
- process sheet/tooling angle
- drawing callout
- springback-adjusted manufacturing intent

### Why this matters
This is the root cause behind several current contradictions:
- heatmap says geometry fits
- bend metrology says the angle is off
- both cannot be trusted equally without a defined truth source

### Decision this unlocks
- expected-angle baseline policy
- when to trust CAD over spec sheet
- when angle deltas are actually meaningful

### Expected answer format
```yaml
bend_angle_truth_source:
  default_source: "cad | drawing | process_sheet | other"
  if_sources_disagree: "resolution rule"
  springback_handling: "..."
  family_exceptions:
    - family: "..."
      source: "..."
```

### Hard follow-up
If final CAD and scan agree globally but bend angle appears locally wrong, should we first assume:
- wrong target definition
- wrong local frame
- wrong expected angle
- actual manufacturing defect

Rank these in order.

---

## 8. Local bend frame definition
### Question
How should the local comparison frame for a bend be defined?

Options:
- by CAD bend line + adjacent CAD faces
- by local scan planes only
- by nearby datums (holes/edges) first, then bend line
- by section slice orthogonal to bend line

### Why this matters
If the local frame is wrong, angle and radius errors are meaningless even when the overlay looks visually right.

### Decision this unlocks
- local bend measurement method ordering
- whether section-slice measurement should dominate
- whether local hole datums should be built into bend metrology

### Expected answer format
```yaml
local_bend_frame:
  preferred_frame_definition: "..."
  minimum_geometry_required: "..."
  if_adjacent_planes_are_short_or_noisy: "fallback rule"
  if_local_holes_exist: "use or ignore"
```

---

## 9. Distinguishing real bends from shell/profile transitions
### Question
What geometric cues best separate a true flange bend from a shell/profile transition on hybrid parts?

Candidates:
- bend-line length
- adjacent face area ratio
- distance from edge
- distance from rolled centerline
- local section topology
- drawing-callout proximity

### Why this matters
This is the core selection problem on parts like `28870000`.
If we can get this rule family right, hybrid parts stop being ambiguous.

### Decision this unlocks
- hybrid CAD target selector scoring
- candidate pruning before scan matching
- label quality rules for future corpus additions

### Expected answer format
```yaml
hybrid_selector_priority:
  strongest_positive_cues:
    - cue: "..."
      weight: high_medium_low
  strongest_negative_cues:
    - cue: "..."
      weight: high_medium_low
  decisive_rule_if_only_one_exists: "..."
```

---

## 10. When should heatmap and bend metrology be expected to agree?
### Question
Under what conditions should local bend metrology and local heatmap residuals strongly correlate, and under what conditions should we explicitly expect them not to?

### Why this matters
We already built the advisory validator, but the real value depends on knowing when disagreement is meaningful.

### Decision this unlocks
- where to trust the heatmap-consistency signal
- which slices should be excluded from contradiction logic
- how to interpret quiet heatmaps with bend-local failure reports

### Expected answer format
```yaml
heatmap_bend_correlation:
  strong_correlation_cases:
    - "..."
  weak_correlation_cases:
    - "..."
  not_expected_to_correlate:
    - "..."
  safest_first_use: "..."
```

### Hard follow-up
Would you ever allow heatmap agreement to overturn a bend failure, or only to flag it for review?

---

## 11. Minimum observability for declaring a bend formed
### Question
What minimum evidence should be required to declare a bend as formed on a partial scan?

Examples:
- one adjacent face visible
- both adjacent faces visible
- local section present
- hole/edge context confirms frame
- partial evidence only allowed as weak support

### Why this matters
This directly controls false positives on partial scans, which is still one of the biggest MAE drivers.

### Decision this unlocks
- `formed` vs `not detected` policy
- observability thresholds
- whether continuity priors are allowed to carry bends forward

### Expected answer format
```yaml
minimum_observability_for_formed:
  required_signals:
    - "..."
  optional_supporting_signals:
    - "..."
  never_enough_on_its_own:
    - "..."
  partial_scan_rule: "..."
```

---

## 12. Ground truth labeling: highest-value additional label
### Question
If we can afford only one new label type in the corpus, what should it be?

Candidates:
- completed bend IDs
- per-bend formed/not formed
- rolled section presence / completion state
- invariant datum regions for partial parts
- good/bad final quality label
- hole datum confidence label

### Why this matters
We need the most leverage per annotation hour.

### Decision this unlocks
- next data-collection policy
- whether to prioritize part-family labels or bend-state labels
- what to request from the shop floor / engineering team

### Expected answer format
```yaml
highest_value_new_label:
  label_name: "..."
  why_it_beats_the_others: "..."
  minimum_annotation_format: "..."
  expected_model_gain: "where it helps"
```

---

## 13. Benchmark design: what should be the true breakthrough metric?
### Question
If we want one metric that best captures whether the system is becoming industrially useful, what should it be?

Candidates:
- bend-count MAE
- exact completed-bend-set accuracy
- full-part registration error after datum alignment
- local bend-angle MAE on observable bends
- rolled-section tolerance pass/fail accuracy
- operator-review save rate

### Why this matters
A lot of optimization direction depends on the wrong metric.

### Decision this unlocks
- what we optimize first
- how we judge new data contributions
- whether alignment or metrology should dominate roadmap priority

### Expected answer format
```yaml
breakthrough_metric:
  primary_metric: "..."
  secondary_metrics:
    - "..."
  why_primary: "..."
  what_not_to_overoptimize_yet:
    - "..."
```

---

## 14. Simplification challenge
### Question
If you had to simplify the system radically without losing accuracy, what would you simplify first?

Examples:
- rely more on hole datums and less on global ICP
- rely more on local sections and less on full 3D bend-line fitting
- split rolled families into a separate pipeline entirely
- use invariant flats for stage alignment and stop trying to register changing geometry globally

### Why this matters
We need to know whether the current complexity is necessary or just accidental.

### Decision this unlocks
- where to cut math
- whether to separate pipelines by family
- how to reduce runtime while improving trustworthiness

### Expected answer format
```yaml
preferred_simplification:
  remove_or_reduce: "..."
  replace_with: "..."
  why_accuracy_should_not_drop: "..."
  best_part_families_for_trying_it: ["..."]
```

---

## 15. Final forced ranking
### Question
Please rank the following as the biggest remaining blocker to industrial-grade performance:
- target definition
- alignment/datum policy
- partial-scan observability
- rolled-section representation
- local bend-frame definition
- lack of labeled truth
- benchmark design

### Why this matters
We need a ruthless priority order.

### Expected answer format
```yaml
blocker_ranking:
  1: "..."
  2: "..."
  3: "..."
  4: "..."
  5: "..."
  6: "..."
  7: "..."
  rationale: "short explanation"
```

---

## My own blunt view going into the meeting
If the professor gives strong answers on only these 4 items, the project direction changes materially:

1. **canonical bend target definition**
2. **alignment hierarchy, especially holes vs surfaces vs invariant flats**
3. **rolled section representation and tolerance semantics**
4. **truth source for expected bend angle when CAD/spec/drawing disagree**

Those four answers would immediately change:
- model routing
- labeling policy
- local metrology
- trust in the current reported deltas

---

## Suggested way to run the session
1. Start with Questions 1, 2, 5, 7.
2. Do not let the discussion stay abstract.
3. Force a concrete answer schema for each one.
4. Only then move to Questions 4, 8, 9, 10.
5. End with Questions 12, 13, 15 so we know what to label and what to optimize.

If time is short, the first four questions are the highest-yield ones.
