# Independent recheck of the Fable 5.1 review follow-through delta

Read-only recheck, 2026-09-12 22:42-22:52 UTC, by an independent reviewer
(not the author). Scope: does the delta described in
`2026-09-12_fable51_review_response_v1.md` do what it claims, resolve receipt
items 1-4 and the two preliminary reviews' P2/observations, and change nothing
else that weakens those reviews' conclusions.

## Decision and limits

- **No P1 found.** No P2 found. Four P3 notes (below).
- Receipt item 1: RESOLVED at source. Item 3: RESOLVED at source. Item 4:
  RESOLVED at source. Item 2: RESOLVED at wording/disclosure level with one
  P3 gap (leaf128 figure not retained); net efficacy remains unqualified by
  design.
- Limits: no test, import, geometry job or benchmark was run; author-reported
  numbers (test passes, CPU, 48.6M/30.8M, needle width occurrence) are not
  reproduced. The reviewed old bytes are unrecoverable (files are untracked in
  git); the delta was inferred from the reviews' line citations, the response,
  and `diff` against the frozen V1/V3 predecessors, not from a byte diff.
- Independence: findings were formed before reading the two concurrent Codex
  delta reviews (`..._extrema_v3_codex_delta_review.md`,
  `..._surface_v4_codex_delta_review.md`), which bind the same hashes below.
  The receipt changed on disk mid-review (2749 -> 4739 bytes, SHA256
  `01766ee2fc4d5fa868c91c5094b7f38bd18fc442378767ae83b76eea33219a78`) to
  record those closures; nothing here relies on that section.
- Only this file was created (in the reviewer's scratch directory, because the
  harness blocked a direct write into the shared checkout). No engine/test file
  was edited.

## Examined bindings

Nine files, SHA-256 identical at start (22:42:29 UTC), mid-review (22:49:55)
and end (22:51:53). Size deltas are against the preliminary reviews' bindings.

| File | Bytes (reviewed -> now) | SHA-256 now |
| --- | ---: | --- |
| `backend/cad_guided_raw_extrema_index_v3.py` | 23208 -> 23978 | `83dea6529003f1d5deae592c9d35e566a77e2c9f6f3af8f641928f5465845bc0` |
| `backend/cad_guided_two_plane_context_v4.py` | 26196 (unchanged) | `e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25` |
| `backend/cad_guided_triangle_surface_v4.py` | 36791 -> 37248 | `f79b73b37be2684f73f7d467d5aa1074f8a27d968cdf52e9fb1d652dae02422c` |
| `backend/cad_guided_global_registration_v4.py` | 36998 -> 37673 | `5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273` |
| `backend/cad_guided_stage_policies_v2.py` | 3328 (unchanged) | `699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d` |
| `tests/test_cad_guided_raw_extrema_index_v3.py` | 32860 -> 33604 | `f78c18d146d4ca2f86e57a9690daa58251e92b2dee207de1a499a61ece6cc572` |
| `tests/test_cad_guided_two_plane_context_v4.py` | 37574 -> 37955 | `ef57b38f23249c8c1555d0c61fd05913a8068027147ca020b082679154d7a945` |
| `tests/test_cad_guided_triangle_surface_v4.py` | 25896 -> 26622 | `cc80b5c7f3f676176804d1d5c15855469ca2c17485672acc0f6549fd46c24db8` |
| `tests/test_cad_guided_global_registration_v4.py` | 32770 -> 36702 | `1cf31b12f373f1c0b8d72eecf5c6bd5ae86f704feeaaf0d14e5dda8031735204` |

Predecessors: all ten files the reviews bound (`raw_extrema_index_v1`,
`two_plane_context_v1/v2/v3`, `triangle_distance_bounds_v3` + its test,
`triangle_surface_v3`, `global_registration_v3` and their tests) hash-match
the reviews' tables; no frozen predecessor was edited. Documents read:
receipt, response (3980 B, `99458961d11b...`), both preliminary reviews, both
delta reviews, `2026-09-12_fable51_progress.md`. All nine files parse (ast).
`_staged`, `_bounds`, `_evaluate`, `query`, `_arithmetic_sentinels`, `_vector`
are ast-source-identical between extrema V1 and V3.

## Item-by-item verification

### Receipt item 1 - counter/flag consistency and one actual mixed-row test: RESOLVED

**Guard** (`backend/cad_guided_global_registration_v4.py:205-213`):
`jointly = normal_ambiguity_uncertain & closest_point_ambiguity_uncertain`;
require counter == 0, or (any(jointly) and counter <= sum(near_tie_counts[jointly])).
Runs after type/shape/range validation of the arrays (:169-179) and of the
counter (:127-128 via :201), on every validated query: adapter (:259) and
`_support_masks` (:268), hence also the stage classifier.

**Producer trace** (`backend/cad_guided_triangle_surface_v4.py`), to decide
whether the bound can be violated by a legitimate COMPLETE output:
- The counter is touched only in the tie pass: pass 1 `continue`s at :555
  before the width test.
- Each (row, triangle) pair is evaluated at most once per walk: internal
  nodes split `rows` into disjoint `near_left`/`~near_left` subsets and push
  each subset to each child once (:596-600); leaves are disjoint permutation
  segments (:602); triangle batches (:510-511) and row batches (:513-514) are
  disjoint slices; rank `k` gives each row one triangle (:520-522).
- :556-559 `competitive` selects pairs, `target` = unique rows. :565-567
  `wide`; :572 counter += count(wide); :573-574 both flags set on exactly
  `target[wide]`. :575-582 EVERY competitive pair, wide included, consumes one
  tie slot for its row or raises `NEAR_TIE_BUDGET_EXCEEDED` (:578), which makes
  the query UNAVAILABLE (:669-671) and thus never validated.
- Tie batches are disjoint row ranges (:614-616); `near_tie_counts[tie_start:tie_stop]
  = tie_counts` (:626) is written before the next batch's reset (:621).
- Flags are only ever set (:573-574, `|=` at :637 and :645), never cleared.

Therefore per row r: wide_pairs(r) <= near_tie_counts(r), and wide_pairs(r) > 0
implies r in `jointly`. Summing over disjoint batches: counter <= sum over
jointly rows of near_tie_counts. The guard is necessary, not sufficient
(other rows can be jointly uncertain via the :637/:645 bands, loosening the
bound), uses the right arrays, cannot reject a legitimate COMPLETE output, and
handles zero rows (counter must be 0). Sum <= 200000*256 = 51.2M; no overflow.
The producer invariant it relies on is pinned by the `interval_width_policy`
metadata equality (:69, :160). Sound.

**Mixed-row test** (`tests/test_cad_guided_global_registration_v4.py:665-698`),
geometry checked against the source (not executed):
- Needle T0 `(0,0,0),(100,0,0),(100+1e-5,0,1e-5)`: shortest edge 1.414e-5,
  shape quality 1e-7 >= `MIN_SHAPE_QUALITY` 1e-8 (admitted, :236), normal
  (0,-1,0); allowed width `down(1e-5*1.414e-5)` ~ 1.4e-10 (:566). Sound T1:
  min edge 5, allowed 5e-5, normal (0,0,1).
- Row 0 `(100, 2e-6, 4e-6)`: needle interior candidate (a,b) ~ (0.59999996,
  0.4), feasible (sum 1-4e-8); residual ~ (0, 2e-6, 0), upper ~ 2e-6. In the
  helper's chart (scale 128, `cad_guided_triangle_distance_bounds_v3.py:177-180`)
  ||r||^2 ~ 2.4e-16 while the outward-rounded support term is ~6e-16 (ulp-level
  widths on 0.78-magnitude coordinates), so `numerator_lower` < 0 and lower = 0
  (:195-200): width ~ 2e-6 >> 1.4e-10, the needle is competitive (lower 0 <=
  up(U*)) -> counter >= 1 and both flags on row 0. T1 for row 0: upper ~
  4.47e-6 < 5e-5, never wide whether or not competitive. distance_upper[0] ~
  2e-6 <= 0.5 -> definite support.
- Row 1 `(50, 1, .05)`: interior of T1 at 0.05; edge gap 1.0 >> geometric
  guard (~4e-9) -> FACE, not region-uncertain. Needle AABB lower = 1.0 >
  up(0.05): box-pruned in the tie pass, so one competitor, no flags.
- Row 2 `(30, 2, 3)`: interior of T1 at exactly 3 -> lower ~ 3 > 0.5 ->
  definite outside. Needle AABB lower 2.99999 passes but its certificate lower
  ~ 3.6055 > up(3): not competitive.
- Asserted masks are the semantically right ones under
  `cad_guided_stage_policies_v2.py:61-76`: `definite_proximity` [T,T,F];
  `proposal_target` = `legacy_unambiguous_support` = distance & `_unambiguous`
  [F,T,F]; `tangent_constraint` [F,T,F]; `correspondence_ambiguous` [T,F,F];
  `definite_outside` [F,F,T]; `_chosen(.5, 1.)` -> `_trim_indices`
  (`cad_guided_global_registration_v1.py:196-200`) on the single finite row ->
  `[1]`. This is exactly the promise the surface review asked to be tested:
  distance-only proximity retained for the wide row, no target/support/tangent,
  sound rows unchanged.
- Caveat: the width occurrence is analytic here (margin ~x2.5 in my ulp count)
  and author-reported (surface test :428-445 and the frozen-V3 refusal); not
  executed.

**Fixtures**: base `_query` (:104-110) still all-false flags with counter 0
(consistent); positive case (:624-638) carries both flags and 2 ties, refuses
counter 3 > 2; negative cases none/normal-only/point-only (:641-651); 256 case
labelled field-maximum (:654-662).

### Receipt item 2 - observations, not net efficacy: RESOLVED (wording); efficacy unqualified

`tests/test_cad_guided_raw_extrema_index_v3.py:378-416`: each full public
build+query batch is timed (:384-389); prints CPU, index visits, sort work,
coordinates and a `common_work` subtotal = coordinate+sort+index (:397-407;
excludes node visits, row visits, hash bytes - disclosed as a subtotal, not an
operation count); comment :408-412 labels the retained assertions (:413-415)
as row/coordinate observations on this fixture, not a net-cost decision; no
directional assertion on sort, common work or CPU, so no V2 gate is imposed.
The response records the 200K-batch regression (48.6M vs 30.8M; 0.34 vs 0.24 s)
as author-reported. Leaf variants: `2026-09-12_fable51_progress.md:45-46`
retains leaf64/leaf256/leaf32 figures; leaf128 is named (:63; response item 5)
but no figure or failure record exists anywhere under `docs/research` (P3-1).

### Receipt item 3 - index-visit convention: RESOLVED

`backend/cad_guided_raw_extrema_index_v3.py:22-31` defines one unit per raw-ID
element touched and states V1's 2-per-row single-sort charge (verified:
`cad_guided_raw_extrema_index_v1.py:194`); :258 charges S before `rows.copy()`;
:268 charges 2n per partitioned segment; :425 `index_work_model` string.
Construction total is now 4S + 2P (P = partitioned-row sum), versus the
reviewed 3S + P. Literal tests match the code: single leaf :558 20 = 4x5; two
leaves :572 600 = 4S+2S (root segment partitioned once), :573 25N coordinates,
:574 sort 100x7, :575 one node visit. Context V4 test :540 `4*memberships +
evaluated_rows` is consistent for single-leaf supports (bind
`two_plane_context_v4.py:166` + root :230 + copy :258 + leaves :292, no
partition). `_partition_gathers` (context test :405-420) restates the :263-266
skip rule exactly. V1 untouched (hash).

### Receipt item 4 - stale comments, no general bitwise claim, byte assertion: RESOLVED

Surface V4 :406-414 now says pass 1 minimizes among evaluated certificates and
states the row-uncertainty policy; :562-564 and :569-571 replace "fatal";
registration V4 :6-7 aligned. No "fatal"/"must pass" text remains. Surface test
:460-492 compares dtype, shape and C-order `tobytes` per result array, tuple
equality for face ids, documents fixture-specific scope, prints build+query CPU
for both versions as consumed observations. Context test :713-746 compares
canonical JSON bytes (repr-based: distinguishes -0.0 and int/float) of the
work-stripped evidence and of each interval; renamed accordingly.

### Review P2s / observations

Surface review P2 (consumer gap), 256 label, pass-1 docstring, soup wording:
closed. Extrema review P2 (net cost): wording closed, efficacy deliberately
open. Accounting clarification and `-0.0` wording: closed.

### Nothing-else-changed check

`diff` of surface V4 vs frozen V3 and registration V4 vs V3 shows only the
declared V4 mechanisms (vertex bound, 256 ties, width policy, work constants,
V4 names/metadata) plus this delta's guard and text; `diff` of extrema V3 vs
V1 shows docstring, budget/leaf constants, axis schedule, partition build, the
copy charge and model strings. Pruning, tie, certificate, enclosure, admission
and failure-boundary code are as the preliminary reviews described. Context V4
and stage policies V2 are byte-identical to the reviewed versions.

## Findings

- **P1: none. P2: none.**
- **P3-1** `docs/research/2026-09-12_fable51_progress.md:45-46` - the leaf128
  variant named at :63 and in the response has no retained figure/failure.
  Repair: add its measurement (or "not completed/refused") on the same line as
  the other variants; receipt item 2 asks that all explored variants be kept.
- **P3-2** `backend/cad_guided_triangle_surface_v4.py:33` - retained V3
  sentence "its own edge-relative interval-width requirement" still reads as a
  gate. Repair: "interval-width test (V4: a failing competitor marks its rows
  uncertain)".
- **P3-3** `tests/test_cad_guided_global_registration_v4.py:679` - the comment
  attributes row 0's width to the short edge; the width actually comes from the
  certificate lower bound being unresolved (0) at ||r|| ~ 2e-6 in a scale-128
  chart, the short edge only setting the 1.4e-10 gate. Optional repair: note
  the mechanism, or assert `answer["distance_lower"][0] == 0.`, so the fixture
  is not mistaken for a geometric-edge case.
- **P3-4** accounting note, no code change: 4S+2P raises attempted index
  visits versus the reviewed bytes (900K capacity fixture: P = 12,470,208 ->
  ~28.5M of the context's 64M parent cap, `two_plane_context_v1.py:66`;
  whole-buffer query loads add E on top). Record the new model in the
  successor resource gate; a refusal point may move, never an extremum.

## What was not checked

- Nothing was executed: no test outcome, CPU figure, needle-width occurrence,
  "283/132 passed" claim or printed work number was reproduced.
- The previously reviewed bytes are unavailable; equivalence of everything
  outside the described delta is inferred from predecessor diffs and the
  reviews' line citations, not proved by a byte diff.
- The certificate helper was read only far enough to estimate the needle's
  lower-bound collapse; its enclosure proof was not re-derived.
- `_trim_indices`, `_v2._rank`, plane initialization V14, mesh offline V15,
  the integration/schedule tests, wiring/runner, protocol/PRE, resource gate
  and any physical/Gold data are out of scope.
