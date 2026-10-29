# Preliminary independent source/math review: surface V4 and registration V4

Review checkpoint: 2026-09-12 22:25 UTC. Reviewer: Codex subordinate
`fable_surface_v4_review`, independent of implementation. Fable 5.1 / max worker
`1519fdc2` remains the sole engine/test implementation owner.

## Disposition and limits

No concrete P1 enclosure, pruning, finite-support or live uncertainty-propagation
blocker was found in the examined bytes. One P2 contract-hardening/test gap is
identified below. Repair that narrow gap before claiming the new uncertainty
policy is validated; it does not call for another broad readiness suite.

This is PRELIMINARY SOURCE REVIEW ONLY, not a frozen independent PRE, execution
authorization, promotion, efficacy result, physical validation or audited F1.
The candidate may still be changing. Rebind and review any later changes before
using this document in an exact-source gate. Matching beginning/end hashes below
support the stated review binding; they do not freeze the repository or attest
against intermediate changes reverted between observations.

No candidate import, collection, Python execution, test run or geometry job was
performed by this reviewer. The coordinator relayed an author report of 132
passing surface/registration tests; this review does not independently verify
that run, its command, custody or bindings. No raw/Gold/input data, artifacts,
caches, credentials, new data, Claude logs or live worker internals were read,
statted, hashed or otherwise inspected. The only edit is this report.

## Examined source bindings

All paths below are repository-relative. The first five files were fully read;
the four primary source/test hashes and sizes matched before and after review.
The stage-policy seam was added by explicit coordinator scope and its binding
was checked again at the end. Total for these five files: 135,783 bytes.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `backend/cad_guided_triangle_surface_v4.py` | 36791 | `b42635fdde34b8e7c682cf4e18db9c2b8cc2955ea5b88ff59f5c0ba79e64ab4a` |
| `backend/cad_guided_global_registration_v4.py` | 36998 | `adde05c8f556b8d7381f7cd391d01747bc28df5b1e9a541094d53a3e434f2940` |
| `tests/test_cad_guided_triangle_surface_v4.py` | 25896 | `0a6f6b1fefdb21e162e7401bf3520be54f08a2c01a9098ba4ad5635d75e5fd62` |
| `tests/test_cad_guided_global_registration_v4.py` | 32770 | `d22fb868ed31fa05ec103b95bc1ca264361ca4c9b80fc33559505515cd013ae3` |
| `backend/cad_guided_stage_policies_v2.py` | 3328 | `699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d` |

Narrow supporting reads: the full inherited distance-certificate helper;
lines 1-205 of its test source, including the independent exact KKT oracle;
and scoped pairwise source/test diffs against the named V3 predecessors. These
are supporting bindings, not a complete execution/import closure:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `backend/cad_guided_triangle_distance_bounds_v3.py` | 10776 | `2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7` |
| `tests/test_cad_guided_triangle_distance_bounds_v3.py` | 19151 | `09deb22ee510fe62e1c69dd1469005ffb949fbea25179755bf713998d2ab254a` |
| `backend/cad_guided_triangle_surface_v3.py` | 32098 | `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10` |
| `backend/cad_guided_global_registration_v3.py` | 35987 | `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0` |
| `tests/test_cad_guided_triangle_surface_v3.py` | 17338 | `e0d0be929915829c95c43e794127818d8a9c1b7fede18981926844c71c09e1bc` |
| `tests/test_cad_guided_global_registration_v3.py` | 30723 | `0e774293e1314d81db6567abf782601304508b5cc71b8f9d9d16e4337a441ace` |

Onboarding read: `AGENTS.md`, newest `docs/CURRENT_STATUS.md` checkpoint,
`docs/REPO_MAP.md`, `docs/operations/astra-workflow.md`, and the Fable handoff
`docs/research/2026-09-12_fable51_clean_f1_handoff.md`. Historical instructions
were not treated as permission to execute or reopen a terminal attempt.

## P2: wide-counter acceptance test does not enforce the new exclusion promise

Evidence:

- `tests/test_cad_guided_global_registration_v4.py:104-110` constructs all
  boolean fields as false. Lines 624-629 then change only
  `interval_width_uncertain_pairs` to 2 and explicitly require the consumer to
  accept the result. Thus the test blesses positive wide-pair telemetry with
  no normal/point-uncertain row.
- `backend/cad_guided_global_registration_v4.py:120-144` validates telemetry
  types, work accounting and completed certificates but does not connect the
  new counter to uncertainty. Lines 201-204 also leave that contradiction
  unchecked. `_unambiguous` at 264-268 then returns true for every such mocked
  row, allowing distance-in-gate rows into `_chosen` at 274-277.
- The CURRENT LIVE PRODUCER does not exhibit that omission:
  `backend/cad_guided_triangle_surface_v4.py:559-568` sets BOTH uncertainty
  flags for every competitive wide pair, and later comparisons use `|=`
  (631 and 639), preserving them. This finding is a structural safety-check
  and regression-control gap, not evidence that the current producer silently
  trusts a wide row.

Minimal repair:

1. Make the positive-counter acceptance fixture semantically consistent by
   setting both uncertainty flags for its affected rows. Add a negative case
   for a positive wide counter with no jointly uncertain row. In the complete
   query validator, enforce necessary counter/flag consistency; at minimum,
   positive wide-pair count must imply a row with both flags. A tighter cheap
   necessary bound is that the counter cannot exceed the summed near-tie
   capacity of rows carrying both flags. These checks do not independently
   prove which particular competitor was wide and must not claim to do so.
2. Add ONE focused mixed-row test using the actual V4 needle/sound geometry
   already in `tests/test_cad_guided_triangle_surface_v4.py:418-442`, passed
   through `_Queries` and the stage classifier. Confirm a wide affected row
   may retain distance-only proximity when its upper bound meets the gate,
   but has no proposal target, legacy unambiguous support or tangent
   constraint; unaffected sound rows retain their original classification.
   Existing consumer tests at 201-226 already cover boolean exclusion and
   no-refill trimming, so reuse those seams instead of expanding scope.

The new test also sets all near-tie counts to 256 without reconstructing their
pair telemetry (628-629). That isolates the accepted field maximum, not a real
256-competitor execution; do not cite it as such. Real surface fan controls are
separate and retain actual pair counts.

## Mathematical and semantic checks

### Vertex upper bound and two-pass finite search

`surface_v4:203-215` rounds the magnitude of every represented-coordinate
difference upward, then squares, sums and takes square root with outward
rounding. Under the inherited round-to-nearest, gradual-underflow and correctly
rounded basic-operation/sqrt model, this gives `Uv >= ||p-v||`. Underflow makes
the bound looser, not smaller. Inputs are finite and coordinate-bounded; failure
to obtain a finite threshold refuses the result (472-484).

Every indexed `v` is copied from represented triangle vertices (283-288).
Therefore the exact union distance `d* <= ||p-v|| <= Uv`. The AABB bound is a
downward L-infinity bound, hence also a Euclidean lower bound (395-398).
Any node/triangle pruned by `Lbox > min(Uv, Ubest)` cannot contain a true
minimizer, because both upper bounds are feasible union-distance bounds.
The correctness argument does not require cKDTree to find the mathematically
nearest vertex: any valid returned vertex index remains a feasible witness;
nearest selection concerns tightness/performance only. Returned tree distances
are not trusted as certificates (477-482).

Pass 2 uses the fixed certificate upper bound plus tie tolerance, not `Uv`
(496-499). It rebuilds the lower envelope (598-607) and preserves every true
minimizer through sound box pruning. Selected materialized points and all
reported distance/error bounds still come from the four certificate candidates
(337-391, 527-549), not from cKDTree or infinite-plane projection. The inherited
helper's feasible convex-hull primal/dual construction and selected-triangle
strong-convexity error bound remain unchanged. No new certificate-math blocker
was found. This is a conditional represented-geometry argument, not a native
CAD or physical-error certificate.

The initial vertex threshold can prune an irrelevant triangle even if that
triangle would supply a numerically smaller certificate upper than another
evaluated witness. Consequently V4 need not globally minimize the list of all
per-triangle certificate upper endpoints exactly as V3 did. That does not break
the union-distance enclosure above. Avoid a general bitwise-V3-equivalence
claim; the generic-soup equality test only establishes its particular fixture
if independently executed and audited.

### Wide intervals, proximity, correspondence and rank

The wide competitor is retained, counted and compared; it is not deleted
(`surface_v4:551-576`). Both uncertainty flags persist through completion.
In registration, definite correspondence support requires both an upper bound
inside the gate and `_unambiguous` (`global_registration_v4:253-277, 431-436`).
Rank uses the further face-interior/non-region-uncertain subset (436-452).
Thus the new flags conservatively exclude affected rows from optimizer targets,
definite registration support and tangent rank.

The explicitly added stage-policy seam is consistent with this distinction:
`stage_policies_v2:61-75` retains distance-only proximity, but computes proposal
targets and tangent constraints from `_unambiguous`. A wide interval with a
valid upper bound below a proximity gate is still legitimate evidence of
represented finite-union proximity; it is NOT correspondence, an observed bend,
a manufactured measurement or acceptance of its normal.

Possible-support accounting can retain uncertain rows but does not promote them
to definite acceptance (`global_registration_v4:280-283, 454-482`). Possible-only
distinct competitors may still block selection (605-627). Empty fixed-support
metrics remain null, not zero-error observations. Reverse sample coverage,
rank, materialized-target optimization, transformation rounding and pose-class
comparison retain their declared engineering limitations.

### Fatal failure and tie capacity

Certificate unavailability still raises (527-532); tie overflow still raises
(570-572); the outer refusal returns every result array as null (663-665).
The adapter charges attempted valid work before raising a query failure
(`global_registration_v4:227-250`). `_select` first clears a stale selection and
requires complete candidate evidence (548-555). Nothing examined permits
salvaging a favorable prefix after a later fatal query. The enclosing workflow
caller was outside this review and must retain that behavior in its own gate.

The 256 tie limit is a new declared versioned capacity, not deletion of excess
competitors: all stored pairs are compared and charged (`surface_v4:621-641`),
and overflow still refuses all rows. It increases worst-case pair work to
`256*255/2 = 32640` comparisons per query row. The existing normal-pair/work
budgets still bound admitted comparison work. No speedup is inferred from the
capacity increase.

## Test delta assessment and narrow documentation repairs

The read source/test diffs retain V3 exact finite-hull containment, local-scale
accuracy controls, holes, outside-plane controls, tie completeness, all-pair
normal comparison, failure clearing and cumulative-budget controls. The
remote-distractor fixture moves from z=1 to z=.3 so it remains inside the new
vertex threshold and still exercises its intended first-pass weak-lower seam;
its exact oracle and expected four witness calls remain in place. This is an
explained fixture adaptation, not a relaxed correctness assertion.

The newly admitted needle fixture asks for exact enclosure and both uncertainty
flags, not the accuracy promise reserved for well-conditioned rows. The policy
change intentionally broadens COMPLETE distance-query outcomes; it does not
justify claiming unchanged whole-query admission or physical support. The P2
consumer gap above is the one concrete new control weakness found.

Two small documentation fixes are warranted: `surface_v4:406-408` still says
every competitor must pass its width gate, and 557-558 still calls a wide
competitive interval fatal. Replace those stale V3 statements with the V4
row-uncertainty policy. Also clarify that pass 1 selects among evaluated
certificate uppers, as explained above. These are not enclosure failures.

The generic-soup comparison at `tests/test_cad_guided_triangle_surface_v4.py:458-476`
asserts fewer triangle/certificate pairs but does not assert lower complete
build-plus-query CPU or total work. Vertex deduplication/tree construction
(`surface_v4:283-287`) and the declared 16-unit lookup charge must not disappear
from any future efficacy argument. No performance qualification is supplied by
this review; resource/process review and exact execution evidence remain
separate responsibilities.

## Handoff

Fable owns the narrow repairs. Codex should inspect their exact delta, preserving
the original review binding and this P2 finding, then use only the separately
authorized source/test gate appropriate to those changes. Do not rerun unchanged
successful suites just to satisfy this document. A physical successor still
requires its separate source/protocol freeze and independent PRE; this report
does not replace either. No engine or test edit, raw access, physical run,
promotion or numeric F1 is authorized or established here.
