# Local CAD-root grouping: fixed paired proof checkpoint

Date: 2026-09-07. Source and invented CAD records only. This continues the
[dense measurement checkpoint](2026-09-07_dense_axis_tube_checkpoint.md) under the
[locked roadmap](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07).
No physical, held-out, email, control or one-shot artifacts are accessed.

Follow-on: a connected-module audit retained this scoped grouping result but
blocked runtime integration on attribution and provenance defects. The subsequent
[attribution repair and face-occurrence handoff](2026-09-07_cad_attribution_provenance_checkpoint.md)
records the user-authorized remediation and new verification. The grouping code
and evidence below remain unchanged; native-root identity is still unverified.

## Proof target frozen before first execution

Positive progress means a better structural partition on IDENTICAL supplied
candidate records, while retaining every original candidate and finite segment.
The comparator is an explicitly declared geometry-only ablation of the new
policy, NOT a replay or benchmark of the native production extractor, EXP10,
or any physical experiment. Oracle grouping partitions exist only in independently
authored tests; owner/face identities are intentionally supplied source metadata.

Four fixed constructions are each evaluated under three deterministic proper
rigid poses, giving 12 cases and 24 input candidate identities:

1. Collinear nearby segments, distinct owners and native root faces: two groups.
2. Same-owner root faces connected only through an unselected flange: two groups.
3. Coincident geometry with distinct owners/native root faces: two groups.
4. Two selected root faces directly sharing a native edge: one group.

All four use geometry compatible with the same geometry-only thresholds. The
predeclared expected comparison is 3/12 exact partitions for the ablation versus
12/12 for local-topology grouping, with false contractions reduced from nine to
zero and the connected-fragment control retained. These are acceptance targets,
not measured results until the verification receipt below is filled in.
The whole paired matrix has fixed 120-second / 512-MiB soft resource targets.
Separate adversarial tests must leave unknown topology, incompatible components,
numerically unsupported inputs and exhausted budgets unresolved. No fixture,
threshold or truth-label adjustment is authorized to rescue a failed result.

## Why this seam precedes geometric contraction

Read-only inspection of the primary runtime source found that
`backend/feature_detection/bend_detector.py::CADBendExtractor.extract_from_mesh`
receives triangle connectivity but runs detection on vertices and emits bend
specifications without native face ownership. `_merge_rolled_specs` then groups
by direction, angle and distances. Nearby separate roots therefore lack an
ownership barrier before averaging geometry and enclosing member endpoints.
For example, the source's default rules admit two otherwise equal rolled specs
with collinear segments [0,10] and [11,21]: midpoint distance is 11 mm and nearest
endpoint gap is 1 mm. This is a source-derived counterexample, not native execution.

`backend/cad_import.py::_tessellate` appends vertices separately for each native
face, offsetting its triangle indices. Mesh vertex-index connectivity therefore
does not establish native face disconnection; coordinate welding would not
establish native identity either. A reliable product adapter must preserve
native face/edge/owner provenance before tessellation and before merging.
The legacy `backend/cad_bend_extractor.py` also has geometric pruning paths; it
was inspected, not executed or changed.

The released [EXP10 result narrative](2026-08-28_exp_cad_0010_ai_consensus_grouped_capacity_result.md)
already records a terminal rejected attempt that merged disconnected collinear
tab roots and lost a true positive. It is motivation only, not reused evaluation
material. This checkpoint does not rerun, subdivide, rescue or promote EXP10.
Whole-solid identity alone is insufficient: tabs can share a solid through a
common flange without sharing a local bend root.

## Conditional grouping rule and geometry

The new [pure grouping module](../../research/geometry/cad_root_grouping.py)
accepts explicit pre-merge candidate identities, original finite axis segments,
owner identities, nominated root-face identities and native face/edge incidence.
The caller must explicitly declare topology COMPLETE; the module does not
verify that claim or bind `source_id` to source bytes.

Edges are namespaced by owner. The root graph is induced ONLY by nominated root
faces: paths through an unselected flange cannot join roots. Each candidate's
own nominated support must itself be connected, without borrowing another
candidate's faces. A shared vertex or coincident coordinates create no edge.
More than two incident faces on an owner-scoped native edge is unresolved.
This does not validate all nonmanifold defects, edge nondegeneracy, connected
face trimming or oriented solid topology; those remain importer obligations.

Within each connected root, EVERY pair of candidates must satisfy the declared
unoriented-axis angle and lateral-distance limits. For unit directions a,b:

```text
angle = atan2(||a cross b||, |a dot b|)
lateral = max over both segment endpoints of distance to the OTHER supporting line
```

The norm of an affine function is convex, so endpoint maxima bound the lateral
distance over an entire finite segment. Closest infinite-line distance would
incorrectly admit intersecting, divergent axes. All-pair checks prevent a greedy
A-B-C chain from joining mutually incompatible endpoints.

Only directly adjacent or overlapping candidate supports additionally require
an axial projected-interval gap below the declared limit. This gap is the maximum
of the interval gaps computed along BOTH axes, NOT Euclidean closest-segment
distance. It is not imposed on nonadjacent pairs of a connected strip, so a long
root split into consecutive fragments can remain one hypothesis. The output
retains every original finite segment; it never substitutes an enclosing span,
averaged axis, radius or physical location.

All threshold margins must strictly exceed separate engineering roundoff
allowances: 512 machine epsilons for angle and
`512*eps*(maximum absolute endpoint coordinate + maximum segment/offset length)`
for distances. These are numerical guards, not calibrated physical uncertainty
or a directed-rounding certificate. Nonfinite ranges and subnormal distance
guards are unresolved. Subnormal segment lengths are rejected BEFORE direction
normalization. Independent pretest review derived the counterexample: with eta
the least positive float, `hypot(2*eta,3*eta)` rounds to `4*eta`; dividing by it
would produce a non-unit direction and understate a lateral separation. Strict
threshold equality is also unresolved. Both fixes precede first execution.

## Bounds, evidence boundary and next integration step

Declared limits are 256 candidates, 4,096 source faces, 32,768 edge incidences,
16,384 candidate-face memberships and at most 32,640 pair evaluations. Required
within-root pair work is counted before evaluation. Any unresolved component
clears ALL groups, including an otherwise valid prefix; all input records remain.
Malformed contracts raise `ValueError`. No native CAD kernel, parser, scanner,
service, file traversal or child process is invoked by this module.

Successful outputs are CAD-root grouping HYPOTHESES, never physical bend counts.
Observed state stays UNKNOWN, event/physical geometry verification stays false,
and production disposition stays NOT_AUTHORIZED. Native topology completeness,
inner/outer-face manufacturing convention, thickness, registration and finite
scan-supported evidence remain separate obligations. The new seam is not wired
into the runtime extractor; it cannot restore provenance already discarded by
the final merged CAD specifications.

This advances locked Priority 3/M3 at a source/synthetic boundary. M0's bounded
checkpoint and three unwaived C15 failures are unchanged. M1 native-runtime
verification, M2 capture/reference/calibration, M4 durable recovery and trusted
evidence, and M3's physical/generalization exit remain open; M5/M6 have not begun.
There is no defensible completion percentage from this test matrix.

The next primary-product step is a provenance-preserving native import/pre-merge
adapter followed by finite expected-to-observed evidence mapping with explicit
registration uncertainty. This does not authorize native or physical execution;
those remain separately gated. M2 and bounded M4 preparation remain parallel
priorities. No UI changes, runtime promotion or physical reevaluation are included.
Terminal EXP35, terminal EXP10, locked acceptance gates and the paused goal remain
unchanged.

## Verification receipt

Root read the entire new implementation, both independent test files and the
package initialization boundary before the first and only execution. Separate
source/math review identified the subnormal normalization and strict-margin
issues described above; both were fixed before execution. **86 checks passed
(83 unit/contract + 3 progress checks) in 0.12 s**. External wall time was 0.19 s;
maximum process RSS was 55,525,376 bytes (53.0 MiB). There was one test process,
no geometry child process and no native CAD import. The exact command:

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_cad_root_grouping.py tests/test_cad_root_grouping_progress.py -q -s
```

| Fixed construction, three poses each | Geometry-only exact partitions | Local-root exact partitions |
| --- | ---: | ---: |
| Collinear nearby segments, different owners | 0/3 | 3/3 |
| Same owner, connection only through unselected flange | 0/3 | 3/3 |
| Coincident geometry, different owners | 0/3 | 3/3 |
| Directly connected local-root fragments | 3/3 | 3/3 |
| **Total** | **3/12** | **12/12** |

All nine incorrect contractions are prevented, and all three connected controls
remain grouped. All 24 candidate identities are accounted for in both lanes;
the new lane also preserves all 24 original finite segments exactly. No original
input or topology record is mutated. The partition totals are 12 groups in the
ablation versus 21 expected and returned by the new lane. This is structural
correctness under the stipulated local-topology contract, NOT a physical bend
count or measured native-runtime improvement. The matrix contains four invented
constructions under rigid transforms, not twelve independently sampled part
families or a generalization confidence estimate.

Independent scope review confirms that every paired case has only two candidates
and passes the shared geometry policy. Consequently, the paired improvement
isolates the supplied ownership/local-topology barrier; an all-pairs versus
single-link geometry difference cannot explain it. The separate three-candidate
chain-abstention control is not counted as a recovered exact partition.

The fixed matrix, including fixture construction and assertions, took
0.0021790829487144947 s; its process-lifetime RSS high-water mark at receipt time
was 55,508,992 bytes. Receipt formatting is outside that matrix timer. Both
predeclared 120 s / 512 MiB soft targets pass. This small-matrix measurement is
not scanner latency, native parsing, maximum pair-cap performance, cancellation
or a repeated statistical resource qualification.

Independent controls retain connected three-fragment strips despite a large
nonadjacent axial gap; reject angle and offset chaining; check both supporting
lines at distant endpoints; preserve separate same-owner roots; namespace edge
identities by owner; and never infer an edge from corner contact. Unknown or
inconsistent topology, nonmanifold edge incidence, a disconnected candidate's
own support, later incompatible components, exact guarded-threshold equality
and exhausted aggregate budgets produce no groups. Subnormal normalization is
rejected; unsupported pair arithmetic abstains. Every returned result remains
research-only, unobserved and without a physical bend count.

No new code, test fixture or acceptance criterion changed after this passing
execution. Older suites were not rerun. The source identities of the new files:

```text
research/geometry/cad_root_grouping.py
cb9fdd88de78b30f32985bd6ec06082db0cb19be5522b99a6e7424abacf6ae74
tests/test_cad_root_grouping.py
1bdc99ed0207f80e361e1a6a9ee10ca15e76d80e9868f0d566ee5768381c896b
tests/test_cad_root_grouping_progress.py
9f34c3b14401ca7a94becd405afbc7d60eb2932d1e5020b26cdf645988ccd81f
```

Read-only runtime source identities at this checkpoint:

```text
backend/feature_detection/bend_detector.py
14e808a33285a8111b03af8b95042aaf6dc45d8f327447a2f6eda11967f537df
backend/cad_import.py
497154603cd435177f8fb71b9aabab793053893ea6f5e5b38baccacc4352d189
backend/cad_bend_extractor.py
a31bda821bbe71431ea589046cce465d14dd0090159d10dffc42a86414613307
```

The prior dense constraint and streamed-window source hashes still match their
recorded checkpoints. AST parsing of all three new Python files, explicit-file
whitespace/final-newline checks for the six touched files, the four whitelisted
local checkpoint links and the scoped diff check pass. All three executed
source/test identities remain unchanged. No protected artifact identity is
accessed or implied by these hashes.

Decision: positive source/synthetic proof is sufficient to retain this grouping
seam for the next provenance-preserving adapter step, not to promote it to native
runtime or floor use. The primary product direction has advanced from the dense
measurement preparatory work into component-aware CAD grouping. Physical,
registration, native-provenance and operational gates remain open as above.
