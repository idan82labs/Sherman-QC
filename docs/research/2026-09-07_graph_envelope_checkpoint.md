# Sparse-graph failure attribution and interval-rank hypotheses

Date: 2026-09-07. Engine-only continuation of the
[locked roadmap](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07).
This is bounded source/synthetic development, not production qualification or
authorization to open physical inputs. The CAD-assisted/operator-reviewed
destination, milestone gates and change control remain unchanged.

Follow-on: the [adjacent-bend seed checkpoint](2026-09-07_adjacent_bend_seed_checkpoint.md)
adds a fixed independent challenge, revises the presumed failure stage, and tests
full cap-ball seeds at the unchanged final error/work limits. This checkpoint's
source and historical measurements remain intact.

## Outcome and implementation boundary

The remaining sparse-mixture noise losses are traced to flange graph cuts, not
radius arithmetic. Two separately declared graph hypotheses improve the retained
[cone-closed baseline](2026-09-07_cone_closed_region_checkpoint.md):

- A mutual interval-rank outer graph recovers the two extra noise losses, from
  16/30 to 18/30 complete-support conditional sparse-mixture cases.
- A symmetric-union interval-rank outer graph also bridges the recorded density
  cuts: all five constructed strata give 30/30 complete-support conditional cases
  in both exact and bounded-noise runs. No individual baseline-matched support or
  conditional support is lost in these matrices.

These are development constructions from the same fixture family, not unseen-part
accuracy. Union is a different sampled-connection policy, not a guarantee about
the true mutual graph. More links can merge unrelated objects or hide good
subsets; all-point refits remain essential but do not establish physical identity.
The old sources, policies, failing receipts and active detector remain intact.

New additive source:

- [Fixed-node graph envelopes](../../research/geometry/position_graph_envelope.py)
  and [independent graph tests](../../tests/test_position_graph_envelope.py).
- [Envelope-graph plane discovery](../../research/geometry/graph_envelope_plane_regions.py)
  and [closure/identity counterexamples](../../tests/test_graph_envelope_plane_regions.py).
- [Whole-component radius integration](../../research/geometry/graph_envelope_bend_candidates.py)
  and [paired matrix and fail-closed tests](../../tests/test_graph_envelope_bend_candidates.py).
- [Independent sparse failure attribution](../../tests/test_sparse_graph_attribution.py).

## Why the baseline loses sparse flanges

The fixed attribution runs 30 exact and 30 bounded-noise sparse mixtures. The
earlier 120 geometries only advance the original RNG stream; they are not used
for additional attribution or policy search. All k, cap, error and fit settings
are unchanged. The 60 case receipts retain induced flange/transition components,
complete cut counts, seam counts and exact-to-noisy edge changes.

- The 12 exact failures are at signed angles +/-30 and +/-60 degrees, across all
  three radii. Both flanges split into five seam-station samples and twenty other
  samples. Every constructed transition remains connected.
- Noise adds two losses: the first flange at -90 degrees/radius 2 and the second
  at +90 degrees/radius 20. Their removed cut links miss the observed kth radius
  by approximately `1.054e-6` and `3.788e-6`, respectively, within `4 epsilon`.
- The older shallow-angle cuts have rank gaps around `0.22–0.48 mm` at radius 2,
  far beyond the stipulated `epsilon=1e-6 mm`. Error-aware rank closure alone
  should not erase these genuine observed density-policy gaps.

This separated the two hypotheses before their candidate matrix was executed:
uncertain ordering under the same mutual policy, versus explicitly changing
symmetrization to permit one-direction neighbor membership. There was no per-case
k selection, distance-cap increase, epsilon increase or favorable-prefix return.

## Derived graph envelope

Canonical distinct OBSERVED rows define fixed node identities. They are deduped
once after enforcing the 256 RAW-row cap. Hypothetical corrected positions that
coincide retain those identities; the definition is not a graph on rededuped
latent positions.

Assume simultaneous position displacements at most epsilon. An observed pair
distance `d_ij` then gives engineering intervals

```text
L_ij = max(0, d_ij - 2*epsilon - guard)
U_ij = d_ij + 2*epsilon + guard
```

For a candidate directed edge i to j, count competitors l excluding BOTH i and j.
With complete kth-distance ties retained:

```text
lower membership: count_l(L_il < U_ij) < k
outer membership: count_l(U_il < L_ij) < k
```

The lower condition leaves fewer than k potentially closer competitors, hence
guarantees membership under the stipulated bounds. If the outer condition fails,
at least k competitors are definitely closer, so membership is excluded. Passing
the outer condition is merely failure to exclude by independent intervals.
Strict comparisons preserve ties; excluding j matters even for k=N-1.

Mutual symmetrization uses AND of the two directions; symmetric union uses OR.
Lower edges additionally require `U_ij <= nominal cap`; outer edges require
`L_ij <= nominal cap`. The lower graph is sufficient, not exhaustive; even a
guaranteed union edge need not be guaranteed in either direction separately.
Lower, guarded-observed and outer graphs are serialized separately, with `edges`
explicitly selecting the OUTER proposal graph.

The interval tests are direct derivations here, not statistical confidence bounds.
The arithmetic guard is not directed rounding or a numerical certificate.
[von Luxburg's primary tutorial, sections 2.2 and 8.1](https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf)
defines mutual versus symmetrized k-neighbor graphs and discusses their differing
behavior across density scales. That background motivates this comparison; it
does not establish physical sheet-metal connectivity, scanner calibration or a
finite-sample success guarantee for this implementation.

## Counterexamples that limit the claim

An individual upper edge can be impossible. For collinear observed coordinates
`0, 1, 1.1, 2`, epsilon `0.3`, and k=2, separate distance intervals retain the
0-to-2 proposal. Yet both middle nodes remain closer for ANY permitted full-3D
corrections. For each observed competitor a in `{1,1.1}`, rationalizing the
query-to-target minus query-to-competitor distance gives the uniform lower bound
`(2-a)*(2+a-2*epsilon)/(2+a+2*epsilon) > 2*epsilon`; moving the other two endpoints
cannot reverse the ordering. The outer graph is not even an individual-edge
realizability witness.

Nor are individually possible edges necessarily jointly possible. An equilateral
triangle of side 1 with epsilon 0.1 can make any chosen edge length 0.8, but not
all three simultaneously: convexity bounds its perimeter below by
`3 - 3*sqrt(3)*0.1 > 3*0.8`. Both counterexamples are persisted tests.

Added cap edges can join separate parallel patches. A graph-level control joins
two patches whose complete union fails the original-epsilon plane test. Other
controls exercise cone-eligible but jointly incompatible points and separated
coplanar supports. Plane compatibility cannot establish physical continuity even
when every point genuinely lies on one plane. Connected radius contamination is
retained and both single/two-radius models abstain rather than remove it.

## Unchanged measurement and bounded work

Discovery uses the predecessor's fixed cone proposal and whole-support plane
refit at ORIGINAL epsilon. It never falls back to a rejected seed, trims points,
iteratively widens a cone or merges approximately equal supports. The new graph
may enlarge a component but does not enlarge the final measurement tolerance.

Every transition association uses a whole remainder component of the declared
outer graph. Single-radius first, then bounded two-radius search, retains the
earlier conditional axis-rotation relaxation. Original joint model compatibility,
surface membership, physical bend count, thickness, tangency and conformance
remain unestablished. Single-model success does not exhaust competing models.

The same caps remain: 256 raw rows, at most 1,024 seed queries, 2,048 unique plane
fits, 64 FINAL refitted regions, 2,016 plane pairs, 256 associations and 255 cuts
per two-radius association. The final-region unit differs from adaptive V1's
raw-seed cap, as already declared in the cone checkpoint. All budget or explicit
numerical failures clear candidate prefixes and mark search incomplete. The
inherited circle proposal still does not classify every solver failure.

Rank comparisons are bounded cubic work on at most 256 nodes, not a scalable
full-scan implementation. Experiments use one geometry process with single-thread
BLAS. Host RAM does not waive the 16 GiB process-tree ceiling or frozen limits.

## Fixed paired result and honest denominators

The unchanged cone baseline and both new policies receive exactly the SAME rows
in 150 exact and 150 nonzero bounded-noise cases: 900 analyzer calls, with two
candidate receipts per observed array. The original seed `2026090717`, case
order, signed angles, radii, strata and error norms are preserved. Construction
IDs are test-only oracles, never inference inputs. Every candidate's whole graph
component is independently recovered by BFS from serialized edges.

| Noisy stratum, 30 cases | Baseline / mutual / union complete-support conditional cases | Baseline / mutual / union associations |
| --- | ---: | ---: |
| Clean | 30 / 30 / 30 | 30 / 30 / 30 |
| Missing middle arc | 30 / 30 / 30 | 54 / 54 / 54 |
| Concentric mixture | 30 / 30 / 30 | 102 / 102 / 90 |
| Sparse flanges | 24 / 24 / 30 | 24 / 24 / 30 |
| Sparse plus mixture | 16 / 18 / 30 | 52 / 54 / 90 |
| Total | 130 / 132 / 150 | 262 / 264 / 294 |

Exact results for both new policies match their noisy columns; the exact baseline
has 132 matched conditional cases and 264 associations. Neither new policy loses
an individual baseline match, has an incomplete search, or fails a checked truth
containment assertion. Mutual retains 84 matched single and 96 matched band
intervals per noise stratum; union retains 90 single and 120 band intervals.

All proposed band intervals in this fixed matrix match complete joint supports
and full constructed radial layers. Single-interval truth assertions apply only
to full-matched single-surface cases; intervals from pooled mixtures and unmatched
alternatives are separately counted, not truth-scored. Union still has 144
unmatched associations per 150-case matrix: 24 missing-arc and 60 in each mixture
stratum. Twelve extra conditional single-radius alternatives in missing-arc cases
remain, for 162 total conditional candidates versus 150 matched constructions.
The increase from 262 to 294 associations is disclosed, not described as reduced
proliferation. This is a coverage improvement with retained ambiguity.

## Verification and next decision

Root's final focused run passes **158 checks in 14.17 seconds**: 78 graph checks,
15 graph/plane controls, 32 integration checks, four independent attribution
checks and 29 retained axis/radius adversarial checks. The graph checks include
750 independently generated corrected-position realizations enclosed by the
lower/outer graphs, as finite tests rather than proof of all perturbations.
Independent source and paired-oracle reviews found no blocking issue. The review
did tighten band counts to include ALL proposed intervals and remapped band rows
independently through their serialized positions before the final run.

The matrix peaks at seven final regions and 118 unique plane evaluations per
candidate call. `/usr/bin/time -l` reports 68,009,984 bytes (about 65 MiB) maximum
process RSS for the complete focused run, with no child geometry workers. This is
not a scanner-scale or product-latency benchmark. Earlier 652/140-test receipts
remain historical; those entire suites were not rerun or claimed as rerun.

Reproduce from `FullProject/`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_position_graph_envelope.py tests/test_graph_envelope_plane_regions.py tests/test_graph_envelope_bend_candidates.py tests/test_sparse_graph_attribution.py tests/test_two_radius_adversarial.py tests/test_axis_radius_independent_bounds.py -q -s
```

`run_matrix()` returns all 600 candidate-case receipts and 20 stratified summaries;
`run_sparse_graph_attribution()` returns the 60 baseline classifications and cut
receipts. Neither helper accepts a physical-input path.

Final additive engine SHA-256 snapshot:

```text
position_graph_envelope.py       8044d58c86327d777956c7d3e58a1bc59b0f1de6149f24dbbdae37a7c019a047
graph_envelope_plane_regions.py  3a9475e0aa8958834a883188f5fcf5598e1d6aefa5d6507fcf76cae93ef09371
graph_envelope_bend_candidates.py c94ad869f3fc91252f5a6046554890b73196159d0d3bd2c5e87f95c5e51b58be
```

All eight checked predecessor hashes (fixed/adaptive/cone plane and association,
plane-conditioned single/two-radius) still match their dated receipts.

Continue the union envelope as an explicitly scoped research hypothesis while
retaining mutual and observed baselines. This closes the named sparse cuts on
this matrix, NOT priority 1 universally and NOT M3 or production readiness.
Next are adjacent-bend/competing-surface/occlusion controls and a bounded
scanner-scale proposal path, followed by CAD-assisted event mapping and local
measurements under the locked order. Capture model/export information was asked
for preparation only; no scan or held-out access is implied.

No runtime promotion, UI/dependency changes, physical/held-out/email reads or
control/issuance-record access occurred. M5/M6 remain unstarted; C15 freeze
failures, terminal EXP35 and the paused long-running goal remain unchanged.
