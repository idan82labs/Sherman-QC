# Adaptive sampling and mixed-radius engine checkpoint

Date: 2026-09-07. Source-only, invented in-memory geometry. This continues the
[position-only candidate-birth work](2026-09-07_position_region_discovery.md).
It is not a physical evaluation, runtime promotion, thickness measurement or new
execution authorization. The long-running goal remains paused.

Continuation: the [cone-closed region checkpoint](2026-09-07_cone_closed_region_checkpoint.md)
records a separate uncertainty-aware alternative and a case-by-case comparison.
The baseline source, policies and receipts below remain unchanged.

## Engine change

Three new modules address the previous fixed-ball sparse-sampling failure and
single-radius mixture failure without changing the V1 engine sources:

- [Adaptive plane seeds and observed graph](../../research/geometry/adaptive_position_regions.py),
  with [91 focused tests](../../tests/test_adaptive_position_regions.py).
- [Plane-conditioned two-radius witness](../../research/geometry/plane_conditioned_two_radius.py),
  with [focused tests](../../tests/test_plane_conditioned_two_radius.py) and
  [independent ambiguity tests](../../tests/test_two_radius_adversarial.py).
- [Position-only integration](../../research/geometry/adaptive_bend_candidates.py)
  and [structural/noise regression receipts](../../tests/test_adaptive_bend_candidates.py).

The integration accepts only positions, units and explicit error/search policies.
It preserves complete associated transition components, tries the existing single
radius first, and then tries two concentric radii after an abstention without an
explicit numerical-unavailability signal.
No point is removed to improve a radius fit. Successful single-radius candidates
do not exclude other multi-surface explanations. These modules are not imported
into the product detector; runtime, UI, evaluator and frozen mechanisms are unchanged.

## Research grounding and declared hypotheses

Nearest-neighbor scales and local plane fitting are established point-set tools;
CGAL documents spacing and normal estimation using neighbor queries. Our capped,
complete-tie, mutual-link policy is a development hypothesis, not a CGAL topology
guarantee or a claim of a new neighborhood method.
[CGAL Point Set Processing](https://doc.cgal.org/latest/Point_set_processing_3/index.html).

Products of implicit polynomials can represent unions without supplied point
memberships. Vidal, Ma and Sastry use this idea for generalized PCA of subspaces.
We borrow the representation idea, not their subspace theorem, for a concentric
circle center proposal. No theorem about complete/noisy circle recovery follows.
[GPCA paper](https://www.cis.jhu.edu/~rvidal/publications/cvpr03-gpca-final.pdf).

Algebraic circle fitting and geometric fitting have different error behavior.
Accordingly, algebraic coefficients only propose a center; acceptance requires
all-point geometric verification and separate conditional radius enclosures.
[Al-Sharadqah and Chernov, circle-fitting error analysis](https://arxiv.org/abs/0907.0421).

### Adaptive neighborhoods and uncertainty

For each unique point and each declared `k`, use its observed distance to the kth
OTHER unique point, closing all distance ties. Require that nominal radius to be
within the explicit cap plus one numerical guard; tie closure can add one more
guard. Skip an over-cap query in full. Do not lower k, clip support, trim residuals
or merge merely similar regions. Only identical support tuples are deduplicated.

Under simultaneous position error `epsilon`, every pair distance and its kth
order statistic change by at most `2 epsilon`. Their difference therefore changes
by at most `4 epsilon`. Diagnostics disclose rank-boundary sensitivity using
`4 epsilon + 2 guard`, and cap sensitivity using `2 epsilon + guard`. This does
not certify selected membership. Epsilon does not expand nominal neighborhoods
or graph edges.

Graph edges require mutual observed k-neighbor membership, including complete
ties, and the observed link-distance cap. They are **sampling policy only**, not
pairwise-possible true-gap links, physical contact or material continuity. This
differs deliberately from V1's possible-gap graph. Fixed-seed-plane extension
still uses induced compatible components. Each disjoint extended-plane pair
retains every whole remainder component linked to both planes.

Caps remain small and explicit: 256 RAW rows before conversion/deduplication,
at most four neighbor counts, 64 accepted regions, 2,016 disjoint pairs, 256
associations and at most 255 radial cuts per two-radius association. An association
can invoke one single and one two-radius wrapper; the latter currently repeats
the baseline single analysis internally. Budgets are not claims about total
primitive operations. Exhaustion or explicitly reported numerical unavailability clears candidate
prefixes and marks search incomplete, including when an earlier candidate passed.

Independent review found an inherited observability limit: the unchanged legacy
circle fitter can swallow a solver exception as a missing proposal or stop its
refinement without exposing a solver-failure reason. New outputs disclose this.
A regression injects an initial solver failure, then independently checks the
later retained two-radius witness against every sample. Its validity does not
depend on the failed proposal, but declared-policy completion must not be read as
complete solver-health reporting or exhaustive feasible-model search. No blanket
guarantee that every internal numerical failure clears results is made.

### Two-radius proposal and conditional bounds

Project the full associated component along the plane-derived axis hypothesis,
using the existing common-rotation error relaxation and a disclosed projection
guard. With normalized projected point `q=(x,y)` and `s=x*x+y*y`, two concentric
circles satisfy

```text
(s - 2 c.q + d1)(s - 2 c.q + d2) = 0,
d_j = ||c||^2 - r_j^2.
```

The coefficients of `s*x` and `s*y` are `-4 c_x` and `-4 c_y`. A rank-eight linear
least-squares fit to `[s*x,s*y,x*x,x*y,y*y,x,y,1]` proposes one center. The other
polynomial identities are not enforced by that fit. Instead, at that center,
enumerate all radial-order cuts. Each band must have enough projected samples,
all points must lie within the common error budget of their minimax band radius,
and both fixed-axis radius diagnostics must pass with the proposed witness radius
inside the corresponding interval. No feasible center or segmentation search
completeness is claimed; proposal/rank failure can lose a real candidate.

Both bands use the SAME full-component rotation budget; a smaller per-band lever
arm is not substituted. A band's interval is conditional on its proposed sample
membership and the axis-rotation relaxation. It is not a global bound over every
partition. The incompatible pooled single-radius interval is not exported by the
two-radius wrapper. Physical face count, inner/outer identity and thickness remain
unavailable; a relaxed sample witness does not prove an original joint model.

Independent tests preserve two essential ambiguities:

- Eight samples can admit radii `{2,4}` about one center and `{2.5,4.5}` about
  another. Tiny angular perturbations make the algebraic design full rank while
  retaining both models within `epsilon=0.001`. Proposed-band intervals need not
  contain the alternative partition's radii. Rank is not identifiability.
- Two-radius samples can also lie within the error budget of TWO flat faces,
  although a pooled SINGLE-plane alternative fails. The current flat check is not
  exhaustive competing-surface model selection.

## Fixed structural development receipt

The 150-case matrix uses five strata, ten signed angles (`+/-30,60,90,120,150`
degrees), and radii `0.5,2,20`. All cases use one scale-relative policy: discovery
`k=8`, graph `k=10`, neighborhood/link cap `0.75R`, error `5e-7R`, axial gap
`0.255R`, radius-width limit `0.05R`, minimum region/band size 9 and angle bounds
0.1 radians. Mixtures add samples at `1.1R`. No per-angle tuning occurred.

| Invented stratum | Conditional cases | Abstentions | Oracle-matched radius checks |
| --- | ---: | ---: | ---: |
| Clean | 30/30 | 0 | 30 single |
| Missing middle angular column | 30/30 | 0 | 30 single |
| Concentric mixture | 30/30 | 0 | 60 band |
| Sparse flange stations over `2R` | 24/30 | 6 | 24 single |
| Sparse flanges plus concentric mixture | 18/30 | 12 | 36 band |

All 150 searches completed; all 84 matched single-radius and 96 matched-band
checks contain their constructed true radius. Multiple candidate alternatives
remain: the missing-column stratum has 42 conditional candidates, not 42 physical
bends. Its positive status does not prove continuity across the missing samples.
The exact two-radius denominator is explicit: 96 evaluated, 96 proposed and 96
oracle-matched band intervals, with zero unmatched bands.

The six sparse failures are all `|30|` degrees; the twelve sparse-mixture failures
are `|30|` or `|60|`. Plane seeds alone do not solve graph/extension connectivity.
A separate regression keeps the `graph k=8` mixed-density seam-fragmentation
failure. Choosing k=10 was a development response to that observed failure, not a
prospective evaluation. The new graph also has a different distance policy from
V1; these results do not isolate a single causal improvement or establish physical
generalization. V1's original failing tests and source remain unchanged.

Root independently executed `run_adaptive_structural_matrix()` in 1.40219 seconds,
peak process RSS 49,397,760 bytes on Darwin. Timing includes only this bounded
synthetic Python/NumPy workload, not a scan/native runtime or process-tree bound.

## Noise sensitivity: important failure retained

The additional fixed-seed (`2026090717`) 150-case matrix perturbs all 14,700
positions in the same constructions. Actual displacement norms range from
`0.2500009243` to `0.9499925473` times the ORIGINAL declared epsilon. No thresholds,
budgets or geometry were changed after observing the outcomes.

| Invented stratum | Conditional cases | Incomplete cases | Matched / proposed interval checks |
| --- | ---: | ---: | ---: |
| Clean | 30/30 | 0 | 30 / 1,407 single |
| Missing middle arc | 30/30 | 0 | 30 / 1,644 single |
| Concentric mixture | 24/30 | 6 | 46 / 1,604 band |
| Sparse flange stations | 24/30 | 0 | 24 / 1,233 single |
| Sparse plus mixture | 16/30 | 6 | 30 / 898 band |

Single denominators include retained abstained association diagnostics, not just
positive models. Band denominators count intervals in retained two-radius models.
Exact observed-row correspondence is used ONLY by the test oracle: both complete
flange supports and the complete transition must match the constructor; a mixture
band must additionally match an entire constructed radial layer. Unmatched
hypotheses are unscored, not silently called accurate or necessarily incorrect.
All 84 matched single and 76 matched band intervals contain their known radius.
Dense and sparse-mixture strata respectively have 23 and 15 fully matched joint
associations, fewer than their 24 and 16 top-level conditional cases.

This is substantial discovery/association instability despite tiny bounded noise,
not stable segmentation. A root replay of the candidate-birth stage for the 60
noisy mixed constructions located all 12 incomplete searches at `+/-30` degrees:
each exceeds the unchanged 64-region candidate cap. The cap was not raised and
no candidate prefix was selected. The expanded alternatives in other cases also
explain why clean-case candidate counts cannot stand in for complete-part recall.

## Executed verification and source snapshot

Root's final combined 12-file geometry suite passed **652 tests in 22.31 seconds**:
the previous 458 regressions plus 194 new tests (91 adaptive discovery/graph,
24 two-radius, 24 independent adversarial and 55 integration tests). This run
includes both fixed 150-case matrices and the final explicit denominator checks.
Independent review confirmed whole-component preservation, scoped graph/interval
claims and the two failure/accounting corrections documented above.
The seven new Python source/test files parsed successfully; 25 scoped local links
in the three updated checkpoint documents resolved. Narrow whitespace checks
reported no errors. No unchanged product-support/UI suite was claimed as rerun.

Reproduce from `FullProject/`, without a corpus runner or application imports:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_adaptive_position_regions.py tests/test_plane_conditioned_two_radius.py tests/test_two_radius_adversarial.py tests/test_adaptive_bend_candidates.py tests/test_position_plane_regions.py tests/test_position_bend_candidates.py tests/test_position_discovery_adversarial.py tests/test_analytic_bend_fixture.py tests/test_axis_conditioned_radius.py tests/test_position_axis_hypothesis.py tests/test_plane_conditioned_radius.py tests/test_axis_radius_independent_bounds.py -q -s
```

The named integration test exposes `run_adaptive_structural_matrix()` and
`run_adaptive_noise_sensitivity_matrix()`; neither accepts a real data path.
Source/review tasks ran in parallel, while these small numerical workloads used
one BLAS thread. No extra dependency or accelerator was needed. The 16 GiB
process-tree ceiling and all tighter frozen limits remain in force; these small
synthetic timings do not establish native-runtime resource margins.

Final source SHA-256 values:

```text
adaptive_position_regions.py      9dbb26dd651f40bba40b0825db491054533a7ae80c09adccb6a66c9eeddd7175
plane_conditioned_two_radius.py   a836857141533e61d4b0af9dc3ac58efc24bfec7dbaca9fd101ad36029d663e6
adaptive_bend_candidates.py       5e946ee1dbb544b7195179f15fece1f97501adaa027aae73e0e7de26363246e0
```

The prior `position_plane_regions.py`, `position_bend_candidates.py` and
`plane_conditioned_radius.py` hashes still match their previous checkpoints
(`ad0278d...`, `8318853f...`, `c2b2a297...`). Their failing development cases and
recorded policies were not overwritten by the new alternative.

## Decision and next engine work

Retain this as an additive research candidate: it demonstrates useful new mixed
and sparse-sampling capability without claiming physical identity or discarding
contaminants. Do not promote it into the active detector from these receipts.
Next, target graph/extension stability under nonuniform sampling and bounded
noise, then adjacent bends and competing flat/multi-surface explanations. A
scanner-scale path, calibrated capture errors, physical tangency/common-object
evidence and a frozen prospective comparison remain necessary for qualification.

No raw scans, reserved incoming-email parts, held-outs, pools/manifests, sealed
truth, control/issuance records or scientific outputs were accessed. No native
geometry job, physical evaluator, runtime service, dependency installation or UI
work ran. C15's retained freeze failures, EXP35's terminal state, frozen resource
budgets and the paused goal are unchanged.
