# Position-only region birth and conditional bend-geometry association

Date: 2026-09-07. Engine-first implementation checkpoint. This is pure in-memory
source/synthetic work, not a physical evaluation, promoted detector, production
measurement or new execution authorization. The long-running goal remains paused.

Continuation: the [adaptive sampling and mixed-radius checkpoint](2026-09-07_adaptive_mixed_radius_checkpoint.md)
adds a separate engine alternative and noisy failure evidence. The V1 source and
receipts below remain unchanged; the continuation is not physical qualification.

## What is new

The previous [plane-to-radius bridge](2026-09-07_plane_conditioned_radius_bridge.md)
required the caller to supply two plane neighborhoods and a cylinder region.
The new bounded front end accepts **one unlabelled position cloud**, an explicit
unit, a simultaneous position-error bound, and declared search/resolution budgets.
It proposes its own plane supports and transition associations, then calls that
same uncertainty-preserving bridge. CAD, labels, exact normals/axis/radius, known
counts, filenames and trained scores are not inference inputs.

Source and independent checks:

- [Plane-region birth](../../research/geometry/position_plane_regions.py) and
  [focused tests](../../tests/test_position_plane_regions.py).
- [Position-only association](../../research/geometry/position_bend_candidates.py)
  and [integration/math tests](../../tests/test_position_bend_candidates.py).
- [Independently constructed adversarial tests](../../tests/test_position_discovery_adversarial.py).

This removes a supplied-region dependency within a deliberately small declared
search. It does **not** prove the discovered regions are physical flanges, that
the common cylinder/plane model exists, or that every bend in a scan is found.
The code is not imported into the product detector. No previous geometry source,
runtime, scoring, UI, native pipeline or frozen mechanism changed in this step.

## Research grounding and chosen hypothesis

Local plane fits and neighborhood queries are established tools, not a claimed
new segmentation invention. CGAL's shape-detection documentation describes region
growing and configurable neighborhood/fit criteria, including seeding sensitivity.
Here every declared observed-ball seed is retained before exact-support
deduplication; there is no greedy assignment or learned ranking. This is a design
choice for auditable candidate birth, not evidence that it outperforms CGAL or
RANSAC. [CGAL Shape Detection](https://doc.cgal.org/latest/Shape_detection/index.html)

Mitra, Nguyen and Guibas analyze how neighborhood size, curvature, density and
noise interact in normal estimation. Demantké et al. study multiscale spherical
neighborhoods. These motivate explicit scale hypotheses; neither paper calibrates
this implementation's error bound or certifies its scales for sheet-metal scans.
[Normal estimation](https://graphics.stanford.edu/~niloy/research/normal_est/normal_estimation_socg_03_ijcga_04.html),
[multiscale neighborhoods](https://isprs-archives.copernicus.org/articles/XXXVIII-5-W12/97/2011/).

Niyogi, Smale and Weinberger's topology-recovery guarantees involve sampling and
geometric regularity assumptions. Those assumptions have not been established
here. Our sample graph therefore never becomes proof of continuous material
connectivity. [Topology-recovery assumptions](https://math.uchicago.edu/~shmuel/NSW1.pdf)

The hypothesis under test is narrower: overlapping, position-derived plane
seeds plus whole-component transition proposals can recover a useful conditional
radius candidate without caller-supplied region labels. The competing failure
explanations are local curved/flat ambiguity, neighborhood or graph-scale
instability, unrelated surfaces entering a component, incomplete fixed-plane
extension, and uncertainty too large for the requested measurement resolution.

## Exact finite search policy

1. Validate at most **256 raw input rows before conversion or deduplication**.
   Lexicographically sort unique positions. Exact repetitions do not create new
   support; near repetitions remain distinct and can affect a density-sensitive
   fit. This is not a scan downsampler or a method for unlimited clouds.
2. At each unique sample and each of one to four explicit radii, form the nominal
   observed Euclidean ball, including a disclosed floating-point guard. Fit its
   entire support using the existing position-only plane witness and normal
   cone. Require the caller's minimum support and angular resolution. Do not trim
   residuals. Deduplicate **identical support tuples only**; retain overlapping
   and nested alternatives. At most 64 accepted seed regions are allowed.
3. Build a graph of pairwise-possible sampled links under the declared `sample_gap`
   and error allowance. This graph is not verified topology.
4. For each seed's **fixed** fitted plane, select compatible samples and take
   connected components in the induced compatible graph. A path cannot travel
   through incompatible samples. Retain each component intersecting any seed
   member and meeting the same minimum size; refit all its points and recheck the
   normal bound. A seed spanning disconnected sampled components is explicitly
   split; split/subminimum-piece counters are exported. No normal rotates while
   extending a region, and no merely similar supports are merged.
5. Enumerate every disjoint extended-plane pair. For each pair, remove its two
   supports from the graph and enumerate every remaining component with at least
   three samples and pairwise-possible links to both planes. Keep the **whole**
   component, including contaminants. Do not choose a shortest path or discard
   poor radial fits. Different pairs can legitimately produce overlapping
   competing associations.
6. Call the unchanged uncertainty-preserving plane/radius bridge on each three-way
   disjoint support. Retain both conditional and abstained diagnostics; no winner
   is selected. A top-level positive status requires at least one conditional
   radius hypothesis, not a verified bend. The physical bend count stays `None`.

Explicit budgets cap unordered pair evaluations at 2,016 and bridge evaluations
at 256. Entire association enumeration is budget-checked before any bridge call.
Exhaustion clears candidates and extended regions, marks search incomplete and
does not publish an arbitrary successful prefix. Seed-region exhaustion similarly
publishes no partial seed set. Numerical unavailability in seed, extension or
bridge analysis is incomplete search, not a silently absent region.

`search_complete_within_declared_policy` means this finite algorithm finished its
declared support/fit checks. It is **not** completeness over every plane, region,
compatible joint model, physical component or scanner-scale cloud. In particular,
TLS-plus-midstrip witness failure is not proof that no minimax plane exists.

## Conditional uncertainty and counterexamples

### Membership and sampled links

If every observation `p_i` has a counterpart `q_i` within `epsilon`, then for
any pair, triangle inequalities give

```text
abs(||q_i-q_j|| - ||p_i-p_j||) <= 2 epsilon.
```

Hence observed distance `d` gives the true-distance interval
`[max(0,d-2epsilon), d+2epsilon]` before numerical guards. A ball boundary within
`2epsilon` is sensitive to admissible perturbations. Seed diagnostics count such
samples on both sides of the observed boundary; selected membership is not
physically verified. The same uniform bound applies to the minimum distance
between two finite sets: take the minimum of all bounded pair distances.

The graph includes an edge when `d <= sample_gap + 2epsilon + numerical_guard`.
Association outputs disclose minimum-distance intervals and whether the upper
bound is within the declared gap. That latter fact concerns a **sample pair**
only, not surface contact or material connection.

Even all pairwise-possible links need not be jointly realizable by a single set
of perturbed positions. Take an equilateral triangle of side 1.2, allowed error
0.1 and link gap 1. Each edge can individually shrink to 1. But the convex
perimeter supporting-plane bound after permitted vertex perturbations is
`3(1.2)-3 sqrt(3)(0.1) > 3`; all three short edges cannot coexist. A persisted
regression checks that the graph exposes this nonverification explicitly.

### Plane compatibility is not physical flatness

For cylindrical samples `(R sin(t), y, R(1-cos(t)))`, `|t|<=phi`, a mid-height
plane has maximum residual `R(1-cos(phi))/2`. When this is at most `epsilon`, an
exact curved patch and compatible planar counterparts both explain the positions.
A tight normal cone conditional on planarity does not resolve that ambiguity.
Tests retain short cylindrical plane hypotheses without claiming verified flanges.
Another independent regression uses two exact cylindrical generator strips that
are also exactly planar even at zero geometric noise.

Pairwise plane compatibility is not transitive either: square-corner samples at
heights 0, 1.5epsilon and 3epsilon admit adjacent two-level planes within
0.75epsilon, while their full three-level union needs 1.5epsilon. Whole-support
checks and fixed-plane extension avoid treating compatibility chains as proof.

The fixed-plane extension itself can lose recall. A seed normal tilted by 0.001
radians excludes distant samples on the same true plane when their lever arm is
10 and the error band only 0.001. This limitation is tested and exported, not
hidden by enlarging the error bound until a candidate appears.

### Preserving complete candidate support

The independent 85-position radius-2 fixture finds two 25-position tangent flange
hypotheses and its 35-position curved component. Adding a connected radius-2.4
contaminant retains all 36 transition positions and forces that candidate to
abstain. A detached arc stays outside the original sampled component.

Conversely, removing an angular strip can leave graph links across the hole;
removing an axial strip can split one constructed bend into two sample candidates.
Tests explicitly preserve these ambiguities. Neither sample connectivity nor
candidate multiplicity establishes physical continuity or bend count.

All simultaneous-error implications remain conditional after data-dependent
subset selection, because the assumed bound applies to **every input sample**.
Estimating that bound from selected fit residuals would not justify this premise.

## Numerical and resource ledger

Distance construction uses scaled coordinates or stable hypotenuse calculations.
The seed module discloses a guard of
`128 machine_epsilon (max_abs_coordinate + max_pair_distance)`; graph uncertainty
uses that guard plus `2epsilon`. Existing plane and radius numerical allowances
remain separately visible. Large offsets may enlarge guards without adding
geometric information. Unsupported ranges and unexpected nonfinite payloads
clear the result and mark it incomplete. In particular, a finite graph threshold
does not excuse an overflowing `distance + uncertainty` output interval.

These are engineering guards, **not directed-rounded interval certificates**.
No hard latency or process-tree memory guarantee follows from the timings below.
No dependency was installed and no native geometry process was started. Parallel
source/review tasks used the host's capacity without increasing frozen budgets.

## Executed verification

Root's final combined eight-file geometry suite passed **458 tests in 1.59 seconds**
after integrating 126 new checks and the existing 332 regressions.
This includes independently authored synthetic support/geometry tests and
regressions for both math-review findings: unavailable analyses must not claim
complete search, and sampled-link upper bounds must remain finite.

The plane-only test includes 1,000 explicit rotation/translation/permutation/scale
cases, each 18 unlabelled samples and 54 seed queries, recovering the same two
known sample supports. This tests one separated-plane construction across
transforms, not arbitrary part families.

Root also executed the optional 1,000-case **end-to-end** sweep: 30 samples per
case, six orders of radius scale, proper rotations, translations, and verified
bounded point errors. Fixed scale-relative search/precision settings were used.
It produced **669 positive hypotheses, 331 abstentions, zero incomplete searches**.
In all 1,000 cases, the retained transition matched the constructed true support
and its conditional radius interval contained the true radius. Timing:
**1.98040 seconds**, peak process RSS **50,921,472 bytes** on Darwin, including
Python, NumPy and pytest. This is not a native pipeline/process-tree benchmark.

These variants share one geometric construction; 669/1,000 is **not accuracy,
physical coverage or calibrated confidence**. The absent detection baseline and
physical truth prevent a generalization or promotion claim.

An additional independent **120-case structural matrix** changes geometry, not
just pose/noise. Ten signed angles (`+/-30,60,90,120,150` degrees) and three radii
(`0.5,2,20`) are used in each stratum. Every case uses the same scale-relative
policy: ball radius `0.4R`, sampled gap `0.405R`, error `5e-7R`, axial gap
`0.255R`, radius-width limit `0.05R`, five axial levels, five flange stations,
and seven interior angular stations. No per-angle tuning occurred.

| Invented stratum | Conditional candidate cases | Abstained cases | Interpretation |
| --- | ---: | ---: | --- |
| Clean tangent flanges and one cylindrical face | 30 | 0 | All 30 recovered valid-model intervals contain the true radius |
| Remove the middle angular sample column | 18 | 12 | Larger 120/150-degree turns exceed the fixed gap; positives at smaller angles still do not prove material continuity |
| Mix in a concentric face at radius `1.1R` | 0 | 30 | All 30 whole components retained; every single-radius bridge abstains |
| Five flange stations spread over `2R` instead of `R` | 0 | 30 | The unchanged ball scale cannot find adequate planar seeds at this sampling density |

There were no incomplete searches or budget failures in this matrix. The clean
and applicable missing-column cases gave 48 matched-original-model containment
checks with zero violations. No single-radius containment is asserted for mixed
faces. An independent graph traversal also verifies that no candidate silently
loses points from its remainder component.

Root executed the matrix in **0.51192 seconds**, peak process RSS **48,316,416
bytes**. Its exact aggregate is now a regression test, not a physical score.
The sparse stratum preserves an initial policy/spacing mismatch as an explicit
negative test: spacing `0.5R` exceeds the `0.4R` ball. The clean angle comparison
keeps the original quarter-fixture spacing `0.25R` to isolate angle; the sparse
failure was not erased by changing its settings.

Reproduce focused integration from `FullProject/` with the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_position_plane_regions.py tests/test_position_bend_candidates.py tests/test_position_discovery_adversarial.py -q
```

The bounded expanded sweep is the explicit
`run_discovery_sweep(expanded=True)` function in the linked integration test;
normal pytest executes its 48-case version. No real data path is accepted.
The independent `run_structural_discovery_matrix()` in the adversarial test
returns the stratified receipt above and is also exercised by normal pytest.

Final new engine-source snapshot (SHA-256):

```text
position_plane_regions.py   ad0278d454cae6e17685e18ff52e0951152b62eb06805c1c5d5998f0b5fae78d
position_bend_candidates.py 8318853f4497459b3244d526df58b00b8a90762ece6413e526ee0be8e203dbc8
```

## Decision and next engine checkpoint

Continue this as a candidate-birth hypothesis: it now demonstrates position-only
end-to-end region association on independently specified invented geometry and
exposes concrete density and mixed-face limitations under fixed policies.
Do not promote it into the active detector or call it floor ready. The next
engineering/scientific gaps are mixed inner/outer surfaces and multiple adjacent
bends, stable discovery under density/coverage changes, joint physical relation
and tangent-contact evidence, capture-error calibration, finite admissible
localization, and realistic native resource behavior. A prospective comparison
still needs a frozen baseline/evaluator and separately authorized physical inputs.

No physical scans, reserved email parts, held-outs, raw datasets, pools/manifests,
saved scientific output or control/issuance/one-shot artifacts were accessed.
C15's three source-freeze failures and EXP35 terminal disposition are unchanged.
Existing unrelated worktree edits remain preserved.
