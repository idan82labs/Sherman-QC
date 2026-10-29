# Dense all-row radius constraints and roadmap position

Date: 2026-09-07. Source and invented arrays only. This continues the
[bounded window-ingestion checkpoint](2026-09-07_streamed_window_checkpoint.md)
under the [locked roadmap](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07).
No physical, held-out, email, control or one-shot artifacts are accessed.

Follow-on: the [component-aware CAD-root grouping checkpoint](2026-09-07_cad_root_grouping_checkpoint.md)
now records a fixed paired structural improvement at the next primary-product
seam. The dense mechanism and evidence below remain unchanged; neither checkpoint
constitutes native runtime or physical validation.

## Position against the locked plan

| Milestone | Evidence-backed status, not percentage complete |
| --- | --- |
| M0 | Bounded source checkpoint delivered; C15's three freeze failures remain unwaived. EXP35 remains terminal. |
| M1 | Decision semantics largely implemented at source/synthetic boundaries; full native runtime/worker verification remains open. |
| M2 | Capture/reference/calibration and repeat-capture validation remain incomplete. |
| M3 | Substantial local geometry development; dense measurement preparation advances here. Primary component-aware CAD mapping and physical/grouped-evaluation exits remain open. |
| M4 | Ownership/artifact safety and declaration gates improved; durable recovery/cancellation, immutable execution and trusted evidence verification remain incomplete. |
| M5 / M6 | Independent prospective physical validation and shadow floor pilot have not begun. |

Priority 1's named noise/density development benchmarks improved materially.
Priority 2 has adjacent/partial/mixed controls, same-face relation diagnostics and
bounded ingestion, but not automatic full-density bend discovery. Priority 3's
component-aware CAD-assisted mapping is the next primary-product deliverable.
Synthetic test counts do not close M2, M3's physical evidence exit, M5 or M6.

Decision for this bounded step: separate a stipulated geometric proposal from
all-row measurement constraints. Do not make universal scan-only discovery a
prerequisite for the CAD-assisted product, or keep extending simulations while
capture/calibration and recoverable operation remain untouched. Reuse this seam
only in its explicit scope; then proceed to expected-event mapping with independent
observed evidence. M2 capture/reference preparation and bounded M4 recovery work
remain parallel priorities, not work completed by this geometry checkpoint.

## What the new mechanism does and does not do

The new [dense radius constraint module](../../research/geometry/dense_axis_tube_radius.py)
consumes up to two million supplied raw rows for ONE stipulated cylinder-support
hypothesis. It freezes a nominal axis line, a direction cone, an axis-point ball,
an original per-point measurement error and a radius prior before reading rows.
Every row contributes; duplicates consume budget. There is no downsampling,
residual trimming, residual-based support assignment, fitting or compression.

This bypasses neither the local engine's 256-raw-row cap nor the prior window
dispatcher's overflow rule. It is a separate O(N) measurement-constraint seam,
NOT an automatic dense-region proposal, whole-scan bend detector, finite CAD
localization or original-joint-model feasibility solver. The caller must justify
the support and axis family separately. In particular, a fitted circle center's
uncertainty is NOT equal to its residual or scanner epsilon. No automatic adapter
from the existing relaxed circle fit is added.

The output distinguishes an empty necessary intersection (only the declared
family/support/prior is excluded), a nonempty necessary intersection (no shared
model existence claim), and invalid/incomplete/empty input (abstention).
Neither result accepts or rejects a part or counts a physical bend.

## Derivation and primary research grounding

For nominal unit direction a0 and nominal axis point o, let v_i=p_i-o and
d_i=||v_i cross a0||. Assume a hypothetical true line contains t with
||t-o||<=b, has unoriented direction angle <=alpha<pi/2 from a0, and each true
surface point x_i satisfies ||x_i-p_i||<=epsilon. The same line and radius R
must apply to all points.

The standard cylinder residual is radial distance to the axis minus radius;
[Shakarji's NIST fitting paper, section 3.6](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955)
uses this geometry in an orthogonal-distance least-squares objective. We do not
use least-squares residuals to discard points or calibrate uncertainty.

Write P_a=I-aa^T. For rank-one directions,
||P_a-P_a0||_2=sin(angle(a,a0)); the general projection-norm/principal-angle
identity is given in
[Edelman, Arias and Smith, section 4.3](https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf#page=35).
Using this identity and the reverse triangle inequality gives our bound:

```text
|R - d_i|
 <= ||P_a[(x_i-p_i) - (t-o)]|| + ||(P_a-P_a0)(p_i-o)||
 <= epsilon + b + sin(alpha)*||p_i-o|| = E_i.

R belongs to radius_prior INTERSECT every [max(0,d_i-E_i), d_i+E_i].
```

The tilt term uses the full 3-D lever arm, not only radial distance. There is no
sqrt(N) reduction: measurement error is a stipulated simultaneous bound, not
independent random noise. The cited work grounds the distance and projector
identities; it does not validate this implementation or a physical QC use.

With b=alpha=0, the unpadded interval is the exact fixed-line sample feasibility
condition: each point can move radially onto the same infinite cylinder within
epsilon. With uncertain line parameters, the row constraints are only necessary:
different rows may require different members of the declared family. The code
conservatively makes no shared-feasibility claim even in the fixed-line case.

One counterexample uses nominal Z direction, b=1, alpha=epsilon=0, prior R=1,
and observed points (+2,0,0),(-2,0,0). Each point's interval contains 1, but their
distance is 4, exceeding any radius-1 circle's diameter. Another takes b=epsilon=0,
points a0 and 2*a0 and alpha>0: their distances to a shared tilted line through
o=0 are sin(theta) and 2*sin(theta), equal only at zero radius. Independent
positive per-row possibilities do not establish a shared cylinder.

A generator line of samples can fit both a plane and a cylinder, however dense.
The check also ignores finite axial/angular face limits and tangency. For a fixed
line and a finite axial interval H, even the full-circle lateral-surface distance
would be sqrt((d-R)^2 + dist(z,H)^2), not independent radial/axial epsilon tests.
These missing constraints must not be relabeled physical coverage.

## Numerical, streaming and resource contract

Distances use cross products and nested hypot, avoiding subtraction of nearly
equal squared norms near the axis. A row-local engineering allowance is added:

```text
g_i = 512*machine_epsilon*(max_abs(p_i)+max_abs(o)+||v_i||+d_i+E_i).
```

It is reported separately from the unchanged measurement epsilon. It covers
ordinary arithmetic scale effects conservatively, not a directed-rounded proof.
The numerical guard depends on each row, not chunk maxima, so rechunking does not
change constraints. Unsupported finite ranges abstain rather than use infinity
to rescue a model. The normalized nominal direction actually used is recorded.
Before any execution, independent review identified that subnormal coordinates
could make a relative allowance vanish while hypot still rounded the distance.
The implementation now abstains whenever nonzero arithmetic scale yields a zero
or subnormal allowance. This conservative range restriction is not an epsilon
change; the exact all-zero arithmetic case needs no padding.

Input budgets are explicit: 2,000,000 rows, 65,536 rows per materialized NumPy
chunk and 4,096 chunks, including empty chunks. Output is constant-size. Memory
is O(chunk_rows), plus fixed receipts; no full scan, distance matrix or point
archive is accumulated. Only extrema and their original stream-row IDs are kept.
Every row is audited, but this is NOT durable raw-data retention or a complete
provenance archive. A future product path must keep the authorized measurement
artifact and bind replay to it separately.

The iterator must reach normal EOF even after an empty geometric intersection.
A later invalid row, interrupted producer or exhausted budget clears all geometry
constraints. Counts cover fully validated and constrained chunks and exclude a
rejected chunk. Source arrays/proposal arrays cannot mutate frozen parameters or
earlier extrema. Producer allocation/blocking before yield remains outside this
cooperative consumer's enforcement; parser isolation and cancellation are open.

## Fixed acceptance challenge, before execution

Two million distinct dense cylinder-support rows are streamed in nominal 8,192-row
chunks. All rows are measured; there is no off-window background. The invented
radius is 2, angular support is 25 to 110 degrees and axial extent is -5 to 5.
Deterministic radial error <=0.5 epsilon and axial error <=0.25 epsilon give a
combined displacement <=sqrt(5)/4 epsilon, with epsilon=1e-6. No RNG, physical
input or scan-sized array is used.
The angular and axial coordinates vary together: the distinct points densely
sample a one-dimensional partial helical trace on the cylinder, not a 2-D areal
coverage grid. Review identified this before execution; the fixed fixture is
retained for all-row arithmetic/resource testing, not surface-coverage evidence.

Acceptance: retain radius 2, count every raw row and reach EOF within fixed
120-second / 512-MiB soft limits. A second two-million-row stream replaces only
the final radius with 2.01; the entire declared fixed-line model must be excluded
although its first 256 rows remain compatible. A separate late-NaN case has
1,999,999 valid rows, including a contradiction in a completed chunk, then one
invalid row; it must abstain rather than return the earlier exclusion. These
limits/fixtures are fixed before the first run; no failure-driven reduction.

Independent unit controls cover nonzero axis-ball/cone errors, true-radius
containment, transformations, long lever arms, non-feasibility counterexamples,
chunk identity, row budgets, invalid input and evidence scope. No older suite is
represented as rerun. Final measured verification is recorded below after root
review and execution.

## Verification and decision

Root fully reviewed the two new test files before the first and only execution.
Independent agents authored the tests and reviewed the derivation, scope and
subnormal fix. **87 checks passed (84 unit/contract + 3 scale checks) in 0.34 s**.
External wall time was 0.41 s; maximum process RSS was 52,887,552 bytes (50.4 MiB).
There was one geometry process and no geometry child process. The exact command:

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_dense_axis_tube_radius.py tests/test_dense_axis_tube_scale.py -q -s
```

| Fixed scale check | Fully constrained raw rows | Result | Analysis + verification time |
| --- | ---: | --- | ---: |
| Dense partial trace | 2,000,000 | Necessary interval contains R=2 | 0.0783285 s |
| Only final radius changed to 2.01 | 2,000,000 | Declared fixed-axis model excluded | 0.0777077 s |
| Contradiction then separate invalid chunk | 1,999,999 | Abstained; interval and extrema cleared | 0.0780712 s |

The first interval is `[1.9999994999993373, 2.000000500000663]` mm. Its extremal
row IDs are 942,054 and 925,973. The late-contradiction lower bound is
2.00999899999859 at row 1,999,999, exceeding the retained upper bound
2.000000500000663. The prior `[1.99,2.02]` admits both individual radii, so this
exclusion comes from their incompatible shared fixed-axis radius, not a prior
that forbids 2.01. The first-256 prefix contains 2 and remains unexcluded.
The tiny interval reflects an invented exact axis and stipulated epsilon; it is
not evidence of nanometer-scale physical measurement accuracy.

Each case processes 245 valid chunks. The invalid case receives one additional
NaN chunk within the total-row cap and does not reach EOF. The valid-prefix
maximum radial distance confirms that the contradiction was actually constrained
before that invalid chunk; no geometric conclusion survives the failure.
Unit tests additionally retain true radius under nonzero point-ball, direction-
cone and simultaneous measurement errors, plus a long axial lever-arm example.
Wrong-axis exclusion coexists with a valid alternative axis. A proven nonempty
but jointly infeasible example remains necessary-only, as intended.

Resource measurements include lazy fixture generation and each test's assertions,
but end before receipt JSON formatting/printing. Sampled process-lifetime RSS
high-water marks were 52,740,096 / 52,772,864 / 52,772,864 bytes, respectively;
these are not per-call deltas or the final process peak. All fixed 120 s / 512 MiB
soft targets pass. This measures 8,192-row chunks and one declared model, not
worst-case chunk size, many-model search, real parsing, cancellation, physical
scanner behavior or complete inspection latency. No repeated benchmark
statistics, new physical families or calibrated uncertainty are established.

No geometry or test was changed after this passing execution. The subnormal fix
was made during independent source review before any run, and its counterexample
is included. Existing local engine and ingestion sources remain unchanged; their
older suites were not rerun. Source identities and static verification are
recorded with this checkpoint below.

```text
research/geometry/dense_axis_tube_radius.py
f221e88e36a3ba5e591d513b8e0e31e3e7b51453a83e3a233a940fb4a3b70f7e
tests/test_dense_axis_tube_radius.py
a3bd74df8dffe2d743820345a939affdad6c7c36f2506eca8584f96b7c5a6ebd
tests/test_dense_axis_tube_scale.py
d2100b94be7f465c1aa3bc6d2a4db435780c0d41b6f626f499562cd48c84f40d
```

SHA-256 comparisons also retain the prior identities for the window dispatcher,
its two test files, the axis/plane-conditioned radius modules and the same-face
module/wrapper. AST parsing of the three new Python files, explicit-file
whitespace/final-newline checks and the new checkpoint's local links pass.

Reuse decision: keep this as a research-only all-row necessary-constraint guard
once support and an axis family have independent justification. Do not wire a
fitted center into it with an invented b, turn nonempty bounds into a formed-bend
claim, or silently discard unsupported rows. Automatic proposal/support discovery
and durable raw evidence retention are not delivered. The next primary-product
step is component-aware expected-event mapping and a finite observed-evidence
contract, with registration uncertainty kept separate. M2 capture/reference and
M4 recovery preparation remain parallel work. This is a bounded reuse decision,
not a claim that Priority 2, M3 or floor readiness is complete.

No runtime promotion, calibration, physical gate, service or UI change is included.
The locked acceptance criteria, C15, terminal EXP35, M5/M6 and paused goal remain
unchanged.
