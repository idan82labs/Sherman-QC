# V9 angular-support policy delta: independent source/math audit

Date: 2026-09-13. Reviewer: Astra/Codex subagent
`v8_flange_policy_math_audit`, independent of the Fable implementation owner.

Scope: the newly declared `support.max_angular_gap_deg` 5 -> 30 change,
`backend/cad_guided_instances.py`, relevant V9 declarations and the connected
synthetic fixture. Source reads and scalar arithmetic only. No engine imports,
test execution, geometry jobs, physical inputs, Gold, attempt artifacts or
private caches were accessed. Earlier reports are unchanged. This report is
neither a whole-candidate acceptance nor a freeze/PRE/launch authorization.

## Outcome

The 30-degree policy is a defensible **prospective sample-support engineering
choice**, not a consequence forced by a 0.3 mm physical sampling calibration.
For fixed fitted geometry and fixed support rows it admits every support
already admitted at 5 degrees and may admit additional, more sparsely sampled
supports. It does not manufacture a measured radius or change the contact
scorer. It also does not guarantee a prediction, improved F1, continuous arc
coverage, material-side identity or a manufacturing-event count.

Two corrections are needed before freezing these snapshots:

1. **P2 — Delta inventory is inconsistent.** The runner declares 5 -> 30, but
   the protocol says three engine changes and omits it; the design says every
   other CONFIG value is unchanged. Include angular-support admission explicitly
   in the change inventory and combined-effect interpretation.
2. **P2 — The causal justification overstates the sampling evidence.** A
   transverse chord of 0.3 mm on radius 0.8 mm does subtend about 21.6 degrees.
   Generic 3D row spacing does not establish that chord. The connected fixture
   actually creates 22.5-degree nominal angular steps, then adds noise.
   Neither radius alone nor an unspecified 0.3 mm spacing proves that 5 degrees
   rejects every carrier. “Margin for phase” is not a phase-robustness proof.

These are bounded declaration/evidence corrections, not a demand to solve
continuous physical coverage before the already-generic conditional contact
development score. Existing uncertainty/coverage qualifiers should remain.

## Exact reviewed snapshots

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_instances.py` | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
| `scripts/run_gold11_mesh_development_v9.py` | `999e4bbc9cc50bdd0954977e8e66d8c09bef9aa6dd2203b111500032636bebe4` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md` | `424e0d2d796fc5d4a7a3858bad56298bc3dee71993a407ecd9315245b395dd46` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md` | `1f4d52492a904703bf84302c8c5d6bc53cfb22157b0a87d9dd6272caeb574522` |
| `tests/test_cad_guided_v9_connected_chain_v1.py` | `90ead166aa2799b9be58ae1595551282b5aa81c63bff2d81907da3655dbf451c` |
| `backend/cad_guided_measurement_v2.py` (arc-filter excerpt only) | `cb63853d45d71a54aae337d4e59c0092d1bd0e8fc939625b0917c04356676328` |
| `docs/research/2026-09-13_fable51_v9_diagnosis.md` (Part E context only) | `2d01be35ab89074d3b023d91a86aa81c7eb489bcae3877bbd9f74b1cb0962ece` |

Line references below are to these snapshots. Concurrent later author edits
require their own delta reconciliation; a hash here does not endorse an
unfinished consumer chain or assert that its tests passed.

## 1. What the support policy actually measures

`backend/cad_guided_instances.py:93` takes the scan-measured axis/center and
`:97` its measured radius. The adapter checks radial consistency with the
declared fit residual at `:110`, then computes cross-section angles about the
measured axis at `:113`. No nominal CAD radius or nominal sweep enters this
adapter (`:1`).

At `:121` the global largest circular gap determines the unwrap origin. The
retained sample arc must span at most 180 degrees (`:128`). This removes the
complementary gap from the represented interval; it does not assert coverage
of unsampled angular endcaps or the full nominal CAD sweep.

For maximum axial gap A, the adapter checks consecutive distinct axial
coordinates (`:130`), partitions the measured axial interval into
`ceil(length/A)` equal slabs (`:137`), and uses closed boundaries with a
numerical guard (`:145`). A boundary station may support both adjacent slabs
by deliberate design. Within **every slab**, it requires:

- At least m unique positions for each of the arc and two flange roles
  (`:87`, `:152`); V9 sets m = 3.
- At least m distinct floating-point angular values (`:155`).
- Sample angular span S at least M, modulo the numerical guard (`:158`);
  V9 sets M = 20 degrees.
- Largest consecutive gap g between those sorted angular values at most G,
  again modulo that guard (`:159`); V9 changes G from 5 to 30 degrees.

The last condition is not an edge-length, nearest-neighbor, surface-area or
per-cross-section triangulation condition. Different angular samples inside
one slab may lie at different axial positions. The unchanged A = 4 mm does
not require each individual axial station to contain three angles.

If k local angles have gaps g_i, then S = sum(g_i). Ignoring only the explicit
roundoff guard, necessary conditions are

    k >= m,  S >= M,  max(g_i) <= G,
    hence k >= max(m, ceil(M/G) + 1).

With M = 20 and m = 3, G = 5 implicitly requires at least five local angular
values, whereas G = 30 can permit only three. Even those three need not be
well distributed: angles 0, 0.01 and 20 degrees meet these angular conditions.
The separate fresh fit and flat-witness conditions still have to pass.

“Distinct angular stations” must be interpreted narrowly. `np.unique` at
`:118` and `:155` removes exact equality, not noise-scale angular clusters.
Many independently perturbed rows on two nominal angle columns may therefore
provide more than three distinct values. This is consistent with the source's
sample-supported contract, but not proof of three independently resolved
cross-section stations. It is particularly relevant to the fixture's per-row
noise at `tests/test_cad_guided_v9_connected_chain_v1.py:69`.

The adapter explicitly publishes `SAMPLE_SUPPORTED_CONDITIONAL`, sampled rather
than physical-trim endpoints, continuous coverage not established, and unknown
manufacturing identity (`backend/cad_guided_instances.py:167`). Its aggregate
per-slab counts, span and maximum gap (`:164`) support that restricted meaning.

## 2. Chord, arc length and axial neighbors are different quantities

For two noiseless points on a common radius-r circular cross-section,

    c = 2 r sin(Delta_theta/2),
    Delta_theta = 2 asin(c/(2r)),      0 <= c <= 2r.

Angles in these equations are radians; multiply by 180/pi for degrees. Thus
r = 0.8 mm and **cross-sectional chord** c = 0.3 mm give
Delta_theta = 21.61384575 degrees. This arithmetic in runner `:180` is correct
under that model.

An **arc-length** step l = 0.3 mm gives l/r = 21.48591732 degrees instead.
For two points on a cylinder with axial separation Delta_z, their 3D distance
d obeys

    d^2 = (2 r sin(Delta_theta/2))^2 + Delta_z^2.

Consequently d = 0.3 mm can mean Delta_theta = 0 with Delta_z = 0.3 mm, or a
range of angular/axial combinations. Nearest-neighbor spacing or row spacing
without directional conditioning cannot be substituted for c. No physical
row-spacing evidence was inspected in this source-only audit.

A fixed 30-degree limit also has a radius-dependent physical scale:

    maximum ideal chord = 2 r sin(15 degrees),
    corresponding arc length = r*pi/6.

At r = 0.8 mm these are 0.41411047 mm and 0.41887902 mm. The same angular
policy allows proportionally larger transverse gaps on larger measured radii.
That is an explicit angular sampling policy, not a constant-mm scan-density
requirement. The support implementation uses measured angles, so radius or
pose errors can alter its actual angular gaps; the nominal calculation does
not bind those errors.

## 3. Exact connected-fixture arithmetic and phase limitation

The source fixture is not a constant-chord sampler. At
`tests/test_cad_guided_v9_connected_chain_v1.py:54`, it sets

    n = max(3, round(r * abs(phi) / spacing) + 1)

and samples `linspace(0, sweep, n)` at `:55`. For r = 0.8 mm, phi = pi/2 and
spacing = 0.3 mm, n = 5: nominal angles 0, 22.5, 45, 67.5, 90 degrees.
The actual nominal arc step is 0.31415927 mm and chord is 0.31214452 mm.
There are also axial steps of 0.3 mm (`:50`), then independent coordinate
noise (`:69`). The connected support policy explicitly uses G = 30 (`:79`).

For ideal fitted tangent planes, the V2 arc exclusion (`:624` through `:646`)
requires r(1-cos(theta)) > e at each end. With r = 0.8 mm and e = 0.04 mm,

    alpha = acos(1-e/r) = 18.19487234 degrees,
    theta in (alpha, 90-alpha),
    continuous usable sweep = 53.61025532 degrees.

The 8-degree nominal guard is weaker here. The noiseless zero-phase fixture
therefore retains angles 22.5, 45 and 67.5: 45-degree sampled span and
22.5-degree gaps. Provided all other conditions pass, 5 and 15 refuse this
support while 25 and 30 admit it. The runner's reported 5/15 versus 25–45
synthetic outcomes are author-reported development observations, not executed
or independently reproduced by this review.

The finite three-column support has a much smaller flat-witness margin than
the continuous envelope:

    finite strip half-width = r*(1-cos(22.5 degrees))/2
                            = 0.030448187 mm,
    headroom above delta=0.03 = 0.000448187 mm.

This is exact clean-model arithmetic, not a prediction of noisy fit success.
Changing G cannot improve this strip width. Noise, fitted-plane errors and
finite sample placement can change the rows admitted by the earlier filter.

A half-step shift of the same angular lattice puts columns at 11.25, 33.75,
56.25 and 78.75 degrees inside the nominal quadrant. Only 33.75 and 56.25
survive the ideal exclusion interval: two exact angle columns. Raising G
cannot recover the lost third column or cure a rank/flat-witness refusal.
Per-row noise may create more than two distinct numeric angles, but that is
not a phase-robust three-column guarantee.

Uniform lattice phase does not change the interior angular step. A 30-degree
limit has 7.5 degrees of headroom over this fixture's 22.5-degree step; that
can accommodate some **gap perturbation**, not arbitrary phase or missing
columns. If retained ordered angles each have a declared error bound eta,
then a nominal gap h can grow to h+2*eta; h = 22.5 and G = 30 require
eta <= 3.75 degrees for that sufficient bound. No such physical error bound
is established here. Removing one interior column instead creates a
45-degree gap, which still fails G = 30.

## 4. Monotonicity and sparse/missing-arc interpretation

The valid G domain is [1e-8, 90] degrees, independently of M in [1e-8, 160]
and m in [3, 512] (`backend/cad_guided_instances.py:27`). Both 5 and 30 are
valid. With all other arguments fixed, G appears only in validation and the
final local-gap inequality (`:162`). Increasing G within its valid domain
cannot turn a supported instance into an unavailable one. Other invalidity,
numeric-resolution, role, winding, axial and local-count checks are unchanged.

This is **not** monotonicity of the full V9 pipeline or F1. V9 simultaneously
changes residual/exclusion policies and selection/initialization mechanisms.
New supports can interact with grouping/cover selection, and additional
predictions may be true or false positives. No score improvement follows
from this support-local monotonicity alone.

At G = 30 an unobserved interval up to that declared consecutive-angle gap
may be bridged by the sample-supported representation. Example: starting
with angles 0, 15, 30, 45, 60 and removing 30 leaves a 30-degree gap that is
admissible. Removing both 15 and 30 leaves a 45-degree gap that is not.
Neither accepted gap proves continuous material between the samples. Missing
angular endcaps are not measured against a CAD angle by this adapter; an
observed sub-arc satisfying the declared M and G can be accepted. These are
facts about the existing generic contract, not grounds to invent a new
continuous-coverage prerequisite for this development score.

## 5. Minimal corrections and decisive synthetic checks

Amend runner `:180`–`:184` to distinguish nominal radius, an assumed
cross-sectional spacing model, and the fixture actually constructed. Replace
“every carrier on this geometry regardless of the fit” with the bounded
uniform-column fixture statement. Describe 30 as a chosen development margin
above its 22.5-degree nominal gaps, with phase/occlusion/count limitations.
It is one plausible value, not an analytically unique optimum or a calibrated
physical threshold. Preserve the consumed-development provenance.

Add the exact 5 -> 30 delta to protocol `:13` and `:57`, and design `:30`–`:37`
and `:61`, so “unchanged” and combined-effect claims match the runner. Freeze
the revised source and declarations together; do not alter the started or
terminal V8 policy. The unchanged generic scoring rules may remain unchanged.

Only a few isolated synthetic checks are needed to pin down this delta:

1. **Threshold and monotonicity:** same fresh scan fit and per-slab support,
   angle grids 0/5/10/15/20 and 0/22.5/45, varying G only. Check equality and
   just-above-30 cases at differences well larger than numerical guard;
   every 5-supported fixture remains supported at 30.
2. **Phase and numeric uniqueness:** zero-phase versus half-step clean
   r=0.8/e=0.04 grids, then controlled per-row angular perturbations. Record
   exact unique-angle counts and the separate flat-witness outcome; do not
   reinterpret a noisy cluster as a physically resolved station.
3. **Missing interior samples:** the 15-degree grid deletions above, while
   holding axial/flange evidence fixed. Preserve the accepted conditional
   coverage qualifiers and the explicit >30-degree refusal.
4. **Local versus global support:** distribute enough global angles across
   several slabs but leave one slab below local angle/count requirements;
   vary only G and ensure that unrelated local refusal remains. Include
   axial-only 0.3 mm neighbors to demonstrate why generic row spacing is not
   an angular-spacing measurement.

These are proposals, not executed checks or a broad new research campaign.
The source-level acceptance check for this review is a consistent frozen
declaration plus correctly scoped sampling claims. The full connected
candidate, custody, actual test results and any physical PRE remain the
responsibility of their separate independent gates.
