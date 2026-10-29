# CAD-engine V1: scope and mathematical contracts

Date: 2026-09-09. Source/synthetic development under the active autonomous
mandate. Frozen R0 remains unchanged. This is not a physical benchmark receipt.

## Registration

The new whole-part mode explicitly assumes common mm without estimating scale.
It is not the stable-datum mode. Twenty-four proper PCA hypotheses receive
bounded point-to-point and trimmed/Huber point-to-plane refinement. There is no
global-convergence or physical-uniqueness claim.

For transformed point `p`, CAD tangent normal `n`, center `c` and characteristic
length `L`, the normalized Jacobian is `[(p-c) × n / L, n]`. Solved rotational
coordinates are `L*omega`, not a fitted scale. The rigid update is
`R_new=Exp(omega)R`, `t_new=Exp(omega)(t-c)+c+delta_t`. Exact Rodrigues rotation
and bounded backtracking avoid applying a nonorthogonal linearized matrix.
[Low's derivation](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf)
supports the local approximation and conditioning treatment; it is not evidence
that these particular search starts solve every partial scan.

Huber IRLS uses `sqrt(min(1, delta/abs(residual)))` on both Jacobian and target.
Trimming and robust weights are optimizer policy, not manufacturing tolerances.
Every generated pose reports complete supplied-cloud forward/backward distance
summaries and coverage, separately from the actual trimmed optimization subset.
Both unweighted and Huber-weighted subset singular values are checked. Weak
normal families cannot establish observability merely by surviving the distance
gate. [Fitzgibbon's robust-registration work](https://www.robots.ox.ac.uk/~vgg/publications/2001/Fitzgibbon01c/fitzgibbon01c.pdf)
motivates robust losses; the implementation is not a reproduction of his LM
algorithm. Geometric-conditioning diagnostics also follow the concern raised by
[Gelfand et al.](https://pixl.cs.princeton.edu/pubs/Gelfand_2003_GSS/stabicp.pdf):
apparent numerical constraint is not independent physical truth.

The objective is unchanged by reversing tangent normals. Therefore it cannot
establish scan inner/outer side, especially for thin shells. Best fit also absorbs
some CAD-relative position/orientation deformation. A rigid transform itself
preserves circle radius and interplane angle, but changed correspondence/ROI
membership can bias their estimates. These distinctions are explicit outputs.

## Full raw support and finite windows

The mesh adapter validates every admitted scalar and face list and retains all
original XYZ rows. Dyadic voxel sampling is registration/proposal preparation,
not replacement metrology. All raw rows map to retained representatives and have
a measured displacement receipt. Distances use scale-safe `hypot`: independent
review found that squared tiny distances could falsely become zero. No continuous
surface, feature-preservation, or physical calibration certificate is implied.

Native transition ROIs use complete boundary enclosures and an inside witness.
Generator contact intervals remain unions, not bounding spans. Entire cylinder
patches and root-to-search flange strips must exclude all non-contact boundaries,
including holes. A window's ends are interior measurement bounds, not physical
trim endpoints. Native-certified periodic seams and complete periodic cavities
receive explicit treatment; unresolved plausible transitions remain visible.

The local wrapper crops conservatively from full raw XYZ without thinning and
rejects oversized local supports. It maps every fitted support row back to the
original byte-bound row and checks curved-support exclusivity across all native
windows, including across the former 16-feature call boundary.

Its fixed proposed display/scoring carrier is the measured inner-arc mid-angle
generator: fitted circle center plus fitted radius times the half-sweep radial
direction derived from a measured flange. It is not a CAD center axis or a
nominal-radius copy. It is never selected against Gold. Historical Gold's carrier
is still unresolved; this proposal does not manufacture convention compatibility.
Window counts and sample-supported segments remain distinct from the required
stage-appropriate observed-bend count, which is still unavailable. Future grouping
must use scan-supported geometry/ownership, not CAD totals or window totals; the
target is not a reconstruction of every manufacturing event.

Local trim uncertainty is conservative exclusion, not a relaxed analytic proof:
selected cylinder/flange/contact geometry retains the strict2e-6mm gate. Larger
edge tolerances enlarge complete edge bounds; incident vertices contribute their
own located point-plus-tolerance boxes. Those envelopes are expanded before
transverse rejection or axial exclusion. Unboundable boundaries refuse their
affected candidate without erasing independent valid transitions. This explicitly
accounts for [OCCT AddOptimal's separate treatment of vertices not already in edges](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_0/src/BRepBndLib/BRepBndLib.cxx).

## Approximate CAD sampling and numerical fixes

Native triangulations supply registration samples only, not measured answers or
certified analytic-surface coverage. Oriented triangle areas allocate deterministic
interior samples. Base2/base3 radical inverses now reverse bounded integer digits
exactly before one float64 division. Triangle interpolation uses
`p0 + (w1*(p1-p0) + w2*(p2-p0))`, adding the anchor last; constant coordinates
are preserved exactly. Exact-rational checks bound arithmetic roundoff, separately
from native tessellation error and physical measurement uncertainty.

An actual four-shape native check exposed rejection of valid cached triangulations.
The narrow fix admits only status0 or128, the informational `IMeshData_Reused`
bit in [OCCT's status definition](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_0/src/IMeshData/IMeshData_Status.hxx).
All error/remesh/outdated/unknown bits still refuse; IsDone and every face,
triangle, winding and location check remain required. Reversed/moved boxes share
their original TShape cache; this checks cache reuse and location/orientation,
not four independently uncached meshes or general coverage certification.

## Actual checks so far

- Mesh final source: 138 focused tests pass, including one-million-vertex full
  support and tiny-distance regressions. Worker CPU .681761 s, peak418,627,584 B;
  parent .708748 s, no budget stop. Independent source reviewer GREEN after repair.
- Whole-part registration: 47 independently authored tests pass in1.66 s;
  worker CPU1.530499 s, sampled peak75,644,928 B, parent1.761217 s, no budget stop.
  Exact/partial oblique poses, noise, Jacobian, proper rigidity, symmetry,
  degeneracy, weighting and complete-supplied-cloud diagnostics are covered.
- Full-raw local measurement73 and connected workflow53 tests pass. These include
  measured+1mm/+5degree constructed deviations and all four endpoint inverse
  transforms; workflow import/pose boundaries are mocked in the glue tests.
- Native finite16-case matrix and nine adversarial local-uncertainty controls
  pass after retained frame-oracle and tolerance-exclusion failures. The analytic
  gate and original ellipse remain unchanged; parent1.253266/1.180316s,
  peak387055616/368115712B, no budget stops, exact source consistency.
- CAD sampler163 focused tests and four constructed native cases pass after
  retained numerical/cache failures. Latest native parent .4670945s,
  sampled peak324,239,360B, no budget stop. The cylinder's independently checked
  analytic surface distance maximum is .00870156mm under this invented model.
- Post-inference scorer47 tests pass using the unchanged legacy geometric
  matching/aggregate code. This tests custody and failure semantics, not a
  physical score. Parent .722730s, peak85,442,560B, no budget stop.
  Per-case P/R/F1 and distinct emitted-count equality were then added using the
  same singleton scorer;49 focused tests pass, parent.742715s, peak85999616B.
- Actual STEP-to-scan connected matrix is1/3: both nominal poses are accurate to
  0.00104838mm but fail finite-sample coverage,91.2064% versus95%. Retained
  p95 distance0.549465mm minus that uniform pose error is still above0.5mm.
  The1-Lipschitz distance-to-set bound proves the8192-sample gate is impossible
  even at the true pose for this fixture. A separately frozen32768-sample
  investigation preserves all error ceilings. Missing-arc abstention is retained;
  its inner-to-outer shifted pose is not a valid acquisition-side solution.
  The32768 successor also1/3: its nominal best candidate has fit-subset rank0
  and plane p950.429645mm. Denser samples changed the optimization basin/trim
  behavior, so the sparse gate diagnosis did not imply successful recovery.
  This negative is retained without relaxing criteria or reusing the attempt.

These are mechanism contracts on constructed inputs, not Gold F1 or production
qualification. The first physical V1 diagnostic has since completed with an
independent post-run audit: both arms emit zero lines and retain all 71 false
negatives; P/F1 are undefined, with no gain or promotion. The separate constructed
stage trace passes six diagnostic controls and retains cap-row loss from trimmed
optimizer subsets. See the [V2 mechanism checkpoint](2026-09-09_surface_registration_v2_mechanism_checkpoint.md)
for the evidence distinction and finite-surface successor. Every new physical
evaluation still requires its own frozen source/config/data/reference/scorer
bindings, budgets and independent pre-run audit.
