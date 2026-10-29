# Analytical finite-bend fixture: geometry, observability and falsification

Date: 2026-09-07. Source-free engineering checkpoint for preparation under
[M2/M3 of the roadmap](2026-09-07_production_generalization_roadmap.md), not a new
detector, evaluator, physical experiment or generalization result.

## Delivered boundary

[The fixture](../../research/geometry/analytic_bend_fixture.py) generates one
stipulated cylindrical bend component and two tangent flanges in memory.
[The focused tests](../../tests/test_analytic_bend_fixture.py) verify its geometry
with independent analytical oracles and metamorphic relations. Default size is
66 sample records; the hard cap is 4,096. No mesh, point-cloud file, scanner model,
fitting, candidate selection or product decision is produced.

The useful innovation here is a falsifiable distinction between an **axis
hypothesis** and **identifiable finite transition support**. The tests demonstrate
why plausible flange geometry alone does not supply the missing observation.
They do not demonstrate that any production or research detector handles it.

## One explicit idealization

Let `c` be the centre-axis origin, `a` a directed unit axis, `n0` a perpendicular
unit start-radial direction, and `k0 = a × n0`. All dimensional quantities use one
declared unit. The default frame is `a=(0,1,0)`, `n0=(0,0,1)`, `k0=(1,0,0)`.
Let inner radius `R>0`, thickness `t>0`, signed turn `0<|theta|<=pi`, axial
coordinate `u` in `[u0,u1]` with `u0<u1`, and positive flange lengths `L0,L1`.

For a radial level `r` equal to `R` or `R+t`, define

```text
n(phi) = cos(phi) n0 + sin(phi) k0
k(phi) = a × n(phi)
sigma  = sign(theta)

bend:   B(u,f,r) = c + u a + r n(theta f),                 0 <= f <= 1
start:  F0(u,v,r) = c + u a + r n0 - sigma v k0,           0 <= v <= L0
end:    F1(u,v,r) = c + u a + r n(theta) + sigma v k(theta), 0 <= v <= L1
```

The cylinder parameterization follows the official Open CASCADE definition;
here we restrict its otherwise infinite axial direction and angular domain,
and explicitly add the flanges. No Open CASCADE dependency is imported.
[Open CASCADE, Geom_CylindricalSurface, detailed description and EvalD0](https://occt3d.com/dev/doc/refman/html/class_geom___cylindrical_surface.html).

The following properties are derived for this model, not empirical findings:

- Outward face normals are `+n(phi)` on the outer face and `-n(phi)` on the inner
  face; the flange normals use their contact radial direction. A negative turn
  reverses the parameter orientation, not the physical meaning of outward.
- At each join the positions, normals and traversal tangents agree. The joins
  are tangent-continuous (G1, or C1 after arc-length reparameterization), not C2:
  cross-sectional curvature jumps between zero and `1/r`. The two normalized
  sampling parameters need not have equal derivative magnitudes at a join.
- On each cylindrical face, `ds^2 = du^2 + r^2 dphi^2`. The coordinate change
  `w=r phi` unrolls that face isometrically; the two principal curvature
  magnitudes are `0` and `1/r`. Inner and outer arc lengths are `R|theta|` and
  `(R+t)|theta|`, differing by `t|theta|`.
- The model stipulates constant radial thickness and straight, parallel axial
  generators. It does **not** locate a neutral material layer, provide a
  K-factor, conserve a specified original blank length/volume, solve plastic
  strain, or predict tooling force, springback, thinning or wrinkling. The two
  face radii must not be interpreted as independent estimates of one radius.

The fixture samples the two broad faces only. It has no end caps, edge faces,
holes, relief cuts, varying curvature, twisted axes, partial formation along the
axis or compound bends. A flat sheet (`theta=0`) and reentrant turns beyond pi
are outside this model; they are rejected instead of relabeled as valid bends.
Positive dimensions do not establish physical manufacturability. Sampled exact
normals are construction oracles, not estimates from a noisy scanner.

## Three different geometric lines

The finite centre axis is `A(u)=c+u a`. A ruling actually on a chosen bend face
is `G(u)=c+u a+r n(theta f*)` for a declared fraction `f*`. They are separated
by radius `r`; neither is automatically an admissible product marker.
`axis_segment()` returns the former; `surface_generator()` returns the latter.
Both retain the stipulated finite axial interval, even when observation samples
cover less of it. The whole expected interval is not observed support.

For `0<|theta|<pi`, the **infinite tangent-plane intersection** is another line:

```text
V(u,r) = c + u a + r n0 + r tan(theta/2) k0
distance(V, A) = r / cos(theta/2)
```

This follows by solving `n0·(V-c)=r` and `n(theta)·(V-c)=r`. Its start-plane
contact offset is `r tan(theta/2)` along `k0`. For a positive turn the virtual
corner is beyond the contact in the direction opposite the actual start flange.
It need not lie on either finite flange. Changing from inner to outer faces moves
the virtual intersection by `t/cos(theta/2)`, although the cylinder centre axis
does not move. At a half-turn the tangent planes are parallel and have no finite
intersection; the cylindrical fixture is still defined.

Thus a plane-pair intersection cannot silently stand for the cylinder axis or a
supported surface ruling. This is a concrete representation distinction worth
testing in a later, separately scoped candidate-birth hypothesis, not a reason
to synthesize bends from every pair of planes.

## Sign, rigid motion, units and topology

`theta` rotates the ordered radial normals about directed `a`. For consistent
face-side normals and `|theta|<pi`, its signed inverse is
`atan2(a·(n0×n1), n0·n1)`. The smaller opening between the outward flange rays
is `pi-|theta|`, a different convention. At `|theta|=pi`, endpoint normals alone
do not distinguish the sign. Unknown normal orientation or mixing inner and
outer normals also invalidates naive signed-angle interpretation.

Reversing `a`, `theta`, and the ordered span to `(-u1,-u0)` preserves the physical
surface. The tests check this gauge change with reversed axial sample order.
A point cloud without an agreed directed-axis/ordered-face convention does not
provide a unique intrinsic signed scalar merely by having two normals.

A proper rigid transform applies `x'=Qx+d` and `n'=Qn`, where `Q^T Q=I` and
`det(Q)=+1`. The fixture transforms its frame and re-evaluates all patches;
the tests compare against independently transformed samples and finite endpoints.
Reflections, shear and scaling are rejected by this SE(3) API. In particular,
an anisotropic affine warp need not preserve cylinders, angles or thickness.
[Lynch and Park, Modern Robotics, section 3.3.1, Homogeneous Transformation Matrices](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/).

Unit conversion is separate from rigid motion. `in_units("m")` from millimetres
scales radius, thickness, origin, span and flange lengths by 0.001; angles,
directions, fractions and component identity do not scale. The API supports only
explicit `m` and `mm`, with no inferred input unit. `milli` denotes `10^-3`.
[BIPM, SI Brochure, ninth edition, section 3 and Table 7](https://www.bipm.org/documents/d/guest/si-brochure-9-pdf).

Each fixture has one stipulated `component_id`. Two components may have collinear
axes yet disjoint finite spans. The test uses spans `[-8,-3]` and `[3,8]` mm:
contracting them to `[-8,8]` fabricates support in a 6 mm gap. Keeping construction
IDs separate is not an algorithm for finding real components. Conversely, a gap
in a sparse scan alone does not establish disconnection; component topology and
capture absence must not be conflated.

## Counterexamples that constrain claims

### Hidden transition: radius and axis are not identifiable from the visible faces

Take a positive quarter-turn, known thickness `t=1` mm, and either `R=2` or
`R=4` mm. Set `c=(-R,0,-R)` and both flange lengths to `10-R`. Both models have
inner flange planes `z=0` and `x=0`, and outer planes `z=1` and `x=1`.
Observe only the same patches with start `x` from -10 to -8 and end `z` from -10
to -8, across the same axial span. The test constructs identical observed
coordinates and exact outward normals on **both** sides of both flanges.
Yet the radii, contact lines and centre axes differ.

The observation channel is therefore non-injective even with exact normals and
known thickness. No method using just these observations can uniquely recover
the radius/axis without additional assumptions or evidence. Matching fixture
metadata is not used to assert the observation equality. With only the start
flange visible, the separate 30-degree and 135-degree fixtures also have identical
observations: even the turn is not determined. With both oriented flange normals
visible, their relative turn can generally remain measurable, so missing
transition visibility does not make **every** quantity unidentifiable.

These examples support retaining unknown/unobserved states. They do not license
calling an unseen transition unformed or missing. Nor does an observation mask
prove that the physical surface was removed: `patches` and sample fractions
stipulate what is observed, not how a camera acquired it.

This is scan-only non-identifiability under the stated observation channel. A CAD
prior can fix a **nominal** radius, axis and expected support; it can choose one
model without resolving the scan-only inverse problem. It cannot independently
verify an unobserved **actual** radius or its conformance. Treating the CAD value
as measured scan evidence would erase precisely the distinction this example
tests. Additional manufacturing constraints are assumptions unless separately
verified; this geometry-only pair does not claim two equal-blank forming paths.

### Short visible arcs: a quantitative conditioning warning, not a confidence score

For a visible circular arc of angle `delta`, the chord and sagitta are

```text
ell = 2 r sin(delta/2)
h   = 2 r sin(delta/4)^2
```

The latter is the stable form of `r(1-cos(delta/2))`. At fixed chord width,

```text
h = r - sqrt(r^2-(ell/2)^2)
  = ell^2 / (4 (r + sqrt(r^2-(ell/2)^2)))
r = ell^2/(8h) + h/2
```

The rationalized form avoids subtracting nearly equal numbers. For `ell << r`,
`h` is approximately `ell^2/(8r)`: large-radius geometry has less curvature signal
over the **same visible footprint**, not necessarily over the same angular
interval.

The test uses `r=1000` mm and `delta=0.002` radians. Its roughly 2 mm-wide visible
arc has only about 0.0005 mm sagitta. Replacing its middle sample column by the
chord midpoint produces exactly planar positional samples with displacement
bounded by that sagitta. This is an explicit perturbation counterexample, not a
simulation or calibration of any scanner. Its noiseless oracle normals are not
included in the equality claim. Normals estimated from these same noisy points
are not independent sensor evidence. Independently measured normals with validated
uncertainty could provide additional information. A coverage count or fit score
is not thereby a calibrated uncertainty or a sufficient observability test.

For a stipulated per-point Euclidean position error bound `epsilon`, the geometry
gives several distinct **sufficient ambiguity** conditions, not one universal
detectability threshold:

- Projection to the fixed, noiseless chord plane changes each point by at most
  `h`. Thus `h<=epsilon` is already compatible with that reference plane.
- A free comparison plane can lie halfway between the chord and arc apex. Its
  corresponding points differ by at most `h/2`; `h<=2 epsilon` allows exact arc
  points to be explained by a plane with that error bound.
- If **both** the arc and planar hypotheses allow error `epsilon`, their midpoint
  observations differ from either hypothesis by at most `h/4`. Hence observation
  sets overlap at least when `h<=4 epsilon`. This does not say every possible
  observed sample set is ambiguous, or that larger sagitta guarantees success.

For any chosen sagitta budget `b`, at fixed chord width the condition `h<=b`
holds when `r>=ell^2/(8b)+b/2`, provided `0<b<=ell/2`. Equivalently at fixed
radius, `ell<=2 sqrt(2rb-b^2)` suffices when `0<b<=r`. Substituting `b=epsilon`,
`2 epsilon` or `4 epsilon` gives the corresponding radius/coverage ambiguity
bound above. These are ideal-model, bounded-error calculations; they are not
measured scanner limits, false-alarm probabilities or a sufficiency criterion
for detecting curvature beyond the bound. No uncertainty in a plane estimated
from the same cloud may be omitted by calling it the fixed reference plane.

The boundary oracle uses `ell=2` mm and `epsilon=0.001` mm. A free halfway plane
reaches `h/2=epsilon` at `r=250.001` mm. Generated cases on both sides of that
boundary verify actual Euclidean residuals and the two-hypothesis midpoint
construction independently of any fixture inverse or detector decision.

## Verification: independent oracles first, metamorphic relations second

Metamorphic testing checks required relations across transformed test cases; the
original research motivates deriving additional tests from existing ones.
It does not make a mutually consistent but wrong generator/detector pair correct.
[Chen, Cheung and Yiu, Metamorphic Testing: A New Approach for Generating Next Test Cases, HKUST-CS98-01 (1998)](https://www.cse.ust.hk/faculty/scc/publ/CS98-01-metamorphictesting.pdf).

Accordingly, the suite includes:

| Check | Oracle or counterexample, beyond repeatability |
| --- | --- |
| Positive/negative quarter-turn | Hand-calculated Cartesian points and outward normals, including nonzero thickness |
| General and boundary angles | Implicit cylinder/plane equations, axial bounds, constant face separation; tiny nonzero and exact half-turn cases |
| Tangent joins | Exact seam points/normals and independently measured secant directions approaching the tangent |
| Line conventions | Solved tangent-plane intersection offset and separate on-face ruling/centre-axis coordinates |
| Transform and units | Proper non-axis-aligned rotation, translation and inverse; all length fields scale, normals/identity do not |
| Gauge and sampling | Directed-axis reversal, nested resampling, permutation/repetition and partial angular support |
| Identifiability | Identical visible face data with different radii/axes or angles; short-arc planar perturbation; disconnected collinear spans |
| Invalid domain | Zero/negative/nonfinite dimensions, unsupported angles/units/faces, invalid frames, reflection/shear/scaling and bounded allocation |

Tests use explicit absolute numerical bounds with relative tolerance disabled for
array comparisons. These are floating-point verification tolerances, not physical
metrology tolerances. Normals and construction labels must not be passed as
privileged inputs to a future scan-only detector. A future detector test should
expose only its declared observation channel and compare predictions against
separately retained analytical truth, including correct abstention cases.

NIST's coordinate-metrology algorithm-testing practice distinguishes software
results from separately supplied reference fits, including numerical uncertainty.
That supports the methodological separation here; this fixture uses no NIST data
or reference-fitting service and does not claim its certification or precision.
[NIST, Algorithm Testing, testing process and uncertainties](https://www.nist.gov/pml/sensor-science/dimensional-metrology/algorithm-testing).

## Executed checkpoint and stop condition

On 2026-09-07 the current source passed **72 tests in 0.08 s**, using Python
3.11.15, NumPy 2.4.6 and pytest 9.1.1, with this command from the Git root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_analytic_bend_fixture.py -q
```

Import inspection found only standard-library dataclasses/math and NumPy in the
fixture; the test adds pytest/NumPy testing and this fixture. The namespace package
path has no repository initializer to execute. Project pytest config, conftest,
plugin autoload, cache and bytecode writes were disabled. No server, detector,
C15/frozen mechanism, corpus, control record or physical input was imported or run.
No dependencies were installed or upgraded. No raw/restricted paths were opened,
listed, statted, hashed or searched. These three new source/test/narrative files
are the only artifacts in this subtask.

The next useful, separately scoped engineering decision is whether a proposed
finite-transition mechanism can recover an explicitly declared observation
channel against these independent oracles and abstain in the ambiguous cases.
Do not count these 72 tests as detector correctness, candidate recall, an unseen
family, physical uncertainty calibration, a new valid experiment, or production
qualification. Physical evidence and a new detector/evaluation change still need
their respective scope and gates. Existing frozen mechanisms and historical
scoring rules remain unchanged.
