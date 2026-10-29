# Certified oriented raw-extrema bounds — independent math review

Date: 2026-09-12. Reviewer: `context_v3_adversarial`.

## Disposition and scope

**Accept the certification hypothesis; reject an unconditional acceleration
claim. Recommend an explicit two-slope shear as the first bounded primitive.**
The proof below preserves the original represented-row leaf program and raw-ID
ties. It does not prove that enough rows will be pruned to avoid the observed
work cap, nor that the whole workflow will reach measurement.

Read completely: canonical AGENTS, the current-status top, the 19,626-byte
`backend/cad_guided_raw_extrema_index_v1.py` (SHA256
`668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121`), and the
20,741-byte V7 scientific diagnosis (SHA256
`eaaef192cb689bf1e7cb0d390aa3a47ce85138d38862332f93e46ef0d25ac271`).
No raw/Gold/cache/attempt artifacts were accessed for this task. No target
imports, executions, tests, fitting, or existing-source edits occurred.

The diagnosis attributes ten slots to the 256-million extrema-coordinate cap,
after roughly 26.6–27.1 million successfully evaluated query rows per slot.
Those narrative counters motivate reducing repeated leaf evaluation; they do
not reveal the distribution, covariance, thickness, or extremal gaps of any
particular raw support. This review does not infer those missing quantities.

Root is separately authoring a new shear-bound primitive. This document reviews
formulas, not that unread implementation. It is not a source qualification,
physical release, new test gate, or permission to relax existing caps.

## 1. Invariant: the original leaf program

All inputs below are their represented finite binary64 values, regarded as
exact real numbers when writing algebra. `RN` denotes the same separate
round-to-nearest/ties-to-even binary64 operation as the existing NumPy ufuncs,
with gradual underflow and no contraction, reassociation, extended accumulator,
or BLAS substitution. For raw row x, direction d, origin o and constant c:

```
t_j = RN(x_j - o_j)                 j = 0,1,2
p_j = RN(t_j * d_j)
s01 = RN(p_0 + p_1)
s2  = RN(s01 + p_2)
v   = RN(s2 + c)
output = +0 if v == 0 else v
```

Keep `_staged` unchanged (source lines 69–77), `_evaluate` raw-row selection and
minimum-ID comparisons unchanged (250–262), and the original `_bounds`
monotone world-box enclosure (240–248). Oriented arithmetic is only a second
certificate for pruning; it never supplies the returned extrema/witness values.

The represented input bounds |x|, |o|, |d|, |c| <= 1e6 put original subtraction
outputs within 2e6, products within 2e12, and final magnitude within
6e12+1e6. Thus the original validated domain is far from binary64 overflow.
Subnormal products and signed zero remain possible. New local arithmetic must
not assume the same safety merely because its source coordinates are bounded.

## 2. Small outward primitives, including subnormals

For finite represented scalar operands a,b and an operation whose exact result
is finite and whose rounded result/required outward neighbors are finite, use:

```
down(a op b) = nextafter(RN(a op b), -infinity)
up(a op b)   = nextafter(RN(a op b), +infinity)
```

The exact result lies between these neighbors. This follows from nearest
rounding: the exact value lies in the rounded result's rounding cell, whose
boundaries lie between adjacent floats. At zero the neighbors are ±eta,
eta=2^-1074, so underflow is enclosed rather than discarded. Exact operations
may be widened unnecessarily; that costs pruning, not correctness.

Interval addition uses the lower/lower and upper/upper pairs; subtraction uses
lower/upper and upper/lower. Interval multiplication takes the minimum of four
outward lower endpoint products and maximum of four outward upper endpoint
products. These formulas handle arbitrary coefficient signs and intervals
straddling zero; do not choose a sign from an uncertain coefficient midpoint.
Min/max/absolute value on finite represented endpoints are exact comparisons.

Reject NaN, unordered intervals and unhandled nonfinite endpoint arithmetic.
Do not suppress an overflow and then treat its clipped finite neighbor as a
certificate. An explicitly classified inability to construct optional local
metadata may deterministically disable that metadata before use; contradictory
finite bounds are an invariant failure, not a reason to silently hide the bug.
Budget/work failure must still trigger the caller's whole-output clearing.

No relative-only `gamma_n` assumption is used. Rump's primary analysis explicitly
distinguishes the absolute underflow contribution for multiplication/division;
that is why relative error alone is unsuitable here. The argument below uses
neighbor spacing instead. See [Rump, Error estimation of floating-point
summation and dot product, Section 2](https://www.tuhh.de/ti3/paper/rump/Ru11.pdf).

## 3. Preferred original-program rounding envelope: one outward ULP

Root proposed the following simpler construction. **Accepted**, with the finite
neighbor and actual rounded-stage-range conditions made explicit.

Trace the unchanged original program at the two sign-selected world-AABB
corners. Retain both corner outputs at every subtraction coordinate, product
coordinate, sum01, sum2, and final addition. The subtraction corner order may
reverse for a negative direction; maximum absolute value of the two endpoints
still bounds every actual subtraction output. Product endpoints are ordered
by the sign selection; both additions are monotone. Consequently each actual
rounded stage output y has |y| <= m, where m is the maximum absolute value of
the relevant two rounded corner-stage outputs.

Define, for finite m >= 0 with finite successor:

```
gap(m) = nextafter(m, +infinity) - m
```

This subtraction is exact: consecutive nonnegative finite binary64 values have
a representable power-of-two spacing (including eta); their difference is
representable. `gap(0)=eta`. The function is nondecreasing over nonnegative
finite binary64 values, including at binade boundaries.

For an individual finite rounded result y, nearest-rounding error is at most
half the larger adjacent gap around y. By sign symmetry that larger gap is
no greater than the spacing toward +infinity from |y|. At a positive power of
two the upward gap is the larger one, so choosing it remains conservative.
Therefore, if |y| <= m, the absolute local rounding error is <= gap(m). This
also covers exact zero, the normal/subnormal boundary, and a product rounded
from eta/2 to zero. No division by |y| is used. An infinite successor is not
admitted; the current bounded original domain cannot approach that case.

This proof is consistent with the documented nearest-neighbor error property
of gradual underflow; it is not an assumption that subnormals have normal
relative precision. See [Oracle Numerical Computation Guide, error properties
of gradual underflow](https://docs.oracle.com/cd/E77782_01/html/E77791/z4000ac020351.html).

Let g_tj, g_pj, g_01, g_2, g_c be these gap bounds for the three subtractions,
three products and three additions. With **each** nonnegative multiplication
and addition rounded outward upward, compute:

```
e_j = up( up(abs(d_j) * g_tj) + g_pj )
E01 = up( up(e_0 + e_1) + g_01 )
E2  = up( up(E01 + e_2) + g_2 )
E   = up(E2 + g_c)
```

For every row, write each rounded operation as its exact operand operation
plus an additive error. Subtraction error is multiplied by d_j; the next local
product error is then added. Addition errors accumulate with coefficient one.
Triangle inequality gives

```
abs(original_staged_output(x) - (sum_j d_j*(x_j-o_j) + c)) <= E.
```

The final conversion of either signed zero to positive zero does not change
its exact real value. This derivation encloses the ORIGINAL arithmetic, not a
new local-coordinate dot product. Retaining only a final world-box range is
insufficient to compute these radii: actual original intermediate stage ranges
are required. They can be obtained in the same two-corner trace rather than by
duplicating the complete corner work.

## 4. General represented-basis certificate

Choose any finite represented anchor h, matrix B and represented per-row local
coordinates z. An approximate rotation, approximate inverse, fixed heuristic,
or even singular B is allowed: none supplies a correctness axiom. Define the
exact reconstruction residual

```
r(x) = x - h - B*z.
```

At construction, compute each residual component with outward interval
operations on the ORIGINAL represented x, h, B, z. For each node store the
componentwise hulls Z containing all represented z and R containing every
per-row residual interval. Union children by endpoint minima/maxima. Every
raw row assigned to that node must contribute; sparse fitting/sampling cannot
replace this certification pass.

The exact identity x=h+Bz+r gives

```
f(x) = d·(x-o)+c
     = (d·(h-o)+c) + sum_l (sum_j d_j*B_jl)*z_l + sum_j d_j*r_j.
```

Compute interval coefficients for each exact sum_j d_j*B_jl and the exact
constant d·(h-o)+c; coefficient rounding must be enclosed. Then evaluate the
right side over Z and R with outward interval arithmetic to obtain J=[Jlo,Jhi].
The local coordinates and residuals are correlated, but replacing their joint
set with a Cartesian box is an over-approximation, never an under-approximation.
No B-transpose-equals-inverse or orthogonality assumption is needed.

The original rounded-program certificate is

```
oriented = [down(Jlo-E), up(Jhi+E)]
combined = [max(world_lower, oriented_lower),
            min(world_upper, oriented_upper)]
```

Both inputs enclose every original staged value, so their intersection does
too. A reversed finite intersection is an implementation/invariant error.
Returning an oriented endpoint as an observed extremum is prohibited.

## 5. Recommended simpler alternative: explicit triangular shear

Fix a permutation (i,j,k) of the original axes and finite represented slopes
a,b with |a|,|b| <= 1. No basis chooser, fit or PCA is needed by the primitive.
For each raw row define represented local coordinates:

```
q = x_i                         # exact alias/value
t = x_j                         # exact alias/value
z = RN( RN(x_k - RN(a*x_i)) - RN(b*x_j) )
r = x_k - a*x_i - b*x_j - z     # exact-real definition, not RN value
```

Certify r using outward interval multiplication/subtraction on the represented
row and represented z. In particular, the singleton computed difference
RN(x_k - reconstructed_x_k) is NOT itself an interval for r. Store node z
min/max and r interval hull. The q/t boxes reuse the original x_i/x_j boxes.

The exact algebraic inverse is x_k=a*q+b*t+z+r, so:

```
f(x) = (d_i+a*d_k)*q + (d_j+b*d_k)*t + d_k*z + d_k*r
       - sum_l d_l*o_l + c.
```

Evaluate every coefficient, constant and interval operation outward, then use
the SAME original-program E and world-box intersection from Section 3/4. This
does not replace leaf subtraction, multiplication or summation with the shear
formula. Nonorthogonality is intrinsic and harmless because the inverse is an
exact identity with a certified residual.

Benefits over the general basis: no matrix inversion or orthogonality proof;
two raw coordinates are preserved exactly; only one local coordinate and one
residual need node hulls; two direction coefficients replace a nine-product
basis transform. With bounded slopes, |z| is at most approximately 3e6, so the
new construction is also far from overflow. Still check finite results.

Dyadic slopes can make an eventual heuristic deterministic but are not an
exact-multiplication proof: multiplication by a general dyadic can round.
Even multiplication by a power of two can underflow. Residual certification is
not optional. Coefficients 0 or ±1 permit some separately proved exact shortcuts,
but such shortcuts are not required for this conservative base construction.

If a later index chooses slopes from a few existing seed rows, those rows may
only choose the heuristic. All indexed rows still certify the hull. Degenerate
or poor choices may use a declared identity/world-box fallback. No need to add
an unbounded eigensolver or a numerical fitting campaign to the primitive.

## 6. Ties, traversal, failure and a useful dominance guarantee

Retain the V1 row permutation, node membership, stable world-axis sort, seed
rows, fixed leaves256 and minimum original raw ID per node. Keep the exact
pruning predicates at source lines 280–281:

```
no_min = L > best_min or (L == best_min and node_min_id >= best_min_id)
no_max = U < best_max or (U == best_max and node_min_id >= best_max_id)
prune iff no_min and no_max
```

No epsilon ties, approximate equality, local-coordinate witness IDs, signed-zero
ordering, or early acceptance of a matching scalar is permitted. Equality at
a bound does not justify skipping a possible smaller raw-ID witness.

Use OLD world bounds for the existing child-priority expression
`max(best_min-world_lower, world_upper-best_max)` and its node-ID tie-break;
use the intersection only for pruning. Then, for a completed query on the same
tree/seeds, leaf-row evaluations cannot exceed V1: any additionally removed
subtree cannot improve either incumbent value or ID. It therefore would not
have changed incumbents in V1; induction leaves the remaining visit order and
incumbents identical while deleting only redundant visits. Tightening bounds
but changing child priorities or spatial ordering loses this simple guarantee.

This is a guarantee about leaf evaluation count, not total work, wall time,
memory, or completion under a cumulative budget. New construction/bound work
can outweigh saved rows or cause a new honest budget refusal. Preserve attempted
precharges and whole-output clearing on any late failure; never salvage the
earlier successful projection prefix. If a new feature is disabled, its already
attempted work remains charged.

## 7. Counterexamples and preimplementation falsifiability

1. **Exact oblique plane remains a full-scan case.** Rows (t,t,z), origin zero,
   direction (1,-1,0), constant zero give +0 for every original staged row:
   multiplication by ±1 is exact and the first sum cancels exactly. The exact
   affine functional is identically zero. Nevertheless a generic positive
   residual/rounding radius makes every nontrivial oriented interval straddle
   zero, so neither extremum can be excluded. The original world boxes also
   straddle zero. Even having the minimum raw ID already seeded cannot prune.
   A shear alone does not solve this. The same issue applies to line supports
   queried in a perpendicular direction.

2. **Thickness alone is insufficient.** Put repeated ±thickness levels in every
   leaf. Even if thickness greatly exceeds rounding error, all leaf maxima can
   equal the global maximum; positive inflation keeps each node competitive.
   Useful pruning depends on node extrema gaps relative to reconstruction and
   rounding inflation, not just a global noise/thickness ratio.

3. **Exact affine values are not staged values.** x=(1,1,1), o=0,
   d=(1,2^-53,-1), c=0 has exact affine value 2^-53 but original staged value
   zero, because RN(1+2^-53)=1. Enclosing only the affine program is unsound.

4. **Relative-only error fails underflow.** x=(eta,0,0), d=(0.5,0,0), o=0,c=0
   yields exact product eta/2 and rounded zero. Its relative error is one, not
   a small gamma_n. `gap(0)=eta` safely covers it. Computing eta/2 in binary64
   and using the resulting zero as a radius would not.

5. **A rounded reconstruction equal to x does not prove residual zero.**
   RN(1+2^-54)=1, while the exact sum differs from 1. Therefore testing equality
   after a floating reconstruction and assigning r=0 is invalid. A represented
   nearly orthogonal basis similarly does not justify B^-1=B^T.

6. **Rounded coefficients can hide extrema.** A local coefficient whose exact
   sum is nonzero may round to zero by cancellation. Multiplying a singleton
   rounded coefficient by a wide local interval omits real variation. Enclose
   coefficient arithmetic before interval multiplication.

The first prototype must expose, not obscure, the exact-flat-normal failure.
A smallest useful future invented efficacy comparison would keep the same
dyadic oblique grid, IDs, tree and query, and vary only three support patterns:
exactly flat; bounded thickness with separated unique extremal levels; repeated
extremal plateaus. A single moderate grid spanning many leaves is enough to
falsify universal pruning; increasing to physical-size fixtures is unnecessary
at this mathematical stage. All completed answers must equal an independent
original-staged oracle bit-for-bit after zero normalization, including IDs.

Record original and intersected interval widths, original error radius,
reconstruction-residual width, node extremal gaps where an invented oracle can
bound them, evaluated rows, total precharged construction/query operations, and
peak accounted live payload. A saving in leaf rows alone is not an efficacy
pass if total work rises. Before integration, reject or revise the performance
hypothesis if noisy/separated invented supports do not save net work under a
fixed declared budget. No speedup threshold or new execution envelope is frozen
by this document. Existing physical cap refusals remain valid worst cases.

## 8. Finite construction/query/memory accounting

Keep one shared immutable raw owner and existing support-index identity/reuse.
Do not allocate another raw copy per local frame. Prefer a streaming pass over
the existing <=256-row leaves, discard per-row z/residual scratch after updating
leaf hulls, then merge child hulls. No full-support retained local-coordinate
array or extra sort is necessary for either certificate.

Let S be total distinct-index memberships (including whole buffer), I the number
of indexes, and K the sum of existing allocated node slots. For support size s,
K_s=2*next_power_of_two(ceil(s/256)); consequently
K < S/64 + 4I. At the prospective context ceilings S=4,000,000 and I=129,
K<63,016. These are capacity formulas, not a measured allocation receipt.

Additional persistent numerical payload, retaining the original world boxes:

| Representation | Added node payload | Added fixed coefficient payload |
| --- | ---: | ---: |
| General 3D basis: three z and three residual intervals | 96 K bytes | 96 I bytes for B and anchor |
| Two-slope shear: one z and one residual interval | 32 K bytes | slopes plus axis metadata, explicitly accounted |

Thus node additions alone are below 6,049,536 bytes (general) or 2,016,512 bytes
(shear) at those context ceilings. Python/container/array-header overhead,
coefficient storage, report records, scratch, old/new arrays during installation,
and every already-live parent/CAD/cache payload must also be charged. These
numbers are not process RSS and do not justify ignoring the cumulative parent
working-memory cap. A leaf primitive needs only a fixed tiny scratch envelope,
but its exact arrays/lifetimes must be reviewed in source before freezing bytes.

An implementable work ledger separately counts coordinate construction, residual
interval primitives, hull reductions, original corner-stage trace, outward-gap
and error propagation, affine-bound interval primitives, comparisons and
nextafter calls. For reference, general residual certification has nine interval
products and twelve interval subtractions per row, plus the explicitly chosen
coordinate constructor. The shear needs two scalar products/two scalar
subtractions to construct z and two interval products/three interval subtractions
to certify r, plus finite checks and four hull endpoint comparisons per row.
All are fixed per-row counts; no recursive arithmetic refinement is needed.

The shear query coefficients and origin constant are node-independent within an
index/query and may be computed once under a precharge. Each node's exact-affine
box then uses four interval products (q,t,z,r) and bounded additions, plus the
two original-corner stage traces and the fixed nine-gap error recurrence.
General-basis query coefficients may likewise be precomputed; its node affine
box uses six interval products. Count every scalar endpoint operation actually
selected by the implementation, including comparisons and nextafter, before
performing it. Vector batching must precharge its full scalar denominator.

Do not relabel this work as free node bookkeeping or overwrite V1 counters with
lower numbers. Preserve the original 9-per-row leaf arithmetic charge and
report the additional model/version. Either add a separately bounded primitive
ledger or explicitly convert it into an existing aggregate work unit; that
choice and its limits require source review. Merely retaining the numeric
256-million cap while changing what a unit means is not an honest comparison.

## Final recommendation

Proceed only with a small, separately reviewed shear certificate primitive:
explicit axes/slopes, original represented rows, outward residual hull, exact
affine interval, original-stage one-outward-ULP envelope, old-world intersection,
and precharged finite work. No basis-selection machinery is necessary to test
that proof boundary. Preserve the original tree/order/seeds and leaf program
when subsequently integrating it.

The mathematical construction is sound under the stated arithmetic assumptions;
its efficacy on the ten consumed V7 capacity failures is **unproved**. Flat
normal queries and repeated extreme plateaus remain legitimate full-scan cases.
Nothing here certifies physical alignment, a measured bend, improved F1, changed
thresholds, extra cap, or a successor physical run.
