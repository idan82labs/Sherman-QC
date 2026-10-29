AUTHORIZE_V72_FIXTURE_ONLY_CORRECTION

# `V72_CASL_SOURCE_FREE_LOCK_AMENDMENT_A1`

The preserved 37/43 run remains immutable evidence that the original lock specification was internally inconsistent; it is not a V72 architecture result. V72 is neither locked nor retired. This amendment authorizes exactly one replacement 43-case lock invocation containing its frozen primary run and internal replay.

The production CASL module must remain byte-identical at SHA-256:

`b0da73bd3fc59fdab968d1bbe87d27f0a4dc215abca48e199589b81f6f01071a`

Only the fixture definitions and, where necessary, fixture-runner construction/assertion code may change as specified below.

## 1. Cap-safe analytic generators

### 1.1 Selected correction

Use **radial/profile-endpoint trimming only**. Do not reduce axial support, thin records, raise or bypass the source cap, or exempt source-free fixtures.

The common axial grid remains exactly

[
z_m=\frac m2,\qquad m=0,\ldots,340,
]

so every affected fixture retains all 341 axial planes, its original axial extent, candidate endpoints, station positions, and axial sampling density.

For **G10, G15, G16, and the exact twofold profile-refinement case only**, replace the inclusive profile reach (75) by the maximal cap-safe inclusive reach

[
R_{\mathrm{fixture}}=73.
]

All profile maps, bend radii, centers, directions, surface formulas, normals, faces, connectivity, component ownership, and candidate rows remain unchanged.

All index sets below are closed integer ranges. Coordinates must be generated from the integer index directly; floating-loop termination, `arange`, accumulated addition, and post-generation clipping are forbidden.

### 1.2 G10 — two G03 bends centered 24 apart

Retain the two frozen translated G03 profile maps (P^{(0)}*{G03}) and (P^{(1)}*{G03}), including the frozen 24-unit center separation.

Replace the former signed half-unit profile indices (k=-150,\ldots,150) by

[
u_k=\frac{k}{2},\qquad k=-146,\ldots,146.
]

Each G03 copy therefore has

[
293
]

profile records per axial plane. The two copies remain distinct and produce

[
N_{\mathrm{G10}}
=341(2\cdot293)
=199{,}826.
]

No point from one bend may acquire ownership, weight, adjacency, or multiplicity from the other bend.

### 1.3 G15 — four-ray crossing

For each of the four frozen ray directions (v_j), (j=0,1,2,3), generate

[
p_{j,k}=c+\frac{k}{2}v_j,
\qquad k=0,\ldots,146.
]

Each ray has 147 samples, including its origin. The four bitwise-identical origins are unioned into one source record:

[
N_{\mathrm{profile,G15}}
=4\cdot147-3
=585,
]

and therefore

[
N_{\mathrm{G15}}
=341\cdot585
=199{,}485.
]

The retained origin carries the union of all frozen incident topology. Duplicate generation must not multiply its weight or effective count.

### 1.4 G16 — two unconnected full parallel lines

For each frozen parallel-line center (c_j), (j=0,1), retain the original direction (v) and generate

[
p_{j,k}=c_j+\frac{k}{2}v,
\qquad k=-146,\ldots,146.
]

Each line has 293 records per plane. They remain separate, unconnected components:

[
N_{\mathrm{G16}}
=341(2\cdot293)
=199{,}826.
]

No bridge, shared face, duplicate ownership, or synthetic adjacency may be introduced.

### 1.5 Exact twofold profile refinement

The coarse and refined versions must use the same closed support interval ([-73,73]).

Coarse profile:

[
u_k=\frac{k}{2},
\qquad k=-146,\ldots,146,
]

giving

[
293\ \text{records per plane},\qquad
N_{\mathrm{coarse}}=341\cdot293=99{,}913.
]

Refined profile:

[
\widetilde u_j=\frac{j}{4},
\qquad j=-292,\ldots,292,
]

giving

[
585\ \text{records per plane},\qquad
N_{\mathrm{refined}}=341\cdot585=199{,}485.
]

The refinement must be exactly nested:

[
\widetilde u_{2k}=u_k
\quad\text{for every }k=-146,\ldots,146.
]

The refined source is regenerated from the analytic profile map. It must not be created by duplicating coarse records, copying vertices with new IDs, assigning multiplicity, or increasing weights.

### 1.6 Why 73 is frozen

Seventy-three is the largest common endpoint on the required grids that keeps every affected fixture within the real source cap:

* A half-unit full profile extended to (73.5) would give (590) records per plane for G10/G16 and (201{,}190) total records.
* Four half-unit rays extended to (73.5) would give (589) records per plane and (200{,}849) total records.
* A quarter-unit refined profile extended to (73.25) would give (587) records per plane and (200{,}167) total records.

No smaller endpoint may be substituted.

### 1.7 Duplicate and tail-faithfulness rules

Generation order remains the frozen order. For any duplicate created by the affected analytic maps:

1. Equality means equality of all three binary64 coordinate bit patterns.
2. Emit one source vertex.
3. Retain the union of its frozen incident faces and adjacency.
4. Use the lowest frozen primitive ordinal only as the serialization owner.
5. Do not sum duplicate weights or count duplicate occurrences in (W), (Q), coverage, or (n_{\mathrm{eff}}).

The runner must additionally assert, without inserting the removed records into the source, that every removed tail record has zero membership in every frozen CASL station-annulus query for that fixture. The audited tails are exactly:

[
|u|\in{73.5,74,74.5,75}
]

for signed half-unit profiles,

[
u\in{73.5,74,74.5,75}
]

for rays, and

[
|u|\in{73.25,73.5,\ldots,75}
]

for the refined quarter-unit profile.

Any nonzero frozen-query membership is a lock failure; it may not be cured by another endpoint edit.

## 2. Six-degree direction-drift boundary

### 2.1 Production comparator remains unchanged

The production decision remains the frozen

[
\delta
=\operatorname{acos}
\left(
\operatorname{clip}
\bigl(\lvert \hat d_0!\cdot!\hat d_1\rvert,-1,1\bigr)
\right)
\le \theta_6,
]

with

[
\theta_6
=\operatorname{float64}!\left(\frac{6\pi}{180}\right)
=\texttt{0x1.acee9f37bebd5p-4}.
]

Conversion of the production predicate to dot/cosine space is **not authorized**.

### 2.2 Exact equality is not constructible on the frozen path

Let

[
c_{\mathrm{out}}
=\texttt{0x1.fd31f94f867c6p-1}
=0.9945218953682733
]

and

[
c_{\mathrm{in}}
=\texttt{0x1.fd31f94f867c7p-1}
=0.9945218953682734.
]

These are adjacent binary64 values:

[
\operatorname{nextafter}(c_{\mathrm{out}},+\infty)
=c_{\mathrm{in}}.
]

The frozen path gives

[
\operatorname{acos}(c_{\mathrm{out}})
=0.10471975511966022

> \theta_6,
> ]

while

[
\operatorname{acos}(c_{\mathrm{in}})
<\theta_6.
]

Because no binary64 dot product exists between these adjacent values, no binary64 direction vector can make the frozen `acos` result equal (\theta_6) exactly. The corrected fixture must therefore bracket the machine boundary rather than claim an impossible equality.

### 2.3 Exact replacement vectors

Use the reference direction

[
\hat d_0=(1,0,0).
]

Nearest failing vector:

[
\hat d_{\mathrm{out}}
=====================

\left(
\texttt{0x1.fd31f94f867c6p-1},
\texttt{0x1.ac2609b3c578ap-4},
0
\right).
]

Nearest passing vector:

[
\hat d_{\mathrm{in}}
====================

\left(
\texttt{0x1.fd31f94f867c7p-1},
\texttt{0x1.ac2609b3c573dp-4},
0
\right).
]

For both vectors, frozen binary64 evaluation of (x^2+y^2) is exactly `1.0`, so normalization cannot move the boundary operands.

The existing six-degree JUnit case remains one of the 43 cases but contains these paired assertions:

[
\operatorname{nextafter}
\bigl(
\hat d_0!\cdot!\hat d_{\mathrm{out}},+\infty
\bigr)
======

\hat d_0!\cdot!\hat d_{\mathrm{in}},
]

[
\delta_{\mathrm{out}}>\theta_6
\quad\text{and therefore FAIL},
]

[
\delta_{\mathrm{in}}\le\theta_6
\quad\text{and therefore PASS}.
]

All five nonzero constants must be loaded from their hexadecimal encodings. The fixture must not call `sin`, `cos`, or `sqrt` to reconstruct them.

This is the exact binary64 inclusive-boundary test: (\hat d_{\mathrm{in}}) is the largest constructible angle accepted by the unchanged production path, and (\hat d_{\mathrm{out}}) is the immediately adjacent rejected value. No angular epsilon or tolerance inflation is permitted.

## 3. One-ulp-below effective count

Retain the frozen formula and comparison:

[
n_{\mathrm{eff}}
=\frac{W^2}{Q},
\qquad
n_{\mathrm{eff}}\ge12.
]

Replace the failed fixture operands by

[
W
=12
=\texttt{0x1.8000000000000p+3},
]

[
Q
=\operatorname{nextafter}(12,+\infty)
=\texttt{0x1.8000000000001p+3}.
]

Using the unchanged binary64 operation order,

[
T=\operatorname{fl}(W\cdot W)
=144
=\texttt{0x1.2000000000000p+7},
]

followed by

[
n_{\mathrm{eff}}
=\operatorname{fl}(T/Q)
=11.999999999999998
=\texttt{0x1.7ffffffffffffp+3}
=\operatorname{nextafter}(12,-\infty).
]

The expected result is therefore rejection by the unchanged inclusive comparator.

The case must inject these exact (W,Q) aggregate inputs into the existing effective-count computation. Direct injection of a precomputed (n_{\mathrm{eff}}) scalar is not authorized. The existing exactly-at-12 fixture remains unchanged.

## 4. Corrected lock authorization and terminal gates

Exactly one new top-level lock invocation is authorized. It must execute all 43 cases in their original frozen order and then execute the existing internal replay within that same invocation.

Before execution:

* Preserve the original ruling, implementation, fixture, runner, and failed JUnit artifacts unchanged.
* Precommit new fixture and runner hashes.
* The diff must contain only the endpoint/index changes, hexadecimal boundary operands, and assertions specified above.
* The CASL implementation hash must still equal `b0da73bd3fc59fdab968d1bbe87d27f0a4dc215abca48e199589b81f6f01071a`.

V72 locks only if all of the following hold simultaneously:

1. Primary result: exactly `43 passed, 0 failed`.
2. Internal replay: exactly `43 passed, 0 failed`.
3. Every canonical per-case result and aggregate artifact is byte-identical between primary and replay.
4. G10 reports exactly 199,826 source records.
5. G15 reports exactly 199,485 source records.
6. G16 reports exactly 199,826 source records.
7. Refined twofold profile reports exactly 199,485 source records.
8. No source, graph, station, annulus, effective-count, memory, or atomicity cap abort occurs.
9. No U01-U04, native visual, reserved physical scan, gold, label, CAD, expected-count, identity, filename, or benchmark artifact is opened during the lock.

Any deviation terminally retires V72 under `V72_CASL_TERMINAL_RETIRE`; no further fixture correction, threshold edit, implementation rescue, or replacement lock run is authorized.

Only after the exact 43/43 primary result and byte-identical 43/43 internal replay have been durably recorded may the unchanged U01-U04 one-shot protocol be opened. Its inputs, ordering, caps, formulas, candidate comparator, and replay rules remain frozen. Native-resolution visual adjudication remains mandatory after U01-U04 and before any reserved physical-scan opening.
