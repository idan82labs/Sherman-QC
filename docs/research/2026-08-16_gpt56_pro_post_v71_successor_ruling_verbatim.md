EXECUTE_ONE

# V72 — `CANDIDATE_CENTERED_ANNULAR_SECANT_LAW_V1`

**Abbreviation:** `CASL_V1`

**Disposition:** one frozen, source-only, deletion-only successor on the exact 48-candidate U01–U04 handoff consumed by V71.

CASL does **not** call V70 or V71, does not reuse either architecture’s decisions, and contains no graph, mesh-geodesic, connected-component, voxel-DTG, edge-closure, Q0, or halo expansion. Its dependency domain is a fixed set of candidate-centered Euclidean annuli. Work is bounded by source-point count, candidate count, station count, and fixed moment arrays before real-scan topology is observed.

The key discriminator is a cross-sectional secant law:

[
\begin{array}{lll}
\text{localized wedge/bend:}
&\theta(r)=\theta_0,
&\text{constant opening angle},[2mm]
\text{return/hem:}
&g(r)=2r\sin \frac{\theta(r)}2=g_0,
&\text{constant parallel-tail gap},[2mm]
\text{smooth circular roll:}
&\kappa(r)=\dfrac{2\cos \frac{\theta(r)}2}{r}=\dfrac1R,
&\text{constant curvature}.
\end{array}
]

A rounded physical bend may obey the circular law in its inner annuli but must transition to a wedge or return law in an outer annular window. A pure smooth roll continues to obey the circular law and receives a strong negative certificate. Flat-face and free-edge candidates have separate exact negative certificates.

CASL is conservative: it suppresses only a candidate with a persistent affirmative **anti-crease** certificate. Missing or ambiguous evidence is `ABSTAIN_KEEP`, never deletion.

---

## 1. Exact insertion and output semantics

CASL consumes the identical frozen candidate records and source-only geometry handed to V71.

For each scan:

[
C={c_0,\ldots,c_{m-1}},\qquad 1\le m\le64,
]

where each candidate is

[
c=(a_c,b_c,H_c),\qquad
L_c=\lVert b_c-a_c\rVert>0.
]

`H_c` is an opaque mapping identifier. It may be written to the audit artifact but may not enter any geometric statistic, comparator, ordering score, or decision.

For measurement only, endpoint order is canonicalized lexicographically:

[
(\alpha_c,\beta_c)=\operatorname{lexsort}(a_c,b_c),\qquad
d_c=\frac{\beta_c-\alpha_c}{L_c}.
]

The original endpoint bytes are returned unchanged.

CASL emits exactly one of:

* `CERTIFIED_KEEP`;
* `ABSTAIN_KEEP`;
* `SUPPRESS`.

The final result is a hash-subset of the frozen V52 result:

[
O_{72}\subseteq O_{52}.
]

CASL may not add, move, rotate, shorten, extend, split, merge, replace, or rerank a marker. No later stage may promote a candidate that V52 did not emit.

---

## 2. Allowed and forbidden inputs

### Allowed

1. Source coordinates

[
P={p_i\in\mathbb R^3}_{i=0}^{N-1},
\qquad 9\le N\le200{,}000.
]

2. The source-only local-spacing scalar for every point,

[
h_i=d_8!\left(p_i,P\setminus{p_i}\right)>0,
]

where (d_8) is Euclidean distance to the eighth distinct source record, and

[
h_0=\operatorname{median}_{i}h_i.
]

The existing frozen generic spacing table may be reused, but its defining formula and source-state hash are part of the lock.

3. Candidate endpoints and opaque hash.

### Forbidden

Candidate score, confidence, rank, route, detector family, provenance, V52/V70/V71 decision, filename, path, scan ID, part ID, family, acquisition identity, expected count, CAD, drawings, labels, gold, pseudo-truth, benchmark outcomes, manual visual judgments, triangle identity, mesh connectivity, adjacency, normals, and other candidates’ geometry.

Candidates are evaluated independently. Nearby candidates cannot change one another’s statistics.

---

## 3. Deterministic preprocessing

All arithmetic is IEEE-754 binary64, round-to-nearest-even, with fast-math, reassociation, contraction, FTZ, and DAZ disabled. The scientific artifact contains no timestamps or nondeterministic telemetry.

Normalize `-0.0` to `+0.0`. Reject every NaN or infinity.

Source records are processed in canonical lexicographic IEEE-total-order of

[
(p_{ix},p_{iy},p_{iz},h_i).
]

The canonical order is produced by a fixed stable radix implementation. Identical records are interchangeable because all accumulated values are identical.

Candidates are processed in lexicographic canonical-endpoint order followed by (H_c), but decisions are candidate-independent and are serialized in ascending (H_c).

All medians use the middle order statistic for odd cardinality and the binary64 arithmetic mean of the two middle values for even cardinality.

Use the correctly rounded binary64 value

[
\pi=\texttt{0x1.921fb54442d18p+1}.
]

All stated comparator boundaries are inclusive. A value one binary64 `nextafter` beyond a boundary fails that comparator.

---

## 4. Candidate scale

Define 17 canonical stations:

[
x_{c,j}
=======

\alpha_c+\frac{j+\frac12}{17}
(\beta_c-\alpha_c),
\qquad j=0,\ldots,16.
]

### Scale pass

For every source point and candidate, compute

[
t_{ic}=d_c\cdot(p_i-\alpha_c).
]

Points with (t_{ic}\notin[0,L_c]) do not participate.

Otherwise assign the point to exactly one station:

[
j(i,c)=
\min!\left(
16,\left\lfloor\frac{17t_{ic}}{L_c}\right\rfloor
\right).
]

For every ((c,j)), retain the eight assigned source records having the smallest tuple

[
\left(
\lVert p_i-x_{c,j}\rVert^2,,
\operatorname{canonical_source_ordinal}(i)
\right).
]

If fewer than eight records are assigned, that station scale is invalid. Otherwise,

[
s_{c,j}=
\operatorname{median}*{i\in N*{8}(c,j)} h_i.
]

If fewer than nine of the 17 station scales are valid, the candidate immediately becomes `ABSTAIN_KEEP`.

Otherwise,

[
\widehat h_c
============

\operatorname{median}*{j:,s*{c,j}\ {\rm valid}}s_{c,j},
]

[
h_c=
\operatorname{clip}_{[0.5h_0,,2h_0]}
(\widehat h_c).
]

No source point or neighbor identity is persisted.

---

## 5. Fixed Euclidean measurement domain

For a scale-valid candidate, define

[
A_c=\min!\left(3h_c,\frac{L_c}{34}\right).
]

In a second canonical source-by-candidate pass, use the same station assignment (j(i,c)). Put

[
\tau_{ic}=d_c\cdot(p_i-x_{c,j}),
]

[
q_{ic}=p_i-x_{c,j}-\tau_{ic}d_c,
\qquad
r_{ic}=\lVert q_{ic}\rVert.
]

The point contributes only when

[
|\tau_{ic}|<A_c
]

and

[
r_{ic}\le \rho_{13},
\qquad
\rho_k=4h_c\left(\frac54\right)^k,
\quad k=0,\ldots,13.
]

Thus

[
\rho_{13}
=========

4h_c\left(\frac54\right)^{13}
\approx72.759576,h_c.
]

Each accepted point contributes to exactly one station and exactly one radial bin.

### Radial bins

Core rings:

[
[0,h_c),\qquad[h_c,2h_c),\qquad[2h_c,4h_c).
]

Thirteen analysis shells:

[
[\rho_k,\rho_{k+1}),
\qquad k=0,\ldots,11,
]

with the final shell

[
[\rho_{12},\rho_{13}]
]

closed on the upper boundary.

### Density and axial weight

[
\omega_{ic}
===========

\operatorname{clip}*{[1/4,,4]}
\left(\frac{h_i}{h_c}\right)^2
\left[
1-\left(\frac{\tau*{ic}}{A_c}\right)^2
\right].
]

The local-spacing factor approximately area-equalizes nonuniform sampling; the clip prevents one sparse record from dominating.

Each radial bin also has four equal axial sub-bins with normalized boundaries

[
-1,\ -\frac12,\ 0,\ \frac12,\ 1
]

in (\tau/A_c).

No point list, image, graph, component, or local submesh is constructed.

---

## 6. Per-shell moments

For every station (j) and analysis shell (k), let

[
u_i=\frac{q_{ic}}{r_{ic}}.
]

Accumulate:

[
W=\sum_i\omega_i,\qquad
Q=\sum_i\omega_i^2,
]

[
\bar r=\frac1W\sum_i\omega_i r_i,
]

[
\mu=\frac1W\sum_i\omega_i u_i,
]

[
S=\frac1W\sum_i\omega_i u_i u_i^{\mathsf T}.
]

The effective sample count is

[
n_{\rm eff}=\frac{W^2}{Q}.
]

The axial-coverage count is the number of the four axial bins whose accumulated weight is at least (0.06W).

A shell is **base-usable** iff

[
n_{\rm eff}\ge12
]

and axial coverage is at least 3.

Project the covariance into the candidate-normal plane:

[
P_c=I-d_cd_c^{\mathsf T},
]

[
C=P_c(S-\mu\mu^{\mathsf T})P_c.
]

Let

[
\lambda_1\ge\lambda_2\ge0
]

be its two nontrivial eigenvalues, with unit principal eigenvector (e) for (\lambda_1). Eigenvector sign is immaterial.

Set

[
v=\lambda_1+\lambda_2,
\qquad
\delta=\frac{\lambda_2}{v}.
]

A shell is **two-ray usable** iff it is base-usable and

[
v\ge0.01,\qquad
\delta\le0.12.
]

Project the mean once more:

[
\mu_\perp=P_c\mu.
]

Define

[
b=
\sqrt{
\operatorname{clip}*{[0,1]}
\left(
\lVert\mu*\perp\rVert^2-(\mu_\perp\cdot e)^2
\right)
},
]

[
a=\sqrt{\max(0,1-b^2)},
]

[
\eta
====

\frac12
\left[
1-
\operatorname{clip}*{[0,1]}
\left(
\frac{|\mu*\perp\cdot e|}
{\max(a,2^{-40})}
\right)
\right],
]

[
\theta=2\operatorname{atan2}(a,b).
]

For an ideal weighted mixture of two rays, (\eta) is the smaller branch weight and (\theta) is their opening angle.

A model shell additionally requires

[
\eta\ge0.08.
]

When (b\ge2^{-20}), define

[
f=
\frac{\mu_\perp-(\mu_\perp\cdot e)e}{b},
]

[
u^\pm=bf\pm ae,
]

followed by one normalization of each (u^\pm).

The unordered set ({u^+,u^-}) is invariant to the sign selected for (e).

### One-ray shell

A base-usable shell is one-ray iff

[
\lVert\mu_\perp\rVert\ge0.96,
\qquad
v\le0.06.
]

Its ray direction is

[
g=\frac{\mu_\perp}{\lVert\mu_\perp\rVert}.
]

### Core contact

For each core ring, compute only (W,Q,n_{\rm eff}), and axial coverage.

Station (j) has `CORE_CONTACT` iff:

[
n_{\rm eff}^{[0,h)}\ge4
]

with axial coverage at least 2, and at least one of

[
n_{\rm eff}^{[h,2h)}\ge6
\quad\text{or}\quad
n_{\rm eff}^{[2h,4h)}\ge6
]

also has axial coverage at least 2.

Every positive or negative station certificate requires `CORE_CONTACT`. An offset or unsupported candidate therefore becomes unknown rather than being deleted from absence alone.

---

## 7. Five-shell model windows

There are nine windows:

[
K_s={s,s+1,s+2,s+3,s+4},
\qquad s=0,\ldots,8.
]

All five constituent shells must satisfy the relevant shell predicate.

For each two-ray shell put

[
g_k=2\bar r_k\sin\frac{\theta_k}{2},
]

[
\kappa_k=
\frac{2\cos(\theta_k/2)}{\bar r_k}.
]

Define dimensionless residuals:

[
E_{\rm wedge}
=============

\frac{\max_{k\in K_s}\theta_k-
\min_{k\in K_s}\theta_k}
{6^\circ},
]

[
E_{\rm return}
==============

\frac{
\max_{k\in K_s}g_k/
\min_{k\in K_s}g_k-1
}{0.20},
]

[
E_{\rm roll}
============

\frac{
\max_{k\in K_s}\kappa_k/
\min_{k\in K_s}\kappa_k-1
}{0.20}.
]

A zero or nonfinite denominator makes that residual (+\infty).

For two unordered ray pairs (U={u^+,u^-}) and (V={v^+,v^-}), define

[
D_{\rm pair}(U,V)
=================

\min_{\pi\in S_2}
\max_{\sigma\in{+,-}}
\arccos
\left[
\operatorname{clip}_{[-1,1]}
\left(
u^\sigma\cdot v^{\pi(\sigma)}
\right)
\right].
]

The window pair drift is the maximum adjacent-shell (D_{\rm pair}).

Bisector drift is

[
D_f=
\max_{\text{adjacent }k,\ell}
\arccos
\left[
\operatorname{clip}*{[-1,1]}(f_k\cdot f*\ell)
\right].
]

### 7.1 Wedge certificate

A window passes `WEDGE` iff:

[
\operatorname{median}\theta_k\in[8^\circ,174^\circ],
]

[
E_{\rm wedge}\le1,
]

[
D_{\rm pair}\le6^\circ,
]

and

[
E_{\rm wedge}+0.25
\le
\min(E_{\rm return},E_{\rm roll}).
]

Equality passes.

### 7.2 Return/hem certificate

A window passes `RETURN` iff:

[
\operatorname{median}\theta_k\in[3^\circ,80^\circ],
]

[
\operatorname{median}g_k\in[1.0h_c,24h_c],
]

[
E_{\rm return}\le1,
\qquad
D_f\le6^\circ,
]

[
\theta_{k+1}\le\theta_k+2^\circ
\quad\text{for every adjacent pair},
]

and

[
E_{\rm return}+0.25
\le
\min(E_{\rm wedge},E_{\rm roll}).
]

### 7.3 Smooth-roll certificate

A window passes `ROLL` iff:

[
\operatorname{median}\theta_k\in[20^\circ,175^\circ],
]

[
R_{\rm med}
===========

\frac{1}{\operatorname{median}\kappa_k}
\in[6h_c,512h_c],
]

[
E_{\rm roll}\le1,
\qquad
D_f\le6^\circ,
]

[
\theta_{k+1}\le\theta_k+2^\circ
\quad\text{for every adjacent pair},
]

and

[
E_{\rm roll}+0.25
\le
\min(E_{\rm wedge},E_{\rm return}).
]

### 7.4 Flat certificate

A shell is flat iff it is two-ray usable and

[
\theta\ge176^\circ,
\qquad
\delta\le0.08,
\qquad
\eta\ge0.10.
]

A five-shell window passes `FLAT` iff all five shells are flat,

[
\max\theta-\min\theta\le3^\circ,
]

and adjacent principal directions obey

[
\arccos(|e_k\cdot e_{k+1}|)\le4^\circ.
]

### 7.5 Edge certificate

A five-shell window passes `EDGE` iff all five shells are one-ray and

[
\arccos
\left[
\operatorname{clip}*{[-1,1]}
(g_k\cdot g*{k+1})
\right]
\le4^\circ
]

for every adjacent pair.

A model equality without the required (0.25) winning margin produces no model. No tie-breaking may turn an ambiguous model into a positive or anti-crease certificate.

---

## 8. Station and candidate decisions

### Station class

For each station:

1. `POSITIVE_WEDGE` if any window passes `WEDGE`.
2. Otherwise `POSITIVE_RETURN` if any window passes `RETURN`.
3. Otherwise `ANTI_ROLL` if two adjacent windows pass `ROLL`.
4. Otherwise `ANTI_FLAT` if two adjacent windows pass `FLAT`.
5. Otherwise `ANTI_EDGE` if two adjacent windows pass `EDGE`.
6. Otherwise `UNKNOWN`.

For audit only, the selected window is the smallest starting index (s). It does not affect the candidate decision.

Positive precedence is deliberate: a rounded bend with circular inner annuli and wedge-like outer annuli cannot be deleted as a roll.

Let

[
P_c=
#{\text{positive stations}}.
]

Let (R_c,F_c,E_c) denote the three anti-class counts. Runs are consecutive in canonical station order.

### Candidate comparator

`CERTIFIED_KEEP` iff

[
P_c\ge4
]

and the longest positive run is at least 3.

`SUPPRESS` iff all of the following hold:

[
P_c=0,
]

[
R_c+F_c+E_c\ge13,
]

[
\max(R_c,F_c,E_c)\ge11,
]

and the longest consecutive run of the dominant anti-class is at least 8.

A dominant-class tie is ordered

[
\texttt{ROLL}<\texttt{FLAT}<\texttt{EDGE}
]

only for canonical serialization.

Every other candidate is `ABSTAIN_KEEP`.

Consequences:

* one positive station is already sufficient to prevent deletion;
* a low-density, partial, occluded, compound, or locally ambiguous candidate remains;
* deletion requires a coherent anti-crease explanation over most of the segment;
* nearby candidates never compete;
* mirrored and opposing folds have identical predicates;
* finite endpoints do not need all 17 stations;
* an exact duplicate receives the same independent class and remains under unchanged V52 ownership.

---

## 9. Intrinsic work and memory bounds

The implementation performs exactly two topology-free source-by-candidate passes.

| Quantity                                        |    Frozen cap |
| ----------------------------------------------- | ------------: |
| Source records                                  |       200,000 |
| Candidates per scan                             |            64 |
| Stations per candidate                          |            17 |
| Core rings                                      |             3 |
| Analysis shells                                 |            13 |
| Model windows per station                       |             9 |
| Scale-pass point–candidate tests                |    12,800,000 |
| Profile-pass point–candidate tests              |    12,800,000 |
| Total point–candidate tests                     |    25,600,000 |
| Top-eight slots                                 |         8,704 |
| Moment bins                                     |        17,408 |
| Symmetric eigensolves                           |        14,144 |
| Window evaluations                              |         9,792 |
| Graph vertices visited                          |             0 |
| Graph edges visited                             |             0 |
| Mesh adjacency reads                            |             0 |
| Persisted source coordinates or point IDs       |             0 |
| CASL scratch, excluding read-only source arrays |        64 MiB |
| Canonical result artifact                       | 262,144 bytes |
| Peak process RSS                                |     (<16) GiB |
| Per-scan wall time                              |    (\le300) s |

The total work bound is known from (N) and (m); it cannot expand because a scan has dense connectivity, holes, nonmanifold triangles, or a long geodesic closure.

---

## 10. Atomic aborts and fail-closed behavior

The following are whole-scan atomic aborts:

* `SOURCE_POINT_CAP`;
* `CANDIDATE_CAP`;
* `NONFINITE_INPUT`;
* `SOURCE_SCALE_CONTRACT` for missing, nonfinite, or nonpositive (h_i) or (h_0);
* `ZERO_LENGTH_CANDIDATE`;
* `POINT_CANDIDATE_TEST_CAP`;
* `SCRATCH_CAP`;
* `MOMENT_NONFINITE`;
* `MOMENT_PSD_DOMAIN`, when an eigenvalue is below (-2^{-40}) after symmetrization;
* `RESULT_BYTES_CAP`;
* `CANONICAL_ENCODING_FAILURE`;
* `ATOMIC_COMMIT_FAILURE`.

Eigenvalues in

[
[-2^{-40},0)
]

are set to zero. Larger violations abort.

Candidate-local insufficiency, including fewer than nine valid station scales, sparse shells, absent core contact, eigendegeneracy, or no winning model, is never a scan abort. It produces `ABSTAIN_KEEP`.

All candidate results remain in memory until the entire scan validates. The final artifact is encoded to a temporary file, size-checked, hashed, flushed, and atomically renamed. No partial candidate mask is publishable.

An append-only invocation or abort record may be committed separately, but it may contain no candidate result on a failed scan.

---

## 11. Compact audit artifact

The canonical artifact contains only:

* architecture/version and constants hash;
* source-state, candidate-stream, executable, and lock hashes;
* (N), candidate count, and fixed work counters;
* for each (H_c):

  * (h_c);
  * 17 one-byte station classes;
  * selected model/window code for non-unknown stations;
  * (P_c,R_c,F_c,E_c);
  * longest runs;
  * final decision;
* total suppression count;
* result SHA-256.

It contains no source coordinate, source index, neighbor identity, shell membership list, point array, triangle, normal, or rendered judgment.

---

## 12. Source-free lock suite

The suite uses only newly specified analytic primitives. It may not use the quarantined procedural morphology, any real scan, or any prior output. Constants are locked before execution.

### Common generator

Unless overridden:

[
h_i=h_0=1,
]

[
c=((0,0,0),(0,0,170)),
]

with axial samples

[
z_m=\frac m2,\qquad m=0,\ldots,340.
]

Profile parameters use half-unit spacing through radius (75). The profile is extruded along (z). Exact duplicate coordinates are unioned once unless a multiplicity fixture explicitly requires them.

The ideal profiles are:

**Wedge**

[
\gamma_\pm(r)
=============

r\left(
\cos\frac{\theta}{2},
\ \pm\sin\frac{\theta}{2}
\right).
]

**Rounded bend**

[
\phi_\pm(r)
===========

\pm\left[
\frac{\theta}{2}
+
\frac{\pi-\theta}{2}
e^{-(r/\rho)^2}
\right],
]

[
\gamma_\pm(r)=r(\cos\phi_\pm(r),\sin\phi_\pm(r)).
]

**Tangent circular roll**

[
\gamma_\pm(r)
=============

\left(
\pm r\sqrt{1-\frac{r^2}{4R^2}},
\frac{r^2}{2R}
\right),
\qquad r\le2R.
]

**Return/hem**, gap (g):

tails

[
\gamma_\pm(x)=(x,\pm g/2),
\qquad x\ge g/2,
]

joined by the left semicircle

[
\gamma(\alpha)
==============

\left(
\frac g2+\frac g2\cos\alpha,
\frac g2\sin\alpha
\right),
\qquad
\alpha\in[\pi/2,3\pi/2].
]

### Mandatory geometric fixtures

| ID  | Geometry                                              | Required result                      |
| --- | ----------------------------------------------------- | ------------------------------------ |
| G01 | (90^\circ) sharp wedge                                | `CERTIFIED_KEEP`                     |
| G02 | (160^\circ) shallow wedge                             | `CERTIFIED_KEEP`                     |
| G03 | rounded (90^\circ,\rho=8)                             | `CERTIFIED_KEEP`                     |
| G04 | rounded (135^\circ,\rho=16)                           | `CERTIFIED_KEEP`                     |
| G05 | return/hem (g=6)                                      | `CERTIFIED_KEEP`                     |
| G06 | reflected/opposing G03                                | same class and station counts as G03 |
| G07 | G03 present only for (z\in[55,115])                   | `CERTIFIED_KEEP`                     |
| G08 | G03 ending at (z=145) and beginning at (z=10)         | not `SUPPRESS`                       |
| G09 | deterministic 25% alternating branch dropout          | not `SUPPRESS`                       |
| G10 | two G03 bends centered 24 units apart, two candidates | neither candidate suppressed         |
| G11 | full flat profile, (\theta=180^\circ)                 | `SUPPRESS`, dominant `FLAT`          |
| G12 | one-ray/free-edge profile                             | `SUPPRESS`, dominant `EDGE`          |
| G13 | tangent roll (R=20)                                   | `SUPPRESS`, dominant `ROLL`          |
| G14 | tangent roll (R=80)                                   | `SUPPRESS`, dominant `ROLL`          |
| G15 | four-ray crossing ({\pm x,\pm y})                     | `ABSTAIN_KEEP`                       |
| G16 | two unconnected full parallel lines, gap 6            | `ABSTAIN_KEEP`                       |
| G17 | G03 with candidate offset by (4h_0)                   | `ABSTAIN_KEEP`, no positive station  |

### Mandatory comparator fixtures

* wedge lower boundary exactly (8^\circ) passes; its `nextafter` below does not;
* wedge upper boundary exactly (174^\circ) passes; its `nextafter` above does not;
* flat boundary exactly (176^\circ) passes;
* model spread ratio exactly (1.20) passes; its `nextafter` above fails;
* direction drift exactly (6^\circ) passes; its `nextafter` above fails;
* (\eta=0.08) passes the model-shell predicate; its `nextafter` below fails;
* (n_{\rm eff}=12) passes; its `nextafter` below fails;
* synthetic station-class vectors exercise the candidate comparator exactly at (P=0), anti total (=13), dominant count (=11), dominant run (=8), and independently one unit below every boundary.

### Mandatory metamorphic fixtures

* arbitrary source permutation;
* arbitrary candidate permutation;
* endpoint reversal;
* reflection;
* signed coordinate-axis permutation;
* translation by ((1024,-512,256));
* similarity scale (0.8), scaling coordinates and all (h)-values together;
* similarity scale (1.25);
* exact twofold profile refinement with correspondingly area-adjusted local spacings;
* arbitrary mesh adjacency, triangle order, and winding supplied alongside the source but ignored.

Every metamorphic pair must have identical decisions and station classes. Source permutation and endpoint reversal must produce byte-identical scientific artifacts.

### Mandatory cap and atomicity fixtures

* (N=200{,}001) gives `SOURCE_POINT_CAP`;
* (m=65) gives `CANDIDATE_CAP`;
* NaN and infinity give `NONFINITE_INPUT`;
* zero or missing (h_i) gives `SOURCE_SCALE_CONTRACT`;
* zero-length candidate gives `ZERO_LENGTH_CANDIDATE`;
* (N=200{,}000,m=64) stress case completes with exactly 25,600,000 point–candidate tests and zero graph counters;
* a forced 262,145-byte result gives `RESULT_BYTES_CAP` with no partial result;
* injected pre-rename write failure gives `ATOMIC_COMMIT_FAILURE` with no final result.

**Lock gate:** all 43 fixtures pass in one frozen run; suite replay is byte-identical; no threshold or formula may change afterward.

### Positive-fixture compatibility

The required positives are not blocked by simultaneous contradictory predicates:

* G01/G02 have rank-one two-ray covariance and constant (\theta), so (E_{\rm wedge}=0).
* G03/G04 may exhibit a circular inner window, but their outer window has constant (\theta); positive precedence prevents a roll deletion.
* G05 has constant (2r\sin(\theta/2)=g), so the return residual is zero in the ideal fixture.
* G07 supplies at least four positive stations with a run of at least three.
* G08/G09/G10 may become partially unknown, but deletion remains impossible unless there are zero positive stations plus 13 coherent anti stations.

---

## 13. U01–U04 mechanical gate

After the source-free lock, execute exactly one primary and one replay on each already exposed scan with the same executable, constants, source state, and candidate bytes.

Mandatory completion counts:

| Scan      | Required candidate decisions | Maximum suppressions |
| --------- | ---------------------------: | -------------------: |
| U01       |                           13 |                    4 |
| U02       |                           20 |                    6 |
| U03       |                            3 |                    1 |
| U04       |                           12 |                    4 |
| **Total** |                       **48** |               **15** |

The tranche passes mechanically only if all of the following hold:

1. All four primary runs and all four replays complete without an atomic abort.
2. All 48 candidate hashes receive exactly one decision.
3. Primary and replay scientific artifacts are byte-identical for every scan.
4. Every work counter is within the frozen caps; all graph, adjacency, Q0, closure, and edge-visit counters are exactly zero or absent.
5. At least 36 of 48 candidates reach at least one five-shell model-window evaluation; an early scale abstention is still a complete candidate decision but does not count here.
6. Total suppressions satisfy

[
1\le S_{\rm total}\le15.
]

7. The per-scan suppression ceilings in the table are respected.
8. Every suppression independently satisfies the frozen (0/13/11/8) candidate comparator.
9. The final result is a hash-subset of the V52 baseline; no endpoint or marker coordinate changes.
10. Peak RSS is below 16 GiB and every run is at most 300 seconds.
11. Every canonical result is at most 262,144 bytes and contains no source-point material.
12. Gold, labels, expected counts, CAD, identity, filenames, benchmark outcomes, and the reserved physical scan remain unopened.

Zero suppression, more than 15 suppressions, any incomplete scan, or failure to reach model evaluation on 36 candidates retires CASL immediately. No constant edit or rerun is authorized.

---

## 14. Native-resolution visual gate

Only after the mechanical artifacts and hashes are frozen:

### Blind first pass

For all 48 baseline candidates, the reviewer first sees:

* undecimated native source geometry;
* the unchanged V52 marker;
* no CASL decision;
* no station class;
* no anti-model reason.

For every marker, record exactly one blind judgment:

* `VISIBLE_PHYSICAL_CREASE`;
* `VISIBLE_NONCREASE`;
* `AMBIGUOUS_OR_OCCLUDED`.

Use the full native view plus three candidate-centered views:

1. view along the candidate direction;
2. first perpendicular side view;
3. orthogonal perpendicular side view.

No smoothing, point deletion, mesh repair, or camera-dependent point resampling is allowed.

### Reveal pass

After the blind ledger is committed, reveal the CASL mask and reasons.

The visual gate passes only if:

1. Every suppressed candidate was blind-labeled `VISIBLE_NONCREASE`.
2. A suppressed candidate labeled `VISIBLE_PHYSICAL_CREASE` or `AMBIGUOUS_OR_OCCLUDED` is an immediate failure.
3. At least one suppression visibly tracks a flat-face streak, free/perimeter edge, dropout boundary, or distributed smooth roll for at least 75% of its marker length.
4. Every V52 marker blind-labeled `VISIBLE_PHYSICAL_CREASE` remains, unless another surviving V52 marker covers at least 90% of the same visible crease span.
5. Every observed rounded, partial, finite-endpoint, return/hem, opposing, and nearby-distinct bend represented by V52 retains at least one marker.
6. No marker has moved or become render-inadmissible.
7. All four scans—not only changed crops—are inspected at native resolution.

Any failure retires V72 with V52 retained. Visual findings may not alter a threshold, station rule, shell radius, or model comparator.

---

## 15. Conditional reserved-scan execution

The reserved physical scan remains sealed unless both the U01–U04 mechanical gate and the blind native-resolution visual gate pass unchanged.

After both pass, the same locked executable and constants are authorized for exactly one primary and one replay on the reserved scan. No new fixture, calibration, threshold, or architecture edit may intervene.

The reserved run must:

* complete twice without abort;
* replay byte-identically;
* remain below 16 GiB and 300 seconds;
* suppress no more than (\lfloor m/3\rfloor) candidates;
* produce only a subset of the V52 markers;
* undergo the same blind native-resolution review before any truth or benchmark information is opened.

A failure is terminal. There is no replacement scan or rescue run.

---

## 16. Why this is the bounded successor

V71 did not establish that source geometry lacks the information needed to distinguish bends from false candidates. It established that conservative topology closure can consume its graph and seed domains before producing any candidate decision. CASL removes that dependency completely.

The literature supports using bounded local covariance and multiscale integral summaries for robust point-cloud feature analysis. Pauly, Keiser, and Gross used local covariance across neighborhood scales to identify persistent line-type features directly on unstructured point samples; their later graph construction is not used here. Yang, Lai, Hu, and Pottmann showed that integrated PCA quantities over bounded neighborhoods provide scale-dependent curvature information with improved noise behavior. Mérigot, Ovsjanikov, and Guibas showed that integral covariance measures can encode sharp-feature directions and angles robustly in point-cloud data. ([CGL ETHZ][1])

The three annular secant laws, their model competition, all numerical thresholds, the station/candidate persistence rule, and the conservative anti-certificate policy are Sherman-specific engineering hypotheses. They are not claimed by those papers. This one execution falsifies the hypothesis directly: CASL must engage the real candidates, delete some but not many, and survive blind native-resolution inspection without sacrificing the difficult bend morphologies.

[1]: https://cgl.ethz.ch/Downloads/Publications/Papers/2003/Pau03b/Pau03b.pdf "https://cgl.ethz.ch/Downloads/Publications/Papers/2003/Pau03b/Pau03b.pdf"
