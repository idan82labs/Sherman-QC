# Noisy plane registration: working design, not a frozen experiment

Date: 2026-09-09. Source/mathematical research only. No new synthetic/native or
physical execution, scanner calibration, runtime promotion or budget authorization
is reported here. The separately frozen exact-mesh V4 connected gate has passed
its three constructed cases; that result does not qualify noisy extraction.

## Why an additive mechanism is needed

V4 deliberately uses near-roundoff coplanarity. Increasing its thresholds would
not establish noisy plane support or preserve its exact curved-chain controls.
Also, its twelve-scan-plane/twenty-four-CAD-plane hypothesis caps are not a
general large-CAD search strategy. A successor needs explicit support generation
and correspondence work accounting, not a largest-area/top-N prefix or simply
larger Cartesian-product enumeration. Preserve every V4 terminal result.

Independent design by `/root/noise_plane_math`: predeclare overlapping original-
mesh topology tiles at two scales. Fit each complete core from original vertices;
retain qualified cores independently of any grown closure. Grow with a fixed
seed uncertainty cone on original shared edges, then validate every original row
of the complete closure. A failed closure cannot erase its predeclared valid core.
This is not post-failure residual trimming. Retain triangle unions, holes,
disconnections and raw identities; no smoothed coordinate replaces a measurement.

This takes neighborhood/whole-region fitting context from
[CGAL shape detection](https://doc.cgal.org/latest/Shape_detection/index.html),
not its single-region ownership. [Polylidar](https://arxiv.org/abs/2002.01886)
provides relevant hole-preserving mesh-boundary work, not scanner metrology or a
performance claim for this implementation. These design references need a final
implementation-specific review before a protocol is frozen.

## Conditional bounded-error mathematics

Require a caller-stipulated simultaneous position bound epsilon_i for each row.
Residual RMS, native tolerance, manufacturing tolerance and point density do not
provide that bound. No physical epsilon has been established for Gold11.

Let observed seed centroid be c, population covariance C, and eigenvalues
lambda_1 <= lambda_2 <= lambda_3. Let e_bar = mean(epsilon_i) and
B^2 = mean((epsilon_i + e_bar)^2). If unknown true points all lie on a single
plane and each differs from its observation by at most epsilon_i, then any
consistent true unit normal n obeys n^T C n <= B^2. This follows by centering
both point sets: each centered normal error is bounded by epsilon_i + e_bar.
For a positive smallest eigengap, decomposition in C's eigenbasis gives

    sin(alpha)^2 <= (B^2 - lambda_1) / (lambda_2 - lambda_1).

An absent gap or bound >=1 leaves orientation unresolved; a negative numerator
cannot be silently clipped into a favorable certificate near numerical error.
There is no 1/sqrt(N) improvement under arbitrary simultaneous bounded errors.
Independent mathematical review also derives the tighter optional bound
n^T C n = Var(n . e_i) <= mean(epsilon_i^2), provided covariance and error
moments use the same declared row weights. The initial implementation retains
the conservative bound above; any change must be prospectively documented.

For a candidate additional row p_j, a necessary relaxation is

    min_{angle(n,n_0)<=alpha} |n . (p_j-c)| <= epsilon_j + e_bar.

Individually compatible rows need not share a jointly feasible plane. Full-
support validation is mandatory and cannot claim completeness from a TLS fit
alone. Shallow curvature entirely within the declared point balls is physically
non-identifiable under this model; compatibility is not proof of flatness.

For three uncertain normals, define eta^2 = sum_i (2 sin(alpha_i/2))^2.
If sigma_min(N_0) <= eta, do not publish an intersection error bound. Otherwise,
with proposed intersection x_0 and seed centroids c_i, the candidate conditional
bound is ||x_* - x_0|| <= ||rho|| / (sigma_min(N_0)-eta), where
rho_i = |n_0i . (x_0-c_i)| + 2 sin(alpha_i/2) ||x_0-c_i|| + e_bar_i.
The first term retains the nominal solve residual. Independent review by
`/root/astra_repo_guidance_review` caught its omission before implementation:
with zero noise/cone angle, N_0=I, c_i=0 and a proposed x_0=(1,0,0), omitting it
incorrectly promises zero error although the true intersection is the origin.
Only an exact nominal intersection makes that term zero. A numerical solve must
not assume exact cancellation. This bound's origin-independent form is important.
These derived bounds still require implementation and floating-point enclosure
review; none is calibrated confidence.

For a fixed proposed orientation and anchor a, a simultaneous heteroscedastic
plane witness exists exactly when all offset intervals
[n . (p_i-a)-epsilon_i, n . (p_i-a)+epsilon_i] have nonempty intersection.
That proves one fixed-normal witness, not feasibility over every orientation.
Keep the observed centroid used in cone membership distinct from the chosen
feasible-strip offset. Numerical interval overlap also needs explicit resolution
handling; an RMS fit alone is insufficient.

Two independently constructed counterexamples must remain explicit controls:

- Observations (+/-1,0,0), (0,+/-1,0), (1,0,1), with respective error bounds
  (0,0,0,0,1), admit plane z=0. Unweighted TLS tilts because of nonzero x-z
  covariance and cannot supply a feasible fixed-normal offset. Failure of that
  witness search means WITNESS_NOT_FOUND, not infeasible geometry.
- Seed corners (+/-10,+/-10,0), each with epsilon=1, have C=diag(100,100,0),
  B^2=4 and alpha=asin(0.2). Additional exact points (0,0,+/-1), epsilon=0,
  each pass the necessary cone relaxation, but any plane through both has
  n_z=0 and cannot fit the seed within its narrow cone. Rowwise compatibility
  is not simultaneous feasibility.

Floating-point covariance/eigenvector/interval uncertainty must be handled
separately from the stipulated physical error balls. It cannot silently widen
epsilon or turn an unresolved boundary into a geometric witness.

## Correspondence research and representation caution

[GraffMatch](https://arxiv.org/html/2212.12745v1) uses pairwise consistency graphs
for line/plane associations and then estimates pose. Its Appendix A invariance
argument transports affine anchors as b' = Rb+t before shifting; graph selection
uses a relaxation, not a guarantee that every ambiguous correspondence is found.
The reported LiDAR results do not establish sheet-metal or metrology performance.

Root's derived implementation caution: independently canonicalizing anchors as
closest points to each frame's origin does not generally satisfy b' = Rb+t.
For example, planes x=0 and y=0 both have zero closest-point anchors. Translate
both by (1,1,0): their new canonical anchors are (1,0,0) and (0,1,0), neither the
transported old anchor (1,1,0). Shifting by the first canonical anchor leaves
x=0 and y=1; it does not recover the original pair's common intersection anchor.
This orthogonal example does not change the lifted Grassmann angle. A stronger
independent control uses x=0 and x+y=0 translated by (1,1,0): independently
canonicalizing and shifting by the first anchor gives x=0 and x+y=1. The lifted
hyperplane-normal angle changes from pi/4 to acos(1/sqrt(3)). This demonstrates
metric change under the proposed naive adaptation, not under transported anchors.
Likewise partial-patch centroids need not correspond. Do not import a shifted
Grassmannian invariant with unverified anchor semantics. This is a warning about
the proposed adaptation, not a claim to have invalidated the paper's theorem.

Unoriented normal-pair angle intervals are simpler rigid invariants, but repeated
orthogonal/parallel planes make them weak. A scalable successor must retain that
ambiguity and finite-face verification, rather than equating graph consistency
with a unique pose. The selected additive implementation seam is a topology-tiled
core/closure proposer followed by angle-indexed, explicitly bounded correspondence
scheduling and the existing complete-row finite-surface verification. Scheduling
quotas may be prospectively incomplete; retain unscheduled witness identities
and distinguish all-scheduled-evaluations-complete from exhaustive search. An
unexpected hard-cap failure must still whole-refuse usable output. No search
implementation or physical protocol is frozen by these working notes.

## Prospective adversarial scope, not execution approval

The independent proposal is twelve geometries x three declared noise fields x
three proper rigid frames =108 cases. Include partial/repeated/near-dependent
planes; remote tiny caps and tile-boundary caps; holes/disconnected tabs/duplicate
coordinates with distinct rows; cylinder-to-plane transition, curved chain,
indistinguishable shallow bow; a beyond-bound contaminant and heteroscedastic
strip. Noise fields are zero, deterministic alternating bounded displacement,
and spatially correlated heterogeneous displacement.

Score required support/core recovery, alternative proposals, all raw denominators,
holes, correct/refused poses and every overflow. A finite tiling cannot guarantee
arbitrarily small patches. Explicit tile/row-visit/membership/closure/report caps
and full-overflow refusal must be set before the small60/45-second1-GiB tier;
large-scan qualification needs its own later audit. Do not run this matrix from
these incomplete design notes.
