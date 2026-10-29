RULING: EXECUTE_ONE

## 1. Intervention identity

**Name:** `V54_EXCLUSIVE_NORMAL_TURN_COVERAGE_V1`
**Short name:** `ENTC_V1`
**Precommit token:** `V54_ENTC_MECHANISM_LOCKED`
**Terminal-failure token:** `V54_ENTC_TERMINAL_RETIRE`

ENTC is the final bounded test of whether the presently represented-but-unselected bends can be recovered by **global source-evidence ownership**, without requiring a second generator route, same-bank route agreement, or a scan-global minimum line span.

## 2. Insertion point and mutation boundary

Insert ENTC **after V52 has produced its final ordered output and while the frozen candidate pool and source-support records remain in memory, but before public-result serialization**.

The intervention is **append-only**:

* Every V52 line, byte order, geometry, score, and metadata field remains an immutable prefix.
* No V52 line may be moved, shortened, replaced, rescored, or removed.
* The frozen candidate pool is not modified and no new candidate geometry is born.
* V53 is neither imported nor invoked.
* Ordinary NMS remains exactly as in V52 and is not called again. The exact frozen V52 NMS conflict predicate is used only as a veto against appending a duplicate.
* Every ENTC line is appended after the complete V52 prefix.

The 27 inherited false positives are therefore outside this intervention. Pruning them without a validated negative certificate would combine two causal hypotheses and place the 52 established V52 true positives at unnecessary risk.

## 3. Complete deterministic algorithm

### 3.1 Source mesh preparation

For each scan, let the source triangle mesh be (M=(V,F)).

1. Merge duplicate undirected mesh edges by sorted source vertex IDs.
2. Let (s) be the median positive finite length of edges having exactly two incident nondegenerate faces.
3. A geometric edge is eligible for normal-turn estimation only when:

   * it has exactly two incident nondegenerate faces; and
   * (0.25s \leq \ell_e \leq 4s).
4. Mark as `BOUNDARY_NEAR` every vertex within two mesh-edge hops of any edge whose incident-face count is not exactly two.
5. For every vertex (v), construct the orientation-free normal dyad

[
D_v=\sum_{f\ni v} A_f,n_fn_f^\mathsf{T},
]

where (A_f) and (n_f) are the face area and unit face normal. The vertex normal is the principal eigenvector of (D_v). The sign is canonicalized by making the first component whose absolute value exceeds (10^{-15}) positive. Use a fixed analytic symmetric-(3\times3) eigensolver; BLAS-dependent eigensolvers are forbidden.

For an eligible edge (e=(v,w)),

[
\delta_e=\arccos!\left(\operatorname{clamp}(|n_v^\mathsf{T}n_w|,0,1)\right),
\qquad
\kappa_e=\delta_e/\ell_e.
]

Calculate the scan-wide excess-turn floor

[
\kappa_0=
\operatorname{median}(\kappa_e)
+3(1.4826)\operatorname{median}
\left|\kappa_e-\operatorname{median}(\kappa_e)\right|.
]

The nonnegative source-only turn mass is

[
h_e=\ell_e\max(0,\delta_e-\kappa_0\ell_e).
]

Edges with (h_e=0) contribute nothing but remain counted for structural audits.

All medians use stable sorting by `(numeric value, edge_id)`. All reductions use ascending edge ID and Neumaier-compensated float64 summation.

### 3.2 Candidate-local source region

For candidate (c):

1. Let ([a_c,b_c]) be its frozen finite line segment, (L_c=|b_c-a_c|), and (u_c=(b_c-a_c)/L_c).
2. Canonicalize the unoriented axis by reversing (u_c,a_c,b_c) when the first component of (u_c) with absolute value above (10^{-15}) is negative.
3. Let (U_c) be the union of all scan-local source vertex IDs in every frozen support bank carried by the candidate.

   * Route names, route counts, bank pairing, and repeated membership are ignored.
   * A vertex appearing in several banks appears once.
4. Let (R_c) be (U_c) plus its one-ring mesh neighbors.
5. Candidate edge set (E_c) contains an eligible source edge when:

   * both endpoints are in (R_c);
   * at least one endpoint is in (U_c);
   * (h_e>0); and
   * its midpoint projection (t_e=(m_e-a_c)^\mathsf{T}u_c) satisfies (0\leq t_e\leq L_c).

For (e\in E_c), with unit edge direction (q_e), define candidate compatibility

[
w_{ce}=h_e\left[1-(q_e^\mathsf{T}u_c)^2\right].
]

This rewards source-normal change occurring transversely to the proposed bend direction. It does not fit a cylinder, mate two skins, or use route agreement.

### 3.3 Axial profile

Divide the complete frozen segment ([0,L_c]) into exactly **24 equal closed-open bins**, except that (t=L_c) belongs to bin 23.

For each bin (k), calculate

[
H_{ck}=\sum h_e,\qquad
W_{ck}=\sum w_{ce},\qquad
n^{\mathrm{eff}}*{ck}
=\frac{H*{ck}^2}{\sum h_e^2}.
]

A bin is **coherent** exactly when:

* (H_{ck}>0);
* (n^{\mathrm{eff}}_{ck}\geq3); and
* (W_{ck}/H_{ck}\geq0.75).

The total effective edge count is

[
n^{\mathrm{eff}}*c=
\frac{\left(\sum*{e\in E_c}h_e\right)^2}
{\sum_{e\in E_c}h_e^2}.
]

Boundary contamination is

[
\beta_c=
\frac{\sum_{e\in E_c:\ e\text{ touches BOUNDARY_NEAR}}w_{ce}}
{\sum_{e\in E_c}w_{ce}}.
]

### 3.4 Global exclusive ownership

Let (S) initially contain the immutable V52 lines, mapped to their frozen originating candidate records.

For each source edge,

[
b_e(S)=\max_{j\in S}w_{je},
]

with zero used when no selected candidate supports that edge.

For an unselected candidate,

[
g_{ce}(S)=\max(0,w_{ce}-b_e(S)).
]

Its marginal coverage is

[
\Delta_c(S)=\sum_e g_{ce}(S),
\qquad
\rho_c(S)=
\frac{\Delta_c(S)}{\sum_e w_{ce}}.
]

For axial bin (k),

[
G_{ck}(S)=\sum_{e\text{ in bin }k}g_{ce}(S).
]

A coherent bin is **novel** when

[
G_{ck}(S)/W_{ck}\geq0.60.
]

An unselected candidate is admissible only if all of the following hold:

1. (L_c\geq24s).
2. (|U_c|\geq64).
3. (n^{\mathrm{eff}}_c\geq48).
4. (\beta_c\leq0.25).
5. At least **16 of 24** bins are coherent.
6. At least **12 of 24** bins are novel.
7. Some contiguous 10-bin window contains at least **9 novel bins**.
8. (\rho_c(S)\geq0.55).
9. The weighted 5th-to-95th percentile span of (t_e), using (g_{ce}(S)) as weights, is at least (0.75L_c).
10. The frozen V52 NMS predicate reports no conflict with any current member of (S).

This is candidate-relative continuity; there is no (0.50B), or any other bounding-box-relative minimum span.

### 3.5 Greedy selection

At each iteration:

1. Recompute admissibility and marginal coverage against the current (S).
2. Select the admissible candidate with the largest tuple, in this exact order:

[
\left(
\Delta_c,\
N^{\mathrm{novel}}_c,\
N^{\mathrm{coherent}}_c,\
R^{\mathrm{one\mbox{-}hole}}_c,\
\rho_c,\
L_c/s
\right),
]

all descending.

3. Break a complete numeric tie by ascending canonical candidate SHA-256.
4. Append that line to the result.
5. Update (b_e(S)) and repeat.

Stop when no candidate is admissible.

At most **16 additions per scan** are permitted. If a seventeenth candidate would remain admissible, do not truncate: emit unchanged V52 for that scan with `ABSTAIN_RUNAWAY_MODEL_COUNT`, and fail the entire V54 run.

The coverage objective

[
F(S)=\sum_e\max_{c\in S}w_{ce}
]

has diminishing marginal gain by construction; the greedy step selects the largest remaining source-explanation increment rather than the next candidate in generator order.

## 4. Candidate population and deterministic ordering

The population is every candidate in the exact frozen pool, subject only to these preprocessing rules:

1. Candidates having exactly identical canonical endpoint float64 bytes are merged. Their source-bank vertex sets are unioned.
2. Any candidate already represented by a V52 output is placed in the initial selected set.
3. Every other candidate remains eligible; no top-(K), generator-score cutoff, route-name filter, family filter, or length ranking is allowed.
4. Candidate canonical bytes are:

   * six little-endian float64 endpoint coordinates;
   * canonical axis sign and endpoint order;
   * all `-0.0` replaced by `+0.0`;
   * NaN or infinity forbidden.
5. SHA-256 of those bytes is the final tie-break key.

Candidate preprocessing is performed in ascending SHA-256 order. Candidate calculations may run in parallel, but reductions inside each candidate and the greedy decision are serial and fixed-order.

## 5. Abstention rules

### Scan-level abstention

Return the exact V52 result, with no ENTC additions, when any of these occurs:

* no valid triangular mesh;
* (s) is nonfinite or nonpositive;
* more than 1% of faces are nonfinite or degenerate;
* any V52 output cannot be mapped to its frozen candidate provenance and source banks;
* more than 4,096 exact-merged candidates;
* more than 8,000,000 candidate-edge incidences after one-ring expansion;
* predicted working memory above 12 GiB;
* actual RSS reaches 16 GiB;
* wall time reaches 300 seconds;
* nonfinite arithmetic enters any selection tuple; or
* the 16-addition runaway condition occurs.

**Any scan-level abstention is an immediate terminal failure of the 30-scan engineering run.** It is not counted as a harmless unchanged case.

### Candidate-level abstention

A candidate is silently excluded, without failing the scan, when:

* it lacks resolvable source-bank vertex IDs;
* its geometry is invalid;
* it conflicts with the current selected set under the frozen NMS predicate;
* (|U_c|<64);
* (L_c<24s);
* its edge set is empty;
* its effective support, boundary, coherence, novelty, continuity, or marginal-span conditions fail; or
* any candidate-local quantity is nonfinite.

No relaxed fallback is permitted.

## 6. Immediate and terminal failure gates

Before scoring, V54 fails immediately on any of the following:

* any prohibited truth, CAD, drawing, identity, expected-count, benchmark, runtime/main, or production read;
* any V53 code-path invocation;
* any modification to candidate-pool or V52 bytes;
* any second NMS invocation;
* any non-prefix V52 serialization;
* any scan-level abstention;
* primary/replay disagreement in result bytes, candidate decisions, 24-bin summaries, or reason codes;
* P95 above 300 seconds or peak RSS at or above 16 GiB;
* serialization of source vertex IDs, edge IDs, face IDs, source coordinates, or support bitsets;
* public JSON above 256 KiB per scan, excluding required visual files.

After primary and replay outputs are frozen, score exactly once. Any failed scoring or visual gate retires V54 under `V54_ENTC_TERMINAL_RETIRE`. No V54b, constant change, subset retention, candidate-specific waiver, or rerun is authorized.

## 7. Minimum positive retention gate

Retain V54 only if **all** conditions hold:

* all 52 previously matched V52 true positives remain present and line-identical;
* at least **8 new true positives**;
* **zero new false positives**;
* additions occur in at least **6 scans** and at least **4 topology families**;
* total score is at least:

  * TP (\geq60);
  * FP (=27);
  * FN (\leq40);
  * precision (\geq0.689655);
  * recall (\geq0.600000);
  * F1 (\geq0.641711);
  * exact-count scans (\geq9/30);
* 30/30 primary/replay byte identity;
* zero additions on every scan that was already exact under V52.

Thus a one-case or two-case improvement is a formal failure even when those lines are true.

## 8. Mandatory visual audit

Render, at the existing native audit resolution with no downsampling:

* every changed scan;
* all six V52-exact scans;
* S011, S016, S024, and S029.

Each changed scan must include:

1. the frozen raw-source overview;
2. V52 lines as an immutable layer;
3. ENTC additions as a separate layer;
4. an axis-normal close-up for every addition;
5. source normal-turn intensity without source IDs;
6. the 24-bin coherent/novel profile and marginal-ownership fraction.

### Required positive signature

Every retained addition must visibly:

* coincide with one physical bend axis over its supported extent;
* follow one localized strip of transverse normal change;
* have support reaching both longitudinal portions of the proposed line;
* terminate where the observed bend terminates, intersects another feature, or genuinely loses source support;
* remain distinct from every V52 line; and
* remain physically valid when viewed both along and normal to the proposed axis.

### Automatic visual failures

Fail the run if:

* any required render is absent, clipped, or not native resolution;
* V52-only overlay pixels differ from the frozen V52 render outside the added-line layer;
* an addition’s displayed marginal-turn 5–95% span is below (0.75L_c);
* displayed 24-bin values disagree with the compact numeric record;
* an addition appears on any formerly exact V52 scan; or
* any source IDs or support bitsets appear in the artifact.

### Manual visual failures

Fail on any visible:

* short or floating axis;
* diagonal traversal of a flat panel;
* free rim, crop edge, cap, wall, seam, weld, or reconstruction-facet explanation;
* parallel-offset or partial duplicate of a V52 line;
* support borrowed from two unrelated physical regions;
* line extending materially beyond the observed fold; or
* S016/S029 addition whose apparent evidence is actually attached to an inherited floating or diagonal V52 false positive.

## 9. Leakage audit

ENTC reads only:

* source coordinates;
* source mesh connectivity;
* normals computed from source faces;
* frozen candidate line geometry; and
* scan-local source vertex membership already carried by frozen candidate support records.

It does **not** read route names or count agreeing routes. Source-bank IDs are unioned and used only as scan-local locality references; they are never compared between scans or serialized.

The worker receives no:

* truth, labels, CAD, drawings, nominal dimensions;
* filename, part number, family, A/B identity, scan identity feature;
* expected bend count;
* benchmark score;
* known exact-case indicator; or
* S011/S016/S024/S029 flag.

The mandatory-case list and topology-family breadth gate exist only in the sealed post-inference scorer/auditor. The 16-line cap is a runaway safety bound: reaching it fails the scan rather than imposing an inferred expected count.

## 10. Causal and research basis

The broad recoverable population is not the nine truth bends absent from the candidate pool; ENTC makes no claim about those. It targets the substantially larger set of represented false negatives by asking a different question from V53: **does this candidate contribute a sustained source-normal-turn explanation that the currently retained line set does not already explain?**

V53 required two named generators to agree on geometry and the same banks over an absolute (0.50B) span. ENTC:

* requires no route pair;
* ignores route identity and multiplicity;
* evaluates a candidate relative to its own finite extent;
* uses a one-sided mesh observable;
* makes all candidate models compete for the same source evidence; and
* requires a large marginal contribution before appending.

It therefore attacks selection/ownership rather than simply admitting more candidates. The zero-new-FP gate and the requirement for at least eight gains across four families prevent a narrow count-increase interpretation.

The established ingredients are limited but relevant. Normal-cycle work provides a principled basis for extracting curvature information from polyhedral meshes through normal variation, although ENTC’s edge token and constants are not a direct implementation of that estimator. ([ACM Digital Library][1]) Global geometric multi-model fitting treats source observations as needing joint model ownership rather than independent greedy acceptance, and geometric set-coverage formulations explicitly address overlapping hypotheses and outliers. ([Springer Link][2]) The coverage objective used here is monotone submodular, so its diminishing-return behavior and deterministic greedy consolidation are established optimization machinery. ([Springer Link][3])

The Sherman-specific, falsifiable hypothesis is that true residual bend candidates own a longitudinally sustained, transverse normal-turn trace that is not already explained by V52, while inherited aliases and arbitrary candidate-pool members do not. The 24-bin construction, excess-turn floor, purity and novelty constants, one-hole rule, and retention gates are engineering precommitments—not published universal bend criteria.

[1]: https://dl.acm.org/doi/10.1145/777792.777839 "https://dl.acm.org/doi/10.1145/777792.777839"
[2]: https://link.springer.com/article/10.1007/s11263-011-0474-7 "https://link.springer.com/article/10.1007/s11263-011-0474-7"
[3]: https://link.springer.com/article/10.1007/BF01588971 "https://link.springer.com/article/10.1007/BF01588971"
