# RULING: OPEN_CAPABILITY

**Capability:** `CALIBRATED_RAW_RANGE_VIEW_BUNDLE_V1`
**Architecture:** `V57_CALIBRATED_MULTIVIEW_TURN_CHANGEPOINT_CONSENSUS_V1`
**Short name:** `CMTCC_V1`
**Capability-opening token:** `OPEN_CALIBRATED_RAW_RANGE_VIEW_BUNDLE_V1_ONCE`
**Precommit token:** `V57_CMTCC_MECHANISM_LOCKED`
**Terminal token:** `V57_CMTCC_TERMINAL_RETIRE`

**One-sentence mechanism:** detect localized surface-normal change-points independently in organized, calibrated, pre-fusion range views; back-project them into 3-D; retain only deterministic three-or-more-view line consensus; then byte-preserve, append to, or prune V52 under explicit multiview visibility certificates.

This is the smallest justified opening because it adds **one source-only information channel—acquisition-view organization and calibration—without opening truth, CAD, identity, RGB, expected counts, or learning**. It attacks both defects that V52 cannot solve append-only: new candidate birth for the nine bends absent from the frozen pool and conservative pruning of the 27 inherited false positives. 

I am not ruling that all future mesh-only mathematics is impossible. I am ruling that another candidate-conditioned mesh-normal, turn-mass, section, or ridge estimator is presently too close to closed PWTS/V54/V55/V56 observables to constitute the smallest clean causal test. V54 already showed that longitudinal turn support can certify a transversely displaced flat-panel axis, while V55 was another mesh-section turn-localization architecture even though its geometry was never reached.  The earlier PWTS record also explicitly prohibited reopening that family by inheritance or renaming. 

## 1. Binding rulings on the V56 protocol questions

### V56 primary outputs

**They remain closed throughout V57 design, implementation, source-free testing, preflight, execution, visualization, scoring, and retention.**

They may not be used for:

* visual diagnosis while selecting V57 predicates or constants;
* choosing fixtures;
* estimating candidate populations or expected line counts;
* comparing V57 behavior against V56;
* generating positive or negative examples;
* selecting scans for special treatment.

After V57 has reached a terminal artifact—pass or fail—the V56 primaries may be unsealed for an explicitly retrospective, truth-free protocol postmortem. Such a postmortem may not revise V57, support a V56 accuracy claim, or authorize another CTMLR execution. This preserves the prediction/postdiction boundary rather than using an unscored benchmark traversal as adaptive development evidence. Preregistration is specifically intended to distinguish prospective tests from analyses formulated after observing outcomes, and reproducibility practice favors fully specified workflows that reduce unintentional experimental error. ([PubMed][1])

### Per-scan child-process supervision

**Yes. It is mandatory.**

Every primary and replay record must run in a fresh, separately supervised child process. Only an OS-backed monotonic deadline and memory controller may terminate it. Completion counts, terminal output, dashboards, inferred progress, or operator judgment must have no connection to the kill path.

### Infrastructure-only CTMLR replication

**Forbidden.**

A V57 that changes only timeout supervision, harness code, serialization, process isolation, or orchestration while retaining CTMLR geometry would be a V56 rescue under a new version label.

A scientifically distinct CTMLR replication would require a separately constituted, untouched corpus that had been sealed before any V56 output was opened, would retain the CTMLR mechanism identity, and would be described as an independent replication rather than a successor architecture. That is **not authorized by this ruling**.

Generic infrastructure may be reused only where it has no CTMLR semantics: canonical JSON, recursive Python-scalar conversion, hashing, atomic file publication, forbidden-key validation, and child-process resource supervision.

## 2. Exact capability and input boundary

For each S001–S030 scan, the independent data custodian must supply one sealed bundle containing:

1. The existing frozen finite source geometry used by V52, bounded to the same approximately 200,000-point representation.
2. The frozen V52 semantic prediction bytes.
3. Between 3 and 64 original undistorted organized range frames.
4. For each frame:

   * a depth array in millimetres;
   * a validity mask;
   * image width and height;
   * a finite (3\times3) pinhole intrinsic matrix (K);
   * a finite rigid (4\times4) transform (T_{\text{world}\leftarrow\text{camera}}).
5. A content hash of the acquisition bundle and, for synthetic data, the already-frozen acquisition-manifest and renderer hashes.

Accepted frames must be either:

* original sensor range frames; or
* byte-reproducible outputs of the acquisition simulator and pose/noise manifest that originally generated the frozen scan.

The following are forbidden:

* new viewpoints;
* increased rendering resolution;
* altered noise, dropout, clipping, or meshing settings;
* rerendering chosen after inspecting V56;
* rendering the fused mesh merely to manufacture “raw” views;
* RGB, intensity, texture, part markings, timestamps, filenames, directory names, acquisition job names, or operator annotations;
* truth, CAD, drawings, expected counts, family labels, A/B identity, benchmark feedback;
* the frozen union pool or V53–V56 outputs, caches, code, station data, or visual artifacts.

The child receives only content-addressed arrays and calibration. The external S-number used by orchestration is not passed to inference.

**Capability failure rule:** if an authentic qualifying bundle is unavailable for even one of the 30 scans, return `V57_CMTCC_CAPABILITY_NOT_OPENED_STOP` before any detector invocation. Do not substitute inferred poses, synthetic camera placement, mesh rendering, or a mesh-only fallback.

Classical range-image fusion combines aligned observations into a cumulative field and may explicitly fill gaps; consequently, pre-fusion observations and their visibility structure are not equivalent to the final fused surface. That is the scientific reason this capability is materially new. ([Stanford Graphics][2])

## 3. Exact deterministic algorithm

### 3.1 Common geometry

Let (B) be the axis-aligned bounding-box diagonal of the frozen finite approximately 200,000 source points. Fail if (B\le 0) or is nonfinite.

For nonzero vectors (a,b), define projective angle

[
\theta(a,b)=\arccos!\left(
\operatorname{clip}
\left(
\frac{|a^\top b|}{|a||b|},-1,1
\right)
\right).
]

Directions are canonicalized by negating them when their first component with magnitude greater than (10^{-12}) is negative.

All geometry is computed in float64. All branch comparisons use values quantized before comparison to:

* (10^{-9}B) for distances;
* (10^{-6}) degrees for angles;
* (10^{-9}) for unitless ratios.

No random sampling is permitted.

### 3.2 Canonical frame selection and pixel budget

A semantic frame hash is calculated from dimensions, depth bytes, validity bytes, (K), and (T); filenames and paths are excluded.

If there are at most 16 frames, use all of them in semantic-hash order.

If there are more than 16:

1. Select the frame with the lexicographically smallest semantic hash.
2. Let (c_v) be its camera centre and (z_v) its world optical axis.
3. Iteratively select the frame maximizing

[
\min_{u\in S}
\left[
\frac{|c_v-c_u|}{B}
+
\frac{\theta(z_v,z_u)}{\pi}
\right],
]

with semantic hash as the only tie-breaker.
4. Stop at 16 selected views.

Choose the smallest integer stride (s\ge1) such that

[
\sum_v
\left\lceil \frac{H_v}{s}\right\rceil
\left\lceil \frac{W_v}{s}\right\rceil
\le 200{,}000.
]

Sample pixels at rows and columns (0,s,2s,\ldots). No phase search is allowed.

At least three selected views must each retain 512 valid lattice samples. The selected camera centres must include a pair separated by at least (0.05B), and their mean viewing rays toward the scan must differ by at least (15^\circ).

### 3.3 Per-view normal field

For every valid lattice point (p):

1. Take its (7\times7) organized-image neighbourhood.
2. Retain finite valid neighbours whose Euclidean distance from (p) is at most (0.020B).
3. Require at least 12 retained neighbours.
4. Compute the TLS covariance and its analytic symmetric (3\times3) eigendecomposition.
5. Require

[
\frac{\lambda_0}{\lambda_0+\lambda_1+\lambda_2}\le0.05,
\qquad
\lambda_0\le\lambda_1\le\lambda_2.
]

6. Use the (\lambda_0) eigenvector as the normal and orient it toward the camera centre.

Eigenvalue or eigenvector ties within (10^{-12}) use lexicographic component order. Calling a nondeterministic external eigensolver is prohibited.

### 3.4 Organized turn change-points

Evaluate exactly these eight unoriented image-lattice directions:

[
(1,0),(2,1),(1,1),(1,2),(0,1),(-1,2),(-1,1),(-2,1).
]

For each corresponding discrete profile, split it at:

* invalid depth runs longer than one lattice sample; or
* adjacent 3-D jumps greater than (0.006B).

At each eligible profile centre, construct left and right normal banks at two physical scales, using cumulative 3-D path distance from the centre:

| Bank  | Included path distance | Minimum normals per side |
| ----- | ---------------------: | -----------------------: |
| Small |   (0.003B) to (0.015B) |                        4 |
| Large |   (0.006B) to (0.030B) |                        6 |

For bank normals (n_j), compute the robust representative:

[
m_0=\operatorname{unit}!\left(\sum_j n_j\right),
]

[
w_j=\min\left(
1,\frac{5^\circ}{\max(\theta(n_j,m_0),10^{-12}!^\circ)}
\right),
]

[
m=\operatorname{unit}!\left(\sum_j w_j n_j\right).
]

Reject a bank if:

* the initial or reweighted vector norm is below (10^{-12});
* median angular residual exceeds (4^\circ); or
* nearest-rank 90th-percentile residual exceeds (8^\circ).

Let

[
\alpha_s=\theta(m_{Ls},m_{Rs}),\qquad
\alpha_l=\theta(m_{Ll},m_{Rl}),
]

and

[
C=\frac{\alpha_s}{\max(\alpha_l,1^\circ)}.
]

A centre is turn-positive only when:

[
15^\circ\le\alpha_s\le165^\circ,
]

[
0.65\le C\le1.35,
]

and the complete path from the end of the left large bank to the end of the right large bank remains connected under the (0.006B) adjacency rule.

Within each profile, identify local maxima over a physical radius of (0.006B). A plateau consists of consecutive centres with scores at least 98% of the local peak. Emit one token at the weighted-median profile position of the plateau, using

[
S=
\frac{\alpha_s\min(C,C^{-1})}
{1+(r_L+r_R)/(8^\circ)},
]

where (r_L,r_R) are the two small-bank median residuals. Ties go to the lower profile ordinal.

The token tangent is

[
t=\operatorname{canonical}\left(
\frac{m_{Ls}\times m_{Rs}}
{|m_{Ls}\times m_{Rs}|}
\right).
]

The token is rejected if the cross-product norm is below (\sin15^\circ).

This two-scale concentration test is the direct distributed-curvature rejection: a uniform cylindrical turn should spread with scale, whereas a localized fold should retain a substantially larger fraction of its turn at the small scale.

### 3.5 Per-view thinning

Partition each selected frame into a fixed (16\times16) grid in normalized pixel coordinates.

Retain at most two tokens per tile, ordered by:

1. higher (S);
2. higher (\alpha_s);
3. (C) closer to 1;
4. lower summed bank residual;
5. row, column, direction ordinal.

The second token is retained only if its token line differs from the first by more than (0.012B) in perpendicular distance or more than (10^\circ) in direction.

Hard caps:

* 512 tokens per view;
* 8,192 tokens per scan.

Cap overflow is terminal; there is no top-(k) truncation beyond the preregistered tile rule.

### 3.6 Deterministic multiview consensus

Every token line is used once as a seed hypothesis. This is exhaustive consensus, not randomized RANSAC.

Token (j) supports seed (i) when:

[
\theta(t_i,t_j)\le6^\circ
]

and

[
\operatorname{dist}(p_j,\ell_i)\le0.006B.
]

For every seed, compute:

* distinct supporting views;
* supporting camera-ray diversity;
* token axial span;
* ten equal-width axial occupancy bins;
* number of bins containing at least two distinct views;
* total token score;
* median seed-line residual.

Seed ordering is:

1. most distinct views;
2. most two-view bins;
3. most occupied bins;
4. greatest quantized total score;
5. lowest median residual;
6. lowest canonical seed bytes.

Fit the highest-ranked seed’s inliers by weighted TLS of the token points, with

[
w_j=\frac{1}{1+(r_j/5^\circ)^2},
]

where (r_j) is summed small-bank roughness. Recompute inliers and refit exactly twice.

Accept a line only if all hold:

* at least 3 distinct views;
* at least one supporting view pair has camera-centre separation (\ge0.020B) and ray angle (\ge15^\circ);
* at least 8 tokens;
* span (\ge0.10B);
* at least 7 of 10 axial bins occupied;
* at least 5 bins contain tokens from two or more views;
* nearest-rank 95th-percentile perpendicular residual (\le0.004B);
* median token-direction error (\le5^\circ).

Endpoints are the minimum and maximum accepted-token axial projections. Accepted inliers are removed; consensus selection repeats. At most 256 raw-view lines may be produced.

Robust consensus fitting is an established response to outlier-contaminated geometric observations; CMTCC makes it deterministic by exhausting the bounded token hypotheses instead of drawing random samples. ([SRI][3])

### 3.7 V52 matching, preservation, birth, and pruning

A raw-view line and V52 line are match-feasible when:

* projective angle (\le4^\circ);
* symmetric midpoint-to-infinite-line distance (\le0.008B);
* shorter-span projected overlap (\ge0.60).

Resolve all feasible matches with one deterministic bipartite assignment:

1. maximize cardinality;
2. maximize summed overlap quantized to (10^{-9});
3. minimize summed distance;
4. minimize summed angle;
5. select the lexicographically smallest assignment.

For every match, emit the **original V52 record byte-for-byte**. The raw line is consumed.

Every unmatched raw line is a new candidate birth.

For each unmatched V52 line, evaluate exactly 11 stations:

[
0.05,0.14,0.23,0.32,0.41,0.50,0.59,0.68,0.77,0.86,0.95.
]

A view is eligible at a station when:

* the station projects inside the frame;
* the nearest valid back-projected lattice point is within (0.008B);
* at least four of the eight profile directions have complete small and large banks.

An eligible view is positive if the **unthinned** token map contains a token within (0.010B) of the station and within (10^\circ) of the V52 direction. Otherwise it is negative.

A station is negative-certified only if at least three eligible views are negative and at least one view pair satisfies the (0.020B/15^\circ) baseline-diversity rule.

Prune the V52 line only when at least 9 of 11 stations are negative-certified. Otherwise retain its original bytes.

Output order is:

1. retained V52 records in original V52 order;
2. unmatched raw-view lines ordered by decreasing view count, occupied bins, span, median turn, then canonical endpoint bytes.

One final exact-duplicate pass suppresses the later line only at:

* angle (\le2^\circ);
* distance (\le0.003B);
* shorter-span overlap (\ge0.90).

V52 wins exact ties.

## 4. Source-free fixture gate

All fixtures are generated without the sealed scans and must pass before any capability bundle is opened geometrically.

### Required positives

1. Sharp (90^\circ) fold, five views: exactly one line.
2. Rounded (60^\circ) fold with radius (0.008B): exactly one line.
3. Shallow (20^\circ) fold with radius (0.004B): exactly one line.
4. One fold absent from two of five views through occlusion: exactly one line from the remaining three.
5. Two parallel folds separated by (0.08B): exactly two lines.
6. Aligned V52 incumbent: one output whose V52 bytes are unchanged.
7. Empty V52 set with an observable fold: one raw-view birth.
8. Flat-panel V52 line with five observable views: the V52 line is pruned and no raw line is born.

### Required negatives

1. Single plane: zero lines.
2. Uniform cylinder accumulating (90^\circ) over (0.25B): zero lines.
3. Pure depth silhouette step: zero lines.
4. Open boundary: zero lines.
5. Two disconnected overlapping panels: zero lines.
6. Crease-like corruption in only one view: zero lines.
7. Agreement in exactly two views: zero lines.
8. Two unresolved folds separated by (0.010B): abstain rather than merge them into one line.

### Required metamorphic and fail-closed tests

* frame permutation and path/name changes produce identical semantic bytes;
* global rigid transformation produces equivariant lines;
* scale factors 0.5 and 2.0 produce scaled-equivalent lines;
* primary/replay are byte-identical;
* malformed (K), nonrigid (T), nonfinite depth, duplicate semantic frame hashes, cap-plus-one tokens, or nonfinite output fail closed.

The fixture suite must report every expected count and every preservation/pruning outcome, not merely a single aggregate pass bit.

## 5. Total schema and operational preflight

Before the first sealed invocation, audit all 30 capability bundles and both planned invocation manifests.

The audit must prove:

* frame count (3)–(64);
* no frame exceeds (8192\times8192);
* no scan exceeds 100,000,000 raw pixels;
* every valid depth is finite and positive;
* every invalid depth is canonical zero;
* (K) is invertible, with positive focal lengths;
* (T) has bottom row exactly ([0,0,0,1]);
* (|R^\top R-I|_\infty\le10^{-9});
* (|\det R-1|\le10^{-9});
* primary/replay input hashes are identical;
* all V52 rows are finite and parseable;
* V52 count is at most 2,048;
* the source, frame, and calibration coordinate systems agree.

For coordinate agreement, choose 4,096 valid pixels per selected view by semantic-hash stride. Their back-projected nearest-source distances must satisfy:

* median (\le0.003B);
* 95th percentile (\le0.015B);
* at least 90% (\le0.020B).

The schema audit may report only dimensions, counts, hashes, finite-value results, calibration residual summaries, and cap margins. It may not emit turn scores, tokens, line hypotheses, V57 predictions, or visualizations.

A separate worst-case source-free stress fixture must exercise:

* 16 selected views;
* exactly 200,000 lattice samples;
* 8,192 tokens;
* 2,048 V52 lines;
* 256 accepted raw lines.

It must complete on the execution host in at most 240 seconds and below 12 GiB RSS. Failure prevents sealed execution.

The frozen lock must bind:

* all source and capability manifests;
* code and dependency hashes;
* constants;
* fixture outputs;
* total schema audit;
* supervisor implementation;
* canonical serializer;
* visualizer and scorer hashes;
* CPU-thread environment.

## 6. Execution order and process supervision

Execute exactly 60 children in this order:

[
\text{S001 primary, S001 replay, S002 primary, S002 replay, }\ldots,\text{ S030 replay}.
]

Before each spawn, append and fsync an `INVOCATION_START` event containing only ordinal and input/code hashes.

Each child must run:

* in a new process group;
* with network disabled;
* with truth, family maps, V56 directories, and scoring code unmounted;
* with one computational thread;
* with deterministic locale and hash environment;
* under kernel-enforced memory isolation.

Use cgroup v2 on Linux or an equivalent kernel job object:

* `memory.max = 17,179,869,184` bytes;
* `pids.max = 64`.

The supervisor starts a monotonic clock immediately before spawn. At

[
300{,}000{,}000{,}000\text{ ns}
]

without a successfully closed child result, it sends an unconditional process-group kill and records `TIMEOUT_300S`. There is no warning threshold and no operator decision.

The child writes only into a temporary directory. The supervisor publishes a result atomically only after:

* clean exit;
* schema validation;
* finite-value validation;
* forbidden-key validation;
* canonical reserialization;
* result hash calculation.

Runtime and RSS metadata are stored outside the semantic prediction bytes. The replay comparison occurs immediately after each replay child completes.

Any human interruption, host restart, supervisor crash, child timeout, OOM kill, or partial artifact is terminal. No resume from the next case is allowed.

After all 60 children:

1. freeze the complete execution manifest;
2. run the independent semantic validator;
3. create all required truth-free visual artifacts;
4. record the blind visual decisions;
5. only then mount truth and family mappings for exactly one scoring call.

No score is produced from fewer than 60 valid records.

## 7. Compact output and cache

Maximum semantic JSON size: **2 MiB per scan**.

Permitted structure:

```text
schema_version
source_sha256
capability_sha256
v52_sha256
scale_B_mm
retained_v52_count
pruned_v52_count
raw_new_count
lines[]
    origin: V52_HOLD | RAW_VIEW_NEW
    line_start_xyz
    line_end_xyz
    v52_ordinal_or_null
    supporting_view_count
    token_count
    occupied_bin_count
    two_view_bin_count
    median_turn_deg
    median_concentration
    p95_residual_over_B
pruned_v52[]
    v52_ordinal
    reason: RAW_VIEW_OBSERVABLE_NEGATIVE
    negative_station_count
    minimum_negative_view_count
```

Prohibited output and cache material:

* depths, validity masks, pixels, normals, profile samples;
* token coordinates or token IDs;
* source points, vertices, faces, neighbourhoods;
* per-view support lists;
* membership or source-support bitsets;
* V56 identifiers or copied V56 statistics.

The visualizer must recompute its evidence from the immutable input bundle and frozen semantic output.

## 8. Retention gates

Every gate is conjunctive.

### Operational and numerical validity

* 30/30 primary records complete.
* 30/30 replay records complete.
* 30/30 semantic prediction pairs byte-identical.
* No nonfinite output.
* No schema or cap violation.
* Every child wall time strictly below 300 seconds.
* Primary P95 wall time at most 270 seconds.
* Maximum process RSS strictly below 16 GiB.
* No child crash, timeout, OOM, or scan-level abstention.

### Aggregate score

The score must satisfy all of:

* TP (\ge60);
* FP (\le22);
* FN (\le40);
* precision (\ge0.72);
* recall (\ge0.60);
* F1 (\ge0.65);
* exact-count accuracy (\ge10/30).

At the limiting integer point TP 60, FP 22, FN 40, F1 is approximately 0.659; this is materially above V52 rather than a nominal one-line gain.

### Mechanism-specific evidence

* At least 8 `RAW_VIEW_NEW` lines are TPs.
* `RAW_VIEW_NEW` precision is at least 0.80.
* At least 2 TPs are absent from the entire frozen union pool, across at least 2 families.
* At least 5 pruned V52 lines are scored FPs.
* **Zero pruned V52 lines may be scored TPs.**
* No retained V52 record may differ bytewise from its original record.

### Family breadth

* At least 7 of 15 families have strictly lower (FP+FN) than V52.
* At least 4 families gain a TP.
* At least 3 families lose an FP.
* No family has lower recall than under V52.
* No more than one family may have higher total (FP+FN).
* Minimum family recall must be at least 0.25.

### Frozen exact-case safety

S005, S006, S007, S013, S026, and S027 must each remain:

* exact-count correct;
* free of FP and FN;
* visually admissible.

A raw-view addition or pruning in one of these cases is permitted only if the final scored set remains perfect and every changed line passes the visual gate.

## 9. Mandatory visual evidence and failures

Render every changed case, all six V52 exact cases, and S011/S016/S024/S029.

For each case produce:

1. A 4096×2304 three-view full-source overlay.
2. For every raw-new line, at least three native-pixel range-view crops showing:

   * projected line;
   * selected transverse direction;
   * small and large banks;
   * local turn maximum;
   * continuous material across the centre.
3. For every pruned V52 line, three negative-view panels at representative certified stations.
4. A full-length axial support plot showing the ten occupancy bins.

A raw-new line passes only when at least three calibrated views visibly show the same continuous fold and the projected axis lies within (0.006B) of the transverse turn maximum.

A prune passes only when the required views visibly show a continuous locally flat or distributed-curvature surface with no centered turn. Mere token absence is not visually sufficient.

Any one of the following retires V57:

* accepted axis on a silhouette, hole, gap, panel interior, or disconnected overlap;
* accepted axis transversely displaced from the visible turn;
* floating or diagonal cross-panel line;
* one physical fold represented twice;
* merged parallel folds;
* pruned line with a visible localized fold in any mandatory negative panel;
* a changed exact V52 case that is visually ambiguous;
* missing, cropped, downscaled, or unreadable evidence.

Visual disagreement is not a reason to adjust a threshold. It is a terminal falsification.

## 10. Leakage audit

The independent validator must establish:

* inference input keys exactly match the whitelist;
* recursive rejection of truth, label, CAD, drawing, filename, path, family, part, expected-count, benchmark, score, V56, and identity-like fields;
* no branch reads the orchestration case token;
* no absolute camera pose, frame count, or acquisition order is used as a family-routing feature;
* calibration is used only for back-projection, projection, visibility, and relative-view diversity;
* frame order and filename-change fixtures pass;
* truth and family maps remain unavailable until the complete output and blind visual decisions are frozen;
* no V56 file is opened, imported, hashed as an input dependency, or used by the visualizer.

## 11. Terminal and no-rescue boundary

Any failure after `V57_CMTCC_MECHANISM_LOCKED` produces `V57_CMTCC_TERMINAL_RETIRE`.

Forbidden rescue actions include:

* changing any angle, distance, scale, count, bank, view, tile, station, matching, pruning, or score constant;
* increasing token or line caps;
* skipping a malformed scan or frame;
* reducing the candidate population after overflow;
* replacing missing raw views with rendered views;
* relaxing three-view consensus to two;
* changing timeout accounting;
* restarting at the failed ordinal;
* rerunning only nondeterministic or visually failed cases;
* creating V57b with the same mechanism;
* importing CTMLR sections, seven stations, E/S/A seeds, signed turn-mass medians, five-of-seven logic, run-three reconstruction, or V56 artifacts.

A successor after failure must pose a genuinely different causal question or open a different prospectively justified capability.

## Research basis and Sherman-specific hypotheses

The established basis is limited:

* Range-fusion research establishes that multiple aligned range observations are integrated into a new cumulative representation, potentially including gap filling; therefore preservation of pre-fusion views is a genuinely different observable. ([Stanford Graphics][2])
* Robust local normal ensembles have been used to distinguish smooth surface, crease discontinuity, and indeterminate orientation on noisy range-derived meshes. ([Oak Ridge National Laboratory][4])
* Ordered-profile methods such as fused penalties and exact change-point optimization establish principled ways to favor local constancy and sparse changes. ([OUP Academic][5])
* Consensus model fitting is an established robust-geometry strategy; V57 replaces random draws with bounded exhaustive hypotheses to satisfy deterministic replay. ([SRI][3])

The following are **Sherman-specific, unvalidated hypotheses**: the two physical bank scales, (0.65) concentration threshold, three-view requirement, tile thinning, consensus tolerances, 9-of-11 negative certificate, raw/V52 matching contract, and all retention gates. The cited research does not establish that these constants will recover Sherman bends. The single frozen execution is the required falsifier.

[1]: https://pubmed.ncbi.nlm.nih.gov/29531091/ "https://pubmed.ncbi.nlm.nih.gov/29531091/"
[2]: https://graphics.stanford.edu/papers/volrange/ "A Volumetric Method for Building Complex Models from Range Images"
[3]: https://www.sri.com/publication/artificial-intelligence-pubs/random-sample-consensus-a-paradigm-for-model-fitting-with-applications-to-image-analysis-and-automated-cartography-2/ "https://www.sri.com/publication/artificial-intelligence-pubs/random-sample-consensus-a-paradigm-for-model-fitting-with-applications-to-image-analysis-and-automated-cartography-2/"
[4]: https://impact.ornl.gov/en/publications/normal-vector-voting-crease-detection-and-curvature-estimation-on/ "Normal vector voting: Crease detection and curvature estimation on large, noisy meshes - Oak Ridge National Laboratory"
[5]: https://academic.oup.com/jrsssb/article-abstract/67/1/91/7110658 "oup.silverchair-cdn.com"
