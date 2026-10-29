# Professor Package — Enterprise F1 Promotion Checkpoint (2026-07-13)

## Decision

Do not promote F1 to runtime/main. Engineering containment, cache determinism,
blind benchmark execution, and visual gating are now reproducible, but the
semantic bend-object model fails the enterprise accuracy and ownership gates.

## Reproducible engineering evidence

- Detector proposal input is deterministically bounded at 192,000 points for
  original and perturbation modes. Full-resolution points remain available for
  atom construction and final ownership geometry.
- Research cache schema v4 stores `bbox_min`, `bbox_max`, and
  `source_point_count`; it does not serialize `bbox_points`. Atom/flange/bend
  cache values retain full floating-point precision.
- Patch-graph F1 reuses the already loaded geometry and candidate bundles when
  evaluating the zero-shot control; the former duplicate detector pass is gone.
- All three large-support PCA paths in `BendDetector` use bounded 3x3 covariance
  eigensolves rather than NumPy's default full N x N SVD allocation.
- Review meshes use deterministic vertex clustering with a hard 120,000-triangle
  display budget; source ownership coordinates and the original camera bounds
  are unchanged.
- Fresh render-inclusive 10839000 profiles now peak at 1,331,544,064 bytes for
  the full scan and 1,315,831,808 bytes for scan 2. Across all 13 isolated jobs,
  the maximum is 1,824,194,560 bytes (1.70 GiB), far below the 16 GiB target.
- Ten independent no-render repeats produce one normalized semantic SHA-256:
  `4c6443c9bbc4d654872ea7066e5efb827ee0e48b0f724324335dc4d622d3bcbd`.
  Only cache-hit telemetry and stage timings are excluded.
- Full Python suite after the Eyal intake, isolation, render-cache, and native
  SLDPRT evidence, fail-closed truth ingestion, and local scan-capacity tranches:
  925 passed, 1 skipped.
  Frontend lint/build pass. npm audit:
  zero vulnerabilities.

## Blind evidence

### Blind scan-only F1, 13 typed scans

The migrated typed slice contains 11 PLY scans and two scan-derived STL meshes;
those STL records are the only release-eligible typed truth for 15651004 and
47959001. No CAD is supplied to F1 inference.

- Accepted exact: 0/13 (0.0%).
- Range contains visible conventional truth: 6/13 (46.15%).
- Raw F1 exact: 3/13 (23.08%).
- Render-admissible exact: 3/13 (23.08%).
- Visually accepted exact: 0/13.
- Reliability: 13/13 successful.
- Render-inclusive P95: 49.803 s (passes the 300 s gate).
- Worst render-inclusive RSS: 1,824,194,560 bytes (1.70 GiB).

### Blind CAD+PLY, 12 typed scans

- Blind inference contract: 12/12.
- Exact visible conventional count: 2/12 (16.67%).
- Pipeline P95: 35.513 s.
- Reliability: 12/12 after installing the declared gmsh STEP fallback.
- Uplift over accepted PLY-only exact: +16.67 points, but absolute accuracy is
  far below the required 95%.

### Promotion report

`output/migration/enterprise-promotion-gates-current.json` is blocked 11/16.
Passing gates are blind CAD contract, CAD uplift, 10/10 determinism, P95
latency, and job reliability. Accuracy, range, visual ownership, truth volume,
holdout, family coverage, and instance geometry remain blocked.

### Expanded Eyal lane, 50 unadjudicated geometry parts

A complete sender audit recovered 51 messages from Eyal Buingu and 50 unique
part packages, not only the initial 13 examples. Each package contains one
drawing PDF, one SLDPRT, and one STL; none contains a PLY scan. The STL role is
therefore quarantined as unadjudicated geometry and is not counted as scan-only
release evidence.

The prediction-hidden, inference-only F1 baseline completed 50/50 parts with
three isolated workers, P95 112.175 seconds, and worst RSS 12,932,694,016 bytes
(12.05 GiB). F1 produced a raw count on all 50 parts, but only 2/50 candidates
were accepted directly from raw F1; 48/50 remained control-guarded or
control-constrained. Exact accuracy is intentionally undefined until the 87
drawing pages are independently adjudicated. See
`docs/research/EYAL_50_ZERO_SHOT_BASELINE_20260713.md`.

The focused seven-part visual probe (all five marker-blocked cases plus both
direct raw-F1 cases) accepted no final markers. It exposes perimeter/hole
capture, duplicate corner clusters, short fragments, curve-family confusion,
and missed internal ownership. The snapshot renderer now reuses one bounded
review mesh across camera views; this is a render-only performance fix.

Before any drawing label existed, the Eyal lane was frozen into a deterministic
20-part development and 30-part holdout split using package hashes only. A
dual-review merge now rejects prediction/holdout exposure and quarantines count
or family disagreements. A separate one-way scorer verifies package, drawing,
geometry, runtime-config, cache-schema, and coordinate-frame identities before
opening development versus holdout metrics. No Eyal score exists yet because
the supplied packages contain no explicit GT and independent reviews are still
pending.

As an adjudication accelerator, the 50 native SLDPRT histories were decoded
with a pinned `cadmpeg 0.2.0` binary. Decode succeeds 50/50; 41 parts have exact
agreement between unique active bend-feature names and their corresponding
flatten-history references. The artifact is prediction-free, split-free, and
metric-ineligible. It is withheld until both first-pass drawing reviews are
frozen.

This evidence cannot be promoted to conventional-bend GT. The two overlaps
with existing manual scan truth prove both outcomes: `37361005` is compatible
at 4 CAD features versus 4 conventional bends; `49125000` is non-equivalent at
3 CAD features versus 2 conventional bends because one feature is the rolled
`R37` transition. The nine-check boundary audit passes and requires the file to
remain reconciliation-only.

The blind-slice builder now independently enforces registry schema v2, the
visible-conventional semantic target, approved manual/dual-review statuses and
provenance, nonnegative integer counts, and a resolved measured-scan source.
Even a CAD feature-history record accidentally marked
`release_metric_eligible=true` is rejected rather than silently entering a
benchmark.

A content-hashed audit of the broader local archive finds 38 unique scan
candidates at the absolute optimistic ceiling: 13 release-eligible, 20
registered but pending, and 5 unregistered PLY candidates. Four duplicate paths
collapse by SHA-256. Even if every pending candidate survives measured-scan
provenance and independent review, 12 new external scans are still required to
reach 50; from the current verified state, 37 additional verified scan records
are required. Eyal's CAD-only 50 cannot fill this scan gap.

The 25 pending unique geometries now have a prediction-hidden, SHA-256-locked
review queue. Twenty have a resolved part ID; five need identity resolution.
Only two geometries (both `49125000`) have an accessible drawing, so 23 are
blocked from drawing-backed independent review. Reviewer templates contain no
legacy total, F1 output, route, or split membership, and CAD references are
withheld until reconciliation.

## Current implemented energy

For atom labels \(z_i\), current code minimizes

\[
E(z)=\sum_i D_i(z_i)
 + 0.32\sum_{(i,j)\in\mathcal E}w_{ij}[z_i\ne z_j]
 + \sum_{\ell\in\mathrm{active}} A(\ell)
 + P_{inc}+P_{frag}+P_{dup}+P_{mass}.
\]

Activation costs are 0.25 for a flange and 0.45/0.60/0.75 for
transition/circular/developable bend classes.

Flange data cost is

\[
D_i(F)=\max(0,
 d_{plane}/\max(2h_i,1)
 + \Delta n/25
 + \min(1,20\kappa_i)
 + s_F),
\]

where \(s_F=-0.4\) for seeded support and 0.35 otherwise.

Bend data cost is

\[
D_i(B)=\max(0,
 d_{line}/\max(0.75w_B,1.25h_i,1)
 + \Delta a/35
 + C_{normal}(i,B)
 + C_{class}(B)
 + s_B),
\]

where \(s_B=-0.45\) for seeded support and 0.45 otherwise, and the normal term
chooses the best candidate flange pair. Component admissibility then applies
support-mass, two-flange contact, line/axis/span, leakage, and duplicate gates.

The diagnostic junction local MAP currently selects non-overlapping path
candidates by adjusted score

\[
S'(p)=S(p)-C_{obj}(p),
\]

with conventional object cost \(3.25\) (+2 for weak contact) and a higher
internal-ridge cost. This diagnostic is not yet the counted object solver.

## What the renders prove

The floating-marker/frame bug remains resolved: no unsupported accepted final
marker was observed in the 13 mismatch reviews. The gate suppresses final dots
when ownership is inconsistent. This is not visual success.

The remaining failures are semantic:

- exterior/free-edge and hole-adjacent ridges compete as conventional bends;
- multiple supports alias the same physical bend/edge family;
- connected corner/junction domains are not decomposed into maximal direct
  flange-pair bend-line paths;
- internal conventional bends remain unowned when perimeter evidence is stronger;
- rolled/rounded transitions are not a conventional line-object family.

Representative cases:

- 49024000: raw count 3 but only one marker-admissible OB2-like arm; two direct
  conventional arms remain embedded in a connected corner complex.
- 38172000 scan 2: 12 raw / 9 render-admissible versus truth 8; duplicate side
  families coexist with missed ownership.
- 40500000: 5 raw / 1 render-admissible versus truth 7; perforated perimeter
  support dominates and most bends are missed.
- ysherman: 2 raw/render-admissible versus truth 12; one local family absorbs
  attention while approximately ten bends remain unowned.
- 49125000: curve/rolled support is correctly blocked as conventional success.

## Explicit questions requiring professor guidance

1. For one connected ridge domain \(R\), what exact MAP/MDL factorization should
   jointly infer countable maximal flange-pair paths \(B_R\), uncounted junctions
   \(J_R\), perimeter edges \(E_R\), curve/rolled support \(C_R\), and residual
   \(U_R\)?
2. What chord rejection inequality should suppress a direct pair such as
   F1–F17 when a sparse path F17–F11–F14–F1 explains overlapping support with
   lower description length?
3. What birth prior should allow an uncovered internal flange-pair corridor to
   create a bend-line object when its ridge evidence is weaker than exterior
   perimeter evidence?
4. Should duplicate suppression be a hard equivalence relation conditioned on
   flange pair + maximal axis interval, or a soft repulsive factor inside the
   local MAP?
5. Should free-edge/perimeter and curve/rolled objects consume separate object
   budgets so they cannot compete with conventional bend lines?
6. What minimum instance-level truth geometry is required to identify these
   factors without leaking count labels into inference?

## Package paths

- Scan-only summary: `output/migration/blind-ply-f1-isolated-v5/summary.json`
- Scan-only operations: `output/migration/blind-ply-f1-isolated-v5/operational_summary.json`
- Visual audit: `output/migration/blind-ply-f1-isolated-v5-review/VISUAL_MISMATCH_AUDIT.md`
- 13 review packs: `output/migration/blind-ply-f1-isolated-v5-review/`
- CAD+PLY summary: `output/migration/blind-cad-ply-v3/summary.json`
- Promotion report: `output/migration/enterprise-promotion-gates-current.json`
- Ten-run outputs: `output/migration/10839000-determinism-v2/`
- Prior semantic package: `docs/PROFESSOR_F1_SEMANTIC_WALL_DROPOFF_20260508.md`
- Eyal 50 baseline: `docs/research/EYAL_50_ZERO_SHOT_BASELINE_20260713.md`
- Eyal 50 visual audit: `output/migration/eyal-50-f1-visual-probe-v1/VISUAL_AUDIT.md`
- Eyal 50 adjudication protocol: `docs/research/EYAL_50_ADJUDICATION_PROTOCOL.md`
- Eyal 50 contract audit: `output/migration/eyal-50-zero-shot-contract-audit-v1.json`
- Eyal SLDPRT evidence: `data/bend_corpus/annotations/eyal_50_sldprt_bend_evidence.json`
- Eyal SLDPRT boundary audit: `output/migration/eyal-50-sldprt-bend-evidence-audit-v1.json`
- Local scan expansion capacity: `output/migration/local-scan-expansion-capacity-v1.json`
- Scan-to-50 data request: `docs/research/SCAN_EXPANSION_TO_50_DATA_GAP_20260713.md`
- Local scan review queue: `output/migration/local-scan-adjudication-v1/`
- Local scan review protocol: `docs/research/LOCAL_SCAN_ADJUDICATION_PROTOCOL_20260713.md`
