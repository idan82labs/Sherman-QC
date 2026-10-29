# Gold11 V5 resource-planning review — 2026-09-11

Decision: the timing architecture is reviewable, but the presently available
evidence does not identify a smallest sufficient physical runtime. Keep all seven
limits unset and release false until the already-planned connected terminal POST
and constructed native-byte timing are available. Choose one prospective,
uniform engineering allocation before the final 85-outcome source gate; do not
derive a success guarantee, per-case setting or retry from this review.

This is a source/narrative-only planning review. No target import, test collection,
execution, native geometry, raw input, manifest, control record, attempt artifact
or active scientific artifact was accessed. The 97 active frozen sources were
not modified. This document is not an exact PRE or permission to run.

## Evidence that may and may not determine the allocation

The released [V3 result](2026-09-10_gold11_mesh_development_v3_result.md) and
[failure-factor audit](2026-09-10_gold11_failure_factor_audit.md) distinguish:

- Six challenger cases / 45 references completed finite screening with zero
  passing starts. Their parent case times were 24.261864–41.372338 seconds.
  They did not complete the expensive refinement-plus-measurement path.
- Three cases / 19 references ended with exit -24 under the old 150-CPU-second
  limit, at parent times 150.707849–150.999727 seconds. No returned RESULT or
  partial established their internal stage. These are censored failures, not
  measured 150-second completed workloads and not evidence specifically of a
  registration or CAD bottleneck.
- The remaining two cases / seven references refused at other known boundaries.
  They must not be averaged into a forecast for completed positive workflows.
- The observed 959.111951-second campaign completion does not budget a new
  22-arm positive workflow. V3's formal release was separately blocked by the
  preserved extra-file inventory failure. Authenticated negative dispositions
  do not erase that custody limit.

The old source used C=150, W=180, scorer CPU=50, scorer wall=60, P=20,
R=30 and T=4200 seconds. Those numbers are historical, not recommended V5 limits.

Root has separately reported OUTPUT-only progress from the still-active
connected packet: first U-R registration used 1047.316910 CPU seconds /
1048.125053125 wall seconds; its required V2 measurement used 1.697320 CPU
seconds and passed its original oracle. The V1 diagnostic failed; three cases
remain. This is explicitly provisional parent-reported progress, not independently
audited terminal evidence, a complete byte-workflow time, or a physical timing
bound. It already contradicts a proposal to give this constructed operating
regime only 300–600 CPU seconds. It does not establish that 1200 seconds is enough.

V5 retains 50,000 registration representatives, 32,768 CAD samples, up to 48
refined starts and 30+30 iterations over 2048/8192 coarse/fine points, plus finite
certificate, prefilter and source-membership limits. Finite operation counts
bound admitted work, not seconds. BVH traversal, ties, support coverage, rejected
starts, native conversion and JSON size prevent a justified linear extrapolation
from a small curved fixture or one successful start. Host memory and a fast CPU
do not change these scientific or resource limits.

## The seven limits are one allocation

Symbols below refer to the existing LIMITS keys, not new runtime parameters.

| Limit | Required scope and prospective check |
| --- | --- |
| C: case_cpu_seconds | Positive exact integer; the inference worker's CPU allowance includes its setup/import and byte workflow. Use completed whole-leaf evidence, not registration alone. Censoring remains a failure. |
| W: case_wall_seconds | Positive finite whole-case acceptance window: preparation, child work, retained-result read/projection and the final case clock check. Choose headroom for wall/CPU divergence and parent work; W=C is not an adequate default. |
| Sc: scorer_cpu_seconds | Positive exact integer; separate pure scorer leaf. Coordinator original-byte reconstruction, request construction and parent revalidation are outside this leaf's CPU budget. |
| Sw: scorer_wall_seconds | Positive finite scorer supervision window, with import/read/serialization and wall headroom above the planned scorer CPU. Do not copy the historical empty-output timing as a positive-output throughput qualification. |
| P: parent_read_seconds | Positive finite deadline for several differently sized operations. In particular, one P bounds the entire all-22 reconstruction before Gold, not just one file read. |
| R: terminalization_reserve_seconds | Positive, strictly less than T; must cover outer cleanup and all post-coordinator custody, not just writing COMPLETED. |
| T: total_wall_seconds | Positive finite outer campaign allowance after STARTED publication; include startup, all 22 case opportunities, full coordinator/scorer work and R. Source preflight before STARTED has its own P and is outside this T clock. |

The current release guard checks positivity, finiteness, exact CPU integer types
and R<T. It does not check this allocation arithmetic, W versus CPU/wall needs,
Sw versus Sc, or equal opportunity across all 22 ordered arm-cases. Exact PRE
must check those explicitly. No runner edit is required to make that declaration.

The workers are serial and numeric thread counts stay one. 22*C+Sc is the sum of
declared direct-worker CPU allowances, not a proved whole-campaign CPU ceiling:
coordinator/outer CPU is separate and process supervision has stated observation
limits. Keep existing 8 GiB leaf, 4 GiB coordinator, 12 GiB whole-tree and 2 GiB
scorer limits; no parallel heavy job or RAM-based cap increase.

## Source-derived wall accounting

In [supervise_case](../../scripts/run_gold11_mesh_development_v5.py), the worker
gets the remainder of W after preparation. Up to four P-bounded preparation
blocks precede it: STARTED publication, paired input read, private input
publication and leaf dispatch publication. Retention, RESULT read and
decode/projection can then require three more P-bounded blocks before the clean
case clock check. Thus an allocation intended to permit a full leaf wall budget
L needs room for preparation and accepted postprocessing as well as L. A
conservative phase-ceiling calculation is W >= L+7P+g, where g is separately
declared administration/OS headroom; it is not a minimal necessary inequality
when the actual bounded blocks are much faster. L itself is not established by
a CPU limit alone.

W is not a hard bound on total function return time. Failed-child cleanup may
take five extra seconds; retention and failure conversion may follow an expired
worker window; the two durable terminal publications occur after the reported
case elapsed time. The shared supervisor explicitly requires its caller to
reserve cleanup within the caller's absolute deadline.

For full-opportunity planning, a conservative source-accounting envelope is:

    Bcase = W + 5*P + 6
    T >= H0 + 22*Bcase + Bs + R

Bcase assumes preparation fits its planned W; the added terms cover the at-most
three post-worker P blocks, two terminal publications, five-second cleanup and
a one-second process sample conservatively across alternative branches. H0 must
include outer/coordinator startup, source verification and loop administration.
Bs is the complete post-inference coordinator/scoring block, not Sw alone:
all-22 reconstruction, source rechecks, reference snapshot/request creation,
scorer dispatch, scorer supervision/cleanup, retained score validation and
terminal publications. Include the remaining coordinator publications in Bs.

This is a transparent conservative allocation model, not an OS hard real-time
proof. Synchronous process creation, filesystem operations, temporary-directory
cleanup and signal delivery have stated limitations. These formulas must not be
advertised as guaranteed physical completion time.

The coordinator checks only whether its own remaining T-R is at least W before
a case. Its clock also starts later than the outer clock. A shorter T can
therefore refuse later ordered cases with TOTAL_WALL_RESERVE_UNAVAILABLE even
when all individual C/W values look generous. If a deliberately clipped
campaign is chosen, declare that restriction in advance and retain every case
in both denominators; do not call it a full 22-case allocation.

## Finalization is a material workload

After coordinator return, the outer parent requires quiescent writers, a fresh
source/control audit, all-22 terminal validation and original-byte projection
reconstruction, then three full campaign inventory passes surrounding COMPLETED
and FINAL_GATE. The 256 KiB control envelopes and original-result bindings remain
mandatory. An expired finalization cannot release a score.

At the declared maximum of 22 complete 128 MiB RESULTS:

- One all-22 RESULT pass is 2816 MiB = 2.75 GiB.
- Parent artifact validation reads a clean RESULT twice: retained-byte validation
  and independent projection reconstruction, totaling 5.5 GiB.
- Three inventory passes add 8.25 GiB of RESULT reads.
- R therefore encounters 13.75 GiB of repeated RESULT reads alone; the earlier
  coordinator pre-Gold pass adds another 2.75 GiB outside R.

This is byte-ceiling arithmetic, not a claim those maximum files will occur.
Wrapper, source, control, score and dispatch I/O is additional; decoding and
canonical revalidation also consume CPU and memory. The inventory's 8 GiB
schema read guard applies per pass, not once across the campaign. The ordinary
119-file source fixture establishes structure, not 128 MiB-per-result throughput.

A defensible reserve calculation is:

    R >= 5 + Hsource + Hterminal_and_reconstruction
             + Hinventory_1 + Hinventory_2 + Hinventory_3
             + Hpublication_and_clock + margin

Use the actual planned artifact-size regime and conservative engineering
headroom for these terms. P must independently fit the largest whole block
using it, especially all-22 pre-Gold reconstruction and outer source preflight.
P and R are not interchangeable. Reusing R=30 only because V3 used it is
unsupported.

## Smallest defensible next planning step

Use the already-planned terminal connected/native evidence; no new physical
probe, broad benchmark or scientific-contract change is required by this review.

1. After independent terminal qualification, tabulate completed stage and whole
   native-byte CPU/wall, exact operating sizes and result-byte sizes. Preserve
   failed/missing stages separately. Use the maximum completed relevant work,
   not the mean, and do not label the native proxy a physical upper bound.
2. Before physical access, choose and record a finite engineering-headroom rule.
   A proportionate default is at least 25% above relevant completed whole-leaf
   CPU, rounded upward to a 60-second integer boundary. This is a planning rule,
   not a confidence interval. It cannot be applied to incomplete stage timings
   as though they were whole-leaf observations.
3. Select W, Sc/Sw, P and R for their actual scopes above. Where no measured
   large-payload timing exists, label the chosen values as conservative
   engineering allowances and preserve budget failure as an allowed outcome.
   Do not falsely require proof of finishing every possible permitted payload.
   Avoid arbitrarily multiplying every physical case by the slowest small
   fixture while claiming predictive accuracy.
4. Compute T from the declared envelopes and reserve, round upward, and state
   the resulting maximum campaign commitment. If that commitment is too large,
   explicitly choose a clipped evaluation or investigate a separately versioned
   performance mechanism; do not silently reduce later case opportunity.
5. Freeze all seven concrete values and the final release source before the
   final 85-outcome source gate, update its exact runner hash, then require
   independent exact physical PRE. Do not use that gate to justify changing
   already-qualified time literals afterward. Once started, no cap increases,
   retries, per-case tuning or inference from Gold.

At present the smallest defensible numerical recommendation is therefore
*not yet determined*, rather than the old 150/180 limits or an invented
success-guaranteeing number. The pending measurements can support one bounded
engineering choice; they cannot remove the possibility of physical abstention,
budget failure, undefined F1 or final custody veto. A new measured F1 still
requires the full frozen evaluation and independent POST.

## Reviewed byte identities

Unchanged source/narrative reads; no source custody claim is made for unexamined
attempts or the active experiment.

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| scripts/run_gold11_mesh_development_v5.py | 153950 | 7c8a2ca60ea7671dd1c2b56339d6c632e70e00df7e6081fd18cb421a6678121f |
| scripts/gold11_owned_process_supervision_v1.py | 17073 | 1b90b86ef77510e32d72d3e08e85417860a8fb06ba0ad3294274a83bfb65ba74 |
| docs/research/2026-09-11_gold11_mesh_development_v5_protocol.md | 11780 | 5aa1b7c0bcc54f9b6eb2a41d000ed9a8c7e7460ea95b74b25b47d37a8027d874 |
| docs/research/2026-09-10_gold11_mesh_development_v3_result.md | 12507 | 9f35217c7bfd318b03139e2818383bf8e86e896be21745dab5e0cbccc0dd7afd |
| docs/research/2026-09-10_gold11_failure_factor_audit.md | 17528 | 4d511b35b1640df437ccef3b4b190ac1a2911139ed72424ea894aabbca1ef5e2 |

