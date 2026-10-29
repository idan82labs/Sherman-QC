# V5 prospective resource allocation — 2026-09-11

Status: **engineering recommendation only; not frozen or released**. The native
attempt is still awaiting its accepted terminal POST. No actual completed native
time is used here. All seven timing literals in the current V5 runner remain
`None`; release remains false. No candidate, policy, scoring rule, input scope,
source test, native frozen source/PRE/archive or existing artifact is changed.

This note completes the numerical planning left open in the prior
[resource review](2026-09-11_gold11_v5_resource_planning_review.md). It uses only
named source and released narrative. No project import, test collection/run,
geometry, simulation, active artifact/private-cache, physical/raw input, manifest
or control-record access occurred. Only this new document is written.

## Conditional CPU rule and seven proposed limits

Let `N` be the **audited terminal native whole-worker CPU seconds**, including
the original/supplemental tests and their overhead. It is admissible for this
rule only when the full native route and required complete integrity evidence
are accepted, with complete retention and no resource/custody failure. The
original accuracy oracle may remain RED under the prospective diagnostic
amendment. A censored/killed worker, partial stage, SAFE_EARLY_REFUSAL, missing
ledger or mere native CPU cap is not `N`.

Use the released connected L-R registration-plus-V2-measurement reference
`1462.253261 + 1.437055 = 1463.690316` seconds as a floor, not a whole-native or
physical upper bound. Define `ceil60(x) = 60*ceil(x/60)` and choose uniformly for
both arms and all eleven cases:

| Existing limit | Proposed value, seconds |
| --- | ---: |
| `case_cpu_seconds`, C | `ceil60(1.25*max(N,1463.690316))` |
| `case_wall_seconds`, W | `L + 7*P + 60`, where `L=ceil60(11*C/10)+120` |
| `scorer_cpu_seconds`, Sc | 120 |
| `scorer_wall_seconds`, Sw | 180 |
| `parent_read_seconds`, P | 180 |
| `terminalization_reserve_seconds`, R | 2400 |
| `total_wall_seconds`, T | `600*ceil((360 + 22*(W+5*P+6) + 3600 + R)/600)` |

The 25% CPU increment is a prospective engineering allowance, not a statistical
confidence bound or a statement that physical cases are at most 25% harder.
`L` allows 10% wall/CPU divergence plus two minutes; the additional 60 seconds in
W is case administration headroom. These margins concern different clock
scopes, not improved scientific accuracy. Round with the written rational
arithmetic, then independently verify the final integer literals.

No sample-count multiplier is applied: 32768 CAD samples do not multiply every
finite optimizer query. Conversely, neither the native proxy nor fixed internal
work caps predict physical triangle/support complexity. Every budget failure
remains a retained refusal/failure with its denominator intact.

## Full-opportunity arithmetic and maximum commitment

Use the source-review accounting variables, without adding runtime parameters:

    g = 60
    H0 = 360
    Bs = 3600
    Bcase = W + 5*P + 6
    T >= H0 + 22*Bcase + Bs + R

W allows four preparation and three accepted postprocessing P-blocks plus L
and g. Bcase additionally reserves the conservative failed/late path: three
post-worker P-blocks, two terminal publications, five-second child cleanup and
one-second process sampling. This intentionally overcounts some alternative
paths; it prevents borrowing the last cases' opportunities for earlier overhead.

Under the existing native ceiling, an admitted `N <= 2520` gives the following
planning endpoints. The right column is a hypothetical cap-near completion,
**not an observed native time and not permission to substitute 2520 for N**.

| Quantity | Connected-floor allocation | Maximum if admitted N ≤2520 |
| --- | ---: | ---: |
| C | 1860 | 3180 |
| L | 2220 | 3660 |
| W | 3540 | 4980 |
| Bcase | 4446 | 5886 |
| 22×Bcase | 97812 | 129492 |
| Unrounded H0+22×Bcase+Bs+R | 104172 | 135852 |
| T, rounded up to ten minutes | 104400 | 136200 |
| Campaign capacity commitment | 29 h | 37 h 50 min |

The execution source preflight has a separate P before the campaign clock.
Thus this option's maximum planned local execution commitment is **37 h 53 min**
including that three-minute preflight, excluding earlier freeze/PRE/source-gate
work. This is a maximum allocation, not predicted elapsed time or a hard OS
termination guarantee. Actual work may finish substantially sooner. If that
commitment is unacceptable, decide before release; do not silently reduce T,
starve later ordered cases, enlarge a started limit or retry a timed-out case.

The minimum and maximum direct-leaf CPU allowance sums are 41040 and 70080
seconds (`22*C+Sc`). They are not whole-campaign CPU limits: outer/coordinator
hashing, reconstruction, inventory and supervision have separate CPU costs.

## What the non-native allowances buy

These are finite engineering allocations to source-defined work. They are not
claims that every operation reaches its individual deadline, measured
throughput, or new per-phase enforcement implemented in the runner.

**P = 180 seconds, including the whole cohort.** In
`score_after_terminal`, one P encloses the complete
`reconstruct_all_terminal_inputs` call and terminal equality checks, before
reference access. That means 22 terminal receipts, 22 compact projections and
up to 22 original RESULTS, not 180 seconds per RESULT. A clean cohort's original
RESULT byte ceiling is `22*128 MiB = 2816 MiB = 2.75 GiB`; JSON decoding, schedule/
contact reprojection and canonical comparisons add CPU. Three minutes is an
explicit engineering choice for this whole block, not evidence that every
maximum-size report finishes within it. Its ordinary timeout must remain fatal
to scoring. No additional maximum-report experiment is required by this choice.
The same P also bounds source preflight, individual reads/publications and other
parent blocks, with nested deadlines preserving stricter absolute clocks.

**Sc/Sw = 120/180 seconds.** The scorer leaf reads at most a 192 MiB private
request and publishes at most 32 MiB. It receives the already reconstructed
compact inputs, not raw meshes or a native workflow. With 71 total reference
lines and at most 512 predictions per case, the unchanged matcher evaluates at
most `2*512*71 = 72704` reference/prediction pairs across both arms, then performs
at most 22 bounded one-to-one assignments and the small case/family aggregation. The
two-minute CPU allocation is proportionate to this pure task; the additional
minute is leaf startup/read/serialization and wall headroom. It is not inferred
from the old empty-output score timing. Coordinator reconstruction and final
parent revalidation are not hidden inside Sc.

**H0 = 360 seconds = 2P.** Reserve one P-equivalent for outer dispatch/process
startup and one for coordinator source/control authentication, setup and loop
administration. The coordinator's own elapsed clock starts later than the
outer campaign clock; H0 explicitly funds that offset. Per-leaf imports and
native execution are charged to C/L, not here. The separate outer pre-STARTED
source check remains outside T.

**Bs = 3600 seconds.** Allocate 18 P-equivalents for the complete post-inference
coordinator/scoring work, plus Sw, six seconds cleanup/sample and 174 seconds
administration: `18*180 + 180 + 6 + 174 = 3600`. The source grouping is:

| Work in/around `score_after_terminal` | P-equivalents |
| --- | ---: |
| Pre-/post-scorer bundle checks, two fixed-catalog reads, selected-reference publication | 5 |
| All-22 reconstruction and durable terminal equality | 1 |
| Scoring-request construction, structural preflight and canonical encoding | 2 |
| Private request publication and scorer dispatch | 2 |
| Scorer receipt publication and retained-score/retention-receipt block | 2 |
| Retained score read, decoding and transport checks | 2 |
| Enriched report construction and durable publication | 2 |
| INFERENCE_TERMINAL and COORDINATOR_COMPLETED publications | 2 |

These are accounting slots, not a claim that the source has 18 independent
P-sized alarms. Several Python/source-verification operations share or lack a
local phase alarm and remain covered by the coordinator's absolute outer
window. Bs is intentionally much larger than Sw alone. It adds no inference,
second scorer, raw-support refit or reference access before all 22 terminals.

**R = 2400 seconds.** Reserve the final 40 minutes after coordinator work for:

| Outer finalization work | Seconds |
| --- | ---: |
| Final current/frozen source and control audit, P-bounded | 180 |
| All-22 terminal/artifact validation, original-byte reprojection and scorer-request/report consistency | 600 |
| Three full campaign inventory passes, 300 each | 900 |
| COMPLETED/FINAL_GATE publication, canonical metadata and clock checks | 360 |
| Owned child cleanup and final sample | 6 |
| Remaining OS/administrative margin | 354 |
| Total | 2400 |

On the all-clean path, final parent validation reads each original RESULT twice
(retention validation and independent reconstruction), and three inventory
passes read it three more times: 110 RESULT reads / at most 13.75 GiB. Compact
projections, receipts, sources, logs and score/control publications are
additional. The earlier coordinator's 2.75 GiB pass belongs to Bs, not R. The
600/300-second groups are allowances, not observed bandwidth. Terminal fill
after an abort uses remaining time and cannot promise a complete successful
campaign; incomplete final custody stays a veto.

## Freeze conditions and preserved scope

This allocation funds all 22 opportunities **conditional on its component
allowances being respected**. The runner only checks enough remaining campaign
time for W before a case; final PRE must independently check the stronger T
arithmetic above. P/R are not interchangeable. Synchronous OS calls, process
creation, signal delivery and sampled RSS prevent a universal completion proof.

Keep the existing serial schedule and one numeric thread, 8 GiB leaf, 4 GiB
coordinator, 12 GiB whole-tree and 2 GiB scorer caps. Keep all 11 cases, six
families, 71 references, both arm definitions, direct endpoint matcher,
whole-case refusals and final custody checks. No physical calibration,
historical comparability or stronger contact-carrier meaning follows from a
larger time allowance.

After the native POST admits the full route/integrity and provides N, root may
choose this finite rule or explicitly choose a different prospective policy.
Then compute/freeze all seven actual values and final runner/test pins before
the 85-outcome runner source gate, followed by the independent exact physical
PRE. This document does not release either gate or a physical run. It does not
alter the active native attempt or authorize its retry.

## Source checkpoint

The reviewed runner is `scripts/run_gold11_mesh_development_v5.py`, 153950 bytes,
SHA256 `7c8a2ca60ea7671dd1c2b56339d6c632e70e00df7e6081fd18cb421a6678121f`.
The prior resource review is 13499 bytes, SHA256
`58b9225883167daaad1a6fbcdc536957b4256082682270df8235e756b18cfff4`.
Relevant source scopes are `supervise_case`, `reconstruct_all_terminal_inputs`,
`score_after_terminal`, `scorer_worker`, `internal_coordinator`,
`verify_parent_artifacts`, `verify_campaign_inventory` and `execute`.
The shared supervision source remains SHA256
`1b90b86ef77510e32d72d3e08e85417860a8fb06ba0ad3294274a83bfb65ba74`.
Matcher counts were derived from the unchanged contact scorer's two `_lane`
calls and `evaluate_f1_against_manual_gold.match_lines`, not actual Gold data.
