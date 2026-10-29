# Sherman QC enterprise F1 promotion checkpoint — 2026-07-15

## Decision

Do not promote F1 to runtime/main.

The research lane now produces render-admissible count equality on 11 of the 13
verified PLY cases and exact, visually aligned final marker sets on 10 of 13.
Case 07 is count-exact but remains visually unresolved, so it is not credited as
a hard visual pass. This is a useful semantic-research checkpoint, not release
acceptance. The product acceptance layer still classifies every current PLY
render as debug/review-only, so release-level visually accepted exact accuracy
remains 0/13.

The current enterprise gate report is:

```text
promotion status: blocked
passed gates:     4 / 16
blocked gates:   12 / 16
```

Passing gates are the explicit blind CAD subprocess contract, 10/10
determinism, P95 latency, and job reliability.

## Corrected blind benchmark evidence

### PLY-only F1

Fresh schema-v5 research inputs were rebuilt for all 13 cases. Cache identity
contains scan path, scan content hash, detector configuration hash, cache
schema, and explicit upstream research-semantics version
`2026-07-15.1`. Full source points are not serialized.

```text
jobs:                              13 / 13 successful
accepted exact accuracy:            0.000000
accepted range containment:         0.538462
render-admissible exact accuracy:   0.846154 (11 / 13)
hard-visual exact accuracy:         0.769231 (10 / 13)
release visual exact accuracy:      0.000000
P95 render-inclusive latency:      43.247 s
worst per-job RSS:                  1,787,314,176 bytes (1.66 GiB)
```

The ten hard-visual research passes are cases 01, 02, 03, 04, 05, 09, 10,
11, 12, and 13. Cases 06–08 remain visual failures. Case 07 now reaches eight
markers after truth-free suppression of a contained same-pair alias, but a
second cross-pair collinear ambiguity prevents visual credit. The current
count/visual split is recorded in
`output/experiments/f1-span-alias-strict-regression-20260715/zero-shot-11count-10visual-checkpoint.json`.

### CAD+PLY

The prior CAD+PLY benchmark was not sufficiently fail-closed because scoring
counts and scan-state labels crossed the inference subprocess boundary, even
though the runtime ignored expected-count overrides in blind mode. The runner
now passes `state=unknown`, no expected counts, no drawing bend angles, and no
schedule hint. Scoring truth is attached and evaluated only after the child
process returns. The promotion checker requires explicit proof of this boundary
for every case.

Corrected evidence:

```text
jobs:                         12 / 12 successful
exact-count accuracy:          0.250000 (3 / 12)
explicit blind contract rate:  1.000000
P95 pipeline latency:         46.335 s
```

The former CAD uplift claim is withdrawn. The uplift gate now also requires
both benchmark lanes to contain the minimum aligned release sample size; it
cannot pass from small or differently evaluated samples.

## Determinism and CI

Schema-v5 determinism was rebuilt rather than inherited from schema v4:

```text
independent process runs: 10
unique result hashes:      1
repeat rate:               1.000000
```

Repository validation:

```text
Python:              959 passed, 1 skipped
frontend ESLint:     pass, zero warnings
frontend production build: pass
manual-assistant UX audit: 10.0 / 10.0 across desktop and mobile
```

The frontend consumes bend annotations only in `reference_mesh_raw`; backend
display objects explicitly declare that frame. Unsupported raw CAD/scan lines
remain blocked by the display gate.

## Truth and corpus audit

Authoritative registry rebuild:

```text
records:                    35
release-eligible scans:     13
release-eligible parts:     10
locked holdout parts:        0
instance geometry coverage:  0%
```

The regenerated 13-case PLY and 12-case CAD+PLY slice manifests are byte-for-
byte identical to the checked-in manifests, apart from registry generation
timestamp/source-root metadata.

Local capacity contains 38 unique geometries. Thirteen are release eligible
and 25 are in a prediction-hidden independent-review queue. Those 25 cannot be
promoted automatically because their visible conventional-bend counts,
measured-scan provenance, and `reference_mesh_raw` instance geometry have not
been adjudicated. Even optimistic acceptance of all local candidates would
still leave 112 scans missing from the 150-scan release threshold.

The Eyal 50 package contains a deterministic 30-part holdout split and decoded
SLDPRT feature-history evidence, but no measured scans or independent visible-
bend ground truth. The CAD feature-history counts are explicitly metric-
ineligible and semantically non-equivalent to visible conventional bend count.
Its frozen inference artifacts use historical cache schema v4; they remain a
pre-truth research baseline, not current schema-v5 promotion evidence.

A fresh complete Gmail audit on 2026-07-15 reconfirmed the delivery boundary:
the exact sender query returned 51 messages and 50 unique packages (45
SharePoint-link messages and six Gmail-attachment messages, including one
duplicate/retry thread). No later message exists after 2026-07-06, and targeted
sender searches found no PLY/scan terminology and no bend-count or ground-truth
language. The extracted corpus remains exactly 50 PDFs, 50 SLDPRTs, and 50
STLs. The STLs remain unadjudicated CAD-derived geometry candidates, not
measured scans and not ground truth.

## Remaining engineering failures

- Case 06: the strongest scan-only plane lattice supplies four defensible
  transitions versus truth six. Proposed ridge additions are face-interior
  false lines.
- Case 07: same-pair contained alias OB8 is now suppressed in favor of dominant
  scan-supported strip OB4, producing count eight without truth. OB1 and OB3
  remain geometrically alias-like but select different inferred flange pairs;
  merging them produces seven and exposes a different missing bend. Count exact
  is therefore not yet a hard visual pass.
- Case 08: deterministic plane intersections recover six visible transitions
  versus truth seven; no supported seventh line has emerged.
- Accepted-count selection still leaves non-singleton/control-guarded results,
  so accepted exact PLY accuracy is below the marker-only research score.

## Material questions requiring professor guidance

1. What object-level prior distinguishes a true missing internal bend from a
   surface-attached perimeter/ridge line when both have two-sided local support?
2. Should maximal bend instances be defined by flange pair plus maximal axis
   interval, or by a sparse junction-path/MDL model that can split observed
   support gaps?
3. Are case-07 OB1 and OB3 two physical bends separated by a real intermediate
   flange, or one bend duplicated by flange segmentation drift? If they are one,
   where is the missing eighth bend?
4. What minimum per-instance annotation geometry is sufficient to learn and
   validate these factors while keeping count truth outside inference?

## Authoritative paths

- Promotion gate: `output/enterprise_promotion_gate/current-20260715.json`
- PLY summary: `output/enterprise_benchmark/blind-ply-schema5-span-alias-20260715/summary.json`
- PLY operations: `output/enterprise_benchmark/blind-ply-schema5-span-alias-20260715/operational_summary.json`
- CAD+PLY summary: `output/enterprise_benchmark/blind-cad-ply-20260715-v4/summary.json`
- Determinism: `output/enterprise_benchmark/f1-determinism-schema5-span-alias-20260715/determinism_summary.json`
- Combined operations: `output/enterprise_benchmark/operational-summary-schema5-span-alias-20260715.json`
- Current count/visual checkpoint: `output/experiments/f1-span-alias-strict-regression-20260715/zero-shot-11count-10visual-checkpoint.json`
- Professor hard-case packet: `output/experiments/f1-10of13/professor-hard-cases-20260715/adjudication_packet.json`
- Local review queue: `output/migration/local-scan-adjudication-v1/review_queue.json`
- Local reviewer template: `output/migration/local-scan-adjudication-v1/reviewer_template.json`
