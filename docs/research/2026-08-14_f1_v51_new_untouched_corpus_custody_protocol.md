# V51 new untouched corpus custody protocol

Status: frozen design; no corpus selected; no detector invocation  
Token: `V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED`

## Non-overlapping roles

- **Corpus custodian:** has never viewed V51 predictions or Gold37 truth. Before
  seeing candidate cases, this person signs the acquisition and sampling rule,
  stages only opaque scan bytes for the detector, and keeps the identity/family
  map outside the detector sandbox.
- **Annotator A and annotator B:** independently mark bend-line endpoints from
  scan geometry only. They cannot see V51 outputs, each other's work, expected
  bend counts, CAD, drawings, filenames, family identity, or benchmark results.
- **Adjudicator:** resolves every non-identical annotation set without V51
  outputs. The adjudicated truth and family map are sealed before invocation.
- **Mechanical launcher:** may run the sealed command after preflight. Because
  this role was exposed to the Gold37 incident, it may not choose cases,
  exclusions, families, annotations, or adjudications and may not inspect any
  prediction before both primary and replay sets are sealed.
- **Read-only scorer/auditor:** opens truth only after the 60 detector results,
  digests, resource manifests, and prediction-only renders are immutable.

One person may not hold two roles above except that the scorer may also perform
the post-score visual audit. The custodian, annotators, and adjudicator must be
independent of the V51 development/incident-exposed operator.

## Acquisition precommit

Before selecting a case, the custodian records and signs:

1. scanner model, firmware, configuration, calibration identifier and date;
2. fixture/orientation policy and capture procedure;
3. reconstruction and cleaning software plus exact configuration hashes;
4. static acquisition-failure criteria that do not mention bend count,
   difficulty, V51, visual preference, or expected output;
5. the deterministic selection rule: first 15 eligible families, then first two
   eligible scans in acquisition-time order from each family;
6. a statement that membership is created after the Pro ruling; or, for a
   previously sealed candidate, its earlier cryptographic seal and access log;
7. an exclusion ledger containing every rejected acquisition and only its
   precommitted acquisition-failure reason.

The final corpus is exactly 30 scans from exactly 15 families, two scans per
family. The detector receives only `O001`–`O030` and geometry staged under an
opaque filename. No expected count, part number, filename, family token, source
folder, or acquisition notes cross the sandbox boundary.

## Disjointness and sealing

The custodian performs a hash-only comparison against the supplied development
and Gold37 hash denylist. Only collision/no-collision is returned. A collision
removes the scan according to the already-signed deterministic rule; it is not
replaced by visual preference. The custodian seals:

- ordered opaque scan manifest and SHA-256 for each staged geometry;
- private identity map and 15-family map outside the detector sandbox;
- annotator A and B files, adjudication log, final truth manifest, and their
  SHA-256 values;
- access log proving that no V51 developer or incident-exposed operator viewed
  geometry, identity, truth, or diagnostic renders before membership was fixed.

## Execution and opening order

1. Run the source-free preflight and require every check to pass.
2. Freeze the generic runner, scorer, renderers, dependency record, CPU/thread
   environment, and canonical schemas before corpus assembly.
3. Run all 30 primary cases in opaque order. No score, inspection, early stop,
   or case-dependent action is allowed.
4. Run all 30 replay cases in the same environment and order.
5. Seal canonical semantic results, primary/replay hashes, scan hashes, point
   counts, compact-cache audits, elapsed times, process-tree RSS, job status,
   and prediction-only native-resolution renders.
6. Require 30/30 primary jobs, 30/30 replay jobs, and byte-identical semantic
   results for every case.
7. Only now may the read-only scorer open the final truth and family map. It
   writes one complete score transaction; no partial result is exposed.
8. Generate prediction/truth overlays and axis-normal cross-section stacks,
   then append one manual visual ruling for every case. No relabeling, waiver,
   matcher change, prediction edit, or V51 rerun is allowed.

The frozen render-admissible marker denominator is the 30 opaque cases. For a
case to be marker-correct, its render-admissible marker count must equal that
case's already-sealed adjudicated bend count. Consequently the `>=0.99` gate
requires 30/30 on this tranche; 29/30 is below threshold. This count does not
replace or modify the line matcher used for precision and recall.

Each append-only visual-audit row contains the opaque case ID; SHA-256-bound
paths for the prediction-only render, scored overlay, and axis-normal stack;
the render-admissible marker count; reviewer identity; and exactly one ruling
from `coherent`, `geometric_failure`, or `render_failure`.

## Terminal gates

The valid transfer gate requires at least 27/30 exact counts, precision and
recall each at least 0.90, every family recall at least 0.80, marker accuracy at
least 0.99, deterministic replay 1.0, 30/30 job success, at most 200,000 working
points, no forbidden serialized source geometry, peak process-tree RSS strictly
below 16 GiB, P95 latency at most 300 seconds, and a manual ruling for all 30
native-resolution overlays.

Any semantic miss terminally retires V51. A pre-truth artifact failure is
invalid and requires fresh adjudication; a post-truth artifact failure consumes
both V51 and this corpus. Runtime/main and production remain closed even after a
pass pending a separate promotion ruling.
