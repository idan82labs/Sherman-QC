# Production line physical accuracy protocol (v1, 2026-10-17)

**Status: written before any physical truth exists, and frozen before any physical case is
registered.** Plan of record: `docs/production/PRODUCTION_LINE_PLAN.md` §5, §8 ("physical truth
campaign: the protocol is written, but execution needs the shop") and §9. Custody and evidence
rules: `AGENTS.md`. Tooling: `domains/production/benchmark/` (case registry and evaluator) and
`domains/production/engine/eval/` (E4 freeze ledger, gate documents).

This protocol does **not** authorize any physical data access by itself (section 10).

## 1. Purpose and claims

The protocol is the only route by which the line's numbers on real parts may be quoted as
accuracy. It supports exactly one decision: **may the line enter a supervised shadow pilot**
for a set of SKU families? In a shadow pilot the line runs next to the shop's normal inspection,
and its verdicts are recorded and compared but never acted on.

It cannot support:
- autonomous accept or reject;
- leaving operator-assist mode (results stay `OPERATOR_ASSIST`, `NOT_AUTHORIZED`);
- promotion evidence for any engine change;
- claims about families that were not evaluated.

Engineering gates G1-G4 (synthetic, and consumed development parts) are a separate ledger. Passing
them is a precondition to start the campaign, not a substitute for it. The synthetic line scorecard
(`apps.production_line.ops.synthetic_scorecard`) is engineering capability only.

## 2. Quantities and conventions

These are the same as the line (ENGINE_IO §1).

- **Flange length** per side: from the OML virtual sharp to the far boundary, either a FREE_EDGE
  (farthest material crossing) or an ADJACENT_BEND (OML to OML). Measured inside the flange plane,
  perpendicular to the bend, in mm.
- **Included angle** between the flanges, in degrees; 180° is flat.
- **Bend-line position**: the OML sharp in the datum frame, in mm. It is only scored where the recipe
  tolerances it.
- **Backgauge length** (incomplete parts): from the just-formed bend's OML to the far boundary
  across still-flat material.

Registry keys: `flange_a_mm`, `flange_b_mm`, `included_deg`, `bend_line_position_mm`, per
`bend_uid` (case_registry `QUANTITY_KEYS`). Side A and B follow the contract's side naming. The
registration form shows the recipe's side labels from the onboarding view, so the measurer never
guesses them. Each measured value is registered with the released recipe's nominal and
tolerance for that quantity (`nominal`, `tol_plus`, `tol_minus`), copied from the pinned recipe
version at registration; the registry refuses a measured value without them.

## 3. Truth instruments and uncertainty

| Quantity | Primary instrument | Required expanded uncertainty U (k = 2) | Acceptable alternative |
|---|---|---|---|
| Flange length | CMM, touch probe; the OML is constructed from fitted outer-surface planes, 3 stations along the bend at 10 / 50 / 90 % | ≤ 0.02 mm | height gauge on a surface plate with a calibrated angle block and OML computed from the measured outer surfaces, U ≤ 0.05 mm; calipers are **not** acceptable |
| Included angle | CMM plane fit on both flanges (same stations) | ≤ 0.05° | digital protractor / bevel on a surface plate, U ≤ 0.2°, at least 3 readings per bend |
| Bend-line position | CMM in a datum frame built from the recipe's datum panel (plane, two free edges or holes) | ≤ 0.03 mm | none |
| Operator verdict (`operator_verdict`) | the shop's existing go/no-go gauge or inspection sheet | IN/OUT only | used only for part-level confusion, never for errors |

Rules:

- **Test uncertainty ratio.** U ≤ 1/4 of the characteristic's tolerance band half-width (TUR ≥ 4:1).
  A characteristic that cannot meet this is registered with its real U, and the evaluator then
  classifies it INDETERMINATE more often. It is never dropped.
- **Gauge R&R before the first registration**: 2 measurers × 3 repeats × 5 parts per
  instrument and method. %GRR ≤ 10 % of tolerance is required, ≤ 30 % is acceptable with the owner's
  sign-off, and anything worse is refused.
- **Record per value**: instrument id, calibration date, method, measurer and temperature. Measure
  at 20 ± 2 °C after 1 h of soak.
- **Blindness.** The measurer never sees the station screen, the PDF or any engine output for that
  part before the truth row is registered. The registry timestamps prove the order: the truth
  `registered_at` must precede the job's `finished_at`, or a second person must attest that the
  output was not shown.
- **Exclusions are declared before the engine output is viewed**: an instrument fault, a damaged
  part, or a part not of the recipe. They are recorded as a registry `note`. A case is never
  excluded after its result has been seen.

## 4. Sample design per SKU family

A **family** is the profile class, plus the thickness band (≤ 1.5, 1.5-3, > 3 mm) and the
material group. Profile classes: L, U/channel, Z, hat, channel with returns, two-direction/box
with tabs, panel with holes and notches.

Per family in the evaluation split:

- at least **5 distinct SKUs** (all, if fewer exist), with at least **3 parts per SKU** from at least
  2 production lots;
- at least 1 complete scan per part, plus **3 reload repeats of one part per SKU** (unclamp,
  re-fixture, rescan) for repeatability;
- at least 2 SKUs scanned mid-sequence (stage 0, stage 1 and one middle stage) for formed/unformed
  state and backgauge length;
- nonconforming parts: every naturally out-of-tolerance part found, plus deliberately mis-bent
  parts where the shop can make them (angle off by 1.5-3× its tolerance, a backgauge off by 1 mm),
  so that each family has truth-OUT characteristics to test missed OUT and false accept.

Grouping:

- `split_group` = the SKU. All revisions, repeats and CAD variants of one SKU are in one group and
  never split across development and evaluation (roadmap M5).
- `sku_family` = the family label above. Reports are per family, plus the worst family.

Size: to bound the **false accept rate below 5 % at 95 % confidence with zero observed**, an
evaluation needs **at least 59 truth-OUT characteristics** (rule of three), pooled over the families
it claims. It needs at least 15 per family to report a per-family false accept. Smaller campaigns
report the counts and the exact (Clopper-Pearson) bound, and claim nothing more.

## 5. Metrics and denominators

Computed by `domains.production.benchmark.evaluator.aggregate` over every registered evaluation
case. Errors are **reported actual − truth**.

- **Errors**, per quantity kind: n, bias, MAE, P95 |error| and max |error|.
  - Over decided characteristics (PASS/FAIL), and separately over all MEASURED characteristics
    (REVIEW and undecided values included).
  - Both views are always reported side by side.
- **NOT_MEASURED rate**: NOT_MEASURED plus missing plus failed jobs, over every registered truth
  quantity that the result does not mark NOT_APPLICABLE. **Failed jobs stay in the denominator.**
- **Characteristic confusion**: truth IN / OUT / INDETERMINATE / UNKNOWN against reported IN / OUT /
  REVIEW / UNDECIDED / NOT_MEASURED / NOT_APPLICABLE.
  - The truth class comes from the truth value against the recipe's nominal and tolerance, with the
    truth's own U as a guard band.
  - The nominal and tolerance are the snapshot stored in the registry row when the case is
    registered (`nominal`, `tol_plus`, `tol_minus`, all required for measured truth; an explicit
    null tolerance means the recipe has none, and the truth class is then UNKNOWN). They are never
    taken from the engine's result. A failed job, a dropped characteristic or an engine-side
    tolerance therefore cannot change a truth class or move a case out of a denominator.
  - INDETERMINATE (truth within U of a limit) is reported, never counted as right or wrong.
- **Part level**: CONFORMING (every registered quantity truth IN) / NONCONFORMING (any truth OUT, or
  wrong or mirrored part) / UNDETERMINED, against the summary state.
- **Repeatability**: P95 |Δ| between reload repeats of one part, per kind.
- **Runtime**: P50 / P95 / max duration and max peak RSS per job.
- Every rate is reported as numerator / denominator. An empty denominator is "n/a", never 0 or 100 %.

## 6. Conformance decision table

| Truth \ line | PASS (green) | FAIL (red) | REVIEW (amber) | UNKNOWN / NOT_MEASURED (grey) |
|---|---|---|---|---|
| IN | correct accept | **false reject** (cost: rework/scrap of a good part) | review cost | coverage cost |
| OUT | **false accept: critical** | correct reject | review (acceptable; counted as missed OUT) | coverage cost (counted as missed OUT) |
| INDETERMINATE | reported only | reported only | reported only | reported only |

At part level, a **green summary on a NONCONFORMING part is a critical false accept.**

## 7. Thresholds for a supervised shadow pilot (go / no-go)

All must hold **overall and in every claimed family**. A pooled number never hides a failing
family.

| Metric | Threshold |
|---|---|
| Characteristic false accept (PASS on truth OUT) | 0 observed, with ≥ 59 truth-OUT characteristics in total (upper 95 % bound ≤ 5 %) |
| Part false accept (ALL_MEASURED_IN_TOLERANCE on a nonconforming part) | 0 |
| Characteristic false reject (FAIL on truth IN) | ≤ 2 % |
| Flange length, decided | abs bias ≤ 0.10 mm; P95 abs error ≤ 0.30 mm (family ≤ 0.40 mm) |
| Included angle, decided | abs bias ≤ 0.20°; P95 abs error ≤ 0.50° (family ≤ 0.60°) |
| Bend-line position, decided | P95 abs error ≤ 0.50 mm |
| Uncertainty calibration | abs(actual − truth) ≤ √(U² + U_truth²) on ≥ 90 % of MEASURED characteristics |
| NOT_MEASURED rate | ≤ 15 % overall; ≤ 25 % in the worst family |
| Decisiveness | ≥ 80 % PASS/FAIL where abs(true deviation) ≤ 0.5·tol or ≥ 1.5·tol |
| Incomplete parts | the formed/unformed state matches the measurer's record for every bend; no NOT_YET_FORMED bend is FAIL |
| Repeatability | P95 abs(Δ) between reloads ≤ 0.30 mm and ≤ 0.30° |
| Runtime | P95 ≤ 45 s; 0 crashes; every failure is an explicit UNKNOWN or INVALID |

A shadow pilot that starts still runs under operator assist. Moving beyond it needs a new protocol
version and an owner decision. Thresholds are frozen with this document and are never changed
after the first evaluation case is registered.

## 8. Development vs evaluation split; no tuning on held-out families

- **The family split is assigned before any evaluation scan exists.** Roughly one third of the
  families, including one with staged scans, go to EVALUATION; the rest are DEVELOPMENT, which
  engineering may use for iteration.
- The split manifest (family → split, with the SKUs known at the time) is a frozen file. Its sha256
  goes into the freeze ledger (section 11).
- Engine, config, thresholds and recipes are frozen (engine bundle id, engine config sha256,
  recipe sha256) before the first evaluation case is scored. **One** frozen configuration is scored
  on all evaluation families.
- Nobody who changes the engine sees evaluation results before the report is written, apart from the
  aggregate verdict. Failures are not tuned against: a change after a failed evaluation is a new
  hypothesis, and it needs new evaluation parts.
- Filenames, serials, metadata and counts are never engine inputs (amendment 4). The engine
  receives the opaque staged scan only.

## 9. Consumed-case policy

- A case is **consumed** as soon as any engine output for it is seen by anyone involved in
  engine development, or it is used for any threshold, feature or configuration choice. A consumed
  case never returns to an unseen split.
- **Every part with a scan currently on the development machine is CONSUMED**
  (`docs/production/BENCHMARK_V2.md`: FRESH has 0 scans on disk). Gold11, Gold37, physical15,
  Eyal-50 delivery files and all `source-assets` scans are consumed or excluded.
- Consumed results are reported only as "consumed development (engineering ledger)".
- After an evaluation run, its cases are consumed. The next evaluation needs newly acquired parts:
  new serials of evaluation-family SKUs, or new evaluation families.
- The registry is append-only and hash-chained (`case_registry`). Its head sha256 at freeze time
  and at report time is recorded, so edits, deletions or reordering are detectable.

## 10. Authorization gates (AGENTS.md)

Before **any** physical scan or truth measurement is opened, stat-ed, hashed or registered under
this protocol:

1. The owner gives a written, dated authorization. It names the cohort (families, SKUs, number of
   parts), the purpose (this protocol, version and sha256), who may access the data, and the
   storage location.
2. Storage stays outside the repository: the registry and the evaluation data root live under the
   line's data root or a separate absolute path. `CaseRegistry` refuses repo paths and protected
   names.
3. An independent reviewer confirms that the freeze (section 11) precedes access, and that the split
   and thresholds are unchanged.
4. HOLDOUT10, sealed truth, the opaque N001-N014 set, EHO4, email attachments and any
   authorization or one-shot control artifacts are never touched by this protocol. A terminal
   one-shot stays terminal.
5. One heavy inspection at a time; process-tree RSS < 16 GiB; the line's 180 s / 4 GiB per scan.
6. Engineering capability and physical generalization are kept in separate ledgers, and the report
   says which one each number belongs to.

## 11. Freeze procedure

Done by the integrator or owner, not by an implementing agent:

```bash
sha256sum docs/research/2026-10-17_production_line_accuracy_protocol.md
python -m domains.production.engine.eval freeze --ledger <ledger> --label "physical-protocol-v1" \
    --file accuracy_protocol=docs/research/2026-10-17_production_line_accuracy_protocol.md \
    --file split_manifest=<split manifest> --engine-config <config> --case-list evaluation=<case list>
python -m domains.production.benchmark.case_registry verify --registry <registry>   # head_sha256 recorded
# at evaluation time, against the recorded head (detects rows removed from the tail):
python -m domains.production.benchmark.case_registry verify --registry <registry> --expect-head <head_sha256>
```

Record the following outside the ledger, as the E4 freeze rules require (the ledger anchor):
- the ledger's first FREEZE `entry_sha256`;
- the engine bundle id;
- the registry head sha256.

## 12. Report

One report per evaluation run. It contains:
- the frozen hashes (protocol, split, engine bundle, config, registry head);
- per family and overall: every section 5 metric with its numerator and denominator, and every
  section 7 threshold with PASS / FAIL;
- the worst family;
- every critical false accept and every abs(error) > 2× tolerance, with the zoomed scan section
  and the truth measurement sheet;
- exclusions declared before viewing;
- the go / no-go decision for the shadow pilot, signed by the owner.

The report is labelled **physical evaluation, protocol v1**, with the cohort named. It never
mixes in consumed development numbers.
