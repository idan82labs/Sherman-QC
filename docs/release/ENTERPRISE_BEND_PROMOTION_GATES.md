# Enterprise Bend Promotion Gates

## Status

F1 remains research-only. An eligibility checker is **not a release authority**.
The legacy aggregate checker `scripts/check_enterprise_promotion_gates.py` is a
historical/development diagnostic; even its `promotion_status=eligible` output
cannot authorize runtime promotion. Its new explicit marker is
`policy_status=LEGACY_AGGREGATE_DIAGNOSTIC_ONLY`.

The prospective, source-only
[`check_enterprise_promotion_gates_v2.py`](../../scripts/check_enterprise_promotion_gates_v2.py)
checks strict, paired, candidate-bound **evidence declarations**, using the fixed
numeric thresholds below. It does not read the referenced evidence bytes,
establish untouched truth, verify a deployed candidate, or authenticate reviewers.
Every result retains `production_use_allowed=false` and
`requires_independent_verification=true`. Historical reports are not upgraded.

Promotion requires independently verified candidate/evidence identity, prospective
protocol and metric-definition approval, reviewed physical qualification, and a
separate scoped release decision. A supervised shadow pilot retains operator
authority and is not autonomous acceptance. No shadow authorization is created
by this document or checker. Autonomous disposition also needs an approved risk
rule, release owner, monitoring and rollback; no acceptable harmful-error rate is
invented here. The API and worker now share a fail-closed legacy-status projection:
booleans and hash-shaped strings cannot grant either release mode.

The gate evaluates the visible, observed conventional-bend task. Raised form
features and rolled/rounded features are typed separately. Partial scans are
never scored against an inferred whole-part total.

## Required Evidence

Scope note — 2026-09-08: the user has prioritized
[CAD+scan offline engine correctness](../research/2026-09-07_production_generalization_roadmap.md#active-scope-amendment--user-confirmed-2026-09-08)
and deferred standalone zero-shot and floor work. PLY-only/uplift requirements
below are not blockers for that offline engineering milestone. This note does
not change the release thresholds/checker or grant release; applicability to a
later CAD-assisted-only release must be reviewed when that release is scoped.

| Area | Release threshold |
| --- | ---: |
| Typed truth | at least 150 reviewed scans across 75 parts |
| Locked holdout | at least 30 untouched parts |
| Family coverage | at least 30 reviewed scans per declared family |
| Instance truth | bend localization geometry on at least 95% of release scans |
| Blind PLY-only exact count | at least 90% |
| Blind CAD+PLY exact count | at least 95% |
| CAD uplift | at least +5 percentage points over PLY-only |
| Range containment | at least 98% |
| Render-admissible instance accuracy | at least 99% |
| Determinism | identical accepted result in 10/10 repeats |
| P95 end-to-end latency | at most 300 seconds |
| Job success rate | at least 99.5% |

## Non-Negotiable Contracts

- Expected counts, manual labels, part feature policies, and validation truth are
  scoring-only in blind benchmarks.
- A CAD+PLY result is blind only when `inference_mode=blind_cad_ply`.
- A result cannot be `render_verified` unless the accepted render artifact
  exists and all owned markers pass quality/admissibility checks.
- Display annotations must be supplied in `reference_mesh_raw`; the frontend
  does not infer coordinate transforms by matching bounding boxes.
- Cache identity includes scan path, scan content hash, schema version,
  detector configuration hash, and an explicit upstream research-semantics
  version for zero-shot results, atom graphs, and generated hypotheses.
- Legacy mail or corpus counts are not release truth until visually adjudicated.
- Blind-slice construction fails closed unless every release-eligible record has
  an approved manual/dual-review status and provenance, the visible-conventional
  semantic target, a valid count, and a resolved measured-scan source. CAD
  feature-history evidence is never an approved truth source.
- Numerical count protection by the control lane is not F1 ownership proof.

## Commands

### Source-only v2 contract verification

The public pure interface is `evaluate_promotion_gates_v2(evidence)`. Its report
separates `evidence_contract_status=eligible|blocked` from
`promotion_status=not_authorized`. See the entirely synthetic
[contract fixture and tests](../../tests/test_enterprise_promotion_gates_v2.py)
for the exact schema. The CLI accepts only explicit evidence/output paths and
exclusively creates a new output; it has no discovery, default input, threshold
override, `--allow-blocked`, or overwrite mode. Running it against physical or
protected evidence still requires the existing prospective authorization.

V2 binds source/configuration/model/semantics digests, protocol/evaluator identities
and the canonical declared scope into each section. Identity consistency is not
provenance verification. Declared families include absent families (zero coverage);
unknown families, duplicate scan IDs, mismatched part partitions, disjoint paired
cohorts, omitted failed jobs, and missing repeat evidence block eligibility.

Metric definitions requiring prospective review before any frozen evaluation:

- Exact counts and range containment use **all expected scans**. Failed/abstained
  results remain failures, not excluded cases. CAD uplift uses the same scan IDs
  and exact rational threshold comparison.
- `render_admissible_instance_precision` is distinct admissible matched reviewed
  truth-instance IDs divided by **all emitted owned instances**. Each prediction
  requires a marker record; duplicate truth ownership earns no double credit.
  It is precision, not localization recall or legacy exact-set accuracy. Separate
  render-evaluated scan coverage requires at least 150 scans. This interpretation
  of the 99% instance threshold is explicit, not a silent change to frozen metrics.
- Operational success and nearest-rank P95 include all expected lane jobs.
- At least ten accepted whole-cohort replays must have the semantic fingerprint
  of the evaluated paired results, not merely agree with each other.

These engineering definitions close aggregate-contract loopholes; they do not
replace native render review, calibrated geometry, reviewer provenance, or
physical measurement evidence.

### Historical commands — not an execution authorization

```bash
python3.11 scripts/build_bend_ground_truth_registry.py
python3.11 scripts/check_enterprise_promotion_gates.py \
  --truth-registry data/bend_corpus/annotations/ground_truth_registry.json \
  --ply-summary /path/to/blind_ply_summary.json \
  --cad-ply-summary /path/to/blind_cad_ply_summary.json \
  --operational-summary /path/to/operational_summary.json \
  --determinism-summary /path/to/determinism_summary.json \
  --output-json output/enterprise_promotion_gate/current.json
```

For historical v1 tooling, `--allow-blocked` only changes the process exit code;
it does not validate provenance or claim promotion readiness. Do not run the
historical corpus commands merely because they appear in this document.
