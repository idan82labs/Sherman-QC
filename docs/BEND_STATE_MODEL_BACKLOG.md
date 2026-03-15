# Bend State Model Backlog

## Goal
Drive the promoted bend-count MAE from the current baseline toward sub-0.5, and eventually toward sub-0.2, by replacing heuristic local bend completion logic with a structured state model.

Current promoted baseline:
- source: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/full_ready_regression_20260311_metrics/summary.json`
- `mae_vs_expected_total_completed = 1.143`
- `mae_vs_expected_completed = 0.062`
- `full_scan_exact_completion_rate = 0.882`

## Phase 0: Data and reproducibility hardening

### 0.1 Remaining truth completion
Deliverables:
- complete missing-data chase from `confirmation_queue`
- especially partial completed-bend counts and bend IDs where available

Acceptance:
- >= 90% of scans have trusted expected totals
- >= 70% of partial scans have trusted completed counts

### 0.2 Nondeterminism audit
Priority family:
- `49024000_A.ply`

Tasks:
- rerun same scan multiple times
- log alignment seed, chosen refinement path, and final completed count
- measure run-to-run variance

Acceptance:
- repeated runs produce stable completed-bend counts on the same input

## Phase 1: State separation in runtime schema

### 1.1 Introduce internal state fields
Runtime additions per bend:
- `physical_completion_state`
- `observability_state`
- `assignment_source`
- `assignment_confidence`
- `visibility_score`

Files likely affected:
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/feature_detection/bend_detector.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/bend_inspection_pipeline.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/frontend/react/src/services/api.ts`

Acceptance:
- API/report schema supports these fields
- current UI labels are derived from them without regression

### 1.2 Persist observability features
Features to persist:
- side support counts
- local support count
- calibrated continuous visibility score `rho_i`
- expected visible fraction from CAD
- realized local coverage estimate
- local density
- residual / planarity metrics

Acceptance:
- corpus outputs expose these consistently
- evaluator can summarize them by family and by failure mode
- `UNOBSERVED` / `PARTIALLY_OBSERVED` can be derived from `rho_i` and posterior confidence rather than ad hoc rule chains

## Phase 2: Local candidate modeling

### 2.1 Explicit candidate generation
Tasks:
- generate local evidence candidates per bend neighborhood
- allow explicit `null` assignment
- store candidate scores and assignment rationale
- define confusable-bend groups for exclusivity

Acceptance:
- no hidden one-shot mapping from local evidence to bend
- per-bend candidate set visible in debug/evaluation output
- candidate sets can be grouped into confusable evidence atoms / cliques for later exclusivity

### 2.2 Observability gate before completion decision
Tasks:
- estimate `rho_i` first
- suppress physical completion claim when visibility is weak

Acceptance:
- `UNOBSERVED` and `PARTIALLY_OBSERVED` become output states derived from visibility/posterior confidence, not ad hoc fallback labels

## Phase 3: Structured inference

### 3.1 Unary potentials
Tasks:
- fit learned conditional unaries from the corpus evidence vectors
- first implementation:
  - observability head `rho_i = h(v_i)`
  - formed / not-formed head `p(C_i = FORMED | A_i = j, g_ij, rho_i)`
- preferred first model family:
  - calibrated gradient-boosted trees
- keep penalized logistic regression as an ablation baseline
- calibrate both heads before they drive structured inference

Acceptance:
- independent-per-bend posterior available for analysis
- calibrated local probabilities available for graph decoding

### 3.2 Sparse coupling
Tasks:
- add hard confusable-bend / evidence-atom exclusivity first
- add shared-flange adjacency only if exclusivity plateaus, and keep it weak + visibility-gated
- add optional precedence prior when data exists

Acceptance:
- improves hard staged families without regressing stable families

### 3.3 Count-aware decoding
Tasks:
- track posterior/score over total completed count `K`
- report count from posterior over `K`
- use posterior median of `K` as the MAE-aligned point estimate in v1
- reconcile bend-wise explanation with reported `K`

Acceptance:
- promoted only if `mae_vs_expected_total_completed` decreases vs current baseline

## Phase 4: Staged-process extension

### 4.1 Ordered scan prior
Tasks:
- use ordered partial scans in evaluator mode
- apply monotonicity first through posterior regularization with slack
- start with count monotonicity only
- defer bend-wise monotonicity until visibility-gated evidence is strong enough
- keep only a weak stage prior where metadata exists; do not introduce a primary stage latent in v1

Acceptance:
- staged monotonic pair rate remains `1.0`
- staged retained bend rate remains `1.0`
- hard staged families improve in count accuracy

### 4.2 Optional temporal runtime mode
Tasks:
- allow multi-scan evaluation of same part when sequence exists

Acceptance:
- no regression for single-scan production endpoint

## Phase 5: Family specialization

Priority families:
- tray/channel partials: `38172000`, `40500000`, `10839000`
- early low-coverage partials: `11979003`
- rolled parts: `47479000`, `47266000`

Tasks:
- add family-conditioned local evidence interpretation
- do not force one local inference rule across all geometries

Acceptance:
- per-family dashboards show gain without cross-family regression

## Regression gates
Every phase must pass:

1. backend tests
2. targeted staged slice
3. full ready-regression corpus
4. reproducibility check on sensitive scans

No promotion if any of these regress materially.

## Immediate next implementation step
The first serious learned unary stage is now in place:
- visibility head bootstrap is trained
- support-aware null-assignment semantics now expose partial-support negatives on hard staged scans
- the first two-head calibrated GBDT bundle is trained at:
  - `/Users/idant/82Labs/Sherman QC/Full Project/output/unary_models/bootstrap_gbdt_20260315_051357`
  - latest pointer: `/Users/idant/82Labs/Sherman QC/Full Project/output/unary_models/latest.json`

Current limitation:
- the state head still has thin negative supervision (`4` bootstrap negatives), so it is useful as a learned separator but not yet strong enough to drive full structured decoding alone

Immediate next step:
- define confusable evidence atoms / cliques
- add hard exclusivity on top of the unary surface
- then move to posterior count decoding over `K`
