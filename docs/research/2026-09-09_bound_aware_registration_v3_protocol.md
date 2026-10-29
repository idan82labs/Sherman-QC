# Bound-aware V3 registration consumers — prospective pure gate

Date2026-09-09. Source-only DRAFT until exact-source independent pre-run GREEN.
The finite-triangle V3 primitive already passed its separate52-case exact gate;
this protocol tests the consumers that decide support, refinement and candidate
eligibility. Neither a primitive pass nor this protocol qualifies a physical run.

## Mechanism and immutable boundaries

Candidate: `backend/cad_guided_global_registration_v3.py`.
Independent tests: `tests/test_cad_guided_global_registration_v3.py`.
The additive consumer uses frozen V3 surface/distance helpers and frozen V1 policy
numbers/V2 Kabsch, Huber and rank helpers. No frozen predecessor is edited or
reexecuted as an experiment. No public PCA wrapper or default product route is
introduced; the separately versioned plane initializer owns starts and custody.

For represented transformed query rows, admit definite distance support only when
the entire distance interval is inside the unchanged correspondence gate. A
lower bound strictly outside is definite outside; the remainder is unresolved.
Normal/point conflict and their uncertainty flags cannot grant unambiguous support.
Face identity alone is not a geometric conflict. Uncertain regions and finite
boundaries are excluded from FACE normal-rank information, not automatically
deleted from distance support. Count/coverage denominators retain all input rows.

Refinement keeps V2's materialized fixed-candidate Kabsch/Huber surrogate and
motion/backtracking rules. Local candidate-distance intervals cannot replace
that fixed-q objective mid-step. Returned world-point restoration/optimizer error
is disclosed separately; no exact-surface descent or certified pose claim follows.

Final P95/RMS upper bounds concern the FIXED definite, uncertainty-excluded
support set only. They do not upper-bound every possible varying true-inlier set.
For possible competitors, pad those definite lower distances with zero for all
additional possible rows: optimistic lower summaries are for ambiguity retention,
not fabricated observations or positive eligibility. Potentially eligible poses
with straddling evidence remain competitors even when not definitely eligible.
Candidate selection compares possible coverage upper/residual lower against a
definite winner. Incomplete and numerical candidate records invalidate selection.

The numerical policy values stay unchanged, but the selected support set differs
from V2. There is no claim that all V3 accepted cases form a subset of V2's accepted
cases. Reverse CAD-sample coverage, normal/rank estimates, pose-class action and
transformation rounding remain explicitly engineering calculations, not interval
certificates. Distance intervals inherit the helper's conditional IEEE model and
represent tessellation, not analytic CAD, scanner calibration or physical truth.

## Fixed test obligations and source closure

The independent test file is frozen before execution. The exact case count and
all hashes are recorded in the independent pre-run receipt, not inferred from
a prior revision's passing count. Required controls include:

- Complete/invalid query schemas, arrays, signs, bounds, source identities and
  all charged resource counters; no missing uncertainty defaulting to false.
- Inclusive upper/strict lower gate boundaries and complete trichotomy; nominal
  favorable distance cannot rescue a straddling interval or refill lost rows.
- Region/normal/point roles, fixed/possible residual denominators and exact-
  rational P95/RMS/coverage checks; null residual is not zero error.
- An actual small finite-query adapter path and cumulative certificate-budget
  refusal after earlier successful queries, retaining original failure reason.
- Materialized-q optimization versus local distance evidence, and potential
  competitor retention when nominal score ordering looks favorable.

Mocks are explicitly decision/ledger/arithmetic seams, never pose-recovery proof.
Only invented arrays and named source are read. No native imports, backend app
startup, scans, CAD files, Gold, retained physical predictions, email or held-outs.

Freeze the consumer, its V1/V2 registration dependencies, V2/V3 triangle modules,
V3 distance helper, this protocol, exact test and named harness: nine files.
One serial worker:60seconds parent wall,45seconds worker CPU,1GiB sampled group
RSS,1MiB stdout. Thread limits1, private HOME/TMPDIR, no pytest plugins/conftest/
cache. The harness retains exclusive STARTED/OUTPUT/COMPLETED and all before/after
digests. Sampled RSS is not an OS memory sandbox. No limits change after start.

Only after independent exact-source pre-run GREEN, execute once:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite bound_aware_registration_v3
```

Every frozen test must pass, no unexplained skips, resource stop or source drift;
independent retained-only post-run audit follows. Preserve any negative unchanged.
A pass advances separately versioned V9 finite-face/file-workflow integration and
the unchanged four-case native workflow gate; it is not physical recovery, a new
Gold11 score or promotion. A successor physical comparison still requires its
own complete freeze/audit and all eleven consumed cases/71 unchanged references.
