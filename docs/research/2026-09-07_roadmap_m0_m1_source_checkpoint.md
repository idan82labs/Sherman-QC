# Roadmap implementation checkpoint: source closure, observation contract, 3D tests

Date: 2026-09-07. Implements the next bounded source-only portion of the
[roadmap](2026-09-07_production_generalization_roadmap.md). This is engineering
progress, not a physical experiment, GPI-A/B result or production qualification.

## Scope and decisions

The user requested proceeding with the roadmap, using Astra for spatial reasoning,
research-grounded testing and deeper math/physics. The independent geometry task
used `gpt-6-astra` at `max` reasoning effort. There is no reliance on an undocumented
3D mode or model confidence as an oracle. The implementation produces explicit
geometry, derivations, counterexamples and executable assertions.

Work remained source-only: no incoming email or held-out parts, raw inputs,
protected manifests, truth, controls, issuance records or existing scientific
outputs were inspected. No detector ran on physical data. No dependencies were
installed, no production release issued, and no terminal experiment rerun.

## M0: C15 ordinary implementation repairs, not a runnable freeze

The parent and worker disagreed on descriptor magic, batch commitment domain and
aggregate-envelope kind. Parent source now uses the existing worker's contract.
Stale worker/parent tests referred to C13 payloads and nonexistent C15 resource or
pool-worker modules; the tests now exercise the C14 resource/C15 geometry contract
using invented data. Scientific core and worker source are unchanged.

The parent deliberately retains zero-SHA/size-1 bindings for worker and worker
tests. Those placeholders were not filled or bypassed in a production path.
Synthetic supervision uses an explicitly test-only exact-byte command, not a
real-input authorization. No core rule, adapter, cohort or scoring change occurred.

| Verification | Result | Boundary |
| --- | --- | --- |
| C15 core | 99 passed, 29.87 s | Dependency-isolated with the worker's exact AST lineage witnesses; native builder/geometry/AI import path not qualified |
| Worker | 36 passed | Synthetic streams, fixed source/test bytes and expressly public C14 disposition only |
| Parent | 96 passed, 3 failed | Remaining failures are real source-freeze gates; not skipped, xfailed or weakened |
| Combined worker/parent | 132 passed, 3 failed, 2.18 s | Functional compatibility is repaired; frozen parent still non-runnable |

Remaining failures:

- `test_source_is_stdlib_only_and_binds_frozen_worker_resource`
- `test_fresh_source_check_only_passes_without_data_modes`
- `test_fresh_parent_ignores_pythonpath_sitecustomize`

All fail because the worker/test bindings remain placeholders. The next C15 gate
is independent review/freeze of exact source bytes under the existing protocol;
any physical diagnostic still requires its separate authority. This checkpoint is
not that review, issuance or permission.

Exact source snapshot (SHA-256, bytes):

```text
core         cc65c7b70ac8072ce3adf8125c48f6d763cf57a45904c06ce42264a22df84d33  77095
core tests   b18a8da9edef86972a8f82c9c5cfa58512866d4b2b12d02150eda4c14b6c4c10  29681
worker       d250cb7dbe7129ec3730dd3bc1705db4c3b42ecb26fefc8cec72bd859056dd08  61505
worker tests 6ad6f1712b18f9236b27b180e5a9d657e3eebda909e54c017bc65903b2157db3  22814
parent       a31857ad8b458c4371837b42be85a54b32973c9aabd4b838811d50f8e1c0c215  223930
parent tests 37897e61225ad895c7febda36bbfd555889211f8af9377e57e2c43776a0365e9  109276
```

## M1: explicit evidence, preserved legacy data

`normalize_observability_state` no longer manufactures an observed state from a
tolerance verdict or physical-completion claim. Explicit unknown, unobserved,
partial and malformed states remain unknown. This is a **runtime behavior change**:
observation-dependent merging/count features may now abstain where they previously
asserted evidence. It is not just a cosmetic label change. Frozen historical
evaluations must retain their exact old source; do not mix this revision into an
old baseline or compare scores without a separately declared experiment.

`BendMatch` now derives missing detail from its explicit observation state instead
of injecting a contradictory default `UNOBSERVED`. Explicitly supplied detail is
preserved, including `PARTIALLY_OBSERVED` before normalization. No legacy detector
`detected_count` or ranking definition was changed. Source-extracted dataclass
checks distinguish the legacy detection summary from the final observation count;
they do not load the detector or qualify native execution.

The new pure application boundary,
`domains/bend/application/finalize_bend_report.py`, adds a versioned
`decision_contract` without overwriting raw matches, measurements, legacy summaries
or posterior estimates. Its typed representation lives in `contracts/ui/bend_results.py`.
It partitions mapped countable targets into observed formed, observed unformed and
unknown; partial observation is a subset of unknown physical state. It rejects
conflicts, duplicate identity as positive evidence, malformed state tokens and
registration abstention. Process-feature rows remain visible but outside bend counts.
Present malformed countability flags retain unknown targets; malformed registration
flags cause abstention. Null countability falls back to the nested CAD flag, with
valid top-level booleans taking precedence. The independent review's original
malformed-flag counterexamples were rerun and confirmed closed.

Version 1 has no qualified-registration state and cannot authorize disposition.
Registration is `ABSTAINED` or `UNVERIFIED`; artifact existence is only
`UNAVAILABLE` or `UNVERIFIED`, never native geometric qualification. Partial
formation is retained in raw metadata but remains unknown in this first contract
until the stage-aware physical taxonomy is independently settled. Partial capture
does not imply partial manufacture.

The **active worker**, not just the unused early mapper, applies this application
boundary before artifact presentation and after artifact attempts. JSON and text
exports share its evidence. The UI validates the version, enums, partition and
per-event coverage; missing/old/malformed contracts are unavailable, not reconstructed
from PASS/FAIL. Counts use mapped-target denominators; posterior/CAD expectation
remains a separate estimate. Unverified tolerance colors and corrective commands
become neutral review states while raw measurement values remain available.
The UI/UX guidance informed this explicit separation and a mobile row-layout fix;
the existing visual design was preserved. Countability is shared by the decision
projection, annotations, chips and review queues, without inferring exclusions
from a rolled-feature name. A feature-local viewer error boundary prevents a
failed 3D asset from erasing the report or its evidence limitations.

The PDF/manifest export path also projects decision evidence rather than legacy
completion. Model counts remain separately labeled estimates; raw tolerance
measurements remain experimental. Unsupported corrective instructions are withheld
from the PDF, and stale qualification flags cannot remove research-only warnings
from pages or detached visuals. A render receipt can establish availability only,
not measurement or registration validity. Assembly checks use in-memory flowable,
renderer and filesystem stubs; no PDF artifact or native renderer was exercised.

This is not the full application/queue migration. Registration-to-measurement
provenance, stage-aware partial-formation ontology, calibrated measurement
uncertainty, durable jobs, authentication and candidate-bound release verification
remain roadmap work. No native runtime or physical conformance claim follows.

## M2/M3 preparation: analytic geometry and observability limits

The [analytic fixture and research note](2026-09-07_analytic_bend_testing_design.md)
define a finite annular cylindrical sector, inner/outer faces, tangent flanges,
directed axes and explicit units. The implementation is small and in-memory,
with no detector, scanning simulation, fitting or source-data lookup.

Independent Cartesian, implicit-surface and tangent oracles precede metamorphic
checks for SE(3), unit conversion, axis-direction gauge, sampling and disjoint
finite support. Two key counterexamples constrain future methods:

1. Identical observed flange points and normals can correspond to different
   radii and cylinder axes when transition/contact regions are hidden. CAD may
   fix a nominal value but does not verify the unobserved actual value.
2. For a short circular arc, `h = L^2 / (4(R + sqrt(R^2-L^2/4)))`. A free plane
   can approximate its positional support within `h/2`; with error allowed under
   both hypotheses, observation sets can overlap when `h <= 4 epsilon`. These are
   sufficient ideal-model ambiguity bounds, not calibrated detector thresholds.

The cylindrical centre axis, an on-face ruling and the infinite tangent-plane
intersection are distinct. Collapsing those representations or connecting
disjoint collinear spans can invent geometry. Conversely, a gap in scan support
does not prove a physical disconnection. The note derives these distinctions and
cites primary mathematical/software-testing references.

This is a geometry model, not a springback/plasticity/neutral-axis/material model.
Exact normals and component labels are construction truth, not privileged inputs
for a future scan-only inference test. No physical family or accuracy point was
added by this work.

`scripts/render_analytic_bend_diagnostic.py` creates a two-panel diagnostic from
the same invented geometry, with an explicit fresh output path. It distinguishes
the three line representations and shows identical visible patches over two
hypothetical hidden radii/axes. Point/normal equality is asserted before writing;
the generated PNG was visually reviewed for matched scales and readable labels.
This is an explanatory analytic figure, not a reconstructed scan or a detector
result. Four output-path guards passed.

## Focused verification

- Combined focused Python verification: **243 passed, 0.78 s** in the integrating
  agent's final run: 103 observation/contract/mapper/worker/source-extracted-class
  checks, 68 isolated PDF/manifest assembly checks and 72 analytic geometry checks.
  Worker tests include reference/artifact failure and adversarial upstream
  completion/qualification declarations. Native/model/qualification providers are
  replaced; worker file IO uses new synthetic temporary output. Export assembly
  uses filesystem/flowable/renderer stubs, with no actual PDF or image generation.
  Mathematical derivations and the diagnostic PNG were separately reviewed.
- Frontend projection: **17 synthetic checks passed**, scoped ESLint passed and
  the full TypeScript/Vite build passed. Build advisories concern existing large
  chunks and outdated Browserslist data; no dependency upgrades were made.
- Compiled-browser QA passed at **1440 x 1000** and **390 x 844**. Every API/auth
  response was invented; only freshly built local static assets were served, with
  all foreign HTTP and WebSocket access blocked. A legacy total of 999 could not
  replace the observed ratio 1/3; model-minus-observed displayed +8. Two blocked
  external HDR requests per viewport exercised the contained viewer fallback.
  No unexpected API requests, uncaught page errors or page-width overflow occurred;
  the mobile job title remained visible. Final desktop/mobile screenshots were
  visually reviewed by the integrating agent. This tests failure handling and
  presentation, not a successful native render, real backend or real account.
- C15 results and limitations are separately recorded above; do not combine their
  tests with these counts into a generalization metric.

No real API/account, physical/native geometry job or existing report migration was
run. Actual PDF layout/fonts remain unverified; passing stubbed assembly checks
does not establish rendered PDF appearance. No physical accuracy, family coverage
or production qualification metric changed.

## Reproduction within the same source-only boundary

From `FullProject/`, after inspecting the named source/test revision:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=.:backend .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_bend_runtime_semantics.py tests/test_bend_decision_contract.py tests/test_bend_worker_decision_boundary.py tests/test_bend_match_observation_boundary.py tests/test_bend_pdf_decision_projection.py backend/tests/test_ui_contracts.py tests/test_analytic_bend_fixture.py -q
node scripts/test-bend-decision-contract.mjs
npm --prefix frontend/react run build
node scripts/qa-bend-decision-ui.mjs
```

The browser QA requires the already installed local Playwright/Chromium; it does
not install a browser or dependencies, and its screenshots use a fresh temporary
directory. The analytic figure renderer similarly requires an explicit new PNG
path whose parent directory already exists and refuses to overwrite that path.

C15 worker/parent (expected to retain the three freeze failures on this snapshot):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest --noconftest -c /dev/null -p no:cacheprovider -q --tb=short tests/test_run_f1_train18_candidate15_geometry_representation_worker_v1.py tests/test_run_f1_train18_candidate15_geometry_representation_failure_localizer_once_v1.py
```

Dependency-isolated core test body, run with the same environment flags through
`.venv/bin/python -` (this does not import the native builder):

```python
import sys
import pytest
import backend
import scripts
from scripts import run_f1_train18_candidate15_geometry_representation_worker_v1 as worker
builder_path = worker.FIXED_SOURCE_BINDINGS['builder_source'][0]
clusterer_path = worker.FIXED_SOURCE_BINDINGS['clusterer_source'][0]
builder, clusterer = worker._execute_lineage_witnesses(
    builder_data=builder_path.read_bytes(), clusterer_data=clusterer_path.read_bytes())
sys.modules['scripts.build_f1_scan_only_candidate_pool'] = builder
scripts.build_f1_scan_only_candidate_pool = builder
sys.modules['backend.f1_line_object_clustering'] = clusterer
backend.f1_line_object_clustering = clusterer
raise SystemExit(pytest.main([
    '--noconftest', '-c', '/dev/null', '-p', 'no:cacheprovider', '-q', '--tb=short',
    'tests/test_f1_train18_candidate15_geometry_representation_failure_localizer_v1.py']))
```

These commands are not permission to launch a C15 data mode or open a held-out.

## Next decision

Close independent C15 source freeze under its existing gate, without reopening
terminal CAD experiments. For science, define one finite-transition mechanism and
its observation channel, then test it against the independent analytic oracles
and mandatory ambiguity/abstention cases before spending physical data. For product
work, carry capture/stage/transform provenance through the real application path
and connect the versioned evidence contract to qualified measurements. Keep the
reserved email parts untouched while selecting mechanisms and thresholds.
