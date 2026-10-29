# F1 transition-mode probe: formal diagnostic STOP

Date: 2026-08-05  
Scope: frozen Gold37 research lane only  
Decision: **STOP `TRANSITION_MODE_PROBE_V1`; do not implement A1 ownership or births from this evidence.**

## Outcome

The professor-directed `TRANSITION_ATOM_V2` representation and inference-safe
probe were implemented and exercised on all 37 frozen cases. The experiment is
deterministic, bounded, compact, and memory-safe, but it did not recover a
single previously unrepresented gold bend. Its two geometry-valid certificates
occur in C010 and C033; neither corresponds to a recovered unrepresented gold
object. The preregistered probe gate therefore fails.

This is a useful falsification. It rules out the current hypothesis that
splitting occupied spatial parents into projective normal-mode children, then
searching connected two-mode plane-pair runs, is sufficient to expose the
missing bend objects. The next tranche must change the smallest plausible
representation or plane-extraction assumption and must first be tested as an
inference-safe probe. It must not add ownership, births, learning, or threshold
tuning.

## Frozen measurements

| Measurement | Result |
|---|---:|
| Frozen cases | 37 |
| Plane-pair records | 938 |
| Multimodal-parent records | 10,457 |
| Transition components | 1,994 |
| Geometry-valid hard certificates | 2 |
| Recovered previously unrepresented gold bends | 0 |
| Independent family groups recovered | 0 / required 8 |
| Frozen A0 false predictions passing the certificate | 0 / 33 |
| First failure stage 1 | 67 gold misses |
| First failure stage 4 | 3 gold misses |
| First failure stage 7 | 70 gold misses |
| Emitted birth columns | 0 |
| Prediction changes | 0 |
| C001 peak RSS | 0.460052 GiB |

Stage 1 means no matching incident plane pair was available. Stage 4 means a
plane pair existed but the transition-mode connected-run certificate failed.
Stage 7 means the bend was represented by the frozen A0 candidate/path space
but rejected downstream. Stages 2, 3, 5, and 6 each have zero post-lock rows.
Gold was consumed only after blind predictions and inference-safe censuses were
frozen.

The 37-case V2 support pool contains 49,212 transition parents, 60,697 mode
children, and 160,880 mode edges. Those broader representation counts must not
be confused with the focused probe census above.

## Determinism and boundedness

- The detector input remains capped at 200,000 points.
- The V2 caps are 4,000 parents, 16,000 modes, four children per parent,
  20,000 parent-plus-mode records, and 200,000 mode edges.
- Compact cache schema 6 and pool schema 3 serialize bounded atom IDs and plane
  statistics, but no source points or point-index arrays.
- C001 used exactly 200,000 working points and peaked at 493,977,600 bytes
  (0.460052 GiB), below the 16 GiB gate.
- The independently rebuilt C001 pool is byte-identical to the first build.
  Both files have SHA-256
  `36a10af46fc295f437cb95522117605675b671066632af9ab4a97c7f172d8b28`.
- The frozen diagnostic census SHA-256 is
  `dd0007286a2182cdfe1bf0c18219b54fb75226db5543b67caf20a3747ce52ac6`.

## Representative certificate census

C010 has 45 plane-pair records, 229 multimodal-parent records, 84 transition
components, and one geometry-valid certificate. Its certificate joins three
parents and six modes across three occupied axial bins, but A0 parent ownership
is unavailable, residual-window support is absent, and its safe reduced-cost
lower bound is positive. It emits no column.

C033 has 15 plane-pair records, 335 multimodal-parent records, 27 transition
components, and one geometry-valid certificate. It likewise does not recover a
post-lock unrepresented gold object and emits no column. The full C010 and C033
censuses are included so the reviewer can audit every field rather than rely on
this summary.

## Compact implementation diff

1. `backend/patch_graph_latent_decomposition.py`
   - Added bounded `TransitionParentAtom`, `SurfaceModeChild`,
     `SurfaceModeEdge`, and `TransitionAtomGraphV2` records.
   - Added deterministic projective-normal mode splitting, canonical IDs/table
     hashes, mode adjacency, serialization validation, and hard caps.
   - Advanced the compact cache to schema 6 without source-point serialization.
2. `scripts/build_f1_scan_only_candidate_pool.py`
   - Added pool schema 3 and the `TRANSITION_ATOM_V2` compact tables and hashes.
   - Preserved frozen candidate geometry and the 200,000-point input contract.
3. `backend/f1_structured_birth_diagnostics.py`
   - Added a read-only transition-mode census with plane-pair, parent,
     component, certificate, bounded-counter, span-endpoint, and first-failure
     records.
   - Kept every birth column disabled and every prediction immutable.
4. `scripts/diagnose_f1_transition_mode_probe.py`
   - Added deterministic per-case assembly and invariant checks for the probe.
5. `scripts/run_f1_transition_mode_probe_batch.py`,
   `scripts/build_f1_transition_probe_postlock_bundles.py`, and
   `scripts/attribute_f1_transition_probe_gold.py`
   - Added bounded batch execution, immutable census/prediction bundles, and
     post-lock gold attribution with the seven-stage cascade.

## Focused regression evidence

Executed from the repository virtual environment:

```text
.venv/bin/python -m pytest -q \
  tests/test_transition_atom_graph_v2.py \
  tests/test_f1_transition_mode_probe.py \
  tests/test_build_f1_transition_probe_postlock_bundles.py \
  tests/test_attribute_f1_transition_probe_gold.py
.......................                                                  [100%]
23 passed in 0.74s
```

The tests cover deterministic normal-sign handling and mode clustering,
canonical serialization and replay, source-point exclusion, caps/schema
guards, inference immutability, certificate census construction, bundle hash
checks, post-lock attribution, and GO/STOP gate arithmetic.

## Required next decision

The evidence does not justify A1 ownership or a birth family. GPT-5.6 Pro should
choose exactly one smallest falsifiable next tranche and specify its invariants,
bounded fields, counterfactual outcome, and STOP/GO threshold. The preferred
decision boundary is between:

- a plane-extraction probe that tests whether missing incident plane modes are
  an upstream extraction/coverage failure; and
- a representation probe that tests a materially different bounded primitive
  (for example transition strips or two-resolution surfels) only if it can
  explain the 67 stage-1 failures without opening an unconstrained proposer.

Until that choice is experimentally supported: no A1 ownership, no emitted
births, no learned potentials, no threshold tuning, and no runtime/main
promotion.
