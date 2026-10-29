# Rerun preparation and promotion audit gates

Date: 2026-09-09. User request: proceed until rerun-ready, with audit gates for
promotion. **Disposition: source/synthetic integration is green; actual existing-
corpus rerun is NOT_READY.** Three external metadata facts remain unverified and
cannot be replaced by declarations or invented fixtures. No physical inputs,
protected reference truth, corpus manifest/control, reserved email or held-out input was accessed.
No incumbent was run, no deployment occurred, and the goal remains paused.

## Delivered engineering work

1. [Import-safe offline engine](../../backend/cad_guided_offline.py),
   `inspect_registered_native`: complete normal Python module, NumPy-only initial
   import, lazy registration/native extraction/measurement. It returns the actual
   payload without legacy app, DB, report or Open3D dependencies. The existing
   registered [pipeline entry](../../backend/bend_inspection_pipeline.py) now only
   delegates and wraps `BendInspectionReport`; there is no duplicate registered
   engine implementation. Default worker/product routing is unchanged.
2. [Bounded case input](../../backend/cad_guided_case_input.py): explicit absolute
   STEP/BREP, point-cloud PLY and datum JSON inputs; no discovery or corpus defaults.
   Raw scan row order is preserved. ASCII, little-endian and big-endian scalar
   vertex-only PLY are supported; XYZ follows property names, not column order.
   Declared float32/64 semantics agree across formats. There is no mesh fallback,
   filtering, resampling, scale guessing, known-pose fallback or truth input.
3. Datum JSON strictly binds the exact captured scan and CAD SHA-256 bytes before
   native loading. Its only fields are schema version, both source digests,
   `reference_frame`, `unit`, `selection_method`, CAD datum positions/normals and
   raw scan datum-row indices. Unknown fields, duplicate keys, nonfinite values,
   stale byte bindings and invalid indices fail. This verifies snapshot identity,
   NOT the declared datums' physical rigidity, units or CAD-frame correctness.
4. Explicit regular-file snapshots are bounded before parsing: CAD 8 MiB, PLY
   64 MiB, datum JSON 2 MiB, header 64 KiB, at most 200,000 scan rows, 5,000 datum
   rows and 32 scalar PLY properties. Unsupported formats/size/geometry are
   unavailable cases, never permission to drop jobs or silently raise limits.
   CAD imports an owned byte copy with native shape retained and OCP selected
   explicitly; no gmsh retry. Final symlinks and FIFOs are rejected. Parent path
   symlinks are not comprehensively rejected; this is not an authorization sandbox.
5. [Paired scoring](../../backend/cad_guided_paired_evaluation.py),
   `score_paired_support_cases`: recomputes incumbent/challenger scores against one
   identical reference mapping and complete declared case/part/family inventory.
   Missing references or lane outputs invalidate comparison. Valid truth is
   retained for unavailable/partial jobs; no exact-case credit is awarded to those
   jobs. Every declared family appears, including absent/undefined families.
   Scalar counts, detached aggregate PASS fields and forged promotion flags do
   not control scoring. Metric deltas remain unknown when either value is undefined.

The input dataclass/read-only arrays are not tamper-proof provenance. Only the
validated load path carries the byte-binding evidence, and independent review
still must establish physical input facts. The loader itself does not enforce
scientific custody; no physical CLI or permissive approval flag was added.

## Fixed verification and first failure

All runtime engine modules in the new file rehearsal are imported whole, not
AST-pruned or replaced. The bounded worker uses a named-source Python importer
that compiles `.py` directly (no stale bytecode) and refuses unnamed backend
modules. Only the old legacy report/wrapper are AST-selected in the retained
66-case report regression. The file rehearsal does not need that legacy layer.

One native job at a time; frozen 300 s wall /240 s worker CPU /2 GiB sampled
process-group RSS /1 MiB output, BLAS/OMP threads 1, isolated owned HOME/cwd/cache,
disabled site startup and Python audit guards. These are bounded engineering
controls, not OS isolation or physical-access authorization.

| Final verification | Result | Parent wall s | Worker CPU s | Worker peak RSS B |
| --- | ---: | ---: | ---: | ---: |
| `case_input` | 47 passed | .211607417 | .189466 | 50,429,952 |
| `offline_source` | 8 passed | .149375792 | .133345 | 66,502,656 |
| `registered_source` | 22 passed | .130461500 | .114990 | 47,857,664 |
| `paired_evaluation` | 50 passed | .387158000 | .355414 | 84,328,448 |
| `native_worker_source` | 25 passed | .126446875 | .115002 | 38,780,928 |
| New file-input native matrix | 16/16 expectations | 2.426774167 | 2.438338 | 372,441,088 |
| Retained registered native matrix | 66/66 expectations | 9.997950417 | 9.907204 | 377,831,424 |

Thus 152 focused tests passed in this turn. The two native matrices are separate
engineering ledgers, not 82 independent physical cases or a pooled performance
score. Both final native runs exit 0, without budget stops. OCP is 7.9.3.1.

[New file matrix](../../scripts/cad_guided_rerun_checks.py): 12 positives =
STEP/BREP × ASCII/LE/BE double-precision PLY × known +1 mm radius/+5 degree sweep,
all obliquely posed with one quarter of each datum plane's samples removed.
Four negatives are stale scan binding, wrong datum frame, unsupported face element
and malformed CAD. All 16 jobs remain in evaluation. The 12 positives call the
actual engine; the four input refusals are explicitly tagged as input-admission
abstentions, never represented as executed engine results. Independent geometry
limits remain 2e-6 mm/degrees. Maximum recovered position error is
4.7374141420755887e-14 mm; radius error 4.529709940470639e-14 mm; sampled endpoint
error 9.222373658585875e-14 mm; sweep-deviation error 5.9401372709544376e-12 degrees.
These exact constructed/file-roundtrip errors are not scanner accuracy.

All-case synthetic file score: TP12/FP0/FN4, P1/R.75/F1.857142857, exact support
count 12/16. Four safe refusals earn no detection/count credit and are not defects.
The unchanged 66-case geometry matrix has TP54/FP0/FN12 and exactly the prior
reported geometry maxima. Its F1 .90 /exact support-count 54/66 remain synthetic,
not a measured Gold gain or achievement of the physical offline targets.

Retained first failure: the first file matrix had **12 failed positive checks and
four passed negative controls**. All 12 positives reached correct pose, radius,
angle and finite endpoints but lacked top-level `uncertainty_status` (the kernel
already labeled individual feature uncertainty). No shortened aggregate was
published. First parent 2.479483500 s, CPU 2.477754 s, peak/group 372,490,240 B,
exit 1, source digests unchanged, no budget stop. The new offline entry now adds
explicit top-level `UNCALIBRATED` and physical `UNKNOWN`; geometry, thresholds,
fixture and denominator are unchanged. Same 16 then passed. The complete 66 was
replayed after this repair and the normal-import refactor. This is development
regression evidence, not independent confirmation.

Before the first run, independent source audit caught and closed lazy CAD binding
selection, ASCII declared-type rounding/strict numeric grammar and malformed
ASCII temporary-list growth. Those were pre-execution findings, not fabricated
failed experiments. Final independent source review found no critical issue in
the repaired input/core/paired boundaries. Earlier free-circle, report and scorer
failures remain in the [prior receipt](2026-09-09_registered_finite_support_checkpoint.md)
and its linked history; unchanged broader suites were not rerun or declared green.

Reproducible source-only commands from the Git root:

```bash
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite case_input
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite offline_source
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite registered_source
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite paired_evaluation
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite native_worker_source
.venv/bin/python -I -S scripts/check_cad_native_provenance.py --rerun-input-only
.venv/bin/python -I -S scripts/check_cad_native_provenance.py --registered-only
```

## Audit gates and promotion decisions

| Gate | Current ruling | Required evidence / allowed advance |
| --- | --- | --- |
| G0 — Engineering integration | GREEN, narrow invented-input scope | Import-safe actual path, unchanged fixed geometry criteria, retained failures, independent source audit, complete scoring denominators, bounded tests and source consistency. Allows this opt-in offline code to be a prepared candidate, not a production route. |
| G1 — Comparable development rerun readiness | BLOCKED | Exact retained incumbent and challenger source/config/model/environment/entrypoint identities; reviewed datum availability/selection/frame/unit binding; approved common event/line/endpoint convention and exact reference revision; same predeclared cases, part/family grouping, evaluator, budgets, all-failure handling and prospective decision rule. Separate physical-development authority must precede protected access. |
| G2 — Development candidate selection | NOT RUN / NOT AUTHORIZED | Apply the G1-frozen rule to the actual paired per-case outputs; audit failures, resources, all declared families and regressions. A demonstrated same-cohort gain may justify retaining/freezing a research challenger. It is not unseen-family evidence or production permission. No rule is retroactively chosen from outcomes. |
| G3 — Prospective qualification | NOT EVALUATED | Candidate frozen before fresh cases, independently reviewed observable truth and convention, untouched family/part groups, measurement and detection evidence, worst-family and full-case reporting under separate scope/authority. New email parts remain untouched until this stage is explicitly approved. |
| G4 — Production / floor | NOT AUTHORIZED | Independent verification of actual evidence bytes against the deployed bundle, approved release-scope applicability, operational/risk criteria, monitoring/rollback and named release authority. Floor/shadow use is a separate decision retaining operator authority. No synthetic or consumed-development PASS grants it. |

These gates preserve the [enterprise policy](../release/ENTERPRISE_BEND_PROMOTION_GATES.md)
and [CAD-assisted-first roadmap](2026-09-07_production_generalization_roadmap.md).
Do not require release-sized cohorts, standalone zero-shot or final .95/.90 scores
before an informative authorized development rerun. Conversely, the paired
module's integer-rational `.95/.90/.80` **sample-support diagnostics** cannot
qualify physical observed-bend targets. Current native feature inventory remains
nonexhaustive; `COMPLETE` means all requested supported features yielded results,
not all observable manufacturing events. Physical observed bend count remains
null, uncertainty uncalibrated and tolerance conformance indeterminate.

## What specifically blocks G1

1. **Historical Gold convention:** the [released audit](2026-08-27_cad_assisted_f1_deep_dive.md)
   explicitly says the target-line convention is not fully signed and requires
   two independent manufacturing authorities for primitive/event and line
   placement. The original scorer accepts bare geometry without determining
   cylinder axis versus flange intersection/skin semantics. Do not silently
   relabel Gold as `CYLINDER_CENTER_AXIS_SUPPORT`, shift references or pad predictions.
2. **Exact incumbent binding:** the released Gold11 narrative identifies the
   retained conservative scan + scan-supported CAD lane (TP30/FP30/FN41, F1 .458,
   exact count 1/11), not its complete executable snapshot/config/entrypoint.
   The old scorer's 13-scan defaults are not the 11-case retained mapping. The
   referenced machine-readable binding was not opened under this source-only scope.
3. **Actual datum compatibility:** no inspected released source/narrative verifies
   that the intended cases contain stable datum selections/reference geometry,
   or a reproducible truth-independent way to obtain them. This adapter does not
   automatically turn arbitrary bare PLY into a datum-qualified input. A separately
   labeled aligned-stage diagnostic is possible only under its own reviewed scope;
   it must not silently replace end-to-end registration evaluation.

Next requires released metadata for those three reviews, or explicit metadata-only
review authority under the existing custody rules. No scan/CAD/label/email opening,
benchmark execution or old terminal-protocol retry is implied. If the historical
reference convention differs, an independently specified compatible measurement
representation or versioned reference bridge is needed; a new ontology cannot
inherit the old .458 baseline by renaming fields. If datum inputs do not exist,
the input/registration course needs an explicit decision, not an invented gate pass.

## Source binding and limitations

The parent captured the same **14 named source digests before and after each
native run**, with source editors paused during execution. Its changed-source
exit is tested. This is consistency evidence, not tamper-proof executed-byte
attestation: transient edit-and-restore races and the complete third-party binary
dependency graph are not cryptographically attested. A release gate still needs
independent full-bundle/evidence verification. Fixed synthetic policy values live
in the source-bound fixture; an actual corpus candidate/config is not yet frozen.

Final native-run SHA-256 values:

```text
1c085c8b133710b9d80956c46f7ecbdc0ecbd103d0161c5f66cdecae9a0c0b9d  backend/bend_inspection_pipeline.py
9e1d2a0904c50fbf5e6118cc8ed8ff93f51626ee29768bce40b26b2d8a360917  backend/cad_guided_case_input.py
3a6a5e1b5a17f5c939f44066ee0d621ef1ace7996422c46640506cacf788b1c3  backend/cad_guided_evaluation.py
d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda  backend/cad_guided_instances.py
d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb  backend/cad_guided_measurement.py
f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f  backend/cad_guided_native_features.py
6c13a90d358e459cda657d66d3474387e4d499b64e7fbd4acc184dd6c467a8a7  backend/cad_guided_offline.py
8d8dfff246815f420d0887c6adc5319029c5f280df863f89c00163d1e82b95a8  backend/cad_guided_paired_evaluation.py
a1d34872ec9466d8201e26c2ab4d43cc2677169a317d8675643090f0e5938475  backend/cad_guided_registration.py
51d77da3456a8a3ccc75e11e6679d3ffca202c05a1f69fa6477654288637ce1b  backend/cad_import.py
a94d7865517d020562443f15420d707162a3249461d901eeba5b01970110e68a  backend/feature_detection/bend_detector.py
6fcaea1284795f1fbe43d403d1c14d43c7f12192276c03e2d869cff084cda025  scripts/cad_guided_rerun_checks.py
598f41a703e96ce3c2da45e63944c802fc17298ff48acca287f1566e32a46c67  scripts/cad_registered_guided_checks.py
568149ef29bd7eae7933ba20d118fa8be243f0e79aac3cf791b43beabb9d9ef2  scripts/check_cad_native_provenance.py
```

The failed first file run differed only in offline-core hash
`8a4a49006bdb0e1e0833e0e821da7c625c404f87dd73eb0877f486d551a53ea0`;
the successful replay adds the explicit aggregate unknown/uncalibrated fields.
Final static verification: 13 changed/new Python sources parse, all 14 recorded
runtime-source digests match, all nine local receipt links resolve, and scoped
whitespace checks pass. No application imports were executed for those checks.
No corpus/performance promotion, zero-shot/UI/floor work or data-custody change
is hidden in this source checkpoint.
