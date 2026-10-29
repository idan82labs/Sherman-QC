# V10 B1 native-feature bridge — independent source audit

Date: 2026-09-13. Reviewer: Codex `/root/v9_material_identity_chain`.

## Decision

**The V1-to-V2 class mismatch is repaired at the actual V17 dispatch, but B1 is not yet accepted unchanged: one P1 capacity regression remains.** The bridge incorrectly applies the standalone kernel's 16-feature limit to the cover workflow, whose existing limit is 64. The remaining test-evidence limitations below do not mean the original class repair is ineffective; they limit what its positive result proves.

This is a bounded source/test audit, not a physical PRE, execution attestation, accuracy improvement, new F1, or permission to retry terminal V9. The author's reported eight passing tests were not executed by this reviewer.

## Custody and pinned snapshot

Read `AGENTS.md`, the current-status checkpoint, the complete 97-line bridge and 315-line test, the complete V16-to-V17 source diff, relevant V17 dispatch/error paths, and narrow frozen callees/test helpers. Source/test reads and hashes only: no module imports, test execution, native geometry execution, physical input, reference, attempt, control, cache, or email access. No engine/test edits. Fable remains their sole editor.

The three B1 files had identical before/after SHA256 values (read tools `8c9856`, `9fe151`, `d16463`, `c59450`):

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_native_feature_bridge_v1.py` | `72cb142f28f5af17b0812e31da7437cda99752dae5a256a6aa6dff523998c1cf` |
| `backend/cad_guided_mesh_offline_v17.py` | `3cb73f0860e00e1778c6883bff873d37e48e7f886acf3ce94c55c3cca32a1f35` |
| `tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py` | `d2a0f9efe87d5e91e9465441648ae0faf6a450916ad100980aac259f24472246` |

Connected source/test anchors:

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_native_features.py` | `f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f` |
| `backend/cad_guided_native_features_v1.py` | `3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b` |
| `backend/cad_guided_measurement.py` | `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb` |
| `backend/cad_guided_measurement_v2.py` | `12bb9ce9e595ada6b4cf86947089ef96f2158ddf36009584442e0a414e6ae008` |
| `backend/cad_guided_context_measurement_v2.py` | `9b48c72b4e3eb87899f8e6911929a869c63d0182dd95e254e5deb5671e623f7c` |
| `backend/cad_guided_support_cover_measurement_v3.py` | `6c6bd75813f6e985fd4e841eb401ae330a5be614a6e874b2e5704256e00ca901` |
| `backend/cad_guided_mesh_offline_v16.py` | `469c444f49cdcd631f04635516304b6af530fe3c813dc1f07cadb51389ed16a7` |
| `tests/test_cad_guided_v9_connected_chain_v1.py` | `8b0399202af65ef5e63d92998cee97e999cc437832e2770c796ff6c14962ed46` |
| `tests/test_cad_guided_completion_v1_engine_integration_v1.py` | `20695f7faf2e70a6b2d0c33ffaea1a05ca3f8170303ea0cfd33254e690536875` |

Both native factories and the V1/V2 kernels retain the hashes from the preceding source review. These observations are limited to the pinned snapshot, not later author edits.

## Accepted repair mechanics

1. **The bridge is actually used by both branches.** V17:531–535 selects the unchanged r0 or v1 extractor; :536 retains its native metadata; :539 invokes the bridge; :636 passes `bridged_features`, not the old factory tuple, to the real cover V3. This is not a correct but unused helper. CAD surface sampling begins only at :546 and whole-part registration follows it.
2. **Conversion preserves the feature contract.** Bridge:76–87 compares V2 field names/order against the V1 dataclass, requires the exact V1 class, passes every field into the V2 constructor, and compares the reconstructed dataclass with the source. The two class definitions are identical at this snapshot. Duplicate IDs refuse. No class alias, duck typing, consumer gate relaxation, geometry rederivation, or default-field omission is used.
3. **Native ownership and limitations are retained.** The helper never mutates `extracted`; V17 retains the native metadata object. Feature IDs, `nominal_source`, signed nominal sweep, origin/axis/start-radial, finite spans, search/fit policy and tuple ordering are carried unchanged. The factory's candidate/window/owner/face/edge and trim-exclusion records, status and reasons stay in that metadata. Native CAD inner-face classification is not promoted to verified scan-side identity.
4. **There is no extra coordinate transform or unit claim.** The bridge's provenance explicitly grants no material-side, unit/frame or measurement authority. Context V2:93–97 remains the feature-to-raw transform, with `p_cad = R p_scan + t`, origin `R.T @ (origin - t)` and directions `R.T @ direction`.
5. **Bridge refusals fail closed before expensive downstream stages.** Exact-type, field-name grammar, duplicate-ID and count failures raise `BridgeRefusal`; V17:540–545 preserves native metadata, emits an explicit bridge refusal and returns without sampling, registration or predictions. Ordinary V2 constructor/serialization exceptions instead reach V17:645–647's generic workflow exception; they also clear predictions/measurements before those later stages, but do not receive the dedicated bridge-refusal receipt. Empty/unavailable native tuples retain the predecessor's later `NATIVE_FEATURES_UNAVAILABLE` handling; this patch does not make that existing path an early exit.
6. **Unknown metrology semantics remain unchanged.** V17 preserves observed count `None`, conformance `INDETERMINATE`, uncalibrated uncertainty, provisional millimetres and no verified stable datum. The real measurement positive test checks generic carrier role, false inner-face eligibility and absent inner-radius deviation. The contact helper still projects the actual emitted endpoints; its copied-carrier scoring reference is transport evidence only.

## P1 — Bridge imposes the wrong feature-cap contract

**Location:** bridge:29–36, :75; cover V3:143; context V2:225–228, :294–297.

`MAX_BRIDGED_FEATURES = min(_V1_MAX_FEATURES, _V2_MAX_FEATURES)` equals 16. Those are limits of the standalone batch inspection entry points, not of the context/cover consumer used here. Cover V3 explicitly admits 1–64 features; context V2 admits up to 512 and measures each feature inside its context loop, subject to its separate window/crop/support budgets.

Therefore any otherwise admissible native set of 17–64 features is now refused before the existing cover consumer can assess it. This is a new, unnecessary capacity restriction for a class-identity repair. No physical data were inspected to determine which cases fall into that interval; the source-contract regression exists independently of that question.

The current boundary test cannot catch it: test:226–230 derives both its accepted and refused sizes from the bridge's own incorrect constant, rather than the connected cover contract.

**Minimum closure:** bind this bridge to the cover's existing 64-feature admission contract without changing frozen cover/native modules or truncating/grouping features. Add explicit boundary assertions for 17 and 64 accepted with exact field/order preservation, and 65 refused whole. A separately declared tighter operational budget would be a mechanism change, not a lossless wiring-only repair; none is justified in this B1 source.

## P2 — Positive regression proves wiring, not independent geometry or thin-skin progress

**Locations:** test:63–65, :117–140, :168–170, :198, :239–315.

The real factory-to-measurement test does close an important old blind spot: both actual native factories produce exact V1 objects; the actual bridge produces V2 objects; real crop/fit/context/planner/cover/publication/contact code then handles them. The unchanged V1-object path is separately required to reproduce `INVALID_SUPPORT_COVER_FEATURES` for both branches. No factory or numerical measurement seam is mocked in those positive/negative measurement tests.

Its geometry-independence wording is nevertheless too strong. `_analytic_rows(feature)` takes the **bridged extractor output** as its origin, axis, start-radial, signed sweep, span and radius. The points are independent of CAD tessellation, but not independent of the adapter under test. An orientation/span convention error in the extractor can therefore be copied into the synthetic scan. The 5 mm radius assertion supplies one external constant; it does not independently validate the other quantities or carrier endpoints.

The test also changes skin thickness to 2.5 mm and explicitly comments that the 1 mm variant leaves outer-arc rows in the fit and produces `ARC_NOT_SINGLE_CIRCLE`. In the ideal 90-degree nominal model, the outer arc survives both plane exclusions when both coordinate projections are below `R-e`; for `R=5`, `e=.04`, a 1 mm skin has such an interval, whereas a 2.5 mm skin does not. Thus the thicker fixture deliberately enters the existing kernel's isolatable subcase. That is acceptable for a wiring smoke test, but it is **not** evidence that the kernel improved or can handle the failed 1 mm case. No 1 mm negative case is executable in the reviewed 315-line test; the failure is retained only as a comment here.

The V17 dispatch tests are also intentionally narrower than a real end-to-end workflow. They use the real STEP reader, real factories and a delegating bridge spy, and assert the exact V2 objects handed to the next stage. But scan/topology, CAD sampling, registration, cover and validator are doubles (:251–269). The refusal test injects `BridgeRefusal` itself (:305–308), rather than sending malformed native output through the actual bridge. These are useful routing checks, not full byte-workflow or runtime fault-path proof. The file mostly discloses these substitutions correctly.

**Minimum evidence closure:** describe the present positives as extractor-derived-coordinate wiring/transport tests; preserve the 1 mm failure as a separately retained negative regression (not silently discarded when choosing 2.5 mm). For the requested independently constructed geometry check, generate scan rows from the independent construction constants/frame, not the returned feature; keep actual factory/bridge/context/cover/validator calls. A full registration redesign or broad corpus run is not required to close B1.

## Bounded next acceptance checks

- Close the 16-versus-64 cap regression with explicit 17/64/65 boundary tests.
- Preserve both real-factory positive cases and both unbridged V1 refusal cases.
- Correct the independence claim and retain the known 1 mm refusal; do not count the thicker fixture as measurement improvement.
- Retain metadata and no-prefix checks on refusal. Prefer one actual bridge failure through the dispatch boundary to complement the currently injected-exception check.
- Re-pin changed files and report focused results actually run. No rerun of unchanged broad suites or physical attempt is implied by this source review.

The original V9 failure remains terminal. B1 can remove that wiring obstruction; it supplies no evidence yet that physical registration will reach measurement more often, that thin-skin fitting improves, or that a new physical F1 exists.

## Append-only delta decision — 14:22 author checkpoint

**ACCEPT the requested B1 source closures at the hashes below.** The prior rejected snapshot and its findings above remain intact. This decision closes the capacity P1 and the requested bounded test-evidence changes; it does not certify execution of the new tests, a full byte-workflow, a physical PRE, or measurement/F1 progress.

The 102-line bridge was read completely; the new test sections and their existing helper/dispatch dependencies were read, with the V17 file hash unchanged from the preceding audit. All three before/after hashes match (tools `d4ee68`, `07f66f`):

| File | Delta SHA256 |
| --- | --- |
| `backend/cad_guided_native_feature_bridge_v1.py` | `fef3980a5d5d4594b48dacaee763451c798d081044b4cf47630f5d7cd6126fea` |
| `backend/cad_guided_mesh_offline_v17.py` | `3cb73f0860e00e1778c6883bff873d37e48e7f886acf3ce94c55c3cca32a1f35` |
| `tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py` | `6b3442507936f45a361f711b949801540a002513bd815587d524b171022a7ec9` |

Scoped reconciliation:

1. **Capacity P1 closed.** Bridge:29–41 no longer imports the standalone kernel limits and sets the cover-bound limit to 64. Test:308–331 explicitly accepts 17 and 64 distinct feature IDs through the real bridge and cover's feature grammar, requires the complete corresponding context-window count, and checks 65 refuses in both the bridge and the real cover. This is a boundary test using repeated geometry with distinct IDs, not evidence of 64 separately observable bends. It no longer derives the important boundary values solely from the implementation constant.
2. **Factory-conditioned test limitation disclosed and independent case added.** Test:15–18 now labels the original positive as factory-conditioned transport. New `_construction_rows` (:222–243) takes no feature argument and reads no factory result: it uses radius 5, 90-degree positive-quadrant arc, 0–30 mm axial construction and fixed tangent-flange directions in the constructor's world frame. Both real factory branches feed the real bridge/context/kernel/cover against those rows (:246–271). The test compares fitted radius, sweep magnitude, axial direction and carrier radial/axial placement with construction constants. This removes the previous extractor-to-scan feedback for this one independent, identity-pose, 90-degree case. It is not general pose/trim/calibration qualification. The original factory-conditioned positives and their real publication/contact checks remain; the new independent case itself checks the cover output rather than repeating the separate top-level publication/contact assertions.
3. **Thin-skin failure retained, not renamed a success.** The executable named 1 mm test (:292–305) uses the real factory, bridge and measurement chain, requires the specific `ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL` window refusal and empty predictions, and distinguishes it from the repaired feature-class refusal. Its analytic rows remain factory-conditioned to reproduce that known subcase. The 2.5 mm positive is still only the existing kernel's isolatable geometry. No kernel source change or thin-skin improvement is claimed by this delta.
4. **Real bridge fault now exercises V17's failure path.** Test:274–289 delegates to the real native factory, substitutes a V2 object where the bridge requires V1, then invokes actual V17 dispatch and the actual bridge. It asserts the exact refusal, retained native metadata and no measurement/prediction prefix or registration call. This supplements, rather than removes, the earlier injected-exception test. The invalid object is intentional fault injection, not a mocked successful bridge.
5. **Previously accepted field/provenance semantics remain.** Apart from the cap and its explanatory comment, the bridge's reconstruction, exact class/field grammar checks, duplicate handling, digest, metadata non-mutation and no-authority declarations are unchanged. V17 is byte-identical to the reviewed source, so both r0/v1 call sites, early failure ordering and the actual V2 handoff remain in place. The independent geometry test explicitly keeps inner-face eligibility false and the carrier role generic.

Execution status: root's 14:22 handoff does not state a fresh test pass count. This reviewer performed no imports, tests, native jobs or physical/artifact access and does not infer a pass from test source. Record the actual focused result at these or subsequently re-pinned hashes before treating the executable regression as demonstrated. No further B1 source blocker was found in this bounded delta.
