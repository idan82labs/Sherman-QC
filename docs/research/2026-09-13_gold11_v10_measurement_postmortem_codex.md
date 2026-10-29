# V10 measurement-stage mechanism postmortem — Codex

Reviewed 2026-09-13 23:14–23:22 UTC. Bounded independent diagnosis, not the physical PRE/POST audit, a replay, or successor approval.

## Decision

The three slots that reached measurement did execute the real kernel-V2 plane path. Their zero support lines are explained by retained, intended single-plane model refusals, not the V9 native-feature class wiring failure. The narrow next mechanism to investigate is **scan-derived layer membership robust to bridging rows and residual contamination**, with unchanged per-selected-plane residual gates and explicit excluded-row accounting. This is a justified experiment, not proof that changing layer selection will produce a line: both-flange success, axis/sweep, arc fitting/tangency, canonical refit, and finite-support gates remain unvisited in these results.

Several author-narrative claims require correction before they are used as the successor hypothesis. In particular, role 0 did split and select a compatible pair in 7 windows; those retained groups still failed planarity.

## Scope and custody

- Read canonical AGENTS/current status, the full V10 protocol, and the full released author result narrative.
- Independently checked `ps -axo pid,ppid,stat,etime,args` for `run_gold11_`, waiter, scorer, pytest and cad-guided processes before retained-artifact reads. Only this inspection shell and its `rg` appeared; no run/test/geometry process was active.
- Read the three named derived `RESULT.json` records below and the named frozen kernel/context/cover/workflow sources. Used read-only `jq` aggregation and source inspection, not application imports or geometry execution.
- No raw scan/CAD, Gold/SELECTED_REFERENCES, control/authorization records, opaque cache, or other attempts were opened. No tests/scorer/replay were run. The terminal attempt is untouched; only this report is written. No moving successor source was reviewed.

## Exact retained denominator and outcomes

| Measurement slot | Contexts | Planned/retained windows | Role-0 WEAK | Role-0 NOT_PLANAR | Source records / cover groups / lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| CG1E0BBA2933 control | 1 | 13 / 13 | 8 | 5 | 0 / 0 / 0 |
| CG1E0BBA2933 challenger | 1 | 13 / 13 | 8 | 5 | 0 / 0 / 0 |
| CG4BDD1371B6 control | 1 | 17 / 17 | 16 | 1 | 0 / 0 / 0 |
| Total | 3 | 43 / 43 | 32 | 11 | 0 / 0 / 0 |

These are 43 context-feature evaluations on two scans, not 43 independent physical bends. The repeated CG1E0 arms contribute the same 13 feature windows twice; 30 distinct scan/feature windows underlie this aggregate. Neither native feature count nor window count is an observed-bend denominator. The 71-reference scoring denominator is separate and unchanged; this review did not open the references or rerun scoring.

All three outputs are `status=UNAVAILABLE`, `phase=COMPLETE`, with `measurements.computation_status=COMPLETE` and `NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`. All 43 window diagnostics have `failure_phase=PLANE_FIT_ROLE_0`. The kernel computes both-flange before/after spectra first, then calls the authoritative `_plane` sequentially: there are **86 flange diagnostic pairs, but only 43 actual `_plane` attempts**, all on role 0. Role-1 diagnostic PASS is therefore not a successful complete bend measurement.

### Selection and post-selection diagnostic cross-tab

| Flange role | Unresolved single group: WEAK / NOT_PLANAR | Incompatible two groups: WEAK | Compatible pair selected: NOT_PLANAR / PASS |
| --- | ---: | ---: | ---: |
| 0 | 30 / 4 | 2 | 7 / 0 |
| 1 | 11 / 0 | 2 | 17 / 13 |

All 45 WEAK diagnostics (32 role 0, 13 role 1) pass the minor-direction spread gate and fail the dimensionless thickness-ratio gate. Thus “insufficient flange extent” is not the recorded weak branch here.

Of 34 unresolved role-0 single groups, 32 have largest gap at most 0.3 mm. The other two are the same CG1E0 gap in both arms: 0.433066 mm, but with just **1 row on one side and 1,271 on the other**. They correctly fail the declared minimum of 12 rows on each side. Absence of a *qualifying* gap must not be described as every gap being below 0.3 mm.

The seven compatible selected role-0 groups have RMS residual 0.01643–0.08820 mm, but maximum residual 0.06234–0.98377 mm: all exceed the unchanged 0.03 mm maximum-residual gate. The 17 selected role-1 failures have RMS 0.00905–0.06238 mm and maximum 0.03380–0.58242 mm. The 13 role-1 diagnostic passes have RMS 0.00454–0.01072 mm and maximum 0.01472–0.02940 mm. Low RMS alone cannot validate the maximum-residual contract.

## Connected path and mathematics

Frozen source anchors:

1. `cad_guided_mesh_offline_v17.py:684–695` passes `scan["raw_points"]` and `bridged_features` into cover V4, validates the publication, and forwards its empty predictions. The records contain no `INVALID_SUPPORT_COVER_FEATURES`, workflow exception, or discarded-prefix outcome.
2. `cad_guided_context_measurement_v3.py:377–428` transforms each nominal feature to raw coordinates, crops full raw rows, applies conditional cap-role exclusions, invokes `_measure`, and retains a typed `_Unavailable` reason plus diagnostics. A support record is created only after successful measurement (`:434–475`). Here none reaches that point.
3. `cad_guided_support_cover_measurement_v4.py:153–168,208–215` accepts complete source-context evidence, plans over zero source records, and produces zero groups/predictions. Retained cover-refit and publication-geometry row visits are zero. Nothing indicates the cover/scorer discarded a successfully measured line.
4. Kernel V2 `:597–659` builds nominally conditioned flange crops, computes both diagnostics/selections, and performs the sequential plane fits. `_plane` at `:431–449` agrees with the diagnostic branch ordering and thresholds at `:197–253`.

For centered row matrix `X`, singular values `s0 >= s1 >= s2` have coordinate units. `s2 / sqrt(N)` is the population RMS orthogonal residual; no sample-covariance `N-1` correction belongs in this declared quantity. The spread test `s1 > 10 * delta * sqrt(N)` is equivalently minor in-plane RMS > 0.3 mm for delta=0.03 mm. The thickness test `s2/s1 < 0.2` is dimensionless. Maximum point-to-plane residual <=0.03 mm is a separate, stricter sample-model check. The inspected arithmetic does not show a denominator, degree/radian, or dimensional bug causing these refusals.

For an ideal pair of parallel populations with common in-plane distribution, normal noise variance `sigma^2`, separation `t`, and mixture fraction `p`, the normal variance is `sigma^2 + p(1-p)t^2`. A balanced 1.5 mm pair can therefore give about 0.75 mm normal RMS even when each population is individually clean. This explains why mixed sheets can violate the ratio test; it is an illustrative model, **not proof** that every retained crop is two physical skins. Warpage, multiple nearby faces, partial spatial support, and projection-axis contamination can also yield a thick spectrum.

The current partition (`kernel:256–281,366–428`) requires a >0.3 mm empty gap in the entire-crop TLS-normal projection. One compatible split retains every row below its threshold. It has no mechanism for assigning a dense inter-layer bridge to neither population. Its pair acceptance uses <=0.1 mm per-group RMS, normal parallelism, a bounded offset, and projected interval overlap; that is a candidate-pair check, not the later <=0.03 mm maximum-residual measurement check. The different thresholds are not an arithmetic contradiction.

All 43 local ROIs report **zero rows removed by the preceding cap-role filter**, although that filter removes rows elsewhere in each full scan. It intentionally removes only conditionally cap-like rows without competing non-cap/unresolved incident ownership. This establishes that the existing role filter did not remove the problematic local population; it does not prove those rows are cut edges or that more aggressive cap deletion is safe.

## Units, geometry, and identity limitations

- All three input receipts say units and coordinate frame are `UNVERIFIED`; the workflow explicitly uses `PROVISIONAL_COMMON_MM_NOT_CALIBRATED` and `ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM`. Millimetre values above are under that declared common-unit assumption, not a calibration certificate.
- Context V3 `:165–169` implements `p_cad = R p_scan + t` by transforming the nominal origin with `R.T @ (origin-t)` and directions with `R.T`. Fitting stays on original scan coordinates. No source-level inversion/unit defect was found at this seam; this does not independently validate the physical pose.
- CAD supplies crop location, nominal orientation and nominal quantities. Fitted normals, axis, circle center/radius and sampled support come from scan rows (`kernel:660–850`). The free-circle path is not replacing the measured radius with the CAD radius.
- Compatible two-layer selection remains `...UNVERIFIED`. The empty inner-eligibility set correctly prevents an inner-face claim even for a two-layer pair. A future successful output may be a generic conditional contact carrier; it must retain unknown side, null inner-radius deviation, unknown observed bend count, and indeterminate conformance. The present three records preserve those restrictions and uncalibrated uncertainty.
- The unvisited arc stage includes an axis-side plane exclusion, free-circle residual and tangency tests, sector/angle/axial support checks, followed by canonical/cover refits and finite-support admission. Nominal outer-arc heuristics explicitly are not isolation certificates. No conclusion about those stages' pass rate follows from the flange diagnostics.

## Smallest justified successor and independent acceptance

Keep registration, native extraction, physical scoring rules and the 0.03/0.04 mm measurement policies unchanged for this mechanism test. Replace the empty-gap-only layer partition with a bounded scan-derived two-population membership model that can assign bridging rows to neither fitted population. Retain all input-row accounting: lower membership, upper membership, excluded bridge/unresolved membership, their exact raw-row binding, policy and attempted/completed work. Do not bury the rejected opposite population in a “contamination fraction”: opposite-skin exclusion and unassigned contamination are distinct quantities.

Declare minimum population support, deterministic ambiguity handling and a maximum **unassigned** fraction before testing. Keep the existing pair compatibility and final per-population maximum-residual checks; do not select a hypothesis because its downstream radius matches CAD/Gold or because it produces a line. The entire-crop TLS normal can itself be unreliable for a thick/short or multiply populated crop, so the model must either validate its projection premise or honestly abstain. Do not expand this into a new registration or scoring redesign.

One independently constructed connected acceptance family can be a finite 90-degree cylindrical bend, nominal radius 2 mm but analytically generated scan radius 2.06 mm, two parallel-offset skins 2.5 mm apart, and independently sampled flange/arc/cut-edge points. Add deterministic <=0.005 mm bounded noise and a declared, bounded fraction of bridge rows that fills all inter-mode gaps below 0.3 mm; rigidly transform the complete construction to an oblique frame. Generate expected axes, sample endpoints and radius from the constructor, never from the native feature factory's output or the fit being tested. These are proposed synthetic parameters, not physical data-derived tuning or executed results.

Minimum connected acceptance is: the new selector separates the stipulated populations without a qualifying empty gap, accounts for every bridge row, and the **real kernel -> context -> cover path** emits the expected generic scan-supported carrier and the independently generated non-nominal radius within a predeclared synthetic error limit. Explicitly exercise both planes, the arc and finite-support gates; a selector-only pass or mocked measurement is insufficient. Keep the known thin-skin/large-radius arc-mixture negative case as a negative, rather than editing its thickness until it passes.

Within that same small family, add row-order and rigid-pose variants plus controls for a warped single sheet, a third population, excessive bridge fraction, and insufficient finite arc support. A clean best-fit subset of a warped/defective surface must not silently erase the deviation. Expected abstention or explicit incomplete/unresolved coverage is acceptable for the controls. These checks diagnose the proposed membership mechanism; they do not promise physical F1 or establish inner-skin identity.

## Required narrative corrections

At the author result narrative hash below:

1. Replace “86 flange fits” with “86 before/after flange diagnostic pairs; 43 first-flange measurement attempts.”
2. Replace “role 0 never split” with the exact cross-tab above. All seven compatible role-0 selections still fail maximum residual.
3. Replace “largest gap 0.03–0.43 mm below the 0.3 mm floor” with the qualifying-gap explanation, including the two duplicated singleton-tail exceptions.
4. Across all 86 pre-selection diagnostics the actual RMS range is **0.32666–1.01022 mm**, normal extent **1.31823–5.79788 mm**, and thickness ratio **0.10471–0.98420**. Accepted pair projected separations span **1.33017–1.83750 mm**, not 1.3–1.6 mm. Neither these ranges nor the pair heuristic establish physical sheet thickness for every crop.
5. Replace “every flange crop contains both skins” / “physical skin is EXCELLENT” with measured diagnostic evidence plus the inferred mixture hypothesis. Thirteen selected role-1 point populations meet the declared sample residual; their identity and physical metrology remain unverified.
6. Replace “the only mechanism between ... and a first physical prediction” with “the first observed measurement blocker; downstream arc and finite-support gates were not exercised.” Mode separation does not guarantee their closure.

## Pinned evidence

Paths beneath `docs/research/gold11_mesh_development_v10_attempt/` are prefixed `attempt/` below. SHA-256 values were read from the retained files, not reconstructed from raw inputs.

| File | SHA-256 |
| --- | --- |
| `docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md` | `e59aff47a8312bfce44139a0a1977a33dff4a0a86039502b8c34fe5f1f27889b` |
| `docs/research/2026-09-13_gold11_mesh_development_v10_result.md` | `f0cf0c90fb053793cf6fd223ca49f154af42d19ee2b4bbf5557de2876ad28b5f` |
| `attempt/source/backend/cad_guided_measurement_v2.py` | `12bb9ce9e595ada6b4cf86947089ef96f2158ddf36009584442e0a414e6ae008` |
| `attempt/source/backend/cad_guided_context_measurement_v3.py` | `e67d1c35cc66a48ddd0ee521ca4c4f23c08f78606751554c17e8b4cdfd042891` |
| `attempt/source/backend/cad_guided_support_cover_measurement_v4.py` | `f6972fd538d27b24d2142a453b56b3846be34e2dc5d79ab30f54d198ce960230` |
| `attempt/source/backend/cad_guided_mesh_offline_v17.py` | `5c4fa6b8f3fb97a23398b9fb00cb418f2f47f0477b93693c6a3c0c4cfd8a5a63` |
| `attempt/cases/CG1E0BBA2933/control/RESULT.json` | `6c4dfe9afdc747f47b5afe4e688d2da5235069c1dd55d9d896881082f907e130` |
| `attempt/cases/CG1E0BBA2933/challenger/RESULT.json` | `1fc6b27587ca5b10643b1f6d2cfdb5eda78b9aaea5d6e5c9411c6f5e2bdda3cc` |
| `attempt/cases/CG4BDD1371B6/control/RESULT.json` | `b73d4481c6053b653af89479b54f3c45593967709eab90e8c8f510486997753c` |

The decisive result is a diagnosed measurement mechanism limitation with correctly retained refusal, not a new score, a manufacturing verdict, or an assurance that the successor will pass.
