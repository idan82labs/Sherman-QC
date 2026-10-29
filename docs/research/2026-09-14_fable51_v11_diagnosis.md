# V11 mechanism diagnosis — from the terminal V10 attempt's retained records — 2026-09-13 23:16 UTC

Author: Claude Code (`claude-fable-5-1`, effort max), sole implementation editor under the standing
consumed-Gold11 development authority. Source of every fact: the terminal V10 attempt
`docs/research/gold11_mesh_development_v10_attempt` (22 retained RESULT.json / COMPLETED.json / SCORED.json
records, read after the observed exit 0; nothing modified), the V10 result narrative
(`2026-09-13_gold11_mesh_development_v10_result.md`) and the frozen V10 sources. The independent V10 POST
and Codex's two read-only postmortems (measurement: `2026-09-13_gold11_v10_measurement_postmortem_codex.md`;
registration: `2026-09-13_gold11_v10_registration_postmortem_codex.md`) are reconciled in the last section
as they arrive (append-only). No physical data, Gold or reference geometry was read for this document.

## A. Facts (retained evidence, not inference)

- A1. 22/22 slots exited 0 under the frozen caps; zero CPU kills (V9: 14). The allowance's guard tripped in 16
  slots at ≈1500 s absolute process CPU; per-candidate refinement cost 195–1422 s CPU (31 candidates completed,
  16 expired at the guard, 28 omitted). Every completed context support reused the diagnosis full-scan query.
- A2. 5 slots published a SUPPORTED measurement context (CG191 ×2: p95 upper 0.38 mm at coverage .91; CG1E0 ×2:
  0.90 mm at .999; CG4BDD control: 0.70 mm at .9999). CG191's native factory refused the solid
  (INVALID_NATIVE_SOLID), leaving nothing to measure; the other three slots measured 13 + 13 + 17 = 43 windows
  with kernel-V2 feature objects delivered through the bridge (the V9 contract defect is closed on physical data).
- A3. 11 slots ended with NO supported context after guard-bounded refinement: 18 candidate rejections by the
  residual gate (p95 upper 1.20–1.62 mm > 1.0 mm, coverage .76–.9997) and 9 by proximity coverage (.46–.61 <
  .75, completion children on other faces); CG64 ×2 had their only candidate expire at the guard.
- A4. 43/43 measured windows were refused at PLANE_FIT_ROLE_0: 32 FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY
  (thickness-ratio branch), 11 FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL. The arc phases were never reached.
- A5. Every flange crop contains BOTH sheet skins: before selection the row spread normal to the crop TLS plane is
  1.3–5.8 mm, rms 0.60–1.01 mm, thickness ratio 0.21–0.98 (gate < 0.2). Where the V9 gap partition found a
  qualifying split, the two layer means sit at ±0.66–0.92 mm (a 1.3–1.6 mm sheet); the pairs were parallel
  within the declared 3° and compatible.
- A6. Role 0 never split (43/43): 30 of its 32 weak fits carry the two-skin signature with largest coordinate
  gap 0.03–0.43 mm, below the 0.3 mm floor: rows bridge the skins (cut edges, the arc start, contaminants), so
  an empty-gap criterion cannot separate populations that are 1.4 mm apart. Role 1 split in 30/43 fits.
- A7. A separated physical skin is excellent: 13 role-1 fits PASS with rms 0.005–0.011 mm and max residual
  0.015–0.029 mm, inside the declared δ = 0.03 mm. 17 role-1 fits split but failed the max-residual gate with
  rms 0.009–0.088 mm and max 0.034–0.98 mm: the gap partition assigns every row below the split threshold to the
  lower skin, so bridging/contaminant rows stay in the kept group.
- A8. The remaining slot classes are the known blockers: CG65/CG84 no finite prefiltered start (CG84: 44
  completion children, 0 passing), CGE4B noisy-plane initialization unavailable, CG191/CG4FAF/CG64
  INVALID_NATIVE_SOLID (the native factory refuses the solid; two of those still registered contexts).
- A9. Resource envelope: worker CPU ≤ 1556 s, sampled RSS ≤ 2.06 GB, slot wall ≤ 1561 s, campaign wall 23853 s;
  stage checkpoints emitted on every returned slot, never truncated.

## Declared observation model for the changed mechanism (to be validated independently)

Physical flange crops are two parallel planar populations (the two skins of a sheet of thickness t ≈ 1.3–1.6 mm
in this cohort; per-skin noise rms σ ≈ 0.005–0.011 mm) plus a minority of BRIDGING rows lying between or beyond
the skins (cut-edge faces, the arc start, dents), continuous in the material-side coordinate. Units: mm, in the
declared common provisional frame (no calibration claim). This model is a development observation from the
V10 diagnostics of three parts (CG1E0, CG4BDD), not a cohort-wide or physical certificate; it is declared so
that the successor's mechanism can be judged against it, and a synthetic fixture built from it is a MODELED
fixture, never physical truth.

## B. Mechanism decision for V11 (bounded to the concrete cause)

- B1 (the change). Measurement kernel V3 = kernel V2 text with a layer separation robust to bridging rows:
  (1) the material-side coordinate along the crop TLS normal, oriented to the nominal radial, unchanged;
  (2) two-population detection by declared, deterministic mode finding on that coordinate (fixed-width bins,
  the two tallest local maxima whose separation lies in [layer_split_gap_mm, max_layer_gap_mm] and whose
  populations each reach min_layer_rows; fewer than two → single population, unchanged V2 outcome);
  (3) band membership: a row belongs to a skin iff its distance to that skin's TLS plane is within the declared
  `layer_membership_band_mm`; rows in neither band are BRIDGING rows (published as a count and fraction, never
  measured); a bounded number of refit iterations (declared) makes membership and planes consistent;
  (4) a declared bound on the bridging fraction (`max_bridging_row_fraction`) refuses the separation when the
  crop is not two skins plus a minority of bridging rows (identity unresolved, all rows kept, as in V2);
  (5) the UNCHANGED V2 pair compatibility (per-group planarity and rank, parallel normals ≤ 3°, offset inside the
  declared gap window, projected interval overlap) and the UNCHANGED smaller-coordinate choice; identity stays
  `TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED`; `INNER_FACE_ELIGIBLE_STATUSES` stays EMPTY; every
  measured window stays a generic conditional contact line.
  The flange gate forms (spread, thickness ratio, orientation, max residual ≤ δ) are unchanged and applied to
  the retained skin rows. Because membership is a geometric band, the gate is not vacuous only if the band is
  wider than δ; the band default is therefore declared ABOVE δ (proposal: 0.10 mm = the declared
  `max_layer_rms_mm`, the layer planarity policy already in force), so rows between δ and the band still fail
  the unchanged gate and are reported, never trimmed. The bridging bound and the published counts keep the
  classification from manufacturing a plane out of a sparse subset. Fit quality never chooses between skins.
- B2 (recorded, not changed in V11). Registration admission: 18 near poses rejected at p95 1.20–1.62 mm with
  coverage up to .9997; the provisional 1.0 mm gate is unchanged in V11 so that the measurement change is the
  only mechanism delta between V10 and V11 on the three measurement-reaching slots.
- B3 (recorded, not changed). Per-candidate refinement cost 195–1422 s CPU bounds the evaluated subset to 1–4
  candidates per slot; a cost successor (fewer refine iterations / rows, or a cheaper diagnosis) is a separate
  versioned decision.
- B4 (recorded, not changed). INVALID_NATIVE_SOLID (CG191, CG4FAF, CG64), no finite start (CG65, CG84) and
  noisy-plane unavailability (CGE4B) persist as deferred blockers.

## Required evidence before any V11 freeze (Codex's condition, adopted)

1. A genuinely POSITIVE real-connected synthetic fixture: a STEP solid with a bend through the REAL native
   factory, the REAL initializer V16 / registration V5 on a synthetic two-skin scan mesh generated from the
   declared observation model (both skins at the sheet thickness, per-skin noise, bridging edge rows, finite
   sampling), the REAL bridge, kernel V3 through context/cover, ending in at least one generic conditional
   contact prediction scored TP by the unchanged matcher against the fixture's own construction line (transport
   truth, never Gold). No mocked geometry seam. Finite coverage and units are stated in the fixture.
2. Adversarial negative controls on the same path: bridging fraction above the bound (refuses, all rows kept),
   a third population (ambiguous, unresolved), an incompatible pair (unresolved), a single skin (unchanged V2
   behaviour), a skin with contaminant rows beyond the band (the unchanged gate refuses), and a forged
   `layer_membership_band_mm` narrower than δ (constructor refusal).
3. The V10 retained diagnostics as a modeled replay: the observed per-window statistics (row counts, layer
   means, rms) drive a synthetic replay fixture; its outcome is a modeled expectation, not a physical claim.

## What V11 does not do

No change to score arithmetic, matcher, thresholds or denominators; no CAD nominal value substituted for a
measured quantity; no relaxation of δ, e, the p95 gate or any physics bound; no Gold in inference; no per-case
tuning; V10 and earlier remain terminal.

## Reconciliation with the independent reviews (append-only)

(pending: independent V10 POST; Codex measurement postmortem; Codex registration postmortem)

## Correction (append-only) — 2026-09-13 23:21 UTC: exact counts and claim strength

- A1: 31 completed, 15 expired, 28 omitted; the guard tripped in 15 of 22 slots; completed-candidate CPU min
  193.96 s, median 416.01 s, max 1422.29 s; ALL 31 completed candidates hit the 30 + 30 coarse/refine
  iteration limits (never converged) — a cost fact for a later successor (B3), not changed in V11.
- A3: 26 rejected candidates: 17 residual-gate only, 8 coverage only, 1 both.
- A6: the 32 role-0 fits that never split have largest gaps 0.028–0.230 mm; the 11 that split (0.343–1.517 mm)
  then failed the residual gate (A7's contaminant mechanism).
- B1 claim: the layer separation is the FIRST OBSERVED blocker; the arc gates never ran and V11 makes no
  claim about them; a layer-only V11 targets a first non-empty output on the admitted slots, not broad F1.

## Reconciliation 1 (append-only) — 2026-09-13 23:23 UTC: Codex registration/refinement postmortem

Read and hash-checked: docs/research/2026-09-13_gold11_v10_registration_postmortem_codex.md (reviewer
`v9_completion_cost_postmortem`; evidence inventory hash c260c63b… over the 22 RESULT + 22 COMPLETED files;
frozen pins V5 ee05a03e…, V16 7dad12ae…, V17 5c4fa6b8…, protocol e59aff47…), and the V11 coordination
checkpoint docs/research/2026-09-13_fable51_v11_codex_coordination.md (93539b00…).
- Counts: adopted verbatim (31 completed / 15 expired / 28 omitted = 74 selected; 15 guard-tripped slots; all
  expiries and omissions are CPU-guard dispositions, none row-quota; 26 rejected contexts with one overlap;
  101 passing start groups before quotas, 27 unscheduled; 5 supported contexts, 3 measurable). The author
  corrections above match.
- Cost attribution: adopted. Completed candidates 16964.55 s CPU (main 15 / 6115.49 s; completion 16 /
  10849.06 s), expired 5660.54 s, completion prefilter 19.80 s; sequential finite surface refinement is the
  dominant cost, not the completion prefilter; CG64's two slots have no completed candidate (an honest empty
  publication); CG4BDD challenger rejected three completed candidates without tripping the guard.
- Reviewer priority 1 (staged/multiresolution refinement with a declared model-aware convergence rule and
  unchanged final validation) and priority 2 (stop registration before it runs when native features are
  unavailable/empty: six INVALID_NATIVE_SOLID slots consumed 8914.70 s = 39.4 % of metered lane CPU — a
  source-proven ordering inefficiency in V17 that saves runtime, not F1): RECORDED as separately versioned
  successors (B3/B4), NOT coupled to V11. The reviewer's own conclusion is adopted: a layer-separation-only
  V11 is justified for a first non-empty output on the three measurable slots; no registration change is
  required before it; neither a non-empty prediction nor a positive F1 is guaranteed because the arc gates
  never ran.
- Admission contract: no gate is loosened (the 1.2–1.6 mm cases stay refused); a future visibility/material
  model needs independently constructed evidence, never a threshold increase.

## Reconciliation 2 (append-only) — 2026-09-13 23:29 UTC: Codex measurement postmortem (e95e99b3…)

Adopted: the exact cross-tab and ranges (recorded as Correction 2 of the V10 narrative); "first observed
blocker" wording; the mixture-hypothesis framing (A5/A7 are measured diagnostics plus an inferred two-population
hypothesis, not a physical thickness certificate). Design requirements adopted into kernel V3: opposite-skin
exclusion and unassigned (bridging) rows are DISTINCT published quantities (`dropped_other_skin_rows`,
`dropped_bridging_rows`, `bridging_fraction`); declared minimum population support (`min_layer_rows`),
deterministic ambiguity (`layer_mode_ambiguity_fraction` on band masses) and a maximum unassigned fraction
(`max_bridging_row_fraction`); the projection premise (the crop TLS normal) is validated after the refits —
the angle between the crop normal and each final skin normal must stay within `max_layer_normal_angle_deg`,
otherwise the selection abstains (`PROJECTION_PREMISE_UNRESOLVED`, all rows kept); the unchanged pair
compatibility and per-population maximum-residual gate follow; no hypothesis is selected by its downstream
radius or by producing a line. Acceptance family adopted for the V11 chain tests: a finite 90° bend with
nominal radius 2 mm and an analytically generated scan radius 2.06 mm, two parallel skins 2.5 mm apart,
independently sampled flange/arc/cut-edge rows, deterministic ≤ 0.005 mm noise, a declared bridge fraction
filling every inter-mode gap below 0.3 mm, rigidly transformed to an oblique frame; expectations from the
constructor only; the REAL kernel → context → cover → contact path must emit the generic carrier and the
non-nominal radius within a predeclared limit, exercising both planes, the arc and the finite-support gates;
row-order and pose variants; controls: warped single sheet, third population, excessive bridge fraction,
insufficient finite arc support; the thin-skin arc-mixture negative stays negative. In addition (root's
requirement) a full-bytes fixture: STEP solid → real native factory → real registration on a two-skin PLY
scan → real bridge/kernel V3/context/cover → prediction.

## Reconciliation 3 (append-only) — 2026-09-14 00:20 UTC: independent V10 POST (8c116089…) accepted; field clarifications

The V10 POST decision is ACCEPT_WITH_FINDINGS (0 blocking, 11 non-blocking): custody of the null-output ledger,
not accuracy. Its candidate p95 range 1.196–1.550 mm is the outer `inlier_plane_p95_mm` field over the 18
residual-gate rejections; the 1.20–1.62 mm quoted in A2/B2 above is the nested
`conditional_context_support.finite_distance_metrics.p95_upper` (recomputed 1.2009–1.6229 mm) over the same 18
rejections (the POST did not extract nested fields); the outer ledger partitions 18 residual = 7 both + 8
support-only + 3 residual-only while the nested ledger reads 17 residual-only + 1 both + 8 coverage-only.
Likewise 9119.65 s / 38.6 % (POST) is whole-worker CPU of the six INVALID_NATIVE_SOLID slots and 8914.695307 s /
39.4 % (this document) is the metered registration-lane CPU of the same slots. CG64's two slots have retained
refinement work but no completed candidate, diagnosis or context (not "nothing beyond the prefilter"); the
measured-window sequence is CG1E0 C 13 / CG1E0 H 13 / CG4BDD C 17. No V11 gate, score or experiment changes.

## Reconciliation 4 (append-only) — 2026-09-14 00:50 UTC: final implemented semantics of B1 after the Codex kernel and consumer-chain audits

The B1 mechanism as implemented (kernel V3, current bytes in `docs/research/2026-09-14_fable51_v11_review_response.md`)
refines the wording above without changing the decision: (1) "bridging rows" are UNASSIGNED between-skin rows —
rows outside both membership bands that lie strictly between the two refitted skin planes — published as
`bridging_rows` / `dropped_bridging_rows` / `bridging_fraction` with the explicit semantics that their cut-edge
identity is unverified; (2) rows outside both bands that are NOT between the planes (air side of the kept skin, far
side of the other skin) are assigned to the nearest plane and are never discarded: in the SELECTED population they
meet the unchanged gates (a displaced selected skin is refused), in the OPPOSITE population only the pair planarity
checks (that skin is never measured; its conformance is unknown); a between-skin displacement of original rows is
an explicit, exactly counted exclusion bounded by `max_bridging_row_fraction` (0.35), never a conformance or coverage
claim (the protocol's earlier blanket "a dented flange is refused NOT_PLANAR" is superseded by this honest split);
(3) mode band masses are exact inclusive coordinate-band row counts; the secondary mode must carry ≥ 25 % of the
primary's band mass; a third population needs band mass ≥ 12 rows (population support, not a single-bin peak);
(4) membership must reach a fixed point within 6 completed refits (completed refits and loop entries published
separately); (5) the band > δ contract is enforced jointly at measurement; (6) the exact V10 counts: role 0 = 34 fits
unresolved as a single population (32 no qualifying gap, 2 singleton-tail), 7 selected, 2 incompatible; 24 selected
fits beyond δ (7 role 0, 17 role 1). B2–B4 remain recorded and unchanged.
