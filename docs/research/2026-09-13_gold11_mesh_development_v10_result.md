# Gold11 full11 V10 — result narrative (terminal attempt; F1 null on both arms)

Written 2026-09-13 23:15 UTC (date -u).

Author: Claude Code (`claude-fable-5-1`, effort max), candidate author. Written from the attempt's own
retained records after the observed outer exit; the independent non-author POST is separate and is
appended below when accepted. V10 is TERMINAL: never reopened, resumed, rescored or edited. Every
number here is a COMBINED consumed-development outcome (bridge V1, refinement allowance V1 with the
declared CPU guard, exact query reuse, the explicit evaluated-subset contract, corrected declarations)
on top of the unchanged V9 mechanisms, policies and the provisional 1.0 mm admission gate. Nothing
here is a speedup measurement, a calibrated accuracy or a comparison with the historical
0.4580152672 (non-comparable; not beaten). No numeric F1 exists: the unchanged matcher and
aggregation contract returned null for both arms and null stays null.

## Headline

| Item | Value |
| --- | --- |
| Outer exit | returncode 0, OBSERVED_BY_WAITING_PARENT (waiter 70512, runner 70513), 2026-09-13 16:26:45 → 23:04:19 UTC |
| Durable terminal slots | 22 of 22 (11 cases × control/challenger), all worker exit 0, no budget stop, no kill |
| Scorer (SCORED.json, `GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V4`, status COMPUTED_DEVELOPMENT_DIAGNOSTIC) | control TP 0 / FP 0 / FN 71, precision null, recall 0.0, F1 **null**, emitted support lines 0; challenger identical |
| Case dispositions | 11/11 REFUSED (EXPLICIT_WORKFLOW_UNAVAILABLE) on both arms; malformed 0 |
| Campaign wall | 23853 s parent wall (bound 104400 s); inference 23725 s |

## What V10 changed in the outcome taxonomy (22 slots)

| Outcome class | V9 (retained ledger) | V10 | Slots |
| --- | ---: | ---: | --- |
| Killed by the 1860 s CPU cap inside refinement (SIGXCPU) | 14 | **0** | — |
| Registration finished under the CPU guard with NO supported context | — | 11 | CG08 ×2, CG4BDD challenger, CG4FAF ×2, CG64 ×2, CGEBA ×2, CGF41 ×2 |
| Supported context reached, native features unavailable (INVALID_NATIVE_SOLID) | — | 2 | CG191 ×2 (no windows to measure) |
| Supported context AND measurement reached; every window refused | 1 (whole-cover refusal before any window) | **3** (43 windows evaluated, 0 records) | CG1E0 ×2, CG4BDD control |
| No finite prefiltered noisy-plane start | 4 | 4 | CG65 ×2, CG84 ×2 |
| Noisy plane initialization unavailable | 2 | 2 | CGE4B ×2 |
| Native solid invalid AND no supported context | — | 4 (counted above in the 11) | CG4FAF ×2, CG64 ×2 |

The allowance did what it was designed to do: the guard tripped in 16 of 22 slots at ~1500 s of
absolute process CPU and every leaf returned a complete result (worker CPU 12–1556 s, peak RSS
0.56–2.06 GB, parent wall 12.5–1561 s per slot). The measurement side was reached with kernel-V2
objects for the first time on physical data (the V9 defect is closed). The score is null because
the measurement kernel refused all 43 physical windows at the FIRST flange plane fit.

## Per-slot ledger (from COMPLETED.json / RESULT.json; CPU = worker CPU seconds)

| Case | Arm | CPU s | Phase | Registration | Ledger completed/expired/omitted | Guard | Measurement |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| CG08 | control | 1509 | registration | no supported context (3 rejected, p95 1.33 mm at coverage .76) | 3/1/0 | tripped (completion lane) | — |
| CG08 | challenger | 1512 | registration | same | 3/1/0 | tripped | — |
| CG191 | control | 1514 | native features | SUPPORTED (p95 .38 mm, cov .91) | 2/1/6 | tripped (main) | none: INVALID_NATIVE_SOLID |
| CG191 | challenger | 1513 | native features | SUPPORTED | 2/1/6 | tripped | none: INVALID_NATIVE_SOLID |
| CG1E0 | control | 1509 | complete | SUPPORTED (p95 .90, cov .999) | 2/1/0 | tripped | 13 windows, 0 records |
| CG1E0 | challenger | 1521 | complete | SUPPORTED | 2/1/0 | tripped | 13 windows, 0 records |
| CG4BDD | control | 1512 | complete | SUPPORTED (p95 .70, cov .9999) | 3/1/0 | tripped | 17 windows, 0 records |
| CG4BDD | challenger | 669 | registration | 3 rejected (p95 1.20–1.34 at cov .9997) | 3/0/0 | not tripped | — |
| CG4FAF | control | 1502 | registration | 4 rejected (p95 1.39–1.52, cov .81–.93) | 4/1/7 | tripped (main) | — |
| CG4FAF | challenger | 1502 | registration | 3 rejected (p95 1.59–1.62; one also coverage .56) | 3/1/7 | tripped | — |
| CG64 | control | 1532 | registration | only candidate EXPIRED at 1503 s | 0/1/1 | tripped | — |
| CG64 | challenger | 1556 | registration | only candidate EXPIRED at 1513 s | 0/1/1 | tripped | — |
| CG65 | control | 31 | registration | no finite prefiltered start | 0/0/0 | — | — |
| CG65 | challenger | 47 | registration | same | 0/0/0 | — | — |
| CG84 | control | 43 | registration | no finite prefiltered start (44 completion children, 0 passing) | 0/0/0 | — | — |
| CG84 | challenger | 63 | registration | same | 0/0/0 | — | — |
| CGE4B | control | 12 | registration | noisy plane initialization unavailable | 0/0/0 | — | — |
| CGE4B | challenger | 13 | registration | same | 0/0/0 | — | — |
| CGEBA | control | 1516 | registration | 1 rejected (coverage .61) | 1/1/0 | tripped | — |
| CGEBA | challenger | 1508 | registration | same | 1/1/0 | tripped | — |
| CGF41 | control | 1507 | registration | 1 rejected (coverage .59) | 1/1/0 | tripped | — |
| CGF41 | challenger | 1520 | registration | same | 1/1/0 | tripped | — |

Candidate refinement cost (per-candidate cost receipts): 195–1422 s CPU each (median ≈ 410 s), so
the 1500 s guard admits one to four candidates per slot; 31 candidates completed, 16 expired at the
guard, 28 omitted. Every completed candidate's context support reused the diagnosis full-scan query
(`REUSED_DIAGNOSIS_FULL_SCAN_QUERY_EXACT_BINDING`). Context rejections: 18 by the residual gate
(p95 upper 1.20–1.62 mm > 1.0 mm, mostly at high coverage), 9 by insufficient proximity coverage
(.46–.61 < .75; completion children on other faces).

## What the measurement kernel saw (43 physical windows, 86 flange fits)

All 43 windows failed at `PLANE_FIT_ROLE_0`: 32 `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`,
11 `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`. The retained both-flange diagnostics show one physical
mechanism:

- Every flange crop contains BOTH sheet skins: before selection the row spread normal to the crop
  plane is 1.3–5.8 mm, the rms residual 0.60–1.01 mm and the thickness ratio 0.21–0.98 (gate: < 0.2).
  Where the V9 gap partition found a qualifying split, the two layer means sit at ±0.66–0.92 mm,
  i.e. a 1.3–1.6 mm sheet thickness.
- Role 0 never split: 30 of its 32 "weak" fits have this two-skin signature with NO qualifying gap
  (largest gap 0.03–0.43 mm below the 0.3 mm floor): rows bridge the two skins (cut edge, arc start),
  so the empty-gap criterion cannot separate them. Role 1 split in 30 of 43 fits.
- When a split happened and the kept skin was clean, the physical skin is EXCELLENT: 13 role-1 fits
  PASS with rms 0.005–0.011 mm and max residual 0.015–0.029 mm — inside the declared δ = 0.03 mm.
  17 role-1 fits split but still failed the max-residual gate: rms 0.009–0.088 mm with max
  0.034–0.98 mm, i.e. bridging/contaminant rows remained in the kept group (the gap partition keeps
  every row below the split threshold).

So the δ = 0.03 mm flange policy is physically feasible for a separated skin; the binding defect is
the layer separation (empty-gap partition) on crops with bridging rows, plus the gate's sensitivity
to residual contaminant rows in the kept group. The arc gates were never reached.

## Resources and custody

All 22 slots within the frozen limits (worker CPU ≤ 1556 s of 1860; peak sampled RSS ≤ 2.06 GB of
8 GB; per-slot wall ≤ 1561 s of 3540; campaign wall 23853 s of 104400). Stage checkpoints were
emitted on every returned slot (0–22 per slot, never truncated). Gold was read only by the post-
terminal scorer path (SELECTED_REFERENCES.json after INFERENCE_TERMINAL). Outer records beside the
attempt directory hold the observed exit; the attempt tree is complete (FINAL_GATE,
COMPLETED.json inventory 309 entries).

## Consequences for the next successor (V11; diagnosis document to follow)

1. Measurement kernel V3: a two-population layer separation robust to bridging rows (mode-based
   skin detection with band membership and published bridging counts), the unchanged pair
   compatibility and the unchanged smaller-coordinate choice, no inner-face eligibility; this is
   the only mechanism between the 43 evaluated windows and a first physical prediction.
2. Registration admission: 18 near poses rejected at p95 1.2–1.6 mm with coverage up to .9997 and
   four slots whose only candidates expired/omitted under the guard — recorded, not changed in V11.
3. Cost: 200–1400 s CPU per candidate refinement caps the evaluated subset at 1–4 candidates;
   recorded for a later successor.

## Correction (append-only) — 2026-09-13 23:21 UTC: exact ledger counts, gap range and claim strength

Recomputed from the 22 retained RESULT.json ledgers after Codex's independent aggregation (23:18 UTC):
- Candidate dispositions: 31 completed, **15** expired (not 16), 28 omitted, 0 failed, 0 not attempted;
  the CPU guard tripped in **15** of 22 slots (not 16): CG4BDD challenger and the six short-circuit slots
  (CG65 ×2, CG84 ×2, CGE4B ×2) never tripped.
- Completed-candidate refinement CPU: n 31, min 193.960664 s, median 416.010668 s, max 1422.29385 s; all 31
  completed candidates ended both the coarse and the refine loops at the 30-iteration limit
  (ITERATION_LIMIT / ITERATION_LIMIT), never by convergence.
- Context rejections: 26 rejected candidates — 17 by the residual gate alone, 8 by proximity coverage alone,
  1 by both (the 18/9 figures above count that overlap twice).
- Role-0 largest coordinate gap: the 32 role-0 fits that never split have largest gaps 0.028–0.230 mm (all
  below the 0.3 mm floor); the 11 role-0 fits that DID split (gaps 0.343–1.517 mm) failed the residual gate
  afterwards. The earlier "0.03–0.43 mm" range wrongly mixed the two groups.
- Claim strength: the flange layer separation is the FIRST OBSERVED blocker of the 43 evaluated windows; the
  arc gates never ran, so no statement is made about them. A layer-only V11 is justified for a first
  non-empty output on the already-admitted slots, not as a broad F1 cure.
No terminal artifact was edited; the sections above stand as written with these corrections.

## Correction 2 (append-only) — 2026-09-13 23:29 UTC: measurement-stage figures per Codex's independent postmortem

Adopted from docs/research/2026-09-13_gold11_v10_measurement_postmortem_codex.md
(e95e99b325b2a7b20a927b07d76f099d2dae3f7edeaf438e34c2db5eddc7f2bd), which read the same three retained
RESULT.json records (6c4dfe9a…, 1fc6b275…, b73d4481…):
1. "86 flange fits" → 86 before/after flange DIAGNOSTIC pairs; only 43 authoritative first-flange `_plane`
   attempts ran (role 0), all refused; role-1 diagnostic passes are not completed bend measurements.
2. "role 0 never split" → exact cross-tab: role 0 = 30 unresolved single-group WEAK + 4 unresolved single-group
   NOT_PLANAR + 2 incompatible two-group WEAK + 7 compatible-pair selections that still failed the maximum
   residual (rms 0.01643–0.08820 mm, max 0.06234–0.98377 mm); role 1 = 11 unresolved WEAK + 2 incompatible +
   17 selected NOT_PLANAR (rms 0.00905–0.06238, max 0.03380–0.58242) + 13 selected PASS (rms 0.00454–0.01072,
   max 0.01472–0.02940). All 45 WEAK diagnostics passed the minor-spread gate and failed the thickness-ratio gate.
3. Gap wording → of 34 unresolved role-0 single groups, 32 have largest gap ≤ 0.3 mm; the other two (the same
   CG1E0 window in both arms, 0.433066 mm) have 1 row on one side of the gap and 1271 on the other and
   correctly fail the 12-row minimum: the absence of a QUALIFYING gap is the finding, not "every gap below
   0.3 mm".
4. Pre-selection diagnostic ranges over all 86 pairs: rms 0.32666–1.01022 mm, normal extent 1.31823–5.79788 mm,
   thickness ratio 0.10471–0.98420; accepted-pair projected separations 1.33017–1.83750 mm (not "1.3–1.6").
   None of these ranges establishes a physical sheet thickness for every crop.
5. "Every flange crop contains both skins" / "physical skin is EXCELLENT" → measured diagnostic evidence plus
   an INFERRED mixture hypothesis (a balanced parallel pair separated by t gives normal rms ≈ sqrt(σ² +
   p(1−p)t²), e.g. ≈ 0.75 mm for t 1.5 mm; warpage, nearby faces, partial support or projection-axis
   contamination can also thicken the spectrum). Thirteen selected role-1 populations meet the declared sample
   residual; their identity and physical metrology remain unverified. Low rms alone never validates the
   maximum-residual contract.
6. "The only mechanism between … and a first physical prediction" → the FIRST OBSERVED measurement blocker;
   the arc, axis/sweep, canonical-refit and finite-support gates were never exercised and mode separation does
   not guarantee their closure.
7. All 43 local ROIs report zero rows removed by the cap-role filter (it did not remove the problematic local
   population; this does not prove those rows are cut edges).
No terminal artifact was edited.

## Correction 3 (append-only) — 2026-09-14 00:20 UTC: independent POST accepted; field-level reconciliation of the p95 and CPU figures; wording

Independent non-author POST (`docs/research/2026-09-13_gold11_mesh_development_v10_post_run_audit.md`, SHA256
8c11608939833bf4c4d85eb7fd9334e2cea3414d0e8c54a7d19f0a8fff7ea3e2, 70713 B; `…_post_run_audit.json`
ddabac9ee54474bff0d2842e33cfc4c342905b9bba4368491d06bfa1dd180c05, 280901 B; both copied verbatim, hash-verified,
immutable): decision ACCEPT_WITH_FINDINGS — execution custody of the terminal V10 attempt accepted, 0 blocking
findings, 11 non-blocking. Codex (root) read all 1069 lines and reports no material count/resource contradiction
from the complementary reviewers. The acceptance validates the NULL-OUTPUT LEDGER and its custody, not any accuracy.
Reconciliations (no figure in this document or in the POST is wrong; they read different fields):
1. p95 ranges. The POST's 1.196–1.550 mm is the OUTER candidate field `inlier_plane_p95_mm` over the 18
   residual-gate rejections (POST lines 975–978), and the POST explicitly did not extract nested fields (lines
   1039–1040). This document's 1.20–1.62 mm (recomputed: 1.2009–1.6229 mm) is the NESTED
   `conditional_context_support.finite_distance_metrics.p95_upper` over the same 18 rejections. The two ledgers
   also partition differently: outer 18 residual (7 both residual+support, 8 support-only) versus nested 17
   residual-only + 1 both + 8 coverage-only (26 rejected contexts, one overlap). Both are correct for their fields.
2. CPU shares. The POST's 9119.65 s = 38.6 % is whole-WORKER CPU of the six INVALID_NATIVE_SOLID slots; the
   diagnosis's 8914.695307 s = 39.4 % is the METERED REGISTRATION-LANE CPU of the same slots. Different
   denominators, both correct.
3. Wording. CG64 (both arms) has RETAINED refinement work (its only candidate expired at the guard) but no
   completed candidate, diagnosis or context — not "nothing beyond the prefilter". The measured-window sequence
   is CG1E0 control 13, CG1E0 challenger 13, CG4BDD control 17 (13/13/17). The 23:15 UTC ledger entry's "16
   slots"/"16 expired" are superseded by Correction 1 (15/15).
No terminal artifact was edited; the POST bytes stay immutable.
