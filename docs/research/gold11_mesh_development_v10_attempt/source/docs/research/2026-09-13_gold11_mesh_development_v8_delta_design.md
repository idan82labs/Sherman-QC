# Gold11 V8 delta design — 2026-09-13

Source-only design record for the V8 successor authored by Claude Code
(`claude-fable-5-1`, effort max) under the standing consumed-Gold11 development
authority. Not a qualification, freeze, PRE, execution request or physical
result. V7 and every earlier attempt remain terminal and unchanged.

## Frozen basis and exact delta inventory

Basis: `scripts/run_gold11_mesh_development_v7.py`, 169180 bytes, SHA256
`e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52`, declared
as `SOURCE_BASIS` and frozen for preservation only (never imported at runtime).
The V8 runner is the V7 text with these classes of change:

1. Identities: attempt/protocol/design paths, `GOLD11_MESH_DEVELOPMENT_V8`,
   `ONE_USE_PARENT_BOUND_LEAF_V8`, custody/inventory schemas, private cache
   prefixes, private scoring/sampler namespaces, V8 error codes, comparator and
   arm identities. Runner custody, supervision, terminal fill, scoring transport
   (`GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6`), limits and the
   contact scorer are the V7 text.
2. Engine wiring in `workflow_entry_and_policies`: mesh offline V15 entry and
   policy, initializer V14 context configuration in mode `v4`, context V4
   budgets, triangle surface V4 budgets, registration V4 budgets, and an
   explicit `REGISTRATION_IMPLEMENTATION` binding check.
3. Scorer projection: `scripts/gold11_schedule_projection_v3.py` replaces V2 in
   `SCORING_SOURCES`; the wrapper schema becomes
   `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V3`. It binds the V15/V14
   identities, the nine context-V4 declarations (kd-partition label, 64-row
   leaves, 512M coordinate maximum) and the V4 registration label on the
   parent evidence and every array binding. The contact projection, matcher
   and all scoring thresholds are unchanged.
4. Declared configuration deltas (`CONFIG_DELTA_FROM_V7`, also frozen in the
   bundle): `registration.max_inlier_plane_p95_mm` 0.5 -> 1.0,
   `surface_budgets.max_ties_per_query` 32 -> 256, context request
   `max_extrema_coordinate_visits` 256M -> 512M, and the NEW
   `workflow.max_measurement_contexts` = 8 (see the measurement-context
   schedule below). Every other CONFIG value,
   certificate cap, sampler policy and limit is unchanged; the arm request
   gains exactly one identity field (`registration_implementation`) and its
   values are otherwise unchanged. The stage sampler names the V8 runner path.

## Engine modules (all NEW files; frozen predecessors untouched)

| Module | Relation to predecessor | Reason (V7 evidence) |
| --- | --- | --- |
| `backend/cad_guided_raw_extrema_index_v3.py` | V1 staged program, query, ties, failure boundary verbatim; rank-partition kd-tree (global halved-root-span axis schedule, stable per-node sort, 64-row leaves) replaces the single longest-axis slab sort; 512M coordinate maximum | 10/22 V7 slots refused at 256M extrema coordinate visits after ~45% of the whole buffer per query |
| `backend/cad_guided_two_plane_context_v4.py` | V3 text bound to extrema V3; schema v4; partition declaration | same |
| `backend/cad_guided_triangle_surface_v4.py` | V3 + sound nearest-vertex pass-1 pruning threshold, 256 tie slots, wide competitive interval -> row uncertainty | CG191 width refusal after a supported context; CG4BDD/CG4FAF challenger near-tie refusals; certificate work dominated registration CPU (CG4FAF control killed at 1860 s) |
| `backend/cad_guided_global_registration_v4.py` | V3 consumer bound to surface V4; consistency check for the wide counter; retained wire strings; `REGISTRATION_IMPLEMENTATION` | same |
| `backend/cad_guided_stage_policies_v2.py` | V1 strings bound to registration V4 | consumer binding |
| `backend/cad_guided_plane_initialization_v14.py` | V13 text on context V4; verbatim V12 copies of `_FaceQueries`/`_prefilter`/`_context_support`/`_configuration` resolved against V4 modules | wiring |
| `backend/cad_guided_mesh_offline_v15.py` | V14 text on initializer V14; V15 identity; registration-implementation binding | wiring |

Exactness evidence (invented, consumed development): extrema V3 returns
byte-identical intervals/witnesses to frozen V1 on the four consumed 32,768-row
populations x 12 queries, a 200K-row folded sheet x 19 queries and the scalar/
blocked oracles; context V4 canonical-bytes-equal to V3 on the connected
fixtures; surface V4 satisfies the same exact convex-hull enclosure oracle as
V3 and is byte-equal to V3 on one generic 400-triangle/189-row fixture.

Cost evidence (invented, consumed development, not a gate): on an 800K-row
folded sheet with 184 CG085-like whole-buffer queries, frozen V1 refuses at
256M coordinate visits after 26.2M evaluated rows while V3 (leaf 64) completes
at 122.0M coordinate visits, 10.19M rows, 150.3M sort work, 4.8 s CPU; leaf
256/128/32 variants were also measured (161M/2.35 s, 142M/3.3 s, 104M/7.3 s).
On the 200K/19-query batch V3's summed common work and build+query CPU are
HIGHER than V1's (sort work dominates small batches). The physical successor's
retained per-case context counters and CPU against the V7 retained counters are
the intended before/after efficacy evidence; no synthetic gate is imposed.

## Interpretation boundary

V8 outcomes are combined consumed-development changes: new mechanisms (extrema
V3/context V4, surface V4/registration V4), new declared capacities (512M
coordinates, 256 ties) and one provisional policy change (1.0 mm pose-admission
gate) act together. No per-change attribution, mechanism-only speedup, or
calibrated accuracy improvement over V7 is claimed; the in-run arm comparison
is schedule/plane-lane only. The 1.0 mm gate admits poses whose definite
support p95 upper bound lies between 0.5 and 1.0 mm; those contexts carry an
unresolved false-pose/uncertainty risk that the unchanged coverage, rank,
measurement-window, fit-residual and contact-scorer gates bound only
partially. Results must carry that label.

## Measurement-context schedule (NEW policy in V14/V15)

Under the 1.0 mm gate, several near-duplicate converged poses can all be
SUPPORTED contexts; the frozen measurement consumer measures every published
context and refuses the whole case at fixed preflight budgets
(`CONTEXT_WINDOW_PREFLIGHT_BUDGET`: contexts x features <= 4096;
`FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET`: contexts x features x raw rows <= 64M;
for 17 features on 581K rows that is at most 6 contexts). V14 therefore adds
`_schedule_measurement_contexts` after the unchanged V12 publication:

- rank the SUPPORTED contexts by their own retained evidence (context p95
  upper ascending, geometric coverage lower descending, candidate index);
- collapse near-duplicate poses with the frozen V2 engineering action metric
  over the refinement rows (`pose_class_distance_mm` = 0.1 mm), keeping the
  best-ranked representative (`SUPPORTED_CONTEXT_DUPLICATE_POSE_CLASS`); this
  collapse applies regardless of the cap, so below the cap the measured set can
  differ from a measure-everything publication only by near-identical windows
  the cover planner would otherwise merge;
- keep at most `max_measurement_contexts`, a cap V15 derives as
  min(policy maximum 8, 4096 // features, 64M // (features x raw rows)),
  never below 1 (`SUPPORTED_CONTEXT_OMITTED_BY_MEASUREMENT_BUDGET`).

Omitted contexts retain their complete evidence and dispositions in
`measurement_context_schedule`; the published index list and the consumer's
recomputation from the SUPPORTED_CONTEXT status agree. This is a deterministic,
label-free capacity schedule, not pose acceptance, uniqueness or a
correspondence certificate. Aggregate local-row budgets inside the consumer can
still refuse a case; that remains a retained outcome.

## Expected new outcome class and accounting notes

- More admitted contexts (1.0 mm gate) can refuse a whole case at the frozen
  measurement consumer's fixed preflight budgets
  (`CONTEXT_WINDOW_PREFLIGHT_BUDGET`: contexts x features <= 4096;
  `FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET`: contexts x features x raw rows <= 64M)
  or its aggregate row budgets. Such a refusal is a retained negative outcome
  of this configuration, not an engine fault; it must be reported as such.
- Extrema V3 index-visit charges are 4S + 2P (P = rows of all partitioned node
  segments), about 28.5M for a 900K-row buffer, within the context's 64M
  parent index-visit cap.

## Deferred blockers (retained honestly)

- CGE4B (both arms): noisy-plane triangle-membership cap (20M hard maximum in
  the frozen V5 budgets) inside the V7 scan-plane lane and the composite lane;
  a repair needs a new noisy-plane/composite chain and is deferred.
- CG64/CG65 (both arms): completed finite prefilter with no passing start; no
  diagnosis basis for a mechanism change.
- CG191, CG64, CG4FAF: `INVALID_NATIVE_SOLID` native-feature refusal; even a
  supported context cannot yield predictions for these three cases.
- Registration wall/CPU limits (3540/1860 s) are unchanged; cases that now pass
  initialization may still exceed them.

## Review status

Independent preliminary reviews (Codex, no P1) covered extrema V3/context V4
and surface V4/registration V4; the follow-through delta is recorded in
`2026-09-12_fable51_review_response_v1.md`. Initializer V14, mesh offline V15,
projection V3 and the V8 runner require their own independent review and the
exact frozen PRE before execution.
