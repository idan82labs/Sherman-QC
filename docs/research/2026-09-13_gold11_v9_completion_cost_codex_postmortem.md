# V9 completion cost and deadline coupling — independent source postmortem

Date: 2026-09-13. Reviewer: Codex subagent `v9_completion_cost_postmortem`.
Scope: bounded read-only source/performance diagnosis; this document is the only
write. No implementation, tests, inference, scoring, raw/Gold/reference reads,
RESULT geometry reads, older-attempt reads or opaque-cache access. The parent
reported actual outer exit 0 and independently quiescent writers before assigning
this review. This review does not independently attest that outer receipt and is
neither the full physical POST nor PRE/launch authority for a successor.

## Conclusion

**V9 protects main-lane queue positions, not main-lane time or completed output.**
The added completion prefilter executes before any registration refinement; its
passing groups then join a shared all-candidates-complete barrier. All work spends
the same 1860-second leaf CPU allowance. A late expensive or failing completion
candidate can therefore prevent publication of a context whose main candidate
already finished. This is a source-proven non-monotone availability mechanism,
not proof that every physical V9 termination took that particular path.

The allowed terminal metadata independently confirms 14 CPU-signal exits out of
22 slots, but explicitly leaves their internal stages unknown. The current-status
narrative attributes the kills to surface refinement; this review has not read
the stage-sampler logs needed to independently establish that attribution, much
less the active lane or candidate number. Do not turn an unknown stage into a
claim that the main lane never ran or that completion alone caused every kill.

Recommended order: close the separately discovered production feature-type
wiring defect; add lane/group cost attribution; remove an exact duplicate final
surface query; establish bounded completion isolation and an end-to-end synthetic
deadline acceptance test. If that still cannot fit the existing leaf allowance,
develop a separately versioned inexpensive proposal optimizer followed by the
unchanged conservative final verifier. Do not simply raise caps or rerun V9.

## Evidence and limits

Read `AGENTS.md`, `docs/CURRENT_STATUS.md`, `docs/REPO_MAP.md`,
`docs/operations/astra-workflow.md`, and the complete V9 protocol. Named source
callees were inspected only where needed for scheduling, resource accounting,
surface-query cost and context publication. The V9 attempt's `COMPLETED.json`
provided the exact 22 case/arm `COMPLETED.json` paths; no directory crawl was used.
Only those terminal metadata files were subsequently inspected and hashed.

The primary terminal file reports 22 durable records, campaign wall
28675.58346658433 seconds, `failures: []`, and
`COMPUTED_PENDING_INDEPENDENT_POST_AUDIT`. The empty campaign-failure list is
consistent with successful terminalization of failed inference slots; it is not
successful geometry on all slots.

| Case | Control exit / wall s | Challenger exit / wall s | Returned disposition |
| --- | ---: | ---: | --- |
| CG08506AE97D | -24 / 1861.37 | -24 / 1861.05 | Both supervisor refusal; internal stage unknown |
| CG191DEC7F81 | -24 / 1861.28 | -24 / 1861.31 | Both supervisor refusal; internal stage unknown |
| CG1E0BBA2933 | -24 / 1860.99 | -24 / 1861.63 | Both supervisor refusal; internal stage unknown |
| CG4BDD1371B6 | 0 / 1626.82 | 0 / 702.23 | Control measurement refusal; challenger no supported context |
| CG4FAF273093 | -24 / 1861.80 | -24 / 1861.79 | Both supervisor refusal; internal stage unknown |
| CG64C412CC73 | -24 / 1862.02 | -24 / 1862.03 | Both supervisor refusal; internal stage unknown |
| CG65CD575B1C | 0 / 30.83 | 0 / 47.06 | Both no finite prefiltered starts |
| CG84C8FA12D2 | 0 / 42.82 | 0 / 63.65 | Both no finite prefiltered starts |
| CGE4B0549E41 | 0 / 12.50 | 0 / 12.75 | Both noisy-plane initialization unavailable |
| CGEBA9A869AC | -24 / 1861.89 | -24 / 1861.90 | Both supervisor refusal; internal stage unknown |
| CGF41F49066A | -24 / 1861.91 | -24 / 1861.97 | Both supervisor refusal; internal stage unknown |

Host `kill -l 24` returned `XCPU` (listing only; no signal sent). The runner sets
both CPU soft and hard limits to 1860 seconds before dispatch. All 14 signal
records have `budget_stop: null`; that field is not proof of no resource stop:
the OS delivered the CPU signal, as distinguished from a supervisor-selected
wall/RSS stop. Their elapsed times range 1860.992–1862.027 seconds and sum to
26062.946 seconds, **90.889% of the campaign wall time**. That percentage is
elapsed wall accounting, not a measured CPU profile or fraction attributable to
completion. The maximum sampled leaf-tree RSS was 2045247488 bytes (1.9048 GiB),
well below its 8 GiB cap; this does not attest continuous RSS between samples.

The two returned CG4BDD records independently show four completed candidates in
control and three in challenger. Control reached `MEASUREMENT_REFUSAL` with
`SUPPORT_COVER_MEASUREMENT_FAILURE:ValueError:INVALID_SUPPORT_COVER_FEATURES`;
challenger returned `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`. The
feature-type cause is reported by current status and belongs to the connected
wiring audit, not this performance review. All eight exit-0 leaves can thus
coexist with zero emitted geometry. No F1 was recomputed here.

## Source-proven findings

### 1. Additive prefilter work precedes the supposedly protected main lane

In `backend/cad_guided_plane_initialization_v15.py:1108`, main passing groups are
counted; line 1109 invokes the complete added lane. Only at lines 1115–1127 is
the main-first refinement list materialized. Completion does not replace main
indices: the remaining slots are filled after the main strata. However,
completion enumeration (729–849), all selected three-support checks (856–907),
new face-index builds (883 and 453–490), and its final ledger encoding (912–923)
occur before **any** main refinement at 1456–1463.

This is real added work even when the main group count already fills all 48
refinement slots. The completion function is not given the remaining refinement
quota and does not skip its finite queries in that condition. The completion
policy permits up to 256 parents, 64 selected children per parent and 4096 total
children (runner 275–284). Every selected child is correctly rechecked on all
three finite supports; a shift along infinite planes is not a license to reuse
old finite-face pass decisions. Any cost optimization must preserve that fact.

Completion budget/refusal exceptions also use the whole-initialization failure
path at V15:1173–1197, invalidating previously found main groups. This is a second
coupling mechanism, separate from CPU exhaustion. It is deliberately conservative
in V9, but additive proposals are therefore not monotone in usable output.

### 2. Main-first refinement still sits behind an all-candidate publication barrier

V15:1441–1468 builds one candidate list, one shared `_Queries` ledger, and runs
coarse refinement, fine refinement, final diagnostics and context support in
sequence for each group. Only **after the entire loop** do lines 1477–1482 mark
search complete, publish contexts and apply the eight-context measurement cap.
The inherited publisher is explicit: V12:1104–1109 requires every candidate to
be attempted and fully diagnosed before publishing any supported context.

An exhausted finite-surface budget raises `_SurfaceFailure` (registration
V4:254–258); V15:1472–1474 rethrows it. The outer exception path clears eligible
contexts at 1485–1497. The final gate at 1503–1515 independently refuses incomplete
candidate evaluation. An OS CPU signal bypasses Python completion entirely, and
the worker only publishes RESULT after the complete engine call returns
(runner 1591–1599). Thus even a main candidate that already has good evidence
is not a returned measurement if a later candidate exhausts the leaf.

Main-first ordering prevents **slot displacement**, but not (a) upfront deadline
consumption, (b) late fatal invalidation, or (c) whole-leaf loss at OS termination.
The existing eight-context cap is downstream of the expensive work and cannot
bound registration cost; pose-class deduplication also occurs after refinement
(V15:1229–1301).

### 3. Abstract quotas do not imply a feasible CPU envelope

Runner 234–243 declares the 1860 CPU / 3540 wall seconds; 1579 and 1205 install
the CPU limit. There is no lane time reserve in the inspected registration path.
Registration `_Queries` is shared across all selected groups (V15:1452) and has
4096 calls / 64 million row / 20 billion abstract-work caps (runner 342–344).
Those quotas bound operations, not elapsed CPU instructions.

At the declared maximum samples and iterations, one selected group can request:

`30*2048 + 30*8192 + 50000 + 8192 + 50000 = 415392` surface-query rows,
in at most `30 + 30 + 2 + 1 = 63` surface calls.

These terms respectively cover coarse/fine iterations, `_diagnose` full scan and
refinement samples, and `_context_support` full scan. Samples can be smaller and
iterations can stop early; this is a source bound, not observed V9 work. For 48
groups the corresponding 19938816 rows / 3024 calls fit under the aggregate
row/call quotas, yet the entire leaf has at most 38.75 CPU seconds per group
even before input preparation, prefiltering, publication and measurement.
Meeting abstract quotas therefore says nothing sufficient about CPU feasibility.

Each surface call is substantial: V4 surface queries perform two finite-hull
passes, per-leaf rank/row loops (507–538), four interval certificates per evaluated
point/triangle pair (340–375), and pairwise near-tie comparisons (615–648).
The interval helper deliberately rounds each arithmetic operation outward
(`cad_guided_triangle_distance_bounds_v3.py:9–12,33–86`). Its guarantees must not
be optimized away by enabling fast-math or replacing conservative bounds with
an uncertified nearest-neighbor distance. The source establishes repeated costly
work, not which primitive dominated any killed physical slot.

### 4. An exact duplicate final full-scan query is available for reuse

Registration V4 `_diagnose` queries `scan @ r.T + t` at line 440. Immediately
after it returns, V15:1467 calls `_context_support`, which queries the same scan
at the same rotation/translation at 1208. No intervening pose update occurs.
An explicit owned immutable query result passed to both consumers could remove
one complete full-scan surface query per completed candidate without changing
the coordinate model or acceptance gates. This is a candidate optimization,
not a measured speedup. It needs strict binding of index, row order/bytes, pose,
query policy and arithmetic/version identity, plus a prospective accounting rule
for performed versus reused work. Never reuse after a partial/unavailable query.

### 5. Frozen completion accounting prose disagrees with the actual callee

Runner 285–297 labels `max_node_visits`, `max_triangle_pairs` and
`max_normal_pairs` as inactive in the completion lane, and says its work units
are only box/query row visits. Actual V15:883 instantiates `_FaceQueries` with
the completion budget/statistics. `_FaceQueries.__call__` at 535–548 explicitly
allocates and charges node visits, triangle pairs, certificate pairs, normal
pairs and full surface work units; `screen` at 498–504 separately adds box rows.
The declared inactivity of those three fields and the row-only formula are
therefore false at the inspected source hash. In contrast, the declared inactive
`max_refined_start_groups` in the completion envelope really is not its refinement
limit: the main envelope controls the combined 48 slots.

This discrepancy does **not** demonstrate a cap violation or explain an OS CPU
kill. Actual enforcement is more extensive than the prose says. It is a frozen
specification/audit gap that a successor must correct. Do not repair the frozen
V9 source or protocol to make the historical record look consistent.

## Bounded successor recommendations and synthetic acceptance checks

These are recommendations to the implementation owner, not new physical-run
authority. Preserve every selected case, refusal, omitted candidate and unchanged
score denominator. Freeze new scheduling/failure semantics before any successor.

1. **Make the cost funnel observable and test the real workflow boundary.**
   Record bounded stage/lane/group identifiers, CPU and wall deltas, query
   counters, completed/attempted/omitted dispositions and last durable checkpoint
   before each costly phase. No raw rows or coordinates need be logged. Verify
   with tiny generated fixtures that the actual public workflow constructs the
   feature type the measurement consumer accepts; no direct construction of the
   downstream type or mock at that seam. This closes the separate wiring issue
   before performance work can claim reachable measurement.

2. **Reuse the identical final query first.** With a counted deterministic
   query double, prove one full-scan query serves both diagnosis and context
   support. Then use independently generated finite meshes with holes, edges,
   multiple surface identities, ties, awkward scales and threshold-near rows:
   original and reused paths must agree on all geometry/gate/uncertainty outputs.
   Changed pose, row permutation, index, query policy or an incomplete result
   must invalidate reuse. Resource records must distinguish actual work from reuse.

3. **Protect a bounded main-lane result, not just list order.** Define a new
   deterministic main/completion schedule with an explicit bounded completion
   share and a reserved final-verification/measurement allowance inside the same
   outer cap. Avoid prefiltering completion-only work that has no refinement
   allocation unless it is separately required diagnostic work. Consider a
   main-first fallback mode or lane-level transactions. An omitted or expired
   candidate must remain unknown, never become a geometrically rejected candidate.
   If publishing completed contexts when another lane is incomplete, revise the
   finite-subset completion contract explicitly; do not quietly remove V9's
   completeness gate or imply global pose uniqueness. Joint metrology can remain
   unavailable even if conditional diagnostic measurements are retained.

   Acceptance: a small independent main-valid synthetic fixture plus an expensive
   completion tail; unchanged main evidence is reached within a predeclared budget,
   tail omissions are retained, and no selected whole-workflow result disappears
   merely because optional completion work consumes its allocation. Add no-main
   completion-success, completion-all-rejected, completion-burst, cap-edge,
   terminal-failure and ambiguous alternative-pose cases. Use fake deterministic
   cost counters for exhaustive scheduler tests and a separately bounded tiny
   process-level CPU-stop test; do not burn 1860 seconds just to test failure.

4. **If necessary, separate inexpensive pose proposals from final conservative
   verification.** `_refine` already labels its fixed-materialized-target Huber
   objective an uncertified proposal surrogate (registration V4:359–365). A new
   bounded optimizer can use a cheaper represented-mesh nearest-point proposal
   path, then run the existing full finite-surface/uncertainty/coverage checks on
   the final nominated poses. This is a new proposal mechanism, not a numerical
   equivalence patch. Use finite triangles/trim-aware proposals, retain alternative
   poses and unknowns, and never use Gold, nominal CAD answers or approximate
   distances as measured truth. Benchmark it on generated independent poses,
   partial scans, sparse arcs, holes, adjacent bends, symmetric alternatives and
   real imposed deviations under the same total synthetic CPU budget. Require
   measured cost reduction and final-verifier correctness, not synthetic F1
   compared to physical F1. Keep the existing conservative refusal if final
   verification is not completed.

No physical improvement is proved by this report. It identifies why adding
geometrically useful starts can still make the deadline-bounded engine less
available, and an actionable route to distinguish and correct that mechanism.

## Hash ledger

SHA-256 of inspected source/protocol and the primary terminal artifact:

```text
4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1 scripts/run_gold11_mesh_development_v9.py
505f313dbf3e0487bf03f3b92ab1d42d90f61d5df3123c34c9a91677d132647e backend/cad_guided_plane_initialization_v15.py
57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87 backend/cad_guided_plane_initialization_v12.py
5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273 backend/cad_guided_global_registration_v4.py
f0a01f666f143184960f78f4cec2aab5b7a39510485b136b85b63fdb4225f0d4 backend/cad_guided_triangle_surface_v4.py
2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7 backend/cad_guided_triangle_distance_bounds_v3.py
e6fcd96aeb85bf3d91543f63f2001511ca22ff13a46c2bd1f1a51ca810fef54f docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md
792d35ad231db751c42f86a2326c8ff77d8ee8be74d4b8e1575aafd73fdbfe29 docs/research/gold11_mesh_development_v9_attempt/COMPLETED.json
```

Case hashes below are for `docs/research/gold11_mesh_development_v9_attempt/`
plus the displayed path. All 22 were hashed from the exact terminal paths and
match their primary terminal-record hash declarations; this is not a full
source/runtime/raw/reference/custody POST audit.

```text
0f6808abdf30eadbea9290fbfc4e88513c7c2d1e93f9c8e9c1e3ebcf8494ecc1 cases/CG08506AE97D/control/COMPLETED.json
683b6af78a05dc23a387f0705f23e7d061018f6fc12513ba5f3b92872f49649b cases/CG08506AE97D/challenger/COMPLETED.json
15b4de6a9e50ffc2c1563e6cea4d2655f0bd94faeb02e696f5872af7ce20e90b cases/CG191DEC7F81/control/COMPLETED.json
4e9e3e93700cf1f8773b2193ca9be75cf03e183ab79fbc83a44ae32245a223d0 cases/CG191DEC7F81/challenger/COMPLETED.json
5f35e6fe2cf131026d523b75c449ef8c627f30a87bbaaca60cd24e0777bb1bf0 cases/CG1E0BBA2933/control/COMPLETED.json
cceb23db884cc6c1b40a38aae75d9210970bd516cd253ac5d828ee528e6721b2 cases/CG1E0BBA2933/challenger/COMPLETED.json
7c264a6ee07bafa9b82bcc6281bd4a9e8af70011279331b9b8aa329872f6c831 cases/CG4BDD1371B6/control/COMPLETED.json
755487d58ffa3681293eab9c72f4a930f8335d15fb4c3f55b8d531b99c53eb00 cases/CG4BDD1371B6/challenger/COMPLETED.json
e6ceb975839f2547ec894af7d329831798c26fdf7da06264bbed714cf60e94dc cases/CG4FAF273093/control/COMPLETED.json
7533f2e66f8c6c6fb14eb610baa7156028f0b7387c4b60664af9e8ab20cb6856 cases/CG4FAF273093/challenger/COMPLETED.json
5d6c43fc7d4ab5b7052ceb43b7bbe7216ee9a29d12cf25ac3a298cb0c838a7f4 cases/CG64C412CC73/control/COMPLETED.json
a2e237c12fd05d23e59c9048e7ac9557270a9ca740ea1ac1555188a01e919b71 cases/CG64C412CC73/challenger/COMPLETED.json
ed8d0ef6a99934b4f692c7286c7aa99d18fdf49d19d4fce3cd9f6f5fe23b82c9 cases/CG65CD575B1C/control/COMPLETED.json
c930457c3a14ed45022cab1743dc6a2a31b65427b9daac07e2115a5e17b9e581 cases/CG65CD575B1C/challenger/COMPLETED.json
e91bd6944e1b259e0dbf80c39c7d0ca239f926c341857e66a42c4f0ac2b27d42 cases/CG84C8FA12D2/control/COMPLETED.json
8ae1bcfabbe60494e1c1f99f923197d69b130276583feb0e22e600b2eb08a679 cases/CG84C8FA12D2/challenger/COMPLETED.json
515204cf79e0d1c0e821b1bb59b98144a6528140a58ec833bb7e4fd232bf73e9 cases/CGE4B0549E41/control/COMPLETED.json
1ca8a5c23db75680d72febbc513314a98e018c536fb23066fb08a0bfd8b4840c cases/CGE4B0549E41/challenger/COMPLETED.json
33af7ac66255e6766eede701277ca295d3df6ec9784c6be350205423cfc180e0 cases/CGEBA9A869AC/control/COMPLETED.json
f24052338842c363fe636d9ea79099ae1c5fb668b665edbcdba210be30e3d2d2 cases/CGEBA9A869AC/challenger/COMPLETED.json
6c8a5aedd0a0fcf9874bbb6151a421f38fd50ed5ee922cd99810fc11e0822275 cases/CGF41F49066A/control/COMPLETED.json
b3c3d2bb41a7161325bc7c3d76f78c4daa16427df51cf0625b302684d46121af cases/CGF41F49066A/challenger/COMPLETED.json
```

Validation actually performed: named-source inspection; exact terminal metadata
projection/count/sum with `jq`; SHA-256 hashing; host signal-name listing; scalar
arithmetic. No tests or timing experiments were executed. Review stayed within
the assigned 30-minute / 60 CPU-second / 1 GiB small-analysis envelope; no heavy
job was started. A failed narrow search for a guessed certificate filename was
corrected using the surface module's actual import; no absent file was invented.
