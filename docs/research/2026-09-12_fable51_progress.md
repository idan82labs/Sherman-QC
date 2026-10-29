# Fable 5.1 clean-F1 implementation ledger

Started 2026-09-12 21:49 UTC. Owner: Claude Code, model `claude-fable-5-1`, effort
`max`, existing Claude Max subscription only. No model fallback, paid API,
billing change, new part, held-out, external deployment or permission bypass.
Codex remains the coordinating/auditing owner; earlier Codex subagents are idle.
Work happens directly in `/Users/idantal/ShermanQC/FullProject` on the existing
dirty/untracked research sources (no worktree: a fresh worktree from `main`
would omit every untracked `backend/cad_guided_*` source and all attempts).

Read before acting: handoff `2026-09-12_fable51_clean_f1_handoff.md`,
`AGENTS.md`, `CLAUDE.md`, newest `docs/CURRENT_STATUS.md`, `docs/REPO_MAP.md`,
the 2026-09-07 roadmap, the 2026-09-09 Gold11 input review checkpoint, the V7
result/diagnosis/POST/scoring POST, and the V2 index result/efficacy/contingency
notes. All existing attempts, frozen bundles and opaque caches stay untouched.

## Ledger

- 2026-09-12 21:49 UTC (date -u) — Acknowledged model/effort/ownership. Process check:
  no Sherman geometry or benchmark process live (only Codex/Claude host
  processes). Progress file created. Next: source inspection of the connected
  path V7 runner -> mesh offline V14 -> plane initialization V13 -> two-plane
  context V3 -> raw extrema V1, and the other V7 stopping mechanisms.
- entry written before 2026-09-12 22:26 UTC (date -u; earlier label was a local-time estimate) — Source inspection complete. Diagnosis basis: V7
  retained counters (derived artifacts only; no raw/Gold): CG085 control
  evaluated 104,408 leaves of 256 rows for 127 extrema queries (~45% of the
  818K-row buffer per whole query); CG4BDD control's single pose had 98.8%
  coverage but context p95 upper 0.666 mm > 0.5 gate; CG191 control had a
  SUPPORTED_CONTEXT candidate before a later width refusal (native solid
  invalid anyway). Decision: successor = new versioned modules only, frozen
  sources untouched, MW (CGE4B noisy membership cap) deferred (deep composite
  chain), PF (CG64/CG65) untouched (no diagnosis basis).
- NEW `backend/cad_guided_raw_extrema_index_v3.py`: rank-partition kd-tree
  (global halved-root-span axis schedule, stable per-node sort, 64-row leaves),
  V1 staged program/query/ties verbatim; declared coordinate maximum 512M.
  NEW `backend/cad_guided_two_plane_context_v4.py` (V3 text, extrema V3).
  Tests: `tests/test_cad_guided_raw_extrema_index_v3.py` 67 passed;
  `tests/test_cad_guided_two_plane_context_v4.py` 79 passed. Command:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider <file> -q`.
  Exactness: byte-identical intervals/witnesses to V1 on the four consumed
  32,768-row populations x 12 queries, a 200K-row invented folded sheet x 19
  queries, and the scalar/blocked oracles. Development efficacy (invented 800K
  folded sheet, 184 CG085-like whole-buffer queries, script in job tmp):
  V1 refuses at 256M coordinate visits after 26.2M rows (142.5K rows/query);
  V3 leaf64 completes at 122.0M coordinate visits, 10.19M rows (55.4K/query),
  sort work 150.3M, CPU 4.8 s; leaf256 161M/2.35 s; leaf32 104M/7.3 s.
  Consumed 32K populations: flat/plateau +4% total work (construction, no
  pruning gain), resolved -67%, folded -5%.
- NEW `backend/cad_guided_triangle_surface_v4.py` (V3 + sound nearest-vertex
  pass-1 pruning threshold, 256 tie slots, wide competitive interval -> row
  uncertainty flags instead of whole-query refusal, 16 declared units per
  vertex lookup); NEW `backend/cad_guided_global_registration_v4.py` (V3
  consumer bound to surface V4, retained wire strings, REGISTRATION_IMPLEMENTATION);
  NEW `backend/cad_guided_stage_policies_v2.py`. Tests
  `tests/test_cad_guided_triangle_surface_v4.py` + `tests/test_cad_guided_global_registration_v4.py`:
  132 passed (includes: frozen V3 refuses a 33/40/96-triangle fan at 32 ties
  and the needle fixture with UNSUPPORTED_COMPETITIVE_TRIANGLE_INTERVAL_WIDTH,
  V4 completes both with exact enclosure; generic 400-triangle soup x 189 rows
  byte-identical to V3 with triangle pairs 1426->1182).
- 2026-09-12 22:26 UTC (date -u) — Codex coordination note received: two read-only preliminary
  reviews are in progress (`2026-09-12_fable51_surface_v4_codex_preliminary_review.md`,
  `2026-09-12_fable51_extrema_v3_codex_preliminary_review.md`); I will read both
  before any freeze. All synthetic leaf-size variants (256/128/64/32) and the
  800K efficacy script runs above are consumed development measurements, not a
  frozen efficacy gate. Ledger timestamps now use `date -u`.
- 2026-09-12 22:36 UTC (date -u) — Codex preliminary reviews read (receipt + two reports,
  no P1). Applied the narrow follow-through; delta recorded in
  `2026-09-12_fable51_review_response_v1.md`. Suites after the delta:
  extrema V3 + context V4 + surface V4 + registration V4 = 283 passed;
  NEW `backend/cad_guided_plane_initialization_v14.py` (V13 text on context
  V4/surface V4/registration V4/stage policies V2, with verbatim V12 copies of
  _FaceQueries/_prefilter/_context_support/_configuration), NEW
  `backend/cad_guided_mesh_offline_v15.py`, NEW `scripts/gold11_schedule_projection_v3.py`;
  `tests/test_cad_guided_context_v4_engine_integration_v1.py` +
  `tests/test_cad_guided_context_v4_schedule_fixture_v1.py` = 101 passed
  (the stale v2-schedule variant, a known terminal failure in the V2
  selection, is replaced by the ported actual-codec fixture);
  `tests/test_cad_guided_plane_initialization_v14_registration.py` 1 passed:
  unmocked V14 registration on the invented two-sheet fixture reaches
  CONDITIONAL_MEASUREMENT_CONTEXTS with a supported context (3 s CPU).
  Working tree vs V7 frozen sources: no drift (all backend/scripts identical).
- 2026-09-12 22:41 UTC (date -u) — Successor wiring complete and test-verified (source only):
  NEW `scripts/run_gold11_mesh_development_v8.py` (V7 text; V15/V14/context V4/
  surface V4 entry bindings; projection V3 in SCORING_SOURCES; declared
  CONFIG deltas: registration p95 gate 0.5->1.0 mm, surface ties 32->256,
  context coordinates 256M->512M; frozen in the bundle as
  `config_delta_from_v7`), NEW protocol `2026-09-13_gold11_mesh_development_v8_protocol.md`
  and design `2026-09-13_gold11_mesh_development_v8_delta_design.md`.
  `tests/test_gold11_mesh_development_v8_source.py` 6 passed (identities,
  unchanged V7 values outside the declared deltas, SOURCES cover the V15
  import closure, JSON-stable arm requests, real constructor bindings for
  both arms, inert freeze/verify round trip + basis/profile refusals).
  Current source hashes recorded in the job tmp file v8_source_hashes.txt and
  below for reviewer binding. Next: three independent read-only reviews
  (wiring; runner/projection/protocol; post-review delta), then freeze, PRE.
  83dea6529003f1d5deae592c9d35e566a77e2c9f6f3af8f641928f5465845bc0  backend/cad_guided_raw_extrema_index_v3.py
  e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25  backend/cad_guided_two_plane_context_v4.py
  f79b73b37be2684f73f7d467d5aa1074f8a27d968cdf52e9fb1d652dae02422c  backend/cad_guided_triangle_surface_v4.py
  5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273  backend/cad_guided_global_registration_v4.py
  699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d  backend/cad_guided_stage_policies_v2.py
  eb1ad06a1847634585aeaa8e79779995d85f6311d17da4aea0dc96b7891b21d7  backend/cad_guided_plane_initialization_v14.py
  5edaae998a56c121a0aabf45fcc5b71d19e4b4aeab10209d42b89897dbd0f4f7  backend/cad_guided_mesh_offline_v15.py
  763e0cd032521f237aaa999601f67e75644fe430b9837bedf1dc2c4d9be82d30  scripts/gold11_schedule_projection_v3.py
  98ae00a0ed63ae1efd15e032a5468c6b9e5fab8d7be4e865df0519c4333cd09d  scripts/run_gold11_mesh_development_v8.py
  65ed24e86453bcc094d08bc09b7c0d7fb15fd814787446c55a0a1dbd0a25cca5  docs/research/2026-09-13_gold11_mesh_development_v8_protocol.md
  6847dff1bc8276be3ecd509cd1dfbbd147ee3ed2d1a588ef323a06241341c921  docs/research/2026-09-13_gold11_mesh_development_v8_delta_design.md
- 2026-09-12 22:56 UTC (date -u) — Independent delta recheck complete (no P1/P2, four P3):
  report copied verbatim from the reviewer's job-tmp output (the background
  reviewer cannot write into the checkout) to
  `2026-09-13_fable51_review_delta_independent_recheck.md`, SHA256
  40ad51fbdab0beb8d0a971296e59028e3f0f6ff9599024e7d1d6904d3248defb. Codex's two
  delta reviews (receipt 2026-09-12 "Subsequent bounded delta closure") closed
  the earlier findings at source; net efficacy stays unqualified by design.
  P3 items queued until the wiring and runner reviews bind their hashes:
  (1) retain the leaf128 figure here: leaf128 = 141.7M coordinate visits /
  12.46M rows / 3.31 s CPU on the 800K/184-query load (measured with leaf
  256/64/32; all consumed development); (2) reword surface V4 line 33
  "interval-width requirement"; (3) note the unresolved-lower-bound mechanism
  in the mixed-row test comment; (4) record index-visit growth (4S+2P, about
  28.5M for a 900K buffer, under the 64M parent cap) in the successor resource
  notes.
- 2026-09-12 22:58 UTC (date -u) — Independent wiring review complete (no P1; P2-1/P2-2 test
  gaps; P3-1..5): `2026-09-13_fable51_wiring_v14_v15_independent_review.md`,
  SHA256 436e21a89b18211fa9c5a5e08d0c83d51d2aea11cb00f0fee6b3c8efb4c9236c.
  Applied now (files outside the still-running runner review): P2-1 wrong/
  non-string registration label refused (parametrized test), P2-2 verbatim
  assertion now covers `_frozen_configuration_v4`, P3-1 additive
  `finite_surface_implementation`/`finite_surface_work_note` fields in V14
  `_base()` (retained V12 wire strings untouched), P3-2 docstring precision,
  P3-4 unused `_previous` alias removed and stale test name renamed; delta
  recheck P3 (2) surface V4 module docstring wording and (3) mixed-row test
  mechanism comment + `distance_lower[0] == 0.` assertion. Not changed: the
  unreachable `two_plane_enabled=False` branch (verbatim V13 text; noted).
  Rerun: surface V4 + registration V4 + integration V1 + schedule fixture V1 +
  V14 registration = 241 passed. Reviewer note retained: predecessors are
  untracked in git; "unchanged" is attested by hash stability plus byte
  equality with the V7 attempt's frozen source copies (checked earlier).
  Queued until the runner review binds: V15 stale comment (P3-4), design-doc
  index-visit accounting note, protocol/design "combined consumed-development
  changes / schedule-lane arm comparison" wording.
- 2026-09-12 23:09 UTC (date -u) — Runner review complete (no P1; P2-1..4, P3-1..5):
  `2026-09-13_fable51_runner_v8_independent_review.md`, SHA256
  576007a86daf10bd51757b58f3c268c40d70fbcb6808bf221a237ec90acbce65, including
  the requested pose-admission-gate assessment (deciding consumer is V14
  `_context_support`; `_select` is off the connected path; real risks are
  measurement preflight refusals from more contexts and wrong-correspondence
  FPs; no new FN mechanism). Applied: P2-1 stage sampler names the V8 runner
  (+ test); P2-2 NEW `tests/test_gold11_schedule_projection_v3.py` (254
  passed, derived from the V2 selection with registration-label cases);
  P2-3 narrow gate evidence: `_context_support` mocked-query test (p95 0.75:
  REJECTED at .5, SUPPORTED at 1.0; coverage gates unchanged); P2-4 resolved
  by a NEW deterministic measurement-context schedule in V14
  (`_schedule_measurement_contexts`: p95-asc/coverage-desc/index ranking,
  pose-class dedup at 0.1 mm with the frozen V2 action metric, cap) with V15
  deriving the cap from the frozen consumer's window/crop budgets and the
  declared policy maximum 8 (`workflow.max_measurement_contexts`, added to
  `CONFIG_DELTA_FROM_V7`); the protocol/design also declare the remaining
  aggregate-budget outcome class; P3: blocker tuple reworded, arm-request
  wording, `residual_gate_value_scope` note, `POSE_ADMISSION_DECLARATION`
  and combined-change/arm-scope labels frozen in the bundle and enriched
  report, V15 stale comment. Docs wording patch (combined consumed-development
  changes; schedule-lane arm comparison) applied to protocol and design.
  Suites: schedule/gate/cap tests added; full new-module rerun below.
- 2026-09-12 23:09 UTC (date -u) — Full new-module rerun: 653 passed (extrema V3 67, context
  V4 79, surface V4 + registration V4 137, projection V3 254, integration V1 +
  schedule fixture V1 + V14 registration 110 incl. schedule/gate/cap, runner
  source 6). Command: the isolated pytest invocation above over the nine
  files. Next: final narrow independent review of the schedule/runner/doc
  deltas, then exclusive `--freeze`, independent PRE, no-overlap, execute.
- 2026-09-12 23:34 UTC (date -u) — Final narrow independent review (schedule + runner/doc
  deltas) complete: `2026-09-13_fable51_context_schedule_independent_review.md`,
  SHA256 7f7b7ed886c7e086d7385c653a4155fe505928ab4993be1c4d7c05881088814e
  (copied verbatim from the reviewer's job-tmp output): no P1; the schedule
  is judged a legitimate label-free capacity policy. Closed: P2-1 cap-branch
  tests (600 features -> cap 6; crop budget 100 -> 2; floor -> 1), P2-2
  stronger unmocked cap assertions + fixture note (one SUPPORTED context),
  P2-3 `SOURCE_PROFILE.source_delta` + docstring name the schedule/cap,
  P2-4 post-review hash drift recorded here: surface V4 now
  f0a01f66-prefixed (docstring-only P3-2 wording, +72 B) and registration V4
  test e108dc3a-prefixed (mechanism comment + `distance_lower[0] == 0.`,
  +225 B) versus the Codex delta-review bindings f79b73b3/1cf31b12; P3-1
  "four" deltas; P3-2 dedup-regardless-of-cap sentence in docstring/design and
  `pose_class_collapse` field, key renamed to
  `measurement_context_schedule_policy`; P3-3 dispositions decided before any
  status mutation; P3-4 explicit `MEASUREMENT_CONTEXT_CAP` phase; P3-6
  positive control for the additive keys in the projection V3 test.
  Full rerun: 657 passed. Final pre-freeze hashes (also in job tmp
  v8_final_hashes.txt):
  83dea6529003f1d5deae592c9d35e566a77e2c9f6f3af8f641928f5465845bc0  backend/cad_guided_raw_extrema_index_v3.py
  e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25  backend/cad_guided_two_plane_context_v4.py
  f0a01f666f143184960f78f4cec2aab5b7a39510485b136b85b63fdb4225f0d4  backend/cad_guided_triangle_surface_v4.py
  5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273  backend/cad_guided_global_registration_v4.py
  699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d  backend/cad_guided_stage_policies_v2.py
  9e3ca41edd05bcc0b6fcdeb9ef0447dbacfe58df8c9458da20f9bb12c3d925bf  backend/cad_guided_plane_initialization_v14.py
  c8f38f496fdd11148cec04a8b5ba0f1366cc548efca59a14b48451b5b5c57148  backend/cad_guided_mesh_offline_v15.py
  763e0cd032521f237aaa999601f67e75644fe430b9837bedf1dc2c4d9be82d30  scripts/gold11_schedule_projection_v3.py
  0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5  scripts/run_gold11_mesh_development_v8.py
  a39e501a01890bfa73daec461a0f27bec8eebb166b5709c3d97248211beef385  docs/research/2026-09-13_gold11_mesh_development_v8_protocol.md
  060e3eaa595cda4e96ef1eb21fb16d74e7ab47f5f8a98c6d79920b13ae83a061  docs/research/2026-09-13_gold11_mesh_development_v8_delta_design.md
  f78c18d146d4ca2f86e57a9690daa58251e92b2dee207de1a499a61ece6cc572  tests/test_cad_guided_raw_extrema_index_v3.py
  ef57b38f23249c8c1555d0c61fd05913a8068027147ca020b082679154d7a945  tests/test_cad_guided_two_plane_context_v4.py
  cc80b5c7f3f676176804d1d5c15855469ca2c17485672acc0f6549fd46c24db8  tests/test_cad_guided_triangle_surface_v4.py
  e108dc3a94651883013d15da655bb25c2a716851277b0b761c64e790f0969752  tests/test_cad_guided_global_registration_v4.py
  7a83b12d58f99fb5f77b350a0ccfabd12e17740a9c295fe895a1fdd4a265cca9  tests/test_gold11_schedule_projection_v3.py
  712602ef7d3334f866d60b76edc9f4ec06a44646c02a064d923558a3374cd49a  tests/test_cad_guided_context_v4_engine_integration_v1.py
  e75497725ed3c7d7b5f208d2d9094d5650bedc894d7c11d4a772508644af45a6  tests/test_cad_guided_context_v4_schedule_fixture_v1.py
  2b5ec9ed84dfa568c32db8a77755d049feff4fbc2af1578b56157a893bdc98df  tests/test_cad_guided_plane_initialization_v14_registration.py
  a7c20b0c887af1d92726be54a976f63d53b12a6da5a7e0072f2d912796e0bdb4  tests/test_gold11_mesh_development_v8_source.py
- 2026-09-12 23:34 UTC (date -u) — Exclusive source-only freeze executed once (no overlapping
  Sherman process per ps): `OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v8.py --freeze`
  exit 0, printed bundle_sha256 43173e8a9c59c95c188e17c0c2901b8067fecf349ec363c05ab024c261fa9326,
  attempt `docs/research/gold11_mesh_development_v8_attempt` (FROZEN.json
  39670 B, file SHA256 03f58d23a3733434cfac6d51c031974088e660d0c880542ceb5c8182f3032961;
  74 frozen sources). External archive `docs/research/gold11_mesh_development_v8_physical_frozen.x9er0oPw/SOURCE.tar`
  (SHA256 35cad3685658cfab5f1ab824c438c6284a0ba47e7970d4fa812164065bfae103,  2839040 B) created for POST custody; no
  raw/Gold access. Next: independent exact physical PRE (reviewer writes to
  job tmp; artifacts copied verbatim with hash verification), then the sole
  `--execute`.
- 2026-09-12 23:51 UTC (date -u) — Independent physical PRE complete (non-author AI auditor,
  read-only; wrote to job tmp, copied verbatim): decision
  GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN, bundle_sha256 recomputed
  independently = 43173e8a9c59c95c188e17c0c2901b8067fecf349ec363c05ab024c261fa9326,
  blocking_findings []. Checks: 74/74 source hashes agree three ways
  (working tree, snapshot, declared); SOURCES tuple = map keys; V7 basis hash
  unchanged; 17/17 runner constants equal the bundle incl. arm_requests and
  runtime metadata; case authority from the frozen review_gold11_inputs_v1.py
  (11 cases, 11 scan / 6 CAD digests, raw17 map digest 2459f4e9…d157, case
  list digest 943268be…813e identical and same order); attempt tree pristine;
  archive tar lists FROZEN.json + 74 sources; protocol limits equal LIMITS;
  no P1 / no unresolved P2 (runner-review P2-3 second half superseded by the
  dedup schedule, recorded as a non-blocking test gap); ps overlap none.
  Files: `docs/research/gold11_mesh_development_v8_attempt/PRE_RUN_AUDIT.json`
  (20736 B, SHA256 41fd61fc6157e44823477e04da034d4b378307055cb325e0bcb19f1d7c7d6554) and
  `docs/research/2026-09-13_gold11_mesh_development_v8_physical_pre_run_audit.md`
  (19219 B, SHA256 a0faba1a32378f84dfe176e9fb1bb839e1c60bc8ec487b84f53b5b158abb2352);
  both hashes verified identical after copy. Next: fresh no-overlap check
  immediately followed by the sole `--execute` (stdout to job tmp; ps-only
  monitoring; no attempt/cache artifact read during the run).
- 2026-09-12 23:52 UTC (date -u) — Sole physical execution launched at 2026-09-12 23:52:07 UTC
  after an in-launcher fresh ps check (no run_gold11_/check_/pytest/cad_guided/
  scoring process) and a tree check (exactly FROZEN.json, PRE_RUN_AUDIT.json,
  source/): `OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v8.py --execute`
  from the repository root, detached in its own session (parent pid 37160;
  internal coordinator and first leaf worker observed via ps within 5 s, tree
  RSS 1.29 GiB). Parent stdout/stderr go to the job scratch file
  `v8_execute_stdout.txt` (outside the checkout). Monitoring is ps-only until
  the parent exits; no attempt or cache artifact is read during the run.
- 2026-09-13 02:31 UTC (date -u) — Sole V8 execution terminal. Parent exited at 2026-09-13
  02:30:42 UTC (external ps monitor; elapsed 9480 s from the monitor start;
  no orphan process). Final printed status: COMPUTED_PENDING_INDEPENDENT_POST_AUDIT,
  durable_terminal_case_records 22, final_campaign_inventory_sha256
  dbeb824d1929e6f3b638cfc3cc1f8ed13899fc5a5da39a62d5fbc7a2912445bb (283 entries,
  115 scientific manifest entries). FINAL_GATE.json present and no
  FINALIZATION_FAILED.json (execute() returned 0 by construction; the detached
  launch did not capture the shell exit code). Observed outer tree RSS peak
  2.41 GiB (20 s ps sampling, not continuous). Reading derived attempt records
  only now (never during the run); independent POST launched.
- 2026-09-13 02:39 UTC (date -u) — V8 derived records read (post-terminal only) and the result
  narrative written: `docs/research/2026-09-13_gold11_mesh_development_v8_result.md`.
  Honest headline: both arms TP 0 / FP 0 / FN 71, precision null, recall 0,
  localized F1 null (undefined by the unchanged scorer; no numeric F1 exists;
  no improvement over V7; historical 0.4580152672 not comparable). Immediate
  causes over 22 slots: 12 no-passing-start (CG08506/CG64/CG65/CG84/CGEBA/CGF41
  both arms), 4 CPU kills inside surface V4 queries during registration
  (CG191/CG4FAF both arms, exit -24), 2 membership cap (CGE4B), 1 residual gate
  rejection at 1.0 mm (CG4BDD challenger, p95 upper 1.2009 mm), 3 admitted
  contexts with zero valid support-cover measurements (CG1E0 both arms p95 upper
  0.9008 mm; CG4BDD control 0.6956 mm; all 43 native windows refused:
  39 FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY, 4 FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL).
  Extrema coordinate-cap refusals: 0 (V7: 10); all completed contexts used
  25.1M–89.2M coordinate visits. Parent wall 9512.744 s, parent peak RSS
  1,557,233,664 B, failures [], final source audit unchanged. POST pending.
- 2026-09-13 02:45 UTC (date -u) — Correction per Codex coordination note: the parent's
  process exit status is UNOBSERVED (detached launch, no waiting parent; ps only
  observed disappearance at 02:30:42 UTC). FINAL_GATE.json present and
  FINALIZATION_FAILED.json absent are consistent with a zero return from
  execute() but do not prove the exit status; the earlier ledger wording
  "returned 0 by construction" is withdrawn. The runner's release condition
  OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT
  is therefore not met by observation; this limitation is recorded in the
  result document and CURRENT_STATUS, and the running POST auditor was
  instructed to retain it rather than infer exit 0.
- 2026-09-13 03:00 UTC (date -u) — Independent POST received and placed verbatim:
  `docs/research/2026-09-13_gold11_mesh_development_v8_post_run_audit.md`
  (44587 B, SHA256 894dfb8a606a02825e4732883449352220b086c5897e76c5ba953a218fca10c1;
  hash verified after copy). Decision ACCEPT (engineering audit only) with the
  exit-status limitation retained; all bindings reconcile; no retry/resume/
  relaxed limit/edited source. V8 is CLOSED and terminal. Result document
  updated (acceptance section; gate-metric table now shows inlier-plane p95
  0.6659/1.1959 mm and context p95 upper 0.6956/1.2009 mm
  for CG4BDD C/H; my earlier "discrepancy" note withdrawn: 0.666 and 0.6956 are
  two different V7 fields reproduced exactly). Per Codex: continue the
  handoff toward an audited numeric F1 with a new versioned successor (no V8
  retry/rescore); Codex has two source-only reviewers on flange-policy math and
  finite-start rejections; I remain sole engine editor.
- 2026-09-13 03:03 UTC (date -u) — V9 diagnosis written: `docs/research/2026-09-13_fable51_v9_diagnosis.md`
  (derived records + synthetic fixtures only; scripts in job scratch:
  completion_analysis.py, completion_feasibility.py, flange_gate_synthetic.py).
  Tested hypotheses: (A) failed third supports lie ON their assigned CAD plane
  (|signed dist| ≤ 0.3 mm) but outside the finite face box by 25–254 mm; no
  alternative third face was ever scheduled for the two passing supports;
  exact-box feasibility finds completions in 4 of 6 no-passing scans (CG08506 4,
  CGEBA 2, CGF41 2, CG64 2; CG84 0; CG65 none). Context lane: endpoint slots
  always proposed, midpoints absent when the heuristic interval is empty, and
  no proposed slot passes in any of 8 scans. (B) Frozen flange gate on
  synthetic strips: single skin σ 0.01 passes, σ 0.02 fails residual; two skins
  1.5 mm apart → WEAK at s3/s2 0.262 (observed reason); derived scan planes show
  105,805 / 126,424 / 117,720 parallel overlapping pairs at ~1.3–1.6 mm gaps
  with CAD thickness faces at 1.5 mm → both skins are in the scans and inside
  the 3.0 mm crop margin. Mechanisms: V9-A triplet completion lane; V9-B
  material-side layer selection (gates unchanged). Codex reviewers remain on
  flange math and finite-start rejections; no competing engine edits.
- 2026-09-13 03:07 UTC (date -u) — Codex 03:05 coordination section read (both source
  audits accepted by root; opaque cache /private/tmp/gold11-v8-coordinator-6ebyksl4
  added to the never-inspect list). Corrections applied: the V8 result document's
  claim that admitted-pose residuals caused the window refusals ("false-pose
  consequence observed") was unsupported and is withdrawn; the global proximity
  p95 is not the local TLS residual (centered SVD is pose-invariant for fixed
  rows). The V9 diagnosis now labels two-skin contamination as the leading
  hypothesis to be discriminated by per-window diagnostics (N, spectrum,
  selected-row/extent provenance, residuals, both flange sides and both
  weak-gate branches), with narrow windows, noise and wrong-context crops as
  competing causes; final gates stay unchanged and no layer is trimmed for fit
  quality. POST addendum requested from the non-author auditor for the
  8-vs-2 former-extrema-slot typo; the original POST is preserved verbatim.
- 2026-09-13 03:12 UTC (date -u) — POST addendum received and placed verbatim:
  `docs/research/2026-09-13_gold11_mesh_development_v8_post_run_audit_addendum.md`
  (6953 B, SHA256 39b2547c350ca93b81ba6cfc8476d08d03ae47c3c75a892283eda0e0af981f8d;
  original POST re-hashed unchanged 894dfb8a…ca10c1). Correction: 8 former
  extrema-refusal slots -> no passing start, 2 (CG1E0 C/H) -> measurement
  refusal; all counts/scores unchanged; decision unchanged. Result document
  acceptance section updated. Both Codex source audits read (flange
  7e5f8ecc…37c0, finite-start 3b4b4da9…7823); V9 scope fixed to the completion
  lane plus per-window diagnostics with declared material-side layer
  selection; cone-feasible witness fallback and finite-support axial search
  recorded as deferred. Measurement kernel V2 drafted
  (`backend/cad_guided_measurement_v2.py`, not yet tested).
- 2026-09-13 03:17 UTC (date -u) — NEW FINDING (diagnosis Part E): the frozen measurement
  policy (δ 0.05, e 0.075) is infeasible for the cohort's nominal geometry
  (all 99 V8 native windows: r 0.78–0.80 mm, 90°): the plane-exclusion band
  leaves a ≤49° usable arc whose sagitta (≤0.073 mm) is below the
  flat-witness requirement (half-width > δ) for ANY e > δ = 0.05; the frozen
  kernel refuses a perfect noise-free single-skin r 0.8/90° bend as
  ARC_COMPATIBLE_WITH_FLAT (script `arc_policy_feasibility.py` in job scratch).
  δ ≤ 0.03 (e 0.04) measures perfect data with margin. Proposed for V9 as a
  separately derived, pre-declared config delta with gate forms unchanged
  (δ 0.03, e 0.04); noise sensitivity documented (σ 0.005 measures, σ ≥ 0.010
  refuses NOT_PLANAR). Kernel V2 smoke-tested: two skins t 1.5 -> V1 WEAK,
  V2 selects inner and measures (both sweep signs); outer-only scan -> single
  layer, identity UNRESOLVED (never labelled inner). Context measurement V2
  drafted (`backend/cad_guided_context_measurement_v2.py`).
- 2026-09-13 03:36 UTC (date -u) — V9 implementation (source only, unfrozen): NEW
  `backend/cad_guided_measurement_v2.py` (both-flange/both-branch diagnostics,
  declared material-side layer selection with explicit sign conventions,
  axis-side arc filter, SINGLE_LAYER_IDENTITY_UNRESOLVED), NEW
  `backend/cad_guided_context_measurement_v2.py`, NEW
  `backend/cad_guided_support_cover_measurement_v3.py`, NEW
  `backend/cad_guided_plane_initialization_v15.py` (V14 text + additive
  two-of-three completion lane with CompletionLanePolicyV1, separate budgets
  and ledger), NEW `backend/cad_guided_mesh_offline_v16.py`, NEW
  `scripts/gold11_schedule_projection_v4.py`, NEW
  `scripts/run_gold11_mesh_development_v9.py` (basis V8 runner
  0e9aa2f6…60c5; CONFIG_DELTA_FROM_V8: measurement δ .05→.03, e .075→.04
  (derived, Part E), NEW completion/layer policies; MEASUREMENT_POLICY_DECLARATION;
  1.0 mm gate unchanged), NEW protocol/delta-design docs. Tests (inert
  namespace): kernel V2 18, context V2 168, support cover V3 82, initializer
  V15 registration 6, completion lane 11 (recovers the correct third face from a
  parallel wrong assignment, both shift signs; disabled/omission/parallel-pair/
  axis-parallel accounting), engine integration 102, projection V4 255, runner
  V9 source (run pending). Frozen predecessors untouched; no freeze yet.
- 2026-09-13 03:38 UTC (date -u) — Combined V9 suite: 648 passed. Two independent non-author
  AI source reviews launched (kernel V2 chain; completion lane/wiring/runner V9),
  reports to job scratch, to be copied verbatim with hashes. Codex reviewers
  continue their source-only audits of the same designs. Pre-review candidate hashes:
  59555408cf92d3bdb7eae711b5ed7edfc68c0d9d518bb683c78e30dd61630d6d  backend/cad_guided_measurement_v2.py
  cc10db40e376ed0786f27babcac40b8261633955a8c9926331f8c9d9414e4ce0  backend/cad_guided_context_measurement_v2.py
  3f53d054165468d9197f7e0e0b3c6b564267880906377c1fb95f964457b2b74d  backend/cad_guided_support_cover_measurement_v3.py
  ce4a61af90db1339ace0914cd359ea6e0afdf22dd5b9f038c1e2d9d7f1eeaba4  backend/cad_guided_plane_initialization_v15.py
  441c618d4998edaca0d401dd22f14d1444f248750a807386fecef19741708769  backend/cad_guided_mesh_offline_v16.py
  06869ff7bb8761eae976f294e733713c3f2d94a83ee0c8d756c90d50afccfa90  scripts/gold11_schedule_projection_v4.py
  f8a9b7f28af81e842f187f6ff76ae87b80b59809561300785064ff63c25027e8  scripts/run_gold11_mesh_development_v9.py
  424e0d2d796fc5d4a7a3858bad56298bc3dee71993a407ecd9315245b395dd46  docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md
  1f4d52492a904703bf84302c8c5d6bc53cfb22157b0a87d9dd6272caeb574522  docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md
  2dd922f7871d10e9718899ecb6719f57da6cdf72db5f89d8fa16ba807993a7f9  docs/research/2026-09-13_fable51_v9_diagnosis.md
  13e2fbf913313cef2ce78a61e2dc4d3a7f16afd75f405bb46188d9347a66f67b  tests/test_cad_guided_measurement_v2.py
  9fd6c11e987cd260e0247154b901ebc67e600e0dba5766787a22b907ffbf2378  tests/test_cad_guided_plane_initialization_v15_completion.py
  462619e1494e845ffcf3db68dfa1da6e1884fcf8682be41802fe0c78e0d238fe  tests/test_gold11_mesh_development_v9_source.py
- 2026-09-13 04:43 UTC (date -u) — V9 CLOSURE CHECKPOINT for Codex's one final bounded review. All Codex
  delta-audit and independent-review findings closed at source (map with hashes and the exact test per
  finding: docs/research/2026-09-13_fable51_v9_review_response.md, SHA256
  31def50d1d4795c22f8c2c79b8899eafe445c388ecfd7041b1c4697da0985de4). Closures this round: context V2
  type-strict identity + per-context measurement revalidation; kernel V2 lossless partition provenance
  (sort normal/centre/split threshold/kept rule), unqualified large gaps, honest names
  (`projected_bounding_interval_overlap`, `min_projected_interval_overlap`, `qualifying_groups_detected`),
  diagnostics retained on numerical failures with explicit phases, attempted diagnostics kept on shared-
  support invalidation; V15 lane byte cap re-applied after the last mutation on every terminal path;
  V16 origin binder binds the parent's unique rejected slot, replacement face/sign, unchanged passing
  correspondences and the lane parent record; envelope V7 malformed-evidence propagation; runner
  `completion_budget_declaration` (inactive inherited fields, work-unit formula), angular-gap 5->30
  declared with fixture-only scope (assumed uniform-column model, not a cohort spacing statistic; not
  phase-robust; one plausible margin), `MEASUREMENT_POLICY_DECLARATION` caveats; durable waiter
  scripts/gold11_v9_execute_waiter.py records the actual outer return code beside the attempt directory;
  protocol/design/diagnosis corrected (four-change inventory, consumer bindings, eligibility contract,
  restricted theorem, H1 fixture-only, per-patch partner coverage 97.5/99.5/92.8%, proxy relabel,
  coupled deltas, Part F). New tests: completion signed-shift expectations (vacuous `or True` removed),
  cap fixed point, live V16 completion-only seam + nine-way stale-mapping tamper matrix, late fatal after
  a passing completion; kernel provenance/sparse/holed-proxy/numerical/invalidation; angular-gap adapter
  checks; connected chain with the REAL crop: positive, generic-unresolved, outer-only, frozen-policy
  refusal, 5 deg refusal vs 30 deg admission, coexisting forced canonical refusal, disabled control,
  emitted-bytes V4 wrapper transport (byte-equal to direct V3; frozen V2 refuses), tamper matrix.
  Actual results at the checkpoint hashes (inert namespace): 826 passed / 0 failed in 31.55 s across
  twelve suites (28/180/82/22/6/102/8/123/255/6/9/5). Checkpoint source hashes:
  a894b96057ca886981a6c8ac4f968519d4c6246af05254eddd03ed41e9864ad1  backend/cad_guided_measurement_v2.py
  0504cd32364a1708d17f8bddafba51a9f9640edc11866e33b37e693a0d4a3534  backend/cad_guided_context_measurement_v2.py
  a298b123218d931911e064a2f35ab52e5553f2c1ad5b25a71e3c149c2d7d9234  backend/cad_guided_support_cover_measurement_v3.py
  505f313dbf3e0487bf03f3b92ab1d42d90f61d5df3123c34c9a91677d132647e  backend/cad_guided_plane_initialization_v15.py
  5b48ad81c2d1796529c6c43cd7f27b48f73413b50f3adb74d9e6cf9608519012  backend/cad_guided_mesh_offline_v16.py
  b7abe653c6f7b9832c61687ec7f1ed68aa44361b8ef467b8cae84bf837bef075  backend/cad_guided_report_envelope_v7.py
  55cf0920b545e0aa204da1b0882298f92ca7015210466b6e8fc777ddafacf10d  scripts/gold11_contact_scoring_v3.py
  32029cbb813381e360bed791e97701cdfb6735cfa6fa12f9a7f90d2477b95708  scripts/gold11_schedule_projection_v4.py
  ad61e12b0a06c0cafb7e515687019e42c6e137e26a419753aaeef2360f5b4241  scripts/run_gold11_mesh_development_v9.py
  a0b8c0b3e3ddb2e3041f25f7bd5ea0a56b4430871996f3f0c957a2823ad6f680  scripts/gold11_v9_execute_waiter.py
  16f20a99766838c6ca210be86582c40089aafeae70540dc6d874848ab5b4c1f2  docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md
  03fa29937f2614971a515028bb9c500cb1736f91f962b275fb9cd81082ae0be9  docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md
  fddc8e3644561ad7c5f8766d166a0f99edfcffb006613544e300967e00a301e7  docs/research/2026-09-13_fable51_v9_diagnosis.md
  (test-file hashes in the response document). Independent reviews copied verbatim:
  docs/research/2026-09-13_fable51_kernel_v2_chain_independent_review.md 7d0afddd373c9b38d4b34d19a2c101294857c0b4e915350a5854068c360b3688,
  docs/research/2026-09-13_fable51_completion_wiring_runner_v9_independent_review.md 4ee023d02c26d426ae11a2774aeb994314a9d476bd0f2cac68a8a668f90e3c28.
  No freeze yet (Codex: no freeze until closure); no physical run, raw/Gold/attempt/cache access.
  Next on Codex's closure: exclusive --freeze, independent PRE, waiter-launched single --execute, POST.
- 2026-09-13 05:06 UTC (date -u) — V9 APPEND-ONLY DELTA CHECKPOINT after the 04:50-04:58 UTC final closure reviews
  (response document now 94ec97a96d4496548f5870d7cbcfbf6f43872c8e9cbcbd147fec8137aada23d1; the 04:40 section and its 826-test history are preserved).
  Delta 1: INNER_FACE_ELIGIBLE_STATUSES = () in kernel V2 and contact V3 (two-layer selection is a retained
  side hypothesis, never eligible; generic contact geometry and hypothesis metadata kept; scorer untouched).
  Delta 2: strict `is` boolean at the context association and cover instance eligibility sites with 0/1
  tamper checks. Delta 3: completion-only and mixed-origin groups through the FULL V16 _registration_binding
  and inspect_cad_guided_mesh_v16 dispatch with a 13-case fault matrix; V16 now reconciles every published
  completion lane against the selected group's origins (mixed groups cannot drop a passed child).
  Delta 4: waiter bytes in the frozen SOURCES; exclusive O_EXCL outer-record reservation, one-time token,
  advisory lock, atomic exit record, unobserved-on-failure; direct --waiter refuses; 9 fake-child tests.
  Nonblocking notes carried: zero-child cap case really schedules zero, malformed-evidence envelope test,
  precise jittered-column assertions, cap normalization and pre-diagnostic-refusal wording.
  Actual results at the delta hashes (inert namespace): 855 passed / 1 skipped / 0 failed in 31.32 s across
  thirteen suites (28/184/82/22/6/117+1s/9/123/255/6/9/5/9). Delta source hashes:
  12bb9ce9e595ada6b4cf86947089ef96f2158ddf36009584442e0a414e6ae008  backend/cad_guided_measurement_v2.py
  9b48c72b4e3eb87899f8e6911929a869c63d0182dd95e254e5deb5671e623f7c  backend/cad_guided_context_measurement_v2.py
  6c6bd75813f6e985fd4e841eb401ae330a5be614a6e874b2e5704256e00ca901  backend/cad_guided_support_cover_measurement_v3.py
  469c444f49cdcd631f04635516304b6af530fe3c813dc1f07cadb51389ed16a7  backend/cad_guided_mesh_offline_v16.py
  4f0cbea8cc97056295ab49092465f27230d0707ff53eaa8915f78feea031927b  scripts/gold11_contact_scoring_v3.py
  4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1  scripts/run_gold11_mesh_development_v9.py
  85913855908ac3a965cb37d7fdd36a8c3464aabd05621184ecd9b4b6e8107688  scripts/gold11_v9_execute_waiter.py
  e6fcd96aeb85bf3d91543f63f2001511ca22ff13a46c2bd1f1a51ca810fef54f  docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md
  d00991ce3ec38d89c668830eeb28cb505086cb64ab0c62c11c0addfb8c48efce  docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md
  da7d2c5c4d253e0877e5049de408fcc6c8bb02f2456da7480424e5151599ecba  docs/research/2026-09-13_fable51_v9_diagnosis.md
  (V15, envelope V7, projection V4 and the registration test unchanged; test-file hashes in the response
  document). No freeze yet; no physical run, raw/Gold/attempt/cache access. Next on Codex's delta
  acceptance: exclusive --freeze, independent PRE/archive, waiter-launched single --execute, POST.
- 2026-09-13 05:12 UTC (date -u) — Cover V3 fault-matrix wiring omission (Codex delta review): the
  `instance_eligibility_int` and `group_eligibility_int` branches were not in the parametrize list, so
  they had never run. Names added; only those two cases executed: 2 passed, 82 deselected in 0.25 s.
  tests/test_cad_guided_support_cover_measurement_v3.py now
  f53fe50dcbe70e8a6119c583b3d574c43caae76a57e5f7812f1a1163e2c41732; no source change; no broad rerun.
  Response document now 5298f201a70edd1d59a44b707e85341726561f7c5888a68f13831321c655ee3b. Codex independently ran the 9 fake-child waiter
  tests (passed) and accepted the identity/boolean source fixes and the launcher delta; the completion
  delta review is finishing; freeze release follows its acceptance. No engine change requested or made.
- 2026-09-13 05:16 UTC (date -u) — V9 SOURCE CLOSURE ACCEPTED by Codex (coordination doc "Source closure ACCEPTED —
  05:08–05:14 UTC": material delta acceptance 2fe8d44b538ec2c6492c44f60adb8132dfc3de8a08be6ef04005fb3faf1072a9,
  completion delta acceptance 4c9c50e5bd0acae0d01dad6f4ee141a63399a9ed3cdf1952b651af384c81a73a, root
  runner/waiter report with delta acceptance d47f66af5fb2b724bc4140ed6dcf5c36c824bf5f4816e429b7a436862b2d5ed8;
  the two added cover cases accepted at test hash f53fe50d…1732). Exclusive source-only freeze executed once
  at 05:15:51 UTC after a fresh ps no-overlap check (no Sherman process):
  OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v9.py --freeze
  exit 0, printed bundle_sha256 c50a8cb9684ba891a97005755529d952229e25b6db83e4549a17cb830ca4b47d, attempt
  docs/research/gold11_mesh_development_v9_attempt (FROZEN.json 51207 B, file SHA256
  67a253ad14702d15853942d70f94260d06e91713b76e4d84f26242ceed37a26e; 85 frozen sources under source/).
  External archive docs/research/gold11_mesh_development_v9_physical_frozen.RdikX7AH/SOURCE.tar (directory
  mode 0700; SHA256 412c9f8e662c845e172cdd19a134031c62a5c4a12866041fd82981da2c6c2326, 3451904 B, tar lists
  92 entries = FROZEN.json + 85 sources + directories) created for POST custody; no raw/Gold/cache access.
  Next: independent non-author exact physical PRE against this bundle and archive (auditor writes to job
  tmp; artifacts copied verbatim with hash verification), then, only if GREEN with no blockers and after a
  fresh no-overlap admission, the ONE waiter-launched --execute (actual outer exit recorded), then POST.
- 2026-09-13 05:35 UTC (date -u) — Independent physical PRE complete (non-author auditor `independent-ai-pre-auditor-v9`,
  read-only; wrote to job tmp, copied verbatim with hash verification): decision
  GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN, blocking_findings [], bundle_sha256 recomputed independently
  = c50a8cb9684ba891a97005755529d952229e25b6db83e4549a17cb830ca4b47d. Checks: 85/85 source hashes agree three
  ways; SOURCES tuple = map keys incl. the waiter; V8 basis hash unchanged; 12/12 accepted closure hashes exact
  (plus frozen contact V2 and planner V1); 17/17 runner constants equal the bundle incl. byte-identical
  arm_requests and runtime metadata (python 3.11.15 darwin/arm64); declared deltas δ .03 / e .04 / gap 30 /
  layer+completion policies+budget declaration, 1.0 mm gate unchanged; 11 cases, 11 scan / 6 CAD digests,
  raw17 digest 2459f4e9…d157 and case-list digest 943268be…813e IDENTICAL to the V8 PRE; attempt tree
  pristine; archive tar = FROZEN.json + 85 sources, sizes equal; protocol limits = LIMITS; waiter contract
  confirmed (O_EXCL reservation, refusals, atomic exit record, unobserved-on-failure, no crash-proof claim);
  Codex acceptance documents hash as expected; ps overlap none at 05:20:44 UTC; nine non-blocking notes
  (NB-1: verify the pid record reads RUNNING after launch). Files:
  docs/research/gold11_mesh_development_v9_attempt/PRE_RUN_AUDIT.json (27755 B,
  750c85f519d15598c504597555a33b30eb20aba30eef4a61dfd1c472e04a037b) and
  docs/research/2026-09-13_gold11_mesh_development_v9_physical_pre_run_audit.md (29261 B,
  54189cdd556fd310bbe7f8ecdef7bfbcbc8ace61667c0e136a79242f3909f31b). Next: fresh no-overlap admission and
  the ONE waiter-launched --execute.
- 2026-09-13 05:36 UTC (date -u) — THE ONE V9 --execute LAUNCHED through the bound waiter after a fresh ps
  no-overlap admission (no Sherman process) and the pristine-tree check (FROZEN.json, PRE_RUN_AUDIT.json,
  source): `.venv/bin/python scripts/gold11_v9_execute_waiter.py` exit 0, printed LAUNCHED 05:36:10 UTC,
  waiter pid 52251 (detached session leader), runner pid 52252 as its direct child running
  `OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v9.py --execute`;
  outer records reserved exclusively beside the attempt directory (…_outer_pid.txt reads
  "RUNNING 25e6c406b3f52ced8bddd9173b9d7345 waiter_pid=52251 runner_pid=52252 2026-09-13 05:36:07 UTC",
  …_outer_exit.json pending, …_outer_stdout.txt owned by the runner). ps at 05:36:16 UTC: coordinator 52253 and
  worker 52254 (1.2 GB RSS) alive under the runner. Monitoring is ps-only (job scratch monitor_v9_wait.sh:
  tree RSS every 10 min); no attempt, cache, raw or Gold artifact is read during the run. The parent's
  actual return code will be recorded by the waiting parent in …_outer_exit.json when the runner exits.
- 2026-09-13 13:35 UTC (date -u) — V9 --execute COMPLETED; the waiting parent OBSERVED the actual outer
  return code: docs/research/gold11_mesh_development_v9_attempt_outer_exit.json = returncode 0,
  observation OBSERVED_BY_WAITING_PARENT (waiter pid 52251, runner pid 52252, started 05:36:07 UTC, finished
  13:34:04 UTC, nonce 25e6c406…; pid record FINISHED … returncode=0). Runner stdout (466 B) final line:
  status COMPUTED_PENDING_INDEPENDENT_POST_AUDIT, release_allowed false, durable_terminal_case_records 22,
  final_campaign_inventory_sha256 4c63a73d8c21ed72349b8db5bc2c7ba6a4d20a8dd91cef031dd127a8dd64b594 (284
  entries, 105 scientific manifest entries). ps-only monitor: tree RSS between 0.70 and 1.97 GiB over the
  7 h 58 min run (limits 12 GiB campaign / 16 GiB repository untouched); no attempt, cache, raw or Gold
  artifact was read during the run. Attempt tree now holds STARTED/COORDINATOR_*/INFERENCE_TERMINAL/
  SCORED/SCORER_*/SELECTED_REFERENCES/COMPLETED/FINAL_GATE plus cases/ and leaf_dispatch/. Next: read the
  derived scoring records, commission the independent POST, then the result narrative (no score claimed
  before POST; null stays null).
- 2026-09-13 13:45 UTC (date -u) — V9 DERIVED RECORDS READ (attempt's own SCORED/INFERENCE_TERMINAL/cases
  records only; no cache/raw/Gold): both arms TP 0 / FP 0 / FN 71, recall 0, precision and F1 NULL (undefined
  by the unchanged scorer); emitted support lines 0; all 22 slots in the denominator. Slot taxonomy: 14 slots
  (CG08506, CG191, CG1E0, CG4FAF, CG64, CGEBA, CGF41 × both arms) terminated by the per-case CPU cap (exit code
  -24 = SIGXCPU at 1860 s CPU, wall 1861-1862 s) inside surface V4 refinement (`_paired_candidates`/leaf walk
  frames in the sampled stacks), returning no workflow stages (V8 had 4 such slots: CG191/CG4FAF); 6 slots
  refused before registration exactly as in V8 (CG65 no finite start, no two-of-three parent; CG84 no finite
  start with 1 completion parent, 44 children scheduled, 0 passing, 68 normal-incompatible; CGE4B scan-plane
  proposals unavailable); CG4BDD challenger: 3 start groups (1 main + 2 completion), all rejected at the 1.0 mm
  context gate (1.196 / 1.318 / 1.338 mm); CG4BDD control: 4 start groups (1 main + 3 completion), the main
  group admitted at 0.666 mm (as in V8) and the completions rejected (0.738 / 0.720 insufficient proximity,
  1.306 outside gate), then the MEASUREMENT phase refused with
  SUPPORT_COVER_MEASUREMENT_FAILURE:ValueError:INVALID_SUPPORT_COVER_FEATURES. SOURCE-CONFIRMED V9 WIRING
  DEFECT: mesh offline V16 (line 33) imports `CADGuidedFeature` from the V1 kernel `cad_guided_measurement` and
  builds the native features with that class (line 692), while support cover V3 (line 144) requires
  `type(feature) is` the kernel-V2 `CADGuidedFeature` (context V2's class). Every real measurement dispatch in
  V9 would therefore refuse; the connected-chain test built kernel-V2 features directly and the engine
  integration test mocks native features (`object()`) and the cover call, so no test exercised V16's actual
  feature construction into cover V3. This is the author's defect, discovered from the terminal records; V9
  stays terminal and is not re-scored. Independent POST commissioned (running, read-only). Next: POST, result
  narrative with honest numbers, then a V10 diagnosis (feature-class binding fixed with a test through V16's
  real feature construction; CPU-cap terminations under the enlarged completion refinement load).
- 2026-09-13 13:41 UTC (date -u) — V9 result narrative written: docs/research/2026-09-13_gold11_mesh_development_v9_result.md
  (12039 B, SHA256 67e9eecf9a4d6f65a405f242836c05123b90b148b681bd3c51b9bbaa19038a2a; POST acceptance section
  to be appended verbatim when the independent POST report arrives). Headline: observed exit 0, parent wall
  28675.6 s, 22 records, both arms TP 0 / FP 0 / FN 71, precision and F1 null; taxonomy 14 CPU-cap
  terminations in surface V4 refinement (V8: 4), 4 no-finite-start, 2 proposals-unavailable, 1 all-groups
  rejected at the 1.0 mm gate, 1 admitted context refused at the cover feature grammar (source-confirmed V16
  kernel-V1 feature class vs cover V3 kernel-V2 class). Kernel-V2 selection and the derived δ/e and 30 deg
  policies were never exercised on cohort data. Codex assigned a separate scoring-ledger reviewer
  (2026-09-13_gold11_v9_scoring_ledger_codex_post_audit.md); no attempt/source mutation or rescore. Next: POST
  acceptance, then a V10 diagnosis (feature-class binding with a real V16-to-cover test; CPU-cap load).
- 2026-09-13 13:42 UTC (date -u) — Codex ledger review confirms CG4BDD/control INVALID_SUPPORT_COVER_FEATURES as a contract
  failure; cost reviewer source-proves V15 runs the completion prefilter before main refinement and all
  refinements before publication (main-first quota protects neither CPU nor already-passing contexts); lane
  attribution of the 14 kills is UNPROVEN. Correction appended to the V9 result narrative (append-only); the
  attribution sentences are now marked hypotheses. Cost report
  2026-09-13_gold11_v9_completion_cost_codex_postmortem.md and scoring-ledger report
  2026-09-13_gold11_v9_scoring_ledger_codex_post_audit.md to be used in the V10 diagnosis after POST. No V9
  edit or rescore. Independent POST still running (read-only).
- 2026-09-13 13:42 UTC (date -u) — Ledger timestamp correction: the entry labelled "2026-09-13 13:45 UTC" (V9 derived records read) carried a hand-typed time; it was written at about 13:41 UTC, before the 13:42 narrative entry. The result narrative's correction section had lost two report names to shell substitution; restored verbatim (no other change), narrative now b9300a02fa2bacdc38a9cbb09eff67292e87d265105e05e959b0ee65817954a1.
- 2026-09-13 13:59 UTC (date -u) — Corrections (Codex): the kernel-V1 CADGuidedFeature import/construction (lines 33/692)
  are in backend/cad_guided_native_features_v1.py, not mesh offline V16 (V16 line 621 passes those V1 objects to
  cover V3 line 144, which requires the kernel-V2 class); CG84 enumerated 44 completion children with ZERO passing
  starts (not progress). Appended to the result narrative (append-only) and fixed in CURRENT_STATUS. Both Codex
  reports read in full and hash-verified: scoring-ledger post audit 87e7f905e3a33c18b13e6b2261f7c545b9ff4e18102e2cf39a21c5a707a5b9ac
  (ACCEPTED_NEGATIVE_SCORING_LEDGER_WITH_LIMITED_SCOPE; nonblocking: legacy `comparison_scope` label NEW_V13…),
  completion-cost postmortem 5edc6ee890310eb4b22b2d367077affe423a608cae96266fe675adb07fb5618f (completion
  prefilter precedes main refinement; all refinements precede publication; per-group source bound 415392 rows /
  63 calls; identical final full-scan query duplicated between _diagnose and _context_support; runner's
  completion accounting prose false: _FaceQueries charges node visits/triangle pairs/normal pairs). V10 diagnosis
  starts now on these inputs (full POST still required; V9 frozen source and attempt preserved).
- 2026-09-13 14:02 UTC (date -u) — V10 diagnosis written: docs/research/2026-09-13_fable51_v10_diagnosis.md (facts A1-A9 from the V9 records and both Codex reports; mechanisms B1 feature-class contract via native features V2 + real extractor-to-cover contract test, B2 cost receipts and bounded STAGE_CHECKPOINT stdout lines, B3 exact reuse of the duplicated final surface query, B4 declared CPU allowances with lane transactions and an EXPLICIT finite-subset publication contract, B5 corrected runner declarations; B6 optimizer deferred). Implementation starts with B1/B5; V9 frozen source and attempt untouched; full POST still pending.
- 2026-09-13 14:03 UTC (date -u) — Independent physical POST complete (non-author auditor, read-only, wrote to job tmp;
  copied verbatim with hash verification): docs/research/2026-09-13_gold11_mesh_development_v9_post_run_audit.md
  (64734 B, bcdd7e4b414d308ca347f40ddda5c78fb53d24f623ae6b6083bb25c1f0b33dca). Decision ACCEPT (execution
  custody + complete negative scoring ledger; engineering audit only), blocking findings none; outer exit
  OBSERVED (returncode 0), all FINAL_GATE/inventory/custody digests reproduced, 85/85 sources identical three
  ways, no retry/resume/relaxed limit; findings: 14 CPU kills in the V15 refine/diagnose loop (stages unknown),
  new outcome class MG = CG4BDD control refused by cover V3's exact-type guard (kernel-V1 objects from native
  adapter V1 through V16), measurement-side V9 changes never exercised on physical data, identity counts null
  not zero, inference wall 3.08x V8's for the same zero-support ledger, 84/85 frozen sources untracked in git.
  POST acceptance appended to the V9 result narrative. V9 closed as terminal (F1 null). Codex native
  feature-contract follow-up read: repair = NEW successor adapter performing explicit lossless kernel-V2
  reconstruction after the unchanged r0/v1 factories (both return kernel-V1 objects), strict V2 checks kept,
  regression through both real factory branches -> actual successor dispatch/bridge -> real context/kernel/
  planner/cover/validator with analytic scan rows. V10 implementation starts with that repair.
- 2026-09-13 14:10 UTC (date -u) — POST reviewer addendum obtained (Codex's three corrections: archive member-hash claim
  withdrawn, 10/11 case pairs = 20/22 slots share cause with CG4BDD the exception, worker CPU stated as unpaired
  totals and as the paired 8-slot comparison 763.757 s -> 2535.780 s all in CG4BDD): copied verbatim with hash
  verification to docs/research/2026-09-13_gold11_mesh_development_v9_post_run_audit_addendum.md (9420 B,
  67237db70c3d2232c62c1ce0ca72e88c21011ce424a5d18ef25a48cf27362a2e); original POST unchanged; ACCEPT stands.
  Pointer appended to the V9 result narrative. V10 B1 in progress: backend/cad_guided_native_feature_bridge_v1.py
  (lossless kernel-V1 -> kernel-V2 reconstruction, fail-closed) and backend/cad_guided_mesh_offline_v17.py (V16
  text + bridge dispatch before CAD sampling/registration for both r0/v1 branches) written; factory-to-measurement
  regression being made to pass.
- 2026-09-13 14:14 UTC (date -u) — V10 B1 closed at source with the required regression (8 passed, inert namespace):
  backend/cad_guided_native_feature_bridge_v1.py (explicit lossless kernel-V1 -> kernel-V2 reconstruction from
  every dataclass field in declared order; refuses wrong class, duplicate ids, over-limit sets without
  truncation, malformed factory results; no authority, no coordinate conversion; frozen factories untouched),
  backend/cad_guided_mesh_offline_v17.py (V16 text + bridge dispatch immediately after the unchanged r0/v1
  factory returns, before CAD sampling and registration; refusal phase NATIVE_FEATURE_BRIDGE with native
  diagnostics retained), tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py: both REAL
  factory branches on an independently constructed OCP bend (r 5, skin 2.5 mm, 30 mm) -> real bridge -> real
  context V2 / kernel V2 / planner V1 / cover V3 / validator / contact V3 projection on analytic rows in the
  feature's own frame (identity registration fixture and keep-all role substitute declared): one supported
  generic carrier (radius 5.0 within .05, TWO_LAYER side hypothesis, eligibility false, no deviation, count
  unknown, conformance INDETERMINATE, endpoints transported, scored TP 1 as transport); frozen V1 objects handed
  directly to cover V3 reproduce the V9 INVALID_SUPPORT_COVER_FEATURES refusal; bridge refusal matrix; actual
  V17 dispatch bridges the real factory output before CAD sampling/registration and hands kernel-V2 objects to
  the cover (both branches); a bridge refusal returns before any costly work. Skin 1 mm would leave outer-arc
  rows in the free-circle fit (kernel refuses ARC_NOT_SINGLE_CIRCLE; the nominal isolation bound 68 deg < 90),
  so the fixture declares 2.5 mm where the bound is 97.5 deg; that is a documented kernel limitation, not a fix.
- 2026-09-13 14:22 UTC (date -u) — Codex B1 audit read (docs/research/2026-09-13_gold11_v10_native_bridge_codex_audit.md,
  c7fe924eda4e0ea18d6dbcb58fe79c39680bb829a44e5afc9a6f6283138a78f3) and V10 coordination read
  (docs/research/2026-09-13_fable51_v10_codex_coordination.md, 7a88214ff5b523d457b5825c3385602978f1fd2179a7a624af7cfb6f09c93e4b).
  B1 P1 closed: bridge limit rebound from the kernel entrypoint cap 16 to the real cover V3 grammar boundary 64
  (context V2 admits 512 records; the cover's own 64 is asserted against the real cover in the test); tests
  added: 17 and 64 accepted with order/field preservation and real-cover acceptance, 65 whole-refused by the
  bridge and by the cover grammar; named 1 mm-skin kernel refusal retained as an executable regression
  (ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL, failure phase ARC_FIT_AND_TANGENCY; nominal bound 68.5 deg <
  90 < 97.2 deg for the 2.5 mm fixture); docstring relabels the extracted-frame rows as factory-conditioned
  transport; NEW independently constructed geometry check (rows from the construction constants, not the
  feature: radius 5, sweep 90, axis +-z, generator on the construction axis within .05 mm, both branches);
  NEW real bridge failure through V17's own dispatch on tampered real factory output (no injected exception).
  B4 design corrections (coordination doc, six points + acceptance matrix) accepted for implementation:
  strict candidate_evaluation_complete, evaluated_subset_complete with disjoint completed/expired/omitted
  reconciliation in a NEW schema validated by every consumer, deterministic per-lane work/group allocation
  with CPU as a declared safety guard and bound realized trace, absolute process CPU incl. preparation, one
  completion allocation, cooperative intra-query stop points (chunked full-scan queries) with declared largest
  uninterruptible operation and serialization headroom, typed allowance expiry only (everything else fatal),
  omitted alternatives unknown, bounded checkpoints (count/bytes, emitted vs durable), exact immutable reuse.
- 2026-09-13 14:36 UTC (date -u) — B1 DELTA RECEIPT (Codex accepted cap 64 / independent constructor-frame
  check / retained 1 mm negative at these exact hashes): backend/cad_guided_native_feature_bridge_v1.py
  fef3980a5d5d4594b48dacaee763451c798d081044b4cf47630f5d7cd6126fea, tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py
  6b3442507936f45a361f711b949801540a002513bd815587d524b171022a7ec9, backend/cad_guided_mesh_offline_v17.py
  3cb73f0860e00e1778c6883bff873d37e48e7f886acf3ce94c55c3cca32a1f35. NEW focused run at 14:35:44 UTC (inert
  namespace): `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=.
  .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider
  tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py -q` -> 13 passed in 2.60 s:
  real_factory_bridge_and_real_measurement_chain[v1,r0], frozen_v1_objects_reproduce_the_v9_refusal[v1,r0],
  independently_constructed_geometry_confirms_the_extractor_frame[v1,r0],
  actual_bridge_failure_through_the_dispatch_boundary_with_real_factory_output,
  one_mm_skin_remains_a_named_kernel_refusal, bridge_limit_is_the_cover_and_context_boundary_of_64,
  bridge_refuses_wrong_class_duplicate_ids_over_limit_and_malformed_results, actual_v17_dispatch_bridges[v1,r0],
  v17_bridge_refusal_returns_before_any_costly_work. The earlier 8-pass receipt is superseded for this delta.
- 2026-09-13 15:38 UTC (date -u) — V10 INTERIM CHECKPOINT (Codex checkpoint request; not a freeze, not a gate). Actual test
  outcomes, including failures: (a) 15:18:03-15:20:53 UTC, 11 suites (bridge, registration V5, initializer V16
  registration/completion/allowance, context V3, cover V4, contact V4, projection V5, connected chain, V17
  integration): 862 passed, 1 skipped, 1 FAILED — test_independent_fixture_and_frozen_numeric_aliases (V17
  integration) because the between-face-build stop had been inserted as a hook line into the V15-verbatim base
  _FaceQueries builder text; fixed by restoring the base text verbatim and moving the typed between-build stop
  into _FaceQueriesV16 (one unchanged V15 build per face after the verbatim whole-set preflight), re-verified with
  the alias test + phase-stop/chunk/accounting/rollback tests (6 passed, 15:24-15:26 UTC). (b) 15:33:15-15:36:07
  UTC, 13 suites (the 11 above + runner V10 source test + waiter V10 fake-child tests): 879 passed, 1 skipped,
  0 failed (`PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=.
  .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider <13 files> -q`). Hashes at that run:
  backend/cad_guided_plane_initialization_v16.py 7dad12ae569c2044681864bb136105e0fea8750a4c1e454797a9f9fd874cb63b
  (B4 delta: rollback restores pre-transaction main groups from a deep snapshot; face queries charge attempted
  rows before the primitive and complete after a COMPLETE answer; cooperative chunk cap; typed phase-boundary
  stops before/during enumeration, before binding, before the face-index build with the first admissible query's
  rows (zero allocation never builds an index), between per-face builds and before evaluation; FAILED disposition
  for surface/unclassified fatal failures in flight; deep-owned reconciled checkpoint snapshots),
  tests/test_cad_guided_plane_initialization_v16_allowance.py b50e781a422a4a5a334151ee5a9eecc609db22df5b6cfcea2247e6aebc1c1168
  (explicit two-candidate fixture with a checkpoint-controlled clock: first candidate supported, second
  OMITTED_BY_CPU_GUARD, exact ledger, subset contract, the ACTUAL initializer subset bound through the REAL V17
  _registration_binding and the omitted candidate / a bare claim refused there; three-candidate surface and
  unclassified fatal variants with completed [0] / failed [1] / not_attempted [2] and zero usable output;
  duplicate-main child then later typed expiry on the modified completion scene through
  register_measurement_contexts_v16: main-group provenance byte-equivalent to a lane-disabled run, attempted child
  evidence retained, the retained main context accepted by the real V17 binder and the dangling pre-fix provenance
  refused; face-query accounting (success / primitive failure after admission / typed pre-admission stop); phase
  stops (guard exceeded before the lane, zero allocation before any build, expiry between two builds, chunk cap
  1024/32768/65536); checkpoint snapshots owned and reconciled at both depths),
  backend/cad_guided_context_measurement_v3.py 2f1439fb01e0959e58f3275975b0150afc3d59ed72b3ea22ba0f510fb2a13c56
  (denominator/pose/window loops over completed candidates only; UNKNOWN_CANDIDATE_CARRIES_CONTEXT_CLAIM),
  tests/test_cad_guided_context_measurement_v3.py dc34ab84ab0c37aa4b718fba4f49925840daf26d6297c66f717eff3250f596b8
  (20 new subset acceptance/fault tests), backend/cad_guided_support_cover_measurement_v4.py f6972fd538d27b24d2142a453b56b3846be34e2dc5d79ab30f54d198ce960230,
  backend/cad_guided_mesh_offline_v17.py 5c4fa6b8f3fb97a23398b9fb00cb418f2f47f0477b93693c6a3c0c4cfd8a5a63,
  scripts/gold11_contact_scoring_v4.py c0bdfadadec64447acb3db6e6509264c53980e8f9b02a130eb22311b8aceba10
  (subset_registration_count int diagnostic replaces the string that broke the count grammar),
  scripts/gold11_schedule_projection_v5.py 996fc87cb85f659d9d604bddb359497abd7bade5306f178703dbac0052499179,
  backend/cad_guided_native_feature_bridge_v1.py fef3980a5d5d4594b48dacaee763451c798d081044b4cf47630f5d7cd6126fea (unchanged),
  tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py e251ed01325b028d348610e3220e2b66fae8300721283f2f9b9c3a45fa94ffa1
  (rebound to cover V4 / context V3 / contact V4 / initializer V16 dispatch), tests/test_cad_guided_v10_connected_chain_v1.py
  99a56ab47e82f6b863a73b517f104173546c62c74fec2c79d422f8a6c9b9a62e, tests/test_cad_guided_completion_v1_engine_integration_v17.py
  18a15d24bff78a1c010ed96b69574671e79b5f4f4ffdc0af0969a90d8793fbf6, tests/test_gold11_contact_scoring_v4.py
  a40fbbfc377d9627ecd8a397a52476496f1f6e3b38ffa9c18b74ab3784cbee68, tests/test_gold11_schedule_projection_v5.py
  d4985a5b1b161aec4e661e459a00d9e2a5833dd6e4629ccf5a35dd0615cd7416, tests/test_cad_guided_support_cover_measurement_v4.py
  062ce6e276ee87b6c2ec0e69282461892643eceb5dce04af30e5031f5b2ced6e, tests/test_cad_guided_plane_initialization_v16_registration.py
  a3c950a6188cf799fb1945525590b1358cdf148cf27edf895cd2a18eb3a00126, tests/test_cad_guided_plane_initialization_v16_completion.py
  699194ebf6b20d6433f6eb4bd1096f7044a72db8d3f60b76bced774dc9b3f3a8. NEW V10 runner/waiter/protocol/design at that run:
  scripts/run_gold11_mesh_development_v10.py ef73dfacf475052ad9b59727a49f80c7a0d6b2210c2651a63c9c7182ddc80217 (V9 text
  ported: basis V9 4567a0f5…, V17/V16/V5 identities, CONFIG.refinement_allowance bound to LIMITS.case_cpu_seconds,
  CONFIG_DELTA_FROM_V9, corrected completion_budget_declaration (values unchanged), scorer V4 / projection V5,
  bounded STAGE_CHECKPOINT stdout printer, gold11-v10- caches, V10 labels), scripts/gold11_v10_execute_waiter.py
  c2648778195b34ee39b45e217c8b566d0998e4d721634866501305ad6b4952fc, tests/test_gold11_mesh_development_v10_source.py
  7116db3500a9541b133c04dd9e3f57ae7e188060bd466df2e3b3160d87e6e37e (7 passed), tests/test_gold11_v10_execute_waiter.py
  9a1542b378118dfec12b245fb5b87ea9b4c758be942e87ddb170e9206e855d37 (9 passed), docs/research/2026-09-13_gold11_mesh_development_v10_delta_design.md
  6476ed7fde1a6f86d9e22fd2e530d1728965b8cf40378cda6ef284bb491241f7, docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md
  1d6ab2c9dab27b23545dfa4f8e0ac6e9a2c203e97c485763040b24ce894364e0.
  Changes AFTER that run (post-receipt, each re-verified focused): B3 ordering bug (Codex): _thaw_query now checks
  INTRINSIC integrity (schema, immutable bytes-backed arrays, digest with shapes, COMPLETE status, grammar against
  the retained rows) before any key comparison and validates against the current index only after every key
  matched; a genuinely different valid index (stepped plate) on the same query object requeries, an identical
  rebuild also requeries: backend/cad_guided_global_registration_v5.py 00c1eab81fe016faa44f0a569e40e06b6b37f4275badadb2dc16656e4bcc6f19,
  tests/test_cad_guided_global_registration_v5.py 14dcb8bc29765bc9687e12930dd120a288d475fc1b88d0c0b1daeeb876095bb0
  (14 passed). Runner provenance wording (Codex): the native-bridge reason now states that only CG4BDD control
  reached measurement and the other 21 slots failed earlier: scripts/run_gold11_mesh_development_v10.py
  163c46658bf2ebe3aa258866d3e7ef431bc9a22974d499b79de10cbe7753db71 (regenerated from the port script; source test
  rerun pending in the final receipt). Under active edit next: the completion chunk test's name/claim (its supports
  are 25 rows; a support longer than the cooperative chunk is being exercised through the real
  _evaluate_completion_supports), a runner-policies-to-V16-registration seam test, the V10 review-response
  checkpoint document, CURRENT_STATUS; then the final 13-suite receipt with final hashes for the delta review,
  exclusive freeze, non-author PRE, single waiter execute, POST. No V9 artifact, frozen source, control or cache
  touched; no physical data read.
- 2026-09-13 15:57 UTC (date -u) — V10 SOURCE CLOSURE CHECKPOINT PUBLISHED for Codex's bounded closure review:
  docs/research/2026-09-13_fable51_v10_review_response.md (hash printed in this session's output and re-hashed by the
  reviewer). FINAL receipt: 13 suites, 885 passed, 1 skipped, 0 failed, pytest exit 0, 15:52:07-15:55:11 UTC
  (183.62 s), inert namespace. Final hashes: bridge fef3980a…, V17 5c4fa6b8…, V5 ee05a03e…, V16 7dad12ae…,
  context V3 e67d1c35…, cover V4 f6972fd5…, contact V4 c0bdfada…, projection V5 996fc87c…, runner V10 f7fd083f…,
  waiter V10 e7ade783…, delta design 0a68d42a…, protocol e59aff47…; tests: bridge e251ed01…, V5 14dcb8bc…,
  V16 registration a3c950a6…, V16 completion 699194eb…, V16 allowance cc34fc7f…, context V3 d5a81f27…, cover V4
  062ce6e2…, contact V4 a40fbbfc…, projection V5 d4985a5b…, chain 99a56ab4…, V17 integration 5f2862f3…, runner
  source 7116db35…, waiter 0df6e722… (full 64-hex values in the checkpoint document). Closed since the 15:3x
  interim receipt: B3 ordering (intrinsic integrity before key match; index-bound validation only after a full
  key match; genuinely different index and identical rebuild both requery; V5 00c1eab8→ee05a03e with the
  row-quota-scope docstring), forged strict whole flag refused by the context V3 helper and by the real V17
  binder while legacy no-ledger evidence keeps the V2 meaning (context V3 e67d1c35…; new fault
  strict_claim_contradicted_by_ledger; strict_claim_with_omitted_row now legacy-shaped so the full consumer
  denominator refusal stays exercised), runner→V16 seam test through the real V17 workflow (runner constructors,
  allowance object and bounded checkpoint printer reach register_measurement_contexts_v16 unchanged; scene-bound
  .025 mm bound / 36-row CAD double declared), 1500-row completion support screened and queried as [1024, 476]
  through the real evaluator and real metered V16 queries, guard expiry after the first successful screen chunk
  and after the first successful query chunk (no next primitive, attempted == completed exactly, child never
  PASSED, rollback marks it EXPIRED_BY_ALLOWANCE_UNKNOWN), V5 atomic-work disclosure (preparation, one chunk
  query, one face-index build) and row-quota scope (query rows only), runner provenance wording (only CG4BDD
  control reached measurement; whole-cover refusal before any window), waiter overlap regex \d+ (+ regression
  test), D1/D2 wording in design/protocol. Next: Codex bounded closure of these hashes → exclusive --freeze →
  independent non-author PRE + archive → fresh no-overlap → ONE waiter execute → ps-only monitoring → observed
  outer exit → independent POST → result narrative. No physical data, attempt, control or cache touched.
- 2026-09-13 15:57 UTC (date -u) — Checkpoint document hash for the record: docs/research/2026-09-13_fable51_v10_review_response.md
  ff6c94daa686b72335562888f0594a9218d3d369c103e94b41e4cd6c71b49e5b; docs/CURRENT_STATUS.md top section added
  (880088581884a52007a199ce4e69b01a803d4544e8c43b73525b04f265351da3). Precheck for the freeze (no side effects):
  no docs/research/gold11_mesh_development_v10_attempt* directory exists; the overlap process table is empty
  (no run_gold11_/pytest/cad_guided/_gold11_vN_exact_scoring processes at 15:57:32 UTC). Waiting for Codex's
  bounded closure of the published hashes before the exclusive --freeze (V9 precedent: freeze follows closure
  acceptance); no idle edits to reviewed files while the review is open.
- 2026-09-13 16:04 UTC (date -u) — V10 BOUNDED SOURCE CLOSURE ACCEPTED by Codex at 16:01 UTC (coordination doc
  3e148087e0e4972921b6d9fa8db7b68413599df9d617be950750515272405cf9: all 25 hashes of the ff6c94da… closure match
  before/after; B2/B4 final audit 7315fc11851cdfbb1a47967533cccde528cd44f72770a0b85edbaaefde0091a9, runner delta
  c3e5fdcd6a0684d09b09a496fdd08eacb1757f34918dcf092f6a15e1295d4684, B3 17920e54…, B1 unchanged fef3980a…).
  Exclusive freeze executed ONCE at 16:03:04 UTC after a fresh ps no-overlap check (no run_gold11_/pytest/
  cad_guided/_gold11_vN_exact_scoring process) and an absent-attempt-directory check:
  OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v10.py --freeze
  exit 0, printed bundle_sha256 ded97290231a6e8fab9ba1680790fd3feead45dde9c9912048bde41462546928, attempt
  docs/research/gold11_mesh_development_v10_attempt (FROZEN.json 56925 B mode 0600, file SHA256
  dd93ccdd9ed0a90c85247b0b76859836f4e05b776e66c492713076592574b980; 96 frozen sources under source/).
  External archive docs/research/gold11_mesh_development_v10_physical_frozen.q4bxmlh3/SOURCE.tar (directory mode
  0700; tar created with COPYFILE_DISABLE=1 --no-xattrs --no-mac-metadata after a first attempt that captured
  macOS AppleDouble ._ entries was deleted and redone; SHA256 2cb7d6fd17012b23e539d754d3dfdea779fc334ad8e2524a74a5ef43bb36e255,
  3858944 B; 97 file members = FROZEN.json + 96 sources, 6 directory entries, 0 unexpected). MEMBER PAYLOAD
  hashes (Codex requirement: not container/listing only): SOURCE.tar.members.sha256.json (37841 B, SHA256
  e62eda42f1313784792fd1b2ce55408782890d97ba4b8f65dab27458dbe4290c) records every member's payload sha256 =
  on-disk snapshot sha256 = FROZEN.json source_sha256 (0 mismatches, 0 missing, 0 extra, FROZEN.json member
  present, canonical bundle sha recomputed = ded97290…). No raw data, Gold, cache or V9 artifact touched.
  Next: independent non-author exact physical PRE (reviewer role, read-only, writes to job tmp; artifacts copied
  verbatim with hash verification into the attempt directory and docs/research), then only if GREEN with no
  blockers and after a fresh no-overlap admission, the ONE waiter-launched --execute.
- 2026-09-13 16:26 UTC (date -u) — Independent physical PRE complete (non-author auditor `independent-ai-pre-auditor-v10`, fresh
  reviewer context, read-only; wrote only to job tmp; two scratch listing/hash files disclosed there and NOT copied):
  decision GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN, blocking_findings [], 13 non-blocking notes, bundle_sha256
  recomputed independently = ded97290231a6e8fab9ba1680790fd3feead45dde9c9912048bde41462546928. Verified: 96/96 source
  hashes agree three ways (working tree = snapshot = declared; 96 regular files, no symlinks, all 0600); SOURCES tuple
  = map keys; V9 basis 4567a0f5… unchanged; 52 runner constants extracted inertly, 17/17 bundle fields equal, both
  arm_requests rebuilt byte-for-byte from constants, common_config == CONFIG; CONFIG vs V9: exactly 7 leaf differences,
  all inside the new refinement_allowance and the corrected completion_budget_declaration, budget values byte-identical,
  cpu_budget_seconds == LIMITS.case_cpu_seconds == 1860, guard 1500; LIMITS identical to V9 and to the protocol table
  (19 values); case authority from the frozen review source (11 cases, 11 scan / 6 CAD digests; raw17 digest
  2459f4e9…d157 and case-list digest 943268be…813e reproduced, identical to the V9 PRE; no raw file opened); attempt
  tree pristine (FROZEN.json + source; outer records absent); archive SOURCE.tar 2cb7d6fd… 3858944 B, 103 entries =
  6 dirs + 97 file members, EVERY member payload streamed and hashed 97/97 = snapshot = declared, manifest e62eda42…
  agrees 97/97; protocol e59aff47… / design 0a68d42a… equal snapshot/bundle/closure table; waiter e7ade783… contract
  confirmed (O_EXCL, flock, atomic exit record, unobserved-on-failure, \d+ regex, direct --waiter refuses; not
  crash-proof); all 25 closure-table hashes match; Codex audit hashes match the coordination closure; no-overlap ps
  checks at 16:06:25Z and 16:15:23Z found nothing; hw.memsize 128 GiB vs the 16/12 GiB limits. Files copied VERBATIM
  with hash verification: docs/research/gold11_mesh_development_v10_attempt/PRE_RUN_AUDIT.json (38389 B, O_EXCL, mode
  0600, 9c14099a88506210d471a326ccc5397d541ae5afea145713749acb1acb3cfa0b; the runner's verify_pre_run_audit predicate
  replicated inertly = True) and docs/research/2026-09-13_gold11_mesh_development_v10_physical_pre_run_audit.md
  (40038 B, 313525fc3284159ba8f1232d6dbc5fb0e97436d90248aa2812f5c1ab8ab63a23). FROZEN.json unchanged (dd93ccdd…).
  Next: fresh no-overlap admission, then the ONE waiter-launched --execute.
- 2026-09-13 16:27 UTC (date -u) — V10 SINGLE EXECUTE LAUNCHED through the frozen durable waiter after a fresh no-overlap check
  (ps at 16:26:45 UTC: no run_gold11_/check_/pytest/cad_guided/_gold11_vN_exact_scoring process), outer records
  absent and the attempt tree exactly FROZEN.json + PRE_RUN_AUDIT.json + source: `.venv/bin/python
  scripts/gold11_v10_execute_waiter.py` printed "LAUNCHED 2026-09-13 16:26:48 UTC waiter 70512 reservation
  docs/research/gold11_mesh_development_v10_attempt_outer_pid.txt no overlapping process; tree pristine", launcher
  exit 0. PRE non-blocking note 1 verified immediately: the reservation record reads "RUNNING
  60084d20c0a6ffa9af8a8d66fc8f7e4b waiter_pid=70512 runner_pid=70513 2026-09-13 16:26:45 UTC"; the exit record is
  UNOBSERVED_PENDING (nonce 60084d20…, schema GOLD11_V10_OUTER_EXIT_RECORD_V1); outer stdout 0 bytes at launch; ps
  shows the waiting parent 70512, runner 70513 (--execute), internal coordinator 70514 and the first inference
  worker 70515 (RSS 1.15 GB at 3 s). Monitoring from here is ps-only plus the outer exit record; no attempt content,
  cache or raw data is read while live. The campaign bound is 104400 s wall with 22 serial slots of at most 3540 s
  wall / 1860 s CPU each; a kill inside a slot is a retained negative outcome, never a cap extension.
- 2026-09-13 23:07 UTC (date -u) — V10 EXECUTE TERMINAL: the waiting parent 70512 observed the actual outer return code 0 for
  runner 70513 (exit record OBSERVED_BY_WAITING_PARENT, started 2026-09-13 16:26:45 UTC, finished 23:04:19 UTC,
  nonce 60084d20…; pid record FINISHED … returncode=0; ps quiescent at 23:05:00 UTC). Outer stdout: one terminal
  line, status COMPUTED_PENDING_INDEPENDENT_POST_AUDIT, release_allowed false, durable_terminal_case_records 22,
  final_campaign_inventory 309 entries (008b07bc…). Attempt tree now holds the terminal records (COMPLETED.json,
  FINAL_GATE.json, INFERENCE_TERMINAL.json, SCORED.json, SCORER_RESULT.json, SCORER_RECEIPT.json,
  SELECTED_REFERENCES.json, cases/<11>/<2 arms>, leaf_dispatch/). HEADLINE, exactly as the unchanged scorer path
  returned (SCORED.json schema GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V4, experiment GOLD11_MESH_DEVELOPMENT_V10,
  status COMPUTED_DEVELOPMENT_DIAGNOSTIC, 11 cases / 6 families / 71 references, bundle ded97290…, reference
  snapshot d16390b2…): control TP 0 / FP 0 / FN 71, precision null, recall 0.0, F1 NULL, emitted support lines 0;
  challenger identical (TP 0 / FP 0 / FN 71, F1 NULL, 0 emitted). NO NUMERIC F1 EXISTS for V10; null stays null;
  V10 is a second zero-prediction terminal outcome. Independent non-author POST commissioned (reviewer
  `independent-ai-post-auditor-v10`, read-only, writes to job tmp) and my own diagnosis from the attempt's retained
  records started; V10 is TERMINAL and will not be reopened, resumed, rescored or edited. Next successor only with a
  documented mechanism, a separately versioned freeze and independent PRE.
- 2026-09-13 23:15 UTC (date -u) — V10 AUTHOR DIAGNOSIS from the attempt's retained records (POST still running; nothing in the
  attempt modified): result narrative docs/research/2026-09-13_gold11_mesh_development_v10_result.md
  (f0cf0c90fb053793cf6fd223ca49f154af42d19ee2b4bbf5557de2876ad28b5f). Taxonomy of the 22 slots: 0 CPU kills (V9: 14);
  the guard tripped in 16 slots at ~1500 s absolute CPU and every leaf returned (worker CPU 12–1556 s, RSS ≤ 2.06
  GB); 11 slots ended with no supported context (18 candidate rejections at p95 1.20–1.62 mm > 1.0 mm mostly at
  coverage .8–.9997; 9 by coverage .46–.61; CG64 ×2 had their only candidate expire at the guard); 5 slots reached
  a SUPPORTED context (CG191 ×2 with INVALID_NATIVE_SOLID native features so nothing to measure; CG1E0 ×2 and
  CG4BDD control measured 13/13/17 windows = 43 physical windows with kernel-V2 objects through the bridge — the
  V9 defect is closed); CG65/CG84 no finite prefiltered start (4), CGE4B noisy initialization unavailable (2).
  Per-candidate refinement cost 195–1422 s CPU (31 completed, 16 expired, 28 omitted); every completed context
  reused the diagnosis query. MEASUREMENT: 43/43 windows refused at PLANE_FIT_ROLE_0 (32
  FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY, 11 FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL). Retained both-flange
  diagnostics: every crop holds BOTH sheet skins (row spread normal to the crop 1.3–5.8 mm, rms 0.60–1.01 mm,
  thickness ratio 0.21–0.98; split layer means ±0.66–0.92 mm = 1.3–1.6 mm sheet); role 0 never split (30/32 weak
  fits carry the two-skin signature with largest gap 0.03–0.43 mm < the 0.3 mm floor: bridging rows between the
  skins defeat the empty-gap partition); role 1 split in 30/43 fits: 13 PASS with rms 0.005–0.011 mm and max
  0.015–0.029 mm (δ 0.03 is physically feasible for a separated skin), 17 split-but-NOT_PLANAR with rms
  0.009–0.088 mm and max 0.034–0.98 mm (contaminant/bridging rows remain in the kept group). Arc gates never
  reached. NEXT MECHANISM (V11, separately versioned): kernel V3 layer separation robust to bridging rows (declared
  two-mode detection on the material-side coordinate, band membership around each skin plane with bounded
  refits, bridging rows excluded and published with a declared fraction bound, unchanged pair compatibility,
  unchanged smaller-coordinate choice, no inner-face eligibility, unchanged gate forms on the retained skin rows)
  with the consumer chain rebound (context V4, cover V5, contact V5, projection V6, mesh offline V18, runner V11).
  Registration p95 rejections and per-candidate cost are recorded for later successors, not changed in V11.
- 2026-09-13 23:26 UTC (date -u) — V11 B1 kernel V3 written and unit-verified: backend/cad_guided_measurement_v3.py (V2 text;
  LAYER_SELECTION_IMPLEMENTATION TWO_MODE_BAND_MEMBERSHIP_ON_MATERIAL_SIDE_COORDINATE_KEEP_SMALLER_COMPATIBLE_PAIR_V2;
  LayerSelectionPolicyV2 = the eight V1 fields + layer_mode_bin_mm 0.05, layer_membership_band_mm 0.1 (declared
  above δ; the unchanged gate is never satisfied by construction), max_bridging_row_fraction 0.35,
  layer_refit_iterations 3, layer_mode_ambiguity_fraction 0.25; deterministic two-mode detection with band
  masses, band membership with bounded refits, bridging rows counted/published/never measured, new unresolved
  status TWO_MODES_MEMBERSHIP_UNRESOLVED, unchanged pair compatibility, unchanged smaller-coordinate choice,
  INNER_FACE_ELIGIBLE_STATUSES still empty, all other text verbatim V2). tests/test_cad_guided_measurement_v3.py:
  29 passed, exit 0 (single skin = V2; disabled = V1 crops; clean two skins measured as V2; two skins with
  cut-edge bridging rows sampled at the scan spacing defeat V2's empty-gap partition and are measured by V3
  with reconstructible group planes; bridging bound refusal keeps all rows; dense edges are a third population
  (ambiguous); contaminant beyond the band is bridging while a contaminant inside the band but beyond δ is
  refused by the unchanged gate; three populations ambiguous; tilted/incompatible pairs unresolved; bounded
  deterministic mode detection incl. bin widening; policy validation; V1 policy type refused; numerical failure
  retains the diagnostic prefix; phases listed; public entry identity fields). Also appended: exact-count and
  claim-strength corrections to the V10 narrative and the V11 diagnosis (31/15/28, 15 guard slots, 26 rejected
  with one overlap, 30+30 iteration limits on all 31 completed, role-0 gaps 0.028–0.230 mm for the 32 unsplit
  fits) and the reconciliation of Codex's registration postmortem (adopted; priorities recorded, not coupled).
- 2026-09-14 00:58 UTC (date -u) — V10 POST ACCEPTED (independent non-author, ACCEPT_WITH_FINDINGS, 0 blocking / 11
  non-blocking; docs/research/2026-09-13_gold11_mesh_development_v10_post_run_audit.md 8c11608939833bf4c4d85eb7fd9334e2cea3414d0e8c54a7d19f0a8fff7ea3e2,
  .json ddabac9ee54474bff0d2842e33cfc4c342905b9bba4368491d06bfa1dd180c05; bytes immutable): custody of the null-output
  ledger, not accuracy. Field reconciliation appended as Correction 3 of the V10 result and Reconciliation 3 of the V11
  diagnosis (outer inlier_plane_p95 1.196–1.550 mm vs nested context p95_upper 1.2009–1.6229 mm over the same 18
  residual rejections; whole-worker 9119.65 s / 38.6 % vs metered lane 8914.70 s / 39.4 %; CG64 = retained refinement
  work, no completed candidate; windows CG1E0 C 13 / H 13 / CG4BDD C 17). V10 stays terminal.
  V11 SOURCE CLOSURES against both Codex audits (kernel 8815d7cd…, consumer chain 521db720…; drift acknowledged, closed
  at current bytes; item table in docs/research/2026-09-14_fable51_v11_review_response.md): kernel V3 — joint band > δ
  contract enforced at measurement; membership fixed point required within the declared refits (completed refits and
  loop entries published separately; exhausted loop abstains with the exact reason); third-population admission by
  exact band-population support (Codex 40/40/6+6 probe ambiguous, 11 rows not); band masses EXACT inclusive
  coordinate-band counts (0.0999 in / 0.1001 out; binary64 interval bounds published); complete MEMBERSHIP_RULE
  (nearest plane, ties → primary plane, UNASSIGNED only between the planes, outside-sandwich rows kept in the nearest
  group for the unchanged gates) with exact partition/kept-set reconstruction tests; exact projection-premise fixture
  (narrow 12°-tilted second population); displaced ORIGINAL rows: air side → kept and refused NOT_PLANAR, between
  skins → exact counted unverified exclusion, above the bound → refused; dense uniform edges refused by the bound;
  headers/tests state the exact V10 counts (34 single-unresolved role-0 = 32 no-gap + 2 singleton-tail, 7 selected,
  2 incompatible; 24 selected fits beyond δ = 7 + 17). Consumer chain — contact V5 inert layer-policy validator mirrors
  the kernel constructor domain and cross-field rules (all-copies-mutated tests; kernel/mirror agreement proved by
  test); every current machine identifier names NATIVE_BRIDGE_V2 (mesh offline V18, projection V6, runner V11, tests;
  bridge V2 alias/docstring say kernel V3); runner CONFIG_DELTA_FROM_V10 declares the final mechanism (exact masses,
  secondary population floor, sandwich rule, refit accounting, excluded-support semantics) and the exact counts;
  protocol/design/diagnosis state the honest split (outside-sandwich deviation refused vs between-skin deformation an
  explicit counted exclusion under 0.35; no universal NOT_PLANAR guarantee). Codex acceptance family added to
  tests/test_cad_guided_v11_connected_chain_v1.py (2.06 mm scan radius vs 2 mm nominal, 2.5 mm skins, cut-edge rows,
  two oblique poses + row-order variant, TP 1 against the CONSTRUCTED reference, seven controls). Light focused
  receipt on the current bytes: 10 files, 912 passed / 1 skipped, pytest exit 0 (kernel V3 36; contact V5 163; chain
  20; runner source + waiter 17; context V4, cover V5, projection V6, V18 integration, factory-to-measurement).
  FULL-BYTES REAL-CONNECTED FIXTURE tests/test_cad_guided_v11_full_bytes_connected_v1.py (STEP → real factory v1 →
  bridge V2 → real mesh/topology/CAD sampling → real initializer V16 + registration V5 under the frozen V11 CONFIG
  policies → real surface roles → context V4 / kernel V3 / cover V5 via inspect_cad_guided_mesh_v18; scan radius 5.06
  vs CAD 5.0, 2.5 mm skins, sheared 35° cut rows inside the span, per-coordinate noise ≤ 0.005 mm, oblique pose,
  strip symmetric under S so the pose truth is the family {truth, S∘truth}). Execution history, all retained:
  (a) revision 1 (square axis-normal cut patches, one process): registered/measured, radius 5.0612, carrier within
  bound, but the real surface-role filter excluded all 558–567 cut rows before the kernel (unlike V10's zero cap-role
  exclusions) — mechanism not exercised, fixture revised; (b) revision 2 (fixture bytes 61f680e9f1ce1ac5c4bcc1c4fc17d0a37ac36909006d5b4fa34733441ee278bf,
  sheared cuts, ALL THREE CASES IN ONE PYTEST PROCESS, 818 s, pytest exit 1): case 1 = geometry/assertion outcome —
  registered/measured, best truth-family pose action 0.32 mm against the then-declared 0.2 mm bound (43 mm values are
  the S-symmetric family, not errors); cases 2–3 = ABSOLUTE CPU-GUARD EXHAUSTION in the shared process
  (PROVISIONAL_REGISTRATION_UNAVAILABLE after the 720 s absolute guard), execution-envelope outcomes, NOT geometry
  results; (c) revision 3 (fixture bytes 5e1eb4c0f4773da5472e4756f89a3177433cf708217a5a8d2ccf9e5480c5b782: connected
  160-row dent patch with unchanged background noise, group raw_support ID disjointness for arc + both flange roles,
  pose family bound to every context feeding an emitted group, 0.5 mm pose bound declared with its reason, in-test
  fresh-process guard, EXECUTION_ENVELOPE guard) under the predeclared per-case fixed envelope: one case per fresh
  process, ulimit -t 900 (CPU), alarm 1200 s (wall), OMP_NUM_THREADS=1, allowance 900/720 s, 4 GiB RSS BUDGET checked
  against /usr/bin/time -l final HWM plus ps samples (not a continuous RSS supervisor); launcher
  /Users/idantal/.claude/jobs/1519fdc2/tmp/run_fullbytes_fresh.sh. Case 1 (non_nominal): START 00:46:18Z END
  00:56:57Z, 639.2 s wall / 635.7 s user CPU, HWM 498,253,824 B, pytest exit 1 — every geometry assertion PASSED
  (envelope guard: no expired/omitted; PARTIAL/COMPLETE; bridge V2 identity; every emitted group's contexts within the
  0.5 mm pose family bound; radius 5.06 ± 0.02, not nominal; carrier within 0.10 mm / 1°; retained raw support nonempty
  for arc + both roles); the ONLY failure is the over-strong exact sheared-row count (131 observed vs 136 nominal:
  sheared rows at d = 2.7 / 12.3 lie 0.2 mm from the crop bounds, inside the pose bound) — an assertion-strength
  issue already flagged by Codex, not an engine defect. Cases 2–3 are running under the same envelope with the SAME
  5e1eb4c0 bytes (no source change to a started case); their results are appended below when they finish; the
  sheared-count claim is narrowed only afterwards (tagged sheared rows must never appear in retained raw support;
  aggregate counts bounded by the interior/boundary construction; dent-ID exclusion stays exact and non-vacuous).
- 2026-09-14 01:40 UTC (date -u) — FULL-BYTES CAMPAIGN 1 (fixture bytes 5e1eb4c0f4773da5472e4756f89a3177433cf708217a5a8d2ccf9e5480c5b782,
  launcher 208279c865d72c9a7e5997322980d649fb7057cafcbe898981f1bf425d203b6c: one fresh process per case, ulimit -t 900,
  perl alarm 1200, OMP_NUM_THREADS=1, allowance 900/720 s, /usr/bin/time -l final HWM) — all three cases terminal, receipts
  retained under the job tmp folder fullbytes_fresh_campaign1_5e1eb4c0/: (1) non_nominal START 00:46:18Z END 00:56:57Z,
  639.2 s real / 635.7 s user, HWM 498,253,824 B, exit 1 — every preceding assertion passed (envelope guard, PARTIAL/COMPLETE,
  bridge V2 identity, every emitted group's contexts within the 0.5 mm pose family, radius 5.06 not nominal, carrier ≤
  0.10 mm / 1°, retained raw support nonempty) and it failed at the over-strong exact sheared-row count (131 vs 136);
  (2) air_side START 00:56:57Z END 01:07:14Z, 617.6 s real / 614.5 s user, HWM 511,279,104 B, exit 1 — empty predictions,
  INDETERMINATE, supported registration and the first inspected NOT_PLANAR window reached, then the same 131-vs-136
  count assertion failed (nothing after that line is claimed); (3) between_skin START 01:07:14Z END 01:17:36Z, 621.7 s
  real / 618.3 s user, HWM 501,383,168 B, exit 1 — a different outcome: UNAVAILABLE / COMPLETE
  ['NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS'] (not the count assertion). DIAGNOSTIC of (3): dump_fullbytes.py
  invoked T._run directly (OMP_NUM_THREADS=1, no ulimit/alarm/time wrapper — NOT an externally bounded execution; 613.2 s
  process CPU by process_time; ps samples ≈ 0.46 GiB RSS; purpose diagnosis only, no receipt claim): registration
  completed 12/12 candidates (0 expired/omitted), 8 supported contexts; ALL 8 windows MEASURED_CONTEXT with the
  changed mechanism exactly as designed (dented role: unassigned = 160 patch + sheared rows, e.g. 291/279/296; other role
  sheared only; outside-sandwich 0; after-selection PASS, max residual 0.0054 mm; zero cap-role exclusions); the cover
  then abstained by the EXISTING context/cover rules: the cross-context arc-ownership rule
  (`_invalidate_overlapping_records`: arc rows claimed by more than one record) marked all six support records
  AMBIGUOUS_SHARED_RAW_OR_DUPLICATED_POSITION_SUPPORT → NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS, and the
  whole-component raw-support cover found no member whose support covers the union of the eight pose-family contexts
  (arc missing 52–217 rows, flange missing 30–202 per record) → NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER. The
  revision-1 positive dump shows the same two-record invalidation with a cover that DID exist (one member covered the
  other), so this is the declared cover behaviour under a larger pose family, a legitimate retained abstention — no gate
  weakened, no prediction forced. Pose-family evidence of the 8 supported contexts (max row action): 0.12, 0.32, 0.37,
  0.51, 0.53, 0.73, 4.02, 4.50 mm — the frozen p95 ≤ 1.0 mm gate also admits far contexts of the S family, so exact and
  interval claims are attributed only to windows whose own context is within the 0.5 mm bound. FIXTURE CLOSURES (Codex
  fixture audits, consumer-chain audit appendages 540aa7bf…/2ac8b116…): connected 160-row patch with unchanged
  background noise; group raw_support ID disjointness (arc + both roles) from the patch AND the tagged sheared rows;
  every emitted group's contexts and every window used for a claim carry their own pose-family check; sheared identity
  narrowed to interior/possible bounds 90/152 (nominal 136; the earlier `> 100` precondition was wrong, as Codex found)
  with no tagged sheared row ever in retained support; refused-context attribution; between-skin case asserts the
  mechanism unconditionally on attributable windows and accounts for BOTH declared cover outcomes (prediction with
  disjoint support, or the abstention verified from the published supports); RSS budget stated as a final-HWM check
  plus ps samples, not continuous enforcement. Light focused receipt after the wording corrections (selected vs
  opposite population; 26 cases; JSON numeric domain): 912 passed / 1 skipped, exit 0. FULL-BYTES CAMPAIGN 2 launched
  01:39 UTC on fixture bytes 6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110 with the SAME launcher
  and envelope (no cap change); results appended when terminal.
- 2026-09-14 (host date -u at write: see the line's own timestamp below) — CORRECTION to the previous entry's clock
  claims (append-only): that entry was written at host 01:35 UTC, not 01:40, and campaign 2 did not "launch at 01:39";
  the binding times are the launcher receipts: campaign 2 case 1 (non_nominal) START 2026-09-14T01:34:35Z END
  2026-09-14T01:45:12Z. All dates in this ledger for the full-bytes campaigns are bound to the launcher START/END
  lines of the per-case receipt files, never to estimates. Campaign 2 case 1 RESULT: pytest exit 0 (1 passed), 637.5 s
  real / 634.0 s user CPU, HWM 516,112,384 B; 5 supported contexts, 5 measured windows all attributable (pose-family
  action 0.15–0.36 mm, both family members present), 1 prediction, measured radius 5.0612 mm (scan 5.06, CAD nominal
  5.0), carrier line distance 0.00026 mm, retained support arc 1140 / flanges 990 + 1020, per-role unassigned rows
  119–136 within the declared [90, 152] interval, outside-sandwich 0, zero cap-role exclusions. Codex reconciled the
  campaign-1 between-skin diagnosis independently from fullbytes_between.json (7e463b2a…, 35,702,457 B). Two
  outstanding narrow items after campaign 2 quiesces (no edit during it): (i) the between-skin abstention oracle
  (lines 515–519 of 6ff4d154…) unions flange columns before one global swap, unlike the planner's per-member swap —
  smallest closure: assert one component containing all records and that every member's ARC support misses the
  component's arc union (arc failure alone proves no cover), checked offline against the retained byte-bound
  fullbytes_between.json plus a focused helper check, no third geometry campaign; (ii) the air-side case is an
  ATTRIBUTABLE-CONTEXT local dent refusal (out-of-bound contexts admitted by the frozen p95 gate may still emit), not
  a whole-workflow guaranteed dent rejection — final claims limited accordingly.
  (written at 2026-09-14T01:45:52Z)
- 2026-09-14 02:08 UTC (date -u) — FULL-BYTES CAMPAIGN 2 TERMINAL, ALL THREE CASES pytest exit 0 at fixture bytes
  6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110 (same launcher/envelope as campaign 1, no cap change;
  receipts under fullbytes_fresh/): non_nominal START 01:34:35Z END 01:45:12Z 637.5 s real / 634.0 s user, HWM
  516,112,384 B — 5 supported contexts, 5 measured windows all attributable (0.15–0.36 mm), 1 prediction, radius
  5.0612 mm (scan 5.06, CAD 5.0), carrier 0.00026 mm, support arc 1140 / flanges 990 + 1020, unassigned 119–136 in
  [90, 152]; air_side START 01:45:12Z END 01:55:27Z 615.0 / 611.7 s, HWM 500,678,656 B — 8 windows all
  FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL, 3 attributable (0.12/0.32/0.36 mm) with outside-sandwich 160 in the dented
  role and 0 in the other, 5 unattributable reported, 0 predictions (an ATTRIBUTABLE-CONTEXT local dent refusal, not
  a whole-workflow guarantee); between_skin START 01:55:27Z END 02:05:46Z 618.4 / 616.2 s, HWM 499,056,640 B — 8
  windows measured, 3 attributable with the patch as an exact exclusion on top of the sheared interval (291/296/279 =
  131/136/119 + 160), no patch or sheared row in any record's retained support, cover branch
  ABSTAINED_NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER (the existing cover rule; no gate weakened, no prediction
  forced). AFTER quiescence (ps 0 pytest): the oracle-only correction (one component holding every record; every
  member's ARC support misses rows of the component's arc union — arc deficit alone proves no cover; no
  flange-permutation oracle) applied → final fixture bytes 8b0556ccfa2f152887a88505e0760f307c99f773c43fe02abcd811803204335a; checked offline
  against the retained campaign-1 diagnostic dump 7e463b2ae6282881cb22697753dc7f4f49d02fd13161accb4c2a497b2fb3186f
  (6 records, component of 6, arc union 1169, arc missing [59, 208, 89, 78, 52, 217]); module collects 3 tests; the
  reconstructed pre-oracle bytes hash back to 6ff4d154… (21 diff lines, oracle block only); no geometry rerun for the
  oracle/prose edit. Final light focused receipt on the final bytes: 912 passed / 1 skipped, pytest exit 0. Claim
  corrections applied: selected vs opposite population (outside-sandwich rows of the opposite skin face only the pair
  planarity checks; conformance unknown; no upper-surface gate), 26 invalid policy cases, JSON numeric domain
  equivalence, attributable-context dent refusal. CLOSURE RECEIPT (complete hash table of the 24 V11 sources/tests/docs
  + all campaign receipts) appended to docs/research/2026-09-14_fable51_v11_review_response.md (af0bda6872a92872c56312c030f24ac0e4cfd4795ead3f6966b1553c47ad71de).
  Prior bounded acceptances stand at their pins (kernel 8fd5c3e6…, contact dd0963c2… / test 12e9944a…, runner
  8432203e…, waiter 1a7be9fd…). READY FOR FINAL BOUNDED ACCEPTANCE of the source bundle; next in order: exclusive
  --freeze, independent non-author PRE + hashed archive, single waiter execute, POST. No physical launch by this entry.
- 2026-09-14 02:18 UTC (date -u) — V11 BOUNDED SOURCE CLOSURE ACCEPTED by Codex (coordination doc
  docs/research/2026-09-13_fable51_v11_codex_coordination.md 716ea68231e4fda11e3f0fbe0595fab75cda2f08fcc06835c34a5d1c5b26f218;
  final consumer acceptance b6ba7bf9e9b2821cf80feb36141e85cf5398095aecf5bcd019d25f59fdeca890; all 24 pins of review_response
  af0bda68… verified independently; light log v11_focused_final.txt bdc5f564… = 912 passed / 1 skipped / exit 0).
  EXCLUSIVE FREEZE executed ONCE at 2026-09-14T02:16:10Z after a fresh ps no-overlap check (0 run_gold11_/pytest/
  cad_guided/_gold11_vN_exact_scoring/execute_waiter processes), an absent-attempt-directory check and a re-check that
  all 24 pins still equal the accepted closure table (0 mismatches):
  OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v11.py --freeze → exit 0,
  {"status": "FROZEN_NOT_EXECUTED", "bundle_sha256": "0a03a2984325c3092e2eaef21d29d5be49e2c626292bfa84984b9cc0447e4c65"};
  attempt docs/research/gold11_mesh_development_v11_attempt (FROZEN.json 61999 B mode 0600, file SHA256
  b4260576f1d6024233c410fd628958446bdea1cea1b5fdfa67394a885e2a86f5; 107 frozen sources under source/; basis V10
  runner f7fd083f… verified by the runner). EXTERNAL ARCHIVE docs/research/gold11_mesh_development_v11_physical_frozen.b9bcd723/
  (directory mode 0700): SOURCE.tar created with COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata, SHA256
  94601f14ed4276cab8b01c7f1afa040d56a4432aa2b51f23319cf32b5ed54e3c, 4,389,376 B, mode 0600; 108 file members =
  FROZEN.json + 107 sources, 6 directory entries, 0 unexpected (no AppleDouble entries). MEMBER PAYLOAD manifest
  SOURCE.tar.members.sha256.json (41,667 B, SHA256 34ecdcc64f0883e546299bb025cfe350e2b962014cf8a592eca891dd152d5d31):
  every member payload sha256 = on-disk snapshot sha256 = FROZEN.json source_sha256 (0 mismatches, 0 missing, 0 extra,
  FROZEN.json member present with its file sha256, canonical bundle sha recomputed = 0a03a298…). No raw data, Gold,
  cache or earlier-attempt artifact touched. Independent non-author PRE commissioned (reviewer role, read-only,
  writes only to job tmp pre_v11/; artifacts to be copied verbatim with hash verification into the attempt directory
  and docs/research). Only if GREEN with no blockers and after a fresh no-overlap admission: the ONE waiter-launched
  --execute under the unchanged frozen limits.
- 2026-09-14 02:33 UTC (date -u) — Independent physical PRE complete (non-author auditor `independent-ai-pre-auditor-v11`,
  fresh reviewer context, read-only; wrote only to job tmp pre_v11/; 9 scratch files disclosed there and NOT copied):
  decision GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN, blocking_findings [], 8 non-blocking notes, bundle_sha256
  recomputed independently = 0a03a2984325c3092e2eaef21d29d5be49e2c626292bfa84984b9cc0447e4c65, written 02:28:14Z.
  Verified: 107/107 source hashes agree three ways (working tree = snapshot = declared; 108 regular files 0600, no
  symlinks); key set = SOURCES; V10 basis f7fd083f… byte-identical to the terminal V10 attempt's frozen entry (96 common
  entries unchanged, 11 new); 21 bundle constants match the runner constants evaluated inertly by AST; CONFIG vs V10:
  exactly the five new layer_policy fields (0.05 / 0.1 / 0.35 / 6 / 0.25), LIMITS and CERTIFICATE_CAPS identical,
  allowance 1860 = case CPU 1860, guard 1500; case authority from the frozen review source (11 cases, 6 CADs, raw17
  digest 2459f4e9… reproduced, cases array = FROZEN order; catalog 4ad243c1… / selected d16390b2… declared; no raw
  file opened); runtime metadata = live venv (CPython 3.11.15, numpy 2.4.6, scipy 1.17.1, cadquery-ocp 7.9.3.1.1);
  attempt tree pristine; frozen limits = protocol table; both arm requests match; archive 94601f14… 4,389,376 B with
  108/108 streamed member payloads = snapshot = FROZEN and the manifest agreeing 108/108; protocol/design name the
  mechanism and basis; review-response closure table 24/24 = working tree; coordination record documents the
  acceptance; no blocking audit finding open; waiter byte-identical to its frozen copy with the O_EXCL/flock/token/
  atomic-exit/unobserved-on-failure/--waiter-refusal contract and the exact --execute command; two ps snapshots 399 s
  apart clean; hw.memsize 128 GiB. Files copied VERBATIM with hash verification (O_EXCL, mode 0600):
  docs/research/gold11_mesh_development_v11_attempt/PRE_RUN_AUDIT.json (26966 B,
  1f18c87ff13c24adde04e66998eec65446e263483ede9549dfb81d586c6b91dd; the runner's verify_pre_run_audit predicate
  replicated inertly = True) and docs/research/2026-09-14_gold11_mesh_development_v11_physical_pre_run_audit.md
  (22546 B, 77219c4c7bb9b7709d9133193bc047298387f8517a18d83112cfb13ab499fedd). FROZEN.json unchanged (b4260576…).
- 2026-09-14 02:33 UTC (date -u) — V11 SINGLE EXECUTE LAUNCHED through the frozen durable waiter after a fresh no-overlap
  admission at 02:31:41Z (0 run_gold11_/check_/pytest/cad_guided/_gold11_vN_exact_scoring/execute_waiter/dump_fullbytes
  processes), outer records absent, attempt tree exactly FROZEN.json + PRE_RUN_AUDIT.json + source, and 107/107 frozen
  pins re-verified against the working tree and snapshot: `.venv/bin/python scripts/gold11_v11_execute_waiter.py`
  printed "LAUNCHED 2026-09-14 02:31:44 UTC waiter 91187 reservation
  docs/research/gold11_mesh_development_v11_attempt_outer_pid.txt no overlapping process; tree pristine", launcher
  exit 0. Reservation record: "RUNNING ab6523a517e4efc908c5eb03dc60ecdb waiter_pid=91187 runner_pid=91188 2026-09-14
  02:31:41 UTC"; exit record UNOBSERVED_PENDING (nonce ab6523a5…, schema GOLD11_V11_OUTER_EXIT_RECORD_V1); ps at +6 s:
  waiting parent 91187, runner 91188 (--execute), internal coordinator 91189, first inference worker 91190 (RSS 1.24 GB).
  Monitoring from here is ps-only plus the outer exit record; no attempt content, cache or raw data is read while
  live. Campaign bound 104400 s wall, 22 serial slots of at most 3540 s wall / 1860 s CPU each (guard 1500 s); a kill
  inside a slot is a retained negative outcome, never a cap extension. Next: observed outer exit → quiescence →
  independent non-author POST → honest physical result (null stays null; a numeric F1 is a baseline with its dominant
  failures; 0.4580152672 non-comparable).
