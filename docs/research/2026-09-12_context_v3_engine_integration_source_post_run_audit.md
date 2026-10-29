# Context V3 engine integration: independent terminal failure POST

Date: 2026-09-12. Disposition: TERMINAL FAIL — 21 passed, 73 failed out of
94 collected outcomes. Publication/custody checks passed; the integration
qualification did not. The dense-support outcome passed and is retained as
narrow context-capacity evidence, not a replacement 1/1 gate or promotion of
the failed 94-outcome gate. No rerun was performed or authorized by this audit.

This reviewer also authored the exact PRE. I did not author the implementation,
tests or harness. This is independent AI technical audit, not human approval.
The PRE missed a concrete cross-field test-policy error despite reading its
constructor and the fixture. That review miss is acknowledged below; the
frozen PRE is preserved rather than rewritten after seeing the outcome.

## Exact invocation and custody

Root reports full read/acceptance of PRE in bea7c5, one actual invocation in
0c1370/session9997, terminal exit 1 and final stdout in ef0f49. Root's external
process assessment cf4a6f found no Sherman worker. These are root-supplied
terminal/process observations, not a new process query by this reviewer.
Root subsequently confirmed all three final-stdout inventory bindings match
the independent measurements below, with status FAIL, outcome gate false and
final publication gate true.

Authorized retained attempt:
`docs/research/cad_engine_v1_synthetic_checks/context_v3_engine_integration-u7kxjxe4`.

| Retained file | Bytes | SHA256 |
| --- | ---: | --- |
| STARTED.json | 9390 | 1b08c8d5ad60fadb185e8667624b7edce7e9ead8478686853102eb278084d77e |
| OUTPUT.txt | 301688 | 8ee9f61c93063454846284eee6955023772806ab8d1efd6b3183cfc12012f744 |
| COMPLETED.json | 19431 | 3b00f135f4f5b7a007fcbe8deb8a4da05fddf8429d65b67190355429c16a87e3 |

Names-only inventory is exactly these three names. Each was read as a regular
no-follow file with stable descriptor/path identity and complete byte count.
Total retained bytes: 330509. The canonical sorted compact JSON name-to-
{bytes,sha256} map SHA256 is
`03b986c9a00e35c3187c4781c8775ee4e7ffdc75fad34ccd4735086accbaa063`.
It matches the bindings root confirmed from actual final stdout. COMPLETED
cannot contain its own final inventory hash; final stdout and independent
re-reading supply that separate publication boundary.

Accepted source archive:
`docs/research/cad_engine_v1_synthetic_checks/context_v3_engine_integration_frozen.6s_689m1/SOURCE.tar`,
1294336 bytes, SHA256
`626db8f139d52c82171ccbabc2977bd1baffffcb0f5d818b1c5e78a7bf817ff5`.
I independently verified exactly 55 unique regular payloads, 1251525 payload
bytes, with exact expected names and byte-for-byte equality to the 54 live
sources plus PRE. No payload was extracted to disk or executed. This agrees
with root's preceding archive verification 1c8c1b.

The exact PRE remains 24557 bytes, SHA256
`621b574778f5f060cade251c121bbfc4a3882ce2d23208414765bf084027d851`.
All 54 source files remain unchanged, total 1226968 bytes, source path-to-SHA
map digest
`74cd2f88eb2ec6e9ac0afce97f50e3d40966358e0db7244ee364c2009b6d3777`.
STARTED, worker postflight, parent postflight, archive and live bindings match.
The unique worker marker in OUTPUT parses to exactly COMPLETED.worker.

Fresh stdlib-only executable/distribution/initializer metadata matches STARTED
and both postflight runtime objects, canonical digest
`3065f28fced0b459c575c783852e8fb3a96d7935a0322ade43e9d16ff0d90473`.
This retains the exact Python3.11.15 executable/stat identity, darwin arm64,
NumPy2.4.6, SciPy1.17.1 and pytest9.1.1 entrypoint bindings enumerated in PRE.
It is not full executable/extension/BLAS attestation. No package or target was
imported for this POST.

## Denominator and failure localization

The worker reports 94 collected nodes, 21 ordinary passed calls and complete
passing phase triplets, 73 failed calls, zero setup/teardown errors, skips,
xfail, xpass, deselections and collection errors; invalid_report is false.
Worker pytest exit code is 1. The harness enforced the frozen 93/1 per-file
collection split before running. I independently reconstructed all 94 ordered
node IDs from the frozen AST literal decorators, without pytest collection.
Their canonical digest exactly matches the retained receipt:

`69baa0af2a2e5cbad43b89712333c0f6580af2d1acabaa6ed1294fd4ae6e4866`.

OUTPUT contains 73 unique FAILED node summaries, all within that exact set.
All 73 corresponding failure sections identify the same policy-order cause:
68 direct ValueErrors and five assertion failures because the expected refusal
reason did not match the actual INVALID_WHOLE_PART_POLICY_ORDER. There is no
second failure cause in those sections. Integration has 20 passes/73 failures;
dense support has its sole pass. Pytest's text summary is 73 failed, 21 passed
in 2.07s; this is not the supervisor's wall measurement.

| Failed integration function (test_ omitted) | Outcomes |
| --- | ---: |
| actual_initializer_scheduler_proposer_finite_prefilter_chain | 2 |
| configuration_owns_context_caps_independent_of_schedule_mode | 4 |
| context_mode_is_strict_and_precedes_array_preparation | 7 |
| context_v3_cannot_silently_disable_pair_lane | 5 |
| tampered_actual_context_cannot_enter_finite_prefilter | 14 |
| actual_initializer_whole_refusal_keeps_no_usable_start_prefix | 2 |
| actual_workflow_binds_all_three_contexts_and_full_raw_transport | 1 |
| every_selected_context_is_bound_before_any_measurement | 12 |
| empty_pair_schedule_does_not_bypass_mandatory_v3_publication | 10 |
| invalid_complete_context_index_table_cannot_publish_prefix | 4 |
| late_workflow_refusal_never_exposes_measurement_prefix | 3 |
| workflow_top_and_policy_publications_are_independently_bound | 4 |
| workflow_invalid_context_mode_never_reaches_native_or_array_decoding | 4 |
| actual_workflow_v2_schedule_request_retains_mandatory_context_v3 | 1 |

These failures occur while constructing test arguments, before the intended
initializer/context/prefilter or workflow-binding call. They do not measure
those production paths' ability to accept or refuse the intended cases.
The preliminary module-identity fix allowed the coherent import graph to load;
it did not cure this separate policy-fixture defect.

## Root cause: invalid fixture combination, not invalid production defaults

Frozen test helper _policy at
`tests/test_cad_guided_context_v3_engine_integration.py:131` constructs
WholePartRegistrationPolicyV1 with max_inlier_plane_p95_mm=0.1 and
max_correspondence_distance_mm=0.5, but omits huber_delta_mm. The omitted field
therefore retains its frozen default 0.25.

The frozen policy __post_init__ at
`backend/cad_guided_global_registration_v1.py:63` requires:

`huber_delta_mm <= max_inlier_plane_p95_mm <= max_correspondence_distance_mm`.

The fixture asks for 0.25 <= 0.1 <= 0.5, which is false. The exact error is
INVALID_WHOLE_PART_POLICY_ORDER at line68. AST-only extraction of constructor
defaults and fixture keyword overrides independently reproduces this
inequality without constructing the class. Other order clauses are satisfied
by the fixture: trim_fraction=0.85 and max_rotation_step_radians=0.15.

Production defaults satisfy 0.25 <= 0.50 <= 2.0. The initializer does not create
this inconsistent override combination on its own; the new test helper does.
The evidence therefore establishes a test-fixture error, not a production-
default regression or a reason to weaken the production admission rule.
It does not prove there are no later integration defects, because those stages
were not reached by these 73 cases.

The independent PRE should have combined the effective inherited defaults with
the fixture overrides and checked this invariant. I read both sites but failed
to perform that cross-field reconciliation. That was a preventable review miss,
not unforeseeable geometry behavior. Source closure, exact hashes and resource
planning cannot substitute for checking that a fixture reaches its intended
stage. The terminal failure is retained to make this distinction explicit.

## Four passing cases did not test their intended refusal

Of the 21 ordinary pytest passes, four are
`test_context_budget_selector_rejects_untyped_or_legacy_profiles` with
dict, boolean, v1 and v2 parameters. Each accepts any ValueError around
_configuration(...). Python evaluates _policy(m) before calling the production
configuration function; it raises the same earlier policy-order error.
Consequently these four cases pass for the wrong reason and provide no evidence
that the intended context-budget selector was reached.

Their recorded pytest status remains PASS; this audit does not rewrite raw
counts into different outcomes. It withholds the claimed semantic coverage.
The other 16 integration passes reconcile as one fixture/numeric-alias control,
nine direct context-configuration snapshot checks, four requested/returned
publication checks, one schedule-capacity marker control and one direct
selected-binding expected-configuration refusal. Together with the dense pass
and the four wrong-reason passes, these account for all 21.

Any NEW successor source should admit one explicit valid baseline policy before
testing invalid downstream inputs, assert all inherited cross-field constraints,
and require the specific intended exception/reason or a reached-stage signal.
A fixture-only correction could explicitly choose a Huber delta no greater
than the declared 0.1 threshold while keeping geometry thresholds unchanged.
This is a source proposal, not an edit or permission to rerun. Frozen production
defaults/type checks, this test revision, PRE and attempt must remain unchanged.
A corrected successor needs separately frozen sources, exact PRE and one new
authorized invocation; later production behavior remains unknown until then.

## Dense capacity: actual pass retained separately

The independently named dense test does not call the bad registration-policy
helper. It completed on exactly 900000 distinct invented grid rows, 1794404
triangles, two450000-row supports and1800000 cumulative indexed memberships.
It required the real context proposer to return COMPLETE, four signed families,
all32 proposed slots, no rejected/absent slots, three completed indexes and
research-only/unknown physical status. The test did not invoke extraction,
native CAD, registration refinement, measurement fitting or scoring.

The one dense stdout record is 3472 bytes including prefix/newline, below8192.
Pytest's progress characters precede that record on the same line; I parsed
the unique fixed marker and complete following JSON, not a fabricated
line-start-only record. It binds raw SHA256
`16614630fbdfb108b12d6d302e7fdedc85ae7f6261310189ba2a292b3c04db06`.
No fixture arrays were regenerated or rehashed in this POST; the original
runtime assertions and retained record are the actual execution evidence.

| Dense metric | Retained value |
| --- | ---: |
| Raw rows / triangles | 900000 / 1794404 |
| Indexed memberships / indexes | 1800000 / 3 |
| Index row counts | 900000 / 450000 / 450000 |
| Leaf rows / node slots | 256 / 8192,4096,4096 |
| Query count / evaluated rows E | 3 / 11 |
| Helper row / index visits | 3600011 / 9000011 |
| Helper coordinate visits / sort work | 34200099 / 35100000 |
| Helper node visits / nodes pruned | 8246 / 27 |
| Helper leaves evaluated | 0 |
| Parent row / index visits | 4500011 / 23360871 |
| Parent sort work / hash bytes | 153530676 / 100621312 |
| Parent-owned / index-owned bytes | 86221376 / 15579648 |
| Modeled parent peak bytes | 219084336 |
| Modeled helper peak bytes | 64866560 |
| Temporary bytes after work | 0 |
| CAD vertex visits | 60 |

These match the PRE's independent formulas with E=11. Node work is8189
construction visits plus57 query-node visits. Seed evaluations suffice for this
axis-aligned fixture; zero leaf evaluations is not missing query work.
All three scan intervals are[0,899], with witnesses0/899,0/899 and450000/450899
as analytically specified. CAD sign/witness checks also completed as part of
the passing outcome. All three indexes share the bound raw owner; scope and
sort axis2 match the fixture. Exact pose duplicates28 do not remove the required
32-slot ledger.

The modeled parent peak is49351120 bytes below the frozen268435456 cap.
This supports the declared dense-support context-capacity case under its
original limits. It does not establish the full4-million-membership boundary,
129 indexes, arbitrary geometry speedup, the real small initialization chain,
full-workflow runtime or physical accuracy. The failed gate cannot be promoted
because this narrower case succeeded.

## Resources, quiescence and publication

| Whole worker / supervisor metric | Actual | Frozen limit |
| --- | ---: | ---: |
| Worker CPU seconds | 2.2794009999999996 | 45 |
| Worker HWM bytes | 412172288 | reported diagnostic |
| Sampled group peak RSS bytes | 412172288 | 1073741824 |
| Supervisor wall seconds | 2.4068723749369383 | 60 |
| Work-stop seconds | 2.3915305421687663 | 55 |
| Output bytes | 301688 | 1048576 |

These CPU/RSS/times cover the whole94-outcome worker, not an isolated dense
benchmark. Parent prepublication time is2.4167939578182995 seconds; root reports
through-inventory wall approximately2.417284 seconds. Parent synchronous I/O is
outside the supervisor clock. RSS sampling is not an OS allocation cap.

The sole observed group member was19635. Retained phases show work stopped at
2.3915305421687663 seconds, child reaped at2.3915338339284062, empty group checked
at2.4064084170386195 and terminal decision at2.4068723749369383.
process_group_confirmed_empty=true, requires_group_attention=false;
budget_stop, primary_error and cleanup_error are null. Postflight errors and
forbidden loaded-module lists are empty. The worker and parent both correctly
publish outcome_gate=false; final publication succeeded without converting
ordinary test failures into supervisor success claims.

No unknown-name contents, private cache, old attempt, raw/Gold/heldout/email
input or unrelated process was inspected. This POST used only the released
three artifacts, accepted archive, bound live sources/PRE and stdlib runtime
metadata. No native operation, signal, target import/collection/execution or
rerun occurred. Only this NEW audit file was written.

## Final decision

Accept the retained evidence and publication custody as an exact terminal
failure record. Do not accept the94-outcome integration gate as qualified.
Retain the independently passing dense context outcome as a narrow engineering
observation and explicitly exclude the four wrong-reason passes from selector
coverage. A corrected, stage-checked successor may address the identified test
defects under a new freeze; this audit does not issue its execution release.

No scientific input was used and no new F1 or physical score exists. Gold11V5
remains terminal/unscored, and previous successful gates are not rerun or
reinterpreted by this failure.

