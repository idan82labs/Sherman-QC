# V9 completion closure — 05:06 delta acceptance

Date: 2026-09-13. Reviewer: Codex Astra `/root/v8_finite_start_audit`.

**ACCEPTED for the bounded completion-lane source/test closure.** The remaining
public V16 binding/measurement-dispatch gap from the preceding closure report
is addressed. No blocking finding remains in this scoped delta. This is not a
physical PRE, execution authorization, independent test run, new F1, geometric
accuracy result, or production qualification.

Only the append-only response delta, named source/test changes and their hashes
were inspected. No imports, tests, geometry, raw/Gold/attempt/cache access or
engine/test edits were performed. Only this new addendum was written. The
preceding report and its limitation at the earlier hashes remain unchanged:
`2026-09-13_gold11_v9_completion_final_closure_codex_audit.md`, SHA-256
`5c83937d6e30356670fc00da9e8d8782626ee39339f25c1001149632806d4f4f`.

## Exact inspected hashes

| File | SHA-256 |
| --- | --- |
| `docs/research/2026-09-13_fable51_v9_review_response.md` | `94ec97a96d4496548f5870d7cbcfbf6f43872c8e9cbcbd147fec8137aada23d1` |
| `backend/cad_guided_mesh_offline_v16.py` | `469c444f49cdcd631f04635516304b6af530fe3c813dc1f07cadb51389ed16a7` |
| `backend/cad_guided_plane_initialization_v15.py` (unchanged) | `505f313dbf3e0487bf03f3b92ab1d42d90f61d5df3123c34c9a91677d132647e` |
| `backend/cad_guided_report_envelope_v7.py` (unchanged) | `b7abe653c6f7b9832c61687ec7f1ed68aa44361b8ef467b8cae84bf837bef075` |
| `tests/test_cad_guided_completion_v1_engine_integration_v1.py` | `20695f7faf2e70a6b2d0c33ffaea1a05ca3f8170303ea0cfd33254e690536875` |
| `tests/test_cad_guided_plane_initialization_v15_completion.py` | `367a059196a9d3c0df894a5c808a6b4f73c9b552774db82ba67ec790cbf381a7` |
| `tests/test_cad_guided_report_envelope_v7.py` | `6c643d4d55d8ae5d7508f4a76ccf06175e951e02eca35f37ac75aa44f46b3eea` |

All match the 05:06 response snapshot. Other agents own the identity chain and
runner/waiter deltas; this addendum does not accept those on their behalf.

## Full public binding and dispatch are now exercised

`tests/test_cad_guided_completion_v1_engine_integration_v1.py:842` constructs
two explicit origin configurations for group 0:

- Completion-only: its former ordinary hypothesis becomes a rejected
  two-of-three parent; the selected group has ordinary origins `[]` and
  completion origins `[0]`.
- Mixed: a fourth rejected hypothesis is the parent, while the selected group
  retains its legitimate passing ordinary origin and the new completion origin
  at exactly the same pose.

The positive test at `:891` calls `_bind` (`:884`), which invokes the public
V16 `_registration_binding` with the actual context/configuration checks.
The fixture wrapper at `:553` through `:558` records calls and delegates to
the original binder; it does not substitute an acceptance result. The test
then invokes `inspect_cad_guided_mesh_v16` through `state.run`, requires all
three context bindings and one measurement/validator dispatch, checks the
published origin-lane labels, and confirms prediction transport (`:894`
through `:904`).

The integration boundaries for byte preparation, native CAD, registration,
measurement computation and measurement-publication validation remain explicit
substitutes (`:527` through `:577`). Accordingly this closes **binding and
dispatch wiring**, not an unmocked extraction-to-physical-measurement run. The
fixture's completion ledger is constructed publication evidence, not measured
support. Actual finite completion and late-failure behavior were covered by
the separately reviewed invented initializer tests. Neither category claims
physical localization or bend-measurement accuracy.

The fault matrix at `:907` through `:947` invokes both the real public binder
and workflow for missing completion origins, a non-passing child, an invalid
parent slot/status, an unusable lane, a passing hypothesis substituted as
parent, and a replacement naming a passing face. It requires the specified
direct refusal and no workflow measurement dispatch. A rejected parent forged
as a main origin is also checked for the completion-only group.

## Origin reconciliation delta is appropriate

`backend/cad_guided_mesh_offline_v16.py:313` through `:324` now reconciles
the selected group's completion indices with all PASSED children claiming
that group **even when the selected index list is empty**. Previously a mixed
group could remove all completion indices and avoid the helper call because
its valid ordinary origin kept the group admissible. The new equality check
closes that omission; absent/malformed lane publication cannot supply claimed
completion origins. Existing exact replacement-history and ordinary-origin
validation remain in force. The mixed missing-index test exercises the newly
required refusal reason.

## The skip is a disclosed non-applicable mutation, not successful coverage

There are fourteen parameter combinations in the fault matrix: two origin
modes times seven faults. The `forged_main_origin` mutation is defined as
setting group origins to `[0]`. In the completion-only fixture, hypothesis 0
is the rejected parent, so this is a genuine forgery and must refuse. In the
mixed fixture, hypothesis 0 is already the legitimate passing main origin;
the same mutation would not be a fault. That one combination explicitly
skips at test `:919`.

Thus thirteen fault combinations are exercised and one is non-applicable.
The skip does not conceal a missing positive mixed-origin test: the mixed
positive path runs separately. Nor does it prove that every other possible
forged ordinary origin on a mixed group is covered. The reported suite result
of **117 passed, 1 skipped** correctly retains this distinction. The appended
response reports **855 passed, 1 skipped, 0 failed** across all thirteen suites;
these remain author-reported results, not reviewer-executed evidence.

## Precision notes resolved

The zero-child cap check at
`tests/test_cad_guided_plane_initialization_v15_completion.py:410` now creates
an axis-unresolved proposal fixture, verifies a populated parent ledger with
zero scheduled children under the ordinary cap (`:425`), then requires the
tiny-cap refusal with an empty retained child list (`:427` through `:429`).
This is a helper-level control-flow fixture with deliberately substituted
plane normals, not a physical geometry claim. It now exercises the intended
empty-child terminal branch.

`tests/test_cad_guided_report_envelope_v7.py:60` now tests malformed top-level
`evidence`, malformed nested initialization, and a retained direct completion
list alongside malformed evidence. It requires unknown counts or the explicit
known-lower-bound status, as appropriate. The earlier untested one-line source
correction is now represented by targeted assertions.

The response correctly limits lane-size accounting to normalized JSON with
`ledger_bytes=null` on non-fatal terminal paths; fatal mutations remain covered
by the outer serialized-report cap. This is the scope accepted in the prior
review, not an expanded all-terminal-path guarantee.

One nonblocking stale comment remains at completion-test `:475`: it says
mixed origins are impossible. The new integration fixture and producer branch
demonstrate that they are supported when exact pose keys merge. Treat that
comment as applying only to the helper's particular completion-only fixture,
not as a system invariant; correct it in a later documentation-only edit.
It does not invalidate the explicit mixed-origin positive/refusal tests or
require another broad review.

## Handoff

The prior completion-lane source closures and the final missing wiring check
are accepted within their declared scope. Proceed with root's remaining
independent checks and prospective freeze/PRE process; this report itself
does not authorize physical execution. Preserve terminal attempts, negative
results, the disclosed skip, provisional units and uncertainty, and the
distinction between wiring evidence and a new measured Gold11 score.
