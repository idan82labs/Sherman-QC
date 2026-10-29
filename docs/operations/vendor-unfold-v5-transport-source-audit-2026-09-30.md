# Independent V5 transport source audit — 2026-09-30

Reviewer: `/root/unfold_surface_generalization`. Runner/worker/selector author:
`/root`. This is source-only review of the draft successor, not PRE authority.
No physical inputs, control records, freeze or native geometry execution were
used. The engine/helper pin placeholders and missing protocol are intentional
draft state and must be finalized before freeze.

## Reviewed source snapshot

- `scripts/evaluate_vendor_cad_upload_unfold_v2.py`: `c568e10dad72c7f6bb307800e6eec418c259823dc2a5dd60898e9b9691f34c2a`.
- `backend/vendor_unfold_worker_v4.py`: `e0ec43f801aa85ba0ec2ed36c0ebda0f4d48361ee1bd7b6d162292077587ba63`.
- `apps/api/services/vendor_setup_service.py`: `29dd07340c38147b4566bd33ae70c4d6fa1323c5e3a1e5796358d04cb7ceccbb`.
- `tests/test_vendor_unfold_worker_v4.py`: `0ed488f6ab99053a301b51aca5fd3e257ca4c1b09d3fa79a1159e6546c72842c`.

These hashes identify the first review snapshot; later author changes require a
delta review and superseding hash receipt before freeze.

## Accepted custody and transport mechanisms

AST comparison confirms the paired runner preserves the incumbent implementations
of raw bindings, source archiving, exact upload route scope, parent-owned case
launch/nonce/exclusive claim before raw access, frozen/PRE validation, sampled
process monitoring, observed PGID cleanup, shared admission check, terminal
case accounting, exclusive execute sentinel and campaign accounting. The five
source bindings and ordering are unchanged. Control then challenger run serially
inside the same claimed case under the existing 60-second/4-GiB whole-case cap
and 330-second campaign cap.

The maintained GET selector has the required V5 unavailable factory and worker
globals in its isolated AST scope. Both controlled paths preserve source and
candidate generation checks and suppress the unbounded folded-flange helper.
The upload route remains exact maintained STEP preprocessing through V1 for both
arms. Therefore V5 does not yet improve CADs that V1 refuses before publication;
that limitation must remain visible in the five-case denominator.

Worker V4 shares the existing in-process admission slot, OS lock, durable journal
and whole-process-group cleanup with earlier profiles, while using an independent
byte cache, nonce, private worker module and explicit source/candidate/engine
binding. Its monitor preflight, deadline, post-exit validation and source-change
checks preserve the incumbent behavior. Independent focused verification ran
`tests/test_vendor_unfold_worker_v4.py` in the isolated pytest configuration:
**62 passed in 0.12 seconds**. Those are mocked supervision/cache tests, not native
integration evidence. Source helper pins must include every new engine/importer,
surface proof, stitch, contract, oracle and oracle bridge dependency before final
worker hashing. No transitive runtime-closure claim is established.

## Blocking integration findings in the reviewed draft

1. **P1 — Successful challenger has no attached oracle.** The runner demands a
   `geometry_oracle` object after an AVAILABLE challenger, but the reviewed V5
   producer does not capture native evidence, call the oracle bridge, or attach
   its result. Every otherwise successful challenger would become a child
   exception after its paired record was written. Wire validation before
   AVAILABLE publication and retain typed oracle refusal evidence.

2. **P1 — Oracle status and feature ownership mismatch.** The runner checks
   `VERIFIED`, whereas the independent oracle returns
   `VERIFIED_UNDER_DECLARED_GEOMETRY_MODEL`. Use the established status without
   strengthening its semantics. Actual upload candidate feature IDs are catalog
   UUIDs; the bridge's default `B1..Bn` mapping is insufficient. Supply a generic,
   source-derived explicit UUID-to-native-cylinder ownership map with complete
   coverage and no nearest/Gold/case-name shortcut. The author reports these
   bridge changes are underway; this review does not assume they have landed.

3. **P1 — App consumer rejects V5.**
   `frontend/react/src/features/part-catalog/pages/vendorUnfoldPreview.ts` accepts
   only mechanism 3/4; its finite-width validator requires mechanism 4/transport
   3. The new selector's mechanism 5/transport 4 result will be unavailable in
   the UI and visual harness. Add an explicit V5 admission contract preserving
   finite-width checks and validating V5 chart/source lineage, winding and oracle
   claims, then test the actual consumer with supported and corrupted payloads.

4. **P1 — Both-arm failure accounting is incomplete.** `_actual_upload_result`
   initially invokes control then challenger without preserving an explicit arm
   receipt around each exception. A control exception prevents challenger entry
   and leaves no paired result; an outer timeout can leave only a completed
   control artifact. Preserve typed completed/failed/not-executed dispositions
   for both arms on every selected case. Attempt challenger after an ordinary
   control refusal only when admission is quiescent and the unchanged whole-case
   budget remains. An integrity/admission/native cleanup failure should halt
   safely and explicitly retain the challenger as not executed. The parent
   should finalize both-arm records even after terminating the case process.

## Decision

The reviewed process/custody mechanics are accepted. **NO-GO for freezing or
executing this draft** until the four integration findings, complete helper
pins, final protocol and real-connected synthetic verification are closed.
The 60-second cap is shared by both arms, so control-first order can consume
challenger time; preserve that declared comparison model and any starvation
outcome. This review makes no support-rate, F1, physical accuracy, calibrated
unit or manufacturing claim.

## Independent delta review — explicit common5 and completed arm accounting

The common5 validator AST is exactly incumbent V2 `_validate_preview`, with
the sole body difference `(3, 4)` becoming `(3, 4, 5)`. Its referenced helpers
and numeric constants remain the incumbent imports, including the nested
`struct` and `PLANAR_TOL` dependencies. The public transport4 contract now calls
this validator on the original mechanism5 payload; there is no mechanism4
shadow or fabricated incumbent limitation.

Reviewed common5 SHA-256:
`ffa6a42bd04f292ab38b34f43d3990285d54b6bfdec1ebae36cc8008d23658a0`.
Reviewed transport4 contract SHA-256:
`163f971c215084653ec1352e6f7614f1566982d368d5a84f7ff3268b08050642`.

The producer captures native evidence and runs the independently constructed
oracle bridge before AVAILABLE publication. The runner preserves its actual
`VERIFIED_UNDER_DECLARED_GEOMETRY_MODEL` status. The frontend has explicit V5
admission, actual bend-line input to its local-strip checker, source/chart and
winding checks, complete child overlays, and the declared oracle check.
The new upload AST scope supplies its explicit V2 runtime through a scoped
`os.environ`; the control scope remains empty and selects incumbent V1 upload.
Both arm byte bindings remain identical while their separately declared
candidate representation policies are retained.

Corrected worker source pins pass `_source_pins()` with 934 declared files;
this source-only call does not touch admission or native geometry. Worker
snapshot SHA-256:
`e005857d485096bb24c9fdb458a293a108d9ab871db112fcb2dca4f11be8d686`.
The common-V2 pin remains `cbcd792106750a866b6fe960e6ff8046cd4c68a842b40c8acc01f551615be76f`.

Independent isolated verification on these authored/synthetic sources:
`test_evaluate_vendor_cad_upload_unfold_v2.py`,
`test_vendor_paired_run_dispositions_v2.py`,
`test_vendor_native_step_upload_v2.py`, and
`test_vendor_unfold_preview_v5.py`: **48 passed in 0.61 seconds**.
No raw dataset, control artifact, native geometry job, or shared admission
record was accessed during this delta review.

One remaining P1 prevents a GREEN source decision: `_paired_terminal` directly
parses a safe existing arm result. A wall/RSS kill may leave that exclusive
write empty or truncated; after cleanup, parsing escapes the execute loop and
prevents terminal receipts for the remaining denominator. Preserve a typed
invalid/incomplete-receipt failure and regress empty, truncated and oversized
arm results without interpreting them as completed AVAILABLE output.
This is a source-only NO-GO pending that narrow repair, not a run authorization.

## Final transport delta decision — GREEN for independent exclusive freeze

The interrupted-write blocker is closed. `_paired_terminal` catches bounded
read, parse and schema failures, records `UNAVAILABLE` with retained error type
and reason, and still writes both-arm parent dispositions. The challenger
cannot retain AVAILABLE when its parent has not established success. Unsafe
writers remain UNOBSERVED and are not read. Empty, truncated, array, invalid
status and oversized authored arm receipts are covered by the repaired suite:
**11 passed in 0.23 seconds**, independently run under isolated source-only
pytest. Ordinary returned control abstentions still attempt challenger inside
the same claimed bytes and whole-case budget; catastrophic exceptions preserve
explicit not-executed/failed dispositions rather than disguising retries.

Final reviewed runner:
`4cf936d5f2d012c2ef4a99ff6835a75f3219febfab714d08c642a1b5b1214c44`.
Final paired-disposition tests:
`67b7897bd3519b7812f6670d1bbd6af124b0418afdee71f9b1510cfac1fcb0c1`.
Additional final reviewed pins:

- V5 producer: `05e5004296bc1555e1cecba91f8435e5550e1d5d9855f96efdebf54d533081c7`.
- Native upload V2: `ba40c980af5316f79158f878dec104afd103f7486c59b6721929bd90adaa8dfe`.
- Maintained GET selector: `29dd07340c38147b4566bd33ae70c4d6fa1323c5e3a1e5796358d04cb7ceccbb`.
- Maintained upload route: `736d0aa7f4d762f5b1d191ee89a15910a890bdac0def40a125a576d7a6033403`.
- Frontend admission: `fc3f7b918d8d2011a9b272ac973a12976d6b11593a16bb2a2ab16eda42735403`.
- Protocol: `494e88879d0df95387e73e17b9a8780636632eaf3266f7cb47eb777124b9466e`.

The preceding common5, public contract and corrected worker pins are unchanged.
The four initial integration blockers and the interrupted-write follow-up are
closed in this transport review. **GREEN for exclusive source freeze and a
separate independent PRE**, not permission to execute or a geometry promotion.
Adding only named test/audit files to `OWNED_SOURCE_FILES` is an anticipated
freeze-closure delta; independently verify the final archive and runner pin
at PRE. Other source changes require renewed delta review.

This reviewer authored the local-strip Python/TypeScript addon and does not
self-audit its critical mathematics. Parent/other independent acceptance of
those files is a separate required gate. The source-only transport decision
does not establish physical availability, F1, CAD agnosticism, continuous
trim-area truth, manufacturing calibration or production release.
