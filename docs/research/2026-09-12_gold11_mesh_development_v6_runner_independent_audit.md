# V6 runner independent source audit

Date: 2026-09-12. Disposition: no additional concrete P1/P2 defect found in the
reviewed revision. This is source review, not qualification, a PRE, timing
release, authority to read inputs, or evidence of a successful physical run.
The independently discovered initial sampler-loading defect is fixed in this
revision; that fix is source-verified, not runtime-verified here.

## Exact scope and identities

I read canonical AGENTS/current checkpoint, the complete current design,
V5-to-V6 source delta and affected custody, policy, projection, worker, scorer,
coordinator and outer-finalization functions. I checked the actual supervisor
V3 receipt construction/finalization, sampler implementation, initializer V13
configuration selector, workflow V14 call/publication seam and projection V2
context validator. Only named source/design files were read. No target import,
collection, test, runner invocation, process observation, native operation,
physical input, Gold, prior-attempt artifact or private-cache access occurred.
Python checks parsed source ASTs and computed source hashes only.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| scripts/run_gold11_mesh_development_v6.py | 166049 | 7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80 |
| docs/research/2026-09-12_gold11_mesh_development_v6_runner_design.md | 17868 | a943ad18577f1a4ee450227c9e879215fccb218dc94a167b0979ab17b96747f2 |
| scripts/run_gold11_mesh_development_v5.py | 153975 | 71d55915db6e6fcb8ae7f8c658de7d7e0c4b93b553eb78413f3914b037e4e481 |
| scripts/gold11_owned_process_supervision_v3.py | 42751 | 9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea |
| scripts/gold11_engine_stage_sampler_v1.py | 8142 | 80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0 |
| scripts/gold11_schedule_projection_v2.py | 14677 | 728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db |
| backend/cad_guided_plane_initialization_v13.py | 38651 | 79ee4de537f1dddcc1dee40d33b25fe3fe8fb667bcd4e30ecc0a811ebe2d989f |
| backend/cad_guided_mesh_offline_v14.py | 33435 | 76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749 |

These are reviewed source identities, not a complete frozen execution closure.
Root owns further source changes and the exact focused qualification gate.

## Concrete integration conclusions

1. **Actual V3 receipts match the new custody predicate.** The supervisor really
   publishes the schema, direct-child reap, attention, escape, still-present
   unknown identity, cleanup-error/discard and control-close fields required by
   `writers_quiescent`. For outer new-session supervision it additionally
   confirms an empty owned group after reaping. Inner receipts leave that group
   field unset, correctly admitted only by the leaf form of the predicate.
   V3's reap path requires its observed owned descendants to be gone. The
   runner does not infer leaf quiescence from exit code alone or revive old
   numeric PID/group signal authority. A missing receipt, including a raised
   supervisor call, leaves the coordinator's quiescent flag false.

2. **Private result retention and deletion are ordered conservatively.**
   `retain_worker_result` rejects before enumeration/stat/content reads unless
   strict quiescence holds, then binds the original private directory identity.
   It retains only the complete/partial derived result; partial bytes are not
   successful inference. The cache context deletes only its fresh identity after
   no dispatch, or after quiescence plus successful retention. Unknown writers
   and retention errors leave opaque private caches. Cleanup failure is marked
   possibly partial and stops later dispatch/scoring. Marking inference dispatch
   attempted before issuing its one-use control can conservatively retain a
   cache even when issuance fails; it does not create unauthorized deletion.
   The scorer's outer finally attempts its receipt after cache handling. Storage
   failure propagates rather than manufacturing a durable receipt.

3. **Coordinator cache and finalization no longer repeat the old unsafe order.**
   Fresh coordinator cache path/device/inode/uid are bound before launch and
   checked against cwd/HOME/TMPDIR before activation. That cache stays opaque
   outside RUN on success and failure. Leaf dispatch records retain their own
   metadata; the outer process never discovers/reclaims leaf caches after an
   interrupted coordinator. The retained campaign schema no longer admits
   `coordinator_work`; an unexpected name remains a metadata veto. In `execute`,
   strict outer-group quiescence precedes source/runtime/control postflight,
   terminal filling, artifact reads and complete-tree inventory. Unknown writers
   produce all 22 nondurable positions, with no fallback scoring. Administrative
   failure publication is not presented as scientific content verification.

4. **The real policy call and pure projection agree.** Source AST comparisons
   show CONFIG, LIMITS, certificate caps, seeded/composite budget maps and V2
   schedule expression unchanged from V5. The actual registration tuple remains
   .25 <= .5 <= 2 mm; the earlier invented integration fixture's invalid tuple
   is not present here. Both arms construct actual context V3 budgets and call
   the initializer's owned configuration selector before raw-byte reads. The
   complete 12-key request contains 23 integer budgets and eight declarations;
   those budget/declaration ASTs exactly match projection V2's independently
   declared maxima/strings. Actual workflow V14 keywords accept the passed
   modes/types. V7/schedule V1 control and composite/schedule V2 challenger stay
   distinct; both honestly declare V14 implementation over V13 transport.
   Projection and revalidation use original returned RESULT bytes and full
   expected context/scan/schedule requests. Contact scorer inputs are obtained
   only through that reconstruction, not arbitrary wrapper unwrapping.

5. **Sampler connection is observational, with bounded incomplete evidence.**
   The initial wrapper reopened a scripts source after a backend-only audit
   hook, which would prevent reaching the engine. Current `leaf_sources` names
   the exact sampler; worker authorization precedes hash-checked compilation,
   which precedes installation of the unchanged backend filesystem hook. The
   wrapper now requires that verified preload and performs no source I/O. It
   supplies only a frozen source-path allowlist and fd 2, not input/result data.
   Ordinary sampler constructor/start/finish/write exceptions do not replace
   the one engine return object or its exception. Root's last narrow addition
   retains `startup_failure_type` outside the finish receipt, so a successful
   finish cannot erase an earlier startup exception; I read that final wrapper
   delta independently. Process termination and hard
   resource failure cannot promise a finally receipt. The frozen sampler's
   complete-byte counter excludes failed partial samples; its receipt exposes
   unknown partial bytes and a still-live thread. The runner does not upgrade
   those values into completion. A missing, malformed or interleaved terminal
   diagnostic must likewise remain incomplete in POST.

The sampler allowance is 524288 bytes plus a 4096-byte framed terminal record,
leaving 520192 bytes under the unchanged 1048576-byte leaf log cap. Both leading
and trailing receipt newlines count. This is not an independently enforced
native-output quota: the supervisor still caps the whole combined stream and a
log overrun may terminate inference. That explicit resource effect is not
scientific result rewriting. Zero 30-second samples do not show no engine work;
a Python sampler cannot certify continuous native-stage observation.

No claim above converts source ordering into real-host integration success,
maximum transport-envelope capacity, registration accuracy, calibrated physical
uncertainty or F1. Darwin coordinator ru_maxrss remains lifetime high water,
explicitly not current RSS; V3 supplies the separate sampled live-tree measure.
The draft's release boolean and timing status still block invocation. Its old
numeric timings are carryovers, not newly approved diagnostic limits.

## Smallest representative connected diagnostic proposal

Use a NEW separately frozen unscored diagnostic, not a subset flag or resume of
the terminal V5 attempt. Select exactly three already-consumed cases using only
previously released dispositions: one with representative cumulative-support
capacity pressure, one with a returned registration/coverage refusal or ambiguity,
and one with earlier CPU exhaustion or unknown unfinished stage. Freeze exact
identities/order and the selection rationale before new raw access; do not select
using new geometry, Gold, favorable outcomes or adaptive substitutions. No new
case identities or input paths are selected by this review.

For the smallest diagnostic representative of both proposed full-run arms,
run those three cases once per arm: **six fixed terminal slots**, serially, with
the same policy constructors, complete raw bytes, native V14 public workflow,
context V3, actual proposal/schedule/prefilter/measurement path, sampler and V3
supervisor. Use the same fresh-cache custody and exact RESULT retention. Retain
and reconstruct every available original-result V2 projection; preserve every
refusal, budget stop and missing stage. A small numerical fixture or mocked
workflow is not a substitute for this connected byte path.

The diagnostic should stop at inference/projection/custody evidence: **no catalog,
selected references, scorer call or F1**. That avoids a new scoring exercise for
what is a capacity/registration/CPU localization question. If scoring is desired
instead, it requires an explicit separately frozen three-case reference scope
and denominator; the all11/71 scorer contract must not be silently relabeled or
weakened. The current all11 V6 orchestrator cannot simply receive three cases:
its exact-11/22 assertions must remain intact. A new thin diagnostic orchestrator
may reuse its reviewed leaf/policy/projection/custody seams under its own honest
schema and six-slot terminal contract.

Record per slot the explicit arm/configuration hash, source-bound input hashes,
exit/budget/custody outcome, CPU and HWM versus sampled-RSS scopes, returned
context memberships/work counters, retained result/projection hashes, and last
observed stack with unknown-later-work language. Set exact CPU/work/outer-wall,
output and memory caps in the new protocol before PRE; do not inherit the V5
numbers as authority, renew a deadline, increase caps after a stop, substitute a
case or rerun a terminal slot. Any new diagnostic still reuses consumed data;
it is not unseen-family validation or a retry of V5 under its old authorization.

The go/no-go decision for a later full11 run is whether this connected diagnostic
produces interpretable bounded terminal evidence without a runner/custody failure
and whether any remaining capacity/CPU stop is understood enough to justify the
declared full-run plan. All-positive geometry is not required. A registration
negative is scientific diagnostic evidence, not automatically a runner bug.
Any change prompted by these consumed outcomes must be declared before the next
separately authorized run; a diagnostic pass does not itself release full11.

## Handoff

Only this new audit was authored. Candidate, frozen predecessors, tests and
protocols were not changed. Root should freeze/review the already scoped focused
runner tests against the final candidate, then use the separate small diagnostic
protocol/PRE. No broad native campaign or additional data exploration is proposed.
