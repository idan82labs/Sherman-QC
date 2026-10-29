# Two-plane schedule V1: independent exact pre-run audit

Date: 2026-09-10. Decision:
**GREEN_FOR_ONE_FROZEN_PURE_TWO_PLANE_SCHEDULE_V1_GATE**.
Remaining blocking findings: **none within the declared scheduler-only scope**.
Execution status at this decision: **UNEXECUTED**.

Reviewer: `/root/production_architecture_review`, an AI reviewer independent of
the scheduler author and root-authored test/protocol/harness route. This reviewer
read the complete 453-line scheduler, 691-line pure dependency, 288-line test and
85-line protocol, and checked the exact harness delta and active execution path.
This is a technical pre-run review, not a human scientific/manufacturing approval.

## Exact five-file freeze

These are the complete source-map entries for the named route. Hashes were
independently checked after the pre-execution test repair below. This audit is
a separate release document, not a sixth source-map entry.

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `backend/cad_guided_two_plane_schedule_v1.py` | 26562 | `1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f` |
| `backend/cad_guided_plane_search_v6.py` | 38804 | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `tests/test_cad_guided_two_plane_schedule_v1.py` | 14852 | `05b7d92bda320cd4fea79bde7753afb6415a375a6d669c1b828c7b01e785fcd9` |
| `scripts/check_cad_guided_measurement.py` | 79874 | `24480e4b18e00a08a9e6efecd654041965c6a2c5a0f3ac10cc8f9f684b741171` |
| `docs/research/2026-09-10_two_plane_schedule_v1_protocol.md` | 4873 | `355851393b3811e77457df3a913ecb0b480e7c43cbe2ec36c6b1ccb6430dda3e` |

The previous harness is preserved at
`docs/research/source_snapshots/two_plane_schedule_pre_628f3a84/check_cad_guided_measurement.py`,
independently checked SHA-256
`628f3a84c98806f1d24c7b47f8d7683911a2f8bfb7f67dc8698887f1d8eda0f4`.
The exact diff adds the scheduler and separately scoped terminal-artifact test
routes, their bindings, and their membership in budget/retention selections.
No supervisor mechanism changed. **This decision releases only the scheduler
route**, not the terminal-artifact suite or any V11/native/physical route.

## Pre-execution finding and verified closure

The originally proposed test SHA
`93ab157b571ee1ad10dcb72a331ef8860461eec547459014b56eac71e74aea85`
contained a Boolean-identity oracle defect in its 16 rank/cone parameters. SVD
produces NumPy scalars; `ratio >= .02 and margin > 0.` therefore returns a NumPy
Boolean, which cannot be compared by identity with `bool(scheduled_matches)`.
This would fail a correct result for a test-type reason, not a geometry failure.

The reviewer reported this before any execution. Root changed the single oracle
line to `admitted = bool(ratio >= .02 and margin > 0.)` (`tests/...:122`). The
reviewer independently proved the exact one-line delta by replacing that line
in memory and recovering the original SHA above. All other test bytes and the
other four closure files remained unchanged. The repaired hash in the table is
the only released test version. No failed runtime attempt, retry or changed
scientific threshold is being hidden: neither version was executed in this audit.

## Mathematical and scheduling review

For the represented two-row matrix with rows a,b, the larger singular value is
the square root of the larger eigenvalue of its Gram matrix. The product of its
singular values is `norm(a cross b)`, so the source's
`s_min = norm(a cross b)/s_max` avoids cancellation in the smaller Gram eigenvalue.
Normal validation makes `s_max` positive; nonfinite arithmetic is fatal. Both
scan and CAD pairs require the .02 nominal ratio and a strictly positive
`s_min - hypot(2*sin(alpha_a/2), 2*sin(alpha_b/2))` margin. This is conditional
binary64 engineering arithmetic, not directed-rounding certification or a
calibrated normal-cone model (`backend/...:187,204,240`).

The dependency's only exercised scheduler seam is the exact `PlaneEquationV6`
type and `_equation` validation. It checks finite, bounded tuple vectors, IDs,
unit normals and anchored offsets, normalizing both equation coefficients
(`backend/cad_guided_plane_search_v6.py:89`). The new scheduler adds the 10-degree
cone ceiling and snapshots typed equations while preserving caller bank order.
It does not invoke the dependency's triplet search or use anchors/offsets as
correspondence-ranking features. Original support qualification and actual
raw/full-face lineage remain unverified caller declarations here.

All at-most-8128 unordered scan pairs are examined before choosing at most64.
The greedy priority is uncovered plane participation, then conditioning, margin,
cone sum and bank-pair index. This is explicit deterministic bank-order policy,
not proof of globally optimal coverage or ID/order invariance. Selected pairs
can still have no compatible CAD match or lose residual quota; all such
dispositions remain visible.

The CAD index contains at most `C(2048,2)=2,096,128` canonical unordered pairs.
Stable angle sorting preserves input-pair ties. The indexed window expands by
the source cone sum, twice the numeric guard and the largest CAD pair cone; every
row inside that window is tested against its actual CAD cone sum and target rank.
Outside-window counts are interval-index observations, not finite-geometry
exclusions. The independently tested chunks are 1/3/65536 pair rows. Small
brute-force fixtures check canonical IDs and complete compatible-window counts;
there is no maximum-size benchmark or physical correspondence oracle.

Repeated compatible layouts are keyed by the exact ordered little-endian int32
pair-index bytes within one immutable CAD table; SHA is diagnostic, not the
equality predicate. For L compatible positions and m nonempty disjoint integer
bins, visit q selects `start_j + q mod bin_length_j`. At fixed m, each position
is exposed after at most `ceil(L/m)` occurrences of that **exact** layout. Actual
one-bin and two-bin schedules use separate clocks. This proves only the stated
repeated-layout exposure property, not per-scan-pair completeness or exposure
within the available at-most64 windows. Layouts occurring too few times remain
incomplete. Both CAD orders are retained for each selected unordered pair.

## Joint quota and bounded failure contract

Triplet counts reserve capacity first. Let H3 and U3 be the caller-declared
retained triplet hypothesis and unsigned-assignment counts, including rejected
records. Available ordered pair room is
`min(floor((Hmax-H3)/32), Umax-U3)`. Both orders form an indivisible two-record
bundle, so odd residual room cannot produce a one-order prefix. At most two
unordered CAD pairs per selected scan pair give at most256 ordered matches;
each reserves one unsigned assignment and32 future sign/context slots.
The final ledger verifies `H3+32*N2 <=8192` and `U3+N2 <=1024` (or smaller declared
joint limits), with no refunds for failed signs, duplicates or absent midpoints.
Triplet starvation is reported explicitly, not interpreted as geometric absence.
The connecting adapter must authenticate these counts against actual retained
triplet records; this helper does not prove their truth or their mutual provenance.

Maximum declared scan selection work is520,192 tests; CAD sort work is
44,018,688 under the explicit `N*ceil(log2 N)` proxy; 64 complete CAD windows
permit at most134,152,192 charged pair tests. Exact compatible-layout keys are
capped at128MiB and the accounted numeric/key/report working payload at256MiB;
report size is capped at32MiB. These logical limits are not Python allocator RSS
or execution-time predictions. Per-operation charging and final report checks
precede success. Ordinary work/metadata/numeric/report failures clear every
usable match and retain bounded attempted statistics, not a favorable prefix.
No existing triplet cap or downstream geometric gate is relaxed by this review.

## Test, imports and execution scope

Static AST enumeration independently confirms **17 functions /68 prospective
outcomes /zero skip or xfail markers**. All four Python sources parse, without
importing them. The repaired SVD oracle independently computes singular values
and margins over16 parameter combinations with fixed2e-14 comparison ceilings.
Other checks cover target-only refusal, incompatible angles, seven joint-budget
boundaries, repeated-layout/mixed-bin exposure, uncovered participation, three
chunk sizes, offset/sign independence, fourteen invalid equations, invalid joint
counts, hard work/storage/report caps, empty/omitted schedules, and clearing after
late-window/final-report failure. These are ordinary synthetic contract checks;
the injected late failures are specific mocks, not a complete exception proof.

Scheduler and dependency imports are stdlib plus NumPy only. Tests use stdlib,
NumPy and pytest and lazily load exactly these two pure modules into a private
package; no backend package/server initialization or other test file is imported.
There is no helper/test file, native, process or network operation beyond the
explicit module-source loading. Installed dependencies are the existing runtime
environment, not a fully byte-attested environment closure.

After root reads this exact audit, one first execution is released:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_schedule_v1
```

Do not use `all`, directly invoke `--worker`, run the companion terminal route
under this decision, or replay the successful context-helper gate. The named
harness path binds exactly the five sources, sets45seconds RLIMIT_CPU before
pytest import, disables plugin autoload/conftest/cacheprovider/bytecode writes,
selects only this test and uses private HOME/TMPDIR with numeric threads fixed1.
Frozen limits:60seconds monitored wall,45seconds worker CPU,1,073,741,824bytes
sampled process-group RSS and1,048,576bytes captured output. The gate is one
owned pytest process; the reviewed test creates no child process of its own.

The harness exclusively retains STARTED/OUTPUT/COMPLETED in a fresh
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v1-*` directory.
Receipts include suite/limits, before/after source maps, child exit, monitored
wall/RSS and output SHA. Setup/hash/receipt IO and cleanup are not all inside the
monitored timer; forwarding/reap are not independently hard-deadlined; sampled
RSS is not an allocation or continuous-peak guarantee. No escaped-native-child
containment is qualified by this pure gate.

## Required post-review and advancement limit

Independent retained-only POST must verify the exact fresh receipt inventory,
five source bindings including repaired test05b7d92b, all68 passing outcomes with
zero skip/failure/xfail, observed parent/worker successful exits, resource evidence
within frozen limits, no stop/drift, and the retained output digest. Missing
terminal handling, an omitted outcome or a CPU kill cannot become success from
exit-zero or a null outer stop field. Preserve every failure; no retry or cap
increase follows from this audit.

A valid pure pass permits further connected V11 qualification only. It does not
generate a pose, prove true-correspondence coverage, bind physical support, accept
registration, recover a measured bend or establish F1. Actual byte-bound supports,
two-plane finite screening, joint consumer accounting, unchanged global rank/
ambiguity/refinement and original .05-mm/free-circle output oracles remain
separate requirements. Native/physical execution needs its own release. Existing
negative attempts and their custody findings remain unchanged.

Audit operations were limited to the named source/protocol/archive, source
hashes/diff and stdlib AST/count/delta checks. No test collection, imports, helper,
native, inference or scorer execution, protected/original input access, or
existing artifact mutation occurred. Only this new pre-run audit was authored.
