# Native diagnostic ledger — independent source review

Date: 2026-09-11. Reviewer: `/root/radius_math_review`.

## Decision and limits

**GREEN for source freeze and a separately prepared exact PRE, not execution.**
The identified source defects are closed in the final helper/test bytes below.
No remaining blocker was found for the eight declared integrity criteria.
This review reports no test passes, native result, accuracy qualification,
physical access permission or promotion. The connected measurement gate remains
the independently recorded 2/4 RED; an integrity ledger cannot change it.

I read the helper and independent test in full, then every final changed range;
the native recorder's policy construction, original byte/context and measurement
oracles, supplemental ordering, and reused predicate bodies; the approved ledger
design, allocation review and integration review; and the relevant native
harness/protocol and supervisor declaration chain. All declared current source
payloads were bounded-read and hashed; all Python payloads were AST-parsed.
There was no project import, pytest collection/execution, geometry calculation,
native call, simulation, inference, refit, scientific-artifact/archive/private-
cache access, or physical/control/raw-data access. Only this new document is
written. The old blocked native PRE is unchanged.

## Exact reviewed source checkpoint

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `tests/native_depth_diagnostic_ledger_v1.py` | 45373 | `596c4225451796bf56fb4f9d66066bbe84ac605730ea7d89cf51d31d42f1742d` |
| `tests/test_native_depth_diagnostic_ledger_v1.py` | 40590 | `c8325a6ed2da733386714da9361a8c3661c99e153b913e4152996c50be1cae66` |
| `tests/test_cad_guided_two_plane_depth_native_v1.py` | 49204 | `368adaa257fcdb2f80eaee6e2530c560f38315cda0c52ec4c8adc7ad46555d55` |
| `scripts/check_cad_guided_two_plane_depth_native_v1.py` | 11931 | `cb36e30d9cb7bcb39dd2afeb45d7e02421361eeb4edb554c0540442650b7ddf5` |
| `docs/research/2026-09-11_two_plane_depth_v1_native_protocol.md` | 22297 | `1216193f9c828a40c48e86debad780f736eb051c9678f1d41feefacd751acc4c` |

The approved ledger design is SHA256
`9ee24c96f708dfd4d2d94f10305ebbec15ecde466dc907c46aaf5d655cccc32f`;
the allocation review is
`eaafd73936edd16c54607ff301a2bb167f5f13ed12d2697acccc5a37a8fb253f`.
These source-review identities do not replace the final PRE/archive ledger.

## Findings and final corrections

1. **Post-accuracy cache state — closed.** The first helper required the native
   fixture loader to have exactly one cached entry. In the corrected execution
   order, the original `_evaluate` can load `ARRAY_SOURCE` before its quantity
   oracle, giving two entries. Thus a legitimate conditional return, including
   a subsequent accuracy failure, would have been refused by the supplement.
   The final helper admits the two source-defined histories, checks typed cache
   size/miss/hit/maxsize telemetry, requires the exact already-loaded fixture
   names, and checks unchanged miss counts, cache sizes and module identities
   around fixed-policy construction. The added three-case control distinguishes
   a valid two-entry history from an unexpected miss or module replacement.
   This is a source-controlled single-worker contract, not protection against
   arbitrary tampering with Python caches/modules.
2. **Stale output in an early refusal — closed.** The supplement now requires
   the exact V13 input-preparation return shape, not a conditional report with
   selected fields nulled. Later registration/binding/native/sampling fields
   must be absent, including null aliases and unknown additions. Its exact
   preparation envelope corresponds to the base refusal branch; other early
   paths remain unqualified. It does not independently reparse the supplied PLY
   or prove that the parser's stated diagnosis is correct.
3. **Reached-stage request maps — closed.** `_policy_binding` now compares both
   top-level scan-plane and two-plane-schedule configuration maps against fresh
   source-requested kwargs, in addition to thirteen policy maps and stage policy.
   This runs before early-return admission. Missing composite configuration or
   a bool-aliased schedule budget cannot receive criterion 4 PASS. On a complete
   route, the unchanged composite/schedule binders additionally compare the
   evidence/initialization copies and every registration binding row.
4. **Signed-integer range overflow — closed.** The original `abs(raw)` range
   test could overflow on `int64` minimum. Direct `raw >= -1e6` and
   `raw <= 1e6` comparisons now precede canonical hashing. The independent test
   requires input admission to fail before scientific predicates. This was an
   input-check edge, not a cause asserted for any executed native result.
5. **Oracle isolation — verified.** The original accuracy attempt follows
   durable CASE_RESULT and precedes the supplement. An ordinary accuracy failure
   still permits the supplement; a supplemental failure never changes the old
   verdict. A new AST-isolated recorder case mutates the supplied result after
   accuracy, retains the original published bytes/verdict, and reports failed
   supplemental integrity. No accuracy assertion was weakened to admit refusal.

## Mathematical and publication scope

| Criterion | Independently reviewed guarantee and boundary |
| --- | --- |
| Publication/source/input | Compact sorted JSON streaming digest exactly matches the existing recorder serializer and the supplied CASE_RESULT receipt. CAD/scan bytes and little-endian raw XYZ/triangle values are hashed before and after predicates; shapes, finite coordinates, index range and nonrepeated triangle vertices are checked. No file is reopened or authenticated by this helper. |
| Raw/native/sampler | The report binds 650 raw rows, 1176 triangles, stipulated 0.05 mm input bound, native unit/action declarations, feature inventory and 32768-sample request. Native CAD triangle/sampler arrays omitted from the public result remain unavailable; their digests are report/source-runtime bindings, not recovered arrays. |
| Context/origin/pose | Every selected context is typed, indexed and bound to the reported rigid pose digest, complete selected/exact group and reciprocal passed-origin list. Assigned support/correspondence identities agree. The existing 1e-8 SE(3) serialization guard checks proper rotation, not registration accuracy. No truth transform is used. |
| Composite/schedule/profile | Fresh fixed constructor policies are compared with returned maps; actual metadata decoding and V12 equation/match/reservation binding are reused. Context V2 schema, complete default budget/capacity profile, five input digests and joint count identities are checked. Four signs and 32 slots per ordered match are inherited schedule/context contracts, not independently regenerated geometry. |
| Raw-support/exact-cover | The unchanged V2 publication validator is called once with actual supplied raw positions. It validates the nested V1 publication, recomputes the entire exact-support cover plan, verifies every group/member/permutation/association projection and relevant finite publication constraints, and must equal the retained validator receipt. It does not repeat the circle or flange fit. Fatal/source/plan refusals cannot substitute for a complete cover proof. |
| Window/nominal associations | Windows are exactly the unique selected-context × native-feature product. Every retained record and group association is visited and bound to its native feature and declared nominal values. Where a group measurement exists, deviations equal measured radius minus nominal radius and measured sweep magnitude minus absolute nominal sweep, within the unchanged 1e-12 arithmetic comparison. This does not use nominal truth to fit or select output. |
| Quantity/carrier/projection | The existing V2 validator supplies typed finite quantity, residual, canonical axis, length, constant radius-offset, role-count and finite-declaration checks. Top-level predictions are exactly the supported V2 group instances. Nested member predictions are never concatenated. Literal contact-sector, sampled-support, raw-frame, unknown material/event/physical-trim and uncalibrated uncertainty semantics remain unchanged. |
| Branch/denominator | A complete conditional or complete-no-measurement disposition requires all eight criteria PASS, exact status/computation/reasons/projection and no discarded prefix. COMPLETE/UNAVAILABLE with zero measured groups is not a malformed positive or missing physical bend. A reason string alone is insufficient. Other missing prerequisites remain NOT_EVALUABLE/UNREACHED or failed, never complete integrity. |

The exact-cover validator's finite endpoint checks are serialization/model-
consistency checks against the already reported fit. Neither it nor the ledger
proves continuous surface coverage, manufactured event identity, physical inner/
outer assignment, measured radius accuracy, a unique joint pose, calibrated
uncertainty or datum conformance. The assurance label
`REPORT_BINDING_PLUS_SOURCE_ENFORCED_RUNTIME_CHECK` is appropriate; a stronger
native-array recomputation or metrology claim would not be.

## Traversal, bounds and failure accounting

New report preflight is iterative, before reused nested validators: depth at
most 64 and 8,000,000 cumulative nodes across both before/after passes. Ordinary
container/scalar types are admitted explicitly; nonfinite values, unsupported
arrays, subclass keys and cyclic/over-deep reports refuse. Each encoded pass is
limited to 129 MiB; CAD and scan buffers to 8 MiB and 1 MiB. JSON fragments are
allocated before their incremental byte check, so these are admission/work
bounds, not a preallocation or instantaneous RSS guarantee.

List limits are 64 native features, 48 contexts, 4096 windows, 512 records/groups/
predictions each, 8192 hypotheses/exact groups and 4096 associations per admitted
record/group. Consequently the complete window product has at most 48×64=3072
members, still within its separate 4096 cap. These maxima are not asserted to
fit simultaneously within the report-node/time limits. Reused metadata, context
publication and exact-cover validators keep their own existing bounds, including
cover graph/link/row/comparison and publication-row guards; they are not charged
as if the new node counter measured their internal work.

Per-criterion counters explicitly count supplemental predicate invocations.
Ordinary per-record failures do not suppress later admitted records or safe
independent criteria. Eight error examples per criterion are retained while
failed counts continue. A reused bulk validator that raises has unknown internal
progress, not a fabricated zero/full visit count; its bulk row says this and the
whole criterion fails. Missing prerequisites never create a successful prefix.

Record and input post-hashes run in `finally` on ordinary paths. A mutation or
post-hash failure vetoes integrity. `BaseException`, including local budget
interruption and KeyboardInterrupt, propagates rather than becoming completion.
The local wall timer does not extend a shorter existing timer. CPU is checked at
boundaries and wall interruption is 120 seconds; neither is a hard OS guarantee.
The supplemental 90 CPU/120 wall seconds are inside, not additional to, the
native 2520 CPU/2815 monitored-work/2820 parent seconds, with sampled 4 GiB RSS
and 1 MiB OUTPUT. Completion is not guaranteed by those engineering ceilings.

The ledger itself is at most 32 KiB and lives only in the existing printed
receipt/OUTPUT, after durable CASE_RESULT. It is not inserted into that original
129 MiB result or its 1 MiB metadata allowance. Four scientific files and four
administrative files are unchanged. Parent quiescent inventory and independent
POST must bind actual bytes, source and process custody; absent/duplicate/
truncated output or any budget failure remains negative. Original accuracy PASS
alone cannot admit a missing, failed or mutated supplemental result.

## Independent tests and source closure

Static AST count gives **16 supplemental functions / 61 prospective outcomes**:

| Fixed test family, in source order | Outcomes |
| --- | ---: |
| Complete conditional / complete-no-measurement mock routes | 2 |
| Truth and forbidden-call bombs | 1 |
| Post-accuracy cache admission / miss / replacement | 3 |
| Signed-minimum raw range | 1 |
| Independent late/source/policy/pose/projection tampering | 17 |
| All late associations and unknown failed-bulk progress | 1 |
| Record/raw/triangle mutation | 3 |
| Node/depth/record/ledger caps | 4 |
| Exact early refusal and eight contradictions | 9 |
| BaseException propagation | 2 |
| No/shorter/longer enclosing timer | 3 |
| Float64 serialization and CPU boundary | 1 |
| Context V2 profile count/input/schema | 4 |
| Unsupported report-tree forms | 4 |
| AST truth/file/engine-call exclusion | 1 |
| Recorder accuracy-first/supplemental-failure/mutation order | 5 |

The original source still declares 20 functions / 39 outcomes. Thus the proposed
combined denominator is **100 = 39 original + 61 supplemental**, not 100 native
cases. There is one actual native case, 38 original pure/mock outcomes and 61
supplemental pure/mock outcomes. The test module explicitly mocks every reused
scientific validator; its two 650×3 /1176×3 invented buffers are identity payloads,
not a geometry-positive experiment. It tests actual new orchestration/common
predicates, plus an AST-isolated recorder with fake inference/publication and one
fresh temporary directory. No pytest fixture is called as an ordinary helper.

I resolved literal/name/starred sequence declarations without executing any
supervisor: 49→62→66→72→91→97→111 unique paths. The final 111 comprise **84 Python,
27 Markdown, 47 backend sources, 2,396,049 bytes**. Every declared file was read
under a 2 MiB cap with O_NOFOLLOW/O_NONBLOCK, regular-file and stable descriptor/
path identity checks. All 84 Python files parsed. All eleven named preloaded
backend entries and the new helper/test/native AST dependencies are declared.
No new missing repository import dependency was found. This is not a fresh
scientific requalification of the inherited backend implementation.

For reproducibility, SHA256 of the canonical compact sorted JSON mapping
`{relative_path: {bytes: length, sha256: digest}}` for those 111 current files is
`e949847c2ffe1796be49db6b99d92ae172f7d2b6189e8869d61968f5bb2d5801`.
This document is not one of those 111; any subsequent declaration change needs
an independently reconciled final ledger. `tests/__init__.py` is the inert
24-byte source, SHA256
`5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b`;
the enclosing repository `__init__.py` was absent. Third-party/runtime/native
binaries are not fully attested by repository source hashes.

The helper itself imports stdlib plus NumPy, not backend/native modules; its
scientific entrypoints must already be loaded by the source-bound single native
case. The fresh policy builder and reused binders are traced source-controlled
calls, not authority to refill a missing fixture cache. The helper has no direct
file/loader/engine/refit dispatch and the `truth` argument has no AST name read.
The package/setup constraints and every final hash must be rechecked in the
separate exact PRE. No command or execution is released by this source review.
