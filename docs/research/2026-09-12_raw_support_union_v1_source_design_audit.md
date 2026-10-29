# Raw-support union V1: independent source/design audit

Date: 2026-09-12.
Reviewer: Codex independent technical reviewer `/root/cad_blocker_closure_review` (AI).
Disposition: **CONDITIONAL SOURCE GREEN**, suitable for preparation of a separately frozen invented-only qualification. This is not an execution, integration, measurement, or production release.

The reviewer read the complete planner, independent test module, design, and sole backend dependency. Review used source inspection and static counting only. No project import, collection, test, simulation, geometry/native invocation, raw/Gold read, retained-result inspection, or private-cache access occurred. The active Gold11 V5 benchmark and its frozen sources, limits, scoring, and evidence remain unchanged. This review is not another prerequisite for its score.

## Exact reviewed bindings

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `backend/cad_guided_raw_support_union_v1.py` | 23355 | `785da4e9624bcc514aad02eac4783c125b1c738764815dcd12080d072a5463ab` |
| `tests/test_cad_guided_raw_support_union_v1.py` | 27080 | `5be1fe8b0d13054e0bfa4002151c3947675811b8271a3bc0f0d1a8aa99fbf2d8` |
| `docs/research/2026-09-12_raw_support_union_v1_design.md` | 9979 | `701bdebc0f22f37bb848212bda491dfde8b88d7c505c1e8c745b20b6712b0192` |
| `backend/cad_guided_exact_support_cover_v1.py` | 10167 | `70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47` |

The supplied three hashes were independently reconciled; the dependency hash was independently computed. The test source contains **28 test functions / 93 prospective parameterized outcomes**. None has executed, and no passing-test claim is made.

## Findings

1. **Global parity is correct at source level.** For a shared flange row, the constraint is `x_i XOR x_j = s XOR t`. The forest stores the correct root relation, including union-by-size root swaps because XOR is symmetric. Checked stars retain every shared-ID equality, including cycles after an earlier contradiction. Inconsistent or disconnected relative parity refuses; a global flip is chosen by raw-ID ordering, not geometry quality.
2. **Ownership and existing-cover branching preserve the intended structural contract.** All structurally admitted records participate regardless of previous fit or finite-support status. Arc-related canonical-position ownership retains conflict members; flange-only sharing does not join components. Existing covers require the complete raw arc union and uniquely aligned flange subsets. The stable existing-cover choice precedes fallback, and no geometry result is consulted or retried here.
3. **Union support is complete and conditional.** The planner checks all three aligned n-ary anchors, complete role unions, cross-role conflicts, and distinct canonical positions. It neither welds row IDs nor drops inconvenient members. Existing-cover duplicate-position behavior is deliberately left for unchanged downstream geometry rejection rather than silently altering that lane. Structural bindings intentionally exclude ignored old measurements/statuses/associations.
4. **Declared accounting and failure handling are coherent.** Snapshot, membership, set-work, sorting, and report-copy paths have their declared precharges/reservations; encoded output is checked before successful return. Fatal input, resource, ordering, or serialization failure clears all groups and success bindings while preserving bounded refusal evidence and attempted counters. This is not a proof that the counters measure literal CPU or every allocator operation.
5. **The invented tests cover the principal branches.** They include an independent small brute-force parity oracle, record/role permutations, contradictory and disconnected parity, pairwise-without-n-ary anchors, raw/coordinate conflicts, failed-member retention, structural existing-cover preservation, duplicate positions, independent budget boundaries, allocation-free full-envelope arithmetic, and late whole-output clearing. Their collinear XYZ labels are deliberately not fitting fixtures. No source-control success establishes a common numerical model or physical bend.

No substantive planner-logic blocker was identified within this bounded source review. There is no claim of exhaustive verification or execution success.

## Capacity and assurance limits

The reviewed 4 GiB accounted reservation accommodates the formula at 1M raw rows, 512 records, and 4M role memberships: 4198408704 B for eight-byte source scalars and 4270408704 B for sixteen-byte source scalars. Those are scalar-accounting identities, not measured RSS. Independent set/order/binding/output caps may still refuse an input at those maxima. The 1024-byte-per-membership factor is a prospective accounting assumption; caller-owned inputs, interpreter/libraries, and later fitting remain outside it and require a separately frozen combined process/tree budget.

Canonical float64 positions define this planner's position domain. Their digest does not independently attest original native bytes or recover original array storage types. A quiescent caller remains required during snapshotting. The bounded seven-field policy schema is not a complete measurement-policy or geometry qualification.

The future wrapper must still validate complete original context publication, every contributing association's role-filter exclusions, original-byte/raw-row lineage, the explicit free-circle policy, and one fresh whole-support fit with finite-support checks. It must not fabricate an existing-cover witness for a union or use an existing-cover fit failure to select fallback. Physical identity, manufacturing count/sign, continuous material, trim truth, calibrated uncertainty, and production conformance remain unestablished.

## Required prospective harness/runtime bindings

Before any run, a new bounded harness/protocol and independent exact PRE must bind the test, planner, cover dependency, and every actual repository initialization/supervision dependency. In particular, ordinary pytest package loading must account for `tests/__init__.py` and the actual repository-root package boundary; this review does not re-attest their current bytes.

The test fixture creates a named synthetic package pointing at `backend/`, imports the planner and its sole relative cover dependency, and cleans up that namespace. It does not import `backend/__init__.py` or an application package. NumPy, pytest, and standard-library imports still use the declared installed runtime and ordinary Python loader. Future qualification must state that trust boundary accurately: disabling bytecode writes does not prove bytecode reads are absent, nor that all definitions came from retained buffers.

The eventual source closure, runtime, 93-outcome denominator, resource limits, clean worker state, archive, output/inventory checks, and terminal POST conditions must be frozen before the first invocation. No harness or protocol existed for execution authorization at this review, and no invocation is released by this document.
