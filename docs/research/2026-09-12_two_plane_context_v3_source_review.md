# Two-plane context V3 — independent source/math/test review

Date: 2026-09-12. Reviewer: Codex AI technical reviewer
`/root/cad_blocker_closure_review`, independent of the source and test authors.
Decision: **conditional source GREEN; no blocking source finding identified in
the exact candidate below.** This is not an exact PRE, permission to execute,
connected qualification, workflow integration, physical improvement or F1.
The 27 test functions contain 76 prospective outcomes; none was imported,
collected or executed for this review.

## Scope and exact bindings

The reviewer fully read V3, its design and tests, and the complete V1/V2/extrema
dependencies. Checks used source reads, stdlib AST inspection and exact hashes
only. No target definitions ran. No native calls, geometry, raw/Gold inputs,
old attempt contents, private caches or physical artifacts were accessed.
The previously accepted standalone extrema POST was read as released narrative
evidence only; its attempt was not reread or rerun.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_two_plane_context_v3.py` | 25845 | `9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d` |
| `tests/test_cad_guided_two_plane_context_v3.py` | 32810 | `adea415e42eae1a83332d4ab49976c6e057686cf1799b40246f7cb255732b309` |
| `docs/research/2026-09-12_two_plane_context_v3_design.md` | 16210 | `fadf4334d1547a09653d138ce722845243d552157c280112aa96650ea4b8b805` |
| `backend/cad_guided_two_plane_context_v1.py` | 36075 | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `backend/cad_guided_two_plane_context_v2.py` | 11130 | `d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663` |
| `backend/cad_guided_raw_extrema_index_v1.py` | 19626 | `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121` |

Standalone evidence:
`docs/research/2026-09-12_raw_extrema_index_v1_source_post_run_audit.md`,7578 B,
SHA256 `558497419cb73cb10c29451c6723c049aa852373fc8433c0b659ba3a350d6b44`.
Its 88/88 pass qualifies the exact standalone helper, not this new adapter.

During review the root changed only the test's capacity print prefix to
`TWO_PLANE_CONTEXT_V3_CAPACITY_JSON=`. Reversing that single publication-only
edit reproduces the fully read original32772 B test SHA256
`1586b1c28f11aad68efedc429488553b9cb1688c4b63bb67c7d5da8a8a262d35`.
Fixtures, assertions and the 76-outcome denominator are unchanged. The prefix
permits unique extraction even if pytest progress dots precede it; the JSON
payload remains bounded to8192 bytes and does not create another artifact.

## Principal source and mathematical conclusions

1. **The complete caller schedule and old central geometry are retained.**
   The reviewer independently compared ASTs: V3 `_slots` (333–355) has exactly
   V1's body, and the public orchestrator's full numerical `with` block is
   AST-identical to V2's. V2 global preflight still precedes input admission;
   unchanged `_planes`, `_schedule`, `_family`, signs and slots remain direct
   dependencies. Up to256 assignments still require all1024 signed families
   and8192 slot records, including explicit rejected/absent outcomes. This
   is schedule equivalence, not identical returned poses: changed scan extrema
   can change votes, exact duplicates and later nonmonotonic admission.

2. **The new cache premise is a private immutable owner, not a digest alone.**
   V1 `_inputs`/`_Work.own` captures shape, dtype and byte count before copying,
   reconstructs bytes-backed arrays, validates all rows and computes fresh
   bindings. V3 checks that bounded immutable base chain (79–88,192–207) and
   shares the same owned scan array. All scan plane supports first pass V1's
   exact sorted raw-ID union of their supplied triangles. V3's support-only
   constructor (137–173) captures assigned row bytes or all original IDs,
   checks sorted unique in-range int64 IDs and freshly hashes the captured
   support before building. Repeated immutable aliases retain strong object
   references; distinct aliases with the same scoped key must additionally
   match the original full row sequence exactly (220–243). Registry resolution
   precedes a projection-cache hit (298–307), so forged equal digests cannot
   bypass that comparison. Whole and explicit-full supports remain separate
   scopes. No source row is inferred from coordinates or a CAD nominal answer.
   These are in-call private-object invariants, not authentication of arbitrary
   externally mutated index internals or a hostile Python process.

3. **Traversal inherits the qualified rounded-program rule without a hidden
   standalone budget reset.** `_ContextIndex` overrides only `__init__`,
   `_charge` and `_memory`; `_build`, `_digest`, `_bounds`, `_evaluate` and
   `query` remain inherited. AST inspection confirms the frozen helper reads
   `self.cap` only in the three overridden methods. All inherited charges reach
   the shared parent ledger. For fixed finite inputs, sign-selected AABB
   corners evaluated by the same staged subtraction/multiplication/addition
   program enclose its rounded row outputs under the stated binary64 rounding,
   gradual-underflow and non-reassociation premises. Pruning requires both
   extrema and smallest-original-ID ties to be unable to improve. Bounds never
   become fabricated row witnesses. CAD projection still dispatches directly
   to V1's chunked matmul; there is no claim of scan-BLAS bitwise equivalence,
   exact-real intervals or physical uncertainty. `chunk_rows=3` does not shrink
   the fixed256-row scan leaves. Reports disclose that changed meaning and the
   additional bounded query domain.

4. **Persistent registry payload participates in later CAD/report admission.**
   The dynamic `_ContextWork.memory` override (111–116) includes every live
   index's numeric bytes, all registry metadata and one extra64KiB scratch
   reservation on every inherited parent call. Temporary, input, cache and
   record charges therefore cannot ignore earlier indexes. Index charges are
   cumulative, including failed attempts; no per-support reset, eviction or
   refund exists. For total indexed membership M, the inherited build plus
   support capture charges2M scan-row visits,5M index visits,19M coordinate
   proxy units,8M hash bytes and the declared stable-sort work; queries add
   their actual evaluated rows, including seeds. Exact-alias comparisons add
   separately disclosed parent index visits. The five new caps are genuine
   prospective restrictions:129 indexes,4M scoped memberships,16M node visits,
   256M coordinate units and3072 queries. They do not follow from V1/V2's old
   input-support caps and do not establish arbitrary128-support capacity.
   The query cap must also fit three times the configured family cap.

   For S support rows, the padded tree allocates
   V=2*2**ceil(log2(ceil(S/256))) node slots, not a compact2K−1 tree.
   Its retained numeric payload is8S+72V bytes; V<S/64+4 implies the design's
   conservative63016-slot upper allowance over129 indexes and4M memberships.
   Support capture, sort/gather/replacement, node allocation and leaf/query
   scratch are precharged. Already-owned context/CAD buffers, cache and report
   payload remain additional. The256MiB parent limit is a numeric/encoded
   payload accounting model, not measured RSS, caller storage or complete
   Python/allocator overhead. Full-envelope admission is not promised.

5. **Late failures clear the entire output and dispose registry cycles.**
   Registry resolution/query failures poison the private registry; inherited
   query failure also invalidates that index. The public exception boundary
   (450–464) rebuilds refusal from `_base`, retaining only bounded attempted
   statistics and discarded counts, not input bindings, projections, indexes,
   families or candidates. The new fixed refusal ceiling is8192 bytes; the
   complete-report cap remains32MiB. The `finally` path (465–467,118–132)
   invalidates completed indexes, clears index/alias ownership and drops raw,
   work and result links on success or refusal. Attempted ownership statistics
   intentionally survive and are not post-return RSS. Process interruption,
   allocation failure beyond Python recovery and externally retained private
   objects are not represented as a successful complete context.

## Prospective tests and honest limits

Independent AST counting finds27 test functions and76 parameterized outcomes,
with no skip/xfail outcome declared. The full test source was reviewed, not
collected. Ordinary invented constructors build their arrays; no legacy pytest
fixture or application initializer is invoked. Coverage includes:

- Caller depths1/5/16/256, every sign/slot, absent and rejected families, empty
  schedules, invalid old/new caps and preflight before array admission.
- Independent Python-float staged row oracles, original-ID ties, explicit
  origins, reversed assignments/normals, noncontiguous inputs and769-row
  nontrivial trees, including tiny/cancelling directions and signed zeros.
- Exact shared bytes-backed raw ownership, forbidden standalone constructor,
  equal and unequal supports, distinct whole/full scope, collisions before
  both cache-hit and cache-miss paths, and complete alias authentication.
- Cross-index counters, original parent limits, all five new guards, later CAD
  and proposal memory admissions, late query/CAD/family/report failure with
  whole-prefix clearing, and successful/refused owner-cycle disposal.

The900000-row test has one whole raw buffer, two assigned supports of four rows,
four scan triangles, four CAD triangles and one assignment. It can establish
that all raw rows remain eligible, including remote extrema witnesses, while
an index is reused through the actual context loop. It is **not** dense-plane,
129-index,4M-membership, full CAD-envelope or complete-workflow realism. Its
less-than2048 query-row and less-than3M parent scan-row conditions are prospective
case-specific work assertions, not yet observed results. CAD scalar-oracle
comparisons use these exactly representable/axis-aligned invented fixtures;
they do not prove general matmul/staged equivalence. Loose-bound/worst-case
pruning controls remain in the separately qualified frozen helper suite.

## Required qualification and integration boundary

The actual project import dependency of the new source is exactly V1, V2 and
the frozen extrema helper, with NumPy and stdlib; tests additionally use pytest
and a private temporary package namespace rooted only at `backend/`. Root and
backend initializers were checked absent. An exact PRE must still reconcile
the eventual complete harness/supervisor/test/package-initializer closure,
ordinary runtime entrypoints and versions, source/PRE markers, output observer
and resource/failure/inventory handling. The current test's dynamic loader uses
ordinary Python imports, not universally source-buffer execution or a full
runtime-binary attestation. Freeze and archive the exact candidate before any
single execution; no invocation is authorized by this review.

The proposed45CPU/55worker/60supervision seconds,1GiB RSS and1MiB output envelope
is a planning proposal for this76-outcome invented gate, not a runtime guarantee
or a change to a begun experiment. The final harness/PRE/POST must state actual
outcomes, capacity counters, resource usage and custody independently.

Even a future pass would qualify only this context seam. An initializer or
workflow must explicitly select and bind V3's schema, full budget, arithmetic
and projections; it must retain all unchanged finite-support, ambiguity and
measurement gates. No accepted registration, observed bend count, uncertainty
or physical F1 can be inferred from these seed proposals. Gold11 V5 remains a
separate terminal unscored failure. No additional broad gate or physical rerun
is authorized here.
