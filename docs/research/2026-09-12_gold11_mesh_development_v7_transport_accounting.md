# V7 full11 request transport — independent conditional byte accounting

Date: 2026-09-12. Reviewer: `process_v2_qualification`. Scope is static source
accounting for the exact proposed V7 owned-producer/coordinator chain, not
arbitrary caller dictionaries, a runtime maximum, a test pass or execution
authority. No target was compiled/imported/executed; no test, native/process
operation, physical input, Gold, control, attempt or private cache was accessed.
No diagnostic/runtime postflight is claimed. Only this new document was written.

## Exact source identities

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `docs/research/2026-09-12_gold11_mesh_development_v7_delta_design.md` | 8879 | 55e905a7bffd25847426826d498624a4ad5b6387d829e66217c03969ba2db403 |
| `scripts/gold11_contact_scoring_v2.py` | 50413 | 9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff |
| `scripts/gold11_schedule_projection_v1.py` | 17231 | a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245 |
| `scripts/gold11_schedule_projection_v2.py` | 14677 | 728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db |
| `scripts/gold11_terminal_artifacts_v4.py` | 24341 | 3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838 |
| `scripts/review_gold11_inputs_v1.py` | 25616 | 41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955 |
| `scripts/run_gold11_mesh_development_v7.py` | 169028 | 27737eff034891e8f7a6d09ba63f5bd17a3947c67ffd553ddfb73a931fef5cb3 |
| `scripts/gold11_owned_process_supervision_v3.py` | 42751 | 9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea |

The named V7/design identities stayed stable during this review. Source-only
AST/literal interpretation used application Python3.11.15 `-I -S -B`, stdlib
JSON/hash arithmetic, and a26,294-byte inert skeleton. No192MiB object or
physical/source-test workload was allocated. The pinned review CASES was used
only for fixed case-token/family strings, never its referenced paths.

## Result

Let J(x) be `json.dumps(x, sort_keys=True, separators=(',', ':'),
allow_nan=False).encode()`, using default ensure_ascii=True and no newline.
All reviewed producer/validator encoders use that same canonical representation.
With M=8×1024²=8,388,608 bytes, the following conditional bound holds:

```text
22 admitted scorer-input bodies          ≤ 22M = 184,549,376
one reference_records_by_case object     ≤  1M =   8,388,608
all remaining request bytes             ≤          26,204
complete encoded scoring request        ≤     192,964,188
literal192MiB request cap                =     201,326,592
remaining conservative slack            ≥       8,362,404 bytes
```

Thus the proposed184MiB payload-plus-<8MiB-overhead argument is valid for the
specified chain. This is an upper bound, not a jointly reachable maximum or
a promise that every result/reference candidate is admitted by earlier gates.

## Why each payload term is bounded

1. `reconstruct_all_terminal_inputs` requires the fixed11 cases and both arms,
   reads each retained SCORING_INPUT under M, and reconstructs every scorer
   input before reference access. For a clean case, V2 revalidation checks
   original-result/projection hashes and canonical stored/rebuilt equality
   under the caller's M wrapper cap. An ADMITTED input is exactly the
   `contact_projection` subtree of that canonical wrapper. Serializing a
   subtree with the same encoder cannot exceed its enclosing serialization.
   Compact publication additionally charges its newline; this proof uses the
   looser ≤M body bound. Unicode escapes do not expand a second time between
   equal canonical objects: both use ensure_ascii=True.
2. A reconstructed REJECTED wrapper produces the fixed supervisor-refusal
   shape with prefix RUNNER_CONTEXT_BINDING_REJECTED and at most160 reason
   characters. Its conservative escaped-ASCII bound is2087 bytes. A retained
   supervisor refusal has exactly six keys, empty predictions, null count,
   False production flag, one of two fixed phases, and at most64 reasons of
   at most512 characters. Even allowing12 JSON bytes per Unicode character
   gives393552 bytes. Both are far below M; neither needs a raw-byte-to-canonical
   assumption to cover arbitrary escaping within those validated reason limits.
3. `score_after_terminal` selects the fixed reference snapshot, then successfully
   publishes `J(selected)` to SELECTED_REFERENCES.json BEFORE constructing the
   request. `publication_cap` sets8MiB explicitly; the qualified publisher
   raises on oversize or unsuccessful publication. Therefore reaching the
   request constructor establishes |J(selected)|≤M. The request reuses the
   same11 keys and only each selected row's `bends` subtree. Replacing each row
   object by its own subtree cannot increase its canonical length. Hence
   |J(reference_records_by_case)|≤M. The catalog's raw8MiB read alone would NOT
   establish this canonical bound; successful canonical snapshot publication
   is the essential additional premise. No reference bytes were read here.

## Fixed overhead recipe and exact constants

The source-pinned CASES has11 unique12-character ASCII tokens and8-character
ASCII part_key values. Reconstruct the nine-field `make_scoring_request` shape
from source. Preserve exact schema, selected-reference SHA, family map, both
full arm requests and scan-plane requests. Replace each of22 inner scorer
bodies by null and the entire reference_records_by_case object by {}. Preserve
all22 binding rows, using four64-character ASCII digests per row, and the
longer allowed projection_kind (EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER,38
characters versus37 for SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW). This dominates
the actual result_sha256=null alternative and any mixture of the two kinds.

| Canonical component | Bytes |
| --- | ---: |
| Complete dominating skeleton | 26294 |
| both arm_requests | 14692 |
| scan_plane_requests_by_arm | 919 |
|22 projection binding rows, with their two-arm maps | 9620 |
| case_family_by_id | 287 |
| skeleton minus22 null bodies and empty reference object | 26294 −22×4 −2 = 26204 |

The component rows are explanatory, not extra addends to the skeleton.
Skeleton SHA256:
`cb2d0628faaea2064b1cbea5458a72dde148939a700d35829df5d5bfb5741e9f`.
Both arm_requests SHA256:
`54cbece489ca6631357d212c2cd4a0b2eae3d762970c1340f2dbde7584aa490b`.
All separators, outer field names, case-map keys, arm names, braces, family
strings and duplicate fixed request declarations are included in26,204.
Replacing each null by a body of length≤M and {} by the reference object of
length≤M gives the exact additive bound without allocating those bodies.

## Explicit provenance premise and retained runtime veto

The binding-row bound is for actual owned receipts. V7 assigns result_sha256
only from `digest(result_blob)`; before that it is absent, including missing
terminal-fill receipts. A later failed path may retain that64-hex digest.
The qualified supervisor contributes process fields, not an engine-controlled
result_sha256; its captured stdout is not merged as receipt keys. Clean
reconstruction validates the result hash. No actual producer path assigning
an unbounded result_sha256 was found.

This is NOT a universal `scorer_arguments` input bound: in its failed-refusal
branch, that consumer does not independently type/length-bound result_sha256.
A foreign hand-built dictionary could therefore exceed this overhead premise.
The exact owned-producer/frozen-case chain, not arbitrary external acceptance,
is the subject of this proof. No new validator edit is proposed solely to
broaden that claim.

V7 still serializes the actual request and checks its full byte length before
private-cache creation/dispatch; the separate private request writer checks
again, and the scorer performs its bounded exact-hash read. These runtime
vetoes remain necessary. The proof is not permission to remove them, truncate
payloads, reduce the denominator or increase a cap.

## Not proved by this request accounting

- The32MiB scorer result is enriched with requests, bindings, stage attribution,
  selected coordinate-frame metadata and policy/provenance fields. Request
  capacity does not establish that enriched report≤32MiB. The existing
  `len(encoded(report))+1` check and refusal remain decisive.
- The256KiB terminal/control/inventory envelopes have separate combined shapes.
  Their maximum successful serialization is not closed by this request proof.
- Total live Python objects, decoded trees, encoded copies, peak RSS, CPU,
  wall time, synchronous I/O, physical result quality and successful scorer
  completion are not byte-transport consequences.

These are scope limits, not a demand for another broad gate or an all-positive
physical diagnostic. This source-only conditional proof does not change the
running diagnostic, any frozen source, V7 release status or scientific policy.

