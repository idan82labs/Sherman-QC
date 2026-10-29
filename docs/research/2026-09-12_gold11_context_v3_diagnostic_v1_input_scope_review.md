# Context V3 diagnostic V1 — independent source-declared input scope

Date: 2026-09-12. Independent reviewer: `process_v2_qualification`.
This is a source-derived comparison record, not an input verification, PRE,
launch authorization or mutable manifest. Only the three named source/doc files
below were read. No physical path was opened, statted, resolved or hashed; no
catalog, control, attempt or private cache was inspected. No target import,
compilation, execution, collection or runtime inspection occurred.

## Source pins

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/review_gold11_inputs_v1.py` | 25616 | 41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955 |
| `scripts/run_gold11_context_v3_diagnostic_v1.py` | 57923 | 08f9e50b52f0225ca54e1920ef21fed2d0840a19941fb20b338b43e6b141a44f |
| `docs/research/2026-09-12_gold11_context_v3_diagnostic_v1_protocol.md` | 12845 | da2db6127916ec2b714fa03c3f04cc901c0ff1263ed35db1b1c8d39d87822ec3 |

## Independent derivation

The review-source SHA was checked before parsing its AST. The single literal
`CASES = json.loads(...)` payload was decoded with duplicate-key rejection,
without running that source. Case keys, unique case tokens, absolute path
strings, 64-character lowercase hexadecimal hashes and repeated-path hash
consistency were checked. Selection/order came from the new runner's literal
`SELECTED_ORDER` and `ARMS`; no paths were inferred from names.

The original declaration contains exactly 11 cases, 11 distinct scan paths
and hashes, six distinct CAD paths and hashes, and 17 distinct total path/hash
bindings. Canonical serialization means UTF-8 JSON, `sort_keys=True`,
`separators=(',', ':')`, `allow_nan=False`, no trailing newline:

- Full raw17 path→SHA map: 2546 serialized bytes; SHA256
  `2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157`.
- Selected six-path map SHA256:
  `543f7907770115a09792e662e36edba8721425c2e1e3778916e6bbaf577f942a`.
- Ordered selected three-case array, with `reference_id` removed, SHA256:
  `19048aa4ac59338406ca69d688da9a16dd98352a17ac05442612531ca02b8842`.

`part_key` is preserved in that selected array as declared bookkeeping; no
reference identifier or reference bytes are included. The full raw17 map is
a provenance declaration, not permission to read its other eleven files.

## Exact selected paths and original declared hashes

Three distinct scan paths plus three distinct CAD paths = six distinct physical
path declarations; their six hashes are also distinct. Both arms use the same
pair for their case. These are source strings, not verified filesystem entries.

| Case order / token / part | Kind | Exact absolute path | Declared SHA256 |
| --- | --- | --- | --- |
| 1 / CG08506AE97D / 10839000 | SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_2.ply` | cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b |
| 1 / CG08506AE97D / 10839000 | CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp` | ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37 |
| 2 / CG4BDD1371B6 / 37361005 | SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.ply` | d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038 |
| 2 / CG4BDD1371B6 / 37361005 | CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.stp` | 81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e |
| 3 / CG4FAF273093 / 49024000 | SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply` | 70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade |
| 3 / CG4FAF273093 / 49024000 | CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.stp` | 3cdfcba8fbea3735e3c9b3610b69ccab2bc8bf681919ef2c86839693ef54479f |

## Fixed six-slot scope

| Slot | Case token | Arm |
| ---: | --- | --- |
| 1 | CG08506AE97D | control |
| 2 | CG08506AE97D | challenger |
| 3 | CG4BDD1371B6 | control |
| 4 | CG4BDD1371B6 | challenger |
| 5 | CG4FAF273093 | control |
| 6 | CG4FAF273093 | challenger |

These six slots are independent of the arm's numerical request contents:
control/challenger changes do not select different input paths or hashes.
This record does not derive or qualify arm requests; the exact bundle/profile
and later PRE bind those separately. Every refusal, timeout or missing result
retains its slot. No substitution, reordered favorable prefix, new input
authority, Gold access, scorer execution or subset F1 follows.

The eventual independent exact PRE should compare the frozen bundle's raw17
authority map/digest, ordered selected cases with `reference_id` stripped,
and six-slot order against this independently derived record. A release-only
source revision must be disclosed and its unchanged scope checked explicitly.
This document is not added to `SOURCES`, is not consumed as a mutable selector,
and does not replace the pinned CASES authority. Physical input bytes remain
unverified in this review and must be checked only within a separately released
exact diagnostic invocation. No physical freeze or launch is claimed here.

