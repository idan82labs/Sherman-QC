# Schedule/context projection V2 exact independent PRE

Date: 2026-09-12. Independent AI technical review, not human approval.
Decision: GREEN for one exact invented-byte projection gate after root's full
acceptance, external archive and no-overlap checks. All237 outcomes remain
prospective. This reviewer performed source reads, AST analysis, hashing and
standard-library runtime metadata checks only: no target import, collection,
test, native call, physical input or attempt/private-cache access.

GOLD11_SCHEDULE_PROJECTION_V2_EXACT_PRE_JSON={"decision":"GREEN","expected_outcomes":237,"runtime_binding_sha256":"3065f28fced0b459c575c783852e8fb3a96d7935a0322ade43e9d16ff0d90473","schema":"GOLD11_SCHEDULE_PROJECTION_V2_EXACT_PRE","scope":"INVENTED_ONLY","source_map_sha256":"c771a39e51ec74a23fb20679c1b8202d9f43c32cc451e3c6961920ca39f12677"}

## Exact closure

The complete new candidate, tests, harness, protocol and source review were
read, together with the complete schedule/contact/legacy import sources and
both older constructor test modules. The unchanged V12 supervisor, tests
initializer and three context sources were fully read earlier in this review
chain and rehashed; context policy declarations were independently rechecked
statically. The harness's full delta against the reviewed contextV3 harness
was also inspected. No remaining blocking finding was identified for this
narrow source/import/policy boundary. No source or test edits were made.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| scripts/check_gold11_schedule_projection_v2.py | 24559 | a8e3417586261bda35aef7a19bc7f4d04ae11b5f562930e839941d06cc4c3290 |
| scripts/check_cad_guided_connected_v12.py | 19437 | 0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694 |
| scripts/gold11_schedule_projection_v2.py | 14677 | 728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db |
| scripts/gold11_schedule_projection_v1.py | 17231 | a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245 |
| scripts/gold11_contact_scoring_v2.py | 50413 | 9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff |
| scripts/gold11_mesh_scoring_v1.py | 13492 | aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234 |
| scripts/evaluate_f1_against_manual_gold.py | 13134 | 1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62 |
| scripts/f1_physical_evaluation_contract_v2.py | 8329 | 28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7 |
| tests/test_gold11_schedule_projection_v2.py | 26090 | 61014e1500742ed40d16d7494bbeccae93e4c2ce722317e676d7123881653a76 |
| tests/test_gold11_schedule_projection_v1.py | 26868 | 8d3113d8017d3e50e096696563daf38c4f62deacb87ca552263a6adcac8faf72 |
| tests/test_gold11_contact_scoring_v2.py | 38711 | f9576b9d4b7267565657ee18a4b344e150a48da9c86f9bf5da2259db3d9fe2cb |
| tests/__init__.py | 24 | 5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b |
| backend/cad_guided_two_plane_context_v1.py | 36075 | 9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a |
| backend/cad_guided_two_plane_context_v2.py | 11130 | d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663 |
| backend/cad_guided_two_plane_context_v3.py | 25845 | 9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d |
| docs/research/2026-09-12_schedule_projection_v2_source_review.md | 14918 | 96b795eea31fa0a2bf272be134ace3676afa4610058e4295b34d9a47450b617d |
| docs/research/2026-09-12_schedule_projection_v2_source_protocol.md | 13110 | 00f8ac840451d1fcc574a834898e5da4cabbad7a9ca4f81212997ca6a5e410ae |

Exactly17 unique sources, total354043 bytes. All15 hardcoded pins match.
Canonical sorted-key compact JSON of the relative-path-to-SHA map, without
newline, hashes to `c771a39e51ec74a23fb20679c1b8202d9f43c32cc451e3c6961920ca39f12677`.
PRE is external to this map and must itself be bound by root's accepted exact
SHA. The root initializer remains absent under direct lstat inspection.

## Exact runtime metadata

Collected with application Python-I-S-B using only standard-library metadata
discovery and initializer-byte hashing; NumPy/SciPy/pytest were not imported.

```json
{"distribution_versions":{"numpy":"2.4.6","pytest":"9.1.1","scipy":"1.17.1"},"executable":"/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11","executable_stat":[16777231,17404606,17257904,1783888533314804818,1783888533315379399],"implementation":"cpython","machine":"arm64","package_initializers":{"numpy":{"bytes":26277,"sha256":"92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a"},"pytest":{"bytes":5644,"sha256":"ee37885c9583f843390684dfb496907f87a0905128e7a3cf4ad227f49ff34c2d"},"scipy":{"bytes":4063,"sha256":"335bc6e0a9909dc7534f9569a3685a92dc8001cb8c63a6da4c239849ff02d4d0"}},"platform":"darwin","python_version":"3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]","scope":"ORDINARY_LOADER_ENTRYPOINTS_METADATA_NOT_FULL_RUNTIME_BINARY_CLOSURE"}
```

Canonical sorted-key compact JSON, allow_nan=False/default ensure_ascii=True,
UTF-8 without newline gives
`3065f28fced0b459c575c783852e8fb3a96d7935a0322ade43e9d16ff0d90473`.
Executable-stat order is device/inode/bytes/mtime_ns/ctime_ns. This is ordinary
loader entrypoint/version evidence, not a transitive extension/shared-library/
bytecode closure. No installation or runtime change is authorized.

## Source, import and policy assessment

V2 passes the same original RESULT bytes unchanged into frozen scheduleV1 and
contactV2, then independently decodes and checks the context publication. It
does not rewrite a workflow schema, clone backend functions, recompute raw
support or geometry, or relabel contact-sector carriers as inner arcs. V14
workflow implementation identity is mandatory alongside the compatible V13
wire schema. Both declared successor arms use contextV3; the explicit scope
does not claim historical paired-control equivalence.

The request has exactly12 keys,23 exact integer budgets and eight closed
arithmetic/scope declarations. Bool/float budget aliases fail; chunk_rows>=3
and queries<=3*requested signed families remain mandatory. This reviewer
independently extracted literal V1/V2/V3 dataclass defaults and maxima, verified
the V2 override AST, and reconciled all23 effective entries with the independent
test map. No backend policy function was imported or evaluated. The public
request accepts valid submaximal policies without replacing them by defaults.

Completed workflows need matching context configuration/digest/initializer
identity at workflow, policies, registration evidence and nested initialization,
plus all selected binding records. The inherited schedule layer authenticates
the context denominator. A completed nonempty schedule needs a COMPLETE helper
with matching budgets and false registration/finite-support/production authority.
The exact three-key empty-schedule sentinel is accepted only jointly with a
complete typed empty schedule, enabled lane and exact integer-zero ordered
matches/slots. An early refusal may lack returned configuration; its presence
map stays false while the caller request stays separate. Present helpers are
closed to consistent COMPLETE/UNAVAILABLE states and false authority. Later
refusals may retain a previously complete helper. These source-review repairs
are present in the final candidate and do not alter frozen dependencies.

Caller mistakes raise before case attribution. Malformed returned evidence
clears contact, context and schedule output together. Strict JSON rejects
duplicate keys, nonfinite values, malformed UTF-8 and depth/node violations.
Parent reconstruction binds both exact byte streams, compares the complete
canonical rebuilt projection and exposes only the unchanged admitted contact
projection or the existing six-key zero-prediction supervisor refusal. Physical
observed count stays unknown and production_use_allowed stays False.

The test creates a private inert scripts package and imports six pure modules:
scheduleV2/V1, contactV2, meshV1, evaluator and metric contract. The two old test
modules are constructor imports only; no old fixture or old test function is
invoked/collected. The evaluator brings NumPy and SciPy.optimize into the loader
closure, but its matcher, filesystem evaluator and main are not called by the
new tests. Its legacy physical-path strings are inert source literals, not
permission to read those paths. The three backend context files enter only as
named static source AST inputs; no backend code or raw-extrema implementation
is imported by this gate.

The harness injects an inert tests package, suppresses tests/__init__.py and
bypasses scripts/__init__.py. The actual target alone is collected under
importlib mode with no conftest, project config, plugin autoload or pytest cache.
The worker tripwire rejects OCP/Open3D/backend import and backend/root initializer
execution; final loaded-module checks reinforce that boundary. NumPy/SciPy are
explicitly allowed, not native-geometry permission. This is a reviewed-source
guard, not arbitrary-code security isolation. No child-producing fixture,
scorer invocation, raw workflow or corpus access appears on the reviewed path.

## Independent prospective denominator

Static cardinality counting of parametrization literals/named literal maps,
without evaluating test code or collecting pytest, yields20 functions and237
outcomes. In source order the counts are
1,8,1,4,4,4,10,28,24,7,69,10,16,10,16,5,1,10,8,1; their sum is237.
Key controls cover all four parent publications; every first/middle/last array
binding field;23 budgets times three invalid types/ranges; all eight declarations
at caller/helper; exact sentinel/helper states; early/later refusal; both intended
arms plus cross-schedule controls; exact original-byte identity; malformed bytes;
retained reconstruction mutations; and whole-result failure clearing.

Fixtures contain two invented contact carriers and three selected contexts,
with two small ordered-pair metadata rows or the exact empty sentinel. Ordinary
serialized workflow/projection fixtures are asserted below102400 bytes. A
delegating spy checks original-byte object identity, and a separate actual inner
projection call checks unchanged output. No physical result or matcher score is
computed by the new tests. Large public input/wrapper caps are not exercised:
V2 adds a third decode, and parser allocation precedes node/depth checking.
This small gate does not qualify128MiB inputs,16MiB wrappers or worst-case memory.

## Fixed execution, custody and terminal rule

One serial worker only:20 CPU seconds,25 work seconds,30 total supervision
seconds including cleanup,536870912-byte sampled group RSS and1048576 output
bytes. Source reads are capped2097152 bytes per named file; administrative
publications131072 bytes each. Thread pools are fixed to one. The broader16GiB
workspace ceiling and no-overlap rule cannot relax this stricter gate. Parent
source/runtime/PRE/archive/publication I/O is outside the30-second supervisor
clock; sampling and clock checks are not hard OS allocation or I/O cancellation.

The parent extracts only unchanged V12 _remaining, _group_sample and
supervise_native_context_test_v3 AST definitions with fixed current constants;
historical imports/main/suite selection are not executed. This ordinary
one-worker wrapper is not the separately qualified native V3 supervisor and
does not establish hostile generation/PGID behavior or arbitrary process trees.

Before execution root must fully read and accept this PRE, hash its exact bytes,
archive all17 sources plus PRE as18 unique regular payloads outside an attempt,
and verify every archive/live byte without extraction/execution. Root must
independently confirm no overlapping heavy writer; this review made no process
observation. Only then invoke once:

```
.venv/bin/python -I -S -B -u scripts/check_gold11_schedule_projection_v2.py --execute --pre-sha256 <root-accepted-exact-PRE-sha256>
```

Exact PRE validation precedes attempt creation. There is no suite/subset/cap
override or retry. External one-shot custody remains mandatory. The unique
attempt retains only STARTED.json, OUTPUT.txt and COMPLETED.json. Source,
runtime, request and PRE are checked before and after; actual loaded package
paths/version/initializer hashes must match. Output/postflight reads occur only
after confirmed empty writers. Missing receipt or unknown cleanup does not
mean no invocation occurred and cannot authorize a retry or favorable subset.

PASS requires237 unique nodes and237 passing setup/call/teardown triplets;
zero failed/skipped/xfail/xpass/error/deselection/collection errors, no duplicate
or invalid reports, worker/parent exit0, no supervision/cleanup/postflight/
publication error, confirmed empty writers and unchanged source/runtime/PRE.
Parent authenticates output against the supervisor hash and exactly one worker
marker. Exact final three-file namespace and content hashes plus actual final
stdout publication gate are mandatory; provisional COMPLETED success alone
is insufficient. Unexpected names veto before their content is read.

Root must separately confirm terminal parent/writer custody and release only
the exact terminal attempt for independent POST. POST must reconcile all
outcomes, resource receipts, archive/live/runtime/PRE and final stdout/inventory
without rebuilding fixtures or rerunning tests. Private worker cache stays
retained and unread. Any failure remains terminal; no reduced fixtures, raised
caps, overwritten receipts or new invocation from this protocol follows it.

GREEN here is technical readiness for that exact small metadata gate, not
measured success. Only an accepted actual POST can qualify the projection seam
for separately reviewed runner integration. No raw-support, schedule execution,
registration, metrology, physical F1, production authority or engine completion
claim follows. V5 and all other terminal failures remain preserved; this PRE
grants no raw/Gold/CAD/held-out/email/private-cache access or physical successor
launch authority.
