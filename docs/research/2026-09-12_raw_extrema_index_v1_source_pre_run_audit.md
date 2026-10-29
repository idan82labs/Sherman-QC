# Raw-extrema index V1 — exact independent PRE

Date: 2026-09-12. Reviewer: Codex AI technical reviewer
`/root/cad_blocker_closure_review`, independent of helper, test and harness
authors. Decision: **GREEN for the one invented-only qualification invocation
specified below, conditional on the external prelaunch requirements.**
Blocking source findings: none. This is technical AI review, not human approval,
an executed test result, engine integration, physical evaluation or new F1.

RAW_EXTREMA_INDEX_V1_EXACT_PRE_JSON={"schema":"RAW_EXTREMA_INDEX_V1_EXACT_PRE","decision":"GREEN","scope":"INVENTED_ONLY","expected_outcomes":88,"source_map_sha256":"3fecbf402fff1e5becd3d9fbf30035cae2df4b48e818550359efc2b448384fed","runtime_binding_sha256":"a767bda3636799a4c226140f9f0d1a1e714620ea3ecd88c27d49a274f52fe976"}

## Exact source ledger

All seven named files were independently read as bounded regular source files,
hashed and reconciled to the final declaration. The helper, complete test module,
math/source review, complete final harness/protocol and the three selected
supervisor functions were read. No target/harness import, pytest collection,
test, numerical simulation, CAD/native API or process signal was executed.
No physical inputs, Gold, prior attempt/scientific artifacts, control records or
private caches were read for this PRE.

| Relative source path | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/check_raw_extrema_index_v1.py` | 22265 | `b26c9fc1218eeaaadd854f19e6802cd0c48b94c24f319661cd29921c0eaffaf8` |
| `scripts/check_cad_guided_connected_v12.py` | 19437 | `0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694` |
| `backend/cad_guided_raw_extrema_index_v1.py` | 19626 | `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121` |
| `tests/test_cad_guided_raw_extrema_index_v1.py` | 31028 | `e93844cfed8526f64dffc184ceb2aa0f952c717b98dbb3e7db3ec43c812a598c` |
| `tests/__init__.py` | 24 | `5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b` |
| `docs/research/2026-09-12_raw_extrema_index_v1_source_review.md` | 14381 | `69103da53991da114f965923ad6b572e6c37cdf12a1ea36bd8f559f1f4ee4f83` |
| `docs/research/2026-09-12_raw_extrema_index_v1_source_protocol.md` | 10719 | `2cca0f2e9b78d8819da8256fc9011b7d23d1ed3d80a526c1800ccd6f6691b707` |

Total: **7 files / 117480 bytes**. Canonical source-map SHA256:
`3fecbf402fff1e5becd3d9fbf30035cae2df4b48e818550359efc2b448384fed`.
Encoding is sorted-key compact JSON mapping every relative path to its SHA256,
without a trailing newline. Five files are Python source and two are Markdown.
The exact repository-root `__init__.py` was confirmed absent. The 24-byte tests
initializer is only the comment `# Sherman QC Test Suite` plus newline; it is
bound conservatively and its execution is suppressed by the synthetic namespace.

## Exact runtime binding

Independently reconstructed with stdlib metadata and bounded file reads only;
NumPy and pytest were not imported. Canonical full-runtime-object SHA256:
`a767bda3636799a4c226140f9f0d1a1e714620ea3ecd88c27d49a274f52fe976`.
Its encoding is the same sorted-key compact JSON convention as above.

| Runtime field | Exact value |
| --- | --- |
| Python | `3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]` |
| Implementation / platform / machine | `cpython` / `darwin` / `arm64` |
| Resolved executable | `/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11` |
| Executable stat: device, inode, size, mtime_ns, ctime_ns | `[16777231,17404606,17257904,1783888533314804818,1783888533315379399]` |
| NumPy distribution | `2.4.6` |
| NumPy initializer | 26277 bytes; `92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a` |
| pytest distribution | `9.1.1` |
| pytest initializer | 5644 bytes; `ee37885c9583f843390684dfb496907f87a0905128e7a3cf4ad227f49ff34c2d` |
| Scope | `ORDINARY_LOADER_ENTRYPOINTS_METADATA_NOT_FULL_RUNTIME_BINARY_CLOSURE` |

The initializer paths are respectively
`/Users/idantal/ShermanQC/FullProject/.venv/lib/python3.11/site-packages/numpy/__init__.py`
and
`/Users/idantal/ShermanQC/FullProject/.venv/lib/python3.11/site-packages/pytest/__init__.py`.
Each distribution is uniquely resolved from that exact site-packages directory.
The application interpreter symlink resolves to the executable above.

## Import, supervision and custody review

The harness constructs only the following three AST function definitions from
the exact V12 source. No historical imports, constants, source lists, fixture
selection or main body are evaluated.

| Function | Source lines | SHA256 of location-free AST dump |
| --- | --- | --- |
| `_remaining` | 22–26 | `04573e9535d55db9bd5ea554f0c4c6c581c7e758d07b53e9042a35c92e0815de` |
| `_group_sample` | 29–38 | `7c38a61e6343c6f28fed61ba4086513b9f204fda3d8ee3b041d0a21dfa374ada` |
| `supervise_native_context_test_v3` | 41–144 | `d4cdba30659e3765c83df5134aee718a6019f80588548cab43deb72f73afdd2d` |

Their namespace supplies Path, tempfile, time, os, subprocess, selectors,
signal, hashlib and this gate's exact CPU/work/wall/RSS/output constants.
Their selected bodies have no other project dependency. This is the existing
short single-worker supervisor, not Gold11 V1/V2 supervision. It rejects an
observed additional worker child and uses sampled process-group lifecycle
checks. It is not generation-aware hostile-process containment. Its monitor
subprocess is separate from the worker group; monitor/parent cost is not worker
CPU or sampled worker-group RSS.

Parent admission verifies all sources, runtime and the unique exact PRE marker
before creating an attempt. The worker is dispatched by the same interpreter
with `-I -S -B -u`, a hash-bound STARTED request, isolated HOME/TMPDIR/cache,
single-thread numeric environment and disabled pytest plugin autoload. Its CPU
rlimit is set before request/package loading. A fresh synthetic `tests`
namespace plus pytest importlib mode suppress package-initializer execution.
The helper loads under a private module name by exact-file loader, without a
backend package import. No fixture reads a dataset or launches a child.

The worker prohibits OCP/Open3D/SciPy imports and backend/root initializer
execution, verifies actual NumPy/pytest entrypoint paths and versions, and
checks forbidden loaded modules after pytest. It compares all sources, runtime
and PRE before/after. Normal Python loaders and trusted, quiescent checkout
semantics remain part of the boundary: `-B` prevents bytecode writes, not all
possible bytecode reads; this is not a full transitive binary/bytecode closure,
adversarial filesystem sandbox or atomic multi-file snapshot. Canonicalizing
only the newly owned cache accommodates macOS path aliases without relaxing
source/input custody.

Stable reads require regular final components, no-follow/nonblocking opens,
bounded bytes and matching descriptor/path identities. Source/PRE reads are
capped at2MiB each; admin records at128KiB. After failure with confirmed empty
writers, parent independently attempts source/runtime/PRE checks even if the
worker ledger is missing. Without quiescence it does not parse or credit output.
The inherited supervisor may hash its own bounded output after failed cleanup;
that does not grant the parent permission to parse it or establish success.
Missing/empty/malformed output, duplicate receipt markers, drift, unexpected
namespace, uncertain cleanup or failed final publication cannot pass.

## Fixed denominator and mathematical acceptance

The independently counted final source has **31 test functions / 88 prospective
outcomes**. Parametrization expands the declared literal tuples and the 12-case
ROUNDING_CASES tuple; none were collected or executed during this audit.
There are no skip/xfail decorators or dependency on another test fixture module.

The observer requires exactly88 unique nodes from the one named test file,
with all three setup/call/teardown phases passed for every node. It requires
zero failures, errors, skips, xfail, xpass, deselections, collection errors,
duplicate/unrecognized reports and missing phases. Xfail detection uses marker
presence, including an empty reason. Worker exit zero alone is insufficient.

The independent scalar oracle reproduces separate binary64 subtraction,
multiplication and ordered addition using individual Python-float operations,
then canonicalizes zero. It does not call the helper's projection or BLAS.
Node/member enclosure, extrema and smallest original raw-ID ties are checked.
Direction-sign corners and the conjunction of minimum/maximum no-improvement
conditions preserve every potentially improving raw witness. This is an
enclosure of the specified rounded program under its stated arithmetic
premises, not an exact-real interval, old-BLAS equivalence or uncertainty bound.
Sentinels and focused rounding controls do not attest every binary/vector path.

The controls also cover anticorrelated loose boxes requiring all-row work,
cross-leaf ties, one-sided improvement, complete supplied support, duplicates
without welding, readonly/strided layouts, caller mutation, captured metadata,
strict admission, attempted/completed work, exact/one-below budgets, late whole
clearing and report/interrupt failures. Success never claims accepted
registration, observed bend count or production permission.

The one900000-row capacity fixture has a fixed bijective reverse modular
population and three axis queries only. The independent analytic smallest-ID
witnesses are0/1,0/41,0/13; whole XYZ/support hashes and population are checked.
The pass-through query spy records constructor counters without another build
or altered query result. Each query is allowed at most519 evaluated rows,
total1557; this is specific to its axis bounds and raw-ID arrangement.
Construction still charges2.7M row visits,4.5M index visits and18M sort units.
It emits one bounded8KiB capacity JSON line inside OUTPUT, not an additional
scientific file.

The fixture's two7.2MB integer arrays are deleted before helper construction;
21.6MB caller XYZ remains. Modeled helper peak86465536 bytes plus that caller
buffer is108065536 bytes, before interpreter/NumPy/pytest and allocator costs.
This supports a plausible bounded run under the fixed512MiB gate, not measured
RSS or a completion guarantee. Remaining fixtures are at most1025 rows, except
one8192-row plate. No fixture/cap may change after the attempt starts.

## Exact invocation and external prelaunch conditions

Freeze one serial worker at **20 CPU / 25 work / 30 supervision seconds,
536870912 bytes sampled group RSS and1048576 bytes output**. The last five
supervision seconds are the cleanup reserve. Repository16GiB and no-heavy-
overlap restrictions also remain binding. Thread pools are fixed to one.
Parent pre/postflight, archive, publication and synchronous I/O are outside the
30-second supervisor clock; no hard OS I/O-deadline claim is made.

Before launch, root must fully read/accept this final PRE; archive exactly the
seven source payloads plus these PRE bytes as eight unique regular payloads
outside any attempt; verify the complete archive and its SHA256; and immediately
recheck the source/runtime/PRE bindings, absent root initializer and absence of
V5/other owned heavy writers. This audit does not claim those future archive
and process checks have already occurred. No physical or cache read is needed.

From `/Users/idantal/ShermanQC/FullProject`, the sole permitted invocation is:

```
.venv/bin/python -I -S -B -u scripts/check_raw_extrema_index_v1.py --execute --pre-sha256 <SHA256-of-these-exact-PRE-bytes>
```

The supplied hash must be the final externally computed hash, not the placeholder.
Do not invoke hidden worker mode directly. External one-shot custody is
mandatory: no retry, suite/limit override, reduced fixture or reopened terminal
attempt. This PRE does not release the union-planner or supervisor-V2 gate.

## Terminal decision and mandatory independent POST

One new unique attempt retains exactly STARTED.json, OUTPUT.txt and
COMPLETED.json. PASS additionally requires zero exit/budget/primary/cleanup/
postflight error, confirmed empty owned writers, unchanged source/runtime/PRE,
one valid worker receipt and exact final three-name/byte/hash inventory. The
parent's final stdout and exit must confirm final publication; an intermediate
COMPLETED outcome flag is not sufficient.

Only after terminal/quiescent evidence is provided may an independent POST
read the exact bounded attempt and archived sources, reconcile all88 outcomes
and full OUTPUT (including the capacity summary), resources, process closure,
three-file inventory and live/archive/runtime/PRE consistency. Preserve every
negative or incomplete outcome and do not rerun. Any qualifying result is
standalone helper evidence only; explicit context integration, aggregate
registry/memory budgets, connected tests and physical scoring remain separate.
V5 stays terminal and unscored, and the roadmap is not closed by this gate.

