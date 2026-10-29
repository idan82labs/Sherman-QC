# Two-plane context V1: independent exact pre-run audit

Date: 2026-09-10. Decision: **GREEN_FOR_ONE_FROZEN_PURE_TWO_PLANE_CONTEXT_V1_GATE**.
Blocking findings: **none within the declared pure-helper scope**.
Execution status at this audit: **UNEXECUTED**. This is permission for the one
command below, not a test result, connected-engine approval or physical release.

Reviewer: `/root/production_architecture_review`, an independent AI reviewer of
the implemented helper, its independently authored test, and root-authored
protocol/harness route. The reviewer contributed the earlier design document;
this is not an independent certification of its own design authorship. Root also
reports a separate author-independent full helper/math review by
`/root/cad_measurement_integration_review`. This decision follows the present
reviewer's full source/test/protocol read and exact release-contract checks; it
does not manufacture a human scientific or manufacturing approval.

## Exact five-file release

All hashes below were independently read from current source bytes and checked
again after review. These five files, and only these five, must appear in this
route's before/after source maps. The pre-audit itself is a separate release
record, not an extra runtime source-map entry.

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `backend/cad_guided_two_plane_context_v1.py` | 36075 | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `tests/test_cad_guided_two_plane_context_v1.py` | 36194 | `16a7e4467007f3a6c3034279c7b34221c2ccd26b51f7472c47ade6a8546cf5d1` |
| `scripts/check_cad_guided_measurement.py` | 79174 | `628f3a84c98806f1d24c7b47f8d7683911a2f8bfb7f67dc8698887f1d8eda0f4` |
| `docs/research/2026-09-10_two_plane_context_proposal_design.md` | 15252 | `f926605558aa31d87ffdcc527964b296cd5d8d7d75cc2bec1043a0af7ddde722` |
| `docs/research/2026-09-10_two_plane_context_v1_protocol.md` | 5685 | `0e4929cf3c88d6b0071ae3cb6873434483eca7ddb67b918844f164d819446356` |

The previous harness is preserved at
`docs/research/source_snapshots/two_plane_context_pre_eae849f8/check_cad_guided_measurement.py`,
SHA-256 `eae849f8da3930913290c02d0aa41ef49c0f50ffd1cc9a93f51b847ac66f0c83`.
An exact textual diff shows four additive route changes only: the named test
selection, three companion source bindings, membership in the 60/45/1-GiB budget
group, and membership in retained-receipt creation. The existing supervisor,
worker dispatch, output loop, source hashing and terminal-receipt code are
unchanged. No frozen V3, V10, cached-control or prior attempt file was edited.

## Imports, command and retained execution contract

The 628-line helper imports only `contextlib`, `dataclasses`, `hashlib`,
`itertools`, `json`, `math` and NumPy. It has no file, native, subprocess, server,
application, scoring or physical-input import. Its default budget construction
does not access the filesystem. The 610-line test imports stdlib, NumPy and
pytest, constructs invented arrays, and loads exactly the named helper with
`spec_from_file_location`; it does not import the backend package, old native
tests or physical fixtures (`tests/...:34`). There is no subprocess or network
call in helper/test source. Installed Python/NumPy/pytest dependencies are the
existing execution environment, not a byte-attested environment closure.

After this exact pre-GREEN, root may execute once:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_context_v1
```

Do not use `--suite all`, invoke `--worker` directly, add a second test route or
change any source/budget after STARTED. The named route selects one exact test
file (`scripts/check_cad_guided_measurement.py:141`). Its source map adds helper,
design and protocol to the selected test and harness (`:490,806`). The worker
sets RLIMIT_CPU before importing pytest, adds the existing Python 3.11
site-packages path, and uses `-c /dev/null --noconftest -p no:cacheprovider -q -s`
with plugin autoload disabled (`:860,957`). This route enables no native/smoke
flag and does not add the backend package namespace to the worker path.

The admitted limits are 60 seconds monitored wall, 45 seconds worker CPU,
1,073,741,824 bytes sampled process-group RSS and 1,048,576 bytes captured
output. The parent creates private HOME/TMPDIR/XDG cache and fixes all listed
numeric thread controls to one (`:971,1098`). One fresh pytest process is launched
in an owned new session. The test itself creates no children. The gate is well
below the repository's 16-GiB ceiling; no full-cap performance claim follows.

The fresh retained directory is under
`docs/research/cad_engine_v1_synthetic_checks/two_plane_context_v1-*`.
STARTED uses exclusive creation and records the suite, limits, test/harness
hashes and exact five-source map. OUTPUT is exclusively created and its retained
bytes are SHA-bound by COMPLETED. COMPLETED retains child exit, monitored wall,
sampled RSS, budget-stop disposition and before/after source maps
(`scripts/check_cad_guided_measurement.py:1006,1074,1119,1216`). This route does
not publish per-fixture geometry artifacts or use V3's terminal-fill helper.
No such additional retention is promised by the pure protocol.

The unchanged legacy supervisor is not perfect containment: its timer excludes
some setup, source hashing, receipt IO and cleanup; output forwarding and reap
are not independently hard-deadlined; RSS is sampled rather than an allocation
cap or continuous-peak guarantee. It does not establish escaped-native-child
custody. These limits are compatible with this reviewed, trusted, no-subprocess
pure test, not approval for a native/physical route. Missing/failed terminal IO,
missing worker resource evidence or a CPU kill is not a pass even when an outer
stop field is null. Parent exit zero alone cannot release the gate.

## Mathematical and custody checks

The rank-two family in `backend/cad_guided_two_plane_context_v1.py:355,387`
checks both two-row normal systems, the nominal .02 conditioning ratio and a
positive conditional cone margin using the Frobenius perturbation bound. It
constructs right-handed central frames, checks determinant +1/orthogonality and
both normal residuals, then uses a checked two-row solve for minimum-norm `t0`.
The two plane equations constrain only two translation components. The retained
`u` is the intersection direction; `t0 + lambda*u` retains its free component.
Unequal noisy included angles produce central seeds only, not an uncertainty
enclosure or exact correspondence guarantee.

The source witness is `anchor + offset*normal`; target signs multiply both the
target normal and corresponding anchored equation. Normal normalization also
normalizes the offset. Source projections use `R.T*u`, source origin and
`u dot t0`; target projections use `u` and target origin (`:433,486`). Thus an
origin shift changes the axial vote without changing physical pose action.
The independent regression explicitly requires lambda=10, not zero, for its
shifted-origin identity fixture (`tests/...:160`). These are engineering
binary64 calculations, not outward-rounded certificates.

Both complete assigned scan supports and complete CAD-face membership are
bound before any pair proposal. The scan raw union is recomputed from owned
triangle indices; CAD row membership must equal the entire selected face-ID
membership. Arrays are copied into immutable byte-backed snapshots with captured
shape, checked dtype/size, full-buffer finite validation and digests
(`backend/...:169,232,282`). Whole-buffer extrema include rows outside qualified
plane cores; no third normal is fabricated from them. Cache identity includes
role, frame, buffer digest, support digest, projection direction, origin and
constant; witnesses retain original row/triangle/local-vertex identities.

All four target sign branches and eight named slots are retained per supplied
ordered assignment. Extrema, endpoint and midpoint votes remain explicitly
heuristic. Exact pose deduplication retains every contributing slot and does
not merge nearby placements by tolerance. Scheduling is caller-supplied and
unauthenticated, not implemented or proved fair by this helper. The 256 ordered
assignment /1024 signed family /8192 slot ceiling is preflight, not proof of
search completeness. Working-byte counters explicitly exclude total Python
object/allocator RSS. Buffer, projection, index, sorting, hash, cache and report
work have separate bounded ledgers. A late malformed/numeric/work/report failure
clears all geometry/candidates and returns only bounded attempted evidence
(`:324,517,531,616`); it cannot release a favorable prefix.

Neither COMPLETE nor usable_proposals means registration. Every result retains
accepted_registration=false, finite_support_match_verified=false, observed count
null, production_use_allowed=false, global uniqueness NOT_ESTABLISHED and axial
context NOT_CERTIFIED. Empty schedules and rejected pairs can complete their
declared evaluation without producing a usable seed. Parallel/cone-ambiguous
systems cannot gain an invented axial rank through extent voting.

## Independently checked test inventory and oracles

Source-only AST parsing, without importing or executing the helper/test/harness,
finds **31 test functions expanding to 110 prospective outcomes**, with no skip,
skipif or xfail markers. Syntax parsing and static parameter counting are not a
pytest collection or test run. The full source was read, including all branches.

The independent suite includes four fixed exact proper poses under sign/order/
origin changes, exact Fraction plane-equation checks, scale changes, independently
changed non-axis rational coordinate frames, and the fixed 1e-8-mm pose-action
ceiling. It constructs CAD context separately from scan intervals; no expected
pose is passed to the helper. Original row permutations, complete extrema
witnesses, chunk sizes 3/6/65536, cache counts and exact submicron deduplication
have explicit assertions. These modest constructs do not prove arbitrary-angle
or full-cap numerical/performance coverage.

Ambiguity controls require unresolved axial sliding, disconnected repeated
context without finite support, empty midpoint slots, and asymmetric partial
coverage that deliberately misses the true pose. Weak-cap coordinates can vote
without inventing a qualified third normal. Rank/cone/angle refusal, malformed
lineage, immutable owned snapshots, declared budget boundaries, empty schedules,
late projection failure, injected later-family exceptions and final report
failure all retain the required unknown/refusal and no usable partial geometry.
The 8192-slot test calls scheduling preflight only. Snapshot-race and exception
injections qualify those mocked boundaries, not arbitrary concurrent mutation,
hostile code, native geometry or whole-part measurement.

## Required post-audit and advancement boundary

Release the result only after a different-author retained-only post-review of
the one fresh STARTED/OUTPUT/COMPLETED set: exact suite and five frozen hashes,
110 passing outcomes, zero skips/xfails/failures, observed successful parent and
worker exits, complete resource evidence within the declared limits, no stop,
no source drift and matching retained output digest. A missing outcome, receipt,
hash binding or budget fact is not inferred from an older gate. Preserve every
failure and partial terminal attempt; no retry or cap increase under this audit.

A successful pure gate permits further connected integration work only. Pair
scheduling, a genuine two-support finite-query adapter, joint triplet/pair quota
accounting, full registration observability, ambiguity/refusal, native feature
custody and unchanged scan-derived free-circle measurement remain unqualified.
The original exact-.05-mm thin-part 0/2 negative is not closed, and the terminal
Gold11 V3 custody-blocked zero-output observations are unchanged. No actual
physical/cached input read, physical successor, F1 gain, calibration, unseen-family
generalization or product promotion is released by this document.

Audit operations were limited to named source/released documents, the specified
archived harness, source hashes/diff and stdlib AST metadata checks. No helper or
test import, pytest/native/inference/scorer run, original input access, new
benchmark, source change or existing-attempt mutation occurred. Only this new
pre-run audit document was authored.
