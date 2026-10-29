# V6 fair plane-search schedule: independent source test protocol

Date: 2026-09-09. Prospective source-only gate, not an execution result.

## Scope and frozen decision

Test file: `tests/test_cad_guided_plane_search_v6.py`.
Target: `backend/cad_guided_plane_search_v6.py`; target revision at initial review
is `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676`.
The root owns target source, its mechanism protocol and the execution harness.
The test author owns only this protocol and the new test file. The original V5
source, all 77 V5 tests and every prior terminal attempt remain unchanged.

The fixed gate contains **115 pytest cases: 77 inherited plus 38 V6 controls**.
Every case must pass. A collection/count mismatch, error, skip, timeout, memory
stop, incomplete candidate ledger or source drift is not a passing gate. Retain
the complete failure and its source closure; do not loosen a geometric assertion,
drop a case or retry a terminal attempt under this protocol.

Only invented arrays/declared plane equations, exact named source files and
small explicitly mocked failure seams are inputs. There is no retained artifact,
native CAD, mesh-file reader, physical part, Gold, held-out, email, filesystem
discovery or application access in the tests. The root's guarded source harness
must install the isolated backend namespace before loading this test, so the
`from backend` import does not initialize the application package. No test or
source import was executed by the author during preparation.

## Preservation of the original evidence contract

The inherited test-function ASTs are mechanically identical after only V5-to-V6
type/module/API renaming and renaming the four quota tests to describe the new
schedule. An executable source comparison enforces this. The only replaced old
helper bodies are `_strata` and `_reconstruct_join`, because the selected ranks
are the explicitly changed mechanism, not a geometric acceptance threshold.

Previously, one slot selected the middle and multiple slots selected endpoint
strata. V6 instead cycles one rank per disjoint integer bin. Thus a six-row layout
at occurrence zero has quota-one rank `[0]`, not `[3]`; quota four gives
`[0, 1, 3, 4]`, not `[0, 1, 3, 5]`. The quota test still requires all eight signs
for each selected unsigned assignment, and the orthogonal control still requires
four unsigned rows, 32 sign rows and 16 proper alternatives, not 24 enumerated
orientations. The 30-CAD control retains its original 435 pair-index records,
300-rank window, 12 searched pairs, 240 compatible unsigned rows, 236 unsigned
omissions and 32 signed records.

All original pose and negative criteria remain unchanged: independently posed
nonorthogonal triads at scales 0.001, 1 and 1000; rotation absolute tolerance
2e-12 and original translation/action tolerances 2e-8 mm; alternate anchors,
unmatched support centroids, coefficient normalization, reflection signs,
independent frames, imperfect congruence and the heterogeneous-cone nominal
Wahba counterexample. Support/rank/cone/backward-residual refusals, complete
sign accounting, all hard-cap/type/unit checks and late SVD/report failures
are preserved. Passing abstention cannot replace a required proper-pose result.

The added geometry-source comparison requires unchanged bank, normal-angle,
interval, intersection, partner-triplet, pair-index, signed-pose and failure
helpers. The exact policy defaults remain 3 support rows, 0.1 mm RMS, 0.02
condition ratio, 0.02 radians extra alignment, 1e-8 mm intersection backward
residual and 1e-10 radians arithmetic angle guard. None is calibrated here.

## Independent schedule oracle and proof being tested

For an exact ordered list of length L and positive m=min(L, quota), define
`a_j=floor(jL/m)`, `b_j=floor((j+1)L/m)` and the integer bin `[a_j,b_j)`.
Occurrence k selects `a_j + (k mod (b_j-a_j))` from each bin. Distinct nonempty
bins imply exactly m unique in-range ranks. Each bin completes a cycle in its
length, so every rank is exposed within any K consecutive occurrences of that
same identity, where `K=max_j(b_j-a_j)=ceil(L/m)`. Within each bin, rank exposure
counts differ by at most one for a prefix of any length. Unequal bins do not
imply perfectly equal exposure counts across the entire list. Empty lists and
zero quotas have no finite coverage certificate.

`_reconstruct_join` independently rebuilds conservative pair windows from scalar
acos-based normal angles, every exact pair/third-plane interval test and both
little-bit rejection masks. Its first occurrence map is keyed by exact `(start,
stop)` in the one canonical immutable CAD pair table. Its second map uses
`struct.pack('<iii', ...)` for each reconstructed compatible ordered CAD-index
triple, concatenated in the original order. It never calls a runtime scheduler,
pair kernel or returned phase to decide the expected rank. It verifies selected
IDs, phases, diagnostic SHA, layout indices, logical byte lengths and every
searched/omitted/hashing-work denominator.

## Added 38-case matrix

- Two source-preservation/policy/workload checks.
- One bounded exhaustive arithmetic check over lengths 0..65 and quotas 0..12;
  up to three maximum-bin cycles, every sliding K-window, uniqueness, rank
  bounds and within-bin prefix fairness. This is not a geometry sweep.
- Eighteen exact-integer negative cases (three arguments crossed with six invalid
  values) and one near-int64-limit arithmetic case using only four output ranks.
- One interleaved pair-identity/state-reset check; one detached ordered-layout
  check covering reordered triples, one changed ID, multiplicity, endian
  canonicalization and ignored capacity tail; one deliberately colliding
  diagnostic-SHA check. Exact byte keys must remain distinct even when their
  diagnostic digests collide, and each new key is hashed only once.
- Three empty/zero-quota checks with no false coverage certificate.
- Five direct boundary checks: pair registry, assignment registry, single-key
  bytes before copying, total-key bytes before new hash/publication, and layout
  shape accounting. Smaller injected limits test refusal semantics, not wider
  production limits. A repeated existing key remains admissible at an exact cap.
- Three explicitly injected late public scheduler failures: pair/runtime,
  assignment/runtime and assignment/budget. Real prior sign/pose work must be
  retained as attempted but every usable pose invalidated; active triplet and
  CAD participation ledgers cannot silently reset. The budget arithmetic itself
  is separately tested without injection above.
- One real repeated-plane-table search: nine exact-axis scan declarations and
  six repeated CAD declarations translated by (7,-11,13) mm. Both identity maps
  must recur, all selected rows must finish, some real proper hypotheses must
  exist, and every proposed alternative must map the independently constructed
  common intersection within 1e-8 mm. Input tuple reversal gives the same result.
  Exact axis ties avoid dependence on differing sub-ULP acos/atan2 tie orders;
  nonidentity rotation remains required by the unchanged original positives.
- One explicit logical counterexample: every list rank can be exposed across
  occurrences while each occurrence's distinct sole correct correspondence is
  omitted. This prevents interpreting scheduling fairness as completeness.

## Bounds, failure handling and permitted conclusion

First gate is one serial guarded job: **60 seconds wall, 45 seconds worker CPU,
1 GiB process-tree RSS, 1 MiB stdout**, enforced by the root-owned supervisor.
No native/physical execution is qualified by this source protocol. The bounded
integer check stores at most 196 small rank lists at once. The real repeated
control uses 9/6 plane records; no large-key or full-size geometry allocation is
performed. Source AST checks read only the four named V5/V6 source/test files.

Runtime limits remain 12 CAD pairs and four unsigned assignments per scan
triplet, 256 triplets, 1,024 unsigned and 8,192 signed rows. Each registry has at
most 256 entries. The maximum canonical layout-key payload is
`256*49104*12=150,847,488` bytes, below the fixed 192 MiB logical key cap. This
counts immutable canonical key payload, not Python containers, transient copies
or total RSS. The later full-size/native scope needs its separately frozen
process budget and capacity evidence; this small 1 GiB gate does not prove it.

Fair exposure at the conservative-window stage does not establish actual pair
compatibility on each occurrence. Fair exposure of a compatible layout is across
different scan triplets, not all CAD assignments for each one. The two stages
can generate changing layouts seen only once. Bank/triplet selection and later
finite-screen/refined-start quotas remain additional omissions. Therefore all
outputs remain nonexhaustive, unverified-support initialization hypotheses, never
accepted registration, calibrated uncertainty, global uniqueness, observed
absence, known bend counts or physical count/F1 improvement. A passing pure gate
only supports this declared scheduling/math/accounting contract.
