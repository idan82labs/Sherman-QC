# Two-plane scheduler V2: exact source PRE

Date: 2026-09-10. Independent reviewer: `radius_math_review`.

Decision: **GREEN for one isolated, supervised source-gate execution only**, after
root fully reads this PRE and verifies an archive containing the eight exact
sources below plus this PRE (nine named regular payloads). No test has been
executed by this review; no passing, connected, native or physical qualification
is claimed. Any source, fixture, denominator or limit change retires this PRE.

## Scope and exact source ledger

The reviewer read all eight source/protocol files completely, including the
691-line V6 dependency, unchanged V1 scheduler, corrected V2 scheduler, all
563 test lines, supervisor and final harness/protocol. Checks were source reads,
SHA-256/byte/type checks and standard-library AST parsing only. No project or
third-party module import, pytest collection, test run, geometry experiment,
raw/physical input, retained scientific artifact or private-cache access occurred.
Only this new audit document was edited.

| Exact repository-relative source | Bytes | SHA-256 |
| --- | ---: | --- |
| `scripts/check_cad_guided_two_plane_schedule_v2.py` | 5933 | `c6d1d527945bb4e171d5bef42c1ef6b25b683791bcad21343acbaad106601548` |
| `scripts/check_gold11_mesh_v4_custody.py` | 10310 | `579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b` |
| `backend/cad_guided_plane_search_v6.py` | 38804 | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `backend/cad_guided_two_plane_schedule_v1.py` | 26562 | `1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f` |
| `backend/cad_guided_two_plane_schedule_v2.py` | 25030 | `07b24b8e53b08a408f56ab94dc442e7db7f4e3ef6a5cd4729629860cace99977` |
| `tests/test_cad_guided_two_plane_schedule_v2.py` | 30321 | `b5ad5cfcb24b6f74bd0644174a822902829adeed13109a469ade3061322b26ad` |
| `tests/__init__.py` | 24 | `5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b` |
| `docs/research/2026-09-10_two_plane_schedule_v2_source_protocol.md` | 3619 | `ad20fe4628644aa515d305285bb20b1287048ecfa7669954d17e65aef4626916` |

Total: eight regular files, **140603 bytes**, excluding this PRE. All are below
the harness's per-source 2-MiB limit. V1 and the exact supervisor remain unchanged.

The initially proposed seven-file closure was not sufficient: default pytest
package discovery imports `tests/__init__.py`. Its current 24 bytes are only a
comment, but omission would leave an actual repository dependency unfrozen.
This was reported before GREEN; root added it to the harness and protocol.
Read-only inspection of installed pytest's `import_path` and package-resolution
source confirmed the dependency. The final eight-file closure closes that issue.
This is repository-source custody, not an attestation of Python, pytest, NumPy,
compiled libraries, the operating system or all possible external interference.

## Prospective denominator and fixture checks

Independent AST count: **29 test functions, 80 outcomes**. Ten parameterized
functions expand to 61 outcomes; the other 19 functions contribute one each.
In source order the multiplicities are:

`8, 1, 5, 1, 1, 5, 1, 1, 1, 6, 3, 1, 2, 12, 5, 4, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1`.

The only pytest marks are parametrization; there are no skip, xfail or deselection
instructions. The intended gate is **80 PASS and zero other dispositions**, not
merely exit zero. The harness records `EXPECTED_OUTCOMES=80` but does not parse
the pytest summary to enforce it; independent POST must do so.

All inference inputs are invented typed normal/anchor/offset/cone declarations.
The largest bank is six scan equations and eight CAD equations: at most 15 scan
pairs and 28 CAD pairs. The largest expected schedule has 256 ordered records.
Tiny independent SVD comparisons are 2-by-3 systems; no mesh, scan sampling,
registration, fit or finite-support measurement is run by these tests.

Oracles cover complete V1-report equality at zero, odd and ample residual
capacity; the independently counted 21 orthogonal target pairs; exact omitted
rank sets, repeated layouts and partial windows; full 8192-slot accounting;
margin priority, confident distractor starvation, ties and limited permutation
examples; SVD/near-unit admission; malformed declarations and budgets; literal
exact/one-less counter caps; one original CAD index/window pass; snapshotted
budgets; distinct cache bytes with deliberately equal diagnostic digests; and
whole-result clearing on original/residual exceptions, memory refusal, malformed
cache or final-report overflow. The toy 48-group selection example explicitly
demonstrates that additional proposals need not preserve downstream selections.

Literal one-window accounting independently checks out: 19 residual unordered
targets, 76 charged mapping/validation comparisons, 3776 retained-list bytes,
42 report records, 46272 reserved report bytes and 43 final row checks. The
initial-memory refusal oracle is also consistent: 65536 V1 fixed bytes +16384
report reservations +924 numeric bytes +84 layout bytes +65536 V2 fixed bytes
= **148464 bytes**, exceeding the stipulated 131072 cap after complete breadth.
These are prospective source-derived expectations, not executed measurements.

## Math, accounting and import review

No remaining blocking source finding was identified.

- V2's complete breadth pass preserves V1 selection, exact compatible-layout
  clocks, matches, inherited budgets and report accounting. The tests compare
  the entire nested V1 result, not only selected IDs. This equivalence assertion
  remains to be executed under the gate.
- Sorting at most two excluded breadth positions and incrementing an omitted
  rank past each exclusion maps it into the complement without duplication.
  Distinct first fair-bin ranks remain distinct; only selected int32 cache
  entries are decoded. There is no second CAD index or geometric window scan.
- With ordered capacity `C=min(floor((H-h)/32), U-u)`, each selected target pair
  consumes two ordered matches, 64 hypothesis slots and two unsigned slots.
  Odd capacity remains unused. No evaluation outcome is accepted as an input
  and no failed sign, absent slot or duplicate pose refunds reservation.
- For exact unit cone centers, the declared row perturbations imply
  `||E||F <= hypot(2*sin(alpha1/2), 2*sin(alpha2/2))`; consequently a positive
  `sigma_min(A)-||E||F` is a conditional rank-independence margin. Its binary64
  implementation and use as scheduling priority are engineering arithmetic,
  not a certified interval, calibrated scanner model or match probability.
- New charges precede bounded work/records. Maximum priority work is384, pair
  visits64, breadth-position checks128, rank/index selections128, skip checks512
  and final fixed-schema row checks320. Conservative V2-only upper bounds are
  386 report records,745472 reserved report bytes and102400 retained-list bytes,
  plus separately accounted V1/cache/fixed payload. These are not Python RSS.
  Mapping plus validation comparisons are explicitly charged; final-ledger work
  is explicitly a row proxy. Initial residual memory failures preserve attempted
  counters after the correction reviewed in the frozen V2 source.
- Final per-pair omissions and joint reservations reconcile breadth plus depth.
  Unvisited windows retain unknown compatibility counts. Any ordinary fatal
  exception or resource refusal clears all usable breadth and residual matches.
  KeyboardInterrupt is deliberately not relabeled as successful or ordinary
  completed evidence. Additional passing proposals can change 48-start selection;
  complete breadth inclusion is not final-output or availability monotonicity.

The test fixture inserts an inert package and loads only the three named pure
backend modules, in dependency order, within each test. It does not import
`backend/__init__.py`, a server or native engine. Their imports are NumPy and
standard-library modules; module bodies define constants, types and functions,
not file/data access. The now-frozen test initializer is comment-only. Tests
import NumPy/pytest but do not call the fixture or target scheduler at module
definition time. No tests create scientific files or subprocesses.

The harness loads the exact hash-bound supervisor into a non-main module name.
Only its supervision function is called; its old main, source reader, old test
path and runner dependencies are not invoked. The worker uses the explicit
named test file, `--noconftest`, disabled pytest plugin autoload/cache provider,
and the inert fixture above.

## Released execution and mandatory POST

After root's full PRE read and exact nine-payload source-plus-PRE archive
verification, the sole released command from `FullProject/` is:

```text
.venv/bin/python -I -S -B scripts/check_cad_guided_two_plane_schedule_v2.py
```

No arguments, case filters, retries or budget overrides. Limits remain one
worker, numerical threads1, **45 CPU /55 work /60 monitored-parent seconds**,
**1073741824 bytes sampled process-group RSS** and **1048576 output bytes**.
The child additionally uses `-u`; its CPU footer is required evidence, not a
substitute for the independent sampled process-group receipt. Missing footer
means unknown CPU/RSS evidence, not zero consumption.

Source reads are bounded, no-follow regular-file reads with identity checks and
before/after hashes. A fresh attempt is append-only, with exactly STARTED.json,
OUTPUT.txt and COMPLETED.json expected and no scientific artifacts. The
supervisor retains and separately identifies its private cache; do not inspect
or clean it. Sampled limits and absolute cleanup deadlines are not hard real-time
filesystem or adversarial process-identity guarantees. Source hashing, receipt
I/O and other unsupervised setup/teardown are not covered by the work clock.

POST must independently verify the exact attempt, terminal exit, all 80 actual
passes and zero other dispositions, bounded complete output and its digest,
CPU/RSS/wall evidence, no budget/primary/cleanup error, confirmed empty worker
group, exact three-file attempt inventory, eight unchanged source hashes equal
to this ledger, and the nine-member archive/PRE binding. No scientific artifact
inspection is released before confirmed group emptiness. Exit zero alone is
insufficient. Any failed/missing obligation remains terminal negative; preserve
the attempt without retry or favorable-prefix salvage.

A future green POST qualifies only the tested small invented-equation scheduling
contracts. It does not qualify maximum-size runtime/RSS, finite-support utility,
translation or pose observability, four-case connected recovery, native/Gold11
performance, calibrated uncertainty, observed bend count or production use.
