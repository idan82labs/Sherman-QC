# Scheduler integration V3 — independent exact PRE

Date: 2026-09-10. Disposition: **GREEN for one prospective fixture-corrected
source gate only**, after root fully reads this PRE and the predecessor RED POST,
verifies the exact archive and unchanged launch state. No project import,
collection, test, native or geometry execution occurred in this review.
The 69 outcomes are prospective; this is not a prediction that they will pass.

## Predecessor and exact correction

Integration V2 remains terminal RED: 25 PASS and 44 setup ERROR outcomes. The
independent POST verified all 44 tracebacks have the same missing `scan` key in
the newly supplied mock `patch_evidence`; no byte-workflow test body ran.
The accepted source-review boundary did not detect that fixture mismatch.
Preserve the original attempt, all 52 sources, PRE, archive and RED POST unchanged.
V3 is separately versioned, not a retry or rewriting of V2's result.

The complete new test, harness and protocol were read. Mechanical AST comparison
against the hash-bound V2 sources confirms:

- All 12 test functions, their decorators, assertions and parameter values are
  identical. The complete test module becomes AST-identical after removing its
  changed leading docstring and the single new `scan` keyword in `patch_evidence`.
- That keyword is a fixed declared envelope with exact format
  `noisy_mesh_metadata_schema_v6`, `schemas=[["schema_version"]]` and
  `root=["D", 0, ["noisy_mesh_planes_v7"]]`. It satisfies the existing legacy-mode
  header checks without changing the binder. It is not complete V7 metadata,
  actual plane preparation, a raw-support certificate or composite-lane evidence.
- The harness differs only in test/protocol/harness source names, addition of the
  RED POST, its private non-main supervisor-module name and fresh attempt prefix.
  Limits, worker flags, test arguments, supervision and failure handling are
  unchanged. No engine, assertion, geometry, policy or threshold changed.

The repaired fixture still checks its two mocked contexts using actual binders
before installing workflow spies. The new literal has exact expected keys,
list shapes, integer schema index and V7 label. The remaining setup path was
traced through schedule, array and context binding in source, including the
ordinary V13 helper constructors. No additional static setup blocker was found;
only the prospective execution can establish the 69 outcomes.

## Denominator, inputs and scientific scope

Independent AST count is 12 functions with multiplicities
`2+1+9+2+7+4+2+2+32+2+2+4=69`. There are no declared skips or xfails. The unchanged
V13 helper supplies ordinary `_arrays` and `_registration` constructors; none of
its tests or decorated fixtures is invoked. Require 69 actual PASS and zero
errors, failures, skips, xfails, xpasses or other dispositions, not merely exit 0.

The 25 preceding configuration/prefilter outcomes remain in this explicitly
frozen 69-outcome successor; they are not silently substituted for the 44
previously unreached outcomes. Two actual positive and eleven negative prefilter
cases retain exactly 50 raw rows, 64 scan triangles, four CAD triangles, a
two-plane bank, no triplets and 64 hypothesis slots. Negative schedules must stop
before context/finite queries. No global ICP or measurement recovery is exercised
by these actual prefilter cases.

The byte fixture remains 72 raw/36 selected rows, two scan triangles, three CAD
triangles/36 samples, two explicitly mocked triplet contexts and zero selected
two-plane pairs. Decode, native extraction/sampling, registration, measurement and
publication validation are mocks; actual configuration and array/context binders
are the target. All required measurement/support policy constructor fields remain
unchanged. This does not qualify complete real metadata, native geometry, physical
registration or metrology. Literal contact/count/pose uncertainty claims are not
upgraded. The connected 2/4 negative and later native/Gold11 gates remain intact.

## Exact 53-source ledger by explicit inheritance and delta

The canonical predecessor ledger is the 52-row table in
`docs/research/2026-09-10_two_plane_schedule_v2_integration_source_pre_run_audit.md`,
15,441 bytes, SHA256
`cf2325d81829c56f1c66acd13d59da671fb8d703d6ef139d8fdb16072e0c1594`.
That exact document was verified, and remains preserved in the predecessor
53-payload archive named in the RED POST. This reference incorporates its exact
path/byte/SHA256 rows, not a directory glob, floating revision or mutable lookup.

Take exactly those 52 rows, remove exactly these three paths, then add exactly
the four rows below:

- `scripts/check_cad_guided_two_plane_schedule_integration_v2.py`
- `tests/test_cad_guided_two_plane_schedule_integration_v2.py`
- `docs/research/2026-09-10_two_plane_schedule_v2_integration_source_protocol.md`

The remaining **49 inherited rows** retain every byte count and SHA256 from that
hash-bound ledger: 46 backend modules, the exact supervisor, the ordinary V13
test helper and `tests/__init__.py`, totaling 1,175,637 bytes. All 49 current
payloads were individually verified; the three removed V2 files also still match
their original hashes. No inherited source is changed or omitted.

| Added source | Bytes | SHA256 |
|---|---:|---|
| `scripts/check_cad_guided_two_plane_schedule_integration_v3.py` | 8494 | `c6755d8bbf4bbf395a5ac04ccb28c6acb0330167c27d9c6d001403236ee39ab9` |
| `tests/test_cad_guided_two_plane_schedule_integration_v3.py` | 29658 | `93120a3adc070c6f379ce5dec3200f39eb23e97afa7a2a43a569e49035d3b8bc` |
| `docs/research/2026-09-10_two_plane_schedule_v2_integration_source_post_run_audit.md` | 8005 | `9ef59b01a46a66a940a52aa2dfcdea390248051c0bb7b416c87c6371aa34c0ad` |
| `docs/research/2026-09-10_two_plane_schedule_v3_integration_source_protocol.md` | 5086 | `2099f39fa375bf141d0e9a6f7a2e37efcbe58af361190c97beeb7ff11e9bf28f` |

This defines exactly **53 unique sources / 1,226,880 bytes**, equal to the new
harness's literal source tuple as a set. Every candidate file was read using
bounded regular-file/no-follow and stable-identity checks, at most 2 MiB each.
All 51 Python sources parse without importing them. The two Markdown payloads
are the new protocol and predecessor RED POST. Preserve the exact predecessor
PRE/archive as the immutable reference resolving the inherited rows.

## Import boundary and fixed resource custody

AST comparison establishes no added imports or dynamic import target. The
previous literal/dynamic import closure remains unchanged and fully represented
by these sources. The new fixture still selects literal backend module names
and the one named V13 helper file. The harness installs an inert backend package
before pytest; the test fixture controls/restores its backend module tree. No
application `backend/__init__.py` startup is executed. Native paths remain behind
the explicit mocks. The legacy header correction adds no I/O or import boundary.

The 24-byte `tests/__init__.py` is explicitly inherited and frozen. `lstat` again
confirms no object, including dangling symlink, at the enclosing repository
`FullProject/__init__.py`. Root must repeat that named absence check immediately
before launch, and POST must repeat it afterward. No scripts-package import is
used. Interpreter, NumPy/SciPy/pytest, stdlib and OS binaries are not attested by
this repository ledger.

Frozen resources are unchanged: one worker, numerical threads one, 45 worker CPU
seconds, 55 monitored work seconds, 60 parent/cleanup seconds, 1,073,741,824 bytes
sampled group RSS and 1,048,576 output bytes. The worker uses `-I -S -B -u`,
disabled plugin autoload, `/dev/null` pytest configuration, `--noconftest`, disabled
cache provider and one named V3 test module. No case or budget override is allowed.

Only the exact hash-bound supervisor definitions are reused, not its old main,
tests or source list. Source/receipt I/O lies outside the supervised clock;
sampled RSS is not a hard allocation or hostile-OS guarantee. The harness only
records the expected denominator; independent POST must verify the actual count.

## One-launch conditions and mandatory POST

After root reads this PRE and the RED POST, archive exactly these 53 named
regular payloads plus this exact PRE: **54 payloads**, no extra, duplicate, link
or AppleDouble member. Verify every archive payload against the defined ledger,
separately record the new PRE/archive hashes, retain the predecessor records and
check that all candidate bytes and the package boundary remain unchanged.
Confirm there is no other heavy worker, then launch this command once:

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_two_plane_schedule_integration_v3.py
```

The fresh `two_plane_schedule_v3_integration-` attempt must contain exactly
`STARTED.json`, `OUTPUT.txt` and `COMPLETED.json`, no scientific artifact. Preserve
the separately named private cache uninspected. After confirmed empty writer
group, independent POST must check full bounded output, 69 PASS/zero other
dispositions, exit 0, CPU/RSS/wall receipts, no budget/primary/cleanup/source error,
all 53 before/after/live hashes, the absent enclosing initializer and all 54 exact
archive payloads including this PRE. Exit zero alone is not sufficient.

Any failed obligation is terminal negative, without retry, cap increase, fixture
reduction or prefix promotion. A GREEN POST would qualify only this invented
integration contract, superseding no historical record and authorizing no
physical access. The later four-case connected packet still needs a separately
frozen protocol/PRE and all four original measurement successes; native-byte and
Gold11 qualification remain later gates. No physical score, accepted pose,
observed count, calibrated uncertainty or factory release follows from this PRE.
