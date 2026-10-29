# Sheared bounds V1 — independent fixture/delta review

2026-09-12; reviewer `context_v3_adversarial`.

Decision: **NO_BLOCKER_FOR_EXACT_INVENTED_PRE**. This is source review, not
collection, execution, a passing outcome or permission to invoke the harness.
The separate PRE reviewer owns final runtime/source-map binding and release.

## Exact reviewed sources

| File | Bytes | SHA256 |
| --- | ---: | --- |
| tests/test_cad_guided_sheared_bounds_v1.py | 16146 | `75b0b73bb11ee8602773faabe2f75330142af754bafdcff481a81f22536a2bc0` |
| scripts/check_sheared_bounds_v1.py | 22373 | `bb4af4868cd325709a6aa3aa7eeb14992b06ecb875528ad7bf39a61ebc4bbcf1` |
| scripts/check_raw_extrema_index_v1.py, comparison baseline only | 22265 | `b26c9fc1218eeaaadd854f19e6802cd0c48b94c24f319661cd29921c0eaffaf8` |
| docs/research/2026-09-12_sheared_bounds_v1_source_protocol.md | 6184 | `a69549dcb9a70bb86e4c3deae310c91630dca868590f2e86f820a5c52040ae87` |

Candidate remains the previously fully reviewed 11420-byte helper SHA256
`36b87ea58c920774602d7f7cf53e817d5d130bf2b98bcc02dd8b8cce947201b6`.
Full test/new harness/delta/protocol/resource review read. Stdlib AST/hash
check `886185` confirms pins/counts. No target import/compile, test/collection,
raw/Gold/cache/attempt access or existing-source edit occurred.

## Denominator and nonvacuous premises

Exactly 14 test functions expand prospectively to **40 items**:

| Control | Items |
| --- | ---: |
| Six axis permutations | 6 |
| Represented non-dyadic slopes | 3 |
| Subnormal/normal-boundary/zero/cancellation/binade/large translation | 6 |
| Merged union and associativity | 2 |
| Exact flat normal limitation | 1 |
| Snapshot mutation | 1 |
| Malformed leaf | 6 |
| Invalid axes | 2 |
| Invalid slopes | 3 |
| Invalid query with successful query premise | 3 |
| Mandatory charger/internal type | 1 |
| First-parent-charge refusal in build/merge/query | 3 |
| Fixed charges and exact 1M merge-count boundary | 1 |
| Incompatible merge shear | 2 |

Loops are not extra items. No skip/xfail/inherited tests or conditional selection.
Largest leaf: 256. The 1M boundary uses 20 small repeated-point summaries, not a
million-row allocation or unique-lineage claim. Set-bit merges reach exactly 1M;
+1 must raise the exact row-limit reason after only admission charging.

All members are checked. Non-dyadic fixtures require a nonzero exact residual.
The flat dyadic fixture asserts scalar outputs zero, then requires the bound
to straddle zero with positive error radius: a limitation, not speedup evidence.

Invalid-query tests first check a valid query; merge negatives use two checked
boxes. Negatives require exact anchored reasons, not any ValueError. Parent
refusal checks exception identity and one attempted 64 charge. No late-callback,
whole-index clearing, forged-box authentication or traversal/tie claim is made.

## Oracle and work validity

Fraction.from_float captures represented values, not ideal decimals. Affine
and residual oracles never call candidate arithmetic. The shear oracle writes
the two products/two subtractions independently, then computes exact residuals.
The scalar oracle writes subtraction/product/add01/add2/constant separately,
with positive-zero normalization and no sum/dot/FMA. It is independent code,
not independent hardware; frozen Python/NumPy arithmetic assumptions remain.

Queries separately check affine containment, staged rounding-error radius,
staged-value containment, world/intersection containment, scalar sign-selected
corner outputs and positive zero. A wide affine interval cannot mask an invalid
error radius. Positive fixtures stay within admission bounds; the range
negative is explicitly above 1e6. Translated non-dyadic fixtures remain below it.

Charges agree with prior reviews: build `61N+78` (15694 at 256), merge 75,
query 390. N=1,17,256 are explicit; positive helpers reconcile callback totals
and 65536 guard. Fixed constants are asserted, not measured RSS/wall time.

## Harness/protocol delta and isolation

AST comparison finds 16 unchanged top-level function/class definitions,
including supervisor extraction, outcome accounting, runtime binding,
publication inventory and CLI. Only `_sources`, `_pre`, `_worker`, `_parent`
change through the displayed identity/closure substitutions. Globals change
to the new candidate/test/PRE/markers, nine-source closure, 40 items, and the
explicit prospective limits: 30 CPU, 40 work-wall, 45 supervision seconds,
1 GiB sampled group RSS and 1 MiB output. These are intentional new limits, not
a claim of identical old budgets or a post-launch increase. Protocol agrees.

The new worker uses isolated/no-site/no-bytecode Python, numerical threads 1,
plugin autoload/conftest/cache disabled and one exact test path. The fixture
loads only the named helper by file spec under a private module name, cleaning
that sys.modules entry afterward; it does not import backend initialization.
Harness guards forbid native/fitting imports and root/backend initializer
execution, then check loaded forbidden modules. This is a source-reviewed
fixture boundary, not an arbitrary-code filesystem sandbox.

Actual collection must have 40 unique nodes, only the named test path, and 40
ordinary passing setup/call/teardown triplets with zero other outcomes.
Source/runtime/PRE checks occur before and after; parent output parsing and
postflight are group-quiescence gated. The inherited supervisor body is not
modified. Parent synchronous I/O remains outside the 45-second supervisor
clock; final publication is a separate exact-three-file gate. Full runtime and
the three selected supervisor bodies are independently checked by PRE owner.

No remaining fixture/delta blocker found. Exact PRE acceptance, no-overlap,
sole invocation and independent POST are still required; no numerical result,
efficacy improvement or physical release is implied here.
