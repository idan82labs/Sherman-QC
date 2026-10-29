# Sheared bounds V1 — independent source review

Date: 2026-09-12. Reviewer: `context_v3_adversarial`.

## Exact disposition

**SOURCE_ACCEPTED_FOR_BOUNDED_INVENTED_QUALIFICATION**, within the trusted
internal-record/caller boundary below. No remaining concrete numerical-proof
or bounded-accounting blocker was found in the reviewed source. This is not a
test pass, PRE release, integrated index, demonstrated speedup or physical claim.

Final candidate: `backend/cad_guided_sheared_bounds_v1.py`, **11,420 bytes**,
**269 lines**, SHA256
`36b87ea58c920774602d7f7cf53e817d5d130bf2b98bcc02dd8b8cce947201b6`.
The entire final file was read in `1f134d`; independent stdlib AST/hash inspection
`d9c9df` agrees. AST parsing, not target import/compile/execution, found six
top-level functions and three classes. Imports are exactly dataclasses.dataclass,
math and NumPy; there are no local-module imports, data/file/process calls,
geometry fitting, or import-time target computations beyond definitions/constants.

The accepted mathematical basis is
`2026-09-12_raw_extrema_oriented_v2_math_review.md`, 21,717 bytes, SHA256
`07c2df1b3a93f49402647869387d351a273ee7d69ab08e4cd48af8a88ec48e4d`.
This review concerns the new primitive, not a change to the existing V1 extrema
index or V3 context. No existing source was edited by this reviewer. No tests,
target imports/execution, raw/Gold/cache/attempt reads, or physical work occurred.

## Numerical proof-to-source correspondence

- `build_leaf_box` (130–168) admits an exact float64 ndarray with 1–256 rows,
  exactly three columns, an explicit axis permutation, and represented slopes
  with magnitude at most one. It makes a local immutable bytes-backed snapshot
  and checks finite coordinates within ±1e6. The <=6,144-byte snapshot prevents
  subsequent caller mutation from altering the produced box; it does not
  authenticate the caller's larger raw-buffer lineage.
- The constructed z uses the prescribed two separately rounded products and
  two separately rounded subtractions. The residual interval is computed from
  the ORIGINAL represented coordinates and represented z, using outward scale
  and subtraction. It encloses the exact identity
  `x_k = a*x_i + b*x_j + z + residual`, not a rounded reconstruction equality.
- `_Ops` interval addition/subtraction use correct cross endpoints; signed
  scalar scaling swaps endpoints for negative coefficients. General interval
  multiplication takes all four rounded endpoint products, their min/max, then
  outward neighbors. Because nearest rounding and nextafter are monotone,
  selecting min/max before the final outward step is conservative as well.
  Subnormal products are enclosed; no normal-only relative-error model appears.
- `merge_boxes` (171–191) requires identical represented axes/slopes and takes
  endpoint min/max hulls. It preserves both child enclosures. Its checked
  positive exact-int row counts and overflow-avoiding sum admission cap merged
  counts at 1,000,000. It does not prove disjoint/raw-ID membership.
- `bound_sheared_box` (231–269) encloses both uncertain direction coefficients
  `d_i+a*d_k`, `d_j+b*d_k` before multiplying by raw-axis boxes. It includes the
  z and residual terms and all three origin products plus the constant. This
  is the exact affine shear formula; omitting any of those terms would be
  unsound, but none is omitted here.
- `_rounded_world_and_error` (194–228) follows original world-box subtraction,
  sign-selected product, sum01, sum2 and final-addition stage ranges. Every
  local error uses the outward positive-neighbor spacing at maximum absolute
  rounded stage magnitude. The proof covers zero, subnormals and binade
  boundaries. Finite/ordered checks reject unsupported nonfinite arithmetic.
  Its error accumulation order sums the three product-error bounds followed by
  the three addition-error bounds; this differs only in positive upward-rounded
  summation order from the math document's recurrence and still encloses the
  same sum of nonnegative local-error bounds.
- The primitive expands the exact affine interval by that original-program
  error, then intersects it with the original rounded world interval. It checks
  the intersection and returns both intervals. Original world zero endpoints
  are explicitly canonicalized to +0. It does not evaluate extrema or return
  raw witnesses and cannot substitute a transformed leaf program.

Within the admitted coordinate/query/slopes domain, numerical magnitudes remain
far from overflow. Nonetheless nonfinite/invalid NumPy operations can raise;
these exceptions are not caught to publish an apparently valid prefix. The
caller must implement the whole-output refusal boundary. Existing arithmetic
sentinels/environment assumptions still belong to the eventual index/caller;
the primitive does not establish processor-wide rounding/underflow behavior.

## Work and memory boundary

Each numeric operation is preceded by `_Ops.take`. Parent charge is called
before incrementing local attempted steps and before the actual operation. A
parent callback must enforce its cap **by raising**, retain attempted aggregate
work, and be trusted/non-mutating; returning false is not a refusal protocol.
If the local 65,536-step cap is exceeded, the parent has already received that
attempted charge and the operation has not run. No rollback of attempted work
is appropriate.

Static successful charges derived from source, not measured execution:

| Call | Additional numeric steps |
| --- | ---: |
| build_leaf_box with n admitted rows | 78 + 61n |
| build_leaf_box with n=256 | 15,694 |
| merge_boxes | 75 |
| bound_sheared_box | 390 |

The build formula comprises 64 admission units; 3n snapshot units; 18n raw
validation units; 4n z operations; 8n+2 signed-scale units; 12n residual interval
subtraction units; 6n residual validity units; 10n reduction inputs; and 12
scalar interval validity units. Merge is 64 admission plus 11 union/sum units.
Query is 165 units through exact-affine validation, 207 original-world/error
units including positive-zero normalization, and 18 expansion/intersection
units. These are an explicit abstract model, not CPU instructions or old V1
coordinate-visit units. Short-circuit validation may do less actual work than
the conservative charge; that is not an undercharge.

The 131,072-byte WORKING_BYTES reservation is a caller obligation in addition to
all caller-owned inputs/indexes/reports. The primitive does not call a memory
budget callback. Array-producing operations are scalar or <=256 elements;
the snapshot, z, signed residual arrays, and nested interval scratch are small
bounded payloads. The fixed reserve is ample for that numerical scratch and
small Python bookkeeping under the stated trusted-size boundary. It is not a
total NumPy/runtime/process RSS cap, a measurement of peak memory, or coverage
of arbitrary allocations made by the external callback. An eventual integrated
caller must also account persistent boxes/arrays, old/new overlap and parent
CAD/context/cache allocations; none is implemented by this leaf primitive.

The initial charger-callability check, shape/type metadata, tuple construction,
callback cost and ordinary Python bookkeeping are expressly outside hardware
operation accounting. The successful fixed admission charges precede the
admitted numeric-value comparisons. Public callables do not expose arbitrary
broadcast shapes to `_Ops.binary`; its internal operands are scalar or a
one-dimensional admitted leaf column.

## Resolved source findings and trusted-record limit

Two concrete review findings were fixed by root before the final pin:

1. Merge admission comparisons and the query box-type check initially preceded
   their first work precharge. Final source precharges 64 admission units first
   and clarifies the initial callback/metadata exclusion.
2. Unrestricted repeated merges could grow a legitimately constructed row-count
   Python integer without bound, weakening the fixed small-Python envelope.
   Final source checks positive exact-int counts and `left <= MAX_NODE_ROWS -
   right` before adding, with MAX_NODE_ROWS=1,000,000.

Root also canonicalized the world interval's signed-zero endpoints under two
charged comparisons. The original version's -0 was numerically safe for pruning
but did not match the original program's positive-zero representation exactly.

`ShearedBox` is an explicitly **trusted internal value**, not a parser or
authentication boundary. Its dataclass is frozen but manually forged records,
`dataclasses.replace`, foreign mutable field contents, or bypassed frozen
attributes are not fully validated by merge/query. Correct use means records
created by this builder/merger, with caller-authenticated stable lineage,
bounded shapes and unchanged contents. Tests may use a deliberately forged
record to exercise a specific arithmetic guard, but must not describe success
of that one guard as complete tamper validation. No public serialized-box API,
raw-ID membership check, support digest, or unique-row attestation exists.

Even two valid boxes can overlap in membership; their union enclosure remains
safe, but summed row_count is not a proof of unique membership. The eventual
index must retain its exact raw-ID permutation, support ownership and minimum
raw-ID node metadata separately. Those are not stored in this primitive.

## Qualification and efficacy boundary

This source is suitable for a small independently frozen invented primitive
check: represented reconstruction residuals against an independent exact-rational
oracle; original staged-value containment; every axis permutation/sign;
cancellation, subnormal and signed-zero cases; merge hull/count cap; malformed
admission; exact parent precharge/late callback failure; and the known flat-plane
non-pruning limitation. Exact fixture source/denominator/runtime/PRE must still
be reviewed before any execution. This document does not invent a suite count
or authorize invocation, and no previous qualified suite needs to be rerun.

There is no full index or parent cumulative integration yet. Keeping old tree,
seeds and world-bound child priorities is only a documented instruction here,
not an executed integration guarantee. The error-enlarged shear box can still
straddle the actual extrema at every node of an exactly flat normal query or
repeated extremal plateau. All-row fallback and honest budget refusal remain
valid outcomes. Leaf-row savings, if later obtained, must be compared with the
new construction/bound cost rather than called a speedup by themselves.

Final disposition remains source-only acceptance of this exact bounded internal
primitive. No geometric accuracy, physical measurement, F1 gain, larger work
cap, or successor physical run is established or released.
