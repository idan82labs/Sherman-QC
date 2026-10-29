# Sherman QC F1 — PGRR-V1 correction-V2 component-ID STOP ruling

You are the external scientific reviewer for Sherman QC F1. Make a narrow, action-oriented ruling from the immutable evidence below. We want to hit walls through bounded execution, kill weak ideas quickly, and visually inspect every valid scientific result. Do not widen scope or tune against truth.

## Required ruling

Choose exactly one:

1. Authorize one fresh correction-V3 attempt that changes only deterministic binary component-ID selection in the producer and independent validator, preceded by the source-free regression and fresh-lock procedure specified below.
2. Issue a formal PGRR architecture STOP if this implementation failure rationally invalidates the architecture family.

If Option 1 is authorized, end with exactly:

`AUTHORIZE_PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_COMPONENT_ID_CORRECTION_V3`

If Option 2 is required, end with exactly:

`PGRR_V1_COMPONENT_ID_FAILURE_ARCHITECTURE_STOP`

Do not authorize a replay of either consumed claim. A V3 authorization must mean a fresh attempt identity, lock, claim, and output root.

## Prior Pro ruling

GPT-5.6 Pro authorized correction V2 because the original run failed before source access when `.resolve()` stripped the virtual environment. Verbatim ruling:

`docs/research/PGRR_V1_PRE_SOURCE_EXECUTION_STOP_CORRECTION_GPT56_PRO_RESPONSE_20260809.txt`

SHA-256:

`7ba6e5c99b4a9265696e6ce7bd78c23ab43ae316a192bc27ea9562df6c5fd4bd`

Authorization token:

`AUTHORIZE_PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_CORRECTION_V2`

## Bound correction-V2 record

- V2 lock self hash: `97f7ce0cc5581076ec703f043af9b8bcfa36d68b017021a42f090daf8b19294f`
- V2 lock file SHA-256: `4c7ca0dd7550a0fdc22f1b1961fe86f994bb89c740a07d3fd19edc0d6a713685`
- V2 consumed claim SHA-256: `12a684358794e8f0033a2bc46164b6b4b88451fe2e78377c079e71bc55431706`
- Corrected source aggregate: `4480be1d6a83d6acd92d0a6e02642c2723a684125b799c53230cddfb4a18f4dc`
- Bound source-free import regression: 16/16 checks passed.
- Full PGRR unit suite before lock: 10/10 passed.
- Exact interpreter: `/Users/idantal/ShermanQC/FullProject/.venv/bin/python`
- NumPy: `2.4.6`, loaded from the bound virtual environment.
- Same original source, 200,000-point cap, PGRR configuration, roles, thresholds, target, evidence schema, resource ceilings, STOP/GO rules, and visual obligations.

## What executed

The fresh V2 claim was consumed before the first worker. The authorized interpreter correction worked. Producer A:

1. launched in the locked virtual environment;
2. imported the candidate worker, NumPy, and PGRR geometry;
3. opened the exact bound raw source;
4. constructed the normal field;
5. entered target-blind candidate construction;
6. failed before writing a producer commitment.

Permuted producer and independent validator were not launched because the earliest-wall rule stopped the tranche. Target/gold/truth/CAD/expected counts/benchmark/runtime/main/production remained closed.

## Exact failure

Producer code:

```python
def _component_id(member_indices: np.ndarray, point_ids: np.ndarray) -> str:
    ids = point_ids[member_indices].view("V32").reshape(len(member_indices))
    return bytes(point_ids[member_indices[int(np.argmin(ids))]]).hex()
```

Traceback boundary:

```text
File "research/f1_pgrr_v1/geometry.py", line 492, in _component_id
    return bytes(point_ids[member_indices[int(np.argmin(ids))]]).hex()
TypeError: data type not ordered
```

The independent validator has the same latent operation:

```python
def _min_id(indices: np.ndarray, ids: np.ndarray) -> str:
    values = ids[indices].view("V32").reshape(len(indices))
    return bytes(ids[indices[int(np.argmin(values))]]).hex()
```

Source-free NumPy 2.4.6 probe on fixed binary rows:

```text
np.unique(void_rows): OK
np.argsort(void_rows, kind="stable"): OK and lexicographically correct
np.lexsort((void_rows, scalar_primary_key)): OK
np.argmin(void_rows): TypeError: data type not ordered
explicit uint8-column np.lexsort: OK and lexicographically correct
```

Repository scan found no other `np.argmin` on a void dtype. All other point-ID order operations use `argsort` or `lexsort` and passed the source-free probe. There are exactly two affected component-minimum functions: producer `_component_id` and independent validator `_min_id`.

## Immutable V2 evidence

- Process record SHA-256: `26ad7ccda25f3942018d7de818d5299229a8c431b4e0d4a424e35ff1d3037e3c`
- Process stderr SHA-256: `7547ac28a1c002f1a1d9e1ebb7bad2463835078824f1876b2b3bc28ed3e1cd84`
- Failure report SHA-256: `aca6acc64e1c87b0ac7cfda963b22a9503e881baca895f85fcabd4af51964243`
- Result SHA-256: `c296d338a88d4c342273a6db029980dde5e82d5f040ceeae137e4543c4569bcf`
- Visual critique SHA-256: `d0c542835469c3ab17ff20959b05749e1975bbdb83b4c1f29c882aef602a1244`
- Visual manifest SHA-256: `03f4513fc0da3d473f753340f54c3dc33a7bf5db48771826262faee579ec93fa`
- Decision: `PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP`
- `architecture_scientifically_evaluated=false`

## Visual critique

No scientific raster was produced by the consumed execution. This is not being hidden or replaced:

- source opened: yes;
- geometry started: yes;
- geometry completed: no;
- target opened: no;
- render started: no;
- candidate/ribbon/handoff counts: unavailable, not zero;
- clipping, grid phase, angular boundaries, misregistration, duplication, selectivity, and causal alignment: not visually evaluable;
- `visual_audit_ok=false` and `review_unresolved=true`.

Creating a render in a separate invocation would be an unauthorized replay. Therefore image absence agrees with the recorded failure point but supplies no evidence for or against PGRR.

## Proposed smallest correction V3

Preserve independent implementations:

Producer only:

```diff
 def _component_id(member_indices, point_ids):
     ids = point_ids[member_indices].view("V32").reshape(len(member_indices))
-    return bytes(point_ids[member_indices[int(np.argmin(ids))]]).hex()
+    local = int(np.argsort(ids, kind="stable")[0])
+    return bytes(point_ids[member_indices[local]]).hex()
```

Independent validator only:

```diff
 def _min_id(indices, ids):
-    values = ids[indices].view("V32").reshape(len(indices))
-    return bytes(ids[indices[int(np.argmin(values))]]).hex()
+    local = min(range(len(indices)), key=lambda offset: bytes(ids[indices[offset]]))
+    return bytes(ids[indices[local]]).hex()
```

Semantics remain exactly “lexicographically smallest 32-byte point ID.” No representation, parameter, threshold, source, sample, graph, pole, ribbon, target, causal rule, evidence field, resource ceiling, or decision criterion changes.

## Mandatory source-free regression before any V3 claim

Use fixed synthetic 32-byte ID arrays only; do not open the source, target, or any outcome.

The regression must require:

1. Producer and validator each equal Python’s exact `min(bytes(row))` oracle.
2. Agreement on adversarial IDs differing at byte 0, middle bytes, byte 31, all-zero/all-255, shared prefixes, and duplicates.
3. Invariance over canonical, reversed, and at least eight fixed explicit permutations (no randomness).
4. Duplicate minima return the same hexadecimal value regardless of stable tie position.
5. Empty-member input fails closed in both implementations.
6. No raw-source, target-ledger, gold, CAD, benchmark, or outcome access.
7. Exact implementation hashes, command, stdout/stderr hashes, and test result are bound into the V3 lock.

Then create a fresh V3 lock that binds the prior V2 lock, consumed claim, process record, result, failure report, visual critique, this Pro ruling, exact two-function diff, corrected source aggregate, successful regression, and every unchanged original scientific binding. Only after the regression passes may a fresh V3 claim be created and consumed before the first worker.

V3 remains exactly one execution. No retry, replacement input, fallback, second correction lane, parameter change, gold/truth/CAD/expected-count access, benchmark feedback, target-outcome tuning, synthetic scientific rescue, source-point serialization, runtime/main integration, production work, or part-specific logic is authorized unless your ruling explicitly changes that boundary.

If V3 reaches rendering, every canonical and negative/causal control image must be manually inspected and reconciled against the scalar claims before any fund/kill decision.

Please rule narrowly and end with exactly one required token.
