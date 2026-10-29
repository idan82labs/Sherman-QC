# CIIP-V1 accounting correction — authorized source-free test results

Date: 2026-08-09  
User authority: one bounded source-free CIIP test batch  
Raw PLY authority: `NONE`  
Scan/smoke authority: `NONE`

## Action-first failure

The first focused accounting run intentionally exercised the new contract definitions and found a real implementation wall:

```text
tests/test_f1_ciip_v1_accounting_correction_source_contract_20260809.py
tests/test_f1_ciip_v1_serialized_bytes_contract_tensor.py

1 failed, 30 passed in 0.33s
```

Failure: an exact-valued `float(T)` caller preclaim did not raise. Python equality allowed `float(T) == T`, and `_set_computed` replaced the float with the computed integer before emitted metadata type validation.

## Bounded correction

The existing strict `_set_computed` remains unchanged. A narrow optional-preclaim check was inserted immediately before its `resource.serialized_bytes` call:

```python
if "serialized_bytes" in built["resource"]:
    _u64_json(built["resource"]["serialized_bytes"], "resource.serialized_bytes")
```

This preserves omission, accepts only an exact non-Boolean Python integer u64 preclaim, rejects ill-typed/range-invalid values before equality, and retains strict mismatch rejection.

## Focused accounting result

Command:

```text
.venv/bin/python -m pytest -q \
  tests/test_f1_ciip_v1_accounting_correction_source_contract_20260809.py \
  tests/test_f1_ciip_v1_serialized_bytes_contract_tensor.py
```

Result:

```text
43 passed in 0.33s
```

## Complete source-free CIIP result

Command:

```text
.venv/bin/python -m pytest -q tests/test_f1_ciip_v1*.py \
  --deselect tests/test_f1_ciip_v1_execution_lock.py::test_content_selected_source_matches_frozen_digest_without_path_semantics
```

Result:

```text
174 passed, 1 deselected in 23.37s
```

The sole deselected test is the only CIIP test that opens and hashes the frozen real-source file. It was not executed. The authorized batch used compact synthetic fixtures and static lock/code evidence only.

## Preserved boundaries

- No raw PLY opened or hashed.
- No scan semantics or smoke role executed.
- No replacement execution lock prepared.
- The prior one-shot claim remains consumed and terminal.
- No gold, CAD, labels, expected counts, benchmarks, replays, synthetic rescue, runtime/main, or production access occurred.
- The prior `INVALID_CII_PERSISTENCE_PREFLIGHT_STOP` remains historical evidence and was not repaired or converted.

## Current bindings

- corrected code aggregate: `70194e3fdafef334c0eb3c4ca01fd304285b90137cffe68f2cf13239894f2301`
- `container.py`: `b6f8696bdbef9ee161076db73ce8bd45c9a51204b182680af5b8b1245e3eb192`
- `semantic_core.py`: `bc91738aee711b10d754e63e484017862797b456fe43bb3977d46364f716b905`
- accounting source-contract tests: `edb2abb2d8008c4933a70d847c91a439ef82f80a79706051c160c402d193b55b`
- self-contained accounting tensor tests: `5dce8386c8d3a0cecc7aa5ccb604fdfb82408869ab3b8e99b16163522fa44657`

The stale V7 code aggregate and execution lock remain unusable with these corrected bytes.
