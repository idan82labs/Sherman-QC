# CIIP-V1 serialized-byte accounting correction record

Date: 2026-08-09  
Authority: `CIIP_V1_SERIALIZED_BYTES_ACCOUNTING_CORRECTION_AUTHORIZED_NO_EXECUTION`  
Execution authority: `NONE`

## Outcome

The authorized paper/code-only correction has been applied. No project module was imported, no source-free test was invoked, no compilation occurred, no native KAT ran, no PLY was opened, and no execution capability was created or consumed.

The sole production behavior change is the deletion of the premature construction-time claim:

```python
"serialized_bytes": 0,
```

from `execute_semantic_role.resource_values` in `research/f1_ciip_v1/semantic_core.py`.

`container.py`, `_set_computed`, the decoder, the independent validator, scientific configuration, schema, native source, geometry, persistence, source selection, and all runtime authority remain unchanged.

## Exact production diff

The complete unified diff is preserved in:

`docs/research/CIIP_V1_SERIALIZED_BYTES_ACCOUNTING_CORRECTION_EXACT_DIFF_20260809.patch`

Before:

- path: `research/f1_ciip_v1/semantic_core.py`
- bytes: `9019`
- SHA-256: `5cd1d36349ff136d12d3bf21827a2a8e6abc774508f243669aed90f787818145`

After:

- path: `research/f1_ciip_v1/semantic_core.py`
- bytes: `8988`
- SHA-256: `bc91738aee711b10d754e63e484017862797b456fe43bb3977d46364f716b905`

The byte delta is exactly the authorized 31-byte line deletion.

## Construction-versus-emission contract

Before encoding, the construction resource object may omit `serialized_bytes`. If a caller supplies it, the value must be a non-Boolean JSON u64 exactly equal to the encoder-computed final length.

The encoder remains the sole authoritative owner of:

```text
T = 1,048,576
  + byte_length(partition_node)
  + byte_length(leaf_aggregate)
  + byte_length(cell_scale_field)
  + byte_length(space_scale_adjacency)
  + byte_length(persistence_disposition)
  + byte_length(rejection)
```

If absent, the encoder inserts `T`. If supplied and unequal or ill-typed, the existing strict computed-field check must fail closed. No zero, null, negative, string, object, enum, max-u64, or other sentinel means “unset.” The emitted resource schema still requires `serialized_bytes`, and both decoder and independent validator still require equality with the exact received byte length.

## Mandatory test definitions

The complete source definitions requested by GPT-5.6 Pro are present but have not been imported or executed:

- `tests/test_f1_ciip_v1_accounting_correction_source_contract_20260809.py`
  - production resource-map omission;
  - stale V7 code/execution binding rejection;
  - one-production-file scientific noninterference allowlist.
- `tests/test_f1_ciip_v1_serialized_bytes_contract_tensor.py`
  - omission-to-authoritative-value;
  - exact preclaim byte identity;
  - zero, wrong-value, type, range, container, and sentinel rejection;
  - emitted closed-key enforcement;
  - decoder and independent-validator tamper rejection for zero, `T-1`, and `T+1`;
  - truncation, trailing-byte, noncontiguous-offset, and declared-table-length non-substitutability;
  - fixed-region/cap boundary arithmetic through a bounded seam;
  - supervisor resource-patch invariance;
  - independent actual-length accounting.

No pass claim is made. Execution of these definitions requires a separate explicit user decision.

## Changed and added file inventory

| Path | Bytes | SHA-256 | Classification |
|---|---:|---|---|
| `research/f1_ciip_v1/semantic_core.py` | 8988 | `bc91738aee711b10d754e63e484017862797b456fe43bb3977d46364f716b905` | sole production change |
| `tests/test_f1_ciip_v1_accounting_correction_source_contract_20260809.py` | 7366 | `43fd2d90f08082c39278bd7b7499962e9a14819bb554a6ec559820e7f51c8225` | new test definitions |
| `tests/test_f1_ciip_v1_serialized_bytes_contract_tensor.py` | 9190 | `ffa7e96bedb55e6f0313d5383022dd4af82f60a404c66b2391b0ba30ccc6ca8a` | new test definitions |
| `docs/research/PROFESSOR_F1_CIIP_V1_SERIALIZED_BYTES_ACCOUNTING_CORRECTION_LEDGER_20260809.md` | 8940 | `9c9c4a4cc64d7c60300ce4f4a3075574be7c5ec3a049489f4ab1c69bbb3ddbf6` | authority/planning ledger |
| `docs/research/CIIP_V1_SERIALIZED_BYTES_ACCOUNTING_CORRECTION_EXACT_DIFF_20260809.patch` | 332 | `e31479c880e38221104eee220f40b35b17a2aed51032d1594462428ddbfea96a` | exact production diff |
| `docs/research/PROFESSOR_F1_CIIP_V1_ONE_SHOT_STOP_ACCOUNTING_RULING_GPT56_PRO_RESPONSE_20260809.txt` | 22061 | `8da9592bd8a68b423253c87abf1b611e843bab346480a1549467ac838d0ca8fd` | preserved Pro ruling |

This record is intentionally excluded from its own inventory; the outer packet manifest binds it.

## Stale bindings

The prior V7 code aggregate is:

`344bd553ce8500c6ea3d2f1f22cfb28254671ba01fa8f94a5599dd975e910814`

The corrected static code aggregate is:

`cafe23bba37dedcf7dbbf506f46fff08557b7eb193f6f8f9e91b59bd317b039f`

Therefore the prior execution-lock commitment:

`54086ad146476dacd1e66217669f359c730646f80b80f6182101f32b19b0158b`

is stale for corrected code. It remains immutable historical evidence and is not edited or reused. No replacement lock has been prepared.

## Byte-identity and noninterference ledger

Static file hashing confirms no authorized production file other than `semantic_core.py` changed relative to the consumed V7 lock. The following protected components retain their prior bytes:

- `config.py`: `f2db944ececf092fdaf659065bd0cd4122f52ceac0a00eab3c30f8ed05d725cc`
- `container.py`: `f7ff7177839e5ef0a96e33fec729b7244b0bf3271850f86f7d8433e3c936b7d2`
- `fields.py`: `988fc04144c75f2eb9ee98b3951a43bb207e99dcfd880e9841d04bf10a7115f0`
- `graph.py`: `a1567fadd8abca12af3e96bb7b3fe221250991cf71a7d80a0716f7ee0f5ec159`
- `independent_semantic_rebuild.py`: `dedfbfa04b765c08befc010deef3bfc3a7b2c1aa2516e79fe693aa9c99984cfc`
- `independent_validator.py`: `5b3dabb827a3cd6b9c7481617db0fbf9ba6d6f72f1df19e3a13c48f66dccf0fd`
- `native/semantic_worker.cpp`: `41c59ab22b176d3d7c2a4584a6dca6671a1af05b64ff99448e9e20c2825feabd`
- `partition.py`: `f4fc8f7075b8728e06c6d0fb5bc2c96636f99784eca7900045ac2d2a4b08119f`
- frozen configuration attachment: `813acdd96f5a8108caf2c87dfdd733b5c856248c4522d2789d733e84016dce04`
- frozen schema attachment: `e996566e2b7da94dfc99b97e8bb59e5b6f1fdb25692a16890b63731b5c16ea4f`
- static resource proof attachment: `9726a1624afa115134135c0451ebf89e3f82241515000a18557988ab721da160`
- safety/terminal attachment: `dd8581b61ea83f22153c5c0e13ee4267312ec1f2b9356c1de3a989acf75a651d`

No scientific table, identifier, geometry, persistence, parameter, gate, or authority behavior is intentionally changed. No dynamic byte-identity claim is made because execution remains closed.

## Preserved terminal state

- prior result: `INVALID_CII_PERSISTENCE_PREFLIGHT_STOP`
- consumed claim: terminal and non-reusable
- `gold_or_labels_accessed: false`
- `replacement_input_authorized: false`
- `rerun_authorized: false`

No corrected execution, test run, or scientific GO is claimed.
