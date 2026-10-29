# Sherman QC F1 — Cubical IIP V1 one-shot STOP review

## Authority and requested ruling

The user explicitly authorized exactly one frozen-V7, gold-free smoke after the prior paper-only review. The smoke has executed exactly once and produced the sole terminal token:

`INVALID_CII_PERSISTENCE_PREFLIGHT_STOP`

No rerun, replacement input, gold/labels, CAD, expected counts, tuning, benchmark, replay, synthetic rescue, runtime/main, or production access is authorized. Review the immutable evidence and select exactly one bounded next step:

1. authorize a paper/code-only correction to the `serialized_bytes` accounting contract and specify its exact semantics and tests, without authorizing another scan; or
2. issue a formal architecture-family STOP.

Do not authorize or propose a rerun in this ruling. A later execution request, if scientifically warranted, must be a separate user decision and fresh immutable lock.

## Frozen execution identity

- V7 package ID: `SHERMAN_QC_F1_CUBICAL_IIP_V1_PREFLIGHT_LOCK_REVIEW_20260809_V7`
- Selected raw-content SHA-256: `050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97`
- Embedded execution-lock commitment: `54086ad146476dacd1e66217669f359c730646f80b80f6182101f32b19b0158b`
- Lock-file SHA-256: `b80a7dde043a6ca5871121a3d203244a2f9079f79dc795e6e7b9cb47cff1e0d9`
- Native binary SHA-256: `fb8594f07506257f93f62537dece416d31c67a8416fa2f3cb9850107dc5bf791`
- Claim-file SHA-256: `f9e2ddd561072c6242563f6c80e555fd90a3cd6476e468278f425cdb634630a3`
- Result SHA-256: `e7280fb49b8b4aad13e11e0c681a5e82261503e83665dd8f2360d225b22c02f0`
- BASE_A process-evidence SHA-256: `4d3e92330fea524954a9b4c8d161995aa2b19fc6a706d6170c4c2e580acc9768`
- BASE_A stderr SHA-256: `54bae9557e6e05850349c4aea8ab2592c6f3006b15be52e282427c4c2cc3ad71`

## Pre-execution assurance result

- Independent final pre-lock audit: CLEAN.
- Synchronized CIIP test slice: 132/132 passed.
- All 20 primary identifier/commitment KATs, independently encoded KATs, native numeric/identifier KATs, frozen allocation-plan recomputation, actual independent-process import isolation, and all 18 V7 tamper classes passed before execution.
- The source was selected only by the frozen content rule. Gold remained closed.

## Observed one-shot result

The supervisor published a durable STOP after the first role:

```text
outcome: INVALID_CII_PERSISTENCE_PREFLIGHT_STOP
failure: BASE_A:WORKER_EXIT_1
role: BASE_A
return_code: 1
peak_process_tree_rss_bytes: 186744832
scratch_bytes: 1179
wall_nanoseconds: 1831429000
cpu_nanoseconds: 2791390000
container_sha256: null
```

The exact exception is:

```text
research.f1_ciip_v1.container.ContainerValidationError:
supplied serialized_bytes disagrees with computed value
```

The direct code path is:

```python
# semantic_core.py
resource_values = {
    ...,
    "serialized_bytes": 0,
    ...,
}
container_bytes = encode_container(metadata_with(resource_values), tables)

# container.py
total_size = METADATA_REGION_BYTES + sum(len(table_bytes[name]) for name in TABLE_NAMES)
_set_computed(built["resource"], "serialized_bytes", total_size)

def _set_computed(target, key, value):
    if key in target and target[key] != value:
        raise ContainerValidationError(f"supplied {key} disagrees with computed value")
```

This appears to expose a contract ambiguity: `serialized_bytes` is a required closed-schema resource key, but its authoritative value is unknowable until canonical tables have been serialized. Supplying the required placeholder `0` is treated as a conflicting claim rather than an unset computed field.

## Questions requiring an exact ruling

1. Is the smallest valid correction to omit `serialized_bytes` from the pre-encoding resource map and let `encode_container` insert the authoritative value, while keeping it mandatory in the emitted container?
2. If omission before encoding violates the closed resource-key set, should a distinguished internal-only sentinel be permitted, with the serialized metadata still containing only the final u64?
3. Should `_set_computed` remain strict for caller-supplied nonzero values and accept only absence (or the exact final value), so forged accounting still fails closed?
4. Which new source-free KAT/tamper cases are mandatory to prove the correction: missing precomputed field accepted internally, zero/nonzero forged values rejected, emitted final value equals exact byte length, decoder requires the field, and metadata mutation fails?
5. Does this STOP reveal only an accounting-layer defect, or does it invalidate any semantic evidence computed transiently before the container exception? No semantic container was published, so the packet treats the attempt as providing no scientific geometry result.

## Research-grounding requirement

Ground the ruling in relevant primary or standards sources where they genuinely apply, and cite them precisely. At minimum, distinguish:

- mathematical persistence/cubical-complex invariants (for example Edelsbrunner–Letscher–Zomorodian 2002; Zomorodian–Carlsson 2005; Kaczynski–Mischaikow–Mrozek 2004),
- deterministic/canonical serialization and reproducible-build principles,
- fail-closed validation and provenance/accounting practice,
- and project-specific engineering conventions that are not literature-established.

Do not use citations as decoration. State explicitly when the recommended correction is an engineering design choice rather than a theorem or established scientific result.

## Required response format

Return exactly these sections:

1. `RULING_TOKEN`
2. `EVIDENCE_INTEGRITY`
3. `ROOT_CAUSE`
4. `LITERATURE_AND_STANDARDS_GROUNDING`
5. `EXACT_AUTHORIZED_CORRECTION_OR_STOP`
6. `MANDATORY_SOURCE_FREE_TESTS`
7. `WHAT_REMAINS_CLOSED`
8. `NEXT_PACKET_REQUIREMENTS`

Choose one token:

- `CIIP_V1_SERIALIZED_BYTES_ACCOUNTING_CORRECTION_AUTHORIZED_NO_EXECUTION`
- `CIIP_V1_ACCOUNTING_CONTRACT_ARCHITECTURE_STOP`

