# EXP_CAD_0034 terminal physical-source RSS-limit result

## Ruling

`EXP_CAD_0034` is complete and terminal with status
`COMPLETE_TERMINAL_PRIMARY_SOURCE_RSS_LIMIT_NO_RERUN`.

The one claimed primary source child exceeded the frozen 4 GiB process-tree
RSS limit and was terminated by the external watchdog. No replay source child
and no renderer child ran. This is negative physical execution-feasibility
evidence under the frozen envelope, not an accuracy, F1, generalization, or
production result.

GPI: `GPI-C_NEGATIVE_PHYSICAL_SOURCE_RUNTIME_INFEASIBLE_NOT_F1_PROGRESS`.

## Exact resource result

- Primary wall time: `74.53849479090422` seconds of a `180` second cap.
- Peak sampled process-tree RSS: `4,316,708,864` bytes.
- Frozen RSS cap: `4,294,967,296` bytes.
- Excess: `21,741,568` bytes = `20.734375` MiB = `0.506210327%`.
- Kill reason: `RSS_LIMIT`.
- Return code: `-15` (`SIGTERM`).
- Timeout: false.
- RSS sampler failure: false.

The terminal orchestration result is
`output/research/autoresearch-loop/exp-cad-0034-cfpdg-v5-physical-attempt1/EXP_CAD_0034_ORCHESTRATION_RESULT.json`,
SHA-256 `dfa30180bfb8edc100fbce925b1cff602bc7527649604681416df6a23fbbac28`.

## Durable source accounting

The durable journal recorded attempted operations in the exact order:

1. hash N009;
2. hash N010;
3. load N009;
4. load N010.

Totals are exactly two scan-hash attempts, two point-cloud-load attempts, and
four byte-open-accounting operations. Evaluator/role/label/gold/quality/
expected-count access is zero.

The last durable stage is `BEFORE_N010_POINT_CLOUD_LOAD`. This is a
pre-operation checkpoint, not proof that N010 parsing did or did not complete.
The external SIGTERM prevented a later completion or exception checkpoint.
Likewise, whether the V5 scientific entry was reached in memory is not
established. What is established is that no source interface report, semantic
payload, visual payload, engine metrics, source result, renderer output, or
native asset was persisted.

## Child and artifact inventory

- Primary source children: 1.
- Replay source children: 0.
- Primary renderer children: 0.
- Replay renderer children: 0.
- Primary source directory: present and empty.
- Replay and renderer roots: absent.
- Native visual review: `NOT_COMPLETED_NO_VISUAL_PAYLOAD_DUE_PRIMARY_FAILURE`.
  The primary failed before a reviewable visual payload existed.
- Attempt files: exactly 9 regular files; no symlinks or special files.

Key hashes:

- immutable lock: `7eea3f2b5798e60e7648a94a4aec3d43c5facff3bab4f259ffb7a907f898f545`;
- detached approval: `ea3ba39eb26d8f791385558ad725a8f7cb12dafd540c27060cadeac61fa60546`;
- launch claim: `caa59b243c8274e9207194d222ef1e6ddb511522195d1b7429e57cbe623ea4a6`;
- source capability: `b9fde508efac1bce00dbb3d240bac644c085d73b1865d6084dfb52d30ea623f5`;
- source child claim: `26ff127d2b906fcd53f44880ad68e2a97eb3924efa07e1616747cb05a72d02d3`;
- process measurement: `1b72e9aff407ee9f1556baf9d53ba57a5abbef1550a65c1585b8e9d3f9ba6b8f`;
- scan journal: `4695bbcf683091e5e288e0a2c25f183bf8ec967105d11f9672dd5abd889ddd10`.

## Operator-observed preclaim launcher note

Before the claimed run, the operator observed a malformed launcher command
with redundant flags for paths that the frozen controller derives internally.
`argparse` rejected it before authority verification, attempt-root creation,
launch-claim creation, child spawn, or scan access. This observation is not
bound by an artifact inside the claimed attempt and is not used for attempt
counts. The persisted attempt establishes exactly one durable launch claim and
one primary child; those are the authoritative counts used above.

## Interpretation and next-step boundary

EXP34 provides no valid pair direction, attribution, role, quality, F1, or
generalization evidence. It demonstrates that the exact physical source path
is not feasible inside the preregistered 4 GiB envelope for this pair. EXP34
must not be rerun, rescued, downsampled, or granted a cap extension. Any later
work requires a fresh experiment identity and a prospectively frozen systems
or scientific hypothesis.
