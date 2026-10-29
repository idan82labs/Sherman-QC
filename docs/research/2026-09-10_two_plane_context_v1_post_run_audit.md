# Two-plane context V1: independent retained post-run audit

Date: 2026-09-10. Decision: **GREEN_VALID_FROZEN_PURE_TWO_PLANE_CONTEXT_V1_GATE**.
Blocking findings: **none within this pure-helper qualification**.

Reviewer: `/root/production_architecture_review`, an AI reviewer independent of
the helper, test and harness/protocol authors. Earlier design contribution is
disclosed in the [pre-audit](2026-09-10_two_plane_context_v1_pre_run_audit.md).
This is an independent retained-evidence review, not a second execution or human
scientific/manufacturing approval.

## Exact execution and retained evidence

Root reports the first named gate was executed once after reading the complete
PRE-GREEN, with observed parent exit zero. The retained child receipt independently
records exit zero. The authorized command was:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_context_v1
```

Exact owned attempt:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_context_v1-6axnp2wc`.
Its immediate inventory contains exactly the three required regular files below,
with no extra, partial or symlink entry. Each file was read in full, hashed and
checked for unchanged metadata around that bounded read. Total: 2,389 bytes.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `STARTED.json` | 1004 | `30011052b8cac7404aa62825ae1a0d9a742c33e549a6f8f3f672ed904bb91baf` |
| `OUTPUT.txt` | 209 | `d27ecabe45178fd8e2c0e3436fb350435bfc81f99998d6b1d17ac3bd0366c96a` |
| `COMPLETED.json` | 1176 | `19c234d83bd05075bebc0f831615bee829c78ab7da6cad607e089db5a2acfcfa` |

OUTPUT has exactly 110 passing progress markers, `110 passed in 0.65s`, and one
worker resource JSON line. There are no failed, skipped, xfailed or omitted
outcomes in this output. This agrees with the independently source-enumerated
31 test functions /110 parameterized outcomes, without collecting or rerunning
pytest during this review. The output's recomputed raw digest equals COMPLETED's
`output_sha256`; its retained directory and suite match STARTED and the release.

## Frozen closure and unchanged source

The exact PRE-GREEN remains unchanged at SHA-256
`f8c9921e47e7ed64aa90b40d8deef640066ba6340b728f2a1c295c634a41d15d`.
STARTED's source map, COMPLETED's source map and independent current-source
rehashing contain exactly the same five entries and values:

| Source | SHA-256 |
| --- | --- |
| `backend/cad_guided_two_plane_context_v1.py` | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `tests/test_cad_guided_two_plane_context_v1.py` | `16a7e4467007f3a6c3034279c7b34221c2ccd26b51f7472c47ade6a8546cf5d1` |
| `scripts/check_cad_guided_measurement.py` | `628f3a84c98806f1d24c7b47f8d7683911a2f8bfb7f67dc8698887f1d8eda0f4` |
| `docs/research/2026-09-10_two_plane_context_proposal_design.md` | `f926605558aa31d87ffdcc527964b296cd5d8d7d75cc2bec1043a0af7ddde722` |
| `docs/research/2026-09-10_two_plane_context_v1_protocol.md` | `0e4929cf3c88d6b0071ae3cb6873434483eca7ddb67b918844f164d819446356` |

The separate harness/test identity fields also match these values. STARTED has
`STARTED_SYNTHETIC_CHECK_NO_OVERWRITE`; COMPLETED reports
`source_before_after_consistent=true`. The independent equality check confirms
that claim rather than relying on the Boolean alone. No live/frozen source or
prior attempt was changed in this audit.

## Resource and terminal checks

| Quantity | Retained observation | Frozen limit |
| --- | ---: | ---: |
| Parent monitored wall | 0.7444459586404264 s | 60 s |
| Worker CPU | 0.7146509999999999 s | 45 s |
| Worker reported high-water RSS | 60,211,200 B | 1,073,741,824 B |
| Sampled process-group peak RSS | 60,211,200 B | 1,073,741,824 B |
| Retained captured output | 209 B | 1,048,576 B |

STARTED and COMPLETED limits match exactly. Child exit is the integer zero,
`budget_stop=null`, and both worker resource fields are present and positive.
No output truncation, CPU termination, declared stop or source drift is observed.
Root's externally observed successful parent exit is explicitly distinguished
from the child exit stored in COMPLETED.

The unchanged supervisor limitations remain: monitored wall excludes some
setup/hash/receipt IO and cleanup; RSS is sampled, not an allocation limit or a
continuous peak guarantee; forwarding/reap are not independently hard-deadlined.
This reviewed pure test creates no subprocess of its own. The receipt does not
prove hostile or escaped-native-process containment, and this review grants no
such claim. No per-fixture geometry artifacts were required or emitted by this
protocol; the three retained receipts are the complete declared artifact set.

## What is qualified, and what is not

The exact frozen independent invented suite passes its helper-level contract:
proper central pose seeds, anchored plane equations and origin transformations,
explicit axial ambiguity, complete supplied support/extrema lineage, exact
deduplication, bounded work, detached snapshots and whole-result refusal on late
failure. Its asynchronous mutation and unexpected-exception controls remain
specific mocks; its 8192-slot test is preflight only, not a full-cap geometry
execution. The modest fixtures do not certify arbitrary-angle numerical accuracy
or large-input performance.

Every usable helper output remains a caller-scheduled seed, not finite support,
accepted registration, a measured bend, an observed count, a pose enclosure or
global uniqueness. The gate supports proceeding with separately reviewed
connected integration work. It does **not** qualify pair scheduling, a two-support
finite-query adapter, joint triplet/pair quotas, whole-part rank/ambiguity checks,
native geometry custody or scan-derived free-circle measurement. A new V11 source
draft outside this closure is neither tested nor approved by this result.

The original exact-.05-mm thin-part diagnostic remains 0/2 nominal outputs. The
terminal Gold11 V3 negative and its artifact-custody block are unchanged. This
pure pass establishes no physical output improvement, F1 gain, calibration,
unseen-family generalization, production permission or physical-run release.
Any connected/native/physical successor needs its own exact frozen scope and
independent pre-run audit. Do not replay this successful unchanged suite merely
to reproduce its result.

Review operations were bounded reads/hashes of the exact three owned receipts,
their immediate inventory, the five current sources and unchanged pre-audit,
plus stdlib JSON/receipt comparisons. No helper/test import, pytest collection,
native/inference/scorer execution, original physical/cache/Gold input access,
retry, cleanup or existing evidence mutation occurred. Only this new post-audit
document was authored.
