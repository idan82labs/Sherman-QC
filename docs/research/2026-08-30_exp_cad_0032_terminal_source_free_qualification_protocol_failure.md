# EXP_CAD_0032 terminal source-free qualification protocol failure

## Ruling

EXP_CAD_0032 / CF-PDG V5 execution-equivalent is consumed and terminal. Two source-free production-scale attempts were launched before an immutable qualification lock existed. Attempt 1 failed import staging; attempt 2 returned from the optimized scientific call but failed strict semantic publication on a non-finite value. No same-identity patch, third run, replay, or serialization rescue is permitted.

The Generalization Progress Indicator is `GPI-C_NEGATIVE_SOURCE_FREE_QUALIFICATION_PROTOCOL_FAILURE_NOT_F1_PROGRESS`. This is not an accuracy, F1, physical-generalization, or production result.

## Attempt 1

- Result: `output/research/autoresearch-loop/exp-cad-0032-cfpdg-v5-production-scale-primary/EXP_CAD_0032_PERFORMANCE_RESULT.json`
- Result SHA-256: `c21b2106de1f60f4571e453b1ec905cb09f31c7e16c007b08d6c7a0f2d48cbc8`
- Failure: `ModuleNotFoundError: No module named 'scripts'`
- Source child: 1; renderer/replay/reference: 0
- Wall: 0.9006795829627663 seconds
- Peak process-tree RSS: 216,956,928 bytes
- Stderr SHA-256: `b7750d0c81bc83fde25b967668166ea179bf06fe0db619d990dc38de5d92f0a4`

## Attempt 2

- Result: `output/research/autoresearch-loop/exp-cad-0032-cfpdg-v5-production-scale-attempt2/EXP_CAD_0032_PERFORMANCE_RESULT.json`
- Result SHA-256: `700402e1ec39a26f982a18a32af32922224becd40962cbb03db52f3fe6ab7f55`
- Failure: `ValueError: non-finite value in canonical payload`
- Source child: 1; renderer/replay/reference: 0
- Wall: 95.42371887504123 seconds
- Peak/exact worker RSS: 3,352,788,992 bytes
- Stderr SHA-256: `75fa19eee1bce271329da7bbb8dd6a14df33f1a96996586a7b15ae5f3945d2c3`

Attempt 2's wall and RSS observations are individually below the preregistered 150-second and 4-GiB limits, but they are **not** gate passes: the process exited nonzero, only one run occurred, and no semantic, visual, operation-counter, replay, render, scaling, or frozen-parent differential artifact was published.

## Frozen source-free envelope

The two attempts used the same deterministic 1,000,000-row-per-member envelope:

- Manifest SHA-256: `1ded23ddb98e531f08c7d1e7aee331f0eb6dcfff48918d80320a61e7f58560b3`
- First points: `0c6596ed4000cd02655f22d34532cf53d877c8de38d50a2df291b4f02d2a41cf`
- First normals: `e672b992d059b09f147b92d745d0e4d47b175b9483e85657499a9536b69ebcfa`
- Second points: `b0a055bcb845b18cad0298e31369fb9e44eb1c67e04b4e2160d4f49591f6d264`
- Second normals: `b802aa087c61d7bdb79b90c8397868c118af8f027203c03b103487200649e8a0`

No physical scan, evaluator manifest, role, label, gold, expected count, filename-derived quality, or production datum was opened.

## Integrity boundary

Neither attempt used a complete immutable qualification lock, and their result files do not bind the exact executed code/preregistration hashes. Source files and the preregistration changed afterward. Therefore the attempt-time implementation is not fully reconstructible and cannot support qualification even apart from the publication failure.

The non-finite publication failure is a harness/protocol mismatch: the frozen parent publication path converts non-finite diagnostic floats to JSON `null` before strict canonical serialization, while the executed EXP32 worker rejected them. Correcting that behavior under EXP32 would be a post-result serialization rescue and is forbidden.

## Successor boundary

Only a fresh source-free identity may prospectively reproduce the exact frozen-parent null canonicalization and bind every consumed path to an immutable lock before execution. It must complete the full three-run production envelope, half-scale check, frozen-parent speed differential, renderer, full-tree equality, and resource gates. Any further protocol, equivalence, runtime, RSS, replay, or render failure retires CF-PDG; no physical successor is authorized by EXP32.
