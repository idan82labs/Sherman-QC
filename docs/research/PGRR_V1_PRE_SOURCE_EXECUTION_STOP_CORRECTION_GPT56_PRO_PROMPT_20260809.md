# Sherman QC F1 — PGRR-V1 pre-source execution STOP and one-line correction ruling

You are the scientific and governance reviewer for the active Sherman QC F1 generalization goal. Please issue a decisive ruling, not a broad redesign.

## Prior authorization

You authorized exactly one gold-free tranche:

`PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_V1`

The complete ruling is preserved at:

`docs/research/PROFESSOR_F1_POST_CIIP_NEXT_ARCHITECTURE_GPT56_PRO_RESPONSE_20260809.txt`

SHA-256:

`591929292169b46087ade50b393a1adcf791f59e3a9b30f5ce7cb1df1b16b2a1`

## What happened

The immutable lock reproduced byte-for-byte, all nine source-free tests passed, and the claim was consumed before the first worker launch as required.

The first target-blind producer process then failed during Python import, before opening the raw source:

```text
Traceback (most recent call last):
  File ".../scripts/run_f1_pgrr_v1_candidate_worker.py", line 26, in <module>
    from research.f1_pgrr_v1 import geometry
  File ".../research/f1_pgrr_v1/geometry.py", line 22, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

The exact cause is one line in the orchestrator:

```python
python = Path(sys.executable).resolve()
```

The launch command began under `.venv/bin/python`, but resolving that symlink produced the base uv-managed interpreter path. Launching the base interpreter directly discarded the virtual-environment site-packages context, so NumPy was unavailable.

The proposed correction is exactly:

```python
python = Path(sys.executable)
```

No other source, configuration, threshold, pole, neighborhood, graph, handoff, causal, evidence, visual, resource, or STOP/GO rule would change.

## Authoritative failure evidence

- result: `output/research/pgrr-v1-c006-frozen-target-causal-preflight-v1/result.json`
  - SHA-256 `ff1c8fc37ef29fe41c672332d318afccf0077749f5891f1fea992da875fcea3f`
  - decision `PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP`
- process record: `output/research/pgrr-v1-c006-frozen-target-causal-preflight-v1/producer-a-process.json`
  - SHA-256 `9e83563de335ced04d22dbfaa1851da2c864ac2386c4c44155586e2f655c693b`
  - `return_code=1`
  - `stdout_bytes=0`
  - stderr SHA-256 `a6853b2da95c217d2d011499b3df7d7903d6a623e8155407ed67e4370f2f81df`
- consumed claim SHA-256: `8eff15c830197548c937b727b0e3ea42d064347487aaddcf7108033a1544b2d1`
- original lock SHA-256: `588d6d6072d115781dcad99196c2a0ed54a3e76993d6f309a55720dc2e19f48a`

## Capability and scientific-access accounting

All of the following are authoritatively false:

- source opened
- geometry started
- target ledger opened
- target handoff started
- any point/normal/pole/ribbon/chain constructed
- any PGRR count observed
- any parameter or threshold inspected against real data
- gold, labels, CAD, expected counts, benchmark feedback, runtime/main, or production access
- source-point serialization
- replacement input

Therefore the failure supplies no scientific evidence for or against `PROJECTIVE_GAUSS_RULING_RIBBON_V1`.

## Visual critique

The mandatory scientific images are all formally `NOT_PRESENT` because no source or geometry object existed. No synthetic substitute was created. The read-only critique is preserved at:

`output/research/pgrr-v1-c006-frozen-target-causal-preflight-v1/visual/visual_critique.md`

SHA-256:

`f14402f72ddf39206016e1c8bf89457e497dbf3bc413e8259456b02dcd291cbe`

The absence of images agrees with the pre-source failure but cannot support any architecture conclusion.

## Requested ruling

Choose exactly one:

1. Authorize one fresh correction execution under a new lock and claim, with only the one-line interpreter-path change above and a source-free regression that proves the worker imports NumPy through the exact unrevolved interpreter path before claim creation. The prior consumed claim, failure, lock, and STOP remain immutable and are not overwritten. The fresh run must keep every original PGRR parameter, input, role, resource ceiling, evidence schema, visual obligation, and no-rerun rule unchanged.

2. Issue a formal PGRR architecture STOP despite the source never having been opened, and explain the scientific basis for treating a pre-import infrastructure failure as architecture evidence.

If authorizing option 1, please end with exactly:

`AUTHORIZE_PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_CORRECTION_V2`

If stopping, please end with exactly:

`PGRR_V1_PRE_SOURCE_FAILURE_ARCHITECTURE_STOP`

Do not authorize any parameter change, alternate source, replay of the consumed claim, truth opening, benchmark, gold, CAD, expected-count access, runtime/main integration, production work, or synthetic scientific rescue.
