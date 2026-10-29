# Gold11 runner V2 — prospective pure gate audit

Date: 2026-09-09. No physical run/freeze or native execution.

The runner's V8 candidate completed its independent pre/post-audited unchanged
four-case connected gate4/4. This gate checks only orchestration and source/mock
contracts:89 mandatory passes plus one unconditional disabled-runtime skip.
Budget60s wall/45s worker CPU/1GiB sampled group RSS/1MiB stdout, one serial
first attempt. No application/native imports or real subprocesses in these tests.
Root read the full runner, tests and test protocol and their V8-only deltas.

## Frozen six-file source closure

| File | SHA-256 |
| --- | --- |
| `scripts/check_cad_guided_measurement.py` | `6d58919f8b5582ec7611751b6ca262c7060a0a099b015dcd644965852f17632d` |
| `scripts/run_gold11_mesh_development_v1.py` | `100b9dc7dfcb5dfe6a3f4bdd77b03e375ff878278efdf9524d0582299a856837` |
| `scripts/run_gold11_mesh_development_v2.py` | `94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422` |
| `tests/test_gold11_mesh_runner_v1.py` | `b963ec4de8efdb00ba14cc828dcdc97627e669a14784f70cdfff3ce7a231c3fa` |
| `tests/test_gold11_mesh_runner_v2.py` | `d13df5c2eec485fdc4640cd1ba0f8f20d40e4aca3869e23e85eeb63edccfce63` |
| `docs/research/2026-09-09_gold11_mesh_runner_v2_test_protocol.md` | `ef2e7c4b72cbefde98c1e83ae887c27a2c6636082db5e446e4cafa557475c4f6` |

The physical protocol and actual backend runtime closure are not read by this
pure route; naming sources in inert constants is not permission to execute them.
The predecessor test is an extra provenance binding, not a collected suite.
Scorer functions/common config remain unchanged; new128MiB full-result transport
applies equally to both arms, while8MiB scoring projection stays fixed.

## Independent release

Independent reviewer `/root/cad_measurement_integration_review` read the complete
1100-line runner, full test/protocols, exact V1-to-V2 runner diff and named harness
closure/isolation/limits route. Every six-file hash above matches. Decision:
GREEN FOR ONE FIRST PURE89-PASS/1-UNCONDITIONAL-SKIP GATE under the stated caps.
No runtime/physical authorization follows. Any failure/stop/drift remains terminal.

Before physical freeze, the separately owned physical protocol must describe
the inherited phase-specific wall enforcement: worker allowance is
max(.01,180-preparation elapsed), parent IO uses20s regions, and coordinator/
whole-run guards remain. It is not a hard180s whole-case postprocessing bound.
Accurate prospective wording is accepted; no runner change is needed for this
pure gate, and no started experiment's limits are altered.

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_mesh_runner_v2
```

## Terminal result and independent post-check

`cad_engine_v1_synthetic_checks/gold11_mesh_runner_v2-0s3nl07n` passed89,
with exactly one unconditional runtime skip. Parent0.6084648752585053s,
workerCPU0.574698s,sampled53,870,592B,workerpeak53,854,208B.
No60/45/1GiB/1MiB stop; all six source hashes agree before/after and with the
table. OUTPUT SHA`512b84bcda96dc2ddec4b4e06e8678628f3351ae61360480a30d93c8b87b80ab`.
Independent reviewer verified all STARTED/OUTPUT/COMPLETED records and exact
hashes: POST-CHECK GREEN FOR PURE ORCHESTRATION ONLY. Printed inference/terminal
statuses belong to invented mocks, not actual physical inference.

The unfrozen physical protocol's accurate phase-specific wall wording is also
independently accepted at SHA`a4c97ffcc34a32dbf6436e65d9316db6c7bc5adafbdf0852f13abab3e636c977`.
Before adding the separate tiny runtime route, root archived the exact used
harness6d589 at `source_snapshots/all_generated_v8_6d58919f/` and verified its
byte-identical SHA. No rerun of this successful suite is needed for additive
route wiring. The runner/test/protocol themselves remain frozen at this table.
Guarded-runtime qualification and a new exact physical bundle audit remain.
