# V4 pure custody gate — prospective freeze, 2026-09-10

Engineering reason: V3 retained44 unexpected temporary artifacts. The new V4
draft must bound complete-tree inventory globally and preserve deadline escape
through an artifact helper that catches ordinary exceptions. This is not a
retry of V3 or qualification of the complete V4 runner.

Candidate runner SHA256:
`559095a685d6ae5badecdb9704952d5e80304dcd728f8afa0f131a25707ebd85`.
Test SHA256:
`68defd348ceced0246f5aa8f0daa637ced09a1263365a8b54f6db6a53f26ad8d`.
Qualified helper:
`3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838`.
Unchanged terminal V3 predecessor:
`77bbcffbf10f05a22d0c72c8a645112e489469b4af92b4dbba3ca6243fcecea7`.

Read only the six named source/protocol files bound by
`scripts/check_gold11_mesh_v4_custody.py`. Independent PRE must bind this exact
harness and protocol before execution. The worker evaluates selected AST
definitions, the hash-bound stdlib helper and tiny invented pytest-owned files.
No raw/Gold/cached prediction, app/server/native module, runner import or runner
dispatch is in this gate. Signals/schema/neutral validation are explicit mocks.

Exactly14 test functions /22 parameterized outcomes, no skips expected. Pass
requires all22 tests, no budget/supervision failure, stopped/empty process group,
and identical six-file pre/post hashes. Every failure stays terminal; a successor
needs a specific repair, new freeze and PRE, not an unchanged retry.

Budget: one worker;45 CPU seconds;55 work wall seconds plus5 seconds reserved
for reaping/group checks inside60 seconds; sampled group RSS1GiB;stdout1MiB.
Numeric threads1, bytecode/plugins disabled, isolated Python/site loading.
The additive harness reuses the reviewed V3 supervisor mechanism. Setup/source
hashing, private-cache creation and final receipt publication are outside its
supervised work timer; this is a bounded engineering gate, not hard real-time
or full-runner custody qualification. Cache remains retained, never silently
cleaned on custody failure. No limit may change after STARTED.

Fresh unique attempt directory retains STARTED.json, OUTPUT.txt, COMPLETED.json;
the worker's tiny invented files remain in its separately retained private cache.
Independent POST checks exact source hashes,22 outcomes, resource/termination
receipts and this limited claim. Full connected runner, actual timer delivery,
schema/leaf validators, whole candidate dispatch and Gold11 remain unqualified.

Prospective command, not permission before PRE:

```sh
.venv/bin/python -I -S -B scripts/check_gold11_mesh_v4_custody.py
```
