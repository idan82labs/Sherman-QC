# V6 refusal helper first bounded gate

Date:2026-09-09. Pre-execution review GREEN for the pure helper route only.
The V5 initializer40/41 negative remains frozen and is not rerun.

## Exact four-file execution closure

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_report_envelope_v6.py` | `c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e` |
| `docs/research/2026-09-09_report_envelope_v6_protocol.md` | `4ebda90a4aefffd0d8c08bd4a2e2b7dfe5212342518a2dc379e1827a08812640` |
| `tests/test_cad_guided_report_envelope_v6.py` | `64f2b4e86636c7e1e7efa6695652bfb09b775da9343957e45e0ddfb38c2f373e` |
| `scripts/check_cad_guided_measurement.py` | `76a73cdb62ac0fb26247137fdd10607daf6c0e6c938030e687840bc11cc3dcee` |

Independent Astra source review and separate test authorship are complete.
Root fully read the helper/protocol and509-line test including final additions.
The307 prospective cases are source-counted, not yet executed. They independently
construct canonical envelopes, enforce every key/type/value boundary, retain
first-event/count evidence under repeated/nested overflow, require null/lower
bounds for unknown pools, and exercise finite byte-prefix and4KiB limits.
Hostile key equality is an ordinary failure, not an escaping exception; process
control exceptions propagate. Oversized headers refuse before key iteration.
No test imports native geometry, application startup, files or physical inputs.

Three source findings were repaired before execution: exact builtin schema keys,
header length before iteration, and exception-safe refusal-evidence collection.
No geometry/scoring threshold was changed. Root checked the new harness route:
only this fixed test, fresh isolated environment, no inherited native/capacity
opt-in, plugin/cache disabled and all four hashes before/after. Previous501cab
harness is archived byte-identically for its terminal V5 attempt.

Command: `.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite report_envelope_v6`.
Fixed60s wall,45s worker CPU,1 GiB sampled group RSS,1 MiB stdout; serial.
All307 intended cases required, no skips. First negative/stop/drift remains
terminal; no automatic retry or changed limit. This gate does not approve
connected geometry, full capacity, a physical rerun or product promotion.
