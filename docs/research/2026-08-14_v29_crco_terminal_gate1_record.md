# V29 CRCO_V1 terminal Gate-1 record

- Profile: `V29_CALIBRATED_RANGE_CONTINUITY_OWNERSHIP_V1`
- Gate 0: passed 16/16 static checks.
- Gate 0 manifest SHA-256: `009685d0889a7e41a317a295ee0df0ac6950ec021699ad179b9c38503d92b435`
- Gate 1: terminal failure before fixture serialization and before any detector call.
- Terminal token: `V29_CRCO_TERMINAL_RETIRE`
- Failed fixture: `SF01_OCCLUDED_90_FOLD`
- Failure: the source-free analytic range builder found incompatible synthetic depths at pixel `(68, 166)` while constructing its continuity path.
- Detector invoked: no.
- Fixture files written: zero.
- Historical data, gold, labels, CAD, drawings, benchmark feedback, runtime/main, and production opened or modified: no.
- Rerun, patch rescue, replacement fixture, or V29b: not authorized and not performed.

Exact exception:

```text
RuntimeError: contradictory synthetic range at (68, 166)
```

Frozen source hashes are preserved in `output/research/f1-v29-crco-v1/precommit_manifest.json`.
