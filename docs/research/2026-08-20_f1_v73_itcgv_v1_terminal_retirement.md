# V73 ITCGV V1 terminal retirement

Date: 2026-08-20

## Disposition

`V73_INTEGRAL_TRANSVERSE_COVARIANCE_GROWTH_VETO_V1` is terminally retired.

The single authorized source-free lock invocation completed its primary process with 31 passed cases and one failed case out of 32. Per the frozen protocol, the replay was not opened and U01-U04, native-resolution adjudication, and the reserved physical scan remain sealed. V73 must not be corrected, rerun, wrapped, or rescued.

Authoritative machine result:

- `output/research/f1-v73-itcgv-source-free-lock-20260820/summary.json`
- `source_free_lock_passed: false`
- `terminal_disposition: V73_TERMINAL_RETIRE`
- primary process exit code: `1`
- primary result: `31 passed, 1 failed, 0 errors`
- replay: `null`

## Exact failure

Case G24 is a canonical-geometry invariance case. It presents the same rounded surface and physical segment as G07 while reversing the candidate axis and shifting the anchor.

The primary process failed at:

```text
assert evaluations[0].geometry_sha256 == ref_values[0].geometry_sha256
observed: 2c53f07a3bb59f7fed3ad0b12bf14255079b308aec6fd4b1200964ecea066a89
expected: dd62f37fb2ea2a4cc365bd3ee958409d98ecd59c218a796f62fa5f927b173587
```

The failure is a geometry-identity/canonicalization defect, not evidence from a physical scan. The decision signature comparison was never reached because the geometry identity assertion failed first. It is nevertheless a lock failure and therefore terminal under the precommitted rules.

## Scientific meaning

V73 produced no physical-generalization result. Passing 31 source-free fixture cases establishes substantial implementation coverage, but it does not authorize interpreting V73 as effective, ineffective, precise, or robust on real parts. The correct conclusion is narrower: the frozen V73 implementation did not satisfy its own invariance contract.

## Next boundary

No V74-style architecture is funded from this result alone. The next phase is measurement repair and acquisition of an untouched, independently annotated physical evaluation corpus. This prevents another synthetic/source-free qualification result from being mistaken for progress on the actual F1 generalization objective.
