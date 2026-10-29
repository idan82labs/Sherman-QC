# Idempotent initialization refusal envelope V6

Date: 2026-09-09. Additive source only; no execution or promotion. The V5
initializer's retained 40/41 result, source, tests and protocol remain unchanged.
Both actual noisy geometry cases passed their frozen criteria; its report-overflow
control exposed a repeated-boundary evidence loss. This helper changes report
custody only, not geometry, acceptance thresholds, schedules or measurements.

## API and first boundary

`bound_initialization_report_v6(result, maximum)` is pure stdlib: no imports of
runtime/native geometry, file IO or data access. `maximum` must be an exact Python
integer from zero through 96 MiB, matching the V5 detailed-report maximum.
Invalid cap or non-dictionary input produces a refusal; booleans are not caps.

Ordinary dictionaries use finite, ASCII-escaped, sorted-key JSON encoding.
An encodable report within the cap remains unchanged. Overflow retains the
existing `INITIALIZATION_REPORT_BUDGET_EXCEEDED` reason. Ordinary encoder failures
use `INITIALIZATION_REPORT_SERIALIZATION_FAILURE`. Exception messages and dynamic
exception-class strings are never collected. `BaseException` is not swallowed.

The fixed refusal schema is `initialization_report_refusal_envelope_v6`.
It is independently bounded to 4 KiB: unavailable status, no transform, empty
hypothesis/start lists, false search completion/admission/production, unknown bend
count and indeterminate conformance. Its source schema is bounded to 96 safe ASCII
characters. At most three source reason codes of 180 safe ASCII characters survive;
malformed or longer source diagnostics set `source_reasons_complete=false`.
This bounded metadata does not authenticate a source or prove unseen execution.

Byte evidence is an **encoded prefix count**, capped at `maximum+1`; it is not a
fabricated complete-report byte size after early overflow or encoder failure.
The exact first limit, limit validity, prefix count and reason are retained.

## Exact canonical recognition and idempotence

A flag or version name is insufficient. Recognition requires the exact complete
key set; exact builtin dictionary/list/string/integer/bool types; every fixed
refusal value; empty lists and no pose; independently bounded source codes/counts;
and a validated fixed-key `first_refusal` record. Extra payload, malformed values,
success status or added poses prevent exemption. A report claiming the V6 schema
or a true discarded-whole flag but failing validation becomes
`INVALID_INITIALIZATION_REFUSAL_ENVELOPE`, even if it fits the detailed cap; its
pose or payload is never passed through as a refusal. Only the validated fixed
schema may exceed a zero detailed cap. Validation's exact compact JSON byte count
uses the already bounded safe-ASCII schema, avoiding general encoder re-entry.

For valid detailed caps, `B(B(report, cap), cap) == B(report, cap)`, including cap
zero. A canonical refusal is returned as a detached same-valued copy: caller
mutation of reason lists or the nested event cannot mutate the prior envelope.
The first discarded counts/reasons/byte evidence are neither reset nor added again.
Recognition is a bounded schema contract, not cryptographic authentication; a
caller who constructs the exact safe schema still supplies declared evidence.

## Outer registration and denominator semantics

Only fixed-depth known containers are inspected: root `hypotheses`, outer
`evidence.candidates`, and `evidence.initialization`. A canonical inner refusal
contributes its prior discarded counts. Its single earliest `first_refusal` event
is copied unchanged into the newly refused outer report. The outer report keeps
its own new reason, limit and prefix count separately. No recursive event chain
or arbitrary metadata payload is admitted.

Potentially aliased record pools combine using **max, never sum**. Counts describe
retained list lengths/prior canonical evidence, not how much unrecorded work might
have happened. The maximum is a conservative known pool size, not the cardinality
of a proven disjoint union. Missing retained-list keys yield zero with
`NO_RETAINED_LIST_PRESENT`. A present non-list container yields null with
`UNKNOWN_MALFORMED_CONTAINER`, not a known zero. If another known pool exists,
the known maximum survives with `KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER`.
Normal list/prior canonical counts use `RETAINED_RECORD_COUNT`. Exact integer
counts are bounded to `2**63-1`; booleans and negative counts are invalid.

## Prospective independent controls

- First overflow, equality at the exact detailed cap, cap zero and repeated
  two/three-pass idempotence; source and detached returned value remain distinct.
- Outer registration wrapping a canonical inner refusal: preserve earliest
  reason/byte event and known discarded hypotheses while recording outer candidates;
  repeated outer bounding must not add or erase counts.
- Every fixed key/value/type, pose, extra-field, invalid-count and source-code
  boundary: forged flagged successes never receive the small-envelope exemption.
- Malformed non-list hypotheses/candidates/evidence/initialization containers;
  explicit null or lower-bound status instead of laundering unknown into zero.
- Nonfinite/unserializable/circular reports and an ordinary encoder exception;
  no dynamic exception text and no swallowed process-control exception.
- Maximum canonical metadata below 4 KiB, exact cap type/range validation,
  nested-event depth restriction and preservation of the original V5 41 fixtures
  under the separately authored V6 initializer without changing their geometry.

First independent source/helper gate: root-owned serial supervision, **60s wall,
45s worker CPU, 1 GiB sampled group RSS, 1 MiB stdout**. This source document does
not execute or authorize native/physical work. Integration and execution are owned
by root and require independent source/test review before the new bounded run.
