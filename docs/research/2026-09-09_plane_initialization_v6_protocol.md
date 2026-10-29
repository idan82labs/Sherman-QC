# V6 connected initializer: unchanged geometry, lossless evidence and refusal

Date:2026-09-09. Prospective source/synthetic successor, not executed. Frozen V5
initializer40-pass/1-fail remains terminal. Both real noisy poses passed; the
last check exposed double report wrapping that erased discarded hypotheses.

## Narrow implementation change

`backend/cad_guided_plane_initialization_v6.py` copies frozen V5 initializer
`c37d2bfa916914c64af74e9edd5638756131914600222c63fe8bf77cfd71d2ee`.
All input ownership, digests, finite support queries, schedules, candidate
refinement, numerical-failure veto, ambiguity and registration gates remain.
Its distinct V6 budget type has exactly the same fields/defaults/hard maxima.
The only functional substitutions are the V7 extractor/compact metadata export
and the V6 idempotent report boundary at all three prior bounding sites.
Versioned API/schema/error names are explicit; V5 source is never monkeypatched.

V5 noisy input/configuration/math helpers and V5 plane search remain frozen.
The compact successor retains all internal raw arrays and uses schema-table evidence only
for diagnostic metadata. Its changed byte metric and separate work limits are
governed by [the codec protocol](2026-09-09_noisy_mesh_evidence_v7_protocol.md).
No compact payload supplied by a caller is admitted as support authority.
The [refusal protocol](2026-09-09_report_envelope_v6_protocol.md) preserves first
discarded evidence across inner/outer boundaries, independent of a100-byte cap.

Pre-execution amendment: the unexecuted initializer was retargeted from V6 to
V7 compact extraction after the separate V6 codec gate had78 passed/13 failed. V7 repairs
exact NumPy float64 JSON transport, not geometry. This initializer has not been
run or retried; all executed V5/V6 codec evidence stays unchanged. Require the
new100-case V7 gate, not the failed V6 source approval, before its first run.

## Finite gate and unchanged independent constructions

`tests/test_cad_guided_plane_initialization_v6.py` mechanically carries all41
controls from frozen V5 test
`b1031fda243055cd38b7f35146912a186702d77e62fe590989e8b4981cb3de60`.
Only imported module, public API, budget class and versioned cap-error names
change. No original fixture, assertion, noise, tolerance, policy or skip changes.
The V5 protocol
`593833164918b8dbc555a2b1620c5cc53681f1be5b56cc5d17eec4fba653fb60`
still specifies the unchanged geometry contract; V6 supersedes only representation
and refusal behavior. V5 tests/source are not rerun by this route.

Two additional controls require actual compact patch metadata to decode into
complete geometry evidence, and actual registration with a100-byte report cap
to preserve the inner discarded hypothesis count/first prefix on outer refusal.
This gives43 intended cases. All are mandatory, no skips. Private mocked tests
remain explicitly scoped; the two real noisy pose positives have no mocks.
Their original start1e-7mm, clean pose.01mm and noisy-row.011mm ceilings remain.
Default global registration thresholds and selected candidate completeness stay
unchanged. No successful measurement or physical F1 is inferred from registration.

Before first execution require independently reviewed codec/helper controls and
independent source/test-delta audit. Root freezes all called sources, protocols,
test and harness. The serial gate is60s wall,45s worker CPU,1 GiB sampled group
RSS,1 MiB output. Original runtime isolation and source-drift/failure retention
apply. A first negative is terminal; no automatic retry or relaxed assertion.
No native kernel, physical file/Gold, email, held-out, server or paid service is
accessed. This is not approval of the later byte/native/measurement workflow or
of an all-case physical benchmark; those require their connected frozen gates.
