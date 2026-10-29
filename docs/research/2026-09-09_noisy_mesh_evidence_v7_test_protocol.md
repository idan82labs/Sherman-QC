# V7 scalar transport: unchanged geometry gate, corrected JSON boundary

Date:2026-09-09. Prospective repair after the terminal V6 gate:78 passed,
13 failed with `UNSUPPORTED_METADATA_VALUE`. No large or connected V6 run
followed. The exact V6 source/test/protocol and retained failure remain frozen.

## Mechanism and comparison

V5's numerical arithmetic retains exact `numpy.float64` scalars (including
terms derived from `numpy.finfo(float).eps`). Its JSON encoder serializes them
as ordinary binary64 numbers. V6's exact builtin-only encoder rejected them.
This was missed in source review and exposed by actual extraction controls.
V7 normalizes ONLY exact `numpy.float64` to builtin `float` at the encoding
boundary. Internal geometry, scalar types/values and lineage arrays stay exact.
Decoder input remains builtin JSON only. No float32/extended/complex/generic
NumPy or subclass conversion is introduced; finite numbers and signed zero
retain their binary64 value. No precision truncation or geometry retuning.

The new test `tests/test_cad_guided_noisy_mesh_evidence_v7.py` copies all91
controls from frozen V6 test
`223e28884e251c35a0dcc03d2250b89ed691338e0443dcb5b6bde8400f56ac45`.
Original fixtures, four positive noisy frames, five finite-support/refusal
variants, exact work/size/cap controls and all geometry gates remain unchanged.
The allowed, prospectively declared test differences are:

- Imported candidate/API becomes V7; wire schema grammar stays V6.
- The logical expected tree models V5's ACTUAL JSON serialization, normalizing
  exact NumPy float64 to builtin float. It no longer incorrectly promises to
  preserve a NumPy scalar class across JSON transport.
- The internal comparator additionally checks NumPy float64 byte identity;
  it still retains exact internal scalar classes and every array byte.
- The old encoder unsupported-np.float64 example becomes unsupported-np.float32;
  np.float64 is now explicitly admitted only by the encoder. The original V6
  refusal assertion remains in its frozen test, not edited retroactively.
- Nine new scalar controls:five finite values (positive/negative zero,.1,
  minimum subnormal,maximum finite),three nonfinite refusals,and one subclass
  conversion-hook refusal. Each finite value must match V5 JSON bytes and
  preserve binary64 bits; decoder still refuses an in-process NumPy scalar.

This gives100 prospective mandatory cases, no skips. Source/type correction,
not an altered geometry oracle, motivates the successor. The prior
[V6 test protocol](2026-09-09_noisy_mesh_evidence_v6_test_protocol.md) governs
every remaining fixture/budget/evidence detail unchanged. Root and independent
Astra reviewers check the exact source/test delta before execution.

## First bounded execution

Freeze the complete V5/V6/V7 source and protocol/test provenance plus actual
harness before dispatch. Serial100-case gate:60s wall,45s worker CPU,1 GiB
sampled process-group RSS,1 MiB output. No native/capacity flags, app imports,
physical data/Gold, email/held-outs or paid services. Retain every failure and
first terminal attempt; no hidden retry, cap change or altered running source.

A pass enables the separately reviewed unchanged large-scan capacity case;
it does not itself qualify large-scan resources, native measurements, physical
F1, observed counts, calibration or production. The full-size case remains
900,000 raw rows/1,800,000 triangles,unchanged noise and ten-face oracle.
