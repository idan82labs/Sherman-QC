# F1 V62 distributed-turn consensus — development result

Date: 2026-08-15

## Status

V62 is a frozen development candidate, not retained evidence. It was designed and executed after V3 truth had been opened, so V3 can only show whether the mechanism is worth a fresh prospective test.

V62 preserves every V61 prediction and adds only source-only lines supported by cumulative normal rotation across wider physical banks. Its fixed gates are: near-bank turn 5–15 degrees, wider-bank cumulative turn 16–80 degrees, concentration 0.18–0.75, at least four supporting views, and strict duplicate suppression against V61.

## Development outcome on consumed V3

| Route | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| V61 | 64 | 23 | 184 | 0.7356 | 0.2581 | 0.3821 |
| V62 | 72 | 27 | 176 | 0.7273 | 0.2903 | 0.4150 |

V62 produced 12 deterministic births: 8 TP and 4 FP. The net F1 gain is +0.0329. Primary/replay semantic identity passed for all 30 cases.

All eight changed cases were visually inspected after scoring. The eight true additions coincide with previously missed low-concentration or rounded transitions. All four false V62 births occur in O004 (three) and O019 (one). They are geometrically plausible neighboring facet boundaries rather than floating lines, but they are not counted as the target bend under the frozen evaluator. This is an ownership/semantic precision risk that a prospective corpus must adjudicate.

## Decision

Freeze this exact V62 specification and fund one fresh prospective V4 test with genuinely new complex profiles. Do not tune V62 further on V3. Retain V61 as the current research candidate until V62 passes or fails V4.

## Artifact hashes

- V62 implementation: `b2a04d9d9cad5c3a40bea0c396092e6f1c68a6c6db33430afb89086630e16817`
- V62 runner: `785a0d31357936aa1e0a584170e69c2b8c732ec09d8f7cba93f3577c1ed9fe08`
- V62 scorer: `4bf6609dcace00a3833b787a6594a03b703905b0184b84277570fe6b1590f8a6`
- V62 execution manifest: `ae1d6c29fb566b208e339cfa984dc98351fcad022401d908096a451555a9db85`
- V62 development score: `390bfaf096c18e21510422938efa8515c729993140b8f22d68bfa58714a9624d`
- Aggregate scored-overlay hash: `1bd5990e0ed6b374141e9c1cceb228c52cb12aa1554375f3e4db63b80a6ce0cf`
