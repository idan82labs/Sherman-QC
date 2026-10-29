# V71 CLEC-MVT terminal retirement

Date: 2026-08-16

## Ruling and implementation

GPT-5.6 Pro selected one bounded successor to V70:
`V71_CANDIDATE_LOCAL_EXACT_CLOSURE_MULTISCALE_VARIATION_TRACK_V1`.
The complete ruling is preserved byte-exact at
`docs/research/2026-08-16_gpt56_pro_post_v70_sparse_field_ruling_verbatim.md`
(SHA-256 `cbec91eeab480ee0e725de0e6005e965c89d8f4fc98ae2fb3be312c2639e7e31`).

The implementation never invoked the V70 executable or its runner. It retained
the frozen V70 field, track, ownership, and NMS mathematics while replacing the
global field with candidate-local exact closure and explicit domain-exit
abstention.

## Source-free lock result

The locked 34-case suite passed once without a failure:

`34 passed in 294.24s`

This includes the unchanged 24 V70 fixtures and L01-L10 exact-closure,
domain-exit, cap, invariance, and dense/sparse-oracle fixtures.

## One-shot U01-U04 result

The authorized primary/replay preflight was executed exactly once. No third
run, replacement input, cap change, tuning, or rescue run was performed.

| Opaque scan | Primary | Replay | Primary wall | Replay wall | Peak RSS |
|---|---|---|---:|---:|---:|
| `S8BB74BDA675656176092` | `GRAPH_EDGE_VISIT_CAP` | `GRAPH_EDGE_VISIT_CAP` | 2.845 s | 2.856 s | 1.45 GB |
| `S9567D0D3F33D80F7C915` | `GRAPH_EDGE_VISIT_CAP` | `GRAPH_EDGE_VISIT_CAP` | 2.939 s | 2.958 s | 1.45 GB |
| `SDB1839480738AC85934F` | `Q0_CAP` | `Q0_CAP` | 2.334 s | 2.344 s | 2.77 GB |
| `S241AA95BFDFC185004F0` | `Q0_CAP` | `Q0_CAP` | 2.635 s | 2.633 s | 3.09 GB |

All four replay outcomes were byte-identical. All eight executions were well
inside the time and memory ceilings, but none reached a complete candidate
decision. Therefore the actionability gate failed before semantic or visual
adjudication.

## Terminal ruling

V71 is terminally retired. The failure is structural: on exposed real scan
density, a conservative candidate-local graph halo is not local enough. It
either visits more than 4,194,304 graph edges or includes more than 32,768 Q0
vertices before the exact sparse field can be evaluated. Raising those caps or
shrinking the domain after seeing this result would be a rescue tuning of the
consumed architecture and is not authorized.

Because V71 produced no committed modified candidate sets, there are no
changed native-resolution overlays to inspect. The visual gate was correctly
not opened. The reserved new physical scan remains sealed.

## Evidence

- Output root: `output/research/f1-v71-clec-mvt-u01-u04-actionability-v1-20260816`
- Summary SHA-256: `8e1bee172562c3f34a764189696081c8d61963980d6171d88dae7dfb4d2a1d29`
- Lock SHA-256: `e6b4ce4b61ed57255b93414f9f7fa1dda6c5d5f09880f57719d8ad87f934ba03`
- Implementation SHA-256: `a28f0da29debdf86bd4e1e094ebf96f1e247afc35227a2d5e7cee1c5f3afab23`
- Fixture SHA-256: `448546b15e062d8611dfa92affd7114900829a2e65bb39b5482f645676fdf9b6`
- Runner SHA-256: `4a9b5009c15c80ba01296f9f39ad5e9a65b0495bc0bf484ca75210b6b161b9e8`

