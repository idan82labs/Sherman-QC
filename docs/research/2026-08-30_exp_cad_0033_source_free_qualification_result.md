# EXP_CAD_0033 — source-free CF-PDG V5 qualification result

## Ruling

`PASS_ALL_PREREGISTERED_SOURCE_FREE_QUALIFICATION_GATES`

GPI: `GPI-C_POSITIVE_SOURCE_ONLY_EXECUTION_FEASIBILITY_NOT_F1_PROGRESS`

EXP33 repaired the qualification-protocol failure that terminally invalidated EXP32. Under a fresh identity, immutable 36-file lock, independent pre-lock review, detached post-lock approval, and durable one-attempt journal, the V5 execution-equivalent CF-PDG engine completed the frozen production-scale synthetic workload within every wall-time, memory, bundle, determinism, and equivalence gate.

This is positive engineering evidence only. No physical scan, evaluator artifact, label, expected count, quality role, bend-line score, or F1 metric was opened.

## Locked execution

- One authorized attempt; exactly 13 fresh children; 13 returned successfully.
- Three 1,000,000-row/member V5 source runs and three renderers.
- One 500,000-row/member scaling run.
- Three frozen-parent and three V5 runs at 100,000 rows/member.
- Exact workload per full run: 8,077 CAD nodes, 19,267 edges, 38 unique fits, all R00–R31, and 306,926 KD/local queries.
- Immutable lock: `cba7e2c9ad737c494f83378aff1deecae3098a51d75b30759d9cbb124885b3eb`.
- Outer result: `4b3173c548e618afc8d2f372721501d6cdf56913d84c9bc7821e37afb5194b1e`.
- Journal: `ac5563fe528d5028503408d0156673e0ab8f6c7f227db2fd007358a6bbf851f7`.

## Performance and replay

Full source wall times were 113.51, 113.31, and 113.03 seconds. Peak process-tree RSS was 3.35, 3.34, and 3.34 GB in decimal bytes, below the 4 GiB cap. Renderer times were 17.83–18.04 seconds, also below their 60-second cap.

The half-scale run took 76.96 seconds, giving a full-median/half ratio of 1.4723 against the 2.5 limit. Maximum source-plus-renderer bundle size was 308,302,663 logical bytes and 321,314,816 allocated bytes, below 2 GiB.

All three full semantic, visual, source-tree, renderer-tree, operation-counter, and render-manifest identities were exact. In the differential lane, frozen parent and V5 produced the same canonical semantic SHA (`796ec508...`) and visual SHA (`e5ff6ba3...`) in all six fresh processes. V5 median wall time was 28.138 seconds versus 32.782 seconds for the parent, a 1.165× parent/V5 ratio.

## Scientific interpretation

The synthetic scientific payload itself was `ABSTAIN / PAIR_SUPPORT_GATE`, with zero central components and no attribution. The native 3000×2400 diagnostic was reviewed at original resolution and truthfully showed the paired synthetic support, exclusions, folds/halo, and explicit absence of active components, attribution, matches, and finite local features. Its three render copies were byte-identical.

The all-abstain fixture means the production-scale result did not exercise nonempty components, attribution, or Dijkstra. Those paths remain covered only by source-free differential/unit fixtures. The two full-run internal `semantic_payload_logical_equivalence=false` and `visual_payload_logical_equivalence=false` fields indicate an unperformed in-run parent comparison; the authoritative six-process parent/V5 canonical-byte differential passed.

## Retain/next-step decision

Retain the V5 execution engine as source-free qualified. Do not promote it or reuse EXP31/32/33 as a physical run. Any physical evaluation must be a fresh successor with a new identity, exact validity-filter lineage, fresh approvals, unchanged thresholds/science, and the same fail-closed resource accounting.

No accuracy, F1, generalization, unseen-family, or production-readiness claim is authorized by EXP33.
