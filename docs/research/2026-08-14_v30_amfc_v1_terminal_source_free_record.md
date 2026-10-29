# V30 AMFC V1 terminal source-free record

Date: 2026-08-14

Objective: preserve the active Sherman QC F1 zero-shot bend generalization objective while testing the professor-authorized `V30_AXIAL_MULTIPATH_FOLD_CERTIFICATE_V1` source-only rescue sidecar.

## Frozen execution outcome

- Gate 0: passed, 22/22 static checks; mechanism SHA-256 `e29a263982188203db885a1a7433e017361786e08d2e4ec055cb89ae94233c95`.
- Gate 1: passed after separately versioned pre-invocation renderer corrections; harness version 4; 10/10 analytic fixtures valid; 10/10 raw 4096×4096 visuals manually inspected and signed; zero prior detector invocations.
- Gate 2: failed on the first and only executed fixture, `SF01_FOLD90_MESH`.
- Expected: one rescue, no rejection reason.
- Observed: zero rescues, reason `AXIAL_COVERAGE`.
- Invocation count at terminal stop: 1.
- Terminal token: `V30_AMFC_SOURCE_FREE_MECHANISM_RETIRE`.
- No rerun, threshold change, fixture replacement, or V30 rescue tuning is authorized or performed.

## Causal accounting

The engine found 72 locally valid paths in total: eight at each of the nine axial stations. Nevertheless, every station was invalid because the accepted path midpoints did not span the frozen minimum of 4s within that station. The median turn error was approximately 5.706041 degrees, so turn agreement was not the failure. The mechanism visually and numerically reached the physical fold, but its minimum-cost node-disjoint paths collapsed into a narrow axial band and could not certify spatially distributed evidence.

This rules out V30 AMFC V1 as frozen. It does not show that tangent-graph evidence is useless; it shows that the selected certificate/optimization primitive cannot guarantee the required within-station axial dispersion without a mechanism change. Any successor must be a new architecture with a new lock and must not be represented as a V30 correction.

## Visual inspection

The 4096×4096 diagnostic overlay was inspected at original resolution. It shows:

- physically correct continuous 90-degree rounded source geometry;
- tail/graph evidence at all nine intended station regions;
- certificate evidence following the actual bend rather than a detached artifact;
- repeated narrow within-station clustering, consistent with the `AXIAL_COVERAGE` rejection;
- no final rescued line.

Artifact: `output/research/f1-v30-amfc-v1/source_free_results/base/SF01_FOLD90_MESH/base/diagnostic.png`

Result SHA-256: `b91c10c516b1a8df26357998e1102941ffe8fc8d18197814674f6880200d78e6`

Diagnostic visual SHA-256: `2e424b2586821ea6533c7dcc4282316ec54721678c169ce92ede97be553e162e`
