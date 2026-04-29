# Patch-Graph F1 Prototype

## Goal

Build an offline-only research lane that treats the visible scan as a typed latent decomposition over connected surface atoms:

- flange instances
- bend instances
- residual / outlier support

This tranche does not replace the current zero-shot runtime. The current engine remains the production/control path.

## Scope

F1 should:

- build a surface-atom graph from the visible scan
- generate flange and bend hypothesis pools from detector and strip seeds
- solve a deterministic typed labeling problem over atoms
- count connected bend components from the labeled support
- emit owned bend regions and render/debug artifacts
- compare its output directly against the current zero-shot control on the focused hard-scan slice

F1 should not:

- mutate runtime zero-shot APIs
- replace selector policy logic in production
- claim full posterior inference

## Focus Benchmark

Use:

- `data/bend_corpus/evaluation_slices/patch_graph_f1_focus.json`

Watch cases:

- `10839000` must stay exact `10`
- `47959001` must stay exact `6`
- `ysherman` must not broaden
- `49204020_04` must not broaden
- `48963008` and `48991006` are secondary hard cases

## Artifacts

Each run should produce:

- atom graph payload
- flange and bend hypotheses
- typed assignment
- owned bend regions
- static overview and focused render artifacts
- control-vs-candidate count comparison

## Success Rule

F1 is only candidate-green if it improves at least one hard full-scan case without worsening the anchor cases or broadening the repeated near-90 canaries.
