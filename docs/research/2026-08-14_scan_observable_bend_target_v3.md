# Scan-observable bend target V3

Status: frozen for the next optimization cycle on 2026-08-14.

## What changed

The F1 task is explicitly limited to what the scanner can observe. A positive is a visible, spatially supported fold-shaped transition with two material surface tails and a localizable finite axis. The detector is not asked to infer how the feature was manufactured.

Consequently, a monolithic bend and a ground/welded joint that produce the same admissible scan observation must receive the same target label. Asking identical input bytes to receive opposite labels is not a difficult generalization problem; it is an undefined target. That hidden-provenance requirement is removed from V3.

This is a clarification and versioning of the existing registry's `visible_observed_conventional_bends` semantics, not a rewrite of past results. The original registry, catalog, V16 report, and all prior research records remain immutable.

## Operational rule

A candidate is positive only when the scan itself supports:

1. two material surface tails;
2. a connecting fold-shaped transition; and
3. a finite bend axis in the raw scan frame.

Free edges, scan boundaries, openings, rims, unsupported plane intersections, and isolated normal/tessellation discontinuities are negative. A visually distinct seam can be negative when its observable geometry fails the fold definition. Provenance alone is never a negative feature.

## Evaluation firewall

- Inputs are renamed to opaque identifiers before inference.
- Part identity, filename meaning, V/X role, expected counts, CAD, drawings, and gold feedback are forbidden inference features.
- A manual native-resolution multiview axis precommit is frozen before predictions are opened.
- Every false positive and false negative is visually inspected at native resolution.
- Evaluator-known roles are disclosed and scored as diagnostic evidence, never independent transfer evidence.

The machine-readable contract is `data/bend_corpus/annotations/scan_observable_bend_target_v3.json`.

## Research consequence

The formal V21 nonidentifiability witness remains valid for the rejected hidden-provenance target. It no longer blocks the actual scan-observable bend-localization objective. V16 remains the frozen comparison baseline while new versions are evaluated against V3.
