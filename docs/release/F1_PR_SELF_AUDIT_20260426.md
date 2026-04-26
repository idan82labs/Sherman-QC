# F1 PR Self-Audit - 2026-04-26

## Purpose

This note records the final pre-review safety audit for the F1 draft PR.

The main question was whether the branch accidentally promotes raw F1 output into runtime behavior. The answer from the audit is no.

## Runtime Import Audit

Search performed:

`rg -n "patch_graph_latent_decomposition|run_patch_graph_latent_decomposition|run_patch_graph_f1" backend domains frontend -S`

Result:

- Only `backend/patch_graph_latent_decomposition.py` defines `run_patch_graph_latent_decomposition(...)`.
- No runtime backend pipeline, domain service, or frontend file imports or calls the F1 solver.

Interpretation:

- F1 remains offline/research-only.
- Runtime bend inspection does not execute the F1 patch-graph solver.
- Product diagnostics only expose structural scan/CAD profile payloads and scan-only overlay rendering support.

## Promotion Logic Audit

Promotion and candidate-source logic is confined to offline validation scripts:

- `scripts/run_patch_graph_f1_prototype.py`
- `scripts/run_f1_accuracy_validation.py`
- related tests and review-package scripts

Important guards:

- raw F1 singleton candidates are tracked as manual-review candidates, not automatic runtime promotions;
- partial scans are scored as visible observed-bend validation targets, not whole-part totals;
- render semantics distinguish `owned_regions_match_accepted_exact` from `scan_context_only_raw_f1_debug_not_final`;
- render-only clustered marker suppression does not change solver/count ownership.

## Runtime Diagnostics Audit

Runtime product-path changes are limited to diagnostic payloads:

- CAD structural target metadata;
- scan structural hypothesis metadata;
- profile selector diagnostics;
- scan-only renderable bend object display support.

These diagnostics do not replace CAD/global bend matching decisions or runtime expected-count logic with F1 output.

## Focused Verification

Commands:

`PYTHONPATH="$(pwd):$(pwd)/backend:$(pwd)/scripts" /opt/homebrew/bin/pytest -q tests/test_run_patch_graph_f1_prototype.py tests/test_run_f1_accuracy_validation.py tests/test_part_profiling.py backend/tests/test_bend_inspection_run_details.py`

Result:

- `49 passed`

Frontend typecheck:

`npm run typecheck`

Result:

- passed

## Remaining Review Warnings

- This is still a large branch and should be reviewed by slice.
- Raw F1 still undercounts `47959001` and `10839000`.
- Raw F1 exact candidates `48991006` and `49024000` still require manual visual review.
- Product diagnostics / scan overlay should be reviewed separately from F1 algorithmic work.
