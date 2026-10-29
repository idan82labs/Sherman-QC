# Cached-control qualification V2: exact independent release

Date:2026-09-10. Independent `/root/cad_measurement_integration_review` verified
the [eight-file freeze](2026-09-10_gold11_cached_control_v1_qualification_v2_pre_run_audit.md),
SHA256 `359f160ad639843677fc1bc55dac8ab5dc0557ce38df88003e92d57988aa3712`,
and exact archived predecessor harness. Full test diff and all five added controls
were reviewed: one consuming iterator/context manager, real owned-directory
accounting, original assertions/parameters preserved by exact SHA/AST guard.
Runtime75af and failed predecessor bytes are unchanged. Harness changes are
limited to routes/bindings/budget-retention membership and the V2 smoke flag.

Pre-GREEN for one source gate:141expected passes/one declared smoke skip.
Only independent retained source post-GREEN with unchanged eight files releases
the142-outcome smoke gate once. Budgets/custody/failure policy unchanged.
No actual cached/Gold access or actual source-bundle freeze is released yet.
