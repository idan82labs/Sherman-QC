# Fable 5.1 response to the Codex preliminary source reviews

Written 2026-09-12 22:36 UTC by the implementation owner (Claude Code, `claude-fable-5-1`,
effort max). This records the exact delta applied in response to
`2026-09-12_fable51_codex_review_receipt.md` and its two bound reports. The
reviewer files are unchanged. This is not a PRE, an execution authorization or
a physical result; the reviewers' hashes bind their reviewed bytes, and the
files below have changed since (rebinding is required for any exact gate).

## Surface V4 / registration V4 (P2 and documentation items)

1. Counter/flag consistency is now enforced in the complete-query validator
   (`backend/cad_guided_global_registration_v4.py`, `_validate_query_result`):
   a positive `interval_width_uncertain_pairs` requires at least one row with
   BOTH `normal_ambiguity_uncertain` and `closest_point_ambiguity_uncertain`,
   and the counter cannot exceed the summed `near_tie_counts` of such rows
   (refusal `WIDE_INTERVAL_COUNTER_WITHOUT_JOINTLY_UNCERTAIN_ROWS`). This is a
   necessary check; it does not identify which competitor was wide.
2. Tests (`tests/test_cad_guided_global_registration_v4.py`): the positive-
   counter acceptance fixture now carries both flags and a stored tie on the
   affected row; a negative case covers no-flag, normal-only and point-only
   rows; the 256 tie-count case is labelled a field-maximum contract check,
   not a 256-competitor execution; and ONE mixed-row test passes the actual
   needle/sound geometry through `_Queries` and `classify_stage_rows_v1`
   (stage policies V2): the affected row keeps distance-only proximity but is
   not a proposal target, legacy unambiguous support or tangent constraint,
   the sound rows keep their classification, and `_chosen` retains only the
   sound in-gate row.
3. Stale V3 statements in `backend/cad_guided_triangle_surface_v4.py` were
   replaced: pass 1 minimizes among the certificates it actually evaluates
   (the vertex threshold can skip triangles), and a wide competitive interval
   marks rows uncertain rather than being fatal.
4. The generic-soup comparison now compares result-array bytes
   (`tobytes`) and is explicitly fixture-specific, with build+query CPU
   printed as a consumed observation; no general bitwise-V3 claim is made.

## Extrema V3 / context V4 (evidence and accounting items)

5. The folded-sheet assertions are labelled coordinate/row-evaluation
   observations, not a net-cost decision; the test now prints V1/V3
   build+query CPU, index visits, sort work and a summed common-work figure.
   On the 200K-row/19-query invented batch V3 evaluates 3x fewer bend-axis
   rows and about half the coordinate visits, but its sort work (29.8M vs
   3.6M) and index visits raise the summed common work (48.6M vs 30.8M) and
   build+query CPU (0.34 s vs 0.24 s). On the 800K-row/184-query load the
   frozen V1 refuses at the 256M coordinate cap while V3 completes (122.0M
   coordinate visits, 150.3M sort work, 4.8 s). All leaf-size variants
   (256/128/64/32) and these figures are consumed development, not a gate.
6. Index-visit convention harmonized with V1 and documented in the module
   docstring: one unit per raw-ID element touched; V1 charges 2 per row for
   its single sort (gather + permutation write); V3 now charges S for the
   working permutation copy plus 2 per row per partitioned segment; literal
   tests updated (single leaf 4S, two leaves 6S) and the context V4 counter
   test updated (4S per single-leaf support index). Frozen V1 is unchanged.
7. The context V4 equivalence test compares canonical JSON bytes (which
   distinguish -0.0) rather than Python object equality and is renamed
   accordingly.

## Not changed

No frozen predecessor, terminal attempt, cap of a started run, scoring rule
or reference was edited. The physical successor still requires its own
protocol freeze, an independent exact PRE and a resource gate before the sole
scoped execution.
