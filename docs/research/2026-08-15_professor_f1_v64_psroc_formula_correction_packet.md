# Sherman QC F1 — V64 PSROC formula correction

Date: 2026-08-15

The prior post-V63 ruling selected `V64_PAIRED_SMOOTH_RIBBON_OWNERSHIP_CERTIFICATE_V1` (`PSROC_V1`) with `RULING: EXECUTE_ONE`. The complete rendered response has been preserved at:

`docs/research/2026-08-15_gpt56_sol_pro_post_v63_ruling_verbatim.md`

One required table cell is incomplete in both the visible DOM and copied text. Under Section 6, the row renders only:

`Largest single inner increment (\max`

The frozen inequality and right-hand-side constant are absent. All other supplied constants and protocol requirements remain unchanged.

Return only:

1. the complete mathematical definition for the **Largest single inner increment** row, including the exact index range, comparison operator, units, and right-hand side; and
2. one sentence confirming whether this correction changes any other PSROC_V1 constant or rule.

Do not redesign the architecture, propose alternatives, relax a gate, add a tuning stage, or authorize execution until the missing constant is unambiguous.

## Resolution

Resolved by GPT-5.6 Sol Pro. The complete answer is preserved in
`2026-08-15_gpt56_sol_pro_v64_psroc_formula_correction_verbatim.md`; the
executable lock is recorded in
`2026-08-15_v64_psroc_locked_specification_correction.md`.
