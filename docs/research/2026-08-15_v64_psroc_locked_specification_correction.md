# V64 PSROC locked specification correction

Date: 2026-08-15

The original `EXECUTE_ONE` ruling remains preserved verbatim in
`2026-08-15_gpt56_sol_pro_post_v63_ruling_verbatim.md`. The formula-only
correction remains preserved verbatim in
`2026-08-15_gpt56_sol_pro_v64_psroc_formula_correction_verbatim.md`.

The incomplete Section 6 row is replaced in the executable lock by exactly:

\[
\max_{10 \le j \le 49} |\delta_j| \le \max(6^\circ, 0.18T),
\qquad \delta_j = \phi_{j+1} - \phi_j,
\]

where both \(|\delta_j|\) and \(T\) are measured in degrees.

The correction changes no other `PSROC_V1` constant, definition, gate,
ordering rule, abstention condition, or execution requirement.
