# Addendum to the Gold11 V8 independent physical POST - correction of one paired-comparison statement

Written 2026-09-13T03:06:54Z (`date -u`). Reviewer: the same independent AI
post-run auditor (Claude Fable 5.1, reviewer role). This addendum corrects the
original POST without modifying it. The original is preserved verbatim at
`docs/research/2026-09-13_gold11_mesh_development_v8_post_run_audit.md`,
44587 bytes, SHA-256
`894dfb8a606a02825e4732883449352220b086c5897e76c5ba953a218fca10c1`
(re-hashed now; identical to the copy in the job tmp directory). Read-only, as
before: only receipts (`COMPLETED.json`), retained projections
(`SCORING_INPUT.json`), leaf logs (`OUTPUT.txt`) and `FROZEN.json` under the
attempt were re-read for this check; no RESULT.json was re-parsed, no raw
data, Gold/reference container or private cache was touched, and no code was
executed. This file is the sole write.

## 1. Sentences corrected (quoted verbatim from the original)

Original, section "Paired comparison with V7", first observation bullet (lines
497-505):

> Every one of those ten slots then failed the unchanged finite prefilter with
> zero passing starts, i.e. the blocking stage moved downstream; V8 did not
> turn any of them into a registration candidate.

Original, section "Findings", item 5 (lines 566-568):

> 5. The ten V7 extrema-coordinate refusals all complete under context V4 but
>    every one lands in the "no passing start" class; the capacity bottleneck
>    moved downstream. The V1/V4 coordinate counters are different units.

Both sentences are wrong for two of the ten slots. The same wrong sentence was
repeated in the auditor's chat relay to the coordinator ("but every one lands
in 'no passing start'"); that relay is corrected by this addendum as well.

Corrected statement: of the ten V7 extrema-coordinate refusal slots
(CG08506AE97D, CG1E0BBA2933, CG84C8FA12D2, CGEBA9A869AC, CGF41F49066A; both
arms), all ten completed their two-plane context under context V4, after
which:

- eight slots failed the unchanged finite prefilter with zero passing start
  groups (PF): CG08506AE97D control and challenger, CG84C8FA12D2 control and
  challenger, CGEBA9A869AC control and challenger, CGF41F49066A control and
  challenger (`exact_passing_start_group_count` 0, reported phase
  `WHOLE_PART_REGISTRATION`, reason `NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS`);
- two slots produced one passing start group each, one attempted registration
  candidate each, a `SUPPORTED_CONTEXT` admitted under the provisional 1.0 mm
  gate (context p95 upper 0.9008134371607799 mm), one scheduled measurement
  context, and were then refused by the frozen measurement consumer in every
  window (MC): CG1E0BBA2933 control and challenger (`exact_passing_start_group_count`
  1, `registration_context_count` 1, `source_context_window_count` 13, reported
  phase `COMPLETE`, reason `NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`, zero
  predictions).

So V8 did turn two of the ten former extrema refusals into an attempted (not
accepted) registration candidate and a scheduled measurement context; none of
the ten produced an emitted support. The rest of the original bullet (context
completion, counter-unit caveat) and the corrected Findings item stand as
written apart from the quoted sentences.

## 2. Dependence of other counts and tables

Re-derived from the 22 receipts and 22 retained projections (exit code,
`reported_workflow_phase`, `reported_workflow_reasons`, projection diagnostics,
and the receipts' `workflow_stage_diagnostics.initialization_work` values):

- Immediate-cause counts: PF 12, MW 2, MC 3, RG 1, CPU 4 (sum 22). These equal
  the original "Immediate causes" table and the original paired-comparison
  class table (EC 10 -> 0, PF 4 -> 12, MW 2 -> 2, IW 2 -> 0, NT 2 -> 0, RG 1 ->
  1, CPU 1 -> 4, MC 0 -> 3). The arithmetic behind them was already the
  corrected mapping: PF 12 = 4 unchanged (CG64C412CC73, CG65CD575B1C; both
  arms) + 8 former EC; MC 3 = 2 former EC (CG1E0BBA2933; both arms) + 1 former
  RG (CG4BDD1371B6 control). Not affected.
- The original per-slot V7 -> V8 transition list already stated
  "CG1E0 C EC 31.674 -> MC 356.501, H EC 48.648 -> MC 392.212" and lists the
  other eight former EC slots as PF. Not affected.
- The original Findings items 3 and 4 already describe CG1E0BBA2933 control and
  challenger as MC slots with an admitted context; they are consistent with the
  correction. Not affected.
- The original "All 22 slots table": all 22 rows (exit code, leaf wall, parent
  wall, worker CPU and high-water RSS for the 18 returned workers, sampler
  record count, log bytes, RESULT bytes and hash prefix) equal the values
  re-derived now from the receipts and logs; for the four CPU-terminated slots
  the "~" fields equal the last sampled values (CPU 1860.224 / 1860.247 /
  1860.196 / 1860.329 s; high-water RSS 1417363456 / 1639366656 / 513409024 /
  539459584 B; sample index 61). Not affected.
- Scores, bindings, custody checks and the arm-comparison statement ("20 of 22
  slots have identical immediate causes in both arms; eight of the nine
  returned pairs have identical initialization and context counters") do not
  depend on the corrected sentences. Not affected.

The error was confined to the two quoted prose sentences (and the chat relay);
no slot changes class, no count changes, and no score changes.

## 3. Decision

The Decision is UNCHANGED: ACCEPT the execution custody and the complete
negative scoring ledger, as an engineering audit only, with the same limits as
the original (parent exit status unobserved, so the runner's
`OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT`
release condition is not met by observation and cannot be repaired by a rerun;
TP 0 / FP 0 / FN 71 in both arms with precision and F1 null and no numeric F1;
four CPU-terminated slots with unknown internal stages; coordinator private
cache retained opaque; raw inputs and Gold not re-read). Reasons: the corrected
sentences were a prose summary inconsistent with the report's own tables; the
per-slot evidence, the immediate-cause classification, the paired class table,
the scores and every custody binding were and remain correct. The correction
does not weaken the negative result and creates no positive claim: the two
CG1E0BBA2933 admissions are provisional development admissions under the
1.0 mm gate with unresolved false-pose risk, not accepted registrations, and
they yielded zero supports. This remains a combined consumed-development
change relative to V7, not a mechanism-only speedup, not a calibrated accuracy
improvement, not manufacturing approval, calibrated metrology or generalization
evidence; the historical F1 0.4580152672 is not comparable and is not beaten.

## 4. Hash

The SHA-256 and byte size of this addendum file cannot be embedded in the file
itself; they are printed at the end of the auditor's report to the coordinator.
