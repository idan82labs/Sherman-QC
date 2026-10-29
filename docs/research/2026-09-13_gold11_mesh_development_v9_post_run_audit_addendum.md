# Addendum to the Gold11 V9 independent physical POST - correction of three statements

Written 2026-09-13T14:09:30Z (`date -u`). Reviewer: the same independent AI
post-run auditor (`independent-ai-post-auditor-v9`, Claude Fable 5.1,
reviewer role). This addendum corrects the original POST without modifying
it, at the coordinating owner's request; the corrections are the reviewer's,
not the author's. The original is preserved verbatim at
`docs/research/2026-09-13_gold11_mesh_development_v9_post_run_audit.md`,
64734 bytes, SHA-256
`bcdd7e4b414d308ca347f40ddda5c78fb53d24f623ae6b6083bb25c1f0b33dca`
(re-hashed now; byte-identical by `diff` to the copy in the job tmp
directory). Read-only, as before: only the original report, the released V8
POST, `FROZEN.json`, the 22 receipts (`COMPLETED.json`), the 22 retained
projections (`SCORING_INPUT.json`), the 22 leaf logs (`OUTPUT.txt`) and the
external archive (`shasum` of the container and `tar -tvf` listing only,
parsed in memory) were re-read for these checks; no RESULT.json was
re-parsed, no raw data, Gold/reference container, opaque cache or other
attempt was touched, no code was executed, and nothing in the repository was
modified. This file is the sole write.

## 1. Archive member identity (original Findings item 8, line 905)

Original (lines 904-906, quoted verbatim):

> the working tree,
> `source/` snapshot and archive members are hash-identical for all 85
> sources;

This overstates what was verified. The archive evidence actually gathered
(original check 2, lines 211-219, which states it correctly) is: SHA-256 of
the tar CONTAINER
`docs/research/gold11_mesh_development_v9_physical_frozen.RdikX7AH/SOURCE.tar`
= `412c9f8e662c845e172cdd19a134031c62a5c4a12866041fd82981da2c6c2326`,
3451904 bytes, equal to the value the freeze printed and the PRE and the
coordination record carry; a `tar -tvf` listing of 92 entries = 6
directories + 86 file members = `FROZEN.json` (51207 B) plus exactly the 85
`source/<path>` members named in `FROZEN.source_sha256` (0 extra, 0
missing); every listed member SIZE equal to the on-disk `source/` snapshot
size (86/86); all file members mode `-rw-------`. No member was extracted
and no per-member hash was computed, so member BYTE identity with the
snapshot or the working tree was not established by this audit (the PRE
also listed only). The container hash binds the archive as a whole to its
freeze-time value; that is the extent of the archive evidence.

Corrected statement for Findings item 8: "the working tree and the
`source/` snapshot are hash-identical for all 85 sources (85/85, check 2);
the external archive is bound by its container hash `412c9f8e...2326`
(3451904 B) and by its member listing (names, sizes, modes), not by
per-member hashes."

Evidence re-checked now: the container re-hashed to the same value and size;
the listing re-parsed in memory gives 92 entries, 86 file members, names
equal to `FROZEN.json` plus the 85 frozen paths, 86/86 sizes equal, modes as
stated; nothing extracted.

## 2. Arm comparison count (original line 840)

Original (lines 840-842, quoted verbatim):

> Arm comparison: 21 of 22 slots have identical immediate causes in both
> arms; the only arm difference is CG4BDD (C supported at 0.696 mm then MG; H
> rejected at 1.196 / 1.318 / 1.338 mm).

The count is wrong: a differing pair involves both of its slots. Corrected
statement: 10 of 11 case pairs (20 of 22 slots) share the same immediate
cause between the two arms - CPU for seven cases (CG08506AE97D,
CG191DEC7F81, CG1E0BBA2933, CG4FAF273093, CG64C412CC73, CGEBA9A869AC,
CGF41F49066A), PF for two (CG65CD575B1C, CG84C8FA12D2), MW for one
(CGE4B0549E41); the exception is CG4BDD1371B6, whose two arms differ:
control MG (measurement contract refusal - the support cover V3 feature-type
guard after one supported context,
`SUPPORT_COVER_MEASUREMENT_FAILURE:ValueError:INVALID_SUPPORT_COVER_FEATURES`)
versus challenger RG (no supported context - all three candidates rejected at
the 1.0 mm gate, `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`). The
parenthetical description of the CG4BDD difference in the original stands.

Evidence re-checked now: the cause of each slot re-derived from the 22
receipts (`exit_code` -24 -> CPU) and, for the 8 returned slots, from the
retained projections' `reported_workflow_phase` / `reported_workflow_reasons`
(`MEASUREMENT_REFUSAL` -> MG; `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`
-> RG; `NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS` -> PF;
`NOISY_PLANE_INITIALIZATION_UNAVAILABLE` -> MW): 7 x 2 CPU, 2 x 2 PF, 1 x 2
MW, CG4BDD MG/RG - ten identical pairs (twenty slots), one differing pair.

## 3. Worker CPU comparison (original lines 817-819)

Original (lines 817-819, quoted verbatim):

> For the 8 slots returned in both
> campaigns the worker CPU rose from 1850.329 s (V8, 18 returned) to
> 2535.780 s (V9, 8 returned); this is not a speed-up.

The sentence labels an UNPAIRED comparison as paired: 1850.329 s is the V8
total over its 18 returned workers and 2535.780 s the V9 total over its 8
returned workers; the two worker sets differ. Corrected statements, both
from the records:

- Unpaired totals of returned-worker CPU: V8 1850.329 s over 18 returned
  workers (released V8 POST); V9 2535.780 s over 8 returned workers (V9
  `INFERENCE_TERMINAL` log lines). Different denominators; not a paired
  comparison and not comparable as totals.
- Paired comparison for the 8 slots returned in BOTH campaigns
  (CG4BDD1371B6, CG65CD575B1C, CG84C8FA12D2, CGE4B0549E41, both arms; every
  V9-returned slot was also returned in V8): V8 763.757 s -> V9 2535.780 s
  (+1772.023 s). Per slot (worker-reported `worker_cpu_seconds`, V8 -> V9):
  CG4BDD C 253.608 -> 1625.392 (+1371.784); CG4BDD H 288.276 -> 701.538
  (+413.262); CG65C C 32.607 -> 30.687 (-1.920); CG65C H 50.297 -> 46.913
  (-3.384); CG84C C 45.105 -> 42.684 (-2.421); CG84C H 66.949 -> 63.487
  (-3.462); CGE4B C 13.429 -> 12.417 (-1.012); CGE4B H 13.486 -> 12.662
  (-0.824). The paired increase lies entirely in CG4BDD (541.884 -> 2326.930
  s; four and three refined candidates instead of one per arm); the other
  six paired slots are 221.873 -> 208.850 s. Neither figure is a speed-up or
  a geometry claim; the 14 killed V9 workers (and the 4 killed V8 workers)
  have no final CPU value and are excluded from both figures, so no
  campaign-wide CPU total exists. (The V9 paired total is 2535.780 s as the
  sum of the three-decimal per-slot values, 2535.781 s at full precision.)

Evidence re-checked now: the V9 values re-read from the eight
`INFERENCE_TERMINAL` lines in `OUTPUT.txt`; the V8 values from the released
V8 POST's all-22 table (its 18 returned rows sum to 1850.329 s; the eight
paired rows sum to 763.757 s).

## 4. Dependence of other claims

- The original check 4 sums ("known worker CPU for the 8 returned workers
  2535.780 s (no total including the killed workers is formed)"), the leaf
  wall sums and Findings item 14 ("3.08x V8's inference wall") are stated
  over the V9 denominators or as leaf-wall sums and are unchanged.
- The class table and the per-slot V8 -> V9 transition list (CPU 14, PF 4,
  MW 2, RG 1, MG 1) already record CG4BDD as the single differing pair; the
  corrected count follows from them. Not affected.
- Original check 2 (source drift 85/85 in working tree and snapshot; archive
  container hash and listing) is stated correctly and is unchanged; only the
  summary sentence in Findings item 8 changes. The runner's own final source
  audit (`complete` true, `unchanged` true, 85 entries) is unaffected.
- Scores, bindings, custody digests, the inventory and scoring-request
  reproductions, the outer-exit evidence and Findings item 3 (the cover V3
  feature-guard refusal) do not depend on the three corrected sentences. No
  other claim of the original changes.

## 5. Decision

The Decision is UNCHANGED: ACCEPT the execution custody and the complete
negative scoring ledger, as an engineering audit only, with the same limits
as the original (parent exit observed by the waiting parent, proving the
outer return code only; TP 0 / FP 0 / FN 71 in both arms with null precision
and F1 and no numeric F1; fourteen CPU-terminated slots with unknown internal
stages; the V9 measurement chain not exercised because of the cover V3
feature-guard refusal on the only slot that reached measurement; coordinator
cache opaque; raw inputs and Gold not re-read). Reasons: the three corrected
sentences were prose summaries; the archive evidence is weaker than Findings
item 8 said (container hash plus member listing, not per-member hashes) but
the source-drift evidence for the snapshot and the working tree is unchanged
and the runner's final source audit reported no change; the arm-comparison
count and the CPU comparison were mislabeled summaries of tables that were
already correct. None of the corrections weakens the negative result or
creates a positive claim. This remains a combined consumed-development change
relative to V8, not a mechanism-only speedup, not a calibrated accuracy
improvement, not manufacturing approval, calibrated metrology, conformance or
generalization evidence; the historical F1 0.4580152672 is not comparable
and is not beaten.

## 6. Hash

The SHA-256 and byte size of this addendum file cannot be embedded in the
file itself; they are printed at the end of the reviewer's report to the
coordinator.
