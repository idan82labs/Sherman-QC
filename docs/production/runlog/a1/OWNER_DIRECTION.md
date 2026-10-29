# Owner direction for A1 (2026-10-18)

Priority from iteration 5 on: raise BROAD line RECALL toward >= 0.80 (iter 4: R 0.708, P 0.650, micro 0.678,
family-macro 0.806 on 24 scans), while holding GOLD_SET_V1 (0.977) and the Gold11 relabel (0.987) at >= 0.95
and not lowering precision. Find out which BROAD gold lines the engine misses and why. Use
iter_4/validation/per_bend_all_final.json and the frozen scorer's FN lists. Also check whether a miss is an
engine miss or a BROAD-label problem (BROAD gold is the older label set). Report label problems separately;
never tune to them. General fixes only: no part-specific logic, thresholds frozen.

## Update after iteration 5 (2026-10-19)
Iteration 5 showed that 36 of the 39 BROAD misses are errors in the older labels. On the frozen BROAD gold, recall can
reach at most 0.75. From now on, measure recall against the gold of record: the Gold11 relabel, GOLD_SET_V1 where a
scan is in it, else BROAD. There the engine is at R 0.981 and P 0.987. Never tune to the frozen BROAD labels.
Remaining engine work:
- wrong-way bends (mirrored hinge search);
- recipe-excluded folds;
- the 2 tab-merge FPs on 38172000 SCAN_1;
- per-bend correctness given CAD (181/203) toward 0.90+, with zero false PASS.

## From the real-part E2E audit (2026-10-19, e2e/REAL_E2E_AUDIT_2026-10-19.md)
- HIGH: on joggle bends (10839000 B1-B4), the 3.6 mm short flanges always read UNKNOWN, and the joggle angle U reaches
  ±8.2°, so these bends can only ever be REVIEW. Also, a high-U REVIEW outranks a clear FAIL as the part's "worst"
  characteristic.
- MEDIUM: on the FULL scan every 90° angle reads 88.0-89.7° (4 FAIL). Check whether this is real springback or a
  systematic angle bias (cross-check against the known-good set). scan_2 measured thickness 1.275 mm against 1.5 mm CAD.
