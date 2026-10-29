# Owner direction for A1 (2026-10-18)

Priority from iteration 5 on: raise BROAD line RECALL toward >= 0.80 (iter 4: R 0.708, P 0.650, micro 0.678,
family-macro 0.806 on 24 scans), while holding GOLD_SET_V1 (0.977) and the Gold11 relabel (0.987) at >= 0.95
and not lowering precision. Find out which BROAD gold lines the engine misses and why. Use
iter_4/validation/per_bend_all_final.json and the frozen scorer's FN lists. Also check whether a miss is an
engine miss or a BROAD-label problem (BROAD gold is the older label set). Report label problems separately;
never tune to them. General fixes only: no part-specific logic, thresholds frozen.
