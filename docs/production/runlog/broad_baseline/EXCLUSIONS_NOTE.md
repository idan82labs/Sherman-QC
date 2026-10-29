# Exclusions note for the broad baseline (2026-10-17)

The scans below are **not physical scans**. Drop them from every broad-baseline
report table, score, and denominator. If a run has already scored them, report
the totals with these rows removed. Do not re-tune anything on the difference.

They are `live_scans/*_LIDAR_DEMO_<stamp>.ply` files. The legacy endpoint
`POST /api/live-scan/demo/load` (`backend/_server_legacy_impl.py`,
`_write_lidar_demo_scan`) made them by resampling `test_files/<part>/scan.stl`.
Each file has the same signature:

- an Open3D binary PLY header with xyz only
- 11284 to 13196 points, which is 12000 or 14000 minus about 6% dropout
- an mtime within 0.1 to 0.7 s of the stamp in the file name
- a point-to-source-mesh distance with a median of 0.053 to 0.055 mm and a p95 of 0.156 to 0.158 mm, which is the 0.08 mm Gaussian jitter the endpoint adds

Per-scan evidence is in
`docs/production/runlog/benchmark_v2/manifest_build/synthetic_demo_exclusions_2026-10-17.json`.
`BENCHMARK_V2.json` now gives all of them the role `EXCLUDED_NOT_A_SCAN`.

| scan_id | part | file | in case_list_broad_v2 |
|---|---|---|---|
| sc_657b87ceeba6 | 47266000 | 47266000_F_LIDAR_DEMO_20260209_154559_545707.ply | yes (AI_GOLD_ACCEPTED) |
| sc_573c999ef6ea | 47266000 | 47266000_F_LIDAR_DEMO_20260209_154744_233337.ply | yes (AI_GOLD_ACCEPTED) |
| sc_123f84b5afcb | 47266000 | 47266000_F_LIDAR_DEMO_20260209_155246_866937.ply | yes (AI_GOLD_ACCEPTED) |
| sc_c2ff6d5b6c2f | 47266000 | 47266000_F_LIDAR_DEMO_20260209_155400_715790.ply | yes (AI_GOLD_ACCEPTED) |
| sc_9018b86ccc9a | 47266000 | 47266000_F_LIDAR_DEMO_20260209_165235_383038.ply | yes (AI_GOLD_ACCEPTED) |
| sc_a0b6add3f6e6 | 47266000 | 47266000_F_LIDAR_DEMO_20260209_211408_450619.ply | yes (AI_GOLD_ACCEPTED) |
| sc_e632352b37f2 | 47266000 | 47266000_F_LIDAR_DEMO_20260209_211721_100645.ply | yes (AI_GOLD_ACCEPTED) |
| sc_43e1dbdce7e3 | 11979003 | 11979003_C_LIDAR_DEMO_20260209_155351_508814.ply | no (already REJECTED_NOT_A_SCAN in review11) |
| sc_d5d7f0cfa36c | 11979003 | 11979003_C_LIDAR_DEMO_20260209_154630_251479.ply | no (already REJECTED_NOT_A_SCAN in review11) |

What this changes in the broad baseline:

- `case_list_broad_v2.json` has 67 cases. Seven of them are the 47266000
  demos above, each with 1 gold line. Removing them leaves 60 cases:
  45 AI_GOLD_ACCEPTED, 11 GOLD11 and 4 GOLD_LINES_HUMAN_NON_GOLD11. It also
  removes 7 gold lines.
- `cad_crosscheck.json` and `cad_crosscheck_auto.json` also list all 9 ids.
  Drop those rows too.
- The ai_bend_lines_v1 label files of the 7 47266000 demos are still on disk
  under `data/bend_corpus/annotations/ai_bend_lines_v1/`. Their asset ids are
  7590fd22d5d86158a6baf337, 5cbf6ce5c255984ca17a4e4b, a11df0bda44de7b856734cd9,
  7f7dd0eace422ac665b7df31, c1e4bcd502f8b3993cbe7722, 387a26b4a83c91527b86793d
  and c9ac548e818e85512c838483. Skip them whenever you iterate that directory.
- No Gold11, Gold37 or physical15 scan is affected.
- The demo source meshes are left as they are. They are sc_9b9fd762f361 for
  47266000 and sc_d45186653eed for 11979003.

A separate label change also affects a case on the list. `sc_9bd69bc6535c`
(47251002_02_V) was relabelled in round 3 from 11 to 12 bends. The new bend is
the 15 deg lanced window tab, the same one as B12 on its X sibling. Its studio
label `e1e41c2d133c34b6b5886eb5.json` was rewritten, so a run scored against
the old 11-line label is stale for that case. See
`docs/production/runlog/benchmark_v2/manifest_build/relabel_r3_2026-10-17.json`.
