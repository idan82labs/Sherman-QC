# 48811000_A native-visual line precommit

Status: frozen before V9 selector inference.

The raw scan was inspected at native resolution in six views with surface-normal coloring. It is a U-shaped bracket with two end flanges and two transition webs. Four conventional bend axes are visible: two lower trough bends and two upper flange/web bends. The large circular opening, four bolt holes, outer silhouettes, and parallel mesh aliases are rejected as non-bends.

The four frozen scan-frame lines are stored in `output/research/f1-new-parts-20260812/v9/holdout_488/manual_visual_precommit.json`. CAD, drawing, sender count, expected count, and prior predictions were not opened or used. `48811000_A` had no prior prediction, annotation, score, or optimization record in the workspace.

The first isolated preprocessing attempt was invalidated before candidate generation because it did not pass the existing strict-blind switch. The strict-blind candidate pool was regenerated and is the sole pool referenced by this precommit.

## Post-freeze result

V9 selected four axes. Against the frozen visual precommit it scored TP=3, FP=1, FN=1, precision=0.75, recall=0.75, F1=0.75. Native-resolution review showed that all four physical bend zones were found; the left-upper representative was a nearby singleton scan-mesh axis tilted by about 10.4 degrees from the consensus line.

The bounded V10 correction replaces only a selected singleton `scan_mesh_adjacency` representative with a nearby line supported by at least three independent source families (within 12 degrees, 5 mm, and 0.80 shorter-span overlap). It cannot add a bend or use part identity, expected count, CAD, or labels.

On the nine comparable full-resolution-mesh development cases, the same frozen rule changed aggregate TP/FP/FN from 20/18/8 (F1 0.606061) to 20/17/8 (F1 0.615385), with no lost true positives. On this optimization case it made one representative replacement and scored TP=4, FP=0, FN=0, F1=1.0. Visual review in four views confirmed that the repaired prediction overlays all four frozen axes cleanly.

This is optimization evidence, not final promotion evidence: V10 was created after inspecting the V9 error on this case. It must pass a different untouched part before promotion.
