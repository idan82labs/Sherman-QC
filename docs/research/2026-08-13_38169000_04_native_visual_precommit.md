# 38169000_04 native visual precommit

- Raw PLY SHA-256: `db1839480738ac85934facd0280f69b439b2fe2f0c7541753f2a6edbb3818aa8`
- Frozen before candidate generation or V12 inference.
- CAD, SLDPRT, and drawing/PDF attachments were not opened.
- Gmail exposed a sender count snippet during link retrieval. That value was
  quarantined and not used to select or fit the manual axes.

Native-resolution inspection identifies a broad main sheet and a three-step
return along one long edge. The supported conventional bend axes are the main
sheet to first return wall, first return wall to narrow shelf, and narrow shelf
to terminal return wall. The large circular opening, repeated lanced slots,
small holes, sheet perimeter, parallel faces, and scan silhouettes are rejected.

The raw six-view was supplemented by a PCA-aligned edge-profile audit. Central
and feature-end 40 mm slices independently show the same three curvature zones;
this prevents the closely spaced parallel returns from being collapsed into a
single silhouette. Axes were fit from the raw sheet frame before inference and
then frozen in `output/research/f1-new-parts-20260813/v12/holdout_381/manual_visual_precommit.json`.
