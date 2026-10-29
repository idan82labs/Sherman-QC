# Sherman QC F1 capability-opening stop — 25 August 2026

The autonomous loop is not at QC accuracy. The retained scan-only V51
development reference remains at precision/recall/F1 0.593, exact count 0.25,
exact localized set 0.083, and minimum-family recall 0. It is research-only.

## What the latest action established

- EXP_SCAN_0002 learned over all 5,787 fused-Ply line objects. F1 regressed to
  0.519 and all nine changed cases failed native-resolution review because the
  selector promoted flat, boundary and parallel aliases.
- The pool nevertheless contains explicit surface/plane/flange ownership on
  2,032 line objects and on 122/163 gold-matching line objects.
- EXP_SCAN_0003 used that ownership, retained V51 unchanged, and admitted at
  most one new line per unclaimed parent pair. It recovered one true bend, but
  added four false boundary/skin events. Recall rose to 0.610 while F1 fell to
  0.585; all three changed cases failed visual review.

The wall is now specific: an adjacent-surface pair does not reveal whether its
shared locus is a material bend, a free boundary, a silhouette, or a duplicate
return skin. The missing discriminator is acquisition continuity or a signed
manufacturing topology—not another score threshold over the same fused PLY.

## Exact unblock paths

### Scan-only

Export authentic pre-fusion depth/range frames for at least two already
consumed development scans, including validity masks, intrinsics, depth scale,
per-frame poses, capture-protocol hash, and the matching fused PLY. Validate
them with `scripts/validate_f1_organized_frame_intake_v1.py`. Do not rerender a
PLY and call it raw data; that cannot restore occlusion or ray history.

### CAD-assisted

Have two distinct named manufacturing/inspection authorities independently
complete all 63 decisions in
`output/research/cad-line-ontology-authority-pack-v1/`. The current validator
still returns `AWAITING_AUTHORITY_REVIEW`. This freezes which CAD events are QC
bends and whether their inspection line is the curvature center, neutral
half-turn, observed surface intersection, or an authority-drawn line.

### Generalization proof

Supply and seal 30 untouched physical scans across 15 morphology families,
with two independent manufactured instances per family, through
`scripts/prepare_f1_physical_intake_v2.py`. Current eligible capacity is zero.

No current result authorizes runtime/main or production use. The loop resumes
from the first opened capability; it does not resume by inventing V74-style
fused-Ply variants.

