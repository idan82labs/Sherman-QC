# EXP CAD 0020 — gated local carrier localization

## Outcome

Retain as a displacement-hypothesis stage only.

Generalization progress indicator: `GPI-C_POSITIVE_GUARDED_LOCALIZATION_NOT_F1_PROGRESS`.

EXP19 was retired because it moved coverage-saturated axes to reduce residual with no coverage gain. EXP20 makes saturation terminal: events with 100% nominal finite-bin support cannot be searched and must retain exact zero shift. Incomplete events are searched only if they gain at least one finite support bin.

## Results

- Thirty-eight fully supported events across N011/N012/N014/N015/N016 remain at exactly zero shift.
- N003 event 3 moves 9.49 mm and improves coverage from 0.0 to 0.625.
- N004 event 2 moves 9.07 mm and improves 0.875 to 1.0.
- N004 event 3 moves 8.43 mm and improves 0.625 to 0.75.

Native-resolution review confirms that only those three incomplete events move and that their yellow hypotheses land on visible green-supported scan neighborhoods. The N011 control shows yellow exactly occluding nominal magenta axes.

## Boundary

The objective is still local radial-annulus support. It can hypothesize where a displaced carrier may be, but it cannot prove cylindrical curvature or the intended bend angle. Trusted good/bad disposition and curvature/normal evidence remain required before an accuracy, F1, or production claim.
