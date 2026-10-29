# V5R3B post-hoc corpus-similarity rejection

Date: 2026-08-15

## Decision

V5R3B is **visually non-representative of the real Sherman QC scan corpus**.
It remains a valid record of one deterministic synthetic stress execution,
but it must not be used as evidence of real-corpus precision, recall, F1, or
architecture validity.  It is consumed and terminal.  It may not be
regenerated, tuned, used for V64 confirmation, or used to set thresholds.

This corrects the earlier truth-free visual precommit.  That review verified
finite sheet topology, calibrated back-projection, and the absence of detached
geometry.  It did not adequately test whether the authored morphology and
mesh sampling matched the production corpus.

## Side-by-side native visual critique

Real references inspected at native resolution:

- `output/research/f1-new-parts-20260814/aug13_v16_retained/native_audits/U01.png`
- `output/research/f1-new-parts-20260814/aug13_v16_retained/native_audits/U02.png`
- `output/research/f1-new-parts-20260814/aug13_v16_retained/native_audits/U03.png`
- `output/research/f1-new-parts-20260814/aug13_v16_retained/native_audits/U04.png`

Synthetic references inspected at native resolution:

- all three V5R3B truth-free review sheets;
- all six native depth-view sets;
- all six fused back-projection audits;
- all six post-truth V62/V63 overlays.

The visual mismatch is systematic:

1. **Wrong silhouette prior.** V5R3B is dominated by rectangular trays,
   guards, and perimeter walls.  The real examples include irregular stamped
   brackets, sparse localized tabs and returns, and large nearly flat panels
   whose meaningful folds occupy only a small portion of the perimeter.
2. **Too many authored bend-like boundaries.** Long parallel walls, paired
   returns, regular apertures, vent arrays, and deliberately broad rolls place
   strong linear structures exactly where the detector is already vulnerable.
   This manufactures easy false-positive opportunities rather than sampling
   ordinary corpus clutter.
3. **Over-regular line geometry.** Synthetic folds are unusually straight,
   complete, mutually parallel, and constant-cross-section.  Real scans show
   incomplete coverage, local deformation, variable support, asymmetric
   termination, and surface contamination.
4. **Feature-density mismatch.** O007's dense slot array and O009's repeated
   return profile are visually over-authored relative to the inspected corpus.
   These cases test a generator vocabulary more than the production domain.
5. **Occlusion and boundary artifacts are too clean.** The synthetic views
   create crisp, repeated visibility boundaries and long free-edge ridges.
   They are easier for a line detector to promote than the mottled, partially
   observed boundaries in real range meshes.
6. **Scale balance is wrong.** The synthetic lane emphasizes large perimeter
   folds and broad rolls while underrepresenting small localized creases,
   short rounded facets, shallow stamped transitions, and isolated mounting
   features embedded in much larger sheets.

## Quantified mesh-spacing mismatch

The public real and synthetic triangle meshes were read without labels.  For
each mesh, triangle-edge lengths were sampled deterministically (up to 300,000
triangles) and the 10th, 50th, and 90th percentiles were measured in millimetres.

| case | vertices | triangles | edge p10 | edge p50 | edge p90 |
|---|---:|---:|---:|---:|---:|
| real U01 | 826,771 | 1,653,126 | 0.126 | 0.178 | 0.505 |
| real U02 | 161,295 | 322,216 | 0.254 | 0.505 | 0.723 |
| real U03 | 1,457,462 | 2,905,978 | 0.274 | 0.563 | 0.823 |
| real U04 | 1,649,387 | 3,284,358 | 0.294 | 0.612 | 0.778 |
| synthetic O001 | 15,956 | 26,917 | 2.440 | 3.767 | 7.497 |
| synthetic O003 | 20,284 | 35,329 | 2.148 | 3.130 | 5.733 |
| synthetic O005 | 10,695 | 17,838 | 2.656 | 3.992 | 7.297 |
| synthetic O007 | 13,666 | 23,222 | 2.249 | 3.511 | 7.979 |
| synthetic O009 | 23,348 | 40,989 | 2.143 | 3.261 | 7.397 |
| synthetic O011 | 11,083 | 17,815 | 2.095 | 3.339 | 8.604 |

The V5R3B median edge spacing is approximately 5–22 times coarser than the
four real scans.  This is not a cosmetic discrepancy: coarse facets and
organized-range fusion boundaries can become strong straight ridges that the
bend detector promotes.

## False-positive evidence

The frozen V62 baseline produced `TP=9`, `FP=9`, `FN=25` on V5R3B.  Native
overlays show false predictions aligned with synthetic perimeter/facet
structures, including two FPs on O003, two on O005, three on O007, and one
each on O001 and O011.  O007 recovered no truth bends but produced three
predictions.  This distribution is consistent with the visual confound above.

The score remains scientifically useful only as evidence that the detector is
sensitive to these synthetic structures.  It is not a clean estimate of
production false-positive behavior.

## Mandatory replacement-lane visual gate

Before any future synthetic scan reaches inference, every case must pass all
of the following using public source only and with truth unopened:

1. Native side-by-side presentation against at least two real corpus scans of
   comparable gross size and morphology.
2. Median triangle-edge spacing within the observed real envelope after the
   same public preprocessing, or an explicitly justified detector-input
   resampling match.  A 5x or larger mismatch is an automatic rejection.
3. No detached sheets, double skins, boundary rails, or renderer seams.
4. No dense repeated vents, paired returns, or broad-roll constructions unless
   a visually comparable real corpus reference is cited.
5. Irregular finite silhouette, localized feature placement, and a realistic
   ratio of flat sheet area to bend-support area.
6. At least one small crease, one short finite bend, and one rounded/shallow
   facet across the batch, but never inserted as repetitive decorative clutter.
7. Realistic partial visibility, mottled coverage, and edge truncation without
   crisp generator-specific straight seams.
8. A blinded human verdict of `CORPUS_LIKE` for every case.  Any ambiguous or
   rejected case rejects the entire prospective lane before inference.

V64 may use only a newly generated and newly sealed lane that passes this
gate.  V5R3B remains excluded.
