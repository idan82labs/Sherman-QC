# F1 zero-shot 10/13 visual checkpoint — 2026-07-15

The verified 13-scan corpus now has 10 hard visual passes under the zero-shot F1 contract. Evaluation truth is used only after inference. No passing run receives an expected count, part-specific target, CAD, or per-line ground truth.

The passing cases are 01, 02, 03, 04, 05, 09, 10, 11, 12, and 13. Every final set matches its count truth and was reviewed in three orthographic views for surface attachment, duplicate lines, hole/free-edge markers, face-interior markers, and missed visible transitions. The machine-readable record and exact audit paths are in `output/experiments/f1-10of13/zero-shot-10of13-checkpoint.json`.

The current-code regression at `output/experiments/f1-span-alias-strict-regression-20260715/summary.json` reports render-admissible count equality `11/13 = 0.846`; hard visual acceptance remains `10/13 = 0.769`. Final marker counts in slice order are `8, 4, 6, 1, 4, 2, 8, 1, 5, 5, 3, 2, 12`, against truths `8, 4, 6, 1, 4, 6, 8, 7, 5, 5, 3, 2, 12`.

Key zero-shot repairs:

- Partial scans can select robust internal strips when their count independently agrees with the scan-policy lower bound; boundary-truncated fragments are excluded only when the internal count supplies that agreement.
- A conservative strip/detector reconciliation completes a short strip from a longer collinear detector corridor and adds only geometrically nonduplicate detector corridors.
- Small-tight scans can use fully observed two-flange incidence when that independent count agrees with the scan-policy lower bound.
- The fragmented repeated-near-90 route deduplicates finite strip corridors, suppresses covered interface births, and adds only uncovered gap-separated interface corridors.
- The small-tight marker contract is scale-aware, so valid compact bracket bends are no longer rejected by thresholds intended for large parts.
- Same-flange-pair aliases are suppressed when a shorter region is almost fully contained on the same physical line and is strongly dominated in span, support mass, and atom count. This removes case-07 OB8 in favor of OB4 without count truth and leaves all other verified marker counts unchanged.

Hard failures remain explicit:

- Case 06: four defensible lines versus six; proposed additions are face-interior false lines.
- Case 07: count eight after same-pair alias suppression, but OB1/OB3 remain a cross-pair collinear ambiguity. Count equality is not credited as visual perfection.
- Case 08: six defensible plane-lattice transitions versus seven; no supported seventh line exists yet.

This checkpoint does not promote F1 to runtime/main. Enterprise promotion remains gated by the full benchmark, deterministic memory/RSS evidence, cache-schema checks, and the release gates.

Ground truth contains bend counts, not per-line coordinates. “Perfect” therefore means exact count plus final-marker admissibility and hard visual alignment—not coordinate-level line-error certification.
