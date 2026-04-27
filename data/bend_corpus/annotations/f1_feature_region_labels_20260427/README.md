# F1 Feature Region Labels - 2026-04-27

These files encode manual owned-region feedback from the 1080p F1 visual review.

- `49024000_feature_region_labels.json`: complete region labels. User validated the part as 3 conventional bends; only `OB2` is accepted as a conventional bend. This proves raw F1 has a missing-support/split-recovery problem, not just duplicate overcount.
- `48991006_feature_region_labels.json`: partial region labels. User identified `OB4` and `OB5` as the same physical bend zone; the remaining regions need a cleaner manual pass because the scan/process quality is poor.

Use `scripts/summarize_f1_feature_region_labels.py` to regenerate the `*_summary.json` files.
