# F1 Exact-Blocker Candidate Ranking

Generated: `2026-04-27T16:05:25`

## Baseline vs Overcomplete

- Baseline `visual_qa_raw4`: range `[4, 4]`, owned `4`, energies `[2040.388482, 2314.872665, 2347.675598]`
- Overcomplete `birth_cap6`: range `[6, 6]`, owned `6`, energies `[2292.862371, 2484.706456, 2551.206459]`

## Ranked Extra-Candidate Regions

| Rank | Bend | Score | Atoms | Pair | Angle | Best baseline match | IoU | Nearest distance | Reasons |
|---:|---|---:|---:|---|---:|---|---:|---:|---|
| 1 | `OB3` | `0.633602` | `33` | `['F27', 'F32']` | `28.6733` | `OB4` | `0.0` | `15.595681` | `['angle_far_from_near90_family']` |
| 2 | `OB2` | `0.527151` | `79` | `['F1', 'F14']` | `84.0991` | `OB2` | `0.105882` | `5.222087` | `['centroid_near_existing_bend']` |
| 3 | `OB4` | `-0.045769` | `5` | `['F11', 'F14']` | `93.6878` | `OB2` | `0.333333` | `1.520717` | `['same_incident_pair_as_baseline', 'centroid_near_existing_bend']` |
| 4 | `OB6` | `-0.438233` | `3` | `['F1', 'F14']` | `168.6591` | `OB3` | `0.6` | `1.305901` | `['likely_duplicate_overlap', 'centroid_near_existing_bend', 'tiny_support', 'angle_far_from_near90_family']` |
| 5 | `OB1` | `-0.524108` | `128` | `['F27', 'F32']` | `128.955` | `OB1` | `1.0` | `0.0` | `['likely_duplicate_overlap', 'same_incident_pair_as_baseline', 'centroid_near_existing_bend', 'angle_far_from_near90_family']` |
| 6 | `OB5` | `-0.834404` | `4` | `['F12', 'F14']` | `155.6175` | `OB4` | `1.0` | `0.0` | `['likely_duplicate_overlap', 'same_incident_pair_as_baseline', 'centroid_near_existing_bend', 'tiny_support', 'angle_far_from_near90_family']` |

## Use

This is diagnostic evidence only. A high-scoring candidate should be reviewed as a possible missing bend, but exact-count promotion still requires a formal local split/birth acceptance rule.
