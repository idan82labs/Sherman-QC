# F1 Visual QA Findings - 2026-04-26

This note records assistant visual inspection of the 1080p review renders after manual feedback. These observations are not a replacement for human bend-count truth; they identify render/owned-region failure modes to drive the next F1 tranche.

## Important Render Semantics

- Several contact sheets show `no accepted dots` because the accepted result is a range, not an exact count.
- In those cases, the visible lines/markers are raw F1 debug owned regions, not accepted final bend dots.
- Current final-dot suppression is correct for range outputs; the visible issue is mainly raw owned-region quality and decomposition semantics.

## `47959001_01PREL_scan1`

- Known/manual target in current annotations: visible observed conventional bends = `5`.
- Current accepted output: range `[4,6]`, exact `None`.
- Visual result: raw F1 shows `4` owned support regions.
- Assessment: raw supports mostly sit on plausible bend zones, but the scan is missing at least one visible conventional bend compared with the 5-bend human target.
- Status: range-safe but not exact-promotion-safe.
- Failure mode: under-separation / missing compact-neighbor bend.

## `10839000_01_full_scan`

- Manual semantics: `8` conventional bends plus `2` raised/form features.
- Current raw F1: `9` owned support regions.
- Visual result: raw markers cluster around the major frame/corner zones. Several labels appear to share the same local bend/feature zone rather than representing distinct conventional bends.
- Assessment: raw F1 is not reliably separating the conventional bend pair structure from raised/form or fragment features. It finds bend zones, but not the correct conventional-object decomposition.
- Status: keep protected by control/accepted logic; do not promote raw F1 exact count.
- Failure mode: conventional-vs-raised feature ambiguity plus duplicate/merged support at corner clusters.

## `10839000_01_scan_1`

- Manual target in current annotations: `4` visible conventional bends plus `2` raised/form features.
- Current accepted output: range `[2,4]`, exact `None`.
- Visual result: raw F1 shows `4` owned support regions.
- Assessment: accepted range contains the conventional target, but final dots are correctly suppressed because the output is not exact. Raw marks are plausible bend-zone evidence, not promotion evidence.
- Status: range-safe; exact promotion not allowed.
- Failure mode: limited partial-scan confidence and raised/form ambiguity.

## `10839000_01_scan_2`

- Manual target in current annotations: `6` visible conventional bends plus `2` raised/form features.
- Current accepted output: range `[6,8]`, exact `None`.
- Visual result: raw F1 shows `10` owned support regions with visible duplicate clusters around the same physical bend/feature areas.
- Assessment: accepted range contains the conventional target, but raw F1 over-fragments/duplicates support and cannot be promoted.
- Status: range-safe; raw exact not safe.
- Failure mode: duplicate support and conventional-vs-raised feature ambiguity.

## Next Algorithmic Targets

1. Keep accepted final-dot suppression for range outputs.
2. Add a clearer visual distinction between accepted final bends and raw debug support.
3. Improve duplicate support collapse where multiple OB labels sit on the same physical bend zone.
4. Add a feature-class gate for raised/form features so they do not compete directly with conventional bend count.
5. Improve compact-neighbor separation on `47959001`, where the accepted range is safe but raw F1 still misses one human-counted visible bend.
