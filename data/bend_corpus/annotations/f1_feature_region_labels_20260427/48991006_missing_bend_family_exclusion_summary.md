# 48991006 Missing-Bend Family Exclusion

This diagnostic applies the current promotion gate for missed-bend candidates:

1. candidate must have local flange/contact recovery,
2. candidate must be separable from existing raw same-pair owned bend families,
3. same-pair alias or ambiguity blocks promotion.

## Result

- Output: `48991006_missing_bend_family_exclusion.json`
- Status: `no_separable_missing_bend_candidate`
- Candidates evaluated: `3`
- Family-exclusion pass count: `0`

Blocked reason counts:

- `contact_not_validated`: `2`
- `same_pair_alias_to_raw_family`: `1`
- `same_pair_ambiguous_with_raw_family`: `1`

Candidate verdicts:

- `IBG1`: blocked by `same_pair_ambiguous_with_raw_family`. It has valid recovered contact, but the closest raw same-pair relation is `OB2` with line distance `6.043297 mm`, span overlap `0.77447`, and axis delta `13.432645°`.
- `IBG2`: blocked by `contact_not_validated` and `same_pair_alias_to_raw_family`. It aliases raw `OB1` with line distance `0.867458 mm`, span overlap `1.0`, and axis delta `21.656764°`.
- `IBH1`: blocked by `contact_not_validated`. It is mostly separable from raw same-pair regions, but its merged axis/contact fit failed earlier, so it is not a safe countable object.

## Decision

Do not promote any current `48991006` interior recovered-contact candidate.

This is now a useful negative-control case: a visually plausible missing-middle signal exists, but the current generated candidates either alias existing raw same-pair families or fail local contact validation.

## Next Work

- Build a new interior support proposal that targets the visually unmarked middle bend directly instead of reusing the `F1-F5` lower/right family.
- Treat `IBH1` as evidence that fragment merging is currently harmful on this scan: the merged support becomes geometrically separable but loses valid axis/contact semantics.
- Require future `48991006` improvements to pass this family-exclusion report before they can be counted or rendered as accepted F1 bends.
