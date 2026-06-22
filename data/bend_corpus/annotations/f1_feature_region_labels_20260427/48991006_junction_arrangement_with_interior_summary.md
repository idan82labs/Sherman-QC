# 48991006 Junction Arrangement With Interior Birth

This reruns the junction-aware bend-line arrangement with the interior missed-bend hypothesis `IBH1` from `48991006_interior_bend_gaps.json` added as an interior birth input.

## Result

- Output: `48991006_junction_bend_arrangement_with_interior.json`
- Existing arrangement status remains: `two_bend_arrangement_conflicted`
- Interior birth candidates tested: `36`
- Interior incidence status:
  - `incidence_rejected`: `36`
  - `incidence_pending`: `0`
  - `incidence_validated`: `0`

## Interpretation

The interior visual signal is real enough to keep as a missed-bend support cue, but current flange-pair attachment cannot validate it as a countable bend-line object.

The highest-scoring interior attachments fail for one of two reasons:

- The candidate has plausible line/axis geometry but weak normal-arc compatibility and only one-sided flange contact.
- The candidate has plausible normal-arc compatibility but the line axis disagrees with the available flange-pair axis.

This means the next blocker is not raw interior support discovery. The blocker is local flange/contact recovery around the interior support:

- find the actual adjacent flange components around `IBH1`,
- repair fragmented flange IDs or infer local flange continuations,
- then re-evaluate `IBH1` against those recovered attachments.

## Decision

Do not promote this to F1 count logic. Keep the interior birth path as a diagnostic-safe input channel. The next implementation step should be a local flange-neighborhood/contact recovery diagnostic for `IBH1`.
