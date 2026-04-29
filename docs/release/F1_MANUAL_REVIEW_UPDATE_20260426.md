# F1 Manual Review Update - 2026-04-26

## Reviewed Cases

### `49024000`

- Manual truth: `3` conventional bends.
- Raw F1 status: rejected.
- Valid raw owned region: `OB2`.
- Invalid raw owned regions: remaining raw marks/owned regions are wrong and must not be promoted.
- Corpus update: prior expected count `4` is marked disputed and the part template now records `3`.

### `48991006`

- Manual truth: not revalidated in this review pass.
- Raw F1 status: rejected.
- Failure: `OB4` and `OB5` sit on the same bend, so raw F1 is duplicating support on one physical bend.
- Scan status: poor processing/scan-quality case; keep as a diagnostic blocker, not clean accuracy evidence.

## Merge Impact

- Neither `49024000` nor `48991006` can be used as raw-F1 exact promotion evidence.
- Both cases remain useful as failure tests before any runtime promotion.
- Required next algorithm work: stronger duplicate owned-region collapse, false support suppression, and bad-scan confidence/abstention.
