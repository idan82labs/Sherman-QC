# T-2026-10-26: E2E W3b flake, studio bend markers missing before bend_pair authoring

- **Status:** OPEN.
- **Owner:** studio / Vendor Setup UI.
- **Priority:** medium. It blocks clean E2E gating, not live inspection.
- **Owner decision, 2026-10-26:** keep the e3-1.13.0 re-pin and ticket this flake.

## Summary

Step **W3b** of the real-part E2E fails intermittently. W3b is the Vendor Setup studio step that authors a `bend_pair` general measurement by clicking on the CAD. When it fails, the step finds **0 bend markers**: `labels: []`, `missing: true` for both target bends. It runs before any E3 job, so engine changes do not affect it.

## Occurrences

| Run | Engine | W3b |
|---|---|---|
| `20261019T-real-fixes3c` | e3-1.12.0 | FAIL (0 markers) |
| `20261019T-real-fixes3d` | e3-1.12.0 | FAIL (0 markers) |
| `20261020T-real-fixes3e` | e3-1.12.0 | PASS |
| `20261026T-real-repin-e3-1-13` | e3-1.13.0 | FAIL (B10, B7: missing, labels []) |
| `20261026T-real-repin-e3-1-13b` | e3-1.13.0 | FAIL (B10, B7: missing, labels []) |
| `20261026T-real-repin-e3-1-13c` | e3-1.13.0 | PASS (13/13) |

Every other step passed in all of these runs. A control run on e3-1.11.0 also passed. The failing assertion is "both bend markers (and their on-line anchors) found" (`scripts/production/run_e2e_real.mjs`, around line 498).

## Suspected cause

There is a race between the studio's marker rendering and the test's anchor read.
- The step waits until `[data-bend-marker]` count equals `nBends` (line 452), sleeps 1 s, then reads the anchors.
- The test's own comment says that once a point is picked, the markers turn into plain labels.
- In the failing runs the markers are gone, or never had their leader anchors, by the time `anchorOf` runs.
- The likely trigger is a re-render or state transition in `VendorSetupPage.tsx` (studio, roughly lines 640–656) after the markers first mount.

## Evidence

- `docs/production/runlog/e2e/20261026T-real-repin-e3-1-13{,b}/steps.json`, plus the `11-FAIL-W3b-wizard.png` screenshot in each.
- `docs/production/runlog/final_eval_2026-10-19/FIXES3.md` §8 (E2E note).
- `docs/production/runlog/e2e/REAL_E2E_AUDIT_2026-10-19.md`.

## Wanted

1. Find out whether the studio really drops its markers (a product bug) or whether only the test reads them too early (a test bug).
2. If it is a product bug, fix it so markers stay stable until the operator picks a point.
3. If it is a test bug, wait on a stable marker-ready signal (for example a `data-markers-ready` attribute) instead of count + sleep.
4. Acceptance: 5 consecutive real E2E runs with W3b passing.

Until fixed, W3b failures with exactly this signature (0 markers, `labels: []`) are known flake. Any other W3b failure is a real failure.
