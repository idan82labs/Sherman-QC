# Regular bend frame repair — 2026-09-29

The regular worker previously detected bends in raw scan coordinates, then aligned
the scan separately for visualization. Untransformed bend endpoints were labelled
as reference coordinates. Matching could ignore spatial separation, while the
display clipper did not establish surface support.

The worker now computes one shared alignment before detection. The progressive
pipeline and local refinement reuse its aligned points and pose. Strict spatial
matching rejects distant or differently oriented candidates. Report geometry is
validated against reference triangles and clipped only within the original finite
segment. Frontend markers and focus reject missing or invalid display provenance.
The CPU renderer draws CAD triangles and tests marker occlusion.

CAD-local angle estimates retain their measured angles, but CAD placeholder
endpoints and radii are no longer presented as independently measured geometry or
zero deviations. Heatmap loading is connected to the regular viewer.

## Verified evidence

- Focused synthetic worker, matching, frame, overlay and report tests: 60 passed.
- Frontend display-contract check, production build and focused ESLint: passed.
- Fresh regular worker job `bend_f60d8b78562740da93f97a2a2bd5b4fd`: exit 0,
  28.68 seconds, peak process-tree RSS 534736 KiB, OMP threads 1.
- All 581263 saved scan points reproduce the persisted transform within
  0.00000853 mm; deviation count matches exactly.
- Dense independent sampling of every displayed segment stays within 2.854 mm
  of CAD triangles and inside the original segment. The markers remain axes,
  not claims of measured skin position.
- Three-page PDF visually inspected; report, overlays and heatmap data generated.
- Physical inputs were only the explicitly requested 37361005 raw PLY and STEP.
  No gold labels were used for inference or scoring in this regular run.

Result details and operator handoff are in the job directory's E2E_VALIDATION.json
and CLI_HANDOFF.md. Existing reports and unrelated workspace edits were preserved.

## Remaining boundary

The regular CAD detector reports three targets, whereas the earlier research
candidate reported four. Complete bend recovery and production metrology are not
established by this fix.

The authenticated regular-app visual check remains pending. Automatic approval
review rejected the prior development-account login because the user had requested
CLI execution and opening later. A sign-in authorization question is pending.
pending_app_metadata.json is prepared without an invented owner; it is not a
registered app job. Next authorized step after sign-in is normal operator binding
and checking CAD, scan, heatmap and bend toggles against this completed result.
Do not bypass authentication, rerun geometry just to open it, or expand to other
physical cases or terminal research experiments.
