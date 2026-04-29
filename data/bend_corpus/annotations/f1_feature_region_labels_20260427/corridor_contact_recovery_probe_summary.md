# Corridor-Backed Contact Recovery Probe

Generated: `2026-04-29`

## Scope

Diagnostic-only F1 run with `--allow-corridor-backed-contact-recovery`.

Inputs:

- `10839000`
- `ysherman`

Output folders:

- `data/bend_corpus/evaluation/f1_corridor_contact_recovery_probe_20260429`
- `data/bend_corpus/evaluation/f1_corridor_contact_recovery_probe_render_ysherman_20260429`

## Result

| Part | Control | Probe raw F1 | Accepted Candidate | Visual Status |
| --- | --- | --- | --- | --- |
| `10839000` | exact `10`, range `[9,10]` | exact `10`, range `[10,10]` | exact `10`, range `[9,10]` via control-aligned tie break | Not a manual-conventional win; still uses legacy target semantics. |
| `ysherman` | range `[10,12]` | exact `12`, range `[12,12]` | exact `12`, range `[12,12]` | Promising count result, but render consistency is not promotion-ready. |

## Visual QA Note

The `ysherman` projection render shows `12` raw owned bend projections, but the 3D overview render reports only `9` support regions after render suppression. This means the count result is promising but not yet acceptable for promotion: the ownership/render path needs to explain why three raw regions suppress from the 3D overview and whether the remaining markers cover all visible bends.

## Engineering Decision

- Keep the new recovery switch diagnostic-only.
- Do not route it into default F1 behavior yet.
- Next step is visual/ownership consistency validation for `ysherman`: all `12` countable regions must render as clear, non-floating, non-clustered bend marks before this can become a route profile.
- `10839000` remains unresolved under manual conventional semantics because the focused slice still treats it as the legacy exact-10 target.
