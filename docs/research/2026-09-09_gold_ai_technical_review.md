# Gold convention: AI technical review and provisional development amendment

Date: 2026-09-09. Reviewer: Codex AI, with a separately assigned AI visual
subreview. **AI_MODEL_ADVISORY; not two human manufacturing signatures.**
User explicitly requested AI review after confirming human decisions probably
do not exist. This review may inform provisional development work; it does not
fill the pending authority templates, redefine historical Gold, or qualify a
production/floor decision.

## Outcome

All **63 displayed CAD proposals** were reviewed across the 11-case pack:
**56 provisionally plausible formed-bend features, seven unresolved, zero
positively rejected.** These are proposal-review dispositions, NOT observed
bend counts, precision, recall or Gold truth. The seven unresolved proposals
remain in the denominator. No candidate was deleted or used to relabel Gold.

The [41-event subreview](2026-09-09_gold_ai_visual_review_b.md) covers C007,
C008, C009, C010, C012 and C013. The remaining 22 are documented below.
This is split review coverage, not independent double annotation or AI consensus.
The reviewers inherited historical project context; this was not a blinded
qualification. Scores were not used to choose the dispositions.

## Evidence integrity and defects

The exact pack manifest SHA-256 is
`c52c7399cd7b3263e8f2dd3b9191c0dc356b7175a4ea9e5cb1cf5069a89e2c3f`.
All 22 named image/sidecar byte hashes were checked against it before review.
The five main-review images were requested at original detail, but the viewing
tool disclosed display resizing from 5,209-pixel widths to 2,048 pixels. This is
six-view visual plausibility review, not pixel-exact dimensional measurement.
The subreview separately records its display-resolution limitation.

1. **Cyan does not always mean a measured intersection.** Thirteen of the 63
   sidecar events say `ABSTAIN_RELOCATION`; 50 say relocated. Source confirms
   [relocation](../../research/f1_cad_observed_intersection_v1.py) retains the
   parent line when fitting abstains. The [renderer](../../scripts/render_f1_cad_line_ontology_comparison_v1.py)
   draws every line in that output cyan without checking the relocation state.
   Thus the legend overstates what was observed. The original pack is retained
   unchanged; abstained cyan carriers are unverified fallbacks in this review.
2. **This is not the historical Gold11 reference denominator.** The pack holds
   63 CAD proposals and includes case C010 from family 47959001, absent from the
   six-family historical Gold11 comparison. Historical Gold11 has 71 reference
   lines. Eleven cases in both records does not establish identical membership.
3. **Four curvature alternatives are absent.** C002:CAD_EVENT_0003,
   C004:CAD_EVENT_0003, C005:CAD_EVENT_0001 and C013:CAD_EVENT_0003 have no curvature-center alternative in their
   sidecars. That is neither evidence of non-bend status nor permission to
   substitute another carrier as if it were the same measured quantity.
4. **Images do not certify endpoints.** C010:CAD_EVENT_0004 has a projected extension
   concern, C012:CAD_EVENT_0010 shows unequal longitudinal extents across conventions,
   and C013's short opening-end features have unresolved splitting/identity.
   No displayed nominal length is promoted to observed material support.

## Main visual review: all 22 proposals

`YES` means plausible as a separately reviewable formed feature, not approved
manufacturing intent or observed truth. `UNRESOLVED` means the available views
cannot safely decide separate event identity. All line conventions and physical
trim endpoints remain unapproved by these images alone.

| Case/index | Exact event ID | Plausibility | Visual basis / restriction |
| --- | --- | --- | --- |
| C001/1 | CAD_EVENT_0003 | YES | Long turn beside the large opening is visible in oblique/reverse views. It occupies a different level from the lower perimeter turn. Relocation abstained; cyan is not measured evidence and the end/corner ownership remains unresolved. |
| C001/2 | CAD_EVENT_0005 | YES | Upper return of a short side wall is visible, with a turned strip distinct from its lower junction. Corner relief prevents exact endpoint approval. |
| C001/3 | CAD_EVENT_0006 | YES | Long front-region turn is traceable across oblique and underside views, separated from the neighboring parallel levels. Projected line overlap is not enough to merge those levels. |
| C001/4 | CAD_EVENT_0007 | YES | Opposite short-side upper return has visible wall and strip surfaces. Its association with the neighboring corner is plausible but not a measured physical trim. |
| C001/5 | CAD_EVENT_0010 | YES | Lower short-side wall-to-sheet junction is spatially distinct from upper return 2. Both surface regions are visible; carrier width obscures the actual radius/termination. |
| C001/6 | CAD_EVENT_0011 | YES | Long outer junction follows the front perimeter below the opening-side structure. Its continuity is visually plausible; displayed full span is not independently observed support. |
| C001/7 | CAD_EVENT_0012 | YES | Remaining lower short-side junction is visible under upper return 4. Apparent crossings with long carriers in oblique projection do not establish shared 3D endpoints. |
| C002/1 | CAD_EVENT_0001 | YES | Long folded edge of the main plate is visible with a narrow turned flange. This is a formed junction rather than one of the nearby circular holes. Corner-end relief remains unmeasured. |
| C002/2 | CAD_EVENT_0002 | YES | Root of a larger upright mounting tab is visible at one end of the plate. The tab and parent sheet support bend plausibility; short end limits remain unresolved. |
| C002/3 | CAD_EVENT_0003 | YES | Root of the smaller upright tab is visible as a separate turn. Its missing curvature-center alternative is an extraction/representation limitation, not proof the tab is unbent. |
| C002/4 | CAD_EVENT_0004 | YES | Opposite larger tab has a distinct raised surface and root turn. Do not merge it with the adjacent long edge merely because their carriers meet in projection. |
| C002/5 | CAD_EVENT_0005 | YES | Opposite smaller tab has a clear local plate-to-tab transition. The hole belongs to the tab, not an additional bend; the exact root span is not resolved through the strokes. |
| C003/1 | CAD_EVENT_0006 | YES | A raised strip at one internal opening has a visible formed root. Relocation abstained, so the displayed cyan line supplies no fitted-intersection confirmation. |
| C003/2 | CAD_EVENT_0008 | YES | The other opening has a distinct raised/turned strip. Cyan is visibly transversely separated from the CAD alternatives; this cannot be resolved by choosing the most visually convenient carrier. |
| C003/3 | CAD_EVENT_0011 | YES | The long plate-to-end-wall junction is evident in the side and oblique views. Its displayed endpoints approach neighboring returns, whose ownership/physical trims remain unresolved. |
| C004/1 | CAD_EVENT_0001 | YES | A long panel-edge turn connects the main plate to an upright flange. Two adjoining surfaces are visible; the notch/end relief needs its own finite-extent rule. |
| C004/2 | CAD_EVENT_0002 | YES | Shorter notched-edge wall visibly turns out of the plate. Nearby holes/slots are not additional bend events. No measured corner endpoints are established. |
| C004/3 | CAD_EVENT_0003 | YES | Opposite long-edge turn is visible as a real flange despite the absent curvature-center alternative. Its upper-looking position in projection does not determine the physical measurement carrier. |
| C004/4 | CAD_EVENT_0004 | YES | Remaining side-wall root is visible and spatially separate from the other three perimeter turns. Mitered/corner geometry prevents independent trim approval. |
| C005/1 | CAD_EVENT_0001 | UNRESOLVED | Central transition between the circular mounting plate and long arm is rounded/compound in the oblique views. Without a curvature alternative, the render cannot establish whether this is a distinct cylindrical bend or a transition/fragment of neighboring turns. |
| C005/2 | CAD_EVENT_0002 | UNRESOLVED | Turn nearest the circular plate is geometrically plausible, but its split from the central rounded transition and changing section is not established. The three displaced carriers do not decide event identity. |
| C005/3 | CAD_EVENT_0003 | UNRESOLVED | Arm-side transition appears smoothly connected to the other rounded regions. A separate bend event versus part of a compound forming feature cannot be determined from these projections; no independent count is assigned. |

Additional unresolved proposals from the subreview are C012:CAD_EVENT_0006,
C013:CAD_EVENT_0003, C013:CAD_EVENT_0005 and C013:CAD_EVENT_0007.

## Mathematical carrier audit

For a circular inner section of radius R, let the tangent-plane unit normals
be n± = (cos h, ±sin h), h = theta/2, and put the curvature center at the origin.
Their equations n±·x = R give the ideal intersection I = (R/cos(h), 0).
A halfway-through-thickness angular-midpoint carrier is
M = (R + t/2, 0). The curvature axis carrier is C = (0, 0).

These are different measured quantities, not numerical approximations to one
point. For R=5 mm, t=1 mm and theta=90 degrees: |C-M|=5.5 mm,
|C-I|=7.0710678119 mm and |M-I|=1.5710678119 mm. Identical directions and spans
do not rescue a C-versus-M comparison from the retained 4 mm transverse gate.
A perfect center-axis estimate can therefore fail a mid-surface reference.
The formula assumes ideal concentric circular skins and tangent flanges;
it is not a calibration model for real springback or compound features.

“Neutral” must not silently mean the material's zero-strain layer. A geometric
half-thickness marker is a chosen representation. Published V-bending research
shows material-dependent neutral-layer shifts and unloading effects; CAD skins
alone do not establish that layer. See the primary
[2024 V-bending study](https://doi.org/10.1016/j.ijplas.2024.103889).

NIST's [algorithm testing method](https://www.nist.gov/pml/sensor-science/dimensional-metrology/algorithm-testing)
compares fits on simulated measurement/form-error data with reference fits.
This supports keeping numerical fit verification distinct from event identity
and scanner qualification; it does not make an easy synthetic F1 comparable
to historical physical Gold.

## Technical recommendation and scoped amendment

For the new development engine, retain **measured cylinder axis** as the
radius/axis measurement carrier. Keep an eventual **surface inspection marker**
as a separate, explicitly named derived representation. Never rename one as the
other or choose between them by whichever obtains a higher consumed Gold score.
Neither a count of CAD cylinders nor a count of successful radius fits is an
observed manufacturing-bend count.

The user's explicit AI-review instruction permits an **AI-reviewed provisional
development lane**. Two human signatures are not a prerequisite for this
advisory lane. The original human-qualified CIBEC/release gate is unchanged;
no AI identity or subagent is entered as a human countersigner.

Before a comparable historical-score diagnostic, freeze the exact Gold11 case
IDs, unchanged 71 references, geometric convention, scorer, source/configuration
identities, resource budgets and all-case failure handling. Scores would be
agreement with retained development labels, not independently qualified physical
accuracy. This 63-proposal pack cannot replace that reference set. Unsupported
jobs must stay in the denominator. A different convention/cohort needs a new
paired baseline and cannot establish uplift over historical F1 0.4580152672.

Still required: a compatible bounded raw-input/datum path and a convention
bridge established independently of score optimization. This review does not
authorize a terminal-protocol retry, new email access, deployment, floor testing
or unbounded physical execution. Original raw inputs, Gold coordinates and
human decision templates were not modified or opened for this review.
