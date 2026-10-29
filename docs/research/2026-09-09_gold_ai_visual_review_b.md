# Consumed CAD-event pack: AI technical visual review B

Date: 2026-09-09. Classification: **AI_MODEL_ADVISORY**.

This is a technical image review by the Codex AI assistant, not a human
manufacturing authority, countersignature, independent Gold truth, calibrated
measurement, or release decision. It covers all **41 assigned CAD events** in
C007, C008, C009, C010, C012 and C013. These are CAD-event review denominators,
not independently established observed-bend counts.

## Scope and method

The user authorized an AI technical review of the consumed advisory pack.
The [pack instructions](2026-08-25_f1_cad_line_ontology_authority_pack_v1.md),
exact `MANIFEST.json`, six named rendered PNGs and their six index-mapping JSON
sidecars were read. No raw CAD/PLY/STL, Gold coordinates, expected counts,
authority templates, controls, email attachments or held-outs were opened.
No inference, native geometry, corpus benchmark or review validator was run.

Each PNG was requested with `tools.view_image(detail="original")`. The tool
reported display downsampling of these roughly 5,209-pixel-wide source images
to roughly 4,200–4,350 pixels. All six views were inspected; small crowded
features were not promoted to certainty merely because their labels were
legible. No image edits or new geometric reconstructions were made.

The pack displays source-only line alternatives: yellow dotted curvature-center
axis, magenta dashed neutral half-turn and cyan solid observed-flank
intersection. An alternative's availability or relocation-state flag was not
treated as proof of physical correctness. In particular, C007 visibly contains
cyan marks even for sidecars marked `ABSTAIN_RELOCATION`; their mere presence
does not establish a successful independent scan fit.

Historical aggregate scores are present in inherited project context and the
pack's motivation, so this is not a claim of complete historical blinding.
Neither those scores nor earlier AI dispositions were used to choose the
per-event recommendations below. The viewed assets themselves declare that
Gold, expected counts and prediction scores are not displayed.

## Byte identity

Before image review, all 12 named asset SHA-256 values were computed and matched
their exact `MANIFEST.json` entries. The manifest's computed SHA-256 also
matches the released pack hash:

`c52c7399cd7b3263e8f2dd3b9191c0dc356b7175a4ea9e5cb1cf5069a89e2c3f`.

Assets are relative to `output/research/cad-line-ontology-authority-pack-v1/`.
This verifies reviewed-byte identity against that manifest, not the truth of
its declarations or independent source/provenance authenticity.

| Case | `visual/<case>.png` SHA-256 | `visual/<case>.json` SHA-256 | Events |
| --- | --- | --- | ---: |
| C007 | `f602944f9916610d2b373ab4d24408b64b6b9bf8e65029a67a95a9d90aecf836` | `95e53eb5e6a305e1e44c7db6af1629713d9ad04ef380bf039d7a7d943cf9a8fc` | 9 |
| C008 | `710d23bf68b86d19857823caf8b60b6c334514f860be3d848199b3469a47b69c` | `77206939fb5ef5774379f0847358fbe136275c55b69b6cace5b09f47908ce382` | 2 |
| C009 | `f8dc227b1d8352c888cb38a5ead079dc1dcca7cc3e54d4f1d5c6f439729c1e3f` | `30a621924bf816a81b61d65db5993e83964aea9ac29bc093fe272ea5f3b3e31e` | 8 |
| C010 | `744eb3a0b260f0294fb3c31d0b5c6604c81c3e00c79e6cd915b06fed8bbd4128` | `c1c78741d2eead29324a087e4923a516a57d78c5725c956e35876f213c99f6e9` | 5 |
| C012 | `390ef029ba9dc2edc4879421134f665606b905b9419af259269cbe74db76389f` | `d8d9807aa2e1c01a1707f7b07d781a95114a7dca21a8c1bb0ea6c5f905543eff` | 5 |
| C013 | `57b81ce54bf8606413d6c5f5a0137dec4f857c5013753d70cf1a7d0e7a83a60a` | `938a6100f6d15be0c04ac230bd29467494196cccb222d76cc2b60b8f88e9b4ee` | 12 |

## Decision semantics

- `YES`: provisionally plausible as a separately reviewable formed-bend feature
  from the rendered geometry. It does **not** approve manufacturing-event
  identity, count truth, acquisition coverage or any displayed line convention.
- `UNRESOLVED`: the available views do not sufficiently distinguish a formed
  bend event from a boundary, small transition, split/compound feature or skin.
- `NO`: positive visual evidence against bend plausibility would be required;
  none of these 41 events warrants that stronger negative finding.

**Line-placement convention remains `UNRESOLVED` for every one of the 41
events. Physical finite endpoints also remain unapproved for every event.**
Image alignment cannot decide whether the inspection definition should be an
off-surface center axis, neutral line or extrapolated flank intersection.
The rows below distinguish geometric plausibility from endpoint concerns; they
do not supply replacement coordinates or silently trim/pad any alternative.

## C007 — nine of nine reviewed

The oblique/reverse/underside views show a tray-like panel with perforated side
walls and local returns/mounting tabs. Multiple events collapse together in an
orthographic projection; their separated heights and local tab locations are
more informative than apparent projected coincidence.

| Visual index | Exact event ID | QC plausibility | Visual support and finite/identity concern |
| ---: | --- | --- | --- |
| 1 | `CAD_EVENT_0001` | YES | The upper-left segment in PC1/PC2 maps to a local wall-to-mounting-tab turn in the oblique/reverse views. Both tab and wall are visible, supporting a formed junction rather than a free hole edge. Line strokes obscure the exact curved junction and tab limits; `ABSTAIN_RELOCATION` prevents treating cyan as an observed fit. |
| 2 | `CAD_EVENT_0004` | YES | A short return at the end of a side wall is visible in reverse and underside views. It is at a different height from the nearby lower feature, so projection overlap alone does not establish a duplicate skin. The short span and corner relief leave exact endpoints unresolved. |
| 3 | `CAD_EVENT_0005` | YES | The lower wall junction near the same end is visible as a distinct local turn in the oblique and underside views. Its placement is geometrically different from the upper return marked 2. The very short overlay masks the transition into the adjacent trimmed edge; no physical trim is approved. |
| 4 | `CAD_EVENT_0006` | YES | A longer return runs along the top of a perforated side-wall section. The flange and wall are visible in both reverse and oblique views. The cyan and CAD carriers are visibly offset; that is a convention issue, not evidence of multiple bends. Stops near the cutouts/end relief require a finite-span rule. |
| 5 | `CAD_EVENT_0008` | YES | The short return at the other side-wall end has visible material on either side of the turn in reverse/underside views. Its elevated location separates it from lower tab event 6. Endpoints lie close to a wall end and cannot be independently read through the thick strokes. |
| 6 | `CAD_EVENT_0009` | YES | A lower mounting-tab root is apparent in oblique and underside views; the tab includes a nearby hole and meets the wall at a local turn. It is not merely a cylinder around that hole. The displayed segment is not independent observed-fit evidence because relocation abstained; short-root endpoint support remains unquantified. |
| 7 | `CAD_EVENT_0011` | YES | A short lower corner/root is traceable between oblique and reverse views, spatially separated from the longer upper return 4. Material turns into the panel/adjacent tab region. Corner blending and the compact overlay prevent approving its exact split from adjacent edge transitions. |
| 8 | `CAD_EVENT_0012` | YES | The central local mounting-tab return is visible on the opposite wall in underside and reverse views. Its local flange occupies a different height from event 4 despite their PC1/PC2 overlap. Relocation abstained; finite tab-root support must not be inferred from the carrier length. |
| 9 | `CAD_EVENT_0014` | YES | Another local wall/tab turn is visible at the mid-side mounting feature, distinct from endpoint tab 6. The oblique view shows the tab surface and wall. Exact segment ends are obscured near the small tab outline, and the abstained relocation provides no scan-fit qualification. |

## C008 — two of two reviewed

| Visual index | Exact event ID | QC plausibility | Visual support and finite/identity concern |
| ---: | --- | --- | --- |
| 1 | `CAD_EVENT_0008` | YES | The shorter line follows a bent interior flange associated with the large opening. Oblique and underside views show this flange distinct from the long perimeter wall. Cyan and the CAD alternatives lie on visibly different transverse carriers in PC1/PC2; visual proximity to the opening edge cannot choose the inspection definition. Opening-end and transition trims remain unresolved. |
| 2 | `CAD_EVENT_0010` | YES | The long perimeter wall-to-panel turn is visible along the side of the part, separately from interior event 1. The scan rendering supports a substantial continuous wall, but the displayed line does not establish which portion of the interrupted/end-tab geometry belongs to this finite event. No extension to an entire apparent side length is authorized. |

## C009 — eight of eight reviewed

The oblique views expose two spatially separated levels of turns around the
rectangular opening/frame. Top-view overlap must not automatically merge those
levels into one event or classify them as two skins of the same cylinder.

| Visual index | Exact event ID | QC plausibility | Visual support and finite/identity concern |
| ---: | --- | --- | --- |
| 1 | `CAD_EVENT_0005` | YES | A short-side upper return is visible in oblique/reverse views, above the lower perimeter junction. The wall and turned strip are distinguishable. Line ends approach open/mitered corners; no measured end plane is available. |
| 2 | `CAD_EVENT_0006` | YES | A long-side upper return is visible beside the central aperture in the oblique and underside views. It occupies a different level from the lower outer event 7. The line is plausible along the formed strip, but endpoint positions near the corner relief are not independently resolved. |
| 3 | `CAD_EVENT_0007` | YES | The opposite short-side upper return is visible above the lower event 8. Its finite strip and surrounding wall distinguish it from a projected duplicate. Displayed alternatives end near the two adjacent corner transitions; their exact physical trims remain uncertain. |
| 4 | `CAD_EVENT_0008` | YES | The other long-side upper return is continuous in the oblique/reverse views. It is separate in height from event 5. Cyan and the CAD alternatives track the same general formed strip without establishing which is the QC carrier. End reliefs are not a basis for extrapolating the line. |
| 5 | `CAD_EVENT_0009` | YES | The long lower wall-to-frame junction can be followed in oblique and reverse views beneath event 4. The two-height structure supports separate geometric bends rather than a skin duplicate. Apparent crossings at projected corners do not identify shared endpoints in 3D. |
| 6 | `CAD_EVENT_0010` | YES | A lower short-side junction follows the outer frame, below upper return 1. Both participating surfaces are visible in reverse/underside views. Corner blending, rather than a visible straight free edge, is the main identity/endpoint ambiguity. |
| 7 | `CAD_EVENT_0011` | YES | A lower long-side junction lies on the perimeter opposite 5, with visible wall/frame material along it. It is distinct from upper event 2. Alternatives approach outer corners but the renderer cannot establish whether the finite support stops before a miter or includes a corner blend. |
| 8 | `CAD_EVENT_0012` | YES | The remaining lower short-side junction appears below upper return 3 in the oblique view and continues to the neighboring lower sides. The projected line crossing in that view is not evidence of an extra event or duplicate. Physical corner trims remain unresolved. |

## C010 — five of five reviewed

| Visual index | Exact event ID | QC plausibility | Visual support and finite/identity concern |
| ---: | --- | --- | --- |
| 1 | `CAD_EVENT_0002` | YES | A short mounting tab bends out of the long panel at one end. The tab and nearby hole are visible in the oblique/reverse views. Relocation abstained; neither the cyan mark nor its apparent tab-width span qualifies an observed line or exact endpoints. |
| 2 | `CAD_EVENT_0003` | YES | The diagonal panel boundary in PC1/PC2 corresponds to a visible change of panel direction in the oblique and underside views. It is geometrically separate from the parallel step lines. Nearby slots and termination reliefs complicate finite extent; relocation abstained. |
| 3 | `CAD_EVENT_0004` | YES | One long step/return junction is visible along the formed region in all three oblique views. The alternatives are conspicuously separated transversely in PC1/PC2, and the cyan end appears to reach past the visible local panel edge in that projection. This is an extension warning, not a proven 3D violation; do not approve its whole segment or choose the best-looking carrier. |
| 4 | `CAD_EVENT_0005` | YES | The second long junction of the step is visible across the part, spatially distinct from 3. Both adjacent surface regions can be seen in reverse/underside views. It is plausible as another formed turn rather than a second skin, but short end transitions and abstained relocation leave the finite observed line unqualified. |
| 5 | `CAD_EVENT_0006` | YES | A second short mounting-tab root is visible near the central/end region, separate from tab 1. Its small turned surface gives positive bend plausibility. The dense small marker and abstained relocation leave its exact root span and endpoint termination uncertain. |

## C012 — five of five reviewed

| Visual index | Exact event ID | QC plausibility | Visual support and finite/identity concern |
| ---: | --- | --- | --- |
| 1 | `CAD_EVENT_0001` | YES | A short terminal return at one end of the tall end wall is visible in reverse/underside views. PC1/PC3 overlaps it with 4, but PC1/PC2 separates their locations. The tight corner overlay and abstained relocation do not establish exact finite support. |
| 2 | `CAD_EVENT_0002` | YES | The long perimeter wall junction is visible along one side of the panel, with material turning out of the main sheet. Cyan is transversely offset from both CAD alternatives in PC1/PC2. The adjacent edge breaks and end-tab geometry require an explicit finite-span convention; line length alone is not observed coverage. |
| 3 | `CAD_EVENT_0006` | UNRESOLVED | The very short mark sits at the end of the formed opening feature, where an aperture edge, local upright/return and the longer flange meet. Oblique views show nearby material but not enough separation to decide whether this is a separate QC bend, a rounded end transition or a split representation of the opening flange. Relocation abstained. Do not count it as an independent observed bend from this pack. |
| 4 | `CAD_EVENT_0009` | YES | The short return at the other end of the tall end wall is visible in oblique/reverse/underside views and spatially distinct from 1. Its local turn is plausible, but the line ends sit at small corner/edge terminations and relocation abstained. |
| 5 | `CAD_EVENT_0010` | YES | The opposite long-side formed junction is visible in oblique and reverse views. The yellow carrier visibly extends farther longitudinally than the magenta/cyan alternatives. This pack therefore leaves a span issue as well as transverse placement: neither shortening to cyan nor extending to yellow is an approved physical endpoint rule. |

## C013 — twelve of twelve reviewed

This case is densest. Three long nearly parallel lines and the tiny features
at the opening ends require explicit distinction between a return, a main
wall bend, a rounded rim and a localized corner transition.

| Visual index | Exact event ID | QC plausibility | Visual support and finite/identity concern |
| ---: | --- | --- | --- |
| 1 | `CAD_EVENT_0001` | YES | A short transverse turn bounds one end of the tall-wall return region. Reverse and underside views show a finite strip connecting the adjacent long formed regions. Corner meeting points are crowded by three alternative carriers; the exact split and endpoints remain unresolved. |
| 2 | `CAD_EVENT_0002` | YES | A long perimeter panel-to-wall bend is visible below the interior opening flange. Oblique/underside views separate it from event 6 and from the end-wall features. Thick line overlays obscure end reliefs; no full-length physical support is inferred from continuity of the nominal carrier. |
| 3 | `CAD_EVENT_0003` | UNRESOLVED | A long outer rim line runs parallel to 4 and 11. The side projection shows a rounded-looking upper return/rim, but these views do not settle whether this candidate is an independent rolled/hem bend, a boundary of the adjacent return, or another skin representation. The sidecar lacks a curvature-center alternative. That absence is not proof of non-bend status, but this carrier cannot be approved as a separate QC event here. |
| 4 | `CAD_EVENT_0004` | YES | A long inner turn between the tall wall and its return is visible in reverse/underside views. It is spatially separated from lower main-panel junction 11, with intervening wall material. Its relationship to the outer rim 3 still needs manufacturing-event adjudication; exact corner termination is not approved. |
| 5 | `CAD_EVENT_0005` | UNRESOLVED | The tiny short line at one opening flange's end is traceable beside longer event 6, but it sits in the compound end transition. Available pixels do not distinguish a separately formed bend from a side/end face or a fragment of that flange's turn. A successful relocation flag cannot supply that event ontology. |
| 6 | `CAD_EVENT_0006` | YES | The longer interior opening flange has an evident turned strip and neighboring panel surface in oblique/reverse/underside views. This is positive bend plausibility independent of the tiny adjacent event 5. The finite endpoint at that compound transition remains unresolved. |
| 7 | `CAD_EVENT_0007` | UNRESOLVED | The corresponding tiny end mark beside opening-flange event 8 lies where an upright edge and the longer bent strip terminate. Material is visible, but the short stroke and corner overlap cannot establish an independent QC event rather than a compound/trim feature. It must remain separate from any count claim pending a clearer ontology decision. |
| 8 | `CAD_EVENT_0008` | YES | The other long interior opening flange is visible as a bent strip in oblique and underside views, separate from perimeter bend 10. Adjacent event 7 leaves the termination/splitting rule unresolved; the visually supported flange must not automatically inherit an approved whole-span endpoint. |
| 9 | `CAD_EVENT_0009` | YES | A short transverse return at the opposite end of the tall wall is visible and distinguishable from event 1 by location. It connects the same general multi-level region but does not visually duplicate 1. Dense corner intersections prevent assigning shared physical endpoints from the plot. |
| 10 | `CAD_EVENT_0010` | YES | The other long perimeter panel-to-wall turn is visible behind/beside opening flange 8. The surfaces and their height difference support a distinct formed bend rather than another skin of 8. End reliefs and nearby short return 9 remain unresolved finite boundaries. |
| 11 | `CAD_EVENT_0011` | YES | The lower tall-wall-to-main-panel junction is visible below the long upper turn 4. An intervening wall separates them in PC1/PC3 and the oblique views. Orthographic near-coincidence is therefore insufficient to merge them; the corner endpoint and relation to rim 3 require explicit review. |
| 12 | `CAD_EVENT_0012` | YES | The long junction on the opposite end wall is visible along the perimeter and separated spatially from the tall-wall group. Two adjoining surface regions are apparent. The three line alternatives track that formed region, but terminal corner blends and displayed line width prevent exact trim approval. |

## Summary and actionable limits

| Case | Reviewed | Provisional YES | UNRESOLVED | NO |
| --- | ---: | ---: | ---: | ---: |
| C007 | 9 | 9 | 0 | 0 |
| C008 | 2 | 2 | 0 | 0 |
| C009 | 8 | 8 | 0 | 0 |
| C010 | 5 | 5 | 0 | 0 |
| C012 | 5 | 4 | 1 | 0 |
| C013 | 12 | 9 | 3 | 0 |
| Total | 41 | 37 | 4 | 0 |

The four unresolved event IDs are `C012:CAD_EVENT_0006`,
`C013:CAD_EVENT_0003`, `C013:CAD_EVENT_0005` and `C013:CAD_EVENT_0007`.
None has been deleted, relabeled as negative, or excluded from the denominator.

Engineering follow-up supported by this review is to retain separate fields for
event identity, geometric carrier convention, finite endpoint convention and
observed support. The visible endpoint/extension questions in C010 event 3 and
C012 event 5 deserve explicit treatment, while C013's neighboring long lines
must not be merged solely on projected proximity. These are review findings,
not permission to alter Gold or optimize a localization threshold against it.

All 41 line conventions and all 41 physical endpoint definitions remain
unapproved. This advisory may organize a research hypothesis or a human review;
it cannot close the pack's two-person manufacturing-signoff gate, establish new
Gold, authorize physical access, validate a score comparison or greenlight
production/floor testing. Authority templates and product source were unchanged
by this visual-review subtask.
