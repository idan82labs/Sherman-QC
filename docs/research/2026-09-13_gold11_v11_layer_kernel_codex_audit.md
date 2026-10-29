# V11 layer kernel — preliminary independent Codex source audit

Review closed 2026-09-13 23:39 UTC. **Not source-closed at this snapshot.** Fable remains the sole engine/test editor; the candidate is an unfrozen draft. This is a bounded source review, not physical PRE, execution evidence, a score forecast, or a request to broaden the successor.

Read canonical instructions/current checkpoint, the V3 focused test file, and the V2-to-V3 source delta plus the connected local kernel paths. No imports, tests, geometry, physical/derived attempt artifacts, raw inputs, Gold, or caches were accessed for this review. The following hashes matched at the initial and 23:39 pre-report repeat checks:

| Source | SHA-256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `12bb9ce9e595ada6b4cf86947089ef96f2158ddf36009584442e0a414e6ae008` |
| `backend/cad_guided_measurement_v3.py` | `aa37dd4f07a533c7b45c611e59930e94c996f7bd4eeeb0d65d3843334c8d6271` |
| `tests/test_cad_guided_measurement_v3.py` | `835a0db3a69aff8cc6c906c4b8c1299890b57d80277d7229e9912fa0a0d7ef70` |

The post-write check observed source drift to kernel `117cd188d07f3f6b32b4fe65b3c8c08edafe4958e67c75943bd1965d0cd8d3f2` and test `db0ffd75a34c39be4eac0566a3ca304cacad8335b257d955bebda52777beb919`. Those newer bytes were not reviewed here. Findings and line anchors below apply only to the pinned earlier snapshot, not an assertion that the active author has left them unresolved in later work.

## Concrete closure requests

### P1 — The anti-trimming band premise is recorded, not enforced

V3 `:467–468` publishes `band_exceeds_declared_residual`, but neither policy validation (`:119–143`) nor `_measure`/selection rejects an enabled policy with `layer_membership_band_mm <= feature.max_fit_residual_mm`. Such policy/feature combinations are valid under both dataclasses. The header's claim that the band is wider than the residual gate is therefore not an invariant of this API.

The defaults 0.1 mm and 0.03 mm satisfy the premise. This is not a claim that the default physical configuration already violates it. It is a reachable configuration-contract defect: a narrowed membership band can exclude exactly the rows the downstream residual gate was supposed to evaluate, while the same selected-carrier route remains available.

Minimum closure: jointly validate enabled layer policy against each feature before selection; reject equality as well as smaller bands, retaining declared disabled behavior. Add below/equal/just-above boundary tests and a configuration test proving a caller cannot merely publish `band_exceeds_declared_residual=false` and proceed. Changing the feature's residual gate to fit the band is not closure.

### P1 — Final projection-premise check can refer to the previous membership

V3 `:506–522` fits planes to the current labels, then computes new labels and assigns them at `:521`. If the last permitted iteration changes membership, the loop ends with **planes fitted to the previous labels, but the newly assigned labels are accepted**. The projection tilt gate (`:526–538`) checks those previous-label planes. Pair compatibility then refits the final member rows (`:543–550`, via `:408–410`), and `_plane` refits again. Neither final-label fit is checked against the crop-normal premise; the later flange orientation gate is against the nominal reference and allows a different angle.

Thus `refit_converged=false` can accompany an accepted pair whose published projection-premise angles are not the angles of its actual accepted row populations. This is a source-level ordering gap, not an executed assertion that one of the existing fixtures or physical parts crosses the boundary.

Minimum closure: either fail closed on an exhausted non-fixed-point membership, or validate the actual final-label planes and publish their relationship to the membership witness under an explicitly bounded finalization rule. Do not append an unbudgeted iteration or silently replace labels after their checks. Test `layer_refit_iterations=1` with a genuine membership change and a final-label tilt crossing the declared premise; require the exact intended disposition, not an OR-list of unrelated failures.

Accounting detail: `refit_iterations_used` is incremented before the `<3 rows` check (`:507–509`), so it can report one iteration when zero TLS refits ran. If the field means attempted loop entries, say so; otherwise count attempted and completed plane refits separately. The current limit is finite, but the telemetry is not an exact completed-work count.

### P2 — Third-population admission uses single-bin support in addition to band mass

The policy comment describes a third mode by its band mass relative to the smaller population. Actual rejection at `:355` also requires **one peak bin** to contain `min_layer_rows`. This may miss a third population with enough rows in its declared band simply because it straddles a bin boundary.

A small independent boundary probe should contain 40 coordinates at 0, 40 at 1.5, and 6 each at 2.98 and 3.02, with default 0.05 mm bins / 0.1 mm band / minimum 12 / ambiguity fraction 0.25. The third band has 12 rows versus a threshold of 10, but each of its peak bins has only 6. The source's single-bin condition skips that candidate. No probe was executed here; this arithmetic specifies the missing regression. A full plane fixture can place those normal offsets on independently balanced in-plane grids.

Minimum closure: decide explicitly whether `min_layer_rows` is band-population support or single-bin peak support. For the stated population-support interpretation, apply it to the candidate band membership rather than one bin, and test bin-phase/noise splitting and the just-below-threshold case. If the extra peak-density requirement is intentional, declare it independently rather than implicitly changing the meaning of `min_layer_rows`; retain the limitation that an adequately sized third population can otherwise be treated as unassigned rows.

### P2 — Published “band mass” is a wider quantized bin window

`_two_modes` documentation (`:312`) says rows within `+/- layer_membership_band_mm`. Actual `mass()` at `:331–333` sums all bins from `i-ceil(band/width)` through `i+ceil(band/width)`, inclusive. At defaults, that is five 0.05 mm bins: approximately +/-0.125 mm around the center, not +/-0.1 mm. This changes the third-mode ambiguity threshold and evidence near its boundary; widened bins can amplify the difference.

Minimum closure: either compute exact coordinate-band mass with explicitly bounded work, or name and publish the actual quantized interval semantics separately from the geometric membership band. Test points just inside/outside +/-0.1 mm and a widened-bin case. Do not claim an exact band probability/mass from this approximation.

## What is sound, and what the tests do not establish

- The new mechanism does not use Gold or nominal radius to select the better measured answer. The final flange/circle/tangency/finite gate forms remain unchanged in the diff, and the inner-eligibility tuple remains empty. Generic conditional contact geometry remains allowed without verified skin identity.
- Population fitting is residual-based row selection: nearest fitted-plane distance drives membership. Therefore “fit quality never selects” is too broad. The accurate distinction is that no downstream measurement/gold-fit score chooses between hypotheses; the new segmentation itself does use fitted residuals.
- The selection mask is generated by nearest-plane membership, with `np.argmin` ties assigned to plane index 0. The final published planes reproduce the last labels even when the TLS loop did not converge, because those labels were just generated from those planes. **Reconstruction consistency does not repair the stale projection-premise fit above.**
- `kept_rule` at `:557` omits the nearest-plane/tie clause and only describes being within the lower plane's band. For nonparallel overlapping bands that is not the complete rule. Publish the nearest-plane and stable plane-index tie semantics explicitly. The test at `tests:155–165` independently recomputes distances but checks only subset inclusion on the whole cloud; it does not require exact equality within the actual flange crop. Add exact cropped-mask/count equality, both labels, excluded rows, overlap and equal-distance ties. Such tests can remain small and synthetic.
- The finite loop bound (1–8 iterations), 4096-bin cap and bounded local-maxima reporting prevent an unbounded iterative search in the reviewed default domain. Additional work includes two membership passes/plane fits per completed iteration plus pair refits; retain it in the successor's declared budget. This audit did not benchmark the cost or verify a new launcher budget.
- `dropped_other_skin_rows` and `dropped_bridging_rows` are correctly separate; total dropped plus kept matches the input at the selection return. On refusal, all rows stay kept and actual dropped counts remain zero, while proposed group/bridge diagnostics remain evidence of the attempted selection.

## Warped-surface and other adversarial gaps, within the declared exploratory model

The current contaminant fixture (`tests:75–82,189–201`) **adds** displaced points while retaining the original clean surface rows. It does not replace a surface patch with a dent or warp. A genuine deformation outside both bands may therefore be excluded as “bridging,” leaving an apparently clean selected population. The published exclusion count makes the operation inspectable, but does not prove excluded rows are cut edges or contamination rather than physical deviation.

Do not block all generic selected-support geometry merely because this distinction is unresolved. Do explicitly scope those rows as **unassigned outside-band rows, not verified cut-edge/bridge identity**, and retain unknown whole-surface coverage, inner identity and conformance. Add a small independently constructed control that replaces a connected patch with warped rows, rather than adding duplicates. Require either abstention or an explicit incomplete/unresolved-support claim with every removed row accounted; never a whole-surface “good”/conformant result. Preserve an inside-band but above-delta deviation as a residual refusal.

The projection test (`tests:317–322`) accepts four possible refusal enums and checks the angle only if the desired enum happened. It does not establish that the projection gate itself was exercised. The three-population test uses 400-row concentrated modes, so it misses the minimum-band-support/bin-phase case above. No current assertion forces nonconvergence, exact membership ties, a replaced warped patch, or exact reconstructed crop equality.

The independent full context/cover positive fixture is separately owned by Fable; this audit does not claim the public-entry fixture is equivalent to that path. These focused closures should precede a stable source-acceptance review. No new physical rerun, score comparison, or broader research gate is requested here.

The V3 header and test preamble also repeat the superseded V10 narrative claims ("86 flange fits," "every crop both skins," "never separated first flange," 1.3–1.6 mm). Correct them using the separate completed V10 measurement postmortem; those statements are not evidence for this draft's success.

## Bounded closure delta — 2026-09-14 01:06 UTC

**ACCEPTED in the bounded kernel source scope.** Both original P1 findings and both P2 algorithm-contract findings are closed at the hashes below. No new implementation blocker was found in the requested delta. This is not executed-test evidence, complete V11 bundle acceptance, physical PRE, or proof of physical accuracy. The earlier rejected snapshot and findings above remain unchanged.

Opening and 01:06 closing checks matched:

| Read source | SHA-256 |
| --- | --- |
| `backend/cad_guided_measurement_v3.py` | `cf978d8617a190ca8db29d1f31bb859a3e123d14c7c21c4fade0cf322b2e1439` |
| `tests/test_cad_guided_measurement_v3.py` | `9bf3cb07d517951e64ba237642aababf45e7b800194d6ffbb179a8514a3085c5` |
| `docs/research/2026-09-14_fable51_v11_review_response.md` | `e02d9db179fef77d79496b953b9c59e8d4bb95f06aa10918cdc8afe82a1b6c88` |

Read the full 100-line author response, current-status narrative, and the changed kernel/test paths; compared the kernel with unchanged V2. No imports, tests, geometry, running-job outputs, attempt artifacts, raw inputs, Gold, or caches were accessed. Fable remains the sole implementation editor. The author-reported 912 passed / 1 skipped is not independently executed or receipt-verified by this review; separate full-connected fixture work remains outside this acceptance.

| Original issue | Source closure and nonvacuous test assertion |
| --- | --- |
| P1 joint band premise | `_measure:803–806` rejects enabled band <= feature delta before cropping/selection. Test `:338–346` requires exact ValueError for equality/below, successful above-bound invocation, and disabled-policy `_Unavailable` rather than the contract ValueError. Private selection remains a helper; measured output goes through the joint check. |
| P1 stale final membership/premise | Selection `:552–596` admits only `new_labels == labels` within the finite loop. At that point published planes were fitted from exactly the admitted labels; the subsequent tilt and pair tests operate on the same partition. Exhaustion/starvation retains all crop rows and an exact unresolved reason; no extra refit occurs. Test `:359–385` uses genuinely tilted original rows, asserts exactly one completed refit then fixed-point refusal under cap 1, and exactly two completed refits with converged selection under the default cap. |
| Refit accounting | `:554–575` distinguishes attempted loop entries from completed two-plane refits. A starved entry increments only attempts. Clean and exhausted-loop tests assert both counts; the starved branch itself is source-verified, not claimed directly exercised by those tests. |
| P2 third-mode support | `:387–394` uses exact band mass >= minimum population rows, not the peak bin. Test `:388–399` implements the 40/40/6/6 boundary construction and asserts ambiguous band mass 12 despite peak count 6; the 11-row variant remains below the floor. |
| P2 exact mass | `:356–366` counts inclusive binary64 interval membership with sorted-coordinate `searchsorted(left/right)`. This is O(N log N) plus bounded O(log N) lookups per histogram maximum, not the previous five-bin window. Test `:400–419` checks 0.0999/0.1001 offsets, exact mass 12 rather than 18, and independently counts widened-bin intervals around the published centers. |
| Projection-premise test | Test `:318–334` now requires the exact premise-refusal enum/reason, converged 600/300 membership, explicit 11–13 degree rejected angle, no pair-compatibility call, and all 900 rows kept. It cannot pass via the earlier OR-list of unrelated failures. |

### Membership and exclusion change

`MEMBERSHIP_RULE` (`:398–419`) now exactly declares nearest absolute-plane distance, stable plane-index-0 ties, inclusive within-band assignment, and unassigned rows only when outside both bands **and strictly between the co-oriented planes**. For co-oriented normals the signed-distance product <0 is the declared between-plane test. Outside-sandwich rows remain assigned to the nearest population. At a membership fixed point, the published planes and rule determine the same admitted row labels; the prior stale-final-fit loophole is gone.

Tests `:437–479` explicitly assert tie, overlapping-band and between/outside labels, then exact actual-crop kept-row equality, both group counts, unassigned counts and outside-sandwich counts. The exact reconstruction uses the production membership helper, so it is an integration-consistency check, not an independent proof of that helper; the separate hand-expected label assertions test the helper's stated rule. The older subset-only assertion remains redundant, not the basis of this acceptance.

Opposite-population exclusions and unassigned-between-plane exclusions remain separate (`:620–625`), and `diagnostics.flange_support` (`:851–859`) explicitly says sample membership subset, no whole-crop planarity certificate, unverified excluded-row identity, and coverage not established. Original-row displacement tests now exist (`tests:482–545`): single-population dent refusal; between-skin displacement with counted exclusion and no displaced retained row; exact above-fraction refusal; air-side dent retained and refused; and a contiguous warped-span refusal. The warped-span assertion intentionally accepts multiple refusal mechanisms, but cannot pass by silently discarding rows and returning a measurement. These tests do not establish physical skin identity and do not need to do so for the generic conditional carrier contract.

The default bounded refit allowance is now 6 rather than the preliminary draft's 3, with constructor maximum 8. That is a candidate policy change to bind in the eventual frozen protocol, not an alteration to any started experiment. The secondary mode also now requires band mass >= `layer_mode_ambiguity_fraction * primary_mass` (`:375–383`); this is explicitly published as `population_mass_floor`, rather than selecting a small displaced tail solely by peak height. Neither change uses downstream measurement, nominal radius or Gold agreement to choose the answer.

### Non-blocking claim correction, not a request for a new outer-surface gate

The author response's blanket statement that an outside-sandwich displaced/foreign population is "refused by the unchanged gate," and kernel header `:47–48`, overstate what is implemented. Every such row is assigned to a population, but only the **selected lower population** reaches the strict feature `_plane` maximum-residual gate; the discarded upper population receives the pair-compatibility RMS/rank/parallelism checks. The existing `far_rows` test (`:459–479`) expressly expects a successful generic measurement with 1% rows 0.7 mm beyond the upper skin assigned to that upper population. They are accounted as opposite-population exclusions, not unassigned rows, and do not necessarily invalidate the selected support.

Narrow the claim to: outside-sandwich rows are not discarded as unassigned; they remain in their nearest population, whose applicable checks may refuse it, while only selected support receives the strict measurement residual test. This is consistent with unknown whole-part conformance and the generic conditional carrier. Do **not** add an outer-skin maximum-residual gate solely to make the prose true. Also replace residual "fit quality never selects" wording with "no downstream measurement/nominal/Gold score chooses a hypothesis"; the segmentation itself plainly uses fitted residuals. These are evidence-description corrections, not remaining runtime blockers in this bounded scope.

## Final kernel claim/hash reconciliation — 2026-09-14 01:39 UTC

**Bounded kernel source acceptance retained** at `backend/cad_guided_measurement_v3.py` SHA-256 `8fd5c3e6a7f6437638f068052d8c71e085980b25fa7d14cca955cf5f66c5d6e6`; focused test file remains exactly `9bf3cb07d517951e64ba237642aababf45e7b800194d6ffbb179a8514a3085c5`. Both hashes matched opening and closing reads. Canonical AGENTS/current status were read. No tests, imports, geometry, live outputs, raw/Gold/attempt artifacts or caches were accessed; only this append is written. This remains a narrow source/claim reconciliation, not full-bundle PRE or independent execution evidence.

Rechecked the four accepted closures in the current source: enabled joint band > delta guard (`:810–813`); completed-refit accounting and mandatory membership fixed point before projection/pair admission (`:559–612`); third-population minimum applied to band mass (`:389–398`); exact inclusive binary64 interval mass via sorted-coordinate searches (`:359–369`). The reviewed membership equation and nearest-plane tie behavior (`:421–426`) remain intact. No new blocker was found in these paths.

The selected-versus-opposite-surface correction is now explicit in the header (`:31–33,48–53`), `_band_membership` docstring (`:413–420`) and published `MEMBERSHIP_RULE` (`:406–409`). The opposite surface receives only its applicable pair checks, is not measured, and has unknown conformance. This is more than a comment-only byte change: the descriptive membership metadata gains the explicit `opposite_surface` entry, without changing the inspected membership computation. The remaining literal "fit quality never selects" at header `:16` is the previously noted non-blocking prose overstatement; interpret it as no downstream measurement/nominal/Gold selection, not absence of residual-based population fitting. It does not reopen the accepted algorithm findings or require a broader gate.
