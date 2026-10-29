# V9 layer kernel: bounded source-delta audit

Date: 2026-09-13. Reviewer: Codex/Astra `v8_flange_policy_math_audit`.
Inspected `backend/cad_guided_measurement_v2.py` fully; SHA256 before and
after reading was
`9a2e7da3a58838c5de74085b197547550bcbcd1b53804754b609bba5302c9baa`.
Baseline reviewed snapshot was
`59555408cf92d3bdb7eae711b5ed7edfc68c0d9d518bb683c78e30dd61630d6d`,
as recorded in the prior layer-design report. All line references here are
to the newer exact kernel snapshot, not an unreviewed subsequent revision.

Scope: selection mathematics, unsupported mixtures, diagnostic retention and
specific earlier source concerns. No imports, tests, experiments, physical
data, Gold, attempt or cache access occurred. Only this report was written;
prior reports and engine/test sources were unchanged. Consumer connections
are work in progress and belong to Fable/the separate chain reviewer; they
are **not accepted by this report**.

## Decision

Several concrete early-snapshot issues are closed at source: signed geometry,
disabled arc mode, the nominal-bound certificate overclaim, actual ray extent,
and common fit/axis/sweep diagnostic propagation. The revised gap partition
and pair checks are a useful, explicitly unverified selection heuristic.

Remaining P2 items concern incomplete selection provenance/diagnostic exits
and names/docstrings that overstate what the heuristic establishes. They do
not require solving arbitrary material geometry before a generic conditional-
contact development score. Per the parent's checked scorer scope, unresolved
material side can remain a generic diagnostic output; it must not silently
become verified inner geometry or manufacturing truth. `INNER_FACE_ELIGIBLE_
STATUSES` and connected claim eligibility are delegated to the chain review,
not greenlit here.

## Closed or materially improved at this source snapshot

| Earlier issue | Delta inspection | Status |
| --- | --- | --- |
| Opposite signs for radial versus fitted normal | Lines 9–24 explicitly distinguish them; layer coordinate uses the TLS normal oriented to +radial at 355–362; enabled arc filtering uses fitted-normal signed distance > exclusion at 638–645 | Closed at source; no executed test claim |
| Disabled layer mode still changed arc mask | Disabled flange path retains all rows at 352; disabled arc path restores `abs(signed)>exclusion` at 646 | Closed at source for row-selection semantics; not byte-identical telemetry/runtime-cost equivalence |
| Nominal formula called complete observed exclusion | Lines 650–670 publish `certificate=False`, explicit idealized assumptions, two qualifying observed gap values, domain refusal instead of clipping, and heuristic-only status | Earlier certificate overclaim closed at source; not actual side-isolation evidence |
| Largest invalid gap hides a smaller qualifying split | Lines 247–265 enumerate every gap meeting the global left/right population floor, rather than inspecting only `argmax` | Closed at source for that exact defect |
| Multiple qualifying gaps silently treated as two layers | Lines 371–373 retain all rows with ambiguous identity when more than one qualifying split exists | Improved, conservatively no selected prefix; sparse/unqualified populations remain outside that definition |
| Pair selection lacks separate geometric checks | Lines 284–333 check per-group RMS/rank, normal angle, mean-normal offset and two projected interval-overlap fractions | Useful declared heuristic qualification, not physical skin/footprint certification |
| `ray_extent_mm` was really second principal extent | Lines 187–233 name both principal extents separately and compute the actual supplied flange ray; callers pass each real ray at 592–598 | Closed at source |
| Plane, axis/sweep and public wrapper lose ordinary unavailable diagnostics | Lines 604–623 attach diagnostics on plane and axis/sweep `_Unavailable`; lines 843–845 preserve attached diagnostics in public error rows | Common stated paths closed; remaining numerical and late-invalidated paths below |

The free-circle objective, `_plane` gates and actual residual checks remain
visible in the inspected source (`:391`, `:451`). This was source comparison,
not a proof from re-executed tests, an exact consumer closure, or promotion
evidence.

## P2 — Published layer telemetry is a bounded summary, not lossless partition provenance

`_partition_layers` computes a stable sort, every qualifying split position,
the gap size and both side counts (`:254`). The public record retains only
the total qualifying count and the **first** `max_layers_reported` gap sizes
(`:365`), not the corresponding positions/counts or complete gap ledger.
For multiple splits it returns all rows with no group membership or complete
group statistics (`:371`). For a selected pair it publishes two means and
counts, but not the TLS projection normal/center used for sorting, local
kept/dropped IDs/bitmap, or pair-fit normals and footprint interval endpoints
(`:344`, `:379`).

The output therefore cannot reconstruct the exact selection/discard partition
from the derived diagnostic record alone. Retained input bytes and this source
could permit a separate recomputation, but that is not the same as explicit
lossless output provenance, and terminal attempts must not be replayed just
to discover which evidence was discarded.

Smallest scoped repair: either call the current schema a bounded diagnostic
summary and explicitly identify its truncation, or add a compact local-index
kept/dropped membership encoding plus the sorting normal/center and complete
or content-bound qualifying split data. The connected consumer must bind
local indices to original raw IDs; do not duplicate unbounded arrays outside
the frozen metadata budget. If only a summary is retained, publish omitted
count and summary scope rather than claiming a full lossless partition.

This is especially useful for the new mechanism: an emitted fit must show
which rows were excluded as a competing layer, not only how many. No new
physical identity certificate is requested to provide this provenance.

## P2 — Some failure exits still lose diagnostics already available

Within the before/selection/after loop (`:592`), `_select_flange_layer` can
raise from `_spectrum` or pair `_group_plane`; those numerical exceptions are
not wrapped with the in-progress diagnostics. `_flange_diagnostics` catches
its own SVD failures (`:203`), but its caller then executes a fresh unguarded
SVD in selection (`:355`). Likewise the `_plane` block attaches only on
`_Unavailable`, not NumPy linear-algebra/floating failures (`:604`). The arc
wrapper also catches only `_Unavailable` (`:679`). The public entrypoint
catches the numerical exceptions later but then obtains `diagnostics=None`
if no attachment exists (`:843`). The computed prefix and failure phase are
lost even though the final result correctly refuses.

The public shared-support invalidation at `:861` clears `measurements`, which
contained successful diagnostics, without first preserving those diagnostics
in an attempted-evidence field. This remains a diagnostic-retention issue even
when clearing the scientific measurement is the correct action.

Smallest scoped repair: attach the available bounded diagnostic state and
explicit incomplete phase on all supported numerical failure exits, without
turning the prefix into measurement evidence. Preserve already-computed
diagnostics before later ownership invalidation. If a phase was intentionally
not evaluated (e.g. ambiguous windows before the diagnostic block), mark that
explicitly rather than claiming both spectra were computed for every exit.
Connected consumers and serializers need their own exact source checks.

## P2 — “Footprint overlap” currently means only two bounding-interval overlaps

The compatibility check projects both groups onto the lower group's first
two PCA directions and compares the one-dimensional min/max intervals
(`:323`). `_interval_overlap` divides interval intersection by the shorter
interval width (`:276`). Thus complete containment of a small bounding box
scores 1 on each axis, regardless of holes or actual support in its interior.

For example, a planar lower population around an annular/trimmed footprint
and an upper planar island entirely within its unsupported central hole can
have perfect normal agreement, accepted separation, zero RMS and interval
overlap 1 on both PCA axes. No material-supported footprint overlap follows.
This is an analytical counterexample to the **name/claim**, not an executed
fixture or evidence of a physical Gold11 error.

For generic conditional-contact development, these interval fractions may
remain a useful explicitly unverified heuristic. The scoped correction is to
name them `projected_bounding_interval_overlap` (or equivalent), identify the
chosen basis and avoid the docstring's unqualified assertion that footprints
overlap (`:289`). Do not infer genuine skin pairing or area overlap from this
proxy. The chain reviewer owns whether subsequent inner-specific eligibility
uses it too strongly; that decision is not resolved here.

The PCA basis is equivariant under rigid transforms when the relevant
eigendirections are unique, up to harmless sign flips. Equal/near-equal
in-plane singular values do not define a unique in-plane basis, and bounding-
interval overlap is basis dependent. No eigengap condition or invariance
qualification is made here. Record the actual basis if exact interpretation
is needed; use a small degenerate/near-degenerate synthetic control before
claiming pose-invariant overlap behavior. This is a limitation to document,
not a demand to solve arbitrary geometry for the next generic benchmark.

## What the new pair checks prove, and what they do not

The inspected implementation requires:

- both selected partition groups have `s2/s1 >= min_plane_spread_ratio`;
- each group's TLS RMS `s3/sqrt(N) <= max_layer_rms_mm` (default 0.1 mm);
- fitted normals, each oriented toward the nominal radial, differ by at most
  the declared angle (default 3 degrees);
- centroid displacement projected on their normalized mean normal lies in
  `[layer_split_gap_mm, max_layer_gap_mm]` (default 0.3–6 mm);
- both projected bounding-interval overlap fractions are at least 0.5.

These are dimensionally consistent engineering checks (`:268`, `:304`,
`:309`, `:319`). The selected lower group still has to pass the stricter
measurement plane gates later (`:605`), so a weak fit cannot become a measured
plane merely by passing the pair RMS threshold. A candidate never switches
to whichever group later fits better: it keeps the lower-coordinate group
only if the pair heuristic passes; otherwise it keeps all rows with unresolved
identity (`:378`). That preserves a useful explicit choice and its failures.

For exact parallel layers with true normal v and separation t, orienting a
reliably identified TLS normal toward +radial makes the paired-point signed
difference positive when radial dot v >0. For nonparallel groups, the
mean-normal projected **centroid** offset is not a uniform pointwise gap.
For symmetric unit normals, that offset equals the average of the two
normal projections divided by `cos(angle/2)`; differing finite footprints
can affect it through tangent-position differences. It is not a thickness
certificate or a guarantee that extended planes do not cross elsewhere.

The initial whole-crop TLS direction also remains an unqualified sorting
proposal (`:355`). For a broad ideal parallel mixture it is the true normal
if thickness variance is the smallest covariance eigenvalue. For a narrow
patch, thickness variance can exceed a tangential variance, making the TLS
normal a tangential direction instead. Later compatibility/plane checks may
reject the resulting partition; the mechanism does not prove every possible
two-skin mixture is recognized. No such completeness requirement is needed
for the generic development score, provided failure/unknown semantics remain.

## Sparse mixtures remain outside “multiple qualifying layers”

A qualifying gap requires at least `min_layer_rows` in the complete prefix
and suffix (`:258`), not in every individual population. Gaps next to a tiny
outermost population can be unqualified and disappear from the published
gap list. Consequently `layers_detected=1` means zero qualifying split,
not proof there is only one physical or geometric layer; `layers_detected=2`
means one qualifying split, not proof of exactly two material skins.

Illustration: 100 well-distributed rows on a lower plane, 100 on an upper
parallel plane 1.5 mm away, plus one row 0.5 mm beyond that upper plane at
the same footprint center. The gap after the singleton has too few rows on
one side to qualify; the main 1.5 mm gap qualifies. In the upper partition,
the singleton contributes TLS RMS approximately
`0.5*sqrt(100)/101 = 0.0495 mm`, below the 0.1 mm pair threshold. With broad
matched footprints and parallel normals the pair can pass and the lower
plane is retained. That may still be a valid generic selected-plane hypothesis;
it does not establish exactly two skins or lossless explanation of the third
population. This example is algebraic, not an executed source probe.

Rename the count as detected qualifying groups/splits or explicitly publish
that scope. Retain unqualified large-gap counts/positions where practical so
minority populations remain auditable. The docstring that “a curved contaminant
fails the planarity test” (`:291`) should likewise say a contaminant exceeding
the declared RMS/rank bounds fails; arbitrarily shallow curvature can satisfy
any positive RMS threshold. These are claim-precision changes, not a request
for unconditional rejection of every low-support contaminant.

## Minimal focused checks for the author

Proposed only; this reviewer ran no tests:

1. Verify the closed sign and disabled-mode paths on both sweep signs, retaining
   actual selected row IDs. Disabled geometry should match the V1 masks even
   though V2 emits extra diagnostics and uses extra work.
2. Check one valid parallel pair, multiple qualifying gaps, a nonqualifying
   largest gap plus a smaller valid gap, and a sparse extra population. Assert
   selection, unresolved status and all declared truncation/provenance fields.
3. Check disjoint/holed footprints with overlapping projected boxes; assert
   the result is documented as a proxy, not observed physical footprint/skin
   certification. A single targeted case suffices to prevent the overclaim.
4. Force each numerical exception boundary and late shared-support invalidation
   with bounded invented inputs/mocks. Preserve the diagnostic ledger as
   attempted/incomplete evidence, never a surviving partial measurement.
5. Bind compact kept/dropped indices through the independently owned consumer
   to original raw rows, and retain all selected/discarded totals without
   exceeding the protocol's metadata/work budgets.

No extra physical rerun is requested by this review. Close the specific source
retention/wording issues, let the chain audit resolve eligibility propagation,
then apply the existing candidate-freeze/PRE process. The report is not final
candidate acceptance, a numeric F1 result or manufacturing approval.

Final WIP note: after this report was drafted, the source hash had advanced to
`cb63853d45d71a54aae337d4e59c0092d1bd0e8fc939625b0917c04356676328`.
That later revision was not inspected in this bounded task. Findings and
closures above apply only to the reviewed `9a2e7da3...` snapshot; they neither
reject nor accept the author's subsequent in-progress edits.
