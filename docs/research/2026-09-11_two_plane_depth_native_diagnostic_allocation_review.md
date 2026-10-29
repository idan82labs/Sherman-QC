# Native depth diagnostic allocation: independent review — 2026-09-11

## Disposition

NO NUMERICAL BLOCKER to the proposed **2520 worker CPU / 2815 work wall / 2820
parent wall seconds, 4 GiB sampled process-group RSS, 1 MiB stdout** as a finite
engineering ceiling for one separately declared native L-R integrity diagnostic.
It is not a completion guarantee, probabilistic runtime bound, qualified
measurement result, or execution release.

This review concerns
[the prospective allocation](2026-09-11_two_plane_depth_native_diagnostic_allocation.md),
SHA256 `c9a59fc1c984defa378c96b8cbdc2cedff49581bf0349c4e8d254464380da9aa`.
The existing 2/4 connected RED remains unchanged. The old native qualification
PRE remains BLOCKED and is not edited. A new prospective diagnostic-order
amendment may allow honest measurement refusal while testing native integrity;
that is a different decision rule from the original 39-outcome qualification.
Neither rule may relabel failed original accuracy assertions as passed.

The numerical choice can be used in the forthcoming draft once root accepts
this review. Activating it still requires the accepted connected terminal POST,
final integrity implementation/oracle review, synchronized frozen limits, exact
source/import closure, independent new PRE, archive and explicit single-run
release. No old artifact or budget is reopened by this review.

## Arithmetic and measured reference

The separately reviewed terminal L-R registration took 1462.253261 CPU seconds;
its V2 measurement call took 1.437055. Registration completed, while measurement
honestly refused at the existing-cover rule before a V2 cover refit. A failed
accuracy/eligibility outcome does not invalidate the observed completed runtime,
but it does limit what downstream work was represented by that runtime.

Independent decimal arithmetic gives:

```text
reference = 1462.253261 + 1.437055 = 1463.690316 s
1.5 * reference + 300 = 2495.535474 s
ceil(2495.535474 / 60) * 60 = 2520 s
```

Thus the CPU allowance is 1056.309684 seconds above the measured reference,
approximately 1.721676 times that reference. The 50% multiplier and additional
300 seconds are explicit engineering choices, not fitted native cost
coefficients or an empirical success quantile. Rounding contributes another
24.464526 seconds; no hidden per-case runtime is being excluded.

| Limit | Proposed value | Interpretation |
|---|---:|---|
| Worker CPU | 2520 s | Whole worker: original pure/mock checks, fixture, exactly one native inference, original oracle evaluation and bounded supplemental integrity checks |
| Work wall | 2815 s | CPU ceiling plus 295 s of wall allowance; not a guarantee about native/OS blocking |
| Parent wall | 2820 s | Five-second difference from work deadline, preserving the supervisor's cleanup reservation |
| Sampled owned-group RSS | 4,294,967,296 B | Unchanged sampled refusal ceiling, not a proved peak-memory bound |
| Stdout | 1,048,576 B | Unchanged aggregate output cap |
| Four scientific files | 144,707,584 B total | 8 MiB CAD + 1 MiB scan + 129 MiB result + 4096 B start |

The time ordering is valid: `2520 <= 2815 <= 2820-5`. Work wall is not CPU plus
a guaranteed separate 295-second service period; both ceilings act on the same
attempt. Native C++ calls, JSON work and filesystem waiting may consume time in
different proportions from the array reference. The supervisor's sampling and
OS cleanup checks are not hard real-time guarantees.

## Why this is a proportionate ceiling, not a forecast

The earlier
[source-derived resource analysis](2026-09-11_two_plane_depth_native_resource_analysis.md),
SHA256 `e7d3cfaff1d61657d47c4a29c30a097664f4b8369dac426240a3ce9224163682`,
separates CAD point count from finite-triangle work. The native fixture has the
same 650 scan rows but 32,768 sampled CAD points instead of 284 CAD vertices.
The ratio 115.3803 is **not** a multiplier on all registration iterations.

With the existing 48-candidate/30+30-iteration caps, the small fixture still has
at most 3024 whole-part finite queries and 1,965,600 scan query rows. Sample-
sensitive additions include at most 1,572,864 reverse KD query rows and up to
49 CAD-point canonicalizations/bindings. CAD sampling's six retained arrays
alone are 3 MiB; that is not the complete native memory footprint. No timing
coefficient is justified for those counts without a native observation.

More important unknowns remain: native STEP/feature work, actual native CAD
tessellation, different face/plane proposals, different finite correspondence
and reverse-coverage behavior, context eligibility, report size and repeated
byte publication checks. Native tessellation changes the triangle index and can
change optimizer/search work even when scan and sample counts remain fixed.
Post-mesh admission limits do not pre-bound the native mesher's internal CPU or
allocation. The reference's absence of a fresh V2 cover fit also means the
native run may do downstream work not present in that particular measurement
timing. None of this is certified to fit inside the declared headroom.

Nevertheless, one measured completed L-R registration on the current mechanism
is a materially better engineering anchor than zero-start historical L cases,
dividing a four-case cap by four, or multiplying by the CAD-point ratio. The
2520-second ceiling grants explicit headroom while remaining substantially
smaller than the full four-case packet's 6000-second CPU cap. Its purpose is a
bounded diagnostic opportunity, with a valid terminal negative outcome if it is
insufficient—not guaranteed reproduction of a success.

The completed connected packet's observed peak was 1,253,244,928 B, versus the
unchanged 4 GiB ceiling (about 3.43 times that observation). Native allocations
are not constrained by that observed ratio; it merely supplies no current
evidence for raising the RSS limit. Preserve the original native meshing,
32,768-sample and scientific work policies. No held optimizer prototype,
parallel heavy worker, reduced sample set or increased engine query quota is
part of this allocation.

## Conditions for final integrity/PRE release

The numerical review is conditional on the following concrete accounting:

1. Exactly one actual native workflow invocation. Original accuracy and new
   integrity predicates consume the same fully retained workflow, not separate
   favorable recomputations. All fixture/oracle/helper work runs inside the
   charged worker interval; no uncharged native preparation is added outside it.
2. Every original qualification outcome/oracle remains identifiable. The new
   integrity decision separately records complete/honestly refused/malformed/
   unavailable cases and preserves the original accuracy failure. A generic
   nonzero pytest exit or missing result is not automatically an acceptable
   diagnostic. Final review must verify the exact new denominator and the
   allowed failure classification, not assume the current 39 is still the whole
   future suite.
3. Supplemental checks have explicit input-byte, traversal/work and output
   bounds before expensive reconstruction; later malformed fields invalidate
   the whole relevant integrity result. Their ledger is computed only after
   durable `CASE_RESULT.json` retention and lives only in the existing receipt/
   `OUTPUT.txt`, capped at 32 KiB within the unchanged 1 MiB stdout limit. It is
   not added to the original `CASE_RESULT.json` or its 1 MiB recorder metadata;
   the original four scientific files remain unchanged. The extra 300 seconds
   is not a license for unbounded replay, hidden geometry or added inference.
4. Preserve the four-file custody contract and regular/no-follow/nonblocking
   inventory reads. Full workflow retention precedes scientific interpretation;
   metadata or transport failure is not repaired by trimming a result prefix.
   Sampled-group quiescence, source hashes and final inventory remain mandatory.
5. Freeze the same CPU/work/parent literals in fixture, supervisor, protocol,
   receipts and new PRE. Re-derive the complete inherited source/import closure
   after the amendments, including test initializer/package-boundary checks,
   diagnostic helper/tests and the exact accepted connected RED POST. The old
   prospective 104 count is not automatically the amended denominator.
6. Budget failure is terminal, retaining unknown/uncompleted stages. No in-place
   increase, repeat, fallback optimizer or physical run is authorized. The new
   native timing, if complete, is still one invented geometry observation and
   cannot by itself qualify a uniform Gold11 budget or its all-22-case total.

During this source review the native harness still declared time constants
`None`, and the native test was being amended in parallel. Its observed hash
changed from the earlier frozen-draft identity to
`fbd2b8e55dedf5d95416ccb0b85f89844166190f3abdd829e37618af23411ee1`.
That is not a complete source/denominator release. The reviewed transport-only
harness identity remained
`f9b145c95f1586f8b72726570813975d0808bb269ee39acd341e305ecb6ea6d9`;
the old native protocol remained
`013dfc4c2f80455eac2925e66da859b6b820a57605aed9ac87dcd8b9a3da5e48`.
The final independent PRE must read the completed amendments rather than
inherit this partial source observation as qualification.

## Review boundary

The allocation note and prior resource analysis were read completely; the
native harness and relevant current fixture/protocol budget/retention seams
were inspected without importing them. Arithmetic used the standard-library
decimal implementation only. This allocation review made no new artifact,
archive, raw/control or native access and ran no tests or geometry. It relies on
the already completed, separately recorded terminal failure analysis for the
released L-R timing and 2/4 disposition. The preservation archive described in
the allocation note was not independently opened here.

The old resource note's requirement for connected GREEN is not retroactively
rewritten: it belonged to native **qualification** planning. This new review
allows consideration of a prospectively declared **integrity diagnostic**
after the connected RED is classified, while preserving its failures. That
narrow decision-order change is not an engine improvement or promotion claim.
