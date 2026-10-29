# Frozen R0 control, AI review and comparable-score boundary

Date: 2026-09-09. User approved a new frozen development baseline, requested a
production-workflow simulation beating the old score, and explicitly asked AI
to review the unsigned convention material. **No comparable Gold11 improvement
has yet been demonstrated.** The completed simulation below is not a calibrated
production-rig simulation or a historical-incumbent comparison.

## Delivered

1. A new R0 **source/configuration snapshot**, separate from historical F1
   0.4580152672: [control policy](../../backend/cad_guided_development_control.py),
   [freeze/verify tool](../../scripts/cad_guided_development_bundle.py), and
   [captured bundle](baselines/cad_datum_registered_r0_20260909.json).
   Bundle identity: `0460d13536cf28bd6bf2977c39d3e732d9c35c219070d9205ae5c314bdf12c46`.
   Eleven files are captured as actual bytes: nine dependency-audited backend
   modules, explicit control policy and bundle tool. Freeze is exclusive-create,
   never overwrite. The historical executable is not reconstructed or relabeled.
2. Explicit registration, measurement, finite-support, scorer and inherited
   feature policies copied unchanged from the earlier fixed file rehearsal.
   Input caps remain 200,000 scan rows/5,000 datum rows; no preprocessing is
   silently enabled. The control has no measured physical score.
3. [Unconnected row-lineage preparation](../../backend/cad_guided_row_lineage.py):
   an integer-rank selection plan retaining every caller-declared datum and
   preserving raw-row order, datum order, reversible retained-row lookup and
   disjoint inspection roles. Prospective limits are one million declared raw
   rows to at most 200,000 engine rows. It reads no coordinates/files and does
   not select datums, bind source bytes or guarantee spatial/feature coverage.
   The actual file loader and product routing are unchanged.
4. [Complete AI technical review](2026-09-09_gold_ai_technical_review.md) of the
   63-proposal rendered pack: 56 provisionally plausible, seven unresolved, zero
   positively rejected. This is advisory proposal review, not observed counts,
   replacement Gold or two human manufacturing decisions. Original templates
   and references remain untouched.

## Verification actually performed

| Check | Result | Parent wall seconds | Worker peak RSS bytes |
| --- | ---: | ---: | ---: |
| Bundle/source/import regressions, final | 36 passed | .145225208 | 41,664,512 |
| Row-lineage contracts, including 1M-to-200K declaration | 85 passed | .226629750 | 80,920,576 |
| Native worker source checks | 25 passed | .137560459 | 38,993,920 |
| Independent circular-carrier math and actual matcher | 16 passed | .286071875 | 82,542,592 |
| Frozen-source actual-file engine rehearsal | 16/16 expectations | 3.944923458 | 384,581,632 |

Thus **162 distinct focused tests pass**, plus the 16-case native rehearsal.
The bundle suite was rerun after adding a provenance-mutation assertion; its
earlier pass is not counted as another independent test set. No tests failed in
these executions. Pre-run independent source review found unsafe executable
configuration extraction, direct aggregate budget bypass and mutable provenance
label capture; all three were fixed before the first bundle freeze/run.

The native command is a fixed selector, not a physical runner:

```bash
.venv/bin/python -I -S scripts/check_cad_native_provenance.py --frozen-control-only
```

The worker requires the exact bundle identity before importing the captured
backend bytes and policy objects. It verifies those policies equal the unchanged
16-case rehearsal parameters. It then runs real STEP/BREP and ASCII/LE/BE PLY
loading, registration, native feature extraction, scan measurement and finite
scoring on files invented in its owned temporary directory. No existing raw
part is opened. Current rehearsal/helper/harness source is separately bound
before/after, not falsely described as archived in the R0 runtime bundle.

Full [machine-readable receipt](baselines/cad_datum_registered_r0_20260909_rehearsal.json):
12 positive engine outputs and four required input refusals; all 16 scored.
TP12/FP0/FN4, P1/R.75/**synthetic support F1 .8571428571**, exact support count
12/16. Known +1 mm radius and +5 degree sweep deviations remain correct after
independently solved oblique pose and partial datum coverage. Maximum radius
error 4.52971e-14 mm; recovered position 4.73741e-14 mm; sampled endpoint
9.22237e-14 mm; sweep-deviation 5.94014e-12 degrees. These are constructed
round-trip numerical errors, not real scanner uncertainty.

Worker wall 3.205465875 s, CPU 2.858559 s; parent/group peak 384,581,632 B.
Exit zero, no budget stop; 17 named source/receipt digests unchanged before/after.
Same fixed 300 s wall /240 s CPU /2 GiB sampled process-group RSS /1 MiB output.
One Python worker and BLAS/OMP thread declarations do not establish one native
OCCT/TBB thread: meshing remains explicitly parallel, native thread count unknown.

The R0 snapshot stores exact interpreter identity and NumPy/SciPy/OCP/proxy/VTK
version/METADATA/WHEEL/RECORD digests. **These are dependency metadata bindings,
not verification of every installed binary, OS library, OCCT resource or dynamic
loader decision.** Source/config replay is proven within this environment;
full runtime attestation and metrology qualification are not established.

## Review findings that affect the engine comparison

- Thirteen of 63 cyan carriers have `ABSTAIN_RELOCATION`: the renderer labels
  retained fallback lines as observed intersections. Do not derive observed
  support or corrected truth from their color.
- This 11-case pack is not the historical Gold11/71-reference cohort. It includes
  family 47959001, absent from historical Gold11; 63 CAD proposals cannot replace
  the 71 reference denominator.
- Center axis, half-thickness midpoint and flange intersection are different
  carriers. Sixteen mathematical/matcher tests confirm the consequence: with
  R5/t1/90-degree geometry, a perfect center axis is 5.5 mm from the midpoint,
  beyond the unchanged 4 mm matching gate. A representation mismatch is not an
  engine localization error, and proximity within tolerance does not establish
  semantic equivalence either.

## Active scope amendment and next gate

The AI-review request authorizes an **AI-reviewed provisional development lane**;
do not keep demanding two human signatures before advisory development work.
Human-qualified manufacturing/release requirements remain separate and unchanged.
No AI reviewer is a fabricated manufacturing authority or human countersigner.

R0 is a new control with no historical score. Its synthetic F1 .857 is **not**
an improvement over old physical-development F1 .458. A comparable historical
diagnostic must preserve the exact Gold11 cases, all 71 original references,
coordinate/line convention and matcher; retain every unsupported/failed job;
freeze the complete input/registration/scoring protocol before physical access.
The resulting metric would measure agreement with retained development labels,
not independently qualified physical accuracy. No new rule may be selected by
which alternative improves consumed Gold scores.

The next required engineering work is a bounded raw-input adapter carrying
source-byte/row lineage and actual stable-datum selection, followed by a reviewed
coordinate/carrier bridge. The declaration-only row planner is not that adapter.
Native feature scope remains a narrow rectangular inner-cylinder/two-flange
contract, not a complete manufacturing-event detector. The original 13-case
run summary reported loaded scan counts above the current cap in 10/11 matched
cases, but current raw headers were not opened, so rejection is not newly proven.

Stop before changing Gold, inheriting old F1 into R0, raising a real-run cap,
dropping unsupported cases, retrying terminal controls, or opening reserved
email/held-outs. Further raw geometry/reference preparation needs its own
bounded development scope and independent execution gate. The prior metadata
review and this rendered-pack AI review do not silently authorize raw-input or
truth-coordinate opening. No product deployment or production/floor gate passed.

Current authorized work remains source/synthetic engineering and the completed
explicit rendered-pack AI review. The paused long-running goal is unchanged.

Final static audit: nine changed/new Python sources parse; ten local receipt
links resolve; scoped whitespace checks pass; the bundle still verifies with
zero captured-source drift. The 22 main and 41 subreview event IDs match the
manifest's exact 63-event set with no omission/duplicate. Independent connected
review found no important numerical, historical-score or authority inconsistency.
