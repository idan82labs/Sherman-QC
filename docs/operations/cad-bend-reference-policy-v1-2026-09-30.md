# CAD bend technical-reference policy V1

2026-09-30. Source-only policy draft; no reference labels or measured score are
established by this document. Scope is the frozen existing 20 unique STEP hashes,
consumed CAD development only. No scans, email, new datasets, sealed labels,
manufacturing approval or unseen-family inference are authorized.

## Frozen evidence campaign

The released inventory `FILES.jsonl` binding is SHA-256
`713c14c33c045744c1d2f9ee79a92d713fdc09d1c3d14457e3df82659a8a72ff`:
20 unique STEP byte hashes and 23 STEP aliases. Non-STEP entries are metadata
only and never decoded. Each representative source is revalidated without
following symlinks, copied once into its immutable case folder, and read only
by the neutral collector. The complete anonymous census and every failure are
retained; prior extraction campaigns are not rerun.

Limits: 60 seconds and 4 GiB sampled process-tree RSS per case; 1,230 seconds
for the campaign; source at most 5,000,000 bytes; 40,000,000 aggregate bytes per
case including source, logs and artifacts, with a separate 1,000,000 combined
log-byte limit. Final accounting is required even after a fast child exit.
Native work is serial under the existing
shared admission lock. Unknown process-group cleanup stops subsequent cases.

First review authored fixture imports and run only synthetic/native-authored
qualification under the shared lock. Then archive exact source/runtime pins,
freeze inventory, limits and collector, obtain independent non-author GREEN PRE
bound to that freeze, and execute exactly once. Exclusive claims and terminal
receipts prevent reruns. Preserve negative/failing runs; successor authority and
fresh qualification/PRE are required before another physical attempt. Two
independent non-collector reference reviewers label the neutral evidence before
any new case-level predictions are released. Exact native identity is the score
target; unverified/assumed units never support millimetre-accuracy claims.

## Reference target and ontology

The target is a finite geometric sheet-bend feature supported by the native CAD
material boundary, not a manufacturing operation count. A feature records its
finite connected forming region, adjoining material, paired material skins when
established, axis/carrier, radii, signed turn/angular domain and trim support.
Face splitting alone does not create another bend; separate finite regions are
not merged merely because their carriers are coaxial. Multiple forming operations
cannot be inferred from a final geometric region without authoritative history.

Every native face and connected material region needs an explicit disposition:
forming feature, planar material, hole/cut wall, fillet/edge round, boss/collar,
rolled/high-turn feature, other geometry, or unresolved. Geometrically proved
negatives and unsupported surfaces are different categories. Analytic cylinders
are neither sufficient evidence of bending nor the complete census: spline skins,
stepped transitions, collars and rolled regions remain in the audit ledger.

## Blind evidence and independent review

Collect all native faces, analytic surface descriptions, exact trim bounds,
adjacency/edge identities, orientation, units and validation failures without
importing the candidate extractor or publishing candidate face groups, counts,
markers, success flags or old results. Stable anonymous case identifiers prevent
filename/count hints. Evidence collection is not an automatic reference oracle.
Pairing cylinders with the same candidate assumptions is corroboration, not
independent truth. Neutral evidence must support a complete region/material review.

Two non-author reviewers independently assign dispositions from neutral evidence
before predictions are opened. Their disagreement ledger and adjudication are
preserved. Missing evidence, insufficient expertise or ambiguous operation identity
stays unresolved; an AI technical review is not qualified human manufacturing
review. Freeze labels, completeness receipts, collector/scorer source hashes and
tolerances before prediction comparison. One terminal collection is retained;
failures do not authorize retries or a smaller cohort.

Reviewers have read released historical corpus aggregates in the current-status
document; this exposure must be disclosed. Case-level predictions remain withheld
until reference freeze. The procedure is independent case-level review, not strict
blindness to all historical information.

## Numerical and completeness gates

Tolerances must follow documented CAD/kernel precision and verified unit metadata,
not candidate errors or a desired score. Unknown units prohibit physical-distance
claims; normalized comparisons must be explicitly dimensionless and independently
frozen. Axis sign equivalence, finite support, angular wrap, split faces, orientation
and radius semantics must be specified. Sampling/meshes may aid review but cannot
certify exact trim coverage or absence of a feature. Invalid solids and missing
adjacency receive failure dispositions, not guessed material.

All 20 hashes remain in the census. Report reference-resolved part coverage, face
and forming-region coverage, disagreement and unresolved regions/cases. A face
count alone does not prove feature completeness. No whole-corpus bend percentage
is valid until every case has a complete reference census, including proved
zero-bend cases. If unresolved regions remain, a resolved-subset score is only a
diagnostic, prominently named as such. Complete-case scoped scores and partial-
reference interval bounds may be reported, with their exact populations and
assumptions. An unbounded unknown bend count makes whole-corpus recall bounds
vacuous; a nontrivial bound requires an independently justified finite missing-
feature budget. Never cherry-pick analytic/easy cases as the headline result.

## Frozen scoring and permitted claims

Use independently reviewed one-to-one matching, not count equality alone. An
exact-native-face membership contract may score feature detection if reference
region membership is independently established; it must not be called location,
radius, trim or angular accuracy. Geometric accuracy needs the separate frozen
geometric checks above. Duplicates are false positives; a merge cannot cover several
reference bends. Candidate refusals retain all known reference bends as false
negatives. Predictions near unresolved regions require an unresolved-match ledger,
not automatic false positives or favorable removal. Freeze empty-case conventions
and reject nonfinite/invalid geometry instead of silently defaulting values.

Publish micro precision/recall/F1, part-macro scores, exact-count agreement,
refusals/zero-output rates, all per-case dispositions and reference coverage.
Separate finite-corpus point estimates from part/family uncertainty. A 90% claim
requires both precision and recall under the predeclared acceptance rule, complete
reference coverage and the unchanged denominator; F1 alone is insufficient.
Agreement with an old extractor, preview availability, flattening isometry and
cylinder-pair counts do not establish bend extraction accuracy. This evaluation
does not establish manufacturing blanks, physical metrology or generalization.
