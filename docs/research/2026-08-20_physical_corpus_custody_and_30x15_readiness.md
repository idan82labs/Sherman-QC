# Physical corpus custody and 30-scan/15-family readiness

Date: 2026-08-20

This audit did not newly open U01-U04 geometry, evaluator-only identity, expected count, CAD, drawing, or label. A lineage cross-check found that U01-U04 geometry was already consumed by the 2026-08-15 V51/V52 real-scan affine-resampling study. “Sealed” in the V73 protocol meant not reopened by V73; it did not restore untouched confirmatory status.

## Inventory

The incoming tree contains 40 physical point-cloud files. Two files in the 2026-07-28 delivery duplicate the 38172000_01 X/V files already represented in the 2026-07-29 intake, leaving 38 non-duplicate incoming scan files.

| Intake | Non-duplicate scans | Known part IDs | Custody status | Permitted use |
|---|---:|---:|---|---|
| 2026-07-16 count-backed PLY | 12 | 12 | consumed by prior capacity/development work; sender counts are not authoritative line truth | development annotation and error analysis |
| 2026-07-29 good/defective pairs | 12 | 6 | heavily consumed by architecture development and visual review | development robustness and paired-scan analysis |
| 2026-08-04 scan-quality parts | 6 | 3 with scans | consumed by visual/domain and scan-quality review | development annotation and scan-quality analysis |
| 2026-08-11 good/defective pairs | 4 | 2 | evaluator-role metadata exists; scans have entered the research context | development paired-scan analysis |
| 2026-08-13 opaque blind tranche | 4 | development-consumed by the Aug-15 V51/V52 real-scan resampling study; not reopened by V73 | evaluator identity/labels remain closed, but geometry is no longer untouched | development robustness only |

The 2026-07-06 50-part delivery contains CAD, drawing candidates, and unadjudicated meshes, not physical scanner point clouds. It can support family-ontology construction only if the research protocol explicitly opens that capability; it cannot substitute for unseen physical scans.

## Readiness ruling

The repository does **not** currently contain an untouched, independently annotated 30-scan/15-family physical confirmation corpus.

- Total physical quantity is sufficient for a development set.
- Known family diversity is at least 23 part IDs before the opaque tranche.
- Confirmatory independence is insufficient because the known scans have already influenced model, architecture, threshold, or visual decisions.
- U01-U04 provide four evaluator-opaque scans, but their geometry already influenced the V51/V52 robustness/error record and therefore cannot serve as untouched confirmation.
- Existing sender counts, CAD/drawings, and good/defective roles do not provide independent manual bend-line ground truth.

## Custody decision

1. Designate every known, previously opened scan as `DEVELOPMENT_CONSUMED`.
2. Classify U01-U04 as `DEVELOPMENT_CONSUMED_EVALUATOR_OPAQUE`; keep their evaluator identity/labels closed and do not reuse them as confirmatory evidence.
3. Use the development-consumed corpus to build the annotation protocol, train annotators, validate overlays, and generate an error taxonomy.
4. Acquire at least 30 new physical scans across at least 15 externally defined morphology families. Prefer two independently manufactured/scanned instances per family.
5. Before any model sees them, assign opaque IDs, hash source files, strip filenames/roles from inference, and freeze family membership in an evaluator-only manifest.
6. Obtain two independent native-resolution manual bend-line selections per scan, visibility/ambiguity tags, and adjudication. Keep adjudicated truth outside the inference process.
7. Run one frozen retained baseline on the new corpus, score with the V2 evaluation contract, and visually adjudicate every prediction marker and every missed gold bend.

## Immediate actionable use of current scans

The consumed corpus is not wasted. It should now answer engineering questions without spending confirmatory evidence:

- How often do independent annotators agree on what is scan-observable as a bend?
- Which false positives are boundaries, holes, ribs, smooth rolls, thickness edges, occlusion seams, or cross-sheet jumps?
- Which false negatives arise before candidate birth versus during ownership/reconciliation/deletion?
- Which scan-quality failures make a bend genuinely unobservable?
- Does the retained baseline preserve predictions across good/defective scans of the same physical family?

Those answers should determine the next architecture. They must not be reported as untouched generalization performance.

## 2026-08-22 intake update

Seven additional good/defective families were located in Gmail deliveries dated 2026-08-17 through 2026-08-20, representing fourteen new PLY scans. Their authenticated SharePoint downloads are pending because no connected signed-in browser session is currently available.

An evaluator-only intake mapping now assigns opaque inference IDs N001-N014. Part identities, V/X roles, source messages, and sender-stated counts are excluded from the inference contract. Sender counts remain non-authoritative and independent manual bend-line annotation is still required.

If these fourteen files are downloaded, hash-sealed, and kept unseen by architecture selection, they provide 14/30 scans and 7/15 families toward the research confirmation gate. At least sixteen more scans across at least eight additional untouched families are still required.
