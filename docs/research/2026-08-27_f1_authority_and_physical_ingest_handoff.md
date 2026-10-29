# F1 authority and physical-ingest handoff — 2026-08-27

## Current outcome

- The frozen three-hour production contract soak passed: 10,800 seconds,
  10,675 isolated child jobs, and 1,067 isolated abstention jobs.
- Neither model lane has new generalization evidence. Both remain GPI-C.
- The CAD-assisted research loop is waiting for two independent real-human
  decisions over the frozen 63-event ontology pack.
- Fourteen new V/X scan files across seven delivered pairs are declared in the
  evaluator-only intake, but none of the PLY bytes is present locally yet.
- The 14 SharePoint links were recovered from the source Gmail messages. Direct
  anonymous download returns HTTP 403; an authenticated SharePoint connection
  is required.
- The product now exposes `/api/bend-inspection/readiness`, renders a
  research-only UI warning fail-closed, prefixes text exports, adds a prominent
  PDF disclaimer, and bakes `RESEARCH ONLY - NOT PRODUCTION QC` into every
  detached overlay PNG while promotion remains false or unavailable.
- Production mode cannot be opened by flipping one status boolean. It also
  requires physical-generalization success plus valid candidate-lock,
  qualification-report, and dual-review-truth SHA-256 bindings.

## Independent CAD ontology review

The review pack is bound to SHA-256
`c52c7399cd7b3263e8f2dd3b9191c0dc356b7175a4ea9e5cb1cf5069a89e2c3f`.

Start the local review application:

```bash
cd /Users/idantal/ShermanQC/FullProject
.venv/bin/python3.11 scripts/serve_f1_cad_line_ontology_review_v1.py --host 127.0.0.1 --port 8765 --no-open
```

Open the printed localhost URL. Reviewer 1 selects `Manufacturing authority`,
uses a real name, reviews all 63 native-resolution events, and exports
`manufacturing_authority_decision.json`. Reviewer 2 works independently,
selects `Countersigning authority`, and exports
`countersigning_authority_decision.json`. The first complete reviews must not
be discussed or copied between reviewers.

The Sol Max file in the pack is an advisory draft only. It must not be used as
a signature or silently prefill either independent review.

Compile the two files fail-closed:

```bash
.venv/bin/python3.11 scripts/finalize_f1_cad_line_ontology_consensus_v1.py \
  --manifest output/research/cad-line-ontology-authority-pack-v1/MANIFEST.json \
  --manufacturing /path/to/manufacturing_authority_decision.json \
  --countersigning /path/to/countersigning_authority_decision.json \
  --output output/research/cad-line-ontology-authority-pack-v1/CONSENSUS_OR_ADJUDICATION.json
```

The compiler emits an actionable mapping only when every event agrees and no
event remains `UNRESOLVED`. Disagreement or matching unresolved decisions emit
an adjudication queue with `actionable_mapping: null` and no model-execution
authority.

## New physical scans

Evaluator-only manifest:

`data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/evaluator_only_manifest.json`

Declared capacity is 14 scans / 7 V-X pairs. It receives zero qualification
capacity credit until the bytes are downloaded under evaluator custody and a
separate custodian supplies acquisition, session, morphology, independent
manufacturing-instance, calibration, and quality-review metadata. Sender bend
counts and V/X roles are provenance only; they are not bend-line truth and may
not reach inference.

Even after successful download, this delivery alone cannot satisfy the final
30-scan / 15-morphology / two-independent-instances-per-family qualification
gate.

After the fourteen original filenames are downloaded into one private folder,
validate and materialize them atomically into opaque N001-N014 custody:

```bash
.venv/bin/python3.11 scripts/materialize_f1_pending_delivery_intake_v1.py \
  --manifest data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/evaluator_only_manifest.json \
  --download-root /private/path/to/downloaded-ply \
  --target-root data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs \
  --lineage-registry data/bend_corpus/custody/physical_lineage_v2.json \
  --workspace-root /Users/idantal/ShermanQC/FullProject \
  --public-receipt data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/public_materialization_receipt.json \
  --evaluator-receipt data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/evaluator_materialization_receipt.json
```

The command validates the entire denominator before copying, rejects content
already consumed in development, rejects duplicate scans and conflicting
opaque targets, and is idempotent for byte-identical existing targets.

## Next empirical action after the gates open

1. CAD-assisted: preregister exactly one signed-CIBEC
   primitive-to-manufacturing-event contraction experiment, retain the frozen
   neutral-half-turn parent, and visually audit every changed case at native
   resolution.
2. Scan-only: open authentic organized range frames (not fused PLY surrogates)
   or record the acquisition-capability stop.
3. Promotion: compare frozen candidates once on an untouched, dual-reviewed,
   30-scan / 15-morphology physical cohort.

No selector tuning, CAD-line relocation, C005 contraction, truth opening, or
promotion claim is authorized before the relevant gate is complete.

## Zero-shot runtime integrity correction

The production worker, domain application entry point, CAD-extraction API, and
legacy compatibility endpoint now operate source-only by default. Part IDs,
drawing-derived expected bend counts, part-specific feature policies, filename
tokens, and expected/completed counts cannot silently alter inference. The
operator-assisted research path remains available only through an explicit
`use_part_truth_hints=True` opt-in.

This is an evaluation-validity improvement, not an accuracy improvement. No
historical F1 score is promoted by this change, and both lanes remain GPI-C
until a frozen candidate is evaluated on admissible physical evidence.

## Local download reconciliation

The Downloads folder contains eight physical scans whose bytes were not under
the corpus tree. Hash and documentary reconciliation showed that all eight had
already been consumed in V12 or the P01-P07/V46-V51 development sequence. They
are now registered by immutable SHA-256 in the physical lineage registry, so
their external storage location cannot allow accidental reuse as untouched
qualification evidence. None receives new qualification credit.

The fourteen scans declared in the Aug-17/20 N001-N014 manifest are not those
files: the frozen source filenames do not match, and the all-or-nothing intake
correctly failed before copying anything.

Mailbox reconciliation also discovered one genuinely new Aug-25 V/X delivery,
reserved as N015-N016 in a separate evaluator-only manifest. Its two PLY links,
native CAD attachment, drawing PDF, sender count, V/X roles, and part identity
remain closed to inference. The link correspondence is hash-audited, but the
PLY bytes still require an authenticated SharePoint download. This increases
declared pending capacity to 16 scans / 8 pairs, while qualification capacity
credit remains zero until custody, metadata, and independent line truth pass.
