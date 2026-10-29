# Five-part CAD upload → bend extraction → unfolding successor protocol

Status: **PROPOSAL ONLY — NO PHYSICAL EXECUTION AUTHORIZED BY THIS DOCUMENT**  
Date: 2026-09-29

## Question and evidence boundary

Determine whether the maintained *fresh upload* workflow can take a previously
approved STEP file, create the catalog part and bend candidates, then return a
source-bound complete flat preview with the same bend identities. The current
public V14 demo does **not** answer this: it serves retained derived fixtures.
The controlled backend runs only under the explicit `v3-development` setting,
and its folded-flange companion is not qualified. No scans, Gold, held-outs,
new email parts, or wider corpus may enter this successor.

This is a five-part consumed-development support test, not a claim that any CAD
format or arbitrary sheet-metal topology will unfold, nor a manufacturing blank.
The source accepts several upload extensions, while the controlled V4 unfolding
engine accepts only `.stp`/`.step`, a maximum five-million-byte source, 1–32
candidates, and bounded planar/cylindrical selected-skin geometry. Format and
geometry refusals count in the denominator. Units remain assumed millimetres,
not independently calibrated.

## Fixed cohort and source-declared bindings

Use exactly these five part identities from the released vendor-preview reports,
in this order. The original terminal V1/V2/V3 and V4 five-part records stay
untouched. `docs/operations/vendor-unfold-preview-2026-09-29.md` reports V3
support **1/5**; `docs/operations/vendor-unfold-production-readiness-2026-09-29.md`
reports the audited V4 fixture result **2/5**. Those are static/pinned-CAD
previews, not fresh-upload outcomes.

The existing, source-only `SOURCES` declaration in
`scripts/build_vendor_unfold_demo.py` lines 14–20 supplies the following names
and hashes. That script sets `cad_root = ROOT.parent /
"source-assets/02_CAD_QC_3D"` (line 30); the paths below are relative to that
root. These are **source-declared expected identities**, not verification of
current raw bytes or a new run freeze.

| Part identity | Released V4 fixture disposition | Native bend markers on supported fixture | STEP filename under declared root | Source-declared SHA-256 |
| --- | --- | ---: | --- | --- |
| 37361005 | AVAILABLE | 4 | `37361005_01.stp` | `81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e` |
| 49001000 | UNPROVEN_NONPLANAR_BSPLINE_SURFACE | — | `49001000_A-PRELIM.stp` | `d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5` |
| 42288005 | NO_CONNECTED_SKIN_COVERS_ALL_CAD_BENDS | — | `42288005_01.stp` | `7afb000cdb1f18eddb4d270c5b8c2c5854d822d67adfd3cfebc5e7ad9e126837` |
| 38172000 | UNAVAILABLE (finite-parent support guard) | — | `38172000_01.stp` | `4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f` |
| 10839000 | AVAILABLE | 12 | `10839000_01.stp` | `ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37` |

An author of the new exclusive freeze must copy exactly these five source-code
bindings, pin the generator source bytes, and cite the released vendor reports
for prior use of this five-part cohort. Independent PRE must verify each
declared raw file's **actual** SHA-256 and the held-out exclusion before any
native invocation; this source-only proposal neither reads nor attests to
those bytes. Do not discover candidates by walking a mutable catalog, source
directory or manifest. A missing/mismatched binding blocks the entire five-case
PRE; do not silently run four or substitute a similarly named file. This is
evidence binding, not permission to add physical parts.

## Candidate and control frozen before any native run

Freeze exact source and dependency bytes, runtime/configuration, five input
paths/hashes, candidate extraction version, planned API route, scorer, visual
review rubric, limits, and exclusive output paths in a new versioned protocol.
Create a complete source archive with independently checked member hashes.
The control is the retained V14 fixture result, clearly labelled *different
entry path*; do not call 2/5 a paired fresh-upload baseline. If a true paired
control is implemented, it must use a separately frozen earlier engine against
the exact same fresh-upload candidate rows within its own bounded, terminal
campaign. No Gold or fixture bend list may feed extraction or candidate choice.

The runner must exercise the real local upload and catalog-extraction boundary
in an isolated temporary catalog/upload root, then invoke the controlled
`v3-development` unfolding path. Never operate on or mutate the live catalog,
existing uploads, saved drafts or public Site. Candidate rows must originate
from that upload's extraction, not the published fixture; record their exact
IDs, endpoints, order, source hash and generation. Verify a bijection between
those rows and the native flat markers before declaring a part AVAILABLE. A
different candidate count or ID is a substantive result, not an invitation to
patch the response using the fixture. Upload response, candidate extraction,
folded display asset, controlled preview and UI validator must all bind to the
same CAD bytes and generation. Concurrent re-upload/stale-generation simulation
must refuse mixed output.

Before a physical campaign, source-only tests must prove the isolated route,
monotonic deadline, exclusive records, hash mismatch refusal, source-change
race, process-group monitoring/cleanup, complete failure receipts and five-case
reservation. Synthetic native fixtures must cover holes, asymmetric trims,
finite-width tabs, independent pose, face/edge ordering, ambiguous skin,
missing/extra candidate, cap exhaustion and unsupported surfaces. A changed
mechanism needs its own independent mathematical review; a favorable fixture
alone is insufficient.

## Fixed execution envelope and failure handling

- Exactly five serial source calls, one heavy/native geometry job at a time,
  fixed order above; `OMP_NUM_THREADS=1`.
- **60 seconds wall and 4 GiB process-tree RSS per entire source case**, not
  60 seconds for preprocessing plus another 60 for unfolding. The parent must
  reserve and enforce the remaining time before every child. Retain the
  existing stricter V3 limits (5 MB CAD, 40 MB result, 100 kB request, 1 MB
  logs, face/triangle/edge/candidate work caps); never relax a started run.
- Freeze the campaign deadline before launch as five 60-second reservations
  plus a separately declared bounded preparation/cleanup allowance. No
  in-flight extension, parallel fallback or case-specific budget.
- Each selected case gets an exclusive terminal row and raw upload, candidate,
  worker and admission receipts, including unsupported, timeout, busy,
  process-monitor failure, hash mismatch and not-executed outcomes. No omitted
  denominator, synthetic substitution, partial flat geometry or success cache
  after failed cleanup. Parent-observed exit and worker quiescence are recorded
  separately from sampled RSS; a sample is not a continuous high-water proof.
- The one-shot runner/paths must be NEW. Do not reopen or overwrite any V1–V4
  terminal attempt. A successor after failure requires a documented engineering
  reason, new immutable protocol and a new independent PRE.

No command is supplied here: source-declared input identities now exist, but no
new approved runner, independently verified raw bindings or independent PRE
exists for this successor. A physical invocation copied from an older report
would violate this protocol.

## Independent PRE, POST and decision rule

Independent non-author PRE must check the exact five released bindings,
held-out exclusion, source/archive closure, isolated upload root, candidate
provenance, deadline and process controls, all-case receipt reservation and
visual rubric. It must refuse any mutable input expansion or attempt reuse.
Only a GREEN PRE with zero blocking findings permits the *single* physical
execution. A second reviewer must not derive inputs from the run's outputs.

Independent POST reconciles every frozen hash against the actual bytes and
every terminal row, outer exit, process cleanup, candidate input digest,
native source/face/marker ownership and frontend admission. Publish the full
five-case disposition table, timings, resource observations and reasons, not
only successful screenshots. Report separately:

1. Fresh-upload bend-extraction count and candidate identity integrity.
2. Complete unfolding support rate, with all five in the denominator.
3. Supported-part geometry/visual integrity and each explicit refusal.
4. Any independently referenced bend precision/recall, only if a separate
   approved bend truth and unchanged matcher exist. CAD-generated candidates
   and V14 fixture markers are **not** independent truth; no F1 otherwise.

The minimum positive result is a verified fresh-upload end-to-end preview for
at least one approved part, with no fabricated marker and no unsupported part
presented as available. Reproducing 2/5 on fresh uploads would close a major
integration gap but not establish CAD-agnostic generalization. Any claimed
increase over 2/5 needs genuinely paired entry-path comparison and independent
POST; unchanged or negative results are retained. No result from this protocol
authorizes default activation, public publishing or cutting release.

## Visual integrity review

For **every** supported part, save deterministic folded and flat views from
top/bottom and four oblique camera directions, plus a screen recording or
captured interaction selecting every bend. Review the native-face/source
mapping and overlays against the same CAD: complete material with real holes
left empty; no filled cutouts, broken trims, duplicate/missing triangles,
overlaps, inverted winding or collapsed flange; every bend line finite and
inside actual developed material; every flange highlight a proper subset with
the same ID/color in folded and flat views. Independently inspect flat area,
bounds and bend lengths against native CAD geometry where an analytic oracle
can be established. Flag unproved dimensions rather than eyeballing them as
correct. Verify order, optional `מהלך` grouping, rotation/pitch/zoom, touch
selection and readable labels at tablet and desktop sizes. For all unavailable
parts capture the folded fallback and precise refusal with flat controls
disabled. Visual attractiveness is not a geometry proof.

The bounded folded companion, true source-bound flange-depth proof, arbitrary
CAD-format support, hostile-upload sandbox, neutral-axis compensation and shop
dimensional validation remain separate gates. Until they pass, describe the
output as an operator **selected-skin visual preview** only.
