# Manufacturing flat-pattern candidate — 2026-09-29

User scope: pursue the established-unfolder option toward manufacturing readiness.
This is a separate candidate, not a relabelling or retry of visual V3. Existing
visual V1/V2/V3 source archives, dispositions, PDF and public Site remain intact.
No scans, Gold, email, held-outs or additional physical CADs are accessed here.

## Latest checkpoint

The controlled offline file workflow V3 completed once: all four frozen synthetic
oracle checks passed in 7.756 seconds. Three supported inputs produced compensated
analytic flat-pattern drafts; the unsupported sharp return correctly returned no
geometry. The independently posed hole part now passes without loosening DXF
seam tolerances. Both independent geometry and math POST audits are GREEN;
all six actual DXF texts pass third-party import with zero errors/fixes.
Manufacturing/cutting release remains disabled.

This is a separate backend candidate. The live vendor Site and its illustrative
skin-development preview/PDF have **not** been replaced or relabelled. Existing
operator tags/groups have not been migrated through a live catalog transaction;
the exact source-bound bend-ID bridge has engineering evidence only.

The external qualification inputs still missing are (a) applicable shop bend
compensation evidence for an identified material/condition, thickness, process
and tooling, and (b) measured known-blank/coupon dimensions with the shop's
declared error limits. These cannot be inferred from folded CAD, synthetic K
values, source units or a confirmation checkbox. The empty profile template
below is intentionally invalid until actual values are supplied.

## Frozen initial supported model

One valid, connected, constant-thickness native solid; two completely accounted
skins with planar flanges and cylindrical bends; tree topology, straight bend
seams, sweep <=175 degrees. Exact line/circle trims only; cylindrical pcurves
must be linear. Formed/freeform surfaces, curved bend trims, cycles requiring
rips, unpaired skins, omitted material, overlaps or warnings refuse the whole
result. No partial cutting geometry is returned.

The engine uses pinned FreeCAD 1.1.3 / commit
`145529fe741292ff0b3977a01195bf0247425794`, SheetMetal 0.8.23 / commit
`3141e205ea340d68e69ff0e257a44e27d9bcb591`, and isolated NetworkX 3.4.2.
Official ARM64 DMG SHA-256:
`f5c0ece7cd7c932466d6effadc0fc6e179b0538a9d9a6a77a6769eae3af2667c`.
Downloaded binary checksum matched; official bundle signature verified before
use. Runtime and dependencies live in ignored `.tools/`; the application Python
environment and installed macOS apps were not upgraded.

The adapter calls raw `unfold`, not the GUI/sketch or approximate cleanup path.
Upstream cleanup snaps/drops small geometry and approximates splines. Its table
interpolation is also defective on interior intervals, so this candidate supports
only an explicit single ANSI K-factor, passed as literal `ansi`.

Primary source references:

- [FreeCAD 1.1.3 release](https://github.com/FreeCAD/FreeCAD/releases/tag/1.1.3).
- [Pinned SheetMetal source](https://github.com/shaise/FreeCAD_SheetMetal/blob/3141e205ea340d68e69ff0e257a44e27d9bcb591/SheetMetalNewUnfolder.py).

## Mathematics and independent checks

ANSI K = neutral-axis distance from the inner skin / sheet thickness.
For bend sweep theta in radians, BA = theta * (Ri + K*t). Upstream BendInfo
reports signed **included** angle, not sweep; DOWN uses the outer-skin radius.
Material/process/tooling, thickness and radius/sweep applicability are explicit
profile inputs with no manufacturing defaults.

The independent native area oracle is the sum of planar skin areas plus each
cylindrical selected-skin area multiplied by (Ri+Kt)/Rs. Area alone is not a
coverage proof: counterpart faces, source patches, cutouts, overlap and every
native face must also be accounted for. Native source-face indices are retained,
but they are not yet certified as the vendor UI's existing bend identities.

Native campaign: independently constructed L/U parts, inner/outer roots,
30/90/135-degree sweeps, different K values, circular holes, independent pose,
invalid parameters, sharp formed appendages, cycles and swallowed upstream
failures. No physical input is read by this campaign.

Before execution, freeze every named source/helper, NetworkX source file and
critical native executable/module into a new exclusive manifest. Independent
technical review must greenlight that exact candidate. One serial process group,
120-second wall cap, 4-GiB RSS cap, 10-MB combined logs and 10-MB result cap, OMP=1,
isolated user preferences. All failures and authored case dispositions are
retained; terminal records cannot be reopened or overwritten. A justified source
change requires a new manifest and a new campaign record.

## Release gates — no manufacturing approval yet

1. Same-byte STEP source hash and actual native unit-entity verification; unit
   names or a target-system-unit setting alone are not source-scale evidence.
2. Complete geometry and independent synthetic/adversarial audits.
3. Exact source-bound bend-ID bridge, preserving vendor order and groups.
4. Analytic cut contours and separate bend/annotation layers, explicit mm and
   1:1 export; round-trip closure, cutout, bounds and dimension checks.
5. A shop-supplied compensation profile applicable to material, thickness,
   process/tooling and bend domain, with independently verified supporting data.
6. Dimensional validation against a known blank/coupon and declared shop error
   limits. Synthetic success does not establish physical calibration.

Both `manufacturing_ready` and cutting release remain false. A profile hash,
`shop_candidate` label or operator checkbox cannot grant qualification. The
public preview/PDF stays labelled illustrative until these gates are closed.

## Current evidence

Pinned runtime health check: FreeCAD reports 1.1.3 and exact source commit;
NetworkX reports 3.4.2; synthetic 1x2x3 box volume=6; explicit ANSI K=.4 returns
.4. No physical CAD was used.

Pure compensation/input checks: 29 passed. These include 2,000 independent
neutral-arc scalar oracles and invalid profile/domain/unit refusals. The native
unit proof is being strengthened from names to actual entity scale/context.
Supervisor failure-transport checks: 5 passed. Native campaign pending exact
source pre-run review and freeze; no manufacturing score or qualification claimed.

The initial source-only unit suite exposed a binding identity issue: OCCT
`Model.Number(Value(i))` returned zero, including for a valid newly generated
millimetre STEP. No physical input was involved. The final validator strongly
retains native model handles and requires exact live handle identity; it never
falls back to names/descriptive equality. The combined final pre-native receipt
is 56 passed in 0.44 seconds, including 17 entity/context adversaries and ten
supervisor failure/strict-record checks. The earlier failing receipt was an
engineering test result, not silently treated as success.

Synthetic V1 freeze (no execution yet):
`artifacts/vendor-manufacturing-unfold/synthetic-v1-frozen.json`, SHA-256
`0651955a4c39d05f2c2aa1b735f19a178d54b89bd91dc1d9211366154d62a785`.
819 exact files cover the candidate/helper sources, critical native modules,
NetworkX and imported Draft Python closure. Minimal child environment excludes
inherited loader/Python-home overrides. Every terminal case/run remains draft
only; malformed/failed/empty receipts cannot be marked COMPLETE.

Pre-run review found a final wall-clock race after leader exit. Before any
native launch, it was closed and regression-tested. V1 freeze/archive remain
preserved without execution; V1b freeze is
`0b31094b3be27236d1075149cfb6c96e10744900f56a4c7877979c355ac30b87`.
V1b source archive SHA-256:
`85e7f7a9c938ba4f48d68446d9f8296f730313ef3f72b0939a9852495c9713f5`.
Adapter `c8919ecbe6e501884b2749976f7bb787074092a2aae9d5b395c0617f6190596a`;
suite `3662c5633a79ada4232cd23f6cc337f003123925c6c01f3b7f4376f0ede143b8`;
supervisor `953692263256b21c135eb6c4c793724ca28283ff916d7cb25f80ab29923e8055`.

Independent non-author math/native-source review and supervisor-safety review
both GREEN for this **synthetic draft-only** campaign, not production. Reviewers
verified all 819 pins and unchanged caps. Independent checks included 56
adversarial circle cases, 17 unit checks, 11 supervisor checks and five syntax
checks. Reviewers ran zero FreeCAD jobs and read no physical CAD.

Frozen command:
`OMP_NUM_THREADS=1 .venv/bin/python scripts/run_vendor_manufacturing_synthetic.py --frozen-manifest artifacts/vendor-manufacturing-unfold/synthetic-v1b-frozen.json --record artifacts/vendor-manufacturing-unfold/synthetic-v1b-run.json`.
It reserves an exclusive new record; do not reuse this command after a terminal
record exists. A successor needs an engineering reason, new frozen source and
independent review under a new record name. Native campaign launch is underway.

V1b completed once and is terminal FAILED: outer exit 1, 0.77 seconds,
195,520 KiB process-group peak RSS. All 49 authored cases retained: 36 passed,
13 failed. Every hole/posed-hole positive refused `ONE_VALID_SOLID_REQUIRED`;
plain 30/90/135-degree bends, both skin roots, three K values and the two-bend
U part passed their independent area/allowance oracles. Negative/refusal cases
also passed. No cap changed and no physical input was used.

The input contract incorrectly demanded `ShapeType == Solid` even when a native
Boolean returned a compound containing exactly one valid solid. V2 engineering
reason: prove and normalize that wrapper while rejecting all extra faces, wires,
vertices, duplicate/ambiguous ownership or multiple solids. Preserve face tables
and root/source identities. Existing hole cases and their expected dimensions
remain unchanged. V1b is not retried or overwritten; V2 needs new frozen source
and independent audit before execution.

The input template is `docs/operations/vendor-compensation-profile-template.json`.
Its null numeric values deliberately fail validation; they are not defaults.
Material must include grade/condition, and tooling/process applicability must
identify where the compensation data was established. A populated template is
still a candidate, not approval. Existing shop records may provide allowance or
bend deduction instead of K; conversion requires their dimensional/angle
convention and independent review, not the defective upstream table path.

Independent V1b POST integrity audit PASS: receipt SHA-256
`03b6f19cd2b8f61248f8e1098e3b79f0101f96cdfa6f2ad4b6f21e01f2a45c03`,
matching frozen sources/caps/runtime. Its 36 passed checks comprise **13**
successful geometry oracles and 23 refusal/integrity checks, not 36 unfolded
parts. All 13 failed positive hole cases remain in the ledger.

V2 candidate admits only a bare solid or a compound with exactly one direct
solid child. Native face, edge and vertex tables must preserve count, uniqueness,
order, identity and orientation. Extra/duplicate topology or reordered identity
refuses. Tests add stray face/wire/vertex, duplicate solid, reordered/reoriented
native tables, positive one-solid wrapper and mirrored hole geometry. Existing
angle, K, hole and pose cases and all analytic/coverage gates are unchanged.

V2 freeze (awaiting exact independent gate; no execution yet):
`98e0a291f3c66ef50cad00438de4d6746964c3622ab88846e2a30ab2c6274715`.
Adapter `581c1a2d8eff8f9cc09b9912ec4526d977e321fd02b5e8b4eb9da82c9f0faa26`;
suite `9065009c8b3842a7138d2e668718b230547836c9952a1f12e9cc75675dc7ccee`.
819 pins; only adapter/suite differ from V1b. Supervisor and all experiment caps
are unchanged. No physical CAD run or public manufacturing publication is implied.

## Native geometry gate closed; file workflow gate follows

V2 is terminal FAILED: 56/57 checks passed, 1.12 seconds, peak 201,104 KiB.
The duplicate-solid negative fixture used the identical native object twice;
the kernel collapsed it before admission. Its failed receipt remains unchanged:
`4d8c71694f9c8ba4c8ed6f654ef483c415e70fe64d1b32853dc77f3c5209c9d7`.
V3 changes only that fixture: use an independent native copy, assert two direct
solids and distinct identity, then require the same refusal. No geometry gate,
expected positive result, compensation value or budget was loosened.

V3 independent PRE and POST are GREEN for this synthetic model. All 57 checks
passed (28 positive geometry oracles, 29 refusal/integrity checks), 1.14 seconds,
peak 202,288 KiB. Receipt `synthetic-v3-run.json` SHA-256:
`cb098bc5a039733de9b5871891230fb70431ca6782e23bc5edc8e9a441c7b44f`.
Freeze `13cc24ada4b74f3ae0fbf92f2645dc6668f6e499ea326455bbf0d82c3d4bae4b`;
source archive `fec58751b172da2d26ebd4f99c6eeace076175f69432545732c213fea88d609f`.
The adapter remains `581c1a2d8eff8f9cc09b9912ec4526d977e321fd02b5e8b4eb9da82c9f0faa26`.
This is not 57 manufacturing-qualified parts or physical accuracy evidence.

The new controlled local engine seam is `backend/vendor_manufacturing_job.py`.
It does not bootstrap the legacy server or load the catalog. Public Python calls:
`freeze_file_job(raw, sha256, profile, ...)`, then
`run_file_job(raw, sha256, profile, frozen, new_record_path, ...)` after independent
audit. Private `--worker` / `--native` modes must not be launched without the outer
supervisor. One lock and process group cover the OCP audit and FreeCAD grandchild;
120-second / 4-GiB / 10-MB log/result caps, OMP=1, isolated preferences and pycache,
source checks before/after execution and exclusive terminal records apply.

Same-byte native STEP SI-millimetre entity/context proof precedes import. The
profile has no defaults or inferred shop authority. Root is explicit native face
or deterministic CAD-only largest planar face; it is not a calibrated shop datum
or a bend-order decision. Native indices and analytic contour rows are retained.
Each mid-skin folded bend marker requires a unique complete source-bound bijection
to the existing vendor candidate IDs; no nearest-available or partial mapping.
Supplied candidate IDs/markers are frozen; automatic source-only STEP IDs are an
alternative, not a claim that existing saved operator tags were migrated.

Cut-only DXF and combined bend-guide DXF are distinct. No bend-guide entities are
allowed in the cut-only file. Both remain draft/not released. Native full-circle
topology/domain certificates are required; a small endpoint gap never becomes a
circle. Clockwise/major arcs preserve their loci. Writer/parser checks include
strict mm, layers, entity ownership, complete endpoint loops and analytic area;
the job also compares native cut area and exact analytic rows. Mathematical
roundoff bounds remain separate from supplier tolerances.

Independent DXF source audit at `af1ac2b5dd7ad87087d6926306fbf721af9d1ee291f8c1d04d462fe300ed894c`:
70 pure tests passed. Isolated ezdxf 1.4.4 (app environment unchanged) read five
independently constructed export fixtures with zero audit errors/fixes; units,
entity types/layers and near-full/major signed arc loci survived. This is limited
importer evidence, not qualification against a shop's CAM or physical blank.
Serializer audit closed radius-scaled circle roundoff and signed included-angle
consistency; 13 pure tests plus independent signed/mirrored probes passed.

Four-case file campaign is separate from V3: L90 with circular hole, two-bend U,
independently posed L90 hole, and sharp-return native-unfold refusal. Geometry is
freshly generated from pinned synthetic builders, never loaded from corpus.
Generation is source-frozen and capped first. Evaluation then freezes exact
generated bytes, profiles, independent area/count oracles and the successful
generation receipt before launch. All four cases are retained in their declared
order. An unrelated unit/import/bridge refusal cannot pass the sharp-return oracle;
that case must refuse at `NATIVE_UNFOLD_REFUSED:`. Job receipt transport checks
reject missing proof, partial geometry on unavailable results and malformed DXF.

Generation V1 PRE freeze (not executed at this checkpoint):
`file-generation-v1-frozen.json`, SHA-256
`e139393b1963305292c05d92ef60e850225f73da4f1b8b75413317adc6261eec`, 1,737 pins.
Archive `file-generation-v1-source.tar.gz`, SHA-256
`1e754ab65febba4282f7863ac05bc0bc49b67e5a7adf027bb4112fc2686a6d2b`.
Job `d35a6abef17d85d6b4a056b59c9d871610bade7124768e2960ded927eaaac492`;
runner `6a40c45a9749e67705000176635e2f47d53a278a053926b89abc8e138a645325`;
generator `69cf84ecc7846b54014c742c37db2da7cb5d906266b09e707fd7af76da5ee0d9`.
Focused job/harness checks: 26 passed. Independent exact PRE is pending.
Use module commands (`.venv/bin/python -m scripts.run_vendor_manufacturing_file_synthetic`),
not a bare script command without repo Python path.

No shop compensation profile or known-blank dimensional truth has been supplied.
The user has been asked for applicable material/thickness/tool/process data.
The public Site and prior illustrative PDF are unchanged; no factory/cutting
release, physical calibration or manufacturing approval is claimed.

## File V1c failure and justified V2 successor

Generation V1 completed once: 0.442 seconds, observed process-group RSS peak
71,344 KiB. All four fresh STEP files were retained (64,085 bytes total), with
explicit synthetic profiles and independent area/count/refusal oracles.
Receipt `ac916a2555f38d1159f2fee0d44fdb7c28f43f1ad1dbfa4cd7b790e12c357fb1`.
Clean generation archive is `88839f1c01942e750d87907eb88639ca3bac413a0afd55417e56315fc2b5e060`;
the original metadata-bearing archive is preserved but not the reviewed payload.

File-evaluation V1 and V1b freezes were superseded **before execution** while
closing contradictory nested evidence flags; they remain preserved and unused.
V1c freeze `2766e1f10f651688b85e1ebf803708ef14cc8fe0dd48fb99cc0b2d8980eac7ef`
and source archive `621068061fe62aaaac4af818197f12804663efb4359dba344403cdb8947e4cbc`
received two exact independent PRE gates. The one execution is terminal FAILED:
all four native grandchildren crashed with SIGSEGV 11 before unfolding. No
partial draft or score was emitted. Aggregate receipt SHA-256:
`29cf255ca007d40adcd26cf01d0bd9da0dd2b4c2e9e8fa7e639db47cec09363c`.
Elapsed 4.186 seconds; maximum observed per-case process-group RSS 379,264 KiB.
Independent POST confirmed integrity and all four failures, not a positive result.

The file worker imported Part before the lazy adapter had initialized FreeCAD.
The successful native suite and generator import FreeCAD first. Pinned Part
startup accesses initialized application/base state; this is a source-supported
cold-start cause hypothesis to be tested, not a favorable interpretation of the
failed run. V2 initializes FreeCAD, verifies exact version/commit, then loads
Part; faulthandler is enabled before native imports. Fatal operational errors stay
FAILED/exit 1 and preserve the child signal. They never become a successful
unsupported-geometry refusal. Aggregate failure reasons are nonempty and retain
the job cause instead of an empty assertion message.

V2 changes only the job and main harness, not compensation, geometry, export,
case definitions, numerical oracles or budgets. The stricter full-source generation
provenance requires a separately frozen fresh Generation V2; V1 receipt/files and
all four failed V1c cases remain intact. Fresh STEP headers may produce new byte
hashes; this is not a claim of a new physical cohort or unseen-family evidence.

Generation V2 independent PRE verified all 1,737 pins and clean archive payloads.
Freeze `94d28d5b46ef50d567c9b627f7aed165852366a23c5e4c194352f79888eda50e`;
archive `6be4ec51d4fbe789fba3f09fcfa73e403452f89d68e766d43eb2676b52929aa8`.
It completed once in 0.447 seconds. Receipt
`a0ecb7817e3bd90cb1a7162ade709e1b4be4b801ef469dba92eead293a85751a`.
Job source `99bc3ba4366903962d21a529cbc5e795daa535bb2c041d5a05e8ea39145d5455`;
harness `c6a614092bfc77187ed85b2db4e383a5a7bcf1468007122fc25f4cb80c44ad1c`.
35 focused pure tests passed, including cold-import order and signal transport.

File V2 exact-byte/profile/oracle freeze (awaiting independent PRE at this entry):
`9f26e5a6d1a9471ae268e40caf0bc89a84f335856bc1bf37f5d23e9afdf4cf5b`.
Source/input archive `43002cb82d205709598583dfbdb65ab2e5023bbe947518bc87fa14bef00ec5db`.
No V1c record is retried or overwritten, and no cap is increased.

V2 exact independent PRE gates both GREEN. The one end-to-end execution is
terminal FAILED overall: **two** successful geometry drafts plus one expected
native-geometry refusal, and one unavailable positive case. The cold native
startup crash is no longer present. L90 hole produced one bend and area
2160.22122533308 mm²; U produced two bends and area 1476.4601535159013 mm².
Both passed complete source units, compensation, coverage, bend-ID bridge and
cut-only/bend-guide DXF transport checks. Sharp return refused at the required
native geometry stage (`ESTIMATED_THICKNESS_MISMATCH`), not a generic input error.
The independently posed L refused `OPEN_OR_AMBIGUOUS_CUT_ENDPOINT`; it has no
partial draft/export. This is 3/4 oracle checks, **not three valid blanks**.

Aggregate `file-evaluation-v2-run.json` SHA-256:
`4c45463856a4b85758d13f459c029a376ed6ce69c2c552a73eb3f80beb5fea71`.
Elapsed 7.230 seconds, maximum observed per-case process-group RSS 668,928 KiB.
Independent POST confirmed exact provenance and all dispositions/caps/false
release flags. The successful drafts remain engineering results only.

Proposed V3 engineering reason: transport finite LINE cut endpoints from actual
native vertices after proving unique oriented trim-endpoint correspondence and
shared native vertex identity at every wire seam. One unchanged 1e-7-mm physical
representation-error budget covers displacement, length and planar projection.
This is source-topology transport, not coordinate-based snapping or hole repair.
Keep the DXF writer and ARC/CIRCLE handling unchanged; genuine gaps or ambiguous
native ownership refuse. Reuse the exact four V2 generated bytes/profiles/oracles
under a new source freeze and independent PRE. V2 remains terminal FAILED.

## File V3 exact independent PRE gates

V3 is frozen at `file-evaluation-v3-frozen.json`, SHA-256
`964d6ce1791d01f28e8f49931a6d52d43862de0391fb8106ac491a6a43903b0b`.
Source archive `file-evaluation-v3-source.tar.gz`, SHA-256
`506fc71c8030f32791f3950c44ffa237da646178193d047d6a9fbcf7acefe1ac`.
Serializer SHA-256
`f4c40ba306e47f55fba8306e9f98bd7d69692687c953e0b831cb7bd80ffa41a5`
is the sole runtime delta from terminal V2. The exact four V2 STEP bytes,
profiles, independent oracles, generation receipt and budgets remain unchanged.

Independent math reviewer `/root/vendor_unfold_math_audit` explicitly returned
**PRE GREEN**, verifying all 1,737 pins / 1,735 per-job pins and all archive/input
bindings. Its checks passed 18 tests, 36 subtests and 13 independently constructed
native-vertex adversaries. Independent geometry reviewer
`/root/vendor_unfold_geometry` separately returned **PRE GREEN** for the exact
same manifest and serializer: 123 focused pure tests plus 36 subtests passed;
five deliberate partial-arc gaps still refused. ARC/CIRCLE extraction and the
DXF writer are unchanged. Both reviewers performed no native jobs or physical
CAD reads. These gates authorize only the four declared synthetic engineering
jobs, not manufacturing release.

The first launch request was denied by an approval check before process creation
because it still regarded PRE as pending. No V3 run record exists at this entry,
and no experiment started. After receiving both explicit exact PRE GREEN reports,
the manifest/archive/serializer hashes and absence of a V3 run record were checked
again. A retry of the same command under the same permission review is justified
by the completed gates above; no alternate executor or bypass is used.

## File V3 terminal engineering result

The same bounded command was admitted and executed exactly once. Terminal
aggregate `file-evaluation-v3-run.json` is **COMPLETE_SYNTHETIC**, SHA-256
`0ac4c3002d8de9b633efb7c47d7efd58a4a73cf6459f5eded95dfd8c05978cc2`.
All four selected cases and individual records are retained. Elapsed wall time
7.756447 seconds; outer exit 0. No budget, input, profile or oracle changed.

| Frozen case | Disposition | Finite nominal bends | Flat area (mm²) |
| --- | --- | ---: | ---: |
| L90_hole | COMPLETE_DRAFT | 1 | 2160.221225333080 |
| U90_two_bends | COMPLETE_DRAFT | 2 | 1476.460153515901 |
| L90_hole_independent_pose | COMPLETE_DRAFT | 1 | 2160.221225333048 |
| sharp_return_refusal | UNAVAILABLE, required native-geometry refusal | — | — |

The posed/unposed L area difference is approximately 3.23e-11 mm², within the
unchanged 1e-6-mm² campaign oracle. This is a synthetic numerical result, not
a supplier dimensional tolerance or physical accuracy claim. The refusal is an
oracle success, **not** a fourth usable blank. Runtime output keeps
`manufacturing_ready=false` and `cut_release_allowed=false` throughout.

Executed command, retained for reproduction evidence (do **not** run again with
these terminal record names):

```sh
.venv/bin/python -m scripts.run_vendor_manufacturing_file_synthetic \
  --mode evaluate \
  --manifest artifacts/vendor-manufacturing-unfold/file-evaluation-v3-frozen.json \
  --directory artifacts/vendor-manufacturing-unfold/file-generated-v2 \
  --generation-record artifacts/vendor-manufacturing-unfold/file-generation-v2-run.json \
  --record artifacts/vendor-manufacturing-unfold/file-evaluation-v3-run.json
```

Exact source/runtime/input pins and per-case freezes are embedded in the manifest
and terminal receipts. Any justified successor needs a separately frozen protocol,
new record names and independent PRE. Next safe action is independent POST on
these named derived synthetic records and the actual DXF texts; no new native
experiment or physical CAD access is authorized by this result.

Independent geometry POST `/root/vendor_unfold_geometry`: **GREEN** for the exact
aggregate/source/archive and all four records. Maximum posed/unposed cut/bend
point or circle-carrier difference 3.40109e-11 mm; area difference 3.22871e-11 mm².
Both L blanks have numerical bounds 54.08407044966673 x 40 mm. Maximum source-bound
bend bridge error 1.58044e-12 mm; maximum LINE transport certificate error
3.99716e-11 mm. All native coverage and finite bend/profile provenance checks hold.
Maximum observed per-job process-tree RSS was 667,040 KiB; all wall/RSS/log/result
caps were respected. These are algorithmic checks, not physical uncertainty.

Independent math reviewer imported the **actual six retained DXF texts** with
isolated ezdxf 1.4.4: all have zero audit errors and zero fixes, R2000/mm metadata,
expected entities/layers, exact native LINE/CIRCLE loci and closed contours.
Cut-only exports have no bend entities; guide exports retain 1/2/1 bends with
source-bound ID bijections. Independent contour area oracles pass the unchanged
1e-6-mm² limit. No review launches a native job, changes runtime result flags,
or qualifies shop dimensions.

Final exact independent math POST `/root/vendor_unfold_math_audit`: **GREEN** for
the same aggregate/manifest/archive, all 1,737 source pins and the four ordered
hash-bound job records. Accepted 4/4 oracles (three drafts plus one intended
native refusal), profile/unit/coverage/bridge/compensation and all frozen caps.
Independent importer entity counts are 9/10 (L cut/guide), 12/14 (U), and 9/10
(posed L). Parsed LINE/CIRCLE loci equal retained native rows; independent
contour area, bounds, hole radius, guide width/containment/hole clearance and
neutral-strip midpoint oracles pass the unchanged 1e-6 numerical criteria.
This separate review does not rewrite the original run's
`independent_importer_evidence=false` field, nor establish shop/CAM compatibility
or physical qualification. No source or terminal result was changed after freeze.

## Qualification handoff and stop boundary

No engineering calculation can recover a shop's neutral-axis compensation from
the folded CAD alone. The supported backend therefore requires an explicit
applicable profile; it does not use synthetic K=.4 as a shop default. If an
existing shop-approved table is available, supply the exact record and its
dimensional/angle convention. If it is not available, the smallest next decision
is the first material/condition + thickness + process/tooling domain to qualify,
followed by actual measured coupon/known-blank records and declared shop limits.
Calibration records and independent validation records must remain distinct.

With those inputs supplied, the successor must freeze their exact hashes,
supported domain, units, candidate and error criteria before evaluating a named
part. Validate cut outline, holes, dimensions, bend direction/allowance, vendor
ID/order/group preservation and an independent CAD/CAM import. Preserve every
refusal and uncertainty. Only then consider a separately reviewed integration
into the maintained vendor flow; an operator confirmation alone is not release.

Until then: no hardware operation, physical calibration claim, automatically
enabled cutting export, catalog migration or public manufacturing-ready label.
The backend candidate and all negative/positive engineering receipts remain
available for development. This stop boundary does not mark manufacturing
readiness or the overall Sherman QC roadmap complete.
