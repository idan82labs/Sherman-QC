# V9 kernel/policy final bounded closure audit

Date: 2026-09-13. Independent reviewer: Astra/Codex
`v8_flange_policy_math_audit`. Author checkpoint: 04:43 UTC.

## Disposition

**GREEN for this bounded source-closure seam, with the nonblocking precision
notes below.** No remaining P1/P2 blocker was identified in the previously
reviewed kernel selection/geometry/provenance mechanisms or the angular-policy
declaration corrections at the exact hashes below. This is not acceptance of
the entire V9 candidate, physical PRE, a freeze or run authorization, a test
execution claim, or production/metrology approval. The material-identity
consumer chain, completion lane, runner and execution waiter have separate
review owners.

This audit read source/tests and released narrative records and checked hashes.
It ran no imports, tests, simulations or geometry and accessed no physical
inputs, Gold, attempt artifacts or private caches. Only this new report was
written; prior reports and all engine/test files remain untouched.

## Bound source snapshots

The closure response SHA256 matched the parent's checkpoint:
`31def50d1d4795c22f8c2c79b8899eafe445c388ecfd7041b1c4697da0985de4`
for `docs/research/2026-09-13_fable51_v9_review_response.md`.

| File | Independently checked SHA256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `a894b96057ca886981a6c8ac4f968519d4c6246af05254eddd03ed41e9864ad1` |
| `backend/cad_guided_instances.py` | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
| `tests/test_cad_guided_measurement_v2.py` | `e0b31029aada927ad6458506060e0b9788ee4f7b3da9dd31b3c65e549f4ec8f9` |
| `tests/test_cad_guided_instances_angular_gap_v9.py` | `0900de7c3657799b71ca1849892ae9ebc09c858e9a229befeaf7d87c0840717d` |
| `tests/test_cad_guided_v9_connected_chain_v1.py` | `14e405921fcf4869870d0274cfdd66705172a58289aecae96d0d1fecc21269cf` |
| `scripts/run_gold11_mesh_development_v9.py` (policy declaration excerpt) | `ad61e12b0a06c0cafb7e515687019e42c6e137e26a419753aaeef2360f5b4241` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md` (kernel/policy sections) | `16f20a99766838c6ca210be86582c40089aafeae70540dc6d874848ab5b4c1f2` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md` (kernel/policy sections) | `03fa29937f2614971a515028bb9c500cb1736f91f962b275fb9cd81082ae0be9` |
| `docs/research/2026-09-13_fable51_v9_diagnosis.md` (relevant B/E/F sections) | `fddc8e3644561ad7c5f8766d166a0f99edfcffb006613544e300967e00a301e7` |

Line references below are to these snapshots. Test results reported by the
author at these hashes are not independently executed results of this audit.

## Prior finding closure

### 1. Layer partition/discard provenance: closed for the declared reconstruction contract

`backend/cad_guided_measurement_v2.py:246` enumerates every gap above the
declared floor and distinguishes qualifying from unqualified large gaps. A
qualifying split still means enough rows in the complete prefix and suffix,
not in every possible physical population; the newly named
`qualifying_groups_detected` and `qualifying_groups_scope` at `:361` say so.

The bounded record now names its truncation and publishes reported/omitted
qualifying-split counts, unqualified-large-gap counts/omissions, the actual
split records, sort normal and center (`:359`, `:387`). For one compatible
selected pair it also publishes the split threshold and exact kept rule
(`:405`, `:412`). Given the same retained local crop rows, the partition is
reconstructed by `(row - sort_center_mm) dot sort_normal < split_threshold_mm`.
The prospective V9 floor is 0.3 mm, well separated from float roundoff at the
declared coordinate scale; this is not an interval-certified claim over every
arbitrarily tiny policy value the general class might accept.

This is reconstruction provenance plus an explicitly bounded diagnostic
summary, **not** a standalone embedded raw-row ledger or a physical skin
certificate. The connected source/raw-row binding remains the consumer
review's responsibility. Multiple qualifying splits retain all rows with
unresolved identity (`:397`); an incompatible pair also retains all (`:409`).
No upper group is chosen merely because the eventual fit improves.

The new source test at `tests/test_cad_guided_measurement_v2.py:260` asserts
exact equality between reconstructed and selected masks. The sparse-population
test at `:280` retains the unqualified singleton gap and documents that its
small RMS contribution need not prevent the main pair selection. It does not
pretend that two qualifying groups prove exactly two material skins.

### 2. Diagnostic retention: closed for completed-prefix numerical/fit paths and late invalidation

The kernel now publishes ordered completed phases and a failure phase
(`backend/cad_guided_measurement_v2.py:617`). Numerical exceptions in the
flange diagnostic/selection loop attach the completed bounded prefix (`:633`);
plane fit (`:638`), axis/sweep (`:645`) and arc fit/tangency (`:709`) attach it
for ordinary `_Unavailable` and supported numerical exceptions. An insufficient
arc count attaches its phase explicitly (`:707`). `_attach` preserves an
existing diagnostic attachment and does not convert attempted evidence into a
measurement (`:721`). Public caught error rows keep the attachment (`:893`).

Successful fits later invalidated by shared support now preserve
`attempted_diagnostics` and an explicit non-surviving-measurement scope before
clearing measurements (`:911`). Source tests cover an injected role-1 SVD
failure through both private and public entries (`tests/test_cad_guided_measurement_v2.py:320`),
success/ordinary-failure phase lists (`:342`), and shared-support invalidation
(`:353`). These close the earlier concrete loss paths without granting a
partial measurement. The wording must retain the completed-prefix boundary
described in the nonblocking notes below.

### 3. Group and footprint-proxy naming: closed

The compatibility check now publishes
`projected_bounding_interval_overlap`, actual projected intervals and their
lower-group TLS basis (`backend/cad_guided_measurement_v2.py:332`). Its
`overlap_scope` explicitly excludes area/material-footprint coverage and
discloses basis ambiguity when the lower in-plane spectrum is nearly equal.
The policy field is `min_projected_interval_overlap` (`:67`), and the refusal
name is `GROUP_PROJECTED_BOUNDING_INTERVALS_DO_NOT_OVERLAP` (`:344`).

The docstring at `:286` no longer promises that arbitrary curved contaminants
fail. It identifies the RMS/rank qualification and explicitly allows shallow
curvature within the declared bound. The holed-footprint/island source test
(`tests/test_cad_guided_measurement_v2.py:300`) asserts overlap 1 while retaining
the proxy-only scope. This is the requested claim correction, not a new
physical overlap mechanism, and is adequate for the generic development path.

### 4. Earlier geometry fixes: preserved

The radial/normal signs remain consistent (`backend/cad_guided_measurement_v2.py:592`):
the crop sort normal is oriented toward positive nominal radial (`:381`),
while `_plane` is called with negative radial (`:639`). With signed sweep
phi, the outgoing flange rays remain `-sign(phi)*k0` and
`sign(phi)*(axis cross end_radial)` (`:598`). The enabled arc rule is signed
distance greater than exclusion against both fitted axis-pointing normals;
disabled mode restores the V1 absolute-band rule (`:659`).

For the exact common-axis circular model, distances against those planes are
`R-rho*cos(theta)` and `R-rho*cos(abs(phi)-theta)`. The nominal outer-arc bound
therefore uses `(R-e)/(R+t)`, not the opposite sign. The kernel retains the
domain check and `certificate=False` with idealized assumptions (`:684`); it
does not turn nominal-radius arithmetic into observed exclusion evidence.
True flange-ray extents are calculated from supplied rays (`:227`), separately
from principal extents. The measured radius remains a free fit to projected
scan rows (`:756`, `:479`), not a copied nominal value.

The plane gates retain population-RMS scaling `s2/sqrt(N) > 10*delta`, the
thickness ratio gate and the actual max-residual check (`:419`). The
flat-witness gate retains minimum strip half-width greater than delta plus
numerical guard (`:479`). The earlier math report's nominal-model limitation
and finite-sampling caveat still apply; passing these sample gates is not
calibrated uncertainty or stable-datum metrology.

### 5. Angular and arc-policy declarations: closed

The runner now distinguishes exact synthetic angular columns, nominal chord
arithmetic and unmeasured whole-scan row spacing (`scripts/run_gold11_mesh_development_v9.py:179`).
It explicitly declares fixture-only evidence, no phase guarantee, fixed-fit/
fixed-row support-local monotonicity only, and a non-unique, uncalibrated
development choice. Protocol `:50` and delta design `:42` now list 5 -> 30;
their combined-effect interpretation includes the support policy. No scorer
threshold change is authorized or implied by this closure.

The exact connected clean fixture still has five nominal angular columns at
22.5-degree spacing, three remaining after ideal r=0.8/e=0.04 exclusion,
and a finite strip half-width of 0.030448187 mm. That is only about 0.000448187
mm above delta=0.03. A half-step phase leaves two exact columns; raising the
gap cap cannot recover the third. The revised protocol `:40` and diagnosis
`:289` retain that limitation. Diagnosis `:125` keeps the narrow-window cause
open for physical crops; `:148` discloses combinatorial pair counts; `:241`
restricts the old-policy impossibility theorem to exact nominal clean geometry;
`:300` stops presenting the 3.5-times-RMS proxy as an expected failure rate.

The new isolated angular test source contains decisive threshold/monotonicity
checks (`tests/test_cad_guided_instances_angular_gap_v9.py:69`), clean phase
counts (`:86`), missing interior columns (`:101`), per-slab rather than global
support and axial-only neighbors (`:110`), and the scalar chord/arc/strip
arithmetic (`:132`). These preserve the sample-supported meaning rather than
inventing continuous coverage. The connected test at
`tests/test_cad_guided_v9_connected_chain_v1.py:301` carries a real fitted
measurement through 5-degree support refusal and 30-degree admission, retaining
null F1 on no emitted carrier. Its module explicitly labels the copied-carrier
scoring reference as transport/arithmetic, not independent localization truth
(`:8`). No synthetic-to-physical accuracy comparison follows.

## Nonblocking precision notes

1. **P3 — “Diagnostics on every refusal” remains too broad if read literally.**
   `AMBIGUOUS_FLANGE_WINDOWS` is raised at kernel `:612` before the diagnostics
   structure is allocated at `:617`; invalid input/alignment similarly reaches
   the public refusal before measurement. Such rows have `diagnostics=None`,
   not an explicit phase object. Within a flange role, numerical failure after
   computing its `before` object but before appending the completed role loses
   that in-progress object; prior completed role records survive. The actual
   supported claim is “completed diagnostic prefix retained on the wrapped
   fit/selection failures; no spectra asserted for pre-diagnostic refusals.”
   This does not invent successful evidence or block generic contact scoring.
   The summary should use that bounded wording; universal all-exit telemetry
   would require an earlier shell record and finer per-role staging.
2. **P3 — One numeric-uniqueness test is imprecise, although the main closure
   tests are decisive.** At angular-test `:93` the comment says two jittered
   columns “still refuse” the 20-degree span, but their nominal separation is
   22.5 degrees. The assertion at `:96` accepts three outcomes, followed by a
   repeated identical call. It establishes only deterministic behavior outside
   the exact-distinct-count refusal, not a specific noise-cluster contract.
   Use the exact returned outcome or explicit retained-angle/span/gap
   assertions when tightening this test; no physical calibration or full new
   experiment is needed. Also the kernel's introductory “largest gap” wording
   (`:17`) should be read as historical shorthand: the executable rule is all
   qualifying gaps with selection only when exactly one qualifies (`:246`).

These do not reopen the closed sign, disabled-mode, provenance-summary,
proxy-naming or angular-policy findings. The test counts in the author
checkpoint (including 28 kernel and 5 angular cases) remain author-reported
execution evidence at matching source hashes; this reviewer assessed their
source assertions only.

## Handoff boundary

This seam has no identified blocker to the **separately reviewed** candidate
freeze/PRE process. Retain the above narrow limitations in the final audit
ledger. No additional physical run, changed score rule, calibration claim or
research expansion is requested. Overall readiness and the sole run remain
subject to the root's other closure audits and established custody gates.
