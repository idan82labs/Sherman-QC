# GPT-5.6 Sol Pro packet — post-PSRO_V1 architecture selection

## Requested ruling

`PAIRED_SKIN_RIBBON_OWNERSHIP_V1` (V18 / PSRO_V1) is terminally stopped at
its preregistered C017 source-only activation gate. Select exactly one next
smallest falsifiable source-only architecture for the native full-resolution
mesh lane, or issue a formal research STOP if the remaining causal information
cannot be recovered under the frozen scan-only constraints.

Do not rescue PSRO_V1, relax its station/reciprocity/thickness/mate gates, add
a scan-level cap, use C017 identity, or propose a sweep. Ground the new
architecture in relevant published research and distinguish established
machinery from the Sherman-specific hypothesis.

## Retained baseline and objective

V16 remains retained and frozen:

- 35 development scans / 25 part families;
- 132 TP / 60 FP / 44 FN;
- precision 0.6875, recall 0.75, F1 0.717391;
- 4/35 exact-count cases;
- minimum-family F1 0.0;
- frozen U02--U05: 15 TP / 0 FP / 0 FN with byte-identical replay.

The active objective remains a trustworthy zero-shot daily-workflow system:
at least 90% exact count, 90% precision, 90% recall, no major family below 80%
recall, 99% render-admissible markers, deterministic CPU replay, P95 <=300 s,
peak RSS <16 GiB, compact caches, and no identity, label, CAD, expected-count,
benchmark-feedback, or target leakage.

## Frozen constraints

- Blind scan-only inference; no CAD, drawings, labels, gold, expected count,
  filename/identity, production feedback, or benchmark feedback.
- Deterministic CPU path.
- Approximately 200,000-point bounded detector input.
- Compact fixed/bounded caches; no source-point, vertex, face-list, or source
  cloud serialization.
- Peak process-tree RSS below 16 GiB; P95 latency at most 300 seconds.
- Research only; runtime/main and production remain closed.
- No part-specific constants, post-hoc rescue, architecture sweep, or tuning
  after the locked source-only outcome.
- One bounded implementation, preregistered source gates, manual native visual
  audit of every activation/mismatch, then one-shot locked gold only if gates
  pass.

## V18 architecture actually executed

The complete Pro ruling is preserved verbatim with SHA-256
`703549bf3f2df050ad3772c799a6926f1493fcaf32d0fd139545eea021af7a83`.

PSRO_V1 asked whether a frozen V11-rejected `scan_mesh_adjacency` line belongs
to a constant-thickness four-owner ribbon:

`A-c-B`, `Abar-m-Bbar`, with reciprocal thickness pairs `A<->Abar` and
`B<->Bbar`.

The historical producer did not expose a reusable BVH, pre-merge face-owner
table, transient side support, or transition-parent graph. The authorized
bounded addition was therefore implemented in one native-mesh pass as:

- one transient Open3D ray scene (no second mesh copy serialized);
- raw pre-opposite-skin-merge planar face regions as deterministic owners;
- a bounded exact shared-edge owner transition-parent graph;
- five fixed stations per near side;
- exactly ten primary rays and two reciprocal center rays per eligible line;
- four distinct owners, thickness stability, thinness/width, reciprocal return,
  conjugate mate, canonical ribbon-key collapse, and incumbent-first frozen V11
  NMS;
- one fixed 224-byte candidate record plus a 64-byte scan header;
- no V17 field, threshold, or decision.

The ray epsilon is a self-hit guard only; recorded thickness is measured from
the selected origin-face centroid. Candidate geometry remains the frozen V11
cropped line.

Frozen code hashes:

- core: `8db85b027fa10daa87373542e7641c9bbc71ed9a8f44eae2e997bd35965ee8d4`
- runner: `8572a0ecd8c8558a1844ab518b2b2d7816e74b34e460a13578faedbc6d4ed2c1`
- construction tests: `26b5a9bddc9d9a5335234c2070ffff8e57ecc736c8fe74197cd6be5a1abbd3a0`

## Source-free construction result

Six tests passed, including:

- finite-thickness paired inner/outer L-fold retains exactly one ribbon;
- perimeter/thickness edge rejects;
- single miter without opposite-skin transition rejects;
- symmetric presentation collapses canonically;
- three rigid transforms preserve the semantic result;
- full face-orientation reversal preserves the result;
- record size is exactly 224 bytes.

The core and retained V16/V17 regression subset pass 32/32 tests.

## U02--U05 label-closed result

U02 ran first. Three V11-rejected adjacency lines were probed; zero survived.
The locked false-positive candidates both reject:

| candidate | masks | reciprocal | opposite owners | retain |
|---|---:|---:|---:|---:|
| SMA1 | 30 / 0 | true / false | 47 / invalid | false |
| SMA2 | 28 / 3 | true / false | 2 / 47 | false |

The third eligible line SMA6 also rejects (masks 31 / 4). The separate cache
is exactly 736 bytes (`64 + 3*224`). Frozen V16 replay A/B remain byte-identical.

U02 resources: 4.39 s wall, 895,582,208-byte max RSS.

U03--U05 remain on frozen V15; PSRO is not invoked, each has zero appends and
only a 64-byte empty sidecar header.

## C017 source-only terminal result

Gold remained closed. All three V11-rejected adjacency lines reject:

| candidate | near owners | opposite owners | masks | reciprocal | mate |
|---|---:|---:|---:|---:|---:|
| SMA3 | 207 / 7 | 29 / 20 | 30 / 1 | true / false | none |
| SMA2 | 210 / 146 | invalid / 103 | 0 / 24 | false / false | none |
| SMA1 | 207 / 148 | 29 / 42 | 9 / 12 | false / true | none |

Zero survives mandatory endpoint/center coverage and reciprocity. No candidate
reaches an admissible opposite-skin mate. C017 resources: 2.04 s wall,
494,370,816-byte max RSS, 736-byte cache.

Native-resolution four-view failure audit:

`output/research/f1-new-parts-20260813/v18_psro_preflight/C017/native_failure_audit.png`

SHA-256:
`05fddac158859f480a7525c4b49b1c11055c79f79f43bcc661c4a51dbfa620bf`

Manual visual finding: all three axes track long plausible structural
transition bands. Valid opposite-surface hits are sparse/asymmetric and absent
at many required endpoints. The result is consistent with fragmented,
occluded, or locally mis-owned scan faces defeating exact face-centroid station
reciprocity, rather than proving that the visible folds do not exist. This
distinction is observational only; no V18 constant was changed and no rerun
rescue occurred.

Evidence hashes:

- terminal report: `b39853817c62011264dc354ef666c5ac532e0e345f2eb56529ebfa74a946a6bf`
- U02 report: `0787549e6d92995e0634a8c3eba34fca0e4d8254e3feeb2fb99b81241e924fb1`
- U02 cache: `e5a07ab1a71899808e59ee830cb9a2cc83947d0bda632762f3fdf003301d56aa`
- C017 report: `fa28aac268b6d59acaee95d96937eff23819c57cab31731f2d8384f58aab1642`
- C017 cache: `c58ae9f74d608a695e2d4a6feb59b2068c2a1ec9ab0e9722f971b282a3ab34fa`

## Scientific interpretation

PSRO demonstrates that strict local reciprocal material ownership can suppress
the known U02 perimeter/thickness aliases. It fails recall because a true or
visually plausible scan transition does not consistently provide a valid
opposite face at fixed endpoint/center stations. The remaining causal gap is
not another scalar threshold on axial evidence. It is how to represent
material-side continuity and interior-fold ownership under missing,
fragmented, noisy, or locally mis-oriented mesh faces without reopening rim,
thickness-wall, or seam false positives.

Potential directions such as bounded geodesic owner completion, probabilistic
or interval ray support, ribbon topology over adjacent surfel modes, or
component-boundary parity are **not authorized suggestions**. They are named
only to clarify the representation question; select one architecture from
first principles or STOP.

## Required response format

1. **RULING:** exactly one intervention name, or formal STOP.
2. **RESEARCH BASIS:** cite established published methods/ideas and state the
   exact narrow Sherman adaptation.
3. **CAUSAL FEATURE:** explain how it tolerates scan fragmentation while still
   rejecting silhouette, opening rim, thickness wall, corner cycle, nearby
   panel pairing, and seam evidence.
4. **EXACT SPECIFICATION:** eligible pool, bounded generator/schema addition if
   needed, deterministic construction, fields, inequalities/predicates,
   insertion point, ranking, duplicate/NMS behavior, and serialization/resource
   bounds. No score sweep.
5. **FALSIFICATION GATES:** source-free constructions; U02 must remain a no-op;
   C017 must activate before gold; native visual artifacts required before any
   labels; one-shot 35-scan and fresh held-out gates; determinism, memory,
   latency, and job reliability.
6. **EXPECTED FAILURE MODE:** the single most likely false-positive mechanism
   and the visual/native artifact that exposes it.

If the causal information is not identifiable from the permitted scan-only
representations, issue a precise STOP and state the smallest scientifically
necessary capability opening. Do not authorize multiple alternatives.

Evidence paths:

- `docs/research/2026-08-14_f1_v18_psro_v1_terminal_stop.md`
- `docs/research/2026-08-14_gpt56_sol_pro_psro_v1_ruling_verbatim.md`
- `backend/f1_paired_skin_ribbon_ownership_v18.py`
- `scripts/run_f1_psro_v18_preflight.py`
- `tests/test_f1_paired_skin_ribbon_ownership_v18.py`
- `output/research/f1-new-parts-20260813/v18_psro_preflight/`
- `output/research/f1-new-parts-20260813/v16_unified_development_regression.json`
