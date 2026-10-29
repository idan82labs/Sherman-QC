# V70 BRMVT terminal runtime retirement

Date: 2026-08-16  
Architecture: `V70_BOUNDARY_REJECTING_MULTISCALE_VARIATION_TRACK_COMPOSITION_V1` (`BRMVT_V1`)  
Status: **terminally retired; no rescue or rerun authorized**

## Outcome

V70 passed its complete locked source-free suite: **24/24 tests** in 373.08 seconds. The suite covered ten positive morphologies, ten negative/ownership morphologies, exact comparator boundaries, input permutation, endpoint reversal, triangle-winding reversal, forbidden-field independence, and exact triangle refinement.

The one-shot exposed-real U01–U04 actionability preflight then failed before producing any candidate decision. Every primary and replay reached the atomic `WALL_TIME_CAP` during the global multiscale source-field stage. All four abort records were byte-identical across their primary/replay pair. No modified candidate output was committed.

| Opaque scan | Vertices | Triangles | V16 candidates | Primary | Replay | Peak RSS | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| `S8BB74BDA675656176092` | 150,194 | 300,000 | 13 | 301.996s | 300.374s | 1.85 GiB | `WALL_TIME_CAP` |
| `S9567D0D3F33D80F7C915` | 161,295 | 322,216 | 20 | 302.439s | 302.441s | 1.94 GiB | `WALL_TIME_CAP` |
| `SDB1839480738AC85934F` | 152,820 | 299,999 | 3 | 302.151s | 301.825s | 3.22 GiB | `WALL_TIME_CAP` |
| `S241AA95BFDFC185004F0` | 153,535 | 299,999 | 12 | 300.516s | 301.906s | 3.49 GiB | `WALL_TIME_CAP` |

The three-candidate case failed identically, so candidate count is not the cost driver. The failure occurs in the all-source-point, six-scale, up-to-1,280-neighbor covariance/percentile field before candidate-local track evaluation.

## Gate accounting

- fixture suite passed: **yes**
- exact 48-candidate exposed stream: **yes**
- primary/replay byte-identical: **yes, 4/4**
- every execution below 8 GiB: **yes**
- no scan abort: **no**
- every execution at most 240 seconds: **no**
- mechanical actionability gate: **failed**
- native-resolution raw-first visual gate: **not opened**
- reserved physical scan: **still sealed**
- truth, gold, labels, CAD, expected counts, identity, and benchmark feedback: **not opened**

Candidate counts recorded on abort are unchanged fallback streams and are not geometry decisions. They cannot be used as actionability evidence.

## Scientific interpretation

This run does **not** falsify the multiscale ridge/track geometry hypothesis on real morphology, because no real candidate reached evaluation. It falsifies this global-field realization under the locked operational envelope.

The smallest legitimate successor question is architectural: can the same boundary-rejecting multiscale evidence be computed sparsely or candidate-locally, with explicit coverage guarantees, without tuning or rescuing V70?

## Immutable evidence

- verbatim Pro ruling: `docs/research/2026-08-16_gpt56_pro_post_v69_real_morphology_ruling_verbatim.md`
- implementation: `backend/f1_boundary_rejecting_multiscale_variation_track_composition_v70.py`
- fixtures: `tests/test_f1_boundary_rejecting_multiscale_variation_track_composition_v70.py`
- runner: `scripts/run_f1_v70_u01_u04_actionability_preflight.py`
- output root: `output/research/f1-v70-brmvt-u01-u04-actionability-v1-20260816`
- lock SHA-256: `f22e1a3c77d58b8683ea5411c422abd07103082f39d2e9d39835b4e24229e7b4`
- summary SHA-256: `2da80bbc07f9ba1233f41bcaf4277fd6e0a46f049c171caa2f6ddb19b69510bc`
- implementation SHA-256: `92f5cb62ec1bb250254d06aac644985668d304c8623ac42f044872de2d162e81`
- fixture SHA-256: `793f815786ef73c116178ffabbeea875ad265b2a3ee68f798fe1394a944c20e7`
- runner SHA-256: `4c4a299307838f56162e8115bd02366f22a6146a277f93717d5280752346c5bc`

