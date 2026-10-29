# CAD bend extraction: measured technical-reference baseline

2026-09-30. Current retained extractor V7 / exact-trim V6 campaign.

## Result

**98 of 127 independently reviewed geometric bend regions were extracted:
77.17% known-bend capture, with 29 known misses.** All 98 emitted features match
one reference feature; there are no unmatched predictions in this run.

This is a real-input CAD detection result on the existing 20 unique STEP models
(23 filename aliases), not a synthetic result or physical Gold11 scan F1.
The reference is an independent AI technical review, not human manufacturing
approval. Eight models retain unresolved geometry, so 77.17% is capture of the
verified positives, **not a complete whole-corpus recall estimate**.

| Metric | Result | Population / qualification |
|---|---:|---|
| Known-bend capture | **77.17% (98/127)** | Verified positive regions across all 20 CADs |
| Precision | **100% (98/98)** | Every prediction matches a verified positive; conditional on the frozen reference ontology |
| Recall, complete-reference subset | **79.38% (77/97)** | 12 CADs with complete references |
| Micro F1, complete-reference subset | **88.51%** | 77 TP, 0 FP, 20 FN on those same 12 CADs |
| Exact bend-count agreement | **50.00% (6/12)** | Complete-reference CADs only |
| Part-macro recall / F1 | **68.90% / 72.98%** | Each of the 12 complete-reference CADs weighted equally |
| Reference-complete coverage | **60.00% (12/20)** | Eight partial references remain in the census |
| Zero-output parts | **15.00% (3/20)** | All three contain verified bends; these were not process failures |
| Optimistic whole-corpus recall ceiling | **77.17%** | Unknown regions assumed to add no missed bends |
| Optimistic whole-corpus F1 ceiling | **87.11%** | Same optimistic assumption; an upper bound, not a measured point F1 |

**The 90% extraction target is not met.** Complete-subset F1 cannot substitute
for whole-corpus recall, precision and reference coverage. At least 17 more of
the currently known 127 bends must be recovered, without losing existing
matches, to reach 90% known-positive recall (115/127 = 90.55%). Additional truth
from unresolved regions can increase that requirement. There is no evidence
here of 90%+ extraction, CAD-agnostic unfolding or manufacturing readiness.

## All CADs retained

Each row is one unique raw-file hash; aliases are not duplicate test cases.
“Capture” below uses the known reference count. A partial row's denominator
can grow when its unresolved geometry is reviewed; 100% on that row does not
establish completeness. All emitted predictions matched, so emitted = matched.

| Case | Representative CAD | Reference | Known bends | Matched | Missed | Capture |
|---|---|---|---:|---:|---:|---:|
| 01 | 47959001_01PREL.stp | Partial | 6 | 6 | 0 | 100.00% |
| 02 | 15651004_01.stp | Partial | 3 | 3 | 0 | 100.00% |
| 03 | 44211000_A.stp | Complete | 1 | 1 | 0 | 100.00% |
| 04 | 49024000_A.stp | Partial | 3 | 0 | 3 | 0.00% |
| 05 | 38172000_01.stp | Complete | 12 | 12 | 0 | 100.00% |
| 06 | 23553000_B.stp | Complete | 16 | 16 | 0 | 100.00% |
| 07 | 49204020_04.stp | Complete | 12 | 12 | 0 | 100.00% |
| 08 | 42288005_01.stp | Partial | 8 | 8 | 0 | 100.00% |
| 09 | 37361005_01.stp | Complete | 4 | 4 | 0 | 100.00% |
| 10 | Default (1).stp | Partial | 3 | 1 | 2 | 33.33% |
| 11 | 48963008_B.stp | Partial | 3 | 1 | 2 | 33.33% |
| 12 | 48811000_A.stp | Complete | 4 | 3 | 1 | 75.00% |
| 13 | 48991006_A 4.stp | Complete | 5 | 1 | 4 | 20.00% |
| 14 | 28870000_A.stp | Partial | 3 | 1 | 2 | 33.33% |
| 15 | 47479000_C.stp | Complete | 1 | 0 | 1 | 0.00% |
| 16 | 11979003_C (1).stp | Complete | 11 | 9 | 2 | 81.82% |
| 17 | 49001000_A-PRELIM.stp | Complete | 5 | 0 | 5 | 0.00% |
| 18 | 10839000/10839000_01.stp | Complete | 12 | 12 | 0 | 100.00% |
| 19 | 40500000_0 1.stp | Complete | 14 | 7 | 7 | 50.00% |
| 20 | 47266000_F.stp | Partial | 1 | 1 | 0 | 100.00% |
| **Total** | **20 unique CADs** | **12 complete / 8 partial** | **127** | **98** | **29** | **77.17%** |

Cases 13, 17 and 19 account for 16/29 known misses (55.17%). Cases 04, 15 and
17 produced zero output despite nine verified bends. This concentration is a
development priority, not proof of a trained/undertrained family effect: this
path is deterministic geometry, and no family-generalization claim is made.

## Why the 29 bends were missed

The post-score [miss ledger](../../artifacts/cad-bend-reference-score-v1-2026-09-30-attempt1/MISS_DIAGNOSIS.json)
traces every missed reference group to the retained extractor receipts and
unchanged source. Both cylindrical skins were recognized for all 29 groups;
the losses occur at later acceptance gates, not initial carrier recognition.

| Mechanism | Missed bends | Evidence / consequence |
|---|---:|---|
| Rectangular trim certificate rejects spline boundaries | **24** | All 48 skins carry `PCURVE_OUTSIDE_CORRIDOR_OR_BACKTRACK`; current representation requires a bounded axis-aligned rectangular footprint |
| Rolled regions with multiple boundary wires / holes | **3** | Six skins carry `MULTIPLE_WIRES_OR_HOLES_UNSUPPORTED`; they also have no planar tangent rails, a second unsupported assumption |
| Curved continuation instead of a second planar flange | **2** | Trim domains pass, but pairing requires two full planar tangent rails per skin; the other boundary continues into a cylinder |

The first category accounts for **82.76% of known misses**. A failed sufficient
rectangle certificate does not prove that the actual curve backtracks or that
the bend is absent. All 48 failed spline skins already have proved active
intervals, and their restricted ranges equal their underlying ranges; another
interval-only fix does not address these recorded refusals.

The main next mechanism is a general finite trimmed-cylinder material domain,
with ordered boundaries, holes and paired-skin ownership, rather than treating
every bend as a rectangle. Bend detection should retain an independently proved
feature even when the separate unfolding path cannot certify it. Rolled-region
and tangent curved-continuation semantics require explicit extensions and
negative hole/fillet controls; removing flange checks globally or enlarging
tolerances would not be a justified fix. This diagnosis identifies the blocking
gates; it does not prove a proposed replacement will recover all 29 bends.

Miss-ledger SHA-256:
`5b06b30dcc630176a24f29c8d069b14503f12bd788a5d629bfba24f01586cd93`.
No engine changes or new CAD execution were performed during this diagnosis.

## What was measured

The frozen ontology is a finite connected geometric sheet-turn region.
Disconnected finite strips are separate features even when one press stroke
might form them; rolled regions are included when their material geometry is
supported. This is not a count of manufacturing operations or user-defined
מהלכים.

Matching is exact native-face union, one to one. The same retained native
topology indexing connects the reference and prediction evidence. Duplicate,
split or merged detections do not receive multiple matches. This evaluation
does **not** score line placement, angle, radius, flange depth, neutral-axis
allowance, developed length or manufacturing flat-pattern correctness.

Two non-collector reviewers inspected the neutral all-face topology and six
unmarked views per model. Both independent initial reviews and their
adjudication are retained. They agreed 127 positive regions; 12 complete
references and eight partial references were frozen before detailed current
prediction comparison. There are 447 unresolved faces among 2,996 native faces;
that is not a count of unknown bends.

This was not a strictly blind trial. Reviewers had read historical narratives
containing some counts and face IDs. One reviewer received the other review's
aggregate after saving initial positives, before final metadata adjudication;
the original and updated records are preserved. Detailed current predictions
were withheld until the reference freeze. Labels remain AI technical judgments
with these disclosed limitations, not certified human ground truth.

The partial-reference bound extension was mathematically reviewed before
comparison. Here T=98 matched positives, M=29 known misses, F=0 proved false
positives and U=0 unscored predictions. Thus precision is 98/98, recall is at
most 98/127, and F1 is at most 196/(196+29)=87.11%. Unbounded unknown missed
bends make lower recall/F1 bounds vacuous, not their upper bounds. See the
[bounds proof](cad-reference-partial-bounds-v1-2026-09-30.md). This independently
audited extension corrects the earlier policy's overly broad statement that
unknown truth makes all recall bounds vacuous; the matching rule is unchanged.

## Frozen evidence and execution

- [Prediction campaign](../../artifacts/vendor-cad-corpus-exact-trim-v6-2026-09-30-attempt1/POST_AUDIT.json): all 20 cases retained; 98 predictions, 19 folded displays and six flat previews. It was not rerun for scoring.
- [Neutral evidence POST](../../artifacts/cad-reference-evidence-v1-2026-09-30-attempt1/POST_AUDIT.json): all 20 native collections complete; no face/edge/topology/pcurve collection errors. Unsupported coefficient representations remain explicit.
- [Neutral visual gallery](../../artifacts/cad-reference-neutral-gallery-v1-2026-09-30-attempt1/MANIFEST.json): all 20 CADs, six views each; surface color is not a bend label.
- [Frozen technical reference](../../artifacts/cad-bend-reference-review-v1-2026-09-30-attempt1/REFERENCE_FROZEN.json): SHA-256 `5fed73aa0b81b3e8ebbb58257a7454daee5cd68e2590ee14ef99e44e1666d23b`.
- [Comparison freeze](../../artifacts/cad-bend-reference-score-v1-2026-09-30-attempt1/FROZEN.json): SHA-256 `a8c592c3d29fa83eff900645d60f8b70b3cfdab32d91f0c83346f8741fbb2748`.
- [Independent PRE](../../artifacts/cad-bend-reference-score-v1-2026-09-30-attempt1/PRE_AUDIT.json): exact source, reviews, input bindings and mathematical extension approved before execution.
- [Independent POST](../../artifacts/cad-bend-reference-score-v1-2026-09-30-attempt1/POST_AUDIT.json): accepted technical-reference comparison with limitations; an independent face-union counter reproduced every case and all aggregate metrics. SHA-256 `78c77a796863d896718e3010df736bcdd0b139954d3a89cfa97b3a27bc7acba8`.
- [Machine-readable score](../../artifacts/cad-bend-reference-score-v1-2026-09-30-attempt1/SCORE.json): SHA-256 `728acc08ed2c39e5f5dfdee15260340b32c895be286888d2b4a046691cb07e47`.
- [Outer terminal receipt](../../artifacts/cad-bend-reference-score-v1-2026-09-30-attempt1-supervision/TERMINAL.json): exit 0, known cleanup, 0.0937078 seconds, 13,408 KiB peak sampled RSS; fixed 30-second / 1-GiB cap. This is scoring time, not CAD processing time.

Focused checks before freezing included 18 collector/supervisor/scorer tests
with six subtests, 11 comparison-contract tests, eight consensus tests and
eight bounds tests. Bounds tests enumerate more than 10,000 synthetic ratios.
These validate the machinery, not the correctness of physical manufacturing
dimensions. No product runtime, UI default or public website was changed.

### Reproduction contract

The recorded terminal invocation, from the repository root, was:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_cad_reference_score_v1.py \
  --out artifacts/cad-bend-reference-score-v1-2026-09-30-attempt1 \
  --frozen-sha256 a8c592c3d29fa83eff900645d60f8b70b3cfdab32d91f0c83346f8741fbb2748
```

That directory is terminal and must not be reused. An authorized independent
reproduction uses the same pinned reference and prediction artifacts with a
new exclusive output directory, a newly recorded freeze and independent PRE.
The freeze command template is:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/score_cad_bend_references_v1.py \
  --freeze --out artifacts/NEW-EXCLUSIVE-COMPARISON-DIRECTORY \
  --reference artifacts/cad-bend-reference-review-v1-2026-09-30-attempt1/REFERENCE_FROZEN.json \
  --reference-sha256 5fed73aa0b81b3e8ebbb58257a7454daee5cd68e2590ee14ef99e44e1666d23b \
  --predictions artifacts/vendor-cad-corpus-exact-trim-v6-2026-09-30-attempt1 \
  --post-sha256 0bef61bca4641323d0ba07b81d49fc9761ceedaee9489c71feb6f183c71598f4
```

After independent PRE, use the supervisor with that new directory and its exact
`FROZEN.json` SHA-256. This only reads released derived evidence; it neither
re-extracts raw CAD nor repairs historical attempts. Future engine improvements
require a separately frozen candidate and audited development run, retaining
the same references, failures and all 20 inputs.
