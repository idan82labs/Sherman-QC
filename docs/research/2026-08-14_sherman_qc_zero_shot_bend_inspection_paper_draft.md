# Source-only zero-shot bend-line inspection from bounded 3D scans

**Working paper — methods and research-governance draft**  
**Confirmatory results, abstract claims, discussion of transfer, and conclusion
remain embargoed until the frozen 30-scan/15-family evaluation is complete.**

## Abstract (structured placeholder)

**Problem.** Reliable bend inspection from incomplete 3D scans requires more
than predicting a bend count: a system must localize individual bend axes,
generalize across part families, remain computationally bounded, and avoid
using CAD, filenames, expected counts, or evaluation labels during inference.

**Method.** We present a deterministic source-only pipeline that converts a
bounded scan representation into competing geometric bend hypotheses, routes
among topology representations, consolidates redundant hypotheses, and applies
frozen ownership and recovery primitives. The current prospective candidate,
V51, adds at most one parallel return axis under a strict geometric gate.

**Evaluation.** The confirmatory protocol uses 30 untouched scans from 15
held-out families, two captures per family, dual independent scan-only
annotation and separate adjudication, blind primary and replay execution, a
frozen one-to-one line matcher, all-case native-resolution visual audit, and
resource/leakage gates.

**Results.** *Reserved until the locked tranche is executed and scored once.*

**Conclusion.** *Reserved. No transfer or production claim is currently
authorized.*

## 1. Introduction

Industrial bend inspection is often framed as a classification or count
problem. That framing is insufficient for quality control. Operators need to
know which physical bend instance was detected, where its axis lies on the
observed part, whether the marker is visually admissible, and whether the same
system behaves consistently on an unseen capture of a related geometry.

This work studies a stricter setting: inference receives scan geometry only.
CAD, drawings, expected bend counts, part identity, filenames, family labels,
and evaluation outcomes are unavailable to the detector. The system must also
run deterministically on CPU, use no more than approximately 200,000 working
points, avoid serializing source point geometry into compact caches, remain
below 16 GiB peak resident memory, and complete within a 300-second P95 target.

The main contribution is therefore not a single geometric heuristic. It is a
combined detection and evidence discipline:

1. a bounded multi-representation geometric hypothesis pipeline;
2. explicit line-instance output rather than count-only output;
3. frozen, falsifiable intervention primitives with no post-result rescue;
4. family-held-out, leakage-resistant evaluation with deterministic replay;
5. mandatory native-resolution visual reconciliation of every case.

## 2. Task definition

For scan points \(X \subset \mathbb{R}^3\), the detector emits a set of bend
line segments \(P=\{p_j\}\). An adjudicated scan-only reference contains bend
segments \(G=\{g_i\}\). A predicted and reference line may match when four
frozen conditions hold: axis-angle difference at most 5 degrees, transverse
offset at most 4 mm, gold-span overlap at least 0.60, and prediction-span
overlap at least 0.60. A one-to-one minimum-cost assignment prevents one
prediction from explaining multiple reference bends.

Primary metrics are exact per-scan bend count, aggregate line precision,
aggregate line recall, line F1, and minimum family recall. Supporting gates are
render-admissible marker accuracy, byte-identical replay, job completion,
latency, memory, point bounds, cache safety, and complete visual audit.

## 3. Source-only geometric pipeline

### 3.1 Bounded scan representation

Each case is processed under a 200,000-point global budget. Source geometry is
used to derive bounded identifiers, topology summaries, line hypotheses, and
compact support relationships. Raw/source point arrays are not serialized into
the candidate cache. This separates the evidence needed for deterministic
selection from the high-volume input representation.

### 3.2 Multi-representation hypothesis generation

The pipeline constructs several complementary geometric views, including
patch-graph latent decomposition, mesh/topology candidates, cylindrical bands,
plane-lattice structures, and line-object groupings. The purpose is not to
assume that every bend has one canonical local signature. Instead, competing
representations expose different failure modes under incomplete coverage,
opposite skins, fragmentation, rims, junctions, and repeated parallel returns.

### 3.3 Frozen route and consolidation chain

The prospective chain is:

`V16 → V43 → V44 → V50 → V47 → V48 → V49 → V51`.

V16 selects a bounded source-only route using two frozen 76-feature models.
Subsequent stages enforce two-sided support, fragment stitching, longest-
incumbent ownership, route-independent elongated-lane handling, nested partial
duplicate suppression, and short-span cross-source duplicate suppression. Each
stage is deterministic and emits typed audit fields.

### 3.4 V51: single-axis parallel-return recovery

V51 is deliberately narrow. It activates only when exactly one surviving
prediction is a long cylindrical-standard axis spanning at least 75% of the
scan bounding-box extent. It considers a candidate only when it comes from a
different cylindrical family, is at least 75% as long as the incumbent, differs
by at most 2 degrees in axis, is offset by at least 8 mm but by no more than 8%
of incumbent length, and overlaps at least 80% of the shorter span. At most the
highest-ranked eligible candidate is appended.

This primitive encodes a falsifiable physical hypothesis: a second persistent,
spatially distinct, nearly parallel transition may be a missed return bend.
Family inequality alone does not prove distinct physical ownership. The
precommitted expected failure is therefore a false append where two cylindrical
families fit the same transition, opposite skins, a rim, or an unsupported
offset locus.

## 4. Research discipline and negative results

The project adopted terminal, one-shot interventions after repeated evidence
that unrestricted iteration on a small labeled corpus can convert geometric
debugging into answer-conditioned tuning. Failed architecture families are
retained as negative results rather than rescued by threshold sweeps. The
history includes local-plane fragments, Cubical IIP, structured decomposition
and residual architectures, mesh/adjacency variants, and several source-only
completion proposals. Their value is diagnostic: they identify representation,
ownership, accounting, and evaluation-protocol failures that a successful
system must avoid.

One V51 attempt on the historical Gold37 corpus was invalidated before detector
execution when an operator printed sealed manual-bend fields during a manifest
check. No prediction or score was produced, but the information exposure made
that corpus unusable for prospective V51 evidence. The architecture was not
geometrically falsified; the evaluation identity was. Gold37 is permanently
restricted to development/diagnostic history for this architecture.

The first new-corpus source-binding lock was also invalidated by its own
fail-closed preflight: direct script execution omitted the repository root from
the import path. This occurred before corpus assembly, truth access, or detector
invocation. The immutable failure record was retained, the launch-only defect
was corrected, and a superseding V2 lock was sealed and fully preflighted.

## 5. Confirmatory evaluation protocol

### 5.1 Corpus construction

An independent custodian, not exposed to V51 predictions or historical truth,
selects the first 15 eligible families and first two eligible scans from each
family under a rule signed before candidate review. Acquisition failures use a
static exclusion ledger. Accepted scans are staged only as `O001` through
`O030`; identity and family mappings remain outside the detector sandbox.

### 5.2 Annotation

Two annotators independently mark bend-line endpoints using scan geometry only.
They cannot see predictions, each other's labels, expected counts, CAD,
drawings, filenames, family membership, or benchmark results. A separate
adjudicator resolves differences before detector invocation. Final truth and
family bytes are sealed by SHA-256.

### 5.3 Blind execution and scoring

All 30 primary cases run in opaque order without inspection, scoring, early
stopping, or case-dependent action. All 30 are then replayed in the same order
and environment. Canonical semantic bytes must match for every case. Only after
both sets are immutable may the scorer open truth and family membership and
write one complete score transaction.

### 5.4 Visual falsification

Every case receives a prediction-only render, a post-score truth/prediction
overlay, and an axis-normal cross-section artifact. For active V51 cases, nine
slices perpendicular to the incumbent axis show local source support around
the incumbent and appended cylindrical families in distinct colors. A true
return should remain spatially separated across stations. A false return tends
to collapse onto one transition/skin, track a rim, or occupy an unsupported
locus. Visual findings cannot modify truth, predictions, matching, or metrics.

### 5.5 Gates

The transfer candidate passes only if all gates pass simultaneously:

- at least 27/30 exact bend counts;
- line precision and recall each at least 0.90;
- every family recall at least 0.80;
- render-admissible marker accuracy at least 0.99 (30/30 under this denominator);
- 30/30 primary and 30/30 replay jobs;
- byte-identical replay for all 30 cases;
- at most 200,000 working points and no forbidden source geometry in caches;
- peak RSS strictly below 16 GiB and primary P95 latency at most 300 seconds;
- complete native-resolution visual audit for all 30 cases.

A semantic miss retires V51. A protocol or artifact failure invalidates the
event according to the precommitted opening stage. There is no near-pass,
family waiver, visual override, threshold sweep, or V51b rescue.

## 6. Results

### 6.1 Pre-confirmatory evidence

The prospective P01–P07 sequence is hypothesis-generating only. It showed one
visually plausible parallel-return activation and an independent no-op, but it
does not provide trusted line F1 because those scans lack independently sealed
manual bend-line truth.

### 6.2 Frozen 30-scan evaluation

**Not yet executed. Do not insert intermediate, partial, development-corpus, or
visually inferred numbers here.**

Required reporting table:

| Metric | Frozen result | Gate |
|---|---:|---:|
| Exact bend count | pending / 30 | >= 27 / 30 |
| Line precision | pending | >= 0.90 |
| Line recall | pending | >= 0.90 |
| Line F1 | pending | reported from aggregate TP/FP/FN |
| Minimum family recall | pending | >= 0.80 |
| Marker accuracy | pending / 30 | >= 0.99 |
| Replay identity | pending / 30 | 30 / 30 |
| Job success | pending / 60 | 60 / 60 |
| Peak RSS | pending | < 16 GiB |
| Primary P95 latency | pending | <= 300 s |
| Visual audits | pending / 30 | 30 / 30 |

## 7. Discussion (embargoed structure)

After the locked result, discuss only precommitted questions:

1. Did V51 improve recall without producing parallel-family false ownership?
2. Which families failed and were failures geometric, representational,
   rendering-related, or acquisition-related?
3. Did axis-normal stacks agree with metric-level error classification?
4. Were resource and determinism constraints compatible with accuracy?
5. What claims are supported for known versus near-neighbor families?

## 8. Limitations

The study targets sheet-metal-like geometry and may not transfer to free-form,
cast, highly reflective, or severely occluded parts. Two scans per family allow
within-family repeatability checks but limited characterization of acquisition
variance. Scan-only annotation can itself be ambiguous when the physical bend
axis is not observable. The current architecture remains a deterministic
research system, not a calibrated uncertainty model or production release.

## 9. Reproducibility status

The valid precommit binds 109 source, model, configuration, dependency, test,
matcher, renderer, serializer, and protocol artifacts. V2 lock SHA-256 is
`e376b1933755419ee792e218d5824dd905a858a7f60b153348ab7f59b93661a3`.
The complete source-free preflight passed 11 checks and is sealed at
`4c41e3fa0b337761581c9cd7e714a3d4036442cac2157a75c401c1bf8f6ffb73`.
Runtime/main and production remain closed.

## 10. Related work and external grounding

### 10.1 Robust primitives and compact geometric structure

RANSAC established the general strategy of fitting geometric models despite
gross observation errors [1]. Efficient RANSAC later applied random sampling to
unorganized point clouds and decomposed them into planes, spheres, cylinders,
cones, tori, and residual points [2]. Sherman QC shares the premise that
primitive hypotheses can provide useful structure under noise, but it does not
equate a fitted primitive with a bend instance. Its ownership and duplicate-
suppression stages exist precisely because several valid local fits may describe
one physical transition.

Graph abstraction is another relevant lineage. Superpoint Graphs partition
large point clouds into homogeneous elements and encode their contextual
relationships in a compact attributed graph [3]. PointNet++ instead learns
hierarchical local features over nested metric-space neighborhoods and explicitly
addresses varying sampling density [4]. Our bounded patch/atom graph is not a
semantic-segmentation network, but it makes a related systems tradeoff: retain
local geometry and contextual adjacency while avoiding repeated processing or
serialization of the complete source cloud.

### 10.2 Edge and wireframe inference from point clouds

Recent work increasingly treats sharp features as structured outputs rather
than isolated edge-point classifications. EC-Net consolidates point sets with
an edge-aware learned objective [5]. PIE-NET proposes overcomplete edge/corner
regions and ranks parametric line, circle, and B-spline curve proposals [6].
PC2WF predicts vertices and their connectivity to reconstruct a wireframe from
raw point clouds [7]. NerVE uses a volumetric representation to extract
piecewise-linear and parametric curves [8], while SepicNet detects sharp-edge
points, groups them, and fits parametric curves to scanned shapes [9].

These systems motivate line/curve-instance evaluation and expose the difficulty
of converting dense or redundant edge evidence into unique physical features.
They are not direct baselines for the present claim. Several rely on supervised
CAD-derived sharp-edge labels, synthetic data, or general object-edge targets;
our target is a scan-observable sheet-metal bend axis under identity-free,
expected-count-free inference. We therefore compare principles and failure
modes rather than claiming equivalence between CAD sharp edges and physical
bends.

### 10.3 Evaluation under family structure and leakage risk

When observations are hierarchically related, random record-level splitting can
measure interpolation among related samples rather than transfer to a new
group. Roberts et al. review blocked cross-validation for hierarchical,
temporal, spatial, and phylogenetic structure [10]. Our family-held-out design
applies the same core principle to repeated captures and near-neighbor parts:
membership, model decisions, and evaluation openings must be separated at the
family level.

Kapoor and Narayanan document data leakage as a recurring source of optimistic
scientific claims and propose explicit model-information reporting to expose
leakage paths [11]. The present work operationalizes that concern through
hash-bound code/model/configuration artifacts, an opaque detector manifest,
process-level denial of truth and identity stores, append-only invocation
events, primary/replay execution before truth opening, and preservation of
invalid attempts. These controls exceed ordinary train/test splitting because
the detector is iteratively researched in a small-data industrial setting where
visual or operational feedback can itself become an answer channel.

### 10.4 Positioning

Our claimed novelty is not the invention of primitive fitting, point-cloud
graphs, proposal ranking, or grouped evaluation. It is the integration of a
deterministic, bounded, scan-only bend-line pipeline with an evaluation protocol
that treats geometric outputs, resource behavior, visual admissibility,
repeated-capture transfer, and information custody as simultaneous falsification
requirements. The narrow V51 parallel-return primitive remains an unconfirmed
project hypothesis and is not presented as established by prior research.

## References

1. M. A. Fischler and R. C. Bolles. “Random Sample Consensus: A Paradigm for
   Model Fitting with Applications to Image Analysis and Automated
   Cartography.” *Communications of the ACM* 24(6), 1981.
   https://doi.org/10.1145/358669.358692
2. R. Schnabel, R. Wahl, and R. Klein. “Efficient RANSAC for Point-Cloud Shape
   Detection.” *Computer Graphics Forum* 26(2), 2007.
   https://doi.org/10.1111/j.1467-8659.2007.01016.x
3. L. Landrieu and M. Simonovsky. “Large-Scale Point Cloud Semantic
   Segmentation with Superpoint Graphs.” *CVPR*, 2018.
   https://openaccess.thecvf.com/content_cvpr_2018/html/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.html
4. C. R. Qi, L. Yi, H. Su, and L. J. Guibas. “PointNet++: Deep Hierarchical
   Feature Learning on Point Sets in a Metric Space.” *NeurIPS*, 2017.
   https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html
5. L. Yu et al. “EC-Net: an Edge-Aware Point Set Consolidation Network.”
   *ECCV*, 2018.
   https://openaccess.thecvf.com/content_ECCV_2018/html/Lequan_Yu_EC-Net_an_Edge-aware_ECCV_2018_paper.html
6. X. Wang et al. “PIE-NET: Parametric Inference of Point Cloud Edges.”
   *NeurIPS*, 2020.
   https://proceedings.neurips.cc/paper_files/paper/2020/hash/e94550c93cd70fe748e6982b3439ad3b-Abstract.html
7. N. Liu et al. “PC2WF: 3D Wireframe Reconstruction from Raw Point Clouds.”
   *ICLR*, 2021. https://openreview.net/forum?id=8X2eaSZxTP
8. L. Zhu et al. “NerVE: Neural Volumetric Edges for Parametric Curve
   Extraction from Point Cloud.” *CVPR*, 2023.
   https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_NerVE_Neural_Volumetric_Edges_for_Parametric_Curve_Extraction_From_Point_CVPR_2023_paper.pdf
9. K. Cherenkova et al. “SepicNet: Sharp Edges Recovery by Parametric Inference
   of Curves in 3D Shapes.” *CVPR Workshops*, 2023.
   https://openaccess.thecvf.com/content/CVPR2023W/StruCo3D/papers/Cherenkova_SepicNet_Sharp_Edges_Recovery_by_Parametric_Inference_of_Curves_in_CVPRW_2023_paper.pdf
10. D. R. Roberts et al. “Cross-validation Strategies for Data with Temporal,
    Spatial, Hierarchical, or Phylogenetic Structure.” *Ecography* 40(8), 2017.
    https://doi.org/10.1111/ecog.02881
11. S. Kapoor and A. Narayanan. “Leakage and the Reproducibility Crisis in
    Machine-Learning-Based Science.” *Patterns* 4(9), 2023.
    https://doi.org/10.1016/j.patter.2023.100804

## 11. Conclusion

**Reserved until the locked evaluation is complete.** The only authorized
current conclusion is that the method and its falsification protocol are ready
for untouched confirmatory evaluation; performance and promotion remain open
questions.

## Internal evidence index (not publication references)

- `2026-08-14_gpt56_sol_pro_post_v51_protocol_invalidation_ruling_verbatim.md`
- `2026-08-14_f1_v51_new_untouched_corpus_custody_protocol.md`
- `2026-08-14_f1_v51_new_corpus_precommit_v1_invalidation.md`
- `2026-08-14_f1_v46_v51_prospective_action_cycle.md`
- `2026-08-14_f1_v51_post_prospective_professor_packet.md`
