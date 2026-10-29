## 1. RULING

`RULING: EXECUTE_ONE`

Authorize exactly one frozen source-only execution of:

`V53_ADJACENCY_LATTICE_BANK_CONSENSUS_V1`
Short name: `V53_ALBC_V1`
Precommit token: `V53_ALBC_MECHANISM_LOCKED`

V53 is an **append-only selector** over the frozen V51 candidate pool. It must not alter candidate generation, V51 selection, V52 cylindrical-consensus recovery, any existing prediction, or any existing threshold.

The intervention tests one narrow hypothesis:

> A long, presently unrepresented candidate is safe to recover when the `mesh_adjacency` and `plane_lattice` routes agree not only on its axis, but on the same two source-mesh surface banks along substantially the full line.

This is preferable to `STOP_FAMILY` because the 0.91 candidate ceiling and V52’s +22 TP / 0 FP result leave a concrete, falsifiable selection hypothesis. It does **not** justify a general selection rewrite or a candidate-birth intervention.

---

## 2. Exact algorithm specification

### 2.1 Insertion point and immutability

Run ALBC after the complete V52 output list has been formed.

1. The frozen V51+V52 predictions form an immutable prefix.
2. ALBC may only append predictions.
3. It may not refit, move, lengthen, shorten, delete, reorder, or rescore any V51 or V52 prediction.
4. No second ordinary NMS is introduced.
5. ALBC uses the exact frozen V52 representation test—`7°`, `6 mm`, `0.5 overlap`—against the immutable prefix and against earlier ALBC additions.

### 2.2 Eligible clusters

Consider a frozen V51 candidate cluster only when all of the following hold:

* It is not represented by the V52 output under the frozen `7° / 6 mm / 0.5-overlap` test.
* It contains at least three **distinct frozen source-route names**.
* At least one member comes from `mesh_adjacency`.
* At least one member comes from `plane_lattice`.
* All relevant geometry is finite.

No candidate may be regenerated or admitted from outside the frozen pool.

### 2.3 Route-pair geometry agreement

For every `mesh_adjacency` member (a) and `plane_lattice` member (p) in an eligible cluster:

1. Use the cluster’s frozen canonical undirected axis as the projection direction.
2. Require undirected angular separation (\le 7^\circ).
3. Require shortest distance between the two infinite axes (\le 6) mm.
4. Project both finite candidate segments onto the canonical axis, producing intervals (I_a) and (I_p).
5. Let (I_c=I_a\cap I_p). Require:

[
\frac{|I_c|}{\min(|I_a|,|I_p|)} \ge 0.80
]

and

[
|I_c| \ge 0.50B,
]

where (B) is the exact frozen V52 bounding-box scale definition.

Thus, central-only agreement cannot qualify.

### 2.4 Two-bank ownership certificate

For each geometrically eligible pair, reconstruct in memory:

* (A_0,A_1): the two source-vertex regions owned by the `mesh_adjacency` hypothesis.
* (P_0,P_1): the two parent-plane source-vertex regions owned by the `plane_lattice` hypothesis.

The route’s existing frozen membership predicate must be used. No support threshold may be changed.

Rules:

* Convert support membership to packed bitsets over deterministic source-vertex ordinals.
* Restrict every set to vertices projecting inside (I_c).
* Within each hypothesis, remove any vertex assigned to both banks.
* If either route does not yield exactly two nonempty, disjoint banks, the pair abstains.
* If the existing route implementation cannot reconstruct its banks without a changed fitting rule, the pair abstains.

For two sets (X,Y), define the overlap coefficient:

[
O(X,Y)=\frac{|X\cap Y|}{\min(|X|,|Y|)}.
]

Evaluate both possible bank correspondences:

[
q_0=\min{O(A_0,P_0),O(A_1,P_1)},
]

[
q_1=\min{O(A_0,P_1),O(A_1,P_0)}.
]

Choose the correspondence with larger (q); exact ties use the unswapped correspondence. Require both matched-bank overlap coefficients to be at least `0.60`.

Divide (I_c) into eight equal axial bins, left-closed/right-open except for the final bin. For each of the two matched-bank intersections, require at least four distinct source vertices in at least seven of the eight bins.

A pair failing either bank’s overlap or axial coverage abstains.

### 2.5 Pair and candidate ordering

Within each cluster, choose one pair using this descending lexicographic tuple:

1. minimum of the two matched-bank overlap coefficients;
2. sum of the two matched-bank overlap coefficients;
3. minimum occupied-bin count across the two banks;
4. (|I_c|/B);
5. lower original `mesh_adjacency` hypothesis ordinal;
6. lower original `plane_lattice` hypothesis ordinal.

The proposed output is the cluster’s existing frozen canonical line, with endpoints clipped to the selected (I_c). There is no parameter refit or endpoint extrapolation.

Order qualifying clusters by the same first four certificate fields, descending, followed by the frozen cluster ordinal ascending. Process them in that order through the unchanged V52 representation test. Append at most one prediction per cluster.

### 2.6 Boundedness and exact abstention

Before reconstructing support sets, count all eligible adjacency–lattice pairs in the scan. If the count exceeds `4096`, ALBC emits no additions for that scan and records `PAIR_CAP_ABSTAIN`. It must not emit a partial prefix.

ALBC also emits the unchanged V52 result for the entire scan upon:

* malformed or non-finite source geometry;
* source-vertex ordinal mismatch;
* inability to reproduce a frozen route’s bank memberships deterministically;
* certificate serialization failure;
* any internal exception before the complete ALBC list is committed.

Candidate-level abstention occurs for:

* missing route pair;
* fewer than three distinct source routes;
* geometric disagreement;
* insufficient common span;
* missing or ambiguous two-bank ownership;
* bank overlap below `0.60`;
* either bank occupying fewer than seven of eight bins;
* representation by V52 or an earlier ALBC addition.

Only aggregate counts, overlap values, bin masks, route ordinals, and accepted line geometry may be serialized. Source vertices, point lists, face lists, and support bitsets remain transient.

---

## 3. Causal argument

V52 established that the candidate pool contains many useful long lines and that a conservative source-consensus selector can recover them without increasing false positives. The next justified question is therefore not “how many more candidates can be emitted,” but whether another class of candidates possesses a comparably strong source-only ownership certificate.

ALBC targets three observed residuals together without changing their generators:

* **Unselected complete bends:** adjacency and plane-lattice candidates already exist but were not V52-eligible.
* **Short central fragments:** a candidate must have at least `0.50B` common span and two-bank evidence in seven of eight bins.
* **Ownership-induced diagonals:** agreement on line geometry is insufficient; both routes must assign the line to substantially the same two source-mesh banks.

It is not a count quota. A cluster with a high score, a missing expected fold, or a visually attractive axis still produces nothing unless the two bank decompositions agree. It cannot repair the known candidate-birth gaps in SF02, SF10, or SF14, and it cannot remove the 27 inherited V52 false positives.

The adjacency and plane-lattice routes share the same reconstruction, so they are not statistically independent. That is the main uncertainty being tested. The zero-FP gate below is therefore binding.

---

## 4. Frozen one-shot success and failure gates

### 4.1 Execution protocol

Before invocation, freeze and hash:

* ALBC code and constants;
* exact bank-membership reconstruction functions;
* V51 candidate pools and cluster order;
* V52 output bytes;
* all 30 source meshes;
* renderer and scorer;
* primary/replay order;
* compact result schema.

Every artifact must retain:

`SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY`

Truth, scored overlays, family identity, capture pairing, and expected counts remain inaccessible until all primary and replay result hashes and source-only audit renders have been committed.

Execute exactly one primary pass and one deterministic replay for each of the 30 scans. No patched rerun, V53b, threshold change, or selective re-execution is authorized.

### 4.2 Immediate failure gates

Retire V53 ALBC if any of the following occurs:

* Any V51 or V52 prediction changes at the semantic or byte level.
* False positives exceed `27`.
* Any new diagonal, cross-panel, unrelated-panel, hole-rim, reconstruction-bridge, free-edge, seam, or short floating axis is found.
* Any of the six previously exact V52 scans ceases to be exact.
* Any primary/replay result or ALBC certificate differs.
* Fewer than `30/30` primary/replay pairs are semantically byte-identical.
* Truth or any forbidden identity signal is accessed during inference.
* A support bitset, source point, vertex list, or face list is serialized.
* Peak process-tree RSS is not below `16 GiB`.
* Total pipeline P95 is not below `300 seconds`.
* Any result is accepted after a scan-level abstention or exception.
* Any constant or ordering rule differs from the precommit.

### 4.3 Minimum positive retention gate

Retain ALBC as a synthetic-only engineering prototype only if **all** of the following hold:

* TP increases from `52` to at least `60`.
* FP remains exactly `27`.
* FN falls to at most `40`.
* Precision is at least `0.689655`.
* Recall is at least `0.600000`.
* F1 is at least `0.641711`.
* Exact-count cases increase from `6/30` to at least `7/30`.
* New true positives occur in at least three topology families.
* At least one topology family improves in both capture A and capture B.
* The number of zero-recall topology families decreases by at least one.
* Every ALBC addition is a scored TP; there may be no unscored or tolerated addition.

Failure of the positive gate retires `V53_ALBC_V1` even when it adds a smaller number of true positives without false positives. Such a result would be too narrow to justify another selector layer.

---

## 5. Mandatory visual signatures

Produce a 4096-pixel audit sheet for every scan receiving an ALBC addition, plus all six previously exact V52 scans and S011, S016, S024, and S029 regardless of whether ALBC fires.

Each accepted candidate needs:

1. orthogonal and oblique raw-mesh views with the proposed axis;
2. separately colored adjacency banks (A_0,A_1);
3. separately colored plane-lattice banks (P_0,P_1);
4. matched bank intersections with all eight axial bins marked;
5. the complete V52+ALBC overlay;
6. side-by-side A/B views when both captures receive a homologous recovery.

A valid addition must visibly show:

* one continuous physical fold aligned with the axis;
* the same two sheet surfaces under both ownership decompositions;
* support resuming on the same banks after holes or occlusion;
* no bank switching to an unrelated panel along the axis;
* no support concentrated solely around the line center;
* endpoints limited by genuine fold termination, crop, or occlusion rather than arbitrary interior truncation;
* no duplicate or near-parallel axis around an already represented fold.

Automatic visual failure occurs when:

* either colored bank crosses from one physical panel to another;
* the proposed line cuts diagonally through a visibly flat panel;
* agreement is driven by a hole rim, cap, seam, triangulation bridge, or boundary;
* the two route decompositions agree only on a small central island;
* an old diagonal in S016 or S029 acquires a new companion or apparent reinforcement;
* any previously exact scan receives an additional line.

Manual review is post-commit adjudication only. It may retain or retire the frozen run but may not alter its predictions.

---

## 6. Leakage audit

ALBC uses only:

* current source-mesh geometry and deterministic vertex ordinals;
* frozen triangle/region ownership already produced by the two candidate routes;
* frozen candidate lines, spans, clusters, and route provenance;
* source-derived bounding-box scale;
* globally precommitted constants.

Route provenance such as `mesh_adjacency` is algorithm provenance, not a filename, part number, family identifier, or expected-count signal.

It does not use:

* truth bends or truth renders;
* CAD, drawings, nominal geometry, or manufacturing metadata;
* filenames, scan IDs, topology-family identity, or capture A/B identity;
* known bend count or expected line locations;
* V51/V52 scoring outcomes on an individual scan;
* post-run benchmark feedback.

The topology-family spread, paired-capture, zero-recall-family, and exact-count gates are applied only after the frozen outputs have been committed. They never enter candidate ranking or acceptance.

Transient bank bitsets are bounded by the current source mesh and are discarded after the scan. Only certificate aggregates are persisted. Consequently, the mechanism remains source-only, part-identity-free, compact, and deterministic.

---

## 7. Research grounding

J-Linkage’s relevant contribution is the use of **shared model-preference sets** to distinguish multiple structures in noisy, outlier-contaminated data. ALBC borrows only that principle: a line is stronger when two structurally different routes assign the same source observations to it. It is not an implementation of J-Linkage. ([Springer][1])

PEARL shows why greedy “largest consensus first” selection can be inadequate in multi-model fitting: model parameters and point ownership are coupled, and spatially coherent assignment can matter as much as raw inlier count. ALBC is a deliberately bounded substitute for a full global relabeling—it tests ownership agreement for append-only candidates while leaving V51+V52 immutable. ([Springer][2])

Graph-Cut RANSAC similarly incorporates spatial coherence into robust inlier/outlier separation. This supports treating longitudinally coherent bank ownership as materially stronger evidence than two nearly coincident line parameters alone. ([CVPR Open Access][3])

Surface-reconstruction literature emphasizes that scanned meshes may contain noise, missing data, nonuniform sampling, and reconstruction-induced defects. That is why an adjacency candidate by itself should not be trusted and why ALBC requires agreement with a second ownership decomposition. ([Ben-Gurion University Research Portal][4])

The specific `0.60` bank-overlap threshold, `0.80` interval agreement, eight-bin certificate, seven-bin requirement, and adjacency–lattice pairing are **Sherman-specific preregistered hypotheses**, not established consequences of those papers. A pass would retain ALBC only as a synthetic engineering prototype; it would not establish physical-LiDAR generalization.

[1]: https://link.springer.com/chapter/10.1007/978-3-540-88682-2_41 "https://link.springer.com/chapter/10.1007/978-3-540-88682-2_41"
[2]: https://link.springer.com/article/10.1007/s11263-011-0474-7 "https://link.springer.com/article/10.1007/s11263-011-0474-7"
[3]: https://openaccess.thecvf.com/content_cvpr_2018/html/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.html"
[4]: https://cris.bgu.ac.il/en/publications/state-of-the-art-in-surface-reconstruction-from-point-clouds/ "https://cris.bgu.ac.il/en/publications/state-of-the-art-in-surface-reconstruction-from-point-clouds/"
