EXECUTE_ONE

## V71 — `CANDIDATE_LOCAL_EXACT_CLOSURE_MULTISCALE_VARIATION_TRACK_V1` (`CLEC_MVT_V1`)

### 1. Architectural ruling

Implement one new, separately locked successor that evaluates the retained BRMVT multiscale field only on the exact source-vertex dependency closure needed by the frozen candidate-local track.

This is **not** a faster V70 run:

* The V70 executable, runner, caches, output directories, and real-scan route are never invoked.
* The scan-global field and scan-global hysteresis graph are removed.
* V71 defines a candidate-local field domain, candidate-local hysteresis, an exact closure proof, and a new abstention state for support that can leave the sparse domain.
* V70 remains terminal under its existing hashes.
* All V70 field, persistence, boundary, strong/weak, track, ownership, and ordinary-NMS numerical constants remain unchanged. No value is edited, relaxed, rescaled, or reselected.

The tested hypothesis is:

> A true candidate need only coincide with a persistent multiscale ridge inside the bounded region that its track can inspect. A candidate may be vetoed only when the sparse computation is mathematically sufficient to prove that the corresponding dense-field hysteresis support cannot enter that region from outside.

The resulting route is deletion-only. It creates no lines, moves no line, changes no endpoint, and changes no candidate score.

---

## 2. Insertion point and immutable parent block

Run the frozen upstream detector through the **V16 thresholded pre-NMS candidate boundary**. On U01–U04 this must reproduce the already frozen 48-row stream: 13, 20, 3, and 12 candidates.

V71 consumes only:

[
{\text{source vertices},\ \text{triangles/adjacency},
\text{candidate endpoints},\ \text{candidate direction},
\text{opaque source and candidate hashes}}.
]

Before V71 receives the records, serialize them through a whitelist that physically excludes:

* candidate score, confidence, rank, and route;
* filename, path, part identity, family, or provenance;
* expected count or any delivery count;
* CAD, labels, truth, benchmark result, or prior visual ruling;
* V69 or V70 decisions;
* candidate information belonging to another scan.

Process candidates in ascending opaque candidate-hash order. Restore original output association only by the opaque hash.

The following V70 mathematical block is copied byte-for-byte into a new V71 module and attested against the frozen V70 lock SHA-256

`f22e1a3c77d58b8683ea5411c422abd07103082f39d2e9d39835b4e24229e7b4`:

1. covariance surface variation at
   [
   K={24,64,160,256,512,1024};
   ]
2. the exact (N=1280) local percentile and relative top-hat calculations;
3. the exact two-scale persistence rule;
4. the exact angular and physical-mesh-boundary penalty;
5. all strong/weak comparators, including equality behavior;
6. the candidate-local partial longitudinal track predicate;
7. geometry-only duplicate ownership;
8. the single ordinary NMS.

A canonical semantic diff must report:

```text
changed_numeric_constants = []
changed_field_formulas = []
changed_comparators = []
changed_track_predicates = []
changed_ownership_predicates = []
changed_nms_rules = []
```

The only permitted semantic differences are:

```text
global_field_domain -> sparse_exact_dependency_domain
global_hysteresis   -> candidate_local_hysteresis
failed_local_track  -> veto_only_if_hysteresis_closed_else_abstain
global_resource_use -> frozen_sparse_caps
```

Any other difference is a pre-execution STOP.

---

## 3. Deterministic source representation

Use the frozen V70 source canonicalization and adjacency construction unchanged. In addition:

* All field arithmetic is binary64.
* NaN, infinity, zero-length candidate segments, invalid triangle indices, or fewer than 1,280 usable source vertices cause an atomic scan abort.
* Exact nearest-neighbor queries are required. Approximate search is forbidden.
* Neighbor order is increasing squared Euclidean distance, then increasing canonical vertex identifier.
* A single 1,280-neighbor query is used for a vertex needing percentile support; its first 1,024 entries are reused for all six covariance scales.
* A single 1,024-neighbor query is used for other closure vertices.
* Covariance accumulation order is the canonical neighbor order.
* Eigenvalues are ordered
  [
  \lambda_1\geq \lambda_2\geq \lambda_3\geq0.
  ]
* The retained surface-variation primitive is
  [
  \operatorname{SV}_k(v)
  ======================

  \frac{\lambda_3(C_k(v))}
  {\lambda_1(C_k(v))+\lambda_2(C_k(v))+\lambda_3(C_k(v))},
  ]
  with the exact V70 zero-trace behavior.

Candidate endpoint reversal, triangle winding reversal, source vertex and face permutation, rigid transformation, reflection, and positive uniform similarity must not change any categorical decision. New V71 distances are source-relative, so no fixed millimetre constant is introduced.

---

## 4. Candidate-local track domain

For candidate (c), let its endpoints be (a_c,b_c), and let

[
p_{c,j}=a_c+\frac{j}{8}(b_c-a_c),
\qquad j=0,\ldots,8.
]

For each (p_{c,j}), obtain the exact 64th-nearest source-vertex distance (r_{c,j}). Define

[
r_c=\operatorname{median}
{r_{c,0},\ldots,r_{c,8}},
]

where the median is the fifth value after ascending numeric sort. No interpolation or clamp is permitted. Equality is included in every domain comparison below.

Materialize (C_c), the **track core**, as the complete set of source vertices that can be read by the frozen V70 candidate-track geometry eligibility predicate under any possible assignment of field labels. Operationally, run that predicate with its field-label tests disabled, while retaining every geometric, axial, transverse, endpoint, and topology eligibility bound.

Thus (C_c) is a conservative field-independent superset of every vertex from which the unchanged track could construct a passing path.

Let (G) be the frozen source adjacency graph, with Euclidean source-edge lengths. Define truncated multi-source graph distance

[
d_G(v,C_c)=\min_{u\in C_c} d_G(v,u).
]

Construct:

[
I_c={v:d_G(v,C_c)\le r_c},
]

[
E_c={v:d_G(v,C_c)\le 2r_c}.
]

Here:

* (C_c) is the region inspected by the track;
* (I_c) is the protected interior halo;
* (E_c) is the complete sparse hysteresis domain;
* (E_c\setminus I_c) is the artificial boundary shell.

Dijkstra ties are resolved by canonical vertex identifier. No graph edge may be clipped, shortened, or replaced. Disconnected core pieces are all used as simultaneous seeds, preserving partial bends and dropout-separated observations.

If (C_c=\varnothing), (r_c\le0), or (r_c) is nonfinite, the candidate is retained as `ABSTAIN_DEGENERATE_LOCAL_DOMAIN`. It is not vetoed.

---

## 5. Exact sparse field closure

Define

[
Q_0=\bigcup_c E_c.
]

For every (q\in Q_0), obtain its exact 1,280-nearest source vertices. Define

[
Q_1
===

Q_0\ \cup
\bigcup_{q\in Q_0}N_{1280}(q).
]

Compute all six covariance surface-variation values for every vertex in (Q_1). Compute the percentile, relative top-hat, persistence, boundary penalty, and strong/weak labels only for vertices in (Q_0).

This is an exact dependency closure:

* (\operatorname{SV}*k(u)) depends only on the raw coordinates in (N*{1024}(u)), which remain available globally even when (u\notin Q_0).
* Every field value at (q\in Q_0) depends only on its own surface variation and on surface-variation values for vertices in (N_{1280}(q)).
* All such neighbor vertices are in (Q_1).
* Consequently, every scalar field value and every strong/weak label in (Q_0) must equal the corresponding value from a direct dense evaluation of the frozen formula.

The implementation must contain an independent small-fixture dense oracle. It must not call the V70 runner.

Global scanning of triangle incidence to identify actual mesh-boundary and nonmanifold edges is permitted. Global construction of the six-scale field is forbidden.

---

## 6. Candidate-local hysteresis and rigorous boundary behavior

For each candidate (c), form the induced graph (G_c=G[E_c]).

Classify its vertices with the unchanged frozen labels:

```text
STRONG
WEAK
OFF
```

Within (G_c), consider each connected component of the subgraph induced by `STRONG ∪ WEAK`.

A component is:

* **active** when it contains at least one `STRONG` vertex;
* **locally inactive and closed** when it contains no `STRONG` vertex, contains no vertex in (E_c\setminus I_c), and no vertex in the component has a source-graph neighbor outside (E_c);
* **externally unresolved** otherwise.

Run the unchanged V70 partial-track constructor using only active vertices in (C_c).

The candidate decision is exactly:

```text
if unchanged_track_predicate_passes:
    KEEP_TRACK
else if any externally_unresolved STRONG/WEAK component intersects C_c:
    KEEP_ABSTAIN_DOMAIN_EXIT
else:
    VETO_COMPLETE_NO_TRACK
```

This closure rule is the key scientific safeguard. A weak core component without a local strong seed could still become active in a dense global field only through a strong/weak path leaving the sparse domain. Such a component is never treated as negative; it forces abstention and retention.

A positive track remains valid even when its larger component touches the artificial boundary. Artificial-boundary contact can prevent a veto, but it can never invalidate an already observed local positive track.

### Morphology consequences

* **Rounded, shallow, and faceted folds:** use the unchanged multiscale response and persistence tests.
* **Partial bends and finite endpoints:** no full-span predicate is added; the unchanged partial-track rule remains controlling.
* **Dropout:** disconnected core pieces are evaluated independently; unresolved continuation outside the sparse domain retains rather than vetoes.
* **Returns, hems, and opposing skins:** field values may share computation, but hysteresis and track construction are separate per candidate and use source topology.
* **Nearby distinct folds:** no candidate can activate, suppress, or lend track vertices to another candidate.
* **Free edges, hole edges, and eroded limits:** the unchanged physical-boundary penalty remains active; an artificial sparse boundary is never represented as a physical mesh boundary.
* **Cylinder or broad smooth roll:** the unchanged local percentile contrast and persistence rules remain responsible for rejection.
* **Seams and isolated short features:** the unchanged track rule remains responsible for rejection.

---

## 7. Duplicate ownership and final NMS

After candidate validity:

1. Pass all `KEEP_TRACK` and all abstained candidates to the unchanged geometry-only duplicate-ownership rule.
2. `VETO_COMPLETE_NO_TRACK` candidates are removed.
3. Ownership ambiguity retains both candidates.
4. Invoke exactly one unchanged ordinary NMS.
5. No later merge, cap, ranker, count rule, or scan-level selection is allowed.

The V71 module must never deserialize or read a candidate score. The frozen NMS contract used here must receive only candidate geometry and opaque identifiers.

---

## 8. Hard resource and atomicity contract

The following equalities pass; cap plus one aborts.

| Quantity                            |              Hard cap |                |         |
| ----------------------------------- | --------------------: | -------------- | ------- |
| Source vertices                     |               200,000 |                |         |
| Source triangles                    |               350,000 |                |         |
| Input candidates                    |                   128 |                |         |
| Unique (Q_0) vertices               |                32,768 |                |         |
| Unique (Q_1) vertices               |                65,536 |                |         |
| Sum of per-candidate (              |                   E_c | ) before union | 262,144 |
| Candidate-local graph-edge visits   |             4,194,304 |                |         |
| Serialized audit, excluding renders |                 8 MiB |                |         |
| Algorithm-owned scratch             |                 4 GiB |                |         |
| Process peak RSS                    | strictly below 16 GiB |                |         |
| Field stage wall time               |             180.000 s |                |         |
| Complete scan wall time             |             240.000 s |                |         |

For (q\in Q_0), query 1,280 neighbors. For (q\in Q_1\setminus Q_0), query 1,024. Therefore the exact maximum retained neighbor-reference workload is

[
1280|Q_0|+1024(|Q_1|-|Q_0|).
]

Neighbor indices are streamed in blocks of 256 query vertices. Full neighbor tables are not serialized.

A cap breach, deadline, allocation failure, nonfinite result, missing dependency, hash mismatch, or deterministic replay mismatch causes:

```text
SCAN_ATOMIC_ABORT
modified_output_count = 0
ordinary_nms_invocation_count = 0
```

There is no candidate truncation, partial commit, fallback scale, approximate search, reduced (K), skipped candidate, or resumed execution.

The source vertex array may be mapped read-only once. It may not be copied into a compact cache or serialized into any result artifact.

---

## 9. Canonical audit artifact

The committed canonical JSON contains no source coordinates, filenames, identity, scores, or visual labels.

Scan-level fields:

```json
{
  "architecture": "V71_CLEC_MVT_V1",
  "source_sha256": "...",
  "candidate_stream_sha256": "...",
  "v70_parent_lock_sha256": "f22e1a3c77d58b8683ea5411c422abd07103082f39d2e9d39835b4e24229e7b4",
  "v71_lock_sha256": "...",
  "vertex_count": 0,
  "triangle_count": 0,
  "candidate_count": 0,
  "q0_count": 0,
  "q1_count": 0,
  "domain_membership_sum": 0,
  "graph_edge_visit_count": 0,
  "neighbor_reference_count": 0,
  "peak_rss_bytes": 0,
  "stage_wall_ns": {},
  "status": "COMPLETE|SCAN_ATOMIC_ABORT",
  "abort_reason": null,
  "output_sha256": "...",
  "replay_sha256": "..."
}
```

Per-candidate fields:

```json
{
  "candidate_hash": "...",
  "local_scale_r": 0.0,
  "track_core_count": 0,
  "interior_halo_count": 0,
  "outer_domain_count": 0,
  "strong_count": 0,
  "weak_count": 0,
  "active_component_count": 0,
  "externally_unresolved_component_count": 0,
  "track_summary": {},
  "decision": "KEEP_TRACK|KEEP_ABSTAIN_DOMAIN_EXIT|ABSTAIN_DEGENERATE_LOCAL_DOMAIN|VETO_COMPLETE_NO_TRACK|SUPPRESS_DUPLICATE",
  "reason": "...",
  "owner_candidate_hash": null
}
```

Also store SHA-256 hashes of the sorted (C_c), (I_c), (E_c), (Q_0), and (Q_1) identifier sequences, but not the sequences themselves.

---

## 10. Source-free one-shot fixture gate

Use the exact existing V70 24-fixture bytes unchanged. They remain mechanistic fixtures, not transfer evidence and not manufacturing-family evidence.

Required categorical results:

* shallow, rounded, faceted, hem, partial, dropout, opposing, nearby-fold, and sinusoidally displaced positives: all `KEEP_TRACK`, not abstention;
* flat, free-edge, hole-edge, jittered-edge, cylinder, seam, and isolated-short-feature negatives: all `VETO_COMPLETE_NO_TRACK`;
* duplicate fixture: exactly one geometry-owned survivor;
* crossing-feature fixture: both distinct features survive;
* all existing comparator and invariance results remain unchanged.

Add exactly these ten sparse-architecture fixtures:

| Fixture                                | Frozen expectation                                                                                                                                  |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `L01_DENSE_SPARSE_FIELD_EQUALITY`      | Every six-scale SV value, contrast value, strong/weak label, and track input in (Q_0) is bit-identical to the independent dense formula oracle.     |
| `L02_REMOTE_DISCONNECTED_GEOMETRY`     | Adding a disconnected source component at (1000r_c) changes no prior domain, field value, or decision.                                              |
| `L03_EXTERNAL_STRONG_SEED_PATH`        | A weak core component connected to a strong seed only outside (E_c) produces `KEEP_ABSTAIN_DOMAIN_EXIT`, never a veto.                              |
| `L04_CLOSED_WEAK_NO_SEED`              | The same weak component closed inside (I_c), with no strong vertex, permits `VETO_COMPLETE_NO_TRACK`.                                               |
| `L05_SHELL_COMPARATOR`                 | Graph distance exactly (r_c) is inside (I_c); `nextafter(r_c,+∞)` is in the shell. Exactly (2r_c) is inside (E_c); `nextafter(2r_c,+∞)` is outside. |
| `L06_OVERLAPPING_DOMAIN_ORDER`         | Candidate permutation changes neither (Q_0,Q_1), field bytes, nor decisions.                                                                        |
| `L07_ENDPOINT_REVERSAL_AND_REFLECTION` | Candidate reversal and reflected geometry preserve decisions and ownership.                                                                         |
| `L08_CAP_EQUALITY`                     | Exact (Q_0), (Q_1), membership, and edge-visit caps complete.                                                                                       |
| `L09_CAP_PLUS_ONE_ATOMICITY`           | Each independent cap-plus-one case atomically aborts with zero output and zero NMS calls.                                                           |
| `L10_200K_350K_128_STRESS`             | Completes under the frozen wall/RSS limits with byte-identical replay and no coordinate-bearing audit data.                                         |

The suite passes only at **34/34** with byte-identical primary/replay artifacts.

### Positive compatibility proof

No new predicate is conjoined with the positive track predicate. (C_c) is defined as a superset of every vertex the unchanged track can inspect, and the field on (C_c\subset Q_0) is exactly dependency-closed. Sparse-boundary uncertainty can only convert a proposed negative into retention; it cannot turn a passing track into a veto. Therefore every pre-existing positive is logically compatible with the simultaneous V71 rules. The unchanged positive fixtures must nevertheless demonstrate `KEEP_TRACK`, preventing a vacuous all-abstain implementation.

Any fixture failure terminally retires V71 before real geometry is opened.

---

## 11. Exposed-real mechanical actionability gate

Only after the source-free lock passes, run one primary and one replay on the already exposed U01–U04 source geometry and their exact frozen 48-row V16 candidate stream.

No visual material or truth is inspected while these mechanical gates are evaluated.

All of the following are required:

[
\text{complete executions}=8/8,
]

[
\text{byte-identical primary/replay}=4/4,
]

[
\max(\text{wall time})\le240.000\text{ s},
]

[
\max(\text{peak RSS})<8\text{ GiB},
]

[
|Q_1|/|V|\le0.55
\quad\text{for every scan},
]

[
\text{abstained candidates}\le4,
]

[
\text{decisive candidates}\ge44/48,
]

[
5\le
\text{V71 pre-NMS removals}
\le24,
]

and at least four removals must be direct `VETO_COMPLETE_NO_TRACK` decisions distributed across at least two opaque scans.

For any scan with at least four input candidates:

[
\text{V71 pre-NMS removal count}
\le
\left\lfloor 0.75,m_{\text{scan}}\right\rfloor .
]

No candidate geometry may change. No source-coordinate artifact may be committed.

These gates reject another near-zero mechanism, another mass-veto mechanism, an architecture that remains effectively global, or an architecture that survives only by abstaining.

A failure is terminal. There is no wider cap, threshold adjustment, retry, resumed child process, alternate sparse radius, or replacement exposed scan.

---

## 12. Raw-first native-resolution visual protocol

If and only if every mechanical gate passes, inspect U01–U04 in a four-stage reveal. The reveal order and opaque candidate order are fixed by hash before rendering.

### Stage A — raw source only

Show the complete native source vertices and triangles without candidates, field, decision, reason, filename, or part identity.

For every scan, inspect:

* the full object under continuous rotation;
* six orthographic views;
* two oblique views;
* native-resolution crops around every candidate region, still without candidate overlays.

The reviewer records visible interior crease corridors, rounded transitions, free/open boundaries, dropout regions, and visually ambiguous regions. Commit and hash this record before Stage B is revealed.

### Stage B — frozen input candidates

Overlay all V16 candidates identically in one neutral style. Show no V71 field or decisions.

For every candidate, record exactly one predecision class:

```text
VISIBLE_PERSISTENT_PHYSICAL_CREASE
NO_VISIBLE_PERSISTENT_PHYSICAL_CREASE
VISUALLY_AMBIGUOUS
```

Commit and hash this record.

### Stage C — raw field evidence

Reveal the unthresholded multiscale response, strong/weak vertices, artificial sparse-domain shell, and candidate track support. Do not reveal keep/veto/abstain status or reason codes.

Commit and hash the field-alignment review.

### Stage D — decisions and final output

Reveal V71 decisions, ownership, and the post-NMS output. Inspect every changed candidate and every abstention again at native resolution.

The visual gate passes only when:

1. every direct veto was classified in Stage B as
   `NO_VISIBLE_PERSISTENT_PHYSICAL_CREASE`;
2. no `VISIBLE_PERSISTENT_PHYSICAL_CREASE` or `VISUALLY_AMBIGUOUS` candidate was vetoed;
3. every duplicate suppression corresponds to the same visible crease as its retained owner;
4. at least four visually unsupported candidates across at least two scans were removed;
5. no decision is attributable to confusing the artificial sparse shell with a physical boundary;
6. every surviving marker is render-admissible and attached to source geometry;
7. no source-visible rounded, partial, finite-endpoint, return, opposing, or nearby distinct bend is lost.

The Stage A/B observations are visual falsification evidence only. They are never fed back into inference or used to alter a constant.

Any violation terminally retires V71.

---

## 13. Reserved physical scan

The reserved scan remains sealed until all source-free, exposed-real mechanical, and exposed-real raw-first visual gates pass unchanged.

Then open it exactly once under an opaque token:

1. open source geometry only;
2. generate the frozen V16 candidate stream;
3. run one V71 primary and one replay;
4. require completion, byte-identical replay, wall time (\le240) s, peak RSS (<8) GiB, and every sparse cap;
5. perform the same Stage A–D raw-first review;
6. freeze all artifacts and hashes before any truth or identity is opened.

After the output lock, its truth may be opened once for scoring. No rerun follows truth opening.

V71 remains eligible for the later locked benchmark only if, on this fresh scan:

[
\operatorname{Recall}*{V71}\ge
\operatorname{Recall}*{V16},
]

[
\operatorname{Precision}*{V71}\ge
\operatorname{Precision}*{V16},
]

and either

[
F1_{V71}>F1_{V16},
]

or

[
F1_{V71}=F1_{V16}=1.
]

No V16 true-positive match may be removed, and render-admissible marker accuracy must be 1.00 on this scan.

Passing the fresh scan is not promotion. Promotion still requires the unchanged locked benchmark of at least 30 scans and 15 families and every threshold stated in the request. Any failure terminally retires V71 without rescue.

---

## 14. Scientific basis and system-specific inference

Local covariance/PCA descriptors, multiple neighborhood scales, and hysteresis-connected line extraction are established point-surface feature machinery. Pauly, Keiser, and Gross explicitly combine local PCA, multiscale classification, and hysteresis for feature-line extraction; Pottmann and colleagues analyze local covariance/integral-neighborhood descriptors as scale-dependent, noise-robust geometric quantities; Demantké and colleagues likewise construct multiscale LiDAR descriptors from local covariance eigenvalues. 

The literature does **not** establish that a V16 candidate whose sparse BRMVT track passes is necessarily a physical sheet-metal bend. That remains the Sherman-specific hypothesis. The scientifically defensible contribution of V71 is narrower: its dependency closure makes the local scalar field exact, and its boundary rule prevents sparse truncation from manufacturing a negative certificate. The exposed-real and fresh one-shot gates then determine whether that exact local geometric signal is actionable on the physical scan domain.
