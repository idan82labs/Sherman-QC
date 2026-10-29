# 1. RULING

## **FORMAL STOP — `SCAN_SURFACE_CAUSAL_NONIDENTIFIABILITY_STOP_V21`**

**No V21 source-only bend detector is authorized.** V16 remains retained and frozen. PNCT and every previously stopped family remain closed. U02, C017, the locked 35, and U02–U05 must not be opened for another surface-only architecture.

This is not a claim that no geometric heuristic could accidentally improve the locked-35 F1. It is a narrower and stronger ruling: **under the frozen target definition, no remaining scan-surface observable can be causally certified to distinguish a true bend from all required aliases.** Continuing would replace an identifiable mechanism with an untestable distributional prior.

The only bounded source-only action authorized is a one-shot proof artifact:

**`SOURCE_OBSERVATION_EQUIVALENCE_WITNESS_V1`**

It records the non-identifiability and then closes the source-only research lane. It is not V21, does not emit production candidates, and does not open any real scan.

The smallest capability that could justify reopening is:

**a co-registered, candidate-local through-thickness ultrasonic B-scan channel**, acquired at three fixed axial stations. A second optical view, denser point cloud, another normal estimator, or a different cross-station transport is insufficient because each remains a measurement of surface geometry.

---

# 2. RESEARCH BASIS

## Established published machinery

Normal-cycle estimators, polynomial surface fitting, and integral invariants can estimate curvature, principal directions, ridges, and other differential surface quantities from sampled or noisy geometry. Integral invariants specifically improve the stability of surface-shape estimates by averaging over neighborhoods, while normal-cycle methods define curvature tensors directly for sampled or polyhedral surfaces. 

Point-cloud edge and corner detectors can also localize seam-like geometric structures; this machinery has been applied directly to robotic weld-seam extraction. That establishes that a weld can be *geometrically localized when it has a visible geometric signature*. It does not establish that point geometry identifies whether an otherwise identical transition was bent, welded, filled, ground, or formed. ([arXiv][1])

The missing information is experimentally measurable with volumetric NDE. Gilmour et al. studied a steel weld whose cap had been ground and polished until no discernible surface feature remained. Phased-array ultrasound still localized the weld from nonsurface volumetric signal and weld-versus-parent microstructure, reporting 0.8 mm mean positional error in their experiment. ([Strathprints][2])

## Sherman-specific conclusion

PNCT established that constant-offset thickness transport is physically appropriate but is not reliably present in the permitted source observation. Once paired-skin evidence is absent, all proposed one-sided quantities reduce to functions of the observed boundary:

[
n(p),\quad k_1(p),\quad k_2(p),\quad K(p),\quad H(p),
]

Gauss-map arcs, normal rotation, developability, geodesic continuation, tail planarity, strip ownership, boundary distance, cross-station consensus, or combinations of them.

Those may identify a **fold-shaped surface transition**. They do not identify **material provenance or through-thickness connectivity** when two constructions have the same observed surface.

That distinction is the Sherman-specific obstruction.

---

# 3. CAUSAL FEATURE

There is no qualifying source-only causal feature under the frozen alias requirement.

Let (H) denote the physical part and let

[
\mathcal O(H)=X
]

be the complete permitted worker observation: source coordinates, faces, adjacency, normals, deterministic source features, deletion pattern, and all fixed sidecars.

Any authorized detector must be a deterministic function

[
D:X\rightarrow {\text{zero axes},\text{one or more axes}}.
]

Now construct:

* (H_B): a monolithic sheet containing a true bend;
* (H_W): two joined sheets with a full-penetration weld or formed joint, machined or ground so that the complete observed boundary is the same as (H_B).

The same triangulation, coordinates, normals, connectivity, sampling, fragmentation, and emitted source features can be used for both:

[
\mathcal O(H_B)=\mathcal O(H_W)=X.
]

But the frozen target requires

[
Y(H_B)=1,\qquad Y(H_W)=0.
]

Therefore,

[
D(\mathcal O(H_B))=D(X)=D(\mathcal O(H_W)),
]

so (D) must be wrong on at least one of the two objects.

This obstruction is unaffected by:

* better curvature estimators;
* one-sided Gauss-map or normal-rotation analysis;
* axial dynamic programming;
* cross-station transport;
* persistence or topology;
* global ranking;
* a different deterministic NMS rule.

They are all functions of (X).

Some individual aliases remain geometrically rejectable:

* A visible opening or free rim can expose a boundary.
* A complete cylinder can expose cycle closure.
* A disconnected nearby panel can fail material-surface continuity.
* Duplicate lines can be handled by NMS.

But solving those aliases does not remove the welded/formed-joint equivalence. One unresolved exact equivalence is enough to invalidate a universal causal claim.

Cropping and fragmentation also create weaker versions of the same issue for terminal caps and opening transitions: a bounded observed neighborhood can be congruent to an interior fold while the distinguishing topology lies outside the observation. Cross-station aggregation cannot reconstruct topology that was never measured.

---

# 4. EXACT SPECIFICATION

## `SOURCE_OBSERVATION_EQUIVALENCE_WITNESS_V1`

### Source-only inputs

Use one immutable source mesh buffer (X), preferably the already validated clean standard-fold positive from the PNCT fixture corpus. No new geometry generator is needed.

The harness creates two fixture records:

* **NI01 — MONOLITHIC_BEND:** required output exactly one axis.
* **NI02 — GROUND_FULL_PENETRATION_JOINT:** required output zero axes.

The worker input for NI01 and NI02 must be the **same byte slice**. Fixture role, physical-construction description, and expected count remain in the external test harness and are never passed to the worker.

All existing deterministic preprocessing is run once and the same resulting feature buffers are shared by both calls.

### Eligibility and equations

Every source-only detector candidate is eligible for the witness. No geometric threshold is involved.

The hard condition is:

[
\operatorname{bytes}(X_{\mathrm{NI01}})
=======================================

\operatorname{bytes}(X_{\mathrm{NI02}}).
]

Consequently, for deterministic worker (D),

[
\operatorname{bytes}(D(X_{\mathrm{NI01}}))
==========================================

\operatorname{bytes}(D(X_{\mathrm{NI02}})).
]

The required semantic outputs are nevertheless:

[
#D(X_{\mathrm{NI01}})=1,\qquad
#D(X_{\mathrm{NI02}})=0.
]

No detector can satisfy both.

### Deterministic ties

No tie rule can affect the result because every candidate score, quantized coordinate, component order, seed order, and NMS input is identical.

If a proposed worker returns different outputs, it has consumed fixture identity, uninitialized state, nondeterminism, or another prohibited channel.

### Fixed serialization

The witness stores:

1. one copy of the existing source fixture;
2. two fixed harness records containing only an expectation and a reference to that same fixture;
3. two worker-output byte buffers for equality comparison.

There is no additional point-cloud, vertex-list, or face-list cache. The mesh is not duplicated.

### Work, memory, and latency

The additional work is one repeated worker invocation and one linear byte comparison:

[
O(|X|+|D(X)|).
]

Hard bounds:

* witness wall time: less than 1 second beyond two worker calls;
* added peak RSS: less than 64 MiB;
* output artifact: less than 1 MiB excluding the already existing fixture;
* exactly one execution;
* no rerun after failure.

### Insertion point

The witness runs **before any V21 detector implementation**, before U02, and before C017.

A valid contradiction closes the source-only lane immediately.

### NMS behavior

NMS is irrelevant to the proof. Whatever NMS output NI01 receives, NI02 receives identically. NMS cannot convert one identical input into both one axis and zero axes.

### Prohibited information

The worker must not receive:

* NI01/NI02 role;
* fixture name;
* construction method;
* expected count;
* filename or path identity;
* gold or labels;
* CAD or drawing;
* material or process record;
* benchmark outcome;
* acquisition ordering;
* production feedback.

## Smallest capability opening

The minimum new observable is:

### **`CO_REGISTERED_THROUGH_THICKNESS_ULTRASONIC_BSCAN`**

For each source-generated candidate strip, acquire B-scans in planes perpendicular to the candidate axis at support quantiles (0.25), (0.50), and (0.75). The measurement must expose through-thickness reflectivity or microstructural response, not merely another estimate of the exterior surface.

The opening remains bounded:

* candidate generation and ranking remain source-only;
* maximum 16 candidate strips per scan;
* three fixed stations per strip;
* station ties use the lowest quantized axial coordinate;
* no candidate may be requested from gold or expected counts;
* only candidate-local ultrasonic tiles are retained;
* no full volumetric reconstruction is required.

The decisive capability fixture must pair:

* a monolithic bend;
* a ground full-penetration weld or formed joint;

with byte-identical optical surface meshes but physically different through-thickness scans. This directly measures the variable that the surface observation lacks. The feasibility is supported by experimental PAUT work in which a ground weld with no discernible surface cue was localized from volumetric and microstructural response. ([Strathprints][2])

If representative monolithic and ground-weld coupons cannot be separated by that channel under a single preregistered rule, the capability opening also stops. No surface-only fallback is then permitted.

---

# 5. FAST FALSIFICATION GATES

## Source-only STOP witness

The complete gate is one positive/negative pair.

### NI01 positive

* Worker input: (X).
* Physical interpretation: monolithic bend.
* Required: exactly one axis.

### NI02 negative

* Worker input: the same (X), byte for byte.
* Physical interpretation: ground welded or formed joint.
* Required: zero axes.

### Hard decision

* One axis on both: NI02 fails.
* Zero axes on both: NI01 fails.
* Any identical nonzero count on both: at least one fails.
* Different outputs: determinism, replay, or prohibited-information failure.

No threshold is exposed and no rescue exists.

## Metamorphic identity

Reuse the frozen PNCT metamorphic corpus verbatim:

* vertex/face permutation;
* seed reversal;
* axis-sign reversal;
* each already frozen rigid transform.

For every transformed instance (T(X)), the positive and negative worker-input buffers must again be identical, and their complete outputs must be byte-identical.

The contradiction must survive every transform. Failure of byte identity is not evidence for identifiability; it is a deterministic implementation failure.

## Replay

Run the complete witness twice in the same isolated environment.

Required:

* identical output bytes;
* identical decision code;
* identical serialized evidence;
* no dependence on allocation order or thread scheduling.

## Mandatory 4096px visual audit

Render NI01 and NI02 independently at native 4096 resolution with exactly the same:

* camera;
* source mesh;
* normals;
* candidate bands;
* curvature or chord overlays;
* line results.

The pixel XOR of the two surface panels must be zero. Captions identifying the physical construction are added only after rendering by the harness.

The manual ruling must state: **no visible surface evidence distinguishes the required positive and negative.**

## Capability-opening preflight

Only after explicitly opening ultrasonic sensing:

* monolithic 90-degree bend: exactly one;
* monolithic noncircular (C^2) bend: exactly one;
* ground full-penetration weld with matched exterior: zero;
* formed or multilayer seam with matched exterior: zero;
* flat panel, rim, opening, cap, cycle and unrelated-panel controls: zero;
* replay and frozen rigid transforms: full-byte identity;
* 4096px composite showing surface, raw B-scan, and accepted/rejected through-thickness evidence.

The matched bend/weld pair is the first gate. Failure there stops before the remaining fixtures.

---

# 6. REAL-SCAN GATES

Under the present ruling, **no real-scan execution is authorized**. U02, C017, the locked 35 and held-out scans remain closed.

If the through-thickness capability is explicitly opened and passes its source-free physical-coupon gate, the real sequence is:

### Gate 1 — U02 label-closed no-op

* Run without opening labels.
* Exact V16 marker bytes must remain unchanged.
* No ultrasonic route activation.
* Wall time, cache, and replay must pass.

Any changed V16 byte or new marker stops the route.

### Gate 2 — C017 exactly one activation

* Exactly one additional axis.
* No V16 marker may move, disappear, or change bytes.
* Manual native-4096 review must inspect both the surface geometry and the three through-thickness slices.
* Review must confirm that the activation is supported by one continuous monolithic sheet response and not by a joint, multilayer seam, terminal wall, opening, or duplicate.

Zero or more than one activation is terminal. A visually ambiguous through-thickness result is terminal.

### Gate 3 — frozen locked 35

For the first bounded tranche, retain only if:

* TP at least 133;
* FP no more than 60;
* FN no more than 43;
* F1 at least (266/369 = 0.720867);
* exact-count scans at least 5/35;
* no existing V16 marker changes;
* no family recall decreases;
* no new family falls below its V16 result;
* deterministic replay remains 100%;
* process-tree RSS remains below 16 GiB;
* P95 remains at most 300 seconds.

This requires at least one recovered TP and zero added FP. A capability-opening experiment should not be retained merely for a fractional F1 gain bought with a new false-positive mode.

### Gate 4 — held-out U02–U05

Required unchanged result:

* 15 TP / 0 FP / 0 FN;
* zero additional markers;
* byte-identical V16 replay on all four scans.

Any held-out activation or byte change rejects the route.

---

# 7. EXPECTED FAILURE MODE

The single terminal signature is:

> **Two fixtures have identical permitted source bytes, identical derived features, identical 4096px renders, and identical worker outputs, while the frozen target requires one axis for one fixture and zero for the other.**

Numerically:

[
X_{\mathrm{bend}}=X_{\mathrm{joint}},
]

[
D(X_{\mathrm{bend}})=D(X_{\mathrm{joint}}),
]

but

[
Y_{\mathrm{bend}}\neq Y_{\mathrm{joint}}.
]

That signature does not indicate a poor threshold, insufficient coverage, bad canonicalization, or an implementation bug. It demonstrates that the target variable is outside the source observation.

Accordingly, **the source-only research direction is stopped at V20**. The next scientifically valid packet requires a through-thickness material-continuity measurement; without that opening, another detector architecture would be an unsupported heuristic rather than a causally identifiable bend representation.

[1]: https://arxiv.org/abs/1809.10468?utm_source=chatgpt.com "Edge and Corner Detection for Unorganized 3D Point Clouds with Application to Robotic Welding"
[2]: https://strathprints.strath.ac.uk/87157/7/Gilmour-etal-IEEE-OJIC-2023-Using-phased-array-ultrasound-to-localize-probes.pdf "Using Phased Array Ultrasound to Localize Probes During the Inspection of Welds"
