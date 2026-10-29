## 1. **RULING**

Authorize exactly one new intervention:

**`PROJECTIVE_NORMAL_CHORD_TRANSPORT_V1` — V20 / PNCT_V1**

PNCT_V1 tests whether an eligible native-mesh adjacency axis supports a **single, axially coherent field of approximately constant-thickness chords transported through a continuous interval of unoriented surface-normal directions**.

This is a genuinely different representation:

* no H0 filtration, union-find, ports, components, persistence, or topology fingerprint;
* no PTCP canonicalization or hash machinery;
* no exact face mate, reciprocal endpoint, reciprocal center, or ownership identity from PSRO;
* no circle, focal-center, cylinder, or tangent-profile fit;
* no source-point or face-list serialization.

The tranche is terminal on any failed gate. No V20b, threshold adjustment, changed fixture, or rerun is authorized.

---

## 2. **RESEARCH BASIS**

### Established published machinery

Shapira, Shamir, and Cohen-Or’s Shape Diameter Function established a volume-based surface descriptor in which local thickness is estimated by probing toward the opposite boundary and robustly aggregating resulting chord lengths. PNCT adopts only the local-diameter principle; it does not adopt their segmentation, smoothing, learned grouping, or inward-orientation assumptions. ([Springer Link][1])

Classical and discrete offset-surface theory treats two surfaces at constant normal distance as parallel offsets. Karpenkov and Wallner explicitly develop curvature theory from parallel surfaces and constant-distance offsets, including the Steiner area relation for offsets. PNCT uses the more elementary consequence relevant here: corresponding traces of two offset skins have parallel tangent lines and are separated along their common normal. ([Geometrie][2])

Pottmann et al. show why neighborhood-integrated geometric quantities are preferable to pointwise differential estimates on noisy geometry: integration supplies stability against noise and minor perturbations. PNCT accordingly aggregates segment length, chord distance, and midpoint transport over bins and stations rather than estimating vertex curvature or differentiating normals. ([UT Austin Computer Science][3])

### Sherman-specific hypothesis

The following is new and is **not** a published bend detector:

> Among already-eligible adjacency axes, a physical constant-thickness fold is characterized by one continuous projective-normal interval in which native cross-sections repeatedly contain two locally parallel surface traces separated by one short, nearly constant normal chord, while the chord midpoints sweep a noncollapsed profile along the fold.

The hypothesis is falsified quickly if:

1. the fragmented synthetic positives do not retain that field;
2. any named alias produces the same unique field;
3. C017 has no source-only activation;
4. U02 activates;
5. a visually obvious cap, rim, seam, cycle, or nearby panel passes.

---

## 3. **CAUSAL FEATURE**

At a station perpendicular to a true bend axis, the two sheet skins appear as two offset planar curves. At a given surface-normal direction, the two curves should provide:

* approximately parallel local tangents;
* a displacement approximately normal to both tangents;
* approximately one material-thickness separation;
* a chord midpoint that moves along the bend profile as normal direction rotates.

Normals are treated projectively, meaning `n` and `-n` are the same direction. Triangle winding and isolated face-normal flips therefore do not alter the observable.

### Alias rejection

**Silhouette or free edge.**
There is only one supported surface trace. Normal bins have no short opposite chord, so chord coverage fails.

**Opening rim.**
A sharp rim supplies isolated planar and wall-normal modes rather than a populated normal interval. A rounded terminal rim can supply apparent diametric chords, but their midpoints collapse near the cap center instead of sweeping a material midsurface. It fails midpoint transport.

**Thickness wall.**
The wall supplies one additional normal mode between two planar skins. It does not supply at least seven populated interior normal bins with one constant chord field.

**Welded seam.**
The two panels terminate independently. A miter leaves the intermediate normal bins empty; a weld bead creates variable chord lengths, extra partners, or a nonparallel hit.

**Formed or overlapping corner seam.**
The overlap introduces more than two traces for at least part of the normal interval. It therefore produces a second short partner, elevated ambiguity, complement-arc support, or axial profile branching.

**Cycle-closing boundary.**
A closed corner or box cycle tends to support both complementary projective-normal arcs or substantial valid support outside the selected arc. PNCT requires exactly one arc to pass.

**Nearby panel.**
An unrelated panel may provide a short hit locally, but its displacement is not consistently normal to both traces, its separation changes with station or normal phase, or its midpoint profile is axially incoherent.

**Duplicated axis.**
PNCT introduces no independent duplicate rule. Activated candidates enter the frozen V16 duplicate canonicalizer with retained V16 lines preseeded and immutable.

### Identifiability boundary

A seam whose **entire observed geometry** is exactly the same two-sided constant-offset surface as a formed bend is not distinguishable by scan geometry alone. If such a seam passes the source-free or native real-scan audit, PNCT is falsified and the correct next ruling is source-only STOP. The smallest scientifically relevant opening would then be a material-sensitive channel, visible seam/bead texture, or an additional view exposing the hidden termination—not labels, filenames, or expected counts.

---

## 4. **EXACT SPECIFICATION**

### 4.1 Eligible source-only pool and insertion point

PNCT reads the frozen V16 native full-resolution candidate ledger at the point:

1. after full-mesh adjacency construction;
2. after V16 route correction;
3. after all existing line-support, angle, station, and basic geometry prerequisites;
4. before rejected candidate geometry is discarded.

A candidate is eligible exactly when:

```text
V16_PRESELECTOR_PASS = true
V16_RETAINED = false
REJECTION_STAGE is at or after the final ownership/alias selector
support_length L >= 12s
both frozen incident seed planes are present
both seed normals project non-degenerately into the plane perpendicular to the axis
```

Here `s` is the already-emitted V16 source scale. PNCT does not recompute or alter it.

Candidates rejected by earlier geometry, support, station, route, or render prerequisites are not eligible. V16-retained lines are never rescored, removed, shifted, or replaced.

An activated candidate is inserted as a supplemental full-mesh candidate immediately before the unchanged V16 duplicate and render-admissibility stages. It inherits the upstream axis, support interval, and marker geometry; PNCT estimates no new line.

### 4.2 Candidate-local cross-sections

For candidate axis line `ell=(o,a)`, with `a` unit length but otherwise unoriented, support length `L`, and scale `s`, use exactly 11 station planes:

```text
P_j passes through o + ((j - 5) * L / 12) a
j = 0,...,10
P_j is perpendicular to a
```

Thus the stations occupy the central `10/12` of the supported axis interval.

For each station:

* query the existing native-mesh BVH;
* consider triangles intersecting the station plane and the radius-`36s` cylinder about `ell`;
* compute exact triangle-plane intersection segments;
* discard intersection segments shorter than `0.10s`;
* ignore triangles coplanar with the station plane;
* treat a signed vertex-plane distance with `abs(distance) <= 2^-36 s` as zero;
* deduplicate coincident intersection endpoints after quantization to `s/4096`.

The transient hard bound is:

```text
maximum intersection segments per station = 8192
```

Any overflow terminates the whole V20 tranche. It is not converted into a candidate rejection and no segments are truncated.

For each segment `i`, retain transiently:

```text
midpoint m_i
unoriented unit tangent line [t_i]
unoriented in-plane normal line [r_i], where r_i = a cross t_i
segment length lambda_i
canonical geometry key quantized at s/4096
```

No segment, vertex, triangle, or coordinate array is serialized.

### 4.3 Projective-normal arcs

Project the two frozen seed normals into `a`-perpendicular space and treat each as an unoriented line `[q_0]`, `[q_1]`.

Those two lines divide the projective normal circle, whose circumference is `pi`, into exactly two complementary arcs. These are not two parameter alternatives or a sweep; they are the two unavoidable representations of an unoriented interval.

For each arc:

* its angular length must be between `25 degrees` and `155 degrees`, inclusive;
* divide it into exactly 12 equal phase bins;
* bins `0` and `11` are the two tail bins;
* bins `1` through `10` are the rotation-interior bins.

Each cross-section segment is assigned to the unique bin containing its projective normal direction. Endpoint equality goes to the lower-index bin.

Both complementary arcs are evaluated. **Exactly one must pass the complete predicate.** Zero passing arcs rejects the candidate; two passing arcs rejects it as closed/cyclic or nonselective.

### 4.4 Normal-chord construction

Within each station and phase bin, define the representative bin normal `r_b` and tangent `t_b`.

For every origin segment `i`, partner candidates are segments in the same station and phase bin satisfying:

```text
abs((m_k - m_i) dot t_b) <= 2s
```

Partner candidates are ordered by absolute tangential displacement and then by canonical segment key. Examine at most the first 32. If more than 32 lie inside the `2s` tangential window, terminate the tranche with `PARTNER_DENSITY_OVERFLOW`; do not truncate.

Let:

```text
Delta = m_k - m_i
d = norm(Delta)
```

A partner is valid exactly when all conditions hold:

```text
1.5s <= d <= 16s

abs(t_i dot t_k) >= cos(20 degrees)

abs(Delta dot t_i) <= sin(20 degrees) d

abs(Delta dot t_k) <= sin(20 degrees) d

abs(Delta dot t_b) <= min(2s, sin(20 degrees) d)
```

For each origin segment, select the valid partner having minimum `d`; ties use canonical geometry key. No reverse-nearest or reciprocal-face requirement exists.

The origin is marked ambiguous when a second valid partner has:

```text
d_second <= 1.25 d_first
```

The chord midpoint is:

```text
h_i = (m_i + m_k) / 2
```

The integer aggregation weight is:

```text
w_i = round_ties_to_even(4096 lambda_i / s)
```

### 4.5 Bin predicates

For station `j`, arc bin `b`, define:

```text
W_raw   = sum w_i over all origin segments
W_valid = sum w_i over origins having a valid chord
W_amb   = sum w_i over ambiguous valid origins
N_valid = number of valid origins
```

A bin is **owned** exactly when:

```text
W_raw corresponds to at least 0.75s of segment length
N_valid >= 2
W_valid / W_raw >= 0.55
W_amb / W_valid <= 0.20
```

Its thickness is the deterministic weighted median of valid `d_i`. Its midpoint is the componentwise deterministic weighted median of valid `h_i`. Weighted-median ties select the lower quantized value and then the lower canonical key.

### 4.6 Station predicates

A station is provisionally complete exactly when:

```text
tail bin 0 is owned
tail bin 11 is owned
at least 7 of the 10 interior bins are owned
there are no two consecutive missing interior bins
sum(W_valid) / sum(W_raw) over the 12 bins >= 0.60
```

For an isolated missing interior bin, interpolate its midpoint linearly from its two adjacent owned bins. No interpolation is permitted across two missing bins.

For the resulting ordered midpoint sequence, define:

```text
P_j = sum of consecutive midpoint step lengths
D_j = distance between first and last midpoint
S_j = maximum individual midpoint step
```

After global thickness `tau` is known, a provisionally complete station finally passes only if:

```text
P_j >= 1.00 tau
D_j >= 0.65 tau
S_j <= 0.40 P_j
```

This is the terminal-cap rejection: a round cap may have stable diametric chords, but their midpoints remain collapsed and produce small `P_j` and `D_j`.

### 4.7 Candidate retain predicate

For one projective-normal arc, pool every valid chord from provisionally complete stations and compute weighted median thickness `tau`.

That arc passes exactly when all conditions hold:

```text
1.5s <= tau <= 16s

at least 8 of 11 stations finally pass

each tail bin is owned in at least 9 of 11 stations

the tail-bin thickness medians differ from tau by at most 0.20 tau

weighted 90th percentile of abs(d_i - tau) / tau <= 0.20

90th percentile of abs(station_median_j - tau) / tau <= 0.15

global ambiguous valid weight / global valid weight <= 0.15

valid chord weight on the complementary arc /
valid chord weight on this arc <= 0.15

median(P_j / tau) over passing stations >= 1.25
```

For axial profile coherence, express each station midpoint profile relative to its station-axis point and in the seed-normal cross-sectional frame. Compare each available phase-bin midpoint with the componentwise median profile over passing stations. The weighted RMS residual must satisfy:

```text
axial_profile_RMS <= 0.35 tau
```

The candidate activates only when **exactly one** complementary arc passes all these conditions.

### 4.8 Ranking, NMS, and duplicate behavior

PNCT uses no learned or weighted score. Activated candidates are ordered lexicographically by:

```text
1. descending number of passing stations
2. descending number of owned interior station-bin cells
3. ascending thickness 90th-percentile deviation
4. ascending axial profile RMS / tau
5. ascending ambiguity fraction
6. ascending frozen upstream candidate key
```

V16-retained lines are inserted ahead of every PNCT line and always win a duplicate conflict.

No new geometric duplicate threshold is introduced. The ordered PNCT candidates enter the **unchanged frozen V16 duplicate predicate and canonicalizer**. Any PNCT candidate duplicating a retained V16 marker is a no-op.

### 4.9 Fixed serialization

One 64-byte little-endian scan header is followed by one 192-byte record per eligible candidate, sorted by the frozen upstream candidate key.

The 192-byte record is:

```text
0..3      magic "PNC1"
4..5      uint16 version = 1
6..7      uint16 flags
8..23     frozen upstream 128-bit candidate key
24..31    uint64 predicate/failure mask
32..47    eight uint16 global quantized metrics
48..59    twelve uint8 selected-arc median coverage values
60..71    twelve uint8 owned-station counts
72..95    twelve uint16 selected-arc bin thickness medians
96..117   eleven uint16 station thickness medians
118..128  eleven uint8 station flags
129..139  eleven uint8 station coverage values
140..151  twelve uint8 complementary-arc coverage values
152..183  eight uint32 structural counters
184..191  zero-filled reserved bytes
```

Quantization is frozen as follows:

```text
normalized lengths: round_ties_to_even(4096 x / s), clipped to uint16
fractions: round_ties_to_even(65535 x), clipped to uint16
uint8 coverage: round_ties_to_even(255 x)
undefined uint16: 0xFFFF
undefined uint8: 0xFF
```

Per-bin sequences are serialized as the lexicographically smaller of the forward and reversed quantized sequence. This handles axis-direction and seed-order reversal; it is not a topology canonicalization.

The cache contains no source point, mesh vertex, face index, intersection segment, chord endpoint, or source coordinate.

### 4.10 Work, memory, and latency bounds

Per eligible candidate:

```text
11 station planes
8192 maximum transient segments per station
32 maximum partner examinations per origin segment
2 complementary projective arcs
12 phase bins per arc
```

Station buffers are reused. Incremental transient memory must remain below `2 MiB` per active worker. The persistent cache is:

```text
64 + 192 * eligible_candidate_count bytes
```

There is no candidate truncation or scan-level selection cap.

Hard execution gates are:

```text
incremental PNCT P95 <= 60 seconds
combined scan P95 <= 300 seconds
incremental process-tree RSS <= 512 MiB
combined process-tree RSS < 16 GiB
zero structural overflows
zero worker failures
```

Any timeout, segment overflow, partner-density overflow, NaN, undefined frame, or incomplete record terminates V20.

---

## 5. **FALSIFICATION GATES**

### 5.1 Mandatory source-free positives

All positives must emit exactly one activated axis.

**SF01 — standard fold**

```text
90-degree double-skin generalized cylinder
mid-radius 12s
thickness 8s
axial source length 40s
planar tails at least 20s
deterministic positional jitter <= 0.03s
5% geometry-key triangle deletion
20% independent triangle-winding reversal
```

**SF02 — axial crop**

The SF01 geometry cropped to a `16s` supported axial interval. It must still activate exactly once.

**SF03 — scale/angle fold**

```text
60-degree bend
inner radius 21s
outer radius 27s
thickness 6s
15% deterministic contiguous stripe deletion on each skin
```

**SF04 — noncircular offset fold**

A C2 planar profile with:

```text
total monotone normal rotation 75 degrees
radius of curvature varying continuously from 18s to 36s
constant normal offset thickness 6s
two planar tails
20% deterministic hash deletion
```

This fixture is mandatory. It ensures PNCT is testing offset-chord transport rather than quietly recreating a circle or focal-center detector.

### 5.2 Mandatory source-free negatives

Each must emit zero axes:

1. flat double skin;
2. single-skin 90-degree bend;
3. silhouette/free-edge sheet;
4. straight through-opening rim;
5. rounded terminal cap and terminal thickness wall;
6. welded miter corner with a convex weld bead;
7. overlapping formed seam and closed-box cycle corner;
8. independent nearby panel at a `6s` gap;
9. concentric full double-wall cylinder.

A final duplicate fixture supplies the SF01 true candidate plus two line-perturbed duplicate hypotheses. After frozen V16 NMS, exactly one axis must remain.

### 5.3 Metamorphic and replay gates

For every positive and negative fixture, the full 64-byte header and every 192-byte candidate record must be byte-identical under:

* triangle permutation;
* cyclic and reversed vertex order within every triangle;
* complete triangle-winding reversal;
* the specified random individual winding reversals;
* candidate-axis reversal;
* seed-plane swap;
* rigid rotations of 17, 53, and 101 degrees plus nonzero translations, after inverse normalization;
* uniform scale factors `0.5` and `2.0`, with `s` transformed identically;
* three independent process launches.

Semantic equality without byte equality is a failure.

### 5.4 Native-resolution source-free audit

Every activation and every fixture mismatch receives a 4096-pixel native audit containing:

* three global mesh projections;
* the 11 station cross-sections;
* all clipped intersection segments;
* valid and ambiguous chords;
* the 12 selected-arc bins and complementary-arc support;
* chord midpoint paths;
* per-station thickness and pass/fail reason;
* the candidate axis in native mesh context.

A numerically passing activation fails visually if any material fraction of its chords visibly:

* joins two points on the same terminal cap;
* lands on an opening wall;
* crosses to an unrelated nearby panel;
* follows a weld bead rather than two skins;
* branches among three or more layers;
* lacks two tail continuations.

### 5.5 Label-closed real-scan gates

No scan identity is supplied to the PNCT worker. The harness alone schedules the scans.

**U02 no-op gate**

```text
zero novel PNCT markers after frozen V16 NMS
V16 marker output byte-identical
zero overflow or structural failure
```

Any U02 activation terminates V20 after its native audit.

**C017 activation gate**

```text
exactly one novel PNCT marker after frozen V16 NMS
the marker is not a duplicate of a retained V16 line
all 11 station records are present
zero overflow or structural failure
```

Zero or more than one novel marker terminates V20. Labels, expected count, and gold remain closed.

The C017 activation must then pass the native visual checklist: two distinct offset skins, a noncollapsed midpoint sweep, both tail continuations, and no cap, wall, opening, seam, cycle, or unrelated-panel chord.

### 5.6 Locked 35-scan gate

Only after every prior gate passes:

1. freeze core, tests, renderer, constants, and serialization hashes;
2. consume one claim;
3. execute the 35 scans once;
4. replay once solely for byte determinism;
5. then open the locked labels.

Retention requires all of:

```text
at least 5 novel true-positive lines
novel-line precision >= 0.90
novel true positives distributed across at least 3 part families
combined recall >= 137 / 176 = 0.778409
combined F1 >= 0.7345
at least 6 exact-count scans
none of the existing 4 exact-count scans becomes non-exact
no family F1 decreases from V16
100% novel marker render admissibility
100% output byte identity on replay
combined P95 <= 300 seconds
peak process-tree RSS < 16 GiB
35/35 successful jobs
```

No threshold may be altered after this result.

### 5.7 Frozen held-out safety gate

On U02–U05, the combined marker result must remain exactly:

```text
15 TP / 0 FP / 0 FN
```

PNCT must contribute zero novel post-NMS markers on all four scans, and the V16 marker files must remain byte-identical. Any novel marker, failed job, replay mismatch, or render failure stops V20.

Passing these gates retains PNCT only as a research lane. Production and runtime/main remain closed.

---

## 6. **EXPECTED FAILURE MODE**

The likeliest failure is **recall loss when the opposite skin is absent over too much of the normal interval**. A genuine bend may visibly retain a smooth outer-skin rotation, while occlusion or mesh fragmentation removes the inner trace in four or more interior bins or more than three stations.

The decisive native diagnostic is:

* one coherent curved surface trace remains visible;
* the surviving chord distances have low dispersion;
* the surviving midpoint path is smooth;
* empty chord bins coincide spatially with genuinely missing opposite-skin geometry;
* the failure reason is coverage alone, not ambiguity, thickness variation, or complement support.

That result would show that constant-thickness ownership is physically appropriate but not observable often enough in the frozen scans. It must still terminate PNCT; lowering the `7/10` or `8/11` gates would be a prohibited rescue.

The likeliest false-positive stressor is a large rounded terminal hem. Its diagnostic is chord endpoints lying on opposite sides of the same cap. The midpoint-collapse predicate should reject it. A visually confirmed hem that nevertheless passes all predicates falsifies the architecture rather than licensing another midpoint threshold.

**AUTHORIZE_PROJECTIVE_NORMAL_CHORD_TRANSPORT_V1_SOURCE_ONLY_TRANCHE**

[1]: https://link.springer.com/article/10.1007/s00371-007-0197-5 "https://link.springer.com/article/10.1007/s00371-007-0197-5"
[2]: https://www.geometrie.tugraz.at/wallner/sdcurv.pdf "https://www.geometrie.tugraz.at/wallner/sdcurv.pdf"
[3]: https://www.cs.utexas.edu/~huangqx/integral_cagd15.pdf "https://www.cs.utexas.edu/~huangqx/integral_cagd15.pdf"

