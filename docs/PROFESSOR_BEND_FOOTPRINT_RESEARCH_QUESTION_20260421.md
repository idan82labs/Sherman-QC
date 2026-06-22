Professor,

I would like your help formulating a scan-only bend inference problem in a mathematically correct way before we implement it.

The application is automated interpretation of 3D scans of bent sheet-metal parts. In the specific setting I care about here, assume there is no nominal CAD available at inference time. The input is only a point cloud sampled from a bent part, for example from lidar or a structured-light scanner. The practical goal is to infer how many discrete bends the part has, where those bends are, and how confident the system should be in that interpretation.

The problem I am trying to solve is that a raw detector of planar regions or candidate bend transitions does not produce a stable bend count. It can under-segment a part into too few large transitions or over-segment it into too many local candidates. I therefore want to move away from counting detector outputs directly and instead define a more physical intermediate object.

The object I have in mind is not a single “bend point,” but a spatially coherent bend-support region: a set of nearby scan points that agree they belong to one physical bend, together with the bend axis, the local transition band across the fold, and the usable span along that axis. Informally, I would call this a bend footprint or bend stripe.

The intended logic is:

1. detect local evidence of bending from point-cloud geometry,
2. group nearby points into a candidate bend-support region only if they agree geometrically,
3. estimate a bend axis and a bounded physical footprint for that region,
4. declare one unique bend object for that footprint,
5. suppress any additional detections that lie inside the same physical footprint so the system does not count the same bend repeatedly.

The intuition is that if one can reliably recover the physical extent of a bend, then one can know where “not to count again.” But I do not want to build this on an intuitive engineering story if the mathematical object is wrong.

The main ingredients I expect the formulation to use are:

- local point neighborhoods,
- robust plane or surface estimation, likely RANSAC-based,
- local curvature or normal-change structure,
- estimates of point spacing or sampling density,
- adjacency or continuity constraints between points belonging to the same bend region,
- and a rule for when two local detections should be considered the same bend because they lie on the same bend footprint.

The point spacing issue matters here. The scanner has some effective sampling density, and the point cloud itself also exposes an empirical average nearest-neighbor spacing. My thought is that any grouping rule should be tied either to the actual measured spacing in the cloud or to a principled sensor-resolution prior, rather than to arbitrary fixed thresholds alone. In other words, if points are assigned to one bend-support region, that assignment should be justified by geometry and by spacing/continuity at the scale the sensor can actually resolve.

To frame the decision cleanly, there are really two competing directions:

1. Current detector-counting direction

- estimate planes or local bend candidates,
- pair or cluster them,
- and treat the surviving candidate set as the bend count.

2. Proposed footprint-inference direction

- estimate local geometric evidence of bending,
- assign agreeing point neighborhoods to latent bend-support regions,
- recover one bounded physical footprint per true bend,
- and count those latent bend footprints instead of counting raw detector outputs.

What I need from you is not just whether the footprint idea is intuitively reasonable, but whether this second direction is mathematically better founded than the first, and if so, how it should be formulated.

What I need from you is not a software suggestion, but a mathematical plan.

The questions are:

1. Is “bend footprint” or “bend stripe” a mathematically sensible latent object for this problem?

More concretely:
- Should a bend be modeled as a bounded subset of the scanned surface together with an axis and local transition geometry, rather than as an unstructured detector event?
- Is there a better formal object than what I am informally calling a footprint?

2. How should one define the membership of scan points in a single bend-support region?

I am looking for a principled rule or energy, not a heuristic description. For example:
- agreement in local normals,
- agreement in curvature sign/magnitude,
- proximity to a common bend axis,
- continuity in geodesic or Euclidean neighborhood structure,
- consistency with two adjacent flange families,
- and consistency with the point spacing imposed by the sensor.

3. How should the physical extent of one bend be defined?

I need a mathematical definition for both:
- the extent across the fold, meaning the transition band between the adjacent flanges,
- and the extent along the bend axis, meaning the usable bend span before one should stop counting the same bend and allow another one.

4. How should duplicate suppression be posed mathematically?

Suppose several local detections are produced from the same physical bend. Under what conditions should they be merged into one latent bend object? I am looking for the right equivalence or clustering criterion:
- same inferred axis within tolerance,
- overlapping footprint,
- compatible adjacent flange normals,
- compatible radius/angle estimate,
- or some joint assignment model.

5. How should point spacing enter the formulation?

If the scan has empirical nearest-neighbor spacing \(h\), or if the sensor gives a known spatial resolution, should grouping and footprint estimation thresholds scale with \(h\)? If so, how? I want to avoid a method that is valid only for one scanner density.

6. How should this be kept compatible with robust estimation methods such as RANSAC?

I want to stay faithful to a robust geometric pipeline, not replace it with a purely learned or black-box clustering method. So I would like a formulation in which:
- robust local plane/surface hypotheses are still estimated,
- but those hypotheses are assembled into higher-level bend objects in a mathematically consistent way.

7. What is the right inference problem?

My current instinct is that the system should infer a set of latent bend footprints

\[
\mathcal{B} = \{B_1, \dots, B_K\}
\]

from a point cloud \(Y\), where each \(B_i\) would carry at least:

- an axis,
- a bounded support region,
- local flange relationships,
- and continuous geometric parameters such as angle/radius if identifiable.

Then the discrete bend count would be

\[
K = |\mathcal{B}|.
\]

But I do not know whether this is the right abstraction, or whether the better object is a graph over surface patches, a marked line process, a latent partition of points, or something else.

8. What should the confidence object be?

Should the output be:
- a single estimated count,
- a count range,
- a posterior over latent bend-footprint sets,
- or a selective estimate with abstention when the footprint decomposition is unstable?

9. How should this be validated?

I would like your recommendation for the right mathematical validation criteria. I assume exact count accuracy alone is not enough. I expect we should also care about:
- stability under downsampling and perturbation,
- correctness of the inferred footprint decomposition,
- and whether duplicate detections from one true bend are successfully merged.

If you think this direction is sound, I would like you to outline a full mathematical plan for it:

- define the latent object precisely,
- define the measurement model from a point cloud,
- define the grouping/assignment criterion for points into bend-support regions,
- define how one estimates axis and bounded footprint,
- define how duplicate suppression should work,
- define how point spacing/resolution should enter,
- define the final decision object and confidence object,
- and define the right validation criteria.

I am deliberately not assuming any prior context here. The design decision I need to make is whether scan-only bend counting should be reformulated as inference over physical bend footprints rather than counting raw geometric candidates, and if so, what the mathematically correct version of that reformulation is.

Thank you.
