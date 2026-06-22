Professor,

I am working on an automated bend-inspection system for bent sheet-metal parts, and I would like your view on the correct mathematical framing for a new capability we are considering.

The system already has a CAD-driven inspection mode. In that mode, a nominal CAD model is loaded, bends are extracted from the nominal geometry, and the scan is used to determine whether the nominal bends are present, measurable, and conforming. That is the authoritative quality-control path.

We are now considering a second mode for cases where no nominal CAD is available. In that mode, the input would be only a scan point cloud, and the system would try to infer the structural profile of the part: likely bend count, dominant bend-angle family, and part family such as repeated orthogonal sheet-metal folds versus rolled or mixed geometry. This would not be used for release, but it could be valuable for intake, reverse-engineering assistance, and early scan triage.

The problem is that the raw scan detector is not itself a reliable semantic answer. On some parts it is close to the expected count, but on other parts it under-segments or over-segments. For one current challenge case, which is believed to have 12 bends all at 90 degrees, the conservative detector returns 7 bends, while a more aggressive multiscale detector returns 15 candidates. If we filter those candidates to near-90-degree bends, we get 11 candidates, but those still collapse to only 6 duplicate line families under simple clustering. In other words, the raw detector appears to be a redundant candidate generator rather than a trustworthy structural interpretation.

That has led us to a possible architecture with two distinct modes:

1. CAD-authoritative nominal mode

- nominal CAD defines bend count, bend topology, and expected angle family
- scan evidence is used to measure and confirm that nominal structure

2. Scan-only structural inference mode

- no CAD is available
- the system generates redundant bend candidates from the scan
- a profile-selection layer chooses the most plausible structural interpretation
- the output is an inferred structure with confidence and abstention, not a release decision

The theoretical questions I need your help with are these:

1. When nominal CAD exists, should it be treated as the authoritative definition of bend count, bend topology, and target angle family, with scan evidence restricted to measuring or supporting that nominal structure? Or should scan-derived structure ever be allowed to override CAD-defined topology if scan evidence is strong?

2. In scan-only mode, what is the correct output object? Should it be:

- an exact bend count,
- a bend-count range,
- a profile-family label plus confidence,
- or a posterior over structural interpretations with an explicit abstention option?

3. For scan-only structural inference, what should success mean theoretically on known parts?

- exact count accuracy,
- count-range containment,
- dominant angle-family correctness,
- structural stability under perturbations such as downsampling or seed changes,
- or some combination of these?

4. Is the right mathematical view that scan-only detection should be decomposed into:

- candidate generation,
- then profile-conditioned consolidation and abstention,

rather than treating the detector output itself as the final structural answer?

5. If nominal CAD exists, is scan-only profiling best understood as a fallback discovery mode only, or is there a principled framework in which it should remain an independent structural hypothesis even in the presence of nominal CAD?

I am not asking for software advice here. I am trying to avoid building the wrong mathematical object. The design decision I need to settle is whether nominal CAD should be the authoritative structural prior whenever it exists, and what the correct decision object should be when it does not.

Thank you.
