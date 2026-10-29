# Mathematical Specification — Expected Angle Truth

Date: 2026-03-29
Status: draft
Priority: high
Dependencies:
- `MATH_SPEC_FEATURE_CLASSIFICATION.md`
- `MATH_SPEC_LOCAL_BEND_METROLOGY.md`
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define which released source supplies the authoritative nominal for bend-angle inspection and when the system must abstain instead of averaging conflicting sources.

## Core principle
An angle is not an inspection truth until it is mapped onto:
- the correct feature family
- the correct geometric primitive
- the released inspection master

Tooling/process angles are causal inputs to manufacturing. They are not inspection truth.

## Scope
This specification applies to:
- countable discrete bend targets whose inspection primitive is angle and optionally radius

This specification does not apply as the primary truth model to:
- rolled/process features
- hems / closed returns
- joggles
- beads
- profile- or gap-controlled transitions

## Common primitive
All competing sources must first be mapped to the same primitive:

`β = arccos(n_1 · n_2)`

where:
- `β`: face-normal angle in `[0, π]`
- `n_1`, `n_2`: adjacent face normals in the nominal released geometry

If a family uses a different public convention, convert by an explicit family map:

`θ = g_f(β)`

The conversion map must be frozen per family and stored as metadata.

## Source mappings
### CAD nominal
`β_CAD = arccos(n_1^CAD · n_2^CAD)`

### Drawing nominal
`β_DWG = g_DWG(θ_callout, f)`

where:
- `θ_callout`: drawing callout angle
- `f`: feature family
- `g_DWG`: convention map from drawing semantics to face-normal angle

### Process/tooling nominal
`β_PROC = g_PROC(θ_tool, f)`

This source is diagnostic only and must not win the inspection truth competition.

## Source precedence
Rank sources as follows:
1. released inspection master:
   - drawing callout if drawing is the contractual master, or
   - CAD + PMI if model-based definition is the contractual master
2. released CAD geometry if no explicit higher-level inspection callout exists
3. process / tooling nominal for diagnostics only

Decision rule:
- if one released source is explicitly marked master, it wins
- if neither released source is explicitly marked master and they disagree materially, return `ABSTAIN_SPEC_AMBIGUITY`
- do not average released sources

## Material disagreement threshold
Let `τ_θ_spec` be the released tolerance on angle if present.

Define ambiguity threshold:

`Δ_amb = max(0.5°, 0.25 * τ_θ_spec)`

If no explicit angle tolerance exists, use `Δ_amb = 0.5°`.

Flag ambiguity if:
- `|β_1 - β_2| > Δ_amb`
- or source conventions cannot be mapped unambiguously to the same primitive
- or the family's correct primitive is not angle at all

## Feature-family routing
### Countable discrete bends
Truth primitive:
- angle `β` or converted `θ`
- optional radius if the release explicitly controls it

### Rolled/process features
Truth primitive:
- centerline placement
- section-profile geometry
- optional radius/arc-extent field if explicitly released

Angle must not be used as the primary released nominal for rolled features.

### Hem / closure families
Truth primitive:
- gap
- closure
- profile / offset

Angle may be retained only as a secondary diagnostic field.

## Output contract
Every target record must expose:
- `measurement_primitive`
- `nominal_source`
- `nominal_source_master`
- `nominal_angle_face_normal_deg`
- `nominal_angle_display_deg`
- `nominal_radius`
- `spec_ambiguity_flag`
- `spec_ambiguity_reason`

## Runtime decision rules
### Released-source agreement
If two released sources map to the same primitive and:

`|β_1 - β_2| <= Δ_amb`

then use the declared master or fallback precedence.

### Released-source ambiguity
If:

`|β_1 - β_2| > Δ_amb`

and no explicit master exists, set:
- `spec_ambiguity_flag = true`
- `measurement_authority = NONE`
- target result to `ABSTAIN_SPEC_AMBIGUITY`

### Process-sheet disagreement
If process/tooling angle disagrees with the released nominal:
- log the disagreement
- expose it in diagnostics
- never replace the released nominal

## Benchmark proof slice
First proving set:
- parts with known CAD/drawing mismatch history
- hybrid families where wrong nominal source previously generated false bend deltas
- parts whose local geometry and global overlay agree but prior angle comparison disagreed

Success criteria:
- no more silent averaging of conflicting nominal sources
- no more process-angle substitution into released bend QC
- explicit abstention on material source ambiguity

## Non-goals
This spec does not define:
- datum-island selection
- local slice estimator internals
- rolled-section tolerance aggregation
- scan-level accept/reject policy

## Short conclusion
The authoritative bend nominal is the released inspection master mapped onto the correct primitive. If that mapping is ambiguous, the system must abstain instead of fabricating a single truth.
