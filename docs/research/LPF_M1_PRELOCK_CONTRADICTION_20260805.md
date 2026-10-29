# LPF-M1 pre-lock contradiction and mandatory audit findings

Date: 2026-08-05  
Status: **BLOCKED BEFORE 37-CASE LOCK; GOLD NOT OPENED**  
Scope: research-only `LOCAL_PLANE_FRAGMENT_PROBE_V1`

## Executive decision

LPF-M1 cannot be frozen under the professor-approved V1 profile as currently
written. The frozen C001 V2 input contains six singleton modes whose
conservative fitted plane residual exceeds the fixed 3.0 mm tolerance. The
prior guidance simultaneously requires:

1. every input mode belongs to exactly one fragment;
2. singleton fragments are retained;
3. every fragment has `surface_fit_p90_mm <= plane_fit_tolerance_mm`;
4. the plane-fit formula and threshold remain unchanged; and
5. there is no second configuration or threshold rescue.

Those statements are jointly unsatisfiable on the frozen input. We therefore
stopped before the 37-case batch, bundle lock, and gold attribution.

A provisional implementation retains each inherited high-fit singleton but
sets `transition_certificate_allowed=false`. That is conservative with respect
to emitted certificates, but it is not professor-approved V1: the exception is
not named in the profile, not represented by a fragment-kind enum, and not
included in `engineering_profile_sha256`. It must not be silently grandfathered
into the experiment.

GPT-5.6 Pro must make one exact ruling:

- **FAIL V1 formally:** the frozen input falsifies the specified profile before
  inference; remove the exception, fail C001 closed, and terminate this tranche;
  or
- **APPROVE a versioned bounded orphan rule:** define a new profile/version,
  exact orphan/singleton semantics, serialization, hard constraints, counters,
  engineering-hash inputs, replay attestations, and STOP/GO implications.

## Direct frozen evidence

C001 uses `plane_fit_tolerance_mm = 3.0`. The six offending V2 modes and the
serialized provisional fragments are:

| V2 mode | Raw conservative fit (mm) | Provisional fragment | Serialized fit | Certificate allowed |
|---|---:|---|---:|---|
| `M_c5cdc0de4aae4307` | 3.573780… | `LPF_19274884b25e15ba` | 3.573780 | false |
| `M_835a882995963044` | **3.5759435084334337** | `LPF_1f71577ba9b9058d` | 3.575944 | false |
| `M_d74b1aaccb6e389e` | 3.367868… | `LPF_7338ac4cb5f5f923` | 3.367868 | false |
| `M_21dcd91654854077` | 3.055962… | `LPF_bc0eb918726bb02a` | 3.055962 | false |
| `M_fa2bf950a002aa45` | 3.425333… | `LPF_c59b904844e755bc` | 3.425333 | false |
| `M_0340004b899fcb1f` | 3.003250… | `LPF_da96f5906905c9a5` | 3.003250 | false |

The maximum raw value is read directly from the frozen pool, not reconstructed
from rounded output. C001 reports 2,372 input modes, 617 local fragments, 425
singletons, six high-fit ineligible singletons, 858 local pairs, 1,652
components, and zero valid certificates. Its provisional engineering-profile
hash is
`2795aa85c2adcfe9a10d7cc24dafc25a52720e3541e637bfb901464052df2a06`,
but the hash formula does not mention the singleton exception.

The contradiction is not a floating-point boundary accident: five of the six
offenders exceed the limit by more than 0.05 mm and the maximum exceeds it by
about 0.576 mm.

## What was implemented before the stop

The research-only implementation exists in isolated new files:

- core local fragment construction and validation;
- inference CLI that hashes the scan/A0 files and parses the V2 pool;
- bounded per-case batch runner;
- inference bundle builder;
- post-lock attribution layer with exact one-to-one matching;
- focused core, memory, bundle, and post-lock tests.

The provisional core preserves every singleton and prevents an over-tolerance
singleton from completing the transition certificate. No prediction, birth,
ownership, learning, or runtime path was modified. The provisional behavior is
useful for exposing the contradiction and measuring boundedness; it is not a
valid locked experiment until Pro rules on it.

## Determinism and memory evidence

Three isolated replays were completed for the required cases:

| Case | Selection reason | Output SHA-256, all 3 runs | Peak RSS across runs | Output bytes |
|---|---|---|---:|---:|
| C001 | 200,000-point reference | `3b73b1dff78777bef60683595ad520d3ec3fec77b342fc149056a19351a88ad7` | 134,037,504 B | 4,198,127 |
| C030 | maximum V2 mode count, 2,465 | `b56e707982413aac2065cbf23eba1b2d3f81833d815257b73b75efca36407d8b` | 126,058,496 B | 10,535,764 |
| C003 | maximum V2 edge count, 6,301 | `25dd2523bd5838c9762557f2bf58c480d5947813f620b30a7260926bb4b7eb99` | 127,959,040 B | 3,066,423 |

All nine probe payloads are byte-identical within case. The overall maximum is
134,037,504 bytes, below 135 MB and far below the 16 GiB hard limit. Per-run
wall time and profile hashes are included in the replay evidence JSON. Resource
profile files differ as expected because they contain timings and RSS; the
canonical probe payloads do not.

The requested focused suite passed: **170 tests in 2.19 seconds**. Passing tests
show internal consistency only; tests cannot approve a representation rule that
the preregistered profile did not contain.

## Read-only audit findings that block locking

### P0 — replay evidence can be bypassed before lock

`build_f1_local_plane_fragment_probe_bundle.py` currently treats a missing
`replay_probe_sha256s` list as `[probe_sha]`. The normal 37-case batch writes one
hash and `replay_count=1` for every case. Consequently a bundle can declare
`locked=true` after one run per case, before the mandated three isolated runs
for C001, C030, and C003. The post-lock script validates the generic lock and
safety flags but does not require a designated-case replay attestation before
opening gold.

Mandatory adjudication/fix: require exactly the designated cases, exactly
three isolated run hashes per case, byte identity, per-run RSS evidence, case
selection attestation (C001/reference, maximum-mode, maximum-edge), and a lock
record whose hash is verified by post-lock attribution. No one-run fallback may
satisfy the lock.

### P1 — unversioned singleton exception changes the experiment

The provisional code checks the plane-fit invariant only when
`len(mode_ids) > 1`, then marks an over-tolerance singleton certificate-ineligible.
The V1 engineering hash covers the general fragment-fit formula and threshold,
but not this exception. Thus the output claims the unchanged V1 profile while
executing a rule absent from the professor response.

Mandatory adjudication/fix: either fail V1, or bump the profile/schema and hash
the exact singleton/orphan rule, enum, eligibility behavior, counters, and all
affected rejection/attestation logic.

### P1 — post-lock result omits mandatory GO criteria

The post-lock `_evaluate_gate()` currently reports only recovery >=14,
independent groups >=8, false certificates <=8, and 37 successful jobs. The
professor’s GO is conjunctive and also requires prediction/birth/ownership/
solver immutability, unchanged V2 and A0 hashes, no cap breach, no forbidden or
source-point serialization, the designated 3x replay attestation, and every
measured process below 16 GiB.

Mandatory fix: expose every condition as an explicit criterion in the result,
derive it from a hash-locked attestation rather than assumed flags, and make any
missing criterion fail closed.

### P1 — engineering/profile validation is incomplete

Payload validation confirms that `engineering_profile_sha256` is well formed
but does not recompute it from the exact profile formulas, thresholds, caps,
first-rejection order, rounding, and exceptional rules. Nested fragment/pair
schemas and `geometry_gate_depth` also need exact recomputation rather than
partial field checks.

Mandatory fix: recompute the engineering hash and exact nested schemas at the
CLI, batch, bundle, replay, and post-lock boundaries. An approved orphan rule
must be among the hashed formula fields.

### P1 — post-lock lexical tie encoding is not bounded

The exact matcher implements lexical tie-breaking with `radix = 1 << E`, where
`E` is the number of eligible bipartite edges. It is exact on current unit
fixtures and passed 1,000 brute-force comparisons, but its integer width grows
linearly with every allowed edge and is not bounded for the permitted component
caps.

Mandatory fix: replace exponential-bit cost encoding with a bounded exact
lexicographic procedure (for example successive feasibility fixing after
maximum-cardinality/minimum-cost lock), and add worst-cap resource tests.

### P2 — inference input projection is broader than the approved contract

The CLI parses and passes the complete candidate-pool mapping into the core.
The core reads approved tables and metadata, and output forbidden-field checks
are recursive, but the boundary does not construct an explicit allowlisted
projection before calling the probe.

Mandatory fix: project exactly the professor-approved input keys, reject extras
or exclude them from the core input, and attest that graph, ownership, solver,
truth, identity, and path fields cannot be consumed or influence an inference
hash.

### P2 — inherited cap/working-cloud attestation must be explicit end to end

The core checks the V2 cap profile and working-point limits, but the locked
payload/bundle does not expose and independently recompute every inherited
`transition_atom_caps`, `working_point_budget`, and
`transient_sampled_point_count` attestation at every boundary.

Mandatory fix: specify which fields must be serialized, table-hashed, and
recomputed in the bundle and post-lock validator without adding source points.

## State of the experiment

- No 37-case LPF inference batch was frozen.
- No LPF bundle was locked.
- Gold was not opened for LPF.
- No 14/8/8 result exists.
- No profile has passed or failed empirically against gold.
- No A1 ownership, birth column, learning, or runtime promotion is authorized.

The only current decision is a pre-lock specification blocker. The next code
change must follow an exact professor ruling; it must not infer permission from
the provisional conservative exception.
