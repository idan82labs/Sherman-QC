# Sherman QC F1 — LPF-M1 pre-lock specification blocker

You are GPT-5.6 Pro extended-thinking acting as the continuing research reviewer
for Sherman QC F1. Continue directly from `02_PRIOR_PROFESSOR_RESPONSE.md`.
This is a GPT file-upload submission, not an email. Inspect every uploaded file.

## Why we stopped

Your `LOCAL_PLANE_FRAGMENT_PROBE_V1` guidance is not executable on frozen C001
without an additional rule:

- every V2 mode must belong to exactly one local fragment;
- singleton fragments must be retained;
- every fragment must have `surface_fit_p90_mm <= plane_fit_tolerance_mm`;
- the fit formula and profile remain unchanged;
- there is no retry, threshold relaxation, or second configuration.

Frozen C001 has six singleton modes above the fixed 3.0 mm tolerance. The
maximum raw V2 value is exactly `3.5759435084334337` mm. We provisionally
retained these singletons and made them certificate-ineligible, but that rule is
not in your response and is not encoded in the engineering-profile hash.

We therefore did **not** run or lock the 37-case experiment and did **not** open
gold. The 3x C001/C030/C003 replays are bounded and byte-identical, and 170
focused tests pass, but those facts cannot authorize a changed experiment.

## Required primary ruling — choose exactly one

### Ruling A: FORMAL PROFILE FAILURE

Declare `LOCAL_PLANE_FRAGMENT_PROBE_V1` infeasible on the frozen input. Specify
that the core must remove the provisional exception, fail C001 closed when an
inherited singleton exceeds the tolerance, and record a pre-inference STOP.
State whether this falsifies only the written LPF profile or the broader local-
plane hypothesis, and name the single permitted next research action, if any.

### Ruling B: VERSIONED BOUNDED ORPHAN/SINGLETON RULE

Approve a new profile/schema version and provide implementation-complete rules:

1. exact profile and schema names;
2. exact predicate identifying an inherited high-fit singleton/orphan;
3. whether it remains a fragment, becomes a distinct `fragment_kind`, or is
   recorded only in a bounded orphan table;
4. whether it may seed a local pair, enter a component, contribute parent mass,
   or affect global-plane coverage;
5. the exact hard rule preventing it from completing a certificate;
6. counters, bounded witnesses, canonical ID/hash inputs, and required nested
   fields;
7. required additions to `engineering_profile_sha256`;
8. invariant/property/schema tests and the formal counterfactual;
9. whether the preregistered 14/8/8 thresholds remain unchanged;
10. whether all existing provisional outputs must be discarded and replayed
    (our default is yes).

Do not suggest a tolerance sweep or silently redefine the 3.0 mm inequality.

## Mandatory audit adjudications

After the primary ruling, adjudicate each audit item in
`01_PRELOCK_BLOCKER_REPORT.md` with **MANDATORY BEFORE LOCK**, **OPTIONAL AFTER
LOCK**, or **REJECT**, and give exact implementation changes for every mandatory
item:

1. P0: the bundle currently locks with one run per case; require designated
   C001/C030/C003 three-run replay/RSS attestations before lock and before gold.
2. P1: the final GO result omits mandatory safety, hash, replay, cap, forbidden-
   data, and memory criteria.
3. P1: validators do not recompute the complete engineering hash, exact nested
   schemas, and gate depth.
4. P1: post-lock lexical matching uses an unbounded `1 << E` tie encoding;
   specify a bounded exact alternative.
5. P2: the CLI passes the full candidate pool rather than an explicit allowed-
   key projection.
6. P2: inherited V2 caps and working-cloud metadata need explicit end-to-end
   attestation.

## Response format

Label every substantive statement **OBSERVED**, **INFERENCE**, or **UNKNOWN**.
Return:

1. `PRIMARY_RULING: A` or `PRIMARY_RULING: B`;
2. the exact rationale;
3. if B, a complete versioned orphan/singleton specification;
4. a six-row mandatory-audit disposition table;
5. an ordered file-by-file patch plan;
6. exact new/changed tests and lock attestations;
7. the single mandatory stop milestone before gold may be opened.

Do not authorize A1 ownership, births, learning, threshold tuning, or
runtime/main promotion. Do not use gold to decide the profile ruling. If any
evidence is insufficient, state UNKNOWN and choose the fail-closed behavior.
