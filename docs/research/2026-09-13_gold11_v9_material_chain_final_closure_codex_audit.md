# V9 material chain: final bounded closure review

Date: 2026-09-13. Reviewer: Codex/Astra `v9_material_identity_chain`.
Scope: source/test-only closure review of the stable Fable checkpoint. AGENTS,
current status and the complete 138-line review response were read. No
imports, tests, physical inputs, Gold, attempts, controls or caches were
accessed. Only this new report was written; Fable retains source ownership.
This is not a physical PRE, execution attestation or production approval.

## Decision

**The connected interface blockers are closed at source; the remaining
two-layer inner-eligibility claim is not.** Review-response row 1 does not
close the previously reported material-side finding: it restates the same
unverified status as the condition for granting inner-specific eligibility.
There is also a small remaining type-strictness omission at two publication
fields. Neither issue calls for another broad research phase. The minimum
repair can preserve all measured contact endpoints and the generic contact
F1 path unchanged.

The new connected test source genuinely exercises the previously missing
measurement-to-projection interfaces. Its positive F1 is explicitly a
copied-carrier transport check, not localization or physical accuracy.

## Exact checkpoint inspected

Review response:
`docs/research/2026-09-13_fable51_v9_review_response.md`, SHA256
`31def50d1d4795c22f8c2c79b8899eafe445c388ecfd7041b1c4697da0985de4`.
The following named files matched its declared hashes:

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `a894b96057ca886981a6c8ac4f968519d4c6246af05254eddd03ed41e9864ad1` |
| `backend/cad_guided_context_measurement_v2.py` | `0504cd32364a1708d17f8bddafba51a9f9640edc11866e33b37e693a0d4a3534` |
| `backend/cad_guided_support_cover_measurement_v3.py` | `a298b123218d931911e064a2f35ab52e5553f2c1ad5b25a71e3c149c2d7d9234` |
| `scripts/gold11_contact_scoring_v3.py` | `55cf0920b545e0aa204da1b0882298f92ca7015210466b6e8fc777ddafacf10d` |
| `scripts/gold11_schedule_projection_v4.py` | `32029cbb813381e360bed791e97701cdfb6735cfa6fa12f9a7f90d2477b95708` |
| `tests/test_cad_guided_v9_connected_chain_v1.py` | `14e405921fcf4869870d0274cfdd66705172a58289aecae96d0d1fecc21269cf` |
| `tests/test_gold11_contact_scoring_v3.py` | `173b6b7912f6c20953ce19ad982cb6c2f61dcf9b026660b27fa762a175c78e1c` |
| `tests/test_gold11_schedule_projection_v4.py` | `9510b9eb422af0844d0a88fbde980c8984bf5a8e5ec0140d1b023419e9e070c1` |

The connected test file was read completely. Contact V2-to-V3 changes were
reviewed as a full source diff, with the changed consumer boundaries and
targeted test seams read directly. No claim is made to have independently
rerun the author's reported 826 passing tests.

## Prior-finding closure ledger

| Finding | Bounded disposition |
| --- | --- |
| Real discovery emitted an extended policy rejected by planner V1 | Closed: seven-field scalar fit policy restored; separate layer-policy object retained and bound in support identity/publication. |
| Cover compared scalar policy with extended policy | Closed: scalar and layer records compared separately; all source records and cover witnesses share the declared layer policy. |
| V4 delegated real V9 bytes to incompatible contact V2 | Closed: V4 calls contact V3 directly through its versioned helper; V2 remains unchanged and is expected to reject new schemas. |
| Ordinary canonical refusal lacked association eligibility | Closed: association construction initializes eligibility and absent canonical deviations before the fit. |
| Overlap invalidation left canonical deviations incompatible with unavailable status | Closed: invalidation clears active canonical association deviations and retains original per-context evidence/support. |
| Per-context material identity not revalidated | Closed: context validator now binds `association.per_context_measurements` to the same status/role/eligibility as its record and window. |
| `_bind_identity` accepts boolean aliases | Helper closed: it now uses exact types. Two direct comparisons outside that helper remain; see P2 below. |
| Disabled selection grants inner eligibility | Closed: `CONTEXT_ASSUMED_NOT_VERIFIED` is absent from the eligible tuple. |
| Two-layer unverified status grants inner eligibility | **Open P1:** still present in both kernel and contact V3. |
| Mocked discovery/old contact fixtures bypass the actual integration seams | Closed at test-source coverage level: the new connected tests use real discovery/cover/transport; current V4 fixture builder uses the new contact grammar. Execution remains author-reported. |

## P1 — Two-layer unverified evidence still grants an actual inner claim

At this exact checkpoint, kernel V2 `:50` and contact V3 `:44` define:

```text
INNER_FACE_ELIGIBLE_STATUSES =
    ("TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED",)
```

The kernel derives `inner_face_claim_eligible` from that tuple (`:754`),
publishes `INNER_FACE_RADIUS_UNVERIFIED` and a nonnull
`radius_deviation_mm` when true (`:801–803`), and the context/cover consumers
permit the corresponding canonical radius-to-nominal-inner-radius deviation.
The contact projection counts these as inner-face-eligible carriers.

The review response supplies no new independently bound material ownership
or arc-side isolation mechanism that could convert the unverified hypothesis
into eligibility for an actual inner-surface answer. The current layer
compatibility uses parallel-plane, gap and projected-interval criteria; its
own scope explicitly remains unverified material. The nominal outer-arc
heuristic is correctly labelled `certificate=false`, not measured exclusion
proof. Renaming the old heuristic honestly closes its overstatement, but does
not establish the missing identity.

The connected two-skin test assumes an ideal sheet's paired skins when
constructing its fixture, then asserts the disputed eligibility true. That
shows the code's enum propagation and fit on that fixture, not that arbitrary
two-layer scan evidence resolves physical material identity. It cannot serve
as independent closure of this finding.

The smallest repair is to make every currently unverified status ineligible
for inner-specific claims, while retaining the two-layer hypothesis string,
scan-derived radius and generic contact generator. Leave the unresolved
inner-radius deviation unavailable. If a separate hypothesis-relative number
is useful, label it as an auxiliary conditional comparison and do not route
it through `inner_face_claim_eligible` or an asserted inner radius identity.
Update the exact assertions and report counts that currently encode the
overclaim. No endpoint, matching-rule or reference change is necessary.

This is **not** a requirement for verified inner material side before the
generic contact-only development F1. That score remains allowed, with its
declared provisional units/frame and absent physical measurement truth.
Unknown material side must remain unknown; it need not suppress an explicitly
generic contact line. No historical-score comparison or floor target follows.

## P2 — Two direct publication comparisons still accept 0/1 as booleans

The type-strict `_bind_identity` fix is correct. However:

- context V2 `:571` compares the association's separate
  `inner_face_claim_eligible` using `==`;
- cover V3 `:375` compares the finite instance's eligibility using `==`
  without invoking `_bind_identity` for that instance in the publication
  validator.

Thus an association-only boolean-to-0/1 mutation, or a cover-instance-only
mutation applied consistently to its prediction alias, can survive those
respective local checks. Contact V3 is stricter for emitted carriers and will
reject a non-boolean there; this is not a demonstrated route to changing the
numeric F1. It is an incomplete local publication guarantee compared with
the review response's all-boundaries type-strict claim.

Use boolean identity/type checking or the existing strict helper at these
two sites, and add the two direct mutation checks. This is a bounded contract
correction, not a new scientific gate or a need for physical execution.

## Connected test coverage and scoring preservation

The complete 413-line connected suite now exercises actual local crop,
kernel fits, context V2, exact planner V1, cover V3, their validators,
contact V3 and the V4 wrapper. Its registration record and keep-all surface
role selector are clearly disclosed substitutes. It does not qualify the
PLY/native input, actual registration accuracy or real cap classification.

Its nine tests cover two-skin contact output, generic singleton output,
outer-only non-inner claims, the old measurement-policy refusal, finite
angular-support refusal, a forced ordinary canonical refusal alongside a
retained positive, disabled selection, V4 exact-byte transport/reconstruction
equal to direct V3, and malformed identity/policy cases. The forced refusal
is an appropriate explicit exception-path test; other cases do not mock the
measurement/discovery result. The reference copied from an emitted carrier
checks transport/matcher arithmetic only, as the author now states.

The current V4 fixture builder no longer reuses the old measurement grammar:
it uses the contact V3 workflow constructor and adds the separately tested
schedule/context envelope. The connected V4 test then replaces that
measurement portion with real emitted cover bytes, checks unchanged endpoints,
compares the direct V3 contact projection and verifies exact reconstruction.

The full contact V2-to-V3 source diff still leaves the coordinate-copying
calculation, matcher imports, one-to-one assignment, 5-degree/4-mm/0.60/0.60
matching thresholds, eleven-case/seventy-one-reference requirement, failure
FN retention and precision/recall/F1 aggregation unchanged. Identity metadata,
grammar and diagnostic identity categories are the added dimensions. The
score remains explicitly generic contact-generator localization, with
observed physical counts unknown and conformance indeterminate.

No additional connected regression was found in this bounded source read.
The inner-eligibility P1 is a retained prior finding, not a new broad research
requirement. The author-reported test counts are not independently executed
evidence from this review.

## Handoff and smallest next action

The P1 and residual P2 sites were sent immediately to the root. Ask the sole
editor for the small eligibility/boolean patch and corresponding focused
assertion updates, preserving the measured geometry and generic contact
score. Review only that exact delta afterward; no new general research phase
is proposed. This report does not authorize a physical PRE or mark the whole
roadmap complete.
