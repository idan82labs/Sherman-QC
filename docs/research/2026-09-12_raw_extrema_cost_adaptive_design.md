# Cost-adaptive raw spatial ordering — 2026-09-12

Source-only contingency, not execution authority. Root reports
V2's sole16 outcomes correct but efficacy failed: folded full12 common work
1,237,422 V1 versus 3,668,475 V2; V1 construction819,200/query418,222;
four-case CPU0.098855s versus1.624711s. These supplied aggregates are the only
run evidence used here. No artifacts, inputs, Gold, caches or target execution.
Existing sources and scoring remain unchanged.

## First reject unaffordable construction

V1 source `_build`, `_bounds`, `_evaluate`, `query` retains all rows, leaf256,
at most7 seeds and one bound per visited node. For S=32768 there are128 leaves
and255 nonpadding nodes. A complete unchanged query has at most
`9*(32768+7)+37*255 = 304410` common arithmetic steps. Nine such remaining
queries cost at most2,739,690, using the declared37-step world-bound supplement,
not hardware instructions or CPU.

The prior prospective80S Morton key cost plus a required6S new leaf-box pass
is already2,818,048: more than all possible remaining query savings after three
calibration queries. Sorting would additionally consume15S sort_work, a SEPARATE
ledger; do not add it to the existing common denominator. Therefore do not
implement that lazy Morton proposal for the twelve-query discriminator merely
to test whether amortization works. This bound is conditional on the proposed
80S construction model, not a lower bound on every possible spatial key.

## Minimal lazy policy for a genuinely cheaper mechanism

Build only the unchanged V1 index initially. Keep its raw ownership, support
hash, seeds and complete output ledger. Finish queries normally, retaining all
initial query costs. Let t be completed queries, R the known remaining count,
E_i actual rows evaluated (including seeds), M_i actual world bounds evaluated,
and `C_i = 9*E_i + 37*M_i`. Do not include build node visits in M_i.

Permit AT MOST ONE new-order construction, only after all three conditions:

1. At least three completed queries individually evaluated at least
   `ceil(S/2)` rows. This deliberately targets repeatedly expensive queries,
   not every moderately expensive one. It uses raw work only.
2. `R*(sum(C_i)/t) >= 2*K`, where K is a source-derived conservative upper
   reservation for the ENTIRE new construction in the SAME common ledger.
   Use exact bounded integer cross-products for the comparison and charge its
   bookkeeping. No query prefix, constructor, key, copy or box rebuild is free.
3. Existing remaining coordinate/sort/index/row/node/hash and working-memory
   caps admit that reservation BEFORE allocating or doing new work. No cap raise.

No build for batches under four queries or after failure.
Reconsider eligibility only after each successful query; once attempted, never
rebuild again. The three-observation and2x margin are proposed fixed heuristic
policy constants, not a calibrated probability or universal economic guarantee.
Freeze before a new run; no case identity, CAD, Gold or accuracy tuning.
Query identities/directions need not be inspected.

This actually excludes the supplied cheap folded batch, even for inexpensive K:
three half-support queries would require at least
`3*9*16384 = 442368` common steps, exceeding its ENTIRE418,222 query ledger.
It cannot reach condition1. Seeds in E only make this inequality conservative.
The policy overhead must still be recorded; skipping rebuild is not literally
zero overhead or a guaranteed CPU win.

Condition2 does NOT certify future savings. Expensive then cheap queries or an
ineffective order can lose money; late expensive queries may not amortize.
Past row counts cannot guarantee positive speedup. Retain the whole result.

## Cheapest plain refinement worth source-designing

Before general Morton, consider one secondary-axis stable sort inside each
fixed4096-row block of V1's longest-axis ordered rows. Blocks contain up to16
existing leaves. Choose the secondary axis once from the remaining root spans,
with deterministic lowest-axis ties; use no fitted basis. Preserve primary
blocks, allowing rows to move BETWEEN leaves within a block. Sorting inside
individual256-row leaves cannot change their boxes and is useless here.

This may tighten two-coordinate patches without a 3D key or per-query shear,
but leaves the third dimension broad. Cancellation, mixed panels and plateaus
remain worst cases. Declare stable tie ordering; extrema use minimum ORIGINAL
raw ID, never first sorted position.

Reuse existing root extrema/seeds and immutable point snapshot. Rebuild world
boxes/minimum-raw metadata from original points after the permutation; do not
reuse stale boxes. Source-design costs: at most12S additional sort_work, a
secondary-coordinate gather, row/permutation copies,6S leaf reductions and
parent unions. A common-work reservation around24S plus bounded node/metadata
work is a feasibility target, NOT an audited operation count. For fully costly
early queries this is below the available nine-query budget even with the2x
margin; exact source must establish K before choosing this experiment.

Chunk local sorting to4096 rows. Retain old and new row arrays/trees during
construction, charging both plus sort/gather scratch and bytes-backed copy
overlap. Do not resnapshot24N raw bytes or use per-row Python objects. Atomically
select the new internal index only after complete validation. A late build or
query failure clears the WHOLE public batch with all attempted costs retained;
do not silently salvage V1 answers after exhausting an aggregate cap.

## Next decision

Source feasibility first: derive exact K, sort work and peak coexistence bytes
for this cheap refinement and the eligibility bookkeeping. If no affordable
implementation fits unchanged caps, stop; no experiment is necessary. Only if
it does, propose a new standalone gate with the same four consumed synthetic
populations, full constructor plus all12 queries, unchanged exact-answer/tie
requirements and frozen efficacy rule. Include a narrow lazy-policy reach and
late-failure control; no huge new matrix, context integration or physical run.
