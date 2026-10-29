# Sherman QC F1 — CIIP-V1 V4 successful post-execution review

You are GPT-5.6 Pro acting as a senior research professor and systems architect. This is a post-execution result review, not a request for another pre-execution permission cycle. We deliberately ran first and are now giving you the result, because the project owner wants progress through bounded action rather than indefinite theory.

## Objective

Improve Sherman QC's F1 generalization for geometric defect detection without using gold labels, CAD, expected object counts, benchmark feedback, production/runtime integration, replay tuning, or replacement inputs in this tranche.

## What CIIP-V1 is

CIIP-V1 is a frozen, gold-free cubical intrinsic invariant pipeline over the content-selected raw point-cloud source. It canonicalizes the source, constructs a deterministic occupied cubical complex and filtration, emits six canonical evidence tables, and independently rebuilds/validates the same semantics. The protocol includes two ordinary producer roles, an input-order-permuted role, and a separately implemented independent validator.

## V4 authorization and immutability

- Attempt: `CORRECTION_V4`
- Executed exactly once; the claim was consumed before the first worker launch.
- No rerun or replacement input is authorized for this consumed attempt.
- Gold/labels/CAD/expected counts were not accessed.
- Raw source SHA-256: `050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97`
- Embedded execution-lock SHA-256: `3a5547ffeda25726bca42e520d569a19f4222651f5f3c7d9a9384cff33279411`
- Lock file SHA-256: `0bca49178814ac0b11742f4402740c26609c874e97c407e00b34c4f19f7d50a4`
- Claim file SHA-256: `c961258b35079676ea9b3f1a3cf8c171b49c3094509b7571bffaa4f04ba295ae`
- Result file SHA-256: `5ee0cc41fae561529f96ed4241742fdae2138479ae332a6c9d31e9b48707dac6`

## Exact terminal result

`CII_PERSISTENCE_PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET`

`failure: null`

All four sequential roles returned code 0 with empty stdout/stderr and no role failure:

1. `BASE_A` — container SHA-256 `7065b3b77a15435d8654d047d9afb50e954e8997375d94e9a3e1606141477817`; CPU 2,745,548,000 ns; wall 1,818,521,958 ns; peak RSS 186,384,384 bytes; scratch 1,947,752 bytes.
2. `BASE_B` — container SHA-256 `b967e1065fd68a40443ec4a2439c2c5c2b140232472b79a418c7dca6b6ae1752`; CPU 2,711,656,000 ns; wall 1,800,932,875 ns; peak RSS 186,384,384 bytes; scratch 1,947,752 bytes.
3. `INPUT_ORDER_PERMUTED` — container SHA-256 `159c090100c3840cca0630590e6aa8fe749bf5c4e098fa8a2db782ea669b32ed`; CPU 2,752,099,999 ns; wall 1,826,929,625 ns; peak RSS 188,530,688 bytes; scratch 1,947,752 bytes.
4. `INDEPENDENT_VALIDATOR` — container SHA-256 `238311d6bc5dba0567ad03b2565a785c97b9048a3f7e3c64b6920de0dfd6d350`; CPU 2,881,503,000 ns; wall 1,923,887,083 ns; peak RSS 188,530,688 bytes; scratch 1,947,752 bytes.

The supervisor compared the six canonical table byte hashes after each role. All four roles agreed exactly. Whole-container hashes differ because supervisor-owned resource metadata differs; semantic table bytes agree. Input-order permutation invariance and independent semantic reconstruction therefore both completed successfully for this source.

The post-run source-free suite is `182 passed, 1 deliberately deselected` (the deselected test is the real-source content-selection test, excluded from the source-free gate).

## Why V3 stopped and why V4 is distinct

V3 completed the three producer roles and their six-table consensus, then stopped before independent semantic reconstruction because the validator required a nonexistent source key named `canonical_coordinate_multiset_sha256`. The immutable lock correctly used `canonical_coordinate_multiset_digest`. The digest algorithms were already equivalent; this was a control-plane alias-contract defect, not a measured semantic disagreement. The correction removed the phantom generic-key requirement and made the real lock key mandatory and exact. Synthetic regression tests cover matching, missing, and wrong digest cases. V3 remains terminal and was not rerun; V4 used a fresh namespace, lock, claim, native build, output, and scratch path.

## Your task

1. State precisely what this successful V4 result establishes and what it does not establish about F1 generalization.
2. Ground your reasoning in relevant primary research where applicable. Cite stable primary sources for cubical persistent homology, persistence stability, and any representation/generalization claim you rely on. Clearly distinguish literature-backed facts from your inference about this implementation.
3. Select exactly one smallest, bounded, action-oriented next tranche that has the highest information value for improving F1 generalization. Do not require another abstract approval loop before the action. Prefer a test that can hit a real empirical wall quickly.
4. Give an executable specification: permitted inputs, forbidden inputs, exact invariants, outputs/metrics, maximum scope, pass/stop rule, and how the result changes the next decision.
5. Keep gold labels, CAD, expected counts, production/runtime/main, benchmark feedback, and tuning closed unless you explicitly argue that continued closure now makes genuine generalization progress impossible. If you argue for opening anything, identify the smallest opening and why it is scientifically necessary.
6. If CIIP-V1 should be stopped as an architecture family despite this pass, issue a formal STOP and identify the evidence that makes further action irrational. Otherwise authorize one concrete next action.

Be decisive. We need the next empirical move, not a long taxonomy of possibilities.
