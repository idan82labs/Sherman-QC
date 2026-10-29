# Perturbation-aware two-plane search depth: prospective successor

Source/retained-evidence proposal. The isolated V2 scheduler now has an audited
80PASS source gate; that proves the tested small scheduling contracts only.
Controlled engine integration is awaiting its69-outcome gate and exact PRE.
No connected recovery is established. The composite connected V1 attempt stays
terminal RED; preserve the old schedule, source archive and negative outcomes.

## Diagnosed allocation bottleneck

Main inspected retained L-R; independent `cad_support_lineage` inspected L-A.
Both complete prefilters retain zero passing starts; neither reaches ICP or
scan measurement. This is not a CPU cap or a numerical/query failure.

L-R now has17 qualified scan planes,3,204 proposed hypotheses and6,440 assigned
support checks. Its selected broad-flange-like pair0/13 receives only the four
ordered assignments11/55,55/11,44/233,233/44. The CAD flange11/33 pairing is not
scheduled for this source pair. There are21 angle-compatible unordered CAD
pairs;19 are omitted. Only128 of252 available ordered matches are reserved,
leaving124 unused; total joint records4,224 of8,192.

L-A likewise has17 qualified planes,1,796 proposed hypotheses and3,608 support
checks. Selected broad-flange-like pair0/2 receives11/44,44/11,44/211,211/44,
again omitting flange11/33 among19 omitted compatible pairs. Only72 of254
available ordered matches are used, leaving182; joint records2,368 of8,192.
L-A has476 individually passing assigned supports, but no pose whose complete
assigned-support set passes. Individual passing supports cannot be combined
across incompatible poses to manufacture a registration.

The source explains the allocation: V1 `_window` unconditionally selects at
most two unordered CAD pairs per selected scan pair and never revisits that
pair. Its repeated-layout fairness clock is shared across different source
pairs, so it proves no coverage of the joint source/target correspondence set.
V1 documents that limitation; this is an engineering capability limitation,
not evidence that its declared fairness theorem or finite rejection is wrong.

The physically plausible omitted mappings are inferred from the independently
constructed fixture and retained plane equations for diagnosis only. They must
never enter inference as IDs, rules, seeded poses or reference-informed choices.
An omitted mapping has not been evaluated. Recovery and sole causality remain
unproved, and finite-support/measurement failures may remain after better search.

## Candidate hypothesis

Introduce a separately versioned, opt-in scheduling policy that allocates more
compatible-target depth to perturbation-robust independent scan pairs under the
same joint8,192-hypothesis and1,024-unsigned maxima. Keep both target orders and
all32 reserved context slots per ordered assignment; failed signs, duplicate
poses and absent midpoint slots do not refund budget after evaluation.

Use the existing normal-independence margin as a primary geometric priority,
not tiny nominal singular-value-ratio differences ahead of uncertainty. For
the represented two-row normal matrix, the existing helper computes
`margin = sigma_min - hypot(2*sin(alpha1/2), 2*sin(alpha2/2))`.
The second term bounds row perturbation in Frobenius norm under declared cones;
spectral perturbation is no greater. This motivates ranking robust independent
pairs but is not a scanner calibration, a certified floating-point bound, a
correspondence likelihood or a recall guarantee.
The chord bound assumes exact unit cone-center directions; represented
normalization and computed ranks remain engineering arithmetic. The underlying
additive singular-value inequality is stated in
[Higham, Singular Value Inequalities, Theorem2](https://nhigham.com/2021/05/04/singular-value-inequalities/).
Our use of it as a residual-budget priority is an engineering inference, not a
published recovery guarantee for this scheduler. Plane-derived geometric
descriptors for guiding nonexhaustive coarse correspondence search also have
precedent in [Theiler and Schindler's ETH project](https://prs.igp.ethz.ch/research/completed_projects/automatic_registration_of_point_clouds.html);
their terrestrial-scan results do not establish performance on sheet-metal QC.
Only the explanatory sources were read; no external datasets were downloaded.

A narrower first design preserves the old first breadth pass, then uses its
otherwise unspent reservation capacity for deeper compatible-target enumeration
of already selected pairs, ranked by perturbation margin. L-A's useful pair was
already second in the original order, so ranking alone cannot fix its missing
target assignments. Remaining depth should visit previously omitted target
pairs, not refund failed signs or repeat evaluated matches. Stop before the
unchanged joint reservation cap. Use explicit deterministic partial-window
selection if the last window cannot fit. Reuse exact compatible-layout/index
evidence only where it preserves owned identity and accounting. All unvisited
pairs and omitted targets must remain visible as policy omissions, not geometric
disprovals. Do not label this global or exhaustive search. Residual depth can
still starve lower-ranked pairs, especially with duplicated confident planes;
this tradeoff requires independent adversarial tests. Preserving old proposals
does not prove preserved final outputs: the48-start refinement selection can
change when additional passing groups are introduced.
The independent design review greenlights source drafting only. Cached-layout
decoding/filtering is still new work: precharge retained views/index lists and
bound cached-index visits/selection, or directly select omitted ranks without
materializing millions of entries. A256-match output cap alone does not bound
cache traversal. Preserve separate immutable breadth and residual ledgers.
Proposal inclusion is conditional on complete V2 evaluation: a late memory or
report failure still clears every usable match, so availability is not monotone.

## Required independent checks before any connected run

- Mathematical review of the normal perturbation/conditioning priority, equal
  or near-equal ranks, non-unit input admission and numeric failure boundaries.
- Invented normal/cone declarations representing robust versus high-cone pairs,
  and the converse: confident distractors/duplicated planes hiding lower-ranked
  pairs. This scheduler sees no raw support sizes, curvature or finite extents;
  these source tests cannot qualify broad-versus-tiny physical usefulness.
  Include zero/odd residual capacities and a partial last window. The connected
  packet must separately exercise the actual curved and finite supports.
- Every selected target pair includes both orders, exact slot/unsigned counting,
  no post-outcome refill, complete omitted/unvisited ledgers and whole-result
  clearing on resource or numerical failure.
- CAD/input-order permutations and geometric ties, with any non-invariance
  explicitly retained rather than silently choosing an order that works.
- Existing four constructed curved inputs, thresholds and accuracy/localization
  oracles unchanged; compare against the retained2/4 control. All four remain
  required for this connected capability gate.

Freeze exact candidate/configuration, source closure, outcome denominator,
budgets and failure handling after independent technical review. Additional
target assignments may increase prefilter cost; prospectively justify CPU/wall
without changing any started limits. The separate fast-ICP proposal review does
not solve these zero-start failures and should not be conflated with this test.
No native or physical Gold11 run is released by this proposal.
