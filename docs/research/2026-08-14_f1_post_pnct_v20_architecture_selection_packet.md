# Sherman QC F1 — post-PNCT V20 architecture-selection packet

## Decision requested

Select exactly one smallest, bounded, source-only architecture to implement and
execute next, or issue a formal source-only research STOP. This is an
action-selection request, not permission for retrospective threshold tuning.

The next architecture must be structurally different from every terminal
family below and must include a fast falsification gate plus mandatory
native-resolution visual review. If the evidence no longer supports a
scientifically identifiable scan-only architecture, say so explicitly and
name the smallest new sensing capability required.

## Objective and retained baseline

- Objective: improve blind scan-only bend-line generalization/F1.
- Retained baseline: V16 unified route, frozen and unchanged.
- Locked 35-scan result: TP 132 / FP 60 / FN 44, precision 0.6875, recall
  0.7500, F1 0.717391, 4/35 exact scans.
- Frozen held-out U02-U05: 15 TP / 0 FP / 0 FN, byte-identical.
- Runtime/main and production remain closed.

## V20 PNCT execution — terminal failure

Authorized architecture: `PROJECTIVE_NORMAL_CHORD_TRANSPORT_V1`.

The implementation, fixed 64+192*N sidecar, deterministic fixtures, tests,
one-shot source-free runner, and 4096px audit renderer were completed in an
isolated research lane. No gold, labels, expected counts, CAD, filenames,
identity, benchmark feedback, or real-scan results entered the worker.

One source-free execution produced:

- SF01 fragmented 90-degree offset fold: PASS.
- SF02 cropped offset fold: PASS.
- SF03 fragmented 60-degree / 6s offset fold: PASS.
- SF04 mandatory noncircular C2 offset fold: FAIL.
- Nine named aliases: zero activations, PASS.
- Replay: PASS.
- Fixed cache layout: PASS.
- Source-free P95: 0.1863 seconds, PASS.
- Peak process RSS: 43,384,832 bytes, PASS.
- Metamorphic full-byte identity: FAIL for axis/seed reversal and selected
  rigid transforms.

The required 4096px visual audit of SF04 was manually inspected. It is a
genuine two-skin constant-offset noncircular fold. Where both traces remain,
the short normal chords have stable ~6s thickness and the midpoint profile is
smooth and noncollapsed. Deterministic 20% mesh deletion removes different
phase fragments across axial stations; only S02, S05, and S08 retain full
support. Eight stations fail frozen owned-bin coverage. This matches PNCT's
predeclared expected failure: the causal constant-thickness field exists but
the opposite skin is not observed often enough. Per the ruling, 7/10 and 8/11
cannot be lowered and no V20b, fixture replacement, or rerun is authorized.

The standard-fold activation visual was also inspected and is physically
clean: chords join the two offset skins, the midpoint sweep is noncollapsed,
both tails continue, and no cap/wall/opening/seam/cycle/unrelated-panel or
three-layer alias is visible.

Therefore PNCT is terminally retired before U02, C017, the locked 35 scans, or
held-out evaluation.

## Terminal architecture families that cannot be renamed or rescued

The following have already failed frozen source-only or real-scan gates:

- local plane fragment / cross-parent rung family;
- Cubical IIP / immutable cubical persistence family;
- SDRA, AMHC, CFAC, LSSC, LEPG source-only geometry families;
- V17 adjacency hysteresis;
- V18 paired-skin ribbon ownership / exact mate identity;
- V19 persistent tube component parity / topology hash canonicalization;
- V20 projective normal chord transport / two-skin constant-offset coverage.

No next proposal may be a threshold adjustment, another canonical hash,
another exact face mate, another circle/focal fit, or PNCT with altered station
or coverage requirements.

## Most important empirical boundary

V20 separates two facts:

1. Constant-offset chord transport is a physically appropriate bend feature.
2. It is not reliably observable in fragmented source meshes when it requires
   both skins at the same station and normal phase.

A next scan-only representation therefore needs either:

- a one-sided observable that can infer a fold from a single surviving skin
  without degenerating into a free-edge/rim/weld alias; or
- a cross-station evidence transport mechanism that remains identifiable and
  bounded without lowering per-station coverage post hoc; or
- a principled argument that neither is identifiable from current scans and a
  formal source-only STOP with the smallest capability opening.

## Required response

Provide:

1. **RULING** — one architecture name and exact authorization, or formal STOP.
2. **RESEARCH BASIS** — published machinery versus Sherman-specific novelty,
   grounded in primary research where possible.
3. **CAUSAL FEATURE** — why the observable identifies a bend and rejects caps,
   openings, rims, seams, welds, cycles, nearby panels, and duplicates.
4. **EXACT SPECIFICATION** — source-only inputs, eligibility, equations,
   deterministic tie rules, fixed serialization, bounded work/memory/latency,
   insertion point, NMS behavior, and prohibited information.
5. **FAST FALSIFICATION GATES** — source-free positives/negatives,
   metamorphic byte identity, replay, and 4096px visual requirements. Test and
   rule out quickly; no tuning or rerun rescue.
6. **REAL-SCAN GATES** — only after source-free pass: label-closed U02 no-op,
   C017 exactly-one activation with manual visual inspection, then frozen
   locked-35 and held-out gates.
7. **EXPECTED FAILURE MODE** — the single most diagnostic visual and numerical
   signature that terminates the architecture.

Gold, labels, CAD, expected counts, filename/identity, runtime/main,
production, benchmark feedback, and post-hoc rescue tuning remain closed unless
you explicitly argue that the smallest scientifically necessary capability
opening is required.
