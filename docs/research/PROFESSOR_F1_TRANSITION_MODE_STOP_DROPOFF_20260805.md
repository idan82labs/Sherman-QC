# Sherman QC F1 — transition-mode diagnostic STOP follow-up

You are GPT-5.6 Pro extended-thinking acting as the research reviewer for the
Sherman QC scan-only, zero-shot bend-inference program. Continue directly from
`02_PRIOR_PROFESSOR_RESPONSE.md`. This is a file-upload research submission,
not an email. Inspect every uploaded file before answering.

## Decision already made by the evidence

`TRANSITION_MODE_PROBE_V1` is a formal STOP:

- 938 plane pairs;
- 10,457 multimodal parents;
- 1,994 transition components;
- only 2 hard geometry certificates;
- 0 recovered previously unrepresented gold bends;
- 0 of the required 8 independent family groups recovered;
- 0 of 33 frozen A0 false predictions pass the certificate;
- post-lock first-failure counts are stage 1 = 67, stage 4 = 3, stage 7 = 70;
- 0 birth columns emitted and 0 predictions changed;
- C001 peak RSS = 0.460052 GiB at the 200,000-point bound;
- C001 replay is byte-identical, SHA-256
  `36a10af46fc295f437cb95522117605675b671066632af9ab4a97c7f172d8b28`.

Treat those as OBSERVED. Do not rescue this probe by relaxing its thresholds.
Do not reinterpret the two certificates as success: the post-lock attribution
shows zero gold recovery.

## Hard prohibitions for your recommendation

Do **not** authorize or design any of the following yet:

- A1 parent/child ownership integration;
- any emitted birth column or birth-family solver wiring;
- learning, fitted potentials, or classifier training;
- threshold sweeps or case/family-specific tuning;
- expected bend counts, filename/identity features, or metadata at inference;
- runtime/main promotion.

The current prediction path remains frozen A0. Gold may be used only after an
inference-safe probe and its output hashes are locked.

## Your single task

Choose the **next smallest falsifiable representation or plane-extraction
tranche**. We need one bounded experiment, not a roadmap of several speculative
features.

First, decide whether the 67 stage-1 misses are best tested as:

1. missing/fragmented incident-plane extraction or coverage; or
2. failure of the V2 spatial-parent/projective-normal-mode primitive to expose
   the relevant transition support.

Then provide an implementation-level protocol containing:

1. the exact hypothesis and counterfactual that would falsify it;
2. the smallest new or changed representation tables and every field;
3. deterministic construction pseudocode, canonical ordering/hashing, and hard
   per-case caps under the 200,000-point input bound;
4. inference-safe census records and first-failure counters, with no source
   points or unbounded point-index serialization;
5. the exact post-lock gold-matching procedure;
6. preregistered GO/STOP thresholds, including minimum recovered
   unrepresented gold objects and independent family groups and a false-certificate
   ceiling against the frozen 33 A0 false predictions;
7. unit, property, schema, memory, and replay tests;
8. an ordered file-by-file patch plan and the single milestone at which we must
   rerun all Gold37 cases and stop again.

Label every substantive statement **OBSERVED**, **INFERENCE**, or **UNKNOWN**.
When the uploaded evidence cannot distinguish alternatives, say UNKNOWN and
design the smallest discriminating probe. Do not recommend broad literature
survey, generic neural learning, or a collection of parallel heuristics.

The desired response is decisive: select one tranche, specify it tightly enough
to implement without interpretation, and state exactly what result would permit
the next professor packet. It must remain research-only even if it passes.
