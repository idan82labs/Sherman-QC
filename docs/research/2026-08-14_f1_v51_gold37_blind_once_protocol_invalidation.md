# V51 Gold37 blind-once protocol invalidation

Date: 2026-08-14  
Status: terminal protocol invalidation; no V51 Gold37 detector execution or score

## Outcome

The authorized token `V51_GOLD37_BLIND_ONCE_LOCKED` was consumed invalidly before detector invocation.

During the precommit/harness audit, an operator command parsed `sealed_gold_manifest.json` and printed the beginning of the JSON object. The printed material included per-case `manual_bends`. This happened before any V51 detector artifact was created or sealed.

The GPT-5.6 Pro ruling states that opening any Gold37 truth consumes the single authorized evaluation and that a post-truth-opening failure cannot be rerun under the V51 identity. Therefore:

- V51 detector cases executed: **0/37**.
- V51 replay cases executed: **0/37**.
- V51 Gold37 score produced: **no**.
- V51 semantic pass/fail conclusion: **not available**.
- Evaluation validity: **invalid**.
- Replacement V51 Gold37 execution: **not authorized and will not be attempted**.
- Runtime/main and production: **remain closed**.

## Bound evidence

- Gold37 inference manifest SHA-256: `bb06613a96a063463bac951cd9a0aa3a31c7c0ce2b894ddab07261abfcae5f66`
- Gold37 truth manifest SHA-256: `276fffd42db78629fe3930545f5a1e046fa45e64ea1eb9dc83bcc5916cca6f8b`
- Verbatim Pro ruling SHA-256: `f467b04066bd72c6e4a55994ae233e5dfdcf924540fb4affd53a4afbfa70bb33`
- Repository HEAD at incident recording: `5df325e33e9883353cbdefaed107742ba5c406f1`
- Incident event: `output/research/f1-v51-gold37-blind-once-20260814/invocation_events.jsonl`

## Scientific consequence

No V51 Gold37 performance claim can be made. The prior P01–P07 evidence remains hypothesis-generating and source-only, but it does not become formal locked F1 evidence. V51 must not be rescued, rerun, or relabeled under a successor suffix after this protocol breach.

The next permissible action is a fresh architecture/evaluation-design ruling that treats V51 Gold37 as unavailable and preserves the exposed Gold37 corpus as development/diagnostic evidence only.
