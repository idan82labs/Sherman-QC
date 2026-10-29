# Angle Truth Control Set

Date: 2026-04-04

## Purpose

Define the completed-part control pack for angle validation against:

1. drawing PDF
2. CAD bend inventory / nominal angles
3. scan-measured bend angles

This is the correct acceptance standard for angle work. `23553000` remains the main target for the short `120°` wing-bend bug, but future angle patches must also hold on the completed-part controls below.

## Selection rule

Use completed/full scans only. Prefer parts that satisfy all or most of:

- drawing PDF available
- full scan available
- low to moderate existing angle MAE
- not already dominated by known hybrid semantics or severe unresolved extraction problems

## Recommended control pack

### Primary target

- `23553000`
  - reason: only strong current `120°` short-wing family in the benchmark slice
  - role: main bug target, not the sole validator

### Stable controls

- `15651004`
  - full-scan countable angle MAE: `0.059`
  - role: single-bend sanity control

- `10839000`
  - full-scan countable angle MAE: `0.253`
  - role: strong multi-bend `90°` control

- `37361005`
  - full-scan countable angle MAE: `0.363`
  - role: moderate multi-bend control with warnings but no fails

### Secondary controls

- `11979003`
  - full-scan countable angle MAE: about `1.97–2.00`
  - role: acceptable but weaker control because existing angle error is already nontrivial

- `48991006`
  - full-scan countable angle MAE: `1.353`
  - role: additional multi-bend control if drawing mapping is clear

- `48963008`
  - full-scan countable angle MAE: `1.993`
  - role: small control set, limited but useful

### Hybrid guard

- `28870000`
  - full-scan countable angle MAE: `6.405`
  - role: semantic guard, not angle-accuracy control
  - reason: protects against collapsing rolled/process geometry into fake discrete bends

### Hard non-control families

These are still worth auditing, but should not be used as “good completed part” controls yet:

- `49125000`
- `47479000`
- `47266000`
- `49024000`
- `23553000`

Reason:
- current full-scan angle error is too high
- using them as controls would blur whether a patch is improving or just moving error around

## PDF availability found

Available drawing PDFs in current workspace / downloads for the main control candidates:

- `10839000`: `/Users/idant/Downloads/שרטוט 10839000.pdf`
- `11979003`: `/Users/idant/Downloads/שרטוט  11979003.PDF`
- `15651004`: `/Users/idant/Downloads/שרטוט  15651004.PDF`
- `23553000`: drawing already used earlier in the investigation
- `28870000`: `/Users/idant/Downloads/שרטוט החלק 28870000.pdf`
- `37361005`: `/Users/idant/Downloads/שרטוט  37361005.PDF`
- `40500000`: `/Users/idant/Downloads/שרטוט   40500000 1.PDF`
- `48963008`: `/Users/idant/Downloads/שרטוט 48963008.pdf`
- `48991006`: `/Users/idant/Downloads/48991006_A 4.pdf`
- `49024000`: `/Users/idant/Downloads/שרטוט 49024000.PDF`
- `49125000`: `/Users/idant/Downloads/49125000_A00.pdf`

## Immediate engineering rule

For any future angle patch:

1. validate on `23553000`
2. validate on stable completed controls:
   - `15651004`
   - `10839000`
   - `37361005`
3. validate on semantic guard:
   - `28870000`
4. only then consider broader benchmark reruns

## Current read

This means the next `23553000` short-wing bend patches should be evaluated against:

- bug target:
  - `23553000`
- accuracy controls:
  - `15651004`
  - `10839000`
  - `37361005`
- semantic guard:
  - `28870000`

That is the smallest control pack that is both credible and efficient.

## Important limitation found on 2026-04-05

A full completed-part scan of the clean benchmark slice shows:

- the short `110–130°` countable bend family appears only on `23553000`
- the stable completed controls (`15651004`, `10839000`, `37361005`) are dominated by `90°` bend families
- `28870000` remains a semantic guard with `22–25°` countable bends plus rolled/process geometry

Implication:

- the control pack is still correct, but not for the same reason on every part
- `15651004`, `10839000`, and `37361005` are regression guards for generic angle behavior
- `28870000` is a semantic guard against hybrid misclassification
- only `23553000` is a like-for-like validator for the short `120°` wing-bend path

Engineering rule:

For short-wing `120°` work, do not over-claim validation from the control pack alone. A patch is only well-validated if:

1. `23553000` improves on the relevant bends
2. the stable controls remain neutral
3. `28870000` semantics remain intact

## Safe execution rule on local hardware

The case runner can consume multiple gigabytes of RSS per geometry evaluation on Apple Silicon. On this `16 GB` machine:

- do not run multiple bend-case evaluations in parallel
- run one case at a time
- cap math libraries to one thread:
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `VECLIB_MAXIMUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`

Parallel case execution is not acceptable for local validation on this hardware.
