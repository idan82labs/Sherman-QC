# Sherman QC F1 — post-CIBEC dual-lane preregistration V1

Date: 2026-08-25

Status: **preregistered plan, not execution authorization**

## Objective

Measure trustworthy zero-shot progress on genuinely untouched physical morphology while keeping scan-only and CAD-assisted claims separate. Every candidate receives a Generalization Progress Indicator (GPI) after its single frozen evaluation.

## Preconditions

No candidate may see or run on the new physical corpus until all of these are true:

1. the source-free contract soak has a strictly validated three-hour PASS;
2. two distinct authorities accept the exact CIBEC review hash;
3. signed CIBEC source-free qualification passes;
4. the 13 consumed physical cases pass independent annotation, native-resolution visual, CAD grouping, replay, latency, memory, job, manifest, and leakage gates;
5. metadata-only inventory confirms at least 30 scans and 15 opaque morphology groups;
6. blind allocation and annotation finish under separated custody, with truth encrypted and unavailable to implementation.

## Frozen baselines

- Scan-only: retained V51 behavior, not V52.
- CAD-assisted: retained scan plus scan-supported legacy CAD union, historical consumed-slice F1 0.4580152672.
- Historical metrics are context only. The untouched comparison is rescored from the independently frozen CIBEC truth.

## Candidate S — scan-only event ownership and flat-face veto

Hypothesis: most current scan-only false positives arise because several candidate routes claim one physical crease region and because long lines cross locally flat support. A deterministic ownership/consolidation stage followed by a source-only flat-face veto can improve precision without reducing recall or minimum-family recall.

Boundary:

- input is only raw-scan-derived compact candidate IR plus permitted source-only support features;
- no STEP, CAD axes, overlay transforms, filenames, identities, family membership, expected counts, labels, truth, or benchmark feedback;
- no new candidate generator is funded;
- the retained V51 candidate population is immutable; the intervention may only consolidate ownership, keep, or veto;
- constants and tie-breaking must be fixed on source-free fixtures and consumed development evidence before the new corpus is visible;
- one primary run and one byte-identical replay; no rescue rerun or threshold change.

## Candidate C — CAD event contraction then scan localization

Hypothesis: CAD recall is useful but native STEP primitives overcount manufacturing events. Independently grouping native primitives into one CIBEC inspection event before scan localization can improve precision while retaining CAD's recall benefit.

Boundary:

- CAD and scan overlay are allowed only in the CAD-assisted lane;
- the independently frozen primary scan truth cannot be changed by CAD correspondence;
- many native STEP faces/axes may contract to one inspection event only under the signed CIBEC mapping;
- a contracted event yields one displayed line localized to observed scan evidence; CAD-only unobserved events remain explanatory, not automatic detections;
- untrusted registration produces a controlled abstention with rescan guidance;
- no expected count, filename, identity, truth, or benchmark feedback;
- constants and tie-breaking freeze before the new corpus is visible;
- one primary run and one byte-identical replay; no rescue rerun or threshold change.

## Untouched evaluation

Each lane uses its own matched frozen baseline and identical opaque case order. Jobs run sequentially with one compute-heavy bend process at a time. The scorer opens truth only after all primary and replay jobs have completed and receipts are frozen.

Required report fields:

- precision, recall, F1;
- exact-count accuracy;
- exact localized target-set accuracy;
- marker admissibility;
- minimum recall across the 15 preregistered morphology groups;
- per-family deltas without revealing family identity to implementation;
- abstention rate and decision coverage;
- deterministic replay rate;
- P95 wall time, peak process-tree RSS, and job-success rate;
- native-resolution visual verdict for every changed case.

## GPI ruling after each candidate

- **GPI-A — verified physical progress:** untouched 30-scan/15-family evidence, strict F1 gain, no exact-count or minimum-family regression, all accuracy/operations/visual gates pass.
- **GPI-B — positive physical indicator:** clean untouched strict F1 gain with no protected-metric regression, but scale, absolute accuracy, or operations remain incomplete.
- **GPI-C — no generalization evidence:** source-free, synthetic, consumed, non-held-out, or no strict untouched F1 gain.
- **GPI-D — negative physical indicator:** clean untouched regression or native visual failure.
- **GPI-F — non-evaluable:** leakage, cross-lane comparison, unmatched baseline, incomplete visual evidence, broken denominator, or invalid protocol.

Promotion is not implied by GPI-A. Autonomous production remains separately gated by precision and recall at least 0.90, exact-count accuracy at least 0.90, marker admissibility at least 0.99, minimum-family recall at least 0.80, replay 1.0, P95 at most 300 seconds, process-tree RSS below 16 GiB, job success at least 0.995, and a native-resolution visual PASS.

## Retain/retire rule

Retain a candidate only on a strict untouched F1 gain with no regression in exact-count accuracy or minimum-family recall and no visual/operational failure. Otherwise retire that version after its single run. Do not reinterpret a GPI-C engineering improvement as physical generalization.
