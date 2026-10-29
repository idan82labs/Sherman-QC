# Luna-max V11 single-scan checkpoint — 2026-09-14

## Outcome

The bounded one-scan objective is met by a fresh reproduction of the existing
research candidate on consumed Gold11 case `CG4BDD1371B6` / part `37361005`.
The frozen 11-case V11 attempt remains terminal and was not reopened.

Fresh artifacts:
`docs/research/gold11_v11_single_scan_luna_max_v1/CG4BDD1371B6/` and the
metric-complete successor
`docs/research/gold11_v11_single_scan_luna_max_v2/CG4BDD1371B6/`.

| quantity | result |
| --- | ---: |
| native STEP axes | 4 |
| scan-supported predictions | 4 |
| Gold references | 4 |
| TP / FP / FN | 4 / 0 / 0 |
| precision / recall / F1 | 1.0 / 1.0 / 1.0 |
| bend-count capture | 100% |

The unchanged one-to-one matcher is used. Pairwise axis errors are
0.074–0.099 degrees and transverse offsets are 0.498–3.779 mm. The visual
artifact has deterministic oblique and plan views with four finite coloured
markers and scan-support points. `RESULT.json` carries scan-derived endpoints,
support distances and per-line spans; STEP radii/sweeps remain explicitly
labelled nominal reference metadata.

## Reproduction and verification

The first attempted command used `python -I -S` and failed before inference
because `-S` disables the venv Matplotlib dependency. It is retained as an
execution-envelope failure, not a geometry result. The valid reproduction was:

```bash
cd /Users/idantal/ShermanQC/FullProject
OMP_NUM_THREADS=1 .venv/bin/python -B \
  scripts/run_gold11_v11_single_case_candidate_v1.py \
  --output docs/research/gold11_v11_single_scan_luna_max_v1/CG4BDD1371B6
```

The V1 command exited 0 and returned four predictions, TP=4, capture=1.0 and
F1=1.0. The V2 wrapper preserves that inference and adds explicit precision
and recall fields:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B \
  scripts/run_gold11_v11_single_scan_luna_max_v2.py
```

It exited 0 with TP=4, FP=0, FN=0, precision=1.0, recall=1.0, F1=1.0 and
100% bend-count capture. Focused contract tests then passed:

```text
4 passed in 1.14s
```

The metric-complete V2 result SHA256 is
`6cac4d3f4cc3945750418f869c628d218ec0daf13a4768e38f07b3380aad3c27`; its
source bindings and visual SHA256 are recorded in that artifact's
`CHECKSUMS.json`. The V2 visual SHA256 is
`024cc476b3221e885c831b2b3c0c93d6bfbbd81af3da37a8f7dea888a7ab389e`, matching
the prior V1 visual exactly. The original V1 result remains preserved with
its own SHA256 (`2de9d340…`).

## Interpretation and next gate

This is a research-only CAD-assisted recovery seam: native STEP supplies a
finite axis proposal, the retained terminal V11 context supplies the pose, and
raw scan support supplies finite endpoints and support statistics. Gold is read
only after inference for scoring. The result does not establish manufacturing
event identity, calibrated units, measured bend radius/deviation, unseen-family
generalization, historical-score comparability, production use or floor release.

Promotion requires a separately frozen multi-part replay with independent
source/PRE/POST audits and adversarial checks for wrong poses, unsupported
trims, duplicate axes, partial coverage and nominal-vs-measured leakage. The
physical V11 full-corpus zero-output attempt remains preserved as a negative
result; no score claim is made for that attempt.
