# Gold11 V11 single-part recovery candidate — 2026-09-14

## Decision

The frozen `gold11_mesh_development_v11_attempt` remains terminal and untouched.
This is a separately versioned research candidate for `CG4BDD1371B6` / part
`37361005`, selected because the consumed Gold reference contains four bends and
the retained V11 control reached a completed context but emitted zero lines.

The candidate runs the maintained V11 policy constructor and the same validated
raw scan/CAD bytes, then adds one controlled recovery seam after the V11
measurement stage:

1. Extract finite bend axes from the native STEP topology.
2. Project each axis through V11's retained completed-context pose.
3. Trim each projected line to its longest contiguous raw-scan-supported run
   using a fixed 3 mm support radius, 201 samples, and the existing frame-
   integrity thresholds.
4. Publish only the scan-derived endpoints and support statistics. STEP radii
   and sweeps are retained as explicitly labelled nominal reference metadata;
   they are not measured answers. No Gold, expected count, case identity, or
   scorer data enters inference.

This is a targeted CAD-assisted recovery, not a claim that the V11 scan-only
measurement kernel has solved arbitrary bends. The output is research-only,
with observed bend count and production/floor release deliberately unknown.

## Reproducible run

```bash
cd /Users/idantal/ShermanQC/FullProject
OMP_NUM_THREADS=1 .venv/bin/python -u \
  scripts/run_gold11_v11_single_case_candidate_v1.py
```

The script binds the approved scan and CAD hashes, confirms that the retained
V11 control is the terminal zero-output case, and opens Gold only at the
post-inference scoring seam.

## Result

Artifact directory:
`docs/research/gold11_v11_single_case_candidate_v1/CG4BDD1371B6/`

| quantity | result |
|---|---:|
| native axes proposed | 4 |
| raw-scan-supported predictions | 4 |
| Gold bends | 4 |
| true positives | 4 |
| false positives / false negatives | 0 / 0 |
| bend-count capture | 100% |
| precision / recall / F1 | 1.0 / 1.0 / 1.0 |

The unchanged one-to-one matcher reports all four matches.  The pairwise
transverse offsets are 0.498–3.779 mm, axis-angle errors are 0.074–0.099°,
and every prediction has full finite-span support.  These are same-input
consumed-development results; they are not unseen-family generalization and do
not prove that historical Gold11 F1 0.4580152672 was beaten.

## Visual review

`VISUAL.png` contains deterministic oblique and plan views of the raw scan
decimation with four labelled, coloured finite lines.  `RESULT.json` contains
the per-line span and scan-support distance data.  The line spans are 150.0,
203.9, and 257.7 mm; median scan distances are 0.539–1.314 mm and p90 values
are 0.693–1.851 mm.  These distances describe support to the targeted axis,
not bend radius or calibrated deviation.

## Promotion gate

Do not promote this seam to the default engine from this one case.  The next
gate is an independently frozen, no-Gold-tuning replay on additional consumed
cases (then an unseen held-out family when authorized), with explicit checks
for false topology axes, duplicate lines, partial coverage, wrong poses,
unsupported trims, and measured-vs-nominal field separation.  Until that audit
passes, this candidate remains an opt-in research artifact.
