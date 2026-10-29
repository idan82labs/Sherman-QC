# PLY versus CAD report — 37361005

23 September 2026 · CG4BDD1371B6 · Research inspection

## Extraction result

Fresh execution recovered **4/4 labelled bends: 100% recall, 100% precision, F1 1.00**, with zero unmatched labels or predictions. This meets the requested >80% extraction criterion on this known consumed case. The method is CAD-guided scan support, not independent PLY-only detection. It proposes STEP bend markers through the retained V11 pose and accepts finite markers supported by raw scan points. The earlier frozen V11 attempt was not rerun or modified.

The unchanged one-to-one matcher allows axis angle ≤5°, transverse offset ≤4 mm, and overlap ≥60% in both directions. This detection score does not mean 100% dimensional accuracy or manufacturing conformance.

## Bend-by-bend analysis

| Bend / reference | Supported span mm | Support | Axis→scan median / P90 mm | Label offset mm / angle ° | Local surface median / P95 mm |
|---|---:|---:|---:|---:|---:|
| B1 / M03 | 257.7 | 100% | 0.539 / 0.748 | 0.848 / 0.074 | 0.203 / 0.705 |
| B2 / M04 | 150.0 | 100% | 0.677 / 0.693 | 3.779 / 0.099 | 0.255 / 1.397 |
| B3 / M02 | 257.7 | 100% | 0.777 / 0.888 | 0.498 / 0.074 | 0.145 / 0.399 |
| B4 / M01 | 203.9 | 100% | 1.314 / 1.851 | 3.395 / 0.085 | 0.189 / 0.674 |

All four CAD references have nominal cylindrical radii **0.8 / 2.3 mm** and nominal sweeps **90° / 90°**. These are paired CAD surface properties. Measured scan bend radius, bend angle, radius error, angle error and tolerance pass/fail are **unavailable** because the retained measurement path did not establish valid flange-plane measurements. The axis-to-label angle above measures marker orientation agreement, not bend angle error.

Support is evaluated at 201 uniformly spaced samples per bend marker within a 3 mm nearest-point gate. All 201 samples on each marker pass, and each marker has one continuous sampled support run. This does not prove continuous physical coverage between samples. Supported lengths are CAD-directed spans retained by the support test, not independently fitted manufacturing lengths. Local surface statistics use raw scan vertices within 3 mm of each finite marker; these neighborhoods can overlap.

B4 has the weakest marker support (median 1.314 mm; P90 1.851 mm), though all sampled positions pass the support gate. B2 has the largest label offset (3.779 mm, close to the 4 mm matching limit) and extends beyond its manually labelled span (150.0 versus 124.9 mm). B4's label offset is 3.395 mm. B1 and B3 have smaller label offsets, 0.848 and 0.498 mm. These observations identify review priorities; they do not establish defects.

![Bend support heatmap](bend_support_heatmap.png)

## Surface-distance heatmap

All **581,263 raw PLY vertices** were transformed using the retained scan-to-CAD rigid pose and compared to **3,312 CAD triangles**. No outliers were removed or alignment optimized for this report. Display uses 40,000 deterministic samples; statistics include every vertex.

| Surface distance statistic | mm |
|---|---:|
| Mean | 0.246 |
| RMS | 0.392 |
| Median | 0.142 |
| P90 | 0.588 |
| P95 | 0.947 |
| P99 | 1.430 |
| Maximum | 2.336 |

![Scan to CAD surface heatmap](surface_heatmap.png)

Colors show unsigned closest-surface distance, not signed material excess/removal, bend-angle error or a calibrated defect measure. The CAD mesher uses 0.10 mm linear and 0.20 rad angular deflection; the approximation error is not certified. Color range ends at 1.430 mm; higher distances share the top color, while full values remain in the data. Statistics are weighted by scan vertex density and include registration error and acquisition artifacts. This scan-to-CAD comparison does not measure unobserved CAD areas or prove whole-part coverage.

## Disposition and traceability

**Bend localization criterion met; dimensional conformance remains unknown.** No manufacturing tolerance or calibrated scan uncertainty was provided. This report preserves the research-only status and does not issue production release. A metrology conclusion requires valid measured bend angles/radii, qualified registration and applicable tolerances.

Inputs: `37361005_01.ply` and `37361005_01.stp`. Both were checked against their established SHA-256 bindings before extraction. Gold labels were consumed only after inference by the unchanged matcher. `analysis.json` includes input bindings, the transform, all matches and local statistics. `surface_distances.npz` contains every transformed scan vertex and its unsigned distance. `extraction/RESULT.json` retains the fresh extraction record. `CHECKSUMS.json` binds the report artifacts and generator.
