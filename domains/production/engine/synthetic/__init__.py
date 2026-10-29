"""E2: synthetic sheet-metal suite and virtual ZEISS-like scanner.

Builds parametric multi-bend parts of seven named families as **real OCP
STEP BReps** (planar panels, true cylindrical bends, holes, notches) for any
stage of the tagged bending order, the matching flat DXF, a recipe draft and
finalized ``PartRecipeV1``, a virtual robotic structured-light scan (PLY),
and exact analytic truth per bend and characteristic. See
``docs/production/ENGINE_SYNTHETIC.md`` and plan §4 E2 / §10 amendments 3-5.

Modules:

- :mod:`.geometry` -- flat blank, exact fold kinematics, loci, conventions
- :mod:`.analytic` -- OML flange / backgauge lengths, bend-line offsets
- :mod:`.families` -- the seven families and their seeded samplers
- :mod:`.injection` -- scenarios and injected errors (as-built model)
- :mod:`.brep` -- OCP pieces, fused solid, deterministic STEP, tessellation
- :mod:`.flat` -- flat DXF (BEND layer + notes) and synthetic drawing PDF
- :mod:`.scanner` -- ScannerModel, scan plans, raycast scanner
- :mod:`.symmetry` -- truth-side self-registration symmetry test
- :mod:`.truth` -- ``truth.json`` assembly
- :mod:`.recipe` -- recipe draft + finalized recipe via the C01 builder
- :mod:`.cases` -- seed sets, tiers, per-case generator, manifest
- :mod:`.cli` -- ``python -m domains.production.engine.synthetic ...``

Independence: this package imports nothing from E1 (``engine.compile``) or E3
(``engine.inspect``); it uses C01 contracts and the C01 recipe builder only.
Truth files (``truth.json``, ``truth_labels.npz``) and ``case.json`` must
never be passed to E3.

Blindness (amendment 4): recipe-side files carry opaque SKUs and names only,
scan point order is random, and the recipe's inspection stage is drawn by
policy. Scans use the scanner model's native spacing (median nearest-neighbour
distance) unless a tier records an explicit point-count override. QUAL seeds
need ``allow_qual=True`` (final G1 run only, amendment 4c).
"""
from __future__ import annotations

from .families import FAMILIES, FeatureRequest, build_family
from .geometry import FoldedPart, FormState, PartDesign
from .scanner import SCAN_PLANS, ScannerModel, ScanPlan

__all__ = [
    "FAMILIES",
    "FeatureRequest",
    "build_family",
    "FoldedPart",
    "FormState",
    "PartDesign",
    "ScannerModel",
    "ScanPlan",
    "SCAN_PLANS",
]
