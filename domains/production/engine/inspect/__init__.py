"""Per-scan inspection engine (E3): one scan + a CompiledPart + its recipe -> ShopfloorInspectionResultV1.

Entry points:

- CLI: ``python -m domains.production.engine.inspect --compiled DIR|JSON --recipe R.json
  --scan SCAN.ply|.stl --out OUT --scan-sha256 HEX [--compiled-sha256 HEX] [--recipe-sha256 HEX]
  [--context C.json] [--declared-stage K --declared-stage-source SIDECAR|OPERATOR|API]
  [--unit-factor 1|25.4|1000] [--time-budget-s 170] [--engine-bundle-id ID] [--config CFG.json]``
  (contract: ``contracts/production/ENGINE_IO.md`` §4; exit 0 result, 2 invalid input, 3 internal).
- Library: :func:`load_inputs` (ENGINE_IO §4 integrity checks, raises
  :class:`InspectInputError`), :func:`inspect_scan` (the pipeline, returns
  :class:`InspectOutcome` and writes the §4.1 output directory) and
  :func:`load_config` / :class:`InspectConfig` (the pinned, hashed configuration).

Pipeline (plan §3.2 with §10 amendments 4, 5 and 9), one module each:
:mod:`.scan_io` (PLY/STL by content, explicit unit factor only) -> :mod:`.cloud`
(voxel cloud, normals, neighbour graph) -> :mod:`.registration` (stage-independent
datum registration from the datum plane, its free edges and its holes; top-K
proper and mirrored poses; in-plane degeneracy) -> :mod:`.articulated` (flange
tree walk, hinge Hough theta search from flat to sweep + 25 deg seeded within
``Ri + 2t + 5 mm`` of the hinge, region-grown planes) -> :mod:`.decode` (per-bend
states with the order as a soft prior, stage model) -> :mod:`.metrology` (5-11
section stations: angle, OML sharp, bend-line position, flange lengths,
backgauge, U) -> :mod:`.result` (contract rows via the factories) ->
:mod:`.engine` (orchestration, validity, heatmap, creases, outputs).

The engine never reads the scan's file name, metadata counts or the
``--context`` it copies (plan §10 amendment 4). Design notes and the measured
G1 numbers: ``docs/production/ENGINE_INSPECT.md``.
"""
from __future__ import annotations

from domains.production.engine.inspect.config import ALGORITHM_VERSIONS, ENGINE_VERSION, ConfigError, InspectConfig, load_config
from domains.production.engine.inspect.engine import InspectInternalError, InspectOptions, InspectOutcome, inspect_scan
from domains.production.engine.inspect.part import INPUT_ERROR_CODES, InspectInputError, Inputs, PartView, load_inputs

__all__ = [
    "ALGORITHM_VERSIONS",
    "ENGINE_VERSION",
    "INPUT_ERROR_CODES",
    "ConfigError",
    "InspectConfig",
    "InspectInputError",
    "InspectInternalError",
    "InspectOptions",
    "InspectOutcome",
    "Inputs",
    "PartView",
    "inspect_scan",
    "load_config",
    "load_inputs",
]
