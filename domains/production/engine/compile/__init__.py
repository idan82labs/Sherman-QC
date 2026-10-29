"""Onboarding compile (E1): folded STEP + confirmed recipe (+ flat) -> CompiledPartV1.

Entry points:

- CLI: ``python -m domains.production.engine.compile --recipe R.json --cad PART.step
  [--flat FLAT.dxf|FLAT.step] --out DIR [--shop-k 0.33] [--config C.json]``
  (contract: ``contracts/production/ENGINE_IO.md`` §3).
- Library: :func:`compile_part` (same behaviour, returns :class:`CompileResult`),
  :func:`analyze_cad`, :func:`reconcile_draft` and :func:`propose_flat_binding`
  for the wizard / C05 before release, :func:`parse_flat`, and
  :func:`stage_transforms_from_compiled` / :func:`stage_points_from_compiled` for
  stage hypotheses (any formed set, e.g. out of order) at inspect time.

Pipeline (one module each): :mod:`.brep` (OCP reader) -> :mod:`.flange_graph`
(panels, bends, walls, features) -> :mod:`.part_model` (datum, tree, skins,
oriented axes, loci) -> :mod:`.reconcile` (marks vs graph) -> :mod:`.flat` /
:mod:`.flat_binding` (flat parse, binding, K, developed-length QA, and the
Method B fold of a bound flat with its own per-bend strips) ->
:mod:`.nominals` (OML flange and backgauge lengths) -> :mod:`.stages`,
:mod:`.sampling`, :mod:`.symmetry` (stage models S_0..S_N) -> :mod:`.compiler`.
The per-feature UNSUPPORTED rule is frozen in :mod:`.unsupported`. Design notes:
``docs/production/ENGINE_COMPILE.md``.

Importing this package does not import OCP; the CAD is only read on demand.
"""
from __future__ import annotations

from domains.production.engine.compile.api import analyze_cad, load_part_model, propose_flat_binding, reconcile_draft
from domains.production.engine.compile.compiler import (
    COMPILE_ERROR_CODES,
    CompileInputError,
    CompileResult,
    CompileTimeout,
    compile_part,
    sha256_file,
)
from domains.production.engine.compile.config import COMPILER_VERSION, DEFAULT_SHOP_K, CompileConfig, load_config
from domains.production.engine.compile.flat import FlatBendLine, FlatParseError, ParsedFlat, parse_flat
from domains.production.engine.compile.reconcile import Mark
from domains.production.engine.compile.stages import stage_points_from_compiled, stage_transforms_from_compiled
from domains.production.engine.compile.unsupported import UNSUPPORTED_RULE_VERSION

__all__ = [
    "COMPILE_ERROR_CODES",
    "COMPILER_VERSION",
    "DEFAULT_SHOP_K",
    "UNSUPPORTED_RULE_VERSION",
    "CompileConfig",
    "CompileInputError",
    "CompileResult",
    "CompileTimeout",
    "FlatBendLine",
    "FlatParseError",
    "Mark",
    "ParsedFlat",
    "analyze_cad",
    "compile_part",
    "load_config",
    "load_part_model",
    "parse_flat",
    "propose_flat_binding",
    "reconcile_draft",
    "sha256_file",
    "stage_points_from_compiled",
    "stage_transforms_from_compiled",
]
