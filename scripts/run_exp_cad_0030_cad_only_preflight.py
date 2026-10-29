#!/usr/bin/env python3
"""Fresh EXP_CAD_0030 identity wrapper for the audited CF-PDG CAD preflight."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_exp_cad_0029_cad_only_preflight as BASE

BASE.EXPECTED_SCHEMA = "EXP_CAD_0030_CFPDG_V3_PREREGISTRATION_V1"
BASE.EXPECTED_EXPERIMENT_ID = "EXP_CAD_0030"
BASE.EXPECTED_ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
BASE.OUTPUT_PREFLIGHT_SCHEMA = "EXP_CAD_0030_CFPDG_V3_CAD_ONLY_PREFLIGHT_V1"
BASE.IDENTITY_WRAPPER_PATH = Path(__file__).resolve()
BASE.__doc__ = "Scan-free CAD actionability preflight for EXP_CAD_0030 CF-PDG V3."


if __name__ == "__main__":
    raise SystemExit(BASE.main())
