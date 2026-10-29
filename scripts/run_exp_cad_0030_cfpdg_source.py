#!/usr/bin/env python3
"""Hash-gated physical source runner for EXP_CAD_0030 CF-PDG V3."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_exp_cad_0029_cfpdg_source as BASE

BASE.RUNNER_IDENTITY_PATH = Path(__file__).resolve()
BASE.__doc__ = __doc__


if __name__ == "__main__":
    raise SystemExit(BASE.main())
