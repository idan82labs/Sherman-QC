"""Part frame for 38172000 X scan derived from gold B06 (long axis) and B02 (short axis) directions.
L = unit(B06 direction) (long, along B01/B03/B05/B06), W = B02 direction orthogonalised, N = L x W.
Coordinates are relative to ORIGIN = gold B06 start point.  Gold lines are only used to define axes.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import numpy as np

from a2_common import gold_lines, unit

_G = gold_lines()
L = unit(_G["B06"][1] - _G["B06"][0])
_w = unit(_G["B02"][1] - _G["B02"][0])
W = unit(_w - L * (_w @ L))
N = np.cross(L, W)
ORIGIN = _G["B06"][0].copy()
R = np.stack([L, W, N])  # rows


def to_part(p):
    return (np.asarray(p) - ORIGIN) @ R.T


def from_part(c):
    return ORIGIN + np.asarray(c) @ R
