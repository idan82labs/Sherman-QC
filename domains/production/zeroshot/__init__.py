"""Production packaging of the frozen zero-shot bend-line lane (zs_lane_v1) and its post-E3 overlay.

E3 stays the only source of measurements and decisions. This package adds a display/REVIEW overlay next to it
(docs/production/runlog/zeroshot_2026-10-25/cad_path/CAD_PATH.md, "Recommended production design"):

- :mod:`.pin` -- the pinned lane: the 60 frozen source files of ``research/zeroshot/zs_lane_v1`` (and what it
  executes) with their sha256 and combined fingerprint, and the vendored selector model with its sha256
  (``assets/zs_lane_v1.pin.json``). The research code is imported read-only, never copied or edited; a
  fingerprint or model mismatch disables the lane (no overlay), it never fails a job.
- :mod:`.lane_cli` -- the runner's zs step (``python -m domains.production.zeroshot.lane_cli``), run alongside
  E3 on the staged ``input/scan.<ext>`` with CAD priors from the CompiledPart only (count cap at full stage,
  CAD thickness, axis-family cap K + 2; no angle window). It never receives a file name, serial or job token.
- :mod:`.overlay` -- ``sherman_zeroshot_overlay_v1``: the versioned job-level overlay merged from E3's
  ``result.json``, the CompiledPart and the zs lines (option 2 gap-fill lines, option 1 unexplained-bend
  REVIEW flag). It can only lower a pass-like verdict to REVIEW; it never creates or upgrades a PASS.
- :mod:`.overlay_cli` -- the runner's merge step (``python -m domains.production.zeroshot.overlay_cli``).
"""
from __future__ import annotations

#: Version of this production wrapper (lane_cli + overlay). The frozen lane itself is pinned by fingerprint.
ZS_RUNTIME_VERSION = "zs-prod-1.0.0"
LANE_ID = "zs_lane_v1"
LINES_SCHEMA = "sherman_zeroshot_lines_v1"
#: Workspace-relative directory of the zs step and overlay outputs (``<job>/zeroshot/``).
WORKSPACE_DIR = "zeroshot"
LINES_FILE = "zs_lines.json"
OVERLAY_FILE = "zeroshot_overlay_v1.json"

__all__ = ["ZS_RUNTIME_VERSION", "LANE_ID", "LINES_SCHEMA", "WORKSPACE_DIR", "LINES_FILE", "OVERLAY_FILE"]
