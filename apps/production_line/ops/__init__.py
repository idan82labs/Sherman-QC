"""Shop-floor operations tools (C12): backup/restore, retention, synthetic scorecard.

- :mod:`.backup`: ``python -m apps.production_line.ops.backup create|verify|restore``.
  Consistent SQLite online backups of the line DBs and the app copy's legacy
  DBs, plus tar archives of the released recipes, all in one hashed backup file.
- :mod:`.retention`: ``python -m apps.production_line.ops.retention``. A dry-run-
  first pruning of archived scans, heavy per-job artifacts and old line events.
- :mod:`.synthetic_scorecard`: ``python -m apps.production_line.ops.synthetic_scorecard``.
  A seeded synthetic grid run end to end through a private LineStore, the
  JobRunner and the real render step. It is scored by E4's scorer and labelled
  as synthetic engineering capability, never physical accuracy.

Runbook: ``docs/operations/shopfloor-runbook.md``. Importing this package
imports nothing heavy.
"""
from __future__ import annotations

__all__ = ["backup", "retention", "synthetic_scorecard"]
