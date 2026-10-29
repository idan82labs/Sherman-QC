"""Jev (TypeSafe System One) bend-candidate dataset and offline harness (C10).

Research only. Nothing in ``apps/``, ``backend/``, ``domains/``, ``contracts/`` or
``frontend/`` may import this package, and no Jev output feeds a runtime
decision. The harness answers one question offline:

    "Is this candidate a real press-brake bend line?"

Modules:

- :mod:`.state_v1` builds coordinate-free, identity-free candidate states and
  enforces the deny-list (fail closed).
- :mod:`.questions_v1` holds the frozen question sets and their hashes.
- :mod:`.request_v1` builds and validates System One request bodies against
  the documented API limits.
- :mod:`.client_v1` is the dry-run-by-default client. A live call needs both
  ``--live`` and ``TYPESAFE_API_KEY`` in the environment.
- :mod:`.labels_v1` and :mod:`.splits_v1` hold the label sidecar rules and the
  grouped, family-level splits.
- :mod:`.export_onboarding_v1` and :mod:`.export_scan_candidates_v1` turn
  explicit input files into ``states.jsonl``, ``labels.jsonl`` and
  ``manifest.json``.
- :mod:`.evaluate_v1` scores answers against labels with frozen operating
  points and baselines.
- :mod:`.cli` is the command line (``python -m research.jev.cli``).

See ``docs/research/JEV_BEND_CANDIDATE_DATASET_V1.md``.
"""
from __future__ import annotations

__all__ = ["HARNESS_VERSION"]

#: Bumped whenever an exported byte would change for the same input.
HARNESS_VERSION = "jev-harness-1.0.0"
