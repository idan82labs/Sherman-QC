"""Frozen CIIP-V1 research primitives.

Only neutral, deterministic identifier helpers are exposed here.  The package
does not import scan readers, geometry, evaluation, or runtime integration.
"""

from . import identifiers

__all__ = ["identifiers"]
