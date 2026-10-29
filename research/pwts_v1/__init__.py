"""Isolated point-native PWTS representation preflight package.

The package initializer deliberately imports only neutral configuration.  In
particular, importing the independent validator must never transitively load
the semantic implementation.
"""

from .config import CONFIG, CONFIG_SHA256, EXPECTED_CONFIG_SHA256

__all__ = ["CONFIG", "CONFIG_SHA256", "EXPECTED_CONFIG_SHA256"]
