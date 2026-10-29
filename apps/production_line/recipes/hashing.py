"""File hashing for the C05 recipe back end, with a stat-keyed cache for read-only checks.

:func:`sha256_file` always reads every byte. :func:`cached_sha256` caches the
digest by ``(realpath, device, inode, size, mtime_ns)``; the onboarding
checklist polls the release gate's read-only checks (CAD, drawing and flat
hashes), and re-hashing a large STEP on every poll is wasteful.

The cache is only used where a stale hit cannot release wrong bytes: the
release re-hashes every byte it freezes (``release_service._freeze_file``)
and E1 re-checks the CAD and flat hashes it is given (CAD_HASH_MISMATCH /
FLAT_HASH_MISMATCH), so a same-size, same-mtime rewrite that the cache could
miss is still refused at release.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

__all__ = ["sha256_file", "cached_sha256", "clear_sha_cache"]

_CHUNK = 1024 * 1024
_CACHE_MAX = 256
_cache: dict[tuple[str, int, int, int, int], str] = {}
_lock = threading.Lock()


def sha256_file(path: Path | str) -> str:
    """Hex sha256 of the file's bytes (always read)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_sha256(path: Path | str) -> str:
    """Hex sha256 of ``path``, cached by (realpath, device, inode, size, mtime_ns)."""
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    digest = sha256_file(path)
    with _lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = digest
    return digest


def clear_sha_cache() -> None:
    """Forget every cached digest (tests)."""
    with _lock:
        _cache.clear()
