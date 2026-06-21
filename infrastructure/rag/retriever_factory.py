from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Iterable, Protocol

from infrastructure.rag.manual_store import (
    SUPPORT_QUERY_COVERAGE_THRESHOLD,
    SUPPORT_SCORE_THRESHOLD,
    ManualRetriever,
    PageRecord,
    RetrievalHit,
)


logger = logging.getLogger(__name__)
_CACHE_LOCK = threading.Lock()
_RETRIEVER_CACHE: dict[str, ManualRetrievalBackend] = {}


class ManualRetrievalBackend(Protocol):
    def retrieve(self, query: str, profile: str, top_k: int = 5) -> list[RetrievalHit]:
        ...


class AdaptiveManualRetriever:
    def __init__(self, pages: list[PageRecord], local_retriever: ManualRetriever):
        self.pages = pages
        self.local_retriever = local_retriever
        self._qdrant_retriever: ManualRetrievalBackend | None = None
        self._qdrant_lock = threading.Lock()
        self._qdrant_retry_after = 0.0

    def retrieve(self, query: str, profile: str, top_k: int = 5) -> list[RetrievalHit]:
        local_hits = self.local_retriever.retrieve(query, profile, top_k=max(top_k, 5))
        if self._local_confident(local_hits):
            return local_hits[:top_k]
        qdrant = self._get_qdrant()
        if qdrant is None:
            return local_hits[:top_k]
        retrieve_with_local = getattr(qdrant, "retrieve_with_local", None)
        try:
            if callable(retrieve_with_local):
                return retrieve_with_local(query, profile, top_k=top_k, local_hits=local_hits)
            return qdrant.retrieve(query, profile, top_k=top_k)
        except Exception as exc:
            self._trip_qdrant_circuit(exc)
            return local_hits[:top_k]

    def _local_confident(self, hits: list[RetrievalHit]) -> bool:
        if not hits:
            return False
        top = hits[0]
        if top.score >= SUPPORT_SCORE_THRESHOLD and top.query_term_coverage >= SUPPORT_QUERY_COVERAGE_THRESHOLD:
            return True
        return top.score >= 8.0 and top.query_term_coverage >= 0.30

    def _get_qdrant(self) -> ManualRetrievalBackend | None:
        if time.time() < self._qdrant_retry_after:
            return None
        if self._qdrant_retriever is not None:
            return self._qdrant_retriever
        with self._qdrant_lock:
            if time.time() < self._qdrant_retry_after:
                return None
            if self._qdrant_retriever is not None:
                return self._qdrant_retriever
            try:
                from infrastructure.rag.qdrant_store import QdrantManualRetriever

                self._qdrant_retriever = QdrantManualRetriever(self.pages, local_retriever=self.local_retriever)
            except Exception as exc:
                self._trip_qdrant_circuit(exc)
                return None
            return self._qdrant_retriever

    def _trip_qdrant_circuit(self, exc: Exception) -> None:
        cooldown = float(os.environ.get("SHERMAN_QDRANT_CIRCUIT_COOLDOWN_SECONDS", "30"))
        self._qdrant_retry_after = time.time() + max(1.0, cooldown)
        logger.info("Qdrant retriever unavailable for adaptive retrieval; using local retriever: %s", exc)


def _pages_fingerprint(pages: list[PageRecord]) -> str:
    payload = [
        {
            "profile": page.profile,
            "manual_id": page.manual_id,
            "page_number": page.page_number,
            "text_hash": hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
            "section_title": page.section_title,
            "visual_heavy": page.visual_heavy,
            "crop_path": page.crop_path,
            "page_image_path": page.page_image_path,
        }
        for page in pages
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_key(backend: str, pages: list[PageRecord]) -> str:
    qdrant_parts = {
        "url": os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        "collection": os.environ.get("SHERMAN_QDRANT_COLLECTION", "sherman_manual_pages"),
        "dense_model": os.environ.get(
            "SHERMAN_QDRANT_DENSE_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        "dense_mode": os.environ.get("SHERMAN_QDRANT_DENSE_MODE", "fastembed"),
        "hash_dim": os.environ.get("SHERMAN_QDRANT_HASH_DIM", "384"),
    }
    payload = {
        "backend": backend,
        "pages": _pages_fingerprint(pages),
        "qdrant": qdrant_parts if backend in {"qdrant", "auto"} else {},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_manual_retriever(pages: Iterable[PageRecord]) -> ManualRetrievalBackend:
    page_list = list(pages)
    backend = os.environ.get("SHERMAN_RETRIEVAL_BACKEND", "local").strip().lower()
    if backend not in {"local", "qdrant", "auto"}:
        logger.warning("Unknown SHERMAN_RETRIEVAL_BACKEND=%s; using local retriever", backend)
        backend = "local"

    key = _cache_key(backend, page_list)
    with _CACHE_LOCK:
        cached = _RETRIEVER_CACHE.get(key)
        if cached is not None:
            return cached

        local_retriever = ManualRetriever(page_list)

        if backend == "local":
            return _RETRIEVER_CACHE.setdefault(key, local_retriever)

        if backend == "auto":
            retriever = AdaptiveManualRetriever(page_list, local_retriever)
            return _RETRIEVER_CACHE.setdefault(key, retriever)

        try:
            from infrastructure.rag.qdrant_store import QdrantManualRetriever

            retriever = QdrantManualRetriever(page_list, local_retriever=local_retriever)
            return _RETRIEVER_CACHE.setdefault(key, retriever)
        except Exception as exc:
            logger.warning("Qdrant retriever requested but unavailable; using local retriever: %s", exc)
            return _RETRIEVER_CACHE.setdefault(key, local_retriever)
