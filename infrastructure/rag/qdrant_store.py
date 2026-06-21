from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from infrastructure.rag.manual_store import (
    DATA_DIR,
    PageRecord,
    RetrievalHit,
    ManualRetriever,
    best_excerpt,
    term_coverage,
    tokenize,
)


class QdrantUnavailable(RuntimeError):
    pass


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "sherman_manual_pages"
DEFAULT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_HASH_DIM = 384
SPARSE_VECTOR_VERSION = "sha1-token-bm25-v1"
PAYLOAD_VERSION = "minimal-payload-v2"
logger = logging.getLogger(__name__)
_COLLECTION_LOCKS: dict[str, threading.Lock] = {}
_COLLECTION_LOCKS_GUARD = threading.Lock()


def _compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _point_id(page: PageRecord) -> str:
    key = f"{page.profile}:{page.manual_id}:{page.page_number}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _indexed_text(page: PageRecord) -> str:
    visual_terms = ""
    if page.visual_heavy or page.crop_path or page.page_image_path:
        visual_terms = " visual diagram drawing figure screenshot screen page crop interface layout"
    return f"{page.section_title}\n{page.manual_id}\n{page.filename}\n{page.text}\n{visual_terms}".strip()


def _token_index(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return values
    return [value / norm for value in values]


class HybridVectorizer:
    def __init__(self, pages: Iterable[PageRecord]):
        self.pages = list(pages)
        self.dense_model_name = os.environ.get("SHERMAN_QDRANT_DENSE_MODEL", DEFAULT_DENSE_MODEL)
        self.hash_dim = int(os.environ.get("SHERMAN_QDRANT_HASH_DIM", str(DEFAULT_HASH_DIM)))
        self._dense_model = None
        self._dense_mode = "hash"
        self._load_fastembed()
        self.dense_dim = self._infer_dense_dim()
        self.idf = self._build_idf()

    @property
    def dense_mode(self) -> str:
        return self._dense_mode

    def _load_fastembed(self) -> None:
        if os.environ.get("SHERMAN_QDRANT_DENSE_MODE", "fastembed").lower() == "hash":
            return
        try:
            from fastembed import TextEmbedding  # type: ignore

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*now uses mean pooling instead of CLS embedding.*",
                    category=UserWarning,
                )
                self._dense_model = TextEmbedding(model_name=self.dense_model_name)
            self._dense_mode = "fastembed"
        except Exception:
            self._dense_model = None
            self._dense_mode = "hash"

    def _infer_dense_dim(self) -> int:
        if self._dense_model is not None:
            try:
                sample = next(iter(self._dense_model.embed(["dimension probe"])))
                return len(sample.tolist() if hasattr(sample, "tolist") else list(sample))
            except Exception:
                self._dense_model = None
                self._dense_mode = "hash"
        return self.hash_dim

    def _build_idf(self) -> dict[str, float]:
        df: dict[str, int] = defaultdict(int)
        tokenized = [set(tokenize(_indexed_text(page), include_query_expansions=True)) for page in self.pages]
        for tokens in tokenized:
            for token in tokens:
                df[token] += 1
        n = max(len(tokenized), 1)
        return {token: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}

    def dense(self, text: str) -> list[float]:
        if self._dense_model is not None:
            try:
                vector = next(iter(self._dense_model.embed([text])))
                values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                return [float(value) for value in values]
            except Exception:
                self._dense_model = None
                self._dense_mode = "hash"

        values = [0.0] * self.hash_dim
        for token in tokenize(text, include_query_expansions=True):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.hash_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[idx] += sign
        return _l2_normalize(values)

    def sparse(self, text: str) -> dict[str, list[int] | list[float]]:
        counts = Counter(tokenize(text, include_query_expansions=True))
        merged: dict[int, float] = defaultdict(float)
        for token, count in counts.items():
            weight = (1.0 + math.log(count)) * self.idf.get(token, 1.0)
            merged[_token_index(token)] += weight
        if not merged:
            return {"indices": [], "values": []}
        items = sorted(merged.items())
        indices = [idx for idx, _value in items]
        values = _l2_normalize([float(value) for _idx, value in items])
        return {"indices": indices, "values": values}


class QdrantManualRetriever:
    def __init__(self, pages: Iterable[PageRecord], local_retriever: ManualRetriever | None = None):
        self.pages = list(pages)
        self.local_retriever = local_retriever or ManualRetriever(self.pages)
        self.url = os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")
        self.collection = os.environ.get("SHERMAN_QDRANT_COLLECTION", DEFAULT_COLLECTION)
        self.timeout = float(os.environ.get("SHERMAN_QDRANT_TIMEOUT", "2"))
        self.api_key = os.environ.get("QDRANT_API_KEY")
        self._validate_url()
        self._ensure_qdrant_available()
        self.vectorizer = HybridVectorizer(self.pages)
        self.page_by_key = {(page.manual_id, page.page_number): page for page in self.pages}
        self.marker_path = DATA_DIR / "qdrant" / f"{self.collection}.fingerprint.json"
        self._ensure_collection()

    def retrieve(self, query: str, profile: str, top_k: int = 5) -> list[RetrievalHit]:
        return self.retrieve_with_local(query, profile, top_k=top_k, local_hits=None)

    def retrieve_with_local(
        self,
        query: str,
        profile: str,
        top_k: int = 5,
        local_hits: list[RetrievalHit] | None = None,
    ) -> list[RetrievalHit]:
        candidate_limit = max(top_k * 6, int(os.environ.get("SHERMAN_QDRANT_CANDIDATE_LIMIT", "30")))
        if local_hits is None:
            local_hits = self.local_retriever.retrieve(query, profile, top_k=candidate_limit)
        else:
            local_hits = local_hits[:candidate_limit]
        try:
            qdrant_points = self._query(query, profile, candidate_limit)
        except QdrantUnavailable as exc:
            logger.warning("Qdrant query unavailable; falling back to local retrieval: %s", exc)
            return self._mark_qdrant_unavailable(local_hits[:top_k])

        merged: dict[tuple[str, int], RetrievalHit] = {
            (hit.page.manual_id, hit.page.page_number): hit for hit in local_hits
        }
        qdrant_rank_by_key: dict[tuple[str, int], tuple[int, float]] = {}
        for rank, point in enumerate(qdrant_points, start=1):
            payload = point.get("payload") or {}
            manual_id = payload.get("manual_id")
            page_number = payload.get("page_number")
            if not isinstance(manual_id, str) or not isinstance(page_number, int):
                continue
            key = (manual_id, page_number)
            if key not in self.page_by_key:
                continue
            qdrant_rank_by_key[key] = (rank, float(point.get("score") or 0.0))
            if key not in merged:
                merged[key] = self._qdrant_only_hit(query, self.page_by_key[key], rank, float(point.get("score") or 0.0))

        ranked: list[tuple[float, RetrievalHit]] = []
        for key, hit in merged.items():
            qrank, qscore = qdrant_rank_by_key.get(key, (0, 0.0))
            boost = max(0.0, 2.0 - (qrank - 1) * 0.08) if qrank else 0.0
            features = dict(hit.rerank_features)
            if qrank:
                features.update(
                    {
                        "qdrant_hybrid": True,
                        "qdrant_rank": qrank,
                        "qdrant_score": round(qscore, 6),
                        "qdrant_dense_mode": self.vectorizer.dense_mode,
                        "qdrant_rank_boost": round(boost, 4),
                    }
                )
            hit_with_features = RetrievalHit(
                rank=hit.rank,
                score=hit.score,
                page=hit.page,
                query_term_coverage=hit.query_term_coverage,
                matched_query_terms=hit.matched_query_terms,
                missing_query_terms=hit.missing_query_terms,
                excerpt=hit.excerpt,
                rerank_features=features,
            )
            ranked.append(
                (
                    hit.score + boost,
                    hit_with_features,
                )
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                rank=rank,
                score=hit.score,
                page=hit.page,
                query_term_coverage=hit.query_term_coverage,
                matched_query_terms=hit.matched_query_terms,
                missing_query_terms=hit.missing_query_terms,
                excerpt=hit.excerpt,
                rerank_features=hit.rerank_features,
            )
            for rank, (_rank_score, hit) in enumerate(ranked[:top_k], start=1)
        ]

    def _mark_qdrant_unavailable(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        marked = []
        for hit in hits:
            features = dict(hit.rerank_features)
            features["qdrant_unavailable"] = True
            marked.append(
                RetrievalHit(
                    rank=hit.rank,
                    score=hit.score,
                    page=hit.page,
                    query_term_coverage=hit.query_term_coverage,
                    matched_query_terms=hit.matched_query_terms,
                    missing_query_terms=hit.missing_query_terms,
                    excerpt=hit.excerpt,
                    rerank_features=features,
                )
            )
        return marked

    def _qdrant_only_hit(self, query: str, page: PageRecord, rank: int, qscore: float) -> RetrievalHit:
        query_tokens = tokenize(query, include_query_expansions=True)
        coverage_tokens = tokenize(query, include_query_expansions=False)
        coverage, matched, missing = term_coverage(coverage_tokens, f"{page.section_title} {page.text}")
        return RetrievalHit(
            rank=rank,
            score=round(coverage * 4.0, 4),
            page=page,
            query_term_coverage=round(coverage, 3),
            matched_query_terms=matched,
            missing_query_terms=missing,
            excerpt=best_excerpt(page.text, query_tokens),
            rerank_features={
                "bm25_score": 0.0,
                "core_coverage": round(coverage, 3),
                "qdrant_hybrid": True,
                "qdrant_score": round(qscore, 6),
                "qdrant_dense_mode": self.vectorizer.dense_mode,
            },
        )

    def _fingerprint(self) -> str:
        payload = {
            "pages": [
                {
                    "profile": page.profile,
                    "manual_id": page.manual_id,
                    "page_number": page.page_number,
                    "text_hash": hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
                    "section_title": page.section_title,
                    "visual_heavy": page.visual_heavy,
                }
                for page in self.pages
            ],
            "dense_model": self.vectorizer.dense_model_name,
            "dense_mode": self.vectorizer.dense_mode,
            "dense_dim": self.vectorizer.dense_dim,
            "sparse_vector_version": SPARSE_VECTOR_VERSION,
            "payload_version": PAYLOAD_VERSION,
        }
        return hashlib.sha256(_compact_json(payload)).hexdigest()

    def _ensure_collection(self) -> None:
        fingerprint = self._fingerprint()
        lock = _collection_lock(self.collection)
        with lock:
            if self._collection_ready(fingerprint):
                return
            self._delete_collection(ignore_missing=True)
            self._create_collection()
            self._upsert_pages()
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker = json.dumps(
                {
                    "fingerprint": fingerprint,
                    "collection": self.collection,
                    "page_count": len(self.pages),
                    "dense_mode": self.vectorizer.dense_mode,
                    "dense_model": self.vectorizer.dense_model_name,
                    "payload_version": PAYLOAD_VERSION,
                },
                ensure_ascii=False,
                indent=2,
            )
            fd, temp_name = tempfile.mkstemp(
                prefix=f"{self.collection}.",
                suffix=".tmp",
                dir=str(self.marker_path.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(marker)
            Path(temp_name).replace(self.marker_path)

    def _collection_ready(self, fingerprint: str) -> bool:
        if not self.marker_path.exists():
            return False
        try:
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if marker.get("fingerprint") != fingerprint:
            return False
        try:
            count = self._count_points()
            return count >= len(self.pages)
        except QdrantUnavailable:
            raise
        except Exception:
            return False

    def _create_collection(self) -> None:
        self._request(
            "PUT",
            f"/collections/{self.collection}",
            {
                "vectors": {
                    "dense": {
                        "size": self.vectorizer.dense_dim,
                        "distance": "Cosine",
                    }
                },
                "sparse_vectors": {
                    "sparse": {
                        "index": {"on_disk": False},
                    }
                },
            },
        )

    def _delete_collection(self, ignore_missing: bool = False) -> None:
        try:
            self._request("DELETE", f"/collections/{self.collection}")
        except QdrantUnavailable:
            raise
        except Exception:
            if not ignore_missing:
                raise

    def _count_points(self) -> int:
        payload = self._request("POST", f"/collections/{self.collection}/points/count", {"exact": True})
        result = payload.get("result") or {}
        return int(result.get("count") or 0)

    def _upsert_pages(self) -> None:
        batch_size = int(os.environ.get("SHERMAN_QDRANT_UPSERT_BATCH_SIZE", "32"))
        points = []
        for page in self.pages:
            text = _indexed_text(page)
            payload = {
                "point_kind": "manual_page",
                "profile": page.profile,
                "manual_id": page.manual_id,
                "filename": page.filename,
                "page_number": page.page_number,
                "section_title": page.section_title,
                "visual_heavy": page.visual_heavy,
                "has_page_image": bool(page.page_image_path),
                "has_crop": bool(page.crop_path),
                "text_hash": hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
            }
            points.append(
                {
                    "id": _point_id(page),
                    "vector": {
                        "dense": self.vectorizer.dense(text),
                        "sparse": self.vectorizer.sparse(text),
                    },
                    "payload": payload,
                }
            )
            if len(points) >= batch_size:
                self._request("PUT", f"/collections/{self.collection}/points?wait=true", {"points": points})
                points = []
        if points:
            self._request("PUT", f"/collections/{self.collection}/points?wait=true", {"points": points})

    def _query(self, query: str, profile: str, limit: int) -> list[dict[str, Any]]:
        profile_filter = {"must": [{"key": "profile", "match": {"value": profile}}]}
        payload = {
            "prefetch": [
                {
                    "query": self.vectorizer.sparse(query),
                    "using": "sparse",
                    "filter": profile_filter,
                    "limit": limit,
                },
                {
                    "query": self.vectorizer.dense(query),
                    "using": "dense",
                    "filter": profile_filter,
                    "limit": limit,
                },
            ],
            "query": {"rrf": {}},
            "with_payload": True,
            "limit": limit,
        }
        response = self._request("POST", f"/collections/{self.collection}/points/query", payload)
        result = response.get("result")
        if isinstance(result, dict):
            points = result.get("points") or []
        else:
            points = result or []
        return [point for point in points if isinstance(point, dict)]

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read().decode("utf-8")
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="ignore")
            raise QdrantUnavailable(f"Qdrant HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise QdrantUnavailable(f"Qdrant is unavailable at {self.url}: {exc}") from exc

    def _ensure_qdrant_available(self) -> None:
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        request = urllib.request.Request(f"{self.url}/readyz", headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 2.0)) as response:
                if response.status >= 400:
                    raise QdrantUnavailable(f"Qdrant readiness returned HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise QdrantUnavailable(f"Qdrant is unavailable at {self.url}: {exc}") from exc

    def _validate_url(self) -> None:
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise QdrantUnavailable("Qdrant URL must be an HTTP(S) URL")
        host = parsed.hostname.lower()
        local_or_internal = host in {"localhost", "127.0.0.1", "::1", "qdrant"}
        if os.environ.get("PRODUCTION", "").strip().lower() in {"1", "true", "yes", "on"}:
            if parsed.scheme == "http" and not local_or_internal and not self.api_key:
                raise QdrantUnavailable("Qdrant remote HTTP URL requires QDRANT_API_KEY in production")


def _collection_lock(collection: str) -> threading.Lock:
    with _COLLECTION_LOCKS_GUARD:
        lock = _COLLECTION_LOCKS.get(collection)
        if lock is None:
            lock = threading.Lock()
            _COLLECTION_LOCKS[collection] = lock
        return lock
