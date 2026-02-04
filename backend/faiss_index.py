"""
FAISS Index Manager for Part Auto-Recognition

Manages vector similarity search for part embeddings using FAISS.
Supports 100-500 parts with <50ms search time target.

Features:
- Add/remove/update embeddings
- K-nearest neighbor search
- Persistence (save/load index)
- Thread-safe operations
"""

import numpy as np
import logging
import os
import json
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIM = 1024
INDEX_VERSION = "v1.0"


@dataclass
class SearchResult:
    """Result of a similarity search"""
    part_id: str
    similarity: float
    distance: float


class FAISSIndexManager:
    """
    Manages FAISS index for part embedding search.

    For 100-500 parts with 1024-dim vectors:
    - Uses IndexFlatIP (inner product = cosine similarity for normalized vectors)
    - CPU-only (fast enough for this scale)
    - Supports add, remove, search operations
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        """
        Initialize FAISS index manager.

        Args:
            index_path: Path to save/load index (optional)
            embedding_dim: Dimension of embeddings (default 1024)
        """
        if not HAS_FAISS:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )

        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path) if index_path else None
        self._lock = threading.RLock()

        # Initialize index (Inner Product for cosine similarity on L2-normalized vectors)
        self._index = faiss.IndexFlatIP(embedding_dim)

        # Mapping from FAISS internal ID to part_id
        self._id_to_part: Dict[int, str] = {}
        self._part_to_id: Dict[str, int] = {}
        self._embeddings: Dict[str, np.ndarray] = {}  # Store for updates/rebuilds
        self._next_id = 0

        # Load existing index if path provided and files exist
        if self.index_path:
            faiss_file = self.index_path.with_suffix('.faiss')
            if faiss_file.exists():
                self.load()

        logger.info(f"FAISS index initialized with {self.count()} parts")

    def add(
        self,
        part_id: str,
        embedding: np.ndarray,
        replace: bool = True,
    ) -> bool:
        """
        Add or update an embedding in the index.

        Args:
            part_id: Unique identifier for the part
            embedding: 1024-dim normalized embedding vector
            replace: If True, replace existing embedding; if False, skip

        Returns:
            True if added/updated, False if skipped
        """
        embedding = self._validate_embedding(embedding)

        with self._lock:
            if part_id in self._part_to_id:
                if not replace:
                    return False
                # Remove existing before re-adding
                self._remove_internal(part_id)

            # Add to index
            internal_id = self._next_id
            self._next_id += 1

            self._index.add(embedding.reshape(1, -1))

            # Update mappings
            self._id_to_part[internal_id] = part_id
            self._part_to_id[part_id] = internal_id
            self._embeddings[part_id] = embedding.copy()

            logger.debug(f"Added part {part_id} to FAISS index (id={internal_id})")
            return True

    def remove(self, part_id: str) -> bool:
        """
        Remove a part from the index.

        Note: FAISS IndexFlatIP doesn't support efficient removal.
        We rebuild the index without the removed part.

        Args:
            part_id: Part identifier to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if part_id not in self._part_to_id:
                return False

            self._remove_internal(part_id)
            return True

    def _remove_internal(self, part_id: str):
        """Internal removal - rebuilds index without the part."""
        # Remove from our tracking
        internal_id = self._part_to_id.pop(part_id, None)
        if internal_id is not None:
            self._id_to_part.pop(internal_id, None)
        self._embeddings.pop(part_id, None)

        # Rebuild index without removed part
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild FAISS index from stored embeddings."""
        self._index.reset()
        self._id_to_part.clear()
        self._part_to_id.clear()
        self._next_id = 0

        for part_id, embedding in self._embeddings.items():
            internal_id = self._next_id
            self._next_id += 1

            self._index.add(embedding.reshape(1, -1))

            self._id_to_part[internal_id] = part_id
            self._part_to_id[part_id] = internal_id

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for most similar parts.

        Args:
            query_embedding: 1024-dim normalized query vector
            k: Number of results to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        query_embedding = self._validate_embedding(query_embedding)

        with self._lock:
            if self._index.ntotal == 0:
                return []

            k = min(k, self._index.ntotal)

            # Search returns (distances, indices) where distances are inner products
            distances, indices = self._index.search(query_embedding.reshape(1, -1), k)

            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:  # Invalid index
                    continue

                part_id = self._id_to_part.get(idx)
                if part_id is None:
                    continue

                # For L2-normalized vectors, inner product = cosine similarity
                similarity = float(dist)

                if similarity < threshold:
                    continue

                results.append(SearchResult(
                    part_id=part_id,
                    similarity=similarity,
                    distance=1.0 - similarity,  # Convert to distance
                ))

            return results

    def get_embedding(self, part_id: str) -> Optional[np.ndarray]:
        """Get the stored embedding for a part."""
        with self._lock:
            return self._embeddings.get(part_id)

    def has_part(self, part_id: str) -> bool:
        """Check if a part exists in the index."""
        with self._lock:
            return part_id in self._part_to_id

    def count(self) -> int:
        """Get number of parts in the index."""
        with self._lock:
            return len(self._embeddings)

    def list_parts(self) -> List[str]:
        """Get all part IDs in the index."""
        with self._lock:
            return list(self._embeddings.keys())

    def save(self, path: Optional[str] = None) -> bool:
        """
        Save index to disk.

        Args:
            path: Path to save (uses default index_path if not specified)

        Returns:
            True if saved successfully
        """
        save_path = Path(path) if path else self.index_path
        if not save_path:
            logger.error("No path specified for saving index")
            return False

        with self._lock:
            try:
                # Save FAISS index
                faiss.write_index(self._index, str(save_path.with_suffix('.faiss')))

                # Save metadata
                metadata = {
                    'version': INDEX_VERSION,
                    'embedding_dim': self.embedding_dim,
                    'count': len(self._embeddings),
                    'id_to_part': self._id_to_part,
                    'part_to_id': self._part_to_id,
                    'next_id': self._next_id,
                }

                with open(save_path.with_suffix('.meta.json'), 'w') as f:
                    json.dump(metadata, f)

                # Save embeddings (numpy format)
                np.savez_compressed(
                    save_path.with_suffix('.npz'),
                    **{k: v for k, v in self._embeddings.items()}
                )

                logger.info(f"Saved FAISS index to {save_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
                return False

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load index from disk.

        Args:
            path: Path to load (uses default index_path if not specified)

        Returns:
            True if loaded successfully
        """
        load_path = Path(path) if path else self.index_path
        if not load_path:
            logger.error("No path specified for loading index")
            return False

        faiss_path = load_path.with_suffix('.faiss')
        meta_path = load_path.with_suffix('.meta.json')
        npz_path = load_path.with_suffix('.npz')

        if not faiss_path.exists() or not meta_path.exists() or not npz_path.exists():
            logger.warning(f"Index files not found at {load_path}")
            return False

        with self._lock:
            try:
                # Load FAISS index
                self._index = faiss.read_index(str(faiss_path))

                # Load metadata
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)

                # Convert string keys back to int for id_to_part
                self._id_to_part = {int(k): v for k, v in metadata['id_to_part'].items()}
                self._part_to_id = metadata['part_to_id']
                self._next_id = metadata['next_id']

                # Load embeddings
                with np.load(npz_path) as data:
                    self._embeddings = {k: data[k] for k in data.files}

                logger.info(f"Loaded FAISS index from {load_path} ({self.count()} parts)")
                return True

            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                # Reset to empty state
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                self._id_to_part.clear()
                self._part_to_id.clear()
                self._embeddings.clear()
                self._next_id = 0
                return False

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validate and normalize embedding."""
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim}-dim embedding, got {len(embedding)}"
            )

        # Ensure L2 normalized
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-6:
            embedding = embedding / norm

        return embedding


# Global index instance
_global_index: Optional[FAISSIndexManager] = None
_global_index_lock = threading.Lock()


def get_faiss_index(index_path: Optional[str] = None) -> FAISSIndexManager:
    """
    Get or create the global FAISS index instance.

    Args:
        index_path: Path to save/load index

    Returns:
        FAISSIndexManager instance
    """
    global _global_index

    with _global_index_lock:
        if _global_index is None:
            # Use default path if not specified
            if index_path is None:
                base_path = os.environ.get(
                    'SHERMAN_DATA_DIR',
                    os.path.join(os.path.dirname(__file__), '..', 'data')
                )
                index_path = os.path.join(base_path, 'faiss_index')

            _global_index = FAISSIndexManager(index_path=index_path)

        return _global_index


# Test function
if __name__ == "__main__":
    import tempfile
    import time

    print("Testing FAISSIndexManager...")

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, 'test_index')
        index = FAISSIndexManager(index_path=index_path)

        # Generate random embeddings (L2 normalized)
        n_parts = 100
        embeddings = {}
        for i in range(n_parts):
            emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings[f"PART-{i:03d}"] = emb

        # Test add
        print(f"\n[1] Adding {n_parts} parts...")
        start = time.time()
        for part_id, emb in embeddings.items():
            index.add(part_id, emb)
        add_time = (time.time() - start) * 1000
        print(f"    Added in {add_time:.1f}ms ({add_time/n_parts:.2f}ms/part)")
        print(f"    Index count: {index.count()}")

        # Test search
        print("\n[2] Testing search...")
        query = embeddings["PART-050"]
        start = time.time()
        results = index.search(query, k=10)
        search_time = (time.time() - start) * 1000
        print(f"    Search time: {search_time:.2f}ms")
        print(f"    Top 5 results:")
        for r in results[:5]:
            print(f"      {r.part_id}: {r.similarity:.4f}")

        # Verify self-match
        if results[0].part_id == "PART-050":
            print("    ✓ Self-match found at top")
        else:
            print("    ✗ Self-match not at top!")

        # Test save/load
        print("\n[3] Testing persistence...")
        index.save()
        print(f"    Saved to {index_path}")

        # Create new index and load
        index2 = FAISSIndexManager(index_path=index_path)
        print(f"    Loaded {index2.count()} parts")

        # Verify search still works
        results2 = index2.search(query, k=10)
        if results2[0].part_id == results[0].part_id:
            print("    ✓ Search results match after reload")
        else:
            print("    ✗ Search results differ after reload!")

        # Test remove
        print("\n[4] Testing remove...")
        index2.remove("PART-050")
        print(f"    Removed PART-050, count: {index2.count()}")

        results3 = index2.search(query, k=10)
        if all(r.part_id != "PART-050" for r in results3):
            print("    ✓ Removed part not in results")
        else:
            print("    ✗ Removed part still in results!")

        # Benchmark search
        print("\n[5] Search benchmark (1000 queries)...")
        start = time.time()
        for _ in range(1000):
            query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            query = query / np.linalg.norm(query)
            index2.search(query, k=10)
        benchmark_time = (time.time() - start) * 1000
        print(f"    1000 searches in {benchmark_time:.1f}ms ({benchmark_time/1000:.3f}ms/search)")

        print("\nAll tests completed!")
