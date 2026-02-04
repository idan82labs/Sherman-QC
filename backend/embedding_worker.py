"""
Embedding Background Worker

Automatically computes embeddings for parts with CAD files but no embeddings.
Updates both the FAISS index and the database.

Usage:
    # Start the worker (runs until interrupted)
    python embedding_worker.py

    # Or use as a module
    from embedding_worker import EmbeddingWorker
    worker = EmbeddingWorker()
    worker.start()
"""

import numpy as np
import logging
import os
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from embedding_service import compute_embedding, EMBEDDING_VERSION
from faiss_index import get_faiss_index
from part_catalog import get_catalog

logger = logging.getLogger(__name__)

# Configuration
POLL_INTERVAL = 30  # seconds between checks for new parts
BATCH_SIZE = 10  # parts to process per batch


class EmbeddingWorker:
    """
    Background worker for computing part embeddings.

    Monitors the part catalog for parts with CAD files but no embeddings,
    computes embeddings, and updates both FAISS index and database.
    """

    def __init__(
        self,
        poll_interval: int = POLL_INTERVAL,
        batch_size: int = BATCH_SIZE,
    ):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "last_run": None,
            "last_error": None,
        }

    def start(self, blocking: bool = True):
        """
        Start the embedding worker.

        Args:
            blocking: If True, run in foreground; if False, run in background thread
        """
        if self._running:
            logger.warning("Embedding worker already running")
            return

        self._running = True
        self._stop_event.clear()

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Embedding worker started in background")

    def stop(self):
        """Stop the embedding worker."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Embedding worker stopped")

    def _run_loop(self):
        """Main worker loop."""
        logger.info("Embedding worker started")

        while self._running and not self._stop_event.is_set():
            try:
                # Process pending parts
                processed = self.process_pending()
                if processed > 0:
                    logger.info(f"Processed {processed} parts")

                self.stats["last_run"] = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Error in embedding worker: {e}")
                self.stats["last_error"] = str(e)

            # Wait for next poll
            self._stop_event.wait(timeout=self.poll_interval)

        logger.info("Embedding worker loop ended")

    def process_pending(self) -> int:
        """
        Process parts that need embeddings.

        Returns:
            Number of parts processed
        """
        catalog = get_catalog()
        faiss_index = get_faiss_index()

        # Find parts needing embeddings
        pending_parts = self._get_pending_parts(catalog)

        if not pending_parts:
            return 0

        processed = 0
        for part in pending_parts[:self.batch_size]:
            try:
                success = self._process_part(part, catalog, faiss_index)
                if success:
                    self.stats["successful"] += 1
                else:
                    self.stats["failed"] += 1

                self.stats["total_processed"] += 1
                processed += 1

            except Exception as e:
                logger.error(f"Failed to process part {part['part_number']}: {e}")
                self.stats["failed"] += 1
                self.stats["last_error"] = str(e)

        # Save FAISS index after batch
        if processed > 0:
            faiss_index.save()

        return processed

    def _get_pending_parts(self, catalog) -> List[Dict[str, Any]]:
        """Get parts that have CAD but no embedding."""
        parts = catalog.list_parts(has_cad=True, has_embedding=False, limit=100)
        return [p.to_dict() for p in parts]

    def _process_part(self, part: Dict[str, Any], catalog, faiss_index) -> bool:
        """
        Process a single part: load CAD, compute embedding, update index.

        Returns:
            True if successful
        """
        part_id = part["id"]
        part_number = part["part_number"]
        cad_path = part["cad_file_path"]

        logger.info(f"Processing part {part_number} ({part_id})")

        if not cad_path:
            logger.warning(f"Part {part_number} has no CAD path")
            return False

        cad_path = Path(cad_path)
        if not cad_path.exists():
            logger.warning(f"CAD file not found: {cad_path}")
            return False

        # Load point cloud from CAD
        points = self._load_cad_points(cad_path)
        if points is None or len(points) < 100:
            logger.warning(f"Failed to load points from {cad_path}")
            return False

        # Compute embedding
        result = compute_embedding(points)
        embedding = result.embedding

        # Add to FAISS index
        faiss_index.add(part_id, embedding, replace=True)

        # Update database
        catalog.set_embedding(
            part_id,
            embedding=embedding,
            version=EMBEDDING_VERSION,
        )

        logger.info(
            f"Computed embedding for {part_number}: "
            f"{result.processing_time_ms:.0f}ms, {result.n_points_processed} points"
        )

        return True

    def _load_cad_points(self, path: Path, n_points: int = 10000) -> Optional[np.ndarray]:
        """Load and sample points from CAD file."""
        if not HAS_OPEN3D:
            logger.error("Open3D not available for CAD loading")
            return None

        suffix = path.suffix.lower()

        try:
            if suffix in ['.stl', '.obj']:
                mesh = o3d.io.read_triangle_mesh(str(path))
                if mesh.is_empty():
                    return None
                pcd = mesh.sample_points_uniformly(number_of_points=n_points)
                return np.asarray(pcd.points)

            elif suffix in ['.ply', '.pcd']:
                pcd = o3d.io.read_point_cloud(str(path))
                if pcd.is_empty():
                    return None
                # Downsample if too many points
                if len(pcd.points) > n_points * 2:
                    pcd = pcd.farthest_point_down_sample(n_points)
                return np.asarray(pcd.points)

            elif suffix in ['.step', '.stp']:
                # STEP files need special handling (not directly supported by Open3D)
                # For now, skip or use fallback
                logger.warning(f"STEP files not directly supported: {path}")
                return None

            else:
                logger.warning(f"Unsupported CAD format: {suffix}")
                return None

        except Exception as e:
            logger.error(f"Failed to load CAD file {path}: {e}")
            return None

    def process_single(self, part_id: str) -> bool:
        """
        Process a single part by ID (on-demand).

        Args:
            part_id: Part ID to process

        Returns:
            True if successful
        """
        catalog = get_catalog()
        faiss_index = get_faiss_index()

        part = catalog.get_part(part_id)
        if not part:
            logger.error(f"Part not found: {part_id}")
            return False

        return self._process_part(part.to_dict(), catalog, faiss_index)


# Global worker instance
_global_worker: Optional[EmbeddingWorker] = None
_worker_lock = threading.Lock()


def get_embedding_worker() -> EmbeddingWorker:
    """Get or create the global embedding worker instance."""
    global _global_worker

    with _worker_lock:
        if _global_worker is None:
            _global_worker = EmbeddingWorker()
        return _global_worker


def start_worker(blocking: bool = False):
    """Start the global embedding worker."""
    worker = get_embedding_worker()
    worker.start(blocking=blocking)


def stop_worker():
    """Stop the global embedding worker."""
    worker = get_embedding_worker()
    worker.stop()


if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("="*60)
    print("  EMBEDDING WORKER")
    print("="*60)

    worker = EmbeddingWorker(poll_interval=10)  # Faster polling for testing

    print(f"\nConfiguration:")
    print(f"  Poll interval: {worker.poll_interval}s")
    print(f"  Batch size: {worker.batch_size}")
    print(f"\nStarting worker (Ctrl+C to stop)...")

    try:
        worker.start(blocking=True)
    except KeyboardInterrupt:
        print("\nStopping worker...")
        worker.stop()

    print(f"\nStats: {worker.stats}")
