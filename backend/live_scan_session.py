"""
Live Scan Session Manager

Manages live scanning sessions for Sherman QC.
Coordinates file watching, part recognition, and scan accumulation.

Session States:
- IDLE: No active session
- IDENTIFYING: First scan received, attempting to identify part
- SCANNING: Part identified, accumulating scan data
- ANALYZING: Scan complete, running QC analysis
- COMPLETE: Analysis done, results available
- ABANDONED: Session timed out or cancelled

Usage:
    from live_scan_session import LiveScanSessionManager

    manager = LiveScanSessionManager(watch_path="/path/to/exports")
    manager.start()

    # Subscribe to events
    manager.on_session_update = lambda session: print(f"Session update: {session.state}")
"""

import os
import time
import logging
import threading
import uuid
import math
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, List, Any
from enum import Enum

from file_watcher import FileWatcher, FileEvent, SUPPORTED_EXTENSIONS, MIN_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Session lifecycle states."""
    IDLE = "idle"
    IDENTIFYING = "identifying"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    ABANDONED = "abandoned"
    ERROR = "error"


@dataclass
class ScanData:
    """Data from a single scan file."""
    file_path: Path
    filename: str
    size_bytes: int
    received_at: datetime = field(default_factory=datetime.now)
    points_count: Optional[int] = None


@dataclass
class RecognitionResult:
    """Result of part recognition."""
    candidates: List[Dict[str, Any]]
    top_match: Optional[str]
    top_similarity: float
    processing_time_ms: float
    is_confident: bool  # True if top match is significantly better than others


@dataclass
class LiveScanSession:
    """
    Represents a live scanning session.

    A session starts when the first scan file is detected and ends when:
    - Analysis is complete
    - Session times out
    - User cancels
    """
    id: str
    state: SessionState
    created_at: datetime
    updated_at: datetime

    # Part identification
    part_id: Optional[str] = None
    part_number: Optional[str] = None
    recognition_result: Optional[RecognitionResult] = None

    # Scan data
    scans: List[ScanData] = field(default_factory=list)
    total_points: int = 0
    coverage_percent: float = 0.0
    gap_clusters: List[Dict[str, Any]] = field(default_factory=list)

    # Analysis results
    analysis_job_id: Optional[str] = None
    analysis_result: Optional[Dict[str, Any]] = None

    # Errors
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "part_id": self.part_id,
            "part_number": self.part_number,
            "recognition": {
                "candidates": self.recognition_result.candidates if self.recognition_result else [],
                "top_match": self.recognition_result.top_match if self.recognition_result else None,
                "top_similarity": self.recognition_result.top_similarity if self.recognition_result else 0,
                "processing_time_ms": self.recognition_result.processing_time_ms if self.recognition_result else 0,
                "is_confident": self.recognition_result.is_confident if self.recognition_result else False,
            } if self.recognition_result else None,
            "scans": [
                {
                    "filename": s.filename,
                    "size_bytes": s.size_bytes,
                    "received_at": s.received_at.isoformat(),
                    "points_count": s.points_count,
                }
                for s in self.scans
            ],
            "total_points": self.total_points,
            "coverage_percent": self.coverage_percent,
            "gap_clusters": self.gap_clusters,
            "analysis_job_id": self.analysis_job_id,
            "error_message": self.error_message,
        }


# Configuration
DEFAULT_SESSION_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_IDLE_TIMEOUT_SECONDS = 60  # 1 minute without new scans
DEFAULT_CONFIDENCE_THRESHOLD = 0.85  # Similarity threshold for auto-accept


class LiveScanSessionManager:
    """
    Manages live scanning sessions.

    Features:
    - Automatic session creation when first scan is detected
    - Part recognition using embedding search
    - Scan accumulation for coverage calculation
    - Session timeout handling
    - Event callbacks for UI updates
    """

    def __init__(
        self,
        watch_path: str,
        session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize session manager.

        Args:
            watch_path: Directory to watch for scan files
            session_timeout_seconds: Maximum session duration
            idle_timeout_seconds: Timeout after last scan
            confidence_threshold: Similarity threshold for auto-recognition
        """
        self.watch_path = Path(watch_path)
        self.session_timeout_seconds = session_timeout_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.confidence_threshold = confidence_threshold

        # Current session
        self._current_session: Optional[LiveScanSession] = None
        self._session_lock = threading.Lock()

        # File watcher
        self._watcher: Optional[FileWatcher] = None

        # Timeout checker
        self._timeout_thread: Optional[threading.Thread] = None
        self._running = False

        # Event callbacks
        self._on_session_update: Optional[Callable[[LiveScanSession], None]] = None
        self._on_file_received: Optional[Callable[[ScanData], None]] = None

        # History
        self._session_history: List[LiveScanSession] = []
        self._max_history = 100

    @property
    def on_session_update(self) -> Optional[Callable[[LiveScanSession], None]]:
        return self._on_session_update

    @on_session_update.setter
    def on_session_update(self, callback: Callable[[LiveScanSession], None]):
        self._on_session_update = callback

    @property
    def on_file_received(self) -> Optional[Callable[[ScanData], None]]:
        return self._on_file_received

    @on_file_received.setter
    def on_file_received(self, callback: Callable[[ScanData], None]):
        self._on_file_received = callback

    def start(self):
        """Start the session manager."""
        if self._running:
            logger.warning("Session manager already running")
            return

        self._running = True

        # Start file watcher
        self._watcher = FileWatcher(
            str(self.watch_path),
            on_file_ready=self._handle_new_file,
        )
        self._watcher.start()

        # Start timeout checker
        self._timeout_thread = threading.Thread(target=self._run_timeout_checker, daemon=True)
        self._timeout_thread.start()

        logger.info(f"LiveScanSessionManager started: watching {self.watch_path}")

    def stop(self):
        """Stop the session manager."""
        self._running = False

        if self._watcher:
            self._watcher.stop()

        if self._timeout_thread:
            self._timeout_thread.join(timeout=2.0)

        logger.info("LiveScanSessionManager stopped")

    def is_running(self) -> bool:
        """Check whether the manager is actively watching for files."""
        return self._running

    def get_current_session(self) -> Optional[LiveScanSession]:
        """Get the current active session."""
        with self._session_lock:
            return self._current_session

    def get_session_history(self) -> List[LiveScanSession]:
        """Get recent session history."""
        with self._session_lock:
            return list(self._session_history)

    def cancel_session(self) -> bool:
        """Cancel the current session."""
        with self._session_lock:
            if self._current_session is None:
                return False

            self._current_session.state = SessionState.ABANDONED
            self._current_session.updated_at = datetime.now()
            self._emit_session_update()
            self._archive_session()
            return True

    def reset_session(self) -> bool:
        """
        Reset/clear the current session regardless of state.

        Used to start fresh after a completed or errored session.
        """
        with self._session_lock:
            if self._current_session is None:
                return True  # Already clear

            # Archive current session if not already archived
            if self._current_session.state not in [SessionState.COMPLETE, SessionState.ABANDONED]:
                self._current_session.state = SessionState.ABANDONED
                self._current_session.updated_at = datetime.now()

            self._archive_session()
            self._emit_session_update()  # This will emit None since session is now cleared
            return True

    def confirm_part(self, part_id: str, part_number: str) -> bool:
        """
        Manually confirm the part identity.

        Used when auto-recognition is not confident enough.
        """
        with self._session_lock:
            if self._current_session is None:
                return False

            if self._current_session.state not in [SessionState.IDENTIFYING, SessionState.SCANNING]:
                return False

            self._current_session.part_id = part_id
            self._current_session.part_number = part_number
            self._current_session.state = SessionState.SCANNING
            self._current_session.updated_at = datetime.now()
            session_id = self._current_session.id
            has_scans = bool(self._current_session.scans)
            self._emit_session_update()
            if has_scans:
                threading.Thread(
                    target=self._update_coverage,
                    args=(session_id,),
                    daemon=True
                ).start()
            return True

    def ingest_scan_file(self, file_path: str, filename: Optional[str] = None) -> ScanData:
        """
        Ingest a scan file directly (bypass filesystem watcher event loop).

        Useful for API-driven demo/testing flows where a file is generated
        programmatically and should be processed immediately.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Scan file not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported scan format: {ext}")

        size = path.stat().st_size
        if size < MIN_FILE_SIZE_BYTES:
            raise ValueError(
                f"Scan file too small ({size} bytes), minimum {MIN_FILE_SIZE_BYTES} bytes"
            )

        event = FileEvent(
            path=path,
            filename=filename or path.name,
            extension=ext,
            size_bytes=size,
        )
        self._handle_new_file(event)
        return ScanData(
            file_path=path,
            filename=event.filename,
            size_bytes=size,
        )

    def complete_scan(self) -> bool:
        """
        Mark scanning as complete and trigger analysis.
        """
        with self._session_lock:
            if self._current_session is None:
                return False

            if self._current_session.state != SessionState.SCANNING:
                return False

            self._current_session.state = SessionState.ANALYZING
            self._current_session.updated_at = datetime.now()
            self._emit_session_update()

            # TODO: Trigger QC analysis in background
            # For now, just mark as complete
            self._current_session.state = SessionState.COMPLETE
            self._current_session.updated_at = datetime.now()
            self._emit_session_update()
            self._archive_session()
            return True

    def _handle_new_file(self, event: FileEvent):
        """Handle a new file from the file watcher."""
        logger.info(f"New scan file: {event.filename}")

        scan_data = ScanData(
            file_path=event.path,
            filename=event.filename,
            size_bytes=event.size_bytes,
        )

        # Notify about file
        if self._on_file_received:
            try:
                self._on_file_received(scan_data)
            except Exception as e:
                logger.error(f"Error in file received callback: {e}")

        with self._session_lock:
            # Start new session if needed
            if self._current_session is None or self._current_session.state in [
                SessionState.COMPLETE, SessionState.ABANDONED, SessionState.ERROR
            ]:
                self._start_new_session(scan_data)
            else:
                self._add_scan_to_session(scan_data)

    def _start_new_session(self, scan_data: ScanData):
        """Start a new session with the first scan."""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        self._current_session = LiveScanSession(
            id=session_id,
            state=SessionState.IDENTIFYING,
            created_at=now,
            updated_at=now,
            scans=[scan_data],
        )

        logger.info(f"Started new session: {session_id}")
        self._emit_session_update()

        # Attempt recognition in background
        threading.Thread(
            target=self._attempt_recognition,
            args=(session_id, scan_data),
            daemon=True
        ).start()

    def _add_scan_to_session(self, scan_data: ScanData):
        """Add a scan to the current session."""
        if self._current_session is None:
            return

        # Skip duplicate files (common when manual ingest and filesystem watcher both fire).
        for existing in self._current_session.scans:
            if existing.file_path == scan_data.file_path:
                logger.info(f"Skipping duplicate scan event for {scan_data.filename}")
                return

        self._current_session.scans.append(scan_data)
        self._current_session.updated_at = datetime.now()
        self._emit_session_update()

        # Calculate coverage if part is identified
        if self._current_session.part_id:
            session_id = self._current_session.id
            threading.Thread(
                target=self._update_coverage,
                args=(session_id,),
                daemon=True
            ).start()

    def _update_coverage(self, session_id: str):
        """Update coverage calculation for the session."""
        try:
            import open3d as o3d
            import numpy as np
            from part_catalog import get_catalog
            from coverage_calculator import CoverageCalculator

            with self._session_lock:
                if not self._current_session or self._current_session.id != session_id:
                    return
                if not self._current_session.part_id:
                    return

                part_id = self._current_session.part_id
                scan_paths = [s.file_path for s in self._current_session.scans]

            # Get CAD file path
            catalog = get_catalog()
            part = catalog.get_part(part_id)
            if not part or not part.cad_file_path:
                logger.warning(f"No CAD file for part {part_id}")
                return

            # Load and merge scan point clouds
            all_scan_points = []
            total_points = 0
            for path in scan_paths:
                try:
                    pcd = o3d.io.read_point_cloud(str(path))
                    pts = np.asarray(pcd.points) if pcd is not None and not pcd.is_empty() else None
                    if (pts is None or len(pts) == 0) and str(path).lower().endswith(('.stl', '.obj')):
                        mesh = o3d.io.read_triangle_mesh(str(path))
                        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                            sampled = mesh.sample_points_uniformly(number_of_points=12000)
                            pts = np.asarray(sampled.points)
                    if pts is not None and len(pts) > 0:
                        all_scan_points.append(pts)
                        total_points += int(len(pts))
                except Exception as e:
                    logger.error(f"Failed to load scan {path}: {e}")

            if not all_scan_points:
                return

            merged_scan = np.vstack(all_scan_points)

            # Load CAD reference
            cad_path = part.cad_file_path
            cad_points = None
            cad_path_lower = cad_path.lower()
            if cad_path_lower.endswith(('.stl', '.obj', '.ply')):
                mesh = o3d.io.read_triangle_mesh(cad_path)
                if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                    cad_pcd = mesh.sample_points_uniformly(number_of_points=50000)
                    if cad_pcd is not None and not cad_pcd.is_empty():
                        cad_points = np.asarray(cad_pcd.points)
                if cad_points is None or len(cad_points) == 0:
                    pcd = o3d.io.read_point_cloud(cad_path)
                    if pcd is not None and not pcd.is_empty():
                        cad_points = np.asarray(pcd.points)
            else:
                pcd = o3d.io.read_point_cloud(cad_path)
                if pcd is not None and not pcd.is_empty():
                    cad_points = np.asarray(pcd.points)

            if cad_points is None or len(cad_points) == 0:
                logger.error(f"Failed to load CAD file: {cad_path}")
                return

            # Calculate coverage
            calculator = CoverageCalculator(voxel_size=2.0, tolerance=3.0)
            coverage_result = calculator.compute_coverage(cad_points, merged_scan)
            coverage_percent = float(coverage_result.coverage_percent)
            gap_clusters = coverage_result.gap_clusters

            # Fallback: canonicalized overlap proxy when absolute alignment is poor.
            if coverage_percent < 1.0:
                proxy_coverage = self._coverage_proxy(cad_points, merged_scan, voxel_norm=0.08)
                if proxy_coverage > coverage_percent:
                    coverage_percent = proxy_coverage
                    # Proxy mode cannot localize physical gaps reliably.
                    gap_clusters = []

            # Update session
            with self._session_lock:
                if self._current_session and self._current_session.id == session_id:
                    self._current_session.coverage_percent = coverage_percent
                    self._current_session.total_points = total_points
                    self._current_session.gap_clusters = gap_clusters
                    self._current_session.updated_at = datetime.now()
                    self._emit_session_update()

            logger.info(
                f"Coverage updated: {coverage_percent:.1f}%, "
                f"gaps: {len(gap_clusters)}"
            )

        except Exception as e:
            logger.error(f"Coverage calculation failed: {e}")
            import traceback
            traceback.print_exc()

    def _attempt_recognition(self, session_id: str, scan_data: ScanData):
        """Attempt to recognize the part from the scan."""
        try:
            # Import here to avoid circular imports
            from embedding_service import compute_embedding
            from faiss_index import get_faiss_index
            from part_catalog import get_catalog
            import open3d as o3d
            import numpy as np
            import time

            start_time = time.time()

            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(scan_data.file_path))
            if pcd.is_empty():
                logger.error(f"Failed to load point cloud: {scan_data.file_path}")
                return

            points = np.asarray(pcd.points)
            scan_data.points_count = len(points)

            # Compute embedding
            emb_result = compute_embedding(points)

            # Search FAISS
            faiss_index = get_faiss_index()
            search_results = faiss_index.search(emb_result.embedding, k=5, threshold=0.0)

            # Get part details
            catalog = get_catalog()
            candidates = []

            for result in search_results:
                part = catalog.get_part(result.part_id)
                if part:
                    # Check if part has CAD file
                    has_cad = bool(part.cad_file_path)

                    # Add warning if similarity is borderline
                    warning = None
                    if result.similarity < 0.7:
                        warning = "Low confidence match"
                    elif not has_cad:
                        warning = "No CAD file available"

                    candidates.append({
                        "part_id": result.part_id,
                        "part_number": part.part_number,
                        "part_name": part.part_name,
                        "customer": part.customer,
                        "similarity": result.similarity,
                        "distance": result.distance,
                        "has_cad": has_cad,
                        "warning": warning,
                    })

            # Calibrate display confidence so alternatives do not appear as overconfident.
            self._calibrate_candidate_confidence(candidates)

            processing_time = (time.time() - start_time) * 1000

            # Determine confidence
            top_match = None
            top_similarity = 0.0
            is_confident = False

            if candidates:
                top_match = candidates[0]["part_number"]
                top_similarity = candidates[0]["similarity"]

                # Confident if top match is significantly better than #2
                if len(candidates) >= 2:
                    gap = candidates[0]["similarity"] - candidates[1]["similarity"]
                    is_confident = top_similarity >= self.confidence_threshold and gap >= 0.05
                else:
                    is_confident = top_similarity >= self.confidence_threshold

            recognition = RecognitionResult(
                candidates=candidates,
                top_match=top_match,
                top_similarity=top_similarity,
                processing_time_ms=processing_time,
                is_confident=is_confident,
            )

            # Update session
            with self._session_lock:
                if self._current_session and self._current_session.id == session_id:
                    self._current_session.recognition_result = recognition
                    self._current_session.total_points = scan_data.points_count

                    if is_confident and candidates:
                        # Auto-confirm part
                        self._current_session.part_id = candidates[0]["part_id"]
                        self._current_session.part_number = candidates[0]["part_number"]
                        self._current_session.state = SessionState.SCANNING
                    else:
                        # Need manual confirmation
                        self._current_session.state = SessionState.IDENTIFYING

                    self._current_session.updated_at = datetime.now()
                    self._emit_session_update()

            # If auto-confirmed, calculate coverage immediately from existing scans.
            if is_confident and candidates:
                threading.Thread(
                    target=self._update_coverage,
                    args=(session_id,),
                    daemon=True
                ).start()

            logger.info(
                f"Recognition complete: top={top_match} ({top_similarity:.3f}), "
                f"confident={is_confident}, time={processing_time:.0f}ms"
            )

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            import traceback
            traceback.print_exc()

            with self._session_lock:
                if self._current_session and self._current_session.id == session_id:
                    self._current_session.state = SessionState.ERROR
                    self._current_session.error_message = str(e)
                    self._emit_session_update()

    def _calibrate_candidate_confidence(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Add `match_confidence` to recognition candidates for UI display.

        This blends absolute similarity, relative rank, and softmax posterior so
        top matches stay high while secondary options are de-emphasized.
        """
        if not candidates:
            return
        import numpy as np

        sims = []
        for c in candidates:
            s = float(c.get("similarity", 0.0))
            if not math.isfinite(s):
                s = 0.0
            sims.append(max(0.0, min(1.0, s)))
        if not sims:
            return

        top_sim = max(1e-6, sims[0])

        # Numerically stable softmax over centered similarities.
        scale = 20.0
        logits = np.asarray([(s - 0.80) * scale for s in sims], dtype=np.float64)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        denom = float(np.sum(exp_logits)) if np.isfinite(np.sum(exp_logits)) else 0.0
        probs = exp_logits / denom if denom > 1e-12 else np.zeros_like(exp_logits)

        for idx, c in enumerate(candidates):
            sim = sims[idx]
            if idx == 0:
                confidence = sim
            else:
                rel = sim / top_sim
                rank_penalty = max(0.40, 1.0 - 0.18 * idx)
                calibrated = sim * rel * rank_penalty
                confidence = 0.55 * calibrated + 0.45 * float(probs[idx])
            c["match_confidence"] = float(max(0.0, min(1.0, confidence)))

    def _canonicalize_points(self, points):
        """Center and normalize point cloud for orientation-invariant overlap proxy."""
        import numpy as np

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)

        center = np.mean(pts, axis=0)
        centered = pts - center
        scale = float(np.max(np.abs(centered)))
        if not np.isfinite(scale) or scale < 1e-9:
            return np.zeros_like(centered)
        normalized = centered / scale

        try:
            cov = (normalized.T @ normalized) / max(1, len(normalized) - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            basis = eigvecs[:, order]
            coords = normalized @ basis
        except Exception:
            coords = normalized

        lo = np.percentile(coords, 2.0, axis=0)
        hi = np.percentile(coords, 98.0, axis=0)
        mid = 0.5 * (lo + hi)
        span = np.maximum(hi - lo, 1e-6)
        out = (coords - mid) / span
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _voxel_set(self, points, voxel_norm: float):
        """Convert normalized points to integer voxel occupancy."""
        import numpy as np

        if points is None or len(points) == 0:
            return set()
        voxel = max(1e-4, float(voxel_norm))
        idx = np.floor(points / voxel).astype(np.int32)
        uniq = np.unique(idx, axis=0)
        return {tuple(map(int, row)) for row in uniq}

    def _coverage_proxy(self, cad_points, scan_points, voxel_norm: float = 0.08) -> float:
        """Alignment-agnostic CAD/scan overlap proxy in percent."""
        import numpy as np

        cad_norm = self._canonicalize_points(cad_points)
        scan_norm = self._canonicalize_points(scan_points)
        cad_vox = self._voxel_set(cad_norm, voxel_norm)
        if not cad_vox or len(scan_norm) == 0:
            return 0.0

        best_cov = 0.0
        for perm in itertools.permutations((0, 1, 2)):
            permuted = scan_norm[:, perm]
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                transformed = permuted * np.asarray(signs, dtype=np.float64)
                scan_vox = self._voxel_set(transformed, voxel_norm)
                if not scan_vox:
                    continue
                inter = len(cad_vox.intersection(scan_vox))
                cov = inter / max(1, len(cad_vox))
                if cov > best_cov:
                    best_cov = cov
        return float(best_cov * 100.0)

    def _run_timeout_checker(self):
        """Background thread to check for session timeouts."""
        while self._running:
            with self._session_lock:
                if self._current_session and self._current_session.state not in [
                    SessionState.COMPLETE, SessionState.ABANDONED, SessionState.ERROR
                ]:
                    now = datetime.now()

                    # Check session timeout
                    session_age = (now - self._current_session.created_at).total_seconds()
                    if session_age > self.session_timeout_seconds:
                        logger.info(f"Session timeout: {self._current_session.id}")
                        self._current_session.state = SessionState.ABANDONED
                        self._current_session.error_message = "Session timeout"
                        self._emit_session_update()
                        self._archive_session()
                        continue

                    # Check idle timeout
                    idle_time = (now - self._current_session.updated_at).total_seconds()
                    if idle_time > self.idle_timeout_seconds:
                        logger.info(f"Idle timeout: {self._current_session.id}")
                        # For now, just log - could trigger auto-complete

            time.sleep(5)  # Check every 5 seconds

    def _emit_session_update(self):
        """Emit session update event."""
        if self._on_session_update and self._current_session:
            try:
                self._on_session_update(self._current_session)
            except Exception as e:
                logger.error(f"Error in session update callback: {e}")

    def _archive_session(self):
        """Move current session to history."""
        if self._current_session:
            self._session_history.append(self._current_session)

            # Trim history
            if len(self._session_history) > self._max_history:
                self._session_history = self._session_history[-self._max_history:]

            self._current_session = None


# Test code
if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("="*60)
    print("LIVE SCAN SESSION MANAGER TEST")
    print("="*60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nWatch path: {tmpdir}")

        def on_update(session: LiveScanSession):
            print(f"  SESSION UPDATE: state={session.state.value}, scans={len(session.scans)}")

        def on_file(scan: ScanData):
            print(f"  FILE RECEIVED: {scan.filename}")

        manager = LiveScanSessionManager(
            tmpdir,
            idle_timeout_seconds=10,
        )
        manager.on_session_update = on_update
        manager.on_file_received = on_file

        manager.start()

        print("\n[1] Creating test scan file...")
        test_file = Path(tmpdir) / "TEST_PART_001.ply"

        # Create minimal PLY
        ply_content = "ply\nformat ascii 1.0\nelement vertex 100\n"
        ply_content += "property float x\nproperty float y\nproperty float z\nend_header\n"
        for i in range(100):
            ply_content += f"{i} {i} {i}\n"

        test_file.write_text(ply_content)
        print(f"    Created: {test_file.name}")

        print("\n[2] Waiting for processing...")
        time.sleep(4)

        session = manager.get_current_session()
        if session:
            print(f"\n[3] Current session:")
            print(f"    ID: {session.id}")
            print(f"    State: {session.state.value}")
            print(f"    Scans: {len(session.scans)}")
        else:
            print("\n[3] No active session")

        manager.stop()
        print("\nTest complete!")
