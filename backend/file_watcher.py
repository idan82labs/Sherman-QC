"""
File Watcher Service for Live Scanning

Monitors a folder for new point cloud files (PLY, STL) exported from VXelements.
Handles debouncing, file locking, and emits events when files are ready.

Usage:
    from file_watcher import FileWatcher, FileEvent

    def on_new_file(event: FileEvent):
        print(f"New file: {event.path}")

    watcher = FileWatcher("/path/to/watch", on_file_ready=on_new_file)
    watcher.start()
"""

import os
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Dict, List
from datetime import datetime
from collections import defaultdict

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    Observer = None
    FileSystemEventHandler = object

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_DEBOUNCE_SECONDS = 2.0  # Wait for file to stabilize
DEFAULT_FILE_STABLE_SECONDS = 1.0  # Time since last modification to consider file stable
SUPPORTED_EXTENSIONS = {'.ply', '.stl', '.obj', '.pcd'}
MIN_FILE_SIZE_BYTES = 500  # Minimum valid file size


@dataclass
class FileEvent:
    """Event emitted when a new file is ready for processing."""
    path: Path
    filename: str
    extension: str
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def part_number_guess(self) -> str:
        """
        Attempt to extract part number from filename.
        VXelements naming: "44211000_A.ply" or "44211000_A_001.ply"
        """
        name = self.path.stem
        # Remove common suffixes like _001, _002, etc.
        import re
        name = re.sub(r'_\d{3}$', '', name)
        return name


class FileEventHandler(FileSystemEventHandler):
    """
    Handles file system events with debouncing and stability checks.
    """

    def __init__(
        self,
        callback: Callable[[FileEvent], None],
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        stable_seconds: float = DEFAULT_FILE_STABLE_SECONDS,
        extensions: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.stable_seconds = stable_seconds
        self.extensions = extensions or SUPPORTED_EXTENSIONS

        # Track file states
        self._file_states: Dict[str, dict] = {}
        self._pending_files: Set[str] = set()
        self._lock = threading.Lock()

        # Background thread for stability checks
        self._checker_running = False
        self._checker_thread: Optional[threading.Thread] = None

    def start_checker(self):
        """Start background thread for file stability checks."""
        if not self._checker_running:
            self._checker_running = True
            self._checker_thread = threading.Thread(target=self._run_checker, daemon=True)
            self._checker_thread.start()

    def stop_checker(self):
        """Stop background checker thread."""
        self._checker_running = False
        if self._checker_thread:
            self._checker_thread.join(timeout=2.0)

    def _run_checker(self):
        """Background loop to check file stability."""
        while self._checker_running:
            self._check_pending_files()
            time.sleep(0.5)

    def _check_pending_files(self):
        """Check if pending files are stable and ready."""
        with self._lock:
            now = time.time()
            ready_files = []

            for filepath in list(self._pending_files):
                state = self._file_states.get(filepath)
                if not state:
                    continue

                # Check if file is stable (no changes for stable_seconds)
                time_since_change = now - state['last_modified']
                if time_since_change < self.stable_seconds:
                    continue

                # Check if debounce period has passed
                time_since_first = now - state['first_seen']
                if time_since_first < self.debounce_seconds:
                    continue

                # Check if file is accessible (not locked)
                path = Path(filepath)
                if not self._is_file_accessible(path):
                    continue

                # Check minimum file size
                try:
                    size = path.stat().st_size
                    if size < MIN_FILE_SIZE_BYTES:
                        logger.debug(f"File too small, skipping: {filepath} ({size} bytes)")
                        self._pending_files.discard(filepath)
                        continue
                except:
                    continue

                ready_files.append((filepath, size))

            # Process ready files outside lock
            for filepath, size in ready_files:
                self._pending_files.discard(filepath)
                del self._file_states[filepath]

        # Emit events outside lock
        for filepath, size in ready_files:
            self._emit_event(Path(filepath), size)

    def _is_file_accessible(self, path: Path) -> bool:
        """Check if file can be opened (not locked by another process)."""
        try:
            with open(path, 'rb') as f:
                # Try to read first few bytes
                f.read(100)
            return True
        except (IOError, OSError):
            return False

    def _emit_event(self, path: Path, size: int):
        """Emit file ready event."""
        try:
            event = FileEvent(
                path=path,
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=size,
            )

            logger.info(f"File ready: {path.name} ({size/1024:.1f} KB)")
            self.callback(event)

        except Exception as e:
            logger.error(f"Error emitting file event for {path}: {e}")

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in self.extensions:
            return

        logger.debug(f"File created: {path.name}")
        self._track_file(str(path))

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in self.extensions:
            return

        logger.debug(f"File modified: {path.name}")
        self._track_file(str(path))

    def _track_file(self, filepath: str):
        """Track file for stability checking."""
        with self._lock:
            now = time.time()

            if filepath not in self._file_states:
                self._file_states[filepath] = {
                    'first_seen': now,
                    'last_modified': now,
                }
            else:
                self._file_states[filepath]['last_modified'] = now

            self._pending_files.add(filepath)


class FileWatcher:
    """
    Watches a directory for new point cloud files.

    Features:
    - Debounces rapid changes (VXelements writes multiple times)
    - Waits for file to be stable (no changes for 1 second)
    - Handles file locking (waits for file to be accessible)
    - Filters by extension (.ply, .stl, .obj, .pcd)
    - Emits events via callback

    Usage:
        watcher = FileWatcher("/path/to/watch", on_file_ready=my_callback)
        watcher.start()
        # ... later ...
        watcher.stop()
    """

    def __init__(
        self,
        watch_path: str,
        on_file_ready: Callable[[FileEvent], None],
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        stable_seconds: float = DEFAULT_FILE_STABLE_SECONDS,
        extensions: Optional[Set[str]] = None,
    ):
        """
        Initialize file watcher.

        Args:
            watch_path: Directory to monitor
            on_file_ready: Callback when file is ready for processing
            debounce_seconds: Minimum time to wait after first detection
            stable_seconds: Time since last modification to consider stable
            extensions: File extensions to watch (default: .ply, .stl, .obj, .pcd)
        """
        if not HAS_WATCHDOG:
            raise ImportError(
                "watchdog library not installed. Install with: pip install watchdog"
            )

        self.watch_path = Path(watch_path)
        if not self.watch_path.exists():
            raise ValueError(f"Watch path does not exist: {watch_path}")
        if not self.watch_path.is_dir():
            raise ValueError(f"Watch path is not a directory: {watch_path}")

        self.on_file_ready = on_file_ready
        self.debounce_seconds = debounce_seconds
        self.stable_seconds = stable_seconds
        self.extensions = extensions or SUPPORTED_EXTENSIONS

        self._observer: Optional[Observer] = None
        self._handler: Optional[FileEventHandler] = None
        self._running = False

        # Statistics
        self.stats = {
            'started_at': None,
            'files_detected': 0,
            'files_processed': 0,
            'last_file': None,
        }

    def start(self):
        """Start watching the directory."""
        if self._running:
            logger.warning("FileWatcher already running")
            return

        # Create callback wrapper to track stats
        def callback_with_stats(event: FileEvent):
            self.stats['files_processed'] += 1
            self.stats['last_file'] = event.filename
            self.on_file_ready(event)

        self._handler = FileEventHandler(
            callback=callback_with_stats,
            debounce_seconds=self.debounce_seconds,
            stable_seconds=self.stable_seconds,
            extensions=self.extensions,
        )

        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.watch_path), recursive=False)

        self._observer.start()
        self._handler.start_checker()
        self._running = True

        self.stats['started_at'] = datetime.now().isoformat()

        logger.info(
            f"FileWatcher started: watching {self.watch_path} "
            f"(debounce={self.debounce_seconds}s, stable={self.stable_seconds}s)"
        )

    def stop(self):
        """Stop watching the directory."""
        if not self._running:
            return

        if self._handler:
            self._handler.stop_checker()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)

        self._running = False
        logger.info("FileWatcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get watcher statistics."""
        return self.stats.copy()

    def scan_existing(self) -> List[FileEvent]:
        """
        Scan for existing files in the watch directory.
        Returns list of FileEvents for files that match the criteria.
        """
        events = []

        for path in self.watch_path.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.extensions:
                continue

            try:
                size = path.stat().st_size
                if size < MIN_FILE_SIZE_BYTES:
                    continue

                events.append(FileEvent(
                    path=path,
                    filename=path.name,
                    extension=path.suffix.lower(),
                    size_bytes=size,
                ))
            except Exception as e:
                logger.warning(f"Error scanning file {path}: {e}")

        return events


# Convenience function for creating and starting watcher
def create_file_watcher(
    watch_path: str,
    on_file_ready: Callable[[FileEvent], None],
    **kwargs
) -> FileWatcher:
    """
    Create and return a FileWatcher instance.

    Args:
        watch_path: Directory to monitor
        on_file_ready: Callback when file is ready
        **kwargs: Additional arguments for FileWatcher

    Returns:
        FileWatcher instance (not started)
    """
    return FileWatcher(watch_path, on_file_ready, **kwargs)


# Test code
if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("="*60)
    print("FILE WATCHER TEST")
    print("="*60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nWatching: {tmpdir}")

        events_received = []

        def on_file(event: FileEvent):
            events_received.append(event)
            print(f"  EVENT: {event.filename} ({event.size_bytes} bytes)")
            print(f"         Part number guess: {event.part_number_guess}")

        watcher = FileWatcher(tmpdir, on_file_ready=on_file, debounce_seconds=1.0)
        watcher.start()

        print("\n[1] Creating test file...")
        test_file = Path(tmpdir) / "44211000_A.ply"

        # Create PLY header
        ply_content = """ply
format ascii 1.0
element vertex 100
property float x
property float y
property float z
end_header
"""
        # Add some vertices
        for i in range(100):
            ply_content += f"{i} {i} {i}\n"

        test_file.write_text(ply_content)
        print(f"    Created: {test_file.name} ({len(ply_content)} bytes)")

        print("\n[2] Waiting for debounce period...")
        time.sleep(3)

        watcher.stop()

        print(f"\n[3] Results:")
        print(f"    Events received: {len(events_received)}")
        print(f"    Stats: {watcher.get_stats()}")

        if events_received:
            print("\n    ✓ File watcher working correctly!")
        else:
            print("\n    ✗ No events received")
            sys.exit(1)
