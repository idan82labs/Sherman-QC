"""
Part Catalog Module - Database Models and Manager for Part Library

Provides:
- Part master data (part number, name, customer, CAD file)
- Point cloud embeddings for auto-recognition
- Bend specifications per part
- Live scan session tracking

This module extends the existing database infrastructure to support
the live scan feature with automatic part recognition.
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Use same database path as main database
DATABASE_PATH = Path(__file__).parent.parent / "data" / "qc_jobs.db"


@dataclass
class Part:
    """Part master data record"""
    id: str
    part_number: str
    part_name: str
    customer: Optional[str] = None
    revision: Optional[str] = None

    # CAD file information
    cad_file_path: Optional[str] = None
    cad_file_hash: Optional[str] = None  # For change detection
    cad_uploaded_at: Optional[str] = None

    # Specifications
    material: Optional[str] = None
    default_tolerance_angle: float = 1.0  # degrees
    default_tolerance_linear: float = 0.5  # mm
    bend_count: Optional[int] = None  # Detected from CAD

    # Auto-recognition embedding
    has_embedding: bool = False
    embedding_version: Optional[str] = None
    embedding_computed_at: Optional[str] = None

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    erp_import_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to API-friendly dict"""
        return asdict(self)

    @property
    def is_ready_for_recognition(self) -> bool:
        """Check if part has CAD and embedding for auto-recognition"""
        return self.cad_file_path is not None and self.has_embedding

    @property
    def status(self) -> str:
        """Get part status string"""
        if not self.cad_file_path:
            return "needs_cad"
        if not self.has_embedding:
            return "needs_embedding"
        return "ready"


@dataclass
class PartBendSpec:
    """Bend specification for a part"""
    id: str
    part_id: str
    bend_id: str  # e.g., "B1", "B2"

    target_angle: float  # degrees
    target_radius: Optional[float] = None  # mm
    tolerance_angle: Optional[float] = None  # degrees, overrides part default
    tolerance_radius: Optional[float] = None  # mm

    # Location hint for matching
    bend_line_start: Optional[str] = None  # JSON [x, y, z]
    bend_line_end: Optional[str] = None  # JSON [x, y, z]

    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LiveScanSession:
    """Live scanning session record"""
    id: str

    # Part association
    part_id: Optional[str] = None
    part_number: Optional[str] = None  # Denormalized for convenience
    recognition_confidence: Optional[float] = None

    # Session state
    status: str = "started"  # started, identifying, scanning, analyzing, completed, abandoned

    # Scan data
    scan_file_path: Optional[str] = None
    point_count: int = 0
    coverage_percent: float = 0.0

    # Recognition results (JSON)
    recognition_results: Optional[str] = None

    # Analysis results
    analysis_job_id: Optional[str] = None

    # Timing
    started_at: str = ""
    last_update_at: str = ""
    completed_at: Optional[str] = None

    # Operator info
    operator_id: Optional[str] = None
    workstation_id: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.recognition_results:
            try:
                data["recognition_results"] = json.loads(self.recognition_results)
            except json.JSONDecodeError:
                pass
        return data


class PartCatalogManager:
    """
    Manages the part catalog database.

    Uses SQLite for simplicity, stores embeddings as BLOB.
    """

    EMBEDDING_DIM = 1024  # PointNet embedding dimension
    CURRENT_EMBEDDING_VERSION = "pointnet_v1"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize part catalog tables"""
        with self.get_connection() as conn:
            # Parts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parts (
                    id TEXT PRIMARY KEY,
                    part_number TEXT UNIQUE NOT NULL,
                    part_name TEXT,
                    customer TEXT,
                    revision TEXT,

                    cad_file_path TEXT,
                    cad_file_hash TEXT,
                    cad_uploaded_at TEXT,

                    material TEXT,
                    default_tolerance_angle REAL DEFAULT 1.0,
                    default_tolerance_linear REAL DEFAULT 0.5,
                    bend_count INTEGER,

                    has_embedding INTEGER DEFAULT 0,
                    embedding_version TEXT,
                    embedding_computed_at TEXT,

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    erp_import_id TEXT,
                    notes TEXT
                )
            """)

            # Part embeddings table (separate for efficiency)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS part_embeddings (
                    part_id TEXT PRIMARY KEY REFERENCES parts(id) ON DELETE CASCADE,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Part bend specifications
            conn.execute("""
                CREATE TABLE IF NOT EXISTS part_bend_specs (
                    id TEXT PRIMARY KEY,
                    part_id TEXT NOT NULL REFERENCES parts(id) ON DELETE CASCADE,
                    bend_id TEXT NOT NULL,

                    target_angle REAL NOT NULL,
                    target_radius REAL,
                    tolerance_angle REAL,
                    tolerance_radius REAL,

                    bend_line_start TEXT,
                    bend_line_end TEXT,
                    notes TEXT,

                    UNIQUE(part_id, bend_id)
                )
            """)

            # Live scan sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_scan_sessions (
                    id TEXT PRIMARY KEY,

                    part_id TEXT REFERENCES parts(id),
                    part_number TEXT,
                    recognition_confidence REAL,

                    status TEXT NOT NULL DEFAULT 'started',

                    scan_file_path TEXT,
                    point_count INTEGER DEFAULT 0,
                    coverage_percent REAL DEFAULT 0.0,

                    recognition_results TEXT,
                    analysis_job_id TEXT,

                    started_at TEXT NOT NULL,
                    last_update_at TEXT NOT NULL,
                    completed_at TEXT,

                    operator_id TEXT,
                    workstation_id TEXT
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parts_part_number ON parts(part_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parts_customer ON parts(customer)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parts_has_embedding ON parts(has_embedding)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_part_bend_specs_part_id ON part_bend_specs(part_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_live_scan_sessions_status ON live_scan_sessions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_live_scan_sessions_part_id ON live_scan_sessions(part_id)")

            logger.info(f"Part catalog initialized at {self.db_path}")

    # ==================== Part CRUD ====================

    def create_part(
        self,
        part_number: str,
        part_name: Optional[str] = None,
        customer: Optional[str] = None,
        revision: Optional[str] = None,
        material: Optional[str] = None,
        default_tolerance_angle: float = 1.0,
        default_tolerance_linear: float = 0.5,
        erp_import_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Part:
        """Create a new part in the catalog"""
        part_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO parts (
                    id, part_number, part_name, customer, revision,
                    material, default_tolerance_angle, default_tolerance_linear,
                    erp_import_id, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                part_id, part_number, part_name, customer, revision,
                material, default_tolerance_angle, default_tolerance_linear,
                erp_import_id, notes, now, now
            ))

        return self.get_part(part_id)

    def get_part(self, part_id: str) -> Optional[Part]:
        """Get part by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM parts WHERE id = ?", (part_id,)
            ).fetchone()

            if row:
                return self._row_to_part(row)
            return None

    def get_part_by_number(self, part_number: str) -> Optional[Part]:
        """Get part by part number"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM parts WHERE part_number = ?", (part_number,)
            ).fetchone()

            if row:
                return self._row_to_part(row)
            return None

    def update_part(
        self,
        part_id: str,
        **kwargs
    ) -> Optional[Part]:
        """Update part fields"""
        allowed_fields = {
            'part_name', 'customer', 'revision', 'material',
            'default_tolerance_angle', 'default_tolerance_linear',
            'notes', 'cad_file_path', 'cad_file_hash', 'cad_uploaded_at',
            'bend_count'
        }

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_part(part_id)

        updates['updated_at'] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [part_id]

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE parts SET {set_clause} WHERE id = ?",
                values
            )

        return self.get_part(part_id)

    def delete_part(self, part_id: str) -> bool:
        """Delete part and all related data"""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM parts WHERE id = ?", (part_id,))
            return cursor.rowcount > 0

    def set_embedding(
        self,
        part_id: str,
        embedding: np.ndarray,
        version: str,
    ) -> bool:
        """
        Set the embedding for a part.

        Args:
            part_id: Part ID
            embedding: 1024-dim embedding vector
            version: Embedding version string

        Returns:
            True if updated successfully
        """
        embedding_blob = embedding.tobytes()
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE parts SET
                    has_embedding = 1,
                    embedding_version = ?,
                    embedding_computed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (version, now, now, part_id)
            )

            if cursor.rowcount > 0:
                # Store embedding in separate table or blob storage
                # For now, we just update the has_embedding flag
                # Actual embedding is stored in FAISS index
                conn.commit()
                return True

            return False

    def list_parts(
        self,
        search: Optional[str] = None,
        customer: Optional[str] = None,
        has_cad: Optional[bool] = None,
        has_embedding: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Part]:
        """List parts with filters"""
        query = "SELECT * FROM parts WHERE 1=1"
        params = []

        if search:
            query += " AND (part_number LIKE ? OR part_name LIKE ?)"
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern])

        if customer:
            query += " AND customer = ?"
            params.append(customer)

        if has_cad is not None:
            if has_cad:
                query += " AND cad_file_path IS NOT NULL"
            else:
                query += " AND cad_file_path IS NULL"

        if has_embedding is not None:
            query += " AND has_embedding = ?"
            params.append(1 if has_embedding else 0)

        query += " ORDER BY part_number ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_part(row) for row in rows]

    def count_parts_filtered(
        self,
        search: Optional[str] = None,
        customer: Optional[str] = None,
        has_cad: Optional[bool] = None,
        has_embedding: Optional[bool] = None,
    ) -> int:
        """Count parts matching filters (for pagination)"""
        query = "SELECT COUNT(*) FROM parts WHERE 1=1"
        params = []

        if search:
            query += " AND (part_number LIKE ? OR part_name LIKE ?)"
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern])

        if customer:
            query += " AND customer = ?"
            params.append(customer)

        if has_cad is not None:
            if has_cad:
                query += " AND cad_file_path IS NOT NULL"
            else:
                query += " AND cad_file_path IS NULL"

        if has_embedding is not None:
            query += " AND has_embedding = ?"
            params.append(1 if has_embedding else 0)

        with self.get_connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def count_parts(
        self,
        has_cad: Optional[bool] = None,
        has_embedding: Optional[bool] = None,
    ) -> Dict[str, int]:
        """Get part counts by status"""
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
            with_cad = conn.execute(
                "SELECT COUNT(*) FROM parts WHERE cad_file_path IS NOT NULL"
            ).fetchone()[0]
            with_embedding = conn.execute(
                "SELECT COUNT(*) FROM parts WHERE has_embedding = 1"
            ).fetchone()[0]
            # Ready = has both CAD and embedding
            ready = conn.execute(
                "SELECT COUNT(*) FROM parts WHERE cad_file_path IS NOT NULL AND has_embedding = 1"
            ).fetchone()[0]
            # Needs embedding = has CAD but no embedding
            needs_embedding = conn.execute(
                "SELECT COUNT(*) FROM parts WHERE cad_file_path IS NOT NULL AND has_embedding = 0"
            ).fetchone()[0]

            return {
                "total": total,
                "with_cad": with_cad,
                "with_embedding": with_embedding,
                "ready": ready,
                "needs_cad": total - with_cad,
                "needs_embedding": needs_embedding,
            }

    def _row_to_part(self, row) -> Part:
        """Convert database row to Part"""
        return Part(
            id=row["id"],
            part_number=row["part_number"],
            part_name=row["part_name"],
            customer=row["customer"],
            revision=row["revision"],
            cad_file_path=row["cad_file_path"],
            cad_file_hash=row["cad_file_hash"],
            cad_uploaded_at=row["cad_uploaded_at"],
            material=row["material"],
            default_tolerance_angle=row["default_tolerance_angle"] or 1.0,
            default_tolerance_linear=row["default_tolerance_linear"] or 0.5,
            bend_count=row["bend_count"],
            has_embedding=bool(row["has_embedding"]),
            embedding_version=row["embedding_version"],
            embedding_computed_at=row["embedding_computed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            erp_import_id=row["erp_import_id"],
            notes=row["notes"],
        )

    # ==================== Embeddings ====================

    def store_embedding(
        self,
        part_id: str,
        embedding: np.ndarray,
        version: str = None,
    ) -> bool:
        """Store embedding for a part"""
        version = version or self.CURRENT_EMBEDDING_VERSION
        now = datetime.now().isoformat()

        # Validate embedding
        if embedding.shape != (self.EMBEDDING_DIM,):
            raise ValueError(f"Expected embedding dim {self.EMBEDDING_DIM}, got {embedding.shape}")

        # Convert to bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()

        with self.get_connection() as conn:
            # Upsert embedding
            conn.execute("""
                INSERT INTO part_embeddings (part_id, embedding, embedding_dim, embedding_version, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(part_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    embedding_dim = excluded.embedding_dim,
                    embedding_version = excluded.embedding_version,
                    created_at = excluded.created_at
            """, (part_id, embedding_bytes, self.EMBEDDING_DIM, version, now))

            # Update part record
            conn.execute("""
                UPDATE parts SET
                    has_embedding = 1,
                    embedding_version = ?,
                    embedding_computed_at = ?,
                    updated_at = ?
                WHERE id = ?
            """, (version, now, now, part_id))

        return True

    def get_embedding(self, part_id: str) -> Optional[np.ndarray]:
        """Get embedding for a part"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT embedding, embedding_dim FROM part_embeddings WHERE part_id = ?",
                (part_id,)
            ).fetchone()

            if row:
                embedding_bytes = row["embedding"]
                dim = row["embedding_dim"]
                return np.frombuffer(embedding_bytes, dtype=np.float32).reshape(dim)
            return None

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all embeddings for building FAISS index.

        Returns:
            Tuple of (part_ids, embeddings_array)
        """
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT part_id, embedding FROM part_embeddings
                ORDER BY part_id
            """).fetchall()

            if not rows:
                return [], np.array([])

            part_ids = [row["part_id"] for row in rows]
            embeddings = np.array([
                np.frombuffer(row["embedding"], dtype=np.float32)
                for row in rows
            ])

            return part_ids, embeddings

    def delete_embedding(self, part_id: str) -> bool:
        """Delete embedding for a part"""
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM part_embeddings WHERE part_id = ?",
                (part_id,)
            )
            conn.execute("""
                UPDATE parts SET
                    has_embedding = 0,
                    embedding_version = NULL,
                    embedding_computed_at = NULL,
                    updated_at = ?
                WHERE id = ?
            """, (now, part_id))

        return True

    # ==================== Bend Specs ====================

    def add_bend_spec(
        self,
        part_id: str,
        bend_id: str,
        target_angle: float,
        target_radius: Optional[float] = None,
        tolerance_angle: Optional[float] = None,
        tolerance_radius: Optional[float] = None,
        bend_line_start: Optional[List[float]] = None,
        bend_line_end: Optional[List[float]] = None,
        notes: Optional[str] = None,
    ) -> PartBendSpec:
        """Add a bend specification to a part"""
        spec_id = str(uuid.uuid4())

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO part_bend_specs (
                    id, part_id, bend_id, target_angle, target_radius,
                    tolerance_angle, tolerance_radius,
                    bend_line_start, bend_line_end, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                spec_id, part_id, bend_id, target_angle, target_radius,
                tolerance_angle, tolerance_radius,
                json.dumps(bend_line_start) if bend_line_start else None,
                json.dumps(bend_line_end) if bend_line_end else None,
                notes
            ))

        return self.get_bend_spec(spec_id)

    def get_bend_spec(self, spec_id: str) -> Optional[PartBendSpec]:
        """Get bend spec by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM part_bend_specs WHERE id = ?", (spec_id,)
            ).fetchone()

            if row:
                return self._row_to_bend_spec(row)
            return None

    def get_bend_specs_for_part(self, part_id: str) -> List[PartBendSpec]:
        """Get all bend specs for a part"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM part_bend_specs WHERE part_id = ? ORDER BY bend_id",
                (part_id,)
            ).fetchall()

            return [self._row_to_bend_spec(row) for row in rows]

    def delete_bend_specs_for_part(self, part_id: str) -> int:
        """Delete all bend specs for a part"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM part_bend_specs WHERE part_id = ?",
                (part_id,)
            )
            return cursor.rowcount

    def _row_to_bend_spec(self, row) -> PartBendSpec:
        """Convert database row to PartBendSpec"""
        return PartBendSpec(
            id=row["id"],
            part_id=row["part_id"],
            bend_id=row["bend_id"],
            target_angle=row["target_angle"],
            target_radius=row["target_radius"],
            tolerance_angle=row["tolerance_angle"],
            tolerance_radius=row["tolerance_radius"],
            bend_line_start=row["bend_line_start"],
            bend_line_end=row["bend_line_end"],
            notes=row["notes"],
        )

    # ==================== Live Scan Sessions ====================

    def create_session(
        self,
        operator_id: Optional[str] = None,
        workstation_id: Optional[str] = None,
    ) -> LiveScanSession:
        """Create a new live scan session"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO live_scan_sessions (
                    id, status, started_at, last_update_at,
                    operator_id, workstation_id
                ) VALUES (?, 'started', ?, ?, ?, ?)
            """, (session_id, now, now, operator_id, workstation_id))

        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[LiveScanSession]:
        """Get session by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_scan_sessions WHERE id = ?",
                (session_id,)
            ).fetchone()

            if row:
                return self._row_to_session(row)
            return None

    def update_session(
        self,
        session_id: str,
        **kwargs
    ) -> Optional[LiveScanSession]:
        """Update session fields"""
        allowed_fields = {
            'part_id', 'part_number', 'recognition_confidence',
            'status', 'scan_file_path', 'point_count', 'coverage_percent',
            'recognition_results', 'analysis_job_id', 'completed_at'
        }

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_session(session_id)

        # Handle JSON serialization for recognition_results
        if 'recognition_results' in updates and not isinstance(updates['recognition_results'], str):
            updates['recognition_results'] = json.dumps(updates['recognition_results'])

        updates['last_update_at'] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [session_id]

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE live_scan_sessions SET {set_clause} WHERE id = ?",
                values
            )

        return self.get_session(session_id)

    def list_active_sessions(self) -> List[LiveScanSession]:
        """List all non-completed sessions"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM live_scan_sessions
                WHERE status NOT IN ('completed', 'abandoned')
                ORDER BY started_at DESC
            """).fetchall()

            return [self._row_to_session(row) for row in rows]

    def _row_to_session(self, row) -> LiveScanSession:
        """Convert database row to LiveScanSession"""
        return LiveScanSession(
            id=row["id"],
            part_id=row["part_id"],
            part_number=row["part_number"],
            recognition_confidence=row["recognition_confidence"],
            status=row["status"],
            scan_file_path=row["scan_file_path"],
            point_count=row["point_count"] or 0,
            coverage_percent=row["coverage_percent"] or 0.0,
            recognition_results=row["recognition_results"],
            analysis_job_id=row["analysis_job_id"],
            started_at=row["started_at"],
            last_update_at=row["last_update_at"],
            completed_at=row["completed_at"],
            operator_id=row["operator_id"],
            workstation_id=row["workstation_id"],
        )


# ==================== CSV Import ====================

def import_parts_from_csv(
    catalog: PartCatalogManager,
    csv_path: str,
    part_number_col: str = "part_number",
    part_name_col: str = "part_name",
    customer_col: str = "customer",
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Import parts from CSV file.

    Args:
        catalog: PartCatalogManager instance
        csv_path: Path to CSV file
        part_number_col: Column name for part number
        part_name_col: Column name for part name
        customer_col: Column name for customer
        skip_existing: Skip parts that already exist

    Returns:
        Dict with counts: imported, skipped, errors
    """
    import csv

    results = {"imported": 0, "skipped": 0, "errors": 0}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            part_number = row.get(part_number_col, "").strip()
            if not part_number:
                results["errors"] += 1
                continue

            # Check if exists
            if skip_existing and catalog.get_part_by_number(part_number):
                results["skipped"] += 1
                continue

            try:
                catalog.create_part(
                    part_number=part_number,
                    part_name=row.get(part_name_col, "").strip() or None,
                    customer=row.get(customer_col, "").strip() or None,
                )
                results["imported"] += 1
            except Exception as e:
                logger.error(f"Error importing {part_number}: {e}")
                results["errors"] += 1

    return results


# ==================== Global Instance ====================

_catalog: Optional[PartCatalogManager] = None


def get_catalog() -> PartCatalogManager:
    """Get the global part catalog manager instance"""
    global _catalog
    if _catalog is None:
        _catalog = PartCatalogManager()
    return _catalog


def init_catalog(db_path: Optional[str] = None) -> PartCatalogManager:
    """Initialize and return the global part catalog manager"""
    global _catalog
    _catalog = PartCatalogManager(db_path)
    return _catalog
