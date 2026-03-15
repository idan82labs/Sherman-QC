"""
Database Module - Multi-Backend Persistence for QC Jobs

Supports both SQLite (development) and PostgreSQL (production).
Automatically detects backend based on DATABASE_URL environment variable.

Usage:
    - SQLite (default): No DATABASE_URL set, uses local file
    - PostgreSQL: Set DATABASE_URL=postgresql://user:pass@host:port/dbname
"""

import os
import json
import sqlite3
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "")
DATABASE_PATH = Path(__file__).parent.parent / "data" / "qc_jobs.db"


def _json_storage_safe(value: Any) -> Any:
    """Recursively sanitize values for strict JSON storage."""
    if isinstance(value, dict):
        return {k: _json_storage_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_storage_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@dataclass
class JobRecord:
    """Database record for an analysis job"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float
    stage: str
    message: str
    error: Optional[str]

    # Part information
    part_id: str
    part_name: str
    material: str
    tolerance: float

    # File paths
    reference_path: str
    scan_path: str
    drawing_path: Optional[str]
    report_path: Optional[str]
    pdf_path: Optional[str]

    # Result (JSON string)
    result_json: Optional[str]

    # Timestamps
    created_at: str
    updated_at: str
    completed_at: Optional[str]

    def to_dict(self) -> Dict:
        """Convert to API-friendly dict"""
        data = asdict(self)
        if self.result_json:
            try:
                data["result"] = json.loads(self.result_json)
            except json.JSONDecodeError:
                data["result"] = None
        del data["result_json"]
        return data


class BaseDatabaseManager(ABC):
    """Abstract base class for database managers"""

    @abstractmethod
    def get_connection(self):
        pass

    @abstractmethod
    def _init_db(self):
        pass

    @abstractmethod
    def create_job(self, job_id: str, part_id: str, part_name: str,
                   material: str, tolerance: float, reference_path: str,
                   scan_path: str, drawing_path: Optional[str] = None) -> JobRecord:
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[JobRecord]:
        pass


class SQLiteDatabaseManager:
    """SQLite database manager for local development"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        # Set busy timeout for concurrent access
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress REAL DEFAULT 0,
                    stage TEXT DEFAULT '',
                    message TEXT DEFAULT '',
                    error TEXT,

                    part_id TEXT NOT NULL,
                    part_name TEXT DEFAULT '',
                    material TEXT DEFAULT '',
                    tolerance REAL DEFAULT 0.1,

                    reference_path TEXT NOT NULL,
                    scan_path TEXT NOT NULL,
                    drawing_path TEXT,
                    report_path TEXT,
                    pdf_path TEXT,

                    result_json TEXT,

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)

            # Single column indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_part_id ON jobs(part_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")

            # Composite indexes for common query patterns
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_part_id_created ON jobs(part_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)")

            logger.info(f"Database initialized at {self.db_path}")

    def _row_to_job(self, row) -> JobRecord:
        """Convert database row to JobRecord"""
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            progress=row["progress"],
            stage=row["stage"],
            message=row["message"],
            error=row["error"],
            part_id=row["part_id"],
            part_name=row["part_name"],
            material=row["material"],
            tolerance=row["tolerance"],
            reference_path=row["reference_path"],
            scan_path=row["scan_path"],
            drawing_path=row["drawing_path"],
            report_path=row["report_path"],
            pdf_path=row["pdf_path"],
            result_json=row["result_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"]
        )

    def create_job(
        self,
        job_id: str,
        part_id: str,
        part_name: str,
        material: str,
        tolerance: float,
        reference_path: str,
        scan_path: str,
        drawing_path: Optional[str] = None
    ) -> JobRecord:
        """Create a new job record"""
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO jobs (
                    job_id, status, progress, stage, message,
                    part_id, part_name, material, tolerance,
                    reference_path, scan_path, drawing_path,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, "pending", 0, "", "Job created",
                part_id, part_name, material, tolerance,
                reference_path, scan_path, drawing_path,
                now, now
            ))

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Get a job by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,)
            ).fetchone()

            if row:
                return self._row_to_job(row)
            return None

    def update_job_progress(
        self,
        job_id: str,
        status: str,
        progress: float,
        stage: str,
        message: str
    ):
        """Update job progress"""
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            conn.execute("""
                UPDATE jobs SET
                    status = ?,
                    progress = ?,
                    stage = ?,
                    message = ?,
                    updated_at = ?
                WHERE job_id = ?
            """, (status, progress, stage, message, now, job_id))

    def update_job_result(
        self,
        job_id: str,
        result: Dict[str, Any],
        report_path: str,
        pdf_path: str
    ):
        """Update job with completed result"""
        now = datetime.now().isoformat()
        safe_result = _json_storage_safe(result)

        with self.get_connection() as conn:
            conn.execute("""
                UPDATE jobs SET
                    status = 'completed',
                    progress = 100,
                    message = 'Analysis complete!',
                    result_json = ?,
                    report_path = ?,
                    pdf_path = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE job_id = ?
            """, (json.dumps(safe_result, allow_nan=False), report_path, pdf_path, now, now, job_id))

    def update_job_error(self, job_id: str, error: str):
        """Mark job as failed with error"""
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            conn.execute("""
                UPDATE jobs SET
                    status = 'failed',
                    message = 'Analysis failed',
                    error = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE job_id = ?
            """, (error, now, now, job_id))

    def list_jobs(
        self,
        status: Optional[str] = None,
        part_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_result: bool = False
    ) -> List[JobRecord]:
        """List jobs with optional filters

        Args:
            status: Filter by job status
            part_id: Filter by part ID
            limit: Maximum number of results (capped at 1000)
            offset: Number of records to skip
            include_result: If False, excludes large result_json field for performance
        """
        # Cap limit to prevent excessive queries
        limit = min(limit, 1000)

        # Select only needed columns for list view (exclude large result_json unless needed)
        if include_result:
            columns = "*"
        else:
            columns = """job_id, status, progress, stage, message, error,
                        part_id, part_name, material, tolerance,
                        reference_path, scan_path, drawing_path, report_path, pdf_path,
                        created_at, updated_at, completed_at"""

        query = f"SELECT {columns} FROM jobs WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if part_id:
            query += " AND part_id = ?"
            params.append(part_id)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            jobs = []
            for row in rows:
                # Handle missing result_json when not included
                row_dict = dict(row)
                if not include_result and 'result_json' not in row_dict:
                    row_dict['result_json'] = None
                jobs.append(self._row_to_job_from_dict(row_dict))
            return jobs

    def _row_to_job_from_dict(self, row_dict: Dict) -> JobRecord:
        """Convert dictionary to JobRecord"""
        return JobRecord(
            job_id=row_dict["job_id"],
            status=row_dict["status"],
            progress=row_dict["progress"],
            stage=row_dict["stage"],
            message=row_dict["message"],
            error=row_dict["error"],
            part_id=row_dict["part_id"],
            part_name=row_dict["part_name"],
            material=row_dict["material"],
            tolerance=row_dict["tolerance"],
            reference_path=row_dict["reference_path"],
            scan_path=row_dict["scan_path"],
            drawing_path=row_dict.get("drawing_path"),
            report_path=row_dict.get("report_path"),
            pdf_path=row_dict.get("pdf_path"),
            result_json=row_dict.get("result_json"),
            created_at=row_dict["created_at"],
            updated_at=row_dict["updated_at"],
            completed_at=row_dict.get("completed_at")
        )

    def count_jobs(
        self,
        status: Optional[str] = None,
        part_id: Optional[str] = None
    ) -> int:
        """Count jobs with optional filters"""
        query = "SELECT COUNT(*) FROM jobs WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if part_id:
            query += " AND part_id = ?"
            params.append(part_id)

        with self.get_connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics"""
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'completed'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'failed'"
            ).fetchone()[0]
            running = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'running'"
            ).fetchone()[0]

            return {
                "total_jobs": total,
                "completed": completed,
                "failed": failed,
                "running": running,
                "pending": total - completed - failed - running
            }


class PostgreSQLDatabaseManager:
    """PostgreSQL database manager for production"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool = None
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install it with: pip install psycopg2-binary"
            )

        conn = psycopg2.connect(self.database_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize PostgreSQL schema"""
        try:
            import psycopg2
        except ImportError:
            logger.warning("psycopg2 not installed, PostgreSQL support disabled")
            return

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id VARCHAR(255) PRIMARY KEY,
                        status VARCHAR(50) NOT NULL DEFAULT 'pending',
                        progress FLOAT DEFAULT 0,
                        stage VARCHAR(100) DEFAULT '',
                        message TEXT DEFAULT '',
                        error TEXT,

                        part_id VARCHAR(255) NOT NULL,
                        part_name VARCHAR(255) DEFAULT '',
                        material VARCHAR(100) DEFAULT '',
                        tolerance FLOAT DEFAULT 0.1,

                        reference_path TEXT NOT NULL,
                        scan_path TEXT NOT NULL,
                        drawing_path TEXT,
                        report_path TEXT,
                        pdf_path TEXT,

                        result_json JSONB,

                        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                        completed_at TIMESTAMP WITH TIME ZONE
                    )
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_part_id ON jobs(part_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)
                """)

                # Create GIN index for JSONB result
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_result ON jobs USING GIN(result_json)
                """)

            logger.info("PostgreSQL database initialized")

    def _row_to_job(self, row: dict) -> JobRecord:
        """Convert database row to JobRecord"""
        result_json = row.get("result_json")
        if result_json and not isinstance(result_json, str):
            result_json = json.dumps(result_json)

        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            progress=row["progress"],
            stage=row["stage"],
            message=row["message"],
            error=row["error"],
            part_id=row["part_id"],
            part_name=row["part_name"],
            material=row["material"],
            tolerance=row["tolerance"],
            reference_path=row["reference_path"],
            scan_path=row["scan_path"],
            drawing_path=row["drawing_path"],
            report_path=row["report_path"],
            pdf_path=row["pdf_path"],
            result_json=result_json,
            created_at=str(row["created_at"]) if row["created_at"] else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] else None,
            completed_at=str(row["completed_at"]) if row["completed_at"] else None
        )

    def create_job(
        self,
        job_id: str,
        part_id: str,
        part_name: str,
        material: str,
        tolerance: float,
        reference_path: str,
        scan_path: str,
        drawing_path: Optional[str] = None
    ) -> JobRecord:
        """Create a new job record"""
        from psycopg2.extras import RealDictCursor

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO jobs (
                        job_id, status, progress, stage, message,
                        part_id, part_name, material, tolerance,
                        reference_path, scan_path, drawing_path
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                """, (
                    job_id, "pending", 0, "", "Job created",
                    part_id, part_name, material, tolerance,
                    reference_path, scan_path, drawing_path
                ))
                row = cur.fetchone()
                return self._row_to_job(dict(row))

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Get a job by ID"""
        from psycopg2.extras import RealDictCursor

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
                row = cur.fetchone()
                if row:
                    return self._row_to_job(dict(row))
                return None

    def update_job_progress(
        self,
        job_id: str,
        status: str,
        progress: float,
        stage: str,
        message: str
    ):
        """Update job progress"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE jobs SET
                        status = %s,
                        progress = %s,
                        stage = %s,
                        message = %s,
                        updated_at = NOW()
                    WHERE job_id = %s
                """, (status, progress, stage, message, job_id))

    def update_job_result(
        self,
        job_id: str,
        result: Dict[str, Any],
        report_path: str,
        pdf_path: str
    ):
        """Update job with completed result"""
        from psycopg2.extras import Json
        safe_result = _json_storage_safe(result)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE jobs SET
                        status = 'completed',
                        progress = 100,
                        message = 'Analysis complete!',
                        result_json = %s,
                        report_path = %s,
                        pdf_path = %s,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE job_id = %s
                """, (Json(safe_result), report_path, pdf_path, job_id))

    def update_job_error(self, job_id: str, error: str):
        """Mark job as failed with error"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE jobs SET
                        status = 'failed',
                        message = 'Analysis failed',
                        error = %s,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE job_id = %s
                """, (error, job_id))

    def list_jobs(
        self,
        status: Optional[str] = None,
        part_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[JobRecord]:
        """List jobs with optional filters"""
        from psycopg2.extras import RealDictCursor

        query = "SELECT * FROM jobs WHERE 1=1"
        params = []

        if status:
            query += " AND status = %s"
            params.append(status)

        if part_id:
            query += " AND part_id = %s"
            params.append(part_id)

        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                return [self._row_to_job(dict(row)) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
                return cur.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM jobs")
                total = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM jobs WHERE status = 'completed'")
                completed = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM jobs WHERE status = 'failed'")
                failed = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
                running = cur.fetchone()[0]

                return {
                    "total_jobs": total,
                    "completed": completed,
                    "failed": failed,
                    "running": running,
                    "pending": total - completed - failed - running
                }


# Type alias for database manager
DatabaseManager = Union[SQLiteDatabaseManager, PostgreSQLDatabaseManager]


def create_database_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """
    Factory function to create appropriate database manager.

    Args:
        database_url: Optional database URL. If not provided, uses DATABASE_URL env var.
                     If no URL is configured, defaults to SQLite.

    Returns:
        Appropriate database manager instance
    """
    url = database_url or DATABASE_URL

    if url and url.startswith("postgresql"):
        logger.info("Using PostgreSQL database")
        return PostgreSQLDatabaseManager(url)
    else:
        logger.info("Using SQLite database")
        return SQLiteDatabaseManager()


# Global database instance
_db: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db
    if _db is None:
        _db = create_database_manager()
    return _db


def init_db(database_url: Optional[str] = None) -> DatabaseManager:
    """Initialize and return the global database manager"""
    global _db
    _db = create_database_manager(database_url)
    return _db
