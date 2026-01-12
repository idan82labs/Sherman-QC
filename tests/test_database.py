"""
Tests for Database Module - SQLite Persistence
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from database import SQLiteDatabaseManager, JobRecord, init_db, get_db


class TestDatabaseManager:
    """Test DatabaseManager class"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = SQLiteDatabaseManager(db_path)
        yield db

        # Cleanup
        os.unlink(db_path)

    def test_init_creates_tables(self, temp_db):
        """Test database initialization creates required tables"""
        with temp_db.get_connection() as conn:
            # Check jobs table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
            ).fetchone()
            assert result is not None
            assert result[0] == "jobs"

    def test_create_job(self, temp_db):
        """Test creating a new job"""
        job = temp_db.create_job(
            job_id="test123",
            part_id="PART-001",
            part_name="Test Part",
            material="Al-5053-H32",
            tolerance=0.1,
            reference_path="/path/to/ref.stl",
            scan_path="/path/to/scan.ply"
        )

        assert job is not None
        assert job.job_id == "test123"
        assert job.part_id == "PART-001"
        assert job.status == "pending"
        assert job.progress == 0

    def test_get_job(self, temp_db):
        """Test retrieving a job by ID"""
        # Create job
        temp_db.create_job(
            job_id="test456",
            part_id="PART-002",
            part_name="Another Part",
            material="Steel",
            tolerance=0.05,
            reference_path="/path/ref.stl",
            scan_path="/path/scan.ply"
        )

        # Retrieve job
        job = temp_db.get_job("test456")

        assert job is not None
        assert job.job_id == "test456"
        assert job.part_id == "PART-002"
        assert job.material == "Steel"

    def test_get_nonexistent_job(self, temp_db):
        """Test retrieving a non-existent job returns None"""
        job = temp_db.get_job("nonexistent")
        assert job is None

    def test_update_job_progress(self, temp_db):
        """Test updating job progress"""
        temp_db.create_job(
            job_id="progress_test",
            part_id="PART-003",
            part_name="Progress Part",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )

        # Update progress
        temp_db.update_job_progress(
            job_id="progress_test",
            status="running",
            progress=50.0,
            stage="align",
            message="Aligning meshes..."
        )

        # Verify update
        job = temp_db.get_job("progress_test")
        assert job.status == "running"
        assert job.progress == 50.0
        assert job.stage == "align"
        assert job.message == "Aligning meshes..."

    def test_update_job_result(self, temp_db):
        """Test updating job with completed result"""
        temp_db.create_job(
            job_id="result_test",
            part_id="PART-004",
            part_name="Result Part",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )

        # Update with result
        result = {"overall_result": "PASS", "quality_score": 95.0}
        temp_db.update_job_result(
            job_id="result_test",
            result=result,
            report_path="/output/report.json",
            pdf_path="/output/report.pdf"
        )

        # Verify update
        job = temp_db.get_job("result_test")
        assert job.status == "completed"
        assert job.progress == 100
        assert job.report_path == "/output/report.json"
        assert job.pdf_path == "/output/report.pdf"
        assert job.completed_at is not None

    def test_update_job_error(self, temp_db):
        """Test marking job as failed"""
        temp_db.create_job(
            job_id="error_test",
            part_id="PART-005",
            part_name="Error Part",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )

        # Mark as failed
        temp_db.update_job_error("error_test", "File not found")

        # Verify update
        job = temp_db.get_job("error_test")
        assert job.status == "failed"
        assert job.error == "File not found"

    def test_list_jobs(self, temp_db):
        """Test listing jobs"""
        # Create multiple jobs
        for i in range(5):
            temp_db.create_job(
                job_id=f"list_test_{i}",
                part_id=f"PART-{i:03d}",
                part_name=f"Part {i}",
                material="Al",
                tolerance=0.1,
                reference_path="/ref.stl",
                scan_path="/scan.ply"
            )

        # List all
        jobs = temp_db.list_jobs()
        assert len(jobs) == 5

        # List with limit
        jobs = temp_db.list_jobs(limit=3)
        assert len(jobs) == 3

    def test_list_jobs_filter_status(self, temp_db):
        """Test filtering jobs by status"""
        # Create jobs with different statuses
        temp_db.create_job(
            job_id="status_test_1",
            part_id="P1",
            part_name="Part 1",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )
        temp_db.update_job_progress("status_test_1", "running", 50, "align", "Aligning...")

        temp_db.create_job(
            job_id="status_test_2",
            part_id="P2",
            part_name="Part 2",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )
        temp_db.update_job_result("status_test_2", {}, "/report.json", "/report.pdf")

        # Filter by running
        running_jobs = temp_db.list_jobs(status="running")
        assert len(running_jobs) == 1
        assert running_jobs[0].job_id == "status_test_1"

        # Filter by completed
        completed_jobs = temp_db.list_jobs(status="completed")
        assert len(completed_jobs) == 1
        assert completed_jobs[0].job_id == "status_test_2"

    def test_delete_job(self, temp_db):
        """Test deleting a job"""
        temp_db.create_job(
            job_id="delete_test",
            part_id="PART-DELETE",
            part_name="Delete Part",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )

        # Verify exists
        job = temp_db.get_job("delete_test")
        assert job is not None

        # Delete
        result = temp_db.delete_job("delete_test")
        assert result is True

        # Verify deleted
        job = temp_db.get_job("delete_test")
        assert job is None

    def test_get_stats(self, temp_db):
        """Test getting job statistics"""
        # Create jobs with different statuses
        temp_db.create_job(
            job_id="stats_1", part_id="P1", part_name="Part 1",
            material="Al", tolerance=0.1,
            reference_path="/ref.stl", scan_path="/scan.ply"
        )

        temp_db.create_job(
            job_id="stats_2", part_id="P2", part_name="Part 2",
            material="Al", tolerance=0.1,
            reference_path="/ref.stl", scan_path="/scan.ply"
        )
        temp_db.update_job_result("stats_2", {}, "/r.json", "/r.pdf")

        temp_db.create_job(
            job_id="stats_3", part_id="P3", part_name="Part 3",
            material="Al", tolerance=0.1,
            reference_path="/ref.stl", scan_path="/scan.ply"
        )
        temp_db.update_job_error("stats_3", "Test error")

        # Get stats
        stats = temp_db.get_stats()

        assert stats["total_jobs"] == 3
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["pending"] == 1


class TestJobRecord:
    """Test JobRecord dataclass"""

    def test_to_dict(self):
        """Test JobRecord serialization"""
        record = JobRecord(
            job_id="test123",
            status="completed",
            progress=100,
            stage="complete",
            message="Done",
            error=None,
            part_id="PART-001",
            part_name="Test Part",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply",
            drawing_path=None,
            report_path="/report.json",
            pdf_path="/report.pdf",
            result_json='{"overall_result": "PASS"}',
            created_at="2026-01-12T00:00:00",
            updated_at="2026-01-12T00:01:00",
            completed_at="2026-01-12T00:01:00"
        )

        data = record.to_dict()

        assert data["job_id"] == "test123"
        assert data["status"] == "completed"
        assert data["result"]["overall_result"] == "PASS"
        assert "result_json" not in data  # Should be converted to result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
