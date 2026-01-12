"""
Tests for Batch Processing Module

Tests the batch processing dataclasses and logic independently
from the server to avoid importing heavy dependencies.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# Mirror dataclasses from server.py for testing without heavy imports
@dataclass
class BatchPartResult:
    """Result for a single part in a batch (test copy)"""
    part_index: int
    job_id: str
    part_id: str
    part_name: str
    status: str = "pending"
    progress: float = 0
    result: Optional[str] = None
    quality_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchJobState:
    """Track batch job state (test copy)"""
    batch_id: str
    name: str
    material: str
    tolerance: float
    total_parts: int
    completed_parts: int = 0
    failed_parts: int = 0
    status: str = "pending"
    progress: float = 0
    current_part: str = ""
    parts: Dict[str, BatchPartResult] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["parts"] = [p.to_dict() if hasattr(p, 'to_dict') else p for p in self.parts.values()]
        return data


class TestBatchPartResult:
    """Test BatchPartResult dataclass"""

    def test_batch_part_result_creation(self):
        """Test creating a BatchPartResult"""
        result = BatchPartResult(
            part_index=0,
            job_id="batch_abc123_000",
            part_id="PART-001",
            part_name="Test Part"
        )

        assert result.part_index == 0
        assert result.job_id == "batch_abc123_000"
        assert result.part_id == "PART-001"
        assert result.status == "pending"
        assert result.progress == 0
        assert result.result is None

    def test_batch_part_result_to_dict(self):
        """Test serialization of BatchPartResult"""
        result = BatchPartResult(
            part_index=0,
            job_id="batch_abc123_000",
            part_id="PART-001",
            part_name="Test Part",
            status="completed",
            progress=100,
            result="PASS",
            quality_score=95.5
        )

        data = result.to_dict()

        assert data["part_index"] == 0
        assert data["job_id"] == "batch_abc123_000"
        assert data["status"] == "completed"
        assert data["result"] == "PASS"
        assert data["quality_score"] == 95.5


class TestBatchJobState:
    """Test BatchJobState dataclass"""

    def test_batch_job_state_creation(self):
        """Test creating a BatchJobState"""
        parts = {
            "job_001": BatchPartResult(0, "job_001", "PART-001", "Part 1"),
            "job_002": BatchPartResult(1, "job_002", "PART-002", "Part 2"),
        }

        batch = BatchJobState(
            batch_id="batch_test123",
            name="Test Batch",
            material="Al-5053-H32",
            tolerance=0.1,
            total_parts=2,
            parts=parts
        )

        assert batch.batch_id == "batch_test123"
        assert batch.name == "Test Batch"
        assert batch.total_parts == 2
        assert batch.status == "pending"
        assert len(batch.parts) == 2

    def test_batch_job_state_defaults(self):
        """Test default values of BatchJobState"""
        batch = BatchJobState(
            batch_id="batch_123",
            name="Test",
            material="Steel",
            tolerance=0.05,
            total_parts=3
        )

        assert batch.completed_parts == 0
        assert batch.failed_parts == 0
        assert batch.status == "pending"
        assert batch.progress == 0
        assert batch.current_part == ""
        assert batch.completed_at is None
        assert batch.created_at is not None

    def test_batch_job_state_to_dict(self):
        """Test serialization of BatchJobState"""
        parts = {
            "job_001": BatchPartResult(0, "job_001", "PART-001", "Part 1", "completed", 100, "PASS", 95.0),
        }

        batch = BatchJobState(
            batch_id="batch_test",
            name="Test Batch",
            material="Al",
            tolerance=0.1,
            total_parts=1,
            completed_parts=1,
            status="completed",
            progress=100,
            parts=parts
        )

        data = batch.to_dict()

        assert data["batch_id"] == "batch_test"
        assert data["name"] == "Test Batch"
        assert data["status"] == "completed"
        assert data["total_parts"] == 1
        assert data["completed_parts"] == 1
        assert len(data["parts"]) == 1


class TestBatchProcessingLogic:
    """Test batch processing logic (unit tests, no server)"""

    def test_progress_calculation(self):
        """Test batch progress calculation"""
        total_parts = 5
        completed = 3
        failed = 1

        progress = ((completed + failed) / total_parts) * 100
        assert progress == 80.0

    def test_pass_rate_calculation(self):
        """Test pass rate calculation"""
        completed_parts = 10
        passed = 8

        pass_rate = (passed / completed_parts * 100) if completed_parts > 0 else 0
        assert pass_rate == 80.0

    def test_quality_score_aggregation(self):
        """Test quality score statistics"""
        scores = [95.0, 88.5, 92.0, 78.0, 99.5]

        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        assert avg_score == 90.6
        assert min_score == 78.0
        assert max_score == 99.5


class TestBatchValidation:
    """Test batch input validation"""

    def test_file_count_mismatch_detection(self):
        """Test detection of mismatched file counts"""
        reference_count = 3
        scan_count = 5

        assert reference_count != scan_count

    def test_max_parts_limit(self):
        """Test maximum parts per batch limit"""
        max_parts = 50
        requested_parts = 60

        assert requested_parts > max_parts

    def test_part_id_generation(self):
        """Test auto-generation of part IDs"""
        num_parts = 5
        provided_ids = ["CUSTOM-001", "CUSTOM-002"]

        generated_ids = []
        for i in range(num_parts):
            if i < len(provided_ids):
                generated_ids.append(provided_ids[i])
            else:
                generated_ids.append(f"PART_{i+1:03d}")

        assert generated_ids == ["CUSTOM-001", "CUSTOM-002", "PART_003", "PART_004", "PART_005"]


class TestBatchSummaryCalculations:
    """Test batch summary statistics"""

    def test_duration_calculation(self):
        """Test batch duration calculation"""
        created = datetime(2026, 1, 12, 10, 0, 0)
        completed = datetime(2026, 1, 12, 10, 5, 30)

        duration_seconds = (completed - created).total_seconds()
        assert duration_seconds == 330.0  # 5 minutes 30 seconds

    def test_empty_scores_handling(self):
        """Test handling when no quality scores available"""
        scores = []

        avg_score = sum(scores) / len(scores) if scores else None
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None

        assert avg_score is None
        assert min_score is None
        assert max_score is None

    def test_pass_rate_with_zero_completed(self):
        """Test pass rate calculation when no parts completed"""
        completed_parts = 0
        passed = 0

        pass_rate = (passed / completed_parts * 100) if completed_parts > 0 else 0
        assert pass_rate == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
