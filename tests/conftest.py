"""
Shared pytest fixtures and configuration for Sherman QC tests.

This module provides:
- Common fixtures for database, QC results, batch jobs
- Path configuration for imports
- Custom markers for test categorization
- Mock servers for webhook testing
"""

import pytest
import sys
import os
import tempfile
import json
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock

# Add backend to path for all tests
backend_path = str(Path(__file__).parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


# ============================================================================
# Configuration - Custom Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_files: marks tests that need CAD/scan files"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async"
    )


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing.

    Yields a configured SQLiteDatabaseManager instance and
    automatically cleans up the database file after the test.
    """
    from database import SQLiteDatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = SQLiteDatabaseManager(db_path)
    yield db

    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass  # File may already be deleted


@pytest.fixture
def temp_db_with_jobs(temp_db):
    """Database fixture pre-populated with sample jobs.

    Creates 5 jobs with varying statuses for testing list/filter operations.
    """
    jobs = []
    for i in range(5):
        job = temp_db.create_job(
            job_id=f"job_{i:03d}",
            part_id=f"PART-{i:03d}",
            part_name=f"Test Part {i}",
            material="Al-5053-H32",
            tolerance=0.1,
            reference_path=f"/path/ref_{i}.stl",
            scan_path=f"/path/scan_{i}.ply"
        )
        jobs.append(job)

    # Update some statuses for filtering tests
    temp_db.update_job_progress("job_001", "running", 50, "align", "Aligning...")
    temp_db.update_job_result("job_002", {"overall_result": "PASS", "quality_score": 95.0}, "/r.json", "/r.pdf")
    temp_db.update_job_result("job_003", {"overall_result": "FAIL", "quality_score": 65.0}, "/r2.json", "/r2.pdf")
    temp_db.update_job_error("job_004", "Test error")

    yield temp_db, jobs


# ============================================================================
# QC Result Fixtures
# ============================================================================

@pytest.fixture
def sample_qc_result() -> Dict[str, Any]:
    """Sample QC result data for testing.

    Returns a complete QC result dictionary with all expected fields.
    """
    return {
        "part_id": "PART-TEST-001",
        "part_name": "Test Bracket Assembly",
        "material": "Al-5053-H32",
        "tolerance": 0.1,
        "timestamp": datetime.now().isoformat(),
        "overall_result": "PASS",
        "quality_score": 92.5,
        "confidence": 85.0,
        "statistics": {
            "total_points": 50000,
            "points_in_tolerance": 47500,
            "points_out_of_tolerance": 2500,
            "pass_rate": 95.0,
            "mean_deviation_mm": 0.02,
            "max_deviation_mm": 0.15,
            "min_deviation_mm": -0.12,
            "std_deviation_mm": 0.04
        },
        "alignment": {
            "fitness": 0.98,
            "rmse": 0.015
        },
        "regions": [
            {
                "name": "Top Surface",
                "point_count": 12000,
                "mean_deviation": 0.01,
                "max_deviation": 0.08,
                "pass_rate": 0.98,
                "status": "OK"
            },
            {
                "name": "Front Edge",
                "point_count": 8000,
                "mean_deviation": 0.05,
                "max_deviation": 0.12,
                "pass_rate": 0.92,
                "status": "ATTENTION"
            }
        ],
        "root_causes": [
            {
                "issue": "Minor edge deviation",
                "likely_cause": "Tool wear",
                "confidence": 0.75,
                "recommendation": "Inspect cutting tool",
                "priority": "medium"
            }
        ],
        "ai_summary": "Part passes overall tolerance requirements with minor edge deviations.",
        "ai_model_used": "claude-3-5-sonnet"
    }


@pytest.fixture
def sample_qc_result_fail() -> Dict[str, Any]:
    """Sample failing QC result for testing failure scenarios."""
    return {
        "part_id": "PART-FAIL-001",
        "part_name": "Test Failed Part",
        "material": "Steel-4140",
        "tolerance": 0.05,
        "timestamp": datetime.now().isoformat(),
        "overall_result": "FAIL",
        "quality_score": 62.0,
        "confidence": 90.0,
        "statistics": {
            "total_points": 45000,
            "points_in_tolerance": 27000,
            "points_out_of_tolerance": 18000,
            "pass_rate": 60.0,
            "mean_deviation_mm": 0.08,
            "max_deviation_mm": 0.25,
            "min_deviation_mm": -0.18,
            "std_deviation_mm": 0.12
        },
        "root_causes": [
            {
                "issue": "Significant surface warping",
                "likely_cause": "Thermal distortion during machining",
                "confidence": 0.85,
                "recommendation": "Review coolant flow and machining parameters",
                "priority": "high"
            }
        ]
    }


# ============================================================================
# Batch Processing Fixtures
# ============================================================================

@pytest.fixture
def sample_batch() -> Dict[str, Any]:
    """Sample batch job data for testing."""
    return {
        "batch_id": "batch_test_001",
        "name": "Morning Production Run",
        "material": "Al-5053-H32",
        "tolerance": 0.1,
        "total_parts": 10,
        "completed_parts": 8,
        "failed_parts": 1,
        "status": "running",
        "progress": 90.0,
        "current_part": "PART-009",
        "created_at": datetime.now().isoformat(),
        "parts": [
            {"part_index": i, "job_id": f"job_{i:03d}", "part_id": f"PART-{i:03d}",
             "part_name": f"Part {i}", "status": "completed" if i < 8 else "pending",
             "result": "PASS" if i < 7 else ("FAIL" if i == 7 else None),
             "quality_score": 90.0 + i if i < 8 else None}
            for i in range(10)
        ]
    }


@pytest.fixture
def sample_batch_summary() -> Dict[str, Any]:
    """Sample batch summary statistics."""
    return {
        "batch_id": "batch_test_001",
        "name": "Morning Production Run",
        "status": "completed",
        "total_parts": 10,
        "passed": 9,
        "failed": 1,
        "pass_rate": 90.0,
        "quality_scores": {
            "min": 85.0,
            "max": 98.0,
            "average": 92.5,
            "std_dev": 3.2
        },
        "duration_seconds": 1200,
        "created_at": "2026-01-12T08:00:00",
        "completed_at": "2026-01-12T08:20:00"
    }


# ============================================================================
# Webhook Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_webhook_server():
    """Mock HTTP server for webhook testing.

    Provides a mock that tracks received requests and can simulate
    various response scenarios.
    """
    class MockWebhookServer:
        def __init__(self):
            self.requests = []
            self.response_status = 200
            self.response_body = "OK"
            self.should_fail = False
            self.fail_count = 0
            self.current_fails = 0

        def configure_failure(self, count: int = 1):
            """Configure server to fail for specified number of requests."""
            self.should_fail = True
            self.fail_count = count
            self.current_fails = 0

        def reset(self):
            """Reset server state."""
            self.requests = []
            self.should_fail = False
            self.fail_count = 0
            self.current_fails = 0

        async def handle_request(self, url: str, payload: Dict, headers: Dict):
            """Simulate handling a webhook request."""
            self.requests.append({
                "url": url,
                "payload": payload,
                "headers": headers,
                "timestamp": datetime.now().isoformat()
            })

            if self.should_fail and self.current_fails < self.fail_count:
                self.current_fails += 1
                raise ConnectionError("Simulated connection failure")

            return self.response_status, self.response_body

    return MockWebhookServer()


@pytest.fixture
def webhook_config():
    """Standard webhook configuration for testing."""
    from webhooks import WebhookConfig, WebhookEvent

    return WebhookConfig(
        id="wh-test-001",
        url="https://test.example.com/webhook",
        name="Test Webhook",
        secret="test-secret-key",
        events={WebhookEvent.JOB_COMPLETED, WebhookEvent.QC_FAIL},
        active=True,
        retry_count=3,
        retry_delay=0.01  # Fast for testing
    )


# ============================================================================
# SPC/Trend Analysis Fixtures
# ============================================================================

@pytest.fixture
def sample_measurement_series():
    """Sample time series data for SPC testing."""
    import numpy as np

    np.random.seed(42)
    n_points = 50

    # Generate realistic measurement data with slight trend
    base_values = np.linspace(0.01, 0.03, n_points)
    noise = np.random.normal(0, 0.005, n_points)
    values = base_values + noise

    timestamps = [
        datetime(2026, 1, 12, 8, 0, 0) +
        __import__('datetime').timedelta(minutes=i * 5)
        for i in range(n_points)
    ]

    return [
        {"timestamp": ts.isoformat(), "value": float(v), "job_id": f"job_{i:03d}"}
        for i, (ts, v) in enumerate(zip(timestamps, values))
    ]


@pytest.fixture
def sample_spc_data():
    """Sample SPC analysis data."""
    return {
        "part_id": "PART-001",
        "characteristic": "diameter",
        "specification": {
            "nominal": 25.0,
            "upper_limit": 25.1,
            "lower_limit": 24.9
        },
        "measurements": [24.98, 25.01, 24.99, 25.02, 24.97, 25.00, 25.01, 24.98],
        "statistics": {
            "mean": 24.995,
            "std_dev": 0.017,
            "cp": 1.96,
            "cpk": 1.85
        }
    }


# ============================================================================
# File/Path Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Temporary directory for test output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_stl_content():
    """Mock binary STL file content for testing."""
    # Minimal valid binary STL header (80 bytes) + triangle count (4 bytes)
    header = b'\x00' * 80
    triangle_count = b'\x00\x00\x00\x00'  # 0 triangles
    return header + triangle_count


# ============================================================================
# Async Event Loop Fixture
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Parametrized Test Data
# ============================================================================

# Common tolerance values for parametrized tests
TOLERANCE_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]

# Common material types for parametrized tests
MATERIAL_TYPES = [
    "Al-5053-H32",
    "Al-6061-T6",
    "Steel-4140",
    "Ti-6Al-4V",
    "Inconel-718"
]

# Quality score thresholds
QUALITY_THRESHOLDS = {
    "excellent": 95.0,
    "good": 85.0,
    "acceptable": 75.0,
    "marginal": 65.0,
    "fail": 0.0
}
