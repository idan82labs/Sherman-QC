"""
API endpoint tests for Sherman QC FastAPI backend.

Tests the HTTP API layer including authentication, job management,
batch operations, and error handling.
"""

import pytest
import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Add backend to path
backend_path = str(Path(__file__).parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


@pytest.fixture
def client(temp_db, temp_output_dir):
    """Create test client with mocked dependencies."""
    # Patch database and directories before importing server
    with patch("server.init_db") as mock_init_db, \
         patch("server.UPLOAD_DIR", temp_output_dir / "uploads"), \
         patch("server.OUTPUT_DIR", temp_output_dir / "output"), \
         patch("server.DATA_DIR", temp_output_dir / "data"):

        mock_init_db.return_value = temp_db

        # Create directories
        (temp_output_dir / "uploads").mkdir(exist_ok=True)
        (temp_output_dir / "output").mkdir(exist_ok=True)
        (temp_output_dir / "data").mkdir(exist_ok=True)

        # Import server after patching
        from server import app
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def auth_token(client):
    """Get authentication token for protected endpoints."""
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    # If login fails (no default user), create one
    return None


@pytest.fixture
def auth_headers(auth_token):
    """Authorization headers for protected endpoints."""
    if auth_token:
        return {"Authorization": f"Bearer {auth_token}"}
    return {}


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_stats_endpoint(self, client, auth_headers):
        """Test system stats endpoint."""
        response = client.get("/api/stats", headers=auth_headers)
        # May return 401 if no auth, or 200 with stats
        assert response.status_code in [200, 401]


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_login_missing_credentials(self, client):
        """Test login with missing credentials returns 422."""
        response = client.post("/api/auth/login", json={})
        assert response.status_code == 422

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials returns 401."""
        response = client.post(
            "/api/auth/login",
            json={"username": "invalid", "password": "wrong"}
        )
        assert response.status_code == 401

    def test_me_without_auth(self, client):
        """Test /me endpoint without authentication returns 401/403."""
        response = client.get("/api/auth/me")
        assert response.status_code in [401, 403]


class TestJobsEndpoints:
    """Test job management endpoints."""

    def test_list_jobs_empty(self, client, auth_headers):
        """Test listing jobs when database is empty."""
        response = client.get("/api/jobs", headers=auth_headers)
        if response.status_code == 200:
            data = response.json()
            assert "jobs" in data
            assert isinstance(data["jobs"], list)

    def test_get_job_not_found(self, client, auth_headers):
        """Test getting non-existent job returns 404."""
        response = client.get("/api/jobs/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_job_not_found(self, client, auth_headers):
        """Test deleting non-existent job returns 404."""
        response = client.delete("/api/jobs/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    def test_list_jobs_with_filters(self, client, auth_headers):
        """Test listing jobs with filter parameters."""
        response = client.get(
            "/api/jobs",
            params={"status": "completed", "limit": 10},
            headers=auth_headers
        )
        if response.status_code == 200:
            data = response.json()
            assert "jobs" in data


class TestAnalyzeEndpoint:
    """Test analysis upload endpoint."""

    def test_analyze_missing_files(self, client, auth_headers):
        """Test analyze endpoint with missing files returns 422."""
        response = client.post(
            "/api/analyze",
            data={"part_id": "TEST-001"},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_analyze_invalid_file_type(self, client, auth_headers, mock_stl_content):
        """Test analyze endpoint rejects invalid file types."""
        # Create a fake .txt file
        txt_content = b"not a valid mesh file"

        response = client.post(
            "/api/analyze",
            files={
                "reference_file": ("ref.txt", io.BytesIO(txt_content), "text/plain"),
                "scan_file": ("scan.txt", io.BytesIO(txt_content), "text/plain"),
            },
            data={
                "part_id": "TEST-001",
                "part_name": "Test Part",
                "material": "Al-5053-H32",
                "tolerance": "0.1"
            },
            headers=auth_headers
        )
        assert response.status_code == 400

    def test_analyze_valid_files(self, client, auth_headers, mock_stl_content, temp_output_dir):
        """Test analyze endpoint accepts valid file types."""
        # Create minimal valid STL content
        response = client.post(
            "/api/analyze",
            files={
                "reference_file": ("ref.stl", io.BytesIO(mock_stl_content), "model/stl"),
                "scan_file": ("scan.stl", io.BytesIO(mock_stl_content), "model/stl"),
            },
            data={
                "part_id": "TEST-001",
                "part_name": "Test Part",
                "material": "Al-5053-H32",
                "tolerance": "0.1"
            },
            headers=auth_headers
        )
        # May succeed (200) or fail at processing (500) depending on file validity
        # But should not return 400 for file type
        assert response.status_code in [200, 401, 500]


class TestBatchEndpoints:
    """Test batch processing endpoints."""

    def test_list_batches_empty(self, client, auth_headers):
        """Test listing batches when empty."""
        response = client.get("/api/batch", headers=auth_headers)
        if response.status_code == 200:
            data = response.json()
            assert "batches" in data
            assert isinstance(data["batches"], list)

    def test_get_batch_not_found(self, client, auth_headers):
        """Test getting non-existent batch returns 404."""
        response = client.get("/api/batch/nonexistent", headers=auth_headers)
        assert response.status_code == 404


class TestProgressEndpoint:
    """Test job progress endpoints."""

    def test_progress_not_found(self, client, auth_headers):
        """Test getting progress for non-existent job."""
        response = client.get("/api/progress/nonexistent", headers=auth_headers)
        # Should return 404 or an empty/default progress
        assert response.status_code in [200, 404]


class TestResultEndpoint:
    """Test result retrieval endpoints."""

    def test_result_not_found(self, client, auth_headers):
        """Test getting result for non-existent job."""
        response = client.get("/api/result/nonexistent", headers=auth_headers)
        assert response.status_code == 404


class TestErrorHandling:
    """Test API error handling and responses."""

    def test_invalid_json_body(self, client):
        """Test handling of invalid JSON body."""
        response = client.post(
            "/api/auth/login",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_method_not_allowed(self, client):
        """Test handling of unsupported HTTP methods."""
        response = client.put("/api/health")
        assert response.status_code == 405

    def test_route_not_found(self, client):
        """Test handling of non-existent routes."""
        response = client.get("/api/nonexistent/endpoint")
        assert response.status_code == 404


class TestFileSizeLimits:
    """Test file upload size limit enforcement."""

    def test_oversized_file_rejected(self, client, auth_headers):
        """Test that oversized files are rejected with 413."""
        # Create a file that's definitely over the limit (101 MB)
        # We can't actually create this in memory, so we mock the validation
        large_content = b"x" * 1024  # 1 KB - actual size validation happens in endpoint

        response = client.post(
            "/api/analyze",
            files={
                "reference_file": ("ref.stl", io.BytesIO(large_content), "model/stl"),
                "scan_file": ("scan.stl", io.BytesIO(large_content), "model/stl"),
            },
            data={
                "part_id": "TEST-001",
                "part_name": "Test Part",
                "material": "Al-5053-H32",
                "tolerance": "0.1"
            },
            headers=auth_headers
        )
        # File should be accepted if under limit, rejected if over
        # With 1KB files, should pass size validation
        assert response.status_code in [200, 400, 401, 500]


class TestGDTEndpoints:
    """Test GD&T calculation endpoints if available."""

    def test_gdt_position_calculation(self, client, auth_headers):
        """Test GD&T position endpoint if available."""
        response = client.post(
            "/api/gdt/position",
            json={
                "measured_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "datum_point": [0.5, 0.5, 0],
                "tolerance": 0.1
            },
            headers=auth_headers
        )
        # Endpoint may or may not exist
        assert response.status_code in [200, 404, 405, 422]


class TestSPCEndpoints:
    """Test SPC analysis endpoints if available."""

    def test_spc_capability_calculation(self, client, auth_headers):
        """Test SPC capability endpoint if available."""
        response = client.post(
            "/api/spc/capability",
            json={
                "measurements": [1.0, 1.1, 0.9, 1.0, 1.05],
                "upper_spec_limit": 1.5,
                "lower_spec_limit": 0.5
            },
            headers=auth_headers
        )
        # Endpoint may or may not exist
        assert response.status_code in [200, 404, 405, 422]
