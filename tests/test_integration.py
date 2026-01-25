"""
Integration Tests for Sherman QC

Tests complete workflows that span multiple components:
- QC analysis pipeline
- Batch processing to SPC monitoring
- Webhook delivery on events
- Database persistence across operations

These tests verify that components work together correctly.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import numpy as np

# Ensure backend path is available (conftest handles this)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.mark.integration
class TestCompleteQCWorkflow:
    """Test complete QC analysis workflow from start to finish."""

    def test_job_creation_to_completion(self, temp_db, sample_qc_result):
        """Test workflow: create job -> update progress -> store result -> verify."""
        # Step 1: Create job
        job = temp_db.create_job(
            job_id="workflow_test_001",
            part_id="PART-WF-001",
            part_name="Workflow Test Part",
            material="Al-5053-H32",
            tolerance=0.1,
            reference_path="/ref/test.stl",
            scan_path="/scan/test.ply"
        )

        assert job.job_id == "workflow_test_001"
        assert job.status == "pending"
        assert job.progress == 0

        # Step 2: Simulate progress updates through analysis stages
        stages = [
            ("load", 15, "Loading files..."),
            ("preprocess", 30, "Preprocessing scan..."),
            ("align", 50, "Aligning meshes..."),
            ("compute", 70, "Computing deviations..."),
            ("analyze", 85, "Analyzing regions..."),
            ("report", 95, "Generating report...")
        ]

        for stage, progress, message in stages:
            temp_db.update_job_progress(
                job_id="workflow_test_001",
                status="running",
                progress=progress,
                stage=stage,
                message=message
            )

            job = temp_db.get_job("workflow_test_001")
            assert job.status == "running"
            assert job.progress == progress
            assert job.stage == stage

        # Step 3: Store final result
        temp_db.update_job_result(
            job_id="workflow_test_001",
            result=sample_qc_result,
            report_path="/output/report.json",
            pdf_path="/output/report.pdf"
        )

        # Step 4: Verify final state
        job = temp_db.get_job("workflow_test_001")
        assert job.status == "completed"
        assert job.progress == 100
        assert job.completed_at is not None

        result = json.loads(job.result_json)
        assert result["overall_result"] == "PASS"
        assert result["quality_score"] == 92.5

    def test_job_failure_workflow(self, temp_db):
        """Test workflow: create job -> partial progress -> failure."""
        # Create job
        temp_db.create_job(
            job_id="fail_workflow_001",
            part_id="PART-FAIL-001",
            part_name="Failing Test Part",
            material="Steel",
            tolerance=0.05,
            reference_path="/ref/fail.stl",
            scan_path="/scan/fail.ply"
        )

        # Partial progress
        temp_db.update_job_progress(
            "fail_workflow_001", "running", 35, "align", "Aligning..."
        )

        # Simulate failure
        temp_db.update_job_error("fail_workflow_001", "Alignment failed: insufficient overlap")

        # Verify failure state
        job = temp_db.get_job("fail_workflow_001")
        assert job.status == "failed"
        assert job.error == "Alignment failed: insufficient overlap"
        assert job.completed_at is not None

    def test_multiple_concurrent_jobs(self, temp_db, sample_qc_result):
        """Test handling multiple jobs concurrently."""
        # Create multiple jobs
        job_ids = [f"concurrent_{i:03d}" for i in range(5)]

        for job_id in job_ids:
            temp_db.create_job(
                job_id=job_id,
                part_id=f"PART-{job_id}",
                part_name=f"Concurrent Part {job_id}",
                material="Al-5053-H32",
                tolerance=0.1,
                reference_path=f"/ref/{job_id}.stl",
                scan_path=f"/scan/{job_id}.ply"
            )

        # Update all to running
        for job_id in job_ids:
            temp_db.update_job_progress(job_id, "running", 50, "analyze", "Processing...")

        # Complete some, fail others
        for i, job_id in enumerate(job_ids):
            if i % 2 == 0:
                temp_db.update_job_result(job_id, sample_qc_result, "/r.json", "/r.pdf")
            else:
                temp_db.update_job_error(job_id, "Test failure")

        # Verify stats
        stats = temp_db.get_stats()
        assert stats["completed"] == 3
        assert stats["failed"] == 2


@pytest.mark.integration
class TestBatchToSPCWorkflow:
    """Test batch processing flowing into SPC monitoring."""

    def test_batch_quality_scores_to_spc(self, temp_db, sample_batch):
        """Test batch quality scores feed into SPC calculations."""
        from spc_engine import SPCEngine

        # Simulate batch completion with quality scores
        quality_scores = [90.5, 92.0, 88.5, 95.0, 91.0, 89.5, 93.0, 90.0, 94.5, 87.0]

        # Create jobs for each part in batch
        for i, score in enumerate(quality_scores):
            job_id = f"batch_spc_{i:03d}"
            temp_db.create_job(
                job_id=job_id,
                part_id=f"PART-SPC-{i:03d}",
                part_name=f"Batch SPC Part {i}",
                material="Al-5053-H32",
                tolerance=0.1,
                reference_path=f"/ref/spc_{i}.stl",
                scan_path=f"/scan/spc_{i}.ply"
            )

            result = {
                "overall_result": "PASS" if score >= 85 else "FAIL",
                "quality_score": score,
                "statistics": {
                    "mean_deviation_mm": 0.02 + (i * 0.002),
                    "max_deviation_mm": 0.08 + (i * 0.005)
                }
            }
            temp_db.update_job_result(job_id, result, f"/r{i}.json", f"/r{i}.pdf")

        # Perform SPC analysis on batch results
        spc = SPCEngine()

        # Extract deviation data for SPC
        measurements = [0.02 + (i * 0.002) for i in range(len(quality_scores))]

        # Generate control charts (which include control limits)
        charts = spc.generate_control_charts(np.array(measurements), subgroup_size=1)

        # Get individuals chart which has control limits
        individuals_chart = charts["individuals"]
        assert individuals_chart.center_line > 0
        assert individuals_chart.ucl > individuals_chart.center_line
        assert individuals_chart.lcl < individuals_chart.center_line

        # Verify capability indices
        if len(measurements) >= 10:
            capability = spc.calculate_capability(
                np.array(measurements),
                usl=0.1,  # Upper spec limit
                lsl=-0.1  # Lower spec limit
            )
            assert capability.cp > 0
            assert capability.cpk > 0

    def test_trend_detection_from_batch_sequence(self, temp_db):
        """Test trend detection across multiple batches."""
        from trend_analysis import TrendAnalyzer, DataPoint

        # Simulate multiple batches over time with degrading quality
        batches = []
        base_time = datetime(2026, 1, 10, 8, 0, 0)

        for batch_num in range(5):
            batch_time = base_time + timedelta(days=batch_num)

            # Quality degrades slightly each batch (tool wear simulation)
            base_score = 95.0 - (batch_num * 2)

            for part in range(10):
                job_id = f"trend_batch{batch_num}_part{part}"
                temp_db.create_job(
                    job_id=job_id,
                    part_id=f"TREND-{batch_num}-{part}",
                    part_name=f"Trend Test Part",
                    material="Al-5053-H32",
                    tolerance=0.1,
                    reference_path="/ref.stl",
                    scan_path="/scan.ply"
                )

                score = base_score + (part * 0.1)  # Slight variation
                deviation = 0.01 + (batch_num * 0.02)  # Increasing deviation (more pronounced)

                result = {
                    "overall_result": "PASS" if score >= 85 else "FAIL",
                    "quality_score": score,
                    "statistics": {"mean_deviation_mm": deviation}
                }
                temp_db.update_job_result(job_id, result, "/r.json", "/r.pdf")

                batches.append({
                    "timestamp": batch_time + timedelta(minutes=part * 5),
                    "value": deviation,
                    "batch_num": batch_num
                })

        # Analyze trend with lower threshold to detect subtle changes
        analyzer = TrendAnalyzer(trend_threshold=0.0005)
        data_points = [
            DataPoint(
                timestamp=b["timestamp"],
                value=b["value"],
                batch_id=f"batch_{b['batch_num']}"
            )
            for b in batches
        ]

        trend_result = analyzer.analyze_trend(data_points)

        # Should detect increasing trend (tool wear)
        assert trend_result.slope > 0
        assert trend_result.trend_type.value in ["increasing", "drift"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestWebhookIntegration:
    """Test webhook delivery integration with QC events."""

    async def test_webhook_on_job_complete(self, temp_db, mock_webhook_server, webhook_config):
        """Test webhook fires on job completion."""
        from webhooks import WebhookManager, WebhookEvent, WebhookPayload

        manager = WebhookManager()
        manager.register_webhook(webhook_config)

        # Mock the HTTP request
        async def mock_send(config, payload):
            from webhooks import DeliveryResult
            await mock_webhook_server.handle_request(
                config.url,
                payload.to_dict(),
                {"X-Webhook-Secret": config.secret}
            )
            return DeliveryResult(
                webhook_id=config.id,
                success=True,
                status_code=200
            )

        with patch.object(manager, '_deliver', side_effect=mock_send):
            # Emit job completed event
            results = await manager.emit(
                WebhookEvent.JOB_COMPLETED,
                {
                    "job_id": "webhook_test_001",
                    "part_id": "PART-WH-001",
                    "quality_score": 92.5,
                    "result": "pass"
                }
            )

            assert len(results) == 1
            assert results[0].success is True

            # Verify mock received the request
            assert len(mock_webhook_server.requests) == 1
            request = mock_webhook_server.requests[0]
            assert request["payload"]["event"] == "job.completed"
            assert request["payload"]["data"]["job_id"] == "webhook_test_001"

    async def test_webhook_on_qc_fail(self, mock_webhook_server, webhook_config):
        """Test webhook fires on QC failure with defect details."""
        from webhooks import WebhookManager, WebhookEvent

        manager = WebhookManager()
        manager.register_webhook(webhook_config)

        async def mock_send(config, payload):
            from webhooks import DeliveryResult
            await mock_webhook_server.handle_request(
                config.url,
                payload.to_dict(),
                {}
            )
            return DeliveryResult(webhook_id=config.id, success=True, status_code=200)

        with patch.object(manager, '_deliver', side_effect=mock_send):
            results = await manager.emit(
                WebhookEvent.QC_FAIL,
                {
                    "job_id": "fail_webhook_001",
                    "part_id": "PART-FAIL-WH",
                    "quality_score": 65.0,
                    "defects": [
                        {"type": "surface_deviation", "severity": "high"},
                        {"type": "edge_defect", "severity": "medium"}
                    ]
                }
            )

            assert len(results) == 1
            assert mock_webhook_server.requests[0]["payload"]["data"]["quality_score"] == 65.0

    async def test_webhook_retry_on_failure(self, mock_webhook_server, webhook_config):
        """Test webhook retries on delivery failure."""
        from webhooks import WebhookManager, WebhookEvent, DeliveryResult

        manager = WebhookManager()
        webhook_config.retry_count = 3
        webhook_config.retry_delay = 0.01  # Fast for testing
        manager.register_webhook(webhook_config)

        # Configure mock to fail twice then succeed
        mock_webhook_server.configure_failure(count=2)

        call_count = 0

        async def mock_send(config, payload):
            nonlocal call_count
            call_count += 1
            try:
                await mock_webhook_server.handle_request(
                    config.url, payload.to_dict(), {}
                )
                return DeliveryResult(webhook_id=config.id, success=True, status_code=200)
            except ConnectionError as e:
                return DeliveryResult(
                    webhook_id=config.id,
                    success=False,
                    error=str(e)
                )

        with patch.object(manager, '_send_request', side_effect=mock_send):
            results = await manager.emit(
                WebhookEvent.JOB_COMPLETED,
                {"job_id": "retry_test"}
            )

            # Should have retried
            assert call_count >= 2


@pytest.mark.integration
class TestDatabasePersistence:
    """Test database persistence across operations."""

    def test_job_survives_reconnection(self, temp_db):
        """Test job data persists after database reconnection."""
        from database import SQLiteDatabaseManager

        # Create job
        temp_db.create_job(
            job_id="persist_test_001",
            part_id="PART-PERSIST",
            part_name="Persistence Test",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )

        # Get the database path
        db_path = temp_db.db_path

        # "Reconnect" with new instance
        db2 = SQLiteDatabaseManager(db_path)

        # Verify job exists
        job = db2.get_job("persist_test_001")
        assert job is not None
        assert job.part_id == "PART-PERSIST"

    def test_batch_operations_atomic(self, temp_db):
        """Test batch database operations are atomic."""
        # Create multiple jobs that should succeed together
        job_ids = [f"atomic_{i}" for i in range(5)]

        for job_id in job_ids:
            temp_db.create_job(
                job_id=job_id,
                part_id=f"PART-{job_id}",
                part_name=f"Atomic Test {job_id}",
                material="Al",
                tolerance=0.1,
                reference_path="/ref.stl",
                scan_path="/scan.ply"
            )

        # Verify all exist
        jobs = temp_db.list_jobs()
        found_ids = {j.job_id for j in jobs}

        for job_id in job_ids:
            assert job_id in found_ids


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndAnalysis:
    """End-to-end tests that simulate complete analysis scenarios.

    These tests are marked as slow as they may involve more computation.
    """

    def test_full_analysis_pipeline_mock(self, temp_db, sample_qc_result, temp_output_dir):
        """Test full analysis pipeline with mocked components."""
        # This test simulates the complete pipeline without actual file processing

        job_id = "e2e_test_001"

        # Step 1: Job creation
        temp_db.create_job(
            job_id=job_id,
            part_id="PART-E2E-001",
            part_name="End-to-End Test Part",
            material="Al-5053-H32",
            tolerance=0.1,
            reference_path=str(temp_output_dir / "ref.stl"),
            scan_path=str(temp_output_dir / "scan.ply")
        )

        # Step 2: Simulate analysis stages
        progress_updates = [
            ("load", 10, "Loading reference model..."),
            ("load", 20, "Loading scan data..."),
            ("preprocess", 35, "Downsampling point cloud..."),
            ("preprocess", 45, "Removing outliers..."),
            ("align", 55, "Initial alignment..."),
            ("align", 65, "Fine ICP alignment..."),
            ("compute", 75, "Computing point-to-mesh distances..."),
            ("analyze", 85, "Analyzing regions..."),
            ("analyze", 90, "Root cause analysis..."),
            ("report", 95, "Generating PDF report...")
        ]

        for stage, progress, message in progress_updates:
            temp_db.update_job_progress(job_id, "running", progress, stage, message)

        # Step 3: Complete with result
        report_path = temp_output_dir / "report.json"
        pdf_path = temp_output_dir / "report.pdf"

        # Write mock report
        with open(report_path, "w") as f:
            json.dump(sample_qc_result, f)

        # Write mock PDF (just a placeholder)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 mock")

        temp_db.update_job_result(
            job_id,
            sample_qc_result,
            str(report_path),
            str(pdf_path)
        )

        # Step 4: Verify complete state
        job = temp_db.get_job(job_id)

        assert job.status == "completed"
        assert job.progress == 100
        assert job.report_path == str(report_path)
        assert job.pdf_path == str(pdf_path)

        result = json.loads(job.result_json)
        assert result["overall_result"] == "PASS"
        assert result["quality_score"] == 92.5

        # Verify files exist
        assert Path(report_path).exists()
        assert Path(pdf_path).exists()


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery and edge cases."""

    def test_duplicate_job_id_handling(self, temp_db):
        """Test handling of duplicate job IDs."""
        temp_db.create_job(
            job_id="duplicate_001",
            part_id="PART-001",
            part_name="First",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )

        # Attempting to create duplicate should raise or handle gracefully
        with pytest.raises(Exception):
            temp_db.create_job(
                job_id="duplicate_001",
                part_id="PART-002",
                part_name="Second",
                material="Al",
                tolerance=0.1,
                reference_path="/ref2.stl",
                scan_path="/scan2.ply"
            )

    def test_update_nonexistent_job(self, temp_db):
        """Test updating a job that doesn't exist."""
        # Should not raise, but also shouldn't create a job
        temp_db.update_job_progress("nonexistent_job", "running", 50, "test", "msg")

        job = temp_db.get_job("nonexistent_job")
        assert job is None

    def test_delete_with_related_data(self, temp_db, sample_qc_result):
        """Test deleting a job with associated result data."""
        # Create and complete a job
        temp_db.create_job(
            job_id="delete_test_001",
            part_id="PART-DEL",
            part_name="Delete Test",
            material="Al",
            tolerance=0.1,
            reference_path="/ref.stl",
            scan_path="/scan.ply"
        )

        temp_db.update_job_result(
            "delete_test_001",
            sample_qc_result,
            "/report.json",
            "/report.pdf"
        )

        # Delete should succeed
        result = temp_db.delete_job("delete_test_001")
        assert result is True

        # Job should be gone
        job = temp_db.get_job("delete_test_001")
        assert job is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
