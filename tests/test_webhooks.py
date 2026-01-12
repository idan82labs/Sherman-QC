"""
Tests for Webhook Notification Framework
"""

import pytest
import asyncio
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from webhooks import (
    WebhookEvent,
    WebhookConfig,
    WebhookPayload,
    DeliveryResult,
    WebhookManager,
    get_webhook_manager,
    create_webhook_manager,
    job_started_payload,
    job_completed_payload,
    qc_fail_payload,
    spc_alert_payload,
    batch_completed_payload,
)


class TestWebhookEvent:
    """Test WebhookEvent enum"""

    def test_job_events(self):
        """Test job event types"""
        assert WebhookEvent.JOB_STARTED.value == "job.started"
        assert WebhookEvent.JOB_COMPLETED.value == "job.completed"
        assert WebhookEvent.JOB_FAILED.value == "job.failed"

    def test_qc_events(self):
        """Test QC event types"""
        assert WebhookEvent.QC_PASS.value == "qc.pass"
        assert WebhookEvent.QC_FAIL.value == "qc.fail"
        assert WebhookEvent.DEFECT_DETECTED.value == "defect.detected"

    def test_batch_events(self):
        """Test batch event types"""
        assert WebhookEvent.BATCH_STARTED.value == "batch.started"
        assert WebhookEvent.BATCH_COMPLETED.value == "batch.completed"
        assert WebhookEvent.BATCH_FAILED.value == "batch.failed"

    def test_spc_events(self):
        """Test SPC event types"""
        assert WebhookEvent.SPC_OUT_OF_CONTROL.value == "spc.out_of_control"
        assert WebhookEvent.SPC_TREND_DETECTED.value == "spc.trend_detected"
        assert WebhookEvent.SPC_CAPABILITY_LOW.value == "spc.capability_low"


class TestWebhookConfig:
    """Test WebhookConfig dataclass"""

    def test_config_defaults(self):
        """Test config with defaults"""
        config = WebhookConfig(
            id="wh-001",
            url="https://example.com/webhook"
        )
        assert config.id == "wh-001"
        assert config.url == "https://example.com/webhook"
        assert config.name == ""
        assert config.secret == ""
        assert config.active is True
        assert config.retry_count == 3
        assert config.retry_delay == 5.0
        assert config.timeout == 30.0

    def test_config_with_events(self):
        """Test config with specific events"""
        config = WebhookConfig(
            id="wh-002",
            url="https://example.com/webhook",
            events={WebhookEvent.JOB_COMPLETED, WebhookEvent.QC_FAIL}
        )
        assert len(config.events) == 2
        assert WebhookEvent.JOB_COMPLETED in config.events

    def test_config_to_dict(self):
        """Test config serialization"""
        config = WebhookConfig(
            id="wh-003",
            url="https://example.com/webhook",
            name="Test Webhook",
            events={WebhookEvent.JOB_COMPLETED}
        )
        data = config.to_dict()
        assert data["id"] == "wh-003"
        assert data["url"] == "https://example.com/webhook"
        assert data["name"] == "Test Webhook"
        assert "job.completed" in data["events"]

    def test_config_stats(self):
        """Test config stats tracking"""
        config = WebhookConfig(
            id="wh-004",
            url="https://example.com/webhook"
        )
        config.total_deliveries = 10
        config.successful_deliveries = 8
        config.failed_deliveries = 2

        data = config.to_dict()
        assert data["stats"]["total_deliveries"] == 10
        assert data["stats"]["successful_deliveries"] == 8
        assert data["stats"]["failed_deliveries"] == 2


class TestWebhookPayload:
    """Test WebhookPayload dataclass"""

    def test_payload_creation(self):
        """Test payload creation"""
        payload = WebhookPayload(
            event=WebhookEvent.JOB_COMPLETED,
            data={"job_id": "123", "result": "pass"}
        )
        assert payload.event == WebhookEvent.JOB_COMPLETED
        assert payload.data["job_id"] == "123"
        assert payload.source == "sherman-qc"
        assert payload.version == "1.0"

    def test_payload_to_dict(self):
        """Test payload serialization"""
        payload = WebhookPayload(
            event=WebhookEvent.QC_FAIL,
            data={"score": 75.5}
        )
        data = payload.to_dict()
        assert data["event"] == "qc.fail"
        assert data["source"] == "sherman-qc"
        assert data["data"]["score"] == 75.5
        assert "timestamp" in data

    def test_payload_to_json(self):
        """Test payload JSON serialization"""
        payload = WebhookPayload(
            event=WebhookEvent.JOB_STARTED,
            data={"job_id": "456"}
        )
        json_str = payload.to_json()
        data = json.loads(json_str)
        assert data["event"] == "job.started"
        assert data["data"]["job_id"] == "456"


class TestDeliveryResult:
    """Test DeliveryResult dataclass"""

    def test_success_result(self):
        """Test successful delivery result"""
        result = DeliveryResult(
            webhook_id="wh-001",
            success=True,
            status_code=200,
            response_body="OK",
            duration_ms=150.5
        )
        assert result.success is True
        assert result.status_code == 200
        assert result.error is None

    def test_failure_result(self):
        """Test failed delivery result"""
        result = DeliveryResult(
            webhook_id="wh-001",
            success=False,
            error="Connection refused",
            attempts=3
        )
        assert result.success is False
        assert result.error == "Connection refused"
        assert result.attempts == 3


class TestWebhookManager:
    """Test WebhookManager class"""

    def test_manager_creation(self):
        """Test manager creation"""
        manager = WebhookManager()
        assert manager.webhooks == {}
        assert manager.storage_path is None

    def test_register_webhook(self):
        """Test webhook registration"""
        manager = WebhookManager()
        config = WebhookConfig(
            id="wh-001",
            url="https://example.com/webhook"
        )

        webhook_id = manager.register_webhook(config)
        assert webhook_id == "wh-001"
        assert "wh-001" in manager.webhooks

    def test_unregister_webhook(self):
        """Test webhook unregistration"""
        manager = WebhookManager()
        config = WebhookConfig(id="wh-001", url="https://example.com")
        manager.register_webhook(config)

        result = manager.unregister_webhook("wh-001")
        assert result is True
        assert "wh-001" not in manager.webhooks

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent webhook"""
        manager = WebhookManager()
        result = manager.unregister_webhook("nonexistent")
        assert result is False

    def test_update_webhook(self):
        """Test webhook update"""
        manager = WebhookManager()
        config = WebhookConfig(
            id="wh-001",
            url="https://example.com",
            name="Original"
        )
        manager.register_webhook(config)

        updated = manager.update_webhook("wh-001", {"name": "Updated"})
        assert updated.name == "Updated"

    def test_update_nonexistent(self):
        """Test updating nonexistent webhook"""
        manager = WebhookManager()
        result = manager.update_webhook("nonexistent", {"name": "Test"})
        assert result is None

    def test_get_webhook(self):
        """Test getting webhook by ID"""
        manager = WebhookManager()
        config = WebhookConfig(id="wh-001", url="https://example.com")
        manager.register_webhook(config)

        webhook = manager.get_webhook("wh-001")
        assert webhook.id == "wh-001"

    def test_list_webhooks(self):
        """Test listing all webhooks"""
        manager = WebhookManager()
        manager.register_webhook(WebhookConfig(id="wh-001", url="https://a.com"))
        manager.register_webhook(WebhookConfig(id="wh-002", url="https://b.com"))

        webhooks = manager.list_webhooks()
        assert len(webhooks) == 2

    def test_generate_signature(self):
        """Test HMAC signature generation"""
        manager = WebhookManager()
        payload = '{"test": "data"}'
        secret = "my-secret-key"

        signature = manager._generate_signature(payload, secret)
        assert len(signature) == 64  # SHA256 hex

    def test_verify_signature(self):
        """Test signature verification"""
        manager = WebhookManager()
        payload = '{"test": "data"}'
        secret = "my-secret-key"

        signature = f"sha256={manager._generate_signature(payload, secret)}"
        assert manager.verify_signature(payload, signature, secret) is True
        assert manager.verify_signature(payload, "sha256=invalid", secret) is False


class TestWebhookManagerAsync:
    """Test async webhook delivery"""

    @pytest.fixture
    def manager(self):
        return WebhookManager()

    @pytest.fixture
    def webhook_config(self):
        return WebhookConfig(
            id="wh-test",
            url="https://example.com/webhook",
            secret="test-secret"
        )

    @pytest.mark.asyncio
    async def test_emit_no_webhooks(self, manager):
        """Test emit with no webhooks"""
        results = await manager.emit(WebhookEvent.JOB_COMPLETED, {"job_id": "123"})
        assert results == []

    @pytest.mark.asyncio
    async def test_emit_inactive_webhook(self, manager, webhook_config):
        """Test emit skips inactive webhooks"""
        webhook_config.active = False
        manager.register_webhook(webhook_config)

        results = await manager.emit(WebhookEvent.JOB_COMPLETED, {"job_id": "123"})
        assert results == []

    @pytest.mark.asyncio
    async def test_emit_filtered_events(self, manager, webhook_config):
        """Test emit respects event filters"""
        webhook_config.events = {WebhookEvent.QC_FAIL}
        manager.register_webhook(webhook_config)

        # Should be filtered out
        results = await manager.emit(WebhookEvent.JOB_COMPLETED, {"job_id": "123"})
        assert results == []

    @pytest.mark.asyncio
    async def test_send_request_success(self, manager, webhook_config):
        """Test successful HTTP request by mocking _send_request"""
        manager.register_webhook(webhook_config)

        # Mock the internal method to return success
        async def mock_send(*args, **kwargs):
            return DeliveryResult(
                webhook_id=webhook_config.id,
                success=True,
                status_code=200,
                response_body="OK",
                duration_ms=100.0
            )

        with patch.object(manager, '_send_request', side_effect=mock_send):
            payload = WebhookPayload(
                event=WebhookEvent.JOB_COMPLETED,
                data={"job_id": "123"}
            )
            results = await manager.emit(WebhookEvent.JOB_COMPLETED, {"job_id": "123"})

            assert len(results) == 1
            assert results[0].success is True
            assert results[0].status_code == 200

    @pytest.mark.asyncio
    async def test_send_request_failure(self, manager, webhook_config):
        """Test failed HTTP request"""
        manager.register_webhook(webhook_config)

        # Mock the internal method to return failure
        async def mock_send(*args, **kwargs):
            return DeliveryResult(
                webhook_id=webhook_config.id,
                success=False,
                error="Connection refused",
                duration_ms=50.0
            )

        with patch.object(manager, '_send_request', side_effect=mock_send):
            results = await manager.emit(WebhookEvent.JOB_COMPLETED, {"job_id": "123"})

            assert len(results) == 1
            assert results[0].success is False
            assert "Connection refused" in results[0].error

    @pytest.mark.asyncio
    async def test_deliver_with_retries(self, manager, webhook_config):
        """Test delivery retries on failure"""
        webhook_config.retry_count = 2
        webhook_config.retry_delay = 0.01  # Fast for testing
        manager.register_webhook(webhook_config)

        call_count = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return DeliveryResult(
                    webhook_id=webhook_config.id,
                    success=False,
                    status_code=500,
                    error="Server error"
                )
            return DeliveryResult(
                webhook_id=webhook_config.id,
                success=True,
                status_code=200
            )

        with patch.object(manager, '_send_request', side_effect=mock_send):
            payload = WebhookPayload(event=WebhookEvent.JOB_COMPLETED, data={})
            result = await manager._deliver(webhook_config, payload)

            # Should succeed on retry
            assert result.success is True


class TestWebhookManagerStorage:
    """Test webhook storage functionality"""

    def test_save_and_load_webhooks(self):
        """Test webhook persistence"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            storage_path = f.name

        try:
            # Create and save webhooks
            manager1 = WebhookManager(storage_path=storage_path)
            manager1.register_webhook(WebhookConfig(
                id="wh-001",
                url="https://example.com/webhook1",
                name="Test 1",
                events={WebhookEvent.JOB_COMPLETED}
            ))
            manager1.register_webhook(WebhookConfig(
                id="wh-002",
                url="https://example.com/webhook2",
                name="Test 2"
            ))

            # Load webhooks in new manager
            manager2 = WebhookManager(storage_path=storage_path)

            assert len(manager2.webhooks) == 2
            assert "wh-001" in manager2.webhooks
            assert manager2.webhooks["wh-001"].name == "Test 1"
            assert WebhookEvent.JOB_COMPLETED in manager2.webhooks["wh-001"].events
        finally:
            Path(storage_path).unlink()


class TestPayloadHelpers:
    """Test payload helper functions"""

    def test_job_started_payload(self):
        """Test job started payload"""
        payload = job_started_payload("job-123", "part-456", "Bracket")
        assert payload["job_id"] == "job-123"
        assert payload["part_id"] == "part-456"
        assert payload["part_name"] == "Bracket"
        assert payload["status"] == "started"

    def test_job_completed_payload(self):
        """Test job completed payload"""
        payload = job_completed_payload(
            job_id="job-123",
            part_id="part-456",
            quality_score=95.5,
            result="pass",
            defect_count=0
        )
        assert payload["job_id"] == "job-123"
        assert payload["quality_score"] == 95.5
        assert payload["result"] == "pass"
        assert payload["defect_count"] == 0

    def test_qc_fail_payload(self):
        """Test QC fail payload"""
        defects = [
            {"type": "scratch", "location": "top"},
            {"type": "dent", "location": "side"}
        ]
        recommendations = ["Inspect tooling", "Check material"]

        payload = qc_fail_payload(
            job_id="job-123",
            part_id="part-456",
            quality_score=65.0,
            defects=defects,
            recommendations=recommendations
        )
        assert payload["quality_score"] == 65.0
        assert payload["defect_count"] == 2
        assert payload["severity"] == "high"  # < 70

    def test_qc_fail_payload_medium_severity(self):
        """Test QC fail payload with medium severity"""
        payload = qc_fail_payload(
            job_id="job-123",
            part_id="part-456",
            quality_score=75.0,
            defects=[{"type": "minor"}],
            recommendations=[]
        )
        assert payload["severity"] == "medium"  # >= 70

    def test_spc_alert_payload(self):
        """Test SPC alert payload"""
        payload = spc_alert_payload(
            metric="cpk",
            value=0.8,
            limit=1.33,
            condition="out_of_control",
            job_ids=["job-1", "job-2", "job-3"]
        )
        assert payload["metric"] == "cpk"
        assert payload["current_value"] == 0.8
        assert payload["limit"] == 1.33
        assert payload["severity"] == "critical"

    def test_spc_alert_payload_warning(self):
        """Test SPC alert payload with warning severity"""
        payload = spc_alert_payload(
            metric="trend",
            value=7,
            limit=6,
            condition="trend",
            job_ids=["job-1"]
        )
        assert payload["severity"] == "warning"

    def test_batch_completed_payload(self):
        """Test batch completed payload"""
        payload = batch_completed_payload(
            batch_id="batch-001",
            name="Morning Run",
            total_parts=100,
            passed=95,
            failed=5,
            avg_quality_score=92.5
        )
        assert payload["batch_id"] == "batch-001"
        assert payload["total_parts"] == 100
        assert payload["pass_rate"] == 95.0
        assert payload["average_quality_score"] == 92.5


class TestSingletonManager:
    """Test singleton manager functions"""

    def test_get_webhook_manager_singleton(self):
        """Test singleton manager creation"""
        # Reset global state
        import webhooks
        webhooks._webhook_manager = None

        manager1 = get_webhook_manager()
        manager2 = get_webhook_manager()

        assert manager1 is manager2

    def test_create_webhook_manager_new_instance(self):
        """Test creating new manager instances"""
        manager1 = create_webhook_manager()
        manager2 = create_webhook_manager()

        assert manager1 is not manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
