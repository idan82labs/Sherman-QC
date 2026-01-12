"""
Webhook Notification Framework

Provides webhook notifications for QC events:
- Job completion (pass/fail)
- Quality alerts (defects detected)
- Batch completion
- SPC out-of-control conditions
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Supported webhook event types"""
    # Job events
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"

    # Quality events
    QC_PASS = "qc.pass"
    QC_FAIL = "qc.fail"
    DEFECT_DETECTED = "defect.detected"

    # Batch events
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"

    # SPC events
    SPC_OUT_OF_CONTROL = "spc.out_of_control"
    SPC_TREND_DETECTED = "spc.trend_detected"
    SPC_CAPABILITY_LOW = "spc.capability_low"

    # System events
    SYSTEM_ERROR = "system.error"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint"""
    id: str
    url: str
    name: str = ""
    secret: str = ""  # For HMAC signature verification
    events: Set[WebhookEvent] = field(default_factory=set)  # Empty = all events
    active: bool = True
    retry_count: int = 3
    retry_delay: float = 5.0  # seconds
    timeout: float = 30.0  # seconds
    headers: Dict[str, str] = field(default_factory=dict)

    # Stats
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_delivery: Optional[str] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "url": self.url,
            "name": self.name,
            "events": [e.value for e in self.events],
            "active": self.active,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "stats": {
                "total_deliveries": self.total_deliveries,
                "successful_deliveries": self.successful_deliveries,
                "failed_deliveries": self.failed_deliveries,
                "last_delivery": self.last_delivery,
                "last_error": self.last_error,
            }
        }


@dataclass
class WebhookPayload:
    """Payload sent to webhook endpoints"""
    event: WebhookEvent
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    source: str = "sherman-qc"
    version: str = "1.0"

    def to_dict(self) -> Dict:
        return {
            "event": self.event.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "version": self.version,
            "data": self.data
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class DeliveryResult:
    """Result of a webhook delivery attempt"""
    webhook_id: str
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 1
    duration_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class WebhookManager:
    """
    Manager for webhook registrations and delivery.

    Usage:
        manager = WebhookManager()
        manager.register_webhook(config)
        await manager.emit(WebhookEvent.JOB_COMPLETED, {"job_id": "123"})
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.storage_path = storage_path
        self._delivery_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Load webhooks from storage if path provided
        if storage_path:
            self._load_webhooks()

    def register_webhook(self, config: WebhookConfig) -> str:
        """Register a new webhook endpoint"""
        self.webhooks[config.id] = config
        self._save_webhooks()
        logger.info(f"Registered webhook: {config.id} -> {config.url}")
        return config.id

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook endpoint"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            self._save_webhooks()
            logger.info(f"Unregistered webhook: {webhook_id}")
            return True
        return False

    def update_webhook(self, webhook_id: str, updates: Dict) -> Optional[WebhookConfig]:
        """Update webhook configuration"""
        if webhook_id not in self.webhooks:
            return None

        webhook = self.webhooks[webhook_id]
        for key, value in updates.items():
            if hasattr(webhook, key):
                setattr(webhook, key, value)

        self._save_webhooks()
        return webhook

    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get a webhook by ID"""
        return self.webhooks.get(webhook_id)

    def list_webhooks(self) -> List[WebhookConfig]:
        """List all registered webhooks"""
        return list(self.webhooks.values())

    async def emit(self, event: WebhookEvent, data: Dict[str, Any]) -> List[DeliveryResult]:
        """
        Emit an event to all subscribed webhooks.

        Args:
            event: The event type
            data: Event data payload

        Returns:
            List of delivery results
        """
        payload = WebhookPayload(event=event, data=data)
        results = []

        for webhook in self.webhooks.values():
            if not webhook.active:
                continue

            # Check if webhook is subscribed to this event
            if webhook.events and event not in webhook.events:
                continue

            result = await self._deliver(webhook, payload)
            results.append(result)

        return results

    async def emit_async(self, event: WebhookEvent, data: Dict[str, Any]):
        """
        Queue an event for asynchronous delivery.
        Does not wait for delivery to complete.
        """
        payload = WebhookPayload(event=event, data=data)
        await self._delivery_queue.put((event, payload))

    async def _deliver(self, webhook: WebhookConfig, payload: WebhookPayload) -> DeliveryResult:
        """Deliver payload to a webhook endpoint with retries"""
        webhook.total_deliveries += 1
        start_time = time.time()

        for attempt in range(1, webhook.retry_count + 1):
            try:
                result = await self._send_request(webhook, payload)
                if result.success:
                    webhook.successful_deliveries += 1
                    webhook.last_delivery = datetime.now().isoformat()
                    webhook.last_error = None
                    self._save_webhooks()
                    return result
                else:
                    # Non-2xx response, might retry
                    if attempt < webhook.retry_count:
                        await asyncio.sleep(webhook.retry_delay * attempt)
            except Exception as e:
                logger.error(f"Webhook delivery error: {e}")
                if attempt == webhook.retry_count:
                    result = DeliveryResult(
                        webhook_id=webhook.id,
                        success=False,
                        error=str(e),
                        attempts=attempt,
                        duration_ms=(time.time() - start_time) * 1000
                    )

        # All retries failed
        webhook.failed_deliveries += 1
        webhook.last_error = result.error or "Delivery failed after retries"
        self._save_webhooks()
        return result

    async def _send_request(self, webhook: WebhookConfig, payload: WebhookPayload) -> DeliveryResult:
        """Send HTTP request to webhook endpoint"""
        start_time = time.time()
        json_body = payload.to_json()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Sherman-QC-Webhook/1.0",
            "X-Webhook-Event": payload.event.value,
            "X-Webhook-Timestamp": payload.timestamp,
            **webhook.headers
        }

        # Add HMAC signature if secret is configured
        if webhook.secret:
            signature = self._generate_signature(json_body, webhook.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    data=json_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook.timeout)
                ) as response:
                    body = await response.text()
                    duration = (time.time() - start_time) * 1000

                    success = 200 <= response.status < 300

                    return DeliveryResult(
                        webhook_id=webhook.id,
                        success=success,
                        status_code=response.status,
                        response_body=body[:1000],  # Truncate response
                        duration_ms=duration
                    )
        except asyncio.TimeoutError:
            return DeliveryResult(
                webhook_id=webhook.id,
                success=False,
                error="Request timed out",
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return DeliveryResult(
                webhook_id=webhook.id,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )

    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature for payload verification"""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature (for incoming webhooks)"""
        expected = self._generate_signature(payload, secret)
        return hmac.compare_digest(f"sha256={expected}", signature)

    async def start_worker(self):
        """Start background worker for async delivery"""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Webhook delivery worker started")

    async def stop_worker(self):
        """Stop background worker"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Webhook delivery worker stopped")

    async def _worker_loop(self):
        """Background worker for processing async deliveries"""
        while self._running:
            try:
                event, payload = await asyncio.wait_for(
                    self._delivery_queue.get(),
                    timeout=1.0
                )

                for webhook in self.webhooks.values():
                    if not webhook.active:
                        continue
                    if webhook.events and event not in webhook.events:
                        continue

                    asyncio.create_task(self._deliver(webhook, payload))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Webhook worker error: {e}")

    def _load_webhooks(self):
        """Load webhooks from storage"""
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for webhook_data in data.get("webhooks", []):
                events = {WebhookEvent(e) for e in webhook_data.get("events", [])}
                config = WebhookConfig(
                    id=webhook_data["id"],
                    url=webhook_data["url"],
                    name=webhook_data.get("name", ""),
                    secret=webhook_data.get("secret", ""),
                    events=events,
                    active=webhook_data.get("active", True),
                    retry_count=webhook_data.get("retry_count", 3),
                    retry_delay=webhook_data.get("retry_delay", 5.0),
                    timeout=webhook_data.get("timeout", 30.0),
                    headers=webhook_data.get("headers", {}),
                )
                self.webhooks[config.id] = config

            logger.info(f"Loaded {len(self.webhooks)} webhooks from storage")
        except Exception as e:
            logger.error(f"Error loading webhooks: {e}")

    def _save_webhooks(self):
        """Save webhooks to storage"""
        if not self.storage_path:
            return

        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "webhooks": [
                    {
                        **w.to_dict(),
                        "secret": w.secret,
                        "headers": w.headers,
                    }
                    for w in self.webhooks.values()
                ]
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving webhooks: {e}")


# Pre-built payload helpers
def job_started_payload(job_id: str, part_id: str, part_name: str) -> Dict:
    """Create payload for job.started event"""
    return {
        "job_id": job_id,
        "part_id": part_id,
        "part_name": part_name,
        "status": "started"
    }


def job_completed_payload(
    job_id: str,
    part_id: str,
    quality_score: float,
    result: str,
    defect_count: int = 0
) -> Dict:
    """Create payload for job.completed event"""
    return {
        "job_id": job_id,
        "part_id": part_id,
        "quality_score": quality_score,
        "result": result,
        "defect_count": defect_count,
        "status": "completed"
    }


def qc_fail_payload(
    job_id: str,
    part_id: str,
    quality_score: float,
    defects: List[Dict],
    recommendations: List[str]
) -> Dict:
    """Create payload for qc.fail event"""
    return {
        "job_id": job_id,
        "part_id": part_id,
        "quality_score": quality_score,
        "defect_count": len(defects),
        "defects": defects[:10],  # Limit to first 10
        "recommendations": recommendations[:5],
        "severity": "high" if quality_score < 70 else "medium"
    }


def spc_alert_payload(
    metric: str,
    value: float,
    limit: float,
    condition: str,
    job_ids: List[str]
) -> Dict:
    """Create payload for SPC alert events"""
    return {
        "metric": metric,
        "current_value": value,
        "limit": limit,
        "condition": condition,  # e.g., "out_of_control", "trend", "run"
        "affected_jobs": job_ids[:20],
        "severity": "critical" if condition == "out_of_control" else "warning"
    }


def batch_completed_payload(
    batch_id: str,
    name: str,
    total_parts: int,
    passed: int,
    failed: int,
    avg_quality_score: float
) -> Dict:
    """Create payload for batch.completed event"""
    return {
        "batch_id": batch_id,
        "name": name,
        "total_parts": total_parts,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(100 * passed / max(1, total_parts), 1),
        "average_quality_score": avg_quality_score,
        "status": "completed"
    }


# Singleton manager instance
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager(storage_path: Optional[str] = None) -> WebhookManager:
    """Get or create the singleton webhook manager"""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager(storage_path)
    return _webhook_manager


def create_webhook_manager(storage_path: Optional[str] = None) -> WebhookManager:
    """Create a new webhook manager instance"""
    return WebhookManager(storage_path)
