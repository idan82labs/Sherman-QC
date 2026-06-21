from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import live_scan_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Live Scan"])

bind(router, "/api/live-scan/demo/options", live_scan_service.get_live_scan_demo_options, ['GET'])
bind(router, "/api/live-scan/demo/load", live_scan_service.load_live_scan_demo_scan, ['POST'])
bind(router, "/api/live-scan/session", live_scan_service.get_live_scan_session, ['GET'])
bind(router, "/api/live-scan/session/points", live_scan_service.get_live_scan_session_points, ['GET'])
bind(router, "/api/live-scan/session/{session_id}/confirm", live_scan_service.confirm_live_scan_part, ['POST'])
bind(router, "/api/live-scan/session/{session_id}/complete", live_scan_service.complete_live_scan, ['POST'])
bind(router, "/api/live-scan/session/{session_id}/cancel", live_scan_service.cancel_live_scan, ['POST'])
bind(router, "/api/live-scan/session/reset", live_scan_service.reset_live_scan_session, ['POST'])
bind(router, "/api/live-scan/start", live_scan_service.start_live_scan_manager, ['POST'])
bind(router, "/api/live-scan/stop", live_scan_service.stop_live_scan_manager, ['POST'])
bind(router, "/api/live-scan/status", live_scan_service.get_live_scan_status, ['GET'])
bind(router, "/api/live-scan/config", live_scan_service.get_live_scan_config, ['GET'])
bind(router, "/api/live-scan/config", live_scan_service.update_live_scan_config, ['POST'])
bind(router, "/api/live-scan/session/stream", live_scan_service.stream_live_scan_session, ['GET'])
