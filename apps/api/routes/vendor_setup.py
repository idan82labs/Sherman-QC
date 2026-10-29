from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.services.bend_inspection_service import require_bend_operator
from apps.api.services.vendor_setup_service import (
    VendorSetupDraft, get_vendor_setup, save_vendor_setup,
)

router = APIRouter(tags=["Vendor Setup"], dependencies=[Depends(require_bend_operator)])
router.add_api_route("/api/parts/{part_id}/vendor-setup", get_vendor_setup, methods=["GET"])
router.add_api_route("/api/parts/{part_id}/vendor-setup", save_vendor_setup, methods=["PUT"])
