from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AnalysisJobResult(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    result_payload: Dict[str, Any] = Field(default_factory=dict)
