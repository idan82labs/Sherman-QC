from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ManualProfile = Literal["cell_operation", "software"]
AssistantMode = Literal["chat", "manual_rag_tool", "retrieval_only"]
SupportState = Literal[
    "chat_answer",
    "supported",
    "partial_support",
    "partial_support_visual_gap",
    "not_found",
    "conflict",
    "clarification",
]


class ManualInfo(BaseModel):
    manual_id: str
    profile: ManualProfile
    filename: str
    page_count: int
    indexed_pages: int
    visual_pages: int


class ManualAsset(BaseModel):
    kind: Literal["page", "crop"]
    manual_id: str
    page_number: int
    path: str
    url: str
    bbox: list[int] | None = None
    element_type: str = "page"


class ManualEvidence(BaseModel):
    citation_id: str
    manual_id: str
    profile: ManualProfile
    page_number: int
    element_type: str
    source_text: str
    excerpt: str
    page_image: ManualAsset | None = None
    crop: ManualAsset | None = None
    visual_required: bool = False
    retrieval: dict[str, Any] = Field(default_factory=dict)


class ManualAssistantChatRequest(BaseModel):
    profile: ManualProfile
    message: str = ""
    attachment_ids: list[str] = Field(default_factory=list)
    ui_language: Literal["en", "he"] = "en"
    answer_language: Literal["follow_ui", "same_as_question", "english_source"] = "follow_ui"
    retrieval_profile: Literal["fast", "accurate"] = "accurate"


class ManualAssistantChatResponse(BaseModel):
    request_id: str
    profile: ManualProfile
    support_state: SupportState
    answer: str
    citations: list[ManualEvidence]
    visual_gap: bool = False
    retrieval_trace: list[ManualEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_profile: ManualProfile | None = None
    model: str = "retrieval-only"
    provider: str = "mock"
    assistant_mode: AssistantMode = "retrieval_only"
    intent: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ManualIngestionResponse(BaseModel):
    status: str
    manuals: list[ManualInfo]
    total_pages: int
    rendered_pages: int
    crop_count: int
    cache_path: str
    warnings: list[str] = Field(default_factory=list)


class ManualAttachmentResponse(BaseModel):
    attachment_id: str
    filename: str
    content_type: str | None
    size_bytes: int
    url: str | None = None
