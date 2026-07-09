from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request, UploadFile, File

from apps.api.services import manual_assistant_service
from domains.manual_assistant.models import ManualAssistantChatRequest

def _manual_assistant_access(
    request: Request,
    authorization: str | None = Header(default=None),
    x_sherman_chat_token: str | None = Header(default=None),
) -> None:
    manual_assistant_service.authorize_manual_request(
        request,
        authorization=authorization,
        x_sherman_chat_token=x_sherman_chat_token,
    )


router = APIRouter(tags=["Manual Assistant"], dependencies=[Depends(_manual_assistant_access)])


@router.post("/api/manual-assistant/chat")
def chat(chat_request: ManualAssistantChatRequest, request: Request):
    return manual_assistant_service.chat(chat_request, http_request=request)


@router.post("/api/sherman-chat/chat")
def sherman_chat(chat_request: ManualAssistantChatRequest, request: Request):
    return manual_assistant_service.chat(chat_request, http_request=request)


@router.post("/api/manual-assistant/retrieval-chat")
def retrieval_chat(request: ManualAssistantChatRequest):
    return manual_assistant_service.retrieval_chat(request)


@router.post("/api/manual-assistant/uploads/photo")
async def upload_photo(file: UploadFile = File(...)):
    return await manual_assistant_service.upload_photo(file)


@router.post("/api/manual-assistant/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    return await manual_assistant_service.transcribe_audio(file)


@router.post("/api/admin/manuals/ingest")
def ingest_manuals(force: bool = False):
    return manual_assistant_service.ingest_manuals(force=force)


@router.get("/api/admin/manuals")
def list_manuals():
    return manual_assistant_service.list_manuals()


@router.get("/api/manual-assistant/assets/{kind}/{manual_id}/{filename}")
def get_asset(kind: str, manual_id: str, filename: str):
    return manual_assistant_service.get_asset(kind, manual_id, filename)
