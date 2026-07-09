from __future__ import annotations

import base64
import hmac
import json
import mimetypes
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from domains.manual_assistant.models import (
    ManualAsset,
    ManualAssistantChatRequest,
    ManualAssistantChatResponse,
    ManualAttachmentResponse,
    ManualEvidence,
    ManualInfo,
    ManualIngestionResponse,
)
from infrastructure.rag.manual_store import (
    DATA_DIR,
    INDEX_PATH,
    MANUALS,
    REPO_ROOT,
    SUPPORT_QUERY_COVERAGE_THRESHOLD,
    SUPPORT_SCORE_THRESHOLD,
    build_index,
    ensure_index,
    new_request_id,
    normalize_text,
)
from infrastructure.rag.retriever_factory import ManualRetrievalBackend, build_manual_retriever


UPLOAD_DIR = DATA_DIR / "uploads"
MOCK_MODEL = "gpt-5.5"
MOCK_PROVIDER = "mock"
OPENAI_PROVIDER = "openai"
CODEX_PROVIDER = "codex"
CHATGPT_OAUTH_PROVIDER = "chatgpt_oauth"
LLM_PROVIDERS = {OPENAI_PROVIDER, CODEX_PROVIDER, CHATGPT_OAUTH_PROVIDER}
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_WIF_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_WIF_JWT_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
DEFAULT_CODEX_CLI_PATH = "/Applications/Codex.app/Contents/Resources/codex"
PROFILE_LABELS = {
    "cell_operation": "Cell Operation",
    "software": "Software",
}
_WIF_TOKEN_CACHE: dict[str, Any] = {}
MAX_PHOTO_UPLOAD_BYTES = int(os.environ.get("SHERMAN_CHAT_MAX_PHOTO_BYTES", str(8 * 1024 * 1024)))
MAX_PHOTO_PIXELS = int(os.environ.get("SHERMAN_CHAT_MAX_PHOTO_PIXELS", str(24_000_000)))
UPLOAD_RETENTION_SECONDS = int(os.environ.get("SHERMAN_CHAT_UPLOAD_RETENTION_SECONDS", str(24 * 60 * 60)))
Image.MAX_IMAGE_PIXELS = MAX_PHOTO_PIXELS


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_production() -> bool:
    return _truthy(os.environ.get("PRODUCTION"))


def _client_is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _manual_api_token_valid(authorization: str | None, x_sherman_chat_token: str | None) -> bool:
    expected = os.environ.get("SHERMAN_CHAT_API_TOKEN") or os.environ.get("MANUAL_ASSISTANT_API_TOKEN")
    if not expected:
        return False
    supplied = x_sherman_chat_token or ""
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            supplied = token
    return hmac.compare_digest(supplied, expected)


def authorize_manual_request(
    request: Request,
    authorization: str | None = None,
    x_sherman_chat_token: str | None = None,
) -> None:
    token_configured = bool(os.environ.get("SHERMAN_CHAT_API_TOKEN") or os.environ.get("MANUAL_ASSISTANT_API_TOKEN"))
    token_valid = _manual_api_token_valid(authorization, x_sherman_chat_token)
    require_auth = token_configured or _truthy(os.environ.get("SHERMAN_CHAT_REQUIRE_AUTH"))
    if require_auth and not token_valid:
        raise HTTPException(status_code=401, detail="ShermanChat authorization is required")
    if _active_provider() == CODEX_PROVIDER and not _client_is_loopback(request):
        if not _truthy(os.environ.get("SHERMAN_CHAT_ALLOW_REMOTE_CODEX")):
            raise HTTPException(status_code=403, detail="Local Codex provider is restricted to loopback clients")


def _include_retrieval_trace() -> bool:
    return _truthy(os.environ.get("SHERMAN_CHAT_INCLUDE_RETRIEVAL_TRACE"))


def _configured_openai_model() -> str:
    return os.environ.get("SHERMAN_CHAT_MODEL") or os.environ.get("OPENAI_MODEL") or MOCK_MODEL


def _openai_workload_identity_configured() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "OPENAI_WIF_SUBJECT_TOKEN_FILE",
            "OPENAI_WIF_IDENTITY_PROVIDER_ID",
            "OPENAI_WIF_SERVICE_ACCOUNT_ID",
        )
    )


def _openai_credentials_configured() -> bool:
    return bool(os.environ.get("OPENAI_ACCESS_TOKEN") or os.environ.get("OPENAI_API_KEY")) or (
        _openai_workload_identity_configured()
    )


def _codex_cli_path() -> str:
    return os.environ.get("CODEX_CLI_PATH") or DEFAULT_CODEX_CLI_PATH


def _codex_available() -> bool:
    return Path(_codex_cli_path()).exists()


def _exchange_openai_workload_identity_token() -> str:
    now = time.time()
    cached_token = _WIF_TOKEN_CACHE.get("access_token")
    cached_expiry = float(_WIF_TOKEN_CACHE.get("expires_at", 0))
    if cached_token and cached_expiry - 60 > now:
        return str(cached_token)

    subject_token_path = Path(os.environ["OPENAI_WIF_SUBJECT_TOKEN_FILE"])
    subject_token = subject_token_path.read_text(encoding="utf-8").strip()
    if not subject_token:
        raise RuntimeError("OpenAI workload identity subject token is empty")

    payload = {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token_type": os.environ.get(
            "OPENAI_WIF_SUBJECT_TOKEN_TYPE",
            OPENAI_WIF_JWT_SUBJECT_TOKEN_TYPE,
        ),
        "subject_token": subject_token,
        "identity_provider_id": os.environ["OPENAI_WIF_IDENTITY_PROVIDER_ID"],
        "service_account_id": os.environ["OPENAI_WIF_SERVICE_ACCOUNT_ID"],
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        os.environ.get("OPENAI_WIF_TOKEN_URL", OPENAI_WIF_TOKEN_URL),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(os.environ.get("SHERMAN_CHAT_OPENAI_TIMEOUT", "45"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            token_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI workload identity token exchange failed with HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"OpenAI workload identity token exchange failed: {exc}") from exc

    access_token = token_payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("OpenAI workload identity token exchange did not return an access token")

    expires_in = int(token_payload.get("expires_in") or 3600)
    _WIF_TOKEN_CACHE["access_token"] = access_token
    _WIF_TOKEN_CACHE["expires_at"] = now + max(60, min(expires_in, 3600))
    return access_token


def _openai_bearer_token() -> str | None:
    static_token = os.environ.get("OPENAI_ACCESS_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if static_token:
        return static_token
    if _openai_workload_identity_configured():
        return _exchange_openai_workload_identity_token()
    return None


def _active_provider() -> str:
    provider = os.environ.get("SHERMAN_CHAT_PROVIDER", MOCK_PROVIDER).strip().lower()
    if provider == OPENAI_PROVIDER and _openai_credentials_configured():
        return OPENAI_PROVIDER
    if provider == CODEX_PROVIDER and _codex_available():
        return CODEX_PROVIDER
    if provider in {CHATGPT_OAUTH_PROVIDER, "oauth", "login_with_chatgpt"}:
        return CHATGPT_OAUTH_PROVIDER
    return MOCK_PROVIDER


def active_provider() -> str:
    return _active_provider()


def _asset_url(kind: str, manual_id: str, path: str | None) -> str | None:
    if not path:
        return None
    return f"/api/manual-assistant/assets/{kind}/{manual_id}/{Path(path).name}"


def _manual_asset(kind: str, manual_id: str, page_number: int, path: str | None, bbox=None):
    if not path:
        return None
    return ManualAsset(
        kind=kind,  # type: ignore[arg-type]
        manual_id=manual_id,
        page_number=page_number,
        path=path,
        url=_asset_url(kind, manual_id, path) or "",
        bbox=bbox,
        element_type="visual_crop" if kind == "crop" else "page",
    )


def _evidence_from_hit(hit, citation_id: str, visual_required: bool = False) -> ManualEvidence:
    page = hit.page
    return ManualEvidence(
        citation_id=citation_id,
        manual_id=page.manual_id,
        profile=page.profile,  # type: ignore[arg-type]
        page_number=page.page_number,
        element_type="visual_page" if page.visual_heavy else "page_text",
        source_text=page.text[:2400],
        excerpt=hit.excerpt,
        page_image=_manual_asset("page", page.manual_id, page.page_number, page.page_image_path),
        crop=_manual_asset("crop", page.manual_id, page.page_number, page.crop_path, page.crop_bbox),
        visual_required=visual_required,
        retrieval={
            "rank": hit.rank,
            "score": hit.score,
            "query_term_coverage": hit.query_term_coverage,
            "matched_query_terms": list(hit.matched_query_terms),
            "missing_query_terms": list(hit.missing_query_terms),
            "visual_heavy": page.visual_heavy,
            "section_title": page.section_title,
            "rerank_features": hit.rerank_features,
        },
    )


def _public_asset(asset: ManualAsset | None) -> ManualAsset | None:
    if asset is None:
        return None
    if hasattr(asset, "model_copy"):
        return asset.model_copy(update={"path": ""})
    return asset.copy(update={"path": ""})


def _public_evidence(item: ManualEvidence) -> ManualEvidence:
    safe_retrieval = {
        key: value
        for key, value in item.retrieval.items()
        if key
        in {
            "rank",
            "score",
            "query_term_coverage",
            "matched_query_terms",
            "missing_query_terms",
            "visual_heavy",
            "section_title",
            "rerank_features",
        }
    }
    update = {
        "source_text": "",
        "page_image": _public_asset(item.page_image),
        "crop": _public_asset(item.crop),
        "retrieval": safe_retrieval,
    }
    if hasattr(item, "model_copy"):
        return item.model_copy(update=update)
    return item.copy(update=update)


def _public_citations(items: list[ManualEvidence]) -> list[ManualEvidence]:
    return [_public_evidence(item) for item in items]


def _public_trace(items: list[ManualEvidence], support_state: str) -> list[ManualEvidence]:
    if support_state == "not_found" or not _include_retrieval_trace():
        return []
    return _public_citations(items)


def _looks_visual_query(message: str) -> bool:
    lowered = message.lower()
    visual_terms = [
        "button",
        "interface",
        "icon",
        "screen",
        "screenshot",
        "drawing",
        "rounded corner",
        "corner",
        "ui",
        "toolguide",
        "panel",
        "diagram",
        "figure",
        "layout",
        "window",
        "כפתור",
        "לחצן",
        "אייקון",
        "סמל",
        "ממשק",
        "מסך",
        "кнопка",
        "значок",
        "иконка",
        "интерфейс",
        "экран",
    ]
    return any(term in lowered for term in visual_terms)


def _needs_visual_attachment(request: ManualAssistantChatRequest) -> bool:
    if request.attachment_ids:
        return False
    lowered = request.message.lower()
    deictic_terms = {
        "this",
        "that",
        "these",
        "those",
        "זה",
        "זו",
        "זאת",
        "הזה",
        "הזו",
        "этот",
        "эта",
        "это",
        "эти",
    }
    return _looks_visual_query(request.message) and (
        any(term in lowered.split() for term in deictic_terms)
        or "this " in lowered
        or "that " in lowered
        or "what does it" in lowered
        or "מה זה" in lowered
        or "что это" in lowered
    )


def _unsupported_external_integration_terms(message: str) -> set[str]:
    lowered = normalize_text(message, include_query_expansions=True).lower()
    terms = {
        "barcode scanner",
        "cloud backup",
        "configure printer",
        "csv invoice",
        "csv invoices",
        "email alert",
        "email alerts",
        "slack",
        "teams",
        "network printer",
        "printer",
        "webhook",
        "webhooks",
        "google drive",
        "email notification",
        "notifications",
        "supplier invoice",
        "supplier invoices",
        "invoice upload",
        "upload invoice",
        "upload invoices",
        "purchase order",
        "erp",
        "plc",
        "wifi",
        "wi-fi",
        "מדפסת",
        "גוגל דרייב",
        "חשבונית",
        "חשבוניות",
        "принтер",
        "счет",
        "счета",
    }
    return {term for term in terms if term in lowered}


def _looks_like_greeting(message: str) -> bool:
    normalized = " ".join(message.lower().strip(" ?!.,").split())
    greetings = {
        "hi",
        "hello",
        "hey",
        "shalom",
        "שלום",
        "היי",
        "привет",
        "здравствуйте",
    }
    return normalized in greetings


def _looks_like_capability_question(message: str) -> bool:
    lowered = message.lower().strip()
    capability_terms = [
        "what can you do",
        "how can you help",
        "help me",
        "who are you",
        "מה אתה",
        "איך אתה יכול לעזור",
    ]
    return any(term in lowered for term in capability_terms)


def _looks_like_manual_question(message: str) -> bool:
    lowered = message.lower()
    manual_terms = [
        "2d",
        "3d",
        "arm",
        "axis",
        "bendmaster",
        "cabinet",
        "corner",
        "dxf",
        "dwg",
        "electrical",
        "emergency",
        "fix",
        "gripper",
        "import",
        "manual",
        "mode",
        "movement",
        "procedure",
        "reference",
        "screen",
        "software",
        "step",
        "teczone",
        "trutops",
        "boost",
        "toolmaster",
        "trubend",
    ]
    question_terms = ["how", "what", "where", "when", "which", "who", "why", "fix", "issue", "problem"]
    if "?" in lowered or any(term in lowered for term in question_terms):
        return True
    return any(term in lowered for term in manual_terms) and (
        "?" in lowered or any(term in lowered for term in question_terms)
    )


def _missing_critical_query_terms(request: ManualAssistantChatRequest, top) -> bool:
    lowered = request.message.lower()
    missing = set(top.missing_query_terms)
    if "dxf" in lowered and "dxf" in missing:
        return True
    if "electrical cabinet" in lowered and {"electrical", "cabinet"}.intersection(missing):
        return True
    return False


def _support_state(request: ManualAssistantChatRequest, hits) -> tuple[str, bool]:
    if not hits:
        return "not_found", False
    top = hits[0]
    if _unsupported_external_integration_terms(request.message):
        return "not_found", False
    if _missing_critical_query_terms(request, top):
        return "not_found", False
    if top.score < SUPPORT_SCORE_THRESHOLD:
        return "not_found", False
    if top.query_term_coverage < 0.30:
        return "not_found", False
    if top.query_term_coverage < SUPPORT_QUERY_COVERAGE_THRESHOLD and top.score < 12:
        return "not_found", False
    visual_required = request.profile == "software" and _looks_visual_query(request.message)
    visual_gap = visual_required and not top.page.crop_path
    if visual_gap:
        return "partial_support_visual_gap", True
    return "supported", False


def _opposite_profile(profile: str) -> str:
    return "software" if profile == "cell_operation" else "cell_operation"


def _profile_hint_from_query(message: str) -> str | None:
    lowered = message.lower()
    cell_terms = [
        "arm",
        "axis",
        "bendmaster",
        "cabinet",
        "control column",
        "electrical",
        "emergency",
        "gripper",
        "manual control",
        "movement",
        "robot",
        "toolmaster",
    ]
    software_terms = [
        "2d",
        "3d",
        "corner",
        "dxf",
        "dwg",
        "igs",
        "import",
        "radial menu",
        "screen",
        "shortcut",
        "step",
        "teczone",
        "trutops",
    ]
    if any(term in lowered for term in cell_terms):
        return "cell_operation"
    if any(term in lowered for term in software_terms):
        return "software"
    return None


def _find_cross_profile_suggestion(
    retriever: ManualRetrievalBackend,
    request: ManualAssistantChatRequest,
):
    hinted_profile = _profile_hint_from_query(request.message)
    if hinted_profile and hinted_profile != request.profile:
        return hinted_profile, []

    other_profile = _opposite_profile(request.profile)
    if hasattr(request, "model_copy"):
        other_request = request.model_copy(update={"profile": other_profile})
    else:
        other_request = request.copy(update={"profile": other_profile})
    other_hits = retriever.retrieve(request.message, other_profile, top_k=3)
    other_support_state, _visual_gap = _support_state(other_request, other_hits)
    if other_support_state == "supported":
        return other_profile, other_hits
    return None, []


def _attachment_paths(attachment_ids: list[str]) -> list[Path]:
    _cleanup_expired_uploads()
    paths: list[Path] = []
    for attachment_id in attachment_ids[:3]:
        if not attachment_id.replace("-", "").isalnum():
            continue
        paths.extend(sorted(UPLOAD_DIR.glob(f"{attachment_id}.*")))
    return [path for path in paths if path.is_file()]


def _cleanup_expired_uploads() -> None:
    if UPLOAD_RETENTION_SECONDS <= 0 or not UPLOAD_DIR.exists():
        return
    cutoff = time.time() - UPLOAD_RETENTION_SECONDS
    for path in UPLOAD_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _openai_content(prompt: str, attachment_ids: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for path in _attachment_paths(attachment_ids):
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        if not mime.startswith("image/"):
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{encoded}",
            }
        )
    return content


def _extract_openai_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _call_openai_responses(payload: dict[str, Any]) -> dict[str, Any]:
    bearer_token = _openai_bearer_token()
    if not bearer_token:
        raise RuntimeError("OpenAI bearer credential is not configured")

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("SHERMAN_CHAT_OPENAI_TIMEOUT", "45"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI Responses API failed with HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"OpenAI Responses API request failed: {exc}") from exc


def _chatgpt_oauth_complete_url() -> str:
    return os.environ.get("SHERMAN_CHATGPT_OAUTH_COMPLETE_URL", "http://127.0.0.1:10000/api/chatgpt/complete")


def _call_chatgpt_oauth_responses(payload: dict[str, Any], http_request: Request | None) -> dict[str, Any]:
    if http_request is None:
        raise HTTPException(status_code=401, detail="Connect ChatGPT before using the GPT-5.5 assistant.")
    cookie = http_request.headers.get("cookie")
    if not cookie:
        raise HTTPException(status_code=401, detail="Connect ChatGPT before using the GPT-5.5 assistant.")

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _chatgpt_oauth_complete_url(),
        data=body,
        headers={
            "Cookie": cookie,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("SHERMAN_CHAT_OPENAI_TIMEOUT", "45"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="ignore")
        if exc.code == 401:
            raise HTTPException(status_code=401, detail="Connect ChatGPT before using the GPT-5.5 assistant.") from exc
        raise RuntimeError(f"ChatGPT OAuth Responses proxy failed with HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"ChatGPT OAuth Responses proxy request failed: {exc}") from exc


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _codex_image_args(attachment_ids: list[str]) -> list[str]:
    args: list[str] = []
    for path in _attachment_paths(attachment_ids):
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        if mime.startswith("image/"):
            args.extend(["--image", str(path)])
    return args


def _safe_codex_cwd() -> str:
    configured = os.environ.get("SHERMAN_CHAT_CODEX_CWD", "/tmp")
    try:
        path = Path(configured).expanduser().resolve()
    except OSError:
        return "/tmp"
    if not path.exists() or not path.is_dir():
        return "/tmp"
    if not _truthy(os.environ.get("SHERMAN_CHAT_ALLOW_CODEX_REPO_CWD")):
        try:
            path.relative_to(REPO_ROOT)
            return "/tmp"
        except ValueError:
            pass
    return str(path)


def _codex_child_env() -> dict[str, str]:
    allowlist = {
        "CODEX_ACCESS_TOKEN",
        "CODEX_HOME",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TMPDIR",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
    }
    extra = os.environ.get("SHERMAN_CHAT_CODEX_ENV_ALLOWLIST", "")
    allowlist.update(item.strip() for item in extra.split(",") if item.strip())
    env = {key: value for key, value in os.environ.items() if key in allowlist}
    env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"))
    env.setdefault("TMPDIR", tempfile.gettempdir())
    env["NO_COLOR"] = "1"
    return env


def _call_local_codex(prompt: str, attachment_ids: list[str] | None = None) -> str:
    output_fd, output_name = tempfile.mkstemp(prefix="sherman-codex-", suffix=".txt")
    os.close(output_fd)
    output_path = Path(output_name)
    effort = os.environ.get("SHERMAN_CHAT_REASONING_EFFORT", "low")
    timeout = float(os.environ.get("SHERMAN_CHAT_CODEX_TIMEOUT", "90"))
    cwd = _safe_codex_cwd()
    command = [
        _codex_cli_path(),
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "-C",
        cwd,
        "--model",
        _configured_openai_model(),
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-o",
        str(output_path),
        *_codex_image_args(attachment_ids or []),
        "--",
        prompt,
    ]
    env = _codex_child_env()
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-500:]
            raise RuntimeError(f"Local Codex failed with exit {result.returncode}: {detail}")
        if output_path.exists():
            answer = output_path.read_text(encoding="utf-8").strip()
            if answer:
                return answer
        return result.stdout.strip()
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Local Codex timed out after {timeout:.0f}s") from exc
    finally:
        output_path.unlink(missing_ok=True)


def _normalize_plan(raw_plan: dict[str, Any], request: ManualAssistantChatRequest) -> dict[str, Any]:
    action = raw_plan.get("action")
    if action not in {"chat_answer", "search_manuals", "needs_clarification"}:
        return _mock_gpt55_plan(request)

    profile = request.profile

    intent = raw_plan.get("intent")
    if not isinstance(intent, str) or not intent:
        intent = "manual_question" if action == "search_manuals" else "general_chat"

    tool_query = raw_plan.get("tool_query")
    if not isinstance(tool_query, str) or not tool_query.strip():
        tool_query = request.message

    answer = raw_plan.get("answer")
    if not isinstance(answer, str):
        answer = None

    return {
        "intent": intent,
        "action": action,
        "profile": profile,
        "tool_query": tool_query.strip(),
        "answer": answer,
    }


def _openai_gpt55_plan(request: ManualAssistantChatRequest) -> dict[str, Any]:
    prompt = {
        "task": "Plan one ShermanAI chat turn.",
        "rules": [
            "Answer normal greetings or capability questions directly.",
            "Use the search_manuals tool only when the user asks for operational, software, troubleshooting, procedure, safety, or manual-grounded information.",
            "The selected profile is a hard boundary. Never switch profiles inside this turn.",
            "If the selected profile appears wrong, still search only the selected profile; the server will suggest a mode switch separately.",
            "Return strict JSON only.",
        ],
        "json_schema": {
            "intent": "short intent label",
            "action": "chat_answer | search_manuals | needs_clarification",
            "profile": "echo the selected profile only",
            "tool_query": "manual search query when action is search_manuals",
            "answer": "direct chat answer when action is chat_answer",
        },
        "selected_profile": request.profile,
        "ui_language": request.ui_language,
        "message": request.message,
        "attachment_count": len(request.attachment_ids),
    }
    payload = {
        "model": _configured_openai_model(),
        "input": [
            {
                "role": "system",
                "content": "You are ShermanAI, a production-support assistant. Plan tool use carefully and return only valid JSON.",
            },
            {
                "role": "user",
                "content": _openai_content(json.dumps(prompt, ensure_ascii=False), request.attachment_ids),
            },
        ],
        "reasoning": {"effort": os.environ.get("SHERMAN_CHAT_REASONING_EFFORT", "low")},
        "text": {"verbosity": "low"},
    }
    text = _extract_openai_text(_call_openai_responses(payload))
    return _normalize_plan(_parse_json_object(text), request)


def _chatgpt_oauth_gpt55_plan(request: ManualAssistantChatRequest, http_request: Request | None) -> dict[str, Any]:
    prompt = {
        "task": "Plan one ShermanAI chat turn.",
        "rules": [
            "Answer normal greetings or capability questions directly.",
            "Use the search_manuals tool only when the user asks for operational, software, troubleshooting, procedure, safety, or manual-grounded information.",
            "The selected profile is a hard boundary. Never switch profiles inside this turn.",
            "If the selected profile appears wrong, still search only the selected profile; the server will suggest a mode switch separately.",
            "Return strict JSON only.",
        ],
        "json_schema": {
            "intent": "short intent label",
            "action": "chat_answer | search_manuals | needs_clarification",
            "profile": "echo the selected profile only",
            "tool_query": "manual search query when action is search_manuals",
            "answer": "direct chat answer when action is chat_answer",
        },
        "selected_profile": request.profile,
        "ui_language": request.ui_language,
        "message": request.message,
        "attachment_count": len(request.attachment_ids),
    }
    payload = {
        "model": _configured_openai_model(),
        "input": [
            {
                "role": "system",
                "content": "You are ShermanAI, a production-support assistant. Plan tool use carefully and return only valid JSON.",
            },
            {
                "role": "user",
                "content": _openai_content(json.dumps(prompt, ensure_ascii=False), request.attachment_ids),
            },
        ],
        "reasoning": {"effort": os.environ.get("SHERMAN_CHAT_REASONING_EFFORT", "low")},
        "text": {"verbosity": "low"},
    }
    text = _extract_openai_text(_call_chatgpt_oauth_responses(payload, http_request))
    return _normalize_plan(_parse_json_object(text), request)


def _codex_gpt55_plan(request: ManualAssistantChatRequest) -> dict[str, Any]:
    prompt = {
        "task": "Plan one ShermanChat turn. Return one strict JSON object only.",
        "rules": [
            "Answer normal greetings or capability questions directly.",
            "Use the search_manuals tool only when the user asks for operational, software, troubleshooting, procedure, safety, or manual-grounded information.",
            "The selected profile is a hard boundary. Never switch profiles inside this turn.",
            "If the selected profile appears wrong, still search only the selected profile; the server will suggest a mode switch separately.",
            "For attached photos, use the image as planning context, but still return JSON only.",
        ],
        "json_schema": {
            "intent": "short intent label",
            "action": "chat_answer | search_manuals | needs_clarification",
            "profile": "echo the selected profile only",
            "tool_query": "manual search query when action is search_manuals",
            "answer": "direct chat answer when action is chat_answer",
        },
        "selected_profile": request.profile,
        "ui_language": request.ui_language,
        "message": request.message,
        "attachment_count": len(request.attachment_ids),
    }
    text = _call_local_codex(
        "You are a JSON-only planner for a manual-grounded factory support chatbot.\n"
        f"{json.dumps(prompt, ensure_ascii=False)}",
        request.attachment_ids,
    )
    return _normalize_plan(_parse_json_object(text), request)


def _assistant_plan(
    request: ManualAssistantChatRequest,
    http_request: Request | None = None,
) -> tuple[dict[str, Any], list[str]]:
    provider = _active_provider()
    if provider == MOCK_PROVIDER:
        return _mock_gpt55_plan(request), []
    try:
        if provider == CODEX_PROVIDER:
            return _codex_gpt55_plan(request), []
        if provider == CHATGPT_OAUTH_PROVIDER:
            return _chatgpt_oauth_gpt55_plan(request, http_request), []
        return _openai_gpt55_plan(request), []
    except HTTPException:
        raise
    except Exception:
        return _mock_gpt55_plan(request), [f"{provider} planning failed; used local mock planner instead."]


def _mock_gpt55_plan(request: ManualAssistantChatRequest) -> dict:
    """Deterministic stand-in for GPT-5.5 tool planning.

    Production replacement: call GPT-5.5 with a search_manuals tool and return
    the same plan shape. This mock intentionally performs no network/API work.
    """

    message = request.message.strip()
    if request.attachment_ids and (not message or message == "[image attached]"):
        return {
            "intent": "image_context",
            "action": "needs_clarification",
            "profile": request.profile,
        }
    if _looks_like_greeting(message):
        return {
            "intent": "greeting",
            "action": "chat_answer",
            "profile": request.profile,
        }
    if _looks_like_capability_question(message):
        return {
            "intent": "capabilities",
            "action": "chat_answer",
            "profile": request.profile,
        }
    if _looks_like_manual_question(message):
        tool_query = message
        hinted_profile = _profile_hint_from_query(message)
        if hinted_profile == "cell_operation" and "bendmaster" not in message.lower():
            tool_query = f"{message} BendMaster manual control unit axes robot"
        return {
            "intent": "manual_question",
            "action": "search_manuals",
            "profile": request.profile,
            "tool_query": tool_query,
        }
    return {
        "intent": "general_chat",
        "action": "chat_answer",
        "profile": request.profile,
    }


def _chat_answer(request: ManualAssistantChatRequest, intent: str) -> str:
    if request.ui_language == "he":
        if intent == "greeting":
            return "היי, אני ShermanAI. שאל אותי על תפעול התא, תוכנה, נהלים, תקלות או בדיקות איכות. כשצריך מקור מאושר, אחפש במדריכים."
        if intent == "capabilities":
            return "אני יכול לענות בשיחה רגילה, לעזור לנסח בדיקות, ולחפש במדריכים כשצריך מקור. לשאלה תפעולית ציין תחנה, פעולה, הודעת שגיאה או מסך."
        return "אני כאן. כתוב את הבעיה, התחנה, החלק או הפעולה, ואם צריך מקור מהמדריך אחפש אותו."

    if intent == "greeting":
        return "Hi, I’m ShermanAI. Ask me about cell operation, software steps, procedures, troubleshooting, or quality checks. When approved manual evidence is needed, I’ll look it up."
    if intent == "capabilities":
        return "I can chat normally, help structure checks, and call the manual-search tool when an answer needs approved documentation. For operational issues, include the station, action, screen, or error message."
    return "I’m here. Describe the issue, station, part, or procedure, and I’ll use manual evidence when it is needed."


def _clarification_answer(request: ManualAssistantChatRequest, intent: str) -> str:
    if request.ui_language == "he":
        if intent == "image_context":
            return "קיבלתי את התמונה, אבל ניתוח תמונה עדיין לא פעיל במצב mock. כתוב מה רואים בתמונה ומה אתה רוצה לבדוק."
        if intent == "visual_attachment_required":
            return "כדי לענות על כפתור, אייקון או אזור שמופיע בתמונה, העלה צילום מסך או ציין את שם המסך/הכפתור במפורש."
        return "אני צריך עוד פרט אחד כדי לענות נכון: באיזו תחנה/מסך/ציר מדובר, ומה הודעת השגיאה או הפעולה שאתה מנסה לבצע?"
    if intent == "image_context":
        return "I received the image, but image reasoning is not active in mock mode yet. Tell me what is shown and what you want to check."
    if intent == "visual_attachment_required":
        return "Upload a screenshot/photo or name the exact screen, button, or icon before I answer that visual question."
    return "I need one more detail to answer safely: which station, screen, axis, or error message are you dealing with?"


def _rag_answer_text(
    request: ManualAssistantChatRequest,
    support_state: str,
    citations: list[ManualEvidence],
    suggested_profile: str | None = None,
) -> str:
    if support_state == "not_found":
        if suggested_profile:
            label = PROFILE_LABELS.get(suggested_profile, suggested_profile)
            if request.ui_language == "he":
                return f"לא מצאתי מקור מאושר בהקשר הנוכחי, אבל זה נראה שייך ל-{label}. החלף הקשר או הוסף תחנה/מסך/ציר."
            return f"I did not find an approved source in the current context. This looks more likely to belong to {label}; switch context or add the station, screen, or axis."
        if request.ui_language == "he":
            return "לא מצאתי מקור מאושר שמתאים לשאלה. הוסף תחנה, שם מסך, ציר, הודעת שגיאה או שם נוהל."
        return "I did not find a matching approved source. Add the station, screen name, axis, error message, or procedure name."

    top = citations[0]
    if request.ui_language == "he":
        visual = " צירפתי מקור חזותי לבדיקה." if top.crop else ""
        return f"לפי המקור במדריך: {top.excerpt}{visual}"

    visual = " I attached the relevant page/crop for verification." if top.crop else ""
    return f"According to the manual source: {top.excerpt}{visual}"


def _openai_grounded_answer(
    request: ManualAssistantChatRequest,
    support_state: str,
    citations: list[ManualEvidence],
    suggested_profile: str | None = None,
) -> str | None:
    if _active_provider() != OPENAI_PROVIDER or support_state == "not_found" or not citations:
        return None

    evidence = [
        {
            "citation_id": item.citation_id,
            "manual_id": item.manual_id,
            "profile": item.profile,
            "page_number": item.page_number,
            "excerpt": item.excerpt,
            "visual_available": bool(item.crop or item.page_image),
        }
        for item in citations
    ]
    prompt = {
        "task": "Answer the operator from approved manual evidence only.",
        "rules": [
            "Do not add facts that are not present in evidence.",
            "Be direct and concise.",
            "If steps are present, preserve their order.",
            "Mention visual evidence when a page or crop is available.",
            "Use the UI language unless the source term must stay in English.",
        ],
        "ui_language": request.ui_language,
        "question": request.message,
        "support_state": support_state,
        "suggested_profile": suggested_profile,
        "evidence": evidence,
    }
    payload = {
        "model": _configured_openai_model(),
        "input": [
            {
                "role": "system",
                "content": "You are ShermanAI. Produce a grounded answer from the supplied manual citations only.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        "reasoning": {"effort": os.environ.get("SHERMAN_CHAT_REASONING_EFFORT", "low")},
        "text": {"verbosity": "low"},
    }
    return _extract_openai_text(_call_openai_responses(payload)) or None


def _codex_grounded_answer(
    request: ManualAssistantChatRequest,
    support_state: str,
    citations: list[ManualEvidence],
    suggested_profile: str | None = None,
) -> str | None:
    if _active_provider() != CODEX_PROVIDER or support_state == "not_found" or not citations:
        return None

    evidence = [
        {
            "citation_id": item.citation_id,
            "manual_id": item.manual_id,
            "profile": item.profile,
            "page_number": item.page_number,
            "excerpt": item.excerpt,
            "source_text": item.source_text[:1800],
            "visual_available": bool(item.crop or item.page_image),
        }
        for item in citations
    ]
    prompt = {
        "task": "Answer the operator from approved manual evidence only.",
        "rules": [
            "Do not add facts that are not present in evidence.",
            "Be direct and concise.",
            "If steps are present, preserve their order.",
            "Mention visual evidence when a page or crop is available.",
            "Use the UI language unless the source term must stay in English.",
        ],
        "ui_language": request.ui_language,
        "question": request.message,
        "support_state": support_state,
        "suggested_profile": suggested_profile,
        "evidence": evidence,
    }
    return (
        _call_local_codex(
            "You are ShermanChat. Produce a grounded answer from the supplied manual citations only.\n"
            f"{json.dumps(prompt, ensure_ascii=False)}",
            request.attachment_ids,
        )
        or None
    )


def _chatgpt_oauth_grounded_answer(
    request: ManualAssistantChatRequest,
    support_state: str,
    citations: list[ManualEvidence],
    suggested_profile: str | None = None,
    http_request: Request | None = None,
) -> str | None:
    if _active_provider() != CHATGPT_OAUTH_PROVIDER or support_state == "not_found" or not citations:
        return None

    evidence = [
        {
            "citation_id": item.citation_id,
            "manual_id": item.manual_id,
            "profile": item.profile,
            "page_number": item.page_number,
            "excerpt": item.excerpt,
            "source_text": item.source_text[:1800],
            "visual_available": bool(item.crop or item.page_image),
        }
        for item in citations
    ]
    prompt = {
        "task": "Answer the operator from approved manual evidence only.",
        "rules": [
            "Do not add facts that are not present in evidence.",
            "Be direct and concise.",
            "If steps are present, preserve their order.",
            "Mention visual evidence when a page or crop is available.",
            "Use the UI language unless the source term must stay in English.",
        ],
        "ui_language": request.ui_language,
        "question": request.message,
        "support_state": support_state,
        "suggested_profile": suggested_profile,
        "evidence": evidence,
    }
    payload = {
        "model": _configured_openai_model(),
        "input": [
            {
                "role": "system",
                "content": "You are ShermanAI. Produce a grounded answer from the supplied manual citations only.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
        "reasoning": {"effort": os.environ.get("SHERMAN_CHAT_REASONING_EFFORT", "low")},
        "text": {"verbosity": "low"},
    }
    return _extract_openai_text(_call_chatgpt_oauth_responses(payload, http_request)) or None


def _llm_grounded_answer(
    request: ManualAssistantChatRequest,
    support_state: str,
    citations: list[ManualEvidence],
    suggested_profile: str | None = None,
    http_request: Request | None = None,
) -> str | None:
    provider = _active_provider()
    if provider == OPENAI_PROVIDER:
        return _openai_grounded_answer(request, support_state, citations, suggested_profile=suggested_profile)
    if provider == CODEX_PROVIDER:
        return _codex_grounded_answer(request, support_state, citations, suggested_profile=suggested_profile)
    if provider == CHATGPT_OAUTH_PROVIDER:
        return _chatgpt_oauth_grounded_answer(
            request,
            support_state,
            citations,
            suggested_profile=suggested_profile,
            http_request=http_request,
        )
    return None


def _answer_text(
    request: ManualAssistantChatRequest,
    support_state: str,
    citations: list[ManualEvidence],
    suggested_profile: str | None = None,
) -> str:
    if support_state == "clarification":
        if request.ui_language == "he":
            return "אני מוכן. בחר מצב תפעול תא או תוכנה ושאל שאלה ספציפית מהמדריך."
        return "I am ready. Choose Cell Operation or Software and ask a specific manual question."

    if support_state == "not_found":
        if suggested_profile:
            label = PROFILE_LABELS.get(suggested_profile, suggested_profile)
            if request.ui_language == "he":
                return f"לא מצאתי תמיכה במצב הנוכחי, אבל נראה שיש התאמה במצב {label}. החלף מצב ונסה שוב."
            return f"I could not support this in the selected profile. It looks more likely to be in {label}; switch modes and try again."
        if request.ui_language == "he":
            return "לא מצאתי תמיכה מספקת במדריכים עבור הפרופיל שנבחר."
        return "I could not find sufficient support for this in the selected manual profile."

    top = citations[0]
    if request.ui_language == "he":
        prefix = "תשובה לפי המדריך בלבד:"
        visual = " מצורף מקור חזותי לעמוד/אזור הרלוונטי." if top.crop else ""
        return f"{prefix} {top.excerpt}{visual}"

    visual = " A page/crop preview is attached for visual verification." if top.crop else ""
    return f"Manual-grounded answer: {top.excerpt}{visual}"


def retrieval_chat(request: ManualAssistantChatRequest) -> ManualAssistantChatResponse:
    if _looks_like_greeting(request.message):
        return ManualAssistantChatResponse(
            request_id=new_request_id(),
            profile=request.profile,
            support_state="clarification",
            answer=_answer_text(request, "clarification", []),
            citations=[],
            visual_gap=False,
            retrieval_trace=[],
            warnings=[],
            model="retrieval-only",
            provider="retrieval",
            assistant_mode="retrieval_only",
            intent="greeting",
        )

    if _needs_visual_attachment(request):
        return ManualAssistantChatResponse(
            request_id=new_request_id(),
            profile=request.profile,
            support_state="clarification",
            answer=_clarification_answer(request, "visual_attachment_required"),
            citations=[],
            visual_gap=False,
            retrieval_trace=[],
            warnings=[],
            model="retrieval-only",
            provider="retrieval",
            assistant_mode="retrieval_only",
            intent="visual_attachment_required",
        )

    pages = ensure_index(render_visuals=True)
    retriever = build_manual_retriever(pages)
    hits = retriever.retrieve(request.message, request.profile, top_k=5)
    support_state, visual_gap = _support_state(request, hits)

    visual_required = request.profile == "software" and _looks_visual_query(request.message)
    trace = [
        _evidence_from_hit(hit, f"t{idx}", visual_required=visual_required)
        for idx, hit in enumerate(hits, start=1)
    ]
    suggested_profile = None
    if support_state == "not_found":
        suggested_profile, _other_hits = _find_cross_profile_suggestion(retriever, request)

    citations = [] if support_state == "not_found" else trace[:3]
    public_citations = _public_citations(citations)
    warnings: list[str] = []
    if request.attachment_ids:
        warnings.append("Uploaded photos are stored, but image reasoning is disabled until OpenAI integration is enabled.")
    if visual_gap:
        warnings.append("The answer needs visual evidence, but no crop was available for the top citation.")

    return ManualAssistantChatResponse(
        request_id=new_request_id(),
        profile=request.profile,
        support_state=support_state,  # type: ignore[arg-type]
        answer=_answer_text(request, support_state, citations, suggested_profile=suggested_profile),
        citations=public_citations,
        visual_gap=visual_gap,
        retrieval_trace=_public_trace(trace, support_state),
        warnings=warnings,
        suggested_profile=suggested_profile,  # type: ignore[arg-type]
        model="retrieval-only",
        provider="retrieval",
        assistant_mode="retrieval_only",
        intent="manual_question",
    )


def chat(request: ManualAssistantChatRequest, http_request: Request | None = None) -> ManualAssistantChatResponse:
    provider = _active_provider()
    model = _configured_openai_model() if provider in LLM_PROVIDERS else MOCK_MODEL
    request_id = new_request_id()

    if _needs_visual_attachment(request):
        return ManualAssistantChatResponse(
            request_id=request_id,
            profile=request.profile,
            support_state="clarification",
            answer=_clarification_answer(request, "visual_attachment_required"),
            citations=[],
            visual_gap=False,
            retrieval_trace=[],
            warnings=[],
            model=model,
            provider=provider,
            assistant_mode="chat",
            intent="visual_attachment_required",
            tool_calls=[],
        )

    plan, plan_warnings = _assistant_plan(request, http_request=http_request)

    if plan["action"] == "chat_answer":
        return ManualAssistantChatResponse(
            request_id=request_id,
            profile=request.profile,
            support_state="chat_answer",  # type: ignore[arg-type]
            answer=plan.get("answer") or _chat_answer(request, plan["intent"]),
            citations=[],
            visual_gap=False,
            retrieval_trace=[],
            warnings=plan_warnings,
            model=model,
            provider=provider,
            assistant_mode="chat",
            intent=plan["intent"],
            tool_calls=[],
        )

    if plan["action"] == "needs_clarification":
        warnings = []
        if request.attachment_ids and provider not in LLM_PROVIDERS:
            warnings.append("Image reasoning is disabled in the mock GPT-5.5 provider.")
        return ManualAssistantChatResponse(
            request_id=request_id,
            profile=request.profile,
            support_state="clarification",
            answer=_clarification_answer(request, plan["intent"]),
            citations=[],
            visual_gap=False,
            retrieval_trace=[],
            warnings=plan_warnings + warnings,
            model=model,
            provider=provider,
            assistant_mode="chat",
            intent=plan["intent"],
            tool_calls=[],
        )

    tool_profile = request.profile
    tool_query = plan.get("tool_query") or request.message
    if hasattr(request, "model_copy"):
        tool_request = request.model_copy(update={"profile": tool_profile, "message": tool_query})
    else:
        tool_request = request.copy(update={"profile": tool_profile, "message": tool_query})

    pages = ensure_index(render_visuals=True)
    retriever = build_manual_retriever(pages)
    hits = retriever.retrieve(tool_query, tool_profile, top_k=5)
    support_state, visual_gap = _support_state(tool_request, hits)
    visual_required = tool_profile == "software" and _looks_visual_query(tool_query)
    trace = [
        _evidence_from_hit(hit, f"t{idx}", visual_required=visual_required)
        for idx, hit in enumerate(hits, start=1)
    ]
    suggested_profile = None
    if support_state == "not_found":
        suggested_profile, _other_hits = _find_cross_profile_suggestion(retriever, request)

    citations = [] if support_state == "not_found" else trace[:3]
    public_citations = _public_citations(citations)
    warnings: list[str] = [*plan_warnings]
    if request.attachment_ids:
        if provider in LLM_PROVIDERS:
            warnings.append("Uploaded photos were sent to GPT-5.5 for planning context when possible.")
        else:
            warnings.append("Uploaded photos are stored, but image reasoning is disabled in the mock GPT-5.5 provider.")
    if visual_gap:
        warnings.append("The answer needs visual evidence, but no crop was available for the top citation.")
    grounded_answer = None
    if citations and provider in LLM_PROVIDERS:
        try:
            grounded_answer = _llm_grounded_answer(
                tool_request,
                support_state,
                citations,
                suggested_profile=suggested_profile,
                http_request=http_request,
            )
        except HTTPException:
            raise
        except Exception:
            warnings.append(f"{provider} answer synthesis failed; used local grounded template instead.")

    tool_call = {
        "name": "search_manuals",
        "arguments": {
            "profile": tool_profile,
            "query": tool_query,
            "top_k": 5,
        },
        "status": "supported" if citations else "no_match",
        "citation_count": len(citations),
    }

    return ManualAssistantChatResponse(
        request_id=request_id,
        profile=request.profile,
        support_state=support_state,  # type: ignore[arg-type]
        answer=grounded_answer
        or _rag_answer_text(tool_request, support_state, citations, suggested_profile=suggested_profile),
        citations=public_citations,
        visual_gap=visual_gap,
        retrieval_trace=_public_trace(trace, support_state),
        warnings=warnings,
        suggested_profile=suggested_profile,  # type: ignore[arg-type]
        model=model,
        provider=provider,
        assistant_mode="manual_rag_tool",
        intent=plan["intent"],
        tool_calls=[tool_call],
    )


def ingest_manuals(force: bool = False) -> ManualIngestionResponse:
    pages = build_index(render_visuals=True, force=force)
    manuals: list[ManualInfo] = []
    rendered_pages = sum(1 for page in pages if page.page_image_path)
    crop_count = sum(1 for page in pages if page.crop_path)
    for manual in MANUALS:
        manual_pages = [page for page in pages if page.manual_id == manual["manual_id"]]
        manuals.append(
            ManualInfo(
                manual_id=manual["manual_id"],
                profile=manual["profile"],  # type: ignore[arg-type]
                filename=manual["filename"],
                page_count=len(manual_pages),
                indexed_pages=sum(1 for page in manual_pages if page.text or page.page_image_path),
                visual_pages=sum(1 for page in manual_pages if page.visual_heavy),
            )
        )
    return ManualIngestionResponse(
        status="ready",
        manuals=manuals,
        total_pages=len(pages),
        rendered_pages=rendered_pages,
        crop_count=crop_count,
        cache_path=str(INDEX_PATH),
    )


def list_manuals() -> ManualIngestionResponse:
    pages = ensure_index(render_visuals=True)
    rendered_pages = sum(1 for page in pages if page.page_image_path)
    crop_count = sum(1 for page in pages if page.crop_path)
    manuals: list[ManualInfo] = []
    for manual in MANUALS:
        manual_pages = [page for page in pages if page.manual_id == manual["manual_id"]]
        manuals.append(
            ManualInfo(
                manual_id=manual["manual_id"],
                profile=manual["profile"],  # type: ignore[arg-type]
                filename=manual["filename"],
                page_count=len(manual_pages),
                indexed_pages=sum(1 for page in manual_pages if page.text or page.page_image_path),
                visual_pages=sum(1 for page in manual_pages if page.visual_heavy),
            )
        )
    return ManualIngestionResponse(
        status="ready",
        manuals=manuals,
        total_pages=len(pages),
        rendered_pages=rendered_pages,
        crop_count=crop_count,
        cache_path=str(INDEX_PATH),
    )


async def upload_photo(file: UploadFile) -> ManualAttachmentResponse:
    _cleanup_expired_uploads()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw = await file.read(MAX_PHOTO_UPLOAD_BYTES + 1)
    if len(raw) > MAX_PHOTO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Photo is too large")
    if not raw:
        raise HTTPException(status_code=400, detail="Photo is empty")
    try:
        with Image.open(BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width * height > MAX_PHOTO_PIXELS:
                raise HTTPException(status_code=413, detail="Photo dimensions are too large")
            normalized = image.convert("RGB")
            output = BytesIO()
            normalized.save(output, format="JPEG", quality=88, optimize=True)
            sanitized = output.getvalue()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Upload must be a valid image") from exc

    attachment_id = new_request_id()
    target = UPLOAD_DIR / f"{attachment_id}.jpg"
    target.write_bytes(sanitized)
    os.chmod(target, 0o600)
    return ManualAttachmentResponse(
        attachment_id=attachment_id,
        filename=file.filename or target.name,
        content_type="image/jpeg",
        size_bytes=target.stat().st_size,
        url=None,
    )


async def transcribe_audio(file: UploadFile):
    raise HTTPException(
        status_code=501,
        detail="Voice transcription requires OpenAI API integration. The local mock assistant supports typed questions and photo attachment storage.",
    )


def get_asset(kind: str, manual_id: str, filename: str) -> FileResponse:
    if kind not in {"page", "crop"}:
        raise HTTPException(status_code=404, detail="Unknown asset kind")
    if manual_id not in {manual["manual_id"] for manual in MANUALS}:
        raise HTTPException(status_code=404, detail="Asset not found")
    if Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Asset not found")
    base = DATA_DIR / ("page_images" if kind == "page" else "crops") / manual_id
    path = (base / filename).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Asset not found") from None
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)
