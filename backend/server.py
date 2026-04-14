"""
FastAPI Backend Server for Scan QC System
Handles file uploads, QC analysis, and PDF generation with real-time progress.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Header, Depends, Body
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
import uvicorn
import asyncio
import aiofiles
import json
import uuid
import shutil
import math
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import os
import re
import fitz  # PyMuPDF for PDF text extraction
from dataclasses import dataclass, field, asdict

# Local imports
from qc_engine import ScanQCEngine, ProgressUpdate
from pdf_generator import generate_pdf_report
from database import get_db, init_db, JobRecord
from auth import (
    get_user_manager, JWTManager, User as AuthUser,
    get_current_user as auth_get_current_user
)
from gdt_engine import GDTEngine, GDTType, GDTResult, create_gdt_engine
from spc_engine import SPCEngine, create_spc_engine
from part_catalog import (
    get_catalog, Part, PartBendSpec, LiveScanSession,
    import_parts_from_csv
)
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sherman Scan QC System API",
    description="""
## AI-Powered Quality Control for Sheet Metal Parts

This API provides comprehensive quality control analysis for CNC-machined and bent sheet metal parts.

### Features

* **3D Scan Analysis** - Compare scan data against CAD reference with ICP alignment
* **AI-Powered Inspection** - Multimodal AI analysis using Claude, GPT-4, or Gemini
* **GD&T Calculations** - ASME Y14.5 compliant geometric dimensioning and tolerancing
* **SPC Analysis** - Statistical Process Control with Cp/Cpk and control charts
* **Report Generation** - PDF reports with deviation heatmaps

### Authentication

Most endpoints require JWT authentication. Get a token via `/api/auth/login`.
Include the token in the `Authorization` header: `Bearer <token>`

### API Versioning

Current API version: v1.0
    """,
    version="1.0.0",
    contact={
        "name": "Sherman QC Support",
        "email": "support@sherman-qc.local"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://sherman-qc.local/license"
    },
    openapi_tags=[
        {
            "name": "Analysis",
            "description": "QC analysis operations - upload files and run inspections"
        },
        {
            "name": "Batch",
            "description": "Batch processing - analyze multiple parts in a single job"
        },
        {
            "name": "Jobs",
            "description": "Job management - list, retrieve, and delete QC jobs"
        },
        {
            "name": "Authentication",
            "description": "User authentication and management"
        },
        {
            "name": "GD&T",
            "description": "Geometric Dimensioning & Tolerancing calculations per ASME Y14.5"
        },
        {
            "name": "SPC",
            "description": "Statistical Process Control - capability and control charts"
        },
        {
            "name": "Reports",
            "description": "Report retrieval and download"
        }
    ]
)

# CORS configuration
# In production, set CORS_ORIGINS environment variable to comma-separated list of allowed origins
# e.g., CORS_ORIGINS="https://app.sherman-qc.com,https://admin.sherman-qc.com"
# For development, defaults to localhost origins
CORS_ORIGINS_ENV = os.environ.get("CORS_ORIGINS", "")
if CORS_ORIGINS_ENV:
    # Production: use explicit origins from environment
    ALLOWED_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_ENV.split(",") if origin.strip()]
else:
    # Development: allow common localhost ports
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# Directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# File size limits (in bytes)
MAX_MESH_FILE_SIZE = 100 * 1024 * 1024  # 100 MB for STL/PLY/OBJ files
MAX_DRAWING_FILE_SIZE = 50 * 1024 * 1024  # 50 MB for drawings/PDFs

# Initialize database
db = init_db()

# In-memory job state for real-time progress (supplements database)
job_progress = {}


def _select_analysis_python() -> str:
    """
    Pick the Python runtime for analysis workers.

    Priority:
    1) ANALYSIS_PYTHON env override
    2) Homebrew Python 3.11 on macOS (more stable Open3D on this machine class)
    3) Current interpreter
    """
    override = os.environ.get("ANALYSIS_PYTHON", "").strip()
    if override:
        return override

    def _supports_analysis(py_bin: str) -> bool:
        try:
            probe = [
                py_bin,
                "-c",
                (
                    "import open3d, numpy, scipy, trimesh, fitz, reportlab; "
                    "print(open3d.__version__)"
                ),
            ]
            proc = subprocess.run(
                probe,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            return proc.returncode == 0
        except Exception:
            return False

    candidates = [
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.11",
        shutil.which("python3.11") or "",
        sys.executable,
    ]
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = str(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists() and _supports_analysis(candidate):
            return candidate
    return sys.executable


ANALYSIS_PYTHON = _select_analysis_python()
logger.info(f"Analysis worker interpreter: {ANALYSIS_PYTHON}")


class JobProgressState:
    """Track real-time job progress in memory"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"
        self.progress = 0
        self.stage = ""
        self.message = ""


def _cleanup_stale_running_jobs(max_age_minutes: float = 20.0) -> int:
    """
    Mark stale `running` jobs as failed after server restart/interruption.

    Prevents UI from showing jobs permanently stuck at old progress values
    when their worker process no longer exists.
    """
    now = datetime.now()
    recovered = 0
    try:
        running_jobs = db.list_jobs(status="running", limit=5000, offset=0)
    except Exception as exc:
        logger.warning(f"Failed to enumerate running jobs for stale cleanup: {exc}")
        return 0

    for job in running_jobs:
        if job.job_id in job_progress:
            # Active in-memory worker in this process.
            continue
        updated_at_raw = job.updated_at or job.created_at
        try:
            updated_at = datetime.fromisoformat(updated_at_raw)
        except Exception:
            updated_at = now
        age_minutes = max(0.0, (now - updated_at).total_seconds() / 60.0)
        if age_minutes < max_age_minutes:
            continue
        msg = (
            "Analysis interrupted before completion. "
            f"Auto-closed stale running job ({age_minutes:.1f} minutes old)."
        )
        try:
            db.update_job_error(job.job_id, msg)
            recovered += 1
            logger.warning(f"Auto-closed stale running job {job.job_id} at {job.progress:.1f}%")
        except Exception as exc:
            logger.warning(f"Failed to auto-close stale running job {job.job_id}: {exc}")

    if recovered:
        logger.warning(f"Recovered {recovered} stale running jobs at startup")
    return recovered


_cleanup_stale_running_jobs()


@dataclass
class BatchPartResult:
    """Result for a single part in a batch"""
    part_index: int
    job_id: str
    part_id: str
    part_name: str
    status: str = "pending"
    progress: float = 0
    result: Optional[str] = None  # PASS/FAIL
    quality_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchJobState:
    """Track batch job state in memory"""
    batch_id: str
    name: str
    material: str
    tolerance: float
    total_parts: int
    completed_parts: int = 0
    failed_parts: int = 0
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0
    current_part: str = ""
    parts: Dict[str, BatchPartResult] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["parts"] = [p.to_dict() if hasattr(p, 'to_dict') else p for p in self.parts.values()]
        return data


# In-memory batch job storage
batch_jobs: Dict[str, BatchJobState] = {}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from technical drawing PDF for context"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(min(3, len(doc))):  # First 3 pages max
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        
        # Clean and limit text
        full_text = "\n".join(text_parts)
        # Extract key specs
        specs = []
        for line in full_text.split('\n'):
            line = line.strip()
            if any(kw in line.lower() for kw in ['tolerance', 'material', 'dimension', 'mm', '±', 'thk', 'radius']):
                specs.append(line)
        
        return " | ".join(specs[:20])  # Limit to 20 key specs
        
    except Exception as e:
        logger.warning(f"Could not extract PDF text: {e}")
        return ""


@app.get("/")
async def root():
    """API root"""
    return {"message": "Scan QC System API", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==========================================================================
# Authentication Endpoints
# ==========================================================================

security = HTTPBearer(auto_error=False)
user_manager = get_user_manager()


# ==========================================================================
# Standardized API Response Models
# ==========================================================================

class APIError(BaseModel):
    """Standardized error response format for all API errors"""
    error: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Machine-readable error code (e.g., ERR_NOT_FOUND)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Job not found",
                "code": "ERR_NOT_FOUND",
                "details": {"job_id": "abc123"}
            }
        }


class PaginatedResponse(BaseModel):
    """Base model for paginated list responses"""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total count of items matching query")
    limit: int = Field(..., description="Maximum items returned")
    offset: int = Field(..., description="Number of items skipped")
    has_more: bool = Field(..., description="Whether more items exist beyond this page")


# Pagination constants
MAX_PAGINATION_LIMIT = 1000
DEFAULT_PAGINATION_LIMIT = 50


def validate_pagination(limit: int, offset: int) -> tuple[int, int]:
    """Validate and constrain pagination parameters."""
    limit = min(max(1, limit), MAX_PAGINATION_LIMIT)
    offset = max(0, offset)
    return limit, offset


# ==========================================================================
# Input Validation Models
# ==========================================================================

class AnalysisRequest(BaseModel):
    """Validated analysis request parameters"""
    part_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Part identifier (e.g., PART-001)"
    )
    part_name: str = Field(
        default="Unnamed Part",
        max_length=200,
        description="Human-readable part name"
    )
    material: str = Field(
        default="Unknown",
        max_length=100,
        description="Material specification (e.g., Al-5053-H32)"
    )
    tolerance: float = Field(
        default=0.1,
        gt=0,
        le=10.0,
        description="Tolerance in mm (0.01 - 10.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "part_id": "BRACKET-001",
                "part_name": "Main Bracket Assembly",
                "material": "Al-5053-H32",
                "tolerance": 0.1
            }
        }


class JobListParams(BaseModel):
    """Query parameters for job listing"""
    status: Optional[str] = Field(None, description="Filter by status: pending, running, completed, failed")
    part_id: Optional[str] = Field(None, description="Filter by part ID")
    limit: int = Field(DEFAULT_PAGINATION_LIMIT, ge=1, le=MAX_PAGINATION_LIMIT)
    offset: int = Field(0, ge=0)

    @property
    def validated(self) -> tuple[int, int]:
        return validate_pagination(self.limit, self.offset)


# ==========================================================================
# Custom Exception Handler
# ==========================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Convert HTTPException to standardized APIError format"""
    # Map status codes to error codes
    code_map = {
        400: "ERR_BAD_REQUEST",
        401: "ERR_UNAUTHORIZED",
        403: "ERR_FORBIDDEN",
        404: "ERR_NOT_FOUND",
        409: "ERR_CONFLICT",
        413: "ERR_PAYLOAD_TOO_LARGE",
        422: "ERR_VALIDATION",
        429: "ERR_RATE_LIMITED",
        500: "ERR_INTERNAL",
        503: "ERR_SERVICE_UNAVAILABLE"
    }

    error_code = code_map.get(exc.status_code, f"ERR_{exc.status_code}")

    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(
            error=str(exc.detail),
            code=error_code,
            details={"status_code": exc.status_code}
        ).model_dump()
    )


from pydantic import ValidationError


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    return JSONResponse(
        status_code=422,
        content=APIError(
            error="Validation failed",
            code="ERR_VALIDATION",
            details={"errors": errors}
        ).model_dump()
    )


# ==========================================================================
# Authentication Models & Endpoints
# ==========================================================================

class LoginRequest(BaseModel):
    """Login request body"""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    """Registration request body"""
    username: str
    email: str
    password: str


class PasswordChangeRequest(BaseModel):
    """Password change request body"""
    current_password: str
    new_password: str


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[AuthUser]:
    """Get current authenticated user from JWT token"""
    if not credentials:
        return None
    return await auth_get_current_user(f"Bearer {credentials.credentials}")


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> AuthUser:
    """Require authentication - raises 401 if not authenticated"""
    if not credentials:
        raise HTTPException(401, "Authentication required")

    user = await auth_get_current_user(f"Bearer {credentials.credentials}")
    if not user:
        raise HTTPException(401, "Invalid or expired token")

    return user


async def require_admin(user: AuthUser = Depends(require_auth)) -> AuthUser:
    """Require admin role"""
    if user.role != "admin":
        raise HTTPException(403, "Admin access required")
    return user


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT token.

    Returns:
        - access_token: JWT token for API authentication
        - token_type: Always "bearer"
        - user: User information
    """
    user = user_manager.authenticate(request.username, request.password)

    if not user:
        raise HTTPException(401, "Invalid username or password")

    token = JWTManager.create_token(user)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user.to_dict()
    }


@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """
    Register a new user (requires admin approval in production).

    Default role is 'operator'. Admin can change roles.
    """
    # Check if username/email already exists
    existing = user_manager.get_user_by_username(request.username)
    if existing:
        raise HTTPException(400, "Username already exists")

    user = user_manager.create_user(
        username=request.username,
        email=request.email,
        password=request.password,
        role="operator"
    )

    if not user:
        raise HTTPException(400, "Registration failed - email may already exist")

    token = JWTManager.create_token(user)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user.to_dict()
    }


@app.get("/api/auth/me")
async def get_me(user: AuthUser = Depends(require_auth)):
    """Get current user information"""
    return user.to_dict()


@app.post("/api/auth/change-password")
async def change_password(
    request: PasswordChangeRequest,
    user: AuthUser = Depends(require_auth)
):
    """Change current user's password"""
    # Verify current password
    from auth import PasswordHasher
    if not PasswordHasher.verify_password(request.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")

    # Update password
    success = user_manager.update_password(user.id, request.new_password)
    if not success:
        raise HTTPException(500, "Password update failed")

    return {"message": "Password changed successfully"}


@app.get("/api/auth/users")
async def list_users(
    admin: AuthUser = Depends(require_admin),
    include_inactive: bool = Query(False)
):
    """List all users (admin only)"""
    users = user_manager.list_users(include_inactive=include_inactive)
    return {"users": [u.to_dict() for u in users]}


@app.put("/api/auth/users/{user_id}")
async def update_user(
    user_id: int,
    email: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    admin: AuthUser = Depends(require_admin)
):
    """Update user details (admin only)"""
    # Validate role if provided
    if role and role not in ["admin", "operator", "viewer"]:
        raise HTTPException(400, "Invalid role. Must be: admin, operator, or viewer")

    user = user_manager.update_user(
        user_id=user_id,
        email=email,
        role=role,
        is_active=is_active
    )

    if not user:
        raise HTTPException(404, "User not found")

    return user.to_dict()


@app.delete("/api/auth/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: AuthUser = Depends(require_admin)
):
    """Deactivate a user (admin only)"""
    if user_id == admin.id:
        raise HTTPException(400, "Cannot delete your own account")

    success = user_manager.delete_user(user_id)
    if not success:
        raise HTTPException(404, "User not found")

    return {"message": f"User {user_id} deactivated"}


# ==========================================================================
# Analysis Endpoints
# ==========================================================================


async def validate_file_size(file: UploadFile, max_size: int, file_type: str) -> int:
    """
    Validate file size and return actual size.
    Raises HTTPException if file exceeds limit.
    """
    # UploadFile.size may be None; read content to determine size
    content = await file.read()
    await file.seek(0)  # Reset for later reading
    size = len(content)

    if size <= 0:
        raise HTTPException(400, f"{file_type} file is empty")

    if size > max_size:
        raise HTTPException(
            413,
            f"{file_type} file exceeds maximum size of {max_size // (1024*1024)}MB "
            f"(got {size // (1024*1024)}MB)"
        )
    return size


def _sanitize_error_message(message: Any) -> str:
    """Remove ANSI escape codes and normalize whitespace for UI-safe errors."""
    text = str(message) if message is not None else ""
    text = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", text)
    text = " ".join(text.split())
    return text.strip()


def _json_safe(value: Any) -> Any:
    """Recursively coerce values to strict JSON-safe primitives."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        num = float(value)
        return num if math.isfinite(num) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _run_analysis_worker_subprocess(
    job_id: str,
    ref_path: str,
    scan_path: str,
    part_id: str,
    part_name: str,
    material: str,
    tolerance: float,
    drawing_context: str,
    output_dir: Path,
    progress_callback,
    drawing_path: Optional[str] = None,
    dimension_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run heavy analysis in an isolated subprocess.

    This protects the API server from native crashes in geometry libraries.
    """
    worker_script = BASE_DIR / "backend" / "analysis_worker.py"
    cmd = [
        ANALYSIS_PYTHON,
        str(worker_script),
        "--job-id",
        job_id,
        "--ref-path",
        ref_path,
        "--scan-path",
        scan_path,
        "--part-id",
        part_id,
        "--part-name",
        part_name,
        "--material",
        material,
        "--tolerance",
        str(tolerance),
        "--drawing-context",
        drawing_context or "",
        "--output-dir",
        str(output_dir),
    ]
    if drawing_path:
        cmd.extend(["--drawing-path", drawing_path])
    if dimension_path:
        cmd.extend(["--dimension-path", dimension_path])

    worker_env = os.environ.copy()
    configured_threads = os.environ.get("ANALYSIS_WORKER_THREADS", "").strip()
    if configured_threads.isdigit() and int(configured_threads) > 0:
        default_threads = str(int(configured_threads))
    else:
        default_threads = str(max(4, min(16, (os.cpu_count() or 8))))
    worker_env.setdefault("OMP_NUM_THREADS", default_threads)
    worker_env.setdefault("OPENBLAS_NUM_THREADS", default_threads)
    worker_env.setdefault("MKL_NUM_THREADS", default_threads)
    worker_env.setdefault("NUMEXPR_NUM_THREADS", default_threads)
    worker_env.setdefault("VECLIB_MAXIMUM_THREADS", default_threads)
    logger.info(
        "Launching analysis worker with %s (threads=%s)",
        ANALYSIS_PYTHON,
        default_threads,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=worker_env,
    )

    result_payload: Optional[Dict[str, Any]] = None
    worker_error: Optional[str] = None
    last_freeform_line = ""

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        if "\t" not in line:
            last_freeform_line = line
            continue
        tag, payload = line.split("\t", 1)
        try:
            data = json.loads(payload)
        except Exception:
            last_freeform_line = line
            continue

        if tag == "PROGRESS":
            try:
                progress_callback(
                    ProgressUpdate(
                        stage=str(data.get("stage", "")),
                        progress=float(data.get("progress", 0)),
                        message=str(data.get("message", "")),
                    )
                )
            except Exception:
                pass
        elif tag == "RESULT":
            result_payload = data
        elif tag == "ERROR":
            worker_error = _sanitize_error_message(data.get("error", "Analysis worker failed"))

    exit_code = proc.wait()
    if exit_code != 0:
        error_msg = worker_error or _sanitize_error_message(last_freeform_line) or (
            f"Analysis worker crashed (exit code {exit_code})"
        )
        return {"ok": False, "error": error_msg, "exit_code": exit_code}

    if not result_payload:
        return {
            "ok": False,
            "error": "Analysis worker finished without a result payload",
            "exit_code": exit_code,
        }

    return {"ok": True, **result_payload}


def _run_bend_worker_subprocess(
    job_id: str,
    cad_path: str,
    scan_path: str,
    part_id: str,
    part_name: str,
    tolerance_angle: float,
    tolerance_radius: float,
    output_dir: Path,
    progress_callback,
) -> Dict[str, Any]:
    """
    Run bend inspection in an isolated subprocess.

    This keeps the API responsive while heavy geometry work is running and
    applies conservative thread limits on macOS laptops.
    """
    worker_script = BASE_DIR / "backend" / "bend_analysis_worker.py"
    cmd = [
        ANALYSIS_PYTHON,
        str(worker_script),
        "--job-id",
        job_id,
        "--cad-path",
        cad_path,
        "--scan-path",
        scan_path,
        "--part-id",
        part_id,
        "--part-name",
        part_name,
        "--tolerance-angle",
        str(tolerance_angle),
        "--tolerance-radius",
        str(tolerance_radius),
        "--output-dir",
        str(output_dir),
    ]

    worker_env = os.environ.copy()
    configured_threads = os.environ.get("BEND_WORKER_THREADS", "").strip()
    if configured_threads.isdigit() and int(configured_threads) > 0:
        default_threads = str(int(configured_threads))
    else:
        default_threads = "1" if sys.platform == "darwin" else str(max(4, min(16, (os.cpu_count() or 8))))
    worker_env.setdefault("OMP_NUM_THREADS", default_threads)
    worker_env.setdefault("OPENBLAS_NUM_THREADS", default_threads)
    worker_env.setdefault("MKL_NUM_THREADS", default_threads)
    worker_env.setdefault("NUMEXPR_NUM_THREADS", default_threads)
    worker_env.setdefault("VECLIB_MAXIMUM_THREADS", default_threads)
    worker_env.setdefault("BEND_PHASE_LOG_PATH", str(output_dir / "bend_worker.phase.log"))
    logger.info(
        "Launching bend worker with %s (threads=%s)",
        ANALYSIS_PYTHON,
        default_threads,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=worker_env,
    )

    result_payload: Optional[Dict[str, Any]] = None
    worker_error: Optional[str] = None
    last_freeform_line = ""

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        if "\t" not in line:
            last_freeform_line = line
            continue
        tag, payload = line.split("\t", 1)
        try:
            data = json.loads(payload)
        except Exception:
            last_freeform_line = line
            continue

        if tag == "PROGRESS":
            try:
                progress_callback(
                    ProgressUpdate(
                        stage=str(data.get("stage", "")),
                        progress=float(data.get("progress", 0)),
                        message=str(data.get("message", "")),
                    )
                )
            except Exception:
                pass
        elif tag == "RESULT":
            result_payload = data
        elif tag == "ERROR":
            worker_error = _sanitize_error_message(data.get("error", "Bend worker failed"))

    exit_code = proc.wait()
    if exit_code != 0:
        error_msg = worker_error or _sanitize_error_message(last_freeform_line) or (
            f"Bend worker crashed (exit code {exit_code})"
        )
        return {"ok": False, "error": error_msg, "exit_code": exit_code}

    if not result_payload:
        return {
            "ok": False,
            "error": "Bend worker finished without a result payload",
            "exit_code": exit_code,
        }

    return {"ok": True, **result_payload}


@app.post("/api/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    reference_file: UploadFile = File(...),
    scan_file: UploadFile = File(...),
    drawing_file: Optional[UploadFile] = File(None),
    dimension_file: Optional[UploadFile] = File(None),
    part_id: str = Form("PART_001"),
    part_name: str = Form("Unnamed Part"),
    material: str = Form("Al-5053-H32"),
    tolerance: float = Form(0.1)
):
    """
    Start QC analysis job.

    - reference_file: Reference STL model
    - scan_file: Scan file (STL, PLY)
    - drawing_file: Optional PDF technical drawing for context
    - dimension_file: Optional XLSX file with dimension specifications (bend angles, tolerances)
    - part_id: Part number/ID
    - part_name: Part name/description
    - material: Material specification
    - tolerance: Tolerance in mm (plus/minus)

    Returns job_id for tracking progress.
    """
    # Validate file types
    ref_ext = Path(reference_file.filename).suffix.lower()
    scan_ext = Path(scan_file.filename).suffix.lower()

    # Reference can be mesh (STL/OBJ/PLY) or CAD (STEP/IGES)
    valid_ref_exts = ['.stl', '.obj', '.ply', '.step', '.stp', '.iges', '.igs']
    if ref_ext not in valid_ref_exts:
        raise HTTPException(400, f"Reference must be STL/OBJ/PLY/STEP/IGES, got {ref_ext}")

    if scan_ext not in ['.stl', '.ply', '.obj', '.pcd']:
        raise HTTPException(400, f"Scan must be STL/PLY/OBJ/PCD, got {scan_ext}")

    # Validate file sizes
    await validate_file_size(reference_file, MAX_MESH_FILE_SIZE, "Reference")
    await validate_file_size(scan_file, MAX_MESH_FILE_SIZE, "Scan")

    # Create job ID
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded files
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    ref_path = job_dir / f"reference{ref_ext}"
    scan_path = job_dir / f"scan{scan_ext}"

    # Use async file I/O to avoid blocking the event loop
    ref_content = await reference_file.read()
    async with aiofiles.open(ref_path, "wb") as f:
        await f.write(ref_content)

    scan_content = await scan_file.read()
    async with aiofiles.open(scan_path, "wb") as f:
        await f.write(scan_content)

    # Save drawing if provided (PDF or image)
    drawing_path = None
    drawing_context = ""
    if drawing_file and drawing_file.filename:
        drawing_ext = Path(drawing_file.filename).suffix.lower()
        valid_drawing_exts = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
        if drawing_ext in valid_drawing_exts:
            # Validate drawing file size
            await validate_file_size(drawing_file, MAX_DRAWING_FILE_SIZE, "Drawing")

            drawing_path = job_dir / f"drawing{drawing_ext}"
            drawing_content = await drawing_file.read()
            async with aiofiles.open(drawing_path, "wb") as f:
                await f.write(drawing_content)
            # Extract text from PDF, or note that image will be analyzed visually
            if drawing_ext == '.pdf':
                drawing_context = extract_text_from_pdf(str(drawing_path))
            else:
                drawing_context = f"[Technical drawing image: {drawing_file.filename}]"

    # Save dimension file if provided (XLSX)
    dimension_path = None
    if dimension_file and dimension_file.filename:
        dim_ext = Path(dimension_file.filename).suffix.lower()
        if dim_ext in ['.xlsx', '.xls']:
            # Validate file size (use drawing limit)
            await validate_file_size(dimension_file, MAX_DRAWING_FILE_SIZE, "Dimension")

            dimension_path = job_dir / f"dimensions{dim_ext}"
            dim_content = await dimension_file.read()
            async with aiofiles.open(dimension_path, "wb") as f:
                await f.write(dim_content)
            logger.info(f"Saved dimension file: {dimension_path}")

    # Create job in database
    db.create_job(
        job_id=job_id,
        part_id=part_id,
        part_name=part_name,
        material=material,
        tolerance=tolerance,
        reference_path=str(ref_path),
        scan_path=str(scan_path),
        drawing_path=str(drawing_path) if drawing_path else None
    )

    # Create in-memory progress tracker
    job_progress[job_id] = JobProgressState(job_id)

    # Start background analysis
    background_tasks.add_task(
        run_analysis,
        job_id,
        str(ref_path),
        str(scan_path),
        part_id,
        part_name,
        material,
        tolerance,
        drawing_context,
        str(drawing_path) if drawing_path else None,
        str(dimension_path) if dimension_path else None
    )

    return {"job_id": job_id, "status": "started"}


async def run_analysis(
    job_id: str,
    ref_path: str,
    scan_path: str,
    part_id: str,
    part_name: str,
    material: str,
    tolerance: float,
    drawing_context: str,
    drawing_path: str = None,
    dimension_path: str = None
):
    """Run QC analysis in background"""
    progress_state = job_progress.get(job_id)
    if not progress_state:
        progress_state = JobProgressState(job_id)
        job_progress[job_id] = progress_state

    progress_state.status = "running"

    # Update database status
    db.update_job_progress(job_id, "running", 0, "init", "Starting analysis...")

    def progress_callback(update: ProgressUpdate):
        """Update progress with error handling to prevent analysis interruption"""
        try:
            progress_state.stage = update.stage
            progress_state.progress = update.progress
            progress_state.message = update.message
            # Periodically update database (every 10% or on stage change)
            if update.progress % 10 == 0 or update.stage in ["load", "preprocess", "align", "analyze", "ai", "complete"]:
                db.update_job_progress(job_id, "running", update.progress, update.stage, update.message)
        except Exception as e:
            # Log error but don't crash the analysis
            logger.warning(f"Progress update failed for job {job_id}: {e}")

    try:
        # Create output directory for this job
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        progress_state.stage = "worker"
        progress_state.progress = max(progress_state.progress, 2)
        progress_state.message = "Launching isolated analysis worker..."
        db.update_job_progress(job_id, "running", progress_state.progress, "worker", progress_state.message)

        worker_result = await asyncio.to_thread(
            _run_analysis_worker_subprocess,
            job_id,
            ref_path,
            scan_path,
            part_id,
            part_name,
            material,
            tolerance,
            drawing_context,
            output_dir,
            progress_callback,
            drawing_path,
            dimension_path,
        )
        if not worker_result.get("ok"):
            raise RuntimeError(worker_result.get("error") or "Analysis worker failed")

        report_path = Path(str(worker_result["report_path"]))
        pdf_path = Path(str(worker_result["pdf_path"]))
        if not report_path.exists():
            raise FileNotFoundError(f"Analysis worker produced no report file: {report_path}")
        with report_path.open("r", encoding="utf-8") as f:
            report_dict = _json_safe(json.load(f))

        # Update in-memory state
        progress_state.status = "completed"
        progress_state.progress = 100
        progress_state.stage = "complete"
        progress_state.message = "Analysis complete!"

        # Update database with result
        db.update_job_result(
            job_id=job_id,
            result=report_dict,
            report_path=str(report_path),
            pdf_path=str(pdf_path)
        )

    except Exception as e:
        import traceback
        clean_error = _sanitize_error_message(e)
        logger.error(f"Analysis failed: {clean_error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        progress_state.status = "failed"
        progress_state.message = f"Error: {clean_error}"

        # Update database with error
        db.update_job_error(job_id, clean_error)


@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str):
    """Get current progress of analysis job"""
    # Try in-memory state first for real-time updates
    progress_state = job_progress.get(job_id)
    if progress_state:
        return {
            "job_id": job_id,
            "status": progress_state.status,
            "progress": progress_state.progress,
            "stage": progress_state.stage,
            "message": progress_state.message,
            "error": None if progress_state.status != "failed" else progress_state.message
        }

    # Fall back to database for persisted jobs
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "message": job.message,
        "error": job.error
    }


@app.get("/api/progress/{job_id}/stream")
async def stream_progress(job_id: str):
    """Stream progress updates via SSE"""
    # Check if job exists (in-memory or database)
    progress_state = job_progress.get(job_id)
    if not progress_state:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        # Job exists in DB but not in memory - create progress state
        progress_state = JobProgressState(job_id)
        progress_state.status = job.status
        progress_state.progress = job.progress
        progress_state.stage = job.stage
        progress_state.message = job.message
        job_progress[job_id] = progress_state

    async def generate():
        last_progress = -1
        while progress_state.status in ["pending", "running"]:
            if progress_state.progress != last_progress:
                data = {
                    "status": progress_state.status,
                    "progress": progress_state.progress,
                    "stage": progress_state.stage,
                    "message": progress_state.message
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = progress_state.progress
            await asyncio.sleep(0.2)

        # Final update
        error = progress_state.message if progress_state.status == "failed" else None
        data = {
            "status": progress_state.status,
            "progress": progress_state.progress,
            "stage": progress_state.stage,
            "message": progress_state.message,
            "error": error
        }
        yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Get analysis result"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status == "running":
        progress_state = job_progress.get(job_id)
        progress = progress_state.progress if progress_state else job.progress
        return _json_safe({"status": "running", "progress": progress})

    if job.status == "failed":
        raise HTTPException(500, f"Analysis failed: {job.error}")

    # Parse result from JSON
    result = None
    if job.result_json:
        try:
            result = json.loads(job.result_json)
        except json.JSONDecodeError:
            result = None

    return _json_safe({
        "status": "completed",
        "result": result,
        "pdf_available": job.pdf_path is not None
    })


@app.get("/api/download/{job_id}/pdf")
async def download_pdf(job_id: str):
    """Download PDF report"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if not job.pdf_path or not Path(job.pdf_path).exists():
        raise HTTPException(404, "PDF not available")

    return FileResponse(
        job.pdf_path,
        media_type="application/pdf",
        filename=f"QC_Report_{job.part_id}.pdf"
    )


@app.get("/api/download/{job_id}/json")
async def download_json(job_id: str):
    """Download JSON report"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if not job.report_path or not Path(job.report_path).exists():
        raise HTTPException(404, "Report not available")

    return FileResponse(
        job.report_path,
        media_type="application/json",
        filename=f"QC_Report_{job.part_id}.json"
    )


@app.get("/api/heatmap/{job_id}/{filename}")
async def get_heatmap(job_id: str, filename: str):
    """Get heatmap image for a job"""
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    heatmap_path = OUTPUT_DIR / job_id / filename
    if not heatmap_path.exists():
        raise HTTPException(404, "Heatmap not found")

    return FileResponse(
        str(heatmap_path),
        media_type="image/png",
        filename=filename
    )


@app.get("/api/heatmaps/{job_id}")
async def list_heatmaps(job_id: str):
    """List all available heatmaps for a job"""
    output_dir = OUTPUT_DIR / job_id
    if not output_dir.exists():
        raise HTTPException(404, "Job not found")

    heatmaps = []
    for png_file in output_dir.glob("heatmap_*.png"):
        heatmaps.append({
            "name": png_file.stem.replace("heatmap_", ""),
            "url": f"/api/heatmap/{job_id}/{png_file.name}"
        })

    return {"job_id": job_id, "heatmaps": heatmaps}


@app.get("/api/files/{job_id}/{filename:path}")
async def get_uploaded_file(job_id: str, filename: str):
    """Serve uploaded files (scan, reference) for 3D viewer.

    Args:
        job_id: The job ID
        filename: The filename (e.g., scan.ply, reference.stl)
    """
    # Validate to prevent path traversal
    if ".." in job_id or ".." in filename:
        raise HTTPException(400, "Invalid path")
    if "/" in job_id or "\\" in job_id:
        raise HTTPException(400, "Invalid job ID")

    file_path = UPLOAD_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    # Determine media type based on extension
    ext = file_path.suffix.lower()
    media_types = {
        ".stl": "application/sla",
        ".ply": "application/x-ply",
        ".obj": "text/plain",
        ".step": "application/step",
        ".stp": "application/step",
        ".iges": "application/iges",
        ".igs": "application/iges",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=filename
    )


@app.get("/api/deviations/{job_id}")
async def get_deviations(job_id: str):
    """Get deviation values for 3D viewer heatmap.

    Returns the deviation array as JSON for rendering heatmaps in the 3D viewer.
    """
    import numpy as np

    # Check output directory for deviations file
    deviations_path = OUTPUT_DIR / job_id / "deviations.npy"
    if not deviations_path.exists():
        raise HTTPException(404, "Deviations not available for this job")

    try:
        deviations = np.load(str(deviations_path))
        return _json_safe({
            "job_id": job_id,
            "count": len(deviations),
            "deviations": deviations.tolist()
        })
    except Exception as e:
        raise HTTPException(500, f"Failed to load deviations: {str(e)}")


@app.head("/api/aligned-scan/{job_id}.ply")
@app.get("/api/aligned-scan/{job_id}.ply")
async def get_aligned_scan(job_id: str):
    """Get aligned scan for 3D viewer.

    Returns the scan AFTER ICP alignment to the reference, so both models
    are in the same coordinate system for proper overlay visualization.
    """
    scan_path = OUTPUT_DIR / job_id / "aligned_scan.ply"
    if not scan_path.exists():
        raise HTTPException(404, "Aligned scan not available for this job")

    return FileResponse(
        str(scan_path),
        media_type="application/x-ply",
        filename="aligned_scan.ply"
    )


@app.head("/api/reference-mesh/{job_id}.ply")
@app.get("/api/reference-mesh/{job_id}.ply")
async def get_reference_mesh(job_id: str):
    """Get converted reference mesh for 3D viewer.

    Returns the reference mesh converted to PLY format for display in 3D viewer.
    Supports both GET and HEAD requests (Three.js loaders use HEAD to check availability).
    URL includes .ply extension so Three.js PLYLoader recognizes the format.
    """
    mesh_path = OUTPUT_DIR / job_id / "reference_mesh.ply"
    if not mesh_path.exists():
        raise HTTPException(404, "Reference mesh not available for this job")

    return FileResponse(
        str(mesh_path),
        media_type="application/x-ply",
        filename="reference_mesh.ply"
    )


@app.get("/api/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed"),
    part_id: Optional[str] = Query(None, description="Filter by part ID"),
    limit: int = Query(DEFAULT_PAGINATION_LIMIT, ge=1, le=MAX_PAGINATION_LIMIT, description="Max jobs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List all analysis jobs with optional filtering and pagination.

    Returns paginated list of jobs with total count for building pagination UI.
    Maximum limit is 1000 items per request.
    """
    # Validate and constrain pagination
    limit, offset = validate_pagination(limit, offset)

    # Get jobs and total count
    jobs_list = db.list_jobs(status=status, part_id=part_id, limit=limit, offset=offset)
    total_count = db.count_jobs(status=status, part_id=part_id)

    jobs_data = [
        _json_safe({
            "job_id": j.job_id,
            "status": j.status,
            "part_id": j.part_id,
            "part_name": j.part_name,
            "material": j.material,
            "tolerance": j.tolerance,
            "created_at": j.created_at,
            "completed_at": j.completed_at,
            "progress": j.progress
        })
        for j in jobs_list
    ]

    return _json_safe({
        "jobs": jobs_data,
        "total": total_count,
        "count": len(jobs_data),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(jobs_data) < total_count
    })


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Get full details for a specific job"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    return _json_safe(job.to_dict())


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Track cleanup errors but don't fail the request
    cleanup_errors = []

    # Delete upload files
    job_upload_dir = UPLOAD_DIR / job_id
    if job_upload_dir.exists():
        try:
            shutil.rmtree(job_upload_dir)
            logger.info(f"Deleted upload directory: {job_upload_dir}")
        except Exception as e:
            cleanup_errors.append(f"upload dir: {e}")
            logger.warning(f"Failed to delete upload directory {job_upload_dir}: {e}")

    # Delete output files
    job_output_dir = OUTPUT_DIR / job_id
    if job_output_dir.exists():
        try:
            shutil.rmtree(job_output_dir)
            logger.info(f"Deleted output directory: {job_output_dir}")
        except Exception as e:
            cleanup_errors.append(f"output dir: {e}")
            logger.warning(f"Failed to delete output directory {job_output_dir}: {e}")

    # Delete from database
    try:
        db.delete_job(job_id)
        logger.info(f"Deleted job from database: {job_id}")
    except Exception as e:
        cleanup_errors.append(f"database: {e}")
        logger.error(f"Failed to delete job from database {job_id}: {e}")
        raise HTTPException(500, "Failed to delete job from database")

    # Remove from in-memory progress
    if job_id in job_progress:
        del job_progress[job_id]

    response = {"message": f"Job {job_id} deleted", "job_id": job_id}
    if cleanup_errors:
        response["warnings"] = cleanup_errors
        logger.warning(f"Job {job_id} deleted with cleanup errors: {cleanup_errors}")

    return response


@app.get("/api/jobs/{job_id}/dimensions")
async def get_job_dimensions(job_id: str):
    """
    Get dimension analysis results for a job.

    Returns comparison of expected (XLSX) vs actual (scan) dimensions with:
    - Per-dimension comparison (expected, CAD, scan, deviation, status)
    - Bend-by-bend analysis with pass/fail status
    - Summary statistics (total, passed, failed, pass rate)
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Get result data
    report_path = OUTPUT_DIR / job_id / "report.json"
    if not report_path.exists():
        return {"dimension_analysis": None, "message": "No analysis data available"}

    try:
        with open(report_path, 'r') as f:
            result_data = json.load(f)

        dimension_analysis = result_data.get("dimension_analysis")

        if not dimension_analysis:
            return _json_safe({
                "job_id": job_id,
                "dimension_analysis": None,
                "message": "No dimension analysis available (XLSX file may not have been provided)"
            })

        return _json_safe({
            "job_id": job_id,
            "dimension_analysis": dimension_analysis,
            "summary": dimension_analysis.get("summary", {}),
            "bend_summary": dimension_analysis.get("bend_summary", {}),
            "failed_dimensions": dimension_analysis.get("failed_dimensions", []),
            "worst_deviations": dimension_analysis.get("worst_deviations", [])
        })
    except Exception as e:
        logger.error(f"Failed to load dimension data for {job_id}: {e}")
        return {"dimension_analysis": None, "error": str(e)}


@app.get("/api/jobs/{job_id}/bends")
async def get_job_bends(job_id: str):
    """Get bend detection results for a job"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Get result data (report.json is current, result.json for backward compatibility)
    result_path = OUTPUT_DIR / job_id / "report.json"
    if not result_path.exists():
        legacy_path = OUTPUT_DIR / job_id / "result.json"
        result_path = legacy_path
    if not result_path.exists():
        return {"bends": [], "message": "No bend data available"}

    try:
        with open(result_path, 'r') as f:
            result_data = json.load(f)

        bend_results = result_data.get("bend_results", [])
        bend_detection = result_data.get("bend_detection_result", {})

        return _json_safe({
            "job_id": job_id,
            "bends": bend_results,
            "detection_result": bend_detection,
            "total_bends": len(bend_results)
        })
    except Exception as e:
        logger.error(f"Failed to load bend data for {job_id}: {e}")
        return {"bends": [], "error": str(e)}


@app.get("/api/jobs/{job_id}/correlations")
async def get_job_correlations(job_id: str):
    """Get 2D-3D correlation results for a job"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Get result data (report.json is current, result.json for backward compatibility)
    result_path = OUTPUT_DIR / job_id / "report.json"
    if not result_path.exists():
        legacy_path = OUTPUT_DIR / job_id / "result.json"
        result_path = legacy_path
    if not result_path.exists():
        return {"correlation": None, "message": "No correlation data available"}

    try:
        with open(result_path, 'r') as f:
            result_data = json.load(f)

        correlation = result_data.get("correlation_2d_3d")
        pipeline_status = result_data.get("pipeline_status")

        return _json_safe({
            "job_id": job_id,
            "correlation": correlation,
            "pipeline_status": pipeline_status
        })
    except Exception as e:
        logger.error(f"Failed to load correlation data for {job_id}: {e}")
        return {"correlation": None, "error": str(e)}


@app.get("/api/jobs/{job_id}/enhanced-analysis")
async def get_enhanced_analysis(job_id: str):
    """Get complete enhanced analysis results for a job"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Get result data (report.json is current, result.json for backward compatibility)
    result_path = OUTPUT_DIR / job_id / "report.json"
    if not result_path.exists():
        legacy_path = OUTPUT_DIR / job_id / "result.json"
        result_path = legacy_path
    if not result_path.exists():
        return {"enhanced_analysis": None, "message": "No enhanced analysis available"}

    try:
        with open(result_path, 'r') as f:
            result_data = json.load(f)

        return _json_safe({
            "job_id": job_id,
            "enhanced_analysis": result_data.get("enhanced_analysis"),
            "bend_results": result_data.get("bend_results", []),
            "bend_detection_result": result_data.get("bend_detection_result"),
            "correlation_2d_3d": result_data.get("correlation_2d_3d"),
            "pipeline_status": result_data.get("pipeline_status")
        })
    except Exception as e:
        logger.error(f"Failed to load enhanced analysis for {job_id}: {e}")
        return {"enhanced_analysis": None, "error": str(e)}


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    stats = db.get_stats()

    # Add storage info
    uploads_size = sum(f.stat().st_size for f in UPLOAD_DIR.rglob("*") if f.is_file())
    output_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())

    stats["storage"] = {
        "uploads_mb": round(uploads_size / (1024 * 1024), 2),
        "output_mb": round(output_size / (1024 * 1024), 2),
        "total_mb": round((uploads_size + output_size) / (1024 * 1024), 2)
    }

    return _json_safe(stats)


# =============================================================================
# Batch Processing API
# =============================================================================


class BatchPartInput(BaseModel):
    """Input model for a single part in a batch"""
    part_id: str
    part_name: str = ""


@app.post("/api/batch/analyze", tags=["Batch"])
async def start_batch_analysis(
    background_tasks: BackgroundTasks,
    reference_files: List[UploadFile] = File(..., description="Reference STL files (one per part)"),
    scan_files: List[UploadFile] = File(..., description="Scan files (one per part, matching order)"),
    batch_name: str = Form("Batch Analysis"),
    part_ids: str = Form("", description="Comma-separated part IDs"),
    part_names: str = Form("", description="Comma-separated part names"),
    material: str = Form("Al-5053-H32"),
    tolerance: float = Form(0.1),
    sequential: bool = Form(True, description="Run parts sequentially (False = parallel)")
):
    """
    Start batch QC analysis for multiple parts.

    Upload matched pairs of reference and scan files.
    Files are paired by index (first reference with first scan, etc.).

    Args:
        reference_files: List of reference STL/PLY files
        scan_files: List of scan files (same count as reference_files)
        batch_name: Name for this batch
        part_ids: Comma-separated list of part IDs (or auto-generated)
        part_names: Comma-separated list of part names
        material: Material specification for all parts
        tolerance: Tolerance in mm for all parts
        sequential: If True, process parts one at a time; if False, run in parallel

    Returns:
        batch_id: ID for tracking batch progress
    """
    # Validate inputs
    if len(reference_files) != len(scan_files):
        raise HTTPException(
            400,
            f"Reference count ({len(reference_files)}) must match scan count ({len(scan_files)})"
        )

    if len(reference_files) == 0:
        raise HTTPException(400, "At least one part required")

    if len(reference_files) > 50:
        raise HTTPException(400, "Maximum 50 parts per batch")

    # Parse part IDs and names
    part_id_list = [p.strip() for p in part_ids.split(",") if p.strip()] if part_ids else []
    part_name_list = [p.strip() for p in part_names.split(",") if p.strip()] if part_names else []

    # Generate batch ID
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"

    # Create batch directory
    batch_dir = UPLOAD_DIR / batch_id
    batch_dir.mkdir(exist_ok=True)

    # Save all files and create part entries
    parts: Dict[str, BatchPartResult] = {}

    for i, (ref_file, scan_file) in enumerate(zip(reference_files, scan_files)):
        # Get or generate part ID and name
        part_id = part_id_list[i] if i < len(part_id_list) else f"PART_{i+1:03d}"
        part_name = part_name_list[i] if i < len(part_name_list) else f"Part {i+1}"

        # Validate file types
        ref_ext = Path(ref_file.filename).suffix.lower()
        scan_ext = Path(scan_file.filename).suffix.lower()

        # Reference can be mesh (STL/OBJ/PLY) or CAD (STEP/IGES)
        valid_ref_exts = ['.stl', '.obj', '.ply', '.step', '.stp', '.iges', '.igs']
        if ref_ext not in valid_ref_exts:
            raise HTTPException(400, f"Part {i+1}: Reference must be STL/OBJ/PLY/STEP/IGES, got {ref_ext}")

        if scan_ext not in ['.stl', '.ply', '.obj', '.pcd']:
            raise HTTPException(400, f"Part {i+1}: Scan must be STL/PLY/OBJ/PCD, got {scan_ext}")

        # Validate file sizes
        await validate_file_size(ref_file, MAX_MESH_FILE_SIZE, f"Part {i+1} reference")
        await validate_file_size(scan_file, MAX_MESH_FILE_SIZE, f"Part {i+1} scan")

        # Create job ID for this part
        job_id = f"{batch_id}_{i:03d}"

        # Create part directory
        part_dir = batch_dir / f"part_{i:03d}"
        part_dir.mkdir(exist_ok=True)

        # Save files using async I/O
        ref_path = part_dir / f"reference{ref_ext}"
        scan_path = part_dir / f"scan{scan_ext}"

        ref_content = await ref_file.read()
        async with aiofiles.open(ref_path, "wb") as f:
            await f.write(ref_content)

        scan_content = await scan_file.read()
        async with aiofiles.open(scan_path, "wb") as f:
            await f.write(scan_content)

        # Create database job entry
        db.create_job(
            job_id=job_id,
            part_id=part_id,
            part_name=part_name,
            material=material,
            tolerance=tolerance,
            reference_path=str(ref_path),
            scan_path=str(scan_path)
        )

        # Create part result entry
        parts[job_id] = BatchPartResult(
            part_index=i,
            job_id=job_id,
            part_id=part_id,
            part_name=part_name
        )

        # Create in-memory progress tracker for this job
        job_progress[job_id] = JobProgressState(job_id)

    # Create batch state
    batch_state = BatchJobState(
        batch_id=batch_id,
        name=batch_name,
        material=material,
        tolerance=tolerance,
        total_parts=len(parts),
        parts=parts
    )
    batch_jobs[batch_id] = batch_state

    # Start batch processing in background
    if sequential:
        background_tasks.add_task(run_batch_sequential, batch_id, material, tolerance)
    else:
        background_tasks.add_task(run_batch_parallel, batch_id, material, tolerance)

    return {
        "batch_id": batch_id,
        "name": batch_name,
        "total_parts": len(parts),
        "status": "started",
        "mode": "sequential" if sequential else "parallel"
    }


async def run_batch_sequential(batch_id: str, material: str, tolerance: float):
    """Process batch parts sequentially"""
    batch_state = batch_jobs.get(batch_id)
    if not batch_state:
        return

    batch_state.status = "running"

    for job_id, part_result in batch_state.parts.items():
        batch_state.current_part = part_result.part_id

        try:
            # Get job info from database
            job = db.get_job(job_id)
            if not job:
                part_result.status = "failed"
                part_result.error = "Job not found in database"
                batch_state.failed_parts += 1
                continue

            # Run analysis
            await run_analysis(
                job_id=job_id,
                ref_path=job.reference_path,
                scan_path=job.scan_path,
                part_id=job.part_id,
                part_name=job.part_name,
                material=material,
                tolerance=tolerance,
                drawing_context=""
            )

            # Update part result
            updated_job = db.get_job(job_id)
            if updated_job:
                if updated_job.status == "completed":
                    part_result.status = "completed"
                    part_result.progress = 100

                    # Extract result info
                    if updated_job.result_json:
                        try:
                            result_data = json.loads(updated_job.result_json)
                            part_result.result = result_data.get("release_decision") or result_data.get("overall_result", "UNKNOWN")
                            part_result.quality_score = result_data.get("quality_score")
                        except json.JSONDecodeError:
                            pass

                    batch_state.completed_parts += 1
                else:
                    part_result.status = "failed"
                    part_result.error = updated_job.error or "Analysis failed"
                    batch_state.failed_parts += 1

        except Exception as e:
            logger.error(f"Batch part {job_id} failed: {e}")
            part_result.status = "failed"
            part_result.error = str(e)
            batch_state.failed_parts += 1

        # Update batch progress
        total_processed = batch_state.completed_parts + batch_state.failed_parts
        batch_state.progress = (total_processed / batch_state.total_parts) * 100

    # Batch complete
    batch_state.status = "completed"
    batch_state.progress = 100
    batch_state.current_part = ""
    batch_state.completed_at = datetime.now().isoformat()


async def run_batch_parallel(batch_id: str, material: str, tolerance: float):
    """Process batch parts in parallel (limited concurrency)"""
    batch_state = batch_jobs.get(batch_id)
    if not batch_state:
        return

    batch_state.status = "running"

    # Create tasks for all parts
    tasks = []
    for job_id, part_result in batch_state.parts.items():
        job = db.get_job(job_id)
        if job:
            task = asyncio.create_task(
                run_single_part(batch_state, job_id, part_result, job, material, tolerance)
            )
            tasks.append(task)

    # Wait for all tasks with limited concurrency
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent analyses

    async def limited_task(task):
        async with semaphore:
            return await task

    await asyncio.gather(*[limited_task(t) for t in tasks])

    # Batch complete
    batch_state.status = "completed"
    batch_state.progress = 100
    batch_state.current_part = ""
    batch_state.completed_at = datetime.now().isoformat()


async def run_single_part(
    batch_state: BatchJobState,
    job_id: str,
    part_result: BatchPartResult,
    job: JobRecord,
    material: str,
    tolerance: float
):
    """Run analysis for a single part in a batch"""
    try:
        batch_state.current_part = part_result.part_id

        await run_analysis(
            job_id=job_id,
            ref_path=job.reference_path,
            scan_path=job.scan_path,
            part_id=job.part_id,
            part_name=job.part_name,
            material=material,
            tolerance=tolerance,
            drawing_context=""
        )

        # Update part result
        updated_job = db.get_job(job_id)
        if updated_job and updated_job.status == "completed":
            part_result.status = "completed"
            part_result.progress = 100

            if updated_job.result_json:
                try:
                    result_data = json.loads(updated_job.result_json)
                    part_result.result = result_data.get("release_decision") or result_data.get("overall_result", "UNKNOWN")
                    part_result.quality_score = result_data.get("quality_score")
                except json.JSONDecodeError:
                    pass

            batch_state.completed_parts += 1
        else:
            part_result.status = "failed"
            part_result.error = updated_job.error if updated_job else "Unknown error"
            batch_state.failed_parts += 1

    except Exception as e:
        logger.error(f"Batch part {job_id} failed: {e}")
        part_result.status = "failed"
        part_result.error = str(e)
        batch_state.failed_parts += 1

    # Update batch progress
    total_processed = batch_state.completed_parts + batch_state.failed_parts
    batch_state.progress = (total_processed / batch_state.total_parts) * 100


@app.get("/api/batch/{batch_id}", tags=["Batch"])
async def get_batch_status(batch_id: str):
    """
    Get current status of a batch job.

    Returns batch-level progress and status of each part.
    """
    batch_state = batch_jobs.get(batch_id)
    if not batch_state:
        raise HTTPException(404, "Batch not found")

    return {
        "batch_id": batch_id,
        "name": batch_state.name,
        "status": batch_state.status,
        "progress": round(batch_state.progress, 1),
        "total_parts": batch_state.total_parts,
        "completed_parts": batch_state.completed_parts,
        "failed_parts": batch_state.failed_parts,
        "current_part": batch_state.current_part,
        "created_at": batch_state.created_at,
        "completed_at": batch_state.completed_at,
        "parts": [part.to_dict() for part in batch_state.parts.values()]
    }


@app.get("/api/batch/{batch_id}/summary", tags=["Batch"])
async def get_batch_summary(batch_id: str):
    """
    Get summary of completed batch analysis.

    Provides aggregate statistics and pass/fail breakdown.
    """
    batch_state = batch_jobs.get(batch_id)
    if not batch_state:
        raise HTTPException(404, "Batch not found")

    if batch_state.status != "completed":
        raise HTTPException(400, "Batch not yet completed")

    # Calculate statistics
    passed = 0
    failed_qc = 0
    scores = []

    for part in batch_state.parts.values():
        if part.status == "completed":
            if part.result == "PASS":
                passed += 1
            elif part.result in ["FAIL", "REJECT"]:
                failed_qc += 1
            if part.quality_score is not None:
                scores.append(part.quality_score)

    avg_score = sum(scores) / len(scores) if scores else None
    min_score = min(scores) if scores else None
    max_score = max(scores) if scores else None

    return {
        "batch_id": batch_id,
        "name": batch_state.name,
        "material": batch_state.material,
        "tolerance": batch_state.tolerance,
        "summary": {
            "total_parts": batch_state.total_parts,
            "completed": batch_state.completed_parts,
            "processing_errors": batch_state.failed_parts,
            "qc_passed": passed,
            "qc_failed": failed_qc,
            "pass_rate": round(passed / batch_state.completed_parts * 100, 1) if batch_state.completed_parts > 0 else 0
        },
        "quality_scores": {
            "average": round(avg_score, 2) if avg_score else None,
            "min": round(min_score, 2) if min_score else None,
            "max": round(max_score, 2) if max_score else None
        },
        "created_at": batch_state.created_at,
        "completed_at": batch_state.completed_at,
        "duration_seconds": (
            (datetime.fromisoformat(batch_state.completed_at) -
             datetime.fromisoformat(batch_state.created_at)).total_seconds()
            if batch_state.completed_at else None
        )
    }


@app.get("/api/batch/{batch_id}/stream", tags=["Batch"])
async def stream_batch_progress(batch_id: str):
    """
    Stream batch progress updates via Server-Sent Events.

    Provides real-time updates as parts are processed.
    """
    batch_state = batch_jobs.get(batch_id)
    if not batch_state:
        raise HTTPException(404, "Batch not found")

    async def generate():
        last_progress = -1
        while batch_state.status in ["pending", "running"]:
            if batch_state.progress != last_progress:
                data = {
                    "status": batch_state.status,
                    "progress": round(batch_state.progress, 1),
                    "completed_parts": batch_state.completed_parts,
                    "failed_parts": batch_state.failed_parts,
                    "current_part": batch_state.current_part
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = batch_state.progress
            await asyncio.sleep(0.5)

        # Final update
        data = {
            "status": batch_state.status,
            "progress": 100,
            "completed_parts": batch_state.completed_parts,
            "failed_parts": batch_state.failed_parts,
            "current_part": ""
        }
        yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/batch", tags=["Batch"])
async def list_batches(
    status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed"),
    limit: int = Query(DEFAULT_PAGINATION_LIMIT, ge=1, le=MAX_PAGINATION_LIMIT, description="Max batches to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List all batch jobs with optional filtering and pagination.

    Returns paginated list of batch jobs with total count.
    """
    # Validate pagination
    limit, offset = validate_pagination(limit, offset)

    # Filter batches
    all_batches = list(batch_jobs.items())
    if status:
        all_batches = [(bid, bs) for bid, bs in all_batches if bs.status == status]

    total_count = len(all_batches)

    # Apply pagination
    paginated = all_batches[offset:offset + limit]

    batches_data = [
        {
            "batch_id": batch_id,
            "name": batch_state.name,
            "status": batch_state.status,
            "progress": round(batch_state.progress, 1),
            "total_parts": batch_state.total_parts,
            "completed_parts": batch_state.completed_parts,
            "failed_parts": batch_state.failed_parts,
            "created_at": batch_state.created_at,
            "completed_at": batch_state.completed_at
        }
        for batch_id, batch_state in paginated
    ]

    return {
        "batches": batches_data,
        "total": total_count,
        "count": len(batches_data),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(batches_data) < total_count
    }


@app.delete("/api/batch/{batch_id}", tags=["Batch"])
async def delete_batch(batch_id: str):
    """
    Delete a batch job and all associated data.

    This will delete all individual job files and database entries.
    """
    batch_state = batch_jobs.get(batch_id)
    if not batch_state:
        raise HTTPException(404, "Batch not found")

    # Don't allow deleting running batches
    if batch_state.status == "running":
        raise HTTPException(400, "Cannot delete running batch")

    # Delete individual jobs
    for job_id in batch_state.parts.keys():
        job = db.get_job(job_id)
        if job:
            # Delete output directory
            job_output_dir = OUTPUT_DIR / job_id
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)

            # Delete from database
            db.delete_job(job_id)

            # Remove from progress tracking
            if job_id in job_progress:
                del job_progress[job_id]

    # Delete batch upload directory
    batch_dir = UPLOAD_DIR / batch_id
    if batch_dir.exists():
        shutil.rmtree(batch_dir)

    # Remove batch from memory
    del batch_jobs[batch_id]

    return {"message": f"Batch {batch_id} deleted", "batch_id": batch_id}


# =============================================================================
# GD&T (Geometric Dimensioning & Tolerancing) API
# =============================================================================

class GDTPointsRequest(BaseModel):
    """Request model for GD&T calculations with point data"""
    points: List[List[float]]  # Nx3 array of points
    tolerance: float  # Tolerance value in mm

class GDTFlatnessRequest(GDTPointsRequest):
    pass

class GDTCylindricityRequest(GDTPointsRequest):
    axis_hint: Optional[List[float]] = None  # Optional axis direction [x, y, z]

class GDTCircularityRequest(GDTPointsRequest):
    plane_normal: Optional[List[float]] = None  # Optional plane normal [x, y, z]

class GDTPositionRequest(GDTPointsRequest):
    nominal_position: List[float]  # [x, y, z] nominal position
    mmc: bool = False  # Apply MMC bonus
    feature_size: Optional[float] = None  # Actual feature size (for MMC)
    feature_mmc: Optional[float] = None  # MMC feature size

class GDTParallelismRequest(GDTPointsRequest):
    datum_normal: List[float]  # [x, y, z] datum plane normal

class GDTPerpendicularityRequest(GDTPointsRequest):
    datum_normal: List[float]  # [x, y, z] datum plane normal


# Global GD&T engine instance
_gdt_engine: Optional[GDTEngine] = None

def get_gdt_engine() -> GDTEngine:
    """Get or create GD&T engine instance"""
    global _gdt_engine
    if _gdt_engine is None:
        _gdt_engine = create_gdt_engine()
    return _gdt_engine


@app.post("/api/gdt/flatness", tags=["GD&T"])
async def calculate_flatness(request: GDTFlatnessRequest):
    """
    Calculate flatness GD&T value.

    Flatness is the zone between two parallel planes containing all surface points.
    Per ASME Y14.5: a form tolerance for surface planarity.
    """
    try:
        engine = get_gdt_engine()
        points = np.array(request.points)

        if len(points) < 3:
            raise HTTPException(400, "Minimum 3 points required for flatness calculation")

        result = engine.calculate_flatness(points, request.tolerance)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Flatness calculation failed: {e}")
        raise HTTPException(500, f"Calculation failed: {str(e)}")


@app.post("/api/gdt/cylindricity", tags=["GD&T"])
async def calculate_cylindricity(request: GDTCylindricityRequest):
    """
    Calculate cylindricity GD&T value.

    Cylindricity is the zone between two coaxial cylinders.
    Per ASME Y14.5: a form tolerance for cylindrical surfaces.
    """
    try:
        engine = get_gdt_engine()
        points = np.array(request.points)
        axis_hint = np.array(request.axis_hint) if request.axis_hint else None

        if len(points) < 6:
            raise HTTPException(400, "Minimum 6 points required for cylindricity calculation")

        result = engine.calculate_cylindricity(points, request.tolerance, axis_hint)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Cylindricity calculation failed: {e}")
        raise HTTPException(500, f"Calculation failed: {str(e)}")


@app.post("/api/gdt/circularity", tags=["GD&T"])
async def calculate_circularity(request: GDTCircularityRequest):
    """
    Calculate circularity (roundness) GD&T value.

    Circularity is the zone between two concentric circles in a plane.
    Per ASME Y14.5: a form tolerance for circular cross-sections.
    """
    try:
        engine = get_gdt_engine()
        points = np.array(request.points)
        plane_normal = np.array(request.plane_normal) if request.plane_normal else None

        if len(points) < 4:
            raise HTTPException(400, "Minimum 4 points required for circularity calculation")

        result = engine.calculate_circularity(points, request.tolerance, plane_normal)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Circularity calculation failed: {e}")
        raise HTTPException(500, f"Calculation failed: {str(e)}")


@app.post("/api/gdt/position", tags=["GD&T"])
async def calculate_position(request: GDTPositionRequest):
    """
    Calculate position GD&T value.

    Position defines the location of a feature relative to datums.
    Supports MMC (Maximum Material Condition) bonus tolerance.
    Per ASME Y14.5: a location tolerance.
    """
    try:
        engine = get_gdt_engine()
        points = np.array(request.points)
        nominal = tuple(request.nominal_position)

        if len(points) < 3:
            raise HTTPException(400, "Minimum 3 points required for position calculation")

        result = engine.calculate_position(
            points,
            nominal,
            request.tolerance,
            mmc=request.mmc,
            feature_size=request.feature_size,
            feature_mmc=request.feature_mmc
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Position calculation failed: {e}")
        raise HTTPException(500, f"Calculation failed: {str(e)}")


@app.post("/api/gdt/parallelism", tags=["GD&T"])
async def calculate_parallelism(request: GDTParallelismRequest):
    """
    Calculate parallelism GD&T value.

    Parallelism is the condition of a surface being equidistant from a datum plane.
    Per ASME Y14.5: an orientation tolerance.
    """
    try:
        engine = get_gdt_engine()
        points = np.array(request.points)
        datum_normal = np.array(request.datum_normal)

        if len(points) < 3:
            raise HTTPException(400, "Minimum 3 points required for parallelism calculation")

        result = engine.calculate_parallelism(points, datum_normal, request.tolerance)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Parallelism calculation failed: {e}")
        raise HTTPException(500, f"Calculation failed: {str(e)}")


@app.post("/api/gdt/perpendicularity", tags=["GD&T"])
async def calculate_perpendicularity(request: GDTPerpendicularityRequest):
    """
    Calculate perpendicularity GD&T value.

    Perpendicularity is the condition of a surface being at 90° to a datum.
    Per ASME Y14.5: an orientation tolerance.
    """
    try:
        engine = get_gdt_engine()
        points = np.array(request.points)
        datum_normal = np.array(request.datum_normal)

        if len(points) < 3:
            raise HTTPException(400, "Minimum 3 points required for perpendicularity calculation")

        result = engine.calculate_perpendicularity(points, datum_normal, request.tolerance)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Perpendicularity calculation failed: {e}")
        raise HTTPException(500, f"Calculation failed: {str(e)}")


@app.get("/api/gdt/types", tags=["GD&T"])
async def get_gdt_types():
    """Get list of supported GD&T types"""
    return {
        "supported_types": [
            {
                "type": "flatness",
                "category": "form",
                "description": "Surface planarity tolerance",
                "endpoint": "/api/gdt/flatness"
            },
            {
                "type": "cylindricity",
                "category": "form",
                "description": "Cylindrical surface tolerance",
                "endpoint": "/api/gdt/cylindricity"
            },
            {
                "type": "circularity",
                "category": "form",
                "description": "Circular cross-section tolerance",
                "endpoint": "/api/gdt/circularity"
            },
            {
                "type": "position",
                "category": "location",
                "description": "Feature location tolerance (supports MMC)",
                "endpoint": "/api/gdt/position"
            },
            {
                "type": "parallelism",
                "category": "orientation",
                "description": "Surface parallelism to datum",
                "endpoint": "/api/gdt/parallelism"
            },
            {
                "type": "perpendicularity",
                "category": "orientation",
                "description": "Surface perpendicularity to datum",
                "endpoint": "/api/gdt/perpendicularity"
            }
        ],
        "reference": "ASME Y14.5M-2018"
    }


# =============================================================================
# SPC (Statistical Process Control) API
# =============================================================================

class SPCCapabilityRequest(BaseModel):
    """Request model for SPC capability calculations"""
    data: List[float]  # Measurement data
    usl: float  # Upper Specification Limit
    lsl: float  # Lower Specification Limit
    target: Optional[float] = None  # Target value
    subgroup_size: int = 1  # Rational subgroup size

class SPCControlChartRequest(BaseModel):
    """Request model for control chart generation"""
    data: List[float]  # Measurement data
    subgroup_size: int = 5  # Rational subgroup size
    labels: Optional[List[str]] = None  # Optional subgroup labels

class SPCHistogramRequest(BaseModel):
    """Request model for histogram generation"""
    data: List[float]  # Measurement data
    num_bins: int = 20  # Number of histogram bins
    lsl: Optional[float] = None
    usl: Optional[float] = None


# Global SPC engine instance
_spc_engine: Optional[SPCEngine] = None

def get_spc_engine() -> SPCEngine:
    """Get or create SPC engine instance"""
    global _spc_engine
    if _spc_engine is None:
        _spc_engine = create_spc_engine()
    return _spc_engine


@app.post("/api/spc/capability", tags=["SPC"])
async def calculate_capability(request: SPCCapabilityRequest):
    """
    Calculate process capability indices (Cp, Cpk, Pp, Ppk).

    Returns capability analysis including:
    - Cp/Cpk (potential/actual process capability)
    - Pp/Ppk (process performance)
    - Capability rating (excellent/good/capable/marginal/poor)
    - Estimated PPM defect rates
    """
    try:
        engine = get_spc_engine()
        data = np.array(request.data)

        if len(data) < 2:
            raise HTTPException(400, "Minimum 2 data points required")

        result = engine.calculate_capability(
            data,
            usl=request.usl,
            lsl=request.lsl,
            target=request.target,
            subgroup_size=request.subgroup_size
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Capability calculation failed: {e}")
        raise HTTPException(500, f"Calculation failed: {str(e)}")


@app.post("/api/spc/control-charts", tags=["SPC"])
async def generate_control_charts(request: SPCControlChartRequest):
    """
    Generate control chart data (X-bar/R or Individuals/MR).

    For subgroup_size > 1: Returns X-bar and Range charts
    For subgroup_size = 1: Returns Individuals and Moving Range charts

    Response includes control limits (UCL, LCL) and out-of-control points.
    """
    try:
        engine = get_spc_engine()
        data = np.array(request.data)

        if len(data) < 2:
            raise HTTPException(400, "Minimum 2 data points required")

        charts = engine.generate_control_charts(
            data,
            subgroup_size=request.subgroup_size,
            labels=request.labels
        )

        return {
            chart_name: chart.to_dict()
            for chart_name, chart in charts.items()
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Control chart generation failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.post("/api/spc/histogram", tags=["SPC"])
async def generate_histogram(request: SPCHistogramRequest):
    """
    Generate histogram data with normal distribution fit.

    Returns bin counts, frequencies, and fitted normal curve for overlay.
    """
    try:
        engine = get_spc_engine()
        data = np.array(request.data)

        if len(data) < 2:
            raise HTTPException(400, "Minimum 2 data points required")

        histogram = engine.generate_histogram(
            data,
            num_bins=request.num_bins,
            lsl=request.lsl,
            usl=request.usl
        )
        return histogram.to_dict()
    except Exception as e:
        logger.error(f"Histogram generation failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.post("/api/spc/analyze", tags=["SPC"])
async def full_spc_analysis(request: SPCCapabilityRequest):
    """
    Perform full SPC analysis including capability, control charts, and histogram.

    This is a convenience endpoint that combines all SPC analyses.
    """
    try:
        engine = get_spc_engine()
        data = np.array(request.data)

        if len(data) < 2:
            raise HTTPException(400, "Minimum 2 data points required")

        # Calculate capability
        capability = engine.calculate_capability(
            data,
            usl=request.usl,
            lsl=request.lsl,
            target=request.target,
            subgroup_size=request.subgroup_size
        )

        # Generate control charts
        charts = engine.generate_control_charts(
            data,
            subgroup_size=request.subgroup_size
        )

        # Generate histogram
        histogram = engine.generate_histogram(data, lsl=request.lsl, usl=request.usl)

        # Analyze stability
        stability = engine.analyze_stability(charts)

        return {
            "capability": capability.to_dict(),
            "control_charts": {
                name: chart.to_dict() for name, chart in charts.items()
            },
            "histogram": histogram.to_dict(),
            "stability": stability,
            "summary": {
                "cpk": round(capability.cpk, 3),
                "rating": capability.rating.value,
                "is_stable": stability["is_stable"],
                "sample_size": capability.sample_size,
                "ppm_defects": round(capability.ppm_total, 1)
            }
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"SPC analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# =============================================================================
# Progressive Bend Inspection API
# =============================================================================

from bend_inspection_pipeline import (
    load_bend_runtime_config,
    load_expected_bend_overrides,
    load_cad_geometry,
    extract_cad_bend_specs,
)


class BendInspectionRequest(BaseModel):
    """Request model for bend inspection"""
    part_id: str = Field(..., description="Part identifier")
    part_name: str = Field(default="Unnamed Part", description="Part name")
    default_tolerance_angle: float = Field(default=1.0, description="Default angle tolerance in degrees")
    default_tolerance_radius: float = Field(default=0.5, description="Default radius tolerance in mm")


class CADBendExtractionRequest(BaseModel):
    """Request for extracting bends from CAD only"""
    part_id: str = Field(..., description="Part identifier")


# In-memory storage for bend inspection results
bend_inspection_results: Dict[str, Dict[str, Any]] = {}


def _resolve_bend_artifact_urls(artifacts: Optional[Dict[str, Any]], job_id: str) -> Dict[str, Any]:
    if not isinstance(artifacts, dict):
        return {}
    resolved: Dict[str, Any] = {}
    for key, value in artifacts.items():
        if isinstance(value, str):
            resolved[key] = value.replace("{job_id}", job_id)
        elif isinstance(value, list):
            resolved[key] = [
                item.replace("{job_id}", job_id) if isinstance(item, str) else item
                for item in value
            ]
        else:
            resolved[key] = value
    return resolved


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object from disk, returning None on any parse/read failure."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    """Parse ISO datetime strings safely for sorting and fallback timestamps."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _build_persisted_bend_job(job_id: str, include_report: bool = False) -> Optional[Dict[str, Any]]:
    """Load a bend-inspection job from output storage."""
    output_dir = OUTPUT_DIR / job_id
    report_path = output_dir / "bend_inspection_report.json"
    if not report_path.exists():
        return None

    report = _safe_read_json(report_path) or {}
    meta = _safe_read_json(output_dir / "bend_inspection_meta.json") or {}

    report_part_id = str(report.get("part_id") or "").strip()
    part_id = str(meta.get("part_id") or report_part_id or job_id).strip()
    part_name = str(meta.get("part_name") or part_id).strip()

    file_time = datetime.fromtimestamp(report_path.stat().st_mtime).isoformat()
    created_at = (
        meta.get("created_at")
        or meta.get("started_at")
        or file_time
    )
    completed_at = meta.get("completed_at") or file_time
    processing_time_ms = meta.get("processing_time_ms")
    if processing_time_ms is None and isinstance(report.get("processing_time_ms"), (int, float)):
        processing_time_ms = report.get("processing_time_ms")

    payload: Dict[str, Any] = {
        "job_id": job_id,
        "status": str(meta.get("status") or "completed"),
        "progress": int(meta.get("progress") or 100),
        "part_id": part_id,
        "part_name": part_name,
        "created_at": created_at,
        "completed_at": completed_at,
        "processing_time_ms": processing_time_ms,
    }
    if isinstance(meta.get("seed_case"), dict):
        payload["seed_case"] = meta.get("seed_case")
    artifacts = _resolve_bend_artifact_urls(meta.get("artifacts"), job_id)
    ref_mesh = output_dir / "reference_mesh.ply"
    if ref_mesh.exists() and "reference_mesh_url" not in artifacts:
        artifacts["reference_mesh_url"] = f"/api/reference-mesh/{job_id}.ply"
    if artifacts:
        payload["artifacts"] = artifacts
    if include_report:
        payload["report"] = report
    return payload


def _list_persisted_bend_jobs(
    status: Optional[str] = None,
    include_report: bool = False,
) -> List[Dict[str, Any]]:
    """Enumerate persisted bend-inspection jobs from output directories."""
    jobs: List[Dict[str, Any]] = []
    for report_path in OUTPUT_DIR.glob("bend_*/bend_inspection_report.json"):
        job_id = report_path.parent.name
        job = _build_persisted_bend_job(job_id, include_report=include_report)
        if not job:
            continue
        if status and job.get("status") != status:
            continue
        jobs.append(job)
    return jobs


@app.post("/api/bend-inspection/analyze", tags=["Bend Inspection"])
async def run_bend_inspection(
    background_tasks: BackgroundTasks,
    cad_file: UploadFile = File(..., description="CAD reference file (PLY, STL, STEP)"),
    scan_file: UploadFile = File(..., description="Scan point cloud file (PLY, STL, PCD)"),
    part_id: str = Form("PART_001"),
    part_name: str = Form("Unnamed Part"),
    default_tolerance_angle: float = Form(1.0),
    default_tolerance_radius: float = Form(0.5),
):
    """
    Run progressive bend inspection.

    This endpoint compares a partial or complete scan against CAD bend specifications.
    It detects which bends have been made and verifies they're within tolerance.

    Use this for:
    - In-process inspection (after 2 of 11 bends, etc.)
    - Final inspection (all bends complete)
    - Progressive QC in production cells

    Returns:
        job_id: ID for tracking/retrieving results
    """
    # Validate file types
    cad_ext = Path(cad_file.filename).suffix.lower()
    scan_ext = Path(scan_file.filename).suffix.lower()

    valid_cad_exts = ['.stl', '.ply', '.obj', '.step', '.stp']
    valid_scan_exts = ['.stl', '.ply', '.obj', '.pcd']

    if cad_ext not in valid_cad_exts:
        raise HTTPException(400, f"CAD file must be STL/PLY/OBJ/STEP, got {cad_ext}")
    if scan_ext not in valid_scan_exts:
        raise HTTPException(400, f"Scan file must be STL/PLY/OBJ/PCD, got {scan_ext}")

    # Validate file sizes
    await validate_file_size(cad_file, MAX_MESH_FILE_SIZE, "CAD")
    await validate_file_size(scan_file, MAX_MESH_FILE_SIZE, "Scan")

    # Create job ID
    job_id = f"bend_{uuid.uuid4().hex[:8]}"

    # Create job directory
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Save files
    cad_path = job_dir / f"cad{cad_ext}"
    scan_path = job_dir / f"scan{scan_ext}"

    cad_content = await cad_file.read()
    async with aiofiles.open(cad_path, "wb") as f:
        await f.write(cad_content)

    scan_content = await scan_file.read()
    async with aiofiles.open(scan_path, "wb") as f:
        await f.write(scan_content)

    # Initialize result storage
    bend_inspection_results[job_id] = {
        "status": "running",
        "progress": 0,
        "part_id": part_id,
        "part_name": part_name,
        "created_at": datetime.now().isoformat(),
    }

    # Run inspection in background
    background_tasks.add_task(
        run_bend_inspection_task,
        job_id,
        str(cad_path),
        str(scan_path),
        part_id,
        part_name,
        default_tolerance_angle,
        default_tolerance_radius,
    )

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Bend inspection started"
    }


async def run_bend_inspection_task(
    job_id: str,
    cad_path: str,
    scan_path: str,
    part_id: str,
    part_name: str,
    tolerance_angle: float,
    tolerance_radius: float,
):
    """Background task for bend inspection."""
    import time

    start_time = time.time()
    result_store = bend_inspection_results.get(job_id)
    if not result_store:
        return

    try:
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        def progress_callback(update: ProgressUpdate):
            try:
                result_store["progress"] = int(max(result_store.get("progress", 0), update.progress))
                result_store["stage"] = update.message or update.stage
            except Exception:
                pass

        result_store["progress"] = 2
        result_store["stage"] = "Launching isolated bend worker..."

        worker_result = await asyncio.to_thread(
            _run_bend_worker_subprocess,
            job_id,
            cad_path,
            scan_path,
            part_id,
            part_name,
            tolerance_angle,
            tolerance_radius,
            output_dir,
            progress_callback,
        )
        if not worker_result.get("ok"):
            raise RuntimeError(worker_result.get("error") or "Bend worker failed")

        report_path = Path(str(worker_result["report_path"]))
        table_path = Path(str(worker_result["table_path"]))
        if not report_path.exists():
            raise FileNotFoundError(f"Bend worker produced no report file: {report_path}")

        with report_path.open("r", encoding="utf-8") as f:
            report_dict = _json_safe(json.load(f))
        table_text = table_path.read_text(encoding="utf-8") if table_path.exists() else ""
        artifacts_payload = _resolve_bend_artifact_urls(worker_result.get("artifacts"), job_id)
        processing_time_ms = float(worker_result.get("processing_time_ms") or ((time.time() - start_time) * 1000.0))
        pipeline_details = _json_safe(worker_result.get("pipeline_details"))
        runtime_config = _json_safe(worker_result.get("runtime_config"))

        completed_at = datetime.now().isoformat()
        metadata = {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "part_id": part_id,
            "part_name": part_name,
            "created_at": result_store.get("created_at"),
            "completed_at": completed_at,
            "processing_time_ms": processing_time_ms,
            "cad_path": cad_path,
            "scan_path": scan_path,
            "artifacts": artifacts_payload,
        }
        meta_path = output_dir / "bend_inspection_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update result store
        result_store["status"] = "completed"
        result_store["progress"] = 100
        result_store["stage"] = "Complete"
        result_store["report"] = report_dict
        result_store["table"] = table_text
        result_store["completed_at"] = completed_at
        result_store["processing_time_ms"] = processing_time_ms
        result_store["pipeline_details"] = pipeline_details
        result_store["runtime_config"] = runtime_config
        if artifacts_payload:
            result_store["artifacts"] = artifacts_payload

        logger.info(f"Bend inspection completed for job {job_id}: "
                   f"{report_dict.get('summary', {}).get('completed_bends', report_dict.get('summary', {}).get('detected', '?'))}/"
                   f"{report_dict.get('summary', {}).get('expected_bends', report_dict.get('summary', {}).get('total_bends', '?'))} bends detected")

    except Exception as e:
        import traceback
        logger.error(f"Bend inspection failed for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        result_store["status"] = "failed"
        result_store["error"] = str(e)


@app.get("/api/bend-inspection/jobs", tags=["Bend Inspection"])
async def list_bend_inspection_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
):
    """List recent bend inspection jobs."""
    jobs_by_id: Dict[str, Dict[str, Any]] = {}

    # Live jobs from in-memory store (running/completed in current process).
    for job_id, result in bend_inspection_results.items():
        if status and result.get("status") != status:
            continue
        jobs_by_id[job_id] = {
            "job_id": job_id,
            "part_id": result.get("part_id"),
            "part_name": result.get("part_name"),
            "status": result.get("status"),
            "progress": result.get("progress"),
            "created_at": result.get("created_at"),
            "completed_at": result.get("completed_at"),
            "processing_time_ms": result.get("processing_time_ms"),
            "report": result.get("report"),
            "artifacts": result.get("artifacts"),
        }

    # Persisted jobs from disk (so history survives process restart).
    persisted = _list_persisted_bend_jobs(status=status, include_report=True)
    for job in persisted:
        current = jobs_by_id.get(job["job_id"])
        # Prefer live running state when available.
        if current and current.get("status") == "running":
            continue
        if current and current.get("status") == "completed":
            # Keep live payload but backfill missing report if needed.
            if not isinstance(current.get("report"), dict) and isinstance(job.get("report"), dict):
                current["report"] = job["report"]
            continue
        jobs_by_id[job["job_id"]] = job

    jobs = list(jobs_by_id.values())

    def _sort_key(job: Dict[str, Any]) -> datetime:
        created = _parse_iso_datetime(job.get("created_at"))
        completed = _parse_iso_datetime(job.get("completed_at"))
        return created or completed or datetime.min

    # Sort by timestamp descending.
    jobs.sort(key=_sort_key, reverse=True)

    return {
        "jobs": jobs[:limit],
        "total": len(jobs)
    }


@app.get("/api/bend-inspection/{job_id}", tags=["Bend Inspection"])
async def get_bend_inspection_result(job_id: str):
    """
    Get bend inspection results.

    Returns the progressive bend inspection report including:
    - Which bends have been detected
    - Angle measurements and deviations
    - Pass/fail status for each bend
    - Progress summary (e.g., "3 of 11 bends complete")
    """
    result = bend_inspection_results.get(job_id)
    if not result:
        # Fall back to persisted job on disk.
        persisted = _build_persisted_bend_job(job_id, include_report=True)
        if persisted:
            table_path = OUTPUT_DIR / job_id / "bend_inspection_table.txt"
            if table_path.exists():
                with open(table_path, "r") as f:
                    persisted["table"] = f.read()
            return persisted
        raise HTTPException(404, "Bend inspection job not found")

    return {
        "job_id": job_id,
        "status": result.get("status"),
        "progress": result.get("progress"),
        "stage": result.get("stage"),
        "part_id": result.get("part_id"),
        "part_name": result.get("part_name"),
        "created_at": result.get("created_at"),
        "completed_at": result.get("completed_at"),
        "report": result.get("report"),
        "table": result.get("table"),
        "pipeline_details": result.get("pipeline_details"),
        "runtime_config": result.get("runtime_config"),
        "artifacts": result.get("artifacts"),
        "error": result.get("error"),
        "processing_time_ms": result.get("processing_time_ms"),
    }


@app.get("/api/bend-inspection/{job_id}/pdf", tags=["Bend Inspection"])
async def download_bend_inspection_pdf(job_id: str):
    pdf_path = OUTPUT_DIR / job_id / "bend_inspection_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Bend inspection PDF not available")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"Bend_Inspection_{job_id}.pdf",
    )


@app.get("/api/bend-inspection/{job_id}/overlay/manifest.json", tags=["Bend Inspection"])
async def get_bend_overlay_manifest(job_id: str):
    manifest_path = OUTPUT_DIR / job_id / "bend_overlay_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Bend overlay manifest not available")
    return FileResponse(
        str(manifest_path),
        media_type="application/json",
        filename="bend_overlay_manifest.json",
    )


@app.get("/api/bend-inspection/{job_id}/overlay/overview.png", tags=["Bend Inspection"])
async def get_bend_overlay_overview(job_id: str):
    overview_path = OUTPUT_DIR / job_id / "bend_overlay_overview.png"
    if not overview_path.exists():
        raise HTTPException(404, "Bend overlay overview not available")
    return FileResponse(
        str(overview_path),
        media_type="image/png",
        filename="bend_overlay_overview.png",
    )


@app.get("/api/bend-inspection/{job_id}/overlay/issues/{filename}", tags=["Bend Inspection"])
async def get_bend_overlay_issue(job_id: str, filename: str):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(400, "Invalid bend issue filename")
    issue_path = OUTPUT_DIR / job_id / safe_name
    if not issue_path.exists():
        raise HTTPException(404, "Bend issue image not available")
    return FileResponse(
        str(issue_path),
        media_type="image/png",
        filename=safe_name,
    )


@app.get("/api/bend-inspection/{job_id}/table", tags=["Bend Inspection"])
async def get_bend_inspection_table(job_id: str):
    """
    Get bend inspection results as ASCII table.

    Returns a formatted table suitable for display in terminal or simple UI.
    """
    result = bend_inspection_results.get(job_id)

    if result and result.get("table"):
        return {"table": result["table"]}

    # Try disk
    table_path = OUTPUT_DIR / job_id / "bend_inspection_table.txt"
    if table_path.exists():
        with open(table_path, "r") as f:
            return {"table": f.read()}

    raise HTTPException(404, "Bend inspection results not found")


@app.post("/api/bend-inspection/extract-cad", tags=["Bend Inspection"])
async def extract_cad_bends(
    cad_file: UploadFile = File(..., description="CAD file to extract bends from"),
    part_id: str = Form("PART_001"),
    default_tolerance_angle: float = Form(1.0),
    default_tolerance_radius: float = Form(0.5),
):
    """
    Extract bend specifications from a CAD file only.

    Use this to preview what bends the system detects in your CAD model
    before running a full inspection.
    """
    # Validate file type
    cad_ext = Path(cad_file.filename).suffix.lower()
    valid_exts = ['.stl', '.ply', '.obj', '.step', '.stp']

    if cad_ext not in valid_exts:
        raise HTTPException(400, f"CAD file must be STL/PLY/OBJ/STEP, got {cad_ext}")

    # Validate file size
    await validate_file_size(cad_file, MAX_MESH_FILE_SIZE, "CAD")

    # Save to temp location
    temp_dir = UPLOAD_DIR / f"temp_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(exist_ok=True)

    try:
        runtime_cfg = load_bend_runtime_config()
        cad_path = temp_dir / f"cad{cad_ext}"
        cad_content = await cad_file.read()
        async with aiofiles.open(cad_path, "wb") as f:
            await f.write(cad_content)

        cad_vertices, cad_triangles, cad_source = load_cad_geometry(
            str(cad_path),
            cad_import_deflection=runtime_cfg.cad_import_deflection,
        )
        bends = extract_cad_bend_specs(
            cad_vertices=cad_vertices,
            cad_triangles=cad_triangles,
            tolerance_angle=default_tolerance_angle,
            tolerance_radius=default_tolerance_radius,
            cad_extractor_kwargs=runtime_cfg.cad_extractor_kwargs,
            cad_detector_kwargs=runtime_cfg.cad_detector_kwargs,
            cad_detect_call_kwargs=runtime_cfg.cad_detect_call_kwargs,
        )
        expected_overrides = load_expected_bend_overrides()
        expected_bend_count = int(expected_overrides.get(part_id, len(bends)))
        if expected_bend_count <= 0:
            expected_bend_count = int(len(bends))

        return {
            "part_id": part_id,
            "cad_source": cad_source,
            "total_bends": len(bends),
            "expected_bend_count": expected_bend_count,
            "bends": [b.to_dict() for b in bends],
            "message": f"Extracted {len(bends)} bend specifications from CAD"
        }

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@app.post("/api/bend-inspection/analyze-from-catalog", tags=["Bend Inspection"])
async def analyze_from_catalog(
    background_tasks: BackgroundTasks,
    part_id: str = Form(..., description="Part ID from catalog"),
    scan_file: UploadFile = File(..., description="Scan file to analyze"),
):
    """
    Run bend inspection using CAD from parts catalog.

    Instead of uploading a CAD file, select a part from the catalog
    that already has a CAD file uploaded.
    """
    # Get part from catalog
    catalog = get_catalog()
    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, f"Part not found: {part_id}")

    if not part.cad_file_path or not Path(part.cad_file_path).exists():
        raise HTTPException(400, f"Part {part.part_number} has no CAD file uploaded")

    # Validate scan file
    scan_ext = Path(scan_file.filename).suffix.lower()
    valid_exts = ['.stl', '.ply', '.obj', '.pcd']
    if scan_ext not in valid_exts:
        raise HTTPException(400, f"Scan file must be STL/PLY/OBJ/PCD, got {scan_ext}")

    await validate_file_size(scan_file, MAX_MESH_FILE_SIZE, "Scan")

    # Create job directory
    job_id = f"bend_{uuid.uuid4().hex[:8]}"
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # Save scan file
        scan_path = job_dir / f"scan{scan_ext}"
        scan_content = await scan_file.read()
        async with aiofiles.open(scan_path, "wb") as f:
            await f.write(scan_content)

        # Initialize async job state
        bend_inspection_results[job_id] = {
            "status": "running",
            "progress": 0,
            "part_id": part.id,
            "part_name": part.part_name or part.part_number,
            "created_at": datetime.now().isoformat(),
        }

        # Reuse the same validated progressive inspection pipeline.
        background_tasks.add_task(
            run_bend_inspection_task,
            job_id,
            str(part.cad_file_path),
            str(scan_path),
            part.id,
            part.part_name or part.part_number,
            part.default_tolerance_angle,
            0.5,
        )

        return {
            "job_id": job_id,
            "status": "started",
            "message": "Catalog-based bend inspection started",
            "part_id": part.id,
            "part_number": part.part_number,
            "part_name": part.part_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in catalog analysis: {e}")
        if job_dir.exists():
            shutil.rmtree(job_dir)
        bend_inspection_results.pop(job_id, None)
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/api/parts/with-cad", tags=["Parts"])
async def get_parts_with_cad():
    """Get all parts that have CAD files uploaded (for analysis picker)."""
    catalog = get_catalog()
    parts = catalog.list_parts(limit=1000, offset=0)

    # Filter to only parts with CAD files
    parts_with_cad = []
    for p in parts:
        if p.cad_file_path and Path(p.cad_file_path).exists():
            bend_specs = catalog.get_bend_specs_for_part(p.id)
            parts_with_cad.append({
                "id": p.id,
                "part_number": p.part_number,
                "part_name": p.part_name,
                "name": p.part_name,  # Backward compatibility for older clients
                "customer": p.customer,
                "has_bend_specs": len(bend_specs) > 0,
            })

    return {"parts": parts_with_cad, "total": len(parts_with_cad)}


@app.delete("/api/bend-inspection/{job_id}", tags=["Bend Inspection"])
async def delete_bend_inspection_job(job_id: str):
    """Delete a bend inspection job and its files."""
    if job_id in bend_inspection_results:
        del bend_inspection_results[job_id]

    # Delete files
    job_dir = UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    output_dir = OUTPUT_DIR / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return {"message": f"Bend inspection job {job_id} deleted"}


# ==================== Part Catalog Endpoints ====================

class PartCreate(BaseModel):
    """Request model for creating a part"""
    part_number: str = Field(..., description="Unique part number")
    part_name: Optional[str] = Field(None, description="Descriptive name")
    customer: Optional[str] = Field(None, description="Customer name")
    revision: Optional[str] = Field(None, description="Revision identifier")
    material: Optional[str] = Field(None, description="Material specification")
    default_tolerance_angle: float = Field(1.0, description="Default angle tolerance (degrees)")
    default_tolerance_linear: float = Field(0.5, description="Default linear tolerance (mm)")
    notes: Optional[str] = Field(None, description="Additional notes")


class PartUpdate(BaseModel):
    """Request model for updating a part"""
    part_name: Optional[str] = None
    customer: Optional[str] = None
    revision: Optional[str] = None
    material: Optional[str] = None
    default_tolerance_angle: Optional[float] = None
    default_tolerance_linear: Optional[float] = None
    notes: Optional[str] = None


class BendSpecCreate(BaseModel):
    """Request model for creating a bend specification"""
    bend_id: str = Field(..., description="Bend identifier (e.g., B1, B2)")
    target_angle: float = Field(..., description="Target angle in degrees")
    target_radius: Optional[float] = Field(None, description="Target radius in mm")
    tolerance_angle: Optional[float] = Field(None, description="Angle tolerance override")
    tolerance_radius: Optional[float] = Field(None, description="Radius tolerance override")
    notes: Optional[str] = None


@app.get("/api/parts", tags=["Part Catalog"])
async def list_parts(
    search: Optional[str] = Query(None, description="Search in part number and name"),
    customer: Optional[str] = Query(None, description="Filter by customer"),
    has_cad: Optional[bool] = Query(None, description="Filter by CAD availability"),
    has_embedding: Optional[bool] = Query(None, description="Filter by embedding availability"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    List parts in the catalog with optional filters.

    Returns parts sorted by part number.
    """
    catalog = get_catalog()
    parts = catalog.list_parts(
        search=search,
        customer=customer,
        has_cad=has_cad,
        has_embedding=has_embedding,
        limit=limit,
        offset=offset,
    )
    # Get filtered count for pagination
    filtered_total = catalog.count_parts_filtered(
        search=search,
        customer=customer,
        has_cad=has_cad,
        has_embedding=has_embedding,
    )
    counts = catalog.count_parts()

    return {
        "parts": [p.to_dict() for p in parts],
        "total": filtered_total,  # Count of filtered results
        "counts": counts,  # Overall catalog stats
    }


@app.post("/api/parts", tags=["Part Catalog"])
async def create_part(part: PartCreate):
    """
    Create a new part in the catalog.

    Part number must be unique.
    """
    catalog = get_catalog()

    # Check for duplicate
    existing = catalog.get_part_by_number(part.part_number)
    if existing:
        raise HTTPException(400, f"Part {part.part_number} already exists")

    new_part = catalog.create_part(
        part_number=part.part_number,
        part_name=part.part_name,
        customer=part.customer,
        revision=part.revision,
        material=part.material,
        default_tolerance_angle=part.default_tolerance_angle,
        default_tolerance_linear=part.default_tolerance_linear,
        notes=part.notes,
    )

    return {"part": new_part.to_dict(), "message": "Part created"}


@app.get("/api/parts/stats", tags=["Part Catalog"])
async def get_parts_stats():
    """Get part catalog statistics."""
    catalog = get_catalog()
    return catalog.count_parts()


@app.get("/api/parts/{part_id}", tags=["Part Catalog"])
async def get_part(part_id: str):
    """
    Get a part by ID.

    Also returns bend specifications if available.
    """
    catalog = get_catalog()
    part = catalog.get_part(part_id)

    if not part:
        raise HTTPException(404, "Part not found")

    bend_specs = catalog.get_bend_specs_for_part(part_id)

    return {
        "part": part.to_dict(),
        "bend_specs": [s.to_dict() for s in bend_specs],
    }


@app.put("/api/parts/{part_id}", tags=["Part Catalog"])
async def update_part(part_id: str, update: PartUpdate):
    """Update a part's details."""
    catalog = get_catalog()

    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")

    updated = catalog.update_part(
        part_id,
        **{k: v for k, v in update.dict().items() if v is not None}
    )

    return {"part": updated.to_dict(), "message": "Part updated"}


@app.delete("/api/parts/{part_id}", tags=["Part Catalog"])
async def delete_part(part_id: str):
    """
    Delete a part and all related data.

    This includes CAD files, embeddings, and bend specs.
    """
    catalog = get_catalog()

    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")

    # Delete CAD file if exists
    if part.cad_file_path:
        cad_path = Path(part.cad_file_path)
        if cad_path.exists():
            cad_path.unlink()

    catalog.delete_part(part_id)

    return {"deleted": True, "message": f"Part {part.part_number} deleted"}


@app.post("/api/parts/{part_id}/cad", tags=["Part Catalog"])
async def upload_part_cad(
    part_id: str,
    cad_file: UploadFile = File(..., description="CAD file (STL, STEP, PLY)"),
    analyze_bends: bool = Form(True, description="Automatically detect bends from CAD"),
):
    """
    Upload a CAD file for a part.

    Optionally analyzes the CAD to extract bend specifications.
    """
    catalog = get_catalog()

    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")

    # Validate file extension
    filename = cad_file.filename or "upload"
    ext = Path(filename).suffix.lower()
    allowed_extensions = {'.stl', '.step', '.stp', '.ply', '.obj'}

    if ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed_extensions}")

    # Create parts CAD directory
    cad_dir = UPLOAD_DIR / "parts_cad"
    cad_dir.mkdir(exist_ok=True)

    # Save file with part number as name
    safe_filename = f"{part.part_number.replace('/', '_')}{ext}"
    cad_path = cad_dir / safe_filename

    content = await cad_file.read()
    async with aiofiles.open(cad_path, "wb") as f:
        await f.write(content)

    # Update part with CAD path
    import hashlib
    file_hash = hashlib.md5(content).hexdigest()

    catalog.update_part(
        part_id,
        cad_file_path=str(cad_path),
        cad_file_hash=file_hash,
        cad_uploaded_at=datetime.now().isoformat(),
    )

    result = {
        "message": "CAD file uploaded",
        "cad_path": str(cad_path),
    }

    # Optionally analyze bends
    if analyze_bends:
        try:
            runtime_cfg = load_bend_runtime_config()
            cad_vertices, cad_triangles, cad_source = load_cad_geometry(
                str(cad_path),
                cad_import_deflection=runtime_cfg.cad_import_deflection,
            )
            bends = extract_cad_bend_specs(
                cad_vertices=cad_vertices,
                cad_triangles=cad_triangles,
                tolerance_angle=part.default_tolerance_angle,
                tolerance_radius=0.5,
                cad_extractor_kwargs=runtime_cfg.cad_extractor_kwargs,
                cad_detector_kwargs=runtime_cfg.cad_detector_kwargs,
                cad_detect_call_kwargs=runtime_cfg.cad_detect_call_kwargs,
            )

            # Clear existing bend specs and add new ones
            catalog.delete_bend_specs_for_part(part_id)
            for bend in bends:
                catalog.add_bend_spec(
                    part_id=part_id,
                    bend_id=bend.bend_id,
                    target_angle=bend.target_angle,
                    target_radius=bend.target_radius,
                    tolerance_angle=bend.tolerance_angle,
                    tolerance_radius=bend.tolerance_radius,
                )

            # Update bend count
            catalog.update_part(part_id, bend_count=len(bends))

            result["cad_source"] = cad_source
            result["bends_detected"] = len(bends)
            result["bends"] = [b.to_dict() for b in bends]
        except Exception as e:
            logger.warning(f"Bend analysis failed: {e}")
            result["bend_analysis_error"] = str(e)

    return result


@app.get("/api/parts/{part_id}/cad", tags=["Part Catalog"])
async def download_part_cad(part_id: str):
    """Download the CAD file for a part."""
    catalog = get_catalog()

    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")

    if not part.cad_file_path:
        raise HTTPException(404, "Part has no CAD file")

    cad_path = Path(part.cad_file_path)
    if not cad_path.exists():
        raise HTTPException(404, "CAD file not found on disk")

    return FileResponse(
        cad_path,
        filename=f"{part.part_number}{cad_path.suffix}",
        media_type="application/octet-stream",
    )


@app.post("/api/parts/{part_id}/bend-specs", tags=["Part Catalog"])
async def add_bend_spec(part_id: str, spec: BendSpecCreate):
    """Add a bend specification to a part."""
    catalog = get_catalog()

    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")

    new_spec = catalog.add_bend_spec(
        part_id=part_id,
        bend_id=spec.bend_id,
        target_angle=spec.target_angle,
        target_radius=spec.target_radius,
        tolerance_angle=spec.tolerance_angle,
        tolerance_radius=spec.tolerance_radius,
        notes=spec.notes,
    )

    return {"bend_spec": new_spec.to_dict(), "message": "Bend spec added"}


@app.get("/api/parts/{part_id}/bend-specs", tags=["Part Catalog"])
async def get_bend_specs(part_id: str):
    """Get all bend specifications for a part."""
    catalog = get_catalog()

    part = catalog.get_part(part_id)
    if not part:
        raise HTTPException(404, "Part not found")

    specs = catalog.get_bend_specs_for_part(part_id)

    return {"bend_specs": [s.to_dict() for s in specs]}


@app.post("/api/parts/import-csv", tags=["Part Catalog"])
async def import_parts_csv(
    csv_file: UploadFile = File(..., description="CSV file with parts"),
    part_number_col: str = Form("part_number", description="Column name for part number"),
    part_name_col: str = Form("part_name", description="Column name for part name"),
    customer_col: str = Form("customer", description="Column name for customer"),
    skip_existing: bool = Form(True, description="Skip parts that already exist"),
):
    """
    Import parts from a CSV file.

    The CSV must have at least a column for part numbers.
    """
    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await csv_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        catalog = get_catalog()
        results = import_parts_from_csv(
            catalog,
            tmp_path,
            part_number_col=part_number_col,
            part_name_col=part_name_col,
            customer_col=customer_col,
            skip_existing=skip_existing,
        )

        return {
            "message": f"Import complete: {results['imported']} imported, {results['skipped']} skipped, {results['errors']} errors",
            "results": results,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/parts/by-number/{part_number}", tags=["Part Catalog"])
async def get_part_by_number(part_number: str):
    """Get a part by its part number."""
    catalog = get_catalog()
    part = catalog.get_part_by_number(part_number)

    if not part:
        raise HTTPException(404, f"Part {part_number} not found")

    bend_specs = catalog.get_bend_specs_for_part(part.id)

    return {
        "part": part.to_dict(),
        "bend_specs": [s.to_dict() for s in bend_specs],
    }


# =============================================================================
# PART RECOGNITION API
# =============================================================================

from embedding_service import compute_embedding, compute_similarity, EMBEDDING_VERSION
from faiss_index import get_faiss_index


class RecognitionRequest(BaseModel):
    """Request model for part recognition"""
    top_k: int = Field(10, ge=1, le=50, description="Number of candidates to return")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    verify_with_icp: bool = Field(False, description="Verify top candidates with ICP alignment")


@app.post("/api/recognize", tags=["Part Recognition"])
async def recognize_part(
    scan_file: UploadFile = File(..., description="Point cloud file (PLY, STL)"),
    top_k: int = Form(10),
    threshold: float = Form(0.5),
):
    """
    Recognize a part from a scan file.

    Computes embedding from scan, searches part catalog for matches.
    Returns ranked candidates with similarity scores.

    For production use, consider using the SSE endpoint for progress updates.
    """
    import tempfile
    import time

    start_time = time.time()

    # Validate file
    filename = scan_file.filename or "scan"
    suffix = Path(filename).suffix.lower()
    if suffix not in ['.ply', '.stl', '.obj', '.pcd']:
        raise HTTPException(400, f"Unsupported file format: {suffix}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await scan_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import open3d as o3d
        import numpy as np

        # Load point cloud
        if suffix in ['.stl', '.obj']:
            mesh = o3d.io.read_triangle_mesh(tmp_path)
            if mesh.is_empty():
                raise HTTPException(400, "Failed to read mesh file")
            if len(mesh.triangles) > 0:
                pcd = mesh.sample_points_uniformly(number_of_points=10000)
            else:
                pcd = o3d.io.read_point_cloud(tmp_path)
                if pcd.is_empty():
                    raise HTTPException(400, "Mesh contains no triangles and no point cloud data")
        else:
            pcd = o3d.io.read_point_cloud(tmp_path)
            if pcd.is_empty():
                raise HTTPException(400, "Failed to read point cloud file")

        points = np.asarray(pcd.points)
        n_points = len(points)

        if n_points < 100:
            raise HTTPException(400, f"Too few points ({n_points}), need at least 100")

        # Compute embedding
        embed_result = compute_embedding(points)
        embedding = embed_result.embedding

        # Search FAISS
        faiss_index = get_faiss_index()
        if faiss_index.count() == 0:
            return {
                "candidates": [],
                "message": "No parts in index. Add parts with CAD files first.",
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        search_results = faiss_index.search(embedding, k=top_k, threshold=threshold)

        # Enrich with part details
        catalog = get_catalog()
        candidates = []

        orphaned_count = 0
        for result in search_results:
            part = catalog.get_part(result.part_id)
            if part:
                candidates.append({
                    "part_id": result.part_id,
                    "part_number": part.part_number,
                    "part_name": part.part_name,
                    "customer": part.customer,
                    "similarity": round(result.similarity, 4),
                    "distance": round(result.distance, 4),
                    "has_cad": part.cad_file_path is not None,
                })
            else:
                # Part in FAISS but not in catalog (orphaned embedding)
                orphaned_count += 1
                candidates.append({
                    "part_id": result.part_id,
                    "part_number": f"[orphaned-{result.part_id[:8]}]",
                    "part_name": None,
                    "customer": None,
                    "similarity": round(result.similarity, 4),
                    "distance": round(result.distance, 4),
                    "has_cad": False,
                    "warning": "Part not found in catalog",
                })

        processing_time = (time.time() - start_time) * 1000

        response = {
            "candidates": candidates,
            "total_in_index": faiss_index.count(),
            "scan_info": {
                "filename": filename,
                "n_points": n_points,
                "embedding_time_ms": embed_result.processing_time_ms,
            },
            "processing_time_ms": round(processing_time, 1),
        }

        if orphaned_count > 0:
            response["warning"] = f"{orphaned_count} embeddings have no matching catalog entry"

        return response

    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.get("/api/recognize/status", tags=["Part Recognition"])
async def get_recognition_status():
    """Get the status of the recognition system."""
    faiss_index = get_faiss_index()
    catalog = get_catalog()
    stats = catalog.count_parts()

    return {
        "index_count": faiss_index.count(),
        "catalog_stats": stats,
        "embedding_version": EMBEDDING_VERSION,
        "status": "ready" if faiss_index.count() > 0 else "no_parts",
    }


@app.post("/api/recognize/compare", tags=["Part Recognition"])
async def compare_embeddings(
    file1: UploadFile = File(..., description="First point cloud"),
    file2: UploadFile = File(..., description="Second point cloud"),
):
    """
    Compare two point clouds directly (without catalog lookup).

    Useful for testing or ad-hoc comparisons.
    """
    import tempfile
    import time
    import open3d as o3d
    import numpy as np

    start_time = time.time()

    embeddings = []
    infos = []

    for i, scan_file in enumerate([file1, file2]):
        filename = scan_file.filename or f"file{i+1}"
        suffix = Path(filename).suffix.lower()

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await scan_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if suffix in ['.stl', '.obj']:
                mesh = o3d.io.read_triangle_mesh(tmp_path)
                if len(mesh.triangles) > 0:
                    pcd = mesh.sample_points_uniformly(number_of_points=10000)
                else:
                    pcd = o3d.io.read_point_cloud(tmp_path)
                    if pcd.is_empty():
                        raise HTTPException(400, f"{filename} has no triangles and no point cloud data")
            else:
                pcd = o3d.io.read_point_cloud(tmp_path)

            points = np.asarray(pcd.points)
            result = compute_embedding(points)
            embeddings.append(result.embedding)
            infos.append({
                "filename": filename,
                "n_points": len(points),
                "embedding_time_ms": result.processing_time_ms,
            })
        finally:
            os.unlink(tmp_path)

    # Compute similarity
    similarity = compute_similarity(embeddings[0], embeddings[1])

    return {
        "similarity": round(similarity, 4),
        "file1": infos[0],
        "file2": infos[1],
        "processing_time_ms": round((time.time() - start_time) * 1000, 1),
    }


@app.post("/api/recognize/reindex", tags=["Part Recognition"])
async def reindex_all_parts(background_tasks: BackgroundTasks):
    """
    Rebuild the FAISS index with all parts that have CAD files.

    Runs in background. Check /api/recognize/status for progress.
    """
    from embedding_worker import get_embedding_worker

    def rebuild_task():
        worker = get_embedding_worker()
        worker.process_pending()

    background_tasks.add_task(rebuild_task)

    return {
        "message": "Reindex started in background",
        "status": "processing",
    }


# ============================================================================
# Live Scan API
# ============================================================================

# Global session manager instance
_live_scan_manager = None
_live_scan_watch_path = None


def get_live_scan_watch_path():
    """Get the current watch path (from env, config, or default)."""
    global _live_scan_watch_path
    if _live_scan_watch_path is None:
        _live_scan_watch_path = os.environ.get(
            "SHERMAN_LIVE_SCAN_PATH",
            str(BASE_DIR / "live_scans")
        )
    return _live_scan_watch_path


def set_live_scan_watch_path(new_path: str):
    """Set a new watch path. Requires manager restart to take effect."""
    global _live_scan_watch_path, _live_scan_manager

    # Validate path exists
    path = Path(new_path)
    if not path.exists():
        os.makedirs(path, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {new_path}")

    _live_scan_watch_path = str(path.resolve())

    # Stop existing manager if running
    if _live_scan_manager is not None and _live_scan_manager._running:
        _live_scan_manager.stop()
        _live_scan_manager = None

    return _live_scan_watch_path


def get_live_scan_manager():
    """Get or create the global live scan session manager."""
    global _live_scan_manager
    if _live_scan_manager is None:
        from live_scan_session import LiveScanSessionManager

        watch_path = get_live_scan_watch_path()

        # Create watch directory if it doesn't exist
        os.makedirs(watch_path, exist_ok=True)

        _live_scan_manager = LiveScanSessionManager(watch_path)

    return _live_scan_manager


LIVE_SCAN_DEMO_ASSETS: Dict[str, Dict[str, str]] = {
    "47266000_F": {
        "part_number": "47266000_F",
        "part_name": "Demo Part 47266000_F",
        "cad_mesh_relpath": "test_files/47266000_F/cad_mesh.stl",
        "scan_mesh_relpath": "test_files/47266000_F/scan.stl",
    },
    "11979003_C": {
        "part_number": "11979003_C",
        "part_name": "Demo Part 11979003_C",
        "cad_mesh_relpath": "test_files/11979003_C/cad_mesh.stl",
        "scan_mesh_relpath": "test_files/11979003_C/scan.stl",
    },
}
DEFAULT_LIVE_SCAN_DEMO_PART = "47266000_F"


class LoadLiveScanDemoRequest(BaseModel):
    """Request model for injecting a demo LiDAR scan."""
    part_number: Optional[str] = Field(
        default=DEFAULT_LIVE_SCAN_DEMO_PART,
        description="Demo part number key (e.g., 47266000_F)",
    )
    point_count: int = Field(
        default=14000,
        ge=2000,
        le=200000,
        description="Number of LiDAR-like points to generate",
    )
    noise_sigma_mm: float = Field(
        default=0.08,
        ge=0.0,
        le=2.0,
        description="Gaussian noise level in mm",
    )
    reset_session: bool = Field(
        default=True,
        description="Reset current live-scan session before injecting demo scan",
    )


def _resolve_live_scan_demo_asset(part_number: Optional[str]) -> Dict[str, str]:
    if part_number:
        key = part_number.strip().upper()
        if key in LIVE_SCAN_DEMO_ASSETS:
            return LIVE_SCAN_DEMO_ASSETS[key]
    return LIVE_SCAN_DEMO_ASSETS[DEFAULT_LIVE_SCAN_DEMO_PART]


def _ensure_demo_part_in_catalog(asset: Dict[str, str]):
    """Create/update demo part in catalog with a CAD mesh path."""
    import hashlib

    catalog = get_catalog()
    part = catalog.get_part_by_number(asset["part_number"])
    if part is None:
        part = catalog.create_part(
            part_number=asset["part_number"],
            part_name=asset["part_name"],
            customer="DEMO",
            material="Aluminum",
            notes="Auto-generated demo part for Live Scan test mode",
        )

    cad_source = (BASE_DIR / asset["cad_mesh_relpath"]).resolve()
    if not cad_source.exists():
        raise FileNotFoundError(f"Demo CAD mesh not found: {cad_source}")

    cad_dir = UPLOAD_DIR / "parts_cad"
    cad_dir.mkdir(parents=True, exist_ok=True)
    cad_target = cad_dir / f"{asset['part_number'].replace('/', '_')}_demo.stl"
    if not cad_target.exists() or cad_target.stat().st_size != cad_source.stat().st_size:
        shutil.copy2(cad_source, cad_target)

    cad_hash = hashlib.md5(cad_target.read_bytes()).hexdigest()
    if (
        part.cad_file_path != str(cad_target)
        or part.cad_file_hash != cad_hash
        or not part.cad_uploaded_at
    ):
        catalog.update_part(
            part.id,
            cad_file_path=str(cad_target),
            cad_file_hash=cad_hash,
            cad_uploaded_at=datetime.now().isoformat(),
        )
        part = catalog.get_part(part.id)

    return part, cad_target


def _ensure_demo_part_indexed(part):
    """Ensure part embedding exists both in DB and in FAISS index."""
    from embedding_worker import get_embedding_worker

    catalog = get_catalog()
    faiss_index = get_faiss_index()

    if (not part.has_embedding) or (not faiss_index.has_part(part.id)):
        worker = get_embedding_worker()
        ok = worker.process_single(part.id)
        if not ok:
            raise RuntimeError(
                f"Failed to compute demo embedding for {part.part_number}. "
                "Ensure demo CAD mesh is valid."
            )
        faiss_index.save()

    updated_part = catalog.get_part(part.id)
    if updated_part is None:
        raise RuntimeError("Part disappeared from catalog while indexing demo part")
    return updated_part


def _write_lidar_demo_scan(
    source_scan_mesh: Path,
    output_scan_path: Path,
    point_count: int,
    noise_sigma_mm: float,
) -> Dict[str, Any]:
    """Generate a LiDAR-like point cloud file from a demo mesh/point cloud source."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(source_scan_mesh))
    points: Optional[np.ndarray] = None

    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        sampled = mesh.sample_points_uniformly(number_of_points=max(point_count * 2, 5000))
        points = np.asarray(sampled.points)
    else:
        pcd = o3d.io.read_point_cloud(str(source_scan_mesh))
        if pcd is not None and not pcd.is_empty():
            points = np.asarray(pcd.points)

    if points is None or len(points) < 200:
        raise ValueError(
            f"Demo source scan is not usable: {source_scan_mesh} "
            "(needs mesh triangles or dense point cloud)"
        )

    rng = np.random.default_rng()

    # Randomly subsample to requested point count.
    if len(points) > point_count:
        idx = rng.choice(len(points), size=point_count, replace=False)
        points = points[idx]

    # Add subtle Gaussian jitter to mimic LiDAR noise.
    if noise_sigma_mm > 0:
        points = points + rng.normal(0.0, noise_sigma_mm, size=points.shape)

    # Add mild dropout pattern (scan shadowing) for realism.
    if len(points) > 4000:
        keep = rng.random(len(points)) > 0.06
        points = points[keep]

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    output_scan_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(output_scan_path), pcd_out, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write demo LiDAR scan: {output_scan_path}")

    return {
        "point_count": int(len(points)),
        "noise_sigma_mm": float(noise_sigma_mm),
    }


@app.get("/api/live-scan/demo/options", tags=["Live Scan"])
async def get_live_scan_demo_options():
    """List built-in demo parts available for one-click Live Scan ingestion."""
    options = []
    for key, asset in LIVE_SCAN_DEMO_ASSETS.items():
        cad_path = (BASE_DIR / asset["cad_mesh_relpath"]).resolve()
        scan_path = (BASE_DIR / asset["scan_mesh_relpath"]).resolve()
        options.append({
            "part_number": key,
            "part_name": asset.get("part_name"),
            "cad_available": cad_path.exists(),
            "scan_available": scan_path.exists(),
        })
    return {
        "default_part_number": DEFAULT_LIVE_SCAN_DEMO_PART,
        "options": options,
    }


@app.post("/api/live-scan/demo/load", tags=["Live Scan"])
async def load_live_scan_demo_scan(request: LoadLiveScanDemoRequest):
    """
    Inject a demo LiDAR scan into Live Scan flow.

    This endpoint is functional (not mock): it ensures the part is in catalog,
    ensures embedding/index presence for recognition, creates a real point-cloud
    scan file under the watch folder, and ingests it into the session pipeline.
    """
    try:
        manager = get_live_scan_manager()
        if not manager.is_running():
            manager.start()
        if request.reset_session:
            manager.reset_session()

        asset = _resolve_live_scan_demo_asset(request.part_number)
        part, _ = _ensure_demo_part_in_catalog(asset)
        part = _ensure_demo_part_indexed(part)

        source_scan = (BASE_DIR / asset["scan_mesh_relpath"]).resolve()
        if not source_scan.exists():
            raise FileNotFoundError(f"Demo scan source not found: {source_scan}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{asset['part_number']}_LIDAR_DEMO_{ts}.ply"
        output_scan = Path(manager.watch_path) / filename

        lidar_meta = _write_lidar_demo_scan(
            source_scan_mesh=source_scan,
            output_scan_path=output_scan,
            point_count=request.point_count,
            noise_sigma_mm=request.noise_sigma_mm,
        )

        manager.ingest_scan_file(str(output_scan), filename=filename)
        session = manager.get_current_session()

        return {
            "message": "Demo LiDAR scan loaded",
            "watch_path": str(manager.watch_path),
            "scan_file": str(output_scan),
            "demo_part": {
                "part_id": part.id,
                "part_number": part.part_number,
                "part_name": part.part_name,
                "has_cad": bool(part.cad_file_path),
                "has_embedding": bool(part.has_embedding),
            },
            "lidar": lidar_meta,
            "session": session.to_dict() if session else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to load live scan demo: {exc}")
        raise HTTPException(500, f"Failed to load live scan demo: {exc}")


@app.get("/api/live-scan/session", tags=["Live Scan"])
async def get_live_scan_session():
    """Get the current live scan session."""
    manager = get_live_scan_manager()
    session = manager.get_current_session()

    if session is None:
        return None

    return session.to_dict()


@app.get("/api/live-scan/session/points", tags=["Live Scan"])
async def get_live_scan_session_points(
    max_points: int = Query(15000, ge=1000, le=100000),
    max_scans: int = Query(4, ge=1, le=12),
):
    """
    Return merged point cloud samples for the current live-scan session.

    Used by the Live Scan viewer to render actual scanned geometry instead of
    synthetic placeholder particles.
    """
    manager = get_live_scan_manager()
    session = manager.get_current_session()
    if session is None or not session.scans:
        return {
            "session_id": None,
            "scan_count": 0,
            "point_count": 0,
            "points": [],
        }

    try:
        import open3d as o3d

        scan_items = session.scans[-max_scans:]
        arrays: List[np.ndarray] = []

        for scan in scan_items:
            path = Path(scan.file_path)
            if not path.exists():
                continue

            ext = path.suffix.lower()
            pts: Optional[np.ndarray] = None

            if ext in {".ply", ".pcd"}:
                pcd = o3d.io.read_point_cloud(str(path))
                if pcd is not None and not pcd.is_empty():
                    pts = np.asarray(pcd.points)
            elif ext in {".stl", ".obj"}:
                mesh = o3d.io.read_triangle_mesh(str(path))
                if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                    sampled = mesh.sample_points_uniformly(number_of_points=min(max_points, 15000))
                    pts = np.asarray(sampled.points)
                else:
                    pcd = o3d.io.read_point_cloud(str(path))
                    if pcd is not None and not pcd.is_empty():
                        pts = np.asarray(pcd.points)

            if pts is not None and len(pts) > 0:
                arrays.append(pts.astype(np.float32, copy=False))

        if not arrays:
            return {
                "session_id": session.id,
                "scan_count": len(session.scans),
                "point_count": 0,
                "points": [],
            }

        merged = np.vstack(arrays)
        if len(merged) > max_points:
            idx = np.random.choice(len(merged), size=max_points, replace=False)
            merged = merged[idx]

        # Round for smaller payload without visible quality loss.
        merged = np.round(merged, 3)

        return {
            "session_id": session.id,
            "scan_count": len(session.scans),
            "point_count": int(len(merged)),
            "points": merged.tolist(),
        }
    except Exception as exc:
        logger.warning(f"Failed to fetch live scan points: {exc}")
        return {
            "session_id": session.id,
            "scan_count": len(session.scans),
            "point_count": 0,
            "points": [],
            "warning": str(exc),
        }


@app.post("/api/live-scan/session/{session_id}/confirm", tags=["Live Scan"])
async def confirm_live_scan_part(
    session_id: str,
    part_id: str = Body(...),
    part_number: str = Body(...),
):
    """Confirm the part identity for a live scan session."""
    manager = get_live_scan_manager()
    session = manager.get_current_session()

    if session is None or session.id != session_id:
        raise HTTPException(404, "Session not found")

    success = manager.confirm_part(part_id, part_number)
    if not success:
        raise HTTPException(400, "Cannot confirm part in current state")

    return {"confirmed": True, "part_number": part_number}


@app.post("/api/live-scan/session/{session_id}/complete", tags=["Live Scan"])
async def complete_live_scan(session_id: str):
    """Mark a live scan session as complete."""
    manager = get_live_scan_manager()
    session = manager.get_current_session()

    if session is None or session.id != session_id:
        raise HTTPException(404, "Session not found")

    success = manager.complete_scan()
    if not success:
        raise HTTPException(400, "Cannot complete scan in current state")

    return {"completed": True}


@app.post("/api/live-scan/session/{session_id}/cancel", tags=["Live Scan"])
async def cancel_live_scan(session_id: str):
    """Cancel a live scan session."""
    manager = get_live_scan_manager()
    session = manager.get_current_session()

    if session is None or session.id != session_id:
        raise HTTPException(404, "Session not found")

    success = manager.cancel_session()
    if not success:
        raise HTTPException(400, "Cannot cancel session")

    return {"cancelled": True}


@app.post("/api/live-scan/session/reset", tags=["Live Scan"])
async def reset_live_scan_session():
    """Reset/clear the current session to start fresh."""
    manager = get_live_scan_manager()
    manager.reset_session()
    return {"reset": True}


@app.post("/api/live-scan/start", tags=["Live Scan"])
async def start_live_scan_manager(watch_path: Optional[str] = Body(None, embed=True)):
    """Start the live scan session manager."""
    if watch_path:
        set_live_scan_watch_path(watch_path)
    manager = get_live_scan_manager()

    if manager.is_running():
        return {"message": "Manager already running", "watch_path": str(manager.watch_path)}

    manager.start()

    return {"message": "Manager started", "watch_path": str(manager.watch_path)}


@app.post("/api/live-scan/stop", tags=["Live Scan"])
async def stop_live_scan_manager():
    """Stop the live scan session manager."""
    manager = get_live_scan_manager()

    if not manager.is_running():
        return {"message": "Manager not running"}

    manager.stop()

    return {"message": "Manager stopped"}


@app.get("/api/live-scan/status", tags=["Live Scan"])
async def get_live_scan_status():
    """Get the live scan manager status."""
    manager = get_live_scan_manager()

    return {
        "running": manager._running,
        "watch_path": str(manager.watch_path),
        "session_history_count": len(manager.get_session_history()),
    }


@app.get("/api/live-scan/config", tags=["Live Scan"])
async def get_live_scan_config():
    """Get the live scan configuration."""
    return {
        "watch_path": get_live_scan_watch_path(),
        "running": _live_scan_manager._running if _live_scan_manager else False,
    }


@app.post("/api/live-scan/config", tags=["Live Scan"])
async def update_live_scan_config(watch_path: str = Body(..., embed=True)):
    """
    Update the live scan watch folder path.

    This will stop any running session manager and reconfigure it
    with the new path. Call /api/live-scan/start to begin watching.
    """
    try:
        new_path = set_live_scan_watch_path(watch_path)
        return {
            "success": True,
            "watch_path": new_path,
            "message": "Watch path updated. Call /api/live-scan/start to begin watching.",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to set watch path: {e}")


@app.get("/api/live-scan/session/stream", tags=["Live Scan"])
async def stream_live_scan_session():
    """
    Stream live scan session updates via Server-Sent Events (SSE).

    The client receives updates whenever:
    - Session state changes
    - New scan file is received
    - Coverage is updated
    - Recognition completes
    """
    manager = get_live_scan_manager()

    async def generate():
        last_update_time = None
        last_session_id = None

        while True:
            session = manager.get_current_session()

            # Check if we should send an update
            should_send = False

            if session is None:
                # No active session
                if last_session_id is not None:
                    # Session ended - send null update
                    should_send = True
                    last_session_id = None
            else:
                # Active session exists
                if session.id != last_session_id:
                    # New session started
                    should_send = True
                    last_session_id = session.id
                    last_update_time = session.updated_at
                elif session.updated_at != last_update_time:
                    # Session was updated
                    should_send = True
                    last_update_time = session.updated_at

            if should_send:
                data = session.to_dict() if session else None
                yield f"data: {json.dumps(data)}\n\n"

            await asyncio.sleep(0.25)  # 4 updates per second max

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# Mount frontend static files
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
