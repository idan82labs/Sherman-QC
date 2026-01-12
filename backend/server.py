"""
FastAPI Backend Server for Scan QC System
Handles file uploads, QC analysis, and PDF generation with real-time progress.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Header, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
import uvicorn
import asyncio
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
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

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Initialize database
db = init_db()

# In-memory job state for real-time progress (supplements database)
job_progress = {}


class JobProgressState:
    """Track real-time job progress in memory"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"
        self.progress = 0
        self.stage = ""
        self.message = ""


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


class LoginRequest(BaseModel):
    """Login request body"""
    username: str
    password: str


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


@app.post("/api/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    reference_file: UploadFile = File(...),
    scan_file: UploadFile = File(...),
    drawing_file: Optional[UploadFile] = File(None),
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
    - part_id: Part number/ID
    - part_name: Part name/description
    - material: Material specification
    - tolerance: Tolerance in mm (plus/minus)

    Returns job_id for tracking progress.
    """
    # Validate file types
    ref_ext = Path(reference_file.filename).suffix.lower()
    scan_ext = Path(scan_file.filename).suffix.lower()

    if ref_ext not in ['.stl', '.obj', '.ply']:
        raise HTTPException(400, f"Reference must be STL/OBJ/PLY, got {ref_ext}")

    if scan_ext not in ['.stl', '.ply', '.obj', '.pcd']:
        raise HTTPException(400, f"Scan must be STL/PLY/OBJ/PCD, got {scan_ext}")

    # Create job ID
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded files
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    ref_path = job_dir / f"reference{ref_ext}"
    scan_path = job_dir / f"scan{scan_ext}"

    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference_file.file, f)

    with open(scan_path, "wb") as f:
        shutil.copyfileobj(scan_file.file, f)

    # Save drawing if provided
    drawing_path = None
    drawing_context = ""
    if drawing_file and drawing_file.filename:
        drawing_ext = Path(drawing_file.filename).suffix.lower()
        if drawing_ext == '.pdf':
            drawing_path = job_dir / f"drawing{drawing_ext}"
            with open(drawing_path, "wb") as f:
                shutil.copyfileobj(drawing_file.file, f)
            drawing_context = extract_text_from_pdf(str(drawing_path))

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
        drawing_context
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
    drawing_context: str
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
        progress_state.stage = update.stage
        progress_state.progress = update.progress
        progress_state.message = update.message
        # Periodically update database (every 10% or on stage change)
        if update.progress % 10 == 0 or update.stage in ["load", "preprocess", "align", "analyze", "ai", "complete"]:
            db.update_job_progress(job_id, "running", update.progress, update.stage, update.message)

    try:
        # Create output directory for this job
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        # Initialize engine with output directory for heatmaps
        engine = ScanQCEngine(
            progress_callback=progress_callback,
            output_dir=str(output_dir)
        )

        # Load files
        engine.load_reference(ref_path)
        engine.load_scan(scan_path)

        # Run analysis (heatmaps will be saved to output_dir)
        report = engine.run_analysis(
            part_id=part_id,
            part_name=part_name,
            material=material,
            tolerance=tolerance,
            drawing_context=drawing_context,
            output_dir=str(output_dir)
        )

        # Convert to dict
        report_dict = report.to_dict()

        # Update heatmap paths to be relative URLs
        if report_dict.get("heatmaps"):
            for key in report_dict["heatmaps"]:
                if report_dict["heatmaps"][key]:
                    # Convert absolute path to relative URL
                    filename = Path(report_dict["heatmaps"][key]).name
                    report_dict["heatmaps"][key] = f"/api/heatmap/{job_id}/{filename}"

        # Save JSON report
        report_path = output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # Generate PDF
        pdf_path = output_dir / f"QC_Report_{part_id}.pdf"
        generate_pdf_report(report_dict, str(pdf_path))

        # Update in-memory state
        progress_state.status = "completed"
        progress_state.progress = 100
        progress_state.message = "Analysis complete!"

        # Update database with result
        db.update_job_result(
            job_id=job_id,
            result=report_dict,
            report_path=str(report_path),
            pdf_path=str(pdf_path)
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        progress_state.status = "failed"
        progress_state.message = f"Error: {str(e)}"

        # Update database with error
        db.update_job_error(job_id, str(e))


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
        return {"status": "running", "progress": progress}

    if job.status == "failed":
        raise HTTPException(500, f"Analysis failed: {job.error}")

    # Parse result from JSON
    result = None
    if job.result_json:
        try:
            result = json.loads(job.result_json)
        except json.JSONDecodeError:
            result = None

    return {
        "status": "completed",
        "result": result,
        "pdf_available": job.pdf_path is not None
    }


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


@app.get("/api/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    part_id: Optional[str] = Query(None, description="Filter by part ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max jobs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List all analysis jobs with optional filtering"""
    jobs_list = db.list_jobs(status=status, part_id=part_id, limit=limit, offset=offset)

    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "part_id": j.part_id,
                "part_name": j.part_name,
                "material": j.material,
                "tolerance": j.tolerance,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
                "progress": j.progress
            }
            for j in jobs_list
        ],
        "count": len(jobs_list),
        "offset": offset,
        "limit": limit
    }


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Get full details for a specific job"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Delete files
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id

    if job_upload_dir.exists():
        shutil.rmtree(job_upload_dir)

    if job_output_dir.exists():
        shutil.rmtree(job_output_dir)

    # Delete from database
    db.delete_job(job_id)

    # Remove from in-memory progress
    if job_id in job_progress:
        del job_progress[job_id]

    return {"message": f"Job {job_id} deleted", "job_id": job_id}


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

    return stats


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

        if ref_ext not in ['.stl', '.obj', '.ply']:
            raise HTTPException(400, f"Part {i+1}: Reference must be STL/OBJ/PLY, got {ref_ext}")

        if scan_ext not in ['.stl', '.ply', '.obj', '.pcd']:
            raise HTTPException(400, f"Part {i+1}: Scan must be STL/PLY/OBJ/PCD, got {scan_ext}")

        # Create job ID for this part
        job_id = f"{batch_id}_{i:03d}"

        # Create part directory
        part_dir = batch_dir / f"part_{i:03d}"
        part_dir.mkdir(exist_ok=True)

        # Save files
        ref_path = part_dir / f"reference{ref_ext}"
        scan_path = part_dir / f"scan{scan_ext}"

        with open(ref_path, "wb") as f:
            shutil.copyfileobj(ref_file.file, f)

        with open(scan_path, "wb") as f:
            shutil.copyfileobj(scan_file.file, f)

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
                            part_result.result = result_data.get("overall_result", "UNKNOWN")
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
                    part_result.result = result_data.get("overall_result", "UNKNOWN")
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
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100)
):
    """List all batch jobs"""
    batches = []

    for batch_id, batch_state in list(batch_jobs.items())[-limit:]:
        if status and batch_state.status != status:
            continue

        batches.append({
            "batch_id": batch_id,
            "name": batch_state.name,
            "status": batch_state.status,
            "progress": round(batch_state.progress, 1),
            "total_parts": batch_state.total_parts,
            "completed_parts": batch_state.completed_parts,
            "failed_parts": batch_state.failed_parts,
            "created_at": batch_state.created_at,
            "completed_at": batch_state.completed_at
        })

    return {"batches": batches, "count": len(batches)}


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


# Mount frontend static files
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
