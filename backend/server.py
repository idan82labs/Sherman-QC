"""
FastAPI Backend Server for Scan QC System
Handles file uploads, QC analysis, and PDF generation with real-time progress.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging
import fitz  # PyMuPDF for PDF text extraction

# Local imports
from qc_engine import ScanQCEngine, ProgressUpdate
from pdf_generator import generate_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Scan QC System API",
    description="AI-Powered Quality Control for Sheet Metal Parts",
    version="1.0.0"
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
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job tracking
jobs = {}


class AnalysisJob:
    """Track analysis job state"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"
        self.progress = 0
        self.stage = ""
        self.message = ""
        self.result = None
        self.error = None
        self.report_path = None
        self.pdf_path = None


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
    - tolerance: Tolerance in mm (±)
    
    Returns job_id for tracking progress.
    """
    # Validate file types
    ref_ext = Path(reference_file.filename).suffix.lower()
    scan_ext = Path(scan_file.filename).suffix.lower()
    
    if ref_ext not in ['.stl', '.obj', '.ply']:
        raise HTTPException(400, f"Reference must be STL/OBJ/PLY, got {ref_ext}")
    
    if scan_ext not in ['.stl', '.ply', '.obj', '.pcd']:
        raise HTTPException(400, f"Scan must be STL/PLY/OBJ/PCD, got {scan_ext}")
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = AnalysisJob(job_id)
    jobs[job_id] = job
    
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
    job = jobs.get(job_id)
    if not job:
        return
    
    job.status = "running"
    
    def progress_callback(update: ProgressUpdate):
        job.stage = update.stage
        job.progress = update.progress
        job.message = update.message
    
    try:
        # Initialize engine
        engine = ScanQCEngine(progress_callback=progress_callback)
        
        # Load files
        engine.load_reference(ref_path)
        engine.load_scan(scan_path)
        
        # Run analysis
        report = engine.run_analysis(
            part_id=part_id,
            part_name=part_name,
            material=material,
            tolerance=tolerance,
            drawing_context=drawing_context
        )
        
        # Convert to dict
        report_dict = report.to_dict()
        
        # Save JSON report
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        # Generate PDF
        pdf_path = output_dir / f"QC_Report_{part_id}.pdf"
        generate_pdf_report(report_dict, str(pdf_path))
        
        job.result = report_dict
        job.report_path = str(report_path)
        job.pdf_path = str(pdf_path)
        job.status = "completed"
        job.progress = 100
        job.message = "Analysis complete!"
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        job.status = "failed"
        job.error = str(e)
        job.message = f"Error: {str(e)}"


@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str):
    """Get current progress of analysis job"""
    job = jobs.get(job_id)
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
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    async def generate():
        last_progress = -1
        while job.status in ["pending", "running"]:
            if job.progress != last_progress:
                data = {
                    "status": job.status,
                    "progress": job.progress,
                    "stage": job.stage,
                    "message": job.message
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = job.progress
            await asyncio.sleep(0.2)
        
        # Final update
        data = {
            "status": job.status,
            "progress": job.progress,
            "stage": job.stage,
            "message": job.message,
            "error": job.error
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
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    if job.status == "running":
        return {"status": "running", "progress": job.progress}
    
    if job.status == "failed":
        raise HTTPException(500, f"Analysis failed: {job.error}")
    
    return {
        "status": "completed",
        "result": job.result,
        "pdf_available": job.pdf_path is not None
    }


@app.get("/api/download/{job_id}/pdf")
async def download_pdf(job_id: str):
    """Download PDF report"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    if not job.pdf_path or not Path(job.pdf_path).exists():
        raise HTTPException(404, "PDF not available")
    
    return FileResponse(
        job.pdf_path,
        media_type="application/pdf",
        filename=f"QC_Report_{job_id}.pdf"
    )


@app.get("/api/download/{job_id}/json")
async def download_json(job_id: str):
    """Download JSON report"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    if not job.report_path or not Path(job.report_path).exists():
        raise HTTPException(404, "Report not available")
    
    return FileResponse(
        job.report_path,
        media_type="application/json",
        filename=f"QC_Report_{job_id}.json"
    )


# Mount frontend static files
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
