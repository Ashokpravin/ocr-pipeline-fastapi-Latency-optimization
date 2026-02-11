import os
import sys
import shutil
import logging
import secrets
import uvicorn
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import aiofiles

# Import Custom Modules
from utils import get_optimal_worker_count, cleanup_resource
from FileIngestor import FileIngestor
from DLA import DLA
from PageProcessor import PageProcessor

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# The specific token provided for Katonic API Management
VALID_API_TOKEN = "REQUEST_API-FASTAPI-OCR eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZWVkMDdlZWY0NmY0MjgyYjgxYjUwOTBiNjQ5NDQ3ZmthdG9uaWMiLCJleHAiOjMzMjk0OTAyNzM0MTI5fQ.19u3omsppcrSKQojCm5aQ7VTtlFi32X7LSV3cBi7UzQ"

# Temporary storage for processing - CREATING OUTPUT DIRECTORIES
BASE_OUTPUT_DIR = Path("/home/katonic/.paddlex/official_models/PP-DocLayout_plus-L/custom-ocr-pipeline/output")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
COMPLETED_DIR = BASE_OUTPUT_DIR / "completed"
COMPLETED_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_OUTPUT_DIR / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global executor for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Job tracking
job_status: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# FASTAPI SETUP
# ============================================================================

# Detect Katonic Workspace Route for Swagger UI Fix
def detect_root_path():
    """Detects the workspace proxy path to ensure /docs works in Katonic."""
    route = os.getenv("ROUTE", "")
    if route:
        if not route.startswith("/"):
            route = "/" + route
        if not route.endswith("/"):
            route = route + "/"
        return route
    return ""

app = FastAPI(
    title="CustomOCR Pipeline API",
    description="Production-grade OCR pipeline with Layout Analysis, Masking, and Enrichment. All processing is asynchronous - returns job ID immediately.",
    version="2.0.0",
    root_path=detect_root_path()  # CRITICAL for Katonic Swagger UI
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUTHENTICATION
# ============================================================================

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validates the Bearer token against the provided Katonic token."""
    token = credentials.credentials
    if token != VALID_API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authentication Token"
        )
    return token

# ============================================================================
# MODELS
# ============================================================================

class JobResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str
    download_url: Optional[str] = None
    created_at: datetime

# ============================================================================
# CORE PIPELINE LOGIC
# ============================================================================

def process_ocr_pipeline(job_id: str, file_path: Path, job_dir: Path):
    """
    Executes the 3-Step CustomOCR Pipeline in background thread.
    1. Ingestion -> 2. DLA -> 3. PageProcessor (Masking & OCR)
    """
    try:
        job_status[job_id]["status"] = "processing"
        job_status[job_id]["message"] = "Starting OCR pipeline..."
        
        # Calculate optimal workers dynamically
        optimal_workers = get_optimal_worker_count(ram_per_worker_gb=1.5, system_reserve_gb=4.0)
        logger.info(f"🚀 Starting Pipeline for job {job_id} with {optimal_workers} workers")

        # --- STEP 1: Ingestion ---
        logger.info(f">>> Job {job_id}: Step 1 - Ingesting File...")
        job_status[job_id]["message"] = "Step 1: Ingesting file..."
        ingestor = FileIngestor(str(job_dir))
        project_dir, image_paths = ingestor.process_input(file_path)

        # --- STEP 2: Layout Analysis (DLA) ---
        logger.info(f">>> Job {job_id}: Step 2 - Layout Analysis...")
        job_status[job_id]["message"] = "Step 2: Analyzing document layout..."
        dla = DLA()
        dla.run_vision_pipeline(image_paths, project_dir, filter_dup=True, merge_visual=False)

        # Clean intermediate files to save space
        cleanup_resource(project_dir / "labeled", force_cleanup=True)
        intermediate_pdf = project_dir / file_path.with_suffix(".pdf").name
        cleanup_resource(intermediate_pdf, force_cleanup=True)

        # --- STEP 3: Masking & OCR ---
        logger.info(f">>> Job {job_id}: Step 3 - Masking & OCR...")
        job_status[job_id]["message"] = "Step 3: Performing OCR and enrichment..."
        page_processor = PageProcessor(str(project_dir), max_workers=optimal_workers)
        
        # Create white boxes over tables/figures
        page_processor.process_and_mask()
        
        # Generate final markdown (Main OCR + Enrichment)
        final_md_path = page_processor.generate_final_markdown()

        # Copy final markdown to completed directory
        completed_md_path = COMPLETED_DIR / f"{job_id}_{file_path.stem}.md"
        shutil.copy2(final_md_path, completed_md_path)

        # Update job status
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["message"] = "Processing completed successfully"
        job_status[job_id]["result_path"] = str(completed_md_path)
        job_status[job_id]["download_url"] = f"/download/{job_id}"
        job_status[job_id]["updated_at"] = datetime.now()

        logger.info(f"✅ Job {job_id} completed. Output saved to {completed_md_path}")

        # Cleanup processing directory
        cleanup_request_files(job_dir)

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["message"] = f"Processing failed: {str(e)}"
        job_status[job_id]["updated_at"] = datetime.now()
        
        # Cleanup on failure
        cleanup_request_files(job_dir)

def cleanup_request_files(directory: Path):
    """Background task to remove temp files after processing completes."""
    try:
        if directory.exists():
            shutil.rmtree(directory)
            logger.info(f"🧹 Cleaned up: {directory}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {directory}: {e}")

# ============================================================================
# ENDPOINTS - SIMPLIFIED
# ============================================================================

@app.get("/")
def health_check():
    return {
        "status": "active",
        "service": "CustomOCR Pipeline API",
        "version": "2.0.0",
        "message": "Use POST /process to submit documents. All processing is asynchronous.",
        "docs_url": f"{detect_root_path()}docs",
        "active_jobs": len([j for j in job_status.values() if j.get("status") in ["queued", "processing"]]),
        "completed_jobs": len([j for j in job_status.values() if j.get("status") == "completed"])
    }

@app.post("/process", response_model=JobResponse)
async def process_document(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """
    **Process a Document**
    
    Uploads a file (PDF, Image, Docx), starts the CustomOCR pipeline asynchronously,
    and returns a job ID immediately.
    
    Use /job/{job_id} to check status and /download/{job_id} to get result.
    """
    
    job_id = secrets.token_hex(8)
    job_dir = BASE_OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    try:
        # Save Uploaded File
        input_path = job_dir / file.filename
        async with aiofiles.open(input_path, "wb") as f:
            content = await file.read()
            await f.write(content)
            
        logger.info(f"📥 Received: {file.filename} ({len(content)} bytes) for job {job_id}")

        # Initialize job status
        job_status[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "message": "Job queued for processing",
            "result_path": None,
            "download_url": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        # Start processing in background thread
        executor.submit(process_ocr_pipeline, job_id, input_path, job_dir)

        # Return immediate response with job ID
        return JobResponse(
            job_id=job_id,
            status="queued",
            filename=file.filename,
            message="Document accepted for processing. Use the job ID to check status.",
            download_url=f"/job/{job_id}",
            created_at=job_status[job_id]["created_at"]
        )

    except Exception as e:
        # Cleanup on fail
        if job_dir.exists():
            cleanup_request_files(job_dir)
        logger.error(f"❌ Failed to accept job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
async def get_job_status(job_id: str, token: str = Depends(verify_token)):
    """Get the status of a processing job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_status[job_id]
    
    response_data = {
        "job_id": job_id,
        "status": job_info["status"],
        "filename": job_info["filename"],
        "message": job_info["message"],
        "created_at": job_info["created_at"],
        "updated_at": job_info.get("updated_at", job_info["created_at"])
    }
    
    if job_info["status"] == "completed":
        response_data["download_url"] = f"/download/{job_id}"
        if job_info.get("result_path") and os.path.exists(job_info["result_path"]):
            response_data["file_size"] = os.path.getsize(job_info["result_path"])
    
    return response_data

@app.get("/download/{job_id}")
async def download_markdown(job_id: str, token: str = Depends(verify_token)):
    """Download the markdown file for a completed job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_status[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Status: {job_info['status']}")
    
    # Try to find the file
    result_path = None
    if job_info.get("result_path") and os.path.exists(job_info["result_path"]):
        result_path = job_info["result_path"]
    else:
        # Search for the file in completed directory
        matching_files = list(COMPLETED_DIR.glob(f"{job_id}_*.md"))
        if matching_files:
            result_path = matching_files[0]
    
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    filename = f"{job_id}_{job_info['filename']}.md"
    
    return FileResponse(
        path=result_path,
        filename=filename,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.get("/jobs")
async def list_jobs(token: str = Depends(verify_token), 
                   status: Optional[str] = None,
                   limit: int = 50):
    """List all jobs with optional status filter."""
    jobs_list = []
    
    for job_id, job_info in sorted(job_status.items(), 
                                   key=lambda x: x[1]["created_at"], 
                                   reverse=True):
        
        if status and job_info["status"] != status:
            continue
            
        job_data = {
            "job_id": job_id,
            "status": job_info["status"],
            "filename": job_info["filename"],
            "message": job_info["message"],
            "created_at": job_info["created_at"],
            "updated_at": job_info.get("updated_at", job_info["created_at"])
        }
        
        if job_info["status"] == "completed":
            job_data["download_url"] = f"/download/{job_id}"
        
        jobs_list.append(job_data)
        
        if len(jobs_list) >= limit:
            break
    
    return {
        "total_jobs": len(jobs_list),
        "jobs": jobs_list
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("🚀 CustomOCR Pipeline API starting up...")
    logger.info(f"📁 Output directory: {BASE_OUTPUT_DIR}")
    logger.info(f"📁 Completed directory: {COMPLETED_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 CustomOCR Pipeline API shutting down...")
    executor.shutdown(wait=False)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8051"))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=120
    )