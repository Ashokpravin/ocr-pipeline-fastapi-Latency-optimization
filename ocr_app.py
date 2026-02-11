"""

================================================================================

CustomOCR Pipeline API (Production-Hardened v2.2.0)

================================================================================


DESCRIPTION:

    Asynchronous document-to-markdown conversion API. Accepts PDF, Office docs,

    and images, processes them through a 3-stage pipeline (Layout Analysis → 

    Masking → OCR), and produces clean, structured Markdown output.


WORKFLOW:

    1. Client uploads file via POST /process

    2. API returns job_id immediately (non-blocking)

    3. Background worker processes: Ingestion → DLA → PageProcessor

    4. Client polls GET /job/{job_id} for status

    5. Client downloads result via GET /download/{job_id}


CONFIGURATION (Environment Variables):

    OCR_OUTPUT_DIR      : Output directory path (default: ./output)

    MAX_UPLOAD_SIZE_MB  : Maximum upload size in MB (default: 500)

    JOB_RETENTION_HOURS : Hours to keep job records (default: 24)

    MAX_WORKERS         : Background processing threads (default: 2)


DEPENDENCIES:

    - FileIngestor: Converts input files to page images

    - DLA: Document Layout Analysis (detects tables, figures)

    - PageProcessor: Masking, OCR, and markdown generation


================================================================================

"""


import os
import sys
import re
import shutil
import logging
import secrets
import uvicorn
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import aiofiles


# =============================================================================
# CONFIGURATION (Environment Variables with Sensible Defaults)
# =============================================================================

# Output directory - where all processing happens
BASE_OUTPUT_DIR = Path(os.getenv("OCR_OUTPUT_DIR", "./output")).resolve()

# File upload constraints
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Job lifecycle management
JOB_RETENTION_HOURS = int(os.getenv("JOB_RETENTION_HOURS", "24"))

# Worker pool size for background processing
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))

# Allowed file extensions (matches FileIngestor capabilities)
ALLOWED_EXTENSIONS: Set[str] = {
    # Documents
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".odp", ".odt",
    # Images
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    # Text/Code (supported by FileIngestor)
    ".json", ".xml", ".txt", ".csv", ".py", ".md", ".html", ".css", ".js"
}


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

# Create output directories
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPLETED_DIR = BASE_OUTPUT_DIR / "completed"
COMPLETED_DIR.mkdir(exist_ok=True)


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler(BASE_OUTPUT_DIR / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Thread pool for CPU-intensive pipeline tasks
executor: Optional[ThreadPoolExecutor] = None

# In-memory job tracking (auto-cleaned periodically)
job_status: Dict[str, Dict[str, Any]] = {}

# Track background cleanup task for proper shutdown
_cleanup_task: Optional[asyncio.Task] = None


# =============================================================================
# LAZY IMPORTS FOR PIPELINE MODULES
# =============================================================================
# [CHANGE] Lazy-load heavy pipeline modules so the API starts fast and
# import errors are caught gracefully instead of crashing the entire server.

_file_ingestor_cls = None
_dla_cls = None
_page_processor_cls = None
_get_optimal_worker_count = None
_cleanup_resource = None


def _load_pipeline_modules():
    """
    Lazily import heavy pipeline modules on first use.
    This prevents import-time crashes if a dependency is missing,
    and speeds up API startup.
    """
    global _file_ingestor_cls, _dla_cls, _page_processor_cls
    global _get_optimal_worker_count, _cleanup_resource

    if _file_ingestor_cls is not None:
        return  # Already loaded

    try:
        from utils import get_optimal_worker_count, cleanup_resource
        _get_optimal_worker_count = get_optimal_worker_count
        _cleanup_resource = cleanup_resource
    except ImportError as e:
        logger.error(f"Failed to import utils module: {e}")
        raise RuntimeError(f"Pipeline dependency missing: utils - {e}")

    try:
        from FileIngestor import FileIngestor
        _file_ingestor_cls = FileIngestor
    except ImportError as e:
        logger.error(f"Failed to import FileIngestor: {e}")
        raise RuntimeError(f"Pipeline dependency missing: FileIngestor - {e}")

    try:
        from DLA import DLA
        _dla_cls = DLA
    except ImportError as e:
        logger.error(f"Failed to import DLA: {e}")
        raise RuntimeError(f"Pipeline dependency missing: DLA - {e}")

    try:
        from PageProcessor import PageProcessor
        _page_processor_cls = PageProcessor
    except ImportError as e:
        logger.error(f"Failed to import PageProcessor: {e}")
        raise RuntimeError(f"Pipeline dependency missing: PageProcessor - {e}")

    logger.info("All pipeline modules loaded successfully")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Removes potentially dangerous characters from filename.
    Prevents path traversal attacks and filesystem issues.
    """
    # Remove path separators and null bytes
    filename = os.path.basename(filename)
    filename = filename.replace("\x00", "")

    # Replace problematic characters with underscore
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)

    # [CHANGE] Remove leading/trailing dots and spaces (Windows safety + hidden file prevention)
    filename = filename.strip('. ')

    # Limit length (filesystem safety)
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]

    # [CHANGE] Fallback if filename becomes empty after sanitization
    if not name:
        name = f"upload_{secrets.token_hex(4)}"

    return f"{name}{ext}"


def validate_file(file: UploadFile, content_length: int) -> None:
    """
    Validates uploaded file before processing.
    Raises HTTPException if validation fails.
    """
    # [CHANGE] Check for zero-byte files
    if content_length == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded. File must have content.")

    # Check file size
    if content_length > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,  # [CHANGE] Use proper 413 status code for payload too large
            detail=f"File too large ({content_length / 1024 / 1024:.1f}MB). Maximum size: {MAX_UPLOAD_SIZE_MB}MB"
        )

    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def validate_job_id(job_id: str) -> None:
    """
    [NEW] Validate job_id format to prevent injection/path traversal attacks.
    Job IDs are always 16-char hex strings generated by secrets.token_hex(8).
    """
    if not job_id or not re.fullmatch(r'[a-f0-9]{16}', job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")


def cleanup_old_jobs() -> int:
    """
    Removes job records older than JOB_RETENTION_HOURS.
    Also cleans up orphaned result files.
    Returns count of removed jobs.
    """
    cutoff = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
    expired_jobs = [
        job_id for job_id, info in job_status.items()
        if info.get("created_at", datetime.now()) < cutoff
    ]

    for job_id in expired_jobs:
        job_info = job_status[job_id]
        # [CHANGE] Also clean up result files for expired jobs
        result_path = job_info.get("result_path")
        if result_path and os.path.exists(result_path):
            try:
                os.remove(result_path)
                logger.info(f"Removed expired result file: {result_path}")
            except OSError as e:
                logger.warning(f"Failed to remove expired result file {result_path}: {e}")

        del job_status[job_id]

    if expired_jobs:
        logger.info(f"Cleaned up {len(expired_jobs)} expired job records")

    return len(expired_jobs)


# =============================================================================
# FASTAPI APPLICATION SETUP (Using Lifespan - modern pattern)
# =============================================================================

def detect_root_path() -> str:
    """Detects workspace proxy path for Katonic deployment (Swagger UI fix)."""
    route = os.getenv("ROUTE", "")
    if route:
        if not route.startswith("/"):
            route = "/" + route
        if not route.endswith("/"):
            route = route + "/"
        return route
    return ""


# [CHANGE] Use modern lifespan context manager instead of deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    global executor, _cleanup_task

    # --- STARTUP ---
    logger.info("=" * 60)
    logger.info("CustomOCR Pipeline API v2.2.0 starting...")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Output Directory  : {BASE_OUTPUT_DIR}")
    logger.info(f"Completed Dir     : {COMPLETED_DIR}")
    logger.info(f"Max Upload Size   : {MAX_UPLOAD_SIZE_MB} MB")
    logger.info(f"Max Workers       : {MAX_WORKERS}")
    logger.info(f"Job Retention     : {JOB_RETENTION_HOURS} hours")
    logger.info(f"Allowed Extensions: {len(ALLOWED_EXTENSIONS)} types")

    # Verify output directory is writable
    try:
        test_file = BASE_OUTPUT_DIR / ".write_test"
        test_file.touch()
        test_file.unlink()
        logger.info("Output directory: Writable ✓")
    except Exception as e:
        logger.error(f"Output directory not writable: {e}")
        raise RuntimeError(f"Cannot write to output directory: {BASE_OUTPUT_DIR}")

    # Initialize thread pool executor
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    logger.info(f"Thread pool initialized with {MAX_WORKERS} workers")

    # Cleanup old jobs from previous runs
    cleanup_old_jobs()

    # Start periodic cleanup background task
    _cleanup_task = asyncio.create_task(periodic_job_cleanup())

    logger.info("=" * 60)
    logger.info("API Ready. Accepting requests.")
    logger.info("=" * 60)

    yield  # Application is running

    # --- SHUTDOWN ---
    logger.info("CustomOCR Pipeline API shutting down...")

    # Cancel periodic cleanup task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass

    # Shutdown thread pool gracefully
    if executor:
        executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Thread pool shut down")

    logger.info("Shutdown complete.")


app = FastAPI(
    title="CustomOCR Pipeline API",
    description=(
        "Production-grade OCR pipeline with Layout Analysis, Masking, and Enrichment. "
        "Upload documents (PDF, Office, Images) and receive structured Markdown output. "
        "All processing is asynchronous — returns job ID immediately."
    ),
    version="2.2.0",
    root_path=detect_root_path(),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# [NEW] GLOBAL EXCEPTION HANDLER
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler. Prevents raw stack traces from leaking
    to clients in production. Logs full traceback server-side.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again later.",
            "error_type": type(exc).__name__
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    [NEW] Explicit HTTP exception handler with structured logging.
    """
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    elif exc.status_code >= 400:
        logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class JobResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str
    download_url: Optional[str] = None
    created_at: datetime


# =============================================================================
# CORE PIPELINE EXECUTION
# =============================================================================

def process_ocr_pipeline(job_id: str, file_path: Path, job_dir: Path):
    """
    Executes the 3-Step CustomOCR Pipeline in background thread.

    Steps:
        1. Ingestion (FileIngestor) - Convert input to page images
        2. Layout Analysis (DLA) - Detect tables, figures, formulas
        3. Processing (PageProcessor) - Masking, OCR, and enrichment
    """
    step_name = "initialization"
    try:
        # [CHANGE] Lazy-load pipeline modules on first job execution
        _load_pipeline_modules()

        job_status[job_id]["status"] = "processing"
        job_status[job_id]["message"] = "Starting OCR pipeline..."
        job_status[job_id]["updated_at"] = datetime.now()

        # Calculate optimal workers dynamically
        optimal_workers = _get_optimal_worker_count(ram_per_worker_gb=1.5, system_reserve_gb=4.0)
        logger.info(f"[Job {job_id}] Starting pipeline with {optimal_workers} workers")

        # --- STEP 1: File Ingestion ---
        step_name = "file_ingestion"
        logger.info(f"[Job {job_id}] Step 1/3 - Ingesting file...")
        job_status[job_id]["message"] = "Step 1/3: Ingesting file..."
        job_status[job_id]["updated_at"] = datetime.now()

        ingestor = _file_ingestor_cls(str(job_dir))
        project_dir, image_paths = ingestor.process_input(file_path)

        # [CHANGE] Validate ingestion output before proceeding
        if not image_paths:
            raise ValueError("File ingestion produced no page images. The file may be empty or corrupted.")

        # --- STEP 2: Document Layout Analysis ---
        step_name = "layout_analysis"
        logger.info(f"[Job {job_id}] Step 2/3 - Analyzing document layout ({len(image_paths)} pages)...")
        job_status[job_id]["message"] = f"Step 2/3: Analyzing layout ({len(image_paths)} pages)..."
        job_status[job_id]["updated_at"] = datetime.now()

        dla = _dla_cls()
        dla.run_vision_pipeline(image_paths, project_dir, filter_dup=True, merge_visual=False)

        # Cleanup intermediate files to save disk space
        try:
            _cleanup_resource(project_dir / "labeled", force_cleanup=True)
            intermediate_pdf = project_dir / file_path.with_suffix(".pdf").name
            if intermediate_pdf.exists():
                _cleanup_resource(intermediate_pdf, force_cleanup=True)
        except Exception as cleanup_err:
            # [CHANGE] Don't fail the job for intermediate cleanup failures
            logger.warning(f"[Job {job_id}] Intermediate cleanup warning: {cleanup_err}")

        # --- STEP 3: Masking & OCR ---
        step_name = "ocr_processing"
        logger.info(f"[Job {job_id}] Step 3/3 - Running OCR and enrichment...")
        job_status[job_id]["message"] = "Step 3/3: Performing OCR and enrichment..."
        job_status[job_id]["updated_at"] = datetime.now()

        page_processor = _page_processor_cls(str(project_dir), max_workers=optimal_workers)
        page_processor.process_and_mask()
        final_md_path = page_processor.generate_final_markdown()

        # [CHANGE] Validate output file exists and has content
        if not final_md_path or not Path(final_md_path).exists():
            raise FileNotFoundError("Pipeline completed but no output markdown file was generated")
        if Path(final_md_path).stat().st_size == 0:
            logger.warning(f"[Job {job_id}] Output markdown is empty (0 bytes)")

        # Copy result to completed directory
        step_name = "result_copy"
        completed_md_path = COMPLETED_DIR / f"{job_id}_{file_path.stem}.md"
        shutil.copy2(final_md_path, completed_md_path)

        # Mark job as completed
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["message"] = "Processing completed successfully"
        job_status[job_id]["result_path"] = str(completed_md_path)
        job_status[job_id]["download_url"] = f"/download/{job_id}"
        job_status[job_id]["updated_at"] = datetime.now()

        logger.info(f"[Job {job_id}] ✓ Completed. Output: {completed_md_path}")

        # Cleanup processing directory (keep only final result)
        cleanup_job_directory(job_dir)

    except Exception as e:
        # [CHANGE] Log full traceback and include step info in error message
        error_msg = str(e)
        logger.error(
            f"[Job {job_id}] Failed at step '{step_name}': {error_msg}\n"
            f"{traceback.format_exc()}"
        )

        # [CHANGE] Guard against job_status key being missing (race condition)
        if job_id in job_status:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["message"] = f"Failed at {step_name}: {error_msg}"
            job_status[job_id]["error_step"] = step_name
            job_status[job_id]["updated_at"] = datetime.now()

        # Cleanup on failure
        cleanup_job_directory(job_dir)


def cleanup_job_directory(directory: Path):
    """Safely removes temporary processing directory."""
    try:
        if directory.exists():
            shutil.rmtree(directory)
            logger.info(f"Cleaned up: {directory.name}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {directory}: {e}")


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/health")
def health_check():
    """
    [CHANGE] Renamed from / to /health for proper health check semantics.
    Root / redirects here. Includes disk space check.
    """
    # [NEW] Check disk space availability
    disk_warning = None
    try:
        disk_usage = shutil.disk_usage(BASE_OUTPUT_DIR)
        free_gb = disk_usage.free / (1024 ** 3)
        if free_gb < 1.0:
            disk_warning = f"Low disk space: {free_gb:.2f}GB remaining"
            logger.warning(disk_warning)
    except OSError:
        disk_warning = "Unable to check disk space"

    return {
        "status": "healthy" if not disk_warning else "degraded",
        "service": "CustomOCR Pipeline API",
        "version": "2.2.0",
        "message": "Use POST /process to submit documents. All processing is asynchronous.",
        "docs_url": f"{detect_root_path()}docs",
        "config": {
            "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
            "max_workers": MAX_WORKERS,
            "job_retention_hours": JOB_RETENTION_HOURS
        },
        "stats": {
            "active_jobs": len([j for j in job_status.values() if j.get("status") in ["queued", "processing"]]),
            "completed_jobs": len([j for j in job_status.values() if j.get("status") == "completed"]),
            "failed_jobs": len([j for j in job_status.values() if j.get("status") == "failed"])
        },
        "warnings": [disk_warning] if disk_warning else []
    }


@app.get("/",include_in_schema=False)
def root():
    """Root endpoint - same as health check."""
    return health_check()


@app.post("/process", response_model=JobResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Upload a document for OCR processing.

    Accepts: PDF, Word, PowerPoint, Excel, Images, Text files
    Returns: Job ID for status tracking

    Use /job/{job_id} to check status and /download/{job_id} to get result.
    """
    job_id = None
    job_dir = None

    try:
        # Read file content
        try:
            content = await file.read()
        except Exception as e:
            logger.error(f"Failed to read uploaded file: {e}")
            raise HTTPException(status_code=400, detail="Failed to read uploaded file. It may be corrupted.")
        finally:
            # [CHANGE] Always close the upload file handle
            await file.close()

        content_length = len(content)

        # Validate file before processing
        validate_file(file, content_length)

        # Generate job ID and create working directory
        job_id = secrets.token_hex(8)
        job_dir = BASE_OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        # Sanitize and save uploaded file
        safe_filename = sanitize_filename(file.filename)
        input_path = job_dir / safe_filename

        async with aiofiles.open(input_path, "wb") as f:
            await f.write(content)

        # [CHANGE] Verify file was written correctly
        written_size = input_path.stat().st_size
        if written_size != content_length:
            raise HTTPException(
                status_code=500,
                detail="File save verification failed. Please retry."
            )

        logger.info(f"[Job {job_id}] Received: {safe_filename} ({content_length / 1024 / 1024:.2f} MB)")

        # Initialize job tracking
        job_status[job_id] = {
            "status": "queued",
            "filename": safe_filename,
            "original_filename": file.filename,
            "file_size_bytes": content_length,
            "message": "Job queued for processing",
            "result_path": None,
            "download_url": None,
            "error_step": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        # [CHANGE] Verify executor is available before submitting
        if executor is None:
            raise HTTPException(status_code=503, detail="Service not ready. Please try again shortly.")

        # Submit to background worker pool
        future: Future = executor.submit(process_ocr_pipeline, job_id, input_path, job_dir)

        # [CHANGE] Add a callback to catch unexpected thread-level errors
        def _on_done(fut: Future):
            exc = fut.exception()
            if exc and job_id in job_status and job_status[job_id]["status"] not in ("completed", "failed"):
                logger.error(f"[Job {job_id}] Thread-level exception: {exc}")
                job_status[job_id]["status"] = "failed"
                job_status[job_id]["message"] = f"Unexpected error: {str(exc)}"
                job_status[job_id]["updated_at"] = datetime.now()

        future.add_done_callback(_on_done)

        return JobResponse(
            job_id=job_id,
            status="queued",
            filename=safe_filename,
            message="Document accepted for processing. Use the job ID to check status.",
            download_url=f"/job/{job_id}",
            created_at=job_status[job_id]["created_at"]
        )

    except HTTPException:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        # Cleanup on unexpected failure
        if job_dir and job_dir.exists():
            cleanup_job_directory(job_dir)
        # [CHANGE] Remove partial job status entry on failure
        if job_id and job_id in job_status:
            del job_status[job_id]
        logger.error(f"Upload failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the current status of a processing job."""
    # [CHANGE] Validate job_id format
    validate_job_id(job_id)

    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = job_status[job_id]

    response_data = {
        "job_id": job_id,
        "status": job_info["status"],
        "filename": job_info["filename"],
        "file_size_mb": round(job_info.get("file_size_bytes", 0) / 1024 / 1024, 2),
        "message": job_info["message"],
        "created_at": job_info["created_at"],
        "updated_at": job_info.get("updated_at", job_info["created_at"])
    }

    if job_info["status"] == "completed":
        response_data["download_url"] = f"/download/{job_id}"
        result_path = job_info.get("result_path")
        if result_path and os.path.exists(result_path):
            response_data["result_size_bytes"] = os.path.getsize(result_path)

    # [CHANGE] Include error step info for failed jobs
    if job_info["status"] == "failed" and job_info.get("error_step"):
        response_data["error_step"] = job_info["error_step"]

    return response_data


@app.get("/download/{job_id}")
async def download_markdown(job_id: str):
    """Download the markdown result for a completed job."""
    # [CHANGE] Validate job_id format
    validate_job_id(job_id)

    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = job_status[job_id]

    if job_info["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job still processing. Please wait.")

    if job_info["status"] == "queued":
        raise HTTPException(status_code=202, detail="Job queued. Processing has not started yet.")

    if job_info["status"] == "failed":
        raise HTTPException(status_code=400, detail=f"Job failed: {job_info['message']}")

    # Locate result file
    result_path = None
    if job_info.get("result_path") and os.path.exists(job_info["result_path"]):
        result_path = job_info["result_path"]
    else:
        # Fallback: search completed directory
        matching_files = list(COMPLETED_DIR.glob(f"{job_id}_*.md"))
        if matching_files:
            result_path = str(matching_files[0])

    if not result_path or not os.path.exists(result_path):
        # [CHANGE] Log the missing file scenario
        logger.error(f"[Job {job_id}] Result file missing: {job_info.get('result_path')}")
        raise HTTPException(status_code=404, detail="Result file not found. It may have been cleaned up.")

    filename = f"{job_id}_{job_info['filename']}.md"

    return FileResponse(
        path=result_path,
        filename=filename,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """
    List all jobs with optional status filter.

    Filter options: queued, processing, completed, failed
    """
    # [CHANGE] Validate status parameter
    valid_statuses = {"queued", "processing", "completed", "failed"}
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status filter: '{status}'. Must be one of: {', '.join(sorted(valid_statuses))}"
        )

    # [CHANGE] Clamp limit to a reasonable range
    limit = max(1, min(limit, 500))

    jobs_list = []

    for job_id, job_info in sorted(
        job_status.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    ):
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
        "total_jobs": len(job_status),
        "returned": len(jobs_list),
        "jobs": jobs_list
    }


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job record and its result file (if exists)."""
    # [CHANGE] Validate job_id format
    validate_job_id(job_id)

    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = job_status[job_id]

    # Prevent deletion of active jobs
    if job_info["status"] in ["queued", "processing"]:
        raise HTTPException(status_code=409, detail="Cannot delete active job. Wait for completion or failure.")

    # Delete result file if exists
    if job_info.get("result_path") and os.path.exists(job_info["result_path"]):
        try:
            os.remove(job_info["result_path"])
            logger.info(f"[Job {job_id}] Result file deleted")
        except OSError as e:
            logger.warning(f"[Job {job_id}] Failed to delete result file: {e}")

    # Remove from tracking
    del job_status[job_id]

    return {"message": f"Job {job_id} deleted successfully"}


# =============================================================================
# BACKGROUND TASK: Periodic Job Cleanup
# =============================================================================

async def periodic_job_cleanup():
    """Runs every hour to cleanup expired job records."""
    while True:
        try:
            await asyncio.sleep(3600)  # 1 hour
            cleanup_old_jobs()
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
            break
        except Exception as e:
            # [CHANGE] Don't let cleanup errors crash the background task
            logger.error(f"Periodic cleanup error: {e}")
            await asyncio.sleep(60)  # Wait a bit before retrying
