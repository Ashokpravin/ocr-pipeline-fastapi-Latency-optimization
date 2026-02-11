"""
Katonic Entry Point — app.py
Katonic runs: uvicorn app:app
This file MUST have an 'app' variable at module level.
"""

import subprocess
import sys
import os
import shutil
import site
import glob
import traceback


def _run(cmd):
    subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _setup_dependencies():
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps_installed")
    if os.path.exists(marker):
        return

    print("=" * 60)
    print(" CustomOCR — First-Run Setup")
    print("=" * 60, flush=True)

    # Step 1: Remove conda opencv
    print("[1/4] Removing conda OpenCV...", flush=True)
    try:
        site_pkg = site.getsitepackages()[0]
    except Exception:
        site_pkg = "/opt/conda/lib/python3.11/site-packages"

    for pattern in ["cv2", "cv2.cpython*", "opencv_python*", "opencv_contrib_python*",
                    "opencv_python_headless*", "opencv_contrib_python_headless*"]:
        for path in glob.glob(os.path.join(site_pkg, pattern)):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    _run("pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null")

    # Step 2: paddleocr without deps
    print("[2/4] Installing PaddleOCR...", flush=True)
    _run("pip install --no-cache-dir --no-deps paddleocr==3.3.2")

    # Step 3: headless opencv
    print("[3/4] Installing headless OpenCV...", flush=True)
    _run("pip install --no-cache-dir --no-deps opencv-python-headless==4.12.0.88 opencv-contrib-python-headless==4.10.0.84")

    # Step 4: verify
    print("[4/4] Verifying cv2...", flush=True)
    result = subprocess.run([sys.executable, "-c", "import cv2; print(f'OpenCV {cv2.__version__}')"],
                            capture_output=True, text=True)
    print(f"  {result.stdout.strip()}", flush=True)
    if result.returncode != 0:
        print(f"  WARNING: {result.stderr.strip()}", flush=True)

    with open(marker, "w") as f:
        f.write("done")
    print("Setup complete!", flush=True)


# =============================================
# RUN SETUP BEFORE ANYTHING ELSE
# =============================================
_setup_dependencies()

# =============================================
# IMPORT THE REAL FASTAPI APP
# =============================================
# This variable MUST be named 'app' — uvicorn looks for it
app = None

try:
    from ocr_app import app
    print(f"[app.py] Loaded ocr_app.app OK: {type(app)}", flush=True)
except Exception as e:
    print(f"[app.py] FAILED to import ocr_app: {e}", flush=True)
    print(traceback.format_exc(), flush=True)

    # Fallback app so container stays alive and you can see the error
    from fastapi import FastAPI
    app = FastAPI(title="CustomOCR - Import Error")

    _import_error = str(e)

    @app.get("/")
    def show_error():
        return {
            "status": "import_error",
            "error": _import_error,
            "fix": "Check ocr_app.py exists and all its dependencies are installed"
        }