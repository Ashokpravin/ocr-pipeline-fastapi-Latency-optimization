"""
Katonic Entry Point — app.py

This file is the ONLY file Katonic allows as main file path.
It handles dependency installation on first run, then starts the real FastAPI app.

WHAT THIS DOES:
  1. Removes conda-installed GUI OpenCV (causes libGL.so.1 error)
  2. Installs requirements.txt
  3. Installs paddleocr with --no-deps (prevents GUI opencv)
  4. Installs headless OpenCV last
  5. Starts uvicorn with the real app from ocr_app.py
"""

import subprocess
import sys
import os
import shutil
import site

def run(cmd, ignore_errors=False):
    """Run a shell command, optionally ignoring failures."""
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and not ignore_errors:
        print(f"    WARN: {result.stderr.strip()[:200]}")
    return result.returncode == 0

def setup_dependencies():
    """One-time dependency setup inside Katonic managed image."""

    # Check if setup already done (marker file)
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps_installed")
    if os.path.exists(marker):
        print("[Setup] Dependencies already installed, skipping...")
        return

    print("=" * 60)
    print(" CustomOCR Pipeline — First-Run Dependency Setup")
    print("=" * 60)

    # --- Step 1: Remove conda-installed OpenCV ---
    print("\n[1/4] Removing conda-installed OpenCV...")

    # Find site-packages
    try:
        site_pkg = site.getsitepackages()[0]
    except Exception:
        site_pkg = "/opt/conda/lib/python3.11/site-packages"

    # Delete opencv files directly (conda packages have no RECORD, pip can't remove them)
    for pattern in [
        "cv2", "cv2.cpython*",
        "opencv_python*", "opencv_contrib_python*",
        "opencv_python_headless*", "opencv_contrib_python_headless*"
    ]:
        import glob
        for path in glob.glob(os.path.join(site_pkg, pattern)):
            print(f"    Removing: {os.path.basename(path)}")
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)

    # Also try standard uninstall methods
    run("conda remove --force --yes opencv-python-headless opencv-python opencv-contrib-python py-opencv libopencv 2>/dev/null", ignore_errors=True)
    run("pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null", ignore_errors=True)
    print("  ✓ OpenCV removed")

    # --- Step 2: Install requirements.txt ---
    print("\n[2/4] Installing requirements.txt...")
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        run(f"pip install --no-cache-dir -r {req_file}")
    else:
        print("  WARNING: requirements.txt not found!")

    # --- Step 3: Install paddleocr without opencv dependency ---
    print("\n[3/4] Installing PaddleOCR (--no-deps)...")
    run("pip install --no-cache-dir --no-deps paddleocr==3.3.2")

    # --- Step 4: Install headless OpenCV LAST ---
    print("\n[4/4] Installing headless OpenCV...")
    run("pip install --no-cache-dir --no-deps opencv-python-headless==4.12.0.88 opencv-contrib-python-headless==4.10.0.84")

    # --- Verify ---
    print("\n[Verify] Checking imports...")
    try:
        verify_code = """
import cv2; print(f'  OpenCV: {cv2.__version__}')
import fastapi; print(f'  FastAPI: {fastapi.__version__}')
import fitz; print(f'  PyMuPDF: {fitz.__version__}')
print('  All OK!')
"""
        subprocess.run([sys.executable, "-c", verify_code], check=True)
    except Exception as e:
        print(f"  Verification warning: {e}")

    # Create marker so we don't repeat on restart
    with open(marker, "w") as f:
        f.write("installed")

    print("\n" + "=" * 60)
    print(" Setup complete! Starting API...")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Step 1: Install dependencies (first run only)
    setup_dependencies()

    # Step 2: Start the real FastAPI app
    import uvicorn
    uvicorn.run(
        "ocr_app:app",  # Import from ocr_app.py
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8050")),
        timeout_keep_alive=120,
        log_level="info"
    )