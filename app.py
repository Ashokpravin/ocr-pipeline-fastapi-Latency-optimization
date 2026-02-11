"""
Katonic Entry Point — app.py

Katonic runs: uvicorn app:app
So this file MUST export a FastAPI 'app' object.

On first import, it installs dependencies (fixes opencv),
then imports the real FastAPI app from ocr_app.py.
"""

import subprocess
import sys
import os
import shutil
import site
import glob


def _run(cmd):
    """Run shell command silently."""
    subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _setup_dependencies():
    """One-time dependency fix inside Katonic managed image."""
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps_installed")
    if os.path.exists(marker):
        return

    print("=" * 60)
    print(" CustomOCR Pipeline — First-Run Setup")
    print("=" * 60)

    # Step 1: Remove conda-installed OpenCV (causes libGL error)
    print("[1/4] Removing conda OpenCV...")
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

    _run("conda remove --force --yes opencv-python-headless opencv-python opencv-contrib-python py-opencv libopencv 2>/dev/null")
    _run("pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null")

    # Step 2: Install requirements
    print("[2/4] Installing requirements...")
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        subprocess.run(f"pip install --no-cache-dir -r {req_file}", shell=True)

    # Step 3: paddleocr without opencv
    print("[3/4] Installing PaddleOCR...")
    _run("pip install --no-cache-dir --no-deps paddleocr==3.3.2")

    # Step 4: headless OpenCV last
    print("[4/4] Installing headless OpenCV...")
    _run("pip install --no-cache-dir --no-deps opencv-python-headless==4.12.0.88 opencv-contrib-python-headless==4.10.0.84")

    # Mark as done
    with open(marker, "w") as f:
        f.write("done")

    print("=" * 60)
    print(" Setup complete!")
    print("=" * 60)


# --- Run setup at import time (before uvicorn loads the app) ---
_setup_dependencies()

# --- Import the real FastAPI app ---
# Your actual FastAPI application must be in ocr_app.py
from ocr_app import app  # noqa: E402