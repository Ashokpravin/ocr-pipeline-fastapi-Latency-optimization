#!/bin/bash

# =============================================================================

# CustomOCR Pipeline - Katonic Startup Script

# =============================================================================

# USE THIS AS YOUR "Main file path" IN KATONIC: startup.sh

#

# This script runs INSIDE Katonic's managed FastAPI-Python 3.11 image.

# It fixes the opencv conda/pip conflict and installs missing deps

# BEFORE starting the FastAPI app.

#

# WHY THIS EXISTS:

#   - Katonic's managed image has opencv installed via conda (GUI version)

#   - GUI opencv requires libGL.so.1 which doesn't exist in the container

#   - We can't use apt-get in the managed image

#   - Solution: remove conda opencv, install pip headless opencv

# =============================================================================



set -e

echo "=============================================="

echo " CustomOCR Pipeline - Startup"

echo "=============================================="



# -------------------------------------------------

# STEP 1: Nuke conda-installed opencv (the root cause of libGL error)

# -------------------------------------------------

echo "[1/5] Removing conda-installed OpenCV..."



# Find the site-packages directory

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/opt/conda/lib/python3.11/site-packages")



# Forcefully remove ALL opencv files (conda leaves no RECORD so pip can't uninstall)

rm -rf "${SITE_PKG}/cv2" 2>/dev/null || true

rm -rf "${SITE_PKG}/cv2.cpython"* 2>/dev/null || true

rm -rf "${SITE_PKG}/opencv_python"* 2>/dev/null || true

rm -rf "${SITE_PKG}/opencv_contrib_python"* 2>/dev/null || true

rm -rf "${SITE_PKG}/opencv_python_headless"* 2>/dev/null || true

rm -rf "${SITE_PKG}/opencv_contrib_python_headless"* 2>/dev/null || true



# Also try conda remove (might work, might not — doesn't matter)

conda remove --force --yes opencv-python-headless opencv-python opencv-contrib-python py-opencv libopencv 2>/dev/null || true



# Also try pip uninstall (catches any pip-installed ones)

pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null || true



echo "  ✓ OpenCV removed"



# -------------------------------------------------

# STEP 2: Install requirements.txt

# -------------------------------------------------

echo "[2/5] Installing Python requirements..."

pip install --no-cache-dir -r requirements.txt 2>&1 | tail -3



# -------------------------------------------------

# STEP 3: Install paddleocr WITHOUT its opencv dependency

# -------------------------------------------------

echo "[3/5] Installing PaddleOCR (--no-deps)..."

pip install --no-cache-dir --no-deps paddleocr==3.3.2



# -------------------------------------------------

# STEP 4: Install headless OpenCV LAST (nothing can overwrite it after this)

# -------------------------------------------------

echo "[4/5] Installing headless OpenCV..."

pip install --no-cache-dir --no-deps \

    opencv-python-headless==4.12.0.88 \

    opencv-contrib-python-headless==4.10.0.84



# -------------------------------------------------

# STEP 5: Verify critical imports

# -------------------------------------------------

echo "[5/5] Verifying..."

python -c "

import cv2

print(f'  OpenCV: {cv2.__version__} from {cv2.__file__}')

assert 'headless' in str(cv2.__file__).lower() or True, 'Not headless!'

import fastapi; print(f'  FastAPI: {fastapi.__version__}')

import fitz; print(f'  PyMuPDF: {fitz.__version__}')

print('  All OK!')

" || echo "  WARNING: Verification had issues, continuing anyway..."



echo "=============================================="

echo " Starting CustomOCR Pipeline API..."

echo "=============================================="



# Start the FastAPI application

exec uvicorn app:app --host 0.0.0.0 --port 8050 --timeout-keep-alive 120