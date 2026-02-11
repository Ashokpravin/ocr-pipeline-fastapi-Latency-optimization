#!/bin/bash
set -e

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing core dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

echo "Installing paddleocr WITHOUT dependencies..."
pip install --no-cache-dir --no-deps paddleocr==3.3.2

echo "Installing headless OpenCV LAST..."
pip install --no-cache-dir opencv-contrib-python==4.10.0.84

echo "Verifying installations..."
python -c "import paddleocr; import cv2; print('PaddleOCR and OpenCV installed successfully')"

echo "All dependencies installed successfully."
