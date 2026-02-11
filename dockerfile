# =============================================================================
# CustomOCR Pipeline API - Dockerfile (Katonic Docker Mode)
# =============================================================================
# Use this ONLY if Katonic supports Dockerfile-based deployment.
# If you must use the managed FastAPI image, use startup.sh instead.
# =============================================================================

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /src

# System dependencies (this is why Docker mode is preferred)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libstdc++6 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps paddleocr==3.3.2 && \
    pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null; true && \
    pip install --no-cache-dir --no-deps \
        opencv-python-headless==4.10.0.84 \
        opencv-contrib-python-headless==4.10.0.84

# Copy application
COPY . .

# Create output directory
RUN mkdir -p /src/output/completed

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8050", "--timeout-keep-alive", "120"]