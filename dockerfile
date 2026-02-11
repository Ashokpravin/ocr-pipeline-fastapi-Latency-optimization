FROM python:3.11-slim



# Avoid interactive prompts

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1



WORKDIR /app



# Install system dependencies required for PDF + OpenCV

RUN apt-get update && apt-get install -y \

    build-essential \

    libgl1 \

    libglib2.0-0 \

    poppler-utils \

    && rm -rf /var/lib/apt/lists/*



# Copy dependency files first (for Docker layer caching)

COPY requirements.txt .

COPY install_deps.sh .



# Give permission to script

RUN chmod +x install_deps.sh



# Install Python dependencies

RUN bash install_deps.sh



# Copy full application

COPY . .



# Expose FastAPI port

EXPOSE 8000



# Start FastAPI

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

