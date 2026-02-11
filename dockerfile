FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /src

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libstdc++6 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY install_deps.sh .

RUN chmod +x install_deps.sh
RUN bash install_deps.sh

COPY . .

EXPOSE 8050

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8050"]
