# ────────────────────────────────────────────────────────────────────
# PTZ Camera Tracking — CUDA-enabled image
#   Base : NVIDIA CUDA 12.1 + cuDNN 8 (Ubuntu 22.04)
#   Run  : docker run --gpus all --rm -it \
#            --env-file .env \
#            -v $(pwd)/yolov8n.pt:/app/yolov8n.pt \
#            -e DISPLAY=$DISPLAY \
#            -v /tmp/.X11-unix:/tmp/.X11-unix \
#            ptz-tracker python 04_updated_track.py
# ────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System deps (OpenCV + GUI + ffmpeg for RTSP) ────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg \
        libgl1 libglib2.0-0 \
        libsm6 libxext6 libxrender1 \
        libgtk-3-0 \
        libavcodec-dev libavformat-dev libswscale-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ── Python deps (torch with CUDA wheels first) ──────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# ── App code ────────────────────────────────────────────────────────
COPY . .

# Default: run the full adaptive tracker
CMD ["python", "04_updated_track.py"]
