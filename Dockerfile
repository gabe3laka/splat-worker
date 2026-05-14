# splat-worker - RunPod serverless worker for 3D Gaussian Splat training
# Mirrors lingbot-map-worker base (CUDA 12.8 + Python 3.10 + Torch 2.8 cu128)
# but adds COLMAP + ffmpeg + gsplat for actual 3DGS reconstruction.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System deps: Python 3.10, COLMAP, ffmpeg, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python-is-python3 \
    git wget curl ca-certificates \
    ffmpeg colmap \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.8.0 cu128 (matches lingbot-map-worker base)
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0

# Core deps
RUN pip install \
    runpod==1.7.* \
    supabase==2.* \
    numpy==1.26.* \
    pillow==10.* \
    opencv-python-headless==4.10.* \
    plyfile==1.* \
    tqdm \
    requests

# gsplat (CUDA-accelerated 3DGS rasterizer/trainer)
RUN pip install gsplat==1.5.0

# Clone gsplat examples (we vendor a stripped trainer entrypoint via rp_handler)
RUN git clone --depth 1 --branch v1.5.0 https://github.com/nerfstudio-project/gsplat.git /opt/gsplat \
    && pip install -r /opt/gsplat/examples/requirements.txt || true

WORKDIR /app
COPY rp_handler.py /app/rp_handler.py

# RunPod serverless entrypoint
CMD ["python", "-u", "rp_handler.py"]
