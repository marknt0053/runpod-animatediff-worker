FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ============================================================
# System packages
# ============================================================

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PyTorch
# ============================================================

RUN pip3 install --no-cache-dir \
    "torch==2.5.1+cu124" \
    "torchvision==0.20.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124


# ============================================================
# CUDA runtime libraries
# ============================================================

RUN pip3 install --no-cache-dir \
    "nvidia-cudnn-cu12==9.1.1.17" \
    "nvidia-cublas-cu12==12.4.5.8" \
    "nvidia-cusolver-cu12==11.6.1.9" \
    "nvidia-cusparse-cu12==12.3.1.170" \
    "nvidia-curand-cu12==10.3.5.147" \
    "nvidia-cuda-nvrtc-cu12==12.4.127" \
    "nvidia-cuda-runtime-cu12==12.4.127" \
    "nvidia-nvjitlink-cu12==12.4.127" \
    "nvidia-nccl-cu12==2.21.5" \
    "nvidia-cuda-cupti-cu12==12.4.127" \
    "nvidia-nvtx-cu12==12.4.127" \
    "nvidia-cufft-cu12==11.2.1.3"


# ============================================================
# Basic Python dependencies
# ============================================================

RUN pip3 install --no-cache-dir \
    networkx \
    filelock \
    "sympy==1.13.1" \
    fsspec \
    typing-extensions \
    jinja2 \
    MarkupSafe \
    mpmath \
    triton \
    pillow \
    numpy


# ============================================================
# Diffusers / Transformers
#
# AnimateDiff .ckpt support:
# MotionAdapter.from_single_file()
# requires diffusers >= 0.30
# ============================================================

RUN pip3 install --no-cache-dir \
    runpod \
    "diffusers==0.40.0" \
    "transformers==4.57.1" \
    "accelerate==1.14.0" \
    "safetensors>=0.8.0" \
    requests


# ============================================================
# Video / image processing
# ============================================================

RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    librosa \
    numba \
    scipy \
    imageio \
    imageio-ffmpeg


# ============================================================
# Wav2Lip
# ============================================================

RUN git clone --depth 1 \
    https://github.com/Rudrabha/Wav2Lip.git \
    /wav2lip

RUN pip3 install --no-cache-dir \
    batch-face \
    tqdm


# ============================================================
# EasyNegativeV2
#
# This is small, so keeping it in the Docker image is OK.
# ============================================================

RUN mkdir -p /workspace/embeddings

RUN pip3 install --no-cache-dir huggingface_hub

RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
p = hf_hub_download( \
    'gsdf/Counterfeit-V3.0', \
    'embedding/EasyNegativeV2.safetensors' \
); \
shutil.copy( \
    p, \
    '/workspace/embeddings/EasyNegativeV2.safetensors' \
)"


# ============================================================
# Wav2Lip model
# ============================================================

RUN mkdir -p /workspace

RUN wget -q \
    -O /workspace/wav2lip.pth \
    "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth"


# ============================================================
# IMPORTANT
#
# DO NOT download Counterfeit or AnimateDiff here.
#
# They are stored on RunPod Network Volume:
#
# /runpod-volume/models/
#
# Counterfeit-V3.0_fix_fp16.safetensors
# mm_sd_v15_v2.ckpt
#
# ============================================================


# ============================================================
# Serverless handler
# ============================================================

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
