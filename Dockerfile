FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PyTorch CUDA 12.4
# ============================================================
RUN pip3 install --no-cache-dir --no-deps \
    "torch==2.5.1+cu124" \
    "torchvision==0.20.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

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
    "nvidia-cufft-cu12==11.2.1.3" \
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
# Diffusers / AnimateDiff
# ============================================================
RUN pip3 install --no-cache-dir \
    runpod \
    "diffusers==0.31.0" \
    "transformers==4.44.2" \
    accelerate \
    safetensors \
    Pillow \
    requests


# ============================================================
# Image / Audio dependencies
# ============================================================
RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    librosa \
    numba \
    scipy


# ============================================================
# Wav2Lip
# ============================================================
RUN git clone https://github.com/Rudrabha/Wav2Lip.git /wav2lip && \
    pip3 install --no-cache-dir batch-face tqdm


# ============================================================
# Hugging Face
# ============================================================
RUN pip3 install --no-cache-dir huggingface_hub


# ============================================================
# EasyNegativeV2 embedding
# ============================================================
RUN mkdir -p /workspace/embeddings && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
src = hf_hub_download(\
    'gsdf/Counterfeit-V3.0', \
    'embedding/EasyNegativeV2.safetensors'\
); \
shutil.copy(src, '/workspace/embeddings/EasyNegativeV2.safetensors')\
"


# ============================================================
# Wav2Lip model
# ============================================================
RUN wget -q -O /workspace/wav2lip.pth \
    "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth"


# ============================================================
# IMPORTANT
#
# 以下の巨大なモデルはDocker imageには含めない。
#
# Network Volumeに置いてあるものを実行時に使用する。
#
# /runpod-volume/models/mm_sd_v15_v2.ckpt
# /runpod-volume/models/Counterfeit-V3.0_fix_fp16.safetensors
#
# ============================================================


# ============================================================
# Handler
# ============================================================
COPY handler.py /handler.py


# ============================================================
# Start RunPod Serverless worker
# ============================================================
CMD ["python3", "-u", "/handler.py"]
