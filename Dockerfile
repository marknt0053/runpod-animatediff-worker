FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg wget git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-deps "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install "nvidia-cudnn-cu12==9.1.1.17" "nvidia-cublas-cu12==12.4.5.8" "nvidia-cusolver-cu12==11.6.1.9" "nvidia-cusparse-cu12==12.3.1.170" "nvidia-curand-cu12==10.3.5.147" "nvidia-cuda-nvrtc-cu12==12.4.127" "nvidia-cuda-runtime-cu12==12.4.127" "nvidia-nvjitlink-cu12==12.4.127" "nvidia-nccl-cu12==2.21.5" "nvidia-cuda-cupti-cu12==12.4.127" "nvidia-nvtx-cu12==12.4.127" && \
    pip3 install networkx filelock sympy==1.13.1 fsspec typing-extensions jinja2 MarkupSafe mpmath triton pillow
RUN pip3 install runpod "diffusers==0.30.3" "transformers==4.44.2" accelerate safetensors Pillow requests

# ghostmixモデルダウンロード
RUN mkdir -p /workspace && \
    pip3 install huggingface_hub && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stable-diffusion-v1-5/stable-diffusion-v1-5', cache_dir='/workspace/hub')"

# AnimateDiffモーションモジュールダウンロード
RUN mkdir -p /workspace && \
    wget -q -O /workspace/mm_sd_v15_v2.ckpt \
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
