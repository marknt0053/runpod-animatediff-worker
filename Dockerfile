FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg wget git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install runpod requests

# ComfyUIインストール
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
RUN pip3 install -r /comfyui/requirements.txt

# AnimateDiff Evolvedインストール
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

# VideoHelperSuiteインストール
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip3 install -r requirements.txt || true

# ghostmixモデルダウンロード
RUN mkdir -p /comfyui/models/checkpoints && \
    wget -q -O /comfyui/models/checkpoints/ghostmix_v20Bakedvae.safetensors \
    "https://huggingface.co/digiplay/GhostMix/resolve/main/ghostmix_v20Bakedvae.safetensors"

# AnimateDiffモーションモジュールダウンロード
RUN mkdir -p /comfyui/models/animatediff_models && \
    wget -q -O /comfyui/models/animatediff_models/mm_sd_v15_v2.ckpt \
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"

COPY workflow.json /workflow.json
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
