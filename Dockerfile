FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg wget git libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-deps "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install "nvidia-cudnn-cu12==9.1.1.17" "nvidia-cublas-cu12==12.4.5.8" "nvidia-cusolver-cu12==11.6.1.9" "nvidia-cusparse-cu12==12.3.1.170" "nvidia-curand-cu12==10.3.5.147" "nvidia-cuda-nvrtc-cu12==12.4.127" "nvidia-cuda-runtime-cu12==12.4.127" "nvidia-nvjitlink-cu12==12.4.127" "nvidia-nccl-cu12==2.21.5" "nvidia-cuda-cupti-cu12==12.4.127" "nvidia-nvtx-cu12==12.4.127" "nvidia-cufft-cu12==11.2.1.3" && \
    pip3 install networkx filelock sympy==1.13.1 fsspec typing-extensions jinja2 MarkupSafe mpmath triton pillow numpy
RUN pip3 install runpod "diffusers==0.31.0" "transformers==4.44.2" accelerate safetensors Pillow requests
RUN pip3 install opencv-python-headless librosa numba scipy
# Wav2Lipセットアップ
RUN git clone https://github.com/Rudrabha/Wav2Lip.git /wav2lip && \
    pip3 install -r /wav2lip/requirements.txt
# Wav2Lipモデルダウンロード
RUN mkdir -p /workspace && \
    wget -q -O /workspace/wav2lip.pth "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z65UepRScVAndSnw" || \
    wget -q -O /workspace/wav2lip.pth "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth"
# AnimateDiffモーションモジュール
RUN wget -q -O /workspace/mm_sd_v15_v2.ckpt "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"
COPY handler.py /handler.py
CMD ["python3", "-u", "/handler.py"]
