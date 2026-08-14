FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg wget git && rm -rf /var/lib/apt/lists/*
RUN pip3 install "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install runpod "diffusers==0.31.0" "transformers==4.44.2" accelerate safetensors Pillow requests
RUN mkdir -p /workspace
COPY handler.py /handler.py
CMD ["python3", "-u", "/handler.py"]
