FROM runpod/worker-comfyui:5.8.4-base

# 必要なライブラリ
RUN pip install runpod requests

# AnimateDiff Evolvedインストール
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

# VideoHelperSuiteインストール
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt || true

# ghostmixモデルダウンロード
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/digiplay/GhostMix/resolve/main/ghostmix_v20Bakedvae.safetensors' \
    --relative-path models/checkpoints \
    --filename 'ghostmix_v20Bakedvae.safetensors' && break; \
    if [ $i -eq 5 ]; then exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# AnimateDiffモーションモジュールダウンロード
RUN mkdir -p /comfyui/models/animatediff_models && \
    wget -q -O /comfyui/models/animatediff_models/mm_sd_v15_v2.ckpt \
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"

# ワークフローとハンドラーをコピー
COPY workflow.json /workflow.json
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
