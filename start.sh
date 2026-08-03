#!/bin/bash
# ComfyUIを先に起動
python3 /comfyui/main.py --listen 0.0.0.0 --port 8188 &
# handler.pyを起動
python3 -u /handler.py
