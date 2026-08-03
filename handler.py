import runpod
import base64
import os
import json
import time
import requests
import tempfile

COMFY_URL = "http://127.0.0.1:8188"

def wait_for_comfy():
    for _ in range(60):
        try:
            requests.get(f"{COMFY_URL}/system_stats", timeout=2)
            return True
        except:
            time.sleep(2)
    return False

def upload_video(video_bytes: bytes) -> str:
    """動画をComfyUIにアップロード"""
    files = {"image": ("input.mp4", video_bytes, "video/mp4")}
    res = requests.post(f"{COMFY_URL}/upload/image", files=files)
    return res.json().get("name", "input.mp4")

def run_workflow(workflow: dict) -> str:
    """ワークフローを実行してprompt_idを返す"""
    res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
    return res.json()["prompt_id"]

def wait_for_result(prompt_id: str, timeout: int = 300) -> bytes:
    """結果の動画ファイルを取得"""
    start = time.time()
    while time.time() - start < timeout:
        hist = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            for node_output in outputs.values():
                for key in ["gifs", "videos"]:
                    files = node_output.get(key, [])
                    if files:
                        filename = files[0]["filename"]
                        subfolder = files[0].get("subfolder", "")
                        res = requests.get(
                            f"{COMFY_URL}/view",
                            params={"filename": filename, "subfolder": subfolder, "type": "output"}
                        )
                        return res.content
        time.sleep(3)
    return None

def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video", "")
    video_bytes = base64.b64decode(video_b64)

    # ComfyUI起動待ち
    if not wait_for_comfy():
        return {"error": "ComfyUI起動タイムアウト"}

    # 動画アップロード
    upload_video(video_bytes)

    # ワークフロー読み込み
    with open("/workflow.json") as f:
        workflow = json.load(f)

    # 実行
    prompt_id = run_workflow(workflow)
    print(f"prompt_id: {prompt_id}")

    # 結果取得
    result = wait_for_result(prompt_id)
    if result is None:
        return {"error": "タイムアウト"}

    return {"video": base64.b64encode(result).decode()}

runpod.serverless.start({"handler": handler})
