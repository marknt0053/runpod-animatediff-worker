import runpod
import base64
import json
import time
import requests
import os

COMFY_URL = "http://127.0.0.1:8188"

def wait_for_comfy(timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{COMFY_URL}/system_stats", timeout=2)
            print("✅ ComfyUI起動確認")
            return True
        except:
            time.sleep(3)
    return False

def upload_video(video_bytes: bytes) -> bool:
    files = {"image": ("input.mp4", video_bytes, "video/mp4")}
    res = requests.post(f"{COMFY_URL}/upload/image", files=files)
    return res.status_code == 200

def run_workflow() -> str:
    with open("/comfyui/workflow.json") as f:
        workflow = json.load(f)
    res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
    data = res.json()
    print("workflow response:", data)
    if "prompt_id" not in data:
        raise Exception(f"ComfyUI error: {data}")
    return data["prompt_id"]

def wait_for_result(prompt_id: str, timeout: int = 300) -> bytes:
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

# ComfyUI起動待ち
print("ComfyUI起動待ち...")
if not wait_for_comfy():
    print("⚠️ ComfyUI起動タイムアウト")

def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video", "")
    video_bytes = base64.b64decode(video_b64)

    # ComfyUI確認
    if not wait_for_comfy(timeout=30):
        return {"error": "ComfyUI未起動"}

    # 動画アップロード
    if not upload_video(video_bytes):
        return {"error": "動画アップロード失敗"}

    # ワークフロー実行
    prompt_id = run_workflow()
    print(f"prompt_id: {prompt_id}")

    # 結果取得
    result = wait_for_result(prompt_id)
    if result is None:
        return {"error": "タイムアウト"}

    return {"video": base64.b64encode(result).decode()}

runpod.serverless.start({"handler": handler})
