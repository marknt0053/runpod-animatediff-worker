import runpod
import base64
import os
import tempfile
import torch
import subprocess
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import json as _json

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("モデルロード中...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        print("モデルロード完了")

def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video", "")
    video_bytes = base64.b64decode(video_b64)

    load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out_frames")
        output_video = os.path.join(tmpdir, "output.mp4")
        audio_path = os.path.join(tmpdir, "audio.aac")
        os.makedirs(frames_dir)
        os.makedirs(out_dir)

        with open(input_video, "wb") as f:
            f.write(video_bytes)

        # 音声抽出
        has_audio = subprocess.run([
            "ffmpeg", "-i", input_video, "-vn", "-acodec", "copy",
            audio_path, "-y"
        ], capture_output=True).returncode == 0

        # 元動画の解像度・FPS取得
        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_entries", "stream_tags=rotate",
            input_video
        ], capture_output=True, text=True)
        print(f"ffprobe stdout: {probe.stdout[:200]}")
        print(f"ffprobe stderr: {probe.stderr[:200]}")

        try:
            probe_data = _json.loads(probe.stdout)
        except Exception as e:
            print(f"ffprobe JSON parse error: {e}")
            probe_data = {}

        orig_w, orig_h, orig_fps = 512, 512, 24
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "video" and s.get("width") and s.get("height"):
                orig_w = s["width"]
                orig_h = s["height"]
                rotate = int(s.get("tags", {}).get("rotate", 0))
                r_frame_rate = s.get("r_frame_rate", "24/1")
                try:
                    num, den = r_frame_rate.split("/")
                    orig_fps = int(int(num) / int(den))
                except:
                    orig_fps = 24
                print(f"ビデオストリーム検出: codec={s.get('codec_name')} {orig_w}x{orig_h} rotate={rotate} fps={orig_fps}")
                if rotate in (90, 270):
                    orig_w, orig_h = orig_h, orig_w
                    print(f"回転補正後: {orig_w}x{orig_h}")
                break

        # アスペクト比を保ちながら長辺512に収める（8の倍数）
        if orig_w >= orig_h:
            new_w = 512
            new_h = max(8, int(orig_h * 512 / orig_w / 8) * 8)
        else:
            new_h = 512
            new_w = max(8, int(orig_w * 512 / orig_h / 8) * 8)
        print(f"リサイズ: {orig_w}x{orig_h} → {new_w}x{new_h}")

        # フレーム抽出（元動画のfpsのまま全フレーム）
        vf_filter = f"scale={new_w}:{new_h}"
        subprocess.run([
            "ffmpeg", "-i", input_video,
            "-vf", vf_filter,
            os.path.join(frames_dir, "frame_%04d.png"), "-y"
        ], capture_output=True)

        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith(".png")
        ])
        print(f"フレーム数: {len(frames)}")

        # 各フレームをSD img2imgで個別変換
        prompt = "anime style, studio ghibli, cel shading, vivid colors, masterpiece, high quality, detailed illustration, smooth animation"
        negative_prompt = "worst quality, low quality, blurry, watermark, realistic, photography, 3d render, noise, grain"

        with torch.no_grad():
            for i, frame_path in enumerate(frames):
                img = Image.open(frame_path).convert("RGB")
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=img,
                    strength=0.55,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=torch.Generator("cuda").manual_seed(42),
                ).images[0]
                result.save(os.path.join(out_dir, f"out_{i:04d}.png"))
                if (i + 1) % 5 == 0:
                    print(f"処理済み: {i+1}/{len(frames)} フレーム")

        print("全フレーム変換完了")

        # 動画結合（元動画と同じfps）
        if has_audio and os.path.exists(audio_path):
            subprocess.run([
                "ffmpeg", "-framerate", str(orig_fps),
                "-i", os.path.join(out_dir, "out_%04d.png"),
                "-i", audio_path,
                "-c:v", "libx264", "-c:a", "aac",
                "-shortest", output_video, "-y"
            ], capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-framerate", str(orig_fps),
                "-i", os.path.join(out_dir, "out_%04d.png"),
                "-c:v", "libx264", output_video, "-y"
            ], capture_output=True)

        with open(output_video, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

        return {"video": result_b64, "frames": len(frames)}

runpod.serverless.start({"handler": handler})
