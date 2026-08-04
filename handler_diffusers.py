import runpod
import base64
import os
import tempfile
import torch
import subprocess
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler
from PIL import Image

MODEL_PATH = "/workspace/ghostmix_v20Bakedvae.safetensors"
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("モデルロード中...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=torch.float16
        )
        pipe = AnimateDiffVideoToVideoPipeline.from_single_file(
            MODEL_PATH,
            motion_adapter=adapter,
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, beta_schedule="linear"
        )
        print("モデルロード完了")

def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video", "")
    video_bytes = base64.b64decode(video_b64)

    load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        output_video = os.path.join(tmpdir, "output.mp4")
        audio_path = os.path.join(tmpdir, "audio.aac")
        os.makedirs(frames_dir)

        with open(input_video, "wb") as f:
            f.write(video_bytes)

        # 音声抽出
        has_audio = subprocess.run([
            "ffmpeg", "-i", input_video, "-vn", "-acodec", "copy",
            audio_path, "-y"
        ], capture_output=True).returncode == 0

        # 元動画の解像度取得
        import json as _json
        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", input_video
        ], capture_output=True, text=True)
        probe_data = _json.loads(probe.stdout)
        orig_w, orig_h = 512, 512
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "video":
                orig_w = s["width"]
                orig_h = s["height"]
                break

        # アスペクト比を保ちながら長辺512に収める（8の倍数に丸める）
        if orig_w >= orig_h:
            new_w = 512
            new_h = max(8, int(orig_h * 512 / orig_w / 8) * 8)
        else:
            new_h = 512
            new_w = max(8, int(orig_w * 512 / orig_h / 8) * 8)
        print(f"リサイズ: {orig_w}x{orig_h} → {new_w}x{new_h}")

        # フレーム抽出（8fps、アスペクト比保持）
        subprocess.run([
            "ffmpeg", "-i", input_video,
            "-vf", f"fps=8,scale={new_w}:{new_h}",
            os.path.join(frames_dir, "frame_%04d.png"), "-y"
        ], capture_output=True)

        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith(".png")
        ])[:24]
        print(f"フレーム数: {len(frames)}")

        # 入力フレームを読み込み
        input_frames = [Image.open(f).convert("RGB") for f in frames]

        # AnimateDiff video-to-video処理
        output = pipe(
            prompt="anime style, masterpiece, high quality, detailed",
            negative_prompt="worst quality, low quality, blurry, watermark",
            video=input_frames,
            height=new_h,
            width=new_w,
            strength=0.6,
            num_inference_steps=10,
            guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(42),
        )

        result_frames = output.frames[0]

        # フレームを保存
        for i, frame in enumerate(result_frames):
            frame.save(os.path.join(tmpdir, f"out_{i:04d}.png"))

        # 動画結合
        if has_audio and os.path.exists(audio_path):
            subprocess.run([
                "ffmpeg", "-framerate", "8",
                "-i", os.path.join(tmpdir, "out_%04d.png"),
                "-i", audio_path,
                "-c:v", "libx264", "-c:a", "aac",
                "-shortest", output_video, "-y"
            ], capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-framerate", "8",
                "-i", os.path.join(tmpdir, "out_%04d.png"),
                "-c:v", "libx264", output_video, "-y"
            ], capture_output=True)

        with open(output_video, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

        return {"video": result_b64, "frames": len(result_frames)}

runpod.serverless.start({"handler": handler})
