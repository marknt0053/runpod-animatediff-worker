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
        pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
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
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_entries", "stream_tags=rotate", input_video
        ], capture_output=True, text=True)
        print(f"ffprobe stdout: {probe.stdout[:200]}")
        print(f"ffprobe stderr: {probe.stderr[:200]}")
        try:
            probe_data = _json.loads(probe.stdout)
        except Exception as e:
            print(f"ffprobe JSON parse error: {e}")
            probe_data = {}
        orig_w, orig_h = 512, 512
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "video" and s.get("width") and s.get("height"):
                orig_w = s["width"]
                orig_h = s["height"]
                # 回転メタデータを確認
                rotate = int(s.get("tags", {}).get("rotate", 0))
                print(f"ビデオストリーム検出: codec={s.get('codec_name')} {orig_w}x{orig_h} rotate={rotate}")
                # 90度または270度回転の場合はwidthとheightを入れ替え
                if rotate in (90, 270):
                    orig_w, orig_h = orig_h, orig_w
                    print(f"回転補正後: {orig_w}x{orig_h}")
                break

        # アスペクト比を保ちながら長辺512に収める（8の倍数に丸める）
        if orig_w >= orig_h:
            new_w = 512
            new_h = max(8, int(orig_h * 512 / orig_w / 8) * 8)
        else:
            new_h = 512
            new_w = max(8, int(orig_w * 512 / orig_h / 8) * 8)
        print(f"アスペクト比確認: orig={orig_w}x{orig_h} → new={new_w}x{new_h}")
        print(f"リサイズ: {orig_w}x{orig_h} → {new_w}x{new_h}")

        # フレーム抽出（8fps、アスペクト比保持、回転補正）
        vf_filter = f"fps=8,scale={new_w}:{new_h}"
        subprocess.run([
            "ffmpeg", "-i", input_video,
            "-vf", vf_filter,
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
            prompt="anime style, studio ghibli, cel shading, vivid colors, masterpiece, high quality, detailed illustration",
            negative_prompt="worst quality, low quality, blurry, watermark, realistic, photography, 3d render, flickering, flicker, particles, leaves, floating objects, sparkles, effects, overlays",
            video=input_frames,
            height=new_h,
            width=new_w,
            strength=0.66,
            num_inference_steps=40,
            guidance_scale=7.6,
            generator=torch.Generator("cuda").manual_seed(123),
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

        # チラつき除去フィルタ
        smoothed_video = os.path.join(tmpdir, "smoothed.mp4")
        subprocess.run([
            "ffmpeg", "-i", output_video,
            "-vf", "tblend=all_mode=average",
            "-c:v", "libx264", smoothed_video, "-y"
        ], capture_output=True)
        if os.path.exists(smoothed_video) and os.path.getsize(smoothed_video) > 0:
            output_video = smoothed_video

        with open(output_video, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

        return {"video": result_b64, "frames": len(result_frames)}

runpod.serverless.start({"handler": handler})
