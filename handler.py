import runpod
import base64
import os
import tempfile
import torch
import subprocess
import numpy as np
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler
from PIL import Image
import json as _json

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

def apply_wav2lip(anime_video, audio_path, output_path, tmpdir):
    """Wav2Lipで口パク合成"""
    try:
        result = subprocess.run([
            "python3", "/wav2lip/inference.py",
            "--checkpoint_path", "/workspace/wav2lip.pth",
            "--face", anime_video,
            "--audio", audio_path,
            "--outfile", output_path,
            "--nosmooth"
        ], capture_output=True, text=True, cwd="/wav2lip")
        if result.returncode == 0:
            print("Wav2Lip処理完了")
            return True
        else:
            print(f"Wav2Lip失敗: {result.stderr[-500:]}")
            return False
    except Exception as e:
        print(f"Wav2Lip例外: {e}")
        return False

def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video", "")
    video_bytes = base64.b64decode(video_b64)

    load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        output_video = os.path.join(tmpdir, "output.mp4")
        wav2lip_video = os.path.join(tmpdir, "wav2lip.mp4")
        audio_path = os.path.join(tmpdir, "audio.aac")
        audio_wav = os.path.join(tmpdir, "audio.wav")
        os.makedirs(frames_dir)

        with open(input_video, "wb") as f:
            f.write(video_bytes)

        # 音声抽出（aac）
        has_audio = subprocess.run([
            "ffmpeg", "-i", input_video, "-vn", "-acodec", "copy",
            audio_path, "-y"
        ], capture_output=True).returncode == 0

        # Wav2Lip用にwav形式でも抽出
        if has_audio:
            subprocess.run([
                "ffmpeg", "-i", input_video, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", audio_wav, "-y"
            ], capture_output=True)

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

        orig_w, orig_h = 512, 512
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "video" and s.get("width") and s.get("height"):
                orig_w = s["width"]
                orig_h = s["height"]
                rotate = int(s.get("tags", {}).get("rotate", 0))
                print(f"ビデオストリーム検出: codec={s.get('codec_name')} {orig_w}x{orig_h} rotate={rotate}")
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

        # 8fpsでフレーム抽出
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

        input_frames = [Image.open(f).convert("RGB") for f in frames]

        # AnimateDiff変換（口の動きは無視してアニメ感最大化）
        output = pipe(
            prompt="anime style, studio ghibli, cel shading, vivid colors, masterpiece, high quality, detailed illustration",
            negative_prompt="worst quality, low quality, blurry, watermark, realistic, photography, 3d render, flickering, flicker, particles, leaves, floating objects, sparkles, effects, overlays",
            video=input_frames,
            height=new_h,
            width=new_w,
            strength=0.66,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42),
        )

        result_frames = output.frames[0]
        result_frames = result_frames[:len(input_frames)]

        # フレームを保存
        for i, frame in enumerate(result_frames):
            frame.save(os.path.join(tmpdir, f"out_{i:04d}.png"))

        # AnimateDiff動画を作成（音声なし）
        anime_no_audio = os.path.join(tmpdir, "anime_no_audio.mp4")
        subprocess.run([
            "ffmpeg", "-framerate", "8",
            "-i", os.path.join(tmpdir, "out_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            anime_no_audio, "-y"
        ], capture_output=True)

        # Wav2Lipで口パク合成
        wav2lip_success = False
        if has_audio and os.path.exists(audio_wav) and os.path.exists("/workspace/wav2lip.pth"):
            wav2lip_success = apply_wav2lip(anime_no_audio, audio_wav, wav2lip_video, tmpdir)

        if wav2lip_success and os.path.exists(wav2lip_video) and os.path.getsize(wav2lip_video) > 0:
            # Wav2Lip成功：音声を付けて最終動画
            subprocess.run([
                "ffmpeg", "-i", wav2lip_video,
                "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac",
                "-shortest", output_video, "-y"
            ], capture_output=True)
            print("Wav2Lip合成成功")
        else:
            # Wav2Lip失敗時はAnimateDiff動画に音声のみ付ける
            if has_audio and os.path.exists(audio_path):
                subprocess.run([
                    "ffmpeg", "-i", anime_no_audio,
                    "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac",
                    "-shortest", output_video, "-y"
                ], capture_output=True)
            else:
                subprocess.run([
                    "ffmpeg", "-i", anime_no_audio,
                    "-c:v", "copy", output_video, "-y"
                ], capture_output=True)
            print("Wav2Lipスキップ（AnimateDiff動画を使用）")

        with open(output_video, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

        return {"video": result_b64, "frames": len(result_frames)}

runpod.serverless.start({"handler": handler})
