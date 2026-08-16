import runpod
import base64
import os
import tempfile
import torch
import subprocess
import numpy as np
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.loaders import FromSingleFileMixin
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
        # まずSD1.5パイプラインとしてsafetensorsを読み込む
        from diffusers import StableDiffusionPipeline
        sd_pipe = StableDiffusionPipeline.from_single_file(
            "/workspace/models/toonyou_jp.safetensors",
            torch_dtype=torch.float16,
        )
        # AnimateDiffVideoToVideoPipelineに変換
        pipe = AnimateDiffVideoToVideoPipeline(
            vae=sd_pipe.vae,
            text_encoder=sd_pipe.text_encoder,
            tokenizer=sd_pipe.tokenizer,
            unet=sd_pipe.unet,
            motion_adapter=adapter,
            scheduler=sd_pipe.scheduler,
        ).to("cuda")
        # DPMSolverMultistepScheduler with karras
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
        print("モデルロード完了")

def apply_wav2lip(anime_video, audio_path, output_path, tmpdir):
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

def process_frames_with_context(frames, pipe, new_h, new_w, context_length=16, context_overlap=4):
    total_frames = len(frames)
    result_frames = [None] * total_frames
    stride = context_length - context_overlap

    prompt = "anime style, studio ghibli, cel shading, vivid colors, masterpiece, high quality, detailed illustration"
    negative_prompt = "worst quality, low quality, blurry, watermark, realistic, photography, noise, grain, flickering, particles, sparkles, floating objects"

    start = 0
    while start < total_frames:
        end = min(start + context_length, total_frames)
        chunk = frames[start:end]
        print(f"処理中: フレーム {start}〜{end-1} ({len(chunk)}フレーム)")

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                video=chunk,
                height=new_h,
                width=new_w,
                strength=0.55,
                num_inference_steps=20,
                guidance_scale=7.0,
                generator=torch.Generator("cuda").manual_seed(42),
            )

        chunk_result = output.frames[0]

        for i, frame in enumerate(chunk_result):
            global_idx = start + i
            if global_idx < total_frames:
                if result_frames[global_idx] is None or i >= context_overlap:
                    result_frames[global_idx] = frame
                else:
                    alpha = i / context_overlap
                    prev = np.array(result_frames[global_idx])
                    curr = np.array(frame)
                    blended = (prev * (1 - alpha) + curr * alpha).astype(np.uint8)
                    result_frames[global_idx] = Image.fromarray(blended)

        if end >= total_frames:
            break
        start += stride

    return [f for f in result_frames if f is not None]

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

        has_audio = subprocess.run([
            "ffmpeg", "-i", input_video, "-vn", "-acodec", "copy",
            audio_path, "-y"
        ], capture_output=True).returncode == 0

        if has_audio:
            subprocess.run([
                "ffmpeg", "-i", input_video, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", audio_wav, "-y"
            ], capture_output=True)

        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_entries", "stream_tags=rotate",
            input_video
        ], capture_output=True, text=True)

        try:
            probe_data = _json.loads(probe.stdout)
        except:
            probe_data = {}

        orig_w, orig_h = 512, 512
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "video" and s.get("width") and s.get("height"):
                orig_w = s["width"]
                orig_h = s["height"]
                rotate = int(s.get("tags", {}).get("rotate", 0))
                if rotate in (90, 270):
                    orig_w, orig_h = orig_h, orig_w
                print(f"ビデオストリーム: {orig_w}x{orig_h} rotate={rotate}")
                break

        if orig_w >= orig_h:
            new_w = 512
            new_h = max(8, int(orig_h * 512 / orig_w / 8) * 8)
        else:
            new_h = 512
            new_w = max(8, int(orig_w * 512 / orig_h / 8) * 8)
        print(f"リサイズ: {orig_w}x{orig_h} → {new_w}x{new_h}")

        vf_filter = f"fps=8,scale={new_w}:{new_h}"
        subprocess.run([
            "ffmpeg", "-i", input_video,
            "-vf", vf_filter,
            os.path.join(frames_dir, "frame_%04d.png"), "-y"
        ], capture_output=True)

        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith(".png")
        ])
        print(f"総フレーム数: {len(frames)}")

        input_frames = [Image.open(f).convert("RGB") for f in frames]

        result_frames = process_frames_with_context(
            input_frames, pipe, new_h, new_w,
            context_length=16,
            context_overlap=4
        )
        print(f"変換完了: {len(result_frames)}フレーム")

        for i, frame in enumerate(result_frames):
            frame.save(os.path.join(tmpdir, f"out_{i:04d}.png"))

        anime_no_audio = os.path.join(tmpdir, "anime_no_audio.mp4")
        subprocess.run([
            "ffmpeg", "-framerate", "8",
            "-i", os.path.join(tmpdir, "out_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            anime_no_audio, "-y"
        ], capture_output=True)

        wav2lip_success = False
        if has_audio and os.path.exists(audio_wav) and os.path.exists("/workspace/wav2lip.pth"):
            wav2lip_success = apply_wav2lip(anime_no_audio, audio_wav, wav2lip_video, tmpdir)

        if wav2lip_success and os.path.exists(wav2lip_video) and os.path.getsize(wav2lip_video) > 0:
            subprocess.run([
                "ffmpeg", "-i", wav2lip_video,
                "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac",
                "-shortest", output_video, "-y"
            ], capture_output=True)
            print("Wav2Lip合成成功")
        else:
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
            print("Wav2Lipスキップ")

        with open(output_video, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

        return {"video": result_b64, "frames": len(result_frames)}

runpod.serverless.start({"handler": handler})
