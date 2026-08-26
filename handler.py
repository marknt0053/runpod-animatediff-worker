import os
import gc
import base64
import tempfile
import subprocess
from pathlib import Path

import runpod
import torch
import numpy as np

from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    AnimateDiffVideoToVideoPipeline,
    MotionAdapter,
    DPMSolverMultistepScheduler,
)


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = "/runpod-volume/models"

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "Counterfeit-V3.0_fix_fp16.safetensors",
)

MOTION_MODULE_PATH = "/runpod-volume/mm_sd_v15_v2.ckpt"

EMBEDDING_PATH = "/runpod-volume/embeddings/EasyNegativeV2.safetensors"


# ============================================================
# Device
# ============================================================

DEVICE = "cuda"
DTYPE = torch.float16


# ============================================================
# Video settings
# ============================================================

OUTPUT_WIDTH = 320
OUTPUT_HEIGHT = 568

FPS = 8

STEPS = 20

CFG = 7.0

DENOISE = 0.55

SEED = 41868074274227


# ============================================================
# Tail extension
# ============================================================
#
# AnimateDiffの最後の数フレームで横転する問題を抑えるため、
# 元動画の最後のフレームを複製して処理用のフレームを追加する。
#
# 重要:
#   動画そのものをFFmpegで延長しない。
#   Python上のフレーム配列だけを延長する。
#
# これにより、
#
#   元67フレーム + 16フレーム = 83フレーム
#
# のように、フレーム数を完全に制御できる。
#
# 最終出力では、この16フレームを必ず削除する。
#

EXTEND_FRAMES = 16


# ============================================================
# AnimateDiff context settings
# ============================================================

CONTEXT_LENGTH = 16
CONTEXT_OVERLAP = 8
CONTEXT_STRIDE = 1


# ============================================================
# Prompt
# ============================================================

PROMPT = (
    "anime style, "
    "studio ghibli, "
    "cel shading, "
    "vivid colors, "
    "masterpiece, "
    "high quality, "
    "detailed illustration"
)


NEGATIVE_PROMPT = (
    "<EasyNegativeV2>, "
    "worst quality, "
    "low quality, "
    "blurry, "
    "watermark, "
    "realistic, "
    "photography, "
    "noise, "
    "grain, "
    "flickering, "
    "particles, "
    "sparkles, "
    "floating objects"
)


# ============================================================
# Global pipeline
# ============================================================

pipe = None


# ============================================================
# Logging
# ============================================================

def log(message):

    print(
        f"[AnimeDiff-Worker] {message}",
        flush=True,
    )


# ============================================================
# GPU check
# ============================================================

def check_gpu():

    log("=" * 70)
    log("GPU CHECK")
    log("=" * 70)

    log(
        f"PyTorch: {torch.__version__}"
    )

    log(
        f"CUDA compiled version: "
        f"{torch.version.cuda}"
    )

    cuda_available = torch.cuda.is_available()

    log(
        f"CUDA available: "
        f"{cuda_available}"
    )

    if not cuda_available:

        log(
            "ERROR: CUDA is not available."
        )

        raise RuntimeError(
            "CUDA is not available. "
            "RunPod Serverless Worker does not "
            "have access to a GPU."
        )

    gpu_count = torch.cuda.device_count()

    log(
        f"GPU count: {gpu_count}"
    )

    for index in range(gpu_count):

        name = torch.cuda.get_device_name(index)

        properties = (
            torch.cuda.get_device_properties(index)
        )

        memory_gb = (
            properties.total_memory
            / 1024**3
        )

        log(
            f"GPU {index}: "
            f"{name} "
            f"({memory_gb:.2f} GB)"
        )

    log(
        f"Selected device: "
        f"{torch.cuda.get_device_name(0)}"
    )

    log("=" * 70)


# ============================================================
# Model check
# ============================================================

def check_models():

    log("=" * 70)
    log("MODEL CHECK")
    log("=" * 70)

    log(
        f"Model directory: "
        f"{MODEL_DIR}"
    )

    if not os.path.isdir(MODEL_DIR):

        raise FileNotFoundError(
            f"Model directory does not exist: "
            f"{MODEL_DIR}"
        )

    # --------------------------------------------------------
    # Counterfeit
    # --------------------------------------------------------

    if not os.path.isfile(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            "Counterfeit model not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint_size = (
        os.path.getsize(
            CHECKPOINT_PATH
        )
        / 1024**3
    )

    log(
        f"Counterfeit model: "
        f"{checkpoint_size:.2f} GB"
    )

    # --------------------------------------------------------
    # Motion module
    # --------------------------------------------------------

    if not os.path.isfile(
        MOTION_MODULE_PATH
    ):

        raise FileNotFoundError(
            "AnimateDiff motion module not found:\n"
            f"{MOTION_MODULE_PATH}"
        )

    motion_size = (
        os.path.getsize(
            MOTION_MODULE_PATH
        )
        / 1024**3
    )

    log(
        f"Motion module: "
        f"{motion_size:.2f} GB"
    )

    # --------------------------------------------------------
    # EasyNegative
    # --------------------------------------------------------

    if os.path.isfile(
        EMBEDDING_PATH
    ):

        embedding_size = (
            os.path.getsize(
                EMBEDDING_PATH
            )
            / 1024**2
        )

        log(
            f"EasyNegativeV2: "
            f"{embedding_size:.2f} MB"
        )

    else:

        log(
            "EasyNegativeV2 not found."
        )

        log(
            "Textual inversion will be skipped."
        )

    log("=" * 70)


# ============================================================
# Model loading
# ============================================================

def load_model():

    global pipe

    if pipe is not None:

        log(
            "Pipeline already loaded."
        )

        return

    log("=" * 70)
    log("MODEL INITIALIZATION")
    log("=" * 70)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    check_gpu()

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    check_models()

    # --------------------------------------------------------
    # CUDA device
    # --------------------------------------------------------

    torch.cuda.set_device(0)

    log(
        f"Using GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # MotionAdapter
    # --------------------------------------------------------

    log(
        "Loading AnimateDiff MotionAdapter..."
    )

    log(
        f"Source: "
        f"{MOTION_MODULE_PATH}"
    )

    motion_adapter = (
        MotionAdapter.from_single_file(
            MOTION_MODULE_PATH,
            torch_dtype=DTYPE,
        )
    )

    log(
        "MotionAdapter loaded."
    )

    # --------------------------------------------------------
    # Stable Diffusion checkpoint
    # --------------------------------------------------------

    log(
        "Loading Counterfeit-V3.0..."
    )

    log(
        f"Source: "
        f"{CHECKPOINT_PATH}"
    )

    sd_pipe = (
        StableDiffusionPipeline.from_single_file(
            CHECKPOINT_PATH,

            torch_dtype=DTYPE,

            safety_checker=None,

            requires_safety_checker=False,
        )
    )

    log(
        "Counterfeit-V3.0 loaded."
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    log(
        "Configuring scheduler..."
    )

    scheduler = (
        DPMSolverMultistepScheduler.from_config(
            sd_pipe.scheduler.config,

            algorithm_type="dpmsolver++",

            solver_order=2,

            use_karras_sigmas=True,
        )
    )

    # --------------------------------------------------------
    # AnimateDiff Video-to-Video Pipeline
    # --------------------------------------------------------

    log(
        "Creating AnimateDiffVideoToVideoPipeline..."
    )

    pipe = (
        AnimateDiffVideoToVideoPipeline(
            vae=sd_pipe.vae,

            text_encoder=sd_pipe.text_encoder,

            tokenizer=sd_pipe.tokenizer,

            unet=sd_pipe.unet,

            motion_adapter=motion_adapter,

            scheduler=scheduler,

            feature_extractor=None,

            image_encoder=None,
        )
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    log(
        "Moving pipeline to GPU..."
    )

    pipe = pipe.to(
        DEVICE,
        dtype=DTYPE,
    )

    # --------------------------------------------------------
    # VAE optimization
    # --------------------------------------------------------

    log(
        "Enabling VAE slicing..."
    )

    pipe.enable_vae_slicing()

    # --------------------------------------------------------
    # EasyNegative
    # --------------------------------------------------------

    if os.path.isfile(
        EMBEDDING_PATH
    ):

        log(
            "Loading EasyNegativeV2..."
        )

        try:

            pipe.load_textual_inversion(
                EMBEDDING_PATH,
                token="EasyNegativeV2",
            )

            log(
                "EasyNegativeV2 loaded."
            )

        except Exception as error:

            log(
                "WARNING: "
                "EasyNegativeV2 failed to load."
            )

            log(
                str(error)
            )

    # --------------------------------------------------------
    # Cleanup SD pipeline
    # --------------------------------------------------------

    del sd_pipe

    gc.collect()

    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Memory information
    # --------------------------------------------------------

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    log(
        f"GPU memory allocated: "
        f"{allocated:.2f} GB"
    )

    log(
        f"GPU memory reserved: "
        f"{reserved:.2f} GB"
    )

    log("=" * 70)
    log("MODEL INITIALIZATION COMPLETE")
    log("=" * 70)


# ============================================================
# Extract video frames
# ============================================================

def extract_frames(
    input_video,
    frames_dir,
):

    os.makedirs(
        frames_dir,
        exist_ok=True,
    )

    output_pattern = os.path.join(
        frames_dir,
        "frame_%06d.png",
    )

    vf = (
        f"fps={FPS},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
    )

    command = [
        "ffmpeg",

        "-hide_banner",
        "-loglevel",
        "error",

        "-noautorotate",

        "-i",
        input_video,

        "-vf",
        vf,

        "-vsync",
        "0",

        output_pattern,

        "-y",
    ]

    log(
        "Extracting frames..."
    )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg frame extraction failed:\n"
            + result.stderr
        )

    frames = sorted(
        [
            os.path.join(
                frames_dir,
                name,
            )

            for name in os.listdir(
                frames_dir
            )

            if name.lower().endswith(
                ".png"
            )
        ]
    )

    if not frames:

        raise RuntimeError(
            "No video frames were extracted."
        )

    log(
        f"Extracted {len(frames)} frames."
    )

    return frames


# ============================================================
# Extend frames
# ============================================================

def extend_frames_for_tail(
    frames,
    extend_count,
):

    if not frames:

        raise ValueError(
            "Cannot extend an empty frame list."
        )

    if extend_count <= 0:

        return list(frames)

    log("=" * 70)
    log("EXTENDING FRAME SEQUENCE FOR TAIL STABILIZATION")
    log("=" * 70)

    original_count = len(frames)

    log(
        f"Original frame count: "
        f"{original_count}"
    )

    log(
        f"Extension frames: "
        f"{extend_count}"
    )

    # --------------------------------------------------------
    # 最後のフレームを複製
    # --------------------------------------------------------

    last_frame = frames[-1]

    extended_frames = (
        list(frames)
        +
        [
            last_frame
            for _ in range(
                extend_count
            )
        ]
    )

    expected_count = (
        original_count
        + extend_count
    )

    actual_count = len(
        extended_frames
    )

    if actual_count != expected_count:

        raise RuntimeError(
            "Frame extension failed: "
            f"got {actual_count}, "
            f"expected {expected_count}"
        )

    log(
        f"Extended frame count: "
        f"{actual_count}"
    )

    log(
        f"Expected frame count: "
        f"{expected_count}"
    )

    log(
        "The final source frame was duplicated "
        f"{extend_count} times."
    )

    return extended_frames


# ============================================================
# Context windows
# ============================================================

def create_context_windows(
    num_frames,
):

    if num_frames <= 0:

        return []

    if num_frames <= CONTEXT_LENGTH:

        return [
            list(
                range(
                    num_frames
                )
            )
        ]

    windows = []

    step = (
        CONTEXT_LENGTH
        - CONTEXT_OVERLAP
    )

    start = 0

    while True:

        end = min(
            start + CONTEXT_LENGTH,
            num_frames,
        )

        # ----------------------------------------------------
        # 通常window
        # ----------------------------------------------------

        indices = list(
            range(
                start,
                end,
            )
        )

        # ----------------------------------------------------
        # 最後のwindow
        # ----------------------------------------------------

        if end >= num_frames:

            # 最後が16フレーム未満になる場合、
            # 最後の16フレームを明示的に使用する。
            if len(indices) < CONTEXT_LENGTH:

                start = max(
                    0,
                    num_frames - CONTEXT_LENGTH,
                )

                end = num_frames

                indices = list(
                    range(
                        start,
                        end,
                    )
                )

            # 重複windowを追加しない
            if (
                not windows
                or windows[-1] != indices
            ):

                windows.append(
                    indices
                )

            break

        windows.append(
            indices
        )

        start += step

    return windows


# ============================================================
# Pyramid weights
# ============================================================

def create_pyramid_weights(
    length,
    is_last_window=False,
):

    if length <= 0:

        return np.array(
            [],
            dtype=np.float32,
        )

    center = (
        length + 1
    ) // 2

    weights = []

    for i in range(length):

        distance = abs(
            i - (length - 1) / 2
        )

        weight = (
            center - distance
        )

        weights.append(
            max(
                weight,
                1.0,
            )
        )

    weights_np = np.asarray(
        weights,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 最後のwindow
    # --------------------------------------------------------
    #
    # 最後のwindowでは後半側を均一に高くする。
    #
    # これにより、動画末尾に向かうフレームが
    # 前のwindowの影響だけで崩れることを防ぐ。
    #

    if is_last_window:

        half = length // 2

        max_weight = float(
            weights_np.max()
        )

        for i in range(
            half,
            length,
        ):

            weights_np[i] = (
                max_weight
            )

    return weights_np


# ============================================================
# AnimateDiff
# ============================================================

def process_video_frames(
    frames,
):

    total_frames = len(frames)

    log("=" * 70)
    log("ANIMATEDIFF PROCESSING")
    log("=" * 70)

    log(
        f"Frames: "
        f"{total_frames}"
    )

    log(
        f"Resolution: "
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
    )

    log(
        f"FPS: "
        f"{FPS}"
    )

    log(
        f"Steps: "
        f"{STEPS}"
    )

    log(
        f"CFG: "
        f"{CFG}"
    )

    log(
        f"Denoising strength: "
        f"{DENOISE}"
    )

    log(
        f"Seed: "
        f"{SEED}"
    )

    log(
        f"Context length: "
        f"{CONTEXT_LENGTH}"
    )

    log(
        f"Context overlap: "
        f"{CONTEXT_OVERLAP}"
    )

    windows = create_context_windows(
        total_frames
    )

    log(
        f"Context windows: "
        f"{len(windows)}"
    )

    # --------------------------------------------------------
    # Window一覧
    # --------------------------------------------------------

    for index, window in enumerate(
        windows,
        start=1,
    ):

        log(
            f"Window {index}: "
            f"{window[0]} - {window[-1]} "
            f"({len(window)} frames)"
        )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    result_accum = np.zeros(
        (
            total_frames,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            3,
        ),
        dtype=np.float32,
    )

    weight_accum = np.zeros(
        total_frames,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Process windows
    # --------------------------------------------------------

    for window_number, indices in enumerate(
        windows,
        start=1,
    ):

        log("=" * 70)

        log(
            f"Context "
            f"{window_number}/{len(windows)}"
        )

        log(
            f"Frames "
            f"{indices[0]} - "
            f"{indices[-1]}"
        )

        # ----------------------------------------------------
        # Last window
        # ----------------------------------------------------

        is_last = (
            window_number
            ==
            len(windows)
        )

        if is_last:

            log(
                "This is the FINAL context window."
            )

        # ----------------------------------------------------
        # Same seed for each context
        # ----------------------------------------------------

        generator = (
            torch.Generator(
                device=DEVICE
            ).manual_seed(
                SEED
            )
        )

        # ----------------------------------------------------
        # Input chunk
        # ----------------------------------------------------

        chunk = [
            frames[index]
            for index in indices
        ]

        log(
            f"Input chunk frames: "
            f"{len(chunk)}"
        )

        try:

            with torch.inference_mode():

                output = pipe(
                    prompt=PROMPT,

                    negative_prompt=NEGATIVE_PROMPT,

                    video=chunk,

                    height=OUTPUT_HEIGHT,

                    width=OUTPUT_WIDTH,

                    strength=DENOISE,

                    num_inference_steps=STEPS,

                    guidance_scale=CFG,

                    generator=generator,

                    output_type="pil",
                )

            output_frames = (
                output.frames[0]
            )

        except torch.cuda.OutOfMemoryError:

            log(
                "CUDA OUT OF MEMORY."
            )

            torch.cuda.empty_cache()

            gc.collect()

            raise RuntimeError(
                "GPU out of memory while "
                "processing an AnimateDiff "
                "context window."
            )

        # ----------------------------------------------------
        # Frame count validation
        # ----------------------------------------------------

        if len(output_frames) != len(indices):

            raise RuntimeError(
                "Output frame count mismatch: "
                f"{len(output_frames)} != "
                f"{len(indices)}"
            )

        # ----------------------------------------------------
        # Weights
        # ----------------------------------------------------

        weights = (
            create_pyramid_weights(
                len(indices),
                is_last_window=is_last,
            )
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        for local_index, frame in enumerate(
            output_frames
        ):

            global_index = (
                indices[local_index]
            )

            frame_np = np.asarray(
                frame,
                dtype=np.float32,
            )

            weight = (
                weights[local_index]
            )

            result_accum[
                global_index
            ] += (
                frame_np * weight
            )

            weight_accum[
                global_index
            ] += weight

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del output
        del output_frames
        del chunk
        del generator

        gc.collect()

        torch.cuda.empty_cache()

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        log(
            f"GPU allocated after window: "
            f"{allocated:.2f} GB"
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    log(
        "Fusing context windows..."
    )

    denominator = np.maximum(
        weight_accum,
        1e-8,
    )

    result_accum /= (
        denominator[
            :, None, None, None
        ]
    )

    result_accum = np.clip(
        result_accum,
        0,
        255,
    ).astype(
        np.uint8
    )

    result_frames = [
        Image.fromarray(
            frame,
            "RGB",
        )

        for frame in result_accum
    ]

    log(
        f"Generated frames: "
        f"{len(result_frames)}"
    )

    return result_frames


# ============================================================
# Save video
# ============================================================

def save_video(
    frames,
    output_path,
):

    output_dir = os.path.dirname(
        output_path
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    frame_pattern = os.path.join(
        output_dir,
        "anime_%06d.png",
    )

    log(
        "Saving generated frames..."
    )

    # --------------------------------------------------------
    # Clean old frames
    # --------------------------------------------------------

    for name in os.listdir(
        output_dir
    ):

        if (
            name.startswith("anime_")
            and name.endswith(".png")
        ):

            try:

                os.remove(
                    os.path.join(
                        output_dir,
                        name,
                    )
                )

            except Exception:

                pass

    # --------------------------------------------------------
    # Save PNG frames
    # --------------------------------------------------------

    for index, frame in enumerate(
        frames
    ):

        frame.save(
            os.path.join(
                output_dir,
                f"anime_{index:06d}.png",
            )
        )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    command = [
        "ffmpeg",

        "-hide_banner",
        "-loglevel",
        "error",

        "-framerate",
        str(FPS),

        "-i",
        frame_pattern,

        "-c:v",
        "libx264",

        "-crf",
        "19",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        output_path,

        "-y",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Video encoding failed:\n"
            + result.stderr
        )

    log(
        "Generated video encoded."
    )

    return output_path


# ============================================================
# Extract audio
# ============================================================

def extract_audio(
    input_video,
    audio_path,
    audio_wav,
):

    # --------------------------------------------------------
    # Original audio
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "ffmpeg",

            "-hide_banner",
            "-loglevel",
            "error",

            "-i",
            input_video,

            "-vn",

            "-acodec",
            "copy",

            audio_path,

            "-y",
        ],
        capture_output=True,
        text=True,
    )

    has_audio = (
        result.returncode == 0
        and os.path.isfile(audio_path)
        and os.path.getsize(
            audio_path
        ) > 0
    )

    if not has_audio:

        return False, False

    # --------------------------------------------------------
    # WAV for Wav2Lip
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "ffmpeg",

            "-hide_banner",
            "-loglevel",
            "error",

            "-i",
            input_video,

            "-vn",

            "-acodec",
            "pcm_s16le",

            "-ar",
            "16000",

            audio_wav,

            "-y",
        ],
        capture_output=True,
        text=True,
    )

    has_wav = (
        result.returncode == 0
        and os.path.isfile(audio_wav)
        and os.path.getsize(
            audio_wav
        ) > 0
    )

    return has_audio, has_wav


# ============================================================
# Wav2Lip
# ============================================================

def apply_wav2lip(
    anime_video,
    audio_path,
    output_path,
):

    checkpoint = (
        "/runpod-volume/wav2lip.pth"
    )

    inference_script = (
        "/wav2lip/inference.py"
    )

    if not os.path.isfile(
        checkpoint
    ):

        log(
            "Wav2Lip checkpoint not found."
        )

        return False

    if not os.path.isfile(
        inference_script
    ):

        log(
            "Wav2Lip inference.py not found."
        )

        return False

    command = [
        "python3",

        inference_script,

        "--checkpoint_path",
        checkpoint,

        "--face",
        anime_video,

        "--audio",
        audio_path,

        "--outfile",
        output_path,

        "--nosmooth",
    ]

    log(
        "Running Wav2Lip..."
    )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd="/wav2lip",
    )

    if result.returncode != 0:

        log(
            "Wav2Lip failed."
        )

        log(
            result.stderr[-4000:]
        )

        return False

    if not os.path.isfile(
        output_path
    ):

        return False

    if os.path.getsize(
        output_path
    ) == 0:

        return False

    log(
        "Wav2Lip completed."
    )

    return True


# ============================================================
# Attach audio
# ============================================================

def attach_audio(
    video_path,
    audio_path,
    output_path,
):

    command = [
        "ffmpeg",

        "-hide_banner",
        "-loglevel",
        "error",

        "-i",
        video_path,

        "-i",
        audio_path,

        "-map",
        "0:v:0",

        "-map",
        "1:a:0",

        "-c:v",
        "copy",

        "-c:a",
        "aac",

        "-shortest",

        "-movflags",
        "+faststart",

        output_path,

        "-y",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Audio muxing failed:\n"
            + result.stderr
        )

    return output_path


# ============================================================
# Handler
# ============================================================

def handler(job):

    job_input = job.get(
        "input",
        {}
    )

    video_b64 = job_input.get(
        "video"
    )

    if not video_b64:

        raise ValueError(
            "input.video is required."
        )

    log("=" * 70)
    log("NEW JOB")
    log("=" * 70)

    # --------------------------------------------------------
    # Decode input
    # --------------------------------------------------------

    try:

        video_bytes = (
            base64.b64decode(
                video_b64
            )
        )

    except Exception as error:

        raise ValueError(
            f"Invalid base64 video: {error}"
        )

    log(
        f"Input video: "
        f"{len(video_bytes) / 1024**2:.2f} MB"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    load_model()

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="animatediff_"
    ) as tmpdir:

        log(
            f"Temporary directory: "
            f"{tmpdir}"
        )

        input_video = os.path.join(
            tmpdir,
            "input.mp4",
        )

        frames_dir = os.path.join(
            tmpdir,
            "frames",
        )

        anime_video = os.path.join(
            tmpdir,
            "anime_no_audio.mp4",
        )

        wav2lip_video = os.path.join(
            tmpdir,
            "wav2lip.mp4",
        )

        output_video = os.path.join(
            tmpdir,
            "output.mp4",
        )

        audio_path = os.path.join(
            tmpdir,
            "audio.aac",
        )

        audio_wav = os.path.join(
            tmpdir,
            "audio.wav",
        )

        # ----------------------------------------------------
        # Write input
        # ----------------------------------------------------

        with open(
            input_video,
            "wb",
        ) as file:

            file.write(
                video_bytes
            )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        has_audio, has_audio_wav = (
            extract_audio(
                input_video,
                audio_path,
                audio_wav,
            )
        )

        log(
            f"Audio available: "
            f"{has_audio}"
        )

        log(
            f"WAV available: "
            f"{has_audio_wav}"
        )

        # ====================================================
        # ORIGINAL FRAMES
        # ====================================================

        frame_paths = extract_frames(
            input_video,
            frames_dir,
        )

        # ----------------------------------------------------
        # Load original frames
        # ----------------------------------------------------

        input_frames = []

        for path in frame_paths:

            with Image.open(
                path
            ) as image:

                input_frames.append(
                    image.convert(
                        "RGB"
                    )
                )

        original_frame_count = len(
            input_frames
        )

        if original_frame_count <= 0:

            raise RuntimeError(
                "Original frame count is zero."
            )

        log(
            f"Original frame count: "
            f"{original_frame_count}"
        )

        # ====================================================
        # EXTEND FRAME SEQUENCE
        # ====================================================

        extended_frames = (
            extend_frames_for_tail(
                input_frames,
                EXTEND_FRAMES,
            )
        )

        extended_frame_count = len(
            extended_frames
        )

        expected_extended_count = (
            original_frame_count
            + EXTEND_FRAMES
        )

        if (
            extended_frame_count
            !=
            expected_extended_count
        ):

            raise RuntimeError(
                "Unexpected extended frame count: "
                f"got {extended_frame_count}, "
                f"expected {expected_extended_count}"
            )

        log(
            f"Extended frame count: "
            f"{extended_frame_count}"
        )

        # ====================================================
        # AnimateDiff
        # ====================================================

        result_frames_extended = (
            process_video_frames(
                extended_frames
            )
        )

        if (
            len(result_frames_extended)
            !=
            extended_frame_count
        ):

            raise RuntimeError(
                "AnimateDiff returned an unexpected "
                "number of frames: "
                f"got {len(result_frames_extended)}, "
                f"expected {extended_frame_count}"
            )

        # ====================================================
        # REMOVE EXTENSION
        # ====================================================
        #
        # ここが非常に重要。
        #
        # AnimateDiffには
        #
        #   original + extension
        #
        # を渡したが、最終動画には
        #
        #   original
        #
        # だけを残す。
        #

        log("=" * 70)
        log("REMOVING TAIL EXTENSION")
        log("=" * 70)

        result_frames = (
            result_frames_extended[
                :original_frame_count
            ]
        )

        if (
            len(result_frames)
            !=
            original_frame_count
        ):

            raise RuntimeError(
                "Failed to restore original "
                "frame count: "
                f"got {len(result_frames)}, "
                f"expected {original_frame_count}"
            )

        log(
            f"Final frame count: "
            f"{len(result_frames)}"
        )

        log(
            f"Removed extension frames: "
            f"{EXTEND_FRAMES}"
        )

        # ----------------------------------------------------
        # Release extended frames
        # ----------------------------------------------------

        del result_frames_extended
        del extended_frames

        gc.collect()

        # ====================================================
        # Encode
        # ====================================================

        save_video(
            result_frames,
            anime_video,
        )

        # ====================================================
        # Wav2Lip
        # ====================================================

        wav2lip_success = False

        if has_audio_wav:

            wav2lip_success = (
                apply_wav2lip(
                    anime_video,
                    audio_wav,
                    wav2lip_video,
                )
            )

        # ====================================================
        # Final video
        # ====================================================

        if (
            wav2lip_success
            and os.path.isfile(
                wav2lip_video
            )
        ):

            log(
                "Using Wav2Lip result."
            )

            attach_audio(
                wav2lip_video,
                audio_path,
                output_video,
            )

        elif has_audio:

            log(
                "Wav2Lip unavailable/failed."
            )

            log(
                "Attaching original audio."
            )

            attach_audio(
                anime_video,
                audio_path,
                output_video,
            )

        else:

            log(
                "Input has no audio."
            )

            subprocess.run(
                [
                    "ffmpeg",

                    "-hide_banner",
                    "-loglevel",
                    "error",

                    "-i",
                    anime_video,

                    "-c",
                    "copy",

                    output_video,

                    "-y",
                ],
                check=True,
            )

        # ====================================================
        # Base64
        # ====================================================

        with open(
            output_video,
            "rb",
        ) as file:

            result_b64 = (
                base64.b64encode(
                    file.read()
                ).decode(
                    "utf-8"
                )
            )

        log("=" * 70)
        log("JOB COMPLETED")
        log("=" * 70)

        return {
            "video": result_b64,

            "frames": original_frame_count,

            "width": OUTPUT_WIDTH,

            "height": OUTPUT_HEIGHT,

            "fps": FPS,
        }


# ============================================================
# RunPod Serverless
# ============================================================

if __name__ == "__main__":

    log("=" * 70)
    log("Starting RunPod Serverless Worker")
    log("=" * 70)

    log(
        "Expected environment:"
    )

    log(
        "CUDA 12.4"
    )

    log(
        "PyTorch 2.5.1+cu124"
    )

    log(
        "Diffusers 0.31.0"
    )

    runpod.serverless.start(
        {
            "handler": handler
        }
    )


# rebuild trigger
