import runpod
import base64
import os
import tempfile
import subprocess
import json as _json
import gc

import torch
import numpy as np

from PIL import Image

from diffusers import (
    AnimateDiffVideoToVideoPipeline,
    MotionAdapter,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"
DTYPE = torch.float16

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

CHECKPOINT_PATH = "/workspace/models/counterfeit_v30.safetensors"

# ComfyUI:
#   mm_sd_v15_v2.ckpt
#
# Diffusers >= 0.30 supports loading original AnimateDiff
# checkpoint format directly with MotionAdapter.from_single_file().
MOTION_MODULE_PATH = "/workspace/models/mm_sd_v15_v2.ckpt"

EASY_NEGATIVE_PATH = "/workspace/embeddings/EasyNegativeV2.safetensors"

# ------------------------------------------------------------
# ComfyUI workflow equivalent
# ------------------------------------------------------------

WIDTH = 320
HEIGHT = 568

FPS = 8

NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.0

# ComfyUI KSampler:
# denoise = 0.55
STRENGTH = 0.55

SEED = 41868074274227

# AnimateDiff context
CONTEXT_LENGTH = 16
CONTEXT_OVERLAP = 4

# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

POSITIVE_PROMPT = (
    "anime style, studio ghibli, cel shading, vivid colors, "
    "masterpiece, high quality, detailed illustration"
)

NEGATIVE_PROMPT = (
    "EasyNegativeV2, "
    "worst quality, low quality, blurry, watermark, realistic, "
    "photography, noise, grain, flickering, particles, "
    "sparkles, floating objects"
)

# ============================================================
# Global pipeline
# ============================================================

pipe = None


# ============================================================
# Utility
# ============================================================

def cleanup_memory():
    """
    GPU memory cleanup.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# Load model
# ============================================================

def load_model():
    global pipe

    if pipe is not None:
        return

    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not os.path.exists(MOTION_MODULE_PATH):
        raise FileNotFoundError(
            f"AnimateDiff motion module not found: {MOTION_MODULE_PATH}"
        )

    # --------------------------------------------------------
    # 1. Load AnimateDiff v2 motion module
    # --------------------------------------------------------

    print("Loading AnimateDiff v2 motion module:")
    print(MOTION_MODULE_PATH)

    adapter = MotionAdapter.from_single_file(
        MOTION_MODULE_PATH,
        torch_dtype=DTYPE,
    )

    print("AnimateDiff v2 motion module loaded.")

    # --------------------------------------------------------
    # 2. Load Counterfeit V3.0
    # --------------------------------------------------------

    print("Loading Counterfeit V3.0...")

    sd_pipe = StableDiffusionPipeline.from_single_file(
        CHECKPOINT_PATH,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )

    print("Counterfeit V3.0 loaded.")

    # --------------------------------------------------------
    # 3. Build AnimateDiff Video-to-Video pipeline
    # --------------------------------------------------------

    print("Building AnimateDiffVideoToVideoPipeline...")

    pipe = AnimateDiffVideoToVideoPipeline(
        vae=sd_pipe.vae,
        text_encoder=sd_pipe.text_encoder,
        tokenizer=sd_pipe.tokenizer,
        unet=sd_pipe.unet,
        motion_adapter=adapter,
        scheduler=sd_pipe.scheduler,
        feature_extractor=None,
        image_encoder=None,
    )

    # --------------------------------------------------------
    # 4. Scheduler
    #
    # ComfyUI:
    #
    # sampler = dpmpp_2s_ancestral
    # scheduler = karras
    #
    # Diffusers does not provide an exact drop-in
    # dpmpp_2s_ancestral implementation.
    #
    # Use DPM-Solver++ 2nd order + Karras as the closest
    # practical Diffusers scheduler for this workflow.
    # --------------------------------------------------------

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        use_karras_sigmas=True,
        lower_order_final=True,
    )

    print("Scheduler:")
    print("  algorithm = DPM-Solver++")
    print("  order     = 2")
    print("  Karras    = True")

    # --------------------------------------------------------
    # 5. Textual inversion
    # --------------------------------------------------------

    if os.path.exists(EASY_NEGATIVE_PATH):

        print("Loading EasyNegativeV2...")

        pipe.load_textual_inversion(
            EASY_NEGATIVE_PATH,
            token="EasyNegativeV2",
        )

        print("EasyNegativeV2 loaded.")

    else:
        print(
            "WARNING: EasyNegativeV2 not found:"
            f" {EASY_NEGATIVE_PATH}"
        )

    # --------------------------------------------------------
    # 6. Move to GPU
    # --------------------------------------------------------

    pipe = pipe.to(DEVICE)

    # --------------------------------------------------------
    # 7. Memory optimizations
    # --------------------------------------------------------

    try:
        pipe.enable_vae_slicing()
        print("VAE slicing enabled.")
    except Exception as e:
        print(f"VAE slicing unavailable: {e}")

    try:
        pipe.enable_vae_tiling()
        print("VAE tiling enabled.")
    except Exception as e:
        print(f"VAE tiling unavailable: {e}")

    # xformers is optional.
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers attention enabled.")
    except Exception as e:
        print(f"xFormers unavailable: {e}")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally DO NOT use:
    #
    # pipe.enable_model_cpu_offload()
    #
    # because the RunPod GPU is used continuously and
    # CPU/GPU transfers can make video inference much slower.
    # --------------------------------------------------------

    print("=" * 60)
    print("MODEL READY")
    print("=" * 60)

    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    print("Motion:")
    print(MOTION_MODULE_PATH)

    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"FPS: {FPS}")
    print(f"Steps: {NUM_INFERENCE_STEPS}")
    print(f"CFG: {GUIDANCE_SCALE}")
    print(f"Strength: {STRENGTH}")
    print(f"Seed: {SEED}")
    print(
        f"Context: {CONTEXT_LENGTH}, "
        f"overlap={CONTEXT_OVERLAP}"
    )


# ============================================================
# Video loading
# ============================================================

def extract_video_frames(
    input_video,
    frames_dir,
):
    """
    ComfyUI VHS_LoadVideo equivalent:

        force_rate = 8
        custom_width = 320
        custom_height = 568
        select_every_nth = 1
    """

    os.makedirs(frames_dir, exist_ok=True)

    output_pattern = os.path.join(
        frames_dir,
        "frame_%06d.png",
    )

    print(
        f"Extracting video: "
        f"{WIDTH}x{HEIGHT} @ {FPS}fps"
    )

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",

            # Do not apply automatic rotation.
            "-noautorotate",

            "-i",
            input_video,

            # ComfyUI:
            # force_rate = 8
            "-vf",
            f"fps={FPS},scale={WIDTH}:{HEIGHT}:"
            "flags=lanczos",

            "-vsync",
            "0",

            output_pattern,

            "-y",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg frame extraction failed:\n"
            + result.stderr
        )

    files = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(".png")
        ]
    )

    if not files:
        raise RuntimeError(
            "No video frames were extracted."
        )

    print(
        f"Extracted {len(files)} frames "
        f"({len(files) / FPS:.2f} sec)"
    )

    frames = []

    for path in files:
        image = Image.open(path).convert("RGB")
        frames.append(image)

    return frames


# ============================================================
# AnimateDiff processing
# ============================================================

def process_single_context(
    frames,
    generator,
):
    """
    Process one AnimateDiff context.

    ComfyUI:
        context_length = 16
    """

    print(
        f"  Processing context: "
        f"{len(frames)} frames"
    )

    with torch.inference_mode():

        result = pipe(
            prompt=POSITIVE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,

            video=frames,

            height=HEIGHT,
            width=WIDTH,

            strength=STRENGTH,

            num_inference_steps=NUM_INFERENCE_STEPS,

            guidance_scale=GUIDANCE_SCALE,

            generator=generator,

            # Do not force a resize inside the pipeline.
            resize_mode="default",

            # Diffusers' video-to-video implementation
            # can otherwise alter the effective number of
            # inference steps depending on strength.
            #
            # We keep the normal behavior here because this
            # corresponds most closely to img2img/video2video.
        )

    return result.frames[0]


# ============================================================
# Context processing
# ============================================================

def process_frames_with_context(
    frames,
):
    """
    Approximate ComfyUI AnimateDiff context behavior.

    ComfyUI:
        context_length = 16
        context_overlap = 4
        closed_loop = false
        fuse_method = pyramid
        use_on_equal_length = false
        start_percent = 0
        guarantee_steps = 1

    Diffusers does not expose ADE_LoopedUniformContextOptions
    directly, so this function implements overlapping
    temporal windows externally.

    For <=16 frames:
        process as one context.

    For >16 frames:
        process overlapping 16-frame windows.

    Results are fused using pyramid-like weighted blending.
    """

    total_frames = len(frames)

    print("=" * 60)
    print("AnimateDiff processing")
    print("=" * 60)

    print(f"Total frames : {total_frames}")
    print(f"Context      : {CONTEXT_LENGTH}")
    print(f"Overlap      : {CONTEXT_OVERLAP}")

    # --------------------------------------------------------
    # Case 1:
    # <=16 frames
    # --------------------------------------------------------

    if total_frames <= CONTEXT_LENGTH:

        print(
            "Video fits into one AnimateDiff context."
        )

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(SEED)

        return process_single_context(
            frames,
            generator,
        )

    # --------------------------------------------------------
    # Case 2:
    # Long video
    # --------------------------------------------------------

    stride = (
        CONTEXT_LENGTH -
        CONTEXT_OVERLAP
    )

    print(
        f"Context stride = {stride}"
    )

    # --------------------------------------------------------
    # Output accumulation
    #
    # Each frame can be produced by multiple contexts.
    # We accumulate weighted pixels and normalize later.
    #
    # This approximates:
    #
    #     fuse_method = pyramid
    #
    # --------------------------------------------------------

    output_accumulator = [
        None
        for _ in range(total_frames)
    ]

    weight_accumulator = np.zeros(
        total_frames,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Generate context windows
    # --------------------------------------------------------

    context_starts = []

    start = 0

    while start < total_frames:

        end = min(
            start + CONTEXT_LENGTH,
            total_frames,
        )

        context_starts.append(
            (start, end)
        )

        if end >= total_frames:
            break

        start += stride

    # --------------------------------------------------------
    # Process each context
    # --------------------------------------------------------

    for context_index, (start, end) in enumerate(
        context_starts
    ):

        print(
            f"[Context {context_index + 1}/"
            f"{len(context_starts)}] "
            f"frames {start} - {end - 1}"
        )

        chunk = frames[start:end]

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use the SAME seed for each context.
        #
        # This is closer to a deterministic ComfyUI workflow
        # than using a continuously changing seed.
        # ----------------------------------------------------

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(SEED)

        chunk_result = process_single_context(
            chunk,
            generator,
        )

        chunk_length = len(chunk_result)

        # ----------------------------------------------------
        # Pyramid-like blending weights
        #
        # Stronger weight in the middle of the context.
        # Lower weight at the edges.
        # ----------------------------------------------------

        weights = np.ones(
            chunk_length,
            dtype=np.float32,
        )

        if chunk_length > 1:

            center = (
                chunk_length - 1
            ) / 2.0

            max_distance = max(
                center,
                1.0,
            )

            for i in range(chunk_length):

                distance = abs(
                    i - center
                )

                normalized = (
                    distance /
                    max_distance
                )

                # Pyramid:
                #
                # center = 1.0
                # edge   = 0.25
                #
                weights[i] = (
                    1.0 -
                    0.75 * normalized
                )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        for local_index in range(
            chunk_length
        ):

            global_index = (
                start + local_index
            )

            if global_index >= total_frames:
                continue

            frame = chunk_result[
                local_index
            ]

            frame_np = np.asarray(
                frame,
                dtype=np.float32,
            )

            weight = weights[
                local_index
            ]

            if output_accumulator[
                global_index
            ] is None:

                output_accumulator[
                    global_index
                ] = (
                    frame_np * weight
                )

            else:

                output_accumulator[
                    global_index
                ] += (
                    frame_np * weight
                )

            weight_accumulator[
                global_index
            ] += weight

        cleanup_memory()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    result_frames = []

    for i in range(total_frames):

        if (
            output_accumulator[i]
            is None
        ):
            raise RuntimeError(
                f"Frame {i} was not generated."
            )

        image_np = (
            output_accumulator[i] /
            max(
                weight_accumulator[i],
                1e-8,
            )
        )

        image_np = np.clip(
            image_np,
            0,
            255,
        ).astype(
            np.uint8
        )

        result_frames.append(
            Image.fromarray(
                image_np,
                mode="RGB",
            )
        )

    return result_frames


# ============================================================
# Audio extraction
# ============================================================

def extract_audio(
    input_video,
    audio_path,
    audio_wav,
):
    """
    Extract original audio.
    """

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

    if result.returncode != 0:
        print(
            "No original audio found."
        )
        return False

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

    if result.returncode != 0:
        print(
            "Could not convert audio to WAV."
        )
        return False

    return (
        os.path.exists(audio_path)
        and os.path.getsize(audio_path) > 0
    )


# ============================================================
# Wav2Lip
# ============================================================

def apply_wav2lip(
    anime_video,
    audio_path,
    output_path,
):

    try:

        result = subprocess.run(
            [
                "python3",
                "/wav2lip/inference.py",

                "--checkpoint_path",
                "/workspace/wav2lip.pth",

                "--face",
                anime_video,

                "--audio",
                audio_path,

                "--outfile",
                output_path,

                "--nosmooth",
            ],

            capture_output=True,
            text=True,

            cwd="/wav2lip",
        )

        if result.returncode == 0:

            print(
                "Wav2Lip processing completed."
            )

            return True

        print(
            "Wav2Lip failed:"
        )

        print(
            result.stderr[-2000:]
        )

        return False

    except Exception as e:

        print(
            f"Wav2Lip exception: {e}"
        )

        return False


# ============================================================
# Save frames
# ============================================================

def save_frames(
    frames,
    output_dir,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    for i, frame in enumerate(frames):

        path = os.path.join(
            output_dir,
            f"out_{i:06d}.png",
        )

        frame.save(
            path,
            format="PNG",
        )


# ============================================================
# Encode video
# ============================================================

def encode_video(
    frames_dir,
    output_video,
):

    input_pattern = os.path.join(
        frames_dir,
        "out_%06d.png",
    )

    print(
        "Encoding video..."
    )

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",

            "-framerate",
            str(FPS),

            "-i",
            input_pattern,

            "-c:v",
            "libx264",

            # Higher quality than the original code.
            "-crf",
            "15",

            "-preset",
            "medium",

            "-pix_fmt",
            "yuv420p",

            output_video,

            "-y",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Video encoding failed:\n"
            + result.stderr
        )

    print(
        f"Video encoded: {output_video}"
    )


# ============================================================
# Probe original video
# ============================================================

def probe_video(
    input_video,
):

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",

            "-print_format",
            "json",

            "-show_streams",

            "-show_entries",
            "stream=codec_type,width,height,"
            "r_frame_rate,duration:stream_tags=rotate",

            input_video,
        ],

        capture_output=True,
        text=True,
    )

    try:
        return _json.loads(
            result.stdout
        )

    except Exception:

        return {}


# ============================================================
# Main handler
# ============================================================

def handler(job):

    job_input = job["input"]

    video_b64 = job_input.get(
        "video",
        "",
    )

    if not video_b64:

        raise ValueError(
            "Input video is missing."
        )

    # --------------------------------------------------------
    # Decode input
    # --------------------------------------------------------

    video_bytes = base64.b64decode(
        video_b64
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    load_model()

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmpdir:

        input_video = os.path.join(
            tmpdir,
            "input.mp4",
        )

        frames_dir = os.path.join(
            tmpdir,
            "frames",
        )

        output_frames_dir = os.path.join(
            tmpdir,
            "output_frames",
        )

        anime_no_audio = os.path.join(
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
        # Save input
        # ----------------------------------------------------

        with open(
            input_video,
            "wb",
        ) as f:

            f.write(
                video_bytes
            )

        # ----------------------------------------------------
        # Probe
        # ----------------------------------------------------

        probe_data = probe_video(
            input_video
        )

        for stream in probe_data.get(
            "streams",
            [],
        ):

            if (
                stream.get(
                    "codec_type"
                )
                == "video"
            ):

                print(
                    "Original video:"
                )

                print(
                    f"  {stream.get('width')}x"
                    f"{stream.get('height')}"
                )

                print(
                    f"  FPS: "
                    f"{stream.get('r_frame_rate')}"
                )

                print(
                    f"  rotation: "
                    f"{stream.get('tags', {}).get('rotate', 0)}"
                )

                break

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        has_audio = extract_audio(
            input_video,
            audio_path,
            audio_wav,
        )

        print(
            f"Audio: {has_audio}"
        )

        # ----------------------------------------------------
        # Extract frames
        #
        # IMPORTANT:
        #
        # Unlike the previous code, this always uses
        # exactly the same resolution as ComfyUI:
        #
        # 320 x 568
        #
        # This is critical for comparison.
        # ----------------------------------------------------

        input_frames = extract_video_frames(
            input_video,
            frames_dir,
        )

        print(
            f"Input frames: "
            f"{len(input_frames)}"
        )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        result_frames = (
            process_frames_with_context(
                input_frames
            )
        )

        print(
            f"Generated frames: "
            f"{len(result_frames)}"
        )

        # ----------------------------------------------------
        # Save PNG frames
        # ----------------------------------------------------

        save_frames(
            result_frames,
            output_frames_dir,
        )

        # ----------------------------------------------------
        # Encode anime video
        # ----------------------------------------------------

        encode_video(
            output_frames_dir,
            anime_no_audio,
        )

        # ----------------------------------------------------
        # Wav2Lip
        # ----------------------------------------------------

        wav2lip_success = False

        if (
            has_audio
            and os.path.exists(audio_wav)
            and os.path.exists(
                "/workspace/wav2lip.pth"
            )
        ):

            print(
                "Running Wav2Lip..."
            )

            wav2lip_success = (
                apply_wav2lip(
                    anime_no_audio,
                    audio_wav,
                    wav2lip_video,
                )
            )

        # ----------------------------------------------------
        # Final audio/video mux
        # ----------------------------------------------------

        if (
            wav2lip_success
            and os.path.exists(
                wav2lip_video
            )
            and os.path.getsize(
                wav2lip_video
            ) > 0
        ):

            print(
                "Muxing Wav2Lip video "
                "with original audio..."
            )

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",

                    "-i",
                    wav2lip_video,

                    "-i",
                    audio_path,

                    "-c:v",
                    "copy",

                    "-c:a",
                    "aac",

                    "-shortest",

                    output_video,

                    "-y",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:

                print(
                    "Wav2Lip mux failed."
                )

                # fallback
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",

                        "-i",
                        anime_no_audio,

                        "-i",
                        audio_path,

                        "-c:v",
                        "copy",

                        "-c:a",
                        "aac",

                        "-shortest",

                        output_video,

                        "-y",
                    ],
                    check=True,
                )

        elif (
            has_audio
            and os.path.exists(audio_path)
        ):

            print(
                "Wav2Lip unavailable. "
                "Muxing original audio."
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",

                    "-i",
                    anime_no_audio,

                    "-i",
                    audio_path,

                    "-c:v",
                    "copy",

                    "-c:a",
                    "aac",

                    "-shortest",

                    output_video,

                    "-y",
                ],
                check=True,
            )

        else:

            print(
                "No audio. "
                "Using video only."
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",

                    "-i",
                    anime_no_audio,

                    "-c",
                    "copy",

                    output_video,

                    "-y",
                ],
                check=True,
            )

        # ----------------------------------------------------
        # Return Base64
        # ----------------------------------------------------

        with open(
            output_video,
            "rb",
        ) as f:

            result_b64 = (
                base64.b64encode(
                    f.read()
                ).decode()
            )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        cleanup_memory()

        return {
            "video": result_b64,
            "frames": len(result_frames),
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "steps": NUM_INFERENCE_STEPS,
            "cfg": GUIDANCE_SCALE,
            "strength": STRENGTH,
            "seed": SEED,
            "motion_module": "mm_sd_v15_v2.ckpt",
        }


# ============================================================
# RunPod
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
)
