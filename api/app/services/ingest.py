"""Upload handling + video frame extraction (ffmpeg). CPU-only, lives in api so the
GPU worker is never blocked decoding video. Frame caps are enforced HERE so the
worker never receives more frames than the model's caps allow."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class Ingested:
    image_paths: list[str]
    source_kind: str  # "images" | "video" | "single"


def is_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTS


def is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def extract_video_frames(
    video_path: str, out_dir: str, *, fps: float, max_frames: int
) -> list[str]:
    """Extract up to max_frames frames at the given fps using ffmpeg."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")
    )
    return frames[:max_frames]


def prepare_inputs(
    job_dir: str,
    saved_files: list[str],
    *,
    fps: float,
    max_frames: int,
) -> Ingested:
    """Turn whatever was uploaded into a capped list of image paths."""
    videos = [f for f in saved_files if is_video(f)]
    images = [f for f in saved_files if is_image(f)]

    if videos:
        frames_dir = os.path.join(job_dir, "frames")
        frames = extract_video_frames(
            videos[0], frames_dir, fps=fps, max_frames=max_frames
        )
        if not frames:
            raise ValueError("ffmpeg produced no frames from the video")
        return Ingested(image_paths=frames, source_kind="video")

    if not images:
        raise ValueError("no usable image or video files in upload")

    images = sorted(images)[:max_frames]
    kind = "single" if len(images) == 1 else "images"
    return Ingested(image_paths=images, source_kind=kind)
