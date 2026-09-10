from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from surgent.schemas import FrameSet


def extract_frames(
    video_path: str,
    output_dir: str,
    max_frames: int = 16,
    stride_s: float = 1.0,
    timestamps_s: Optional[List[float]] = None,
) -> FrameSet:
    """
    Extract frames from a video at a fixed time stride.

    Returns paths to saved PNGs and their timestamps in seconds.
    """
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "opencv-python is required for frame extraction."
        ) from exc

    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_paths: List[str] = []
    timestamps: List[float] = []

    if timestamps_s:
        targets = sorted(
            {
                max(0.0, float(t))
                for t in timestamps_s
            }
        )[:max_frames]
        target_frames = [
            int(round(t * fps)) if fps > 0 else int(round(t))
            for t in targets
        ]
        for idx, frame_idx in enumerate(target_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ts_s = (frame_idx / fps) if fps > 0 else float(idx)
            out_path = output_root / f"frame_{idx:04d}.png"
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(str(out_path))
            timestamps.append(ts_s)
    else:
        stride_frames = int(round(stride_s * fps)) if fps > 0 else 1
        if stride_frames < 1:
            stride_frames = 1

        frame_idx = 0
        saved = 0

        while saved < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride_frames == 0:
                ts_s = (frame_idx / fps) if fps > 0 else float(saved)
                out_path = output_root / f"frame_{saved:04d}.png"
                cv2.imwrite(str(out_path), frame)
                frame_paths.append(str(out_path))
                timestamps.append(ts_s)
                saved += 1

            frame_idx += 1

    cap.release()
    return FrameSet(frame_paths=frame_paths, timestamps_s=timestamps)


def extract_laparoscopy_frames(
    video_path: str,
    output_dir: str = "tmp/surgical_agent_frames",
    max_frames: int = 16,
    stride_s: float = 1.0,
    timestamps_s: Optional[List[float]] = None,
) -> dict:
    """
    Extracts frames from a laparoscopy video and returns paths + timestamps.
    """
    if os.getenv("SURGICAL_AGENT_VERBOSE") == "1":
        print("[tool] extract_laparoscopy_frames: starting")
    frame_set = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        max_frames=max_frames,
        stride_s=stride_s,
        timestamps_s=timestamps_s,
    )
    if os.getenv("SURGICAL_AGENT_VERBOSE") == "1":
        count = len(frame_set.frame_paths)
        first_ts = frame_set.timestamps_s[0] if count else None
        last_ts = frame_set.timestamps_s[-1] if count else None
        print(
            f"[tool] extract_laparoscopy_frames: extracted={count}, "
            f"first_ts={first_ts}, last_ts={last_ts}"
        )
    return {
        "frame_paths": list(frame_set.frame_paths),
        "timestamps_s": list(frame_set.timestamps_s),
    }
