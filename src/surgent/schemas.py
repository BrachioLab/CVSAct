from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

CRITERION_DEFINITIONS: Dict[str, str] = {
    "C1": "Two and only two tubular structures are visible entering the gallbladder.",
    "C2": "The hepatocystic triangle is cleared of fat and fibrous tissue.",
    "C3": "The lower third of the gallbladder is detached from the liver bed.",
}

DEFAULT_CVS_CRITERIA: List[str] = ["C1", "C2", "C3"]


@dataclass
class FrameSet:
    """A small set of extracted frame paths and their timestamps (seconds)."""

    frame_paths: List[str]
    timestamps_s: List[float]


@dataclass
class CVSRequest:
    """Request for Critical View of Safety (CVS) criterion predictions."""

    video_path: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    criteria: List[str] = field(default_factory=lambda: list(DEFAULT_CVS_CRITERIA))
    criterion_definitions: Dict[str, str] = field(
        default_factory=lambda: dict(CRITERION_DEFINITIONS)
    )
    max_frames: int = 16
    stride_s: float = 1.0
    output_dir: str = "tmp/surgical_agent_frames"
    model_path: Optional[str] = None


@dataclass
class CVSResult:
    """Per-criterion prediction output."""

    predictions: Dict[str, str]
    confidences: Dict[str, float]
    notes: str = "Research-only output. Not for clinical use."
