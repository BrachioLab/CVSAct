from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from langchain.tools import tool

from .memory_hm3 import HM3Memory


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_bbox_norm(value: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    if len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return None
    x1 = _clamp(x1, 0.0, 1.0)
    y1 = _clamp(y1, 0.0, 1.0)
    x2 = _clamp(x2, 0.0, 1.0)
    y2 = _clamp(y2, 0.0, 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _parse_bbox_px(value: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return None
    x1 = int(round(_clamp(x1, 0.0, float(max(0, width - 1)))))
    y1 = int(round(_clamp(y1, 0.0, float(max(0, height - 1)))))
    x2 = int(round(_clamp(x2, 1.0, float(width))))
    y2 = int(round(_clamp(y2, 1.0, float(height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_norm_to_px(
    bbox_norm: Tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n = bbox_norm
    w = float(width)
    h = float(height)
    x1 = x1n * w
    y1 = y1n * h
    x2 = x2n * w
    y2 = y2n * h
    if margin_ratio > 0:
        margin_ratio = _clamp(float(margin_ratio), 0.0, 0.45)
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * margin_ratio
        y1 -= bh * margin_ratio
        x2 += bw * margin_ratio
        y2 += bh * margin_ratio
    x1 = int(round(_clamp(x1, 0.0, max(0.0, w - 1))))
    y1 = int(round(_clamp(y1, 0.0, max(0.0, h - 1))))
    x2 = int(round(_clamp(x2, 1.0, w)))
    y2 = int(round(_clamp(y2, 1.0, h)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return (x1, y1, x2, y2)


@tool
def zoom(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    iteration: int,
    bbox_norm_xyxy: Optional[List[float]] = None,
    bbox_xyxy: Optional[List[float]] = None,
    frame_crops: Optional[List[Dict[str, Any]]] = None,
    target_frame_id: Optional[str] = None,
    margin_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    Spatial Zoom Tool for short-term frame crops used in iterative evidence refinement.

    Purpose:
      `zoom` performs a deterministic spatial crop over the current short-term perception
      pool `P_s` and writes cropped images to temporary files. The tool is intentionally
      non-generative: it does not call an LLM and does not infer what to crop. The
      controller must provide the crop region (where to zoom) and optional target frame
      selection. This keeps decision authority in the high-level agent while making zooming
      behavior transparent and reproducible.

    Why this tool exists:
      In the coarse-to-fine loop, `clip_explorer` scopes a local temporal window and
      populates `P_s` with full-frame images. Sometimes the evidence needed by
      `clip_analyzer` is too small or partially obscured at full scale (e.g., tiny tubular
      structures, boundary planes, clip interfaces). `zoom` lets the agent magnify a region
      and then run `clip_analyzer` again on these cropped views without re-running temporal
      sampling.

    Input contract:
      interval:
        Dict like {"start": int, "end": int}. Used for traceability and result-memory
        linkage; not used to compute crops.
      reason:
        Controller-provided justification for why zoom is being applied.
      memory:
        Runtime HM3 memory object. `zoom` reads from `memory.sensory.short_term_pool` and
        overwrites it with zoomed frame entries.
      iteration:
        Runtime iteration index used in filenames and result memory.
      bbox_norm_xyxy (optional):
        Normalized crop box [x1, y1, x2, y2] in [0, 1] coordinates. Preferred input.
      bbox_xyxy (optional):
        Pixel crop box [x1, y1, x2, y2]. Used only if `bbox_norm_xyxy` is not provided.
      frame_crops (optional):
        Per-frame crop instructions for multi-frame zoom in one tool call.
        Each item can provide:
          {
            "frame_id": "<id from P_s>",
            "bbox_norm_xyxy": [x1,y1,x2,y2],  # preferred
            "bbox_xyxy": [x1,y1,x2,y2],        # optional pixel alternative
            "margin_ratio": 0.0..0.45          # optional per-frame override
          }
        This allows slightly different zoom regions per frame. If both per-frame and
        global boxes are provided, per-frame boxes take precedence for matching frames;
        global boxes are used as fallback for non-specified frames.
      target_frame_id (optional):
        If provided, crop only that frame in `P_s`; otherwise crop all frames currently
        present in `P_s`. Ignored when `frame_crops` is provided and contains explicit
        frame IDs.
      margin_ratio:
        Optional expansion around the requested box (fraction of box width/height). This
        guards against over-tight crops and preserves local context.

    Operational behavior:
      1. Validate that `P_s` has frames with readable paths.
      2. Build crop plan from `frame_crops` (if provided) plus optional global fallback.
      3. Select one, many, or all frames from `P_s` based on the crop plan.
      4. Resolve crop coordinates independently per selected frame.
      5. Crop each selected frame and save as PNG under a temp directory.
      6. Replace `P_s` with the zoomed-frame entries only.
      7. Write a result-memory entry with crop parameters and output paths.

    Side effects:
      - Updates `memory.sensory.short_term_pool` in-place to point to zoomed files.
      - Persists image files to:
          `SURGENT_ZOOM_DIR` or `<temp>/surgent_zoom`.
      - Appends tool output to HM3 Result Memory via `memory.record_result(...)`.
      - Does NOT clear `P_s`; the next `clip_analyzer` step is expected to consume it.

    Output schema (dict):
      {
        "tool": "zoom",
        "interval": {...},
        "reason": str,
        "target_frame_id": str | null,
        "requested_frame_crops": int,
        "applied_frame_crops": int,
        "bbox_mode": "norm" | "px" | "per_frame" | "mixed",
        "bbox_norm_xyxy": [float, float, float, float] | null,
        "num_source_frames": int,
        "num_zoomed_frames": int,
        "zoomed": [
          {
            "type": "frame",
            "frame_id": str,
            "path": str,
            "source_path": str,
            "crop_xyxy": [int, int, int, int],
            "source_size": [int, int]
          },
          ...
        ],
        "error": str | null
      }
    """
    source_pool = list(memory.sensory.short_term_pool)
    output: Dict[str, Any] = {
        "tool": "zoom",
        "interval": interval,
        "reason": reason,
        "target_frame_id": target_frame_id,
        "requested_frame_crops": len(frame_crops or []),
        "applied_frame_crops": 0,
        "bbox_mode": (
            "mixed"
            if (frame_crops and (bbox_norm_xyxy is not None or bbox_xyxy is not None))
            else (
                "per_frame"
                if frame_crops
                else ("norm" if bbox_norm_xyxy is not None else "px")
            )
        ),
        "bbox_norm_xyxy": bbox_norm_xyxy,
        "num_source_frames": len(source_pool),
        "num_zoomed_frames": 0,
        "zoomed": [],
        "error": None,
    }
    if not source_pool:
        output["error"] = "short_term_pool is empty; run clip_explorer first"
        memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
        return output

    norm_bbox = _parse_bbox_norm(bbox_norm_xyxy or [])
    if norm_bbox is None and bbox_norm_xyxy is not None:
        output["error"] = "invalid bbox_norm_xyxy; expected [x1,y1,x2,y2] with x2>x1, y2>y1"
        memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
        return output
    if norm_bbox is None and bbox_xyxy is None and not frame_crops:
        output["error"] = "missing crop box; provide bbox_norm_xyxy or bbox_xyxy"
        memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
        return output

    per_frame_plan: Dict[str, Dict[str, Any]] = {}
    if frame_crops:
        for spec in frame_crops:
            if not isinstance(spec, dict):
                continue
            frame_id = str(spec.get("frame_id", "")).strip()
            if not frame_id:
                continue
            spec_norm = _parse_bbox_norm(spec.get("bbox_norm_xyxy") or [])
            spec_px = spec.get("bbox_xyxy")
            if spec_norm is None and spec_px is None:
                continue
            spec_margin = spec.get("margin_ratio")
            try:
                parsed_margin = float(spec_margin) if spec_margin is not None else float(margin_ratio)
            except Exception:
                parsed_margin = float(margin_ratio)
            per_frame_plan[frame_id] = {
                "bbox_norm": spec_norm,
                "bbox_px": spec_px,
                "margin_ratio": _clamp(parsed_margin, 0.0, 0.45),
            }
        if not per_frame_plan and norm_bbox is None and bbox_xyxy is None:
            output["error"] = "frame_crops provided but no valid crop specs were found"
            memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
            return output

    selected = source_pool
    if per_frame_plan:
        selected = [
            item for item in source_pool
            if str(item.get("frame_id", "")) in per_frame_plan
            or norm_bbox is not None
            or bbox_xyxy is not None
        ]
        if not selected:
            output["error"] = "none of frame_crops.frame_id matched short_term_pool"
            memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
            return output
    elif target_frame_id:
        selected = [
            item for item in source_pool
            if str(item.get("frame_id", "")) == str(target_frame_id)
        ]
        if not selected:
            output["error"] = f"target_frame_id not found in short_term_pool: {target_frame_id}"
            memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
            return output

    _zoom_base = Path(os.getenv("SURGENT_ZOOM_DIR", str(Path(tempfile.gettempdir()) / "surgent_zoom")))
    _vid = getattr(getattr(memory, "metadata", None), "video_id", None) or "unknown"
    zoom_dir = _zoom_base / _vid
    zoom_dir.mkdir(parents=True, exist_ok=True)

    zoomed_entries: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected):
        source_path = str(item.get("path", ""))
        frame_id = str(item.get("frame_id", f"unknown_{idx}"))
        if not source_path or not os.path.exists(source_path):
            continue
        with Image.open(source_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            spec = per_frame_plan.get(frame_id)
            spec_norm = spec.get("bbox_norm") if spec else None
            spec_px = spec.get("bbox_px") if spec else None
            spec_margin = float(spec.get("margin_ratio")) if spec else float(margin_ratio)
            if spec_norm is not None:
                crop_xyxy = _bbox_norm_to_px(spec_norm, w, h, margin_ratio=spec_margin)
                bbox_mode = "norm"
            elif spec_px is not None:
                px_bbox = _parse_bbox_px(spec_px, w, h)
                if px_bbox is None:
                    continue
                crop_xyxy = px_bbox
                bbox_mode = "px"
            elif norm_bbox is not None:
                crop_xyxy = _bbox_norm_to_px(norm_bbox, w, h, margin_ratio=margin_ratio)
                bbox_mode = "norm"
            else:
                px_bbox = _parse_bbox_px(bbox_xyxy or [], w, h)
                if px_bbox is None:
                    continue
                crop_xyxy = px_bbox
                bbox_mode = "px"
            crop = rgb.crop(crop_xyxy)
            out_name = f"zoom_iter{int(iteration):03d}_{frame_id}_{idx:02d}.png"
            out_path = zoom_dir / out_name
            crop.save(out_path, format="PNG")
            zoomed_entries.append({
                "type": "frame",
                "frame_id": frame_id,
                "path": str(out_path),
                "source_path": source_path,
                "crop_xyxy": list(crop_xyxy),
                "source_size": [w, h],
                "bbox_mode": bbox_mode,
                "margin_ratio": spec_margin if spec is not None else float(margin_ratio),
            })

    if not zoomed_entries:
        output["error"] = "no zoomed frames were produced"
        memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
        return output

    memory.sensory.short_term_pool = zoomed_entries
    output["zoomed"] = zoomed_entries
    output["num_zoomed_frames"] = len(zoomed_entries)
    output["applied_frame_crops"] = len([z for z in zoomed_entries if str(z.get("frame_id", "")) in per_frame_plan])
    if norm_bbox is not None:
        output["bbox_norm_xyxy"] = [float(v) for v in norm_bbox]
    memory.record_result(tool="zoom", interval=interval, output=output, iteration=iteration)
    return output


def tool_registry() -> Dict[str, Any]:
    return {
        "zoom": zoom,
    }
