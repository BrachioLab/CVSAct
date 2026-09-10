from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
try:
    from langchain.tools import tool
except Exception:  # pragma: no cover - optional dependency for offline testing
    class _ToolShim:
        def __init__(self, fn: Any) -> None:
            self._fn = fn

        def invoke(self, args: Dict[str, Any]) -> Any:
            return self._fn(**args)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self._fn(*args, **kwargs)

    def tool(fn: Any) -> Any:  # type: ignore[misc]
        return _ToolShim(fn)

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - optional dependency for offline testing
    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: Any) -> None:
            self.content = content

    class SystemMessage:  # type: ignore[no-redef]
        def __init__(self, content: Any) -> None:
            self.content = content

from .memory_hm3 import HM3Memory


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _interval_limits() -> Tuple[int, int, int]:
    max_n = max(1, _safe_int(os.getenv("SURGENT_MAX_INTERVAL_FRAMES", "12"), 12))
    min_n = max(1, _safe_int(os.getenv("SURGENT_MIN_INTERVAL_FRAMES", "1"), 1))
    if min_n > max_n:
        min_n = max_n
    default_n = _safe_int(os.getenv("SURGENT_DEFAULT_INTERVAL_FRAMES", str(max_n)), max_n)
    default_n = min(max(default_n, min_n), max_n)
    return min_n, max_n, default_n


def _clip_limits() -> Tuple[int, int, int]:
    max_n = max(1, _safe_int(os.getenv("SURGENT_MAX_CLIP_FRAMES", "5"), 5))
    min_n = max(1, _safe_int(os.getenv("SURGENT_MIN_CLIP_FRAMES", "1"), 1))
    if min_n > max_n:
        min_n = max_n
    default_n = _safe_int(os.getenv("SURGENT_DEFAULT_CLIP_FRAMES", str(max_n)), max_n)
    default_n = min(max(default_n, min_n), max_n)
    return min_n, max_n, default_n


def _clamp_sample_count(requested: int, *, min_n: int, max_n: int, available: int) -> int:
    max_possible = min(max_n, max(0, int(available)))
    if max_possible <= 0:
        return 0
    min_effective = min(min_n, max_possible)
    return min(max(int(requested), min_effective), max_possible)


def _resolve_visible_bounds(memory: HM3Memory, total_frames: int) -> Tuple[int, int]:
    if total_frames <= 0:
        return 0, -1
    visible_start = max(0, int(getattr(memory, "visible_start_idx", 0) or 0))
    raw_visible_end = getattr(memory, "visible_end_idx", None)
    if raw_visible_end is None:
        visible_end = total_frames - 1
    else:
        visible_end = int(raw_visible_end)
    visible_end = min(max(0, visible_end), total_frames - 1)
    visible_start = min(visible_start, visible_end)
    return visible_start, visible_end


@tool
def interval_localizer(
    query: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
) -> Dict[str, Any]:
    """
    Interval Localizer. Interval Localizer identifies frame intervals T_long = [t_i, t_j]
    most relevant to the query by leveraging contextual signals in HM3. It determines
    sampling frame number N1 for each interval for adaptive granularity, uniformly samples
    N1 frames from T_long, composites frames into compact 3x2 grids with frame id
    overlays, and uses these grids to update the long-term perception pool P_l. By
    integrating evidence over stored information, it constrains the agent's working
    scope to query-aligned segments and surrounding context, reducing token consumption.

    Input:
      query: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)

    Output (dict):
      {
        "tool": "interval_localizer",
        "interval": {"start": int, "end": int},
        "N1": int,
        "reason": str,
        "sampled_indices": [int, ...],
        "mosaics": [{"type":"mosaic","path":str,"frame_ids":[str,...]}, ...],
        "hm3_context": {...}
      }
    """
    hm3_context = _hm3_compact_for_scoping(memory)
    decision = _interval_decision(model, query, hm3_context)
    requested_interval = decision.get("interval") or {}
    start_idx = int(requested_interval.get("start", 0))
    end_idx = int(requested_interval.get("end", start_idx))
    frames_meta = (memory.metadata.frames if memory.metadata else [])
    if not frames_meta:
        return {
            "tool": "interval_localizer",
            "interval": requested_interval,
            "N1": decision.get("N1"),
            "reason": decision.get("reason", ""),
            "error": "no frames metadata available",
        }
    visible_start, visible_end = _resolve_visible_bounds(memory, len(frames_meta))
    start_idx = max(visible_start, min(start_idx, visible_end))
    end_idx = max(start_idx, min(end_idx, visible_end))
    interval = {"start": start_idx, "end": end_idx}
    available = end_idx - start_idx + 1
    if available <= 0:
        return {
            "tool": "interval_localizer",
            "interval": interval,
            "N1": decision.get("N1"),
            "reason": decision.get("reason", ""),
            "error": "empty interval",
        }
    interval_min, interval_max, interval_default = _interval_limits()
    N1_requested = _safe_int(decision.get("N1"), interval_default)
    N1 = _clamp_sample_count(
        N1_requested,
        min_n=interval_min,
        max_n=interval_max,
        available=available,
    )
    indices = _uniform_indices(start_idx, end_idx, N1)
    frames: List[Image.Image] = []
    frame_ids: List[str] = []
    for idx in indices:
        path = frames_meta[idx].path
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = _resize_short_edge(img, 256)
            img = _draw_index_overlay(img, frames_meta[idx].frame_id)
            frames.append(img)
            frame_ids.append(frames_meta[idx].frame_id)
    mosaics = _make_mosaics(frames, rows=2, cols=3)
    _mosaic_base = Path(os.getenv("SURGENT_MOSAIC_DIR", "tmp/surgent_mosaics"))
    _vid = getattr(getattr(memory, "metadata", None), "video_id", None) or "unknown"
    output_dir = _mosaic_base / _vid
    output_dir.mkdir(parents=True, exist_ok=True)
    mosaic_entries: List[Dict[str, Any]] = []
    for i, mosaic in enumerate(mosaics):
        fname = f"mosaic_{i:03d}_start{indices[0]}_end{indices[-1]}.png"
        path = output_dir / fname
        mosaic.save(path, format="PNG")
        mosaic_entries.append({
            "type": "mosaic",
            "path": str(path),
            "frame_ids": list(frame_ids),
        })
    memory.sensory.long_term_pool = mosaic_entries
    output = {
        "tool": "interval_localizer",
        "interval": interval,
        "N1": N1,
        "reason": decision.get("reason", ""),
        "sampled_indices": indices,
        "mosaics": mosaic_entries,
        "hm3_context": hm3_context,
    }
    return output


@tool
def clip_explorer(
    query: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    resize_short_edge: int = 512,
) -> Dict[str, Any]:
    """
    Clip Explorer. The Clip Explorer serves a similar role to the Interval Localizer,
    but does not alter the agent's focused segments (long-term perception pool).
    Instead, it performs tentative, fine-grained probing within local frame intervals
    T_local around the focused segments. Given the short duration of the frame intervals
    it focuses on, we directly store the frames corresponding to these intervals in the
    short-term perception pool within Sensory Memory, without using a tiling method,
    and employ a fixed sampling frame number N2. Overall, this tool maintains the
    agent's global focus while enabling rapid, low-overhead hypothesis testing and
    evidence gathering at the micro-event level.

    Input:
      query: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      resize_short_edge: int

    Output (dict):
      {
        "tool": "clip_explorer",
        "interval": {"start": int, "end": int},
        "N2": int,
        "reason": str,
        "sampled_indices": [int, ...],
        "sampled": [{"type":"frame","frame_id":str,"path":str}, ...],
        "hm3_context": {...}
      }
    """
    hm3_context = _hm3_compact_for_scoping(memory)
    decision = _clip_decision(model, query, hm3_context)
    requested_interval = decision.get("interval") or {}
    start_idx = int(requested_interval.get("start", 0))
    end_idx = int(requested_interval.get("end", start_idx))
    frames_meta = (memory.metadata.frames if memory.metadata else [])
    if not frames_meta:
        return {
            "tool": "clip_explorer",
            "interval": requested_interval,
            "N2": decision.get("N2"),
            "reason": decision.get("reason", ""),
            "error": "no frames metadata available",
        }
    visible_start, visible_end = _resolve_visible_bounds(memory, len(frames_meta))
    start_idx = max(visible_start, min(start_idx, visible_end))
    end_idx = max(start_idx, min(end_idx, visible_end))
    interval = {"start": start_idx, "end": end_idx}
    clip_min, clip_max, clip_default = _clip_limits()
    legacy_n2_override = os.getenv("SURGENT_N2")
    if legacy_n2_override is not None:
        N2_requested = _safe_int(legacy_n2_override, clip_default)
    else:
        N2_requested = _safe_int(decision.get("N2"), clip_default)
    N2_fixed = _clamp_sample_count(
        N2_requested,
        min_n=clip_min,
        max_n=clip_max,
        available=(end_idx - start_idx + 1),
    )
    indices = _uniform_indices(start_idx, end_idx, N2_fixed)
    sampled: List[Dict[str, Any]] = []
    for idx in indices:
        path = frames_meta[idx].path
        sampled.append({
            "type": "frame",
            "frame_id": frames_meta[idx].frame_id,
            "path": path,
        })
    memory.sensory.short_term_pool = sampled
    output = {
        "tool": "clip_explorer",
        "interval": interval,
        "N2": N2_fixed,
        "reason": decision.get("reason", ""),
        "sampled_indices": indices,
        "sampled": sampled,
        "hm3_context": hm3_context,
    }
    return output


def _resize_short_edge(img: Image.Image, short_edge: int) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return img
    if w <= h:
        new_w = short_edge
        new_h = int(round(h * (short_edge / w)))
    else:
        new_h = short_edge
        new_w = int(round(w * (short_edge / h)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _draw_index_overlay(img: Image.Image, frame_id: str) -> Image.Image:
    rgba = img.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    # Use a much larger index label for readability in mosaics.
    base = max(24, int(min(rgba.size) * 0.12))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", base)
    except Exception:
        font = ImageFont.load_default()
    text = str(frame_id)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    padding = max(6, base // 5)
    rect_w = text_bbox[2] - text_bbox[0] + padding * 2
    rect_h = text_bbox[3] - text_bbox[1] + padding * 2
    draw.rectangle((0, 0, rect_w, rect_h), fill=(0, 0, 0, 160))
    draw.text((padding, padding), text, fill=(255, 255, 255, 255), font=font)
    return rgba.convert("RGB")


def _pad_to_size(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    w, h = img.size
    x = max((target_w - w) // 2, 0)
    y = max((target_h - h) // 2, 0)
    canvas.paste(img, (x, y))
    return canvas


def _make_mosaics(frames: List[Image.Image], rows: int, cols: int) -> List[Image.Image]:
    per = rows * cols
    if not frames:
        return []
    mosaics: List[Image.Image] = []
    for i in range(0, len(frames), per):
        batch = frames[i:i + per]
        if len(batch) < per:
            batch = batch + [Image.new("RGB", (1, 1), color=(0, 0, 0))] * (per - len(batch))
        max_w = max(im.size[0] for im in batch)
        max_h = max(im.size[1] for im in batch)
        tile_size = (max_w, max_h)
        mosaic = Image.new("RGB", (cols * max_w, rows * max_h), color=(0, 0, 0))
        for j, im in enumerate(batch):
            row = j // cols
            col = j % cols
            tile = _pad_to_size(im, tile_size)
            mosaic.paste(tile, (col * max_w, row * max_h))
        mosaics.append(mosaic)
    return mosaics


def _uniform_indices(start_idx: int, end_idx: int, n: int) -> List[int]:
    if n <= 0:
        return []
    total = end_idx - start_idx + 1
    if total <= 0:
        return []
    n = min(n, total)
    if n == 1:
        return [start_idx]
    step = (total - 1) / (n - 1)
    indices = []
    for i in range(n):
        idx = int(round(start_idx + i * step))
        idx = max(start_idx, min(end_idx, idx))
        if indices and idx <= indices[-1]:
            idx = min(end_idx, indices[-1] + 1)
        indices.append(idx)
    return indices


def _interval_decision(model: Any, query: str, hm3_context: Dict[str, Any]) -> Dict[str, Any]:
    interval_min, interval_max, interval_default = _interval_limits()
    if model is None:
        return {"interval": {"start": 0, "end": 0}, "N1": interval_default, "reason": "default"}
    visible_start = hm3_context.get("metadata", {}).get("visible_start_idx", 0)
    visible_end = hm3_context.get("metadata", {}).get("visible_end_idx", 0)
    system_prompt = (
        "You are Interval Localizer. Return JSON with keys: "
        "{interval: {start:int, end:int}, N1:int, reason:str}. "
        f"N1 must be between {interval_min} and {interval_max}. "
        f"IMPORTANT: You can ONLY access frames in the visible window [{visible_start}, {visible_end}]. "
        f"Your interval start and end MUST be within [{visible_start}, {visible_end}]. "
        "Frames outside this range do not exist yet."
    )
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{query}\n\nHM3 context:\n{json.dumps(hm3_context)}"),
    ])
    return _safe_json(response)


def _clip_decision(model: Any, query: str, hm3_context: Dict[str, Any]) -> Dict[str, Any]:
    clip_min, clip_max, clip_default = _clip_limits()
    if model is None:
        return {"interval": {"start": 0, "end": 0}, "N2": clip_default, "reason": "default"}
    visible_start = hm3_context.get("metadata", {}).get("visible_start_idx", 0)
    visible_end = hm3_context.get("metadata", {}).get("visible_end_idx", 0)
    system_prompt = (
        "You are Clip Explorer. Return JSON with keys: "
        "{interval: {start:int, end:int}, N2:int, reason:str}. "
        f"N2 must be between {clip_min} and {clip_max}. "
        f"IMPORTANT: You can ONLY access frames in the visible window [{visible_start}, {visible_end}]. "
        f"Your interval start and end MUST be within [{visible_start}, {visible_end}]. "
        "Frames outside this range do not exist yet."
    )
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{query}\n\nHM3 context:\n{json.dumps(hm3_context)}"),
    ])
    return _safe_json(response)


def _safe_json(response: Any) -> Dict[str, Any]:
    text = getattr(response, "content", None)
    if not isinstance(text, str):
        text = str(response)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {}


def _hm3_compact_for_scoping(memory: HM3Memory) -> Dict[str, Any]:
    frame_id_by_idx: Dict[int, str] = {}
    if memory.metadata:
        for fr in memory.metadata.frames:
            frame_id_by_idx[int(fr.idx)] = str(fr.frame_id)

    def _interval_to_frame_ids(interval: Any) -> Dict[str, Any]:
        if not isinstance(interval, dict):
            return {}
        start = interval.get("start")
        end = interval.get("end")
        out: Dict[str, Any] = {}
        if isinstance(start, int):
            out["start_frame_id"] = frame_id_by_idx.get(start, str(start))
        if isinstance(end, int):
            out["end_frame_id"] = frame_id_by_idx.get(end, str(end))
        return out

    def _sanitize_pool(items: List[Any]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                keep = {}
                if "type" in item:
                    keep["type"] = item["type"]
                if "path" in item:
                    keep["path"] = item["path"]
                if "frame_id" in item:
                    keep["frame_id"] = item["frame_id"]
                if "frame_ids" in item:
                    keep["frame_ids"] = item["frame_ids"]
                sanitized.append(keep)
        return sanitized

    metadata: Dict[str, Any] = {}
    if memory.metadata:
        visible_start, visible_end = _resolve_visible_bounds(memory, len(memory.metadata.frames))
        metadata = {
            "video_id": memory.metadata.video_id,
            "image_dir": memory.metadata.image_dir,
            "num_frames": memory.metadata.num_frames,
            "visible_start_idx": visible_start,
            "visible_end_idx": visible_end,
            "num_visible_frames": max(0, visible_end - visible_start + 1),
        }
    results_tail: List[Dict[str, Any]] = []
    for entry in memory.results.events[-3:]:
        results_tail.append(
            {
                "iteration": entry.iteration,
                "tool": entry.tool,
                "interval": _interval_to_frame_ids(entry.interval),
            }
        )
    return {
        "metadata": metadata,
        "long_term_pool": _sanitize_pool(memory.sensory.long_term_pool[-3:]),
        "short_term_pool": _sanitize_pool(memory.sensory.short_term_pool[-3:]),
        "results_tail": results_tail,
        "working_tail": [entry.model_dump() for entry in memory.working.entries[-3:]],
    }


def tool_registry() -> Dict[str, Any]:
    return {
        "interval_localizer": interval_localizer,
        "clip_explorer": clip_explorer,
    }
