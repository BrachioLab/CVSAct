from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResultMemoryEntry(BaseModel):
    iteration: int = Field(ge=0)
    interval: Dict[str, Any]
    tool: str
    output: Dict[str, Any]


class WorkingMemoryEntry(BaseModel):
    iteration: int = Field(ge=0)
    objective: str
    reasoning_trace: str
    tool: Optional[str] = None


@dataclass
class SensoryMemory:
    """Perceptual buffers from the paper: long-term P_l and short-term P_s.

    Sensory Memory consists of a long-term perception pool P_l and a short-term perception
    pool P_s for storing perceptual information such as key frames F. P_l maintains a
    snapshot of the base frame interval currently processed by the pipeline and acts as a
    lightweight, volatile buffer for the agent's most recent temporal focus. In contrast,
    P_s temporarily holds fine-grained clips sampled during local exploration and analysis,
    serving as a transient workspace for rapid hypothesis testing and evidence accumulation
    at the micro-event level. Once analysis results are written into Result Memory, P_s is
    cleared to free resources for the next reasoning cycle.
    """
    long_term_pool: List[Dict[str, Any]] = field(default_factory=list)  # P_l in paper
    short_term_pool: List[Dict[str, Any]] = field(default_factory=list)  # P_s in paper


@dataclass
class ResultMemory:
    """Dynamic workspace storing iteration index, analyzed intervals, and tool outputs.

    Result Memory is the dynamic workspace that records intermediate results generated
    throughout the agent's operation. It stores the iteration index, analyzed frame
    intervals, and corresponding tool outputs. By maintaining temporally ordered entries,
    Result Memory captures a fine-grained history of perception cues, allowing the
    controller to reflect on past reasoning steps, avoid redundant actions, and adapt
    strategy based on accumulated evidence.
    """
    events: List[ResultMemoryEntry] = field(default_factory=list)

    def add(
        self,
        tool: str,
        interval: Dict[str, Any],
        output: Dict[str, Any],
        iteration: int,
    ) -> None:
        entry = ResultMemoryEntry(
            iteration=iteration,
            interval=interval,
            tool=tool,
            output=output,
        )
        self.events.append(entry)


@dataclass
class WorkingMemory:
    """Controller reasoning traces and intended objectives prior to tool calls.

    Working Memory records the controller's reasoning traces and intended objectives prior
    to each tool invocation. By externalizing these traces from the MLLM context into
    Working Memory, tool-related context can be cleared after every call. This reduces
    redundancy, mitigates context overflow, and sharpens the focus of subsequent reasoning.
    Persisted traces provide an auditable history supporting error analysis and strategy
    adaptation.
    """
    entries: List[WorkingMemoryEntry] = field(default_factory=list)

    def add(
        self,
        iteration: int,
        objective: str,
        reasoning_trace: str,
        tool: Optional[str] = None,
    ) -> None:
        entry = WorkingMemoryEntry(
            iteration=iteration,
            objective=objective,
            reasoning_trace=reasoning_trace,
            tool=tool,
        )
        self.entries.append(entry)


@dataclass
class HM3Memory:
    sensory: SensoryMemory = field(default_factory=SensoryMemory)
    results: ResultMemory = field(default_factory=ResultMemory)
    working: WorkingMemory = field(default_factory=WorkingMemory)
    metadata: Optional[VideoMetadata] = None
    visible_start_idx: int = 0
    visible_end_idx: Optional[int] = None
    iteration_cursor: int = 0
    video_criteria_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record_result(
        self,
        tool: str,
        interval: Dict[str, Any],
        output: Dict[str, Any],
        iteration: int,
    ) -> None:
        self.results.add(tool=tool, interval=interval, output=output, iteration=iteration)

    def record_working(
        self,
        iteration: int,
        objective: str,
        reasoning_trace: str,
        tool: Optional[str] = None,
    ) -> None:
        self.working.add(
            iteration=iteration,
            objective=objective,
            reasoning_trace=reasoning_trace,
            tool=tool,
        )

    def clear_short_term(self) -> None:
        self.sensory.short_term_pool.clear()

    def update_video_criteria_status(self, frame_prediction: Dict[str, Dict[str, Any]]) -> None:
        """Max-accumulate frame-level criteria scores into video-level status.

        For each criterion, keeps the higher score seen so far (any frame satisfied → video satisfied).
        Updates reason only when a higher score replaces the previous.
        """
        for crit, payload in frame_prediction.items():
            crit = crit.strip().upper()
            if not isinstance(payload, dict):
                continue
            try:
                new_score = max(0.0, min(1.0, float(payload.get("score", 0.0) or 0.0)))
            except Exception:
                new_score = 0.0
            new_reason = str(payload.get("reason") or "")
            prev = self.video_criteria_status.get(crit, {})
            prev_score = float(prev.get("score", 0.0))
            if new_score >= prev_score:
                self.video_criteria_status[crit] = {"score": new_score, "reason": new_reason}

    def set_visible_window(self, start_idx: int = 0, end_idx: Optional[int] = None) -> None:
        start = max(0, int(start_idx))
        if end_idx is None:
            self.visible_start_idx = start
            self.visible_end_idx = None
            return
        end = max(0, int(end_idx))
        if end < start:
            end = start
        self.visible_start_idx = start
        self.visible_end_idx = end

    def record_result_and_clear_short_term(
        self,
        tool: str,
        interval: Dict[str, Any],
        output: Dict[str, Any],
        iteration: int,
    ) -> None:
        self.record_result(tool=tool, interval=interval, output=output, iteration=iteration)
        self.clear_short_term()

    def snapshot(self, tail: int = 3, max_output_chars: int = 300) -> Dict[str, Any]:
        frame_id_by_idx: Dict[int, str] = {}
        if self.metadata:
            for fr in self.metadata.frames:
                frame_id_by_idx[int(fr.idx)] = str(fr.frame_id)

        def _interval_to_frame_ids(interval: Dict[str, Any]) -> Dict[str, Any]:
            start = interval.get("start")
            end = interval.get("end")
            out: Dict[str, Any] = {}
            if isinstance(start, int):
                out["start_frame_id"] = frame_id_by_idx.get(start, str(start))
            if isinstance(end, int):
                out["end_frame_id"] = frame_id_by_idx.get(end, str(end))
            return out

        def _indices_to_frame_ids(indices: Any) -> List[str]:
            if not isinstance(indices, list):
                return []
            out: List[str] = []
            for value in indices:
                if isinstance(value, int):
                    out.append(frame_id_by_idx.get(value, str(value)))
            return out

        def _truncate(value: Any) -> str:
            raw = str(value)
            if len(raw) <= max_output_chars:
                return raw
            return f"{raw[:max_output_chars]}... (truncated)"

        def _serialize_pool(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            serialized: List[Dict[str, Any]] = []
            for item in items[-tail:]:
                if not isinstance(item, dict):
                    continue
                keep: Dict[str, Any] = {}
                for key in ("type", "path", "frame_id", "frame_ids", "idx"):
                    if key in item:
                        keep[key] = item[key]
                serialized.append(keep)
            return serialized

        # Keys whose dict/list values should be preserved without truncation
        _PRESERVE_KEYS = {"rationale", "cvs_predictions"}

        def _serialize_result_output(output: Dict[str, Any]) -> Dict[str, Any]:
            keep: Dict[str, Any] = {}
            for key, value in output.items():
                if key in {"frames", "mosaics", "sampled", "hm3_context"}:
                    continue
                if key == "sampled_indices":
                    keep["sampled_frame_ids"] = _indices_to_frame_ids(value)
                    continue
                if key == "interval" and isinstance(value, dict):
                    keep[key] = _interval_to_frame_ids(value)
                    continue
                if key in _PRESERVE_KEYS:
                    keep[key] = value
                    continue
                if isinstance(value, (dict, list)):
                    keep[key] = _truncate(value)
                else:
                    keep[key] = value
            return keep

        metadata: Dict[str, Any] = {}
        if self.metadata:
            total = int(self.metadata.num_frames)
            v_start = max(0, int(self.visible_start_idx or 0))
            raw_end = self.visible_end_idx
            v_end = (total - 1) if raw_end is None else min(int(raw_end), total - 1)
            v_end = max(0, v_end)
            v_start = min(v_start, v_end)
            metadata = {
                "video_id": self.metadata.video_id,
                "image_dir": self.metadata.image_dir,
                "num_frames": total,
                "visible_start_idx": v_start,
                "visible_end_idx": v_end,
                "num_visible_frames": max(0, v_end - v_start + 1),
            }

        results_tail: List[Dict[str, Any]] = []
        for entry in self.results.events[-tail:]:
            results_tail.append(
                {
                    "iteration": entry.iteration,
                    "tool": entry.tool,
                    "interval": _interval_to_frame_ids(entry.interval),
                    "output": _serialize_result_output(entry.output),
                }
            )

        return {
            "sensory": {
                "long_term_pool": _serialize_pool(self.sensory.long_term_pool),
                "short_term_pool": _serialize_pool(self.sensory.short_term_pool),
            },
            "results_tail": results_tail,
            "working_tail": [entry.model_dump() for entry in self.working.entries[-tail:]],
            "metadata": metadata,
            "video_criteria_status": dict(self.video_criteria_status),
        }

    def set_metadata(
        self,
        video_id: str,
        image_dir: str,
        frames: List[Dict[str, Any]],
        num_frames: int,
    ) -> None:
        self.metadata = VideoMetadata(
            video_id=str(video_id),
            image_dir=str(image_dir),
            frames=[FrameRef(**frame) for frame in frames],
            num_frames=int(num_frames),
        )
class FrameRef(BaseModel):
    idx: int = Field(ge=0)
    frame_id: str
    path: str


class VideoMetadata(BaseModel):
    video_id: str
    image_dir: str
    frames: List[FrameRef]
    num_frames: int = Field(ge=0)
