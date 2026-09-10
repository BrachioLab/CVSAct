from __future__ import annotations

import inspect
import json
import os
import operator
import re
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from typing_extensions import Annotated

try:  # pragma: no cover - optional dependency for template
    from langgraph.graph import END, START, StateGraph
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "LangGraph is required for src/surgent/graph.py. "
        "Install it (e.g., pip install langgraph)."
    ) from exc

from .audit_simple_actions import AUDIT_V11_SIMPLE_TARGET_COUNT, normalize_actor_slot_actions
from .controller import build_controller_prompt, record_result_from_tool
from .memory_hm3 import FrameRef, HM3Memory, VideoMetadata
from .prompts import CONTROLLER_SYSTEM_PROMPT
from .schemas import CRITERION_DEFINITIONS, CVSRequest
from .tools_spatial import tool_registry as spatial_registry
from .tools_temporal import tool_registry as temporal_registry
from .tools_understanding import tool_registry as understanding_registry


_LOG_OMIT_KEYS = {"memory", "model", "hm3_context"}
_DECISION_FALLBACK: Dict[str, Any] = {
    "action": "ANSWER",
    "action_args": {},
    "reason": "parse_error",
    "decision": "uncertain",
    "confidence": 0.0,
    "criteria_status": {},
    "stop": True,
}
_MM_TO_TEMPORAL_TOOL: Dict[str, str] = {
    "scene_snapper": "interval_localizer",
    "clip_analyzer": "clip_explorer",
    "zoom": "clip_explorer",
}


INITIAL_ANALYSIS_TOOLS = {"scene_cvs_analyzer", "action_rec"}
CONTROLLER_EXPLORATION_TOOLS = {"scene_snapper", "clip_analyzer", "zoom"}
ALL_TOOL_NAMES = INITIAL_ANALYSIS_TOOLS | CONTROLLER_EXPLORATION_TOOLS


def _effective_max_steps(ctx: Optional["SurgentContext"] = None) -> int:
    """Max steps for the controller, reserving slots for finalize tools."""
    raw = int(os.getenv("SURGENT_MAX_STEPS", "10"))
    num_finalize = len(ctx.finalize_tools) if ctx and ctx.finalize_tools else 0
    return max(1, raw - num_finalize)


@dataclass
class SurgentContext:
    models: Dict[str, Any]
    allowed_tools: Optional[set] = None
    taxonomy_catalog: Optional[Dict[str, Any]] = None
    preferred_tools: Optional[List[str]] = None
    finalize_tools: Optional[List[str]] = None
    scene_preset: str = "direct"
    analyze_actions: bool = False
    recommend_actions: bool = False
    fixed_k: Optional[int] = None
    cvs_context_mode: str = "latest"  # latest | current | full | video_summary
    one_per_actor: bool = False
    action_rec_rules: str = "default"


class SurgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    llm_calls: int
    ctx: SurgentContext
    hm3: HM3Memory
    iteration: int
    start_iteration: int
    criterion: str
    decision: Optional[Dict[str, Any]]
    last_temporal: Optional[Dict[str, Any]]
    _finalize_phase: bool


def _tool_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    registry.update(temporal_registry())
    registry.update(_mm_tool_registry())
    return registry


def _mm_tool_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    registry.update(understanding_registry())
    registry.update(spatial_registry())
    return registry


def _filtered_mm_tools(ctx: Optional[SurgentContext] = None) -> Dict[str, Any]:
    registry = _mm_tool_registry()
    if ctx is not None and ctx.allowed_tools is not None:
        registry = {k: v for k, v in registry.items() if k in ctx.allowed_tools}
    return registry


def _controller_mm_tools(ctx: Optional[SurgentContext] = None) -> Dict[str, Any]:
    """Tools the controller may choose after preferred auto-calls finish."""
    registry = _filtered_mm_tools(ctx)
    return {k: v for k, v in registry.items() if k in CONTROLLER_EXPLORATION_TOOLS}


def _build_query(criterion: str) -> str:
    criteria = [
        token
        for token in [x.strip().upper() for x in re.split(r"[,\s]+", str(criterion)) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    if len(criteria) > 1:
        defs = "\n".join([f"- {c}: {CRITERION_DEFINITIONS[c]}" for c in criteria])
        return (
            f"Task: Jointly assess CVS criteria {', '.join(criteria)} for this surgical video.\n"
            f"Criteria definitions:\n{defs}\n"
            "Goal: Gather visual evidence and recommend the next action that maximizes "
            "progress toward satisfying all criteria together.\n"
            "Return final satisfied only when all criteria are satisfied."
        )
    criterion_def = CRITERION_DEFINITIONS.get(criterion, "")
    return (
        f"Task: Assess CVS {criterion} for this surgical video.\n"
        f"Criterion {criterion} definition: {criterion_def}\n"
        "Goal: Gather visual evidence from the video and decide one of "
        "{Satisfied, Unsatisfied, Uncertain}.\n"
        "You must cite the best supporting evidence intervals (and any contradicting intervals) "
        "and give a calibrated confidence score in [0,1]."
    )


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    # Fallback 1: extract JSON from fenced code blocks (```json ... ```).
    for match in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    # Fallback 2: extract the outermost object-like span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


_RECOMMENDED_ACTION_FIELDS = (
    "actor_role", "tool_type", "action_code", "target_structure",
    "target_context_1", "target_context_2", "intention", "evidence",
)

_EMPTY_ACTION: Dict[str, Any] = {
    "rank": 0,
    "actor_role": "(not set)",
    "tool_type": "(not set)",
    "action_code": "(not set)",
    "target_structure": "(not set)",
    "target_context_1": "(not set)",
    "target_context_2": "(not set)",
    "intention": "(not set)",
    "confidence": 0.0,
    "evidence": "uncertain",
}


def _parse_recommended_actions(raw: Any, target_count: int = AUDIT_V11_SIMPLE_TARGET_COUNT) -> List[Dict[str, Any]]:
    """Validate recommended_actions and normalize to the actor-slot schema."""
    actions: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            action: Dict[str, Any] = {}
            for field in _RECOMMENDED_ACTION_FIELDS:
                action[field] = str(entry.get(field) or "(not set)").strip() or "(not set)"
            try:
                action["confidence"] = max(0.0, min(1.0, float(entry.get("confidence", 0.0) or 0.0)))
            except Exception:
                action["confidence"] = 0.0
            action["rank"] = 0  # will be re-assigned below
            actions.append(action)
    normalized = normalize_actor_slot_actions(actions)
    return normalized[:target_count]


def _parse_controller_decision(response: Any, allowed_actions: set[str]) -> Dict[str, Any]:
    content = getattr(response, "content", "")
    parse_failed = False
    if isinstance(content, dict):
        raw = content
    else:
        raw = _extract_json_object(content if isinstance(content, str) else str(content))
        if not raw:
            parse_failed = True
    if not isinstance(raw, dict) or not raw:
        out = dict(_DECISION_FALLBACK)
        out["_parse_failed"] = True
        return out
    if "action" not in raw:
        out = dict(_DECISION_FALLBACK)
        out["_parse_failed"] = True
        return out
    action = str(raw.get("action", "ANSWER"))
    if action not in allowed_actions:
        out = dict(_DECISION_FALLBACK)
        out["_parse_failed"] = True
        return out
    action_args = raw.get("action_args")
    if not isinstance(action_args, dict):
        action_args = {}
    reason_value = raw.get("reason")
    if not reason_value and isinstance(action_args, dict):
        reason_value = action_args.get("reason")
    reason = str(reason_value or "").strip() or "controller_no_reason"
    stop = bool(raw.get("stop", action == "ANSWER"))
    # Extract frame-level criteria prediction (new key, fall back to old key)
    frame_pred = _parse_criteria_status(
        raw.get("current_frame_criteria_prediction") or raw.get("criteria_status")
    )
    # Derive decision and confidence from criteria scores (not from LLM)
    scores = [float(v.get("score", 0.0)) for v in frame_pred.values()] if frame_pred else []
    if scores and all(s >= 0.5 for s in scores):
        decision_label = "satisfied"
    elif scores:
        decision_label = "unsatisfied"
    else:
        decision_label = "uncertain"
    confidence = sum(scores) / len(scores) if scores else 0.0
    # Extract and validate recommended_actions
    recommended_actions = _parse_recommended_actions(raw.get("recommended_actions"))

    return {
        "action": action,
        "action_args": action_args,
        "reason": reason,
        "decision": decision_label,
        "confidence": confidence,
        "criteria_status": frame_pred,
        "recommended_actions": recommended_actions,
        "stop": stop,
        "_parse_failed": parse_failed,
    }


def _normalize_final_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "satisfied": "satisfied",
        "met": "satisfied",
        "unsatisfied": "unsatisfied",
        "not_met": "unsatisfied",
        "not met": "unsatisfied",
        "uncertain": "uncertain",
        "unknown": "uncertain",
    }
    return mapping.get(text, "uncertain")


def _score_to_label(score: float) -> str:
    """Derive a human-readable label from a satisfaction score."""
    if score >= 0.7:
        return "satisfied"
    if score <= 0.3:
        return "unsatisfied"
    return "uncertain"


def _parse_criteria_status(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    allowed = set(CRITERION_DEFINITIONS.keys())
    out: Dict[str, Dict[str, Any]] = {}
    for key, payload in value.items():
        crit = str(key or "").strip().upper()
        if crit not in allowed or not isinstance(payload, dict):
            continue
        # New format: single "score" field (float 0..1)
        if "score" in payload:
            try:
                score = max(0.0, min(1.0, float(payload.get("score", 0.0) or 0.0)))
            except Exception:
                score = 0.0
            out[crit] = {
                "score": score,
                "reason": str(payload.get("reason") or payload.get("evidence") or "").strip(),
            }
        else:
            # Legacy format: "decision" + "confidence" → convert to score
            decision = _normalize_final_decision(payload.get("decision"))
            try:
                confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
            except Exception:
                confidence = 0.0
            if decision == "satisfied":
                score = 0.5 + 0.5 * confidence
            elif decision == "unsatisfied":
                score = 0.5 - 0.5 * confidence
            else:
                score = 0.5
            out[crit] = {
                "score": max(0.0, min(1.0, score)),
                "reason": str(payload.get("reason") or payload.get("evidence") or "").strip(),
            }
    return out


def _estimate_explored_frame_coverage(memory: HM3Memory) -> tuple[int, int]:
    total_frames = int(memory.metadata.num_frames) if memory.metadata else 0
    if total_frames <= 0:
        return 0, 0
    visible_start, visible_end = _resolve_visible_bounds(memory, total_frames)
    visible_total = max(0, visible_end - visible_start + 1)
    if visible_total <= 0:
        return 0, 0
    covered = [False] * visible_total
    _EXPLORATION_TOOLS = {
        "interval_localizer", "clip_explorer",
        "scene_cvs_analyzer", "scene_cvs_act_analyzer",
        "scene_snapper", "clip_analyzer", "action_rec",
    }
    for entry in memory.results.events:
        if entry.tool not in _EXPLORATION_TOOLS:
            continue
        interval = entry.interval if isinstance(entry.interval, dict) else {}
        try:
            start = int(interval.get("start", 0))
            end = int(interval.get("end", start))
        except Exception:
            continue
        start = max(0, min(total_frames - 1, start))
        end = max(0, min(total_frames - 1, end))
        if end < start:
            start, end = end, start
        if end < visible_start or start > visible_end:
            continue
        start = max(start, visible_start)
        end = min(end, visible_end)
        for idx in range(start, end + 1):
            covered[idx - visible_start] = True
    return sum(1 for flag in covered if flag), visible_total


def _enforce_unsatisfied_coverage_guard(decision: Dict[str, Any], memory: HM3Memory) -> Dict[str, Any]:
    action = str(decision.get("action", ""))
    final_decision = _normalize_final_decision(decision.get("decision"))
    if action != "ANSWER" or final_decision != "unsatisfied":
        return decision
    covered_count, total_frames = _estimate_explored_frame_coverage(memory)
    if total_frames <= 0 or covered_count >= total_frames:
        return decision
    previous_reason = str(decision.get("reason") or "").strip()
    guarded = dict(decision)
    guarded["action"] = "scene_snapper"
    guarded["action_args"] = {}
    guarded["decision"] = "uncertain"
    guarded["stop"] = False
    guarded["confidence"] = 0.0
    guarded["coverage_guard_triggered"] = True
    guarded["coverage"] = {
        "covered_frames": covered_count,
        "total_frames": total_frames,
    }
    guarded_reason = (
        f"coverage_guard: explored {covered_count}/{total_frames} visible-window frames; "
        "collect more evidence before concluding unsatisfied."
    )
    guarded["reason"] = (
        f"{guarded_reason} Previous answer rationale: {previous_reason}"
        if previous_reason
        else guarded_reason
    )
    return guarded


def _resolve_visible_bounds(memory: HM3Memory, total_frames: int) -> tuple[int, int]:
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


def _frame_id_to_idx_map(memory: HM3Memory) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not memory.metadata:
        return out
    for fr in memory.metadata.frames:
        try:
            out[str(fr.frame_id)] = int(fr.idx)
        except Exception:
            continue
    return out


def _coerce_interval_index(value: Any, frame_id_to_idx: Dict[str, int]) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value or "").strip()
    if not text:
        return None
    if text in frame_id_to_idx:
        return int(frame_id_to_idx[text])
    try:
        return int(text)
    except Exception:
        return None


def _normalize_interval_arg(value: Any, *, memory: Optional[HM3Memory] = None) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict) and "items" in value and set(value.keys()).issubset({"len", "items"}):
        return _normalize_interval_arg(value.get("items"), memory=memory)
    if isinstance(value, dict):
        frame_id_to_idx = _frame_id_to_idx_map(memory) if memory is not None else {}
        start_raw = (
            value.get("start")
            if "start" in value
            else value.get("start_index", value.get("start_idx", value.get("start_frame_id")))
        )
        end_raw = (
            value.get("end")
            if "end" in value
            else value.get("end_index", value.get("end_idx", value.get("end_frame_id")))
        )
        start_idx = _coerce_interval_index(start_raw, frame_id_to_idx)
        end_idx = _coerce_interval_index(end_raw, frame_id_to_idx)
        if start_idx is None and end_idx is None:
            return None
        if start_idx is None:
            start_idx = end_idx
        if end_idx is None:
            end_idx = start_idx
        if start_idx is None or end_idx is None:
            return None
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        return {"start": int(start_idx), "end": int(end_idx)}
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return {"start": int(value[0]), "end": int(value[1])}
        except Exception:
            return None
    return None


def _clamp_interval_to_visible_window(memory: HM3Memory, interval: Dict[str, Any]) -> Dict[str, int]:
    total_frames = int(memory.metadata.num_frames) if memory.metadata else 0
    if total_frames <= 0:
        return {"start": 0, "end": 0}
    visible_start, visible_end = _resolve_visible_bounds(memory, total_frames)
    try:
        start = int(interval.get("start", visible_start))
    except Exception:
        start = visible_start
    try:
        end = int(interval.get("end", start))
    except Exception:
        end = start
    start = max(visible_start, min(start, visible_end))
    end = max(start, min(end, visible_end))
    return {"start": start, "end": end}


def _normalize_single_subquestion(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return "Is the target CVS evidence clearly visible in this scoped clip?"
    # Keep only one question at a time; if multiple are present, keep the first.
    if "?" in text:
        first = text.split("?", 1)[0].strip()
        return f"{first}?"
    # If multiple clauses are chained, keep only the first clause.
    for sep in ("; ", " and ", " then ", ". "):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    return text if text.endswith("?") else f"{text}?"


def _tool_signature(tool_fn: Any) -> str:
    sig = inspect.signature(tool_fn)
    params = []
    for name, param in sig.parameters.items():
        if param.default is inspect._empty:
            params.append(name)
        else:
            params.append(f"{name}={param.default!r}")
    return f"{tool_fn.__name__}({', '.join(params)})"


def get_tool_help() -> str:
    lines = ["Available tools:"]
    for tool_name, tool_fn in _tool_registry().items():
        doc = (tool_fn.__doc__ or "").strip()
        lines.append(f"- {tool_name}: {_tool_signature(tool_fn)}")
        if doc:
            for line in doc.splitlines():
                lines.append(f"  {line}")
    return "\n".join(lines)


def _truncate_text(text: Any, max_chars: int = 600) -> str:
    raw = str(text)
    if len(raw) <= max_chars:
        return raw
    return f"{raw[:max_chars]}... (truncated {len(raw) - max_chars} chars)"


def _summarize_sequence(values: List[Any], head: int = 2, tail: int = 2) -> Dict[str, Any]:
    if len(values) <= head + tail:
        return {"len": len(values), "items": values}
    return {
        "len": len(values),
        "head": values[:head],
        "tail": values[-tail:],
    }


def _compact_for_log(value: Any, key: str = "") -> Any:
    if key in _LOG_OMIT_KEYS:
        return "<omitted>"
    if key == "prompt" and isinstance(value, str):
        return value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate_text(value) if isinstance(value, str) else value
    if isinstance(value, list):
        compact_items = [_compact_for_log(v) for v in value]
        return _summarize_sequence(compact_items)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if k in _LOG_OMIT_KEYS:
                continue
            if k == "metadata" and isinstance(v, dict):
                out[k] = {
                    "video_id": v.get("video_id"),
                    "image_dir": v.get("image_dir"),
                    "num_frames": v.get("num_frames"),
                    "frames_len": len(v.get("frames", []) or []),
                }
                continue
            out[k] = _compact_for_log(v, key=k)
        return out
    return _truncate_text(value)


def _resolve_log_level(env_name: str, default_level: int) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        return default_level
    try:
        return int(raw)
    except Exception:
        return default_level


def _render_step_log(level: int, label: str, payload: Dict[str, Any]) -> Optional[str]:
    if level <= 0:
        return None
    if level == 1:
        return _format_level1_log(label, payload)
    if "Parse Error Raw Response" in label or "Parse Error Retry Raw Response" in label:
        return f"{label} {json.dumps(payload, default=str)}"
    return f"{label} {json.dumps(_compact_for_log(payload), default=str)}"


def _log_step(label: str, payload: Dict[str, Any]) -> None:
    trace_file = os.getenv("SURGENT_TRACE_FILE")

    def _append_trace(text: str) -> None:
        if not trace_file:
            return
        try:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n\n")
        except Exception:
            pass

    level_raw = os.getenv("SURGENT_LOG_LEVEL")
    if level_raw is None:
        default_level = 2 if os.getenv("SURGENT_LOG_STEPS") == "1" else 0
    else:
        try:
            default_level = int(level_raw)
        except Exception:
            default_level = 0
    console_level = _resolve_log_level("SURGENT_CONSOLE_LOG_LEVEL", default_level)
    trace_level = _resolve_log_level("SURGENT_TRACE_LOG_LEVEL", default_level)

    min_level = 2 if "prompt" in label else 1
    if console_level >= min_level:
        rendered_console = _render_step_log(console_level, label, payload)
        if rendered_console:
            print(rendered_console, file=sys.stderr, flush=True)
            print("", file=sys.stderr, flush=True)
    if trace_level >= min_level:
        rendered_trace = _render_step_log(trace_level, label, payload)
        if rendered_trace:
            _append_trace(rendered_trace)


def _format_level1_log(label: str, payload: Dict[str, Any]) -> str:
    def _fmt_key_value(key: str, value: Any) -> str:
        text = "" if value is None else str(value)
        if "\n" not in text:
            return f"{key}: {text}"
        first, *rest = text.splitlines()
        if not rest:
            return f"{key}: {first}"
        return f"{key}: {first}\n" + "\n".join(f"  {r}" for r in rest)

    def _caps(value: Any) -> str:
        return str(value or "").upper()

    decision = payload.get("decision") if isinstance(payload, dict) else None
    is_controller_decision = (
        ("llm_call response" in label or "forced_answer_call response" in label)
        and isinstance(decision, dict)
    )
    if is_controller_decision:
        header = "[AGENT-STEP] Forced Answer" if "forced_answer" in label else "[AGENT-STEP] Controller Decision"
        lines = [
            header,
            _fmt_key_value("Action", _caps(decision.get("action"))),
            _fmt_key_value("Reason", decision.get("reason")),
            _fmt_key_value("Stop", decision.get("stop")),
        ]
        criteria_status = decision.get("criteria_status")
        if isinstance(criteria_status, dict) and criteria_status:
            status_parts = []
            for c, info in criteria_status.items():
                if isinstance(info, dict):
                    score = info.get("score")
                    if score is not None:
                        label = _score_to_label(float(score))
                        status_parts.append(f"{c}={float(score):.2f} ({label})")
                    else:
                        # Legacy fallback
                        status_parts.append(
                            f"{c}={info.get('decision', '?')} ({info.get('confidence', '?')})"
                        )
                else:
                    status_parts.append(f"{c}={info}")
            lines.append(_fmt_key_value("Criteria Status", ", ".join(status_parts)))
        rec_actions = decision.get("recommended_actions")
        if isinstance(rec_actions, list) and rec_actions:
            action_lines = []
            for a in rec_actions:
                if isinstance(a, dict) and a.get("action_code", "(not set)") != "(not set)":
                    action_lines.append(
                        f"[rank {a.get('rank', '?')}] "
                        f"{a.get('actor_role', '(not set)')}/{a.get('tool_type', '(not set)')} "
                        f"{a.get('action_code', '(not set)')} -> {a.get('target_structure', '(not set)')} "
                        f"| intention: {a.get('intention', '(not set)')}"
                    )
            if action_lines:
                lines.append(_fmt_key_value("Recommended Actions", "; ".join(action_lines)))
        return "\n".join(lines)

    if "temporal_call" in label:
        return (
            "[AGENT-STEP] Temporal Call\n"
            f"{_fmt_key_value('Tool', _caps(payload.get('tool')))}\n"
            f"{_fmt_key_value('Iteration', ((payload.get('args') or {}).get('iteration')))}"
        )

    if "temporal_output" in label:
        output = payload.get("output") if isinstance(payload, dict) else {}
        interval = output.get("interval") if isinstance(output, dict) else {}
        return (
            "[AGENT-STEP] Temporal Observation\n"
            f"{_fmt_key_value('Tool', _caps(payload.get('tool')))}\n"
            f"{_fmt_key_value('Interval', interval)}\n"
            f"{_fmt_key_value('Sampled Count', len((output.get('sampled_indices') or [])) if isinstance(output, dict) else 0)}\n"
            f"{_fmt_key_value('Reason', output.get('reason') if isinstance(output, dict) else '')}"
        )

    if "mm_call" in label:
        args = payload.get("args") if isinstance(payload, dict) else {}
        return (
            "[AGENT-STEP] Understanding Call\n"
            f"{_fmt_key_value('Tool', _caps(payload.get('tool')))}\n"
            f"{_fmt_key_value('Interval', (args or {}).get('interval'))}\n"
            f"{_fmt_key_value('Subquestion', (args or {}).get('subquestion', ''))}\n"
            f"{_fmt_key_value('Reason', (args or {}).get('reason', ''))}"
        )

    if "mm_output" in label:
        output = payload.get("output") if isinstance(payload, dict) else {}
        tool_name = str(payload.get("tool") or "").lower() if isinstance(payload, dict) else ""
        if tool_name == "zoom":
            return (
                "[AGENT-STEP] Understanding Observation\n"
                f"{_fmt_key_value('Tool', _caps(payload.get('tool')))}\n"
                f"{_fmt_key_value('Zoomed Frames', (output or {}).get('num_zoomed_frames', 0))}\n"
                f"{_fmt_key_value('Bbox Mode', (output or {}).get('bbox_mode', ''))}\n"
                f"{_fmt_key_value('Bbox', (output or {}).get('bbox_norm_xyxy', ''))}\n"
                f"{_fmt_key_value('Reason', (output or {}).get('reason', ''))}\n"
                f"{_fmt_key_value('Error', (output or {}).get('error', ''))}"
            )
        if tool_name in ("scene_cvs_analyzer", "scene_cvs_act_analyzer"):
            preds = (output or {}).get("cvs_predictions", {})
            pred_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in preds.items())
            actions = (output or {}).get("actions_ranked", [])
            if isinstance(actions, list) and actions:
                action_lines = []
                for a in actions:
                    if isinstance(a, dict):
                        action_lines.append(
                            f"[rank {a.get('rank', '?')}] "
                            f"{a.get('actor_role', '(not set)')}/{a.get('tool_type', '(not set)')} "
                            f"{a.get('action_code', '(not set)')} -> {a.get('target_structure', '(not set)')} "
                            f"ctx=({a.get('target_context_1', '(not set)')}, {a.get('target_context_2', '(not set)')}) "
                            f"| intention: {a.get('intention', '(not set)')}"
                        )
                action_str = "; ".join(action_lines) if action_lines else "(no action)"
            else:
                # Fallback to recommended_action
                action = (output or {}).get("recommended_action", {})
                if isinstance(action, dict) and action:
                    action_str = (
                        f"{action.get('actor_role', '(not set)')}/{action.get('tool_type', '(not set)')} "
                        f"{action.get('action_code', '(not set)')} -> {action.get('target_structure', '(not set)')} "
                        f"ctx=({action.get('target_context', '(not set)')}) "
                        f"| intention: {action.get('intention', '(not set)')}"
                    )
                else:
                    action_str = "(no action)"
            # Include rationale if present (cot preset)
            rationale = (output or {}).get("rationale", {})
            rationale_str = ""
            if isinstance(rationale, dict) and rationale:
                rationale_parts = []
                for c, r in rationale.items():
                    if r:
                        rationale_parts.append(f"{c.upper()}: {r}")
                if rationale_parts:
                    rationale_str = "; ".join(rationale_parts)
            lines = [
                "[AGENT-STEP] Understanding Observation",
                _fmt_key_value('Tool', _caps(payload.get('tool'))),
                _fmt_key_value('CVS Predictions', pred_str),
            ]
            if rationale_str:
                lines.append(_fmt_key_value('Rationale', rationale_str))
            lines.append(_fmt_key_value('Actions', action_str))
            lines.append(_fmt_key_value('Summary', (output or {}).get('summary', '')))
            return "\n".join(lines)
        if tool_name == "action_rec":
            actions = (output or {}).get("actions_ranked", [])
            action_lines = []
            if isinstance(actions, list) and actions:
                for a in actions:
                    if isinstance(a, dict):
                        action_lines.append(
                            f"[rank {a.get('rank', '?')}] "
                            f"{a.get('actor_role', '(not set)')}/{a.get('tool_type', '(not set)')} "
                            f"{a.get('action_code', '(not set)')} -> {a.get('target_structure', '(not set)')} "
                            f"ctx=({a.get('target_context_1', '(not set)')}, {a.get('target_context_2', '(not set)')}) "
                            f"| intention: {a.get('intention', '(not set)')}"
                        )
            action_str = "; ".join(action_lines) if action_lines else "(no action)"
            lines = [
                "[AGENT-STEP] Understanding Observation",
                _fmt_key_value('Tool', _caps(payload.get('tool'))),
                _fmt_key_value('Summary/Answer', (output or {}).get('summary') or (output or {}).get('answer')),
                _fmt_key_value('Actions', action_str),
                _fmt_key_value('Reasoning', (output or {}).get('reasoning', '')),
            ]
            return "\n".join(lines)
        return (
            "[AGENT-STEP] Understanding Observation\n"
            f"{_fmt_key_value('Tool', _caps(payload.get('tool')))}\n"
            f"{_fmt_key_value('Summary/Answer', (output or {}).get('summary') or (output or {}).get('answer'))}\n"
            f"{_fmt_key_value('Confidence', (output or {}).get('confidence', ''))}"
        )

    if "Parse Error Raw Response" in label:
        return (
            "[AGENT-STEP] Parse Error Raw Response\n"
            f"{_fmt_key_value('Content', payload.get('content') if isinstance(payload, dict) else '')}"
        )

    if "Parse Error Retry Raw Response" in label:
        return (
            "[AGENT-STEP] Parse Error Retry Raw Response\n"
            f"{_fmt_key_value('Content', payload.get('content') if isinstance(payload, dict) else '')}"
        )

    return f"{label} {json.dumps(_compact_for_log(payload), default=str)}"


def _auto_first_step(state: SurgentState) -> Dict[str, Any]:
    """Auto-call preferred tools before the controller decides.

    Iterates through ctx.preferred_tools in order and injects a decision
    for the first tool that hasn't been called yet.  Once all preferred
    tools have results in memory, returns empty so the router falls
    through to llm_call.
    """
    ctx = state.get("ctx")
    preferred = ctx.preferred_tools if ctx is not None else None
    if not preferred:
        return {}

    hm3 = state["hm3"]
    start_iteration = state.get("start_iteration", 0)
    called_tools = {event.tool for event in hm3.results.events if event.iteration >= start_iteration}

    for tool_name in preferred:
        if tool_name in called_tools:
            continue
        # Found an uncalled preferred tool — inject decision (counts as a step)
        _log_step(
            "[agent-step] auto_first_step:",
            {"action": tool_name, "reason": f"preferred-tool: {tool_name} not yet called"},
        )
        return {
            "decision": {
                "action": tool_name,
                "action_args": {},
                "reason": f"preferred-tool: auto-calling {tool_name}",
                "decision": "uncertain",
                "confidence": 0.0,
                "criteria_status": {},
                "stop": False,
            },
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # All preferred tools already called
    return {}


def _route_auto_first_step(state: SurgentState) -> str:
    """Route after auto_first_step.

    If a preferred tool decision was injected, route to temporal_node
    (for tools that need interval_localizer/clip_explorer first) or
    directly to mm_node.  Otherwise fall through to llm_call.
    """
    decision = state.get("decision")
    if decision is None:
        return "llm_call"
    action = str(decision.get("action", ""))
    if not action or action == "ANSWER":
        return "llm_call"
    # Route through temporal prereq if the tool needs one
    if action in _MM_TO_TEMPORAL_TOOL:
        return "temporal_node"
    return "mm_node"


def _auto_finalize_step(state: SurgentState) -> Dict[str, Any]:
    """Auto-call finalize tools after the controller says ANSWER / forced_answer.

    Iterates through ctx.finalize_tools in order and injects a decision
    for the first tool that hasn't been called yet this round.  Once all
    finalize tools have results in memory, returns ``{"decision": None}``
    so the router falls through to END.
    """
    ctx = state.get("ctx")
    finalize = ctx.finalize_tools if ctx is not None else None
    if not finalize:
        return {"decision": None}

    hm3 = state["hm3"]
    start_iteration = state.get("start_iteration", 0)
    called_tools = {event.tool for event in hm3.results.events if event.iteration >= start_iteration}

    for tool_name in finalize:
        if tool_name in called_tools:
            continue
        # Found an uncalled finalize tool — inject decision
        _log_step(
            "[agent-step] auto_finalize_step:",
            {"action": tool_name, "reason": f"finalize-tool: {tool_name} not yet called"},
        )
        return {
            "decision": {
                "action": tool_name,
                "action_args": {},
                "reason": f"finalize-tool: auto-calling {tool_name}",
                "decision": "uncertain",
                "confidence": 0.0,
                "criteria_status": {},
                "stop": False,
            },
            "llm_calls": state.get("llm_calls", 0) + 1,
            "_finalize_phase": True,
        }

    # All finalize tools already called — clear decision so router proceeds to END
    return {"decision": None}


def _route_auto_finalize_step(state: SurgentState) -> str:
    """Route after auto_finalize_step.

    If a finalize tool decision was injected, route to temporal_node or
    mm_node.  If all finalize tools done, route to END.
    """
    decision = state.get("decision")
    if decision is None:
        return END
    action = str(decision.get("action", ""))
    if not action or action == "ANSWER":
        return END
    if action in _MM_TO_TEMPORAL_TOOL:
        return "temporal_node"
    return "mm_node"


def _already_called_tools(hm3: HM3Memory, start_iteration: int) -> set[str]:
    """Return the set of MM tools that have already produced results this run."""
    return {
        entry.tool
        for entry in hm3.results.events
        if entry.iteration >= start_iteration
    }


def _llm_call(state: SurgentState) -> Dict[str, Any]:
    mm_tools = _filtered_mm_tools(state.get("ctx"))
    controller_tools = _controller_mm_tools(state.get("ctx"))
    # Remove tools that have already been called (e.g. scene_cvs_analyzer after
    # auto-first-step).  The controller can still use zoom, clip_analyzer, etc.
    already_called = _already_called_tools(state["hm3"], state.get("start_iteration", 0))
    # Also exclude finalize tools — they run automatically at the end,
    # the controller should not call them directly.
    ctx = state.get("ctx")
    finalize_set = set(ctx.finalize_tools) if ctx and ctx.finalize_tools else set()
    exclude_from_schema = already_called | finalize_set
    mm_tools_for_schema = {k: v for k, v in controller_tools.items() if k not in exclude_from_schema}
    mm_actions = sorted(mm_tools_for_schema.keys())
    criteria_in_scope = [
        token
        for token in [x.strip().upper() for x in re.split(r"[,\s]+", str(state.get("criterion") or "")) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    is_multi_criteria = len(criteria_in_scope) > 1

    # Skip recommended_actions when action_rec is a finalize tool —
    # the controller only decides CVS, action_rec runs separately at the end.
    skip_rec_actions = "action_rec" in finalize_set

    recommended_actions_schema = ""
    rec_actions_instruction = ""
    if not skip_rec_actions:
        recommended_actions_schema = (
            '  "recommended_actions": [\n'
            f"    // Always populate with exactly {AUDIT_V11_SIMPLE_TARGET_COUNT} next-action slots.\n"
            "    // Stage 1: form initial hypotheses from scene_cvs_analyzer output.\n"
            "    // Stage 2: actively refine by identifying what blocks each unsatisfied criterion\n"
            "    //   and what action most effectively reduces that gap.\n"
            "    // Use exactly one actor_role each for camera, left_instrument, right_instrument, and other.\n"
            '    {"rank":1,"actor_role":"<str>","tool_type":"<str>","action_code":"<str>","target_structure":"<str>","target_context_1":"<str>","target_context_2":"<str>","intention":"<str>","confidence":<float 0..1>,"evidence":"<short phrase>"},\n'
            '    {"rank":2, ...},\n'
            '    {"rank":3, ...},\n'
            '    {"rank":4, ...}\n'
            "  ],\n"
        )
        rec_actions_instruction = (
            "Always populate recommended_actions with exactly four actor-slot next actions: camera, left_instrument, right_instrument, other. "
            "This is prospective next-step planning from the visible video prefix and current CVS state, not retrospective clip labeling. "
            "Use CAMERA_NO_CHANGE when camera should remain stable, CAMERA_UNCERTAIN when camera recommendation is actually uncertain, "
            "and reserve the other slot for actions such as ICG_SWITCH or no additional other action.\n"
        )

    if is_multi_criteria:
        decision_schema = (
            "\nReturn STRICT JSON only with keys exactly:\n"
            "{\n"
            f'  "action": "ANSWER" | {" | ".join([json.dumps(a) for a in mm_actions])},\n'
            '  "action_args": { ... },\n'
            '  "reason": "<For ANSWER: explain progress toward all criteria and cite strongest evidence. For non-ANSWER: short intent sentence focused on maximizing joint progress.>",\n'
            '  "current_frame_criteria_prediction": {\n'
            '    "C1": {"score":<float 0..1 where 0.0=clearly unsatisfied, 0.5=uncertain, 1.0=clearly satisfied>,"reason":"<evidence from current frame>"},\n'
            '    "C2": {"score":<float 0..1>,"reason":"<evidence>"},\n'
            '    "C3": {"score":<float 0..1>,"reason":"<evidence>"}\n'
            "  },\n"
            f"{recommended_actions_schema}"
            '  "stop": <bool>\n'
            "}\n"
            "Set stop=true when you have gathered enough visual evidence to judge CVS for the current frame. "
            "Set stop=false if you need more tool calls to resolve uncertain scores. "
            "stop=true does NOT mean CVS is satisfied — it means you are done examining.\n"
            "Always populate current_frame_criteria_prediction with your best current estimates, even when action != ANSWER.\n"
            f"{rec_actions_instruction}"
            "Address all unsatisfied criteria, not just the lowest-scoring one. Avoid repeatedly validating already high-score ones unless contradictions appear.\n"
            'Optional routing hint: set action_args.reuse_short_term=true to reuse current P_s frames '
            "for zoom/clip_analyzer without a new clip_explorer step.\n"
            'Optional focus hint: set action_args.focus_criterion to the criterion currently targeted.\n'
            'If "action" == "zoom", include either action_args.bbox_norm_xyxy=[x1,y1,x2,y2] in [0,1], action_args.bbox_xyxy in pixels, or '
            "action_args.frame_crops=[{frame_id,bbox_norm_xyxy|bbox_xyxy}] for per-frame zoom boxes.\n"
            'If "action" == "clip_analyzer", "action_args.subquestion" must contain exactly one focused question.\n'
            "Do not ask multiple questions in one clip_analyzer step.\n"
            "Do not emit tool_calls. Do not include markdown."
        )
    else:
        single_crit = criteria_in_scope[0] if criteria_in_scope else "C1"
        decision_schema = (
            "\nReturn STRICT JSON only with keys exactly:\n"
            "{\n"
            f'  "action": "ANSWER" | {" | ".join([json.dumps(a) for a in mm_actions])},\n'
            '  "action_args": { ... },\n'
            '  "reason": "<For ANSWER: write a natural, surgeon-facing explanation. Start with 1-2 concise summary sentences, then briefly cite the strongest frame IDs or frame ranges and why they support the decision. Use practical operative wording, not robotic phrasing. If evidence is ambiguous, state what is missing and what should be checked next. For non-ANSWER actions: short intent sentence.>",\n'
            '  "current_frame_criteria_prediction": {\n'
            f'    "{single_crit}": {{"score":<float 0..1 where 0.0=clearly unsatisfied, 0.5=uncertain, 1.0=clearly satisfied>,"reason":"<evidence from current frame>"}}\n'
            "  },\n"
            f"{recommended_actions_schema}"
            '  "stop": <bool>\n'
            "}\n"
            "Set stop=true when you have gathered enough visual evidence to judge CVS for the current frame. "
            "Set stop=false if you need more tool calls to resolve uncertain scores. "
            "stop=true does NOT mean CVS is satisfied — it means you are done examining.\n"
            "Always populate current_frame_criteria_prediction with your best current estimates, even when action != ANSWER.\n"
            f"{rec_actions_instruction}"
            'Optional routing hint: set action_args.reuse_short_term=true to reuse current P_s frames '
            "for zoom/clip_analyzer without a new clip_explorer step.\n"
            'If "action" == "zoom", include either action_args.bbox_norm_xyxy=[x1,y1,x2,y2] in [0,1], action_args.bbox_xyxy in pixels, or '
            "action_args.frame_crops=[{frame_id,bbox_norm_xyxy|bbox_xyxy}] for per-frame zoom boxes.\n"
            'If "action" == "clip_analyzer", "action_args.subquestion" must contain exactly one focused question.\n'
            "Do not ask multiple questions in one clip_analyzer step.\n"
            "Do not emit tool_calls. Do not include markdown."
        )

    user_prompt = build_controller_prompt(
        state["criterion"], state["hm3"], mm_tools_for_schema,
        analyze_actions=state["ctx"].analyze_actions,
        recommend_actions=state["ctx"].recommend_actions,
        llm_calls=state.get("llm_calls", 0),
        max_steps=_effective_max_steps(state["ctx"]),
        num_preferred_tools=len(state["ctx"].preferred_tools or []),
        taxonomy_catalog=state["ctx"].taxonomy_catalog,
        fixed_k=state["ctx"].fixed_k,
    ) + decision_schema
    model = state["ctx"].models["controller"]
    _log_step("[agent-step] llm_call prompt:", {"prompt": user_prompt})
    response = model.invoke([
        SystemMessage(content=CONTROLLER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    log_level = int(os.getenv("SURGENT_LOG_LEVEL", "0") or 0)
    allowed_actions = {"ANSWER", *mm_tools_for_schema.keys()}
    decision = _parse_controller_decision(response, allowed_actions=allowed_actions)
    parse_failed = bool(decision.get("_parse_failed", False))
    if parse_failed and log_level >= 3:
        _log_step(
            "[AGENT-STEP] Parse Error Raw Response:",
            {"content": getattr(response, "content", None)},
        )
    if parse_failed:
        retry_instruction = (
            "\nYour previous output was not valid controller JSON.\n"
            "Return exactly one valid JSON object with the required keys only.\n"
            "Do not include markdown, code fences, or extra text."
        )
        retry_response = model.invoke([
            SystemMessage(content=CONTROLLER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt + retry_instruction),
        ])
        retry_decision = _parse_controller_decision(retry_response, allowed_actions=allowed_actions)
        retry_parse_failed = bool(retry_decision.get("_parse_failed", False))
        if not retry_parse_failed:
            response = retry_response
            decision = retry_decision
            parse_failed = False
        else:
            if log_level >= 3:
                _log_step(
                    "[AGENT-STEP] Parse Error Retry Raw Response:",
                    {"content": getattr(retry_response, "content", None)},
                )
    decision.pop("_parse_failed", None)
    decision = _enforce_unsatisfied_coverage_guard(decision, state["hm3"])
    state["hm3"].update_video_criteria_status(decision.get("criteria_status", {}))
    if decision.get("coverage_guard_triggered"):
        _log_step(
            "[agent-step] coverage_guard:",
            {
                "action": decision.get("action"),
                "decision": decision.get("decision"),
                "reason": decision.get("reason"),
                "coverage": decision.get("coverage"),
            },
        )
    state["hm3"].record_working(
        iteration=state["iteration"],
        objective=decision.get("reason", ""),
        reasoning_trace=decision.get("reason", ""),
        tool="controller",
    )
    if log_level >= 2:
        _log_step(
            "[agent-step] llm_call response:",
            {
                "content": getattr(response, "content", None),
                "tool_calls": getattr(response, "tool_calls", None),
                "decision": decision,
            },
        )
    elif log_level == 1:
        _log_step(
            "[agent-step] llm_call response:",
            {
                "decision": {
                    "action": decision.get("action"),
                    "reason": decision.get("reason"),
                    "decision": decision.get("decision"),
                    "confidence": decision.get("confidence"),
                    "stop": decision.get("stop"),
                }
            },
        )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "decision": decision,
    }


def _forced_answer_call(state: SurgentState) -> Dict[str, Any]:
    """Final LLM call forced when the step limit is reached.

    Re-uses the same controller prompt but appends an instruction requiring
    the model to produce an ANSWER decision using all evidence gathered so far.
    """
    mm_tools = _filtered_mm_tools(state.get("ctx"))
    criteria_in_scope = [
        token
        for token in [x.strip().upper() for x in re.split(r"[,\s]+", str(state.get("criterion") or "")) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    is_multi_criteria = len(criteria_in_scope) > 1

    # Skip recommended_actions when action_rec is a finalize tool —
    # forced_answer only decides CVS, action_rec runs separately after.
    ctx = state.get("ctx")
    finalize_set = set(ctx.finalize_tools) if ctx and ctx.finalize_tools else set()
    skip_rec_actions = "action_rec" in finalize_set

    forced_recommended_actions_schema = ""
    forced_rec_actions_instruction = ""
    if not skip_rec_actions:
        forced_recommended_actions_schema = (
            '  "recommended_actions": [\n'
            f"    // Your final best {AUDIT_V11_SIMPLE_TARGET_COUNT} actor-slot next actions synthesized from all tool outputs.\n"
            '    {"rank":1,"actor_role":"<str>","tool_type":"<str>","action_code":"<str>","target_structure":"<str>","target_context_1":"<str>","target_context_2":"<str>","intention":"<str>","confidence":<float 0..1>,"evidence":"<short phrase>"},\n'
            '    {"rank":2, ...},\n'
            '    {"rank":3, ...},\n'
            '    {"rank":4, ...}\n'
            "  ],\n"
        )
        forced_rec_actions_instruction = (
            "Always populate recommended_actions with your final best four actor-slot next actions: "
            "camera, left_instrument, right_instrument, other. Use CAMERA_NO_CHANGE when camera should not change, "
            "CAMERA_UNCERTAIN when uncertain, and reserve other for actions such as ICG_SWITCH.\n"
        )

    if is_multi_criteria:
        answer_schema = (
            "\nYou have reached the step limit. You MUST produce a final ANSWER now using all evidence gathered so far.\n"
            "Return STRICT JSON only with keys exactly:\n"
            "{\n"
            '  "action": "ANSWER",\n'
            '  "action_args": { ... },\n'
            '  "reason": "<explain progress toward all criteria and cite strongest evidence>",\n'
            '  "current_frame_criteria_prediction": {\n'
            '    "C1": {"score":<float 0..1 where 0.0=clearly unsatisfied, 0.5=uncertain, 1.0=clearly satisfied>,"reason":"<brief evidence>"},\n'
            '    "C2": {"score":<float 0..1>,"reason":"<brief evidence>"},\n'
            '    "C3": {"score":<float 0..1>,"reason":"<brief evidence>"}\n'
            "  },\n"
            f"{forced_recommended_actions_schema}"
            '  "stop": true\n'
            "}\n"
            "Always populate current_frame_criteria_prediction with your best estimates from all evidence gathered.\n"
            f"{forced_rec_actions_instruction}"
            "Do not emit tool_calls. Do not include markdown."
        )
    else:
        single_crit = criteria_in_scope[0] if criteria_in_scope else "C1"
        answer_schema = (
            "\nYou have reached the step limit. You MUST produce a final ANSWER now using all evidence gathered so far.\n"
            "Return STRICT JSON only with keys exactly:\n"
            "{\n"
            '  "action": "ANSWER",\n'
            '  "action_args": { ... },\n'
            '  "reason": "<explain your assessment citing the strongest evidence gathered so far>",\n'
            '  "current_frame_criteria_prediction": {\n'
            f'    "{single_crit}": {{"score":<float 0..1 where 0.0=clearly unsatisfied, 0.5=uncertain, 1.0=clearly satisfied>,"reason":"<brief evidence>"}}\n'
            "  },\n"
            f"{forced_recommended_actions_schema}"
            '  "stop": true\n'
            "}\n"
            "Always populate current_frame_criteria_prediction with your best estimates from all evidence gathered.\n"
            f"{forced_rec_actions_instruction}"
            "Do not emit tool_calls. Do not include markdown."
        )

    user_prompt = build_controller_prompt(
        state["criterion"], state["hm3"], mm_tools,
        analyze_actions=state["ctx"].analyze_actions,
        recommend_actions=state["ctx"].recommend_actions,
        llm_calls=state.get("llm_calls", 0),
        max_steps=_effective_max_steps(state["ctx"]),
        num_preferred_tools=len(state["ctx"].preferred_tools or []),
        taxonomy_catalog=state["ctx"].taxonomy_catalog,
        fixed_k=state["ctx"].fixed_k,
    ) + answer_schema
    model = state["ctx"].models["controller"]
    _log_step("[agent-step] forced_answer_call prompt:", {"prompt": user_prompt})
    response = model.invoke([
        SystemMessage(content=CONTROLLER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    allowed_actions = {"ANSWER"}
    decision = _parse_controller_decision(response, allowed_actions=allowed_actions)
    decision.pop("_parse_failed", None)
    # Force ANSWER regardless of what the model returned.
    decision["action"] = "ANSWER"
    decision["stop"] = True
    state["hm3"].update_video_criteria_status(decision.get("criteria_status", {}))
    state["hm3"].record_working(
        iteration=state["iteration"],
        objective=decision.get("reason", ""),
        reasoning_trace=f"forced_answer (step limit reached): {decision.get('reason', '')}",
        tool="controller",
    )
    log_level = int(os.getenv("SURGENT_LOG_LEVEL", "0") or 0)
    if log_level >= 1:
        _log_step(
            "[agent-step] forced_answer_call response:",
            {
                "decision": {
                    "action": decision.get("action"),
                    "reason": decision.get("reason"),
                    "decision": decision.get("decision"),
                    "confidence": decision.get("confidence"),
                    "stop": decision.get("stop"),
                }
            },
        )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "decision": decision,
    }


def _temporal_node(state: SurgentState) -> Dict[str, Any]:
    decision = state.get("decision") or dict(_DECISION_FALLBACK)
    mm_action = str(decision.get("action", ""))
    temporal_name = _MM_TO_TEMPORAL_TOOL.get(mm_action)
    if not temporal_name:
        return {"last_temporal": None}
    temporal_tools = temporal_registry()
    tool = temporal_tools.get(temporal_name)
    if tool is None:
        return {"last_temporal": None}
    hm3 = state["hm3"]
    ctx = state["ctx"]
    iteration = state["iteration"]
    args = {
        "memory": hm3,
        "model": ctx.models.get("controller"),
        "iteration": iteration,
        "query": _build_query(state["criterion"]),
    }
    hm3.record_working(
        iteration=iteration,
        objective=str(decision.get("reason") or ""),
        reasoning_trace=f"invoke {temporal_name} for criterion {state['criterion']}",
        tool=temporal_name,
    )
    _log_step("[agent-step] temporal_call:", {"tool": temporal_name, "args": args})
    observation = tool.invoke(args)
    _log_step("[agent-step] temporal_output:", {"tool": temporal_name, "output": observation})
    if isinstance(observation, dict) and "interval" in observation:
        record_result_from_tool(
            hm3,
            tool=temporal_name,
            interval=observation.get("interval") or {},
            output=observation,
            iteration=iteration,
        )
    tool_message = ToolMessage(
        content=json.dumps({"tool": temporal_name, "output": observation}),
        tool_call_id=f"temporal-{uuid.uuid4().hex}",
    )
    return {
        "messages": [tool_message],
        "last_temporal": observation if isinstance(observation, dict) else {"raw": observation},
    }


def _has_prior_result(hm3: HM3Memory, tool_name: str, interval: Dict[str, Any]) -> bool:
    """Check if a tool has already been called on this exact interval."""
    target_start = interval.get("start")
    target_end = interval.get("end")
    for entry in hm3.results.events:
        if entry.tool != tool_name:
            continue
        prev = entry.interval if isinstance(entry.interval, dict) else {}
        if prev.get("start") == target_start and prev.get("end") == target_end:
            return True
    return False


def _mm_node(state: SurgentState) -> Dict[str, Any]:
    decision = state.get("decision") or dict(_DECISION_FALLBACK)
    name = str(decision.get("action", "ANSWER"))
    if name == "ANSWER":
        return {}
    tools_by_name = _filtered_mm_tools(state.get("ctx"))
    tool = tools_by_name.get(name)
    if tool is None:
        return {"iteration": state["iteration"] + 1}
    hm3 = state["hm3"]
    ctx = state["ctx"]
    iteration = state["iteration"]
    args = dict(decision.get("action_args") or {})
    args.pop("reuse_short_term", None)
    args["reason"] = str(decision.get("reason") or args.get("reason", ""))
    last_temporal = state.get("last_temporal") or {}
    temporal_interval: Optional[Dict[str, Any]] = None
    if isinstance(last_temporal, dict):
        candidate = _normalize_interval_arg(last_temporal.get("interval"), memory=hm3)
        if isinstance(candidate, dict):
            temporal_interval = _clamp_interval_to_visible_window(hm3, candidate)
    parsed_interval = _normalize_interval_arg(args.get("interval"), memory=hm3)
    if parsed_interval is not None:
        args["interval"] = _clamp_interval_to_visible_window(hm3, parsed_interval)
    elif temporal_interval is not None:
        args["interval"] = temporal_interval
    elif name in {"scene_cvs_analyzer", "scene_cvs_act_analyzer", "action_rec"}:
        # Default to the current evaluation frame
        current_end = hm3.visible_end_idx
        if current_end is None and hm3.metadata:
            current_end = hm3.metadata.num_frames - 1
        current_end = max(0, int(current_end or 0))
        args["interval"] = _clamp_interval_to_visible_window(hm3, {"start": current_end, "end": current_end})
    elif name in {"clip_analyzer", "zoom", "scene_snapper"}:
        args["interval"] = _clamp_interval_to_visible_window(hm3, {"start": 0, "end": 0})
    # Duplicate call guard (safety net): scene_cvs_analyzer should already be
    # removed from the schema after the first call, but if the controller
    # still requests it, skip the call and let it try again with the tool
    # removed from the available actions on the next llm_call.
    resolved_interval = args.get("interval")
    if resolved_interval and name in {"scene_cvs_analyzer", "scene_cvs_act_analyzer"} and _has_prior_result(hm3, name, resolved_interval):
        _log_step(
            "[agent-step] duplicate_guard:",
            {"tool": name, "interval": resolved_interval, "action": "skipped, returning to controller"},
        )
        tool_message = ToolMessage(
            content=json.dumps({
                "tool": name,
                "duplicate_guard": True,
                "note": (
                    f"{name} was already called on interval {resolved_interval}. "
                    "This tool is no longer available. Use zoom, clip_analyzer, "
                    "scene_snapper, or set stop=true to finalize."
                ),
            }),
            tool_call_id=f"mm-dup-{uuid.uuid4().hex}",
        )
        return {
            "messages": [tool_message],
            "iteration": iteration + 1,
        }
    if name == "clip_analyzer":
        if "subquestion" not in args:
            args["subquestion"] = "Is the target CVS evidence clearly visible in this scoped clip?"
        args["subquestion"] = _normalize_single_subquestion(args.get("subquestion"))
    if name == "scene_snapper":
        args["memory"] = hm3
        args["model"] = ctx.models.get("scene")
        args["iteration"] = iteration
    elif name == "clip_analyzer":
        args["memory"] = hm3
        args["model"] = ctx.models.get("clip")
        args["iteration"] = iteration
    elif name == "scene_cvs_analyzer":
        args["memory"] = hm3
        args["model"] = ctx.models.get("scene")
        args["iteration"] = iteration
        args["taxonomy_catalog"] = ctx.taxonomy_catalog or {}
        args["criterion"] = state.get("criterion", "")
        args["preset"] = ctx.scene_preset or "direct"
    elif name == "scene_cvs_act_analyzer":
        args["memory"] = hm3
        args["model"] = ctx.models.get("scene")
        args["iteration"] = iteration
        args["taxonomy_catalog"] = ctx.taxonomy_catalog or {}
        args["criterion"] = state.get("criterion", "")
        args["preset"] = ctx.scene_preset or "direct"
        args["action_mode"] = "recommend" if ctx.recommend_actions else "predict"
        if ctx.fixed_k is not None:
            args["fixed_k"] = ctx.fixed_k
    elif name == "action_rec":
        args["memory"] = hm3
        args["model"] = ctx.models.get("scene")
        args["iteration"] = iteration
        args["taxonomy_catalog"] = ctx.taxonomy_catalog or {}
        args["criterion"] = state.get("criterion", "")
        if ctx.fixed_k is not None:
            args["fixed_k"] = ctx.fixed_k
        args["cvs_context_mode"] = ctx.cvs_context_mode
        if ctx.one_per_actor:
            args["one_per_actor"] = True
        args["action_rec_rules"] = ctx.action_rec_rules
    elif name == "zoom":
        args["memory"] = hm3
        args["iteration"] = iteration
        if "target_frame_id" in args and args["target_frame_id"] is not None:
            args["target_frame_id"] = str(args["target_frame_id"])
    working_args = {k: v for k, v in args.items() if k not in {"memory", "model", "taxonomy_catalog"}}
    hm3.record_working(
        iteration=iteration,
        objective=str(decision.get("reason") or ""),
        reasoning_trace=(
            f"invoke {name} with args "
            f"{json.dumps(_compact_for_log(working_args), default=str)}"
        ),
        tool=name,
    )
    _log_step("[agent-step] mm_call:", {"tool": name, "args": args, "decision": decision})
    observation = tool.invoke(args)
    _log_step("[agent-step] mm_output:", {"tool": name, "output": observation})
    # Record mm tool result so auto_first_step / auto_finalize_step
    # can detect that this tool has already been called.
    if isinstance(observation, dict):
        record_result_from_tool(
            hm3,
            tool=name,
            interval=observation.get("interval") or args.get("interval") or {},
            output=observation,
            iteration=iteration,
        )
    tool_message = ToolMessage(
        content=json.dumps({"tool": name, "output": observation}),
        tool_call_id=f"mm-{uuid.uuid4().hex}",
    )
    return {
        "messages": [tool_message],
        "iteration": iteration + 1,
    }


def _route_after_llm(state: SurgentState) -> str:
    max_steps = _effective_max_steps(state.get("ctx"))
    if state.get("llm_calls", 0) >= max_steps:
        return "forced_answer"
    decision = state.get("decision") or dict(_DECISION_FALLBACK)
    action = str(decision.get("action", "ANSWER"))
    stop = bool(decision.get("stop", action == "ANSWER"))
    if stop or action == "ANSWER":
        ctx = state.get("ctx")
        if ctx is not None and ctx.finalize_tools:
            return "auto_finalize_step"
        return END
    mm_tools = _controller_mm_tools(state.get("ctx"))
    if action in mm_tools:
        action_args = decision.get("action_args")
        reuse_short_term = (
            isinstance(action_args, dict)
            and bool(action_args.get("reuse_short_term"))
            and action in {"zoom", "clip_analyzer"}
            and bool(state["hm3"].sensory.short_term_pool)
        )
        if reuse_short_term:
            return "mm_node"
        if action in _MM_TO_TEMPORAL_TOOL:
            return "temporal_node"
        return "mm_node"
    return END


def _should_continue(state: SurgentState) -> str:
    """Backward-compatible alias for tests/importers."""
    if state.get("decision") is None:
        max_steps = _effective_max_steps(state.get("ctx"))
        if state.get("llm_calls", 0) >= max_steps:
            return "forced_answer"
        messages = state.get("messages", [])
        if messages and getattr(messages[-1], "tool_calls", None):
            return "tool_node"
        return END
    return _route_after_llm(state)


def _tool_node(state: SurgentState) -> Dict[str, Any]:
    """Backward-compatible shim; graph execution now uses temporal/mm split nodes."""
    if state.get("decision"):
        temporal_update = _temporal_node(state)
        mm_update = _mm_node(state)
        merged_messages: List[AnyMessage] = []
        merged_messages.extend(temporal_update.get("messages", []))
        merged_messages.extend(mm_update.get("messages", []))
        update: Dict[str, Any] = {"messages": merged_messages}
        if "iteration" in mm_update:
            update["iteration"] = mm_update["iteration"]
        if "last_temporal" in temporal_update:
            update["last_temporal"] = temporal_update["last_temporal"]
        return update
    messages = list(state.get("messages", []))
    if not messages:
        return {}
    hm3 = state["hm3"]
    ctx = state["ctx"]
    iteration = state["iteration"]
    out_messages: List[AnyMessage] = []
    for tool_call in getattr(messages[-1], "tool_calls", []) or []:
        name = str(tool_call.get("name", ""))
        args = dict(tool_call.get("args") or {})
        hm3.record_working(
            iteration=iteration,
            objective=str(args.get("reason", "")),
            reasoning_trace=f"invoke {name}",
            tool=name or None,
        )
        if name in temporal_registry():
            temporal_args = dict(args)
            temporal_args["memory"] = hm3
            temporal_args["model"] = ctx.models.get("controller")
            temporal_args["iteration"] = iteration
            temporal_args["query"] = _build_query(state["criterion"])
            observation = temporal_registry()[name].invoke(temporal_args)
            if isinstance(observation, dict) and "interval" in observation:
                record_result_from_tool(
                    hm3,
                    tool=name,
                    interval=observation.get("interval") or {},
                    output=observation,
                    iteration=iteration,
                )
            out_messages.append(
                ToolMessage(
                    content=json.dumps({"tool": name, "output": observation}),
                    tool_call_id=str(tool_call.get("id") or f"legacy-temporal-{uuid.uuid4().hex}"),
                )
            )
        elif name in _mm_tool_registry():
            mm_args = dict(args)
            mm_args["memory"] = hm3
            mm_args["iteration"] = iteration
            if name == "clip_analyzer":
                mm_args["model"] = ctx.models.get("clip")
            elif name == "scene_snapper":
                mm_args["model"] = ctx.models.get("scene")
            observation = _mm_tool_registry()[name].invoke(mm_args)
            # Record mm tool result so auto_first_step / auto_finalize_step
            # can detect that this tool has already been called.
            if isinstance(observation, dict):
                record_result_from_tool(
                    hm3,
                    tool=name,
                    interval=observation.get("interval") or {},
                    output=observation,
                    iteration=iteration,
                )
            out_messages.append(
                ToolMessage(
                    content=json.dumps({"tool": name, "output": observation}),
                    tool_call_id=str(tool_call.get("id") or f"legacy-mm-{uuid.uuid4().hex}"),
                )
            )
    return {"messages": out_messages, "iteration": iteration + 1}


def _route_after_mm(state: SurgentState) -> str:
    """Route after mm_node: handle finalize phase or loop back to auto_first_step."""
    # If we're in the finalize phase, route back to auto_finalize_step
    if state.get("_finalize_phase", False):
        return "auto_finalize_step"
    ctx = state.get("ctx")
    preferred = ctx.preferred_tools if ctx is not None else None
    if preferred:
        hm3 = state["hm3"]
        start_iteration = state.get("start_iteration", 0)
        called_tools = {event.tool for event in hm3.results.events if event.iteration >= start_iteration}
        for tool_name in preferred:
            if tool_name not in called_tools:
                return "auto_first_step"
    return "llm_call"


def _route_forced_answer(state: SurgentState) -> str:
    """Route after forced_answer: finalize tools if configured, otherwise END."""
    ctx = state.get("ctx")
    if ctx is not None and ctx.finalize_tools:
        return "auto_finalize_step"
    return END


def build_agent_graph() -> StateGraph:
    graph = StateGraph(SurgentState)
    graph.add_node("auto_first_step", _auto_first_step)
    graph.add_node("llm_call", _llm_call)
    graph.add_node("forced_answer", _forced_answer_call)
    graph.add_node("temporal_node", _temporal_node)
    graph.add_node("mm_node", _mm_node)
    graph.add_node("auto_finalize_step", _auto_finalize_step)
    graph.add_edge(START, "auto_first_step")
    graph.add_conditional_edges("auto_first_step", _route_auto_first_step, ["temporal_node", "mm_node", "llm_call"])
    graph.add_conditional_edges("llm_call", _route_after_llm, ["temporal_node", "mm_node", "forced_answer", "auto_finalize_step", END])
    graph.add_conditional_edges("forced_answer", _route_forced_answer, ["auto_finalize_step", END])
    graph.add_edge("temporal_node", "mm_node")
    graph.add_conditional_edges("mm_node", _route_after_mm, ["auto_first_step", "auto_finalize_step", "llm_call"])
    graph.add_conditional_edges("auto_finalize_step", _route_auto_finalize_step, ["temporal_node", "mm_node", END])
    return graph


def _populate_hm3_metadata_from_image_paths(hm3: HM3Memory, image_paths: List[str]) -> None:
    if not image_paths:
        return
    image_dir = str(Path(image_paths[0]).parent)
    frames = []
    for idx, path in enumerate(image_paths):
        frame_id = Path(path).stem
        frames.append(FrameRef(idx=idx, frame_id=frame_id, path=path))
    hm3.metadata = VideoMetadata(
        video_id=Path(image_dir).name,
        image_dir=image_dir,
        frames=frames,
        num_frames=len(frames),
    )


def run_cvs_agent_with_memory(
    *,
    criterion: str,
    ctx: SurgentContext,
    hm3: HM3Memory,
    compiled_graph: Optional[Any] = None,
) -> Dict[str, Any]:
    graph = compiled_graph or build_agent_graph().compile()
    start_iteration = int(getattr(hm3, "iteration_cursor", 0) or 0)
    agent_state: SurgentState = {
        "messages": [HumanMessage(content="Begin analysis.")],
        "llm_calls": 0,
        "ctx": ctx,
        "hm3": hm3,
        "iteration": start_iteration,
        "start_iteration": start_iteration,
        "criterion": criterion,
        "decision": None,
        "last_temporal": None,
        "_finalize_phase": False,
    }
    final_state = graph.invoke(agent_state)
    next_iteration = final_state.get("iteration", start_iteration)
    try:
        hm3.iteration_cursor = int(next_iteration)
    except Exception:
        hm3.iteration_cursor = start_iteration
    return final_state


def run_cvs_agent_sync(
    request: CVSRequest,
    model_name: Optional[str] = None,
    criterion: Optional[str] = None,
    ctx: Optional[SurgentContext] = None,
) -> str:
    if request.criteria and len(request.criteria) != 1:
        raise SystemExit(
            "Only one CVS criterion is allowed per run. "
            "Provide a single criterion (e.g., C1)."
        )
    if ctx is None:
        raise SystemExit("SurgentContext is required to run the LangGraph agent.")

    resolved_criterion = criterion or (request.criteria[0] if request.criteria else "C1")
    graph = build_agent_graph().compile()
    hm3 = HM3Memory()
    _populate_hm3_metadata_from_image_paths(hm3, request.image_paths)
    final_state = run_cvs_agent_with_memory(
        criterion=resolved_criterion,
        ctx=ctx,
        hm3=hm3,
        compiled_graph=graph,
    )
    final_decision = final_state.get("decision")
    if isinstance(final_decision, dict) and str(final_decision.get("action", "")) == "ANSWER":
        answer_payload = {
            "action": "ANSWER",
            "action_args": final_decision.get("action_args") or {},
            "reason": str(final_decision.get("reason") or ""),
            "decision": _normalize_final_decision(final_decision.get("decision")),
            "confidence": float(final_decision.get("confidence", 0.0) or 0.0),
            "stop": True,
        }
        return json.dumps(answer_payload, ensure_ascii=False, indent=2)
    messages = final_state.get("messages", [])
    if messages:
        return str(messages[-1].content or "")
    return "{}"
