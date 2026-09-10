from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
# langgraph imports removed; direct model call used in scene_snapper

from .audit_simple_actions import (
    augment_taxonomy_catalog_for_audit_v11_simple,
    build_audit_v11_simple_prompt_block,
    normalize_actor_slot_actions,
)
from .memory_hm3 import HM3Memory
from .shared_prompt_cores import (
    _action_schema_block,
    _cvs_definition_block,
    _fixed_k_constraint_block,
    _taxonomy_options_block,
    SYSTEM_PREAMBLE,
    build_action_rec_prompt_body,
    build_action_rec_response_schema,
    build_scene_cvs_frame_intro,
    build_scene_cvs_prompt_body,
    build_scene_cvs_response_schema,
)
from .schemas import CRITERION_DEFINITIONS, DEFAULT_CVS_CRITERIA


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _load_scene_images(pool: List[Any], max_images: int = 6) -> List[Image.Image]:
    images: List[Image.Image] = []
    for item in pool:
        if isinstance(item, Image.Image):
            images.append(item)
        elif isinstance(item, str) and os.path.exists(item):
            images.append(Image.open(item).convert("RGB"))
        elif isinstance(item, dict) and "path" in item and os.path.exists(item["path"]):
            images.append(Image.open(item["path"]).convert("RGB"))
        if len(images) >= max_images:
            break
    return images


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    for match in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_zoom_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y", "zoom", "needed", "necessary"}


def _normalize_bbox_norm(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return None
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _parse_clip_analyzer_payload(text: str) -> tuple[str, float, bool, Optional[List[float]], str]:
    raw = (text or "").strip()
    if not raw:
        return "template-only", 0.0, False, None, ""
    parsed = _extract_json_object(raw)
    if parsed:
        answer = str(parsed.get("answer") or parsed.get("analysis") or "").strip() or "template-only"
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        needs_zoom = _parse_zoom_bool(parsed.get("needs_zoom"))
        bbox_norm = _normalize_bbox_norm(
            parsed.get("suggested_zoom_bbox_norm_xyxy")
            or parsed.get("zoom_bbox_norm_xyxy")
            or parsed.get("bbox_norm_xyxy")
        )
        zoom_reason = str(parsed.get("zoom_reason") or parsed.get("zoom_rationale") or "").strip()
        return answer, confidence, needs_zoom, bbox_norm, zoom_reason

    answer = raw
    confidence = 0.0
    needs_zoom = False
    zoom_reason = ""
    bbox_norm: Optional[List[float]] = None

    answer_match = re.search(r"(?im)^answer:\s*(.+)$", raw)
    if answer_match:
        answer = answer_match.group(1).strip()

    conf_match = re.search(r"(?im)^confidence:\s*([0-9]*\.?[0-9]+)\s*$", raw)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except Exception:
            confidence = 0.0
    else:
        loose_conf = re.search(r"([0-9]*\.?[0-9]+)", raw)
        if loose_conf:
            try:
                confidence = float(loose_conf.group(1))
            except Exception:
                confidence = 0.0
    needs_zoom_match = re.search(r"(?im)^needs[_\s-]*zoom:\s*(.+)$", raw)
    if needs_zoom_match:
        needs_zoom = _parse_zoom_bool(needs_zoom_match.group(1))
    zoom_reason_match = re.search(r"(?im)^zoom[_\s-]*reason:\s*(.+)$", raw)
    if zoom_reason_match:
        zoom_reason = zoom_reason_match.group(1).strip()
    bbox_match = re.search(
        r"(?im)^(?:suggested[_\s-]*zoom[_\s-]*bbox[_\s-]*norm[_\s-]*xyxy|zoom[_\s-]*bbox[_\s-]*norm[_\s-]*xyxy|bbox[_\s-]*norm[_\s-]*xyxy)\s*:\s*(\[[^\]]+\])",
        raw,
    )
    if bbox_match:
        try:
            bbox_norm = _normalize_bbox_norm(json.loads(bbox_match.group(1)))
        except Exception:
            bbox_norm = None

    confidence = max(0.0, min(1.0, confidence))
    return answer or "template-only", confidence, needs_zoom, bbox_norm, zoom_reason


@tool
def scene_snapper(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
) -> Dict[str, Any]:
    """
    Summarize long-range frames stored in P_l and write results into Result Memory.

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)

    Output (dict):
      {
        "tool": "scene_snapper",
        "interval": {...},
        "reason": str,
        "summary": str
      }
    """
    frames: List[Any] = list(memory.sensory.long_term_pool)
    summary = "template-only: no scene captioning implemented"
    if model is not None:
        images = _load_scene_images(frames)
        if images:
            frame_paths = []
            for item in frames:
                if isinstance(item, dict) and "path" in item:
                    frame_paths.append(str(item["path"]))
            start_frame = int(interval.get("start", 0))
            end_frame = int(interval.get("end", start_frame))
            system_prompt = (
                "You are a surgical video frame captioning assistant.\n"
                "The frames are from a laparoscopic cholecystectomy procedure.\n"
                "Each extracted frame displays the global frame id in white text at the top-left.\n"
                "Each picture contains one 3x2 mosaic (6 frames), ordered row-major from top-left to bottom-right.\n\n"
                f"Caption the provided {len(frame_paths)} frames sampled uniformly from frame range "
                f"{start_frame}-{end_frame}.\n"
                "The frames represent a continuous sequence from the video.\n\n"
                "Write one concise English sentence describing the main surgical scene and action.\n"
                "Focus on:\n"
                "- anatomical structures (gallbladder, liver, cystic duct, cystic artery, Calot's triangle)\n"
                "- surgical instruments and their interactions (traction, dissection, clipping, cautery)\n"
                "- whether the operative field is clearly exposed or obscured\n\n"
                "If key structures are not visible, explicitly state \"structures not clearly visible\"."
            )
            user_parts = [{"type": "text", "text": "Caption the scene in one concise sentence."}]
            for img in images:
                user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})
            response = model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_parts),
            ])
            summary = (response.content or "").strip() or summary
    output = {
        "tool": "scene_snapper",
        "interval": interval,
        "reason": reason,
        "frames": frames,
        "summary": summary,
    }
    memory.record_result(tool="scene_snapper", interval=interval, output=output, iteration=iteration)
    return output


@tool
def clip_analyzer(
    interval: Dict[str, Any],
    subquestion: str,
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
) -> Dict[str, Any]:
    """
    Clip Analyzer. Clip Analyzer is focused on capturing fine-grained local semantic
    details. Given the frames F stored in short-term perception pool P_s and a
    sub-question subquestion, this tool jointly examines these frames, synthesizing
    cross-clip observations to infer detailed spatial and semantic information.
    It then produces an answer A_sub to the sub-question along with a confidence
    score S_sub that reflects the reliability of its reasoning. This tool enables
    precise semantic discrimination and temporal understanding within local
    segments, providing essential fine-grained evidence and validation support
    for long-form video reasoning. A_sub, S_sub = ClipAnalyzer(F, subquestion), F in P_s.

    Input:
      interval: {"start": int, "end": int}
      subquestion: str
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)

    Output (dict):
      {
        "tool": "clip_analyzer",
        "interval": {...},
        "subquestion": str,
        "reason": str,
        "answer": str,
        "confidence": float,
        "needs_zoom": bool,
        "suggested_zoom_bbox_norm_xyxy": [float, float, float, float] | None,
        "zoom_reason": str
      }
    """
    frames: List[Any] = list(memory.sensory.short_term_pool)
    answer = "template-only"
    confidence = 0.0
    needs_zoom = False
    suggested_zoom_bbox_norm_xyxy: Optional[List[float]] = None
    zoom_reason = ""
    if model is not None:
        images = _load_scene_images(frames)
        if images:
            num_frames = len(frames)
            start_frame = int(interval.get("start", 0))
            end_frame = int(interval.get("end", start_frame))
            system_prompt = (
                "You are an expert video frame analyst specializing in surgical videos.\n\n"
                "The frames are from a laparoscopic cholecystectomy (gallbladder removal) procedure.\n"
                "Focus on clinically relevant visual evidence: anatomical structures (gallbladder, liver bed, "
                "hepatocystic triangle/Calot's triangle, cystic duct, cystic artery), tissue dissection state, "
                "and instrument interactions (traction, dissection, clipping, cutting, cautery).\n"
                "If key structures are not clearly visible or the view is occluded (smoke/blur/blood), "
                "explicitly say so rather than guessing.\n\n"
                f"Analyze the provided {num_frames} frames sampled from frame range {start_frame} - {end_frame} "
                "and answer the given question.\n"
                "The frames represent a continuous sequence from the video. Analyze them collectively to provide "
                "a comprehensive answer.\n"
                "Each extracted frame displays the global frame id in white text at the top-left.\n\n"
                "Analyze this frame sequence and answer the question below.\n"
                "Return STRICT JSON only with this schema:\n"
                "{\n"
                '  "answer": "<concise clinically grounded answer>",\n'
                '  "confidence": <float 0..1>,\n'
                '  "needs_zoom": <true|false>,\n'
                '  "zoom_reason": "<why higher magnification is needed or empty>",\n'
                '  "suggested_zoom_bbox_norm_xyxy": [x1,y1,x2,y2] or null\n'
                "}\n"
                "Zoom guidance:\n"
                "- Set needs_zoom=true only when the current evidence is limited by scale/visibility.\n"
                "- If needs_zoom=true, provide suggested_zoom_bbox_norm_xyxy in normalized coordinates [0,1].\n"
                "- Ensure x2>x1 and y2>y1.\n"
                "- If needs_zoom=false, set suggested_zoom_bbox_norm_xyxy to null.\n\n"
                f"Question:\n{subquestion}"
            )
            user_parts = [{"type": "text", "text": "Analyze the frame sequence and answer the question."}]
            for img in images:
                user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})
            response = model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_parts),
            ])
            text = (response.content or "").strip()
            (
                answer,
                confidence,
                needs_zoom,
                suggested_zoom_bbox_norm_xyxy,
                zoom_reason,
            ) = _parse_clip_analyzer_payload(text)
    output = {
        "tool": "clip_analyzer",
        "interval": interval,
        "subquestion": subquestion,
        "reason": reason,
        "answer": answer,
        "confidence": confidence,
        "needs_zoom": needs_zoom,
        "suggested_zoom_bbox_norm_xyxy": suggested_zoom_bbox_norm_xyxy,
        "zoom_reason": zoom_reason,
        "frames": frames,
    }
    memory.record_result_and_clear_short_term(
        tool="clip_analyzer",
        interval=interval,
        output=output,
        iteration=iteration,
    )
    return output


def _enforce_fixed_k(actions_ranked: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper; actor-slot mode now always returns 4 slots."""
    del k
    return normalize_actor_slot_actions(actions_ranked)


def _parse_scene_cvs_output(text: str) -> Dict[str, Any]:
    parsed = _extract_json_object(text)
    if not parsed:
        return {"raw_text": text}
    return parsed


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


@tool
def scene_cvs_act_analyzer(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    taxonomy_catalog: Dict[str, Any],
    criterion: str,
    preset: str = "direct",
    action_mode: str = "predict",
    fixed_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scene CVS + Action Analyzer (combined). Frame-level CVS criterion prediction
    tool that uses a subrubric approach: the model generates its own diagnostic
    checklist, then predicts CVS scores and ranked surgical actions. This mirrors
    the baseline subrubric preset for maximum action prediction accuracy.

    Loads the current evaluation frame directly from metadata (visible_end_idx)
    and optionally includes P_s frames from clip_explorer. Produces structured
    CVS predictions with per-criterion confidence scores, a self-generated
    checklist, and four actor-slot next actions from the taxonomy.

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      taxonomy_catalog: dict (runtime) - loaded taxonomy with options
      criterion: str (runtime) - active criteria e.g. "C1,C2,C3"

    Output (dict):
      {
        "tool": "scene_cvs_act_analyzer",
        "interval": {...},
        "reason": str,
        "cvs_predictions": {"C1": float, "C2": float, "C3": float},
        "checklist": {"C1": [...], "C2": [...], "C3": [...]},
        "actions_ranked": [{rank, actor_role, tool_type, action_code, evidence, ...}, ...],
        "recommended_action": first_action or {},
        "summary": str
      }
    """
    import re as _re
    criteria = [
        token
        for token in [x.strip().upper() for x in _re.split(r"[,\s]+", str(criterion or "")) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    if not criteria:
        criteria = list(DEFAULT_CVS_CRITERIA)

    # Load individual frames: current frame + optionally P_s frames from clip_explorer
    frames: List[Dict[str, Any]] = []

    # Always include current evaluation frame from metadata
    if memory.metadata and memory.metadata.frames:
        current_idx = memory.visible_end_idx
        if current_idx is None:
            current_idx = memory.metadata.num_frames - 1
        current_idx = max(0, min(int(current_idx), len(memory.metadata.frames) - 1))
        current_frame = memory.metadata.frames[current_idx]
        frames.append({
            "type": "frame",
            "frame_id": str(current_frame.frame_id),
            "path": str(current_frame.path),
            "idx": int(current_frame.idx),
        })

    # Optionally include P_s frames from clip_explorer (if available, excluding duplicates)
    current_paths = {f["path"] for f in frames}
    for item in memory.sensory.short_term_pool:
        if isinstance(item, dict) and item.get("type") == "frame" and item.get("path"):
            if item["path"] not in current_paths:
                frames.append(item)
                current_paths.add(item["path"])

    default_output: Dict[str, Any] = {
        "tool": "scene_cvs_act_analyzer",
        "interval": interval,
        "reason": reason,
        "cvs_predictions": {c: 0.0 for c in criteria},
        "checklist": {c: [] for c in criteria},
        "actions_ranked": [],
        "recommended_action": {},
        "summary": "no analysis performed",
    }

    if model is None or not frames:
        if not frames:
            default_output["summary"] = "no frames available from metadata"
        memory.record_result(tool="scene_cvs_act_analyzer", interval=interval, output=default_output, iteration=iteration)
        return default_output

    images = _load_scene_images(frames)
    if not images:
        default_output["summary"] = "no loadable images from metadata frames"
        memory.record_result(tool="scene_cvs_act_analyzer", interval=interval, output=default_output, iteration=iteration)
        return default_output

    # Current frame is always frames[0]
    current_frame_id = frames[0].get("frame_id", "")
    num_frames = len(frames)

    system_prompt = SYSTEM_PREAMBLE

    audit_simple_catalog = augment_taxonomy_catalog_for_audit_v11_simple(taxonomy_catalog)
    taxonomy_block = _taxonomy_options_block(audit_simple_catalog)
    audit_simple_block = build_audit_v11_simple_prompt_block()

    if num_frames == 1:
        frame_intro = (
            "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
            f"This is frame {current_frame_id} — the CURRENT evaluation frame.\n"
        )
    else:
        frame_intro = (
            f"You are viewing {num_frames} frames from a laparoscopic cholecystectomy video.\n"
            f"Frame {current_frame_id} is the CURRENT evaluation frame — scores must reflect "
            f"the state at this frame. Earlier frames provide temporal context.\n"
        )

    # Build prompt based on preset (mirrors baseline.py presets)
    effective_preset = preset if preset in ("direct", "cot", "subrubric") else "direct"

    if effective_preset in ("direct", "cot"):
        include_rationale = effective_preset == "cot"
        rationale_schema = ""
        rationale_instruction = ""
        if include_rationale:
            rationale_schema = '  "rationale": { "c1": "<string>", "c2": "<string>", "c3": "<string>" },\n'
            rationale_instruction = (
                "\n- Provide a short rationale per CVS criterion before giving scores."
                "\n- Output rationale before pred."
            )
        del action_mode
        action_task = (
            f"Task 2: Predict exactly {AUDIT_V11_SIMPLE_TARGET_COUNT} NEXT surgical actions that should happen next.\n"
            "This is prospective next-action planning from the visible video prefix and current CVS state,\n"
            "not retrospective labeling of the full clip and not a description of only the current visible action.\n"
            "Return exactly one action slot each for camera, left_instrument, right_instrument, and other.\n"
            "Use ONLY values from the allowed audit_v11 simple interface and taxonomy below.\n"
            'Use "(not set)" only when a slot is truly absent or unsupported by the evidence.\n'
            "If CVS is already adequate, still return all four slots, using CAMERA_NO_CHANGE for a stable camera\n"
            "and no additional other action when appropriate.\n"
        )
        action_important = (
            "IMPORTANT:\n"
            "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
            "- actor_role must be exactly one of camera, left_instrument, right_instrument, other.\n"
            "- Provide exactly one slot for each actor_role.\n"
            "- Address all unsatisfied CVS criteria, not just the lowest-scoring one.\n"
            "  Each action's intention should reflect which specific criterion it targets.\n"
            "- Camera recommendations must use CAMERA_NO_CHANGE when the current view should be maintained.\n"
            "- Use CAMERA_UNCERTAIN only when camera recommendation is actually uncertain.\n"
            '- Use the other slot for actions such as ICG_SWITCH, or leave it as no additional other action.\n'
            f"{_fixed_k_constraint_block(fixed_k or AUDIT_V11_SIMPLE_TARGET_COUNT)}"
            f"\n{audit_simple_block}\n"
        )
        user_prompt = (
            f"{frame_intro}"
            "\n"
            "Task 1: Predict Critical View of Safety (CVS) confidence scores.\n"
            "Return a probability in [0,1] for each criterion.\n"
            "\n"
            "Criterion definitions:\n"
            f"{_cvs_definition_block()}\n"
            "\n"
            f"{action_task}\n"
            "Allowed taxonomy values:\n"
            f"{taxonomy_block}\n"
            "\n"
            f"{action_important}"
            f"{rationale_instruction}\n"
            "\n"
            "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
            "{\n"
            f"{rationale_schema}"
            '  "pred": { "c1": float, "c2": float, "c3": float },\n'
            f"{_action_schema_block(fixed_k=fixed_k)}\n"
            "}\n"
        )
    else:
        # subrubric preset
        del action_mode
        action_step = (
            "STEP 3 — Next-action planning:\n"
            f"Predict exactly {AUDIT_V11_SIMPLE_TARGET_COUNT} NEXT surgical actions that should happen next.\n"
            "This is prospective next-action planning from the visible video prefix and current CVS state,\n"
            "not retrospective clip labeling and not a description of only the current visible action.\n"
            "Return exactly one action slot each for camera, left_instrument, right_instrument, and other.\n"
            "Use ONLY values from the allowed audit_v11 simple interface and taxonomy below.\n"
            'Use "(not set)" only when a slot is truly absent or unsupported by the evidence.\n'
            "If CVS is already adequate, still return all four slots, using CAMERA_NO_CHANGE for a stable camera\n"
            "and no additional other action when appropriate.\n"
        )
        action_important = (
            "IMPORTANT:\n"
            "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
            "- actor_role must be exactly one of camera, left_instrument, right_instrument, other.\n"
            "- Provide exactly one slot for each actor_role.\n"
            "- Address all unsatisfied CVS criteria, not just the lowest-scoring one.\n"
            "  Each action's intention should reflect which specific criterion it targets.\n"
            "- Camera recommendations must use CAMERA_NO_CHANGE when the current view should be maintained.\n"
            "- Use CAMERA_UNCERTAIN only when camera recommendation is actually uncertain.\n"
            '- Use the other slot for actions such as ICG_SWITCH, or leave it as no additional other action.\n'
            f"{_fixed_k_constraint_block(fixed_k or AUDIT_V11_SIMPLE_TARGET_COUNT)}"
            f"\n{audit_simple_block}\n"
        )
        user_prompt = (
            f"{frame_intro}"
            "\n"
            "STEP 1 — Self-rubric checklist:\n"
            "For each CVS criterion (C1, C2, C3), generate diagnostic questions that\n"
            "help assess whether the criterion is met. Answer each question as\n"
            "yes/no/uncertain based on the CURRENT evaluation frame (use earlier frames for context).\n"
            "\n"
            "Criterion definitions:\n"
            f"{_cvs_definition_block()}\n"
            "\n"
            "STEP 2 — CVS prediction:\n"
            "Using the checklist answers as evidence, predict confidence scores\n"
            "in [0,1] for C1, C2, C3 at the CURRENT evaluation frame.\n"
            "\n"
            f"{action_step}\n"
            "Allowed taxonomy values:\n"
            f"{taxonomy_block}\n"
            "\n"
            f"{action_important}"
            "\n"
            "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
            "{\n"
            '  "checklist": {\n'
            '    "C1": [ {"question": "<string>", "answer": "yes"|"no"|"uncertain"}, ... ],\n'
            '    "C2": [ ... ],\n'
            '    "C3": [ ... ]\n'
            "  },\n"
            '  "pred": { "c1": float, "c2": float, "c3": float },\n'
            f"{_action_schema_block(fixed_k=fixed_k)}\n"
            "}\n"
        )

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": user_prompt}
    ]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    text = (response.content or "").strip()
    parsed = _parse_scene_cvs_output(text)

    # Extract checklist (self-generated)
    checklist_raw = parsed.get("checklist", {})
    if not isinstance(checklist_raw, dict):
        checklist_raw = {}
    checklist: Dict[str, List[Dict[str, str]]] = {}
    for c in criteria:
        items = checklist_raw.get(c, [])
        if isinstance(items, list):
            checklist[c] = [
                {
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "uncertain")),
                }
                for item in items
                if isinstance(item, dict)
            ]
        else:
            checklist[c] = []

    # Extract pred → cvs_predictions (normalize lowercase keys to uppercase)
    pred_raw = parsed.get("pred", {})
    if not isinstance(pred_raw, dict):
        pred_raw = {}
    cvs_predictions: Dict[str, float] = {}
    for c in criteria:
        # Try both lowercase and uppercase key
        val = pred_raw.get(c.lower(), pred_raw.get(c, 0.0))
        try:
            cvs_predictions[c] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            cvs_predictions[c] = 0.0

    # Extract actions_ranked
    actions_raw = parsed.get("actions", parsed.get("actions_ranked", []))
    if not isinstance(actions_raw, list):
        actions_raw = []
    actions_ranked: List[Dict[str, Any]] = []
    for a in actions_raw:
        if not isinstance(a, dict):
            continue
        actions_ranked.append({
            "rank": a.get("rank"),
            "actor_role": str(a.get("actor_role", "(not set)")),
            "tool_type": str(a.get("tool_type", "(not set)")),
            "action_code": str(a.get("action_code", "(not set)")),
            "target_structure": str(a.get("target_structure", "(not set)")),
            "target_context_1": str(a.get("target_context_1", "(not set)")),
            "target_context_2": str(a.get("target_context_2", "(not set)")),
            "intention": str(a.get("intention", "(not set)")),
            "confidence": _safe_float(a.get("confidence", 0.0)),
            "evidence": str(a.get("evidence", "")),
        })

    actions_ranked = _enforce_fixed_k(actions_ranked, fixed_k or AUDIT_V11_SIMPLE_TARGET_COUNT)

    # Backward compat: recommended_action from first ranked action
    recommended_action: Dict[str, Any] = {}
    if actions_ranked:
        first = actions_ranked[0]
        recommended_action = {
            "actor_role": first.get("actor_role", "(not set)"),
            "tool_type": first.get("tool_type", "(not set)"),
            "action_code": first.get("action_code", "(not set)"),
            "target_structure": first.get("target_structure", "(not set)"),
            "target_context": first.get("target_context_1", "(not set)"),
            "target_context_2": first.get("target_context_2", "(not set)"),
            "intention": first.get("intention", "(not set)"),
            "sentence": "",
            "confidence": first.get("confidence", 0.0),
        }

    # Extract rationale (cot preset)
    rationale_raw = parsed.get("rationale", {})
    if not isinstance(rationale_raw, dict):
        rationale_raw = {}
    rationale: Dict[str, str] = {}
    if effective_preset == "cot":
        for c in criteria:
            val = rationale_raw.get(c.lower(), rationale_raw.get(c, ""))
            rationale[c.lower()] = str(val) if val else ""

    # Derive summary from checklist/pred
    pred_parts = [f"{c}={cvs_predictions.get(c, 0.0):.2f}" for c in criteria]
    summary = str(parsed.get("summary", "")).strip()
    if not summary:
        summary = f"CVS scores: {', '.join(pred_parts)}. {len(actions_ranked)} action(s) predicted."

    output: Dict[str, Any] = {
        "tool": "scene_cvs_act_analyzer",
        "preset": effective_preset,
        "interval": interval,
        "reason": reason,
        "cvs_predictions": cvs_predictions,
        "checklist": checklist,
        "actions_ranked": actions_ranked,
        "recommended_action": recommended_action,
        "summary": summary,
        "frames": frames,
    }
    if effective_preset == "cot" and rationale:
        output["rationale"] = rationale
    memory.record_result(tool="scene_cvs_act_analyzer", interval=interval, output=output, iteration=iteration)
    return output


def _get_latest_cvs_from_memory(memory: HM3Memory) -> Dict[str, Any]:
    """Extract latest CVS results from memory for use in action_rec.

    Checks for both ``scene_cvs_analyzer`` (CVS-only) and
    ``scene_cvs_act_analyzer`` (combined CVS+action) tool names.
    """
    _CVS_TOOL_NAMES = {"scene_cvs_analyzer", "scene_cvs_act_analyzer"}
    for event in reversed(memory.results.events):
        if event.tool in _CVS_TOOL_NAMES:
            output = event.output if isinstance(event.output, dict) else {}
            return {
                "cvs_predictions": output.get("cvs_predictions", {}),
                "checklist": output.get("checklist", {}),
                "actions_ranked": output.get("actions_ranked", []),
                "summary": output.get("summary", ""),
            }
    return {}


def _load_action_rec_frames(memory: HM3Memory, max_context: int = 5) -> List[Dict[str, Any]]:
    """Load current frame + up to max_context previous frames for action recommendation."""
    frames: List[Dict[str, Any]] = []
    if not (memory.metadata and memory.metadata.frames):
        return frames
    current_idx = memory.visible_end_idx
    if current_idx is None:
        current_idx = memory.metadata.num_frames - 1
    current_idx = max(0, min(int(current_idx), len(memory.metadata.frames) - 1))
    visible_start = max(0, int(memory.visible_start_idx or 0))
    context_start = max(visible_start, current_idx - max_context)
    for i in range(context_start, current_idx):
        ctx_frame = memory.metadata.frames[i]
        frames.append({"type": "frame", "frame_id": str(ctx_frame.frame_id),
                        "path": str(ctx_frame.path), "idx": int(ctx_frame.idx)})
    current_frame = memory.metadata.frames[current_idx]
    frames.append({"type": "frame", "frame_id": str(current_frame.frame_id),
                    "path": str(current_frame.path), "idx": int(current_frame.idx)})
    return frames


def _build_cvs_status_block(memory: HM3Memory, criteria: List[str]) -> str:
    """Build explicit score+rationale lines from the latest scene_cvs_analyzer results."""
    cvs_context = _get_latest_cvs_from_memory(memory)
    cvs_preds = cvs_context.get("cvs_predictions", {})
    rationale = cvs_context.get("rationale", {})
    lines = []
    for c in criteria:
        score = cvs_preds.get(c, cvs_preds.get(c.lower(), None))
        rationale_text = str(rationale.get(c.lower(), rationale.get(c, "")) or "").strip() or "no rationale available"
        if score is None:
            lines.append(f"  {c}: not yet assessed -- {rationale_text}")
            continue
        try:
            score_f = float(score)
            lines.append(f"  {c}: {score_f:.2f} -- {rationale_text}")
        except (TypeError, ValueError):
            lines.append(f"  {c}: unknown -- {rationale_text}")
    return "\n".join(lines) if lines else "  No CVS scores available yet."


def _get_all_cvs_from_memory(memory: HM3Memory) -> List[Dict[str, Any]]:
    """Extract ALL CVS results from memory (oldest-first) for full-history mode."""
    _CVS_TOOL_NAMES = {"scene_cvs_analyzer", "scene_cvs_act_analyzer"}
    results = []
    for event in memory.results.events:
        if event.tool in _CVS_TOOL_NAMES:
            output = event.output if isinstance(event.output, dict) else {}
            results.append({
                "iteration": event.iteration,
                "cvs_predictions": output.get("cvs_predictions", {}),
                "checklist": output.get("checklist", {}),
                "actions_ranked": output.get("actions_ranked", []),
                "summary": output.get("summary", ""),
            })
    return results


def _get_video_cvs_summary(memory: HM3Memory, criteria: List[str]) -> str:
    """Build a human-readable block from the video-level running CVS summary."""
    vcs = memory.video_criteria_status
    if not vcs:
        return "  No video-level CVS summary available yet."
    lines = []
    for c in criteria:
        entry = vcs.get(c, vcs.get(c.upper(), {}))
        if not isinstance(entry, dict):
            lines.append(f"  {c}: not yet assessed")
            continue
        score = entry.get("score")
        reason = entry.get("reason", "")
        if score is not None:
            try:
                score_f = float(score)
                label = "satisfied" if score_f >= 0.7 else ("unsatisfied" if score_f <= 0.3 else "uncertain")
                line = f"  {c}: {score_f:.2f} ({label})"
                if reason:
                    line += f" — {reason[:150]}"
                lines.append(line)
            except (TypeError, ValueError):
                lines.append(f"  {c}: unknown")
        else:
            lines.append(f"  {c}: not yet assessed")
    return "\n".join(lines) if lines else "  No video-level CVS summary available yet."


def _build_full_cvs_history_block(memory: HM3Memory, criteria: List[str]) -> str:
    """Build a full CVS history block showing all past assessments."""
    all_cvs = _get_all_cvs_from_memory(memory)
    if not all_cvs:
        return ""
    lines = ["CVS assessment history (oldest to newest):"]
    for i, entry in enumerate(all_cvs):
        iter_num = entry.get("iteration", i)
        preds = entry.get("cvs_predictions", {})
        summary = entry.get("summary", "")
        checklist = entry.get("checklist", {})
        score_parts = []
        for c in criteria:
            s = preds.get(c, preds.get(c.lower()))
            if s is not None:
                try:
                    score_parts.append(f"{c}={float(s):.2f}")
                except (TypeError, ValueError):
                    score_parts.append(f"{c}=?")
        lines.append(f"  [{iter_num}] {', '.join(score_parts)}")
        if summary:
            lines.append(f"       Summary: {summary[:200]}")
        if isinstance(checklist, dict) and checklist:
            for c in criteria:
                items = checklist.get(c, [])
                if isinstance(items, list) and items:
                    for item in items:
                        if isinstance(item, dict):
                            q = item.get("question", "")
                            a = item.get("answer", "uncertain")
                            lines.append(f"       {c}: {q} → {a}")
    return "\n".join(lines) + "\n"


def _parse_criteria_list(criterion: str) -> List[str]:
    """Parse a comma/space-separated criterion string into a list of valid criteria."""
    import re as _re
    criteria = [
        token
        for token in [x.strip().upper() for x in _re.split(r"[,\s]+", str(criterion or "")) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    if not criteria:
        criteria = list(DEFAULT_CVS_CRITERIA)
    return criteria


@tool
def left_rec(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    taxonomy_catalog: Dict[str, Any],
    criterion: str,
) -> Dict[str, Any]:
    """
    Left Instrument Recommender. Generates the next recommended action for the
    LEFT instrument (almost always a Grasper used for retraction/exposure).

    In cholecystectomy, the left instrument almost always retracts the gallbladder
    to create exposure. The key question is WHICH DIRECTION to retract.

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      taxonomy_catalog: dict (runtime)
      criterion: str (runtime)

    Output (dict):
      {
        "tool": "left_rec",
        "interval": {...},
        "reason": str,
        "reasoning": str,
        "action": {actor_role, tool_type, action_code, ...},
        "summary": str
      }
    """
    criteria = _parse_criteria_list(criterion)
    frames = _load_action_rec_frames(memory)

    default_output: Dict[str, Any] = {
        "tool": "left_rec",
        "interval": interval,
        "reason": reason,
        "reasoning": "",
        "action": {},
        "summary": "no analysis performed",
    }

    if model is None or not frames:
        if not frames:
            default_output["summary"] = "no frames available from metadata"
        memory.record_result(tool="left_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    images = _load_scene_images(frames)
    if not images:
        default_output["summary"] = "no loadable images from metadata frames"
        memory.record_result(tool="left_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    current_frame_id = frames[-1].get("frame_id", "")
    num_frames = len(frames)
    cvs_status_block = _build_cvs_status_block(memory, criteria)
    cvs_summary = _get_latest_cvs_from_memory(memory).get("summary", "")
    audit_simple_catalog = augment_taxonomy_catalog_for_audit_v11_simple(taxonomy_catalog)
    taxonomy_block = _taxonomy_options_block(audit_simple_catalog)
    audit_simple_block = build_audit_v11_simple_prompt_block()

    system_prompt = (
        "You are a surgical vision assistant specializing in laparoscopic cholecystectomy. "
        "Follow the output format exactly."
    )

    if num_frames == 1:
        frame_intro = (
            "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
            f"This is frame {current_frame_id} — the CURRENT evaluation frame.\n"
        )
    else:
        frame_intro = (
            f"You are viewing {num_frames} frames from a laparoscopic cholecystectomy video.\n"
            f"Frame {current_frame_id} is the CURRENT evaluation frame (last image). "
            f"Earlier frames provide temporal context for how the scene evolved.\n"
        )

    user_prompt = (
        f"{frame_intro}"
        "\n"
        "You are recommending the NEXT action for the LEFT instrument (almost always a Grasper).\n"
        "\n"
        "In cholecystectomy, the left instrument almost always retracts the gallbladder to create\n"
        "exposure. The key question is WHICH DIRECTION to retract:\n"
        "  - RETRACT_LATERAL: Pull gallbladder leftward (from camera perspective) to expose medial HCT\n"
        "  - RETRACT_MEDIAL: Pull rightward to expose lateral aspect\n"
        "  - RETRACT_UPWARD: Pull cephalad to lift gallbladder off liver bed\n"
        "  - RETRACT_MEDIAL_TO_LATERAL / RETRACT_LATERAL_TO_MEDIAL: Rotation between positions\n"
        "  - RETRACT_MAINTAIN: Current retraction is adequate, maintain position\n"
        "  - RETRACT_DOWNWARD: Rare, pull inferiorly\n"
        "\n"
        "If the gallbladder is already well-retracted and the hepatocystic triangle is well-exposed\n"
        "for the target criterion, recommend RETRACT_MAINTAIN.\n"
        "If exposure needs to change, recommend the direction that would best serve the\n"
        "unsatisfied CVS criteria.\n"
        "\n"
        "CVS criterion definitions:\n"
        f"{_cvs_definition_block()}\n"
        "\n"
        f"Current CVS status:\n{cvs_status_block}\n"
    )
    if cvs_summary:
        user_prompt += f"Scene summary: {cvs_summary}\n"

    user_prompt += (
        "\n"
        "Allowed taxonomy values:\n"
        f"{taxonomy_block}\n"
        "\n"
        "IMPORTANT:\n"
        "- actor_role MUST be \"left_instrument\".\n"
        "- tool_type MUST be \"Grasper\".\n"
        "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
        "- Return exactly 1 action.\n"
        "\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        "{\n"
        '  "reasoning": "<why this retraction direction>",\n'
        '  "action": {\n'
        '    "actor_role": "left_instrument",\n'
        '    "tool_type": "Grasper",\n'
        '    "action_code": "<RETRACT_* value from allowed action_code>",\n'
        '    "target_structure": "<from allowed target_structure>",\n'
        '    "target_context_1": "<from allowed target_context or (not set)>",\n'
        '    "target_context_2": "<(not set)>",\n'
        '    "intention": "<from allowed intention>",\n'
        '    "confidence": <float 0..1>\n'
        "  }\n"
        "}\n"
    )

    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    text = (response.content or "").strip()
    parsed = _extract_json_object(text)

    reasoning = str(parsed.get("reasoning", "")).strip()
    action_raw = parsed.get("action", {})
    if not isinstance(action_raw, dict):
        action_raw = {}

    action: Dict[str, Any] = {}
    if action_raw:
        action = {
            "actor_role": str(action_raw.get("actor_role", "left_instrument")),
            "tool_type": str(action_raw.get("tool_type", "Grasper")),
            "action_code": str(action_raw.get("action_code", "(not set)")),
            "target_structure": str(action_raw.get("target_structure", "(not set)")),
            "target_context_1": str(action_raw.get("target_context_1", "(not set)")),
            "target_context_2": str(action_raw.get("target_context_2", "(not set)")),
            "intention": str(action_raw.get("intention", "(not set)")),
            "confidence": _safe_float(action_raw.get("confidence", 0.0)),
        }

    summary = f"Left rec: {action.get('action_code', '(none)')}"
    if reasoning:
        summary = f"{reasoning[:200]}. {summary}"

    output: Dict[str, Any] = {
        "tool": "left_rec",
        "interval": interval,
        "reason": reason,
        "reasoning": reasoning,
        "action": action,
        "summary": summary,
        "frames": frames,
    }
    memory.record_result(tool="left_rec", interval=interval, output=output, iteration=iteration)
    return output


@tool
def right_rec(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    taxonomy_catalog: Dict[str, Any],
    criterion: str,
) -> Dict[str, Any]:
    """
    Right Instrument Recommender. Generates the next recommended action for the
    RIGHT instrument (active/working tool: Hook, Maryland, Scissors, Irrigator).

    The right instrument performs the active surgical work: dissection, cutting,
    cautery, aspiration. Target depends on which CVS criterion needs most
    improvement.

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      taxonomy_catalog: dict (runtime)
      criterion: str (runtime)

    Output (dict):
      {
        "tool": "right_rec",
        "interval": {...},
        "reason": str,
        "reasoning": str,
        "action": {actor_role, tool_type, action_code, ...},
        "summary": str
      }
    """
    criteria = _parse_criteria_list(criterion)
    frames = _load_action_rec_frames(memory)

    default_output: Dict[str, Any] = {
        "tool": "right_rec",
        "interval": interval,
        "reason": reason,
        "reasoning": "",
        "action": {},
        "summary": "no analysis performed",
    }

    if model is None or not frames:
        if not frames:
            default_output["summary"] = "no frames available from metadata"
        memory.record_result(tool="right_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    images = _load_scene_images(frames)
    if not images:
        default_output["summary"] = "no loadable images from metadata frames"
        memory.record_result(tool="right_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    current_frame_id = frames[-1].get("frame_id", "")
    num_frames = len(frames)
    cvs_status_block = _build_cvs_status_block(memory, criteria)
    cvs_summary = _get_latest_cvs_from_memory(memory).get("summary", "")
    audit_simple_catalog = augment_taxonomy_catalog_for_audit_v11_simple(taxonomy_catalog)
    taxonomy_block = _taxonomy_options_block(audit_simple_catalog)
    audit_simple_block = build_audit_v11_simple_prompt_block()

    system_prompt = (
        "You are a surgical vision assistant specializing in laparoscopic cholecystectomy. "
        "Follow the output format exactly."
    )

    if num_frames == 1:
        frame_intro = (
            "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
            f"This is frame {current_frame_id} — the CURRENT evaluation frame.\n"
        )
    else:
        frame_intro = (
            f"You are viewing {num_frames} frames from a laparoscopic cholecystectomy video.\n"
            f"Frame {current_frame_id} is the CURRENT evaluation frame (last image). "
            f"Earlier frames provide temporal context for how the scene evolved.\n"
        )

    user_prompt = (
        f"{frame_intro}"
        "\n"
        "You are recommending the NEXT action for the RIGHT instrument (active/working tool).\n"
        "\n"
        "The right instrument performs the active surgical work: dissection, cutting, cautery,\n"
        "aspiration. Target depends on which CVS criterion needs most improvement:\n"
        "  - C1 (two structures) / C2 (triangle clearance) → dissect in hepatocystic triangle area.\n"
        "    Specify precise target_context (between cystic duct and artery, near artery, near duct, etc.)\n"
        "  - C3 (lower third detachment) → dissect cystic plate / gallbladder-liver interface.\n"
        "\n"
        "If no active work is visible or needed, you may return action_code=NO_ACTION.\n"
        "\n"
        "CVS criterion definitions:\n"
        f"{_cvs_definition_block()}\n"
        "\n"
        f"Current CVS status:\n{cvs_status_block}\n"
    )
    if cvs_summary:
        user_prompt += f"Scene summary: {cvs_summary}\n"

    user_prompt += (
        "\n"
        "Allowed taxonomy values:\n"
        f"{taxonomy_block}\n"
        "\n"
        "IMPORTANT:\n"
        "- actor_role MUST be \"right_instrument\".\n"
        "- tool_type from: Hook, Maryland, Scissors, Irrigator, Unknown.\n"
        "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
        "- Return exactly 1 action.\n"
        "\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        "{\n"
        '  "reasoning": "<what needs dissection and why>",\n'
        '  "action": {\n'
        '    "actor_role": "right_instrument",\n'
        '    "tool_type": "<Hook|Maryland|Scissors|Irrigator|Unknown>",\n'
        '    "action_code": "<DISSECT|COAGULATE_HEMOSTASIS|IRRIGATOR_ASPIRATE|NO_ACTION|... from allowed action_code>",\n'
        '    "target_structure": "<from allowed target_structure>",\n'
        '    "target_context_1": "<from allowed target_context or (not set)>",\n'
        '    "target_context_2": "<from allowed target_context or (not set)>",\n'
        '    "intention": "<from allowed intention>",\n'
        '    "confidence": <float 0..1>\n'
        "  }\n"
        "}\n"
    )

    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    text = (response.content or "").strip()
    parsed = _extract_json_object(text)

    reasoning = str(parsed.get("reasoning", "")).strip()
    action_raw = parsed.get("action", {})
    if not isinstance(action_raw, dict):
        action_raw = {}

    action: Dict[str, Any] = {}
    if action_raw:
        action = {
            "actor_role": str(action_raw.get("actor_role", "right_instrument")),
            "tool_type": str(action_raw.get("tool_type", "Unknown")),
            "action_code": str(action_raw.get("action_code", "(not set)")),
            "target_structure": str(action_raw.get("target_structure", "(not set)")),
            "target_context_1": str(action_raw.get("target_context_1", "(not set)")),
            "target_context_2": str(action_raw.get("target_context_2", "(not set)")),
            "intention": str(action_raw.get("intention", "(not set)")),
            "confidence": _safe_float(action_raw.get("confidence", 0.0)),
        }

    summary = f"Right rec: {action.get('action_code', '(none)')}"
    if reasoning:
        summary = f"{reasoning[:200]}. {summary}"

    output: Dict[str, Any] = {
        "tool": "right_rec",
        "interval": interval,
        "reason": reason,
        "reasoning": reasoning,
        "action": action,
        "summary": summary,
        "frames": frames,
    }
    memory.record_result(tool="right_rec", interval=interval, output=output, iteration=iteration)
    return output


@tool
def camera_rec(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    taxonomy_catalog: Dict[str, Any],
    criterion: str,
) -> Dict[str, Any]:
    """
    Camera Recommender. Generates the next recommended action for the CAMERA.

    Decision tree:
      - Field of view too far / structures small → CAMERA_ZOOM_IN
      - View too close / can't see whole field → CAMERA_ZOOM_OUT
      - Hepatocystic triangle not centered/visible → CAMERA_REPOSITION
      - View is adequate → NO_ACTION (recommend no camera change)

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      taxonomy_catalog: dict (runtime)
      criterion: str (runtime)

    Output (dict):
      {
        "tool": "camera_rec",
        "interval": {...},
        "reason": str,
        "reasoning": str,
        "action": {actor_role, tool_type, action_code, ...},
        "summary": str
      }
    """
    criteria = _parse_criteria_list(criterion)
    frames = _load_action_rec_frames(memory)

    default_output: Dict[str, Any] = {
        "tool": "camera_rec",
        "interval": interval,
        "reason": reason,
        "reasoning": "",
        "action": {},
        "summary": "no analysis performed",
    }

    if model is None or not frames:
        if not frames:
            default_output["summary"] = "no frames available from metadata"
        memory.record_result(tool="camera_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    images = _load_scene_images(frames)
    if not images:
        default_output["summary"] = "no loadable images from metadata frames"
        memory.record_result(tool="camera_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    current_frame_id = frames[-1].get("frame_id", "")
    num_frames = len(frames)
    cvs_status_block = _build_cvs_status_block(memory, criteria)
    cvs_summary = _get_latest_cvs_from_memory(memory).get("summary", "")
    audit_simple_catalog = augment_taxonomy_catalog_for_audit_v11_simple(taxonomy_catalog)
    taxonomy_block = _taxonomy_options_block(audit_simple_catalog)
    audit_simple_block = build_audit_v11_simple_prompt_block()

    system_prompt = (
        "You are a surgical vision assistant specializing in laparoscopic cholecystectomy. "
        "Follow the output format exactly."
    )

    if num_frames == 1:
        frame_intro = (
            "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
            f"This is frame {current_frame_id} — the CURRENT evaluation frame.\n"
        )
    else:
        frame_intro = (
            f"You are viewing {num_frames} frames from a laparoscopic cholecystectomy video.\n"
            f"Frame {current_frame_id} is the CURRENT evaluation frame (last image). "
            f"Earlier frames provide temporal context for how the scene evolved.\n"
        )

    user_prompt = (
        f"{frame_intro}"
        "\n"
        "You are recommending the NEXT action for the CAMERA.\n"
        "\n"
        "Decision tree:\n"
        "  - Field of view too far / structures appear small → CAMERA_ZOOM_IN\n"
        "  - View too close / cannot see the whole operative field → CAMERA_ZOOM_OUT\n"
        "  - Hepatocystic triangle not centered or not visible → CAMERA_REPOSITION\n"
        "  - View is adequate for assessing CVS criteria → NO_ACTION (no camera change needed)\n"
        "\n"
        "If the current camera position provides adequate visualization for assessing CVS criteria,\n"
        "return action_code=NO_ACTION.\n"
        "\n"
        "CVS criterion definitions:\n"
        f"{_cvs_definition_block()}\n"
        "\n"
        f"Current CVS status:\n{cvs_status_block}\n"
    )
    if cvs_summary:
        user_prompt += f"Scene summary: {cvs_summary}\n"

    user_prompt += (
        "\n"
        "Allowed taxonomy values:\n"
        f"{taxonomy_block}\n"
        "\n"
        "IMPORTANT:\n"
        "- actor_role MUST be \"camera\".\n"
        "- tool_type MUST be \"camera\".\n"
        "- action_code must be one of: CAMERA_ZOOM_IN, CAMERA_ZOOM_OUT, CAMERA_REPOSITION, NO_ACTION.\n"
        "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
        "- Return exactly 1 action.\n"
        "\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        "{\n"
        '  "reasoning": "<why camera should/should not move>",\n'
        '  "action": {\n'
        '    "actor_role": "camera",\n'
        '    "tool_type": "camera",\n'
        '    "action_code": "<CAMERA_ZOOM_IN|CAMERA_ZOOM_OUT|CAMERA_REPOSITION|NO_ACTION>",\n'
        '    "target_structure": "<from allowed target_structure or (not set)>",\n'
        '    "target_context_1": "<(not set)>",\n'
        '    "target_context_2": "<(not set)>",\n'
        '    "intention": "<from allowed intention>",\n'
        '    "confidence": <float 0..1>\n'
        "  }\n"
        "}\n"
    )

    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    text = (response.content or "").strip()
    parsed = _extract_json_object(text)

    reasoning = str(parsed.get("reasoning", "")).strip()
    action_raw = parsed.get("action", {})
    if not isinstance(action_raw, dict):
        action_raw = {}

    action: Dict[str, Any] = {}
    if action_raw:
        action = {
            "actor_role": str(action_raw.get("actor_role", "camera")),
            "tool_type": str(action_raw.get("tool_type", "camera")),
            "action_code": str(action_raw.get("action_code", "NO_ACTION")),
            "target_structure": str(action_raw.get("target_structure", "(not set)")),
            "target_context_1": str(action_raw.get("target_context_1", "(not set)")),
            "target_context_2": str(action_raw.get("target_context_2", "(not set)")),
            "intention": str(action_raw.get("intention", "(not set)")),
            "confidence": _safe_float(action_raw.get("confidence", 0.0)),
        }

    summary = f"Camera rec: {action.get('action_code', '(none)')}"
    if reasoning:
        summary = f"{reasoning[:200]}. {summary}"

    output: Dict[str, Any] = {
        "tool": "camera_rec",
        "interval": interval,
        "reason": reason,
        "reasoning": reasoning,
        "action": action,
        "summary": summary,
        "frames": frames,
    }
    memory.record_result(tool="camera_rec", interval=interval, output=output, iteration=iteration)
    return output


@tool
def action_rec_old(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    taxonomy_catalog: Dict[str, Any],
    criterion: str,
    fixed_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Action Recommender (old, standalone). Generates specific, surgeon-facing
    next-action recommendations based on the current surgical scene and CVS
    criterion status. Loads the current evaluation frame directly from metadata
    (visible_end_idx) plus up to 5 previous frames for temporal context,
    and uses any existing CVS predictions from memory.

    Unlike scene_cvs_analyzer (which predicts what actions ARE happening),
    this tool reasons about what actions SHOULD happen next to improve CVS
    criteria. It incorporates surgical heuristics:

    LEFT instrument (usually Grasper):
      - Retraction direction matters: left-of-camera = lateral/anterior,
        right-of-camera = medial/posterior, sometimes upward.
      - Should decide if current retraction serves the target criterion;
        if not, recommend rotating direction.

    RIGHT instrument (active/working — Hook, Maryland, Scissors, Irrigator):
      - Usually dissecting. Target depends on criterion:
        C1/C2 → dissect hepatocystic triangle area
        C3 → dissect cystic plate / gallbladder-liver interface
      - When cystic duct and artery are partially visible, specify
        precise target_context (between duct and artery, etc).

    Camera:
      - Too far → suggest zoom in
      - Too close / can't see whole field → suggest zoom out
      - Can't see hepatocystic triangle clearly → reposition camera

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      taxonomy_catalog: dict (runtime) - loaded taxonomy with options
      criterion: str (runtime) - active criteria e.g. "C1,C2,C3"

    Output (dict):
      {
        "tool": "action_rec",
        "interval": {...},
        "reason": str,
        "actions_ranked": [{rank, actor_role, tool_type, action_code, ...}, ...],
        "reasoning": str,
        "summary": str
      }
    """
    import re as _re
    criteria = [
        token
        for token in [x.strip().upper() for x in _re.split(r"[,\s]+", str(criterion or "")) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    if not criteria:
        criteria = list(DEFAULT_CVS_CRITERIA)

    # Load individual frames: current frame + up to 5 previous frames
    frames: List[Dict[str, Any]] = []

    if memory.metadata and memory.metadata.frames:
        current_idx = memory.visible_end_idx
        if current_idx is None:
            current_idx = memory.metadata.num_frames - 1
        current_idx = max(0, min(int(current_idx), len(memory.metadata.frames) - 1))

        # Include up to 5 previous frames for temporal context (oldest first)
        visible_start = max(0, int(memory.visible_start_idx or 0))
        context_start = max(visible_start, current_idx - 5)
        for i in range(context_start, current_idx):
            ctx_frame = memory.metadata.frames[i]
            frames.append({
                "type": "frame",
                "frame_id": str(ctx_frame.frame_id),
                "path": str(ctx_frame.path),
                "idx": int(ctx_frame.idx),
            })

        # Current frame last
        current_frame = memory.metadata.frames[current_idx]
        frames.append({
            "type": "frame",
            "frame_id": str(current_frame.frame_id),
            "path": str(current_frame.path),
            "idx": int(current_frame.idx),
        })

    default_output: Dict[str, Any] = {
        "tool": "action_rec_old",
        "interval": interval,
        "reason": reason,
        "actions_ranked": [],
        "reasoning": "",
        "summary": "no analysis performed",
    }

    if model is None or not frames:
        if not frames:
            default_output["summary"] = "no frames available from metadata"
        memory.record_result(tool="action_rec_old", interval=interval, output=default_output, iteration=iteration)
        return default_output

    images = _load_scene_images(frames)
    if not images:
        default_output["summary"] = "no loadable images from metadata frames"
        memory.record_result(tool="action_rec_old", interval=interval, output=default_output, iteration=iteration)
        return default_output

    # Current frame is always frames[-1]
    current_frame_id = frames[-1].get("frame_id", "")
    num_frames = len(frames)

    # Get existing CVS predictions from memory for context
    cvs_context = _get_latest_cvs_from_memory(memory)
    cvs_preds = cvs_context.get("cvs_predictions", {})
    cvs_summary = cvs_context.get("summary", "")

    # Build CVS status block for the prompt
    cvs_status_lines = []
    for c in criteria:
        score = cvs_preds.get(c, cvs_preds.get(c.lower(), None))
        if score is not None:
            try:
                score_f = float(score)
                label = "satisfied" if score_f >= 0.7 else ("unsatisfied" if score_f <= 0.3 else "uncertain")
                cvs_status_lines.append(f"  {c}: {score_f:.2f} ({label})")
            except (TypeError, ValueError):
                cvs_status_lines.append(f"  {c}: unknown")
        else:
            cvs_status_lines.append(f"  {c}: not yet assessed")
    cvs_status_block = "\n".join(cvs_status_lines) if cvs_status_lines else "  No CVS scores available yet."

    taxonomy_block = _taxonomy_options_block(taxonomy_catalog)

    system_prompt = (
        "You are a surgical vision assistant specializing in laparoscopic cholecystectomy. "
        "Follow the output format exactly."
    )

    if num_frames == 1:
        frame_intro = (
            "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
            f"This is frame {current_frame_id} — the CURRENT evaluation frame.\n"
        )
    else:
        frame_intro = (
            f"You are viewing {num_frames} frames from a laparoscopic cholecystectomy video.\n"
            f"Frame {current_frame_id} is the CURRENT evaluation frame (last image). "
            f"Earlier frames provide temporal context for how the scene evolved.\n"
        )

    user_prompt = (
        f"{frame_intro}"
        "\n"
        "Your task: Recommend the NEXT surgical actions that should be performed to\n"
        "improve the Critical View of Safety (CVS) criteria.\n"
        "\n"
        "CVS criterion definitions:\n"
        f"{_cvs_definition_block()}\n"
        "\n"
        f"Current CVS status:\n{cvs_status_block}\n"
    )
    if cvs_summary:
        user_prompt += f"Scene summary: {cvs_summary}\n"

    user_prompt += (
        "\n"
        "SURGICAL REASONING GUIDELINES:\n"
        "\n"
        "LEFT instrument (almost always a Grasper for retraction/exposure):\n"
        "- Retraction direction relative to camera view matters:\n"
        "  * Pulling LEFT of camera = lateral/anterior retraction\n"
        "  * Pulling RIGHT of camera = medial/posterior retraction\n"
        "  * Pulling UPWARD = cephalad retraction\n"
        "- Evaluate if current retraction direction serves the target criterion.\n"
        "  If the current exposure is adequate for the criterion being worked on, recommend\n"
        "  maintaining current retraction. If not, recommend rotating direction\n"
        "  (e.g., RETRACT_LATERALLY to RETRACT_MEDIALLY or vice versa).\n"
        "- If retraction is steady and adequate, use RETRACT_MAINTAIN.\n"
        "\n"
        "RIGHT instrument (active/working tool — usually Hook, Maryland, Scissors, or Irrigator):\n"
        "- The right instrument does the active work (dissection, cutting, cautery).\n"
        "- Target depends on which criterion needs improvement:\n"
        "  * C1 (two structures) / C2 (triangle clearance) → dissect in the hepatocystic triangle.\n"
        "    Specify target_structure precisely (e.g., peritoneum, adhesion, fat_tissue).\n"
        "    When cystic duct and cystic artery are partially visible, use target_context\n"
        "    to specify the location (between cystic duct and artery, lateral to cystic duct, etc).\n"
        "  * C3 (lower third detachment) → dissect the cystic plate / gallbladder-liver interface.\n"
        "    Target the plane between gallbladder wall and liver bed.\n"
        "\n"
        "CAMERA:\n"
        "- If the field of view is too far / structures are small → recommend ZOOM_IN.\n"
        "- If the view is too close / cannot see the whole operative field → recommend ZOOM_OUT.\n"
        "- If the hepatocystic triangle is not centered or visible → recommend REPOSITION camera.\n"
        "\n"
        "CRITICAL — INSTRUMENT IDENTIFICATION FIRST:\n"
        "Before recommending any action, carefully examine the frame to identify which\n"
        "instruments are actually present and visible. The tool_type in each recommendation\n"
        "MUST match the instrument you observe in the image. Common right instruments:\n"
        "  - Hook (cautery hook): thin L-shaped tip\n"
        "  - Maryland (Maryland dissector): long, thin, curved jaw tips\n"
        "  - Scissors: two blades\n"
        "  - Irrigator: tube-like\n"
        "Do NOT default to Hook — if you see Maryland (curved jaw tips), use Maryland.\n"
        "Recommend actions that the CURRENTLY VISIBLE instrument can perform.\n"
        "\n"
        "PRIORITIZATION:\n"
        "- Address all unsatisfied criteria, not just the lowest-scoring one.\n"
        "- Each action's intention should reflect which specific criterion it targets.\n"
        "- Include BOTH retraction and dissection actions if both are needed.\n"
        "- Rank actions by importance: the action most likely to advance any unsatisfied criterion first.\n"
        "\n"
        "Allowed taxonomy values:\n"
        f"{taxonomy_block}\n"
        "\n"
        "IMPORTANT:\n"
        "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
        "- actor_role must match the tool side: left_instrument for grasper/retraction,\n"
        "  right_instrument for active dissection/cutting.\n"
        '- If unsure of actor, set actor_role="unknown".\n'
        + (f"- Return exactly {fixed_k} ranked recommended actions.\n" if fixed_k is not None
           else "- Return 1-3 ranked recommended actions.\n")
        + ("\n" + _fixed_k_constraint_block(fixed_k) if fixed_k is not None else "")
        + "\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        "{\n"
        '  "reasoning": "<2-3 sentences: what criterion needs most improvement, what the current scene shows, and what should change>",\n'
        '  "actions_ranked": [\n'
        + (f"    // exactly {fixed_k} action objects, ranked 1 to {fixed_k}\n" if fixed_k is not None else "")
        + "    {\n"
        '      "rank": 1,\n'
        '      "actor_role": "<from allowed actor_role>",\n'
        '      "tool_type": "<from allowed tool_type>",\n'
        '      "action_code": "<from allowed action_code>",\n'
        '      "target_structure": "<from allowed target_structure>",\n'
        '      "target_context_1": "<from allowed target_context or (not set)>",\n'
        '      "target_context_2": "<from allowed target_context or (not set)>",\n'
        '      "intention": "<from allowed intention>",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": user_prompt}
    ]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    text = (response.content or "").strip()
    parsed = _parse_scene_cvs_output(text)

    # Extract reasoning
    reasoning = str(parsed.get("reasoning", "")).strip()

    # Extract actions_ranked
    actions_raw = parsed.get("actions", parsed.get("actions_ranked", []))
    if not isinstance(actions_raw, list):
        actions_raw = []
    actions_ranked: List[Dict[str, Any]] = []
    for a in actions_raw:
        if not isinstance(a, dict):
            continue
        actions_ranked.append({
            "rank": a.get("rank"),
            "actor_role": str(a.get("actor_role", "(not set)")),
            "tool_type": str(a.get("tool_type", "(not set)")),
            "action_code": str(a.get("action_code", "(not set)")),
            "target_structure": str(a.get("target_structure", "(not set)")),
            "target_context_1": str(a.get("target_context_1", "(not set)")),
            "target_context_2": str(a.get("target_context_2", "(not set)")),
            "intention": str(a.get("intention", "(not set)")),
            "confidence": _safe_float(a.get("confidence", 0.0)),
        })

    # Enforce fixed_k if set
    if fixed_k is not None:
        actions_ranked = _enforce_fixed_k(actions_ranked, fixed_k)

    summary = f"{len(actions_ranked)} action(s) recommended."
    if reasoning:
        summary = f"{reasoning[:200]}. {summary}"

    output: Dict[str, Any] = {
        "tool": "action_rec_old",
        "interval": interval,
        "reason": reason,
        "actions_ranked": actions_ranked,
        "reasoning": reasoning,
        "summary": summary,
        "frames": frames,
    }
    memory.record_result(tool="action_rec_old", interval=interval, output=output, iteration=iteration)
    return output


@tool
def scene_cvs_analyzer(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    taxonomy_catalog: Dict[str, Any],
    criterion: str,
    preset: str = "direct",
) -> Dict[str, Any]:
    """
    Scene CVS Analyzer (CVS-only). Frame-level CVS criterion prediction tool.
    The model generates its own diagnostic checklist and predicts CVS scores.
    Does NOT predict or recommend actions — use action_rec for that.

    Loads the current evaluation frame directly from metadata (visible_end_idx)
    and optionally includes P_s frames from clip_explorer. Produces structured
    CVS predictions with per-criterion confidence scores and a self-generated
    checklist.

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      taxonomy_catalog: dict (runtime) - loaded taxonomy with options
      criterion: str (runtime) - active criteria e.g. "C1,C2,C3"

    Output (dict):
      {
        "tool": "scene_cvs_analyzer",
        "interval": {...},
        "reason": str,
        "cvs_predictions": {"C1": float, "C2": float, "C3": float},
        "checklist": {"C1": [...], "C2": [...], "C3": [...]},
        "summary": str
      }
    """
    import re as _re
    criteria = [
        token
        for token in [x.strip().upper() for x in _re.split(r"[,\s]+", str(criterion or "")) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    if not criteria:
        criteria = list(DEFAULT_CVS_CRITERIA)

    # Load individual frames: current frame + optionally P_s frames from clip_explorer
    frames: List[Dict[str, Any]] = []

    # Always include current evaluation frame from metadata
    if memory.metadata and memory.metadata.frames:
        current_idx = memory.visible_end_idx
        if current_idx is None:
            current_idx = memory.metadata.num_frames - 1
        current_idx = max(0, min(int(current_idx), len(memory.metadata.frames) - 1))
        current_frame = memory.metadata.frames[current_idx]
        frames.append({
            "type": "frame",
            "frame_id": str(current_frame.frame_id),
            "path": str(current_frame.path),
            "idx": int(current_frame.idx),
        })

    # Optionally include P_s frames from clip_explorer (if available, excluding duplicates)
    current_paths = {f["path"] for f in frames}
    for item in memory.sensory.short_term_pool:
        if isinstance(item, dict) and item.get("type") == "frame" and item.get("path"):
            if item["path"] not in current_paths:
                frames.append(item)
                current_paths.add(item["path"])

    default_output: Dict[str, Any] = {
        "tool": "scene_cvs_analyzer",
        "interval": interval,
        "reason": reason,
        "cvs_predictions": {c: 0.0 for c in criteria},
        "checklist": {c: [] for c in criteria},
        "summary": "no analysis performed",
    }

    if model is None or not frames:
        if not frames:
            default_output["summary"] = "no frames available from metadata"
        memory.record_result(tool="scene_cvs_analyzer", interval=interval, output=default_output, iteration=iteration)
        return default_output

    images = _load_scene_images(frames)
    if not images:
        default_output["summary"] = "no loadable images from metadata frames"
        memory.record_result(tool="scene_cvs_analyzer", interval=interval, output=default_output, iteration=iteration)
        return default_output

    # Current frame is always frames[0]
    current_frame_id = frames[0].get("frame_id", "")
    num_frames = len(frames)

    system_prompt = (
        "You are a surgical vision assistant specializing in laparoscopic cholecystectomy. "
        "Follow the output format exactly."
    )

    effective_preset = preset if preset in ("direct", "cot", "subrubric") else "direct"
    user_prompt = build_scene_cvs_prompt_body(preset=effective_preset)

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": user_prompt}
    ]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    text = (response.content or "").strip()
    parsed = _parse_scene_cvs_output(text)

    # Extract checklist (self-generated)
    checklist_raw = parsed.get("checklist", {})
    if not isinstance(checklist_raw, dict):
        checklist_raw = {}
    checklist: Dict[str, List[Dict[str, str]]] = {}
    for c in criteria:
        items = checklist_raw.get(c, [])
        if isinstance(items, list):
            checklist[c] = [
                {
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "uncertain")),
                }
                for item in items
                if isinstance(item, dict)
            ]
        else:
            checklist[c] = []

    # Extract pred → cvs_predictions (normalize lowercase keys to uppercase)
    pred_raw = parsed.get("pred", {})
    if not isinstance(pred_raw, dict):
        pred_raw = {}
    cvs_predictions: Dict[str, float] = {}
    for c in criteria:
        val = pred_raw.get(c.lower(), pred_raw.get(c, 0.0))
        try:
            cvs_predictions[c] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            cvs_predictions[c] = 0.0

    # Extract rationale (cot preset)
    rationale_raw = parsed.get("rationale", {})
    if not isinstance(rationale_raw, dict):
        rationale_raw = {}
    rationale: Dict[str, str] = {}
    if effective_preset == "cot":
        for c in criteria:
            val = rationale_raw.get(c.lower(), rationale_raw.get(c, ""))
            rationale[c.lower()] = str(val) if val else ""

    # Derive summary
    pred_parts = [f"{c}={cvs_predictions.get(c, 0.0):.2f}" for c in criteria]
    summary = str(parsed.get("summary", "")).strip()
    if not summary:
        summary = f"CVS scores: {', '.join(pred_parts)}."

    output: Dict[str, Any] = {
        "tool": "scene_cvs_analyzer",
        "preset": effective_preset,
        "interval": interval,
        "reason": reason,
        "cvs_predictions": cvs_predictions,
        "checklist": checklist,
        "summary": summary,
        "frames": frames,
    }
    if effective_preset == "cot" and rationale:
        output["rationale"] = rationale
    memory.record_result(tool="scene_cvs_analyzer", interval=interval, output=output, iteration=iteration)
    return output


@tool
def action_rec(
    interval: Dict[str, Any],
    reason: str,
    *,
    memory: HM3Memory,
    model: Any,
    iteration: int,
    taxonomy_catalog: Dict[str, Any],
    criterion: str,
    fixed_k: Optional[int] = None,
    cvs_context_mode: str = "latest",
    one_per_actor: bool = False,
    action_rec_rules: str = "default",
) -> Dict[str, Any]:
    """
    Action Recommender (with CVS context). Generates specific, surgeon-facing
    next-action recommendations based on the current surgical scene and CVS
    criterion status from a prior scene_cvs_analyzer call.

    Loads the current evaluation frame directly from metadata (visible_end_idx)
    plus up to 5 previous frames for temporal context, and uses existing CVS
    predictions/checklist from memory to inform recommendations.

    Unlike scene_cvs_act_analyzer (which does both CVS + action in one call),
    this tool is designed to run AFTER scene_cvs_analyzer so it can benefit
    from CVS results already in memory. It incorporates surgical heuristics:

    LEFT instrument (usually Grasper):
      - Retraction direction matters: left-of-camera = lateral/anterior,
        right-of-camera = medial/posterior, sometimes upward.
      - Should decide if current retraction serves the target criterion;
        if not, recommend rotating direction.

    RIGHT instrument (active/working — Hook, Maryland, Scissors, Irrigator):
      - Usually dissecting. Target depends on criterion:
        C1/C2 → dissect hepatocystic triangle area
        C3 → dissect cystic plate / gallbladder-liver interface
      - When cystic duct and artery are partially visible, specify
        precise target_context (between duct and artery, etc).

    Camera:
      - Too far → suggest zoom in
      - Too close / can't see whole field → suggest zoom out
      - Can't see hepatocystic triangle clearly → reposition camera

    Input:
      interval: {"start": int, "end": int}
      reason: str
      memory: HM3Memory (runtime)
      model: LLM client (runtime)
      iteration: int (runtime)
      taxonomy_catalog: dict (runtime) - loaded taxonomy with options
      criterion: str (runtime) - active criteria e.g. "C1,C2,C3"
      fixed_k: int or None (runtime) - if set, produce exactly k actions

    Output (dict):
      {
        "tool": "action_rec",
        "interval": {...},
        "reason": str,
        "actions_ranked": [{rank, actor_role, tool_type, action_code, ...}, ...],
        "reasoning": str,
        "summary": str
      }
    """
    import re as _re
    criteria = [
        token
        for token in [x.strip().upper() for x in _re.split(r"[,\s]+", str(criterion or "")) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    if not criteria:
        criteria = list(DEFAULT_CVS_CRITERIA)

    # Load frames: current frame + up to 5 previous frames
    frames = _load_action_rec_frames(memory)

    default_output: Dict[str, Any] = {
        "tool": "action_rec",
        "interval": interval,
        "reason": reason,
        "actions_ranked": [],
        "reasoning": "",
        "summary": "no analysis performed",
    }

    if model is None or not frames:
        if not frames:
            default_output["summary"] = "no frames available from metadata"
        memory.record_result(tool="action_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    images = _load_scene_images(frames)
    if not images:
        default_output["summary"] = "no loadable images from metadata frames"
        memory.record_result(tool="action_rec", interval=interval, output=default_output, iteration=iteration)
        return default_output

    # Current frame is always frames[-1]
    current_frame_id = frames[-1].get("frame_id", "")
    num_frames = len(frames)

    # ── Build CVS context based on cvs_context_mode ──
    cvs_summary = ""
    checklist_context = ""

    if cvs_context_mode == "current":
        # Scores only — no checklist, no summary (mimics baseline CoT info level)
        cvs_status_block = _build_cvs_status_block(memory, criteria)

    elif cvs_context_mode == "full":
        # Full CVS history from all past assessments
        cvs_status_block = _build_cvs_status_block(memory, criteria)
        cvs_context = _get_latest_cvs_from_memory(memory)
        cvs_summary = cvs_context.get("summary", "")
        checklist_context = _build_full_cvs_history_block(memory, criteria)

    elif cvs_context_mode == "video_summary":
        # Video-level running CVS summary (max-accumulated scores + reasons)
        cvs_status_block = _get_video_cvs_summary(memory, criteria)

    else:  # "latest" (default) — original behavior
        cvs_context = _get_latest_cvs_from_memory(memory)
        cvs_summary = cvs_context.get("summary", "")
        cvs_checklist = cvs_context.get("checklist", {})
        cvs_status_block = _build_cvs_status_block(memory, criteria)
        if isinstance(cvs_checklist, dict) and cvs_checklist:
            checklist_lines = ["CVS checklist from prior assessment:"]
            for c in criteria:
                items = cvs_checklist.get(c, [])
                if isinstance(items, list) and items:
                    checklist_lines.append(f"  {c}:")
                    for item in items:
                        if isinstance(item, dict):
                            q = item.get("question", "")
                            a = item.get("answer", "uncertain")
                            checklist_lines.append(f"    - {q} → {a}")
            checklist_context = "\n".join(checklist_lines) + "\n"

    system_prompt = SYSTEM_PREAMBLE

    user_prompt = (
        f"{build_action_rec_prompt_body(
            taxonomy_catalog=taxonomy_catalog,
            cvs_status_block=cvs_status_block,
            cvs_summary=cvs_summary,
            checklist_context=checklist_context,
            fixed_k=fixed_k,
            one_per_actor=one_per_actor,
            action_rec_rules=action_rec_rules,
        )}"
    )

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": user_prompt}
    ]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    text = (response.content or "").strip()
    parsed = _parse_scene_cvs_output(text)

    # Extract reasoning
    reasoning = str(parsed.get("reasoning", "")).strip()

    # Extract actions_ranked
    actions_raw = parsed.get("actions", parsed.get("actions_ranked", []))
    if not isinstance(actions_raw, list):
        actions_raw = []
    actions_ranked: List[Dict[str, Any]] = []
    for a in actions_raw:
        if not isinstance(a, dict):
            continue
        actions_ranked.append({
            "rank": a.get("rank"),
            "actor_role": str(a.get("actor_role", "(not set)")),
            "tool_type": str(a.get("tool_type", "(not set)")),
            "action_code": str(a.get("action_code", "(not set)")),
            "target_structure": str(a.get("target_structure", "(not set)")),
            "target_context_1": str(a.get("target_context_1", "(not set)")),
            "target_context_2": str(a.get("target_context_2", "(not set)")),
            "intention": str(a.get("intention", "(not set)")),
            "confidence": _safe_float(a.get("confidence", 0.0)),
            "evidence": str(a.get("evidence", "")),
        })

    actions_ranked = _enforce_fixed_k(actions_ranked, fixed_k or AUDIT_V11_SIMPLE_TARGET_COUNT)

    summary = f"{len(actions_ranked)} action(s) recommended."
    if reasoning:
        summary = f"{reasoning[:200]}. {summary}"

    output: Dict[str, Any] = {
        "tool": "action_rec",
        "interval": interval,
        "reason": reason,
        "actions_ranked": actions_ranked,
        "reasoning": reasoning,
        "summary": summary,
        "frames": frames,
    }
    memory.record_result(tool="action_rec", interval=interval, output=output, iteration=iteration)
    return output


def tool_registry() -> Dict[str, Any]:
    return {
        "scene_snapper": scene_snapper,
        "scene_cvs_analyzer": scene_cvs_analyzer,
        "action_rec": action_rec,
        "clip_analyzer": clip_analyzer,
    }
