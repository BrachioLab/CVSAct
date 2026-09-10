from __future__ import annotations

import inspect
import json
import os
import re
from typing import Any, Dict, List, Optional

from .audit_simple_actions import (
    AUDIT_V11_SIMPLE_TARGET_COUNT,
    augment_taxonomy_catalog_for_audit_v11_simple,
    build_audit_v11_simple_prompt_block,
)
from .memory_hm3 import HM3Memory
from .prompts import CONTROLLER_USER_PROMPT_TEMPLATE
from .schemas import CRITERION_DEFINITIONS

_RUBRIC_LABEL_TYPE_DEFS: Dict[str, str] = {
    "precondition": (
        "Visibility or identifiability prerequisite for judging the criterion; "
        "not direct proof of satisfaction."
    ),
    "evidence": "Direct visual evidence supporting or refuting criterion satisfaction.",
    "modifier": (
        "Factors that affect confidence by influencing observability "
        "(e.g., camera angle, instrument occlusion), not the surgical state itself."
    ),
}

_RUBRIC_ITEMS: Dict[str, list[Dict[str, Any]]] = {
    "C1": [
        {"id": "C1-1", "text": "The gallbladder neck (infundibulum) region is clearly visible and in-frame.", "weight": 0.1, "label_type": "precondition"},
        {"id": "C1-2", "text": "There are exactly two tubular attachment tracks entering the gallbladder wall.", "weight": 0.3, "label_type": "evidence"},
        {"id": "C1-3", "text": "The two tubular structures are visually distinguishable from each other at the gallbladder entry.", "weight": 0.3, "label_type": "evidence"},
        {"id": "C1-4", "text": "The gallbladder entry site is sufficiently exposed that no additional tubular structure could plausibly be hidden.", "weight": 0.1, "label_type": "evidence"},
        {"id": "C1-5", "text": "Camera angle and exposure are sufficient to confidently assess the number of tubular structures entering the gallbladder.", "weight": 0.1, "label_type": "modifier"},
        {"id": "C1-6", "text": "No instrument is present, or if present, it does not obscure the intersection between the gallbladder and the tubular structures.", "weight": 0.1, "label_type": "modifier"},
    ],
    "C2": [
        {"id": "C2-1", "text": "The hepatocystic triangle anatomy - cystic duct, common hepatic duct, and inferior liver edge - is all clearly identifiable and in-frame.", "weight": 0.1, "label_type": "precondition"},
        {"id": "C2-2", "text": "Minimal fat or fibrous tissue is within the hepatocystic triangle.", "weight": 0.2, "label_type": "evidence"},
        {"id": "C2-3", "text": "Liver parenchyma is visible within the hepatocystic triangle (i.e., between the cystic duct and the liver edge), not merely beneath the gallbladder.", "weight": 0.3, "label_type": "evidence"},
        {"id": "C2-4", "text": "No continuous sheet of fat or fibrous tissue spans across and occludes the interior of the hepatocystic triangle.", "weight": 0.2, "label_type": "evidence"},
        {"id": "C2-5", "text": "Camera angle and exposure are sufficient to confidently assess hepatocystic triangle clearance.", "weight": 0.1, "label_type": "modifier"},
        {"id": "C2-6", "text": "No instrument is present, or if present, it does not obscure the interior of the hepatocystic triangle.", "weight": 0.1, "label_type": "modifier"},
    ],
    "C3": [
        {"id": "C3-1", "text": "The boundary between the gallbladder wall and liver at the lower third is clearly visible and in-frame.", "weight": 0.1, "label_type": "precondition"},
        {"id": "C3-2", "text": "A clearly visible separation interface indicating detachment is present between the gallbladder wall and the liver bed in the lower third.", "weight": 0.3, "label_type": "evidence"},
        {"id": "C3-3", "text": "The exposed liver bed (cystic plate) surface is visible at the attachment site of the lower third of the gallbladder.", "weight": 0.2, "label_type": "evidence"},
        {"id": "C3-4", "text": "The detachment extends across a substantial portion of the lower third, not just a focal point.", "weight": 0.1, "label_type": "evidence"},
        {"id": "C3-5", "text": "No continuous tissue bridge connects the lower third of the gallbladder to the liver bed.", "weight": 0.1, "label_type": "evidence"},
        {"id": "C3-6", "text": "Camera angle and exposure are sufficient to confidently assess gallbladder detachment from the liver bed.", "weight": 0.1, "label_type": "modifier"},
        {"id": "C3-7", "text": "No instrument is present, or if present, it does not obscure the gallbladder-liver interface.", "weight": 0.1, "label_type": "modifier"},
    ],
}


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
        "scene_cvs_analyzer",
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
        for i in range(start, end + 1):
            covered[i - visible_start] = True
    covered_count = sum(1 for v in covered if v)
    return covered_count, visible_total


def _build_prior_calls_summary(memory: HM3Memory) -> str:
    """Summarize all prior tool calls so the controller can see what's been done."""
    if not memory.results.events:
        return ""
    from collections import defaultdict
    calls: dict[str, list[str]] = defaultdict(list)
    for entry in memory.results.events:
        interval = entry.interval if isinstance(entry.interval, dict) else {}
        interval_str = f"[{interval.get('start', '?')},{interval.get('end', '?')}]"
        calls[entry.tool].append(interval_str)
    lines = ["Prior tool calls (do NOT repeat the same tool on the same interval):"]
    for tool_name, intervals in calls.items():
        lines.append(f"  {tool_name}: {', '.join(intervals)}")
    return "\n".join(lines)


def _build_exploration_policy_block(memory: HM3Memory) -> str:
    covered_count, total_frames = _estimate_explored_frame_coverage(memory)
    ratio = (covered_count / total_frames) if total_frames > 0 else 0.0
    visible_line = "- Visible window: unknown."
    if memory.metadata and int(memory.metadata.num_frames) > 0:
        start_idx, end_idx = _resolve_visible_bounds(memory, int(memory.metadata.num_frames))
        visible_line = (
            f"- Visible window: frames {start_idx}..{end_idx} "
            f"({max(0, end_idx - start_idx + 1)} frame(s))."
        )
    return (
        "Exploration Policy:\n"
        f"{visible_line}\n"
        "- Keep exploring different temporal parts of the video while evidence is incomplete.\n"
        "- You MUST NOT conclude decision=\"unsatisfied\" unless all different frame regions have been explored.\n"
        "- If coverage is not complete, continue tool calls instead of final unsatisfied answer.\n"
        "- If you still choose unsatisfied, explicitly justify full exploration coverage.\n"
        f"- Current explored coverage (visible window): {covered_count}/{total_frames} frames ({ratio:.1%})."
    )


def _build_rubric_questions_block(criterion: str) -> str:
    criteria = [
        token
        for token in [x.strip().upper() for x in re.split(r"[,\s]+", str(criterion)) if x.strip()]
        if token in _RUBRIC_ITEMS
    ]
    if not criteria:
        criteria = [str(criterion).upper()]
    lines = ["Preferred Rubric Questions To Consider:"]
    lines.append(
        "Prioritize these rubric questions when planning actions. "
        "You may ask additional questions only when necessary and you must justify why. "
        "When multiple criteria are in scope, address all unsatisfied criteria — not just the weakest. "
        "to maximize joint progress toward all criteria."
    )
    for c in criteria:
        items = _RUBRIC_ITEMS.get(c, [])
        if not items:
            continue
        lines.append(f"- Criterion {c}:")
        for item in items:
            item_id = str(item.get("id", ""))
            text = str(item.get("text", ""))
            weight = item.get("weight", "")
            label_type = str(item.get("label_type", ""))
            lines.append(
                f"- {item_id} [{label_type}, weight={weight}]: {text}"
            )
    lines.append("Label type definitions:")
    for key, desc in _RUBRIC_LABEL_TYPE_DEFS.items():
        lines.append(f"- {key}: {desc}")
    return "\n".join(lines)


def _build_stage_context_block(
    llm_calls: int,
    max_steps: int,
    num_preferred_tools: int,
    fixed_k: Optional[int] = None,
) -> str:
    """Build a stage-awareness block for the controller prompt.

    Preferred tools are auto-called before the controller decides and do not
    count as reasoning steps.  The two-stage split is based on the remaining
    *real* (non-auto) steps.
    """
    real_steps_used = max(0, llm_calls - num_preferred_tools)
    real_steps_total = max(1, max_steps - num_preferred_tools)
    steps_remaining = max(0, real_steps_total - real_steps_used)
    del fixed_k
    action_count = AUDIT_V11_SIMPLE_TARGET_COUNT

    if real_steps_used < real_steps_total // 2:
        stage = 1
        stage_label = "Stage 1 — CVS Evidence Improvement"
        focus = (
            "Goal: improve the accuracy and confidence of CVS assessment.\n"
            "At each step:\n"
            "  1. Identify which CVS criterion is most uncertain or blocking progress.\n"
            "  2. Specify what anatomical evidence is missing.\n"
            "  3. Select and call the most informative tool to gather that evidence.\n"
            "  4. Update internal assessment of CVS and confidence.\n"
            "Transition to Stage 2 early if:\n"
            "  - CVS state is sufficiently clear for action recommendation, OR\n"
            "  - Additional tool calls are unlikely to provide new information."
        )
    else:
        stage = 2
        stage_label = "Stage 2 — Recommendation Improvement"
        focus = (
            "Goal: refine and prioritize the next surgical actions.\n"
            "At each step:\n"
            "  1. Identify the unsatisfied CVS criterion.\n"
            "  2. Determine what is currently blocking its satisfaction.\n"
            "  3. Generate candidate next actions that most effectively and safely reduce that gap.\n"
            "  4. Refine one action per actor slot: camera, left_instrument, right_instrument, other.\n"
            "If a critical uncertainty prevents safe recommendation, you may call one additional tool, "
            "then continue refining actions.\n"
            "When actions are refined, set stop=true."
        )

    action_count_line = f"- Action output: exactly {action_count} actor-slot next actions in recommended_actions."
    swap_policy = (
        "- Swap policy: each actor slot (camera, left_instrument, right_instrument, other) "
        "should have at most one action. When new evidence suggests a better action "
        "for a slot (e.g. RETRACT_LATERAL → RETRACT_MEDIAL_TO_LATERAL), replace the "
        "old one instead of keeping both. Use CAMERA_NO_CHANGE when camera should stay stable. "
        "Reserve the other slot for actions such as ICG_SWITCH or leave it as no additional other action."
    )

    return (
        f"Current Stage: {stage_label}\n"
        f"- Steps used: {real_steps_used}/{real_steps_total} "
        f"(auto-calls excluded: {num_preferred_tools})\n"
        f"- Steps remaining: {steps_remaining}\n"
        f"- Focus: {focus}\n"
        f"{action_count_line}\n"
        f"{swap_policy}\n"
    )


def _taxonomy_options_block_for_controller(catalog: Optional[Dict[str, Any]]) -> str:
    """Build a taxonomy options block so the controller knows valid values
    for its recommended_actions output."""
    if not catalog:
        return ""
    audit_catalog = augment_taxonomy_catalog_for_audit_v11_simple(catalog)
    options = audit_catalog.get("options", {})
    if not isinstance(options, dict) or not options.get("action_code"):
        return ""
    intention_descs = audit_catalog.get("intention_descs", {})
    _fields = (
        "actor_role", "tool_type", "action_code",
        "target_structure", "target_context", "intention",
    )
    lines = ["Allowed taxonomy values for recommended_actions:"]
    for field in _fields:
        vals = options.get(field, [])
        if field == "intention" and intention_descs:
            items = []
            for v in vals:
                desc = intention_descs.get(v, "")
                items.append(f'"{v}": {desc}' if desc else f'"{v}"')
            lines.append(f"  {field}: [{', '.join(items)}]")
        else:
            lines.append(f"  {field}: {json.dumps(vals, ensure_ascii=False)}")
    lines.append("")
    lines.append(build_audit_v11_simple_prompt_block())
    return "\n".join(lines)


def build_controller_prompt(
    criterion: str,
    memory: HM3Memory,
    available_tools: Dict[str, Any],
    *,
    analyze_actions: bool = False,
    recommend_actions: bool = False,
    llm_calls: int = 0,
    max_steps: int = 10,
    num_preferred_tools: int = 0,
    taxonomy_catalog: Optional[Dict[str, Any]] = None,
    fixed_k: Optional[int] = None,
) -> str:
    memory_snapshot = memory.snapshot()
    tool_lines = []
    for name, fn in available_tools.items():
        params: list[str] = []
        args_schema = getattr(fn, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "model_fields"):
            for p_name, field in args_schema.model_fields.items():
                if p_name in {"memory", "model", "iteration"}:
                    continue
                if field.is_required():
                    params.append(p_name)
                else:
                    params.append(f"{p_name}={field.default!r}")
        else:
            sig = inspect.signature(fn)
            for p_name, p in sig.parameters.items():
                if p_name in {"memory", "model", "iteration"}:
                    continue
                if p.default is inspect._empty:
                    params.append(p_name)
                else:
                    params.append(f"{p_name}={p.default!r}")
        tool_lines.append(f"- {name}: {name}({', '.join(params)})")
    fps = os.getenv("SURGENT_VIDEO_FPS", "unknown")
    total_frames = memory.metadata.num_frames if memory.metadata else 0
    visible_start, visible_end = _resolve_visible_bounds(memory, total_frames)
    num_visible = max(0, visible_end - visible_start + 1) if total_frames > 0 else 0
    duration_seconds: str
    try:
        fps_value = float(fps)
        duration_seconds = f"{(total_frames / fps_value):.2f}" if fps_value > 0 else "unknown"
    except Exception:
        duration_seconds = "unknown"
    criteria = [
        token
        for token in [x.strip().upper() for x in re.split(r"[,\s]+", str(criterion)) if x.strip()]
        if token in CRITERION_DEFINITIONS
    ]
    if len(criteria) > 1:
        defs = " ".join([f"{c}: {CRITERION_DEFINITIONS[c]}" for c in criteria])
        question_text = (
            f"Jointly assess CVS criteria {', '.join(criteria)} for this surgical video. "
            f"Definitions: {defs}. "
            "Primary objective: recommend actions that get closest to satisfying all criteria together, "
            "not repeatedly re-validating already high-score criteria. "
            "Report per-criterion satisfaction as a score in [0,1] where 0=clearly unsatisfied, 0.5=uncertain, 1.0=clearly satisfied. "
            "Provide a video-level decision in {Satisfied, Unsatisfied, Uncertain}, where Satisfied means all criteria scores >= 0.7."
        )
    else:
        criterion_def = CRITERION_DEFINITIONS.get(criterion, "")
        question_text = (
            f"Assess CVS {criterion} for this surgical video. "
            f"Definition: {criterion_def}. "
            "Provide a video-level decision in {Satisfied, Unsatisfied, Uncertain}."
        )
    user_prompt = CONTROLLER_USER_PROMPT_TEMPLATE.format(
        total_frames=total_frames,
        visible_start=visible_start,
        visible_end=visible_end,
        num_visible=num_visible,
        duration_seconds=duration_seconds,
        fps=fps,
        memory_json=json.dumps(memory_snapshot, indent=2),
        question_text=question_text,
    )
    rubric_block = _build_rubric_questions_block(criterion)
    exploration_policy_block = _build_exploration_policy_block(memory)

    # Build a summary of prior tool calls so the controller knows what's been done
    prior_calls_summary = _build_prior_calls_summary(memory)

    cvs_priority_block = (
        "CVS Analysis:\n"
        "- scene_cvs_analyzer provides frame-level CVS predictions. Use these as your primary evidence.\n"
        "- Your current_frame_criteria_prediction should reflect CVS status AT THE CURRENT FRAME.\n"
        "- Video-level aggregation happens automatically — you only need to judge the current frame.\n"
        "- Set stop=true when you have enough evidence to judge CVS AND have refined your recommended actions.\n"
        "  stop=true does NOT mean CVS is satisfied — it means examination is complete.\n"
        "- Do NOT re-run scene_cvs_analyzer on the same frame interval. Duplicate calls are blocked.\n"
        "- To resolve uncertain scores or to gather evidence for action refinement, use zoom\n"
        "  (to magnify specific structures) or clip_analyzer (to ask a focused question about\n"
        "  specific visual evidence such as instrument positions or tissue state).\n"
        + (f"\n{prior_calls_summary}" if prior_calls_summary else "")
    )

    action_analysis_block = ""
    if analyze_actions:
        action_analysis_block = (
            "Action Analysis Context:\n"
            "- scene_cvs_analyzer reports what is CURRENTLY happening in the scene (current actions) "
            "and recommends what SHOULD be done next.\n"
            "- Analyze the recommendations: if a recommended action is the same as what is already being done, "
            "you do not need to recommend it again.\n"
            "- If you need more evidence to confirm the right action to take next, "
            "feel free to explore more using available tools.\n"
            "- Balance your effort between gathering evidence for CVS prediction "
            "and for action recommendation.\n"
        )

    recommend_actions_block = ""
    if recommend_actions:
        recommend_actions_block = (
            "Action Recommendation Context:\n"
            "- scene_cvs_analyzer RECOMMENDS next actions, not retrospective clip labels or current-only visible actions.\n"
            "- Its actions_ranked output contains what the surgeon SHOULD do next to improve unsatisfied CVS criteria.\n"
            "- Treat scene_cvs_analyzer recommendations as initial candidates. Actively refine them into exactly four actor slots:\n"
            "  - right_instrument\n"
            "  - left_instrument\n"
            "  - camera\n"
            "  - other\n"
            "  - Use zoom to inspect instrument positions and verify what each instrument is doing.\n"
            "  - Use clip_analyzer to ask targeted questions (e.g. 'What is the right instrument grasping?',\n"
            "    'Is the hepatocystic triangle exposed?') to confirm or revise action choices.\n"
            "- In Stage 2, your primary goal is to improve action quality, not just CVS scores.\n"
            "  Determine what is blocking each unsatisfied criterion and what action would most\n"
            "  effectively and safely reduce that gap.\n"
        )

    stage_block = _build_stage_context_block(llm_calls, max_steps, num_preferred_tools, fixed_k=fixed_k)
    taxonomy_block = _taxonomy_options_block_for_controller(taxonomy_catalog)

    return (
        "Available MM actions:\n"
        + "\n".join(tool_lines)
        + "\n\n"
        "Action args must exclude injected runtime fields: memory, model, iteration.\n\n"
        f"{stage_block}\n"
        f"{cvs_priority_block}\n\n"
        + (f"{action_analysis_block}\n" if action_analysis_block else "")
        + (f"{recommend_actions_block}\n" if recommend_actions_block else "")
        + f"{exploration_policy_block}\n\n"
        + (f"{taxonomy_block}\n\n" if taxonomy_block else "")
        + f"{user_prompt}\n"
        + (f"{rubric_block}\n\n" if rubric_block else "")
    )


def default_controller_output() -> Dict[str, Any]:
    return {
        "action": "ANSWER",
        "action_args": {},
        "reason": "template_only",
        "confidence": 0.0,
        "stop": True,
    }


def record_working_from_response(
    memory: HM3Memory,
    response: Dict[str, Any],
    iteration: int,
) -> None:
    action = response.get("action")
    if not action or action == "ANSWER":
        return
    objective = str(response.get("working_memory_note") or response.get("reason") or "")
    trace = str(response.get("thought") or response.get("reason") or "")
    memory.record_working(
        iteration=iteration,
        objective=objective,
        reasoning_trace=trace,
        tool=str(action),
    )


def record_result_from_tool(
    memory: HM3Memory,
    tool: str,
    interval: Dict[str, Any],
    output: Dict[str, Any],
    iteration: int,
) -> None:
    memory.record_result(
        tool=tool,
        interval=interval,
        output=output,
        iteration=iteration,
    )
