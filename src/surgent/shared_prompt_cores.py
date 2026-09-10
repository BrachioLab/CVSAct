from __future__ import annotations

from typing import Any, Dict, Optional

from .audit_simple_actions import AUDIT_V11_SIMPLE_TARGET_COUNT, load_audit_v11_simple_options
from .schemas import CRITERION_DEFINITIONS


SYSTEM_PREAMBLE = (
    "You are a surgical vision assistant specializing in laparoscopic cholecystectomy.\n"
    "Return ONLY valid JSON with no markdown fencing or extra text."
)


def _cvs_definition_block() -> str:
    lines = []
    for criterion in ("C1", "C2", "C3"):
        definition = CRITERION_DEFINITIONS.get(criterion, "")
        if criterion == "C3" and definition:
            definition = "The lower third of the gallbladder is detached from the cystic plate."
        if definition:
            lines.append(f"- {criterion}: {definition}")
    return "\n".join(lines)


def _taxonomy_options_block(catalog: Dict[str, Any]) -> str:
    options = catalog.get("options", {}) if isinstance(catalog, dict) else {}
    lines = []
    for field in ("actor_role", "tool_type", "action_code", "target_structure", "target_context"):
        vals = options.get(field, [])
        if isinstance(vals, list):
            lines.append(f"  {field}: {vals}")
    return "\n".join(lines)


def _normalize_target_context_values() -> list[str]:
    simple_options = load_audit_v11_simple_options()
    values = list((simple_options.get("interface_options") or {}).get("target_context", []))
    normalized = []
    for value in values:
        if value == "between cystic artery and liver bed":
            value = "between cystic artery and cystic plate"
        if value == "the position of the grasper moved more towards the the gallbladder body":
            value = "the position of the grasper moved more towards the gallbladder body"
        normalized.append(value)
    return list(dict.fromkeys(normalized))


def _target_context_block() -> str:
    values = _normalize_target_context_values()
    lines = []
    for idx, value in enumerate(values):
        suffix = " |" if idx < len(values) - 1 else ""
        lines.append(f'"{value}"{suffix}')
    return "\n".join(lines)


def _action_schema_block(fixed_k: Optional[int] = None) -> str:
    del fixed_k
    return (
        '  "actions": [\n'
        "    {\n"
        '      "actor_role": "camera",\n'
        '      "tool_type": "camera",\n'
        '      "action_code": "<string>",\n'
        '      "evidence": "<short phrase>"\n'
        "    },\n"
        "    {\n"
        '      "actor_role": "left_instrument",\n'
        '      "tool_type": "Grasper",\n'
        '      "action_code": "<string>",\n'
        '      "target_context_1": "<optional>",\n'
        '      "target_context_2": "<optional>",\n'
        '      "evidence": "<short phrase>"\n'
        "    },\n"
        "    {\n"
        '      "actor_role": "right_instrument",\n'
        '      "tool_type": "<observed instrument>",\n'
        '      "action_code": "<string>",\n'
        '      "target_structure": "<string>",\n'
        '      "target_context_1": "<optional>",\n'
        '      "target_context_2": "<optional>",\n'
        '      "evidence": "<short phrase>"\n'
        "    },\n"
        "    {\n"
        '      "actor_role": "other",\n'
        '      "action_code": "<string>",\n'
        '      "evidence": "<short phrase>"\n'
        "    }\n"
        "  ]"
    )


def _fixed_k_constraint_block(k: int) -> str:
    return (
        f"You MUST provide exactly {AUDIT_V11_SIMPLE_TARGET_COUNT} actions in actions.\n"
        "Use exactly one slot each for camera, left_instrument, right_instrument, and other.\n"
        f"Ignore any older fixed-k={k} convention and follow the four actor slots.\n"
    )


def build_scene_cvs_frame_intro(num_frames: int, current_frame_id: str) -> str:
    if num_frames == 1:
        return (
            "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
            f"This is frame {current_frame_id} — the CURRENT evaluation frame.\n"
        )
    return (
        f"You are viewing {num_frames} frames from a laparoscopic cholecystectomy video.\n"
        f"Frame {current_frame_id} is the CURRENT evaluation frame — scores must reflect "
        "the state at this frame. Earlier frames provide temporal context.\n"
    )


def build_scene_cvs_prompt_body(*, preset: str) -> str:
    del preset
    return (
        "## CVS Criteria Definitions\n\n"
        f"{_cvs_definition_block()}\n\n"
        "## Task: CVS Confidence Scoring\n\n"
        "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
        "For each CVS criterion, provide a short rationale, then a probability in [0, 1].\n\n"
        "Output schema:\n"
        "{\n"
        '  "rationale": { "c1": "<string>", "c2": "<string>", "c3": "<string>" },\n'
        '  "pred": { "c1": <float>, "c2": <float>, "c3": <float> }\n'
        "}\n"
    )


def build_scene_cvs_response_schema(*, preset: str) -> str:
    del preset
    return (
        "{\n"
        '  "rationale": { "c1": "<string>", "c2": "<string>", "c3": "<string>" },\n'
        '  "pred": { "c1": float, "c2": float, "c3": float }\n'
        "}\n"
    )


def _action_rec_conservative_visible_rules_block() -> str:
    return (
        "### Conservative Visible-Evidence Rules\n\n"
        "Only recommend actions supported by visible evidence. CVS state may guide\n"
        "priority, but it must not override the current visual action.\n\n"
        "CVS-aware right-instrument targeting:\n"
        "  - Do not infer a right-instrument action from CVS state alone. First decide\n"
        "    whether the right instrument is visibly active in the current frame or\n"
        "    short temporal context.\n"
        "  - If C1/C2 are already satisfied and C3 remains unsatisfied, do not\n"
        "    automatically choose CysticPlate.\n"
        "  - Emit right_instrument = \"(not set)\" when there is no clear active\n"
        "    right-instrument manipulation. For that slot, use \"(not set)\" for\n"
        "    action_code, tool_type, target_structure, and target_context fields.\n"
        "  - Use CysticPlate only when the right instrument is visibly dissecting,\n"
        "    peeling, cauterizing, or separating tissue at the gallbladder-liver\n"
        "    interface, cystic plate, or liver bed.\n"
        "  - Use HepatocysticTriangle when the visible action is general dissection or\n"
        "    exposure within Calot's triangle and no specific duct, artery, or cystic\n"
        "    plate interface is clearly targeted.\n"
        "  - Use CysticDuct or CysticArtery only when the action is visibly focused on\n"
        "    one of those tubular structures.\n"
        "  - CVS status can guide what would be surgically useful, but the emitted\n"
        "    action must match the visible current action.\n\n"
        "Camera action rules:\n"
        "  - Do not recommend a camera action unless the view problem is clear.\n"
        "  - Use CAMERA_NO_CHANGE when the operative field is adequately framed and key\n"
        "    anatomy is visible enough for the current task.\n"
        "  - Use CAMERA_ZOOM_IN only when the view is clearly too far away and the\n"
        "    target anatomy or instrument-tissue interaction is too small to assess.\n"
        "  - Use CAMERA_ZOOM_OUT only when the view is clearly too close, so surrounding\n"
        "    context is missing or the instrument/target relationship cannot be\n"
        "    understood.\n"
        "  - Use CAMERA_REPOSITION only when the field is clearly off-center, relevant\n"
        "    anatomy is partly outside the frame, or only part of the hepatocystic\n"
        "    triangle is visible despite needing the full triangle for assessment.\n"
        "  - Do not use CAMERA_REPOSITION merely because CVS is incomplete; use it only\n"
        "    when reframing would visibly improve assessment or action.\n"
        "  - If uncertain between a camera movement and no movement, prefer\n"
        "    CAMERA_NO_CHANGE.\n\n"
    )


def build_action_rec_prompt_body(
    *,
    taxonomy_catalog: Dict[str, Any],
    cvs_status_block: str = "",
    cvs_summary: str = "",
    checklist_context: str = "",
    fixed_k: Optional[int] = None,
    one_per_actor: bool = False,
    action_rec_rules: str = "default",
) -> str:
    del taxonomy_catalog
    del cvs_summary
    del checklist_context
    del fixed_k
    del one_per_actor
    normalized_rules = str(action_rec_rules or "default").strip().lower().replace("-", "_")

    context_block = ""
    if cvs_status_block.strip():
        context_block = (
            "Current CVS status (from prior assessment):\n"
            f"{cvs_status_block.rstrip()}\n\n"
        )
    rules_block = ""
    if normalized_rules in {"conservative_visible", "visible_conservative"}:
        rules_block = _action_rec_conservative_visible_rules_block()

    return (
        "## Task: Next-Action Recommendation\n\n"
        "Recommend the next surgical actions to improve CVS achievement.\n"
        "These are prospective (what should happen next), not retrospective labels.\n\n"
        "Output exactly 4 action slots in fixed order: camera, left_instrument,\n"
        'right_instrument, other. Each slot has a required "evidence" field (short phrase).\n\n'
        f"{context_block}"
        f"{rules_block}"
        "### Instrument Identification (do this first)\n\n"
        "Before recommending actions, identify which instruments are visible in the frame.\n"
        "The tool_type in each slot MUST match what you observe.\n"
        "Common right instruments:\n"
        "  - Hook: thin L-shaped tip\n"
        "  - Maryland: long curved jaw tips\n"
        "  - Scissors: two blades\n"
        "  - Irrigator: tube-like\n"
        "Do NOT default to Hook.\n\n"
        "### Action Taxonomy\n\n"
        'CAMERA (actor_role: "camera")\n'
        '  tool_type: "camera"\n'
        "  action_code: one of\n"
        "    CAMERA_ZOOM_IN | CAMERA_ZOOM_OUT | CAMERA_REPOSITION |\n"
        "    CAMERA_NO_CHANGE | CAMERA_UNCERTAIN\n"
        "  No target_structure or target_context.\n"
        "  Guidelines:\n"
        "    - Structures too small -> CAMERA_ZOOM_IN\n"
        "    - View too close / cannot see full field -> CAMERA_ZOOM_OUT\n"
        "    - Triangle not centered or visible -> CAMERA_REPOSITION\n"
        "    - View adequate -> CAMERA_NO_CHANGE\n\n"
        'LEFT INSTRUMENT (actor_role: "left_instrument")\n'
        '  tool_type: "Grasper"\n'
        "  action_code: one of\n"
        "    KEEP_RETRACT_LATERAL | KEEP_RETRACT_MEDIAL | KEEP_RETRACT_UPWARD |\n"
        "    RETRACT_LATERAL | RETRACT_MEDIAL |\n"
        "    RETRACT_LATERAL_TO_MEDIAL | RETRACT_LATERAL_TO_UPWARD |\n"
        "    RETRACT_MEDIAL_TO_LATERAL | RETRACT_UPWARD_TO_LATERAL\n"
        "  No target_structure. Add target_context only if genuinely useful.\n"
        "  Guidelines:\n"
        "    - Pulling LEFT of image = lateral/anterior retraction\n"
        "    - Pulling RIGHT of image = medial/posterior retraction\n"
        "    - Pulling UPWARD = cephalad retraction\n"
        "    - If current retraction is adequate -> use KEEP_RETRACT_* code\n"
        "    - If exposure is insufficient -> recommend direction change\n"
        "    - RETRACT_X_TO_Y means the grasper is currently retracting in direction X\n"
        "      and should change to direction Y. For example, RETRACT_LATERAL_TO_MEDIAL\n"
        "      means it is currently pulling left (lateral) and should switch to pulling\n"
        "      right (medial).\n\n"
        'RIGHT INSTRUMENT (actor_role: "right_instrument")\n'
        "  tool_type: must match the visible instrument\n"
        "  Allowed actions per tool (physical constraints):\n"
        "    Hook:      DISSECT | COAGULATE_HEMOSTASIS | COUNTERTRACTION_ASSIST |\n"
        "               RETRACT_DOWNWARD | TOOL_WITHDRAW_UNBLOCKS_VIEW\n"
        "    Maryland:  DISSECT | sweeping | TOOL_WITHDRAW_UNBLOCKS_VIEW\n"
        "    Irrigator: COUNTERTRACTION_ASSIST | IRRIGATOR_ASPIRATE\n"
        "    Scissors:  DISSECT\n"
        "    clipper:   CLIP\n\n"
        "  target_structure (required, independent of tool/action):\n"
        "    CysticArtery | CysticDuct | CysticPlate |\n"
        "    GallbladderNeck_Infundibulum | HepatocysticTriangle\n\n"
        "  Choosing target_structure:\n"
        "    - If the hepatocystic triangle is not well delineated and the next action\n"
        "      is general dissection in that region, use HepatocysticTriangle.\n"
        "    - If the cystic artery or cystic duct are at least partially visible,\n"
        "      even if not fully skeletonized, and the next dissection should focus\n"
        "      around one of them, prefer the more specific target (CysticArtery,\n"
        "      CysticDuct).\n"
        "    - If the two tubular structures are visible but the next action targets\n"
        "      the general space between them rather than skeletonizing either one,\n"
        "      use HepatocysticTriangle with target_context \"between presumed cystic\n"
        "      duct and presumed cystic artery\".\n"
        "    - Similarly, if the next action should target the gallbladder-cystic plate\n"
        "      interface, use CysticPlate.\n\n"
        "  Optional target_context (see shared list below).\n"
        "  If the next action targets a specific side of a structure, specify that\n"
        "  side in target_context (e.g., \"between presumed cystic duct and presumed\n"
        "  cystic artery\" or \"between cystic artery and cystic plate\"). Use both\n"
        "  target_context_1 and target_context_2 if the action spans both sides.\n\n"
        "  This is the only slot that uses target_structure.\n\n"
        "  Action guidelines:\n"
        "    - C1/C2 improvement -> dissect in the hepatocystic triangle area\n"
        "    - C3 improvement -> dissect the cystic plate / gallbladder-plate interface\n"
        "    - COUNTERTRACTION_ASSIST: the right tool is helping the grasper change\n"
        "      retraction direction rather than actively dissecting\n"
        "    - IRRIGATOR_ASPIRATE: an irrigator is clearing blood or fluid from the field\n"
        "    - TOOL_WITHDRAW_UNBLOCKS_VIEW: the tool should be removed from the scene\n"
        "      to unblock the view. Target should be the specific region being blocked,\n"
        "      not the whole triangle\n"
        "    - COAGULATE_HEMOSTASIS: electrocautery is needed to stop active bleeding\n"
        "    - CLIP: a clipper should place a clip on the target structure\n"
        "    - sweeping: the tool should sweep across tissue without true dissection\n"
        "      (e.g., clearing blood or lightly moving tissue)\n"
        "    - RETRACT_DOWNWARD: the tool is pushing tissue down to improve\n"
        "      visualization, not dissecting\n\n"
        'OTHER (actor_role: "other")\n'
        "  action_code: ICG_SWITCH | (not set)\n"
        "  Omit tool_type and target fields unless truly needed.\n"
        "  Use \"(not set)\" when no other action is justified.\n\n"
        "### Allowed target_context values (for left or right, optional)\n\n"
        f"{_target_context_block()}\n\n"
        "### Output Schema\n\n"
        "{\n"
        f"{_action_schema_block()}\n"
        "}\n"
    )


def build_action_rec_response_schema() -> str:
    return "{\n" + _action_schema_block() + "\n}\n"


def build_action_rec_taxonomy_only_prompt_body(
    *,
    taxonomy_catalog: Dict[str, Any],
    fixed_k: Optional[int] = None,
) -> str:
    del fixed_k
    return (
        "## Task: Next-Action Recommendation\n\n"
        "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
        "Recommend the next surgical actions the surgeon should perform.\n"
        "These are prospective recommendations, not retrospective labels.\n\n"
        "Output exactly 4 action slots in fixed order: camera, left_instrument,\n"
        'right_instrument, other. Each slot has a required "evidence" field (short phrase).\n\n'
        "Allowed taxonomy values:\n"
        f"{_taxonomy_options_block(taxonomy_catalog)}\n\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        "{\n"
        f"{_action_schema_block()}\n"
        "}\n"
    )


def build_action_rec_no_guideline_prompt_body(
    *,
    taxonomy_catalog: Dict[str, Any],
    fixed_k: Optional[int] = None,
) -> str:
    del taxonomy_catalog
    del fixed_k
    return (
        "## Task: Next-Action Recommendation\n\n"
        "Recommend the next surgical actions to improve CVS achievement.\n"
        "These are prospective (what should happen next), not retrospective labels.\n\n"
        "Output exactly 4 action slots in fixed order: camera, left_instrument,\n"
        'right_instrument, other. Each slot has a required "evidence" field (short phrase).\n\n'
        "### Instrument Identification\n\n"
        "Before recommending actions, identify which instruments are visible in the frame.\n"
        "The tool_type in each slot MUST match what you observe.\n"
        "Common right instruments:\n"
        "  - Hook: thin L-shaped tip\n"
        "  - Maryland: long curved jaw tips\n"
        "  - Scissors: two blades\n"
        "  - Irrigator: tube-like\n"
        "Do NOT default to Hook.\n\n"
        "### Action Taxonomy\n\n"
        'CAMERA (actor_role: "camera")\n'
        '  tool_type: "camera"\n'
        "  action_code: one of\n"
        "    CAMERA_ZOOM_IN | CAMERA_ZOOM_OUT | CAMERA_REPOSITION |\n"
        "    CAMERA_NO_CHANGE | CAMERA_UNCERTAIN\n"
        "  No target_structure or target_context.\n\n"
        'LEFT INSTRUMENT (actor_role: "left_instrument")\n'
        '  tool_type: "Grasper"\n'
        "  action_code: one of\n"
        "    KEEP_RETRACT_LATERAL | KEEP_RETRACT_MEDIAL | KEEP_RETRACT_UPWARD |\n"
        "    RETRACT_LATERAL | RETRACT_MEDIAL |\n"
        "    RETRACT_LATERAL_TO_MEDIAL | RETRACT_LATERAL_TO_UPWARD |\n"
        "    RETRACT_MEDIAL_TO_LATERAL | RETRACT_UPWARD_TO_LATERAL\n"
        "  No target_structure. Add target_context only if genuinely useful.\n\n"
        'RIGHT INSTRUMENT (actor_role: "right_instrument")\n'
        "  tool_type: must match the visible instrument\n"
        "  Allowed actions per tool:\n"
        "    Hook:      DISSECT | COAGULATE_HEMOSTASIS | COUNTERTRACTION_ASSIST |\n"
        "               RETRACT_DOWNWARD | TOOL_WITHDRAW_UNBLOCKS_VIEW\n"
        "    Maryland:  DISSECT | sweeping | TOOL_WITHDRAW_UNBLOCKS_VIEW\n"
        "    Irrigator: COUNTERTRACTION_ASSIST | IRRIGATOR_ASPIRATE\n"
        "    Scissors:  DISSECT\n"
        "    clipper:   CLIP\n\n"
        "  target_structure:\n"
        "    CysticArtery | CysticDuct | CysticPlate |\n"
        "    GallbladderNeck_Infundibulum | HepatocysticTriangle\n"
        "  Optional target_context: see shared list below.\n"
        "  This is the only slot that uses target_structure.\n\n"
        'OTHER (actor_role: "other")\n'
        "  action_code: ICG_SWITCH | (not set)\n"
        "  Omit tool_type and target fields unless truly needed.\n\n"
        "### Allowed target_context values (for left or right, optional)\n\n"
        f"{_target_context_block()}\n\n"
        "### Output Schema\n\n"
        "{\n"
        f"{_action_schema_block()}\n"
        "}\n"
    )


def build_baseline_bridge_instruction() -> str:
    return (
        "First produce the CVS scoring JSON, then use those scores as current CVS status\n"
        "to produce the action recommendation JSON.\n\n"
        "Return a single JSON object combining both:\n"
        "{\n"
        '  "rationale": { "c1": "...", "c2": "...", "c3": "..." },\n'
        '  "pred": { "c1": <float>, "c2": <float>, "c3": <float> },\n'
        '  "actions": [ ... ]\n'
        "}\n"
    )


def build_baseline_combined_response_schema() -> str:
    return (
        "{\n"
        '  "rationale": { "c1": "<string>", "c2": "<string>", "c3": "<string>" },\n'
        '  "pred": { "c1": <float>, "c2": <float>, "c3": <float> },\n'
        f"{_action_schema_block()}\n"
        "}\n"
    )
