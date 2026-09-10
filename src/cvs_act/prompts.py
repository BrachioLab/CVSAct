"""Prompt builder and LLM invocation for CVS action descriptions."""

import os

from langchain_core.messages import HumanMessage, SystemMessage

from .io import img_to_data_url
from .schemas import (
    _CRIT_DEF_LOWER,
    VERBS,
    TOOLS,
    ITEMS_PRIORITY,
    REGIONS_PRIORITY,
    SCENE_FLAGS,
    DIRECTIONS,
    ALLOW_OTHER_ITEMS,
    ALLOW_OTHER_REGIONS,
)


def describe_transition_action(
    *,
    frame_ids,                 # frames in chronological order for the window
    frame_dir,
    llm,
    clip_type,                 # "coarse" or "fine"
    criterion,                 # "c1" / "c2" / "c3"
    video_data_entry,          # video_data[vid]
    start_idx, end_idx,        # indices into raw scores
    max_deltas=3,
    include_action_track=True, # optionally return coarse audit actions
):
    """
    One-criterion-per-call prompt builder + LLM caller.
    Forces: tool roles (left/right/camera), start vs end, deltas, ranked actions.
    Uses controlled vocab with item vs region ontology.
    """

    # ---- Criterion context with rater counts (X/3 annotators) ----
    crit_name, crit_desc = _CRIT_DEF_LOWER[criterion]
    raw = video_data_entry[criterion]["raw"]
    raters_start = int(round(raw[start_idx] * 3))
    raters_end = int(round(raw[end_idx] * 3))

    if raters_end > raters_start:
        mind_change = "unsatisfied->satisfied"
    elif raters_end < raters_start:
        mind_change = "satisfied->unsatisfied"
    else:
        mind_change = "no_change"

    crit_text = (
        f"{crit_name} ({crit_desc}) changes from "
        f"{raters_start}/3 to {raters_end}/3 annotators "
        f"({mind_change})"
    )

    if len(frame_ids) == 0:
        return None

    # ---- Select boundary frames (start/end) + tiny context buffer ----
    start_fid = frame_ids[0]
    end_fid = frame_ids[-1]

    boundary_fids = []
    boundary_fids.extend(frame_ids[: min(3, len(frame_ids))])
    if len(frame_ids) >= 5:
        boundary_fids.append(frame_ids[len(frame_ids) // 2])
    boundary_fids.extend(frame_ids[max(0, len(frame_ids) - 3):])

    # Deduplicate while preserving order
    seen = set()
    boundary_fids = [x for x in boundary_fids if not (x in seen or seen.add(x))]

    # ---- action_track schema/instruction ----
    if include_action_track:
        action_track_instruction = (
            "- action_track: 1-4 coarse audit actions; each must use controlled vocab fields.\n"
            "- Each action_track entry must have actor_side, tool, verb, item, region, direction."
        )
        action_track_schema = (
            '"action_track": ['
            '{"actor_side":"left|right|camera|unclear","tool":"...","verb":"...","item":"...","region":"...","direction":"..."}'
            "]"
        )
    else:
        action_track_instruction = "- action_track: MUST be an empty list []."
        action_track_schema = '"action_track": []'

    # ---- Controlled vocab strings for prompt ----
    verbs_str = ", ".join(VERBS)
    tools_str = ", ".join(TOOLS)
    items_str = ", ".join(ITEMS_PRIORITY)
    regions_str = ", ".join(REGIONS_PRIORITY)
    flags_str = ", ".join(SCENE_FLAGS)
    directions_str = ", ".join(DIRECTIONS)

    # ---- Escape-hatch rules ----
    other_item_rule = (
        'If none of the listed items fit, you MAY output item as "Other:<short name>". '
        "But you must prefer the provided item list whenever possible."
        if ALLOW_OTHER_ITEMS else
        "You MUST choose item from the provided list; otherwise use Null."
    )
    other_region_rule = (
        'If none of the listed regions fit, you MAY output region as "Other:<short name>". '
        "But you must prefer the provided region list whenever possible."
        if ALLOW_OTHER_REGIONS else
        'You MUST choose region from the provided list; otherwise use "Unclear".'
    )

    # ---- Prompt ----
    prompt = f"""You are a surgical video analysis assistant.

    You are viewing frames from a laparoscopic cholecystectomy video.
    We focus on ONE Critical View of Safety (CVS) criterion, and we want the most plausible
    actions that caused the annotator belief to change during this time window.

    Criterion transition:
    {crit_text}

    Clip granularity: {clip_type}
    Frames: chronological order.

    CONTROLLED VOCAB (use these whenever applicable):
    - verb ∈ [{verbs_str}]
    - tool ∈ [{tools_str}]
    - item (OBJECT/MATERIAL manipulated; PRIORITIZE EARLY ENTRIES) ∈ [{items_str}]
    - region (WHERE the action occurs; PRIORITIZE EARLY ENTRIES) ∈ [{regions_str}]
    - direction ∈ [{directions_str}]
    - scene_flags: choose any subset of [{flags_str}]
    {other_item_rule}
    {other_region_rule}

    CRITICAL ONTOLOGY RULES:
    - "Hepatocystic triangle" is a REGION, NOT an item.
    - item must be an OBJECT or MATERIAL (e.g., Fat, Connective tissue, Cystic-duct).
    - region must be a PLACE (e.g., Hepatocystic triangle, Gallbladder neck/infundibulum).

    Your job:
    A) Identify the role of each tool (left, right, camera) using ONLY what is visible.
    B) Describe START vs END state relevant to the criterion (brief).
        B2) Optionally describe ONE intermediate state (MID state) if it helps clarify tool roles,
        viewpoint changes, or the criterion transition.
    C) List the TOP {max_deltas} criterion-relevant deltas (ranked) with explicit actor/tool grounding.
    D) Output a RANKED LIST of 1-3 important actions (often BOTH retraction and dissection).
       - Do NOT force a single "final action".

    ABSOLUTE RULES (must follow):
    1) DO NOT GUESS ACTORS.
       - If you are not confident whether left or right tool caused an action/delta, set actor_side="unclear".
    2) NO ABSTRACT "TRACTION" CLAIMS.
       - If verb is Retract, you MUST specify:
         (a) actor_side (left/right/unclear),
         (b) item being retracted (often Gallbladder),
         (c) direction (from the direction vocab).
       - Do NOT write "increases traction" without grounding; use Retract with fields instead.
    3) Separate dissection vs retraction:
       - Dissect = clearing fat/tissue, separating planes, sweeping tissue.
       - Retract = pulling/lifting/holding tissue under tension to change exposure.
    4) For verb=="Dissect": item MUST NOT be Null and MUST NOT be a region.
       - Prefer item=Fat or Connective tissue when clearing tissue.
    5) Use ONLY controlled vocab for tool/verb/item/region/direction in structured fields.
       - Free text is allowed ONLY in "one_sentence" and "rationale".
    6) If mind_change is "no_change", actions_ranked MUST be [].
    7) If you cannot find any clear criterion-relevant action, actions_ranked MUST be [].
    8) IMPORTANT: If BOTH a stable retraction AND tissue clearing are present, include BOTH.
       - Ranking guidance: choose the action most responsible for the criterion flip as rank 1.
       - If uncertain between retraction vs dissection, prefer retraction as rank 1.
    9) VIEWPOINT / SURFACE VISIBILITY (IMPORTANT):
       - If the visible surface of the gallbladder changes (anterior ↔ posterior),
         explicitly mention it in start_state and end_state.
       - If the camera/tool rotation reveals a new surface, include this as a delta.
       - Only say "posterior view" if it is clearly visible; otherwise say "rotated view" or "new surface exposed".

    MID STATE RULE:
    - If any important tool, anatomy, or viewpoint is only visible in middle frames,
      you MUST include a mid_state.
    - mid_state should correspond to a single evidence_frame that is most informative.
    - Do NOT summarize all frames; only choose the most useful intermediate snapshot.


    Output format (STRICT JSON only; no markdown; no extra keys):
    {{
      "criterion": "{criterion.upper()}",
      "mind_change": "{mind_change}",

      "scene_flags": [],

      "tool_roles": {{
        "left":  {{"tool":"{TOOLS[0]}|...|Null","verb":"{VERBS[0]}|...|Null","item":"{ITEMS_PRIORITY[0]}|...|Null","region":"{REGIONS_PRIORITY[0]}|...|Unclear","direction":"{DIRECTIONS[0]}|...|unclear"}},
        "right": {{"tool":"{TOOLS[0]}|...|Null","verb":"{VERBS[0]}|...|Null","item":"{ITEMS_PRIORITY[0]}|...|Null","region":"{REGIONS_PRIORITY[0]}|...|Unclear","direction":"{DIRECTIONS[0]}|...|unclear"}},
        "camera": {{"tool":"Null","verb":"Null","item":"Null","region":"Unclear","direction":"unclear"}}
      }},

      "start_state": {{
        "what_is_visible": ["..."],
        "confidence": 0.0
      }},
      "end_state": {{
        "what_is_visible": ["..."],
        "confidence": 0.0
      }},

      "mid_state": {{
        "what_is_visible": ["..."],
        "evidence_frame": 0,
        "confidence": 0.0
      }},

      "deltas": [
        {{
          "delta": "...",
          "actor_side": "left|right|camera|unclear",
          "tool": "{TOOLS[0]}|...|Null",
          "verb": "{VERBS[0]}|...|Null",
          "item": "{ITEMS_PRIORITY[0]}|...|Null",
          "region": "{REGIONS_PRIORITY[0]}|...|Unclear",
          "direction": "{DIRECTIONS[0]}|...|unclear",
          "evidence_frame": {start_fid},
          "mechanism": "short, concrete, observable"
        }}
      ],

      "actions_ranked": [
        {{
          "rank": 1,
          "importance": 0.0,
          "actor_side": "left|right|camera|unclear",
          "tool": "{TOOLS[0]}|...|Null",
          "verb": "{VERBS[0]}|...|Null",
          "item": "{ITEMS_PRIORITY[0]}|...|Null",
          "region": "{REGIONS_PRIORITY[0]}|...|Unclear",
          "direction": "{DIRECTIONS[0]}|...|unclear",
          "one_sentence": "8-18 words, must match tool/verb/item/region/direction",
          "rationale": "one sentence mapping action -> criterion change",
          "confidence": 0.0
        }}
      ],

      {action_track_schema}
    }}

    CONSTRAINTS:
    - start_state.what_is_visible: 2-5 bullets, each <= 12 words.
    - end_state.what_is_visible: 2-5 bullets, each <= 12 words.
    - mid_state: null if not needed; otherwise must include what_is_visible (2-5 bullets),
      evidence_frame (one of the provided frame IDs, NOT the first or last), and confidence.
    - deltas: 1 to {max_deltas} items, ranked most important first.
    - evidence_frame MUST be one of the provided frame IDs.
    - actions_ranked: 0 to 3 items total.
      - If non-empty: ranks must be 1..K with no gaps.
      - importance and confidence are floats in [0,1].
    - If mind_change=="no_change": actions_ranked MUST be [].
    {action_track_instruction}
    """

    # ---- Build multimodal user message ----
    user_parts = [{
        "type": "text",
        "text": (
            f"Here are boundary-focused frames for the transition window (chronological). "
            f"Start frame is {start_fid}, end frame is {end_fid}."
        )
    }]

    def add_frame(fid):
        fpath = os.path.join(frame_dir, f"frame_{fid:06d}.png")
        if os.path.isfile(fpath):
            user_parts.append({"type": "text", "text": f"Frame {fid}:"})
            user_parts.append({"type": "image_url", "image_url": {"url": img_to_data_url(fpath)}})

    for fid in boundary_fids:
        add_frame(fid)

    # ---- Invoke LLM ----
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_parts)])
    content = response.content

    # Some models return a list of content blocks instead of a string
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return content.strip()
