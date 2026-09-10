#!/usr/bin/env python3
"""
Baseline CVS + action prediction script.

Processes a video's frames (every 5th key frame) with a single LLM call per
frame.  Each call predicts C1/C2/C3 scores AND a recommended action combo
using the surgical action taxonomy.  Video-level CVS is MAX(frame-level) per
criterion.

Four prompt presets:
  direct   – predict CVS + action directly (no rationale)
  cot      – predict CVS + action with chain-of-thought rationale
  subrubric – generate a self-rubric checklist, then predict CVS + action
  auto     – automatically gather ALL heuristics from surgent modules into one prompt

EXAMPLES
  python scripts/run/baseline.py --image-dir data/processed/CVS_Challenge_SAGES_v1/frames/train/00371a60-3206-40aa-862c-43826769e962
  python scripts/run/baseline.py --image-dir ... --preset cot
  python scripts/run/baseline.py --image-dir ... --preset subrubric --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PIL import Image  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from surgent.audit_simple_actions import normalize_actor_slot_actions  # noqa: E402
from surgent.shared_prompt_cores import (  # noqa: E402
    build_action_rec_no_guideline_prompt_body,
    build_action_rec_prompt_body,
    build_action_rec_taxonomy_only_prompt_body,
    build_baseline_bridge_instruction,
    build_baseline_combined_response_schema,
    build_scene_cvs_prompt_body,
    SYSTEM_PREAMBLE,
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

CRITERION_DEFINITIONS = {
    "C1": "Two and only two tubular structures are visible entering the gallbladder.",
    "C2": "The hepatocystic triangle is cleared of fat and fibrous tissue.",
    "C3": "The lower third of the gallbladder is detached from the liver bed.",
}

DEFAULT_TAXONOMY_PATH = (
    REPO_ROOT
    / "hf_repos"
    / "cvs-act"
    / "taxonomy"
    / "action_taxonomy.json"
)

TAXONOMY_FIELDS = (
    "actor_role",
    "tool_type",
    "action_code",
    "target_structure",
    "target_context",
    "intention",
)

SYSTEM_PROMPT = (
    SYSTEM_PREAMBLE
)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _default_qwen_base_url() -> Optional[str]:
    candidate_paths = [
        Path("/mnt/md0/weiqiuy/ips/carnaroli.txt"),
        Path("/shared_data0/weiqiuy/ips/carnaroli.txt"),
    ]
    for path in candidate_paths:
        try:
            if path.exists():
                host = path.read_text().strip()
                if host:
                    return f"http://{host}:8001/v1"
        except Exception:
            continue
    return None


def _init_model(model_id: str, temperature: float = 0.0) -> Any:
    """Initialise a langchain chat model, mirroring surgent's _init_model."""
    model_lower = str(model_id).lower()
    is_google_vertex = any(tok in model_lower for tok in ("gemini", "vertex", "google"))
    is_qwen = model_lower.startswith("qwen/")
    api_key: Optional[str]
    if is_google_vertex:
        api_key = None
    elif is_qwen:
        api_key = os.getenv("QWEN_API_KEY") or "brachiokey"
    elif "anthropic" in model_lower or "claude" in model_lower:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif "openai" in model_lower or "gpt" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    kwargs: Dict[str, Any] = {"temperature": temperature, "max_retries": 20}
    if "/" in str(model_id):
        inferred_provider = str(model_id).split("/", 1)[0].strip().lower()
        if inferred_provider:
            kwargs["model_provider"] = inferred_provider
    base_url: Optional[str] = None
    if is_qwen:
        base_url = os.getenv("QWEN_BASE_URL") or _default_qwen_base_url()
        kwargs["base_url"] = base_url
        kwargs["model_provider"] = "openai"
    if api_key:
        kwargs["api_key"] = api_key
    provider_hint = str(kwargs.get("model_provider", "")).strip().lower()
    if provider_hint == "openai" or model_lower.startswith("openai/"):
        openai_kwargs: Dict[str, Any] = {"temperature": temperature, "max_retries": 20}
        if base_url:
            openai_kwargs["base_url"] = base_url
        if api_key:
            openai_kwargs["api_key"] = api_key
        return ChatOpenAI(model=str(model_id), **openai_kwargs)
    return init_chat_model(model_id, **kwargs)


def _invoke_model(
    model: Any,
    prompt_text: str,
    images: List[Image.Image],
    system_prompt: str,
) -> str:
    """Invoke a langchain chat model with text + images, return raw text response."""
    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_parts),
    ])
    content = getattr(response, "content", "")
    return str(content) if not isinstance(content, str) else content


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _repair_truncated_json(text: str) -> Dict[str, Any]:
    t = text.rstrip()
    if t.endswith(","):
        t = t[:-1]
    in_string = False
    i = 0
    while i < len(t):
        ch = t[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        i += 1
    if in_string:
        t += '"'
    stack: list[str] = []
    in_str = False
    i = 0
    while i < len(t):
        ch = t[i]
        if ch == "\\" and in_str:
            i += 2
            continue
        if ch == '"':
            in_str = not in_str
            i += 1
            continue
        if in_str:
            i += 1
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack:
                stack.pop()
        i += 1
    t += "".join(reversed(stack))
    return json.loads(t)


def extract_json_obj(text: str) -> Dict[str, Any]:
    t = _strip_code_fences(text or "")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return _repair_truncated_json(t)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def sanitize_tag(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._") or "value"


# ──────────────────────────────────────────────────────────────
# Taxonomy loading  (reuses logic from run_surgent_sequential.py)
# ──────────────────────────────────────────────────────────────

def load_taxonomy(path: Path) -> Dict[str, Any]:
    """Load taxonomy JSON and return options dict per field."""
    catalog: Dict[str, Any] = {
        "path": str(path),
        "version": "unknown",
        "options": {f: [] for f in TAXONOMY_FIELDS},
    }
    if not path.exists():
        return catalog
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return catalog
    if isinstance(payload, dict) and isinstance(payload.get("actors"), dict):
        catalog["version"] = str(payload.get("taxonomy_version") or "unknown")
        actors = payload.get("actors", {})
        right_components = (
            actors.get("right", {}).get("components", {})
            if isinstance(actors.get("right"), dict)
            else {}
        )
        left_components = (
            actors.get("left", {}).get("components", {})
            if isinstance(actors.get("left"), dict)
            else {}
        )
        camera_components = (
            actors.get("camera", {}).get("components", {})
            if isinstance(actors.get("camera"), dict)
            else {}
        )
        catalog["options"]["actor_role"] = [
            "camera",
            "left_instrument",
            "right_instrument",
            "other",
        ]
        catalog["options"]["tool_type"] = list(right_components.get("tool_type", []))
        if "camera" not in {str(x).lower() for x in catalog["options"]["tool_type"]}:
            catalog["options"]["tool_type"].append("camera")
        action_codes = (
            list(right_components.get("action_code", []))
            + list(camera_components.get("action_code", []))
            + list(left_components.get("retraction_direction_code", []))
        )
        catalog["options"]["action_code"] = list(dict.fromkeys(action_codes))
        catalog["options"]["target_structure"] = list(right_components.get("target_structure", []))
        catalog["options"]["target_context"] = list(right_components.get("target_context", []))
        catalog["options"]["intention"] = []
        catalog["intention_descs"] = {}
        catalog["recommendation_descs"] = {f: {} for f in TAXONOMY_FIELDS}
        catalog["field_meta"] = {}
        return catalog
    meta = payload.get("_meta", {})
    if isinstance(meta, dict):
        catalog["version"] = str(meta.get("version") or "unknown")
    fields = payload.get("fields", {})
    if isinstance(fields, dict):
        for field, entries in fields.items():
            if field not in catalog["options"] or not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and "value" in entry:
                    catalog["options"][field].append(str(entry["value"]))
    intentions = payload.get("intentions", [])
    if isinstance(intentions, list):
        for entry in intentions:
            if isinstance(entry, dict) and "value" in entry:
                val = str(entry["value"])
                if val not in catalog["options"]["intention"]:
                    catalog["options"]["intention"].append(val)
    # Build intention descriptions
    intention_descs: Dict[str, str] = {}
    if isinstance(intentions, list):
        for entry in intentions:
            if isinstance(entry, dict):
                val = str(entry.get("value", ""))
                desc = str(entry.get("description", ""))
                if val and desc:
                    intention_descs[val] = desc
    catalog["intention_descs"] = intention_descs

    # Build per-value recommendation descriptions (v10+)
    rec_descs: Dict[str, Dict[str, str]] = {f: {} for f in TAXONOMY_FIELDS}
    if isinstance(fields, dict):
        for field, entries in fields.items():
            if field not in rec_descs or not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    val = str(entry.get("value", ""))
                    desc = str(entry.get("recommendation_description", ""))
                    if val and desc:
                        rec_descs[field][val] = desc
    catalog["recommendation_descs"] = rec_descs

    # Build per-field recommendation descriptions (field_meta / field_guidelines)
    field_meta_out: Dict[str, str] = {}
    for key in ("field_meta", "field_guidelines"):
        fm_block = payload.get(key, {})
        if isinstance(fm_block, dict) and fm_block:
            for field in TAXONOMY_FIELDS:
                entry = fm_block.get(field, {})
                if isinstance(entry, dict):
                    desc = str(entry.get("recommendation_description", "")).strip()
                    if desc and field not in field_meta_out:
                        field_meta_out[field] = desc
            if field_meta_out:
                break
    catalog["field_meta"] = field_meta_out

    return catalog


def _taxonomy_options_block(catalog: Dict[str, Any]) -> str:
    """Build a human-readable block of allowed taxonomy values for the prompt."""
    options = catalog.get("options", {})
    intention_descs = catalog.get("intention_descs", {})
    rec_descs = catalog.get("recommendation_descs", {})
    field_meta = catalog.get("field_meta", {})
    has_rec_descs = any(bool(d) for d in rec_descs.values()) if rec_descs else False
    lines = []
    for field in TAXONOMY_FIELDS:
        vals = options.get(field, [])
        field_rec = rec_descs.get(field, {}) if has_rec_descs else {}
        field_int = intention_descs if field == "intention" else {}
        fm_desc = field_meta.get(field, "") if isinstance(field_meta, dict) else ""
        # Use rich per-value format when recommendation descriptions are available
        if field_rec or field_int:
            if fm_desc:
                lines.append(f"{field}: {fm_desc}")
            else:
                lines.append(f"{field}:")
            for v in vals:
                desc = field_rec.get(v, "") or field_int.get(v, "")
                if desc:
                    lines.append(f"  - {v}: {desc}")
                else:
                    lines.append(f"  - {v}")
        else:
            if fm_desc:
                lines.append(f"{field}: {fm_desc}")
                for v in vals:
                    lines.append(f"  - {v}")
            else:
                lines.append(f"  {field}: {json.dumps(vals)}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Frame collection
# ──────────────────────────────────────────────────────────────

def _frame_sort_key(path: Path) -> Tuple[int, Any, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), stem)
    match = re.search(r"(\d+)", stem)
    if match:
        return (1, int(match.group(1)), stem)
    return (2, stem, stem)


def collect_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise SystemExit(f"image_dir not found or not a directory: {image_dir}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=_frame_sort_key)
    return paths


def select_key_frames(paths: List[Path], every_n: int = 5) -> List[Path]:
    """Select every N-th frame (key frames)."""
    return paths[::every_n]


# ──────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────

def _cvs_definition_block() -> str:
    lines = []
    for c in ("C1", "C2", "C3"):
        lines.append(f"- {c}: {CRITERION_DEFINITIONS[c]}")
    return "\n".join(lines)


def _action_schema_block(fixed_k: Optional[int] = None) -> str:
    count_comment = ""
    if fixed_k is not None:
        count_comment = f"  // exactly {fixed_k} action objects, ranked 1 to {fixed_k}\n"
    return (
        '    "actions_ranked": [\n'
        f"{count_comment}"
        "      {\n"
        '        "rank": 1,\n'
        '        "actor_role": "<from allowed actor_role>",\n'
        '        "tool_type": "<from allowed tool_type>",\n'
        '        "action_code": "<from allowed action_code>",\n'
        '        "target_structure": "<from allowed target_structure>",\n'
        '        "target_context_1": "<from allowed target_context or (not set)>",\n'
        '        "target_context_2": "<from allowed target_context or (not set)>",\n'
        '        "intention": "<from allowed intention>",\n'
        '        "confidence": 0.0\n'
        "      }\n"
        "    ]"
    )


def _fixed_k_constraint_block(k: int) -> str:
    """Build the fixed-k constraint instruction block."""
    return (
        f"FIXED-K CONSTRAINT:\n"
        f"You MUST provide exactly {k} actions in actions_ranked.\n"
        f"Rank them from most appropriate (rank 1) to least appropriate (rank {k}).\n"
        f"Do not output more or fewer than {k}.\n"
        f"Each action must be a concrete surgical action (include actor, motion, and target when applicable).\n"
        f"Do not include explanations or extra text.\n"
    )


def _enforce_fixed_k(actions: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Pad or truncate actions to exactly k entries."""
    slot_actor_roles = {"right_instrument", "left_instrument", "camera", "other"}
    if any(str(a.get("actor_role", "")) in slot_actor_roles for a in actions):
        return normalize_actor_slot_actions(actions)
    result = actions[:k]
    while len(result) < k:
        rank = len(result) + 1
        result.append({
            "rank": rank,
            "actor_role": "(not set)",
            "tool_type": "(not set)",
            "action_code": "(not set)",
            "target_structure": "(not set)",
            "target_context_1": "(not set)",
            "target_context_2": "(not set)",
            "intention": "(not set)",
            "confidence": 0.0,
        })
    for i, a in enumerate(result):
        a["rank"] = i + 1
    return result


def _causal_frame_intro(num_frames: int, current_frame_id: str) -> str:
    """Build a frame intro block for causal (multi-frame) mode."""
    if num_frames == 1:
        return "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
    return (
        f"You are viewing {num_frames} frames from a laparoscopic cholecystectomy video, "
        f"presented in chronological order.\n"
        f"The LAST image (frame {current_frame_id}) is the CURRENT evaluation frame.\n"
        f"Earlier frames provide temporal context showing how the surgical scene evolved.\n"
        f"All predictions (CVS scores and actions) must reflect the state at the CURRENT "
        f"(last) frame. Use earlier frames only as context.\n"
    )


def _apply_causal_intro(prompt_text: str, num_frames: int, current_frame_id: str) -> str:
    """Replace the single-frame intro in a prompt with a causal multi-frame intro."""
    single_frame_marker = "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
    causal_intro = _causal_frame_intro(num_frames, current_frame_id)
    if single_frame_marker in prompt_text:
        return prompt_text.replace(single_frame_marker, causal_intro, 1)
    # If the marker is not found, prepend the causal intro
    return causal_intro + "\n" + prompt_text


def _video_level_intro(frame_paths: List[Path]) -> str:
    """Build a multi-frame intro block for video-level (all-frames-at-once) mode.

    Replaces the single-frame marker in the base prompt so that the model
    knows it is seeing ALL key frames and should reason across them.
    """
    n = len(frame_paths)
    # Frame reference table: idx -> frame_id (stem)
    ref_lines = [f"  idx={i}: {p.stem}" for i, p in enumerate(frame_paths)]
    ref_table = "\n".join(ref_lines)

    return (
        f"You are viewing {n} temporally ordered key frames from a laparoscopic "
        f"cholecystectomy video, spaced approximately 5 seconds apart.\n"
        "\n"
        "Frame reference table:\n"
        f"{ref_table}\n"
        "\n"
        "VIDEO-LEVEL REASONING GUIDANCE:\n"
        "- Assess each CVS criterion across ALL frames, not just a single one.\n"
        "- Once a criterion is clearly satisfied in any frame, it remains satisfied\n"
        "  for the video even if later frames show occlusion or a different camera angle.\n"
        "- Transient or ambiguous appearances in a single frame do not constitute\n"
        "  satisfaction — look for consistent, clear evidence.\n"
        "- Reason about the temporal progression: criteria may become satisfied as\n"
        "  the surgery progresses (e.g., dissection reveals structures over time).\n"
        "- When citing evidence, reference frames by their ID from the table above.\n"
        "- Your predictions (CVS scores and actions) should reflect the VIDEO-LEVEL\n"
        "  assessment, not any single frame.\n"
    )


def _apply_video_level_intro(prompt_text: str, frame_paths: List[Path]) -> str:
    """Replace the single-frame intro in a prompt with a video-level multi-frame intro."""
    single_frame_marker = "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
    video_intro = _video_level_intro(frame_paths)
    if single_frame_marker in prompt_text:
        return prompt_text.replace(single_frame_marker, video_intro, 1)
    return video_intro + "\n" + prompt_text


def _action_task_block(action_mode: str, taxonomy_catalog: Dict[str, Any], fixed_k: Optional[int] = None) -> str:
    """Build the action task text + IMPORTANT block for the given action mode."""
    if action_mode == "predict":
        if fixed_k is not None:
            task = (
                f"Task 2: Predict exactly {fixed_k} most likely current surgical actions visible in this frame.\n"
                f"Rank them from most likely (1) to least likely ({fixed_k}).\n"
                "Use ONLY values from the allowed taxonomy below.\n"
                'Use "(not set)" for unknown fields.\n'
            )
        else:
            task = (
                "Task 2: Predict the most likely current surgical action(s) visible in this frame.\n"
                "Use ONLY values from the allowed taxonomy below.\n"
                'Use "(not set)" for unknown fields.\n'
            )
        important = (
            "IMPORTANT:\n"
            "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
            "- actor_role must match the tool side visible in the image.\n"
        )
    else:  # recommend
        if fixed_k is not None:
            task = (
                f"Task 2: Recommend exactly {fixed_k} next surgical actions the surgeon should perform.\n"
                f"Rank them from most appropriate (1) to least appropriate ({fixed_k}).\n"
                "Use ONLY values from the allowed taxonomy below.\n"
                'Use "(not set)" for unknown fields.\n'
            )
        else:
            task = (
                "Task 2: Recommend the next surgical action(s) the surgeon should perform to\n"
                "increase CVS satisfaction for criteria that are NOT yet satisfied.\n"
                "Based on your CVS scores from Task 1, identify which criteria are unsatisfied and\n"
                "recommend 1-3 ranked actions that would most effectively improve them.\n"
                "Use ONLY values from the allowed taxonomy below.\n"
                'Use "(not set)" for unknown fields. If all criteria are already satisfied,\n'
                "return actions_ranked as [].\n"
            )
        task += (
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
        )
        important = (
            "IMPORTANT:\n"
            "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
            "- actor_role should indicate which actor performs the recommended action\n"
            "  (left_instrument for retraction, right_instrument for dissection, camera for repositioning).\n"
            "- Address all unsatisfied CVS criteria, not just the lowest-scoring one.\n"
            "  Each action's intention should reflect which specific criterion it targets.\n"
        )
    if fixed_k is not None:
        important += _fixed_k_constraint_block(fixed_k)
    return (
        f"{task}\n"
        "Allowed taxonomy values:\n"
        f"{_taxonomy_options_block(taxonomy_catalog)}\n"
        "\n"
        f"{important}"
    )


def _action_step_block(action_mode: str, step_label: str = "STEP 3", fixed_k: Optional[int] = None) -> str:
    """Build the action step text for subrubric/auto presets."""
    if action_mode == "predict":
        if fixed_k is not None:
            return (
                f"{step_label} — Action prediction:\n"
                f"Predict exactly {fixed_k} most likely current surgical actions visible in this frame.\n"
                f"Rank them from most likely (1) to least likely ({fixed_k}).\n"
                "Use ONLY values from the allowed taxonomy below.\n"
                'Use "(not set)" for unknown fields.\n'
            )
        return (
            f"{step_label} — Action prediction:\n"
            "Predict the most likely current surgical action(s) visible in this frame.\n"
            "Use ONLY values from the allowed taxonomy below.\n"
            'Use "(not set)" for unknown fields.\n'
        )
    else:  # recommend
        if fixed_k is not None:
            return (
                f"{step_label} — Action recommendation:\n"
                f"Recommend exactly {fixed_k} next surgical actions the surgeon should perform.\n"
                f"Rank them from most appropriate (1) to least appropriate ({fixed_k}).\n"
                "Use ONLY values from the allowed taxonomy below.\n"
                'Use "(not set)" for unknown fields.\n'
            )
        return (
            f"{step_label} — Action recommendation:\n"
            "Based on your CVS scores, recommend the next surgical action(s) the surgeon\n"
            "should perform to increase CVS satisfaction for criteria that are NOT yet satisfied.\n"
            "Identify which criteria are unsatisfied and recommend 1-3 ranked actions that would\n"
            "most effectively improve them.\n"
            "Use ONLY values from the allowed taxonomy below.\n"
            'Use "(not set)" for unknown fields. If all criteria are already satisfied,\n'
            "return actions_ranked as [].\n"
        )


def _action_important_block(action_mode: str, fixed_k: Optional[int] = None) -> str:
    """Build the IMPORTANT block for subrubric/auto presets."""
    if action_mode == "predict":
        block = (
            "IMPORTANT:\n"
            "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
            "- actor_role must match the tool side visible in the image.\n"
        )
    else:  # recommend
        block = (
            "IMPORTANT:\n"
            "- Use ONLY controlled vocab values from the allowed taxonomy.\n"
            "- actor_role should indicate which actor performs the recommended action\n"
            "  (left_instrument for retraction, right_instrument for dissection, camera for repositioning).\n"
            "- Address all unsatisfied CVS criteria, not just the lowest-scoring one.\n"
            "  Each action's intention should reflect which specific criterion it targets.\n"
        )
    if fixed_k is not None:
        block += _fixed_k_constraint_block(fixed_k)
    return block


def build_direct_prompt(
    taxonomy_catalog: Dict[str, Any],
    *,
    include_rationale: bool,
    action_mode: str = "recommend",
    fixed_k: Optional[int] = None,
    action_rec_rules: str = "default",
) -> str:
    """Build a prompt for the 'direct' or 'cot' preset."""
    rationale_schema = ""
    rationale_instruction = ""
    if include_rationale:
        rationale_schema = '  "rationale": { "c1": "<string>", "c2": "<string>", "c3": "<string>" },\n'
        rationale_instruction = (
            "\n- Provide a short rationale per CVS criterion before giving scores."
            "\n- Output rationale before pred."
        )

    if action_mode == "recommend":
        action_block = build_action_rec_prompt_body(
            taxonomy_catalog=taxonomy_catalog,
            cvs_status_block="",
            fixed_k=fixed_k,
            action_rec_rules=action_rec_rules,
        )
        cvs_block = build_scene_cvs_prompt_body(preset="cot" if include_rationale else "direct")
        prompt = (
            f"{cvs_block}"
            "\n"
            f"{action_block}\n"
            f"{build_baseline_bridge_instruction()}\n"
            "Combined output schema:\n"
            f"{build_baseline_combined_response_schema()}"
        )
        return prompt

    action_block = _action_task_block(action_mode, taxonomy_catalog, fixed_k=fixed_k)

    prompt = (
        "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
        "\n"
        "Task 1: Predict Critical View of Safety (CVS) confidence scores.\n"
        "Return a probability in [0,1] for each criterion.\n"
        "\n"
        "Criterion definitions:\n"
        f"{_cvs_definition_block()}\n"
        "\n"
        f"{action_block}"
        f"{rationale_instruction}\n"
        "\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        "{\n"
        f"{rationale_schema}"
        '  "pred": { "c1": float, "c2": float, "c3": float },\n'
        f"{_action_schema_block(fixed_k=fixed_k)}\n"
        "}\n"
    )
    return prompt


def build_subrubric_prompt(
    taxonomy_catalog: Dict[str, Any],
    *,
    action_mode: str = "recommend",
    fixed_k: Optional[int] = None,
) -> str:
    """Build a single-call prompt for the 'subrubric' preset.

    The model generates a self-rubric checklist AND then uses it to predict
    CVS scores + actions, all in one call.
    """
    action_step = _action_step_block(action_mode, step_label="STEP 3", fixed_k=fixed_k)
    action_important = _action_important_block(action_mode, fixed_k=fixed_k)

    prompt = (
        "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
        "\n"
        "STEP 1 — Self-rubric checklist:\n"
        "For each CVS criterion (C1, C2, C3), generate diagnostic questions that\n"
        "help assess whether the criterion is met. Answer each question as\n"
        "yes/no/uncertain based only on the image.\n"
        "\n"
        "Criterion definitions:\n"
        f"{_cvs_definition_block()}\n"
        "\n"
        "STEP 2 — CVS prediction:\n"
        "Using the checklist answers as evidence, predict confidence scores\n"
        "in [0,1] for C1, C2, C3.\n"
        "\n"
        f"{action_step}\n"
        "Allowed taxonomy values:\n"
        f"{_taxonomy_options_block(taxonomy_catalog)}\n"
        "\n"
        f"{action_important}\n"
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
    return prompt


# ──────────────────────────────────────────────────────────────
# Auto prompt builder: collect ALL surgent heuristics
# ──────────────────────────────────────────────────────────────

def _collect_surgent_heuristics() -> str:
    """Programmatically gather all heuristics from surgent modules."""
    from surgent.controller import _RUBRIC_ITEMS, _RUBRIC_LABEL_TYPE_DEFS
    from surgent.schemas import CRITERION_DEFINITIONS

    sections: List[str] = []

    # ── 1. Rubric items with weights and label types ──
    rubric_lines = ["DETAILED RUBRIC (use these to assess each criterion):"]
    rubric_lines.append("")
    rubric_lines.append("Label type definitions:")
    for lt, desc in _RUBRIC_LABEL_TYPE_DEFS.items():
        rubric_lines.append(f"  - {lt}: {desc}")
    rubric_lines.append("")
    for criterion_key in ("C1", "C2", "C3"):
        defn = CRITERION_DEFINITIONS.get(criterion_key, "")
        rubric_lines.append(f"{criterion_key}: {defn}")
        for item in _RUBRIC_ITEMS.get(criterion_key, []):
            rubric_lines.append(
                f"  - {item['id']} [{item['label_type']}, weight={item['weight']}]: {item['text']}"
            )
        rubric_lines.append("")
    sections.append("\n".join(rubric_lines))

    # ── 2. Score thresholds ──
    sections.append(
        "SCORE THRESHOLDS:\n"
        "- score >= 0.7 → satisfied\n"
        "- score <= 0.3 → unsatisfied\n"
        "- 0.3 < score < 0.7 → uncertain\n"
        "- Video-level rule: if any clearly visible frame provides decisive evidence that\n"
        "  the criterion is satisfied, classify the video as satisfied, even if other frames\n"
        "  are ambiguous, obscured, or not assessable."
    )

    # ── 3. Left instrument heuristics ──
    sections.append(
        "LEFT INSTRUMENT HEURISTICS (almost always a Grasper for retraction/exposure):\n"
        "- Retraction direction matters:\n"
        "  * RETRACT_LATERAL: Pull gallbladder leftward (from camera perspective) to expose medial HCT\n"
        "  * RETRACT_MEDIAL: Pull rightward to expose lateral aspect\n"
        "  * RETRACT_UPWARD: Pull cephalad to lift gallbladder off liver bed\n"
        "  * RETRACT_MEDIAL_TO_LATERAL / RETRACT_LATERAL_TO_MEDIAL: Rotation between positions\n"
        "  * RETRACT_MAINTAIN: Current retraction is adequate, maintain position\n"
        "  * RETRACT_DOWNWARD: Rare, pull inferiorly\n"
        "- If the gallbladder is already well-retracted and the hepatocystic triangle is\n"
        "  well-exposed for the target criterion, recommend RETRACT_MAINTAIN.\n"
        "- If exposure needs to change, recommend the direction that would best serve the\n"
        "  unsatisfied CVS criteria."
    )

    # ── 4. Right instrument heuristics ──
    sections.append(
        "RIGHT INSTRUMENT HEURISTICS (active/working tool — Hook, Maryland, Scissors, Irrigator):\n"
        "- The right instrument performs the active surgical work: dissection, cutting, cautery, aspiration.\n"
        "- Target depends on which CVS criterion needs most improvement:\n"
        "  * C1 (two structures) / C2 (triangle clearance) → dissect in hepatocystic triangle area.\n"
        "    Specify precise target_context (between cystic duct and artery, near artery, near duct, etc.)\n"
        "  * C3 (lower third detachment) → dissect cystic plate / gallbladder-liver interface.\n"
        "- When cystic duct and artery are partially visible, specify precise target_context.\n"
        "- If no active work is visible or needed, return action_code=NO_ACTION."
    )

    # ── 5. Camera heuristics ──
    sections.append(
        "CAMERA HEURISTICS:\n"
        "- Field of view too far / structures appear small → CAMERA_ZOOM_IN\n"
        "- View too close / cannot see the whole operative field → CAMERA_ZOOM_OUT\n"
        "- Hepatocystic triangle not centered or not visible → CAMERA_REPOSITION\n"
        "- View is adequate for assessing CVS criteria → NO_ACTION (no camera change needed)"
    )

    # ── 6. Action prioritization ──
    sections.append(
        "ACTION PRIORITIZATION:\n"
        "- Address all unsatisfied criteria, not just the lowest-scoring one.\n"
        "- Each action's intention should reflect which specific criterion it targets.\n"
        "- Include BOTH retraction and dissection actions if both are needed.\n"
        "- Rank actions by importance: the action most likely to advance any unsatisfied criterion first.\n"
        "- Primary objective: recommend actions that maximize progress toward satisfying all\n"
        "  criteria together, not repeatedly re-validating already high-score criteria."
    )

    # ── 7. Key anatomical context ──
    sections.append(
        "ANATOMICAL CONTEXT:\n"
        "- Key structures: gallbladder, liver, liver bed (cystic plate), cystic duct,\n"
        "  cystic artery, hepatocystic triangle (Calot's triangle), common hepatic duct,\n"
        "  inferior liver edge, gallbladder neck (infundibulum), peritoneum.\n"
        "- Surgical instruments: Grasper (retraction), Hook (cautery/dissection),\n"
        "  Maryland (dissection), Scissors (cutting), Irrigator (aspiration).\n"
        "- Instrument interactions: traction, dissection, clipping, cautery, cutting."
    )

    return "\n\n".join(sections)


def build_auto_prompt(
    taxonomy_catalog: Dict[str, Any],
    *,
    action_mode: str = "recommend",
    fixed_k: Optional[int] = None,
) -> str:
    """Build the 'auto' preset prompt: all surgent heuristics in one prompt."""
    heuristics = _collect_surgent_heuristics()
    action_step = _action_step_block(action_mode, step_label="STEP 3", fixed_k=fixed_k)
    action_important = _action_important_block(action_mode, fixed_k=fixed_k)

    # For auto+recommend, add extra actor guidance
    extra_actor_guidance = ""
    if action_mode == "recommend":
        extra_actor_guidance = (
            "Consider all three actors:\n"
            "  - LEFT instrument (retraction direction)\n"
            "  - RIGHT instrument (dissection/cutting target)\n"
            "  - Camera (zoom/reposition if needed)\n"
            "Use the prioritization heuristics above. Address all unsatisfied criteria.\n"
        )

    prompt = (
        "You are viewing a single frame from a laparoscopic cholecystectomy video.\n"
        "\n"
        "You have access to the full knowledge base of a specialized surgical vision system.\n"
        "Use ALL of the following heuristics to make your assessment.\n"
        "\n"
        "═══════════════════════════════════════════════════════\n"
        "SURGENT KNOWLEDGE BASE\n"
        "═══════════════════════════════════════════════════════\n"
        "\n"
        f"{heuristics}\n"
        "\n"
        "═══════════════════════════════════════════════════════\n"
        "TASKS\n"
        "═══════════════════════════════════════════════════════\n"
        "\n"
        "STEP 1 — Rubric-guided assessment:\n"
        "For each CVS criterion (C1, C2, C3), evaluate EVERY rubric item above.\n"
        "For each item, answer yes/no/uncertain based on the image and note the weight.\n"
        "Consider label types: evidence items are most important, preconditions must be met\n"
        "for evidence to be meaningful, and modifiers affect confidence.\n"
        "\n"
        "STEP 2 — CVS prediction:\n"
        "Using the rubric answers and weights as evidence, predict confidence scores\n"
        "in [0,1] for C1, C2, C3. Weight your assessment by the rubric item weights.\n"
        "\n"
        f"{action_step}"
        f"{extra_actor_guidance}\n"
        "Allowed taxonomy values:\n"
        f"{_taxonomy_options_block(taxonomy_catalog)}\n"
        "\n"
        f"{action_important}\n"
        "Return ONLY valid JSON (no markdown, no extra keys) with this schema:\n"
        "{\n"
        '  "checklist": {\n'
        '    "C1": [ {"id": "<rubric item id>", "question": "<rubric text>", "answer": "yes"|"no"|"uncertain", "weight": <float>}, ... ],\n'
        '    "C2": [ ... ],\n'
        '    "C3": [ ... ]\n'
        "  },\n"
        '  "rationale": { "c1": "<reasoning from rubric>", "c2": "<reasoning>", "c3": "<reasoning>" },\n'
        '  "pred": { "c1": float, "c2": float, "c3": float },\n'
        f"{_action_schema_block(fixed_k=fixed_k)}\n"
        "}\n"
    )
    return prompt


# ──────────────────────────────────────────────────────────────
# Output parsing / validation
# ──────────────────────────────────────────────────────────────

def parse_frame_output(raw_text: str) -> Dict[str, Any]:
    """Parse LLM output JSON and return structured dict."""
    parsed = extract_json_obj(raw_text)
    return parsed


def extract_pred(parsed: Dict[str, Any]) -> Dict[str, Optional[float]]:
    pred_raw = parsed.get("pred", {})
    if not isinstance(pred_raw, dict):
        return {"c1": None, "c2": None, "c3": None}
    out: Dict[str, Optional[float]] = {}
    for c in ("c1", "c2", "c3"):
        val = pred_raw.get(c)
        try:
            out[c] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            out[c] = None
    return out


def extract_actions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = parsed.get("actions", parsed.get("actions_ranked", []))
    if not isinstance(actions, list):
        return []
    cleaned = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        cleaned.append({
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
    slot_actor_roles = {"right_instrument", "left_instrument", "camera", "other"}
    if any(a["actor_role"] in slot_actor_roles for a in cleaned):
        return normalize_actor_slot_actions(cleaned)
    return cleaned


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline CVS + action prediction per frame (single LLM call)."
    )
    p.add_argument(
        "--image-dir",
        required=True,
        help="Directory of ordered frame images for one video.",
    )
    p.add_argument(
        "--preset",
        choices=["direct", "cot", "subrubric", "auto"],
        default="direct",
        help="Prompt preset: direct (no rationale), cot (with rationale), subrubric (self-rubric + predict), auto (all surgent heuristics).",
    )
    p.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model ID for the LLM (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--taxonomy-json",
        default=str(DEFAULT_TAXONOMY_PATH),
        help="Path to taxonomy JSON (default: taxonomy_v10.json).",
    )
    p.add_argument(
        "--keyframe-step",
        type=int,
        default=5,
        help="Select every N-th frame as a key frame (default: 5).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of key frames to process.",
    )
    p.add_argument(
        "--out-jsonl",
        default=None,
        help="Output JSONL path. Default: outputs/baseline/<video_id>__<preset>__<model>.jsonl",
    )
    p.add_argument(
        "--trace-dir",
        default="outputs/traces_baseline",
        help="Base directory for trace output.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )
    p.add_argument(
        "--action-mode",
        choices=["predict", "recommend"],
        default="recommend",
        help="Action mode: predict (describe current visible action) or recommend (suggest next action to improve CVS). Default: recommend.",
    )
    p.add_argument(
        "--causal",
        action="store_true",
        help="Causal multi-frame mode: for each key frame, pass ALL preceding key frames "
             "as context images. The model predicts CVS and actions for the CURRENT (last) frame.",
    )
    p.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help="If specified, force exactly this many ranked actions at each key frame "
             "(e.g. --fixed-k 3 outputs exactly 3 actions per frame).",
    )
    p.add_argument(
        "--action-rec-rules",
        choices=["default", "conservative_visible"],
        default="default",
        help=(
            "Action-rec prompt rule variant. Use conservative_visible to require "
            "visible right-instrument evidence and conservative camera moves."
        ),
    )
    p.add_argument(
        "--action-prompt-variant",
        choices=["default", "no_cvs", "no_cvs_no_desc", "no_cvs_no_guideline"],
        default="default",
        help=(
            "Action recommendation prompt variant. default combines CVS scoring "
            "with action recommendation; no_cvs runs action recommendation only; "
            "no_cvs_no_guideline runs action recommendation only with actor-wise "
            "taxonomy sections but without surgical guideline prose; no_cvs_no_desc "
            "runs action recommendation only with taxonomy/schema but without "
            "surgical guideline prose."
        ),
    )
    p.add_argument(
        "--no-rec-descs",
        action="store_true",
        help="Strip recommendation_description from taxonomy block (use terse value-only lists). "
             "Useful for A/B testing whether per-value descriptions help or hurt.",
    )
    p.add_argument(
        "--include-field-meta",
        action="store_true",
        help="Include per-field recommendation_description guidance from taxonomy (default: off).",
    )
    p.add_argument(
        "--video-level",
        action="store_true",
        help="Video-level mode: send all key frames in a single LLM call "
             "for direct video-level CVS prediction (instead of per-frame).",
    )
    p.add_argument(
        "--dryrun",
        action="store_true",
        help="Print prompt for first frame and exit without calling the model.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    args = parse_args()
    image_dir = Path(args.image_dir)
    video_id = image_dir.name

    # Collect and subsample frames
    all_frames = collect_image_paths(image_dir)
    if not all_frames:
        raise SystemExit(f"No images found in {image_dir}")
    key_frames = select_key_frames(all_frames, every_n=max(1, args.keyframe_step))
    if args.max_frames is not None:
        key_frames = key_frames[: max(0, args.max_frames)]
    if not key_frames:
        raise SystemExit("No key frames selected after filtering.")
    causal = args.causal
    print(
        f"Video: {video_id}  |  total frames: {len(all_frames)}  |  key frames: {len(key_frames)}"
        + (f"  |  causal=True" if causal else "")
    )

    # Load taxonomy
    taxonomy_path = Path(args.taxonomy_json)
    taxonomy_catalog = load_taxonomy(taxonomy_path)
    if args.no_rec_descs:
        taxonomy_catalog["recommendation_descs"] = {}
    if not args.include_field_meta:
        taxonomy_catalog["field_meta"] = {}
    extra = ""
    if args.no_rec_descs:
        extra += " (rec-descs stripped)"
    if args.include_field_meta:
        extra += " (field-meta included)"
    print(f"Taxonomy: {taxonomy_catalog['version']} from {taxonomy_path}{extra}")

    # Build base prompt (will be adapted per-frame in causal mode)
    preset = args.preset
    action_mode = args.action_mode
    fixed_k = args.fixed_k
    action_rec_rules = args.action_rec_rules
    action_prompt_variant = args.action_prompt_variant
    if action_rec_rules != "default":
        print(f"Action-rec rules: {action_rec_rules}")
    if action_prompt_variant != "default":
        print(f"Action prompt variant: {action_prompt_variant}")
    if action_prompt_variant == "no_cvs":
        base_prompt = build_action_rec_prompt_body(
            taxonomy_catalog=taxonomy_catalog,
            cvs_status_block="",
            fixed_k=fixed_k,
            action_rec_rules=action_rec_rules,
        )
    elif action_prompt_variant == "no_cvs_no_guideline":
        base_prompt = build_action_rec_no_guideline_prompt_body(
            taxonomy_catalog=taxonomy_catalog,
            fixed_k=fixed_k,
        )
    elif action_prompt_variant == "no_cvs_no_desc":
        base_prompt = build_action_rec_taxonomy_only_prompt_body(
            taxonomy_catalog=taxonomy_catalog,
            fixed_k=fixed_k,
        )
    elif preset == "direct":
        base_prompt = build_direct_prompt(taxonomy_catalog, include_rationale=False, action_mode=action_mode, fixed_k=fixed_k, action_rec_rules=action_rec_rules)
    elif preset == "cot":
        base_prompt = build_direct_prompt(taxonomy_catalog, include_rationale=True, action_mode=action_mode, fixed_k=fixed_k, action_rec_rules=action_rec_rules)
    elif preset == "subrubric":
        base_prompt = build_subrubric_prompt(taxonomy_catalog, action_mode=action_mode, fixed_k=fixed_k)
    elif preset == "auto":
        base_prompt = build_auto_prompt(taxonomy_catalog, action_mode=action_mode, fixed_k=fixed_k)
    else:
        raise SystemExit(f"Unknown preset: {preset}")

    # Preset tag for output paths: includes action_mode, fixed_k, and _causal suffix when set
    preset_tag = preset
    if action_mode == "predict":
        preset_tag = f"{preset_tag}_predict"
    if fixed_k is not None:
        preset_tag = f"{preset_tag}_fixedk{fixed_k}"
    if action_prompt_variant != "default":
        preset_tag = f"{preset_tag}_{action_prompt_variant}"
    if causal:
        preset_tag = f"{preset_tag}_causal"
    if args.video_level:
        preset_tag = f"{preset_tag}_videolevel"
    if args.no_rec_descs:
        preset_tag = f"{preset_tag}_norecdescs"
    if args.include_field_meta:
        preset_tag = f"{preset_tag}_fmeta"
    if action_rec_rules != "default":
        preset_tag = f"{preset_tag}_arecrules-{sanitize_tag(action_rec_rules).replace('_', '-')}"

    # Dryrun
    if args.dryrun:
        print("===== SYSTEM PROMPT =====")
        print(SYSTEM_PROMPT)
        if args.video_level:
            demo_prompt = _apply_video_level_intro(base_prompt, key_frames)
            print(f"\n===== USER PROMPT (video-level, {len(key_frames)} frames) =====")
            print(demo_prompt)
        elif causal:
            # Show what the prompt looks like for the last frame with all context
            demo_prompt = _apply_causal_intro(base_prompt, len(key_frames), key_frames[-1].stem)
            print(f"\n===== USER PROMPT (causal, {len(key_frames)} frames) =====")
            print(demo_prompt)
        else:
            print("\n===== USER PROMPT =====")
            print(base_prompt)
        print(f"\n[dryrun] Would process {len(key_frames)} key frames.")
        return 0

    # Init model (langchain, consistent with surgent sequential runner)
    model_lower = args.model.lower()
    use_temperature = args.temperature
    # gpt-5 and above only support default temperature (1)
    if model_lower.startswith("gpt-5") or model_lower.startswith("gpt-6") or model_lower.startswith("gpt-7"):
        use_temperature = 1.0

    model = _init_model(args.model, temperature=use_temperature)
    print(f"Model: {args.model} (langchain)")

    # Output paths
    model_tag = sanitize_tag(args.model)
    tax_tag = taxonomy_path.stem  # e.g. "taxonomy_v10"
    if args.out_jsonl:
        out_jsonl = Path(args.out_jsonl)
    else:
        out_jsonl = (
            REPO_ROOT
            / "outputs"
            / preset_tag
            / f"{video_id}__{preset_tag}__{model_tag}__{tax_tag}.jsonl"
        )

    date_folder = datetime.now().strftime("%Y%m%d")
    trace_dir = Path(args.trace_dir) / date_folder / f"{video_id}__{preset_tag}__{model_tag}__{tax_tag}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output JSONL: {out_jsonl}")
    print(f"Trace dir: {trace_dir}")

    # ── Video-level mode: single LLM call with all key frames ──
    if args.video_level:
        print(f"Video-level mode: sending all {len(key_frames)} key frames in one call ...")
        images: List[Image.Image] = [load_image(p) for p in key_frames]
        prompt_text = _apply_video_level_intro(base_prompt, key_frames)

        raw_response = ""
        parsed: Dict[str, Any] = {}
        parse_ok = False
        parse_errors: List[str] = []

        try:
            raw_response = _invoke_model(model, prompt_text, images, SYSTEM_PROMPT)
            parsed = parse_frame_output(raw_response)
            parse_ok = True
        except Exception as exc:
            parse_errors.append(f"exception:{type(exc).__name__}:{exc}")

        pred = extract_pred(parsed)
        actions = extract_actions(parsed)
        if fixed_k is not None:
            actions = _enforce_fixed_k(actions, fixed_k)
        rationale = parsed.get("rationale") if preset in ("cot", "auto") else None
        checklist = parsed.get("checklist") if preset in ("subrubric", "auto") else None

        row: Dict[str, Any] = {
            "schema_version": "baseline.v1",
            "video_id": video_id,
            "preset": preset_tag,
            "model": args.model,
            "step_index": 0,
            "frame_id": "video",
            "frame_path": str(image_dir),
            "video_level": True,
            "num_key_frames": len(key_frames),
            "pred": pred,
            "actions_ranked": actions,
            "parse_ok": parse_ok,
            "parse_errors": parse_errors,
            "raw_response": raw_response,
            "taxonomy_version": taxonomy_catalog["version"],
            "taxonomy_path": str(taxonomy_path),
        }
        if rationale is not None:
            row["rationale"] = rationale
        if checklist is not None:
            row["checklist"] = checklist

        # Print summary
        c1, c2, c3 = pred.get("c1"), pred.get("c2"), pred.get("c3")
        print(f"  Video-level: C1={c1}  C2={c2}  C3={c3}  parse_ok={parse_ok}")
        if actions:
            for a in actions:
                print(
                    f"    [rank {a.get('rank')}] "
                    f"{a.get('actor_role')}/{a.get('tool_type')} "
                    f"{a.get('action_code')} -> {a.get('target_structure')} "
                    f"ctx=({a.get('target_context_1')}, {a.get('target_context_2')}) "
                    f"| intention: {a.get('intention')} "
                    f"| conf={a.get('confidence')}"
                )
        else:
            print("    (no actions)")

        video_pred = pred  # direct video-level prediction

        print(f"\nVideo-level CVS: C1={video_pred['c1']}  C2={video_pred['c2']}  C3={video_pred['c3']}")

        # Write JSONL (single row)
        frame_rows_vl = [row]
        with out_jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Write video-level summary JSON
        video_summary = {
            "schema_version": "baseline.video.v1",
            "video_id": video_id,
            "preset": preset_tag,
            "model": args.model,
            "video_level": True,
            "taxonomy_version": taxonomy_catalog["version"],
            "taxonomy_path": str(taxonomy_path),
            "num_key_frames": len(key_frames),
            "num_total_frames": len(all_frames),
            "keyframe_step": args.keyframe_step,
            "video_pred": video_pred,
        }
        video_summary_path = out_jsonl.with_suffix(".video_summary.json")
        video_summary_path.write_text(
            json.dumps(video_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write trace copies
        trace_steps_path = trace_dir / "steps.jsonl"
        with trace_steps_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        trace_summary_path = trace_dir / "video_summary.json"
        trace_summary_path.write_text(
            json.dumps(video_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Run metadata
        run_meta = {
            "command": " ".join(sys.argv),
            "video_id": video_id,
            "preset": preset_tag,
            "video_level": True,
            "model": args.model,
            "taxonomy_version": taxonomy_catalog["version"],
            "num_key_frames": len(key_frames),
            "output_jsonl": str(out_jsonl),
            "trace_dir": str(trace_dir),
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        (trace_dir / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\nWrote 1 video-level record to {out_jsonl}")
        print(f"Video summary: {video_summary_path}")
        print(f"Trace dir: {trace_dir}")
        return 0

    # Process each key frame
    frame_rows: List[Dict[str, Any]] = []

    for step_idx, frame_path in enumerate(key_frames):
        frame_id = frame_path.stem

        if causal:
            # Causal mode: pass all key frames from index 0..step_idx (inclusive)
            context_paths = key_frames[: step_idx + 1]
            num_context = len(context_paths)
            print(f"[{step_idx + 1}/{len(key_frames)}] Processing frame {frame_id} (causal: {num_context} frames) ...")

            images: List[Image.Image] = [load_image(p) for p in context_paths]
            prompt_text = _apply_causal_intro(base_prompt, num_context, frame_id)
        else:
            print(f"[{step_idx + 1}/{len(key_frames)}] Processing frame {frame_id} ...")
            images = [load_image(frame_path)]
            prompt_text = base_prompt

        # Single LLM call (langchain)
        raw_response = ""
        parsed: Dict[str, Any] = {}
        parse_ok = False
        parse_errors: List[str] = []

        try:
            raw_response = _invoke_model(model, prompt_text, images, SYSTEM_PROMPT)
            parsed = parse_frame_output(raw_response)
            parse_ok = True
        except Exception as exc:
            parse_errors.append(f"exception:{type(exc).__name__}:{exc}")

        pred = extract_pred(parsed)
        actions = extract_actions(parsed)
        if fixed_k is not None:
            actions = _enforce_fixed_k(actions, fixed_k)
        rationale = parsed.get("rationale") if preset in ("cot", "auto") else None
        checklist = parsed.get("checklist") if preset in ("subrubric", "auto") else None

        row: Dict[str, Any] = {
            "schema_version": "baseline.v1",
            "video_id": video_id,
            "preset": preset_tag,
            "model": args.model,
            "step_index": step_idx,
            "frame_id": frame_id,
            "frame_path": str(frame_path),
            "pred": pred,
            "actions_ranked": actions,
            "parse_ok": parse_ok,
            "parse_errors": parse_errors,
            "raw_response": raw_response,
            "taxonomy_version": taxonomy_catalog["version"],
            "taxonomy_path": str(taxonomy_path),
        }
        if causal:
            row["causal"] = True
            row["num_context_frames"] = step_idx + 1
        if rationale is not None:
            row["rationale"] = rationale
        if checklist is not None:
            row["checklist"] = checklist

        frame_rows.append(row)

        # Summary
        c1 = pred.get("c1")
        c2 = pred.get("c2")
        c3 = pred.get("c3")
        print(f"  C1={c1}  C2={c2}  C3={c3}  parse_ok={parse_ok}")
        if actions:
            for a in actions:
                print(
                    f"    [rank {a.get('rank')}] "
                    f"{a.get('actor_role')}/{a.get('tool_type')} "
                    f"{a.get('action_code')} -> {a.get('target_structure')} "
                    f"ctx=({a.get('target_context_1')}, {a.get('target_context_2')}) "
                    f"| intention: {a.get('intention')} "
                    f"| conf={a.get('confidence')}"
                )
        else:
            print("    (no actions)")

    # Compute video-level CVS: MAX(frame-level) per criterion
    video_pred: Dict[str, Optional[float]] = {"c1": None, "c2": None, "c3": None}
    for c in ("c1", "c2", "c3"):
        vals = [r["pred"][c] for r in frame_rows if r["pred"].get(c) is not None]
        video_pred[c] = max(vals) if vals else None

    print(f"\nVideo-level CVS (max over frames): C1={video_pred['c1']}  C2={video_pred['c2']}  C3={video_pred['c3']}")

    # Write frame-level JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in frame_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write video-level summary JSON
    video_summary = {
        "schema_version": "baseline.video.v1",
        "video_id": video_id,
        "preset": preset_tag,
        "model": args.model,
        "causal": causal,
        "taxonomy_version": taxonomy_catalog["version"],
        "taxonomy_path": str(taxonomy_path),
        "num_key_frames": len(key_frames),
        "num_total_frames": len(all_frames),
        "keyframe_step": args.keyframe_step,
        "video_pred": video_pred,
        "frame_preds": [
            {"frame_id": r["frame_id"], "pred": r["pred"]}
            for r in frame_rows
        ],
    }
    video_summary_path = out_jsonl.with_suffix(".video_summary.json")
    video_summary_path.write_text(
        json.dumps(video_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write trace copies
    trace_steps_path = trace_dir / "steps.jsonl"
    with trace_steps_path.open("w", encoding="utf-8") as f:
        for row in frame_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    trace_summary_path = trace_dir / "video_summary.json"
    trace_summary_path.write_text(
        json.dumps(video_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Run metadata
    run_meta = {
        "command": " ".join(sys.argv),
        "video_id": video_id,
        "preset": preset_tag,
        "causal": causal,
        "model": args.model,
        "taxonomy_version": taxonomy_catalog["version"],
        "num_key_frames": len(key_frames),
        "output_jsonl": str(out_jsonl),
        "trace_dir": str(trace_dir),
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
    }
    (trace_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nWrote {len(frame_rows)} frame records to {out_jsonl}")
    print(f"Video summary: {video_summary_path}")
    print(f"Trace dir: {trace_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
