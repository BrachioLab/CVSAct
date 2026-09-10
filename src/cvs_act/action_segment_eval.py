"""Utilities for action-segment extraction and evaluation.

This module supports notebooks that:
- convert timestamped audit annotations into a compact actor-wise format,
- extract actor-wise timestamped actions from free-form LLM segment outputs, and
- evaluate extracted actions with the multi-granularity F1@k spec.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .audit_timestamped_tool_interface import (
    CAMERA_ACTION_CODES,
    EXTRA_ACTION_CODES,
    EXTRA_TOOL_TYPES,
    RETRACTION_DIRECTION_OPTIONS,
)


LEFT_EXACT_TO_MEDIUM = {
    "KEEP_RETRACT_LATERAL": "lateral",
    "RETRACT_LATERAL": "lateral",
    "RETRACT_MEDIAL_TO_LATERAL": "lateral",
    "RETRACT_UPWARD_TO_LATERAL": "lateral",
    "KEEP_RETRACT_MEDIAL": "medial",
    "RETRACT_MEDIAL": "medial",
    "RETRACT_LATERAL_TO_MEDIAL": "medial",
    "KEEP_RETRACT_UPWARD": "upward",
    "RETRACT_LATERAL_TO_UPWARD": "upward",
}

LEFT_EXACT_TO_COARSE = {
    "KEEP_RETRACT_LATERAL": "maintain",
    "KEEP_RETRACT_MEDIAL": "maintain",
    "KEEP_RETRACT_UPWARD": "maintain",
    "RETRACT_LATERAL": "change",
    "RETRACT_LATERAL_TO_MEDIAL": "change",
    "RETRACT_LATERAL_TO_UPWARD": "change",
    "RETRACT_MEDIAL": "change",
    "RETRACT_MEDIAL_TO_LATERAL": "change",
    "RETRACT_UPWARD_TO_LATERAL": "change",
}

LEFT_CODE_TO_DIRECTION_FIELDS = {
    "KEEP_RETRACT_LATERAL": ("no", "lateral", "lateral"),
    "KEEP_RETRACT_MEDIAL": ("no", "medial", "medial"),
    "KEEP_RETRACT_UPWARD": ("no", "upward", "upward"),
    "RETRACT_LATERAL": ("yes", "not_retracted", "lateral"),
    "RETRACT_MEDIAL": ("yes", "not_retracted", "medial"),
    "RETRACT_LATERAL_TO_MEDIAL": ("yes", "lateral", "medial"),
    "RETRACT_LATERAL_TO_UPWARD": ("yes", "lateral", "upward"),
    "RETRACT_MEDIAL_TO_LATERAL": ("yes", "medial", "lateral"),
    "RETRACT_UPWARD_TO_LATERAL": ("yes", "upward", "lateral"),
}

LEFT_DIRECTION_FIELDS_TO_CODE = {fields: code for code, fields in LEFT_CODE_TO_DIRECTION_FIELDS.items()}

ACTION_VERBS = {
    "DISSECT": "dissects around",
    "RETRACT": "retracts",
    "ASPIRATE": "aspirates",
    "IRRIGATE": "irrigates",
    "COUNTERTRACTION_ASSIST": "provides countertraction on",
    "CLIP": "clips",
    "CUT": "cuts",
    "GRASP": "grasps",
    "ICG_SWITCH": "switches ICG",
}

CAMERA_DESCRIPTIONS = {
    "CAMERA_UNCERTAIN": "The camera is uncertain.",
    "CAMERA_ZOOM_OUT": "The camera zooms out.",
    "CAMERA_ZOOM_IN": "The camera zooms in.",
    "CAMERA_REPOSITION": "The camera repositions.",
    "CAMERA_NO_CHANGE": "The camera does not change.",
}

NATURAL_CODE_TRANSLATIONS = {
    "KEEP_RETRACT_LATERAL": "keep retracted lateral",
    "KEEP_RETRACT_MEDIAL": "keep retracted medial",
    "KEEP_RETRACT_UPWARD": "keep retracted upward",
    "RETRACT_LATERAL": "retract to lateral",
    "RETRACT_MEDIAL": "retract to medial",
    "RETRACT_LATERAL_TO_MEDIAL": "retract from lateral to medial",
    "RETRACT_LATERAL_TO_UPWARD": "retract from lateral to upward",
    "RETRACT_MEDIAL_TO_LATERAL": "retract from medial to lateral",
    "RETRACT_UPWARD_TO_LATERAL": "retract from upward to lateral",
    "CAMERA_UNCERTAIN": "The camera is uncertain",
    "CAMERA_ZOOM_OUT": "The camera zooms out",
    "CAMERA_ZOOM_IN": "The camera zooms in",
    "CAMERA_REPOSITION": "The camera repositions",
    "CAMERA_NO_CHANGE": "The camera does not change",
    "DISSECT": "dissect",
    "COUNTERTRACTION_ASSIST": "countertraction assist",
    "ICG_SWITCH": "switch ICG",
}

ACTION_CODE_ALIASES = {
    "IRRIGATOR_COUNTERTRACTION_ASSIST": "COUNTERTRACTION_ASSIST",
}

CAMERA_EXACT_TO_MEDIUM = {
    "CAMERA_ZOOM_IN": "zoom",
    "CAMERA_ZOOM_OUT": "zoom",
    "CAMERA_REPOSITION": "reposition",
    "CAMERA_UNCERTAIN": "no_change",
    "CAMERA_NO_CHANGE": "no_change",
}

CAMERA_EXACT_TO_COARSE = {
    "CAMERA_ZOOM_IN": "change",
    "CAMERA_ZOOM_OUT": "change",
    "CAMERA_REPOSITION": "change",
    "CAMERA_UNCERTAIN": "no_change",
    "CAMERA_NO_CHANGE": "no_change",
}

STRUCTURED_PREDICTION_METHOD = "structured_prediction"
STRUCTURED_PREDICTION_DETERMINISTIC_METHOD = "structured_prediction_deterministic"

IOU_THRESHOLDS = [0.10, 0.25, 0.50]
GRANULARITIES = ["exact", "medium", "coarse"]
ACTORS = ["left", "camera", "right", "other"]
EVAL_ACTORS = ["left", "camera", "right", "other"]


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_set(value: Any) -> bool:
    return value not in (None, "", "(not set)", "Null", "null")


def frame_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def taxonomy_options(annotation_root: Path) -> Dict[str, List[str]]:
    taxonomy = read_json(annotation_root / "taxonomy_v10.json")
    options = {
        field: (
            sorted(dict.fromkeys(canonical_action_code(item["value"]) for item in items))
            if field == "action_code"
            else [item["value"] for item in items]
        )
        for field, items in taxonomy["fields"].items()
    }
    for code in EXTRA_ACTION_CODES:
        if code not in options["action_code"]:
            options["action_code"].append(code)
    for tool in EXTRA_TOOL_TYPES:
        if tool not in options["tool_type"]:
            options["tool_type"].append(tool)
    return options


def load_audit_records(audit_dir: Path) -> List[dict]:
    records: List[dict] = []
    for path in sorted(audit_dir.glob("*.json")):
        for record in read_json(path):
            record = dict(record)
            record["_source_path"] = str(path)
            records.append(record)
    return records


def sorted_segments(rows: List[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("start_frame") is None,
            row.get("start_frame") or -1,
            row.get("end_frame") or -1,
            json.dumps(row, sort_keys=True),
        ),
    )


def left_direction_fields(seg: dict) -> Tuple[Any, Any, Any]:
    code = seg.get("action_code", "(not set)")
    derived = LEFT_CODE_TO_DIRECTION_FIELDS.get(code, ("unsure", "unsure", "unsure"))
    changed = seg.get("changed")
    start_direction = seg.get("start_direction")
    end_direction = seg.get("end_direction")
    if changed in (None, "", "unsure"):
        changed = derived[0]
    if start_direction in (None, "", "unsure"):
        start_direction = derived[1]
    if end_direction in (None, "", "unsure"):
        end_direction = derived[2]
    return changed, start_direction, end_direction


def humanize_code(value: Any) -> str:
    text = canonical_action_code(value)
    if not is_set(text):
        return ""
    if text in NATURAL_CODE_TRANSLATIONS:
        return NATURAL_CODE_TRANSLATIONS[text]
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.replace("_", " ").replace("-", " ").lower()


def canonical_action_code(value: Any) -> str:
    text = str(value or "").strip()
    return ACTION_CODE_ALIASES.get(text, text)


def is_deterministic_structured_method(method_name: Any) -> bool:
    text = str(method_name or "")
    if text.endswith("_cvs_context"):
        text = text.removesuffix("_cvs_context")
    return text == STRUCTURED_PREDICTION_DETERMINISTIC_METHOD


def segmentation_source_method(method_name: Any) -> str:
    text = str(method_name or "")
    suffix = "_cvs_context" if text.endswith("_cvs_context") else ""
    base = text.removesuffix(suffix)
    if base == STRUCTURED_PREDICTION_DETERMINISTIC_METHOD:
        return STRUCTURED_PREDICTION_METHOD + suffix
    return text


def _normalize_prompt_left_direction(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return {
        "left": "lateral",
        "lateral": "lateral",
        "right": "medial",
        "medial": "medial",
        "upward": "upward",
        "not_retracted": "not_retracted",
    }.get(text)


def _normalize_prompt_changed(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"yes", "no"}:
        return text
    return None


def _structured_left_code(grasper_retraction: dict) -> Optional[str]:
    changed = _normalize_prompt_changed(grasper_retraction.get("changed"))
    start_direction = _normalize_prompt_left_direction(grasper_retraction.get("start_direction"))
    end_direction = _normalize_prompt_left_direction(grasper_retraction.get("end_direction"))
    if changed == "no":
        if end_direction and not start_direction:
            start_direction = end_direction
        if start_direction and not end_direction:
            end_direction = start_direction
    elif changed == "yes":
        if end_direction and not start_direction and end_direction in {"lateral", "medial"}:
            start_direction = "not_retracted"
    if not (changed and start_direction and end_direction):
        return None
    return LEFT_DIRECTION_FIELDS_TO_CODE.get((changed, start_direction, end_direction))


def _structured_camera_code(camera_movement: Any) -> Optional[str]:
    text = str(camera_movement or "").strip().lower()
    return {
        "zoom_in": "CAMERA_ZOOM_IN",
        "zoom_out": "CAMERA_ZOOM_OUT",
        "reposition": "CAMERA_REPOSITION",
        "unclear_movement": "CAMERA_UNCERTAIN",
        "unsure": "CAMERA_UNCERTAIN",
        "not_moving": "CAMERA_NO_CHANGE",
    }.get(text)


def build_deterministic_structured_extraction(
    method_name: str,
    prediction_text: str,
    clip_start: int,
    clip_end: int,
) -> dict:
    parsed = extract_json_object(prediction_text)
    rows = {"method": method_name, "left": [], "right": [], "camera": [], "other": []}
    for segment in parsed.get("segments", []) or []:
        start = frame_or_none(segment.get("start_frame"))
        end = frame_or_none(segment.get("end_frame"))
        if start is None or end is None:
            continue
        description = complete_sentence(segment.get("action_description"))

        grasper_retraction = segment.get("grasper_retraction") or {}
        left_code = _structured_left_code(grasper_retraction)
        if left_code:
            left_row = {
                "start_frame": start,
                "end_frame": end,
                "retraction_direction_code": left_code,
            }
            evidence = grasper_retraction.get("description") or description
            if is_set(evidence):
                left_row["evidence"] = str(evidence).strip()
            rows["left"].append(left_row)

        active_tools = segment.get("active_tools") or []
        if isinstance(active_tools, dict):
            active_tools = [active_tools]
        for tool_row in active_tools:
            if not isinstance(tool_row, dict):
                continue
            right_row = {
                "start_frame": start,
                "end_frame": end,
                "tool_type": tool_row.get("tool_type"),
                "action_code": canonical_action_code(tool_row.get("action_code")),
                "target_structure": tool_row.get("target_structure"),
                "target_context_1": tool_row.get("target_context_1", "(not set)"),
                "target_context_2": tool_row.get("target_context_2", "(not set)"),
            }
            evidence = tool_row.get("evidence") or description
            if is_set(evidence):
                right_row["evidence"] = str(evidence).strip()
            rows["right"].append(right_row)

        camera_code = _structured_camera_code(segment.get("camera_movement"))
        if camera_code:
            camera_row = {
                "start_frame": start,
                "end_frame": end,
                "action_code": camera_code,
            }
            if is_set(description):
                camera_row["evidence"] = description
            rows["camera"].append(camera_row)
    return rows


def complete_sentence(text: Any) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    return out if out.endswith((".", "!", "?")) else out + "."


def left_segment_description(seg: dict) -> str:
    if is_set(seg.get("description")):
        return str(seg.get("description"))
    changed = seg.get("changed")
    start_direction = humanize_code(seg.get("start_direction"))
    end_direction = humanize_code(seg.get("end_direction"))
    if changed == "no" and end_direction:
        return f"The grasper keeps the gallbladder neck retracted {end_direction}."
    if start_direction and end_direction:
        return f"The grasper retracts the gallbladder neck from {start_direction} to {end_direction}."
    code = humanize_code(seg.get("action_code"))
    return f"The grasper retracts the gallbladder neck with action {code}." if code else "The grasper retracts the gallbladder neck."


def right_segment_description(seg: dict) -> str:
    if is_set(seg.get("description")):
        return str(seg.get("description"))
    tool = humanize_code(seg.get("tool_type")) or "right instrument"
    action_code = canonical_action_code(seg.get("action_code"))
    verb = ACTION_VERBS.get(action_code, humanize_code(action_code) or "acts")
    target = humanize_code(seg.get("target_structure"))
    contexts = [humanize_code(seg.get("target_context_1")), humanize_code(seg.get("target_context_2"))]
    contexts = [context for context in contexts if context and context != "not set"]
    target_phrase = f" the {target}" if target else ""
    if contexts:
        return f"The {tool} {verb}{target_phrase}, {', and '.join(contexts)}."
    return f"The {tool} {verb}{target_phrase}."


def camera_segment_description(seg: dict) -> str:
    if is_set(seg.get("description")):
        return str(seg.get("description"))
    code = seg.get("action_code", seg.get("camera_movement"))
    return CAMERA_DESCRIPTIONS.get(code, f"The camera {humanize_code(code)}.")


def other_segment_description(action: dict) -> str:
    existing = action.get("one_sentence") or action.get("generated_sentence") or action.get("original_sentence")
    if is_set(existing):
        return str(existing)
    code = canonical_action_code(action.get("action_code"))
    if code == "ICG_SWITCH":
        return "The ICG is switched."
    tool = humanize_code(action.get("tool_type"))
    verb = ACTION_VERBS.get(code, humanize_code(code) or "acts")
    target = humanize_code(action.get("target_structure"))
    subject = f"The {tool}" if tool else "The other action"
    return f"{subject} {verb}{(' the ' + target) if target else ''}."


def naturalize_simple_segment(actor: str, seg: dict) -> dict:
    out = dict(seg)
    if actor == "left":
        out["retraction_direction_code"] = humanize_code(seg.get("retraction_direction_code"))
        out["start_direction"] = humanize_code(seg.get("start_direction"))
        out["end_direction"] = humanize_code(seg.get("end_direction"))
        out["changed"] = {"yes": "changed", "no": "not changed"}.get(seg.get("changed"), humanize_code(seg.get("changed")))
    elif actor == "right":
        out["target_structure"] = humanize_code(seg.get("target_structure"))
        out["target_context_1"] = humanize_code(seg.get("target_context_1"))
        out["target_context_2"] = humanize_code(seg.get("target_context_2"))
        out["tool_type"] = humanize_code(seg.get("tool_type"))
        out["action_code"] = humanize_code(seg.get("action_code"))
        out["triplet"] = [humanize_code(value) for value in seg.get("triplet", [])]
    elif actor == "camera":
        out["action_code"] = humanize_code(seg.get("action_code"))
        out["camera_movement"] = humanize_code(seg.get("camera_movement"))
    elif actor == "other":
        for key in ["target_structure", "target_context_1", "target_context_2", "actor_role", "tool_type", "action_code"]:
            out[key] = humanize_code(seg.get(key))
    out["description"] = complete_sentence(seg.get("description"))
    return out


def naturalize_simple_actions(record: dict) -> dict:
    out = {key: value for key, value in record.items() if key not in ACTORS}
    for actor in ACTORS:
        out[actor] = [naturalize_simple_segment(actor, seg) for seg in record.get(actor, [])]
    return out


def convert_record_to_simple_actions(record: dict) -> dict:
    coarse = record.get("coarse", {})
    out = {
        "example_id": record.get("example_id"),
        "video_id": record.get("video_id"),
        "criterion": record.get("criterion"),
        "mind_change": record.get("mind_change"),
        "frame_range": [frame_or_none(coarse.get("start_frame")), frame_or_none(coarse.get("end_frame"))],
        "source_path": record.get("_source_path"),
        "left": [],
        "right": [],
        "camera": [],
        "other": [],
    }
    for action in coarse.get("annotation", {}).get("actions_ranked", []):
        rank = action.get("rank")
        actor_role = action.get("actor_role", "(not set)")
        handled = False

        for seg in action.get("left_action_segments") or []:
            changed, start_direction, end_direction = left_direction_fields(seg)
            left_row = {
                "start_frame": frame_or_none(seg.get("start_frame")),
                "end_frame": frame_or_none(seg.get("end_frame")),
                "retraction_direction_code": seg.get("action_code", "(not set)"),
                "changed": changed,
                "start_direction": start_direction,
                "end_direction": end_direction,
                "description": "",
                "rank": rank,
            }
            left_row["description"] = left_segment_description({**seg, **left_row, "action_code": left_row["retraction_direction_code"]})
            out["left"].append(left_row)
            handled = True

        for seg in action.get("right_action_segments") or []:
            triplet = [
                seg.get("tool_type", "(not set)"),
                canonical_action_code(seg.get("action_code", "(not set)")),
                seg.get("target_structure", "(not set)"),
            ]
            right_row = {
                "start_frame": frame_or_none(seg.get("start_frame")),
                "end_frame": frame_or_none(seg.get("end_frame")),
                "target_structure": triplet[2],
                "target_context_1": seg.get("target_context_1", "(not set)"),
                "target_context_2": seg.get("target_context_2", "(not set)"),
                "tool_type": triplet[0],
                "action_code": triplet[1],
                "triplet": triplet,
                "description": "",
                "rank": rank,
            }
            right_row["description"] = right_segment_description({**seg, **right_row})
            out["right"].append(right_row)
            handled = True

        for seg in action.get("camera_action_segments") or []:
            camera_row = {
                "start_frame": frame_or_none(seg.get("start_frame")),
                "end_frame": frame_or_none(seg.get("end_frame")),
                "action_code": seg.get("action_code", seg.get("camera_movement", "(not set)")),
                "camera_movement": seg.get("camera_movement", seg.get("action_code", "(not set)")),
                "description": "",
                "rank": rank,
            }
            camera_row["description"] = camera_segment_description({**seg, **camera_row})
            out["camera"].append(camera_row)
            handled = True

        action_code = canonical_action_code(action.get("action_code"))
        if (not handled and actor_role not in {"left_instrument", "right_instrument", "camera"}) or action_code == "ICG_SWITCH":
            out["other"].append(
                {
                    "start_frame": frame_or_none(action.get("action_start_frame", action.get("start_frame"))),
                    "end_frame": frame_or_none(action.get("action_end_frame", action.get("end_frame"))),
                    "target_structure": action.get("target_structure", "(not set)"),
                    "target_context_1": action.get("target_context_1", "(not set)"),
                    "target_context_2": action.get("target_context_2", "(not set)"),
                    "actor_role": actor_role,
                    "tool_type": action.get("tool_type", "(not set)"),
                    "action_code": action_code or "(not set)",
                    "description": other_segment_description(action),
                    "rank": rank,
                }
            )

    for actor in ACTORS:
        out[actor] = merge_adjacent_segments(sorted_segments(out[actor]), actor)
    return out


def build_simple_options(simple_records: List[dict], annotation_root: Path) -> dict:
    options = taxonomy_options(annotation_root)
    observed_left = sorted(
        {
            seg["retraction_direction_code"]
            for record in simple_records
            for seg in record["left"]
            if is_set(seg.get("retraction_direction_code"))
        }
    )
    observed_right = sorted(
        {
            tuple(seg["triplet"])
            for record in simple_records
            for seg in record["right"]
            if all(is_set(value) for value in seg.get("triplet", []))
        }
    )
    observed_camera = sorted(
        {
            seg["action_code"]
            for record in simple_records
            for seg in record["camera"]
            if is_set(seg.get("action_code"))
        }
    )
    observed_other = sorted(
        {
            seg["action_code"]
            for record in simple_records
            for seg in record["other"]
            if is_set(seg.get("action_code"))
        }
    )
    camera_options = [code for code in CAMERA_ACTION_CODES if code in observed_camera]
    camera_options += [code for code in observed_camera if code not in set(camera_options)]
    return {
        "source": "audit_timestamped_tool_interface.py + taxonomy_v10.json + observed audit_v11 GT",
        "interface_options": {
            "retraction_direction_options": RETRACTION_DIRECTION_OPTIONS,
            "camera_action_codes": CAMERA_ACTION_CODES,
            "tool_type": options["tool_type"],
            "action_code": options["action_code"],
            "target_structure": options["target_structure"],
            "target_context": options["target_context"],
        },
        "left_retraction_direction_code_options": observed_left,
        "right_triplet_options": [list(triplet) for triplet in observed_right],
        "camera_action_code_options": camera_options,
        "other_action_code_options": observed_other,
    }


def build_frame_ids(start_frame: int, end_frame: int, frame_stride: int = 150, inbetween_slots: Sequence[int] = (1, 2, 3, 4)) -> List[int]:
    anchor_frame_ids = list(range(int(start_frame), int(end_frame) + 1, frame_stride))
    if not anchor_frame_ids or anchor_frame_ids[-1] != int(end_frame):
        anchor_frame_ids.append(int(end_frame))
    frame_ids: List[int] = []
    for left, right in zip(anchor_frame_ids, anchor_frame_ids[1:]):
        frame_ids.append(left)
        gap = right - left
        for slot in sorted({int(slot) for slot in inbetween_slots}):
            candidate = left + round(gap * slot / 5)
            if left < candidate < right:
                frame_ids.append(candidate)
    frame_ids.append(anchor_frame_ids[-1])
    return sorted(dict.fromkeys(frame_ids))


def build_messages_for_record(record: dict, user_prompt: str, frames_dir: Path, model_id: str, temperature: float, enable_thinking: bool, frame_labels: Optional[Dict[int, str]] = None) -> Tuple[list, dict]:
    coarse = record["coarse"]
    start_frame = int(coarse["start_frame"])
    end_frame = int(coarse["end_frame"])
    video_id = record["video_id"]
    frame_ids = build_frame_ids(start_frame, end_frame)
    frame_paths = [frames_dir / video_id / f"frame_{frame_id:06d}.png" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frame files for {record['example_id']}: {missing[:5]}")

    user_content = [{"type": "text", "text": user_prompt}]
    frame_records = []
    for frame_id, path in zip(frame_ids, frame_paths):
        label = (frame_labels.get(frame_id) if frame_labels else None) or f"Frame {frame_id:06d}"
        user_content.append({"type": "text", "text": label})
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
        frame_records.append({"frame_id": frame_id, "label": label, "path": str(path), "sha256": file_sha256(path)})

    system_prompt = "You are a surgical video understanding assistant. You will receive laparoscopic video frames in chronological order. Each image is preceded by a text label containing its exact frame number."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    prompt_record = {
        "model": model_id,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "source": "audit_v11",
        "example_id": record["example_id"],
        "video_id": video_id,
        "criterion": record.get("criterion"),
        "mind_change": record.get("mind_change"),
        "frame_range": [start_frame, end_frame],
        "frames": frame_records,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    return messages, prompt_record


def default_segmentation_prompts(options: Optional[dict] = None) -> Dict[str, str]:
    continuity = """
Segment continuity requirements:
- Cover the full provided frame range with no gaps.
- The first segment must start at the first provided frame label.
- The final segment must end at the last provided frame label.
- Each segment's end_frame must equal the next segment's start_frame, e.g. 000150-000180, then 000180-000300. Do not skip intervals like 000150-000180 followed by 000210-000300.
""".strip()
    task_intro = """
The required anchor frames are sampled every 5 seconds from one continuous video segment. Optional in-between frames may also be included.

Task: segment the video into multiple frame-number ranges where each range contains one continuous different surgical action or no action.
""".strip()

    frame_rules = f"""
Use only the provided frames and their text frame labels. Do not infer exact boundaries between sampled frames more precisely than the visible frame numbers.
For start_frame and end_frame, copy the actual frame numbers from the visible text labels, e.g. Frame 000150 means 150. Do not use the frame's zero-based or one-based position/index in the provided sequence.

{continuity}

If something changes, like the retraction direction of the gallbladder between two frames, you should also report that.
""".strip()

    output_fields = """
For each segment, report:
- start_frame: actual frame number from a provided text label, not a sequence index
- end_frame: actual frame number from a provided text label, not a sequence index
- label: action or no_action
- detailed action_description (3-5 sentences)
""".strip()

    description = f"""
{task_intro}

{frame_rules}

{output_fields}

Return only a Markdown table. Do not include extra explanation.
""".strip()
    hints = f"""
{task_intro}

Actions you should be reporting:
1. Did the grasper's retraction direction of the gallbladder neck change? If so, where did it start and where did it end? Examples of acceptable answers: (here left and right can be replaced with left/right/upward)
    - Started in the [left], and dragged to the [right] in the end.
    - Started not retracted, and then it became retracted to the [left].
    - Started retracted to the [left], and it was kept that way until the end.
    - Started retracted to the [left] a bit, and then it was retracted more that way.
    - If you are not sure, you can always report unsure instead of forcing an answer.
    - Try to say which direction the gallbladder is retracted to if you talk about the grasper retracting the gallbladder neck, or explicitly say unclear direction if you don't know.
2. Is there a tool actively operating in the area?
    - If a tool is just hanging there without interacting with any tissue, then you can ignore it.
    - Include: tool type, action, target.
    - Action: Operating includes dissecting, irrigating, aspirating, coagulating, retracting.
    - Tool type: The tool can be Maryland grasper, hook, scissors, irrigator, or clipper. Withdrawing is an action, not a tool type: use it only when the tool was previously blocking a region and is now removed from there.
    - Target: Target can be hepatocystic triangle if it is general in that area and the tissues are not clearly delineated, or be more specific as cystic duct, cystic artery, cystic plate, which describe the proxy area. Even if those ducts or regions are not perfectly delineated, if you see they are somewhat delineated and the tool is clearly operating onto those areas, then prefer the more specific regions. For withdrawal, if the tool started blocking only a smaller region (e.g. cystic plate) and then is withdrawn, then you should say it's withdrawn from the "cystic plate" instead of the more general "hepatocystic triangle".
    - Target context: Also describe where the target is relative to visible anatomy when it is clear, for example between cystic artery and cystic plate, near the cystic duct, close to the gallbladder neck, near the base of the hepatocystic triangle, behind the two tubular structures, or whether the view is moving from posterior to anterior / anterior to posterior. If no such context is visible, say not set or omit it.
    - If you are not sure, always report unsure instead of forcing an answer. In prose/table outputs, you can report unsure for some subset of (action, tool type and target) tuples and still have answers for others.
3. Is there any significant camera movements?
    - Did the camera zoom in, zoom out, reposition, moving in an unclear way, or not moving?
    - If you are not sure, you can always report unsure instead of forcing an answer.
4. Is the smaller segment blurry or obscured such that it's hard to judge?

{frame_rules}

{output_fields}

Return only a Markdown table. Do not include extra explanation.
""".strip()
    right_ontology = ""
    if options:
        right_triplets = options["right_triplet_options"]
        right_tool_types = sorted({row[0] for row in right_triplets})
        right_action_codes = sorted({canonical_action_code(row[1]) for row in right_triplets})
        right_target_structures = sorted({row[2] for row in right_triplets})
        right_ontology = f"""

Right-tool ontology for active_tools:
- tool_type must be exactly one of:
{json.dumps(right_tool_types, indent=2)}
- action_code must be exactly one of:
{json.dumps(right_action_codes, indent=2)}
- target_structure must be exactly one of:
{json.dumps(right_target_structures, indent=2)}
- target_context_1 and target_context_2 must each be exactly one of these allowed target contexts:
{json.dumps(options['interface_options']['target_context'], indent=2)}
- Use "(not set)" for target_context_1 or target_context_2 when no listed context is visible.
- Only include an active_tools row when tool_type, action_code, and target_structure can all be selected from the listed options. If the active tool is uncertain, leave active_tools empty and explain the uncertainty in action_description.
- Use action_code "COUNTERTRACTION_ASSIST" when a tool, including an irrigator, provides temporary countertraction/pushing to assist exposure or regrasping.
- Use action_code "TOOL_WITHDRAW_UNBLOCKS_VIEW" when a tool is withdrawn because it was blocking the camera view or key anatomy.
""".rstrip()

    structured = f"""
{hints.removesuffix('Return only a Markdown table. Do not include extra explanation.').strip()}
{right_ontology}

Return STRICT JSON only. Do not wrap the JSON in Markdown fences.
Use schema:
{{"segments": [{{"start_frame": 150, "end_frame": 180, "label": "action | no_action | unsure", "action_description": "...", "grasper_retraction": {{"changed": "yes | no | unsure", "start_direction": "left | right | upward | downward | not_retracted | unclear | unsure", "end_direction": "left | right | upward | downward | not_retracted | unclear | unsure", "description": "..."}}, "active_tools": [{{"tool_type": "Hook", "action_code": "DISSECT", "target_structure": "CysticArtery", "target_context_1": "(not set)", "target_context_2": "(not set)", "evidence": "short visible evidence"}}], "camera_movement": "zoom_in | zoom_out | reposition | unclear_movement | not_moving | unsure", "visibility": "clear | blurry | obscured | blurry_and_obscured | unsure"}}]}}
""".strip()
    return {"description_only": description, "hint_questions": hints, "structured_prediction": structured}


def build_extraction_prompt(example_id: str, frame_range: Sequence[int], method_name: str, prediction_text: str, options: dict) -> str:
    right_triplets = options["right_triplet_options"]
    right_tool_types = sorted({row[0] for row in right_triplets})
    right_action_codes = sorted({canonical_action_code(row[1]) for row in right_triplets})
    right_target_structures = sorted({row[2] for row in right_triplets})
    return f"""
Convert the model's action segmentation output into simple timestamped actions using ONLY the allowed audit_v11 options.

Clip:
- example_id: {example_id}
- frame range: {frame_range[0]}-{frame_range[1]}

Allowed left retraction_direction_code options:
{json.dumps(options['left_retraction_direction_code_options'], indent=2)}

Allowed right options:
- tool_type must be exactly one of:
{json.dumps(right_tool_types, indent=2)}
- action_code must be exactly one of:
{json.dumps(right_action_codes, indent=2)}
- target_structure must be exactly one of:
{json.dumps(right_target_structures, indent=2)}
- target_context_1 and target_context_2 must each be exactly one of:
{json.dumps(options['interface_options']['target_context'], indent=2)}

Allowed camera action_code options:
{json.dumps(options['camera_action_code_options'], indent=2)}

Allowed other action_code options:
{json.dumps(options['other_action_code_options'], indent=2)}

Rules:
- Use the prediction text only. Do not inspect GT labels.
- Preserve timestamps from the prediction.
- For left, output only retraction_direction_code.
- For right, output tool_type, action_code, target_structure, target_context_1, target_context_2.
- For camera, output only allowed camera action_code.
- Use other for actions such as ICG.
- If an actor is absent, use an empty list.
- If no allowed option matches, omit the row instead of inventing a new option.
- Return STRICT JSON only. No markdown.

Schema:
{{"method": "{method_name}", "left": [{{"start_frame": 150, "end_frame": 300, "retraction_direction_code": "KEEP_RETRACT_LATERAL", "confidence": 0.0, "evidence": "short phrase"}}], "right": [{{"start_frame": 150, "end_frame": 300, "tool_type": "Hook", "action_code": "DISSECT", "target_structure": "HepatocysticTriangle", "target_context_1": "(not set)", "target_context_2": "(not set)", "confidence": 0.0, "evidence": "short phrase"}}], "camera": [{{"start_frame": 150, "end_frame": 300, "action_code": "CAMERA_ZOOM_OUT", "confidence": 0.0, "evidence": "short phrase"}}], "other": [{{"start_frame": 150, "end_frame": 300, "action_code": "ICG_SWITCH", "confidence": 0.0, "evidence": "short phrase"}}]}}

Prediction text from method `{method_name}`:
{prediction_text}
""".strip()


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(cleaned[start : end + 1])


def normalize_extraction(raw: dict, clip_start: int, clip_end: int, options: dict) -> dict:
    allowed_left = set(options["left_retraction_direction_code_options"])
    allowed_right = {(row[0], canonical_action_code(row[1]), row[2]) for row in options["right_triplet_options"]}
    allowed_ctx = set(options["interface_options"]["target_context"])
    allowed_camera = set(options["camera_action_code_options"])
    allowed_other = {canonical_action_code(code) for code in options["other_action_code_options"]}

    out = {"left": [], "right": [], "camera": [], "other": []}
    for row in raw.get("left", []) or []:
        code = row.get("retraction_direction_code")
        if code not in allowed_left:
            continue
        out["left"].append(_basic_row(row, clip_start, clip_end, {"retraction_direction_code": code}))
    for row in raw.get("right", []) or []:
        triplet = (row.get("tool_type"), canonical_action_code(row.get("action_code")), row.get("target_structure"))
        if triplet not in allowed_right:
            continue
        ctx1 = row.get("target_context_1", "(not set)")
        ctx2 = row.get("target_context_2", "(not set)")
        ctx1 = ctx1 if ctx1 in allowed_ctx else "(not set)"
        ctx2 = ctx2 if ctx2 in allowed_ctx else "(not set)"
        out["right"].append(
            _basic_row(
                row,
                clip_start,
                clip_end,
                {
                    "tool_type": triplet[0],
                    "action_code": triplet[1],
                    "target_structure": triplet[2],
                    "triplet": list(triplet),
                    "target_context_1": ctx1,
                    "target_context_2": ctx2,
                },
            )
        )
    for row in raw.get("camera", []) or []:
        code = row.get("action_code")
        if code not in allowed_camera:
            continue
        out["camera"].append(_basic_row(row, clip_start, clip_end, {"action_code": code}))
    for row in raw.get("other", []) or []:
        code = canonical_action_code(row.get("action_code"))
        if code not in allowed_other:
            continue
        out["other"].append(_basic_row(row, clip_start, clip_end, {"action_code": code}))

    for actor in ACTORS:
        out[actor] = merge_adjacent_segments(sorted_segments(out[actor]), actor)
    return out


def _basic_row(row: dict, clip_start: int, clip_end: int, extra: dict) -> dict:
    start = frame_or_none(row.get("start_frame"))
    end = frame_or_none(row.get("end_frame"))
    if start is None or end is None:
        raise ValueError(f"Missing frame interval in row: {row}")
    start = max(int(clip_start), min(int(clip_end), start))
    end = max(int(clip_start), min(int(clip_end), end))
    if end < start:
        start, end = end, start
    out = {"start_frame": start, "end_frame": end}
    out.update(extra)
    if "confidence" in row:
        out["confidence"] = row.get("confidence")
    if "evidence" in row:
        out["evidence"] = row.get("evidence")
    return out


def actor_label(row: dict, actor: str, granularity: str) -> Any:
    actor = eval_source_actor(actor)
    if actor == "left":
        code = row.get("retraction_direction_code")
        if granularity == "exact":
            return code
        if granularity == "medium":
            return LEFT_EXACT_TO_MEDIUM.get(code, code)
        return LEFT_EXACT_TO_COARSE.get(code, code)
    if actor == "camera":
        code = row.get("action_code")
        if granularity == "exact":
            return code
        if granularity == "medium":
            return CAMERA_EXACT_TO_MEDIUM.get(code, code)
        return CAMERA_EXACT_TO_COARSE.get(code, code)
    if actor == "right":
        if granularity == "exact":
            return (
                row.get("tool_type"),
                row.get("action_code"),
                row.get("target_structure"),
                row.get("target_context_1", "(not set)"),
                row.get("target_context_2", "(not set)"),
            )
        if granularity == "medium":
            return (row.get("tool_type"), row.get("action_code"), row.get("target_structure"))
        return (row.get("action_code"), row.get("target_structure"))
    if actor == "other":
        return row.get("action_code")
    raise ValueError(actor)


RIGHT_CONTEXT_ABSENT_VALUES = {"(not set)", "", None}


def _right_context_set(label: Any) -> set:
    if not isinstance(label, tuple) or len(label) < 5:
        return set()
    return {
        value
        for value in label[3:5]
        if value not in RIGHT_CONTEXT_ABSENT_VALUES
    }


def right_exact_labels_match(left_label: Any, right_label: Any) -> bool:
    """Match exact right labels with unordered, partial target-context overlap.

    The core right-tool label is still exact: tool_type, action_code, and
    target_structure must all match. The two target-context slots are treated as
    an unordered optional set: if both labels omit context, that is a match; if
    either side specifies context, at least one non-empty context must overlap.
    """
    if not isinstance(left_label, tuple) or not isinstance(right_label, tuple):
        return left_label == right_label
    if len(left_label) < 5 or len(right_label) < 5:
        return left_label == right_label
    if left_label[:3] != right_label[:3]:
        return False
    left_contexts = _right_context_set(left_label)
    right_contexts = _right_context_set(right_label)
    if not left_contexts and not right_contexts:
        return True
    return bool(left_contexts & right_contexts)


def labels_match(left_label: Any, right_label: Any, actor: str, granularity: str) -> bool:
    if eval_source_actor(actor) == "right" and granularity == "exact":
        return right_exact_labels_match(left_label, right_label)
    return left_label == right_label


def merge_adjacent_segments(segments: List[dict], actor: str) -> List[dict]:
    if not segments:
        return []
    merged: List[dict] = []
    for seg in sorted_segments(segments):
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        if prev.get("end_frame") == seg.get("start_frame") and labels_match(
            actor_label(prev, actor, "exact"),
            actor_label(seg, actor, "exact"),
            actor,
            "exact",
        ):
            prev["end_frame"] = seg.get("end_frame")
        else:
            merged.append(dict(seg))
    return merged


def compute_iou(s_p: int, e_p: int, s_g: int, e_g: int) -> float:
    intersection = max(0, min(e_p, e_g) - max(s_p, s_g))
    union = max(e_p, e_g) - min(s_p, s_g)
    return 0.0 if union == 0 else intersection / union


def f1_at_k(pred_segments: List[dict], gt_segments: List[dict], actor: str, granularity: str, k: float) -> dict:
    pred_mapped = [
        (seg["start_frame"], seg["end_frame"], actor_label(seg, actor, granularity))
        for seg in pred_segments
    ]
    gt_mapped = [
        (seg["start_frame"], seg["end_frame"], actor_label(seg, actor, granularity))
        for seg in gt_segments
    ]
    tp = 0
    fp = 0
    matched_gt = set()
    for s_p, e_p, l_p in sorted(pred_mapped, key=lambda x: x[0]):
        best_iou = 0.0
        best_idx = None
        for idx, (s_g, e_g, l_g) in enumerate(gt_mapped):
            if idx in matched_gt or not labels_match(l_p, l_g, actor, granularity):
                continue
            iou = compute_iou(s_p, e_p, s_g, e_g)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= k:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_mapped) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def is_left_change_segment(seg: dict) -> bool:
    return LEFT_EXACT_TO_COARSE.get(seg.get("retraction_direction_code")) == "change"


def eval_source_actor(actor: str) -> str:
    return "left" if actor == "left_change" else actor


def evaluate_actor_by_video(pred_by_video: Dict[str, List[dict]], gt_by_video: Dict[str, List[dict]], actor: str) -> List[dict]:
    rows = []
    for granularity in GRANULARITIES:
        if actor == "other" and granularity != "exact":
            continue
        for k in IOU_THRESHOLDS:
            scores = []
            details = []
            for video_id, gt_segments in sorted(gt_by_video.items()):
                if not gt_segments:
                    continue
                pred_segments = pred_by_video.get(video_id, [])
                metric = f1_at_k(pred_segments, gt_segments, actor, granularity, k)
                scores.append(metric["f1"])
                details.append({"video_id": video_id, **metric})
            rows.append(
                {
                    "actor": actor,
                    "granularity": granularity,
                    "iou": k,
                    "f1_mean": mean(scores) if scores else None,
                    "f1_std": pstdev(scores) if len(scores) > 1 else 0.0,
                    "n_videos": len(scores),
                    "details": details,
                }
            )
    return rows


def build_actor_video_maps(records: List[dict], actor: str) -> Dict[str, List[dict]]:
    by_video: Dict[str, List[dict]] = defaultdict(list)
    source_actor = eval_source_actor(actor)
    for record in records:
        rows = list(record.get(source_actor, []))
        if actor == "left_change":
            rows = [row for row in rows if is_left_change_segment(row)]
        by_video[record["video_id"]].extend(rows)
    return {video_id: merge_adjacent_segments(sorted_segments(rows), actor) for video_id, rows in by_video.items()}


def evaluate_method_predictions(pred_records: List[dict], gt_records: List[dict]) -> List[dict]:
    rows = []
    for actor in EVAL_ACTORS:
        pred_by_video = build_actor_video_maps(pred_records, actor)
        gt_by_video = build_actor_video_maps(gt_records, actor)
        rows.extend(evaluate_actor_by_video(pred_by_video, gt_by_video, actor))
    return rows


def build_judge_prompt(example_id: str, frame_range: Sequence[int], gt_simple: dict, candidate: Any, method_name: str, candidate_kind: str) -> str:
    return f"""
You are evaluating surgical action segment predictions against GT timestamped actions.

First write a concise rationale, then assign a numeric score for each rubric item.
Return STRICT JSON only. No markdown.

Clip:
- example_id: {example_id}
- frame range: {frame_range[0]}-{frame_range[1]}

GT simple actions:
{json.dumps(gt_simple, indent=2)}

Candidate kind: {candidate_kind}
Method: {method_name}
Candidate:
{candidate if isinstance(candidate, str) else json.dumps(candidate, indent=2)}

Rubric items:
- left_action: left retraction direction/action correctness.
- left_time: left temporal localization.
- right_tool: right tool correctness.
- right_action: right action label correctness.
- right_target: right target_structure correctness.
- right_context: right target_context correctness.
- right_time: right temporal localization.
- camera_action: camera action label correctness.
- camera_time: camera temporal localization.
- other_action: other/ICG action correctness, or 1 if no GT and no prediction.
- other_time: other/ICG temporal localization, or 1 if no GT and no prediction.

Scores are 0, 0.5, or 1.
Return schema:
{{"method": "{method_name}", "candidate_kind": "{candidate_kind}", "rubric": [{{"item": "left_action", "rationale": "short reason first", "score": 0}}], "summary": "short summary"}}
""".strip()
