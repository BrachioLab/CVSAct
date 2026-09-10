"""Right-tool annotation helpers extracted from the Qwen evaluation notebook."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from openai import OpenAI
from tqdm.auto import tqdm


DEFAULT_PROMPT_VERSION = "2026-03-24-v2"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_IP_PATH = Path("/shared_data0/weiqiuy/ips/carnaroli.txt")
DEFAULT_API_KEY = "brachiokey"
DEFAULT_RIGHT_TOOL_VERSION = "rtv1"

TOOL_TYPES = ["Grasper", "Maryland", "Hook", "Irrigator", "Scissors", "Clipper"]
ANSWER_VOCAB = ("Yes", "No", "Uncertain")

REFERENCE_IMAGE_PATHS = {
    "holes": Path("few_shot_examples/shape/2d/holes_vs_no_holes.png"),
    "sharp": Path("few_shot_examples/shape/2d/sharp_vs_blunt.png"),
    "tapered": Path("few_shot_examples/shape/2d/tapered_vs_constant.png"),
    "tube": Path("few_shot_examples/shape/2d/tube_vs_with_tiip.png"),
    "two_tips": Path("few_shot_examples/shape/2d/two_tips_vs_one.png"),
    "wire": Path("few_shot_examples/shape/2d/wire_vs_not_wire.png"),
    "hollow": Path("few_shot_examples/shape/2d/hollow_vs_solid_tip.png"),
    "hinge": Path("few_shot_examples/shape/2d/hinge_vs_no_hinge.png"),
    "serration": Path("few_shot_examples/shape/2d/serration_vs_smooth.png"),
    "pivot": Path("few_shot_examples/shape/2d/pivot_vs_parallel.png"),
}


def read_default_base_url(ip_path: Path = DEFAULT_IP_PATH) -> str:
    carnaroli_ip = ip_path.read_text().strip()
    return f"http://{carnaroli_ip}:8001/v1"

INSTRUMENT_RUBRIC = [
    {
        "id": 1,
        "question": "Does the distal tube have hole-like openings?",
        "description": {
            "yes": "Yes if there are one or more visible holes on the tube.",
            "no": "No if the tube surface is smooth without holes.",
        },
        "reference_name": "holes",
        "reference_description": "Reference image for distinguishing visible hole-like openings on a tube from a smooth tube.",
        "scores": {
            "Grasper": ("No", 2),
            "Maryland": ("No", 2),
            "Hook": ("No", 2),
            "Irrigator": ("Yes", 2),
            "Scissors": ("No", 2),
            "Clipper": ("No", 2),
        },
    },
    {
        "id": 2,
        "question": "Does the tip (each tip, if multiple) have a cutting edge (blade)?",
        "description": {
            "yes": "Yes if a sharp cutting edge used for cutting is visible.",
            "no": "No if the tip does not have a blade.",
        },
        "reference_name": "sharp",
        "reference_description": "Reference image for distinguishing a sharp cutting edge from a non-bladed tip.",
        "scores": {
            "Grasper": ("No", 2),
            "Maryland": ("No", 2),
            "Hook": (None, 0),
            "Irrigator": ("No", 2),
            "Scissors": ("Yes", 2),
            "Clipper": ("No", 2),
        },
    },
    {
        "id": 3,
        "question": "Does the tip (each tip, if multiple) continuously narrow from base to end?",
        "description": {
            "yes": "Yes if the width decreases smoothly toward the tip (tapered).",
            "no": "No if the width is mostly constant or only slightly changes (not tapered).",
        },
        "reference_name": "tapered",
        "reference_description": "Reference image for distinguishing a tapered tip from a non-tapered tip.",
        "scores": {
            "Grasper": ("No", 2),
            "Maryland": ("Yes", 2),
            "Hook": ("No", 2),
            "Irrigator": ("No", 2),
            "Scissors": ("Yes", 2),
            "Clipper": (None, 0),
        },
    },
    {
        "id": 4,
        "question": "Does the distal end consist of a tube only (without any separate tip structure)?",
        "description": {
            "yes": "Yes if the end is just a tube.",
            "no": "No if there is a distinct tip or multiple tips attached.",
        },
        "reference_name": "tube",
        "reference_description": "Reference image for distinguishing a tube-only distal end from one with a distinct tip structure.",
        "scores": {
            "Grasper": ("No", 2),
            "Maryland": ("No", 2),
            "Hook": ("No", 2),
            "Irrigator": ("Yes", 2),
            "Scissors": ("No", 2),
            "Clipper": ("No", 2),
        },
    },
    {
        "id": 5,
        "question": "Does the tip split into two visible parts?",
        "description": {
            "yes": "Yes if two separate tips/jaws are visible.",
            "no": "No if only a single continuous tip is visible.",
        },
        "reference_name": "two_tips",
        "reference_description": "Reference image for distinguishing two visible tips or jaws from a single continuous tip.",
        "scores": {
            "Grasper": ("Yes", 1),
            "Maryland": ("Yes", 1),
            "Hook": ("No", 1),
            "Irrigator": ("No", 1),
            "Scissors": ("Yes", 1),
            "Clipper": ("Yes", 1),
        },
    },
    {
        "id": 6,
        "question": "Does the tip appear wire-like (very thin and rod-like, much thinner than typical jaws or blades)?",
        "description": {
            "yes": "Yes if the tip resembles a thin metal wire.",
            "no": "No if the tip is flat, broad, or has noticeable thickness.",
        },
        "reference_name": "wire",
        "reference_description": "Reference image for distinguishing a wire-like tip from a flat, broad, or thick tip.",
        "scores": {
            "Grasper": ("No", 2),
            "Maryland": ("No", 2),
            "Hook": ("Yes", 2),
            "Irrigator": ("No", 2),
            "Scissors": ("No", 2),
            "Clipper": ("No", 2),
        },
    },
    {
        "id": 7,
        "question": "Does the tip appear hollow (having an opening or lumen) rather than solid?",
        "description": {
            "yes": "Yes if an opening or inner cavity is visible.",
            "no": "No if the tip appears solid with no opening.",
        },
        "reference_name": "hollow",
        "reference_description": "Reference image for distinguishing a hollow tip from a solid tip.",
        "scores": {
            "Grasper": ("Yes", 2),
            "Maryland": ("No", 2),
            "Hook": ("No", 2),
            "Irrigator": ("No", 2),
            "Scissors": ("No", 2),
            "Clipper": ("No", 2),
        },
    },
    {
        "id": 8,
        "question": "Is a hinge (pivot joint) visible where the tip is attached to the shaft?",
        "description": {
            "yes": "Yes if a pivot, pin, or joint structure is visible between the tip and where it connects to the shaft.",
            "no": "No if no such joint is visible.",
        },
        "reference_name": "hinge",
        "reference_description": "Reference image for distinguishing a visible hinge or pivot from no hinge.",
        "scores": {
            "Grasper": ("Yes", 1),
            "Maryland": ("Yes", 1),
            "Hook": ("No", 1),
            "Irrigator": ("No", 1),
            "Scissors": ("Yes", 1),
            "Clipper": ("Yes", 1),
        },
    },
    {
        "id": 9,
        "question": "Do any visible tips have serrations (small, repeated tooth-like ridges along the edge)?",
        "description": {
            "yes": "Yes if a saw/comb-like pattern is visible.",
            "no": "No if the edge is smooth or flat.",
        },
        "reference_name": "serration",
        "reference_description": "Reference image for distinguishing serrated edges from smooth edges.",
        "scores": {
            "Grasper": ("Yes", 1),
            "Maryland": ("Yes", 2),
            "Hook": ("No", 2),
            "Irrigator": ("No", 2),
            "Scissors": ("No", 2),
            "Clipper": ("No", 2),
        },
    },
    {
        "id": 10,
        "question": "Do the two tips originate from a common pivot point at the base (opening like a fan)?",
        "description": {
            "yes": "Yes if both tips meet at a single joint and spread apart from that point.",
            "no": "No if the tips remain parallel and do not converge to a single pivot point.",
        },
        "reference_name": "pivot",
        "reference_description": "Reference image for distinguishing tips that fan open from a common pivot versus tips that remain parallel.",
        "scores": {
            "Grasper": ("Yes", 2),
            "Maryland": ("Yes", 2),
            "Hook": (None, 0),
            "Irrigator": (None, 0),
            "Scissors": ("Yes", 2),
            "Clipper": ("No", 2),
        },
    },
]

FRAME_CHECKS = [
    {
        "id": 1,
        "question": "Is the active working instrument (the one performing dissection, cutting, aspiration, or similar, not the instrument retracting the gallbladder) present in the frame?",
        "value": 1,
    },
    {
        "id": 2,
        "question": "Is the tip of the actively working instrument obscured by tissue?",
        "value": -1,
    },
    {
        "id": 3,
        "question": "Is the tip of the actively working instrument blurry?",
        "value": -1,
    },
]

QUESTION_SPECS = [
    {
        "qid": f"instrument_{item['id']}",
        "group": "instrument",
        "question": item["question"],
        "description": item["description"],
        "reference_name": item["reference_name"],
        "reference_description": item["reference_description"],
    }
    for item in INSTRUMENT_RUBRIC
] + [
    {
        "qid": f"frame_check_{item['id']}",
        "group": "frame_check",
        "question": item["question"],
        "description": None,
        "reference_name": None,
        "reference_description": None,
    }
    for item in FRAME_CHECKS
]

QUESTION_LOOKUP = {item["qid"]: item for item in QUESTION_SPECS}
RIGHT_TOOL_VERSION_SPECS: Dict[str, Dict[str, str]] = {
    "rtv0": {
        "name": "direct_whole_clip",
        "description": "Direct whole-clip right-tool prediction using all clip frames in chronological order.",
        "method_key": "prompt1_direct_whole_clip",
        "prediction_level": "clip",
    },
    "rtv1": {
        "name": "questionnaire_with_shapes_frame_aggregation",
        "description": "Legacy questionnaire-with-shape-examples inference with per-frame tool aggregation.",
        "method_key": "prompt2_all_in_one_json_with_shapes",
        "prediction_level": "frame",
    },
    "rtv2": {
        "name": "questionnaire_with_shapes_best_single_frame_clip",
        "description": "Questionnaire-with-shape-examples inference with clip-level tool prediction chosen from the frame with the highest winning aggregated tool score.",
        "method_key": "prompt2_all_in_one_json_with_shapes",
        "prediction_level": "clip",
    },
}
TEST_DEV_EXAMPLE_UIDS = [
    f"ex{i:02d}_coarse_01__mid" for i in range(1, 11)
]

QUESTIONNAIRE_SYSTEM_PROMPT = (
    "You answer surgical instrument questionnaires from images. "
    "Return strict JSON only."
)
DIRECT_LARGE_SYSTEM_PROMPT = (
    "You identify the active working right-side laparoscopic instrument from a whole clip. "
    "Return strict JSON only."
)
_TAG_LABELS = {"start": "First frame", "mid": "Middle frame", "end": "Last frame"}


def encode_image_file(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return mime, b64


def image_path_to_data_url(image_path: Path) -> str:
    mime, b64 = encode_image_file(image_path)
    return f"data:{mime};base64,{b64}"


def get_question_frame_paths(image_path: Path) -> List[tuple[str, Path]]:
    target_path = Path(image_path)
    try:
        frame_id = int(target_path.stem.split("_")[-1])
    except Exception:
        return [("target_frame", target_path)]

    frame_paths: List[tuple[str, Path]] = [("target_frame", target_path)]
    before_path = target_path.with_name(f"frame_{frame_id - 30:06d}{target_path.suffix}")
    after_path = target_path.with_name(f"frame_{frame_id + 30:06d}{target_path.suffix}")
    if before_path.exists():
        frame_paths.append(("frame_minus_1s", before_path))
    if after_path.exists():
        frame_paths.append(("frame_plus_1s", after_path))
    return frame_paths


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    mapping = {
        "yes": "Yes",
        "no": "No",
        "uncertain": "Uncertain",
        "unknown": "Unknown",
        "(absent)": "(absent)",
        "true": "Yes",
        "false": "No",
    }
    return mapping.get(text.lower(), text)


def normalize_tool_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    mapping = {
        "grasper": "Grasper",
        "maryland": "Maryland",
        "hook": "Hook",
        "irrigator": "Irrigator",
        "scissors": "Scissors",
        "clipper": "Clipper",
        "unknown": "Unknown",
        "(absent)": "(absent)",
        "(not set)": "(absent)",
        "not set": "(absent)",
        "none": "(absent)",
    }
    return mapping.get(text.lower(), text)


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned.startswith("{"):
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
    return json.loads(cleaned)


def load_reference_images(root_dir: Path) -> Dict[str, Dict[str, str]]:
    images: Dict[str, Dict[str, str]] = {}
    for name, rel_path in REFERENCE_IMAGE_PATHS.items():
        path = root_dir / rel_path
        mime, b64 = encode_image_file(path)
        images[name] = {
            "path": str(path),
            "mime": mime,
            "b64": b64,
        }
    return images


def get_right_tool_version_spec(version: str = DEFAULT_RIGHT_TOOL_VERSION) -> Dict[str, str]:
    try:
        return dict(RIGHT_TOOL_VERSION_SPECS[version])
    except KeyError as exc:
        valid = ", ".join(sorted(RIGHT_TOOL_VERSION_SPECS))
        raise ValueError(f"Unsupported right-tool version: {version}. Valid options: {valid}") from exc


def build_questionnaire_prompt(include_shape_examples: bool = True) -> str:
    question_blocks = []
    for item in QUESTION_SPECS:
        block = f"- {item['qid']}: {item['question']}"
        if item.get("description"):
            block += f"\n  Yes definition: {item['description'].get('yes', '')}"
            block += f"\n  No definition: {item['description'].get('no', '')}"
        block += "\n  Allowed answers: Yes, No, Uncertain"
        question_blocks.append(block)

    prompt = """
You are a surgical instrument identification assistant.
You are viewing one or more frames from the same laparoscopic cholecystectomy clip.
The first image is the target frame. Additional images, when shown, are neighboring frames from about 1 second before or after.
Your task is to answer the exact questionnaire below about the actively working RIGHT-side instrument.

Rules:
- Answer every question with exactly one of: Yes, No, Uncertain.
- For the 10 instrument questions, judge the visible instrument morphology only.
- For the 3 frame-check questions, judge image visibility or instrument presence only.
- Use all provided frames together when they help resolve visibility or ambiguity.
- If there is no clearly operating tool, answer Uncertain.
- If the active working instrument is not clearly present, use Uncertain for morphology questions when needed.
- If you cannot clearly see evidence for either the Yes features or the No features, answer Uncertain.
- Do not skip any question.
""".strip()

    if include_shape_examples:
        prompt += (
            "\n\nYou will also be shown 10 reference example images for the morphology "
            "questions. Use them as visual guidance for the shape concepts, then answer "
            "the questionnaire on the target frame."
        )

    answer_schema = ",\n".join(
        f'    "{item["qid"]}": "Yes|No|Uncertain"' for item in QUESTION_SPECS
    )
    prompt += f"""

Questions:
{chr(10).join(question_blocks)}

Return STRICT JSON only with this schema:
{{
  "answers": {{
{answer_schema}
  }},
  "brief_reasoning": "1-3 short sentences"
}}
"""
    return prompt.strip()


def build_prompt2_all_in_one_json_with_shapes_content(
    *,
    image_path: Path,
    reference_images: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": build_questionnaire_prompt(include_shape_examples=True),
        }
    ]
    for item in INSTRUMENT_RUBRIC:
        ref = reference_images[item["reference_name"]]
        content.append(
            {
                "type": "text",
                "text": (
                    f"Reference for instrument_{item['id']}: {item['question']}\n"
                    f"Yes definition: {item['description'].get('yes', '')}\n"
                    f"No definition: {item['description'].get('no', '')}"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{ref['mime']};base64,{ref['b64']}"},
            }
        )
    for _, frame_path in get_question_frame_paths(image_path):
        content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(frame_path)}})
    return content


def build_direct_large_prompt() -> str:
    tool_types_str = ", ".join(TOOL_TYPES)
    return f"""
You are a surgical video analysis assistant.

You are viewing one or more frames from the same laparoscopic cholecystectomy clip in chronological order.
We want to determine whether the RIGHT instrument is the active working instrument in this clip,
and if so, what tool it is.

CONTROLLED VOCABULARY for tool_type:
  [{tool_types_str}, Unknown, (not set)]

INSTRUMENT IDENTIFICATION NOTES:
- The RIGHT instrument is the active/working instrument in the surgical field.
- It is typically visible on the right side of the frame, performing the primary surgical action
  such as dissecting, cutting, cauterizing, or aspirating/irrigating.
- The working instrument is usually Hook, Maryland, Scissors, or Irrigator, but it may also be
  Grasper, Clipper, Unknown, or absent.
- Hook has a distinctive L-shaped or J-shaped hooked tip used for electrocautery dissection.
- Maryland is a bipolar dissector/forceps with two fine tapered jaws that open from a common pivot.
- Scissors have two blades that open and close for cutting.
- Irrigator is a smooth tubular suction-irrigation instrument, often with a hole-like opening.
- Grasper has two grasping jaws and is often broader/blunter than Maryland.
- Clipper has two jaws but they stay relatively parallel instead of opening like a fan from a pivot.

Your task:
1. Decide whether the active working RIGHT instrument is present and identifiable across this clip.
2. If yes, identify the tool type.

RULES:
- Use ONLY the controlled vocabulary for tool_type.
- Use all provided frames together, not just one frame.
- If the working right instrument is not clearly present or cannot be identified across the clip, set right_present=false.
- If right_present is false, set right_tool_type to "(not set)".

Output format (STRICT JSON only; no markdown fences; no extra keys):
{{
  "right_present": true/false,
  "right_tool_type": "Grasper|Maryland|Hook|Irrigator|Scissors|Clipper|Unknown|(not set)",
  "confidence": 0.0,
  "reasoning": "brief explanation of what you see"
}}

CONSTRAINTS:
- confidence is a float in [0,1].
- reasoning should be 1-2 sentences max.
""".strip()


def parse_direct_prediction_json(raw_text: str) -> Dict[str, Any]:
    try:
        parsed = extract_json_object(raw_text)
    except Exception as exc:
        return {
            "raw": raw_text,
            "parse_error": str(exc),
            "right_present": False,
            "right_tool_type": "(absent)",
            "confidence": None,
            "reasoning": "",
            "predicted_tool": "(absent)",
        }

    right_present = bool(parsed.get("right_present", False))
    right_tool_type = normalize_tool_label(parsed.get("right_tool_type", "(absent)"))
    predicted_tool = right_tool_type if right_present else "(absent)"
    if predicted_tool == "(not set)":
        predicted_tool = "(absent)"
    return {
        "raw": raw_text,
        "parse_error": None,
        "right_present": right_present,
        "right_tool_type": right_tool_type,
        "confidence": parsed.get("confidence"),
        "reasoning": parsed.get("reasoning", ""),
        "predicted_tool": normalize_tool_label(predicted_tool),
    }


def get_clip_image_paths(example: Mapping[str, Any]) -> List[tuple[str, Path]]:
    paths: List[tuple[str, Path]] = []
    for fid, path_str in example.get("all_frame_paths", []):
        tag = (example.get("frame_tags") or {}).get(fid)
        label = _TAG_LABELS.get(tag, f"Frame {fid}")
        paths.append((label, Path(path_str)))
    return paths


def run_prompt1_direct_whole_clip(
    *,
    client: OpenAI,
    model_id: str,
    example: Mapping[str, Any],
) -> Dict[str, Any]:
    frame_paths = get_clip_image_paths(example)
    prompt_record = {
        "method": "prompt1_direct_whole_clip",
        "model": model_id,
        "messages": [
            {"role": "system", "content": DIRECT_LARGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_direct_large_prompt()},
                    *[
                        {
                            "type": "image",
                            "label": label,
                            "path": str(path),
                        }
                        for label, path in frame_paths
                    ],
                ],
            },
        ],
        "frames": [(label, str(path)) for label, path in frame_paths],
    }
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": DIRECT_LARGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_direct_large_prompt()},
                    *[
                        {"type": "image_url", "image_url": {"url": image_path_to_data_url(path)}}
                        for _, path in frame_paths
                    ],
                ],
            },
        ],
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw_text = response.choices[0].message.content or ""
    result = parse_direct_prediction_json(raw_text)
    result["prompt_record"] = prompt_record
    return result


def compute_normalized_tool_scores(answer_map: Mapping[str, Any]) -> Dict[str, Any]:
    tool_breakdown: Dict[str, Any] = {}
    for tool in TOOL_TYPES:
        scored_items = []
        earned_points = 0
        total_possible = 0
        for item in INSTRUMENT_RUBRIC:
            expected, points = item["scores"].get(tool, (None, 0))
            if expected is None or points <= 0:
                continue
            total_possible += points
            observed = normalize_answer(answer_map.get(f"instrument_{item['id']}", "Uncertain"))
            expected = normalize_answer(expected)
            matched = observed == expected
            if matched:
                earned_points += points
            scored_items.append(
                {
                    "question_id": f"instrument_{item['id']}",
                    "expected": expected,
                    "observed": observed,
                    "points": points,
                    "matched": matched,
                    "earned": points if matched else 0,
                }
            )
        normalized_score = (earned_points / total_possible) if total_possible else 0.0
        tool_breakdown[tool] = {
            "earned_points": earned_points,
            "total_possible": total_possible,
            "normalized_score": normalized_score,
            "items": scored_items,
        }

    sorted_tools = sorted(
        ((tool, info["normalized_score"]) for tool, info in tool_breakdown.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    predicted_tool = sorted_tools[0][0] if sorted_tools else "Unknown"
    return {
        "tool_breakdown": tool_breakdown,
        "tool_scores": {tool: info["normalized_score"] for tool, info in tool_breakdown.items()},
        "sorted_tools": sorted_tools,
        "predicted_tool": predicted_tool,
    }


def aggregate_tool_prediction(answer_map: Mapping[str, Any]) -> Dict[str, Any]:
    present_answer = normalize_answer(answer_map.get("frame_check_1", "Uncertain"))
    if present_answer != "Yes":
        return {
            "predicted_tool": "(absent)",
            "tool_scores": {},
            "sorted_tools": [],
            "tool_breakdown": {},
        }
    return compute_normalized_tool_scores(answer_map)


def parse_questionnaire_response(raw_text: str) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text)
    answers = {
        q["qid"]: normalize_answer(parsed.get("answers", {}).get(q["qid"], "Uncertain"))
        for q in QUESTION_SPECS
    }
    return {
        "raw": raw_text,
        "answers": answers,
        "brief_reasoning": parsed.get("brief_reasoning", ""),
        "aggregation": aggregate_tool_prediction(answers),
    }


def run_prompt2_all_in_one_json_with_shapes(
    *,
    client: OpenAI,
    model_id: str,
    image_path: Path,
    reference_images: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": QUESTIONNAIRE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt2_all_in_one_json_with_shapes_content(
                    image_path=image_path,
                    reference_images=reference_images,
                ),
            },
        ],
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw_text = response.choices[0].message.content or ""
    return parse_questionnaire_response(raw_text)


def collect_annotation_answer_map(clip: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    row_map = {ann["row_id"].split("__")[-1]: ann for ann in clip["annotations"]}
    out = {}
    for row_key, ann in row_map.items():
        out[row_key] = {
            "sequence_answer": normalize_answer(ann.get("sequence_answer", "")),
            "frame_answers": {
                str(key): normalize_answer(value)
                for key, value in ann.get("frame_answers", {}).items()
            },
        }
    return out


def effective_gt_answer(answer_entry: Optional[Mapping[str, Any]], frame_id: int) -> str:
    if answer_entry is None:
        return ""
    frame_value = normalize_answer(answer_entry.get("frame_answers", {}).get(str(frame_id), ""))
    if frame_value:
        return frame_value
    return normalize_answer(answer_entry.get("sequence_answer", ""))


def derive_gt_tool_name(answer_map: Mapping[str, Mapping[str, Any]], frame_id: int) -> str:
    tool_answer = effective_gt_answer(answer_map.get("tool_name"), frame_id)
    if tool_answer:
        return tool_answer
    present_answer = effective_gt_answer(answer_map.get("frame_check_1"), frame_id)
    if present_answer == "No":
        return "(absent)"
    return ""


def derive_clip_level_tool_name(clip: Mapping[str, Any]) -> str:
    audited = clip.get("audited_right_tool_gt") or []
    normalized = [normalize_tool_label(value) for value in audited if normalize_tool_label(value)]
    return normalized[0] if normalized else ""


def build_eval_examples(
    *,
    annotation_dir: Path,
    frames_dir: Path,
    clip_types: Iterable[str] = ("coarse", "fine"),
    frame_tags: Iterable[str] = ("start", "mid", "end"),
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    allowed_clip_types = set(clip_types)
    allowed_frame_tags = tuple(frame_tags)
    for ann_path in sorted(annotation_dir.glob("*.json")):
        data = json.loads(ann_path.read_text())
        for clip in data.get("clips", []):
            clip_type = clip.get("clip_type")
            if clip_type not in allowed_clip_types:
                continue
            answer_map = collect_annotation_answer_map(clip)
            clip_level_tool_name = derive_clip_level_tool_name(clip)
            frame_tags_map = {
                int(fid_str): tag
                for fid_str, tag in clip.get("frame_tags", {}).items()
                if tag in allowed_frame_tags
            }
            ordered_frame_ids = [int(fid) for fid in clip.get("frame_ids", [])]
            if not ordered_frame_ids:
                ordered_frame_ids = sorted(frame_tags_map)
            all_frame_paths = []
            for fid in ordered_frame_ids:
                image_path = frames_dir / clip["video_id"] / f"frame_{fid:06d}.png"
                if image_path.exists():
                    all_frame_paths.append((fid, str(image_path)))
            tag_to_frame = {}
            for fid_str, tag in clip.get("frame_tags", {}).items():
                if tag in allowed_frame_tags:
                    tag_to_frame[tag] = int(fid_str)
            for tag in allowed_frame_tags:
                if tag not in tag_to_frame:
                    continue
                frame_id = tag_to_frame[tag]
                image_path = frames_dir / clip["video_id"] / f"frame_{frame_id:06d}.png"
                if not image_path.exists():
                    continue
                gt_answers = {}
                labeled_question_count = 0
                for question in QUESTION_SPECS:
                    gt_value = effective_gt_answer(answer_map.get(question["qid"]), frame_id)
                    gt_answers[question["qid"]] = gt_value
                    if gt_value:
                        labeled_question_count += 1
                gt_tool_name = derive_gt_tool_name(answer_map, frame_id)
                examples.append(
                    {
                        "example_uid": f"{clip['clip_uid']}__{tag}",
                        "video_id": clip["video_id"],
                        "clip_uid": clip["clip_uid"],
                        "clip_type": clip_type,
                        "criterion": clip["criterion"],
                        "frame_tag": tag,
                        "frame_id": frame_id,
                        "image_path": str(image_path),
                        "gt_answers": gt_answers,
                        "gt_tool_name": gt_tool_name,
                        "clip_level_tool_name": clip_level_tool_name,
                        "all_frame_paths": all_frame_paths,
                        "frame_tags": frame_tags_map,
                        "tool_matches_clip_level": bool(gt_tool_name)
                        and bool(clip_level_tool_name)
                        and gt_tool_name == clip_level_tool_name,
                        "labeled_question_count": labeled_question_count,
                    }
                )
    return examples


def filter_examples_by_split(
    examples: Iterable[Mapping[str, Any]],
    split: str = "all",
) -> List[Dict[str, Any]]:
    example_list = [dict(example) for example in examples]
    if split == "all":
        return example_list

    dev_uid_set = set(TEST_DEV_EXAMPLE_UIDS)
    if split == "test_dev":
        return [example for example in example_list if example["example_uid"] in dev_uid_set]
    if split == "test_rest":
        return [example for example in example_list if example["example_uid"] not in dev_uid_set]
    raise ValueError(f"Unsupported split: {split}")


def cache_entry_is_current(entry: Any, prompt_version: str) -> bool:
    return isinstance(entry, dict) and entry.get("prompt_version") == prompt_version


def create_openai_client(
    *,
    base_url: Optional[str] = None,
    api_key: str = DEFAULT_API_KEY,
) -> OpenAI:
    if base_url is None:
        base_url = read_default_base_url()
    return OpenAI(base_url=base_url, api_key=api_key)


def run_inference_for_examples(
    *,
    client: OpenAI,
    model_id: str,
    prompt_version: str,
    examples: List[Dict[str, Any]],
    reference_images: Mapping[str, Mapping[str, str]],
    cache: Optional[MutableMapping[str, Any]] = None,
    overwrite: bool = False,
    right_tool_version: str = DEFAULT_RIGHT_TOOL_VERSION,
) -> Dict[str, Any]:
    version_spec = get_right_tool_version_spec(right_tool_version)
    cache_dict: Dict[str, Any] = dict(cache or {})
    method_key = version_spec["method_key"]
    work_examples = examples
    if method_key == "prompt1_direct_whole_clip":
        seen_clip_uids = set()
        work_examples = []
        for ex in examples:
            clip_uid = ex["clip_uid"]
            if clip_uid in seen_clip_uids:
                continue
            seen_clip_uids.add(clip_uid)
            work_examples.append(ex)

    for ex in tqdm(work_examples, total=len(work_examples), desc=f"Right tool {right_tool_version}"):
        ex_key = ex["example_uid"]
        ex_cache = dict(cache_dict.get(ex_key, {}))
        if overwrite or not cache_entry_is_current(ex_cache.get(method_key), prompt_version):
            if method_key == "prompt1_direct_whole_clip":
                clip_examples = [item for item in examples if item["clip_uid"] == ex["clip_uid"]]
                result = run_prompt1_direct_whole_clip(
                    client=client,
                    model_id=model_id,
                    example=ex,
                )
                result["prompt_version"] = prompt_version
                result["right_tool_version"] = right_tool_version
                for clip_example in clip_examples:
                    clip_cache = dict(cache_dict.get(clip_example["example_uid"], {}))
                    clip_cache[method_key] = dict(result)
                    cache_dict[clip_example["example_uid"]] = clip_cache
                continue
            else:
                result = run_prompt2_all_in_one_json_with_shapes(
                    client=client,
                    model_id=model_id,
                    image_path=Path(ex["image_path"]),
                    reference_images=reference_images,
                )
            result["prompt_version"] = prompt_version
            result["right_tool_version"] = right_tool_version
            ex_cache[method_key] = result
        cache_dict[ex_key] = ex_cache
    return cache_dict


def _tool_scores_from_prediction(
    prediction: Mapping[str, Any],
    *,
    method_key: str,
) -> Dict[str, float]:
    if method_key == "prompt1_direct_whole_clip":
        predicted_tool = normalize_tool_label(prediction.get("predicted_tool", ""))
        if not predicted_tool or predicted_tool not in TOOL_TYPES:
            return {}
        confidence = prediction.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        return {predicted_tool: confidence}

    aggregation = prediction.get("aggregation", {}) if isinstance(prediction, Mapping) else {}
    tool_scores = aggregation.get("normalized_scores") or aggregation.get("tool_scores") or {}
    return {tool: float(score) for tool, score in tool_scores.items()}


def _build_frame_prediction_rows(
    *,
    examples: Iterable[Mapping[str, Any]],
    cache: Mapping[str, Any],
    method_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        prediction = cache.get(ex["example_uid"], {}).get(method_key, {})
        aggregation = prediction.get("aggregation", {})
        if method_key == "prompt1_direct_whole_clip":
            pred_tool = normalize_tool_label(prediction.get("predicted_tool", ""))
        else:
            pred_tool = normalize_tool_label(aggregation.get("predicted_tool", ""))
        gt_tool = normalize_tool_label(ex.get("gt_tool_name", ""))
        tool_scores = _tool_scores_from_prediction(prediction, method_key=method_key)
        winning_score = max(tool_scores.values()) if tool_scores else None
        rows.append(
            {
                "example_uid": ex["example_uid"],
                "video_id": ex["video_id"],
                "clip_uid": ex["clip_uid"],
                "clip_type": ex.get("clip_type"),
                "criterion": ex["criterion"],
                "frame_tag": ex["frame_tag"],
                "frame_id": ex["frame_id"],
                "image_path": ex["image_path"],
                "gt_tool_name": gt_tool,
                "clip_gt_tool_name": normalize_tool_label(ex.get("clip_level_tool_name", "")),
                "predicted_tool_name": pred_tool,
                "correct": int(bool(gt_tool) and bool(pred_tool) and gt_tool == pred_tool),
                "answers": prediction.get("answers", {}),
                "aggregation": aggregation,
                "tool_scores": tool_scores,
                "winning_score": winning_score,
                "gt_tool_score": tool_scores.get(normalize_tool_label(ex.get("clip_level_tool_name", ""))),
                "brief_reasoning": prediction.get("brief_reasoning", ""),
                "prompt_version": prediction.get("prompt_version"),
                "right_tool_version": prediction.get("right_tool_version"),
            }
        )
    return rows


def _build_best_single_frame_clip_rows(
    *,
    examples: Iterable[Mapping[str, Any]],
    cache: Mapping[str, Any],
    method_key: str,
) -> List[Dict[str, Any]]:
    frame_rows = _build_frame_prediction_rows(examples=examples, cache=cache, method_key=method_key)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in frame_rows:
        grouped.setdefault(row["clip_uid"], []).append(row)

    clip_rows: List[Dict[str, Any]] = []
    for clip_uid, rows in sorted(grouped.items()):
        valid_rows = [row for row in rows if row.get("tool_scores")]
        if not valid_rows:
            continue

        def candidate_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
            return (
                float(row.get("winning_score") or 0.0),
                float(row.get("gt_tool_score") or 0.0),
                -int(row.get("frame_id") or 0),
            )

        best_row = max(valid_rows, key=candidate_key)
        clip_gt_tool = normalize_tool_label(best_row.get("clip_gt_tool_name", ""))
        predicted_tool = normalize_tool_label(best_row.get("predicted_tool_name", ""))
        clip_rows.append(
            {
                "clip_uid": clip_uid,
                "video_id": best_row.get("video_id"),
                "clip_type": best_row.get("clip_type"),
                "criterion": best_row.get("criterion"),
                "chosen_example_uid": best_row.get("example_uid"),
                "chosen_frame_tag": best_row.get("frame_tag"),
                "chosen_frame_id": best_row.get("frame_id"),
                "clip_gt_tool_name": clip_gt_tool,
                "predicted_tool_name": predicted_tool,
                "gt_tool_score": best_row.get("gt_tool_score"),
                "winning_score": best_row.get("winning_score"),
                "tool_scores": best_row.get("tool_scores", {}),
                "correct": int(bool(clip_gt_tool) and bool(predicted_tool) and clip_gt_tool == predicted_tool),
                "brief_reasoning": best_row.get("brief_reasoning", ""),
                "prompt_version": best_row.get("prompt_version"),
                "right_tool_version": best_row.get("right_tool_version"),
            }
        )
    return clip_rows


def _build_direct_clip_rows(
    *,
    examples: Iterable[Mapping[str, Any]],
    cache: Mapping[str, Any],
    method_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(str(example["clip_uid"]), []).append(example)

    clip_rows: List[Dict[str, Any]] = []
    for clip_uid, clip_examples in sorted(grouped.items()):
        representative = sorted(
            clip_examples,
            key=lambda item: (str(item.get("frame_tag", "")), int(item.get("frame_id") or 0)),
        )[0]
        prediction = cache.get(representative["example_uid"], {}).get(method_key, {})
        predicted_tool = normalize_tool_label(prediction.get("predicted_tool", ""))
        clip_gt_tool = normalize_tool_label(representative.get("clip_level_tool_name", ""))
        try:
            confidence = float(prediction.get("confidence"))
        except (TypeError, ValueError):
            confidence = None
        clip_rows.append(
            {
                "clip_uid": clip_uid,
                "video_id": representative.get("video_id"),
                "clip_type": representative.get("clip_type"),
                "criterion": representative.get("criterion"),
                "chosen_example_uid": representative.get("example_uid"),
                "chosen_frame_tag": representative.get("frame_tag"),
                "chosen_frame_id": representative.get("frame_id"),
                "clip_gt_tool_name": clip_gt_tool,
                "predicted_tool_name": predicted_tool,
                "gt_tool_score": None,
                "winning_score": confidence,
                "tool_scores": _tool_scores_from_prediction(prediction, method_key=method_key),
                "correct": int(bool(clip_gt_tool) and bool(predicted_tool) and clip_gt_tool == predicted_tool),
                "brief_reasoning": prediction.get("reasoning", ""),
                "prompt_version": prediction.get("prompt_version"),
                "right_tool_version": prediction.get("right_tool_version"),
            }
        )
    return clip_rows


def build_prediction_rows(
    *,
    examples: Iterable[Mapping[str, Any]],
    cache: Mapping[str, Any],
    method_key: Optional[str] = None,
    right_tool_version: str = DEFAULT_RIGHT_TOOL_VERSION,
) -> List[Dict[str, Any]]:
    version_spec = get_right_tool_version_spec(right_tool_version)
    resolved_method_key = method_key or version_spec["method_key"]
    if resolved_method_key == "prompt1_direct_whole_clip":
        return _build_direct_clip_rows(
            examples=examples,
            cache=cache,
            method_key=resolved_method_key,
        )
    if version_spec["prediction_level"] == "clip":
        return _build_best_single_frame_clip_rows(
            examples=examples,
            cache=cache,
            method_key=resolved_method_key,
        )
    return _build_frame_prediction_rows(
        examples=examples,
        cache=cache,
        method_key=resolved_method_key,
    )


def summarize_predictions(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    gt_key = "gt_tool_name"
    if rows and any("clip_gt_tool_name" in row for row in rows):
        gt_key = "clip_gt_tool_name"
    evaluated = [row for row in rows if row.get(gt_key) and row.get("predicted_tool_name")]
    correct = sum(int(row["correct"]) for row in evaluated)
    summary = {
        "n_rows": len(rows),
        "n_evaluated": len(evaluated),
        "n_correct": correct,
        "accuracy": (correct / len(evaluated)) if evaluated else None,
    }
    if rows and any(row.get("clip_type") is not None for row in rows):
        by_clip_type: Dict[str, Dict[str, Any]] = {}
        for clip_type in sorted({row.get("clip_type") for row in rows if row.get("clip_type")}):
            split_rows = [row for row in evaluated if row.get("clip_type") == clip_type]
            split_correct = sum(int(row["correct"]) for row in split_rows)
            by_clip_type[clip_type] = {
                "n_evaluated": len(split_rows),
                "n_correct": split_correct,
                "accuracy": (split_correct / len(split_rows)) if split_rows else None,
            }
        summary["by_clip_type"] = by_clip_type
    return summary
