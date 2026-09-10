"""Camera annotation helpers extracted from the organ-question notebook."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm


DEFAULT_PROMPT_VERSION = "2026-03-30-camera-organ-v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_IP_PATH = Path("/shared_data0/weiqiuy/ips/carnaroli.txt")
DEFAULT_API_KEY = "brachiokey"
DEFAULT_CAMERA_VERSION = "v1"
DEFAULT_EVAL_SEED = 42
DEFAULT_NOTEBOOK_SAMPLE_N_PER_CATEGORY = 3

CAMERA_VERSION_SPECS: Dict[str, Dict[str, str]] = {
    "v0": {
        "name": "direct_whole_clip_camera_action",
        "description": "Direct whole-clip camera action prediction among zoom in / zoom out / reposition / not set.",
        "method_key": "camera_direct_whole_clip",
    },
    "v1": {
        "name": "fixed_organs_size_vote",
        "description": "Predict size change for a fixed list of organs, then majority-vote camera action from organ change answers.",
        "method_key": "camera_fixed_organs_size_vote",
    },
    "v2": {
        "name": "adaptive_organs_detect_then_size_vote",
        "description": "First detect visible organs, then predict size change only for predicted organs, then majority-vote camera action.",
        "method_key": "camera_adaptive_organs_size_vote",
    },
}

ORGANS = [
    ("gallbladder", "gallbladder"),
    ("liver", "liver"),
    ("hepatocystic_triangle", "hepatocystic triangle"),
    ("omentum", "omentum"),
    ("cystic_duct", "cystic duct"),
    ("cystic_artery", "cystic artery"),
]
ORGAN_NAME_BY_ID = dict(ORGANS)

CAMERA_ACTION_CODES = [
    "CAMERA_ZOOM_IN",
    "CAMERA_ZOOM_OUT",
    "CAMERA_REPOSITION",
    "(not set)",
]
CAMERA_GT_PRIORITY = [
    "CAMERA_ZOOM_IN",
    "CAMERA_ZOOM_OUT",
    "CAMERA_REPOSITION",
]

ORGAN_DETECTION_SYSTEM_PROMPT = (
    "You are a surgical anatomy assistant. "
    "You are viewing the first frame of a laparoscopic cholecystectomy video clip. "
    "List all anatomical structures that are clearly visible in this frame from the following options: "
    "gallbladder, liver, hepatocystic triangle, omentum, cystic duct, cystic artery. "
    'Reply with a comma-separated list. If none are visible, reply "none". '
    "Do not add any explanation."
)

SIZE_QUESTION_SYSTEM_PROMPT = (
    "You are a surgical anatomy assistant. "
    "You are viewing frames from a laparoscopic cholecystectomy video clip in chronological order. "
    "Your task is to answer a visual question about how an anatomical structure has changed in size "
    "from the first frame to the last frame. "
    "Reply with exactly one of: Becomes larger / Becomes smaller / Stayed the same / Uncertain. "
    "Do not add any explanation."
)

DIRECT_CAMERA_SYSTEM_PROMPT = (
    "You identify the camera action in a laparoscopic cholecystectomy clip. "
    "Return strict JSON only."
)

DIRECT_CAMERA_PROMPT = """
You are a surgical video analysis assistant.
You are viewing one or more frames from the same laparoscopic cholecystectomy clip in chronological order.

Task:
Predict the dominant camera action across this clip.

Allowed action codes:
- CAMERA_ZOOM_IN: the camera moves closer and structures appear larger.
- CAMERA_ZOOM_OUT: the camera moves farther away and structures appear smaller.
- CAMERA_REPOSITION: the camera changes angle / framing / viewpoint without a clear zoom change, or the motion is too uncertain to map cleanly to zoom in or zoom out.
- (not set): no meaningful camera motion is visible.

Rules:
- Use all provided frames together.
- Focus only on camera behavior, not tool motion.
- If the visual evidence is ambiguous between same-view and reposition, choose CAMERA_REPOSITION.
- Return strict JSON only.

Return this schema:
{
  "predicted_action_code": "CAMERA_ZOOM_IN|CAMERA_ZOOM_OUT|CAMERA_REPOSITION|(not set)",
  "confidence": 0.0,
  "reasoning": "1-3 short sentences"
}
""".strip()

_TAG_LABELS = {"start": "First frame", "mid": "Middle frame", "end": "Last frame"}


def get_camera_version_spec(version: str = DEFAULT_CAMERA_VERSION) -> Dict[str, str]:
    try:
        return dict(CAMERA_VERSION_SPECS[version])
    except KeyError as exc:
        valid = ", ".join(sorted(CAMERA_VERSION_SPECS))
        raise ValueError(f"Unsupported camera version: {version}. Valid options: {valid}") from exc


def read_default_base_url(ip_path: Path = DEFAULT_IP_PATH) -> str:
    carnaroli_ip = ip_path.read_text().strip()
    return f"http://{carnaroli_ip}:8001/v1"


def create_openai_client(
    *,
    base_url: Optional[str] = None,
    api_key: str = DEFAULT_API_KEY,
) -> OpenAI:
    if base_url is None:
        base_url = read_default_base_url()
    return OpenAI(base_url=base_url, api_key=api_key)


def encode_image_file(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return mime, b64


def image_path_to_data_url(path: Path) -> str:
    mime, b64 = encode_image_file(path)
    return f"data:{mime};base64,{b64}"


def compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def normalize_camera_action_code(value: Any) -> str:
    if value is None:
        return "(not set)"
    text = str(value).strip()
    if not text:
        return "(not set)"
    upper = text.upper()
    if upper in CAMERA_ACTION_CODES:
        return upper
    lower = text.lower()
    if lower in {"camera_zoom_in", "zoom_in", "zoomin"}:
        return "CAMERA_ZOOM_IN"
    if lower in {"camera_zoom_out", "zoom_out", "zoomout"}:
        return "CAMERA_ZOOM_OUT"
    if lower in {"camera_reposition", "reposition", "uncertain"}:
        return "CAMERA_REPOSITION"
    if lower in {"same", "stayed the same", "none", "not set", "(not set)"}:
        return "(not set)"
    return "(not set)"


def normalize_size_answer(text: Any) -> str:
    if not text:
        return "Stayed the same"
    cleaned = str(text).split("</think>")[-1].strip()
    lower = cleaned.lower()
    if "uncertain" in lower:
        return "Uncertain"
    if "becomes larger" in lower or lower == "larger":
        return "Becomes larger"
    if "becomes smaller" in lower or lower == "smaller":
        return "Becomes smaller"
    if "stayed the same" in lower or "stays the same" in lower or lower == "same":
        return "Stayed the same"
    if "larger" in lower:
        return "Becomes larger"
    if "smaller" in lower:
        return "Becomes smaller"
    if "same" in lower:
        return "Stayed the same"
    return "Stayed the same"


def map_size_answer_to_camera_code(size_answer: str) -> str:
    size_answer = normalize_size_answer(size_answer)
    mapping = {
        "Becomes larger": "CAMERA_ZOOM_IN",
        "Becomes smaller": "CAMERA_ZOOM_OUT",
        "Stayed the same": "(not set)",
        "Uncertain": "CAMERA_REPOSITION",
    }
    return mapping.get(size_answer, "CAMERA_REPOSITION")


def parse_frame_id(keyframe_str: str | None) -> Optional[int]:
    if not keyframe_str:
        return None
    match = re.search(r"frame_(\d+)$", str(keyframe_str))
    return int(match.group(1)) if match else None


def load_video_frame_ids(labels_dir: Path) -> Dict[str, List[int]]:
    video_frame_ids: Dict[str, List[int]] = {}
    for csv_path in sorted(labels_dir.glob("*/frame.csv")):
        df = pd.read_csv(csv_path)
        video_frame_ids[csv_path.parent.name] = df["frame_id"].astype(int).tolist()
    return video_frame_ids


def derive_camera_gt_action(actions_ranked: Iterable[Mapping[str, Any]]) -> str:
    camera_codes = [
        normalize_camera_action_code(action.get("action_code"))
        for action in actions_ranked
        if action.get("actor_role") == "camera"
    ]
    for code in CAMERA_GT_PRIORITY:
        if code in camera_codes:
            return code
    return "(not set)"


def _make_clip_paths(
    *,
    frame_ids: Iterable[int],
    video_id: str,
    frames_dir: Path,
    frame_tags: Mapping[int, str],
) -> List[tuple[int, str]]:
    paths: List[tuple[int, str]] = []
    for fid in frame_ids:
        frame_path = frames_dir / video_id / f"frame_{int(fid):06d}.png"
        if frame_path.exists():
            paths.append((int(fid), str(frame_path)))
    if paths:
        return paths

    # Fallback to tagged frames only if sampled frame list is empty.
    tagged_paths: List[tuple[int, str]] = []
    for fid in sorted(frame_tags):
        frame_path = frames_dir / video_id / f"frame_{int(fid):06d}.png"
        if frame_path.exists():
            tagged_paths.append((int(fid), str(frame_path)))
    return tagged_paths


def build_eval_examples(
    *,
    annotation_dir: Path,
    frames_dir: Path,
    labels_dir: Path,
    include_clip_types: Iterable[str] = ("coarse", "fine"),
) -> List[Dict[str, Any]]:
    video_frame_ids = load_video_frame_ids(labels_dir)
    include_set = set(include_clip_types)
    examples: List[Dict[str, Any]] = []

    for ann_path in sorted(annotation_dir.glob("*.json")):
        records = json.loads(ann_path.read_text())
        for item in records:
            video_id = item["video_id"]

            if "coarse" in include_set:
                coarse = item.get("coarse") or {}
                ann = coarse.get("annotation") or {}
                keyframes = coarse.get("keyframes") or {}
                start_frame_id = parse_frame_id(keyframes.get("start"))
                mid_frame_id = parse_frame_id(keyframes.get("mid"))
                end_frame_id = parse_frame_id(keyframes.get("end"))
                if start_frame_id is not None and end_frame_id is not None:
                    clip_start = coarse.get("start_frame") or start_frame_id
                    clip_end = coarse.get("end_frame") or end_frame_id
                    sampled_frame_ids = [
                        fid for fid in video_frame_ids.get(video_id, [])
                        if clip_start <= fid <= clip_end
                    ]
                    frame_tags = {
                        fid: tag
                        for fid, tag in [
                            (start_frame_id, "start"),
                            (mid_frame_id, "mid"),
                            (end_frame_id, "end"),
                        ]
                        if fid is not None
                    }
                    all_frame_paths = _make_clip_paths(
                        frame_ids=sampled_frame_ids,
                        video_id=video_id,
                        frames_dir=frames_dir,
                        frame_tags=frame_tags,
                    )
                    if all_frame_paths:
                        examples.append(
                            {
                                "example_uid": item["example_id"],
                                "clip_uid": item["example_id"],
                                "video_id": video_id,
                                "criterion": item.get("criterion"),
                                "clip_type": "coarse",
                                "all_frame_paths": all_frame_paths,
                                "frame_tags": frame_tags,
                                "gt_action_code": derive_camera_gt_action(ann.get("actions_ranked") or []),
                            }
                        )

            if "fine" in include_set:
                for fine in item.get("fine") or []:
                    ann = fine.get("annotation") or {}
                    keyframes = fine.get("keyframes") or {}
                    start_frame_id = parse_frame_id(keyframes.get("start"))
                    end_frame_id = parse_frame_id(keyframes.get("end"))
                    mid_frame_id = fine.get("mid_frame")
                    if mid_frame_id is None and start_frame_id is not None and end_frame_id is not None:
                        mid_frame_id = (start_frame_id + end_frame_id) // 2
                    if start_frame_id is None or end_frame_id is None:
                        continue
                    clip_start = fine.get("start_frame") or start_frame_id
                    clip_end = fine.get("end_frame") or end_frame_id
                    sampled_frame_ids = [
                        fid for fid in video_frame_ids.get(video_id, [])
                        if clip_start <= fid <= clip_end
                    ]
                    if not sampled_frame_ids:
                        sampled_frame_ids = sorted({start_frame_id, int(mid_frame_id), end_frame_id})
                    frame_tags = {
                        int(fid): tag
                        for fid, tag in [
                            (start_frame_id, "start"),
                            (mid_frame_id, "mid"),
                            (end_frame_id, "end"),
                        ]
                        if fid is not None
                    }
                    all_frame_paths = _make_clip_paths(
                        frame_ids=sampled_frame_ids,
                        video_id=video_id,
                        frames_dir=frames_dir,
                        frame_tags=frame_tags,
                    )
                    if all_frame_paths:
                        examples.append(
                            {
                                "example_uid": fine["fine_id"],
                                "clip_uid": fine["fine_id"],
                                "video_id": video_id,
                                "criterion": item.get("criterion"),
                                "clip_type": "fine",
                                "all_frame_paths": all_frame_paths,
                                "frame_tags": frame_tags,
                                "gt_action_code": derive_camera_gt_action(ann.get("actions_ranked") or []),
                            }
                        )

    return examples


def select_notebook_eval_examples(
    all_examples: Iterable[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_EVAL_SEED,
    n_per_category: int = DEFAULT_NOTEBOOK_SAMPLE_N_PER_CATEGORY,
) -> List[Dict[str, Any]]:
    """Match the notebook's balanced 12-example coarse subset.

    Selects coarse clips only, with distinct videos across:
    - zoom_in
    - zoom_out
    - reposition
    - stayed_same
    """

    coarse_examples = [dict(ex) for ex in all_examples if ex.get("clip_type") == "coarse"]
    rng = random.Random(seed)

    zoom_in = [ex for ex in coarse_examples if ex.get("gt_action_code") == "CAMERA_ZOOM_IN"]
    zoom_out = [ex for ex in coarse_examples if ex.get("gt_action_code") == "CAMERA_ZOOM_OUT"]
    reposition = [ex for ex in coarse_examples if ex.get("gt_action_code") == "CAMERA_REPOSITION"]
    stayed_same = [ex for ex in coarse_examples if ex.get("gt_action_code") == "(not set)"]

    used_videos = set()
    selected: List[Dict[str, Any]] = []

    def pick(candidates: List[Dict[str, Any]], category_name: str) -> List[Dict[str, Any]]:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        picked: List[Dict[str, Any]] = []
        for ex in shuffled:
            if ex["video_id"] not in used_videos:
                ex["selected_category"] = category_name
                picked.append(ex)
                used_videos.add(ex["video_id"])
            if len(picked) == n_per_category:
                break
        return picked

    for category_name, bucket in [
        ("zoom_in", zoom_in),
        ("zoom_out", zoom_out),
        ("reposition", reposition),
        ("stayed_same", stayed_same),
    ]:
        selected.extend(pick(bucket, category_name))

    return selected


def get_clip_image_paths(ex: Mapping[str, Any], uses_first_frame_only: bool = False) -> List[tuple[str, Path]]:
    all_frame_paths = ex.get("all_frame_paths") or []
    if not all_frame_paths:
        return []
    if uses_first_frame_only:
        first_fid, first_path = all_frame_paths[0]
        _ = first_fid
        return [("First frame", Path(first_path))]

    frame_tags = ex.get("frame_tags", {})
    paths: List[tuple[str, Path]] = []
    for fid, path_str in all_frame_paths:
        tag = frame_tags.get(fid)
        label = _TAG_LABELS.get(tag, f"Frame {fid}")
        paths.append((label, Path(path_str)))
    return paths


def build_prompt_signature(prompt_record: Mapping[str, Any]) -> str:
    canonical = json.dumps(prompt_record, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_prompt_record(
    *,
    method_key: str,
    prompt_version: str,
    model_id: str,
    user_text: str,
    system_prompt: str,
    frame_paths: List[tuple[str, Path]],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    image_records = [
        {
            "type": "image",
            "label": label,
            "path": str(path),
            "sha256": compute_file_sha256(path),
        }
        for label, path in frame_paths
    ]
    payload = {
        "method": method_key,
        "model": model_id,
        "prompt_version": prompt_version,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}, *image_records]},
        ],
        "frames": [(label, str(path)) for label, path in frame_paths],
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _build_chat_messages(
    *,
    system_prompt: str,
    user_text: str,
    frame_paths: List[tuple[str, Path]],
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for _, path in frame_paths:
        user_content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(path)}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _direct_camera_prompt_record(
    *,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
) -> Dict[str, Any]:
    frame_paths = get_clip_image_paths(example, uses_first_frame_only=False)
    return _build_prompt_record(
        method_key="camera_direct_whole_clip",
        prompt_version=prompt_version,
        model_id=model_id,
        user_text=DIRECT_CAMERA_PROMPT,
        system_prompt=DIRECT_CAMERA_SYSTEM_PROMPT,
        frame_paths=frame_paths,
    )


def _fixed_organ_prompt_records(
    *,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
    organ_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    frame_paths = get_clip_image_paths(example, uses_first_frame_only=False)
    selected_organ_ids = list(organ_ids) if organ_ids is not None else [organ_id for organ_id, _ in ORGANS]
    prompt_records: List[Dict[str, Any]] = []
    for organ_id in selected_organ_ids:
        organ_name = ORGAN_NAME_BY_ID[organ_id]
        user_text = _build_size_question(organ_name)
        prompt_records.append(
            _build_prompt_record(
                method_key="camera_fixed_organs_size_vote",
                prompt_version=prompt_version,
                model_id=model_id,
                user_text=user_text,
                system_prompt=SIZE_QUESTION_SYSTEM_PROMPT,
                frame_paths=frame_paths,
                extra={"organ_id": organ_id},
            )
        )
    return prompt_records


def _adaptive_detection_prompt_record(
    *,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
) -> Dict[str, Any]:
    detect_frame_paths = get_clip_image_paths(example, uses_first_frame_only=True)
    detect_text = (
        "What anatomical structures from the following list are visible in this frame: "
        "gallbladder, liver, hepatocystic triangle, omentum, cystic duct, cystic artery? "
        'Reply with a comma-separated list of visible structures, or "none" if none are visible.'
    )
    return _build_prompt_record(
        method_key="camera_adaptive_organs_size_vote_detection",
        prompt_version=prompt_version,
        model_id=model_id,
        user_text=detect_text,
        system_prompt=ORGAN_DETECTION_SYSTEM_PROMPT,
        frame_paths=detect_frame_paths,
    )


def _extract_prompt_signatures(entry: Mapping[str, Any]) -> List[str]:
    signatures = entry.get("prompt_signatures")
    if isinstance(signatures, list) and signatures:
        return [str(sig) for sig in signatures]

    prompt_records: List[Mapping[str, Any]] = []
    if isinstance(entry.get("prompt_record"), dict):
        prompt_records.append(entry["prompt_record"])
    if isinstance(entry.get("detection_prompt_record"), dict):
        prompt_records.append(entry["detection_prompt_record"])
    for organ_result in entry.get("organ_results") or []:
        prompt_record = organ_result.get("prompt_record")
        if isinstance(prompt_record, dict):
            prompt_records.append(prompt_record)
    return [build_prompt_signature(record) for record in prompt_records]


def _expected_prompt_signatures(
    *,
    camera_version: str,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
    cache_entry: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    if camera_version == "v0":
        return [build_prompt_signature(_direct_camera_prompt_record(
            model_id=model_id,
            prompt_version=prompt_version,
            example=example,
        ))]
    if camera_version == "v1":
        return [
            build_prompt_signature(record)
            for record in _fixed_organ_prompt_records(
                model_id=model_id,
                prompt_version=prompt_version,
                example=example,
            )
        ]

    organ_ids = list((cache_entry or {}).get("detected_organs") or [])
    expected_records = [
        _adaptive_detection_prompt_record(
            model_id=model_id,
            prompt_version=prompt_version,
            example=example,
        ),
        *_fixed_organ_prompt_records(
            model_id=model_id,
            prompt_version=prompt_version,
            example=example,
            organ_ids=organ_ids,
        ),
    ]
    return [build_prompt_signature(record) for record in expected_records]


def run_direct_camera_prediction(
    *,
    client: OpenAI,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
) -> Dict[str, Any]:
    frame_paths = get_clip_image_paths(example, uses_first_frame_only=False)
    prompt_record = _direct_camera_prompt_record(
        model_id=model_id,
        prompt_version=prompt_version,
        example=example,
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=_build_chat_messages(
            system_prompt=DIRECT_CAMERA_SYSTEM_PROMPT,
            user_text=DIRECT_CAMERA_PROMPT,
            frame_paths=frame_paths,
        ),
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw_text = response.choices[0].message.content or ""
    parsed = extract_json_object(raw_text)
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "raw": raw_text,
        "predicted_action_code": normalize_camera_action_code(parsed.get("predicted_action_code", "(not set)")),
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "prompt_record": prompt_record,
        "prompt_signatures": [build_prompt_signature(prompt_record)],
    }


def _build_size_question(organ_name: str) -> str:
    return (
        f"Does the {organ_name} become larger, become smaller, or stay the same "
        f"in the last frame compared to the first frame?\n"
        "Options:\n"
        f"- Becomes larger: The {organ_name} appears larger in the last frame than in the first frame.\n"
        f"- Becomes smaller: The {organ_name} appears smaller in the last frame than in the first frame.\n"
        f"- Stayed the same: The {organ_name} is approximately the same size in both frames.\n"
        f"- Uncertain: The {organ_name} disappears in the last frame or it is unclear whether it has changed in size.\n"
        "Reply with exactly one of: Becomes larger / Becomes smaller / Stayed the same / Uncertain."
    )


def _parse_detected_organs(raw_text: str) -> List[str]:
    cleaned = (raw_text or "").split("</think>")[-1].strip().lower()
    if not cleaned or cleaned == "none":
        return []
    pieces = re.split(r"[,;\n]+", cleaned)
    detected: List[str] = []
    normalized_name_map = {name.lower(): organ_id for organ_id, name in ORGANS}
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        for organ_id, organ_name in ORGANS:
            if piece == organ_name.lower():
                detected.append(organ_id)
                break
        else:
            normalized = piece.replace("_", " ").strip()
            if normalized in normalized_name_map:
                detected.append(normalized_name_map[normalized])
    seen = set()
    return [organ_id for organ_id in detected if not (organ_id in seen or seen.add(organ_id))]


def _vote_camera_action(mapped_actions: Iterable[str]) -> Dict[str, Any]:
    counts: Dict[str, int] = {code: 0 for code in CAMERA_ACTION_CODES}
    for code in mapped_actions:
        norm = normalize_camera_action_code(code)
        counts[norm] = counts.get(norm, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (item[1], -CAMERA_ACTION_CODES.index(item[0])),
        reverse=True,
    )
    predicted = ranked[0][0] if ranked and ranked[0][1] > 0 else "(not set)"
    return {
        "predicted_action_code": predicted,
        "vote_counts": counts,
        "winning_vote_count": counts.get(predicted, 0),
    }


def run_fixed_organs_prediction(
    *,
    client: OpenAI,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
) -> Dict[str, Any]:
    frame_paths = get_clip_image_paths(example, uses_first_frame_only=False)
    organ_results: List[Dict[str, Any]] = []
    prompt_records = _fixed_organ_prompt_records(
        model_id=model_id,
        prompt_version=prompt_version,
        example=example,
    )
    for prompt_record, (organ_id, organ_name) in zip(prompt_records, ORGANS):
        user_text = _build_size_question(organ_name)
        response = client.chat.completions.create(
            model=model_id,
            messages=_build_chat_messages(
                system_prompt=SIZE_QUESTION_SYSTEM_PROMPT,
                user_text=user_text,
                frame_paths=frame_paths,
            ),
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw_text = response.choices[0].message.content or ""
        size_answer = normalize_size_answer(raw_text)
        organ_results.append(
            {
                "organ_id": organ_id,
                "organ_name": organ_name,
                "raw": raw_text,
                "size_answer": size_answer,
                "mapped_action_code": map_size_answer_to_camera_code(size_answer),
                "prompt_record": prompt_record,
            }
        )

    vote = _vote_camera_action(result["mapped_action_code"] for result in organ_results)
    return {
        "organ_results": organ_results,
        **vote,
        "confidence": (vote["winning_vote_count"] / len(organ_results)) if organ_results else 0.0,
        "reasoning": "",
        "prompt_signatures": [build_prompt_signature(record) for record in prompt_records],
    }


def run_adaptive_organs_prediction(
    *,
    client: OpenAI,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
) -> Dict[str, Any]:
    detect_frame_paths = get_clip_image_paths(example, uses_first_frame_only=True)
    detect_text = (
        "What anatomical structures from the following list are visible in this frame: "
        "gallbladder, liver, hepatocystic triangle, omentum, cystic duct, cystic artery? "
        'Reply with a comma-separated list of visible structures, or "none" if none are visible.'
    )
    detect_prompt_record = _adaptive_detection_prompt_record(
        model_id=model_id,
        prompt_version=prompt_version,
        example=example,
    )
    detection_response = client.chat.completions.create(
        model=model_id,
        messages=_build_chat_messages(
            system_prompt=ORGAN_DETECTION_SYSTEM_PROMPT,
            user_text=detect_text,
            frame_paths=detect_frame_paths,
        ),
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    detection_raw = detection_response.choices[0].message.content or ""
    detected_organs = _parse_detected_organs(detection_raw)

    frame_paths = get_clip_image_paths(example, uses_first_frame_only=False)
    organ_prompt_records = _fixed_organ_prompt_records(
        model_id=model_id,
        prompt_version=prompt_version,
        example=example,
        organ_ids=detected_organs,
    )
    organ_results: List[Dict[str, Any]] = []
    for organ_id, prompt_record in zip(detected_organs, organ_prompt_records):
        organ_name = ORGAN_NAME_BY_ID[organ_id]
        user_text = _build_size_question(organ_name)
        response = client.chat.completions.create(
            model=model_id,
            messages=_build_chat_messages(
                system_prompt=SIZE_QUESTION_SYSTEM_PROMPT,
                user_text=user_text,
                frame_paths=frame_paths,
            ),
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw_text = response.choices[0].message.content or ""
        size_answer = normalize_size_answer(raw_text)
        organ_results.append(
            {
                "organ_id": organ_id,
                "organ_name": organ_name,
                "raw": raw_text,
                "size_answer": size_answer,
                "mapped_action_code": map_size_answer_to_camera_code(size_answer),
                "prompt_record": prompt_record,
            }
        )

    vote = _vote_camera_action(result["mapped_action_code"] for result in organ_results)
    if not organ_results:
        vote["predicted_action_code"] = "(not set)"
    return {
        "detection_raw": detection_raw,
        "detected_organs": detected_organs,
        "detection_prompt_record": detect_prompt_record,
        "organ_results": organ_results,
        **vote,
        "confidence": (vote["winning_vote_count"] / len(organ_results)) if organ_results else 0.0,
        "reasoning": "",
        "prompt_signatures": [
            build_prompt_signature(detect_prompt_record),
            *[build_prompt_signature(record) for record in organ_prompt_records],
        ],
    }


def cache_entry_is_current(
    entry: Any,
    *,
    camera_version: str,
    model_id: str,
    prompt_version: str,
    example: Mapping[str, Any],
) -> bool:
    if not isinstance(entry, dict):
        return False
    cached_signatures = sorted(_extract_prompt_signatures(entry))
    expected_signatures = sorted(
        _expected_prompt_signatures(
            camera_version=camera_version,
            model_id=model_id,
            prompt_version=prompt_version,
            example=example,
            cache_entry=entry,
        )
    )
    return bool(cached_signatures) and cached_signatures == expected_signatures


def run_inference_for_examples(
    *,
    client: OpenAI,
    model_id: str,
    prompt_version: str,
    examples: List[Dict[str, Any]],
    cache: Optional[MutableMapping[str, Any]] = None,
    overwrite: bool = False,
    camera_version: str = DEFAULT_CAMERA_VERSION,
) -> Dict[str, Any]:
    version_spec = get_camera_version_spec(camera_version)
    cache_dict: Dict[str, Any] = dict(cache or {})
    method_key = version_spec["method_key"]
    progress_desc = f"Camera {camera_version}"
    for ex in tqdm(examples, total=len(examples), desc=progress_desc):
        ex_key = ex["example_uid"]
        ex_cache = dict(cache_dict.get(ex_key, {}))
        if overwrite or not cache_entry_is_current(
            ex_cache.get(method_key),
            camera_version=camera_version,
            model_id=model_id,
            prompt_version=prompt_version,
            example=ex,
        ):
            if camera_version == "v0":
                result = run_direct_camera_prediction(
                    client=client,
                    model_id=model_id,
                    prompt_version=prompt_version,
                    example=ex,
                )
            elif camera_version == "v1":
                result = run_fixed_organs_prediction(
                    client=client,
                    model_id=model_id,
                    prompt_version=prompt_version,
                    example=ex,
                )
            else:
                result = run_adaptive_organs_prediction(
                    client=client,
                    model_id=model_id,
                    prompt_version=prompt_version,
                    example=ex,
                )
            result["prompt_version"] = prompt_version
            result["camera_version"] = camera_version
            ex_cache[method_key] = result
        cache_dict[ex_key] = ex_cache
    return cache_dict


def build_prediction_rows(
    *,
    examples: Iterable[Mapping[str, Any]],
    cache: Mapping[str, Any],
    camera_version: str = DEFAULT_CAMERA_VERSION,
) -> List[Dict[str, Any]]:
    version_spec = get_camera_version_spec(camera_version)
    method_key = version_spec["method_key"]
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        prediction = cache.get(ex["example_uid"], {}).get(method_key, {})
        pred_action = normalize_camera_action_code(prediction.get("predicted_action_code", "(not set)"))
        gt_action = normalize_camera_action_code(ex.get("gt_action_code", "(not set)"))
        rows.append(
            {
                "example_uid": ex["example_uid"],
                "video_id": ex["video_id"],
                "clip_uid": ex["clip_uid"],
                "clip_type": ex["clip_type"],
                "criterion": ex["criterion"],
                "gt_action_code": gt_action,
                "predicted_action_code": pred_action,
                "correct": int(gt_action == pred_action),
                "confidence": prediction.get("confidence"),
                "prompt_version": prediction.get("prompt_version"),
                "camera_version": prediction.get("camera_version"),
                "vote_counts": prediction.get("vote_counts", {}),
                "detected_organs": prediction.get("detected_organs", []),
                "organ_results": prediction.get("organ_results", []),
            }
        )
    return rows


def summarize_predictions(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    evaluated = [row for row in rows if row.get("gt_action_code") and row.get("predicted_action_code")]
    correct = sum(int(row["correct"]) for row in evaluated)

    by_clip_type: Dict[str, Dict[str, Any]] = {}
    by_class: Dict[str, Dict[str, Any]] = {}
    by_clip_type_and_class: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for clip_type in sorted({row.get("clip_type") for row in evaluated if row.get("clip_type")}):
        split_rows = [row for row in evaluated if row.get("clip_type") == clip_type]
        split_correct = sum(int(row["correct"]) for row in split_rows)
        by_clip_type[clip_type] = {
            "n_evaluated": len(split_rows),
            "n_correct": split_correct,
            "accuracy": (split_correct / len(split_rows)) if split_rows else None,
        }

    for action_code in CAMERA_ACTION_CODES:
        class_rows = [row for row in evaluated if row.get("gt_action_code") == action_code]
        class_correct = sum(int(row["correct"]) for row in class_rows)
        by_class[action_code] = {
            "n_evaluated": len(class_rows),
            "n_correct": class_correct,
            "accuracy": (class_correct / len(class_rows)) if class_rows else None,
        }
        by_clip_type_and_class[action_code] = {}
        for clip_type in sorted({row.get("clip_type") for row in class_rows if row.get("clip_type")}):
            split_rows = [row for row in class_rows if row.get("clip_type") == clip_type]
            split_correct = sum(int(row["correct"]) for row in split_rows)
            by_clip_type_and_class[action_code][clip_type] = {
                "n_evaluated": len(split_rows),
                "n_correct": split_correct,
                "accuracy": (split_correct / len(split_rows)) if split_rows else None,
            }

    return {
        "n_rows": len(rows),
        "n_evaluated": len(evaluated),
        "n_correct": correct,
        "accuracy": (correct / len(evaluated)) if evaluated else None,
        "by_clip_type": by_clip_type,
        "by_class": by_class,
        "by_clip_type_and_class": by_clip_type_and_class,
    }
