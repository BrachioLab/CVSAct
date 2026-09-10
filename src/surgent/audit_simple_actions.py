from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


AUDIT_V11_SIMPLE_TARGET_COUNT = 4
AUDIT_V11_SIMPLE_SLOT_ORDER = (
    "camera",
    "left_instrument",
    "right_instrument",
    "other",
)

_SLOT_ALIASES = {
    "right": "right_instrument",
    "right_instrument": "right_instrument",
    "left": "left_instrument",
    "left_instrument": "left_instrument",
    "camera": "camera",
    "other": "other",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hf_taxonomy_path() -> Path:
    return _repo_root() / "hf_repos" / "cvs-act" / "taxonomy" / "action_taxonomy.json"


@lru_cache(maxsize=1)
def load_audit_v11_simple_options() -> Dict[str, Any]:
    """Load the exact audit_v11 simple-action option inventory."""
    hf_taxonomy_path = _hf_taxonomy_path()
    if hf_taxonomy_path.exists():
        try:
            payload = json.loads(hf_taxonomy_path.read_text())
            actors = payload.get("actors", {}) if isinstance(payload, dict) else {}
            left_codes = (
                actors.get("left", {}).get("components", {}).get("retraction_direction_code", [])
            )
            right = actors.get("right", {}).get("components", {})
            camera_codes = actors.get("camera", {}).get("components", {}).get("action_code", [])
            return {
                "source": str(hf_taxonomy_path),
                "interface_options": {
                    "retraction_direction_options": [
                        "unsure",
                        "left",
                        "right",
                        "upward",
                        "downward",
                        "not_retracted",
                        "unclear",
                    ],
                    "camera_action_codes": camera_codes,
                    "tool_type": right.get("tool_type", []),
                    "action_code": sorted(
                        dict.fromkeys(
                            list(right.get("action_code", []))
                            + list(camera_codes)
                            + list(left_codes)
                        )
                    ),
                    "target_structure": right.get("target_structure", []),
                    "target_context": right.get("target_context", []),
                },
                "left_retraction_direction_code_options": left_codes,
                "right_triplet_options": [],
                "camera_action_code_options": camera_codes,
                "other_action_code_options": [],
            }
        except Exception:
            pass

    annotation_root = (
        _repo_root()
        / "data"
        / "processed"
        / "CVS_Challenge_SAGES_v1"
        / "cvs_act_annotations"
        / "v1"
    )
    audit_v11_dir = annotation_root / "audit_v11"
    try:
        from cvs_act.action_segment_eval import (
            build_simple_options,
            convert_record_to_simple_actions,
            load_audit_records,
        )

        records = load_audit_records(audit_v11_dir)
        simple_records = [convert_record_to_simple_actions(record) for record in records]
        return build_simple_options(simple_records, annotation_root)
    except Exception:
        return {
            "source": "fallback",
            "interface_options": {
                "retraction_direction_options": [
                    "unsure",
                    "left",
                    "right",
                    "upward",
                    "downward",
                    "not_retracted",
                    "unclear",
                ],
                "camera_action_codes": [
                    "CAMERA_ZOOM_IN",
                    "CAMERA_ZOOM_OUT",
                    "CAMERA_REPOSITION",
                    "CAMERA_UNCERTAIN",
                ],
                "tool_type": ["Hook", "Grasper", "camera", "Maryland", "(not set)", "Irrigator", "Scissors", "clipper"],
                "action_code": [
                    "(not set)",
                    "CAMERA_REPOSITION",
                    "CAMERA_ZOOM_IN",
                    "CAMERA_ZOOM_OUT",
                    "CAMERA_UNCERTAIN",
                    "CAMERA_NO_CHANGE",
                    "DISSECT",
                    "COUNTERTRACTION_ASSIST",
                    "IRRIGATOR_ASPIRATE",
                    "CLIP",
                    "ICG_SWITCH",
                    "KEEP_RETRACT_LATERAL",
                    "KEEP_RETRACT_UPWARD",
                    "KEEP_RETRACT_MEDIAL",
                    "RETRACT_LATERAL",
                    "RETRACT_MEDIAL",
                    "RETRACT_UPWARD",
                    "RETRACT_LATERAL_TO_MEDIAL",
                    "RETRACT_MEDIAL_TO_LATERAL",
                    "RETRACT_LATERAL_TO_UPWARD",
                    "RETRACT_UPWARD_TO_LATERAL",
                ],
                "target_structure": [
                    "HepatocysticTriangle",
                    "GallbladderNeck_Infundibulum",
                    "CysticPlate",
                    "CysticArtery",
                    "CysticDuct",
                    "TwoTubularStructures",
                    "Gallbladder",
                    "BloodOrFluid",
                    "(not set)",
                ],
                "target_context": ["(not set)"],
            },
            "left_retraction_direction_code_options": [
                "KEEP_RETRACT_LATERAL",
                "KEEP_RETRACT_MEDIAL",
                "KEEP_RETRACT_UPWARD",
                "RETRACT_LATERAL",
                "RETRACT_LATERAL_TO_MEDIAL",
                "RETRACT_LATERAL_TO_UPWARD",
                "RETRACT_MEDIAL",
                "RETRACT_MEDIAL_TO_LATERAL",
                "RETRACT_UPWARD_TO_LATERAL",
            ],
            "right_triplet_options": [],
            "camera_action_code_options": [
                "CAMERA_ZOOM_IN",
                "CAMERA_ZOOM_OUT",
                "CAMERA_REPOSITION",
                "CAMERA_UNCERTAIN",
                "CAMERA_NO_CHANGE",
            ],
            "other_action_code_options": ["ICG_SWITCH"],
        }


def augment_taxonomy_catalog_for_audit_v11_simple(catalog: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extend taxonomy_catalog with audit_v11 simple action options."""
    out: Dict[str, Any] = deepcopy(catalog) if isinstance(catalog, dict) else {}
    options = out.setdefault("options", {})
    for field in ("actor_role", "tool_type", "action_code", "target_structure", "target_context", "intention"):
        vals = options.get(field)
        if not isinstance(vals, list):
            options[field] = []
    simple = load_audit_v11_simple_options()
    interface = simple.get("interface_options", {}) if isinstance(simple, dict) else {}

    _extend_unique(options["actor_role"], ["left_instrument", "right_instrument", "camera", "other"])
    _extend_unique(options["tool_type"], interface.get("tool_type", []))
    _extend_unique(options["action_code"], interface.get("action_code", []))
    _extend_unique(options["target_structure"], interface.get("target_structure", []))
    _extend_unique(options["target_context"], interface.get("target_context", []))
    return out


def build_audit_v11_simple_prompt_block() -> str:
    """Human-readable actor-wise action schema guidance for prompts."""
    simple = load_audit_v11_simple_options()
    left_codes = simple.get("left_retraction_direction_code_options", [])
    right_triplets = simple.get("right_triplet_options", [])
    camera_codes = simple.get("camera_action_code_options", [])
    other_codes = simple.get("other_action_code_options", [])
    target_context = (
        simple.get("interface_options", {}).get("target_context", [])
        if isinstance(simple.get("interface_options"), dict)
        else []
    )
    target_context = [
        "between cystic artery and cystic plate"
        if value == "between cystic artery and liver bed"
        else value
        for value in target_context
    ]
    target_context = list(dict.fromkeys(target_context))
    lines = [
        "AUDIT_V11 SIMPLE ACTION INTERFACE:",
        f"- Return exactly {AUDIT_V11_SIMPLE_TARGET_COUNT} action slots, one per actor_role: camera, left_instrument, right_instrument, other.",
        "- This is NOT retrospective clip labeling. Predict what should happen next from the visible prefix of the video and the current CVS state.",
        '- Every slot must include an "evidence" field. If a slot is genuinely uncertain, explicitly say "uncertain" in evidence.',
        "- right_instrument slot: choose exactly one concrete active-tool recommendation. This is the only slot that should usually use target_structure and optional target_context fields.",
        f"- Allowed right-instrument (tool_type, action_code, target_structure) triplets: {json.dumps(right_triplets, ensure_ascii=False)}",
        "- left_instrument slot: tool_type should normally be Grasper and action_code must be one of the allowed left retraction direction codes. Do not add target_structure. Add target_context only if it is genuinely useful for specifying the retraction plane/location.",
        f"- Allowed left retraction direction codes: {json.dumps(left_codes, ensure_ascii=False)}",
        f"- If a left_instrument recommendation uses target_context_1 or target_context_2, each must be one of: {json.dumps(target_context, ensure_ascii=False)}",
        '- camera slot: tool_type MUST be "camera". Use CAMERA_NO_CHANGE when no camera adjustment should be made. Use CAMERA_UNCERTAIN when camera recommendation is actually uncertain. Camera should not use target_structure or target_context.',
        f"- Allowed camera action codes: {json.dumps(camera_codes, ensure_ascii=False)}",
        "- other slot: reserve for extra non-left/right/camera actions such as ICG. If no justified other action is needed, set action_code to (not set) and say so in evidence. Other should usually omit tool_type, target_structure, and target_context.",
        f"- Allowed other action codes: {json.dumps(other_codes, ensure_ascii=False)}",
        f"- If a right_instrument recommendation uses target_context_1 or target_context_2, each must be one of: {json.dumps(target_context, ensure_ascii=False)}",
        '- Do not output an "intention" field in this simplified interface.',
    ]
    return "\n".join(lines)


def normalize_actor_role(value: Any) -> Optional[str]:
    return _SLOT_ALIASES.get(str(value or "").strip().lower())


def normalize_actor_slot_actions(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep at most one action per actor slot and pad missing slots."""
    kept: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in actions:
        if not isinstance(raw, dict):
            continue
        actor_role = normalize_actor_role(raw.get("actor_role"))
        if actor_role is None or actor_role in seen:
            continue
        item = dict(raw)
        item["actor_role"] = actor_role
        item.setdefault("evidence", "")
        kept.append(item)
        seen.add(actor_role)
    for actor_role in AUDIT_V11_SIMPLE_SLOT_ORDER:
        if actor_role in seen:
            continue
        kept.append(_default_slot_action(actor_role))
    for idx, item in enumerate(kept):
        item["rank"] = idx + 1
    return kept[:AUDIT_V11_SIMPLE_TARGET_COUNT]


def _default_slot_action(actor_role: str) -> Dict[str, Any]:
    if actor_role == "camera":
        return {
            "rank": 0,
            "actor_role": "camera",
            "tool_type": "camera",
            "action_code": "CAMERA_UNCERTAIN",
            "target_structure": "(not set)",
            "target_context_1": "(not set)",
            "target_context_2": "(not set)",
            "intention": "(not set)",
            "confidence": 0.0,
            "evidence": "uncertain",
        }
    if actor_role == "left_instrument":
        return {
            "rank": 0,
            "actor_role": "left_instrument",
            "tool_type": "Grasper",
            "action_code": "(not set)",
            "target_structure": "(not set)",
            "target_context_1": "(not set)",
            "target_context_2": "(not set)",
            "intention": "(not set)",
            "confidence": 0.0,
            "evidence": "uncertain",
        }
    if actor_role == "right_instrument":
        return {
            "rank": 0,
            "actor_role": "right_instrument",
            "tool_type": "(not set)",
            "action_code": "(not set)",
            "target_structure": "(not set)",
            "target_context_1": "(not set)",
            "target_context_2": "(not set)",
            "intention": "(not set)",
            "confidence": 0.0,
            "evidence": "uncertain",
        }
    return {
        "rank": 0,
        "actor_role": "other",
        "tool_type": "(not set)",
        "action_code": "(not set)",
        "target_structure": "(not set)",
        "target_context_1": "(not set)",
        "target_context_2": "(not set)",
        "intention": "(not set)",
        "confidence": 0.0,
        "evidence": "no additional other action",
    }


def _extend_unique(target: List[Any], values: Sequence[Any]) -> None:
    existing = {str(v) for v in target}
    for value in values:
        text = str(value)
        if text in existing:
            continue
        target.append(value)
        existing.add(text)
