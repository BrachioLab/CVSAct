#!/usr/bin/env python3
"""Run Surgent as a sequential online agent with persistent HM3 memory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_community.cache import InMemoryCache, SQLiteCache  # noqa: E402
from langchain_core.globals import set_llm_cache  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from surgent.graph import ALL_TOOL_NAMES, SurgentContext, build_agent_graph, run_cvs_agent_with_memory  # noqa: E402
from surgent.memory_hm3 import FrameRef, HM3Memory, VideoMetadata  # noqa: E402
from surgent.schemas import DEFAULT_CVS_CRITERIA  # noqa: E402

TAXONOMY_EMPTY = "(not set)"
TAXONOMY_FIELDS = (
    "actor_role",
    "tool_type",
    "action_code",
    "target_structure",
    "target_context",
    "intention",
)


def _default_taxonomy_path() -> Path:
    hf_repo_taxonomy = REPO_ROOT / "hf_repos" / "cvs-act" / "taxonomy" / "action_taxonomy.json"
    if hf_repo_taxonomy.exists():
        return hf_repo_taxonomy
    fallback_base_dir = (
        REPO_ROOT
        / "data"
        / "processed"
        / "CVS_Challenge_SAGES_v1"
        / "cvs_act_annotations"
        / "v1"
    )
    base_dirs = [fallback_base_dir]
    candidates: List[Tuple[int, int, float, Path]] = []
    for priority, base_dir in enumerate(base_dirs):
        for path in base_dir.glob("taxonomy_v*.json"):
            stem = path.stem
            m = re.search(r"taxonomy_v(\d+)$", stem, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                version = int(m.group(1))
            except Exception:
                continue
            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            # Prefer highest version first, then newest mtime, then preferred location.
            candidates.append((version, -priority, mtime, path))
    best_path: Optional[Path] = None
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_path = candidates[0][3]
    if best_path is not None:
        return best_path
    return fallback_base_dir / "taxonomy_v3.json"


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


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
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
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _load_taxonomy_catalog(path: Path) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {
        "path": str(path),
        "version": "unknown",
        "options": {field: [] for field in TAXONOMY_FIELDS},
        "aliases": {field: {} for field in TAXONOMY_FIELDS},
    }
    if not path.exists():
        return catalog
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return catalog
    if not isinstance(payload, dict):
        return catalog
    if isinstance(payload.get("actors"), dict):
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
        catalog["intention_descs"] = {}
        catalog["recommendation_descs"] = {field: {} for field in TAXONOMY_FIELDS}
        catalog["field_meta"] = {}
        catalog["options"] = {
            "actor_role": ["camera", "left_instrument", "right_instrument", "other"],
            "tool_type": list(right_components.get("tool_type", [])),
            "action_code": list(
                dict.fromkeys(
                    list(right_components.get("action_code", []))
                    + list(camera_components.get("action_code", []))
                    + list(left_components.get("retraction_direction_code", []))
                )
            ),
            "target_structure": list(right_components.get("target_structure", [])),
            "target_context": list(right_components.get("target_context", [])),
            "intention": [],
        }
        if "camera" not in {str(x).lower() for x in catalog["options"]["tool_type"]}:
            catalog["options"]["tool_type"].append("camera")
        catalog["aliases"] = {field: {} for field in TAXONOMY_FIELDS}
        return catalog
    meta = payload.get("_meta", {})
    if isinstance(meta, dict):
        catalog["version"] = str(meta.get("version") or "unknown")

    fields = payload.get("fields", {})
    options: Dict[str, List[str]] = {field: [] for field in TAXONOMY_FIELDS}
    options_lower: Dict[str, set[str]] = {field: set() for field in TAXONOMY_FIELDS}

    def _add(field_name: str, value: Any) -> None:
        field = str(field_name or "").strip()
        if field not in options:
            return
        text = str(value or "").strip()
        if not text:
            return
        lowered = text.lower()
        if lowered in options_lower[field]:
            return
        options_lower[field].add(lowered)
        options[field].append(text)

    if isinstance(fields, dict):
        for field, entries in fields.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and "value" in entry:
                    _add(str(field), entry.get("value"))
    intentions = payload.get("intentions", [])
    if isinstance(intentions, list):
        for entry in intentions:
            if isinstance(entry, dict) and "value" in entry:
                _add("intention", entry.get("value"))
    intention_descs: Dict[str, str] = {}
    if isinstance(intentions, list):
        for entry in intentions:
            if isinstance(entry, dict):
                val = str(entry.get("value", ""))
                desc = str(entry.get("description", ""))
                if val and desc:
                    intention_descs[val] = desc
    catalog["intention_descs"] = intention_descs
    for field in TAXONOMY_FIELDS:
        if TAXONOMY_EMPTY.lower() not in options_lower[field]:
            options[field].append(TAXONOMY_EMPTY)
            options_lower[field].add(TAXONOMY_EMPTY.lower())
    aliases: Dict[str, Dict[str, str]] = {field: {} for field in TAXONOMY_FIELDS}

    def _normalize_field_name(raw: Any) -> Optional[str]:
        text = str(raw or "").strip().lower()
        if not text:
            return None
        direct = {
            "actor_role": "actor_role",
            "tool_type": "tool_type",
            "action_code": "action_code",
            "target_structure": "target_structure",
            "target_context": "target_context",
            "intention": "intention",
        }
        if text in direct:
            return direct[text]
        if text.endswith("_remap"):
            text = text[: -len("_remap")]
            if text in direct:
                return direct[text]
        return None

    def _add_alias(field_name: Any, source_value: Any, target_value: Any) -> None:
        field = _normalize_field_name(field_name)
        if field is None:
            return
        src = str(source_value or "").strip()
        dst = str(target_value or "").strip()
        if not src or not dst:
            return
        aliases[field][src.lower()] = dst

    merges = payload.get("merges", {})
    if isinstance(merges, dict):
        for field_name, entries in merges.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                to_map = entry.get("to_map")
                if isinstance(to_map, dict):
                    for src, dst in to_map.items():
                        _add_alias(field_name, src, dst)
                to_value = entry.get("to")
                from_values = entry.get("from")
                if isinstance(to_value, str) and isinstance(from_values, list):
                    for src in from_values:
                        _add_alias(field_name, src, to_value)

    for key, block in payload.items():
        if not isinstance(key, str) or not key.endswith("_rules") or not isinstance(block, dict):
            continue
        for rule_name, rule_value in block.items():
            if not isinstance(rule_name, str) or not rule_name.endswith("_remap") or not isinstance(rule_value, dict):
                continue
            field_name = _normalize_field_name(rule_name)
            if field_name is None:
                continue
            for src, dst in rule_value.items():
                _add_alias(field_name, src, dst)

    version_text = str(catalog.get("version") or "")
    version_match = re.search(r"v(\d+)", version_text, flags=re.IGNORECASE)
    if version_match:
        norm_map_path = path.with_name(f"normalization_map_v{version_match.group(1)}.json")
        if norm_map_path.exists():
            try:
                norm_payload = json.loads(norm_map_path.read_text(encoding="utf-8"))
            except Exception:
                norm_payload = {}
            norm_map = norm_payload.get("normalization_map", {})
            if isinstance(norm_map, dict):
                for field_name, field_map in norm_map.items():
                    if not isinstance(field_map, dict):
                        continue
                    for src, dst in field_map.items():
                        _add_alias(field_name, src, dst)

    # Build per-value recommendation descriptions (v10+)
    rec_descs: Dict[str, Dict[str, str]] = {field: {} for field in TAXONOMY_FIELDS}
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
    field_meta: Dict[str, str] = {}
    for key in ("field_meta", "field_guidelines"):
        fm_block = payload.get(key, {})
        if isinstance(fm_block, dict) and fm_block:
            for field in TAXONOMY_FIELDS:
                entry = fm_block.get(field, {})
                if isinstance(entry, dict):
                    desc = str(entry.get("recommendation_description", "")).strip()
                    if desc and field not in field_meta:
                        field_meta[field] = desc
            if field_meta:
                break
    catalog["field_meta"] = field_meta

    catalog["options"] = options
    catalog["aliases"] = aliases
    return catalog


def _sanitize_taxonomy_value(
    value: Any,
    allowed_values: Sequence[str],
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    text = str(value or "").strip()
    if not text:
        return TAXONOMY_EMPTY
    invalid_tokens = {"none", "null", "n/a", "na", "unspecified", "not set", "(not set)"}
    if text.lower() in invalid_tokens:
        return TAXONOMY_EMPTY
    if isinstance(aliases, Mapping):
        mapped = aliases.get(text.lower())
        if mapped:
            text = str(mapped).strip()
            if not text:
                return TAXONOMY_EMPTY
    if not allowed_values:
        return text
    by_lower = {str(v).strip().lower(): str(v) for v in allowed_values if str(v).strip()}
    return by_lower.get(text.lower(), TAXONOMY_EMPTY)


def _as_float_01(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = default
    return max(0.0, min(1.0, out))


def _default_predicted_taxonomy_action(
    *,
    criterion: Optional[str],
    taxonomy_version: str,
    taxonomy_path: Path,
) -> Dict[str, Any]:
    return {
        "criterion": str(criterion or ""),
        "rank": 1,
        "actor_role": TAXONOMY_EMPTY,
        "tool_type": TAXONOMY_EMPTY,
        "action_code": TAXONOMY_EMPTY,
        "target_structure": TAXONOMY_EMPTY,
        "target_context_1": TAXONOMY_EMPTY,
        "target_context_2": TAXONOMY_EMPTY,
        "intention": TAXONOMY_EMPTY,
        "generated_sentence": "",
        "confidence": 0.0,
        "taxonomy_version": str(taxonomy_version or "unknown"),
        "taxonomy_path": str(taxonomy_path),
    }


def _predict_taxonomy_action(
    *,
    model: Any,
    hm3: HM3Memory,
    criterion: Optional[str],
    decision: Mapping[str, Any],
    taxonomy_catalog: Mapping[str, Any],
    taxonomy_path: Path,
) -> Dict[str, Any]:
    taxonomy_version = str(taxonomy_catalog.get("version") or "unknown")
    out = _default_predicted_taxonomy_action(
        criterion=criterion,
        taxonomy_version=taxonomy_version,
        taxonomy_path=taxonomy_path,
    )
    options = taxonomy_catalog.get("options", {})
    if not isinstance(options, dict):
        return out
    aliases = taxonomy_catalog.get("aliases", {})
    if not isinstance(aliases, dict):
        aliases = {}

    actor_values = [str(x) for x in options.get("actor_role", [])]
    tool_values = [str(x) for x in options.get("tool_type", [])]
    code_values = [str(x) for x in options.get("action_code", [])]
    target_values = [str(x) for x in options.get("target_structure", [])]
    context_values = [str(x) for x in options.get("target_context", [])]
    intention_values = [str(x) for x in options.get("intention", [])]
    actor_aliases = aliases.get("actor_role", {}) if isinstance(aliases.get("actor_role"), dict) else {}
    tool_aliases = aliases.get("tool_type", {}) if isinstance(aliases.get("tool_type"), dict) else {}
    code_aliases = aliases.get("action_code", {}) if isinstance(aliases.get("action_code"), dict) else {}
    target_aliases = aliases.get("target_structure", {}) if isinstance(aliases.get("target_structure"), dict) else {}
    context_aliases = aliases.get("target_context", {}) if isinstance(aliases.get("target_context"), dict) else {}
    intention_aliases = aliases.get("intention", {}) if isinstance(aliases.get("intention"), dict) else {}
    if not code_values:
        return out

    decision_obj = dict(decision) if isinstance(decision, Mapping) else {}
    action_args = decision_obj.get("action_args", {})
    if not isinstance(action_args, dict):
        action_args = {}

    memory_snapshot = hm3.snapshot(tail=4, max_output_chars=240)
    user_prompt = (
        "Task: predict ONE ranked next surgical action using the provided taxonomy.\n"
        "This recommendation is for surgeon-facing action planning (not tool-routing calls like scene_snapper).\n"
        f"Active criterion: {criterion or ''}\n"
        f"Controller decision: {json.dumps(decision_obj, ensure_ascii=False)}\n"
        f"Memory snapshot: {json.dumps(memory_snapshot, ensure_ascii=False)}\n\n"
        "Allowed options:\n"
        f"- actor_role: {json.dumps(actor_values, ensure_ascii=False)}\n"
        f"- tool_type: {json.dumps(tool_values, ensure_ascii=False)}\n"
        f"- action_code: {json.dumps(code_values, ensure_ascii=False)}\n"
        f"- target_structure: {json.dumps(target_values, ensure_ascii=False)}\n"
        f"- target_context: {json.dumps(context_values, ensure_ascii=False)}\n"
        f"- intention: {json.dumps(intention_values, ensure_ascii=False)}\n\n"
        "Return STRICT JSON only with keys exactly:\n"
        "{\n"
        '  "actor_role": "<value from actor_role>",\n'
        '  "tool_type": "<value from tool_type>",\n'
        '  "action_code": "<value from action_code>",\n'
        '  "target_structure": "<value from target_structure>",\n'
        '  "target_context_1": "<value from target_context or (not set)>",\n'
        '  "target_context_2": "<value from target_context or (not set)>",\n'
        '  "intention": "<value from intention>",\n'
        '  "generated_sentence": "<short surgeon-facing sentence>",\n'
        '  "confidence": <float 0..1>\n'
        "}\n"
        "Do not include markdown. Do not include extra keys."
    )
    system_prompt = (
        "You map evidence to a controlled action taxonomy. "
        "Always choose values from allowed options. "
        f'Use "{TAXONOMY_EMPTY}" when unknown.'
    )
    content: Any = ""
    parsed: Dict[str, Any] = {}
    if model is not None:
        try:
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = getattr(response, "content", "")
            parsed = content if isinstance(content, dict) else _extract_json_object(str(content))
        except Exception:
            parsed = {}
    if not parsed:
        parsed = {}

    context_raw = parsed.get("target_context", None)
    context_values_raw: List[Any] = []
    if isinstance(context_raw, list):
        context_values_raw = context_raw
    elif context_raw is not None:
        context_values_raw = [context_raw]
    context_values_raw.extend(
        [
            parsed.get("target_context_1"),
            parsed.get("target_context_2"),
        ]
    )
    context_clean = [str(x).strip() for x in context_values_raw if str(x or "").strip()]

    sentence = str(
        parsed.get("generated_sentence")
        or parsed.get("one_sentence")
        or parsed.get("sentence")
        or action_args.get("generated_sentence")
        or action_args.get("one_sentence")
        or ""
    ).strip()
    actor = _sanitize_taxonomy_value(
        parsed.get("actor_role", action_args.get("actor_role")),
        actor_values,
        aliases=actor_aliases,
    )
    tool = _sanitize_taxonomy_value(
        parsed.get("tool_type", action_args.get("tool_type")),
        tool_values,
        aliases=tool_aliases,
    )
    action_code = _sanitize_taxonomy_value(
        parsed.get("action_code", action_args.get("action_code")),
        code_values,
        aliases=code_aliases,
    )
    target = _sanitize_taxonomy_value(
        parsed.get("target_structure", action_args.get("target_structure")),
        target_values,
        aliases=target_aliases,
    )
    context_1 = _sanitize_taxonomy_value(
        context_clean[0] if context_clean else action_args.get("target_context_1"),
        context_values,
        aliases=context_aliases,
    )
    context_2 = _sanitize_taxonomy_value(
        context_clean[1] if len(context_clean) > 1 else action_args.get("target_context_2"),
        context_values,
        aliases=context_aliases,
    )
    intention = _sanitize_taxonomy_value(
        parsed.get("intention", action_args.get("intention")),
        intention_values,
        aliases=intention_aliases,
    )
    confidence = _as_float_01(
        parsed.get("confidence", action_args.get("confidence", decision_obj.get("confidence", 0.0))),
        default=0.0,
    )
    if not sentence and action_code != TAXONOMY_EMPTY:
        sentence = f"Recommend {action_code} to progress {criterion or 'CVS'} evidence."

    out.update(
        {
            "actor_role": actor,
            "tool_type": tool,
            "action_code": action_code,
            "target_structure": target,
            "target_context_1": context_1,
            "target_context_2": context_2,
            "intention": intention,
            "generated_sentence": sentence,
            "confidence": confidence,
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sequential online Surgent runner: at each step t, the agent can only access "
            "frames <= t (or <= lookback window), while HM3 memory persists across steps."
        )
    )
    p.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing ordered frame images for one video.",
    )
    p.add_argument(
        "--criterion",
        default=None,
        help=(
            "Optional focus criterion for action targeting (e.g., C1). "
            "Predictions are still emitted for C1/C2/C3 at each step."
        ),
    )
    p.add_argument(
        "--criteria",
        default=None,
        help=(
            "Optional focus criteria for action targeting, e.g. C1,C2. "
            "Predictions are still emitted for C1/C2/C3 at each step."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Global default model id for scene/clip/controller (can override per role with env).",
    )
    p.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "Subsample selected evaluation frames by N "
            "(default: 1; with keyframe mode this means every selected keyframe)."
        ),
    )
    p.add_argument(
        "--start-frame-idx",
        type=int,
        default=0,
        help="First frame index to evaluate.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of sequential steps to run after stepping/subsampling.",
    )
    p.add_argument(
        "--max-source-frame",
        type=int,
        default=None,
        help=(
            "Stop after processing frames up to this source frame number "
            "(e.g. --max-source-frame 150 stops after source frame 150). "
            "Source frame numbers are extracted from frame filenames."
        ),
    )
    p.add_argument(
        "--lookback-frames",
        type=int,
        default=None,
        help="If set, visibility window is [t-lookback+1, t] instead of [0, t].",
    )
    p.add_argument(
        "--sampling-mode",
        choices=["keyframes", "stride"],
        default="keyframes",
        help=(
            "keyframes: evaluate only fixed-pattern keyframes "
            "(tools can still inspect prior non-key frames); "
            "stride: evaluate every --frame-step over all frames."
        ),
    )
    p.add_argument(
        "--keyframe-interval",
        type=int,
        default=150,
        help=(
            "When --sampling-mode=keyframes, select keyframes on this source-frame interval "
            "(default: 150)."
        ),
    )
    p.add_argument(
        "--frame-csv",
        default=None,
        help=(
            "Deprecated and ignored. Keyframes are selected by --keyframe-interval."
        ),
    )
    p.add_argument(
        "--cache",
        choices=["none", "memory", "sqlite"],
        default="none",
        help="LLM cache backend.",
    )
    p.add_argument(
        "--verbose",
        type=int,
        nargs="?",
        const=2,
        choices=[0, 1, 2, 3],
        default=0,
        help="Requested verbosity level. Runtime keeps console at readable level 1 while saving traces at level 3.",
    )
    p.add_argument(
        "--out-jsonl",
        default=None,
        help=(
            "Output JSONL path. Default: "
            "outputs/surgent_sequential_agent/<video_id>__multi__C1-C2-C3.jsonl"
        ),
    )
    p.add_argument(
        "--memory-tail",
        type=int,
        default=0,
        help="If >0, include HM3 snapshot tail in each output row.",
    )
    p.add_argument(
        "--trace-dir",
        default="outputs/traces_sequential",
        help="Base directory for detailed per-run traces (always enabled).",
    )
    p.add_argument(
        "--taxonomy-json",
        default=str(_default_taxonomy_path()),
        help=(
            "Path to taxonomy JSON for structured recommended-action prediction "
            "(default: latest taxonomy_v*.json)."
        ),
    )
    p.add_argument(
        "--tools",
        default=",".join(sorted(ALL_TOOL_NAMES)),
        help=(
            "Comma-separated list of allowed tools "
            f"(default: {','.join(sorted(ALL_TOOL_NAMES))})."
        ),
    )
    p.add_argument(
        "--preferred-tools",
        default="scene_cvs_analyzer,action_rec",
        help=(
            "Comma-separated list of tools to auto-call before the controller decides, "
            "in order. Set to empty string to disable. "
            "(default: scene_cvs_analyzer,action_rec)."
        ),
    )
    p.add_argument(
        "--scene-preset",
        choices=["direct", "cot", "subrubric"],
        default="direct",
        help=(
            "Prompt preset for scene_cvs_analyzer: "
            "direct (predict CVS + action, no rationale), "
            "cot (with chain-of-thought rationale), "
            "subrubric (self-rubric checklist then predict). "
            "(default: direct)."
        ),
    )
    p.add_argument(
        "--analyze-actions",
        action="store_true",
        default=False,
        help=(
            "(Legacy, kept for backward compat) "
            "Add action analysis context to the controller prompt."
        ),
    )
    p.add_argument(
        "--recommend-actions",
        action="store_true",
        default=True,
        help=(
            "Enable recommend-actions mode for scene_cvs_analyzer (default: on). "
            "Use --no-recommend-actions to disable."
        ),
    )
    p.add_argument(
        "--no-recommend-actions",
        dest="recommend_actions",
        action="store_false",
        help="Disable recommend-actions mode for scene_cvs_analyzer.",
    )
    p.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help=(
            "Legacy fixed-k compatibility flag. Action recommendation now uses four actor slots "
            "(camera, left_instrument, right_instrument, other) regardless of the numeric value. "
            "Affects scene_cvs_analyzer output."
        ),
    )
    p.add_argument(
        "--no-rec-descs",
        action="store_true",
        help="Strip recommendation_description from taxonomy block (use terse value-only lists).",
    )
    p.add_argument(
        "--include-field-meta",
        action="store_true",
        help="Include per-field recommendation_description guidance from taxonomy (default: off).",
    )
    p.add_argument(
        "--one-per-actor",
        action="store_true",
        default=False,
        help="Legacy flag; action recommendation now always uses one slot per actor_role.",
    )
    p.add_argument(
        "--cvs-context-mode",
        choices=["latest", "current", "full", "video_summary"],
        default="latest",
        help=(
            "CVS context passed to action_rec: "
            "'latest' (default) = latest scene_cvs_analyzer output (scores+checklist+summary); "
            "'current' = current frame scores only (no checklist/summary, like baseline); "
            "'full' = all frames' CVS history; "
            "'video_summary' = video-level running CVS summary (max-accumulated)."
        ),
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
        "--finalize-tools",
        type=str,
        default=None,
        help=(
            "Comma-separated tools to auto-call at END of each round "
            "(after controller ANSWER, before final output). "
            "e.g. --finalize-tools action_rec"
        ),
    )
    return p.parse_args()


def _frame_sort_key(path: Path) -> Tuple[int, Any, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), stem)
    match = re.search(r"(\d+)", stem)
    if match:
        return (1, int(match.group(1)), stem)
    return (2, stem, stem)


def _collect_image_paths(image_dir: Path) -> List[str]:
    if not image_dir.exists():
        raise SystemExit(f"image_dir not found: {image_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"image_dir is not a directory: {image_dir}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=_frame_sort_key)
    return [str(p) for p in paths]


def _frame_id_to_int(value: str) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _select_pattern_keyframe_indices(
    frames: Sequence[FrameRef],
    interval: int = 150,
) -> Tuple[List[int], Dict[int, int]]:
    step = max(1, int(interval))
    source_to_idx: Dict[int, int] = {}
    ordered_source_idx: List[Tuple[int, int]] = []
    for fr in frames:
        source_id = _frame_id_to_int(fr.frame_id)
        if source_id is None:
            continue
        if source_id not in source_to_idx:
            source_to_idx[source_id] = int(fr.idx)
            ordered_source_idx.append((source_id, int(fr.idx)))
    ordered_source_idx.sort(key=lambda x: x[0])
    ordered_source_ids = [x[0] for x in ordered_source_idx]
    ordered_indices = [x[1] for x in ordered_source_idx]
    if not ordered_source_ids:
        return [], {}

    selected_indices: List[int] = []
    source_id_by_index: Dict[int, int] = {}
    seen_indices = set()
    max_source_id = ordered_source_ids[-1]
    for target_source_id in range(0, max_source_id + 1, step):
        matched_idx = source_to_idx.get(int(target_source_id))
        if matched_idx is None:
            # Causal fallback: use nearest available source frame <= target pattern frame.
            pos = bisect_right(ordered_source_ids, int(target_source_id)) - 1
            if pos >= 0:
                matched_idx = ordered_indices[pos]
        if matched_idx is None or matched_idx in seen_indices:
            continue
        seen_indices.add(matched_idx)
        selected_indices.append(matched_idx)
        source_id_by_index[matched_idx] = int(target_source_id)

    selected_indices.sort()
    return selected_indices, source_id_by_index


def _normalize_final_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"satisfied", "satisfy", "yes", "true", "1"}:
        return "satisfied"
    if text in {"unsatisfied", "not_satisfied", "no", "false", "0"}:
        return "unsatisfied"
    return "uncertain"


def _safe_token(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("._")
    return token or "value"


def _preferred_tools_tag(preferred_tools: Sequence[str]) -> str:
    """Build a short tag from the preferred tools list for use in output paths."""
    if not preferred_tools:
        return "pref-none"
    _abbrev: Dict[str, str] = {
        "scene_cvs_analyzer": "cvs",
        "scene_cvs_act_analyzer": "cvsact",
        "scene_snapper": "snap",
        "clip_analyzer": "clip",
        "action_rec": "arec",
        "left_rec": "left",
        "right_rec": "right",
        "camera_rec": "cam",
        "zoom": "zoom",
    }
    parts = [_abbrev.get(t, _safe_token(t)) for t in preferred_tools]
    return "pref-" + "_".join(parts)


def _allowed_tools_tag(allowed_tools: Sequence[str]) -> str:
    """Build a stable tag for non-default allowed tool sets."""
    default_tools = sorted(ALL_TOOL_NAMES)
    current_tools = sorted({str(tool).strip() for tool in allowed_tools if str(tool).strip()})
    if current_tools == default_tools:
        return ""
    _abbrev: Dict[str, str] = {
        "scene_cvs_analyzer": "cvs",
        "scene_cvs_act_analyzer": "cvsact",
        "scene_snapper": "snap",
        "clip_analyzer": "clip",
        "action_rec": "arec",
        "left_rec": "left",
        "right_rec": "right",
        "camera_rec": "cam",
        "zoom": "zoom",
    }
    parts = [_abbrev.get(tool, _safe_token(tool)) for tool in current_tools]
    return "_tools-" + "_".join(parts)


def _allocate_trace_run_dir(
    *,
    trace_dir: Path,
    out_jsonl: Path,
) -> Path:
    """Derive trace directory from the output JSONL path so they stay in sync.

    out_jsonl: .../outputs/surgent_sequential_agent/cot/pref_tag/vid__multi__...jsonl
    result:    trace_dir/YYYYMMDD/cot/pref_tag/vid__multi__.../
    """
    date_folder = datetime.now().strftime("%Y%m%d")
    # Mirror the directory structure under surgent_sequential_agent/
    sa_base = REPO_ROOT / "outputs" / "surgent_sequential_agent"
    try:
        rel = out_jsonl.resolve().relative_to(sa_base.resolve())
    except ValueError:
        # out_jsonl is not under the expected base (e.g. custom --out-jsonl)
        rel = Path(out_jsonl.stem)
    # rel = cot/pref_tag/filename.jsonl  ->  use parent dirs + stem as dir name
    trace_run_dir = trace_dir / date_folder / rel.parent / rel.stem
    if trace_run_dir.exists():
        shutil.rmtree(trace_run_dir, ignore_errors=True)
    trace_run_dir.mkdir(parents=True, exist_ok=True)
    return trace_run_dir


def _write_run_meta(
    *,
    path: Path,
    command: str,
    video_id: str,
    criteria: Sequence[str],
    focus_criteria: Sequence[str],
    model_default: str,
    cache_backend: str,
    forced_console_verbose: int,
    forced_trace_verbose: int,
    output_jsonl: Path,
    trace_steps_jsonl: Path,
    status: str,
    rows_written: int,
    total_steps: int,
    last_step_index: Optional[int],
    total_elapsed_seconds: Optional[float] = None,
) -> None:
    lines = [
        f"status: {status}",
        f"rows_written: {rows_written}",
        f"total_steps: {total_steps}",
        f"command: {command}",
        f"video_id: {video_id}",
        f"criteria: {','.join(criteria)}",
        f"focus_criteria: {','.join(focus_criteria)}",
        f"model_default: {model_default}",
        f"cache: {cache_backend}",
        f"forced_console_verbose: {forced_console_verbose}",
        f"forced_trace_verbose: {forced_trace_verbose}",
        f"output_jsonl: {output_jsonl}",
        f"trace_steps_jsonl: {trace_steps_jsonl}",
    ]
    if last_step_index is not None:
        lines.append(f"last_step_index: {last_step_index}")
    if total_elapsed_seconds is not None:
        lines.append(f"total_elapsed_seconds: {total_elapsed_seconds}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_cache_token(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("._")
    return token or "model"


def _configure_cache(cache_backend: str, scene_model_id: str, clip_model_id: str, controller_model_id: str) -> None:
    if cache_backend == "memory":
        set_llm_cache(InMemoryCache())
    elif cache_backend == "none":
        set_llm_cache(None)
    else:
        cache_path_env = os.getenv("SURGENT_LLM_CACHE_PATH")
        if cache_path_env:
            cache_path = Path(cache_path_env)
        else:
            cache_name = (
                "langchain_llm_cache__"
                f"scene-{_safe_cache_token(scene_model_id)}__"
                f"clip-{_safe_cache_token(clip_model_id)}__"
                f"controller-{_safe_cache_token(controller_model_id)}.db"
            )
            cache_path = Path(".cache") / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(cache_path)))


def _init_model(prefix: str, default_model: str) -> Any:
    model_id = os.getenv(f"{prefix}_MODEL", default_model)
    base_url = os.getenv(f"{prefix}_BASE_URL")
    explicit_key = os.getenv(f"{prefix}_API_KEY")
    explicit_provider = os.getenv(f"{prefix}_MODEL_PROVIDER")
    model_lower = str(model_id).lower()
    is_google_vertex = any(tok in model_lower for tok in ("gemini", "vertex", "google"))
    is_qwen = model_lower.startswith("qwen/")
    api_key: Optional[str]
    if is_google_vertex:
        api_key = None
    elif is_qwen:
        api_key = os.getenv("QWEN_API_KEY") or "brachiokey"
    elif explicit_key:
        api_key = explicit_key
    elif "anthropic" in model_lower or "claude" in model_lower:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif "openai" in model_lower or "gpt" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    kwargs: Dict[str, Any] = {"temperature": 0, "max_retries": 20}
    if explicit_provider:
        kwargs["model_provider"] = explicit_provider
    elif "/" in str(model_id):
        inferred_provider = str(model_id).split("/", 1)[0].strip().lower()
        if inferred_provider:
            kwargs["model_provider"] = inferred_provider
    if is_qwen:
        kwargs["model_provider"] = "openai"
        if not base_url:
            base_url = os.getenv("QWEN_BASE_URL") or _default_qwen_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    provider_hint = str(kwargs.get("model_provider", "")).strip().lower()
    # Preserve raw model ids like "openai/gpt-oss-120b" for OpenAI-compatible servers.
    if provider_hint == "openai" or str(model_id).lower().startswith("openai/"):
        openai_kwargs: Dict[str, Any] = {"temperature": 0, "max_retries": 20}
        if base_url:
            openai_kwargs["base_url"] = base_url
        if api_key:
            openai_kwargs["api_key"] = api_key
        return ChatOpenAI(model=str(model_id), **openai_kwargs)
    return init_chat_model(model_id, **kwargs)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _parse_criteria_arg(criterion: Optional[str], criteria_csv: Optional[str]) -> List[str]:
    if criteria_csv:
        raw = [x.strip().upper() for x in str(criteria_csv).split(",") if x.strip()]
    elif criterion:
        raw = [str(criterion).strip().upper()]
    else:
        raw = [str(c).upper() for c in DEFAULT_CVS_CRITERIA]
    seen = set()
    deduped: List[str] = []
    for c in raw:
        if not c or c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    allowed = {str(c).upper() for c in DEFAULT_CVS_CRITERIA}
    invalid = [c for c in deduped if c not in allowed]
    if invalid:
        raise SystemExit(
            f"Unsupported criteria: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return deduped or [str(c).upper() for c in DEFAULT_CVS_CRITERIA]


def _build_hm3_with_metadata(image_dir: Path, frames: List[FrameRef]) -> HM3Memory:
    hm3 = HM3Memory()
    hm3.metadata = VideoMetadata(
        video_id=image_dir.name,
        image_dir=str(image_dir),
        frames=frames,
        num_frames=len(frames),
    )
    return hm3


def _extract_scene_cvs_output(
    hm3: HM3Memory,
    iteration_start: int,
    iteration_end: int,
) -> Optional[Dict[str, Any]]:
    """Extract the latest scene_cvs_analyzer/scene_cvs_act_analyzer result from HM3 memory."""
    _CVS_TOOL_NAMES = {"scene_cvs_analyzer", "scene_cvs_act_analyzer"}
    for event in reversed(hm3.results.events):
        if event.tool not in _CVS_TOOL_NAMES:
            continue
        if event.iteration < iteration_start or event.iteration > iteration_end:
            continue
        output = event.output if isinstance(event.output, dict) else {}

        # Prefer direct actions_ranked from new subrubric output
        actions_ranked_raw = output.get("actions_ranked", [])
        if isinstance(actions_ranked_raw, list) and actions_ranked_raw:
            actions_ranked = []
            for a in actions_ranked_raw:
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
                    "confidence": float(a.get("confidence", 0.0) or 0.0),
                    "evidence": str(a.get("evidence", "")),
                })
        else:
            # Fallback: construct from recommended_action (legacy format)
            recommended_action = output.get("recommended_action", {})
            if not isinstance(recommended_action, dict):
                recommended_action = {}
            actions_ranked = []
            if recommended_action:
                target_context_raw = recommended_action.get("target_context", "(not set)")
                if isinstance(target_context_raw, list):
                    target_context_1 = str(target_context_raw[0]) if len(target_context_raw) > 0 else "(not set)"
                    target_context_2 = str(target_context_raw[1]) if len(target_context_raw) > 1 else "(not set)"
                else:
                    target_context_1 = str(target_context_raw or "(not set)")
                    target_context_2 = str(recommended_action.get("target_context_2", "(not set)"))
                actions_ranked.append({
                    "rank": 1,
                    "actor_role": str(recommended_action.get("actor_role", "(not set)")),
                    "tool_type": str(recommended_action.get("tool_type", "(not set)")),
                    "action_code": str(recommended_action.get("action_code", "(not set)")),
                    "target_structure": str(recommended_action.get("target_structure", "(not set)")),
                    "target_context_1": target_context_1,
                    "target_context_2": target_context_2,
                    "intention": str(recommended_action.get("intention", "(not set)")),
                    "confidence": float(recommended_action.get("confidence", 0.0) or 0.0),
                })

        cvs_predictions = output.get("cvs_predictions", {})
        if not isinstance(cvs_predictions, dict):
            cvs_predictions = {}
        # Normalize CVS predictions to lowercase keys like baseline
        pred = {}
        for k, v in cvs_predictions.items():
            try:
                pred[k.lower()] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                pred[k.lower()] = 0.0

        recommended_action = output.get("recommended_action", {})
        if not isinstance(recommended_action, dict):
            recommended_action = {}

        result: Dict[str, Any] = {
            "pred": pred,
            "actions_ranked": actions_ranked,
            "checklist": output.get("checklist", {}),
            "recommended_action": recommended_action,
            "summary": output.get("summary", ""),
            "interval": output.get("interval", {}),
        }
        # Include legacy fields if present
        rationale = output.get("rationale")
        if rationale:
            result["rationale"] = rationale
        rubric_scores = output.get("rubric_scores")
        if rubric_scores:
            result["rubric_scores"] = rubric_scores
        return result
    return None


def _extract_action_rec_output(
    hm3: HM3Memory,
    iteration_start: int,
    iteration_end: int,
) -> Optional[Dict[str, Any]]:
    """Extract the latest action_rec result from HM3 memory for the given iteration range."""
    for event in reversed(hm3.results.events):
        if event.tool != "action_rec":
            continue
        if event.iteration < iteration_start or event.iteration > iteration_end:
            continue
        output = event.output if isinstance(event.output, dict) else {}
        actions_raw = output.get("actions_ranked", [])
        actions_ranked = []
        if isinstance(actions_raw, list):
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
                    "confidence": float(a.get("confidence", 0.0) or 0.0),
                    "evidence": str(a.get("evidence", "")),
                })
        return {
            "actions_ranked": actions_ranked,
            "reasoning": output.get("reasoning", ""),
            "summary": output.get("summary", ""),
            "interval": output.get("interval", {}),
        }
    return None


def _extract_instrument_rec_output(
    hm3: HM3Memory,
    tool_name: str,
    iteration_start: int,
    iteration_end: int,
) -> Optional[Dict[str, Any]]:
    """Extract the latest left_rec/right_rec/camera_rec result from HM3 memory for the given iteration range."""
    for event in reversed(hm3.results.events):
        if event.tool != tool_name:
            continue
        if event.iteration < iteration_start or event.iteration > iteration_end:
            continue
        output = event.output if isinstance(event.output, dict) else {}
        action = output.get("action", {})
        if not isinstance(action, dict):
            action = {}
        return {
            "action": action,
            "reasoning": output.get("reasoning", ""),
            "summary": output.get("summary", ""),
            "interval": output.get("interval", {}),
        }
    return None


def _score_to_label(score: float) -> str:
    """Derive a human-readable label from a satisfaction score."""
    if score >= 0.7:
        return "satisfied"
    if score <= 0.3:
        return "unsatisfied"
    return "uncertain"


def _default_criteria_status(criteria: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {
        c: {
            "score": 0.0,
            "reason": "not evaluated yet",
        }
        for c in criteria
    }


def _normalize_criteria_status(
    raw_status: Any,
    criteria: Sequence[str],
    previous_status: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out = _default_criteria_status(criteria)
    for c in criteria:
        prev = previous_status.get(c, {})
        out[c] = {
            "score": max(0.0, min(1.0, float(prev.get("score", 0.0) or 0.0))),
            "reason": str(prev.get("reason") or ""),
        }
    if not isinstance(raw_status, dict):
        return out
    for key, payload in raw_status.items():
        c = str(key or "").strip().upper()
        if c not in out or not isinstance(payload, dict):
            continue
        # New format: single "score" field
        if "score" in payload:
            try:
                score = max(0.0, min(1.0, float(payload.get("score", 0.0) or 0.0)))
            except Exception:
                score = float(out[c]["score"])
            out[c] = {
                "score": score,
                "reason": str(payload.get("reason") or payload.get("evidence") or out[c]["reason"]),
            }
        else:
            # Legacy format: "decision" + "confidence" → convert to score
            decision = _normalize_final_decision(payload.get("decision", "uncertain"))
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
            out[c] = {
                "score": max(0.0, min(1.0, score)),
                "reason": str(payload.get("reason") or payload.get("evidence") or out[c]["reason"]),
            }
    return out


def _infer_focus_criterion(
    criteria: Sequence[str],
    criteria_status: Dict[str, Dict[str, Any]],
    action_args: Dict[str, Any],
) -> Optional[str]:
    hinted = str(action_args.get("focus_criterion") or "").strip().upper()
    if hinted in criteria:
        return hinted
    unresolved = [c for c in criteria if criteria_status.get(c, {}).get("score", 0) < 0.7]
    if not unresolved:
        return None
    unresolved.sort(key=lambda c: float(criteria_status.get(c, {}).get("score", 0.0)))
    return unresolved[0]


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    args = parse_args()
    image_dir = Path(args.image_dir)
    image_paths = _collect_image_paths(image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")

    requested_focus_criteria = _parse_criteria_arg(args.criterion, args.criteria)
    prediction_criteria = [str(c).upper() for c in DEFAULT_CVS_CRITERIA]
    global_model_default = str(args.model or "gpt-5-mini")
    taxonomy_path = Path(args.taxonomy_json)
    taxonomy_catalog = _load_taxonomy_catalog(taxonomy_path)
    if args.no_rec_descs:
        taxonomy_catalog["recommendation_descs"] = {}
    if not args.include_field_meta:
        taxonomy_catalog["field_meta"] = {}
    taxonomy_tag = taxonomy_path.stem  # e.g. "taxonomy_v8"
    taxonomy_options = taxonomy_catalog.get("options", {})
    taxonomy_ready = isinstance(taxonomy_options, dict) and bool(taxonomy_options.get("action_code"))
    if taxonomy_ready:
        extra = ""
        if args.no_rec_descs:
            extra += " (rec-descs stripped)"
        if args.include_field_meta:
            extra += " (field-meta included)"
        print(
            "Structured action taxonomy loaded: "
            f"{taxonomy_catalog.get('version', 'unknown')} from {taxonomy_path}{extra}"
        )
    else:
        print(
            "Structured action taxonomy unavailable or missing action_code options. "
            f"Using fallback empty predictions from {taxonomy_path}."
        )

    scene_model_id = str(os.getenv("SURGENT_SCENE_MODEL", global_model_default))
    clip_model_id = str(os.getenv("SURGENT_CLIP_MODEL", global_model_default))
    controller_model_id = str(os.getenv("SURGENT_CONTROLLER_MODEL", global_model_default))
    _configure_cache(
        cache_backend=str(args.cache).strip().lower(),
        scene_model_id=scene_model_id,
        clip_model_id=clip_model_id,
        controller_model_id=controller_model_id,
    )

    forced_trace_verbose = 3
    forced_console_verbose = 1
    if int(args.verbose) != forced_console_verbose:
        print(
            "Requested verbose="
            f"{int(args.verbose)}; forcing console level={forced_console_verbose} "
            f"and trace-file level={forced_trace_verbose}."
        )
    os.environ["SURGENT_LOG_LEVEL"] = str(forced_trace_verbose)
    os.environ["SURGENT_TRACE_LOG_LEVEL"] = str(forced_trace_verbose)
    os.environ["SURGENT_CONSOLE_LOG_LEVEL"] = str(forced_console_verbose)
    os.environ["SURGENT_LOG_STEPS"] = "1"

    scene_model = _init_model("SURGENT_SCENE", global_model_default)
    clip_model = _init_model("SURGENT_CLIP", global_model_default)
    controller_model = _init_model("SURGENT_CONTROLLER", global_model_default)
    allowed_tools = {t.strip() for t in str(args.tools).split(",") if t.strip()}
    invalid_tools = allowed_tools - ALL_TOOL_NAMES
    if invalid_tools:
        raise SystemExit(f"Unknown tools: {invalid_tools}. Allowed: {sorted(ALL_TOOL_NAMES)}")
    print(f"Allowed tools: {sorted(allowed_tools)}")
    preferred_tools_raw = str(args.preferred_tools or "").strip()
    preferred_tools = [t.strip() for t in preferred_tools_raw.split(",") if t.strip()] if preferred_tools_raw else []
    invalid_preferred = set(preferred_tools) - ALL_TOOL_NAMES
    if invalid_preferred:
        raise SystemExit(f"Unknown preferred tools: {invalid_preferred}. Allowed: {sorted(ALL_TOOL_NAMES)}")
    if preferred_tools:
        print(f"Preferred tools (auto-call): {preferred_tools}")
    finalize_tools_raw = str(args.finalize_tools or "").strip()
    finalize_tools = [t.strip() for t in finalize_tools_raw.split(",") if t.strip()] if finalize_tools_raw else []
    invalid_finalize = set(finalize_tools) - ALL_TOOL_NAMES
    if invalid_finalize:
        raise SystemExit(f"Unknown finalize tools: {invalid_finalize}. Allowed: {sorted(ALL_TOOL_NAMES)}")
    if finalize_tools:
        print(f"Finalize tools (auto-call at END): {finalize_tools}")
    scene_preset = str(args.scene_preset or "direct").strip().lower()
    if scene_preset not in ("direct", "cot", "subrubric"):
        scene_preset = "direct"
    print(f"Scene preset: {scene_preset}")
    pref_tag = _preferred_tools_tag(preferred_tools)
    analyze_actions = bool(args.analyze_actions)
    if analyze_actions:
        print("Action analysis context: ENABLED")
    recommend_actions = bool(args.recommend_actions)
    if recommend_actions:
        print("Recommend actions mode: ENABLED (scene_cvs_analyzer will recommend next actions)")
    fixed_k = args.fixed_k
    if fixed_k is not None:
        print(f"Fixed-K mode: ENABLED (exactly {fixed_k} actions per frame)")
    cvs_context_mode = args.cvs_context_mode
    if cvs_context_mode != "latest":
        print(f"CVS context mode: {cvs_context_mode}")
    action_rec_rules = str(args.action_rec_rules or "default")
    if action_rec_rules != "default":
        print(f"Action-rec rules: {action_rec_rules}")
    ctx = SurgentContext(
        models={"scene": scene_model, "clip": clip_model, "controller": controller_model},
        allowed_tools=allowed_tools,
        taxonomy_catalog=taxonomy_catalog,
        preferred_tools=preferred_tools,
        finalize_tools=finalize_tools or None,
        scene_preset=scene_preset,
        analyze_actions=analyze_actions,
        recommend_actions=recommend_actions,
        fixed_k=fixed_k,
        cvs_context_mode=cvs_context_mode,
        one_per_actor=bool(args.one_per_actor),
        action_rec_rules=action_rec_rules,
    )

    frames = [FrameRef(idx=i, frame_id=Path(p).stem, path=p) for i, p in enumerate(image_paths)]
    hm3 = _build_hm3_with_metadata(image_dir=image_dir, frames=frames)

    frame_step = max(1, int(args.frame_step))
    start_idx = max(0, int(args.start_frame_idx))
    frame_indices = list(range(start_idx, len(image_paths), frame_step))
    pattern_source_by_frame_index: Dict[int, int] = {}
    if str(args.sampling_mode).strip().lower() == "keyframes":
        pattern_indices, pattern_source_by_frame_index = _select_pattern_keyframe_indices(
            frames=frames,
            interval=max(1, int(args.keyframe_interval)),
        )
        frame_indices = [idx for idx in pattern_indices if idx >= start_idx]
        if frame_step > 1:
            frame_indices = frame_indices[::frame_step]
        print(
            f"Sampling mode=keyframes fixed pattern interval={max(1, int(args.keyframe_interval))} "
            f"(selected={len(frame_indices)})"
        )
    if args.max_source_frame is not None:
        max_src = int(args.max_source_frame)
        frame_indices = [
            idx for idx in frame_indices
            if (_frame_id_to_int(frames[idx].frame_id) or 0) <= max_src
        ]
        print(f"Filtered to source frame <= {max_src} (remaining={len(frame_indices)})")
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(0, int(args.max_frames))]
    if not frame_indices:
        raise SystemExit("No evaluation frame indices selected after sampling/filtering.")

    # Include max_steps in the output subdirectory when explicitly set
    max_steps_env = os.environ.get("SURGENT_MAX_STEPS")
    if max_steps_env is not None:
        pref_tag_dir = f"{pref_tag}_steps{max_steps_env}"
    else:
        pref_tag_dir = pref_tag
    # _aa and _ra suffixes removed: recommend_actions is now default-on,
    # and analyze_actions context is folded into the two-stage controller.
    if fixed_k is not None:
        pref_tag_dir = f"{pref_tag_dir}_fixedk{fixed_k}"
    if args.no_rec_descs:
        pref_tag_dir = f"{pref_tag_dir}_norecdescs"
    if args.include_field_meta:
        pref_tag_dir = f"{pref_tag_dir}_fmeta"
    if args.one_per_actor:
        pref_tag_dir = f"{pref_tag_dir}_1peractor"
    tools_tag = _allowed_tools_tag(sorted(allowed_tools))
    if tools_tag:
        pref_tag_dir = f"{pref_tag_dir}{tools_tag}"
    _CVS_CTX_SUFFIXES = {
        "current": "_cvsctx-current",
        "full": "_cvsctx-full",
        "video_summary": "_cvsctx-vidsummary",
    }
    if cvs_context_mode in _CVS_CTX_SUFFIXES:
        pref_tag_dir = f"{pref_tag_dir}{_CVS_CTX_SUFFIXES[cvs_context_mode]}"
    if action_rec_rules != "default":
        pref_tag_dir = f"{pref_tag_dir}_arecrules-{_safe_token(action_rec_rules).replace('_', '-')}"
    if finalize_tools:
        _ft_abbrev: Dict[str, str] = {
            "scene_cvs_analyzer": "cvs",
            "scene_cvs_act_analyzer": "cvsact",
            "scene_snapper": "snap",
            "clip_analyzer": "clip",
            "action_rec": "arec",
            "left_rec": "left",
            "right_rec": "right",
            "camera_rec": "cam",
            "zoom": "zoom",
        }
        ft_tag = "-".join(_ft_abbrev.get(t, _safe_token(t)) for t in finalize_tools)
        pref_tag_dir = f"{pref_tag_dir}_final-{ft_tag}"

    out_jsonl = (
        Path(args.out_jsonl)
        if args.out_jsonl
        else REPO_ROOT
        / "outputs"
        / "surgent_sequential_agent"
        / scene_preset
        / pref_tag_dir
        / f"{image_dir.name}__multi__{'-'.join(prediction_criteria)}__{_safe_token(global_model_default)}__{_safe_token(taxonomy_tag)}.jsonl"
    )
    trace_run_dir = _allocate_trace_run_dir(
        trace_dir=Path(args.trace_dir),
        out_jsonl=out_jsonl,
    )
    os.environ["SURGENT_TRACE_FILE"] = str(trace_run_dir / "trace.log")
    os.environ["SURGENT_MOSAIC_DIR"] = str(trace_run_dir / "mosaics")
    print(f"Detailed trace dir: {trace_run_dir}")
    trace_steps_jsonl = trace_run_dir / "steps_live.jsonl"
    print(f"Live step trace JSONL: {trace_steps_jsonl}")

    graph = build_agent_graph().compile()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    prev_signature: Optional[str] = None
    multi_criterion_token = ",".join(prediction_criteria)
    focus_criteria = requested_focus_criteria or list(prediction_criteria)
    total_steps = len(frame_indices)
    run_meta_path = trace_run_dir / "run_meta.txt"
    _write_run_meta(
        path=run_meta_path,
        command=" ".join(sys.argv),
        video_id=image_dir.name,
        criteria=prediction_criteria,
        focus_criteria=focus_criteria,
        model_default=global_model_default,
        cache_backend=str(args.cache).strip().lower(),
        forced_console_verbose=forced_console_verbose,
        forced_trace_verbose=forced_trace_verbose,
        output_jsonl=out_jsonl,
        trace_steps_jsonl=trace_steps_jsonl,
        status="running",
        rows_written=rows_written,
        total_steps=total_steps,
        last_step_index=None,
    )
    run_start_time = time.monotonic()
    with out_jsonl.open("w", encoding="utf-8") as out_f, trace_steps_jsonl.open("w", encoding="utf-8") as trace_steps_f:
        for step_idx, current_idx in enumerate(frame_indices):
            step_start_time = time.monotonic()
            if args.lookback_frames is None:
                visible_start = 0
            else:
                lb = max(1, int(args.lookback_frames))
                visible_start = max(0, current_idx - lb + 1)

            hm3.set_visible_window(start_idx=visible_start, end_idx=current_idx)
            iteration_before = hm3.iteration_cursor
            final_state = run_cvs_agent_with_memory(
                criterion=multi_criterion_token,
                ctx=ctx,
                hm3=hm3,
                compiled_graph=graph,
            )
            iteration_after = hm3.iteration_cursor
            decision = final_state.get("decision", {})
            if not isinstance(decision, dict):
                decision = {}
            action = str(decision.get("action") or "ANSWER")
            action_args = decision.get("action_args") or {}
            reason = str(decision.get("reason") or "")
            controller_recommended_actions = decision.get("recommended_actions", [])

            # Frame-level: directly from this step's controller decision (no carryforward)
            current_frame_pred = _normalize_criteria_status(
                raw_status=decision.get("criteria_status"),
                criteria=prediction_criteria,
                previous_status=_default_criteria_status(prediction_criteria),
            )

            # Video-level: read from memory (already max-accumulated by graph.py)
            video_criteria_status: Dict[str, Dict[str, Any]] = {}
            for c in prediction_criteria:
                mem_entry = hm3.video_criteria_status.get(c, {})
                video_criteria_status[c] = {
                    "score": max(0.0, min(1.0, float(mem_entry.get("score", 0.0) or 0.0))),
                    "reason": str(mem_entry.get("reason") or "not evaluated yet"),
                }

            # Derive from frame-level prediction (not from LLM's decision field)
            frame_scores = [current_frame_pred[c]["score"] for c in prediction_criteria]
            confidence = sum(frame_scores) / len(frame_scores) if frame_scores else 0.0
            final_decision = "satisfied" if all(s >= 0.5 for s in frame_scores) else "unsatisfied"

            # Video-level all_satisfied
            all_satisfied = all(
                video_criteria_status.get(c, {}).get("score", 0) >= 0.5
                for c in prediction_criteria
            )
            active_criterion = _infer_focus_criterion(focus_criteria, current_frame_pred, action_args)

            # Extract scene_cvs_output and action_rec_output from memory
            scene_cvs_output = _extract_scene_cvs_output(
                hm3,
                iteration_start=iteration_before,
                iteration_end=iteration_after,
            )
            action_rec_output = _extract_action_rec_output(
                hm3,
                iteration_start=iteration_before,
                iteration_end=iteration_after,
            )

            # Action priority:
            #   1. action_rec (has CVS context + frame access)
            #   2. scene_cvs_act_analyzer actions_ranked (combined tool, only if used)
            #   3. controller recommended_actions (text-only)
            #   4. _predict_taxonomy_action() (last resort)
            _action_rec_actions = (
                action_rec_output.get("actions_ranked", [])
                if isinstance(action_rec_output, dict)
                else []
            )
            _action_rec_has_valid_actions = (
                isinstance(_action_rec_actions, list)
                and any(
                    isinstance(a, dict) and str(a.get("action_code", "(not set)")) != "(not set)"
                    for a in _action_rec_actions
                )
            )
            _scene_cvs_actions = (
                scene_cvs_output.get("actions_ranked", [])
                if isinstance(scene_cvs_output, dict)
                else []
            )
            _scene_cvs_has_valid_actions = (
                isinstance(_scene_cvs_actions, list)
                and any(
                    isinstance(a, dict) and str(a.get("action_code", "(not set)")) != "(not set)"
                    for a in _scene_cvs_actions
                )
            )
            _controller_has_valid_actions = (
                isinstance(controller_recommended_actions, list)
                and any(
                    isinstance(a, dict) and str(a.get("action_code", "(not set)")) != "(not set)"
                    for a in controller_recommended_actions
                )
            )

            if _action_rec_has_valid_actions:
                # Prefer action_rec (has CVS context + frame access)
                _first = _action_rec_actions[0] if _action_rec_actions else {}
                predicted_taxonomy_action = {
                    "criterion": str(active_criterion or ""),
                    "rank": 1,
                    "actor_role": str(_first.get("actor_role", TAXONOMY_EMPTY)),
                    "tool_type": str(_first.get("tool_type", TAXONOMY_EMPTY)),
                    "action_code": str(_first.get("action_code", TAXONOMY_EMPTY)),
                    "target_structure": str(_first.get("target_structure", TAXONOMY_EMPTY)),
                    "target_context_1": str(_first.get("target_context_1", TAXONOMY_EMPTY)),
                    "target_context_2": str(_first.get("target_context_2", TAXONOMY_EMPTY)),
                    "intention": str(_first.get("intention", TAXONOMY_EMPTY)),
                    "generated_sentence": "",
                    "confidence": float(_first.get("confidence", 0.0) or 0.0),
                    "taxonomy_version": str(taxonomy_catalog.get("version") or "unknown"),
                    "taxonomy_path": str(taxonomy_path),
                    "source": "action_rec",
                }
            elif _scene_cvs_has_valid_actions:
                # Fallback: scene_cvs_act_analyzer actions (combined tool)
                _first = _scene_cvs_actions[0] if _scene_cvs_actions else {}
                predicted_taxonomy_action = {
                    "criterion": str(active_criterion or ""),
                    "rank": 1,
                    "actor_role": str(_first.get("actor_role", TAXONOMY_EMPTY)),
                    "tool_type": str(_first.get("tool_type", TAXONOMY_EMPTY)),
                    "action_code": str(_first.get("action_code", TAXONOMY_EMPTY)),
                    "target_structure": str(_first.get("target_structure", TAXONOMY_EMPTY)),
                    "target_context_1": str(_first.get("target_context_1", TAXONOMY_EMPTY)),
                    "target_context_2": str(_first.get("target_context_2", TAXONOMY_EMPTY)),
                    "intention": str(_first.get("intention", TAXONOMY_EMPTY)),
                    "generated_sentence": "",
                    "confidence": float(_first.get("confidence", 0.0) or 0.0),
                    "taxonomy_version": str(taxonomy_catalog.get("version") or "unknown"),
                    "taxonomy_path": str(taxonomy_path),
                    "source": "scene_cvs_act_analyzer",
                }
            elif _controller_has_valid_actions:
                # Fall back to controller actions
                _first = controller_recommended_actions[0] if controller_recommended_actions else {}
                predicted_taxonomy_action = {
                    "criterion": str(active_criterion or ""),
                    "rank": 1,
                    "actor_role": str(_first.get("actor_role", TAXONOMY_EMPTY)),
                    "tool_type": str(_first.get("tool_type", TAXONOMY_EMPTY)),
                    "action_code": str(_first.get("action_code", TAXONOMY_EMPTY)),
                    "target_structure": str(_first.get("target_structure", TAXONOMY_EMPTY)),
                    "target_context_1": str(_first.get("target_context_1", TAXONOMY_EMPTY)),
                    "target_context_2": str(_first.get("target_context_2", TAXONOMY_EMPTY)),
                    "intention": str(_first.get("intention", TAXONOMY_EMPTY)),
                    "generated_sentence": "",
                    "confidence": float(_first.get("confidence", 0.0) or 0.0),
                    "taxonomy_version": str(taxonomy_catalog.get("version") or "unknown"),
                    "taxonomy_path": str(taxonomy_path),
                    "source": "controller",
                }
            else:
                predicted_taxonomy_action = _predict_taxonomy_action(
                    model=ctx.models.get("controller"),
                    hm3=hm3,
                    criterion=active_criterion,
                    decision=decision,
                    taxonomy_catalog=taxonomy_catalog,
                    taxonomy_path=taxonomy_path,
                )
            left_rec_output = None
            right_rec_output = None
            camera_rec_output = None
            hm3.record_result(
                tool="criteria_status_tracker",
                interval={"start": visible_start, "end": current_idx},
                output={
                    "current_frame_criteria_prediction": current_frame_pred,
                    "video_criteria_status": video_criteria_status,
                    "all_criteria_satisfied": all_satisfied,
                    "controller_action": action,
                    "controller_decision": final_decision,
                },
                iteration=int(iteration_after),
            )
            signature = json.dumps(
                {
                    "active_criterion": active_criterion,
                    "criteria_scores": {c: current_frame_pred[c]["score"] for c in prediction_criteria},
                    "video_criteria_scores": {c: video_criteria_status[c]["score"] for c in prediction_criteria},
                    "action": action,
                    "decision": final_decision,
                    "action_args": action_args,
                    "predicted_taxonomy_action_code": predicted_taxonomy_action.get("action_code"),
                    "predicted_taxonomy_criterion": predicted_taxonomy_action.get("criterion"),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            changed = prev_signature is None or signature != prev_signature
            prev_signature = signature

            # Build authoritative actions_ranked:
            #   action_rec > scene_cvs_act_analyzer > controller
            def _build_actions_ranked_final(actions: list, source: str) -> list:
                result = []
                for i, a in enumerate(actions[:4]):
                    if not isinstance(a, dict):
                        continue
                    result.append({
                        "rank": i + 1,
                        "actor_role": str(a.get("actor_role", "(not set)")),
                        "tool_type": str(a.get("tool_type", "(not set)")),
                        "action_code": str(a.get("action_code", "(not set)")),
                        "target_structure": str(a.get("target_structure", "(not set)")),
                        "target_context_1": str(a.get("target_context_1", "(not set)")),
                        "target_context_2": str(a.get("target_context_2", "(not set)")),
                        "intention": str(a.get("intention", "(not set)")),
                        "confidence": float(a.get("confidence", 0.0) or 0.0),
                        "evidence": str(a.get("evidence", "")),
                        "source": source,
                    })
                return result

            if _action_rec_has_valid_actions:
                actions_ranked_final = _build_actions_ranked_final(_action_rec_actions, "action_rec")
            elif _scene_cvs_has_valid_actions:
                actions_ranked_final = _build_actions_ranked_final(_scene_cvs_actions, "scene_cvs_act_analyzer")
            elif _controller_has_valid_actions:
                actions_ranked_final = _build_actions_ranked_final(controller_recommended_actions, "controller")
            else:
                actions_ranked_final = None  # let load_predictions() fall back to concatenation

            row = {
                "schema_version": "surgent.sequential.agent.multi.v6",
                "video_id": image_dir.name,
                "criteria": list(prediction_criteria),
                "focus_criteria": list(focus_criteria),
                "sampling_mode": str(args.sampling_mode),
                "pattern_source_frame_id": pattern_source_by_frame_index.get(current_idx),
                "labeled_source_frame_id": pattern_source_by_frame_index.get(current_idx),
                "criterion": active_criterion,
                "step_index": step_idx,
                "current_frame_index": current_idx,
                "current_frame_id": frames[current_idx].frame_id,
                "visible_window": {
                    "start_index": visible_start,
                    "end_index": current_idx,
                    "start_frame_id": frames[visible_start].frame_id,
                    "end_frame_id": frames[current_idx].frame_id,
                },
                "active_criterion": active_criterion,
                "all_criteria_satisfied": all_satisfied,
                "current_frame_criteria_prediction": {
                    c.lower(): {
                        "score": current_frame_pred[c]["score"],
                        "label": _score_to_label(current_frame_pred[c]["score"]),
                        "reason": current_frame_pred[c]["reason"],
                    }
                    for c in prediction_criteria
                },
                "video_level_criteria_status": video_criteria_status,
                "criteria_status": video_criteria_status,  # backward compat
                "cvs_predictions": {  # backward compat — uses frame-level
                    c.lower(): {
                        "score": current_frame_pred[c]["score"],
                        "label": _score_to_label(current_frame_pred[c]["score"]),
                        "reason": current_frame_pred[c]["reason"],
                    }
                    for c in prediction_criteria
                },
                "agent": {
                    "action": action,
                    "action_args": action_args,
                    "decision": final_decision,  # derived, not from LLM
                    "confidence": confidence,    # derived, not from LLM
                    "reason": reason,
                },
                "predicted_taxonomy_action": predicted_taxonomy_action,
                "scene_cvs_output": scene_cvs_output,
                "action_rec_output": action_rec_output,
                "left_rec_output": left_rec_output,
                "right_rec_output": right_rec_output,
                "camera_rec_output": camera_rec_output,
                "controller_recommended_actions": controller_recommended_actions,
                **({"actions_ranked": actions_ranked_final} if actions_ranked_final is not None else {}),
                "changed": changed,
                "timing": {
                    "step_elapsed_seconds": round(time.monotonic() - step_start_time, 2),
                    "total_elapsed_seconds": round(time.monotonic() - run_start_time, 2),
                },
                "memory": {
                    "iteration_before": iteration_before,
                    "iteration_after": iteration_after,
                    "result_events": len(hm3.results.events),
                    "working_entries": len(hm3.working.entries),
                },
            }
            if int(args.memory_tail) > 0:
                row["memory_snapshot"] = hm3.snapshot(tail=int(args.memory_tail))
            serialized = json.dumps(row, ensure_ascii=False)
            out_f.write(serialized)
            out_f.write("\n")
            out_f.flush()
            trace_steps_f.write(serialized)
            trace_steps_f.write("\n")
            trace_steps_f.flush()
            rows_written += 1
            _write_run_meta(
                path=run_meta_path,
                command=" ".join(sys.argv),
                video_id=image_dir.name,
                criteria=prediction_criteria,
                focus_criteria=focus_criteria,
                model_default=global_model_default,
                cache_backend=str(args.cache).strip().lower(),
                forced_console_verbose=forced_console_verbose,
                forced_trace_verbose=forced_trace_verbose,
                output_jsonl=out_jsonl,
                trace_steps_jsonl=trace_steps_jsonl,
                status="running",
                rows_written=rows_written,
                total_steps=total_steps,
                last_step_index=step_idx,
            )
            decision_summary = ", ".join([
                f"{c}={current_frame_pred[c]['score']:.2f} (vid={video_criteria_status[c]['score']:.2f})"
                for c in prediction_criteria
            ])
            print(
                f"[step {step_idx + 1}/{len(frame_indices)}] "
                f"frame={frames[current_idx].frame_id} target={active_criterion or 'none'} "
                f"decision={final_decision} all_satisfied={all_satisfied} changed={changed} "
                f"criteria=[{decision_summary}]"
            )
            if scene_cvs_output is not None:
                _pred = scene_cvs_output.get("pred", {})
                if _pred:
                    _pred_str = "  ".join(
                        f"{k.upper()}={v:.2f}" if isinstance(v, float) else f"{k.upper()}={v}"
                        for k, v in _pred.items()
                    )
                    print(f"  CVS predictions: {_pred_str}")
                for _a in scene_cvs_output.get("actions_ranked", []):
                    print(
                        f"    [rank {_a.get('rank')}] "
                        f"{_a.get('actor_role', '(not set)')}/{_a.get('tool_type', '(not set)')} "
                        f"{_a.get('action_code', '(not set)')} -> {_a.get('target_structure', '(not set)')} "
                        f"ctx=({_a.get('target_context_1', '(not set)')}, {_a.get('target_context_2', '(not set)')}) "
                        f"| intention: {_a.get('intention', '(not set)')} "
                        f"| conf={_a.get('confidence', 0.0)}"
                    )
                if not scene_cvs_output.get("actions_ranked"):
                    print("    (no recommended action from scene_cvs_analyzer)")
            else:
                print("  (no scene_cvs_analyzer output this step)")
            if _action_rec_has_valid_actions:
                print(f"  Authoritative actions: action_rec")
                for _a in _action_rec_actions:
                    if isinstance(_a, dict) and str(_a.get("action_code", "(not set)")) != "(not set)":
                        print(
                            f"    [arec rank {_a.get('rank')}] "
                            f"{_a.get('actor_role', '(not set)')}/{_a.get('tool_type', '(not set)')} "
                            f"{_a.get('action_code', '(not set)')} -> {_a.get('target_structure', '(not set)')} "
                            f"ctx=({_a.get('target_context_1', '(not set)')}, {_a.get('target_context_2', '(not set)')}) "
                            f"| intention: {_a.get('intention', '(not set)')} "
                            f"| conf={_a.get('confidence', 0.0)}"
                        )
            elif _scene_cvs_has_valid_actions:
                print(f"  Authoritative actions: scene_cvs_act_analyzer")
            elif _controller_has_valid_actions:
                print(f"  Authoritative actions: controller (no tool actions)")
                for _a in controller_recommended_actions:
                    if isinstance(_a, dict) and str(_a.get("action_code", "(not set)")) != "(not set)":
                        print(
                            f"    [ctrl rank {_a.get('rank')}] "
                            f"{_a.get('actor_role', '(not set)')}/{_a.get('tool_type', '(not set)')} "
                            f"{_a.get('action_code', '(not set)')} -> {_a.get('target_structure', '(not set)')} "
                            f"ctx=({_a.get('target_context_1', '(not set)')}, {_a.get('target_context_2', '(not set)')}) "
                            f"| intention: {_a.get('intention', '(not set)')} "
                            f"| conf={_a.get('confidence', 0.0)}"
                        )

    total_elapsed = time.monotonic() - run_start_time
    total_elapsed_str = f"{total_elapsed:.1f}s"
    if total_elapsed >= 60:
        minutes = int(total_elapsed // 60)
        seconds = total_elapsed % 60
        total_elapsed_str = f"{minutes}m {seconds:.1f}s"

    print(f"Wrote {rows_written} sequential agent records to {out_jsonl}")
    _write_run_meta(
        path=run_meta_path,
        command=" ".join(sys.argv),
        video_id=image_dir.name,
        criteria=prediction_criteria,
        focus_criteria=focus_criteria,
        model_default=global_model_default,
        cache_backend=str(args.cache).strip().lower(),
        forced_console_verbose=forced_console_verbose,
        forced_trace_verbose=forced_trace_verbose,
        output_jsonl=out_jsonl,
        trace_steps_jsonl=trace_steps_jsonl,
        status="completed",
        rows_written=rows_written,
        total_steps=total_steps,
        last_step_index=(rows_written - 1) if rows_written > 0 else None,
        total_elapsed_seconds=round(total_elapsed, 2),
    )
    print(f"Detailed trace saved to: {trace_run_dir}")
    print(f"Total time: {total_elapsed_str} ({total_elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
