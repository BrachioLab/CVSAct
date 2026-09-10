#!/usr/bin/env python3
"""Evaluate CVS-Act v1 predictions with CVS + dual recommendation actions.

This combines:
- CVS frame/video metrics from the legacy evaluation path
- recommendation-style action metrics against audit_v11 GT
- recommendation-style action metrics against synthetic GT

This public entrypoint reads trained-annotator and synthetic action labels from
the HF-style local repo checkout under ``hf_repos/cvs-act`` while preserving
the existing local CVS frame/video label evaluation path.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from evaluate_all import (  # noqa: E402
    CRITERIA,
    collect_pred_files_recursive,
    extract_model_from_pred,
    extract_taxonomy_from_pred,
    extract_video_id_from_pred,
)
from evaluate_baseline import evaluate_cvs, load_predictions  # noqa: E402
from evaluate_audit_v11_simple_recommendation import (  # noqa: E402
    build_summary_rows as build_recommendation_summary_rows,
    choose_start_frame,
    frame_match_f1,
    gt_records_by_video,
    inspect_prediction_file,
    is_complete_prediction,
    keyframes_in_clip,
    parse_modified_since,
    pred_actor_row,
    prediction_frame_number,
    start_frame_pm1_keyframes,
    gt_actor_row,
    write_json,
)
from cvs_act.action_segment_eval import GRANULARITIES, convert_record_to_simple_actions, load_audit_records  # noqa: E402


EVAL_ACTORS = ["left", "camera", "right", "other"]

DEFAULT_SYNTHETIC_PRESETS = {
    "v3.1": {
        "records_jsonl": "hf_repos/cvs-act/sages_synthetic/test.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trained-records-jsonl",
        default="hf_repos/cvs-act/sages_trained_annotator/test.jsonl",
        help="HF-style JSONL containing trained-annotator CVS-Act v1 action labels. Used only with --action-gt-source hf.",
    )
    parser.add_argument(
        "--action-gt-source",
        choices=["audit_v11", "hf"],
        default="audit_v11",
        help="Action GT source for recommendation metrics. audit_v11 reads local coarse/fine annotations; hf reads --trained-records-jsonl.",
    )
    parser.add_argument(
        "--audit-actions-dir",
        default="data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/audit_v11",
        help="Local audit_v11 annotation directory used when --action-gt-source audit_v11.",
    )
    parser.add_argument(
        "--audit-clip-level",
        choices=["coarse", "fine", "both"],
        default="both",
        help="Which audit_v11 clip sections to evaluate as action GT records when --action-gt-source audit_v11.",
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=["outputs/cot_audit_v11_simple", "outputs/surgent_sequential_agent_audit_v11_simple"],
        help="Prediction output roots to scan recursively for JSONL files.",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Optional CVS labels directory. If omitted, auto-resolve train/test labels.",
    )
    parser.add_argument(
        "--synthetic-preset",
        default="v3.1",
        help="Synthetic GT preset name. Default is the built-in v3.1 preset.",
    )
    parser.add_argument(
        "--synthetic-records-jsonl",
        default="",
        help="Optional override path to synthetic GT records JSONL.",
    )
    parser.add_argument(
        "--skip-synthetic-action-eval",
        action="store_true",
        help="Skip synthetic-GT recommendation evaluation.",
    )
    parser.add_argument(
        "--no-synthetic-fine-from-coarse",
        action="store_true",
        help="Do not add synthetic fine-start records derived from synthetic coarse records.",
    )
    parser.add_argument(
        "--metrics-base",
        default="metrics_audit_v11_simple_combined",
        help="Directory where combined summary artifacts will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to only these model names.",
    )
    parser.add_argument(
        "--taxonomy",
        nargs="+",
        default=None,
        help="Filter to only these taxonomy versions.",
    )
    parser.add_argument(
        "--common-only",
        action="store_true",
        help="Only evaluate videos that exist in all method/model/taxonomy groups.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include prediction files even if they appear incomplete for their video.",
    )
    parser.add_argument(
        "--modified-since",
        default="",
        help="Optional local timestamp cutoff like '2026-05-24 17:00:00'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress and skip reasons.",
    )
    return parser.parse_args()


def method_key(method_label: str, model: str, taxonomy: str) -> str:
    return f"{method_label}/{model}/{taxonomy}"


def parse_example_frame_range(example_id: str) -> list[int]:
    match = re.search(r"__[cf]_(\d+)_(\d+)$", example_id)
    if not match:
        raise ValueError(f"Could not parse frame range from example_id: {example_id}")
    return [int(match.group(1)), int(match.group(2))]


def _audit_section_record(record: dict, section: dict, clip_level: str) -> dict:
    section_record = dict(record)
    section_record["coarse"] = section
    section_record["example_id"] = section.get("fine_id") or record.get("example_id")
    section_record["clip_level"] = clip_level
    return section_record


def load_audit_v11_action_records(actions_dir: Path, clip_level: str) -> List[dict]:
    raw_records = load_audit_records(actions_dir)
    records: List[dict] = []
    include_coarse = clip_level in {"coarse", "both"}
    include_fine = clip_level in {"fine", "both"}

    for raw_record in raw_records:
        if include_coarse and raw_record.get("coarse"):
            simple = convert_record_to_simple_actions(_audit_section_record(raw_record, raw_record["coarse"], "coarse"))
            simple["clip_level"] = "coarse"
            records.append(simple)
        if include_fine:
            for fine in raw_record.get("fine") or []:
                simple = convert_record_to_simple_actions(_audit_section_record(raw_record, fine, "fine"))
                simple["clip_level"] = "fine"
                records.append(simple)

    records.sort(
        key=lambda record: (
            record.get("video_id") or "",
            tuple(record.get("frame_range") or [0, 0]),
            record.get("criterion") or "",
            record.get("clip_level") or "",
            record.get("example_id") or "",
        )
    )
    return records


def resolve_synthetic_source(args: argparse.Namespace) -> Path:
    preset = DEFAULT_SYNTHETIC_PRESETS.get(args.synthetic_preset)
    if not preset and not args.synthetic_records_jsonl:
        raise ValueError(
            f"Unknown synthetic preset '{args.synthetic_preset}'. "
            f"Known presets: {sorted(DEFAULT_SYNTHETIC_PRESETS)}"
        )
    records_jsonl = args.synthetic_records_jsonl or preset["records_jsonl"]
    return ROOT_DIR / records_jsonl


def load_public_action_records(records_jsonl_path: Path) -> List[dict]:
    records: List[dict] = []
    with records_jsonl_path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {records_jsonl_path}:{line_num}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                continue
            record = {
                "example_id": row.get("example_id"),
                "video_id": row.get("video_id"),
                "criterion": row.get("criterion"),
                "frame_range": row.get("frame_range") or [row.get("frame_start"), row.get("frame_end")],
                "mind_change": row.get("mind_change"),
                "left": row.get("left_actions", []),
                "right": row.get("right_actions", []),
                "camera": row.get("camera_actions", []),
                "other": row.get("other_actions", []),
            }
            records.append(record)
    records.sort(
        key=lambda record: (
            record.get("video_id") or "",
            tuple(record.get("frame_range") or [0, 0]),
            record.get("criterion") or "",
            record.get("example_id") or "",
        )
    )
    return records


def load_synthetic_action_records(records_jsonl_path: Path) -> List[dict]:
    records: List[dict] = []
    for record in load_public_action_records(records_jsonl_path):
        if (
            (not record.get("frame_range") or any(value is None for value in record.get("frame_range", [])))
            and record.get("example_id")
        ):
            record["frame_range"] = parse_example_frame_range(str(record["example_id"]))
        records.append(record)
    return records


def add_synthetic_fine_records_from_coarse(records: Sequence[dict]) -> List[dict]:
    """Treat each synthetic coarse clip's start frame as a fine-start label."""
    out: List[dict] = []
    for record in records:
        coarse_record = dict(record)
        coarse_record["clip_level"] = "coarse"
        out.append(coarse_record)

        fine_record = dict(record)
        fine_record["clip_level"] = "fine"
        fine_record["example_id"] = f"{record.get('example_id') or record.get('video_id')}__synthetic_fine_start"
        out.append(fine_record)
    return out


def evaluate_action_method_frames(pred_frames_by_video: Dict[str, List[dict]], gt_records: List[dict]) -> List[dict]:
    by_metric_actor_granularity: Dict[tuple[str, str, str], List[float]] = defaultdict(list)
    details: Dict[tuple[str, str, str], List[dict]] = defaultdict(list)

    gt_by_video = gt_records_by_video(gt_records)
    for video_id, records in sorted(gt_by_video.items()):
        pred_frames = pred_frames_by_video.get(video_id)
        if not pred_frames:
            continue
        pred_frames_sorted = sorted(pred_frames, key=prediction_frame_number)
        available = [prediction_frame_number(frame) for frame in pred_frames_sorted]
        by_frame_num = {prediction_frame_number(frame): frame for frame in pred_frames_sorted}

        for record in records:
            clip_start, clip_end = [int(value) for value in record["frame_range"]]
            clip_level = record.get("clip_level", "coarse")
            onset_frame = choose_start_frame(clip_start, available)
            if clip_level == "fine":
                frame_groups = {
                    "start_keyframe": [onset_frame] if onset_frame is not None else [],
                }
            else:
                onset_pm1_frames = start_frame_pm1_keyframes(clip_start, available)
                clip_frames = keyframes_in_clip(clip_start, clip_end, available)
                frame_groups = {
                    "start_keyframe": [onset_frame] if onset_frame is not None else [],
                    "start_keyframe_pm1": list(onset_pm1_frames),
                    "clip_keyframes": list(clip_frames),
                }

            for actor in EVAL_ACTORS:
                granularities = ["exact"] if actor == "other" else list(GRANULARITIES)
                for granularity in granularities:
                    for metric_name, frame_nums in frame_groups.items():
                        if not frame_nums:
                            continue
                        if metric_name == "start_keyframe":
                            pred_row = pred_actor_row(by_frame_num[frame_nums[0]], actor)
                            gt_row = gt_actor_row(record, actor, clip_start)
                            score = frame_match_f1(pred_row, gt_row, actor, granularity)
                            detail = {
                                "video_id": video_id,
                                "example_id": record.get("example_id"),
                                "clip_level": clip_level,
                                "frame": frame_nums[0],
                                "score": score,
                            }
                        elif metric_name == "start_keyframe_pm1":
                            gt_row = gt_actor_row(record, actor, clip_start)
                            frame_scores = [
                                frame_match_f1(pred_actor_row(by_frame_num[frame_num], actor), gt_row, actor, granularity)
                                for frame_num in frame_nums
                            ]
                            score = max(frame_scores) if frame_scores else 0.0
                            detail = {
                                "video_id": video_id,
                                "example_id": record.get("example_id"),
                                "clip_level": clip_level,
                                "frames": list(frame_nums),
                                "score": score,
                            }
                        else:
                            scores = []
                            for frame_num in frame_nums:
                                pred_row = pred_actor_row(by_frame_num[frame_num], actor)
                                gt_row = gt_actor_row(record, actor, frame_num)
                                scores.append(frame_match_f1(pred_row, gt_row, actor, granularity))
                            score = mean(scores)
                            detail = {
                                "video_id": video_id,
                                "example_id": record.get("example_id"),
                                "clip_level": clip_level,
                                "n_keyframes": len(frame_nums),
                                "score": score,
                            }
                        key = (metric_name, actor, granularity)
                        by_metric_actor_granularity[key].append(score)
                        details[key].append(detail)

    rows = []
    for (metric_name, actor, granularity), scores in sorted(by_metric_actor_granularity.items()):
        rows.append(
            {
                "metric": metric_name,
                "actor": actor,
                "granularity": granularity,
                "f1_mean": mean(scores) if scores else None,
                "f1_std": pstdev(scores) if len(scores) > 1 else 0.0,
                "n_examples": len(scores),
                "details": details[(metric_name, actor, granularity)],
            }
        )
    return rows


def collect_prediction_groups(
    pred_roots: Sequence[Path],
    gt_by_video: Dict[str, List[dict]],
    include_partial: bool,
    modified_since: float | None,
    model_filter: set[str] | None,
    taxonomy_filter: set[str] | None,
    verbose: bool,
) -> tuple[Dict[str, Dict[str, List[dict]]], Dict[str, Dict[str, str]], List[dict]]:
    frames_by_method: Dict[str, Dict[str, List[dict]]] = defaultdict(dict)
    paths_by_method: Dict[str, Dict[str, str]] = defaultdict(dict)
    skipped: List[dict] = []

    for pred_root in pred_roots:
        for method_label, pred_path_str in collect_pred_files_recursive(str(pred_root)):
            pred_path = Path(pred_path_str)
            if modified_since is not None and pred_path.stat().st_mtime < modified_since:
                continue

            health = inspect_prediction_file(pred_path)
            if health["bad_json_lines"] and not include_partial:
                skipped.append(
                    {
                        "pred_path": str(pred_path),
                        "reason": "malformed_jsonl",
                        "bad_json_lines": health["bad_json_lines"],
                    }
                )
                continue

            pred_frames = load_predictions(str(pred_path))
            if not pred_frames:
                skipped.append({"pred_path": str(pred_path), "reason": "empty_or_unreadable"})
                continue

            video_id = extract_video_id_from_pred(pred_frames)
            if not video_id:
                skipped.append({"pred_path": str(pred_path), "reason": "missing_video_id"})
                continue
            video_id = str(video_id)

            gt_records = gt_by_video.get(video_id)
            if not gt_records:
                skipped.append({"pred_path": str(pred_path), "video_id": video_id, "reason": "missing_gt"})
                continue

            if not include_partial and not is_complete_prediction(pred_frames, gt_records):
                skipped.append({"pred_path": str(pred_path), "video_id": video_id, "reason": "partial_prediction"})
                continue

            model = extract_model_from_pred(pred_frames, str(pred_path))
            taxonomy = extract_taxonomy_from_pred(pred_frames, str(pred_path))
            if model_filter and model not in model_filter:
                continue
            if taxonomy_filter and taxonomy not in taxonomy_filter:
                continue

            key = method_key(method_label, model, taxonomy)
            frames_by_method[key][video_id] = pred_frames
            paths_by_method[key][video_id] = str(pred_path)
            if verbose:
                print(f"EVAL {pred_path} -> {key}")

    return frames_by_method, paths_by_method, skipped


def aggregate_cvs_rows(rows: Sequence[dict]) -> dict:
    summary: dict = {"n_cvs_videos": len(rows)}
    if not rows:
        return summary

    frame_maps = [row["frame_mAP"] for row in rows if row.get("frame_mAP") is not None]
    video_maps = [row["video_mAP"] for row in rows if row.get("video_mAP") is not None]
    if frame_maps:
        summary["frame_mAP"] = sum(frame_maps) / len(frame_maps)
    if video_maps:
        summary["video_mAP"] = sum(video_maps) / len(video_maps)

    frame_ap = {}
    video_ap = {}
    for criterion in CRITERIA:
        frame_vals = [row["frame_ap"].get(criterion) for row in rows if row.get("frame_ap", {}).get(criterion) is not None]
        video_vals = [row["video_ap"].get(criterion) for row in rows if row.get("video_ap", {}).get(criterion) is not None]
        frame_ap[criterion] = (sum(frame_vals) / len(frame_vals)) if frame_vals else None
        video_ap[criterion] = (sum(video_vals) / len(video_vals)) if video_vals else None
    summary["frame_ap"] = frame_ap
    summary["video_ap"] = video_ap
    return summary


def aggregate_recommendation_headline(
    rows: Sequence[dict],
    include_actors: set[str] | None = None,
) -> dict[tuple[str, str], float]:
    by_metric_granularity: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        actor = row.get("actor")
        if actor == "other":
            continue
        if include_actors is not None and actor not in include_actors:
            continue
        metric = str(row.get("metric"))
        granularity = str(row.get("granularity"))
        value = row.get("f1_mean")
        if value is None:
            continue
        by_metric_granularity[(metric, granularity)].append(float(value))
    return {
        key: (sum(values) / len(values))
        for key, values in by_metric_granularity.items()
        if values
    }


def aggregate_recommendation_headline_by_clip_level(
    rows: Sequence[dict],
    clip_level: str,
    include_actors: set[str] | None = None,
) -> dict[tuple[str, str], float]:
    by_metric_granularity: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        actor = row.get("actor")
        if actor == "other":
            continue
        if include_actors is not None and actor not in include_actors:
            continue
        scores = [
            float(detail["score"])
            for detail in row.get("details", [])
            if detail.get("clip_level") == clip_level and detail.get("score") is not None
        ]
        if not scores:
            continue
        metric = str(row.get("metric"))
        granularity = str(row.get("granularity"))
        by_metric_granularity[(metric, granularity)].append(sum(scores) / len(scores))
    return {
        key: (sum(values) / len(values))
        for key, values in by_metric_granularity.items()
        if values
    }


def build_combined_summary_rows(
    cvs_results: Dict[str, List[dict]],
    audit_recommendation_results: Dict[str, List[dict]],
    synthetic_recommendation_results: Dict[str, List[dict]],
) -> List[dict]:
    rows: List[dict] = []
    for method, method_rows in sorted(cvs_results.items()):
        summary = aggregate_cvs_rows(method_rows)
        rows.append(
            {
                "method": method,
                "category": "cvs",
                "metric": "frame_mAP",
                "actor": "",
                "granularity": "",
                "value": summary.get("frame_mAP"),
                "std": None,
                "n_examples": summary.get("n_cvs_videos", 0),
            }
        )
        rows.append(
            {
                "method": method,
                "category": "cvs",
                "metric": "video_mAP",
                "actor": "",
                "granularity": "",
                "value": summary.get("video_mAP"),
                "std": None,
                "n_examples": summary.get("n_cvs_videos", 0),
            }
        )
        for criterion in CRITERIA:
            rows.append(
                {
                    "method": method,
                    "category": "cvs",
                    "metric": f"frame_ap_{criterion}",
                    "actor": "",
                    "granularity": "",
                    "value": summary.get("frame_ap", {}).get(criterion),
                    "std": None,
                    "n_examples": summary.get("n_cvs_videos", 0),
                }
            )
            rows.append(
                {
                    "method": method,
                    "category": "cvs",
                    "metric": f"video_ap_{criterion}",
                    "actor": "",
                    "granularity": "",
                    "value": summary.get("video_ap", {}).get(criterion),
                    "std": None,
                    "n_examples": summary.get("n_cvs_videos", 0),
                }
            )

    for row in build_recommendation_summary_rows(audit_recommendation_results):
        rows.append(
            {
                "method": row["method"],
                "category": "recommendation_audit_v11",
                "metric": row["metric"],
                "actor": row["actor"],
                "granularity": row["granularity"],
                "value": row["f1_mean"],
                "std": row["f1_std"],
                "n_examples": row["n_examples"],
            }
        )

    for row in build_recommendation_summary_rows(synthetic_recommendation_results):
        rows.append(
            {
                "method": row["method"],
                "category": "recommendation_synthetic",
                "metric": row["metric"],
                "actor": row["actor"],
                "granularity": row["granularity"],
                "value": row["f1_mean"],
                "std": row["f1_std"],
                "n_examples": row["n_examples"],
            }
        )

    return rows


RIGHT_HAND_COMPONENTS = {
    "tool_type": ("tool_type",),
    "action_code": ("action_code",),
    "target_structure": ("target_structure",),
    "target_context_1": ("target_context_1",),
    "target_context_2": ("target_context_2",),
    "action_target": ("action_code", "target_structure"),
    "tool_action_target": ("tool_type", "action_code", "target_structure"),
    "exact_tuple": (
        "tool_type",
        "action_code",
        "target_structure",
        "target_context_1",
        "target_context_2",
    ),
}


def _right_component_label(row: dict, fields: Sequence[str]) -> tuple:
    return tuple(str(row.get(field, "(not set)")) for field in fields)


def _right_component_match(pred_row: dict, gt_row: dict, fields: Sequence[str]) -> float:
    return 1.0 if _right_component_label(pred_row, fields) == _right_component_label(gt_row, fields) else 0.0


def evaluate_right_hand_breakdown(
    pred_frames_by_video: Dict[str, List[dict]],
    gt_records: List[dict],
) -> list[dict]:
    """Evaluate right-instrument matches by individual fields and combinations."""
    scores_by_key: Dict[tuple[str, str], List[float]] = defaultdict(list)
    details_by_key: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    confusions_by_key: Dict[tuple[str, str], Dict[tuple[tuple, tuple], int]] = defaultdict(lambda: defaultdict(int))

    gt_by_video = gt_records_by_video(gt_records)
    for video_id, records in sorted(gt_by_video.items()):
        pred_frames = pred_frames_by_video.get(video_id)
        if not pred_frames:
            continue
        pred_frames_sorted = sorted(pred_frames, key=prediction_frame_number)
        available = [prediction_frame_number(frame) for frame in pred_frames_sorted]
        by_frame_num = {prediction_frame_number(frame): frame for frame in pred_frames_sorted}

        for record in records:
            clip_start, clip_end = [int(value) for value in record["frame_range"]]
            start_frame = choose_start_frame(clip_start, available)
            if record.get("clip_level") == "fine":
                frame_groups = {
                    "start_keyframe": [start_frame] if start_frame is not None else [],
                }
            else:
                start_pm1_frames = start_frame_pm1_keyframes(clip_start, available)
                clip_frames = keyframes_in_clip(clip_start, clip_end, available)
                frame_groups = {
                    "start_keyframe": [start_frame] if start_frame is not None else [],
                    "start_keyframe_pm1": list(start_pm1_frames),
                    "clip_keyframes": list(clip_frames),
                }

            for metric_name, frame_nums in frame_groups.items():
                if not frame_nums:
                    continue
                gt_frame = clip_start
                gt_row = gt_actor_row(record, "right", gt_frame)
                for component, fields in RIGHT_HAND_COMPONENTS.items():
                    frame_scores = []
                    for frame_num in frame_nums:
                        pred_row = pred_actor_row(by_frame_num[frame_num], "right")
                        score = _right_component_match(pred_row, gt_row, fields)
                        frame_scores.append(score)
                        if metric_name != "start_keyframe_pm1" or score == max(frame_scores):
                            pred_label = _right_component_label(pred_row, fields)
                            gt_label = _right_component_label(gt_row, fields)
                            confusions_by_key[(metric_name, component)][(gt_label, pred_label)] += 1
                    if metric_name == "start_keyframe_pm1":
                        score = max(frame_scores) if frame_scores else 0.0
                    else:
                        score = sum(frame_scores) / len(frame_scores)
                    key = (metric_name, component)
                    scores_by_key[key].append(score)
                    details_by_key[key].append(
                        {
                            "video_id": video_id,
                            "example_id": record.get("example_id"),
                            "clip_level": record.get("clip_level", "coarse"),
                            "frames": list(frame_nums),
                            "score": score,
                            "gt": list(_right_component_label(gt_row, fields)),
                        }
                    )

    rows = []
    for (metric_name, component), scores in sorted(scores_by_key.items()):
        confusions = sorted(
            (
                {"gt": list(gt), "pred": list(pred), "count": count}
                for (gt, pred), count in confusions_by_key[(metric_name, component)].items()
                if gt != pred
            ),
            key=lambda item: (-item["count"], item["gt"], item["pred"]),
        )
        rows.append(
            {
                "metric": metric_name,
                "component": component,
                "match_mean": (sum(scores) / len(scores)) if scores else None,
                "n_examples": len(scores),
                "top_confusions": confusions[:12],
                "details": details_by_key[(metric_name, component)],
            }
        )
    return rows


def build_right_hand_breakdown_rows(results: Dict[str, list[dict]]) -> list[dict]:
    rows = []
    for method, method_rows in sorted(results.items()):
        for row in method_rows:
            rows.append(
                {
                    "method": method,
                    "metric": row["metric"],
                    "component": row["component"],
                    "match_mean": row["match_mean"],
                    "n_examples": row["n_examples"],
                }
            )
    return rows


def write_right_hand_breakdown_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "metric", "component", "match_mean", "n_examples"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "category", "metric", "actor", "granularity", "value", "std", "n_examples"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_compact_summary(
    cvs_results: Dict[str, List[dict]],
    audit_recommendation_results: Dict[str, List[dict]],
    synthetic_recommendation_results: Dict[str, List[dict]],
) -> dict:
    methods = sorted(set(cvs_results) | set(audit_recommendation_results) | set(synthetic_recommendation_results))
    summary: dict = {"methods": {}}
    for method in methods:
        audit_headline = aggregate_recommendation_headline(
            audit_recommendation_results.get(method, [])
        )
        audit_fine_start_headline = {
            key: value
            for key, value in aggregate_recommendation_headline_by_clip_level(
                audit_recommendation_results.get(method, []),
                clip_level="fine",
            ).items()
            if key[0] == "start_keyframe"
        }
        synthetic_headline = aggregate_recommendation_headline(
            synthetic_recommendation_results.get(method, [])
        )
        synthetic_fine_start_headline = {
            key: value
            for key, value in aggregate_recommendation_headline_by_clip_level(
                synthetic_recommendation_results.get(method, []),
                clip_level="fine",
            ).items()
            if key[0] == "start_keyframe"
        }
        summary["methods"][method] = {
            "cvs": aggregate_cvs_rows(cvs_results.get(method, [])),
            "recommendation_audit_v11_headline": {
                f"{key[0]}/{key[1]}": value for key, value in audit_headline.items()
            },
            "recommendation_audit_v11_fine_start_headline": {
                f"{key[0]}/{key[1]}": value
                for key, value in audit_fine_start_headline.items()
            },
            "recommendation_synthetic_headline": {
                f"{key[0]}/{key[1]}": value for key, value in synthetic_headline.items()
            },
            "recommendation_synthetic_fine_start_headline": {
                f"{key[0]}/{key[1]}": value
                for key, value in synthetic_fine_start_headline.items()
            },
        }

    if len(methods) == 2:
        lhs, rhs = methods
        lhs_data = summary["methods"][lhs]
        rhs_data = summary["methods"][rhs]
        deltas = {
            "cvs": {},
            "recommendation_audit_v11_headline": {},
            "recommendation_audit_v11_fine_start_headline": {},
            "recommendation_synthetic_headline": {},
            "recommendation_synthetic_fine_start_headline": {},
        }
        for metric in ("frame_mAP", "video_mAP"):
            lval = lhs_data["cvs"].get(metric)
            rval = rhs_data["cvs"].get(metric)
            if lval is not None and rval is not None:
                deltas["cvs"][metric] = rval - lval
        for section in (
            "recommendation_audit_v11_headline",
            "recommendation_audit_v11_fine_start_headline",
            "recommendation_synthetic_headline",
            "recommendation_synthetic_fine_start_headline",
        ):
            keys = sorted(set(lhs_data[section]) | set(rhs_data[section]))
            for key in keys:
                lval = lhs_data[section].get(key)
                rval = rhs_data[section].get(key)
                if lval is not None and rval is not None:
                    deltas[section][key] = rval - lval
        summary["comparison"] = {
            "baseline_method": lhs,
            "compare_method": rhs,
            "delta_compare_minus_baseline": deltas,
        }
    return summary


def render_compact_summary(compact_summary: dict) -> str:
    methods = list(compact_summary.get("methods", {}).keys())
    lines = ["## Compact Summary"]
    lines.append("")

    def _fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "-"

    def _append_section(
        title: str,
        lhs_name: str,
        rhs_name: str,
        rows: list[tuple[str, float | None, float | None]],
    ) -> None:
        if not rows:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"| Metric | {lhs_name} | {rhs_name} | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for metric, lhs, rhs in rows:
            delta = None if lhs is None or rhs is None else rhs - lhs
            lines.append(f"| {metric} | {_fmt(lhs)} | {_fmt(rhs)} | {_fmt(delta)} |")
        lines.append("")

    comparison = compact_summary.get("comparison")
    if comparison and len(methods) == 2:
        lhs_name = comparison["baseline_method"]
        rhs_name = comparison["compare_method"]
        lhs_data = compact_summary["methods"][lhs_name]
        rhs_data = compact_summary["methods"][rhs_name]

        cvs_rows = [
            ("frame_mAP", lhs_data["cvs"].get("frame_mAP"), rhs_data["cvs"].get("frame_mAP")),
            ("video_mAP", lhs_data["cvs"].get("video_mAP"), rhs_data["cvs"].get("video_mAP")),
            ("frame_ap/c1", lhs_data["cvs"].get("frame_ap", {}).get("c1"), rhs_data["cvs"].get("frame_ap", {}).get("c1")),
            ("frame_ap/c2", lhs_data["cvs"].get("frame_ap", {}).get("c2"), rhs_data["cvs"].get("frame_ap", {}).get("c2")),
            ("frame_ap/c3", lhs_data["cvs"].get("frame_ap", {}).get("c3"), rhs_data["cvs"].get("frame_ap", {}).get("c3")),
            ("video_ap/c1", lhs_data["cvs"].get("video_ap", {}).get("c1"), rhs_data["cvs"].get("video_ap", {}).get("c1")),
            ("video_ap/c2", lhs_data["cvs"].get("video_ap", {}).get("c2"), rhs_data["cvs"].get("video_ap", {}).get("c2")),
            ("video_ap/c3", lhs_data["cvs"].get("video_ap", {}).get("c3"), rhs_data["cvs"].get("video_ap", {}).get("c3")),
        ]
        _append_section("CVS", lhs_name, rhs_name, cvs_rows)

        for title, section_key in (
            ("Recommendation vs audit_v11 GT", "recommendation_audit_v11_headline"),
            (
                "Fine-start recommendation vs audit_v11 GT",
                "recommendation_audit_v11_fine_start_headline",
            ),
            ("Recommendation vs synthetic GT", "recommendation_synthetic_headline"),
            (
                "Fine-start recommendation vs synthetic GT",
                "recommendation_synthetic_fine_start_headline",
            ),
        ):
            metric_order = [
                "start_keyframe/exact",
                "start_keyframe/medium",
                "start_keyframe/coarse",
                "clip_keyframes/exact",
                "clip_keyframes/medium",
                "clip_keyframes/coarse",
            ]
            rows = [
                (
                    metric,
                    lhs_data[section_key].get(metric),
                    rhs_data[section_key].get(metric),
                )
                for metric in metric_order
            ]
            _append_section(title, lhs_name, rhs_name, rows)
    else:
        lines.append("| Method | Frame mAP | Video mAP |")
        lines.append("| --- | ---: | ---: |")
        for method in methods:
            data = compact_summary["methods"][method]
            lines.append(
                f"| {method} | {_fmt(data['cvs'].get('frame_mAP'))} | {_fmt(data['cvs'].get('video_mAP'))} |"
            )
    return "\n".join(lines) + "\n"


def _print_recommendation_section(
    methods: Sequence[str],
    width: int,
    title: str,
    recommendation_results: Dict[str, List[dict]],
) -> None:
    if not any(recommendation_results.get(method) for method in methods):
        return
    print(f"\n  {title}")
    print(
        f"  {'Method':<{width}s} {'Metric':<16s} {'Actor':<8s} {'Granularity':<11s} "
        f"{'Mean':>8s} {'Std':>8s} {'n':>4s}"
    )
    print(
        f"  {'-'*(width-2)} {'-'*16} {'-'*8} {'-'*11} {'-'*8} {'-'*8} {'-'*4}"
    )
    for method in methods:
        for row_data in recommendation_results.get(method, []):
            print(
                f"  {method:<{width}s} {row_data['metric']:<16s} {row_data['actor']:<8s} "
                f"{row_data['granularity']:<11s} {row_data['f1_mean']:8.3f} "
                f"{row_data['f1_std']:8.3f} {row_data['n_examples']:4d}"
            )


def _short_headline_method_name(method: str, include_taxonomy: bool = False) -> str:
    parts = method.split("/")
    model = parts[-2] if len(parts) >= 2 else method
    taxonomy = parts[-1] if len(parts) >= 1 else ""
    if model.startswith("claude-"):
        model_name = "claude"
    elif model.startswith("gemini-"):
        model_name = "gemini"
    elif model.startswith("gpt-"):
        model_name = "gpt"
    else:
        model_name = model

    is_surgent = method.startswith("surgent_") or any(
        part.startswith("pref-") for part in parts
    )
    method_name = "SurGent" if is_surgent else "Baseline"

    if "no_cvs_no_desc" in method:
        variant = "no-cvs-no-desc"
    elif "no_cvs_no_guideline" in method:
        variant = "no-cvs-no-guideline"
    elif "no_cvs" in method:
        variant = "no-cvs"
    elif "arecrules-conservative-visible" in method:
        variant = "conservative-visible"
    else:
        variant = "default"

    label = f"{method_name} {variant} {model_name}"
    if include_taxonomy and taxonomy:
        label = f"{label} {taxonomy}"
    return label


def _headline_methods(methods: Sequence[str]) -> list[str]:
    current_methods = [
        method
        for method in methods
        if method.endswith("/cvs_act_current_simple_v1")
    ]
    return sorted(current_methods or methods)


def _print_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    header_line = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join(
        "---" if index == 0 else "---:"
        for index, _ in enumerate(headers)
    ) + " |"
    print(header_line)
    print(separator)
    for row in rows:
        print("| " + " | ".join(
            cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])
            for index, cell in enumerate(row)
        ) + " |")


def print_headline_tables(
    cvs_results: Dict[str, List[dict]],
    audit_recommendation_results: Dict[str, List[dict]],
) -> None:
    methods = _headline_methods(sorted(set(cvs_results) | set(audit_recommendation_results)))
    if not methods:
        return
    include_taxonomy = len({method.split("/")[-1] for method in methods}) > 1
    title_suffix = ""
    if methods and all(method.endswith("/cvs_act_current_simple_v1") for method in methods):
        title_suffix = " (cvs_act_current_simple_v1)"

    cvs_rows = []
    for method in methods:
        summary = aggregate_cvs_rows(cvs_results.get(method, []))
        if "frame_mAP" not in summary:
            continue
        cvs_rows.append(
            [
                _short_headline_method_name(method, include_taxonomy=include_taxonomy),
                f"{summary['frame_mAP']:.3f}",
                f"{summary['video_mAP']:.3f}",
            ]
        )

    def _build_action_rows(include_actors: set[str] | None = None) -> list[list[str]]:
        rows = []
        for method in methods:
            headline = aggregate_recommendation_headline(
                audit_recommendation_results.get(method, []),
                include_actors=include_actors,
            )
            if not headline:
                continue
            rows.append(
                [
                    _short_headline_method_name(method, include_taxonomy=include_taxonomy),
                    f"{headline.get(('clip_keyframes', 'coarse'), 0.0):.3f}",
                    f"{headline.get(('clip_keyframes', 'medium'), 0.0):.3f}",
                    f"{headline.get(('clip_keyframes', 'exact'), 0.0):.3f}",
                    f"{headline.get(('start_keyframe', 'coarse'), 0.0):.3f}",
                    f"{headline.get(('start_keyframe', 'medium'), 0.0):.3f}",
                    f"{headline.get(('start_keyframe', 'exact'), 0.0):.3f}",
                ]
            )
        return rows

    action_rows = _build_action_rows()
    action_left_right_rows = _build_action_rows(include_actors={"left", "right"})
    action_actor_rows = {
        actor: _build_action_rows(include_actors={actor})
        for actor in ("left", "right", "camera")
    }

    def _build_pm1_rows(include_actors: set[str] | None = None) -> list[list[str]]:
        rows = []
        for method in methods:
            headline = aggregate_recommendation_headline(
                audit_recommendation_results.get(method, []),
                include_actors=include_actors,
            )
            if not headline:
                continue
            rows.append(
                [
                    _short_headline_method_name(method, include_taxonomy=include_taxonomy),
                    f"{headline.get(('start_keyframe_pm1', 'coarse'), 0.0):.3f}",
                    f"{headline.get(('start_keyframe_pm1', 'medium'), 0.0):.3f}",
                    f"{headline.get(('start_keyframe_pm1', 'exact'), 0.0):.3f}",
                ]
            )
        return rows

    def _build_fine_start_rows(include_actors: set[str] | None = None) -> list[list[str]]:
        rows = []
        for method in methods:
            headline = aggregate_recommendation_headline_by_clip_level(
                audit_recommendation_results.get(method, []),
                clip_level="fine",
                include_actors=include_actors,
            )
            if not headline:
                continue
            rows.append(
                [
                    _short_headline_method_name(method, include_taxonomy=include_taxonomy),
                    f"{headline.get(('start_keyframe', 'coarse'), 0.0):.3f}",
                    f"{headline.get(('start_keyframe', 'medium'), 0.0):.3f}",
                    f"{headline.get(('start_keyframe', 'exact'), 0.0):.3f}",
                ]
            )
        return rows

    start_pm1_rows = _build_pm1_rows()
    start_pm1_left_right_rows = _build_pm1_rows(include_actors={"left", "right"})
    start_pm1_actor_rows = {
        actor: _build_pm1_rows(include_actors={actor})
        for actor in ("left", "right", "camera")
    }
    fine_start_rows = _build_fine_start_rows()
    fine_start_left_right_rows = _build_fine_start_rows(include_actors={"left", "right"})

    if (
        cvs_rows
        or action_rows
        or action_left_right_rows
        or any(action_actor_rows.values())
        or start_pm1_rows
        or start_pm1_left_right_rows
        or any(start_pm1_actor_rows.values())
        or fine_start_rows
        or fine_start_left_right_rows
    ):
        print()
        print(f"HEADLINE TABLES{title_suffix}")
        print("=" * 80)
    if cvs_rows:
        print()
        print("CVS")
        _print_markdown_table(["Method", "Frame mAP", "Video mAP"], cvs_rows)
    if action_rows:
        print()
        print("Action F1 vs audit_v11 GT")
        _print_markdown_table(
            [
                "Method",
                "Clip coarse",
                "Clip medium",
                "Clip exact",
                "Start coarse",
                "Start medium",
                "Start exact",
            ],
            action_rows,
        )
    if fine_start_rows:
        print()
        print("Fine Start Action F1 vs audit_v11 GT")
        _print_markdown_table(
            ["Method", "Fine start coarse", "Fine start medium", "Fine start exact"],
            fine_start_rows,
        )
    if start_pm1_rows:
        print()
        print("Start Action F1 vs audit_v11 GT (+/-1 keyframe)")
        _print_markdown_table(
            ["Method", "Start +/-1 coarse", "Start +/-1 medium", "Start +/-1 exact"],
            start_pm1_rows,
        )
    if action_left_right_rows:
        print()
        print("Action F1 vs audit_v11 GT (left/right only)")
        _print_markdown_table(
            [
                "Method",
                "Clip coarse",
                "Clip medium",
                "Clip exact",
                "Start coarse",
                "Start medium",
                "Start exact",
            ],
            action_left_right_rows,
        )
    if fine_start_left_right_rows:
        print()
        print("Fine Start Action F1 vs audit_v11 GT (left/right only)")
        _print_markdown_table(
            ["Method", "Fine start coarse", "Fine start medium", "Fine start exact"],
            fine_start_left_right_rows,
        )
    for actor in ("left", "right", "camera"):
        rows = action_actor_rows[actor]
        if not rows:
            continue
        print()
        print(f"Action F1 vs audit_v11 GT ({actor} only)")
        _print_markdown_table(
            [
                "Method",
                "Clip coarse",
                "Clip medium",
                "Clip exact",
                "Start coarse",
                "Start medium",
                "Start exact",
            ],
            rows,
        )
    if start_pm1_left_right_rows:
        print()
        print("Start Action F1 vs audit_v11 GT (+/-1 keyframe, left/right only)")
        _print_markdown_table(
            ["Method", "Start +/-1 coarse", "Start +/-1 medium", "Start +/-1 exact"],
            start_pm1_left_right_rows,
        )
    for actor in ("left", "right", "camera"):
        rows = start_pm1_actor_rows[actor]
        if not rows:
            continue
        print()
        print(f"Start Action F1 vs audit_v11 GT (+/-1 keyframe, {actor} only)")
        _print_markdown_table(
            ["Method", "Start +/-1 coarse", "Start +/-1 medium", "Start +/-1 exact"],
            rows,
        )


def print_right_hand_breakdown(right_hand_results: Dict[str, list[dict]]) -> None:
    methods = _headline_methods(sorted(right_hand_results))
    if not methods:
        return

    wanted = [
        ("clip_keyframes", "action_code", "Clip action"),
        ("clip_keyframes", "target_structure", "Clip target"),
        ("clip_keyframes", "action_target", "Clip action+target"),
        ("clip_keyframes", "tool_action_target", "Clip tool+action+target"),
        ("clip_keyframes", "exact_tuple", "Clip exact tuple"),
        ("start_keyframe", "action_target", "Start action+target"),
        ("start_keyframe_pm1", "action_target", "Start +/-1 action+target"),
    ]
    rows_by_method = {}
    for method in methods:
        rows_by_method[method] = {
            (row["metric"], row["component"]): row.get("match_mean")
            for row in right_hand_results.get(method, [])
        }

    table_rows = []
    include_taxonomy = len({method.split("/")[-1] for method in methods}) > 1
    for method in methods:
        values = rows_by_method[method]
        row = [_short_headline_method_name(method, include_taxonomy=include_taxonomy)]
        for metric, component, _label in wanted:
            value = values.get((metric, component))
            row.append(f"{value:.3f}" if value is not None else "-")
        table_rows.append(row)

    print()
    print("Right-Hand Fine-Grained Action Breakdown vs audit_v11 GT")
    _print_markdown_table(
        ["Method"] + [label for _metric, _component, label in wanted],
        table_rows,
    )


def print_summary(
    cvs_results: Dict[str, List[dict]],
    audit_recommendation_results: Dict[str, List[dict]],
    synthetic_recommendation_results: Dict[str, List[dict]],
) -> None:
    methods = sorted(set(cvs_results) | set(audit_recommendation_results) | set(synthetic_recommendation_results))
    if not methods:
        print("No results to display.")
        return

    width = max(len(method) for method in methods) + 2

    print()
    print("=" * 160)
    print("AGGREGATED RESULTS")
    print("=" * 160)

    has_cvs = any(cvs_results.get(method) for method in methods)
    if has_cvs:
        print("\n  CVS Scores (averaged across videos)")
        print(
            f"  {'Method':<{width}s} {'Frame mAP':>10s} {'F.c1':>6s} {'F.c2':>6s} {'F.c3':>6s}"
            f"  {'Video mAP':>10s} {'V.c1':>6s} {'V.c2':>6s} {'V.c3':>6s} {'n':>4s}"
        )
        print(
            f"  {'-'*(width-2)} {'-'*10} {'-'*6} {'-'*6} {'-'*6}"
            f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*4}"
        )
        for method in methods:
            summary = aggregate_cvs_rows(cvs_results.get(method, []))
            if "frame_mAP" not in summary:
                continue
            row = f"  {method:<{width}s} {summary['frame_mAP']:10.3f}"
            for criterion in CRITERIA:
                value = summary["frame_ap"].get(criterion)
                row += f" {value:6.3f}" if value is not None else f" {'N/A':>6s}"
            row += f"  {summary['video_mAP']:10.3f}"
            for criterion in CRITERIA:
                value = summary["video_ap"].get(criterion)
                row += f" {value:6.3f}" if value is not None else f" {'N/A':>6s}"
            row += f" {summary['n_cvs_videos']:4d}"
            print(row)

    _print_recommendation_section(methods, width, "Recommendation Action Metrics vs audit_v11 GT", audit_recommendation_results)
    _print_recommendation_section(methods, width, "Recommendation Action Metrics vs synthetic GT", synthetic_recommendation_results)


def main() -> int:
    args = parse_args()
    pred_roots = [ROOT_DIR / path for path in args.output_dirs]
    metrics_dir = ROOT_DIR / args.metrics_base

    trained_records_jsonl = ROOT_DIR / args.trained_records_jsonl
    audit_actions_dir = ROOT_DIR / args.audit_actions_dir
    if args.action_gt_source == "audit_v11":
        audit_gt_records = load_audit_v11_action_records(audit_actions_dir, args.audit_clip_level)
        action_gt_config = {
            "source": "audit_v11",
            "actions_dir": str(audit_actions_dir),
            "clip_level": args.audit_clip_level,
            "n_records": len(audit_gt_records),
        }
    else:
        audit_gt_records = load_public_action_records(trained_records_jsonl)
        action_gt_config = {
            "source": "hf",
            "trained_records_jsonl": str(trained_records_jsonl),
            "n_records": len(audit_gt_records),
        }
    audit_gt_by_video = gt_records_by_video(audit_gt_records)
    synthetic_config = None
    synthetic_gt_records: List[dict] = []
    if not args.skip_synthetic_action_eval:
        synthetic_records_jsonl = resolve_synthetic_source(args)
        synthetic_gt_records = load_synthetic_action_records(synthetic_records_jsonl)
        synthetic_fine_from_coarse = not args.no_synthetic_fine_from_coarse
        if synthetic_fine_from_coarse:
            synthetic_gt_records = add_synthetic_fine_records_from_coarse(synthetic_gt_records)
        synthetic_config = {
            "preset": args.synthetic_preset,
            "records_jsonl": str(synthetic_records_jsonl),
            "fine_from_coarse": synthetic_fine_from_coarse,
            "n_records": len(synthetic_gt_records),
        }
    modified_since = parse_modified_since(args.modified_since)
    model_filter = set(args.models) if args.models else None
    taxonomy_filter = set(args.taxonomy) if args.taxonomy else None

    frames_by_method, paths_by_method, skipped = collect_prediction_groups(
        pred_roots=pred_roots,
        gt_by_video=audit_gt_by_video,
        include_partial=args.include_partial,
        modified_since=modified_since,
        model_filter=model_filter,
        taxonomy_filter=taxonomy_filter,
        verbose=args.verbose,
    )

    if args.common_only and frames_by_method:
        common_video_ids = set.intersection(*(set(by_video.keys()) for by_video in frames_by_method.values()))
        for method in list(frames_by_method.keys()):
            frames_by_method[method] = {
                video_id: pred_frames
                for video_id, pred_frames in frames_by_method[method].items()
                if video_id in common_video_ids
            }
            paths_by_method[method] = {
                video_id: pred_path
                for video_id, pred_path in paths_by_method[method].items()
                if video_id in common_video_ids
            }
        print(f"Common videos across {len(frames_by_method)} group(s): {len(common_video_ids)}")

    cvs_results: Dict[str, List[dict]] = {}
    audit_recommendation_results: Dict[str, List[dict]] = {}
    synthetic_recommendation_results: Dict[str, List[dict]] = {}
    right_hand_results: Dict[str, List[dict]] = {}

    for method, by_video in sorted(frames_by_method.items()):
        audit_gt_subset = [record for record in audit_gt_records if str(record.get("video_id")) in by_video]
        audit_recommendation_results[method] = evaluate_action_method_frames(by_video, audit_gt_subset)
        right_hand_results[method] = evaluate_right_hand_breakdown(by_video, audit_gt_subset)
        if synthetic_gt_records:
            synthetic_gt_subset = [record for record in synthetic_gt_records if str(record.get("video_id")) in by_video]
            synthetic_recommendation_results[method] = evaluate_action_method_frames(by_video, synthetic_gt_subset)

        cvs_rows: List[dict] = []
        for video_id, pred_frames in sorted(by_video.items()):
            cvs_result = evaluate_cvs(pred_frames, video_id, labels_dir=args.labels_dir, verbose=args.verbose)
            if not cvs_result:
                skipped.append(
                    {
                        "pred_path": paths_by_method[method].get(video_id),
                        "video_id": video_id,
                        "reason": "missing_cvs_labels",
                    }
                )
                continue
            cvs_rows.append(
                {
                    "video_id": video_id,
                    "pred_path": paths_by_method[method].get(video_id),
                    **cvs_result,
                }
            )
        cvs_results[method] = cvs_rows

    summary_rows = build_combined_summary_rows(cvs_results, audit_recommendation_results, synthetic_recommendation_results)
    right_hand_summary_rows = build_right_hand_breakdown_rows(right_hand_results)
    output = {
        "trained_records_jsonl": str(trained_records_jsonl),
        "action_gt_config": action_gt_config,
        "output_dirs": [str(path) for path in pred_roots],
        "labels_dir": args.labels_dir,
        "synthetic_config": synthetic_config,
        "modified_since": args.modified_since,
        "common_only": args.common_only,
        "models": args.models,
        "taxonomy": args.taxonomy,
        "results": {
            method: {
                "cvs": cvs_results.get(method, []),
                "recommendation_audit_v11": audit_recommendation_results.get(method, []),
                "recommendation_synthetic": synthetic_recommendation_results.get(method, []),
                "right_hand_breakdown_audit_v11": right_hand_results.get(method, []),
            }
            for method in sorted(
                set(cvs_results)
                | set(audit_recommendation_results)
                | set(synthetic_recommendation_results)
                | set(right_hand_results)
            )
        },
        "skipped": skipped,
    }
    compact_summary = build_compact_summary(cvs_results, audit_recommendation_results, synthetic_recommendation_results)
    compact_summary_text = render_compact_summary(compact_summary)
    write_json(metrics_dir / "combined_eval_results.json", output)
    write_json(metrics_dir / "compact_summary.json", compact_summary)
    write_json(metrics_dir / "right_hand_breakdown_audit_v11.json", right_hand_results)
    write_summary_csv(metrics_dir / "combined_eval_summary.csv", summary_rows)
    write_right_hand_breakdown_csv(metrics_dir / "right_hand_breakdown_audit_v11.csv", right_hand_summary_rows)
    (metrics_dir / "compact_summary.txt").write_text(compact_summary_text)
    print_summary(cvs_results, audit_recommendation_results, synthetic_recommendation_results)
    print()
    print(compact_summary_text, end="")
    print_headline_tables(cvs_results, audit_recommendation_results)
    print_right_hand_breakdown(right_hand_results)
    print(f"\nWrote {metrics_dir / 'combined_eval_results.json'}")
    print(f"Wrote {metrics_dir / 'combined_eval_summary.csv'}")
    print(f"Wrote {metrics_dir / 'compact_summary.json'}")
    print(f"Wrote {metrics_dir / 'compact_summary.txt'}")
    print(f"Wrote {metrics_dir / 'right_hand_breakdown_audit_v11.json'}")
    print(f"Wrote {metrics_dir / 'right_hand_breakdown_audit_v11.csv'}")
    print(f"Evaluated {len(audit_recommendation_results)} method group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
