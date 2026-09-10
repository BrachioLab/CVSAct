#!/usr/bin/env python3
"""Evaluate audit_v11_simple recommendation outputs without segment IoU.

This script reports two recommendation-style metrics:

1. `start_keyframe`: compare the predicted action at the keyframe immediately
   before an action clip starts (or the closest available keyframe at/near the
   start) against the GT label active at the clip start.
2. `clip_keyframes`: compare predictions against GT over all 5-second keyframes
   inside each action clip and average the per-clip match rate.

For these single-label-per-actor comparisons, micro-F1 equals exact match
accuracy, so we report the match rate as `f1_mean`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from evaluate_all import (  # noqa: E402
    collect_pred_files_recursive,
    extract_model_from_pred,
    extract_taxonomy_from_pred,
    extract_video_id_from_pred,
)
from evaluate_baseline import load_predictions, parse_frame_number  # noqa: E402
from cvs_act.action_segment_eval import (  # noqa: E402
    GRANULARITIES,
    actor_label,
    convert_record_to_simple_actions,
    load_audit_records,
    write_json,
)
from surgent.audit_simple_actions import normalize_actor_slot_actions  # noqa: E402


ACTOR_ROLE_TO_SIMPLE = {
    "left_instrument": "left",
    "right_instrument": "right",
    "camera": "camera",
    "other": "other",
}
EVAL_ACTORS = ["left", "camera", "right", "other"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actions-dir",
        default="data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/audit_v11",
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=["outputs/cot_audit_v11_simple"],
    )
    parser.add_argument(
        "--metrics-base",
        default="metrics_audit_v11_simple_recommendation",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
    )
    parser.add_argument(
        "--modified-since",
        default="",
        help="Optional local timestamp cutoff like '2026-05-24 17:00:00'. Only prediction files modified on/after this time are used.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args()


def load_gt_simple_records(actions_dir: Path) -> List[dict]:
    raw_records = load_audit_records(actions_dir)
    simple_records = [convert_record_to_simple_actions(record) for record in raw_records]
    simple_records.sort(
        key=lambda record: (
            record.get("video_id") or "",
            tuple(record.get("frame_range") or [0, 0]),
            record.get("criterion") or "",
            record.get("example_id") or "",
        )
    )
    return simple_records


def gt_records_by_video(simple_records: Iterable[dict]) -> Dict[str, List[dict]]:
    by_video: Dict[str, List[dict]] = defaultdict(list)
    for record in simple_records:
        by_video[str(record["video_id"])].append(record)
    return by_video


def prediction_frame_number(pred_frame: dict) -> int:
    frame_id = pred_frame.get("frame_id")
    if frame_id is None:
        return -1
    return parse_frame_number(str(frame_id))


def inspect_prediction_file(pred_path: Path) -> dict:
    total_lines = 0
    bad_json_lines: List[int] = []
    with pred_path.open() as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad_json_lines.append(line_num)
    return {
        "total_lines": total_lines,
        "bad_json_lines": bad_json_lines,
    }


def is_complete_prediction(pred_frames: Sequence[dict], gt_simple_records: Sequence[dict]) -> bool:
    if not pred_frames or not gt_simple_records:
        return False
    pred_max = max(prediction_frame_number(frame) for frame in pred_frames)
    gt_max = max(int(record["frame_range"][1]) for record in gt_simple_records if record.get("frame_range"))
    return pred_max >= gt_max


def parse_modified_since(text: str) -> float | None:
    if not text.strip():
        return None
    return datetime.fromisoformat(text.strip()).timestamp()


def _empty_row(actor: str) -> dict:
    if actor == "left":
        return {"retraction_direction_code": "(not set)"}
    if actor == "right":
        return {
            "tool_type": "(not set)",
            "action_code": "(not set)",
            "target_structure": "(not set)",
            "target_context_1": "(not set)",
            "target_context_2": "(not set)",
        }
    if actor == "camera":
        return {"action_code": "(not set)"}
    return {"action_code": "(not set)"}


def _row_from_action(action: dict, actor: str) -> dict:
    if actor == "left":
        return {"retraction_direction_code": action.get("action_code", "(not set)")}
    if actor == "right":
        return {
            "tool_type": action.get("tool_type", "(not set)"),
            "action_code": action.get("action_code", "(not set)"),
            "target_structure": action.get("target_structure", "(not set)"),
            "target_context_1": action.get("target_context_1", "(not set)"),
            "target_context_2": action.get("target_context_2", "(not set)"),
        }
    return {"action_code": action.get("action_code", "(not set)")}


def pred_actor_row(pred_frame: dict, actor: str) -> dict:
    actions = pred_frame.get("actions", pred_frame.get("actions_ranked", []))
    if not isinstance(actions, list):
        return _empty_row(actor)
    normalized = normalize_actor_slot_actions([action for action in actions if isinstance(action, dict)])
    for action in normalized:
        bucket = ACTOR_ROLE_TO_SIMPLE.get(str(action.get("actor_role") or ""))
        if bucket == actor:
            return _row_from_action(action, actor)
    return _empty_row(actor)


def gt_actor_row(gt_record: dict, actor: str, frame_num: int) -> dict:
    rows = list(gt_record.get(actor, []))
    for row in rows:
        start = int(row.get("start_frame", -1))
        end = int(row.get("end_frame", -1))
        if start <= frame_num < end or (frame_num == end and end == int(gt_record["frame_range"][1])):
            return row
    return _empty_row(actor)


def keyframes_in_clip(start_frame: int, end_frame: int, available_frames: Sequence[int]) -> List[int]:
    return [frame for frame in available_frames if start_frame <= frame <= end_frame]


def choose_start_frame(start_frame: int, available_frames: Sequence[int]) -> int | None:
    if not available_frames:
        return None
    before = [frame for frame in available_frames if frame <= start_frame]
    if before:
        return max(before)
    return min(available_frames, key=lambda frame: abs(frame - start_frame))


def start_frame_pm1_keyframes(start_frame: int, available_frames: Sequence[int]) -> List[int]:
    onset_frame = choose_start_frame(start_frame, available_frames)
    if onset_frame is None:
        return []
    available = list(available_frames)
    try:
        onset_index = available.index(onset_frame)
    except ValueError:
        return [onset_frame]
    start_index = max(0, onset_index - 1)
    end_index = min(len(available), onset_index + 2)
    return available[start_index:end_index]


def frame_match_f1(pred_row: dict, gt_row: dict, actor: str, granularity: str) -> float:
    pred_label = actor_label(pred_row, actor, granularity)
    gt_label = actor_label(gt_row, actor, granularity)
    return 1.0 if pred_label == gt_label else 0.0


def method_key(method_label: str, model: str, taxonomy: str) -> str:
    return f"{method_label}/{model}/{taxonomy}"


def collect_method_frames(
    pred_roots: Sequence[Path],
    gt_by_video: Dict[str, List[dict]],
    include_partial: bool,
    modified_since: float | None,
    verbose: bool,
) -> tuple[Dict[str, Dict[str, List[dict]]], List[dict]]:
    frames_by_method: Dict[str, Dict[str, List[dict]]] = defaultdict(dict)
    skipped: List[dict] = []

    for pred_root in pred_roots:
        for method_label, pred_path_str in collect_pred_files_recursive(str(pred_root)):
            pred_path = Path(pred_path_str)
            if modified_since is not None and pred_path.stat().st_mtime < modified_since:
                continue
            health = inspect_prediction_file(pred_path)
            if health["bad_json_lines"] and not include_partial:
                skipped.append({"pred_path": str(pred_path), "reason": "malformed_jsonl"})
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
            key = method_key(method_label, model, taxonomy)
            frames_by_method[key][video_id] = pred_frames
            if verbose:
                print(f"EVAL {pred_path} -> {key}")

    return frames_by_method, skipped


def evaluate_method_frames(pred_frames_by_video: Dict[str, List[dict]], gt_records: List[dict]) -> List[dict]:
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
            onset_frame = choose_start_frame(clip_start, available)
            onset_pm1_frames = start_frame_pm1_keyframes(clip_start, available)
            clip_frames = keyframes_in_clip(clip_start, clip_end, available)

            for actor in EVAL_ACTORS:
                granularities = ["exact"] if actor == "other" else list(GRANULARITIES)
                for granularity in granularities:
                    if onset_frame is not None:
                        pred_row = pred_actor_row(by_frame_num[onset_frame], actor)
                        gt_row = gt_actor_row(record, actor, clip_start)
                        score = frame_match_f1(pred_row, gt_row, actor, granularity)
                        key = ("start_keyframe", actor, granularity)
                        by_metric_actor_granularity[key].append(score)
                        details[key].append(
                            {
                                "video_id": video_id,
                                "example_id": record.get("example_id"),
                                "frame": onset_frame,
                                "score": score,
                            }
                        )

                    if onset_pm1_frames:
                        gt_row = gt_actor_row(record, actor, clip_start)
                        frame_scores = []
                        for frame_num in onset_pm1_frames:
                            pred_row = pred_actor_row(by_frame_num[frame_num], actor)
                            frame_scores.append(
                                frame_match_f1(pred_row, gt_row, actor, granularity)
                            )
                        score = max(frame_scores) if frame_scores else 0.0
                        key = ("start_keyframe_pm1", actor, granularity)
                        by_metric_actor_granularity[key].append(score)
                        details[key].append(
                            {
                                "video_id": video_id,
                                "example_id": record.get("example_id"),
                                "frames": list(onset_pm1_frames),
                                "score": score,
                            }
                        )

                    if clip_frames:
                        scores = []
                        for frame_num in clip_frames:
                            pred_row = pred_actor_row(by_frame_num[frame_num], actor)
                            gt_row = gt_actor_row(record, actor, frame_num)
                            scores.append(frame_match_f1(pred_row, gt_row, actor, granularity))
                        clip_score = mean(scores)
                        key = ("clip_keyframes", actor, granularity)
                        by_metric_actor_granularity[key].append(clip_score)
                        details[key].append(
                            {
                                "video_id": video_id,
                                "example_id": record.get("example_id"),
                                "n_keyframes": len(clip_frames),
                                "score": clip_score,
                            }
                        )

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


def build_summary_rows(results: Dict[str, List[dict]]) -> List[dict]:
    rows: List[dict] = []
    for method, method_rows in sorted(results.items()):
        for row in method_rows:
            rows.append(
                {
                    "method": method,
                    "metric": row["metric"],
                    "actor": row["actor"],
                    "granularity": row["granularity"],
                    "f1_mean": row["f1_mean"],
                    "f1_std": row["f1_std"],
                    "n_examples": row["n_examples"],
                }
            )
    return rows


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "metric", "actor", "granularity", "f1_mean", "f1_std", "n_examples"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    actions_dir = ROOT_DIR / args.actions_dir
    pred_roots = [ROOT_DIR / path for path in args.output_dirs]
    metrics_dir = ROOT_DIR / args.metrics_base

    gt_records = load_gt_simple_records(actions_dir)
    gt_by_video = gt_records_by_video(gt_records)
    modified_since = parse_modified_since(args.modified_since)
    pred_frames_by_method, skipped = collect_method_frames(
        pred_roots=pred_roots,
        gt_by_video=gt_by_video,
        include_partial=args.include_partial,
        modified_since=modified_since,
        verbose=args.verbose,
    )

    results: Dict[str, List[dict]] = {}
    for method, by_video in sorted(pred_frames_by_method.items()):
        gt_subset = [record for record in gt_records if str(record.get("video_id")) in by_video]
        results[method] = evaluate_method_frames(by_video, gt_subset)

    summary_rows = build_summary_rows(results)
    write_json(
        metrics_dir / "recommendation_eval_results.json",
        {
            "actions_dir": str(actions_dir),
            "output_dirs": [str(path) for path in pred_roots],
            "modified_since": args.modified_since,
            "results": results,
            "skipped": skipped,
        },
    )
    write_summary_csv(metrics_dir / "recommendation_eval_summary.csv", summary_rows)

    print(f"Wrote {metrics_dir / 'recommendation_eval_results.json'}")
    print(f"Wrote {metrics_dir / 'recommendation_eval_summary.csv'}")
    print(f"Evaluated {len(results)} method group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
