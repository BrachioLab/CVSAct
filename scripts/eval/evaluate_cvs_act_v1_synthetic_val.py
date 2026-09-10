#!/usr/bin/env python3
"""Evaluate CVS-Act action recommendations against the synthetic val split.

This is the training-side companion to
``notebooks/training/audit_v11_train_val_split_template.ipynb``.  It evaluates
baseline/agent JSONL outputs against the exact ``train_val_split/val.json`` file
created by that notebook, using the same recommendation metric implementation
as the audit v11 evaluators.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence


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
from evaluate_audit_v11_simple_recommendation import (  # noqa: E402
    build_summary_rows,
    evaluate_method_frames,
    gt_records_by_video,
    inspect_prediction_file,
    is_complete_prediction,
    method_key,
    parse_modified_since,
    write_summary_csv,
)
from evaluate_baseline import load_predictions  # noqa: E402
from cvs_act.action_segment_eval import write_json  # noqa: E402


DEFAULT_VAL_JSON = (
    ROOT_DIR
    / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_synthetic_data"
    / "audit_v11_cvsctx_v3_1/train_val_split/val.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-json",
        default=str(DEFAULT_VAL_JSON.relative_to(ROOT_DIR)),
        help="Synthetic validation JSON created by the train/val split notebook.",
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        required=True,
        help="One or more prediction output directories to evaluate.",
    )
    parser.add_argument(
        "--metrics-base",
        default="metrics_cvs_act_v1_synthetic_val",
        help="Output metrics directory, relative to the repo root unless absolute.",
    )
    parser.add_argument("--models", nargs="*", default=[], help="Optional model id filter.")
    parser.add_argument("--taxonomy", nargs="*", default=[], help="Optional taxonomy version filter.")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--common-only", action="store_true")
    parser.add_argument(
        "--modified-since",
        default="",
        help="Optional local timestamp cutoff like '2026-05-24 17:00:00'.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT_DIR / path


def _parse_example_frame_range(example_id: str) -> list[int] | None:
    match = re.search(r"__c_(\d+)_(\d+)(?:$|__)", example_id)
    if not match:
        match = re.search(r"_c_(\d+)_(\d+)(?:$|__)", example_id)
    if not match:
        return None
    return [int(match.group(1)), int(match.group(2))]


def load_synthetic_val_records(path: Path) -> List[dict]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a JSON list")
    out: List[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        item = dict(record)
        if "frame_range" not in item:
            frame_range = _parse_example_frame_range(str(item.get("example_id", "")))
            if frame_range is None:
                raise ValueError(f"Could not infer frame_range for {item.get('example_id')}")
            item["frame_range"] = frame_range
        out.append(item)
    return out


def collect_method_frames(
    pred_roots: Sequence[Path],
    gt_by_video: Dict[str, List[dict]],
    *,
    include_partial: bool,
    modified_since: float | None,
    model_filter: set[str] | None,
    taxonomy_filter: set[str] | None,
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
            if verbose:
                print(f"EVAL {pred_path} -> {key}")

    return frames_by_method, skipped


def build_compact_summary(results: Dict[str, List[dict]]) -> dict:
    methods = {}
    for method, rows in sorted(results.items()):
        clip_exact = [
            float(row["f1_mean"])
            for row in rows
            if row.get("metric") == "clip_keyframes"
            and row.get("granularity") == "exact"
            and row.get("f1_mean") is not None
        ]
        clip_exact_no_other = [
            float(row["f1_mean"])
            for row in rows
            if row.get("metric") == "clip_keyframes"
            and row.get("granularity") == "exact"
            and row.get("actor") != "other"
            and row.get("f1_mean") is not None
        ]
        actor_exact = {
            str(row["actor"]): row["f1_mean"]
            for row in rows
            if row.get("metric") == "clip_keyframes" and row.get("granularity") == "exact"
        }
        methods[method] = {
            "clip_keyframes_exact_macro_actor_f1": mean(clip_exact) if clip_exact else None,
            "clip_keyframes_exact_macro_actor_f1_no_other": mean(clip_exact_no_other)
            if clip_exact_no_other
            else None,
            "clip_keyframes_exact_by_actor": actor_exact,
        }
    return {"methods": methods}


def write_compact_text(path: Path, compact: dict) -> None:
    lines = [
        "## Synthetic Val Compact Summary",
        "",
        "| Method | Clip exact macro F1 | Clip exact macro F1 no-other |",
        "| --- | ---: | ---: |",
    ]
    for method, row in sorted(compact.get("methods", {}).items()):
        macro = row.get("clip_keyframes_exact_macro_actor_f1")
        macro_no_other = row.get("clip_keyframes_exact_macro_actor_f1_no_other")
        macro_text = f"{macro:.3f}" if macro is not None else "nan"
        macro_no_other_text = f"{macro_no_other:.3f}" if macro_no_other is not None else "nan"
        lines.append(f"| {method} | {macro_text} | {macro_no_other_text} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    val_json = _resolve_repo_path(args.val_json)
    pred_roots = [_resolve_repo_path(path) for path in args.output_dirs]
    metrics_dir = _resolve_repo_path(args.metrics_base)

    gt_records = load_synthetic_val_records(val_json)
    gt_by_video = gt_records_by_video(gt_records)
    modified_since = parse_modified_since(args.modified_since)
    model_filter = set(args.models) if args.models else None
    taxonomy_filter = set(args.taxonomy) if args.taxonomy else None

    frames_by_method, skipped = collect_method_frames(
        pred_roots,
        gt_by_video,
        include_partial=args.include_partial,
        modified_since=modified_since,
        model_filter=model_filter,
        taxonomy_filter=taxonomy_filter,
        verbose=args.verbose,
    )

    if args.common_only and frames_by_method:
        common_video_ids = set.intersection(*(set(by_video) for by_video in frames_by_method.values()))
        for method in list(frames_by_method):
            frames_by_method[method] = {
                video_id: frames
                for video_id, frames in frames_by_method[method].items()
                if video_id in common_video_ids
            }
        print(f"Common videos across {len(frames_by_method)} group(s): {len(common_video_ids)}")

    results: Dict[str, List[dict]] = {}
    for method, by_video in sorted(frames_by_method.items()):
        gt_subset = [record for record in gt_records if str(record.get("video_id")) in by_video]
        results[method] = evaluate_method_frames(by_video, gt_subset)

    summary_rows = build_summary_rows(results)
    compact = build_compact_summary(results)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        metrics_dir / "synthetic_val_eval_results.json",
        {
            "val_json": str(val_json),
            "output_dirs": [str(path) for path in pred_roots],
            "modified_since": args.modified_since,
            "common_only": args.common_only,
            "models": args.models,
            "taxonomy": args.taxonomy,
            "n_gt_records": len(gt_records),
            "n_gt_videos": len(gt_by_video),
            "n_pred_videos_by_method": {
                method: len(by_video) for method, by_video in sorted(frames_by_method.items())
            },
            "results": results,
            "skipped": skipped,
        },
    )
    write_summary_csv(metrics_dir / "synthetic_val_eval_summary.csv", summary_rows)
    write_json(metrics_dir / "compact_summary.json", compact)
    write_compact_text(metrics_dir / "compact_summary.txt", compact)

    print(f"GT records: {len(gt_records)}  videos: {len(gt_by_video)}")
    for method, by_video in sorted(frames_by_method.items()):
        print(f"{method}: {len(by_video)} video(s)")
    print(f"Wrote {metrics_dir / 'synthetic_val_eval_results.json'}")
    print(f"Wrote {metrics_dir / 'synthetic_val_eval_summary.csv'}")
    print(f"Wrote {metrics_dir / 'compact_summary.json'}")
    print(f"Wrote {metrics_dir / 'compact_summary.txt'}")
    print(f"Evaluated {len(results)} method group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
